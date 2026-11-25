"""Black-box coverage for the 14-tool MCP registry over HTTP and stdio."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from starlette.testclient import TestClient

from emulatrx_preview.mcp_tools import CANONICAL_TOOLS, LEGACY_TOOL_ALIASES, RESOURCE_URIS

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "examples" / "tte_preview_v1.json").read_text(encoding="utf-8"))
CCW_PROTOCOL = json.loads((ROOT / "examples" / "tte_grace_period_ccw_v1.json").read_text(encoding="utf-8"))


class McpAgentSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        (base / "web").write_text("w" * 40, encoding="utf-8")
        (base / "agent").write_text("a" * 40, encoding="utf-8")
        self.allow = base / "ingest"
        self.allow.mkdir()
        patch = mock.patch.dict(
            os.environ,
            {
                "EMULATRX_RUNS_DIR": str(base / "runs"),
                "EMULATRX_WEB_PASSWORD_FILE": str(base / "web"),
                "EMULATRX_MCP_TOKEN_FILE": str(base / "agent"),
                "EMULATRX_ALLOWED_HOSTS": "testserver",
                "EMULATRX_OFFLINE_RCT_CATALOG": "1",
                "EMULATRX_MCP_ALLOWED_INGEST_DIR": str(self.allow),
                "EMULATRX_CODE_COMMIT": "b" * 40,
                "EMULATRX_CODE_DIRTY": "false",
            },
        )
        patch.start()
        self.addCleanup(patch.stop)
        import importlib

        from emulatrx_preview import app

        self.service = importlib.reload(app)
        self.mcp_auth = {"Authorization": "Bearer " + "a" * 40, "Accept": "application/json, text/event-stream"}
        self.client = TestClient(self.service.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def mcp_call(self, name, arguments, rpc_id=1):
        response = self.client.post(
            "/mcp",
            headers=self.mcp_auth,
            json={"jsonrpc": "2.0", "id": rpc_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        if payload.get("result", {}).get("isError"):
            raise AssertionError(response.text)
        content = payload["result"]["content"]
        text = content[0]["text"]
        return json.loads(text)

    def mcp_error(self, name, arguments):
        response = self.client.post(
            "/mcp",
            headers=self.mcp_auth,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["result"]["isError"], response.text)
        return response.text

    def test_http_five_stage_workflow_resources_and_fail_closed(self):
        listed = self.client.post("/mcp", headers=self.mcp_auth, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual(listed.status_code, 200, listed.text)
        for name in (*CANONICAL_TOOLS, *LEGACY_TOOL_ALIASES):
            self.assertIn(name, listed.text)
        resources = self.client.post("/mcp", headers=self.mcp_auth, json={"jsonrpc": "2.0", "id": 2, "method": "resources/list"})
        self.assertEqual(resources.status_code, 200, resources.text)
        for uri in RESOURCE_URIS:
            self.assertIn(uri, resources.text)
            read = self.client.post(
                "/mcp",
                headers=self.mcp_auth,
                json={"jsonrpc": "2.0", "id": 3, "method": "resources/read", "params": {"uri": uri}},
            )
            self.assertEqual(read.status_code, 200, read.text)
            self.assertNotIn("http://", read.text.lower().replace("http/1.1", ""))

        idea = self.mcp_call("parse_research_idea", {"idea_text": "利格列汀对比格列美脲治疗2型糖尿病心血管结局"})
        self.assertIn("pico", idea)
        self.assertIn("ctgov_queries", idea)
        ingested = self.mcp_call("ingest_cohort_session", {"sample_id": "t2dm_cv_carolina"})
        session_id = ingested["session_id"]
        self.assertIn("profile", ingested)
        audited = self.mcp_call(
            "audit_target_rct_alignment",
            {"nct_id": "NCT01243424", "session_id": session_id, "project_mode": "user_cohort"},
        )
        self.assertEqual(audited["nct_id"], "NCT01243424")
        self.assertIn("overall_alignment_status", audited)
        drafted = self.mcp_call(
            "draft_tte_protocol",
            {
                "design_mode": "static",
                "session_id": session_id,
                "nct_id": "NCT01243424",
                "confirm_arm_mismatch": True,
                "confirm_protocol_mismatch": True,
            },
        )
        self.assertEqual(drafted["schema"], "tte-protocol-v1")
        validated = self.mcp_call("validate_tte_protocol", {"protocol": drafted["protocol_draft"], "session_id": session_id})
        self.assertTrue(validated["valid"])
        diagnosed = self.mcp_call(
            "diagnose_tte_feasibility",
            {"session_id": session_id, "column_mapping": drafted["column_mapping"]},
        )
        self.assertIn(diagnosed["gate_status"], ("passed", "blocked"))
        self.assertIn("waterfall", diagnosed)
        started = self.mcp_call(
            "run_static_or_landmark_tte",
            {
                "project_mode": "user_cohort",
                "session_id": session_id,
                "nct_id": "NCT01243424",
                "column_mapping": drafted["column_mapping"],
                "bootstrap_iterations": 20,
                "confirm_arm_mismatch": True,
                "confirm_protocol_mismatch": True,
                "wait_seconds": 30,
            },
        )
        self.assertEqual(started["state"], "completed")
        summary = self.mcp_call("get_tte_result_summary", {"run_id": started["run_id"]})
        self.assertEqual(summary["run_type"], "static_or_landmark")
        self.assertEqual(summary["gate_status"], "passed")
        self.assertIn("tier_1_unadjusted_crude", summary["clinical_comparison_ladder"])
        self.assertIn("tier_2_conventional_ps", summary["clinical_comparison_ladder"])
        self.assertIn("tier_3_target_trial_emulation", summary["clinical_comparison_ladder"])
        self.assertIn("headline_progression_zh", summary["what_tte_controlled_beyond_psm"])
        exported = self.mcp_call(
            "export_tte_artifact",
            {"run_id": started["run_id"], "artifact_name": "audit_report.md", "mode": "text"},
        )
        self.assertIn("临床专家三阶结果对比", exported["text"])
        self.assertEqual(len(exported["sha256"]), 64)
        deleted = self.mcp_call("delete_cohort_session", {"session_id": session_id})
        self.assertTrue(deleted.get("deleted", True))

        ccw_ok = self.mcp_call("validate_ccw_protocol", {"ccw_protocol": CCW_PROTOCOL})
        self.assertTrue(ccw_ok["valid"])
        ccw_run = self.mcp_call(
            "run_grace_period_ccw",
            {"synthetic_scenario": "null_effect", "seed": 42, "cases": 700, "bootstrap_replicates": 15},
        )
        self.assertEqual(ccw_run["gate_status"], "passed")
        ccw_summary = self.mcp_call("get_tte_result_summary", {"run_id": ccw_run["run_id"]})
        self.assertEqual(ccw_summary["run_type"], "grace_period_ccw")
        self.assertIsNotNone(ccw_summary["clinical_comparison_ladder"]["tier_3_target_trial_emulation"])
        self.assertIn("非临床疗效证据", ccw_summary["evidence_boundary_notice"])

        mismatch = self.mcp_error(
            "draft_tte_protocol",
            {"design_mode": "static", "nct_id": "NCT01243424", "pico": {"intervention_query_en": "Aspirin", "comparator_query_en": "Placebo"}},
        )
        self.assertTrue("ARM_MISMATCH_CONFIRMATION_REQUIRED" in mismatch or "PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED" in mismatch)
        self.assertIn("PATH_TRAVERSAL_BLOCKED", self.mcp_error("ingest_cohort_session", {"file_path": "/etc/passwd"}))

        ticket = self.mcp_call("ingest_cohort_session", {"issue_upload_ticket": True, "filename": "ward.csv"})
        raw = "subject_id,treatment,time_day,event\n" + "\n".join(f"H{i:04d},{i % 2},{30+i},{i % 2}" for i in range(12)) + "\n"
        uploaded = self.client.post(ticket["upload_url"], content=raw.encode("utf-8"), headers={"Content-Type": "text/csv"})
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        self.assertNotIn("H0001", uploaded.text)
        reused = self.client.post(ticket["upload_url"], content=raw.encode("utf-8"), headers={"Content-Type": "text/csv"})
        self.assertGreaterEqual(reused.status_code, 400)

        blocked_protocol = json.loads(json.dumps(PROTOCOL))
        blocked_protocol["estimand"]["diagnostic_gates"]["positivity_epsilon"] = 0.49
        blocked = self.mcp_call("start_synthetic_tte", {"protocol": blocked_protocol, "cases": 300, "seed": 1, "bootstrap_iterations": 20})
        for _ in range(100):
            status = self.mcp_call("get_tte_run_status", {"run_id": blocked["run_id"]})
            if status["state"] not in ("queued", "running"):
                break
            time.sleep(0.1)
        self.assertEqual(status["status"], "gate_blocked")
        omitted = self.mcp_call("get_tte_result_summary", {"run_id": blocked["run_id"]})
        self.assertEqual(omitted["gate_status"], "blocked")
        self.assertIsNone(omitted["clinical_comparison_ladder"])
        self.assertIn("omitted", omitted["estimates_omitted_reason"])

    def test_stdio_lists_tools_resources_and_reads_local_csv(self):
        runs = Path(self.temp.name) / "stdio-runs"
        csv_path = Path(self.temp.name) / "local.csv"
        csv_path.write_text(
            "subject_id,treatment,time_day,event,age\n" + "\n".join(f"L{i:04d},{i % 2},{40+i},{i % 3},{50+i}" for i in range(12)) + "\n",
            encoding="utf-8",
        )
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "emulatrx_tte",
                "mcp-serve",
                "--runs-dir",
                str(runs),
            ],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(proc.kill)

        def rpc(message):
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
            if "id" not in message:
                return None
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise AssertionError(proc.stderr.read())
                if line.startswith("{"):
                    payload = json.loads(line)
                    if payload.get("id") == message["id"]:
                        return payload

        init = rpc(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "suite", "version": "0"}},
            }
        )
        self.assertIn("result", init, msg=str(init))
        rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [item["name"] for item in tools["result"]["tools"]]
        for name in (*CANONICAL_TOOLS, *LEGACY_TOOL_ALIASES):
            self.assertIn(name, names)
        listed = rpc({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})
        uris = [item["uri"] for item in listed["result"]["resources"]]
        for uri in RESOURCE_URIS:
            self.assertIn(uri, uris)
        read = rpc({"jsonrpc": "2.0", "id": 4, "method": "resources/read", "params": {"uri": "emulatrx://guide/error-and-gate-codes"}})
        self.assertIn("result", read)
        ingested = rpc(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "ingest_cohort_session", "arguments": {"file_path": str(csv_path)}},
            }
        )
        self.assertFalse(ingested["result"].get("isError"), msg=json.dumps(ingested)[:1000])
        body = json.loads(ingested["result"]["content"][0]["text"])
        self.assertTrue(body["missing_clinical_semantics"])
        self.assertNotIn("L0001", json.dumps(body))
        proc.terminate()


if __name__ == "__main__":
    unittest.main()
