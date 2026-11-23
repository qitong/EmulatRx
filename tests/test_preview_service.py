"""Real synthetic TTE browser/MCP flows and service boundaries."""

import base64
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from starlette.testclient import TestClient

from emulatrx_tte._common import ContractError, code_provenance
from emulatrx_preview.tte_runner import TTERunner


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "examples" / "tte_preview_v1.json").read_text(encoding="utf-8"))
ELEMENT_KEYS = ("eligibility", "time_zero", "treatment_strategies", "assignment_mode", "followup", "outcomes", "estimand")


class PreviewServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        (base / "web").write_text("w" * 40)
        (base / "agent").write_text("a" * 40)
        patch = mock.patch.dict(os.environ, {
            "EMULATRX_RUNS_DIR": str(base / "runs"),
            "EMULATRX_WEB_PASSWORD_FILE": str(base / "web"),
            "EMULATRX_MCP_TOKEN_FILE": str(base / "agent"),
            "EMULATRX_CODE_COMMIT": "b" * 40,
            "EMULATRX_CODE_DIRTY": "false",
            "EMULATRX_ALLOWED_HOSTS": "testserver",
            "EMULATRX_TTE_DEPENDENCY_COMMIT": "fedbc754a1fea22bde559b2575c462829db23f95",
        })
        patch.start()
        self.addCleanup(patch.stop)
        from emulatrx_preview import app
        self.service = importlib.reload(app)
        self.web_auth = {"Authorization": "Basic " + base64.b64encode(("preview:" + "w" * 40).encode()).decode()}
        self.mcp_auth = {"Authorization": "Bearer " + "a" * 40, "Accept": "application/json, text/event-stream"}
        self.client = TestClient(self.service.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def form(self, protocol=None, cases="2000", bootstrap="100"):
        protocol = protocol or PROTOCOL
        values = {"study_title": protocol["study_title"], "cases": cases, "seed": "42", "bootstrap_iterations": bootstrap}
        values.update({key: json.dumps(protocol[key]) for key in ELEMENT_KEYS})
        return values

    def wait_run(self, run_id):
        for _ in range(200):
            result = self.service.runner.get(run_id)
            if result["state"] not in ("queued", "running"):
                return result
            time.sleep(0.1)
        self.fail("TTE run did not finish in 20 seconds")

    def mcp_call(self, name, arguments, rpc_id=1):
        response = self.client.post("/mcp", headers=self.mcp_auth,
                                    json={"jsonrpc": "2.0", "id": rpc_id, "method": "tools/call",
                                          "params": {"name": name, "arguments": arguments}})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["result"]
        self.assertFalse(result["isError"], result)
        return json.loads(result["content"][0]["text"])

    def test_browser_seven_element_full_flow_and_benchmark(self):
        home = self.client.get("/", headers=self.web_auth)
        self.assertEqual(home.status_code, 200)
        self.assertIn("七要素", home.text)
        self.assertIn("非真实患者", home.text)
        validated = self.client.post("/validate", headers=self.web_auth, data=self.form())
        self.assertEqual(validated.status_code, 200, validated.text[:300])
        self.assertIn("规范化方案", validated.text)
        created = self.client.post("/runs", headers=self.web_auth, data=self.form(), follow_redirects=False)
        self.assertEqual(created.status_code, 303, created.text)
        run_id = created.headers["location"].split("/")[-1]
        result = self.wait_run(run_id)
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["gate_status"], "passed")
        self.assertEqual(result["cohort_rows"], 2000)
        self.assertEqual(result["eligible_rows"], 954)
        self.assertGreaterEqual(len(result["cohort_columns"]), 10)
        self.assertEqual(result["preview_manifest"]["code_commit"], "b" * 40)
        self.assertEqual(result["diagnostics"]["gate_status"], "passed")
        self.assertGreater(result["diagnostics"]["ess_diagnostics"]["min_ess_ratio_observed"], 0.25)
        row = result["estimates"]["co_presented_comparison_table"][0]
        self.assertAlmostEqual(row["rd_primary"], -0.0309042318, delta=0.005)
        self.assertAlmostEqual(row["shr_primary"], 0.820865552, delta=0.03)
        self.assertAlmostEqual(row["cshr_primary"], 0.7649949755, delta=0.03)
        self.assertLessEqual(row["cif_primary_active"] + row["cif_competing_active"], 1)
        result_page = self.client.get(created.headers["location"], headers=self.web_auth)
        self.assertEqual(result_page.status_code, 200)
        for expected in ("2000 行", "954 行", "筛选流失", "倾向分数分布", "目标事件 CIF", "竞争事件 CIF", "RD", "RR(CIF)", "sHR", "csHR", "rd_primary_ci_lower", "方法限制与警示", "完整审计报告"):
            self.assertIn(expected, result_page.text)
        cohort = self.client.get(f"/runs/{run_id}/artifacts/cohort/cohort.csv", headers=self.web_auth)
        self.assertEqual(cohort.status_code, 200)
        rows = list(csv.DictReader(cohort.text.splitlines()))
        self.assertEqual(len(rows), 2000)
        self.assertEqual(set(row["event_code"] for row in rows), {"0", "1", "2"})
        selected = self.client.get(f"/runs/{run_id}/artifacts/screen/eligible_cohort.csv", headers=self.web_auth)
        self.assertEqual(len(list(csv.DictReader(selected.text.splitlines()))), 954)
        digest = hashlib.sha256(selected.content).hexdigest()
        diag_manifest = json.loads(self.client.get(f"/runs/{run_id}/artifacts/diagnostics/run_manifest.json", headers=self.web_auth).text)
        self.assertEqual(diag_manifest["cohort_sha256"], digest)
        self.assertEqual(self.client.get(f"/runs/{run_id}/artifacts/cohort/cohort.csv").status_code, 401)
        restarted = TTERunner(self.service.runner.root)
        self.assertEqual(restarted.get(run_id)["cohort_rows"], 2000)

    def test_mcp_validate_run_status_and_chunked_artifact(self):
        listed = self.client.post("/mcp", headers=self.mcp_auth, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual(listed.status_code, 200, listed.text)
        for tool in ("validate_tte_protocol", "start_synthetic_tte", "get_tte_run", "read_tte_artifact"):
            self.assertIn(tool, listed.text)
        validated = self.mcp_call("validate_tte_protocol", {"protocol": PROTOCOL})
        self.assertEqual(validated["normalized_protocol"]["study_title"], PROTOCOL["study_title"])
        started = self.mcp_call("start_synthetic_tte", {"protocol": PROTOCOL, "cases": 1000, "seed": 42, "bootstrap_iterations": 20})
        final = self.wait_run(started["run_id"])
        self.assertEqual(final["state"], "completed")
        remote = self.mcp_call("get_tte_run", {"run_id": started["run_id"]})
        self.assertEqual(remote["cohort_rows"], 1000)
        self.assertIn("estimates", remote)
        chunk = self.mcp_call("read_tte_artifact", {"run_id": started["run_id"], "name": "estimates/audit_report.md", "limit": 128})
        self.assertEqual(chunk["encoding"], "base64")
        self.assertLessEqual(chunk["next_offset"], 128)
        self.assertEqual(self.client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).status_code, 401)

    def test_blocked_gate_exposes_diagnostics_without_estimates(self):
        protocol = json.loads(json.dumps(PROTOCOL))
        protocol["estimand"]["diagnostic_gates"]["positivity_epsilon"] = 0.49
        started = self.service.runner.start(protocol, cases=1000, seed=42, bootstrap_iterations=20)
        result = self.wait_run(started["run_id"])
        self.assertEqual(result["state"], "blocked")
        self.assertEqual(result["gate_status"], "blocked")
        self.assertEqual(result["cohort_rows"], 1000)
        self.assertNotIn("estimates", result)
        self.assertIn("POSITIVITY_VIOLATION", str(result["diagnostics"]["gate_failures"]))
        self.assertFalse(any("estimates.json" in name for name in result["available_artifacts"]))
        self.assertEqual(self.client.get(f"/runs/{started['run_id']}/artifacts/estimates/estimates.json", headers=self.web_auth).status_code, 404)
        page = self.client.get(f"/runs/{started['run_id']}", headers=self.web_auth)
        self.assertIn("估计结果与曲线不生成", page.text)
        self.assertNotIn("完整审计报告", page.text)

    def test_edited_eligibility_actually_changes_screening_and_blocks_empty_risk_set(self):
        protocol = json.loads(json.dumps(PROTOCOL))
        protocol["eligibility"]["inclusion_criteria"][0]["value"] = 3.5
        started = self.service.runner.start(protocol, cases=1000, seed=42, bootstrap_iterations=20)
        result = self.wait_run(started["run_id"])
        self.assertEqual(result["state"], "blocked")
        self.assertEqual(result["eligible_rows"], 0)
        self.assertEqual(result["attrition"][1]["excluded_n"], "1000")
        self.assertIn("NO_ELIGIBLE_SYNTHETIC_RISK_SET", str(result["diagnostics"]["gate_failures"]))
        self.assertNotIn("estimates", result)
        missing_variable = json.loads(json.dumps(PROTOCOL))
        missing_variable["eligibility"]["exclusion_criteria"][0]["variable"] = "prior_event_before_t0"
        with self.assertRaises(ContractError) as raised:
            self.service.runner.validate(missing_variable)
        self.assertEqual(raised.exception.code, "UNAVAILABLE_ELIGIBILITY_VARIABLE")

    def test_authentication_input_and_path_gates(self):
        self.assertEqual(self.client.get("/").status_code, 401)
        self.assertEqual(self.client.get("/", headers=self.mcp_auth).status_code, 401)
        self.assertEqual(self.client.post("/mcp", headers=self.web_auth, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).status_code, 401)
        self.assertEqual(self.client.post("/runs", headers={**self.web_auth, "Origin": "https://wrong.example"}, data=self.form()).status_code, 400)
        self.assertEqual(self.client.post("/runs", headers=self.web_auth, json={"patient_file": "forbidden"}).status_code, 400)
        self.assertEqual(self.client.post("/runs", headers=self.web_auth, data={**self.form(), "patient_id": "x"}).status_code, 400)
        self.assertEqual(self.client.post("/runs", headers=self.web_auth, data=self.form(cases="5001")).status_code, 400)
        changed = json.loads(json.dumps(PROTOCOL))
        changed["time_zero"]["day_offset"] = 1
        self.assertEqual(self.client.post("/validate", headers=self.web_auth, data=self.form(changed)).status_code, 400)
        with self.assertRaises(ContractError):
            self.service.runner.artifact_path("a" * 32, "../../secrets/web_password")
        self.assertTrue(self.service.runner._slot.acquire(blocking=False))
        try:
            with self.assertRaises(ContractError) as raised:
                self.service.runner.start(PROTOCOL, cases=1000, seed=42, bootstrap_iterations=20)
            self.assertEqual(raised.exception.code, "BUSY")
        finally:
            self.service.runner._slot.release()

    def test_build_provenance_and_interrupted_restart(self):
        self.assertEqual(code_provenance(), ("b" * 40, False))
        with mock.patch.dict(os.environ, {"EMULATRX_CODE_COMMIT": "bad"}):
            with self.assertRaises(ContractError):
                code_provenance()
        run_id = "c" * 32
        path = self.service.runner.root / run_id
        path.mkdir()
        (path / "status.json").write_text(json.dumps({"run_id": run_id, "state": "running"}))
        recovered = TTERunner(self.service.runner.root)
        self.assertEqual(recovered.get(run_id)["state"], "interrupted")


if __name__ == "__main__":
    unittest.main()
