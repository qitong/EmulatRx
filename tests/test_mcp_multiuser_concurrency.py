"""Per-user folders, header rejection, and a 10-user MCP execution pool."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from starlette.testclient import TestClient

from emulatrx_tte._common import ContractError
from emulatrx_tte.sample_cohorts import _build_carolina_csv


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _tiny_csv(tag: str) -> str:
    lines = ["subject_id,treatment,time_day,event,age"]
    for index in range(24):
        lines.append(f"{tag}{index:04d},{index % 2},{40 + index},{1 if index % 4 == 0 else 0},{45 + (index % 12)}")
    return "\n".join(lines) + "\n"


class MultiUserAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        (base / "web").write_text("w" * 40, encoding="utf-8")
        (base / "agent").write_text("a" * 40, encoding="utf-8")
        patch = mock.patch.dict(
            os.environ,
            {
                "EMULATRX_RUNS_DIR": str(base / "runs"),
                "EMULATRX_WEB_PASSWORD_FILE": str(base / "web"),
                "EMULATRX_MCP_TOKEN_FILE": str(base / "agent"),
                "EMULATRX_ALLOWED_HOSTS": "testserver",
                "EMULATRX_OFFLINE_RCT_CATALOG": "1",
                "EMULATRX_CODE_COMMIT": "b" * 40,
                "EMULATRX_CODE_DIRTY": "false",
                "EMULATRX_MAX_CONCURRENT_RUNS": "8",
                "EMULATRX_SLOT_WAIT_SECONDS": "180",
            },
        )
        patch.start()
        self.addCleanup(patch.stop)
        import importlib

        from emulatrx_preview import app

        self.service = importlib.reload(app)
        self.runs = Path(self.service.runner.root)
        self.mcp_auth = {"Authorization": "Bearer " + "a" * 40, "Accept": "application/json, text/event-stream"}
        self.client = TestClient(self.service.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.http_lock = threading.Lock()

    def mcp_call(self, name, arguments, headers=None, rpc_id=1):
        merged = dict(self.mcp_auth)
        if headers:
            merged.update(headers)
        with self.http_lock:
            response = self.client.post(
                "/mcp",
                headers=merged,
                json={"jsonrpc": "2.0", "id": rpc_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        if payload.get("result", {}).get("isError"):
            raise AssertionError(response.text)
        return json.loads(payload["result"]["content"][0]["text"])

    def mcp_error(self, name, arguments, headers=None) -> str:
        merged = dict(self.mcp_auth)
        if headers:
            merged.update(headers)
        with self.http_lock:
            response = self.client.post(
                "/mcp",
                headers=merged,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["result"]["isError"], response.text)
        return response.text

    def _xlsx_bytes(self) -> bytes:
        import pandas as pd

        frame = pd.read_csv(io.BytesIO(_build_carolina_csv()))
        buffer = io.BytesIO()
        frame.to_excel(buffer, index=False, engine="openpyxl")
        return buffer.getvalue()

    def _wait_summary(self, run_id: str, headers: dict) -> dict:
        status = {}
        for _ in range(240):
            status = self.mcp_call("get_tte_run_status", {"run_id": run_id}, headers=headers)
            if status["state"] not in ("queued", "running"):
                break
            time.sleep(0.25)
        self.assertNotIn(status.get("state"), ("queued", "running"), status)
        summary = self.mcp_call("get_tte_result_summary", {"run_id": run_id}, headers=headers)
        return summary

    def test_physical_folders_xlsx_and_cross_user_denial(self):
        zhang = {"X-EmulatRx-User": "dr_zhang"}
        ticket = self.mcp_call(
            "ingest_cohort_session",
            {"issue_upload_ticket": True, "filename": "carolina.xlsx"},
            headers=zhang,
        )
        self.assertEqual(ticket["issue_upload_ticket"], True)
        uploaded = self.client.post(
            ticket["upload_url"],
            files={"file": ("carolina.xlsx", self._xlsx_bytes(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        body = uploaded.json()
        session_id = body["session_id"]
        self.assertEqual(body["user_namespace"], "dr_zhang")
        session_dir = self.runs / "users" / "dr_zhang" / "sessions" / session_id
        self.assertTrue(session_dir.is_dir(), session_dir)
        for directory in (
            self.runs / "users" / "dr_zhang",
            self.runs / "users" / "dr_zhang" / "sessions",
            self.runs / "users" / "dr_zhang" / "runs",
            session_dir,
        ):
            self.assertEqual(_mode(directory), 0o700, directory)

        csv_ticket = self.mcp_call(
            "ingest_cohort_session",
            {"issue_upload_ticket": True, "filename": "carolina.csv"},
            headers=zhang,
        )
        csv_upload = self.client.post(
            csv_ticket["upload_url"],
            files={"file": ("carolina.csv", _build_carolina_csv(), "text/csv")},
        )
        self.assertEqual(csv_upload.status_code, 200, csv_upload.text)
        csv_session = csv_upload.json()["session_id"]
        self.assertTrue((self.runs / "users" / "dr_zhang" / "sessions" / csv_session).is_dir())

        mapping = body["default_mapping"]
        diagnosed = self.mcp_call(
            "diagnose_tte_feasibility",
            {"session_id": session_id, "column_mapping": mapping},
            headers=zhang,
        )
        self.assertIn(diagnosed["gate_status"], ("passed", "blocked"))
        started = self.mcp_call(
            "run_static_or_landmark_tte",
            {
                "project_mode": "user_cohort",
                "session_id": session_id,
                "column_mapping": mapping,
                "bootstrap_iterations": 20,
                "wait_seconds": 90,
                "confirm_arm_mismatch": True,
                "confirm_protocol_mismatch": True,
            },
            headers=zhang,
        )
        self.assertEqual(started["state"], "completed", started)
        summary = self.mcp_call("get_tte_result_summary", {"run_id": started["run_id"]}, headers=zhang)
        self.assertEqual(summary["user_namespace"], "dr_zhang")
        self.assertEqual(summary["gate_status"], "passed")
        ladder = summary["clinical_comparison_ladder"]
        self.assertTrue(ladder["tier_1_unadjusted_crude"])
        tier2 = ladder["tier_2_conventional_ps"]
        self.assertIn("psm_1to1", tier2)
        self.assertIn("simple_iptw", tier2)
        self.assertTrue(tier2["psm_1to1"])
        self.assertTrue(tier2["simple_iptw"])
        self.assertTrue(ladder["tier_3_target_trial_emulation"])
        self.assertTrue(summary["what_tte_controlled_beyond_psm"])
        status = self.mcp_call("get_tte_run_status", {"run_id": started["run_id"]}, headers=zhang)
        self.assertEqual(status["user_namespace"], "dr_zhang")
        exported = self.mcp_call(
            "export_tte_artifact",
            {"run_id": started["run_id"], "artifact_name": "dualmode_report.html", "mode": "text"},
            headers=zhang,
        )
        self.assertIn("<", exported["text"])
        run_dir = self.runs / "users" / "dr_zhang" / "runs" / started["run_id"]
        self.assertTrue((run_dir / "dualmode_report.html").is_file())
        self.assertEqual(_mode(run_dir), 0o700)

        li = {"X-EmulatRx-User": "dr_li"}
        for tool, arguments, needle in (
            ("get_tte_result_summary", {"run_id": started["run_id"]}, "RUN_NOT_FOUND"),
            ("ingest_cohort_session", {"session_id": session_id}, "SESSION_NOT_FOUND"),
            ("export_tte_artifact", {"run_id": started["run_id"], "artifact_name": "dualmode_report.html"}, "RUN_NOT_FOUND"),
            ("delete_cohort_session", {"session_id": session_id}, "SESSION_NOT_FOUND"),
        ):
            self.assertIn(needle, self.mcp_error(tool, arguments, headers=li))
        self.assertTrue(session_dir.is_dir())
        self.assertTrue(run_dir.is_dir())

    def test_invalid_user_header_is_rejected_before_tools(self):
        listed = self.client.post(
            "/mcp",
            headers={**self.mcp_auth, "X-EmulatRx-User": ""},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
        self.assertEqual(listed.status_code, 200, listed.text)
        too_long = "a" * 64
        for raw in ("../root", "a/b", "bad space", too_long, ".", ".."):
            response = self.client.post(
                "/mcp",
                headers={**self.mcp_auth, "X-EmulatRx-User": raw},
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
            self.assertEqual(response.status_code, 400, raw)
            payload = response.json()
            self.assertEqual(payload["code"], "INVALID_USER_NAMESPACE")
            self.assertEqual(payload["error_code"], "INVALID_USER_NAMESPACE")
            self.assertTrue(str(payload["error"]).startswith("INVALID_USER_NAMESPACE"))

        alias = {"X-WorkBuddy-User": "dr_alias"}
        saved = self.mcp_call(
            "ingest_cohort_session",
            {"csv_text": _tiny_csv("A"), "filename": "alias.csv"},
            headers=alias,
        )
        self.assertEqual(saved["user_namespace"], "dr_alias")
        self.assertTrue((self.runs / "users" / "dr_alias" / "sessions" / saved["session_id"]).is_dir())

    def test_ten_users_run_without_busy_or_crosstalk(self):
        errors: list[BaseException] = []
        found: dict[str, dict] = {}

        def worker(index: int) -> None:
            user = f"user_{index:02d}"
            headers = {"X-EmulatRx-User": user}
            try:
                saved = self.mcp_call(
                    "ingest_cohort_session",
                    {"csv_text": _tiny_csv(user), "filename": f"{user}.csv"},
                    headers=headers,
                    rpc_id=index,
                )
                mapping = saved["default_mapping"]
                diagnosed = self.mcp_call(
                    "diagnose_tte_feasibility",
                    {"session_id": saved["session_id"], "column_mapping": mapping},
                    headers=headers,
                    rpc_id=index,
                )
                if "BUSY" in json.dumps(diagnosed):
                    raise AssertionError(diagnosed)
                started = self.mcp_call(
                    "run_static_or_landmark_tte",
                    {
                        "project_mode": "user_cohort",
                        "session_id": saved["session_id"],
                        "column_mapping": mapping,
                        "bootstrap_iterations": 20,
                        "wait_seconds": 0,
                        "confirm_arm_mismatch": True,
                        "confirm_protocol_mismatch": True,
                    },
                    headers=headers,
                    rpc_id=index,
                )
                summary = self._wait_summary(started["run_id"], headers)
                if summary.get("user_namespace") != user or summary.get("run_id") != started["run_id"]:
                    raise AssertionError(summary)
                found[user] = {"session_id": saved["session_id"], "run_id": started["run_id"], "summary": summary}
            except BaseException as exc:  # noqa: BLE001 - collect worker failures
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(1, 11)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=240)
        alive = [thread.name for thread in threads if thread.is_alive()]
        self.assertEqual(alive, [])
        self.assertEqual(errors, [])
        self.assertEqual(len(found), 10)
        for index in range(1, 11):
            user = f"user_{index:02d}"
            self.assertTrue((self.runs / "users" / user).is_dir(), user)
            self.assertTrue((self.runs / "users" / user / "sessions" / found[user]["session_id"]).is_dir())
            self.assertTrue((self.runs / "users" / user / "runs" / found[user]["run_id"]).is_dir())
        alien = self.mcp_error(
            "get_tte_result_summary",
            {"run_id": found["user_02"]["run_id"]},
            headers={"X-EmulatRx-User": "user_01"},
        )
        self.assertIn("RUN_NOT_FOUND", alien)
        self.assertNotIn("429", alien)

    def test_stdio_rejects_path_namespace(self):
        completed = subprocess.run(
            [sys.executable, "-m", "emulatrx_tte", "mcp-serve", "--user-namespace", "../root"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("INVALID_USER_NAMESPACE", completed.stderr + completed.stdout)


class SlotEnvTests(unittest.TestCase):
    def test_zero_wait_single_slot_is_busy(self):
        from emulatrx_preview.tte_runner import TTERunner

        protocol = json.loads((Path(__file__).resolve().parents[1] / "examples" / "tte_preview_v1.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"EMULATRX_MAX_CONCURRENT_RUNS": "1", "EMULATRX_SLOT_WAIT_SECONDS": "0"}):
                runner = TTERunner(directory)
                self.assertTrue(runner._slot.acquire(blocking=False))
                try:
                    with self.assertRaises(ContractError) as raised:
                        runner.start(protocol, cases=1000, seed=1, bootstrap_iterations=20)
                    self.assertEqual(raised.exception.code, "BUSY")
                finally:
                    runner._slot.release()
