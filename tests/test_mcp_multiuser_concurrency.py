"""Per-user folders, header rejection, and a 10-user MCP execution pool."""

from __future__ import annotations

import asyncio
import base64
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


class Round2ReviewFixTests(unittest.TestCase):
    """Cleanup race, raw namespace headers, missing-session 404, and CCW run reads."""

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
            },
        )
        patch.start()
        self.addCleanup(patch.stop)
        import importlib

        from emulatrx_preview import app

        self.service = importlib.reload(app)
        self.client = TestClient(self.service.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.web_auth = {"Authorization": "Basic " + base64.b64encode(b"preview:" + b"w" * 40).decode()}

    def _asgi_status(self, path: str, headers: list[tuple[bytes, bytes]], method: str = "GET") -> int:
        sent: list[dict] = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 1),
            "server": ("testserver", 80),
        }
        asyncio.run(self.service.app(scope, receive, send))
        return next(message["status"] for message in sent if message["type"] == "http.response.start")

    def test_raw_header_whitespace_is_rejected_and_empty_is_default(self):
        from emulatrx_preview.user_context import InvalidUserNamespace, coerce_user_namespace, namespace_from_headers

        self.assertEqual(coerce_user_namespace(None), "default")
        self.assertEqual(coerce_user_namespace(""), "default")
        self.assertEqual(namespace_from_headers({}), "default")
        self.assertEqual(namespace_from_headers({b"x-emulatrx-user": b""}), "default")
        for raw in ("   ", "\t", " dr_zhang", "dr_zhang ", "bad space"):
            with self.assertRaises(InvalidUserNamespace):
                coerce_user_namespace(raw)
            with self.assertRaises(InvalidUserNamespace):
                namespace_from_headers({b"x-emulatrx-user": raw.encode("latin-1")})
            with self.assertRaises(InvalidUserNamespace):
                namespace_from_headers({b"x-workbuddy-user": raw.encode("latin-1")})

        basic = b"Basic " + base64.b64encode(b"preview:" + b"w" * 40)
        for raw in (b"   ", b"\t", b" dr_zhang", b"dr_zhang "):
            status = self._asgi_status(
                "/api/workbench/bootstrap",
                [(b"authorization", basic), (b"host", b"testserver"), (b"x-emulatrx-user", raw)],
            )
            self.assertEqual(status, 400, raw)
        self.assertEqual(
            self._asgi_status(
                "/api/workbench/bootstrap",
                [(b"authorization", basic), (b"host", b"testserver"), (b"x-emulatrx-user", b"")],
            ),
            200,
        )
        self.assertEqual(
            self._asgi_status(
                "/api/sessions/upload-ticket/../secret",
                [(b"host", b"testserver")],
                method="POST",
            ),
            401,
        )
        self.assertEqual(
            self._asgi_status(
                "/api/sessions/upload-ticket/..",
                [(b"host", b"testserver")],
                method="POST",
            ),
            401,
        )

    def test_missing_and_cross_user_delete_is_404(self):
        missing = self.client.delete("/api/sessions/" + "cd" * 16, headers=self.web_auth)
        self.assertEqual(missing.status_code, 404, missing.text)
        self.assertEqual(missing.json()["code"], "SESSION_NOT_FOUND")
        self.assertFalse(missing.json()["session_deleted"])
        invalid = self.client.delete("/api/sessions/not-a-session", headers=self.web_auth)
        self.assertEqual(invalid.status_code, 400, invalid.text)

        runner = self.service.runner
        from emulatrx_preview.user_context import CURRENT_USER_NAMESPACE

        token = CURRENT_USER_NAMESPACE.set("dr_zhang")
        try:
            saved = runner.save_session_cohort(_tiny_csv("Z").encode("utf-8"), "z.csv")
        finally:
            CURRENT_USER_NAMESPACE.reset(token)
        foreign = self.client.delete(
            f"/api/sessions/{saved['session_id']}",
            headers={**self.web_auth, "X-EmulatRx-User": "dr_li"},
        )
        self.assertEqual(foreign.status_code, 404, foreign.text)
        self.assertEqual(foreign.json()["code"], "SESSION_NOT_FOUND")
        self.assertTrue((runner.root / "users" / "dr_zhang" / "sessions" / saved["session_id"]).is_dir())
        own = self.client.delete(
            f"/api/sessions/{saved['session_id']}",
            headers={**self.web_auth, "X-EmulatRx-User": "dr_zhang"},
        )
        self.assertEqual(own.status_code, 200, own.text)

    def test_completed_ccw_run_get_and_namespaced_artifact(self):
        from emulatrx_preview.user_context import CURRENT_USER_NAMESPACE

        runner = self.service.runner
        source = runner.root / "ccw-src"
        source.mkdir()
        (source / "ccw_preview_report.html").write_text("<html>ccw-report-body</html>", encoding="utf-8")
        (source / "ccw_estimates.json").write_text("{}", encoding="utf-8")
        token = CURRENT_USER_NAMESPACE.set("dr_zhang")
        try:
            status = runner.adopt_ccw_directory(source, gate_status="passed", seed=3, case_count=12)
        finally:
            CURRENT_USER_NAMESPACE.reset(token)
        run_id = status["run_id"]
        anonymous = self.client.get(f"/api/dual-runs/{run_id}", headers=self.web_auth)
        self.assertEqual(anonymous.status_code, 404, anonymous.text)
        fetched = self.client.get(
            f"/api/dual-runs/{run_id}",
            headers={**self.web_auth, "X-EmulatRx-User": "dr_zhang"},
        )
        self.assertEqual(fetched.status_code, 200, fetched.text)
        body = fetched.json()
        self.assertEqual(body["kind"], "grace_period_ccw")
        self.assertNotIn("normalized_protocol", body)
        self.assertIn("ccw-report-body", body["report_html"])
        self.assertNotIn("rendered_html", body)
        page = self.client.get(f"/runs/{run_id}", headers={**self.web_auth, "X-EmulatRx-User": "dr_zhang"})
        self.assertEqual(page.status_code, 200, page.text)
        self.assertIn("ccw-report-body", page.text)
        artifact = self.client.get(
            f"/runs/{run_id}/artifacts/ccw_preview_report.html",
            headers={**self.web_auth, "X-EmulatRx-User": "dr_zhang"},
        )
        self.assertEqual(artifact.status_code, 200, artifact.text)
        self.assertIn("ccw-report-body", artifact.text)
        hidden = self.client.get(
            f"/runs/{run_id}/artifacts/ccw_preview_report.html",
            headers=self.web_auth,
        )
        self.assertEqual(hidden.status_code, 404, hidden.text)
        bare = runner.root / "ccw-src-2"
        bare.mkdir()
        (bare / "ccw_diagnostics.json").write_text("{}", encoding="utf-8")
        quiet = runner.adopt_ccw_directory(bare, gate_status="blocked", seed=4, case_count=8)
        quiet_get = self.client.get(f"/api/dual-runs/{quiet['run_id']}", headers=self.web_auth)
        self.assertEqual(quiet_get.status_code, 200, quiet_get.text)
        self.assertNotIn("report_html", quiet_get.json())


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

    def test_cleanup_never_deletes_a_run_being_allocated_or_adopted(self):
        from emulatrx_preview import tte_runner as runner_mod
        from emulatrx_preview.tte_runner import TTERunner

        protocol = json.loads(
            (Path(__file__).resolve().parents[1] / "examples" / "tte_preview_v1.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = TTERunner(directory)
            fresh = Path(directory) / ("ab" * 16)
            fresh.mkdir(mode=0o700)
            (fresh / "keep.txt").write_text("alive", encoding="utf-8")
            runner.cleanup()
            self.assertTrue(fresh.is_dir())
            self.assertTrue((fresh / "keep.txt").is_file())
            old = time.time() - 120
            os.utime(fresh, (old, old))
            runner.cleanup()
            self.assertFalse(fresh.exists())

            removed: list[str] = []
            real_rmtree = runner_mod.shutil.rmtree
            real_copy = runner_mod.shutil.copyfile
            real_allocate = runner._allocate_run_dir
            allocate_entered = threading.Event()
            allocate_release = threading.Event()
            copy_entered = threading.Event()
            copy_release = threading.Event()

            def spy_rmtree(path, *args, **kwargs):
                removed.append(Path(path).name)
                return real_rmtree(path, *args, **kwargs)

            def hooked_allocate(ns):
                run_id, path = real_allocate(ns)
                allocate_entered.set()
                self.assertTrue(allocate_release.wait(5), "allocation hook was not released")
                return run_id, path

            def slow_copy(src, dst, *args, **kwargs):
                copy_entered.set()
                self.assertTrue(copy_release.wait(5), "adopt copy hook was not released")
                return real_copy(src, dst, *args, **kwargs)

            runner._execute = lambda *_args, **_kwargs: runner._slot.release()
            runner._allocate_run_dir = hooked_allocate
            holder: dict = {}

            def run_start():
                holder["status"] = runner.start(protocol, cases=100, seed=1, bootstrap_iterations=20)

            starter = threading.Thread(target=run_start)
            starter.start()
            self.assertTrue(allocate_entered.wait(5))
            cleanup_finished = threading.Event()

            def run_cleanup():
                runner.cleanup()
                cleanup_finished.set()

            with mock.patch.object(runner_mod.shutil, "rmtree", spy_rmtree):
                cleaner = threading.Thread(target=run_cleanup)
                cleaner.start()
                time.sleep(0.2)
                self.assertFalse(cleanup_finished.is_set(), "cleanup ran before the new run had status.json")
                born = [path for path in Path(directory).iterdir() if len(path.name) == 32 and path.is_dir()]
                self.assertEqual(len(born), 1)
                self.assertTrue(born[0].is_dir())
                allocate_release.set()
                self.assertTrue(starter.join(5) is None and not starter.is_alive())
                self.assertTrue(cleaner.join(5) is None and not cleaner.is_alive())
                status = holder["status"]
                published = Path(directory) / status["run_id"]
                self.assertTrue((published / "status.json").is_file())
                self.assertNotIn(status["run_id"], removed)

                source = Path(directory) / "ccw-src"
                source.mkdir()
                (source / "ccw_preview_report.html").write_text("<html>kept</html>", encoding="utf-8")
                adopted: dict = {}

                def run_adopt():
                    adopted["status"] = runner.adopt_ccw_directory(source, gate_status="passed", seed=9, case_count=11)

                with mock.patch.object(runner_mod.shutil, "copyfile", slow_copy):
                    adopter = threading.Thread(target=run_adopt)
                    adopter.start()
                    self.assertTrue(copy_entered.wait(5))
                    for _ in range(20):
                        runner.cleanup()
                    copy_release.set()
                    self.assertTrue(adopter.join(5) is None and not adopter.is_alive())
                adopted_id = adopted["status"]["run_id"]
                adopted_dir = Path(directory) / adopted_id
                self.assertTrue(adopted_dir.is_dir())
                self.assertTrue((adopted_dir / "status.json").is_file())
                self.assertTrue((adopted_dir / "ccw_preview_report.html").is_file())
                saved = json.loads((adopted_dir / "status.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["state"], "completed")
                self.assertNotIn(adopted_id, removed)
