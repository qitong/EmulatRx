"""End-to-end service boundaries using only synthetic inputs."""

import base64
import importlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from starlette.testclient import TestClient

from emulatrx_demo import SIMULATION_LABEL
from emulatrx_demo.__main__ import _code_provenance
from emulatrx_demo._common import ContractError
from emulatrx_preview.runner import Runner


class PreviewServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / "web").write_text("web-" + "w" * 40)
        (root / "mcp").write_text("agent-" + "a" * 40)
        patcher = mock.patch.dict(os.environ, {
            "EMULATRX_RUNS_DIR": str(root / "runs"),
            "EMULATRX_WEB_PASSWORD_FILE": str(root / "web"),
            "EMULATRX_MCP_TOKEN_FILE": str(root / "mcp"),
            "EMULATRX_CODE_COMMIT": "b" * 40,
            "EMULATRX_CODE_DIRTY": "false",
            "EMULATRX_ALLOWED_HOSTS": "testserver",
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        from emulatrx_preview import app
        self.service = importlib.reload(app)
        self.web_auth = {"Authorization": "Basic " + base64.b64encode(("preview:" + (root / "web").read_text()).encode()).decode()}
        self.mcp_auth = {"Authorization": "Bearer " + (root / "mcp").read_text()}
        self.client = TestClient(self.service.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_authentication_and_browser_end_to_end(self):
        self.assertEqual(self.client.get("/").status_code, 401)
        self.assertEqual(self.client.get("/mcp", headers=self.web_auth).status_code, 401)
        self.assertEqual(self.client.get("/", headers=self.mcp_auth).status_code, 401)
        page = self.client.get("/", headers=self.web_auth)
        self.assertEqual(page.status_code, 200)
        self.assertIn("非真实患者", page.text)
        created = self.client.post("/runs", data={"template_version": "1.0", "case_count": "12", "seed": "42"},
                                   headers=self.web_auth, follow_redirects=False)
        self.assertEqual(created.status_code, 303, created.text)
        run_page = self.client.get(created.headers["location"], headers=self.web_auth)
        self.assertEqual(run_page.status_code, 200)
        self.assertIn("code_commit", run_page.text)
        self.assertIn("Arbitrary synthetic", run_page.text)
        csv_response = self.client.get(created.headers["location"] + "/cases.csv", headers=self.web_auth)
        self.assertEqual(csv_response.status_code, 200)
        self.assertIn(SIMULATION_LABEL, csv_response.text)
        self.assertEqual(self.client.get(created.headers["location"] + "/cases.csv").status_code, 401)

    def test_mcp_streamable_http_calls_same_runner(self):
        headers = {**self.mcp_auth, "Accept": "application/json, text/event-stream"}
        listed = self.client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=headers)
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertIn("run_synthetic_demo", listed.text)
        called = self.client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "run_synthetic_demo", "arguments": {"case_count": 12, "seed": 42}}}, headers=headers)
        self.assertEqual(called.status_code, 200, called.text)
        self.assertIn("synthetic_demo_generated", called.text)
        self.assertIn("b" * 40, called.text)
        self.assertEqual(self.client.post("/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"}).status_code, 401)

    def test_error_gates_and_recovery(self):
        data = {"template_version": "1.0", "case_count": "12", "seed": "42"}
        self.assertEqual(self.client.post("/runs", data=data, headers={**self.web_auth, "Origin": "https://other.example", "Host": "testserver"}).status_code, 403)
        self.assertEqual(self.client.post("/runs", json={"patient_file": "x"}, headers=self.web_auth).status_code, 415)
        self.assertEqual(self.client.post("/runs", data={**data, "patient_id": "x"}, headers=self.web_auth).status_code, 400)
        self.assertEqual(self.client.post("/runs", data={**data, "case_count": "10001"}, headers=self.web_auth).status_code, 400)
        self.assertEqual(self.client.get("/runs/../bad", headers=self.web_auth).status_code, 404)
        result = self.service.runner.run({"template_version": "1.0", "scenario": "synthetic_tutorial_v1", "case_count": 12, "seed": 42, "simulation_label": SIMULATION_LABEL})
        restarted = Runner(self.service.runner.root)
        self.assertEqual(restarted.get(result["run_id"])["summary"], result["summary"])
        self.assertEqual(result["manifest"]["code_commit"], "b" * 40)

    def test_busy_gate(self):
        runner = self.service.runner
        self.assertTrue(runner._slot.acquire(blocking=False))
        try:
            with self.assertRaises(ContractError) as raised:
                runner.run({"template_version": "1.0", "scenario": "synthetic_tutorial_v1", "case_count": 1, "seed": 1, "simulation_label": SIMULATION_LABEL})
            self.assertEqual(raised.exception.code, "BUSY")
        finally:
            runner._slot.release()

    def test_build_provenance_requires_valid_commit(self):
        self.assertEqual(_code_provenance(), ("b" * 40, False))
        with mock.patch.dict(os.environ, {"EMULATRX_CODE_COMMIT": "bad"}):
            with self.assertRaises(ContractError):
                _code_provenance()


if __name__ == "__main__":
    unittest.main()
