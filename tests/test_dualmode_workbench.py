"""End-to-end Web REST API & MCP service tests for the Issue #15 Dual-Mode TTE Workbench."""

from __future__ import annotations

import base64
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
from urllib.error import URLError

from starlette.testclient import TestClient

from emulatrx_tte import llm_assist, rct_matcher
from emulatrx_tte.sample_cohorts import generate_sample_cohort_csv_bytes


class _FakeHTTPResp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


MOCK_STUDIES = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT03036124",
                    "briefTitle": "Dapagliflozin in Patients With Chronic Heart Failure (DAPA-HF)",
                    "officialTitle": "Effect of Dapagliflozin on Worsening Heart Failure or Cardiovascular Death",
                },
                "statusModule": {"overallStatus": "COMPLETED"},
                "designModule": {
                    "studyType": "INTERVENTIONAL",
                    "phases": ["PHASE3"],
                    "enrollmentInfo": {"count": 4744},
                    "designInfo": {"allocation": "RANDOMIZED"},
                },
                "conditionsModule": {"conditions": ["Heart Failure", "Type 2 Diabetes"]},
                "armsInterventionsModule": {
                    "armGroups": [
                        {"label": "Dapagliflozin 10 mg", "type": "EXPERIMENTAL"},
                        {"label": "Sitagliptin / Standard Care", "type": "ACTIVE_COMPARATOR"},
                    ],
                    "interventions": [
                        {"name": "Dapagliflozin", "type": "DRUG"},
                        {"name": "Sitagliptin", "type": "DRUG"},
                    ],
                },
                "eligibilityModule": {
                    "minimumAge": "18 Years",
                    "maximumAge": "N/A",
                    "sex": "ALL",
                    "eligibilityCriteria": "Inclusion Criteria:\n* Age >= 45 years\n* eGFR >= 30 mL/min/1.73m2",
                },
                "outcomesModule": {
                    "primaryOutcomes": [
                        {
                            "measure": "Worsening Heart Failure or Cardiovascular Death",
                            "timeFrame": "Up to 18 months",
                        }
                    ]
                },
            },
            "hasResults": True,
        }
    ]
}


class DualModeWorkbenchServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        (self.base / "web_secret").write_text("w" * 40, encoding="utf-8")
        (self.base / "mcp_secret").write_text("m" * 40, encoding="utf-8")
        self.env_patch = mock.patch.dict(
            os.environ,
            {
                "EMULATRX_RUNS_DIR": str(self.base / "data"),
                "EMULATRX_WEB_PASSWORD_FILE": str(self.base / "web_secret"),
                "EMULATRX_MCP_TOKEN_FILE": str(self.base / "mcp_secret"),
                "EMULATRX_CODE_COMMIT": "c" * 40,
                "EMULATRX_CODE_DIRTY": "false",
                "EMULATRX_ALLOWED_HOSTS": "testserver",
                "EMULATRX_TTE_DEPENDENCY_COMMIT": "c" * 40,
            },
            clear=False,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

        from emulatrx_preview import app as preview_app
        self.service = importlib.reload(preview_app)
        self.web_auth = {
            "Authorization": "Basic " + base64.b64encode(("preview:" + "w" * 40).encode()).decode()
        }
        self.mcp_auth = {
            "Authorization": "Bearer " + "m" * 40,
            "Accept": "application/json, text/event-stream",
        }
        self.client = TestClient(self.service.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def wait_dual_run(self, run_id: str) -> dict:
        for _ in range(200):
            resp = self.client.get(f"/api/dual-runs/{run_id}", headers=self.web_auth)
            self.assertEqual(resp.status_code, 200, resp.text)
            payload = resp.json()
            if payload["state"] not in ("queued", "running"):
                return payload
            time.sleep(0.1)
        self.fail(f"Dual-mode run {run_id} did not finish within 20s")

    def mcp_call(self, name: str, arguments: dict, rpc_id: int = 1) -> dict:
        resp = self.client.post(
            "/mcp",
            headers=self.mcp_auth,
            json={"jsonrpc": "2.0", "id": rpc_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        res = resp.json()["result"]
        self.assertFalse(res["isError"], res)
        return json.loads(res["content"][0]["text"])

    def test_path1_user_cohort_csv_upload_privacy_guard_rct_match_and_physical_session_wipe(self):
        """E2E Path 1: Upload real-header CSV -> LLM privacy assertion -> Match RCT -> Map & Run user_cohort -> Wipe session."""
        # 1. Verify Workbench Home & Sample CSV Download
        home = self.client.get("/", headers=self.web_auth)
        self.assertEqual(home.status_code, 200)
        self.assertIn("目标试验模拟（TTE）双轨研究工作台", home.text)
        self.assertIn("我有一批自有队列数据", home.text)
        self.assertIn("我只有研究想法", home.text)

        dl = self.client.get("/api/samples/cardiorenal_sglt2i/download", headers=self.web_auth)
        self.assertEqual(dl.status_code, 200)
        self.assertIn("attachment", dl.headers["content-disposition"])
        self.assertIn("Hosp-2023-0001", dl.text)
        self.assertNotIn("SYN-", dl.text)

        # 2. Upload CSV via multipart/form-data while mocking OpenAI LLM & intercepting outbound HTTP payload
        captured_llm_bodies: list[str] = []

        def fake_llm(req, timeout):
            body_txt = req.data.decode("utf-8")
            captured_llm_bodies.append(body_txt)
            if "rankings" in body_txt:
                ans = {"rankings": [{"nct_id": "NCT03036124", "score": 96, "reason_zh": "达格列净心肾终点高度对齐"}]}
            else:
                ans = {
                    "domain_summary_zh": "2型糖尿病合并心衰风险队列",
                    "condition_query_en": "Heart Failure Type 2 Diabetes",
                    "intervention_query_en": "Dapagliflozin",
                    "search_query_en": "Dapagliflozin Heart Failure",
                }
            return _FakeHTTPResp(json.dumps({"choices": [{"message": {"content": json.dumps(ans)}}]}).encode("utf-8"))

        def fake_ctgov(req, timeout):
            return _FakeHTTPResp(json.dumps(MOCK_STUDIES).encode("utf-8"))

        with mock.patch.dict(
            os.environ,
            {
                "EMULATRX_LLM_API_KEY_FILE": str(self.base / "missing_llm_secret"),
                "EMULATRX_LLM_API_KEY": "sk-test-e2e-key",
                "EMULATRX_LLM_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/",
                "EMULATRX_LLM_MODEL": "qwen-plus",
            },
            clear=False,
        ):
            with mock.patch.object(llm_assist, "urlopen_llm", side_effect=fake_llm), mock.patch.object(
                rct_matcher, "urlopen_ctgov", side_effect=fake_ctgov
            ):
                up = self.client.post(
                    "/api/sessions/upload",
                    headers=self.web_auth,
                    files={"file": ("hospital_export.csv", dl.content, "text/csv")},
                    data={"user_note": "心内科脱敏出院随访队列"},
                )
                self.assertEqual(up.status_code, 200, up.text)
                up_json = up.json()
                session_id = up_json["session_id"]
                self.assertEqual(up_json["profile"]["row_count"], 1500)
                self.assertEqual(up_json["llm_status"]["mode"], "llm_assisted")

                # PRIVACY REDLINE ASSERTION on E2E HTTP upload: zero Hosp-2023-* IDs in LLM request!
                self.assertGreaterEqual(len(captured_llm_bodies), 1)
                for body_str in captured_llm_bodies:
                    self.assertNotIn("Hosp-2023-", body_str)
                    self.assertNotIn("deid_patient_id", json.loads(body_str)["messages"][1]["content"])

                # 3. Search ClinicalTrials.gov candidate RCTs
                srch = self.client.post(
                    "/api/rcts/search",
                    headers=self.web_auth,
                    json={
                        "query": up_json["interpretation"]["search_query_en"],
                        "condition": up_json["interpretation"]["condition_query_en"],
                        "intervention": up_json["interpretation"]["intervention_query_en"],
                    },
                )
                self.assertEqual(srch.status_code, 200, srch.text)
                rct_card = srch.json()["candidates"][0]
                self.assertEqual(rct_card["nct_id"], "NCT03036124")
                self.assertEqual(rct_card["match_score"], 96)

                # 4. Draft ColumnMappingSpec aligned to NCT03036124
                draft = self.client.post(
                    "/api/rcts/draft",
                    headers=self.web_auth,
                    json={"project_mode": "user_cohort", "session_id": session_id, "target_rct": rct_card},
                )
                self.assertEqual(draft.status_code, 200, draft.text)
                mapping = draft.json()["mapping"]
                self.assertEqual(mapping["treatment_column"], "initiated_drug")
                self.assertEqual(mapping["comparator_value"], "Sitagliptin")
                self.assertEqual(mapping["active_value"], "Dapagliflozin")
                self.assertEqual(mapping["time_unit"], "months")
                mapping["bootstrap_iterations"] = 30

                # 5. Launch user_cohort TTE run
                launched = self.client.post(
                    "/api/dual-runs",
                    headers=self.web_auth,
                    json={
                        "project_mode": "user_cohort",
                        "session_id": session_id,
                        "mapping": mapping,
                        "target_rct": rct_card,
                    },
                )
                self.assertEqual(launched.status_code, 202, launched.text)
                run_id = launched.json()["run_id"]
                final = self.wait_dual_run(run_id)
                self.assertEqual(final["state"], "completed")
                self.assertEqual(final["gate_status"], "passed")
                self.assertEqual(final["estimates"]["source_mode"], "user_cohort")
                self.assertIn("真实队列目标试验模拟（真项目）分析报告", final["rendered_html"])
                self.assertIn("观察性因果推断四大识别前提", final["rendered_html"])
                self.assertIn("class='cif-step'", final["rendered_html"])
                self.assertIn("Love Plot", final["rendered_html"])

                # Check standalone printable/downloadable HTML report
                rep = self.client.get(f"/runs/{run_id}/report?download=1", headers=self.web_auth)
                self.assertEqual(rep.status_code, 200)
                self.assertIn("attachment", rep.headers["content-disposition"])
                self.assertIn("真实队列目标试验模拟（真项目）分析报告", rep.text)
                self.assertNotIn("<textarea", rep.text)

                # 6. Verify session directory & run directory exist before delete, then physically wipe via DELETE
                sess_dir = self.service.runner.sessions_root / session_id
                run_dir = self.service.runner.root / run_id
                self.assertTrue(sess_dir.is_dir())
                self.assertTrue(run_dir.is_dir())

                wiped = self.client.delete(f"/api/sessions/{session_id}", headers=self.web_auth)
                self.assertEqual(wiped.status_code, 200, wiped.text)
                wipe_data = wiped.json()
                self.assertEqual(wipe_data["status"], "session_deleted")
                self.assertFalse(wipe_data["session_exists_on_disk"])
                self.assertFalse(sess_dir.exists())
                self.assertFalse(run_dir.exists())
                self.assertEqual(self.client.get(f"/api/dual-runs/{run_id}", headers=self.web_auth).status_code, 404)

    def test_path2_idea_to_simulated_cohort_and_deterministic_offline_fallback(self):
        """E2E Path 2: Clinical Idea (no LLM key, offline CT.gov fallback) -> PICO -> Target RCT -> simulated_cohort report."""
        env_no_key = {k: v for k, v in os.environ.items() if k not in ("EMULATRX_LLM_API_KEY", "OPENAI_API_KEY")}
        with mock.patch.dict(os.environ, env_no_key, clear=True):
            with mock.patch.object(rct_matcher, "urlopen_ctgov", side_effect=URLError("offline")):
                parsed = self.client.post(
                    "/api/idea/parse",
                    headers=self.web_auth,
                    json={"idea_text": "帕博利珠单抗联合含铂化疗对比单纯含铂双药化疗一线治疗晚期非小细胞肺癌，评估进展死亡与非肿瘤竞争死亡"},
                )
                self.assertEqual(parsed.status_code, 200, parsed.text)
                pico = parsed.json()["pico"]
                self.assertEqual(pico["assist_mode"], "deterministic_rules")

                srch = self.client.post(
                    "/api/rcts/search",
                    headers=self.web_auth,
                    json={"query": pico["search_query_en"]},
                )
                self.assertEqual(srch.status_code, 200)
                self.assertEqual(srch.json()["source"], "offline_curated_catalog_fallback")
                rct_card = srch.json()["candidates"][0]
                self.assertEqual(rct_card["nct_id"], "NCT02578680")

                draft = self.client.post(
                    "/api/rcts/draft",
                    headers=self.web_auth,
                    json={"project_mode": "simulated_cohort", "target_rct": rct_card, "pico": pico},
                )
                self.assertEqual(draft.status_code, 200)
                self.assertIn("NCT02578680", draft.json()["protocol"]["study_title"])

                launched = self.client.post(
                    "/api/dual-runs",
                    headers=self.web_auth,
                    json={
                        "project_mode": "simulated_cohort",
                        "target_rct": rct_card,
                        "pico": pico,
                        "sim_config": {
                            "cases": 800,
                            "confounding_strength": "medium",
                            "effect_direction": "protective_strong",
                            "competing_rate_level": "medium",
                            "weight_method": "overlap",
                            "seed": 42,
                            "bootstrap_iterations": 25,
                        },
                    },
                )
                self.assertEqual(launched.status_code, 202, launched.text)
                run_id = launched.json()["run_id"]
                final = self.wait_dual_run(run_id)
                self.assertEqual(final["state"], "completed")
                self.assertEqual(final["estimates"]["source_mode"], "simulated_cohort")
                self.assertIn("目标试验模拟（无数据模拟预演项目）可行性报告", final["rendered_html"])
                self.assertIn("预设模拟真值 vs 加权 TTE 估计恢复对比表", final["rendered_html"])
                self.assertIn("class='cif-step'", final["rendered_html"])

    def test_mcp_dualmode_tools_end_to_end(self):
        with mock.patch.object(rct_matcher, "urlopen_ctgov", side_effect=URLError("offline")):
            listed = self.client.post("/mcp", headers=self.mcp_auth, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            self.assertEqual(listed.status_code, 200)
            for tool_name in (
                "search_target_rcts",
                "start_simulated_rct_tte",
                "profile_cohort_session",
                "start_user_cohort_tte",
                "delete_cohort_session",
            ):
                self.assertIn(tool_name, listed.text)

            rcts = self.mcp_call("search_target_rcts", {"idea_text": "达格列净治疗心力衰竭"})
            self.assertEqual(rcts["candidates"][0]["nct_id"], "NCT03036124")

            sim_run = self.mcp_call(
                "start_simulated_rct_tte",
                {"nct_id": "NCT03036124", "cases": 600, "bootstrap_iterations": 20},
            )
            self.wait_dual_run(sim_run["run_id"])
            sim_status = self.mcp_call("get_tte_run", {"run_id": sim_run["run_id"]})
            self.assertEqual(sim_status["state"], "completed")

            _, raw_onc = generate_sample_cohort_csv_bytes("oncology_io_nsclc")
            prof_res = self.mcp_call(
                "profile_cohort_session",
                {"csv_base64": base64.b64encode(raw_onc).decode("ascii"), "filename": "onc.csv"},
            )
            sid = prof_res["session_id"]
            uc_run = self.mcp_call(
                "start_user_cohort_tte",
                {"session_id": sid, "nct_id": "NCT02578680", "bootstrap_iterations": 20},
            )
            self.wait_dual_run(uc_run["run_id"])
            uc_status = self.mcp_call("get_tte_run", {"run_id": uc_run["run_id"]})
            self.assertEqual(uc_status["state"], "completed")

            chunk = self.mcp_call("read_tte_artifact", {"run_id": uc_run["run_id"], "name": "bundle/audit_report.md", "limit": 256})
            self.assertIn("真实队列目标试验模拟", base64.b64decode(chunk["data"]).decode("utf-8"))

            del_res = self.mcp_call("delete_cohort_session", {"session_id": sid})
            self.assertEqual(del_res["status"], "session_deleted")
            self.assertFalse(del_res["session_exists_on_disk"])

    def test_ymadmin_secret_file_contract_and_zero_secret_leak_in_web_responses(self):
        secret_path = self.base / "ymadmin_llm_secret"
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("EMULATRX_LLM_API_KEY_FILE", "EMULATRX_LLM_API_KEY", "OPENAI_API_KEY")
        }
        # 1. Missing secret file -> container/service boots cleanly and reports unmounted key deterministic fallback
        with mock.patch.object(llm_assist, "DEFAULT_LLM_SECRET_FILE", self.base / "no_default_secret"):
            with mock.patch.dict(os.environ, {**clean_env, "EMULATRX_LLM_API_KEY_FILE": str(secret_path)}, clear=True):
                boot_no_key = self.client.get("/api/workbench/bootstrap", headers=self.web_auth)
                self.assertEqual(boot_no_key.status_code, 200)
                self.assertFalse(boot_no_key.json()["llm_status"]["enabled"])
                self.assertIn("当前未挂载模型密钥，已启用确定性规则引擎兜底", boot_no_key.json()["llm_status"]["label_zh"])

                # 2. Provision secret file -> service activates LLM mode without leaking secret anywhere
                secret_val = "sk-YMADMIN-BAILIAN-DO-NOT-ECHO-1122334455"
                secret_path.write_text(secret_val + "\n", encoding="utf-8")
                boot_with_key = self.client.get("/api/workbench/bootstrap", headers=self.web_auth)
                self.assertEqual(boot_with_key.status_code, 200)
                self.assertTrue(boot_with_key.json()["llm_status"]["enabled"])
                self.assertEqual(boot_with_key.json()["llm_status"]["key_source"], "secret_file")
                self.assertNotIn(secret_val, boot_with_key.text)

                home_with_key = self.client.get("/", headers=self.web_auth)
                self.assertEqual(home_with_key.status_code, 200)
                self.assertNotIn(secret_val, home_with_key.text)


if __name__ == "__main__":
    unittest.main()
