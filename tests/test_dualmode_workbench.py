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
                self.assertEqual(rct_card["arm_alignment_status"], "exact_dual_arm")
                self.assertEqual(rct_card["match_score"], 0.96)

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

    def test_codex_privacy_counterexample_e2e_zero_leak_across_upload_note_search_draft(self):
        """Regression test (Blocking 1): 20-row cohort with site_group='Alice Smith' & user_note='Patient identifier 123456789' leaks 0 tokens across all endpoints."""
        rows = ["deid_id,site_group,initiated_drug,followup_months,endpoint,age_years"]
        for i in range(20):
            site = "Alice Smith" if i < 12 else "Bob Jones"
            drug = "Dapagliflozin" if i % 2 == 0 else "Sitagliptin"
            ep = "censored" if i < 10 else ("hf_hospitalization" if i < 16 else "cv_death")
            rows.append(f"PT-{i + 1:04d},{site},{drug},{10.0 + i * 0.8:.1f},{ep},{55 + (i % 15)}")
        csv_bytes = ("\n".join(rows) + "\n").encode("utf-8")

        captured_llm_bodies: list[str] = []

        def fake_llm(req, timeout):
            body_txt = req.data.decode("utf-8")
            captured_llm_bodies.append(body_txt)
            if "rankings" in body_txt:
                ans = {"rankings": [{"nct_id": "NCT03036124", "score": 97, "reason_zh": "达格列净心衰终点匹配"}]}
            else:
                ans = {
                    "domain_summary_zh": "心衰观察性队列",
                    "condition_query_en": "Heart Failure",
                    "intervention_query_en": "Dapagliflozin",
                    "search_query_en": "Dapagliflozin Heart Failure",
                }
            return _FakeHTTPResp(json.dumps({"choices": [{"message": {"content": json.dumps(ans)}}]}).encode("utf-8"))

        def fake_ctgov(req, timeout):
            return _FakeHTTPResp(json.dumps(MOCK_STUDIES).encode("utf-8"))

        with mock.patch.dict(
            os.environ,
            {
                "EMULATRX_LLM_API_KEY_FILE": str(self.base / "absent_secret"),
                "EMULATRX_LLM_API_KEY": "sk-codex-regression-key",
            },
            clear=False,
        ):
            with mock.patch.object(llm_assist, "urlopen_llm", side_effect=fake_llm), mock.patch.object(
                rct_matcher, "urlopen_ctgov", side_effect=fake_ctgov
            ):
                # 1. Initial upload profiling
                up = self.client.post(
                    "/api/sessions/upload",
                    headers=self.web_auth,
                    files={"file": ("alice_site.csv", csv_bytes, "text/csv")},
                    data={"user_note": "Patient identifier 123456789 达格列净心衰"},
                )
                self.assertEqual(up.status_code, 200, up.text)
                sid = up.json()["session_id"]
                # Local browser profile keeps raw category labels for local user selection
                site_col = next(c for c in up.json()["profile"]["columns"] if c["name"] == "site_group")
                self.assertEqual(site_col["top_categories"][0]["value"], "Alice Smith")

                # 2. Update session note with new identifier & person name
                note_resp = self.client.post(
                    f"/api/sessions/{sid}/note",
                    headers=self.web_auth,
                    json={"user_note": "Updated patient identifier 987654321 investigator Carol White 心衰"},
                )
                self.assertEqual(note_resp.status_code, 200, note_resp.text)

                # 3. Search & LLM-rerank candidate RCTs bound to this session
                srch = self.client.post(
                    "/api/rcts/search",
                    headers=self.web_auth,
                    json={
                        "session_id": sid,
                        "project_mode": "user_cohort",
                        "query": "Dapagliflozin Heart Failure",
                        "condition": "Heart Failure",
                        "intervention": "Dapagliflozin",
                        "context_summary": {
                            "domain_summary_zh": "Patient identifier 123456789 Alice Smith 达格列净心衰",
                            "search_query_en": "Dapagliflozin Heart Failure",
                        },
                    },
                )
                self.assertEqual(srch.status_code, 200, srch.text)
                self.assertEqual(srch.json()["ranking_mode"], "llm_assisted")
                rct_card = srch.json()["candidates"][0]

                # 4. Draft mapping aligned to selected RCT
                draft = self.client.post(
                    "/api/rcts/draft",
                    headers=self.web_auth,
                    json={"project_mode": "user_cohort", "session_id": sid, "target_rct": rct_card},
                )
                self.assertEqual(draft.status_code, 200, draft.text)

        # Every outbound LLM request across upload, note update, and RCT rerank must contain ZERO sensitive tokens
        self.assertGreaterEqual(len(captured_llm_bodies), 3)
        for idx, body_str in enumerate(captured_llm_bodies):
            for forbidden in ("Alice Smith", "Bob Jones", "Carol White", "123456789", "987654321", "PT-00"):
                self.assertNotIn(forbidden, body_str, f"Leak '{forbidden}' in LLM call #{idx + 1}: {body_str}")
        self.assertIn("CATEGORY_LEVEL_1", captured_llm_bodies[0])

    def test_session_wipe_unconditional_run_sweep_on_corrupt_session_meta(self):
        """Regression test (Blocking 2): corrupt session_meta.json while session dir exists still deletes all derived runs via status.json sweep."""
        up = self.client.post(
            "/api/sessions/upload",
            headers=self.web_auth,
            json={"sample_id": "cardiorenal_sglt2i"},
        )
        self.assertEqual(up.status_code, 200, up.text)
        sid = up.json()["session_id"]
        mapping = up.json()["default_mapping"]
        mapping["bootstrap_iterations"] = 20

        launched = self.client.post(
            "/api/dual-runs",
            headers=self.web_auth,
            json={"project_mode": "user_cohort", "session_id": sid, "mapping": mapping},
        )
        self.assertEqual(launched.status_code, 202, launched.text)
        run_id_1 = launched.json()["run_id"]
        final = self.wait_dual_run(run_id_1)
        self.assertEqual(final["state"], "completed")

        # Create a second orphaned run directory referencing the same session_id in status.json
        run_id_2 = "b" * 32
        orphan_run_dir = self.service.runner.root / run_id_2
        orphan_run_dir.mkdir(mode=0o700)
        (orphan_run_dir / "status.json").write_text(
            json.dumps({"run_id": run_id_2, "session_id": sid, "state": "completed"}),
            encoding="utf-8",
        )
        (orphan_run_dir / "weighted_cohort.csv").write_text("id,w\n1,0.5\n", encoding="utf-8")

        # Corrupt session_meta.json while session directory still exists on disk (Codex probe)
        sess_dir = self.service.runner.sessions_root / sid
        run_dir_1 = self.service.runner.root / run_id_1
        (sess_dir / "session_meta.json").write_bytes(b"{corrupt_partial_json_without_run_ids")
        self.assertTrue(sess_dir.is_dir())
        self.assertTrue(run_dir_1.is_dir())
        self.assertTrue(orphan_run_dir.is_dir())

        wiped = self.client.delete(f"/api/sessions/{sid}", headers=self.web_auth)
        self.assertEqual(wiped.status_code, 200, wiped.text)
        wdata = wiped.json()
        self.assertTrue(wdata["session_deleted"])
        self.assertFalse(wdata["session_exists_on_disk"])
        self.assertIn(run_id_1, wdata["purged_runs"])
        self.assertIn(run_id_2, wdata["purged_runs"])
        self.assertFalse(sess_dir.exists())
        self.assertFalse(run_dir_1.exists())
        self.assertFalse(orphan_run_dir.exists())

    def test_streaming_25mb_upload_limit_returns_413_cleanly(self):
        """Regression test (Non-blocking 1): oversized >25 MB multipart uploads abort in streaming chunks and return HTTP 413."""
        # 1. Oversized Content-Length header rejected immediately with 413
        resp_hdr = self.client.post(
            "/api/sessions/upload",
            headers={
                **self.web_auth,
                "Content-Type": "multipart/form-data; boundary=----OversizedBoundary",
                "Content-Length": str(30 * 1024 * 1024),
            },
            content=b"------OversizedBoundary--\r\n",
        )
        self.assertEqual(resp_hdr.status_code, 413, resp_hdr.text)
        self.assertEqual(resp_hdr.json()["code"], "COHORT_FILE_TOO_LARGE")

        # 2. Actual >25 MB file body (within 128 KiB wire overhead so chunked upload.read(65536) loop triggers) returns 413
        oversized_file_bytes = b"id,arm,time,event\n" + (b"1,0,12.0,0\n" * ((25 * 1024 * 1024 // 11) + 2048))
        self.assertGreater(len(oversized_file_bytes), 25 * 1024 * 1024)
        self.assertLess(len(oversized_file_bytes), 25 * 1024 * 1024 + 64 * 1024)
        resp_stream = self.client.post(
            "/api/sessions/upload",
            headers=self.web_auth,
            files={"file": ("huge_cohort.csv", oversized_file_bytes, "text/csv")},
        )
        self.assertEqual(resp_stream.status_code, 413, resp_stream.text)
        self.assertEqual(resp_stream.json()["code"], "COHORT_FILE_TOO_LARGE")
        self.assertIn("25 MB", resp_stream.json()["error"])

    def test_common_english_cohort_categories_patient_claims_study_full_4step_e2e(self):
        """Regression test (Codex Round 2): 20-row cohort with data_origin='patient'/'claims'/'study' & column 'source' completes all 4 steps with 0 false-positive 400s."""
        rows = ["deid_id,data_origin,source,initiated_drug,followup_months,endpoint,age_years"]
        origins = ["patient"] * 8 + ["claims"] * 7 + ["study"] * 5
        sources = ["hospital"] * 10 + ["source"] * 10
        for i in range(20):
            drug = "Dapagliflozin" if i % 2 == 0 else "Sitagliptin"
            if i in (0, 2, 4, 1, 3, 5, 7):
                ep = "hf_hospitalization"
            elif i in (6, 8, 9, 11):
                ep = "cv_death"
            else:
                ep = "censored"
            fu = 6.0 + (i % 10) * 1.8
            age = 52 + (i % 18)
            rows.append(f"PT-{i + 1:04d},{origins[i]},{sources[i]},{drug},{fu:.1f},{ep},{age}")
        csv_bytes = ("\n".join(rows) + "\n").encode("utf-8")

        claims_trial_payload = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT03036124",
                            "briefTitle": "Adult patients in claims study of Dapagliflozin in Heart Failure",
                            "officialTitle": "Hospital study of Dapagliflozin vs Sitagliptin in adult patients",
                        },
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {
                            "studyType": "INTERVENTIONAL",
                            "phases": ["PHASE3"],
                            "enrollmentInfo": {"count": 4744},
                            "designInfo": {"allocation": "RANDOMIZED"},
                        },
                        "conditionsModule": {"conditions": ["Heart Failure"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Dapagliflozin 10 mg", "type": "EXPERIMENTAL"},
                                {"label": "Sitagliptin / Standard Control", "type": "ACTIVE_COMPARATOR"},
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
                            "eligibilityCriteria": "Inclusion Criteria:\n* Adult patients in claims study >= 18 years",
                        },
                        "outcomesModule": {
                            "primaryOutcomes": [
                                {
                                    "measure": "Worsening Heart Failure or Cardiovascular Death in Study Patients",
                                    "timeFrame": "Up to 18 months",
                                }
                            ]
                        },
                    },
                    "hasResults": True,
                }
            ]
        }

        captured_user_envelopes: list[dict] = []

        def fake_llm(req, timeout):
            outer = json.loads(req.data.decode("utf-8"))
            user_env = json.loads(outer["messages"][1]["content"])
            captured_user_envelopes.append(user_env)
            if "public_reference_context" in user_env:
                ans = {"rankings": [{"nct_id": "NCT03036124", "score": 96, "reason_zh": "心衰终点与暴露策略一致"}]}
            else:
                ans = {
                    "domain_summary_zh": "心衰观察性队列",
                    "condition_query_en": "Heart Failure",
                    "intervention_query_en": "Dapagliflozin",
                    "search_query_en": "Dapagliflozin Heart Failure",
                }
            return _FakeHTTPResp(json.dumps({"choices": [{"message": {"content": json.dumps(ans)}}]}).encode("utf-8"))

        def fake_ctgov(req, timeout):
            return _FakeHTTPResp(json.dumps(claims_trial_payload).encode("utf-8"))

        with mock.patch.dict(
            os.environ,
            {
                "EMULATRX_LLM_API_KEY_FILE": str(self.base / "absent_secret"),
                "EMULATRX_LLM_API_KEY": "sk-deepseek-online-sim",
                "EMULATRX_LLM_BASE_URL": "https://api.deepseek.com",
                "EMULATRX_LLM_MODEL": "deepseek-chat",
            },
            clear=False,
        ):
            with mock.patch.object(llm_assist, "urlopen_llm", side_effect=fake_llm), mock.patch.object(
                rct_matcher, "urlopen_ctgov", side_effect=fake_ctgov
            ):
                # Step 1: Upload 20-row cohort with data_origin='patient'/'claims'/'study' -> HTTP 200
                up = self.client.post(
                    "/api/sessions/upload",
                    headers=self.web_auth,
                    files={"file": ("claims_cohort.csv", csv_bytes, "text/csv")},
                    data={"user_note": "real-world patient claims study 达格列净治疗心衰"},
                )
                self.assertEqual(up.status_code, 200, up.text)
                sid = up.json()["session_id"]
                self.assertEqual(up.json()["profile"]["row_count"], 20)

                # Step 2: Update session note -> HTTP 200
                note_resp = self.client.post(
                    f"/api/sessions/{sid}/note",
                    headers=self.web_auth,
                    json={"user_note": "hospital baseline patient claims cohort 心力衰竭与达格列净"},
                )
                self.assertEqual(note_resp.status_code, 200, note_resp.text)

                # Step 3: Search & LLM-rerank candidate RCT containing 'Adult patients in claims study' -> HTTP 200
                srch = self.client.post(
                    "/api/rcts/search",
                    headers=self.web_auth,
                    json={
                        "session_id": sid,
                        "project_mode": "user_cohort",
                        "query": up.json()["interpretation"]["search_query_en"],
                        "condition": up.json()["interpretation"]["condition_query_en"],
                        "intervention": up.json()["interpretation"]["intervention_query_en"],
                        "context_summary": up.json()["interpretation"],
                    },
                )
                self.assertEqual(srch.status_code, 200, srch.text)
                self.assertEqual(srch.json()["ranking_mode"], "llm_assisted")
                rct_card = srch.json()["candidates"][0]
                self.assertIn("Adult patients in claims study", rct_card["brief_title"])

                # Step 4: Draft column mapping (HTTP 200) & run user_cohort TTE pipeline to completion
                draft = self.client.post(
                    "/api/rcts/draft",
                    headers=self.web_auth,
                    json={"project_mode": "user_cohort", "session_id": sid, "target_rct": rct_card},
                )
                self.assertEqual(draft.status_code, 200, draft.text)
                mapping = draft.json()["mapping"]
                mapping["covariate_columns"] = ["age_years"]
                mapping["eligibility_criteria"] = []
                mapping["max_abs_smd"] = 0.50
                mapping["bootstrap_iterations"] = 20

                launched = self.client.post(
                    "/api/dual-runs",
                    headers=self.web_auth,
                    json={"project_mode": "user_cohort", "session_id": sid, "mapping": mapping, "target_rct": rct_card},
                )
                self.assertEqual(launched.status_code, 202, launched.text)
                final = self.wait_dual_run(launched.json()["run_id"])
                self.assertEqual(final["state"], "completed", final)
                self.assertEqual(final["gate_status"], "passed", final)

        # Verify all 3 LLM calls had strictly sanitized `cohort_outbound_context` (converted to CATEGORY_LEVEL_*)
        self.assertEqual(len(captured_user_envelopes), 3)
        prof_ctx = captured_user_envelopes[0]["cohort_outbound_context"]
        cols_sent = {c["name"]: c for c in prof_ctx["columns"]}
        self.assertEqual(
            cols_sent["data_origin"]["distinct_categories"],
            ["CATEGORY_LEVEL_1", "CATEGORY_LEVEL_2", "CATEGORY_LEVEL_3"],
        )
        self.assertEqual(
            cols_sent["source"]["distinct_categories"],
            ["CATEGORY_LEVEL_1", "CATEGORY_LEVEL_2"],
        )
        for env in captured_user_envelopes:
            cohort_ctx_json = json.dumps(env["cohort_outbound_context"], ensure_ascii=False)
            for raw_cat in ("patient", "claims", "study", "hospital"):
                self.assertNotIn(f'"{raw_cat}"', cohort_ctx_json)
        # Meanwhile Layer B (`public_reference_context`) safely carried the public ClinicalTrials.gov title
        self.assertIn(
            "Adult patients in claims study",
            json.dumps(captured_user_envelopes[2]["public_reference_context"]),
        )

    def test_nct06642272_sample_cohort_demotion_and_fail_closed_422_arm_mismatch_gate(self):
        """E2E Product QA regression: built-in cardio-renal sample cohort vs NCT06642272 (Empagliflozin vs Dapagliflozin) enforces <=0.45 cap and HTTP 422 gate."""
        nct06642272_and_exact_payload = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT06642272",
                            "briefTitle": "Empagliflozin Versus Dapagliflozin in Type 2 Diabetes and Cardiorenal Risk",
                            "officialTitle": "Randomized Trial Comparing Empagliflozin and Dapagliflozin",
                        },
                        "statusModule": {"overallStatus": "RECRUITING"},
                        "designModule": {
                            "studyType": "INTERVENTIONAL",
                            "phases": ["PHASE4"],
                            "enrollmentInfo": {"count": 800},
                            "designInfo": {"allocation": "RANDOMIZED"},
                        },
                        "conditionsModule": {"conditions": ["Type 2 Diabetes", "Heart Failure"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Empagliflozin 10 mg", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Empagliflozin"]},
                                {"label": "Dapagliflozin 10 mg", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Dapagliflozin"]},
                            ],
                            "interventions": [
                                {"name": "Empagliflozin", "type": "DRUG"},
                                {"name": "Dapagliflozin", "type": "DRUG"},
                            ],
                        },
                        "eligibilityModule": {"minimumAge": "18 Years", "maximumAge": "N/A", "sex": "ALL", "eligibilityCriteria": "* Adults >= 45 years"},
                        "outcomesModule": {"primaryOutcomes": [{"measure": "HF Hospitalization or CV Death", "timeFrame": "Up to 18 months"}]},
                    },
                    "hasResults": False,
                },
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT01095653",
                            "briefTitle": "Dapagliflozin Compared With Sitagliptin in Patients With Type 2 Diabetes",
                            "officialTitle": "Active-Controlled Randomized Trial of Dapagliflozin Versus Sitagliptin",
                        },
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {
                            "studyType": "INTERVENTIONAL",
                            "phases": ["PHASE3"],
                            "enrollmentInfo": {"count": 1450},
                            "designInfo": {"allocation": "RANDOMIZED"},
                        },
                        "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Dapagliflozin 10 mg", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin"]},
                                {"label": "Sitagliptin 100 mg", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Sitagliptin"]},
                            ],
                            "interventions": [
                                {"name": "Dapagliflozin", "type": "DRUG"},
                                {"name": "Sitagliptin", "type": "DRUG"},
                            ],
                        },
                        "eligibilityModule": {"minimumAge": "18 Years", "maximumAge": "N/A", "sex": "ALL", "eligibilityCriteria": "* Adults >= 45 years"},
                        "outcomesModule": {"primaryOutcomes": [{"measure": "Cardiorenal Composite", "timeFrame": "Up to 12 months"}]},
                    },
                    "hasResults": True,
                },
            ]
        }

        def fake_deepseek(req, timeout):
            body_txt = req.data.decode("utf-8")
            if "public_reference_context" in body_txt:
                # Simulate DeepSeek returning 0.88 & recommended=true for NCT06642272!
                ans = {
                    "rankings": [
                        {"nct_id": "NCT06642272", "match_score": 0.88, "recommended": True, "reason_zh": "虽对照臂为恩格列净但含达格列净"},
                        {"nct_id": "NCT01095653", "match_score": 0.94, "recommended": True, "reason_zh": "达格列净对比西格列汀双臂严格一致"},
                    ]
                }
            else:
                # Even if DeepSeek omitted Sitagliptin in its own text, deterministic cohort lock preserves comparator_query_en='Sitagliptin'!
                ans = {
                    "domain_summary_zh": "2型糖尿病合并心肾风险队列",
                    "condition_query_en": "Type 2 Diabetes Heart Failure",
                    "intervention_query_en": "Dapagliflozin",
                    "search_query_en": "Dapagliflozin Type 2 Diabetes",
                }
            return _FakeHTTPResp(json.dumps({"choices": [{"message": {"content": json.dumps(ans, ensure_ascii=False)}}]}).encode("utf-8"))

        def fake_ctgov(req, timeout):
            return _FakeHTTPResp(json.dumps(nct06642272_and_exact_payload).encode("utf-8"))

        with mock.patch.dict(
            os.environ,
            {
                "EMULATRX_LLM_API_KEY_FILE": str(self.base / "absent_secret"),
                "EMULATRX_LLM_API_KEY": "sk-deepseek-live-qa",
                "EMULATRX_OFFLINE_RCT_CATALOG": "0",
            },
            clear=False,
        ):
            with mock.patch.object(llm_assist, "urlopen_llm", side_effect=fake_deepseek), mock.patch.object(
                rct_matcher, "urlopen_ctgov", side_effect=fake_ctgov
            ):
                # 1. Load built-in cardio-renal sample cohort -> must extract BOTH Dapagliflozin AND Sitagliptin
                up = self.client.post(
                    "/api/sessions/upload",
                    headers=self.web_auth,
                    json={"sample_id": "cardiorenal_sglt2i"},
                )
                self.assertEqual(up.status_code, 200, up.text)
                sid = up.json()["session_id"]
                interp = up.json()["interpretation"]
                self.assertEqual(interp["intervention_query_en"], "Dapagliflozin")
                self.assertEqual(interp["comparator_query_en"], "Sitagliptin")
                self.assertIn("Sitagliptin", interp["search_query_en"])

                # 2. Search candidate RCTs -> NCT01095653 ranks #1; NCT06642272 is capped <=0.45 & is_recommended=False
                srch = self.client.post(
                    "/api/rcts/search",
                    headers=self.web_auth,
                    json={
                        "session_id": sid,
                        "project_mode": "user_cohort",
                        "query": interp["search_query_en"],
                        "condition": interp["condition_query_en"],
                        "intervention": interp["intervention_query_en"],
                        "comparator": interp["comparator_query_en"],
                        "context_summary": interp,
                    },
                )
                self.assertEqual(srch.status_code, 200, srch.text)
                cands = srch.json()["candidates"]
                self.assertEqual(cands[0]["nct_id"], "NCT01095653")
                self.assertEqual(cands[0]["arm_alignment_status"], "exact_dual_arm")
                self.assertTrue(cands[0]["is_recommended"])

                nct066 = next(c for c in cands if c["nct_id"] == "NCT06642272")
                self.assertEqual(nct066["arm_alignment_status"], "single_arm_or_comparator_mismatch")
                self.assertLessEqual(nct066["match_score"], 0.45)
                self.assertFalse(nct066["is_recommended"])

                # 3. Fail-Closed 422 Gate: submitting NCT06642272 WITHOUT confirm_arm_mismatch=True MUST return HTTP 422
                draft_422 = self.client.post(
                    "/api/rcts/draft",
                    headers=self.web_auth,
                    json={"project_mode": "user_cohort", "session_id": sid, "target_rct": nct066},
                )
                self.assertEqual(draft_422.status_code, 422, draft_422.text)
                self.assertEqual(draft_422.json()["error_code"], "ARM_MISMATCH_CONFIRMATION_REQUIRED")

                mapping = up.json()["default_mapping"]
                mapping["bootstrap_iterations"] = 20
                run_422 = self.client.post(
                    "/api/v1/projects/user-cohort/runs",
                    headers=self.web_auth,
                    json={"session_id": sid, "mapping": mapping, "target_rct": nct066},
                )
                self.assertEqual(run_422.status_code, 422, run_422.text)
                self.assertEqual(run_422.json()["error_code"], "ARM_MISMATCH_CONFIRMATION_REQUIRED")

                # 4. Submitting WITH confirm_arm_mismatch=True succeeds and persists arm_alignment_warning in JSON, Markdown & HTML
                draft_ok = self.client.post(
                    "/api/rcts/draft",
                    headers=self.web_auth,
                    json={
                        "project_mode": "user_cohort",
                        "session_id": sid,
                        "target_rct": nct066,
                        "confirm_arm_mismatch": True,
                    },
                )
                self.assertEqual(draft_ok.status_code, 200, draft_ok.text)
                self.assertTrue(draft_ok.json()["target_rct"]["arm_alignment_warning"])

                run_ok = self.client.post(
                    "/api/v1/projects/user-cohort/runs",
                    headers=self.web_auth,
                    json={
                        "session_id": sid,
                        "mapping": mapping,
                        "target_rct": nct066,
                        "confirm_arm_mismatch": True,
                    },
                )
                self.assertEqual(run_ok.status_code, 202, run_ok.text)
                final = self.wait_dual_run(run_ok.json()["run_id"])
                self.assertEqual(final["state"], "completed", final)
                trct_est = final["estimates"]["target_rct"]
                self.assertEqual(trct_est["arm_alignment_status"], "single_arm_or_comparator_mismatch")
                self.assertIn("NCT06642272", trct_est["arm_alignment_warning"])
                self.assertIn("arm-mismatch-callout", final["rendered_html"])
                self.assertIn("对照不一致参考声明", final["rendered_html"])
                self.assertIn("arm_alignment_warning", final["audit_report_markdown"])


if __name__ == "__main__":
    unittest.main()
