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
                for art_url in (
                    f"/runs/{run_id}/report",
                    f"/runs/{run_id}/artifacts/bundle/estimates.json",
                    f"/runs/{run_id}/artifacts/bundle/cohort.csv",
                    f"/runs/{run_id}/artifacts/preview_manifest.json",
                ):
                    r_art = self.client.get(art_url, headers=self.web_auth)
                    self.assertEqual(r_art.status_code, 200, f"{art_url}: {r_art.text[:200]}")

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
                        "conditionsModule": {"conditions": ["Type 2 Diabetes", "Heart Failure"]},
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
                        "outcomesModule": {"primaryOutcomes": [{"measure": "Heart Failure Hospitalization or Cardiorenal Composite", "timeFrame": "Up to 18 months"}]},
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

    def test_round4_probes_1a_1b_binary_01_cohort_and_idea_llm_hallucinated_comparator(self):
        """E2E regression for all 4 Codex Round-4 probes: 1A shared background, 1B <2 armGroups, 2 binary 0/1 cohort, 3 idea LLM comparator."""
        round4_ctgov_payload = {
            "studies": [
                # NCT06642272: Empagliflozin vs Dapagliflozin
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT06642272", "briefTitle": "Empagliflozin vs Dapagliflozin in T2DM"},
                        "statusModule": {"overallStatus": "RECRUITING"},
                        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE4"], "enrollmentInfo": {"count": 800}, "designInfo": {"allocation": "RANDOMIZED"}},
                        "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Empagliflozin 10 mg", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Empagliflozin"]},
                                {"label": "Dapagliflozin 10 mg", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Dapagliflozin"]},
                            ],
                            "interventions": [{"name": "Empagliflozin", "type": "DRUG"}, {"name": "Dapagliflozin", "type": "DRUG"}],
                        },
                        "outcomesModule": {"primaryOutcomes": [{"measure": "HF or CV Death", "timeFrame": "12 months"}]},
                    }
                },
                # NCT09990001: Dapagliflozin + Metformin vs Metformin (missing Sitagliptin!)
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT09990001", "briefTitle": "Dapagliflozin + Metformin vs Metformin Alone"},
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE3"], "enrollmentInfo": {"count": 600}, "designInfo": {"allocation": "RANDOMIZED"}},
                        "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Dapagliflozin + Metformin", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin", "Drug: Metformin"]},
                                {"label": "Metformin + Placebo", "type": "PLACEBO_COMPARATOR", "interventionNames": ["Drug: Metformin", "Drug: Placebo"]},
                            ],
                            "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Metformin", "type": "DRUG"}],
                        },
                        "outcomesModule": {"primaryOutcomes": [{"measure": "Composite", "timeFrame": "12 months"}]},
                    }
                },
                # NCT09990002: Single-arm Dapagliflozin + Sitagliptin (`len(armGroups) == 1`)
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT09990002", "briefTitle": "Single-Arm Dapagliflozin + Sitagliptin"},
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE2"], "enrollmentInfo": {"count": 120}},
                        "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Dapagliflozin + Sitagliptin Fixed Combo", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin", "Drug: Sitagliptin"]}
                            ],
                            "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Sitagliptin", "type": "DRUG"}],
                        },
                        "outcomesModule": {"primaryOutcomes": [{"measure": "HbA1c", "timeFrame": "6 months"}]},
                    }
                },
                # NCT01095653: True head-to-head Dapagliflozin + Metformin vs Sitagliptin + Metformin
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT01095653", "briefTitle": "Dapagliflozin + Metformin vs Sitagliptin + Metformin"},
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE3"], "enrollmentInfo": {"count": 1450}, "designInfo": {"allocation": "RANDOMIZED"}},
                        "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Dapagliflozin 10 mg + Metformin", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin", "Drug: Metformin"]},
                                {"label": "Sitagliptin 100 mg + Metformin", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Sitagliptin", "Drug: Metformin"]},
                            ],
                            "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Sitagliptin", "type": "DRUG"}, {"name": "Metformin", "type": "DRUG"}],
                        },
                        "outcomesModule": {"primaryOutcomes": [{"measure": "Cardiorenal Composite", "timeFrame": "12 months"}]},
                    }
                },
                # NCT03036124: Dapagliflozin vs Placebo in Heart Failure
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT03036124", "briefTitle": "Dapagliflozin vs Placebo in Heart Failure (DAPA-HF)"},
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE3"], "enrollmentInfo": {"count": 4744}, "designInfo": {"allocation": "RANDOMIZED"}},
                        "conditionsModule": {"conditions": ["Heart Failure"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Dapagliflozin 10 mg", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin"]},
                                {"label": "Matching Placebo", "type": "PLACEBO_COMPARATOR", "interventionNames": ["Drug: Placebo"]},
                            ],
                            "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Placebo", "type": "DRUG"}],
                        },
                        "outcomesModule": {"primaryOutcomes": [{"measure": "Worsening HF or CV Death", "timeFrame": "18 months"}]},
                    }
                },
            ]
        }

        def fake_ctgov(req, timeout):
            return _FakeHTTPResp(json.dumps(round4_ctgov_payload).encode("utf-8"))

        # Mock LLM that hallucinates `comparator_query_en="Sitagliptin"` even on a single-intervention idea!
        def fake_llm_hallucinate_comp(req, timeout):
            body_txt = req.data.decode("utf-8")
            if "idea_text" in body_txt:
                ans = {
                    "population": "心力衰竭患者",
                    "intervention": "达格列净 (Dapagliflozin)",
                    "comparator": "西格列汀 (Sitagliptin)",
                    "primary_outcome": "心衰恶化住院或心血管死亡",
                    "competing_event": "非心血管竞争死亡",
                    "condition_query_en": "Heart Failure",
                    "intervention_query_en": "Dapagliflozin",
                    "comparator_query_en": "Sitagliptin",
                    "search_query_en": "Dapagliflozin Sitagliptin Heart Failure",
                }
            elif "public_reference_context" in body_txt:
                ans = {"rankings": [{"nct_id": "NCT06642272", "match_score": 0.91, "recommended": True, "reason_zh": "测试重排"}]}
            else:
                ans = {
                    "domain_summary_zh": "SGLT2队列",
                    "condition_query_en": "Type 2 Diabetes",
                    "intervention_query_en": "Dapagliflozin",
                    "comparator_query_en": "Empagliflozin",
                    "search_query_en": "Dapagliflozin Type 2 Diabetes",
                }
            return _FakeHTTPResp(json.dumps({"choices": [{"message": {"content": json.dumps(ans, ensure_ascii=False)}}]}).encode("utf-8"))

        with mock.patch.dict(
            os.environ,
            {
                "EMULATRX_LLM_API_KEY_FILE": str(self.base / "absent_secret"),
                "EMULATRX_LLM_API_KEY": "sk-round4-test",
                "EMULATRX_OFFLINE_RCT_CATALOG": "0",
            },
            clear=False,
        ):
            with mock.patch.object(llm_assist, "urlopen_llm", side_effect=fake_llm_hallucinate_comp), mock.patch.object(
                rct_matcher, "urlopen_ctgov", side_effect=fake_ctgov
            ):
                # --- Probe 1A & 1B E2E: Cohort `Dapagliflozin_Plus_Metformin` vs `Sitagliptin_Plus_Metformin` ---
                combo_rows = ["deid_id,initiated_drug,followup_months,endpoint,age_years"]
                for i in range(20):
                    drug = "Dapagliflozin_Plus_Metformin" if i % 2 == 0 else "Sitagliptin_Plus_Metformin"
                    ep = "hf_hospitalization" if i < 7 else ("cv_death" if i < 11 else "censored")
                    combo_rows.append(f"PT-{i + 1:04d},{drug},{8.0 + i * 0.9:.1f},{ep},{54 + (i % 14)}")
                up_combo = self.client.post(
                    "/api/sessions/upload",
                    headers=self.web_auth,
                    files={"file": ("combo_met.csv", ("\n".join(combo_rows) + "\n").encode("utf-8"), "text/csv")},
                )
                self.assertEqual(up_combo.status_code, 200, up_combo.text)
                sid_combo = up_combo.json()["session_id"]

                srch_combo = self.client.post(
                    "/api/rcts/search",
                    headers=self.web_auth,
                    json={"session_id": sid_combo, "project_mode": "user_cohort", "query": "Dapagliflozin Sitagliptin"},
                )
                self.assertEqual(srch_combo.status_code, 200)
                by_nct = {c["nct_id"]: c for c in srch_combo.json()["candidates"]}
                # True combo head-to-head gets exact_dual_arm; Dapa+Met vs Met AND 1-arm combo both get <=0.45 & 422!
                self.assertEqual(by_nct["NCT01095653"]["arm_alignment_status"], "exact_dual_arm")
                for bad_id in ("NCT09990001", "NCT09990002"):
                    self.assertEqual(by_nct[bad_id]["arm_alignment_status"], "single_arm_or_comparator_mismatch")
                    self.assertLessEqual(by_nct[bad_id]["match_score"], 0.45)
                    self.assertFalse(by_nct[bad_id]["is_recommended"])
                    r_bad = self.client.post(
                        "/api/rcts/draft",
                        headers=self.web_auth,
                        json={"project_mode": "user_cohort", "session_id": sid_combo, "target_rct": by_nct[bad_id]},
                    )
                    self.assertEqual(r_bad.status_code, 422, r_bad.text)
                    self.assertEqual(r_bad.json()["error_code"], "ARM_MISMATCH_CONFIRMATION_REQUIRED")

                # --- Probe 2 E2E: Anonymous binary `treatment_sglt2_user = 0/1` CSV ---
                bin_rows = ["deid_id,treatment_sglt2_user,followup_months,endpoint,age_years"]
                for i in range(20):
                    arm_01 = 1 if i % 2 == 0 else 0
                    ep = "hf_hospitalization" if i < 7 else ("cv_death" if i < 11 else "censored")
                    bin_rows.append(f"PT-{i + 1:04d},{arm_01},{8.0 + i * 0.9:.1f},{ep},{52 + (i % 15)}")
                up_bin = self.client.post(
                    "/api/sessions/upload",
                    headers=self.web_auth,
                    files={"file": ("sglt2_01.csv", ("\n".join(bin_rows) + "\n").encode("utf-8"), "text/csv")},
                )
                self.assertEqual(up_bin.status_code, 200, up_bin.text)
                sid_bin = up_bin.json()["session_id"]

                srch_bin = self.client.post(
                    "/api/rcts/search",
                    headers=self.web_auth,
                    json={"session_id": sid_bin, "project_mode": "user_cohort", "query": "SGLT2 inhibitor"},
                )
                self.assertEqual(srch_bin.status_code, 200)
                bin_nct066 = next(c for c in srch_bin.json()["candidates"] if c["nct_id"] == "NCT06642272")
                # Must NEVER be single_intervention_matched or recommended on a 0/1 user_cohort!
                self.assertNotEqual(bin_nct066["arm_alignment_status"], "single_intervention_matched")
                self.assertFalse(bin_nct066["is_recommended"])
                self.assertLessEqual(bin_nct066["match_score"], 0.45)

                draft_bin_422 = self.client.post(
                    "/api/rcts/draft",
                    headers=self.web_auth,
                    json={"project_mode": "user_cohort", "session_id": sid_bin, "target_rct": bin_nct066},
                )
                self.assertEqual(draft_bin_422.status_code, 422, draft_bin_422.text)
                self.assertEqual(draft_bin_422.json()["error_code"], "ARM_MISMATCH_CONFIRMATION_REQUIRED")

                # When researcher explicitly declares `1 = 达格列净` and `0 = 西格列汀`, NCT01095653 succeeds while NCT06642272 still 422s!
                draft_bin_still_422 = self.client.post(
                    "/api/rcts/draft",
                    headers=self.web_auth,
                    json={
                        "project_mode": "user_cohort",
                        "session_id": sid_bin,
                        "target_rct": bin_nct066,
                        "intervention": "达格列净",
                        "comparator": "西格列汀",
                    },
                )
                self.assertEqual(draft_bin_still_422.status_code, 422)

                draft_bin_exact_ok = self.client.post(
                    "/api/rcts/draft",
                    headers=self.web_auth,
                    json={
                        "project_mode": "user_cohort",
                        "session_id": sid_bin,
                        "target_rct": by_nct["NCT01095653"],
                        "intervention": "达格列净 + 二甲双胍",
                        "comparator": "西格列汀 + 二甲双胍",
                    },
                )
                self.assertEqual(draft_bin_exact_ok.status_code, 200, draft_bin_exact_ok.text)
                self.assertEqual(draft_bin_exact_ok.json()["target_rct"]["arm_alignment_status"], "exact_dual_arm")

                # --- Probe 3 E2E: Single-intervention idea ignores LLM-hallucinated comparator ---
                parsed_idea = self.client.post(
                    "/api/idea/parse",
                    headers=self.web_auth,
                    json={"idea_text": "评估达格列净在心衰患者中的心肾获益"},
                )
                self.assertEqual(parsed_idea.status_code, 200, parsed_idea.text)
                pico_single = parsed_idea.json()["pico"]
                self.assertFalse(pico_single["comparator_explicit"])
                self.assertEqual(pico_single["comparator_query_en"], "")
                self.assertNotIn("Sitagliptin", pico_single["search_query_en"])

                srch_idea = self.client.post(
                    "/api/rcts/search",
                    headers=self.web_auth,
                    json={
                        "project_mode": "simulated_cohort",
                        "query": pico_single["search_query_en"],
                        "condition": pico_single["condition_query_en"],
                        "intervention": pico_single["intervention_query_en"],
                        "context_summary": pico_single,
                    },
                )
                self.assertEqual(srch_idea.status_code, 200)
                dapa_hf = next(c for c in srch_idea.json()["candidates"] if c["nct_id"] == "NCT03036124")
                self.assertEqual(dapa_hf["arm_alignment_status"], "single_intervention_matched")
                self.assertTrue(dapa_hf["is_recommended"])

                draft_idea_ok = self.client.post(
                    "/api/rcts/draft",
                    headers=self.web_auth,
                    json={"project_mode": "simulated_cohort", "target_rct": dapa_hf, "pico": pico_single},
                )
                self.assertEqual(draft_idea_ok.status_code, 200, draft_idea_ok.text)

    def test_round5_blockers_1a_1b_and_direct_simulated_run_alignment_gate(self):
        """E2E regression for Codex Round-5 Blockers 1A, 1B (full regimen drug-set equality) and Blocker 2 (unconditional simulated run alignment gate)."""
        round5_ctgov_payload = {
            "studies": [
                # NCT06642272: Empagliflozin vs Dapagliflozin
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT06642272", "briefTitle": "Empagliflozin vs Dapagliflozin"},
                        "statusModule": {"overallStatus": "RECRUITING"},
                        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE4"], "enrollmentInfo": {"count": 800}, "designInfo": {"allocation": "RANDOMIZED"}},
                        "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Empagliflozin 10 mg", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Empagliflozin"]},
                                {"label": "Dapagliflozin 10 mg", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Dapagliflozin"]},
                            ],
                            "interventions": [{"name": "Empagliflozin", "type": "DRUG"}, {"name": "Dapagliflozin", "type": "DRUG"}],
                        },
                        "outcomesModule": {"primaryOutcomes": [{"measure": "Cardiorenal Composite", "timeFrame": "12 months"}]},
                    }
                },
                # NCT09990011: Dapagliflozin vs Sitagliptin (monotherapy — NO Metformin!)
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT09990011", "briefTitle": "Dapagliflozin vs Sitagliptin Monotherapy"},
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE3"], "enrollmentInfo": {"count": 900}, "designInfo": {"allocation": "RANDOMIZED"}},
                        "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Dapagliflozin 10 mg", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin"]},
                                {"label": "Sitagliptin 100 mg", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Sitagliptin"]},
                            ],
                            "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Sitagliptin", "type": "DRUG"}],
                        },
                        "outcomesModule": {"primaryOutcomes": [{"measure": "Composite", "timeFrame": "12 months"}]},
                    }
                },
                # NCT09990012: Dapagliflozin vs Sitagliptin + Glimepiride (extra active drug in comparator!)
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT09990012", "briefTitle": "Dapagliflozin vs Sitagliptin Plus Glimepiride"},
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE3"], "enrollmentInfo": {"count": 700}, "designInfo": {"allocation": "RANDOMIZED"}},
                        "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Dapagliflozin 10 mg", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin"]},
                                {"label": "Sitagliptin 100 mg + Glimepiride 4 mg", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Sitagliptin", "Drug: Glimepiride"]},
                            ],
                            "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Sitagliptin", "type": "DRUG"}, {"name": "Glimepiride", "type": "DRUG"}],
                        },
                        "outcomesModule": {"primaryOutcomes": [{"measure": "Composite", "timeFrame": "12 months"}]},
                    }
                },
                # NCT01095653: Exact Dapagliflozin + Metformin vs Sitagliptin + Metformin
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT01095653", "briefTitle": "Dapagliflozin + Metformin vs Sitagliptin + Metformin"},
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE3"], "enrollmentInfo": {"count": 1450}, "designInfo": {"allocation": "RANDOMIZED"}},
                        "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Dapagliflozin 10 mg + Metformin", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin", "Drug: Metformin"]},
                                {"label": "Sitagliptin 100 mg + Metformin", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Sitagliptin", "Drug: Metformin"]},
                            ],
                            "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Sitagliptin", "type": "DRUG"}, {"name": "Metformin", "type": "DRUG"}],
                        },
                        "outcomesModule": {"primaryOutcomes": [{"measure": "Composite", "timeFrame": "12 months"}]},
                    }
                },
                # NCT09990001: Exact add-on Dapagliflozin + Metformin vs Metformin + Placebo
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT09990001", "briefTitle": "Dapagliflozin + Metformin vs Metformin Alone"},
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE3"], "enrollmentInfo": {"count": 600}, "designInfo": {"allocation": "RANDOMIZED"}},
                        "conditionsModule": {"conditions": ["Type 2 Diabetes", "Heart Failure"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Dapagliflozin + Metformin", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin", "Drug: Metformin"]},
                                {"label": "Metformin + Placebo", "type": "PLACEBO_COMPARATOR", "interventionNames": ["Drug: Metformin", "Drug: Placebo"]},
                            ],
                            "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Metformin", "type": "DRUG"}],
                        },
                        "outcomesModule": {"primaryOutcomes": [{"measure": "Heart Failure Hospitalization or Cardiorenal Composite", "timeFrame": "18 months"}]},
                    }
                },
            ]
        }

        def fake_ctgov(req, timeout):
            content_url = req.full_url
            if "/studies/NCT06642272" in content_url:
                return _FakeHTTPResp(json.dumps(round5_ctgov_payload["studies"][0]).encode("utf-8"))
            return _FakeHTTPResp(json.dumps(round5_ctgov_payload).encode("utf-8"))

        with mock.patch.dict(os.environ, {"EMULATRX_OFFLINE_RCT_CATALOG": "0"}, clear=False):
            with mock.patch.object(rct_matcher, "urlopen_ctgov", side_effect=fake_ctgov):
                # --- Blocker 1A: CSV `Dapagliflozin_Plus_Metformin` vs `Sitagliptin_Plus_Metformin` against `NCT09990011` (`Dapagliflozin` vs `Sitagliptin`, no Metformin) ---
                combo_rows = ["deid_id,initiated_drug,followup_months,endpoint,age_years"]
                for i in range(20):
                    drug = "Dapagliflozin_Plus_Metformin" if i % 2 == 0 else "Sitagliptin_Plus_Metformin"
                    ep = "hf_hospitalization" if i < 7 else ("cv_death" if i < 11 else "censored")
                    combo_rows.append(f"PT-{i + 1:04d},{drug},{8.0 + i * 0.9:.1f},{ep},{54 + (i % 14)}")
                up_combo = self.client.post(
                    "/api/sessions/upload",
                    headers=self.web_auth,
                    files={"file": ("combo_met.csv", ("\n".join(combo_rows) + "\n").encode("utf-8"), "text/csv")},
                )
                self.assertEqual(up_combo.status_code, 200)
                sid_combo = up_combo.json()["session_id"]

                srch_combo = self.client.post(
                    "/api/rcts/search",
                    headers=self.web_auth,
                    json={"session_id": sid_combo, "project_mode": "user_cohort", "query": "Dapagliflozin Sitagliptin"},
                )
                self.assertEqual(srch_combo.status_code, 200)
                combo_by_nct = {c["nct_id"]: c for c in srch_combo.json()["candidates"]}
                # Exact combo `NCT01095653` is `exact_dual_arm`; `NCT09990011` (missing Metformin) is NOT exact_dual_arm, capped <=0.72, is_recommended=False!
                self.assertEqual(combo_by_nct["NCT01095653"]["arm_alignment_status"], "exact_dual_arm")
                no_met_rct = combo_by_nct["NCT09990011"]
                self.assertNotEqual(no_met_rct["arm_alignment_status"], "exact_dual_arm")
                self.assertLessEqual(no_met_rct["match_score"], 0.72)
                self.assertFalse(no_met_rct["is_recommended"])

                for ep_path in ("/api/rcts/draft", "/api/dual-runs"):
                    r_422 = self.client.post(
                        ep_path,
                        headers=self.web_auth,
                        json={"project_mode": "user_cohort", "session_id": sid_combo, "target_rct": no_met_rct},
                    )
                    self.assertEqual(r_422.status_code, 422, f"{ep_path}: {r_422.text}")
                    self.assertEqual(r_422.json()["error_code"], "ARM_MISMATCH_CONFIRMATION_REQUIRED")

                # --- Blocker 1B: CSV `Dapagliflozin` vs `Sitagliptin` against `NCT09990012` (`Dapagliflozin` vs `Sitagliptin + Glimepiride`) ---
                up_mono = self.client.post(
                    "/api/sessions/upload",
                    headers=self.web_auth,
                    json={"sample_id": "cardiorenal_sglt2i"},
                )
                self.assertEqual(up_mono.status_code, 200)
                sid_mono = up_mono.json()["session_id"]

                srch_mono = self.client.post(
                    "/api/rcts/search",
                    headers=self.web_auth,
                    json={"session_id": sid_mono, "project_mode": "user_cohort", "query": "Dapagliflozin Sitagliptin"},
                )
                self.assertEqual(srch_mono.status_code, 200)
                mono_by_nct = {c["nct_id"]: c for c in srch_mono.json()["candidates"]}
                extra_glim_rct = mono_by_nct["NCT09990012"]
                self.assertNotEqual(extra_glim_rct["arm_alignment_status"], "exact_dual_arm")
                self.assertLessEqual(extra_glim_rct["match_score"], 0.45)
                self.assertFalse(extra_glim_rct["is_recommended"])

                for ep_path in ("/api/rcts/draft", "/api/dual-runs"):
                    r_1b = self.client.post(
                        ep_path,
                        headers=self.web_auth,
                        json={"project_mode": "user_cohort", "session_id": sid_mono, "target_rct": extra_glim_rct},
                    )
                    self.assertEqual(r_1b.status_code, 422, f"{ep_path}: {r_1b.text}")
                    self.assertEqual(r_1b.json()["error_code"], "ARM_MISMATCH_CONFIRMATION_REQUIRED")

                # Verify exact add-on (`Dapagliflozin + Metformin` vs `Metformin`) still gets `exact_dual_arm` (`200`)
                addon_draft_ok = self.client.post(
                    "/api/rcts/draft",
                    headers=self.web_auth,
                    json={
                        "project_mode": "user_cohort",
                        "session_id": sid_mono,
                        "target_rct": mono_by_nct["NCT09990001"],
                        "intervention": "Dapagliflozin + Metformin",
                        "comparator": "Metformin",
                    },
                )
                self.assertEqual(addon_draft_ok.status_code, 200, addon_draft_ok.text)
                self.assertEqual(addon_draft_ok.json()["target_rct"]["arm_alignment_status"], "exact_dual_arm")

                # --- Blocker 2: Direct simulated run with NO `pico` / NO `idea_text` / NO `confirm_arm_mismatch` MUST synchronously 422! ---
                nct066_card = mono_by_nct["NCT06642272"]
                direct_sim_payload = {
                    "project_mode": "simulated_cohort",
                    "target_rct": nct066_card,
                    "intervention": "Dapagliflozin",
                    "comparator": "Sitagliptin",
                    "sim_config": {"cases": 400, "bootstrap_iterations": 20},
                }
                for sim_ep in ("/api/dual-runs", "/api/v1/projects/simulated/runs"):
                    sim_422 = self.client.post(sim_ep, headers=self.web_auth, json=direct_sim_payload)
                    self.assertEqual(sim_422.status_code, 422, f"{sim_ep}: {sim_422.text}")
                    self.assertEqual(sim_422.json()["error_code"], "ARM_MISMATCH_CONFIRMATION_REQUIRED")

                # Also verify sim_config label overrides (`treatment_label` / `comparator_label`) with NO top-level fields or pico return 422
                sim_cfg_only_422 = self.client.post(
                    "/api/dual-runs",
                    headers=self.web_auth,
                    json={
                        "project_mode": "simulated_cohort",
                        "target_rct": nct066_card,
                        "sim_config": {
                            "treatment_label": "Dapagliflozin",
                            "comparator_label": "Sitagliptin",
                            "cases": 400,
                            "bootstrap_iterations": 20,
                        },
                    },
                )
                self.assertEqual(sim_cfg_only_422.status_code, 422, sim_cfg_only_422.text)
                self.assertEqual(sim_cfg_only_422.json()["error_code"], "ARM_MISMATCH_CONFIRMATION_REQUIRED")

                # Also verify protocol.treatment_strategies and 1-arm target_rct with zero pico/idea_text/overrides return 422
                proto_only_422 = self.client.post(
                    "/api/dual-runs",
                    headers=self.web_auth,
                    json={
                        "project_mode": "simulated_cohort",
                        "target_rct": nct066_card,
                        "protocol": {
                            "treatment_strategies": [
                                {"id": "arm_comparator", "label": "Sitagliptin", "role": "control"},
                                {"id": "arm_active", "label": "Dapagliflozin", "role": "active"},
                            ]
                        },
                        "sim_config": {"cases": 400, "bootstrap_iterations": 20},
                    },
                )
                self.assertEqual(proto_only_422.status_code, 422, proto_only_422.text)
                self.assertEqual(proto_only_422.json()["error_code"], "ARM_MISMATCH_CONFIRMATION_REQUIRED")

                single_arm_target = {
                    "nct_id": "NCT09990002",
                    "brief_title": "Single-Arm Dapagliflozin + Sitagliptin",
                    "experimental_arms": ["Dapagliflozin + Sitagliptin"],
                    "comparator_arms": [],
                    "arm_groups_detail": [
                        {"label": "Dapagliflozin + Sitagliptin", "type": "EXPERIMENTAL", "description": "", "intervention_names": ["Drug: Dapagliflozin", "Drug: Sitagliptin"]}
                    ],
                    "interventions": ["Drug: Dapagliflozin", "Drug: Sitagliptin"],
                }
                single_arm_no_pico_422 = self.client.post(
                    "/api/dual-runs",
                    headers=self.web_auth,
                    json={
                        "project_mode": "simulated_cohort",
                        "target_rct": single_arm_target,
                        "sim_config": {"cases": 400, "bootstrap_iterations": 20},
                    },
                )
                self.assertEqual(single_arm_no_pico_422.status_code, 422, single_arm_no_pico_422.text)
                self.assertEqual(single_arm_no_pico_422.json()["error_code"], "ARM_MISMATCH_CONFIRMATION_REQUIRED")

                # Also verify MCP `start_simulated_rct_tte` rejects mismatched NCT06642272 without confirm_arm_mismatch
                mcp_resp = self.client.post(
                    "/mcp",
                    headers=self.mcp_auth,
                    json={
                        "jsonrpc": "2.0",
                        "id": 99,
                        "method": "tools/call",
                        "params": {
                            "name": "start_simulated_rct_tte",
                            "arguments": {
                                "nct_id": "NCT06642272",
                                "intervention": "Dapagliflozin",
                                "comparator": "Sitagliptin",
                                "cases": 400,
                                "bootstrap_iterations": 20,
                            },
                        },
                    },
                )
                self.assertEqual(mcp_resp.status_code, 200)
                self.assertTrue(mcp_resp.json()["result"]["isError"], mcp_resp.text)
                self.assertIn("ARM_MISMATCH_CONFIRMATION_REQUIRED", mcp_resp.text)

                # With `confirm_arm_mismatch=True`, direct `/api/dual-runs` succeeds (`202` -> `completed`) and records `arm_alignment_warning`!
                sim_ok = self.client.post(
                    "/api/dual-runs",
                    headers=self.web_auth,
                    json={**direct_sim_payload, "confirm_arm_mismatch": True},
                )
                self.assertEqual(sim_ok.status_code, 202, sim_ok.text)
                sim_final = self.wait_dual_run(sim_ok.json()["run_id"])
                self.assertEqual(sim_final["state"], "completed", sim_final)
                sim_trct = sim_final["estimates"]["target_rct"]
                self.assertEqual(sim_trct["arm_alignment_status"], "single_arm_or_comparator_mismatch")
                self.assertIn("NCT06642272", sim_trct["arm_alignment_warning"])
                self.assertIn("arm-mismatch-callout", sim_final["rendered_html"])
                self.assertIn("arm_alignment_warning", sim_final["audit_report_markdown"])

    def test_pico4d_nct03959501_nct00842556_422_protocol_mismatch_gate_and_carolina_flaura_e2e(self):
        """Live QA 4D PICO-T E2E: NCT03959501 (<=0.48), NCT00842556 (<=0.20), 422 PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED, and CAROLINA/FLAURA positive E2E."""
        live_qa_ctgov_studies = {
            "studies": [
                # 1. NCT03959501: Dapagliflozin vs Sitagliptin, uncomplicated T2DM/Ketosis, 24-week blood ketone change
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT03959501",
                            "briefTitle": "Comparison of Effects of Dapagliflozin and Sitagliptin on Blood Ketone Levels in Type 2 Diabetes",
                        },
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE4"], "enrollmentInfo": {"count": 60}, "designInfo": {"allocation": "RANDOMIZED"}},
                        "conditionsModule": {"conditions": ["Type 2 Diabetes", "Ketosis"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Dapagliflozin", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin"]},
                                {"label": "Sitagliptin", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Sitagliptin"]},
                            ],
                            "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Sitagliptin", "type": "DRUG"}],
                        },
                        "eligibilityModule": {
                            "healthyVolunteers": False,
                            "eligibilityCriteria": "Inclusion Criteria:\n* Type 2 Diabetes\nExclusion Criteria:\n* Chronic heart failure NYHA II-IV",
                        },
                        "outcomesModule": {
                            "primaryOutcomes": [{"measure": "Change in blood ketone bodies at 24 weeks", "timeFrame": "24 weeks"}]
                        },
                    }
                },
                # 2. NCT00842556: Healthy-volunteer PK/DDI study of Dapagliflozin + Sitagliptin
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT00842556",
                            "briefTitle": "Drug-Drug Interaction Study of Dapagliflozin and Sitagliptin in Healthy Subjects",
                        },
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE1"], "enrollmentInfo": {"count": 18}, "designInfo": {"allocation": "RANDOMIZED"}},
                        "conditionsModule": {"conditions": ["Healthy Volunteers"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Dapagliflozin", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin"]},
                                {"label": "Sitagliptin", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Sitagliptin"]},
                            ],
                            "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Sitagliptin", "type": "DRUG"}],
                        },
                        "eligibilityModule": {
                            "healthyVolunteers": True,
                            "eligibilityCriteria": "Inclusion Criteria:\n* Healthy volunteers aged 18-45 years",
                        },
                        "outcomesModule": {
                            "primaryOutcomes": [{"measure": "AUC and Cmax of Dapagliflozin and Sitagliptin", "timeFrame": "Day 1 to Day 14"}]
                        },
                    }
                },
                # 3. NCT03036124: DAPA-HF landmark placebo-controlled HF outcome reference
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT03036124",
                            "briefTitle": "Dapagliflozin in Patients With Chronic Heart Failure (DAPA-HF)",
                        },
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE3"], "enrollmentInfo": {"count": 4744}, "designInfo": {"allocation": "RANDOMIZED"}},
                        "conditionsModule": {"conditions": ["Heart Failure", "Type 2 Diabetes"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Dapagliflozin 10 mg", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin"]},
                                {"label": "Matching Placebo", "type": "PLACEBO_COMPARATOR", "interventionNames": ["Drug: Placebo"]},
                            ],
                            "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Placebo", "type": "DRUG"}],
                        },
                        "eligibilityModule": {"healthyVolunteers": False, "eligibilityCriteria": "Inclusion Criteria:\n* Symptomatic HFrEF"},
                        "outcomesModule": {
                            "primaryOutcomes": [{"measure": "Worsening Heart Failure or Cardiovascular Death", "timeFrame": "Up to 18 months"}]
                        },
                    }
                },
                # 4–5. Official ClinicalTrials.gov API v2 records: CAROLINA double-dummy and FLAURA SoC OR.
                json.loads((Path(__file__).resolve().parent / "fixtures" / "ctgov" / "NCT01243424.json").read_text(encoding="utf-8")),
                json.loads((Path(__file__).resolve().parent / "fixtures" / "ctgov" / "NCT02296125.json").read_text(encoding="utf-8")),
            ]
        }

        # Spin up a real local loopback HTTP fake DeepSeek gateway (`127.0.0.1:0`) trying to inflate NCT03959501 to 0.95 & NCT00842556 to 0.92!
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading as _threading

        class _LoopbackDeepSeekHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body_txt = self.rfile.read(length).decode("utf-8")
                if "public_reference_context" in body_txt:
                    ans = {
                        "rankings": [
                            {"nct_id": "NCT03959501", "match_score": 0.95, "recommended": True, "reason_zh": "双臂为达格列净与西格列汀"},
                            {"nct_id": "NCT00842556", "match_score": 0.92, "recommended": True, "reason_zh": "同时包含达格列净与西格列汀"},
                            {"nct_id": "NCT01243424", "match_score": 0.97, "recommended": True, "reason_zh": "CAROLINA 四维完全一致"},
                            {"nct_id": "NCT02296125", "match_score": 0.97, "recommended": True, "reason_zh": "FLAURA 四维完全一致"},
                        ]
                    }
                elif "Osimertinib" in body_txt:
                    ans = {
                        "domain_summary_zh": "EGFR突变肺癌队列",
                        "condition_query_en": "EGFR-Mutated Non-Small Cell Lung Cancer",
                        "intervention_query_en": "Osimertinib",
                        "search_query_en": "Osimertinib Gefitinib EGFR Non-Small Cell Lung Cancer",
                    }
                elif "Linagliptin" in body_txt:
                    ans = {
                        "domain_summary_zh": "2型糖尿病高心血管风险队列",
                        "condition_query_en": "Type 2 Diabetes Cardiovascular Risk",
                        "intervention_query_en": "Linagliptin",
                        "search_query_en": "Linagliptin Glimepiride Type 2 Diabetes Cardiovascular",
                    }
                else:
                    ans = {
                        "domain_summary_zh": "临床观察性队列",
                        "condition_query_en": "Type 2 Diabetes Heart Failure",
                        "intervention_query_en": "Dapagliflozin",
                        "search_query_en": "Dapagliflozin Sitagliptin Heart Failure",
                    }
                payload = json.dumps({"choices": [{"message": {"content": json.dumps(ans, ensure_ascii=False)}}]}, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        loopback_srv = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackDeepSeekHandler)
        self.addCleanup(loopback_srv.server_close)
        self.addCleanup(loopback_srv.shutdown)
        _threading.Thread(target=loopback_srv.serve_forever, daemon=True).start()
        loopback_url = f"http://127.0.0.1:{loopback_srv.server_address[1]}/v1"

        by_id_raw = {s["protocolSection"]["identificationModule"]["nctId"]: s for s in live_qa_ctgov_studies["studies"]}

        def fake_ctgov(req, timeout):
            for nid, st_obj in by_id_raw.items():
                if f"/studies/{nid}" in req.full_url:
                    return _FakeHTTPResp(json.dumps(st_obj).encode("utf-8"))
            return _FakeHTTPResp(json.dumps(live_qa_ctgov_studies).encode("utf-8"))

        # Verify wrong Bearer token on /mcp and unauthenticated REST endpoint return 401
        self.assertEqual(self.client.post("/api/rcts/search", json={"query": "Dapagliflozin"}).status_code, 401)
        self.assertEqual(
            self.client.post(
                "/mcp",
                headers={"Authorization": "Bearer " + "x" * 40, "Accept": "application/json, text/event-stream"},
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            ).status_code,
            401,
        )

        with mock.patch.dict(
            os.environ,
            {
                "EMULATRX_LLM_API_KEY_FILE": str(self.base / "absent_secret"),
                "EMULATRX_LLM_API_KEY": "sk-pico4d-qa-test",
                "EMULATRX_LLM_BASE_URL": loopback_url,
                "EMULATRX_LLM_MODEL": "deepseek-chat",
                "EMULATRX_OFFLINE_RCT_CATALOG": "0",
            },
            clear=False,
        ):
            with mock.patch.object(rct_matcher, "urlopen_ctgov", side_effect=fake_ctgov):
                # --- Part A: Cardio-renal HF sample cohort vs NCT03959501 & NCT00842556 ---
                up_hf = self.client.post(
                    "/api/sessions/upload",
                    headers=self.web_auth,
                    json={"sample_id": "cardiorenal_sglt2i"},
                )
                self.assertEqual(up_hf.status_code, 200, up_hf.text)
                sid_hf = up_hf.json()["session_id"]
                mapping_hf = up_hf.json()["default_mapping"]
                mapping_hf["bootstrap_iterations"] = 20

                srch_hf = self.client.post(
                    "/api/rcts/search",
                    headers=self.web_auth,
                    json={"session_id": sid_hf, "project_mode": "user_cohort", "query": "Dapagliflozin Sitagliptin Heart Failure", "max_results": 6},
                )
                self.assertEqual(srch_hf.status_code, 200, srch_hf.text)
                hf_by_nct = {c["nct_id"]: c for c in srch_hf.json()["candidates"]}

                ketone_card = hf_by_nct["NCT03959501"]
                self.assertEqual(ketone_card["arm_alignment_status"], "exact_dual_arm")
                self.assertNotEqual(ketone_card["population_alignment"], "matched")
                self.assertEqual(ketone_card["outcome_alignment"], "surrogate_or_mismatched")
                self.assertNotEqual(ketone_card["overall_alignment_status"], "protocol_aligned")
                self.assertLessEqual(ketone_card["match_score"], 0.48)
                self.assertFalse(ketone_card["is_recommended"])
                self.assertIn("血酮", ketone_card["dimension_audit"]["outcome"]["pill_zh"])
                self.assertIn("心力衰竭", ketone_card["dimension_audit"]["population"]["pill_zh"])

                pk_card = hf_by_nct["NCT00842556"]
                self.assertEqual(pk_card["design_suitability"], "non_efficacy_pk_or_healthy")
                self.assertEqual(pk_card["overall_alignment_status"], "non_efficacy_excluded")
                self.assertLessEqual(pk_card["match_score"], 0.20)
                self.assertFalse(pk_card["is_recommended"])
                self.assertIn("健康受试者", pk_card["dimension_audit"]["population"]["pill_zh"])
                self.assertIn("药代指标", pk_card["dimension_audit"]["outcome"]["pill_zh"])

                # Landmark HF outcome trial DAPA-HF is also surfaced alongside as a reference card!
                self.assertIn("NCT03036124", hf_by_nct)

                # Unconfirmed NCT03959501 & NCT00842556 MUST block with 422 PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED across all endpoints!
                for bad_card in (ketone_card, pk_card):
                    for ep in ("/api/rcts/draft", "/api/dual-runs"):
                        r_422 = self.client.post(
                            ep,
                            headers=self.web_auth,
                            json={"project_mode": "user_cohort", "session_id": sid_hf, "mapping": mapping_hf, "target_rct": bad_card},
                        )
                        self.assertEqual(r_422.status_code, 422, f"{bad_card['nct_id']} on {ep}: {r_422.text}")
                        self.assertEqual(r_422.json()["error_code"], "PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED")

                    r_v1_422 = self.client.post(
                        "/api/v1/projects/user-cohort/runs",
                        headers=self.web_auth,
                        json={"session_id": sid_hf, "mapping": mapping_hf, "target_rct": bad_card},
                    )
                    self.assertEqual(r_v1_422.status_code, 422, r_v1_422.text)
                    self.assertEqual(r_v1_422.json()["error_code"], "PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED")

                    mcp_422 = self.client.post(
                        "/mcp",
                        headers=self.mcp_auth,
                        json={
                            "jsonrpc": "2.0",
                            "id": 101,
                            "method": "tools/call",
                            "params": {
                                "name": "start_user_cohort_tte",
                                "arguments": {"session_id": sid_hf, "nct_id": bad_card["nct_id"], "bootstrap_iterations": 20},
                            },
                        },
                    )
                    self.assertEqual(mcp_422.status_code, 200)
                    self.assertTrue(mcp_422.json()["result"]["isError"], mcp_422.text)
                    self.assertIn("PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED", mcp_422.text)

                # Confirmed NCT03959501 with `confirm_protocol_mismatch=True` succeeds and persists `protocol_alignment_warning`!
                draft_ketone_ok = self.client.post(
                    "/api/rcts/draft",
                    headers=self.web_auth,
                    json={
                        "project_mode": "user_cohort",
                        "session_id": sid_hf,
                        "target_rct": ketone_card,
                        "confirm_protocol_mismatch": True,
                    },
                )
                self.assertEqual(draft_ketone_ok.status_code, 200, draft_ketone_ok.text)
                self.assertTrue(draft_ketone_ok.json()["target_rct"]["protocol_alignment_warning"])

                run_ketone_ok = self.client.post(
                    "/api/v1/projects/user-cohort/runs",
                    headers=self.web_auth,
                    json={
                        "session_id": sid_hf,
                        "mapping": mapping_hf,
                        "target_rct": ketone_card,
                        "confirm_protocol_mismatch": True,
                    },
                )
                self.assertEqual(run_ketone_ok.status_code, 202, run_ketone_ok.text)
                ketone_final = self.wait_dual_run(run_ketone_ok.json()["run_id"])
                self.assertEqual(ketone_final["state"], "completed", ketone_final)
                ketone_trct = ketone_final["estimates"]["target_rct"]
                self.assertEqual(ketone_trct["overall_alignment_status"], "partial_protocol_reference")
                self.assertIn("NCT03959501", ketone_trct["protocol_alignment_warning"])
                self.assertIn("protocol-mismatch-callout", ketone_final["rendered_html"])
                self.assertIn("protocol_alignment_warning", ketone_final["audit_report_markdown"])

                # Also verify MCP `start_user_cohort_tte` succeeds when `confirm_protocol_mismatch=True` is explicitly passed!
                mcp_ketone_ok = self.mcp_call(
                    "start_user_cohort_tte",
                    {
                        "session_id": sid_hf,
                        "nct_id": "NCT03959501",
                        "bootstrap_iterations": 20,
                        "confirm_protocol_mismatch": True,
                    },
                )
                self.assertEqual(self.wait_dual_run(mcp_ketone_ok["run_id"])["state"], "completed")

                # --- Part B: True 4D Positive Controls (CAROLINA NCT01243424 & FLAURA NCT02296125) run with zero confirmation friction ---
                for sample_key, expected_nct in (("sample_t2dm_cv_carolina", "NCT01243424"), ("sample_nsclc_egfr_tki", "NCT02296125")):
                    up_pos = self.client.post(
                        "/api/sessions/upload",
                        headers=self.web_auth,
                        json={"sample_id": sample_key},
                    )
                    self.assertEqual(up_pos.status_code, 200, up_pos.text)
                    sid_pos = up_pos.json()["session_id"]
                    interp_pos = up_pos.json()["interpretation"]
                    srch_pos = self.client.post(
                        "/api/rcts/search",
                        headers=self.web_auth,
                        json={
                            "session_id": sid_pos,
                            "project_mode": "user_cohort",
                            "query": interp_pos["search_query_en"],
                            "condition": interp_pos["condition_query_en"],
                            "intervention": interp_pos["intervention_query_en"],
                            "comparator": interp_pos["comparator_query_en"],
                        },
                    )
                    self.assertEqual(srch_pos.status_code, 200, srch_pos.text)
                    top_pos = srch_pos.json()["candidates"][0]
                    self.assertEqual(top_pos["nct_id"], expected_nct)
                    self.assertEqual(top_pos["arm_alignment_status"], "exact_dual_arm")
                    self.assertEqual(top_pos["overall_alignment_status"], "protocol_aligned")
                    self.assertTrue(top_pos["is_recommended"])
                    self.assertGreaterEqual(top_pos["match_score"], 0.85)
                    arm_blob = " ".join(top_pos.get("experimental_arms") or []) + " " + " ".join(top_pos.get("comparator_arms") or [])
                    if expected_nct == "NCT01243424":
                        self.assertIn("glimepiride placebo", arm_blob)
                        self.assertIn("linagliptin placebo", arm_blob)
                    else:
                        self.assertIn("Gefitinib", arm_blob)
                        self.assertIn("Erlotinib", arm_blob)
                        self.assertIn("Placebo", arm_blob)

                    map_pos = up_pos.json()["default_mapping"]
                    map_pos["bootstrap_iterations"] = 20
                    draft_pos = self.client.post(
                        "/api/rcts/draft",
                        headers=self.web_auth,
                        json={"project_mode": "user_cohort", "session_id": sid_pos, "target_rct": top_pos},
                    )
                    self.assertIn(draft_pos.status_code, (200, 202), draft_pos.text)
                    self.assertEqual(draft_pos.json()["target_rct"]["arm_alignment_status"], "exact_dual_arm")
                    run_pos = self.client.post(
                        "/api/dual-runs",
                        headers=self.web_auth,
                        json={
                            "project_mode": "user_cohort",
                            "session_id": sid_pos,
                            "mapping": map_pos,
                            "target_rct": top_pos,
                        },
                    )
                    self.assertIn(run_pos.status_code, (200, 202), run_pos.text)
                    pos_run_id = run_pos.json()["run_id"]
                    fin_pos = self.wait_dual_run(pos_run_id)
                    self.assertEqual(fin_pos["state"], "completed", fin_pos)
                    self.assertEqual(fin_pos["estimates"]["target_rct"]["overall_alignment_status"], "protocol_aligned")

                    # Verify report view & artifact download routes return 200 before purge, and 404 after `DELETE /api/cohorts/{session_id}`
                    for ok_url in (
                        f"/runs/{pos_run_id}/report",
                        f"/runs/{pos_run_id}/artifacts/bundle/estimates.json",
                        f"/runs/{pos_run_id}/artifacts/preview_manifest.json",
                    ):
                        self.assertEqual(self.client.get(ok_url, headers=self.web_auth).status_code, 200)

                    del_cohort = self.client.delete(f"/api/cohorts/{sid_pos}", headers=self.web_auth)
                    self.assertEqual(del_cohort.status_code, 200, del_cohort.text)
                    self.assertTrue(del_cohort.json()["session_deleted"])
                    self.assertEqual(self.client.get(f"/api/cohorts/{sid_pos}", headers=self.web_auth).status_code, 404)
                    self.assertEqual(self.client.get(f"/api/dual-runs/{pos_run_id}", headers=self.web_auth).status_code, 404)
                    self.assertEqual(
                        self.client.get(f"/runs/{pos_run_id}/artifacts/bundle/estimates.json", headers=self.web_auth).status_code,
                        404,
                    )

    def test_round7_custom_egfr_flaura_e2e_step2_pico_override_422_and_privacy(self):
        """Round-7 E2E regression: custom 30-row EGFR/NSCLC CSV matches FLAURA with zero false 422, while Step 2/explicit outcome override enforces 422 across Web/REST/MCP with zero PII leak."""
        home_html = self.client.get("/", headers=self.web_auth).text
        for dom_id in ("rct-cond-input", "rct-out-input", "step3-cond-hidden", "step3-out-hidden"):
            self.assertIn(f'id="{dom_id}"', home_html)

        csv_lines = ["patient_key,arm_drug,time_days,os_event,age,ecog"]
        for i in range(30):
            drug = "Osimertinib" if i % 2 == 0 else "Gefitinib or Erlotinib"
            t_days = 150 + (i * 23) % 520
            ev = 1 if (i % 3 == 0) else 0
            age = 50 + ((i * 7) % 22)
            ecog = (i // 2) % 2
            csv_lines.append(f"PT-ONC-{i + 1:03d},{drug},{t_days},{ev},{age},{ecog}")
        custom_csv = ("\n".join(csv_lines) + "\n").encode("utf-8")

        captured_llm_payloads: list[str] = []

        def fake_llm_audit(req, timeout):
            raw_str = req.data.decode("utf-8")
            captured_llm_payloads.append(raw_str)
            if "public_reference_context" in raw_str:
                ans = {
                    "rankings": [
                        {"nct_id": "NCT02296125", "match_score": 0.96, "recommended": True, "reason_zh": "奥希替尼对比吉非替尼一线治疗"},
                    ]
                }
            else:
                ans = {
                    "domain_summary_zh": "EGFR 突变晚期非小细胞肺癌靶向队列",
                    "condition_query_en": "EGFR-Mutated Non-Small Cell Lung Cancer",
                    "intervention_query_en": "Osimertinib",
                    "comparator_query_en": "Gefitinib or Erlotinib",
                    "search_query_en": "Osimertinib Gefitinib EGFR Non-Small Cell Lung Cancer",
                }
            return _FakeHTTPResp(json.dumps({"choices": [{"message": {"content": json.dumps(ans, ensure_ascii=False)}}]}).encode("utf-8"))

        with mock.patch.dict(
            os.environ,
            {
                "EMULATRX_LLM_API_KEY_FILE": str(self.base / "absent_secret"),
                "EMULATRX_LLM_API_KEY": "sk-round7-test",
                "EMULATRX_OFFLINE_RCT_CATALOG": "1",
            },
            clear=False,
        ):
            with mock.patch.object(llm_assist, "urlopen_llm", side_effect=fake_llm_audit):
                # 1. Blocker 1 E2E: Upload exact 30-row CSV + Step 2 note "EGFR-mutated non-small cell lung cancer, overall survival"
                up = self.client.post(
                    "/api/sessions/upload",
                    headers=self.web_auth,
                    files={"file": ("egfr_nsclc_30.csv", custom_csv, "text/csv")},
                    data={"user_note": "EGFR-mutated non-small cell lung cancer, overall survival"},
                )
                self.assertEqual(up.status_code, 200, up.text)
                sid = up.json()["session_id"]
                interp = up.json()["interpretation"]
                self.assertNotIn("Chronic Kidney Disease", interp["condition_query_en"])
                self.assertIn("EGFR-Mutated Non-Small Cell Lung Cancer", interp["condition_query_en"])

                srch_ok = self.client.post(
                    "/api/rcts/search",
                    headers=self.web_auth,
                    json={
                        "session_id": sid,
                        "project_mode": "user_cohort",
                        "query": interp["search_query_en"],
                        "condition": interp["condition_query_en"],
                        "intervention": interp["intervention_query_en"],
                        "comparator": interp["comparator_query_en"],
                        "outcome": interp["outcome_query_en"],
                    },
                )
                self.assertEqual(srch_ok.status_code, 200, srch_ok.text)
                flaura_card = srch_ok.json()["candidates"][0]
                self.assertEqual(flaura_card["nct_id"], "NCT02296125")
                self.assertEqual(flaura_card["arm_alignment_status"], "exact_dual_arm")
                self.assertEqual(flaura_card["overall_alignment_status"], "protocol_aligned")
                self.assertTrue(flaura_card["is_recommended"])

                draft_ok = self.client.post(
                    "/api/rcts/draft",
                    headers=self.web_auth,
                    json={"project_mode": "user_cohort", "session_id": sid, "target_rct": flaura_card},
                )
                self.assertEqual(draft_ok.status_code, 200, draft_ok.text)
                map_30 = draft_ok.json()["mapping"]
                map_30["eligibility_criteria"] = []
                map_30["max_abs_smd"] = 0.65
                map_30["min_ess_ratio"] = 0.10
                map_30["bootstrap_iterations"] = 20

                for run_endpoint in ("/api/dual-runs", "/api/v1/projects/user-cohort/runs"):
                    r_ok = self.client.post(
                        run_endpoint,
                        headers=self.web_auth,
                        json={"project_mode": "user_cohort", "session_id": sid, "mapping": map_30, "target_rct": flaura_card},
                    )
                    self.assertEqual(r_ok.status_code, 202, r_ok.text)
                    fin_ok = self.wait_dual_run(r_ok.json()["run_id"])
                    self.assertEqual(fin_ok["state"], "completed", fin_ok)
                    self.assertEqual(fin_ok["estimates"]["target_rct"]["overall_alignment_status"], "protocol_aligned")

                # 2. Blocker 2 E2E (explicit `outcome="Blood ketone change"` override on draft, dual-runs, v1 user-cohort, and MCP)
                for endpoint, body in (
                    (
                        "/api/rcts/draft",
                        {"project_mode": "user_cohort", "session_id": sid, "target_rct": flaura_card, "outcome": "Blood ketone change"},
                    ),
                    (
                        "/api/dual-runs",
                        {"project_mode": "user_cohort", "session_id": sid, "mapping": map_30, "target_rct": flaura_card, "outcome": "Blood ketone change"},
                    ),
                    (
                        "/api/v1/projects/user-cohort/runs",
                        {"session_id": sid, "mapping": map_30, "target_rct": flaura_card, "outcome": "Blood ketone change"},
                    ),
                ):
                    r_422 = self.client.post(endpoint, headers=self.web_auth, json=body)
                    self.assertEqual(r_422.status_code, 422, f"{endpoint} -> {r_422.text}")
                    self.assertEqual(r_422.json()["error_code"], "PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED")

                mcp_exp_422 = self.client.post(
                    "/mcp",
                    headers=self.mcp_auth,
                    json={
                        "jsonrpc": "2.0",
                        "id": 201,
                        "method": "tools/call",
                        "params": {
                            "name": "start_user_cohort_tte",
                            "arguments": {
                                "session_id": sid,
                                "nct_id": "NCT02296125",
                                "outcome": "Blood ketone change",
                                "bootstrap_iterations": 20,
                            },
                        },
                    },
                )
                self.assertEqual(mcp_exp_422.status_code, 200)
                self.assertTrue(mcp_exp_422.json()["result"]["isError"], mcp_exp_422.text)
                self.assertIn("PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED", mcp_exp_422.text)

                # 3. Blocker 2 E2E (Step 2 persisted `condition`/`outcome` overrides WITH untrusted PII tokens -> carried forward even when omitted in Step 3/4!)
                srch_mismatch = self.client.post(
                    "/api/rcts/search",
                    headers=self.web_auth,
                    json={
                        "session_id": sid,
                        "project_mode": "user_cohort",
                        "query": "Osimertinib Gefitinib NSCLC",
                        "condition": "EGFR-mutated non-small cell lung cancer Alice Smith",
                        "intervention": "Osimertinib",
                        "comparator": "Gefitinib or Erlotinib",
                        "outcome": "Blood ketone change 987654321",
                    },
                )
                self.assertEqual(srch_mismatch.status_code, 200, srch_mismatch.text)
                flaura_demoted = next(c for c in srch_mismatch.json()["candidates"] if c["nct_id"] == "NCT02296125")
                self.assertEqual(flaura_demoted["outcome_alignment"], "surrogate_or_mismatched")
                self.assertLessEqual(flaura_demoted["match_score"], 0.48)
                self.assertFalse(flaura_demoted["is_recommended"])

                # Verify Step 2 persisted `override_outcome` blocks subsequent draft / dual-runs / v1 / MCP even when `outcome` is omitted!
                for endpoint, body in (
                    ("/api/rcts/draft", {"project_mode": "user_cohort", "session_id": sid, "target_rct": flaura_demoted}),
                    ("/api/dual-runs", {"project_mode": "user_cohort", "session_id": sid, "mapping": map_30, "target_rct": flaura_demoted}),
                    ("/api/v1/projects/user-cohort/runs", {"session_id": sid, "mapping": map_30, "target_rct": flaura_demoted}),
                ):
                    r_persisted_422 = self.client.post(endpoint, headers=self.web_auth, json=body)
                    self.assertEqual(r_persisted_422.status_code, 422, f"Persisted override {endpoint} -> {r_persisted_422.text}")
                    self.assertEqual(r_persisted_422.json()["error_code"], "PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED")

                mcp_persisted_422 = self.client.post(
                    "/mcp",
                    headers=self.mcp_auth,
                    json={
                        "jsonrpc": "2.0",
                        "id": 202,
                        "method": "tools/call",
                        "params": {
                            "name": "start_user_cohort_tte",
                            "arguments": {"session_id": sid, "nct_id": "NCT02296125", "bootstrap_iterations": 20},
                        },
                    },
                )
                self.assertEqual(mcp_persisted_422.status_code, 200)
                self.assertTrue(mcp_persisted_422.json()["result"]["isError"], mcp_persisted_422.text)
                self.assertIn("PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED", mcp_persisted_422.text)

        # Verify 3-layer LLM privacy guard: ZERO leak of `Alice`, `Smith`, `987654321`, or `PT-ONC-` in any outbound LLM request!
        self.assertGreaterEqual(len(captured_llm_payloads), 2)
        for p_str in captured_llm_payloads:
            for forbidden_tok in ("Alice", "Smith", "987654321", "PT-ONC-"):
                self.assertNotIn(forbidden_tok, p_str)

    def test_round8_short_surrogate_outcomes_422_and_os_pfs_still_pass(self):
        """Round-8 E2E: ORR/CRP/CEA/LDH/PSA vs FLAURA stay unconfirmed 422 with score clamped below a mocked DeepSeek 0.99; OS and PFS still draft and run."""
        csv_lines = ["patient_key,arm_drug,time_days,os_event,age,ecog"]
        for i in range(30):
            drug = "Osimertinib" if i % 2 == 0 else "Gefitinib or Erlotinib"
            csv_lines.append(
                f"PT-R8-{i + 1:03d},{drug},{150 + (i * 23) % 520},{1 if i % 3 == 0 else 0},{50 + ((i * 7) % 22)},{(i // 2) % 2}"
            )
        custom_csv = ("\n".join(csv_lines) + "\n").encode("utf-8")

        def fake_llm(req, timeout):
            raw_str = req.data.decode("utf-8")
            if "public_reference_context" in raw_str:
                ans = {
                    "rankings": [
                        {"nct_id": "NCT02296125", "match_score": 0.99, "recommended": True, "reason_zh": "DeepSeek 高分仍须服从终点硬上限"},
                    ]
                }
            else:
                ans = {
                    "domain_summary_zh": "EGFR 突变非小细胞肺癌",
                    "condition_query_en": "EGFR-Mutated Non-Small Cell Lung Cancer",
                    "intervention_query_en": "Osimertinib",
                    "comparator_query_en": "Gefitinib or Erlotinib",
                    "outcome_query_en": "Overall Survival",
                    "search_query_en": "Osimertinib Gefitinib EGFR Non-Small Cell Lung Cancer",
                }
            return _FakeHTTPResp(json.dumps({"choices": [{"message": {"content": json.dumps(ans, ensure_ascii=False)}}]}).encode("utf-8"))

        with mock.patch.dict(
            os.environ,
            {
                "EMULATRX_LLM_API_KEY_FILE": str(self.base / "absent_secret"),
                "EMULATRX_LLM_API_KEY": "sk-round8-test",
                "EMULATRX_OFFLINE_RCT_CATALOG": "1",
            },
            clear=False,
        ):
            with mock.patch.object(llm_assist, "urlopen_llm", side_effect=fake_llm):
                up = self.client.post(
                    "/api/sessions/upload",
                    headers=self.web_auth,
                    files={"file": ("egfr_nsclc_r8.csv", custom_csv, "text/csv")},
                    data={"user_note": "EGFR-mutated non-small cell lung cancer, overall survival"},
                )
                self.assertEqual(up.status_code, 200, up.text)
                sid = up.json()["session_id"]
                mapping = up.json()["default_mapping"]
                mapping["eligibility_criteria"] = []
                mapping["max_abs_smd"] = 0.65
                mapping["min_ess_ratio"] = 0.10
                mapping["bootstrap_iterations"] = 20
                flaura = next(r for r in rct_matcher.OFFLINE_CURATED_RCTS if r["nct_id"] == "NCT02296125")

                for bad_out in ("ORR", "CRP", "CEA", "LDH", "PSA"):
                    srch = self.client.post(
                        "/api/rcts/search",
                        headers=self.web_auth,
                        json={
                            "session_id": sid,
                            "project_mode": "user_cohort",
                            "query": "Osimertinib Gefitinib EGFR Non-Small Cell Lung Cancer",
                            "condition": "EGFR-mutated Non-Small Cell Lung Cancer",
                            "intervention": "Osimertinib",
                            "comparator": "Gefitinib or Erlotinib",
                            "outcome": bad_out,
                        },
                    )
                    self.assertEqual(srch.status_code, 200, srch.text)
                    card = next(c for c in srch.json()["candidates"] if c["nct_id"] == "NCT02296125")
                    self.assertNotEqual(card["overall_alignment_status"], "protocol_aligned", bad_out)
                    self.assertFalse(card["is_recommended"], bad_out)
                    self.assertLessEqual(card["match_score"], 0.48, bad_out)
                    self.assertEqual(card.get("ranking_mode"), "llm_assisted", bad_out)

                    for endpoint, body in (
                        (
                            "/api/rcts/draft",
                            {
                                "project_mode": "user_cohort",
                                "session_id": sid,
                                "target_rct": flaura,
                                "condition": "EGFR-mutated Non-Small Cell Lung Cancer",
                                "intervention": "Osimertinib",
                                "comparator": "Gefitinib or Erlotinib",
                                "outcome": bad_out,
                            },
                        ),
                        (
                            "/api/dual-runs",
                            {
                                "project_mode": "user_cohort",
                                "session_id": sid,
                                "mapping": mapping,
                                "target_rct": flaura,
                                "condition": "EGFR-mutated Non-Small Cell Lung Cancer",
                                "intervention": "Osimertinib",
                                "comparator": "Gefitinib or Erlotinib",
                                "outcome": bad_out,
                            },
                        ),
                    ):
                        blocked = self.client.post(endpoint, headers=self.web_auth, json=body)
                        self.assertEqual(blocked.status_code, 422, f"{bad_out} {endpoint} -> {blocked.text}")
                        self.assertEqual(blocked.json()["error_code"], "PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED")

                for good_out in ("OS", "PFS"):
                    srch_ok = self.client.post(
                        "/api/rcts/search",
                        headers=self.web_auth,
                        json={
                            "session_id": sid,
                            "project_mode": "user_cohort",
                            "query": "Osimertinib Gefitinib EGFR Non-Small Cell Lung Cancer",
                            "condition": "EGFR-mutated Non-Small Cell Lung Cancer",
                            "intervention": "Osimertinib",
                            "comparator": "Gefitinib or Erlotinib",
                            "outcome": good_out,
                        },
                    )
                    self.assertEqual(srch_ok.status_code, 200, srch_ok.text)
                    card_ok = next(c for c in srch_ok.json()["candidates"] if c["nct_id"] == "NCT02296125")
                    self.assertEqual(card_ok["overall_alignment_status"], "protocol_aligned", good_out)
                    self.assertTrue(card_ok["is_recommended"], good_out)
                    draft_ok = self.client.post(
                        "/api/rcts/draft",
                        headers=self.web_auth,
                        json={
                            "project_mode": "user_cohort",
                            "session_id": sid,
                            "target_rct": card_ok,
                            "outcome": good_out,
                        },
                    )
                    self.assertEqual(draft_ok.status_code, 200, f"{good_out} draft -> {draft_ok.text}")
                    run_ok = self.client.post(
                        "/api/dual-runs",
                        headers=self.web_auth,
                        json={
                            "project_mode": "user_cohort",
                            "session_id": sid,
                            "mapping": mapping,
                            "target_rct": card_ok,
                            "outcome": good_out,
                        },
                    )
                    self.assertEqual(run_ok.status_code, 202, f"{good_out} dual-runs -> {run_ok.text}")
                    finished = self.wait_dual_run(run_ok.json()["run_id"])
                    self.assertEqual(finished["state"], "completed", finished)


if __name__ == "__main__":
    unittest.main()
