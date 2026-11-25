"""Offline unit & CLI tests for Issue #15 dual-mode TTE: privacy-guarded LLM assist, ClinicalTrials.gov matcher, cohort_ingest, and pipelines."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from urllib.error import URLError

import numpy as np
import pandas as pd

from emulatrx_tte import (
    ContractError,
    GateBlockedError,
    SOURCE_MODE_SIMULATED_COHORT,
    SOURCE_MODE_USER_COHORT,
    assert_llm_payload_privacy_safe,
    build_simulated_protocol_and_dgp,
    estimate_competing_risks,
    fetch_rct_by_nct,
    generate_sample_cohort_csv_bytes,
    get_llm_status,
    ingest_user_cohort,
    interpret_cohort_profile,
    parse_research_idea_pico,
    parse_timeframe_to_horizons,
    profile_cohort_dataframe,
    read_tabular_cohort_bytes,
    recommend_column_mapping,
    run_simulated_cohort_pipeline,
    run_user_cohort_pipeline,
    sanitize_profile_for_llm,
    search_candidate_rcts,
)
from emulatrx_tte import llm_assist, rct_matcher


ROOT = Path(__file__).resolve().parents[1]


class _FakeHTTPResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


MOCK_CTGOV_SEARCH_PAYLOAD = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT03036124",
                    "briefTitle": "Study to Evaluate the Effect of Dapagliflozin in Chronic Heart Failure (DAPA-HF)",
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
                        {"label": "Dapagliflozin 10mg", "type": "EXPERIMENTAL"},
                        {"label": "Matching Placebo / Standard Care", "type": "PLACEBO_COMPARATOR"},
                    ],
                    "interventions": [
                        {"name": "Dapagliflozin", "type": "DRUG"},
                        {"name": "Placebo", "type": "DRUG"},
                    ],
                },
                "eligibilityModule": {
                    "minimumAge": "18 Years",
                    "maximumAge": "N/A",
                    "sex": "ALL",
                    "eligibilityCriteria": "Inclusion Criteria:\n* Adults >= 18 years with symptomatic HFrEF\n* eGFR >= 30 mL/min/1.73m2",
                },
                "outcomesModule": {
                    "primaryOutcomes": [
                        {
                            "measure": "Composite of Worsening Heart Failure or Cardiovascular Death",
                            "timeFrame": "Up to 18 months",
                        }
                    ]
                },
            },
            "hasResults": True,
        }
    ]
}


class PrivacyGuardAndLLMAssistTests(unittest.TestCase):
    def test_llm_request_body_contains_zero_patient_ids_or_individual_rows(self):
        """STRICT PRIVACY REDLINE: Verify outbound LLM HTTP request never contains subject IDs or row values."""
        for sample_id in ("cardiorenal_sglt2i", "oncology_io_nsclc"):
            fname, raw = generate_sample_cohort_csv_bytes(sample_id)
            df, _, _ = read_tabular_cohort_bytes(raw, fname)
            id_col = "deid_patient_id" if "deid_patient_id" in df.columns else "case_no"
            all_subject_ids = df[id_col].astype(str).tolist()
            self.assertGreaterEqual(len(all_subject_ids), 1200)
            self.assertFalse(any(sid.startswith("SYN-") for sid in all_subject_ids))

            profile = profile_cohort_dataframe(df, filename=fname, input_sha256=hashlib.sha256(raw).hexdigest())

            captured_requests: list[bytes] = []

            def fake_urlopen_llm(request, timeout):
                captured_requests.append(request.data)
                reply = {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "domain_summary_zh": "AI 识别：心肾/肿瘤队列汇总特征",
                                        "condition_query_en": "Heart Failure",
                                        "intervention_query_en": "Dapagliflozin",
                                        "search_query_en": "Dapagliflozin Heart Failure",
                                    }
                                )
                            }
                        }
                    ]
                }
                return _FakeHTTPResponse(json.dumps(reply).encode("utf-8"))

            # Simulate existing env without secret file
            with mock.patch.dict(
                os.environ,
                {
                    "EMULATRX_LLM_API_KEY_FILE": str(Path(tempfile.gettempdir()) / "nonexistent-secret-key"),
                    "EMULATRX_LLM_API_KEY": "sk-mock-test-key",
                    "EMULATRX_LLM_MODEL": "gpt-4o-mini",
                },
                clear=False,
            ):
                self.assertTrue(get_llm_status()["enabled"])
                with mock.patch.object(llm_assist, "urlopen_llm", side_effect=fake_urlopen_llm):
                    res = interpret_cohort_profile(
                        profile,
                        user_note="队列备注含意外患者号 Hosp-2023-0001 请自动脱敏",
                        forbidden_values=tuple(all_subject_ids[:100]),
                    )

            self.assertEqual(res["assist_mode"], "llm_assisted")
            self.assertEqual(len(captured_requests), 1)
            sent_text = captured_requests[0].decode("utf-8")

            # Assert NOT A SINGLE patient ID or raw user_note phrase appears in the LLM request body
            for sid in all_subject_ids:
                self.assertNotIn(sid, sent_text, f"Privacy leak: {sid} found in outbound LLM payload!")
            self.assertNotIn("Hosp-2023-", sent_text)
            self.assertNotIn("ONC-880", sent_text)
            self.assertNotIn("队列备注含意外患者号", sent_text)

            # Parse inner user message JSON envelope and verify only schema + aggregate stats + whitelisted concepts exist
            outer = json.loads(sent_text)
            inner_envelope = json.loads(outer["messages"][1]["content"])
            self.assertIn("cohort_outbound_context", inner_envelope)
            inner_payload = inner_envelope["cohort_outbound_context"]
            self.assertEqual(inner_payload["privacy_contract"], "schema_and_aggregates_only_zero_individual_rows")
            self.assertIn("whitelisted_domain_concepts", inner_payload)
            self.assertNotIn("user_note", inner_payload)
            col_names_sent = {c["name"] for c in inner_payload["columns"]}
            self.assertNotIn(id_col, col_names_sent)
            for c in inner_payload["columns"]:
                if "distinct_categories" in c:
                    self.assertLessEqual(len(c["distinct_categories"]), 20)

        # Also verify active runtime blocker raises PRIVACY_VIOLATION_BLOCKED if a raw ID or standalone numeric ID slips in
        with self.assertRaises(ContractError) as ctx:
            assert_llm_payload_privacy_safe(b'{"leak": "Hosp-2023-0042"}')
        self.assertEqual(ctx.exception.code, "PRIVACY_VIOLATION_BLOCKED")
        with self.assertRaises(ContractError) as ctx_num:
            assert_llm_payload_privacy_safe(b'{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"Patient 123456789"}]}')
        self.assertEqual(ctx_num.exception.code, "PRIVACY_VIOLATION_BLOCKED")

    def test_whitelist_anonymous_placeholders_block_alice_smith_and_note_identifiers(self):
        """Regression test for Codex review Blocking 1: non-whitelisted categories (even count>=5) become CATEGORY_LEVEL_* and raw user_note never leaves."""
        df = pd.DataFrame(
            {
                "deid_id": [f"PT-{i + 1:04d}" for i in range(20)],
                "site_group": ["Alice Smith"] * 12 + ["Bob Jones"] * 8,
                "initiated_drug": ["Dapagliflozin"] * 10 + ["Sitagliptin"] * 10,
                "rare_status": ["censored"] * 17 + ["death"] * 3,
                "followup_months": [12.0 + i * 0.5 for i in range(20)],
                "endpoint": ["censored"] * 10 + ["hf_hospitalization"] * 6 + ["cv_death"] * 4,
            }
        )
        prof = profile_cohort_dataframe(df, filename="site_cohort.csv")

        # Local UI profile preserves raw category labels for the user's own column mapping dropdowns
        site_col_local = next(c for c in prof["columns"] if c["name"] == "site_group")
        self.assertEqual(site_col_local["top_categories"][0]["value"], "Alice Smith")

        captured: list[str] = []

        def fake_llm(req, timeout):
            captured.append(req.data.decode("utf-8"))
            return _FakeHTTPResponse(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "domain_summary_zh": "心衰队列",
                                            "condition_query_en": "Heart Failure",
                                            "intervention_query_en": "Dapagliflozin",
                                            "search_query_en": "Dapagliflozin Heart Failure",
                                            "rankings": [{"nct_id": "NCT03036124", "score": 95, "reason_zh": "高度匹配"}],
                                        }
                                    )
                                }
                            }
                        ]
                    }
                ).encode("utf-8")
            )

        with mock.patch.dict(
            os.environ,
            {
                "EMULATRX_LLM_API_KEY_FILE": str(Path(tempfile.gettempdir()) / "absent-key-file"),
                "EMULATRX_LLM_API_KEY": "sk-whitelist-test",
            },
            clear=False,
        ):
            with mock.patch.object(llm_assist, "urlopen_llm", side_effect=fake_llm):
                interp = interpret_cohort_profile(
                    prof,
                    user_note="Patient identifier 123456789 site lead Alice Smith 达格列净治疗心衰",
                    forbidden_values=("Alice Smith", "Bob Jones", "123456789"),
                )
                self.assertEqual(interp["assist_mode"], "llm_assisted")

                reranked, mode = llm_assist.rerank_and_explain_rcts(
                    [
                        {
                            "nct_id": "NCT03036124",
                            "brief_title": "DAPA-HF",
                            "phases": ["PHASE3"],
                            "enrollment_count": 4744,
                            "conditions": ["Heart Failure"],
                            "experimental_arms": ["Dapagliflozin"],
                            "comparator_arms": ["Placebo"],
                            "primary_outcomes": ["Worsening HF or CV Death"],
                        }
                    ],
                    {"query": "Dapagliflozin Heart Failure Patient identifier 123456789 Alice Smith"},
                    forbidden_values=("Alice Smith", "Bob Jones", "123456789"),
                )
                self.assertEqual(mode, "llm_assisted")
                self.assertEqual(reranked[0]["nct_id"], "NCT03036124")

        self.assertEqual(len(captured), 2)
        for body in captured:
            for forbidden in ("Alice Smith", "Bob Jones", "123456789", "PT-0001", "Patient identifier"):
                self.assertNotIn(forbidden, body)

        first_envelope = json.loads(json.loads(captured[0])["messages"][1]["content"])
        first_inner = first_envelope["cohort_outbound_context"]
        cols_by_name = {c["name"]: c for c in first_inner["columns"]}
        self.assertEqual(cols_by_name["site_group"]["distinct_categories"], ["CATEGORY_LEVEL_1", "CATEGORY_LEVEL_2"])
        # Whitelisted drugs with count=10 >= 5 stay literal; rare_status 'death' with count=3 < 5 becomes CATEGORY_LEVEL_2
        self.assertEqual(cols_by_name["initiated_drug"]["distinct_categories"], ["Dapagliflozin", "Sitagliptin"])
        self.assertEqual(cols_by_name["rare_status"]["distinct_categories"], ["censored", "CATEGORY_LEVEL_2"])

        # Verify column-name vs cell-value separation & Layer A/B isolation:
        # Column named 'source' does NOT false-positive when another column had forbidden cell value 'source',
        # while an unredacted forbidden value inside `distinct_categories` DOES fail-closed!
        msgs_ok = llm_assist.build_verified_llm_messages(
            "System prompt mentioning patient rows and clinical study trials.",
            cohort_outbound_context={
                "privacy_contract": "schema_and_aggregates_only_zero_individual_rows",
                "columns": [
                    {"name": "source", "dtype_kind": "categorical", "inferred_role": "categorical_covariate", "distinct_categories": ["CATEGORY_LEVEL_1"]},
                    {"name": "data_origin", "dtype_kind": "categorical", "inferred_role": "categorical_covariate", "distinct_categories": ["CATEGORY_LEVEL_1", "CATEGORY_LEVEL_2"]},
                ],
                "whitelisted_domain_concepts": ["Heart Failure", "Dapagliflozin"],
            },
            public_reference_context={
                "candidates": [{"nct_id": "NCT03036124", "brief_title": "Adult patients in claims study from hospital source"}]
            },
            forbidden_values=("patient", "claims", "study", "hospital", "source", "level"),
        )
        self.assertEqual(len(msgs_ok), 2)

        with self.assertRaises(ContractError) as ctx_cell_leak:
            llm_assist.build_verified_llm_messages(
                "System prompt",
                cohort_outbound_context={
                    "columns": [{"name": "data_origin", "distinct_categories": ["claims"]}],
                },
                forbidden_values=("patient", "claims"),
            )
        self.assertEqual(ctx_cell_leak.exception.code, "PRIVACY_VIOLATION_BLOCKED")

    def test_deterministic_fallback_without_llm_key_and_on_llm_network_error(self):
        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ("EMULATRX_LLM_API_KEY", "OPENAI_API_KEY")
        }
        with mock.patch.dict(os.environ, env_clean, clear=True):
            status = get_llm_status()
            self.assertFalse(status["enabled"])
            self.assertEqual(status["mode"], "deterministic_rules")
            self.assertIn("本地规则引擎模式", status["label_zh"])

            pico = parse_research_idea_pico("达格列净对比西格列汀在2型糖尿病合并心衰患者中的心衰住院与竞争死亡")
            self.assertEqual(pico["assist_mode"], "deterministic_rules")
            self.assertIn("Dapagliflozin", pico["search_query_en"])
            self.assertIn("Heart Failure", pico["search_query_en"])

        # With key configured, network failure smoothly falls back to deterministic rules without raising
        with mock.patch.dict(os.environ, {"EMULATRX_LLM_API_KEY": "sk-test"}, clear=False):
            with mock.patch.object(llm_assist, "urlopen_llm", side_effect=URLError("timeout")):
                pico_fb = parse_research_idea_pico("帕博利珠单抗联合含铂化疗一线治疗晚期非小细胞肺癌")
                self.assertEqual(pico_fb["assist_mode"], "deterministic_rules")
                self.assertIn("Pembrolizumab", pico_fb["search_query_en"])

    def test_secret_file_priority_empty_file_fallback_zero_key_leak_and_deepseek_bailian_gateways(self):
        # 1. URL normalization for DeepSeek, Alibaba Bailian (DashScope), and OpenAI
        self.assertEqual(llm_assist.normalize_llm_base_url("https://api.deepseek.com/"), "https://api.deepseek.com/v1")
        self.assertEqual(llm_assist.normalize_llm_base_url("https://api.deepseek.com/v1/"), "https://api.deepseek.com/v1")
        self.assertEqual(
            llm_assist.normalize_llm_base_url("https://dashscope.aliyuncs.com/compatible-mode"),
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(
            llm_assist.normalize_llm_base_url("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(llm_assist.normalize_llm_base_url("https://api.openai.com/v1"), "https://api.openai.com/v1")

        with tempfile.TemporaryDirectory() as td:
            tbase = Path(td)
            secret_file = tbase / "llm_api_key"
            default_secret_file = tbase / "default_run_secret"
            empty_file = tbase / "empty_secret"
            empty_file.write_text("   \n\t  ", encoding="utf-8")

            secret_token = "sk-TOP-PRIORITY-FILE-SECRET-99887766"
            secret_file.write_text(f"  {secret_token}\n", encoding="utf-8")
            default_secret_file.write_text("sk-DEFAULT-RUN-SECRET-554433\n", encoding="utf-8")

            # 2. Empty secret file + no env keys -> clean deterministic fallback & honest banner
            clean_env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("EMULATRX_LLM_API_KEY_FILE", "EMULATRX_LLM_API_KEY", "OPENAI_API_KEY")
            }
            with mock.patch.object(llm_assist, "DEFAULT_LLM_SECRET_FILE", tbase / "absent_default"):
                with mock.patch.dict(os.environ, {**clean_env, "EMULATRX_LLM_API_KEY_FILE": str(empty_file)}, clear=True):
                    st_empty = get_llm_status()
                    self.assertFalse(st_empty["enabled"])
                    self.assertEqual(st_empty["mode"], "deterministic_rules")
                    self.assertIn("当前未挂载模型密钥，已启用确定性规则引擎兜底", st_empty["label_zh"])

            # 3. Default /run/secrets/llm_api_key auto-pickup when EMULATRX_LLM_API_KEY_FILE is unset
            with mock.patch.object(llm_assist, "DEFAULT_LLM_SECRET_FILE", default_secret_file):
                with mock.patch.dict(
                    os.environ,
                    {**clean_env, "EMULATRX_LLM_API_KEY": "sk-LOWER-PRIORITY-ENV"},
                    clear=True,
                ):
                    st_def = get_llm_status()
                    self.assertTrue(st_def["enabled"])
                    self.assertEqual(st_def["key_source"], "default_secret_file")
                    self.assertNotIn("sk-DEFAULT-RUN-SECRET-554433", json.dumps(st_def, ensure_ascii=False))

            # 4. Explicit EMULATRX_LLM_API_KEY_FILE wins over EMULATRX_LLM_API_KEY & OPENAI_API_KEY + DeepSeek-R1 <think> & ```json parsing
            seen_auth: list[str] = []
            seen_urls: list[str] = []
            seen_bodies: list[dict] = []

            def fake_deepseek_reasoner(req, timeout):
                seen_urls.append(req.full_url)
                seen_auth.append(req.get_header("Authorization"))
                seen_bodies.append(json.loads(req.data.decode("utf-8")))
                reasoning_reply = (
                    "<think>\nLet us extract PICO carefully for Heart Failure and Dapagliflozin.\n</think>\n"
                    "Here is the JSON result:\n```json\n"
                    + json.dumps(
                        {
                            "population": "慢性心衰成人患者",
                            "intervention": "达格列净 (Dapagliflozin)",
                            "comparator": "西格列汀 (Sitagliptin)",
                            "primary_outcome": "心衰住院或心血管死亡",
                            "competing_event": "非心血管竞争死亡",
                            "condition_query_en": "Heart Failure",
                            "intervention_query_en": "Dapagliflozin",
                            "search_query_en": "Dapagliflozin Heart Failure",
                        },
                        ensure_ascii=False,
                    )
                    + "\n```"
                )
                return _FakeHTTPResponse(
                    json.dumps({"choices": [{"message": {"content": reasoning_reply}}]}).encode("utf-8")
                )

            with mock.patch.dict(
                os.environ,
                {
                    **clean_env,
                    "EMULATRX_LLM_API_KEY_FILE": str(secret_file),
                    "EMULATRX_LLM_API_KEY": "sk-env-override-should-lose",
                    "OPENAI_API_KEY": "sk-openai-should-lose",
                    "EMULATRX_LLM_BASE_URL": "https://api.deepseek.com/",
                    "EMULATRX_LLM_MODEL": "deepseek-reasoner",
                },
                clear=True,
            ):
                st_file = get_llm_status()
                self.assertTrue(st_file["enabled"])
                self.assertEqual(st_file["key_source"], "secret_file")
                self.assertEqual(st_file["gateway_host"], "api.deepseek.com")
                self.assertNotIn(secret_token, json.dumps(st_file, ensure_ascii=False))

                with mock.patch.object(llm_assist, "urlopen_llm", side_effect=fake_deepseek_reasoner):
                    pico_ds = parse_research_idea_pico("达格列净对比西格列汀治疗慢性心衰")
                self.assertEqual(pico_ds["assist_mode"], "llm_assisted")
                self.assertEqual(pico_ds["population"], "慢性心衰成人患者")
                self.assertEqual(seen_urls[0], "https://api.deepseek.com/v1/chat/completions")
                self.assertEqual(seen_auth[0], f"Bearer {secret_token}")
                # Must NOT send OpenAI-only response_format or temperature on deepseek-reasoner
                self.assertNotIn("response_format", seen_bodies[0])
                self.assertNotIn("temperature", seen_bodies[0])
                self.assertNotIn(secret_token, json.dumps(pico_ds, ensure_ascii=False))


class RCTMatcherTests(unittest.TestCase):
    def test_live_ctgov_mock_search_exact_nct_and_explicit_offline_fallback(self):
        horizons, admin_end = parse_timeframe_to_horizons("Up to 18 months")
        self.assertEqual(horizons, [90.0, 180.0, 365.0])
        self.assertGreater(admin_end, 365.0)

        def fake_ctgov_ok(request, timeout):
            if "/studies/NCT03036124" in request.full_url:
                return _FakeHTTPResponse(json.dumps(MOCK_CTGOV_SEARCH_PAYLOAD["studies"][0]).encode("utf-8"))
            return _FakeHTTPResponse(json.dumps(MOCK_CTGOV_SEARCH_PAYLOAD).encode("utf-8"))

        env_no_llm = {k: v for k, v in os.environ.items() if k not in ("EMULATRX_LLM_API_KEY", "OPENAI_API_KEY", "EMULATRX_OFFLINE_RCT_CATALOG")}
        with mock.patch.dict(os.environ, env_no_llm, clear=True):
            with mock.patch.object(rct_matcher, "urlopen_ctgov", side_effect=fake_ctgov_ok):
                res = search_candidate_rcts("Dapagliflozin Heart Failure", condition="Heart Failure", intervention="Dapagliflozin")
                self.assertEqual(res["source"], "clinicaltrials_gov_api_v2")
                self.assertEqual(res["candidates"][0]["nct_id"], "NCT03036124")
                self.assertEqual(len(res["candidates"][0]["receipt_sha256"]), 64)

                exact = fetch_rct_by_nct("NCT03036124")
                self.assertEqual(exact["source"], "clinicaltrials_gov_api_v2")
                self.assertEqual(exact["nct_id"], "NCT03036124")

            # Simulate complete network outage -> MUST explicitly label source="offline_curated_catalog_fallback"
            with mock.patch.object(rct_matcher, "urlopen_ctgov", side_effect=URLError("network unreachable")):
                offline_res = search_candidate_rcts("Pembrolizumab Non-Small Cell Lung Cancer")
                self.assertEqual(offline_res["source"], "offline_curated_catalog_fallback")
                self.assertGreaterEqual(len(offline_res["candidates"]), 1)
                self.assertEqual(offline_res["candidates"][0]["source"], "offline_curated_catalog_fallback")
                self.assertEqual(offline_res["candidates"][0]["nct_id"], "NCT02578680")

    def test_nct06642272_dual_arm_alignment_score_cap_and_bilingual_query_translation(self):
        """Live QA regression: NCT06642272 (Empagliflozin vs Dapagliflozin) against Dapagliflozin vs Sitagliptin is capped <=0.45 and never recommended."""
        live_studies_payload = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT06642272",
                            "briefTitle": "Head-to-Head Comparison of Empagliflozin vs Dapagliflozin in Type 2 Diabetes and Heart Failure",
                            "officialTitle": "Randomized Active-Controlled Trial of Empagliflozin Versus Dapagliflozin",
                        },
                        "statusModule": {"overallStatus": "RECRUITING"},
                        "designModule": {
                            "studyType": "INTERVENTIONAL",
                            "phases": ["PHASE4"],
                            "enrollmentInfo": {"count": 1200},
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
                        "eligibilityModule": {"minimumAge": "18 Years", "maximumAge": "N/A", "sex": "ALL", "eligibilityCriteria": "* T2DM"},
                        "outcomesModule": {"primaryOutcomes": [{"measure": "HF Hospitalization or CV Death", "timeFrame": "12 months"}]},
                    },
                    "hasResults": False,
                },
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT02471404",
                            "briefTitle": "Class Comparison of Empagliflozin Versus Linagliptin in Type 2 Diabetes",
                            "officialTitle": "Randomized Study of Empagliflozin vs Linagliptin",
                        },
                        "statusModule": {"overallStatus": "COMPLETED"},
                        "designModule": {
                            "studyType": "INTERVENTIONAL",
                            "phases": ["PHASE3"],
                            "enrollmentInfo": {"count": 900},
                            "designInfo": {"allocation": "RANDOMIZED"},
                        },
                        "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                        "armsInterventionsModule": {
                            "armGroups": [
                                {"label": "Empagliflozin 10 mg", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Empagliflozin"]},
                                {"label": "Linagliptin 5 mg", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Linagliptin"]},
                            ],
                            "interventions": [
                                {"name": "Empagliflozin", "type": "DRUG"},
                                {"name": "Linagliptin", "type": "DRUG"},
                            ],
                        },
                        "eligibilityModule": {"minimumAge": "18 Years", "maximumAge": "N/A", "sex": "ALL", "eligibilityCriteria": "* T2DM"},
                        "outcomesModule": {"primaryOutcomes": [{"measure": "Cardiorenal Composite", "timeFrame": "12 months"}]},
                    },
                    "hasResults": True,
                },
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT01095653",
                            "briefTitle": "Dapagliflozin Compared With Sitagliptin in Patients With Type 2 Diabetes",
                            "officialTitle": "Randomized Double-Blind Active-Controlled Phase 3 Study of Dapagliflozin vs Sitagliptin",
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
                        "eligibilityModule": {"minimumAge": "18 Years", "maximumAge": "N/A", "sex": "ALL", "eligibilityCriteria": "* T2DM"},
                        "outcomesModule": {"primaryOutcomes": [{"measure": "Glycemic & Cardiorenal Outcome", "timeFrame": "48 weeks"}]},
                    },
                    "hasResults": True,
                },
            ]
        }

        seen_ctgov_urls: list[str] = []

        def fake_ctgov(req, timeout):
            seen_ctgov_urls.append(req.full_url)
            return _FakeHTTPResponse(json.dumps(live_studies_payload).encode("utf-8"))

        # Mock DeepSeek intentionally trying to boost NCT06642272 to match_score=0.88 & recommended=True!
        def fake_deepseek_overrate(req, timeout):
            bad_llm_ranking = {
                "rankings": [
                    {"nct_id": "NCT06642272", "match_score": 0.88, "recommended": True, "reason_zh": "含达格列净但在对照臂与恩格列净互比"},
                    {"nct_id": "NCT02471404", "match_score": 0.90, "recommended": True, "reason_zh": "SGLT2i 对比 DPP-4i 同机制参考"},
                    {"nct_id": "NCT01095653", "match_score": 0.92, "recommended": True, "reason_zh": "达格列净头对头对比西格列汀完全一致"},
                ]
            }
            return _FakeHTTPResponse(
                json.dumps({"choices": [{"message": {"content": json.dumps(bad_llm_ranking, ensure_ascii=False)}}]}).encode("utf-8")
            )

        with mock.patch.dict(
            os.environ,
            {
                "EMULATRX_LLM_API_KEY_FILE": str(Path(tempfile.gettempdir()) / "absent-key"),
                "EMULATRX_LLM_API_KEY": "sk-deepseek-qa-test",
                "EMULATRX_OFFLINE_RCT_CATALOG": "0",
            },
            clear=False,
        ):
            with mock.patch.object(rct_matcher, "urlopen_ctgov", side_effect=fake_ctgov), mock.patch.object(
                llm_assist, "urlopen_llm", side_effect=fake_deepseek_overrate
            ):
                # Pass Chinese condition/intervention/comparator to also verify bilingual English translation on ClinicalTrials.gov URL!
                res = search_candidate_rcts(
                    "达格列净 对比 西格列汀 2型糖尿病",
                    condition="2型糖尿病",
                    intervention="达格列净",
                    comparator="西格列汀",
                )

        # 1. Verify bilingual translation produced English `Dapagliflozin` + `Sitagliptin` + `Type+2+Diabetes` on CT.gov URL
        self.assertGreaterEqual(len(seen_ctgov_urls), 1)
        first_url_lower = seen_ctgov_urls[0].lower()
        self.assertIn("dapagliflozin", first_url_lower)
        self.assertIn("sitagliptin", first_url_lower)
        self.assertIn("diabetes", first_url_lower)

        # 2. Verify strict ordering: exact_dual_arm (#1) > class_dual_arm (#2, <=0.72) > single_arm_or_comparator_mismatch (#3, <=0.45)
        cands = res["candidates"]
        self.assertEqual([c["nct_id"] for c in cands[:3]], ["NCT01095653", "NCT02471404", "NCT06642272"])

        exact_c = cands[0]
        self.assertEqual(exact_c["arm_alignment_status"], "exact_dual_arm")
        self.assertTrue(exact_c["is_recommended"])
        self.assertGreaterEqual(exact_c["match_score"], 0.85)

        class_c = cands[1]
        self.assertEqual(class_c["arm_alignment_status"], "class_dual_arm")
        self.assertLessEqual(class_c["match_score"], 0.72)
        self.assertFalse(class_c["is_recommended"])

        mismatch_c = cands[2]
        self.assertEqual(mismatch_c["nct_id"], "NCT06642272")
        self.assertEqual(mismatch_c["arm_alignment_status"], "single_arm_or_comparator_mismatch")
        self.assertLessEqual(mismatch_c["match_score"], 0.45)
        self.assertFalse(mismatch_c["is_recommended"])
        self.assertEqual(mismatch_c["arm_alignment_label"], "对照不一致 · 仅供参考")

    def test_round4_shared_background_drug_and_single_or_missing_armgroups_audit(self):
        """Round-4 Probes 1A & 1B: shared background Metformin never masks missing Sitagliptin, and <2 armGroups never fabricate arms."""
        # Probe 1A: Cohort is `Dapagliflozin + Metformin` vs `Sitagliptin + Metformin`
        rct_missing_sita = {
            "nct_id": "NCT09990001",
            "brief_title": "Dapagliflozin Plus Metformin Versus Metformin Monotherapy",
            "experimental_arms": ["Dapagliflozin + Metformin"],
            "comparator_arms": ["Metformin + Placebo"],
            "interventions": ["Drug: Dapagliflozin", "Drug: Metformin", "Drug: Placebo"],
        }
        audit_1a_bad = rct_matcher.audit_study_arm_alignment(
            rct_missing_sita,
            intervention="Dapagliflozin + Metformin",
            comparator="Sitagliptin + Metformin",
            project_mode="user_cohort",
        )
        self.assertEqual(audit_1a_bad["arm_alignment_status"], "single_arm_or_comparator_mismatch")
        self.assertLessEqual(audit_1a_bad["arm_alignment_score_cap"], 0.45)
        self.assertFalse(audit_1a_bad["is_recommended"])
        self.assertTrue(audit_1a_bad["requires_mismatch_confirmation"])

        rct_true_combo = {
            "nct_id": "NCT01095653",
            "brief_title": "Dapagliflozin + Metformin vs Sitagliptin + Metformin",
            "experimental_arms": ["Dapagliflozin 10 mg + Metformin"],
            "comparator_arms": ["Sitagliptin 100 mg + Metformin"],
            "interventions": ["Drug: Dapagliflozin", "Drug: Sitagliptin", "Drug: Metformin"],
        }
        audit_1a_ok = rct_matcher.audit_study_arm_alignment(
            rct_true_combo,
            intervention="Dapagliflozin + Metformin",
            comparator="Sitagliptin + Metformin",
            project_mode="user_cohort",
        )
        self.assertEqual(audit_1a_ok["arm_alignment_status"], "exact_dual_arm")
        self.assertTrue(audit_1a_ok["is_recommended"])
        self.assertFalse(audit_1a_ok["requires_mismatch_confirmation"])

        # True add-on cohort (`Dapagliflozin + Metformin` vs `Metformin`) DOES match `rct_missing_sita`!
        audit_addon_ok = rct_matcher.audit_study_arm_alignment(
            rct_missing_sita,
            intervention="Dapagliflozin + Metformin",
            comparator="Metformin",
            project_mode="user_cohort",
        )
        self.assertEqual(audit_addon_ok["arm_alignment_status"], "exact_dual_arm")

        # Probe 1B-1: Single-arm combination trial (`len(armGroups) == 1` with both drugs in `interventions`)
        raw_single_arm_api = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT09990002", "briefTitle": "Single-Arm Fixed-Dose Dapagliflozin + Sitagliptin Study"},
                "statusModule": {"overallStatus": "COMPLETED"},
                "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE2"], "designInfo": {"allocation": "NON_RANDOMIZED"}},
                "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                "armsInterventionsModule": {
                    "armGroups": [
                        {"label": "Fixed-Dose Dapagliflozin + Sitagliptin", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin", "Drug: Sitagliptin"]}
                    ],
                    "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Sitagliptin", "type": "DRUG"}],
                },
                "outcomesModule": {"primaryOutcomes": [{"measure": "HbA1c change", "timeFrame": "24 weeks"}]},
            }
        }
        norm_single = rct_matcher._normalize_api_study(raw_single_arm_api)
        self.assertEqual(len(norm_single["arm_groups_detail"]), 1)
        self.assertEqual(norm_single["comparator_arms"], [])
        audit_1b_single = rct_matcher.audit_study_arm_alignment(
            norm_single,
            intervention="Dapagliflozin",
            comparator="Sitagliptin",
            project_mode="user_cohort",
        )
        self.assertEqual(audit_1b_single["arm_alignment_status"], "single_arm_or_comparator_mismatch")
        self.assertLessEqual(audit_1b_single["arm_alignment_score_cap"], 0.45)
        self.assertFalse(audit_1b_single["is_recommended"])
        with self.assertRaises(ContractError) as ctx_1b1:
            rct_matcher.verify_and_enrich_target_rct(
                norm_single,
                intervention="Dapagliflozin",
                comparator="Sitagliptin",
                project_mode="user_cohort",
                confirm_arm_mismatch=False,
            )
        self.assertEqual(ctx_1b1.exception.code, "ARM_MISMATCH_CONFIRMATION_REQUIRED")

        # Probe 1B-2: Sparse registry record with `0` armGroups and 2 study-level interventions
        raw_zero_armgroups_api = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT09990003", "briefTitle": "Unlinked Dapagliflozin and Sitagliptin Study"},
                "statusModule": {"overallStatus": "COMPLETED"},
                "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE3"]},
                "conditionsModule": {"conditions": ["Type 2 Diabetes"]},
                "armsInterventionsModule": {
                    "armGroups": [],
                    "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Sitagliptin", "type": "DRUG"}],
                },
                "outcomesModule": {"primaryOutcomes": [{"measure": "HbA1c", "timeFrame": "24 weeks"}]},
            }
        }
        norm_zero = rct_matcher._normalize_api_study(raw_zero_armgroups_api)
        self.assertEqual(len(norm_zero["arm_groups_detail"]), 0)
        audit_1b_zero = rct_matcher.audit_study_arm_alignment(
            norm_zero,
            intervention="Dapagliflozin",
            comparator="Sitagliptin",
            project_mode="user_cohort",
        )
        self.assertNotEqual(audit_1b_zero["arm_alignment_status"], "exact_dual_arm")
        self.assertLessEqual(audit_1b_zero["arm_alignment_score_cap"], 0.45)
        self.assertFalse(audit_1b_zero["is_recommended"])
        with self.assertRaises(ContractError) as ctx_1b2:
            rct_matcher.verify_and_enrich_target_rct(
                norm_zero,
                intervention="Dapagliflozin",
                comparator="Sitagliptin",
                project_mode="user_cohort",
                confirm_arm_mismatch=False,
            )
        self.assertEqual(ctx_1b2.exception.code, "ARM_MISMATCH_CONFIRMATION_REQUIRED")

    def test_round5_full_regimen_drug_set_equality_for_exact_dual_arm(self):
        """Round-5 Blocker 1A & 1B: exact_dual_arm requires identical full active drug sets on both arms (no missing shared background, no extra active drugs)."""
        # Blocker 1A: Cohort `Dapagliflozin + Metformin` vs `Sitagliptin + Metformin` against RCT `Dapagliflozin` vs `Sitagliptin` (missing Metformin)
        rct_mono_h2h = {
            "nct_id": "NCT09990011",
            "brief_title": "Dapagliflozin Monotherapy vs Sitagliptin Monotherapy",
            "experimental_arms": ["Dapagliflozin 10 mg"],
            "comparator_arms": ["Sitagliptin 100 mg"],
            "interventions": ["Drug: Dapagliflozin", "Drug: Sitagliptin"],
        }
        audit_1a = rct_matcher.audit_study_arm_alignment(
            rct_mono_h2h,
            intervention="Dapagliflozin + Metformin",
            comparator="Sitagliptin + Metformin",
            project_mode="user_cohort",
        )
        self.assertNotEqual(audit_1a["arm_alignment_status"], "exact_dual_arm")
        self.assertEqual(audit_1a["arm_alignment_status"], "class_dual_arm")
        self.assertLessEqual(audit_1a["arm_alignment_score_cap"], 0.72)
        self.assertFalse(audit_1a["is_recommended"])
        self.assertTrue(audit_1a["requires_mismatch_confirmation"])
        with self.assertRaises(ContractError) as ctx_1a:
            rct_matcher.verify_and_enrich_target_rct(
                rct_mono_h2h,
                intervention="Dapagliflozin + Metformin",
                comparator="Sitagliptin + Metformin",
                project_mode="user_cohort",
                confirm_arm_mismatch=False,
            )
        self.assertEqual(ctx_1a.exception.code, "ARM_MISMATCH_CONFIRMATION_REQUIRED")

        # Blocker 1B: Cohort `Dapagliflozin` vs `Sitagliptin` against RCT `Dapagliflozin` vs `Sitagliptin + Glimepiride` (extra drug in comparator)
        rct_extra_comp = {
            "nct_id": "NCT09990012",
            "brief_title": "Dapagliflozin vs Sitagliptin Plus Glimepiride",
            "experimental_arms": ["Dapagliflozin 10 mg"],
            "comparator_arms": ["Sitagliptin 100 mg + Glimepiride 4 mg"],
            "interventions": ["Drug: Dapagliflozin", "Drug: Sitagliptin", "Drug: Glimepiride"],
        }
        audit_1b = rct_matcher.audit_study_arm_alignment(
            rct_extra_comp,
            intervention="Dapagliflozin",
            comparator="Sitagliptin",
            project_mode="user_cohort",
        )
        self.assertNotEqual(audit_1b["arm_alignment_status"], "exact_dual_arm")
        self.assertEqual(audit_1b["arm_alignment_status"], "single_arm_or_comparator_mismatch")
        self.assertLessEqual(audit_1b["arm_alignment_score_cap"], 0.45)
        self.assertFalse(audit_1b["is_recommended"])
        self.assertTrue(audit_1b["requires_mismatch_confirmation"])
        with self.assertRaises(ContractError) as ctx_1b:
            rct_matcher.verify_and_enrich_target_rct(
                rct_extra_comp,
                intervention="Dapagliflozin",
                comparator="Sitagliptin",
                project_mode="user_cohort",
                confirm_arm_mismatch=False,
            )
        self.assertEqual(ctx_1b.exception.code, "ARM_MISMATCH_CONFIRMATION_REQUIRED")

        # Positive checks: exact monotherapy, exact background combo, exact add-on, and exact Drug vs Placebo
        self.assertEqual(
            rct_matcher.audit_study_arm_alignment(
                rct_mono_h2h,
                intervention="Dapagliflozin",
                comparator="Sitagliptin",
                project_mode="user_cohort",
            )["arm_alignment_status"],
            "exact_dual_arm",
        )
        rct_placebo = {
            "nct_id": "NCT03036124",
            "brief_title": "Dapagliflozin vs Placebo",
            "experimental_arms": ["Dapagliflozin 10 mg"],
            "comparator_arms": ["Matching Placebo"],
            "interventions": ["Drug: Dapagliflozin", "Drug: Placebo"],
        }
        self.assertEqual(
            rct_matcher.audit_study_arm_alignment(
                rct_placebo,
                intervention="Dapagliflozin",
                comparator="Placebo",
                project_mode="simulated_cohort",
            )["arm_alignment_status"],
            "exact_dual_arm",
        )

    def test_pico4d_nct03959501_ketone_surrogate_nct00842556_pk_exclusion_trap_and_carolina_flaura_positive_controls(self):
        """4D PICO-T regression: NCT03959501 (<=0.48), NCT00842556 (<=0.20), Exclusion Criteria trap, and CAROLINA/FLAURA 4D aligned."""
        # 1. NCT03959501: Dapagliflozin vs Sitagliptin in T2DM/Ketosis, 24-week blood ketone primary outcome
        nct03959501_raw = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT03959501",
                    "briefTitle": "Effects of Dapagliflozin Versus Sitagliptin on Blood Ketone Bodies in Type 2 Diabetes",
                },
                "statusModule": {"overallStatus": "COMPLETED"},
                "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE4"], "enrollmentInfo": {"count": 60}, "designInfo": {"allocation": "RANDOMIZED"}},
                "conditionsModule": {"conditions": ["Type 2 Diabetes Mellitus", "Ketosis"]},
                "armsInterventionsModule": {
                    "armGroups": [
                        {"label": "Dapagliflozin 5 mg", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin"]},
                        {"label": "Sitagliptin 50 mg", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Sitagliptin"]},
                    ],
                    "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Sitagliptin", "type": "DRUG"}],
                },
                "eligibilityModule": {
                    "healthyVolunteers": False,
                    "minimumAge": "20 Years",
                    "eligibilityCriteria": "Inclusion Criteria:\n* Uncomplicated Type 2 diabetes\nExclusion Criteria:\n* Chronic heart failure NYHA II-IV",
                },
                "outcomesModule": {
                    "primaryOutcomes": [{"measure": "Change from baseline in fasting blood ketone bodies (beta-hydroxybutyrate) at 24 weeks", "timeFrame": "24 weeks"}]
                },
            }
        }
        norm_ketone = rct_matcher._normalize_api_study(nct03959501_raw)
        audit_ketone = rct_matcher.audit_study_pico4d_alignment(
            norm_ketone,
            intervention="Dapagliflozin",
            comparator="Sitagliptin",
            target_condition="Type 2 Diabetes Heart Failure",
            target_outcome="Heart Failure Hospitalization or Cardiovascular Death",
            target_followup_days=1095.0,
            project_mode="user_cohort",
        )
        self.assertEqual(audit_ketone["arm_alignment_status"], "exact_dual_arm")
        self.assertEqual(audit_ketone["population_alignment"], "partial")
        self.assertEqual(audit_ketone["outcome_alignment"], "surrogate_or_mismatched")
        self.assertEqual(audit_ketone["followup_alignment"], "shorter_than_target")
        self.assertEqual(audit_ketone["overall_alignment_status"], "partial_protocol_reference")
        self.assertLessEqual(audit_ketone["effective_score_cap"], 0.48)
        self.assertFalse(audit_ketone["is_recommended"])

        with self.assertRaises(ContractError) as ctx_ketone:
            rct_matcher.verify_and_enrich_target_rct(
                norm_ketone,
                intervention="Dapagliflozin",
                comparator="Sitagliptin",
                target_condition="Type 2 Diabetes Heart Failure",
                target_outcome="Heart Failure Hospitalization or Cardiovascular Death",
                target_followup_days=1095.0,
                project_mode="user_cohort",
            )
        self.assertEqual(ctx_ketone.exception.code, "PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED")

        # 2. NCT00842556: Healthy-volunteer PK/DDI study of Dapagliflozin + Sitagliptin
        nct00842556_raw = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00842556",
                    "briefTitle": "Pharmacokinetic Drug-Drug Interaction Study Between Dapagliflozin and Sitagliptin in Healthy Subjects",
                },
                "statusModule": {"overallStatus": "COMPLETED"},
                "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE1"], "enrollmentInfo": {"count": 18}, "designInfo": {"allocation": "RANDOMIZED"}},
                "conditionsModule": {"conditions": ["Healthy"]},
                "armsInterventionsModule": {
                    "armGroups": [
                        {"label": "Dapagliflozin", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin"]},
                        {"label": "Sitagliptin", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Sitagliptin"]},
                    ],
                    "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Sitagliptin", "type": "DRUG"}],
                },
                "eligibilityModule": {
                    "healthyVolunteers": True,
                    "minimumAge": "18 Years",
                    "eligibilityCriteria": "Inclusion Criteria:\n* Healthy adult male and female subjects",
                },
                "outcomesModule": {
                    "primaryOutcomes": [{"measure": "Cmax and AUC(0-inf) of Dapagliflozin and Sitagliptin", "timeFrame": "Day 1 to Day 14"}]
                },
            }
        }
        norm_pk = rct_matcher._normalize_api_study(nct00842556_raw)
        audit_pk = rct_matcher.audit_study_pico4d_alignment(
            norm_pk,
            intervention="Dapagliflozin",
            comparator="Sitagliptin",
            target_condition="Type 2 Diabetes Heart Failure",
            target_outcome="Heart Failure Hospitalization or Cardiovascular Death",
            target_followup_days=540.0,
            project_mode="user_cohort",
        )
        self.assertEqual(audit_pk["design_suitability"], "non_efficacy_pk_or_healthy")
        self.assertEqual(audit_pk["overall_alignment_status"], "non_efficacy_excluded")
        self.assertLessEqual(audit_pk["effective_score_cap"], 0.20)
        self.assertFalse(audit_pk["is_recommended"])

        with self.assertRaises(ContractError) as ctx_pk:
            rct_matcher.verify_and_enrich_target_rct(
                norm_pk,
                intervention="Dapagliflozin",
                comparator="Sitagliptin",
                target_condition="Type 2 Diabetes Heart Failure",
                target_outcome="Heart Failure Hospitalization or Cardiovascular Death",
                target_followup_days=540.0,
                project_mode="user_cohort",
            )
        self.assertEqual(ctx_pk.exception.code, "PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED")

        # 3. Exclusion Criteria Trap Test: Heart failure mentioned ONLY under Exclusion Criteria must NOT match HF population!
        trap_raw = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT09998888", "briefTitle": "Dapagliflozin vs Sitagliptin Outcomes in T2DM"},
                "statusModule": {"overallStatus": "COMPLETED"},
                "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE3"], "enrollmentInfo": {"count": 1200}, "designInfo": {"allocation": "RANDOMIZED"}},
                "conditionsModule": {"conditions": ["Type 2 Diabetes Mellitus"]},
                "armsInterventionsModule": {
                    "armGroups": [
                        {"label": "Dapagliflozin 10 mg", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Dapagliflozin"]},
                        {"label": "Sitagliptin 100 mg", "type": "ACTIVE_COMPARATOR", "interventionNames": ["Drug: Sitagliptin"]},
                    ],
                    "interventions": [{"name": "Dapagliflozin", "type": "DRUG"}, {"name": "Sitagliptin", "type": "DRUG"}],
                },
                "eligibilityModule": {
                    "healthyVolunteers": False,
                    "eligibilityCriteria": "Inclusion Criteria:\n* Adults >= 40 years with Type 2 Diabetes\nExclusion Criteria:\n* Chronic heart failure NYHA II-IV or prior heart failure hospitalization",
                },
                "outcomesModule": {"primaryOutcomes": [{"measure": "Cardiovascular Death or MACE", "timeFrame": "36 months"}]},
            }
        }
        norm_trap = rct_matcher._normalize_api_study(trap_raw)
        audit_trap = rct_matcher.audit_study_pico4d_alignment(
            norm_trap,
            intervention="Dapagliflozin",
            comparator="Sitagliptin",
            target_condition="Type 2 Diabetes Heart Failure",
            target_outcome="Cardiovascular Death",
            target_followup_days=730.0,
            project_mode="user_cohort",
        )
        self.assertNotEqual(audit_trap["population_alignment"], "matched")
        self.assertEqual(audit_trap["population_alignment"], "partial")
        self.assertLessEqual(audit_trap["effective_score_cap"], 0.55)
        self.assertFalse(audit_trap["is_recommended"])

        # 4. Positive Controls: CAROLINA (NCT01243424) & FLAURA (NCT02296125) achieve 4-for-4 protocol_aligned
        carolina = next(r for r in rct_matcher.OFFLINE_CURATED_RCTS if r["nct_id"] == "NCT01243424")
        audit_carolina = rct_matcher.audit_study_pico4d_alignment(
            carolina,
            intervention="Linagliptin",
            comparator="Glimepiride",
            target_condition="Type 2 Diabetes Cardiovascular Risk",
            target_outcome="Cardiovascular Death Myocardial Infarction Stroke MACE",
            target_followup_days=730.0,
            project_mode="user_cohort",
        )
        self.assertEqual(audit_carolina["overall_alignment_status"], "protocol_aligned")
        self.assertTrue(audit_carolina["is_recommended"])
        self.assertEqual(audit_carolina["effective_score_cap"], 1.0)

        flaura = next(r for r in rct_matcher.OFFLINE_CURATED_RCTS if r["nct_id"] == "NCT02296125")
        audit_flaura = rct_matcher.audit_study_pico4d_alignment(
            flaura,
            intervention="Osimertinib",
            comparator="Gefitinib or Erlotinib",
            target_condition="EGFR-Mutated Non-Small Cell Lung Cancer",
            target_outcome="Progression-Free Survival and Overall Survival",
            target_followup_days=540.0,
            project_mode="user_cohort",
        )
        self.assertEqual(audit_flaura["arm_alignment_status"], "exact_dual_arm")
        self.assertEqual(audit_flaura["overall_alignment_status"], "protocol_aligned")
        self.assertTrue(audit_flaura["is_recommended"])
        self.assertEqual(audit_flaura["effective_score_cap"], 1.0)

    def test_round7_egfr_vs_ckd_disambiguation_and_symmetric_user_cohort_outcome_audit(self):
        """Round-7 Blocker 1 & 2: Oncology EGFR != CKD, custom 30-row EGFR/NSCLC CSV matches FLAURA cleanly, and user_cohort rejects explicit surrogate/mismatched outcome."""
        # --- Blocker 1: Exact 30-row custom CSV (`patient_key,arm_drug,time_days,os_event,age,ecog`) + note ---
        csv_lines = ["patient_key,arm_drug,time_days,os_event,age,ecog"]
        for i in range(30):
            drug = "Osimertinib" if i % 2 == 0 else "Gefitinib or Erlotinib"
            t_days = 180 + (i * 19) % 540
            ev = 1 if (i % 3 == 0) else 0
            age = 52 + ((i * 7) % 20)
            ecog = (i // 2) % 2
            csv_lines.append(f"ONC-CUST-{i + 1:03d},{drug},{t_days},{ev},{age},{ecog}")
        raw_csv = ("\n".join(csv_lines) + "\n").encode("utf-8")
        df_onc, _, _ = read_tabular_cohort_bytes(raw_csv, "custom_egfr_nsclc.csv")
        prof_onc = profile_cohort_dataframe(df_onc, filename="custom_egfr_nsclc.csv")
        map_onc = recommend_column_mapping(prof_onc)

        for note_str in (
            "EGFR-mutated non-small cell lung cancer, overall survival",
            "EGFR+ NSCLC, progression-free survival",
            "EGFR exon 19 L858R lung cancer, overall survival",
        ):
            interp_onc = llm_assist.interpret_cohort_profile(
                prof_onc,
                user_note=note_str,
                default_mapping=map_onc,
            )
            self.assertNotIn("Chronic Kidney Disease", interp_onc["condition_query_en"])
            self.assertNotIn("慢性肾脏病", interp_onc["domain_summary_zh"])
            self.assertIn("EGFR-Mutated Non-Small Cell Lung Cancer", interp_onc["condition_query_en"])

            flaura = next(r for r in rct_matcher.OFFLINE_CURATED_RCTS if r["nct_id"] == "NCT02296125")
            audit_custom = rct_matcher.audit_study_pico4d_alignment(
                flaura,
                intervention=interp_onc["intervention_query_en"],
                comparator=interp_onc["comparator_query_en"],
                target_condition=interp_onc["condition_query_en"],
                target_outcome=interp_onc["outcome_query_en"],
                target_followup_days=interp_onc["target_followup_days"],
                project_mode="user_cohort",
            )
            self.assertEqual(audit_custom["population_alignment"], "matched")
            self.assertEqual(audit_custom["outcome_alignment"], "matched")
            self.assertEqual(audit_custom["overall_alignment_status"], "protocol_aligned")
            self.assertEqual(audit_custom["effective_score_cap"], 1.0)
            self.assertTrue(audit_custom["is_recommended"])

        # Genuine CKD strings still classify as CKD, while sample_cardiorenal_sglt2_dpp4 (with baseline `egfr_ml_min`) resolves T2DM + HF cleanly
        ckd_concepts = llm_assist.extract_whitelisted_clinical_concepts("Patients with chronic kidney disease and diabetic nephropathy (CKD)")
        self.assertIn("Chronic Kidney Disease", ckd_concepts["conditions_en"])
        egfr_only_concepts = llm_assist.extract_whitelisted_clinical_concepts("EGFR-mutated lung carcinoma with EGFR TKI")
        self.assertNotIn("Chronic Kidney Disease", egfr_only_concepts["conditions_en"])
        self.assertIn("EGFR-Mutated Non-Small Cell Lung Cancer", egfr_only_concepts["conditions_en"])

        fname_cr, raw_cr = generate_sample_cohort_csv_bytes("sample_cardiorenal_sglt2_dpp4")
        df_cr, _, _ = read_tabular_cohort_bytes(raw_cr, fname_cr)
        prof_cr = profile_cohort_dataframe(df_cr, filename=fname_cr)
        interp_cr = llm_assist.interpret_cohort_profile(prof_cr, sample_id="cardiorenal_sglt2i")
        self.assertEqual(interp_cr["condition_query_en"], "Type 2 Diabetes Heart Failure")
        self.assertNotIn("Chronic Kidney Disease", interp_cr["condition_query_en"])

        # --- Blocker 2(a): Symmetric explicit target_outcome audit in `user_cohort` mode ---
        flaura = next(r for r in rct_matcher.OFFLINE_CURATED_RCTS if r["nct_id"] == "NCT02296125")
        for pmode in ("user_cohort", "simulated_cohort"):
            audit_ketone_vs_flaura = rct_matcher.audit_study_pico4d_alignment(
                flaura,
                intervention="Osimertinib",
                comparator="Gefitinib or Erlotinib",
                target_condition="EGFR-mutated Non-Small Cell Lung Cancer",
                target_outcome="Blood ketone change",
                target_followup_days=540.0,
                project_mode=pmode,
            )
            self.assertEqual(audit_ketone_vs_flaura["outcome_alignment"], "surrogate_or_mismatched")
            self.assertNotEqual(audit_ketone_vs_flaura["overall_alignment_status"], "protocol_aligned")
            self.assertLessEqual(audit_ketone_vs_flaura["effective_score_cap"], 0.48)
            self.assertFalse(audit_ketone_vs_flaura["is_recommended"])
            with self.assertRaises(ContractError) as ctx_b2:
                rct_matcher.verify_and_enrich_target_rct(
                    {**flaura, "match_score": 0.96},
                    intervention="Osimertinib",
                    comparator="Gefitinib or Erlotinib",
                    target_condition="EGFR-mutated Non-Small Cell Lung Cancer",
                    target_outcome="Blood ketone change",
                    target_followup_days=540.0,
                    project_mode=pmode,
                )
            self.assertEqual(ctx_b2.exception.code, "PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED")

        # Unlisted explicit functional outcome (`6-minute walk distance`) also mismatches FLAURA in `user_cohort` mode
        audit_unlisted = rct_matcher.audit_study_pico4d_alignment(
            flaura,
            intervention="Osimertinib",
            comparator="Gefitinib or Erlotinib",
            target_condition="EGFR-mutated Non-Small Cell Lung Cancer",
            target_outcome="6-minute walk distance",
            project_mode="user_cohort",
        )
        self.assertEqual(audit_unlisted["outcome_alignment"], "surrogate_or_mismatched")
        self.assertLessEqual(audit_unlisted["effective_score_cap"], 0.48)

    def test_round8_short_explicit_outcomes_fail_closed_against_flaura(self):
        """Round-8: ORR/CRP/CEA/LDH/PSA never certify FLAURA; bare OS/PFS still align; short tokens use word boundaries."""
        flaura = next(r for r in rct_matcher.OFFLINE_CURATED_RCTS if r["nct_id"] == "NCT02296125")
        pico = dict(
            intervention="Osimertinib",
            comparator="Gefitinib or Erlotinib",
            target_condition="EGFR-mutated Non-Small Cell Lung Cancer",
            target_followup_days=540.0,
            project_mode="user_cohort",
        )
        clin_os, surr_os = rct_matcher.classify_outcome_domains("OS")
        self.assertIn("mortality_os", clin_os)
        self.assertNotIn("tumor_response_rate", surr_os)
        clin_pfs, _ = rct_matcher.classify_outcome_domains("PFS")
        self.assertIn("oncology_progression", clin_pfs)
        _, surr_bleed = rct_matcher.classify_outcome_domains("Corrected calcium and major hemorrhage")
        self.assertNotIn("tumor_response_rate", surr_bleed)
        _, surr_orr = rct_matcher.classify_outcome_domains("ORR")
        self.assertEqual(surr_orr, {"tumor_response_rate"})

        for good_out in ("OS", "PFS"):
            audit_ok = rct_matcher.audit_study_pico4d_alignment(flaura, target_outcome=good_out, **pico)
            self.assertEqual(audit_ok["outcome_alignment"], "matched", good_out)
            self.assertEqual(audit_ok["overall_alignment_status"], "protocol_aligned", good_out)
            self.assertTrue(audit_ok["is_recommended"], good_out)
            self.assertEqual(audit_ok["effective_score_cap"], 1.0, good_out)
            enriched = rct_matcher.verify_and_enrich_target_rct(
                {**flaura, "match_score": 0.99},
                target_outcome=good_out,
                **pico,
            )
            self.assertEqual(enriched["match_score"], 0.99)
            self.assertTrue(enriched["is_recommended"])

        for bad_out in ("ORR", "CRP", "CEA", "LDH", "PSA"):
            audit_bad = rct_matcher.audit_study_pico4d_alignment(flaura, target_outcome=bad_out, **pico)
            self.assertEqual(audit_bad["outcome_alignment"], "surrogate_or_mismatched", bad_out)
            self.assertNotEqual(audit_bad["overall_alignment_status"], "protocol_aligned", bad_out)
            self.assertFalse(audit_bad["is_recommended"], bad_out)
            self.assertLessEqual(audit_bad["effective_score_cap"], 0.48, bad_out)
            with self.assertRaises(ContractError) as ctx_bad:
                rct_matcher.verify_and_enrich_target_rct(
                    {**flaura, "match_score": 0.99},
                    target_outcome=bad_out,
                    **pico,
                )
            self.assertEqual(ctx_bad.exception.code, "PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED")

        # ORR listed only as a secondary tumor-response surrogate is at most secondary_only, never protocol_aligned.
        flaura_sec_orr = {
            **flaura,
            "secondary_outcomes": [
                {"measure": "Overall Survival (OS)", "timeFrame": "Up to 36 months"},
                {"measure": "Objective Response Rate (ORR) per RECIST 1.1", "timeFrame": "18 months"},
            ],
        }
        audit_sec = rct_matcher.audit_study_pico4d_alignment(flaura_sec_orr, target_outcome="ORR", **pico)
        self.assertEqual(audit_sec["outcome_alignment"], "secondary_only")
        self.assertLessEqual(audit_sec["effective_score_cap"], 0.65)
        self.assertNotEqual(audit_sec["overall_alignment_status"], "protocol_aligned")
        self.assertFalse(audit_sec["is_recommended"])

        # Unlisted 3-letter token must not substring-match inside a longer word.
        decoy = {
            **flaura,
            "primary_outcomes": [{"measure": "Corrected hemorrhage control at week 12", "timeFrame": "12 weeks"}],
            "secondary_outcomes": [],
        }
        audit_decoy = rct_matcher.audit_study_pico4d_alignment(decoy, target_outcome="ZZZ", **pico)
        self.assertEqual(audit_decoy["outcome_alignment"], "surrogate_or_mismatched")
        self.assertFalse(audit_decoy["is_recommended"])
        hit = {
            **decoy,
            "primary_outcomes": [{"measure": "Change in ZZZ biomarker", "timeFrame": "12 weeks"}],
        }
        audit_hit = rct_matcher.audit_study_pico4d_alignment(hit, target_outcome="ZZZ", **pico)
        self.assertEqual(audit_hit["outcome_alignment"], "matched")


class CohortIngestAndDualModePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def run_cli(self, subcommand: str, args: list[str]):
        return subprocess.run(
            [sys.executable, "-m", "emulatrx_tte", subcommand, *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1"},
        )

    def test_bom_tsv_constant_collinear_protection_missingness_and_zero_competing_graceful(self):
        rng = np.random.default_rng(909)
        n = 320
        age = rng.normal(62.0, 9.0, size=n)
        stage = rng.choice(["Stage_II", "Stage_III", "Stage_IV"], size=n)
        arm = np.where(rng.uniform(size=n) < 0.5, "Drug_Active", "Drug_Comparator")
        fu_years = rng.uniform(0.2, 2.5, size=n)
        # Single-event dataset (zero competing events!) + constant column + collinear duplicate column + missing values
        status = np.where(rng.uniform(size=n) < 0.28, "Primary_Relapse", "Alive_Free")
        df = pd.DataFrame(
            {
                "hosp_id": [f"HOSP-{1000 + i}" for i in range(n)],
                "age_yr": [np.nan if i < 6 else round(float(age[i]), 1) for i in range(n)],
                "age_duplicate": [np.nan if i < 6 else round(float(age[i]) * 2.0 + 5.0, 2) for i in range(n)],
                "constant_flag": [1] * n,
                "constant_cat": ["SAME_CENTER"] * n,
                "tumor_stage": stage,
                "regimen": arm,
                "followup_years": [round(float(y), 3) for y in fu_years],
                "endpoint": status,
            }
        )
        # Write as UTF-8-BOM tab-delimited TSV
        tsv_bytes = ("\ufeff" + df.to_csv(sep="\t", index=False)).encode("utf-8")
        tsv_path = self.base / "real_style.tsv"
        tsv_path.write_bytes(tsv_bytes)

        parsed_df, delim, enc = read_tabular_cohort_bytes(tsv_bytes, "real_style.tsv")
        self.assertEqual(delim, "\t")
        self.assertEqual(enc, "utf-8-sig")
        self.assertEqual(len(parsed_df), n)

        mapping = {
            "id_column": "hosp_id",
            "study_title": "鲁棒性测试：常数列剔除、年份换算与无竞争事件退化",
            "treatment_column": "regimen",
            "comparator_value": "Drug_Comparator",
            "active_value": "Drug_Active",
            "time_column": "followup_years",
            "time_unit": "years",
            "outcome_column": "endpoint",
            "primary_event_values": ["Primary_Relapse"],
            "competing_event_values": [],  # Explicitly empty competing events!
            "censor_values": ["Alive_Free"],
            "covariate_columns": ["age_yr", "age_duplicate", "constant_flag", "constant_cat", "tumor_stage"],
            "eligibility_criteria": [
                {"id": "inc_age_50", "kind": "inclusion", "variable": "age_yr", "operator": ">=", "value": 50}
            ],
            "horizon_days": [180.0, 365.0],
            "administrative_end_day": 950.0,
            "weight_method": "overlap",
        }
        out_run = self.base / "user_robust_out"
        res, code = run_user_cohort_pipeline(tsv_path, mapping, out_dir=out_run, bootstrap_iterations=25)
        self.assertEqual(code, 0, res)
        self.assertEqual(res["source_mode"], SOURCE_MODE_USER_COHORT)

        ingest_man = json.loads((out_run / "ingest_manifest.json").read_text(encoding="utf-8"))
        dropped_cols = {d["column"] for d in ingest_man["dropped_constant_or_collinear_covariates"]}
        self.assertIn("constant_flag", dropped_cols)
        self.assertIn("constant_cat", dropped_cols)
        self.assertIn("age_duplicate", dropped_cols)
        self.assertIsNotNone(ingest_man["zero_competing_events_notice"])
        self.assertAlmostEqual(ingest_man["time_conversion"]["multiplier_to_days"], 365.25)

        est = json.loads((out_run / "estimates.json").read_text(encoding="utf-8"))
        self.assertEqual(est["source_mode"], SOURCE_MODE_USER_COHORT)
        self.assertIn("真实队列目标试验模拟（真项目）分析报告", est["report_badge"])
        self.assertEqual(len(est["causal_assumptions"]), 4)
        self.assertEqual(est["co_presented_comparison_table"][0]["cif_competing_active"], 0.0)
        audit_md = (out_run / "audit_report.md").read_text(encoding="utf-8")
        self.assertIn("真实队列目标试验模拟（真项目）分析报告", audit_md)
        self.assertNotIn("synthetic-only benchmark cohort", audit_md)

    def test_cli_run_user_cohort_and_run_simulated_and_exit_code_3_gate_block(self):
        fname, raw = generate_sample_cohort_csv_bytes("oncology_io_nsclc")
        cohort_path = self.base / fname
        cohort_path.write_bytes(raw)
        df, _, _ = read_tabular_cohort_bytes(raw, fname)
        prof = profile_cohort_dataframe(df, filename=fname)
        mapping = recommend_column_mapping(prof)
        map_path = self.base / "mapping.json"
        map_path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")

        cli_user_out = self.base / "cli_user_out"
        r_uc = self.run_cli(
            "run-user-cohort",
            ["--cohort", str(cohort_path), "--mapping", str(map_path), "--bootstrap-iterations", "20", "--out", str(cli_user_out)],
        )
        self.assertEqual(r_uc.returncode, 0, r_uc.stderr)
        self.assertTrue((cli_user_out / "estimates.json").is_file())

        # Gate block (exit code 3) on impossible positivity_epsilon=0.49 in user_cohort
        strict_map = dict(mapping, positivity_epsilon=0.49)
        strict_map_path = self.base / "strict_map.json"
        strict_map_path.write_text(json.dumps(strict_map), encoding="utf-8")
        blk_out = self.base / "cli_user_blocked"
        r_blk = self.run_cli(
            "run-user-cohort",
            ["--cohort", str(cohort_path), "--mapping", str(strict_map_path), "--bootstrap-iterations", "20", "--out", str(blk_out)],
        )
        self.assertEqual(r_blk.returncode, 3, r_blk.stderr)
        self.assertFalse((blk_out / "weighted_cohort.csv").exists())
        self.assertFalse((blk_out / "estimates.json").exists())

        # Verify estimate --source-mode user_cohort rejects missing ingest_manifest.json
        ingest_only = self.base / "ing_only"
        ingest_user_cohort(cohort_path, mapping, out_dir=ingest_only)
        from emulatrx_tte import diagnose_cohort
        diag_only = self.base / "diag_only"
        diagnose_cohort(ingest_only / "normalized_protocol.json", ingest_only / "standardized_cohort.csv", weight_method="overlap", out_dir=diag_only)
        with self.assertRaises(ContractError) as ctx_no_man:
            estimate_competing_risks(
                ingest_only / "normalized_protocol.json",
                diagnostics_dir=diag_only,
                seed=42,
                bootstrap_iterations=20,
                out_dir=self.base / "est_no_man",
                source_mode=SOURCE_MODE_USER_COHORT,
            )
        self.assertEqual(ctx_no_man.exception.code, "MISSING_USER_COHORT_INGEST_MANIFEST")

        # CLI run-simulated end-to-end
        sim_proto, sim_dgp = build_simulated_protocol_and_dgp(
            sim_config={"confounding_strength": "medium", "effect_direction": "protective_strong"}
        )
        sp_path = self.base / "sim_proto.json"
        sd_path = self.base / "sim_dgp.json"
        sp_path.write_text(json.dumps(sim_proto), encoding="utf-8")
        sd_path.write_text(json.dumps(sim_dgp), encoding="utf-8")
        cli_sim_out = self.base / "cli_sim_out"
        r_sim = self.run_cli(
            "run-simulated",
            [
                "--protocol",
                str(sp_path),
                "--dgp-params",
                str(sd_path),
                "--seed",
                "42",
                "--cases",
                "700",
                "--bootstrap-iterations",
                "20",
                "--out",
                str(cli_sim_out),
            ],
        )
        self.assertEqual(r_sim.returncode, 0, r_sim.stderr)
        sim_est = json.loads((cli_sim_out / "estimates.json").read_text(encoding="utf-8"))
        self.assertEqual(sim_est["source_mode"], SOURCE_MODE_SIMULATED_COHORT)
        self.assertIn("目标试验模拟（无数据模拟预演项目）可行性报告", sim_est["report_badge"])
        self.assertGreaterEqual(len(sim_est["simulation_recovery_table"]), 3)

    def test_live_double_dummy_placebo_and_investigator_choice_or_arms(self):
        """Live CT.gov fixtures: CAROLINA double-dummy placebos and FLAURA gefitinib-or-erlotinib stay exact dual-arm."""
        fixture_dir = Path(__file__).resolve().parent / "fixtures" / "ctgov"

        def load(nct_id: str) -> dict:
            return json.loads((fixture_dir / f"{nct_id}.json").read_text(encoding="utf-8"))

        active = set(rct_matcher.normalize_intervention_spec("linagliptin (linagliptin, glimepiride placebo)")["drug_ids"]) - {"placebo_soc"}
        self.assertEqual(active, {"linagliptin"})
        active = set(rct_matcher.normalize_intervention_spec("glimepiride 1-4 mg QD (glimepiride, linagliptin placebo)")["drug_ids"]) - {"placebo_soc"}
        self.assertEqual(active, {"glimepiride"})
        for placebo_label in (
            "Drug: glimepiride placebo",
            "Other: Placebo matching Gefitinib",
            "linagliptin 5 mg placebo",
            "sitagliptin-matching placebo",
            "gefitinib/erlotinib placebo",
            "placebo matching osimertinib",
            "placebo to match glimepiride",
            "placebo for sitagliptin",
            "placebo (glimepiride)",
            "利格列汀安慰剂",
            "匹配格列美脲的安慰剂",
        ):
            spec = rct_matcher.normalize_intervention_spec(placebo_label)
            self.assertEqual(set(spec["drug_ids"]) - {"placebo_soc"}, set(), placebo_label)
            self.assertIn("placebo_soc", spec["drug_ids"], placebo_label)
        self.assertEqual(
            set(rct_matcher.normalize_intervention_spec("Placebo + Metformin")["drug_ids"]) - {"placebo_soc"},
            {"metformin"},
        )
        combo = rct_matcher.normalize_intervention_spec("Sitagliptin + Glimepiride")
        self.assertEqual(combo["regimen_mode"], "all")
        self.assertEqual(set(combo["drug_ids"]), {"sitagliptin", "glimepiride"})
        choice = rct_matcher.normalize_intervention_spec("Gefitinib or Erlotinib")
        self.assertEqual(set(choice["drug_ids"]), {"gefitinib", "erlotinib"})
        self.assertEqual(
            set(rct_matcher.normalize_intervention_spec("AZD9291 + placebo Erlotinib / placebo Gefitinib")["drug_ids"]) - {"placebo_soc"},
            {"osimertinib"},
        )
        self.assertEqual(
            set(rct_matcher.normalize_intervention_spec("Erlotinib or Gefitinib + placebo AZD9291")["drug_ids"]) - {"placebo_soc"},
            {"erlotinib", "gefitinib"},
        )
        for code_name, canonical in (
            ("AZD9291", "osimertinib"),
            ("AZD-9291", "osimertinib"),
            ("BI-1356", "linagliptin"),
            ("MK-0431", "sitagliptin"),
            ("BMS-512148", "dapagliflozin"),
            ("BI-10773", "empagliflozin"),
            ("MK-3475", "pembrolizumab"),
        ):
            self.assertIn(canonical, rct_matcher.normalize_intervention_spec(code_name)["drug_ids"], code_name)

        carolina = rct_matcher._normalize_api_study(load("NCT01243424"))
        self.assertIn("linagliptin, glimepiride placebo", carolina["experimental_arms"][0])
        self.assertIn("linagliptin placebo", carolina["comparator_arms"][0])
        audit_carolina = rct_matcher.audit_study_pico4d_alignment(
            carolina,
            intervention="利格列汀",
            comparator="格列美脲",
            target_condition="Type 2 Diabetes Cardiovascular Risk",
            target_outcome="Cardiovascular Death Myocardial Infarction Stroke MACE",
            target_followup_days=730.0,
            project_mode="user_cohort",
        )
        self.assertEqual(audit_carolina["arm_alignment_status"], "exact_dual_arm")
        self.assertEqual(audit_carolina["population_alignment"], "matched")
        self.assertEqual(audit_carolina["outcome_alignment"], "matched")
        self.assertEqual(audit_carolina["followup_alignment"], "adequate")
        self.assertEqual(audit_carolina["overall_alignment_status"], "protocol_aligned")
        self.assertTrue(audit_carolina["is_recommended"])
        scored_carolina = rct_matcher._score_and_audit_study(
            carolina,
            ["Linagliptin", "Glimepiride", "Diabetes", "Cardiovascular"],
            intervention="Linagliptin",
            comparator="Glimepiride",
            target_condition="Type 2 Diabetes Cardiovascular Risk",
            target_outcome="Cardiovascular Death Myocardial Infarction Stroke MACE",
            target_followup_days=730.0,
            project_mode="user_cohort",
        )
        self.assertGreaterEqual(scored_carolina["match_score"], 0.85)

        flaura = rct_matcher._normalize_api_study(load("NCT02296125"))
        self.assertIn("Placebo Erlotinib", " ".join(flaura["experimental_arms"]))
        self.assertIn("Gefitinib", flaura["comparator_arms"][0])
        self.assertIn("Erlotinib", flaura["comparator_arms"][0])
        for comparator in ("Gefitinib or Erlotinib", "吉非替尼/厄洛替尼"):
            audit_flaura = rct_matcher.audit_study_pico4d_alignment(
                flaura,
                intervention="Osimertinib",
                comparator=comparator,
                target_condition="EGFR-Mutated Non-Small Cell Lung Cancer",
                target_outcome="Progression-Free Survival and Overall Survival",
                target_followup_days=540.0,
                project_mode="user_cohort",
            )
            self.assertEqual(audit_flaura["arm_alignment_status"], "exact_dual_arm", comparator)
            self.assertEqual(audit_flaura["overall_alignment_status"], "protocol_aligned", comparator)
            self.assertTrue(audit_flaura["is_recommended"], comparator)
        for narrow in ("Gefitinib", "吉非替尼"):
            audit_narrow = rct_matcher.audit_study_pico4d_alignment(
                flaura,
                intervention="Osimertinib",
                comparator=narrow,
                target_condition="EGFR-Mutated Non-Small Cell Lung Cancer",
                target_outcome="Progression-Free Survival and Overall Survival",
                target_followup_days=540.0,
                project_mode="user_cohort",
            )
            self.assertEqual(audit_narrow["arm_alignment_status"], "class_dual_arm", narrow)
            self.assertLessEqual(audit_narrow["effective_score_cap"], 0.72, narrow)
            self.assertFalse(audit_narrow["is_recommended"], narrow)
            self.assertTrue(audit_narrow["requires_mismatch_confirmation"], narrow)
            with self.assertRaises(ContractError) as narrow_ctx:
                rct_matcher.verify_and_enrich_target_rct(
                    flaura,
                    intervention="Osimertinib",
                    comparator=narrow,
                    target_condition="EGFR-Mutated Non-Small Cell Lung Cancer",
                    target_outcome="Progression-Free Survival and Overall Survival",
                    target_followup_days=540.0,
                    project_mode="user_cohort",
                )
            self.assertEqual(narrow_ctx.exception.code, "ARM_MISMATCH_CONFIRMATION_REQUIRED", narrow)
        curated_flaura = next(r for r in rct_matcher.OFFLINE_CURATED_RCTS if r["nct_id"] == "NCT02296125")
        curated_carolina = next(r for r in rct_matcher.OFFLINE_CURATED_RCTS if r["nct_id"] == "NCT01243424")
        self.assertEqual(
            set(rct_matcher.normalize_intervention_spec(curated_flaura["comparator_arms"][0])["drug_ids"]),
            {"gefitinib", "erlotinib"},
        )
        self.assertEqual(
            set(rct_matcher.normalize_intervention_spec(curated_flaura["experimental_arms"][0])["drug_ids"]),
            {"osimertinib"},
        )
        self.assertEqual(
            rct_matcher.audit_study_arm_alignment(
                curated_flaura,
                intervention="Osimertinib",
                comparator="Gefitinib",
                project_mode="user_cohort",
            )["arm_alignment_status"],
            "class_dual_arm",
        )
        self.assertEqual(
            rct_matcher.audit_study_arm_alignment(
                curated_flaura,
                intervention="Osimertinib",
                comparator="Gefitinib or Erlotinib",
                project_mode="user_cohort",
            )["arm_alignment_status"],
            "exact_dual_arm",
        )
        from emulatrx_tte.sample_cohorts import SAMPLE_COHORT_SPECS

        self.assertEqual(SAMPLE_COHORT_SPECS["nsclc_egfr_flaura"]["comparator_intervention_en"], "Gefitinib or Erlotinib")
        offline_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("EMULATRX_LLM_API_KEY", "OPENAI_API_KEY", "EMULATRX_LLM_API_KEY_FILE")
        }
        offline_env["EMULATRX_OFFLINE_RCT_CATALOG"] = "1"
        with mock.patch.dict(os.environ, offline_env, clear=True):
            for sample_id, study in (
                ("t2dm_cv_carolina", curated_carolina),
                ("nsclc_egfr_flaura", curated_flaura),
            ):
                spec = SAMPLE_COHORT_SPECS[sample_id]
                live_study = carolina if sample_id == "t2dm_cv_carolina" else flaura
                for source_study in (live_study, study):
                    aligned = rct_matcher.audit_study_pico4d_alignment(
                        source_study,
                        intervention=spec["active_intervention_en"],
                        comparator=spec["comparator_intervention_en"],
                        target_condition=spec["condition_en"],
                        target_outcome=spec["outcome_en"],
                        target_followup_days=spec["target_followup_days"],
                        project_mode="user_cohort",
                    )
                    self.assertEqual(aligned["arm_alignment_status"], "exact_dual_arm", sample_id)
                    self.assertEqual(aligned["overall_alignment_status"], "protocol_aligned", sample_id)
                    self.assertTrue(aligned["is_recommended"], sample_id)
                offline_res = search_candidate_rcts(
                    f"{spec['active_intervention_en']} {spec['comparator_intervention_en']}",
                    condition=spec["condition_en"],
                    intervention=spec["active_intervention_en"],
                    comparator=spec["comparator_intervention_en"],
                    outcome=spec["outcome_en"],
                    target_followup_days=spec["target_followup_days"],
                    project_mode="user_cohort",
                    max_results=10,
                )
                self.assertEqual(offline_res["source"], "offline_curated_catalog_fallback", sample_id)
                expected_nct = "NCT01243424" if sample_id == "t2dm_cv_carolina" else "NCT02296125"
                card = next(c for c in offline_res["candidates"] if c["nct_id"] == expected_nct)
                self.assertEqual(card["arm_alignment_status"], "exact_dual_arm", sample_id)
                self.assertEqual(card["overall_alignment_status"], "protocol_aligned", sample_id)
                self.assertTrue(card["is_recommended"], sample_id)
        scored_flaura = rct_matcher._score_and_audit_study(
            flaura,
            ["Osimertinib", "Gefitinib", "Erlotinib", "EGFR", "Lung"],
            intervention="Osimertinib",
            comparator="Gefitinib or Erlotinib",
            target_condition="EGFR-Mutated Non-Small Cell Lung Cancer",
            target_outcome="Progression-Free Survival and Overall Survival",
            target_followup_days=540.0,
            project_mode="user_cohort",
        )
        self.assertGreaterEqual(scored_flaura["match_score"], 0.85)
        self.assertEqual(scored_flaura["arm_alignment_status"], "exact_dual_arm")
        with self.assertRaises(ContractError) as orr_ctx:
            rct_matcher.verify_and_enrich_target_rct(
                flaura,
                intervention="Osimertinib",
                comparator="Gefitinib or Erlotinib",
                target_condition="EGFR-Mutated Non-Small Cell Lung Cancer",
                target_outcome="ORR",
                target_followup_days=540.0,
                project_mode="user_cohort",
            )
        self.assertEqual(orr_ctx.exception.code, "PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED")

        blocked = rct_matcher.audit_study_arm_alignment(
            {
                "experimental_arms": ["Dapagliflozin 10 mg"],
                "comparator_arms": ["Sitagliptin 100 mg + Glimepiride 4 mg"],
            },
            intervention="Dapagliflozin",
            comparator="Sitagliptin",
            project_mode="user_cohort",
        )
        self.assertNotEqual(blocked["arm_alignment_status"], "exact_dual_arm")
        self.assertEqual(blocked["arm_alignment_status"], "single_arm_or_comparator_mismatch")
        self.assertLessEqual(blocked["arm_alignment_score_cap"], 0.45)

        ketone = rct_matcher.audit_study_pico4d_alignment(
            rct_matcher._normalize_api_study(load("NCT03959501")),
            intervention="Dapagliflozin",
            comparator="Sitagliptin",
            target_condition="Type 2 Diabetes Heart Failure",
            target_outcome="Heart Failure Hospitalization or Cardiovascular Death",
            target_followup_days=1095.0,
            project_mode="user_cohort",
        )
        self.assertLessEqual(ketone["effective_score_cap"], 0.48)
        self.assertFalse(ketone["is_recommended"])
        pk = rct_matcher.audit_study_pico4d_alignment(
            rct_matcher._normalize_api_study(load("NCT00842556")),
            intervention="Dapagliflozin",
            comparator="Sitagliptin",
            target_condition="Type 2 Diabetes Heart Failure",
            target_outcome="Heart Failure Hospitalization or Cardiovascular Death",
            target_followup_days=540.0,
            project_mode="user_cohort",
        )
        self.assertLessEqual(pk["effective_score_cap"], 0.20)
        self.assertEqual(pk["overall_alignment_status"], "non_efficacy_excluded")
        same_class = rct_matcher.audit_study_arm_alignment(
            rct_matcher._normalize_api_study(load("NCT06642272")),
            intervention="Dapagliflozin",
            comparator="Sitagliptin",
            project_mode="user_cohort",
        )
        self.assertEqual(same_class["arm_alignment_status"], "single_arm_or_comparator_mismatch")
        self.assertLessEqual(same_class["arm_alignment_score_cap"], 0.45)


if __name__ == "__main__":
    unittest.main()
