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

            # Assert NOT A SINGLE patient ID from the entire CSV appears in the LLM request body
            for sid in all_subject_ids:
                self.assertNotIn(sid, sent_text, f"Privacy leak: {sid} found in outbound LLM payload!")
            self.assertNotIn("Hosp-2023-", sent_text)
            self.assertNotIn("ONC-880", sent_text)
            self.assertIn("[REDACTED_ID]", sent_text)

            # Parse inner user message JSON and verify only schema + aggregate stats exist
            outer = json.loads(sent_text)
            inner_payload = json.loads(outer["messages"][1]["content"])
            self.assertEqual(inner_payload["privacy_contract"], "schema_and_aggregates_only_zero_individual_rows")
            col_names_sent = {c["name"] for c in inner_payload["columns"]}
            self.assertNotIn(id_col, col_names_sent)
            for c in inner_payload["columns"]:
                if "distinct_categories" in c:
                    self.assertLessEqual(len(c["distinct_categories"]), 20)

        # Also verify active runtime blocker raises PRIVACY_VIOLATION_BLOCKED if a raw ID slips in
        with self.assertRaises(ContractError) as ctx:
            assert_llm_payload_privacy_safe(b'{"leak": "Hosp-2023-0042"}')
        self.assertEqual(ctx.exception.code, "PRIVACY_VIOLATION_BLOCKED")

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


if __name__ == "__main__":
    unittest.main()
