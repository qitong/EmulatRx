"""Unit and synthetic benchmark tests for P1 bilateral grace-period CCW (Issue #17)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import emulation_utils as eu
from emulatrx_tte import (
    CCW_PROTOCOL_VERSION,
    ContractError,
    PROHIBITED_CLAIMS,
    SOURCE_MODE_SYNTHETIC_ONLY,
    clone_and_censor,
    generate_synthetic_ccw_cohort,
    run_ccw_synthetic_pipeline,
    run_grace_period_ccw,
    validate_ccw_protocol,
    validate_ccw_protocol_dict,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CCW_PROTOCOL = ROOT / "examples" / "tte_grace_period_ccw_v1.json"
CCW_SCHEMA_PATH = ROOT / "schemas" / "tte-ccw-v1.schema.json"


class TTEGracePeriodCCWTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.protocol = json.loads(EXAMPLE_CCW_PROTOCOL.read_text(encoding="utf-8"))

    def run_cli(self, subcommand: str, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "emulatrx_tte", subcommand, *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            },
        )

    def write_json(self, obj: dict, name: str) -> Path:
        path = self.base / name
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def test_schema_and_example_protocol_validation_and_reproducibility(self) -> None:
        schema = json.loads(CCW_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(schema["additionalProperties"])
        for req in (
            "protocol_id",
            "protocol_version",
            "source_mode",
            "grace_period_spec",
            "covariates",
            "ipcw_spec",
            "diagnostic_gates",
        ):
            self.assertIn(req, schema["required"])

        out1 = self.base / "val1"
        res1 = self.run_cli("validate-ccw", ["--protocol", str(EXAMPLE_CCW_PROTOCOL), "--out", str(out1)])
        self.assertEqual(res1.returncode, 0, res1.stderr)
        norm1 = out1 / "normalized_protocol.json"
        rep1 = out1 / "protocol_check_report.md"
        man1 = json.loads((out1 / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(norm1.is_file())
        self.assertTrue(rep1.is_file())
        self.assertEqual(man1["protocol_version"], CCW_PROTOCOL_VERSION)
        self.assertEqual(man1["source_mode"], SOURCE_MODE_SYNTHETIC_ONLY)
        for rel, digest in man1["output_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((out1 / rel).read_bytes()).hexdigest())

        out2 = self.base / "val2"
        res2 = self.run_cli("validate-ccw", ["--protocol", str(norm1), "--out", str(out2)])
        self.assertEqual(res2.returncode, 0, res2.stderr)
        self.assertEqual(norm1.read_bytes(), (out2 / "normalized_protocol.json").read_bytes())

    def test_contract_exit_code_2_non_syn_id_divisibility_and_time_travel_guards(self) -> None:
        # 1. Non-divisible grace_period_days
        bad_div = copy.deepcopy(self.protocol)
        bad_div["grace_period_spec"]["grace_period_days"] = 65
        p_div = self.write_json(bad_div, "bad_div.json")
        r_div = self.run_cli("validate-ccw", ["--protocol", str(p_div), "--out", str(self.base / "out_div")])
        self.assertEqual(r_div.returncode, 2)
        self.assertIn("NON_DIVISIBLE_TIME_GRID", r_div.stderr)
        self.assertFalse((self.base / "out_div").exists())

        # 2. Non-divisible evaluation horizon
        bad_hor = copy.deepcopy(self.protocol)
        bad_hor["grace_period_spec"]["evaluation_horizons_days"] = [60, 125, 180]
        p_hor = self.write_json(bad_hor, "bad_hor.json")
        r_hor = self.run_cli("validate-ccw", ["--protocol", str(p_hor), "--out", str(self.base / "out_hor")])
        self.assertEqual(r_hor.returncode, 2)
        self.assertIn("NON_DIVISIBLE_TIME_GRID", r_hor.stderr)

        # 3. Non-synthetic source_mode
        bad_src = copy.deepcopy(self.protocol)
        bad_src["source_mode"] = "real_world_ehr"
        p_src = self.write_json(bad_src, "bad_src.json")
        r_src = self.run_cli("validate-ccw", ["--protocol", str(p_src), "--out", str(self.base / "out_src")])
        self.assertEqual(r_src.returncode, 2)
        self.assertIn("UNSUPPORTED_SOURCE_MODE", r_src.stderr)

        # 4. Post-baseline covariate leak (offset > 0)
        bad_base_cov = copy.deepcopy(self.protocol)
        bad_base_cov["covariates"]["baseline_covariates"][0]["assessed_day_offset"] = 5
        p_bc = self.write_json(bad_base_cov, "bad_bc.json")
        r_bc = self.run_cli("validate-ccw", ["--protocol", str(p_bc), "--out", str(self.base / "out_bc")])
        self.assertEqual(r_bc.returncode, 2)
        self.assertIn("POST_BASELINE_COVARIATE_LEAK", r_bc.stderr)

        # 5. Pre-period time-varying covariate time travel (offset >= 0 relative to period decision start)
        bad_tv = copy.deepcopy(self.protocol)
        bad_tv["covariates"]["time_varying_covariates"] = [
            {"name": "x_tv_lab", "type": "continuous", "offset_from_period_start_days": 0}
        ]
        p_tv = self.write_json(bad_tv, "bad_tv.json")
        r_tv = self.run_cli("validate-ccw", ["--protocol", str(p_tv), "--out", str(self.base / "out_tv")])
        self.assertEqual(r_tv.returncode, 2)
        self.assertIn("TIME_VARYING_COVARIATE_TIME_TRAVEL", r_tv.stderr)

        # 6. Non-SYN-* subject_id rejected with ContractError (exit_code == 2)
        gen = generate_synthetic_ccw_cohort(self.protocol, seed=7, n_cases=40)
        bad_df = gen["cohort_df"].copy()
        bad_df.loc[3, "subject_id"] = "REAL-000004"
        with self.assertRaises(ContractError) as ctx:
            clone_and_censor(bad_df, self.protocol)
        self.assertEqual(ctx.exception.exit_code, 2)
        self.assertEqual(ctx.exception.code, "NON_SYNTHETIC_SUBJECT_ID")

        # 7. Negative initiation_day or non-positive duration_day rejected
        neg_init_df = gen["cohort_df"].copy()
        neg_init_df.loc[0, "initiation_day"] = -2.0
        with self.assertRaises(ContractError) as ctx_init:
            run_grace_period_ccw(self.protocol, neg_init_df, seed=7, bootstrap_iterations=10)
        self.assertEqual(ctx_init.exception.exit_code, 2)
        self.assertEqual(ctx_init.exception.code, "INVALID_INITIATION_DAY")

    def test_three_subject_bilateral_clone_shared_early_events_and_censoring_exactness(self) -> None:
        micro_df = pd.DataFrame(
            [
                # Subject 1: dies/events at day 20 without initiating -> shared early event in BOTH clones
                {
                    "subject_id": "SYN-000001",
                    "initiation_day": np.nan,
                    "duration_day": 20.0,
                    "event_code": 1,
                    "x_age_std": 0.2,
                    "x_biomarker_std": -0.1,
                    "x_comorbidity_bin": 1,
                    "x_high_risk_bin": 0,
                },
                # Subject 2: initiates at day 30 (period 3), primary event at day 120
                {
                    "subject_id": "SYN-000002",
                    "initiation_day": 30.0,
                    "duration_day": 120.0,
                    "event_code": 1,
                    "x_age_std": 0.5,
                    "x_biomarker_std": 0.4,
                    "x_comorbidity_bin": 0,
                    "x_high_risk_bin": 1,
                },
                # Subject 3: survives past day 60 without initiating (censored day 120)
                {
                    "subject_id": "SYN-000003",
                    "initiation_day": np.nan,
                    "duration_day": 120.0,
                    "event_code": 0,
                    "x_age_std": -0.4,
                    "x_biomarker_std": 0.1,
                    "x_comorbidity_bin": 0,
                    "x_high_risk_bin": 0,
                },
            ]
        )

        clones_df, attrition = clone_and_censor(micro_df, self.protocol)
        self.assertEqual(len(clones_df), 6)
        self.assertEqual(attrition["shared_early_events_within_grace_uninitiated"]["total_events"], 1)
        self.assertEqual(attrition["shared_early_events_within_grace_uninitiated"]["primary_events"], 1)

        by_key = {
            (row["subject_id"], row["clone_strategy"]): row
            for _, row in clones_df.iterrows()
        }

        # SYN-000001: day-20 uninitiated event counted in BOTH Strategy 0 and Strategy 1
        s1_c0 = by_key[("SYN-000001", "never_initiate")]
        s1_c1 = by_key[("SYN-000001", "initiate_within_grace")]
        self.assertEqual((s1_c0["T_clone"], s1_c0["event_clone"], s1_c0["artificial_censored"]), (20.0, 1, 0))
        self.assertEqual((s1_c1["T_clone"], s1_c1["event_clone"], s1_c1["artificial_censored"]), (20.0, 1, 0))

        # SYN-000002: day-30 initiator -> Strategy 0 censored at start of period 3 (T_clone=20, event=0); Strategy 1 uncensored
        s2_c0 = by_key[("SYN-000002", "never_initiate")]
        s2_c1 = by_key[("SYN-000002", "initiate_within_grace")]
        self.assertEqual((s2_c0["T_clone"], s2_c0["event_clone"], s2_c0["artificial_censored"], int(s2_c0["censor_period"])), (20.0, 0, 1, 3))
        self.assertEqual((s2_c1["T_clone"], s2_c1["event_clone"], s2_c1["artificial_censored"]), (120.0, 1, 0))

        # SYN-000003: day-120 uninitiated survivor -> Strategy 1 censored at t=60 (period 7); Strategy 0 uncensored
        s3_c0 = by_key[("SYN-000003", "never_initiate")]
        s3_c1 = by_key[("SYN-000003", "initiate_within_grace")]
        self.assertEqual((s3_c1["T_clone"], s3_c1["event_clone"], s3_c1["artificial_censored"], int(s3_c1["censor_period"])), (60.0, 0, 1, 7))
        self.assertEqual((s3_c0["T_clone"], s3_c0["event_clone"], s3_c0["artificial_censored"]), (120.0, 0, 0))

    def test_null_causal_effect_immortal_time_debiasing_and_cluster_bootstrap_coverage(self) -> None:
        dgp_null = {"true_log_hr_event1": 0.0, "true_log_hr_event2": 0.0}
        gen = generate_synthetic_ccw_cohort(self.protocol, seed=42, n_cases=2500, dgp_params=dgp_null)

        # Ground-truth counterfactual RD is identically 0.0 under strict null
        for h_row in gen["dgp_truth"]["horizon_truth_table"]:
            self.assertEqual(h_row["true_rd_event1"], 0.0)
            self.assertEqual(h_row["true_rr_cif_event1"], 1.0)

        res, code = run_grace_period_ccw(
            self.protocol,
            gen["cohort_df"],
            seed=42,
            bootstrap_iterations=200,
        )
        self.assertEqual(code, 0)
        est_payload = res["estimates"]
        self.assertEqual(est_payload["variance_method"], "subject_cluster_bootstrap_with_ipcw_refit")

        for h_key in ("60", "120", "180"):
            nv = est_payload["naive_immortal_time_comparison"]["horizons"][h_key]
            est = est_payload["horizons"][h_key]
            # Naive post-hoc static classification exhibits strong false-positive protective bias (< -0.04)
            self.assertLess(nv["naive_rd"], -0.04, msg=f"Horizon {h_key}: {nv}")
            self.assertLess(nv["naive_within_grace_rd"], -0.04, msg=f"Horizon {h_key}: {nv}")
            # Bilateral arm-specific IPCW-CCW removes immortal-time bias and 95% cluster bootstrap CI covers 0.0
            self.assertLess(abs(est["ccw_rd_event1"]), 0.03, msg=f"Horizon {h_key}: {est}")
            self.assertLessEqual(est["ccw_rd_event1_ci_lower"], 0.0, msg=f"Horizon {h_key}: {est}")
            self.assertGreaterEqual(est["ccw_rd_event1_ci_upper"], 0.0, msg=f"Horizon {h_key}: {est}")
            self.assertAlmostEqual(
                nv["immortal_time_bias_rd_inflation"],
                nv["naive_rd"] - est["ccw_rd_event1"],
                places=6,
            )

    def test_protective_effect_recovery_byte_identical_reproducibility_and_html_report(self) -> None:
        out_a = self.base / "prot_run_a"
        out_b = self.base / "prot_run_b"

        res_a = self.run_cli(
            "run-ccw-synthetic",
            [
                "--protocol",
                str(EXAMPLE_CCW_PROTOCOL),
                "--seed",
                "108",
                "--cases",
                "2500",
                "--bootstrap-iterations",
                "150",
                "--out",
                str(out_a),
            ],
        )
        res_b = self.run_cli(
            "run-ccw-synthetic",
            [
                "--protocol",
                str(EXAMPLE_CCW_PROTOCOL),
                "--seed",
                "108",
                "--cases",
                "2500",
                "--bootstrap-iterations",
                "150",
                "--out",
                str(out_b),
            ],
        )
        self.assertEqual(res_a.returncode, 0, res_a.stderr)
        self.assertEqual(res_b.returncode, 0, res_b.stderr)

        man_a = json.loads((out_a / "run_manifest.json").read_text(encoding="utf-8"))
        man_b = json.loads((out_b / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(man_a["output_paths"], man_b["output_paths"])
        self.assertEqual(man_a["output_sha256"], man_b["output_sha256"])

        expected_files = {
            "PURPOSE.txt",
            "audit_report.md",
            "balance_post_grace.csv",
            "ccw_attrition.json",
            "ccw_diagnostics.json",
            "ccw_estimates.json",
            "ccw_preview_report.html",
            "cohort.csv",
            "curves/ccw_cif_curves.csv",
            "curves/naive_cif_curves.csv",
            "dgp_truth.json",
            "normalized_protocol.json",
            "protocol_check_report.md",
        }
        self.assertEqual(set(man_a["output_paths"]), expected_files)
        for rel in expected_files:
            bytes_a = (out_a / rel).read_bytes()
            bytes_b = (out_b / rel).read_bytes()
            self.assertEqual(bytes_a, bytes_b, f"Mismatch in {rel}")
            self.assertEqual(man_a["output_sha256"][rel], hashlib.sha256(bytes_a).hexdigest())

        truth = json.loads((out_a / "dgp_truth.json").read_text(encoding="utf-8"))
        estimates = json.loads((out_a / "ccw_estimates.json").read_text(encoding="utf-8"))
        self.assertEqual(estimates["source_mode"], SOURCE_MODE_SYNTHETIC_ONLY)
        self.assertEqual(estimates["prohibited_claims"], list(PROHIBITED_CLAIMS))

        true_rd_180 = truth["horizons"]["180"]["true_rd_event1"]
        est_180 = estimates["horizons"]["180"]
        self.assertLess(true_rd_180, -0.05)
        self.assertLess(est_180["ccw_rd_event1_ci_upper"], 0.0)
        self.assertLessEqual(est_180["ccw_rd_event1_ci_lower"], true_rd_180)
        self.assertGreaterEqual(est_180["ccw_rd_event1_ci_upper"], true_rd_180)

        html_text = (out_a / "ccw_preview_report.html").read_text(encoding="utf-8")
        self.assertIn("朴素事后分组假阳性曲线", html_text)
        self.assertIn("双侧宽限期分臂 IPCW-CCW 校正曲线", html_text)
        self.assertIn("合成真值 (Ground Truth", html_text)
        self.assertIn("synthetic_only", html_text)
        self.assertIn("非真实患者数据", html_text)
        self.assertNotIn("http://", html_text)
        self.assertNotIn("https://", html_text)
        self.assertIn("truth_cif_initiate_within_grace", truth)
        self.assertIn("truth_cif_never_initiate", truth)
        self.assertEqual(len(truth["truth_cif_initiate_within_grace"]), 19)
        self.assertEqual(len(truth["truth_cif_never_initiate"]), 19)
        step_paths = re.findall(r"<path class='cif-step' d='([^']+)'", html_text)
        truth_paths = re.findall(r"<path class='cif-truth-step' d='([^']+)'", html_text)
        self.assertEqual(len(step_paths), 4)
        self.assertEqual(len(truth_paths), 4)
        for p_cmd in [*step_paths, *truth_paths]:
            self.assertIn(" H", p_cmd)
            self.assertIn(" V", p_cmd)
            self.assertNotIn(" L", p_cmd)

    def test_time_varying_covariate_ipcw_and_admin_censoring_truncation_options(self) -> None:
        tv_proto = copy.deepcopy(self.protocol)
        tv_proto["protocol_id"] = "ccw_tv_admin_ipcw_v1"
        tv_proto["covariates"]["time_varying_covariates"] = [
            {
                "name": "x_tv_severity",
                "type": "continuous",
                "offset_from_period_start_days": -1,
                "description": "Pre-period synthetic biomarker assessed 1 day before period decision start",
            }
        ]
        tv_proto["ipcw_spec"] = {
            "truncate_quantiles": [0.02, 0.98],
            "include_admin_ipcw": True,
        }
        out_tv = self.base / "tv_run"
        res, code = run_ccw_synthetic_pipeline(
            tv_proto,
            seed=99,
            n_cases=1800,
            bootstrap_iterations=40,
            out_dir=out_tv,
        )
        self.assertEqual(code, 0, res)
        self.assertTrue((out_tv / "person_period_covariates.csv").is_file())
        self.assertTrue((out_tv / "ccw_estimates.json").is_file())

        # Inject time-travel measurement day >= (k-1)*Delta into person_period_covariates.csv -> exit_code 2
        pp_df = pd.read_csv(out_tv / "person_period_covariates.csv", dtype={"subject_id": str})
        cohort_df = pd.read_csv(out_tv / "cohort.csv", dtype={"subject_id": str})
        pp_df.loc[0, "x_tv_severity__assessed_day"] = 0.0  # Period 1 decision start is 0.0 -> not strictly prior!
        with self.assertRaises(ContractError) as ctx:
            run_grace_period_ccw(
                tv_proto,
                cohort_df,
                seed=99,
                bootstrap_iterations=10,
                person_period_covariates=pp_df,
            )
        self.assertEqual(ctx.exception.exit_code, 2)
        self.assertEqual(ctx.exception.code, "TIME_VARYING_COVARIATE_TIME_TRAVEL")

    def test_positivity_and_gate_violation_blocks_with_exit_code_3_without_estimates(self) -> None:
        strict_proto = copy.deepcopy(self.protocol)
        strict_proto["diagnostic_gates"]["max_stabilized_weight"] = 1.25
        strict_proto["diagnostic_gates"]["min_ess_ratio"] = 0.95
        p_strict = self.write_json(strict_proto, "strict_gate.json")

        extreme_dgp = {
            "initiation_covariate_slopes": [2.2, -2.0, 1.8, 1.6],
        }
        p_dgp = self.write_json(extreme_dgp, "extreme_dgp.json")

        out_blocked = self.base / "blocked_run"
        res = self.run_cli(
            "run-ccw-synthetic",
            [
                "--protocol",
                str(p_strict),
                "--seed",
                "42",
                "--cases",
                "600",
                "--bootstrap-iterations",
                "30",
                "--dgp-params",
                str(p_dgp),
                "--out",
                str(out_blocked),
            ],
        )
        self.assertEqual(res.returncode, 3, res.stderr)
        self.assertIn("DIAGNOSTIC_GATE_BLOCKED", res.stderr)
        self.assertTrue((out_blocked / "ccw_diagnostics.json").is_file())
        self.assertTrue((out_blocked / "ccw_attrition.json").is_file())
        self.assertTrue((out_blocked / "run_manifest.json").is_file())
        # Critical Fail-Closed invariant: NEVER produce ccw_estimates.json, curves, or preview HTML when blocked
        self.assertFalse((out_blocked / "ccw_estimates.json").exists())
        self.assertFalse((out_blocked / "ccw_preview_report.html").exists())
        self.assertFalse((out_blocked / "curves").exists())

        diag = json.loads((out_blocked / "ccw_diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(diag["gate_status"], "blocked")
        failure_codes = {f["code"] for f in diag["gate_failures"]}
        self.assertTrue(failure_codes & {"MAX_WEIGHT_EXCEEDED", "ESS_RATIO_BELOW_MIN"})

    def test_legacy_adjust_immortal_time_bias_raises_runtime_error_with_migration_guide(self) -> None:
        dummy_df = pd.DataFrame(
            {
                "id": ["SYN-000001", "SYN-000002"],
                "t_trt": [10.0, np.nan],
                "event": [0, 1],
                "t_end": [50.0, 30.0],
                "x": [0.1, -0.2],
            }
        )
        with self.assertRaises(RuntimeError) as ctx:
            eu.adjust_immortal_time_bias(dummy_df, "id", "t_trt", "event", "t_end", ["x"])
        msg = str(ctx.exception)
        self.assertIn("emulatrx_tte.ccw", msg)
        self.assertIn("grace-period", msg)
        self.assertIn("pooled", msg)

    def test_time_varying_confounder_true_null_probe_eliminates_strategy1_bias(self) -> None:
        """Regression test for Review Blocking #1: Strategy 1 IPCW must condition on pre-period X_tv."""
        tv_null_proto = {
            "protocol_id": "ccw_tv_true_null_probe",
            "protocol_version": "tte-ccw-v1",
            "source_mode": "synthetic_only",
            "grace_period_spec": {
                "grace_period_days": 60,
                "period_length_days": 10,
                "max_followup_days": 120,
                "evaluation_horizons_days": [60, 120],
            },
            "covariates": {
                "baseline_covariates": [
                    {"name": "x_base_noise", "type": "continuous", "assessed_day_offset": 0}
                ],
                "time_varying_covariates": [
                    {"name": "x_tv_bin", "type": "binary", "offset_from_period_start_days": -1}
                ],
            },
            "ipcw_spec": {"truncate_quantiles": None, "include_admin_ipcw": False},
            "diagnostic_gates": {
                "min_ess_ratio": 0.25,
                "max_post_grace_smd": 0.10,
                "max_stabilized_weight": 12.0,
                "min_uncensored_subjects_per_clone": 50,
            },
        }

        def _simulate_tv_null_cohort(seed: int, concentrated_period1: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
            rng = np.random.default_rng(seed)
            n = 4000
            k_total = 12
            m_g = 6
            delta = 10.0
            x_base = np.round(rng.standard_normal(n), 6)
            # Pre-period binary confounder driving BOTH initiation (0.85 vs 0.15) and 120d risk (0.50 vs 0.10)
            z_tv = (rng.random(n) < 0.50).astype(np.int64)

            p_ev_step = np.where(z_tv == 1, 1.0 - (1.0 - 0.50) ** (1.0 / k_total), 1.0 - (1.0 - 0.10) ** (1.0 / k_total))
            if concentrated_period1:
                p_init_step = np.where(z_tv == 1, 0.85, 0.15)
            else:
                p_init_step = np.where(z_tv == 1, 1.0 - (1.0 - 0.85) ** (1.0 / m_g), 1.0 - (1.0 - 0.15) ** (1.0 / m_g))

            active = np.ones(n, dtype=bool)
            initiated = np.zeros(n, dtype=bool)
            init_day = np.full(n, np.nan, dtype=np.float64)
            dur_day = np.full(n, 120.0, dtype=np.float64)
            ev_code = np.zeros(n, dtype=np.int64)
            exit_k = np.full(n, k_total, dtype=np.int64)

            for k in range(1, k_total + 1):
                if k == 1 or (not concentrated_period1 and k <= m_g):
                    can_init = active & (~initiated)
                    do_init = can_init & (rng.random(n) < p_init_step)
                    initiated[do_init] = True
                    init_day[do_init] = (k - 1) * delta + 2.0

                # Strict null causal effect: hazard depends ONLY on z_tv, NEVER on initiated
                ev_now = active & (rng.random(n) < p_ev_step)
                dur_day[ev_now] = (k - 1) * delta + 7.0
                ev_code[ev_now] = 1
                exit_k[ev_now] = k
                active[ev_now] = False

            cohort_df = pd.DataFrame(
                {
                    "subject_id": [f"SYN-{i + 1:06d}" for i in range(n)],
                    "initiation_day": init_day,
                    "duration_day": dur_day,
                    "event_code": ev_code,
                    "x_base_noise": x_base,
                }
            )
            pp_records = []
            for i in range(n):
                sid = f"SYN-{i + 1:06d}"
                zv = int(z_tv[i])
                for k in range(1, int(exit_k[i]) + 1):
                    pp_records.append(
                        {
                            "subject_id": sid,
                            "period_idx": k,
                            "x_tv_bin": zv,
                            "x_tv_bin__assessed_day": (k - 1) * delta - 1.0,
                        }
                    )
            return cohort_df, pd.DataFrame(pp_records)

        for mode_label, conc in (("concentrated_k1", True), ("multi_period_grace", False)):
            cohort_df, pp_df = _simulate_tv_null_cohort(seed=2026 if conc else 2027, concentrated_period1=conc)
            res, code = run_grace_period_ccw(
                tv_null_proto,
                cohort_df,
                seed=31,
                bootstrap_iterations=80,
                person_period_covariates=pp_df,
            )
            self.assertEqual(code, 0, msg=f"{mode_label} blocked unexpectedly: {res.get('gate_failures')}")
            self.assertEqual(res["gate_status"], "passed")
            est_120 = res["estimates"]["horizons"]["120"]
            self.assertLess(abs(est_120["ccw_rd_event1"]), 0.035, msg=f"{mode_label}: {est_120}")
            self.assertLessEqual(est_120["ccw_rd_event1_ci_lower"], 0.0, msg=f"{mode_label}: {est_120}")
            self.assertGreaterEqual(est_120["ccw_rd_event1_ci_upper"], 0.0, msg=f"{mode_label}: {est_120}")

    def test_positivity_collapse_masked_by_extreme_quantile_truncation_blocks_fail_closed(self) -> None:
        """Regression test for Review Blocking #2: truncation cannot hide positivity failure or over-clip."""
        # Invalid quantile bounds rejected with ContractError (exit code 2)
        for bad_q in ([0.60, 0.40], [0.50, 0.50], [-0.05, 0.95], [0.05, 1.05]):
            bad_p = copy.deepcopy(self.protocol)
            bad_p["ipcw_spec"]["truncate_quantiles"] = bad_q
            with self.assertRaises(ContractError) as ctx_q:
                validate_ccw_protocol_dict(bad_p)
            self.assertEqual(ctx_q.exception.exit_code, 2)

        # 5,000-subject cohort with 0.98 / 0.02 initiation probabilities by binary baseline risk stratum
        rng = np.random.default_rng(5050)
        n = 5000
        stratum = (rng.random(n) < 0.50).astype(np.int64)
        p_init = np.where(stratum == 1, 0.98, 0.02)
        did_init = rng.random(n) < p_init
        init_day = np.where(did_init, rng.choice([5.0, 15.0, 25.0, 35.0, 45.0, 55.0], size=n), np.nan)
        ev_u = rng.random(n)
        dur_day = np.where(ev_u < 0.25, rng.uniform(65.0, 175.0, size=n), 180.0)
        ev_code = np.where(ev_u < 0.25, 1, 0)

        collapse_df = pd.DataFrame(
            {
                "subject_id": [f"SYN-{i + 1:06d}" for i in range(n)],
                "initiation_day": init_day,
                "duration_day": np.round(dur_day, 4),
                "event_code": ev_code,
                "x_age_std": np.round(rng.standard_normal(n), 6),
                "x_biomarker_std": np.round(rng.standard_normal(n), 6),
                "x_comorbidity_bin": (rng.random(n) < 0.4).astype(np.int64),
                "x_high_risk_bin": stratum,
            }
        )

        for trunc_spec, out_sub in (
            (None, "collapse_none"),
            ([0.49, 0.51], "collapse_4951"),
            ([0.01, 0.99], "collapse_0199"),
        ):
            proto_c = copy.deepcopy(self.protocol)
            proto_c["ipcw_spec"]["truncate_quantiles"] = trunc_spec
            proto_c["diagnostic_gates"]["max_stabilized_weight"] = 2.0
            proto_c["diagnostic_gates"]["min_ess_ratio"] = 0.05
            proto_c["diagnostic_gates"]["max_post_grace_smd"] = 2.0
            proto_c["diagnostic_gates"]["min_uncensored_subjects_per_clone"] = 10

            out_dir = self.base / out_sub
            res, code = run_grace_period_ccw(
                proto_c,
                collapse_df,
                seed=42,
                bootstrap_iterations=20,
                out_dir=out_dir,
            )
            self.assertEqual(code, 3, msg=f"trunc={trunc_spec} should fail closed with code 3: {res}")
            self.assertEqual(res["gate_status"], "blocked")
            self.assertFalse((out_dir / "ccw_estimates.json").exists())
            diag = json.loads((out_dir / "ccw_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(diag["gate_status"], "blocked")
            self.assertGreater(diag["pre_truncation_max_stabilized_weight"], 20.0)
            codes = {f["code"] for f in diag["gate_failures"]}
            self.assertIn("PRE_TRUNCATION_WEIGHT_EXCEEDED", codes)
            if trunc_spec == [0.49, 0.51]:
                self.assertIn("EXCESSIVE_WEIGHT_TRUNCATION", codes)
                self.assertGreater(diag["truncated_weight_fraction"]["max_across_clones"], 0.80)

        # Verify CLI run-ccw-synthetic also fails closed (exit code 3, no ccw_estimates.json) under [0.49, 0.51]
        cli_proto = copy.deepcopy(self.protocol)
        cli_proto["ipcw_spec"]["truncate_quantiles"] = [0.49, 0.51]
        cli_proto["diagnostic_gates"]["max_stabilized_weight"] = 2.0
        p_cli = self.write_json(cli_proto, "cli_collapse_4951.json")
        p_dgp_collapse = self.write_json(
            {"initiation_covariate_slopes": [0.0, 0.0, 0.0, 7.8], "grace_initiation_intercept": -3.9},
            "dgp_collapse.json",
        )
        out_cli = self.base / "cli_collapse_out"
        cli_res = self.run_cli(
            "run-ccw-synthetic",
            [
                "--protocol",
                str(p_cli),
                "--seed",
                "42",
                "--cases",
                "5000",
                "--bootstrap-iterations",
                "20",
                "--dgp-params",
                str(p_dgp_collapse),
                "--out",
                str(out_cli),
            ],
        )
        self.assertEqual(cli_res.returncode, 3, cli_res.stderr)
        self.assertFalse((out_cli / "ccw_estimates.json").exists())
        cli_diag = json.loads((out_cli / "ccw_diagnostics.json").read_text(encoding="utf-8"))
        self.assertGreater(cli_diag["pre_truncation_max_stabilized_weight"], 20.0)


if __name__ == "__main__":
    unittest.main()
