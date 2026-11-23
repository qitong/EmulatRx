"""Analytical benchmarks, fail-closed gate tests, determinism, and contract checks for Issue #9."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from lifelines import CoxPHFitter
import numpy as np
import pandas as pd

from emulatrx_tte import (
    PROHIBITED_CLAIMS,
    SOURCE_MODE_SYNTHETIC_ONLY,
    ContractError,
    GateBlockedError,
    compute_evalue,
    compute_reverse_km_left_limit,
    diagnose_cohort,
    estimate_competing_risks,
    evaluate_step_curve,
    fit_fine_gray_shr,
    fit_weighted_cause_specific_cox,
    generate_synthetic_cohort,
    run_synthetic_pipeline,
    weighted_aalen_johansen_cif,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PROTOCOL = ROOT / "examples" / "tte_competing_risk_v1.json"
SKILL_MD = ROOT / "skills" / "emulatrx-tte" / "SKILL.md"
CONTRACT_DOC = ROOT / "docs" / "tte-cli-and-agent-contract-v1.md"
OVERVIEW_DOC = ROOT / "docs" / "FEATURE_API_OVERVIEW.md"


def _closed_form_cif(lam1: float, lam2: float, t: float) -> tuple[float, float, float]:
    """Return (true_cif1, true_cif2, biased_one_minus_km1) for constant cause-specific hazards."""
    tot = lam1 + lam2
    prob_any = 1.0 - math.exp(-tot * t)
    cif1 = (lam1 / tot) * prob_any
    cif2 = (lam2 / tot) * prob_any
    biased_km1 = 1.0 - math.exp(-lam1 * t)
    return cif1, cif2, biased_km1


def _geskus_reference_fine_gray_log_shr(
    durations: np.ndarray,
    events: np.ndarray,
    arm: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Independent reference solver evaluating exact Fine & Gray (1999) partial score equation."""
    ev1_times = np.unique(durations[events == 1])
    g_ev1 = compute_reverse_km_left_limit(durations, events, weights, ev1_times)
    g_ti = np.ones_like(durations, dtype=np.float64)
    comp = events == 2
    if np.any(comp):
        g_ti[comp] = compute_reverse_km_left_limit(durations, events, weights, durations[comp])

    def score(beta: float) -> float:
        r = np.exp(arm * beta)
        u = 0.0
        for m_idx, tm in enumerate(ev1_times):
            fail_m = (durations == tm) & (events == 1)
            d_m = weights[fail_m]
            D_m = float(np.sum(d_m))
            w_risk = np.where(
                durations >= tm,
                weights,
                np.where((durations < tm) & comp, weights * (g_ev1[m_idx] / g_ti), 0.0),
            )
            s0 = float(np.sum(w_risk * r))
            s1 = float(np.sum(w_risk * r * arm))
            u += float(np.sum(d_m * arm[fail_m])) - D_m * (s1 / s0)
        return u

    lo, hi = -4.0, 4.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class TTECompetingRisksBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.protocol = json.loads(EXAMPLE_PROTOCOL.read_text(encoding="utf-8"))

    def run_cli(self, subcommand: str, args: list[str]):
        return subprocess.run(
            [sys.executable, "-m", "emulatrx_tte", subcommand, *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def write_json(self, obj: dict, name: str) -> Path:
        path = self.base / name
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def test_double_exponential_analytical_cif_naive_km_bias_and_cox_fine_gray_alignment(self):
        # Pure double-exponential competing-risks DGP (zero outcome covariate slope -> exact marginal closed-form CIFs)
        lam1_ctrl = 0.0030
        lam2_ctrl = 0.0026
        cshr1_true = 0.60
        cshr2_true = 1.00
        lam1_act = lam1_ctrl * cshr1_true
        lam2_act = lam2_ctrl * cshr2_true

        dgp = {
            "propensity_intercepts": [0.0, -0.10],
            "propensity_slopes": [[0.0, 0.0, 0.0, 0.0], [0.45, -0.35, 0.30, 0.25]],
            "baseline_hazard_primary": lam1_ctrl,
            "baseline_hazard_competing": lam2_ctrl,
            "random_censor_hazard": 0.0004,
            "pre_landmark_hazard_multiplier": 1.0,
            "log_cshr_primary_by_arm": [0.0, math.log(cshr1_true)],
            "log_cshr_competing_by_arm": [0.0, math.log(cshr2_true)],
            "outcome_covariate_log_hr_primary": [0.0, 0.0, 0.0, 0.0],
            "outcome_covariate_log_hr_competing": [0.0, 0.0, 0.0, 0.0],
        }
        dgp_path = self.write_json(dgp, "dgp_analytical.json")
        out_dir = self.base / "analytical_run"

        res, exit_code = run_synthetic_pipeline(
            EXAMPLE_PROTOCOL,
            seed=202611,
            cases=4000,
            weight_method="overlap",
            bootstrap_iterations=120,
            out_dir=out_dir,
            dgp_params_path=dgp_path,
        )
        self.assertEqual(exit_code, 0, res)

        estimates = json.loads((out_dir / "estimates.json").read_text(encoding="utf-8"))
        self.assertEqual(estimates["source_mode"], SOURCE_MODE_SYNTHETIC_ONLY)
        self.assertEqual(estimates["prohibited_claims"], list(PROHIBITED_CLAIMS))
        self.assertEqual(estimates["bootstrap_config"]["baseline_propensity_weights"], "fixed")
        self.assertFalse(estimates["bootstrap_config"]["censoring_ipcw_reestimated_in_bootstrap"])
        self.assertIn("conditional on estimated Stage-1 baseline weights", estimates["bootstrap_config"]["variance_scope_note"])

        # 1. Verify probability conservation CIF_1 + CIF_2 + S == 1.0 across every exported curve row
        for arm_id in ("strategy_control", "strategy_active"):
            c_df = pd.read_csv(out_dir / "curves" / f"cif_{arm_id}.csv")
            prob_sum = (
                c_df["cif_primary"].to_numpy()
                + c_df["cif_competing"].to_numpy()
                + c_df["overall_survival"].to_numpy()
            )
            np.testing.assert_allclose(prob_sum, 1.0, atol=1e-9)

        # 2. Verify naive 1-KM upward bias vs weighted Aalen-Johansen accuracy at t=365 days
        t_check = 365.0
        true_c1_ctrl, true_c2_ctrl, true_naive_ctrl = _closed_form_cif(lam1_ctrl, lam2_ctrl, t_check)
        true_c1_act, true_c2_act, true_naive_act = _closed_form_cif(lam1_act, lam2_act, t_check)

        ctrl_365 = estimates["arm_cif_summaries"]["strategy_control"]["horizons"]["365"]
        act_365 = estimates["arm_cif_summaries"]["strategy_active"]["horizons"]["365"]

        # Naive 1-KM must show pronounced expected upward bias (> +0.08 in control, > +0.05 in active)
        self.assertGreater(ctrl_365["naive_km_primary_biased_ref"] - ctrl_365["cif_primary"], 0.08)
        self.assertGreater(act_365["naive_km_primary_biased_ref"] - act_365["cif_primary"], 0.05)
        self.assertAlmostEqual(ctrl_365["naive_km_primary_biased_ref"], true_naive_ctrl, delta=0.03)
        self.assertAlmostEqual(act_365["naive_km_primary_biased_ref"], true_naive_act, delta=0.03)

        # Weighted Aalen-Johansen CIF_1 and CIF_2 converge accurately to closed-form values
        self.assertAlmostEqual(ctrl_365["cif_primary"], true_c1_ctrl, delta=0.025)
        self.assertAlmostEqual(ctrl_365["cif_competing"], true_c2_ctrl, delta=0.025)
        self.assertAlmostEqual(act_365["cif_primary"], true_c1_act, delta=0.025)
        self.assertAlmostEqual(act_365["cif_competing"], true_c2_act, delta=0.025)

        # Bootstrap 95% CIs cover analytical RD_1(365) and RR_CIF_1(365)
        contrast = estimates["contrasts"]["strategy_active__vs__strategy_control"]
        h365 = contrast["horizons"]["365"]
        true_rd1 = true_c1_act - true_c1_ctrl
        true_rr1 = true_c1_act / true_c1_ctrl
        self.assertLessEqual(h365["rd_primary_ci_lower"], true_rd1)
        self.assertGreaterEqual(h365["rd_primary_ci_upper"], true_rd1)
        self.assertLessEqual(h365["rr_cif_primary_ci_lower"], true_rr1)
        self.assertGreaterEqual(h365["rr_cif_primary_ci_upper"], true_rr1)

        # 3. Cross-validate csHR_1, csHR_2, and Fine-Gray sHR against lifelines & independent Geskus solver
        self.assertAlmostEqual(contrast["cause_specific_cox_primary"]["point"], cshr1_true, delta=0.08)
        self.assertAlmostEqual(contrast["cause_specific_cox_competing"]["point"], cshr2_true, delta=0.10)

        wdf = pd.read_csv(out_dir / "weighted_cohort.csv")
        t_vec = wdf["duration_day"].to_numpy(dtype=float)
        e_vec = wdf["event_code"].to_numpy(dtype=int)
        z_vec = (wdf["treatment_arm"] == "strategy_active").astype(int).to_numpy()
        norm_w = wdf["normalized_weight"].to_numpy(dtype=float)
        w_vec = 0.5 * len(wdf) * norm_w

        cph1 = CoxPHFitter()
        cph1.fit(
            pd.DataFrame({"T": t_vec, "E": (e_vec == 1).astype(int), "Z": z_vec, "W": w_vec}),
            duration_col="T",
            event_col="E",
            weights_col="W",
            robust=True,
        )
        self.assertAlmostEqual(
            contrast["cause_specific_cox_primary"]["point"],
            float(np.exp(cph1.params_["Z"])),
            places=4,
        )
        self.assertAlmostEqual(
            contrast["cause_specific_cox_primary"]["se_robust_log_cshr"],
            float(cph1.standard_errors_["Z"]),
            places=2,
        )

        ref_log_shr = _geskus_reference_fine_gray_log_shr(t_vec, e_vec, z_vec.astype(float), w_vec)
        self.assertAlmostEqual(
            contrast["fine_gray_shr_primary"]["point"],
            math.exp(ref_log_shr),
            places=5,
        )

    def test_high_competing_mortality_drives_shr_below_one_with_mandatory_warnings(self):
        # Counterexample: true primary csHR_1 == 1.0 (zero biological effect on primary event),
        # while competing mortality hazard is ~3.3x higher in active arm (log_cshr_2 = 1.20).
        dgp_counter = {
            "propensity_intercepts": [0.0, 0.0],
            "propensity_slopes": [[0.0, 0.0, 0.0, 0.0], [0.30, -0.25, 0.20, 0.20]],
            "baseline_hazard_primary": 0.0025,
            "baseline_hazard_competing": 0.0045,
            "random_censor_hazard": 0.0003,
            "pre_landmark_hazard_multiplier": 1.0,
            "log_cshr_primary_by_arm": [0.0, 0.0],
            "log_cshr_competing_by_arm": [0.0, 1.20],
            "outcome_covariate_log_hr_primary": [0.15, 0.10, 0.15, 0.10],
            "outcome_covariate_log_hr_competing": [0.15, 0.10, 0.15, 0.10],
        }
        dgp_path = self.write_json(dgp_counter, "dgp_high_competing.json")
        out_dir = self.base / "counterexample_run"

        r_cli = self.run_cli(
            "run-synthetic",
            [
                "--protocol",
                str(EXAMPLE_PROTOCOL),
                "--seed",
                "90210",
                "--cases",
                "2500",
                "--weight-method",
                "overlap",
                "--bootstrap-iterations",
                "100",
                "--dgp-params",
                str(dgp_path),
                "--out",
                str(out_dir),
            ],
        )
        self.assertEqual(r_cli.returncode, 0, r_cli.stderr)

        est = json.loads((out_dir / "estimates.json").read_text(encoding="utf-8"))
        audit_md = (out_dir / "audit_report.md").read_text(encoding="utf-8")
        contrast = est["contrasts"]["strategy_active__vs__strategy_control"]

        shr = contrast["fine_gray_shr_primary"]
        cshr1 = contrast["cause_specific_cox_primary"]
        cshr2 = contrast["cause_specific_cox_competing"]

        # Primary cause-specific HR stays near 1.0 (CI covers 1.0), competing csHR_2 >> 1.0,
        # YET Fine-Gray sHR < 1.0 (upper 95% CI < 1.0) and RD_1(365) < 0!
        self.assertLessEqual(cshr1["ci_lower"], 1.05)
        self.assertGreaterEqual(cshr1["ci_upper"], 0.95)
        self.assertGreater(cshr2["point"], 2.5)
        self.assertGreater(cshr2["ci_lower"], 2.0)
        self.assertLess(shr["point"], 0.80)
        self.assertLess(shr["ci_upper"], 0.95)
        self.assertLess(contrast["horizons"]["365"]["rd_primary"], -0.05)
        self.assertGreater(contrast["horizons"]["365"]["rd_competing"], 0.15)

        # Mandatory side-by-side co-presentation & divergence alert & Hernan (2010) warning
        self.assertTrue(est["methodological_warnings"]["competing_mortality_divergence_alert"]["triggered"])
        self.assertIn("Hernán", est["methodological_warnings"]["hazard_ratio_selection_bias_warning"])
        self.assertIn("MUST NEVER be interpreted", est["methodological_warnings"]["subdistribution_vs_cause_specific_warning"])

        for row in est["co_presented_comparison_table"]:
            for required_key in (
                "cif_primary_ref",
                "cif_primary_active",
                "cif_competing_ref",
                "cif_competing_active",
                "shr_primary",
                "cshr_primary",
                "cshr_competing",
                "evalue_rr_cif_point",
            ):
                self.assertIsNotNone(row[required_key], f"Missing {required_key} in co-presented table")

        self.assertIn("Fine-Gray sHR (CI)", audit_md)
        self.assertIn("Primary csHR_1 (CI)", audit_md)
        self.assertIn("Competing csHR_2 (CI)", audit_md)
        self.assertIn("Hernán", audit_md)
        self.assertIn("competing_mortality_divergence_alert", audit_md)
        self.assertIn("synthetic_only", audit_md)
        self.assertIn("非临床疗效证据", audit_md)

    def test_three_arm_landmark_and_joint_ipcw_compatibility(self):
        # 1. Three-arm generalized_overlap & pairwise_overlap
        proto_3 = copy.deepcopy(self.protocol)
        proto_3["treatment_strategies"].append(
            {"id": "strategy_experimental_c", "label": "Synthetic Regimen C", "role": "experimental"}
        )
        proto_3["estimand"]["target_estimand"] = "GENERALIZED_ATO"
        p3_gen = self.write_json(proto_3, "proto_3arm_gen.json")

        gen_out = self.base / "run_3arm_gen"
        r_gen = self.run_cli(
            "run-synthetic",
            [
                "--protocol",
                str(p3_gen),
                "--seed",
                "77",
                "--cases",
                "900",
                "--weight-method",
                "generalized_overlap",
                "--bootstrap-iterations",
                "60",
                "--out",
                str(gen_out),
            ],
        )
        self.assertEqual(r_gen.returncode, 0, r_gen.stderr)
        est_gen = json.loads((gen_out / "estimates.json").read_text(encoding="utf-8"))
        self.assertEqual(est_gen["reference_arm"]["id"], "strategy_control")
        self.assertEqual(
            set(est_gen["contrasts"].keys()),
            {
                "strategy_active__vs__strategy_control",
                "strategy_experimental_c__vs__strategy_control",
            },
        )
        self.assertEqual(len(est_gen["co_presented_comparison_table"]), 6)  # 2 contrasts * 3 horizons
        for aid in ("strategy_control", "strategy_active", "strategy_experimental_c"):
            self.assertTrue((gen_out / "curves" / f"cif_{aid}.csv").is_file())

        # Pairwise overlap 3-arm via staged diagnose -> estimate
        proto_3_pair = copy.deepcopy(proto_3)
        proto_3_pair["estimand"]["target_estimand"] = "PAIRWISE_ATO"
        p3_pair = self.write_json(proto_3_pair, "proto_3arm_pair.json")
        s3_dir = self.base / "s3"
        d3_dir = self.base / "d3"
        e3_dir = self.base / "e3"
        generate_synthetic_cohort(p3_pair, seed=88, cases=900, out_dir=s3_dir)
        _, d_code = diagnose_cohort(p3_pair, s3_dir / "cohort.csv", weight_method="pairwise_overlap", out_dir=d3_dir)
        self.assertEqual(d_code, 0)
        r_e3 = self.run_cli(
            "estimate",
            [
                "--protocol",
                str(p3_pair),
                "--diagnostics-dir",
                str(d3_dir),
                "--seed",
                "88",
                "--bootstrap-iterations",
                "50",
                "--out",
                str(e3_dir),
            ],
        )
        self.assertEqual(r_e3.returncode, 0, r_e3.stderr)
        est_pair = json.loads((e3_dir / "estimates.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(est_pair["transitivity_warning"])
        self.assertIn("non-transitive", est_pair["transitivity_warning"].lower())
        self.assertEqual(len(list((e3_dir / "curves").glob("*.csv"))), 4)  # 2 active pairs * 2 arms each

        # 2. Landmark design + covariate_cox & arm_stratified joint IPCW + custom --horizons
        lm_proto = copy.deepcopy(self.protocol)
        lm_proto["assignment_mode"] = "observational_landmark"
        lm_proto["time_zero"]["landmark_day"] = 30
        p_lm = self.write_json(lm_proto, "proto_landmark.json")

        for ipcw_mode in ("arm_stratified", "covariate_cox"):
            lm_out = self.base / f"lm_{ipcw_mode}"
            r_lm = self.run_cli(
                "run-synthetic",
                [
                    "--protocol",
                    str(p_lm),
                    "--seed",
                    "314",
                    "--cases",
                    "600",
                    "--weight-method",
                    "overlap",
                    "--bootstrap-iterations",
                    "40",
                    "--horizons",
                    "120,240",
                    "--censoring-ipcw",
                    ipcw_mode,
                    "--out",
                    str(lm_out),
                ],
            )
            self.assertEqual(r_lm.returncode, 0, r_lm.stderr)
            est_lm = json.loads((lm_out / "estimates.json").read_text(encoding="utf-8"))
            self.assertEqual(est_lm["censoring_model"]["censoring_ipcw_mode"], ipcw_mode)
            self.assertEqual(est_lm["bootstrap_config"]["baseline_propensity_weights"], "fixed")
            self.assertTrue(est_lm["bootstrap_config"]["censoring_ipcw_reestimated_in_bootstrap"])
            self.assertIn("baseline_propensity_weights=fixed", (lm_out / "audit_report.md").read_text(encoding="utf-8"))
            self.assertEqual(est_lm["horizons_days"], [120.0, 240.0])
            c_lm = pd.read_csv(lm_out / "curves" / "cif_strategy_control.csv")
            self.assertAlmostEqual(float(c_lm["time_day"].iloc[0]), 30.0, places=6)
            self.assertAlmostEqual(float(c_lm["time_since_landmark_day"].iloc[0]), 0.0, places=6)

    def test_fail_closed_gate_blocking_exit_code_3_and_contract_exit_code_2(self):
        # Construct extreme confounding cohort that fails diagnose gate (exit code 3)
        dgp_extreme = {
            "propensity_intercepts": [0.0, 0.0],
            "propensity_slopes": [[0.0, 0.0, 0.0, 0.0], [8.5, -7.5, 4.0, 4.0]],
        }
        dgp_path = self.write_json(dgp_extreme, "dgp_extreme.json")
        strict_proto = copy.deepcopy(self.protocol)
        strict_proto["estimand"]["diagnostic_gates"] = {
            "max_abs_smd": 0.05,
            "min_ess_ratio": 0.65,
            "positivity_epsilon": 0.05,
        }
        strict_path = self.write_json(strict_proto, "strict_proto.json")

        # A. run-synthetic on blocked cohort -> exit code 3, publishes diagnostics ONLY, NEVER weighted_cohort or estimates
        blocked_pipe_dir = self.base / "blocked_pipe"
        r_pipe = self.run_cli(
            "run-synthetic",
            [
                "--protocol",
                str(strict_path),
                "--seed",
                "11",
                "--cases",
                "300",
                "--weight-method",
                "iptw",
                "--bootstrap-iterations",
                "50",
                "--dgp-params",
                str(dgp_path),
                "--out",
                str(blocked_pipe_dir),
            ],
        )
        self.assertEqual(r_pipe.returncode, 3, r_pipe.stderr)
        self.assertIn("DIAGNOSTIC_GATE_BLOCKED", r_pipe.stderr)
        pipe_files = {str(p.relative_to(blocked_pipe_dir)) for p in blocked_pipe_dir.rglob("*") if p.is_file()}
        self.assertEqual(
            pipe_files,
            {
                "normalized_protocol.json",
                "attrition_table.csv",
                "balance_diagnostics.csv",
                "diagnostics_summary.json",
                "run_manifest.json",
            },
        )
        self.assertFalse((blocked_pipe_dir / "weighted_cohort.csv").exists())
        self.assertFalse((blocked_pipe_dir / "estimates.json").exists())

        # B. estimate on blocked diagnostics dir -> exit code 3, publishes NOTHING, SDK raises GateBlockedError
        est_from_blocked = self.base / "est_from_blocked"
        r_est_blk = self.run_cli(
            "estimate",
            [
                "--protocol",
                str(strict_path),
                "--diagnostics-dir",
                str(blocked_pipe_dir),
                "--seed",
                "11",
                "--bootstrap-iterations",
                "50",
                "--out",
                str(est_from_blocked),
            ],
        )
        self.assertEqual(r_est_blk.returncode, 3, r_est_blk.stderr)
        self.assertIn("ESTIMATE_GATE_BLOCKED", r_est_blk.stderr)
        self.assertFalse(est_from_blocked.exists())

        with self.assertRaises(GateBlockedError) as ctx_sdk:
            estimate_competing_risks(
                strict_path,
                diagnostics_dir=blocked_pipe_dir,
                seed=11,
                bootstrap_iterations=50,
                out_dir=est_from_blocked,
            )
        self.assertEqual(ctx_sdk.exception.exit_code, 3)
        self.assertEqual(ctx_sdk.exception.code, "UPSTREAM_GATE_NOT_PASSED")

        # C. Passed diagnose, then delete or tamper weighted_cohort.csv or pass mismatched protocol -> exit code 3
        synth_ok = self.base / "synth_ok"
        generate_synthetic_cohort(EXAMPLE_PROTOCOL, seed=42, cases=400, out_dir=synth_ok)

        # C1: Mismatched protocol
        diag_ok1 = self.base / "diag_ok1"
        diagnose_cohort(EXAMPLE_PROTOCOL, synth_ok / "cohort.csv", weight_method="overlap", out_dir=diag_ok1)
        alt_proto = copy.deepcopy(self.protocol)
        alt_proto["study_title"] = "Different Unmatched Protocol"
        alt_proto_path = self.write_json(alt_proto, "alt_proto.json")
        r_mismatch = self.run_cli(
            "estimate",
            [
                "--protocol",
                str(alt_proto_path),
                "--diagnostics-dir",
                str(diag_ok1),
                "--seed",
                "42",
                "--bootstrap-iterations",
                "30",
                "--out",
                str(self.base / "est_mismatch"),
            ],
        )
        self.assertEqual(r_mismatch.returncode, 3)
        self.assertIn("PROTOCOL_DIAGNOSTICS_MISMATCH", r_mismatch.stderr)
        self.assertFalse((self.base / "est_mismatch").exists())

        # C2: Tampered weighted_cohort.csv (SHA-256 mismatch with run_manifest.json)
        wc_file = diag_ok1 / "weighted_cohort.csv"
        wc_orig = wc_file.read_bytes()
        wc_file.write_bytes(wc_orig + b"\n")
        r_tamper = self.run_cli(
            "estimate",
            [
                "--protocol",
                str(EXAMPLE_PROTOCOL),
                "--weighted-cohort",
                str(wc_file),
                "--seed",
                "42",
                "--bootstrap-iterations",
                "30",
                "--out",
                str(self.base / "est_tamper"),
            ],
        )
        self.assertEqual(r_tamper.returncode, 3)
        self.assertIn("MANIFEST_INTEGRITY_MISMATCH", r_tamper.stderr)
        self.assertFalse((self.base / "est_tamper").exists())

        # C3: Missing weighted_cohort.csv
        wc_file.unlink()
        r_missing_wc = self.run_cli(
            "estimate",
            [
                "--protocol",
                str(EXAMPLE_PROTOCOL),
                "--diagnostics-dir",
                str(diag_ok1),
                "--seed",
                "42",
                "--bootstrap-iterations",
                "30",
                "--out",
                str(self.base / "est_missing_wc"),
            ],
        )
        self.assertEqual(r_missing_wc.returncode, 3)
        self.assertIn("MISSING_WEIGHTED_COHORT", r_missing_wc.stderr)
        self.assertFalse((self.base / "est_missing_wc").exists())

        # D. ContractError -> exit code 2 (non-competing protocol, existing out dir, non-synthetic source_mode)
        single_ev_proto = copy.deepcopy(self.protocol)
        single_ev_proto["outcomes"]["competing_event_codes"] = []
        single_ev_proto["outcomes"]["competing_event_handling"] = "none"
        single_ev_proto["estimand"]["effect_measures"] = ["RD", "HR"]
        p_single = self.write_json(single_ev_proto, "single_ev.json")
        r_single = self.run_cli(
            "run-synthetic",
            [
                "--protocol",
                str(p_single),
                "--seed",
                "1",
                "--cases",
                "200",
                "--weight-method",
                "overlap",
                "--bootstrap-iterations",
                "30",
                "--out",
                str(self.base / "single_out"),
            ],
        )
        self.assertEqual(r_single.returncode, 2)
        self.assertIn("COMPETING_RISKS_REQUIRED", r_single.stderr)

        with self.assertRaises(ContractError) as ctx_src:
            estimate_competing_risks(
                EXAMPLE_PROTOCOL,
                diagnostics_dir=diag_ok1,
                seed=1,
                bootstrap_iterations=10,
                out_dir=self.base / "bad_src",
                source_mode="real_patient_ehr",
            )
        self.assertEqual(ctx_src.exception.code, "UNSUPPORTED_SOURCE_MODE")

        # E. Non-SYN subject_id cohort passed through diagnose MUST be rejected by estimate (exit code 2)
        non_syn_df = pd.read_csv(synth_ok / "cohort.csv").rename(columns={"synthetic_case_id": "subject_id"})
        non_syn_df["subject_id"] = [f"REAL-{i:05d}" for i in range(len(non_syn_df))]
        non_syn_csv = self.base / "non_syn_cohort.csv"
        non_syn_df.to_csv(non_syn_csv, index=False)
        diag_non_syn = self.base / "diag_non_syn"
        diagnose_cohort(EXAMPLE_PROTOCOL, non_syn_csv, weight_method="overlap", out_dir=diag_non_syn)
        r_non_syn = self.run_cli(
            "estimate",
            [
                "--protocol",
                str(EXAMPLE_PROTOCOL),
                "--diagnostics-dir",
                str(diag_non_syn),
                "--seed",
                "1",
                "--bootstrap-iterations",
                "10",
                "--out",
                str(self.base / "est_non_syn"),
            ],
        )
        self.assertEqual(r_non_syn.returncode, 2)
        self.assertIn("NON_SYNTHETIC_COHORT_REJECTED", r_non_syn.stderr)
        self.assertFalse((self.base / "est_non_syn").exists())

    def test_evalue_blocking_and_rr_cif_horizon_consistency(self):
        # Direct sHR rejected
        with self.assertRaises(ContractError) as ctx_shr:
            compute_evalue(0.72, ci_lower=0.55, ci_upper=0.92, measure_type="sHR", has_competing_risks=True)
        self.assertEqual(ctx_shr.exception.code, "UNSUPPORTED_EVALUE_MEASURE")

        # Competing-risks csHR / HR with incidence > 0.15 rejected
        for mtype in ("csHR", "HR"):
            with self.assertRaises(ContractError) as ctx_hr:
                compute_evalue(0.68, measure_type=mtype, outcome_incidence=0.24, has_competing_risks=True)
            self.assertEqual(ctx_hr.exception.code, "UNSUPPORTED_EVALUE_MEASURE")

        # Verify pipeline horizon E-values match direct compute_evalue(RR_CIF)
        out_dir = self.base / "ev_run"
        res, code = run_synthetic_pipeline(
            EXAMPLE_PROTOCOL,
            seed=555,
            cases=700,
            weight_method="overlap",
            bootstrap_iterations=80,
            out_dir=out_dir,
        )
        self.assertEqual(code, 0, res)
        est = json.loads((out_dir / "estimates.json").read_text(encoding="utf-8"))
        for row in est["co_presented_comparison_table"]:
            rr = row["rr_cif_primary"]
            lo = row["rr_cif_primary_ci_lower"]
            hi = row["rr_cif_primary_ci_upper"]
            direct = compute_evalue(rr, ci_lower=lo, ci_upper=hi, measure_type="RR_CIF", has_competing_risks=True)
            self.assertAlmostEqual(row["evalue_rr_cif_point"], direct["point_estimate"], places=9)
            self.assertAlmostEqual(row["evalue_rr_cif_ci_limit"], direct["ci_limit_closer_to_null"], places=9)

    def test_byte_identical_determinism_sha256_manifest_and_agent_skill_contract_docs(self):
        run_a = self.base / "det_a"
        run_b = self.base / "det_b"
        for target in (run_a, run_b):
            r = self.run_cli(
                "run-synthetic",
                [
                    "--protocol",
                    str(EXAMPLE_PROTOCOL),
                    "--seed",
                    "2026",
                    "--cases",
                    "500",
                    "--weight-method",
                    "overlap",
                    "--bootstrap-iterations",
                    "75",
                    "--out",
                    str(target),
                ],
            )
            self.assertEqual(r.returncode, 0, r.stderr)

        files_a = sorted(str(p.relative_to(run_a)) for p in run_a.rglob("*") if p.is_file())
        files_b = sorted(str(p.relative_to(run_b)) for p in run_b.rglob("*") if p.is_file())
        self.assertEqual(files_a, files_b)

        for required in (
            "normalized_protocol.json",
            "attrition_table.csv",
            "balance_diagnostics.csv",
            "estimates.json",
            "audit_report.md",
            "run_manifest.json",
            "curves/cif_strategy_control.csv",
            "curves/cif_strategy_active.csv",
        ):
            self.assertIn(required, files_a)

        for rel in files_a:
            if rel == "run_manifest.json":
                continue
            self.assertEqual(
                (run_a / rel).read_bytes(),
                (run_b / rel).read_bytes(),
                f"Non-deterministic bytes in {rel}",
            )

        man_a = json.loads((run_a / "run_manifest.json").read_text(encoding="utf-8"))
        man_b = json.loads((run_b / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([k for k in man_a if k.endswith("_utc")], ["run_at_utc"])
        expected_non_manifest = [f for f in files_a if f != "run_manifest.json"]
        self.assertEqual(man_a["output_paths"], expected_non_manifest)
        for rel, digest in man_a["output_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((run_a / rel).read_bytes()).hexdigest())

        man_a.pop("run_at_utc")
        man_b.pop("run_at_utc")
        self.assertEqual(man_a, man_b)

        # Verify external Agent SKILL.md, CLI contract doc, and FEATURE_API_OVERVIEW sync
        self.assertTrue(SKILL_MD.is_file())
        self.assertTrue(CONTRACT_DOC.is_file())
        self.assertTrue(OVERVIEW_DOC.is_file())

        skill_text = SKILL_MD.read_text(encoding="utf-8")
        contract_text = CONTRACT_DOC.read_text(encoding="utf-8")
        overview_text = OVERVIEW_DOC.read_text(encoding="utf-8")

        for subcmd in ("validate-protocol", "synth", "diagnose", "estimate", "run-synthetic"):
            self.assertIn(subcmd, skill_text)
            self.assertIn(subcmd, contract_text)
        self.assertIn("synthetic_only", skill_text)
        self.assertIn("Hernán", skill_text)
        self.assertIn("emulatrx_tte.estimate", overview_text)
        self.assertIn("emulatrx_tte.run-synthetic", overview_text)
        self.assertIn("skills/emulatrx-tte/SKILL.md", overview_text)
        self.assertIn("baseline_propensity_weights", contract_text)
        self.assertNotIn("能力卡 10（排队预告｜尚未合并）", overview_text)

    def test_covariate_cox_fine_gray_joint_ipcw_score_oracle_and_sandwich_consistency(self):
        """Regression test for Round 1 review: covariate_cox Fine-Gray unified IPCW risk set & Lin-Wei SE."""
        from emulatrx_tte.competing_risks import (
            _compute_covariate_censoring_survival_matrix,
            _compute_stabilized_joint_ipcw_matrix,
            _fit_weighted_counting_process_cox,
        )

        # Exact Round-1 review counterexample (rng=728, n=400, covariate-dependent censoring + prior competing events)
        rng = np.random.default_rng(728)
        n = 400
        arm = np.array([0] * 200 + [1] * 200, dtype=np.int64)
        X = rng.standard_normal((n, 1))
        t1 = rng.exponential(scale=8.0, size=n)
        t2 = rng.exponential(scale=10.0, size=n)
        c_scale = 6.0 * np.exp(-1.2 * X[:, 0])
        tc = np.minimum(rng.exponential(scale=c_scale, size=n), 30.0)
        durations = np.minimum(np.minimum(t1, t2), tc)
        events = np.zeros(n, dtype=np.int64)
        events[(t1 <= t2) & (t1 <= tc)] = 1
        events[(t2 < t1) & (t2 <= tc)] = 2
        weights = np.ones(n, dtype=np.float64)
        pos_floor = 0.001

        self.assertGreater(int(np.sum(events == 1)), 80)
        self.assertGreater(int(np.sum(events == 2)), 80)
        self.assertGreater(int(np.sum(events == 0)), 100)

        ev1_times = np.unique(durations[events == 1])
        comp_mask = events == 2

        # Build independent joint-weighted Fine-Gray risk-set matrix W_oracle(i, m) and failure matrix D_oracle(i, m):
        # - Still in follow-up (T_i >= t_m): w_i * M_i(t_m) = w_i * G_{A_i}(t_m-) / G_i(t_m- | A_i, X_i)
        # - Prior competing event (T_i < t_m, E_i == 2):
        #     w_i * M_i(t_m) * clip(G_i(t_m- | A_i, X_i) / G_i(T_i- | A_i, X_i), 0, 1)
        #   = w_i * G_{A_i}(t_m-) / G_i(T_i- | A_i, X_i)
        m_nat = _compute_stabilized_joint_ipcw_matrix(
            durations,
            events,
            arm,
            2,
            X,
            weights,
            ev1_times,
            censoring_ipcw="covariate_cox",
            admin_end_day=30.0,
            positivity_floor=pos_floor,
        )
        g_ev1 = _compute_covariate_censoring_survival_matrix(
            durations,
            events,
            arm,
            2,
            X,
            weights,
            ev1_times,
            admin_end_day=30.0,
            positivity_floor=pos_floor,
        )
        g_ti = np.ones(n, dtype=np.float64)
        comp_idx = np.flatnonzero(comp_mask)
        g_comp = _compute_covariate_censoring_survival_matrix(
            durations,
            events,
            arm,
            2,
            X,
            weights,
            durations[comp_idx],
            admin_end_day=30.0,
            positivity_floor=pos_floor,
        )
        for pos, subj in enumerate(comp_idx):
            g_ti[subj] = max(float(g_comp[subj, pos]), pos_floor)

        g_arm_ev1 = np.ones((2, ev1_times.size), dtype=np.float64)
        for k in (0, 1):
            mk = arm == k
            g_arm_ev1[k, :] = np.maximum(
                compute_reverse_km_left_limit(durations[mk], events[mk], weights[mk], ev1_times),
                pos_floor,
            )

        D_oracle = ((durations[:, None] == ev1_times[None, :]) & (events[:, None] == 1)) * (weights[:, None] * m_nat)
        W_still = (durations[:, None] >= ev1_times[None, :]) * (weights[:, None] * m_nat)
        W_comp_direct = (
            (durations[:, None] < ev1_times[None, :])
            & comp_mask[:, None]
        ) * (weights[:, None] * (g_arm_ev1[arm, :] / g_ti[:, None]))
        W_oracle = W_still + W_comp_direct

        # Verify algebraic identity M_i(t_m) * clip(G_i(t_m-)/G_i(T_i-), 0, 1) == G_{A_i}(t_m-)/G_i(T_i-)
        W_comp_factored = (
            (durations[:, None] < ev1_times[None, :])
            & comp_mask[:, None]
        ) * (weights[:, None] * m_nat * np.clip(g_ev1 / g_ti[:, None], 0.0, 1.0))
        np.testing.assert_allclose(W_comp_factored, W_comp_direct, rtol=1e-13, atol=1e-13)

        z_float = arm.astype(np.float64)

        def oracle_score_hess_se(beta: float) -> tuple[float, float, float]:
            r = np.exp(z_float * beta)
            D_m = np.sum(D_oracle, axis=0)
            A = W_oracle * r[:, None]
            S0 = np.sum(A, axis=0)
            S1 = np.sum(A * z_float[:, None], axis=0)
            zbar = S1 / S0
            score_val = float(np.sum(D_oracle * z_float[:, None]) - np.sum(D_m * zbar))
            hess_val = float(np.sum(D_m * zbar * (1.0 - zbar)))
            h_im = A * (D_m / S0)[None, :]
            u_i = np.sum((D_oracle - h_im) * (z_float[:, None] - zbar[None, :]), axis=1)
            meat = (n / (n - 1.0)) * float(np.sum(u_i * u_i))
            se_val = math.sqrt(meat) / hess_val
            return score_val, hess_val, se_val

        # Bisection root of independent oracle score equation
        lo, hi = -2.0, 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if oracle_score_hess_se(mid)[0] > 0.0:
                lo = mid
            else:
                hi = mid
        beta_oracle = 0.5 * (lo + hi)
        _, _, se_oracle = oracle_score_hess_se(beta_oracle)

        fit_cov = fit_fine_gray_shr(
            durations,
            events,
            arm,
            2,
            weights,
            censoring_ipcw="covariate_cox",
            X_cov=X,
            admin_end_day=30.0,
            positivity_floor=pos_floor,
        )
        self.assertTrue(fit_cov["converged"])
        beta_est = fit_cov["beta"][0]
        se_est = fit_cov["se_robust"][0]

        # Score residual at fit_fine_gray_shr beta MUST vanish to < 1e-10 and match oracle beta & Lin-Wei SE
        score_at_est, _, _ = oracle_score_hess_se(beta_est)
        self.assertLess(abs(score_at_est), 1e-10)
        self.assertAlmostEqual(beta_est, beta_oracle, places=9)
        self.assertAlmostEqual(se_est, se_oracle, places=9)

        # Verify multi-chunk boundary invariance (chunk_size=7 across >15 chunks vs full matrix)
        fit_small_chunk = _fit_weighted_counting_process_cox(
            z_float[:, None],
            int(ev1_times.size),
            lambda m0, m1: (D_oracle[:, m0:m1], W_oracle[:, m0:m1]),
            chunk_size=7,
        )
        self.assertAlmostEqual(fit_small_chunk["beta"][0], beta_est, places=11)
        self.assertAlmostEqual(fit_small_chunk["se_robust"][0], se_est, places=11)

        # Also verify arm_stratified mode score equation consistency on the same cohort with non-uniform weights
        w_nonunif = 0.5 + rng.uniform(0.0, 1.5, size=n)
        fit_strat = fit_fine_gray_shr(
            durations,
            events,
            arm,
            2,
            w_nonunif,
            censoring_ipcw="arm_stratified",
            admin_end_day=30.0,
            positivity_floor=pos_floor,
        )
        g_strat_ev1 = np.ones((2, ev1_times.size), dtype=np.float64)
        g_strat_ti = np.ones(n, dtype=np.float64)
        for k in (0, 1):
            mk = arm == k
            g_strat_ev1[k, :] = compute_reverse_km_left_limit(durations[mk], events[mk], w_nonunif[mk], ev1_times)
            ck = mk & comp_mask
            g_strat_ti[ck] = np.maximum(
                compute_reverse_km_left_limit(durations[mk], events[mk], w_nonunif[mk], durations[ck]),
                1e-15,
            )
        D_s = ((durations[:, None] == ev1_times[None, :]) & (events[:, None] == 1)) * w_nonunif[:, None]
        W_s = (durations[:, None] >= ev1_times[None, :]) * w_nonunif[:, None] + (
            (durations[:, None] < ev1_times[None, :]) & comp_mask[:, None]
        ) * (w_nonunif[:, None] * np.clip(g_strat_ev1[arm, :] / g_strat_ti[:, None], 0.0, 1.0))
        r_s = np.exp(z_float * fit_strat["beta"][0])
        D_sm = np.sum(D_s, axis=0)
        zbar_s = np.sum(W_s * r_s[:, None] * z_float[:, None], axis=0) / np.sum(W_s * r_s[:, None], axis=0)
        score_strat = float(np.sum(D_s * z_float[:, None]) - np.sum(D_sm * zbar_s))
        self.assertLess(abs(score_strat), 1e-10)


if __name__ == "__main__":
    unittest.main()
