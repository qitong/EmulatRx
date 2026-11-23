"""Unit and end-to-end synthetic tests for emulatrx_tte and fixed E-value utilities (Issue #8)."""

from __future__ import annotations

import ast
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

import numpy as np
import pandas as pd

import emulation_utils as eu
from emulatrx_tte import (
    ContractError,
    METHODOLOGICAL_DISCLAIMER,
    compute_evalue,
    compute_weights,
    diagnose_cohort,
    generate_synthetic_cohort,
    validate_protocol_dict,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PROTOCOL = ROOT / "examples" / "tte_competing_risk_v1.json"
SCHEMA_PATH = ROOT / "schemas" / "tte-protocol-v1.schema.json"


def _vanderweele_ding_evalue_from_hr(hr: float, common: bool) -> tuple[float, float]:
    if common:
        rr = 1.0 if hr == 1.0 else (1.0 - 0.5 ** math.sqrt(hr)) / (1.0 - 0.5 ** math.sqrt(1.0 / hr))
    else:
        rr = hr
    rr_star = rr if rr >= 1.0 else 1.0 / rr
    ev = 1.0 if rr_star == 1.0 else rr_star + math.sqrt(rr_star * (rr_star - 1.0))
    return rr, ev


class TTEProtocolContractTests(unittest.TestCase):
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

    def write_protocol(self, obj: dict, name: str = "protocol.json") -> Path:
        path = self.base / name
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def test_schema_file_is_valid_draft_2020_12_json(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(schema["additionalProperties"])
        for req in (
            "protocol_version",
            "study_title",
            "simulation_or_data_label",
            "eligibility",
            "time_zero",
            "treatment_strategies",
            "assignment_mode",
            "followup",
            "outcomes",
            "estimand",
        ):
            self.assertIn(req, schema["required"])

    def test_example_protocol_normalizes_reproducibly_with_disclaimer(self):
        out1 = self.base / "val1"
        res1 = self.run_cli("validate-protocol", ["--protocol", str(EXAMPLE_PROTOCOL), "--out", str(out1)])
        self.assertEqual(res1.returncode, 0, res1.stderr)

        norm_path = out1 / "normalized_protocol.json"
        report_path = out1 / "protocol_check_report.md"
        manifest_path = out1 / "run_manifest.json"
        self.assertTrue(norm_path.is_file())
        self.assertTrue(report_path.is_file())
        self.assertTrue(manifest_path.is_file())

        normalized = json.loads(norm_path.read_text(encoding="utf-8"))
        report_text = report_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        expected_phrase = (
            "observable-rule checking detects only structural/timestamp rule violations "
            "in supplied metadata/columns and cannot detect unmeasured confounding, "
            "measurement error, informative censoring, or prove causal validity."
        )
        self.assertIn(expected_phrase, normalized["methodological_disclaimer"].lower())
        self.assertIn(expected_phrase, report_text.lower())
        self.assertIn(expected_phrase, manifest["methodological_disclaimer"].lower())
        self.assertEqual(manifest["input_sha256"], hashlib.sha256(EXAMPLE_PROTOCOL.read_bytes()).hexdigest())
        for rel, digest in manifest["output_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((out1 / rel).read_bytes()).hexdigest())

        # Re-validating normalized_protocol.json yields byte-identical normalized_protocol.json
        out2 = self.base / "val2"
        res2 = self.run_cli("validate-protocol", ["--protocol", str(norm_path), "--out", str(out2)])
        self.assertEqual(res2.returncode, 0, res2.stderr)
        self.assertEqual(norm_path.read_bytes(), (out2 / "normalized_protocol.json").read_bytes())

    def test_rejects_missing_time_zero_and_post_t0_eligibility_and_competing_and_horizon(self):
        # 1. Missing time_zero
        no_tz = copy.deepcopy(self.protocol)
        del no_tz["time_zero"]
        p_no_tz = self.write_protocol(no_tz, "no_tz.json")
        r1 = self.run_cli("validate-protocol", ["--protocol", str(p_no_tz), "--out", str(self.base / "o1")])
        self.assertEqual(r1.returncode, 2)
        self.assertIn("MISSING_TIME_ZERO", r1.stderr)
        self.assertFalse((self.base / "o1").exists())

        # 2. Empty time_zero
        empty_tz = copy.deepcopy(self.protocol)
        empty_tz["time_zero"] = {}
        p_empty_tz = self.write_protocol(empty_tz, "empty_tz.json")
        r1b = self.run_cli("validate-protocol", ["--protocol", str(p_empty_tz), "--out", str(self.base / "o1b")])
        self.assertEqual(r1b.returncode, 2)
        self.assertIn("MISSING_TIME_ZERO", r1b.stderr)

        # 3. Post-time-zero eligibility criterion -> POST_TIME_ZERO_ELIGIBILITY with Hint
        post_elig = copy.deepcopy(self.protocol)
        post_elig["eligibility"]["inclusion_criteria"][0]["assessed_day_offset"] = 7
        p_post = self.write_protocol(post_elig, "post_elig.json")
        r2 = self.run_cli("validate-protocol", ["--protocol", str(p_post), "--out", str(self.base / "o2")])
        self.assertEqual(r2.returncode, 2)
        self.assertIn("POST_TIME_ZERO_ELIGIBILITY", r2.stderr)
        self.assertIn("Hint:", r2.stderr)
        self.assertFalse((self.base / "o2").exists())

        # 4. Unhandled competing event -> UNHANDLED_COMPETING_EVENT
        unhandled = copy.deepcopy(self.protocol)
        unhandled["outcomes"]["competing_event_handling"] = "none"
        p_unhandled = self.write_protocol(unhandled, "unhandled.json")
        r3 = self.run_cli("validate-protocol", ["--protocol", str(p_unhandled), "--out", str(self.base / "o3")])
        self.assertEqual(r3.returncode, 2)
        self.assertIn("UNHANDLED_COMPETING_EVENT", r3.stderr)

        # 5. RD / RR_CIF with empty horizon_days -> MISSING_HORIZON
        no_horizon = copy.deepcopy(self.protocol)
        no_horizon["followup"]["horizon_days"] = []
        p_no_h = self.write_protocol(no_horizon, "no_horizon.json")
        r4 = self.run_cli("validate-protocol", ["--protocol", str(p_no_h), "--out", str(self.base / "o4")])
        self.assertEqual(r4.returncode, 2)
        self.assertIn("MISSING_HORIZON", r4.stderr)

        # 6. Duplicate strategy IDs or < 2 strategies -> INVALID_STRATEGIES
        dup_strat = copy.deepcopy(self.protocol)
        dup_strat["treatment_strategies"][1]["id"] = dup_strat["treatment_strategies"][0]["id"]
        p_dup = self.write_protocol(dup_strat, "dup_strat.json")
        r5 = self.run_cli("validate-protocol", ["--protocol", str(p_dup), "--out", str(self.base / "o5")])
        self.assertEqual(r5.returncode, 2)
        self.assertIn("INVALID_STRATEGIES", r5.stderr)

        one_strat = copy.deepcopy(self.protocol)
        one_strat["treatment_strategies"] = [one_strat["treatment_strategies"][0]]
        with self.assertRaises(ContractError) as ctx:
            validate_protocol_dict(one_strat)
        self.assertEqual(ctx.exception.code, "INVALID_STRATEGIES")

        # 7. Missing or out-of-range diagnostic gates & existing output preservation
        bad_gate = copy.deepcopy(self.protocol)
        bad_gate["estimand"]["diagnostic_gates"]["positivity_epsilon"] = 0.55
        with self.assertRaises(ContractError) as ctx_gate:
            validate_protocol_dict(bad_gate)
        self.assertEqual(ctx_gate.exception.code, "INVALID_DIAGNOSTIC_GATE")

        keep_dir = self.base / "existing"
        keep_dir.mkdir()
        (keep_dir / "keep.txt").write_text("preserve", encoding="utf-8")
        r_exist = self.run_cli("validate-protocol", ["--protocol", str(EXAMPLE_PROTOCOL), "--out", str(keep_dir)])
        self.assertEqual(r_exist.returncode, 2)
        self.assertIn("OUTPUT_EXISTS", r_exist.stderr)
        self.assertEqual((keep_dir / "keep.txt").read_text(encoding="utf-8"), "preserve")
        self.assertEqual(list(self.base.glob(".emulatrx-tte-stage-*")), [])


class TTESyntheticAndWeightingTests(unittest.TestCase):
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

    def test_seeded_synth_is_byte_identical_and_strictly_labeled(self):
        decoy = self.base / "patient_records.csv"
        decoy.write_text("patient_id,secret\nP1,DECOY_PATIENT_SECRET\n", encoding="utf-8")

        out_a = self.base / "synth_a"
        out_b = self.base / "synth_b"
        ra = self.run_cli(
            "synth",
            ["--protocol", str(EXAMPLE_PROTOCOL), "--seed", "123", "--cases", "400", "--out", str(out_a)],
        )
        rb = self.run_cli(
            "synth",
            ["--protocol", str(EXAMPLE_PROTOCOL), "--seed", "123", "--cases", "400", "--out", str(out_b)],
        )
        self.assertEqual(ra.returncode, 0, ra.stderr)
        self.assertEqual(rb.returncode, 0, rb.stderr)

        for fname in ("cohort.csv", "dgp_truth.json", "PURPOSE.txt"):
            self.assertEqual((out_a / fname).read_bytes(), (out_b / fname).read_bytes())

        ma = json.loads((out_a / "run_manifest.json").read_text(encoding="utf-8"))
        mb = json.loads((out_b / "run_manifest.json").read_text(encoding="utf-8"))
        ma.pop("run_at_utc")
        mb.pop("run_at_utc")
        self.assertEqual(ma, mb)

        dgp_raw = (out_a / "dgp_truth.json").read_text(encoding="utf-8")
        dgp_obj = json.loads(dgp_raw)
        self.assertEqual(dgp_obj["parameter_type"], "synthetic_dgp_parameters")
        self.assertIn("synthetic_dgp_parameters", dgp_obj)
        self.assertNotIn("true clinical efficacy", dgp_raw.lower())

        df = pd.read_csv(out_a / "cohort.csv")
        self.assertEqual(len(df), 400)
        self.assertEqual(df["synthetic_case_id"].iloc[0], "SYN-000001")
        self.assertEqual(df["synthetic_case_id"].iloc[-1], "SYN-000400")
        self.assertEqual(set(df["event_code"].unique()), {0, 1, 2})
        for p in out_a.rglob("*"):
            if p.is_file():
                self.assertNotIn("DECOY_PATIENT_SECRET", p.read_text(encoding="utf-8"))

        # Out-of-range cases/seed rejected with code 2
        r_bad = self.run_cli(
            "synth",
            ["--protocol", str(EXAMPLE_PROTOCOL), "--seed", "1", "--cases", "20001", "--out", str(self.base / "bad")],
        )
        self.assertEqual(r_bad.returncode, 2)
        self.assertIn("INVALID_CASE_COUNT", r_bad.stderr)

    def test_two_arm_overlap_exact_balance_and_bounds_and_iptw_truncation(self):
        synth_dir = self.base / "synth"
        generate_synthetic_cohort(EXAMPLE_PROTOCOL, seed=2026, cases=800, out_dir=synth_dir)

        ow_dir = self.base / "diag_ow"
        res_ow = self.run_cli(
            "diagnose",
            [
                "--protocol",
                str(EXAMPLE_PROTOCOL),
                "--cohort",
                str(synth_dir / "cohort.csv"),
                "--weight-method",
                "overlap",
                "--out",
                str(ow_dir),
            ],
        )
        self.assertEqual(res_ow.returncode, 0, res_ow.stderr)
        summary = json.loads((ow_dir / "diagnostics_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["gate_status"], "passed")
        self.assertIsNone(summary["transitivity_warning"])
        self.assertGreater(summary["max_abs_smd_unweighted"], 0.15)
        self.assertLess(summary["max_abs_smd_weighted"], 1e-6)

        wdf = pd.read_csv(ow_dir / "weighted_cohort.csv")
        raw_w = wdf["raw_weight"].to_numpy(dtype=float)
        ps_active = wdf["ps_strategy_active"].to_numpy(dtype=float)
        tilt_h = wdf["tilting_h"].to_numpy(dtype=float)
        self.assertTrue(np.all((raw_w > 0.0) & (raw_w < 1.0)))
        self.assertTrue(np.all((tilt_h > 0.0) & (tilt_h <= 0.25 + 1e-12)))
        # Verify h(e) = e * (1 - e) is maximized closest to e = 0.50
        max_h_idx = int(np.argmax(tilt_h))
        min_dist_idx = int(np.argmin(np.abs(ps_active - 0.5)))
        self.assertEqual(max_h_idx, min_dist_idx)

        # Stabilized IPTW with quantile truncation
        siptw_dir = self.base / "diag_siptw"
        res_siptw = self.run_cli(
            "diagnose",
            [
                "--protocol",
                str(EXAMPLE_PROTOCOL),
                "--cohort",
                str(synth_dir / "cohort.csv"),
                "--weight-method",
                "stabilized_iptw",
                "--truncate-quantiles",
                "0.02,0.98",
                "--out",
                str(siptw_dir),
            ],
        )
        self.assertEqual(res_siptw.returncode, 0, res_siptw.stderr)
        sum_siptw = json.loads((siptw_dir / "diagnostics_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(sum_siptw["gate_status"], "passed")
        self.assertEqual(sum_siptw["truncation_quantiles"], [0.02, 0.98])
        lo_b, hi_b = sum_siptw["truncation_bounds"]
        w_siptw = pd.read_csv(siptw_dir / "weighted_cohort.csv")["raw_weight"].to_numpy(dtype=float)
        self.assertGreaterEqual(float(np.min(w_siptw)), lo_b - 1e-9)
        self.assertLessEqual(float(np.max(w_siptw)), hi_b + 1e-9)

    def test_landmark_slicing_attrition_and_post_landmark_covariate_leak_blocked(self):
        lm_proto = copy.deepcopy(self.protocol)
        lm_proto["assignment_mode"] = "observational_landmark"
        lm_proto["time_zero"]["landmark_day"] = 30
        lm_proto["eligibility"]["inclusion_criteria"][0]["assessed_day_offset"] = 15  # allowed <= 30
        lm_proto_path = self.base / "landmark_protocol.json"
        lm_proto_path.write_text(json.dumps(lm_proto), encoding="utf-8")

        synth_dir = self.base / "lm_synth"
        generate_synthetic_cohort(lm_proto_path, seed=77, cases=500, out_dir=synth_dir)
        raw_df = pd.read_csv(synth_dir / "cohort.csv")
        pre_landmark_n = int((raw_df["duration_day"] <= 30.0).sum())
        self.assertGreater(pre_landmark_n, 0, "Expected pre-landmark events/censoring for landmark_day=30")

        diag_dir = self.base / "lm_diag"
        result, code = diagnose_cohort(
            lm_proto_path,
            synth_dir / "cohort.csv",
            weight_method="overlap",
            out_dir=diag_dir,
        )
        self.assertEqual(code, 0)
        self.assertEqual(result["gate_status"], "passed")

        with (diag_dir / "attrition_table.csv").open(encoding="utf-8") as f:
            attr_rows = list(csv.DictReader(f))
        self.assertEqual(int(attr_rows[0]["remaining_n"]), 500)
        self.assertEqual(int(attr_rows[1]["excluded_n"]), pre_landmark_n)
        self.assertEqual(int(attr_rows[1]["remaining_n"]), 500 - pre_landmark_n)

        w_lm = pd.read_csv(diag_dir / "weighted_cohort.csv")
        self.assertEqual(len(w_lm), 500 - pre_landmark_n)
        self.assertTrue((w_lm["duration_day"] > 30.0).all())
        np.testing.assert_allclose(
            w_lm["duration_since_landmark_day"].to_numpy(),
            w_lm["duration_day"].to_numpy() - 30.0,
            atol=1e-6,
        )

        # Inject a post-landmark covariate column (assessed at day 45 > landmark_day=30) into cohort CSV
        leaked_df = raw_df.copy()
        leaked_df["post_landmark_lab"] = 1.23
        leaked_df["post_landmark_lab__assessed_day_offset"] = 45
        leaked_csv = self.base / "leaked_cohort.csv"
        leaked_df.to_csv(leaked_csv, index=False)

        r_leak = self.run_cli(
            "diagnose",
            [
                "--protocol",
                str(lm_proto_path),
                "--cohort",
                str(leaked_csv),
                "--weight-method",
                "overlap",
                "--out",
                str(self.base / "leak_out"),
            ],
        )
        self.assertEqual(r_leak.returncode, 2)
        self.assertIn("POST_LANDMARK_COVARIATE_LEAK", r_leak.stderr)
        self.assertFalse((self.base / "leak_out").exists())

    def test_multi_arm_generalized_overlap_vs_pairwise_overlap_transitivity_metadata(self):
        proto_3arm = copy.deepcopy(self.protocol)
        proto_3arm["treatment_strategies"].append(
            {"id": "strategy_alt", "label": "Synthetic Regimen C", "role": "comparator"}
        )
        proto_3arm["estimand"]["target_estimand"] = "GENERALIZED_ATO"
        proto_3arm_path = self.base / "proto_3arm.json"
        proto_3arm_path.write_text(json.dumps(proto_3arm), encoding="utf-8")

        synth_3arm = self.base / "synth_3arm"
        generate_synthetic_cohort(proto_3arm_path, seed=99, cases=900, out_dir=synth_3arm)

        # Two-arm 'overlap' must be rejected for 3 arms
        r_bad_method = self.run_cli(
            "diagnose",
            [
                "--protocol",
                str(proto_3arm_path),
                "--cohort",
                str(synth_3arm / "cohort.csv"),
                "--weight-method",
                "overlap",
                "--out",
                str(self.base / "bad_ow3"),
            ],
        )
        self.assertEqual(r_bad_method.returncode, 2)
        self.assertIn("INVALID_WEIGHT_METHOD", r_bad_method.stderr)

        # Generalized overlap (Li & Li 2019)
        gen_dir = self.base / "gen_ow"
        r_gen = self.run_cli(
            "diagnose",
            [
                "--protocol",
                str(proto_3arm_path),
                "--cohort",
                str(synth_3arm / "cohort.csv"),
                "--weight-method",
                "generalized_overlap",
                "--out",
                str(gen_dir),
            ],
        )
        self.assertEqual(r_gen.returncode, 0, r_gen.stderr)
        gen_sum = json.loads((gen_dir / "diagnostics_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(gen_sum["gate_status"], "passed")
        self.assertIsNone(gen_sum["transitivity_warning"])
        gen_w = pd.read_csv(gen_dir / "weighted_cohort.csv")["raw_weight"].to_numpy(dtype=float)
        self.assertTrue(np.all((gen_w > 0.0) & (gen_w < 1.0)))

        # Verify K=2 generalized_overlap reduces identically to two-arm overlap
        rng = np.random.default_rng(5)
        X2 = rng.standard_normal((200, 3))
        a2 = (rng.uniform(size=200) < 0.45).astype(np.int64)
        w_two = compute_weights(X2, a2, ["a0", "a1"], weight_method="overlap")
        w_gen2 = compute_weights(X2, a2, ["a0", "a1"], weight_method="generalized_overlap")
        np.testing.assert_allclose(w_two.raw_weights, w_gen2.raw_weights, atol=1e-12)

        # Pairwise overlap: exact balance per pair + mandatory transitivity_warning
        pair_dir = self.base / "pair_ow"
        r_pair = self.run_cli(
            "diagnose",
            [
                "--protocol",
                str(proto_3arm_path),
                "--cohort",
                str(synth_3arm / "cohort.csv"),
                "--weight-method",
                "pairwise_overlap",
                "--out",
                str(pair_dir),
            ],
        )
        self.assertEqual(r_pair.returncode, 0, r_pair.stderr)
        pair_sum = json.loads((pair_dir / "diagnostics_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(pair_sum["gate_status"], "passed")
        self.assertIsInstance(pair_sum["transitivity_warning"], str)
        self.assertIn("non-transitive", pair_sum["transitivity_warning"].lower())
        self.assertIn("subpopulations differ", pair_sum["transitivity_warning"].lower())
        self.assertLess(pair_sum["max_abs_smd_weighted"], 1e-6)
        self.assertEqual(len(pair_sum["ess_diagnostics"]["pairwise"]), 3)

    def test_diagnostic_gate_blocks_with_exit_code_3_on_positivity_ess_smd_and_zero_risk_set(self):
        # 1. Extreme confounding / near-separation -> POSITIVITY_VIOLATION & ESS_RATIO_BELOW_MIN (exit code 3)
        dgp_extreme = {
            "propensity_intercepts": [0.0, 0.0],
            "propensity_slopes": [[0.0, 0.0, 0.0, 0.0], [8.5, -7.5, 4.0, 4.0]],
        }
        dgp_path = self.base / "dgp_extreme.json"
        dgp_path.write_text(json.dumps(dgp_extreme), encoding="utf-8")

        extreme_synth = self.base / "extreme_synth"
        generate_synthetic_cohort(
            EXAMPLE_PROTOCOL,
            seed=11,
            cases=300,
            out_dir=extreme_synth,
            dgp_params_path=dgp_path,
        )

        strict_proto = copy.deepcopy(self.protocol)
        strict_proto["estimand"]["diagnostic_gates"] = {
            "max_abs_smd": 0.05,
            "min_ess_ratio": 0.60,
            "positivity_epsilon": 0.05,
        }
        strict_proto_path = self.base / "strict_proto.json"
        strict_proto_path.write_text(json.dumps(strict_proto), encoding="utf-8")

        blocked_dir = self.base / "blocked_diag"
        r_block = self.run_cli(
            "diagnose",
            [
                "--protocol",
                str(strict_proto_path),
                "--cohort",
                str(extreme_synth / "cohort.csv"),
                "--weight-method",
                "iptw",
                "--out",
                str(blocked_dir),
            ],
        )
        self.assertEqual(r_block.returncode, 3, r_block.stderr)
        self.assertIn("DIAGNOSTIC_GATE_BLOCKED", r_block.stderr)
        # Must publish diagnostic artifacts WITHOUT weighted_cohort.csv
        published_files = {p.name for p in blocked_dir.iterdir() if p.is_file()}
        self.assertEqual(
            published_files,
            {"attrition_table.csv", "balance_diagnostics.csv", "diagnostics_summary.json", "run_manifest.json"},
        )
        blocked_summary = json.loads((blocked_dir / "diagnostics_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(blocked_summary["gate_status"], "blocked")
        failure_codes = {f["code"] for f in blocked_summary["gate_failures"]}
        self.assertIn("POSITIVITY_VIOLATION", failure_codes)
        self.assertIn("ESS_RATIO_BELOW_MIN", failure_codes)

        # 2. Zero risk set after landmark slicing -> exit code 3 with ZERO_RISK_SET
        df_zero = pd.read_csv(extreme_synth / "cohort.csv")
        df_zero.loc[df_zero["treatment_arm"] == "strategy_active", "duration_day"] = 10.0
        zero_csv = self.base / "zero_arm.csv"
        df_zero.to_csv(zero_csv, index=False)

        lm_proto = copy.deepcopy(self.protocol)
        lm_proto["assignment_mode"] = "observational_landmark"
        lm_proto["time_zero"]["landmark_day"] = 30
        lm_path = self.base / "lm_zero.json"
        lm_path.write_text(json.dumps(lm_proto), encoding="utf-8")

        zero_out = self.base / "zero_out"
        r_zero = self.run_cli(
            "diagnose",
            [
                "--protocol",
                str(lm_path),
                "--cohort",
                str(zero_csv),
                "--weight-method",
                "overlap",
                "--out",
                str(zero_out),
            ],
        )
        self.assertEqual(r_zero.returncode, 3, r_zero.stderr)
        zero_summary = json.loads((zero_out / "diagnostics_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(zero_summary["gate_status"], "blocked")
        self.assertEqual(zero_summary["gate_failures"][0]["code"], "ZERO_RISK_SET")
        self.assertFalse((zero_out / "weighted_cohort.csv").exists())

        # 3. Heavily truncated IPTW violating ultra-tight max_abs_smd -> exit code 3 with SMD_EXCEEDS_MAX
        smd_proto = copy.deepcopy(self.protocol)
        smd_proto["estimand"]["diagnostic_gates"] = {
            "max_abs_smd": 0.005,
            "min_ess_ratio": 0.10,
            "positivity_epsilon": 0.0001,
        }
        smd_proto_path = self.base / "smd_proto.json"
        smd_proto_path.write_text(json.dumps(smd_proto), encoding="utf-8")
        normal_synth = self.base / "normal_synth"
        generate_synthetic_cohort(EXAMPLE_PROTOCOL, seed=19, cases=250, out_dir=normal_synth)
        smd_out = self.base / "smd_out"
        r_smd = self.run_cli(
            "diagnose",
            [
                "--protocol",
                str(smd_proto_path),
                "--cohort",
                str(normal_synth / "cohort.csv"),
                "--weight-method",
                "stabilized_iptw",
                "--truncate-quantiles",
                "0.25,0.75",
                "--out",
                str(smd_out),
            ],
        )
        self.assertEqual(r_smd.returncode, 3, r_smd.stderr)
        smd_summary = json.loads((smd_out / "diagnostics_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(smd_summary["gate_status"], "blocked")
        self.assertIn("SMD_EXCEEDS_MAX", {f["code"] for f in smd_summary["gate_failures"]})
        self.assertFalse((smd_out / "weighted_cohort.csv").exists())


class EValueFormulaAndBoundaryTests(unittest.TestCase):
    def test_duplicate_perform_balancing_method_removed_in_emulation_utils(self):
        tree = ast.parse((ROOT / "emulation_utils.py").read_text(encoding="utf-8"))
        defs = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "perform_balancing_method"
        ]
        self.assertEqual(len(defs), 1, f"Expected single perform_balancing_method definition, got {len(defs)}")

    def test_vanderweele_ding_analytical_values_including_hr4_and_hr5(self):
        # Common outcome dataset (incidence = 0.30 > 0.15)
        df_common = pd.DataFrame({"event": [1] * 30 + [0] * 70})
        # Rare outcome dataset (incidence = 0.10 <= 0.15)
        df_rare = pd.DataFrame({"event": [1] * 10 + [0] * 90})

        # 1. HR = 4.0 and HR = 5.0 with incidence = 0.30 (previously ZeroDivisionError and NaN!)
        for hr_val in (4.0, 5.0, 0.25, 2.2):
            expected_rr, expected_ev = _vanderweele_ding_evalue_from_hr(hr_val, common=True)
            eu_res = eu.calculate_evalues(df_common, "event", hr_val)
            self.assertFalse(eu_res["is_rare_outcome"])
            self.assertAlmostEqual(float(eu_res["e_values"]["point_estimate"]), expected_ev, places=4)

            tte_res = compute_evalue(
                hr_val,
                measure_type="HR",
                outcome_incidence=0.30,
                has_competing_risks=False,
            )
            self.assertAlmostEqual(tte_res["approximated_rr"], expected_rr, places=10)
            self.assertAlmostEqual(tte_res["point_estimate"], expected_ev, places=10)

        # Symmetry check: HR = 0.25 and HR = 4.0 yield identical E-values
        ev_4 = compute_evalue(4.0, measure_type="HR", outcome_incidence=0.30)["point_estimate"]
        ev_025 = compute_evalue(0.25, measure_type="HR", outcome_incidence=0.30)["point_estimate"]
        self.assertAlmostEqual(ev_4, ev_025, places=10)

        # 2. Rare outcome HR = 2.5 (incidence = 0.10 <= 0.15) -> direct RR formula
        _, expected_rare = _vanderweele_ding_evalue_from_hr(2.5, common=False)
        eu_rare = eu.calculate_evalues(df_rare, "event", 2.5)
        self.assertTrue(eu_rare["is_rare_outcome"])
        self.assertNotIn("ci_limit_closer_to_null", eu_rare["e_values"])  # 3-arg call backward-compatible
        self.assertAlmostEqual(float(eu_rare["e_values"]["point_estimate"]), expected_rare, places=4)

    def test_confidence_interval_crossing_null_vs_non_crossing(self):
        df_common = pd.DataFrame({"event": [1] * 30 + [0] * 70})

        # CI crossing 1.0 -> ci_limit_closer_to_null == 1.0
        eu_cross = eu.calculate_evalues(df_common, "event", 1.8, ci_lower=0.85, ci_upper=3.10)
        self.assertEqual(eu_cross["e_values"]["ci_limit_closer_to_null"], "1.0000")

        tte_cross = compute_evalue(1.8, ci_lower=0.85, ci_upper=3.10, measure_type="RR")
        self.assertEqual(tte_cross["ci_limit_closer_to_null"], 1.0)

        # CI strictly > 1.0 -> E-value of ci_lower
        _, exp_lo = _vanderweele_ding_evalue_from_hr(1.6, common=True)
        eu_pos = eu.calculate_evalues(df_common, "event", 2.8, ci_lower=1.6, ci_upper=4.5)
        self.assertAlmostEqual(float(eu_pos["e_values"]["ci_limit_closer_to_null"]), exp_lo, places=4)

        # Protective CI strictly < 1.0 -> E-value of ci_upper
        _, exp_hi = _vanderweele_ding_evalue_from_hr(0.75, common=True)
        eu_prot = eu.calculate_evalues(df_common, "event", 0.50, ci_lower=0.30, ci_upper=0.75)
        self.assertAlmostEqual(float(eu_prot["e_values"]["ci_limit_closer_to_null"]), exp_hi, places=4)

        tte_prot = compute_evalue(
            0.50,
            ci_lower=0.30,
            ci_upper=0.75,
            measure_type="HR",
            outcome_incidence=0.30,
            has_competing_risks=False,
        )
        self.assertAlmostEqual(tte_prot["ci_limit_closer_to_null"], exp_hi, places=10)

    def test_competing_risks_and_shr_boundaries_enforced(self):
        # 1. Fine-Gray sHR rejected unconditionally
        with self.assertRaises(ContractError) as ctx_shr:
            compute_evalue(1.5, measure_type="sHR", outcome_incidence=0.10, has_competing_risks=True)
        self.assertEqual(ctx_shr.exception.code, "UNSUPPORTED_EVALUE_MEASURE")

        # 2. Competing-risks csHR with outcome_incidence > 0.15 rejected
        with self.assertRaises(ContractError) as ctx_cshr:
            compute_evalue(1.8, measure_type="csHR", outcome_incidence=0.30, has_competing_risks=True)
        self.assertEqual(ctx_cshr.exception.code, "UNSUPPORTED_EVALUE_MEASURE")

        with self.assertRaises(ContractError) as ctx_cshr_implicit:
            compute_evalue(1.8, measure_type="csHR", outcome_incidence=0.25)
        self.assertEqual(ctx_cshr_implicit.exception.code, "UNSUPPORTED_EVALUE_MEASURE")

        # 3. HR with has_competing_risks=True and outcome_incidence > 0.15 rejected
        with self.assertRaises(ContractError) as ctx_hr_comp:
            compute_evalue(2.0, measure_type="HR", outcome_incidence=0.22, has_competing_risks=True)
        self.assertEqual(ctx_hr_comp.exception.code, "UNSUPPORTED_EVALUE_MEASURE")

        # 4. RR_CIF supported and carries competing-pathway & unmeasured-confounding disclaimer
        rr_cif_res = compute_evalue(0.70, ci_lower=0.55, ci_upper=0.88, measure_type="RR_CIF", has_competing_risks=True)
        _, expected_rr_cif_ev = _vanderweele_ding_evalue_from_hr(0.70, common=False)
        self.assertAlmostEqual(rr_cif_res["point_estimate"], expected_rr_cif_ev, places=10)
        self.assertIn("never proves absence of unmeasured confounding", rr_cif_res["interpretation_note"])
        self.assertIn("competing events/mortality", rr_cif_res["interpretation_note"])


if __name__ == "__main__":
    unittest.main()
