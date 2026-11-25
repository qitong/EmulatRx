"""Deterministic RCT-DUPLICATE benchmarking and negative-control diagnostics."""

from __future__ import annotations

import json
import math
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

from emulatrx_tte._common import ContractError
from emulatrx_tte.benchmarking import (
    OFFLINE_RESULTS_FIXTURES,
    compute_baseline_comparability,
    compute_rct_duplicate_metrics,
    evaluate_negative_control_outcome,
    execute_benchmark_request,
    generate_hidden_confounding_benchmark_cohort,
    is_benchmarking_enabled,
    parse_clinicaltrials_results_study,
    resolve_rct_public_results,
    run_rct_benchmark_and_nco,
)
from emulatrx_tte.weighting import compute_weights


ROOT = Path(__file__).resolve().parents[1]
ALIGNED = {
    "overall_alignment": "aligned",
    "population_aligned": True,
    "arms_aligned": True,
    "outcome_aligned": True,
    "timeframe_aligned": True,
}
_PASSED_ESTIMATES = {
    "co_presented_comparison_table": [
        {
            "cshr_primary": 0.90,
            "cshr_primary_ci_lower": 0.75,
            "cshr_primary_ci_upper": 1.08,
        }
    ]
}


def _carolina_target(**overrides) -> dict:
    target = {
        "nct_id": "NCT01243424",
        "overall_alignment_status": "protocol_aligned",
        "population_alignment": "matched",
        "arm_alignment_status": "exact_dual_arm",
        "outcome_alignment": "matched",
        "followup_alignment": "adequate",
        "experimental_arms": ["Linagliptin 5 mg once daily"],
        "comparator_arms": ["Glimepiride 1–4 mg once daily"],
        "conditions": ["Diabetes Mellitus, Type 2"],
        "primary_outcomes": [
            {
                "measure": "Time to First Occurrence of 3P-MACE (Cardiovascular Death, Non-fatal Myocardial Infarction, or Non-fatal Stroke)",
                "timeFrame": "Up to approximately 8 years (median follow-up 6.3 years)",
            }
        ],
        "parsed_admin_end_day": 1095.0,
    }
    target.update(overrides)
    return target


RWD_HR = {
    "gate_passed": True,
    "effect_scale": "hr",
    "point_estimate": 0.90,
    "ci_lower": 0.75,
    "ci_upper": 1.08,
    "baseline_characteristics": {
        "age": {"mean": 70.0, "sd": 8.0},
        "female": {"proportion": 0.55},
    },
}


def _guard(test, report):
    guard = report["interpretive_guardrail"]
    test.assertIs(guard["numerical_proximity_proves_causality"], False)
    test.assertIs(guard["clinical_efficacy_claim_prohibited"], True)
    test.assertIn("绝不可作为", guard["summary_zh"])
    test.assertIs(report.get("ipd_used", False), False)


def _max_abs_weighted_smd(frame, weights, columns):
    arm = frame["treatment"].to_numpy()
    weight = np.asarray(weights, dtype=float)
    values = []
    for column in columns:
        covariate = frame[column].to_numpy(dtype=float)
        def mean(mask):
            return float(np.sum(weight[mask] * covariate[mask]) / np.sum(weight[mask]))
        diff = abs(mean(arm == 1) - mean(arm == 0))
        pooled = math.sqrt(
            0.5 * (float(np.var(covariate[arm == 1], ddof=1)) + float(np.var(covariate[arm == 0], ddof=1)))
        )
        values.append(0.0 if pooled == 0.0 and diff == 0.0 else diff / pooled)
    return max(values)


class HiddenConfoundingTests(unittest.TestCase):
    def _weighted(self, strength):
        generated = generate_hidden_confounding_benchmark_cohort(
            hidden_confounding_strength=strength,
            seed=20261125,
        )
        frame = generated["cohort"]
        self.assertNotIn("U", frame.columns)
        self.assertEqual(generated["observed_covariates"], ["age", "bmi", "prior_cvd"])
        self.assertEqual(generated["ground_truth_audit"]["true_hr_treatment_on_primary"], 1.0)
        self.assertEqual(generated["ground_truth_audit"]["true_rd_treatment_on_nco"], 0.0)
        design = frame[generated["observed_covariates"]].to_numpy(dtype=float)
        fitted = compute_weights(design, frame["treatment"].to_numpy(dtype=int), ["control", "treated"], weight_method="overlap")
        return frame, np.asarray(fitted.analysis_weights, dtype=float)

    def test_overlap_balances_observed_covariates_while_nco_detects_hidden_bias(self):
        frame, weights = self._weighted(1.2)
        self.assertLess(_max_abs_weighted_smd(frame, weights, ["age", "bmi", "prior_cvd"]), 0.02)
        probed = evaluate_negative_control_outcome(
            frame,
            treatment_col="treatment",
            weights=weights,
            negative_control_outcome_col="negative_control_outcome",
            outcome_col="outcome_event",
        )
        self.assertTrue(probed["residual_confounding_alert"])
        self.assertEqual(probed["alert_code"], "NEGATIVE_CONTROL_BIAS_DETECTED")
        self.assertLess(probed["p_nco"], 0.01)
        self.assertTrue(probed["ci_excludes_null"])
        _guard(self, probed)
        report = run_rct_benchmark_and_nco(
            RWD_HR,
            ALIGNED,
            nct_id="NCT01243424",
            cohort=frame,
            treatment_col="treatment",
            outcome_col="outcome_event",
            weights=weights,
            negative_control_outcome_col="negative_control_outcome",
            retrieved_at_utc="2026-11-25T00:00:00Z",
        )
        self.assertEqual(report["overall_validity_flag"], "high_residual_bias_risk")
        self.assertFalse(report["consistency_conclusion_allowed"])
        self.assertIs(report["numerical_proximity_proves_causality"], False)
        _guard(self, report)

    def test_clean_null_negative_control_covers_null(self):
        frame, weights = self._weighted(0.0)
        self.assertLess(_max_abs_weighted_smd(frame, weights, ["age", "bmi", "prior_cvd"]), 0.02)
        probed = evaluate_negative_control_outcome(
            frame,
            treatment_col="treatment",
            weights=weights,
            negative_control_outcome_col="negative_control_outcome",
            outcome_col="outcome_event",
        )
        self.assertFalse(probed["residual_confounding_alert"])
        self.assertFalse(probed["ci_excludes_null"])
        self.assertGreaterEqual(probed["ci_lower"], 0.0)
        self.assertFalse(probed["inconclusive_warning"])
        _guard(self, probed)

    def test_time_to_event_negative_control_uses_lin_wei_cox(self):
        frame, weights = self._weighted(1.2)
        probed = evaluate_negative_control_outcome(
            frame,
            treatment_col="treatment",
            weights=weights,
            negative_control_outcome_col="negative_control_outcome",
            negative_control_time_col="negative_control_time",
            outcome_col="outcome_event",
        )
        self.assertEqual(probed["primary_estimand"], "hr")
        self.assertEqual(probed["weighted_hr"]["variance"], "lin_wei_robust_sandwich")
        self.assertTrue(probed["residual_confounding_alert"])
        self.assertLess(probed["p_nco"], 0.01)


class OracleMetricTests(unittest.TestCase):
    def test_rct_duplicate_and_baseline_smd_match_hand_calculation(self):
        # theta = ln(0.90), ln(0.98)
        # SE = (ln upper - ln lower) / (2 * 1.959963984540054)
        # z = (theta_rwd - theta_rct) / sqrt(se_rwd^2 + se_rct^2) = -0.7018343393704267
        metrics = compute_rct_duplicate_metrics(
            scale="hr",
            rwd_point=0.90,
            rwd_lower=0.75,
            rwd_upper=1.08,
            rct_point=0.98,
            rct_lower=0.84,
            rct_upper=1.14,
            ni_margin=1.3,
        )
        self.assertAlmostEqual(metrics["standardized_difference_z"], -0.7018343393704267, places=8)
        self.assertAlmostEqual(metrics["p_difference"], 0.48278248087680387, places=8)
        self.assertTrue(metrics["standardized_difference_within_196"])
        self.assertAlmostEqual(metrics["prediction_interval_95"][0], 0.7725820977124557, places=8)
        self.assertAlmostEqual(metrics["prediction_interval_95"][1], 1.2431041346203282, places=8)
        self.assertTrue(metrics["estimate_within_95_pi"])
        direction = metrics["regulatory_direction_agreement"]
        self.assertTrue(direction["agree"])
        self.assertEqual(direction["rwd_direction"], "nonsignificant")
        self.assertTrue(direction["non_inferiority_agreement"]["agree"])
        self.assertTrue(direction["non_inferiority_agreement"]["rct_meets_ni"])
        self.assertAlmostEqual(direction["non_inferiority_agreement"]["ni_margin"], 1.3)
        _guard(self, metrics)

        rd = compute_rct_duplicate_metrics(
            scale="rd",
            rwd_point=0.02,
            rwd_lower=-0.01,
            rwd_upper=0.05,
            rct_point=-0.01,
            rct_lower=-0.04,
            rct_upper=0.02,
        )
        self.assertAlmostEqual(rd["standardized_difference_z"], 1.3859038243496777, places=8)
        self.assertAlmostEqual(rd["prediction_interval_95"][0], -0.052426406871192856, places=8)
        self.assertAlmostEqual(rd["prediction_interval_95"][1], 0.03242640687119285, places=8)

        # Continuous SMD = (70-64) / sqrt((8^2 + 9.5^2) / 2) = 0.6832091307714864
        # Binary SMD = (0.55-0.399) / sqrt((0.55*0.45 + 0.399*0.601) / 2) = 0.30591035995839616
        parsed = parse_clinicaltrials_results_study(OFFLINE_RESULTS_FIXTURES["NCT01243424"])
        self.assertAlmostEqual(parsed["point_estimate"], 0.98)
        self.assertEqual(parsed["ci_lower"], 0.84)
        self.assertEqual(parsed["ci_upper"], 1.14)
        self.assertEqual(parsed["n_treatment"], 3023)
        self.assertEqual(parsed["n_control"], 3010)
        self.assertAlmostEqual(parsed["baseline_characteristics"]["age"]["mean"], 64.0)
        self.assertAlmostEqual(parsed["baseline_characteristics"]["age"]["sd"], 9.5)
        self.assertAlmostEqual(parsed["baseline_characteristics"]["female"]["proportion"], 0.399)
        self.assertAlmostEqual(parsed["ni_margin"], 1.3)
        table = compute_baseline_comparability(
            RWD_HR["baseline_characteristics"],
            parsed["baseline_characteristics"],
        )
        by_name = {row["covariate"]: row for row in table["rows"]}
        self.assertAlmostEqual(by_name["age"]["smd_approx"], 0.6832091307714864, places=8)
        self.assertAlmostEqual(by_name["female"]["smd_approx"], 0.30591035995839616, places=8)
        self.assertTrue(by_name["age"]["transportability_shift_flag"])
        self.assertTrue(by_name["female"]["transportability_shift_flag"])
        self.assertEqual(table["n_shifted_covariates"], 2)
        self.assertAlmostEqual(table["max_abs_baseline_smd_approx"], 0.6832091307714864, places=8)
        self.assertIn("可运输性", table["interpretation_zh"])

    def test_raw_v2_results_section_is_parsed(self):
        study = {
            "protocolSection": {"identificationModule": {"nctId": "NCT00000001", "officialTitle": "Raw fixture"}},
            "hasResults": True,
            "resultsSection": {
                "participantFlowModule": {
                    "periods": [
                        {
                            "milestones": [
                                {
                                    "type": "STARTED",
                                    "achievements": [
                                        {"groupId": "FG000", "numSubjects": "120"},
                                        {"groupId": "FG001", "numSubjects": "118"},
                                    ],
                                }
                            ]
                        }
                    ]
                },
                "outcomeMeasuresModule": {
                    "outcomeMeasures": [
                        {
                            "type": "PRIMARY",
                            "title": "Custom hazard",
                            "timeFrame": "52 weeks",
                            "analyses": [
                                {
                                    "paramType": "Hazard Ratio (HR)",
                                    "paramValue": "1.25",
                                    "ciLowerLimit": "1.05",
                                    "ciUpperLimit": "1.40",
                                }
                            ],
                        }
                    ]
                },
                "baselineCharacteristicsModule": {"groups": [], "measures": []},
            },
        }
        parsed = parse_clinicaltrials_results_study(study)
        self.assertEqual(parsed["effect_scale"], "hr")
        self.assertAlmostEqual(parsed["point_estimate"], 1.25)
        self.assertEqual(parsed["n_treatment"], 120)
        self.assertEqual(parsed["primary_outcome_timeframe"], "52 weeks")

    def test_offline_resolution_does_not_open_the_network(self):
        with mock.patch("emulatrx_tte.benchmarking.urlopen") as opener:
            resolved = resolve_rct_public_results("NCT02296125", retrieved_at_utc="2026-11-25T00:00:00Z")
            opener.assert_not_called()
        self.assertAlmostEqual(resolved["aggregate"]["point_estimate"], 0.46)
        self.assertEqual(resolved["aggregate"]["n_treatment"], 279)
        self.assertEqual(resolved["source_receipt"]["ipd_used"], False)
        self.assertEqual(len(resolved["source_receipt"]["payload_sha256"]), 64)
        with mock.patch.dict(os.environ, {"EMULATRX_OFFLINE_RCT_CATALOG": "1"}):
            with mock.patch("emulatrx_tte.benchmarking.urlopen") as opener:
                fallback = resolve_rct_public_results("NCT01243424", allow_live=True)
                opener.assert_not_called()
        self.assertEqual(fallback["status"], "resolved")
        with mock.patch("emulatrx_tte.benchmarking.urlopen", side_effect=URLError("offline")):
            rescued = resolve_rct_public_results("NCT02296125", allow_live=True)
        self.assertEqual(rescued["status"], "resolved")
        self.assertAlmostEqual(rescued["aggregate"]["ci_lower"], 0.37)


class NegativeGateTests(unittest.TestCase):
    def test_missing_results_section_and_empty_citation_fail_closed(self):
        missing = run_rct_benchmark_and_nco(
            RWD_HR,
            ALIGNED,
            nct_id="NCT09990001",
            retrieved_at_utc="2026-11-25T00:00:00Z",
        )
        self.assertEqual(missing["gate_code"], "BENCHMARK_UNAVAILABLE_NO_RESULTS_SECTION")
        self.assertFalse(missing["benchmarkable"])
        self.assertIsNone(missing["effect_comparison"])
        self.assertFalse(missing["consistency_conclusion_allowed"])
        self.assertIn("resultsSection 缺失", missing["message_zh"])
        _guard(self, missing)
        ongoing = run_rct_benchmark_and_nco(RWD_HR, ALIGNED, nct_id="NCT06642272")
        self.assertEqual(ongoing["status"], "unavailable_no_public_results")
        empty = run_rct_benchmark_and_nco(
            RWD_HR,
            ALIGNED,
            nct_id="NCT01243424",
            curated_published_summary={
                "effect_scale": "hr",
                "point_estimate": 0.5,
                "ci_lower": 0.4,
                "ci_upper": 0.7,
                "n_treatment": 10,
                "n_control": 10,
                "primary_outcome_label": "unsourced",
                "source_citation": "",
            },
        )
        self.assertEqual(empty["gate_code"], "BENCHMARK_UNAVAILABLE_NO_RESULTS_SECTION")
        self.assertIsNone(empty["effect_comparison"])
        self.assertFalse(empty["consistency_conclusion_allowed"])
        _guard(self, empty)

    def test_protocol_or_endpoint_mismatch_blocks_even_after_confirmation(self):
        mismatched = run_rct_benchmark_and_nco(
            RWD_HR,
            {
                "overall_alignment": "aligned",
                "population_aligned": True,
                "arms_aligned": True,
                "outcome_aligned": False,
                "timeframe_aligned": True,
            },
            nct_id="NCT01243424",
            confirm_protocol_mismatch=True,
            confirm_arm_mismatch=True,
        )
        self.assertEqual(mismatched["gate_code"], "BENCHMARK_BLOCKED_PROTOCOL_OR_ENDPOINT_MISMATCH")
        self.assertEqual(mismatched["status"], "blocked_protocol_mismatch")
        self.assertIsNone(mismatched["effect_comparison"])
        self.assertFalse(mismatched["consistency_conclusion_allowed"])
        _guard(self, mismatched)
        surrogate = run_rct_benchmark_and_nco(
            RWD_HR,
            {
                "overall_alignment_status": "protocol_aligned",
                "population_alignment": "matched",
                "arm_alignment_status": "exact_dual_arm",
                "outcome_alignment": "surrogate_or_mismatched",
                "followup_alignment": "adequate",
            },
            nct_id="NCT01243424",
            confirm_protocol_mismatch=True,
        )
        self.assertEqual(surrogate["gate_code"], "BENCHMARK_BLOCKED_PROTOCOL_OR_ENDPOINT_MISMATCH")

    def test_upstream_gate_scale_mismatch_sparse_nco_and_guardrail(self):
        blocked = run_rct_benchmark_and_nco(
            {**RWD_HR, "gate_passed": False},
            ALIGNED,
            nct_id="NCT01243424",
        )
        self.assertEqual(blocked["gate_code"], "BENCHMARK_BLOCKED_UPSTREAM_DIAGNOSTIC_FAILED")
        self.assertEqual(blocked["status"], "blocked_upstream_gate")
        self.assertIsNone(blocked["effect_comparison"])
        _guard(self, blocked)
        scaled = run_rct_benchmark_and_nco(
            {
                "gate_passed": True,
                "effect_scale": "rd",
                "point_estimate": 0.02,
                "ci_lower": -0.01,
                "ci_upper": 0.05,
            },
            ALIGNED,
            nct_id="NCT01243424",
        )
        self.assertEqual(scaled["gate_code"], "BENCHMARK_SCALE_MISMATCH")
        self.assertIsNone(scaled["effect_comparison"])
        _guard(self, scaled)
        sparse = pd.DataFrame(
            {
                "treatment": [1, 1, 1, 1, 0, 0, 0, 0],
                "outcome_event": [0, 0, 0, 0, 0, 0, 0, 0],
                "negative_control_outcome": [1, 0, 0, 0, 1, 0, 0, 0],
            }
        )
        probed = evaluate_negative_control_outcome(
            sparse,
            treatment_col="treatment",
            weights=np.ones(len(sparse)),
            negative_control_outcome_col="negative_control_outcome",
            outcome_col="outcome_event",
        )
        self.assertEqual(probed["status"], "insufficient_events_inconclusive")
        self.assertFalse(probed["residual_confounding_alert"])
        self.assertTrue(probed["inconclusive_warning"])
        _guard(self, probed)
        invalid = evaluate_negative_control_outcome(
            sparse.assign(negative_control_outcome=sparse["outcome_event"]),
            treatment_col="treatment",
            weights=np.ones(len(sparse)),
            negative_control_outcome_col="outcome_event",
            outcome_col="outcome_event",
        )
        self.assertEqual(invalid["gate_code"], "INVALID_NEGATIVE_CONTROL_COLUMN")
        outside = evaluate_negative_control_outcome(
            sparse.assign(negative_control_outcome=[0, 1, 2, 0, 0, 1, 0, 0]),
            treatment_col="treatment",
            weights=np.ones(len(sparse)),
            negative_control_outcome_col="negative_control_outcome",
            outcome_col="outcome_event",
        )
        self.assertEqual(outside["gate_code"], "INVALID_NEGATIVE_CONTROL_COLUMN")
        for report in (blocked, scaled, probed, invalid, outside):
            self.assertIs(report["interpretive_guardrail"]["numerical_proximity_proves_causality"], False)


class BenchmarkInterfaceTests(unittest.TestCase):
    def test_feature_flag_parser(self):
        with mock.patch.dict(os.environ, {"EMULATRX_ENABLE_BENCHMARKING": "0"}):
            self.assertFalse(is_benchmarking_enabled())
        with mock.patch.dict(os.environ, {"EMULATRX_ENABLE_BENCHMARKING": "off"}):
            self.assertFalse(is_benchmarking_enabled())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EMULATRX_ENABLE_BENCHMARKING", None)
            self.assertTrue(is_benchmarking_enabled())

    def test_schema_covers_benchmark_report_keys(self):
        schema = json.loads((ROOT / "schemas" / "tte-benchmark-v1.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["title"], "EmulatRx TTE benchmark and negative-control report v1")
        report = run_rct_benchmark_and_nco(RWD_HR, ALIGNED, nct_id="NCT01243424", retrieved_at_utc="2026-11-25T00:00:00Z")
        for key in schema["required"]:
            self.assertIn(key, report)
        self.assertEqual(report["status"], "benchmarked")
        self.assertTrue(report["consistency_conclusion_allowed"])
        self.assertIs(report["ipd_used"], False)
        _guard(self, report)

    def test_cli_writes_json_and_html_and_honors_the_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "demo"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "emulatrx_tte",
                    "benchmark-rct",
                    "--synthetic-demo",
                    "hidden_bias",
                    "--nct-id",
                    "NCT01243424",
                    "--n",
                    "800",
                    "--seed",
                    "20261125",
                    "--output-dir",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "EMULATRX_OFFLINE_RCT_CATALOG": "1"},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads((out / "benchmark_report.json").read_text(encoding="utf-8"))
            html = (out / "benchmark_report.html").read_text(encoding="utf-8")
            self.assertTrue(payload["negative_control"]["residual_confounding_alert"])
            self.assertEqual(payload["overall_validity_flag"], "high_residual_bias_risk")
            self.assertIn("绝不可作为", html)
            self.assertIn("NEGATIVE_CONTROL_BIAS_DETECTED", html)
            self.assertIn("Source Receipt", html)
            self.assertNotIn("hidden_u", html)
            disabled = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "emulatrx_tte",
                    "benchmark-rct",
                    "--synthetic-demo",
                    "clean_null",
                    "--output-dir",
                    str(Path(directory) / "disabled"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                env={**os.environ, "EMULATRX_ENABLE_BENCHMARKING": "0", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(disabled.returncode, 2, disabled.stderr)
            self.assertIn("BENCHMARKING_FEATURE_DISABLED", disabled.stderr)
            self.assertFalse((Path(directory) / "disabled").exists())
        missing = subprocess.run(
            [sys.executable, "-m", "emulatrx_tte", "benchmark-rct", "--output-dir", str(ROOT / "should-not-create")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(missing.returncode, 2)
        self.assertIn("MISSING_BENCHMARK_INPUT", missing.stderr)


try:
    import starlette.testclient  # noqa: F401

    _HAS_PREVIEW_SERVICE = True
except ImportError:
    _HAS_PREVIEW_SERVICE = False


@unittest.skipUnless(_HAS_PREVIEW_SERVICE, "preview service dependencies are not installed")
class BenchmarkServiceTests(unittest.TestCase):
    def setUp(self):
        import importlib

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        (base / "web").write_text("w" * 40)
        (base / "agent").write_text("a" * 40)
        self.patch = mock.patch.dict(
            os.environ,
            {
                "EMULATRX_RUNS_DIR": str(base / "runs"),
                "EMULATRX_WEB_PASSWORD_FILE": str(base / "web"),
                "EMULATRX_MCP_TOKEN_FILE": str(base / "agent"),
                "EMULATRX_CODE_COMMIT": "b" * 40,
                "EMULATRX_CODE_DIRTY": "false",
                "EMULATRX_ALLOWED_HOSTS": "testserver",
                "EMULATRX_TTE_DEPENDENCY_COMMIT": "152af0610d4317cd18691da1173cd297566fee09",
                "EMULATRX_OFFLINE_RCT_CATALOG": "1",
                "EMULATRX_ENABLE_BENCHMARKING": "1",
            },
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)
        from emulatrx_preview import app

        self.service = importlib.reload(app)
        self.web = {"Authorization": "Basic " + __import__("base64").b64encode(b"preview:" + b"w" * 40).decode()}
        self.mcp = {"Authorization": "Bearer " + "a" * 40, "Accept": "application/json, text/event-stream"}
        self.client = starlette.testclient.TestClient(self.service.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_rest_mcp_auth_panel_and_feature_flag(self):
        denied = self.client.post("/api/v1/tte/benchmark", json={"synthetic_demo": "clean_null", "n": 400})
        self.assertEqual(denied.status_code, 401)
        home = self.client.get("/", headers=self.web)
        self.assertEqual(home.status_code, 200)
        self.assertNotIn("benchmark-diagnostic-panel", home.text)
        self.assertIn("方案 → 运行 → 结果", home.text)
        created = self.client.post(
            "/api/v1/tte/benchmark",
            headers=self.web,
            json={"synthetic_demo": "hidden_bias", "nct_id": "NCT01243424", "n": 800, "seed": 20261125},
        )
        self.assertEqual(created.status_code, 200, created.text[:500])
        body = created.json()
        self.assertEqual(body["overall_validity_flag"], "high_residual_bias_risk")
        self.assertIs(body["interpretive_guardrail"]["numerical_proximity_proves_causality"], False)
        self.assertIs(body["ipd_used"], False)
        rejected = self.client.post(
            "/api/v1/tte/benchmark",
            headers=self.web,
            json={"patients": [{"id": "1"}], "nct_id": "NCT01243424", "tte_result": RWD_HR, "alignment_audit": ALIGNED},
        )
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(rejected.json()["code"], "IPD_NOT_ACCEPTED")
        self.assertIs(rejected.json()["interpretive_guardrail"]["numerical_proximity_proves_causality"], False)
        listed = self.client.post(
            "/mcp",
            headers=self.mcp,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertIn("benchmark_rct_and_negative_control", listed.text)
        called = self.client.post(
            "/mcp",
            headers=self.mcp,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "benchmark_rct_and_negative_control",
                    "arguments": {"synthetic_demo": "clean_null", "nct_id": "NCT01243424", "n": 400, "seed": 20261125},
                },
            },
        )
        self.assertEqual(called.status_code, 200, called.text)
        tool_body = called.json()["result"]
        self.assertFalse(tool_body.get("isError"), tool_body)
        tool_report = json.loads(tool_body["content"][0]["text"])
        self.assertIn("interpretive_guardrail", tool_report)
        self.assertIs(tool_report["interpretive_guardrail"]["numerical_proximity_proves_causality"], False)
        with mock.patch.dict(os.environ, {"EMULATRX_ENABLE_BENCHMARKING": "0"}):
            disabled = self.client.post(
                "/api/v1/tte/benchmark",
                headers=self.web,
                json={"synthetic_demo": "clean_null", "n": 200},
            )
            self.assertEqual(disabled.status_code, 503)
            self.assertEqual(disabled.json()["error_code"], "BENCHMARKING_FEATURE_DISABLED")
            mcp_off = self.client.post(
                "/mcp",
                headers=self.mcp,
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "benchmark_rct_and_negative_control",
                        "arguments": {"synthetic_demo": "clean_null"},
                    },
                },
            )
            self.assertEqual(mcp_off.status_code, 200, mcp_off.text)
            mcp_payload = json.loads(mcp_off.json()["result"]["content"][0]["text"])
            self.assertEqual(mcp_payload["error_code"], "BENCHMARKING_FEATURE_DISABLED")

    def _plant_run(self, run_id: str, status: dict) -> None:
        path = self.service.runner.root / run_id
        path.mkdir()
        (path / "status.json").write_text(json.dumps(status), encoding="utf-8")

    def test_http_and_mcp_reject_spoofed_alignment_bad_nct_and_baseline(self):
        blocked_id = "a" * 32
        self._plant_run(
            blocked_id,
            {
                "run_id": blocked_id,
                "state": "completed",
                "kind": "grace_period_ccw",
                "gate_status": "blocked",
                "target_rct": {
                    "nct_id": "NCT01243424",
                    "overall_alignment_status": "protocol_mismatched",
                    "population_alignment": "matched",
                    "arm_alignment_status": "arm_mismatched",
                    "outcome_alignment": "surrogate_or_mismatched",
                    "followup_alignment": "adequate",
                },
            },
        )
        spoof = {
            "run_id": blocked_id,
            "nct_id": "NCT01243424",
            "tte_result": {**RWD_HR, "gate_passed": True},
            "alignment_audit": {"overall_alignment": "aligned"},
        }
        blocked = self.client.post("/api/v1/tte/benchmark", headers=self.web, json=spoof)
        self.assertEqual(blocked.status_code, 200, blocked.text[:500])
        body = blocked.json()
        self.assertIn(
            body["gate_code"],
            {"BENCHMARK_BLOCKED_UPSTREAM_DIAGNOSTIC_FAILED", "BENCHMARK_BLOCKED_PROTOCOL_OR_ENDPOINT_MISMATCH"},
        )
        self.assertIs(body["consistency_conclusion_allowed"], False)
        self.assertIsNone(body["effect_comparison"])
        self.assertNotEqual(body["status"], "benchmarked")

        swapped_id = "b" * 32
        self._plant_run(
            swapped_id,
            {
                "run_id": swapped_id,
                "state": "completed",
                "kind": "grace_period_ccw",
                "gate_status": "passed",
                "gate_passed": True,
                "target_rct": _carolina_target(),
                "estimates": _PASSED_ESTIMATES,
            },
        )
        swapped = self.client.post(
            "/api/v1/tte/benchmark",
            headers=self.web,
            json={
                "run_id": swapped_id,
                "nct_id": "NCT02296125",
                "tte_result": RWD_HR,
                "alignment_audit": ALIGNED,
            },
        )
        self.assertEqual(swapped.status_code, 200, swapped.text[:500])
        swapped_body = swapped.json()
        self.assertEqual(swapped_body["gate_code"], "BENCHMARK_BLOCKED_PROTOCOL_OR_ENDPOINT_MISMATCH")
        self.assertIs(swapped_body["consistency_conclusion_allowed"], False)
        self.assertIsNone(swapped_body["effect_comparison"])

        mcp_spoof = self.client.post(
            "/mcp",
            headers=self.mcp,
            json={
                "jsonrpc": "2.0",
                "id": 41,
                "method": "tools/call",
                "params": {
                    "name": "benchmark_rct_and_negative_control",
                    "arguments": {
                        "run_id": blocked_id,
                        "nct_id": "NCT01243424",
                        "tte_result": {**RWD_HR, "gate_passed": True},
                        "alignment_audit": {"overall_alignment": "aligned"},
                    },
                },
            },
        )
        self.assertEqual(mcp_spoof.status_code, 200, mcp_spoof.text[:800])
        tool_body = mcp_spoof.json()["result"]
        self.assertFalse(tool_body.get("isError"), tool_body)
        tool_report = json.loads(tool_body["content"][0]["text"])
        self.assertIn(
            tool_report["gate_code"],
            {"BENCHMARK_BLOCKED_UPSTREAM_DIAGNOSTIC_FAILED", "BENCHMARK_BLOCKED_PROTOCOL_OR_ENDPOINT_MISMATCH"},
        )
        self.assertIs(tool_report["consistency_conclusion_allowed"], False)

        hostile = self.client.post(
            "/api/v1/tte/benchmark",
            headers=self.web,
            json={
                "nct_id": "NCT01243424",
                "study_json": {"nctId": "javascript:alert(1)", "hasResults": False},
                "tte_result": RWD_HR,
                "alignment_audit": ALIGNED,
            },
        )
        self.assertIn(hostile.status_code, {400, 422})
        self.assertEqual(hostile.json()["code"], "INVALID_NCT_ID")

        overflow = self.client.post(
            "/api/v1/tte/benchmark",
            headers=self.web,
            json={
                "nct_id": "NCT01243424",
                "tte_result": {
                    **RWD_HR,
                    "baseline_characteristics": {"female": {"proportion": 1.5}},
                },
                "alignment_audit": ALIGNED,
            },
        )
        self.assertIn(overflow.status_code, {400, 422}, overflow.text[:500])
        self.assertNotEqual(overflow.status_code, 500)
        self.assertEqual(overflow.json()["code"], "INVALID_BASELINE_SUMMARY")


class RequestBoundaryTests(unittest.TestCase):
    def test_embedded_patient_rows_are_rejected_before_benchmarking(self):
        with self.assertRaises(Exception) as caught:
            execute_benchmark_request({"patients": [{"age": 1}], "synthetic_demo": "clean_null"})
        self.assertEqual(caught.exception.code, "IPD_NOT_ACCEPTED")


class PersistedRunAndContractRegressionTests(unittest.TestCase):
    def test_spoofed_body_cannot_upgrade_persisted_gate_or_alignment(self):
        failed_gate = execute_benchmark_request(
            {
                "nct_id": "NCT01243424",
                "tte_result": {**RWD_HR, "gate_passed": True},
                "alignment_audit": {"overall_alignment": "aligned"},
            },
            run_record={
                "gate_status": "blocked",
                "gate_passed": True,
                "target_rct": _carolina_target(),
                "estimates": _PASSED_ESTIMATES,
            },
        )
        self.assertEqual(failed_gate["gate_code"], "BENCHMARK_BLOCKED_UPSTREAM_DIAGNOSTIC_FAILED")
        self.assertIs(failed_gate["consistency_conclusion_allowed"], False)
        self.assertIsNone(failed_gate["effect_comparison"])
        self.assertNotEqual(failed_gate["status"], "benchmarked")

        mismatched = execute_benchmark_request(
            {
                "nct_id": "NCT01243424",
                "tte_result": RWD_HR,
                "alignment_audit": ALIGNED,
            },
            run_record={
                "gate_status": "passed",
                "target_rct": _carolina_target(
                    overall_alignment_status="protocol_mismatched",
                    arm_alignment_status="arm_mismatched",
                    outcome_alignment="surrogate_or_mismatched",
                ),
                "estimates": _PASSED_ESTIMATES,
            },
        )
        self.assertEqual(mismatched["gate_code"], "BENCHMARK_BLOCKED_PROTOCOL_OR_ENDPOINT_MISMATCH")
        self.assertIs(mismatched["consistency_conclusion_allowed"], False)
        self.assertIsNone(mismatched["effect_comparison"])

    def test_stored_nct_cannot_be_benchmarked_against_a_different_trial(self):
        swapped = execute_benchmark_request(
            {
                "nct_id": "NCT02296125",
                "tte_result": RWD_HR,
                "alignment_audit": ALIGNED,
            },
            run_record={
                "gate_status": "passed",
                "target_rct": _carolina_target(),
                "estimates": _PASSED_ESTIMATES,
            },
        )
        self.assertEqual(swapped["gate_code"], "BENCHMARK_BLOCKED_PROTOCOL_OR_ENDPOINT_MISMATCH")
        self.assertIs(swapped["consistency_conclusion_allowed"], False)
        self.assertIsNone(swapped["effect_comparison"])
        self.assertNotEqual(swapped["status"], "benchmarked")

        same = execute_benchmark_request(
            {"nct_id": "NCT01243424"},
            run_record={
                "gate_status": "passed",
                "target_rct": _carolina_target(),
                "estimates": _PASSED_ESTIMATES,
            },
        )
        self.assertEqual(same["status"], "benchmarked")
        self.assertEqual(same["source_receipt"]["nct_id"], "NCT01243424")

    def test_sparse_negative_control_is_not_a_consistency_signoff(self):
        sparse = pd.DataFrame(
            {
                "treatment": [1, 1, 1, 1, 0, 0, 0, 0],
                "outcome_event": [0, 0, 0, 0, 0, 0, 0, 0],
                "negative_control_outcome": [1, 0, 0, 0, 1, 0, 0, 0],
            }
        )
        report = run_rct_benchmark_and_nco(
            RWD_HR,
            ALIGNED,
            nct_id="NCT01243424",
            cohort=sparse,
            weights=np.ones(len(sparse)),
            negative_control_outcome_col="negative_control_outcome",
            outcome_col="outcome_event",
        )
        self.assertEqual(report["negative_control"]["status"], "insufficient_events_inconclusive")
        self.assertIs(report["consistency_conclusion_allowed"], False)
        self.assertEqual(report["overall_validity_flag"], "nco_inconclusive_underpowered")
        self.assertIsNotNone(report["effect_comparison"])

    def test_hostile_nct_and_source_url_fail_closed(self):
        with self.assertRaises(ContractError) as caught:
            parse_clinicaltrials_results_study({"nctId": "javascript:alert(1)", "hasResults": False})
        self.assertEqual(caught.exception.code, "INVALID_NCT_ID")
        with self.assertRaises(ContractError) as caught_url:
            parse_clinicaltrials_results_study(
                {
                    "protocolSection": {"identificationModule": {"nctId": "NCT01243424"}},
                    "source_url": "javascript:alert(1)",
                    "hasResults": False,
                }
            )
        self.assertEqual(caught_url.exception.code, "INVALID_SOURCE_URL")
        with self.assertRaises(ContractError) as caught_curated:
            resolve_rct_public_results(
                "NCT01243424",
                curated_published_summary={
                    "effect_scale": "hr",
                    "point_estimate": 0.9,
                    "ci_lower": 0.8,
                    "ci_upper": 1.1,
                    "n_treatment": 100,
                    "n_control": 100,
                    "primary_outcome_label": "3P-MACE",
                    "source_citation": "Example citation",
                    "source_url": "javascript:alert(1)",
                },
            )
        self.assertEqual(caught_curated.exception.code, "INVALID_SOURCE_URL")

    def test_out_of_range_baseline_summaries_raise_structured_error(self):
        with self.assertRaises(ContractError) as proportion:
            compute_baseline_comparability(
                {"female": {"proportion": 1.5}},
                {"female": {"proportion": 0.4}},
            )
        self.assertEqual(proportion.exception.code, "INVALID_BASELINE_SUMMARY")
        with self.assertRaises(ContractError) as negative:
            compute_baseline_comparability(
                {"age": {"mean": 70.0, "sd": -1.0}},
                {"age": {"mean": 64.0, "sd": 9.5}},
            )
        self.assertEqual(negative.exception.code, "INVALID_BASELINE_SUMMARY")
        with self.assertRaises(ContractError) as below:
            compute_baseline_comparability(
                {"female": {"proportion": -0.1}},
                {"female": {"proportion": 0.4}},
            )
        self.assertEqual(below.exception.code, "INVALID_BASELINE_SUMMARY")


if __name__ == "__main__":
    unittest.main()
