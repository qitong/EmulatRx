"""White-box oracles for the clinician-facing 3-tier evidence ladder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from emulatrx_tte.ccw import generate_synthetic_ccw_cohort, run_grace_period_ccw
from emulatrx_tte.clinical_ladder import build_binary_exposure_ladder
from emulatrx_tte.cohort_ingest import (
    detect_missing_clinical_semantics,
    profile_cohort_dataframe,
    read_tabular_cohort_bytes,
)


def _confounded_protective_cohort(n: int = 2400, seed: int = 11, horizon: float = 365.0):
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    comorbidity = rng.binomial(1, 0.35, size=n).astype(np.float64)
    treated_prob = expit(-0.1 + 1.7 * risk + 0.8 * comorbidity)
    arm = (rng.random(n) < treated_prob).astype(np.int64)
    rate1 = 0.0016 * np.exp(1.35 * risk + 0.7 * comorbidity - 0.95 * arm)
    rate2 = 0.00035 * np.exp(0.25 * risk)
    t1 = rng.exponential(1.0 / rate1)
    t2 = rng.exponential(1.0 / rate2)
    censor = rng.uniform(horizon * 0.55, horizon * 1.4, size=n)
    time = np.minimum(np.minimum(t1, t2), censor)
    event = np.zeros(n, dtype=np.int64)
    event[(t1 <= t2) & (t1 <= censor)] = 1
    event[(t2 < t1) & (t2 <= censor)] = 2
    X = np.column_stack([risk, comorbidity])
    return time, event, arm, X, ["risk", "comorbidity"], horizon


def _competing_heavy_cohort(n: int = 1800, seed: int = 19, horizon: float = 365.0):
    rng = np.random.default_rng(seed)
    age = rng.normal(size=n)
    arm = rng.binomial(1, 0.5, size=n).astype(np.int64)
    rate1 = 0.0009 * np.exp(0.15 * age)
    rate2 = 0.0045 * np.exp(0.2 * age)
    t1 = rng.exponential(1.0 / rate1)
    t2 = rng.exponential(1.0 / rate2)
    censor = np.full(n, horizon * 1.2)
    time = np.minimum(np.minimum(t1, t2), censor)
    event = np.zeros(n, dtype=np.int64)
    event[(t1 <= t2) & (t1 <= censor)] = 1
    event[(t2 < t1) & (t2 <= censor)] = 2
    return time, event, arm, np.column_stack([age]), ["age"], horizon


class ClinicalLadderOracleTests(unittest.TestCase):
    def test_baseline_confounding_crude_harm_removed_by_ps_and_overlap(self):
        time, event, arm, X, names, horizon = _confounded_protective_cohort()
        bundle = build_binary_exposure_ladder(
            time,
            event,
            arm,
            X,
            names,
            horizon,
            grouping="raw_exposure",
            design_family="static",
            compute_overlap_tier3=True,
        )
        tier1 = bundle["clinical_comparison_ladder"]["tier_1_unadjusted_crude"]
        tier2 = bundle["clinical_comparison_ladder"]["tier_2_conventional_ps"]
        tier3 = bundle["clinical_comparison_ladder"]["tier_3_target_trial_emulation"]
        self.assertGreater(tier1["crude_rd_cif"], 0.0, msg=json.dumps(tier1, ensure_ascii=False))
        self.assertGreaterEqual(tier2["psm_1to1"]["matched_pairs"], 10)
        self.assertLess(tier2["psm_1to1"]["psm_rd"], 0.0)
        self.assertLess(tier2["simple_iptw"]["iptw_rd"], 0.0)
        self.assertLess(tier3["tte_rd"], 0.0)
        self.assertEqual(tier3["tte_retained_n"], int(arm.size))
        self.assertGreater(tier2["psm_1to1"]["discarded_subjects_pct"], 0.0)
        self.assertIn("风险差", bundle["what_tte_controlled_beyond_psm"]["headline_progression_zh"])

    def test_grace_period_null_immortal_time_survives_psm_but_not_ccw(self):
        protocol = json.loads(
            Path("examples/tte_grace_period_ccw_v1.json").read_text(encoding="utf-8")
        )
        protocol["diagnostic_gates"]["max_post_grace_smd"] = 0.20
        dgp = {"true_log_hr_event1": 0.0, "true_log_hr_event2": 0.0}
        generated = generate_synthetic_ccw_cohort(protocol, seed=42, n_cases=2000, dgp_params=dgp)
        result, code = run_grace_period_ccw(
            protocol,
            generated["cohort_df"],
            seed=42,
            bootstrap_iterations=20,
            person_period_covariates=generated["person_period_df"],
        )
        self.assertEqual(code, 0, msg=str(result.get("gate_failures")))
        estimates = result["estimates"]
        ladder = estimates["clinical_comparison_ladder"]
        tier1 = ladder["tier_1_unadjusted_crude"]
        tier2 = ladder["tier_2_conventional_ps"]
        tier3 = ladder["tier_3_target_trial_emulation"]
        self.assertLessEqual(tier1["crude_rd_cif"], -0.10)
        self.assertLessEqual(tier2["psm_1to1"]["psm_rd"], -0.10)
        self.assertLessEqual(tier2["simple_iptw"]["iptw_rd"], -0.10)
        self.assertLess(tier2["psm_1to1"]["post_psm_max_abs_smd"], 0.10)
        self.assertAlmostEqual(tier3["tte_rd"], 0.0, delta=0.05)
        immortal = next(
            item
            for item in estimates["what_tte_controlled_beyond_psm"]["extra_tte_controls"]
            if item["category"] == "immortal_time_bias_elimination"
        )
        self.assertTrue(immortal["active"])
        self.assertGreater(immortal["quantitative_impact"]["absolute_false_benefit_removed"], 0.05)
        self.assertIn("不朽时间", immortal["clinical_explanation_zh"])

    def test_competing_risk_km_overestimates_cif(self):
        time, event, arm, X, names, horizon = _competing_heavy_cohort()
        bundle = build_binary_exposure_ladder(
            time,
            event,
            arm,
            X,
            names,
            horizon,
            grouping="raw_exposure",
            design_family="static",
            compute_overlap_tier3=True,
            context={"competing_events": int(np.sum(event == 2))},
        )
        tier1 = bundle["clinical_comparison_ladder"]["tier_1_unadjusted_crude"]
        self.assertGreater(tier1["unadjusted_km_risk_treated"], tier1["unadjusted_cif_treated"] + 0.05)
        self.assertGreater(tier1["unadjusted_km_risk_comparator"], tier1["unadjusted_cif_comparator"] + 0.05)
        item = next(
            row
            for row in bundle["what_tte_controlled_beyond_psm"]["extra_tte_controls"]
            if row["category"] == "competing_risks_correction"
        )
        self.assertTrue(item["active"])
        self.assertGreater(item["quantitative_impact"]["competing_events"], 0)
        self.assertGreater(item["quantitative_impact"]["km_overestimation_treated"], 0.05)
        self.assertGreater(item["quantitative_impact"]["km_overestimation_comparator"], 0.05)

    def test_gb18030_opaque_codes_quarantine_pii_and_ask_semantics(self):
        frame = pd.DataFrame(
            {
                "subject_id": [f"H{i:04d}" for i in range(12)],
                "treatment": [i % 2 for i in range(12)],
                "time_day": [30 + i for i in range(12)],
                "event": [0, 1, 0, 1, 0, 0, 1, 0, 2, 1, 0, 0],
                "age": [50 + i for i in range(12)],
                "姓名": [f"张三-SECRET-{i}" for i in range(12)],
            }
        )
        raw = frame.to_csv(index=False).encode("gb18030")
        decoded, _delimiter, encoding = read_tabular_cohort_bytes(raw, "ward.csv")
        self.assertEqual(encoding, "gb18030")
        self.assertIn("姓名", decoded.columns)
        profile = profile_cohort_dataframe(decoded, filename="ward.csv")
        semantics = detect_missing_clinical_semantics(decoded, profile)
        blob = json.dumps(semantics, ensure_ascii=False)
        self.assertNotIn("SECRET", blob)
        self.assertIn("姓名", semantics["pii_columns_quarantined"])
        fields = {item["field"] for item in semantics["missing_clinical_semantics"]}
        self.assertIn("treatment", fields)
        self.assertIn("outcome", fields)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ward.csv"
            path.write_bytes(raw)
            reloaded = path.read_bytes()
            self.assertTrue(reloaded.startswith(raw[:4]) or encoding == "gb18030")


if __name__ == "__main__":
    unittest.main()
