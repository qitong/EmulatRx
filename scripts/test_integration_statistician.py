"""
End-to-end integration test for the mimic-free synthetic emulation pipeline.

Proves the synthetic data source flows through the REAL production pipeline with
NO database:

    Informatician.get_df_dict()   (dataframe_utils.py)
        -> the Statistician's causal-inference functions   (emulation_utils.py)
        -> an adjusted treatment hazard ratio

Ground truth baked into the synthetic cohort: treatment hazard ratio ~= 0.65.

Findings (recorded here so the pipeline's coupling is documented in-repo):
  * run_cox_model(...)  -> WORKS. Adjusted (multivariable) Cox on the synthetic
    cohort recovers HR ~= 0.70 with a 95% CI that covers the 0.65 ground truth.
  * perform_iptw(...) + calculate_hazard_ratios(...) -> WORKS. IPTW-weighted Cox
    recovers HR ~= 0.67; causallib's covariate_balance shows the strong
    confounders (ecog_ps, stage_iv) become better balanced after weighting.
  * perform_psm(...) -> WORKS after the check_balance_after_matching fix. It
    previously raised KeyError: 'mean' because check_balance_after_matching
    indexed a groupby(...).agg(['mean','std']) MultiIndex Series with the scalar
    'mean' (which resolved against the variable-name level under recent pandas);
    the fix selects the stat via the MultiIndex level, so PSM now runs and
    produces post-matching balance.
"""

import os
import sys

# --- Environment must be prepared BEFORE importing dataframe_utils, because that
# module reads DB_* via os.getenv at import time and imports matplotlib.pyplot. ---
os.environ.setdefault("MPLBACKEND", "Agg")  # no GUI backend during Cox plotting

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# No database anywhere in this test: drop any DB_* config so the synthetic
# CSV-cache path is the only thing that can be exercised.
for _k in ("DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"):
    os.environ.pop(_k, None)

import numpy as np
import pytest

from dataframe_utils import Informatician
import emulation_utils as eu

TRIAL_ID = "NCT00000099"
TRIALS_DIR = os.path.join(ROOT, "trials")
GROUND_TRUTH_HR = 0.65

# Covariate split requested by the plan.
CATEGORICAL = ["sex_male", "ecog_ps", "smoking", "histology_adeno", "stage_iv", "egfr_mut"]
CONTINUOUS = ["age", "pdl1_tps", "hemoglobin", "albumin", "ldh", "neutrophils"]
TREATMENT = ["treatment"]
OUTCOME = ["event"]
DURATION = ["duration_day"]


@pytest.fixture(scope="module")
def df_cd():
    """Run the real Informatician (no llm, no DB) and return (df, cols_dict)."""
    # Belt-and-suspenders: ensure DB_* stays unset even if another test set it.
    for k in ("DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"):
        os.environ.pop(k, None)
    informatician = Informatician(
        trialID=TRIAL_ID,
        trial_info={},
        llm=None,
        dataset="synthetic",
        root_dir=TRIALS_DIR,
    )
    df, cols_dict = informatician.get_df_dict()
    return df, cols_dict


def test_informatician_returns_without_db(df_cd):
    """Informatician.get_df_dict returns (df, cols_dict) from the CSV cache,
    with no llm and no database."""
    df, cols_dict = df_cd

    assert len(df) > 0, "Informatician returned an empty dataframe"
    assert cols_dict["patient_id"] == "stay_id"

    # Treatment / outcome / duration columns must be present and usable.
    treatment_col = cols_dict["treatment_cols"]          # "treatment"
    outcome_col = cols_dict["outcome_cols"][0]           # "event"
    duration_col = cols_dict["duration_cols"][0]         # "duration_day"
    for col in (treatment_col, outcome_col, duration_col):
        assert col in df.columns, f"expected column {col!r} missing from df"

    # Sanity on the treatment-effect substrate.
    assert set(df[treatment_col].unique()) <= {0, 1}
    assert set(df[outcome_col].unique()) <= {0, 1}
    assert (df[duration_col] > 0).all()

    # All requested covariates survived get_df_dict's space-normalisation.
    for cov in CATEGORICAL + CONTINUOUS:
        assert cov in df.columns, f"covariate {cov!r} missing after get_df_dict"


def _unadjusted_hr(df):
    """Crude unadjusted treatment HR (module reuse convenience for reporting)."""
    cph, _ = eu.calculate_hazard_ratios(df, TREATMENT, OUTCOME, DURATION, [], [])
    return cph.summary.loc["treatment", "exp(coef)"]


def test_statistician_recovers_effect(df_cd):
    """The repo's OWN causal-inference functions, fed the synthetic (df, cols_dict),
    recover an adjusted treatment hazard ratio consistent with the 0.65 ground
    truth -- with no database in the loop."""
    df, cols_dict = df_cd

    recovered = {}

    # ---- Repo function #1: run_cox_model (adjusted multivariable Cox) ----
    cox_summary, _fig, cph_info = eu.run_cox_model(
        df, TREATMENT, OUTCOME, DURATION,
        outcome_definition=OUTCOME,
        categorical_covariates=CATEGORICAL,
        continuous_covariates=CONTINUOUS,
        state={},
    )
    cox_hr = cox_summary["exp(coef)"]["treatment"]
    cox_lo = cox_summary["exp(coef) lower 95%"]["treatment"]
    cox_hi = cox_summary["exp(coef) upper 95%"]["treatment"]
    recovered["run_cox_model"] = (cox_hr, cox_lo, cox_hi)

    assert 0.45 <= cox_hr <= 0.95, f"adjusted Cox HR {cox_hr:.4f} implausible"
    assert cox_lo <= GROUND_TRUTH_HR <= cox_hi, (
        f"adjusted Cox 95% CI ({cox_lo:.4f}, {cox_hi:.4f}) "
        f"does not cover ground-truth HR {GROUND_TRUTH_HR}"
    )
    assert isinstance(cph_info, str) and "Hazard Ratio" in cph_info

    # ---- Repo function #2: perform_iptw -> calculate_hazard_ratios ----
    results = {}
    iptw_out = eu.perform_iptw(
        df, CATEGORICAL, CONTINUOUS, [],
        TREATMENT, OUTCOME,
        outcome_definition=OUTCOME,
        break_flag=False,
        results=results,
        state={},
    )
    df_weighted, iptw_info, results, iptw_break = iptw_out
    assert iptw_break is False, f"IPTW flagged residual imbalance: {iptw_info}"
    assert "iptw_weight" in df_weighted.columns

    cph_iptw, _ = eu.calculate_hazard_ratios(
        df_weighted, TREATMENT, OUTCOME, DURATION, CATEGORICAL, CONTINUOUS
    )
    iptw_hr = cph_iptw.summary.loc["treatment", "exp(coef)"]
    iptw_lo = cph_iptw.summary.loc["treatment", "exp(coef) lower 95%"]
    iptw_hi = cph_iptw.summary.loc["treatment", "exp(coef) upper 95%"]
    recovered["perform_iptw+calculate_hazard_ratios"] = (iptw_hr, iptw_lo, iptw_hi)

    assert 0.45 <= iptw_hr <= 0.95, f"IPTW-weighted Cox HR {iptw_hr:.4f} implausible"
    assert iptw_lo <= GROUND_TRUTH_HR <= iptw_hi, (
        f"IPTW-weighted Cox 95% CI ({iptw_lo:.4f}, {iptw_hi:.4f}) "
        f"does not cover ground-truth HR {GROUND_TRUTH_HR}"
    )

    # ---- Post-adjustment balance from the repo function's own machinery ----
    # perform_iptw stashes causallib's evaluation; its covariate_balance table
    # carries both 'unweighted' (before) and 'weighted' (after) absolute SMDs.
    covariate_balance = results["All IPTW results"].evaluated_metrics.covariate_balance
    improved = []
    for confounder in ("ecog_ps", "stage_iv"):
        before = covariate_balance.loc[confounder, "unweighted"]
        after = covariate_balance.loc[confounder, "weighted"]
        if after < before:
            improved.append((confounder, before, after))
    assert improved, (
        "IPTW did not improve balance on any strong confounder "
        f"(ecog_ps/stage_iv):\n{covariate_balance}"
    )

    # Emit a human-readable summary (visible with `pytest -s` / on failure).
    print("\n[integration] recovered adjusted treatment HR (ground truth ~0.65):")
    print(f"  unadjusted Cox                 HR={_unadjusted_hr(df):.4f}")
    for name, (hr, lo, hi) in recovered.items():
        print(f"  {name:<32} HR={hr:.4f}  95% CI=({lo:.4f}, {hi:.4f})")
    for confounder, before, after in improved:
        print(f"  balance {confounder:<10} |SMD| {before:.4f} -> {after:.4f} (IPTW)")


def test_perform_psm_runs(df_cd):
    """perform_psm runs cleanly after the check_balance_after_matching MultiIndex fix.

    Previously raised KeyError: 'mean' at emulation_utils.py check_balance_after_matching
    (indexing a groupby.agg(['mean','std']) MultiIndex Series with the scalar 'mean')
    under recent pandas. The fix selects the stat via the MultiIndex level.
    """
    df, _ = df_cd
    results, state = {}, {}
    out = eu.perform_psm(
        df, CATEGORICAL, CONTINUOUS, [],
        TREATMENT, OUTCOME,
        outcome_definition=OUTCOME,
        break_flag=False,
        results=results,
        state=state,
    )
    # Completes without the old KeyError and produces balancing output.
    assert out is not None
    assert any("Balance" in str(k) for k in results), results.keys()
