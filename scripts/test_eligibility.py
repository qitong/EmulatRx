"""Eligibility / EC-optimization data path for the realistic synthetic cohort.

Verifies that the generator produces a raw patient POOL that includes ineligible
patients, derives the analysis cohort by applying operational eligibility criteria,
and emits the per-criterion matched-patient files that the eligibility-criteria
optimization analysis consumes.
"""
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import synthetic_config as cfg  # noqa: E402

TRIAL = "NCT00000099"
CACHE = os.path.join(ROOT, "trials", TRIAL, "dataframes")
PY = sys.executable


def _gen():
    subprocess.run(
        [PY, os.path.join(ROOT, "scripts", "generate_synthetic_oncology.py"),
         "--trial-id", TRIAL, "--no-postgres"],
        check=True, cwd=ROOT,
    )


def test_raw_pool_larger_than_cohort():
    _gen()
    raw = pd.read_csv(os.path.join(CACHE, "big_df.csv"))
    cohort = pd.read_csv(os.path.join(CACHE, "dataframe.csv"))
    assert len(raw) > len(cohort), "big_df (raw pool) must be larger than the eligible cohort"


def test_per_criterion_files_and_sql_exist():
    for name in list(cfg.INCLUSION_CRITERIA) + list(cfg.EXCLUSION_CRITERIA):
        assert os.path.exists(os.path.join(CACHE, f"matched_patients_ec_{name}.csv")), name
    for name in cfg.INCLUSION_CRITERIA:
        assert os.path.exists(os.path.join(ROOT, "trials", TRIAL, "include", f"{name}.sql"))
    for name in cfg.EXCLUSION_CRITERIA:
        assert os.path.exists(os.path.join(ROOT, "trials", TRIAL, "exclude", f"{name}.sql"))


def test_cohort_satisfies_eligibility():
    raw = pd.read_csv(os.path.join(CACHE, "big_df.csv"))
    cohort_ids = set(pd.read_csv(os.path.join(CACHE, "matched_patients.csv"))["stay_id"])
    cohort = raw[raw["stay_id"].isin(cohort_ids)]
    # inclusion: advanced (stage 3/4) + ECOG 0-2; exclusion: no prior systemic therapy
    assert set(cohort["ecog_ps"].unique()) <= {0, 1, 2}
    assert set(cohort["stage"].unique()) <= {3, 4}
    assert (cohort["prior_systemic_therapy"] == 0).all()
    # raw pool DOES contain ineligible patients (otherwise the criteria are vacuous)
    assert (raw["ecog_ps"] > 2).any()
    assert (raw["stage"] < 3).any()
    assert (raw["prior_systemic_therapy"] == 1).any()


def test_ec_optimization_powerset_builds():
    for k in ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"]:
        os.environ.pop(k, None)
    from dataframe_utils import Informatician
    info = Informatician(trialID=TRIAL, trial_info={}, llm=None, dataset="synthetic",
                         root_dir=os.path.join(ROOT, "trials"))
    info.get_df_dict()
    state = info.get_permuted_ec_for_optimize()
    ps = state["variables"]["ec_patients_powerset_dataframe"]
    n_crit = len(cfg.INCLUSION_CRITERIA) + len(cfg.EXCLUSION_CRITERIA)
    assert len(ps) == 2 ** n_crit, f"powerset should have 2^{n_crit} rows, got {len(ps)}"
    # applying ALL criteria yields the smallest set == the derived cohort
    cohort_n = len(pd.read_csv(os.path.join(CACHE, "matched_patients.csv")))
    assert ps["MatchedPatientsNumber"].min() == cohort_n
