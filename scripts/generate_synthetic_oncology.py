#!/usr/bin/env python
"""Generate a confounded synthetic advanced-NSCLC dataset and emit the exact
DataframeBuilder CSV cache that EmulatRx's DataframeBuilder/Informatician read.

Realistic-cohort model: a raw patient POOL (``cfg.N_RAW``) is drawn that includes
patients who FAIL eligibility (early stage, ECOG 3-4, prior systemic therapy). The
analysis cohort is then DERIVED by applying the operational eligibility criteria in
``synthetic_config`` (INCLUSION_CRITERIA / EXCLUSION_CRITERIA). Each criterion's
matched-patient set and the derived cohort are written to disk, so the
eligibility-criteria-optimization analysis (which permutes over per-criterion patient
sets) has real inputs — not an eligible-by-construction cohort.

Baseline covariates come from literature-typical advanced-NSCLC distributions
(``synthetic_config.py``); treatment is assigned via a confounded propensity model;
Weibull survival times have a hazard driven by treatment (conditional HR =
``cfg.HR_TARGET``) plus covariates. Everything is written under
``trials/<trial>/`` in every form the DataframeBuilder cache checks, so downstream
emulation short-circuits to these files instead of touching Postgres/MIMIC.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Make ``import synthetic_config`` work regardless of the caller's cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import synthetic_config as cfg  # noqa: E402


def _sample_covariate(spec, n):
    """Draw ``n`` samples for a single covariate given its spec tuple."""
    kind = spec[0]
    if kind == "normal":
        _, mean, sd, lo, hi = spec
        return np.clip(np.random.normal(mean, sd, n), lo, hi)
    if kind == "lognormal":
        _, meanlog, sdlog, lo, hi = spec
        return np.clip(np.random.lognormal(meanlog, sdlog, n), lo, hi)
    if kind == "bin":
        _, p = spec
        return np.random.binomial(1, p, n).astype(int)
    if kind == "cat":
        _, level_probs = spec
        levels = list(level_probs.keys())
        probs = np.array(list(level_probs.values()), dtype=float)
        probs = probs / probs.sum()
        return np.random.choice(levels, size=n, p=probs).astype(int)
    raise ValueError(f"Unknown covariate spec kind: {kind!r}")


def _zscore(x):
    x = np.asarray(x, dtype=float)
    std = x.std()
    if std == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def _apply_predicate(df, predicate):
    """Evaluate a (column, op, values) eligibility predicate over ``df``."""
    col, op, val = predicate
    s = df[col]
    if op == "in":
        return s.isin(val)
    if op == "==":
        return s == val
    if op == "!=":
        return s != val
    if op == ">=":
        return s >= val
    if op == "<=":
        return s <= val
    if op == ">":
        return s > val
    if op == "<":
        return s < val
    raise ValueError(f"Unknown eligibility op: {op!r}")


def generate(trial_id):
    """Draw the raw pool and return (raw_df, col_dict, covariate_names, elig_var_names)."""
    np.random.seed(cfg.SEED)
    n = cfg.N_RAW

    covariate_names = list(cfg.COVARIATES)
    cov = {name: _sample_covariate(spec, n) for name, spec in cfg.COVARIATES.items()}

    # Eligibility-determining variables (extra columns beyond the analysis covariates).
    elig_var_names = list(cfg.ELIGIBILITY_VARS)
    elig = {name: _sample_covariate(spec, n) for name, spec in cfg.ELIGIBILITY_VARS.items()}

    # Overrides so the raw pool contains ineligible patients:
    #  - ecog_ps spans 0-4 (cohort keeps 0-2 via the ECOG criterion)
    #  - stage_iv is DERIVED from stage (= 1 iff stage 4), consistent with `stage`
    cov["ecog_ps"] = _sample_covariate(("cat", cfg.ECOG_RAW), n)
    cov["stage_iv"] = (elig["stage"] == 4).astype(int)

    # z-scored versions for any "_z" term in the propensity / outcome models.
    zcols = {}
    for model in (cfg.PROPENSITY, cfg.OUTCOME_BETAS):
        for key in model:
            if key.endswith("_z"):
                zcols[key] = _zscore(cov[key[:-2]])

    def feature(key):
        if key.endswith("_z"):
            return zcols[key]
        return np.asarray(cov[key], dtype=float)

    # --- Confounded treatment assignment (propensity model) -----------------
    lin = np.full(n, float(cfg.PROPENSITY.get("intercept", 0.0)))
    for key, weight in cfg.PROPENSITY.items():
        if key == "intercept":
            continue
        lin = lin + weight * feature(key)
    p_treat = np.clip(1.0 / (1.0 + np.exp(-lin)), 0.05, 0.95)
    treatment = np.random.binomial(1, p_treat).astype(int)

    # --- Weibull survival with treatment + covariate-driven hazard ----------
    k = cfg.WEIBULL_SHAPE
    linpred = np.log(cfg.HR_TARGET) * treatment
    for key, beta in cfg.OUTCOME_BETAS.items():
        linpred = linpred + beta * feature(key)
    lam = cfg.BASELINE_MEDIAN_SURV_DAYS / (np.log(2) ** (1.0 / k))
    u = np.random.uniform(0.0, 1.0, n)
    t_event = lam * ((-np.log(u)) * np.exp(-linpred)) ** (1.0 / k)
    censor = np.random.exponential(cfg.MEDIAN_FOLLOWUP_DAYS / np.log(2), n)
    duration_day = np.maximum(np.minimum(t_event, censor), 1.0)
    event = (t_event <= censor).astype(int)

    treatment_time_delta = np.where(treatment == 1, 0.0, np.nan)
    stay_id = 900000 + np.arange(n)
    subject_id = 800000 + np.arange(n)

    ordered = {
        "stay_id": stay_id,
        "subject_id": subject_id,
        "treatment": treatment,
        "treatment_time_delta": treatment_time_delta,
        "event": event,
        "duration_day": duration_day,
    }
    for name in covariate_names:
        ordered[name] = cov[name]
    for name in elig_var_names:
        ordered[name] = elig[name]
    raw_df = pd.DataFrame(ordered)

    col_dict = {
        "covariate_cols": covariate_names,
        "treatment_cols": "treatment",
        "outcome_cols": ["event"],
        "duration_cols": ["duration_day"],
        "treatment_time_cols": "treatment_time_delta",
        "patient_id": "stay_id",
    }
    return raw_df, col_dict, covariate_names, elig_var_names


def compute_eligibility(raw_df):
    """Apply the operational eligibility criteria to the raw pool.

    Returns (cohort_ids, criteria) where criteria maps criterion name ->
    (kind, patient_id_set): for 'include' the patients satisfying the criterion,
    for 'exclude' the patients hit by the exclusion (i.e. those to be removed).
    Cohort = (intersection of inclusion sets) - (union of exclusion sets).
    """
    ids = raw_df["stay_id"]
    criteria = {}
    include_sets = []
    for name, pred in cfg.INCLUSION_CRITERIA.items():
        s = set(ids[_apply_predicate(raw_df, pred)].tolist())
        criteria[name] = ("include", s)
        include_sets.append(s)
    exclude_union = set()
    for name, pred in cfg.EXCLUSION_CRITERIA.items():
        s = set(ids[_apply_predicate(raw_df, pred)].tolist())
        criteria[name] = ("exclude", s)
        exclude_union |= s
    eligible = set.intersection(*include_sets) if include_sets else set(ids.tolist())
    cohort_ids = sorted(eligible - exclude_union)
    return cohort_ids, criteria


_ANALYSIS_COLS = ["stay_id", "subject_id", "treatment", "treatment_time_delta",
                  "event", "duration_day"]


def write_cache(raw_df, cohort_ids, criteria, col_dict, covariate_names,
                elig_var_names, trial_id):
    root = os.path.dirname(_SCRIPT_DIR)
    trial_dir = os.path.join(root, "trials", trial_id)
    df_dir = os.path.join(trial_dir, "dataframes")
    os.makedirs(df_dir, exist_ok=True)
    for sub in ("include", "exclude", "treatment", "outduration", "adverseevents"):
        os.makedirs(os.path.join(trial_dir, sub), exist_ok=True)

    # The analysis cohort: the raw pool filtered to eligible patients, standard columns.
    cohort_df = raw_df[raw_df["stay_id"].isin(cohort_ids)][
        _ANALYSIS_COLS + covariate_names].reset_index(drop=True)

    # Primary contract files (the analysis cohort).
    cohort_df.to_csv(os.path.join(df_dir, "dataframe.csv"), index=False)
    with open(os.path.join(df_dir, "col_dict.json"), "w") as fh:
        json.dump(col_dict, fh, indent=4)

    # big_df is the RAW POOL (all patients incl. ineligible) so EC-optimization can
    # re-derive HRs for different eligibility subsets. Extra eligibility columns are
    # carried along (clean_data selects only covariates/treatment/outcome/duration/id).
    raw_df.to_csv(os.path.join(df_dir, "big_df.csv"), index=False)
    raw_df[["stay_id", "subject_id"] + covariate_names + elig_var_names].to_csv(
        os.path.join(df_dir, "merged_X.csv"), index=False)
    raw_df[["stay_id", "treatment", "treatment_time_delta"]].to_csv(
        os.path.join(df_dir, "treatment_df.csv"), index=False)
    raw_df[["stay_id", "event", "duration_day"]].to_csv(
        os.path.join(df_dir, "outduration_df.csv"), index=False)

    # Derived cohort membership.
    pd.DataFrame({"stay_id": cohort_ids}).to_csv(
        os.path.join(df_dir, "matched_patients.csv"), index=False)

    # Per-criterion matched patients + a placeholder .sql (DataframeBuilder keys
    # criteria by the .sql filename; the EC-optimization path reads the CSVs).
    for name, (kind, patient_set) in criteria.items():
        pd.DataFrame({"stay_id": sorted(patient_set)}).to_csv(
            os.path.join(df_dir, f"matched_patients_ec_{name}.csv"), index=False)
        col, op, val = (cfg.INCLUSION_CRITERIA if kind == "include"
                        else cfg.EXCLUSION_CRITERIA)[name]
        sql = (f"-- synthetic {kind} criterion: {col} {op} {val}\n"
               f"SELECT stay_id FROM synthetic.cohort WHERE {col} {op} {val!r};\n")
        with open(os.path.join(trial_dir, kind, f"{name}.sql"), "w") as fh:
            fh.write(sql)

    return df_dir, cohort_df


def load_postgres(df, covariate_names):
    """Optionally load the analysis cohort into Postgres under a ``synthetic``
    schema. Wholly non-fatal: any failure (no DB, refused connection, missing
    psycopg2/sqlalchemy, or missing DB_* env) prints a warning and returns without
    raising, so the CSV cache (already written) remains the source of truth.
    """
    try:
        from sqlalchemy import create_engine, text

        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        database = os.getenv("DB_NAME")
        missing = [k for k, v in (
            ("DB_USER", user), ("DB_PASSWORD", password), ("DB_HOST", host),
            ("DB_PORT", port), ("DB_NAME", database)) if not v]
        if missing:
            raise RuntimeError(f"missing env {', '.join(missing)}")

        engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS synthetic;"))
        df.to_sql("cohort", engine, schema="synthetic", if_exists="replace", index=False)
        df[["stay_id", "subject_id"] + covariate_names].to_sql(
            "covariates", engine, schema="synthetic", if_exists="replace", index=False)
        df[["stay_id", "treatment", "treatment_time_delta"]].to_sql(
            "treatment", engine, schema="synthetic", if_exists="replace", index=False)
        df[["stay_id", "event", "duration_day"]].to_sql(
            "outcome", engine, schema="synthetic", if_exists="replace", index=False)
        print(f"[synthetic] postgres load ok: wrote {len(df)} rows to synthetic.cohort "
              f"(+ covariates/treatment/outcome) on {host}:{port}/{database}")
    except Exception as exc:  # noqa: BLE001 - deliberately catch-all, non-fatal
        print(f"[synthetic] postgres load skipped: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", default="NCT00000099",
                        help="Synthetic trial id (cache dir under trials/).")
    parser.add_argument("--postgres", action="store_true",
                        help="Also load the synthetic cohort into Postgres "
                             "(schema 'synthetic'). Non-fatal: skipped with a "
                             "warning if no DB is reachable.")
    parser.add_argument("--no-postgres", action="store_true",
                        help="Skip the optional Postgres load (default). Retained "
                             "for backward compatibility; the CSV cache is always "
                             "written regardless.")
    args = parser.parse_args()

    raw_df, col_dict, covariate_names, elig_var_names = generate(args.trial_id)
    cohort_ids, criteria = compute_eligibility(raw_df)
    df_dir, cohort_df = write_cache(raw_df, cohort_ids, criteria, col_dict,
                                    covariate_names, elig_var_names, args.trial_id)

    crit_summary = ", ".join(f"{name}({kind})={len(s)}"
                             for name, (kind, s) in criteria.items())
    print(f"[synthetic] trial={args.trial_id} raw_pool={len(raw_df)} "
          f"cohort={len(cohort_df)} "
          f"event_rate={float(cohort_df['event'].mean()):.3f} "
          f"treated_frac={float(cohort_df['treatment'].mean()):.3f} -> {df_dir}")
    print(f"[synthetic] eligibility criteria: {crit_summary}")

    if args.postgres and not args.no_postgres:
        load_postgres(cohort_df, covariate_names)


if __name__ == "__main__":
    main()
