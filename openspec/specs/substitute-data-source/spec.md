# substitute-data-source (delta)

## ADDED Requirements

### Requirement: Synthetic oncology data source satisfies the Informatician interface
The system SHALL provide a synthetic real-world-data source for an advanced-NSCLC cohort that satisfies the existing `DataframeBuilder`/`Informatician` runtime contract without requiring MIMIC-IV. The source SHALL expose a patient table keyed on `stay_id` with columns `treatment` (0/1), `treatment_time_delta`, an outcome column and a duration column for survival, and baseline oncology covariates, plus a `col_dict` describing `patient_id`, `treatment_cols`, `outcome_cols`, `duration_cols`, `treatment_time_cols`, and `covariate_cols`.

#### Scenario: Cohort table matches the runtime contract
- **WHEN** the synthetic generator runs for the target trial id
- **THEN** it writes `trials/<id>/dataframes/dataframe.csv` whose columns are `stay_id, subject_id, treatment, treatment_time_delta, <outcome>, <duration>, <covariates...>`
- **AND** it writes `col_dict.json` whose `patient_id` is `stay_id`, `treatment_cols` is `treatment`, and `outcome_cols`/`duration_cols` reference the generated survival columns

#### Scenario: Confounded treatment effect is present
- **WHEN** the cohort is generated
- **THEN** treatment assignment depends on covariates (a covariate-dependent propensity), and survival time depends on both treatment and covariates with a target hazard ratio
- **AND** a Cox model fit on the raw generated data recovers the target hazard ratio within its confidence interval

### Requirement: Pipeline runs on the synthetic source without MIMIC-IV
The system SHALL allow `Informatician.get_df_dict` to return `(df, col_dict)` for the synthetic trial and the Statistician to run covariate balancing and survival analysis on it, with no connection to a MIMIC-IV database.

#### Scenario: End-to-end via CSV cache without a database
- **WHEN** the synthetic cache exists and `Informatician.get_df_dict(<id>)` is called with no database reachable
- **THEN** it returns `(df, col_dict)` from the cache without raising
- **AND** feeding the result to the Statistician's PSM/IPTW/Cox produces post-matching standardized mean differences below 0.1 on key covariates and an adjusted hazard ratio near the target

#### Scenario: Postgres path rebuilds the same cohort from SQL
> DEFERRED (partial): `--postgres` loads the `synthetic` schema, but the pure from-SQL rebuild with the CSV cache cleared is not yet fully wired — `get_dataframe`'s `col_dict` construction (dataframe_utils.py:502) still hardcodes MIMIC column names, and synthetic few-shot SQL was not added. Tracked as a follow-up. The CSV-cache route below is the verified path. See docs/superpowers/reports/2026-07-16-mimic-free-emulation-verify.md.
- **WHEN** the CSV cache is cleared and `DataframeBuilder(dataset='synthetic')` runs against the loaded `synthetic` Postgres schema
- **THEN** `get_dataframe` rebuilds an equivalent `dataframe.csv` from SQL queries keyed on `stay_id`

### Requirement: MIMIC-specific assembly is guarded by dataset
The system SHALL NOT run the MIMIC-specific covariate/treatment munging when `dataset='synthetic'`.

#### Scenario: Synthetic branch bypasses MIMIC column munging
- **WHEN** `dataset='synthetic'` is selected
- **THEN** the MIMIC-only column logic in `_build_dataframe_X`/`_build_dataframe_treatment` (e.g. `abs_basophils`, `sofa`, `_max`/`_min` rules, `intime`/`outtime`) is not executed
- **AND** the synthetic covariate table loads without `KeyError`
