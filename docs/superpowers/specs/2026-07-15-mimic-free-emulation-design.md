---
comet_change: mimic-free-emulation
role: technical-design
canonical_spec: openspec
archived-with: 2026-07-16-mimic-free-emulation
status: final
---

# Technical Design: mimic-free-emulation

## Problem recap

EmulatRx's Informatician/Statistician need a PostgreSQL real-world-data source in MIMIC-IV shape. We have no MIMIC-IV (credentialed, ICU-only, wrong domain — the target is oncology/NSCLC). Also, trial-parsing hardcodes OpenAI GPT-4o. Goal: run end-to-end on a synthetic oncology substitute + DeepSeek only.

## Key finding driving the approach

`DataframeBuilder` is hardwired to MIMIC's ICU shape, not schema-agnostic:
- Every merge joins on `stay_id` (`_build_dataframe_X:280`, `_build_big_dataframe:421,424`, `_get_matched_patients_by_ec:454`, `get_dataframe:496`).
- `_build_dataframe_X` (267-304) hardcodes MIMIC-derived column munging (`_max`/`_min` suffix rules, `variables_keep_min`, direct refs to `abs_basophils`/`sofa`) → `KeyError` on any non-MIMIC covariate table. The existing `dataset='insight'` branch only swaps `scheme_name`/`X_table_names` but still runs this MIMIC-specific builder.
- `_build_dataframe_treatment.clean_treatment_time` (309-355) assumes ICU `intime`/`outtime`/`treatment_time`.

Therefore **real OMOP (Option A) does not plug in** — it would require faking ICU-shaped tables anyway. Decision: generate synthetic data directly in the shape the code needs.

Second finding: `get_dataframe` (481-517) returns from the **CSV cache** (`dataframe.csv` + `col_dict.json`) before touching Postgres, and `DataframeBuilder.__init__` wraps the DB connect in try/except (non-fatal). So synthetic data injected as the cache runs the pipeline with **no Postgres at all**.

## Decisions (confirmed with user)

- **Approach B + C**: synthetic generator emits BOTH a Postgres `synthetic` schema (new `DataframeBuilder` branch + few-shot SQL; exercises LLM SQL generation) AND the CSV cache (guaranteed end-to-end run without Postgres).
- **Fidelity: confounded effect** — `X → propensity(A)`, `(A, X) → survival hazard` with a target HR, so PSM/IPTW/Cox demonstrate confounding adjustment.
- **Seed: blend all 5 NSCLC papers** into a generic advanced-NSCLC cohort; trial meta via CT.gov API (`trialistUtils/web_utils.get_trial_info`).

## The runtime contract (from `col_dict.json` + `get_dataframe:502-510`)

`dataframe.csv` columns, in order:
`stay_id, subject_id, treatment, treatment_time_delta, <outcome_cols>, <duration_cols>, <covariate_cols>`

`col_dict.json`:
```json
{
  "covariate_cols": ["age","sex","ecog_ps","smoking","histology","stage","pdl1_tps","biomarker_status","...labs..."],
  "treatment_cols": "treatment",
  "outcome_cols": ["event"],
  "duration_cols": ["duration_day"],
  "treatment_time_cols": "treatment_time_delta",
  "patient_id": "stay_id"
}
```
- `treatment`: 0/1. `event`: 0/1 (progression/death). `duration_day`: float days to event/censor. `treatment_time_delta`: float days (0 for treated at baseline, NaN for control).

## Components

### 1. Synthetic generator — `scripts/generate_synthetic_oncology.py`
Inputs: seed distributions blended from the 5 papers in `/Users/qitonghu/Desktop/0816 长沙/papers/` (baseline covariate means/SDs & proportions, arm sizes, primary HR, median follow-up) + optional CT.gov meta. Deterministic via a fixed seed.

Generative model (per patient i):
1. `X_i` ~ blended marginals (age Normal; sex/ECOG/smoking/histology/stage/PD-L1/biomarker categorical; a handful of labs Normal/log-Normal).
2. `A_i` ~ Bernoulli(σ(β0 + βᵀ standardized(X_i)))  — confounding via covariate-dependent propensity.
3. Survival `T_i` ~ Weibull with `log-hazard = γ·A_i + δᵀ X_i`, `γ = log(HR_target)`; censoring `C_i` ~ Exponential(median follow-up). `duration_day = min(T_i, C_i)`, `event = 1[T_i ≤ C_i]`. `treatment_time_delta = 0` if A=1 else NaN.

Outputs (both):
- **CSV cache**: `trials/<TRIAL_ID>/dataframes/dataframe.csv` + `col_dict.json` (and, if needed, `big_df.csv`/`merged_X.csv` so `_build_big_dataframe`'s cache also short-circuits).
- **Postgres**: schema `synthetic`, a denormalized `synthetic.cohort` table (all columns) + minimal split tables for the SQL-generation demo.

### 2. New `dataset='synthetic'` branch — `dataframe_utils.py`
- `__init__`: `scheme_name='synthetic'`, `X_table_names=['cohort']`.
- Add a synthetic-specific build path so `_build_dataframe_X`/`_build_dataframe_treatment` do **not** run the MIMIC munging: a `SELECT *`-style read of `synthetic.cohort` keyed on `stay_id`, or short-circuit entirely on the CSV cache. Guard the MIMIC-specific code behind `if self.dataset == 'mimic'`.
- Few-shot SQL examples for the Informatician matching the `synthetic` schema (eligibility/treatment/outcome returning `stay_id`).

### 3. Task 2 — DeepSeek trial-parsing
- `trialistUtils/main.py::parse_clinical_trial`: replace the hardcoded `OpenaiClient(api_key='sk-…', …)` with a client built from env (`OpenaiClient(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url=os.getenv('DEEPSEEK_BASE_URL','https://api.deepseek.com'), model=os.getenv('DEEPSEEK_MODEL','deepseek-chat'))`), or accept an injected `llm`. Preserve the `get_chatgpt_response(prompt, content)` contract.
- `trialistUtils/parse_info.py`: same — remove the module-level hardcoded key and client.
- Grep guard: no committed source file contains `sk-`.

## Data flow (downstream unchanged)
```
generator → {CSV cache, synthetic Postgres}
          → DataframeBuilder(dataset='synthetic') → Informatician.get_df_dict → (df, col_dict)
          → Statistician (PSM/IPTW/Cox in emulation_utils) → estimates
```
`emulation_utils.py` and the LangGraph orchestration are untouched.

## Verification strategy
- **Unit**: generator output has exactly the contract columns; `col_dict.json` validates; seeded covariate means/proportions within tolerance; a Cox fit on the raw synthetic data recovers `HR_target` within its CI.
- **Integration**: `Informatician.get_df_dict(TRIAL_ID)` returns `(df, col_dict)` from cache with no DB; feeding it to the Statistician's PSM/IPTW/Cox produces post-matching SMD < 0.1 on key covariates and an adjusted HR near target.
- **Postgres path**: with the cache cleared and `dataset='synthetic'`, `get_dataframe` rebuilds the same df from `synthetic.*` via SQL.
- **End-to-end**: notebook run for the synthetic trial reaches the Statistician stage and prints a result, using `LLM_MODEL='deepseek_api'`.
- **Task 2**: `grep -rn 'sk-' *.py trialistUtils/` returns nothing; a parse call routes to DeepSeek.

## Risks / boundary conditions
- Synthetic data is illustrative, not clinical evidence (documented in proposal non-goals).
- The Statistician agent / notebook may hardcode some outcome-definition or column names expecting MIMIC (`hospitaldischargestatus`); if so, add a thin mapping or align `col_dict` naming — handle in build.
- Empty/degenerate cohorts (all-censored, perfect separation in propensity) — clamp propensity, ensure event rate > 0.
- Postgres optional; the CSV-cache path is the guaranteed route.
