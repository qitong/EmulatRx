# Tasks: mimic-free-emulation

> Status after Build phase. `[x]` done, `[~]` partial/optional, `[-]` descoped (with reason), `[ ]` deferred to Verify.

## A. Substitute data source (Task 1)

- [x] A1. Design decision: Option A (OMOP) vs B (synthetic) — chose **B+C** (synthetic emitting CSV cache + optional Postgres). Rationale in the Design Doc: `DataframeBuilder` is hardwired to MIMIC's ICU shape, so real OMOP would require fabricating that shape anyway.
- [~] A2. Postgres: generator supports `--postgres` to load a `synthetic` schema (non-fatal when no DB). A local Postgres instance is **not** required — the CSV-cache path is the guaranteed route, so no `DB_*` need be set.
- [-] A3a. OMOP sample load — **descoped** (Option A rejected in design; see A1).
- [x] A3b. Synthetic generator (`scripts/generate_synthetic_oncology.py` + `scripts/synthetic_config.py`): confounded NSCLC cohort (covariate-dependent propensity + Weibull survival, target HR 0.65), deterministic seed. Emits the full CSV cache.
- [x] A4. `dataset='synthetic'` branch in `DataframeBuilder`; MIMIC-only munging guarded behind `if self.dataset=='mimic'`. Verified `get_dataframe` returns `(df, col_dict)` from cache with no DB (`scripts/test_builder_cache.py`).
- [~] A5. Few-shot SQL for the synthetic schema — **deferred**. Only needed for the pure Postgres→SQL-generation route; the cache path (the verified route) bypasses SQL generation. Note: `get_dataframe`'s `col_dict` construction (dataframe_utils.py:502) still hardcodes MIMIC column names, so a from-scratch SQL rebuild is not yet fully wired — tracked for a follow-up.
- [x] A6. Synthetic dataset runs through `DataframeBuilder`/`Informatician` → cohort table (X/A/Y/T). Full notebook wiring exercised in Verify (C1).

## B. DeepSeek trial-parsing (Task 2)

- [x] B1. `trialistUtils/main.py::parse_clinical_trial` uses env-configured `OpenaiClient()` (DeepSeek); hardcoded key removed.
- [x] B2. `trialistUtils/parse_info.py` builds an env client lazily; hardcoded key/client removed.
- [x] B3. No committed key in tracked source (`scripts/test_no_secrets.py` greps and passes). NOTE: old keys remain in git **history** — rotate/revoke.
- [~] B4. Client construction verified offline (model=deepseek-chat, base=deepseek.com). A live parse call is exercised in Verify.

## C. Integration & verification

- [x] C0. Integration test: `Informatician.get_df_dict` (no DB) → `emulation_utils` recovers the effect — `run_cox_model` HR 0.70 CI (0.61,0.80); `perform_iptw` HR 0.67 CI (0.58,0.79); IPTW improves balance. (`scripts/test_integration_statistician.py`; `perform_psm` xfailed on a pre-existing pandas bug at emulation_utils.py:41.)
- [x] C2. Docs (`CLAUDE.md` + `README.md`): how to generate synthetic data + run on DeepSeek.

## Deferred to the Verify phase (tracked by comet-verify, not build tasks)

- Full notebook end-to-end (`LLM_MODEL='deepseek_api'`, `TRIAL_ID='NCT00000099'`) reaching the Statistician stage on DeepSeek.
- A live DeepSeek trial-parse call (B4) beyond the offline client-construction check.
