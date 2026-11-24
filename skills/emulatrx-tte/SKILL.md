---
name: emulatrx-tte
description: Run deterministic Target Trial Emulation (TTE) protocol validation, ClinicalTrials.gov target RCT matching, privacy-safe user cohort ingestion (`user_cohort`), target-RCT simulated cohort rehearsal (`simulated_cohort`), synthetic competing-risks benchmarking (`synthetic_only`), overlap/IPTW diagnostic gating, and weighted competing-risks survival analysis (Aalen-Johansen CIF, Fine-Gray sHR, cause-specific Cox csHR, horizon RD/RR_CIF, E-values) via `python -m emulatrx_tte` or the Dual-Mode Web/MCP Workbench.
---

# `emulatrx-tte` External Agent Execution Contract

Deterministic Python CLI, SDK (`emulatrx_tte`), and Web/MCP Dual-Mode Workbench (`emulatrx_preview`) for seven-element Target Trial Emulation (TTE) protocol checking, `ClinicalTrials.gov` target RCT matching, propensity/overlap diagnostic gating, and weighted competing-risks survival analysis.

## 1. Non-Negotiable Provenance, Privacy & Claim Guardrails

Every run explicitly locks one of three `source_mode` (`project_mode`) contracts:

1. **`"synthetic_only"` (default CLI benchmark)**:
   - Accepts only seeded synthetic cohorts (`SYN-000001..`). Rejects any non-`SYN-*` subject ID (`NON_SYNTHETIC_COHORT_REJECTED`).
   - Enforces `prohibited_claims`: `非真实患者数据`, `非临床疗效证据`, `非因果实证证明`.
2. **`"simulated_cohort"` (no-data target-RCT feasibility rehearsal)**:
   - Aligns treatment arms and follow-up horizons with a matched `ClinicalTrials.gov` target RCT (`emulatrx_tte/rct_matcher.py`), generates a virtual confounded cohort (`SIM-000001..`), and outputs **🟣《目标试验模拟（无数据模拟预演项目）可行性报告》** including a DGP truth vs weighted estimate recovery table (`simulation_recovery_table`).
   - Must never be cited as clinical efficacy or safety evidence.
3. **`"user_cohort"` (de-identified owned-cohort real TTE project)**:
   - Ingests de-identified `.csv` / `.tsv` cohorts via `emulatrx_tte/cohort_ingest.py`, producing a verified `ingest_manifest.json` (recording `input_sha256`, `standardized_cohort_sha256`, column mapping, time unit conversion to days, categorical one-hot encoding, zero-variance/collinear covariate protection, and step-by-step attrition waterfall).
   - Outputs **🟢《真实队列目标试验模拟（真项目）分析报告》**, explicitly stating the four observational causal identification assumptions (`Consistency`, `Conditional Exchangeability / No Unmeasured Confounding`, `Positivity`, `Conditional Non-Informative Censoring`) and residual unmeasured confounding limitations.
   - **Strict LLM Privacy Redline**: Whenever optional LLM assistance (`emulatrx_tte/llm_assist.py`) is enabled, outbound payloads are scrubbed by `sanitize_profile_for_llm` + `assert_llm_payload_privacy_safe` to contain **only column names, inferred types, missingness rates, deduplicated short category labels (<= 20 non-ID labels), and aggregate quantiles/means/SDs — NEVER a single individual patient record or subject ID**. When no LLM key is set, 100% of steps fall back deterministically without error.
4. **Zero hardcoded clinical thresholds & atomic non-overwrite staging**:
   - Read gate thresholds (`max_abs_smd`, `min_ess_ratio`, `positivity_epsilon`) and default `horizon_days` from the validated protocol JSON, pass `--seed` and `--bootstrap-iterations` explicitly, and always pass a **new, non-existent** `--out` directory (`OUTPUT_EXISTS` exits `2`).

## 2. Workflow & CLI Subcommands

Run from the repository root with Python 3.10+ and scientific dependencies (`numpy`, `scipy`, `pandas`, `scikit-learn`, `lifelines`).

### Option A: Single-Command End-to-End Synthetic Benchmark (`run-synthetic`, `source_mode="synthetic_only"`)

```bash
python -m emulatrx_tte run-synthetic \
  --protocol examples/tte_competing_risk_v1.json \
  --seed 42 \
  --cases 800 \
  --weight-method overlap \
  --bootstrap-iterations 200 \
  --out /tmp/tte-run-synthetic
```

### Option B: Owned-Cohort Real TTE Project (`run-user-cohort`, `source_mode="user_cohort"`)

```bash
python -m emulatrx_tte run-user-cohort \
  --cohort /path/to/deidentified_cohort.csv \
  --mapping /path/to/column_mapping.json \
  [--target-rct /path/to/target_rct.json] \
  --weight-method overlap \
  --seed 42 \
  --bootstrap-iterations 100 \
  --out /tmp/tte-user-cohort-run
```

### Option C: No-Data Target-RCT Simulated Rehearsal (`run-simulated`, `source_mode="simulated_cohort"`)

```bash
python -m emulatrx_tte run-simulated \
  --protocol /path/to/simulated_protocol.json \
  [--dgp-params /path/to/sim_dgp.json] \
  --seed 42 \
  --cases 1500 \
  --weight-method overlap \
  --bootstrap-iterations 100 \
  --out /tmp/tte-simulated-run
```

### Option D: Stepwise Staged Pipeline (`validate-protocol` → `synth` → `diagnose` → `estimate`)

```bash
python -m emulatrx_tte validate-protocol \
  --protocol examples/tte_competing_risk_v1.json \
  --out /tmp/tte-step1-protocol

python -m emulatrx_tte synth \
  --protocol /tmp/tte-step1-protocol/normalized_protocol.json \
  --seed 42 --cases 800 \
  --out /tmp/tte-step2-synth

python -m emulatrx_tte diagnose \
  --protocol /tmp/tte-step1-protocol/normalized_protocol.json \
  --cohort /tmp/tte-step2-synth/cohort.csv \
  --weight-method overlap \
  --out /tmp/tte-step3-diagnose

python -m emulatrx_tte estimate \
  --protocol /tmp/tte-step1-protocol/normalized_protocol.json \
  --diagnostics-dir /tmp/tte-step3-diagnose \
  --seed 42 --bootstrap-iterations 200 \
  --censoring-ipcw none \
  --out /tmp/tte-step4-estimate
```

## 3. Exit Codes & Fail-Closed Gate Rules

| Exit Code | Meaning | Required Agent Action |
| --- | --- | --- |
| `0` | Contract & diagnostic gates passed; full artifacts + `run_manifest.json` atomically published | Parse `estimates.json`, `audit_report.md`, `curves/*.csv`, and verify `run_manifest.json` |
| `2` | `ContractError` (schema violation, `MISSING_TIME_ZERO`, `INVALID_TIME_ZERO`, `POST_TIME_ZERO_ELIGIBILITY`, `POST_LANDMARK_COVARIATE_LEAK`, `INVALID_STRATEGIES`, `UNHANDLED_COMPETING_EVENT`, `UNSUPPORTED_EVALUE_MEASURE`, `MISSING_USER_COHORT_INGEST_MANIFEST`, `PRIVACY_VIOLATION_BLOCKED`, `OUTPUT_EXISTS`, etc.) | Stop immediately; inspect the error code + `Hint:` on `stderr` and fix protocol/mapping/arguments before retrying into a new `--out` directory |
| `3` | Statistical / upstream gate blocked (`POSITIVITY_VIOLATION`, `ESS_RATIO_BELOW_MIN`, `SMD_EXCEEDS_MAX`, `ZERO_RISK_SET`, `UPSTREAM_GATE_NOT_PASSED`, `MISSING_WEIGHTED_COHORT`, `MANIFEST_INTEGRITY_MISMATCH`, `PROTOCOL_DIAGNOSTICS_MISMATCH`, `INGEST_LINEAGE_MISMATCH`) | **Fail closed**: report the diagnostic gate failure from `diagnostics_summary.json` / `stderr`. Never fabricate survival estimates, never bypass `diagnose`, and never copy an unverified `weighted_cohort.csv` |

## 4. Mandatory Side-by-Side Competing-Risks Interpretation Rules

When reporting `estimates.json` (`co_presented_comparison_table`) or `audit_report.md`:

1. **Co-present all five core quantities together**: Always report Fine–Gray `sHR` (primary), primary cause-specific `csHR_1`, competing mortality `csHR_2`, primary cumulative incidence `CIF_1(t)` (`RD_1(t)`, `RR_CIF_1(t)`), and competing cumulative incidence `CIF_2(t)` (`RD_2(t)`) side by side relative to reference arm `treatment_strategies[0]`.
2. **Guard against competing-mortality `sHR < 1` misinterpretation**: If competing mortality is elevated (`csHR_2 > 1`, `RD_2(t) > 0`), subjects in the active arm die of the competing cause earlier and remain in the Fine–Gray subdistribution risk set via IPCW weights $\hat{G}(t^-)/\hat{G}(T_i^-)$, depressing `CIF_1(t)` and yielding `sHR < 1` even when primary biological hazard is unchanged (`csHR_1 ≈ 1`). Check `methodological_warnings.competing_mortality_divergence_alert`.
3. **State hazard-ratio selection bias (Hernán 2010)**: Note that `csHR_1` and `csHR_2` condition on remaining event-free up to $t^-$ and remain subject to built-in post-baseline selection bias; anchor absolute magnitude on explicit horizon `RD_1(t)` and `RR_CIF_1(t)`.
4. **Enforce E-value competing-risks boundaries**: Report sensitivity E-values strictly from horizon `RR_CIF_1(t)` (`evalue_rr_cif_point` and `evalue_rr_cif_ci_limit`). Never apply single-event E-value conversions to `sHR`, and never convert competing-risks `csHR` / `HR` when cumulative incidence `> 0.15`.
5. **State censoring & bootstrap variance scope**: When `--censoring-ipcw none` is used, report the arm-conditional independent right-censoring assumption; note that CIF/`RD`/`RR_CIF` bootstrap CIs hold Stage-1 baseline propensity weights fixed (`baseline_propensity_weights: "fixed"`).

For full CLI flags, JSON field schemas, Web/MCP endpoints, and Python SDK signatures, read [`docs/tte-cli-and-agent-contract-v1.md`](../../docs/tte-cli-and-agent-contract-v1.md) and [`docs/FEATURE_API_OVERVIEW.md`](../../docs/FEATURE_API_OVERVIEW.md).
