---
name: emulatrx-tte
description: Run deterministic Target Trial Emulation (TTE) protocol validation, synthetic competing-risks cohort benchmarking, overlap/IPTW diagnostic gating, and weighted competing-risks survival analysis (Aalen-Johansen CIF, Fine-Gray sHR, cause-specific Cox csHR, horizon RD/RR_CIF, E-values) via `python -m emulatrx_tte`. Trigger when validating TTE seven-element protocols, diagnosing propensity/overlap balance gates, estimating survival under competing mortality, or generating SHA-256 audited synthetic TTE bundles.
---

# `emulatrx-tte` External Agent Execution Contract

Deterministic, offline Python CLI and SDK (`emulatrx_tte`) for seven-element Target Trial Emulation (TTE) protocol checking, propensity/overlap diagnostic gating, and weighted competing-risks survival analysis.

## 1. Non-Negotiable Data & Claim Guardrails

Every call operates under strict **synthetic-only** provenance (`source_mode: "synthetic_only"`):

1. **Zero real patient data**: Supply only seeded synthetic cohorts (`SYN-000001..`). Never pass real-world EHR/IPD records or live trial result payloads.
2. **Prohibited claims (`prohibited_claims`)**: Never describe any output as real patient data (`非真实患者数据`), clinical treatment efficacy/safety evidence (`非临床疗效证据`), or empirical proof of causal validity / absence of unmeasured confounding (`非因果实证证明`).
3. **Zero hardcoded clinical thresholds**: Read all gate thresholds (`max_abs_smd`, `min_ess_ratio`, `positivity_epsilon`) and default `horizon_days` from the validated protocol JSON, and pass `--seed` and `--bootstrap-iterations` explicitly on every run.
4. **Atomic non-overwrite staging**: Always pass a **new, non-existent** `--out` directory. Existing paths raise `OUTPUT_EXISTS` (exit code `2`) and are never overwritten.

## 2. Workflow & CLI Subcommands

Run from the repository root with Python 3.10+ and scientific dependencies (`numpy`, `scipy`, `pandas`, `scikit-learn`, `lifelines`).

### Option A: Single-Command End-to-End Synthetic Benchmark (`run-synthetic`)

Chains `validate-protocol` → `synth` → `diagnose` → `estimate` in a temporary workspace and atomically publishes the verified audit bundle:

```bash
python -m emulatrx_tte run-synthetic \
  --protocol examples/tte_competing_risk_v1.json \
  --seed 42 \
  --cases 800 \
  --weight-method overlap \
  --bootstrap-iterations 200 \
  --out /tmp/tte-run-synthetic
```

Optional sensitivity flags: `--dgp-params <json>`, `--truncate-quantiles q_low,q_high`, `--horizons 90,180,365`, `--censoring-ipcw <none|arm_stratified|covariate_cox>`, `--ci-level 0.95`.

### Option B: Stepwise Staged Pipeline (`validate-protocol` → `synth` → `diagnose` → `estimate`)

```bash
# Step 1: Validate seven-element protocol & observable timestamp rules
python -m emulatrx_tte validate-protocol \
  --protocol examples/tte_competing_risk_v1.json \
  --out /tmp/tte-step1-protocol

# Step 2: Generate seeded synthetic competing-risks cohort (event 0=censor, 1=primary, 2=competing)
python -m emulatrx_tte synth \
  --protocol /tmp/tte-step1-protocol/normalized_protocol.json \
  --seed 42 \
  --cases 800 \
  --out /tmp/tte-step2-synth

# Step 3: Slice landmark risk set, fit overlap/IPTW weights, and enforce SAP diagnostic gates
python -m emulatrx_tte diagnose \
  --protocol /tmp/tte-step1-protocol/normalized_protocol.json \
  --cohort /tmp/tte-step2-synth/cohort.csv \
  --weight-method overlap \
  --out /tmp/tte-step3-diagnose

# Step 4: Estimate weighted Aalen-Johansen CIF, Fine-Gray sHR, cause-specific csHR_1/csHR_2, and RR_CIF E-values
python -m emulatrx_tte estimate \
  --protocol /tmp/tte-step1-protocol/normalized_protocol.json \
  --diagnostics-dir /tmp/tte-step3-diagnose \
  --seed 42 \
  --bootstrap-iterations 200 \
  --censoring-ipcw none \
  --out /tmp/tte-step4-estimate
```

**Completion criterion**: Process exits `0`, `run_manifest.json` exists in `--out`, and every file in `output_sha256` matches its on-disk SHA-256 digest.

## 3. Exit Codes & Fail-Closed Gate Rules

| Exit Code | Meaning | Required Agent Action |
| --- | --- | --- |
| `0` | Contract & diagnostic gates passed; full artifacts + `run_manifest.json` atomically published | Parse `estimates.json`, `audit_report.md`, `curves/*.csv`, and verify `run_manifest.json` |
| `2` | `ContractError` (schema violation, `MISSING_TIME_ZERO`, `INVALID_TIME_ZERO`, `POST_TIME_ZERO_ELIGIBILITY`, `POST_LANDMARK_COVARIATE_LEAK`, `INVALID_STRATEGIES`, `UNHANDLED_COMPETING_EVENT`, `UNSUPPORTED_EVALUE_MEASURE`, `OUTPUT_EXISTS`, etc.) | Stop immediately; inspect the error code + `Hint:` on `stderr` and fix protocol/arguments before retrying into a new `--out` directory |
| `3` | Statistical / upstream gate blocked (`POSITIVITY_VIOLATION`, `ESS_RATIO_BELOW_MIN`, `SMD_EXCEEDS_MAX`, `ZERO_RISK_SET`, `UPSTREAM_GATE_NOT_PASSED`, `MISSING_WEIGHTED_COHORT`, `MANIFEST_INTEGRITY_MISMATCH`, `PROTOCOL_DIAGNOSTICS_MISMATCH`) | **Fail closed**: report the diagnostic gate failure from `diagnostics_summary.json` / `stderr`. Never fabricate survival estimates, never bypass `diagnose`, and never copy an unverified `weighted_cohort.csv` |

- On exit code `3`, `diagnose` and `run-synthetic` publish diagnostic evidence (`attrition_table.csv`, `balance_diagnostics.csv`, `diagnostics_summary.json`, `run_manifest.json`) while **omitting** `weighted_cohort.csv` and `estimates.json`.
- On exit code `3`, `estimate` publishes **nothing** and refuses estimation whenever `gate_status != "passed"`, `weighted_cohort.csv` is missing, SHA-256 digests mismatch `run_manifest.json`, or the protocol differs from upstream diagnostics.

## 4. Mandatory Side-by-Side Competing-Risks Interpretation Rules

When reporting `estimates.json` (`co_presented_comparison_table`) or `audit_report.md`:

1. **Co-present all five core quantities together**: Always report Fine–Gray `sHR` (primary), primary cause-specific `csHR_1`, competing mortality `csHR_2`, primary cumulative incidence `CIF_1(t)` (`RD_1(t)`, `RR_CIF_1(t)`), and competing cumulative incidence `CIF_2(t)` (`RD_2(t)`) side by side relative to reference arm `treatment_strategies[0]`.
2. **Guard against competing-mortality `sHR < 1` misinterpretation**: If competing mortality is elevated (`csHR_2 > 1`, `RD_2(t) > 0`), subjects in the active arm die of the competing cause earlier and remain in the Fine–Gray subdistribution risk set via IPCW weights $\hat{G}(t^-)/\hat{G}(T_i^-)$, depressing `CIF_1(t)` and yielding `sHR < 1` even when primary biological hazard is unchanged (`csHR_1 ≈ 1`). Check `methodological_warnings.competing_mortality_divergence_alert` and state explicitly whether `sHR < 1` reflects competing mortality rather than direct primary protection.
3. **State hazard-ratio selection bias (Hernán 2010)**: Note that `csHR_1` and `csHR_2` condition on remaining event-free up to $t^-$ and remain subject to built-in post-baseline selection bias; anchor absolute clinical/methodological magnitude on explicit horizon `RD_1(t)` and `RR_CIF_1(t)`.
4. **Enforce E-value competing-risks boundaries**: Report sensitivity E-values strictly from horizon `RR_CIF_1(t)` (`evalue_rr_cif_point` and `evalue_rr_cif_ci_limit`, reflecting total cumulative incidence pathways). Never apply single-event E-value conversions to `sHR`, and never convert competing-risks `csHR` / `HR` when cumulative incidence `> 0.15`.
5. **State censoring & bootstrap variance scope**: When `--censoring-ipcw none` is used, report the arm-conditional independent right-censoring assumption; use `--censoring-ipcw arm_stratified` or `--censoring-ipcw covariate_cox` for censoring sensitivity checks. Note that CIF/`RD`/`RR_CIF` bootstrap CIs hold Stage-1 baseline propensity weights fixed (`baseline_propensity_weights: "fixed"`, conditional on `weighted_cohort.csv` weights without propensity model re-estimation), while re-estimating censoring IPCW per replicate when `--censoring-ipcw != "none"`.

For full CLI flags, JSON field schemas, and Python SDK signatures, read [`docs/tte-cli-and-agent-contract-v1.md`](../../docs/tte-cli-and-agent-contract-v1.md).
