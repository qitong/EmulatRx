# EmulatRx TTE Foundation v1 (`emulatrx_tte`)

`emulatrx_tte` provides deterministic, offline-verifiable Target Trial Emulation (TTE) protocol validation, synthetic competing-risks / multi-arm cohort generation, propensity weighting & diagnostic gating, and VanderWeele & Ding (2017) E-value sensitivity calculations with explicit competing-risks validity boundaries.

This package leaves the existing `v1-codebase.ipynb` workflow, `environment.yml`, and `emulatrx_demo` contracts untouched.

## CLI Entry Points

Always pass a **new** `--out` directory; existing destinations are never overwritten.

```bash
# 1. Validate TTE seven-element protocol & observable timestamp rules
python -m emulatrx_tte validate-protocol \
  --protocol examples/tte_competing_risk_v1.json \
  --out /tmp/tte-protocol-check

# 2. Generate seeded, purely synthetic competing-risks benchmark cohort
python -m emulatrx_tte synth \
  --protocol examples/tte_competing_risk_v1.json \
  --seed 42 \
  --cases 600 \
  --out /tmp/tte-synth-cohort

# 3. Run landmark slicing, overlap/IPTW weighting, and SAP diagnostic gate
python -m emulatrx_tte diagnose \
  --protocol examples/tte_competing_risk_v1.json \
  --cohort /tmp/tte-synth-cohort/cohort.csv \
  --weight-method overlap \
  --out /tmp/tte-diagnose
```

## Exit Code Contract

| Exit Code | Meaning | Output Directory Behavior |
| --- | --- | --- |
| `0` | Success (`validate-protocol`, `synth`, or `diagnose` with `gate_status: "passed"`) | Full outputs + `run_manifest.json` atomically published |
| `2` | `ContractError` (malformed JSON, missing `time_zero`, `INVALID_TIME_ZERO` such as `observational_baseline` with `landmark_day > 0`, `POST_TIME_ZERO_ELIGIBILITY`, `INVALID_STRATEGIES` including missing/ambiguous/non-index-0 `control`/`comparator`/`reference` arm, `UNHANDLED_COMPETING_EVENT`, `MISSING_HORIZON`, `POST_LANDMARK_COVARIATE_LEAK`, etc.) | Nothing published; temporary stage cleaned up |
| `3` | Statistical diagnostic gate blocked (`POSITIVITY_VIOLATION`, `ESS_RATIO_BELOW_MIN`, `SMD_EXCEEDS_MAX`, or `ZERO_RISK_SET`) | Publishes `attrition_table.csv`, `balance_diagnostics.csv`, `diagnostics_summary.json` (`gate_status: "blocked"`), and `run_manifest.json`, omitting `weighted_cohort.csv` |

## Supported Weighting Methods

- `overlap` (Li, Morgan & Zaslavsky, *JASA* 2018): two-arm Overlap Weighting ($w_1 = 1 - \hat{e}(X), w_0 = \hat{e}(X) \in (0, 1)$, tilting function $h(X) = \hat{e}(X)(1-\hat{e}(X))$ maximized at $\hat{e}(X) = 0.50$), achieving exact finite-sample mean balance on linear logistic covariates via unpenalized MLE.
- `stabilized_iptw` and `iptw`: marginal-stabilized or unstabilized inverse probability of treatment weights across $\ge 2$ arms, with optional `--truncate-quantiles q_low,q_high`.
- `generalized_overlap` (Li & Li, *Ann Appl Stat* 2019): $w_k(X) \propto [\hat{e}_k(X)]^{-1} / \sum_j [\hat{e}_j(X)]^{-1}$ targeting the common multi-arm equipoise subpopulation.
- `pairwise_overlap`: runs separate two-arm overlap weighting for each unordered pair $(j, k)$ and stamps an explicit `transitivity_warning` in `diagnostics_summary.json` noting pairwise ATO subpopulations differ across pairs and estimates are non-transitive.

## Methodological & Sensitivity Boundaries

1. **Observable-rule checking boundary**: `validate-protocol` and `diagnose` detect only structural and explicit timestamp rule violations in supplied metadata/columns (`assessed_day_offset > landmark_day`). They cannot detect unmeasured confounding, measurement error, informative censoring, or prove causal validity.
2. **Synthetic DGP boundary**: `dgp_truth.json` records `synthetic_dgp_parameters` generated from `np.random.default_rng(seed)` for estimator verification only; it contains zero patient data and carries zero clinical meaning.
3. **E-value competing-risks boundary**: `emulatrx_tte.evalue.compute_evalue` supports `"RR"`, `"RR_CIF"`, and single-event `"HR"` (fixing the `0.5 ** sqrt(HR)` exponentiation formula in `emulation_utils.py::calculate_evalues`). It explicitly rejects Fine–Gray `"sHR"` and competing-risks `"csHR"` / `"HR"` when `outcome_incidence > 0.15` (`UNSUPPORTED_EVALUE_MEASURE`), requiring explicit horizon `"RR_CIF"` instead. An E-value never proves absence of unmeasured confounding.

## Competing-Risks Estimation & External Agent Contract (Issue #9)

For `python -m emulatrx_tte estimate`, `python -m emulatrx_tte run-synthetic`, weighted Aalen–Johansen CIF, Fine–Gray `sHR`, cause-specific Cox `csHR`, and external Agent invocation rules, see:
- [`docs/tte-cli-and-agent-contract-v1.md`](tte-cli-and-agent-contract-v1.md)
- [`skills/emulatrx-tte/SKILL.md`](../skills/emulatrx-tte/SKILL.md)
