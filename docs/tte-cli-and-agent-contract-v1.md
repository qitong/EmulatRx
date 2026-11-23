# EmulatRx TTE CLI & Python SDK Contract v1 (`emulatrx_tte`)

`emulatrx_tte` provides a deterministic, offline-verifiable Target Trial Emulation (TTE) protocol checker, seeded synthetic competing-risks / multi-arm cohort generator, overlap/IPTW propensity diagnostic gate, and weighted competing-risks survival analysis workbench.

All legacy files (`v1-codebase.ipynb`, `environment.yml`, `emulatrx_demo/`) remain untouched. Every estimation artifact locks `source_mode: "synthetic_only"` and explicit `prohibited_claims` (`非真实患者数据`, `非临床疗效证据`, `非因果实证证明`).

---

## 1. Global Invariants & Exit Code Contract

1. **Atomic sibling staging & non-overwrite**: Every CLI subcommand writes into a temporary sibling directory `.emulatrx-tte-stage-*`, computes SHA-256 digests for all published files in `run_manifest.json`, and atomically renames to `--out`. Passing an existing `--out` raises `OUTPUT_EXISTS` (exit code `2`) without modifying the existing directory.
2. **Byte-for-byte determinism**: Wall-clock execution time is recorded solely in `run_manifest.json["run_at_utc"]`. Given identical protocol JSON, random `--seed`, CLI parameters, Git commit, and pinned Python/scientific stack, all data/estimate/report files are byte-for-byte identical across runs.
3. **Zero hardcoded clinical thresholds**: Diagnostic gates (`max_abs_smd`, `min_ess_ratio`, `positivity_epsilon`) and default `horizon_days` come strictly from the validated protocol JSON (`schemas/tte-protocol-v1.schema.json`); `--seed` and `--bootstrap-iterations` are explicit required parameters.

| Exit Code | Exception / Gate State | Published Directory Behavior |
| --- | --- | --- |
| `0` | Validation, diagnostics gate, and estimation succeeded | Publishes full output bundle + `run_manifest.json` |
| `2` | `ContractError` (input JSON/schema error, observable timestamp rule leak, unsupported E-value measure, invalid CLI parameter, or `OUTPUT_EXISTS`) | Publishes **nothing**; cleans up staging directory; writes `<CODE>: <detail>` to `stderr` |
| `3` | `GateBlockedError` / statistical diagnostic gate blocked (`gate_status: "blocked"`) | `diagnose` & `run-synthetic`: publish diagnostic files only (**omitting** `weighted_cohort.csv`, `estimates.json`, `curves/`, `audit_report.md`).<br>`estimate`: publishes **nothing** and refuses estimation 100% |

---

## 2. CLI Reference (`python -m emulatrx_tte`)

### 2.1 `validate-protocol`
```bash
python -m emulatrx_tte validate-protocol \
  --protocol <protocol.json> \
  --out <new_dir>
```
- **Checks**: JSON strictness (`<= 64 KiB`, no duplicate keys, no `NaN`/`Infinity`), explicit `time_zero` (`day_offset == 0`; `observational_baseline` requires `landmark_day == 0`; `observational_landmark` requires `landmark_day > 0`), no post-baseline/post-landmark eligibility criteria (`POST_TIME_ZERO_ELIGIBILITY`) or covariates (`POST_LANDMARK_COVARIATE_LEAK`), unique reference arm at `treatment_strategies[0]` (`role` in `control|comparator|reference`), explicit competing-event handling (`UNHANDLED_COMPETING_EVENT`), explicit horizons for `RD`/`RR_CIF` (`MISSING_HORIZON`), and explicit `diagnostic_gates`.
- **Outputs (`0`)**: `normalized_protocol.json`, `protocol_check_report.md`, `run_manifest.json`.

### 2.2 `synth`
```bash
python -m emulatrx_tte synth \
  --protocol <protocol.json> \
  --seed <0..4294967295> \
  --cases <1..20000> \
  [--dgp-params <dgp_override.json>] \
  --out <new_dir>
```
- **Behavior**: Generates seeded synthetic baseline covariates, two-arm logistic or $\ge 3$-arm multinomial treatment assignments, piecewise-exponential latent primary (`event_code=1`) and competing (`event_code=2`) failure times, and independent administrative/random right-censoring (`event_code=0`).
- **Outputs (`0`)**: `cohort.csv`, `dgp_truth.json` (`parameter_type: "synthetic_dgp_parameters"`), `PURPOSE.txt`, `run_manifest.json`.

### 2.3 `diagnose`
```bash
python -m emulatrx_tte diagnose \
  --protocol <protocol.json> \
  --cohort <cohort.csv> \
  --weight-method <overlap|stabilized_iptw|iptw|generalized_overlap|pairwise_overlap> \
  [--truncate-quantiles q_low,q_high] \
  [--covariate-offsets <offsets.json>] \
  --out <new_dir>
```
- **Behavior**: Performs landmark risk-set slicing (`duration_day > landmark_day`), checks covariate timestamp offsets, fits exact unpenalized MLE binary/multinomial propensity models, evaluates positivity, Kish ESS ratio, and post-weighting `|SMD|` against `estimand.diagnostic_gates`.
- **Outputs (`0`, `gate_status="passed"`)**: `attrition_table.csv`, `balance_diagnostics.csv`, `diagnostics_summary.json`, `weighted_cohort.csv`, `run_manifest.json`.
- **Outputs (`3`, `gate_status="blocked"`)**: `attrition_table.csv`, `balance_diagnostics.csv`, `diagnostics_summary.json`, `run_manifest.json` (`weighted_cohort.csv` omitted).

### 2.4 `estimate`
```bash
python -m emulatrx_tte estimate \
  --protocol <protocol.json> \
  --diagnostics-dir <passed_diagnose_dir> \
  [--weighted-cohort <passed_diagnose_dir/weighted_cohort.csv>] \
  --seed <0..4294967295> \
  --bootstrap-iterations <1..5000> \
  [--horizons <h1,h2,...>] \
  [--censoring-ipcw <none|arm_stratified|covariate_cox>] \
  [--ci-level 0.95] \
  --out <new_dir>
```
- **Fail-closed gate verification (exit code `3` on failure)**:
  1. Verifies `diagnostics_summary.json` and `run_manifest.json` exist and match recorded SHA-256 digests;
  2. Verifies `gate_status == "passed"` in both summary and manifest;
  3. Verifies `weighted_cohort.csv` exists and matches `run_manifest.json["output_sha256"]["weighted_cohort.csv"]`;
  4. Verifies `normalized_protocol_sha256`, `simulation_or_data_label`, `landmark_day`, `treatment_strategies`, and `diagnostic_gates` match upstream diagnostics.
- **Estimators**:
  1. **Weighted Aalen–Johansen CIF**: Simultaneous estimation of primary (`event=1`) and competing (`event=2`) cumulative incidence curves plus biased `1 - KM` reference (`naive_km_primary_biased_ref`), supporting 2-arm and $\ge 3$-arm protocols (contrasted against `treatment_strategies[0]`). Computes $\text{RD}_1(t)$, $\text{RD}_2(t)$, and $\text{RR}_{\text{CIF},1}(t)$ with deterministic subject-level stratified bootstrap CIs (`np.random.default_rng(seed)`).
  2. **Fine–Gray Subdistribution Hazard (`sHR`)**: Time-varying reverse-KM IPCW risk sets $\hat{G}(t_m^-)/\hat{G}(T_i^-)$ (Fine & Gray 1999; Geskus 2011) combined with baseline propensity/overlap weights and subject-clustered Lin–Wei sandwich variance.
  3. **Weighted Cause-Specific Cox (`csHR`)**: Fitted simultaneously for primary (`event=1`) and competing (`event=2`) causes with subject-clustered sandwich variance.
  4. **Competing-Risks E-values**: Computed at each horizon strictly from $\text{RR}_{\text{CIF},1}(t)$ (point estimate and CI limit closer to `1.0`).
- **Outputs (`0`)**:
  - `estimates.json` (`source_mode`, `prohibited_claims`, `censoring_model`, `bootstrap_config`, `methodological_warnings`, `co_presented_comparison_table`, `arm_cif_summaries`, `contrasts`)
  - `curves/*.csv` (`curves/cif_<arm_id>.csv` for joint weighting; `curves/cif_<pair_id>__<arm_id>.csv` for `pairwise_overlap`)
  - `audit_report.md` (co-presented Markdown comparison table + mandatory methodological warnings + non-clinical claim locks)
  - `run_manifest.json`

### 2.5 `run-synthetic`
```bash
python -m emulatrx_tte run-synthetic \
  --protocol <protocol.json> \
  --seed <0..4294967295> \
  --cases <1..20000> \
  --weight-method <overlap|stabilized_iptw|iptw|generalized_overlap|pairwise_overlap> \
  --bootstrap-iterations <1..5000> \
  [--dgp-params <dgp_override.json>] \
  [--truncate-quantiles q_low,q_high] \
  [--horizons <h1,h2,...>] \
  [--censoring-ipcw <none|arm_stratified|covariate_cox>] \
  [--ci-level 0.95] \
  --out <new_dir>
```
- **Outputs (`0`, `gate_status="passed"`)**: `normalized_protocol.json`, `protocol_check_report.md`, `dgp_truth.json`, `PURPOSE.txt`, `attrition_table.csv`, `balance_diagnostics.csv`, `diagnostics_summary.json`, `weighted_cohort.csv`, `estimates.json`, `curves/*.csv`, `audit_report.md`, `run_manifest.json`.
- **Outputs (`3`, `gate_status="blocked"`)**: `normalized_protocol.json`, `attrition_table.csv`, `balance_diagnostics.csv`, `diagnostics_summary.json`, `run_manifest.json` (`weighted_cohort.csv` and `estimates.json` are never published).

---

## 3. Python SDK Reference

```python
from pathlib import Path
from emulatrx_tte import (
    ContractError,               # exit_code == 2
    GateBlockedError,            # exit_code == 3
    SOURCE_MODE_SYNTHETIC_ONLY,  # "synthetic_only"
    PROHIBITED_CLAIMS,
    validate_protocol_dict,
    load_and_validate_protocol,
    validate_protocol_command,
    generate_synthetic_cohort,
    compute_weights,
    kish_ess,
    diagnose_cohort,
    weighted_aalen_johansen_cif,
    fit_fine_gray_shr,
    fit_weighted_cause_specific_cox,
    estimate_competing_risks,
    run_synthetic_pipeline,
    compute_evalue,
)
```

- `estimate_competing_risks(protocol_path, *, diagnostics_dir=..., weighted_cohort_path=..., seed=..., bootstrap_iterations=..., out_dir=..., horizons_days=None, censoring_ipcw="none", ci_level=0.95) -> dict`: raises `ContractError` (`exit_code=2`) on contract errors and `GateBlockedError` (`exit_code=3`) on blocked/missing/tampered upstream diagnostics.
- `run_synthetic_pipeline(protocol_path, *, seed=..., cases=..., weight_method=..., bootstrap_iterations=..., out_dir=..., dgp_params_path=None, truncate_quantiles=None, horizons_days=None, censoring_ipcw="none", ci_level=0.95) -> tuple[dict, int]`: returns `(result_dict, 0 | 3)` and raises `ContractError` on contract errors.
- `compute_evalue(estimate, ci_lower=None, ci_upper=None, measure_type="RR", outcome_incidence=None, has_competing_risks=False) -> dict`: rejects `measure_type="sHR"` unconditionally and rejects `"csHR"` / `"HR"` when competing risks exist and `outcome_incidence > 0.15` (`UNSUPPORTED_EVALUE_MEASURE`).

---

## 4. Methodological Guardrails Embedded in Every Output

1. **Subdistribution vs Cause-Specific Co-Presentation**: `estimates.json["co_presented_comparison_table"]` and `audit_report.md` co-present `sHR`, `csHR_primary`, `csHR_competing`, `CIF_primary(t)`, `CIF_competing(t)`, `RD_primary(t)`, `RD_competing(t)`, `RR_CIF_primary(t)`, and `evalue_rr_cif` in every row, with automatic `competing_mortality_divergence_alert` detection when high competing mortality drives `sHR < 1`.
2. **Built-in Hazard-Ratio Selection Bias (Hernán 2010)**: Explicitly warns that `csHR` conditions on event-free survival up to $t^-$ and suffers from post-baseline selection bias even after baseline propensity/overlap balance.
3. **Censoring Independence & Joint IPCW Sensitivity**: Records the baseline-weighted independent right-censoring assumption under `censoring_ipcw="none"` and provides `arm_stratified` reverse-KM IPCW and stabilized `covariate_cox` IPCW modes for sensitivity analysis.
