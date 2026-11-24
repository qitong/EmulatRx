# EmulatRx Bilateral Grace-Period Clone–Censor–Weight (CCW) Contract v1 (`tte-ccw-v1`)

`emulatrx_tte.ccw` implements a deterministic, offline-verifiable **Bilateral Grace-Period Treatment Clone–Censor–Weight (CCW)** estimator and synthetic immortal-time bias benchmark workbench.

All outputs strictly enforce `source_mode: "synthetic_only"`, `SYN-*` subject identifier guards, and `prohibited_claims` (`非真实患者数据`, `非临床疗效证据`, `非因果实证证明`). Legacy notebook (`v1-codebase.ipynb`), `environment.yml`, and `emulatrx_demo/` remain untouched, while the flawed legacy prototype `emulation_utils.adjust_immortal_time_bias` is explicitly fenced with a `RuntimeError` migration guard.

---

## 1. Clinical Scenario, Target Estimand & Identification Assumptions

### 1.1 Immortal-Time Bias Under Delayed Treatment Initiation
When an observational study compares **"initiate target therapy within a $g$-day grace period after $t_0=0$"** (Strategy 1) against **"never initiate target therapy during follow-up"** (Strategy 0):
- **Naive post-hoc static classification** (assigning patients who initiated on day 30 to the treated group at day 0 without cloning) guarantees that delayed initiators survived days 0–30 event-free, while dumping patients who died on day 15 prior to initiating into the uninitiated group. Even after full baseline propensity weighting, this induces severe spurious protective bias (`naive_rd < 0` under a strict null effect).
- **Fixed-landmark slicing** discards all high-risk events in $(0, g]$ and changes the target population.

### 1.2 Target Estimand
At each explicit evaluation horizon $t^* \in (0, H_{\max}]$, `emulatrx_tte.ccw` estimates the marginal cumulative incidence of primary events (`CIF_1(t*)`) and competing events (`CIF_2(t*)`), absolute risk difference `RD(t*)`, cumulative incidence ratio `RR_CIF(t*)`, and total-effect `RR_CIF` E-value between:
- **Strategy 1 (`initiate_within_grace`)**: initiate within $[0, g]$ days (early events at $t \le g$ prior to initiation are natural strategy-consistent events; survivors past $t=g$ who did not initiate by $g$ violate Strategy 1 at $t=g$);
- **Strategy 0 (`never_initiate`)**: remain uninitiated throughout follow-up (subjects who initiate at $T^{\text{init}}_i \le T_i$ violate Strategy 0 at the start of their initiation period).

### 1.3 Mandatory Causal Identification Assumptions
1. **Longitudinal conditional exchangeability**: all baseline ($L_0$) and pre-period time-varying ($L_{k-1}$) confounders jointly affecting treatment initiation and outcomes are measured without error and included in the arm-specific censoring models (**CCW cannot remove unmeasured confounding**).
2. **Consistency & no interference**: grace-period initiation follows the protocol-sustained natural initiation distribution (or protocol equivalence across initiation days in $[0, g]$) without cross-subject interference.
3. **Longitudinal positivity**: within every covariate stratum, survivors to $t=g$ have positive probability of having initiated by $g$, and uninitiated subjects in each period $k$ have positive probability of remaining uninitiated.
4. **Arm-conditional independent right-censoring**: administrative censoring is conditionally independent of latent failure times given measured covariates and clone strategy (or reweighted via `ipcw_spec.include_admin_ipcw=true`).

---

## 2. Protocol Schema (`schemas/tte-ccw-v1.schema.json`) & Exit Code Contract

All clinical time windows and diagnostic thresholds are passed explicitly via JSON (zero hardcoded clinical thresholds):
- `protocol_id`, `protocol_version: "tte-ccw-v1"`, `source_mode: "synthetic_only"`
- `grace_period_spec`: `grace_period_days` ($g > 0$), `period_length_days` ($\Delta > 0$), `max_followup_days` ($H_{\max} > g$), `evaluation_horizons_days` (all positive integer multiples of $\Delta$)
- `covariates`: `baseline_covariates` (`assessed_day_offset <= 0`) and optional `time_varying_covariates` (`offset_from_period_start_days < 0`, strictly before period $k$'s decision start $(k-1)\Delta$)
- `ipcw_spec`: optional `truncate_quantiles` (`[q_low, q_high]` or `null`) and `include_admin_ipcw` (`bool`, default `false`)
- `diagnostic_gates`: `min_ess_ratio`, `max_post_grace_smd`, `max_stabilized_weight`, `min_uncensored_subjects_per_clone`

| Exit Code | Meaning | Artifact Behavior |
| --- | --- | --- |
| `0` | Contract & diagnostic gates passed | Publishes full CCW bundle (`ccw_estimates.json`, `ccw_diagnostics.json`, `ccw_attrition.json`, `curves/*.csv`, `audit_report.md`, `ccw_preview_report.html`, `run_manifest.json`) |
| `2` | `ContractError` (schema error, non-`SYN-*` ID, non-divisible step $\Delta$, post-baseline/time-travel covariate leak, `OUTPUT_EXISTS`) | Publishes **nothing**; writes `<CODE>: <detail>` to `stderr` |
| `3` | Fail-closed diagnostic gate blocked (`UNCENSORED_BELOW_MIN`, `MAX_WEIGHT_EXCEEDED`, `ESS_RATIO_BELOW_MIN`, `POST_GRACE_SMD_EXCEEDS_MAX`) | Publishes diagnostic evidence (`normalized_protocol.json`, `ccw_diagnostics.json`, `ccw_attrition.json`, `balance_post_grace.csv`, `run_manifest.json`); **100% omits `ccw_estimates.json`** |

---

## 3. Symmetric Bilateral Cloning, Arm-Specific IPCW & Cluster Bootstrap

1. **`clone_and_censor`**:
   - **Strategy 1 (`initiate_within_grace`)**: artificial censoring is identically `0` for $k = 1, \dots, m_g$ ($t \le g$). Entering period $m_g + 1$ ($T_i > g$), subjects with `initiation_day is None` or `initiation_day > g` are artificially censored at $t = g$ (`T_clone = g, event_clone = 0, artificial_censored = 1`).
   - **Strategy 0 (`never_initiate`)**: subjects with $T^{\text{init}}_i \le T_i$ are artificially censored at the start of period $k_{\text{init}} = \max(1, \lceil T^{\text{init}}_i / \Delta \rceil)$ prior to period-$k_{\text{init}}$ outcomes (`T_clone = (k_init - 1) * Δ, event_clone = 0, artificial_censored = 1`).
2. **Strictly Separated Arm-Specific Stabilized IPCW (No Pooling)**:
   - **Strategy 1**: $W^{(1)}_{i,k} = 1.0$ for $k \le m_g$; on $T_i > g$, fits unpenalized MLE logistic regression $\hat{P}(\text{Init}_i \le g \mid T_i > g, L_i)$ with pure marginal numerator $\hat{P}(\text{Init}_i \le g \mid T_i > g)$ for $k \ge m_g + 1$.
   - **Strategy 0**: on Strategy 0 pre-decision uninitiated person-periods, fits unpenalized MLE Pooled Logistic regression (period indicators + $L_{i,k-1}$) for $\hat{P}(A_{i,k}=0 \mid \bar{A}_{i,k-1}=0, T_i > (k-1)\Delta, L_{i,k-1}, k)$ with pure period-specific marginal numerator $\hat{P}(A_{i,k}=0 \mid \bar{A}_{i,k-1}=0, T_i > (k-1)\Delta, k)$, multiplied cumulatively across $j \le k$.
3. **Subject-Cluster Bootstrap (`subject_cluster_bootstrap_with_ipcw_refit`)**:
   - Resamples `subject_id` with replacement **prior to cloning** and refits both arm-specific IPCW models and weighted discrete-time Aalen–Johansen CIFs inside every replicate.

---

## 4. CLI & Python SDK Usage

```bash
# 1. Validate CCW protocol JSON
python -m emulatrx_tte validate-ccw \
  --protocol examples/tte_grace_period_ccw_v1.json \
  --out /tmp/ccw-step1-validate

# 2. Generate seeded synthetic delayed-initiation cohort + counterfactual truth
python -m emulatrx_tte synth-ccw \
  --protocol examples/tte_grace_period_ccw_v1.json \
  --seed 42 --cases 2500 \
  --out /tmp/ccw-step2-synth

# 3. Run end-to-end synthetic CCW benchmark + interactive Chinese SVG HTML comparison report
python -m emulatrx_tte run-ccw-synthetic \
  --protocol examples/tte_grace_period_ccw_v1.json \
  --seed 42 --cases 2500 \
  --bootstrap-iterations 200 \
  --out /tmp/ccw-run-synthetic
```
