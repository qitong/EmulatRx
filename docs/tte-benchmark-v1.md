# TTE 公开结果对标与阴性对照诊断（tte-benchmark-v1）

本模块把**已经完成**的目标试验模拟（TTE）与目标 RCT 的**研究层面公开汇总**做统计接近度比较，并可用预先指定的阴性对照结局（negative control outcome, NCO）探测残余混杂。它不读取、不保存、不外传个体患者数据（`ipd_used: false`）。

数值接近**不能**证明无未测量混杂、因果效应已经确立或临床有效。每份报告都带有：

- `numerical_proximity_proves_causality: false`
- `clinical_efficacy_claim_prohibited: true`

独立回滚开关：`EMULATRX_ENABLE_BENCHMARKING=0`（也接受 `false` / `off` / `no`）。关闭后 `POST /api/v1/tte/benchmark` 返回 HTTP `503`，MCP 工具返回 `BENCHMARKING_FEATURE_DISABLED`，工作台对标面板隐藏。默认开启，无需重新部署即可切回。

## 失败关闭

1. `BENCHMARK_BLOCKED_UPSTREAM_DIAGNOSTIC_FAILED`：`tte_result.gate_passed` 不是 `true`（例如 ESS、SMD 或正值性门禁未过）。`effect_comparison` 为 `null`。
2. `BENCHMARK_BLOCKED_PROTOCOL_OR_ENDPOINT_MISMATCH`：`overall_alignment` 不是 `aligned`（仓库里的 `overall_alignment_status = protocol_aligned` 视为对齐），或人群、双臂、主要终点、随访窗任一维度不匹配。此前为了跑探索性 TTE 而设置的 `confirm_protocol_mismatch` / `confirm_arm_mismatch` **不能**解锁一致性结论。
3. `BENCHMARK_UNAVAILABLE_NO_RESULTS_SECTION`：`hasResults` 为假、`resultsSection` 没有可用的主要结局汇总，且没有带非空 `source_citation` 的人工摘要。空引用字符串同样拒绝。
4. `BENCHMARK_SCALE_MISMATCH`：公开结果与 TTE 的效应尺度（`hr` / `rr` / `rd`）无法配对。
5. `NEGATIVE_CONTROL_BIAS_DETECTED`：阴性对照的加权 95% 区间排除无效值。此时 `overall_validity_flag = high_residual_bias_risk`，一致性结论改为不允许。事件过少时状态为 `insufficient_events_inconclusive`，不得声称“没有偏倚”。

## 公开汇总的两种来源

- **clinicaltrials_results_section**：解析 ClinicalTrials.gov API v2 的 `resultsSection`（`outcomeMeasuresModule.outcomeMeasures[].analyses[]` 的 `paramType` / `paramValue` / `ciLowerLimit` / `ciUpperLimit`，以及 `baselineCharacteristicsModule`）。离线夹具覆盖 `NCT01243424`（CAROLINA，HR 0.98，95% CI 0.84–1.14，非劣效界值 1.3）、`NCT02296125`（FLAURA，PFS HR 0.46，95% CI 0.37–0.57）以及没有结果的 `NCT09990001` / `NCT06642272`。运行时可选只读请求 `https://clinicaltrials.gov/api/v2/studies/{NCT_ID}`（超时后回退夹具）。单元测试与 `EMULATRX_OFFLINE_RCT_CATALOG=1` 不发起网络请求。
- **curated_published_summary**：调用方提供尺度、点估计、置信区间、两组样本量、主要结局名称，以及非空 `source_citation`（DOI / PMID / 引文）。区间必须包住点估计；比率尺度必须为正。

来源收据包含 `nct_id`、`source_mode`、`source_url`、`source_citation`、`retrieved_at_utc`、`has_posted_results`、汇总输入的 SHA-256、官方题目、主要结局与时间窗，以及 `ipd_used: false`。

## 基线可运输性

连续变量近似 SMD：

\[
\frac{\bar x_{\mathrm{RWD}}-\bar x_{\mathrm{RCT}}}{\sqrt{(s_{\mathrm{RWD}}^2+s_{\mathrm{RCT}}^2)/2}}
\]

二分类：

\[
\frac{p_{\mathrm{RWD}}-p_{\mathrm{RCT}}}{\sqrt{(p_{\mathrm{RWD}}(1-p_{\mathrm{RWD}})+p_{\mathrm{RCT}}(1-p_{\mathrm{RCT}}))/2}}
\]

绝对值为 0.20 及以上记 `transportability_shift_flag`。这是目标人群可运输性或效应修饰，不是治疗组内部混杂。

## RCT-DUPLICATE 指标

尺度必须相同。比率尺度使用 \(\theta=\ln(\hat\theta)\)，风险差使用 \(\theta=\hat\theta\)。标准误

\[
\mathrm{SE}=\frac{c_U-c_L}{2\times 1.959963984540054}
\]

（比率尺度先取对数）。零或负的标准误直接拒绝。

1. 标准化差 \(z=(\theta_{\mathrm{RWD}}-\theta_{\mathrm{RCT}})/\sqrt{\mathrm{SE}_{\mathrm{RWD}}^2+\mathrm{SE}_{\mathrm{RCT}}^2}\)，双侧 \(p=2\Phi(-|z|)\)，以及 \(|z|<1.959963984540054\)。
2. 95% 预测区间以 **1.96** 计算：比率尺度为 \(\exp(\theta_{\mathrm{RCT}}\pm 1.96\sqrt{\mathrm{SE}_{\mathrm{RCT}}^2+\mathrm{SE}_{\mathrm{RWD}}^2})\)，风险差不加指数。
3. 相对无效值（比率尺度为 1，风险差为 0）的统计显著性与方向是否一致。若给出 `ni_margin`，再比较两侧是否都满足 `ci_upper < ni_margin`（CAROLINA 界值为 1.3）。

## 阴性对照

输入为队列、治疗列、主分析的阳性权重 \(w_i>0\)、预先指定且取值仅为 0/1 的阴性对照列。该列不能与治疗列或主要结局列相同。任一臂事件数 `< 5` 时无结论。

加权风险 \(\hat p_a=\sum_{A_i=a} w_i N_i / \sum_{A_i=a} w_i\)，影响函数方差

\[
\widehat{\mathrm{Var}}(\hat p_a)=\frac{\sum_{A_i=a} w_i^2 (N_i-\hat p_a)^2}{(\sum_{A_i=a} w_i)^2}
\]

由此得到风险差及其稳健区间，以及 \(\ln\widehat{\mathrm{RR}}\) 的 delta-method 方差 \(\widehat{\mathrm{Var}}(\hat p_a)/\hat p_a^2\)。若同时提供阳性的时间列，再用仓库内 Lin–Wei 稳健三明治方差拟合加权 Cox 风险比，并以此作为主检验。主检验 95% 区间排除无效值则 `residual_confounding_alert = true`。即使观察协变量的 \(|\mathrm{SMD}|\) 已经小于 0.10 或 0.02，这个警报仍然有效。

`generate_hidden_confounding_benchmark_cohort` 用固定种子生成队列：观察协变量为 `age`、`bmi`、`prior_cvd`；隐藏因素 `U` 只出现在 `ground_truth_audit`，不进入分析协变量。治疗对主要结局和阴性对照的结构系数严格为 0（真实 HR = 1，真实 RD = 0）。`hidden_confounding_strength = 0` 时阴性对照覆盖无效值。

## 调用

```bash
python -m emulatrx_tte benchmark-rct \
  --synthetic-demo hidden_bias \
  --nct-id NCT01243424 \
  --output-dir /tmp/emulatrx-benchmark-demo

python -m emulatrx_tte benchmark-rct \
  --tte-result-json /tmp/tte_result.json \
  --alignment-json /tmp/alignment.json \
  --nct-id NCT01243424 \
  --output-dir /tmp/emulatrx-benchmark
```

产物为 `benchmark_report.json` 与自包含的 `benchmark_report.html`。退出码 `0` 表示报告已写出（其中包括失败关闭的诊断结论）；退出码 `2` 表示契约错误或功能开关关闭。

REST：`POST /api/v1/tte/benchmark`（与其他 `/api/v1/tte/*` 相同的 HTTP Basic）。MCP：`benchmark_rct_and_negative_control`。两者都可以传 `run_id`、`nct_id`、可选 `negative_control_outcome_col` 与 `curated_published_summary`。请求体里的个体行记录会被 `IPD_NOT_ACCEPTED` 拒绝。
