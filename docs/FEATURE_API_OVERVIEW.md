# `qitong/EmulatRx` 功能与核心 API 概览（Feature & API Overview）

- **GitHub `main` 基线**：[`11e58c2d7ed2028f4a314c33743ff831c30bb831`](https://github.com/qitong/EmulatRx/commit/11e58c2d7ed2028f4a314c33743ff831c30bb831)（[Issue #8](https://github.com/qitong/EmulatRx/issues/8) / [PR #10](https://github.com/qitong/EmulatRx/pull/10) 与 [Issue #9](https://github.com/qitong/EmulatRx/issues/9) / [PR #14](https://github.com/qitong/EmulatRx/pull/14) **均已合并关闭**；最新 `main` CI [run `45156950106`](https://github.com/qitong/EmulatRx/actions/runs/45156950106) 双作业总计 **45 项**离线测试全绿；实时核对时间：`2026-11-24 01:21 CST / 2026-11-23 17:21 UTC`）
- **定位说明**：以当前已合并 `main`（含 `emulatrx_demo` 标准库套件、`#8` 阶段一确定性 TTE 基础包与 `#9` 阶段二竞争风险生存分析工作台、端到端合成流水线及外部 Agent `skills/emulatrx-tte/SKILL.md` 契约）为唯一已实现事实源；尚未合入 `main` 的独立预览分支（如 [PR #13](https://github.com/qitong/EmulatRx/pull/13)）单独标明**“尚未合并入 `main`”**。所有合成与模拟输出严格锁定 `source_mode: "synthetic_only"` 与 `prohibited_claims`，仅供方法学教学与算法验证，**不构成真实患者数据、临床疗效证据或因果实证证明**。

---

## 1. 产品速览（当前 `main` 已可完成的 4 条核心流程）

```text
流程 A（离线合成演示）  : 固定模板 JSON ──► emulatrx_demo validate ──► demo 按种子生成 A/B 组虚拟分值 CSV + SHA-256 清单
流程 B（公开登记收据）  : 公开 NCT 编号 ──► 请求官方 API v2 白名单叶字段 ──► 内存投影哈希 ──► 仅落盘 12 键来源收据 JSON
流程 C（确定性 TTE 闭环）: 七要素方案 JSON ──► emulatrx_tte validate-protocol ──► synth 合成竞争风险/多臂队列 ──► diagnose 地标切片 + 重叠加权/IPTW 门禁 ──► estimate 加权 Aalen–Johansen CIF + Fine–Gray sHR + 原因特异 Cox csHR + RR_CIF E-value + SHA-256 审计包（或由 run-synthetic 一键串联）
流程 D（实验性本地原型）: 合成肿瘤参数 ──► 生成池化缓存 CSV（覆写式） ──► 纳排过滤 ──► 基线 PSM / IPTW 平衡 + 单事件 Cox
```

- **主要成功分支**：
  - 流程 A / B / C 校验与统计门禁全部通过后，均先在同父目录临时文件夹写齐产物与 `run_manifest.json` SHA-256 清单，再**原子重命名**发布至新输出目录，标准输出打印单行 JSON 摘要，进程退出码为 `0`。
  - 流程 C 在通过诊断门禁后，强制同表并列输出目标事件与竞争死亡双累计发生率曲线（`CIF_1(t)`、`CIF_2(t)`）、指定时点风险差/累计发生率比（`RD(t)`、`RR_CIF(t)`）、Fine–Gray 亚分布风险比（`sHR`）、目标/竞争原因特异 Cox（`csHR_1`、`csHR_2`）及 `RR_CIF(t)` E-value 敏感性参考，并内嵌防误读方法学警示与非临床声明锁。
  - 流程 D（实验性本地接口）在协变量加权/匹配后不平衡比例满足阈值时，返回调整后风险比（HR）、95% 置信区间及平衡诊断。
- **主要失败与阻断分支**：
  - **契约与显式时间戳规则阻断（退出码 `2`）**：流程 A / B / C 遇 JSON 格式错误、缺少 `time_zero`、基线模式误配 `landmark_day > 0`（`INVALID_TIME_ZERO`）、将 $t_0$/地标后评估条件放入入排或倾向模型（`POST_TIME_ZERO_ELIGIBILITY` / `POST_LANDMARK_COVARIATE_LEAK`）、策略缺少唯一索引 `0` 参考/对照臂（`INVALID_STRATEGIES`）、未声明竞争事件处理方式（`UNHANDLED_COMPETING_EVENT` / `COMPETING_RISKS_REQUIRED`）、误对 `sHR` 套用单事件 E-value（`UNSUPPORTED_EVALUE_MEASURE`）、输出目录已存在（`OUTPUT_EXISTS`）或公开元数据 HTTP 异常时，向 `stderr` 输出稳定错误码，清理临时目录并以退出码 `2` 终止，**不覆写既有目录，也不回退旧缓存**。
  - **Fail-Closed 统计与上游门禁阻断（退出码 `3`）**：
    - `emulatrx_tte diagnose` 与 `run-synthetic` 在检出空风险集（`ZERO_RISK_SET`）、正值性违背（`POSITIVITY_VIOLATION`）、Kish 有效样本量比例过低（`ESS_RATIO_BELOW_MIN`）或加权后最大 `|SMD|` 超限（`SMD_EXCEEDS_MAX`）时，仅原子发布诊断证据（`attrition_table.csv`、`balance_diagnostics.csv`、`diagnostics_summary.json` [`gate_status: "blocked"`]、`run_manifest.json`；`run-synthetic` 另含 `normalized_protocol.json`），**显式省略 `weighted_cohort.csv` 与 `estimates.json`** 并以退出码 `3` 阻断下游效应估计。
    - `emulatrx_tte estimate` 对 `gate_status != "passed"`、缺失或篡改 `weighted_cohort.csv`（SHA-256 与 `run_manifest.json` 不一致）或方案与诊断元数据不一致的输入 **100% 拒绝估计、不发布任何目录** 并以退出码 `3` 退出。

---

## 2. 入口与运行状态

当前 `main` 分支**无 Web 前端路由、无常驻 HTTP 服务端 API、无任何线上部署**（GitHub Deployments = `0`、Environments = `0`、Releases = `0`）。最新已合并 `main` CI（[run `45156950106`](https://github.com/qitong/EmulatRx/actions/runs/45156950106)）双作业总计 **45 项**离线测试全绿：`validate` 标准库作业 **26 项**（模板校验 11 项 + 合成演示 7 项 = **18 项**，加公开来源收据模拟 HTTP **8 项**）+ `tte-foundation` 科学计算作业 **19 项**（阶段一协议/加权/门禁/E-value 基座 **12 项** + 阶段二竞争风险解析基准与联合权重 oracle **7 项**）。

| 入口类型 | 实际调用入口 | 本地状态 | GitHub Actions CI 状态 | 线上部署 | 可核对源码与测试证据 |
| --- | --- | --- | --- | --- | --- |
| **标准库 CLI**<br>模板校验与合成演示 | `python -m emulatrx_demo validate \| demo` | 可离线零依赖运行（Python 3.11 标准库 + Git） | **已覆盖且全绿**：最新 `main` CI [run `45156950106`](https://github.com/qitong/EmulatRx/actions/runs/45156950106) 双作业共 **45 项**中，`validate` 作业运行 **26 项**标准库测试（本入口占 **18 项** = 模板校验 11 项 + 合成演示 7 项；另来源收据 **8 项**，`tte-foundation` 作业 **19 项**） | 未部署 | `emulatrx_demo/__main__.py`、`tests/test_demo_contract.py`、`tests/test_synthetic_demo.py` |
| **标准库 CLI**<br>公开试验来源收据 | `python -m emulatrx_demo metadata --nct-id <NCT########> --out <dir>` | 联网直连 `https://clinicaltrials.gov/api/v2` 可运行 | **已覆盖且全绿**：属于最新 `main` CI（双作业共 **45 项**）中 `validate` 作业 **26 项**里的 **8 项**（CI 使用完全合成模拟 HTTP，不发真实外网请求；另模板/合成演示 **18 项**，`tte-foundation` **19 项**） | 未部署 | `emulatrx_demo/public_metadata.py`、`tests/test_public_metadata.py` |
| **确定性科学计算 CLI、SDK 与外部 Agent Skill（#8 / PR #10 与 #9 / PR #14 已合并）**<br>TTE 方案检查、合成竞争风险队列、重叠加权门禁、竞争风险生存分析工作台与一键流水线 | `python -m emulatrx_tte validate-protocol \| synth \| diagnose \| estimate \| run-synthetic`；`from emulatrx_tte import estimate_competing_risks, run_synthetic_pipeline, compute_evalue`；外部 Agent 规范 `skills/emulatrx-tte/SKILL.md` | 需 Python 3.11 + 最小固定科学计算栈（`numpy==1.26.0`、`scipy==1.15.1`、`pandas==2.2.3`、`scikit-learn==1.5.2`、`lifelines==0.30.0`），已在 `main` 全离线确定性可调，带原子新目录防覆写保护 | **已覆盖且全绿**：最新 `main` CI [run `45156950106`](https://github.com/qitong/EmulatRx/actions/runs/45156950106) 双作业共 **45 项**中，`tte-foundation` 作业离线运行 **19 项**测试全部通过——`tests/test_tte_foundation.py` **12 项**（阶段一协议/加权/门禁/E-value 基座）+ `tests/test_tte_competing_risks.py` **7 项**（阶段二竞争风险解析基准、高竞争死亡反例、三臂兼容、联合权重 oracle、Fail-Closed 门禁退出码 `3`、E-value 阻断与逐字节 SHA-256 复现；另 `validate` 作业 **26 项**） | 未部署 | `emulatrx_tte/`、`skills/emulatrx-tte/SKILL.md`、`docs/tte-cli-and-agent-contract-v1.md`、`tests/test_tte_foundation.py`、`tests/test_tte_competing_risks.py` |
| **实验性本地接口**<br>合成肿瘤队列脚本与统计模块 | `python scripts/generate_synthetic_oncology.py`；`import emulation_utils` / `from dataframe_utils import Informatician` | **实验性本地接口**：`emulation_utils.calculate_evalues` 幂运算公式缺陷与重复定义的 `perform_balancing_method` **已由 `#8`（PR #10）合并入 `main` 修复并纳入 `tte-foundation` CI**；但 `generate_synthetic_oncology.py` 无防覆写保护、**仍会直接覆写 `trials/<trial-id>/` 既有缓存输出**，且 `adjust_immortal_time_bias` 仍为未修复实验原型 | `calculate_evalues` 与去重检查已由 `tte-foundation`（**19 项**中的阶段一 12 项套件）覆盖；其余旧版管线函数未纳入 CI | 未部署 | `scripts/generate_synthetic_oncology.py`、`emulation_utils.py`、`scripts/test_integration_statistician.py` |
| **遗留交互式 Notebook** | `jupyter notebook v1-codebase.ipynb` | 实验性交互入口，依赖外部模型 API Key 或本地 Ollama；**按 `Item #40` 拍板保持冻结不修改、不升级 `langgraph==0.2.56`** | 未覆盖 | 未部署 | `v1-codebase.ipynb`、`environment.yml`、`CLAUDE.md` |

---

## 3. 鉴权与数据边界

- **鉴权现状**：**无服务端鉴权、无用户登录或 Token 签发机制，仅限本地命令行与本地 Python 进程使用**。`emulatrx_demo validate | demo` 与 `emulatrx_tte` 全部命令均**零密钥、无数据库依赖且全离线无需外网**；仅 `emulatrx_demo metadata` 需免密钥公开只读请求官方 `https://clinicaltrials.gov/api/v2`（CI 中则使用完全合成模拟 HTTP 响应，不发真实外网请求）；遗留 Notebook 若需调用外部大模型或本地数据库，仅从进程环境变量读取 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `DB_*`，禁止硬编码密钥。
- **四类数据隔离红线**：
  1. **完全合成数据与声明锁**：`emulatrx_demo demo`、`emulatrx_tte synth` / `estimate` / `run-synthetic` 仅由固定随机种子生成与分析虚构记录（`SYN-000001..`），`dgp_truth.json` 严格标记为 `synthetic_dgp_parameters`，`estimates.json`、`audit_report.md` 与 `run_manifest.json` 强制写入 `source_mode: "synthetic_only"` 与三项 `prohibited_claims`（非真实患者数据、非临床疗效证据、非因果实证证明）。
  2. **公开试验登记元数据（方案 A 实时最小化）**：`emulatrx_demo metadata` 仅向官方 `https://clinicaltrials.gov/api/v2` 请求 17 个结构化叶路径并在内存中做白名单二次校验，**磁盘只保存 12 个来源与双 SHA-256 哈希字段（`source/metadata.json`），绝不落盘真实试验标题/入排/结果正文，绝不向任何合成生成器供数，故障时绝不回退历史缓存**。
  3. **研究方案与临床参数零硬编码**：`emulatrx_tte` 依据 `schemas/tte-protocol-v1.schema.json` 强制由研究方案/SAP 或显式 CLI/函数参数传入门禁阈值（`max_abs_smd`、`min_ess_ratio`、`positivity_epsilon`）、评估时点 `horizon_days`、`--seed` 与 `--bootstrap-iterations`；`observational_baseline` 强制 `landmark_day == 0`，`treatment_strategies[0]` 强制为唯一参考/对照臂。
  4. **真实患者级资料（IPD / MIMIC-IV）**：仓库内零真实患者记录；`estimate` 拒绝任何非 `"synthetic_only"` 模式输入。未获合法数据授权、伦理批准与签署版统计分析计划（SAP）前禁止接入真实数据。

---

## 4. Agent 可用能力卡（12 张 `main` 已实现能力）

### 能力卡 1：`emulatrx_demo.validate`（合成演示模板契约校验）
- **当前 Agent 可调状态**：**已实现（`main` 可直接通过 CLI 确定性调用）**
- **用途**：离线校验合成演示模板 JSON 是否符合 `synthetic-demo-template-v1` 严格五字段契约并生成输入哈希留痕。
- **调用入口**：`python -m emulatrx_demo validate --template <json_path> --out <new_dir>`
- **最小输入**：`<= 16 KiB` UTF-8 JSON 文件（必含 `template_version="1.0"`、`scenario="synthetic_tutorial_v1"`、`case_count` `1..10000`、`seed` `0..4294967295`、`simulation_label`）。
- **主要输出**：stdout 单行 JSON（`status: "validated_only"`）；新目录下生成 `normalized_template.json` 与 `run_manifest.json`。
- **失败状态**：退出码 `2`，stderr 打印 `INVALID_JSON` / `MISSING_FIELD` / `UNKNOWN_FIELD` / `INVALID_CASE_COUNT` / `OUTPUT_EXISTS` / `CODE_VERSION_UNAVAILABLE` 等稳定码，不生成目标目录。
- **权限或前提**：Python 3.11+，当前仓库具备有效 Git HEAD 提交；无需联网或密钥。
- **证据**：`docs/synthetic-demo-contract-v1.md`、`tests/test_demo_contract.py`。

### 能力卡 2：`emulatrx_demo.demo`（固定种子完全合成教学病例生成）
- **当前 Agent 可调状态**：**已实现（`main` 可直接通过 CLI 确定性调用）**
- **用途**：按固定种子确定性生成 `SYN-000001` 格式的 `A/B` 分组与 `0..100` 任意教学分值 CSV 及全文件 SHA-256 清单（无任何疾病或疗效含义）。
- **调用入口**：`python -m emulatrx_demo demo --template <json_path> --out <new_dir>`
- **最小输入**：同能力卡 1 的合法模板 JSON 与不存在的新输出目录路径。
- **主要输出**：stdout 单行 JSON（`status: "synthetic_demo_generated"`）；新目录下产出 `synthetic/cases.csv`、`results/summary.json`、`PURPOSE.txt`、`run_manifest.json`（同模板同种子两次运行除 `run_at_utc` 外逐字节一致）。
- **失败状态**：同能力卡 1（退出码 `2`，原子回滚无残留）。
- **权限或前提**：离线标准库 + Git，无需外部依赖。
- **证据**：`emulatrx_demo/__main__.py`、`tests/test_synthetic_demo.py`。

### 能力卡 3：`emulatrx_demo.metadata`（公开试验登记实时来源收据）
- **当前 Agent 可调状态**：**已实现（`main` 可在具备外网访问权限时通过 CLI 调用）**
- **用途**：实时核验指定公开 ClinicalTrials.gov 试验记录可达性与白名单结构，仅落盘双 SHA-256 来源收据以供审计留痕。
- **调用入口**：`python -m emulatrx_demo metadata --nct-id <NCT########> --out <new_dir>`
- **最小输入**：符合 `^NCT[0-9]{8}$` 的公开登记号与新输出目录路径。
- **主要输出**：stdout 单行 JSON（`status: "source_receipt_created"`）；新目录下仅生成 `source/metadata.json`（含 `apiVersion`、`dataTimestamp`、请求/投影字段路径、`rawResponseSha256`、`projectionSha256` 及非疗效声明）。
- **失败状态**：退出码 `2`，错误码含 `INVALID_NCT_ID`、`HTTP_ERROR`（含 404 与任何重定向）、`NETWORK_ERROR`、`RESPONSE_TOO_LARGE`、`MISSING_FIELD`、`NCT_MISMATCH`、`OUTPUT_EXISTS`。
- **权限或前提**：允许直连 `https://clinicaltrials.gov/api/v2`；无需 API Key。
- **证据**：`docs/public-metadata-receipt-v1.md`、`tests/test_public_metadata.py`。

### 能力卡 4：`scripts.generate_synthetic_oncology`（实验性本地合成肿瘤队列脚本）
- **当前 Agent 可调状态**：**实验性本地接口（需科学计算依赖，无原子防覆写保护）**
- **用途**：生成带基线混杂与 Weibull 生存时间（预设合成条件 HR 参数 = `0.65`）的合成晚期 NSCLC 原始池与纳排后分析集缓存。
- **调用入口**：`python scripts/generate_synthetic_oncology.py --trial-id NCT00000099 --no-postgres`
- **最小输入**：合成试验标识（默认 `NCT00000099`，配置见 `scripts/synthetic_config.py`）。
- **主要输出**：写入 `trials/<trial-id>/dataframes/` 下的 `dataframe.csv`、`col_dict.json`、`big_df.csv` 及单条纳排匹配集 CSV。
- **失败状态与风险**：缺少 `numpy`/`pandas` 时报 `ModuleNotFoundError`；**注意：该实验性脚本没有原子新目录保护，会直接覆写 `trials/<trial-id>/` 下的既有缓存文件，亦不生成 `run_manifest.json`**。
- **权限或前提**：本地安装 `numpy`、`pandas`；默认 `--no-postgres` 无需数据库。
- **证据**：`scripts/generate_synthetic_oncology.py`、`scripts/test_synthetic.py`。

### 能力卡 5：`emulation_utils` 实验性本地基线平衡、单事件生存与已修复 `calculate_evalues`
- **当前 Agent 可调状态**：**本地 Python 导入接口（其中 `calculate_evalues` 已由 `#8` / PR #10 合并入 `main` 修复并进 CI；其余函数仍为实验性本地接口）**
- **用途**：提供单行每患基线 DataFrame 的贪心倾向匹配（`perform_psm`）、非稳定逆概率加权（`perform_iptw`）、稳健标准误单事件 Cox（`run_cox_model`）及 VanderWeele & Ding (2017) E-value 计算（`calculate_evalues`）。
- **调用入口**：`import emulation_utils as eu`（`eu.calculate_evalues(balanced_df, primary_outcome_col, hr, ci_lower=None, ci_upper=None)`）。
- **最小输入**：包含二元结局列的 `pandas.DataFrame`、结局列名、正数风险比 `hr` 及可选置信区间上下限。
- **主要输出**：包含 `outcome_incidence`、`is_rare_outcome`、`calculation_method` 与 `e_values`（`point_estimate` 及可选 `ci_limit_closer_to_null`）的字典。
- **失败状态与已知限制**：对于存在竞争风险的结局，请使用能力卡 9/10（`emulatrx_tte.evalue.compute_evalue` 或 `emulatrx_tte estimate`）；**同时 `adjust_immortal_time_bias` 仍存在漏写治疗克隆宽限期违约删失与混池拟合删失模型的已知局限，在后续动态策略阶段修复前严禁调用**。
- **权限或前提**：`calculate_evalues` 仅需 `numpy`、`pandas`；其余函数需 `scikit-learn`、`causallib`、`lifelines`。
- **证据**：`emulation_utils.py`、`tests/test_tte_foundation.py`。

### 能力卡 6：`emulatrx_tte.validate-protocol`（目标试验七要素契约与显式规则检查器｜`main` 已合入）
- **当前 Agent 可调状态**：**已实现（`main` 可直接通过 CLI / SDK 确定性调用）**
- **用途**：离线校验目标试验模拟（TTE）七要素协议 JSON，显式拦截缺失 `time_zero`、基线模式混入地标偏移、$t_0$/地标后入排或协变量泄漏、策略参考臂缺失/冲突、竞争事件未声明处理策略、风险差/累计发生率比缺评估时点等结构与时间戳规则冲突。
- **调用入口**：`python -m emulatrx_tte validate-protocol --protocol <json_path> --out <new_dir>`（Python SDK：`emulatrx_tte.validate_protocol_dict` / `load_and_validate_protocol` / `validate_protocol_command`）。
- **最小输入**：符合 [`schemas/tte-protocol-v1.schema.json`](../schemas/tte-protocol-v1.schema.json) 的 `<= 64 KiB` UTF-8 JSON（参见 [`examples/tte_competing_risk_v1.json`](../examples/tte_competing_risk_v1.json)）。
- **主要输出**：stdout 单行 JSON（`status: "protocol_validated"`）；新目录原子生成 `normalized_protocol.json`、`protocol_check_report.md`、`run_manifest.json`。
- **失败状态**：退出码 `2`，stderr 输出 `MISSING_TIME_ZERO`、`INVALID_TIME_ZERO`、`POST_TIME_ZERO_ELIGIBILITY`、`POST_LANDMARK_COVARIATE_LEAK`、`INVALID_STRATEGIES`、`UNHANDLED_COMPETING_EVENT`、`MISSING_HORIZON`、`MISSING_DIAGNOSTIC_GATE`、`INVALID_DIAGNOSTIC_GATE`、`OUTPUT_EXISTS` 等稳定码及 `Hint:` 提示，不生成目标目录。
- **权限或前提**：Python 3.11+ 与最小固定科学计算栈，无需联网或密钥。
- **证据**：`emulatrx_tte/protocol.py`、`schemas/tte-protocol-v1.schema.json`、`tests/test_tte_foundation.py`。

### 能力卡 7：`emulatrx_tte.synth`（固定种子完全合成竞争风险与多臂基准生成器｜`main` 已合入）
- **当前 Agent 可调状态**：**已实现（`main` 可直接通过 CLI / SDK 确定性调用）**
- **用途**：基于 `np.random.default_rng(seed)` 全离线、逐字节可复现地生成带基线混杂的两臂或多臂队列（`SYN-000001..`），包含目标事件（`1`）、竞争死亡/事件（`2`）、删失（`0`）及可选地标前事件，并输出解析参考生成参数。
- **调用入口**：`python -m emulatrx_tte synth --protocol <json_path> --seed <int> --cases <1..20000> [--dgp-params <json_path>] --out <new_dir>`
- **最小输入**：合法七要素协议 JSON、非负整数种子、`1..20000` 样本量与新输出目录。
- **主要输出**：stdout 单行 JSON（`status: "synthetic_tte_cohort_generated"`）；新目录原子产出 `cohort.csv`、`dgp_truth.json`（`parameter_type: "synthetic_dgp_parameters"`）、`PURPOSE.txt`、`run_manifest.json`。
- **失败状态**：协议非法、种子/样本量越界（`INVALID_SEED` / `INVALID_CASE_COUNT`）、DGP 参数非法或目标目录已存在时以退出码 `2` 终止，无残留目录。
- **权限或前提**：全离线，零外部数据访问。
- **证据**：`emulatrx_tte/synthetic.py`、`tests/test_tte_foundation.py`。

### 能力卡 8：`emulatrx_tte.diagnose`（地标切片、两臂/多臂重叠加权与稳定/截尾 IPTW 诊断门禁｜`main` 已合入）
- **当前 Agent 可调状态**：**已实现（`main` 可直接通过 CLI / SDK 确定性调用）**
- **用途**：执行可选地标风险集切片与未来协变量泄漏拦截，基于无正则极大似然倾向模型计算两臂 `overlap`（ATO）、`iptw` / `stabilized_iptw`（支持 `--truncate-quantiles`）、多臂 `generalized_overlap`（Li & Li 2019）或 `pairwise_overlap`（强制附带 `transitivity_warning`），并按方案 `diagnostic_gates` 执行门禁裁决。
- **调用入口**：`python -m emulatrx_tte diagnose --protocol <json_path> --cohort <csv_path> --weight-method <overlap|stabilized_iptw|iptw|generalized_overlap|pairwise_overlap> [--truncate-quantiles q_low,q_high] [--covariate-offsets <json_path>] --out <new_dir>`
- **最小输入**：合法协议 JSON、队列 CSV、加权方法及新输出目录。
- **主要输出**：
  - 门禁通过（退出码 `0`）：原子发布 `attrition_table.csv`、`balance_diagnostics.csv`、`diagnostics_summary.json`（`gate_status: "passed"`）、`weighted_cohort.csv` 与 `run_manifest.json`（含 `normalized_protocol_sha256`）。
  - 门禁阻断（退出码 `3`）：原子发布诊断证据（`gate_status: "blocked"`），**不产出 `weighted_cohort.csv`**。
- **失败状态**：契约/时间泄漏违例返回退出码 `2`（不发布目录）；正值性/ESS/SMD/空风险集违背方案门禁返回退出码 `3`。
- **权限或前提**：Python 3.11 + 最小固定科学计算栈，全离线运行。
- **证据**：`emulatrx_tte/weighting.py`、`emulatrx_tte/diagnostics.py`、`tests/test_tte_foundation.py`。

### 能力卡 9：`emulatrx_tte.evalue.compute_evalue`（E-value 敏感性计算与竞争风险适用边界｜`main` 已合入）
- **当前 Agent 可调状态**：**已实现（`main` 可直接通过 SDK 调用，并由 `estimate` 自动内联调用）**
- **用途**：按 VanderWeele & Ding (*Ann Intern Med* 2017) 计算点估计与置信区间近无效侧限值 E-value，强制阻断竞争风险场景下的单事件公式误用。
- **调用入口**：`from emulatrx_tte import compute_evalue`（`compute_evalue(estimate, ci_lower=None, ci_upper=None, measure_type="RR", outcome_incidence=None, has_competing_risks=False)`）。
- **最小输入**：正数效应点估计 `estimate`、`measure_type`（`"RR"` | `"RR_CIF"` | `"HR"` | 罕见结局 `"csHR"`）及 HR 转换所需的 `outcome_incidence`。
- **主要输出**：返回 `approximated_rr`、`point_estimate`、`ci_limit_closer_to_null`（CI 跨 `1.0` 时精确返回 `1.0`）及显式 `interpretation_note`。
- **失败状态**：传入 `"sHR"`，或在竞争风险且 `outcome_incidence > 0.15` 时传入 `"csHR"` / `"HR"`，显式抛出 `ContractError("UNSUPPORTED_EVALUE_MEASURE", ...)`。
- **权限或前提**：Python 3.11 + `numpy`。
- **证据**：`emulatrx_tte/evalue.py`、`tests/test_tte_foundation.py`、`tests/test_tte_competing_risks.py`。

### 能力卡 10：`emulatrx_tte.estimate`（Fail-Closed 加权竞争风险生存分析工作台｜#9 / PR #14 已合并入 `main`）
- **当前 Agent 可调状态**：**已实现（PR #14 已合并入 `main`，可直接通过 CLI 与 Python SDK 确定性调用）**
- **用途**：对通过上游 `diagnose` 门禁校验的加权队列同时估计目标事件（`1`）与竞争死亡（`2`）的加权 Aalen–Johansen 累计发生率曲线（`CIF_1(t)`、`CIF_2(t)` 及有偏 `1-KM` 对照参考）、指定时点绝对风险差 `RD(t)` 与累计发生率比 `RR_CIF(t)`（含确定性同种子个体级分层 Bootstrap CI：固定复用 `weighted_cohort.csv` 第一阶段基线倾向权重 `baseline_propensity_weights="fixed"`，在开启 `--censoring-ipcw` 时逐次重估删失权重）、Fine–Gray 亚分布比例风险比（`sHR`，反向 KM / 联合稳定化 IPCW 风险集 + 个体聚类稳健方差）、目标/竞争原因特异加权 Cox（`csHR_1`、`csHR_2`）及 `RR_CIF(t)` E-value 敏感性参考，强制同表并列展示并输出防误读警示。
- **调用入口**：`python -m emulatrx_tte estimate --protocol <json_path> --diagnostics-dir <diag_dir> [--weighted-cohort <csv_path>] --seed <int> --bootstrap-iterations <1..5000> [--horizons h1,h2,...] [--censoring-ipcw none|arm_stratified|covariate_cox] [--ci-level 0.95] --out <new_dir>`（Python SDK：`emulatrx_tte.estimate_competing_risks`、`weighted_aalen_johansen_cif`、`fit_fine_gray_shr`、`fit_weighted_cause_specific_cox`）。
- **最小输入**：声明 `competing_event_handling="explicit_cif_and_subdistribution"` 的合法协议 JSON、通过门禁的 `diagnostics_dir`、显式整数随机种子、Bootstrap 次数与新输出目录。
- **主要输出**：stdout 单行 JSON（`status: "tte_competing_risks_estimated"`）；新目录原子产出 `estimates.json`（锁定 `source_mode: "synthetic_only"`、`prohibited_claims`、`co_presented_comparison_table`、`methodological_warnings`）、`curves/*.csv`、`audit_report.md`、`run_manifest.json`。
- **失败状态**：
  - 退出码 `2`（`ContractError`）：协议未开启竞争风险、参数越界、目标目录已存在或路径冲突，不发布任何文件；
  - 退出码 `3`（`GateBlockedError`）：上游 `gate_status != "passed"`、缺失 `weighted_cohort.csv`、SHA-256 与 `run_manifest.json` 不一致或协议与诊断元数据不匹配，**100% 拒绝估计且不发布任何文件**。
- **权限或前提**：Python 3.11 + 最小固定科学计算栈，全离线运行。
- **证据**：`emulatrx_tte/competing_risks.py`、`docs/tte-cli-and-agent-contract-v1.md`、`tests/test_tte_competing_risks.py`。

### 能力卡 11：`emulatrx_tte.run-synthetic`（端到端离线合成竞争风险流水线与 SHA-256 审计包｜#9 / PR #14 已合并入 `main`）
- **当前 Agent 可调状态**：**已实现（PR #14 已合并入 `main`，可直接通过 CLI 与 Python SDK 确定性调用）**
- **用途**：一键离线串联 `validate-protocol` → `synth` → `diagnose` → `estimate`，使用同级临时目录原子发布完整协议、流失与平衡诊断、双 CIF 曲线、并列效应估计与全文件 SHA-256 清单，支持同方案同种子逐字节复现核验。
- **调用入口**：`python -m emulatrx_tte run-synthetic --protocol <json_path> --seed <int> --cases <1..20000> --weight-method <choice> --bootstrap-iterations <1..5000> [--dgp-params <json_path>] [--truncate-quantiles q_low,q_high] [--horizons h1,h2,...] [--censoring-ipcw none|arm_stratified|covariate_cox] [--ci-level 0.95] --out <new_dir>`（Python SDK：`emulatrx_tte.run_synthetic_pipeline`）。
- **最小输入**：合法竞争风险协议 JSON、种子、样本量、加权方法、Bootstrap 次数与新输出目录。
- **主要输出**：
  - 门禁全部通过（退出码 `0`）：原子发布 `normalized_protocol.json`、`protocol_check_report.md`、`dgp_truth.json`、`PURPOSE.txt`、`attrition_table.csv`、`balance_diagnostics.csv`、`diagnostics_summary.json`、`weighted_cohort.csv`、`estimates.json`、`curves/*.csv`、`audit_report.md` 与 `run_manifest.json`（两次同种子运行除 `run_at_utc` 外逐字节一致）。
  - 诊断门禁阻断（退出码 `3`）：原子发布 `normalized_protocol.json`、`attrition_table.csv`、`balance_diagnostics.csv`、`diagnostics_summary.json`（`gate_status: "blocked"`）与 `run_manifest.json`，**绝不发布 `weighted_cohort.csv` 与 `estimates.json`**。
- **失败状态**：协议/参数非法或输出目录已存在返回退出码 `2`（无目录发布）；诊断门禁未通过返回退出码 `3`。
- **权限或前提**：Python 3.11 + 最小固定科学计算栈，全离线运行。
- **证据**：`emulatrx_tte/pipeline.py`、`tests/test_tte_competing_risks.py`。

### 能力卡 12：`skills/emulatrx-tte/SKILL.md`（外部 Agent 标准调用技能与防误读契约｜#9 / PR #14 已合并入 `main`）
- **当前 Agent 可调状态**：**已实现（PR #14 已合并入 `main`，可供外部 Agent 直接加载执行）**
- **用途**：向外部 Agent 提供标准化的子命令调用序列、退出码 `0/2/3` 处理规则、Fail-Closed 门禁阻断纪律、竞争风险五量同表解读规则（防止高竞争死亡致 `sHR < 1` 被误读为生物学保护作用、标注 Hernán 2010 `csHR` 选择偏倚、锁定 `RR_CIF` E-value 边界）以及 `synthetic_only` / `prohibited_claims` 合规红线。
- **入口路径**：[`skills/emulatrx-tte/SKILL.md`](../skills/emulatrx-tte/SKILL.md) 与契约参考 [`docs/tte-cli-and-agent-contract-v1.md`](tte-cli-and-agent-contract-v1.md)。
- **证据**：`skills/emulatrx-tte/SKILL.md`、`docs/tte-cli-and-agent-contract-v1.md`、`tests/test_tte_competing_risks.py`。

### 最小完全合成调用示例（当前 `main` 离线可复现）

```bash
# 1. 标准库固定模板合成演示（零外部依赖）
python -m emulatrx_demo demo \
  --template examples/synthetic_demo_v1.json \
  --out /tmp/emulatrx-example-run

# 2. TTE 端到端合成竞争风险流水线（#9 / PR #14 已合入 main：协议检查 -> 合成队列 -> 重叠加权门禁 -> CIF/sHR/csHR/E-value 审计包）
python -m emulatrx_tte run-synthetic \
  --protocol examples/tte_competing_risk_v1.json \
  --seed 42 --cases 600 \
  --weight-method overlap \
  --bootstrap-iterations 200 \
  --out /tmp/tte-run-synthetic

# 3. 分步流水线调用（validate-protocol -> synth -> diagnose -> estimate）
python -m emulatrx_tte validate-protocol \
  --protocol examples/tte_competing_risk_v1.json \
  --out /tmp/tte-protocol-check
python -m emulatrx_tte synth \
  --protocol /tmp/tte-protocol-check/normalized_protocol.json \
  --seed 42 --cases 600 \
  --out /tmp/tte-synth
python -m emulatrx_tte diagnose \
  --protocol /tmp/tte-protocol-check/normalized_protocol.json \
  --cohort /tmp/tte-synth/cohort.csv \
  --weight-method overlap \
  --out /tmp/tte-diagnose
python -m emulatrx_tte estimate \
  --protocol /tmp/tte-protocol-check/normalized_protocol.json \
  --diagnostics-dir /tmp/tte-diagnose \
  --seed 42 --bootstrap-iterations 200 \
  --out /tmp/tte-estimate
```

---

## 5. 主要 Branch 与关键逻辑分支（GitHub 实时快照：`2026-11-24 01:21 CST`）

| Git 分支 | 当前提交 / 位置 | 关联 PR / Issue 与状态 |
| --- | --- | --- |
| `main` | [`11e58c2`](https://github.com/qitong/EmulatRx/commit/11e58c2d7ed2028f4a314c33743ff831c30bb831)（远端默认分支） | 已合并 [PR #4](https://github.com/qitong/EmulatRx/pull/4)（#1 模板契约）、[PR #5](https://github.com/qitong/EmulatRx/pull/5)（#3 合成演示）、[PR #6](https://github.com/qitong/EmulatRx/pull/6)（#2 来源收据）、[PR #10](https://github.com/qitong/EmulatRx/pull/10)（[#8](https://github.com/qitong/EmulatRx/issues/8) 阶段一确定性 TTE 基础包）及 [PR #14](https://github.com/qitong/EmulatRx/pull/14)（[#9](https://github.com/qitong/EmulatRx/issues/9) 阶段二竞争风险生存分析工作台与外部 Agent `SKILL.md`）；最新 `main` CI [run `45156950106`](https://github.com/qitong/EmulatRx/actions/runs/45156950106) 双作业总计 **45/45** 全绿（`validate` **26** 项 [`18+8`] + `tte-foundation` **19** 项 [`12+7`]），未部署 |
| `qitong/emulatrx-feature-api-overview` | 基于 `origin/main`（[`11e58c2`](https://github.com/qitong/EmulatRx/commit/11e58c2d7ed2028f4a314c33743ff831c30bb831)）变基收口的只读概览文档分支 | 关联开放 [Issue #11](https://github.com/qitong/EmulatRx/issues/11) 与可审阅 [PR #12](https://github.com/qitong/EmulatRx/pull/12)，在 PR #10 与 PR #14 相继合入后校准本文档至实时已合并 `main` 基线与双作业 45 项 CI 口径并关闭 Issue #11 |
| `qitong/emulatrx-ymadmin-preview` | 快照时刻 `8b8bc63`（在途持续演进的远端预览分支） | 关联开放 [PR #13](https://github.com/qitong/EmulatRx/pull/13)（在途演进的独立私有合成预览工作台分支）；**尚未合并入 `main`**，不计入当前 `main` 已实现能力 |
| `qitong/emulatrx-issue9-competing-risks` | `152af06`（远端历史分支，已合入 `11e58c2`） | 关联 [Issue #9](https://github.com/qitong/EmulatRx/issues/9) 与 [PR #14](https://github.com/qitong/EmulatRx/pull/14)，**均已合并关闭** |
| `qitong/emulatrx-issue8-tte-foundation` | `5ecfd02`（远端历史分支，已合入 `ed9791d`） | 关联 [Issue #8](https://github.com/qitong/EmulatRx/issues/8) 与 [PR #10](https://github.com/qitong/EmulatRx/pull/10)，**均已合并关闭** |
| `qitong/emulatrx-contract`、`qitong/emulatrx-issue3-synthetic`、`qitong/emulatrx-issue2-live-metadata` | `5289f00`、`2663802`、`47d4dbc`（远端历史分支） | 对应 [PR #4](https://github.com/qitong/EmulatRx/pull/4) / [PR #5](https://github.com/qitong/EmulatRx/pull/5) / [PR #6](https://github.com/qitong/EmulatRx/pull/6) **均已合并关闭**，无未决变更 |

- **关键业务与治理逻辑分支**：
  - **校验/门禁与估计全部通过（退出码 `0`）**：输入符合七要素契约、上游/当次统计诊断门禁 `gate_status == "passed"` 且目标目录不存在 → 同级临时目录写齐产物与 SHA-256 `run_manifest.json` 后原子重命名发布。
  - **契约/显式偏倚规则阻断（退出码 `2`）**：结构缺失、时间零点/地标偏移矛盾、$t_0$ 后入排或协变量泄漏、参考臂未明确定义、单事件 E-value 误用、目录已存在或公开元数据校验失败 → 立即清理临时目录、保留现场不被污染，返回稳定错误码。
  - **Fail-Closed 统计与上游门禁阻断（退出码 `3`）**：`diagnose` / `run-synthetic` 检出正值性违背、ESS 比例不足、加权后残余失衡或空风险集 → 仅发布流失与平衡诊断证据（不产出 `weighted_cohort.csv` 与 `estimates.json`）；`estimate` 检出未过门禁、缺表、哈希篡改或协议失配 → 100% 拒绝估计且不发布任何文件。
  - **待人工审批（`change_class=framework`）**：涉及核心契约、因果/统计推断逻辑或证据边界的 PR，即使离线合成 CI 与异 Agent 独立复核全过，仍保持待审状态，**必须由启曈本人审阅合并，绝不自动合并或部署**。

---

## 6. 限制与下一步

- **当前 `main` 明确限制（含 `#8` / PR #10 与 `#9` / PR #14 已合并边界）**：
  1. **聚焦静态基线与固定地标竞争风险设计**：`#8`（PR #10）与 `#9`（PR #14）已在 `main` 完整覆盖静态基线分配（`observational_baseline`）与显式固定地标分配（`observational_landmark`）下的重叠加权/IPTW 门禁与竞争风险生存估计，尚未包含双侧宽限期克隆删失加权（CCW）或时变混杂 g-formula；
  2. **显式规则检查与敏感性指标的方法学边界**：`validate-protocol`、`diagnose` 与 `RR_CIF` E-value 无法检测真实世界中的未测混杂、测量误差或未建模信息性删失，不能证明因果有效性；
  3. **遗留模块冻结与实验性本地接口限制**：按启曈在 `Item #40` 的拍板，`v1-codebase.ipynb` 与 `environment.yml`（`langgraph==0.2.56`）保持冻结不动；`scripts/generate_synthetic_oncology.py` 仍为无原子防覆写保护的实验性本地脚本（可能直接覆写 `trials/<trial-id>/` 既有缓存输出），`emulation_utils.adjust_immortal_time_bias` 简化 CCW 原型尚未修复。
- **在途与下一步**：
  1. 审阅合入 [PR #12](https://github.com/qitong/EmulatRx/pull/12) 关闭 [Issue #11](https://github.com/qitong/EmulatRx/issues/11)，完成阶段一（`#8` / PR #10）与阶段二（`#9` / PR #14）合入后 `docs/FEATURE_API_OVERVIEW.md` 的实时 `main`（`11e58c2`）基线与双作业 45 项 CI 口径收口；
  2. 独立推进在途开放 [PR #13](https://github.com/qitong/EmulatRx/pull/13)（私有合成预览工作台分支，当前尚未合并入 `main`）；
  3. **后续待启曈决定的扩展点**：在 `#8`、`#9` 已验收合入基础上，是否启动第二批动态策略能力（双侧宽限期分臂 IPCW 克隆删失加权 CCW、逐月序贯目标试验、离散时间 Pooled Logistic / 参数 g-formula），以及未来若需接入获授权真实队列时的独立数据合规与签署版 SAP 审批流程。
