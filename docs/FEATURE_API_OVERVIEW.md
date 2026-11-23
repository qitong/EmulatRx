# `qitong/EmulatRx` 功能与核心 API 概览（Feature & API Overview）

- **GitHub `main` 基线**：[`bed656b02444b8c2b3989595745dab5854ef128c`](https://github.com/qitong/EmulatRx/commit/bed656b02444b8c2b3989595745dab5854ef128c)；**本文档随 [Issue #8](https://github.com/qitong/EmulatRx/issues/8) / [Draft PR #10](https://github.com/qitong/EmulatRx/pull/10) 同步更新**（更新时间：`2026-11-23 CST`）
- **定位说明**：以 `main` 既有能力加本 PR（`#8` 阶段一确定性 TTE 基础包）交付能力为已实现范围；排队中的阶段二变更（[#9](https://github.com/qitong/EmulatRx/issues/9)）单独标明**“排队待合入、尚未合并”**。所有合成与模拟输出仅供方法学教学与算法验证，**不构成真实患者证据或临床疗效结论**。

---

## 1. 产品速览（`main` 既有流程 + `#8` 新增确定性 TTE 流程）

```text
流程 A（离线合成演示）  : 固定模板 JSON ──► emulatrx_demo validate ──► demo 按种子生成 A/B 组虚拟分值 CSV + SHA-256 清单
流程 B（公开登记收据）  : 公开 NCT 编号 ──► 请求官方 API v2 白名单叶字段 ──► 内存投影哈希 ──► 仅落盘 12 键来源收据 JSON
流程 C（确定性 TTE 门禁）: 七要素方案 JSON ──► emulatrx_tte validate-protocol ──► synth 合成竞争风险/多臂队列 ──► diagnose 地标切片 + 重叠加权/IPTW 门禁 + evalue 敏感性参考
流程 D（实验性本地原型）: 合成肿瘤参数 ──► 生成池化缓存 CSV（覆写式） ──► 纳排过滤 ──► 基线 PSM / IPTW 平衡 + 单事件 Cox
```

- **主要成功分支**：
  - 流程 A / B / C 校验与统计门禁全部通过后，均先在同父目录临时文件夹写齐产物与 `run_manifest.json` SHA-256 清单，再**原子重命名**发布至新输出目录，标准输出打印单行 JSON 摘要，进程退出码为 `0`。
  - 流程 D（实验性本地接口）在协变量加权/匹配后不平衡比例满足阈值时，返回调整后风险比（HR）、95% 置信区间及平衡诊断。
- **主要失败与阻断分支**：
  - **契约与显式时间戳规则阻断（退出码 `2`）**：流程 A / B / C 遇 JSON 格式错误、缺少 `time_zero`、基线模式误配 `landmark_day > 0`（`INVALID_TIME_ZERO`）、将 $t_0$/地标后评估条件放入入排或倾向模型（`POST_TIME_ZERO_ELIGIBILITY` / `POST_LANDMARK_COVARIATE_LEAK`）、策略缺少唯一索引 `0` 参考/对照臂（`INVALID_STRATEGIES`）、未声明竞争事件处理方式（`UNHANDLED_COMPETING_EVENT`）、输出目录已存在（`OUTPUT_EXISTS`）或公开元数据 HTTP 异常时，向 `stderr` 输出稳定错误码，清理临时目录并以退出码 `2` 终止，**不覆写既有目录，也不回退旧缓存**。
  - **统计诊断门禁阻断（退出码 `3`）**：`emulatrx_tte diagnose` 在地标切片后若出现空风险集（`ZERO_RISK_SET`）、倾向得分违背正值性阈值（`POSITIVITY_VIOLATION`）、Kish 有效样本量比例过低（`ESS_RATIO_BELOW_MIN`）或加权后最大协变量 `|SMD|` 超出方案阈值（`SMD_EXCEEDS_MAX`），会原子发布完整诊断文件（`attrition_table.csv`、`balance_diagnostics.csv`、`diagnostics_summary.json` [`gate_status: "blocked"`]、`run_manifest.json`），**显式省略 `weighted_cohort.csv`** 并以退出码 `3` 阻断下游自动估计。

---

## 2. 入口与运行状态

当前仓库**无 Web 前端路由、无常驻 HTTP 服务端 API、无任何线上部署**（GitHub Deployments = `0`、Environments = `0`、Releases = `0`）。

| 入口类型 | 实际调用入口 | 本地状态 | GitHub Actions CI 状态 | 线上部署 | 可核对源码与测试证据 |
| --- | --- | --- | --- | --- | --- |
| **标准库 CLI**<br>模板校验与合成演示 | `python -m emulatrx_demo validate \| demo` | 可离线零依赖运行（Python 3.11 标准库 + Git） | **已覆盖且全绿**：`Synthetic template contract / validate` 作业运行 **26 项**标准库测试（模板校验 11 项 + 合成演示 7 项 + 来源收据模拟 HTTP 8 项） | 未部署 | `emulatrx_demo/__main__.py`、`tests/test_demo_contract.py`、`tests/test_synthetic_demo.py` |
| **标准库 CLI**<br>公开试验来源收据 | `python -m emulatrx_demo metadata --nct-id <NCT########> --out <dir>` | 联网直连 `https://clinicaltrials.gov/api/v2` 可运行 | **已覆盖且全绿**：属于上述 `validate` 作业 **26 项**中的 **8 项**（CI 使用完全合成模拟 HTTP，不发真实外网请求） | 未部署 | `emulatrx_demo/public_metadata.py`、`tests/test_public_metadata.py` |
| **确定性科学计算 CLI 与 SDK（#8 新增）**<br>TTE 方案规则检查、合成竞争风险队列、重叠加权门禁与 E-value 边界 | `python -m emulatrx_tte validate-protocol \| synth \| diagnose`；`from emulatrx_tte import compute_evalue` | 需 Python 3.11 + 最小固定科学计算栈（`numpy==1.26.0`、`scipy==1.15.1`、`pandas==2.2.3`、`scikit-learn==1.5.2`、`lifelines==0.30.0`），全离线确定性运行，带原子新目录防覆写保护 | **已纳入 `.github/workflows/synthetic-contract.yml` 新增 `tte-foundation` 作业且全绿**：离线运行 `tests/test_tte_foundation.py`（覆盖正向精确平衡/有界性/逐字节复现/E-value 解析核对与反向退出码 `2`/`3` 阻断） | 未部署 | `emulatrx_tte/`、`schemas/tte-protocol-v1.schema.json`、`docs/tte-foundation-v1.md`、`tests/test_tte_foundation.py` |
| **实验性本地接口**<br>合成肿瘤队列脚本与统计模块 | `python scripts/generate_synthetic_oncology.py`；`import emulation_utils` / `from dataframe_utils import Informatician` | **实验性本地接口**：`emulation_utils.calculate_evalues` 幂运算公式缺陷与重复定义的 `perform_balancing_method` **已随 `#8` 修复并纳入 `tte-foundation` CI**；但 `generate_synthetic_oncology.py` 仍会直接覆写 `trials/<trial-id>/` 缓存，且 `adjust_immortal_time_bias` 仍为未修复实验原型 | `calculate_evalues` 与去重检查已由 `tte-foundation` 覆盖；其余旧版管线函数未纳入 CI | 未部署 | `scripts/generate_synthetic_oncology.py`、`emulation_utils.py`、`scripts/test_integration_statistician.py` |
| **遗留交互式 Notebook** | `jupyter notebook v1-codebase.ipynb` | 实验性交互入口，依赖外部模型 API Key 或本地 Ollama；**按 `Item #40` 拍板保持冻结不修改、不升级 `langgraph==0.2.56`** | 未覆盖 | 未部署 | `v1-codebase.ipynb`、`environment.yml`、`CLAUDE.md` |

---

## 3. 鉴权与数据边界

- **鉴权现状**：**无服务端鉴权、无用户登录或 Token 签发机制，仅限本地命令行与本地 Python 进程使用**。`emulatrx_demo` 与 `emulatrx_tte` 全部命令无需任何密钥、数据库或外网连接；遗留 Notebook 若需调用外部大模型或本地数据库，仅从进程环境变量读取 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `DB_*`，禁止硬编码密钥。
- **四类数据隔离红线**：
  1. **完全合成数据**：`emulatrx_demo demo`、`emulatrx_tte synth` 与 `generate_synthetic_oncology.py` 仅由固定随机种子（SHA-256 计数器或 `np.random.default_rng(seed)`）和预设数学参数生成虚构记录（`SYN-000001..`），`dgp_truth.json` 严格标记为 `synthetic_dgp_parameters`，不含任何真实患者信息且绝不表述为真实临床疗效。
  2. **公开试验登记元数据（方案 A 实时最小化）**：`emulatrx_demo metadata` 仅向官方 `https://clinicaltrials.gov/api/v2` 请求 17 个结构化叶路径并在内存中做白名单二次校验，**磁盘只保存 12 个来源与双 SHA-256 哈希字段（`source/metadata.json`），绝不落盘真实试验标题/入排/结果正文，绝不向任何合成生成器供数，故障时绝不回退历史缓存**。
  3. **研究方案与临床参数**：`emulatrx_tte validate-protocol` 依据 `schemas/tte-protocol-v1.schema.json` 强制由研究方案/SAP 显式传入全部时间窗与诊断门禁参数（`max_abs_smd`、`min_ess_ratio`、`positivity_epsilon`），代码零硬编码临床阈值；`observational_baseline` 强制 `landmark_day == 0`，`treatment_strategies` 强制索引 `0` 为唯一 `control`/`comparator`/`reference` 参考臂。输出单显式声明：**仅检测输入结构与已观测时间戳显式冲突，不能检测未测混杂、测量误差或信息性删失，不证明因果有效**。
  4. **真实患者级资料（IPD / MIMIC-IV）**：仓库内零真实患者记录；未获合法数据授权、伦理批准与签署版统计分析计划（SAP）前禁止接入，任何合成或演示输出严禁表述为临床疗效证据。

---

## 4. Agent 可用能力卡（9 张已实现能力 + 1 张排队未合并预告）

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
- **当前 Agent 可调状态**：**本地 Python 导入接口（其中 `calculate_evalues` 已在 `#8` 修复并进 CI；其余函数仍为实验性本地接口）**
- **用途**：提供单行每患基线 DataFrame 的贪心倾向匹配（`perform_psm`）、非稳定逆概率加权（`perform_iptw`）、稳健标准误单事件 Cox（`run_cox_model`）及 VanderWeele & Ding (2017) E-value 计算（`calculate_evalues`）。
- **调用入口**：`import emulation_utils as eu`（`eu.calculate_evalues(balanced_df, primary_outcome_col, hr, ci_lower=None, ci_upper=None)`）。
- **最小输入**：包含二元结局列的 `pandas.DataFrame`、结局列名、正数风险比 `hr` 及可选置信区间上下限。
- **主要输出**：包含 `outcome_incidence`、`is_rare_outcome`、`calculation_method` 与 `e_values`（`point_estimate` 及可选 `ci_limit_closer_to_null`）的字典。
- **失败状态与已知限制**：`#8` 已修复原第 986 行 `0.5 ** np.sqrt(...)` 幂运算缺陷并清理重复定义的 `perform_balancing_method`；**但对于存在竞争风险的结局，请改用下方能力卡 9 `emulatrx_tte.evalue.compute_evalue`；同时 `adjust_immortal_time_bias` 仍存在漏写治疗克隆宽限期违约删失的已知缺陷，在后续动态策略阶段修复前严禁调用**。
- **权限或前提**：`calculate_evalues` 仅需 `numpy`、`pandas`；其余函数需 `scikit-learn`、`causallib`、`lifelines`。
- **证据**：`emulation_utils.py`、`tests/test_tte_foundation.py`、`scripts/test_integration_statistician.py`。

### 能力卡 6：`emulatrx_tte.validate-protocol`（目标试验七要素契约与显式规则检查器｜#8 本 PR 实现）
- **当前 Agent 可调状态**：**已在本分支实现并通过 CI（随 PR #10 合入 `main`）**
- **用途**：离线校验目标试验模拟（TTE）七要素协议 JSON，显式拦截缺失 `time_zero`、基线模式混入地标偏移、$t_0$/地标后入排或协变量泄漏、策略参考臂缺失/冲突、竞争事件未声明处理策略、风险差/累计发生率比缺评估时点等结构与时间戳规则冲突。
- **调用入口**：`python -m emulatrx_tte validate-protocol --protocol <json_path> --out <new_dir>`（Python SDK：`emulatrx_tte.validate_protocol_dict` / `load_and_validate_protocol`）。
- **最小输入**：符合 [`schemas/tte-protocol-v1.schema.json`](../schemas/tte-protocol-v1.schema.json) 的 `<= 64 KiB` UTF-8 JSON（参见 [`examples/tte_competing_risk_v1.json`](../examples/tte_competing_risk_v1.json)，所有时间窗与 `diagnostic_gates` 须显式传入，且 `treatment_strategies[0]` 须为唯一 `control`/`comparator`/`reference` 参考臂）。
- **主要输出**：stdout 单行 JSON（`status: "protocol_validated"`）；新目录原子生成 `normalized_protocol.json`、`protocol_check_report.md`、`run_manifest.json`（均内嵌显式方法学免责声明）。
- **失败状态**：退出码 `2`，stderr 输出 `MISSING_TIME_ZERO`、`INVALID_TIME_ZERO`、`POST_TIME_ZERO_ELIGIBILITY`、`POST_LANDMARK_COVARIATE_LEAK`、`INVALID_STRATEGIES`、`UNHANDLED_COMPETING_EVENT`、`MISSING_HORIZON`、`MISSING_DIAGNOSTIC_GATE`、`INVALID_DIAGNOSTIC_GATE`、`OUTPUT_EXISTS` 等稳定码及 `Hint:` 修正提示，不生成目标目录。
- **权限或前提**：Python 3.11+ 与最小固定科学计算栈，无需联网或密钥。
- **证据**：`emulatrx_tte/protocol.py`、`schemas/tte-protocol-v1.schema.json`、`tests/test_tte_foundation.py`。

### 能力卡 7：`emulatrx_tte.synth`（固定种子完全合成竞争风险与多臂基准生成器｜#8 本 PR 实现）
- **当前 Agent 可调状态**：**已在本分支实现并通过 CI（随 PR #10 合入 `main`）**
- **用途**：基于 `np.random.default_rng(seed)` 全离线、逐字节可复现地生成带基线混杂的两臂或多臂（多项逻辑斯蒂）队列（`SYN-000001..`），包含目标事件（`1`）、竞争死亡/事件（`2`）、行政/独立删失（`0`）及可选地标前事件，并输出解析参考生成参数供估计器验证。
- **调用入口**：`python -m emulatrx_tte synth --protocol <json_path> --seed <int> --cases <1..20000> [--dgp-params <json_path>] --out <new_dir>`
- **最小输入**：合法的七要素协议 JSON、非负整数随机种子、`1..20000` 样本量与新输出目录。
- **主要输出**：stdout 单行 JSON（`status: "synthetic_tte_cohort_generated"`）；新目录原子产出 `cohort.csv`、`dgp_truth.json`（顶层固定标注 `parameter_type: "synthetic_dgp_parameters"`）、`PURPOSE.txt`、`run_manifest.json`。
- **失败状态**：协议非法、种子/样本量越界（`INVALID_SEED` / `INVALID_CASE_COUNT`）、DGP 参数非法或目标目录已存在时以退出码 `2` 终止，无残留目录。
- **权限或前提**：全离线，不读取任何患者文件、数据库或公开试验登记响应。
- **证据**：`emulatrx_tte/synthetic.py`、`tests/test_tte_foundation.py`。

### 能力卡 8：`emulatrx_tte.diagnose`（地标切片、两臂/多臂重叠加权与稳定/截尾 IPTW 诊断门禁｜#8 本 PR 实现）
- **当前 Agent 可调状态**：**已在本分支实现并通过 CI（随 PR #10 合入 `main`）**
- **用途**：执行可选地标风险集切片与未来协变量泄漏拦截，基于无正则极大似然倾向模型计算两臂 Overlap Weighting（ATO，个体原始权重严格位于 `(0,1)`、线性协变量有限样本精确平衡 `|SMD| < 1e-6`）、`iptw` / `stabilized_iptw`（支持 `--truncate-quantiles` 分位数截尾）、多臂 `generalized_overlap`（Li & Li 2019 共同均势亚群）或 `pairwise_overlap`（成对 ATO，强制写入非传递性警告 `transitivity_warning`），并按方案 `diagnostic_gates` 执行门禁裁决。
- **调用入口**：`python -m emulatrx_tte diagnose --protocol <json_path> --cohort <csv_path> --weight-method <overlap|stabilized_iptw|iptw|generalized_overlap|pairwise_overlap> [--truncate-quantiles q_low,q_high] [--covariate-offsets <json_path>] --out <new_dir>`
- **最小输入**：合法协议 JSON、含 `synthetic_case_id`/`treatment_arm`/`duration_day`/`event_code` 与基线协变量的队列 CSV、加权方法及新输出目录。
- **主要输出**：
  - 门禁通过（退出码 `0`）：原子发布 `attrition_table.csv`、`balance_diagnostics.csv`、`diagnostics_summary.json`（`gate_status: "passed"`）、`weighted_cohort.csv` 与 `run_manifest.json`。
  - 门禁阻断（退出码 `3`）：原子发布 `attrition_table.csv`、`balance_diagnostics.csv`、`diagnostics_summary.json`（`gate_status: "blocked"` 及具体 `gate_failures`）与 `run_manifest.json`，**不产出 `weighted_cohort.csv`**。
- **失败状态**：输入契约错误或地标后协变量泄漏（`POST_LANDMARK_COVARIATE_LEAK`）返回退出码 `2`（不发布目录）；正值性/ESS/SMD/空风险集违背方案门禁返回退出码 `3`（发布诊断证据但阻断加权队列流出）。
- **权限或前提**：Python 3.11 + 最小固定科学计算栈，全离线运行。
- **证据**：`emulatrx_tte/weighting.py`、`emulatrx_tte/diagnostics.py`、`tests/test_tte_foundation.py`。

### 能力卡 9：`emulatrx_tte.evalue.compute_evalue`（E-value 敏感性计算与竞争风险适用边界｜#8 本 PR 实现）
- **当前 Agent 可调状态**：**已在本分支实现并通过 CI（随 PR #10 合入 `main`）**
- **用途**：按 VanderWeele & Ding (*Ann Intern Med* 2017) 计算点估计与置信区间近无效侧限值 E-value，并强制拦截竞争风险场景下的单事件效应量误用。
- **调用入口**：`from emulatrx_tte import compute_evalue`（`compute_evalue(estimate, ci_lower=None, ci_upper=None, measure_type="RR", outcome_incidence=None, has_competing_risks=False)`）。
- **最小输入**：正数效应点估计 `estimate`、`measure_type`（`"RR"` | `"RR_CIF"` | `"HR"` | 罕见结局 `"csHR"`）及 HR 转换所需的 `outcome_incidence`。
- **主要输出**：返回 `approximated_rr`、`point_estimate`、`ci_limit_closer_to_null`（CI 跨 `1.0` 时精确返回 `1.0`）及显式 `interpretation_note`（声明 E-value 不证明无未测混杂；`"RR_CIF"` 反映含竞争死亡通路在内的总累计发生率效应）。
- **失败状态**：若传入 Fine–Gray `"sHR"`，或在存在竞争风险且 `outcome_incidence > 0.15` 时传入 `"csHR"` / `"HR"`，显式抛出 `ContractError("UNSUPPORTED_EVALUE_MEASURE", ...)`。
- **权限或前提**：Python 3.11 + `numpy`。
- **证据**：`emulatrx_tte/evalue.py`、`tests/test_tte_foundation.py`。

### 能力卡 10（排队预告｜尚未合并）：[#9](https://github.com/qitong/EmulatRx/issues/9) 竞争风险估计器工作台与外部 Agent `SKILL.md`
- **当前 Agent 可调状态**：**排队待合入，尚未合并至 `main`（当前不可调用）**
- **状态说明**：待本 PR（[#10](https://github.com/qitong/EmulatRx/pull/10) / [#8](https://github.com/qitong/EmulatRx/issues/8)）经人工审阅合入 `main` 后，紧接在独立分支实施 [#9](https://github.com/qitong/EmulatRx/issues/9)：消费 `diagnose` 通过门禁产出的 `weighted_cohort.csv`，交付加权 Aalen–Johansen 累计发生率（CIF）与指定时点风险差/风险比（`RD(t)` / `RR_CIF(t)`）、Fine–Gray 亚分布风险回归（`sHR`）、原因特异加权 Cox（`csHR`）并置对比工作台，以及外部 Agent `SKILL.md` 与完整审计报告导出。

### 最小完全合成调用示例（离线可复现）

```bash
# 1. 标准库固定模板合成演示（零外部依赖）
python -m emulatrx_demo demo \
  --template examples/synthetic_demo_v1.json \
  --out /tmp/emulatrx-example-run

# 2. TTE 七要素方案校验 -> 合成竞争风险队列 -> 重叠加权诊断门禁（#8 新增）
python -m emulatrx_tte validate-protocol \
  --protocol examples/tte_competing_risk_v1.json \
  --out /tmp/tte-protocol-check
python -m emulatrx_tte synth \
  --protocol examples/tte_competing_risk_v1.json \
  --seed 42 --cases 600 \
  --out /tmp/tte-synth
python -m emulatrx_tte diagnose \
  --protocol examples/tte_competing_risk_v1.json \
  --cohort /tmp/tte-synth/cohort.csv \
  --weight-method overlap \
  --out /tmp/tte-diagnose
```

---

## 5. 主要 Branch 与关键逻辑分支

| Git 分支 | 当前提交 / 位置 | 关联 PR / Issue 与状态 |
| --- | --- | --- |
| `main` | [`bed656b`](https://github.com/qitong/EmulatRx/commit/bed656b02444b8c2b3989595745dab5854ef128c)（远端默认分支） | 已合并 [PR #4](https://github.com/qitong/EmulatRx/pull/4)（#1 模板契约）、[PR #5](https://github.com/qitong/EmulatRx/pull/5)（#3 合成演示）、[PR #6](https://github.com/qitong/EmulatRx/pull/6)（#2 来源收据）；`validate` 标准库 CI 26/26 全绿，未部署 |
| `qitong/emulatrx-issue8-tte-foundation` | 基于 `origin/main`（`bed656b`）的阶段一特性分支 | 关联开放 [Issue #8](https://github.com/qitong/EmulatRx/issues/8) 与 [Draft PR #10](https://github.com/qitong/EmulatRx/pull/10)（`change_class=framework`）：交付 `emulatrx_tte` 协议检查、合成竞争风险生成器、重叠加权/IPTW 门禁、E-value 修复、`tte-foundation` 离线 CI，并按跨库规范在本分支同步更新本文档；待异构复核收口与启曈本人审阅合并 |
| `qitong/emulatrx-feature-api-overview` | 基于 `origin/main`（`bed656b`）的初始概览文档分支 | 关联 [Issue #11](https://github.com/qitong/EmulatRx/issues/11) 与 [PR #12](https://github.com/qitong/EmulatRx/pull/12)，提供本文档初始基线底稿 |
| `qitong/emulatrx-contract`、`qitong/emulatrx-issue3-synthetic`、`qitong/emulatrx-issue2-live-metadata` | `5289f00`、`2663802`、`47d4dbc`（远端历史分支） | 对应 PR #4 / #5 / #6 **均已合并关闭**，无未决变更 |

- **关键业务与治理逻辑分支**：
  - **校验/门禁通过（退出码 `0`）**：`emulatrx_demo` 或 `emulatrx_tte` 输入符合契约、统计门禁达标且目标目录不存在 → 同级临时目录写齐产物与 SHA-256 `run_manifest.json` 后原子重命名发布。
  - **契约/显式偏倚规则阻断（退出码 `2`）**：结构缺失、时间零点/地标偏移矛盾、$t_0$ 后入排或协变量泄漏、参考臂未明确定义、目录已存在或公开元数据校验失败 → 立即清理临时目录、保留现场不被污染、不回退任何历史缓存，返回稳定错误码。
  - **统计诊断门禁阻断（退出码 `3`）**：`emulatrx_tte diagnose` 检出正值性违背、有效样本量比例不足、加权后残余失衡或切片后空风险集 → 发布完整流失与平衡诊断证据（`gate_status: "blocked"`），不产出 `weighted_cohort.csv`，阻止带病进入下游效应估计。
  - **待人工审批（`change_class=framework`）**：涉及核心契约、因果/统计推断逻辑或证据边界的 PR，即使离线合成 CI 与异 Agent 独立复核全过，仍保持待审状态，**必须由启曈本人审阅合并，绝不自动合并或部署**。

---

## 6. 限制与下一步

- **当前明确限制（含 `#8` 交付边界）**：
  1. **阶段二竞争风险效应估计器尚未合并**：`#8` 完成协议显式规则检查、合成竞争风险队列、倾向加权门禁与 E-value 修复，但加权 Aalen–Johansen CIF、Fine–Gray 亚分布风险回归（`sHR`）与加权原因特异 Cox（`csHR`）估计命令位于排队的 [Issue #9](https://github.com/qitong/EmulatRx/issues/9)，尚未合并；
  2. **显式规则检查的方法学边界**：`validate-protocol` 与 `diagnose` 仅能拦截输入元数据/列中显式声明的结构与时间戳违例，无法检测未测混杂、测量误差或信息性删失，亦不能证明因果有效性；
  3. **遗留模块冻结与实验性限制**：按启曈在 `Item #40` 的拍板，`v1-codebase.ipynb` 与 `environment.yml`（`langgraph==0.2.56`）保持不动；`scripts/generate_synthetic_oncology.py` 仍为无防覆写保护的实验脚本，`emulation_utils.adjust_immortal_time_bias` 简化 CCW 原型尚未修复。
- **在途与排队下一步**：
  1. 完成 [Draft PR #10](https://github.com/qitong/EmulatRx/pull/10)（[Issue #8](https://github.com/qitong/EmulatRx/issues/8)）审阅收口并由启曈本人审阅合入 `main`；
  2. `#8` 合入后紧接实施 [Issue #9](https://github.com/qitong/EmulatRx/issues/9)（竞争风险 Aalen–Johansen CIF / Fine–Gray sHR / 原因特异 Cox csHR 成套工作台 + 外部 Agent `SKILL.md` + 完整审计报告导出），并在同 PR 同步更新本文档；
  3. **后续待启曈决定的扩展点**：`#8`、`#9` 验收完成后，是否启动第二批动态策略能力（双侧宽限期分臂 IPCW 克隆删失加权 CCW、逐月序贯目标试验、离散时间 Pooled Logistic / g-formula），以及未来若需接入真实获授权队列时的独立数据合规与 SAP 签署流程。
