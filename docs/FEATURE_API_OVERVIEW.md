# `qitong/EmulatRx` 功能与核心 API 概览（Feature & API Overview）

- **GitHub `main` 基线与当前分支**：基于 `origin/main` [`67bb45f3dbcffa304260cbfaa2c1b0c67523f72f`](https://github.com/qitong/EmulatRx/commit/67bb45f3dbcffa304260cbfaa2c1b0c67523f72f)（已合入 `#8` / [PR #10](https://github.com/qitong/EmulatRx/pull/10)、`#9` / [PR #14](https://github.com/qitong/EmulatRx/pull/14)、`#11` / [PR #12](https://github.com/qitong/EmulatRx/pull/12) 及私有预览基座 [PR #13](https://github.com/qitong/EmulatRx/pull/13)），并在本分支（关联 [Issue #15](https://github.com/qitong/EmulatRx/issues/15)）完成**面向研究者的「研究想法 / 自有数据」双入口、`ClinicalTrials.gov` 目标 RCT 实时匹配与「真项目 (`user_cohort`) / 无数据模拟预演 (`simulated_cohort`) / 合成基准 (`synthetic_only`)」双轨 TTE 工作台**升级。
- **定位说明**：大模型（OpenAI 兼容接口，未配置密钥时 100% 自动降级为确定性规则引擎）仅用于解析临床想法 PICO、解读严格脱敏的表级汇总画像（**铁律：绝不向外部模型发送任何一行患者记录或个体 ID**）与候选目标 RCT 排序/字段映射草拟；底层倾向加权门禁与竞争风险生存估计 100% 由确定性 `emulatrx_tte` 引擎计算。

---

## 1. 产品速览（端到端核心流程）

```text
路径一（自有数据 → 真项目 user_cohort）:
  上传脱敏 CSV/TSV（或一键载入内置心肾/肿瘤真实临床表头样表）
    ──► cohort_ingest 隐私安全画像（0 个体行外发）+ 疾病/药物线索提取
    ──► rct_matcher 实时检索 ClinicalTrials.gov API v2 选定目标 RCT（离线显式回退精选目录）
    ──► 可视化字段映射（暴露对照臂[0]/试验臂[1]、随访时间天/月/年换算、结局 1/2/0、多分类 One-Hot 与常数列保护、基线纳排瀑布）
    ──► 确定性重叠加权/IPTW 门禁 ──► 🟢《真实队列目标试验模拟（真项目）分析报告》+ 一键物理清除会话数据

路径二（研究想法 → 无数据模拟预演 simulated_cohort）:
  输入临床研究想法（内置多学科中文示例）──► 解析结构化 PICO
    ──► rct_matcher 实时检索并选定对标目标 RCT
    ──► 自动转译中文七要素方案卡片 + 直观配置样本量/混杂强度/真实效应方向/竞争死亡率
    ──► 虚拟生成对标队列 + 门禁 + 竞争风险估计 ──► 🟣《目标试验模拟（无数据模拟预演项目）可行性报告》（含模拟真值 vs 估计恢复对比表）

路径三（确定性合成基准 synthetic_only，100% 向后兼容）:
  七要素方案 JSON ──► emulatrx_tte validate-protocol ──► synth ──► diagnose ──► estimate（或 run-synthetic 一键串联）

辅助标准库套件:
  emulatrx_demo validate | demo（零依赖合成演示）与 emulatrx_demo metadata（公开 NCT 12 键来源收据）
```

---

## 2. 入口、鉴权与运行状态

| 入口类型 | 实际调用入口 | 鉴权与隔离边界 | GitHub Actions CI 覆盖 | 可核对源码与测试证据 |
| --- | --- | --- | --- | --- |
| **Web 四步双轨研究工作台**<br>想法/数据双入口 + 目标 RCT 匹配 + 真/模双轨看板 | `GET /`（四步交互工作台）；REST API：`/api/workbench/bootstrap`、`/api/samples/{id}/download`、`POST /api/sessions/upload`、`DELETE /api/sessions/{id}`、`POST /api/idea/parse`、`POST /api/rcts/search`、`POST /api/rcts/draft`、`POST /api/dual-runs`（及 v1 路由 `/api/v1/projects/user-cohort/runs`、`/api/v1/projects/simulated/runs`）、`/runs/{id}/report`；兼容教学入口 `/protocol/new` | HTTP Basic 鉴权（`preview:<WEB_PASSWORD>`）+ 同源校验 + `/healthz` 存活探针；上传数据隔离于 `sessions/<session_id>/`，支持 `DELETE /api/sessions/<session_id>` 一键物理清除；对标 RCT 双臂不一致时强制 `confirm_arm_mismatch=true`（否则返回 `422 ARM_MISMATCH_CONFIRMATION_REQUIRED`） | `preview-service.yml`（`service` 作业运行 `test_preview_service.py` 与 `test_dualmode_workbench.py`；`arm64-image` 构建 ARM64 镜像并执行双轨冒烟） | `emulatrx_preview/app.py`、`emulatrx_preview/dualmode_views.py`、`emulatrx_preview/tte_runner.py`、`tests/test_preview_service.py`、`tests/test_dualmode_workbench.py` |
| **Streamable HTTP MCP 工具集**<br>面向外部 Agent 的双轨工具 | `POST /mcp`：`search_target_rcts`、`start_simulated_rct_tte`、`profile_cohort_session`、`start_user_cohort_tte`、`delete_cohort_session`、`validate_tte_protocol`、`start_synthetic_tte`、`get_tte_run`、`read_tte_artifact` | 独立 `Authorization: Bearer <MCP_TOKEN>` 鉴权（与 Web 密码强制不同）+ `allowed_hosts` 校验 | `preview-service.yml` 全覆盖 | `emulatrx_preview/app.py`、`tests/test_preview_service.py`、`tests/test_dualmode_workbench.py` |
| **确定性科学计算 CLI、SDK 与外部 Agent Skill**<br>`emulatrx_tte` | `python -m emulatrx_tte validate-protocol \| synth \| diagnose \| estimate \| run-synthetic \| run-simulated \| run-user-cohort`；外部 Agent 规范 `skills/emulatrx-tte/SKILL.md` | 本地 CLI / SDK 零数据库依赖；`user_cohort` 强制校验 `ingest_manifest.json` 的输入 SHA-256 与派生队列哈希链；`synthetic_only` 强制校验 `SYN-*` ID | `synthetic-contract.yml` 中 `tte-foundation` 作业运行 `test_tte_*.py`（含 `test_tte_foundation.py`、`test_tte_competing_risks.py`、`test_tte_dualmode.py`） | `emulatrx_tte/`、`skills/emulatrx-tte/SKILL.md`、`docs/tte-cli-and-agent-contract-v1.md` |
| **标准库 CLI**<br>`emulatrx_demo` | `python -m emulatrx_demo validate \| demo \| metadata` | 标准库零外部科学计算依赖 | `synthetic-contract.yml` 中 `validate` 作业 **26 项**全绿 | `emulatrx_demo/`、`tests/test_demo_contract.py`、`tests/test_synthetic_demo.py`、`tests/test_public_metadata.py` |
| **遗留交互式 Notebook 与实验脚本** | `v1-codebase.ipynb`、`scripts/generate_synthetic_oncology.py`、`emulation_utils.py` | 按既有边界保持 `v1-codebase.ipynb` 与 `environment.yml` 冻结不动 | `emulation_utils.calculate_evalues` 已纳入 `tte-foundation` | `v1-codebase.ipynb`、`environment.yml`、`emulation_utils.py` |

---

## 3. 数据隐私红线与三态证据边界

1. **LLM 表头画像隐私铁律与 `ymadmin` Secret 契约（0 行个体记录外发、密钥零回显）**：
   - `emulatrx_tte/llm_assist.py` 读取密钥严格遵循优先级：`EMULATRX_LLM_API_KEY_FILE`（或默认 `/run/secrets/llm_api_key` 非空文件）> `EMULATRX_LLM_API_KEY` > `OPENAI_API_KEY`，并原生兼容 DeepSeek（`deepseek-chat` / `deepseek-reasoner`）、阿里云百炼 DashScope（`qwen-plus` 等）及标准 OpenAI 网关（自动规范化 `EMULATRX_LLM_BASE_URL` 末尾斜杠与 `/v1`、剥离 `<think>...</think>` 思考块与 ```` ```json ```` 围栏，不锁定 OpenAI 专有 `json_schema`）。
   - 通过 `sanitize_profile_for_llm` 与发送前拦截器 `assert_llm_payload_privacy_safe` 双重锁定：发往外部网关的请求体**仅允许包含列名、推断类型、缺失率、非主键短分类枚举值（≤20 项且不含 ID 模式）及数值列分位数/均值/标准差汇总，严禁包含任何一行患者记录或患者 ID（如 `Hosp-2023-0001`、`ONC-880001`）**，且任何日志/响应/审计包绝不回显密钥。
   - 当未挂载密钥文件、密钥文件为空或网络异常时，容器正常启动并 100% 自动回退本地确定性规则与双语医学词典，顶栏诚实展示 `当前未挂载模型密钥，已启用确定性规则引擎兜底（本地规则引擎模式）`。
2. **公开 `ClinicalTrials.gov` 检索与显式离线兜底**：
   - `emulatrx_tte/rct_matcher.py` 优先实时请求官方 `https://clinicaltrials.gov/api/v2/studies`（及 `/studies/{nct_id}`）并生成原始响应 SHA-256 收据哈希；仅当运行环境无外网或连接超时时，回退至内置代表性公开 RCT 目录，且**每条记录显式标记 `source: "offline_curated_catalog_fallback"`**，绝不伪装为实时联网响应。
3. **三态报告与声明锁严格区分**：
   - `user_cohort` → **🟢《真实队列目标试验模拟（真项目）分析报告》**：解除 `SYN-*` 前缀限制，强制校验 `ingest_manifest.json` 哈希谱系，输出**观察性因果推断四大核心前提**（一致性、条件可交换性/无未测混杂、正值性、条件独立右删失）及未测混杂局限声明，支持会话数据一键物理擦除；
   - `simulated_cohort` → **🟣《目标试验模拟（无数据模拟预演项目）可行性报告》**：输出预设模拟真值 vs 加权估计恢复对比表（`simulation_recovery_table`），明确锁定“无数据模拟立项预演、非临床疗效证据”；
   - `synthetic_only` → 保持既有 `run-synthetic` 与 `estimate` 全部契约、`SYN-*` 校验及逐字节确定性 100% 不变。

---

## 4. 核心模块与 Agent 能力总览

- **`emulatrx_tte.validate-protocol` / `emulatrx_tte.synth` / `emulatrx_tte.diagnose` / `emulatrx_tte.estimate` / `emulatrx_tte.run-synthetic`**：确定性七要素协议校验、合成竞争风险生成、重叠加权/IPTW 门禁（退出码 `3` Fail-Closed 阻断）、加权 Aalen–Johansen 双 CIF、Fine–Gray `sHR`、原因特异 Cox `csHR_1`/`csHR_2` 与 `RR_CIF` E-value 计算（详见 [`docs/tte-cli-and-agent-contract-v1.md`](tte-cli-and-agent-contract-v1.md) 与 [`skills/emulatrx-tte/SKILL.md`](../skills/emulatrx-tte/SKILL.md)）。
- **`emulatrx_tte.rct_matcher`**：`search_candidate_rcts`、`fetch_rct_by_nct`、`audit_study_arm_alignment` 与 `verify_and_enrich_target_rct`，支持中英双语药物/药理机制类别（SGLT2i、DPP-4i、GLP-1RA、EGFR-TKI、PD-1/PD-L1、含铂化疗等）规范化转译、共享背景治疗（如双臂共用 Metformin）差集剥离与严格 `armGroups` 结构核验（单臂或缺失 `armGroups` 绝不伪造双臂），逐臂核验 `arm_alignment_status`（`exact_dual_arm` / `class_dual_arm` 上限 `0.72` / `single_arm_or_comparator_mismatch` 硬上限 `0.45` / `unmatched_interventions` 上限 `0.25`；`single_intervention_matched` 仅限未显式指定对照药的研究想法模式），支持 `0/1` 编码队列补充 `intervention` / `comparator` 覆盖声明，并对一切非精确双臂对齐试验执行 `confirm_arm_mismatch` 显式确认门禁（未确认返回 `422 ARM_MISMATCH_CONFIRMATION_REQUIRED`）。
- **`emulatrx_tte.cohort_ingest`**：`read_tabular_cohort_bytes`（支持 UTF-8 / UTF-8-BOM / GB18030 CSV 与 TSV）、`profile_cohort_dataframe`、`recommend_column_mapping`、`ingest_user_cohort`（含暴露/时间单位/结局码 `0/1/2` 映射、无竞争事件优雅退化、多分类协变量稳健 One-Hot、常数/共线协变量自动剔除保护及逐步纳排瀑布 `ingest_attrition.csv`）。
- **`emulatrx_tte.pipeline`（`run-simulated` / `run-user-cohort`）**：串联对标虚拟预演或自有队列真项目全流程并原子发布完整审计包。
- **`emulatrx_preview`（Web 双轨工作台 + MCP 服务）**：提供零 JSON 门槛的四步中文向导界面（内置心肾代谢 1,500 例与晚期肺癌 1,200 例两套真实表头演示样表及 4 组临床想法示例）、Step-After 精确阶梯 CIF SVG、Love Plot、倾向得分重叠图、纳排瀑布图、独立可打印 HTML 报告及双轨 MCP 工具。
