# `qitong/EmulatRx` 功能与核心 API 概览（Feature & API Overview）

- **GitHub `main` 基线与当前分支**：基于 `origin/main` [`a9f98790`](https://github.com/qitong/EmulatRx/commit/a9f9879062f937e52b05fac6845571f4b1c0809d)（已合入 `#8` / [PR #10](https://github.com/qitong/EmulatRx/pull/10)、`#9` / [PR #14](https://github.com/qitong/EmulatRx/pull/14)、`#11` / [PR #12](https://github.com/qitong/EmulatRx/pull/12)、私有预览基座 [PR #13](https://github.com/qitong/EmulatRx/pull/13) 以及 `#17` P1 双侧宽限期治疗克隆–删失–加权 CCW [PR #18](https://github.com/qitong/EmulatRx/pull/18)），并在当前待合并分支 `qitong/emulatrx-dualmode-rct-workbench`（关联开放 [PR #16](https://github.com/qitong/EmulatRx/pull/16) / [Issue #15](https://github.com/qitong/EmulatRx/issues/15)，`change_class=framework`，**当前 PR 分支在途待人工审阅合并，尚未合入 `main`**）完成**面向研究者的「研究想法 / 自有数据」双入口、`ClinicalTrials.gov` 四维 PICO-T 目标 RCT 实时核验与「真项目 (`user_cohort`) / 无数据模拟预演 (`simulated_cohort`) / 合成基准 (`synthetic_only`)」双轨 TTE 工作台**升级。
- **离线自包含 HTML 版本**：同目录同步维护零外链中文离线概览页 [`docs/FEATURE_API_OVERVIEW.html`](FEATURE_API_OVERVIEW.html)。
- **定位说明**：大模型（DeepSeek / 阿里云百炼 DashScope / 标准 OpenAI 兼容接口，未挂载密钥时 100% 自动降级为本地确定性规则引擎）仅用于解析临床想法 PICO、解读严格脱敏的表级汇总画像（**铁律：绝不向外部模型发送任何一行患者记录、个体 ID 或非白名单原始分类值**）与候选目标 RCT 排序说明；底层五维 PICO-T 对齐硬上限、倾向加权门禁、竞争风险生存估计与 P1 双侧宽限期分臂 IPCW-CCW（`emulatrx_tte.ccw`，已于 `a9f98790` 合入 `main`）100% 由确定性 `emulatrx_tte` 引擎计算。

---

## 1. 仓库能力矩阵（`main` 已合入 vs 在途开放 PR #16）

| 归属与状态 | 模块 / 能力 | 核心交付物与功能边界 |
| --- | --- | --- |
| **`main` 已合入（`a9f98790`）** | `emulatrx_demo`（#1/#2/#3 → PR #4/#5/#6） | 零第三方依赖合成模板契约校验（`validate`）、固定种子虚拟病例生成（`demo`）与公开 `ClinicalTrials.gov` 12 键双 SHA-256 来源收据（`metadata`） |
| **`main` 已合入（`a9f98790`）** | `emulatrx_tte` 静态/地标竞争风险基座（#8/#9 → PR #10/#14） | 七要素方案校验（`emulatrx_tte.validate-protocol`）、合成竞争风险队列（`emulatrx_tte.synth`）、重叠加权/IPTW 门禁（`emulatrx_tte.diagnose`，退出码 `3` 阻断）、加权 Aalen–Johansen 双 CIF / Fine–Gray `sHR` / 原因特异 Cox `csHR` / `RR_CIF` E-value（`emulatrx_tte.estimate`、`emulatrx_tte.run-synthetic`）及 `skills/emulatrx-tte/SKILL.md` |
| **`main` 已合入（`a9f98790`）** | `emulatrx_tte.ccw` P1 双侧宽限期 CCW（#17 → PR #18） | `tte-ccw-v1` 协议校验（`validate-ccw`）、延迟启动合成队列与反事实真值（`synth-ccw`）、对称双侧克隆 + 分臂独立 IPCW（基线依从或 `X_tv` 逐期启动风险率乘积）+ 分位数感知截尾/截尾前极值门禁 + 患者主键聚类重拟合 Bootstrap + 中文 SVG 报告（`run-ccw-synthetic`），并将遗留缺陷函数 `emulation_utils.adjust_immortal_time_bias` 收口为 `RuntimeError` |
| **`main` 已合入（`a9f98790`）** | 私网合成预览容器基座（PR #13） | ARM64 确定性容器构建（`Dockerfile.preview`）、HTTP Basic 与 MCP Bearer 凭证隔离、非文本域合成表单与分页/打印报告基座 |
| **当前分支待合并（PR #16 / #15）** | 双入口四步工作台 + 四维 PICO-T 对齐门禁 + 双轨 TTE | 「自有脱敏队列 / 临床研究想法」双入口（含 4 套内置样表：心肾心衰、KEYNOTE-189 免疫肺癌、CAROLINA 糖尿病心血管、FLAURA EGFR 肺癌）、实时 `ClinicalTrials.gov` 检索 + 五维 PICO-T 对齐硬上限与 `422` 确认门禁、25 MB 流式上传保护、三层脱敏 LLM 辅助、会话一键物理擦除，以及 `user_cohort` 真项目 / `simulated_cohort` 无数据模拟预演全链路 Web + REST + CLI + MCP 支持 |

---

## 2. 鉴权、密钥挂载与网络/数据隔离边界

1. **Web 界面与 REST API 鉴权**：
   - 采用 HTTP Basic 鉴权（`preview:<WEB_PASSWORD>`，由只读密钥文件 `EMULATRX_WEB_PASSWORD_FILE` 注入，部署规范对应 `EMULATRX_PREVIEW_USER` / `EMULATRX_PREVIEW_PASS`）与 `Origin`/`Host` 同源校验；`ymadmin` 私网部署仅监听 Tailscale 私网地址 `100.121.92.1:8602`（本机 SSH 隧道 `127.0.0.1:18602`），零公网暴露；仅 `/healthz` 为无状态存活探针。
2. **MCP 端点独立鉴权（`/mcp`）**：
   - 采用独立 `Authorization: Bearer <MCP_TOKEN>`（由 `EMULATRX_MCP_TOKEN_FILE` / `EMULATRX_PREVIEW_MCP_TOKEN` 注入），启动时通过常量时间比对强制校验与 Web 密码绝不相同，亦绝不复用 GigaTIME 或 MedGemma 等其他服务 Token。
3. **LLM 网关只读密钥与三层外发隐私红线**：
   - `emulatrx_tte/llm_assist.py` 按优先级读取只读挂载文件 `EMULATRX_LLM_API_KEY_FILE=/run/secrets/llm_api_key`（`0400`，容器非特权用户 `UID 10001`）> `EMULATRX_LLM_API_KEY` > `OPENAI_API_KEY`；未挂载密钥或网络异常时 100% 自动回退本地确定性规则引擎，任何接口与日志零密钥回显。
   - 外发载荷严格执行三层隔离（`system` 提示词 / `cohort_outbound_context` 白名单表头汇总与匿名 `CATEGORY_LEVEL_*` 占位符 / `public_reference_context` 公开试验元数据），经 `sanitize_profile_for_llm` 与 `assert_llm_payload_privacy_safe` 双重校验，**确保个体患者行记录与非白名单单元格值 0 外发**。
4. **会话级隔离存储与一键物理清除**：
   - 上传队列按 `sessions/<session_id>/` 私网隔离落盘（`0700`），调用 `DELETE /api/cohorts/{session_id}`（或 `DELETE /api/sessions/{session_id}`、MCP `delete_cohort_session`）即使在 `session_meta.json` 损坏时仍无条件全盘扫描 `status.json` 同步物理擦除全部派生运行目录，删除后再次读取会话、运行或下载产物一律返回 `404`。

---

## 3. 全量调用入口总表（Web 路由、REST API、CLI、Python SDK 与 MCP 工具）

| 入口层 | 具体路由 / 命令 / 函数 | 鉴权要求 | 核心输入 | 主要返回产物 |
| --- | --- | --- | --- | --- |
| **Web 页面** | `GET /`、`GET /protocol/new`、`POST /validate`、`POST /runs`、`GET /runs/{run_id}`、`GET /runs/{run_id}/report[?download=1]`、`GET /runs/{run_id}/artifacts/{name:path}` | HTTP Basic | 浏览器交互、四步双轨向导或合成教学表单 | 双轨四步工作台、四维 PICO-T 徽章卡片、Step-After CIF / Love Plot / 重叠图 HTML 报告与附件下载 |
| **健康探针** | `GET /healthz` | 免鉴权（仅存活状态） | 无 | `{"status": "ok", "service": "emulatrx-dualmode-tte-workbench"}` |
| **REST · 初始化与样表** | `GET /api/workbench/bootstrap`、`GET /api/samples/{sample_id}/download` | HTTP Basic | `sample_id`（`cardiorenal_sglt2i`、`oncology_io_nsclc`、`t2dm_cv_carolina`、`nsclc_egfr_flaura`） | LLM 模式状态、4 套内置样表元数据/CSV 字节流与 4 组临床想法预设 |
| **REST · 会话上传与清除** | `POST /api/sessions/upload`、`POST /api/sessions/{id}/note`、`GET\|DELETE /api/cohorts/{id}`（同义路由 `/api/sessions/{id}`） | HTTP Basic | `multipart/form-data` CSV/TSV（≤25 MB，`64 KiB` 分块流式读取）或 JSON `sample_id` | `session_id`、脱敏 `CohortProfile`、领域线索 `interpretation`、`default_mapping`；`DELETE` 返回物理擦除文件与运行清单 |
| **REST · 想法与四维检索** | `POST /api/idea/parse`、`POST /api/rcts/search`、`POST /api/rcts/draft` | HTTP Basic | 中文临床想法或检索词、`condition` / `intervention` / `comparator` / `outcome`（自动持久化至会话并在 Step 2→3→4 传递）、`session_id`、可选确认标记 | 结构化 PICO、带 `dimension_audit` 四维胶囊与硬上限评分的候选 RCT 列表、对齐后的映射/七要素方案草稿 |
| **REST · 双轨异步运行** | `POST /api/dual-runs`、`POST /api/v1/projects/user-cohort/runs`、`POST /api/v1/projects/simulated/runs`、`GET /api/dual-runs/{run_id}` | HTTP Basic | `session_id` + `mapping`（或 `pico` + `sim_config`）+ `target_rct` + 可选 `condition`/`intervention`/`comparator`/`outcome` 覆写 + 偏离确认标记 | `202 Accepted`（`run_id`、`status_url`、`report_url`），完成后返回诊断、双 CIF、五量并列表、恢复对比表与渲染 HTML |
| **MCP 工具集** | `POST /mcp` 与 `python -m emulatrx_tte mcp-serve`（stdio）共用 14 个规范工具：`parse_research_idea`、`ingest_cohort_session`、`delete_cohort_session`、`search_target_rcts`、`audit_target_rct_alignment`、`draft_tte_protocol`、`validate_tte_protocol`、`validate_ccw_protocol`、`diagnose_tte_feasibility`、`run_static_or_landmark_tte`、`run_grace_period_ccw`、`get_tte_run_status`、`get_tte_result_summary`、`export_tte_artifact`。兼容别名：`profile_cohort_session`、`start_user_cohort_tte`、`start_simulated_rct_tte`、`start_synthetic_tte`、`get_tte_run`、`read_tte_artifact`。只读资源：`emulatrx://schemas/tte-protocol-v1`、`emulatrx://schemas/tte-ccw-v1`、`emulatrx://samples/catalog`、`emulatrx://guide/error-and-gate-codes` | `Bearer <MCP_TOKEN>`（HTTP）；stdio 为本地进程 | JSON-RPC `tools/list` / `tools/call` / `resources/list` / `resources/read` | `get_tte_result_summary` 在门禁通过时返回 `clinical_comparison_ladder`（`tier_1_unadjusted_crude` → `tier_2_conventional_ps` 的 `psm_1to1` / `simple_iptw` → `tier_3_target_trial_emulation`）与 `what_tte_controlled_beyond_psm`；`gate_status != passed` 时省略点估计。产物可用 `text` / `json` / `base64_chunk` 导出 |
| **确定性 CLI & SDK** | `python -m emulatrx_tte validate-protocol \| synth \| diagnose \| estimate \| run-synthetic \| run-simulated \| run-user-cohort \| validate-ccw \| synth-ccw \| run-ccw-synthetic \| mcp-serve`；`python -m emulatrx_demo validate \| demo \| metadata` | 本地进程（零网络/零密钥，`metadata` 除外） | 协议 JSON / 脱敏队列 CSV + 映射 JSON / `--seed` / `--out <new_dir>` | 原子发布的规范化协议、纳排瀑布、平衡诊断、CIF 曲线、`estimates.json` / `ccw_estimates.json`、`ccw_preview_report.html` 与 SHA-256 `run_manifest.json` |

---

## 4. 显式 Fail-Closed 错误码与退出码边界

| 层级 | 状态码 / 退出码 | 稳定错误码（Error Code） | 触发条件与强制阻断行为 |
| --- | --- | --- | --- |
| **CLI / 统计引擎** | 退出码 `0` | — | 协议契约与全部统计门禁通过，同级临时目录写齐全部产物与 `run_manifest.json` 后原子重命名发布 |
| **CLI / 统计引擎** | 退出码 `2` | `INVALID_JSON`、`MISSING_TIME_ZERO`、`POST_TIME_ZERO_ELIGIBILITY`、`POST_LANDMARK_COVARIATE_LEAK`、`NON_SYNTHETIC_COHORT_REJECTED`、`UNSUPPORTED_EVALUE_MEASURE`、`MISSING_USER_COHORT_INGEST_MANIFEST`、`OUTPUT_EXISTS` 等 | 协议结构/时间零点穿越/非 `SYN-*` 合成主键/目标目录已存在等契约违例：**立即终止且不发布任何目标目录** |
| **CLI / 统计引擎** | 退出码 `3` | `POSITIVITY_VIOLATION`、`ESS_RATIO_BELOW_MIN`、`SMD_EXCEEDS_MAX`、`ZERO_RISK_SET`、CCW `UNCENSORED_BELOW_MIN` / `PRE_TRUNCATION_WEIGHT_EXCEEDED` / `EXCESSIVE_WEIGHT_TRUNCATION` / `MAX_WEIGHT_EXCEEDED` / `POST_GRACE_SMD_EXCEEDS_MAX`、`UPSTREAM_GATE_NOT_PASSED` | 统计正值性、有效样本量、残余失衡或截尾前极值违背方案门禁：**仅发布诊断与流失证据包，100% 拒绝产出加权生存点估计（`estimates.json` / `ccw_estimates.json`）** |
| **HTTP / MCP 服务** | `401`（及 `403` 主机保护） | `authentication required` | 未携带凭证、密码/Token 错误、或混用 Web Basic 凭证访问 `/mcp`（及反之）时立即拦截 |
| **HTTP / MCP 服务** | `404` | `SESSION_NOT_FOUND` / `RUN_NOT_FOUND` / `ARTIFACT_NOT_FOUND` | 访问未知、过期或已通过一键物理清除擦除的会话、派生运行及产物路径时返回 `404` |
| **HTTP / MCP 服务** | `413` | `COHORT_FILE_TOO_LARGE` | 上传请求头或 `64 KiB` 流式分块读取累计超过 `25 MB` 上限时立即中断读取并清理缓冲 |
| **HTTP / MCP 服务** | `400` | `PRIVACY_VIOLATION_BLOCKED`、`ORIGIN_MISMATCH`、`INVALID_MAPPING` 等 | 发送前隐私拦截器检出个体 ID/敏感词泄漏风险、跨源请求或非法字段参数时阻断请求 |
| **HTTP / MCP 服务** | `422` | `ARM_MISMATCH_CONFIRMATION_REQUIRED` | 目标 RCT 双臂与研究干预/对照不一致（如 `NCT06642272` 同类互比、缺联合背景药、单臂/`0` 分组或匿名 `0/1` 队列未声明通用名）且未传 `confirm_arm_mismatch=true` |
| **HTTP / MCP 服务** | `422` | `PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED` | 双臂药物虽匹配但四维 PICO-T 任一维度偏离——健康受试者/药代试验（如 `NCT00842556` 上限 `0.20`）、缺失核心合并症如心衰（上限 `0.55`，含 `Exclusion Criteria` 陷阱拦截）、主要终点为血酮/HbA1c 等短期替代指标（如 `NCT03959501` 上限 `0.48`）或随访窗过短——且未传 `confirm_protocol_mismatch=true` |
| **HTTP / MCP 服务** | `429` | `BUSY` | 单容器有界并发槽位占用时拒绝超额并发计算请求 |

---

## 5. 自动化测试矩阵、执行命令与覆盖边界（White-Box Oracle & Black-Box CI Matrix）

所有 GitHub Actions 工作流在 `pull_request` 与 `push: branches: [main]` 双事件下自动触发，全仓本地一键发现命令为：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tests -p 'test_*.py' -v
```

| CI 作业（工作流文件） | 精确执行命令 | 测试性质与核心验证内容 |
| --- | --- | --- |
| **`validate`**<br>(`.github/workflows/synthetic-contract.yml`) | `python -m unittest discover -s tests -p 'test_demo_contract.py' -v`<br>`python -m unittest discover -s tests -p 'test_synthetic_demo.py' -v`<br>`python -m unittest discover -s tests -p 'test_public_metadata.py' -v` | **标准库契约与公开来源收据套件（26 项：`test_demo_contract.py` 11 + `test_synthetic_demo.py` 7 + `test_public_metadata.py` 8）**：零第三方依赖验证固定五字段模板契约、原子防覆写、逐字节确定性及 `ClinicalTrials.gov` 模拟 HTTP 12 键收据边界 |
| **`tte-foundation`**<br>(`.github/workflows/synthetic-contract.yml`) | `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_tte_*.py' -v` | **白盒数学解析基准与对抗性因果探针套件（48 项，涵盖 `test_tte_foundation.py`、`test_tte_competing_risks.py`、`test_tte_ccw.py`、`test_tte_dualmode.py`、`test_tte_clinical_ladder.py`）**：双指数竞争风险解析真值对比、协变量依赖删失联合 IPCW Fine–Gray 得分方程与 Lin–Wei 稳健方差 oracle、P1 双侧宽限期 CCW 零效应消偏/保护效应恢复/`X_tv` 时变协变量/截尾前极值阻断、三阶证据梯队（基线混杂下粗分析有害而 PSM/IPTW/Overlap TTE 恢复保护效应、宽限期真零效应下粗分析与 1:1 PSM 仍 `RD ≤ -0.10` 而双侧 CCW 回到约 0、竞争死亡下 `1-KM` 高估 CIF）、五维 PICO-T 对齐（`NCT03959501` / `NCT00842556` / `Exclusion Criteria` 陷阱 / 肿瘤 `EGFR` 突变与肾脏 `CKD` 消歧 / `user_cohort` 对称终点匹配 / 显式短缩写 `ORR`/`CRP`/`CEA`/`LDH`/`PSA` 对 FLAURA 失败关闭 / `OS`/`PFS` 仍对齐 / CAROLINA 双盲双模拟安慰剂与 FLAURA 研究者选择 `Gefitinib or Erlotinib` 正对照）、三层 LLM 脱敏拦截器及 `user_cohort`/`simulated_cohort` CLI 闭环 |
| **`service`**<br>(`.github/workflows/preview-service.yml`) | `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_preview_service.py' -v`<br>`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_dualmode_workbench.py' -v`<br>`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_mcp_agent_suite.py' -v` | **公开入口端到端黑盒服务套件（27 项：`test_preview_service.py` 11 + `test_dualmode_workbench.py` 14 + `test_mcp_agent_suite.py` 2；`python -m unittest discover -s tests` 全仓合计 101 项 Web / REST / MCP）**：通过本地回环（`127.0.0.1`）模拟 OpenAI 兼容网关与公开 `NCT*` 固定夹具（含官方 CAROLINA 双模拟臂与 FLAURA 研究者选择臂），全链路核验双入口运行与 HTML/JSON/CSV/manifest 下载 `200`、Web Basic 与 `/mcp` Bearer 鉴权隔离 `401`、`25 MB` 流式超限 `413`、Step 2 `condition`/`outcome` 覆写持久化及双臂/四维偏离 `422`（`ARM_MISMATCH_CONFIRMATION_REQUIRED` & `PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED`，含未确认的 `ORR`/`CRP`/`CEA`/`LDH`/`PSA`）、`DELETE /api/cohorts/{id}` 损坏元数据强扫物理清除后 `404`、14 工具五阶段 HTTP 闭环、stdio `mcp-serve` 工具/资源列表、上传票据单次使用与路径穿越阻断 |
| **`arm64-image`**<br>(`.github/workflows/preview-service.yml`) | `docker build --platform linux/arm64 -f Dockerfile.preview ...` + 容器内 `run-synthetic` / `run-ccw-synthetic` CLI 冒烟 + 容器实启 `127.0.0.1:18602` HTTP/MCP 黑盒冒烟 | **生产同构 ARM64 容器黑盒冒烟与镜像归档**：校验 OCI 架构与提交溯源锁，执行静态竞争风险与 P1 CCW 容器命令冒烟，并真实拉起 `uvicorn` 容器通过宿主机 `127.0.0.1:18602` 验证 `/healthz`、Basic vs Bearer 凭证隔离、`tools/list` 含 14 个规范工具、`resources/list` 含 4 个只读资源、CAROLINA 离线四维匹配与会话擦除 `404` |

- **CI 隔离原则与线上最小实机边界**：
  - CI 100% 使用确定性合成队列（`SYN-*` 或固定种子医院/肿瘤表头合成样表）、公开注册试验 `NCT*` 静态 JSON 夹具与本地 `127.0.0.1` 回环模拟大模型网关，**绝不依赖外网稳定性、绝不在 CI 存放真实大模型 API Key、绝不触碰任何真实患者个体数据（IPD）**。
  - `ymadmin` 私网部署保留最小实机冒烟核验收据：核对精确 OCI 镜像摘要、`127.0.0.1:18602/healthz`（及关联服务端口）存活、`/run/secrets/llm_api_key` 只读挂载权限（`0400`）与一键镜像回滚命令。

---

## 6. 主要 Branch 与关键 PR 状态（GitHub 快照：`2026-11-24`）

| Git 分支 | 当前提交 / 位置 | 关联 PR / Issue 与状态 |
| --- | --- | --- |
| `main` | [`a9f98790`](https://github.com/qitong/EmulatRx/commit/a9f9879062f937e52b05fac6845571f4b1c0809d)（远端默认分支） | **已合并** [PR #4](https://github.com/qitong/EmulatRx/pull/4)、[PR #5](https://github.com/qitong/EmulatRx/pull/5)、[PR #6](https://github.com/qitong/EmulatRx/pull/6)、[PR #10](https://github.com/qitong/EmulatRx/pull/10)（[#8](https://github.com/qitong/EmulatRx/issues/8)）、[PR #12](https://github.com/qitong/EmulatRx/pull/12)（[#11](https://github.com/qitong/EmulatRx/issues/11)）、[PR #13](https://github.com/qitong/EmulatRx/pull/13)、[PR #14](https://github.com/qitong/EmulatRx/pull/14)（[#9](https://github.com/qitong/EmulatRx/issues/9)）及 **[PR #18](https://github.com/qitong/EmulatRx/pull/18)（[#17](https://github.com/qitong/EmulatRx/issues/17) P1 双侧宽限期治疗克隆–删失–加权 CCW，合并提交 `a9f98790`）** |
| `qitong/emulatrx-dualmode-rct-workbench` | **当前 PR 分支**（已同步合入 `origin/main` [`a9f98790`](https://github.com/qitong/EmulatRx/commit/a9f9879062f937e52b05fac6845571f4b1c0809d)） | 关联开放 **[PR #16](https://github.com/qitong/EmulatRx/pull/16)**（[#15](https://github.com/qitong/EmulatRx/issues/15) 双入口、四维 PICO-T 目标 RCT 匹配与真/模双轨 TTE 工作台）；**当前分支在途待审阅合并，尚未合入 `main`** |
| `qitong/emulatrx-p1-grace-period-ccw` 及早期功能分支 | 远端已合并历史分支 | 对应 [PR #18](https://github.com/qitong/EmulatRx/pull/18)（`a9f98790`）及 [PR #4](https://github.com/qitong/EmulatRx/pull/4)–[#14](https://github.com/qitong/EmulatRx/pull/14) **均已合并关闭**；`v1-codebase.ipynb` 与 `environment.yml` 按拍板保持冻结不修改 |

---

## 7. 五步可视化向导与秒级预检（Issue #19）

工作台 `GET /` 与 `GET /dual` 使用同一单焦点五步向导，一次只展开当前步：

1. **选择起点与队列**：脱敏 CSV/TSV 或 CAROLINA / FLAURA 样表，或两行研究想法。
2. **锁定对标 RCT**：候选卡片上的五维 PICO-T 信号；双臂或方案偏离仍须显式确认。
3. **时间轴与角色映射**：基线窗、t₀、地标窗与随访窗，以及治疗、时间、终点、协变量槽位。
4. **秒级均衡预检**：`POST /api/v1/tte/diagnose-preflight` 返回纳排漏斗、加权前/后 `|SMD|` 与 `gate_passed`，不跑生存 bootstrap。
5. **三阶去偏结果看板**：粗分析 → 1:1 PSM / sIPTW → 完整 TTE，并展示 `what_tte_controlled_beyond_psm`。

`diagnose-preflight` 与 `POST /api/v1/projects/user-cohort/runs`、`POST /api/v1/projects/simulated/runs` 使用同一套 `verify_and_enrich_target_rct` 门禁。未确认的双臂偏离返回 `422 ARM_MISMATCH_CONFIRMATION_REQUIRED`，未确认的 PICO-T 偏离返回 `422 PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED`。`gate_passed=false` 时界面不放行最终估计。鉴权、CSP、`X-Content-Type-Options: nosniff` 与 `record_url` 的 `https://` 白名单保持不变。
