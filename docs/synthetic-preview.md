# Private Dual-Mode TTE Workbench & Preview Service

`emulatrx_preview` serves both the interactive Chinese four-step **Dual-Mode Target Trial Emulation (TTE) Workbench** at `/` and the Streamable HTTP MCP endpoint at `/mcp`, backed by deterministic `emulatrx_tte` weighting gates and competing-risks survival estimators.

## Two End-User Clinical Workflows + Legacy Synthetic Benchmark

1. **Path 1 — Owned Cohort Real Project (`project_mode = "user_cohort"`, 🟢 真项目)**:
   - Drag-and-drop upload de-identified `.csv` / `.tsv` cohorts (up to 50,000 rows / 25 MB), or click to load/download the two built-in realistic hospital/oncology header demo cohorts (`cardiorenal_sglt2i` 1,500 rows with `Hosp-2023-0001..`; `oncology_io_nsclc` 1,200 rows with `ONC-880001..`).
   - Automatic privacy-safe **Data Profiling** (row/column counts, candidate exposure/time/outcome columns, covariate summaries, missingness). Outbound LLM assistance (when `EMULATRX_LLM_API_KEY` / `OPENAI_API_KEY` is configured) sends **only column-level aggregate summaries and <=20 non-ID category labels — zero individual patient rows**. Without an API key, the workbench automatically runs in **本地规则引擎模式**.
   - Live `ClinicalTrials.gov` API v2 target RCT search & selection (with explicit `offline_curated_catalog_fallback` labeling when offline).
   - Visual column mapping (Comparator arm anchored at index 0 vs Active arm at index 1; days/months/years time conversion; primary `1` / competing `2` / censor `0` outcome mapping; categorical one-hot encoding & zero-variance protection; interactive baseline eligibility builder) → outputs **🟢《真实队列目标试验模拟（真项目）分析报告》** with four observational causal assumptions, SVG charts, and **one-click physical session wipe** (`DELETE /api/sessions/<session_id>`).
2. **Path 2 — Clinical Idea Simulated Rehearsal (`project_mode = "simulated_cohort"`, 🟣 无数据模拟预演项目)**:
   - Enter a free-text Chinese clinical research idea (4 built-in presets included) → extract PICO → select a matched `ClinicalTrials.gov` target RCT → configure sample size, confounding strength, preset treatment effect direction, and competing mortality rate → outputs **🟣《目标试验模拟（无数据模拟预演项目）可行性报告》** with a DGP truth vs estimate recovery table.
3. **Backward-Compatible Synthetic Teaching Benchmark (`project_mode = "synthetic_only"`)**:
   - Accessible via `/protocol/new` and MCP `validate_tte_protocol` / `start_synthetic_tte`, preserving the seeded 2,000-row (`SYN-000001..`) benchmark unchanged.

## Security, Resource & Deployment Boundaries

- Browser & REST routes require HTTP Basic authentication (`preview:<WEB_PASSWORD>`); `/mcp` requires a distinct `Bearer <MCP_TOKEN>`; `/healthz` provides unauthenticated container health checks.
- User-uploaded cohorts are stored strictly under `/data/tte/sessions/<session_id>/` (UID `10001`, 24-hour automatic retention cleanup, plus immediate physical deletion via `DELETE /api/sessions/<session_id>` or MCP `delete_cohort_session`).
