# Private synthetic TTE preview

The browser and Streamable HTTP MCP endpoint at `/mcp` use the same bounded runner around the pinned `emulatrx_tte` protocol, synthetic cohort, diagnostic gate, and competing-risks commands. The default workflow uses 2,000 seeded, fully synthetic generated rows and 100 bootstrap iterations. It applies the editable baseline eligibility criteria before diagnosis; the bundled preview example retains 954 rows with seed 42. It provides a seven-element editable protocol, normalized protocol, full generated and eligible cohort CSVs, eligibility and landmark attrition, propensity and balance diagnostics, effective sample size and weight summaries, target and competing event cumulative incidence curves, RD/RR at specified horizons, sHR and cause-specific HR together, and the complete audit report and manifests.

The analysis implementation is pinned to reviewed PR #14 source commit `152af0610d4317cd18691da1173cd297566fee09`, now merged into main through merge commit `11e58c2d7ed2028f4a314c33743ff831c30bb831`. The displayed dependency SHA must match the built image. The preview never invokes the legacy notebook, LLMs, PostgreSQL, or `emulation_utils.py`.

**All cases, treatment arms, covariates, events, times, and estimates are synthetic.** They are not patient data, clinical efficacy evidence, or proof of a causal effect. Diagnostic thresholds in the example protocol are editable statistical method gates, not clinical thresholds. No patient file upload endpoint exists. Do not paste patient-level content into the protocol form.

## Human flow

1. Open the private URL with the browser's `preview` username and its separate password. The home page explains the synthetic-only boundary and loads the runnable Chinese teaching example in one click. No JSON editing is needed.
2. Complete the seven-element form with controlled synthetic baseline-variable selectors, numeric eligibility comparisons, strategy labels, follow-up horizons and teaching diagnostic gates. Time zero, two-arm observational assignment, event codes, ATO overlap weighting and the four supported effect measures are explained and fixed to the reviewed preview design. These fixed choices prevent the web form from implying support for unimplemented designs. The example has 2,000 rows, seed 42 and 100 bootstrap iterations. The form checks each editable field and displays errors beside the relevant control.
3. Select **查看方案摘要**, review the human-readable protocol, then choose **确认并运行合成 TTE** or **返回修改**. The normalized JSON is available only under technical details. The run page displays the current stage and any failure reason, refreshing automatically.
4. The result page follows study reading order: protocol, screening attrition, synthetic variable list and a 25-row paginated horizontally scrollable table, propensity/balance/ESS and weight gates, event-time **step** CIF curves, horizon RD/RR(CIF) and sHR/csHR, warnings, and a readable **研究方案与分析报告**. Generated and screened full CSVs, audit Markdown, JSON and SHA-256 manifests remain downloadable. The report is self-contained HTML with print styles and an HTML download option.
5. If eligibility leaves no usable arm or a diagnostic gate blocks, the page and report show gate failures and available diagnostics. Estimation is skipped: estimates, curves, and estimate audit report are absent. The original generated synthetic cohort remains downloadable for inspection.

## MCP tools

Send Streamable HTTP JSON-RPC requests to `/mcp` with `Authorization: Bearer <agent token>` and `Accept: application/json, text/event-stream`.

- `validate_tte_protocol(protocol)` returns the normalized seven-element contract.
- `start_synthetic_tte(protocol, cases=2000, seed=42, bootstrap_iterations=100)` returns a random run ID and initial status. It uses two-arm ATO overlap weighting with explicit competing events and RD, RR(CIF), sHR and csHR in this preview.
- `get_tte_run(run_id)` returns state, bounded diagnostics and, only after a passed gate, the joint estimate table. Large responses fall back to artifact references.
- `read_tte_artifact(run_id, name, offset=0, limit=16384)` reads only an artifact listed in the run manifest. It returns base64 bytes in chunks of at most 16 KiB with `next_offset`; use this for the full cohort, report or curves when needed.

The web and MCP credentials are distinct local files. Never place either in the repository, PR, application logs, or shared configuration.

## Resource and access boundaries

- The service protocol limit is 16 KiB, browser form limit 64 KiB, MCP request limit 20 KiB, MCP summary limit 32 KiB, and MCP artifact chunk limit 16 KiB. Output is capped at 8 MiB per run. One run executes at a time and has a 90-second deadline. Runs receive random 128-bit IDs, survive restart when completed, and are retained for at most 24 hours or 100 runs. Interrupted work is marked interrupted on restart.
- The independent ARM64 Compose project runs with one worker, UID 10001, a read-only root filesystem, a 768 MiB memory limit, one CPU, a 64 MiB tmpfs, no Linux capabilities, and a private `/data/tte` run directory. Docker publishes only the selected Tailscale IPv4 on port 8602; it must never bind to `0.0.0.0` or the public IP.
- Tailscale Serve HTTPS is preferred when the tailnet administrator enables it. If Serve is disabled, direct HTTP on the host's Tailscale IPv4 stays within Tailscale's encrypted private network. Do not enable Funnel. The browser requires Basic auth and MCP requires its own Bearer token in addition to tailnet membership.
- The image embeds a full clean build commit and the pinned TTE dependency commit. The service runs the existing `validate-protocol`, `synth`, `diagnose`, and `estimate` commands, inserting a narrow synthetic baseline eligibility screen between `synth` and `diagnose` because PR #14's `run-synthetic` command does not itself apply eligibility. The screen's output hash must equal the diagnostics manifest's input-cohort hash; the estimates manifest must point to the exact diagnostics manifest. The service manifest hashes every published artifact, including the generated and screened cohorts. The image does not mount `.git`.

## Reproducible checks and deployment

Use Python 3.11 and the pinned `requirements-preview.txt`:

```sh
python3.11 -m venv .venv-preview
.venv-preview/bin/pip install -r requirements-preview.txt httpx==0.28.1
.venv-preview/bin/python -m unittest discover -s tests -p 'test_preview_service.py' -v
```

Existing TTE tests run in the retained CI workflow. The preview CI runs service flow tests, builds a native ARM64 image, runs the 1,000-row synthetic pipeline inside that image, and exports the exact image artifact with a SHA-256 checksum. The deployed image should be that artifact, identified by its digest and full source commit. CI checks and an independent scientific review of the exact PR #14 dependency must be green before updating the server.

The deployment uses an immutable release directory under `/opt/emulatrx-preview/releases/<tested-full-sha>` on `ymadmin`, `/opt/emulatrx-preview/secrets/{web_password,mcp_token}` readable only by UID 10001, and `/opt/emulatrx-preview/data` owned by UID 10001. Set `.env` values `EMULATRX_CODE_COMMIT` to the tested image commit, `EMULATRX_TTE_DEPENDENCY_COMMIT` to the reviewed analysis commit, `EMULATRX_ALLOWED_HOSTS` to the private Tailscale host/IP and local test hosts, and `EMULATRX_BIND_IP` to `tailscale ip -4` when Serve is unavailable. Load the tested ARM64 image, then run `sudo docker compose --env-file .env -f compose.preview.yaml -p emulatrx-preview up -d --no-build`. Check the private bind socket, browser and MCP full flows through an SSH tunnel, container health, and GigaTIME `/healthz` before and after.

To stop the preview and remove its private listener, run `sudo docker compose --env-file .env -f compose.preview.yaml -p emulatrx-preview down` from the active release directory. Keep the previous tested release and image for rollback; run its matching Compose file and `.env` with `up -d --no-build`. Preserve data and secret files until deliberately rotated or removed. Never tag newer code with an older commit SHA.
