# Private synthetic TTE preview

The browser and Streamable HTTP MCP endpoint at `/mcp` use the same bounded runner around the pinned `emulatrx_tte` protocol, synthetic cohort, diagnostic gate, and competing-risks commands. The default workflow uses 2,000 seeded, fully synthetic generated rows and 100 bootstrap iterations. It applies the editable baseline eligibility criteria before diagnosis; the bundled preview example retains 954 rows with seed 42. It provides a seven-element editable protocol, normalized protocol, full generated and eligible cohort CSVs, eligibility and landmark attrition, propensity and balance diagnostics, effective sample size and weight summaries, target and competing event cumulative incidence curves, RD/RR at specified horizons, sHR and cause-specific HR together, and the complete audit report and manifests.

The analysis implementation is a **pinned PR #14 preview dependency**, not a claim that Draft PR #14 is merged or approved. The displayed dependency SHA must match the built image. Deploy a newer dependency only after its review and CI are satisfactory. PR #10 is already merged into main. The preview never invokes the legacy notebook, LLMs, PostgreSQL, or `emulation_utils.py`.

**All cases, treatment arms, covariates, events, times, and estimates are synthetic.** They are not patient data, clinical efficacy evidence, or proof of a causal effect. Diagnostic thresholds in the example protocol are editable statistical method gates, not clinical thresholds. No patient file upload endpoint exists. Do not paste patient-level content into the protocol form.

## Human flow

1. Open the private URL with the browser's `preview` username and its separate password. The home page presents each of the seven TTE elements as a labeled editable JSON section with a short explanation. The example protocol is prefilled.
2. Choose the synthetic row count (100–5,000, default 2,000), seed, and bootstrap iterations (20–300, default 100). Select **Validate protocol** to inspect the normalized protocol and correct any structural or timestamp error. The preview accepts only generated baseline covariates in eligibility; it rejects unavailable or post-event variables. Select **Run synthetic TTE** to start the bounded asynchronous job.
3. The run page refreshes while work proceeds. On completion it shows total rows and column names, a sample, the full CSV download, attrition, balance, propensity distribution, ESS and weights, two CIF curves, the joint estimate table and full audit report. The SHA-256 manifests and all permitted artifacts are downloadable.
4. If eligibility leaves no usable arm or a diagnostic gate blocks, the page shows gate failures and available diagnostics. Estimation is skipped: the weighted cohort, estimates, curves, and audit report are absent. The original generated synthetic cohort remains downloadable for inspection.

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

The deployment uses `/opt/emulatrx-preview/source` on `ymadmin`, `/opt/emulatrx-preview/secrets/{web_password,mcp_token}` readable only by UID 10001, and `/opt/emulatrx-preview/data` owned by UID 10001. Set `.env` values `EMULATRX_CODE_COMMIT` to the tested image commit, `EMULATRX_TTE_DEPENDENCY_COMMIT` to the reviewed analysis commit, `EMULATRX_ALLOWED_HOSTS` to the private Tailscale host/IP and local test hosts, and `EMULATRX_BIND_IP` to `tailscale ip -4` when Serve is unavailable. Load the tested ARM64 image, then run `sudo docker compose --env-file .env -f compose.preview.yaml -p emulatrx-preview up -d --no-build`. Check the private bind socket, browser and MCP full flows through an SSH tunnel, container health, and GigaTIME `/healthz` before and after.

To stop the preview and remove its private listener: `cd /opt/emulatrx-preview/source && sudo docker compose --env-file .env -f compose.preview.yaml -p emulatrx-preview down`. Keep the previous tested image and checkout for rollback; restore that exact checkout and its matching `.env` image commit before `up -d --no-build`. Preserve data and secret files until deliberately rotated or removed. Never tag newer code with an older commit SHA.
