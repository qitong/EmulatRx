# Private synthetic preview

The preview has two entry points to the same bounded runner: an authenticated browser form and a Streamable HTTP MCP endpoint at `/mcp`. The only calculation is the merged `emulatrx_demo` deterministic fixed-template generator. Every case identifier, group, and 0–100 teaching score is synthetic. The score is not a health outcome or treatment effect. This service does not provide TTE, competing-risk estimates, patient ingestion, notebook execution, LLMs, PostgreSQL, or `emulation_utils.py`.

An optional public ClinicalTrials.gov source receipt accepts only an `NCT` ID followed by eight digits. The existing `metadata` command obtains registration provenance and hashes from the official API. It never enters the synthetic calculation and does not store results data. This call requires outbound access to ClinicalTrials.gov; synthetic runs do not.

## Limits and isolation

- Browser forms accept only `case_count` (1–10,000), `seed` (0–4,294,967,295), and the fixed template version, or one NCT ID. No upload endpoint exists. MCP tools expose the same generator and receipt path.
- Browser bodies are capped at 4 KiB, MCP bodies at 4 KiB, and the underlying template at 16 KiB. One run runs at a time; synthetic runs time out at 25 seconds and public receipts at 40 seconds. A published run is capped at 2 MiB.
- Each run receives a random 128-bit ID and its own directory. Results are available for up to 24 hours and at most 100 completed runs. An interrupted run is discarded at startup. Only the service-controlled CSV path is downloadable.
- MCP requires its own Bearer token. The browser requires a separate `preview` Basic-auth password. Both secrets come from local mounted files, never from the repository. By default the host port binds to `127.0.0.1:8602`. When tailnet Serve is enabled, publish it only through Tailscale Serve HTTPS. If Serve is disabled by tailnet policy, set `EMULATRX_BIND_IP` to the host's Tailscale IPv4; the HTTP connection then travels inside Tailscale's encrypted private network. Never bind to a public interface or enable Funnel.
- The container runs as UID 10001, read-only except `/data` and a 16 MiB `/tmp`, with a single worker and bounded CPU, memory and process count. Build with an exact clean commit SHA; the Docker image embeds it in `EMULATRX_CODE_COMMIT`, and each run manifest records it. A missing or invalid revision fails closed.

## Local tests

Use Python 3.11:

```sh
python3.11 -m venv .venv-preview
.venv-preview/bin/pip install -r requirements-preview.txt httpx==0.28.1
.venv-preview/bin/python -m unittest discover -s tests -p 'test_preview_service.py' -v
```

Existing synthetic contract and metadata tests remain in the original CI. The preview workflow also builds the image on an ARM64 runner.

## Deployment and rollback

Deploy from a clean checkout of the reviewed full commit. Set `EMULATRX_CODE_COMMIT` to the tested image commit and `EMULATRX_ALLOWED_HOSTS` to the private Tailscale DNS name or IPv4 with port plus local test hosts. Provision two distinct random 32+ character secret files at `/opt/emulatrx-preview/secrets/{web_password,mcp_token}` readable only by UID 10001, and `/opt/emulatrx-preview/data` owned by UID 10001. Run `sudo docker compose -f compose.preview.yaml -p emulatrx-preview up -d --no-build` after loading the tested ARM64 image, or `up -d --build` where the image registry is reachable. Check browser and MCP authentication and an actual synthetic run before exposure. If Serve is enabled, use `tailscale serve --bg --https=443 8602` with loopback binding. If it is disabled, set `EMULATRX_BIND_IP` to `tailscale ip -4` and verify Docker listens only on that private IP, not on `0.0.0.0` or the public IP. Confirm `tailscale funnel status` has no Funnel entry. Verify GigaTIME `/healthz` before and after without touching its container.

For rollback, run `tailscale serve clear --https=443` if this preview owns that Serve entry, then `sudo docker compose -f compose.preview.yaml -p emulatrx-preview down` from its deployment directory. With direct private-IP binding, `down` removes the listener. Keep the data directory until its retention period has elapsed or remove it deliberately. A previous tested image or checkout can be restored with its full commit SHA and `up -d`; never tag a newer source with an older SHA.
