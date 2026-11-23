"""Private browser and MCP entry points for the pinned synthetic TTE pipeline."""

import base64
from contextlib import asynccontextmanager
import html
import json
import os
from pathlib import Path
import secrets
from urllib.parse import parse_qs, urlsplit

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Mount, Route

from emulatrx_tte._common import ContractError
from .tte_runner import TTERunner
from .tte_views import ELEMENTS, form, result_view


NOTICE = "完全合成 TTE 方法学预览；非真实患者、非临床疗效证据、非因果实证证明。禁止上传患者数据。"
DEFAULT_PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / "examples" / "tte_preview_v1.json").read_text(encoding="utf-8"))
SIMULATION_LABEL = DEFAULT_PROTOCOL["simulation_or_data_label"]
DEPENDENCY_COMMIT = os.environ.get("EMULATRX_TTE_DEPENDENCY_COMMIT", "development checkout")
runner = TTERunner(os.environ.get("EMULATRX_RUNS_DIR", "/tmp/emulatrx-tte-preview-runs"))


def _read_secret(name):
    path = os.environ.get(name)
    if not path:
        raise RuntimeError(f"{name} is required")
    value = Path(path).read_text(encoding="utf-8").strip()
    if len(value) < 32 or len(value) > 256 or not value.isascii():
        raise RuntimeError(f"{name} must contain 32..256 ASCII characters")
    return value


WEB_PASSWORD = _read_secret("EMULATRX_WEB_PASSWORD_FILE")
MCP_TOKEN = _read_secret("EMULATRX_MCP_TOKEN_FILE")
if secrets.compare_digest(WEB_PASSWORD, MCP_TOKEN):
    raise RuntimeError("web and MCP credentials must differ")

allowed_hosts = [item.strip() for item in os.environ.get("EMULATRX_ALLOWED_HOSTS", "localhost:8000").split(",") if item.strip()]
if not allowed_hosts:
    raise RuntimeError("EMULATRX_ALLOWED_HOSTS must not be empty")
mcp = FastMCP("EmulatRx synthetic TTE preview", instructions=NOTICE,
              stateless_http=True, json_response=True, max_request_body_size=20 * 1024,
              transport_security=TransportSecuritySettings(allowed_hosts=allowed_hosts,
                  allowed_origins=[f"https://{host}" for host in allowed_hosts] + [f"http://{host}" for host in allowed_hosts]))


@mcp.tool(description="Validate and normalize a seven-element synthetic TTE protocol. No patient data is accepted.")
def validate_tte_protocol(protocol: dict) -> dict:
    return {"status": "valid", "source_mode": "synthetic_only", "normalized_protocol": runner.validate(protocol),
            "analysis_dependency_commit": DEPENDENCY_COMMIT}


@mcp.tool(description="Start an asynchronous, fully synthetic TTE protocol → cohort → diagnostics → estimates → report run. The default is 2,000 synthetic rows.")
def start_synthetic_tte(protocol: dict, cases: int = 2000, seed: int = 42, bootstrap_iterations: int = 100) -> dict:
    return runner.start(protocol, cases=cases, seed=seed, bootstrap_iterations=bootstrap_iterations)


@mcp.tool(description="Read the status and bounded summary of a synthetic TTE run. Estimates are omitted when the diagnostic gate blocks.")
def get_tte_run(run_id: str) -> dict:
    result = runner.get(run_id)
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > 32 * 1024:
        return {"run_id": result["run_id"], "state": result["state"], "gate_status": result.get("gate_status"),
                "cohort_rows": result.get("cohort_rows"), "available_artifacts": result.get("available_artifacts", []),
                "truncated": True, "hint": "Read specific artifacts with read_tte_artifact"}
    return result


@mcp.tool(description="Read a permitted synthetic artifact in base64 chunks of at most 16 KiB; use next_offset for pagination.")
def read_tte_artifact(run_id: str, name: str, offset: int = 0, limit: int = 16384) -> dict:
    return runner.read_artifact(run_id, name, offset=offset, limit=limit)


def page(body, status=200, refresh=False):
    refresh_tag = '<meta http-equiv="refresh" content="2">' if refresh else ""
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">{refresh_tag}
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EmulatRx · 合成 TTE 预览</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;background:#f3f6f9;color:#19324a;margin:0}}
main{{max-width:1120px;margin:32px auto;padding:28px;background:white;border-radius:12px;box-shadow:0 2px 12px #19324a16}}
h1{{margin:0 0 6px}}h2{{margin-top:30px;border-bottom:1px solid #dce5eb;padding-bottom:5px}}
.notice{{background:#fff2d8;border-left:5px solid #a65a00;padding:12px;margin:20px 0}}.error{{background:#fff0ef;border-left:5px solid #b42318;padding:12px}}
fieldset{{border:1px solid #d7e0e7;border-radius:8px;margin:16px 0;padding:14px}}legend{{font-weight:700;color:#164b76}}fieldset p{{margin:0 0 8px;color:#506577}}
input,textarea{{font:inherit;box-sizing:border-box;border:1px solid #abbcca;border-radius:5px;padding:8px}}input[type=number]{{width:150px}}input[name=study_title]{{width:100%}}
textarea{{width:100%;min-height:90px;font:13px/1.4 ui-monospace,monospace}}label{{display:inline-block;margin-right:20px}}
button{{font:inherit;background:#174c78;color:white;border:0;border-radius:6px;padding:10px 18px;cursor:pointer}}button:hover{{background:#0e3657}}
pre{{white-space:pre-wrap;word-break:break-word;background:#eef2f6;padding:14px;border-radius:6px;max-height:550px;overflow:auto}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:6px 9px;border-bottom:1px solid #dce5eb;text-align:left;white-space:nowrap}}
figure{{display:inline-block;max-width:48%;min-width:330px;margin:8px 1% 8px 0}}figure svg{{width:100%;height:auto;background:#f8fafc}}figcaption{{font-weight:700}}
.bar{{background:#1a6aaa;height:12px;min-width:2px}}.legend{{font-size:12px}}a{{color:#174c78}}code{{font-size:13px}}</style></head><body><main>
<h1>EmulatRx 合成 TTE 工作台</h1><div class="notice"><strong>模拟标识：</strong>{html.escape(NOTICE)}<br>
分析依赖：PR #14 固定提交 <code>{html.escape(DEPENDENCY_COMMIT)}</code>（预览依赖，不代表已合入 main）。</div>{body}</main></body></html>"""
    return HTMLResponse(document, status_code=status, headers={"Cache-Control": "no-store", "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'", "X-Content-Type-Options": "nosniff"})


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


async def _form_fields(request):
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        raise ContractError("ORIGIN_MISMATCH", "request origin does not match host")
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/x-www-form-urlencoded":
        raise ContractError("INVALID_MEDIA_TYPE", "only the fixed form is accepted")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 64 * 1024:
            raise ContractError("INPUT_TOO_LARGE", "form exceeds 64 KiB")
        body.extend(chunk)
    fields = parse_qs(body.decode("utf-8"), strict_parsing=True)
    expected = {"study_title", "cases", "seed", "bootstrap_iterations", *[key for key, _, _ in ELEMENTS]}
    if set(fields) != expected or any(len(values) != 1 for values in fields.values()):
        raise ContractError("INVALID_FORM", "unexpected or missing form field")
    return {key: values[0] for key, values in fields.items()}


def _protocol_from_form(fields):
    protocol = {"protocol_version": "1.0", "study_title": fields["study_title"],
                "simulation_or_data_label": SIMULATION_LABEL}
    for key, _, _ in ELEMENTS:
        protocol[key] = json.loads(fields[key], object_pairs_hook=_strict_pairs,
                                   parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    protocol["covariates"] = DEFAULT_PROTOCOL["covariates"]
    return protocol


async def home(request):
    return page(form(DEFAULT_PROTOCOL))


async def validate_form(request):
    fields = {}
    try:
        fields = await _form_fields(request)
        normalized = runner.validate(_protocol_from_form(fields))
        return page(form(DEFAULT_PROTOCOL, values=fields, normalized=normalized))
    except (ContractError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        detail = f"{exc.code}: {exc}" if isinstance(exc, ContractError) else str(exc)
        return page(form(DEFAULT_PROTOCOL, values=fields, error=detail), status=400)


async def create_run(request):
    fields = {}
    try:
        fields = await _form_fields(request)
        protocol = _protocol_from_form(fields)
        run = runner.start(protocol, cases=int(fields["cases"]), seed=int(fields["seed"]),
                           bootstrap_iterations=int(fields["bootstrap_iterations"]))
        return RedirectResponse(f"/runs/{run['run_id']}", status_code=303)
    except (ContractError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        detail = f"{exc.code}: {exc}" if isinstance(exc, ContractError) else str(exc)
        return page(form(DEFAULT_PROTOCOL, values=fields, error=detail), status=400)


async def show_run(request):
    try:
        result = runner.get(request.path_params["run_id"])
    except (ContractError, OSError, ValueError):
        return page("运行不存在或已过期。", status=404)
    return page(result_view(result, runner.root), refresh=result["state"] in ("queued", "running"))


async def download_artifact(request):
    try:
        path = runner.artifact_path(request.path_params["run_id"], request.path_params["name"])
    except (ContractError, OSError, ValueError):
        return page("产物不存在或不可访问。", status=404)
    return FileResponse(path, filename=path.name, headers={"Cache-Control": "no-store"})


async def health(request):
    return JSONResponse({"status": "ok", "service": "emulatrx-synthetic-tte-preview"})


mcp_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        yield


inner = Starlette(routes=[Route("/healthz", health), Route("/", home),
                          Route("/validate", validate_form, methods=["POST"]),
                          Route("/runs", create_run, methods=["POST"]),
                          Route("/runs/{run_id}", show_run),
                          Route("/runs/{run_id}/artifacts/{name:path}", download_artifact),
                          Mount("/", app=mcp_app)], lifespan=lifespan)


class AuthenticationGate:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/healthz":
            return await self.app(scope, receive, send)
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        authorization = headers.get(b"authorization", b"")
        if scope["path"].startswith("/mcp"):
            expected = b"Bearer " + MCP_TOKEN.encode("ascii")
            valid = secrets.compare_digest(authorization, expected)
            challenge = b"Bearer"
        else:
            try:
                scheme, encoded = authorization.split(b" ", 1)
                decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
                user, password = decoded.split(":", 1)
                valid = scheme.lower() == b"basic" and user == "preview" and secrets.compare_digest(password, WEB_PASSWORD)
            except (ValueError, UnicodeError):
                valid = False
            challenge = b'Basic realm="EmulatRx synthetic TTE preview"'
        if not valid:
            response = JSONResponse({"error": "authentication required"}, status_code=401,
                                    headers={"WWW-Authenticate": challenge.decode("ascii"), "Cache-Control": "no-store"})
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)


app = AuthenticationGate(inner)
