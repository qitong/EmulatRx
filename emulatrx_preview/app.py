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
from .tte_views import cohort_view, form, home_view, report_view, result_view, review_view
from .web_protocol import FIELD_NAMES, FormError, protocol_from_fields


NOTICE = "完全合成 TTE 方法学预览；非真实患者、非临床疗效证据、非因果实证证明。禁止上传患者数据。"
DEFAULT_PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / "examples" / "tte_preview_v1.json").read_text(encoding="utf-8"))
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
<style>:root{{--ink:#18332f;--muted:#536965;--line:#d5dfd9;--paper:#fbfaf6;--accent:#176b5e;--soft:#e9f3ee;--warn:#fff1d7}}
*{{box-sizing:border-box}}body{{font:16px/1.65 'PingFang SC','Noto Sans CJK SC','Source Han Sans SC',sans-serif;background:#eef2ef;color:var(--ink);margin:0}}
main{{max-width:1160px;margin:24px auto;padding:clamp(18px,3vw,42px);background:var(--paper);border:1px solid var(--line);box-shadow:0 8px 28px #18332f12}}
h1,h2,h3{{font-family:'Noto Serif CJK SC','Songti SC',serif;line-height:1.35}}h1{{margin:0 0 6px;font-size:1.8rem}}h2{{margin:36px 0 14px;font-size:1.55rem;border-bottom:1px solid var(--line);padding-bottom:9px}}h3{{margin:24px 0 8px;font-size:1.16rem}}
p{{max-width:78ch}}.eyebrow{{font-size:.78rem;letter-spacing:.12em;color:var(--accent);font-weight:700}}
.notice{{background:var(--warn);border-left:5px solid #a26416;padding:13px 16px;margin:20px 0;font-size:.94rem}}.error,.field-error{{background:#fff0ec;color:#912b1d;border-left:4px solid #ae4937;padding:9px 12px}}.field-error{{margin:8px 0;font-size:.9rem}}
.hero{{padding:28px 30px;background:#e8f2ec;border-left:6px solid var(--accent)}}.hero h2{{border:0;margin:0 0 12px;font-size:2rem}}.hero p{{font-size:1.08rem}}.hero-warning{{font-weight:700}}
.flow-intro{{margin:24px 0}}.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px;margin:18px 0}}.metric-grid>div{{border:1px solid var(--line);background:#fff;padding:14px 16px}}.metric-grid strong{{display:block;font:700 1.35rem/1.3 'Noto Serif CJK SC','Songti SC',serif;color:var(--accent)}}.metric-grid span{{display:block;color:var(--muted);font-size:.85rem;margin-top:4px}}
fieldset{{border:1px solid var(--line);margin:18px 0;padding:18px 20px;background:#fff}}legend{{font-weight:700;color:var(--accent);font-size:1.08rem;padding:0 8px}}fieldset p{{margin:4px 0 14px;color:var(--muted)}}.field-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}}.control{{display:block;font-weight:650}}.control span,.control small{{display:block}}.control small{{font-weight:400;font-size:.82rem;color:var(--muted);line-height:1.45;margin:4px 0 7px}}
input,select{{font:inherit;width:100%;border:1px solid #a9bcb3;border-radius:3px;padding:9px 10px;background:#fff;color:var(--ink)}}input:focus-visible,select:focus-visible,button:focus-visible,a:focus-visible,summary:focus-visible{{outline:3px solid #e1a44b;outline-offset:2px}}
.actions{{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0}}button,.primary-link{{display:inline-block;font:700 1rem/1.3 inherit;background:var(--accent);color:white;border:1px solid var(--accent);padding:11px 19px;cursor:pointer;text-decoration:none}}button:hover,.primary-link:hover{{background:#12574c;color:#fff}}button.secondary{{background:#fff;color:var(--accent)}}button.secondary:hover{{background:var(--soft)}}
pre{{white-space:pre-wrap;word-break:break-word;background:#eef2ef;padding:14px;max-height:550px;overflow:auto;font-size:.83rem}}details{{border:1px solid var(--line);background:#fff;margin:15px 0;padding:10px 14px}}summary{{cursor:pointer;font-weight:700;color:var(--accent)}}
.scroll{{overflow:auto;max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}td,th{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{background:#eef3ef}}.variable-list{{columns:2;line-height:1.8}}
figure{{display:inline-block;vertical-align:top;max-width:48%;min-width:330px;margin:8px 1% 8px 0}}figure svg{{width:100%;height:auto;background:#f8faf8}}figcaption{{font-weight:700}}.bar{{background:var(--accent);height:12px;min-width:2px}}.legend{{font-size:.8rem}}a{{color:#125e53}}code{{font-size:.83rem;word-break:break-all}}
.run-overview{{background:#f3f7f3;border-top:4px solid var(--accent);padding:4px 18px 16px}}.run-overview h2{{margin-top:18px}}.run-overview nav{{padding:10px 0;border-top:1px solid var(--line)}}.key-result{{padding:14px 18px;background:var(--soft);font-weight:650}}
@media(max-width:700px){{main{{margin:0;padding:18px}}.hero{{padding:20px}}.hero h2{{font-size:1.55rem}}.variable-list{{columns:1}}figure{{min-width:100%;max-width:100%}}}}
@media print{{body{{background:white;color:#111}}main{{border:0;box-shadow:none;max-width:none;margin:0;padding:0}}.notice{{background:white;border:1px solid #aaa}}.actions,.run-overview nav,.print-hide{{display:none!important}}details{{border:0;padding:0}}details>summary{{display:none}}details>*{{display:block}}figure{{break-inside:avoid}}.scroll{{overflow:visible}}table{{font-size:9pt}}td,th{{white-space:normal;padding:4px}}}}
</style></head><body><main>
<h1>EmulatRx 合成 TTE 工作台</h1><div class="notice"><strong>数据边界：</strong>{html.escape(NOTICE)}</div>
<details class="version-note"><summary>代码版本与分析来源</summary><p>分析依赖：已合入 main 的 PR #14，预览固定源提交 <code>{html.escape(DEPENDENCY_COMMIT)}</code>。</p></details>{body}</main></body></html>"""
    return HTMLResponse(document, status_code=status, headers={"Cache-Control": "no-store", "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'", "X-Content-Type-Options": "nosniff"})


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
    fields = parse_qs(body.decode("utf-8"), strict_parsing=True, keep_blank_values=True)
    if set(fields) != FIELD_NAMES or any(len(values) != 1 for values in fields.values()):
        raise ContractError("INVALID_FORM", "unexpected or missing form field")
    return {key: values[0] for key, values in fields.items()}


def _protocol_from_form(fields):
    protocol, settings = protocol_from_fields(fields, DEFAULT_PROTOCOL)
    return runner.validate(protocol), settings


async def home(request):
    return page(home_view())


async def new_protocol(request):
    return page(form(DEFAULT_PROTOCOL))


async def edit_protocol(request):
    try:
        fields = await _form_fields(request)
        return page(form(DEFAULT_PROTOCOL, values=fields))
    except (ContractError, ValueError, UnicodeError) as exc:
        return page(form(DEFAULT_PROTOCOL, errors={"study_title": str(exc)}), status=400)


async def validate_form(request):
    fields = {}
    try:
        fields = await _form_fields(request)
        normalized, settings = _protocol_from_form(fields)
        return page(review_view(fields, normalized, settings))
    except (ContractError, FormError, ValueError, UnicodeError) as exc:
        field = exc.field if isinstance(exc, FormError) else "study_title"
        detail = f"{exc.code}: {exc}" if isinstance(exc, ContractError) else str(exc)
        return page(form(DEFAULT_PROTOCOL, values=fields, errors={field: detail}), status=400)


async def create_run(request):
    fields = {}
    try:
        fields = await _form_fields(request)
        protocol, settings = _protocol_from_form(fields)
        run = runner.start(protocol, **settings)
        return RedirectResponse(f"/runs/{run['run_id']}", status_code=303)
    except (ContractError, FormError, ValueError, UnicodeError) as exc:
        field = exc.field if isinstance(exc, FormError) else "study_title"
        detail = f"{exc.code}: {exc}" if isinstance(exc, ContractError) else str(exc)
        return page(form(DEFAULT_PROTOCOL, values=fields, errors={field: detail}), status=400)


async def show_run(request):
    try:
        result = runner.get(request.path_params["run_id"])
    except (ContractError, OSError, ValueError):
        return page("运行不存在或已过期。", status=404)
    return page(result_view(result, runner.root), refresh=result["state"] in ("queued", "running"))


async def show_cohort(request):
    try:
        result = runner.get(request.path_params["run_id"])
        if result["state"] not in ("completed", "blocked"):
            return page("队列尚未生成。", status=409)
        raw_page = request.query_params.get("page", "1")
        if not raw_page.isascii() or not raw_page.isdecimal():
            raise ValueError("invalid page")
        page_number = int(raw_page)
        return page(cohort_view(result, runner.root, page_number))
    except (ContractError, OSError, ValueError):
        return page("队列页不存在。", status=404)


async def show_report(request):
    try:
        result = runner.get(request.path_params["run_id"])
        if result["state"] not in ("completed", "blocked"):
            return page("报告尚未生成。", status=409)
        response = page(report_view(result, runner.root))
        if request.query_params.get("download") == "1":
            response.headers["Content-Disposition"] = f'attachment; filename="emulatrx-synthetic-report-{result["run_id"]}.html"'
        return response
    except (ContractError, OSError, ValueError):
        return page("报告不存在或已过期。", status=404)


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
                          Route("/protocol/new", new_protocol),
                          Route("/protocol/edit", edit_protocol, methods=["POST"]),
                          Route("/validate", validate_form, methods=["POST"]),
                          Route("/runs", create_run, methods=["POST"]),
                          Route("/runs/{run_id}/cohort", show_cohort),
                          Route("/runs/{run_id}/report", show_report),
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
