"""Authenticated private preview. No patient data or file uploads are accepted."""

import asyncio
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

from emulatrx_demo import SCENARIO, SIMULATION_LABEL, TEMPLATE_VERSION
from emulatrx_demo._common import ContractError
from .runner import Runner


NOTICE = "教学模拟，非真实患者、非临床疗效证据。无 TTE、竞争风险估计、真实数据或治疗效果分析。"
runner = Runner(os.environ.get("EMULATRX_RUNS_DIR", "/tmp/emulatrx-preview-runs"))


def _read_secret(name):
    path = os.environ.get(name)
    if not path:
        raise RuntimeError(f"{name} is required")
    value = Path(path).read_text(encoding="utf-8").strip()
    if len(value) < 32 or len(value) > 256:
        raise RuntimeError(f"{name} must contain 32..256 characters")
    return value


WEB_PASSWORD = _read_secret("EMULATRX_WEB_PASSWORD_FILE")
MCP_TOKEN = _read_secret("EMULATRX_MCP_TOKEN_FILE")
if secrets.compare_digest(WEB_PASSWORD, MCP_TOKEN):
    raise RuntimeError("web and MCP credentials must differ")

allowed_hosts = [item.strip() for item in os.environ.get("EMULATRX_ALLOWED_HOSTS", "localhost:8000").split(",") if item.strip()]
if not allowed_hosts:
    raise RuntimeError("EMULATRX_ALLOWED_HOSTS must not be empty")
mcp = FastMCP("EmulatRx synthetic preview", instructions=NOTICE,
              stateless_http=True, json_response=True, max_request_body_size=4096,
              transport_security=TransportSecuritySettings(allowed_hosts=allowed_hosts,
                  allowed_origins=[f"https://{host}" for host in allowed_hosts] + [f"http://{host}" for host in allowed_hosts]))


@mcp.tool(description="Generate only seeded synthetic teaching cases and an arbitrary teaching score. No clinical emulation.")
def run_synthetic_demo(case_count: int = 12, seed: int = 42) -> dict:
    template = {"template_version": TEMPLATE_VERSION, "scenario": SCENARIO,
                "case_count": case_count, "seed": seed, "simulation_label": SIMULATION_LABEL}
    return runner.run(template)


@mcp.tool(description="Read a recent synthetic run by its server generated run ID.")
def get_synthetic_run(run_id: str) -> dict:
    return runner.get(run_id)


@mcp.tool(description="Optionally fetch a provenance-only receipt for a public ClinicalTrials.gov NCT ID. It does not affect the synthetic demo.")
def fetch_public_source_receipt(nct_id: str) -> dict:
    return runner.receipt(nct_id)


def page(body, title="EmulatRx · 合成演示预览", status=200):
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;background:#f5f7fa;color:#17324d;margin:0}}
main{{max-width:850px;margin:40px auto;padding:28px;background:white;border-radius:12px;box-shadow:0 2px 12px #17324d16}}
h1{{margin:0 0 8px}}.notice{{background:#fff2d8;border-left:5px solid #a65a00;padding:12px;margin:20px 0}}
label{{display:block;margin:15px 0}}input{{font:inherit;padding:8px;width:180px}}button{{font:inherit;background:#174c78;color:white;border:0;border-radius:6px;padding:10px 18px;cursor:pointer}}
pre{{white-space:pre-wrap;word-break:break-word;background:#eef2f6;padding:14px;border-radius:6px;overflow:auto}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:6px;border-bottom:1px solid #ddd;text-align:left}}
a{{color:#174c78}}</style></head><body><main><h1>EmulatRx 合成演示预览</h1>
<div class="notice"><strong>模拟标识：</strong>{html.escape(NOTICE)}</div>{body}</main></body></html>"""
    return HTMLResponse(document, status_code=status, headers={"Cache-Control": "no-store", "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'", "X-Content-Type-Options": "nosniff"})


async def home(request):
    return page("""<p>选择固定模板的病例数量和种子。病例、分组和分数完全由种子生成；分数没有医学含义。</p>
<form method="post" action="/runs"><input type="hidden" name="template_version" value="1.0">
<label>合成病例数量（1–10000）<br><input name="case_count" type="number" min="1" max="10000" value="12" required></label>
<label>种子（0–4294967295）<br><input name="seed" type="number" min="0" max="4294967295" value="42" required></label>
<button type="submit">运行合成演示</button></form>
<h2>可选：公开登记来源收据</h2><p>仅保存 ClinicalTrials.gov 来源、时间和哈希；不会载入试验结果，也不会进入合成计算。</p>
<form method="post" action="/receipts"><label>NCT 编号<br><input name="nct_id" pattern="NCT[0-9]{8}" placeholder="NCT00000000" required></label><button type="submit">获取来源收据</button></form>
<p>每次运行有独立编号，最多保存 24 小时或 100 次运行。最多同时运行 1 次，合成计算上限 25 秒，收据上限 40 秒，产物上限 2 MiB。</p>""")


async def create_run(request):
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        return page("请求来源不匹配。", status=403)
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/x-www-form-urlencoded":
        return page("只接受固定表单。", status=415)
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 4096:
            return page("输入过大。", status=413)
        body.extend(chunk)
    try:
        fields = parse_qs(body.decode("utf-8"), strict_parsing=True)
        if set(fields) != {"template_version", "case_count", "seed"} or any(len(v) != 1 for v in fields.values()):
            raise ValueError("表单字段不匹配")
        if fields["template_version"][0] != TEMPLATE_VERSION:
            raise ValueError("模板版本不匹配")
        template = {"template_version": TEMPLATE_VERSION, "scenario": SCENARIO,
                    "case_count": int(fields["case_count"][0]), "seed": int(fields["seed"][0]),
                    "simulation_label": SIMULATION_LABEL}
        result = await asyncio.to_thread(runner.run, template)
    except (UnicodeError, ValueError, ContractError) as exc:
        detail = exc.code if isinstance(exc, ContractError) else "INVALID_FORM"
        return page(f"<p>输入或运行被拒绝：{html.escape(detail)}</p><p><a href='/'>返回</a></p>", status=400)
    return RedirectResponse(f"/runs/{result['run_id']}", status_code=303)


async def create_receipt(request):
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        return page("请求来源不匹配。", status=403)
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/x-www-form-urlencoded":
        return page("只接受固定表单。", status=415)
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 4096:
            return page("输入过大。", status=413)
        body.extend(chunk)
    try:
        fields = parse_qs(body.decode("utf-8"), strict_parsing=True)
        if set(fields) != {"nct_id"} or len(fields["nct_id"]) != 1:
            raise ValueError("invalid form")
        result = await asyncio.to_thread(runner.receipt, fields["nct_id"][0])
    except (UnicodeError, ValueError, ContractError) as exc:
        detail = exc.code if isinstance(exc, ContractError) else "INVALID_FORM"
        return page(f"<p>来源收据被拒绝：{html.escape(detail)}</p><p><a href='/'>返回</a></p>", status=400)
    return RedirectResponse(f"/runs/{result['run_id']}", status_code=303)


async def show_run(request):
    try:
        result = runner.get(request.path_params["run_id"])
    except (ContractError, OSError):
        return page("运行不存在或已过期。", status=404)
    if result["kind"] == "public_source_receipt":
        return page(f"<p><strong>收据编号：</strong>{result['run_id']}</p>"
                    "<p>公开登记来源收据与合成计算相互独立，不含研究结果数值。</p>"
                    f"<pre>{html.escape(json.dumps(result['receipt'], ensure_ascii=False, indent=2))}</pre>"
                    "<p><a href='/'>返回</a></p>")
    summary = result["summary"]
    manifest = result["manifest"]
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row.values()) + "</tr>" for row in result["sample_cases"])
    columns = "".join(f"<th>{html.escape(k)}</th>" for k in result["sample_cases"][0]) if result["sample_cases"] else ""
    return page(f"<p><strong>运行编号：</strong>{result['run_id']}</p>"
                f"<h2>合成汇总</h2><pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>"
                f"<h2>前十个合成病例</h2><table><tr>{columns}</tr>{rows}</table>"
                f"<p><a href='/runs/{result['run_id']}/cases.csv'>下载合成 CSV</a></p>"
                f"<h2>运行清单与可追溯来源</h2><pre>{html.escape(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre>"
                "<p><a href='/'>新建运行</a></p>")


async def cases(request):
    try:
        path = runner.cases_path(request.path_params["run_id"])
    except (ContractError, OSError):
        return page("运行不存在或已过期。", status=404)
    return FileResponse(path, media_type="text/csv", filename="synthetic-cases.csv", headers={"Cache-Control": "no-store"})


async def health(request):
    return JSONResponse({"status": "ok", "service": "emulatrx-synthetic-preview"})


mcp_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        yield


inner = Starlette(routes=[Route("/healthz", health), Route("/", home),
                          Route("/runs", create_run, methods=["POST"]),
                          Route("/receipts", create_receipt, methods=["POST"]),
                          Route("/runs/{run_id}", show_run),
                          Route("/runs/{run_id}/cases.csv", cases),
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
            challenge = b'Basic realm="EmulatRx synthetic preview"'
        if not valid:
            response = JSONResponse({"error": "authentication required"}, status_code=401,
                                    headers={"WWW-Authenticate": challenge.decode("ascii"), "Cache-Control": "no-store"})
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)


app = AuthenticationGate(inner)
