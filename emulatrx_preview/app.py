"""Private browser, REST API, and MCP entry points for the Dual-Mode TTE Workbench & legacy synthetic benchmark."""

from __future__ import annotations

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
from starlette.requests import Request as StarletteRequest
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route

from emulatrx_tte import (
    CLINICAL_IDEA_PRESETS,
    SAMPLE_COHORT_SPECS,
    SOURCE_MODE_SIMULATED_COHORT,
    SOURCE_MODE_USER_COHORT,
    build_simulated_protocol_and_dgp,
    fetch_rct_by_nct,
    generate_sample_cohort_csv_bytes,
    get_llm_status,
    parse_research_idea_pico,
    recommend_column_mapping,
    search_candidate_rcts,
    verify_and_enrich_target_rct,
)
from emulatrx_tte._common import ContractError, MAX_USER_COHORT_BYTES
from emulatrx_tte.benchmarking import benchmarking_disabled_payload, interpretive_guardrail, is_benchmarking_enabled
from emulatrx_tte.llm_assist import extract_untrusted_Note_forbidden_tokens
from .dualmode_views import render_dualmode_run_view, workbench_home_view
from .mcp_tools import benchmark_from_payload, register_tools
from .tte_runner import TTERunner
from .tte_views import cohort_view, form, report_view, result_view, review_view
from .web_protocol import FIELD_NAMES, FormError, protocol_from_fields


UPLOAD_READ_CHUNK_BYTES = 65536
MAX_MULTIPART_OVERHEAD_BYTES = 128 * 1024


class _UploadTooLargeError(Exception):
    """Raised when an incoming multipart cohort upload exceeds the 25 MB limit."""


def _public_interpretation(interp: dict) -> dict:
    return {k: v for k, v in interp.items() if k != "sanitized_llm_payload"}


def _contract_error_json(exc: ContractError) -> JSONResponse:
    if exc.code in ("ARM_MISMATCH_CONFIRMATION_REQUIRED", "PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED"):
        status_http = 422
    elif exc.code == "COHORT_FILE_TOO_LARGE":
        status_http = 413
    elif exc.code == "BUSY":
        status_http = 429
    else:
        status_http = 400
    return JSONResponse(
        {"error": f"{exc.code}: {exc}", "code": exc.code, "error_code": exc.code},
        status_code=status_http,
        headers={"Cache-Control": "no-store"},
    )


NOTICE = (
    "双轨数据与证据边界：🟢 自有队列真项目（user_cohort）按会话私网隔离并支持一键物理清除，发往外部大模型的请求仅含表级汇总统计、严禁外发任何个体患者行记录，结果受观察性因果识别四大前提约束；"
    "🟣 无数据模拟预演（simulated_cohort）与合成基准（synthetic_only）均为非真实患者数据、非临床疗效证据。"
)
DEFAULT_PROTOCOL = json.loads(
    (Path(__file__).resolve().parents[1] / "examples" / "tte_preview_v1.json").read_text(encoding="utf-8")
)
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

allowed_hosts = [
    item.strip()
    for item in os.environ.get("EMULATRX_ALLOWED_HOSTS", "localhost:8000").split(",")
    if item.strip()
]
if not allowed_hosts:
    raise RuntimeError("EMULATRX_ALLOWED_HOSTS must not be empty")

mcp = FastMCP(
    "EmulatRx Dual-Mode TTE Workbench",
    instructions=NOTICE,
    stateless_http=True,
    json_response=True,
    max_request_body_size=2 * 1024 * 1024,
    transport_security=TransportSecuritySettings(
        allowed_hosts=allowed_hosts,
        allowed_origins=[f"https://{host}" for host in allowed_hosts]
        + [f"http://{host}" for host in allowed_hosts],
    ),
)


# MCP tools and resources are registered once and shared with `python -m emulatrx_tte mcp-serve`.
_allow_ingest = [
    Path(item).expanduser().resolve()
    for item in os.environ.get("EMULATRX_MCP_ALLOWED_INGEST_DIR", "").split(os.pathsep)
    if item.strip()
]
register_tools(mcp, runner, transport="http", allow_ingest_dirs=_allow_ingest)


# ----------------------------------------------------------------------
# HTML Layout & Origin Helpers
# ----------------------------------------------------------------------


def page(body, status=200, refresh=False, title="EmulatRx · 目标试验模拟（TTE）双轨工作台"):
    refresh_tag = '<meta http-equiv="refresh" content="2">' if refresh else ""
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">{refresh_tag}
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>:root{{--ink:#18332f;--muted:#536965;--line:#d5dfd9;--paper:#fbfaf6;--accent:#176b5e;--soft:#e9f3ee;--warn:#fff1d7}}
*{{box-sizing:border-box}}body{{font:16px/1.65 'PingFang SC','Noto Sans CJK SC','Source Han Sans SC',sans-serif;background:#eef2ef;color:var(--ink);margin:0}}
main{{max-width:1180px;margin:22px auto;padding:clamp(18px,3vw,40px);background:var(--paper);border:1px solid var(--line);box-shadow:0 8px 28px #18332f12}}
h1,h2,h3{{font-family:'Noto Serif CJK SC','Songti SC',serif;line-height:1.35}}h1{{margin:0 0 6px;font-size:1.82rem}}h2{{margin:32px 0 14px;font-size:1.48rem;border-bottom:1px solid var(--line);padding-bottom:8px}}h3{{margin:18px 0 8px;font-size:1.14rem}}
p{{max-width:86ch}}.eyebrow{{font-size:.78rem;letter-spacing:.1em;color:var(--accent);font-weight:700}}
.notice{{background:var(--warn);border-left:5px solid #a26416;padding:12px 16px;margin:16px 0;font-size:.92rem}}.error,.field-error{{background:#fff0ec;color:#912b1d;border-left:4px solid #ae4937;padding:10px 14px}}.field-error{{margin:8px 0;font-size:.9rem}}
.hero{{padding:28px 30px;background:#e8f2ec;border-left:6px solid var(--accent)}}.hero h2{{border:0;margin:0 0 12px;font-size:2rem}}.hero p{{font-size:1.08rem}}.hero-warning{{font-weight:700}}
.flow-intro{{margin:24px 0}}.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px;margin:16px 0}}.metric-grid>div{{border:1px solid var(--line);background:#fff;padding:13px 15px;border-radius:5px}}.metric-grid strong{{display:block;font:700 1.28rem/1.3 'Noto Serif CJK SC','Songti SC',serif;color:var(--accent)}}.metric-grid span{{display:block;color:var(--muted);font-size:.84rem;margin-top:4px}}
fieldset{{border:1px solid var(--line);border-radius:6px;margin:16px 0;padding:18px 20px;background:#fff}}legend{{font-weight:700;color:var(--accent);font-size:1.06rem;padding:0 8px}}fieldset p{{margin:4px 0 12px;color:var(--muted)}}.field-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:14px}}.control{{display:block;font-weight:650}}.control span,.control small{{display:block}}.control small{{font-weight:400;font-size:.82rem;color:var(--muted);line-height:1.45;margin:4px 0 7px}}
input,select{{font:inherit;width:100%;border:1px solid #a9bcb3;border-radius:4px;padding:8px 10px;background:#fff;color:var(--ink)}}input:focus-visible,select:focus-visible,button:focus-visible,a:focus-visible,summary:focus-visible{{outline:3px solid #e1a44b;outline-offset:2px}}
.actions{{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}}button,.primary-link{{display:inline-block;font:700 .96rem/1.3 inherit;background:var(--accent);color:white;border:1px solid var(--accent);border-radius:4px;padding:10px 18px;cursor:pointer;text-decoration:none}}button:hover,.primary-link:hover{{background:#12574c;color:#fff}}button.secondary{{background:#fff;color:var(--accent)}}button.secondary:hover{{background:var(--soft)}}
pre{{white-space:pre-wrap;word-break:break-word;background:#eef2ef;padding:14px;max-height:550px;overflow:auto;font-size:.82rem}}details{{border:1px solid var(--line);border-radius:5px;background:#fff;margin:14px 0;padding:10px 14px}}summary{{cursor:pointer;font-weight:700;color:var(--accent)}}
.scroll{{overflow:auto;max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:.88rem}}td,th{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{background:#eef3ef}}.variable-list{{columns:2;line-height:1.8}}
figure{{display:inline-block;vertical-align:top;max-width:48.5%;min-width:320px;margin:8px 1% 8px 0}}figure svg{{width:100%;height:auto;background:#f8faf8}}figcaption{{font-weight:700;margin-bottom:6px}}.bar{{background:var(--accent);height:12px;min-width:2px}}.legend{{font-size:.8rem;margin-top:6px}}a{{color:#125e53}}code{{font-size:.82rem;word-break:break-all}}
.run-overview{{background:#f3f7f3;border-top:4px solid var(--accent);padding:4px 18px 16px}}.run-overview h2{{margin-top:18px}}.run-overview nav{{padding:10px 0;border-top:1px solid var(--line)}}.key-result{{padding:14px 18px;background:var(--soft);font-weight:650}}
@media(max-width:700px){{main{{margin:0;padding:16px}}.hero{{padding:20px}}.hero h2{{font-size:1.55rem}}.variable-list{{columns:1}}figure{{min-width:100%;max-width:100%}}}}
@media print{{body{{background:white;color:#111}}main{{border:0;box-shadow:none;max-width:none;margin:0;padding:0}}.notice{{background:white;border:1px solid #aaa}}.actions,.run-overview nav,.print-hide,.wb-topbar,.stepper{{display:none!important}}details{{border:0;padding:0}}details>summary{{display:none}}details>*{{display:block}}figure{{break-inside:avoid}}.scroll{{overflow:visible}}table{{font-size:9pt}}td,th{{white-space:normal;padding:4px}}}}
</style></head><body><main>
<h1>EmulatRx · 目标试验模拟（TTE）双轨研究工作台</h1><div class="notice"><strong>方法学与数据隐私边界：</strong>{html.escape(NOTICE)}</div>
<details class="version-note print-hide"><summary>代码版本与确定性统计引擎来源</summary><p>底层加权门禁与竞争风险生存估计 100% 由确定性 <code>emulatrx_tte</code> 计算；源提交 <code>{html.escape(DEPENDENCY_COMMIT)}</code>。</p></details>{body}</main></body></html>"""
    return HTMLResponse(
        document,
        status_code=status,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _check_same_origin(request):
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        raise ContractError("ORIGIN_MISMATCH", "request origin does not match host")


async def _read_json_body(request, max_bytes=256 * 1024):
    _check_same_origin(request)
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_bytes:
            raise ContractError("INPUT_TOO_LARGE", "JSON request exceeds size limit")
        body.extend(chunk)
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("INVALID_JSON", f"invalid JSON payload: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractError("INVALID_TYPE", "JSON body must be an object")
    return data


async def _form_fields(request):
    _check_same_origin(request)
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


# ----------------------------------------------------------------------
# Dual-Mode Workbench REST API Endpoints
# ----------------------------------------------------------------------


async def api_workbench_bootstrap(request):
    return JSONResponse(
        {
            "llm_status": get_llm_status(),
            "samples": list(SAMPLE_COHORT_SPECS.values()),
            "ideas": list(CLINICAL_IDEA_PRESETS),
        },
        headers={"Cache-Control": "no-store"},
    )


async def api_download_sample(request):
    sample_id = request.path_params["sample_id"]
    try:
        filename, raw = generate_sample_cohort_csv_bytes(sample_id)
    except ValueError:
        return JSONResponse({"error": "未知的样表编号"}, status_code=404)
    return Response(
        raw,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


async def api_upload_with_ticket(request):
    """Single-use ticket stream. Authentication is the unguessable ticket, not Web Basic."""
    try:
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > MAX_USER_COHORT_BYTES:
            raise ContractError("COHORT_FILE_TOO_LARGE", "上传数据文件超过 25 MB 限制。")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_USER_COHORT_BYTES:
                raise ContractError("COHORT_FILE_TOO_LARGE", "上传数据文件超过 25 MB 限制。")
            body.extend(chunk)
        saved = runner.consume_upload_ticket(request.path_params["ticket_id"], bytes(body))
        from .mcp_tools import _session_view

        return JSONResponse(_session_view(saved), headers={"Cache-Control": "no-store"})
    except ContractError as exc:
        return _contract_error_json(exc)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400, headers={"Cache-Control": "no-store"})


async def api_upload_or_load_session(request):
    try:
        _check_same_origin(request)
        cl_header = request.headers.get("content-length")
        if cl_header is not None:
            try:
                cl_int = int(cl_header)
            except ValueError as exc:
                raise ContractError("INVALID_CONTENT_LENGTH", "请求头 Content-Length 格式无效。") from exc
            if cl_int > MAX_USER_COHORT_BYTES + MAX_MULTIPART_OVERHEAD_BYTES:
                raise _UploadTooLargeError("上传请求超过 25 MB 上限（Content-Length 超限），已拒绝接收。")

        ctype = request.headers.get("content-type", "").lower()
        if ctype.startswith("application/json"):
            from emulatrx_tte.sample_cohorts import SAMPLE_ID_ALIASES

            payload = await _read_json_body(request)
            raw_sample_id = str(payload.get("sample_id", "")).strip()
            sample_id = SAMPLE_ID_ALIASES.get(raw_sample_id, raw_sample_id)
            if sample_id not in SAMPLE_COHORT_SPECS:
                raise ContractError("INVALID_SAMPLE_ID", f"未知的内置样表编号：{raw_sample_id}")
            filename, raw = generate_sample_cohort_csv_bytes(sample_id)
            saved = runner.save_session_cohort(
                raw,
                filename,
                session_id=payload.get("session_id"),
                user_note=str(payload.get("user_note", "")),
                sample_id=sample_id,
            )
        elif ctype.startswith("multipart/form-data"):
            seen_wire_bytes = 0

            async def bounded_receive():
                nonlocal seen_wire_bytes
                message = await request.receive()
                if message.get("type") == "http.request":
                    chunk_bytes = message.get("body", b"") or b""
                    seen_wire_bytes += len(chunk_bytes)
                    if seen_wire_bytes > MAX_USER_COHORT_BYTES + MAX_MULTIPART_OVERHEAD_BYTES:
                        raise _UploadTooLargeError("流式上传数据超过 25 MB 上限，已立即中止接收。")
                return message

            bounded_req = StarletteRequest(request.scope, receive=bounded_receive)
            form_data = await bounded_req.form(max_files=1, max_fields=6, max_part_size=64 * 1024)
            try:
                upload = form_data.get("file")
                if upload is None or not hasattr(upload, "read"):
                    raise ContractError("MISSING_FILE", "请选择要上传的 CSV / TSV 队列文件。")
                buf = bytearray()
                try:
                    while True:
                        chunk = await upload.read(UPLOAD_READ_CHUNK_BYTES)
                        if not chunk:
                            break
                        if len(buf) + len(chunk) > MAX_USER_COHORT_BYTES:
                            raise _UploadTooLargeError("上传队列文件超过 25 MB 上限，已中止读取并清理临时缓冲。")
                        buf.extend(chunk)
                finally:
                    await upload.close()
                raw = bytes(buf)
                filename = getattr(upload, "filename", None) or "uploaded_cohort.csv"
                user_note = str(form_data.get("user_note") or "")
                session_id = str(form_data.get("session_id") or "") or None
            finally:
                await form_data.close()
            saved = runner.save_session_cohort(raw, filename, session_id=session_id, user_note=user_note)
        else:
            raise ContractError("UNSUPPORTED_CONTENT_TYPE", "请通过文件上传表单或样表载入接口提交。")

        return JSONResponse(
            {
                "session_id": saved["session_id"],
                "profile": saved["profile"],
                "interpretation": _public_interpretation(saved["interpretation"]),
                "default_mapping": saved["default_mapping"],
                "llm_status": get_llm_status(),
            },
            headers={"Cache-Control": "no-store"},
        )
    except _UploadTooLargeError as exc:
        return JSONResponse(
            {"error": f"COHORT_FILE_TOO_LARGE: {exc}", "code": "COHORT_FILE_TOO_LARGE"},
            status_code=413,
            headers={"Cache-Control": "no-store"},
        )
    except ContractError as exc:
        status_http = 413 if exc.code == "COHORT_FILE_TOO_LARGE" else 400
        return JSONResponse(
            {"error": f"{exc.code}: {exc}", "code": exc.code},
            status_code=status_http,
            headers={"Cache-Control": "no-store"},
        )


async def api_update_session_note(request):
    try:
        payload = await _read_json_body(request, max_bytes=16 * 1024)
        res = runner.update_session_note(request.path_params["session_id"], str(payload.get("user_note", "")))
        res["interpretation"] = _public_interpretation(res["interpretation"])
        return JSONResponse(res, headers={"Cache-Control": "no-store"})
    except ContractError as exc:
        return JSONResponse({"error": f"{exc.code}: {exc}", "code": exc.code}, status_code=400)


async def api_get_session(request):
    try:
        sess = runner.get_session(request.path_params["session_id"])
    except ContractError as exc:
        status_http = 404 if exc.code == "SESSION_NOT_FOUND" else 400
        return JSONResponse({"error": f"{exc.code}: {exc}", "code": exc.code}, status_code=status_http, headers={"Cache-Control": "no-store"})
    return JSONResponse(
        {
            "session_id": sess["session_id"],
            "profile": sess["profile"],
            "interpretation": _public_interpretation(sess["interpretation"]),
            "default_mapping": sess["default_mapping"],
        },
        headers={"Cache-Control": "no-store"},
    )


async def api_delete_session(request):
    try:
        _check_same_origin(request)
        res = runner.delete_session(request.path_params["session_id"])
        return JSONResponse(res, status_code=200, headers={"Cache-Control": "no-store"})
    except ContractError as exc:
        return JSONResponse(
            {"error": f"{exc.code}: {exc}", "code": exc.code, "session_deleted": False},
            status_code=400,
            headers={"Cache-Control": "no-store"},
        )
    except (OSError, RuntimeError) as exc:
        return JSONResponse(
            {"error": f"SESSION_WIPE_FAILED: {exc}", "code": "SESSION_WIPE_FAILED", "session_deleted": False},
            status_code=500,
            headers={"Cache-Control": "no-store"},
        )


async def api_parse_idea(request):
    try:
        payload = await _read_json_body(request, max_bytes=32 * 1024)
        pico = parse_research_idea_pico(
            str(payload.get("idea_text", "")),
            explicit_comparator=str(payload.get("comparator", "")),
        )
        return JSONResponse({"pico": pico, "llm_status": get_llm_status()}, headers={"Cache-Control": "no-store"})
    except ContractError as exc:
        return JSONResponse({"error": f"{exc.code}: {exc}", "code": exc.code}, status_code=400)


async def api_search_rcts(request):
    try:
        payload = await _read_json_body(request, max_bytes=64 * 1024)
        session_id = str(payload.get("session_id", "")).strip()
        project_mode = str(payload.get("project_mode", "")).strip() or (
            SOURCE_MODE_USER_COHORT if session_id else ""
        )
        cond_in = str(payload.get("condition", "")).strip()
        intr_in = str(payload.get("intervention", "")).strip()
        comp_in = str(payload.get("comparator", "")).strip()
        out_in = str(payload.get("outcome", "")).strip()
        fu_in = float(payload["target_followup_days"]) if isinstance(payload.get("target_followup_days"), (int, float)) else None
        if session_id and (any((cond_in, intr_in, comp_in, out_in)) or fu_in is not None):
            runner.save_session_pico_overrides(
                session_id,
                condition=cond_in,
                intervention=intr_in,
                comparator=comp_in,
                outcome=out_in,
                target_followup_days=fu_in,
            )

        forbidden_list = list(runner.get_session_forbidden_values(session_id)) if session_id else []
        ctx_summary = payload.get("context_summary") if isinstance(payload.get("context_summary"), dict) else None
        sess_for_search = None
        if session_id:
            try:
                sess_for_search = runner.get_session(session_id)
                if ctx_summary is None:
                    ctx_summary = sess_for_search["interpretation"]
                else:
                    merged_ctx = dict(sess_for_search["interpretation"])
                    merged_ctx.update({k: v for k, v in ctx_summary.items() if v not in (None, "")})
                    ctx_summary = merged_ctx
            except ContractError:
                sess_for_search = None
        if session_id or project_mode == SOURCE_MODE_USER_COHORT:
            raw_probe = " ".join(
                [
                    str(payload.get("query", "")),
                    cond_in,
                    intr_in,
                    comp_in,
                    out_in,
                    json.dumps(ctx_summary or {}, ensure_ascii=False),
                ]
            )
            for tok in extract_untrusted_Note_forbidden_tokens(raw_probe):
                if tok not in forbidden_list:
                    forbidden_list.append(tok)
        res = search_candidate_rcts(
            str(payload.get("query", "")),
            condition=cond_in,
            intervention=intr_in,
            comparator=comp_in,
            outcome=out_in,
            target_followup_days=fu_in,
            project_mode=project_mode,
            context_summary=ctx_summary,
            max_results=int(payload.get("max_results", 6)),
            forbidden_values=tuple(forbidden_list),
        )
        return JSONResponse(res, headers={"Cache-Control": "no-store"})
    except ContractError as exc:
        return _contract_error_json(exc)


async def api_draft_mapping_or_protocol(request):
    try:
        payload = await _read_json_body(request, max_bytes=128 * 1024)
        pmode = str(payload.get("project_mode", ""))
        target_rct = payload.get("target_rct") if isinstance(payload.get("target_rct"), dict) else None
        if not target_rct and str(payload.get("nct_id", "")).strip():
            target_rct = fetch_rct_by_nct(str(payload["nct_id"]).strip())
        confirm_arm = bool(payload.get("confirm_arm_mismatch") is True)
        confirm_proto = bool(payload.get("confirm_protocol_mismatch") is True)
        cond_ovr = str(payload.get("condition", "")).strip()
        intr_ovr = str(payload.get("intervention", "")).strip()
        comp_ovr = str(payload.get("comparator", "")).strip()
        out_ovr = str(payload.get("outcome", "")).strip()
        fu_ovr = float(payload["target_followup_days"]) if isinstance(payload.get("target_followup_days"), (int, float)) else None

        if pmode == SOURCE_MODE_USER_COHORT:
            session_id = str(payload.get("session_id", ""))
            if any((cond_ovr, intr_ovr, comp_ovr, out_ovr)) or fu_ovr is not None:
                runner.save_session_pico_overrides(
                    session_id,
                    condition=cond_ovr,
                    intervention=intr_ovr,
                    comparator=comp_ovr,
                    outcome=out_ovr,
                    target_followup_days=fu_ovr,
                )
            sess = runner.get_session(session_id)
            mapping = recommend_column_mapping(sess["profile"], target_rct=target_rct)
            if target_rct:
                pctx = runner.resolve_user_cohort_pico_context(
                    sess,
                    mapping,
                    condition_override=cond_ovr,
                    intervention_override=intr_ovr,
                    comparator_override=comp_ovr,
                    outcome_override=out_ovr,
                    target_followup_days_override=fu_ovr,
                )
                target_rct = verify_and_enrich_target_rct(
                    target_rct,
                    intervention=pctx["intervention"],
                    comparator=pctx["comparator"],
                    target_condition=pctx["target_condition"],
                    target_outcome=pctx["target_outcome"],
                    target_followup_days=pctx["target_followup_days"],
                    project_mode=SOURCE_MODE_USER_COHORT,
                    confirm_arm_mismatch=confirm_arm,
                    confirm_protocol_mismatch=confirm_proto,
                )
            return JSONResponse(
                {"project_mode": pmode, "mapping": mapping, "target_rct": target_rct, "pico_context": pctx if target_rct else None},
                headers={"Cache-Control": "no-store"},
            )
        if pmode == SOURCE_MODE_SIMULATED_COHORT:
            pico = payload.get("pico") if isinstance(payload.get("pico"), dict) else None
            sim_cfg = dict(payload.get("sim_config")) if isinstance(payload.get("sim_config"), dict) else {}
            if intr_ovr:
                sim_cfg.setdefault("intervention", intr_ovr)
            if comp_ovr:
                sim_cfg.setdefault("comparator", comp_ovr)
            proto_in = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else None
            eff_pico, sim_intr, sim_comp = runner.resolve_simulated_effective_pico(
                pico=pico,
                idea_text=str(payload.get("idea_text", "")),
                condition=cond_ovr,
                intervention=intr_ovr,
                comparator=comp_ovr,
                outcome=out_ovr,
                sim_config=sim_cfg,
                protocol=proto_in,
                target_rct=target_rct,
            )
            if target_rct:
                target_rct = verify_and_enrich_target_rct(
                    target_rct,
                    intervention=sim_intr,
                    comparator=sim_comp,
                    target_condition=str((eff_pico or {}).get("condition_query_en") or "").strip(),
                    target_outcome=str((eff_pico or {}).get("outcome_query_en") or (eff_pico or {}).get("primary_outcome") or "").strip(),
                    target_followup_days=fu_ovr,
                    project_mode=SOURCE_MODE_SIMULATED_COHORT,
                    confirm_arm_mismatch=confirm_arm,
                    confirm_protocol_mismatch=confirm_proto,
                )
            proto, dgp = build_simulated_protocol_and_dgp(
                target_rct=target_rct,
                pico=eff_pico,
                sim_config=sim_cfg,
            )
            return JSONResponse(
                {"project_mode": pmode, "protocol": proto, "dgp_preview": dgp, "target_rct": target_rct},
                headers={"Cache-Control": "no-store"},
            )
        raise ContractError("UNSUPPORTED_SOURCE_MODE", f"Unsupported project_mode: {pmode}")
    except ContractError as exc:
        return _contract_error_json(exc)


async def _launch_dual_run_payload(payload: dict, forced_mode: str | None = None) -> JSONResponse:
    pmode = forced_mode or str(payload.get("project_mode", ""))
    target_rct = payload.get("target_rct") if isinstance(payload.get("target_rct"), dict) else None
    if not target_rct and str(payload.get("nct_id", "")).strip():
        target_rct = fetch_rct_by_nct(str(payload["nct_id"]).strip())
    confirm_arm = bool(payload.get("confirm_arm_mismatch") is True)
    confirm_proto = bool(payload.get("confirm_protocol_mismatch") is True)
    cond_ovr = str(payload.get("condition", "")).strip()
    intr_ovr = str(payload.get("intervention", "")).strip()
    comp_ovr = str(payload.get("comparator", "")).strip()
    out_ovr = str(payload.get("outcome", "")).strip()
    fu_ovr = float(payload["target_followup_days"]) if isinstance(payload.get("target_followup_days"), (int, float)) else None

    if pmode == SOURCE_MODE_USER_COHORT:
        session_id = str(payload.get("session_id", ""))
        if any((cond_ovr, intr_ovr, comp_ovr, out_ovr)) or fu_ovr is not None:
            runner.save_session_pico_overrides(
                session_id,
                condition=cond_ovr,
                intervention=intr_ovr,
                comparator=comp_ovr,
                outcome=out_ovr,
                target_followup_days=fu_ovr,
            )
        sess = runner.get_session(session_id)
        mapping = payload.get("mapping")
        if mapping is None:
            mapping = recommend_column_mapping(sess["profile"], target_rct=target_rct)
        if not isinstance(mapping, dict):
            raise ContractError("INVALID_MAPPING", "mapping must be an object")
        status = runner.start_user_cohort(
            session_id,
            mapping,
            target_rct=target_rct,
            condition_override=cond_ovr,
            intervention_override=intr_ovr,
            comparator_override=comp_ovr,
            outcome_override=out_ovr,
            target_followup_days_override=fu_ovr,
            confirm_arm_mismatch=confirm_arm,
            confirm_protocol_mismatch=confirm_proto,
        )
    elif pmode == SOURCE_MODE_SIMULATED_COHORT:
        pico = payload.get("pico") if isinstance(payload.get("pico"), dict) else None
        idea_txt = str(payload.get("idea_text", ""))
        proto_in = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else None
        sim_config = dict(payload.get("sim_config")) if isinstance(payload.get("sim_config"), dict) else {}
        if intr_ovr:
            sim_config["intervention"] = intr_ovr
        if comp_ovr:
            sim_config["comparator"] = comp_ovr
        status = runner.start_simulated_cohort(
            target_rct=target_rct,
            pico=pico,
            idea_text=idea_txt,
            condition=cond_ovr,
            intervention=intr_ovr,
            comparator=comp_ovr,
            outcome=out_ovr,
            protocol=proto_in,
            sim_config=sim_config,
            confirm_arm_mismatch=confirm_arm,
            confirm_protocol_mismatch=confirm_proto,
        )
    else:
        raise ContractError("UNSUPPORTED_SOURCE_MODE", f"Unsupported project_mode: {pmode}")
    return JSONResponse(
        {
            **status,
            "status_url": f"/api/dual-runs/{status['run_id']}",
            "view_url": f"/runs/{status['run_id']}",
            "report_url": f"/runs/{status['run_id']}/report",
        },
        status_code=202,
        headers={"Cache-Control": "no-store"},
    )


async def api_start_dual_run(request):
    try:
        payload = await _read_json_body(request, max_bytes=128 * 1024)
        return await _launch_dual_run_payload(payload)
    except ContractError as exc:
        return _contract_error_json(exc)


async def api_start_v1_user_cohort_run(request):
    try:
        payload = await _read_json_body(request, max_bytes=128 * 1024)
        return await _launch_dual_run_payload(payload, forced_mode=SOURCE_MODE_USER_COHORT)
    except ContractError as exc:
        return _contract_error_json(exc)


async def api_start_v1_simulated_run(request):
    try:
        payload = await _read_json_body(request, max_bytes=128 * 1024)
        return await _launch_dual_run_payload(payload, forced_mode=SOURCE_MODE_SIMULATED_COHORT)
    except ContractError as exc:
        return _contract_error_json(exc)


async def api_get_dual_run(request):
    try:
        result = runner.get(request.path_params["run_id"])
    except (ContractError, OSError, ValueError):
        return JSONResponse({"error": "运行不存在或已清除。"}, status_code=404)
    out = dict(result)
    if result.get("state") not in ("queued", "running") and result.get("kind") != "synthetic_tte":
        out["rendered_html"] = render_dualmode_run_view(result, runner.root)
    return JSONResponse(out, headers={"Cache-Control": "no-store"})


# ----------------------------------------------------------------------
# Browser Pages (Workbench Home + Dual-Mode & Legacy Run/Report Views)
# ----------------------------------------------------------------------


async def home(request):
    return page(workbench_home_view())


async def api_v1_tte_benchmark(request):
    if not is_benchmarking_enabled():
        return JSONResponse(
            benchmarking_disabled_payload(),
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    try:
        payload = await _read_json_body(request)
        report = benchmark_from_payload(payload, runner)
        return JSONResponse(report, headers={"Cache-Control": "no-store"})
    except ContractError as exc:
        status_code = 404 if exc.code in {"RUN_NOT_FOUND", "INVALID_RUN_ID"} else 400
        return JSONResponse(
            {
                "error": f"{exc.code}: {exc}",
                "code": exc.code,
                "interpretive_guardrail": interpretive_guardrail(),
            },
            status_code=status_code,
            headers={"Cache-Control": "no-store"},
        )


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
    is_running = result["state"] in ("queued", "running")
    if result.get("kind", "synthetic_tte") != "synthetic_tte":
        return page(render_dualmode_run_view(result, runner.root), refresh=is_running)
    return page(result_view(result, runner.root), refresh=is_running)


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
        if result.get("kind", "synthetic_tte") != "synthetic_tte":
            pmode = result.get("project_mode", "dual")
            title_prefix = (
                "真实队列目标试验模拟（真项目）分析报告"
                if pmode == SOURCE_MODE_USER_COHORT
                else "目标试验模拟（无数据模拟预演项目）可行性报告"
            )
            response = page(
                render_dualmode_run_view(result, runner.root, standalone_report=True),
                title=f"EmulatRx · {title_prefix}",
            )
            if request.query_params.get("download") == "1":
                slug = "user-cohort-report" if pmode == SOURCE_MODE_USER_COHORT else "simulated-rehearsal-report"
                response.headers["Content-Disposition"] = f'attachment; filename="emulatrx-{slug}-{result["run_id"]}.html"'
            return response

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
    return JSONResponse({"status": "ok", "service": "emulatrx-dualmode-tte-workbench"})


mcp_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        yield


inner = Starlette(
    routes=[
        Route("/healthz", health),
        Route("/", home),
        Route("/api/workbench/bootstrap", api_workbench_bootstrap),
        Route("/api/samples/{sample_id}/download", api_download_sample),
        Route("/api/sessions/upload", api_upload_or_load_session, methods=["POST"]),
        Route("/api/sessions/upload-ticket/{ticket_id}", api_upload_with_ticket, methods=["POST"]),
        Route("/api/sessions/{session_id}/note", api_update_session_note, methods=["POST"]),
        Route("/api/sessions/{session_id}", api_get_session, methods=["GET"]),
        Route("/api/sessions/{session_id}", api_delete_session, methods=["DELETE"]),
        Route("/api/cohorts/{session_id}", api_get_session, methods=["GET"]),
        Route("/api/cohorts/{session_id}", api_delete_session, methods=["DELETE"]),
        Route("/api/idea/parse", api_parse_idea, methods=["POST"]),
        Route("/api/rcts/search", api_search_rcts, methods=["POST"]),
        Route("/api/rcts/draft", api_draft_mapping_or_protocol, methods=["POST"]),
        Route("/api/dual-runs", api_start_dual_run, methods=["POST"]),
        Route("/api/dual-runs/{run_id}", api_get_dual_run),
        Route("/api/v1/projects/user-cohort/runs", api_start_v1_user_cohort_run, methods=["POST"]),
        Route("/api/v1/projects/simulated/runs", api_start_v1_simulated_run, methods=["POST"]),
        Route("/api/v1/tte/benchmark", api_v1_tte_benchmark, methods=["POST"]),
        Route("/protocol/new", new_protocol),
        Route("/protocol/edit", edit_protocol, methods=["POST"]),
        Route("/validate", validate_form, methods=["POST"]),
        Route("/runs", create_run, methods=["POST"]),
        Route("/runs/{run_id}/cohort", show_cohort),
        Route("/runs/{run_id}/report", show_report),
        Route("/runs/{run_id}", show_run),
        Route("/runs/{run_id}/artifacts/{name:path}", download_artifact),
        Mount("/", app=mcp_app),
    ],
    lifespan=lifespan,
)


class AuthenticationGate:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/healthz" or scope["path"].startswith("/api/sessions/upload-ticket/"):
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
                valid = (
                    scheme.lower() == b"basic"
                    and user == "preview"
                    and secrets.compare_digest(password, WEB_PASSWORD)
                )
            except (ValueError, UnicodeError):
                valid = False
            challenge = b'Basic realm="EmulatRx synthetic TTE preview"'
        if not valid:
            response = JSONResponse(
                {"error": "authentication required"},
                status_code=401,
                headers={"WWW-Authenticate": challenge.decode("ascii"), "Cache-Control": "no-store"},
            )
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)


app = AuthenticationGate(inner)
