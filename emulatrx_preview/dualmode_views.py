"""Modern Chinese four-step Dual-Mode Workbench UI and server-rendered dual-mode report views."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

from emulatrx_tte._common import (
    OBSERVATIONAL_CAUSAL_ASSUMPTIONS,
    SOURCE_MODE_SIMULATED_COHORT,
    SOURCE_MODE_USER_COHORT,
)
from emulatrx_tte.llm_assist import get_llm_status
from emulatrx_tte.sample_cohorts import CLINICAL_IDEA_PRESETS, SAMPLE_COHORT_SPECS


def esc(val: object) -> str:
    return html.escape(str(val), quote=True)


def _fmt_num(val: object, digits: int = 4) -> str:
    if val is None or val == "":
        return "—"
    try:
        f = float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return str(val)
    if not math.isfinite(f):
        return "—"
    return f"{f:.{digits}f}"


def _step_cif_svg(curve_paths: list[Path], column: str, title: str, arm_labels: dict[str, str]) -> str:
    series = []
    for cpath in curve_paths:
        with cpath.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        pts = [(float(r["time_day"]), float(r[column])) for r in rows]
        series.append((rows[0]["treatment_arm"], pts))
    if not series:
        return ""
    max_day = max(pt[0] for _, pts in series for pt in pts) or 1.0
    max_y = max((pt[1] for _, pts in series for pt in pts), default=0.25)
    y_top = min(1.0, max(0.20, math.ceil(max_y * 10.0 + 0.5) / 10.0))
    colors = ("#475569", "#0f766e", "#7c3aed", "#b45309")
    paths_html = []
    legends = []
    for idx, (arm_id, pts) in enumerate(series):
        d_cmd = f"M48.0,235.0 " + " ".join(
            f"H{48 + 600 * (x / max_day):.1f} V{235 - 205 * min(y / y_top, 1.0):.1f}"
            for x, y in pts
        )
        col = colors[idx % len(colors)]
        paths_html.append(f"<path class='cif-step' d='{d_cmd}' fill='none' stroke='{col}' stroke-width='2.6'/>")
        legends.append(f"<span style='color:{col};font-weight:700'>●</span> {esc(arm_labels.get(arm_id, arm_id))}")
    return (
        f"<figure class='chart-card'><figcaption>{esc(title)}</figcaption>"
        f"<svg viewBox='0 0 680 272' role='img' aria-label='{esc(title)}'>"
        "<line x1='48' y1='30' x2='652' y2='30' stroke='#e2e8f0' stroke-dasharray='4 4'/>"
        "<line x1='48' y1='132.5' x2='652' y2='132.5' stroke='#e2e8f0' stroke-dasharray='4 4'/>"
        "<path d='M48 24 V235 H656' fill='none' stroke='#64748b' stroke-width='1.3'/>"
        f"<text x='6' y='35' font-size='11' fill='#475569'>{y_top:.2f}</text>"
        f"<text x='6' y='136' font-size='11' fill='#475569'>{y_top * 0.5:.2f}</text>"
        "<text x='18' y='239' font-size='11' fill='#475569'>0.0</text>"
        "<text x='48' y='258' font-size='11' fill='#475569'>0 天</text>"
        f"<text x='605' y='258' font-size='11' fill='#475569'>{max_day:.0f} 天</text>"
        f"{''.join(paths_html)}</svg>"
        f"<div class='legend'>{' &nbsp;·&nbsp; '.join(legends)}</div></figure>"
    )


def _love_plot_svg(balance_rows: list[dict], cov_labels: dict[str, str], max_gate_smd: float) -> str:
    if not balance_rows:
        return ""
    h = max(180, 56 + 28 * len(balance_rows))
    max_smd = max(
        max_gate_smd * 2.0,
        *(
            float(r[k])
            for r in balance_rows
            for k in ("abs_smd_unweighted", "abs_smd_weighted")
            if r.get(k) not in (None, "", "inf")
        ),
        0.25,
    )
    x0, xw = 220.0, 420.0
    gate_x = x0 + xw * min(max_gate_smd / max_smd, 1.0)
    elems = [
        f"<line x1='{x0}' y1='22' x2='{x0}' y2='{h - 28}' stroke='#64748b'/>",
        f"<line x1='{gate_x:.1f}' y1='22' x2='{gate_x:.1f}' y2='{h - 28}' stroke='#dc2626' stroke-dasharray='5 4' stroke-width='1.5'/>",
        f"<text x='{gate_x + 4:.1f}' y='34' font-size='11' fill='#dc2626'>门禁阈值 {max_gate_smd:g}</text>",
    ]
    for idx, r in enumerate(balance_rows):
        cy = 48 + idx * 26
        lbl = cov_labels.get(r["covariate"], r["covariate"])[:26]
        su = float(r["abs_smd_unweighted"]) if r.get("abs_smd_unweighted") not in (None, "", "inf") else 0.0
        sw = float(r["abs_smd_weighted"]) if r.get("abs_smd_weighted") not in (None, "", "inf") else 0.0
        xu = x0 + xw * min(su / max_smd, 1.0)
        x_wt = x0 + xw * min(sw / max_smd, 1.0)
        elems.append(f"<line x1='{x0}' y1='{cy}' x2='{x0 + xw}' y2='{cy}' stroke='#f1f5f9'/>")
        elems.append(f"<text x='{x0 - 8}' y='{cy + 4}' text-anchor='end' font-size='11.5' fill='#1e293b'>{esc(lbl)}</text>")
        elems.append(f"<line x1='{xu:.1f}' y1='{cy}' x2='{x_wt:.1f}' y2='{cy}' stroke='#94a3b8' stroke-width='1.4'/>")
        elems.append(f"<circle cx='{xu:.1f}' cy='{cy}' r='4.5' fill='#f59e0b'/>")
        elems.append(f"<circle cx='{x_wt:.1f}' cy='{cy}' r='4.5' fill='#0f766e'/>")
    return (
        "<figure class='chart-card'><figcaption>协变量均衡 Love Plot（加权前 vs 加权后 |SMD|）</figcaption>"
        f"<svg viewBox='0 0 675 {h}' role='img' aria-label='Love Plot'>"
        + "".join(elems)
        + f"<text x='{x0}' y='{h - 8}' font-size='11' fill='#475569'>0.00</text>"
        + f"<text x='{x0 + xw - 30}' y='{h - 8}' font-size='11' fill='#475569'>{max_smd:.2f}</text>"
        "</svg><div class='legend'><span style='color:#f59e0b;font-weight:700'>●</span> 加权前 |SMD| &nbsp;·&nbsp; "
        "<span style='color:#0f766e;font-weight:700'>●</span> 加权后 |SMD| &nbsp;·&nbsp; "
        "<span style='color:#dc2626'>┆</span> 门禁上限</div></figure>"
    )


def _propensity_overlap_svg(ps_overlap: dict, arm_labels: dict[str, str]) -> str:
    counts_by_arm = ps_overlap.get("counts_by_arm", {})
    arms = list(counts_by_arm.keys())
    if len(arms) < 2:
        return ""
    c0 = counts_by_arm[arms[0]]
    c1 = counts_by_arm[arms[1]]
    max_c = max([*c0, *c1, 1])
    bars = []
    for b in range(10):
        x_base = 52 + b * 58
        h0 = 170 * (c0[b] / max_c)
        h1 = 170 * (c1[b] / max_c)
        bars.append(f"<rect x='{x_base}' y='{215 - h0:.1f}' width='22' height='{h0:.1f}' fill='#64748b' opacity='0.82'/>")
        bars.append(f"<rect x='{x_base + 24}' y='{215 - h1:.1f}' width='22' height='{h1:.1f}' fill='#0f766e' opacity='0.88'/>")
        bars.append(f"<text x='{x_base + 23}' y='234' text-anchor='middle' font-size='10.5' fill='#475569'>{b/10:.1f}-{ (b+1)/10:.1f}</text>")
    return (
        "<figure class='chart-card'><figcaption>两臂倾向得分重叠分布（P(试验策略 | 基线协变量)）</figcaption>"
        "<svg viewBox='0 0 660 252' role='img' aria-label='倾向得分重叠图'>"
        "<path d='M44 25 V215 H642' fill='none' stroke='#64748b'/>"
        + "".join(bars)
        + "</svg>"
        f"<div class='legend'><span style='color:#64748b;font-weight:700'>■</span> 对照臂：{esc(arm_labels.get(arms[0], arms[0]))} &nbsp;·&nbsp; "
        f"<span style='color:#0f766e;font-weight:700'>■</span> 试验臂：{esc(arm_labels.get(arms[1], arms[1]))}</div></figure>"
    )


def _attrition_waterfall_html(attrition_rows: list[dict], arm_labels: dict[str, str]) -> str:
    if not attrition_rows:
        return "<p>无筛选记录。</p>"
    top_n = max(int(attrition_rows[0].get("remaining_n", 1) or 1), 1)
    tr_list = []
    for r in attrition_rows:
        rem = int(r.get("remaining_n", 0) or 0)
        exc = int(r.get("excluded_n", 0) or 0)
        pct = min(100.0, max(2.0, 100.0 * rem / top_n)) if rem > 0 else 0.0
        try:
            ac = json.loads(r.get("arm_counts_json", "{}"))
            arm_str = " / ".join(f"{arm_labels.get(k, k)}: {v}" for k, v in ac.items())
        except Exception:
            arm_str = "—"
        tr_list.append(
            "<tr>"
            f"<td><strong>{esc(r.get('description') or r.get('step_id', ''))}</strong></td>"
            f"<td><div style='background:#e2e8f0;width:150px;height:12px;border-radius:3px'>"
            f"<div style='background:#0f766e;width:{pct:.1f}%;height:12px;border-radius:3px'></div></div></td>"
            f"<td><strong>{rem}</strong></td>"
            f"<td>{f'-{exc}' if exc > 0 else '0'}</td>"
            f"<td>{esc(arm_str)}</td>"
            "</tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr>"
        "<th>纳排与完整性过滤步骤</th><th>留存比例</th><th>剩余例数</th><th>本步剔除</th><th>两臂留存分布</th>"
        "</tr></thead><tbody>" + "".join(tr_list) + "</tbody></table></div>"
    )


def render_dualmode_run_view(result: dict, root: Path, *, standalone_report: bool = False) -> str:
    """Server-side HTML view for both `/runs/{id}` and `/runs/{id}/report` on dual-mode runs."""
    state = result.get("state", "unknown")
    run_id = result["run_id"]
    pmode = result.get("project_mode", SOURCE_MODE_SIMULATED_COHORT)
    is_user = pmode == SOURCE_MODE_USER_COHORT

    if state in ("queued", "running"):
        badge = "🟢 自有队列真项目 TTE" if is_user else "🟣 无数据模拟预演项目 TTE"
        return (
            f"<section class='mode-banner {'user-mode' if is_user else 'sim-mode'}'>"
            f"<h2>{badge} · 正在分析计算中</h2>"
            f"<p>当前阶段：<code>{esc(result.get('stage', 'running'))}</code>；页面每 2 秒自动刷新（运行编号 <code>{esc(run_id)}</code>）。</p>"
            "</section>"
        )
    if state in ("failed", "interrupted"):
        return (
            f"<div class='error'><strong>分析未完成：</strong>{esc(result.get('error', state))}</div>"
            "<p><a class='primary-link' href='/'>返回双轨研究工作台</a></p>"
        )

    proto = result["normalized_protocol"]
    diag = result["diagnostics"]
    gate = result["gate_status"]
    trct = result.get("target_rct") or proto.get("target_rct") or {}
    arm_labels = {s["id"]: s["label"] for s in proto["treatment_strategies"]}
    ingest_man = result.get("ingest_manifest") or {}
    cov_labels = ingest_man.get("covariate_display_labels") or {
        "x_age_std": "模拟标准化基线年龄",
        "x_biomarker_std": "模拟标准化生物标志物",
        "x_comorbidity_bin": "模拟二元基础合并症",
        "x_high_risk_bin": "模拟二元高危临床分期",
    }

    if is_user:
        banner_cls = "user-mode"
        report_title = "🟢 真实队列目标试验模拟（真项目）分析报告"
        subtitle = (
            f"数据来源：用户上传脱敏自有队列（文件：{esc(ingest_man.get('input_filename', 'cohort.csv'))}，"
            f"原始 SHA-256：<code>{esc((ingest_man.get('input_sha256') or '')[:16])}…</code>） · 观察性因果推断模式（source_mode=user_cohort）"
        )
    else:
        banner_cls = "sim-mode"
        report_title = "🟣 目标试验模拟（无数据模拟预演项目）可行性报告"
        subtitle = "数据来源：基于对标 RCT 参数虚拟生成的立项预演队列（source_mode=simulated_cohort） · 仅供方案可行性与偏倚控制预演，绝非真实临床疗效证据"

    sess_id = result.get("session_id")
    wipe_btn = (
        f" <button type='button' class='danger-btn print-hide' onclick=\"purgeSession('{esc(sess_id)}')\">🗑️ 一键清除本次会话上传数据</button>"
        if (is_user and sess_id)
        else ""
    )
    purge_script = (
        "<script>async function purgeSession(sid){if(!confirm('确认物理删除本次会话上传的原始队列及全部派生分析文件吗？'))return;"
        "const r=await fetch('/api/sessions/'+sid,{method:'DELETE'});const d=await r.json();"
        "alert('会话数据已彻底物理清除（共清理 '+d.purged_files+' 个文件）。');window.location.href='/';}</script>"
        if (is_user and sess_id)
        else ""
    )

    parts = [
        purge_script,
        f"<section class='mode-banner {banner_cls}'>"
        f"<p class='eyebrow'>{'OWNED COHORT REAL TTE PROJECT' if is_user else 'NO-DATA SIMULATED TTE FEASIBILITY REHEARSAL'}</p>"
        f"<h2>{report_title}</h2>"
        f"<p><strong>研究标题：</strong>{esc(proto['study_title'])}</p>"
        f"<p>{subtitle}</p>",
    ]
    if trct.get("nct_id"):
        rct_role_label = "参考框架 RCT（仅借用纳排与随访窗）" if trct.get("arm_alignment_warning") else "对标目标 RCT"
        parts.append(
            f"<p><strong>{rct_role_label}：</strong><a href='{esc(trct.get('record_url', 'https://clinicaltrials.gov/study/' + trct['nct_id']))}' target='_blank' rel='noopener'>"
            f"{esc(trct['nct_id'])}</a> — {esc(trct.get('brief_title', ''))} "
            f"（来源标识：<code>{esc(trct.get('source', 'clinicaltrials_gov_api_v2'))}</code>"
            + (f"，分组对齐状态：<code>{esc(trct['arm_alignment_status'])}</code>" if trct.get("arm_alignment_status") else "")
            + "）</p>"
        )
        if trct.get("arm_alignment_warning"):
            parts.append(
                "<div class='notice arm-mismatch-callout' style='background:#fffbeb;border-left:5px solid #d97706;color:#92400e'>"
                f"<strong>⚠️ 对照不一致参考声明：</strong>{esc(trct['arm_alignment_warning'])}"
                "</div>"
            )
    parts.append(
        "<div class='actions print-hide'>"
        f"<a class='primary-link' href='/runs/{esc(run_id)}/report?download=1'>下载独立 HTML 报告</a>"
        + (f" <a class='secondary-link' href='/runs/{esc(run_id)}/report'>打开打印友好版报告</a>" if not standalone_report else f" <a class='secondary-link' href='/runs/{esc(run_id)}'>返回交互结果页</a>")
        + (f" <a class='secondary-link' href='/runs/{esc(run_id)}/artifacts/bundle/audit_report.md'>下载审计 Markdown</a>" if gate == "passed" else "")
        + " <a class='secondary-link' href='/'>返回双轨工作台新建项目</a>"
        + wipe_btn
        + "</div></section>"
    )

    parts.append(
        "<div class='metric-grid'>"
        f"<div><strong>{result.get('cohort_rows', 0)} 例</strong><span>{'上传原始总行数' if is_user else '虚拟生成样本量'}</span></div>"
        f"<div><strong>{result.get('eligible_rows', 0)} 例</strong><span>纳排与完整性筛选后风险集</span></div>"
        f"<div><strong>{'通过' if gate == 'passed' else '阻断'}</strong><span>倾向加权诊断门禁</span></div>"
        f"<div><strong>{_fmt_num(diag.get('max_abs_smd_weighted'), 4)}</strong><span>加权后最大 |SMD|</span></div>"
        f"<div><strong>{_fmt_num(diag.get('ess_diagnostics', {}).get('min_ess_ratio_observed'), 3)}</strong><span>最小有效样本量比 ESS</span></div>"
        "</div>"
    )

    if is_user:
        zero_note = (ingest_man.get("zero_competing_events_notice") or "")
        if zero_note:
            parts.append(f"<div class='notice'><strong>无竞争事件退化提示：</strong>{esc(zero_note)}</div>")
        dropped = ingest_man.get("dropped_constant_or_collinear_covariates") or []
        if dropped:
            d_txt = "、".join(f"{d['column']} ({d['reason']})" for d in dropped)
            parts.append(f"<div class='notice'><strong>常数/共线协变量自动保护：</strong>已自动剔除零方差或完全共线列：{esc(d_txt)}，防止倾向模型矩阵奇异。</div>")
        parts.append(
            "<h2>一、观察性因果推断四大识别前提与局限声明</h2>"
            "<div class='assumption-box'><ul>"
            + "".join(f"<li>{esc(a)}</li>" for a in OBSERVATIONAL_CAUSAL_ASSUMPTIONS)
            + "</ul><p><strong>未测混杂局限声明：</strong>本研究系基于既有观察性队列对标随机对照试验（Target Trial Emulation），"
            "尽管通过显式时间零点对齐、基线重叠加权/IPTW 门禁及竞争风险模型控制已测基线混杂与生存选择偏倚，"
            "仍无法排除未记录临床混杂因素、信息性删失或入组前选择偏倚，请结合下方指定时点 RR_CIF E-value 审慎解读。</p></div>"
        )
    else:
        parts.append(
            "<h2>一、无数据模拟预演声明与方案定位</h2>"
            "<div class='assumption-box sim-box'>"
            "<p><strong>模拟立项可行性预演声明：</strong>本报告全部个体时间与结局均根据目标 RCT 的干预对比结构与您设定的基线混杂强度/风险率参数虚拟生成，"
            "<strong>不包含任何真实患者数据，绝不可引用为临床治疗有效或安全的实证证据</strong>；用于正式提取医院专病库数据前检验七要素方案闭环、有效样本量损耗与偏倚消除能力。</p></div>"
        )

    parts.append("<h2>二、队列纳排瀑布（Attrition Waterfall）</h2>")
    parts.append(_attrition_waterfall_html(result.get("attrition", []), arm_labels))

    parts.append("<h2>三、倾向得分重叠与协变量均衡门禁诊断</h2>")
    max_gate_smd = float(proto["estimand"]["diagnostic_gates"]["max_abs_smd"])
    parts.append(_love_plot_svg(result.get("balance", []), cov_labels, max_gate_smd))
    if gate == "passed":
        parts.append(_propensity_overlap_svg(result.get("propensity_overlap", {}), arm_labels))

    if gate != "passed":
        failures_json = esc(json.dumps(diag.get("gate_failures", []), ensure_ascii=False, indent=2))
        parts.append(
            "<div class='error'><strong>统计诊断门禁未通过（Fail-Closed 阻断，退出码 3）：</strong>"
            "<p>为防止在严重正值性违背、有效样本量过低或残余协变量失衡时输出误导性效应估计，系统已按预设门禁自动阻断下游生存分析。</p>"
            f"<pre>{failures_json}</pre></div>"
        )
    else:
        curves_dir = Path(root) / run_id / "bundle" / "curves"
        curve_files = sorted(curves_dir.glob("*.csv")) if curves_dir.is_dir() else []
        parts.append("<h2>四、加权 Aalen–Johansen 累计发生率阶梯曲线（Step-After CIF）</h2>")
        parts.append(_step_cif_svg(curve_files, "cif_primary", "主要结局事件累计发生率 CIF_1(t)", arm_labels))
        parts.append(_step_cif_svg(curve_files, "cif_competing", "竞争风险事件累计发生率 CIF_2(t)", arm_labels))

        est = result["estimates"]
        parts.append("<h2>五、竞争风险核心五量并列效应表与 E-value 敏感性参考</h2>")
        rows_html = []
        for row in est.get("co_presented_comparison_table", []):
            rows_html.append(
                "<tr>"
                f"<td><strong>{row['horizon_day']:g} 天</strong></td>"
                f"<td>{_fmt_num(row['cif_primary_ref'])}</td>"
                f"<td>{_fmt_num(row['cif_primary_active'])}</td>"
                f"<td>{_fmt_num(row['rd_primary'])} [{_fmt_num(row['rd_primary_ci_lower'])}, {_fmt_num(row['rd_primary_ci_upper'])}]</td>"
                f"<td>{_fmt_num(row['rr_cif_primary'])} [{_fmt_num(row['rr_cif_primary_ci_lower'])}, {_fmt_num(row['rr_cif_primary_ci_upper'])}]</td>"
                f"<td>{_fmt_num(row['cif_competing_ref'])} / {_fmt_num(row['cif_competing_active'])}</td>"
                f"<td>{_fmt_num(row['shr_primary'])} [{_fmt_num(row['shr_primary_ci_lower'])}, {_fmt_num(row['shr_primary_ci_upper'])}]</td>"
                f"<td>{_fmt_num(row['cshr_primary'])} [{_fmt_num(row['cshr_primary_ci_lower'])}, {_fmt_num(row['cshr_primary_ci_upper'])}]</td>"
                f"<td>{_fmt_num(row['cshr_competing'])}</td>"
                f"<td>{_fmt_num(row['evalue_rr_cif_point'], 2)}（CI界：{_fmt_num(row['evalue_rr_cif_ci_limit'], 2)}）</td>"
                "</tr>"
            )
        parts.append(
            "<div class='scroll'><table><thead><tr>"
            "<th>评估时点</th><th>对照 CIF_1</th><th>试验 CIF_1</th><th>主要风险差 RD_1 (95%CI)</th>"
            "<th>累计发生率比 RR_CIF (95%CI)</th><th>竞争 CIF_2 (对照/试验)</th>"
            "<th>Fine–Gray sHR (95%CI)</th><th>主要原因 csHR_1 (95%CI)</th><th>竞争原因 csHR_2</th><th>RR_CIF E-value</th>"
            "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div>"
        )

        if not is_user and est.get("simulation_recovery_table"):
            parts.append("<h2>六、预设模拟真值 vs 加权 TTE 估计恢复对比表（方案可行性核验）</h2>")
            rec_rows = []
            for rr in est["simulation_recovery_table"]:
                cov_badge = "✓ 95%CI 覆盖真值" if rr.get("covered_in_95ci") is True else ("未覆盖" if rr.get("covered_in_95ci") is False else "参考对比")
                rec_rows.append(
                    "<tr>"
                    f"<td>{esc(rr['metric'])}</td>"
                    f"<td><strong>{_fmt_num(rr['dgp_reference_truth'])}</strong></td>"
                    f"<td>{_fmt_num(rr['weighted_estimate'])} [{_fmt_num(rr['ci_lower'])}, {_fmt_num(rr['ci_upper'])}]</td>"
                    f"<td>{esc(cov_badge)}</td>"
                    "</tr>"
                )
            parts.append(
                "<div class='scroll'><table><thead><tr>"
                "<th>评估指标</th><th>预设模拟 DGP 真值</th><th>加权 TTE 估计值 (95%CI)</th><th>偏倚消除与区间覆盖</th>"
                "</tr></thead><tbody>" + "".join(rec_rows) + "</tbody></table></div>"
            )

    tech_payload = {
        "normalized_protocol": proto,
        "ingest_manifest": ingest_man if is_user else None,
        "preview_manifest": result.get("preview_manifest"),
    }
    parts.append(
        "<details class='tech-drawer print-hide'><summary>查看底层技术详情（规范化七要素协议 JSON / 字段映射与 SHA-256 审计清单）</summary>"
        f"<pre>{esc(json.dumps(tech_payload, ensure_ascii=False, indent=2))}</pre></details>"
    )
    return "".join(parts)


def workbench_home_view() -> str:
    """Render the interactive 4-step Chinese Dual-Mode Target Trial Emulation Workbench."""
    llm_info = get_llm_status()
    samples_list = list(SAMPLE_COHORT_SPECS.values())
    ideas_list = list(CLINICAL_IDEA_PRESETS)

    boot_json = json.dumps(
        {
            "llm_status": llm_info,
            "samples": samples_list,
            "ideas": ideas_list,
        },
        ensure_ascii=False,
    )

    llm_badge_cls = "pill-ai" if llm_info["enabled"] else "pill-local"
    return f"""
<style>
.wb-topbar{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;padding:12px 16px;background:#f1f6f4;border:1px solid var(--line);border-radius:6px;margin:12px 0 20px}}
.pill{{display:inline-flex;align-items:center;gap:6px;padding:4px 11px;border-radius:999px;font-size:.82rem;font-weight:700}}
.pill-ai{{background:#dcfce7;color:#166534;border:1px solid #86efac}}
.pill-local{{background:#e0f2fe;color:#075985;border:1px solid #7dd3fc}}
.pill-priv{{background:#fef3c7;color:#92400e;border:1px solid #fcd34d}}
.stepper{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:18px 0 24px}}
.step-tab{{padding:11px 12px;border:1px solid var(--line);background:#fff;border-radius:6px;font-size:.9rem;font-weight:700;color:var(--muted);cursor:pointer;text-align:left;transition:.15s}}
.step-tab.active{{border-color:var(--accent);background:#e6f4f0;color:var(--ink);box-shadow:0 2px 8px #176b5e1a}}
.step-tab small{{display:block;font-weight:400;font-size:.76rem;color:var(--muted)}}
.entry-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px;margin:16px 0}}
.entry-card{{border:2px solid var(--line);border-radius:8px;padding:20px 22px;background:#fff;cursor:pointer;transition:.15s}}
.entry-card.selected-user{{border-color:#0f766e;background:#f0fdfa;box-shadow:0 6px 20px #0f766e17}}
.entry-card.selected-sim{{border-color:#6d28d9;background:#faf5ff;box-shadow:0 6px 20px #6d28d917}}
.badge-user{{display:inline-block;background:#0f766e;color:#fff;padding:3px 10px;border-radius:4px;font-size:.8rem;font-weight:700}}
.badge-sim{{display:inline-block;background:#6d28d9;color:#fff;padding:3px 10px;border-radius:4px;font-size:.8rem;font-weight:700}}
.dropzone{{border:2px dashed #64998e;border-radius:8px;padding:24px;text-align:center;background:#f7fbf9;margin:14px 0;transition:.15s}}
.dropzone.dragover{{background:#e0f2fe;border-color:#0284c7}}
.sample-box{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin:14px 0}}
.sample-card{{border:1px solid #cbd5e1;border-radius:6px;padding:14px 16px;background:#fff}}
.rct-grid{{display:grid;grid-template-columns:1fr;gap:14px;margin:16px 0}}
.rct-card{{border:1.5px solid #cbd5e1;border-radius:8px;padding:16px 18px;background:#fff}}
.rct-card.selected{{border-color:var(--accent);background:#f0fdf9}}
.rct-meta{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px;font-size:.83rem}}
.tag{{background:#f1f5f9;color:#334155;padding:2px 8px;border-radius:4px;font-weight:600}}
.tag-score{{background:#dcfce7;color:#166534;font-weight:700}}
.tag-score-warn{{background:#fef3c7;color:#92400e;font-weight:700}}
.tag-score-mismatch{{background:#fee2e2;color:#991b1b;font-weight:700}}
.tag-align-exact{{background:#dcfce7;color:#166534;border:1px solid #86efac;font-weight:700}}
.tag-align-class{{background:#fef3c7;color:#92400e;border:1px solid #fcd34d;font-weight:700}}
.tag-align-mismatch{{background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;font-weight:700}}
.tag-fallback{{background:#ffedd5;color:#9a3412}}
.mismatch-confirm-box{{background:#fffbeb;border:1.5px solid #f59e0b;border-radius:6px;padding:12px 16px;margin:12px 0;color:#78350f}}
.mode-banner{{padding:20px 24px;border-radius:8px;margin:16px 0}}
.mode-banner.user-mode{{background:#ecfdf5;border-left:6px solid #0f766e}}
.mode-banner.sim-mode{{background:#f5f3ff;border-left:6px solid #6d28d9}}
.assumption-box{{background:#fff;border:1px solid #cbd5e1;border-radius:6px;padding:14px 18px;margin:12px 0}}
.danger-btn{{background:#b91c1c!important;border-color:#b91c1c!important;color:#fff!important}}
.danger-btn:hover{{background:#991b1b!important}}
.secondary-link{{display:inline-block;padding:10px 16px;border:1px solid var(--accent);color:var(--accent);background:#fff;text-decoration:none;font-weight:700}}
.preset-chips{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin:12px 0}}
.preset-chip{{border:1px solid #cbd5e1;background:#fff;color:var(--ink);text-align:left;padding:10px 12px;border-radius:6px;font-weight:600;font-size:.88rem}}
.preset-chip:hover{{background:#f5f3ff;border-color:#6d28d9;color:var(--ink)}}
.outcome-chip-row{{display:grid;grid-template-columns:1.5fr repeat(3,1fr);gap:8px;align-items:center;padding:7px 10px;border-bottom:1px solid #e2e8f0}}
.cov-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px;margin:10px 0}}
.cov-item{{display:flex;align-items:flex-start;gap:8px;padding:8px 10px;border:1px solid #e2e8f0;border-radius:5px;background:#fff;font-size:.88rem}}
.cov-item input{{width:auto;margin-top:4px}}
.elig-row{{display:grid;grid-template-columns:1fr 1.4fr 1fr 1fr auto;gap:8px;margin:8px 0;align-items:center}}
.hidden{{display:none!important}}
.idea-input{{width:100%;min-height:92px;padding:10px 12px;border:1px solid #94a3b8;border-radius:5px;font:inherit;line-height:1.5}}
.chart-card{{background:#fff;border:1px solid var(--line);padding:12px;border-radius:6px}}
@media(max-width:760px){{.stepper{{grid-template-columns:1fr 1fr}}.elig-row,.outcome-chip-row{{grid-template-columns:1fr}}}}
</style>

<div class="wb-topbar">
  <div>
    <span id="llm-status-pill" class="pill {llm_badge_cls}">{esc(llm_info['label_zh'])}</span>
    <span class="pill pill-priv">🔒 隐私红线：个体患者行记录 0 外发 · 会话私网隔离</span>
  </div>
  <div id="session-bar" class="hidden">
    <span id="session-label" style="font-size:.84rem;color:var(--muted);margin-right:8px"></span>
    <button type="button" class="danger-btn" style="padding:7px 13px;font-size:.85rem" onclick="wipeCurrentSession()">🗑️ 一键清除本次会话上传数据</button>
  </div>
</div>

<div class="stepper" role="tablist" aria-label="目标试验模拟四步向导">
  <button type="button" class="step-tab active" id="tab-step-1" onclick="goStep(1)">第 1 步 · 选择起点<small>自有队列上传 / 临床研究想法</small></button>
  <button type="button" class="step-tab" id="tab-step-2" onclick="goStep(2)">第 2 步 · 匹配目标 RCT<small>ClinicalTrials.gov 实时检索选定</small></button>
  <button type="button" class="step-tab" id="tab-step-3" onclick="goStep(3)">第 3 步 · 方案与字段对齐<small>零 JSON 映射 / 模拟预演参数</small></button>
  <button type="button" class="step-tab" id="tab-step-4" onclick="goStep(4)">第 4 步 · 双态分析报告<small>门禁诊断、CIF 曲线与审计包</small></button>
</div>

<div id="wb-alert" class="error hidden" role="alert"></div>

<!-- ==================== STEP 1 ==================== -->
<section id="panel-step-1">
  <div class="entry-cards">
    <div id="card-mode-user" class="entry-card selected-user" onclick="selectMode('user_cohort')">
      <span class="badge-user">🟢 路径一 · 真项目模式 (user_cohort)</span>
      <h3 style="margin:10px 0 6px">我有一批自有队列数据 → 匹配目标 RCT → 跑真项目 TTE</h3>
      <p style="margin:0;font-size:.92rem;color:var(--muted)">拖拽上传医院/专病脱敏 CSV/TSV（或一键载入下方真实临床表头样表演示），自动完成隐私安全画像、匹配对标 RCT、可视化映射列名并输出真项目因果分析报告。</p>
    </div>
    <div id="card-mode-sim" class="entry-card" onclick="selectMode('simulated_cohort')">
      <span class="badge-sim">🟣 路径二 · 无数据模拟预演 (simulated_cohort)</span>
      <h3 style="margin:10px 0 6px">我只有研究想法（尚无数据）→ 匹配目标 RCT → 模拟预演可行性</h3>
      <p style="margin:0;font-size:.92rem;color:var(--muted)">用中文描述临床问题，自动提取 PICO 并实时检索 ClinicalTrials.gov 对标试验，虚拟生成带混杂与竞争死亡的队列预演方案可行性与样本量需求。</p>
    </div>
  </div>

  <!-- Path 1: User Cohort Upload & Sample Demo -->
  <div id="subpanel-user">
    <fieldset>
      <legend>上传脱敏队列文件（CSV / TSV）或一键载入真实临床表头样表</legend>
      <div id="dropzone" class="dropzone">
        <p style="margin:0 auto 10px;font-weight:700">拖拽 <code>.csv</code> / <code>.tsv</code> 队列文件至此处，或点击选择本地文件</p>
        <p style="margin:0 auto 12px;font-size:.85rem;color:var(--muted)">支持 UTF-8 / UTF-8-BOM / GB18030 编码，自动识别逗号或制表符分隔（上限 50,000 行 / 25 MB；无需任何 SYN- 前缀）</p>
        <input type="file" id="cohort-file-input" accept=".csv,.tsv,.txt,.xlsx" style="max-width:340px">
      </div>

      <h3 style="margin-top:18px">手头没带数据？一键载入真实医院/肿瘤专病表头样表当场演示（也可先下载 CSV 再手动上传）：</h3>
      <div class="sample-box" id="sample-cards-container"></div>

      <label class="control" style="margin-top:14px">
        <span>可选临床背景补充说明（当表头为缩写或匿名编码如 x1, x2 时辅助精准检索目标 RCT）</span>
        <input type="text" id="user-domain-note" placeholder="例如：2型糖尿病合并心衰高危门诊队列，比较达格列净与西格列汀..." maxlength="260">
      </label>
    </fieldset>

    <div id="profile-result-box" class="hidden"></div>
  </div>

  <!-- Path 2: Clinical Idea Input -->
  <div id="subpanel-sim" class="hidden">
    <fieldset>
      <legend>输入临床研究方向 / 立项想法（中文自由文本）</legend>
      <p>点击下方多学科示例可一键填入，也可直接输入您自己的临床对比设想：</p>
      <div class="preset-chips" id="idea-presets-container"></div>
      <label class="control" style="margin-top:12px">
        <span>研究问题与目标人群、干预对比描述</span>
        <div style="margin-top:6px">
          <input type="text" id="idea-input-box" class="idea-input" value="{esc(CLINICAL_IDEA_PRESETS[0]['idea_text'])}" maxlength="1000">
        </div>
      </label>
      <div class="actions">
        <button type="button" id="btn-parse-idea" onclick="parseIdeaAndSearchRCTs()">解析 PICO 并实时检索 ClinicalTrials.gov 目标 RCT →</button>
      </div>
    </fieldset>
    <div id="pico-result-box" class="hidden"></div>
  </div>
</section>

<!-- ==================== STEP 2 ==================== -->
<section id="panel-step-2" class="hidden">
  <fieldset>
    <legend>ClinicalTrials.gov 候选目标随机对照试验（Target RCT）检索与选定</legend>
    <div class="field-grid" style="grid-template-columns:1.6fr 1fr 1fr auto">
      <label class="control">
        <span>检索关键词（支持中英文疾病/药物词，或输入精确登记号如 <code>NCT03036124</code>）</span>
        <input type="text" id="rct-query-input" placeholder="例如：Dapagliflozin Heart Failure 或 NCT03036124">
      </label>
      <label class="control">
        <span>干预策略 (Intervention / 1代表药物)</span>
        <input type="text" id="rct-intr-input" placeholder="如：达格列净 / Dapagliflozin">
      </label>
      <label class="control">
        <span>对照策略 (Comparator / 0代表对照)</span>
        <input type="text" id="rct-comp-input" placeholder="如：西格列汀（研究想法未限定可留空）">
      </label>
      <div style="display:flex;align-items:flex-end">
        <button type="button" onclick="executeRCTSearch()">重新检索并核验双臂</button>
      </div>
    </div>
    <div id="rct-source-banner" style="margin-top:12px"></div>
    <div id="rct-cards-list" class="rct-grid"></div>
    <div id="rct-mismatch-confirm-box" class="mismatch-confirm-box hidden">
      <label style="display:flex;align-items:flex-start;gap:10px;cursor:pointer;font-weight:700">
        <input type="checkbox" id="confirm-arm-mismatch-checkbox" name="confirm_arm_mismatch" value="true" style="width:auto;margin-top:4px" onchange="onToggleMismatchConfirm(this.checked)">
        <span>我已知晓该 RCT 的干预/对照臂与本研究不一致，仅借用其纳排与随访时间窗作为参考框架</span>
      </label>
      <p id="rct-mismatch-detail-text" style="margin:6px 0 0 26px;font-size:.86rem;color:#92400e"></p>
    </div>
    <div class="actions" id="rct-step2-actions">
      <button type="button" id="btn-confirm-selected-rct" onclick="confirmSelectedTargetRCT()">使用所选目标 RCT 进入第 3 步（方案与字段对齐） →</button>
    </div>
  </fieldset>
</section>

<!-- ==================== STEP 3 ==================== -->
<section id="panel-step-3" class="hidden">
  <div id="selected-rct-summary"></div>

  <!-- Step 3A: User Cohort Visual Mapping -->
  <div id="step3-user-mapping" class="hidden">
    <fieldset>
      <legend>1 · 暴露分组列与两臂角色映射（对照臂锚定 Index 0）</legend>
      <div class="field-grid">
        <label class="control"><span>暴露/治疗分组列</span><select id="map-trt-col" onchange="onTrtColChange()"></select></label>
        <label class="control"><span>对照臂取值（Comparator，参考组 0）</span><select id="map-comp-val"></select></label>
        <label class="control"><span>试验臂取值（Active，研究组 1）</span><select id="map-act-val"></select></label>
      </div>
    </fieldset>

    <fieldset>
      <legend>2 · 随访时间列与时间单位自动换算</legend>
      <div class="field-grid">
        <label class="control"><span>随访时长列</span><select id="map-time-col"></select></label>
        <label class="control"><span>原始时间单位（自动换算为标准「天」）</span>
          <select id="map-time-unit">
            <option value="days">天（× 1.0）</option>
            <option value="months">月（× 30.4375 天）</option>
            <option value="years">年（× 365.25 天）</option>
          </select>
        </label>
        <label class="control"><span>研究标题（用于报告页眉）</span><input type="text" id="map-study-title" maxlength="160"></label>
      </div>
    </fieldset>

    <fieldset>
      <legend>3 · 结局状态列与竞争风险事件编码映射（1=主要结局 / 2=竞争风险 / 0=删失存活）</legend>
      <p>系统已自动识别结局类别；若您的数据集无竞争风险事件，可将全部非主要事件映射为「删失/存活 (0)」，引擎将自动优雅退化为标准加权生存估计并给出明确说明。</p>
      <label class="control" style="max-width:360px;margin-bottom:12px"><span>结局状态列</span><select id="map-out-col" onchange="onOutColChange()"></select></label>
      <div id="outcome-values-mapper" style="border:1px solid #cbd5e1;border-radius:6px;background:#f8fafc"></div>
    </fieldset>

    <fieldset>
      <legend>4 · 基线混杂协变量勾选（自动处理多分类 One-Hot 编码、缺失值剔除与零方差保护）</legend>
      <div id="covariate-checkbox-list" class="cov-grid"></div>
    </fieldset>

    <fieldset>
      <legend>5 · 基线纳排筛选规则（在队列原始字段上逐条过滤并生成纳排瀑布图）</legend>
      <div id="eligibility-builder-rows"></div>
      <button type="button" class="secondary" style="margin-top:8px;padding:7px 14px;font-size:.88rem" onclick="addEligibilityRow()">+ 新增一条基线纳排条件</button>
    </fieldset>

    <fieldset>
      <legend>6 · 对标评估时点、倾向加权方式与统计诊断门禁</legend>
      <div class="field-grid">
        <label class="control"><span>评估时点（天，逗号分隔，已按队列有效随访裁剪）</span><input type="text" id="map-horizons" value="90, 180, 365"></label>
        <label class="control"><span>行政随访截尾上限（天，须大于最大评估时点）</span><input type="number" id="map-admin-end" step="any" min="10" value="540"></label>
        <label class="control"><span>倾向得分加权方法</span>
          <select id="map-weight-method">
            <option value="overlap">重叠加权 Overlap Weighting（推荐 · 聚焦临床均势重叠人群 ATO）</option>
            <option value="stabilized_iptw">稳定化逆概率加权 Stabilized IPTW（全人群平均效应 ATE）</option>
          </select>
        </label>
        <label class="control"><span>最大加权 |SMD| 门禁</span><input type="number" id="map-max-smd" step="0.01" min="0.01" value="0.10"></label>
        <label class="control"><span>最小有效样本量比 ESS 门禁</span><input type="number" id="map-min-ess" step="0.01" min="0.05" max="1" value="0.20"></label>
        <label class="control"><span>Bootstrap 重抽样次数（20–300）</span><input type="number" id="map-boot" step="10" min="20" max="300" value="80"></label>
      </div>
    </fieldset>

    <div class="actions">
      <button type="button" id="btn-run-user-tte" onclick="launchUserCohortRun()">🟢 运行自有队列真项目 TTE 分析 →</button>
      <button type="button" class="secondary" onclick="goStep(2)">← 返回重选目标 RCT</button>
      <button type="button" class="danger-btn" onclick="wipeCurrentSession()">🗑️ 一键清除本次会话上传数据</button>
    </div>
  </div>

  <!-- Step 3B: Simulated Cohort Protocol Card & Rehearsal Params -->
  <div id="step3-sim-config" class="hidden">
    <div id="sim-seven-element-card"></div>
    <fieldset>
      <legend>模拟预演队列生成与加权分析参数（零 JSON 可视化配置）</legend>
      <div class="field-grid">
        <label class="control"><span>对标对照臂显示名称（Index 0）</span><input type="text" id="sim-comp-label" maxlength="90"></label>
        <label class="control"><span>对标试验臂显示名称（Index 1）</span><input type="text" id="sim-act-label" maxlength="90"></label>
        <label class="control"><span>模拟队列样本量 N（600–5,000）</span><input type="number" id="sim-cases" min="600" max="5000" step="100" value="1500"></label>
        <label class="control"><span>基线混杂偏倚强度</span>
          <select id="sim-confounding">
            <option value="low">低度基线混杂（接近轻度失衡）</option>
            <option value="medium" selected>中度基线混杂（典型真实世界观察性失衡）</option>
            <option value="high">高度基线混杂（强选择偏倚压力测试）</option>
          </select>
        </label>
        <label class="control"><span>预设真实治疗效应方向（用于检验估计恢复）</span>
          <select id="sim-effect">
            <option value="protective_strong">显著保护效应（预设主要 csHR = 0.65）</option>
            <option value="protective_moderate" selected>中等保护效应（预设主要 csHR = 0.75）</option>
            <option value="null">无效零假设检验（预设主要 csHR = 1.00）</option>
            <option value="competing_divergence">高竞争死亡分歧反例（主要 csHR=1.00，竞争死亡升高）</option>
          </select>
        </label>
        <label class="control"><span>竞争风险事件发生率水平</span>
          <select id="sim-competing-level">
            <option value="low">低竞争死亡风险</option>
            <option value="medium" selected>中等竞争死亡风险</option>
            <option value="high">高竞争死亡风险（高龄/晚期肿瘤场景）</option>
          </select>
        </label>
        <label class="control"><span>随访评估时点（天，源自对标 RCT TimeFrame）</span><input type="text" id="sim-horizons" value="90, 180, 365"></label>
        <label class="control"><span>加权方法</span>
          <select id="sim-weight-method">
            <option value="overlap" selected>重叠加权 Overlap Weighting（ATO）</option>
            <option value="stabilized_iptw">稳定化逆概率加权 Stabilized IPTW（ATE）</option>
          </select>
        </label>
        <label class="control"><span>随机种子 Seed（保障逐字节可复现）</span><input type="number" id="sim-seed" min="0" max="99999999" value="42"></label>
      </div>
    </fieldset>
    <div class="actions">
      <button type="button" id="btn-run-sim-tte" onclick="launchSimulatedRun()">🟣 虚拟生成队列并运行模拟 TTE 预演 →</button>
      <button type="button" class="secondary" onclick="goStep(2)">← 返回重选目标 RCT</button>
    </div>
  </div>
</section>

<!-- ==================== STEP 4 ==================== -->
<section id="panel-step-4" class="hidden">
  <div id="step4-result-container">
    <p>请先完成前序步骤并启动真项目或模拟预演分析。</p>
  </div>
</section>

<details class="compat-drawer print-hide" style="margin-top:38px">
  <summary>查看底层技术详情与经典七要素合成教学基准入口（向后兼容模式）</summary>
  <p>除上方双轨临床研究工作台外，系统完整保留阶段二合成基准教学流水线（<strong>方案 → 运行 → 结果</strong>）：<strong>默认示例</strong>使用固定种子 42 生成 <strong>2,000 行</strong>、<strong>10 列</strong>完全合成竞争风险队列（默认入排筛选后留存 <strong>954 行</strong>，支持<strong>全量 CSV 下载</strong>）。模拟预演与合成教学模式均为<strong>非真实患者</strong>数据，仅供方法学核验。</p>
  <p><a class="secondary-link" href="/protocol/new">载入可运行的合成教学实例</a></p>
</details>

<script id="wb-boot-data" type="application/json">{boot_json}</script>
<script>
const BOOT = JSON.parse(document.getElementById('wb-boot-data').textContent);
const STATE = {{
  mode: 'user_cohort',
  sessionId: null,
  profile: null,
  interpretation: null,
  defaultMapping: null,
  pico: null,
  rctCandidates: [],
  selectedRctIdx: -1,
  selectedRct: null,
  confirmArmMismatch: false,
  outcomeCategoryCache: {{}},
  currentRunId: null,
}};

function showError(msg) {{
  const box = document.getElementById('wb-alert');
  if (!msg) {{ box.classList.add('hidden'); box.textContent = ''; return; }}
  box.textContent = msg;
  box.classList.remove('hidden');
  window.scrollTo({{ top: 0, behavior: 'smooth' }});
}}

function goStep(n) {{
  showError('');
  for (let i = 1; i <= 4; i++) {{
    document.getElementById('panel-step-' + i).classList.toggle('hidden', i !== n);
    document.getElementById('tab-step-' + i).classList.toggle('active', i === n);
  }}
}}

function selectMode(mode) {{
  STATE.mode = mode;
  document.getElementById('card-mode-user').className = 'entry-card' + (mode === 'user_cohort' ? ' selected-user' : '');
  document.getElementById('card-mode-sim').className = 'entry-card' + (mode === 'simulated_cohort' ? ' selected-sim' : '');
  document.getElementById('subpanel-user').classList.toggle('hidden', mode !== 'user_cohort');
  document.getElementById('subpanel-sim').classList.toggle('hidden', mode !== 'simulated_cohort');
}}

function escHtml(s) {{
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

function initBootCards() {{
  const sc = document.getElementById('sample-cards-container');
  sc.innerHTML = BOOT.samples.map(s => `
    <div class="sample-card">
      <strong style="color:var(--accent)">${{escHtml(s.title_zh)}}</strong>
      <p style="font-size:.84rem;color:var(--muted);margin:6px 0 12px">${{escHtml(s.subtitle_zh)}}</p>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button type="button" style="padding:8px 13px;font-size:.86rem" onclick="loadSampleCohort('${{escHtml(s.sample_id)}}')">⚡ 一键载入样表并画像</button>
        <a class="secondary-link" style="padding:7px 11px;font-size:.84rem" href="/api/samples/${{escHtml(s.sample_id)}}/download">⬇ 下载原始 CSV 样表</a>
      </div>
    </div>
  `).join('');

  const ic = document.getElementById('idea-presets-container');
  ic.innerHTML = BOOT.ideas.map((item, idx) => `
    <button type="button" class="preset-chip" onclick="pickIdeaPreset(${{idx}})">
      <span style="display:block;font-size:.75rem;color:#6d28d9">${{escHtml(item.domain_zh)}}</span>
      ${{escHtml(item.title_zh)}}
    </button>
  `).join('');

  const dz = document.getElementById('dropzone');
  const fi = document.getElementById('cohort-file-input');
  dz.addEventListener('dragover', e => {{ e.preventDefault(); dz.classList.add('dragover'); }});
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => {{
    e.preventDefault(); dz.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) uploadCohortFile(e.dataTransfer.files[0]);
  }});
  fi.addEventListener('change', () => {{ if (fi.files && fi.files[0]) uploadCohortFile(fi.files[0]); }});
}}

function pickIdeaPreset(idx) {{
  const item = BOOT.ideas[idx];
  if (item) document.getElementById('idea-input-box').value = item.idea_text;
}}

function updateSessionBar() {{
  const bar = document.getElementById('session-bar');
  const lbl = document.getElementById('session-label');
  if (STATE.sessionId && STATE.profile) {{
    bar.classList.remove('hidden');
    lbl.textContent = `当前隔离会话：${{STATE.profile.filename}} (${{STATE.profile.row_count}} 行)`;
  }} else {{
    bar.classList.add('hidden');
  }}
}}

async function wipeCurrentSession() {{
  if (!STATE.sessionId) return;
  const resp = await fetch('/api/sessions/' + STATE.sessionId, {{ method: 'DELETE' }});
  const data = await resp.json();
  STATE.sessionId = null; STATE.profile = null; STATE.interpretation = null;
  document.getElementById('profile-result-box').classList.add('hidden');
  document.getElementById('profile-result-box').innerHTML = '';
  updateSessionBar();
  goStep(1);
  alert(`已物理清除会话上传与派生文件（共删除 ${{data.purged_files || 0}} 个文件）。`);
}}

async function loadSampleCohort(sampleId) {{
  showError('');
  const note = document.getElementById('user-domain-note').value;
  const resp = await fetch('/api/sessions/upload', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ sample_id: sampleId, user_note: note, session_id: STATE.sessionId }})
  }});
  const data = await resp.json();
  if (!resp.ok) {{ showError(data.error || '载入样表失败'); return; }}
  onCohortProfileLoaded(data);
}}

async function uploadCohortFile(file) {{
  showError('');
  const fd = new FormData();
  fd.append('file', file);
  fd.append('user_note', document.getElementById('user-domain-note').value || '');
  if (STATE.sessionId) fd.append('session_id', STATE.sessionId);
  const resp = await fetch('/api/sessions/upload', {{ method: 'POST', body: fd }});
  const data = await resp.json();
  if (!resp.ok) {{ showError(data.error || '上传解析失败'); return; }}
  onCohortProfileLoaded(data);
}}

function onCohortProfileLoaded(data) {{
  STATE.sessionId = data.session_id;
  STATE.profile = data.profile;
  STATE.interpretation = data.interpretation;
  STATE.defaultMapping = data.default_mapping;
  updateSessionBar();

  const p = data.profile;
  const trtHtml = (p.candidate_treatment_columns || []).slice(0, 2).map(tc =>
    `<div><strong><code>${{escHtml(tc.column)}}</code></strong>：${{(tc.categories||[]).map(c => `${{escHtml(c.value)}} (${{c.count}}例)`).join(' vs ')}}</div>`
  ).join('') || '<div>请在下一步手动指定分组列</div>';

  const timeHtml = (p.candidate_time_columns || []).slice(0, 2).map(tc =>
    `<div><strong><code>${{escHtml(tc.column)}}</code></strong>（推断单位：${{escHtml(tc.inferred_unit)}}）：中位数 ${{tc.median}}，四分位 [${{tc.p25}}, ${{tc.p75}}]，范围 ${{tc.min}}–${{tc.max}}</div>`
  ).join('') || '<div>请在下一步指定随访时间列</div>';

  const outHtml = (p.candidate_outcome_columns || []).slice(0, 2).map(oc =>
    `<div><strong><code>${{escHtml(oc.column)}}</code></strong>：${{(oc.categories||[]).map(c => `${{escHtml(c.value)}} (${{c.count}}例)`).join(' · ')}}</div>`
  ).join('') || '<div>请在下一步指定结局列</div>';

  const covRows = (p.columns || []).map(c => {{
    let summary = '主键标识列（已自动脱敏屏蔽，绝不外发）';
    if (c.inferred_role !== 'identifier') {{
      if (c.numeric_summary) {{
        summary = `均值 ${{c.numeric_summary.mean}} ± ${{c.numeric_summary.sd}}，中位数 ${{c.numeric_summary.median}} [${{c.numeric_summary.min}}, ${{c.numeric_summary.max}}]`;
      }} else if (c.top_categories && c.top_categories.length) {{
        summary = c.top_categories.slice(0, 4).map(x => `${{escHtml(x.value)}} (${{(x.rate*100).toFixed(1)}}%)`).join('，');
      }} else {{
        summary = `去重取值 ${{c.n_unique}} 种`;
      }}
    }}
    return `<tr><td><code>${{escHtml(c.name)}}</code></td><td>${{escHtml(c.inferred_role)}}</td><td>${{(c.missing_rate*100).toFixed(1)}}%</td><td>${{summary}}</td></tr>`;
  }}).join('');

  const box = document.getElementById('profile-result-box');
  box.classList.remove('hidden');
  box.innerHTML = `
    <fieldset style="border-color:#0f766e;background:#f0fdfa">
      <legend>自动隐私安全数据画像（Data Profiling · 0 行个体记录外发）</legend>
      <div class="metric-grid">
        <div><strong>${{p.row_count}} 行</strong><span>总观测样本数</span></div>
        <div><strong>${{p.column_count}} 列</strong><span>数据表字段数</span></div>
        <div><strong><code>${{escHtml((p.input_sha256||'').slice(0,12))}}…</code></strong><span>输入文件 SHA-256</span></div>
        <div><strong>${{escHtml(data.interpretation.assist_mode === 'llm_assisted' ? 'AI 表头辅助' : '确定性规则画像')}}</strong><span>画像解析模式</span></div>
      </div>
      <p><strong>临床领域与干预线索识别：</strong>${{escHtml(data.interpretation.domain_summary_zh)}}（推荐检索词：<code>${{escHtml(data.interpretation.search_query_en)}}</code>）</p>
      <div class="field-grid" style="margin:12px 0">
        <div class="sample-card"><strong>候选暴露/治疗分组列</strong>${{trtHtml}}</div>
        <div class="sample-card"><strong>候选随访时间列</strong>${{timeHtml}}</div>
        <div class="sample-card"><strong>候选结局状态列</strong>${{outHtml}}</div>
      </div>
      <details><summary>展开查看全部 ${{p.column_count}} 个字段的基线统计汇总与缺失率表</summary>
        <div class="scroll"><table><thead><tr><th>字段名</th><th>自动推断角色</th><th>缺失率</th><th>分布汇总（均值±SD / 分类占比）</th></tr></thead><tbody>${{covRows}}</tbody></table></div>
      </details>
      <div class="actions">
        <button type="button" onclick="searchRCTsFromProfile()">下一步：基于队列线索实时检索 ClinicalTrials.gov 目标 RCT →</button>
      </div>
    </fieldset>
  `;
}}

async function searchRCTsFromProfile() {{
  if (!STATE.interpretation) return;
  const note = document.getElementById('user-domain-note').value;
  if (note && STATE.sessionId) {{
    const nr = await fetch(`/api/sessions/${{STATE.sessionId}}/note`, {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{ user_note: note }})
    }});
    if (nr.ok) STATE.interpretation = (await nr.json()).interpretation;
  }}
  document.getElementById('rct-query-input').value = STATE.interpretation.search_query_en || '';
  document.getElementById('rct-intr-input').value = STATE.interpretation.intervention_query_en || '';
  document.getElementById('rct-comp-input').value = STATE.interpretation.comparator_query_en || '';
  goStep(2);
  await executeRCTSearch(
    STATE.interpretation.condition_query_en || '',
    STATE.interpretation.intervention_query_en || '',
    STATE.interpretation.comparator_query_en || ''
  );
}}

async function parseIdeaAndSearchRCTs() {{
  showError('');
  const ideaText = document.getElementById('idea-input-box').value;
  const resp = await fetch('/api/idea/parse', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ idea_text: ideaText }})
  }});
  const data = await resp.json();
  if (!resp.ok) {{ showError(data.error || '解析研究想法失败'); return; }}
  STATE.pico = data.pico;
  const pbox = document.getElementById('pico-result-box');
  pbox.classList.remove('hidden');
  pbox.innerHTML = `
    <fieldset style="border-color:#6d28d9;background:#faf5ff">
      <legend>已提取结构化 PICO 要素</legend>
      <ul>
        <li><strong>目标人群 (P)：</strong>${{escHtml(STATE.pico.population)}}</li>
        <li><strong>试验干预 (I)：</strong>${{escHtml(STATE.pico.intervention)}}</li>
        <li><strong>对照策略 (C)：</strong>${{escHtml(STATE.pico.comparator)}}</li>
        <li><strong>主要终点 (O)：</strong>${{escHtml(STATE.pico.primary_outcome)}}</li>
        <li><strong>潜在竞争风险：</strong>${{escHtml(STATE.pico.competing_event)}}</li>
      </ul>
    </fieldset>
  `;
  document.getElementById('rct-query-input').value = STATE.pico.search_query_en || '';
  document.getElementById('rct-intr-input').value = STATE.pico.intervention_query_en || '';
  document.getElementById('rct-comp-input').value = STATE.pico.comparator_query_en || '';
  goStep(2);
  await executeRCTSearch(
    STATE.pico.condition_query_en || '',
    STATE.pico.intervention_query_en || '',
    STATE.pico.comparator_query_en || ''
  );
}}

function onSelectRCTRadio(idx) {{
  STATE.selectedRctIdx = idx;
  STATE.confirmArmMismatch = false;
  const cb = document.getElementById('confirm-arm-mismatch-checkbox');
  if (cb) cb.checked = false;
  document.querySelectorAll('#rct-cards-list .rct-card').forEach((el, i) => {{
    el.classList.toggle('selected', i === idx);
  }});
  const cand = STATE.rctCandidates[idx];
  const box = document.getElementById('rct-mismatch-confirm-box');
  const detailEl = document.getElementById('rct-mismatch-detail-text');
  if (cand && cand.requires_mismatch_confirmation) {{
    box.classList.remove('hidden');
    detailEl.textContent = `当前研究对照：${{cand.expected_arms_summary || '—'}} ｜ 该试验实际分组：${{cand.trial_arms_summary || '—'}}`;
  }} else {{
    box.classList.add('hidden');
    detailEl.textContent = '';
  }}
}}

function onToggleMismatchConfirm(checked) {{
  STATE.confirmArmMismatch = Boolean(checked);
}}

async function executeRCTSearch(cond='', intr='', comp='') {{
  showError('');
  const q = document.getElementById('rct-query-input').value;
  const intrBox = document.getElementById('rct-intr-input').value.trim();
  const compBox = document.getElementById('rct-comp-input').value.trim();
  const ctx = STATE.mode === 'user_cohort' ? (STATE.interpretation || {{}}) : (STATE.pico || {{}});
  const effIntr = intrBox || intr || ctx.intervention_query_en || '';
  const effComp = compBox || comp || ctx.comparator_query_en || '';
  const effCond = cond || ctx.condition_query_en || '';
  const listEl = document.getElementById('rct-cards-list');
  listEl.innerHTML = '<p>正在检索 ClinicalTrials.gov 公开注册随机对照试验并核验双臂干预对齐度…</p>';
  const resp = await fetch('/api/rcts/search', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{
      query: q,
      condition: effCond,
      intervention: effIntr,
      comparator: effComp,
      session_id: STATE.sessionId || '',
      project_mode: STATE.mode,
      context_summary: ctx
    }})
  }});
  const data = await resp.json();
  if (!resp.ok) {{ listEl.innerHTML = ''; showError(data.error || '检索失败'); return; }}
  STATE.rctCandidates = data.candidates || [];
  STATE.confirmArmMismatch = false;
  const cb = document.getElementById('confirm-arm-mismatch-checkbox');
  if (cb) cb.checked = false;
  document.getElementById('rct-mismatch-confirm-box').classList.add('hidden');

  // Auto-preselect #1 ONLY when it is exact_dual_arm, or single_intervention_matched in idea mode!
  const firstStatus = STATE.rctCandidates[0] && STATE.rctCandidates[0].arm_alignment_status;
  const allowPreselect = firstStatus === 'exact_dual_arm' || (STATE.mode !== 'user_cohort' && ['single_intervention_matched', 'unspecified'].includes(firstStatus));
  STATE.selectedRctIdx = allowPreselect ? 0 : -1;

  const isFallback = data.source === 'offline_curated_catalog_fallback';
  document.getElementById('rct-source-banner').innerHTML = `
    <span class="tag ${{isFallback ? 'tag-fallback' : 'tag-score'}}">${{escHtml(data.source_label_zh || data.source)}}</span>
    <span style="font-size:.85rem;color:var(--muted);margin-left:8px">检索词：<code>${{escHtml(data.query_used)}}</code> · 共返回 ${{STATE.rctCandidates.length}} 项候选目标 RCT</span>
  `;
  listEl.innerHTML = STATE.rctCandidates.map((c, idx) => {{
    const po = (c.primary_outcomes && c.primary_outcomes[0]) || {{ measure: '—', timeFrame: '—' }};
    const reasons = (c.match_reasons || []).map(r => `<li>${{escHtml(r)}}</li>`).join('');
    const elig = ((c.eligibility_summary && c.eligibility_summary.highlights) || []).slice(0, 3).map(h => `<li>${{escHtml(h)}}</li>`).join('');
    const st = c.arm_alignment_status || 'unspecified';
    const rawSc = typeof c.match_score === 'number' ? c.match_score : 0.5;
    const pct = c.match_score_pct != null ? c.match_score_pct : Math.round(rawSc <= 1 ? rawSc * 100 : rawSc);

    let alignCls = 'tag-align-exact';
    let scoreCls = 'tag-score';
    let alignText = c.arm_alignment_label || '双臂干预一致 · 推荐对标';
    if (st === 'class_dual_arm') {{
      alignCls = 'tag-align-class';
      scoreCls = 'tag-score-warn';
      alignText = c.arm_alignment_label || '同机制类别参考 · 需确认';
    }} else if (st === 'single_arm_or_comparator_mismatch' || st === 'unmatched_interventions') {{
      alignCls = 'tag-align-mismatch';
      scoreCls = 'tag-score-mismatch';
      alignText = c.arm_alignment_label || '对照不一致 · 仅供参考';
    }}
    const isChecked = idx === STATE.selectedRctIdx;
    const mismatchBanner = (st === 'single_arm_or_comparator_mismatch' || st === 'unmatched_interventions' || st === 'class_dual_arm')
      ? `<div style="background:#fffbeb;border-left:4px solid #d97706;padding:8px 12px;margin:8px 0;font-size:.86rem;color:#92400e">
           <strong>当前研究对照：</strong>${{escHtml(c.expected_arms_summary || '—')}} &nbsp;|&nbsp;
           <strong>该试验分组：</strong>${{escHtml(c.trial_arms_summary || '—')}}<br>
           <span>${{escHtml(c.arm_alignment_detail || '')}}</span>
         </div>`
      : '';

    return `
      <div class="rct-card ${{isChecked ? 'selected' : ''}}">
        <div class="rct-meta">
          <label style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;font-weight:700">
            <input type="radio" name="target_rct_choice" value="${{idx}}" ${{isChecked ? 'checked' : ''}} style="width:auto" onchange="onSelectRCTRadio(${{idx}})">
            <span>选择</span>
          </label>
          <a href="${{escHtml(c.record_url)}}" target="_blank" rel="noopener"><strong>${{escHtml(c.nct_id)}} ↗</strong></a>
          <span class="tag ${{alignCls}}">${{escHtml(alignText)}}</span>
          <span class="tag ${{scoreCls}}">匹配度 ${{pct}}%</span>
          <span class="tag">${{escHtml((c.phases || []).join('/'))}}</span>
          <span class="tag">${{escHtml(c.overall_status)}}</span>
          ${{c.enrollment_count ? `<span class="tag">入组规模 N=${{c.enrollment_count}}</span>` : ''}}
        </div>
        <h3 style="margin:6px 0 8px">${{escHtml(c.brief_title)}}</h3>
        <p style="margin:4px 0;font-size:.9rem"><strong>试验组 (Active)：</strong>${{escHtml((c.experimental_arms||[]).join('；'))}} &nbsp;|&nbsp; <strong>对照组 (Comparator)：</strong>${{escHtml((c.comparator_arms||[]).join('；'))}}</p>
        ${{mismatchBanner}}
        <p style="margin:4px 0;font-size:.9rem"><strong>主要终点与随访期：</strong>${{escHtml(po.measure)}}（随访窗：<code>${{escHtml(po.timeFrame)}}</code> → 建议评估天数：<code>${{(c.parsed_horizon_days||[]).join(', ')}}</code> 天）</p>
        ${{elig ? `<details style="margin:8px 0;padding:6px 10px"><summary style="font-size:.86rem">查看核心纳排标准要点</summary><ul style="font-size:.85rem;margin:6px 0">${{elig}}</ul></details>` : ''}}
        <div style="background:#f8fafc;padding:8px 12px;border-radius:5px;margin:8px 0;font-size:.86rem"><strong>与当前研究的匹配核验：</strong><ul style="margin:4px 0">${{reasons}}</ul></div>
        <button type="button" class="${{c.requires_mismatch_confirmation ? 'secondary' : ''}}" onclick="chooseTargetRCT(${{idx}})">${{c.requires_mismatch_confirmation ? '设为参考框架（需勾选下方不一致确认）' : `选定 ${{escHtml(c.nct_id)}} 作为对标目标试验 →`}}</button>
      </div>
    `;
  }}).join('');
}}

async function confirmSelectedTargetRCT() {{
  if (STATE.selectedRctIdx < 0 || !STATE.rctCandidates[STATE.selectedRctIdx]) {{
    showError('请先在上方候选列表中勾选一项随机对照试验（若候选均为「对照不一致」，需手动点选并勾选下方不一致知情确认框）。');
    return;
  }}
  await chooseTargetRCT(STATE.selectedRctIdx);
}}

async function chooseTargetRCT(idx) {{
  const rct = STATE.rctCandidates[idx];
  if (!rct) return;
  if (STATE.selectedRctIdx !== idx) {{
    onSelectRCTRadio(idx);
  }}
  if (rct.requires_mismatch_confirmation && !STATE.confirmArmMismatch) {{
    showError(`所选试验 ${{rct.nct_id}} 的干预/对照臂与本研究不一致（${{rct.arm_alignment_label || rct.arm_alignment_status}}）。若仅借用其纳排与随访时间窗，请先勾选列表下方的「我已知晓该 RCT 的干预/对照臂与本研究不一致…」确认框后再继续。`);
    return;
  }}
  const intrOvr = document.getElementById('rct-intr-input').value.trim();
  const compOvr = document.getElementById('rct-comp-input').value.trim();
  const resp = await fetch('/api/rcts/draft', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{
      project_mode: STATE.mode,
      session_id: STATE.sessionId,
      target_rct: rct,
      pico: STATE.pico,
      intervention: intrOvr,
      comparator: compOvr,
      confirm_arm_mismatch: STATE.confirmArmMismatch
    }})
  }});
  const data = await resp.json();
  if (!resp.ok) {{ showError(data.error || '生成方案草稿失败'); return; }}
  STATE.selectedRct = data.target_rct || rct;

  const activeRct = STATE.selectedRct;
  const warnHtml = activeRct.arm_alignment_warning
    ? `<div class="notice arm-mismatch-callout" style="background:#fffbeb;border-left-color:#d97706;color:#92400e"><strong>⚠️ 对照不一致参考声明：</strong>${{escHtml(activeRct.arm_alignment_warning)}}</div>`
    : '';
  document.getElementById('selected-rct-summary').innerHTML = `
    <div class="notice" style="background:#eef6f4;border-left-color:var(--accent)">
      <strong>${{activeRct.arm_alignment_warning ? '当前参考框架 RCT（仅借用纳排与随访窗）：' : '当前对标目标 RCT：'}}</strong><a href="${{escHtml(activeRct.record_url)}}" target="_blank" rel="noopener">${{escHtml(activeRct.nct_id)}} ↗</a> — ${{escHtml(activeRct.brief_title)}}
    </div>
    ${{warnHtml}}
  `;

  if (STATE.mode === 'user_cohort') {{
    document.getElementById('step3-user-mapping').classList.remove('hidden');
    document.getElementById('step3-sim-config').classList.add('hidden');
    populateUserMappingForm(data.mapping);
  }} else {{
    document.getElementById('step3-user-mapping').classList.add('hidden');
    document.getElementById('step3-sim-config').classList.remove('hidden');
    populateSimConfigForm(data.protocol, rct);
  }}
  goStep(3);
}}

function populateUserMappingForm(mapping) {{
  const cols = (STATE.profile && STATE.profile.columns) || [];
  const trtSel = document.getElementById('map-trt-col');
  const timeSel = document.getElementById('map-time-col');
  const outSel = document.getElementById('map-out-col');

  trtSel.innerHTML = cols.filter(c => c.top_categories && c.top_categories.length >= 2).map(c =>
    `<option value="${{escHtml(c.name)}}" ${{c.name === mapping.treatment_column ? 'selected' : ''}}>${{escHtml(c.name)}} (${{c.n_unique}} 类)</option>`
  ).join('');
  timeSel.innerHTML = cols.filter(c => c.dtype_kind === 'numeric').map(c =>
    `<option value="${{escHtml(c.name)}}" ${{c.name === mapping.time_column ? 'selected' : ''}}>${{escHtml(c.name)}}</option>`
  ).join('');
  outSel.innerHTML = cols.filter(c => c.top_categories && c.top_categories.length >= 2).map(c =>
    `<option value="${{escHtml(c.name)}}" ${{c.name === mapping.outcome_column ? 'selected' : ''}}>${{escHtml(c.name)}} (${{c.n_unique}} 类)</option>`
  ).join('');

  document.getElementById('map-time-unit').value = mapping.time_unit || 'days';
  document.getElementById('map-study-title').value = mapping.study_title || '';
  document.getElementById('map-horizons').value = (mapping.horizon_days || [90, 180, 365]).join(', ');
  document.getElementById('map-admin-end').value = mapping.administrative_end_day || 540;
  document.getElementById('map-weight-method').value = mapping.weight_method || 'overlap';

  onTrtColChange(mapping.comparator_value, mapping.active_value);
  onOutColChange(mapping.primary_event_values, mapping.competing_event_values, mapping.censor_values);
  renderCovariateCheckboxes(mapping.covariate_columns || []);

  const eb = document.getElementById('eligibility-builder-rows');
  eb.innerHTML = '';
  (mapping.eligibility_criteria || []).forEach(crit => addEligibilityRow(crit));
}}

function onTrtColChange(prefComp, prefAct) {{
  const colName = document.getElementById('map-trt-col').value;
  const colObj = ((STATE.profile && STATE.profile.columns) || []).find(c => c.name === colName);
  const cats = (colObj && colObj.top_categories) || [];
  const compSel = document.getElementById('map-comp-val');
  const actSel = document.getElementById('map-act-val');
  const defComp = prefComp || (cats[0] && cats[0].value) || '';
  const defAct = prefAct || (cats[1] && cats[1].value) || defComp;
  compSel.innerHTML = cats.map(c => `<option value="${{escHtml(c.value)}}" ${{c.value === defComp ? 'selected' : ''}}>${{escHtml(c.value)}} (${{c.count}} 例)</option>`).join('');
  actSel.innerHTML = cats.map(c => `<option value="${{escHtml(c.value)}}" ${{c.value === defAct ? 'selected' : ''}}>${{escHtml(c.value)}} (${{c.count}} 例)</option>`).join('');
  renderCovariateCheckboxes(getSelectedCovariates());
}}

function onOutColChange(primList, compList, censList) {{
  const colName = document.getElementById('map-out-col').value;
  const colObj = ((STATE.profile && STATE.profile.columns) || []).find(c => c.name === colName);
  const cats = (colObj && colObj.top_categories) || [];
  const box = document.getElementById('outcome-values-mapper');
  box.innerHTML = `<div class="outcome-chip-row" style="font-weight:700;background:#eef2f6"><div>原始结局列取值（例数）</div><div>主要结局事件 (1)</div><div>竞争风险事件 (2)</div><div>删失 / 存活 (0)</div></div>` +
    cats.map((c, idx) => {{
      const v = c.value;
      let role = '0';
      if (Array.isArray(primList) && primList.includes(v)) role = '1';
      else if (Array.isArray(compList) && compList.includes(v)) role = '2';
      else if (!primList && idx === 0) role = '1';
      return `
        <div class="outcome-chip-row" data-out-val="${{escHtml(v)}}">
          <div><code>${{escHtml(v)}}</code> <small style="color:var(--muted)">(${{c.count}} 例)</small></div>
          <label><input type="radio" name="out_role_${{idx}}" value="1" ${{role==='1'?'checked':''}} style="width:auto"> 主要事件 (1)</label>
          <label><input type="radio" name="out_role_${{idx}}" value="2" ${{role==='2'?'checked':''}} style="width:auto"> 竞争事件 (2)</label>
          <label><input type="radio" name="out_role_${{idx}}" value="0" ${{role==='0'?'checked':''}} style="width:auto"> 删失/存活 (0)</label>
        </div>
      `;
    }}).join('');
  renderCovariateCheckboxes(getSelectedCovariates());
}}

function getSelectedCovariates() {{
  return Array.from(document.querySelectorAll('.cov-check:checked')).map(el => el.value);
}}

function renderCovariateCheckboxes(selectedNames) {{
  const trtCol = document.getElementById('map-trt-col').value;
  const timeCol = document.getElementById('map-time-col').value;
  const outCol = document.getElementById('map-out-col').value;
  const idCol = STATE.profile && STATE.profile.id_column_guess;
  const cols = ((STATE.profile && STATE.profile.columns) || []).filter(c =>
    ![trtCol, timeCol, outCol, idCol].includes(c.name) && c.inferred_role !== 'identifier' && c.inferred_role !== 'constant_or_unusable'
  );
  const selSet = new Set(selectedNames && selectedNames.length ? selectedNames : cols.map(c => c.name));
  document.getElementById('covariate-checkbox-list').innerHTML = cols.map(c => {{
    const badge = c.dtype_kind === 'numeric' ? (c.inferred_role === 'binary_covariate' ? '二元 0/1' : '连续数值') : `多分类 (${{c.n_unique}}类 One-Hot)`;
    return `
      <label class="cov-item">
        <input type="checkbox" class="cov-check" value="${{escHtml(c.name)}}" ${{selSet.has(c.name) ? 'checked' : ''}}>
        <span><strong><code>${{escHtml(c.name)}}</code></strong><br><small style="color:var(--muted)">${{badge}} · 缺失率 ${{(c.missing_rate*100).toFixed(1)}}%</small></span>
      </label>
    `;
  }}).join('');
}}

function addEligibilityRow(crit) {{
  const cols = ((STATE.profile && STATE.profile.columns) || []).filter(c => c.inferred_role !== 'identifier');
  const c = crit || {{ kind: 'inclusion', variable: (cols[0] && cols[0].name) || '', operator: '>=', value: '' }};
  const row = document.createElement('div');
  row.className = 'elig-row';
  row.innerHTML = `
    <select class="el-kind">
      <option value="inclusion" ${{c.kind==='inclusion'?'selected':''}}>纳入 (Include)</option>
      <option value="exclusion" ${{c.kind==='exclusion'?'selected':''}}>排除 (Exclude)</option>
    </select>
    <select class="el-var">${{cols.map(col => `<option value="${{escHtml(col.name)}}" ${{col.name===c.variable?'selected':''}}>${{escHtml(col.name)}}</option>`).join('')}}</select>
    <select class="el-op">${{['>=','<=','>','<','==','!='].map(op => `<option value="${{op}}" ${{op===c.operator?'selected':''}}>${{op}}</option>`).join('')}}</select>
    <input type="text" class="el-val" value="${{escHtml(c.value)}}" placeholder="阈值">
    <button type="button" class="secondary" style="padding:7px 10px" onclick="this.parentElement.remove()">删除</button>
  `;
  document.getElementById('eligibility-builder-rows').appendChild(row);
}}

function populateSimConfigForm(proto, rct) {{
  const s0 = proto.treatment_strategies[0];
  const s1 = proto.treatment_strategies[1];
  document.getElementById('sim-comp-label').value = s0.label;
  document.getElementById('sim-act-label').value = s1.label;
  document.getElementById('sim-horizons').value = (proto.followup.horizon_days || [90, 180, 365]).join(', ');
  const po = (rct.primary_outcomes && rct.primary_outcomes[0]) || {{ measure: '主要临床结局', timeFrame: '12 months' }};
  document.getElementById('sim-seven-element-card').innerHTML = `
    <fieldset style="border-color:#6d28d9;background:#faf5ff">
      <legend>已自动转译的对标目标试验七要素方案卡片（中文可读版）</legend>
      <ol style="margin:6px 0;line-height:1.75">
        <li><strong>入排标准 (Eligibility)：</strong>对标 ${{escHtml(rct.nct_id)}} 成人准入人群，基线第 0 天完成评估筛选；</li>
        <li><strong>时间零点 (Time Zero)：</strong>合格患者首次起始对标治疗策略当日记为第 0 天（杜绝永生时间偏倚）；</li>
        <li><strong>治疗策略 (Strategies)：</strong>对照臂「${{escHtml(s0.label)}}」vs 试验臂「${{escHtml(s1.label)}}」；</li>
        <li><strong>分配模式 (Assignment)：</strong>观察性基线混杂分配，采用倾向得分重叠加权 / 稳定化 IPTW 平衡基线特征；</li>
        <li><strong>随访窗口 (Follow-up)：</strong>解析自对标终点时间窗（<code>${{escHtml(po.timeFrame)}}</code>），在 ${{(proto.followup.horizon_days||[]).join('、')}} 天评估累计发生率；</li>
        <li><strong>结局定义 (Outcomes)：</strong>主要终点「${{escHtml(po.measure)}}」（编码 1）与非目标疾病竞争死亡（编码 2）双事件显式建模；</li>
        <li><strong>目标估计量与门禁 (Estimand & Gates)：</strong>并列估计 CIF、RD(t)、RR_CIF(t)、Fine–Gray sHR、原因特异 csHR 及 E-value，受正值性/ESS/|SMD| 三重门禁保护。</li>
      </ol>
    </fieldset>
  `;
}}

async function launchUserCohortRun() {{
  showError('');
  const primVals = [], compVals = [], censVals = [];
  document.querySelectorAll('#outcome-values-mapper [data-out-val]').forEach(row => {{
    const val = row.getAttribute('data-out-val');
    const checked = row.querySelector('input[type="radio"]:checked');
    const role = checked ? checked.value : '0';
    if (role === '1') primVals.push(val);
    else if (role === '2') compVals.push(val);
    else censVals.push(val);
  }});
  const horizons = document.getElementById('map-horizons').value.split(',').map(x => parseFloat(x.trim())).filter(x => Number.isFinite(x) && x > 0);
  const elig = [];
  document.querySelectorAll('#eligibility-builder-rows .elig-row').forEach((r, i) => {{
    const v = r.querySelector('.el-val').value.trim();
    if (!v) return;
    const numV = Number(v);
    elig.push({{
      id: `user_elig_${{i+1}}`,
      kind: r.querySelector('.el-kind').value,
      variable: r.querySelector('.el-var').value,
      operator: r.querySelector('.el-op').value,
      value: Number.isFinite(numV) && v !== '' ? numV : v
    }});
  }});

  const mapping = {{
    id_column: (STATE.profile && STATE.profile.id_column_guess) || '',
    study_title: document.getElementById('map-study-title').value.trim(),
    treatment_column: document.getElementById('map-trt-col').value,
    comparator_value: document.getElementById('map-comp-val').value,
    active_value: document.getElementById('map-act-val').value,
    time_column: document.getElementById('map-time-col').value,
    time_unit: document.getElementById('map-time-unit').value,
    outcome_column: document.getElementById('map-out-col').value,
    primary_event_values: primVals,
    competing_event_values: compVals,
    censor_values: censVals,
    covariate_columns: getSelectedCovariates(),
    eligibility_criteria: elig,
    horizon_days: horizons,
    administrative_end_day: parseFloat(document.getElementById('map-admin-end').value) || 540,
    weight_method: document.getElementById('map-weight-method').value,
    max_abs_smd: parseFloat(document.getElementById('map-max-smd').value) || 0.10,
    min_ess_ratio: parseFloat(document.getElementById('map-min-ess').value) || 0.20,
    positivity_epsilon: 0.01,
    bootstrap_iterations: parseInt(document.getElementById('map-boot').value, 10) || 80,
    seed: 42
  }};

  const resp = await fetch('/api/dual-runs', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{
      project_mode: 'user_cohort',
      session_id: STATE.sessionId,
      mapping: mapping,
      target_rct: STATE.selectedRct,
      intervention: document.getElementById('rct-intr-input').value.trim(),
      comparator: document.getElementById('rct-comp-input').value.trim(),
      confirm_arm_mismatch: STATE.confirmArmMismatch
    }})
  }});
  const data = await resp.json();
  if (!resp.ok) {{ showError(data.error || '启动真项目 TTE 分析失败'); return; }}
  pollDualRun(data.run_id);
}}

async function launchSimulatedRun() {{
  showError('');
  const horizons = document.getElementById('sim-horizons').value.split(',').map(x => parseFloat(x.trim())).filter(x => Number.isFinite(x) && x > 0);
  const simConfig = {{
    comparator_label: document.getElementById('sim-comp-label').value.trim(),
    active_label: document.getElementById('sim-act-label').value.trim(),
    cases: parseInt(document.getElementById('sim-cases').value, 10) || 1500,
    confounding_strength: document.getElementById('sim-confounding').value,
    effect_direction: document.getElementById('sim-effect').value,
    competing_rate_level: document.getElementById('sim-competing-level').value,
    horizon_days: horizons,
    weight_method: document.getElementById('sim-weight-method').value,
    seed: parseInt(document.getElementById('sim-seed').value, 10) || 42,
    bootstrap_iterations: 80
  }};
  const resp = await fetch('/api/dual-runs', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{
      project_mode: 'simulated_cohort',
      target_rct: STATE.selectedRct,
      pico: STATE.pico,
      sim_config: simConfig,
      intervention: document.getElementById('rct-intr-input').value.trim(),
      comparator: document.getElementById('rct-comp-input').value.trim(),
      confirm_arm_mismatch: STATE.confirmArmMismatch
    }})
  }});
  const data = await resp.json();
  if (!resp.ok) {{ showError(data.error || '启动模拟预演失败'); return; }}
  pollDualRun(data.run_id);
}}

async function pollDualRun(runId) {{
  STATE.currentRunId = runId;
  goStep(4);
  const box = document.getElementById('step4-result-container');
  box.innerHTML = `<div class="notice"><strong>正在运行确定性 TTE 门禁与竞争风险生存估计（运行号 <code>${{escHtml(runId)}}</code>）…</strong><p>正在执行纳排漏斗、极大似然倾向加权、正值性/ESS/|SMD| 门禁核查、加权 Aalen–Johansen CIF 与 Fine–Gray sHR / 原因特异 Cox 估计，请稍候约 1–3 秒。</p></div>`;
  for (let i = 0; i < 180; i++) {{
    const r = await fetch(`/api/dual-runs/${{runId}}`);
    const d = await r.json();
    if (d.state !== 'queued' && d.state !== 'running') {{
      box.innerHTML = d.rendered_html || `<p>已完成：<a href="/runs/${{escHtml(runId)}}">打开完整结果页</a></p>`;
      return;
    }}
    await new Promise(res => setTimeout(res, 400));
  }}
  box.innerHTML = `<p>分析仍在后台执行，请点击查看：<a href="/runs/${{escHtml(runId)}}">/runs/${{escHtml(runId)}}</a></p>`;
}}

initBootCards();
</script>
"""
