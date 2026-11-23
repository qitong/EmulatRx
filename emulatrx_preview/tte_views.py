"""Small server-rendered views for the synthetic TTE protocol and results."""

import csv
import html
import json
from pathlib import Path


ELEMENTS = (
    ("eligibility", "1 · 纳入与排除", "基线前条件；时间偏移必须不晚于 time zero。"),
    ("time_zero", "2 · Time zero", "入组锚点与 landmark 天数。"),
    ("treatment_strategies", "3 · 治疗策略", "合成对照臂须在第一位；当前预览为两臂。"),
    ("assignment_mode", "4 · 策略分配", "基线观察性分配或显式 landmark。"),
    ("followup", "5 · 随访", "指定评估时点和行政结束日。"),
    ("outcomes", "6 · 结局", "目标事件、竞争事件与删失编码。"),
    ("estimand", "7 · 目标估计量与诊断门禁", "门禁阈值为方法学基准参数，不是临床阈值。"),
)


def esc(value):
    return html.escape(str(value), quote=True)


def protocol_summary(protocol):
    criteria = protocol["eligibility"]
    rows = []
    for kind in ("inclusion_criteria", "exclusion_criteria"):
        rows.extend({"kind": "纳入" if kind == "inclusion_criteria" else "排除", **item} for item in criteria[kind])
    strategies = protocol["treatment_strategies"]
    gates = protocol["estimand"]["diagnostic_gates"]
    return ("<div class='protocol-summary'>"
            "<h3>1 · 纳入与排除</h3>" + table(rows, ["kind", "variable", "operator", "value", "assessed_day_offset"]) +
            f"<h3>2 · Time zero</h3><p>锚点 {esc(protocol['time_zero']['anchor_event'])}；landmark 第 {esc(protocol['time_zero']['landmark_day'])} 天。</p>"
            "<h3>3 · 治疗策略</h3>" + table(strategies, ["id", "label", "role"]) +
            f"<h3>4 · 分配模式</h3><p>{esc(protocol['assignment_mode'])}</p>"
            f"<h3>5 · 随访</h3><p>评估时点 {esc(', '.join(str(x) for x in protocol['followup']['horizon_days']))} 天；行政结束第 {esc(protocol['followup']['administrative_end_day'])} 天。</p>"
            f"<h3>6 · 结局</h3><p>目标事件编码 {esc(protocol['outcomes']['primary_event_code'])}；竞争事件编码 {esc(protocol['outcomes']['competing_event_codes'])}；删失编码 {esc(protocol['outcomes']['censor_code'])}。</p>"
            f"<h3>7 · 目标估计量</h3><p>{esc(protocol['estimand']['target_estimand'])}；结果指标 {esc(', '.join(protocol['estimand']['effect_measures']))}。</p>"
            f"<p>方案中的方法学门禁：最大 |SMD| {esc(gates['max_abs_smd'])}、最小 ESS 比 {esc(gates['min_ess_ratio'])}、倾向下界 {esc(gates['positivity_epsilon'])}。</p></div>")


def form(default_protocol, values=None, normalized=None, error=None):
    values = values or {}
    parts = ["<p>编辑七要素方案，先校验并查看规范化方案，再运行完全合成 TTE。每个区域使用 JSON；页面不会接收患者文件。</p>"]
    if error:
        parts.append(f"<div class='error'><strong>校验失败：</strong>{esc(error)}</div>")
    if normalized:
        parts.append("<h2>规范化方案</h2><p>下列内容已通过 PR #14 契约的结构与时间规则校验；统计门禁在合成队列生成后检查。</p>")
        parts.append(protocol_summary(normalized))
        parts.append(f"<pre>{esc(json.dumps(normalized, ensure_ascii=False, indent=2))}</pre>")
    parts.append("<form method='post' action='/validate'><fieldset><legend>方案标题</legend>")
    title = values.get("study_title", default_protocol["study_title"])
    parts.append(f"<input name='study_title' maxlength='160' value='{esc(title)}' required></fieldset>")
    for key, label, hint in ELEMENTS:
        value = values.get(key, json.dumps(default_protocol[key], ensure_ascii=False, indent=2))
        parts.append(f"<fieldset><legend>{esc(label)}</legend><p>{esc(hint)}</p>"
                     f"<textarea name='{key}' rows='{7 if key in ('eligibility','treatment_strategies','estimand') else 4}' spellcheck='false' required>{esc(value)}</textarea></fieldset>")
    parts.append("<fieldset><legend>运行设置</legend><p>固定的完全合成数据生成机制；种子决定可复现队列。</p>")
    for key, label, default, lo, hi in (("cases", "合成队列行数", 2000, 100, 5000),
                                        ("seed", "随机种子", 42, 0, 4294967295),
                                        ("bootstrap_iterations", "Bootstrap 次数", 100, 20, 300)):
        value = values.get(key, str(default))
        parts.append(f"<label>{esc(label)} <input type='number' name='{key}' min='{lo}' max='{hi}' value='{esc(value)}' required></label>")
    parts.append("</fieldset><div class='actions'><button type='submit'>1. 校验方案</button> "
                 "<button type='submit' formaction='/runs'>2. 运行合成 TTE</button></div></form>")
    return "".join(parts)


def table(rows, columns, labels=None):
    if not rows:
        return "<p>无记录。</p>"
    labels = labels or {}
    head = "".join(f"<th>{esc(labels.get(key, key))}</th>" for key in columns)
    body = "".join("<tr>" + "".join(f"<td>{esc(row.get(key, ''))}</td>" for key in columns) + "</tr>" for row in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _fmt(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def curves_svg(curve_paths, column, title):
    series = []
    for path in curve_paths:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        points = [(float(row["time_day"]), float(row[column])) for row in rows]
        stride = max(1, len(points) // 150)
        points = points[::stride] + ([points[-1]] if points[-1] not in points[::stride] else [])
        series.append((rows[0]["treatment_arm"], points))
    if not series:
        return ""
    max_day = max(point[0] for _, points in series for point in points) or 1.0
    colors = ("#1769aa", "#d3672e", "#7d4aa0", "#288d69")
    lines = []
    legend = []
    for index, (name, points) in enumerate(series):
        path_data = " ".join(f"{40 + 620*x/max_day:.1f},{240 - 210*y:.1f}" for x, y in points)
        color = colors[index % len(colors)]
        lines.append(f"<polyline points='{path_data}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        legend.append(f"<span style='color:{color}'>●</span> {esc(name)}")
    return (f"<figure><figcaption>{esc(title)}</figcaption><svg viewBox='0 0 690 275' role='img' aria-label='{esc(title)}'>"
            "<path d='M40 20 V240 H665' fill='none' stroke='#64748b'/>"
            "<text x='4' y='32'>1.0</text><text x='8' y='242'>0</text>"
            f"<text x='615' y='262'>{max_day:.0f} 天</text>{''.join(lines)}</svg>"
            f"<div class='legend'>{' · '.join(legend)}</div></figure>")


def propensity_histogram(weighted_path, active_id):
    bins = [0] * 10
    with weighted_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            value = float(row[f"ps_{active_id}"])
            bins[min(9, max(0, int(value * 10)))] += 1
    maximum = max(bins) or 1
    rows = []
    for index, count in enumerate(bins):
        width = 100 * count / maximum
        rows.append(f"<tr><td>{index/10:.1f}–{(index+1)/10:.1f}</td><td><div class='bar' style='width:{width:.1f}%'></div></td><td>{count}</td></tr>")
    return "<div class='scroll'><table><tr><th>倾向区间</th><th>人数分布</th><th>行数</th></tr>" + "".join(rows) + "</table></div>"


def result_view(result, root):
    state = result["state"]
    if state in ("queued", "running"):
        return f"<p>运行编号 <code>{esc(result['run_id'])}</code> · 正在生成队列、诊断和估计。页面会自动刷新。</p>"
    if state in ("failed", "interrupted"):
        return f"<div class='error'>运行未完成：{esc(result.get('error', state))}</div><p><a href='/'>返回方案</a></p>"
    run_id = result["run_id"]
    root = Path(root) / run_id
    base = f"/runs/{run_id}/artifacts/"
    gate = result["gate_status"]
    diagnostic = result["diagnostics"]
    parts = [f"<p>运行编号 <code>{run_id}</code> · 生成完全合成队列 <strong>{result['cohort_rows']} 行</strong> · 方案筛选后 <strong>{result['eligible_rows']} 行</strong> · 诊断门禁 <strong>{esc(gate)}</strong></p>",
             f"<p>变量列（{len(result['cohort_columns'])} 列）：{esc(', '.join(result['cohort_columns']))}</p>",
             f"<p><a href='{base}cohort/cohort.csv'>下载完整生成队列 CSV</a> · <a href='{base}screen/eligible_cohort.csv'>下载完整筛选后队列 CSV</a> · <a href='{base}preview_manifest.json'>下载预览清单</a> · <a href='{base}cohort/run_manifest.json'>下载生成清单</a></p>",
             "<h2>1 · 规范化七要素方案</h2>",
             protocol_summary(result["normalized_protocol"]),
             f"<pre>{esc(json.dumps(result['normalized_protocol'], ensure_ascii=False, indent=2))}</pre>",
             f"<p><a href='{base}protocol_check/normalized_protocol.json'>下载规范化方案</a></p>",
             "<h2>2 · 合成数据预览</h2>",
             table(result["sample_cases"], result["cohort_columns"]),
             "<h2>3 · 筛选流失</h2>",
             table(result["attrition"], ["step_id", "remaining_n", "excluded_n", "excluded_primary_events", "excluded_competing_events", "arm_counts_json"]),
             "<h2>4 · 诊断与门禁</h2>",
             f"<p>未加权最大 |SMD|：{_fmt(diagnostic.get('max_abs_smd_unweighted'))}；加权最大 |SMD|：{_fmt(diagnostic.get('max_abs_smd_weighted'))}；最小 ESS 比：{_fmt(diagnostic.get('ess_diagnostics', {}).get('min_ess_ratio_observed'))}。</p>",
             f"<p>倾向分数范围：{_fmt(diagnostic.get('propensity_diagnostics', {}).get('min_propensity'))}–{_fmt(diagnostic.get('propensity_diagnostics', {}).get('max_propensity'))}。门禁阈值来自方案，不代表临床阈值。</p>"]
    ess = diagnostic.get("ess_diagnostics", {}).get("by_arm", {})
    parts.append(table([{"arm": arm, **stats} for arm, stats in ess.items()], ["arm", "n", "ess", "ess_ratio", "raw_weight_min", "raw_weight_max"]))
    parts.append("<h3>协变量均衡</h3>" + table(result["balance"], ["covariate", "abs_smd_unweighted", "abs_smd_weighted"]))
    if gate == "blocked":
        parts.append("<div class='error'><strong>诊断门禁未通过。估计结果与曲线不生成。</strong>"
                     f"<pre>{esc(json.dumps(diagnostic.get('gate_failures', []), ensure_ascii=False, indent=2))}</pre></div>")
        parts.append(f"<p><a href='{base}diagnostics/diagnostics_summary.json'>下载诊断</a> · <a href='{base}diagnostics/balance_diagnostics.csv'>下载均衡表</a> · <a href='{base}screen/eligibility_attrition.csv'>下载筛选流失表</a></p>")
    else:
        active_id = result["normalized_protocol"]["treatment_strategies"][1]["id"]
        parts.append("<h3>倾向分数分布</h3>" + propensity_histogram(root / "diagnostics" / "weighted_cohort.csv", active_id))
        curves = sorted((root / "estimates" / "curves").glob("*.csv"))
        parts.append("<h2>5 · 目标事件与竞争事件累计发生率</h2>")
        parts.append(curves_svg(curves, "cif_primary", "目标事件 CIF") + curves_svg(curves, "cif_competing", "竞争事件 CIF"))
        comparison = [{key: _fmt(value) for key, value in row.items()} for row in result["estimates"]["co_presented_comparison_table"]]
        parts.append("<h2>6 · 指定时点对比估计</h2>")
        parts.append(table(comparison, ["horizon_day", "cif_primary_ref", "cif_primary_active", "cif_competing_ref", "cif_competing_active",
                                        "rd_primary", "rd_primary_ci_lower", "rd_primary_ci_upper", "rr_cif_primary", "rr_cif_primary_ci_lower", "rr_cif_primary_ci_upper",
                                        "rd_competing", "rd_competing_ci_lower", "rd_competing_ci_upper",
                                        "shr_primary", "shr_primary_ci_lower", "shr_primary_ci_upper",
                                        "cshr_primary", "cshr_primary_ci_lower", "cshr_primary_ci_upper",
                                        "cshr_competing", "cshr_competing_ci_lower", "cshr_competing_ci_upper",
                                        "evalue_rr_cif_point", "evalue_rr_cif_ci_limit"],
                           {"horizon_day": "天", "rd_primary": "RD 目标", "rr_cif_primary": "RR(CIF) 目标", "rd_competing": "RD 竞争",
                            "shr_primary": "sHR 目标", "cshr_primary": "csHR 目标", "cshr_competing": "csHR 竞争", "evalue_rr_cif_point": "E-value"}))
        parts.append("<p>sHR 与 csHR 描述不同的风险集；所有数值仅来自合成基准，不能解释为真实治疗效果。Bootstrap 区间固定已估计的基线倾向权重，不包含重新拟合倾向模型的不确定性。</p>")
        parts.append("<h3>方法限制与警示</h3>" + f"<pre>{esc(json.dumps(result['estimates']['methodological_warnings'], ensure_ascii=False, indent=2))}</pre>")
        report = (root / "estimates" / "audit_report.md").read_text(encoding="utf-8")
        parts.append(f"<h2>7 · 完整审计报告</h2><p><a href='{base}estimates/audit_report.md'>下载 Markdown 报告</a> · <a href='{base}estimates/estimates.json'>下载完整估计 JSON</a> · <a href='{base}estimates/run_manifest.json'>下载估计清单</a></p><pre>{esc(report)}</pre>")
    parts.append(f"<h2>产物与清单</h2><p>可下载产物 {len(result['available_artifacts'])} 项；审计提交 {esc(result['preview_manifest']['code_commit'])}。</p>")
    parts.append("<p><a href='/'>新建方案</a></p>")
    return "".join(parts)
