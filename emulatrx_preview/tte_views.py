"""Small server-rendered views for the synthetic TTE protocol and results."""

import csv
import html
from itertools import islice
import json
from pathlib import Path

from .web_protocol import default_fields, FIELD_NAMES, OPERATORS, VARIABLE_LABELS


def esc(value):
    return html.escape(str(value), quote=True)


def protocol_summary(protocol):
    criteria = protocol["eligibility"]
    rows = []
    for kind in ("inclusion_criteria", "exclusion_criteria"):
        rows.extend({"kind": "纳入" if kind == "inclusion_criteria" else "排除",
                     "variable": VARIABLE_LABELS.get(item["variable"], item["variable"]),
                     "operator": OPERATORS.get(item["operator"], item["operator"]),
                     "value": item["value"], "day": item["assessed_day_offset"]}
                    for item in criteria[kind])
    strategies = protocol["treatment_strategies"]
    gates = protocol["estimand"]["diagnostic_gates"]
    strategies_readable = [{"role": "对照策略" if item["role"] == "control" else "比较策略",
                            "label": item["label"]} for item in strategies]
    return ("<div class='protocol-summary'>"
            f"<p><strong>研究问题：</strong>{esc(protocol['study_title'])}</p>"
            "<h3>1 · 目标人群与筛选</h3>" + table(rows, ["kind", "variable", "operator", "value", "day"],
                                               {"kind": "条件", "variable": "合成变量", "operator": "判断", "value": "阈值", "day": "评估日"}) +
            f"<h3>2 · 时间零点</h3><p>以合格策略首次启动为第 0 天；本教学实例在第 {esc(protocol['time_zero']['landmark_day'])} 天建立风险集。</p>"
            "<h3>3 · 比较策略</h3>" + table(strategies_readable, ["role", "label"], {"role": "角色", "label": "名称"}) +
            "<h3>4 · 策略分配</h3><p>基线观察性分配；通过合成协变量估计倾向分数，使用重叠加权。</p>"
            f"<h3>5 · 随访</h3><p>在 {esc('、'.join(str(x) for x in protocol['followup']['horizon_days']))} 天对比；"
            f"行政随访到第 {esc(protocol['followup']['administrative_end_day'])} 天。</p>"
            "<h3>6 · 结局</h3><p>目标事件（编码 1）与竞争事件（编码 2）分别计入累计发生率；编码 0 表示删失。</p>"
            "<h3>7 · 估计与门禁</h3><p>目标人群为重叠人群（ATO）；并列报告 RD、RR(CIF)、sHR 与 csHR。"
            f"教学诊断门禁：最大 |SMD| {esc(gates['max_abs_smd'])}、最小 ESS 比 {esc(gates['min_ess_ratio'])}、"
            f"正性下界 {esc(gates['positivity_epsilon'])}；这些不是临床阈值。</p></div>")


def home_view():
    return ("<section class='hero'><p class='eyebrow'>研究方法教学 · 私网预览</p>"
            "<h2>从研究方案到竞争风险报告</h2>"
            "<p>用一份可读的七要素方案，生成完全合成队列，检查入排筛选、加权诊断与门禁，"
            "再阅读目标事件和竞争事件的估计及审计报告。</p>"
            "<p class='hero-warning'>完全合成教学数据，非患者资料或临床疗效证据。</p>"
            "<p><a class='primary-link' href='/protocol/new'>载入可运行的合成教学实例</a></p></section>"
            "<section class='flow-intro'><h2>方案 → 运行 → 结果</h2><p>默认示例使用固定种子 42；修改条件后人数会变化。</p>"
            "<div class='metric-grid'><div><strong>2,000 行</strong><span>默认生成队列</span></div>"
            "<div><strong>10 列</strong><span>合成结局与基线变量</span></div>"
            "<div><strong>954 行</strong><span>默认筛选后，随编辑而变化</span></div>"
            "<div><strong>全量 CSV 下载</strong><span>运行后可导出两份队列</span></div></div>"
            "<p>先建立并确认方案，再运行；诊断门禁通过才显示估计。"
            "结果页按方案、筛选、加权诊断、竞争风险估计和可打印报告的顺序呈现。</p></section>")


def form(default_protocol, values=None, errors=None):
    fields = {**default_fields(default_protocol), **(values or {})}
    errors = errors or {}

    def error_for(name):
        return f"<p class='field-error' role='alert'>{esc(errors[name])}</p>" if name in errors else ""

    def input_field(name, label, hint, kind="text", attributes=""):
        return (f"<label class='control'><span>{esc(label)}</span><small>{esc(hint)}</small>"
                f"<input id='{name}' name='{name}' type='{kind}' value='{esc(fields[name])}' {attributes} required></label>"
                + error_for(name))

    def select_field(name, label, choices):
        options = "".join(f"<option value='{esc(value)}'{' selected' if fields[name] == value else ''}>{esc(text)}</option>"
                          for value, text in choices.items())
        return (f"<label class='control'><span>{esc(label)}</span><select id='{name}' name='{name}' required>{options}</select></label>"
                + error_for(name))

    parts = ["<p class='eyebrow'>步骤 1 / 3 · 建立研究方案</p><h2>合成 TTE 教学方案</h2>"
             "<p>下列设置已载入可运行的教学示例。改动后先看方案摘要，再确认运行。"
             "输入只用于生成完全合成数据，不能上传患者资料。</p>"]
    if errors:
        parts.append("<div class='error' role='alert'><strong>请检查标出的字段。</strong>修改后可重新生成方案摘要。</div>")
    parts.append("<form id='plan-editor' method='post' action='/validate'>")
    parts.append("<fieldset><legend>研究问题</legend>" + input_field("study_title", "方案标题", "用于标识本次教学分析；不输入患者信息。", attributes="maxlength='160'") + "</fieldset>")
    for prefix, heading, explanation in (("inc", "纳入", "保留符合此条件的合成记录。"),
                                         ("exc", "排除", "移除符合此条件的合成记录。")):
        parts.append(f"<fieldset><legend>1 · 目标人群：{heading}条件</legend><p>{explanation}所有变量在第 0 天合成测量；此处的阈值仅为教学示例。</p><div class='field-grid'>")
        parts.append(select_field(f"{prefix}_variable", "合成基线变量", VARIABLE_LABELS))
        parts.append(select_field(f"{prefix}_operator", "比较方式", OPERATORS))
        parts.append(input_field(f"{prefix}_value", "比较数值", "标准化指标无临床单位；二元标记只能填 0 或 1。", "number", "step='any'"))
        parts.append("</div></fieldset>")
    parts.append("<fieldset><legend>2 · 时间零点</legend><p>合格策略首次启动为第 0 天，筛选和协变量测量均在第 0 天完成。"
                 "本教学实例不开放其他时间零点，以避免在当前合成生成器中误用时间顺序。</p></fieldset>")
    parts.append("<fieldset><legend>3 · 比较策略</legend><p>仅改变显示名称；合成队列始终为对照与比较两臂。</p><div class='field-grid'>")
    parts.append(input_field("control_label", "对照策略名称", "固定为合成对照臂。", attributes="maxlength='80'"))
    parts.append(input_field("active_label", "比较策略名称", "固定为合成比较臂。", attributes="maxlength='80'"))
    parts.append("</div></fieldset>")
    parts.append("<fieldset><legend>4 · 策略分配</legend><p>基线观察性分配；根据下列四个合成基线变量估计倾向分数，"
                 "使用两臂重叠加权。此预览不提供随机分配或多臂方案。</p><ul class='variable-list'>")
    parts.extend(f"<li><code>{esc(key)}</code>：{esc(label)}</li>" for key, label in VARIABLE_LABELS.items())
    parts.append("</ul></fieldset>")
    parts.append("<fieldset><legend>5 · 随访与评估时点</legend><p>在指定天数比较累计发生率；随访结束日须晚于全部评估时点。</p><div class='field-grid'>")
    for index in (1, 2, 3):
        parts.append(input_field(f"horizon_{index}", f"第 {index} 个评估时点（天）", "按递增顺序填写正数。", "number", "min='1' step='any'"))
    parts.append(input_field("administrative_end_day", "行政随访结束（天）", "晚于最后一个评估时点。", "number", "min='1' step='any'"))
    parts.append("</div></fieldset>")
    parts.append("<fieldset><legend>6 · 目标事件与竞争事件</legend><p>合成目标事件编码 1；合成竞争事件编码 2；"
                 "编码 0 表示删失。竞争事件不会被当成普通删失。事件机制由固定合成生成器决定，不能代表真实诊断。</p></fieldset>")
    parts.append("<fieldset><legend>7 · 目标估计量、加权与诊断门禁</legend>"
                 "<p>目标是合成重叠人群（ATO）；并列报告指定时点风险差 RD、累计发生率比 RR(CIF)、"
                 "亚分布风险比 sHR 与原因别风险比 csHR。下列数值是教学诊断门禁，不是临床阈值。</p><div class='field-grid'>")
    parts.append(input_field("max_abs_smd", "最大加权 |SMD|", "协变量均衡的演示门禁，数值越小越严格。", "number", "min='0' step='any'"))
    parts.append(input_field("min_ess_ratio", "最小有效样本量比（ESS）", "筛选加权后样本信息量；范围 0–1。", "number", "min='0' max='1' step='any'"))
    parts.append(input_field("positivity_epsilon", "正性下界", "倾向分数过于靠近 0 或 1 时阻断估计；小于 0.5。", "number", "min='0' max='0.49' step='any'"))
    parts.append("</div></fieldset>")
    parts.append("<fieldset><legend>生成设置</legend><p>固定合成机制与种子保证结果可复现；默认生成 2,000 行教学队列。</p><div class='field-grid'>")
    parts.append(input_field("cases", "合成队列行数", "100–5,000 行；默认 2,000 行。", "number", "min='100' max='5000' step='1'"))
    parts.append(input_field("seed", "随机种子", "相同方案与种子可重现队列。", "number", "min='0' max='4294967295' step='1'"))
    parts.append(input_field("bootstrap_iterations", "Bootstrap 次数", "用于演示区间估计；20–300 次。", "number", "min='20' max='300' step='1'"))
    parts.append("</div></fieldset><div class='actions'><button type='submit'>查看方案摘要 →</button></div></form>")
    return "".join(parts)


def review_view(fields, normalized, settings):
    hidden = "".join(f"<input type='hidden' name='{esc(key)}' value='{esc(fields[key])}'>" for key in sorted(FIELD_NAMES))
    return ("<p class='eyebrow'>步骤 2 / 3 · 确认方案</p><h2>运行前方案摘要</h2>"
            "<p>请核对目标人群、时间、策略和门禁。确认后才会生成完全合成队列。"
            "统计门禁将在生成和筛选之后检查；未通过时不会显示估计。</p>"
            + protocol_summary(normalized)
            + f"<p><strong>生成设置：</strong>{settings['cases']} 行；种子 {settings['seed']}；"
              f"Bootstrap {settings['bootstrap_iterations']} 次。结果以实际筛选与诊断为准。</p>"
            + f"<details><summary>技术详情：规范化方案 JSON</summary><pre>{esc(json.dumps(normalized, ensure_ascii=False, indent=2))}</pre></details>"
            + f"<form method='post' action='/runs'>{hidden}<div class='actions'>"
              "<button type='submit'>确认并运行合成 TTE →</button> "
              "<button type='submit' formaction='/protocol/edit' class='secondary'>返回修改</button>"
              "</div></form>")


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


def curves_svg(curve_paths, column, title, arm_labels=None):
    series = []
    for path in curve_paths:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        points = [(float(row["time_day"]), float(row[column])) for row in rows]
        series.append((rows[0]["treatment_arm"], points))
    if not series:
        return ""
    max_day = max(point[0] for _, points in series for point in points) or 1.0
    colors = ("#1769aa", "#d3672e", "#7d4aa0", "#288d69")
    lines = []
    legend = []
    for index, (name, points) in enumerate(series):
        path_data = "M40.0,240.0 " + " ".join(
            f"H{40 + 620*x/max_day:.1f} V{240 - 210*y:.1f}" for x, y in points)
        color = colors[index % len(colors)]
        lines.append(f"<path class='cif-step' d='{path_data}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        legend.append(f"<span style='color:{color}'>●</span> {esc((arm_labels or {}).get(name, name))}")
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


COHORT_LABELS = {"synthetic_case_id": "合成记录号", "simulation_or_data_label": "数据标识",
                 "treatment_arm": "分配策略", "treatment_idx": "策略序号",
                 "duration_day": "随访天数", "event_code": "事件编码（0 删失/1 目标/2 竞争）",
                 **VARIABLE_LABELS}


def cohort_view(result, root, page_number):
    """Show a bounded page; downloads remain the complete generated CSV."""
    total = result["cohort_rows"]
    page_size = 25
    pages = (total + page_size - 1) // page_size
    if not 1 <= page_number <= pages:
        raise ValueError("invalid page")
    path = Path(root) / result["run_id"] / "cohort" / "cohort.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(islice(reader, (page_number - 1) * page_size, page_number * page_size))
    base = f"/runs/{result['run_id']}"
    controls = []
    if page_number > 1:
        controls.append(f"<a href='{base}/cohort?page={page_number - 1}'>← 上一页</a>")
    if page_number < pages:
        controls.append(f"<a href='{base}/cohort?page={page_number + 1}'>下一页 →</a>")
    return ("<p class='eyebrow'>合成队列 · 可分页查看</p><h2>全部变量与记录</h2>"
            f"<p>共 {total} 行、{len(result['cohort_columns'])} 列；第 {page_number} / {pages} 页，"
            f"每页最多 {page_size} 行。表格可横向滚动；完整数据请下载 CSV。</p>"
            f"<p><a href='{base}/artifacts/cohort/cohort.csv'>下载完整生成队列 CSV</a> · "
            f"<a href='{base}/artifacts/screen/eligible_cohort.csv'>下载完整筛选后队列 CSV</a></p>"
            + table(rows, result["cohort_columns"], COHORT_LABELS)
            + f"<p class='actions'>{' · '.join(controls)} · <a href='{base}'>返回结果</a></p>")


def comparison_table(result):
    def interval(row, key):
        return f"{_fmt(row[key])} [{_fmt(row[key + '_ci_lower'])}, {_fmt(row[key + '_ci_upper'])}]"

    risks = []
    hazards = []
    for row in result["estimates"]["co_presented_comparison_table"]:
        risks.append({"day": _fmt(row["horizon_day"]), "primary_ref": _fmt(row["cif_primary_ref"]),
                      "primary_active": _fmt(row["cif_primary_active"]),
                      "rd_primary": interval(row, "rd_primary"), "rr_primary": interval(row, "rr_cif_primary"),
                      "competing_ref": _fmt(row["cif_competing_ref"]),
                      "competing_active": _fmt(row["cif_competing_active"]),
                      "rd_competing": interval(row, "rd_competing")})
        hazards.append({"day": _fmt(row["horizon_day"]), "shr": interval(row, "shr_primary"),
                        "cshr_primary": interval(row, "cshr_primary"),
                        "cshr_competing": interval(row, "cshr_competing"),
                        "evalue": _fmt(row["evalue_rr_cif_point"]),
                        "evalue_ci": _fmt(row["evalue_rr_cif_ci_limit"])})
    return ("<h3>累计发生率与指定时点差异</h3><p>方括号内为 95% Bootstrap 区间。</p>"
            + table(risks, ["day", "primary_ref", "primary_active", "rd_primary", "rr_primary",
                            "competing_ref", "competing_active", "rd_competing"],
                    {"day": "天", "primary_ref": "目标 CIF 对照", "primary_active": "目标 CIF 比较",
                     "rd_primary": "目标 RD（95%区间）", "rr_primary": "目标 RR(CIF)（95%区间）",
                     "competing_ref": "竞争 CIF 对照", "competing_active": "竞争 CIF 比较",
                     "rd_competing": "竞争 RD（95%区间）"})
            + "<h3>亚分布与原因别风险比</h3>"
            + table(hazards, ["day", "shr", "cshr_primary", "cshr_competing", "evalue", "evalue_ci"],
                    {"day": "天", "shr": "目标 sHR（95%区间）", "cshr_primary": "目标 csHR（95%区间）",
                     "cshr_competing": "竞争 csHR（95%区间）", "evalue": "E-value 点估计",
                     "evalue_ci": "E-value 区间界限"}))


def attrition_table(result):
    names = {strategy["id"]: ("对照" if strategy["role"] == "control" else "比较")
             for strategy in result["normalized_protocol"]["treatment_strategies"]}
    rows = []
    for entry in result["attrition"]:
        step = entry["step_id"]
        label = {"generated_synthetic_cohort": "生成队列", "initial_cohort": "合格队列",
                 "landmark_risk_set_slice": "时间零点风险集"}.get(step)
        if label is None:
            label = "纳入筛选" if step.startswith("inc_") else "排除筛选" if step.startswith("exc_") else step
        try:
            arms = json.loads(entry.get("arm_counts_json", "{}"))
            counts = "；".join(f"{names.get(arm, arm)} {count}" for arm, count in arms.items())
        except (ValueError, TypeError):
            counts = "—"
        rows.append({"step": label, "remaining": entry.get("remaining_n", ""),
                     "excluded": entry.get("excluded_n", ""), "arms": counts})
    return table(rows, ["step", "remaining", "excluded", "arms"],
                 {"step": "筛选步骤", "remaining": "剩余行数", "excluded": "本步排除", "arms": "各策略行数"})


def gate_failure_view(diagnostic, protocol):
    """Explain blocked estimates in Chinese; retain exact machine reasons in disclosure."""
    gates = protocol["estimand"]["diagnostic_gates"]
    propensity = diagnostic.get("propensity_diagnostics", {})
    ess = diagnostic.get("ess_diagnostics", {})
    explanations = {
        "NO_ELIGIBLE_SYNTHETIC_RISK_SET": "入排筛选后至少一个比较策略没有合格合成记录，无法建立两臂风险集。请放宽筛选条件后重试。",
        "ZERO_RISK_SET": "时间零点后至少一个策略的风险集为空，无法比较两臂。请检查筛选与时间零点。",
        "POSITIVITY_VIOLATION": ("正性门禁未通过：观察到的倾向分数范围为 "
                                  f"{_fmt(propensity.get('min_propensity'))}–{_fmt(propensity.get('max_propensity'))}；"
                                  f"方案要求远离 0 和 1，正性下界为 {esc(gates['positivity_epsilon'])}。"),
        "ESS_RATIO_BELOW_MIN": ("有效样本量门禁未通过：最小 ESS 比为 "
                                f"{_fmt(ess.get('min_ess_ratio_observed'))}，低于方案下限 {esc(gates['min_ess_ratio'])}。"),
        "SMD_EXCEEDS_MAX": ("协变量均衡门禁未通过：加权后最大 |SMD| 为 "
                            f"{_fmt(diagnostic.get('max_abs_smd_weighted'))}，高于方案上限 {esc(gates['max_abs_smd'])}。"),
    }
    failures = diagnostic.get("gate_failures", [])
    items = "".join(f"<li>{explanations.get(item.get('code'), '其他诊断门禁未通过；请查看技术详情。')}</li>"
                    for item in failures if isinstance(item, dict))
    if not items:
        items = "<li>诊断门禁未通过；请检查方案与技术详情。</li>"
    raw = esc(json.dumps(failures, ensure_ascii=False, indent=2))
    return (f"<ul>{items}</ul><details class='technical-details print-hide'><summary>技术详情：门禁代码和原始诊断</summary>"
            f"<pre>{raw}</pre></details>")


def warning_view(warnings, comparison):
    """Carry methodological and run-specific warnings into both readable surfaces."""
    parts = ["<p>亚分布风险比 sHR 与原因别风险比 csHR 使用不同风险集；风险比要与目标和竞争事件的累计发生率一同阅读。"
             "csHR 条件于仍无事件者，存在事件后选择问题。规则检查无法发现未测混杂、测量误差或信息性删失。</p>"]
    alert = warnings.get("competing_mortality_divergence_alert", {})
    if alert.get("triggered"):
        first = comparison[0] if comparison else {}
        parts.append("<div class='error'><strong>本次运行触发竞争事件分歧警示。</strong>"
                     "<p>合成比较策略的目标事件 sHR 下降，但目标 csHR 未同步下降，竞争事件 csHR 升高。"
                     "目标事件累计发生率降低可能由竞争事件更早发生造成，不能视为直接保护作用。"
                     f"本次 sHR {_fmt(first.get('shr_primary'))}、目标 csHR {_fmt(first.get('cshr_primary'))}、"
                     f"竞争 csHR {_fmt(first.get('cshr_competing'))}。</p></div>")
    parts.append("<details class='technical-details print-hide'><summary>技术详情：原始方法警示</summary>"
                 f"<pre>{esc(json.dumps(warnings, ensure_ascii=False, indent=2))}</pre></details>")
    return "".join(parts)


def report_view(result, root):
    """Self-contained, print-friendly scientific reading surface, with no JSON required."""
    run_id = result["run_id"]
    base = f"/runs/{run_id}/artifacts/"
    manifest = result["preview_manifest"]
    diagnostic = result["diagnostics"]
    arm_labels = {strategy["id"]: strategy["label"] for strategy in result["normalized_protocol"]["treatment_strategies"]}
    parts = ["<p class='eyebrow'>研究方案与分析报告 · 完全合成教学实例</p>"
             f"<h2>{esc(result['normalized_protocol']['study_title'])}</h2>"
             "<p><strong>资料性质：</strong>完全合成教学数据；无真实患者资料，不能作为临床疗效或因果实证证据。</p>"
             f"<p>运行编号 {esc(run_id)}；启动时间 {esc(result.get('started_utc', '未记录'))}；"
             f"完成时间 {esc(result.get('finished_utc', '未记录'))}。</p>"
             "<p class='print-hide'><a href='?download=1'>下载此 HTML 报告</a> · "
             "使用浏览器打印或另存 PDF（⌘/Ctrl+P） · "
             f"<a href='/runs/{run_id}'>返回交互结果</a></p>"
             "<h2>一、研究方案</h2>", protocol_summary(result["normalized_protocol"]),
             "<h2>二、合成数据与分析方法</h2>"
             f"<p>生成队列 {result['cohort_rows']} 行、{len(result['cohort_columns'])} 列；"
             f"基线资格筛选后 {result['eligible_rows']} 行。固定种子 {result['seed']}；"
             f"Bootstrap {result['bootstrap_iterations']} 次。合成变量：{esc('、'.join(VARIABLE_LABELS.values()))}。</p>"
             "<p>在合格风险集上估计基线倾向分数并使用两臂重叠权重；先检查协变量均衡、ESS 与正性门禁。"
             "通过门禁后，以 Aalen–Johansen 累计发生率和原因别/亚分布风险模型并列呈现目标及竞争事件。</p>"
             "<h2>三、筛选与诊断</h2>",
             attrition_table(result),
             f"<p>门禁状态：<strong>{'通过' if result['gate_status'] == 'passed' else '阻断'}</strong>；"
             f"加权最大 |SMD| {_fmt(diagnostic.get('max_abs_smd_weighted'))}；"
             f"最小 ESS 比 {_fmt(diagnostic.get('ess_diagnostics', {}).get('min_ess_ratio_observed'))}。</p>",
             table([{"arm": arm_labels.get(arm, arm), **values} for arm, values in diagnostic.get("ess_diagnostics", {}).get("by_arm", {}).items()],
                   ["arm", "n", "ess", "ess_ratio", "raw_weight_min", "raw_weight_max"],
                   {"arm": "策略", "n": "行数", "ess": "ESS", "ess_ratio": "ESS 比", "raw_weight_min": "最小权重", "raw_weight_max": "最大权重"})]
    if result["gate_status"] == "passed":
        curves = sorted((Path(root) / run_id / "estimates" / "curves").glob("*.csv"))
        parts.extend(["<h2>四、目标与竞争事件累计发生率</h2>",
                      curves_svg(curves, "cif_primary", "目标事件 CIF（阶梯曲线）", arm_labels),
                      curves_svg(curves, "cif_competing", "竞争事件 CIF（阶梯曲线）", arm_labels),
                      "<h2>五、各时点对比估计</h2>", comparison_table(result),
                      "<p>RD 为指定时点累计发生率差，RR(CIF) 为累计发生率比；sHR 和 csHR 使用不同风险集，"
                      "必须与目标和竞争事件 CIF 一起阅读。</p>"])
    else:
        parts.append("<div class='error'><strong>诊断门禁未通过，不生成估计。</strong>"
                     + gate_failure_view(diagnostic, result["normalized_protocol"]) + "</div>")
    if result["gate_status"] == "passed":
        parts.append("<h2>六、运行特定警示</h2>" + warning_view(
            result["estimates"]["methodological_warnings"],
            result["estimates"]["co_presented_comparison_table"]))
    parts.extend(["<h2>七、方法限制与版本</h2>"
                  "<p>仅检查可观察的方案结构和时间规则；不能识别未测混杂、测量误差或信息性删失。"
                  "Bootstrap 区间固定第一阶段倾向权重，未计入其模型估计不确定性。"
                  "所有阈值和数字仅用于合成教学，不得解释为临床决策依据。</p>"
                  f"<p>服务 Git 提交 {esc(manifest['code_commit'])}；分析依赖 PR #14 提交 "
                  f"{esc(manifest.get('tte_dependency_commit', '未记录'))}（已合入 main，预览固定源版本）；"
                  f"镜像版本 emulatrx-preview:{esc(manifest['code_commit'])}。</p>"
                  f"<p class='print-hide'><a href='{base}preview_manifest.json'>下载哈希清单</a> · "
                  f"<a href='{base}protocol_check/normalized_protocol.json'>下载机器可读方案</a></p>"])
    if result["gate_status"] == "passed":
        parts.append(f"<p class='print-hide'><a href='{base}estimates/audit_report.md'>下载估计审计 Markdown</a></p>")
    return "".join(parts)


def result_view(result, root):
    state = result["state"]
    if state in ("queued", "running"):
        stages = {"queued": "等待开始", "protocol_validation": "核对方案", "synthetic_generation": "生成完全合成队列",
                  "eligibility_screening": "执行资格筛选", "diagnostics": "检查加权与诊断门禁",
                  "estimation": "估计目标与竞争事件", "artifact_audit": "整理报告与审计清单"}
        label = stages.get(result.get("stage"), "处理中")
        return ("<p class='eyebrow'>步骤 3 / 3 · 正在分析</p><h2>合成 TTE 正在运行</h2>"
                f"<p>当前阶段：<strong>{esc(label)}</strong>。页面每 2 秒自动刷新。</p>"
                f"<p>运行编号 <code>{esc(result['run_id'])}</code>；若门禁未通过，将只显示诊断及原因。</p>")
    if state in ("failed", "interrupted"):
        return f"<div class='error'>运行未完成：{esc(result.get('error', state))}</div><p><a href='/protocol/new'>返回方案</a></p>"
    run_id = result["run_id"]
    root = Path(root) / run_id
    base = f"/runs/{run_id}/artifacts/"
    gate = result["gate_status"]
    diagnostic = result["diagnostics"]
    gate_label = "通过" if gate == "passed" else "阻断"
    parts = ["<section class='run-overview' aria-labelledby='run-overview-title'>"
             "<h2 id='run-overview-title'>本次运行概览</h2>"
             f"<p>运行编号 <code>{run_id}</code> · 完全合成方法学预览</p>"
             "<div class='metric-grid'>"
             f"<div><strong>{result['cohort_rows']} 行</strong><span>生成队列</span></div>"
             f"<div><strong>{len(result['cohort_columns'])} 列</strong><span>全部变量</span></div>"
             f"<div><strong>{result['eligible_rows']} 行</strong><span>资格筛选后</span></div>"
             f"<div><strong>{gate_label}</strong><span>诊断门禁</span></div></div>"
             f"<p>变量列：{esc('、'.join(COHORT_LABELS.get(column, column) for column in result['cohort_columns']))}</p>"
             f"<p><a href='{base}cohort/cohort.csv'>下载完整生成队列 CSV</a> · "
             f"<a href='{base}screen/eligible_cohort.csv'>下载完整筛选后队列 CSV</a> · "
             f"<a href='/runs/{run_id}/cohort?page=1'>分页查看全部记录</a> · "
             f"<a href='/runs/{run_id}/report'>阅读、打印或下载研究方案与分析报告</a> · "
             f"<a href='{base}preview_manifest.json'>下载预览清单</a> · "
             f"<a href='{base}cohort/run_manifest.json'>下载生成清单</a></p>"
             "<nav aria-label='结果面板'><a href='#protocol'>方案</a> · <a href='#cohort'>队列</a> · "
             "<a href='#attrition'>筛选流失</a> · <a href='#diagnostics'>诊断与门禁</a>"
             + (" · <a href='#estimates'>估计结果</a> · <a href='#audit'>审计报告</a>" if gate == "passed" else "")
             + "</nav></section>"]
    if gate == "passed" and result["estimates"]["co_presented_comparison_table"]:
        first = result["estimates"]["co_presented_comparison_table"][0]
        parts.append(f"<p class='key-result'>第 {esc(first['horizon_day'])} 天合成基准：目标事件 "
                     f"RD {_fmt(first['rd_primary'])}；RR(CIF) {_fmt(first['rr_cif_primary'])}；"
                     f"sHR {_fmt(first['shr_primary'])}；csHR {_fmt(first['cshr_primary'])}。"
                     "<a href='#estimates'>查看全部时点与区间</a></p>")
    parts.extend(["<h2 id='protocol'>1 · 规范化七要素方案</h2>",
             protocol_summary(result["normalized_protocol"]),
             f"<details><summary>查看完整规范化 JSON</summary><pre>{esc(json.dumps(result['normalized_protocol'], ensure_ascii=False, indent=2))}</pre></details>",
             f"<p><a href='{base}protocol_check/normalized_protocol.json'>下载规范化方案</a></p>",
             "<h2 id='cohort'>2 · 合成数据预览</h2>",
             f"<p>共 {result['cohort_rows']} 行，以下预览前 {len(result['sample_cases'])} 行；"
             f"<a href='/runs/{run_id}/cohort?page=1'>打开每页 25 行的队列表</a>，或下载上方完整 CSV。</p>",
             "<details><summary>查看部分合成记录与所有变量列</summary>",
             table(result["sample_cases"], result["cohort_columns"], COHORT_LABELS), "</details>",
             "<h2 id='attrition'>3 · 筛选流失</h2>",
             attrition_table(result),
             "<h2 id='diagnostics'>4 · 诊断与门禁</h2>",
             f"<p>未加权最大 |SMD|：{_fmt(diagnostic.get('max_abs_smd_unweighted'))}；加权最大 |SMD|：{_fmt(diagnostic.get('max_abs_smd_weighted'))}；最小 ESS 比：{_fmt(diagnostic.get('ess_diagnostics', {}).get('min_ess_ratio_observed'))}。</p>",
             f"<p>倾向分数范围：{_fmt(diagnostic.get('propensity_diagnostics', {}).get('min_propensity'))}–{_fmt(diagnostic.get('propensity_diagnostics', {}).get('max_propensity'))}。门禁阈值来自方案，不代表临床阈值。</p>"])
    ess = diagnostic.get("ess_diagnostics", {}).get("by_arm", {})
    arm_labels = {strategy["id"]: strategy["label"] for strategy in result["normalized_protocol"]["treatment_strategies"]}
    parts.append(table([{"arm": arm_labels.get(arm, arm), **stats} for arm, stats in ess.items()], ["arm", "n", "ess", "ess_ratio", "raw_weight_min", "raw_weight_max"],
                       {"arm": "策略", "n": "行数", "ess": "有效样本量 ESS", "ess_ratio": "ESS 比", "raw_weight_min": "最小权重", "raw_weight_max": "最大权重"}))
    balance_rows = [{**row, "covariate": VARIABLE_LABELS.get(row["covariate"], row["covariate"])} for row in result["balance"]]
    parts.append("<h3>协变量均衡</h3>" + table(balance_rows, ["covariate", "abs_smd_unweighted", "abs_smd_weighted"],
                                             {"covariate": "合成变量", "abs_smd_unweighted": "加权前 |SMD|", "abs_smd_weighted": "加权后 |SMD|"}))
    if gate == "blocked":
        parts.append("<div class='error'><strong>诊断门禁未通过。估计结果与曲线不生成。</strong>"
                     + gate_failure_view(diagnostic, result["normalized_protocol"]) + "</div>")
        parts.append(f"<p><a href='{base}diagnostics/diagnostics_summary.json'>下载诊断</a> · <a href='{base}diagnostics/balance_diagnostics.csv'>下载均衡表</a> · <a href='{base}screen/eligibility_attrition.csv'>下载筛选流失表</a></p>")
    else:
        active_id = result["normalized_protocol"]["treatment_strategies"][1]["id"]
        parts.append("<h3>倾向分数分布</h3>" + propensity_histogram(root / "diagnostics" / "weighted_cohort.csv", active_id))
        curves = sorted((root / "estimates" / "curves").glob("*.csv"))
        parts.append("<h2 id='curves'>5 · 目标事件与竞争事件累计发生率</h2>")
        parts.append(curves_svg(curves, "cif_primary", "目标事件 CIF", arm_labels)
                     + curves_svg(curves, "cif_competing", "竞争事件 CIF", arm_labels))
        parts.append("<h2 id='estimates'>6 · 指定时点对比估计</h2>")
        parts.append(comparison_table(result))
        parts.append("<p>sHR 与 csHR 描述不同的风险集；所有数值仅来自合成基准，不能解释为真实治疗效果。Bootstrap 区间固定已估计的基线倾向权重，不包含重新拟合倾向模型的不确定性。</p>")
        warnings = result["estimates"]["methodological_warnings"]
        parts.append("<h3>方法限制与警示</h3>" + warning_view(
            warnings, result["estimates"]["co_presented_comparison_table"]))
        report = (root / "estimates" / "audit_report.md").read_text(encoding="utf-8")
        parts.append(f"<h2 id='audit'>7 · 完整审计报告</h2><p><a href='/runs/{run_id}/report'>阅读和打印中文研究方案与分析报告</a> · <a href='{base}estimates/audit_report.md'>下载审计 Markdown</a> · <a href='{base}estimates/estimates.json'>下载完整估计 JSON</a> · <a href='{base}estimates/run_manifest.json'>下载估计清单</a></p><details><summary>查看完整审计 Markdown（技术详情）</summary><pre>{esc(report)}</pre></details>")
    parts.append(f"<h2>产物与清单</h2><p>可下载产物 {len(result['available_artifacts'])} 项；审计提交 {esc(result['preview_manifest']['code_commit'])}。</p>")
    parts.append("<p><a href='/protocol/new'>新建方案</a></p>")
    return "".join(parts)
