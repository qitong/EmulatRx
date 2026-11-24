"""Constrained Chinese web controls mapped to the reviewed synthetic TTE contract."""

from copy import deepcopy
import math


VARIABLE_LABELS = {
    "x_age_std": "合成标准化年龄指标（无临床单位）",
    "x_biomarker_std": "合成标准化生物标志物指标（无临床单位）",
    "x_comorbidity_bin": "合成二元合并症标记（0/1）",
    "x_high_risk_bin": "合成二元高风险标记（0/1）",
}
OPERATORS = {">=": "大于或等于", ">": "大于", "<=": "小于或等于",
             "<": "小于", "==": "等于", "!=": "不等于"}
FIELD_NAMES = frozenset({
    "study_title", "inc_variable", "inc_operator", "inc_value",
    "exc_variable", "exc_operator", "exc_value", "control_label", "active_label",
    "horizon_1", "horizon_2", "horizon_3", "administrative_end_day",
    "max_abs_smd", "min_ess_ratio", "positivity_epsilon",
    "cases", "seed", "bootstrap_iterations",
})


class FormError(ValueError):
    def __init__(self, field, message):
        self.field = field
        super().__init__(message)


def default_fields(protocol):
    inc = protocol["eligibility"]["inclusion_criteria"][0]
    exc = protocol["eligibility"]["exclusion_criteria"][0]
    horizons = protocol["followup"]["horizon_days"]
    gates = protocol["estimand"]["diagnostic_gates"]
    return {
        "study_title": protocol["study_title"],
        "inc_variable": inc["variable"], "inc_operator": inc["operator"], "inc_value": str(inc["value"]),
        "exc_variable": exc["variable"], "exc_operator": exc["operator"], "exc_value": str(exc["value"]),
        "control_label": protocol["treatment_strategies"][0]["label"],
        "active_label": protocol["treatment_strategies"][1]["label"],
        "horizon_1": str(horizons[0]), "horizon_2": str(horizons[1]), "horizon_3": str(horizons[2]),
        "administrative_end_day": str(protocol["followup"]["administrative_end_day"]),
        "max_abs_smd": str(gates["max_abs_smd"]), "min_ess_ratio": str(gates["min_ess_ratio"]),
        "positivity_epsilon": str(gates["positivity_epsilon"]),
        "cases": "2000", "seed": "42", "bootstrap_iterations": "100",
    }


def _number(fields, name):
    try:
        value = float(fields[name])
    except (ValueError, KeyError, OverflowError) as exc:
        raise FormError(name, "请输入有效数字。") from exc
    if not math.isfinite(value):
        raise FormError(name, "请输入有限数字。")
    return value


def _integer(fields, name, minimum, maximum):
    try:
        raw = fields[name]
        if not raw.isascii() or not raw.isdecimal():
            raise ValueError
        value = int(raw)
    except (ValueError, KeyError) as exc:
        raise FormError(name, "请输入整数。") from exc
    if not minimum <= value <= maximum:
        raise FormError(name, f"请输入 {minimum} 至 {maximum} 之间的整数。")
    return value


def protocol_from_fields(fields, default_protocol):
    """Use only reviewed synthetic variables and fixed design choices for web input."""
    if set(fields) != FIELD_NAMES:
        raise FormError("study_title", "表单字段缺失或多余，请重新载入教学示例。")
    title = fields["study_title"].strip()
    if not 1 <= len(title) <= 160:
        raise FormError("study_title", "研究标题须为 1 至 160 个字。")
    protocol = deepcopy(default_protocol)
    protocol["study_title"] = title
    for prefix, kind, criterion_id in (("inc", "inclusion_criteria", "inc_synthetic_baseline"),
                                       ("exc", "exclusion_criteria", "exc_synthetic_baseline")):
        variable = fields[f"{prefix}_variable"]
        operator = fields[f"{prefix}_operator"]
        if variable not in VARIABLE_LABELS:
            raise FormError(f"{prefix}_variable", "请选择页面列出的合成基线变量。")
        if operator not in OPERATORS:
            raise FormError(f"{prefix}_operator", "请选择受支持的数值比较方式。")
        value = _number(fields, f"{prefix}_value")
        if variable.endswith("_bin") and value not in (0, 1):
            raise FormError(f"{prefix}_value", "二元标记只能与 0 或 1 比较。")
        protocol["eligibility"][kind] = [{"id": criterion_id, "variable": variable,
                                            "operator": operator, "value": value,
                                            "assessed_day_offset": 0}]
    for index, key in enumerate(("control_label", "active_label")):
        label = fields[key].strip()
        if not 1 <= len(label) <= 80:
            raise FormError(key, "策略名称须为 1 至 80 个字。")
        protocol["treatment_strategies"][index]["label"] = label
    horizons = [_number(fields, f"horizon_{index}") for index in (1, 2, 3)]
    if any(value <= 0 for value in horizons) or horizons != sorted(set(horizons)):
        raise FormError("horizon_1", "三个评估时点须为递增且不重复的正数天。")
    end_day = _number(fields, "administrative_end_day")
    if end_day <= horizons[-1]:
        raise FormError("administrative_end_day", "随访结束日须晚于最后一个评估时点。")
    protocol["followup"] = {"horizon_days": horizons, "administrative_end_day": end_day}
    gates = protocol["estimand"]["diagnostic_gates"]
    for key in ("max_abs_smd", "min_ess_ratio", "positivity_epsilon"):
        gates[key] = _number(fields, key)
    if gates["max_abs_smd"] <= 0:
        raise FormError("max_abs_smd", "最大 |SMD| 门禁须大于 0。")
    if not 0 < gates["min_ess_ratio"] <= 1:
        raise FormError("min_ess_ratio", "ESS 比门禁须在 0 与 1 之间。")
    if not 0 < gates["positivity_epsilon"] < 0.5:
        raise FormError("positivity_epsilon", "正性下界须大于 0 且小于 0.5。")
    settings = {"cases": _integer(fields, "cases", 100, 5000),
                "seed": _integer(fields, "seed", 0, 2**32 - 1),
                "bootstrap_iterations": _integer(fields, "bootstrap_iterations", 20, 300)}
    return protocol, settings
