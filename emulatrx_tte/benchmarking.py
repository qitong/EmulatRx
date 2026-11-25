"""Study-level RCT published-results benchmarking and negative-control diagnostics.

Public ClinicalTrials.gov ``resultsSection`` aggregates and caller-supplied citations are
study-level statistics only (``ipd_used`` is always false). Numerical proximity to a target
RCT is not evidence of causality, of no unmeasured confounding, or of clinical efficacy.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from ._common import ContractError
from .competing_risks import fit_weighted_cause_specific_cox
from .weighting import compute_weights


SCHEMA_VERSION = "tte-benchmark-v1"
Z_975 = 1.959963984540054
PI_Z = 1.96
CTGOV_STUDY_URL = "https://clinicaltrials.gov/api/v2/studies/{nct_id}"
OBSERVED_COVARIATES = ("age", "bmi", "prior_cvd")

INTERPRETIVE_GUARDRAIL = {
    "numerical_proximity_proves_causality": False,
    "clinical_efficacy_claim_prohibited": True,
    "summary_zh": (
        "对标指标仅反映加权观察性估计与公开试验汇总结果在统计分布上的接近程度；"
        "数值相近可能源于多种偏倚相互抵消，绝不可作为无未测混杂、因果效应确立或临床有效性的证明。"
    ),
    "summary_en": (
        "Benchmark agreement reflects statistical proximity to published aggregate RCT summaries only; "
        "numerical proximity never proves absence of unmeasured confounding, causality, or clinical efficacy."
    ),
}

_ALIGNED_OVERALL = frozenset({"aligned", "protocol_aligned"})
_OK_POPULATION = frozenset({"matched", "aligned"})
_OK_ARMS = frozenset({"exact_dual_arm", "aligned"})
_OK_OUTCOME = frozenset({"matched", "aligned"})
_OK_TIME = frozenset({"adequate", "aligned", "matched"})


def is_benchmarking_enabled() -> bool:
    """Independent rollback switch. Default on; ``0`` / ``false`` / ``off`` / ``no`` disables it."""
    raw = os.environ.get("EMULATRX_ENABLE_BENCHMARKING", "1")
    return str(raw).strip().lower() not in {"0", "false", "off", "no"}


def interpretive_guardrail() -> dict:
    return dict(INTERPRETIVE_GUARDRAIL)


def benchmarking_disabled_payload() -> dict:
    return {
        "error_code": "BENCHMARKING_FEATURE_DISABLED",
        "gate_code": "BENCHMARKING_FEATURE_DISABLED",
        "status": "disabled",
        "benchmarkable": False,
        "effect_comparison": None,
        "consistency_conclusion_allowed": False,
        "ipd_used": False,
        "message_zh": "RCT 对标与阴性对照诊断已由 EMULATRX_ENABLE_BENCHMARKING 关闭，可立即回滚且无需重新部署。",
        "message_en": "RCT benchmarking and negative-control diagnostics are disabled by EMULATRX_ENABLE_BENCHMARKING.",
        "interpretive_guardrail": interpretive_guardrail(),
    }


def _json_float(value):
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _offline_catalog_forced() -> bool:
    return os.environ.get("EMULATRX_OFFLINE_RCT_CATALOG", "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_nct(nct_id: str | None) -> str:
    text = str(nct_id or "").strip().upper()
    if len(text) != 11 or not text.startswith("NCT") or not text[3:].isdigit():
        raise ContractError("INVALID_NCT_ID", f"NCT id must look like NCT########, got {nct_id!r}")
    return text


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "aligned"}:
            return True
        if token in {"0", "false", "no", "mismatched", "unaligned"}:
            return False
    return None


def _dimension_ok(audit: dict, explicit_key: str, repo_key: str, ok_values: frozenset[str]) -> bool | None:
    if explicit_key in audit:
        flag = _as_bool(audit.get(explicit_key))
        if flag is None:
            return False
        return flag
    if repo_key in audit and audit.get(repo_key) not in (None, ""):
        return str(audit.get(repo_key)) in ok_values
    return None


def audit_benchmark_eligibility(tte_result: dict, alignment_audit: dict) -> dict:
    """Fail closed when the TTE gate failed or PICO-T alignment is not exact.

    Earlier ``confirm_protocol_mismatch`` / ``confirm_arm_mismatch`` flags are ignored.
    An exploratory TTE is not a license to declare consistency with the target RCT.
    """
    if not isinstance(tte_result, dict) or not isinstance(alignment_audit, dict):
        raise ContractError("INVALID_BENCHMARK_INPUT", "tte_result and alignment_audit must be objects")
    base = {
        "effect_comparison": None,
        "consistency_conclusion_allowed": False,
        "benchmarkable": False,
        "ipd_used": False,
        "mismatch_confirmations_do_not_authorize_consistency": True,
        "interpretive_guardrail": interpretive_guardrail(),
    }
    if tte_result.get("gate_passed") is not True:
        return {
            **base,
            "status": "blocked_upstream_gate",
            "gate_code": "BENCHMARK_BLOCKED_UPSTREAM_DIAGNOSTIC_FAILED",
            "message_zh": "上游 TTE 诊断门禁未通过（gate_passed 不是 true），拒绝进行目标 RCT 对标，禁止输出一致性结论。",
            "message_en": "Upstream TTE diagnostic gate did not pass (gate_passed is not true); benchmarking is refused.",
        }
    overall = alignment_audit.get("overall_alignment", alignment_audit.get("overall_alignment_status"))
    overall_ok = str(overall) in _ALIGNED_OVERALL
    dimension_flags = [
        _dimension_ok(alignment_audit, "population_aligned", "population_alignment", _OK_POPULATION),
        _dimension_ok(alignment_audit, "arms_aligned", "arm_alignment_status", _OK_ARMS),
        _dimension_ok(alignment_audit, "outcome_aligned", "outcome_alignment", _OK_OUTCOME),
        _dimension_ok(alignment_audit, "timeframe_aligned", "followup_alignment", _OK_TIME),
    ]
    present = [flag for flag in dimension_flags if flag is not None]
    dimensions_ok = all(present) if present else overall_ok
    if not overall_ok or not dimensions_ok:
        return {
            **base,
            "status": "blocked_protocol_mismatch",
            "gate_code": "BENCHMARK_BLOCKED_PROTOCOL_OR_ENDPOINT_MISMATCH",
            "message_zh": (
                "目标试验在人群、干预/对照、主要硬终点或随访时间窗上未完全对齐。"
                "即使此前以 confirm_protocol_mismatch 或 confirm_arm_mismatch 放行了探索性 TTE，"
                "也禁止据此输出与目标 RCT 的一致性结论。"
            ),
            "message_en": (
                "Population, arms, primary outcome, or timeframe is not aligned. "
                "Exploratory mismatch confirmations do not authorize a consistency conclusion."
            ),
        }
    return {
        "status": "eligible",
        "gate_code": None,
        "benchmarkable": True,
        "effect_comparison": None,
        "consistency_conclusion_allowed": False,
        "ipd_used": False,
        "interpretive_guardrail": interpretive_guardrail(),
    }


def _parse_float(value):
    if value is None or value == "":
        return None
    try:
        number = float(str(value).replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _scale_from_param_type(param_type: str) -> str | None:
    text = str(param_type or "").strip().lower()
    if "hazard" in text or text in {"hr", "hazard_ratio"}:
        return "hr"
    if "risk difference" in text or "mean difference" in text or text in {"rd", "risk_difference"}:
        return "rd"
    if "risk ratio" in text or "rate ratio" in text or text in {"rr", "risk_ratio"}:
        return "rr"
    return None


def _measurements(measure: dict) -> list[dict]:
    found: list[dict] = []
    for klass in measure.get("classes") or []:
        class_title = str(klass.get("title") or "")
        for category in klass.get("categories") or []:
            cat_title = str(category.get("title") or class_title)
            for item in category.get("measurements") or []:
                found.append({**item, "_class_title": class_title, "_category_title": cat_title})
    return found


def _prefer_total(measurements: list[dict], groups: dict[str, str]) -> dict | None:
    total = []
    other = []
    for item in measurements:
        title = groups.get(str(item.get("groupId")), "")
        blob = f"{title} {item.get('_class_title', '')}".lower()
        if "total" in blob or str(item.get("groupId", "")).endswith("999"):
            total.append(item)
        else:
            other.append(item)
    if total:
        return total[0]
    return other[0] if other else None


def _baseline_from_module(module: dict) -> dict:
    groups = {str(group.get("id")): str(group.get("title") or "") for group in module.get("groups") or []}
    baseline: dict[str, dict] = {}
    for measure in module.get("measures") or []:
        title = str(measure.get("title") or "")
        param = str(measure.get("paramType") or "")
        unit = str(measure.get("unitOfMeasure") or "")
        dispersion = str(measure.get("dispersionType") or "").lower()
        chosen = _prefer_total(_measurements(measure), groups)
        if chosen is None:
            continue
        value = _parse_float(chosen.get("value"))
        spread = _parse_float(chosen.get("spread"))
        title_l = title.lower()
        if value is None:
            continue
        if "age" in title_l and "mean" in param.lower():
            baseline["age"] = {"mean": value, "sd": spread if spread is not None else None}
        elif "female" in title_l or "female" in str(chosen.get("_class_title") or "").lower():
            if "%" in unit or "percent" in unit.lower() or value > 1.0:
                baseline["female"] = {"proportion": value / 100.0 if value > 1.0 else value}
            else:
                denoms = []
                for denom in measure.get("denoms") or []:
                    for count in denom.get("counts") or []:
                        if str(count.get("groupId")) == str(chosen.get("groupId")):
                            denoms.append(_parse_float(count.get("value")))
                denom_value = next((item for item in denoms if item), None)
                if denom_value and denom_value > 0 and "count" in param.lower():
                    baseline["female"] = {"proportion": value / denom_value}
                elif value <= 1.0:
                    baseline["female"] = {"proportion": value}
        elif "mean" in param.lower() and spread is not None and ("sd" in dispersion or "standard deviation" in dispersion):
            key = "bmi" if "bmi" in title_l or "body mass" in title_l else None
            if key and key not in baseline:
                baseline[key] = {"mean": value, "sd": spread}
    return {key: val for key, val in baseline.items() if val}


def _arm_sample_sizes(results_section: dict) -> tuple[int | None, int | None]:
    flow = results_section.get("participantFlowModule") or {}
    for period in flow.get("periods") or []:
        for milestone in period.get("milestones") or []:
            if str(milestone.get("type") or "").upper() != "STARTED":
                continue
            counts = []
            for achievement in milestone.get("achievements") or []:
                num = _parse_float(achievement.get("numSubjects"))
                if num is not None:
                    counts.append(int(num))
            if len(counts) >= 2:
                return counts[0], counts[1]
    return None, None


def _primary_analysis(results_section: dict) -> dict | None:
    module = results_section.get("outcomeMeasuresModule") or {}
    measures = list(module.get("outcomeMeasures") or [])
    primary = [item for item in measures if str(item.get("type") or "").upper() == "PRIMARY"] or measures
    for measure in primary:
        for analysis in measure.get("analyses") or []:
            scale = _scale_from_param_type(str(analysis.get("paramType") or ""))
            point = _parse_float(analysis.get("paramValue"))
            lower = _parse_float(analysis.get("ciLowerLimit"))
            upper = _parse_float(analysis.get("ciUpperLimit"))
            if scale is None or point is None or lower is None or upper is None:
                continue
            margin = _parse_float(analysis.get("nonInferiorityMargin"))
            return {
                "effect_scale": scale,
                "point_estimate": point,
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": _parse_float(analysis.get("pValue")),
                "ni_margin": margin,
                "primary_outcome_title": str(measure.get("title") or ""),
                "primary_outcome_timeframe": str(measure.get("timeFrame") or ""),
            }
    return None


def parse_clinicaltrials_results_study(study: dict) -> dict | None:
    """Parse a ClinicalTrials.gov API v2 study JSON object into study-level aggregates."""
    if not isinstance(study, dict):
        return None
    protocol = study.get("protocolSection") or {}
    identification = protocol.get("identificationModule") or {}
    nct_id = str(identification.get("nctId") or study.get("nctId") or "")
    has_results = bool(study.get("hasResults"))
    results = study.get("resultsSection")
    if not has_results or not isinstance(results, dict):
        return {
            "nct_id": nct_id,
            "has_posted_results": False,
            "official_title": str(identification.get("officialTitle") or ""),
            "usable": False,
        }
    effect = _primary_analysis(results)
    if effect is None:
        return {
            "nct_id": nct_id,
            "has_posted_results": False,
            "official_title": str(identification.get("officialTitle") or ""),
            "usable": False,
        }
    n_treatment, n_control = _arm_sample_sizes(results)
    outcomes = (protocol.get("outcomesModule") or {}).get("primaryOutcomes") or []
    protocol_time = str(outcomes[0].get("timeFrame") or "") if outcomes else ""
    return {
        "nct_id": nct_id,
        "has_posted_results": True,
        "usable": True,
        "official_title": str(identification.get("officialTitle") or ""),
        "primary_outcome_title": effect["primary_outcome_title"] or (str(outcomes[0].get("measure")) if outcomes else ""),
        "primary_outcome_timeframe": effect["primary_outcome_timeframe"] or protocol_time,
        "effect_scale": effect["effect_scale"],
        "point_estimate": effect["point_estimate"],
        "ci_lower": effect["ci_lower"],
        "ci_upper": effect["ci_upper"],
        "p_value": effect["p_value"],
        "ni_margin": effect["ni_margin"],
        "n_treatment": n_treatment,
        "n_control": n_control,
        "baseline_characteristics": _baseline_from_module(results.get("baselineCharacteristicsModule") or {}),
    }


def _v2_study(
    *,
    nct_id: str,
    official_title: str,
    has_results: bool,
    primary_title: str = "",
    timeframe: str = "",
    effect_scale: str = "hr",
    point_estimate: float | None = None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    n_treatment: int | None = None,
    n_control: int | None = None,
    age_mean: float | None = None,
    age_sd: float | None = None,
    female_percent: float | None = None,
    ni_margin: float | None = None,
    arm_titles: tuple[str, str] = ("Experimental", "Active comparator"),
) -> dict:
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "officialTitle": official_title},
            "outcomesModule": {
                "primaryOutcomes": [{"measure": primary_title, "timeFrame": timeframe}] if primary_title else []
            },
        },
        "hasResults": has_results,
    }
    if not has_results:
        return study
    param = {"hr": "Hazard Ratio (HR)", "rr": "Risk Ratio", "rd": "Risk Difference"}[effect_scale]
    analysis = {
        "paramType": param,
        "paramValue": f"{point_estimate}",
        "ciPctValue": "95",
        "ciLowerLimit": f"{ci_lower}",
        "ciUpperLimit": f"{ci_upper}",
        "pValue": "",
    }
    if ni_margin is not None:
        analysis["nonInferiorityMargin"] = f"{ni_margin}"
    measures = []
    if age_mean is not None:
        measures.append(
            {
                "title": "Age",
                "paramType": "MEAN",
                "dispersionType": "Standard Deviation",
                "unitOfMeasure": "years",
                "classes": [
                    {
                        "categories": [
                            {
                                "measurements": [
                                    {"groupId": "BG999", "value": f"{age_mean}", "spread": f"{age_sd}"}
                                ]
                            }
                        ]
                    }
                ],
            }
        )
    if female_percent is not None:
        measures.append(
            {
                "title": "Female",
                "paramType": "MEAN",
                "unitOfMeasure": "%",
                "classes": [
                    {"categories": [{"measurements": [{"groupId": "BG999", "value": f"{female_percent}"}]}]}
                ],
            }
        )
    study["resultsSection"] = {
        "baselineCharacteristicsModule": {
            "groups": [
                {"id": "BG000", "title": arm_titles[0]},
                {"id": "BG001", "title": arm_titles[1]},
                {"id": "BG999", "title": "Total"},
            ],
            "measures": measures,
        },
        "outcomeMeasuresModule": {
            "outcomeMeasures": [
                {
                    "type": "PRIMARY",
                    "title": primary_title,
                    "timeFrame": timeframe,
                    "analyses": [analysis],
                }
            ]
        },
        "participantFlowModule": {
            "groups": [
                {"id": "FG000", "title": arm_titles[0]},
                {"id": "FG001", "title": arm_titles[1]},
            ],
            "periods": [
                {
                    "title": "Overall Study",
                    "milestones": [
                        {
                            "type": "STARTED",
                            "achievements": [
                                {"groupId": "FG000", "numSubjects": str(n_treatment)},
                                {"groupId": "FG001", "numSubjects": str(n_control)},
                            ],
                        }
                    ],
                }
            ],
        },
    }
    return study


OFFLINE_RESULTS_FIXTURES: dict[str, dict] = {
    "NCT01243424": _v2_study(
        nct_id="NCT01243424",
        official_title="CAROLINA: Cardiovascular Outcome Study of Linagliptin Versus Glimepiride in Patients With Type 2 Diabetes",
        has_results=True,
        primary_title="Time to first occurrence of cardiovascular death, non-fatal myocardial infarction, or non-fatal stroke (3P-MACE)",
        timeframe="Median follow-up 6.3 years",
        effect_scale="hr",
        point_estimate=0.98,
        ci_lower=0.84,
        ci_upper=1.14,
        n_treatment=3023,
        n_control=3010,
        age_mean=64.0,
        age_sd=9.5,
        female_percent=39.9,
        ni_margin=1.3,
        arm_titles=("Linagliptin", "Glimepiride"),
    ),
    "NCT02296125": _v2_study(
        nct_id="NCT02296125",
        official_title="FLAURA: Osimertinib Versus Gefitinib or Erlotinib in Previously Untreated EGFR Mutation-Positive Advanced NSCLC",
        has_results=True,
        primary_title="Progression-free survival",
        timeframe="Up to approximately 20 months",
        effect_scale="hr",
        point_estimate=0.46,
        ci_lower=0.37,
        ci_upper=0.57,
        n_treatment=279,
        n_control=277,
        age_mean=64.0,
        age_sd=10.2,
        female_percent=63.0,
        arm_titles=("Osimertinib", "Gefitinib or Erlotinib"),
    ),
    "NCT09990001": _v2_study(
        nct_id="NCT09990001",
        official_title="Ongoing study without a posted ClinicalTrials.gov results section",
        has_results=False,
    ),
    "NCT06642272": _v2_study(
        nct_id="NCT06642272",
        official_title="Illustrative ongoing study fixture with hasResults false",
        has_results=False,
    ),
}

_FIXTURE_CITATIONS = {
    "NCT01243424": "Rosenstock et al. JAMA. 2019;322(12):1155-1166. doi:10.1001/jama.2019.13772",
    "NCT02296125": "Soria et al. N Engl J Med. 2018;378:113-125. doi:10.1056/NEJMoa1713137",
}


def _fetch_live_study(nct_id: str, timeout: float) -> dict:
    url = CTGOV_STUDY_URL.format(nct_id=nct_id)
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "EmulatRx-benchmark/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _validate_effect_bounds(scale: str, point: float, lower: float, upper: float) -> None:
    if not (lower <= point <= upper):
        raise ContractError("INVALID_EFFECT_BOUNDS", "Confidence interval must contain the point estimate")
    if scale in {"hr", "rr"} and (point <= 0.0 or lower <= 0.0 or upper <= 0.0):
        raise ContractError("INVALID_EFFECT_BOUNDS", "Ratio-scale estimates and intervals must be positive")


def _curated_aggregate(summary: dict, nct_id: str | None) -> dict:
    if not isinstance(summary, dict):
        raise ContractError("INVALID_CURATED_SUMMARY", "curated_published_summary must be an object")
    citation = str(summary.get("source_citation") or "").strip()
    if not citation:
        raise ContractError(
            "BENCHMARK_UNAVAILABLE_NO_RESULTS_SECTION",
            "curated_published_summary requires a non-empty source_citation",
        )
    scale = str(summary.get("effect_scale") or "").strip().lower()
    if scale not in {"hr", "rr", "rd"}:
        raise ContractError("INVALID_CURATED_SUMMARY", "effect_scale must be hr, rr, or rd")
    point = _parse_float(summary.get("point_estimate"))
    lower = _parse_float(summary.get("ci_lower"))
    upper = _parse_float(summary.get("ci_upper"))
    if point is None or lower is None or upper is None:
        raise ContractError("INVALID_CURATED_SUMMARY", "point_estimate and CI bounds are required")
    _validate_effect_bounds(scale, point, lower, upper)
    n_treatment = summary.get("n_treatment")
    n_control = summary.get("n_control")
    try:
        n_treatment_i = int(n_treatment)
        n_control_i = int(n_control)
    except (TypeError, ValueError) as exc:
        raise ContractError("INVALID_CURATED_SUMMARY", "n_treatment and n_control must be integers") from exc
    if n_treatment_i <= 0 or n_control_i <= 0:
        raise ContractError("INVALID_CURATED_SUMMARY", "arm sample sizes must be positive")
    label = str(summary.get("primary_outcome_label") or "").strip()
    if not label:
        raise ContractError("INVALID_CURATED_SUMMARY", "primary_outcome_label is required")
    return {
        "nct_id": nct_id or "",
        "has_posted_results": False,
        "usable": True,
        "source_mode": "curated_published_summary",
        "source_citation": citation,
        "official_title": str(summary.get("official_title") or label),
        "primary_outcome_title": label,
        "primary_outcome_timeframe": str(summary.get("primary_outcome_timeframe") or ""),
        "effect_scale": scale,
        "point_estimate": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "ni_margin": _parse_float(summary.get("ni_margin")),
        "n_treatment": n_treatment_i,
        "n_control": n_control_i,
        "baseline_characteristics": summary.get("baseline_characteristics") or {},
        "raw_input": summary,
    }


def resolve_rct_public_results(
    nct_id: str | None = None,
    *,
    study_json: dict | None = None,
    curated_published_summary: dict | None = None,
    allow_live: bool = False,
    live_timeout_seconds: float = 8.0,
    retrieved_at_utc: str | None = None,
) -> dict:
    """Resolve study-level public aggregates. Never reads or stores patient-level data."""
    normalized = _normalize_nct(nct_id) if nct_id else ""
    retrieved = retrieved_at_utc or _utc_now()
    if curated_published_summary is not None:
        try:
            aggregate = _curated_aggregate(curated_published_summary, normalized or None)
        except ContractError as exc:
            if exc.code == "BENCHMARK_UNAVAILABLE_NO_RESULTS_SECTION":
                return _unavailable(normalized, retrieved, exc.code, str(exc))
            raise
        return _receipt_from_aggregate(aggregate, retrieved_at_utc=retrieved, raw_for_hash=curated_published_summary)
    source_mode = "clinicaltrials_results_section"
    live_error = None
    if study_json is None and allow_live and normalized and not _offline_catalog_forced():
        try:
            study_json = _fetch_live_study(normalized, live_timeout_seconds)
            source_mode = "clinicaltrials_results_section"
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            live_error = str(exc)
            study_json = None
    if study_json is None and normalized in OFFLINE_RESULTS_FIXTURES:
        study_json = OFFLINE_RESULTS_FIXTURES[normalized]
        source_mode = "clinicaltrials_results_section"
    if not isinstance(study_json, dict):
        return _unavailable(normalized, retrieved, "BENCHMARK_UNAVAILABLE_NO_RESULTS_SECTION", live_error or "no fixture")
    parsed = parse_clinicaltrials_results_study(study_json)
    if not parsed or not parsed.get("usable"):
        return _unavailable(
            normalized or str((parsed or {}).get("nct_id") or ""),
            retrieved,
            "BENCHMARK_UNAVAILABLE_NO_RESULTS_SECTION",
            "目标 NCT 暂无公开结果摘要（resultsSection 缺失），不可对标，禁止输出一致性结论",
            official_title=str((parsed or {}).get("official_title") or ""),
        )
    if normalized and not parsed.get("nct_id"):
        parsed["nct_id"] = normalized
    parsed["source_mode"] = source_mode
    parsed["source_citation"] = _FIXTURE_CITATIONS.get(parsed.get("nct_id") or normalized, "")
    return _receipt_from_aggregate(parsed, retrieved_at_utc=retrieved, raw_for_hash=study_json)


def _unavailable(nct_id: str, retrieved: str, code: str, detail: str, official_title: str = "") -> dict:
    message = "目标 NCT 暂无公开结果摘要（resultsSection 缺失），不可对标，禁止输出一致性结论"
    return {
        "status": "unavailable_no_public_results",
        "gate_code": code,
        "benchmarkable": False,
        "effect_comparison": None,
        "consistency_conclusion_allowed": False,
        "ipd_used": False,
        "message_zh": message,
        "message_en": "The target NCT has no usable public resultsSection aggregate, so benchmarking and consistency conclusions are forbidden.",
        "detail": detail,
        "source_receipt": {
            "nct_id": nct_id,
            "source_mode": "clinicaltrials_results_section",
            "source_url": CTGOV_STUDY_URL.format(nct_id=nct_id) if nct_id else "",
            "source_citation": "",
            "retrieved_at_utc": retrieved,
            "has_posted_results": False,
            "payload_sha256": _canonical_sha256({"nct_id": nct_id, "has_posted_results": False}),
            "official_title": official_title,
            "primary_outcome_title": "",
            "primary_outcome_timeframe": "",
            "ipd_used": False,
        },
        "interpretive_guardrail": interpretive_guardrail(),
    }


def _receipt_from_aggregate(aggregate: dict, *, retrieved_at_utc: str, raw_for_hash: dict) -> dict:
    nct_id = str(aggregate.get("nct_id") or "")
    hash_body = {
        "effect_scale": aggregate.get("effect_scale"),
        "point_estimate": aggregate.get("point_estimate"),
        "ci_lower": aggregate.get("ci_lower"),
        "ci_upper": aggregate.get("ci_upper"),
        "n_treatment": aggregate.get("n_treatment"),
        "n_control": aggregate.get("n_control"),
        "ni_margin": aggregate.get("ni_margin"),
        "baseline_characteristics": aggregate.get("baseline_characteristics") or {},
        "primary_outcome_title": aggregate.get("primary_outcome_title"),
    }
    receipt = {
        "nct_id": nct_id,
        "source_mode": aggregate.get("source_mode") or "clinicaltrials_results_section",
        "source_url": CTGOV_STUDY_URL.format(nct_id=nct_id) if nct_id else "",
        "source_citation": str(aggregate.get("source_citation") or ""),
        "retrieved_at_utc": retrieved_at_utc,
        "has_posted_results": bool(aggregate.get("has_posted_results")),
        "payload_sha256": _canonical_sha256(hash_body),
        "official_title": str(aggregate.get("official_title") or ""),
        "primary_outcome_title": str(aggregate.get("primary_outcome_title") or ""),
        "primary_outcome_timeframe": str(aggregate.get("primary_outcome_timeframe") or ""),
        "ipd_used": False,
        "raw_payload_sha256": _canonical_sha256(raw_for_hash) if isinstance(raw_for_hash, dict) else "",
    }
    return {
        "status": "resolved",
        "gate_code": None,
        "benchmarkable": True,
        "aggregate": {key: value for key, value in aggregate.items() if key != "raw_input"},
        "source_receipt": receipt,
        "ipd_used": False,
        "interpretive_guardrail": interpretive_guardrail(),
    }


def _shared_baseline_rows(rwd_baseline: dict, rct_baseline: dict) -> list[dict]:
    rows = []
    rct_keys = {str(key).lower(): value for key, value in (rct_baseline or {}).items() if isinstance(value, dict)}
    for key, rwd in (rwd_baseline or {}).items():
        if not isinstance(rwd, dict):
            continue
        rct = rct_keys.get(str(key).lower())
        if not isinstance(rct, dict):
            continue
        if "mean" in rwd and "mean" in rct:
            mean_rwd = _parse_float(rwd.get("mean"))
            mean_rct = _parse_float(rct.get("mean"))
            sd_rwd = _parse_float(rwd.get("sd"))
            sd_rct = _parse_float(rct.get("sd"))
            if None in (mean_rwd, mean_rct, sd_rwd, sd_rct):
                continue
            denom = math.sqrt((sd_rwd**2 + sd_rct**2) / 2.0)
            smd = 0.0 if denom == 0.0 and mean_rwd == mean_rct else (math.inf if denom == 0.0 else (mean_rwd - mean_rct) / denom)
            rows.append(
                {
                    "covariate": str(key),
                    "kind": "continuous",
                    "rwd_mean": mean_rwd,
                    "rwd_sd": sd_rwd,
                    "rct_mean": mean_rct,
                    "rct_sd": sd_rct,
                    "smd_approx": _json_float(smd) if math.isfinite(smd) else None,
                    "transportability_shift_flag": bool(math.isfinite(smd) and abs(smd) >= 0.20) or not math.isfinite(smd),
                }
            )
        elif "proportion" in rwd and "proportion" in rct:
            p_rwd = _parse_float(rwd.get("proportion"))
            p_rct = _parse_float(rct.get("proportion"))
            if p_rwd is None or p_rct is None:
                continue
            denom = math.sqrt((p_rwd * (1.0 - p_rwd) + p_rct * (1.0 - p_rct)) / 2.0)
            smd = 0.0 if denom == 0.0 and p_rwd == p_rct else (math.inf if denom == 0.0 else (p_rwd - p_rct) / denom)
            rows.append(
                {
                    "covariate": str(key),
                    "kind": "binary",
                    "rwd_proportion": p_rwd,
                    "rct_proportion": p_rct,
                    "smd_approx": _json_float(smd) if math.isfinite(smd) else None,
                    "transportability_shift_flag": bool(math.isfinite(smd) and abs(smd) >= 0.20) or not math.isfinite(smd),
                }
            )
    return rows


def compute_baseline_comparability(rwd_baseline: dict | None, rct_baseline: dict | None) -> dict:
    """Approximate RWD-versus-RCT baseline SMD. This is transportability, not arm confounding."""
    rows = _shared_baseline_rows(rwd_baseline or {}, rct_baseline or {})
    finite = [abs(row["smd_approx"]) for row in rows if row.get("smd_approx") is not None]
    shifted = sum(1 for row in rows if row["transportability_shift_flag"])
    return {
        "rows": rows,
        "max_abs_baseline_smd_approx": max(finite) if finite else None,
        "n_shifted_covariates": shifted,
        "shift_threshold": 0.20,
        "interpretation_zh": (
            "RWD 与 RCT 基线差异反映目标人群可运输性或效应修饰，而不是试验内部治疗组间混杂。"
            "绝对近似 SMD ≥ 0.20 标记为 transportability_shift_flag。"
        ),
        "interpretation_en": (
            "An RWD–RCT baseline shift reflects target-population transportability or effect modification, "
            "not internal treatment-arm confounding. Rows with absolute approximate SMD ≥ 0.20 are flagged."
        ),
    }


def _effect_triple(block: dict) -> tuple[float, float, float] | None:
    point = _parse_float(block.get("point_estimate"))
    lower = _parse_float(block.get("ci_lower"))
    upper = _parse_float(block.get("ci_upper"))
    if point is None or lower is None or upper is None:
        return None
    return point, lower, upper


def extract_tte_effects(tte_result: dict) -> dict[str, tuple[float, float, float]]:
    found: dict[str, tuple[float, float, float]] = {}
    nested = tte_result.get("effects") if isinstance(tte_result.get("effects"), dict) else {}
    for scale in ("hr", "rr", "rd"):
        block = None
        if isinstance(tte_result.get(scale), dict):
            block = tte_result[scale]
        elif isinstance(nested.get(scale), dict):
            block = nested[scale]
        if block is not None:
            triple = _effect_triple(block)
            if triple is not None:
                found[scale] = triple
    primary = str(tte_result.get("effect_scale") or "").lower()
    if primary in {"hr", "rr", "rd"} and "point_estimate" in tte_result:
        triple = _effect_triple(tte_result)
        if triple is not None:
            found[primary] = triple
    return found


def _theta_se(scale: str, point: float, lower: float, upper: float) -> tuple[float, float]:
    _validate_effect_bounds(scale, point, lower, upper)
    if scale in {"hr", "rr"}:
        theta = math.log(point)
        se = (math.log(upper) - math.log(lower)) / (2.0 * Z_975)
    else:
        theta = point
        se = (upper - lower) / (2.0 * Z_975)
    if not math.isfinite(se) or se <= 0.0:
        raise ContractError("INVALID_EFFECT_BOUNDS", "Standard error must be positive")
    return theta, se


def _significance_direction(point: float, lower: float, upper: float, null: float) -> str:
    if upper < null:
        return "significant_below_null"
    if lower > null:
        return "significant_above_null"
    return "nonsignificant"


def compute_rct_duplicate_metrics(
    *,
    scale: str,
    rwd_point: float,
    rwd_lower: float,
    rwd_upper: float,
    rct_point: float,
    rct_lower: float,
    rct_upper: float,
    ni_margin: float | None = None,
) -> dict:
    """RCT-DUPLICATE metrics (Franklin et al., Circulation 2021) plus the clinical-scale difference."""
    if scale not in {"hr", "rr", "rd"}:
        raise ContractError("BENCHMARK_SCALE_MISMATCH", f"Unsupported effect scale {scale}")
    theta_rwd, se_rwd = _theta_se(scale, rwd_point, rwd_lower, rwd_upper)
    theta_rct, se_rct = _theta_se(scale, rct_point, rct_lower, rct_upper)
    se_diff = math.sqrt(se_rwd**2 + se_rct**2)
    z_stat = (theta_rwd - theta_rct) / se_diff
    p_difference = float(2.0 * norm.sf(abs(z_stat)))
    half = PI_Z * se_diff
    if scale in {"hr", "rr"}:
        pi = [math.exp(theta_rct - half), math.exp(theta_rct + half)]
        within_pi = bool(pi[0] <= rwd_point <= pi[1])
    else:
        pi = [theta_rct - half, theta_rct + half]
        within_pi = bool(pi[0] <= rwd_point <= pi[1])
    null = 1.0 if scale in {"hr", "rr"} else 0.0
    rwd_dir = _significance_direction(rwd_point, rwd_lower, rwd_upper, null)
    rct_dir = _significance_direction(rct_point, rct_lower, rct_upper, null)
    ni_block = None
    if ni_margin is not None:
        margin = float(ni_margin)
        rwd_ni = bool(rwd_upper < margin)
        rct_ni = bool(rct_upper < margin)
        ni_block = {
            "ni_margin": margin,
            "rule": "ci_upper < ni_margin",
            "rwd_meets_ni": rwd_ni,
            "rct_meets_ni": rct_ni,
            "agree": rwd_ni == rct_ni,
        }
    return {
        "effect_scale": scale,
        "method": "RCT-DUPLICATE",
        "citation": "Franklin JM, Patorno E, Desai RJ, et al. Circulation. 2021;143:1002-1013.",
        "rwd": {"point_estimate": rwd_point, "ci_lower": rwd_lower, "ci_upper": rwd_upper, "theta": theta_rwd, "se": se_rwd},
        "rct": {"point_estimate": rct_point, "ci_lower": rct_lower, "ci_upper": rct_upper, "theta": theta_rct, "se": se_rct},
        "clinical_scale_difference": _json_float(rwd_point - rct_point),
        "standardized_difference_z": _json_float(z_stat),
        "p_difference": p_difference,
        "standardized_difference_within_196": bool(abs(z_stat) < Z_975),
        "prediction_interval_95": [_json_float(pi[0]), _json_float(pi[1])],
        "estimate_within_95_pi": within_pi,
        "regulatory_direction_agreement": {
            "null_value": null,
            "rwd_direction": rwd_dir,
            "rct_direction": rct_dir,
            "agree": rwd_dir == rct_dir,
            "non_inferiority_agreement": ni_block,
        },
        "interpretive_guardrail": interpretive_guardrail(),
    }


def _weighted_arm_variance(outcome: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ContractError("INVALID_WEIGHTS", "Arm weight sum must be positive")
    estimate = float(np.sum(weights * outcome) / total)
    variance = float(np.sum(weights**2 * (outcome - estimate) ** 2) / (total**2))
    return estimate, variance


def _interval_excludes(lower: float, upper: float, null: float) -> bool:
    return bool(upper < null or lower > null)


def _rr_block(p1: float, v1: float, p0: float, v0: float) -> dict:
    rr = p1 / p0
    var_log = v1 / (p1**2) + v0 / (p0**2)
    se = math.sqrt(var_log)
    z_stat = math.log(rr) / se
    lower = math.exp(math.log(rr) - Z_975 * se)
    upper = math.exp(math.log(rr) + Z_975 * se)
    return {
        "estimate": _json_float(rr),
        "se_log": _json_float(se),
        "ci_lower": _json_float(lower),
        "ci_upper": _json_float(upper),
        "z": _json_float(z_stat),
        "p_value": float(2.0 * norm.sf(abs(z_stat))),
        "null_value": 1.0,
        "ci_excludes_null": _interval_excludes(lower, upper, 1.0),
    }


def _rd_block(p1: float, v1: float, p0: float, v0: float) -> dict:
    rd = p1 - p0
    se = math.sqrt(v1 + v0)
    if se <= 0.0:
        raise ContractError("INVALID_EFFECT_BOUNDS", "NCO risk-difference SE must be positive")
    z_stat = rd / se
    lower = rd - Z_975 * se
    upper = rd + Z_975 * se
    return {
        "estimate": _json_float(rd),
        "se": _json_float(se),
        "ci_lower": _json_float(lower),
        "ci_upper": _json_float(upper),
        "z": _json_float(z_stat),
        "p_value": float(2.0 * norm.sf(abs(z_stat))),
        "null_value": 0.0,
        "ci_excludes_null": _interval_excludes(lower, upper, 0.0),
    }


def _invalid_nco(detail: str) -> dict:
    return {
        "status": "invalid_negative_control",
        "gate_code": "INVALID_NEGATIVE_CONTROL_COLUMN",
        "residual_confounding_alert": False,
        "inconclusive_warning": False,
        "alert_code": None,
        "p_nco": None,
        "ci_excludes_null": False,
        "message_zh": detail,
        "message_en": detail,
        "interpretive_guardrail": interpretive_guardrail(),
    }


def _binary_01(series: pd.Series) -> np.ndarray | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        return None
    if not np.all(np.isin(values, (0.0, 1.0))):
        return None
    return values.astype(np.int64)


def evaluate_negative_control_outcome(
    cohort: pd.DataFrame,
    *,
    treatment_col: str,
    weights,
    negative_control_outcome_col: str,
    outcome_col: str | None = None,
    negative_control_time_col: str | None = None,
    horizon_time: float | None = None,
) -> dict:
    """Weighted NCO residual-confounding probe (Lipsitch et al., Epidemiology 2010).

    Sparse events stay inconclusive. A null NCO is never reported as proof of no bias.
    """
    if not isinstance(cohort, pd.DataFrame):
        raise ContractError("INVALID_NEGATIVE_CONTROL_COLUMN", "cohort must be a DataFrame")
    if _blank(negative_control_outcome_col) or negative_control_outcome_col not in cohort.columns:
        return _invalid_nco("阴性对照列缺失。")
    if negative_control_outcome_col == treatment_col or (
        outcome_col and negative_control_outcome_col == outcome_col
    ):
        return _invalid_nco("阴性对照列不能与治疗列或主要结局列相同。")
    if treatment_col not in cohort.columns:
        return _invalid_nco("治疗列缺失。")
    treatment = _binary_01(cohort[treatment_col])
    outcome = _binary_01(cohort[negative_control_outcome_col])
    if treatment is None or outcome is None:
        return _invalid_nco("阴性对照或治疗列含有 {0, 1} 以外的取值。")
    weight = np.asarray(weights, dtype=float)
    if weight.shape[0] != len(cohort) or not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
        raise ContractError("INVALID_WEIGHTS", "weights must be finite and strictly positive, one per row")
    events_1 = int(np.sum((treatment == 1) & (outcome == 1)))
    events_0 = int(np.sum((treatment == 0) & (outcome == 1)))
    if events_1 < 5 or events_0 < 5 or not np.any(treatment == 1) or not np.any(treatment == 0):
        return {
            "status": "insufficient_events_inconclusive",
            "gate_code": None,
            "residual_confounding_alert": False,
            "inconclusive_warning": True,
            "alert_code": None,
            "alert_severity": None,
            "p_nco": None,
            "ci_excludes_null": False,
            "events_treated": events_1,
            "events_control": events_0,
            "message_zh": "任一治疗组阴性对照事件数少于 5，结果无结论，不得声称队列干净或没有偏倚。",
            "message_en": "Fewer than 5 negative-control events in an arm. The probe is inconclusive and is not evidence of no bias.",
            "interpretive_guardrail": interpretive_guardrail(),
        }
    mask1 = treatment == 1
    mask0 = treatment == 0
    p1, v1 = _weighted_arm_variance(outcome[mask1].astype(float), weight[mask1])
    p0, v0 = _weighted_arm_variance(outcome[mask0].astype(float), weight[mask0])
    rd = _rd_block(p1, v1, p0, v0)
    rr = _rr_block(p1, v1, p0, v0) if p1 > 0.0 and p0 > 0.0 else None
    hr = None
    if negative_control_time_col:
        if negative_control_time_col not in cohort.columns:
            return _invalid_nco("阴性对照时间列缺失。")
        times = pd.to_numeric(cohort[negative_control_time_col], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(times)) or np.any(times <= 0.0):
            return _invalid_nco("阴性对照时间必须为正数。")
        event = outcome.astype(np.int64)
        if horizon_time is not None:
            horizon = float(horizon_time)
            event = np.where(times <= horizon, event, 0).astype(np.int64)
            times = np.minimum(times, horizon)
        fit = fit_weighted_cause_specific_cox(
            times,
            event,
            treatment.astype(np.int64),
            2,
            weight,
            cause_code=1,
        )
        if fit.get("converged") and fit.get("hr") and fit["hr"][0] is not None and fit["se_robust"][0]:
            hr = {
                "estimate": _json_float(fit["hr"][0]),
                "ci_lower": _json_float(fit["ci_lower"][0]),
                "ci_upper": _json_float(fit["ci_upper"][0]),
                "se_log": _json_float(fit["se_robust"][0]),
                "z": _json_float(math.log(fit["hr"][0]) / fit["se_robust"][0]),
                "p_value": float(2.0 * norm.sf(abs(math.log(fit["hr"][0]) / fit["se_robust"][0]))),
                "null_value": 1.0,
                "ci_excludes_null": _interval_excludes(fit["ci_lower"][0], fit["ci_upper"][0], 1.0),
                "variance": "lin_wei_robust_sandwich",
            }
    if hr is not None:
        primary = hr
        primary_name = "hr"
    elif rr is not None:
        primary = rr
        primary_name = "rr"
    else:
        primary = rd
        primary_name = "rd"
    alert = bool(primary["ci_excludes_null"])
    message_zh = (
        "预先指定的阴性对照在加权后仍显著偏离无效值（即使观察协变量加权后 |SMD| 可以小于 0.10 甚至 0.02）。"
        "这提示可能存在未测量混杂（例如健康使用者、衰弱或监测偏倚），主要 TTE 估计因此不可靠。"
        if alert
        else "阴性对照的加权 95% 区间覆盖无效值。这只说明该探针未报警，不能解释为没有残余混杂。"
    )
    message_en = (
        "The prespecified negative control shifted after weighting even if observed covariates achieved max |SMD| < 0.10 or < 0.02. "
        "Hidden confounding (healthy-user, frailty, or surveillance bias) likely compromises the primary TTE estimate."
        if alert
        else "The weighted negative-control interval covers the null. That is not evidence that residual confounding is absent."
    )
    return {
        "status": "evaluated",
        "gate_code": "NEGATIVE_CONTROL_BIAS_DETECTED" if alert else None,
        "residual_confounding_alert": alert,
        "inconclusive_warning": False,
        "alert_code": "NEGATIVE_CONTROL_BIAS_DETECTED" if alert else None,
        "alert_severity": "high" if alert else None,
        "primary_estimand": primary_name,
        "estimate": primary["estimate"],
        "ci_lower": primary["ci_lower"],
        "ci_upper": primary["ci_upper"],
        "p_nco": primary["p_value"],
        "null_value": primary["null_value"],
        "ci_excludes_null": primary["ci_excludes_null"],
        "weighted_risk_treated": _json_float(p1),
        "weighted_risk_control": _json_float(p0),
        "weighted_rd": rd,
        "weighted_rr": rr,
        "weighted_hr": hr,
        "events_treated": events_1,
        "events_control": events_0,
        "citation": "Lipsitch M, Tchetgen Tchetgen E, Cohen T. Epidemiology. 2010;21(3):383-388.",
        "message_zh": message_zh,
        "message_en": message_en,
        "interpretive_guardrail": interpretive_guardrail(),
    }


def generate_hidden_confounding_benchmark_cohort(
    *,
    n: int = 2500,
    seed: int = 20261125,
    hidden_confounding_strength: float = 1.2,
) -> dict:
    """Seeded cohort whose observed covariates are balanceable while U still confounds.

    Treatment and both outcomes depend on hidden ``U`` when strength is positive. The structural
    coefficient of treatment on the primary hazard and on the negative control is exactly zero.
    ``U`` is returned only inside ``ground_truth_audit``, never as an analysis covariate.
    """
    if n < 50:
        raise ContractError("INVALID_COHORT_SIZE", "n must be at least 50")
    strength = float(hidden_confounding_strength)
    rng = np.random.default_rng(int(seed))
    age = rng.normal(64.0, 9.0, n)
    bmi = rng.normal(28.0, 4.5, n)
    prior = rng.binomial(1, 0.32, n).astype(float)
    hidden_u = rng.normal(0.0, 1.0, n)
    logit = (
        0.25 * ((age - 64.0) / 9.0)
        + 0.20 * ((bmi - 28.0) / 4.5)
        + 0.45 * prior
        + strength * 2.4 * hidden_u
    )
    treatment = rng.binomial(1, expit(logit)).astype(np.int64)
    primary_hazard = np.exp(-1.7 - strength * 0.7 * hidden_u + 0.12 * prior)
    primary_draw = rng.exponential(1.0 / primary_hazard)
    primary_horizon = 4.0
    outcome_event = (primary_draw <= primary_horizon).astype(np.int64)
    outcome_time = np.maximum(np.minimum(primary_draw, primary_horizon), 1e-3)
    nco_hazard = np.exp(-0.85 - strength * 1.15 * hidden_u)
    nco_draw = rng.exponential(1.0 / nco_hazard)
    nco_horizon = 2.5
    nco_event = (nco_draw <= nco_horizon).astype(np.int64)
    nco_time = np.maximum(np.minimum(nco_draw, nco_horizon), 1e-3)
    cohort = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "prior_cvd": prior.astype(np.int64),
            "treatment": treatment,
            "outcome_event": outcome_event,
            "outcome_time": outcome_time,
            "negative_control_outcome": nco_event,
            "negative_control_time": nco_time,
        }
    )
    return {
        "cohort": cohort,
        "observed_covariates": list(OBSERVED_COVARIATES),
        "treatment_col": "treatment",
        "outcome_col": "outcome_event",
        "negative_control_outcome_col": "negative_control_outcome",
        "negative_control_time_col": "negative_control_time",
        "ground_truth_audit": {
            "hidden_confounder": "U",
            "hidden_u": hidden_u,
            "true_hr_treatment_on_primary": 1.0,
            "true_hr_treatment_on_nco": 1.0,
            "true_rd_treatment_on_nco": 0.0,
            "hidden_confounding_strength": strength,
            "seed": int(seed),
            "n": int(n),
            "observed_covariates": list(OBSERVED_COVARIATES),
            "note": "U is not part of the observed analysis covariate list.",
        },
    }


def _weighted_baseline(cohort: pd.DataFrame, weights: np.ndarray) -> dict:
    total = float(np.sum(weights))
    baseline = {}
    for column, kind in (("age", "continuous"), ("bmi", "continuous"), ("prior_cvd", "binary")):
        values = cohort[column].to_numpy(dtype=float)
        mean = float(np.sum(weights * values) / total)
        if kind == "continuous":
            variance = float(np.sum(weights * (values - mean) ** 2) / total)
            baseline[column] = {"mean": mean, "sd": math.sqrt(max(variance, 0.0))}
        else:
            baseline[column] = {"proportion": mean}
    return baseline


def _fit_primary_hr(cohort: pd.DataFrame, weights: np.ndarray) -> dict:
    fit = fit_weighted_cause_specific_cox(
        cohort["outcome_time"].to_numpy(dtype=float),
        cohort["outcome_event"].to_numpy(dtype=int),
        cohort["treatment"].to_numpy(dtype=int),
        2,
        weights,
        cause_code=1,
    )
    if not fit.get("converged") or fit["hr"][0] is None:
        raise ContractError("PRIMARY_HR_UNAVAILABLE", "Weighted primary Cox model did not converge in the synthetic demo")
    return {
        "gate_passed": True,
        "effect_scale": "hr",
        "point_estimate": float(fit["hr"][0]),
        "ci_lower": float(fit["ci_lower"][0]),
        "ci_upper": float(fit["ci_upper"][0]),
    }


def run_synthetic_benchmark_demo(
    mode: str,
    nct_id: str = "NCT01243424",
    *,
    n: int = 2500,
    seed: int = 20261125,
    retrieved_at_utc: str | None = None,
) -> dict:
    if mode not in {"hidden_bias", "clean_null"}:
        raise ContractError("INVALID_SYNTHETIC_DEMO", "synthetic demo must be hidden_bias or clean_null")
    strength = 1.2 if mode == "hidden_bias" else 0.0
    generated = generate_hidden_confounding_benchmark_cohort(n=n, seed=seed, hidden_confounding_strength=strength)
    cohort = generated["cohort"]
    design = cohort[list(OBSERVED_COVARIATES)].to_numpy(dtype=float)
    arms = cohort["treatment"].to_numpy(dtype=int)
    fitted = compute_weights(design, arms, ["control", "treated"], weight_method="overlap")
    weights = np.asarray(fitted.analysis_weights, dtype=float)
    tte_result = _fit_primary_hr(cohort, weights)
    tte_result["baseline_characteristics"] = _weighted_baseline(cohort, weights)
    alignment = {
        "overall_alignment": "aligned",
        "population_aligned": True,
        "arms_aligned": True,
        "outcome_aligned": True,
        "timeframe_aligned": True,
    }
    report = run_rct_benchmark_and_nco(
        tte_result,
        alignment,
        nct_id=nct_id,
        cohort=cohort,
        treatment_col="treatment",
        outcome_col="outcome_event",
        weights=weights,
        negative_control_outcome_col="negative_control_outcome",
        negative_control_time_col="negative_control_time",
        retrieved_at_utc=retrieved_at_utc,
    )
    audit = generated["ground_truth_audit"]
    report["synthetic_demo"] = {
        "mode": mode,
        "seed": int(seed),
        "n": int(n),
        "hidden_confounding_strength": strength,
        "true_hr_treatment_on_primary": audit["true_hr_treatment_on_primary"],
        "true_rd_treatment_on_nco": audit["true_rd_treatment_on_nco"],
        "observed_covariates": list(OBSERVED_COVARIATES),
        "hidden_confounder_excluded_from_analysis": True,
    }
    return report


def _rwd_baseline(tte_result: dict) -> dict:
    for key in ("baseline_characteristics", "weighted_baseline"):
        block = tte_result.get(key)
        if isinstance(block, dict):
            return block
    return {}


def _scale_mismatch(message: str) -> dict:
    return {
        "status": "scale_mismatch",
        "gate_code": "BENCHMARK_SCALE_MISMATCH",
        "benchmarkable": False,
        "effect_comparison": None,
        "consistency_conclusion_allowed": False,
        "ipd_used": False,
        "message_zh": "RWD 与公开 RCT 的效应尺度不一致，且 TTE 结果中没有可配对的同一尺度，已失败关闭。",
        "message_en": message,
        "interpretive_guardrail": interpretive_guardrail(),
    }


def _attach_nco(report: dict, nco: dict | None) -> dict:
    report["negative_control"] = nco
    alert = bool(nco and nco.get("residual_confounding_alert"))
    if alert:
        report["overall_validity_flag"] = "high_residual_bias_risk"
        report["consistency_conclusion_allowed"] = False
    elif report.get("effect_comparison") is not None and nco and nco.get("status") == "invalid_negative_control":
        report["overall_validity_flag"] = "not_interpretable"
        report["consistency_conclusion_allowed"] = False
    elif report.get("effect_comparison") is not None:
        report["overall_validity_flag"] = report.get("overall_validity_flag") or "numerical_proximity_only"
    else:
        report["overall_validity_flag"] = "not_interpretable"
    report["interpretive_guardrail"] = interpretive_guardrail()
    report["ipd_used"] = False
    report["numerical_proximity_proves_causality"] = False
    return report


def run_rct_benchmark_and_nco(
    tte_result: dict,
    alignment_audit: dict,
    *,
    nct_id: str | None = None,
    study_json: dict | None = None,
    curated_published_summary: dict | None = None,
    cohort: pd.DataFrame | None = None,
    treatment_col: str = "treatment",
    outcome_col: str | None = None,
    weights=None,
    negative_control_outcome_col: str | None = None,
    negative_control_time_col: str | None = None,
    horizon_time: float | None = None,
    allow_live: bool = False,
    retrieved_at_utc: str | None = None,
    confirm_protocol_mismatch: bool = False,
    confirm_arm_mismatch: bool = False,
) -> dict:
    """Benchmark a finished TTE against public RCT aggregates and optionally probe an NCO.

    ``confirm_protocol_mismatch`` and ``confirm_arm_mismatch`` are accepted and deliberately ignored.
    """
    del confirm_protocol_mismatch, confirm_arm_mismatch
    eligibility = audit_benchmark_eligibility(tte_result, alignment_audit)
    if eligibility["status"] != "eligible":
        eligibility["schema_version"] = SCHEMA_VERSION
        eligibility["overall_validity_flag"] = "not_interpretable"
        return eligibility
    try:
        resolved = resolve_rct_public_results(
            nct_id,
            study_json=study_json,
            curated_published_summary=curated_published_summary,
            allow_live=allow_live,
            retrieved_at_utc=retrieved_at_utc,
        )
    except ContractError as exc:
        if exc.code == "BENCHMARK_UNAVAILABLE_NO_RESULTS_SECTION":
            blocked = _unavailable(str(nct_id or ""), retrieved_at_utc or _utc_now(), exc.code, str(exc))
            blocked["schema_version"] = SCHEMA_VERSION
            blocked["overall_validity_flag"] = "not_interpretable"
            return blocked
        raise
    if resolved["status"] != "resolved":
        resolved["schema_version"] = SCHEMA_VERSION
        resolved["overall_validity_flag"] = "not_interpretable"
        return resolved
    aggregate = resolved["aggregate"]
    rct_scale = str(aggregate["effect_scale"])
    available = extract_tte_effects(tte_result)
    if rct_scale not in available:
        mismatch = _scale_mismatch(
            f"RCT reports {rct_scale} but the TTE result has scales {sorted(available)}"
        )
        mismatch["source_receipt"] = resolved["source_receipt"]
        mismatch["schema_version"] = SCHEMA_VERSION
        mismatch["overall_validity_flag"] = "not_interpretable"
        return mismatch
    rwd_point, rwd_lower, rwd_upper = available[rct_scale]
    try:
        comparison = compute_rct_duplicate_metrics(
            scale=rct_scale,
            rwd_point=rwd_point,
            rwd_lower=rwd_lower,
            rwd_upper=rwd_upper,
            rct_point=float(aggregate["point_estimate"]),
            rct_lower=float(aggregate["ci_lower"]),
            rct_upper=float(aggregate["ci_upper"]),
            ni_margin=aggregate.get("ni_margin"),
        )
    except ContractError as exc:
        if exc.code == "INVALID_EFFECT_BOUNDS":
            invalid = {
                "status": "invalid_effect_bounds",
                "gate_code": "BENCHMARK_INVALID_EFFECT_BOUNDS",
                "benchmarkable": False,
                "effect_comparison": None,
                "consistency_conclusion_allowed": False,
                "ipd_used": False,
                "source_receipt": resolved["source_receipt"],
                "message_zh": "效应区间无效或标准误不是正数，已失败关闭。",
                "message_en": str(exc),
                "schema_version": SCHEMA_VERSION,
                "overall_validity_flag": "not_interpretable",
                "interpretive_guardrail": interpretive_guardrail(),
            }
            return invalid
        raise
    baseline = compute_baseline_comparability(_rwd_baseline(tte_result), aggregate.get("baseline_characteristics") or {})
    nco = None
    if negative_control_outcome_col:
        if cohort is None or weights is None:
            nco = _invalid_nco("已请求阴性对照，但没有可用的队列与阳性权重。")
        else:
            nco = evaluate_negative_control_outcome(
                cohort,
                treatment_col=treatment_col,
                weights=weights,
                negative_control_outcome_col=negative_control_outcome_col,
                outcome_col=outcome_col,
                negative_control_time_col=negative_control_time_col,
                horizon_time=horizon_time,
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "benchmarked",
        "gate_code": None,
        "benchmarkable": True,
        "effect_comparison": comparison,
        "baseline_comparability": baseline,
        "source_receipt": resolved["source_receipt"],
        "consistency_conclusion_allowed": nco is None or not nco.get("residual_confounding_alert"),
        "overall_validity_flag": "numerical_proximity_only",
        "ipd_used": False,
        "numerical_proximity_proves_causality": False,
        "interpretive_guardrail": interpretive_guardrail(),
        "message_zh": "已完成研究层面的统计对标。数值接近不是因果或临床有效性的证明。",
        "message_en": "Study-level statistical benchmarking completed. Proximity is not causal or clinical-efficacy evidence.",
    }
    if nco and nco.get("residual_confounding_alert"):
        report["consistency_conclusion_allowed"] = False
    return _attach_nco(report, nco)


def tte_result_from_run_record(run_record: dict) -> tuple[dict, dict]:
    """Build a study-level TTE effect dict from a completed run summary. Drops row samples."""
    gate_passed = run_record.get("gate_passed") is True or run_record.get("gate_status") == "passed"
    estimates = run_record.get("estimates") if isinstance(run_record.get("estimates"), dict) else {}
    table = estimates.get("co_presented_comparison_table") or []
    row = table[0] if table else {}
    effects = {}
    if row.get("cshr_primary") is not None and row.get("cshr_primary_ci_lower") is not None:
        effects["hr"] = {
            "point_estimate": row["cshr_primary"],
            "ci_lower": row["cshr_primary_ci_lower"],
            "ci_upper": row["cshr_primary_ci_upper"],
        }
    if row.get("rr_cif_primary") is not None and row.get("rr_cif_primary_ci_lower") is not None:
        effects["rr"] = {
            "point_estimate": row["rr_cif_primary"],
            "ci_lower": row["rr_cif_primary_ci_lower"],
            "ci_upper": row["rr_cif_primary_ci_upper"],
        }
    if row.get("rd_primary") is not None and row.get("rd_primary_ci_lower") is not None:
        effects["rd"] = {
            "point_estimate": row["rd_primary"],
            "ci_lower": row["rd_primary_ci_lower"],
            "ci_upper": row["rd_primary_ci_upper"],
        }
    primary = "hr" if "hr" in effects else ("rr" if "rr" in effects else ("rd" if "rd" in effects else None))
    tte_result = {"gate_passed": gate_passed, "effects": effects}
    if primary:
        tte_result["effect_scale"] = primary
        tte_result.update(effects[primary])
    target = run_record.get("target_rct") if isinstance(run_record.get("target_rct"), dict) else {}
    alignment = {key: target[key] for key in target if "align" in key or key in {"population_alignment", "outcome_alignment", "followup_alignment", "arm_alignment_status", "overall_alignment_status"}}
    return tte_result, alignment


def execute_benchmark_request(payload: dict, *, run_record: dict | None = None, cohort: pd.DataFrame | None = None, weights=None) -> dict:
    """Service entry. Rejects embedded patient-level tables."""
    if not isinstance(payload, dict):
        raise ContractError("INVALID_JSON", "Benchmark body must be an object")
    forbidden = [key for key in ("cohort_rows", "patients", "ipd", "records", "subject_rows") if key in payload]
    if forbidden:
        raise ContractError("IPD_NOT_ACCEPTED", "Patient-level records are not accepted by the benchmark endpoint")
    if payload.get("synthetic_demo"):
        return run_synthetic_benchmark_demo(
            str(payload["synthetic_demo"]),
            nct_id=str(payload.get("nct_id") or "NCT01243424"),
            n=int(payload.get("n") or 2500),
            seed=int(payload.get("seed") or 20261125),
        )
    tte_result = payload.get("tte_result") if isinstance(payload.get("tte_result"), dict) else None
    alignment = payload.get("alignment_audit") if isinstance(payload.get("alignment_audit"), dict) else None
    if run_record is not None:
        derived_tte, derived_alignment = tte_result_from_run_record(run_record)
        tte_result = tte_result or derived_tte
        alignment = alignment or derived_alignment
    if tte_result is None or alignment is None:
        raise ContractError("MISSING_BENCHMARK_INPUT", "Provide tte_result and alignment_audit, a run_id, or synthetic_demo")
    nco_col = payload.get("negative_control_outcome_col") or None
    return run_rct_benchmark_and_nco(
        tte_result,
        alignment,
        nct_id=payload.get("nct_id"),
        study_json=payload.get("study_json") if isinstance(payload.get("study_json"), dict) else None,
        curated_published_summary=payload.get("curated_published_summary") if isinstance(payload.get("curated_published_summary"), dict) else None,
        cohort=cohort,
        treatment_col=str(payload.get("treatment_col") or "treatment"),
        outcome_col=payload.get("outcome_col"),
        weights=weights,
        negative_control_outcome_col=str(nco_col) if nco_col else None,
        negative_control_time_col=payload.get("negative_control_time_col"),
        allow_live=bool(payload.get("allow_live")),
        confirm_protocol_mismatch=bool(payload.get("confirm_protocol_mismatch")),
        confirm_arm_mismatch=bool(payload.get("confirm_arm_mismatch")),
    )


def render_benchmark_report_html(report: dict) -> str:
    """Self-contained bilingual HTML. No external assets and no patient rows."""
    guard = report.get("interpretive_guardrail") or interpretive_guardrail()
    nco = report.get("negative_control") or {}
    alert = bool(nco.get("residual_confounding_alert")) or report.get("overall_validity_flag") == "high_residual_bias_risk"
    blocked = report.get("status") != "benchmarked"
    red = ""
    if alert or blocked:
        red = (
            "<section class='banner red'><strong>失败关闭 / 偏倚警报</strong><p>"
            + html.escape(str(report.get("message_zh") or nco.get("message_zh") or report.get("gate_code") or ""))
            + "</p><p>"
            + html.escape(str(nco.get("message_en") or report.get("message_en") or ""))
            + "</p></section>"
        )
    receipt = report.get("source_receipt") or {}
    receipt_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(receipt.get(key, '')))}</td></tr>"
        for key in (
            "nct_id",
            "source_mode",
            "source_url",
            "source_citation",
            "retrieved_at_utc",
            "has_posted_results",
            "payload_sha256",
            "official_title",
            "primary_outcome_title",
            "primary_outcome_timeframe",
            "ipd_used",
        )
    )
    baseline = (report.get("baseline_comparability") or {}).get("rows") or []
    baseline_body = "".join(
        "<tr><td>{cov}</td><td>{smd}</td><td>{flag}</td></tr>".format(
            cov=html.escape(str(row.get("covariate"))),
            smd=html.escape(str(row.get("smd_approx"))),
            flag="是" if row.get("transportability_shift_flag") else "否",
        )
        for row in baseline
    ) or "<tr><td colspan='3'>无可对照的研究层面基线摘要</td></tr>"
    comparison = report.get("effect_comparison") or {}
    direction = comparison.get("regulatory_direction_agreement") or {}
    metrics = (
        f"<p>standardized_difference_z = {html.escape(str(comparison.get('standardized_difference_z')))}</p>"
        f"<p>p_difference = {html.escape(str(comparison.get('p_difference')))}</p>"
        f"<p>within |z|&lt;1.96: {html.escape(str(comparison.get('standardized_difference_within_196')))}</p>"
        f"<p>95% prediction interval: {html.escape(str(comparison.get('prediction_interval_95')))}</p>"
        f"<p>estimate_within_95_pi: {html.escape(str(comparison.get('estimate_within_95_pi')))}</p>"
        f"<p>regulatory direction agree: {html.escape(str(direction.get('agree')))}</p>"
        f"<p>non-inferiority: {html.escape(str(direction.get('non_inferiority_agreement')))}</p>"
        if comparison
        else "<p>effect_comparison = null（一致性结论已禁止）</p>"
    )
    nco_card = (
        f"<p>status = {html.escape(str(nco.get('status')))}</p>"
        f"<p>alert_code = {html.escape(str(nco.get('alert_code')))}</p>"
        f"<p>p_nco = {html.escape(str(nco.get('p_nco')))}</p>"
        f"<p>95% CI = [{html.escape(str(nco.get('ci_lower')))}, {html.escape(str(nco.get('ci_upper')))}]</p>"
        f"<p>{html.escape(str(nco.get('message_zh') or ''))}</p>"
        if nco
        else "<p>未请求阴性对照。</p>"
    )
    return f"""<!DOCTYPE html>
<html lang="zh-Hans"><head><meta charset="utf-8"><title>EmulatRx RCT 对标与阴性对照</title>
<style>
body{{font-family:Georgia,serif;margin:32px;color:#1c1917;background:#fafaf9}}
h1{{font-size:1.6rem}} h2{{font-size:1.15rem;margin-top:1.4rem}}
.banner{{padding:14px 16px;border-radius:8px;margin:12px 0}}
.amber{{background:#fffbeb;border:1px solid #d97706}}
.red{{background:#fef2f2;border:1px solid #b91c1c}}
table{{border-collapse:collapse;width:100%;background:#fff}}
th,td{{border:1px solid #d6d3d1;padding:6px 8px;text-align:left;vertical-align:top}}
.card{{background:#fff;border:1px solid #d6d3d1;border-radius:8px;padding:12px 14px;margin:10px 0}}
code{{font-family:ui-monospace,monospace}}
</style></head><body>
<h1>目标 RCT 公开结果对标与阴性对照诊断</h1>
<section class="banner amber"><strong>非因果护栏</strong><p>{html.escape(guard['summary_zh'])}</p><p>{html.escape(guard['summary_en'])}</p>
<p>numerical_proximity_proves_causality = false · clinical_efficacy_claim_prohibited = true · ipd_used = false</p></section>
{red}
<h2>Source Receipt 来源收据</h2>
<table>{receipt_rows}</table>
<h2>Baseline Comparability 基线可运输性</h2>
<div class="card"><table><tr><th>协变量</th><th>SMD_approx</th><th>运输性偏移</th></tr>{baseline_body}</table>
<p>{html.escape(str((report.get('baseline_comparability') or {}).get('interpretation_zh') or ''))}</p></div>
<h2>RCT-DUPLICATE 三项指标</h2>
<div class="card">{metrics}</div>
<h2>阴性对照偏倚探针</h2>
<div class="card">{nco_card}</div>
<p>overall_validity_flag = <code>{html.escape(str(report.get('overall_validity_flag')))}</code>
· consistency_conclusion_allowed = <code>{html.escape(str(report.get('consistency_conclusion_allowed')))}</code></p>
</body></html>
"""


def write_benchmark_report(report: dict, output_dir) -> dict:
    from pathlib import Path

    target = Path(output_dir)
    if target.exists():
        raise ContractError("OUTPUT_EXISTS", f"Refusing to overwrite {target}")
    target.mkdir(parents=True)
    json_path = target / "benchmark_report.json"
    html_path = target / "benchmark_report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(render_benchmark_report_html(report), encoding="utf-8")
    return {"benchmark_report_json": str(json_path), "benchmark_report_html": str(html_path), "status": report.get("status")}
