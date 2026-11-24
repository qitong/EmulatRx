"""Bilateral grace-period Clone-Censor-Weight (CCW) protocol validation, synthetic DGP, arm-specific IPCW, and subject-cluster bootstrap competing-risks workbench."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from ._common import (
    ContractError,
    GateBlockedError,
    MAX_BOOTSTRAP_ITERATIONS,
    MAX_SEED,
    MAX_SYNTH_CASES,
    METHODOLOGICAL_DISCLAIMER,
    PROHIBITED_CLAIMS,
    SOURCE_MODE_SYNTHETIC_ONLY,
    atomic_publish,
    check_destination,
    code_provenance,
    dependency_versions,
    is_strict_int,
    is_strict_number,
    json_bytes,
    read_strict_json,
    sha256_bytes,
    sha256_file,
    utc_now_iso,
)
from .evalue import compute_evalue
from .weighting import fit_binary_logistic_mle, kish_ess


CCW_PROTOCOL_VERSION = "tte-ccw-v1"

CCW_METHODOLOGICAL_DISCLAIMER = (
    f"{METHODOLOGICAL_DISCLAIMER} In bilateral grace-period Clone-Censor-Weight (CCW), "
    "unbiased recovery of the protocol-sustained contrast requires longitudinal conditional "
    "exchangeability (no unmeasured baseline or time-varying confounders of treatment initiation "
    "and outcomes), longitudinal positivity across periods, regime consistency within the grace "
    "window, and arm-conditional non-informative administrative censoring."
)

CCW_IDENTIFICATION_ASSUMPTIONS = (
    "1. 纵向条件可交换性 (Longitudinal conditional exchangeability): 所有共同影响宽限期/随访期给药启动（人工删失）与结局/竞争死亡的基线及离散期前协变量均被无误差测量并正确纳入分臂删失模型（tte-ccw-v1 中宽限期 time_varying_covariates 限定为不受 (0, g] 内前期给药影响的期前混杂因素；给药后治疗–混杂反馈需序贯试验或参数 g-formula）；CCW 无法消除未测混杂。",
    "2. 策略一致性与无干扰 (Consistency & no interference): 宽限期 [0, g] 内遵循观测自然启动分布（逾期未启动者在 t=g 强制启动的协议持续对比）或预设协议等价，且个体间互不干扰。",
    "3. 纵向正性 (Longitudinal positivity): 在各协变量层内，存活至 t=g 者有正概率在 <=g 启动治疗，且各期未启动者有正概率继续不启动治疗。",
    "4. 条件非信息性行政删失 (Arm-conditional independent right-censoring): 给定已测协变量与克隆策略臂（或叠加行政删失 IPCW 后），行政失访独立于潜在事件时间。",
)

REQUIRED_CCW_ROOT_FIELDS = (
    "protocol_id",
    "protocol_version",
    "source_mode",
    "grace_period_spec",
    "covariates",
    "ipcw_spec",
    "diagnostic_gates",
)
OPTIONAL_CCW_ROOT_FIELDS = (
    "study_title",
    "simulation_or_data_label",
    "methodological_disclaimer",
)

RESERVED_CCW_COLUMNS = frozenset(
    (
        "subject_id",
        "synthetic_case_id",
        "simulation_or_data_label",
        "initiation_day",
        "duration_day",
        "event_code",
        "period_idx",
        "clone_arm",
        "clone_strategy",
        "T_clone",
        "event_clone",
        "artificial_censored",
        "censor_period",
        "stabilized_ipcw",
    )
)


def _check_exact_keys(
    obj: dict,
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
    context: str,
    missing_code: str = "MISSING_FIELD",
) -> None:
    allowed = set(required) | set(optional)
    missing = sorted(set(required) - set(obj.keys()))
    if missing:
        raise ContractError(missing_code, f"Missing required field(s) in {context}: {', '.join(missing)}")
    unknown = sorted(set(obj.keys()) - allowed)
    if unknown:
        raise ContractError("UNKNOWN_FIELD", f"Unknown field(s) in {context}: {', '.join(unknown)}")


def _require_nonempty_str(value: object, field_name: str, code: str = "INVALID_TYPE", max_len: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(code, f"'{field_name}' must be a non-empty string")
    s = value.strip()
    if len(s) > max_len:
        raise ContractError(code, f"'{field_name}' exceeds maximum length {max_len}")
    return s


def _normalize_numeric_day(value: object, field_name: str, code: str) -> int | float:
    if not is_strict_number(value):
        raise ContractError(code, f"'{field_name}' must be a finite number (not boolean)")
    fval = float(value)
    if is_strict_int(value):
        return int(value)
    if abs(fval - round(fval)) <= 1e-12:
        return int(round(fval))
    return fval


def _check_positive_multiple(
    value: int | float,
    step: int | float,
    field_name: str,
) -> int:
    val_f = float(value)
    step_f = float(step)
    ratio = val_f / step_f
    mult = int(round(ratio))
    tol = 1e-9 * max(1.0, abs(step_f))
    if mult < 1 or abs(val_f - mult * step_f) > tol:
        raise ContractError(
            "NON_DIVISIBLE_TIME_GRID",
            f"{field_name}={val_f:g} must be a positive integer multiple of period_length_days={step_f:g}. "
            "Hint: Align grace_period_days, max_followup_days, and evaluation_horizons_days to exact multiples "
            "of period_length_days.",
        )
    return mult


def _validate_covariate_name(raw_name: object, context: str, seen_names: set[str]) -> str:
    name = _require_nonempty_str(raw_name, f"{context}.name", "INVALID_COVARIATES", max_len=128)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ContractError(
            "INVALID_COVARIATES",
            f"{context}.name '{name}' must match ^[A-Za-z][A-Za-z0-9_]*$",
        )
    if name in RESERVED_CCW_COLUMNS or name.endswith("__assessed_day_offset") or name.endswith("__assessed_day"):
        raise ContractError(
            "INVALID_COVARIATES",
            f"{context}.name '{name}' collides with a reserved CCW column identifier",
        )
    if name in seen_names:
        raise ContractError("INVALID_COVARIATES", f"Duplicate covariate name '{name}' in protocol")
    seen_names.add(name)
    return name


def validate_ccw_protocol_dict(obj: dict) -> dict:
    """Validate and normalize a tte-ccw-v1 protocol dictionary, raising ContractError on any violation."""
    if not isinstance(obj, dict):
        raise ContractError("INVALID_TYPE", "CCW protocol must be a JSON object")

    _check_exact_keys(
        obj,
        required=REQUIRED_CCW_ROOT_FIELDS,
        optional=OPTIONAL_CCW_ROOT_FIELDS,
        context="ccw_protocol",
    )

    protocol_id = _require_nonempty_str(obj["protocol_id"], "protocol_id", "INVALID_PROTOCOL_ID", max_len=128)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", protocol_id):
        raise ContractError(
            "INVALID_PROTOCOL_ID",
            f"protocol_id '{protocol_id}' must match ^[A-Za-z0-9_-]+$",
        )

    if not isinstance(obj["protocol_version"], str):
        raise ContractError("INVALID_TYPE", "protocol_version must be a string")
    if obj["protocol_version"] != CCW_PROTOCOL_VERSION:
        raise ContractError(
            "UNKNOWN_VERSION",
            f"Unsupported protocol_version '{obj['protocol_version']}'; expected '{CCW_PROTOCOL_VERSION}'",
        )

    if not isinstance(obj["source_mode"], str):
        raise ContractError("INVALID_TYPE", "source_mode must be a string")
    if obj["source_mode"] != SOURCE_MODE_SYNTHETIC_ONLY:
        raise ContractError(
            "UNSUPPORTED_SOURCE_MODE",
            f"Unsupported source_mode '{obj['source_mode']}'; v1 strictly requires '{SOURCE_MODE_SYNTHETIC_ONLY}'",
        )

    study_title = (
        _require_nonempty_str(obj["study_title"], "study_title")
        if "study_title" in obj
        else f"Synthetic Grace-Period CCW Study ({protocol_id})"
    )
    sim_label = (
        _require_nonempty_str(obj["simulation_or_data_label"], "simulation_or_data_label")
        if "simulation_or_data_label" in obj
        else "完全合成宽限期 CCW 基准，非真实患者数据、非临床疗效证据"
    )

    # 1. grace_period_spec
    gp = obj["grace_period_spec"]
    if not isinstance(gp, dict):
        raise ContractError("INVALID_TYPE", "grace_period_spec must be an object")
    _check_exact_keys(
        gp,
        required=("grace_period_days", "period_length_days", "max_followup_days", "evaluation_horizons_days"),
        context="grace_period_spec",
    )

    period_len = _normalize_numeric_day(
        gp["period_length_days"], "grace_period_spec.period_length_days", "INVALID_GRACE_PERIOD_SPEC"
    )
    grace_days = _normalize_numeric_day(
        gp["grace_period_days"], "grace_period_spec.grace_period_days", "INVALID_GRACE_PERIOD_SPEC"
    )
    max_followup = _normalize_numeric_day(
        gp["max_followup_days"], "grace_period_spec.max_followup_days", "INVALID_GRACE_PERIOD_SPEC"
    )

    if float(period_len) <= 0.0:
        raise ContractError("INVALID_GRACE_PERIOD_SPEC", "period_length_days must be > 0")
    if float(grace_days) <= 0.0:
        raise ContractError("INVALID_GRACE_PERIOD_SPEC", "grace_period_days must be > 0")
    if float(max_followup) <= float(grace_days):
        raise ContractError(
            "INVALID_GRACE_PERIOD_SPEC",
            f"max_followup_days ({max_followup:g}) must strictly exceed grace_period_days ({grace_days:g})",
        )

    _check_positive_multiple(grace_days, period_len, "grace_period_spec.grace_period_days")
    _check_positive_multiple(max_followup, period_len, "grace_period_spec.max_followup_days")

    raw_horizons = gp["evaluation_horizons_days"]
    if not isinstance(raw_horizons, list) or len(raw_horizons) == 0:
        raise ContractError(
            "MISSING_HORIZON",
            "grace_period_spec.evaluation_horizons_days must be a non-empty list of positive evaluation horizons",
        )
    horizons_norm: list[int | float] = []
    seen_horizons_float: list[float] = []
    for idx, raw_h in enumerate(raw_horizons):
        h_val = _normalize_numeric_day(
            raw_h,
            f"grace_period_spec.evaluation_horizons_days[{idx}]",
            "INVALID_HORIZON",
        )
        hf = float(h_val)
        if hf <= 0.0 or hf > float(max_followup):
            raise ContractError(
                "INVALID_HORIZON",
                f"evaluation_horizons_days[{idx}]={hf:g} must lie in (0, max_followup_days={float(max_followup):g}]",
            )
        _check_positive_multiple(
            h_val,
            period_len,
            f"grace_period_spec.evaluation_horizons_days[{idx}]",
        )
        if any(abs(hf - prev) <= 1e-9 for prev in seen_horizons_float):
            raise ContractError("INVALID_HORIZON", f"Duplicate evaluation horizon: {hf:g}")
        seen_horizons_float.append(hf)
        horizons_norm.append(h_val)
    horizons_norm.sort(key=float)

    # 2. covariates (baseline <= 0; pre-period time-varying strictly < 0 relative to period decision start)
    cov_block = obj["covariates"]
    if not isinstance(cov_block, dict):
        raise ContractError("INVALID_TYPE", "covariates must be an object")
    _check_exact_keys(
        cov_block,
        required=("baseline_covariates",),
        optional=("time_varying_covariates",),
        context="covariates",
    )

    raw_base_covs = cov_block["baseline_covariates"]
    if not isinstance(raw_base_covs, list) or len(raw_base_covs) == 0:
        raise ContractError(
            "INVALID_COVARIATES",
            "covariates.baseline_covariates must be a non-empty list",
        )

    seen_cov_names: set[str] = set()
    norm_base_covs: list[dict] = []
    for idx, item in enumerate(raw_base_covs):
        ctx = f"covariates.baseline_covariates[{idx}]"
        if not isinstance(item, dict):
            raise ContractError("INVALID_TYPE", f"{ctx} must be an object")
        _check_exact_keys(
            item,
            required=("name", "type", "assessed_day_offset"),
            optional=("description",),
            context=ctx,
        )
        name = _validate_covariate_name(item["name"], ctx, seen_cov_names)
        ctype = _require_nonempty_str(item["type"], f"{ctx}.type", "INVALID_COVARIATES")
        if ctype not in ("continuous", "binary"):
            raise ContractError("INVALID_COVARIATES", f"{ctx}.type must be 'continuous' or 'binary'")
        offset = _normalize_numeric_day(item["assessed_day_offset"], f"{ctx}.assessed_day_offset", "INVALID_COVARIATES")
        if float(offset) > 0.0:
            raise ContractError(
                "POST_BASELINE_COVARIATE_LEAK",
                f"Baseline covariate '{name}' has assessed_day_offset={float(offset):g} > 0. "
                "Hint: Baseline covariates must be measured at or before time_zero (offset <= 0) to prevent "
                "conditioning on post-baseline / post-initiation variables.",
            )
        entry: dict[str, Any] = {
            "name": name,
            "type": ctype,
            "assessed_day_offset": offset,
        }
        if "description" in item:
            if not isinstance(item["description"], str):
                raise ContractError("INVALID_TYPE", f"{ctx}.description must be a string")
            entry["description"] = item["description"]
        norm_base_covs.append(entry)

    raw_tv_covs = cov_block.get("time_varying_covariates", [])
    if not isinstance(raw_tv_covs, list):
        raise ContractError("INVALID_TYPE", "covariates.time_varying_covariates must be a list")
    norm_tv_covs: list[dict] = []
    for idx, item in enumerate(raw_tv_covs):
        ctx = f"covariates.time_varying_covariates[{idx}]"
        if not isinstance(item, dict):
            raise ContractError("INVALID_TYPE", f"{ctx} must be an object")
        _check_exact_keys(
            item,
            required=("name", "type", "offset_from_period_start_days"),
            optional=("description",),
            context=ctx,
        )
        name = _validate_covariate_name(item["name"], ctx, seen_cov_names)
        ctype = _require_nonempty_str(item["type"], f"{ctx}.type", "INVALID_COVARIATES")
        if ctype not in ("continuous", "binary"):
            raise ContractError("INVALID_COVARIATES", f"{ctx}.type must be 'continuous' or 'binary'")
        rel_off = _normalize_numeric_day(
            item["offset_from_period_start_days"],
            f"{ctx}.offset_from_period_start_days",
            "INVALID_COVARIATES",
        )
        if float(rel_off) >= 0.0:
            raise ContractError(
                "TIME_VARYING_COVARIATE_TIME_TRAVEL",
                f"Time-varying covariate '{name}' has offset_from_period_start_days={float(rel_off):g} >= 0. "
                "Hint: Pre-period time-varying covariates for period k must be measured strictly before period k's "
                "decision start (k-1)*period_length_days (offset_from_period_start_days < 0) to prevent time-travel bias.",
            )
        tv_entry: dict[str, Any] = {
            "name": name,
            "type": ctype,
            "offset_from_period_start_days": rel_off,
        }
        if "description" in item:
            if not isinstance(item["description"], str):
                raise ContractError("INVALID_TYPE", f"{ctx}.description must be a string")
            tv_entry["description"] = item["description"]
        norm_tv_covs.append(tv_entry)

    # 3. ipcw_spec
    ipcw = obj["ipcw_spec"]
    if not isinstance(ipcw, dict):
        raise ContractError("INVALID_TYPE", "ipcw_spec must be an object")
    _check_exact_keys(
        ipcw,
        required=(),
        optional=("truncate_quantiles", "include_admin_ipcw"),
        context="ipcw_spec",
    )
    raw_trunc = ipcw.get("truncate_quantiles", None)
    norm_trunc: list[float] | None = None
    if raw_trunc is not None:
        if not isinstance(raw_trunc, list) or len(raw_trunc) != 2:
            raise ContractError(
                "INVALID_IPCW_SPEC",
                "ipcw_spec.truncate_quantiles must be null or [q_low, q_high]",
            )
        if not (is_strict_number(raw_trunc[0]) and is_strict_number(raw_trunc[1])):
            raise ContractError("INVALID_IPCW_SPEC", "ipcw_spec.truncate_quantiles elements must be finite numbers")
        q_lo, q_hi = float(raw_trunc[0]), float(raw_trunc[1])
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ContractError(
                "INVALID_IPCW_SPEC",
                f"ipcw_spec.truncate_quantiles [{q_lo}, {q_hi}] must satisfy 0.0 <= q_low < q_high <= 1.0",
            )
        norm_trunc = [q_lo, q_hi]

    include_admin_ipcw = ipcw.get("include_admin_ipcw", False)
    if not isinstance(include_admin_ipcw, bool):
        raise ContractError("INVALID_IPCW_SPEC", "ipcw_spec.include_admin_ipcw must be a boolean")

    # 4. diagnostic_gates (four core gates required explicitly, plus optional pre-truncation / clipping gates)
    gates = obj["diagnostic_gates"]
    if not isinstance(gates, dict):
        raise ContractError("INVALID_TYPE", "diagnostic_gates must be an object")
    _check_exact_keys(
        gates,
        required=(
            "min_ess_ratio",
            "max_post_grace_smd",
            "max_stabilized_weight",
            "min_uncensored_subjects_per_clone",
        ),
        optional=(
            "max_pre_truncation_weight",
            "max_truncated_weight_fraction",
        ),
        context="diagnostic_gates",
        missing_code="MISSING_DIAGNOSTIC_GATE",
    )

    min_ess = gates["min_ess_ratio"]
    if not is_strict_number(min_ess) or not (0.0 < float(min_ess) <= 1.0):
        raise ContractError(
            "INVALID_DIAGNOSTIC_GATE",
            "diagnostic_gates.min_ess_ratio must be a finite number in (0.0, 1.0]",
        )

    max_smd = gates["max_post_grace_smd"]
    if not is_strict_number(max_smd) or float(max_smd) <= 0.0:
        raise ContractError(
            "INVALID_DIAGNOSTIC_GATE",
            "diagnostic_gates.max_post_grace_smd must be a finite positive number",
        )

    max_w = gates["max_stabilized_weight"]
    if not is_strict_number(max_w) or float(max_w) < 1.0:
        raise ContractError(
            "INVALID_DIAGNOSTIC_GATE",
            "diagnostic_gates.max_stabilized_weight must be a finite number >= 1.0",
        )

    min_uncens = gates["min_uncensored_subjects_per_clone"]
    if not is_strict_int(min_uncens) or min_uncens < 1:
        raise ContractError(
            "INVALID_DIAGNOSTIC_GATE",
            "diagnostic_gates.min_uncensored_subjects_per_clone must be an integer >= 1",
        )

    norm_gates: dict[str, Any] = {
        "min_ess_ratio": float(min_ess),
        "max_post_grace_smd": float(max_smd),
        "max_stabilized_weight": float(max_w),
        "min_uncensored_subjects_per_clone": int(min_uncens),
    }
    if "max_pre_truncation_weight" in gates:
        max_pre_w = gates["max_pre_truncation_weight"]
        if not is_strict_number(max_pre_w) or float(max_pre_w) < 1.0:
            raise ContractError(
                "INVALID_DIAGNOSTIC_GATE",
                "diagnostic_gates.max_pre_truncation_weight must be a finite number >= 1.0",
            )
        norm_gates["max_pre_truncation_weight"] = float(max_pre_w)

    if "max_truncated_weight_fraction" in gates:
        max_trunc_f = gates["max_truncated_weight_fraction"]
        if not is_strict_number(max_trunc_f) or not (0.0 < float(max_trunc_f) <= 1.0):
            raise ContractError(
                "INVALID_DIAGNOSTIC_GATE",
                "diagnostic_gates.max_truncated_weight_fraction must be a finite number in (0.0, 1.0]",
            )
        norm_gates["max_truncated_weight_fraction"] = float(max_trunc_f)

    if "methodological_disclaimer" in obj and not isinstance(obj["methodological_disclaimer"], str):
        raise ContractError("INVALID_TYPE", "methodological_disclaimer must be a string")

    return {
        "protocol_id": protocol_id,
        "protocol_version": CCW_PROTOCOL_VERSION,
        "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
        "study_title": study_title,
        "simulation_or_data_label": sim_label,
        "grace_period_spec": {
            "grace_period_days": grace_days,
            "period_length_days": period_len,
            "max_followup_days": max_followup,
            "evaluation_horizons_days": horizons_norm,
        },
        "covariates": {
            "baseline_covariates": norm_base_covs,
            "time_varying_covariates": norm_tv_covs,
        },
        "ipcw_spec": {
            "truncate_quantiles": norm_trunc,
            "include_admin_ipcw": include_admin_ipcw,
        },
        "diagnostic_gates": norm_gates,
        "methodological_disclaimer": CCW_METHODOLOGICAL_DISCLAIMER,
    }


def load_and_validate_ccw_protocol(path: Path) -> tuple[bytes, dict]:
    """Read strict UTF-8 JSON file and validate against the tte-ccw-v1 contract."""
    raw, obj = read_strict_json(path)
    normalized = validate_ccw_protocol_dict(obj)
    return raw, normalized


def _render_ccw_protocol_report_md(normalized: dict) -> str:
    gp = normalized["grace_period_spec"]
    m_g = int(round(float(gp["grace_period_days"]) / float(gp["period_length_days"])))
    k_total = int(round(float(gp["max_followup_days"]) / float(gp["period_length_days"])))
    gates = normalized["diagnostic_gates"]
    ipcw = normalized["ipcw_spec"]
    base_names = ", ".join(c["name"] for c in normalized["covariates"]["baseline_covariates"])
    tv_covs = normalized["covariates"]["time_varying_covariates"]
    tv_summary = (
        ", ".join(f"{c['name']} (offset={c['offset_from_period_start_days']}d)" for c in tv_covs)
        if tv_covs
        else "None (baseline covariates only)"
    )
    horizons_str = ", ".join(f"{h:g}" for h in gp["evaluation_horizons_days"])

    lines = [
        f"# Bilateral Grace-Period CCW Protocol Check Report (`{normalized['protocol_id']}`)",
        "",
        f"- **Protocol Version**: `{normalized['protocol_version']}`",
        f"- **Source Mode**: `{normalized['source_mode']}`",
        f"- **Study Title**: {normalized['study_title']}",
        f"- **Simulation / Data Label**: {normalized['simulation_or_data_label']}",
        f"- **Grace Period ($g$)**: `{gp['grace_period_days']}` days (`m_g = {m_g}` discrete periods of `Δ = {gp['period_length_days']}` days)",
        f"- **Max Follow-up ($H_\\max$)**: `{gp['max_followup_days']}` days (`K = {k_total}` total discrete periods)",
        f"- **Evaluation Horizons ($t^*$)**: `[{horizons_str}]` days",
        f"- **Baseline Covariates ($L_0$)**: `{base_names}`",
        f"- **Pre-Period Time-Varying Covariates ($L_{{k-1}}$)**: {tv_summary}",
        f"- **IPCW Spec**: `truncate_quantiles={ipcw['truncate_quantiles']}`, `include_admin_ipcw={ipcw['include_admin_ipcw']}`",
        (
            f"- **Diagnostic Gates (SAP-supplied)**: `min_ess_ratio={gates['min_ess_ratio']}`, "
            f"`max_post_grace_smd={gates['max_post_grace_smd']}`, "
            f"`max_stabilized_weight={gates['max_stabilized_weight']}`, "
            f"`min_uncensored_subjects_per_clone={gates['min_uncensored_subjects_per_clone']}`"
        ),
        "",
        "## Causal Identification Assumptions",
        *[f"- {item}" for item in CCW_IDENTIFICATION_ASSUMPTIONS],
        "",
        "## Prohibited Claims",
        *[f"- {claim}" for claim in PROHIBITED_CLAIMS],
        "",
        "## Methodological Disclaimer",
        normalized["methodological_disclaimer"],
        "",
    ]
    return "\n".join(lines)


def validate_ccw_protocol(protocol_path: Path, out_dir: Path) -> dict:
    """Validate a CCW v1 protocol file and atomically publish normalized protocol + report + SHA-256 manifest."""
    check_destination(out_dir)
    raw, normalized = load_and_validate_ccw_protocol(protocol_path)
    commit, code_dirty = code_provenance()
    deps = dependency_versions()
    norm_bytes = json_bytes(normalized)
    report_bytes = _render_ccw_protocol_report_md(normalized).encode("utf-8")
    output_paths = ["normalized_protocol.json", "protocol_check_report.md"]

    def _populate(stage: Path) -> None:
        (stage / "normalized_protocol.json").write_bytes(norm_bytes)
        (stage / "protocol_check_report.md").write_bytes(report_bytes)
        manifest = {
            "status": "ccw_protocol_validated",
            "activity": "tte_ccw_protocol_validation",
            "protocol_id": normalized["protocol_id"],
            "protocol_version": CCW_PROTOCOL_VERSION,
            "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
            "prohibited_claims": list(PROHIBITED_CLAIMS),
            "simulation_or_data_label": normalized["simulation_or_data_label"],
            "methodological_disclaimer": normalized["methodological_disclaimer"],
            "input_sha256": sha256_bytes(raw),
            "normalized_protocol_sha256": sha256_bytes(norm_bytes),
            "code_commit": commit,
            "code_dirty": code_dirty,
            "dependency_versions": deps,
            "output_paths": output_paths,
            "output_sha256": {p: sha256_file(stage / p) for p in output_paths},
            "run_at_utc": utc_now_iso(),
        }
        (stage / "run_manifest.json").write_bytes(json_bytes(manifest))

    atomic_publish(out_dir, _populate)
    return {
        "status": "ccw_protocol_validated",
        "protocol_id": normalized["protocol_id"],
        "protocol_version": CCW_PROTOCOL_VERSION,
        "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
        "simulation_or_data_label": normalized["simulation_or_data_label"],
        "output_path": str(out_dir),
    }


# ---------------------------------------------------------------------------
# Module 2: Seeded Delayed-Initiation & Competing-Risks Synthetic DGP + Truth
# ---------------------------------------------------------------------------

ALLOWED_CCW_DGP_KEYS = frozenset(
    (
        "true_log_hr_event1",
        "true_log_hr_event2",
        "grace_initiation_intercept",
        "post_grace_initiation_intercept",
        "initiation_covariate_slopes",
        "initiation_tv_covariate_slopes",
        "baseline_daily_hazard_event1",
        "baseline_daily_hazard_event2",
        "admin_censor_daily_hazard",
        "outcome_covariate_log_hr_event1",
        "outcome_covariate_log_hr_event2",
        "outcome_tv_covariate_log_hr_event1",
        "outcome_tv_covariate_log_hr_event2",
        "admin_censor_covariate_log_hr",
        "n_truth",
    )
)


def _resolve_ccw_protocol_input(protocol: Path | str | dict) -> tuple[bytes, dict]:
    if isinstance(protocol, dict):
        norm = validate_ccw_protocol_dict(protocol)
        return json_bytes(norm), norm
    if isinstance(protocol, (Path, str)):
        return load_and_validate_ccw_protocol(Path(protocol))
    raise ContractError("INVALID_TYPE", "protocol must be a dict or filesystem Path")


def _resolve_ccw_dgp_override(dgp_params: Path | str | dict | None) -> tuple[bytes | None, dict | None]:
    if dgp_params is None:
        return None, None
    if isinstance(dgp_params, dict):
        return json_bytes(dgp_params), dict(dgp_params)
    if isinstance(dgp_params, (Path, str)):
        raw, obj = read_strict_json(Path(dgp_params), io_error_code="DGP_PARAMS_IO_ERROR")
        return raw, obj
    raise ContractError("INVALID_TYPE", "dgp_params must be None, dict, or filesystem Path")


def _default_ccw_dgp_parameters(num_base_covs: int, num_tv_covs: int) -> dict[str, Any]:
    init_pat = [0.30, -0.25, 0.28, 0.22]
    ev1_pat = [0.35, 0.30, 0.35, 0.40]
    ev2_pat = [0.30, 0.20, 0.40, 0.25]
    tv_init_pat = [0.30, 0.25]
    tv_ev1_pat = [0.25, 0.20]
    tv_ev2_pat = [0.15, 0.15]
    return {
        "true_log_hr_event1": -0.45,
        "true_log_hr_event2": 0.0,
        "grace_initiation_intercept": -1.95,
        "post_grace_initiation_intercept": -3.40,
        "initiation_covariate_slopes": [round(init_pat[j % len(init_pat)], 6) for j in range(num_base_covs)],
        "initiation_tv_covariate_slopes": [round(tv_init_pat[j % len(tv_init_pat)], 6) for j in range(num_tv_covs)],
        "baseline_daily_hazard_event1": 0.0025,
        "baseline_daily_hazard_event2": 0.0010,
        "admin_censor_daily_hazard": 0.0005,
        "outcome_covariate_log_hr_event1": [round(ev1_pat[j % len(ev1_pat)], 6) for j in range(num_base_covs)],
        "outcome_covariate_log_hr_event2": [round(ev2_pat[j % len(ev2_pat)], 6) for j in range(num_base_covs)],
        "outcome_tv_covariate_log_hr_event1": [round(tv_ev1_pat[j % len(tv_ev1_pat)], 6) for j in range(num_tv_covs)],
        "outcome_tv_covariate_log_hr_event2": [round(tv_ev2_pat[j % len(tv_ev2_pat)], 6) for j in range(num_tv_covs)],
        "admin_censor_covariate_log_hr": [0.0 for _ in range(num_base_covs)],
        "n_truth": 50000,
    }


def _merge_and_validate_ccw_dgp_params(
    defaults: dict[str, Any],
    override: dict | None,
    *,
    num_base_covs: int,
    num_tv_covs: int,
) -> dict[str, Any]:
    params = dict(defaults)
    if override is not None:
        unknown = sorted(set(override.keys()) - ALLOWED_CCW_DGP_KEYS)
        if unknown:
            raise ContractError("UNKNOWN_DGP_PARAM", f"Unknown CCW DGP parameter(s): {', '.join(unknown)}")
        params.update(override)

    def _vec(name: str, expected_len: int) -> list[float]:
        val = params[name]
        if not isinstance(val, list) or len(val) != expected_len:
            raise ContractError("INVALID_DGP_PARAM", f"{name} must be a list of length {expected_len}")
        out: list[float] = []
        for idx, x in enumerate(val):
            if not is_strict_number(x):
                raise ContractError("INVALID_DGP_PARAM", f"{name}[{idx}] must be a finite number")
            out.append(float(x))
        return out

    for scalar_name in (
        "true_log_hr_event1",
        "true_log_hr_event2",
        "grace_initiation_intercept",
        "post_grace_initiation_intercept",
    ):
        if not is_strict_number(params[scalar_name]):
            raise ContractError("INVALID_DGP_PARAM", f"{scalar_name} must be a finite number")

    bh1 = params["baseline_daily_hazard_event1"]
    bh2 = params["baseline_daily_hazard_event2"]
    adm_h = params["admin_censor_daily_hazard"]
    if not is_strict_number(bh1) or float(bh1) <= 0.0:
        raise ContractError("INVALID_DGP_PARAM", "baseline_daily_hazard_event1 must be > 0")
    if not is_strict_number(bh2) or float(bh2) < 0.0:
        raise ContractError("INVALID_DGP_PARAM", "baseline_daily_hazard_event2 must be >= 0")
    if not is_strict_number(adm_h) or float(adm_h) < 0.0:
        raise ContractError("INVALID_DGP_PARAM", "admin_censor_daily_hazard must be >= 0")

    n_truth = params["n_truth"]
    if not is_strict_int(n_truth) or not (1000 <= n_truth <= 500000):
        raise ContractError("INVALID_DGP_PARAM", "n_truth must be an integer in 1000..500000")

    return {
        "true_log_hr_event1": float(params["true_log_hr_event1"]),
        "true_log_hr_event2": float(params["true_log_hr_event2"]),
        "grace_initiation_intercept": float(params["grace_initiation_intercept"]),
        "post_grace_initiation_intercept": float(params["post_grace_initiation_intercept"]),
        "initiation_covariate_slopes": _vec("initiation_covariate_slopes", num_base_covs),
        "initiation_tv_covariate_slopes": _vec("initiation_tv_covariate_slopes", num_tv_covs),
        "baseline_daily_hazard_event1": float(bh1),
        "baseline_daily_hazard_event2": float(bh2),
        "admin_censor_daily_hazard": float(adm_h),
        "outcome_covariate_log_hr_event1": _vec("outcome_covariate_log_hr_event1", num_base_covs),
        "outcome_covariate_log_hr_event2": _vec("outcome_covariate_log_hr_event2", num_base_covs),
        "outcome_tv_covariate_log_hr_event1": _vec("outcome_tv_covariate_log_hr_event1", num_tv_covs),
        "outcome_tv_covariate_log_hr_event2": _vec("outcome_tv_covariate_log_hr_event2", num_tv_covs),
        "admin_censor_covariate_log_hr": _vec("admin_censor_covariate_log_hr", num_base_covs),
        "n_truth": int(n_truth),
    }


def _sample_baseline_covariate_matrix(
    rng: np.random.Generator,
    n: int,
    base_covs: list[dict],
) -> np.ndarray:
    p = len(base_covs)
    X = np.zeros((n, p), dtype=np.float64)
    z0 = rng.standard_normal(n)
    for j, spec in enumerate(base_covs):
        if spec["type"] == "continuous":
            raw_z = z0 if j == 0 else (0.25 * z0 + math.sqrt(1.0 - 0.25**2) * rng.standard_normal(n))
            X[:, j] = np.round(raw_z, 6)
        else:
            prev_signal = 0.35 * X[:, 0] if j > 0 else 0.35 * z0
            prob = expit(-0.25 - 0.15 * (j % 3) + prev_signal)
            X[:, j] = (rng.random(n) < prob).astype(np.float64)
    return X


def _sample_tv_covariates_cube(
    rng: np.random.Generator,
    X_base: np.ndarray,
    tv_covs: list[dict],
    k_total: int,
) -> np.ndarray:
    """Generate pre-period time-varying covariates L_{i, k-1} of shape (n, k_total, p_tv)."""
    n = X_base.shape[0]
    p_tv = len(tv_covs)
    if p_tv == 0:
        return np.zeros((n, k_total, 0), dtype=np.float64)
    cube = np.zeros((n, k_total, p_tv), dtype=np.float64)
    anchor = X_base[:, 0]
    for k_idx in range(k_total):
        for j, spec in enumerate(tv_covs):
            prev_val = anchor if k_idx == 0 else cube[:, k_idx - 1, j]
            if spec["type"] == "continuous":
                cube[:, k_idx, j] = np.round(0.60 * prev_val + 0.80 * rng.standard_normal(n), 6)
            else:
                prob = expit(-0.30 + 0.45 * prev_val)
                cube[:, k_idx, j] = (rng.random(n) < prob).astype(np.float64)
    return cube


def _compute_ccw_counterfactual_truth(
    *,
    protocol: dict,
    dgp: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Compute counterfactual CIF_1, CIF_2, RD, and RR_CIF at evaluation horizons.

    Uses exact per-subject discrete-time Markov state recursion over n_truth synthetic subjects
    drawn from a deterministic auxiliary seed, eliminating event-level Monte Carlo noise and
    preserving exact 0.0 risk difference under true_log_hr_event1 == 0.0 & true_log_hr_event2 == 0.0.
    """
    gp = protocol["grace_period_spec"]
    delta = float(gp["period_length_days"])
    grace_days = float(gp["grace_period_days"])
    max_followup = float(gp["max_followup_days"])
    m_g = int(round(grace_days / delta))
    k_total = int(round(max_followup / delta))
    horizons = [float(h) for h in gp["evaluation_horizons_days"]]

    base_covs = protocol["covariates"]["baseline_covariates"]
    tv_covs = protocol["covariates"]["time_varying_covariates"]
    n_truth = int(dgp["n_truth"])
    truth_seed = int((seed + 1_048_583) % (MAX_SEED + 1))
    rng_truth = np.random.default_rng(truth_seed)

    X_base = _sample_baseline_covariate_matrix(rng_truth, n_truth, base_covs)
    X_tv = _sample_tv_covariates_cube(rng_truth, X_base, tv_covs, k_total)

    alpha_base = X_base @ np.asarray(dgp["initiation_covariate_slopes"], dtype=np.float64)
    beta1_base = X_base @ np.asarray(dgp["outcome_covariate_log_hr_event1"], dtype=np.float64)
    beta2_base = X_base @ np.asarray(dgp["outcome_covariate_log_hr_event2"], dtype=np.float64)
    alpha_tv_vec = np.asarray(dgp["initiation_tv_covariate_slopes"], dtype=np.float64)
    beta1_tv_vec = np.asarray(dgp["outcome_tv_covariate_log_hr_event1"], dtype=np.float64)
    beta2_tv_vec = np.asarray(dgp["outcome_tv_covariate_log_hr_event2"], dtype=np.float64)

    bh1 = float(dgp["baseline_daily_hazard_event1"])
    bh2 = float(dgp["baseline_daily_hazard_event2"])
    theta1 = float(dgp["true_log_hr_event1"])
    theta2 = float(dgp["true_log_hr_event2"])
    a0_grace = float(dgp["grace_initiation_intercept"])

    # Strategy 0 (never_initiate): state s0_alive
    s0_alive = np.ones(n_truth, dtype=np.float64)
    s0_cif1 = np.zeros(n_truth, dtype=np.float64)
    s0_cif2 = np.zeros(n_truth, dtype=np.float64)

    # Strategy 1 (initiate_within_grace): states s1_uninit, s1_init
    s1_uninit = np.ones(n_truth, dtype=np.float64)
    s1_init = np.zeros(n_truth, dtype=np.float64)
    s1_cif1 = np.zeros(n_truth, dtype=np.float64)
    s1_cif2 = np.zeros(n_truth, dtype=np.float64)

    curve_by_period: dict[int, dict[str, float]] = {}

    for k in range(1, k_total + 1):
        k_idx = k - 1
        tv_init = X_tv[:, k_idx, :] @ alpha_tv_vec if tv_covs else 0.0
        tv_b1 = X_tv[:, k_idx, :] @ beta1_tv_vec if tv_covs else 0.0
        tv_b2 = X_tv[:, k_idx, :] @ beta2_tv_vec if tv_covs else 0.0

        # Period-start treatment transition under Strategy 1
        if k <= m_g:
            p_init = expit(a0_grace + alpha_base + tv_init)
            newly_init = s1_uninit * p_init
            s1_uninit = s1_uninit - newly_init
            s1_init = s1_init + newly_init
        elif k == m_g + 1:
            # Protocol-sustained grace regime: any survivor uninitiated by t=g initiates at t=g+
            s1_init = s1_init + s1_uninit
            s1_uninit = np.zeros_like(s1_uninit)

        # Within-period untreated & treated transition probabilities
        lam1_u = bh1 * np.exp(beta1_base + tv_b1)
        lam2_u = bh2 * np.exp(beta2_base + tv_b2)
        tot_u = lam1_u + lam2_u
        ev_prob_u = -np.expm1(-tot_u * delta)
        q1_u = np.where(tot_u > 0.0, (lam1_u / np.maximum(tot_u, 1e-30)) * ev_prob_u, 0.0)
        q2_u = np.where(tot_u > 0.0, (lam2_u / np.maximum(tot_u, 1e-30)) * ev_prob_u, 0.0)
        surv_u = np.clip(1.0 - q1_u - q2_u, 0.0, 1.0)

        lam1_t = lam1_u * math.exp(theta1)
        lam2_t = lam2_u * math.exp(theta2)
        tot_t = lam1_t + lam2_t
        ev_prob_t = -np.expm1(-tot_t * delta)
        q1_t = np.where(tot_t > 0.0, (lam1_t / np.maximum(tot_t, 1e-30)) * ev_prob_t, 0.0)
        q2_t = np.where(tot_t > 0.0, (lam2_t / np.maximum(tot_t, 1e-30)) * ev_prob_t, 0.0)
        surv_t = np.clip(1.0 - q1_t - q2_t, 0.0, 1.0)

        # Advance Strategy 0
        s0_cif1 = s0_cif1 + s0_alive * q1_u
        s0_cif2 = s0_cif2 + s0_alive * q2_u
        s0_alive = s0_alive * surv_u

        # Advance Strategy 1
        s1_cif1 = s1_cif1 + s1_uninit * q1_u + s1_init * q1_t
        s1_cif2 = s1_cif2 + s1_uninit * q2_u + s1_init * q2_t
        s1_uninit = s1_uninit * surv_u
        s1_init = s1_init * surv_t

        curve_by_period[k] = {
            "cif1_s1": float(np.mean(s1_cif1)),
            "cif2_s1": float(np.mean(s1_cif2)),
            "surv_s1": float(np.mean(s1_uninit + s1_init)),
            "cif1_s0": float(np.mean(s0_cif1)),
            "cif2_s0": float(np.mean(s0_cif2)),
            "surv_s0": float(np.mean(s0_alive)),
        }

    horizons_dict: dict[str, dict[str, Any]] = {}
    horizon_table: list[dict[str, Any]] = []
    for h_raw in gp["evaluation_horizons_days"]:
        k_h = int(round(float(h_raw) / delta))
        row_p = curve_by_period[k_h]
        cif1_1 = row_p["cif1_s1"]
        cif1_0 = row_p["cif1_s0"]
        cif2_1 = row_p["cif2_s1"]
        cif2_0 = row_p["cif2_s0"]
        rd_1 = cif1_1 - cif1_0
        rr_1 = (cif1_1 / cif1_0) if cif1_0 > 0.0 else None
        rd_2 = cif2_1 - cif2_0
        rr_2 = (cif2_1 / cif2_0) if cif2_0 > 0.0 else None
        entry = {
            "horizon_day": h_raw,
            "period_index": k_h,
            "strategy_1_initiate_within_grace": {
                "CIF_1": round(cif1_1, 10),
                "CIF_2": round(cif2_1, 10),
                "survival": round(row_p["surv_s1"], 10),
            },
            "strategy_0_never_initiate": {
                "CIF_1": round(cif1_0, 10),
                "CIF_2": round(cif2_0, 10),
                "survival": round(row_p["surv_s0"], 10),
            },
            "CIF_1_strategy_1": round(cif1_1, 10),
            "CIF_1_strategy_0": round(cif1_0, 10),
            "CIF_2_strategy_1": round(cif2_1, 10),
            "CIF_2_strategy_0": round(cif2_0, 10),
            "true_rd_event1": round(rd_1, 10),
            "true_rr_cif_event1": round(rr_1, 10) if rr_1 is not None else None,
            "true_rd_event2": round(rd_2, 10),
            "true_rr_cif_event2": round(rr_2, 10) if rr_2 is not None else None,
        }
        horizons_dict[f"{float(h_raw):g}"] = entry
        horizon_table.append(entry)

    truth_time_days = [round(k * delta, 6) for k in range(0, k_total + 1)]
    truth_cif1_s1 = [0.0] + [round(curve_by_period[k]["cif1_s1"], 10) for k in range(1, k_total + 1)]
    truth_cif1_s0 = [0.0] + [round(curve_by_period[k]["cif1_s0"], 10) for k in range(1, k_total + 1)]
    truth_cif2_s1 = [0.0] + [round(curve_by_period[k]["cif2_s1"], 10) for k in range(1, k_total + 1)]
    truth_cif2_s0 = [0.0] + [round(curve_by_period[k]["cif2_s0"], 10) for k in range(1, k_total + 1)]

    return {
        "truth_auxiliary_seed": truth_seed,
        "n_truth_counterfactual": n_truth,
        "horizons": horizons_dict,
        "horizon_truth_table": horizon_table,
        "curves": {
            "time_days": truth_time_days,
            "truth_cif_initiate_within_grace": truth_cif1_s1,
            "truth_cif_never_initiate": truth_cif1_s0,
            "truth_cif2_initiate_within_grace": truth_cif2_s1,
            "truth_cif2_never_initiate": truth_cif2_s0,
        },
    }


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_synthetic_ccw_cohort(
    protocol: Path | str | dict,
    seed: int,
    n_cases: int,
    dgp_params: Path | str | dict | None = None,
    *,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a deterministic synthetic delayed-initiation competing-risks cohort and DGP truth.

    When out_dir is provided, atomically publishes cohort.csv, dgp_truth.json, PURPOSE.txt,
    (optional person_period_covariates.csv), and run_manifest.json.
    """
    if out_dir is not None:
        check_destination(out_dir)

    if not is_strict_int(seed) or not (0 <= seed <= MAX_SEED):
        raise ContractError("INVALID_SEED", f"seed must be an integer in 0..{MAX_SEED}")
    if not is_strict_int(n_cases) or not (1 <= n_cases <= MAX_SYNTH_CASES):
        raise ContractError("INVALID_CASE_COUNT", f"n_cases must be an integer in 1..{MAX_SYNTH_CASES}")

    raw_proto, norm_proto = _resolve_ccw_protocol_input(protocol)
    raw_dgp_bytes, override_dgp = _resolve_ccw_dgp_override(dgp_params)

    base_covs = norm_proto["covariates"]["baseline_covariates"]
    tv_covs = norm_proto["covariates"]["time_varying_covariates"]
    dgp = _merge_and_validate_ccw_dgp_params(
        _default_ccw_dgp_parameters(len(base_covs), len(tv_covs)),
        override_dgp,
        num_base_covs=len(base_covs),
        num_tv_covs=len(tv_covs),
    )

    gp = norm_proto["grace_period_spec"]
    delta = float(gp["period_length_days"])
    grace_days = float(gp["grace_period_days"])
    max_followup = float(gp["max_followup_days"])
    m_g = int(round(grace_days / delta))
    k_total = int(round(max_followup / delta))

    rng = np.random.default_rng(seed)
    X_base = _sample_baseline_covariate_matrix(rng, n_cases, base_covs)
    X_tv = _sample_tv_covariates_cube(rng, X_base, tv_covs, k_total)

    alpha_base = X_base @ np.asarray(dgp["initiation_covariate_slopes"], dtype=np.float64)
    beta1_base = X_base @ np.asarray(dgp["outcome_covariate_log_hr_event1"], dtype=np.float64)
    beta2_base = X_base @ np.asarray(dgp["outcome_covariate_log_hr_event2"], dtype=np.float64)
    adm_base = X_base @ np.asarray(dgp["admin_censor_covariate_log_hr"], dtype=np.float64)

    alpha_tv_vec = np.asarray(dgp["initiation_tv_covariate_slopes"], dtype=np.float64)
    beta1_tv_vec = np.asarray(dgp["outcome_tv_covariate_log_hr_event1"], dtype=np.float64)
    beta2_tv_vec = np.asarray(dgp["outcome_tv_covariate_log_hr_event2"], dtype=np.float64)

    bh1 = float(dgp["baseline_daily_hazard_event1"])
    bh2 = float(dgp["baseline_daily_hazard_event2"])
    adm_h = float(dgp["admin_censor_daily_hazard"])
    theta1 = float(dgp["true_log_hr_event1"])
    theta2 = float(dgp["true_log_hr_event2"])

    active = np.ones(n_cases, dtype=bool)
    initiated = np.zeros(n_cases, dtype=bool)
    init_day = np.full(n_cases, np.nan, dtype=np.float64)
    duration_day = np.full(n_cases, max_followup, dtype=np.float64)
    event_code = np.zeros(n_cases, dtype=np.int64)
    exit_period = np.full(n_cases, k_total, dtype=np.int64)

    for k in range(1, k_total + 1):
        if not np.any(active):
            break
        k_idx = k - 1
        tv_init = X_tv[:, k_idx, :] @ alpha_tv_vec if tv_covs else np.zeros(n_cases, dtype=np.float64)
        tv_b1 = X_tv[:, k_idx, :] @ beta1_tv_vec if tv_covs else np.zeros(n_cases, dtype=np.float64)
        tv_b2 = X_tv[:, k_idx, :] @ beta2_tv_vec if tv_covs else np.zeros(n_cases, dtype=np.float64)

        # 1. Start-of-period treatment initiation decision for uninitiated active subjects
        a0_k = dgp["grace_initiation_intercept"] if k <= m_g else dgp["post_grace_initiation_intercept"]
        p_init = expit(a0_k + alpha_base + tv_init)
        u_init_draw = rng.random(n_cases)
        u_init_pos = rng.uniform(0.05, 0.35, size=n_cases)

        can_init = active & (~initiated)
        new_init = can_init & (u_init_draw < p_init)
        if np.any(new_init):
            initiated[new_init] = True
            init_day[new_init] = np.round(((k - 1) + u_init_pos[new_init]) * delta, 6)

        # 2. Within-period outcome (1=primary, 2=competing) conditional on current treatment status
        z_curr = initiated.astype(np.float64)
        lam1 = bh1 * np.exp(beta1_base + tv_b1 + theta1 * z_curr)
        lam2 = bh2 * np.exp(beta2_base + tv_b2 + theta2 * z_curr)
        tot_lam = lam1 + lam2
        ev_prob = -np.expm1(-tot_lam * delta)
        q1 = np.where(tot_lam > 0.0, (lam1 / np.maximum(tot_lam, 1e-30)) * ev_prob, 0.0)
        q2 = np.where(tot_lam > 0.0, (lam2 / np.maximum(tot_lam, 1e-30)) * ev_prob, 0.0)

        u_ev_draw = rng.random(n_cases)
        u_ev_pos = rng.uniform(0.55, 0.98, size=n_cases)

        ev1_mask = active & (u_ev_draw < q1)
        ev2_mask = active & (~ev1_mask) & (u_ev_draw < (q1 + q2))
        ev_any = ev1_mask | ev2_mask
        if np.any(ev_any):
            duration_day[ev_any] = np.round(((k - 1) + u_ev_pos[ev_any]) * delta, 6)
            event_code[ev1_mask] = 1
            event_code[ev2_mask] = 2
            exit_period[ev_any] = k
            active[ev_any] = False

        # 3. End-of-period administrative right-censoring (for k < K)
        if k < k_total and adm_h > 0.0 and np.any(active):
            lam_c = adm_h * np.exp(adm_base)
            p_cens = -np.expm1(-lam_c * delta)
            u_cens = rng.random(n_cases)
            cens_mask = active & (u_cens < p_cens)
            if np.any(cens_mask):
                duration_day[cens_mask] = round(k * delta, 6)
                event_code[cens_mask] = 0
                exit_period[cens_mask] = k
                active[cens_mask] = False

    truth_info = _compute_ccw_counterfactual_truth(protocol=norm_proto, dgp=dgp, seed=seed)

    base_cov_names = [c["name"] for c in base_covs]
    cohort_fieldnames = [
        "subject_id",
        "simulation_or_data_label",
        "initiation_day",
        "duration_day",
        "event_code",
        *base_cov_names,
    ]
    cohort_rows: list[dict[str, Any]] = []
    for i in range(n_cases):
        sid = f"SYN-{i + 1:06d}"
        row: dict[str, Any] = {
            "subject_id": sid,
            "simulation_or_data_label": norm_proto["simulation_or_data_label"],
            "initiation_day": "" if np.isnan(init_day[i]) else f"{init_day[i]:.6f}",
            "duration_day": f"{duration_day[i]:.6f}",
            "event_code": int(event_code[i]),
        }
        for j, cname in enumerate(base_cov_names):
            val = X_base[i, j]
            if base_covs[j]["type"] == "binary":
                row[cname] = int(round(val))
            else:
                row[cname] = f"{val:.6f}"
        cohort_rows.append(row)

    pp_rows: list[dict[str, Any]] = []
    pp_fieldnames: list[str] = []
    if tv_covs:
        tv_names = [c["name"] for c in tv_covs]
        pp_fieldnames = [
            "subject_id",
            "period_idx",
            *tv_names,
            *[f"{name}__assessed_day" for name in tv_names],
        ]
        for i in range(n_cases):
            sid = f"SYN-{i + 1:06d}"
            max_k_i = int(exit_period[i])
            for k in range(1, max_k_i + 1):
                pp_row: dict[str, Any] = {
                    "subject_id": sid,
                    "period_idx": k,
                }
                for j, spec in enumerate(tv_covs):
                    cname = spec["name"]
                    val = X_tv[i, k - 1, j]
                    pp_row[cname] = int(round(val)) if spec["type"] == "binary" else f"{val:.6f}"
                    meas_day = (k - 1) * delta + float(spec["offset_from_period_start_days"])
                    pp_row[f"{cname}__assessed_day"] = f"{meas_day:.6f}"
                pp_rows.append(pp_row)

    n_init_grace = int(np.sum((~np.isnan(init_day)) & (init_day <= grace_days)))
    n_init_late = int(np.sum((~np.isnan(init_day)) & (init_day > grace_days)))
    n_never = int(np.sum(np.isnan(init_day)))

    dgp_truth_payload = {
        "parameter_type": "synthetic_ccw_dgp_parameters",
        "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
        "prohibited_claims": list(PROHIBITED_CLAIMS),
        "protocol_id": norm_proto["protocol_id"],
        "protocol_version": CCW_PROTOCOL_VERSION,
        "simulation_or_data_label": norm_proto["simulation_or_data_label"],
        "methodological_disclaimer": norm_proto["methodological_disclaimer"],
        "seed": seed,
        "n_cases": n_cases,
        "truth_auxiliary_seed": truth_info["truth_auxiliary_seed"],
        "n_truth_counterfactual": truth_info["n_truth_counterfactual"],
        "grace_period_spec": norm_proto["grace_period_spec"],
        "dgp_parameters": dgp,
        "observed_cohort_summary": {
            "initiated_within_grace": n_init_grace,
            "initiated_after_grace": n_init_late,
            "never_initiated": n_never,
            "event1_count": int(np.sum(event_code == 1)),
            "event2_count": int(np.sum(event_code == 2)),
            "admin_censored_count": int(np.sum(event_code == 0)),
        },
        "horizons": truth_info["horizons"],
        "horizon_truth_table": truth_info["horizon_truth_table"],
        "curves": truth_info["curves"],
        "truth_cif_initiate_within_grace": truth_info["curves"]["truth_cif_initiate_within_grace"],
        "truth_cif_never_initiate": truth_info["curves"]["truth_cif_never_initiate"],
    }

    if out_dir is None:
        df_mem = pd.DataFrame(
            {
                "subject_id": [f"SYN-{i + 1:06d}" for i in range(n_cases)],
                "simulation_or_data_label": norm_proto["simulation_or_data_label"],
                "initiation_day": init_day,
                "duration_day": duration_day,
                "event_code": event_code,
                **{
                    c["name"]: (
                        X_base[:, j].astype(np.int64)
                        if c["type"] == "binary"
                        else X_base[:, j].astype(np.float64)
                    )
                    for j, c in enumerate(base_covs)
                },
            }
        )
        pp_df = pd.DataFrame(pp_rows) if tv_covs else None
        return {
            "status": "synthetic_ccw_cohort_generated",
            "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
            "protocol": norm_proto,
            "cohort_df": df_mem,
            "person_period_df": pp_df,
            "dgp_truth": dgp_truth_payload,
        }

    commit, code_dirty = code_provenance()
    deps = dependency_versions()
    purpose_text = (
        "SYNTHETIC-ONLY BENCHMARK COHORT (tte-ccw-v1)\n"
        f"Protocol ID: {norm_proto['protocol_id']}\n"
        f"Label: {norm_proto['simulation_or_data_label']}\n"
        f"Seed: {seed} | Cases: {n_cases}\n\n"
        + "\n".join(PROHIBITED_CLAIMS)
        + "\n"
    )

    output_paths = ["PURPOSE.txt", "cohort.csv", "dgp_truth.json"]
    if tv_covs:
        output_paths.append("person_period_covariates.csv")
    output_paths.sort()

    def _populate_synth(stage: Path) -> None:
        _write_csv_rows(stage / "cohort.csv", cohort_fieldnames, cohort_rows)
        if tv_covs:
            _write_csv_rows(stage / "person_period_covariates.csv", pp_fieldnames, pp_rows)
        (stage / "dgp_truth.json").write_bytes(json_bytes(dgp_truth_payload))
        (stage / "PURPOSE.txt").write_bytes(purpose_text.encode("utf-8"))
        manifest = {
            "status": "synthetic_ccw_cohort_generated",
            "activity": "tte_ccw_synthetic_cohort_generation",
            "protocol_id": norm_proto["protocol_id"],
            "protocol_version": CCW_PROTOCOL_VERSION,
            "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
            "prohibited_claims": list(PROHIBITED_CLAIMS),
            "simulation_or_data_label": norm_proto["simulation_or_data_label"],
            "methodological_disclaimer": norm_proto["methodological_disclaimer"],
            "seed": seed,
            "case_count": n_cases,
            "protocol_sha256": sha256_bytes(raw_proto),
            "normalized_protocol_sha256": sha256_bytes(json_bytes(norm_proto)),
            "dgp_params_sha256": sha256_bytes(raw_dgp_bytes) if raw_dgp_bytes is not None else None,
            "code_commit": commit,
            "code_dirty": code_dirty,
            "dependency_versions": deps,
            "output_paths": output_paths,
            "output_sha256": {p: sha256_file(stage / p) for p in output_paths},
            "run_at_utc": utc_now_iso(),
        }
        (stage / "run_manifest.json").write_bytes(json_bytes(manifest))

    atomic_publish(out_dir, _populate_synth)
    return {
        "status": "synthetic_ccw_cohort_generated",
        "protocol_id": norm_proto["protocol_id"],
        "protocol_version": CCW_PROTOCOL_VERSION,
        "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
        "simulation_or_data_label": norm_proto["simulation_or_data_label"],
        "seed": seed,
        "case_count": n_cases,
        "observed_cohort_summary": dgp_truth_payload["observed_cohort_summary"],
        "output_path": str(out_dir),
    }


# ---------------------------------------------------------------------------
# Module 3: Bilateral Clone-Censor-Weight, Arm-Specific IPCW Gates & Bootstrap
# ---------------------------------------------------------------------------


def _load_cohort_dataframe(cohort: pd.DataFrame | Path | str) -> pd.DataFrame:
    if isinstance(cohort, pd.DataFrame):
        return cohort.copy()
    if isinstance(cohort, (Path, str)):
        p = Path(cohort)
        try:
            return pd.read_csv(p, dtype={"subject_id": str})
        except (OSError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
            raise ContractError("COHORT_IO_ERROR", f"Failed to read cohort CSV '{p}': {exc}") from exc
    raise ContractError("INVALID_TYPE", "cohort must be a pandas.DataFrame or CSV Path")


def _load_person_period_dataframe(pp: pd.DataFrame | Path | str | None) -> pd.DataFrame | None:
    if pp is None:
        return None
    if isinstance(pp, pd.DataFrame):
        return pp.copy()
    if isinstance(pp, (Path, str)):
        p = Path(pp)
        try:
            return pd.read_csv(p, dtype={"subject_id": str})
        except (OSError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
            raise ContractError("COHORT_IO_ERROR", f"Failed to read person-period CSV '{p}': {exc}") from exc
    raise ContractError("INVALID_TYPE", "person_period_covariates must be None, DataFrame, or CSV Path")


def validate_ccw_cohort_df(
    cohort: pd.DataFrame | Path | str,
    protocol: dict,
    person_period_covariates: pd.DataFrame | Path | str | None = None,
) -> dict[str, Any]:
    """Validate synthetic CCW cohort & optional pre-period covariates against tte-ccw-v1 rules."""
    df = _load_cohort_dataframe(cohort)
    if len(df) == 0:
        raise ContractError("EMPTY_COHORT", "CCW input cohort must contain at least 1 subject")

    base_covs = protocol["covariates"]["baseline_covariates"]
    tv_covs = protocol["covariates"]["time_varying_covariates"]
    base_names = [c["name"] for c in base_covs]

    required_cols = ["subject_id", "initiation_day", "duration_day", "event_code", *base_names]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ContractError(
            "MISSING_COHORT_COLUMN",
            f"Cohort is missing required column(s): {', '.join(missing_cols)}",
        )

    # 1. Strict subject_id non-empty, unique, and 100% SYN-* prefix
    raw_ids = df["subject_id"]
    if raw_ids.isna().any():
        raise ContractError("INVALID_SUBJECT_ID", "subject_id contains missing/null entries")
    subj_ids: list[str] = []
    for idx, val in enumerate(raw_ids.tolist()):
        if not isinstance(val, str) or not val.strip():
            raise ContractError("INVALID_SUBJECT_ID", f"Row {idx} has empty or non-string subject_id")
        s = val.strip()
        if not s.startswith("SYN-"):
            raise ContractError(
                "NON_SYNTHETIC_SUBJECT_ID",
                f"Row {idx} subject_id '{s}' violates synthetic_only guardrail: all subject_id values "
                "must carry the 'SYN-' prefix.",
            )
        subj_ids.append(s)
    if len(set(subj_ids)) != len(subj_ids):
        raise ContractError("DUPLICATE_SUBJECT_ID", "Cohort contains duplicate subject_id values")

    # 2. duration_day > 0
    dur_raw = df["duration_day"]
    if dur_raw.map(lambda v: isinstance(v, (bool, np.bool_))).any():
        raise ContractError("INVALID_DURATION_DAY", "duration_day must be numeric, not boolean")
    dur_num = pd.to_numeric(dur_raw, errors="coerce").to_numpy(dtype=np.float64)
    if np.any(np.isnan(dur_num)) or not np.all(np.isfinite(dur_num)) or np.any(dur_num <= 0.0):
        raise ContractError("INVALID_DURATION_DAY", "duration_day must be finite and strictly > 0 for all subjects")

    # 3. event_code in {0, 1, 2}
    ev_raw = df["event_code"]
    if ev_raw.map(lambda v: isinstance(v, (bool, np.bool_))).any():
        raise ContractError("INVALID_EVENT_CODE", "event_code must be integer in {0, 1, 2}, not boolean")
    ev_num = pd.to_numeric(ev_raw, errors="coerce").to_numpy(dtype=np.float64)
    if np.any(np.isnan(ev_num)) or not np.all(np.isfinite(ev_num)):
        raise ContractError("INVALID_EVENT_CODE", "event_code contains missing or non-numeric values")
    if np.any(ev_num != np.floor(ev_num)):
        raise ContractError("INVALID_EVENT_CODE", "event_code must be an exact integer in {0, 1, 2}")
    ev_int = ev_num.astype(np.int64)
    if not np.all(np.isin(ev_int, (0, 1, 2))):
        raise ContractError("INVALID_EVENT_CODE", "event_code values must belong to {0, 1, 2}")

    # 4. initiation_day >= 0 or missing
    init_arr = np.full(len(df), np.nan, dtype=np.float64)
    for idx, val in enumerate(df["initiation_day"].tolist()):
        if val is None or (isinstance(val, str) and val.strip() == ""):
            continue
        if isinstance(val, (float, np.floating)) and math.isnan(float(val)):
            continue
        if pd.isna(val):
            continue
        if isinstance(val, (bool, np.bool_)):
            raise ContractError("INVALID_INITIATION_DAY", f"Row {idx} initiation_day cannot be boolean")
        try:
            fval = float(val)
        except (TypeError, ValueError) as exc:
            raise ContractError("INVALID_INITIATION_DAY", f"Row {idx} initiation_day '{val}' is non-numeric") from exc
        if not math.isfinite(fval) or fval < 0.0:
            raise ContractError(
                "INVALID_INITIATION_DAY",
                f"Row {idx} initiation_day={fval} must be non-negative (>= 0) or missing",
            )
        init_arr[idx] = fval

    # 5. Embedded baseline covariate offset leak check + numeric check
    for col in df.columns:
        for suffix in ("__assessed_day_offset", "_assessed_day_offset"):
            if col.endswith(suffix) and len(col) > len(suffix):
                target = col[: -len(suffix)]
                offs = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
                if np.any(np.isnan(offs)) or not np.all(np.isfinite(offs)):
                    raise ContractError("INVALID_COHORT_SCHEMA", f"Offset column '{col}' has non-numeric values")
                if float(np.max(offs)) > 0.0:
                    raise ContractError(
                        "POST_BASELINE_COVARIATE_LEAK",
                        f"Baseline covariate '{target}' has embedded offset {float(np.max(offs)):g} > 0 in '{col}'",
                    )

    X_base = np.zeros((len(df), len(base_names)), dtype=np.float64)
    for j, cname in enumerate(base_names):
        col_s = df[cname]
        if col_s.map(lambda v: isinstance(v, (bool, np.bool_))).any():
            raise ContractError("INVALID_COVARIATE_VALUES", f"Covariate '{cname}' must be numeric, not boolean")
        cnum = pd.to_numeric(col_s, errors="coerce").to_numpy(dtype=np.float64)
        if np.any(np.isnan(cnum)) or not np.all(np.isfinite(cnum)):
            raise ContractError(
                "INVALID_COVARIATE_VALUES",
                f"Baseline covariate '{cname}' contains missing or non-finite values",
            )
        X_base[:, j] = cnum

    # 6. Optional pre-period time-varying covariates
    gp = protocol["grace_period_spec"]
    delta = float(gp["period_length_days"])
    max_followup = float(gp["max_followup_days"])
    k_total = int(round(max_followup / delta))

    T_eff = np.minimum(dur_num, max_followup)
    E_eff = np.where(dur_num > max_followup, 0, ev_int).astype(np.int64)
    k_exit = np.maximum(1, np.ceil(T_eff / delta - 1e-12).astype(np.int64))
    k_exit = np.minimum(k_exit, k_total)

    pp_df = _load_person_period_dataframe(person_period_covariates)
    X_tv = np.zeros((len(df), k_total, len(tv_covs)), dtype=np.float64)
    if tv_covs:
        if pp_df is None:
            raise ContractError(
                "MISSING_TIME_VARYING_DATA",
                "Protocol declares time_varying_covariates, so person_period_covariates table is required",
            )
        tv_names = [c["name"] for c in tv_covs]
        req_pp = ["subject_id", "period_idx", *tv_names]
        miss_pp = [c for c in req_pp if c not in pp_df.columns]
        if miss_pp:
            raise ContractError(
                "MISSING_COHORT_COLUMN",
                f"person_period_covariates is missing column(s): {', '.join(miss_pp)}",
            )
        pidx_num = pd.to_numeric(pp_df["period_idx"], errors="coerce").to_numpy(dtype=np.float64)
        if np.any(np.isnan(pidx_num)) or np.any(pidx_num != np.floor(pidx_num)):
            raise ContractError("INVALID_PERIOD_INDEX", "period_idx in person_period_covariates must be integer")
        pidx_int = pidx_num.astype(np.int64)
        if np.any((pidx_int < 1) | (pidx_int > k_total)):
            raise ContractError("INVALID_PERIOD_INDEX", f"period_idx must lie in 1..{k_total}")

        # Anti-time-travel check on explicit measurement day columns if present
        for cname in tv_names:
            for mcol in (f"{cname}__assessed_day", f"{cname}_assessed_day"):
                if mcol in pp_df.columns:
                    mdays = pd.to_numeric(pp_df[mcol], errors="coerce").to_numpy(dtype=np.float64)
                    if np.any(np.isnan(mdays)) or not np.all(np.isfinite(mdays)):
                        raise ContractError("INVALID_COHORT_SCHEMA", f"'{mcol}' contains non-numeric values")
                    decision_starts = (pidx_int - 1) * delta
                    viol = np.flatnonzero(mdays >= decision_starts - 1e-12)
                    if viol.size > 0:
                        r0 = int(viol[0])
                        raise ContractError(
                            "TIME_VARYING_COVARIATE_TIME_TRAVEL",
                            f"Row {r0} of person_period_covariates has {mcol}={mdays[r0]:g} >= "
                            f"period {pidx_int[r0]} decision start {decision_starts[r0]:g}. "
                            "Hint: Pre-period time-varying covariates must be measured strictly prior to "
                            "(k-1)*period_length_days.",
                        )

        subj_to_row = {sid: idx for idx, sid in enumerate(subj_ids)}
        filled = np.zeros((len(df), k_total), dtype=bool)
        pp_sids = pp_df["subject_id"].astype(str).str.strip().tolist()
        tv_vals = np.zeros((len(pp_df), len(tv_names)), dtype=np.float64)
        for j, cname in enumerate(tv_names):
            cnum = pd.to_numeric(pp_df[cname], errors="coerce").to_numpy(dtype=np.float64)
            if np.any(np.isnan(cnum)) or not np.all(np.isfinite(cnum)):
                raise ContractError(
                    "INVALID_COVARIATE_VALUES",
                    f"Time-varying covariate '{cname}' has missing or non-finite values",
                )
            tv_vals[:, j] = cnum

        for r_idx, sid in enumerate(pp_sids):
            if not sid.startswith("SYN-"):
                raise ContractError(
                    "NON_SYNTHETIC_SUBJECT_ID",
                    f"person_period_covariates subject_id '{sid}' must carry 'SYN-' prefix",
                )
            if sid not in subj_to_row:
                continue
            i_subj = subj_to_row[sid]
            k_val = int(pidx_int[r_idx])
            if filled[i_subj, k_val - 1]:
                raise ContractError(
                    "DUPLICATE_PERSON_PERIOD",
                    f"Duplicate person-period record for ({sid}, period_idx={k_val})",
                )
            filled[i_subj, k_val - 1] = True
            X_tv[i_subj, k_val - 1, :] = tv_vals[r_idx, :]

        for i_subj in range(len(df)):
            req_k = int(k_exit[i_subj])
            if not np.all(filled[i_subj, :req_k]):
                miss_k = int(np.flatnonzero(~filled[i_subj, :req_k])[0] + 1)
                raise ContractError(
                    "MISSING_PERSON_PERIOD",
                    f"Subject {subj_ids[i_subj]} is missing person-period covariate record for period_idx={miss_k}",
                )

    has_init = (~np.isnan(init_arr)) & (init_arr <= T_eff + 1e-12)
    init_day_eff = np.where(has_init, init_arr, np.nan)
    k_init = np.zeros(len(df), dtype=np.int64)
    if np.any(has_init):
        raw_ki = np.ceil(init_day_eff[has_init] / delta - 1e-12).astype(np.int64)
        k_init[has_init] = np.clip(np.maximum(1, raw_ki), 1, k_total)

    return {
        "df": df,
        "subject_ids": subj_ids,
        "duration_raw": dur_num,
        "event_raw": ev_int,
        "initiation_raw": init_arr,
        "T_eff": T_eff,
        "E_eff": E_eff,
        "has_init": has_init,
        "init_day_eff": init_day_eff,
        "k_init": k_init,
        "k_exit": k_exit,
        "X_base": X_base,
        "X_tv": X_tv,
        "base_cov_names": base_names,
        "tv_cov_names": [c["name"] for c in tv_covs],
    }


def clone_and_censor(
    cohort: pd.DataFrame | Path | str,
    protocol: dict | Path | str,
    person_period_covariates: pd.DataFrame | Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Perform symmetric bilateral cloning and artificial censoring for grace-period TTE.

    Returns
    -------
    clones_df : pandas.DataFrame
        2N-row DataFrame (Strategy 0 'never_initiate' and Strategy 1 'initiate_within_grace')
        with columns subject_id, clone_arm, clone_strategy, initiation_day, duration_day,
        event_code, T_clone, event_clone, artificial_censored, censor_period, last_uncensored_period,
        and baseline covariates.
    ccw_attrition : dict
        Detailed bilateral clone and period-by-period artificial censoring / event attrition summary.
    """
    _, norm_proto = _resolve_ccw_protocol_input(protocol)
    validated = validate_ccw_cohort_df(cohort, norm_proto, person_period_covariates=person_period_covariates)

    gp = norm_proto["grace_period_spec"]
    delta = float(gp["period_length_days"])
    grace_days = float(gp["grace_period_days"])
    max_followup = float(gp["max_followup_days"])
    m_g = int(round(grace_days / delta))
    k_total = int(round(max_followup / delta))

    subj_ids = validated["subject_ids"]
    T_eff = validated["T_eff"]
    E_eff = validated["E_eff"]
    has_init = validated["has_init"]
    init_day_eff = validated["init_day_eff"]
    k_init = validated["k_init"]
    k_exit = validated["k_exit"]
    X_base = validated["X_base"]
    base_names = validated["base_cov_names"]
    n = len(subj_ids)

    # Strategy 0 ('never_initiate'): artificially censored at start of period k_init (prior to period k_init outcome)
    s0_art_cens = has_init.astype(np.int64)
    s0_T_clone = np.where(has_init, (k_init - 1) * delta, T_eff)
    s0_E_clone = np.where(has_init, 0, E_eff).astype(np.int64)
    s0_censor_period = [int(k_init[i]) if has_init[i] else None for i in range(n)]
    s0_last_period = np.where(has_init, k_init - 1, k_exit).astype(np.int64)

    # Strategy 1 ('initiate_within_grace'):
    #   - If T_eff <= g: never violates strategy (artificial_censored = 0, keeps T_eff, E_eff)
    #   - If T_eff > g: adherent iff has_init and init_day_eff <= g; else censored at t=g (period m_g + 1)
    adherent_by_g = has_init & (init_day_eff <= grace_days + 1e-12)
    s1_violates_at_g = (T_eff > grace_days + 1e-12) & (~adherent_by_g)
    s1_art_cens = s1_violates_at_g.astype(np.int64)
    s1_T_clone = np.where(s1_violates_at_g, grace_days, T_eff)
    s1_E_clone = np.where(s1_violates_at_g, 0, E_eff).astype(np.int64)
    s1_censor_period = [int(m_g + 1) if s1_violates_at_g[i] else None for i in range(n)]
    s1_last_period = np.where(s1_violates_at_g, m_g, k_exit).astype(np.int64)

    df0_dict: dict[str, Any] = {
        "subject_id": subj_ids,
        "clone_arm": np.zeros(n, dtype=np.int64),
        "clone_strategy": ["never_initiate"] * n,
        "initiation_day": validated["initiation_raw"],
        "duration_day": validated["duration_raw"],
        "event_code": validated["event_raw"],
        "T_clone": s0_T_clone,
        "event_clone": s0_E_clone,
        "artificial_censored": s0_art_cens,
        "censor_period": s0_censor_period,
        "last_uncensored_period": s0_last_period,
    }
    df1_dict: dict[str, Any] = {
        "subject_id": subj_ids,
        "clone_arm": np.ones(n, dtype=np.int64),
        "clone_strategy": ["initiate_within_grace"] * n,
        "initiation_day": validated["initiation_raw"],
        "duration_day": validated["duration_raw"],
        "event_code": validated["event_raw"],
        "T_clone": s1_T_clone,
        "event_clone": s1_E_clone,
        "artificial_censored": s1_art_cens,
        "censor_period": s1_censor_period,
        "last_uncensored_period": s1_last_period,
    }
    for j, cname in enumerate(base_names):
        df0_dict[cname] = X_base[:, j]
        df1_dict[cname] = X_base[:, j]

    clones_df = pd.concat([pd.DataFrame(df0_dict), pd.DataFrame(df1_dict)], ignore_index=True)

    # Period-by-period attrition audit tables
    s1_period_rows: list[dict[str, Any]] = []
    s0_period_rows: list[dict[str, Any]] = []
    for k in range(1, k_total + 1):
        t_start = (k - 1) * delta
        t_end = k * delta

        # Strategy 1 period k
        if k <= m_g:
            s1_enter = T_eff > t_start + 1e-12
            s1_art_k = np.zeros(n, dtype=bool)
            s1_risk = s1_enter
        elif k == m_g + 1:
            s1_enter = T_eff > t_start + 1e-12
            s1_art_k = s1_enter & (~adherent_by_g)
            s1_risk = s1_enter & adherent_by_g
        else:
            s1_enter = (T_eff > t_start + 1e-12) & adherent_by_g
            s1_art_k = np.zeros(n, dtype=bool)
            s1_risk = s1_enter

        s1_ev_in_k = s1_risk & (k_exit == k)
        s1_period_rows.append(
            {
                "period_idx": k,
                "start_day": round(t_start, 6),
                "end_day": round(t_end, 6),
                "in_grace_window": bool(k <= m_g),
                "entering_pre_decision_n": int(np.sum(s1_enter)),
                "artificially_censored_at_period_start_n": int(np.sum(s1_art_k)),
                "at_risk_for_outcome_n": int(np.sum(s1_risk)),
                "primary_events_n": int(np.sum(s1_ev_in_k & (E_eff == 1))),
                "competing_events_n": int(np.sum(s1_ev_in_k & (E_eff == 2))),
                "admin_censored_n": int(np.sum(s1_ev_in_k & (E_eff == 0))),
            }
        )

        # Strategy 0 period k
        s0_enter = (T_eff > t_start + 1e-12) & ((~has_init) | (k_init >= k))
        s0_art_k = s0_enter & has_init & (k_init == k)
        s0_risk = s0_enter & (~s0_art_k)
        s0_ev_in_k = s0_risk & (k_exit == k)
        s0_period_rows.append(
            {
                "period_idx": k,
                "start_day": round(t_start, 6),
                "end_day": round(t_end, 6),
                "in_grace_window": bool(k <= m_g),
                "entering_pre_decision_n": int(np.sum(s0_enter)),
                "artificially_censored_at_period_start_n": int(np.sum(s0_art_k)),
                "at_risk_for_outcome_n": int(np.sum(s0_risk)),
                "primary_events_n": int(np.sum(s0_ev_in_k & (E_eff == 1))),
                "competing_events_n": int(np.sum(s0_ev_in_k & (E_eff == 2))),
                "admin_censored_n": int(np.sum(s0_ev_in_k & (E_eff == 0))),
            }
        )

    pre_g_mask = T_eff <= grace_days + 1e-12
    shared_early_uninit = pre_g_mask & (~has_init) & (E_eff != 0)
    ccw_attrition = {
        "protocol_id": norm_proto["protocol_id"],
        "protocol_version": CCW_PROTOCOL_VERSION,
        "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
        "prohibited_claims": list(PROHIBITED_CLAIMS),
        "grace_period_days": gp["grace_period_days"],
        "period_length_days": gp["period_length_days"],
        "max_followup_days": gp["max_followup_days"],
        "grace_period_steps": m_g,
        "total_periods": k_total,
        "total_subjects": n,
        "total_clones_created": 2 * n,
        "shared_early_events_within_grace_uninitiated": {
            "total_events": int(np.sum(shared_early_uninit)),
            "primary_events": int(np.sum(shared_early_uninit & (E_eff == 1))),
            "competing_events": int(np.sum(shared_early_uninit & (E_eff == 2))),
        },
        "strategy_1_initiate_within_grace": {
            "initial_clones": n,
            "pre_or_at_grace_primary_events": int(np.sum(pre_g_mask & (E_eff == 1))),
            "pre_or_at_grace_competing_events": int(np.sum(pre_g_mask & (E_eff == 2))),
            "pre_or_at_grace_admin_censored": int(np.sum(pre_g_mask & (E_eff == 0))),
            "surviving_past_grace_boundary": int(np.sum(~pre_g_mask)),
            "artificially_censored_at_grace_end": int(np.sum(s1_violates_at_g)),
            "uncensored_entering_post_grace": int(np.sum((~pre_g_mask) & adherent_by_g)),
            "post_grace_primary_events": int(np.sum((~pre_g_mask) & adherent_by_g & (E_eff == 1))),
            "post_grace_competing_events": int(np.sum((~pre_g_mask) & adherent_by_g & (E_eff == 2))),
            "post_grace_admin_censored": int(np.sum((~pre_g_mask) & adherent_by_g & (E_eff == 0))),
            "period_table": s1_period_rows,
        },
        "strategy_0_never_initiate": {
            "initial_clones": n,
            "total_artificially_censored_on_initiation": int(np.sum(s0_art_cens)),
            "artificially_censored_within_grace": int(np.sum(has_init & (k_init <= m_g))),
            "artificially_censored_post_grace": int(np.sum(has_init & (k_init > m_g))),
            "uncensored_entering_post_grace": int(s0_period_rows[m_g]["at_risk_for_outcome_n"]),
            "uncensored_primary_events": int(np.sum((~has_init) & (E_eff == 1))),
            "uncensored_competing_events": int(np.sum((~has_init) & (E_eff == 2))),
            "uncensored_admin_censored": int(np.sum((~has_init) & (E_eff == 0))),
            "period_table": s0_period_rows,
        },
    }
    return clones_df, ccw_attrition


def _fit_logistic_with_col_guard(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit unpenalized MLE binary logistic regression while dropping zero-variance columns cleanly."""
    n = X.shape[0]
    y_mean = float(np.mean(y))
    if n == 0 or y_mean <= 0.0 or y_mean >= 1.0:
        return np.full(n, y_mean if n > 0 else 1.0, dtype=np.float64)
    if X.shape[1] == 0:
        return np.full(n, y_mean, dtype=np.float64)
    col_ptp = np.ptp(X, axis=0)
    keep = col_ptp > 1e-12
    if not np.any(keep):
        return np.full(n, y_mean, dtype=np.float64)
    return fit_binary_logistic_mle(X[:, keep], y)


def _fit_and_predict_logistic_with_col_guard(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-13,
) -> np.ndarray:
    """Fit unpenalized damped IRLS logistic regression on (X_train, y_train) and predict on X_eval."""
    n_tr = X_train.shape[0]
    n_ev = X_eval.shape[0]
    if n_ev == 0:
        return np.zeros(0, dtype=np.float64)
    if n_tr == 0:
        return np.full(n_ev, 0.5, dtype=np.float64)
    y_mean = float(np.mean(y_train))
    if y_mean <= 0.0 or y_mean >= 1.0 or X_train.shape[1] == 0:
        return np.full(n_ev, y_mean, dtype=np.float64)

    keep = np.ptp(X_train, axis=0) > 1e-12
    if not np.any(keep):
        return np.full(n_ev, y_mean, dtype=np.float64)

    X_sub = X_train[:, keep].astype(np.float64)
    p = X_sub.shape[1]
    X_aug = np.column_stack([np.ones(n_tr, dtype=np.float64), X_sub])
    y_f = y_train.astype(np.float64)
    beta = np.zeros(p + 1, dtype=np.float64)

    def _neg_loglik(b: np.ndarray) -> float:
        eta = np.clip(X_aug @ b, -35.0, 35.0)
        return float(np.sum(np.logaddexp(0.0, eta) - y_f * eta))

    nll = _neg_loglik(beta)
    for _ in range(max_iter):
        eta = np.clip(X_aug @ beta, -35.0, 35.0)
        prob = expit(eta)
        grad = X_aug.T @ (prob - y_f)
        if float(np.max(np.abs(grad))) <= tol:
            break
        w = prob * (1.0 - prob)
        if float(np.max(w)) < 1e-15:
            break
        hess = (X_aug * np.maximum(w, 1e-14)[:, None]).T @ X_aug
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(hess + 1e-12 * np.eye(p + 1), grad, rcond=None)

        alpha = 1.0
        accepted = False
        while alpha >= 1e-6:
            cand = beta - alpha * step
            if np.max(np.abs(X_aug @ cand)) > 35.0 and np.max(np.abs(eta)) >= 34.5:
                break
            cand_nll = _neg_loglik(cand)
            if cand_nll <= nll + 1e-12:
                beta = cand
                nll = cand_nll
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    X_ev_aug = np.column_stack([np.ones(n_ev, dtype=np.float64), X_eval[:, keep].astype(np.float64)])
    return expit(np.clip(X_ev_aug @ beta, -35.0, 35.0))


def _fit_pooled_admin_ipcw_multiplier(
    *,
    risk_mask: np.ndarray,
    k_exit: np.ndarray,
    E_eff: np.ndarray,
    X_base: np.ndarray,
    X_tv: np.ndarray,
    k_total: int,
) -> np.ndarray:
    """Fit within-arm stabilized administrative-censoring IPCW multipliers across periods 1..K."""
    n = X_base.shape[0]
    mult = np.ones((n, k_total), dtype=np.float64)
    if k_total <= 1:
        return mult

    active_js: list[int] = []
    p_adm_num = np.ones(k_total - 1, dtype=np.float64)
    elig_mask = np.zeros((n, k_total - 1), dtype=bool)
    remain_mask = np.zeros((n, k_total - 1), dtype=bool)

    for j in range(1, k_total):
        # Eligible for end-of-period j administrative censoring: in clone outcome risk set for j and event-free in j
        event_in_j = (k_exit == j) & (E_eff != 0)
        elig_j = risk_mask[:, j - 1] & (~event_in_j)
        elig_mask[:, j - 1] = elig_j
        n_el = int(np.sum(elig_j))
        if n_el == 0:
            continue
        adm_cens_j = elig_j & (k_exit == j) & (E_eff == 0)
        rem_j = elig_j & (~adm_cens_j)
        remain_mask[:, j - 1] = rem_j
        p_num = float(np.sum(rem_j)) / float(n_el)
        p_adm_num[j - 1] = p_num
        if 0.0 < p_num < 1.0:
            active_js.append(j)

    step_factor = np.ones((n, k_total - 1), dtype=np.float64)
    if active_js:
        subj_parts: list[np.ndarray] = []
        j_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for j in active_js:
            idx_j = np.flatnonzero(elig_mask[:, j - 1])
            subj_parts.append(idx_j)
            j_parts.append(np.full(idx_j.size, j, dtype=np.int64))
            y_parts.append(remain_mask[idx_j, j - 1].astype(np.float64))

        subj_cat = np.concatenate(subj_parts)
        j_cat = np.concatenate(j_parts)
        y_cat = np.concatenate(y_parts)

        feat_blocks: list[np.ndarray] = []
        if len(active_js) > 1:
            ind = np.column_stack([(j_cat == j_val).astype(np.float64) for j_val in active_js[1:]])
            feat_blocks.append(ind)
        feat_blocks.append(X_base[subj_cat])
        if X_tv.shape[2] > 0:
            feat_blocks.append(X_tv[subj_cat, j_cat - 1, :])
        X_adm = np.column_stack(feat_blocks)
        p_adm_hat = _fit_logistic_with_col_guard(X_adm, y_cat)

        offset = 0
        for j in active_js:
            idx_j = np.flatnonzero(elig_mask[:, j - 1])
            cnt = idx_j.size
            p_sub = p_adm_hat[offset : offset + cnt]
            step_factor[idx_j, j - 1] = p_adm_num[j - 1] / np.clip(p_sub, 1e-12, 1.0)
            offset += cnt

    cum_adm = np.cumprod(step_factor, axis=1)
    mult[:, 1:] = cum_adm
    return mult


def _fit_arm_specific_ipcw_and_aj(
    *,
    T_eff: np.ndarray,
    E_eff: np.ndarray,
    has_init: np.ndarray,
    init_day_eff: np.ndarray,
    k_init: np.ndarray,
    k_exit: np.ndarray,
    X_base: np.ndarray,
    X_tv: np.ndarray,
    delta: float,
    grace_days: float,
    m_g: int,
    k_total: int,
    truncate_quantiles: list[float] | None,
    include_admin_ipcw: bool,
) -> dict[str, Any]:
    """Fit strictly separated Strategy 1 and Strategy 0 stabilized IPCW models and discrete AJ CIFs."""
    n = T_eff.shape[0]
    k_grid = np.arange(1, k_total + 1, dtype=np.int64)

    # --- Strategy 1 ('initiate_within_grace') ---
    adherent_g = has_init & (init_day_eff <= grace_days + 1e-12)
    R1 = k_exit[:, None] >= k_grid[None, :]
    R1[:, m_g:] = R1[:, m_g:] & adherent_g[:, None]

    W1 = np.where(R1, 1.0, 0.0)
    S_g = k_exit >= (m_g + 1)
    if np.any(S_g):
        idx_sg = np.flatnonzero(S_g)
        y1_sg = adherent_g[idx_sg].astype(np.float64)
        p1_num = float(np.mean(y1_sg))
        if 0.0 < p1_num < 1.0:
            if X_tv.shape[2] > 0:
                # Fit discrete-time grace-window initiation hazard P(A_{i,k}=1 | A_bar_{k-1}=0, T>(k-1)Delta, k, X_base, X_tv)
                k_g_grid = np.arange(1, m_g + 1, dtype=np.int64)
                E1_g = (k_exit[:, None] >= k_g_grid[None, :]) & (
                    (~has_init[:, None]) | (k_init[:, None] >= k_g_grid[None, :])
                )
                A1_g = E1_g & has_init[:, None] & (k_init[:, None] == k_g_grid[None, :])

                lam_sg = np.zeros((idx_sg.size, m_g), dtype=np.float64)
                active_g_ks: list[int] = []
                for k in range(1, m_g + 1):
                    n_ek = int(np.sum(E1_g[:, k - 1]))
                    if n_ek == 0:
                        continue
                    sum_ak = int(np.sum(A1_g[:, k - 1]))
                    if sum_ak == 0:
                        lam_sg[:, k - 1] = 0.0
                    elif sum_ak == n_ek:
                        lam_sg[:, k - 1] = 1.0
                    else:
                        active_g_ks.append(k)

                if active_g_ks:
                    tr_s: list[np.ndarray] = []
                    tr_k: list[np.ndarray] = []
                    tr_y: list[np.ndarray] = []
                    ev_s: list[np.ndarray] = []
                    ev_k: list[np.ndarray] = []
                    for k in active_g_ks:
                        idx_ek = np.flatnonzero(E1_g[:, k - 1])
                        tr_s.append(idx_ek)
                        tr_k.append(np.full(idx_ek.size, k, dtype=np.int64))
                        tr_y.append(A1_g[idx_ek, k - 1].astype(np.float64))
                        ev_s.append(idx_sg)
                        ev_k.append(np.full(idx_sg.size, k, dtype=np.int64))

                    tr_s_cat = np.concatenate(tr_s)
                    tr_k_cat = np.concatenate(tr_k)
                    tr_y_cat = np.concatenate(tr_y)
                    ev_s_cat = np.concatenate(ev_s)
                    ev_k_cat = np.concatenate(ev_k)

                    tr_feats: list[np.ndarray] = []
                    ev_feats: list[np.ndarray] = []
                    if len(active_g_ks) > 1:
                        tr_feats.append(
                            np.column_stack([(tr_k_cat == kv).astype(np.float64) for kv in active_g_ks[1:]])
                        )
                        ev_feats.append(
                            np.column_stack([(ev_k_cat == kv).astype(np.float64) for kv in active_g_ks[1:]])
                        )
                    tr_feats.append(X_base[tr_s_cat])
                    tr_feats.append(X_tv[tr_s_cat, tr_k_cat - 1, :])
                    ev_feats.append(X_base[ev_s_cat])
                    ev_feats.append(X_tv[ev_s_cat, ev_k_cat - 1, :])

                    pred_lam = _fit_and_predict_logistic_with_col_guard(
                        np.column_stack(tr_feats),
                        tr_y_cat,
                        np.column_stack(ev_feats),
                    )
                    n_sg = idx_sg.size
                    for pos_idx, k in enumerate(active_g_ks):
                        lam_sg[:, k - 1] = pred_lam[pos_idx * n_sg : (pos_idx + 1) * n_sg]

                p1_hat = 1.0 - np.prod(1.0 - np.clip(lam_sg, 0.0, 1.0), axis=1)
            else:
                p1_hat = _fit_logistic_with_col_guard(X_base[idx_sg], y1_sg)

            adh_local = adherent_g[idx_sg]
            adh_idx = idx_sg[adh_local]
            w1_post = p1_num / np.clip(p1_hat[adh_local], 1e-12, 1.0)
            W1[adh_idx, m_g:] = w1_post[:, None] * R1[adh_idx, m_g:].astype(np.float64)
        elif p1_num == 0.0:
            W1[:, m_g:] = 0.0

    if include_admin_ipcw:
        adm_m1 = _fit_pooled_admin_ipcw_multiplier(
            risk_mask=R1,
            k_exit=k_exit,
            E_eff=E_eff,
            X_base=X_base,
            X_tv=X_tv,
            k_total=k_total,
        )
        W1 = W1 * adm_m1

    W1_pre_trunc = W1.copy()
    s1_start_k = 1 if include_admin_ipcw else (m_g + 1)
    if truncate_quantiles is not None:
        for k in range(s1_start_k, k_total + 1):
            mk = R1[:, k - 1]
            if np.any(mk):
                lo, hi = np.quantile(W1[mk, k - 1], truncate_quantiles)
                W1[mk, k - 1] = np.clip(W1[mk, k - 1], float(lo), float(hi))

    active_r1_mask = R1[:, (s1_start_k - 1) :]
    if np.any(active_r1_mask):
        clipped1 = np.abs(W1[:, (s1_start_k - 1) :][active_r1_mask] - W1_pre_trunc[:, (s1_start_k - 1) :][active_r1_mask]) > 1e-12
        trunc_frac_1 = float(np.mean(clipped1))
    else:
        trunc_frac_1 = 0.0

    # --- Strategy 0 ('never_initiate') ---
    E0 = (k_exit[:, None] >= k_grid[None, :]) & ((~has_init[:, None]) | (k_init[:, None] >= k_grid[None, :]))
    R0 = E0 & ((~has_init[:, None]) | (k_init[:, None] > k_grid[None, :]))

    p0_num = np.ones(k_total, dtype=np.float64)
    active_ks: list[int] = []
    for k in range(1, k_total + 1):
        n_ent = int(np.sum(E0[:, k - 1]))
        if n_ent == 0:
            continue
        pn = float(np.sum(R0[:, k - 1])) / float(n_ent)
        p0_num[k - 1] = pn
        if 0.0 < pn < 1.0:
            active_ks.append(k)

    factor0 = np.ones((n, k_total), dtype=np.float64)
    if active_ks:
        subj_parts0: list[np.ndarray] = []
        k_parts0: list[np.ndarray] = []
        y_parts0: list[np.ndarray] = []
        for k in active_ks:
            idx_k = np.flatnonzero(E0[:, k - 1])
            subj_parts0.append(idx_k)
            k_parts0.append(np.full(idx_k.size, k, dtype=np.int64))
            y_parts0.append(R0[idx_k, k - 1].astype(np.float64))

        subj_cat0 = np.concatenate(subj_parts0)
        k_cat0 = np.concatenate(k_parts0)
        y_cat0 = np.concatenate(y_parts0)

        feat0: list[np.ndarray] = []
        if len(active_ks) > 1:
            ind0 = np.column_stack([(k_cat0 == kv).astype(np.float64) for kv in active_ks[1:]])
            feat0.append(ind0)
        feat0.append(X_base[subj_cat0])
        if X_tv.shape[2] > 0:
            feat0.append(X_tv[subj_cat0, k_cat0 - 1, :])
        X_pool0 = np.column_stack(feat0)
        p0_hat_all = _fit_logistic_with_col_guard(X_pool0, y_cat0)

        offset = 0
        for k in active_ks:
            idx_k = np.flatnonzero(E0[:, k - 1])
            cnt = idx_k.size
            p_sub = p0_hat_all[offset : offset + cnt]
            factor0[idx_k, k - 1] = p0_num[k - 1] / np.clip(p_sub, 1e-12, 1.0)
            offset += cnt

    W0 = np.cumprod(factor0, axis=1) * R0.astype(np.float64)
    if include_admin_ipcw:
        adm_m0 = _fit_pooled_admin_ipcw_multiplier(
            risk_mask=R0,
            k_exit=k_exit,
            E_eff=E_eff,
            X_base=X_base,
            X_tv=X_tv,
            k_total=k_total,
        )
        W0 = W0 * adm_m0

    W0_pre_trunc = W0.copy()
    if truncate_quantiles is not None:
        for k in range(1, k_total + 1):
            mk = R0[:, k - 1]
            if np.any(mk):
                lo, hi = np.quantile(W0[mk, k - 1], truncate_quantiles)
                W0[mk, k - 1] = np.clip(W0[mk, k - 1], float(lo), float(hi))

    if np.any(R0):
        trunc_frac_0 = float(np.mean(np.abs(W0[R0] - W0_pre_trunc[R0]) > 1e-12))
    else:
        trunc_frac_0 = 0.0

    # --- Weighted Discrete-Time Aalen-Johansen Recursion ---
    def _discrete_aj(R_mat: np.ndarray, W_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cif1 = np.zeros(k_total + 1, dtype=np.float64)
        cif2 = np.zeros(k_total + 1, dtype=np.float64)
        surv = np.ones(k_total + 1, dtype=np.float64)
        for k in range(1, k_total + 1):
            rk = R_mat[:, k - 1]
            wk = W_mat[:, k - 1]
            yk = float(np.sum(wk[rk])) if np.any(rk) else 0.0
            if yk > 0.0:
                in_k = rk & (k_exit == k)
                d1 = float(np.sum(wk[in_k & (E_eff == 1)]))
                d2 = float(np.sum(wk[in_k & (E_eff == 2)]))
                h1 = d1 / yk
                h2 = d2 / yk
            else:
                h1 = 0.0
                h2 = 0.0
            cif1[k] = cif1[k - 1] + surv[k - 1] * h1
            cif2[k] = cif2[k - 1] + surv[k - 1] * h2
            surv[k] = max(0.0, surv[k - 1] * (1.0 - h1 - h2))
        return cif1, cif2, surv

    cif1_s1, cif2_s1, surv_s1 = _discrete_aj(R1, W1)
    cif1_s0, cif2_s0, surv_s0 = _discrete_aj(R0, W0)

    return {
        "R1": R1,
        "W1": W1,
        "W1_pre_trunc": W1_pre_trunc,
        "trunc_frac_1": trunc_frac_1,
        "R0": R0,
        "W0": W0,
        "W0_pre_trunc": W0_pre_trunc,
        "trunc_frac_0": trunc_frac_0,
        "cif1_s1": cif1_s1,
        "cif2_s1": cif2_s1,
        "surv_s1": surv_s1,
        "cif1_s0": cif1_s0,
        "cif2_s0": cif2_s0,
        "surv_s0": surv_s0,
    }


def _weight_distribution_summary(weights: np.ndarray) -> dict[str, float | None]:
    if weights.size == 0:
        return {
            "count": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    qs = np.quantile(weights, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "count": int(weights.size),
        "min": round(float(np.min(weights)), 8),
        "p01": round(float(qs[0]), 8),
        "p05": round(float(qs[1]), 8),
        "median": round(float(qs[2]), 8),
        "mean": round(float(np.mean(weights)), 8),
        "p95": round(float(qs[3]), 8),
        "p99": round(float(qs[4]), 8),
        "max": round(float(np.max(weights)), 8),
    }


def _compute_post_grace_smd_table(
    *,
    X_base: np.ndarray,
    cov_names: list[str],
    mask0: np.ndarray,
    w0: np.ndarray,
    mask1: np.ndarray,
    w1: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    has_both = bool(np.any(mask0) and np.any(mask1))
    X0 = X_base[mask0]
    X1 = X_base[mask1]
    w0_sub = w0[mask0]
    w1_sub = w1[mask1]
    sum_w0 = float(np.sum(w0_sub)) if X0.shape[0] > 0 else 0.0
    sum_w1 = float(np.sum(w1_sub)) if X1.shape[0] > 0 else 0.0

    for j, cname in enumerate(cov_names):
        if not has_both:
            rows.append(
                {
                    "covariate": cname,
                    "unweighted_mean_strategy_0": None,
                    "unweighted_mean_strategy_1": None,
                    "weighted_mean_strategy_0": None,
                    "weighted_mean_strategy_1": None,
                    "pooled_sd": None,
                    "abs_smd_unweighted": None,
                    "abs_smd_weighted": None,
                }
            )
            continue

        x0 = X0[:, j]
        x1 = X1[:, j]
        mu_u0 = float(np.mean(x0))
        mu_u1 = float(np.mean(x1))
        v0 = float(np.var(x0, ddof=1)) if x0.size > 1 else 0.0
        v1 = float(np.var(x1, ddof=1)) if x1.size > 1 else 0.0
        pooled_sd = math.sqrt(0.5 * (v0 + v1))
        if pooled_sd <= 1e-12:
            full_sd = float(np.std(X_base[:, j], ddof=1)) if X_base.shape[0] > 1 else 0.0
            pooled_sd = full_sd

        mu_w0 = float(np.sum(w0_sub * x0) / sum_w0) if sum_w0 > 0.0 else mu_u0
        mu_w1 = float(np.sum(w1_sub * x1) / sum_w1) if sum_w1 > 0.0 else mu_u1

        diff_u = abs(mu_u1 - mu_u0)
        diff_w = abs(mu_w1 - mu_w0)
        if pooled_sd <= 1e-12:
            smd_u = 0.0 if diff_u <= 1e-12 else float("inf")
            smd_w = 0.0 if diff_w <= 1e-12 else float("inf")
        else:
            smd_u = diff_u / pooled_sd
            smd_w = diff_w / pooled_sd

        rows.append(
            {
                "covariate": cname,
                "unweighted_mean_strategy_0": round(mu_u0, 8),
                "unweighted_mean_strategy_1": round(mu_u1, 8),
                "weighted_mean_strategy_0": round(mu_w0, 8),
                "weighted_mean_strategy_1": round(mu_w1, 8),
                "pooled_sd": round(pooled_sd, 8),
                "abs_smd_unweighted": round(smd_u, 8) if math.isfinite(smd_u) else smd_u,
                "abs_smd_weighted": round(smd_w, 8) if math.isfinite(smd_w) else smd_w,
            }
        )
    return rows


def _evaluate_ccw_diagnostic_gates(
    *,
    protocol: dict,
    fit_res: dict[str, Any],
    X_base: np.ndarray,
    cov_names: list[str],
    X_tv: np.ndarray | None = None,
    tv_cov_names: list[str] | None = None,
) -> dict[str, Any]:
    gp = protocol["grace_period_spec"]
    delta = float(gp["period_length_days"])
    grace_days = float(gp["grace_period_days"])
    m_g = int(round(grace_days / delta))
    gates = protocol["diagnostic_gates"]
    trunc_q = protocol["ipcw_spec"]["truncate_quantiles"]

    R1: np.ndarray = fit_res["R1"]
    W1: np.ndarray = fit_res["W1"]
    W1_pre: np.ndarray = fit_res["W1_pre_trunc"]
    R0: np.ndarray = fit_res["R0"]
    W0: np.ndarray = fit_res["W0"]
    W0_pre: np.ndarray = fit_res["W0_pre_trunc"]
    trunc_frac_1 = float(fit_res["trunc_frac_1"])
    trunc_frac_0 = float(fit_res["trunc_frac_0"])
    max_trunc_frac = max(trunc_frac_1, trunc_frac_0)

    # t = g+ corresponds to period m_g + 1 outcome risk set (index m_g)
    mask1_pg = R1[:, m_g]
    mask0_pg = R0[:, m_g]
    n1_pg = int(np.sum(mask1_pg))
    n0_pg = int(np.sum(mask0_pg))
    ess1_pg = kish_ess(W1[mask1_pg, m_g]) if n1_pg > 0 else 0.0
    ess0_pg = kish_ess(W0[mask0_pg, m_g]) if n0_pg > 0 else 0.0
    ess_ratio1_pg = (ess1_pg / n1_pg) if n1_pg > 0 else 0.0
    ess_ratio0_pg = (ess0_pg / n0_pg) if n0_pg > 0 else 0.0

    X_smd_cols = [X_base]
    smd_cov_names = list(cov_names)
    if X_tv is not None and X_tv.shape[2] > 0 and tv_cov_names:
        for j, tname in enumerate(tv_cov_names):
            X_smd_cols.append(X_tv[:, 0:1, j])
            smd_cov_names.append(f"{tname}__k1")
            if m_g > 1:
                X_smd_cols.append(np.mean(X_tv[:, :m_g, j], axis=1, keepdims=True))
                smd_cov_names.append(f"{tname}__grace_mean")
                X_smd_cols.append(X_tv[:, m_g - 1 : m_g, j])
                smd_cov_names.append(f"{tname}__k{m_g}")
    X_smd = np.column_stack(X_smd_cols)

    smd_rows = _compute_post_grace_smd_table(
        X_base=X_smd,
        cov_names=smd_cov_names,
        mask0=mask0_pg,
        w0=W0[:, m_g],
        mask1=mask1_pg,
        w1=W1[:, m_g],
    )

    valid_w_smds = [r["abs_smd_weighted"] for r in smd_rows if r["abs_smd_weighted"] is not None]
    valid_u_smds = [r["abs_smd_unweighted"] for r in smd_rows if r["abs_smd_unweighted"] is not None]
    max_w_smd = max(valid_w_smds) if valid_w_smds else float("inf")
    max_u_smd = max(valid_u_smds) if valid_u_smds else float("inf")

    w1_pg_vals = W1[:, m_g:][R1[:, m_g:]] if np.any(R1[:, m_g:]) else np.array([], dtype=np.float64)
    w0_all_vals = W0[R0] if np.any(R0) else np.array([], dtype=np.float64)
    w0_pg_vals = W0[:, m_g:][R0[:, m_g:]] if np.any(R0[:, m_g:]) else np.array([], dtype=np.float64)
    w1_pre_pg_vals = W1_pre[:, m_g:][R1[:, m_g:]] if np.any(R1[:, m_g:]) else np.array([], dtype=np.float64)
    w0_pre_all_vals = W0_pre[R0] if np.any(R0) else np.array([], dtype=np.float64)

    all_uncens_w = np.concatenate([W1[R1], W0[R0]])
    overall_max_w = float(np.max(all_uncens_w)) if all_uncens_w.size > 0 else 0.0
    all_pre_trunc_w = np.concatenate([W1_pre[R1], W0_pre[R0]])
    pre_trunc_max_w = float(np.max(all_pre_trunc_w)) if all_pre_trunc_w.size > 0 else 0.0

    horizon_ess: list[dict[str, Any]] = []
    for h_raw in gp["evaluation_horizons_days"]:
        k_h = int(round(float(h_raw) / delta))
        m1_h = R1[:, k_h - 1]
        m0_h = R0[:, k_h - 1]
        n1_h = int(np.sum(m1_h))
        n0_h = int(np.sum(m0_h))
        e1_h = kish_ess(W1[m1_h, k_h - 1]) if n1_h > 0 else 0.0
        e0_h = kish_ess(W0[m0_h, k_h - 1]) if n0_h > 0 else 0.0
        horizon_ess.append(
            {
                "horizon_day": h_raw,
                "period_index": k_h,
                "strategy_1_uncensored_n": n1_h,
                "strategy_1_ess": round(e1_h, 6),
                "strategy_1_ess_ratio": round(e1_h / n1_h, 8) if n1_h > 0 else 0.0,
                "strategy_0_uncensored_n": n0_h,
                "strategy_0_ess": round(e0_h, 6),
                "strategy_0_ess_ratio": round(e0_h / n0_h, 8) if n0_h > 0 else 0.0,
            }
        )

    failures: list[dict[str, Any]] = []
    min_uncens = int(gates["min_uncensored_subjects_per_clone"])
    if n1_pg < min_uncens or n0_pg < min_uncens:
        failures.append(
            {
                "code": "UNCENSORED_BELOW_MIN",
                "detail": (
                    f"Post-grace t=g+ uncensored count (strategy_1={n1_pg}, strategy_0={n0_pg}) "
                    f"is below min_uncensored_subjects_per_clone={min_uncens}"
                ),
                "metric_value": min(n1_pg, n0_pg),
                "threshold": min_uncens,
            }
        )

    max_allowed_w = float(gates["max_stabilized_weight"])
    effective_max_pre_w = float(
        gates.get("max_pre_truncation_weight", max_allowed_w * 2.0 if trunc_q is not None else max_allowed_w)
    )
    effective_max_trunc_frac = float(gates.get("max_truncated_weight_fraction", 0.10))
    quantile_span = float(trunc_q[1] - trunc_q[0]) if trunc_q is not None else None

    if pre_trunc_max_w > effective_max_pre_w:
        failures.append(
            {
                "code": "PRE_TRUNCATION_WEIGHT_EXCEEDED",
                "detail": (
                    f"Pre-truncation maximum stabilized IPCW weight {pre_trunc_max_w:.6f} exceeds "
                    f"max_pre_truncation_weight={effective_max_pre_w:.6f} (longitudinal positivity violation)"
                ),
                "metric_value": round(pre_trunc_max_w, 8),
                "threshold": effective_max_pre_w,
            }
        )

    if trunc_q is not None and (
        (quantile_span is not None and quantile_span < 0.80 - 1e-12) or (max_trunc_frac > effective_max_trunc_frac)
    ):
        failures.append(
            {
                "code": "EXCESSIVE_WEIGHT_TRUNCATION",
                "detail": (
                    f"Excessive IPCW quantile clipping detected: truncate_quantiles={trunc_q} "
                    f"(span={quantile_span:.4f}, min_required_span=0.80), max clone truncated_weight_fraction="
                    f"{max_trunc_frac:.6f} (max_truncated_weight_fraction={effective_max_trunc_frac:.6f})"
                ),
                "metric_value": round(max_trunc_frac, 8),
                "threshold": effective_max_trunc_frac,
            }
        )

    if overall_max_w > max_allowed_w:
        failures.append(
            {
                "code": "MAX_WEIGHT_EXCEEDED",
                "detail": (
                    f"Post-truncation maximum stabilized IPCW weight {overall_max_w:.6f} exceeds "
                    f"diagnostic_gates.max_stabilized_weight={max_allowed_w:.6f}"
                ),
                "metric_value": round(overall_max_w, 8),
                "threshold": max_allowed_w,
            }
        )

    min_ess_gate = float(gates["min_ess_ratio"])
    min_observed_ess_ratio = min(
        ess_ratio1_pg,
        ess_ratio0_pg,
        *(min(row["strategy_1_ess_ratio"], row["strategy_0_ess_ratio"]) for row in horizon_ess),
    )
    if min_observed_ess_ratio < min_ess_gate:
        failures.append(
            {
                "code": "ESS_RATIO_BELOW_MIN",
                "detail": (
                    f"Minimum clone ESS/N_uncensored ratio across t=g+ and evaluation horizons "
                    f"({min_observed_ess_ratio:.6f}) is below diagnostic_gates.min_ess_ratio={min_ess_gate:.6f}"
                ),
                "metric_value": round(min_observed_ess_ratio, 8),
                "threshold": min_ess_gate,
            }
        )

    max_smd_gate = float(gates["max_post_grace_smd"])
    if (not math.isfinite(max_w_smd)) or max_w_smd > max_smd_gate:
        failures.append(
            {
                "code": "POST_GRACE_SMD_EXCEEDS_MAX",
                "detail": (
                    f"Post-grace t=g+ weighted max |SMD| ({max_w_smd:.6f}) exceeds "
                    f"diagnostic_gates.max_post_grace_smd={max_smd_gate:.6f}"
                ),
                "metric_value": round(max_w_smd, 8) if math.isfinite(max_w_smd) else None,
                "threshold": max_smd_gate,
            }
        )

    gate_status = "blocked" if failures else "passed"
    trunc_frac_payload = {
        "strategy_1": round(trunc_frac_1, 8),
        "strategy_0": round(trunc_frac_0, 8),
        "max_across_clones": round(max_trunc_frac, 8),
        "quantile_span": round(quantile_span, 8) if quantile_span is not None else None,
    }
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_version": CCW_PROTOCOL_VERSION,
        "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
        "prohibited_claims": list(PROHIBITED_CLAIMS),
        "methodological_disclaimer": protocol["methodological_disclaimer"],
        "gate_status": gate_status,
        "gate_failures": failures,
        "diagnostic_gates": {
            **dict(gates),
            "effective_max_pre_truncation_weight": effective_max_pre_w,
            "effective_max_truncated_weight_fraction": effective_max_trunc_frac,
        },
        "pre_truncation_max_stabilized_weight": round(pre_trunc_max_w, 8),
        "post_truncation_max_stabilized_weight": round(overall_max_w, 8),
        "max_stabilized_weight": round(overall_max_w, 8),
        "truncated_weight_fraction": trunc_frac_payload,
        "post_grace_entry_summary": {
            "grace_period_days": gp["grace_period_days"],
            "post_grace_period_idx": m_g + 1,
            "strategy_1_uncensored_n": n1_pg,
            "strategy_0_uncensored_n": n0_pg,
            "strategy_1_ess": round(ess1_pg, 6),
            "strategy_1_ess_ratio": round(ess_ratio1_pg, 8),
            "strategy_0_ess": round(ess0_pg, 6),
            "strategy_0_ess_ratio": round(ess_ratio0_pg, 8),
            "max_unweighted_abs_smd": round(max_u_smd, 8) if math.isfinite(max_u_smd) else None,
            "max_weighted_abs_smd": round(max_w_smd, 8) if math.isfinite(max_w_smd) else None,
            "covariate_balance_rows": smd_rows,
        },
        "horizon_ess_summary": horizon_ess,
        "stabilized_weight_summary": {
            "overall_max_weight": round(overall_max_w, 8),
            "pre_truncation_max_stabilized_weight": round(pre_trunc_max_w, 8),
            "post_truncation_max_stabilized_weight": round(overall_max_w, 8),
            "truncated_weight_fraction": trunc_frac_payload,
            "strategy_1_post_grace": _weight_distribution_summary(w1_pg_vals),
            "strategy_0_all_periods": _weight_distribution_summary(w0_all_vals),
            "strategy_0_post_grace": _weight_distribution_summary(w0_pg_vals),
            "pre_truncation_strategy_1_post_grace": _weight_distribution_summary(w1_pre_pg_vals),
            "pre_truncation_strategy_0_all_periods": _weight_distribution_summary(w0_pre_all_vals),
        },
    }


def _compute_naive_post_hoc_comparison(
    *,
    T_eff: np.ndarray,
    E_eff: np.ndarray,
    has_init: np.ndarray,
    init_day_eff: np.ndarray,
    k_exit: np.ndarray,
    X_base: np.ndarray,
    delta: float,
    grace_days: float,
    k_total: int,
    horizons: list[int | float],
    ccw_fit: dict[str, Any],
) -> dict[str, Any]:
    """Compute naive post-hoc static grouping CIFs (without cloning) to quantify immortal-time bias."""
    n = T_eff.shape[0]
    k_grid = np.arange(1, k_total + 1, dtype=np.int64)
    at_risk_obs = k_exit[:, None] >= k_grid[None, :]

    def _static_group_aj(g_ind: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        res_c1 = []
        res_c2 = []
        for arm_val in (0, 1):
            arm_mask = g_ind == arm_val
            cif1 = np.zeros(k_total + 1, dtype=np.float64)
            cif2 = np.zeros(k_total + 1, dtype=np.float64)
            surv = np.ones(k_total + 1, dtype=np.float64)
            for k in range(1, k_total + 1):
                rk = arm_mask & at_risk_obs[:, k - 1]
                yk = float(np.sum(weights[rk])) if np.any(rk) else 0.0
                if yk > 0.0:
                    in_k = rk & (k_exit == k)
                    d1 = float(np.sum(weights[in_k & (E_eff == 1)]))
                    d2 = float(np.sum(weights[in_k & (E_eff == 2)]))
                    h1 = d1 / yk
                    h2 = d2 / yk
                else:
                    h1 = 0.0
                    h2 = 0.0
                cif1[k] = cif1[k - 1] + surv[k - 1] * h1
                cif2[k] = cif2[k - 1] + surv[k - 1] * h2
                surv[k] = max(0.0, surv[k - 1] * (1.0 - h1 - h2))
            res_c1.append(cif1)
            res_c2.append(cif2)
        return res_c1[0], res_c1[1], res_c2[0], res_c2[1]

    # Post-hoc static grouping 1: initiated within grace period <= g vs did not initiate within <= g
    g_grace = (has_init & (init_day_eff <= grace_days + 1e-12)).astype(np.int64)
    p_g = float(np.mean(g_grace))
    if 0.0 < p_g < 1.0:
        ps_g = _fit_logistic_with_col_guard(X_base, g_grace.astype(np.float64))
        w_iptw_g = np.where(
            g_grace == 1,
            p_g / np.clip(ps_g, 1e-6, 1.0),
            (1.0 - p_g) / np.clip(1.0 - ps_g, 1e-6, 1.0),
        )
    else:
        w_iptw_g = np.ones(n, dtype=np.float64)

    w_unit = np.ones(n, dtype=np.float64)
    c1_g0_iptw, c1_g1_iptw, c2_g0_iptw, c2_g1_iptw = _static_group_aj(g_grace, w_iptw_g)
    c1_g0_unw, c1_g1_unw, _, _ = _static_group_aj(g_grace, w_unit)

    # Post-hoc static grouping 2: ever initiated during full follow-up vs never initiated
    g_ever = has_init.astype(np.int64)
    p_ev = float(np.mean(g_ever))
    if 0.0 < p_ev < 1.0:
        ps_ev = _fit_logistic_with_col_guard(X_base, g_ever.astype(np.float64))
        w_iptw_ev = np.where(
            g_ever == 1,
            p_ev / np.clip(ps_ev, 1e-6, 1.0),
            (1.0 - p_ev) / np.clip(1.0 - ps_ev, 1e-6, 1.0),
        )
    else:
        w_iptw_ev = np.ones(n, dtype=np.float64)
    c1_ev0_iptw, c1_ev1_iptw, c2_ev0_iptw, c2_ev1_iptw = _static_group_aj(g_ever, w_iptw_ev)
    c1_ev0_unw, c1_ev1_unw, _, _ = _static_group_aj(g_ever, w_unit)

    horizons_map: dict[str, dict[str, Any]] = {}
    table_rows: list[dict[str, Any]] = []
    for h_raw in horizons:
        k_h = int(round(float(h_raw) / delta))
        ccw_rd = float(ccw_fit["cif1_s1"][k_h] - ccw_fit["cif1_s0"][k_h])
        ccw_rr = (
            float(ccw_fit["cif1_s1"][k_h] / ccw_fit["cif1_s0"][k_h])
            if ccw_fit["cif1_s0"][k_h] > 0.0
            else None
        )

        # Primary naive post-hoc reference (PRD §2.3 Naive Ever-Initiated, baseline IPTW adjusted)
        naive_rd = float(c1_ev1_iptw[k_h] - c1_ev0_iptw[k_h])
        naive_rr = float(c1_ev1_iptw[k_h] / c1_ev0_iptw[k_h]) if c1_ev0_iptw[k_h] > 0.0 else None

        naive_unw_rd = float(c1_ev1_unw[k_h] - c1_ev0_unw[k_h])
        naive_unw_rr = float(c1_ev1_unw[k_h] / c1_ev0_unw[k_h]) if c1_ev0_unw[k_h] > 0.0 else None

        naive_grace_rd = float(c1_g1_iptw[k_h] - c1_g0_iptw[k_h])
        naive_grace_rr = float(c1_g1_iptw[k_h] / c1_g0_iptw[k_h]) if c1_g0_iptw[k_h] > 0.0 else None
        naive_grace_unw_rd = float(c1_g1_unw[k_h] - c1_g0_unw[k_h])

        bias_inflation = naive_rd - ccw_rd
        entry = {
            "horizon_day": h_raw,
            "period_index": k_h,
            "naive_cif_event1_group1": round(float(c1_ev1_iptw[k_h]), 8),
            "naive_cif_event1_group0": round(float(c1_ev0_iptw[k_h]), 8),
            "naive_cif_event2_group1": round(float(c2_ev1_iptw[k_h]), 8),
            "naive_cif_event2_group0": round(float(c2_ev0_iptw[k_h]), 8),
            "naive_rd": round(naive_rd, 8),
            "naive_rr_cif": round(naive_rr, 8) if naive_rr is not None else None,
            "naive_unweighted_rd": round(naive_unw_rd, 8),
            "naive_unweighted_rr_cif": round(naive_unw_rr, 8) if naive_unw_rr is not None else None,
            "naive_within_grace_rd": round(naive_grace_rd, 8),
            "naive_within_grace_rr_cif": round(naive_grace_rr, 8) if naive_grace_rr is not None else None,
            "naive_within_grace_unweighted_rd": round(naive_grace_unw_rd, 8),
            "ccw_rd": round(ccw_rd, 8),
            "ccw_rr_cif": round(ccw_rr, 8) if ccw_rr is not None else None,
            "immortal_time_bias_rd_inflation": round(bias_inflation, 8),
        }
        horizons_map[f"{float(h_raw):g}"] = entry
        table_rows.append(entry)

    return {
        "grouping_definition": (
            "Post-hoc static Naive Ever-Initiated classification from t0=0 without cloning (baseline "
            "stabilized-IPTW adjusted for L0): Group 1 = ever initiated during follow-up vs Group 0 = never "
            "initiated (plus post-hoc initiated-within-grace <= g and unweighted references)."
        ),
        "methodology_warning": (
            "Immortal-time bias warning: classifying delayed initiators into the treated arm at t0=0 "
            "guarantees event-free survival from day 0 until treatment initiation while attributing all "
            "early pre-initiation deaths/events to the uninitiated arm, inducing spurious protective RD < 0 "
            "even under a strict null causal effect and full baseline IPTW adjustment."
        ),
        "curves": {
            "time_days": [round(k * delta, 6) for k in range(0, k_total + 1)],
            "naive_group1_cif1": [round(float(x), 8) for x in c1_ev1_iptw],
            "naive_group0_cif1": [round(float(x), 8) for x in c1_ev0_iptw],
            "naive_group1_cif2": [round(float(x), 8) for x in c2_ev1_iptw],
            "naive_group0_cif2": [round(float(x), 8) for x in c2_ev0_iptw],
            "naive_grace1_cif1": [round(float(x), 8) for x in c1_g1_iptw],
            "naive_grace0_cif1": [round(float(x), 8) for x in c1_g0_iptw],
        },
        "horizons": horizons_map,
        "comparison_table": table_rows,
    }


def _run_subject_cluster_bootstrap(
    *,
    validated: dict[str, Any],
    delta: float,
    grace_days: float,
    m_g: int,
    k_total: int,
    truncate_quantiles: list[float] | None,
    include_admin_ipcw: bool,
    seed: int,
    bootstrap_iterations: int,
    ci_level: float,
) -> dict[str, Any]:
    """Resample subject_id with replacement BEFORE cloning and refit arm-specific IPCW + AJ per replicate."""
    n = len(validated["subject_ids"])
    T_eff = validated["T_eff"]
    E_eff = validated["E_eff"]
    has_init = validated["has_init"]
    init_day_eff = validated["init_day_eff"]
    k_init = validated["k_init"]
    k_exit = validated["k_exit"]
    X_base = validated["X_base"]
    X_tv = validated["X_tv"]

    rng = np.random.default_rng((seed + 7_919) % (MAX_SEED + 1))

    boot_cif1_s1 = np.full((bootstrap_iterations, k_total + 1), np.nan, dtype=np.float64)
    boot_cif2_s1 = np.full((bootstrap_iterations, k_total + 1), np.nan, dtype=np.float64)
    boot_cif1_s0 = np.full((bootstrap_iterations, k_total + 1), np.nan, dtype=np.float64)
    boot_cif2_s0 = np.full((bootstrap_iterations, k_total + 1), np.nan, dtype=np.float64)
    valid_reps = 0

    for b in range(bootstrap_iterations):
        idx = rng.integers(0, n, size=n)
        rep = _fit_arm_specific_ipcw_and_aj(
            T_eff=T_eff[idx],
            E_eff=E_eff[idx],
            has_init=has_init[idx],
            init_day_eff=init_day_eff[idx],
            k_init=k_init[idx],
            k_exit=k_exit[idx],
            X_base=X_base[idx],
            X_tv=X_tv[idx] if X_tv.shape[2] > 0 else X_tv,
            delta=delta,
            grace_days=grace_days,
            m_g=m_g,
            k_total=k_total,
            truncate_quantiles=truncate_quantiles,
            include_admin_ipcw=include_admin_ipcw,
        )
        # Require at least 1 uncensored post-grace subject in each clone if original cohort had post-grace subjects
        if np.any(rep["R1"][:, m_g]) and np.any(rep["R0"][:, m_g]):
            boot_cif1_s1[valid_reps] = rep["cif1_s1"]
            boot_cif2_s1[valid_reps] = rep["cif2_s1"]
            boot_cif1_s0[valid_reps] = rep["cif1_s0"]
            boot_cif2_s0[valid_reps] = rep["cif2_s0"]
            valid_reps += 1

    if valid_reps == 0:
        # Degenerate ultra-small micro-cohort fallback where post-grace subset may be tiny
        for b in range(bootstrap_iterations):
            idx = rng.integers(0, n, size=n)
            rep = _fit_arm_specific_ipcw_and_aj(
                T_eff=T_eff[idx],
                E_eff=E_eff[idx],
                has_init=has_init[idx],
                init_day_eff=init_day_eff[idx],
                k_init=k_init[idx],
                k_exit=k_exit[idx],
                X_base=X_base[idx],
                X_tv=X_tv[idx] if X_tv.shape[2] > 0 else X_tv,
                delta=delta,
                grace_days=grace_days,
                m_g=m_g,
                k_total=k_total,
                truncate_quantiles=truncate_quantiles,
                include_admin_ipcw=include_admin_ipcw,
            )
            boot_cif1_s1[valid_reps] = rep["cif1_s1"]
            boot_cif2_s1[valid_reps] = rep["cif2_s1"]
            boot_cif1_s0[valid_reps] = rep["cif1_s0"]
            boot_cif2_s0[valid_reps] = rep["cif2_s0"]
            valid_reps += 1

    alpha = 1.0 - float(ci_level)
    q_lo = 0.5 * alpha
    q_hi = 1.0 - 0.5 * alpha

    c1_s1 = boot_cif1_s1[:valid_reps]
    c2_s1 = boot_cif2_s1[:valid_reps]
    c1_s0 = boot_cif1_s0[:valid_reps]
    c2_s0 = boot_cif2_s0[:valid_reps]

    return {
        "valid_replicates": valid_reps,
        "q_lo": q_lo,
        "q_hi": q_hi,
        "c1_s1": c1_s1,
        "c2_s1": c2_s1,
        "c1_s0": c1_s0,
        "c2_s0": c2_s0,
    }


def _render_ccw_audit_report_md(
    *,
    protocol: dict,
    attrition: dict,
    diagnostics: dict,
    estimates: dict,
) -> str:
    gp = protocol["grace_period_spec"]
    pg = diagnostics["post_grace_entry_summary"]
    lines = [
        f"# 双侧宽限期治疗克隆–删失–加权（CCW）与不朽时间偏倚校正审计报告 (`{protocol['protocol_id']}`)",
        "",
        f"- **数据与声明锁 (`source_mode`)**: `{SOURCE_MODE_SYNTHETIC_ONLY}`",
        f"- **研究标题**: {protocol['study_title']}",
        f"- **模拟标签**: {protocol['simulation_or_data_label']}",
        f"- **时间网格**: 宽限期 $g = {gp['grace_period_days']}$ 天 | 离散期步长 $\\Delta = {gp['period_length_days']}$ 天 | 最大随访 $H_\\max = {gp['max_followup_days']}$ 天",
        f"- **方差与置信区间方法 (`variance_method`)**: `{estimates['variance_method']}` (Bootstrap 重抽样 `{estimates['bootstrap_config']['bootstrap_iterations']}` 次，有效 `{estimates['bootstrap_config']['valid_replicates']}` 次)",
        f"- **门禁状态 (`gate_status`)**: `{diagnostics['gate_status']}`",
        "",
        "## 1. 双侧克隆与人工删失审计摘要",
        f"- 原始受试者数 $N = {attrition['total_subjects']}$，对称展开双克隆记录 $2N = {attrition['total_clones_created']}$",
        f"- 宽限期 $\\le g$ 内未启动前共享早期事件：共 `{attrition['shared_early_events_within_grace_uninitiated']['total_events']}` 例（目标事件 `{attrition['shared_early_events_within_grace_uninitiated']['primary_events']}`，竞争事件 `{attrition['shared_early_events_within_grace_uninitiated']['competing_events']}`），对称计入两策略克隆",
        f"- **策略 1 克隆 (`initiate_within_grace`)**: 存活跨越 $t=g$ 共 `{attrition['strategy_1_initiate_within_grace']['surviving_past_grace_boundary']}` 人，其中逾期未启动在 $t=g$ 人工删失 `{attrition['strategy_1_initiate_within_grace']['artificially_censored_at_grace_end']}` 人，保留进入 $t=g^+$ `{pg['strategy_1_uncensored_n']}` 人（ESS 比 `{pg['strategy_1_ess_ratio']:.4f}`）",
        f"- **策略 0 克隆 (`never_initiate`)**: 全随访期因启动给药在当期期初人工删失 `{attrition['strategy_0_never_initiate']['total_artificially_censored_on_initiation']}` 人，保留进入 $t=g^+$ `{pg['strategy_0_uncensored_n']}` 人（ESS 比 `{pg['strategy_0_ess_ratio']:.4f}`）",
        f"- **$t = g^+$ 协变量均衡**: 未加权最大 `|SMD|` = `{pg['max_unweighted_abs_smd']:.4f}` → 分臂稳定化 IPCW 加权后最大 `|SMD|` = `{pg['max_weighted_abs_smd']:.4f}`（门禁上限 `{diagnostics['diagnostic_gates']['max_post_grace_smd']}`），全局最大稳定化权重 `{diagnostics['stabilized_weight_summary']['overall_max_weight']:.4f}`",
        "",
        "## 2. 朴素事后分组（不朽时间偏倚） vs 双侧分臂 IPCW-CCW 并列对照表",
        "",
        "| 评估时点 $t^*$ (天) | 朴素事后 `RD(t*)` | 双侧 CCW `RD(t*)` [95% CI] | 不朽时间偏倚膨胀量 (`naive_rd - ccw_rd`) | 朴素事后 `RR_CIF` | 双侧 CCW `RR_CIF(t*)` [95% CI] | `RR_CIF` E-value (点估计 / CI 限) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    naive_by_h = estimates["naive_immortal_time_comparison"]["horizons"]
    for h_est in estimates["horizon_estimates"]:
        h_key = f"{float(h_est['horizon_day']):g}"
        nv = naive_by_h[h_key]
        ev = h_est["evalue_rr_cif_event1"]
        ev_str = (
            f"{ev['point_estimate']:.3f} / {ev['ci_limit_closer_to_null']:.3f}"
            if ev is not None and ev.get("ci_limit_closer_to_null") is not None
            else (f"{ev['point_estimate']:.3f} / —" if ev is not None else "—")
        )
        rr_str = (
            f"{h_est['ccw_rr_cif_event1']:.4f} [{h_est['ccw_rr_cif_event1_ci_lower']:.4f}, {h_est['ccw_rr_cif_event1_ci_upper']:.4f}]"
            if h_est["ccw_rr_cif_event1"] is not None and h_est["ccw_rr_cif_event1_ci_lower"] is not None
            else "—"
        )
        nv_rr_str = f"{nv['naive_rr_cif']:.4f}" if nv["naive_rr_cif"] is not None else "—"
        lines.append(
            f"| {h_key} | {nv['naive_rd']:+.4f} | "
            f"{h_est['ccw_rd_event1']:+.4f} [{h_est['ccw_rd_event1_ci_lower']:+.4f}, {h_est['ccw_rd_event1_ci_upper']:+.4f}] | "
            f"{nv['immortal_time_bias_rd_inflation']:+.4f} | {nv_rr_str} | {rr_str} | {ev_str} |"
        )

    lines.extend(
        [
            "",
            "## 3. 因果识别前提（强制披露）",
            *[f"- {item}" for item in CCW_IDENTIFICATION_ASSUMPTIONS],
            "",
            "## 4. 禁止性声明锁（Prohibited Claims）",
            *[f"- {claim}" for claim in PROHIBITED_CLAIMS],
            "",
            "## 5. 方法学免责声明",
            protocol["methodological_disclaimer"],
            "",
        ]
    )
    return "\n".join(lines)


def _svg_step_path(points: list[tuple[float, float]], x0: float, y0: float, w: float, h: float, max_x: float, max_y: float) -> str:
    if not points:
        return ""
    sx0 = x0 + w * (points[0][0] / max_x)
    sy0 = y0 - h * min(1.0, max(0.0, points[0][1] / max_y))
    cmds = [f"M{sx0:.1f},{sy0:.1f}"]
    for tx, ty in points[1:]:
        px = x0 + w * (tx / max_x)
        py = y0 - h * min(1.0, max(0.0, ty / max_y))
        cmds.append(f"H{px:.1f} V{py:.1f}")
    return " ".join(cmds)


def _render_ccw_preview_report_html(
    *,
    protocol: dict,
    attrition: dict,
    diagnostics: dict,
    estimates: dict,
    ccw_curve_rows: list[dict[str, Any]],
    naive_curve_rows: list[dict[str, Any]],
    dgp_truth: dict[str, Any] | None = None,
) -> str:
    """Render a self-contained offline Chinese HTML report comparing Naive vs Bilateral CCW."""
    esc = lambda v: html.escape(str(v), quote=True)
    gp = protocol["grace_period_spec"]
    grace_days = float(gp["grace_period_days"])
    max_followup = float(gp["max_followup_days"])
    pg = diagnostics["post_grace_entry_summary"]
    gates = diagnostics["diagnostic_gates"]
    w_sum = diagnostics["stabilized_weight_summary"]

    ccw_s0 = [(float(r["time_day"]), float(r["strategy_0_cif_event1"])) for r in ccw_curve_rows]
    ccw_s1 = [(float(r["time_day"]), float(r["strategy_1_cif_event1"])) for r in ccw_curve_rows]
    nv_s0 = [(float(r["time_day"]), float(r["naive_group0_cif_event1"])) for r in naive_curve_rows]
    nv_s1 = [(float(r["time_day"]), float(r["naive_group1_cif_event1"])) for r in naive_curve_rows]

    truth_curves_obj = (dgp_truth or {}).get("curves", {})
    if (
        truth_curves_obj
        and "time_days" in truth_curves_obj
        and "truth_cif_never_initiate" in truth_curves_obj
        and "truth_cif_initiate_within_grace" in truth_curves_obj
    ):
        t_days_tr = [float(t) for t in truth_curves_obj["time_days"]]
        truth_s0: list[tuple[float, float]] | None = list(
            zip(t_days_tr, [float(y) for y in truth_curves_obj["truth_cif_never_initiate"]])
        )
        truth_s1: list[tuple[float, float]] | None = list(
            zip(t_days_tr, [float(y) for y in truth_curves_obj["truth_cif_initiate_within_grace"]])
        )
    else:
        truth_s0 = None
        truth_s1 = None

    peak_cif = max(
        0.10,
        *(y for _, y in ccw_s0),
        *(y for _, y in ccw_s1),
        *(y for _, y in nv_s0),
        *(y for _, y in nv_s1),
        *(y for _, y in (truth_s0 or [])),
        *(y for _, y in (truth_s1 or [])),
    )
    y_max = min(1.0, math.ceil(peak_cif * 11.5) / 10.0)

    def _build_cif_svg(
        title: str,
        subtitle: str,
        pts0: list[tuple[float, float]],
        pts1: list[tuple[float, float]],
        label0: str,
        label1: str,
        highlight_immortal: bool,
    ) -> str:
        x_left, y_bot, plot_w, plot_h = 56.0, 228.0, 440.0, 184.0
        gx = x_left + plot_w * (grace_days / max_followup)
        p0 = _svg_step_path(pts0, x_left, y_bot, plot_w, plot_h, max_followup, y_max)
        p1 = _svg_step_path(pts1, x_left, y_bot, plot_w, plot_h, max_followup, y_max)
        truth_svg_paths = ""
        truth_legend_items = ""
        if truth_s0 is not None and truth_s1 is not None:
            pt0 = _svg_step_path(truth_s0, x_left, y_bot, plot_w, plot_h, max_followup, y_max)
            pt1 = _svg_step_path(truth_s1, x_left, y_bot, plot_w, plot_h, max_followup, y_max)
            truth_svg_paths = (
                f"<path class='cif-truth-step' d='{pt0}' fill='none' stroke='#4338ca' stroke-width='2.0' stroke-dasharray='6,4'/>"
                f"<path class='cif-truth-step' d='{pt1}' fill='none' stroke='#b45309' stroke-width='2.0' stroke-dasharray='6,4'/>"
            )
            truth_legend_items = (
                "<span><i style='border-top:2px dashed #4338ca;background:transparent;height:0'></i>合成真值 (Ground Truth: 策略 0 全程不启动)</span>"
                "<span><i style='border-top:2px dashed #b45309;background:transparent;height:0'></i>合成真值 (Ground Truth: 策略 1 宽限期内启动)</span>"
            )
        y_ticks = []
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            yv = y_max * frac
            py = y_bot - plot_h * frac
            y_ticks.append(
                f"<line x1='{x_left}' y1='{py:.1f}' x2='{x_left + plot_w}' y2='{py:.1f}' stroke='#e2e8f0' stroke-dasharray='3,3'/>"
                f"<text x='{x_left - 8}' y='{py + 4:.1f}' text-anchor='end' font-size='11' fill='#475569'>{yv:.2f}</text>"
            )
        x_ticks = []
        for h_val in sorted({0.0, grace_days, *(float(h) for h in gp["evaluation_horizons_days"]), max_followup}):
            px = x_left + plot_w * (h_val / max_followup)
            x_ticks.append(
                f"<line x1='{px:.1f}' y1='{y_bot}' x2='{px:.1f}' y2='{y_bot + 5}' stroke='#64748b'/>"
                f"<text x='{px:.1f}' y='{y_bot + 18}' text-anchor='middle' font-size='11' fill='#475569'>{h_val:g}d</text>"
            )
        shade_color = "rgba(239,68,68,0.10)" if highlight_immortal else "rgba(14,165,233,0.08)"
        shade_label = "不朽时间错分窗 [0,g]" if highlight_immortal else "宽限期策略允许窗 [0,g]"
        stroke1 = "#dc2626" if highlight_immortal else "#0f766e"
        return (
            f"<div class='chart-card'><h3>{esc(title)}</h3><p class='chart-sub'>{esc(subtitle)}</p>"
            f"<svg viewBox='0 0 520 268' role='img' aria-label='{esc(title)}'>"
            f"<rect x='{x_left}' y='{y_bot - plot_h:.1f}' width='{gx - x_left:.1f}' height='{plot_h:.1f}' fill='{shade_color}'/>"
            + "".join(y_ticks)
            + "".join(x_ticks)
            + f"<line x1='{gx:.1f}' y1='{y_bot - plot_h:.1f}' x2='{gx:.1f}' y2='{y_bot:.1f}' stroke='#94a3b8' stroke-width='1.5' stroke-dasharray='5,4'/>"
            f"<text x='{(x_left + gx) / 2:.1f}' y='{y_bot - plot_h + 15:.1f}' text-anchor='middle' font-size='10.5' fill='#334155'>{esc(shade_label)} (g={grace_days:g}d)</text>"
            f"<path d='M{x_left},{y_bot - plot_h:.1f} V{y_bot:.1f} H{x_left + plot_w:.1f}' fill='none' stroke='#334155' stroke-width='1.4'/>"
            + truth_svg_paths
            + f"<path class='cif-step' d='{p0}' fill='none' stroke='#1d4ed8' stroke-width='2.6'/>"
            f"<path class='cif-step' d='{p1}' fill='none' stroke='{stroke1}' stroke-width='2.6'/>"
            "</svg>"
            "<div class='legend'>"
            f"<span><i style='background:#1d4ed8'></i>{esc(label0)}</span>"
            f"<span><i style='background:{stroke1}'></i>{esc(label1)}</span>"
            + truth_legend_items
            + "</div></div>"
        )

    svg_naive = _build_cif_svg(
        "朴素事后分组假阳性曲线（Naive Ever-Initiated）",
        "错误做法：在 t0=0 按随访期是否曾给药静态分组（含基线 IPTW），延迟给药者在给药前必然存活，人为压低红线制造虚假保护效应",
        nv_s0,
        nv_s1,
        "事后未启动组 CIF_1(t)",
        "事后曾启动组 CIF_1(t) [含不朽时间偏倚]",
        True,
    )
    svg_ccw = _build_cif_svg(
        "双侧宽限期分臂 IPCW-CCW 校正曲线（Bilateral CCW）",
        "正确做法：对称克隆保留 <=g 早期未给药前事件，在违背策略时刻人工删失并分臂独立拟合稳定化 IPCW 恢复无偏边际 CIF_1(t)",
        ccw_s0,
        ccw_s1,
        "策略 0 克隆（全程不启动）CIF_1(t)",
        "策略 1 克隆（宽限期内启动）CIF_1(t)",
        False,
    )

    # SMD Love Plot SVG
    smd_rows = pg["covariate_balance_rows"]
    max_smd_plot = max(
        0.25,
        float(gates["max_post_grace_smd"]) * 1.6,
        *(float(r["abs_smd_unweighted"] or 0.0) for r in smd_rows),
        *(float(r["abs_smd_weighted"] or 0.0) for r in smd_rows),
    )
    smd_h = max(140, 52 + 34 * len(smd_rows))
    smd_x0, smd_w = 165.0, 320.0
    gate_x = smd_x0 + smd_w * (float(gates["max_post_grace_smd"]) / max_smd_plot)
    smd_items = []
    for idx, r in enumerate(smd_rows):
        cy = 38.0 + idx * 32.0
        su = float(r["abs_smd_unweighted"] or 0.0)
        sw = float(r["abs_smd_weighted"] or 0.0)
        xu = smd_x0 + smd_w * min(1.0, su / max_smd_plot)
        xw = smd_x0 + smd_w * min(1.0, sw / max_smd_plot)
        smd_items.append(
            f"<line x1='{smd_x0}' y1='{cy:.1f}' x2='{smd_x0 + smd_w}' y2='{cy:.1f}' stroke='#f1f5f9'/>"
            f"<text x='{smd_x0 - 10}' y='{cy + 4:.1f}' text-anchor='end' font-size='12' fill='#1e293b'>{esc(r['covariate'])}</text>"
            f"<line x1='{xu:.1f}' y1='{cy:.1f}' x2='{xw:.1f}' y2='{cy:.1f}' stroke='#94a3b8' stroke-width='1.5'/>"
            f"<circle cx='{xu:.1f}' cy='{cy:.1f}' r='5' fill='#f59e0b'/>"
            f"<circle cx='{xw:.1f}' cy='{cy:.1f}' r='5' fill='#0f766e'/>"
        )
    smd_svg = (
        f"<svg viewBox='0 0 520 {smd_h}' role='img' aria-label='t=g+ 协变量均衡 SMD 图'>"
        f"<line x1='{smd_x0}' y1='16' x2='{smd_x0}' y2='{smd_h - 24}' stroke='#334155'/>"
        f"<line x1='{gate_x:.1f}' y1='16' x2='{gate_x:.1f}' y2='{smd_h - 24}' stroke='#dc2626' stroke-dasharray='4,3' stroke-width='1.5'/>"
        f"<text x='{gate_x + 4:.1f}' y='26' font-size='10.5' fill='#dc2626'>门禁 {gates['max_post_grace_smd']}</text>"
        + "".join(smd_items)
        + f"<text x='{smd_x0}' y='{smd_h - 6}' font-size='11' fill='#475569'>0.00</text>"
        f"<text x='{smd_x0 + smd_w}' y='{smd_h - 6}' text-anchor='end' font-size='11' fill='#475569'>{max_smd_plot:.2f}</text>"
        "</svg>"
    )

    # Comparison table rows
    truth_horizons = (dgp_truth or {}).get("horizons", {})
    comp_trs = []
    naive_by_h = estimates["naive_immortal_time_comparison"]["horizons"]
    for h_est in estimates["horizon_estimates"]:
        hk = f"{float(h_est['horizon_day']):g}"
        nv = naive_by_h[hk]
        tr_cell = ""
        if truth_horizons and hk in truth_horizons:
            tr_cell = f"<td><code>{truth_horizons[hk]['true_rd_event1']:+.4f}</code></td>"
        ev = h_est["evalue_rr_cif_event1"]
        ev_txt = f"{ev['point_estimate']:.2f} (CI: {ev['ci_limit_closer_to_null']:.2f})" if ev and ev.get("ci_limit_closer_to_null") is not None else "—"
        rr_txt = (
            f"{h_est['ccw_rr_cif_event1']:.3f} [{h_est['ccw_rr_cif_event1_ci_lower']:.3f}, {h_est['ccw_rr_cif_event1_ci_upper']:.3f}]"
            if h_est["ccw_rr_cif_event1"] is not None and h_est["ccw_rr_cif_event1_ci_lower"] is not None
            else "—"
        )
        comp_trs.append(
            "<tr>"
            f"<td><strong>{esc(hk)} 天</strong></td>"
            + tr_cell
            + f"<td class='bias-val'>{nv['naive_rd']:+.4f}</td>"
            f"<td class='ccw-val'><strong>{h_est['ccw_rd_event1']:+.4f}</strong> [{h_est['ccw_rd_event1_ci_lower']:+.4f}, {h_est['ccw_rd_event1_ci_upper']:+.4f}]</td>"
            f"<td>{nv['immortal_time_bias_rd_inflation']:+.4f}</td>"
            f"<td>{rr_txt}</td>"
            f"<td>{h_est['strategy_1_initiate_within_grace']['CIF_2']:.4f} vs {h_est['strategy_0_never_initiate']['CIF_2']:.4f}</td>"
            f"<td>{esc(ev_txt)}</td>"
            "</tr>"
        )
    truth_th = "<th>合成 DGP 真值 RD</th>" if truth_horizons else ""

    s1_att = attrition["strategy_1_initiate_within_grace"]
    s0_att = attrition["strategy_0_never_initiate"]
    shared_ev = attrition["shared_early_events_within_grace_uninitiated"]

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>双侧宽限期 CCW 与不朽时间偏倚校正对比报告 · {esc(protocol['protocol_id'])}</title>
<style>
  :root {{ color-scheme: light; --bg: #f8fafc; --card: #ffffff; --ink: #0f172a; --muted: #475569; --border: #cbd5e1; --teal: #0f766e; --red: #b91c1c; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif; margin: 0; background: var(--bg); color: var(--ink); line-height: 1.55; }}
  .wrap {{ max-width: 1160px; margin: 0 auto; padding: 24px 20px 48px; }}
  .guard-banner {{ background: #fffbeb; border: 1.5px solid #f59e0b; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px; }}
  .guard-banner strong {{ color: #92400e; }}
  .guard-banner ul {{ margin: 6px 0 0 18px; padding: 0; font-size: 13px; color: #78350f; }}
  h1 {{ margin: 0 0 6px; font-size: 24px; }}
  .meta {{ color: var(--muted); font-size: 14px; margin-bottom: 20px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 22px; }}
  .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .kpi .num {{ font-size: 22px; font-weight: 700; color: var(--teal); }}
  .kpi .lbl {{ font-size: 12.5px; color: var(--muted); }}
  .two-col {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 16px; margin-bottom: 22px; }}
  .chart-card, .section-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; margin-bottom: 18px; }}
  .chart-card h3, .section-card h2 {{ margin: 0 0 6px; font-size: 17px; }}
  .chart-sub {{ margin: 0 0 10px; font-size: 12.5px; color: var(--muted); }}
  svg {{ width: 100%; height: auto; display: block; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 12.5px; margin-top: 8px; color: #334155; }}
  .legend i {{ display: inline-block; width: 12px; height: 3px; vertical-align: middle; margin-right: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
  th, td {{ border-bottom: 1px solid #e2e8f0; padding: 9px 10px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  .bias-val {{ color: var(--red); font-weight: 600; }}
  .ccw-val {{ color: var(--teal); }}
  .funnel-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }}
  .funnel-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; font-size: 13.5px; }}
  .view-tabs {{ display: flex; gap: 8px; margin-bottom: 12px; }}
  .view-tabs button {{ border: 1px solid var(--border); background: #fff; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; }}
  .view-tabs button.active {{ background: var(--teal); color: #fff; border-color: var(--teal); }}
</style>
</head>
<body>
<div class="wrap">
  <div class="guard-banner" role="region" aria-label="合成数据与非临床声明锁">
    <strong>合成数据与方法学边界声明锁（source_mode = "{esc(SOURCE_MODE_SYNTHETIC_ONLY)}"）</strong>
    <ul>
      {"".join(f"<li>{esc(c)}</li>" for c in PROHIBITED_CLAIMS)}
    </ul>
  </div>

  <h1>{esc(protocol['study_title'])}</h1>
  <p class="meta">
    协议 ID：<code>{esc(protocol['protocol_id'])}</code> · 版本：<code>{esc(protocol['protocol_version'])}</code> ·
    宽限期 g = <strong>{gp['grace_period_days']}</strong> 天（步长 Δ = {gp['period_length_days']} 天，最大随访 {gp['max_followup_days']} 天） ·
    方差估计：<code>{esc(estimates['variance_method'])}</code>（Bootstrap {estimates['bootstrap_config']['bootstrap_iterations']} 次）
  </p>

  <div class="kpi-grid">
    <div class="kpi"><div class="num">{attrition['total_subjects']} → {attrition['total_clones_created']}</div><div class="lbl">原始合成受试者 → 对称双侧克隆</div></div>
    <div class="kpi"><div class="num">{shared_ev['total_events']} 例</div><div class="lbl">宽限期 ≤g 未启动前双臂共享早期事件</div></div>
    <div class="kpi"><div class="num">{pg['max_unweighted_abs_smd']:.3f} → {pg['max_weighted_abs_smd']:.3f}</div><div class="lbl">t=g+ 最大 |SMD|（未加权 → 分臂 IPCW 加权）</div></div>
    <div class="kpi"><div class="num">{pg['strategy_1_ess_ratio']:.2f} / {pg['strategy_0_ess_ratio']:.2f}</div><div class="lbl">t=g+ 有效样本量比 ESS/N（策略1 / 策略0）</div></div>
  </div>

  <div class="view-tabs">
    <button type="button" class="active" onclick="setMode('both', this)">并排对比视图</button>
    <button type="button" onclick="setMode('naive', this)">仅看朴素事后假阳性曲线</button>
    <button type="button" onclick="setMode('ccw', this)">仅看双侧 CCW 校正曲线</button>
  </div>
  <div class="two-col" id="curve-grid">
    <div id="panel-naive">{svg_naive}</div>
    <div id="panel-ccw">{svg_ccw}</div>
  </div>

  <div class="section-card">
    <h2>评估时点效应与不朽时间偏倚定量对照表</h2>
    <table>
      <thead>
        <tr>
          <th>评估时点 t*</th>
          {truth_th}
          <th>朴素事后 RD(t*)</th>
          <th>双侧 CCW RD(t*) [95% CI]</th>
          <th>不朽时间偏倚膨胀 (naive - ccw)</th>
          <th>双侧 CCW RR_CIF(t*) [95% CI]</th>
          <th>竞争事件 CIF_2（策略1 vs 策略0）</th>
          <th>RR_CIF 总效应 E-value</th>
        </tr>
      </thead>
      <tbody>
        {"".join(comp_trs)}
      </tbody>
    </table>
  </div>

  <div class="two-col">
    <div class="section-card">
      <h2>双克隆与人工删失漏斗（Clone–Censor Attrition Funnel）</h2>
      <div class="funnel-grid">
        <div class="funnel-box">
          <strong>策略 1 克隆（initiate_within_grace，宽限期 ≤ {gp['grace_period_days']}d 启动）</strong>
          <ul>
            <li>t = 0 初始克隆数：<code>{s1_att['initial_clones']}</code></li>
            <li>≤ g 自然结局（人工删失恒为 0）：目标事件 <code>{s1_att['pre_or_at_grace_primary_events']}</code>，竞争事件 <code>{s1_att['pre_or_at_grace_competing_events']}</code>，行政删失 <code>{s1_att['pre_or_at_grace_admin_censored']}</code></li>
            <li>存活跨越 t = g 人数：<code>{s1_att['surviving_past_grace_boundary']}</code></li>
            <li>t = g 届满逾期未启动人工删失：<code>{s1_att['artificially_censored_at_grace_end']}</code></li>
            <li>进入 t = g+ 未删失风险集：<code>{s1_att['uncensored_entering_post_grace']}</code>（ESS = {pg['strategy_1_ess']:.1f}）</li>
          </ul>
        </div>
        <div class="funnel-box">
          <strong>策略 0 克隆（never_initiate，全程不启动）</strong>
          <ul>
            <li>t = 0 初始克隆数：<code>{s0_att['initial_clones']}</code></li>
            <li>宽限期 ≤ g 内因给药在当期期初人工删失：<code>{s0_att['artificially_censored_within_grace']}</code></li>
            <li>进入 t = g+ 未删失风险集：<code>{s0_att['uncensored_entering_post_grace']}</code>（ESS = {pg['strategy_0_ess']:.1f}）</li>
            <li>宽限期后 &gt; g 因延迟给药人工删失：<code>{s0_att['artificially_censored_post_grace']}</code></li>
            <li>全程未人工删失结局：目标事件 <code>{s0_att['uncensored_primary_events']}</code>，竞争事件 <code>{s0_att['uncensored_competing_events']}</code></li>
          </ul>
        </div>
      </div>
      <p class="chart-sub" style="margin-top:10px">全局最大稳定化 IPCW 权重：<code>{w_sum['overall_max_weight']:.4f}</code>（门禁上限 <code>{gates['max_stabilized_weight']}</code>）</p>
    </div>

    <div class="section-card">
      <h2>t = g+ 基线协变量均衡诊断（未加权橙点 vs 分臂 IPCW 绿点）</h2>
      {smd_svg}
      <div class="legend">
        <span><i style="background:#f59e0b;height:8px;width:8px;border-radius:50%"></i>未加权 |SMD|</span>
        <span><i style="background:#0f766e;height:8px;width:8px;border-radius:50%"></i>分臂稳定化 IPCW 加权后 |SMD|</span>
      </div>
    </div>
  </div>

  <div class="section-card">
    <h2>显式因果识别前提与方法学说明</h2>
    <ul>
      {"".join(f"<li>{esc(a)}</li>" for a in CCW_IDENTIFICATION_ASSUMPTIONS)}
    </ul>
    <p class="chart-sub">{esc(protocol['methodological_disclaimer'])}</p>
  </div>
</div>
<script>
function setMode(mode, btn) {{
  document.querySelectorAll('.view-tabs button').forEach(function(b) {{ b.classList.remove('active'); }});
  btn.classList.add('active');
  document.getElementById('panel-naive').style.display = (mode === 'ccw') ? 'none' : 'block';
  document.getElementById('panel-ccw').style.display = (mode === 'naive') ? 'none' : 'block';
}}
</script>
</body>
</html>
"""


def run_ccw_synthetic_pipeline(
    protocol_path: Path | str | dict,
    *,
    seed: int,
    n_cases: int,
    out_dir: Path,
    bootstrap_iterations: int = 200,
    dgp_params: Path | str | dict | None = None,
    ci_level: float = 0.95,
) -> tuple[dict[str, Any], int]:
    """Run end-to-end synthetic CCW benchmark (validate-ccw -> synth-ccw -> run_grace_period_ccw) with HTML report."""
    check_destination(out_dir)
    gen = generate_synthetic_ccw_cohort(
        protocol_path,
        seed=seed,
        n_cases=n_cases,
        dgp_params=dgp_params,
        out_dir=None,
    )
    norm_proto = gen["protocol"]
    dgp_truth = gen["dgp_truth"]
    cohort_df: pd.DataFrame = gen["cohort_df"]
    pp_df: pd.DataFrame | None = gen["person_period_df"]

    # Format cohort.csv bytes deterministically matching synth-ccw
    base_covs = norm_proto["covariates"]["baseline_covariates"]
    base_names = [c["name"] for c in base_covs]
    cohort_fields = [
        "subject_id",
        "simulation_or_data_label",
        "initiation_day",
        "duration_day",
        "event_code",
        *base_names,
    ]
    csv_lines = [",".join(cohort_fields)]
    for idx in range(len(cohort_df)):
        init_v = cohort_df["initiation_day"].iloc[idx]
        init_s = "" if pd.isna(init_v) else f"{float(init_v):.6f}"
        dur_s = f"{float(cohort_df['duration_day'].iloc[idx]):.6f}"
        ev_s = str(int(cohort_df["event_code"].iloc[idx]))
        cov_strs = []
        for j, c in enumerate(base_covs):
            cv = cohort_df[c["name"]].iloc[idx]
            cov_strs.append(str(int(cv)) if c["type"] == "binary" else f"{float(cv):.6f}")
        # Quote simulation_or_data_label safely via csv module
        row_vals = [
            str(cohort_df["subject_id"].iloc[idx]),
            str(cohort_df["simulation_or_data_label"].iloc[idx]),
            init_s,
            dur_s,
            ev_s,
            *cov_strs,
        ]
        csv_lines.append(",".join(f'"{v}"' if ("," in v or '"' in v) else v for v in row_vals))
    cohort_bytes = ("\n".join(csv_lines) + "\n").encode("utf-8")

    purpose_bytes = (
        "SYNTHETIC-ONLY BENCHMARK COHORT & CCW AUDIT BUNDLE (tte-ccw-v1)\n"
        f"Protocol ID: {norm_proto['protocol_id']}\n"
        f"Label: {norm_proto['simulation_or_data_label']}\n"
        f"Seed: {seed} | Cases: {n_cases}\n\n"
        + "\n".join(PROHIBITED_CLAIMS)
        + "\n"
    ).encode("utf-8")

    extra_files: list[tuple[str, bytes]] = [
        ("PURPOSE.txt", purpose_bytes),
        ("cohort.csv", cohort_bytes),
        ("dgp_truth.json", json_bytes(dgp_truth)),
    ]
    if pp_df is not None:
        pp_buf = pp_df.to_csv(index=False, lineterminator="\n").encode("utf-8")
        extra_files.append(("person_period_covariates.csv", pp_buf))

    def _html_with_truth(**kwargs: Any) -> str:
        return _render_ccw_preview_report_html(**kwargs, dgp_truth=dgp_truth)

    res, exit_code = run_grace_period_ccw(
        protocol_path,
        cohort_df,
        seed=seed,
        bootstrap_iterations=bootstrap_iterations,
        ci_level=ci_level,
        person_period_covariates=pp_df,
        out_dir=out_dir,
        extra_stage_files=extra_files,
        render_html_fn=_html_with_truth,
    )
    if exit_code == 0:
        res["status"] = "ccw_synthetic_pipeline_completed"
    else:
        res["status"] = "ccw_synthetic_pipeline_blocked"
    return res, exit_code


def run_grace_period_ccw(
    protocol: Path | str | dict,
    cohort: pd.DataFrame | Path | str,
    *,
    seed: int,
    bootstrap_iterations: int = 200,
    ci_level: float = 0.95,
    person_period_covariates: pd.DataFrame | Path | str | None = None,
    out_dir: Path | None = None,
    extra_stage_files: list[tuple[str, bytes]] | None = None,
    render_html_fn: Any = _render_ccw_preview_report_html,
) -> tuple[dict[str, Any], int]:
    """Execute bilateral grace-period CCW, fail-closed IPCW diagnostic gating, and subject-cluster bootstrap.

    Returns
    -------
    (result_dict, exit_code)
      - exit_code == 0 when gate_status == "passed"
      - exit_code == 3 when gate_status == "blocked" (publishes diagnostics/attrition/manifest only;
        NEVER produces ccw_estimates.json)
      - Raises ContractError (exit_code == 2) on any protocol/cohort contract violation.
    """
    if out_dir is not None:
        check_destination(out_dir)

    if not is_strict_int(seed) or not (0 <= seed <= MAX_SEED):
        raise ContractError("INVALID_SEED", f"seed must be an integer in 0..{MAX_SEED}")
    if not is_strict_int(bootstrap_iterations) or not (1 <= bootstrap_iterations <= MAX_BOOTSTRAP_ITERATIONS):
        raise ContractError(
            "INVALID_BOOTSTRAP_ITERATIONS",
            f"bootstrap_iterations must be an integer in 1..{MAX_BOOTSTRAP_ITERATIONS}",
        )
    if not is_strict_number(ci_level) or not (0.0 < float(ci_level) < 1.0):
        raise ContractError("INVALID_CI_LEVEL", "ci_level must lie strictly in (0.0, 1.0)")

    raw_proto, norm_proto = _resolve_ccw_protocol_input(protocol)
    validated = validate_ccw_cohort_df(cohort, norm_proto, person_period_covariates=person_period_covariates)
    _, attrition = clone_and_censor(validated["df"], norm_proto, person_period_covariates=person_period_covariates)

    gp = norm_proto["grace_period_spec"]
    delta = float(gp["period_length_days"])
    grace_days = float(gp["grace_period_days"])
    max_followup = float(gp["max_followup_days"])
    m_g = int(round(grace_days / delta))
    k_total = int(round(max_followup / delta))
    ipcw_spec = norm_proto["ipcw_spec"]

    fit_res = _fit_arm_specific_ipcw_and_aj(
        T_eff=validated["T_eff"],
        E_eff=validated["E_eff"],
        has_init=validated["has_init"],
        init_day_eff=validated["init_day_eff"],
        k_init=validated["k_init"],
        k_exit=validated["k_exit"],
        X_base=validated["X_base"],
        X_tv=validated["X_tv"],
        delta=delta,
        grace_days=grace_days,
        m_g=m_g,
        k_total=k_total,
        truncate_quantiles=ipcw_spec["truncate_quantiles"],
        include_admin_ipcw=bool(ipcw_spec["include_admin_ipcw"]),
    )

    diagnostics = _evaluate_ccw_diagnostic_gates(
        protocol=norm_proto,
        fit_res=fit_res,
        X_base=validated["X_base"],
        cov_names=validated["base_cov_names"],
        X_tv=validated["X_tv"],
        tv_cov_names=validated["tv_cov_names"],
    )

    balance_fieldnames = [
        "covariate",
        "unweighted_mean_strategy_0",
        "unweighted_mean_strategy_1",
        "weighted_mean_strategy_0",
        "weighted_mean_strategy_1",
        "pooled_sd",
        "abs_smd_unweighted",
        "abs_smd_weighted",
    ]
    balance_rows = diagnostics["post_grace_entry_summary"]["covariate_balance_rows"]

    # Fail-closed exit code 3: publish ONLY diagnostics & attrition, NEVER ccw_estimates.json!
    if diagnostics["gate_status"] == "blocked":
        if out_dir is None:
            return (
                {
                    "status": "ccw_diagnostic_gate_blocked",
                    "gate_status": "blocked",
                    "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
                    "protocol_id": norm_proto["protocol_id"],
                    "gate_failures": diagnostics["gate_failures"],
                    "diagnostics": diagnostics,
                    "attrition": attrition,
                },
                3,
            )

        commit, code_dirty = code_provenance()
        deps = dependency_versions()
        norm_bytes = json_bytes(norm_proto)
        diag_bytes = json_bytes(diagnostics)
        attr_bytes = json_bytes(attrition)

        def _populate_blocked(stage: Path) -> None:
            (stage / "normalized_protocol.json").write_bytes(norm_bytes)
            (stage / "ccw_diagnostics.json").write_bytes(diag_bytes)
            (stage / "ccw_attrition.json").write_bytes(attr_bytes)
            _write_csv_rows(stage / "balance_post_grace.csv", balance_fieldnames, balance_rows)
            written = ["balance_post_grace.csv", "ccw_attrition.json", "ccw_diagnostics.json", "normalized_protocol.json"]
            if extra_stage_files:
                for rel_p, content_b in extra_stage_files:
                    # Never write estimates or curves when blocked
                    if "estimate" in rel_p or rel_p.startswith("curves/"):
                        continue
                    dest = stage / rel_p
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(content_b)
                    if rel_p not in written:
                        written.append(rel_p)
            written.sort()
            manifest = {
                "status": "ccw_diagnostic_gate_blocked",
                "gate_status": "blocked",
                "activity": "tte_grace_period_ccw_estimation",
                "protocol_id": norm_proto["protocol_id"],
                "protocol_version": CCW_PROTOCOL_VERSION,
                "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
                "prohibited_claims": list(PROHIBITED_CLAIMS),
                "simulation_or_data_label": norm_proto["simulation_or_data_label"],
                "methodological_disclaimer": norm_proto["methodological_disclaimer"],
                "seed": seed,
                "case_count": len(validated["subject_ids"]),
                "bootstrap_iterations": bootstrap_iterations,
                "protocol_sha256": sha256_bytes(raw_proto),
                "normalized_protocol_sha256": sha256_bytes(norm_bytes),
                "code_commit": commit,
                "code_dirty": code_dirty,
                "dependency_versions": deps,
                "output_paths": written,
                "output_sha256": {p: sha256_file(stage / p) for p in written},
                "run_at_utc": utc_now_iso(),
            }
            (stage / "run_manifest.json").write_bytes(json_bytes(manifest))

        atomic_publish(out_dir, _populate_blocked)
        return (
            {
                "status": "ccw_diagnostic_gate_blocked",
                "gate_status": "blocked",
                "protocol_id": norm_proto["protocol_id"],
                "protocol_version": CCW_PROTOCOL_VERSION,
                "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
                "simulation_or_data_label": norm_proto["simulation_or_data_label"],
                "seed": seed,
                "case_count": len(validated["subject_ids"]),
                "gate_failures": diagnostics["gate_failures"],
                "output_path": str(out_dir),
            },
            3,
        )

    # Gate passed -> compute naive immortal-time comparison + subject-cluster bootstrap
    naive_comp = _compute_naive_post_hoc_comparison(
        T_eff=validated["T_eff"],
        E_eff=validated["E_eff"],
        has_init=validated["has_init"],
        init_day_eff=validated["init_day_eff"],
        k_exit=validated["k_exit"],
        X_base=validated["X_base"],
        delta=delta,
        grace_days=grace_days,
        k_total=k_total,
        horizons=gp["evaluation_horizons_days"],
        ccw_fit=fit_res,
    )

    boot = _run_subject_cluster_bootstrap(
        validated=validated,
        delta=delta,
        grace_days=grace_days,
        m_g=m_g,
        k_total=k_total,
        truncate_quantiles=ipcw_spec["truncate_quantiles"],
        include_admin_ipcw=bool(ipcw_spec["include_admin_ipcw"]),
        seed=seed,
        bootstrap_iterations=bootstrap_iterations,
        ci_level=float(ci_level),
    )

    q_lo, q_hi = boot["q_lo"], boot["q_hi"]
    b_c1_s1 = boot["c1_s1"]
    b_c2_s1 = boot["c2_s1"]
    b_c1_s0 = boot["c1_s0"]
    b_c2_s0 = boot["c2_s0"]

    horizon_estimates: list[dict[str, Any]] = []
    horizons_est_map: dict[str, dict[str, Any]] = {}
    for h_raw in gp["evaluation_horizons_days"]:
        k_h = int(round(float(h_raw) / delta))
        h_key = f"{float(h_raw):g}"

        c1_1 = float(fit_res["cif1_s1"][k_h])
        c1_0 = float(fit_res["cif1_s0"][k_h])
        c2_1 = float(fit_res["cif2_s1"][k_h])
        c2_0 = float(fit_res["cif2_s0"][k_h])

        rd1 = c1_1 - c1_0
        rd2 = c2_1 - c2_0
        rr1 = (c1_1 / c1_0) if c1_0 > 1e-12 else None
        rr2 = (c2_1 / c2_0) if c2_0 > 1e-12 else None

        c1_1_lo, c1_1_hi = np.quantile(b_c1_s1[:, k_h], [q_lo, q_hi])
        c1_0_lo, c1_0_hi = np.quantile(b_c1_s0[:, k_h], [q_lo, q_hi])
        c2_1_lo, c2_1_hi = np.quantile(b_c2_s1[:, k_h], [q_lo, q_hi])
        c2_0_lo, c2_0_hi = np.quantile(b_c2_s0[:, k_h], [q_lo, q_hi])

        rd1_draws = b_c1_s1[:, k_h] - b_c1_s0[:, k_h]
        rd1_lo, rd1_hi = np.quantile(rd1_draws, [q_lo, q_hi])
        rd2_draws = b_c2_s1[:, k_h] - b_c2_s0[:, k_h]
        rd2_lo, rd2_hi = np.quantile(rd2_draws, [q_lo, q_hi])

        pos_rr1 = (b_c1_s0[:, k_h] > 1e-12) & (b_c1_s1[:, k_h] > 1e-12)
        if rr1 is not None and rr1 > 0.0 and int(np.sum(pos_rr1)) >= max(5, int(0.5 * boot["valid_replicates"])):
            rr1_draws = b_c1_s1[pos_rr1, k_h] / b_c1_s0[pos_rr1, k_h]
            rr1_lo_f = float(np.quantile(rr1_draws, q_lo))
            rr1_hi_f = float(np.quantile(rr1_draws, q_hi))
            # Ensure percentile CI encloses point estimate for E-value consistency if near-boundary
            e_lo = min(rr1_lo_f, rr1)
            e_hi = max(rr1_hi_f, rr1)
            evalue_res = compute_evalue(
                rr1,
                ci_lower=e_lo,
                ci_upper=e_hi,
                measure_type="RR_CIF",
                has_competing_risks=True,
            )
        elif rr1 is not None and rr1 > 0.0:
            rr1_lo_f = None
            rr1_hi_f = None
            evalue_res = compute_evalue(
                rr1,
                measure_type="RR_CIF",
                has_competing_risks=True,
            )
        else:
            rr1_lo_f = None
            rr1_hi_f = None
            evalue_res = None

        pos_rr2 = (b_c2_s0[:, k_h] > 1e-12) & (b_c2_s1[:, k_h] > 1e-12)
        if rr2 is not None and rr2 > 0.0 and int(np.sum(pos_rr2)) >= max(5, int(0.5 * boot["valid_replicates"])):
            rr2_draws = b_c2_s1[pos_rr2, k_h] / b_c2_s0[pos_rr2, k_h]
            rr2_lo_f = float(np.quantile(rr2_draws, q_lo))
            rr2_hi_f = float(np.quantile(rr2_draws, q_hi))
        else:
            rr2_lo_f = None
            rr2_hi_f = None

        # Enrich naive comparison row with CCW CIs
        naive_comp["horizons"][h_key]["ccw_rd_ci_lower"] = round(float(rd1_lo), 8)
        naive_comp["horizons"][h_key]["ccw_rd_ci_upper"] = round(float(rd1_hi), 8)
        naive_comp["horizons"][h_key]["ccw_rr_cif_ci_lower"] = round(rr1_lo_f, 8) if rr1_lo_f is not None else None
        naive_comp["horizons"][h_key]["ccw_rr_cif_ci_upper"] = round(rr1_hi_f, 8) if rr1_hi_f is not None else None

        h_entry = {
            "horizon_day": h_raw,
            "period_index": k_h,
            "strategy_1_initiate_within_grace": {
                "CIF_1": round(c1_1, 8),
                "CIF_1_ci_lower": round(float(c1_1_lo), 8),
                "CIF_1_ci_upper": round(float(c1_1_hi), 8),
                "CIF_2": round(c2_1, 8),
                "CIF_2_ci_lower": round(float(c2_1_lo), 8),
                "CIF_2_ci_upper": round(float(c2_1_hi), 8),
                "survival": round(float(fit_res["surv_s1"][k_h]), 8),
            },
            "strategy_0_never_initiate": {
                "CIF_1": round(c1_0, 8),
                "CIF_1_ci_lower": round(float(c1_0_lo), 8),
                "CIF_1_ci_upper": round(float(c1_0_hi), 8),
                "CIF_2": round(c2_0, 8),
                "CIF_2_ci_lower": round(float(c2_0_lo), 8),
                "CIF_2_ci_upper": round(float(c2_0_hi), 8),
                "survival": round(float(fit_res["surv_s0"][k_h]), 8),
            },
            "ccw_rd_event1": round(rd1, 8),
            "ccw_rd_event1_ci_lower": round(float(rd1_lo), 8),
            "ccw_rd_event1_ci_upper": round(float(rd1_hi), 8),
            "ccw_rr_cif_event1": round(rr1, 8) if rr1 is not None else None,
            "ccw_rr_cif_event1_ci_lower": round(rr1_lo_f, 8) if rr1_lo_f is not None else None,
            "ccw_rr_cif_event1_ci_upper": round(rr1_hi_f, 8) if rr1_hi_f is not None else None,
            "ccw_rd_event2": round(rd2, 8),
            "ccw_rd_event2_ci_lower": round(float(rd2_lo), 8),
            "ccw_rd_event2_ci_upper": round(float(rd2_hi), 8),
            "ccw_rr_cif_event2": round(rr2, 8) if rr2 is not None else None,
            "ccw_rr_cif_event2_ci_lower": round(rr2_lo_f, 8) if rr2_lo_f is not None else None,
            "ccw_rr_cif_event2_ci_upper": round(rr2_hi_f, 8) if rr2_hi_f is not None else None,
            "evalue_rr_cif_event1": evalue_res,
        }
        horizon_estimates.append(h_entry)
        horizons_est_map[h_key] = h_entry

    ccw_curve_rows: list[dict[str, Any]] = []
    naive_curve_rows: list[dict[str, Any]] = []
    for k in range(0, k_total + 1):
        t_day = round(k * delta, 6)
        ccw_curve_rows.append(
            {
                "period_idx": k,
                "time_day": t_day,
                "strategy_0_cif_event1": f"{fit_res['cif1_s0'][k]:.8f}",
                "strategy_1_cif_event1": f"{fit_res['cif1_s1'][k]:.8f}",
                "strategy_0_cif_event2": f"{fit_res['cif2_s0'][k]:.8f}",
                "strategy_1_cif_event2": f"{fit_res['cif2_s1'][k]:.8f}",
                "strategy_0_survival": f"{fit_res['surv_s0'][k]:.8f}",
                "strategy_1_survival": f"{fit_res['surv_s1'][k]:.8f}",
            }
        )
        naive_curve_rows.append(
            {
                "period_idx": k,
                "time_day": t_day,
                "naive_group0_cif_event1": f"{naive_comp['curves']['naive_group0_cif1'][k]:.8f}",
                "naive_group1_cif_event1": f"{naive_comp['curves']['naive_group1_cif1'][k]:.8f}",
                "naive_group0_cif_event2": f"{naive_comp['curves']['naive_group0_cif2'][k]:.8f}",
                "naive_group1_cif_event2": f"{naive_comp['curves']['naive_group1_cif2'][k]:.8f}",
            }
        )

    estimates_payload = {
        "protocol_id": norm_proto["protocol_id"],
        "protocol_version": CCW_PROTOCOL_VERSION,
        "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
        "prohibited_claims": list(PROHIBITED_CLAIMS),
        "identification_assumptions": list(CCW_IDENTIFICATION_ASSUMPTIONS),
        "methodological_disclaimer": norm_proto["methodological_disclaimer"],
        "simulation_or_data_label": norm_proto["simulation_or_data_label"],
        "grace_period_spec": gp,
        "ipcw_spec": ipcw_spec,
        "variance_method": "subject_cluster_bootstrap_with_ipcw_refit",
        "bootstrap_config": {
            "seed": seed,
            "bootstrap_iterations": bootstrap_iterations,
            "valid_replicates": boot["valid_replicates"],
            "ci_level": float(ci_level),
            "resampling_unit": "subject_id_pre_cloning",
            "ipcw_reestimated_in_bootstrap": True,
        },
        "horizons": horizons_est_map,
        "horizon_estimates": horizon_estimates,
        "naive_immortal_time_comparison": naive_comp,
    }

    if out_dir is None:
        return (
            {
                "status": "ccw_estimated",
                "gate_status": "passed",
                "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
                "protocol_id": norm_proto["protocol_id"],
                "attrition": attrition,
                "diagnostics": diagnostics,
                "estimates": estimates_payload,
                "curves": {
                    "ccw": ccw_curve_rows,
                    "naive": naive_curve_rows,
                },
            },
            0,
        )

    commit, code_dirty = code_provenance()
    deps = dependency_versions()
    norm_bytes = json_bytes(norm_proto)
    proto_rep_bytes = _render_ccw_protocol_report_md(norm_proto).encode("utf-8")
    attr_bytes = json_bytes(attrition)
    diag_bytes = json_bytes(diagnostics)
    est_bytes = json_bytes(estimates_payload)
    audit_md_bytes = _render_ccw_audit_report_md(
        protocol=norm_proto,
        attrition=attrition,
        diagnostics=diagnostics,
        estimates=estimates_payload,
    ).encode("utf-8")

    html_bytes: bytes | None = None
    if render_html_fn is not None:
        html_bytes = render_html_fn(
            protocol=norm_proto,
            attrition=attrition,
            diagnostics=diagnostics,
            estimates=estimates_payload,
            ccw_curve_rows=ccw_curve_rows,
            naive_curve_rows=naive_curve_rows,
        ).encode("utf-8")

    ccw_curve_fields = [
        "period_idx",
        "time_day",
        "strategy_0_cif_event1",
        "strategy_1_cif_event1",
        "strategy_0_cif_event2",
        "strategy_1_cif_event2",
        "strategy_0_survival",
        "strategy_1_survival",
    ]
    naive_curve_fields = [
        "period_idx",
        "time_day",
        "naive_group0_cif_event1",
        "naive_group1_cif_event1",
        "naive_group0_cif_event2",
        "naive_group1_cif_event2",
    ]

    def _populate_passed(stage: Path) -> None:
        (stage / "curves").mkdir(parents=True, exist_ok=True)
        (stage / "normalized_protocol.json").write_bytes(norm_bytes)
        (stage / "protocol_check_report.md").write_bytes(proto_rep_bytes)
        (stage / "ccw_attrition.json").write_bytes(attr_bytes)
        (stage / "ccw_diagnostics.json").write_bytes(diag_bytes)
        _write_csv_rows(stage / "balance_post_grace.csv", balance_fieldnames, balance_rows)
        (stage / "ccw_estimates.json").write_bytes(est_bytes)
        _write_csv_rows(stage / "curves" / "ccw_cif_curves.csv", ccw_curve_fields, ccw_curve_rows)
        _write_csv_rows(stage / "curves" / "naive_cif_curves.csv", naive_curve_fields, naive_curve_rows)
        (stage / "audit_report.md").write_bytes(audit_md_bytes)
        written = [
            "audit_report.md",
            "balance_post_grace.csv",
            "ccw_attrition.json",
            "ccw_diagnostics.json",
            "ccw_estimates.json",
            "curves/ccw_cif_curves.csv",
            "curves/naive_cif_curves.csv",
            "normalized_protocol.json",
            "protocol_check_report.md",
        ]
        if html_bytes is not None:
            (stage / "ccw_preview_report.html").write_bytes(html_bytes)
            written.append("ccw_preview_report.html")
        if extra_stage_files:
            for rel_p, content_b in extra_stage_files:
                dest = stage / rel_p
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content_b)
                if rel_p not in written:
                    written.append(rel_p)
        written.sort()
        manifest = {
            "status": "ccw_estimated",
            "gate_status": "passed",
            "activity": "tte_grace_period_ccw_estimation",
            "protocol_id": norm_proto["protocol_id"],
            "protocol_version": CCW_PROTOCOL_VERSION,
            "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
            "prohibited_claims": list(PROHIBITED_CLAIMS),
            "simulation_or_data_label": norm_proto["simulation_or_data_label"],
            "methodological_disclaimer": norm_proto["methodological_disclaimer"],
            "variance_method": "subject_cluster_bootstrap_with_ipcw_refit",
            "seed": seed,
            "case_count": len(validated["subject_ids"]),
            "bootstrap_iterations": bootstrap_iterations,
            "ci_level": float(ci_level),
            "protocol_sha256": sha256_bytes(raw_proto),
            "normalized_protocol_sha256": sha256_bytes(norm_bytes),
            "code_commit": commit,
            "code_dirty": code_dirty,
            "dependency_versions": deps,
            "output_paths": written,
            "output_sha256": {p: sha256_file(stage / p) for p in written},
            "run_at_utc": utc_now_iso(),
        }
        (stage / "run_manifest.json").write_bytes(json_bytes(manifest))

    atomic_publish(out_dir, _populate_passed)
    return (
        {
            "status": "ccw_estimated",
            "gate_status": "passed",
            "protocol_id": norm_proto["protocol_id"],
            "protocol_version": CCW_PROTOCOL_VERSION,
            "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
            "simulation_or_data_label": norm_proto["simulation_or_data_label"],
            "variance_method": "subject_cluster_bootstrap_with_ipcw_refit",
            "seed": seed,
            "case_count": len(validated["subject_ids"]),
            "bootstrap_iterations": bootstrap_iterations,
            "output_path": str(out_dir),
        },
        0,
    )
