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
    "1. 纵向条件可交换性 (Longitudinal conditional exchangeability): 所有共同影响宽限期/随访期给药启动（人工删失）与结局/竞争死亡的基线及离散期前协变量均被无误差测量并正确纳入分臂删失模型；CCW 无法消除未测混杂。",
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

    # 4. diagnostic_gates (all four required explicitly, zero hardcoded clinical defaults)
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
        "diagnostic_gates": {
            "min_ess_ratio": float(min_ess),
            "max_post_grace_smd": float(max_smd),
            "max_stabilized_weight": float(max_w),
            "min_uncensored_subjects_per_clone": int(min_uncens),
        },
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
    init_pat = [0.40, 0.35, 0.35, 0.30]
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

    return {
        "truth_auxiliary_seed": truth_seed,
        "n_truth_counterfactual": n_truth,
        "horizons": horizons_dict,
        "horizon_truth_table": horizon_table,
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

