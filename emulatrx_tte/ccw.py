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
