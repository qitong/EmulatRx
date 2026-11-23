"""Landmark risk-set slicing, covariate balance / Kish ESS / positivity diagnostics, and exit-code-3 gate."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import (
    ContractError,
    METHODOLOGICAL_DISCLAIMER,
    PROTOCOL_VERSION,
    atomic_publish,
    code_provenance,
    dependency_versions,
    is_strict_int,
    json_bytes,
    read_strict_json,
    sha256_bytes,
    sha256_file,
    utc_now_iso,
)
from .protocol import load_and_validate_protocol
from .weighting import (
    ALLOWED_WEIGHT_METHODS,
    PAIRWISE_TRANSITIVITY_WARNING,
    compute_weights,
    kish_ess,
    parse_truncate_quantiles,
)


RESERVED_COHORT_COLUMNS = frozenset(
    (
        "synthetic_case_id",
        "subject_id",
        "simulation_or_data_label",
        "simulation_label",
        "treatment_arm",
        "treatment_idx",
        "duration_day",
        "event_code",
    )
)


def _is_offset_column(col: str) -> tuple[bool, str | None]:
    for suffix in ("__assessed_day_offset", "_assessed_day_offset"):
        if col.endswith(suffix) and len(col) > len(suffix):
            return True, col[: -len(suffix)]
    return False, None


def _load_covariate_offsets_arg(offsets_arg: Path | dict[str, int] | None) -> dict[str, int]:
    if offsets_arg is None:
        return {}
    if isinstance(offsets_arg, Path):
        _, raw_map = read_strict_json(offsets_arg, io_error_code="COVARIATE_OFFSETS_IO_ERROR")
    elif isinstance(offsets_arg, dict):
        raw_map = offsets_arg
    else:
        raise ContractError("INVALID_TYPE", "covariate_day_offsets must be a path or dict")
    result: dict[str, int] = {}
    for k, v in raw_map.items():
        if not isinstance(k, str) or not k.strip() or not is_strict_int(v):
            raise ContractError(
                "INVALID_COVARIATES",
                f"Covariate day offset '{k}: {v}' must map non-empty covariate name to integer day offset",
            )
        result[k.strip()] = v
    return result


def _resolve_and_check_covariates(
    df: pd.DataFrame,
    protocol: dict,
    explicit_offsets: dict[str, int],
) -> list[str]:
    landmark_day = int(protocol["time_zero"]["landmark_day"])

    # 1. Inspect every offset column embedded in cohort.csv
    csv_offset_targets: set[str] = set()
    for col in df.columns:
        is_off, target_cov = _is_offset_column(col)
        if not is_off or target_cov is None:
            continue
        csv_offset_targets.add(target_cov)
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            raise ContractError("INVALID_COHORT_SCHEMA", f"Offset column '{col}' contains non-numeric/missing values")
        max_off = float(vals.max()) if len(vals) > 0 else 0.0
        if max_off > landmark_day:
            raise ContractError(
                "POST_LANDMARK_COVARIATE_LEAK",
                f"Covariate '{target_cov}' is flagged via '{col}' with assessed_day_offset={max_off:g} "
                f"> landmark_day={landmark_day}. Hint: Exclude post-landmark covariates from propensity "
                "models to prevent conditioning on post-treatment / future variables.",
            )

    # 2. Inspect explicit offsets dict / CLI file
    for cov_name, off in explicit_offsets.items():
        if off > landmark_day:
            raise ContractError(
                "POST_LANDMARK_COVARIATE_LEAK",
                f"Covariate '{cov_name}' has assessed_day_offset={off} > landmark_day={landmark_day}. "
                "Hint: Propensity covariates must be measured at or before time_zero/landmark_day.",
            )

    # 3. Determine active propensity covariate columns
    if "covariates" in protocol:
        cov_names = [c["name"] for c in protocol["covariates"]]
        for c in protocol["covariates"]:
            if c["assessed_day_offset"] > landmark_day:
                raise ContractError(
                    "POST_LANDMARK_COVARIATE_LEAK",
                    f"Protocol covariate '{c['name']}' has assessed_day_offset={c['assessed_day_offset']} "
                    f"> landmark_day={landmark_day}.",
                )
        for extra_cov in sorted((csv_offset_targets | set(explicit_offsets.keys())) - set(cov_names)):
            if extra_cov in df.columns:
                cov_names.append(extra_cov)
    else:
        cov_names = []
        for col in df.columns:
            is_off, _ = _is_offset_column(col)
            if col in RESERVED_COHORT_COLUMNS or is_off:
                continue
            cov_names.append(col)

    if not cov_names:
        raise ContractError("INVALID_COHORT_SCHEMA", "No baseline covariate columns found for propensity modeling")

    missing = [c for c in cov_names if c not in df.columns]
    if missing:
        raise ContractError("INVALID_COHORT_SCHEMA", f"Cohort CSV is missing covariate column(s): {', '.join(missing)}")

    for c in cov_names:
        series = pd.to_numeric(df[c], errors="coerce")
        if series.isna().any() or not np.all(np.isfinite(series.to_numpy(dtype=np.float64))):
            raise ContractError("INVALID_COHORT_SCHEMA", f"Covariate column '{c}' contains missing or non-numeric values")

    return cov_names


def _compute_pair_smd_rows(
    *,
    pair_id: str,
    arm_j_id: str,
    arm_k_id: str,
    X_j: np.ndarray,
    X_k: np.ndarray,
    w_j: np.ndarray,
    w_k: np.ndarray,
    cov_names: list[str],
) -> list[dict]:
    rows: list[dict] = []
    has_both = (X_j.shape[0] > 0) and (X_k.shape[0] > 0)
    w_j_sum = float(np.sum(w_j)) if X_j.shape[0] > 0 else 0.0
    w_k_sum = float(np.sum(w_k)) if X_k.shape[0] > 0 else 0.0

    for col_idx, cov in enumerate(cov_names):
        if not has_both:
            rows.append(
                {
                    "pair_id": pair_id,
                    "arm_reference": arm_j_id,
                    "arm_comparator": arm_k_id,
                    "covariate": cov,
                    "unweighted_mean_ref": None,
                    "unweighted_mean_comp": None,
                    "weighted_mean_ref": None,
                    "weighted_mean_comp": None,
                    "pooled_baseline_sd": None,
                    "abs_smd_unweighted": None,
                    "abs_smd_weighted": None,
                }
            )
            continue

        xj = X_j[:, col_idx]
        xk = X_k[:, col_idx]
        mean_u_j = float(np.mean(xj))
        mean_u_k = float(np.mean(xk))
        var_j = float(np.var(xj, ddof=1)) if xj.size > 1 else 0.0
        var_k = float(np.var(xk, ddof=1)) if xk.size > 1 else 0.0
        pooled_sd = math.sqrt(0.5 * (var_j + var_k))

        mean_w_j = float(np.sum(w_j * xj) / w_j_sum) if w_j_sum > 0.0 else mean_u_j
        mean_w_k = float(np.sum(w_k * xk) / w_k_sum) if w_k_sum > 0.0 else mean_u_k

        diff_u = abs(mean_u_k - mean_u_j)
        diff_w = abs(mean_w_k - mean_w_j)
        if pooled_sd == 0.0:
            smd_u = 0.0 if diff_u == 0.0 else float("inf")
            smd_w = 0.0 if diff_w == 0.0 else float("inf")
        else:
            smd_u = diff_u / pooled_sd
            smd_w = diff_w / pooled_sd

        rows.append(
            {
                "pair_id": pair_id,
                "arm_reference": arm_j_id,
                "arm_comparator": arm_k_id,
                "covariate": cov,
                "unweighted_mean_ref": mean_u_j,
                "unweighted_mean_comp": mean_u_k,
                "weighted_mean_ref": mean_w_j,
                "weighted_mean_comp": mean_w_k,
                "pooled_baseline_sd": pooled_sd,
                "abs_smd_unweighted": smd_u,
                "abs_smd_weighted": smd_w,
            }
        )
    return rows


def _write_attrition_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "step_order",
        "step_id",
        "description",
        "remaining_n",
        "excluded_n",
        "excluded_primary_events",
        "excluded_competing_events",
        "excluded_censored",
        "arm_counts_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _fmt_optional_float(val: float | None) -> str:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return "" if val is None else "inf"
    return f"{val:.12f}"


def _write_balance_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "pair_id",
        "arm_reference",
        "arm_comparator",
        "covariate",
        "unweighted_mean_ref",
        "unweighted_mean_comp",
        "weighted_mean_ref",
        "weighted_mean_comp",
        "pooled_baseline_sd",
        "abs_smd_unweighted",
        "abs_smd_weighted",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "pair_id": r["pair_id"],
                    "arm_reference": r["arm_reference"],
                    "arm_comparator": r["arm_comparator"],
                    "covariate": r["covariate"],
                    "unweighted_mean_ref": _fmt_optional_float(r["unweighted_mean_ref"]),
                    "unweighted_mean_comp": _fmt_optional_float(r["unweighted_mean_comp"]),
                    "weighted_mean_ref": _fmt_optional_float(r["weighted_mean_ref"]),
                    "weighted_mean_comp": _fmt_optional_float(r["weighted_mean_comp"]),
                    "pooled_baseline_sd": _fmt_optional_float(r["pooled_baseline_sd"]),
                    "abs_smd_unweighted": _fmt_optional_float(r["abs_smd_unweighted"]),
                    "abs_smd_weighted": _fmt_optional_float(r["abs_smd_weighted"]),
                }
            )


def diagnose_cohort(
    protocol_path: Path,
    cohort_path: Path,
    *,
    weight_method: str,
    out_dir: Path,
    truncate_quantiles: str | tuple[float, float] | None = None,
    covariate_day_offsets: Path | dict[str, int] | None = None,
) -> tuple[dict, int]:
    """Run landmark slicing, propensity weighting, balance/ESS/positivity diagnostics, and gate evaluation.

    Returns (result_dict, exit_code) where exit_code is 0 on gate pass and 3 on gate block.
    ContractError is raised (CLI exit code 2) on malformed inputs or observable timestamp rule leaks.
    """
    raw_proto, protocol = load_and_validate_protocol(protocol_path)
    if weight_method not in ALLOWED_WEIGHT_METHODS:
        raise ContractError(
            "INVALID_WEIGHT_METHOD",
            f"Unsupported weight_method '{weight_method}'. Allowed: {sorted(ALLOWED_WEIGHT_METHODS)}",
        )
    strategies = protocol["treatment_strategies"]
    arm_ids = [s["id"] for s in strategies]
    num_arms = len(arm_ids)
    if weight_method == "overlap" and num_arms != 2:
        raise ContractError(
            "INVALID_WEIGHT_METHOD",
            f"Two-arm 'overlap' requires 2 treatment strategies (protocol defines {num_arms}). "
            "Hint: Use 'generalized_overlap' or 'pairwise_overlap'.",
        )
    parsed_trunc = parse_truncate_quantiles(truncate_quantiles)
    explicit_offsets = _load_covariate_offsets_arg(covariate_day_offsets)

    try:
        raw_cohort_bytes = cohort_path.read_bytes()
    except OSError as exc:
        raise ContractError("COHORT_IO_ERROR", str(exc)) from exc

    try:
        df = pd.read_csv(cohort_path)
    except Exception as exc:
        raise ContractError("INVALID_COHORT_SCHEMA", f"Failed to parse cohort CSV: {exc}") from exc

    id_col = "synthetic_case_id" if "synthetic_case_id" in df.columns else ("subject_id" if "subject_id" in df.columns else None)
    if id_col is None:
        raise ContractError("INVALID_COHORT_SCHEMA", "Cohort CSV must include 'synthetic_case_id' (or 'subject_id')")
    for req_col in ("treatment_arm", "duration_day", "event_code"):
        if req_col not in df.columns:
            raise ContractError("INVALID_COHORT_SCHEMA", f"Cohort CSV missing required column '{req_col}'")
    if len(df) == 0:
        raise ContractError("INVALID_COHORT_SCHEMA", "Cohort CSV contains 0 rows")

    # Validate treatment arms, event codes, durations
    arm_to_idx = {aid: idx for idx, aid in enumerate(arm_ids)}
    unknown_arms = sorted(set(df["treatment_arm"].astype(str).unique()) - set(arm_ids))
    if unknown_arms:
        raise ContractError(
            "INVALID_COHORT_SCHEMA",
            f"Cohort CSV contains unknown treatment_arm value(s): {unknown_arms} (expected {arm_ids})",
        )

    durations = pd.to_numeric(df["duration_day"], errors="coerce")
    if durations.isna().any() or (durations < 0.0).any():
        raise ContractError("INVALID_COHORT_SCHEMA", "duration_day must be finite and >= 0")

    events_num = pd.to_numeric(df["event_code"], errors="coerce")
    if events_num.isna().any() or not np.all(np.floor(events_num) == events_num):
        raise ContractError("INVALID_COHORT_SCHEMA", "event_code must be integer-valued")
    events_int = events_num.astype(int)
    primary_code = protocol["outcomes"]["primary_event_code"]
    censor_code = protocol["outcomes"]["censor_code"]
    competing_codes = set(protocol["outcomes"]["competing_event_codes"])
    valid_codes = {censor_code, primary_code} | competing_codes
    bad_codes = sorted(set(events_int.unique()) - valid_codes)
    if bad_codes:
        raise ContractError(
            "INVALID_COHORT_SCHEMA",
            f"Cohort CSV contains undeclared event_code(s): {bad_codes} (allowed: {sorted(valid_codes)})",
        )

    cov_names = _resolve_and_check_covariates(df, protocol, explicit_offsets)

    # Landmark risk-set slicing
    landmark_day = int(protocol["time_zero"]["landmark_day"])
    initial_arm_counts = {aid: int((df["treatment_arm"] == aid).sum()) for aid in arm_ids}
    attrition_rows = [
        {
            "step_order": 0,
            "step_id": "initial_cohort",
            "description": "Initial input cohort prior to landmark/baseline risk-set filtering",
            "remaining_n": int(len(df)),
            "excluded_n": 0,
            "excluded_primary_events": 0,
            "excluded_competing_events": 0,
            "excluded_censored": 0,
            "arm_counts_json": json.dumps(initial_arm_counts, sort_keys=True),
        }
    ]

    if landmark_day > 0:
        keep_mask = durations > float(landmark_day)
        step_desc = f"Landmark risk-set slice: retain subjects event-free and uncensored past landmark_day={landmark_day}"
    else:
        keep_mask = durations > 0.0
        step_desc = "Baseline risk-set check: retain subjects with strictly positive follow-up duration_day > 0"

    df_excl = df.loc[~keep_mask]
    excl_events = events_int.loc[~keep_mask]
    df_ret = df.loc[keep_mask].copy().reset_index(drop=True)
    ret_arm_counts = {aid: int((df_ret["treatment_arm"] == aid).sum()) for aid in arm_ids}
    attrition_rows.append(
        {
            "step_order": 1,
            "step_id": "landmark_risk_set_slice",
            "description": step_desc,
            "remaining_n": int(len(df_ret)),
            "excluded_n": int(len(df_excl)),
            "excluded_primary_events": int((excl_events == primary_code).sum()),
            "excluded_competing_events": int(excl_events.isin(competing_codes).sum()),
            "excluded_censored": int((excl_events == censor_code).sum()),
            "arm_counts_json": json.dumps(ret_arm_counts, sort_keys=True),
        }
    )

    gates = protocol["estimand"]["diagnostic_gates"]
    gate_max_smd = float(gates["max_abs_smd"])
    gate_min_ess = float(gates["min_ess_ratio"])
    gate_pos_eps = float(gates["positivity_epsilon"])

    gate_failures: list[dict] = []
    zero_arms = [aid for aid, cnt in ret_arm_counts.items() if cnt == 0]
    if zero_arms:
        gate_failures.append(
            {
                "code": "ZERO_RISK_SET",
                "detail": f"Empty risk set after slicing in treatment strategy arm(s): {zero_arms}",
            }
        )

    balance_rows: list[dict] = []
    ess_by_group: dict = {}
    propensity_summary: dict = {}
    w_res = None

    if not zero_arms:
        X = df_ret[cov_names].to_numpy(dtype=np.float64)
        arm_idx = df_ret["treatment_arm"].map(arm_to_idx).to_numpy(dtype=np.int64)
        w_res = compute_weights(
            X,
            arm_idx,
            arm_ids,
            weight_method=weight_method,
            truncate_quantiles=parsed_trunc,
        )

        if w_res.pairwise_results is not None:
            pair_summaries = {}
            min_ps = 1.0
            max_ps = 0.0
            min_ess_observed = 1.0
            for pr in w_res.pairwise_results:
                X_sub = X[pr.subset_indices]
                mask_j = pr.binary_arm_indicator == 0
                mask_k = pr.binary_arm_indicator == 1
                balance_rows.extend(
                    _compute_pair_smd_rows(
                        pair_id=pr.pair_id,
                        arm_j_id=pr.arm_j_id,
                        arm_k_id=pr.arm_k_id,
                        X_j=X_sub[mask_j],
                        X_k=X_sub[mask_k],
                        w_j=pr.raw_weights[mask_j],
                        w_k=pr.raw_weights[mask_k],
                        cov_names=cov_names,
                    )
                )
                n_j = int(np.sum(mask_j))
                n_k = int(np.sum(mask_k))
                ess_j = kish_ess(pr.raw_weights[mask_j])
                ess_k = kish_ess(pr.raw_weights[mask_k])
                ratio_j = ess_j / n_j if n_j > 0 else 0.0
                ratio_k = ess_k / n_k if n_k > 0 else 0.0
                min_ess_observed = min(min_ess_observed, ratio_j, ratio_k)
                ps_all = np.concatenate([pr.propensity_k_given_pair, 1.0 - pr.propensity_k_given_pair])
                p_min = float(np.min(ps_all))
                p_max = float(np.max(ps_all))
                min_ps = min(min_ps, p_min)
                max_ps = max(max_ps, p_max)
                pair_summaries[pr.pair_id] = {
                    "arm_j": {"id": pr.arm_j_id, "n": n_j, "ess": round(ess_j, 6), "ess_ratio": round(ratio_j, 6)},
                    "arm_k": {"id": pr.arm_k_id, "n": n_k, "ess": round(ess_k, 6), "ess_ratio": round(ratio_k, 6)},
                    "propensity_min": round(p_min, 8),
                    "propensity_max": round(p_max, 8),
                    "tilting_h_max": round(float(np.max(pr.tilting_h)), 8),
                }
            ess_by_group = {"pairwise": pair_summaries, "min_ess_ratio_observed": round(min_ess_observed, 8)}
            propensity_summary = {
                "min_propensity": round(min_ps, 8),
                "max_propensity": round(max_ps, 8),
            }
        else:
            assert w_res.propensity_matrix is not None
            assert w_res.raw_weights is not None
            assert w_res.tilting_h is not None
            for j in range(num_arms):
                for k in range(j + 1, num_arms):
                    mask_j = arm_idx == j
                    mask_k = arm_idx == k
                    balance_rows.extend(
                        _compute_pair_smd_rows(
                            pair_id=f"{arm_ids[j]}__vs__{arm_ids[k]}",
                            arm_j_id=arm_ids[j],
                            arm_k_id=arm_ids[k],
                            X_j=X[mask_j],
                            X_k=X[mask_k],
                            w_j=w_res.raw_weights[mask_j],
                            w_k=w_res.raw_weights[mask_k],
                            cov_names=cov_names,
                        )
                    )
            arm_ess_map = {}
            min_ess_observed = 1.0
            for k, aid in enumerate(arm_ids):
                mask_k = arm_idx == k
                n_k = int(np.sum(mask_k))
                ess_k = kish_ess(w_res.raw_weights[mask_k])
                ratio_k = ess_k / n_k if n_k > 0 else 0.0
                min_ess_observed = min(min_ess_observed, ratio_k)
                arm_ess_map[aid] = {
                    "n": n_k,
                    "ess": round(ess_k, 6),
                    "ess_ratio": round(ratio_k, 6),
                    "raw_weight_min": round(float(np.min(w_res.raw_weights[mask_k])), 8),
                    "raw_weight_max": round(float(np.max(w_res.raw_weights[mask_k])), 8),
                }
            ess_by_group = {"by_arm": arm_ess_map, "min_ess_ratio_observed": round(min_ess_observed, 8)}
            min_ps = float(np.min(w_res.propensity_matrix))
            max_ps = float(np.max(w_res.propensity_matrix))
            propensity_summary = {
                "min_propensity": round(min_ps, 8),
                "max_propensity": round(max_ps, 8),
                "tilting_h_min": round(float(np.min(w_res.tilting_h)), 8),
                "tilting_h_max": round(float(np.max(w_res.tilting_h)), 8),
            }

        # Evaluate quantitative SAP diagnostic gates
        weighted_smds = [r["abs_smd_weighted"] for r in balance_rows if r["abs_smd_weighted"] is not None]
        unweighted_smds = [r["abs_smd_unweighted"] for r in balance_rows if r["abs_smd_unweighted"] is not None]
        max_weighted_smd = float(max(weighted_smds)) if weighted_smds else float("nan")
        max_unweighted_smd = float(max(unweighted_smds)) if unweighted_smds else float("nan")

        if min_ps < gate_pos_eps or max_ps > (1.0 - gate_pos_eps):
            gate_failures.append(
                {
                    "code": "POSITIVITY_VIOLATION",
                    "detail": (
                        f"Observed propensity bounds [{min_ps:.6g}, {max_ps:.6g}] violate protocol "
                        f"[positivity_epsilon={gate_pos_eps}, {1.0 - gate_pos_eps:.6g}]"
                    ),
                }
            )
        if min_ess_observed < gate_min_ess:
            gate_failures.append(
                {
                    "code": "ESS_RATIO_BELOW_MIN",
                    "detail": (
                        f"Minimum arm Kish ESS ratio {min_ess_observed:.6f} < protocol "
                        f"min_ess_ratio={gate_min_ess:.6f}"
                    ),
                }
            )
        if not math.isfinite(max_weighted_smd) or max_weighted_smd > gate_max_smd:
            gate_failures.append(
                {
                    "code": "SMD_EXCEEDS_MAX",
                    "detail": (
                        f"Maximum post-weighting |SMD| {max_weighted_smd:.8f} > protocol "
                        f"max_abs_smd={gate_max_smd:.8f}"
                    ),
                }
            )
    else:
        for j in range(num_arms):
            for k in range(j + 1, num_arms):
                balance_rows.extend(
                    _compute_pair_smd_rows(
                        pair_id=f"{arm_ids[j]}__vs__{arm_ids[k]}",
                        arm_j_id=arm_ids[j],
                        arm_k_id=arm_ids[k],
                        X_j=np.empty((0, len(cov_names))),
                        X_k=np.empty((0, len(cov_names))),
                        w_j=np.empty(0),
                        w_k=np.empty(0),
                        cov_names=cov_names,
                    )
                )
        max_weighted_smd = None
        max_unweighted_smd = None

    gate_status = "passed" if len(gate_failures) == 0 else "blocked"
    exit_code = 0 if gate_status == "passed" else 3

    diagnostics_summary = {
        "gate_status": gate_status,
        "gate_failures": gate_failures,
        "weight_method": weight_method,
        "simulation_or_data_label": protocol["simulation_or_data_label"],
        "methodological_disclaimer": METHODOLOGICAL_DISCLAIMER,
        "transitivity_warning": (
            PAIRWISE_TRANSITIVITY_WARNING if weight_method == "pairwise_overlap" else None
        ),
        "landmark_day": landmark_day,
        "initial_n": int(len(df)),
        "retained_n": int(len(df_ret)),
        "retained_arm_counts": ret_arm_counts,
        "covariates": cov_names,
        "protocol_diagnostic_gates": gates,
        "max_abs_smd_unweighted": (
            round(max_unweighted_smd, 12)
            if (max_unweighted_smd is not None and math.isfinite(max_unweighted_smd))
            else None
        ),
        "max_abs_smd_weighted": (
            round(max_weighted_smd, 12)
            if (max_weighted_smd is not None and math.isfinite(max_weighted_smd))
            else None
        ),
        "ess_diagnostics": ess_by_group,
        "propensity_diagnostics": propensity_summary,
        "truncation_quantiles": list(parsed_trunc) if parsed_trunc is not None else None,
        "truncation_bounds": (
            list(w_res.truncation_bounds)
            if (w_res is not None and w_res.truncation_bounds is not None)
            else None
        ),
    }

    output_paths = ["attrition_table.csv", "balance_diagnostics.csv", "diagnostics_summary.json"]
    if gate_status == "passed":
        output_paths.append("weighted_cohort.csv")

    commit, code_dirty = code_provenance()
    deps = dependency_versions()

    def _populate(stage: Path) -> None:
        _write_attrition_csv(stage / "attrition_table.csv", attrition_rows)
        _write_balance_csv(stage / "balance_diagnostics.csv", balance_rows)
        (stage / "diagnostics_summary.json").write_bytes(json_bytes(diagnostics_summary))

        if gate_status == "passed":
            assert w_res is not None
            out_df = df_ret.copy()
            out_df["duration_since_landmark_day"] = (
                pd.to_numeric(out_df["duration_day"]) - float(landmark_day)
            ).map(lambda v: f"{float(v):.8f}")

            if w_res.pairwise_results is not None:
                for pr in w_res.pairwise_results:
                    raw_col = np.full(len(out_df), np.nan, dtype=np.float64)
                    norm_col = np.full(len(out_df), np.nan, dtype=np.float64)
                    ps_col = np.full(len(out_df), np.nan, dtype=np.float64)
                    raw_col[pr.subset_indices] = pr.raw_weights
                    norm_col[pr.subset_indices] = pr.normalized_weights
                    ps_col[pr.subset_indices] = pr.propensity_k_given_pair
                    out_df[f"ps__{pr.pair_id}"] = [
                        "" if np.isnan(v) else f"{v:.10f}" for v in ps_col
                    ]
                    out_df[f"raw_weight__{pr.pair_id}"] = [
                        "" if np.isnan(v) else f"{v:.10f}" for v in raw_col
                    ]
                    out_df[f"normalized_weight__{pr.pair_id}"] = [
                        "" if np.isnan(v) else f"{v:.10f}" for v in norm_col
                    ]
            else:
                assert w_res.propensity_matrix is not None
                assert w_res.raw_weights is not None
                assert w_res.normalized_weights is not None
                assert w_res.analysis_weights is not None
                assert w_res.tilting_h is not None
                for k, aid in enumerate(arm_ids):
                    out_df[f"ps_{aid}"] = [f"{v:.10f}" for v in w_res.propensity_matrix[:, k]]
                out_df["tilting_h"] = [f"{v:.10f}" for v in w_res.tilting_h]
                out_df["raw_weight"] = [f"{v:.10f}" for v in w_res.raw_weights]
                out_df["normalized_weight"] = [f"{v:.10f}" for v in w_res.normalized_weights]
                out_df["analysis_weight"] = [f"{v:.10f}" for v in w_res.analysis_weights]

            out_df.to_csv(stage / "weighted_cohort.csv", index=False, lineterminator="\n")

        manifest = {
            "status": f"tte_diagnostics_{gate_status}",
            "gate_status": gate_status,
            "activity": "tte_propensity_weighting_and_diagnostic_gate",
            "protocol_version": PROTOCOL_VERSION,
            "weight_method": weight_method,
            "simulation_or_data_label": protocol["simulation_or_data_label"],
            "methodological_disclaimer": METHODOLOGICAL_DISCLAIMER,
            "protocol_sha256": sha256_bytes(raw_proto),
            "cohort_sha256": sha256_bytes(raw_cohort_bytes),
            "code_commit": commit,
            "code_dirty": code_dirty,
            "dependency_versions": deps,
            "output_paths": output_paths,
            "output_sha256": {p: sha256_file(stage / p) for p in output_paths},
            "run_at_utc": utc_now_iso(),
        }
        (stage / "run_manifest.json").write_bytes(json_bytes(manifest))

    atomic_publish(out_dir, _populate)
    return (
        {
            "status": f"tte_diagnostics_{gate_status}",
            "gate_status": gate_status,
            "weight_method": weight_method,
            "simulation_or_data_label": protocol["simulation_or_data_label"],
            "output_path": str(out_dir),
            "gate_failures": gate_failures,
        },
        exit_code,
    )
