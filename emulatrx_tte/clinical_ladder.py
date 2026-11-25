"""Clinician-facing 3-tier evidence ladder: crude analysis, conventional PS, full TTE.

Tier 1 reproduces the spreadsheet / Kaplan–Meier contrast a clinician would run before
any design rules. Tier 2 adds 1:1 caliper PSM and stabilized baseline IPTW on the same
grouping. Tier 3 reports the engine's target-trial estimate and a plain-language account
of what that extra design controlled. Point estimates are omitted whenever the caller
marks the TTE gate as not passed.
"""

from __future__ import annotations

import html
import math
from typing import Any

import numpy as np

from .competing_risks import (
    evaluate_step_curve,
    fit_weighted_cause_specific_cox,
    weighted_aalen_johansen_cif,
)
from .weighting import compute_weights, fit_binary_logistic_mle, kish_ess


_SMD_IMBALANCE = 0.10
_PS_CLIP = 1e-4
_MIN_PAIRS = 10


def _r(value: float | None, digits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _ci(lo: float | None, hi: float | None) -> list[float] | None:
    a, b = _r(lo), _r(hi)
    if a is None or b is None:
        return None
    return [a, b]


def _fmt_pp(value: float | None) -> str:
    if value is None:
        return "未能估计"
    return f"{value * 100:+.1f} 个百分点"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "未能估计"
    return f"{value:.{digits}f}"


def risks_at_horizon(
    time: np.ndarray,
    event: np.ndarray,
    horizon: float,
    weights: np.ndarray | None = None,
    *,
    origin_day: float = 0.0,
) -> dict[str, float | None]:
    """KM risk (1-S, competing events censored) and Aalen–Johansen CIF_1 at ``horizon``."""
    t_arr = np.asarray(time, dtype=np.float64)
    e_arr = np.asarray(event, dtype=np.int64)
    if t_arr.size == 0:
        return {"km_risk": None, "cif": None, "events": 0, "n": 0}
    w_arr = np.ones(t_arr.size, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(t_arr) & np.isfinite(w_arr) & (w_arr > 0.0) & (t_arr > float(origin_day))
    if not np.any(keep):
        return {"km_risk": None, "cif": None, "events": 0, "n": int(t_arr.size)}
    curve = weighted_aalen_johansen_cif(
        t_arr[keep],
        e_arr[keep],
        w_arr[keep],
        origin_day=float(origin_day),
    )
    cif = float(evaluate_step_curve(curve["time_day"], curve["cif_primary"], [float(horizon)])[0])
    km = float(evaluate_step_curve(curve["time_day"], curve["naive_km_primary_biased_ref"], [float(horizon)])[0])
    return {
        "km_risk": _r(km),
        "cif": _r(cif),
        "events": int(np.sum(e_arr[keep] == 1)),
        "n": int(np.sum(keep)),
    }


def cox_hr_wald(
    time: np.ndarray,
    event: np.ndarray,
    arm_idx: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float | None, list[float] | None]:
    """Unadjusted or weighted cause-specific Cox HR for arm 1 vs arm 0. None if an arm has zero events."""
    t_arr = np.asarray(time, dtype=np.float64)
    e_arr = np.asarray(event, dtype=np.int64)
    a_arr = np.asarray(arm_idx, dtype=np.int64)
    w_arr = np.ones(t_arr.size, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if int(np.sum((e_arr == 1) & (a_arr == 1) & (w_arr > 0))) == 0:
        return None, None
    if int(np.sum((e_arr == 1) & (a_arr == 0) & (w_arr > 0))) == 0:
        return None, None
    try:
        fitted = fit_weighted_cause_specific_cox(
            t_arr,
            e_arr,
            a_arr,
            2,
            w_arr,
            cause_code=1,
        )
    except Exception:
        return None, None
    hr = fitted["hr"][0] if fitted.get("hr") else None
    lo = fitted["ci_lower"][0] if fitted.get("ci_lower") else None
    hi = fitted["ci_upper"][0] if fitted.get("ci_upper") else None
    if hr is None:
        return None, None
    return _r(hr, 6), _ci(lo, hi)


def signed_smds(
    X: np.ndarray,
    names: list[str],
    arm_idx: np.ndarray,
    weights: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Signed SMD (treated arm 1 minus comparator arm 0) / pooled unweighted SD."""
    X_arr = np.asarray(X, dtype=np.float64)
    a_arr = np.asarray(arm_idx, dtype=np.int64)
    w_arr = np.ones(a_arr.size, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    if X_arr.ndim != 2 or X_arr.shape[0] != a_arr.size:
        return rows
    treated = a_arr == 1
    control = a_arr == 0
    for col, name in enumerate(names):
        xt = X_arr[treated, col]
        xc = X_arr[control, col]
        wt = w_arr[treated]
        wc = w_arr[control]
        if xt.size < 2 or xc.size < 2:
            rows.append({"covariate": name, "smd": None})
            continue
        var_t = float(np.var(xt, ddof=1))
        var_c = float(np.var(xc, ddof=1))
        pooled = math.sqrt(0.5 * (var_t + var_c))
        sum_t = float(np.sum(wt))
        sum_c = float(np.sum(wc))
        mean_t = float(np.sum(wt * xt) / sum_t) if sum_t > 0 else float(np.mean(xt))
        mean_c = float(np.sum(wc * xc) / sum_c) if sum_c > 0 else float(np.mean(xc))
        if pooled == 0.0:
            smd = 0.0 if abs(mean_t - mean_c) == 0.0 else None
        else:
            smd = (mean_t - mean_c) / pooled
        rows.append({"covariate": name, "smd": _r(smd, 6)})
    return rows


def _propensity(X: np.ndarray, arm_idx: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64)
    y = (np.asarray(arm_idx) == 1).astype(np.float64)
    if X_arr.ndim != 2 or X_arr.shape[1] == 0:
        X_arr = np.zeros((y.size, 1), dtype=np.float64)
    return np.clip(fit_binary_logistic_mle(X_arr, y), _PS_CLIP, 1.0 - _PS_CLIP)


def match_one_to_one(logit_ps: np.ndarray, arm_idx: np.ndarray) -> dict[str, Any]:
    """Greedy nearest-neighbor matching on the logit propensity score, without replacement."""
    logit = np.asarray(logit_ps, dtype=np.float64)
    arm = np.asarray(arm_idx, dtype=np.int64)
    finite = np.isfinite(logit)
    sd = float(np.std(logit[finite], ddof=1)) if int(np.sum(finite)) > 1 else 0.0
    caliper = max(0.2 * sd, 0.05) if math.isfinite(sd) else 0.05
    treated = [int(i) for i in np.flatnonzero((arm == 1) & finite)]
    controls = [int(i) for i in np.flatnonzero((arm == 0) & finite)]
    treated.sort(key=lambda i: (float(logit[i]), i))
    unused = set(controls)
    pairs: list[tuple[int, int]] = []
    for ti in treated:
        best: int | None = None
        best_d: float | None = None
        lt = float(logit[ti])
        for ci in unused:
            dist = abs(float(logit[ci]) - lt)
            if dist <= caliper + 1e-12 and (best_d is None or dist < best_d - 1e-15 or (abs(dist - best_d) <= 1e-15 and ci < best)):
                best = ci
                best_d = dist
        if best is not None:
            unused.remove(best)
            pairs.append((ti, best))
    n_eligible = int(np.sum((arm == 0) | (arm == 1)))
    matched_pairs = len(pairs)
    matched_idx = np.array([i for pair in pairs for i in pair], dtype=np.int64)
    status = "matched" if matched_pairs >= _MIN_PAIRS else "insufficient_matches"
    return {
        "status": status,
        "caliper": _r(caliper, 6),
        "matched_pairs": matched_pairs,
        "matched_n_total": 2 * matched_pairs,
        "matched_index": matched_idx,
        "discarded_subjects_count": int(n_eligible - 2 * matched_pairs),
        "discarded_subjects_pct": _r((n_eligible - 2 * matched_pairs) / n_eligible, 6) if n_eligible else None,
    }


def _stabilized_iptw(ps: np.ndarray, arm_idx: np.ndarray) -> np.ndarray:
    arm = np.asarray(arm_idx, dtype=np.int64)
    prob = float(np.mean(arm == 1))
    prob = min(max(prob, 1e-6), 1.0 - 1e-6)
    e_hat = np.clip(ps, 1e-6, 1.0 - 1e-6)
    weights = np.where(arm == 1, prob / e_hat, (1.0 - prob) / (1.0 - e_hat)).astype(np.float64)
    lo, hi = np.quantile(weights, [0.01, 0.99])
    if float(hi) > float(lo) and (float(np.max(weights)) > 20.0 or float(np.min(weights)) < 0.05):
        weights = np.clip(weights, float(lo), float(hi))
    return weights


def _arm_bundle(
    time: np.ndarray,
    event: np.ndarray,
    arm_idx: np.ndarray,
    horizon: float,
    weights: np.ndarray | None,
    *,
    origin_day: float,
) -> dict[str, Any]:
    treated = arm_idx == 1
    control = arm_idx == 0
    w = np.ones(arm_idx.size, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    tr = risks_at_horizon(time[treated], event[treated], horizon, w[treated], origin_day=origin_day)
    cr = risks_at_horizon(time[control], event[control], horizon, w[control], origin_day=origin_day)
    hr, hr_ci = cox_hr_wald(time, event, arm_idx, w)
    rd_cif = None if tr["cif"] is None or cr["cif"] is None else _r(tr["cif"] - cr["cif"])
    rd_km = None if tr["km_risk"] is None or cr["km_risk"] is None else _r(tr["km_risk"] - cr["km_risk"])
    rr = None
    if tr["cif"] is not None and cr["cif"] is not None and cr["cif"] > 0:
        rr = _r(tr["cif"] / cr["cif"], 6)
    return {
        "n_treated": int(np.sum(treated)),
        "n_comparator": int(np.sum(control)),
        "events_treated": int(np.sum((event == 1) & treated)),
        "events_comparator": int(np.sum((event == 1) & control)),
        "km_treated": tr["km_risk"],
        "km_comparator": cr["km_risk"],
        "cif_treated": tr["cif"],
        "cif_comparator": cr["cif"],
        "rd_cif": rd_cif,
        "rd_km": rd_km,
        "rr_cif": rr,
        "hr": hr,
        "hr_ci_95": hr_ci,
        "weights": w,
    }


def build_binary_exposure_ladder(
    time: np.ndarray,
    event: np.ndarray,
    arm_idx: np.ndarray,
    X: np.ndarray,
    covariate_names: list[str],
    horizon: float,
    *,
    grouping: str,
    design_family: str,
    origin_day: float = 0.0,
    raw_n_treated: int | None = None,
    raw_n_comparator: int | None = None,
    tier3: dict[str, Any] | None = None,
    compute_overlap_tier3: bool = False,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute tiers 1–2 on a binary exposure and attach a supplied or overlap-weighted tier 3."""
    t_arr = np.asarray(time, dtype=np.float64)
    e_arr = np.asarray(event, dtype=np.int64)
    a_arr = np.asarray(arm_idx, dtype=np.int64)
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.shape[0] != a_arr.size:
        X_arr = np.zeros((a_arr.size, 0), dtype=np.float64)
        covariate_names = []

    crude = _arm_bundle(t_arr, e_arr, a_arr, horizon, None, origin_day=origin_day)
    smd_rows = signed_smds(X_arr, list(covariate_names), a_arr)
    imbalanced = [
        {"covariate": row["covariate"], "smd": row["smd"]}
        for row in smd_rows
        if row["smd"] is not None and abs(row["smd"]) > _SMD_IMBALANCE
    ]
    imbalanced.sort(key=lambda row: abs(row["smd"]), reverse=True)
    finite_smds = [abs(row["smd"]) for row in smd_rows if row["smd"] is not None]
    tier1 = {
        "tier_id": "tier_1_unadjusted_crude",
        "label_zh": "第一阶：原始未校正直接分析",
        "grouping": grouping,
        "primary_horizon_days": _r(horizon, 4),
        "n_treated": crude["n_treated"],
        "n_comparator": crude["n_comparator"],
        "raw_input_n_treated": int(raw_n_treated if raw_n_treated is not None else crude["n_treated"]),
        "raw_input_n_comparator": int(raw_n_comparator if raw_n_comparator is not None else crude["n_comparator"]),
        "analytic_n_treated": crude["n_treated"],
        "analytic_n_comparator": crude["n_comparator"],
        "events_treated": crude["events_treated"],
        "events_comparator": crude["events_comparator"],
        "unadjusted_km_risk_treated": crude["km_treated"],
        "unadjusted_km_risk_comparator": crude["km_comparator"],
        "unadjusted_cif_treated": crude["cif_treated"],
        "unadjusted_cif_comparator": crude["cif_comparator"],
        "crude_rd_km": crude["rd_km"],
        "crude_rd_cif": crude["rd_cif"],
        "crude_rr_cif": crude["rr_cif"],
        "crude_hr": crude["hr"],
        "crude_hr_ci_95": crude["hr_ci_95"],
        "max_unadjusted_abs_smd": _r(max(finite_smds), 6) if finite_smds else None,
        "imbalanced_baseline_covariates": imbalanced,
        "baseline_smd": smd_rows,
    }

    ps = _propensity(X_arr if X_arr.shape[1] else np.zeros((a_arr.size, 1)), a_arr)
    logit = np.log(ps / (1.0 - ps))
    matched = match_one_to_one(logit, a_arr)
    psm_block: dict[str, Any] = {
        "status": matched["status"],
        "caliper_logit": matched["caliper"],
        "matched_pairs": matched["matched_pairs"],
        "matched_n_total": matched["matched_n_total"],
        "discarded_subjects_count": matched["discarded_subjects_count"],
        "discarded_subjects_pct": matched["discarded_subjects_pct"],
    }
    post_psm_rows: list[dict[str, Any]] = []
    if matched["status"] == "matched":
        idx = matched["matched_index"]
        psm = _arm_bundle(t_arr[idx], e_arr[idx], a_arr[idx], horizon, None, origin_day=origin_day)
        post_psm_rows = signed_smds(X_arr[idx], list(covariate_names), a_arr[idx])
        post_abs = [abs(row["smd"]) for row in post_psm_rows if row["smd"] is not None]
        psm_block.update(
            {
                "post_psm_max_abs_smd": _r(max(post_abs), 6) if post_abs else None,
                "balanced_baseline_covariates": [
                    {
                        "covariate": pre["covariate"],
                        "pre_smd": pre["smd"],
                        "post_psm_smd": post["smd"],
                    }
                    for pre, post in zip(smd_rows, post_psm_rows)
                ],
                "psm_km_risk_treated": psm["km_treated"],
                "psm_km_risk_comparator": psm["km_comparator"],
                "psm_cif_treated": psm["cif_treated"],
                "psm_cif_comparator": psm["cif_comparator"],
                "psm_rd": psm["rd_cif"],
                "psm_rd_cif": psm["rd_cif"],
                "psm_rd_km": psm["rd_km"],
                "psm_hr": psm["hr"],
                "psm_hr_ci_95": psm["hr_ci_95"],
            }
        )
    else:
        psm_block.update(
            {
                "post_psm_max_abs_smd": None,
                "balanced_baseline_covariates": [],
                "psm_km_risk_treated": None,
                "psm_km_risk_comparator": None,
                "psm_cif_treated": None,
                "psm_cif_comparator": None,
                "psm_rd": None,
                "psm_rd_cif": None,
                "psm_rd_km": None,
                "psm_hr": None,
                "psm_hr_ci_95": None,
            }
        )

    iptw_w = _stabilized_iptw(ps, a_arr)
    iptw = _arm_bundle(t_arr, e_arr, a_arr, horizon, iptw_w, origin_day=origin_day)
    iptw_smd = signed_smds(X_arr, list(covariate_names), a_arr, iptw_w)
    iptw_abs = [abs(row["smd"]) for row in iptw_smd if row["smd"] is not None]
    iptw_block = {
        "iptw_ess": _r(kish_ess(iptw_w), 4),
        "post_iptw_max_abs_smd": _r(max(iptw_abs), 6) if iptw_abs else None,
        "iptw_cif_treated": iptw["cif_treated"],
        "iptw_cif_comparator": iptw["cif_comparator"],
        "iptw_rd": iptw["rd_cif"],
        "iptw_hr": iptw["hr"],
        "iptw_hr_ci_95": iptw["hr_ci_95"],
        "post_iptw_smd": iptw_smd,
    }
    primary_method = "psm_1to1" if matched["status"] == "matched" else "simple_iptw"
    if primary_method == "psm_1to1":
        primary = {
            "risk_treated": psm_block["psm_cif_treated"],
            "risk_comparator": psm_block["psm_cif_comparator"],
            "rd": psm_block["psm_rd"],
            "hr": psm_block["psm_hr"],
            "hr_ci_95": psm_block["psm_hr_ci_95"],
        }
    else:
        primary = {
            "risk_treated": iptw_block["iptw_cif_treated"],
            "risk_comparator": iptw_block["iptw_cif_comparator"],
            "rd": iptw_block["iptw_rd"],
            "hr": iptw_block["iptw_hr"],
            "hr_ci_95": iptw_block["iptw_hr_ci_95"],
        }
    tier2 = {
        "tier_id": "tier_2_conventional_ps",
        "label_zh": "第二阶：传统常规倾向性评分调整 — 1:1 PSM 与简单基线 IPTW",
        "psm_1to1": psm_block,
        "simple_iptw": iptw_block,
        "primary_method": primary_method,
        **primary,
    }

    if tier3 is None and compute_overlap_tier3 and X_arr.shape[1] > 0:
        tier3 = _overlap_tier3(t_arr, e_arr, a_arr, X_arr, horizon, origin_day=origin_day)
    if tier3 is not None:
        tier3 = {
            "tier_id": "tier_3_target_trial_emulation",
            "label_zh": "第三阶：完整目标试验模拟 TTE",
            **tier3,
        }

    ladder = {
        "design_family": design_family,
        "primary_horizon_days": _r(horizon, 4),
        "tier_1_unadjusted_crude": tier1,
        "tier_2_conventional_ps": tier2,
        "tier_3_target_trial_emulation": tier3,
    }
    attribution = build_attribution(ladder, context or {}, post_psm_rows=post_psm_rows, crude_smd=smd_rows)
    return {"clinical_comparison_ladder": ladder, "what_tte_controlled_beyond_psm": attribution}


def _overlap_tier3(
    time: np.ndarray,
    event: np.ndarray,
    arm_idx: np.ndarray,
    X: np.ndarray,
    horizon: float,
    *,
    origin_day: float,
) -> dict[str, Any]:
    fitted = compute_weights(X, arm_idx.astype(np.int64), ["comparator", "treated"], weight_method="overlap")
    weights = np.asarray(fitted.raw_weights, dtype=np.float64)
    bundle = _arm_bundle(time, event, arm_idx, horizon, weights, origin_day=origin_day)
    smd_rows = signed_smds(X, [f"x{i}" for i in range(X.shape[1])], arm_idx, weights)
    abs_smds = [abs(row["smd"]) for row in smd_rows if row["smd"] is not None]
    return {
        "tte_cif_treated": bundle["cif_treated"],
        "tte_cif_comparator": bundle["cif_comparator"],
        "tte_rd": bundle["rd_cif"],
        "tte_rd_ci_95": None,
        "tte_rr_cif": bundle["rr_cif"],
        "tte_rr_ci_95": None,
        "shr": None,
        "shr_ci_95": None,
        "cshr_1": bundle["hr"],
        "cshr_1_ci_95": bundle["hr_ci_95"],
        "evalue_point": None,
        "evalue_ci_bound": None,
        "weighting_scheme": "overlap",
        "tte_retained_n": int(arm_idx.size),
        "tte_ess": _r(kish_ess(weights), 4),
        "post_tte_max_abs_smd": _r(max(abs_smds), 6) if abs_smds else None,
        "post_tte_smd": smd_rows,
        "source": "overlap_weighting_engine",
    }


def build_attribution(
    ladder: dict[str, Any],
    context: dict[str, Any],
    *,
    post_psm_rows: list[dict[str, Any]],
    crude_smd: list[dict[str, Any]],
) -> dict[str, Any]:
    tier1 = ladder["tier_1_unadjusted_crude"]
    tier2 = ladder["tier_2_conventional_ps"]
    tier3 = ladder.get("tier_3_target_trial_emulation")
    crude_rd = tier1.get("crude_rd_cif")
    psm_rd = tier2["psm_1to1"].get("psm_rd")
    tier2_rd = tier2.get("rd")
    tte_rd = None if tier3 is None else tier3.get("tte_rd")
    tte_hr = None
    if tier3 is not None:
        tte_hr = tier3.get("shr") if tier3.get("shr") is not None else tier3.get("cshr_1")

    headline = (
        f"同一主要时点上，第一阶未校正风险差为 {_fmt_pp(crude_rd)}（粗 HR {_fmt_num(tier1.get('crude_hr'))}），"
        f"第二阶{('1:1 PSM' if tier2.get('primary_method') == 'psm_1to1' else '简单 IPTW')}风险差为 {_fmt_pp(tier2_rd)}"
        f"（HR {_fmt_num(tier2.get('hr'))}）"
    )
    if tte_rd is None:
        headline += "；目标试验门禁未通过，第三阶点估计按失败关闭规则省略。"
    else:
        headline += f"，第三阶完整 TTE 风险差为 {_fmt_pp(tte_rd)}（sHR/csHR {_fmt_num(tte_hr)}）。"
        if crude_rd is not None and tte_rd is not None:
            removed = crude_rd - tte_rd
            headline += f"从粗分析到 TTE，风险差共移动 {_fmt_pp(removed)}。"

    tte_smd_map = {}
    if tier3 and isinstance(tier3.get("post_tte_smd"), list):
        tte_smd_map = {row["covariate"]: row.get("smd") for row in tier3["post_tte_smd"]}
    elif isinstance(context.get("tte_smd_by_name"), dict):
        tte_smd_map = context["tte_smd_by_name"]
    psm_map = {row["covariate"]: row.get("smd") for row in post_psm_rows}
    named_tte = context.get("tte_smd_by_name") if isinstance(context.get("tte_smd_by_name"), dict) else {}
    confounders = []
    for row in crude_smd:
        name = row["covariate"]
        confounders.append(
            {
                "name": name,
                "crude_smd": row.get("smd"),
                "psm_smd": psm_map.get(name),
                "tte_smd": tte_smd_map.get(name, named_tte.get(name)),
            }
        )

    excluded_n = int(context.get("excluded_n") or 0)
    landmark_day = float(context.get("landmark_day") or 0.0)
    eligibility_active = excluded_n > 0 or landmark_day > 0.0
    competing_n = int(context.get("competing_events") or 0)
    km_over_t = None
    km_over_c = None
    if tier1.get("unadjusted_km_risk_treated") is not None and tier1.get("unadjusted_cif_treated") is not None:
        km_over_t = _r(tier1["unadjusted_km_risk_treated"] - tier1["unadjusted_cif_treated"])
    if tier1.get("unadjusted_km_risk_comparator") is not None and tier1.get("unadjusted_cif_comparator") is not None:
        km_over_c = _r(tier1["unadjusted_km_risk_comparator"] - tier1["unadjusted_cif_comparator"])
    immortal_active = ladder.get("design_family") in {"grace_period_ccw", "landmark"} or bool(context.get("immortal_time_design"))
    false_benefit_pp = None
    if psm_rd is not None and tte_rd is not None:
        false_benefit_pp = _r(abs(psm_rd - tte_rd))
    discarded = tier2["psm_1to1"].get("discarded_subjects_pct")
    tv_names = list(context.get("time_varying_covariates") or [])
    ipcw_mode = str(context.get("censoring_ipcw") or "none")
    retention_active = discarded is not None or ipcw_mode != "none" or bool(tv_names)

    items = [
        {
            "category": "eligibility_and_time_zero_alignment",
            "label_zh": "纳排漏斗与时间零点 t0 严格对齐",
            "active": bool(eligibility_active),
            "quantitative_impact": {
                "excluded_n": excluded_n,
                "landmark_day": _r(landmark_day, 4),
                "raw_input_n": int(tier1["raw_input_n_treated"] + tier1["raw_input_n_comparator"]),
                "analytic_n": int(tier1["analytic_n_treated"] + tier1["analytic_n_comparator"]),
            },
            "clinical_explanation_zh": (
                f"时间零点对齐与纳排瀑布剔除了 {excluded_n} 例现患、不合格或 t0 前已发生事件的记录"
                f"（地标窗 landmark_day={landmark_day:g}）。常规按基线分组不会自动丢掉这些穿越记录。"
                if eligibility_active
                else "本方案时间零点即为基线，纳排瀑布没有额外剔除现患或零点前事件记录。"
            ),
        },
        {
            "category": "immortal_time_bias_elimination",
            "label_zh": "不朽时间偏倚消除",
            "active": bool(immortal_active and tte_rd is not None),
            "quantitative_impact": {
                "psm_rd": psm_rd,
                "tte_rd": tte_rd,
                "absolute_false_benefit_removed": false_benefit_pp,
                "grace_period_days": context.get("grace_period_days"),
            },
            "clinical_explanation_zh": (
                f"按“曾否用药”事后分组再做 1:1 PSM，风险差仍为 {_fmt_pp(psm_rd)}，因为用药前的存活被算进用药组。"
                f"双侧克隆–删失–加权 TTE 的风险差为 {_fmt_pp(tte_rd)}，绝对值相差 {_fmt_pp(false_benefit_pp)}。"
                "这部分差额是不朽时间造成的假阳性获益，不是药物效应。"
                if immortal_active and tte_rd is not None
                else "本方案没有宽限期延迟给药或地标后移，不朽时间偏倚不是本次数值变化的主要来源。"
            ),
        },
        {
            "category": "competing_risks_correction",
            "label_zh": "竞争死亡风险高估修正",
            "active": competing_n > 0,
            "quantitative_impact": {
                "competing_events": competing_n,
                "km_overestimation_treated": km_over_t,
                "km_overestimation_comparator": km_over_c,
            },
            "clinical_explanation_zh": (
                f"队列中有 {competing_n} 例竞争事件（event=2）。把竞争死亡当作失访的 1−KM，"
                f"在用药组高估累计发生率 { _fmt_pp(km_over_t) }、在对照组高估 { _fmt_pp(km_over_c) }。"
                "Aalen–Johansen CIF 与 Fine–Gray sHR 把这部分竞争死亡从目标事件里分开。"
                if competing_n > 0
                else "没有观察到竞争事件，1−KM 与 CIF 在本时点上一致。"
            ),
        },
        {
            "category": "target_population_retention_and_time_varying_ipcw",
            "label_zh": "目标重叠人群全保留与纵向时变依从混杂控制",
            "active": bool(retention_active and tte_rd is not None),
            "quantitative_impact": {
                "discarded_subjects_pct": discarded,
                "tte_retained_n": None if tier3 is None else tier3.get("tte_retained_n"),
                "tte_ess": None if tier3 is None else tier3.get("tte_ess"),
                "time_varying_covariates": tv_names,
                "censoring_ipcw": ipcw_mode,
                "evalue_point": None if tier3 is None else tier3.get("evalue_point"),
                "evalue_ci_bound": None if tier3 is None else tier3.get("evalue_ci_bound"),
            },
            "clinical_explanation_zh": (
                f"1:1 PSM 丢弃了合格样本的 {((discarded or 0) * 100):.1f}%。"
                f"重叠加权 / CCW 保留目标重叠人群（TTE 分析人数 {None if tier3 is None else tier3.get('tte_retained_n')}）。"
                + (
                    f"并按期前时变协变量 {', '.join(tv_names)} 与删失模型（{ipcw_mode}）做了纵向校正。"
                    if tv_names or ipcw_mode != "none"
                    else "本运行没有额外的期前时变协变量需要进入 IPCW。"
                )
                + (
                    f"未测混杂 E-value 点估计为 {_fmt_num(tier3.get('evalue_point'))}。"
                    if tier3 is not None and tier3.get("evalue_point") is not None
                    else ""
                )
            ),
        },
    ]
    takeaway_bits = [item["clinical_explanation_zh"] for item in items if item["active"]]
    if not takeaway_bits:
        takeaway_bits = ["三阶数值若接近，说明在本队列上基线混杂、时间对齐和竞争风险都没有把结论方向改写。"]
    return {
        "headline_progression_zh": headline,
        "baseline_confounders_controlled": confounders,
        "extra_tte_controls": items,
        "attribution_items": items,
        "clinician_takeaway_zh": " ".join(takeaway_bits),
    }


def render_clinical_ladder_markdown(bundle: dict[str, Any]) -> str:
    ladder = bundle["clinical_comparison_ladder"]
    note = bundle["what_tte_controlled_beyond_psm"]
    t1 = ladder["tier_1_unadjusted_crude"]
    t2 = ladder["tier_2_conventional_ps"]
    t3 = ladder.get("tier_3_target_trial_emulation")
    lines = [
        "# 临床专家三阶结果对比：原始直接分析 vs 常规倾向评分(PSM/IPTW) vs 完整目标试验模拟(TTE)",
        "",
        note["headline_progression_zh"],
        "",
        "| 阶梯 | 用药组风险 | 对照组风险 | 风险差 | HR / sHR |",
        "| --- | ---: | ---: | ---: | --- |",
        (
            f"| 第一阶 未校正 | { _fmt_num(t1.get('unadjusted_cif_treated'), 3) } | "
            f"{ _fmt_num(t1.get('unadjusted_cif_comparator'), 3) } | {_fmt_pp(t1.get('crude_rd_cif'))} | "
            f"{_fmt_num(t1.get('crude_hr'))} |"
        ),
        (
            f"| 第二阶 {t2.get('primary_method')} | {_fmt_num(t2.get('risk_treated'), 3)} | "
            f"{_fmt_num(t2.get('risk_comparator'), 3)} | {_fmt_pp(t2.get('rd'))} | {_fmt_num(t2.get('hr'))} |"
        ),
    ]
    if t3 is None:
        lines.append("| 第三阶 完整 TTE | 门禁未通过，点估计省略 | — | — | — |")
    else:
        lines.append(
            f"| 第三阶 完整 TTE | {_fmt_num(t3.get('tte_cif_treated'), 3)} | "
            f"{_fmt_num(t3.get('tte_cif_comparator'), 3)} | {_fmt_pp(t3.get('tte_rd'))} | "
            f"{_fmt_num(t3.get('shr') if t3.get('shr') is not None else t3.get('cshr_1'))} |"
        )
    lines.extend(["", "## 为什么做完 TTE 结果变了", ""])
    for item in note["extra_tte_controls"]:
        flag = "已启用" if item["active"] else "未触发"
        lines.append(f"- **{item['label_zh']}**（{flag}）：{item['clinical_explanation_zh']}")
    lines.append("")
    lines.append(note["clinician_takeaway_zh"])
    lines.append("")
    return "\n".join(lines)


def render_clinical_ladder_html(bundle: dict[str, Any]) -> str:
    """Self-contained HTML callout. No external URLs."""
    esc = lambda value: html.escape(str(value), quote=True)
    markdown_rows = render_clinical_ladder_markdown(bundle).splitlines()
    note = bundle["what_tte_controlled_beyond_psm"]
    ladder = bundle["clinical_comparison_ladder"]
    t1 = ladder["tier_1_unadjusted_crude"]
    t2 = ladder["tier_2_conventional_ps"]
    t3 = ladder.get("tier_3_target_trial_emulation")

    def cell(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    t3_cells = (
        "<td colspan='4'>门禁未通过，不输出第三阶点估计</td>"
        if t3 is None
        else (
            f"<td>{cell(t3.get('tte_cif_treated'))}</td><td>{cell(t3.get('tte_cif_comparator'))}</td>"
            f"<td>{cell(t3.get('tte_rd'))}</td><td>{cell(t3.get('shr') if t3.get('shr') is not None else t3.get('cshr_1'))}</td>"
        )
    )
    items = "".join(
        f"<li><strong>{esc(item['label_zh'])}</strong>：{esc(item['clinical_explanation_zh'])}</li>"
        for item in note["extra_tte_controls"]
        if item["active"]
    )
    return (
        "<section class='clinical-ladder' style='border:1px solid #99f6e4;border-left:6px solid #0f766e;"
        "background:#f0fdfa;padding:14px 16px;margin:16px 0;border-radius:8px'>"
        "<h2 style='margin:0 0 8px;font-size:1.15rem'>临床专家三阶结果对比：原始直接分析 vs 常规倾向评分(PSM/IPTW) vs 完整目标试验模拟(TTE)</h2>"
        f"<p><strong>{esc(note['headline_progression_zh'])}</strong></p>"
        "<div class='scroll'><table><thead><tr><th>阶梯</th><th>用药组 CIF</th><th>对照组 CIF</th><th>风险差</th><th>HR / sHR</th></tr></thead><tbody>"
        f"<tr><td>第一阶 未校正</td><td>{cell(t1.get('unadjusted_cif_treated'))}</td>"
        f"<td>{cell(t1.get('unadjusted_cif_comparator'))}</td><td>{cell(t1.get('crude_rd_cif'))}</td><td>{cell(t1.get('crude_hr'))}</td></tr>"
        f"<tr><td>第二阶 {esc(t2.get('primary_method'))}</td><td>{cell(t2.get('risk_treated'))}</td>"
        f"<td>{cell(t2.get('risk_comparator'))}</td><td>{cell(t2.get('rd'))}</td><td>{cell(t2.get('hr'))}</td></tr>"
        f"<tr><td>第三阶 完整 TTE</td>{t3_cells}</tr>"
        "</tbody></table></div>"
        f"<ul>{items}</ul><p>{esc(note['clinician_takeaway_zh'])}</p>"
        f"<details><summary>阶梯原文</summary><pre>{esc(chr(10).join(markdown_rows))}</pre></details>"
        "</section>"
    )


def _map_binary_from_frame(frame, protocol: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    import pandas as pd

    strategies = protocol["treatment_strategies"]
    ref_id = str(strategies[0]["id"])
    active = next((s for s in strategies if s.get("role") == "active"), strategies[1] if len(strategies) > 1 else strategies[0])
    act_id = str(active["id"])
    arm = frame["treatment_arm"].astype(str)
    keep = arm.isin([ref_id, act_id])
    sub = frame.loc[keep]
    arm_idx = np.where(sub["treatment_arm"].astype(str).to_numpy() == act_id, 1, 0).astype(np.int64)
    time = pd.to_numeric(sub["duration_day"], errors="coerce").to_numpy(dtype=np.float64)
    raw_event = pd.to_numeric(sub["event_code"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    primary = int(protocol["outcomes"]["primary_event_code"])
    competing = {int(code) for code in protocol["outcomes"]["competing_event_codes"]}
    event = np.zeros(raw_event.size, dtype=np.int64)
    event[raw_event == primary] = 1
    event[np.isin(raw_event, list(competing))] = 2
    cov_names = [name for name in protocol.get("_covariate_names", []) if name in sub.columns]
    if not cov_names:
        cov_names = [
            col
            for col in sub.columns
            if col not in {"treatment_arm", "duration_day", "event_code", "duration_since_landmark_day"}
            and not str(col).startswith(("ps__", "raw_weight__", "normalized_weight__", "analysis_weight__"))
            and pd.to_numeric(sub[col], errors="coerce").notna().mean() > 0.9
        ]
    if cov_names:
        X = np.column_stack([pd.to_numeric(sub[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) for name in cov_names])
    else:
        X = np.zeros((len(sub), 0), dtype=np.float64)
    return time, event, arm_idx, X, cov_names


def attach_static_ladder(estimates: dict, *, frame, protocol: dict, diagnostics: dict | None = None) -> dict:
    """Mutate a passed-gate estimates payload with the 3-tier ladder. Never called on blocked runs."""
    diag = diagnostics or {}
    protocol = dict(protocol)
    if diag.get("covariates"):
        protocol["_covariate_names"] = list(diag["covariates"])
    time, event, arm_idx, X, names = _map_binary_from_frame(frame, protocol)
    horizons = list(estimates.get("horizons_days") or [])
    horizon = float(horizons[-1]) if horizons else float(protocol["followup"]["horizon_days"][-1])
    origin = float(estimates.get("landmark_day") or protocol["time_zero"].get("landmark_day") or 0.0)
    ref_id = estimates["reference_arm"]["id"]
    table = estimates.get("co_presented_comparison_table") or []
    primary_rows = [row for row in table if abs(float(row.get("horizon_day", -1)) - horizon) < 1e-6]
    row = primary_rows[-1] if primary_rows else (table[-1] if table else None)
    tier3 = None
    if row is not None:
        contrast = None
        for value in (estimates.get("contrasts") or {}).values():
            if value.get("active_arm_id") == row.get("active_arm_id") or contrast is None:
                contrast = value
                if value.get("active_arm_id") == row.get("active_arm_id"):
                    break
        shr = None if contrast is None else contrast.get("fine_gray_shr_primary") or {}
        cshr = None if contrast is None else contrast.get("cause_specific_cox_primary") or {}
        ess = None
        ess_diag = (diag.get("ess_diagnostics") or {}).get("by_arm") or {}
        if ess_diag:
            ess = _r(sum(float(item.get("ess") or 0.0) for item in ess_diag.values()), 4)
        tte_smd = {}
        tier3 = {
            "tte_cif_treated": _r(row.get("cif_primary_active")),
            "tte_cif_comparator": _r(row.get("cif_primary_ref")),
            "tte_rd": _r(row.get("rd_primary")),
            "tte_rd_ci_95": _ci(row.get("rd_primary_ci_lower"), row.get("rd_primary_ci_upper")),
            "tte_rr_cif": _r(row.get("rr_cif_primary"), 6),
            "tte_rr_ci_95": _ci(row.get("rr_cif_primary_ci_lower"), row.get("rr_cif_primary_ci_upper")),
            "shr": _r(row.get("shr_primary"), 6),
            "shr_ci_95": _ci(row.get("shr_primary_ci_lower"), row.get("shr_primary_ci_upper")),
            "cshr_1": _r(row.get("cshr_primary"), 6),
            "cshr_1_ci_95": _ci(row.get("cshr_primary_ci_lower"), row.get("cshr_primary_ci_upper")),
            "evalue_point": _r(row.get("evalue_rr_cif_point"), 4),
            "evalue_ci_bound": _r(row.get("evalue_rr_cif_ci_limit"), 4),
            "weighting_scheme": estimates.get("weight_method"),
            "tte_retained_n": int(diag.get("retained_n") or arm_idx.size),
            "tte_ess": ess,
            "post_tte_max_abs_smd": _r(diag.get("max_abs_smd_weighted"), 6),
            "reference_arm_id": ref_id,
            "fine_gray_detail": {"point": shr.get("point") if isinstance(shr, dict) else None},
            "cause_specific_detail": {"point": cshr.get("point") if isinstance(cshr, dict) else None},
        }
        del tte_smd
    initial = diag.get("initial_arm_counts") or {}
    active_id = str(protocol["treatment_strategies"][1]["id"]) if len(protocol["treatment_strategies"]) > 1 else str(protocol["treatment_strategies"][0]["id"])
    for strategy in protocol["treatment_strategies"]:
        if strategy.get("role") == "active":
            active_id = str(strategy["id"])
    if tier3 is not None and "analysis_weight" in getattr(frame, "columns", []):
        import pandas as pd

        weight_keep = frame["treatment_arm"].astype(str).isin([ref_id, active_id])
        weight_values = pd.to_numeric(frame.loc[weight_keep, "analysis_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if weight_values.size == arm_idx.size and names:
            weighted_rows = signed_smds(X, names, arm_idx, weight_values)
            tier3["post_tte_smd"] = weighted_rows
            finite_w = [abs(row["smd"]) for row in weighted_rows if row["smd"] is not None]
            if finite_w and tier3.get("post_tte_max_abs_smd") is None:
                tier3["post_tte_max_abs_smd"] = _r(max(finite_w), 6)
    context = {
        "excluded_n": int(diag.get("landmark_excluded_n") or max(0, int(diag.get("initial_n") or arm_idx.size) - int(diag.get("retained_n") or arm_idx.size))),
        "landmark_day": origin,
        "competing_events": int(np.sum(event == 2)),
        "immortal_time_design": origin > 0.0,
        "censoring_ipcw": (estimates.get("censoring_model") or {}).get("censoring_ipcw_mode", "none"),
        "time_varying_covariates": [],
    }
    bundle = build_binary_exposure_ladder(
        time,
        event,
        arm_idx,
        X,
        names,
        horizon,
        grouping="raw_exposure",
        design_family="landmark" if origin > 0 else "static",
        origin_day=origin,
        raw_n_treated=None if not initial else int(initial.get(active_id, 0)),
        raw_n_comparator=None if not initial else int(initial.get(ref_id, 0)),
        tier3=tier3,
        context=context,
    )
    estimates["clinical_comparison_ladder"] = bundle["clinical_comparison_ladder"]
    estimates["what_tte_controlled_beyond_psm"] = bundle["what_tte_controlled_beyond_psm"]
    return bundle


def attach_ccw_ladder(
    estimates: dict,
    *,
    time: np.ndarray,
    event: np.ndarray,
    ever_initiated: np.ndarray,
    X: np.ndarray,
    covariate_names: list[str],
    diagnostics: dict,
    grace_days: float,
    tv_names: list[str] | None = None,
) -> dict:
    """Attach the ladder to a passed CCW estimates payload. Caller must not invoke this when the gate is blocked."""
    horizons = estimates.get("horizon_estimates") or []
    primary = horizons[-1]
    horizon = float(primary["horizon_day"])
    tier3 = {
        "tte_cif_treated": _r(primary["strategy_1_initiate_within_grace"]["CIF_1"]),
        "tte_cif_comparator": _r(primary["strategy_0_never_initiate"]["CIF_1"]),
        "tte_rd": _r(primary["ccw_rd_event1"]),
        "tte_rd_ci_95": _ci(primary.get("ccw_rd_event1_ci_lower"), primary.get("ccw_rd_event1_ci_upper")),
        "tte_rr_cif": _r(primary.get("ccw_rr_cif_event1"), 6),
        "tte_rr_ci_95": _ci(primary.get("ccw_rr_cif_event1_ci_lower"), primary.get("ccw_rr_cif_event1_ci_upper")),
        "shr": None,
        "shr_ci_95": None,
        "cshr_1": None,
        "cshr_1_ci_95": None,
        "evalue_point": _r(((primary.get("evalue_rr_cif_event1") or {}) if isinstance(primary.get("evalue_rr_cif_event1"), dict) else {}).get("point_estimate"), 4),
        "evalue_ci_bound": _r(((primary.get("evalue_rr_cif_event1") or {}) if isinstance(primary.get("evalue_rr_cif_event1"), dict) else {}).get("ci_limit_closer_to_null"), 4),
        "weighting_scheme": "bilateral_clone_censor_weight",
        "tte_retained_n": int(np.asarray(ever_initiated).size),
        "tte_ess": _r((diagnostics.get("post_grace_entry_summary") or {}).get("strategy_1_ess_ratio"), 6),
        "post_tte_max_abs_smd": _r((diagnostics.get("post_grace_entry_summary") or {}).get("max_weighted_abs_smd"), 6),
        "source": "ccw_engine",
    }
    tte_smd = {}
    for row in (diagnostics.get("post_grace_entry_summary") or {}).get("covariate_balance_rows") or []:
        name = row.get("covariate")
        if name:
            weighted = row.get("abs_smd_weighted")
            tte_smd[name] = _r(weighted, 6)
    context = {
        "excluded_n": 0,
        "landmark_day": 0.0,
        "grace_period_days": grace_days,
        "competing_events": int(np.sum(np.asarray(event) == 2)),
        "immortal_time_design": True,
        "censoring_ipcw": "arm_specific_ipcw",
        "time_varying_covariates": list(tv_names or []),
        "tte_smd_by_name": tte_smd,
    }
    bundle = build_binary_exposure_ladder(
        np.asarray(time, dtype=np.float64),
        np.asarray(event, dtype=np.int64),
        np.asarray(ever_initiated, dtype=np.int64),
        np.asarray(X, dtype=np.float64),
        list(covariate_names),
        horizon,
        grouping="naive_ever_initiated",
        design_family="grace_period_ccw",
        tier3=tier3,
        context=context,
    )
    estimates["clinical_comparison_ladder"] = bundle["clinical_comparison_ladder"]
    estimates["what_tte_controlled_beyond_psm"] = bundle["what_tte_controlled_beyond_psm"]
    return bundle
