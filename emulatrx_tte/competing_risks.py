"""Weighted competing-risks survival analysis workbench (Aalen-Johansen CIF, Fine-Gray sHR, cause-specific Cox csHR)."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._common import (
    ALLOWED_SOURCE_MODES,
    ContractError,
    GateBlockedError,
    MAX_BOOTSTRAP_ITERATIONS,
    MAX_SEED,
    METHODOLOGICAL_DISCLAIMER,
    OBSERVATIONAL_CAUSAL_ASSUMPTIONS,
    PROHIBITED_CLAIMS,
    PROTOCOL_VERSION,
    SOURCE_MODE_SIMULATED_COHORT,
    SOURCE_MODE_SYNTHETIC_ONLY,
    SOURCE_MODE_USER_COHORT,
    atomic_publish,
    check_destination,
    claims_for_source_mode,
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
from .protocol import load_and_validate_protocol
from .weighting import ALLOWED_WEIGHT_METHODS, PAIRWISE_TRANSITIVITY_WARNING


ALLOWED_CENSORING_IPCW = frozenset(("none", "arm_stratified", "covariate_cox"))

SUBDISTRIBUTION_VS_CSHR_WARNING = (
    "Competing-risks interpretation guardrail (Fine & Gray 1999; Austin & Fine 2017): "
    "Fine-Gray subdistribution hazard ratio (sHR) reflects cumulative incidence dynamics in a risk set "
    "where subjects experiencing competing mortality/events remain at risk via IPCW weights. An observed "
    "sHR < 1.0 can arise solely from elevated competing mortality in the active arm (csHR_competing > 1.0) "
    "depleting subjects before primary events occur, and MUST NEVER be interpreted in isolation as a direct "
    "protective biological effect on the primary outcome. Always interpret sHR alongside primary csHR, "
    "competing csHR, primary CIF_1(t), and competing CIF_2(t) in the co-presented table."
)

HR_SELECTION_BIAS_WARNING = (
    "Hazard-ratio built-in selection bias warning (Hernán, Epidemiology 2010): "
    "Even cause-specific Cox hazard ratios (csHR) condition at each failure time t on surviving free of "
    "both primary and competing events up to t-, inducing post-baseline selection bias (differential "
    "depletion of susceptibles) even under perfect baseline randomization or overlap weighting. Pair "
    "hazard summaries with explicit horizon cumulative incidence contrasts RD(t) and RR_CIF(t)."
)

BASELINE_INDEPENDENT_CENSORING_ASSUMPTION = (
    "Baseline-weight-only censoring assumption (censoring_ipcw='none'): right-censoring is assumed "
    "independent of latent primary and competing event times conditional on treatment strategy within the "
    "baseline-weighted pseudo-population. Pass censoring_ipcw='arm_stratified' or 'covariate_cox' for "
    "joint baseline x IPCW censoring sensitivity analysis."
)


def parse_horizons_arg(
    horizons_arg: str | list[float] | tuple[float, ...] | None,
    *,
    default_horizons: list[float],
    landmark_day: float,
    admin_end_day: float,
) -> list[float]:
    """Parse and validate evaluation horizons against landmark_day and administrative_end_day."""
    if horizons_arg is None:
        raw_list = list(default_horizons)
    elif isinstance(horizons_arg, str):
        parts = [p.strip() for p in horizons_arg.split(",") if p.strip()]
        if not parts:
            raise ContractError("MISSING_HORIZON", "--horizons must specify at least one positive day value")
        raw_list = []
        for idx, p in enumerate(parts):
            try:
                raw_list.append(float(p))
            except ValueError as exc:
                raise ContractError("INVALID_FOLLOWUP", f"--horizons[{idx}] '{p}' is not a valid number") from exc
    elif isinstance(horizons_arg, (list, tuple)):
        raw_list = list(horizons_arg)
    else:
        raise ContractError("INVALID_TYPE", "horizons_days must be None, comma-separated string, or list of numbers")

    if len(raw_list) == 0:
        raise ContractError(
            "MISSING_HORIZON",
            "Competing-risks estimation requires at least one explicit evaluation horizon in followup.horizon_days "
            "or --horizons.",
        )

    out: list[float] = []
    for idx, val in enumerate(raw_list):
        if isinstance(val, (np.floating, np.integer)) and not isinstance(val, (bool, np.bool_)):
            val = float(val)
        if not is_strict_number(val) or float(val) <= 0.0:
            raise ContractError("INVALID_FOLLOWUP", f"Horizon [{idx}]={val} must be a positive finite number")
        hf = float(val)
        if landmark_day > 0.0 and hf <= landmark_day:
            raise ContractError(
                "INVALID_FOLLOWUP",
                f"Horizon {hf:g} must strictly exceed time_zero.landmark_day={landmark_day:g}",
            )
        if hf >= admin_end_day:
            raise ContractError(
                "INVALID_FOLLOWUP",
                f"Horizon {hf:g} must be strictly less than administrative_end_day={admin_end_day:g}",
            )
        out.append(hf)

    if len(set(out)) != len(out):
        raise ContractError("INVALID_FOLLOWUP", "Evaluation horizons must not contain duplicate values")
    return sorted(out)


def compute_reverse_km_left_limit(
    durations: np.ndarray,
    event_types: np.ndarray,
    weights: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    """Evaluate baseline-weighted reverse Kaplan-Meier censoring survival G_hat(t-) at eval_times.

    Event-before-censoring tie convention (Fine & Gray 1999; Geskus 2011): events (event_types != 0)
    at time c occur at c, immediately prior to censorings (event_types == 0) at c+. Thus G_hat(t-)
    multiplies (1 - dN_C(c) / Y_C(c)) strictly over censoring times c < t.
    """
    t_arr = np.asarray(durations, dtype=np.float64)
    e_arr = np.asarray(event_types, dtype=np.int64)
    w_arr = np.asarray(weights, dtype=np.float64)
    q_arr = np.asarray(eval_times, dtype=np.float64)

    cens_mask = e_arr == 0
    if not np.any(cens_mask) or q_arr.size == 0:
        return np.ones(q_arr.shape, dtype=np.float64)

    c_times = np.unique(t_arr[cens_mask])
    factors = np.ones(c_times.size, dtype=np.float64)
    for idx, c_val in enumerate(c_times):
        d_c = float(np.sum(w_arr[cens_mask & (t_arr == c_val)]))
        y_c = float(np.sum(w_arr[(t_arr > c_val) | (cens_mask & (t_arr == c_val))]))
        if y_c > 0.0:
            factors[idx] = max(0.0, 1.0 - d_c / y_c)
        else:
            factors[idx] = 0.0

    g_cum = np.cumprod(factors)
    pos = np.searchsorted(c_times, q_arr, side="left")
    out = np.ones(q_arr.shape, dtype=np.float64)
    has_prev = pos > 0
    out[has_prev] = g_cum[pos[has_prev] - 1]
    return out


def _fit_weighted_counting_process_cox(
    Z: np.ndarray,
    num_event_times: int,
    build_chunk_fn: Callable[[int, int], tuple[np.ndarray, np.ndarray]],
    *,
    ci_level: float = 0.95,
    chunk_size: int = 256,
    max_iter: int = 80,
    tol: float = 1e-11,
) -> dict:
    """Fit weighted counting-process partial likelihood with subject-clustered sandwich variance.

    build_chunk_fn(m_start, m_end) returns (d_chunk, W_chunk) each of shape (n, m_end - m_start),
    where d_chunk[i, m] >= 0 is subject i's failure weight at event time m and W_chunk[i, m] >= 0 is
    subject i's risk-set weight at event time m.
    """
    n, p = Z.shape
    if num_event_times == 0:
        return {
            "converged": False,
            "num_events_times": 0,
            "beta": [None] * p,
            "se_robust": [None] * p,
            "hr": [None] * p,
            "ci_lower": [None] * p,
            "ci_upper": [None] * p,
            "warning": "ZERO_CAUSE_EVENTS",
        }

    # Precompute total failure weight per subject to detect zero-event arms cleanly
    d_subj = np.zeros(n, dtype=np.float64)
    for m0 in range(0, num_event_times, chunk_size):
        m1 = min(num_event_times, m0 + chunk_size)
        d_ch, _ = build_chunk_fn(m0, m1)
        d_subj += np.sum(d_ch, axis=1)

    total_weighted_events = float(np.sum(d_subj))
    if total_weighted_events <= 0.0:
        return {
            "converged": False,
            "num_events_times": num_event_times,
            "beta": [None] * p,
            "se_robust": [None] * p,
            "hr": [None] * p,
            "ci_lower": [None] * p,
            "ci_upper": [None] * p,
            "warning": "ZERO_WEIGHTED_EVENTS",
        }

    # Check monotone likelihood for binary indicator columns (zero events in an active arm or reference arm)
    ref_mask = np.all(Z == 0.0, axis=1)
    if np.any(ref_mask) and float(np.sum(d_subj[ref_mask])) <= 0.0:
        return {
            "converged": False,
            "num_events_times": num_event_times,
            "beta": [None] * p,
            "se_robust": [None] * p,
            "hr": [None] * p,
            "ci_lower": [None] * p,
            "ci_upper": [None] * p,
            "warning": "ZERO_EVENTS_IN_REFERENCE_ARM",
        }
    for j in range(p):
        col_pos = Z[:, j] > 0.0
        if np.any(col_pos) and np.all((Z[:, j] == 0.0) | (Z[:, j] == 1.0)):
            if float(np.sum(d_subj[col_pos])) <= 0.0:
                return {
                    "converged": False,
                    "num_events_times": num_event_times,
                    "beta": [None] * p,
                    "se_robust": [None] * p,
                    "hr": [None] * p,
                    "ci_lower": [None] * p,
                    "ci_upper": [None] * p,
                    "warning": f"ZERO_EVENTS_IN_ARM_COLUMN_{j}",
                }

    def _eval_ll_grad_hess(beta_vec: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        eta = np.clip(Z @ beta_vec, -20.0, 20.0)
        r = np.exp(eta)
        ll_val = float(np.sum(d_subj * eta))
        grad_vec = Z.T @ d_subj
        hess_mat = np.zeros((p, p), dtype=np.float64)

        for m0 in range(0, num_event_times, chunk_size):
            m1 = min(num_event_times, m0 + chunk_size)
            d_ch, W_ch = build_chunk_fn(m0, m1)
            D_ch = np.sum(d_ch, axis=0)
            valid = D_ch > 0.0
            if not np.any(valid):
                continue
            if not np.all(valid):
                W_ch = W_ch[:, valid]
                D_ch = D_ch[valid]
            A = W_ch * r[:, None]
            S0 = np.sum(A, axis=0)
            S0_safe = np.maximum(S0, 1e-300)
            ll_val -= float(np.sum(D_ch * np.log(S0_safe)))
            S1 = Z.T @ A  # (p, B)
            Z_bar = (S1 / S0_safe[None, :]).T  # (B, p)
            grad_vec = grad_vec - (Z_bar.T @ D_ch)
            S2 = np.einsum("im,ik,il->mkl", A, Z, Z)
            V_m = (S2 / S0_safe[:, None, None]) - (Z_bar[:, :, None] * Z_bar[:, None, :])
            hess_mat += np.einsum("m,mkl->kl", D_ch, V_m)

        return ll_val, grad_vec, hess_mat

    beta = np.zeros(p, dtype=np.float64)
    ll, grad, hess = _eval_ll_grad_hess(beta)
    converged = False

    for _ in range(max_iter):
        if float(np.max(np.abs(grad))) <= tol:
            converged = True
            break
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(hess + 1e-10 * np.eye(p), grad, rcond=None)

        alpha = 1.0
        accepted = False
        while alpha >= 1e-6:
            cand = beta + alpha * step
            if float(np.max(np.abs(Z @ cand))) > 20.0 and float(np.max(np.abs(Z @ beta))) >= 19.5:
                break
            cand_ll, cand_grad, cand_hess = _eval_ll_grad_hess(cand)
            if cand_ll >= ll - 1e-11:
                beta, ll, grad, hess = cand, cand_ll, cand_grad, cand_hess
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    else:
        if float(np.max(np.abs(grad))) <= 1e-7:
            converged = True

    if float(np.max(np.abs(grad))) <= 1e-7:
        converged = True

    # Subject-clustered Lin-Wei / Geskus sandwich score residuals U_i(beta_hat)
    eta_hat = np.clip(Z @ beta, -20.0, 20.0)
    r_hat = np.exp(eta_hat)
    M_resid = np.zeros(n, dtype=np.float64)
    Q_resid = np.zeros((n, p), dtype=np.float64)

    for m0 in range(0, num_event_times, chunk_size):
        m1 = min(num_event_times, m0 + chunk_size)
        d_ch, W_ch = build_chunk_fn(m0, m1)
        D_ch = np.sum(d_ch, axis=0)
        valid = D_ch > 0.0
        if not np.any(valid):
            continue
        if not np.all(valid):
            d_ch = d_ch[:, valid]
            W_ch = W_ch[:, valid]
            D_ch = D_ch[valid]
        A = W_ch * r_hat[:, None]
        S0_safe = np.maximum(np.sum(A, axis=0), 1e-300)
        Z_bar = ((Z.T @ A) / S0_safe[None, :]).T  # (B, p)
        h_ch = A * (D_ch / S0_safe)[None, :]
        dm_ch = d_ch - h_ch
        M_resid += np.sum(dm_ch, axis=1)
        Q_resid += dm_ch @ Z_bar

    U_subj = M_resid[:, None] * Z - Q_resid  # (n, p), exact sum_i U_subj[i] == grad
    df_scale = float(n) / float(max(n - 1, 1))
    meat = df_scale * (U_subj.T @ U_subj)
    h_inv = np.linalg.pinv(hess, rcond=1e-12)
    cov_rob = h_inv @ meat @ h_inv
    se_rob = np.sqrt(np.maximum(np.diag(cov_rob), 0.0))

    z_crit = float(norm.ppf(0.5 + 0.5 * ci_level))
    hrs = [float(math.exp(beta[j])) for j in range(p)]
    ci_lo = [float(math.exp(beta[j] - z_crit * se_rob[j])) for j in range(p)]
    ci_hi = [float(math.exp(beta[j] + z_crit * se_rob[j])) for j in range(p)]

    return {
        "converged": converged,
        "num_events_times": num_event_times,
        "beta": [float(x) for x in beta],
        "se_robust": [float(x) for x in se_rob],
        "hr": hrs,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "warning": None if converged else "NON_CONVERGED_OR_SEPARATED",
    }


def _compute_covariate_censoring_survival_matrix(
    durations: np.ndarray,
    event_types: np.ndarray,
    arm_idx: np.ndarray,
    num_arms: int,
    X_cov: np.ndarray | None,
    base_weights: np.ndarray,
    eval_times: np.ndarray,
    *,
    admin_end_day: float,
    positivity_floor: float,
) -> np.ndarray:
    """Compute subject-specific censoring survival G_hat(t- | A_i, X_i) of shape (n, len(eval_times))."""
    n = durations.size
    m = eval_times.size
    if m == 0:
        return np.ones((n, 0), dtype=np.float64)

    if X_cov is None or X_cov.shape[1] == 0:
        out = np.ones((n, m), dtype=np.float64)
        for k in range(num_arms):
            mask = arm_idx == k
            if np.any(mask):
                g_k = compute_reverse_km_left_limit(
                    durations[mask], event_types[mask], base_weights[mask], eval_times
                )
                out[mask, :] = np.maximum(g_k[None, :], positivity_floor)
        return out

    # Standardize covariates + arm indicators for censoring proportional-hazards model
    col_sd = np.std(X_cov, axis=0)
    col_mean = np.mean(X_cov, axis=0)
    X_std = (X_cov - col_mean[None, :]) / np.where(col_sd > 1e-12, col_sd, 1.0)[None, :]
    if num_arms >= 2:
        arm_dummies = np.column_stack([(arm_idx == k).astype(np.float64) for k in range(1, num_arms)])
        Z_cens = np.column_stack([arm_dummies, X_std])
    else:
        Z_cens = X_std

    # Random/informative censoring events strictly before administrative_end_day
    cens_event_mask = (event_types == 0) & (durations < float(admin_end_day) - 1e-9)
    c_times = np.unique(durations[cens_event_mask])
    if c_times.size < 2:
        # Fall back cleanly to arm-stratified reverse KM if virtually no pre-administrative censoring
        out = np.ones((n, m), dtype=np.float64)
        for k in range(num_arms):
            mask = arm_idx == k
            if np.any(mask):
                g_k = compute_reverse_km_left_limit(
                    durations[mask], event_types[mask], base_weights[mask], eval_times
                )
                out[mask, :] = np.maximum(g_k[None, :], positivity_floor)
        return out

    def _cens_chunk(m0: int, m1: int) -> tuple[np.ndarray, np.ndarray]:
        ct = c_times[m0:m1]
        d_ch = ((durations[:, None] == ct[None, :]) & cens_event_mask[:, None]) * base_weights[:, None]
        w_ch = (
            (durations[:, None] > ct[None, :])
            | ((durations[:, None] == ct[None, :]) & (event_types[:, None] == 0))
        ) * base_weights[:, None]
        return d_ch, w_ch

    cens_fit = _fit_weighted_counting_process_cox(
        Z_cens,
        int(c_times.size),
        _cens_chunk,
        ci_level=0.95,
    )
    if cens_fit["beta"][0] is None:
        gamma = np.zeros(Z_cens.shape[1], dtype=np.float64)
    else:
        gamma = np.asarray(cens_fit["beta"], dtype=np.float64)

    lin_pred = np.clip(Z_cens @ gamma, -10.0, 10.0)
    risk_c = np.exp(lin_pred)

    # Breslow baseline censoring hazard increments at c_times
    d_lambda0 = np.zeros(c_times.size, dtype=np.float64)
    for idx, c_val in enumerate(c_times):
        d_num = float(np.sum(base_weights[cens_event_mask & (durations == c_val)]))
        in_rc = (durations > c_val) | ((durations == c_val) & (event_types == 0))
        denom = float(np.sum(base_weights[in_rc] * risk_c[in_rc]))
        if denom > 0.0:
            d_lambda0[idx] = d_num / denom

    cum_lambda0 = np.cumsum(d_lambda0)
    pos = np.searchsorted(c_times, eval_times, side="left")
    lam0_eval = np.zeros(m, dtype=np.float64)
    has_prev = pos > 0
    lam0_eval[has_prev] = cum_lambda0[pos[has_prev] - 1]
    g_mat = np.exp(-np.outer(risk_c, lam0_eval))
    return np.maximum(g_mat, positivity_floor)


def _compute_stabilized_joint_ipcw_matrix(
    durations: np.ndarray,
    event_types: np.ndarray,
    arm_idx: np.ndarray,
    num_arms: int,
    X_cov: np.ndarray | None,
    base_weights: np.ndarray,
    eval_times: np.ndarray,
    *,
    censoring_ipcw: str,
    admin_end_day: float,
    positivity_floor: float,
) -> np.ndarray:
    """Return multiplier matrix M(i, m) so joint weight at eval_times[m] is base_weights[i] * M(i, m)."""
    n = durations.size
    m = eval_times.size
    if m == 0 or censoring_ipcw == "none":
        return np.ones((n, m), dtype=np.float64)

    if censoring_ipcw == "arm_stratified":
        # Unstabilized inverse arm-stratified reverse KM 1 / G_a(t-) for sensitivity inspection
        mult = np.ones((n, m), dtype=np.float64)
        for k in range(num_arms):
            mask = arm_idx == k
            if np.any(mask):
                g_k = compute_reverse_km_left_limit(
                    durations[mask], event_types[mask], base_weights[mask], eval_times
                )
                mult[mask, :] = 1.0 / np.maximum(g_k[None, :], positivity_floor)
        return mult

    # censoring_ipcw == "covariate_cox": stabilized G_a(t-) / G(t- | A_i, X_i)
    g_cond = _compute_covariate_censoring_survival_matrix(
        durations,
        event_types,
        arm_idx,
        num_arms,
        X_cov,
        base_weights,
        eval_times,
        admin_end_day=admin_end_day,
        positivity_floor=positivity_floor,
    )
    g_num = np.ones((n, m), dtype=np.float64)
    for k in range(num_arms):
        mask = arm_idx == k
        if np.any(mask):
            g_k = compute_reverse_km_left_limit(
                durations[mask], event_types[mask], base_weights[mask], eval_times
            )
            g_num[mask, :] = np.maximum(g_k[None, :], positivity_floor)
    return g_num / g_cond


def weighted_aalen_johansen_cif(
    durations: np.ndarray,
    event_types: np.ndarray,
    weights: np.ndarray,
    *,
    origin_day: float = 0.0,
    joint_ipcw_multipliers: np.ndarray | None = None,
) -> dict:
    """Compute weighted Aalen-Johansen CIF_1(t), CIF_2(t), overall survival S(t), and biased naive 1-KM_1(t).

    Parameters
    ----------
    durations : (n,) float array of follow-up times (> origin_day)
    event_types : (n,) int array with 0=censored, 1=primary event, 2=competing event
    weights : (n,) positive baseline weights
    origin_day : time zero / landmark anchor day
    joint_ipcw_multipliers : optional (n, M) time-varying censoring multipliers aligned to unique event times

    Returns
    -------
    dict with keys:
      time_day, time_since_landmark_day, weighted_at_risk,
      cif_primary, cif_competing, overall_survival, naive_km_primary_biased_ref
    """
    t_arr = np.asarray(durations, dtype=np.float64)
    e_arr = np.asarray(event_types, dtype=np.int64)
    w_arr = np.asarray(weights, dtype=np.float64)

    if t_arr.size == 0:
        raise ContractError("EMPTY_COHORT_ARM", "Cannot estimate Aalen-Johansen CIF on an empty arm")

    ev_mask = (e_arr == 1) | (e_arr == 2)
    ev_times = np.unique(t_arr[ev_mask])
    m_len = ev_times.size

    if m_len == 0:
        return {
            "time_day": np.array([float(origin_day)], dtype=np.float64),
            "time_since_landmark_day": np.array([0.0], dtype=np.float64),
            "weighted_at_risk": np.array([float(np.sum(w_arr))], dtype=np.float64),
            "cif_primary": np.array([0.0], dtype=np.float64),
            "cif_competing": np.array([0.0], dtype=np.float64),
            "overall_survival": np.array([1.0], dtype=np.float64),
            "naive_km_primary_biased_ref": np.array([0.0], dtype=np.float64),
        }

    cif1 = np.zeros(m_len + 1, dtype=np.float64)
    cif2 = np.zeros(m_len + 1, dtype=np.float64)
    surv = np.ones(m_len + 1, dtype=np.float64)
    naive_surv1 = np.ones(m_len + 1, dtype=np.float64)
    at_risk = np.zeros(m_len + 1, dtype=np.float64)
    at_risk[0] = float(np.sum(w_arr))

    if joint_ipcw_multipliers is None:
        # Fast O(n log n) exact reduction without (n, m_len) allocation
        order = np.argsort(t_arr, kind="mergesort")
        t_sorted = t_arr[order]
        e_sorted = e_arr[order]
        w_sorted = w_arr[order]

        uniq_t, start_idx = np.unique(t_sorted, return_index=True)
        w_tot_u = np.add.reduceat(w_sorted, start_idx)
        w_e1_u = np.add.reduceat(w_sorted * (e_sorted == 1), start_idx)
        w_e2_u = np.add.reduceat(w_sorted * (e_sorted == 2), start_idx)
        y_u = np.cumsum(w_tot_u[::-1])[::-1]

        ev_u_mask = (w_e1_u > 0.0) | (w_e2_u > 0.0)
        y_ev = y_u[ev_u_mask]
        d1_ev = w_e1_u[ev_u_mask]
        d2_ev = w_e2_u[ev_u_mask]

        dh1 = np.where(y_ev > 0.0, d1_ev / y_ev, 0.0)
        dh2 = np.where(y_ev > 0.0, d2_ev / y_ev, 0.0)
        dh_tot = np.clip(dh1 + dh2, 0.0, 1.0)

        surv[1:] = np.cumprod(1.0 - dh_tot)
        s_prev = surv[:-1]
        cif1[1:] = np.cumsum(s_prev * dh1)
        cif2[1:] = np.cumsum(s_prev * dh2)
        naive_surv1[1:] = np.cumprod(1.0 - np.clip(dh1, 0.0, 1.0))
        at_risk[1:] = y_ev
    else:
        for j, tj in enumerate(ev_times):
            w_eff = w_arr * joint_ipcw_multipliers[:, j]
            in_risk = t_arr >= tj
            yj = float(np.sum(w_eff[in_risk]))
            at_risk[j + 1] = yj
            if yj <= 0.0:
                cif1[j + 1] = cif1[j]
                cif2[j + 1] = cif2[j]
                surv[j + 1] = surv[j]
                naive_surv1[j + 1] = naive_surv1[j]
                continue
            d1 = float(np.sum(w_eff[(t_arr == tj) & (e_arr == 1)]))
            d2 = float(np.sum(w_eff[(t_arr == tj) & (e_arr == 2)]))
            dh1_j = d1 / yj
            dh2_j = d2 / yj
            dh_tot_j = min(1.0, max(0.0, dh1_j + dh2_j))
            cif1[j + 1] = cif1[j] + surv[j] * dh1_j
            cif2[j + 1] = cif2[j] + surv[j] * dh2_j
            surv[j + 1] = surv[j] * (1.0 - dh_tot_j)
            naive_surv1[j + 1] = naive_surv1[j] * (1.0 - min(1.0, max(0.0, dh1_j)))

    full_times = np.concatenate([[float(origin_day)], ev_times])
    return {
        "time_day": full_times,
        "time_since_landmark_day": full_times - float(origin_day),
        "weighted_at_risk": at_risk,
        "cif_primary": np.clip(cif1, 0.0, 1.0),
        "cif_competing": np.clip(cif2, 0.0, 1.0),
        "overall_survival": np.clip(surv, 0.0, 1.0),
        "naive_km_primary_biased_ref": np.clip(1.0 - naive_surv1, 0.0, 1.0),
    }


def evaluate_step_curve(curve_times: np.ndarray, curve_values: np.ndarray, horizons: list[float]) -> np.ndarray:
    """Right-continuous step-function evaluation at horizons."""
    h_arr = np.asarray(horizons, dtype=np.float64)
    idx = np.searchsorted(curve_times, h_arr, side="right") - 1
    idx = np.clip(idx, 0, curve_values.size - 1)
    return curve_values[idx]


def fit_weighted_cause_specific_cox(
    durations: np.ndarray,
    event_types: np.ndarray,
    arm_idx: np.ndarray,
    num_arms: int,
    weights: np.ndarray,
    *,
    cause_code: int,
    ci_level: float = 0.95,
    censoring_ipcw: str = "none",
    X_cov: np.ndarray | None = None,
    admin_end_day: float = math.inf,
    positivity_floor: float = 0.01,
) -> dict:
    """Fit weighted cause-specific Cox model for cause_code (1=primary or 2=competing) vs arm 0."""
    t_arr = np.asarray(durations, dtype=np.float64)
    e_arr = np.asarray(event_types, dtype=np.int64)
    a_arr = np.asarray(arm_idx, dtype=np.int64)
    w_arr = np.asarray(weights, dtype=np.float64)

    Z = np.column_stack([(a_arr == k).astype(np.float64) for k in range(1, num_arms)])
    ev_times = np.unique(t_arr[e_arr == cause_code])

    ipcw_mult: np.ndarray | None = None
    if censoring_ipcw != "none" and ev_times.size > 0:
        ipcw_mult = _compute_stabilized_joint_ipcw_matrix(
            t_arr,
            e_arr,
            a_arr,
            num_arms,
            X_cov,
            w_arr,
            ev_times,
            censoring_ipcw=censoring_ipcw,
            admin_end_day=admin_end_day,
            positivity_floor=positivity_floor,
        )

    def _build_chunk(m0: int, m1: int) -> tuple[np.ndarray, np.ndarray]:
        tm = ev_times[m0:m1]
        if ipcw_mult is None:
            w_eff = w_arr[:, None]
        else:
            w_eff = w_arr[:, None] * ipcw_mult[:, m0:m1]
        d_ch = ((t_arr[:, None] == tm[None, :]) & (e_arr[:, None] == cause_code)) * w_eff
        w_ch = (t_arr[:, None] >= tm[None, :]) * w_eff
        return d_ch, w_ch

    return _fit_weighted_counting_process_cox(
        Z,
        int(ev_times.size),
        _build_chunk,
        ci_level=ci_level,
    )


def fit_fine_gray_shr(
    durations: np.ndarray,
    event_types: np.ndarray,
    arm_idx: np.ndarray,
    num_arms: int,
    weights: np.ndarray,
    *,
    ci_level: float = 0.95,
    censoring_ipcw: str = "none",
    X_cov: np.ndarray | None = None,
    admin_end_day: float = math.inf,
    positivity_floor: float = 0.01,
) -> dict:
    """Fit Fine-Gray (1999) / Geskus (2011) subdistribution proportional hazards model for primary event=1.

    Subjects experiencing competing events (event_types == 2) at T_i < t_m remain in the subdistribution
    risk set at t_m with time-varying IPCW weight w_i * G_hat(t_m-) / G_hat(T_i-), where G_hat is estimated
    via reverse Kaplan-Meier (pooled when censoring_ipcw='none', arm-stratified when 'arm_stratified', or
    covariate-adjusted Cox when 'covariate_cox'). Standard errors use subject-clustered Lin-Wei sandwich variance.
    """
    t_arr = np.asarray(durations, dtype=np.float64)
    e_arr = np.asarray(event_types, dtype=np.int64)
    a_arr = np.asarray(arm_idx, dtype=np.int64)
    w_arr = np.asarray(weights, dtype=np.float64)
    n = t_arr.size

    Z = np.column_stack([(a_arr == k).astype(np.float64) for k in range(1, num_arms)])
    ev1_times = np.unique(t_arr[e_arr == 1])
    if ev1_times.size == 0:
        return _fit_weighted_counting_process_cox(Z, 0, lambda a, b: (np.empty((n, 0)), np.empty((n, 0))), ci_level=ci_level)

    comp_mask = e_arr == 2
    comp_indices = np.flatnonzero(comp_mask)

    # Precompute G_hat(t_m-) and G_hat(T_i-) according to censoring model
    if censoring_ipcw == "none":
        g_at_ev1 = compute_reverse_km_left_limit(t_arr, e_arr, w_arr, ev1_times)  # (M,)
        g_at_ti = np.ones(n, dtype=np.float64)
        if comp_indices.size > 0:
            g_at_ti[comp_indices] = np.maximum(
                compute_reverse_km_left_limit(t_arr, e_arr, w_arr, t_arr[comp_indices]),
                1e-15,
            )
        natural_mult = None
    elif censoring_ipcw == "arm_stratified":
        g_ev1_by_arm = np.ones((num_arms, ev1_times.size), dtype=np.float64)
        g_at_ti = np.ones(n, dtype=np.float64)
        for k in range(num_arms):
            mask_k = a_arr == k
            if np.any(mask_k):
                g_ev1_by_arm[k, :] = compute_reverse_km_left_limit(
                    t_arr[mask_k], e_arr[mask_k], w_arr[mask_k], ev1_times
                )
                comp_k = mask_k & comp_mask
                if np.any(comp_k):
                    g_at_ti[comp_k] = np.maximum(
                        compute_reverse_km_left_limit(
                            t_arr[mask_k], e_arr[mask_k], w_arr[mask_k], t_arr[comp_k]
                        ),
                        1e-15,
                    )
        g_at_ev1 = g_ev1_by_arm  # (K, M)
        natural_mult = None
    else:  # covariate_cox
        all_query_times = np.concatenate([ev1_times, t_arr[comp_indices]])
        g_all = _compute_covariate_censoring_survival_matrix(
            t_arr,
            e_arr,
            a_arr,
            num_arms,
            X_cov,
            w_arr,
            all_query_times,
            admin_end_day=admin_end_day,
            positivity_floor=positivity_floor,
        )
        g_at_ev1 = g_all[:, : ev1_times.size]  # (n, M)
        g_at_ti = np.ones(n, dtype=np.float64)
        for idx_pos, subj_i in enumerate(comp_indices):
            g_at_ti[subj_i] = max(float(g_all[subj_i, ev1_times.size + idx_pos]), positivity_floor)
        natural_mult = _compute_stabilized_joint_ipcw_matrix(
            t_arr,
            e_arr,
            a_arr,
            num_arms,
            X_cov,
            w_arr,
            ev1_times,
            censoring_ipcw="covariate_cox",
            admin_end_day=admin_end_day,
            positivity_floor=positivity_floor,
        )

    def _build_fg_chunk(m0: int, m1: int) -> tuple[np.ndarray, np.ndarray]:
        tm = ev1_times[m0:m1]
        if natural_mult is None:
            w_nat = w_arr[:, None]
        else:
            w_nat = w_arr[:, None] * natural_mult[:, m0:m1]

        d_ch = ((t_arr[:, None] == tm[None, :]) & (e_arr[:, None] == 1)) * w_nat
        still_at_risk = (t_arr[:, None] >= tm[None, :]) * w_nat

        comp_prior = (t_arr[:, None] < tm[None, :]) & comp_mask[:, None]
        if not np.any(comp_prior):
            return d_ch, still_at_risk

        if censoring_ipcw == "none":
            ipcw_ratio = g_at_ev1[None, m0:m1] / g_at_ti[:, None]
        elif censoring_ipcw == "arm_stratified":
            ipcw_ratio = g_at_ev1[a_arr, m0:m1] / g_at_ti[:, None]
        else:
            ipcw_ratio = g_at_ev1[:, m0:m1] / g_at_ti[:, None]

        ipcw_ratio = np.clip(ipcw_ratio, 0.0, 1.0)
        # Multiply by w_nat (= w_i * M_i(t_m) under covariate_cox, or w_i otherwise) so subjects
        # remaining in follow-up and subjects retained after prior competing events share identical
        # stabilized censoring weights: w_i * M_i(t_m) * G_i(t_m-)/G_i(T_i-) = w_i * G_{A_i}(t_m-)/G_i(T_i-).
        w_comp = comp_prior * (w_nat * ipcw_ratio)
        return d_ch, still_at_risk + w_comp

    return _fit_weighted_counting_process_cox(
        Z,
        int(ev1_times.size),
        _build_fg_chunk,
        ci_level=ci_level,
    )


def _bootstrap_arm_cif_horizons(
    durations_by_arm: list[np.ndarray],
    events_by_arm: list[np.ndarray],
    weights_by_arm: list[np.ndarray],
    covariates_by_arm: list[np.ndarray | None],
    *,
    horizons: list[float],
    origin_day: float,
    bootstrap_iterations: int,
    seed: int,
    censoring_ipcw: str,
    admin_end_day: float,
    positivity_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subject-level bootstrap returning (boot_cif1, boot_cif2) of shape (B, K, H)."""
    num_arms = len(durations_by_arm)
    num_h = len(horizons)
    h_arr = np.asarray(horizons, dtype=np.float64)
    rng = np.random.default_rng(seed)

    boot_cif1 = np.zeros((bootstrap_iterations, num_arms, num_h), dtype=np.float64)
    boot_cif2 = np.zeros((bootstrap_iterations, num_arms, num_h), dtype=np.float64)

    if censoring_ipcw == "none":
        # Pre-sort each arm once so every bootstrap replicate is O(n_k) without sorting
        prepped = []
        for k in range(num_arms):
            tk = durations_by_arm[k]
            ek = events_by_arm[k]
            wk = weights_by_arm[k]
            order = np.argsort(tk, kind="mergesort")
            ts = tk[order]
            es = ek[order]
            ws = wk[order]
            uniq_t, start_idx = np.unique(ts, return_index=True)
            h_pos = np.searchsorted(uniq_t, h_arr, side="right") - 1
            prepped.append((ts.size, ws, es == 1, es == 2, start_idx, h_pos))

        for b in range(bootstrap_iterations):
            for k in range(num_arms):
                n_k, ws, is_e1, is_e2, start_idx, h_pos = prepped[k]
                draw = rng.integers(0, n_k, size=n_k)
                mult = np.bincount(draw, minlength=n_k).astype(np.float64)
                wb = ws * mult
                w_tot_u = np.add.reduceat(wb, start_idx)
                w_e1_u = np.add.reduceat(wb * is_e1, start_idx)
                w_e2_u = np.add.reduceat(wb * is_e2, start_idx)
                y_u = np.cumsum(w_tot_u[::-1])[::-1]

                dh1 = np.divide(w_e1_u, y_u, out=np.zeros_like(w_e1_u), where=y_u > 0.0)
                dh2 = np.divide(w_e2_u, y_u, out=np.zeros_like(w_e2_u), where=y_u > 0.0)
                dh_tot = np.clip(dh1 + dh2, 0.0, 1.0)
                surv_after = np.cumprod(1.0 - dh_tot)
                s_prev = np.empty_like(surv_after)
                s_prev[0] = 1.0
                if surv_after.size > 1:
                    s_prev[1:] = surv_after[:-1]
                c1_u = np.clip(np.cumsum(s_prev * dh1), 0.0, 1.0)
                c2_u = np.clip(np.cumsum(s_prev * dh2), 0.0, 1.0)

                val1 = np.where(h_pos >= 0, c1_u[np.maximum(h_pos, 0)], 0.0)
                val2 = np.where(h_pos >= 0, c2_u[np.maximum(h_pos, 0)], 0.0)
                boot_cif1[b, k, :] = val1
                boot_cif2[b, k, :] = val2
        return boot_cif1, boot_cif2

    # Joint IPCW sensitivity mode: recompute censoring survival per replicate
    for b in range(bootstrap_iterations):
        t_list = []
        e_list = []
        w_list = []
        a_list = []
        x_list = []
        for k in range(num_arms):
            n_k = durations_by_arm[k].size
            draw = rng.integers(0, n_k, size=n_k)
            t_list.append(durations_by_arm[k][draw])
            e_list.append(events_by_arm[k][draw])
            w_list.append(weights_by_arm[k][draw])
            a_list.append(np.full(n_k, k, dtype=np.int64))
            if covariates_by_arm[k] is not None:
                x_list.append(covariates_by_arm[k][draw])

        t_all = np.concatenate(t_list)
        e_all = np.concatenate(e_list)
        w_all = np.concatenate(w_list)
        a_all = np.concatenate(a_list)
        x_all = np.vstack(x_list) if x_list else None

        for k in range(num_arms):
            mk = a_all == k
            tk = t_all[mk]
            ek = e_all[mk]
            wk = w_all[mk]
            ev_k = np.unique(tk[(ek == 1) | (ek == 2)])
            if ev_k.size == 0:
                continue
            mult_all = _compute_stabilized_joint_ipcw_matrix(
                t_all,
                e_all,
                a_all,
                num_arms,
                x_all,
                w_all,
                ev_k,
                censoring_ipcw=censoring_ipcw,
                admin_end_day=admin_end_day,
                positivity_floor=positivity_floor,
            )
            aj_k = weighted_aalen_johansen_cif(
                tk,
                ek,
                wk,
                origin_day=origin_day,
                joint_ipcw_multipliers=mult_all[mk, :],
            )
            boot_cif1[b, k, :] = evaluate_step_curve(aj_k["time_day"], aj_k["cif_primary"], horizons)
            boot_cif2[b, k, :] = evaluate_step_curve(aj_k["time_day"], aj_k["cif_competing"], horizons)

    return boot_cif1, boot_cif2


def _compute_rr_cif_evalue_entry(
    rr_point: float | None,
    rr_lo: float | None,
    rr_hi: float | None,
) -> dict | None:
    if rr_point is None or not math.isfinite(rr_point) or rr_point <= 0.0:
        return None

    # Handle edge case where percentile bootstrap lower limit hits 0.0 in small samples
    if rr_lo is not None and rr_hi is not None:
        if rr_lo <= 0.0:
            if rr_hi >= 1.0:
                res = compute_evalue(rr_point, measure_type="RR_CIF", has_competing_risks=True)
                res["ci_lower"] = 0.0
                res["ci_upper"] = float(rr_hi)
                res["ci_limit_closer_to_null"] = 1.0
                return res
            if rr_point < 1.0 and rr_hi > 0.0:
                res = compute_evalue(rr_point, ci_upper=rr_hi, measure_type="RR_CIF", has_competing_risks=True)
                res["ci_lower"] = 0.0
                return res
        if rr_lo > 0.0 and rr_hi > 0.0 and rr_lo <= rr_hi:
            return compute_evalue(
                rr_point,
                ci_lower=rr_lo,
                ci_upper=rr_hi,
                measure_type="RR_CIF",
                has_competing_risks=True,
            )
    return compute_evalue(rr_point, measure_type="RR_CIF", has_competing_risks=True)


def _verify_upstream_diagnostics_gate(
    protocol: dict,
    diagnostics_dir: Path,
    weighted_cohort_path: Path,
    *,
    source_mode: str = SOURCE_MODE_SYNTHETIC_ONLY,
    ingest_manifest_path: Path | None = None,
) -> tuple[dict, dict, bytes, pd.DataFrame, dict | None]:
    """Fail-closed verification of upstream diagnose gate, manifest SHA-256s, and protocol alignment."""
    if not diagnostics_dir.is_dir():
        raise GateBlockedError(
            "MISSING_DIAGNOSTICS_DIR",
            f"Diagnostics directory does not exist: {diagnostics_dir}",
        )

    summary_path = diagnostics_dir / "diagnostics_summary.json"
    manifest_path = diagnostics_dir / "run_manifest.json"
    if not summary_path.is_file():
        raise GateBlockedError(
            "MISSING_DIAGNOSTICS_SUMMARY",
            f"Missing diagnostics_summary.json in {diagnostics_dir}; refusing unverified estimation.",
        )
    if not manifest_path.is_file():
        raise GateBlockedError(
            "MISSING_DIAGNOSTICS_MANIFEST",
            f"Missing run_manifest.json in {diagnostics_dir}; refusing unverified estimation.",
        )

    try:
        _, diag_summary = read_strict_json(summary_path, max_bytes=512 * 1024)
        raw_diag_manifest, diag_manifest = read_strict_json(manifest_path, max_bytes=512 * 1024)
    except ContractError as exc:
        raise GateBlockedError(
            "CORRUPT_DIAGNOSTICS_METADATA",
            f"Upstream diagnostics metadata invalid ({exc.code}): {exc}",
        ) from exc

    output_hashes = diag_manifest.get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise GateBlockedError(
            "MANIFEST_INTEGRITY_MISMATCH",
            "Upstream run_manifest.json is missing 'output_sha256' hash dictionary.",
        )

    expected_summary_hash = output_hashes.get("diagnostics_summary.json")
    if not expected_summary_hash or sha256_file(summary_path) != expected_summary_hash:
        raise GateBlockedError(
            "MANIFEST_INTEGRITY_MISMATCH",
            "diagnostics_summary.json SHA-256 does not match upstream run_manifest.json.",
        )

    gate_status = diag_summary.get("gate_status")
    manifest_gate_status = diag_manifest.get("gate_status")
    if gate_status != "passed" or manifest_gate_status != "passed":
        upstream_failures = diag_summary.get("gate_failures")
        raise GateBlockedError(
            "UPSTREAM_GATE_NOT_PASSED",
            f"Upstream diagnostic gate_status='{gate_status}' (manifest='{manifest_gate_status}'); "
            "fail-closed competing-risks estimation refused.",
            gate_failures=upstream_failures if isinstance(upstream_failures, list) and upstream_failures else None,
        )

    if not weighted_cohort_path.is_file():
        raise GateBlockedError(
            "MISSING_WEIGHTED_COHORT",
            f"Required gated cohort '{weighted_cohort_path}' is missing despite gate check request.",
        )

    expected_wc_hash = output_hashes.get("weighted_cohort.csv")
    if not expected_wc_hash:
        raise GateBlockedError(
            "MANIFEST_INTEGRITY_MISMATCH",
            "Upstream run_manifest.json does not certify 'weighted_cohort.csv'.",
        )
    actual_wc_hash = sha256_file(weighted_cohort_path)
    if actual_wc_hash != expected_wc_hash:
        raise GateBlockedError(
            "MANIFEST_INTEGRITY_MISMATCH",
            f"weighted_cohort.csv SHA-256 ({actual_wc_hash}) mismatches upstream run_manifest.json ({expected_wc_hash}).",
        )

    # Protocol alignment checks
    norm_proto_hash = sha256_bytes(json_bytes(protocol))
    if "normalized_protocol_sha256" in diag_manifest:
        if diag_manifest["normalized_protocol_sha256"] != norm_proto_hash:
            raise GateBlockedError(
                "PROTOCOL_DIAGNOSTICS_MISMATCH",
                "Normalized protocol SHA-256 does not match upstream diagnostics run_manifest.json.",
            )

    arm_ids = [s["id"] for s in protocol["treatment_strategies"]]
    retained_counts = diag_summary.get("retained_arm_counts")
    if (
        diag_summary.get("simulation_or_data_label") != protocol["simulation_or_data_label"]
        or diag_summary.get("landmark_day") != protocol["time_zero"]["landmark_day"]
        or diag_summary.get("protocol_diagnostic_gates") != protocol["estimand"]["diagnostic_gates"]
        or not isinstance(retained_counts, dict)
        or set(retained_counts.keys()) != set(arm_ids)
    ):
        raise GateBlockedError(
            "PROTOCOL_DIAGNOSTICS_MISMATCH",
            "Protocol strategies, landmark_day, simulation_or_data_label, or diagnostic_gates do not match "
            "upstream diagnostics_summary.json.",
        )

    weight_method = diag_summary.get("weight_method")
    if weight_method not in ALLOWED_WEIGHT_METHODS:
        raise GateBlockedError(
            "INVALID_DIAGNOSTICS_WEIGHT_METHOD",
            f"Upstream diagnostics_summary.json carries unsupported weight_method='{weight_method}'.",
        )

    try:
        wdf = pd.read_csv(weighted_cohort_path)
    except Exception as exc:
        raise GateBlockedError(
            "CORRUPT_WEIGHTED_COHORT",
            f"Failed to read weighted_cohort.csv: {exc}",
        ) from exc

    if len(wdf) != int(diag_summary.get("retained_n", -1)) or len(wdf) == 0:
        raise GateBlockedError(
            "WEIGHTED_COHORT_ROW_MISMATCH",
            f"weighted_cohort.csv row count ({len(wdf)}) mismatches diagnostics retained_n ({diag_summary.get('retained_n')}).",
        )

    ingest_manifest: dict | None = None
    if source_mode == SOURCE_MODE_SYNTHETIC_ONLY:
        if "synthetic_case_id" not in wdf.columns:
            raise ContractError(
                "NON_SYNTHETIC_COHORT_REJECTED",
                "weighted_cohort.csv must contain 'synthetic_case_id' with 'SYN-...' identifiers under source_mode='synthetic_only'.",
            )
        case_ids = wdf["synthetic_case_id"].astype(str)
        if not bool(case_ids.str.match(r"^SYN-[0-9]+$").all()):
            raise ContractError(
                "NON_SYNTHETIC_COHORT_REJECTED",
                "All synthetic_case_id entries must match '^SYN-[0-9]+$' under source_mode='synthetic_only'.",
            )
    elif source_mode == SOURCE_MODE_SIMULATED_COHORT:
        id_col_sim = "synthetic_case_id" if "synthetic_case_id" in wdf.columns else ("subject_id" if "subject_id" in wdf.columns else None)
        if id_col_sim is None:
            raise ContractError(
                "INVALID_SIMULATED_COHORT",
                "weighted_cohort.csv must contain 'synthetic_case_id' or 'subject_id' under source_mode='simulated_cohort'.",
            )
        case_ids = wdf[id_col_sim].astype(str)
        if not bool(case_ids.str.match(r"^(?:SIM|SYN)-[0-9]+$").all()):
            raise ContractError(
                "INVALID_SIMULATED_COHORT",
                "All identifiers must match '^(SIM|SYN)-[0-9]+$' under source_mode='simulated_cohort'.",
            )
    elif source_mode == SOURCE_MODE_USER_COHORT:
        if "subject_id" not in wdf.columns and "synthetic_case_id" not in wdf.columns:
            raise ContractError("INVALID_COHORT_SCHEMA", "weighted_cohort.csv must include 'subject_id' under source_mode='user_cohort'.")
        cand_paths = [
            p
            for p in (
                ingest_manifest_path,
                diagnostics_dir / "ingest_manifest.json",
                diagnostics_dir.parent / "ingest" / "ingest_manifest.json",
                diagnostics_dir.parent / "01_ingest" / "ingest_manifest.json",
            )
            if p is not None
        ]
        found_manifest_path = next((p for p in cand_paths if p.is_file()), None)
        if found_manifest_path is None:
            raise ContractError(
                "MISSING_USER_COHORT_INGEST_MANIFEST",
                "source_mode='user_cohort' requires a verified cohort_ingest ingest_manifest.json.",
            )
        _, ingest_manifest = read_strict_json(found_manifest_path, max_bytes=512 * 1024, io_error_code="INGEST_MANIFEST_IO_ERROR")
        if ingest_manifest.get("source_mode") != SOURCE_MODE_USER_COHORT:
            raise ContractError("INVALID_INGEST_MANIFEST", "ingest_manifest.json must declare source_mode='user_cohort'.")
        input_sha = ingest_manifest.get("input_sha256", "")
        std_sha = ingest_manifest.get("standardized_cohort_sha256", "")
        if not isinstance(input_sha, str) or len(input_sha) != 64:
            raise ContractError("INVALID_INGEST_MANIFEST", "ingest_manifest.json missing valid 64-hex input_sha256.")
        if std_sha != diag_manifest.get("cohort_sha256"):
            raise GateBlockedError(
                "INGEST_LINEAGE_MISMATCH",
                "standardized_cohort_sha256 in ingest_manifest.json does not match diagnostics cohort_sha256.",
            )

    for col in ("treatment_arm", "duration_day", "event_code"):
        if col not in wdf.columns:
            raise GateBlockedError("CORRUPT_WEIGHTED_COHORT", f"weighted_cohort.csv missing '{col}'")

    if set(wdf["treatment_arm"].astype(str).unique()) != set(arm_ids):
        raise GateBlockedError(
            "WEIGHTED_COHORT_ARM_MISMATCH",
            "weighted_cohort.csv does not contain observations in every protocol treatment arm.",
        )

    return diag_summary, diag_manifest, raw_diag_manifest, wdf, ingest_manifest


def _write_curve_csv(path: Path, arm_id: str, curve: dict, *, pair_id: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "treatment_arm",
        *(["pair_id"] if pair_id is not None else []),
        "time_day",
        "time_since_landmark_day",
        "weighted_at_risk",
        "cif_primary",
        "cif_competing",
        "overall_survival",
        "naive_km_primary_biased_ref",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        n_rows = curve["time_day"].size
        for i in range(n_rows):
            row = {
                "treatment_arm": arm_id,
                "time_day": f"{curve['time_day'][i]:.8f}",
                "time_since_landmark_day": f"{curve['time_since_landmark_day'][i]:.8f}",
                "weighted_at_risk": f"{curve['weighted_at_risk'][i]:.8f}",
                "cif_primary": f"{curve['cif_primary'][i]:.10f}",
                "cif_competing": f"{curve['cif_competing'][i]:.10f}",
                "overall_survival": f"{curve['overall_survival'][i]:.10f}",
                "naive_km_primary_biased_ref": f"{curve['naive_km_primary_biased_ref'][i]:.10f}",
            }
            if pair_id is not None:
                row["pair_id"] = pair_id
            writer.writerow(row)


def _fmt_est_ci(point: float | None, lo: float | None, hi: float | None, digits: int = 4) -> str:
    if point is None or not math.isfinite(point):
        return "NA"
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} ({lo:.{digits}f}, {hi:.{digits}f})"


def _render_audit_report_md(estimates: dict) -> str:
    smode = estimates["source_mode"]
    if smode == SOURCE_MODE_USER_COHORT:
        heading = "# 🟢 真实队列目标试验模拟（真项目）分析报告\n\n## 1. 项目模式、来源留痕与观察性因果识别前提 (User Cohort Provenance & Causal Assumptions)"
    elif smode == SOURCE_MODE_SIMULATED_COHORT:
        heading = "# 🟣 目标试验模拟（无数据模拟预演项目）可行性报告\n\n## 1. 模拟预演声明与对标试验边界 (Simulation Rehearsal Mode & Prohibited Claims)"
    else:
        heading = "# EmulatRx TTE Competing-Risks Survival Audit Report\n\n## 1. 数据来源与禁用声明锁 (Source Mode & Prohibited Claims)"

    lines = [
        heading,
        "",
        f"- **Source Mode (`source_mode`)**: `{smode}`",
        f"- **Study Title**: {estimates['study_title']}",
        f"- **Data / Simulation Label**: {estimates['simulation_or_data_label']}",
        f"- **Reference Strategy (`treatment_strategies[0]`)**: `{estimates['reference_arm']['id']}` ({estimates['reference_arm']['label']})",
        f"- **Weighting Method**: `{estimates['weight_method']}`",
        f"- **Censoring IPCW Mode**: `{estimates['censoring_model']['censoring_ipcw_mode']}`",
        f"- **Bootstrap Configuration**: `seed={estimates['bootstrap_config']['seed']}`, "
        f"`iterations={estimates['bootstrap_config']['bootstrap_iterations']}`, "
        f"`ci_level={estimates['bootstrap_config']['ci_level']}`, "
        f"`baseline_propensity_weights={estimates['bootstrap_config']['baseline_propensity_weights']}`, "
        f"`censoring_ipcw_reestimated_in_bootstrap={estimates['bootstrap_config']['censoring_ipcw_reestimated_in_bootstrap']}`",
    ]
    if "target_rct" in estimates and isinstance(estimates["target_rct"], dict):
        trct = estimates["target_rct"]
        lines.append(
            f"- **Target RCT Benchmark**: `{trct.get('nct_id', '')}` — {trct.get('brief_title', '')} "
            f"(`source={trct.get('source', 'n/a')}`)"
        )
    if smode == SOURCE_MODE_USER_COHORT and isinstance(estimates.get("ingest_lineage"), dict):
        lin = estimates["ingest_lineage"]
        lines.append(
            f"- **Uploaded Input File & SHA-256**: `{lin.get('input_filename', 'cohort.csv')}` "
            f"(`sha256={lin.get('input_sha256', '')}`)"
        )
        if lin.get("zero_competing_events_notice"):
            lines.append(f"- **Competing Event Status**: {lin['zero_competing_events_notice']}")
    lines.append("- **Prohibited Claims / Boundary Statements (`prohibited_claims`)**:")
    for claim in estimates["prohibited_claims"]:
        lines.append(f"  - {claim}")
    if smode == SOURCE_MODE_USER_COHORT and estimates.get("causal_assumptions"):
        lines.append("- **观察性因果推断四大核心前提 (`causal_assumptions`)**:")
        for assumption in estimates["causal_assumptions"]:
            lines.append(f"  - {assumption}")

    lines.extend(
        [
            "",
            "## 2. 竞争风险核心效应强制并列对照表 (Mandatory Co-Presented Competing-Risks Table)",
            "",
            "| Contrast (Active vs Ref) | Horizon (d) | CIF_1 Ref | CIF_1 Active | RD_1 (CI) | RR_CIF_1 (CI) | CIF_2 Ref | CIF_2 Active | RD_2 (CI) | Fine-Gray sHR (CI) | Primary csHR_1 (CI) | Competing csHR_2 (CI) | E-value RR_CIF (Point / CI) |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
        ]
    )

    for row in estimates["co_presented_comparison_table"]:
        ev_pt = f"{row['evalue_rr_cif_point']:.4f}" if row["evalue_rr_cif_point"] is not None else "NA"
        ev_ci = f"{row['evalue_rr_cif_ci_limit']:.4f}" if row["evalue_rr_cif_ci_limit"] is not None else "NA"
        lines.append(
            f"| `{row['contrast_id']}` | {row['horizon_day']:g} | "
            f"{row['cif_primary_ref']:.4f} | {row['cif_primary_active']:.4f} | "
            f"{_fmt_est_ci(row['rd_primary'], row['rd_primary_ci_lower'], row['rd_primary_ci_upper'])} | "
            f"{_fmt_est_ci(row['rr_cif_primary'], row['rr_cif_primary_ci_lower'], row['rr_cif_primary_ci_upper'])} | "
            f"{row['cif_competing_ref']:.4f} | {row['cif_competing_active']:.4f} | "
            f"{_fmt_est_ci(row['rd_competing'], row['rd_competing_ci_lower'], row['rd_competing_ci_upper'])} | "
            f"{_fmt_est_ci(row['shr_primary'], row['shr_primary_ci_lower'], row['shr_primary_ci_upper'])} | "
            f"{_fmt_est_ci(row['cshr_primary'], row['cshr_primary_ci_lower'], row['cshr_primary_ci_upper'])} | "
            f"{_fmt_est_ci(row['cshr_competing'], row['cshr_competing_ci_lower'], row['cshr_competing_ci_upper'])} | "
            f"{ev_pt} / {ev_ci} |"
        )

    mw = estimates["methodological_warnings"]
    lines.extend(
        [
            "",
            "## 3. 方法学警示与解读红线 (Mandatory Methodological Warnings)",
            "",
            f"1. **亚分布风险与高竞争死亡防误读警示 (sHR vs csHR)**: {mw['subdistribution_vs_cause_specific_warning']}",
            f"2. **风险比固有选择偏倚警示 (Hernán 2010)**: {mw['hazard_ratio_selection_bias_warning']}",
            f"3. **删失假设与联合 IPCW 敏感性说明**: {estimates['censoring_model']['independence_assumption']}",
            f"4. **Bootstrap 置信区间条件范围 (`baseline_propensity_weights=fixed`)**: {estimates['bootstrap_config']['variance_scope_note']}",
            f"5. **E-value 敏感性参考边界**: 仅基于指定时点累计发生率比 `RR_CIF(t)` 计算累计发生总效应敏感性参考值；显式拒绝直接对 `sHR` 套用单事件公式，且拒绝在竞争风险累计发生率 `> 0.15` 时对 `HR`/`csHR` 套用单事件转换。{mw['observable_rule_disclaimer']}",
        ]
    )
    if mw["competing_mortality_divergence_alert"]["triggered"]:
        lines.append(
            f"6. **竞争死亡分歧自动告警 (`competing_mortality_divergence_alert`)**: "
            f"{mw['competing_mortality_divergence_alert']['detail']}"
        )
    if estimates["transitivity_warning"]:
        lines.append(f"7. **多臂成对加权非传递性警示**: {estimates['transitivity_warning']}")

    lines.append("")
    return "\n".join(lines)


def estimate_competing_risks(
    protocol_path: Path,
    *,
    diagnostics_dir: Path | None = None,
    weighted_cohort_path: Path | None = None,
    seed: int,
    bootstrap_iterations: int,
    out_dir: Path,
    horizons_days: str | list[float] | tuple[float, ...] | None = None,
    censoring_ipcw: str = "none",
    ci_level: float = 0.95,
    source_mode: str = SOURCE_MODE_SYNTHETIC_ONLY,
    ingest_manifest_path: Path | None = None,
) -> dict:
    """Validate upstream diagnostics gate and run weighted competing-risks survival analysis.

    Raises ContractError (exit code 2) on malformed CLI/protocol inputs or existing destination,
    and GateBlockedError (exit code 3) if upstream diagnostics gate did not pass or weighted_cohort.csv
    is missing/tampered/mismatched.
    """
    check_destination(out_dir)

    if not isinstance(source_mode, str) or source_mode not in ALLOWED_SOURCE_MODES:
        raise ContractError(
            "UNSUPPORTED_SOURCE_MODE",
            f"source_mode must be one of {sorted(ALLOWED_SOURCE_MODES)} (got '{source_mode}').",
        )
    if not is_strict_int(seed) or not (0 <= seed <= MAX_SEED):
        raise ContractError("INVALID_SEED", f"seed must be an integer in 0..{MAX_SEED}")
    if not is_strict_int(bootstrap_iterations) or not (1 <= bootstrap_iterations <= MAX_BOOTSTRAP_ITERATIONS):
        raise ContractError(
            "INVALID_BOOTSTRAP_ITERATIONS",
            f"bootstrap_iterations must be an integer in 1..{MAX_BOOTSTRAP_ITERATIONS}",
        )
    if not is_strict_number(ci_level) or not (0.0 < float(ci_level) < 1.0):
        raise ContractError("INVALID_CI_LEVEL", "ci_level must be strictly between 0.0 and 1.0")
    ci_level_f = float(ci_level)

    if not isinstance(censoring_ipcw, str) or censoring_ipcw not in ALLOWED_CENSORING_IPCW:
        raise ContractError(
            "INVALID_CENSORING_IPCW",
            f"censoring_ipcw must be one of {sorted(ALLOWED_CENSORING_IPCW)}",
        )

    raw_proto, protocol = load_and_validate_protocol(protocol_path)
    competing_codes = protocol["outcomes"]["competing_event_codes"]
    if (
        protocol["outcomes"]["competing_event_handling"] != "explicit_cif_and_subdistribution"
        or len(competing_codes) == 0
    ):
        raise ContractError(
            "COMPETING_RISKS_REQUIRED",
            "emulatrx_tte estimate requires outcomes.competing_event_handling="
            "'explicit_cif_and_subdistribution' and non-empty competing_event_codes.",
        )

    landmark_day = float(protocol["time_zero"]["landmark_day"])
    admin_end_day = float(protocol["followup"]["administrative_end_day"])
    horizons = parse_horizons_arg(
        horizons_days,
        default_horizons=protocol["followup"]["horizon_days"],
        landmark_day=landmark_day,
        admin_end_day=admin_end_day,
    )

    if diagnostics_dir is None and weighted_cohort_path is None:
        raise ContractError(
            "MISSING_DIAGNOSTICS_INPUT",
            "Provide --diagnostics-dir (or --weighted-cohort inside a passed diagnostics directory).",
        )
    if diagnostics_dir is not None and weighted_cohort_path is not None:
        expected_wc = diagnostics_dir / "weighted_cohort.csv"
        if weighted_cohort_path.resolve() != expected_wc.resolve():
            raise ContractError(
                "DIAGNOSTICS_PATH_CONFLICT",
                f"--weighted-cohort ({weighted_cohort_path}) must reside inside --diagnostics-dir ({diagnostics_dir}).",
            )
        resolved_diag_dir = diagnostics_dir
        resolved_wc_path = weighted_cohort_path
    elif diagnostics_dir is not None:
        resolved_diag_dir = diagnostics_dir
        resolved_wc_path = diagnostics_dir / "weighted_cohort.csv"
    else:
        assert weighted_cohort_path is not None
        resolved_wc_path = weighted_cohort_path
        resolved_diag_dir = weighted_cohort_path.parent

    # Fail-closed upstream gate & SHA-256 verification (raises GateBlockedError -> exit code 3)
    diag_summary, diag_manifest, raw_diag_manifest, wdf, ingest_manifest = _verify_upstream_diagnostics_gate(
        protocol,
        resolved_diag_dir,
        resolved_wc_path,
        source_mode=source_mode,
        ingest_manifest_path=ingest_manifest_path,
    )

    strategies = protocol["treatment_strategies"]
    arm_ids = [s["id"] for s in strategies]
    arm_to_idx = {aid: idx for idx, aid in enumerate(arm_ids)}
    num_arms = len(strategies)
    weight_method = diag_summary["weight_method"]
    cov_names = list(diag_summary.get("covariates") or [])
    positivity_floor = max(float(protocol["estimand"]["diagnostic_gates"]["positivity_epsilon"]), 1e-3)

    durations_all = pd.to_numeric(wdf["duration_day"], errors="raise").to_numpy(dtype=np.float64)
    raw_events = pd.to_numeric(wdf["event_code"], errors="raise").to_numpy(dtype=np.int64)
    comp_set = set(competing_codes)
    event_types_all = np.zeros_like(raw_events)
    event_types_all[raw_events == 1] = 1
    for cc in comp_set:
        event_types_all[raw_events == cc] = 2
    arm_idx_all = wdf["treatment_arm"].astype(str).map(arm_to_idx).to_numpy(dtype=np.int64)
    X_all = wdf[cov_names].to_numpy(dtype=np.float64) if cov_names else None

    alpha = 1.0 - ci_level_f
    q_lo = 0.5 * alpha
    q_hi = 1.0 - 0.5 * alpha

    curve_specs: list[tuple[str, str, dict, str | None]] = []  # (rel_path, arm_id, curve_dict, pair_id)
    arm_cif_summaries: dict = {}
    contrasts_dict: dict = {}
    co_table_rows: list[dict] = []
    divergence_messages: list[str] = []

    if weight_method != "pairwise_overlap":
        if "normalized_weight" in wdf.columns:
            w_norm_col = pd.to_numeric(wdf["normalized_weight"], errors="raise").to_numpy(dtype=np.float64)
            if np.any(~np.isfinite(w_norm_col)) or np.any(w_norm_col <= 0.0):
                raise GateBlockedError(
                    "CORRUPT_WEIGHTED_COHORT", "Baseline normalized_weight must be finite and strictly positive"
                )
            w_base_all = np.zeros_like(w_norm_col)
            arm_target_mass = float(len(wdf)) / float(num_arms)
            for k in range(num_arms):
                mk = arm_idx_all == k
                w_base_all[mk] = arm_target_mass * (w_norm_col[mk] / float(np.sum(w_norm_col[mk])))
        elif "raw_weight" in wdf.columns:
            w_raw_col = pd.to_numeric(wdf["raw_weight"], errors="raise").to_numpy(dtype=np.float64)
            if np.any(~np.isfinite(w_raw_col)) or np.any(w_raw_col <= 0.0):
                raise GateBlockedError("CORRUPT_WEIGHTED_COHORT", "Baseline raw_weight must be finite and strictly positive")
            w_base_all = np.zeros_like(w_raw_col)
            arm_target_mass = float(len(wdf)) / float(num_arms)
            for k in range(num_arms):
                mk = arm_idx_all == k
                w_base_all[mk] = arm_target_mass * (w_raw_col[mk] / float(np.sum(w_raw_col[mk])))
        else:
            raise GateBlockedError(
                "CORRUPT_WEIGHTED_COHORT", "weighted_cohort.csv missing 'normalized_weight'/'raw_weight'"
            )

        if np.any(~np.isfinite(w_base_all)) or np.any(w_base_all <= 0.0):
            raise GateBlockedError("CORRUPT_WEIGHTED_COHORT", "Baseline weights must be finite and strictly positive")

        # 1. Per-arm weighted Aalen-Johansen CIF curves
        arm_curves: list[dict] = []
        t_by_arm: list[np.ndarray] = []
        e_by_arm: list[np.ndarray] = []
        w_by_arm: list[np.ndarray] = []
        x_by_arm: list[np.ndarray | None] = []

        for k, aid in enumerate(arm_ids):
            mk = arm_idx_all == k
            tk = durations_all[mk]
            ek = event_types_all[mk]
            wk = w_base_all[mk]
            xk = X_all[mk] if X_all is not None else None
            t_by_arm.append(tk)
            e_by_arm.append(ek)
            w_by_arm.append(wk)
            x_by_arm.append(xk)

            ipcw_mult_k = None
            if censoring_ipcw != "none":
                ev_k = np.unique(tk[(ek == 1) | (ek == 2)])
                if ev_k.size > 0:
                    mult_full = _compute_stabilized_joint_ipcw_matrix(
                        durations_all,
                        event_types_all,
                        arm_idx_all,
                        num_arms,
                        X_all,
                        w_base_all,
                        ev_k,
                        censoring_ipcw=censoring_ipcw,
                        admin_end_day=admin_end_day,
                        positivity_floor=positivity_floor,
                    )
                    ipcw_mult_k = mult_full[mk, :]

            curve_k = weighted_aalen_johansen_cif(
                tk,
                ek,
                wk,
                origin_day=landmark_day,
                joint_ipcw_multipliers=ipcw_mult_k,
            )
            arm_curves.append(curve_k)
            rel_curve = f"curves/cif_{aid}.csv"
            curve_specs.append((rel_curve, aid, curve_k, None))

        # 2. Deterministic subject-level bootstrap for CIF, RD(t), RR_CIF(t)
        boot_cif1, boot_cif2 = _bootstrap_arm_cif_horizons(
            t_by_arm,
            e_by_arm,
            w_by_arm,
            x_by_arm,
            horizons=horizons,
            origin_day=landmark_day,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            censoring_ipcw=censoring_ipcw,
            admin_end_day=admin_end_day,
            positivity_floor=positivity_floor,
        )

        # Summarize per-arm horizon CIFs
        cif1_pts = np.zeros((num_arms, len(horizons)), dtype=np.float64)
        cif2_pts = np.zeros((num_arms, len(horizons)), dtype=np.float64)
        naive_km_pts = np.zeros((num_arms, len(horizons)), dtype=np.float64)
        for k, aid in enumerate(arm_ids):
            cif1_pts[k, :] = evaluate_step_curve(arm_curves[k]["time_day"], arm_curves[k]["cif_primary"], horizons)
            cif2_pts[k, :] = evaluate_step_curve(arm_curves[k]["time_day"], arm_curves[k]["cif_competing"], horizons)
            naive_km_pts[k, :] = evaluate_step_curve(
                arm_curves[k]["time_day"], arm_curves[k]["naive_km_primary_biased_ref"], horizons
            )
            h_entries = {}
            for h_idx, h_val in enumerate(horizons):
                c1_lo, c1_hi = np.quantile(boot_cif1[:, k, h_idx], [q_lo, q_hi])
                c2_lo, c2_hi = np.quantile(boot_cif2[:, k, h_idx], [q_lo, q_hi])
                h_entries[f"{h_val:g}"] = {
                    "horizon_day": h_val,
                    "cif_primary": round(float(cif1_pts[k, h_idx]), 10),
                    "cif_primary_ci_lower": round(float(c1_lo), 10),
                    "cif_primary_ci_upper": round(float(c1_hi), 10),
                    "cif_competing": round(float(cif2_pts[k, h_idx]), 10),
                    "cif_competing_ci_lower": round(float(c2_lo), 10),
                    "cif_competing_ci_upper": round(float(c2_hi), 10),
                    "naive_km_primary_biased_ref": round(float(naive_km_pts[k, h_idx]), 10),
                }
            arm_cif_summaries[aid] = {
                "role": strategies[k]["role"],
                "label": strategies[k]["label"],
                "n_subjects": int(t_by_arm[k].size),
                "curve_csv": f"curves/cif_{aid}.csv",
                "horizons": h_entries,
            }

        # 3. Joint Fine-Gray sHR + Cause-Specific Cox csHR_1 & csHR_2
        fg_fit = fit_fine_gray_shr(
            durations_all,
            event_types_all,
            arm_idx_all,
            num_arms,
            w_base_all,
            ci_level=ci_level_f,
            censoring_ipcw=censoring_ipcw,
            X_cov=X_all,
            admin_end_day=admin_end_day,
            positivity_floor=positivity_floor,
        )
        cox1_fit = fit_weighted_cause_specific_cox(
            durations_all,
            event_types_all,
            arm_idx_all,
            num_arms,
            w_base_all,
            cause_code=1,
            ci_level=ci_level_f,
            censoring_ipcw=censoring_ipcw,
            X_cov=X_all,
            admin_end_day=admin_end_day,
            positivity_floor=positivity_floor,
        )
        cox2_fit = fit_weighted_cause_specific_cox(
            durations_all,
            event_types_all,
            arm_idx_all,
            num_arms,
            w_base_all,
            cause_code=2,
            ci_level=ci_level_f,
            censoring_ipcw=censoring_ipcw,
            X_cov=X_all,
            admin_end_day=admin_end_day,
            positivity_floor=positivity_floor,
        )

        ref_id = arm_ids[0]
        for k in range(1, num_arms):
            act_id = arm_ids[k]
            contrast_id = f"{act_id}__vs__{ref_id}"
            j_col = k - 1

            shr_pt = fg_fit["hr"][j_col]
            shr_lo = fg_fit["ci_lower"][j_col]
            shr_hi = fg_fit["ci_upper"][j_col]
            cshr1_pt = cox1_fit["hr"][j_col]
            cshr1_lo = cox1_fit["ci_lower"][j_col]
            cshr1_hi = cox1_fit["ci_upper"][j_col]
            cshr2_pt = cox2_fit["hr"][j_col]
            cshr2_lo = cox2_fit["ci_lower"][j_col]
            cshr2_hi = cox2_fit["ci_upper"][j_col]

            if (
                shr_pt is not None
                and cshr1_pt is not None
                and cshr2_pt is not None
                and shr_pt < 0.95
                and cshr1_pt >= 0.90
                and cshr2_pt > 1.25
            ):
                divergence_messages.append(
                    f"Contrast '{contrast_id}': Fine-Gray sHR={shr_pt:.4f} < 1 while primary csHR_1={cshr1_pt:.4f} "
                    f"is near/above null and competing csHR_2={cshr2_pt:.4f} > 1. Lower primary cumulative incidence "
                    "is driven by elevated competing mortality removing subjects early, NOT direct primary protection."
                )

            horizon_contrasts = {}
            for h_idx, h_val in enumerate(horizons):
                c1_ref = float(cif1_pts[0, h_idx])
                c1_act = float(cif1_pts[k, h_idx])
                c2_ref = float(cif2_pts[0, h_idx])
                c2_act = float(cif2_pts[k, h_idx])

                rd1 = c1_act - c1_ref
                rd1_boot = boot_cif1[:, k, h_idx] - boot_cif1[:, 0, h_idx]
                rd1_lo, rd1_hi = np.quantile(rd1_boot, [q_lo, q_hi])

                rd2 = c2_act - c2_ref
                rd2_boot = boot_cif2[:, k, h_idx] - boot_cif2[:, 0, h_idx]
                rd2_lo, rd2_hi = np.quantile(rd2_boot, [q_lo, q_hi])

                rr1: float | None = (c1_act / c1_ref) if c1_ref > 0.0 else None
                rr1_lo: float | None = None
                rr1_hi: float | None = None
                valid_rr = boot_cif1[:, 0, h_idx] > 0.0
                if rr1 is not None and np.any(valid_rr):
                    rr_boot = boot_cif1[valid_rr, k, h_idx] / boot_cif1[valid_rr, 0, h_idx]
                    q_vals = np.quantile(rr_boot, [q_lo, q_hi])
                    rr1_lo = float(q_vals[0])
                    rr1_hi = float(q_vals[1])

                ev_entry = _compute_rr_cif_evalue_entry(rr1, rr1_lo, rr1_hi)

                horizon_contrasts[f"{h_val:g}"] = {
                    "horizon_day": h_val,
                    "cif_primary_ref": round(c1_ref, 10),
                    "cif_primary_active": round(c1_act, 10),
                    "rd_primary": round(rd1, 10),
                    "rd_primary_ci_lower": round(float(rd1_lo), 10),
                    "rd_primary_ci_upper": round(float(rd1_hi), 10),
                    "rr_cif_primary": round(rr1, 10) if rr1 is not None else None,
                    "rr_cif_primary_ci_lower": round(rr1_lo, 10) if rr1_lo is not None else None,
                    "rr_cif_primary_ci_upper": round(rr1_hi, 10) if rr1_hi is not None else None,
                    "cif_competing_ref": round(c2_ref, 10),
                    "cif_competing_active": round(c2_act, 10),
                    "rd_competing": round(rd2, 10),
                    "rd_competing_ci_lower": round(float(rd2_lo), 10),
                    "rd_competing_ci_upper": round(float(rd2_hi), 10),
                    "evalue_rr_cif": ev_entry,
                }

                co_table_rows.append(
                    {
                        "contrast_id": contrast_id,
                        "reference_arm_id": ref_id,
                        "active_arm_id": act_id,
                        "horizon_day": h_val,
                        "cif_primary_ref": round(c1_ref, 10),
                        "cif_primary_active": round(c1_act, 10),
                        "rd_primary": round(rd1, 10),
                        "rd_primary_ci_lower": round(float(rd1_lo), 10),
                        "rd_primary_ci_upper": round(float(rd1_hi), 10),
                        "rr_cif_primary": round(rr1, 10) if rr1 is not None else None,
                        "rr_cif_primary_ci_lower": round(rr1_lo, 10) if rr1_lo is not None else None,
                        "rr_cif_primary_ci_upper": round(rr1_hi, 10) if rr1_hi is not None else None,
                        "cif_competing_ref": round(c2_ref, 10),
                        "cif_competing_active": round(c2_act, 10),
                        "rd_competing": round(rd2, 10),
                        "rd_competing_ci_lower": round(float(rd2_lo), 10),
                        "rd_competing_ci_upper": round(float(rd2_hi), 10),
                        "shr_primary": round(shr_pt, 10) if shr_pt is not None else None,
                        "shr_primary_ci_lower": round(shr_lo, 10) if shr_lo is not None else None,
                        "shr_primary_ci_upper": round(shr_hi, 10) if shr_hi is not None else None,
                        "cshr_primary": round(cshr1_pt, 10) if cshr1_pt is not None else None,
                        "cshr_primary_ci_lower": round(cshr1_lo, 10) if cshr1_lo is not None else None,
                        "cshr_primary_ci_upper": round(cshr1_hi, 10) if cshr1_hi is not None else None,
                        "cshr_competing": round(cshr2_pt, 10) if cshr2_pt is not None else None,
                        "cshr_competing_ci_lower": round(cshr2_lo, 10) if cshr2_lo is not None else None,
                        "cshr_competing_ci_upper": round(cshr2_hi, 10) if cshr2_hi is not None else None,
                        "evalue_rr_cif_point": (
                            round(ev_entry["point_estimate"], 10) if ev_entry is not None else None
                        ),
                        "evalue_rr_cif_ci_limit": (
                            round(ev_entry["ci_limit_closer_to_null"], 10)
                            if (ev_entry is not None and ev_entry["ci_limit_closer_to_null"] is not None)
                            else None
                        ),
                    }
                )

            contrasts_dict[contrast_id] = {
                "reference_arm_id": ref_id,
                "active_arm_id": act_id,
                "fine_gray_shr_primary": {
                    "point": round(shr_pt, 10) if shr_pt is not None else None,
                    "ci_lower": round(shr_lo, 10) if shr_lo is not None else None,
                    "ci_upper": round(shr_hi, 10) if shr_hi is not None else None,
                    "se_robust_log_shr": (
                        round(fg_fit["se_robust"][j_col], 10) if fg_fit["se_robust"][j_col] is not None else None
                    ),
                    "converged": fg_fit["converged"],
                    "warning": fg_fit["warning"],
                },
                "cause_specific_cox_primary": {
                    "point": round(cshr1_pt, 10) if cshr1_pt is not None else None,
                    "ci_lower": round(cshr1_lo, 10) if cshr1_lo is not None else None,
                    "ci_upper": round(cshr1_hi, 10) if cshr1_hi is not None else None,
                    "se_robust_log_cshr": (
                        round(cox1_fit["se_robust"][j_col], 10) if cox1_fit["se_robust"][j_col] is not None else None
                    ),
                    "converged": cox1_fit["converged"],
                    "warning": cox1_fit["warning"],
                },
                "cause_specific_cox_competing": {
                    "point": round(cshr2_pt, 10) if cshr2_pt is not None else None,
                    "ci_lower": round(cshr2_lo, 10) if cshr2_lo is not None else None,
                    "ci_upper": round(cshr2_hi, 10) if cshr2_hi is not None else None,
                    "se_robust_log_cshr": (
                        round(cox2_fit["se_robust"][j_col], 10) if cox2_fit["se_robust"][j_col] is not None else None
                    ),
                    "converged": cox2_fit["converged"],
                    "warning": cox2_fit["warning"],
                },
                "horizons": horizon_contrasts,
            }
    else:
        # Pairwise overlap weighting (>=3 arms): evaluate each active arm k against reference arm 0
        ref_id = arm_ids[0]
        for k in range(1, num_arms):
            act_id = arm_ids[k]
            pair_id = f"{ref_id}__vs__{act_id}"
            contrast_id = f"{act_id}__vs__{ref_id}"
            raw_col = f"raw_weight__{pair_id}"
            if raw_col not in wdf.columns:
                raise GateBlockedError("CORRUPT_WEIGHTED_COHORT", f"Missing pairwise weight column '{raw_col}'")

            pair_mask = (arm_idx_all == 0) | (arm_idx_all == k)
            t_pair = durations_all[pair_mask]
            e_pair = event_types_all[pair_mask]
            bin_arm = (arm_idx_all[pair_mask] == k).astype(np.int64)
            w_raw_pair = pd.to_numeric(wdf.loc[pair_mask, raw_col], errors="raise").to_numpy(dtype=np.float64)
            if np.any(~np.isfinite(w_raw_pair)) or np.any(w_raw_pair <= 0.0):
                raise GateBlockedError("CORRUPT_WEIGHTED_COHORT", f"Invalid weights in '{raw_col}'")

            w_pair = np.zeros_like(w_raw_pair)
            pair_target_mass = 0.5 * float(w_raw_pair.size)
            for b_arm in (0, 1):
                mb = bin_arm == b_arm
                w_pair[mb] = pair_target_mass * (w_raw_pair[mb] / float(np.sum(w_raw_pair[mb])))

            x_pair = X_all[pair_mask] if X_all is not None else None

            pair_curves = []
            for b_arm, aid in ((0, ref_id), (1, act_id)):
                mb = bin_arm == b_arm
                ipcw_mult_b = None
                if censoring_ipcw != "none":
                    ev_b = np.unique(t_pair[mb][(e_pair[mb] == 1) | (e_pair[mb] == 2)])
                    if ev_b.size > 0:
                        mult_p = _compute_stabilized_joint_ipcw_matrix(
                            t_pair,
                            e_pair,
                            bin_arm,
                            2,
                            x_pair,
                            w_pair,
                            ev_b,
                            censoring_ipcw=censoring_ipcw,
                            admin_end_day=admin_end_day,
                            positivity_floor=positivity_floor,
                        )
                        ipcw_mult_b = mult_p[mb, :]
                c_b = weighted_aalen_johansen_cif(
                    t_pair[mb],
                    e_pair[mb],
                    w_pair[mb],
                    origin_day=landmark_day,
                    joint_ipcw_multipliers=ipcw_mult_b,
                )
                pair_curves.append(c_b)
                rel_c = f"curves/cif_{pair_id}__{aid}.csv"
                curve_specs.append((rel_c, aid, c_b, pair_id))

            # Derive deterministic per-pair seed offset so pairwise bootstrap streams are reproducible
            boot_c1, boot_c2 = _bootstrap_arm_cif_horizons(
                [t_pair[bin_arm == 0], t_pair[bin_arm == 1]],
                [e_pair[bin_arm == 0], e_pair[bin_arm == 1]],
                [w_pair[bin_arm == 0], w_pair[bin_arm == 1]],
                [
                    x_pair[bin_arm == 0] if x_pair is not None else None,
                    x_pair[bin_arm == 1] if x_pair is not None else None,
                ],
                horizons=horizons,
                origin_day=landmark_day,
                bootstrap_iterations=bootstrap_iterations,
                seed=(seed + 10007 * k) % (MAX_SEED + 1),
                censoring_ipcw=censoring_ipcw,
                admin_end_day=admin_end_day,
                positivity_floor=positivity_floor,
            )

            fg_p = fit_fine_gray_shr(
                t_pair,
                e_pair,
                bin_arm,
                2,
                w_pair,
                ci_level=ci_level_f,
                censoring_ipcw=censoring_ipcw,
                X_cov=x_pair,
                admin_end_day=admin_end_day,
                positivity_floor=positivity_floor,
            )
            cox1_p = fit_weighted_cause_specific_cox(
                t_pair,
                e_pair,
                bin_arm,
                2,
                w_pair,
                cause_code=1,
                ci_level=ci_level_f,
                censoring_ipcw=censoring_ipcw,
                X_cov=x_pair,
                admin_end_day=admin_end_day,
                positivity_floor=positivity_floor,
            )
            cox2_p = fit_weighted_cause_specific_cox(
                t_pair,
                e_pair,
                bin_arm,
                2,
                w_pair,
                cause_code=2,
                ci_level=ci_level_f,
                censoring_ipcw=censoring_ipcw,
                X_cov=x_pair,
                admin_end_day=admin_end_day,
                positivity_floor=positivity_floor,
            )

            shr_pt, shr_lo, shr_hi = fg_p["hr"][0], fg_p["ci_lower"][0], fg_p["ci_upper"][0]
            cshr1_pt, cshr1_lo, cshr1_hi = cox1_p["hr"][0], cox1_p["ci_lower"][0], cox1_p["ci_upper"][0]
            cshr2_pt, cshr2_lo, cshr2_hi = cox2_p["hr"][0], cox2_p["ci_lower"][0], cox2_p["ci_upper"][0]

            c1_ref_vec = evaluate_step_curve(pair_curves[0]["time_day"], pair_curves[0]["cif_primary"], horizons)
            c1_act_vec = evaluate_step_curve(pair_curves[1]["time_day"], pair_curves[1]["cif_primary"], horizons)
            c2_ref_vec = evaluate_step_curve(pair_curves[0]["time_day"], pair_curves[0]["cif_competing"], horizons)
            c2_act_vec = evaluate_step_curve(pair_curves[1]["time_day"], pair_curves[1]["cif_competing"], horizons)

            arm_cif_summaries[f"{pair_id}__{ref_id}"] = {
                "pair_id": pair_id,
                "arm_id": ref_id,
                "curve_csv": f"curves/cif_{pair_id}__{ref_id}.csv",
            }
            arm_cif_summaries[f"{pair_id}__{act_id}"] = {
                "pair_id": pair_id,
                "arm_id": act_id,
                "curve_csv": f"curves/cif_{pair_id}__{act_id}.csv",
            }

            horizon_contrasts = {}
            for h_idx, h_val in enumerate(horizons):
                c1_ref = float(c1_ref_vec[h_idx])
                c1_act = float(c1_act_vec[h_idx])
                c2_ref = float(c2_ref_vec[h_idx])
                c2_act = float(c2_act_vec[h_idx])
                rd1 = c1_act - c1_ref
                rd1_lo, rd1_hi = np.quantile(boot_c1[:, 1, h_idx] - boot_c1[:, 0, h_idx], [q_lo, q_hi])
                rd2 = c2_act - c2_ref
                rd2_lo, rd2_hi = np.quantile(boot_c2[:, 1, h_idx] - boot_c2[:, 0, h_idx], [q_lo, q_hi])

                rr1 = (c1_act / c1_ref) if c1_ref > 0.0 else None
                rr1_lo, rr1_hi = None, None
                valid_rr = boot_c1[:, 0, h_idx] > 0.0
                if rr1 is not None and np.any(valid_rr):
                    rr_boot = boot_c1[valid_rr, 1, h_idx] / boot_c1[valid_rr, 0, h_idx]
                    qv = np.quantile(rr_boot, [q_lo, q_hi])
                    rr1_lo, rr1_hi = float(qv[0]), float(qv[1])

                ev_entry = _compute_rr_cif_evalue_entry(rr1, rr1_lo, rr1_hi)
                horizon_contrasts[f"{h_val:g}"] = {
                    "horizon_day": h_val,
                    "cif_primary_ref": round(c1_ref, 10),
                    "cif_primary_active": round(c1_act, 10),
                    "rd_primary": round(rd1, 10),
                    "rd_primary_ci_lower": round(float(rd1_lo), 10),
                    "rd_primary_ci_upper": round(float(rd1_hi), 10),
                    "rr_cif_primary": round(rr1, 10) if rr1 is not None else None,
                    "rr_cif_primary_ci_lower": round(rr1_lo, 10) if rr1_lo is not None else None,
                    "rr_cif_primary_ci_upper": round(rr1_hi, 10) if rr1_hi is not None else None,
                    "cif_competing_ref": round(c2_ref, 10),
                    "cif_competing_active": round(c2_act, 10),
                    "rd_competing": round(rd2, 10),
                    "rd_competing_ci_lower": round(float(rd2_lo), 10),
                    "rd_competing_ci_upper": round(float(rd2_hi), 10),
                    "evalue_rr_cif": ev_entry,
                }
                co_table_rows.append(
                    {
                        "contrast_id": contrast_id,
                        "reference_arm_id": ref_id,
                        "active_arm_id": act_id,
                        "horizon_day": h_val,
                        "cif_primary_ref": round(c1_ref, 10),
                        "cif_primary_active": round(c1_act, 10),
                        "rd_primary": round(rd1, 10),
                        "rd_primary_ci_lower": round(float(rd1_lo), 10),
                        "rd_primary_ci_upper": round(float(rd1_hi), 10),
                        "rr_cif_primary": round(rr1, 10) if rr1 is not None else None,
                        "rr_cif_primary_ci_lower": round(rr1_lo, 10) if rr1_lo is not None else None,
                        "rr_cif_primary_ci_upper": round(rr1_hi, 10) if rr1_hi is not None else None,
                        "cif_competing_ref": round(c2_ref, 10),
                        "cif_competing_active": round(c2_act, 10),
                        "rd_competing": round(rd2, 10),
                        "rd_competing_ci_lower": round(float(rd2_lo), 10),
                        "rd_competing_ci_upper": round(float(rd2_hi), 10),
                        "shr_primary": round(shr_pt, 10) if shr_pt is not None else None,
                        "shr_primary_ci_lower": round(shr_lo, 10) if shr_lo is not None else None,
                        "shr_primary_ci_upper": round(shr_hi, 10) if shr_hi is not None else None,
                        "cshr_primary": round(cshr1_pt, 10) if cshr1_pt is not None else None,
                        "cshr_primary_ci_lower": round(cshr1_lo, 10) if cshr1_lo is not None else None,
                        "cshr_primary_ci_upper": round(cshr1_hi, 10) if cshr1_hi is not None else None,
                        "cshr_competing": round(cshr2_pt, 10) if cshr2_pt is not None else None,
                        "cshr_competing_ci_lower": round(cshr2_lo, 10) if cshr2_lo is not None else None,
                        "cshr_competing_ci_upper": round(cshr2_hi, 10) if cshr2_hi is not None else None,
                        "evalue_rr_cif_point": (
                            round(ev_entry["point_estimate"], 10) if ev_entry is not None else None
                        ),
                        "evalue_rr_cif_ci_limit": (
                            round(ev_entry["ci_limit_closer_to_null"], 10)
                            if (ev_entry is not None and ev_entry["ci_limit_closer_to_null"] is not None)
                            else None
                        ),
                    }
                )

            contrasts_dict[contrast_id] = {
                "pair_id": pair_id,
                "reference_arm_id": ref_id,
                "active_arm_id": act_id,
                "fine_gray_shr_primary": {
                    "point": round(shr_pt, 10) if shr_pt is not None else None,
                    "ci_lower": round(shr_lo, 10) if shr_lo is not None else None,
                    "ci_upper": round(shr_hi, 10) if shr_hi is not None else None,
                    "se_robust_log_shr": (
                        round(fg_p["se_robust"][0], 10) if fg_p["se_robust"][0] is not None else None
                    ),
                    "converged": fg_p["converged"],
                    "warning": fg_p["warning"],
                },
                "cause_specific_cox_primary": {
                    "point": round(cshr1_pt, 10) if cshr1_pt is not None else None,
                    "ci_lower": round(cshr1_lo, 10) if cshr1_lo is not None else None,
                    "ci_upper": round(cshr1_hi, 10) if cshr1_hi is not None else None,
                    "se_robust_log_cshr": (
                        round(cox1_p["se_robust"][0], 10) if cox1_p["se_robust"][0] is not None else None
                    ),
                    "converged": cox1_p["converged"],
                    "warning": cox1_p["warning"],
                },
                "cause_specific_cox_competing": {
                    "point": round(cshr2_pt, 10) if cshr2_pt is not None else None,
                    "ci_lower": round(cshr2_lo, 10) if cshr2_lo is not None else None,
                    "ci_upper": round(cshr2_hi, 10) if cshr2_hi is not None else None,
                    "se_robust_log_cshr": (
                        round(cox2_p["se_robust"][0], 10) if cox2_p["se_robust"][0] is not None else None
                    ),
                    "converged": cox2_p["converged"],
                    "warning": cox2_p["warning"],
                },
                "horizons": horizon_contrasts,
            }

    censoring_note = (
        BASELINE_INDEPENDENT_CENSORING_ASSUMPTION
        if censoring_ipcw == "none"
        else (
            f"Joint baseline x censoring IPCW sensitivity mode ('{censoring_ipcw}'): time-varying inverse "
            f"probability of censoring weights (positivity floor={positivity_floor:g}) combined with baseline "
            "treatment weights."
        )
    )

    active_claims = list(claims_for_source_mode(source_mode))
    estimates_payload = {
        "source_mode": source_mode,
        "prohibited_claims": active_claims,
        "protocol_version": PROTOCOL_VERSION,
        "study_title": protocol["study_title"],
        "simulation_or_data_label": protocol["simulation_or_data_label"],
        "target_estimand": protocol["estimand"]["target_estimand"],
        "weight_method": weight_method,
        "transitivity_warning": (
            PAIRWISE_TRANSITIVITY_WARNING if weight_method == "pairwise_overlap" else None
        ),
        "landmark_day": int(landmark_day),
        "horizons_days": horizons,
        "reference_arm": {
            "id": strategies[0]["id"],
            "label": strategies[0]["label"],
            "role": strategies[0]["role"],
        },
        "censoring_model": {
            "censoring_ipcw_mode": censoring_ipcw,
            "fine_gray_censoring_estimator": (
                "reverse_kaplan_meier_pooled"
                if censoring_ipcw == "none"
                else (
                    "reverse_kaplan_meier_arm_stratified"
                    if censoring_ipcw == "arm_stratified"
                    else "covariate_adjusted_cox_ipcw"
                )
            ),
            "positivity_floor": positivity_floor,
            "independence_assumption": censoring_note,
        },
        "bootstrap_config": {
            "seed": seed,
            "bootstrap_iterations": bootstrap_iterations,
            "ci_level": ci_level_f,
            "sampling_unit": "individual_subject_stratified_by_arm",
            "baseline_propensity_weights": "fixed",
            "censoring_ipcw_reestimated_in_bootstrap": bool(censoring_ipcw != "none"),
            "variance_scope_note": (
                "Bootstrap confidence intervals resample individual subjects stratified by treatment arm "
                "while holding Stage-1 baseline propensity/overlap weights from weighted_cohort.csv fixed "
                "(baseline_propensity_weights='fixed'; propensity models are not re-estimated per replicate, "
                "so CIs are conditional on estimated Stage-1 baseline weights and exclude first-stage "
                "propensity estimation uncertainty). "
                + (
                    "Under censoring_ipcw='none', no censoring IPCW model is fitted in bootstrap replicates."
                    if censoring_ipcw == "none"
                    else (
                        f"Under censoring_ipcw='{censoring_ipcw}', censoring survival IPCW weights are "
                        "re-estimated within each bootstrap replicate."
                    )
                )
            ),
        },
        "evalue_policy": {
            "supported_competing_risks_measure": "RR_CIF",
            "blocked_measures": [
                "sHR (unconditionally blocked: subdistribution HR has no single-event RR conversion)",
                "csHR / HR when competing risks exist and outcome_incidence > 0.15",
            ],
        },
        "methodological_warnings": {
            "subdistribution_vs_cause_specific_warning": SUBDISTRIBUTION_VS_CSHR_WARNING,
            "hazard_ratio_selection_bias_warning": HR_SELECTION_BIAS_WARNING,
            "observable_rule_disclaimer": METHODOLOGICAL_DISCLAIMER,
            "competing_mortality_divergence_alert": {
                "triggered": len(divergence_messages) > 0,
                "detail": " | ".join(divergence_messages) if divergence_messages else None,
            },
        },
        "co_presented_comparison_table": co_table_rows,
        "arm_cif_summaries": arm_cif_summaries,
        "contrasts": contrasts_dict,
    }

    if source_mode == SOURCE_MODE_USER_COHORT:
        estimates_payload["report_badge"] = "🟢 真实队列目标试验模拟（真项目）分析报告"
        estimates_payload["causal_assumptions"] = list(OBSERVATIONAL_CAUSAL_ASSUMPTIONS)
        if ingest_manifest is not None:
            estimates_payload["ingest_lineage"] = {
                "input_filename": ingest_manifest.get("input_filename"),
                "input_sha256": ingest_manifest.get("input_sha256"),
                "standardized_cohort_sha256": ingest_manifest.get("standardized_cohort_sha256"),
                "raw_row_count": ingest_manifest.get("raw_row_count"),
                "eligible_row_count": ingest_manifest.get("eligible_row_count"),
                "column_mapping": ingest_manifest.get("column_mapping"),
                "covariate_display_labels": ingest_manifest.get("covariate_display_labels"),
                "dropped_constant_or_collinear_covariates": ingest_manifest.get("dropped_constant_or_collinear_covariates"),
                "zero_competing_events_notice": ingest_manifest.get("zero_competing_events_notice"),
            }
    elif source_mode == SOURCE_MODE_SIMULATED_COHORT:
        estimates_payload["report_badge"] = "🟣 目标试验模拟（无数据模拟预演项目）可行性报告"

    if "target_rct" in protocol:
        estimates_payload["target_rct"] = protocol["target_rct"]

    audit_md_bytes = _render_audit_report_md(estimates_payload).encode("utf-8")
    estimates_bytes = json_bytes(estimates_payload)

    curve_rel_paths = [rel for rel, _, _, _ in curve_specs]
    output_paths = sorted(["estimates.json", "audit_report.md", *curve_rel_paths])

    commit, code_dirty = code_provenance()
    deps = dependency_versions()
    wc_sha = sha256_file(resolved_wc_path)

    def _populate(stage: Path) -> None:
        (stage / "estimates.json").write_bytes(estimates_bytes)
        (stage / "audit_report.md").write_bytes(audit_md_bytes)
        for rel_path, aid, c_dict, p_id in curve_specs:
            _write_curve_csv(stage / rel_path, aid, c_dict, pair_id=p_id)

        manifest = {
            "status": "tte_competing_risks_estimated",
            "activity": "tte_weighted_competing_risks_estimation",
            "source_mode": source_mode,
            "prohibited_claims": active_claims,
            "protocol_version": PROTOCOL_VERSION,
            "simulation_or_data_label": protocol["simulation_or_data_label"],
            "methodological_disclaimer": METHODOLOGICAL_DISCLAIMER,
            "weight_method": weight_method,
            "censoring_ipcw": censoring_ipcw,
            "seed": seed,
            "bootstrap_iterations": bootstrap_iterations,
            "ci_level": ci_level_f,
            "horizons_days": horizons,
            "protocol_sha256": sha256_bytes(raw_proto),
            "normalized_protocol_sha256": sha256_bytes(json_bytes(protocol)),
            "diagnostics_manifest_sha256": sha256_bytes(raw_diag_manifest),
            "weighted_cohort_sha256": wc_sha,
            "code_commit": commit,
            "code_dirty": code_dirty,
            "dependency_versions": deps,
            "output_paths": output_paths,
            "output_sha256": {p: sha256_file(stage / p) for p in output_paths},
            "run_at_utc": utc_now_iso(),
        }
        if source_mode == SOURCE_MODE_USER_COHORT and ingest_manifest is not None:
            manifest["input_sha256"] = ingest_manifest.get("input_sha256")
        (stage / "run_manifest.json").write_bytes(json_bytes(manifest))

    atomic_publish(out_dir, _populate)
    return {
        "status": "tte_competing_risks_estimated",
        "source_mode": source_mode,
        "simulation_or_data_label": protocol["simulation_or_data_label"],
        "weight_method": weight_method,
        "censoring_ipcw": censoring_ipcw,
        "seed": seed,
        "bootstrap_iterations": bootstrap_iterations,
        "horizons_days": horizons,
        "output_path": str(out_dir),
    }
