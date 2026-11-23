"""Exact propensity score estimation, overlap/IPTW weighting, and multi-arm weighting algorithms."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax

from ._common import ContractError


ALLOWED_WEIGHT_METHODS = frozenset(
    ("overlap", "stabilized_iptw", "iptw", "generalized_overlap", "pairwise_overlap")
)

PAIRWISE_TRANSITIVITY_WARNING = (
    "Non-transitivity warning (pairwise_overlap): each unordered arm pair (j, k) tilts toward its own "
    "pairwise equipoise subpopulation h_{jk}(X) = e_{k|{j,k}}(X) * (1 - e_{k|{j,k}}(X)). Because "
    "pairwise ATO subpopulations differ across pairs and estimates are non-transitive "
    "(effect(A vs B) + effect(B vs C) need not equal effect(A vs C)), use 'generalized_overlap' when a "
    "single common multi-arm equipoise population is required."
)


@dataclass(frozen=True)
class PairWeightResult:
    pair_id: str
    arm_j_idx: int
    arm_k_idx: int
    arm_j_id: str
    arm_k_id: str
    subset_indices: np.ndarray
    binary_arm_indicator: np.ndarray  # 1 for arm_k, 0 for arm_j
    propensity_k_given_pair: np.ndarray
    tilting_h: np.ndarray
    raw_weights: np.ndarray
    normalized_weights: np.ndarray
    analysis_weights: np.ndarray
    truncation_bounds: tuple[float, float] | None


@dataclass(frozen=True)
class WeightingResult:
    weight_method: str
    propensity_matrix: np.ndarray | None  # (n, K) for joint models; None for pairwise_overlap
    tilting_h: np.ndarray | None
    raw_weights: np.ndarray | None
    normalized_weights: np.ndarray | None
    analysis_weights: np.ndarray | None
    pairwise_results: tuple[PairWeightResult, ...] | None
    truncation_quantiles: tuple[float, float] | None
    truncation_bounds: tuple[float, float] | None
    transitivity_warning: str | None


def parse_truncate_quantiles(spec: str | tuple[float, float] | None) -> tuple[float, float] | None:
    if spec is None:
        return None
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise ContractError("INVALID_TRUNCATE_QUANTILES", "truncate_quantiles tuple must have 2 floats")
        q_low, q_high = float(spec[0]), float(spec[1])
    elif isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 2:
            raise ContractError(
                "INVALID_TRUNCATE_QUANTILES",
                f"--truncate-quantiles '{spec}' must have format 'q_low,q_high' (e.g. '0.01,0.99')",
            )
        try:
            q_low, q_high = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise ContractError("INVALID_TRUNCATE_QUANTILES", f"Non-numeric quantile in '{spec}'") from exc
    else:
        raise ContractError("INVALID_TRUNCATE_QUANTILES", "Invalid truncate_quantiles specification")

    if not (np.isfinite(q_low) and np.isfinite(q_high)) or not (0.0 <= q_low < q_high <= 1.0):
        raise ContractError(
            "INVALID_TRUNCATE_QUANTILES",
            f"truncate_quantiles ({q_low}, {q_high}) must satisfy 0.0 <= q_low < q_high <= 1.0",
        )
    return (q_low, q_high)


def fit_binary_logistic_mle(X: np.ndarray, y: np.ndarray, *, max_iter: int = 100, tol: float = 1e-13) -> np.ndarray:
    """Fit unpenalized binary logistic regression via damped IRLS / Newton-Raphson.

    Satisfies the exact MLE score equation X_aug.T @ (y - e_hat) = 0 to machine precision on
    non-separated data (required for exact finite-sample Overlap Weighting balance, Li et al. 2018),
    while clamping linear predictors in [-35, 35] so separated cohorts yield finite (0, 1) probabilities
    that cleanly trigger positivity gates without NaN/overflow.
    """
    n, p = X.shape
    X_aug = np.column_stack([np.ones(n, dtype=np.float64), X.astype(np.float64)])
    y_f = y.astype(np.float64)
    beta = np.zeros(p + 1, dtype=np.float64)

    def _neg_loglik(b: np.ndarray) -> float:
        eta = np.clip(X_aug @ b, -35.0, 35.0)
        # Stable negative log-likelihood: sum(log(1 + exp(eta)) - y * eta)
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
        # Tiny ridge floor only when Hessian is singular/near-separated; vanishes as grad -> 0.
        hess = (X_aug * np.maximum(w, 1e-14)[:, None]).T @ X_aug
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(hess + 1e-12 * np.eye(p + 1), grad, rcond=None)

        # Backtracking line search to prevent overshoot under quasi-separation
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

    eta_final = np.clip(X_aug @ beta, -35.0, 35.0)
    return expit(eta_final)


def fit_multinomial_logistic_mle(X: np.ndarray, arm_idx: np.ndarray, num_arms: int) -> np.ndarray:
    """Fit unpenalized multinomial logistic regression with arm 0 as reference."""
    if num_arms == 2:
        p1 = fit_binary_logistic_mle(X, (arm_idx == 1).astype(np.float64))
        return np.column_stack([1.0 - p1, p1])

    n, p = X.shape
    X_aug = np.column_stack([np.ones(n, dtype=np.float64), X.astype(np.float64)])
    d = p + 1
    Y = np.eye(num_arms, dtype=np.float64)[arm_idx]  # (n, K)

    def _obj_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        B = theta.reshape((num_arms - 1, d))
        logits = np.zeros((n, num_arms), dtype=np.float64)
        logits[:, 1:] = np.clip(X_aug @ B.T, -35.0, 35.0)
        log_probs = logits - np.log(np.sum(np.exp(logits - np.max(logits, axis=1, keepdims=True)), axis=1, keepdims=True)) - np.max(logits, axis=1, keepdims=True)
        nll = -float(np.sum(Y * log_probs))
        probs = softmax(logits, axis=1)
        grad_B = (probs[:, 1:] - Y[:, 1:]).T @ X_aug  # (K-1, d)
        return nll, grad_B.ravel()

    init = np.zeros((num_arms - 1) * d, dtype=np.float64)
    res = minimize(
        _obj_and_grad,
        init,
        jac=True,
        method="BFGS",
        options={"gtol": 1e-11, "maxiter": 2000},
    )
    B_hat = res.x.reshape((num_arms - 1, d))
    logits_final = np.zeros((n, num_arms), dtype=np.float64)
    logits_final[:, 1:] = np.clip(X_aug @ B_hat.T, -35.0, 35.0)
    return softmax(logits_final, axis=1)


def _normalize_by_arm(raw_weights: np.ndarray, arm_idx: np.ndarray, num_arms: int) -> tuple[np.ndarray, np.ndarray]:
    norm_w = np.zeros_like(raw_weights, dtype=np.float64)
    analysis_w = np.zeros_like(raw_weights, dtype=np.float64)
    for k in range(num_arms):
        mask = arm_idx == k
        n_k = int(np.sum(mask))
        if n_k == 0:
            continue
        s_k = float(np.sum(raw_weights[mask]))
        if s_k <= 0.0:
            norm_w[mask] = 1.0 / n_k
        else:
            norm_w[mask] = raw_weights[mask] / s_k
        analysis_w[mask] = n_k * norm_w[mask]
    return norm_w, analysis_w


def _apply_truncation(
    raw_weights: np.ndarray,
    truncate_quantiles: tuple[float, float] | None,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    if truncate_quantiles is None:
        return raw_weights, None
    q_low, q_high = truncate_quantiles
    lo, hi = np.quantile(raw_weights, [q_low, q_high])
    lo_f, hi_f = float(lo), float(hi)
    return np.clip(raw_weights, lo_f, hi_f), (lo_f, hi_f)


def compute_weights(
    X: np.ndarray,
    arm_idx: np.ndarray,
    arm_ids: list[str],
    *,
    weight_method: str,
    truncate_quantiles: tuple[float, float] | None = None,
) -> WeightingResult:
    num_arms = len(arm_ids)
    if weight_method not in ALLOWED_WEIGHT_METHODS:
        raise ContractError(
            "INVALID_WEIGHT_METHOD",
            f"Unsupported weight_method '{weight_method}'. Allowed: {sorted(ALLOWED_WEIGHT_METHODS)}",
        )
    if weight_method == "overlap" and num_arms != 2:
        raise ContractError(
            "INVALID_WEIGHT_METHOD",
            f"Two-arm 'overlap' requires exactly 2 treatment strategies (got {num_arms}). "
            "Hint: Use 'generalized_overlap' or 'pairwise_overlap' for >=3 strategies.",
        )

    n = X.shape[0]
    if weight_method == "pairwise_overlap":
        pair_results: list[PairWeightResult] = []
        for j in range(num_arms):
            for k in range(j + 1, num_arms):
                pair_mask = (arm_idx == j) | (arm_idx == k)
                sub_idx = np.flatnonzero(pair_mask)
                X_sub = X[sub_idx]
                y_sub = (arm_idx[sub_idx] == k).astype(np.int64)
                e_k = fit_binary_logistic_mle(X_sub, y_sub)
                h_sub = e_k * (1.0 - e_k)
                w_raw = np.where(y_sub == 1, 1.0 - e_k, e_k)
                w_raw, bounds = _apply_truncation(w_raw, truncate_quantiles)
                w_norm, w_analysis = _normalize_by_arm(w_raw, y_sub, 2)
                pair_results.append(
                    PairWeightResult(
                        pair_id=f"{arm_ids[j]}__vs__{arm_ids[k]}",
                        arm_j_idx=j,
                        arm_k_idx=k,
                        arm_j_id=arm_ids[j],
                        arm_k_id=arm_ids[k],
                        subset_indices=sub_idx,
                        binary_arm_indicator=y_sub,
                        propensity_k_given_pair=e_k,
                        tilting_h=h_sub,
                        raw_weights=w_raw,
                        normalized_weights=w_norm,
                        analysis_weights=w_analysis,
                        truncation_bounds=bounds,
                    )
                )
        return WeightingResult(
            weight_method=weight_method,
            propensity_matrix=None,
            tilting_h=None,
            raw_weights=None,
            normalized_weights=None,
            analysis_weights=None,
            pairwise_results=tuple(pair_results),
            truncation_quantiles=truncate_quantiles,
            truncation_bounds=None,
            transitivity_warning=PAIRWISE_TRANSITIVITY_WARNING,
        )

    ps_mat = fit_multinomial_logistic_mle(X, arm_idx, num_arms)
    e_assigned = ps_mat[np.arange(n), arm_idx]

    if weight_method == "overlap":
        e1 = ps_mat[:, 1]
        tilting_h = e1 * (1.0 - e1)
        w_raw = np.where(arm_idx == 1, 1.0 - e1, e1)
    elif weight_method == "generalized_overlap":
        inv_ps = 1.0 / np.clip(ps_mat, 1e-15, 1.0)
        sum_inv = np.sum(inv_ps, axis=1)
        tilting_h = 1.0 / sum_inv
        w_raw = inv_ps[np.arange(n), arm_idx] / sum_inv
    elif weight_method == "iptw":
        tilting_h = np.ones(n, dtype=np.float64)
        w_raw = 1.0 / np.clip(e_assigned, 1e-15, 1.0)
    else:  # stabilized_iptw
        tilting_h = np.ones(n, dtype=np.float64)
        marginal_p = np.array([np.mean(arm_idx == k) for k in range(num_arms)], dtype=np.float64)
        w_raw = marginal_p[arm_idx] / np.clip(e_assigned, 1e-15, 1.0)

    w_raw, trunc_bounds = _apply_truncation(w_raw, truncate_quantiles)
    w_norm, w_analysis = _normalize_by_arm(w_raw, arm_idx, num_arms)
    return WeightingResult(
        weight_method=weight_method,
        propensity_matrix=ps_mat,
        tilting_h=tilting_h,
        raw_weights=w_raw,
        normalized_weights=w_norm,
        analysis_weights=w_analysis,
        pairwise_results=None,
        truncation_quantiles=truncate_quantiles,
        truncation_bounds=trunc_bounds,
        transitivity_warning=None,
    )


def kish_ess(weights: np.ndarray) -> float:
    if weights.size == 0:
        return 0.0
    sum_w = float(np.sum(weights))
    sum_w2 = float(np.sum(weights * weights))
    if sum_w <= 0.0 or sum_w2 <= 0.0:
        return 0.0
    return float((sum_w * sum_w) / sum_w2)
