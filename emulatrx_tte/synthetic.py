"""Seeded, purely synthetic competing-risks and multi-arm cohort generator for TTE benchmarking."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.special import expit, softmax

from ._common import (
    ContractError,
    MAX_SEED,
    MAX_SYNTH_CASES,
    METHODOLOGICAL_DISCLAIMER,
    PROTOCOL_VERSION,
    atomic_publish,
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
from .protocol import load_and_validate_protocol


DEFAULT_COVARIATES = (
    {"name": "x_age_std", "type": "continuous", "assessed_day_offset": 0},
    {"name": "x_biomarker_std", "type": "continuous", "assessed_day_offset": 0},
    {"name": "x_comorbidity_bin", "type": "binary", "assessed_day_offset": 0},
    {"name": "x_high_risk_bin", "type": "binary", "assessed_day_offset": 0},
)

ALLOWED_DGP_KEYS = frozenset(
    (
        "propensity_intercepts",
        "propensity_slopes",
        "baseline_hazard_primary",
        "baseline_hazard_competing",
        "random_censor_hazard",
        "pre_landmark_hazard_multiplier",
        "log_cshr_primary_by_arm",
        "log_cshr_competing_by_arm",
        "outcome_covariate_log_hr_primary",
        "outcome_covariate_log_hr_competing",
    )
)


def _default_dgp_parameters(num_arms: int, num_covs: int, has_competing: bool) -> dict:
    # Arm 0 is reference (intercept=0, slopes=0); arms 1..K-1 receive confounded logit parameters.
    intercepts = [0.0]
    slopes = [[0.0] * num_covs]
    if num_arms == 2:
        arm_patterns = [[0.55, -0.45, 0.50, 0.40]]
        arm_intercepts = [-0.15]
    else:
        arm_patterns = [
            [0.36, -0.28, 0.32, 0.28],
            [-0.26, 0.34, -0.22, 0.24],
            [0.22, 0.24, 0.30, -0.26],
        ]
        arm_intercepts = [-0.08 * k for k in range(1, num_arms)]

    for idx in range(num_arms - 1):
        pat = arm_patterns[idx % len(arm_patterns)]
        intercepts.append(round(arm_intercepts[idx], 6))
        slopes.append([round(pat[j % len(pat)], 6) for j in range(num_covs)])

    log_cshr_1 = [0.0] + [round(-0.38 - 0.10 * (k - 1), 6) for k in range(1, num_arms)]
    log_cshr_2 = (
        [0.0] + [round(-0.15 + 0.08 * (k - 1), 6) for k in range(1, num_arms)]
        if has_competing
        else [0.0] * num_arms
    )
    cov_hr_1 = [round([0.30, 0.25, 0.35, 0.40][j % 4], 6) for j in range(num_covs)]
    cov_hr_2 = (
        [round([0.35, 0.15, 0.45, 0.25][j % 4], 6) for j in range(num_covs)]
        if has_competing
        else [0.0] * num_covs
    )
    return {
        "propensity_intercepts": intercepts,
        "propensity_slopes": slopes,
        "baseline_hazard_primary": 0.0028,
        "baseline_hazard_competing": 0.0016 if has_competing else 0.0,
        "random_censor_hazard": 0.0005,
        "pre_landmark_hazard_multiplier": 1.0,
        "log_cshr_primary_by_arm": log_cshr_1,
        "log_cshr_competing_by_arm": log_cshr_2,
        "outcome_covariate_log_hr_primary": cov_hr_1,
        "outcome_covariate_log_hr_competing": cov_hr_2,
    }


def _merge_and_validate_dgp_params(
    defaults: dict,
    override: dict | None,
    *,
    num_arms: int,
    num_covs: int,
    has_competing: bool,
) -> dict:
    params = dict(defaults)
    if override is not None:
        unknown = sorted(set(override.keys()) - ALLOWED_DGP_KEYS)
        if unknown:
            raise ContractError("UNKNOWN_DGP_PARAM", f"Unknown DGP parameter(s): {', '.join(unknown)}")
        params.update(override)

    def _vec(name: str, expected_len: int) -> list[float]:
        val = params[name]
        if not isinstance(val, list) or len(val) != expected_len:
            raise ContractError("INVALID_DGP_PARAM", f"{name} must be a list of length {expected_len}")
        out = []
        for idx, x in enumerate(val):
            if not is_strict_number(x):
                raise ContractError("INVALID_DGP_PARAM", f"{name}[{idx}] must be a finite number")
            out.append(float(x))
        return out

    intercepts = _vec("propensity_intercepts", num_arms)
    raw_slopes = params["propensity_slopes"]
    if not isinstance(raw_slopes, list) or len(raw_slopes) != num_arms:
        raise ContractError("INVALID_DGP_PARAM", f"propensity_slopes must have {num_arms} arm rows")
    slopes: list[list[float]] = []
    for k, row in enumerate(raw_slopes):
        if not isinstance(row, list) or len(row) != num_covs:
            raise ContractError("INVALID_DGP_PARAM", f"propensity_slopes[{k}] must have length {num_covs}")
        srow = []
        for j, x in enumerate(row):
            if not is_strict_number(x):
                raise ContractError("INVALID_DGP_PARAM", f"propensity_slopes[{k}][{j}] must be finite")
            srow.append(float(x))
        slopes.append(srow)

    bh1 = params["baseline_hazard_primary"]
    bh2 = params["baseline_hazard_competing"]
    rc = params["random_censor_hazard"]
    pre_mult = params["pre_landmark_hazard_multiplier"]
    if not is_strict_number(bh1) or float(bh1) <= 0.0:
        raise ContractError("INVALID_DGP_PARAM", "baseline_hazard_primary must be > 0")
    if not is_strict_number(bh2) or float(bh2) < 0.0:
        raise ContractError("INVALID_DGP_PARAM", "baseline_hazard_competing must be >= 0")
    if has_competing and float(bh2) <= 0.0:
        raise ContractError(
            "INVALID_DGP_PARAM",
            "baseline_hazard_competing must be > 0 when protocol declares competing_event_codes",
        )
    if not has_competing:
        bh2 = 0.0
    if not is_strict_number(rc) or float(rc) < 0.0:
        raise ContractError("INVALID_DGP_PARAM", "random_censor_hazard must be >= 0")
    if not is_strict_number(pre_mult) or float(pre_mult) <= 0.0:
        raise ContractError("INVALID_DGP_PARAM", "pre_landmark_hazard_multiplier must be > 0")

    return {
        "propensity_intercepts": intercepts,
        "propensity_slopes": slopes,
        "baseline_hazard_primary": float(bh1),
        "baseline_hazard_competing": float(bh2),
        "random_censor_hazard": float(rc),
        "pre_landmark_hazard_multiplier": float(pre_mult),
        "log_cshr_primary_by_arm": _vec("log_cshr_primary_by_arm", num_arms),
        "log_cshr_competing_by_arm": _vec("log_cshr_competing_by_arm", num_arms),
        "outcome_covariate_log_hr_primary": _vec("outcome_covariate_log_hr_primary", num_covs),
        "outcome_covariate_log_hr_competing": _vec("outcome_covariate_log_hr_competing", num_covs),
    }


def _piecewise_exponential_sample(
    u: np.ndarray,
    rate_post: np.ndarray,
    landmark_day: float,
    pre_multiplier: float,
) -> np.ndarray:
    """Exact inverse-CDF draw for piecewise-constant hazard on [0, landmark_day] and (landmark_day, inf)."""
    neg_log_u = -np.log(np.clip(u, 1e-15, 1.0))
    if landmark_day <= 0.0 or pre_multiplier == 1.0:
        return neg_log_u / rate_post
    rate_pre = rate_post * pre_multiplier
    cum_hazard_landmark = rate_pre * landmark_day
    in_pre = neg_log_u <= cum_hazard_landmark
    t_pre = neg_log_u / rate_pre
    t_post = landmark_day + (neg_log_u - cum_hazard_landmark) / rate_post
    return np.where(in_pre, t_pre, t_post)


def _analytical_cif_at_reference_x0(
    *,
    strategies: list[dict],
    horizons: list[float],
    landmark_day: float,
    dgp: dict,
    has_competing: bool,
) -> dict:
    """Closed-form conditional CIF(t | arm, X=0, T > landmark_day) at each protocol horizon."""
    by_arm = {}
    bh1 = dgp["baseline_hazard_primary"]
    bh2 = dgp["baseline_hazard_competing"]
    for k, strat in enumerate(strategies):
        lam1 = bh1 * math.exp(dgp["log_cshr_primary_by_arm"][k])
        lam2 = bh2 * math.exp(dgp["log_cshr_competing_by_arm"][k]) if has_competing else 0.0
        total_lam = lam1 + lam2
        horizon_map = {}
        for h in horizons:
            dt = max(0.0, float(h) - float(landmark_day))
            total_event_prob = 1.0 - math.exp(-total_lam * dt) if total_lam > 0.0 else 0.0
            cif1 = (lam1 / total_lam) * total_event_prob if total_lam > 0.0 else 0.0
            cif2 = (lam2 / total_lam) * total_event_prob if (has_competing and total_lam > 0.0) else 0.0
            horizon_map[f"{h:g}"] = {
                "horizon_day": float(h),
                "cif_primary": round(cif1, 10),
                "cif_competing": round(cif2, 10),
            }
        by_arm[strat["id"]] = {
            "cshr_primary_vs_arm0": round(math.exp(dgp["log_cshr_primary_by_arm"][k]), 10),
            "cshr_competing_vs_arm0": (
                round(math.exp(dgp["log_cshr_competing_by_arm"][k]), 10) if has_competing else None
            ),
            "horizons_conditional_on_x0_and_landmark_survival": horizon_map,
        }
    return by_arm


def generate_synthetic_cohort(
    protocol_path: Path,
    *,
    seed: int,
    cases: int,
    out_dir: Path,
    dgp_params_path: Path | None = None,
) -> dict:
    if not is_strict_int(seed) or not (0 <= seed <= MAX_SEED):
        raise ContractError("INVALID_SEED", f"seed must be an integer in 0..{MAX_SEED}")
    if not is_strict_int(cases) or not (1 <= cases <= MAX_SYNTH_CASES):
        raise ContractError("INVALID_CASE_COUNT", f"cases must be an integer in 1..{MAX_SYNTH_CASES}")

    raw_proto, protocol = load_and_validate_protocol(protocol_path)
    raw_dgp: bytes | None = None
    override_dgp: dict | None = None
    if dgp_params_path is not None:
        raw_dgp, override_dgp = read_strict_json(dgp_params_path, io_error_code="DGP_PARAMS_IO_ERROR")

    strategies = protocol["treatment_strategies"]
    num_arms = len(strategies)
    cov_specs = protocol.get("covariates") or [dict(c) for c in DEFAULT_COVARIATES]
    num_covs = len(cov_specs)
    competing_codes = protocol["outcomes"]["competing_event_codes"]
    has_competing = len(competing_codes) > 0
    competing_code = competing_codes[0] if has_competing else None
    primary_code = protocol["outcomes"]["primary_event_code"]
    censor_code = protocol["outcomes"]["censor_code"]
    landmark_day = float(protocol["time_zero"]["landmark_day"])
    admin_end_day = float(protocol["followup"]["administrative_end_day"])

    dgp = _merge_and_validate_dgp_params(
        _default_dgp_parameters(num_arms, num_covs, has_competing),
        override_dgp,
        num_arms=num_arms,
        num_covs=num_covs,
        has_competing=has_competing,
    )

    rng = np.random.default_rng(seed)

    # 1. Generate correlated continuous & binary baseline covariates deterministically
    X = np.zeros((cases, num_covs), dtype=np.float64)
    latent = rng.standard_normal(cases)
    for j, spec in enumerate(cov_specs):
        if spec["type"] == "continuous":
            noise = rng.standard_normal(cases)
            rho = 0.30 if j > 0 else 0.0
            X[:, j] = rho * latent + math.sqrt(1.0 - rho * rho) * noise
        else:
            p_bin = expit(-0.35 + 0.55 * latent + (0.25 * X[:, 0] if j > 0 else 0.0))
            X[:, j] = (rng.uniform(size=cases) < p_bin).astype(np.float64)

    # 2. Treatment assignment (2-arm logistic or >=3-arm multinomial logit)
    intercepts = np.asarray(dgp["propensity_intercepts"], dtype=np.float64)
    slopes = np.asarray(dgp["propensity_slopes"], dtype=np.float64)
    logits = X @ slopes.T + intercepts[None, :]
    logits = logits - logits[:, [0]]  # anchor arm 0 at 0
    if num_arms == 2:
        p1 = expit(logits[:, 1])
        true_ps = np.column_stack([1.0 - p1, p1])
        arm_idx = (rng.uniform(size=cases) < p1).astype(np.int64)
    else:
        true_ps = softmax(logits, axis=1)
        u_arm = rng.uniform(size=cases)[:, None]
        cum_ps = np.cumsum(true_ps, axis=1)
        cum_ps[:, -1] = 1.0
        arm_idx = (u_arm >= cum_ps).sum(axis=1).astype(np.int64)

    # 3. Competing latent event times + independent censoring
    beta_cov_1 = np.asarray(dgp["outcome_covariate_log_hr_primary"], dtype=np.float64)
    gamma_arm_1 = np.asarray(dgp["log_cshr_primary_by_arm"], dtype=np.float64)
    rate_1 = dgp["baseline_hazard_primary"] * np.exp(X @ beta_cov_1 + gamma_arm_1[arm_idx])
    t1 = _piecewise_exponential_sample(
        rng.uniform(size=cases), rate_1, landmark_day, dgp["pre_landmark_hazard_multiplier"]
    )

    if has_competing:
        beta_cov_2 = np.asarray(dgp["outcome_covariate_log_hr_competing"], dtype=np.float64)
        gamma_arm_2 = np.asarray(dgp["log_cshr_competing_by_arm"], dtype=np.float64)
        rate_2 = dgp["baseline_hazard_competing"] * np.exp(X @ beta_cov_2 + gamma_arm_2[arm_idx])
        t2 = _piecewise_exponential_sample(
            rng.uniform(size=cases), rate_2, landmark_day, dgp["pre_landmark_hazard_multiplier"]
        )
    else:
        t2 = np.full(cases, np.inf, dtype=np.float64)

    if dgp["random_censor_hazard"] > 0.0:
        c_rand = -np.log(np.clip(rng.uniform(size=cases), 1e-15, 1.0)) / dgp["random_censor_hazard"]
        c_time = np.minimum(c_rand, admin_end_day)
    else:
        c_time = np.full(cases, admin_end_day, dtype=np.float64)

    duration_day = np.minimum(np.minimum(t1, t2), c_time)
    event_code = np.full(cases, censor_code, dtype=np.int64)
    primary_wins = (t1 <= t2) & (t1 <= c_time)
    competing_wins = (t2 < t1) & (t2 <= c_time)
    event_code[primary_wins] = primary_code
    if has_competing and competing_code is not None:
        event_code[competing_wins] = competing_code

    arm_ids = [s["id"] for s in strategies]
    cov_names = [c["name"] for c in cov_specs]

    analytical_summary = _analytical_cif_at_reference_x0(
        strategies=strategies,
        horizons=protocol["followup"]["horizon_days"],
        landmark_day=landmark_day,
        dgp=dgp,
        has_competing=has_competing,
    )

    dgp_truth = {
        "parameter_type": "synthetic_dgp_parameters",
        "simulation_or_data_label": protocol["simulation_or_data_label"],
        "methodological_disclaimer": METHODOLOGICAL_DISCLAIMER,
        "non_clinical_notice": (
            "Mathematical data-generating process (DGP) parameters for offline estimator verification only. "
            "Generated from a pseudo-random seed; carries zero patient data and zero clinical meaning."
        ),
        "seed": seed,
        "case_count": cases,
        "landmark_day": int(landmark_day),
        "pre_landmark_event_or_censor_count": int(np.sum(duration_day <= landmark_day)),
        "arm_counts": {arm_ids[k]: int(np.sum(arm_idx == k)) for k in range(num_arms)},
        "event_counts": {
            "censor_0": int(np.sum(event_code == censor_code)),
            "primary_1": int(np.sum(event_code == primary_code)),
            "competing": (
                int(np.sum(event_code == competing_code))
                if (has_competing and competing_code is not None)
                else 0
            ),
        },
        "synthetic_dgp_parameters": {
            **dgp,
            "covariate_schema": cov_specs,
            "treatment_strategy_ids": arm_ids,
            "analytical_reference_quantities": analytical_summary,
        },
    }

    commit, code_dirty = code_provenance()
    deps = dependency_versions()
    output_paths = ["cohort.csv", "dgp_truth.json", "PURPOSE.txt"]

    def _populate(stage: Path) -> None:
        cohort_path = stage / "cohort.csv"
        with cohort_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            header = [
                "synthetic_case_id",
                "simulation_or_data_label",
                "treatment_arm",
                "treatment_idx",
                "duration_day",
                "event_code",
                *cov_names,
            ]
            writer.writerow(header)
            label = protocol["simulation_or_data_label"]
            for i in range(cases):
                cov_formatted = [
                    str(int(X[i, j])) if cov_specs[j]["type"] == "binary" else f"{X[i, j]:.8f}"
                    for j in range(num_covs)
                ]
                writer.writerow(
                    [
                        f"SYN-{i + 1:06d}",
                        label,
                        arm_ids[int(arm_idx[i])],
                        int(arm_idx[i]),
                        f"{duration_day[i]:.8f}",
                        int(event_code[i]),
                        *cov_formatted,
                    ]
                )

        (stage / "dgp_truth.json").write_bytes(json_bytes(dgp_truth))
        (stage / "PURPOSE.txt").write_text(
            f"{protocol['simulation_or_data_label']}\n"
            "Offline seeded synthetic competing-risks / multi-arm benchmark cohort for EmulatRx TTE.\n"
            "All subject identifiers (SYN-000001..), covariates, treatment arms, and event times are "
            "generated deterministically from np.random.default_rng(seed).\n"
            "No patient records, trial results, public registration responses, databases, or APIs are read.\n",
            encoding="utf-8",
        )
        manifest = {
            "status": "synthetic_tte_cohort_generated",
            "activity": "offline_synthetic_competing_risks_generation",
            "protocol_version": PROTOCOL_VERSION,
            "simulation_or_data_label": protocol["simulation_or_data_label"],
            "methodological_disclaimer": METHODOLOGICAL_DISCLAIMER,
            "seed": seed,
            "case_count": cases,
            "input_sha256": sha256_bytes(raw_proto),
            "dgp_params_sha256": sha256_bytes(raw_dgp) if raw_dgp is not None else None,
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
        "status": "synthetic_tte_cohort_generated",
        "simulation_or_data_label": protocol["simulation_or_data_label"],
        "seed": seed,
        "case_count": cases,
        "output_path": str(out_dir),
    }
