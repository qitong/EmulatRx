"""End-to-end atomic synthetic TTE competing-risks pipeline (validate-protocol -> synth -> diagnose -> estimate)."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from ._common import (
    ContractError,
    MAX_BOOTSTRAP_ITERATIONS,
    MAX_SEED,
    MAX_SYNTH_CASES,
    METHODOLOGICAL_DISCLAIMER,
    OBSERVATIONAL_CAUSAL_ASSUMPTIONS,
    PROHIBITED_CLAIMS,
    PROTOCOL_VERSION,
    SIMULATED_COHORT_CLAIMS,
    SOURCE_MODE_SIMULATED_COHORT,
    SOURCE_MODE_SYNTHETIC_ONLY,
    SOURCE_MODE_USER_COHORT,
    USER_COHORT_CAUSAL_CLAIMS,
    atomic_publish,
    check_destination,
    code_provenance,
    dependency_versions,
    is_strict_int,
    is_strict_number,
    json_bytes,
    sha256_bytes,
    sha256_file,
    utc_now_iso,
)
from .cohort_ingest import ingest_user_cohort
from .competing_risks import (
    ALLOWED_CENSORING_IPCW,
    estimate_competing_risks,
    parse_horizons_arg,
)
from .diagnostics import diagnose_cohort
from .protocol import load_and_validate_protocol, validate_protocol_command, validate_protocol_dict
from .synthetic import generate_synthetic_cohort
from .weighting import ALLOWED_WEIGHT_METHODS, parse_truncate_quantiles


def run_synthetic_pipeline(
    protocol_path: Path,
    *,
    seed: int,
    cases: int,
    weight_method: str,
    bootstrap_iterations: int,
    out_dir: Path,
    dgp_params_path: Path | None = None,
    truncate_quantiles: str | tuple[float, float] | None = None,
    horizons_days: str | list[float] | tuple[float, ...] | None = None,
    censoring_ipcw: str = "none",
    ci_level: float = 0.95,
) -> tuple[dict, int]:
    """Run end-to-end synthetic competing-risks pipeline with atomic sibling staging.

    Exit code contract:
      - Raises ContractError (CLI exit code 2) if protocol, arguments, or destination violate contract
        (publishes nothing).
      - Returns (result_dict, 3) if diagnostic gate blocks (atomically publishes normalized_protocol.json,
        attrition_table.csv, balance_diagnostics.csv, diagnostics_summary.json, and run_manifest.json;
        NEVER publishes weighted_cohort.csv or estimates.json).
      - Returns (result_dict, 0) if all gates pass (atomically publishes full protocol, diagnostics,
        CIF curves, estimates.json, audit_report.md, and SHA-256 run_manifest.json).
    """
    check_destination(out_dir)

    if not is_strict_int(seed) or not (0 <= seed <= MAX_SEED):
        raise ContractError("INVALID_SEED", f"seed must be an integer in 0..{MAX_SEED}")
    if not is_strict_int(cases) or not (1 <= cases <= MAX_SYNTH_CASES):
        raise ContractError("INVALID_CASE_COUNT", f"cases must be an integer in 1..{MAX_SYNTH_CASES}")
    if weight_method not in ALLOWED_WEIGHT_METHODS:
        raise ContractError(
            "INVALID_WEIGHT_METHOD",
            f"Unsupported weight_method '{weight_method}'. Allowed: {sorted(ALLOWED_WEIGHT_METHODS)}",
        )
    if not is_strict_int(bootstrap_iterations) or not (1 <= bootstrap_iterations <= MAX_BOOTSTRAP_ITERATIONS):
        raise ContractError(
            "INVALID_BOOTSTRAP_ITERATIONS",
            f"bootstrap_iterations must be an integer in 1..{MAX_BOOTSTRAP_ITERATIONS}",
        )
    if not is_strict_number(ci_level) or not (0.0 < float(ci_level) < 1.0):
        raise ContractError("INVALID_CI_LEVEL", "ci_level must be strictly between 0.0 and 1.0")
    if not isinstance(censoring_ipcw, str) or censoring_ipcw not in ALLOWED_CENSORING_IPCW:
        raise ContractError(
            "INVALID_CENSORING_IPCW",
            f"censoring_ipcw must be one of {sorted(ALLOWED_CENSORING_IPCW)}",
        )
    parsed_trunc = parse_truncate_quantiles(truncate_quantiles)

    raw_proto, protocol = load_and_validate_protocol(protocol_path)
    competing_codes = protocol["outcomes"]["competing_event_codes"]
    if (
        protocol["outcomes"]["competing_event_handling"] != "explicit_cif_and_subdistribution"
        or len(competing_codes) == 0
    ):
        raise ContractError(
            "COMPETING_RISKS_REQUIRED",
            "run-synthetic requires outcomes.competing_event_handling='explicit_cif_and_subdistribution' "
            "and non-empty competing_event_codes.",
        )

    num_arms = len(protocol["treatment_strategies"])
    if weight_method == "overlap" and num_arms != 2:
        raise ContractError(
            "INVALID_WEIGHT_METHOD",
            f"Two-arm 'overlap' requires 2 treatment strategies (protocol defines {num_arms}). "
            "Hint: Use 'generalized_overlap' or 'pairwise_overlap'.",
        )

    landmark_day = float(protocol["time_zero"]["landmark_day"])
    admin_end_day = float(protocol["followup"]["administrative_end_day"])
    horizons = parse_horizons_arg(
        horizons_days,
        default_horizons=protocol["followup"]["horizon_days"],
        landmark_day=landmark_day,
        admin_end_day=admin_end_day,
    )

    raw_dgp_bytes: bytes | None = None
    if dgp_params_path is not None:
        try:
            raw_dgp_bytes = dgp_params_path.read_bytes()
        except OSError as exc:
            raise ContractError("DGP_PARAMS_IO_ERROR", str(exc)) from exc

    commit, code_dirty = code_provenance()
    deps = dependency_versions()

    with tempfile.TemporaryDirectory(prefix=".emulatrx-tte-work-", dir=out_dir.parent) as work_str:
        work = Path(work_str)
        val_dir = work / "01_protocol"
        synth_dir = work / "02_synth"
        diag_dir = work / "03_diagnose"
        est_dir = work / "04_estimate"

        validate_protocol_command(protocol_path, val_dir)
        generate_synthetic_cohort(
            protocol_path,
            seed=seed,
            cases=cases,
            out_dir=synth_dir,
            dgp_params_path=dgp_params_path,
        )
        diag_res, diag_code = diagnose_cohort(
            protocol_path,
            synth_dir / "cohort.csv",
            weight_method=weight_method,
            out_dir=diag_dir,
            truncate_quantiles=parsed_trunc,
        )

        if diag_code == 3:
            blocked_files = [
                ("normalized_protocol.json", val_dir / "normalized_protocol.json"),
                ("attrition_table.csv", diag_dir / "attrition_table.csv"),
                ("balance_diagnostics.csv", diag_dir / "balance_diagnostics.csv"),
                ("diagnostics_summary.json", diag_dir / "diagnostics_summary.json"),
            ]
            blocked_paths = sorted(rel for rel, _ in blocked_files)

            def _populate_blocked(stage: Path) -> None:
                for rel, src in blocked_files:
                    shutil.copyfile(src, stage / rel)
                manifest = {
                    "status": "tte_synthetic_pipeline_blocked",
                    "gate_status": "blocked",
                    "activity": "tte_end_to_end_synthetic_competing_risks_pipeline",
                    "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
                    "prohibited_claims": list(PROHIBITED_CLAIMS),
                    "protocol_version": PROTOCOL_VERSION,
                    "simulation_or_data_label": protocol["simulation_or_data_label"],
                    "methodological_disclaimer": METHODOLOGICAL_DISCLAIMER,
                    "seed": seed,
                    "case_count": cases,
                    "weight_method": weight_method,
                    "bootstrap_iterations": bootstrap_iterations,
                    "ci_level": float(ci_level),
                    "censoring_ipcw": censoring_ipcw,
                    "horizons_days": horizons,
                    "truncation_quantiles": list(parsed_trunc) if parsed_trunc is not None else None,
                    "protocol_sha256": sha256_bytes(raw_proto),
                    "normalized_protocol_sha256": sha256_bytes(json_bytes(protocol)),
                    "dgp_params_sha256": sha256_bytes(raw_dgp_bytes) if raw_dgp_bytes is not None else None,
                    "code_commit": commit,
                    "code_dirty": code_dirty,
                    "dependency_versions": deps,
                    "output_paths": blocked_paths,
                    "output_sha256": {p: sha256_file(stage / p) for p in blocked_paths},
                    "run_at_utc": utc_now_iso(),
                }
                (stage / "run_manifest.json").write_bytes(json_bytes(manifest))

            atomic_publish(out_dir, _populate_blocked)
            return (
                {
                    "status": "tte_synthetic_pipeline_blocked",
                    "gate_status": "blocked",
                    "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
                    "simulation_or_data_label": protocol["simulation_or_data_label"],
                    "seed": seed,
                    "case_count": cases,
                    "weight_method": weight_method,
                    "gate_failures": diag_res["gate_failures"],
                    "output_path": str(out_dir),
                },
                3,
            )

        estimate_competing_risks(
            protocol_path,
            diagnostics_dir=diag_dir,
            seed=seed,
            bootstrap_iterations=bootstrap_iterations,
            out_dir=est_dir,
            horizons_days=horizons,
            censoring_ipcw=censoring_ipcw,
            ci_level=float(ci_level),
        )

        files_to_copy: list[tuple[str, Path]] = [
            ("normalized_protocol.json", val_dir / "normalized_protocol.json"),
            ("protocol_check_report.md", val_dir / "protocol_check_report.md"),
            ("dgp_truth.json", synth_dir / "dgp_truth.json"),
            ("PURPOSE.txt", synth_dir / "PURPOSE.txt"),
            ("attrition_table.csv", diag_dir / "attrition_table.csv"),
            ("balance_diagnostics.csv", diag_dir / "balance_diagnostics.csv"),
            ("diagnostics_summary.json", diag_dir / "diagnostics_summary.json"),
            ("weighted_cohort.csv", diag_dir / "weighted_cohort.csv"),
            ("estimates.json", est_dir / "estimates.json"),
            ("audit_report.md", est_dir / "audit_report.md"),
        ]
        for curve_file in sorted((est_dir / "curves").glob("*.csv")):
            files_to_copy.append((f"curves/{curve_file.name}", curve_file))

        output_paths = sorted(rel for rel, _ in files_to_copy)

        def _populate_passed(stage: Path) -> None:
            for rel, src in files_to_copy:
                dest = stage / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
            manifest = {
                "status": "tte_synthetic_pipeline_completed",
                "gate_status": "passed",
                "activity": "tte_end_to_end_synthetic_competing_risks_pipeline",
                "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
                "prohibited_claims": list(PROHIBITED_CLAIMS),
                "protocol_version": PROTOCOL_VERSION,
                "simulation_or_data_label": protocol["simulation_or_data_label"],
                "methodological_disclaimer": METHODOLOGICAL_DISCLAIMER,
                "seed": seed,
                "case_count": cases,
                "weight_method": weight_method,
                "bootstrap_iterations": bootstrap_iterations,
                "ci_level": float(ci_level),
                "censoring_ipcw": censoring_ipcw,
                "horizons_days": horizons,
                "truncation_quantiles": list(parsed_trunc) if parsed_trunc is not None else None,
                "protocol_sha256": sha256_bytes(raw_proto),
                "normalized_protocol_sha256": sha256_bytes(json_bytes(protocol)),
                "dgp_params_sha256": sha256_bytes(raw_dgp_bytes) if raw_dgp_bytes is not None else None,
                "code_commit": commit,
                "code_dirty": code_dirty,
                "dependency_versions": deps,
                "output_paths": output_paths,
                "output_sha256": {p: sha256_file(stage / p) for p in output_paths},
                "run_at_utc": utc_now_iso(),
            }
            (stage / "run_manifest.json").write_bytes(json_bytes(manifest))

        atomic_publish(out_dir, _populate_passed)

    return (
        {
            "status": "tte_synthetic_pipeline_completed",
            "gate_status": "passed",
            "source_mode": SOURCE_MODE_SYNTHETIC_ONLY,
            "simulation_or_data_label": protocol["simulation_or_data_label"],
            "seed": seed,
            "case_count": cases,
            "weight_method": weight_method,
            "bootstrap_iterations": bootstrap_iterations,
            "censoring_ipcw": censoring_ipcw,
            "horizons_days": horizons,
            "output_path": str(out_dir),
        },
        0,
    )


def build_simulated_protocol_and_dgp(
    *,
    target_rct: dict | None = None,
    pico: dict | None = None,
    sim_config: dict | None = None,
) -> tuple[dict, dict]:
    """Draft a normalized simulated_cohort protocol v1 and intuitive DGP parameters from Target RCT + UI settings."""
    cfg = sim_config or {}
    trct = target_rct or {}
    pico_d = pico or {}

    comp_label = str(
        cfg.get("comparator_label")
        or ((trct.get("comparator_arms") or [None])[0])
        or pico_d.get("comparator")
        or "对标对照治疗臂（Comparator）"
    ).strip()[:100]
    act_label = str(
        cfg.get("active_label")
        or ((trct.get("experimental_arms") or [None])[0])
        or pico_d.get("intervention")
        or "对标试验干预臂（Active）"
    ).strip()[:100]

    nct_id = str(trct.get("nct_id", "")).strip()
    rct_title = str(trct.get("brief_title", "")).strip()
    study_title = str(
        cfg.get("study_title")
        or (f"模拟预演（对标 {nct_id}）：{act_label} vs {comp_label}" if nct_id else f"无数据目标试验模拟预演：{act_label} vs {comp_label}")
    ).strip()[:200]

    horizons_raw = cfg.get("horizon_days") or trct.get("parsed_horizon_days") or [90.0, 180.0, 365.0]
    horizons = sorted({float(h) for h in horizons_raw if float(h) > 0.0})
    admin_end = float(cfg.get("administrative_end_day") or trct.get("parsed_admin_end_day") or max(540.0, horizons[-1] + 90.0))
    if admin_end <= horizons[-1]:
        admin_end = horizons[-1] + 90.0

    weight_method = str(cfg.get("weight_method", "overlap"))
    protocol_obj: dict = {
        "protocol_version": "1.0",
        "project_mode": SOURCE_MODE_SIMULATED_COHORT,
        "study_title": study_title,
        "simulation_or_data_label": "无数据虚拟队列立项可行性模拟预演（非真实患者数据、非临床疗效证据）",
        "eligibility": {
            "inclusion_criteria": [
                {
                    "id": "inc_sim_baseline_index",
                    "variable": "x_age_std",
                    "operator": ">=",
                    "value": -2.5,
                    "assessed_day_offset": 0,
                    "description": "基线标准化年龄/体能准入指数（模拟立项入排筛选）",
                }
            ],
            "exclusion_criteria": [],
        },
        "time_zero": {
            "anchor_event": "simulated_strategy_initiation",
            "day_offset": 0,
            "landmark_day": 0,
            "description": "虚拟队列目标治疗策略起始日（第 0 天）",
        },
        "treatment_strategies": [
            {"id": "arm_comparator", "label": comp_label, "role": "control"},
            {"id": "arm_active", "label": act_label, "role": "active"},
        ],
        "assignment_mode": "observational_baseline",
        "followup": {
            "horizon_days": horizons,
            "administrative_end_day": admin_end,
        },
        "outcomes": {
            "primary_event_code": 1,
            "competing_event_codes": [2],
            "censor_code": 0,
            "competing_event_handling": "explicit_cif_and_subdistribution",
        },
        "estimand": {
            "target_estimand": "ATO" if weight_method == "overlap" else "ATE",
            "effect_measures": ["RD", "RR_CIF", "sHR", "csHR"],
            "diagnostic_gates": {
                "max_abs_smd": float(cfg.get("max_abs_smd", 0.10)),
                "min_ess_ratio": float(cfg.get("min_ess_ratio", 0.25)),
                "positivity_epsilon": float(cfg.get("positivity_epsilon", 0.01)),
            },
        },
        "covariates": [
            {"name": "x_age_std", "type": "continuous", "assessed_day_offset": 0, "description": "模拟标准化基线年龄"},
            {"name": "x_biomarker_std", "type": "continuous", "assessed_day_offset": 0, "description": "模拟标准化器官功能/生物标志物"},
            {"name": "x_comorbidity_bin", "type": "binary", "assessed_day_offset": 0, "description": "模拟二元基础合并症"},
            {"name": "x_high_risk_bin", "type": "binary", "assessed_day_offset": 0, "description": "模拟二元高危临床分期"},
        ],
    }
    if nct_id and rct_title:
        po = (trct.get("primary_outcomes") or [{}])[0]
        protocol_obj["target_rct"] = {
            "nct_id": nct_id,
            "brief_title": rct_title,
            "record_url": str(trct.get("record_url", f"https://clinicaltrials.gov/study/{nct_id}")),
            "source": str(trct.get("source", "clinicaltrials_gov_api_v2")),
            "receipt_sha256": str(trct.get("receipt_sha256", "")),
            "primary_outcome_measure": str(po.get("measure", "")),
            "primary_outcome_timeframe": str(po.get("timeFrame", "")),
        }

    normalized = validate_protocol_dict(protocol_obj)

    confounding = str(cfg.get("confounding_strength", "medium")).lower()
    slope_scale = {"low": 0.25, "medium": 0.55, "high": 1.05, "extreme": 6.5}.get(confounding, 0.55)

    effect_preset = str(cfg.get("effect_direction", "protective_moderate")).lower()
    import math as _math
    true_hr_map = {
        "protective_strong": 0.65,
        "protective_moderate": 0.75,
        "null": 1.00,
        "harmful": 1.30,
    }
    true_cshr1 = float(cfg.get("true_primary_cshr") or true_hr_map.get(effect_preset, 0.75))
    true_cshr2 = float(cfg.get("true_competing_cshr") or (2.40 if effect_preset == "competing_divergence" else 0.95))
    if effect_preset == "competing_divergence":
        true_cshr1 = 1.00

    comp_level = str(cfg.get("competing_rate_level", "medium")).lower()
    bh_comp = {"low": 0.0007, "medium": 0.0016, "high": 0.0036}.get(comp_level, 0.0016)

    dgp_params = {
        "propensity_intercepts": [0.0, -0.10],
        "propensity_slopes": [
            [0.0, 0.0, 0.0, 0.0],
            [round(0.55 * slope_scale / 0.55, 4), round(-0.45 * slope_scale / 0.55, 4), round(0.45 * slope_scale / 0.55, 4), round(0.35 * slope_scale / 0.55, 4)],
        ],
        "baseline_hazard_primary": 0.0026,
        "baseline_hazard_competing": bh_comp,
        "random_censor_hazard": 0.0005,
        "pre_landmark_hazard_multiplier": 1.0,
        "log_cshr_primary_by_arm": [0.0, round(_math.log(max(true_cshr1, 0.05)), 6)],
        "log_cshr_competing_by_arm": [0.0, round(_math.log(max(true_cshr2, 0.05)), 6)],
        "outcome_covariate_log_hr_primary": [0.22, 0.18, 0.28, 0.32],
        "outcome_covariate_log_hr_competing": [0.28, 0.12, 0.35, 0.22],
    }
    return normalized, dgp_params


def run_simulated_cohort_pipeline(
    protocol_path: Path,
    *,
    seed: int,
    cases: int,
    weight_method: str = "overlap",
    bootstrap_iterations: int = 100,
    out_dir: Path,
    dgp_params_path: Path | None = None,
    horizons_days: str | list[float] | tuple[float, ...] | None = None,
    censoring_ipcw: str = "none",
    ci_level: float = 0.95,
) -> tuple[dict, int]:
    """Run no-data target-RCT simulated rehearsal pipeline (`source_mode = 'simulated_cohort'`)."""
    check_destination(out_dir)
    raw_proto, protocol = load_and_validate_protocol(protocol_path)
    landmark_day = float(protocol["time_zero"]["landmark_day"])
    admin_end_day = float(protocol["followup"]["administrative_end_day"])
    horizons = parse_horizons_arg(
        horizons_days,
        default_horizons=protocol["followup"]["horizon_days"],
        landmark_day=landmark_day,
        admin_end_day=admin_end_day,
    )
    commit, code_dirty = code_provenance()
    deps = dependency_versions()

    with tempfile.TemporaryDirectory(prefix=".emulatrx-sim-work-", dir=out_dir.parent) as work_str:
        work = Path(work_str)
        val_dir = work / "01_protocol"
        synth_dir = work / "02_synth"
        diag_dir = work / "03_diagnose"
        est_dir = work / "04_estimate"

        validate_protocol_command(protocol_path, val_dir)
        generate_synthetic_cohort(
            protocol_path,
            seed=seed,
            cases=cases,
            out_dir=synth_dir,
            dgp_params_path=dgp_params_path,
            id_prefix="SIM",
        )
        diag_res, diag_code = diagnose_cohort(
            protocol_path,
            synth_dir / "cohort.csv",
            weight_method=weight_method,
            out_dir=diag_dir,
        )
        if diag_code == 3:
            blocked_files = [
                ("normalized_protocol.json", val_dir / "normalized_protocol.json"),
                ("dgp_truth.json", synth_dir / "dgp_truth.json"),
                ("attrition_table.csv", diag_dir / "attrition_table.csv"),
                ("balance_diagnostics.csv", diag_dir / "balance_diagnostics.csv"),
                ("diagnostics_summary.json", diag_dir / "diagnostics_summary.json"),
            ]
            blocked_paths = sorted(rel for rel, _ in blocked_files)

            def _pop_blk(stage: Path) -> None:
                for rel, src in blocked_files:
                    shutil.copyfile(src, stage / rel)
                manifest = {
                    "status": "tte_simulated_pipeline_blocked",
                    "gate_status": "blocked",
                    "source_mode": SOURCE_MODE_SIMULATED_COHORT,
                    "prohibited_claims": list(SIMULATED_COHORT_CLAIMS),
                    "protocol_sha256": sha256_bytes(raw_proto),
                    "code_commit": commit,
                    "code_dirty": code_dirty,
                    "dependency_versions": deps,
                    "output_paths": blocked_paths,
                    "output_sha256": {p: sha256_file(stage / p) for p in blocked_paths},
                    "run_at_utc": utc_now_iso(),
                }
                (stage / "run_manifest.json").write_bytes(json_bytes(manifest))

            atomic_publish(out_dir, _pop_blk)
            return (
                {
                    "status": "tte_simulated_pipeline_blocked",
                    "gate_status": "blocked",
                    "source_mode": SOURCE_MODE_SIMULATED_COHORT,
                    "gate_failures": diag_res["gate_failures"],
                    "output_path": str(out_dir),
                },
                3,
            )

        estimate_competing_risks(
            protocol_path,
            diagnostics_dir=diag_dir,
            seed=seed,
            bootstrap_iterations=bootstrap_iterations,
            out_dir=est_dir,
            horizons_days=horizons,
            censoring_ipcw=censoring_ipcw,
            ci_level=ci_level,
            source_mode=SOURCE_MODE_SIMULATED_COHORT,
        )

        import json as _json
        dgp_truth = _json.loads((synth_dir / "dgp_truth.json").read_text(encoding="utf-8"))
        estimates = _json.loads((est_dir / "estimates.json").read_text(encoding="utf-8"))

        # Build simulation truth vs estimate recovery table
        ref_id = protocol["treatment_strategies"][0]["id"]
        act_id = protocol["treatment_strategies"][1]["id"]
        ana = dgp_truth["synthetic_dgp_parameters"]["analytical_reference_quantities"]
        contrast = next(iter(estimates["contrasts"].values()))
        recovery_rows = [
            {
                "metric": "主要结局原因特异风险比 csHR_1",
                "dgp_reference_truth": round(float(ana[act_id]["cshr_primary_vs_arm0"]), 4),
                "weighted_estimate": contrast["cause_specific_cox_primary"]["point"],
                "ci_lower": contrast["cause_specific_cox_primary"]["ci_lower"],
                "ci_upper": contrast["cause_specific_cox_primary"]["ci_upper"],
                "covered_in_95ci": bool(
                    contrast["cause_specific_cox_primary"]["ci_lower"] is not None
                    and contrast["cause_specific_cox_primary"]["ci_lower"]
                    <= float(ana[act_id]["cshr_primary_vs_arm0"])
                    <= contrast["cause_specific_cox_primary"]["ci_upper"]
                ),
            },
            {
                "metric": "竞争死亡原因特异风险比 csHR_2",
                "dgp_reference_truth": round(float(ana[act_id]["cshr_competing_vs_arm0"]), 4),
                "weighted_estimate": contrast["cause_specific_cox_competing"]["point"],
                "ci_lower": contrast["cause_specific_cox_competing"]["ci_lower"],
                "ci_upper": contrast["cause_specific_cox_competing"]["ci_upper"],
                "covered_in_95ci": bool(
                    contrast["cause_specific_cox_competing"]["ci_lower"] is not None
                    and contrast["cause_specific_cox_competing"]["ci_lower"]
                    <= float(ana[act_id]["cshr_competing_vs_arm0"])
                    <= contrast["cause_specific_cox_competing"]["ci_upper"]
                ),
            },
        ]
        for h_val in horizons:
            hk = f"{h_val:g}"
            if hk in ana[act_id]["horizons_conditional_on_x0_and_landmark_survival"] and hk in contrast["horizons"]:
                true_rd_x0 = (
                    ana[act_id]["horizons_conditional_on_x0_and_landmark_survival"][hk]["cif_primary"]
                    - ana[ref_id]["horizons_conditional_on_x0_and_landmark_survival"][hk]["cif_primary"]
                )
                h_est = contrast["horizons"][hk]
                recovery_rows.append(
                    {
                        "metric": f"第 {hk} 天主要结局累计发生率差 RD_1({hk}d) [参考 X=0 基准]",
                        "dgp_reference_truth": round(float(true_rd_x0), 4),
                        "weighted_estimate": h_est["rd_primary"],
                        "ci_lower": h_est["rd_primary_ci_lower"],
                        "ci_upper": h_est["rd_primary_ci_upper"],
                        "covered_in_95ci": None,
                    }
                )
        estimates["simulation_recovery_table"] = recovery_rows
        (est_dir / "estimates.json").write_bytes(json_bytes(estimates))

        files_to_copy: list[tuple[str, Path]] = [
            ("normalized_protocol.json", val_dir / "normalized_protocol.json"),
            ("protocol_check_report.md", val_dir / "protocol_check_report.md"),
            ("cohort.csv", synth_dir / "cohort.csv"),
            ("dgp_truth.json", synth_dir / "dgp_truth.json"),
            ("PURPOSE.txt", synth_dir / "PURPOSE.txt"),
            ("attrition_table.csv", diag_dir / "attrition_table.csv"),
            ("balance_diagnostics.csv", diag_dir / "balance_diagnostics.csv"),
            ("diagnostics_summary.json", diag_dir / "diagnostics_summary.json"),
            ("weighted_cohort.csv", diag_dir / "weighted_cohort.csv"),
            ("estimates.json", est_dir / "estimates.json"),
            ("audit_report.md", est_dir / "audit_report.md"),
        ]
        for curve_file in sorted((est_dir / "curves").glob("*.csv")):
            files_to_copy.append((f"curves/{curve_file.name}", curve_file))
        output_paths = sorted(rel for rel, _ in files_to_copy)

        def _pop_ok(stage: Path) -> None:
            for rel, src in files_to_copy:
                dest = stage / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
            manifest = {
                "status": "tte_simulated_pipeline_completed",
                "gate_status": "passed",
                "source_mode": SOURCE_MODE_SIMULATED_COHORT,
                "prohibited_claims": list(SIMULATED_COHORT_CLAIMS),
                "seed": seed,
                "case_count": cases,
                "weight_method": weight_method,
                "bootstrap_iterations": bootstrap_iterations,
                "horizons_days": horizons,
                "code_commit": commit,
                "code_dirty": code_dirty,
                "dependency_versions": deps,
                "output_paths": output_paths,
                "output_sha256": {p: sha256_file(stage / p) for p in output_paths},
                "run_at_utc": utc_now_iso(),
            }
            (stage / "run_manifest.json").write_bytes(json_bytes(manifest))

        atomic_publish(out_dir, _pop_ok)

    return (
        {
            "status": "tte_simulated_pipeline_completed",
            "gate_status": "passed",
            "source_mode": SOURCE_MODE_SIMULATED_COHORT,
            "seed": seed,
            "case_count": cases,
            "weight_method": weight_method,
            "horizons_days": horizons,
            "output_path": str(out_dir),
        },
        0,
    )


def run_user_cohort_pipeline(
    cohort_path: Path,
    mapping: dict,
    *,
    out_dir: Path,
    target_rct: dict | None = None,
    seed: int = 42,
    bootstrap_iterations: int = 100,
    weight_method: str | None = None,
    censoring_ipcw: str = "none",
    ci_level: float = 0.95,
) -> tuple[dict, int]:
    """Run end-to-end user-uploaded cohort TTE pipeline (`source_mode = 'user_cohort'`)."""
    check_destination(out_dir)
    chosen_weight = str(weight_method or mapping.get("weight_method") or "overlap")
    if chosen_weight not in ALLOWED_WEIGHT_METHODS:
        raise ContractError("INVALID_WEIGHT_METHOD", f"Unsupported weight_method '{chosen_weight}'")

    commit, code_dirty = code_provenance()
    deps = dependency_versions()

    with tempfile.TemporaryDirectory(prefix=".emulatrx-user-work-", dir=out_dir.parent) as work_str:
        work = Path(work_str)
        ingest_dir = work / "01_ingest"
        val_dir = work / "02_protocol"
        diag_dir = work / "03_diagnose"
        est_dir = work / "04_estimate"

        ingest_res = ingest_user_cohort(cohort_path, mapping, out_dir=ingest_dir, target_rct=target_rct)
        proto_path = ingest_dir / "normalized_protocol.json"
        std_cohort_path = ingest_dir / "standardized_cohort.csv"
        ingest_manifest_path = ingest_dir / "ingest_manifest.json"

        validate_protocol_command(proto_path, val_dir)
        diag_res, diag_code = diagnose_cohort(
            proto_path,
            std_cohort_path,
            weight_method=chosen_weight,
            out_dir=diag_dir,
        )

        if diag_code == 3:
            blocked_files = [
                ("normalized_protocol.json", val_dir / "normalized_protocol.json"),
                ("ingest_manifest.json", ingest_manifest_path),
                ("ingest_attrition.csv", ingest_dir / "ingest_attrition.csv"),
                ("attrition_table.csv", diag_dir / "attrition_table.csv"),
                ("balance_diagnostics.csv", diag_dir / "balance_diagnostics.csv"),
                ("diagnostics_summary.json", diag_dir / "diagnostics_summary.json"),
            ]
            blocked_paths = sorted(rel for rel, _ in blocked_files)

            def _pop_user_blk(stage: Path) -> None:
                for rel, src in blocked_files:
                    shutil.copyfile(src, stage / rel)
                manifest = {
                    "status": "tte_user_cohort_pipeline_blocked",
                    "gate_status": "blocked",
                    "source_mode": SOURCE_MODE_USER_COHORT,
                    "prohibited_claims": list(USER_COHORT_CAUSAL_CLAIMS),
                    "causal_assumptions": list(OBSERVATIONAL_CAUSAL_ASSUMPTIONS),
                    "input_sha256": ingest_res["input_sha256"],
                    "weight_method": chosen_weight,
                    "code_commit": commit,
                    "code_dirty": code_dirty,
                    "dependency_versions": deps,
                    "output_paths": blocked_paths,
                    "output_sha256": {p: sha256_file(stage / p) for p in blocked_paths},
                    "run_at_utc": utc_now_iso(),
                }
                (stage / "run_manifest.json").write_bytes(json_bytes(manifest))

            atomic_publish(out_dir, _pop_user_blk)
            return (
                {
                    "status": "tte_user_cohort_pipeline_blocked",
                    "gate_status": "blocked",
                    "source_mode": SOURCE_MODE_USER_COHORT,
                    "input_sha256": ingest_res["input_sha256"],
                    "raw_row_count": ingest_res["raw_row_count"],
                    "eligible_row_count": ingest_res["eligible_row_count"],
                    "gate_failures": diag_res["gate_failures"],
                    "output_path": str(out_dir),
                },
                3,
            )

        estimate_competing_risks(
            proto_path,
            diagnostics_dir=diag_dir,
            seed=seed,
            bootstrap_iterations=bootstrap_iterations,
            out_dir=est_dir,
            censoring_ipcw=censoring_ipcw,
            ci_level=ci_level,
            source_mode=SOURCE_MODE_USER_COHORT,
            ingest_manifest_path=ingest_manifest_path,
        )

        files_to_copy: list[tuple[str, Path]] = [
            ("normalized_protocol.json", val_dir / "normalized_protocol.json"),
            ("protocol_check_report.md", val_dir / "protocol_check_report.md"),
            ("ingest_manifest.json", ingest_manifest_path),
            ("ingest_attrition.csv", ingest_dir / "ingest_attrition.csv"),
            ("standardized_cohort.csv", std_cohort_path),
            ("attrition_table.csv", diag_dir / "attrition_table.csv"),
            ("balance_diagnostics.csv", diag_dir / "balance_diagnostics.csv"),
            ("diagnostics_summary.json", diag_dir / "diagnostics_summary.json"),
            ("weighted_cohort.csv", diag_dir / "weighted_cohort.csv"),
            ("estimates.json", est_dir / "estimates.json"),
            ("audit_report.md", est_dir / "audit_report.md"),
        ]
        for curve_file in sorted((est_dir / "curves").glob("*.csv")):
            files_to_copy.append((f"curves/{curve_file.name}", curve_file))
        output_paths = sorted(rel for rel, _ in files_to_copy)

        def _pop_user_ok(stage: Path) -> None:
            for rel, src in files_to_copy:
                dest = stage / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
            manifest = {
                "status": "tte_user_cohort_pipeline_completed",
                "gate_status": "passed",
                "source_mode": SOURCE_MODE_USER_COHORT,
                "prohibited_claims": list(USER_COHORT_CAUSAL_CLAIMS),
                "causal_assumptions": list(OBSERVATIONAL_CAUSAL_ASSUMPTIONS),
                "input_sha256": ingest_res["input_sha256"],
                "raw_row_count": ingest_res["raw_row_count"],
                "eligible_row_count": ingest_res["eligible_row_count"],
                "weight_method": chosen_weight,
                "seed": seed,
                "bootstrap_iterations": bootstrap_iterations,
                "code_commit": commit,
                "code_dirty": code_dirty,
                "dependency_versions": deps,
                "output_paths": output_paths,
                "output_sha256": {p: sha256_file(stage / p) for p in output_paths},
                "run_at_utc": utc_now_iso(),
            }
            (stage / "run_manifest.json").write_bytes(json_bytes(manifest))

        atomic_publish(out_dir, _pop_user_ok)

    return (
        {
            "status": "tte_user_cohort_pipeline_completed",
            "gate_status": "passed",
            "source_mode": SOURCE_MODE_USER_COHORT,
            "input_sha256": ingest_res["input_sha256"],
            "raw_row_count": ingest_res["raw_row_count"],
            "eligible_row_count": ingest_res["eligible_row_count"],
            "weight_method": chosen_weight,
            "output_path": str(out_dir),
        },
        0,
    )
