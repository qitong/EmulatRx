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
    PROHIBITED_CLAIMS,
    PROTOCOL_VERSION,
    SOURCE_MODE_SYNTHETIC_ONLY,
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
from .competing_risks import (
    ALLOWED_CENSORING_IPCW,
    estimate_competing_risks,
    parse_horizons_arg,
)
from .diagnostics import diagnose_cohort
from .protocol import load_and_validate_protocol, validate_protocol_command
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
