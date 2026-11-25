"""Command-line entry point for emulatrx_tte (validate-protocol, synth, diagnose, estimate, run-synthetic, run-simulated, run-user-cohort, validate-ccw, synth-ccw, run-ccw-synthetic, benchmark-rct)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ._common import ALLOWED_SOURCE_MODES, ContractError, GateBlockedError, read_strict_json
from .ccw import (
    generate_synthetic_ccw_cohort,
    run_ccw_synthetic_pipeline,
    validate_ccw_protocol,
)
from .competing_risks import ALLOWED_CENSORING_IPCW, estimate_competing_risks
from .diagnostics import diagnose_cohort
from .pipeline import (
    run_simulated_cohort_pipeline,
    run_synthetic_pipeline,
    run_user_cohort_pipeline,
)
from .protocol import validate_protocol_command
from .synthetic import generate_synthetic_cohort
from .weighting import ALLOWED_WEIGHT_METHODS


def _parse_strict_cli_int(raw: str) -> int:
    try:
        return int(raw, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer value: '{raw}'") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m emulatrx_tte")
    subparsers = parser.add_subparsers(dest="command", required=True)

    vp = subparsers.add_parser(
        "validate-protocol",
        help="Validate a TTE v1 protocol JSON and check observable structural/timestamp rules",
    )
    vp.add_argument("--protocol", type=Path, required=True)
    vp.add_argument("--out", type=Path, required=True)

    sy = subparsers.add_parser(
        "synth",
        help="Generate a seeded, purely synthetic competing-risks / multi-arm TTE cohort",
    )
    sy.add_argument("--protocol", type=Path, required=True)
    sy.add_argument("--seed", type=_parse_strict_cli_int, required=True)
    sy.add_argument("--cases", type=_parse_strict_cli_int, required=True)
    sy.add_argument("--dgp-params", type=Path, default=None)
    sy.add_argument("--out", type=Path, required=True)

    dg = subparsers.add_parser(
        "diagnose",
        help="Run landmark slicing, propensity/overlap weighting, balance/ESS diagnostics, and gate checks",
    )
    dg.add_argument("--protocol", type=Path, required=True)
    dg.add_argument("--cohort", type=Path, required=True)
    dg.add_argument(
        "--weight-method",
        required=True,
        choices=sorted(ALLOWED_WEIGHT_METHODS),
    )
    dg.add_argument("--truncate-quantiles", default=None)
    dg.add_argument("--covariate-offsets", type=Path, default=None)
    dg.add_argument("--out", type=Path, required=True)

    es = subparsers.add_parser(
        "estimate",
        help="Run fail-closed weighted competing-risks survival analysis (CIF, Fine-Gray sHR, Cox csHR, E-value)",
    )
    es.add_argument("--protocol", type=Path, required=True)
    es.add_argument("--diagnostics-dir", type=Path, default=None)
    es.add_argument("--weighted-cohort", type=Path, default=None)
    es.add_argument("--seed", type=_parse_strict_cli_int, required=True)
    es.add_argument("--bootstrap-iterations", type=_parse_strict_cli_int, required=True)
    es.add_argument("--horizons", default=None)
    es.add_argument(
        "--censoring-ipcw",
        default="none",
        choices=sorted(ALLOWED_CENSORING_IPCW),
    )
    es.add_argument("--ci-level", type=float, default=0.95)
    es.add_argument(
        "--source-mode",
        default="synthetic_only",
        choices=sorted(ALLOWED_SOURCE_MODES),
    )
    es.add_argument("--ingest-manifest", type=Path, default=None)
    es.add_argument("--out", type=Path, required=True)

    rs = subparsers.add_parser(
        "run-synthetic",
        help="Run end-to-end offline synthetic TTE competing-risks pipeline with SHA-256 audit bundle",
    )
    rs.add_argument("--protocol", type=Path, required=True)
    rs.add_argument("--seed", type=_parse_strict_cli_int, required=True)
    rs.add_argument("--cases", type=_parse_strict_cli_int, required=True)
    rs.add_argument(
        "--weight-method",
        required=True,
        choices=sorted(ALLOWED_WEIGHT_METHODS),
    )
    rs.add_argument("--bootstrap-iterations", type=_parse_strict_cli_int, required=True)
    rs.add_argument("--dgp-params", type=Path, default=None)
    rs.add_argument("--truncate-quantiles", default=None)
    rs.add_argument("--horizons", default=None)
    rs.add_argument(
        "--censoring-ipcw",
        default="none",
        choices=sorted(ALLOWED_CENSORING_IPCW),
    )
    rs.add_argument("--ci-level", type=float, default=0.95)
    rs.add_argument("--out", type=Path, required=True)

    rsim = subparsers.add_parser(
        "run-simulated",
        help="Run no-data target-RCT simulated cohort TTE rehearsal pipeline (source_mode=simulated_cohort)",
    )
    rsim.add_argument("--protocol", type=Path, required=True)
    rsim.add_argument("--seed", type=_parse_strict_cli_int, required=True)
    rsim.add_argument("--cases", type=_parse_strict_cli_int, required=True)
    rsim.add_argument("--weight-method", default="overlap", choices=sorted(ALLOWED_WEIGHT_METHODS))
    rsim.add_argument("--bootstrap-iterations", type=_parse_strict_cli_int, default=100)
    rsim.add_argument("--dgp-params", type=Path, default=None)
    rsim.add_argument("--horizons", default=None)
    rsim.add_argument("--censoring-ipcw", default="none", choices=sorted(ALLOWED_CENSORING_IPCW))
    rsim.add_argument("--ci-level", type=float, default=0.95)
    rsim.add_argument("--out", type=Path, required=True)

    ruc = subparsers.add_parser(
        "run-user-cohort",
        help="Run user-uploaded cohort ingestion, attrition, diagnostics, and TTE estimation (source_mode=user_cohort)",
    )
    ruc.add_argument("--cohort", type=Path, required=True)
    ruc.add_argument("--mapping", type=Path, required=True, help="Path to ColumnMappingSpec JSON file")
    ruc.add_argument("--target-rct", type=Path, default=None, help="Optional Target RCT JSON file")
    ruc.add_argument("--seed", type=_parse_strict_cli_int, default=42)
    ruc.add_argument("--bootstrap-iterations", type=_parse_strict_cli_int, default=100)
    ruc.add_argument("--weight-method", default=None, choices=sorted(ALLOWED_WEIGHT_METHODS))
    ruc.add_argument("--censoring-ipcw", default="none", choices=sorted(ALLOWED_CENSORING_IPCW))
    ruc.add_argument("--ci-level", type=float, default=0.95)
    ruc.add_argument("--out", type=Path, required=True)

    vc = subparsers.add_parser(
        "validate-ccw",
        help="Validate a tte-ccw-v1 grace-period CCW protocol JSON and anti-time-travel rules",
    )
    vc.add_argument("--protocol", type=Path, required=True)
    vc.add_argument("--out", type=Path, required=True)

    sc = subparsers.add_parser(
        "synth-ccw",
        help="Generate a seeded synthetic delayed-initiation competing-risks cohort and counterfactual truth",
    )
    sc.add_argument("--protocol", type=Path, required=True)
    sc.add_argument("--seed", type=_parse_strict_cli_int, required=True)
    sc.add_argument("--cases", type=_parse_strict_cli_int, required=True)
    sc.add_argument("--dgp-params", type=Path, default=None)
    sc.add_argument("--out", type=Path, required=True)

    rc = subparsers.add_parser(
        "run-ccw-synthetic",
        help="Run end-to-end synthetic bilateral grace-period CCW benchmark with HTML comparison report",
    )
    rc.add_argument("--protocol", type=Path, required=True)
    rc.add_argument("--seed", type=_parse_strict_cli_int, required=True)
    rc.add_argument("--cases", type=_parse_strict_cli_int, required=True)
    rc.add_argument("--bootstrap-iterations", type=_parse_strict_cli_int, default=200)
    rc.add_argument("--dgp-params", type=Path, default=None)
    rc.add_argument("--ci-level", type=float, default=0.95)
    rc.add_argument("--out", type=Path, required=True)

    br = subparsers.add_parser(
        "benchmark-rct",
        help="Benchmark a finished TTE against public RCT aggregates and optional negative-control diagnostics",
    )
    br.add_argument("--output-dir", type=Path, required=True)
    br.add_argument("--nct-id", default="NCT01243424")
    br.add_argument("--synthetic-demo", choices=("hidden_bias", "clean_null"), default=None)
    br.add_argument("--seed", type=_parse_strict_cli_int, default=20261125)
    br.add_argument("--n", type=_parse_strict_cli_int, default=2500)
    br.add_argument("--tte-result-json", type=Path, default=None)
    br.add_argument("--alignment-json", type=Path, default=None)
    br.add_argument("--study-json", type=Path, default=None)
    br.add_argument("--curated-summary-json", type=Path, default=None)
    br.add_argument("--cohort-csv", type=Path, default=None)
    br.add_argument("--treatment-col", default="treatment")
    br.add_argument("--outcome-col", default="outcome_event")
    br.add_argument("--weights-col", default="analysis_weight")
    br.add_argument("--negative-control-outcome-col", default=None)
    br.add_argument("--negative-control-time-col", default=None)

    args = parser.parse_args(argv)

    try:
        if args.command == "validate-protocol":
            result = validate_protocol_command(args.protocol, args.out)
            exit_code = 0
        elif args.command == "synth":
            result = generate_synthetic_cohort(
                args.protocol,
                seed=args.seed,
                cases=args.cases,
                out_dir=args.out,
                dgp_params_path=args.dgp_params,
            )
            exit_code = 0
        elif args.command == "diagnose":
            result, exit_code = diagnose_cohort(
                args.protocol,
                args.cohort,
                weight_method=args.weight_method,
                out_dir=args.out,
                truncate_quantiles=args.truncate_quantiles,
                covariate_day_offsets=args.covariate_offsets,
            )
        elif args.command == "estimate":
            result = estimate_competing_risks(
                args.protocol,
                diagnostics_dir=args.diagnostics_dir,
                weighted_cohort_path=args.weighted_cohort,
                seed=args.seed,
                bootstrap_iterations=args.bootstrap_iterations,
                out_dir=args.out,
                horizons_days=args.horizons,
                censoring_ipcw=args.censoring_ipcw,
                ci_level=args.ci_level,
                source_mode=args.source_mode,
                ingest_manifest_path=args.ingest_manifest,
            )
            exit_code = 0
        elif args.command == "run-simulated":
            result, exit_code = run_simulated_cohort_pipeline(
                args.protocol,
                seed=args.seed,
                cases=args.cases,
                weight_method=args.weight_method,
                bootstrap_iterations=args.bootstrap_iterations,
                out_dir=args.out,
                dgp_params_path=args.dgp_params,
                horizons_days=args.horizons,
                censoring_ipcw=args.censoring_ipcw,
                ci_level=args.ci_level,
            )
        elif args.command == "run-user-cohort":
            from ._common import read_strict_json
            _, mapping_obj = read_strict_json(args.mapping, io_error_code="MAPPING_IO_ERROR")
            rct_obj = None
            if args.target_rct is not None:
                _, rct_obj = read_strict_json(args.target_rct, io_error_code="TARGET_RCT_IO_ERROR")
            result, exit_code = run_user_cohort_pipeline(
                args.cohort,
                mapping_obj,
                out_dir=args.out,
                target_rct=rct_obj,
                seed=args.seed,
                bootstrap_iterations=args.bootstrap_iterations,
                weight_method=args.weight_method,
                censoring_ipcw=args.censoring_ipcw,
                ci_level=args.ci_level,
            )
        elif args.command == "validate-ccw":
            result = validate_ccw_protocol(args.protocol, args.out)
            exit_code = 0
        elif args.command == "synth-ccw":
            result = generate_synthetic_ccw_cohort(
                args.protocol,
                seed=args.seed,
                n_cases=args.cases,
                dgp_params=args.dgp_params,
                out_dir=args.out,
            )
            exit_code = 0
        elif args.command == "benchmark-rct":
            from .benchmarking import (
                execute_benchmark_request,
                is_benchmarking_enabled,
                run_synthetic_benchmark_demo,
                write_benchmark_report,
            )

            if not is_benchmarking_enabled():
                print("BENCHMARKING_FEATURE_DISABLED: benchmarking is turned off", file=sys.stderr)
                return 2
            if args.synthetic_demo:
                result = run_synthetic_benchmark_demo(
                    args.synthetic_demo,
                    nct_id=args.nct_id,
                    n=args.n,
                    seed=args.seed,
                )
            else:
                if args.tte_result_json is None or args.alignment_json is None:
                    raise ContractError(
                        "MISSING_BENCHMARK_INPUT",
                        "benchmark-rct requires --synthetic-demo or both --tte-result-json and --alignment-json",
                    )
                _, tte_result = read_strict_json(args.tte_result_json, max_bytes=2_000_000)
                _, alignment = read_strict_json(args.alignment_json, max_bytes=2_000_000)
                study_json = None
                curated = None
                if args.study_json is not None:
                    _, study_json = read_strict_json(args.study_json, max_bytes=2_000_000)
                if args.curated_summary_json is not None:
                    _, curated = read_strict_json(args.curated_summary_json, max_bytes=2_000_000)
                cohort = None
                weights = None
                if args.cohort_csv is not None:
                    import pandas as pd

                    cohort = pd.read_csv(args.cohort_csv)
                    if args.weights_col not in cohort.columns:
                        raise ContractError("INVALID_WEIGHTS", f"cohort is missing weights column {args.weights_col}")
                    weights = cohort[args.weights_col].to_numpy()
                result = execute_benchmark_request(
                    {
                        "nct_id": args.nct_id,
                        "tte_result": tte_result,
                        "alignment_audit": alignment,
                        "study_json": study_json,
                        "curated_published_summary": curated,
                        "treatment_col": args.treatment_col,
                        "outcome_col": args.outcome_col,
                        "negative_control_outcome_col": args.negative_control_outcome_col,
                        "negative_control_time_col": args.negative_control_time_col,
                    },
                    cohort=cohort,
                    weights=weights,
                )
            write_benchmark_report(result, args.output_dir)
            exit_code = 0
        elif args.command == "run-ccw-synthetic":
            result, exit_code = run_ccw_synthetic_pipeline(
                args.protocol,
                seed=args.seed,
                n_cases=args.cases,
                bootstrap_iterations=args.bootstrap_iterations,
                dgp_params=args.dgp_params,
                ci_level=args.ci_level,
                out_dir=args.out,
            )
        else:
            result, exit_code = run_synthetic_pipeline(
                args.protocol,
                seed=args.seed,
                cases=args.cases,
                weight_method=args.weight_method,
                bootstrap_iterations=args.bootstrap_iterations,
                out_dir=args.out,
                dgp_params_path=args.dgp_params,
                truncate_quantiles=args.truncate_quantiles,
                horizons_days=args.horizons,
                censoring_ipcw=args.censoring_ipcw,
                ci_level=args.ci_level,
            )
    except ContractError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2
    except GateBlockedError as exc:
        codes = ", ".join(f["code"] for f in exc.gate_failures)
        print(f"ESTIMATE_GATE_BLOCKED: {exc.code}: {exc} [{codes}]", file=sys.stderr)
        blocked_payload = {
            "status": "estimate_gate_blocked",
            "gate_status": "blocked",
            "error_code": exc.code,
            "gate_failures": exc.gate_failures,
        }
        print(json.dumps(blocked_payload, ensure_ascii=False, sort_keys=True))
        return 3

    if exit_code == 3:
        codes = ", ".join(f["code"] for f in result.get("gate_failures", []))
        print(f"DIAGNOSTIC_GATE_BLOCKED: {codes}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
