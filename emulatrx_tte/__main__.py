"""Command-line entry point for emulatrx_tte (validate-protocol, synth, diagnose, estimate, run-synthetic)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ._common import ContractError, GateBlockedError
from .ccw import (
    generate_synthetic_ccw_cohort,
    run_ccw_synthetic_pipeline,
    validate_ccw_protocol,
)
from .competing_risks import ALLOWED_CENSORING_IPCW, estimate_competing_risks
from .diagnostics import diagnose_cohort
from .pipeline import run_synthetic_pipeline
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
            )
            exit_code = 0
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
