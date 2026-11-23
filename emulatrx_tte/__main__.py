"""Command-line entry point for emulatrx_tte (validate-protocol, synth, diagnose)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ._common import ContractError
from .diagnostics import diagnose_cohort
from .protocol import validate_protocol_command
from .synthetic import generate_synthetic_cohort
from .weighting import ALLOWED_WEIGHT_METHODS


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
    sy.add_argument("--seed", type=int, required=True)
    sy.add_argument("--cases", type=int, required=True)
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
        else:
            result, exit_code = diagnose_cohort(
                args.protocol,
                args.cohort,
                weight_method=args.weight_method,
                out_dir=args.out,
                truncate_quantiles=args.truncate_quantiles,
                covariate_day_offsets=args.covariate_offsets,
            )
    except ContractError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2

    if exit_code == 3:
        codes = ", ".join(f["code"] for f in result.get("gate_failures", []))
        print(f"DIAGNOSTIC_GATE_BLOCKED: {codes}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
