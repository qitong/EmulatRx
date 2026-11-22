"""Run with ``python -m emulatrx_demo validate`` from a clean clone."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from . import SCENARIO, SIMULATION_LABEL, TEMPLATE_VERSION


FIELDS = ("template_version", "scenario", "case_count", "seed", "simulation_label")
MAX_TEMPLATE_BYTES = 16 * 1024
MAX_SEED = 2**32 - 1


class ContractError(Exception):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("INVALID_JSON", f"duplicate key: {key}")
        result[key] = value
    return result


def _read_template(path):
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_TEMPLATE_BYTES + 1)
    except OSError as exc:
        raise ContractError("TEMPLATE_IO_ERROR", str(exc)) from exc
    if len(raw) > MAX_TEMPLATE_BYTES:
        raise ContractError("INVALID_JSON", "template exceeds 16 KiB")
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError("INVALID_JSON", str(exc)) from exc
    if not isinstance(obj, dict):
        raise ContractError("INVALID_TYPE", "template must be a JSON object")
    return raw, obj


def _validate(obj):
    missing = sorted(set(FIELDS) - obj.keys())
    if missing:
        raise ContractError("MISSING_FIELD", ", ".join(missing))
    unknown = sorted(obj.keys() - set(FIELDS))
    if unknown:
        raise ContractError("UNKNOWN_FIELD", ", ".join(unknown))
    for key in FIELDS:
        expected = int if key in ("case_count", "seed") else str
        if type(obj[key]) is not expected:
            raise ContractError("INVALID_TYPE", f"{key} must be {expected.__name__}")
    if obj["template_version"] != TEMPLATE_VERSION:
        raise ContractError("UNKNOWN_VERSION", obj["template_version"])
    if obj["scenario"] != SCENARIO:
        raise ContractError("INVALID_SCENARIO", obj["scenario"])
    if obj["case_count"] <= 0:
        raise ContractError("INVALID_CASE_COUNT", "case_count must be positive")
    if not 0 <= obj["seed"] <= MAX_SEED:
        raise ContractError("INVALID_SEED", f"seed must be in 0..{MAX_SEED}")
    if obj["simulation_label"] != SIMULATION_LABEL:
        raise ContractError("INVALID_SIMULATION_LABEL", "simulation_label must match the contract")
    return {key: obj[key] for key in FIELDS}


def _code_provenance():
    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
            timeout=5, check=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ContractError("CODE_VERSION_UNAVAILABLE", "git commit could not be resolved") from exc
    commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ContractError("CODE_VERSION_UNAVAILABLE", "git commit is invalid")
    return commit, bool(status.stdout.strip())


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def validate(template, out):
    # Check the destination before preparing any result files.
    if os.path.lexists(out):
        raise ContractError("OUTPUT_EXISTS", str(out))
    if not out.parent.is_dir():
        raise ContractError("OUTPUT_PARENT_MISSING", str(out.parent))
    raw, obj = _read_template(template)
    normalized = _validate(obj)
    commit, code_dirty = _code_provenance()
    manifest = {
        "status": "validated_only",
        "source_type": "fixed_synthetic_template",
        "activity": "template_validation",
        "simulation_label": SIMULATION_LABEL,
        "template_version": TEMPLATE_VERSION,
        "seed": normalized["seed"],
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "code_commit": commit,
        "code_dirty": code_dirty,
        "output_paths": ["normalized_template.json", "run_manifest.json"],
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    stage = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=".emulatrx-stage-", dir=out.parent))
        (stage / "normalized_template.json").write_bytes(_json_bytes(normalized))
        (stage / "run_manifest.json").write_bytes(_json_bytes(manifest))
        # Publish only after both complete files exist. A prior destination is never overwritten.
        if os.path.lexists(out):
            raise ContractError("OUTPUT_EXISTS", str(out))
        stage.rename(out)
        stage = None
    except OSError as exc:
        raise ContractError("OUTPUT_IO_ERROR", str(exc)) from exc
    finally:
        if stage is not None:
            shutil.rmtree(stage)
    return {"status": "validated_only", "simulation_label": SIMULATION_LABEL,
            "output_path": str(out)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m emulatrx_demo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("validate", help="validate a synthetic template")
    command.add_argument("--template", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate(args.template, args.out)
    except ContractError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
