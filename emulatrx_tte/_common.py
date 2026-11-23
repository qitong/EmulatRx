"""Shared error, provenance, strict JSON, and atomic directory staging helpers for emulatrx_tte."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile


PROTOCOL_VERSION = "1.0"
PACKAGE_VERSION = "0.2.0"
MAX_PROTOCOL_BYTES = 64 * 1024
MAX_SYNTH_CASES = 20000
MAX_BOOTSTRAP_ITERATIONS = 5000
MAX_SEED = 2**32 - 1

SOURCE_MODE_SYNTHETIC_ONLY = "synthetic_only"

PROHIBITED_CLAIMS = (
    "非真实患者数据 (not_real_patient_data: synthetic-only benchmark cohort; contains zero real-world patient records)",
    "非临床疗效证据 (not_clinical_efficacy_evidence: must never be cited or interpreted as clinical treatment efficacy or safety evidence)",
    "非因果实证证明 (not_causal_proof: observable-rule checks, propensity balance, and sensitivity E-values do not prove absence of unmeasured confounding or causal validity)",
)

METHODOLOGICAL_DISCLAIMER = (
    "Methodological boundary: observable-rule checking detects only structural/timestamp "
    "rule violations in supplied metadata/columns and cannot detect unmeasured confounding, "
    "measurement error, informative censoring, or prove causal validity."
)


class ContractError(Exception):
    """Deterministic contract / input validation error mapped to CLI exit code 2."""

    exit_code = 2

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


class GateBlockedError(Exception):
    """Fail-closed statistical / upstream diagnostic gate block mapped to CLI exit code 3."""

    exit_code = 3

    def __init__(self, code: str, detail: str, *, gate_failures: list[dict] | None = None):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.gate_failures = (
            list(gate_failures) if gate_failures is not None else [{"code": code, "detail": detail}]
        )


def is_strict_int(value: object) -> bool:
    return type(value) is int


def is_strict_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def check_destination(out: Path) -> None:
    if os.path.lexists(out):
        raise ContractError("OUTPUT_EXISTS", f"Destination already exists: {out}")
    if not out.parent.is_dir():
        raise ContractError("OUTPUT_PARENT_MISSING", f"Parent directory does not exist: {out.parent}")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("INVALID_JSON", f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_strict_json(
    path: Path,
    *,
    max_bytes: int = MAX_PROTOCOL_BYTES,
    io_error_code: str = "INPUT_IO_ERROR",
) -> tuple[bytes, dict]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
    except OSError as exc:
        raise ContractError(io_error_code, str(exc)) from exc
    if len(raw) > max_bytes:
        raise ContractError("INVALID_JSON", f"JSON input exceeds {max_bytes} bytes: {path}")
    try:
        obj = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"Non-finite JSON constant: {token}")),
        )
    except ContractError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError("INVALID_JSON", str(exc)) from exc
    if not isinstance(obj, dict):
        raise ContractError("INVALID_TYPE", "Root JSON value must be an object")
    return raw, obj


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def code_provenance() -> tuple[str, bool]:
    image_commit = os.environ.get("EMULATRX_CODE_COMMIT")
    if image_commit is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", image_commit):
            raise ContractError("CODE_VERSION_UNAVAILABLE", "Invalid image commit SHA")
        if os.environ.get("EMULATRX_CODE_DIRTY") != "false":
            raise ContractError("CODE_VERSION_UNAVAILABLE", "Image source state is not clean")
        return image_commit, False
    root = Path(__file__).resolve().parents[1]
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ContractError("CODE_VERSION_UNAVAILABLE", "Git commit provenance could not be resolved") from exc
    commit = rev.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ContractError("CODE_VERSION_UNAVAILABLE", "Resolved Git commit SHA is invalid")
    return commit, bool(status.stdout.strip())


def dependency_versions(packages: tuple[str, ...] = ("numpy", "scipy", "pandas", "scikit-learn", "lifelines")) -> dict:
    external = {}
    for pkg in packages:
        try:
            external[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            external[pkg] = None
    return {
        "python": platform.python_version(),
        "emulatrx_tte": PACKAGE_VERSION,
        "external_packages": external,
    }


def atomic_publish(out: Path, populate_stage_fn) -> None:
    """Create a sibling staging directory, populate all files + run_manifest.json, and rename atomically."""
    check_destination(out)
    stage: Path | None = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=".emulatrx-tte-stage-", dir=out.parent))
        populate_stage_fn(stage)
        if os.path.lexists(out):
            raise ContractError("OUTPUT_EXISTS", f"Destination already exists: {out}")
        stage.rename(out)
        stage = None
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError("OUTPUT_IO_ERROR", str(exc)) from exc
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
