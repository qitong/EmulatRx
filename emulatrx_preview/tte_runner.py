"""Bounded, restart-aware orchestration of synthetic TTE commands."""

import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

from emulatrx_tte import validate_protocol_dict
from emulatrx_tte._common import ContractError
from .tte_screen import screen_cohort, validate_screening_criteria


RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
MAX_PROTOCOL_BYTES = 16 * 1024
MAX_CASES = 5000
MAX_BOOTSTRAP = 300
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
RETENTION_SECONDS = 24 * 60 * 60
MAX_RUNS = 100
RUN_TIMEOUT_SECONDS = 90
MCP_CHUNK_BYTES = 16 * 1024


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _csv_rows(path, limit=None):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader) if limit is None else list(row for _, row in zip(range(limit), reader))
        return reader.fieldnames or [], rows


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class TTERunner:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._slot = threading.BoundedSemaphore(1)
        self._maintenance = threading.Lock()
        self.cleanup(startup=True)

    def _path(self, run_id):
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ContractError("INVALID_RUN_ID", "invalid run ID")
        path = self.root / run_id
        if path.is_symlink() or not path.is_dir() or time.time() - path.stat().st_mtime > RETENTION_SECONDS:
            raise ContractError("RUN_NOT_FOUND", "run expired or unavailable")
        return path

    def _status(self, path):
        return json.loads((path / "status.json").read_text(encoding="utf-8"))

    def _write_status(self, path, status):
        temp = path / ".status-new.json"
        temp.write_bytes(_bytes(status))
        temp.replace(path / "status.json")

    def cleanup(self, startup=False):
        with self._maintenance:
            now = time.time()
            retained = []
            for path in self.root.iterdir():
                if path.is_symlink():
                    path.unlink()
                    continue
                if not path.is_dir() or not RUN_ID.fullmatch(path.name):
                    continue
                if now - path.stat().st_mtime > RETENTION_SECONDS:
                    shutil.rmtree(path)
                    continue
                try:
                    status = self._status(path)
                except (OSError, ValueError):
                    shutil.rmtree(path)
                    continue
                if startup and status["state"] in ("queued", "running"):
                    status.update(state="interrupted", error="Server restarted before this run finished")
                    self._write_status(path, status)
                retained.append(path)
            for path in sorted(retained, key=lambda p: p.stat().st_mtime)[:-MAX_RUNS]:
                shutil.rmtree(path)

    def validate(self, protocol):
        if not isinstance(protocol, dict):
            raise ContractError("INVALID_TYPE", "protocol must be a JSON object")
        try:
            raw = _bytes(protocol)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ContractError("INVALID_PROTOCOL", "protocol is not finite JSON") from exc
        if len(raw) > MAX_PROTOCOL_BYTES:
            raise ContractError("PROTOCOL_TOO_LARGE", "protocol exceeds 16 KiB service limit")
        normalized = validate_protocol_dict(protocol)
        validate_screening_criteria(normalized)
        if len(normalized["treatment_strategies"]) != 2 or normalized["estimand"]["target_estimand"] != "ATO":
            raise ContractError("UNSUPPORTED_PREVIEW_DESIGN", "preview supports two-arm ATO overlap designs")
        if (normalized["outcomes"]["competing_event_handling"] != "explicit_cif_and_subdistribution"
                or not normalized["outcomes"]["competing_event_codes"]):
            raise ContractError("UNSUPPORTED_PREVIEW_DESIGN", "preview requires an explicit competing event")
        if not {"RD", "RR_CIF", "sHR", "csHR"}.issubset(normalized["estimand"]["effect_measures"]):
            raise ContractError("UNSUPPORTED_PREVIEW_DESIGN", "preview requires RD, RR_CIF, sHR, and csHR")
        return normalized

    def start(self, protocol, *, cases=2000, seed=42, bootstrap_iterations=100, weight_method="overlap"):
        normalized = self.validate(protocol)
        if type(cases) is not int or not 100 <= cases <= MAX_CASES:
            raise ContractError("INVALID_CASE_COUNT", "preview cases must be 100..5000")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ContractError("INVALID_SEED", "seed must be 0..4294967295")
        if type(bootstrap_iterations) is not int or not 20 <= bootstrap_iterations <= MAX_BOOTSTRAP:
            raise ContractError("INVALID_BOOTSTRAP_ITERATIONS", "preview bootstrap iterations must be 20..300")
        if weight_method != "overlap" or len(normalized["treatment_strategies"]) != 2:
            raise ContractError("INVALID_WEIGHT_METHOD", "preview supports two-arm overlap weighting")
        if not self._slot.acquire(blocking=False):
            raise ContractError("BUSY", "another TTE run is in progress")
        try:
            self.cleanup()
            run_id = uuid.uuid4().hex
            path = self.root / run_id
            path.mkdir(mode=0o700)
            (path / "protocol.json").write_bytes(_bytes(protocol))
            status = {"run_id": run_id, "state": "queued", "kind": "synthetic_tte",
                      "case_count": cases, "seed": seed, "bootstrap_iterations": bootstrap_iterations,
                      "weight_method": weight_method, "gate_status": None, "stage": "queued",
                      "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            self._write_status(path, status)
            threading.Thread(target=self._execute, args=(path, status, normalized), daemon=True).start()
            return status
        except Exception:
            self._slot.release()
            raise

    def _command(self, command, deadline, env, permitted=(0,)):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("TTE run exceeded 90 seconds")
        result = subprocess.run(command, env=env, capture_output=True, text=True,
                                timeout=remaining, check=False)
        if result.returncode not in permitted:
            raise RuntimeError("TTE command failed")
        return result.returncode

    def _empty_eligibility_gate(self, path, screen):
        diagnostics = path / "diagnostics"
        diagnostics.mkdir(mode=0o700)
        payload = {"gate_status": "blocked", "initial_n": screen["eligible_rows"],
                   "gate_failures": [{"code": "NO_ELIGIBLE_SYNTHETIC_RISK_SET",
                                      "detail": "No eligible synthetic rows remain in both strategy arms after screening"}],
                   "weight_method": "overlap"}
        (diagnostics / "diagnostics_summary.json").write_bytes(_bytes(payload))
        with (diagnostics / "balance_diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
            stream.write("covariate,abs_smd_unweighted,abs_smd_weighted\n")
        with (diagnostics / "attrition_table.csv").open("w", encoding="utf-8", newline="") as stream:
            stream.write("step_id,remaining_n,excluded_n,excluded_primary_events,excluded_competing_events,arm_counts_json\n")

    def _execute(self, path, status, protocol):
        try:
            status = dict(status, state="running")

            def advance(stage):
                status["stage"] = stage
                self._write_status(path, status)

            advance("protocol_validation")
            deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
            env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            protocol_path = str(path / "protocol.json")
            prefix = [sys.executable, "-m", "emulatrx_tte"]
            self._command([*prefix, "validate-protocol", "--protocol", protocol_path,
                           "--out", str(path / "protocol_check")], deadline, env)
            advance("synthetic_generation")
            self._command([*prefix, "synth", "--protocol", protocol_path, "--seed", str(status["seed"]),
                           "--cases", str(status["case_count"]), "--out", str(path / "cohort")], deadline, env)
            cohort_file = path / "cohort" / "cohort.csv"
            cohort_manifest = json.loads((path / "cohort" / "run_manifest.json").read_text(encoding="utf-8"))
            if cohort_manifest["output_sha256"]["cohort.csv"] != _sha256(cohort_file):
                raise RuntimeError("generated cohort hash mismatch")
            advance("eligibility_screening")
            screen = screen_cohort(protocol, cohort_file, path / "screen")
            status["cohort_rows"] = screen["generated_rows"]
            status["eligible_rows"] = screen["eligible_rows"]
            status["screen_arm_counts"] = screen["arm_counts"]
            status["cohort_columns"], _ = _csv_rows(cohort_file, limit=1)
            eligible_file = path / "screen" / "eligible_cohort.csv"
            if screen["eligible_rows"] == 0 or any(count == 0 for count in screen["arm_counts"].values()):
                self._empty_eligibility_gate(path, screen)
                status["gate_status"] = "blocked"
            else:
                advance("diagnostics")
                diag_code = self._command([*prefix, "diagnose", "--protocol", protocol_path,
                                           "--cohort", str(eligible_file), "--weight-method", "overlap",
                                           "--out", str(path / "diagnostics")], deadline, env, permitted=(0, 3))
                status["gate_status"] = "blocked" if diag_code == 3 else "passed"
                diag_manifest = json.loads((path / "diagnostics" / "run_manifest.json").read_text(encoding="utf-8"))
                if diag_manifest["cohort_sha256"] != _sha256(eligible_file) or diag_manifest["gate_status"] != status["gate_status"]:
                    raise RuntimeError("diagnostic input or gate hash mismatch")
                if diag_code == 0:
                    advance("estimation")
                    self._command([*prefix, "estimate", "--protocol", protocol_path,
                                   "--diagnostics-dir", str(path / "diagnostics"), "--seed", str(status["seed"]),
                                   "--bootstrap-iterations", str(status["bootstrap_iterations"]),
                                   "--out", str(path / "estimates")], deadline, env)
                    est_manifest = json.loads((path / "estimates" / "run_manifest.json").read_text(encoding="utf-8"))
                    if est_manifest["diagnostics_manifest_sha256"] != _sha256(path / "diagnostics" / "run_manifest.json"):
                        raise RuntimeError("estimate diagnostic lineage mismatch")
            advance("artifact_audit")
            files = [p for p in path.rglob("*") if p.is_file()]
            total_bytes = sum(p.stat().st_size for p in files)
            if total_bytes > MAX_ARTIFACT_BYTES:
                raise RuntimeError("run artifacts exceed 8 MiB")
            protocol_manifest = json.loads((path / "protocol_check" / "run_manifest.json").read_text(encoding="utf-8"))
            status["artifact_bytes"] = total_bytes
            manifest = {"source_mode": "synthetic_only", "code_commit": protocol_manifest["code_commit"],
                        "code_dirty": protocol_manifest["code_dirty"],
                        "tte_dependency_commit": os.environ.get("EMULATRX_TTE_DEPENDENCY_COMMIT"),
                        "gate_status": status["gate_status"], "cohort_rows": screen["generated_rows"],
                        "eligible_rows": screen["eligible_rows"], "dependency_versions": protocol_manifest["dependency_versions"],
                        "output_sha256": {str(p.relative_to(path)): _sha256(p) for p in files
                                          if p.name != "status.json" and not p.name.startswith(".status-")}}
            (path / "preview_manifest.json").write_bytes(_bytes(manifest))
            status["state"] = "completed" if status["gate_status"] == "passed" else "blocked"
            status["stage"] = "complete"
            status["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._write_status(path, status)
        except (subprocess.TimeoutExpired, TimeoutError):
            self._write_status(path, dict(status, state="failed", error="运行超过 90 秒，请减少队列行数或 Bootstrap 次数。",
                                          finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds")))
        except ContractError as exc:
            self._write_status(path, dict(status, state="failed", error=f"{exc.code}: {exc}",
                                          finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds")))
        except (OSError, ValueError, RuntimeError):
            self._write_status(path, dict(status, state="failed", error="运行在当前阶段失败；请检查方案、门禁或资源配额。",
                                          finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds")))
        finally:
            self._slot.release()

    def get(self, run_id):
        path = self._path(run_id)
        status = self._status(path)
        if status["state"] not in ("completed", "blocked"):
            return status
        diagnostic = json.loads((path / "diagnostics" / "diagnostics_summary.json").read_text(encoding="utf-8"))
        _, screening_attrition = _csv_rows(path / "screen" / "eligibility_attrition.csv", limit=30)
        _, diagnosis_attrition = _csv_rows(path / "diagnostics" / "attrition_table.csv", limit=30)
        _, balance = _csv_rows(path / "diagnostics" / "balance_diagnostics.csv", limit=50)
        _, sample = _csv_rows(path / "cohort" / "cohort.csv", limit=8)
        manifest = json.loads((path / "preview_manifest.json").read_text(encoding="utf-8"))
        result = {**status,
                  "normalized_protocol": json.loads((path / "protocol_check" / "normalized_protocol.json").read_text(encoding="utf-8")),
                  "diagnostics": diagnostic, "attrition": screening_attrition + diagnosis_attrition,
                  "balance": balance, "sample_cases": sample, "preview_manifest": manifest,
                  "available_artifacts": self.artifacts(run_id)}
        if status["gate_status"] == "passed":
            estimates = json.loads((path / "estimates" / "estimates.json").read_text(encoding="utf-8"))
            result["estimates"] = {"co_presented_comparison_table": estimates["co_presented_comparison_table"],
                                   "arm_cif_summaries": estimates["arm_cif_summaries"],
                                   "methodological_warnings": estimates["methodological_warnings"]}
        return result

    def artifacts(self, run_id):
        path = self._path(run_id)
        status = self._status(path)
        if status["state"] not in ("completed", "blocked"):
            return []
        names = ["protocol.json", "preview_manifest.json",
                 "protocol_check/normalized_protocol.json", "protocol_check/protocol_check_report.md", "protocol_check/run_manifest.json",
                 "cohort/cohort.csv", "cohort/dgp_truth.json", "cohort/PURPOSE.txt", "cohort/run_manifest.json",
                 "screen/eligible_cohort.csv", "screen/eligibility_attrition.csv",
                 "diagnostics/attrition_table.csv", "diagnostics/balance_diagnostics.csv",
                 "diagnostics/diagnostics_summary.json", "diagnostics/weighted_cohort.csv", "diagnostics/run_manifest.json",
                 "estimates/estimates.json", "estimates/audit_report.md", "estimates/run_manifest.json"]
        curve_dir = path / "estimates" / "curves"
        if curve_dir.is_dir():
            names.extend(f"estimates/curves/{p.name}" for p in sorted(curve_dir.glob("*.csv")) if p.is_file() and not p.is_symlink())
        return [name for name in names if (path / name).is_file() and not (path / name).is_symlink()]

    def artifact_path(self, run_id, name):
        if name not in self.artifacts(run_id):
            raise ContractError("ARTIFACT_NOT_FOUND", "artifact is unavailable for this run")
        path = self._path(run_id) / name
        if path.is_symlink() or not path.is_file():
            raise ContractError("ARTIFACT_NOT_FOUND", "artifact is unavailable")
        return path

    def read_artifact(self, run_id, name, offset=0, limit=MCP_CHUNK_BYTES):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= MCP_CHUNK_BYTES:
            raise ContractError("INVALID_RANGE", "offset/limit outside allowed range")
        path = self.artifact_path(run_id, name)
        size = path.stat().st_size
        if offset > size:
            raise ContractError("INVALID_RANGE", "offset exceeds artifact size")
        with path.open("rb") as stream:
            stream.seek(offset)
            chunk = stream.read(limit)
        return {"name": name, "offset": offset, "total_bytes": size, "next_offset": offset + len(chunk),
                "encoding": "base64", "data": base64.b64encode(chunk).decode("ascii")}
