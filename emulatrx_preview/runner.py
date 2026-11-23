"""One bounded execution path shared by the browser and MCP tool."""

import csv
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

from emulatrx_demo.__main__ import _validate
from emulatrx_demo._common import ContractError
from emulatrx_demo.public_metadata import NCT_RE


RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
MAX_INPUT_BYTES = 16 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_RUNS = 100
RETENTION_SECONDS = 24 * 60 * 60
RUN_TIMEOUT_SECONDS = 25


class Runner:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._slot = threading.BoundedSemaphore(1)
        self._maintenance = threading.Lock()
        self.cleanup()

    def cleanup(self):
        """Discard expired and interrupted work, never following symlinks."""
        with self._maintenance:
            now = time.time()
            complete = []
            for path in self.root.iterdir():
                if path.is_symlink():
                    path.unlink()
                    continue
                if not path.is_dir():
                    continue
                if not RUN_ID.fullmatch(path.name):
                    # Do not touch files that were not created by this service.
                    continue
                complete_run = ((path / "output" / "run_manifest.json").is_file() or
                                (path / "output" / "source" / "metadata.json").is_file())
                if not complete_run or now - path.stat().st_mtime > RETENTION_SECONDS:
                    shutil.rmtree(path)
                else:
                    complete.append(path)
            for path in sorted(complete, key=lambda p: p.stat().st_mtime)[:-MAX_RUNS]:
                shutil.rmtree(path)

    def run(self, template):
        if not isinstance(template, dict):
            raise ContractError("INVALID_TYPE", "template must be a JSON object")
        _validate(template)
        raw = (json.dumps(template, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        if len(raw) > MAX_INPUT_BYTES:
            raise ContractError("INVALID_JSON", "template exceeds 16 KiB")
        if not self._slot.acquire(blocking=False):
            raise ContractError("BUSY", "another run is in progress")
        run_dir = None
        try:
            self.cleanup()
            run_id = uuid.uuid4().hex
            run_dir = self.root / run_id
            run_dir.mkdir(mode=0o700)
            (run_dir / "template.json").write_bytes(raw)
            command = [sys.executable, "-m", "emulatrx_demo", "demo", "--template", str(run_dir / "template.json"), "--out", str(run_dir / "output")]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=RUN_TIMEOUT_SECONDS, check=False)
            except subprocess.TimeoutExpired as exc:
                raise ContractError("TIMEOUT", "synthetic generation exceeded 25 seconds") from exc
            if result.returncode:
                # The CLI emits error codes, but its stderr can contain local paths.
                raise ContractError("RUN_FAILED", "synthetic generation failed")
            size = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
            if size > MAX_OUTPUT_BYTES:
                raise ContractError("OUTPUT_TOO_LARGE", "run exceeds 2 MiB")
            return self.get(run_id)
        except Exception:
            if run_dir is not None:
                shutil.rmtree(run_dir, ignore_errors=True)
            raise
        finally:
            self._slot.release()

    def get(self, run_id):
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ContractError("INVALID_RUN_ID", "invalid run id")
        path = self.root / run_id
        if path.is_symlink() or not path.is_dir() or time.time() - path.stat().st_mtime > RETENTION_SECONDS:
            raise ContractError("RUN_NOT_FOUND", "run expired or unavailable")
        output = path / "output"
        receipt_path = output / "source" / "metadata.json"
        if receipt_path.is_file():
            return {"run_id": run_id, "kind": "public_source_receipt",
                    "receipt": json.loads(receipt_path.read_text(encoding="utf-8")), "retention_hours": 24}
        with (output / "results" / "summary.json").open(encoding="utf-8") as stream:
            summary = json.load(stream)
        with (output / "run_manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        with (output / "synthetic" / "cases.csv").open(encoding="utf-8", newline="") as stream:
            sample = list(row for _, row in zip(range(10), csv.DictReader(stream)))
        return {"run_id": run_id, "kind": "synthetic_demo", "summary": summary, "manifest": manifest, "sample_cases": sample,
                "retention_hours": 24}

    def receipt(self, nct_id):
        if not isinstance(nct_id, str) or not NCT_RE.fullmatch(nct_id):
            raise ContractError("INVALID_NCT_ID", "expected NCT followed by eight digits")
        if not self._slot.acquire(blocking=False):
            raise ContractError("BUSY", "another run is in progress")
        run_dir = None
        try:
            self.cleanup()
            run_id = uuid.uuid4().hex
            run_dir = self.root / run_id
            run_dir.mkdir(mode=0o700)
            command = [sys.executable, "-m", "emulatrx_demo", "metadata", "--nct-id", nct_id,
                       "--out", str(run_dir / "output")]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=40, check=False)
            except subprocess.TimeoutExpired as exc:
                raise ContractError("TIMEOUT", "source receipt exceeded 40 seconds") from exc
            if result.returncode:
                raise ContractError("RECEIPT_FAILED", "public source receipt failed")
            size = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
            if size > MAX_OUTPUT_BYTES:
                raise ContractError("OUTPUT_TOO_LARGE", "run exceeds 2 MiB")
            return self.get(run_id)
        except Exception:
            if run_dir is not None:
                shutil.rmtree(run_dir, ignore_errors=True)
            raise
        finally:
            self._slot.release()

    def cases_path(self, run_id):
        result = self.get(run_id)
        if result["kind"] != "synthetic_demo":
            raise ContractError("RUN_NOT_FOUND", "CSV unavailable")
        return self.root / run_id / "output" / "synthetic" / "cases.csv"
