"""Bounded, restart-aware orchestration for legacy synthetic, simulated-cohort, and session-isolated user-cohort TTE runs."""

from __future__ import annotations

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

from emulatrx_tte import (
    build_simulated_protocol_and_dgp,
    interpret_cohort_profile,
    profile_cohort_dataframe,
    read_tabular_cohort_bytes,
    recommend_column_mapping,
    run_simulated_cohort_pipeline,
    run_user_cohort_pipeline,
    validate_protocol_dict,
    verify_and_enrich_target_rct,
)
from emulatrx_tte._common import (
    ContractError,
    MAX_USER_COHORT_BYTES,
    SOURCE_MODE_SIMULATED_COHORT,
    SOURCE_MODE_SYNTHETIC_ONLY,
    SOURCE_MODE_USER_COHORT,
)
from emulatrx_tte.llm_assist import (
    SAFE_CONTROLLED_CLINICAL_WORDS,
    extract_untrusted_Note_forbidden_tokens,
    is_whitelisted_clinical_category,
)
from .tte_screen import screen_cohort, validate_screening_criteria


RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
MAX_PROTOCOL_BYTES = 16 * 1024
MAX_CASES = 5000
MAX_BOOTSTRAP = 300
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
RETENTION_SECONDS = 24 * 60 * 60
MAX_RUNS = 100
MAX_SESSIONS = 60
RUN_TIMEOUT_SECONDS = 90
MCP_CHUNK_BYTES = 16 * 1024


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Atomically write bytes via a same-directory temporary file and `os.replace`."""
    tmp = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _collect_cohort_forbidden_values(df, profile: dict, user_note: str = "") -> list[str]:
    """Collect non-whitelisted string values from cohort columns and untrusted tokens from `user_note`."""
    seen: set[str] = set()
    ordered: list[str] = []

    def _add(tok: str) -> None:
        s = tok.strip()
        if len(s) < 3 or s.lower() in SAFE_CONTROLLED_CLINICAL_WORDS:
            return
        if s not in seen and len(ordered) < 2000:
            seen.add(s)
            ordered.append(s)

    id_col = profile.get("id_column_guess")
    if id_col and id_col in df.columns:
        for val in df[id_col].dropna().astype(str).iloc[:500]:
            _add(val)

    for col in df.columns:
        ser = df[col]
        if str(ser.dtype) in ("object", "string", "category") or str(ser.dtype).startswith("str"):
            vc = ser.dropna().astype(str).value_counts()
            for raw_val, cnt in list(vc.items())[:300]:
                val_s = str(raw_val).strip()
                if not is_whitelisted_clinical_category(val_s, int(cnt)):
                    _add(val_s)

    for tok in extract_untrusted_Note_forbidden_tokens(user_note):
        _add(tok)

    return ordered


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


def _propensity_bins_by_arm(weighted_csv_path: Path, arm_ids: list[str], active_id: str) -> dict:
    """Compute 10-bin propensity score distribution split by reference vs active arm for overlap visualization."""
    bins_by_arm = {aid: [0] * 10 for aid in arm_ids}
    ps_col = f"ps_{active_id}"
    if not weighted_csv_path.is_file():
        return {"bin_edges": [round(i / 10, 1) for i in range(11)], "counts_by_arm": bins_by_arm}
    with weighted_csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or ps_col not in reader.fieldnames:
            return {"bin_edges": [round(i / 10, 1) for i in range(11)], "counts_by_arm": bins_by_arm}
        for row in reader:
            arm = row.get("treatment_arm", "")
            if arm not in bins_by_arm:
                continue
            try:
                val = float(row[ps_col])
            except (ValueError, TypeError):
                continue
            idx = min(9, max(0, int(val * 10)))
            bins_by_arm[arm][idx] += 1
    return {
        "bin_edges": [round(i / 10, 1) for i in range(11)],
        "counts_by_arm": bins_by_arm,
    }


def _read_curve_series(curves_dir: Path) -> list[dict]:
    if not curves_dir.is_dir():
        return []
    out = []
    for cpath in sorted(curves_dir.glob("*.csv")):
        if cpath.is_symlink() or not cpath.is_file():
            continue
        with cpath.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        arm_id = rows[0].get("treatment_arm", cpath.stem.replace("cif_", ""))
        points = [
            {
                "t": round(float(r["time_day"]), 4),
                "cif1": round(float(r["cif_primary"]), 6),
                "cif2": round(float(r["cif_competing"]), 6),
                "surv": round(float(r.get("overall_survival", 0.0)), 6),
            }
            for r in rows
        ]
        out.append({"arm_id": arm_id, "points": points})
    return out


class TTERunner:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.sessions_root = self.root / "sessions"
        self.sessions_root.mkdir(mode=0o700, parents=True, exist_ok=True)
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

    def _session_path(self, session_id: str) -> Path:
        if not isinstance(session_id, str) or not SESSION_ID.fullmatch(session_id):
            raise ContractError("INVALID_SESSION_ID", "无效的会话编号。")
        path = self.sessions_root / session_id
        if path.is_symlink() or not path.is_dir() or time.time() - path.stat().st_mtime > RETENTION_SECONDS:
            raise ContractError("SESSION_NOT_FOUND", "会话不存在、已清除或已过期。")
        return path

    def _status(self, path):
        return json.loads((path / "status.json").read_text(encoding="utf-8"))

    def _write_status(self, path, status):
        _atomic_write_bytes(path / "status.json", _bytes(status))

    def cleanup(self, startup=False):
        with self._maintenance:
            now = time.time()
            retained = []
            for path in self.root.iterdir():
                if path.name == "sessions":
                    continue
                if path.is_symlink():
                    path.unlink()
                    continue
                if not path.is_dir() or not RUN_ID.fullmatch(path.name):
                    continue
                if now - path.stat().st_mtime > RETENTION_SECONDS:
                    shutil.rmtree(path, ignore_errors=True)
                    continue
                try:
                    status = self._status(path)
                except (OSError, ValueError):
                    shutil.rmtree(path, ignore_errors=True)
                    continue
                if startup and status["state"] in ("queued", "running"):
                    status.update(state="interrupted", error="Server restarted before this run finished")
                    self._write_status(path, status)
                retained.append(path)
            for path in sorted(retained, key=lambda p: p.stat().st_mtime)[:-MAX_RUNS]:
                shutil.rmtree(path, ignore_errors=True)

            # Cleanup expired isolated sessions
            if self.sessions_root.is_dir():
                sess_retained = []
                for spath in self.sessions_root.iterdir():
                    if spath.is_symlink():
                        spath.unlink()
                        continue
                    if not spath.is_dir() or not SESSION_ID.fullmatch(spath.name):
                        continue
                    if now - spath.stat().st_mtime > RETENTION_SECONDS:
                        shutil.rmtree(spath, ignore_errors=True)
                        continue
                    sess_retained.append(spath)
                for spath in sorted(sess_retained, key=lambda p: p.stat().st_mtime)[:-MAX_SESSIONS]:
                    shutil.rmtree(spath, ignore_errors=True)

    # ------------------------------------------------------------------
    # Session-Isolated Cohort Management (User-Cohort Track)
    # ------------------------------------------------------------------

    def save_session_cohort(
        self,
        raw_bytes: bytes,
        filename: str,
        *,
        session_id: str | None = None,
        user_note: str = "",
        sample_id: str | None = None,
    ) -> dict:
        if len(raw_bytes) > MAX_USER_COHORT_BYTES:
            raise ContractError("COHORT_FILE_TOO_LARGE", "上传数据文件超过 25 MB 限制。")

        df, delimiter, encoding = read_tabular_cohort_bytes(raw_bytes, filename=filename)
        input_sha = hashlib.sha256(raw_bytes).hexdigest()

        # Collect all non-whitelisted cohort strings & note identifiers so every outbound LLM call in this session enforces 0 leakage
        profile = profile_cohort_dataframe(df, filename=filename, input_sha256=input_sha)
        forbidden_list = _collect_cohort_forbidden_values(df, profile, user_note=user_note)
        forbidden_tokens = tuple(forbidden_list)
        default_mapping = recommend_column_mapping(profile)

        interpretation = interpret_cohort_profile(
            profile,
            user_note=user_note,
            forbidden_values=forbidden_tokens,
            default_mapping=default_mapping,
            sample_id=sample_id,
        )

        self.cleanup()
        if session_id and SESSION_ID.fullmatch(session_id) and (self.sessions_root / session_id).is_dir():
            sid = session_id
            sdir = self.sessions_root / sid
            raw_dir = sdir / "raw"
            if raw_dir.is_dir():
                shutil.rmtree(raw_dir)
        else:
            sid = uuid.uuid4().hex
            sdir = self.sessions_root / sid
            sdir.mkdir(mode=0o700, parents=True, exist_ok=True)

        raw_dir = sdir / "raw"
        raw_dir.mkdir(mode=0o700, exist_ok=True)
        safe_fname = SAFE_FILENAME_RE.sub("_", Path(filename).name).strip("_") or "cohort.csv"
        cohort_file = raw_dir / safe_fname
        _atomic_write_bytes(cohort_file, raw_bytes)

        meta_path = sdir / "session_meta.json"
        prev_runs = []
        if meta_path.is_file():
            try:
                prev_runs = json.loads(meta_path.read_text(encoding="utf-8")).get("run_ids", [])
            except Exception:
                prev_runs = []

        meta = {
            "session_id": sid,
            "filename": safe_fname,
            "cohort_relpath": f"raw/{safe_fname}",
            "input_sha256": input_sha,
            "detected_delimiter": delimiter,
            "detected_encoding": encoding,
            "sample_id": sample_id,
            "user_note": (user_note or "")[:300],
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_ids": prev_runs,
        }
        _atomic_write_bytes(sdir / "profile.json", _bytes(profile))
        _atomic_write_bytes(sdir / "interpretation.json", _bytes(interpretation))
        _atomic_write_bytes(sdir / "default_mapping.json", _bytes(default_mapping))
        _atomic_write_bytes(sdir / "forbidden_tokens.json", _bytes(forbidden_list))
        _atomic_write_bytes(meta_path, _bytes(meta))

        return {
            "session_id": sid,
            "meta": meta,
            "profile": profile,
            "interpretation": interpretation,
            "default_mapping": default_mapping,
        }

    def get_session(self, session_id: str) -> dict:
        sdir = self._session_path(session_id)
        meta = json.loads((sdir / "session_meta.json").read_text(encoding="utf-8"))
        profile = json.loads((sdir / "profile.json").read_text(encoding="utf-8"))
        interpretation = json.loads((sdir / "interpretation.json").read_text(encoding="utf-8"))
        default_mapping = json.loads((sdir / "default_mapping.json").read_text(encoding="utf-8"))
        cohort_path = sdir / meta["cohort_relpath"]
        return {
            "session_id": session_id,
            "cohort_path": cohort_path,
            "meta": meta,
            "profile": profile,
            "interpretation": interpretation,
            "default_mapping": default_mapping,
        }

    def get_session_forbidden_values(self, session_id: str) -> tuple[str, ...]:
        """Return persisted forbidden cohort/note tokens for a session (fail-safe empty tuple if session not found)."""
        if not isinstance(session_id, str) or not SESSION_ID.fullmatch(session_id):
            return ()
        fpath = self.sessions_root / session_id / "forbidden_tokens.json"
        if not fpath.is_file():
            return ()
        try:
            vals = json.loads(fpath.read_text(encoding="utf-8"))
            if isinstance(vals, list):
                return tuple(str(v) for v in vals if isinstance(v, str) and v.strip())
        except Exception:
            return ()
        return ()

    def update_session_note(self, session_id: str, user_note: str) -> dict:
        sess = self.get_session(session_id)
        sdir = self._session_path(session_id)
        existing_forbidden = list(self.get_session_forbidden_values(session_id))
        for tok in extract_untrusted_Note_forbidden_tokens(user_note):
            if tok not in existing_forbidden and len(existing_forbidden) < 2000:
                existing_forbidden.append(tok)
        _atomic_write_bytes(sdir / "forbidden_tokens.json", _bytes(existing_forbidden))
        interp = interpret_cohort_profile(
            sess["profile"],
            user_note=user_note,
            forbidden_values=tuple(existing_forbidden),
            default_mapping=sess["default_mapping"],
            sample_id=sess["meta"].get("sample_id"),
        )
        meta = dict(
            sess["meta"],
            user_note=(user_note or "")[:300],
            updated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        _atomic_write_bytes(sdir / "interpretation.json", _bytes(interp))
        _atomic_write_bytes(sdir / "session_meta.json", _bytes(meta))
        return {"session_id": session_id, "interpretation": interp, "meta": meta}

    def delete_session(self, session_id: str) -> dict:
        """Physically delete isolated session directory AND unconditionally sweep all derived run directories."""
        if not isinstance(session_id, str) or not SESSION_ID.fullmatch(session_id):
            raise ContractError("INVALID_SESSION_ID", "无效的会话编号。")
        sdir = self.sessions_root / session_id
        with self._maintenance:
            purged_files = 0
            candidate_run_dirs: dict[str, Path] = {}

            # 1. Read run_ids recorded in session_meta.json if readable (tolerating corrupt or missing JSON)
            if sdir.is_dir() and not sdir.is_symlink():
                try:
                    meta = json.loads((sdir / "session_meta.json").read_text(encoding="utf-8"))
                    if isinstance(meta, dict) and isinstance(meta.get("run_ids"), list):
                        for rid in meta["run_ids"]:
                            if isinstance(rid, str) and RUN_ID.fullmatch(rid):
                                rpath = self.root / rid
                                if os.path.lexists(rpath):
                                    candidate_run_dirs[rid] = rpath
                except Exception:
                    pass

            # 2. Unconditionally scan all run directories in self.root for status.json referencing session_id
            for rpath in self.root.iterdir():
                if rpath.name == "sessions" or not RUN_ID.fullmatch(rpath.name):
                    continue
                if rpath.is_dir() and not rpath.is_symlink():
                    matches_session = False
                    try:
                        st = self._status(rpath)
                        if isinstance(st, dict) and st.get("session_id") == session_id:
                            matches_session = True
                    except Exception:
                        st_file = rpath / "status.json"
                        try:
                            if st_file.is_file() and session_id in st_file.read_text(encoding="utf-8", errors="ignore"):
                                matches_session = True
                        except OSError:
                            pass
                    if matches_session:
                        candidate_run_dirs[rpath.name] = rpath

            # 3. Physically remove every matched derived run directory
            purged_runs: list[str] = []
            for rid, rpath in sorted(candidate_run_dirs.items()):
                if rpath.is_symlink():
                    rpath.unlink()
                    purged_files += 1
                    purged_runs.append(rid)
                elif rpath.is_dir():
                    purged_files += sum(1 for p in rpath.rglob("*") if os.path.lexists(p) and not p.is_dir())
                    shutil.rmtree(rpath)
                    purged_runs.append(rid)

            # 4. Physically remove the session directory itself
            if sdir.is_symlink():
                sdir.unlink()
                purged_files += 1
            elif sdir.is_dir():
                purged_files += sum(1 for p in sdir.rglob("*") if os.path.lexists(p) and not p.is_dir())
                shutil.rmtree(sdir)

            # 5. Post-deletion disk verification: never report session_deleted=True if any residue remains
            residual_runs = [rid for rid, rp in candidate_run_dirs.items() if os.path.lexists(rp)]
            for rpath in self.root.iterdir():
                if rpath.name == "sessions" or not RUN_ID.fullmatch(rpath.name):
                    continue
                if rpath.is_dir() and not rpath.is_symlink():
                    try:
                        if self._status(rpath).get("session_id") == session_id:
                            residual_runs.append(rpath.name)
                    except Exception:
                        pass

            if os.path.lexists(sdir) or residual_runs:
                raise RuntimeError(
                    f"SESSION_WIPE_INCOMPLETE: residual paths remain for session {session_id} "
                    f"(session_exists={os.path.lexists(sdir)}, residual_runs={residual_runs})"
                )

            return {
                "status": "session_deleted",
                "session_deleted": True,
                "session_id": session_id,
                "purged_files": purged_files,
                "purged_runs": purged_runs,
                "session_exists_on_disk": False,
            }

    # ------------------------------------------------------------------
    # Legacy Synthetic-Only Validation & Execution (100% Backward Compatible)
    # ------------------------------------------------------------------

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
            status = {
                "run_id": run_id,
                "state": "queued",
                "kind": "synthetic_tte",
                "project_mode": SOURCE_MODE_SYNTHETIC_ONLY,
                "case_count": cases,
                "seed": seed,
                "bootstrap_iterations": bootstrap_iterations,
                "weight_method": weight_method,
                "gate_status": None,
                "stage": "queued",
                "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._write_status(path, status)
            threading.Thread(target=self._execute, args=(path, status, normalized), daemon=True).start()
            return status
        except Exception:
            self._slot.release()
            raise

    # ------------------------------------------------------------------
    # New Dual-Mode Run Starters (`simulated_cohort` & `user_cohort`)
    # ------------------------------------------------------------------

    def resolve_user_cohort_arms(
        self,
        sess: dict,
        mapping: dict,
        *,
        intervention_override: str = "",
        comparator_override: str = "",
    ) -> tuple[str, str]:
        """Resolve effective (active_arm, comparator_arm) strings for dual-arm RCT alignment verification."""
        meta = sess.get("meta") or {}
        interp = sess.get("interpretation") or {}

        ovr_act = (intervention_override or str(mapping.get("active_drug_label") or meta.get("override_intervention") or "")).strip()
        ovr_comp = (comparator_override or str(mapping.get("comparator_drug_label") or meta.get("override_comparator") or "")).strip()

        act_raw = ovr_act or str(mapping.get("active_value") or "").strip()
        comp_raw = ovr_comp or str(mapping.get("comparator_value") or "").strip()
        if act_raw in ("", "0", "1", "yes", "no", "true", "false"):
            act_raw = str(interp.get("intervention_query_en") or act_raw).strip()
        if comp_raw in ("", "0", "1", "yes", "no", "true", "false"):
            comp_raw = str(interp.get("comparator_query_en") or comp_raw).strip()
        return act_raw, comp_raw

    def save_session_arm_overrides(self, session_id: str, intervention: str = "", comparator: str = "") -> None:
        """Persist researcher-supplied active/comparator drug labels for binary 0/1 user-cohort sessions."""
        if not session_id or not SESSION_ID.fullmatch(session_id):
            return
        sdir = self.sessions_root / session_id
        meta_file = sdir / "session_meta.json"
        if not meta_file.is_file():
            return
        with self._maintenance:
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                return
            changed = False
            if intervention.strip():
                meta["override_intervention"] = intervention.strip()[:100]
                changed = True
            if comparator.strip():
                meta["override_comparator"] = comparator.strip()[:100]
                changed = True
            if changed:
                _atomic_write_bytes(meta_file, _bytes(meta))

    def start_simulated_cohort(
        self,
        *,
        target_rct: dict | None = None,
        pico: dict | None = None,
        sim_config: dict | None = None,
        confirm_arm_mismatch: bool = False,
    ) -> dict:
        cfg = dict(sim_config or {})
        cases = int(cfg.get("cases", 1200))
        seed = int(cfg.get("seed", 42))
        bootstrap_iterations = int(cfg.get("bootstrap_iterations", 80))
        weight_method = str(cfg.get("weight_method", "overlap"))

        if not 100 <= cases <= MAX_CASES:
            raise ContractError("INVALID_CASE_COUNT", f"模拟样本量须在 100 至 {MAX_CASES} 之间。")
        if not 0 <= seed <= 2**32 - 1:
            raise ContractError("INVALID_SEED", "随机种子须在有效非负整数范围内。")
        if not 20 <= bootstrap_iterations <= MAX_BOOTSTRAP:
            raise ContractError("INVALID_BOOTSTRAP_ITERATIONS", f"Bootstrap 次数须在 20 至 {MAX_BOOTSTRAP} 之间。")
        if weight_method not in ("overlap", "stabilized_iptw", "iptw"):
            raise ContractError("INVALID_WEIGHT_METHOD", f"不支持的加权方式：{weight_method}")

        if target_rct and isinstance(pico, dict):
            sim_intr = str(cfg.get("intervention") or pico.get("active_arm_label") or pico.get("intervention_query_en") or pico.get("intervention") or "").strip()
            sim_comp = str(cfg.get("comparator") or pico.get("comparator_arm_label") or pico.get("comparator_query_en") or "").strip()
            if not sim_comp and pico.get("comparator_explicit") is True:
                sim_comp = str(pico.get("comparator") or "").strip()
            target_rct = verify_and_enrich_target_rct(
                target_rct,
                intervention=sim_intr,
                comparator=sim_comp,
                project_mode=SOURCE_MODE_SIMULATED_COHORT,
                confirm_arm_mismatch=confirm_arm_mismatch,
            )

        normalized_proto, dgp_params = build_simulated_protocol_and_dgp(
            target_rct=target_rct,
            pico=pico,
            sim_config=cfg,
        )
        if not self._slot.acquire(blocking=False):
            raise ContractError("BUSY", "当前已有另一个 TTE 分析任务正在运行，请稍候。")
        try:
            self.cleanup()
            run_id = uuid.uuid4().hex
            path = self.root / run_id
            path.mkdir(mode=0o700)
            _atomic_write_bytes(path / "protocol.json", _bytes(normalized_proto))
            _atomic_write_bytes(path / "dgp_params.json", _bytes(dgp_params))
            if target_rct:
                _atomic_write_bytes(path / "target_rct.json", _bytes(target_rct))
            if pico:
                _atomic_write_bytes(path / "pico.json", _bytes(pico))
            status = {
                "run_id": run_id,
                "state": "queued",
                "kind": "simulated_cohort_tte",
                "project_mode": SOURCE_MODE_SIMULATED_COHORT,
                "case_count": cases,
                "seed": seed,
                "bootstrap_iterations": bootstrap_iterations,
                "weight_method": weight_method,
                "sim_config": cfg,
                "gate_status": None,
                "stage": "queued",
                "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._write_status(path, status)
            threading.Thread(target=self._execute_simulated, args=(path, status), daemon=True).start()
            return status
        except Exception:
            self._slot.release()
            raise

    def start_user_cohort(
        self,
        session_id: str,
        mapping: dict,
        *,
        target_rct: dict | None = None,
        seed: int = 42,
        bootstrap_iterations: int = 80,
        weight_method: str | None = None,
        intervention_override: str = "",
        comparator_override: str = "",
        confirm_arm_mismatch: bool = False,
    ) -> dict:
        if intervention_override.strip() or comparator_override.strip():
            self.save_session_arm_overrides(session_id, intervention_override, comparator_override)
        sess = self.get_session(session_id)
        cohort_path = sess["cohort_path"]
        if not cohort_path.is_file():
            raise ContractError("SESSION_COHORT_MISSING", "会话中的队列数据文件已不存在，请重新上传。")

        chosen_weight = str(weight_method or mapping.get("weight_method") or "overlap")
        if chosen_weight not in ("overlap", "stabilized_iptw", "iptw"):
            raise ContractError("INVALID_WEIGHT_METHOD", f"不支持的加权方式：{chosen_weight}")
        seed_i = int(mapping.get("seed", seed))
        boot_i = int(mapping.get("bootstrap_iterations", bootstrap_iterations))
        if not 20 <= boot_i <= MAX_BOOTSTRAP:
            raise ContractError("INVALID_BOOTSTRAP_ITERATIONS", f"Bootstrap 次数须在 20 至 {MAX_BOOTSTRAP} 之间。")

        if target_rct:
            eff_act, eff_comp = self.resolve_user_cohort_arms(
                sess,
                mapping,
                intervention_override=intervention_override,
                comparator_override=comparator_override,
            )
            target_rct = verify_and_enrich_target_rct(
                target_rct,
                intervention=eff_act,
                comparator=eff_comp,
                project_mode=SOURCE_MODE_USER_COHORT,
                confirm_arm_mismatch=confirm_arm_mismatch,
            )

        if not self._slot.acquire(blocking=False):
            raise ContractError("BUSY", "当前已有另一个 TTE 分析任务正在运行，请稍候。")
        try:
            self.cleanup()
            run_id = uuid.uuid4().hex
            path = self.root / run_id
            path.mkdir(mode=0o700)
            _atomic_write_bytes(path / "mapping.json", _bytes(mapping))
            if target_rct:
                _atomic_write_bytes(path / "target_rct.json", _bytes(target_rct))

            # Register run_id atomically inside session metadata under maintenance lock
            with self._maintenance:
                sdir = self._session_path(session_id)
                smeta = dict(sess["meta"])
                smeta["run_ids"] = list(dict.fromkeys([*smeta.get("run_ids", []), run_id]))
                _atomic_write_bytes(sdir / "session_meta.json", _bytes(smeta))

            status = {
                "run_id": run_id,
                "session_id": session_id,
                "state": "queued",
                "kind": "user_cohort_tte",
                "project_mode": SOURCE_MODE_USER_COHORT,
                "seed": seed_i,
                "bootstrap_iterations": boot_i,
                "weight_method": chosen_weight,
                "gate_status": None,
                "stage": "queued",
                "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._write_status(path, status)
            threading.Thread(
                target=self._execute_user_cohort,
                args=(path, status, cohort_path, mapping, target_rct),
                daemon=True,
            ).start()
            return status
        except Exception:
            self._slot.release()
            raise

    def _command(self, command, deadline, env, permitted=(0,)):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("TTE run exceeded 90 seconds")
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=remaining, check=False)
        if result.returncode not in permitted:
            raise RuntimeError(result.stderr.strip() or "TTE command failed")
        return result.returncode

    def _empty_eligibility_gate(self, path, screen):
        diagnostics = path / "diagnostics"
        diagnostics.mkdir(mode=0o700)
        payload = {
            "gate_status": "blocked",
            "initial_n": screen["eligible_rows"],
            "gate_failures": [
                {
                    "code": "NO_ELIGIBLE_SYNTHETIC_RISK_SET",
                    "detail": "No eligible synthetic rows remain in both strategy arms after screening",
                }
            ],
            "weight_method": "overlap",
        }
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
                raise RuntimeError("run artifacts exceed limit")
            protocol_manifest = json.loads((path / "protocol_check" / "run_manifest.json").read_text(encoding="utf-8"))
            status["artifact_bytes"] = total_bytes
            manifest = {
                "source_mode": "synthetic_only",
                "code_commit": protocol_manifest["code_commit"],
                "code_dirty": protocol_manifest["code_dirty"],
                "tte_dependency_commit": os.environ.get("EMULATRX_TTE_DEPENDENCY_COMMIT"),
                "gate_status": status["gate_status"],
                "cohort_rows": screen["generated_rows"],
                "eligible_rows": screen["eligible_rows"],
                "dependency_versions": protocol_manifest["dependency_versions"],
                "output_sha256": {
                    str(p.relative_to(path)): _sha256(p)
                    for p in files
                    if p.name != "status.json" and not p.name.startswith(".status-")
                },
            }
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

    def _execute_simulated(self, path: Path, status: dict) -> None:
        try:
            status = dict(status, state="running", stage="simulated_generation_and_estimation")
            self._write_status(path, status)
            bundle_dir = path / "bundle"
            res, code = run_simulated_cohort_pipeline(
                path / "protocol.json",
                seed=status["seed"],
                cases=status["case_count"],
                weight_method=status["weight_method"],
                bootstrap_iterations=status["bootstrap_iterations"],
                out_dir=bundle_dir,
                dgp_params_path=path / "dgp_params.json",
            )
            diag = json.loads((bundle_dir / "diagnostics_summary.json").read_text(encoding="utf-8"))
            run_man = json.loads((bundle_dir / "run_manifest.json").read_text(encoding="utf-8"))
            status["cohort_rows"] = status["case_count"]
            status["eligible_rows"] = diag.get("retained_n", 0)
            if (bundle_dir / "cohort.csv").is_file():
                status["cohort_columns"], _ = _csv_rows(bundle_dir / "cohort.csv", limit=1)
            else:
                status["cohort_columns"] = []
            status["gate_status"] = "blocked" if code == 3 else "passed"

            files = [p for p in path.rglob("*") if p.is_file() and p.name != "status.json" and not p.name.startswith(".status-")]
            preview_man = {
                "source_mode": SOURCE_MODE_SIMULATED_COHORT,
                "code_commit": run_man["code_commit"],
                "code_dirty": run_man["code_dirty"],
                "tte_dependency_commit": os.environ.get("EMULATRX_TTE_DEPENDENCY_COMMIT"),
                "gate_status": status["gate_status"],
                "cohort_rows": status["cohort_rows"],
                "eligible_rows": status["eligible_rows"],
                "dependency_versions": run_man["dependency_versions"],
                "output_sha256": {str(p.relative_to(path)): _sha256(p) for p in files},
            }
            (path / "preview_manifest.json").write_bytes(_bytes(preview_man))
            status["state"] = "completed" if code == 0 else "blocked"
            status["stage"] = "complete"
            status["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._write_status(path, status)
        except ContractError as exc:
            self._write_status(
                path,
                dict(
                    status,
                    state="failed",
                    error=f"{exc.code}: {exc}",
                    finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
        except Exception as exc:
            self._write_status(
                path,
                dict(
                    status,
                    state="failed",
                    error=f"模拟预演运行失败：{exc}",
                    finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
        finally:
            self._slot.release()

    def _execute_user_cohort(self, path: Path, status: dict, cohort_path: Path, mapping: dict, target_rct: dict | None) -> None:
        try:
            status = dict(status, state="running", stage="cohort_ingestion_and_estimation")
            self._write_status(path, status)
            bundle_dir = path / "bundle"
            res, code = run_user_cohort_pipeline(
                cohort_path,
                mapping,
                out_dir=bundle_dir,
                target_rct=target_rct,
                seed=status["seed"],
                bootstrap_iterations=status["bootstrap_iterations"],
                weight_method=status["weight_method"],
            )
            diag = json.loads((bundle_dir / "diagnostics_summary.json").read_text(encoding="utf-8"))
            run_man = json.loads((bundle_dir / "run_manifest.json").read_text(encoding="utf-8"))
            status["cohort_rows"] = res["raw_row_count"]
            status["eligible_rows"] = diag.get("retained_n", res["eligible_row_count"])
            status["input_sha256"] = res["input_sha256"]
            status["gate_status"] = "blocked" if code == 3 else "passed"
            if (bundle_dir / "standardized_cohort.csv").is_file():
                status["cohort_columns"], _ = _csv_rows(bundle_dir / "standardized_cohort.csv", limit=1)
            else:
                status["cohort_columns"] = []

            files = [p for p in path.rglob("*") if p.is_file() and p.name != "status.json" and not p.name.startswith(".status-")]
            preview_man = {
                "source_mode": SOURCE_MODE_USER_COHORT,
                "code_commit": run_man["code_commit"],
                "code_dirty": run_man["code_dirty"],
                "tte_dependency_commit": os.environ.get("EMULATRX_TTE_DEPENDENCY_COMMIT"),
                "gate_status": status["gate_status"],
                "cohort_rows": status["cohort_rows"],
                "eligible_rows": status["eligible_rows"],
                "input_sha256": res["input_sha256"],
                "dependency_versions": run_man["dependency_versions"],
                "output_sha256": {str(p.relative_to(path)): _sha256(p) for p in files},
            }
            (path / "preview_manifest.json").write_bytes(_bytes(preview_man))
            status["state"] = "completed" if code == 0 else "blocked"
            status["stage"] = "complete"
            status["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._write_status(path, status)
        except ContractError as exc:
            self._write_status(
                path,
                dict(
                    status,
                    state="failed",
                    error=f"{exc.code}: {exc}",
                    finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
        except Exception as exc:
            self._write_status(
                path,
                dict(
                    status,
                    state="failed",
                    error=f"自有队列分析失败：{exc}",
                    finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
        finally:
            self._slot.release()

    def get(self, run_id):
        path = self._path(run_id)
        status = self._status(path)
        if status["state"] not in ("completed", "blocked"):
            return status

        kind = status.get("kind", "synthetic_tte")
        if kind == "synthetic_tte":
            diagnostic = json.loads((path / "diagnostics" / "diagnostics_summary.json").read_text(encoding="utf-8"))
            _, screening_attrition = _csv_rows(path / "screen" / "eligibility_attrition.csv", limit=30)
            _, diagnosis_attrition = _csv_rows(path / "diagnostics" / "attrition_table.csv", limit=30)
            _, balance = _csv_rows(path / "diagnostics" / "balance_diagnostics.csv", limit=50)
            _, sample = _csv_rows(path / "cohort" / "cohort.csv", limit=8)
            manifest = json.loads((path / "preview_manifest.json").read_text(encoding="utf-8"))
            result = {
                **status,
                "normalized_protocol": json.loads((path / "protocol_check" / "normalized_protocol.json").read_text(encoding="utf-8")),
                "diagnostics": diagnostic,
                "attrition": screening_attrition + diagnosis_attrition,
                "balance": balance,
                "sample_cases": sample,
                "preview_manifest": manifest,
                "available_artifacts": self.artifacts(run_id),
            }
            if status["gate_status"] == "passed":
                estimates = json.loads((path / "estimates" / "estimates.json").read_text(encoding="utf-8"))
                result["estimates"] = {
                    "co_presented_comparison_table": estimates["co_presented_comparison_table"],
                    "arm_cif_summaries": estimates["arm_cif_summaries"],
                    "methodological_warnings": estimates["methodological_warnings"],
                }
            return result

        # Dual-mode bundle (`simulated_cohort_tte` or `user_cohort_tte`)
        bdir = path / "bundle"
        norm_proto = json.loads((bdir / "normalized_protocol.json").read_text(encoding="utf-8"))
        diagnostic = json.loads((bdir / "diagnostics_summary.json").read_text(encoding="utf-8"))
        _, diag_attr = _csv_rows(bdir / "attrition_table.csv", limit=30)
        ingest_attr = []
        if (bdir / "ingest_attrition.csv").is_file():
            _, ingest_attr = _csv_rows(bdir / "ingest_attrition.csv", limit=30)
        # Avoid duplicate baseline row between ingest_attrition final row and diag_attr initial_cohort row
        merged_attr = list(ingest_attr)
        for row in diag_attr:
            if merged_attr and row.get("step_id") == "initial_cohort":
                continue
            merged_attr.append(row)

        _, balance = _csv_rows(bdir / "balance_diagnostics.csv", limit=80)
        manifest = json.loads((path / "preview_manifest.json").read_text(encoding="utf-8"))
        target_rct = json.loads((path / "target_rct.json").read_text(encoding="utf-8")) if (path / "target_rct.json").is_file() else norm_proto.get("target_rct")
        ingest_manifest = json.loads((bdir / "ingest_manifest.json").read_text(encoding="utf-8")) if (bdir / "ingest_manifest.json").is_file() else None

        arm_ids = [s["id"] for s in norm_proto["treatment_strategies"]]
        active_id = arm_ids[1] if len(arm_ids) > 1 else arm_ids[0]
        ps_hist = _propensity_bins_by_arm(bdir / "weighted_cohort.csv", arm_ids, active_id)

        result = {
            **status,
            "normalized_protocol": norm_proto,
            "target_rct": target_rct,
            "ingest_manifest": ingest_manifest,
            "diagnostics": diagnostic,
            "attrition": merged_attr,
            "balance": balance,
            "propensity_overlap": ps_hist,
            "preview_manifest": manifest,
            "available_artifacts": self.artifacts(run_id),
        }
        if status["gate_status"] == "passed":
            estimates = json.loads((bdir / "estimates.json").read_text(encoding="utf-8"))
            result["estimates"] = estimates
            result["curve_series"] = _read_curve_series(bdir / "curves")
            result["audit_report_markdown"] = (bdir / "audit_report.md").read_text(encoding="utf-8")
        return result

    def artifacts(self, run_id):
        path = self._path(run_id)
        status = self._status(path)
        if status["state"] not in ("completed", "blocked"):
            return []
        kind = status.get("kind", "synthetic_tte")
        if kind == "synthetic_tte":
            names = [
                "protocol.json",
                "preview_manifest.json",
                "protocol_check/normalized_protocol.json",
                "protocol_check/protocol_check_report.md",
                "protocol_check/run_manifest.json",
                "cohort/cohort.csv",
                "cohort/dgp_truth.json",
                "cohort/PURPOSE.txt",
                "cohort/run_manifest.json",
                "screen/eligible_cohort.csv",
                "screen/eligibility_attrition.csv",
                "diagnostics/attrition_table.csv",
                "diagnostics/balance_diagnostics.csv",
                "diagnostics/diagnostics_summary.json",
                "diagnostics/weighted_cohort.csv",
                "diagnostics/run_manifest.json",
                "estimates/estimates.json",
                "estimates/audit_report.md",
                "estimates/run_manifest.json",
            ]
            curve_dir = path / "estimates" / "curves"
            if curve_dir.is_dir():
                names.extend(
                    f"estimates/curves/{p.name}"
                    for p in sorted(curve_dir.glob("*.csv"))
                    if p.is_file() and not p.is_symlink()
                )
            return [name for name in names if (path / name).is_file() and not (path / name).is_symlink()]

        # Dual-mode bundle artifacts (note: for user_cohort, expose aggregated reports/curves/manifests, omit raw individual rows if desired; standardized/weighted cohorts stay inside isolated session/run and are wiped on session delete)
        candidates = [
            "preview_manifest.json",
            "target_rct.json",
            "bundle/normalized_protocol.json",
            "bundle/protocol_check_report.md",
            "bundle/ingest_manifest.json",
            "bundle/ingest_attrition.csv",
            "bundle/dgp_truth.json",
            "bundle/attrition_table.csv",
            "bundle/balance_diagnostics.csv",
            "bundle/diagnostics_summary.json",
            "bundle/estimates.json",
            "bundle/audit_report.md",
            "bundle/run_manifest.json",
        ]
        if kind == "simulated_cohort_tte":
            candidates.extend(["bundle/cohort.csv", "bundle/weighted_cohort.csv"])
        curve_dir = path / "bundle" / "curves"
        if curve_dir.is_dir():
            candidates.extend(
                f"bundle/curves/{p.name}"
                for p in sorted(curve_dir.glob("*.csv"))
                if p.is_file() and not p.is_symlink()
            )
        return [name for name in candidates if (path / name).is_file() and not (path / name).is_symlink()]

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
        return {
            "name": name,
            "offset": offset,
            "total_bytes": size,
            "next_offset": offset + len(chunk),
            "encoding": "base64",
            "data": base64.b64encode(chunk).decode("ascii"),
        }
