"""Shared MCP tool and resource registry for stdio and Streamable HTTP."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import tempfile
import time

from emulatrx_tte import (
    SOURCE_MODE_SIMULATED_COHORT,
    SOURCE_MODE_SYNTHETIC_ONLY,
    SOURCE_MODE_USER_COHORT,
    build_simulated_protocol_and_dgp,
    fetch_rct_by_nct,
    generate_sample_cohort_csv_bytes,
    get_llm_status,
    parse_research_idea_pico,
    recommend_column_mapping,
    search_candidate_rcts,
    verify_and_enrich_target_rct,
)
from emulatrx_tte._common import ContractError, MAX_USER_COHORT_BYTES
from emulatrx_tte.ccw import run_ccw_synthetic_pipeline, validate_ccw_protocol_dict
from emulatrx_tte.cohort_ingest import detect_missing_clinical_semantics, ingest_user_cohort
from emulatrx_tte.diagnostics import diagnose_cohort
from emulatrx_tte.llm_assist import extract_untrusted_Note_forbidden_tokens
from emulatrx_tte.protocol import validate_protocol_dict
from emulatrx_tte.rct_matcher import audit_study_pico4d_alignment
from emulatrx_tte.synthetic import generate_synthetic_cohort

ROOT = Path(__file__).resolve().parents[1]
CSV_TEXT_LIMIT = 512 * 1024
CANONICAL_TOOLS = (
    "parse_research_idea",
    "ingest_cohort_session",
    "delete_cohort_session",
    "search_target_rcts",
    "audit_target_rct_alignment",
    "draft_tte_protocol",
    "validate_tte_protocol",
    "validate_ccw_protocol",
    "diagnose_tte_feasibility",
    "run_static_or_landmark_tte",
    "run_grace_period_ccw",
    "get_tte_run_status",
    "get_tte_result_summary",
    "export_tte_artifact",
)
LEGACY_TOOL_ALIASES = (
    "profile_cohort_session",
    "start_user_cohort_tte",
    "start_simulated_rct_tte",
    "start_synthetic_tte",
    "get_tte_run",
    "read_tte_artifact",
)
RESOURCE_URIS = (
    "emulatrx://schemas/tte-protocol-v1",
    "emulatrx://schemas/tte-ccw-v1",
    "emulatrx://samples/catalog",
    "emulatrx://guide/error-and-gate-codes",
)


def _read_repo_text(*parts: str) -> str:
    path = ROOT.joinpath(*parts)
    return path.read_text(encoding="utf-8")


def _session_view(saved: dict) -> dict:
    profile = saved["profile"]
    semantics = detect_missing_clinical_semantics_from_profile(profile)
    public_profile = {
        "filename": profile.get("filename"),
        "row_count": profile.get("row_count"),
        "column_count": profile.get("column_count"),
        "input_sha256": profile.get("input_sha256"),
        "candidate_treatment_columns": profile.get("candidate_treatment_columns"),
        "candidate_time_columns": profile.get("candidate_time_columns"),
        "candidate_outcome_columns": profile.get("candidate_outcome_columns"),
        "pii_columns_quarantined": semantics["pii_columns_quarantined"],
    }
    interpretation = saved.get("interpretation") or {}
    meta = saved.get("meta") or {}
    return {
        "session_id": saved["session_id"],
        "user_namespace": saved.get("user_namespace") or meta.get("user_namespace") or "default",
        "profile": public_profile,
        "interpretation": {key: value for key, value in interpretation.items() if key != "sanitized_llm_payload"},
        "default_mapping": saved.get("default_mapping"),
        "inferred_pico": {
            "condition": interpretation.get("condition_query_en"),
            "intervention": interpretation.get("intervention_query_en"),
            "comparator": interpretation.get("comparator_query_en"),
            "outcome": interpretation.get("outcome_query_en") or interpretation.get("primary_outcome"),
        },
        "missing_clinical_semantics": semantics["missing_clinical_semantics"],
    }


def detect_missing_clinical_semantics_from_profile(profile: dict) -> dict:
    """Reuse the cohort semantic probe without re-reading row values."""
    import pandas as pd

    columns = profile.get("columns") or []
    frame = pd.DataFrame({str(column.get("name")): [] for column in columns}) if columns else pd.DataFrame()
    return detect_missing_clinical_semantics(frame, profile)


_STATIC_ESS_HINT = "有效样本量过低。减少极端权重协变量，或改用 overlap 并检查组别样本量。"
_CCW_ESS_HINT = "克隆后有效样本不足。延长随访或增大合成样本量后再预检。"


def _remediation(failures: list[dict], *, is_ccw: bool = False) -> list[dict]:
    """Map gate codes to Chinese hints. Static propensity ESS and CCW cloning ESS share a code but not a hint."""
    hints = {
        "POSITIVITY_VIOLATION": "缩小极不平衡的基线协变量，或放宽方案里的 positivity_epsilon 后再预检。",
        "ESS_RATIO_BELOW_MIN": _CCW_ESS_HINT if is_ccw else _STATIC_ESS_HINT,
        "SMD_EXCEEDS_MAX": "加权后仍有协变量 |SMD| 超门限。从方案里去掉该协变量或改用更能重叠的加权。",
        "POST_GRACE_SMD_EXCEEDS_MAX": "宽限期后克隆人群仍不平衡。检查时变协变量是否期后测量，或放宽 max_post_grace_smd。",
    }
    out = []
    for failure in failures or []:
        code = str(failure.get("code"))
        out.append({"code": code, "hint_zh": hints.get(code, "按诊断明细调整方案后重新预检，未过门禁不会输出点估计。"), "detail": failure.get("detail")})
    return out


def register_tools(mcp, runner, *, transport: str = "http", allow_ingest_dirs: list[Path] | None = None) -> None:
    """Register the canonical 14 tools, legacy aliases, and 4 read-only resources."""
    allowed = [path.resolve() for path in (allow_ingest_dirs or [])]

    def _resolve_file(file_path: str) -> Path:
        candidate = Path(file_path).expanduser().resolve()
        if transport != "stdio":
            if not allowed or not any(candidate == root or root in candidate.parents for root in allowed):
                raise ContractError("PATH_TRAVERSAL_BLOCKED", "PATH_TRAVERSAL_BLOCKED: file_path is outside the configured ingest directory")
        if not candidate.is_file():
            raise ContractError("COHORT_FILE_NOT_FOUND", "file_path does not exist")
        return candidate

    def _ingest(
        sample_id: str = "",
        csv_base64: str = "",
        csv_text: str = "",
        file_path: str = "",
        filename: str = "cohort.csv",
        user_note: str = "",
        session_id: str = "",
        issue_upload_ticket: bool = False,
    ) -> dict:
        if issue_upload_ticket:
            if transport == "stdio":
                raise ContractError("UPLOAD_TICKET_HTTP_ONLY", "issue_upload_ticket is only available on the HTTP MCP server")
            ticket = runner.issue_upload_ticket(filename)
            return {
                "issue_upload_ticket": True,
                "ticket_id": ticket["ticket_id"],
                "session_id": ticket["session_id"],
                "upload_url": f"/api/sessions/upload-ticket/{ticket['ticket_id']}",
                "expires_in_seconds": 600,
                "max_bytes": MAX_USER_COHORT_BYTES,
            }
        if session_id.strip() and not any((sample_id.strip(), csv_base64.strip(), csv_text.strip(), file_path.strip())):
            saved = runner.get_session(session_id.strip())
            return _session_view(saved)
        if sample_id.strip():
            fname, raw = generate_sample_cohort_csv_bytes(sample_id.strip())
        elif file_path.strip():
            path = _resolve_file(file_path.strip())
            raw = path.read_bytes()
            fname = filename if filename != "cohort.csv" else path.name
        elif csv_text.strip():
            raw = csv_text.encode("utf-8")
            if len(raw) > CSV_TEXT_LIMIT:
                raise ContractError("CSV_TEXT_TOO_LARGE", "csv_text exceeds 512 KiB; use file_path or an upload ticket")
            fname = filename
        elif csv_base64.strip():
            raw = base64.b64decode(csv_base64.encode("ascii"), validate=True)
            fname = filename
        else:
            raise ContractError("MISSING_COHORT_INPUT", "Provide file_path, csv_text, csv_base64, sample_id, session_id, or issue_upload_ticket")
        saved = runner.save_session_cohort(raw, fname, user_note=user_note, sample_id=sample_id.strip() or None)
        return _session_view(saved)

    def _wait(run_id: str, wait_seconds: float) -> dict:
        deadline = time.time() + max(0.0, float(wait_seconds))
        status = runner.get(run_id)
        while status.get("state") in ("queued", "running") and time.time() < deadline:
            time.sleep(0.1)
            status = runner.get(run_id)
        return status

    def _load_target(nct_id: str = "", target_rct: dict | None = None) -> dict | None:
        if isinstance(target_rct, dict) and target_rct.get("nct_id"):
            return target_rct
        if nct_id.strip():
            return fetch_rct_by_nct(nct_id.strip())
        return None

    @mcp.tool(description="Parse a clinical research idea into PICO and ClinicalTrials.gov query terms.")
    def parse_research_idea(
        idea_text: str,
        condition: str = "",
        intervention: str = "",
        comparator: str = "",
        outcome: str = "",
    ) -> dict:
        pico = parse_research_idea_pico(idea_text, explicit_comparator=comparator.strip())
        if condition.strip():
            pico["condition_query_en"] = condition.strip()
        if intervention.strip():
            pico["intervention_query_en"] = intervention.strip()
        if comparator.strip():
            pico["comparator_query_en"] = comparator.strip()
            pico["comparator_explicit"] = True
        if outcome.strip():
            pico["primary_outcome"] = outcome.strip()
        return {
            "pico": pico,
            "comparator_explicit": bool(pico.get("comparator_explicit")),
            "ctgov_queries": {
                "query": pico.get("search_query_en", ""),
                "condition": pico.get("condition_query_en", ""),
                "intervention": pico.get("intervention_query_en", ""),
                "comparator": pico.get("comparator_query_en", ""),
                "outcome": pico.get("primary_outcome", ""),
            },
        }

    @mcp.tool(description="Ingest a de-identified cohort from a local file, a small csv_text table, a sample, an existing session, or an HTTP upload ticket.")
    def ingest_cohort_session(
        sample_id: str = "",
        csv_base64: str = "",
        csv_text: str = "",
        file_path: str = "",
        filename: str = "cohort.csv",
        user_note: str = "",
        session_id: str = "",
        issue_upload_ticket: bool = False,
    ) -> dict:
        return _ingest(sample_id, csv_base64, csv_text, file_path, filename, user_note, session_id, issue_upload_ticket)

    @mcp.tool(description="Legacy alias of ingest_cohort_session.")
    def profile_cohort_session(
        sample_id: str = "",
        csv_base64: str = "",
        filename: str = "cohort.csv",
        user_note: str = "",
        csv_text: str = "",
        file_path: str = "",
        session_id: str = "",
    ) -> dict:
        return _ingest(sample_id, csv_base64, csv_text, file_path, filename, user_note, session_id, False)

    @mcp.tool(description="Physically delete a cohort session and every derived run.")
    def delete_cohort_session(session_id: str) -> dict:
        return runner.delete_session(session_id)

    @mcp.tool(description="Search ClinicalTrials.gov API v2, with an explicit offline catalog fallback and five-dimension alignment caps.")
    def search_target_rcts(
        query: str = "",
        condition: str = "",
        intervention: str = "",
        comparator: str = "",
        outcome: str = "",
        idea_text: str = "",
        session_id: str = "",
        project_mode: str = "",
        target_followup_days: float = 0,
        max_results: int = 5,
    ) -> dict:
        pico = None
        ctx_summary = None
        sid = session_id.strip()
        cond_in, intr_in, comp_in, out_in = condition.strip(), intervention.strip(), comparator.strip(), outcome.strip()
        if sid and any((cond_in, intr_in, comp_in, out_in)):
            runner.save_session_pico_overrides(sid, condition=cond_in, intervention=intr_in, comparator=comp_in, outcome=out_in)
        if idea_text.strip():
            pico = parse_research_idea_pico(idea_text, explicit_comparator=comp_in)
            query = query or pico["search_query_en"]
            condition = cond_in or pico["condition_query_en"]
            intervention = intr_in or pico["intervention_query_en"]
            comparator = comp_in or pico.get("comparator_query_en", "")
            outcome = out_in or pico.get("primary_outcome", "")
            ctx_summary = pico
        elif sid:
            try:
                sess = runner.get_session(sid)
                interp = sess["interpretation"]
                meta = sess.get("meta") or {}
                ctx_summary = interp
                query = query or interp.get("search_query_en", "")
                condition = cond_in or str(meta.get("override_condition") or interp.get("condition_query_en", ""))
                intervention = intr_in or str(meta.get("override_intervention") or interp.get("intervention_query_en", ""))
                comparator = comp_in or str(meta.get("override_comparator") or interp.get("comparator_query_en", ""))
                outcome = out_in or str(meta.get("override_outcome") or interp.get("outcome_query_en", ""))
            except ContractError:
                pass
        forbidden_list = list(runner.get_session_forbidden_values(sid)) if sid else []
        if sid:
            for token in extract_untrusted_Note_forbidden_tokens(f"{query} {condition} {intervention} {comparator} {outcome}"):
                if token not in forbidden_list:
                    forbidden_list.append(token)
        mode = project_mode.strip() or (SOURCE_MODE_USER_COHORT if sid else SOURCE_MODE_SIMULATED_COHORT)
        res = search_candidate_rcts(
            query,
            condition=condition,
            intervention=intervention,
            comparator=comparator,
            outcome=outcome,
            project_mode=mode,
            context_summary=ctx_summary,
            max_results=max(1, min(int(max_results), 8)),
            forbidden_values=tuple(forbidden_list),
            target_followup_days=float(target_followup_days) if target_followup_days else None,
        )
        return {"pico": pico, "llm_status": get_llm_status(), **res}

    @mcp.tool(description="Fetch one NCT record and return the five-dimension PICO-T alignment audit.")
    def audit_target_rct_alignment(
        nct_id: str = "",
        target_rct: dict | None = None,
        session_id: str = "",
        pico: dict | None = None,
        project_mode: str = "",
        condition: str = "",
        intervention: str = "",
        comparator: str = "",
        outcome: str = "",
        target_followup_days: float = 0,
    ) -> dict:
        study = _load_target(nct_id, target_rct)
        if study is None:
            raise ContractError("MISSING_TARGET_RCT", "Provide nct_id or target_rct")
        ctx = pico or {}
        if session_id.strip():
            sess = runner.get_session(session_id.strip())
            interp = sess.get("interpretation") or {}
            ctx = {**interp, **ctx}
        audit = audit_study_pico4d_alignment(
            study,
            intervention=intervention or str(ctx.get("intervention_query_en") or ""),
            comparator=comparator or str(ctx.get("comparator_query_en") or ""),
            target_condition=condition or str(ctx.get("condition_query_en") or ""),
            target_outcome=outcome or str(ctx.get("primary_outcome") or ctx.get("outcome_query_en") or ""),
            target_followup_days=float(target_followup_days) if target_followup_days else None,
            project_mode=project_mode or (SOURCE_MODE_USER_COHORT if session_id.strip() else SOURCE_MODE_SIMULATED_COHORT),
        )
        dims = audit.get("dimension_audit") or {}
        return {
            "nct_id": study.get("nct_id"),
            "target_rct": {**study, **audit},
            "design_suitability": (dims.get("design") or {}).get("status", audit.get("design_suitability")),
            "population_alignment": (dims.get("population") or {}).get("status", audit.get("population_alignment")),
            "arm_alignment_status": audit.get("arm_alignment_status"),
            "outcome_alignment": (dims.get("outcome") or {}).get("status", audit.get("outcome_alignment")),
            "followup_alignment": (dims.get("followup") or {}).get("status", audit.get("followup_alignment")),
            "overall_alignment_status": audit.get("overall_alignment_status"),
            "requires_confirmation": bool(audit.get("requires_mismatch_confirmation")),
            "required_confirmation_flag": (
                "confirm_arm_mismatch"
                if audit.get("requires_mismatch_confirmation")
                or audit.get("arm_alignment_status") not in (None, "exact_dual_arm", "single_intervention_matched", "unspecified", "aligned")
                else "confirm_protocol_mismatch"
            ),
            "divergence_explanation_zh": audit.get("protocol_alignment_warning") or audit.get("arm_alignment_warning") or "",
        }

    @mcp.tool(description="Draft an inspectable static, landmark, or grace-period CCW protocol. Unconfirmed PICO-T mismatch fails closed.")
    def draft_tte_protocol(
        design_mode: str = "static",
        target_rct: dict | None = None,
        nct_id: str = "",
        session_id: str = "",
        pico: dict | None = None,
        mapping_overrides: dict | None = None,
        confirm_arm_mismatch: bool = False,
        confirm_protocol_mismatch: bool = False,
        project_mode: str = "",
    ) -> dict:
        study = _load_target(nct_id, target_rct)
        mode = project_mode or (SOURCE_MODE_USER_COHORT if session_id.strip() else SOURCE_MODE_SIMULATED_COHORT)
        ctx = dict(pico or {})
        mapping = dict(mapping_overrides or {})
        if session_id.strip():
            sess = runner.get_session(session_id.strip())
            mapping = {**sess["default_mapping"], **mapping}
            interp = sess.get("interpretation") or {}
            ctx = {**interp, **ctx}
        if study is not None:
            study = verify_and_enrich_target_rct(
                study,
                intervention=str(ctx.get("intervention_query_en") or mapping.get("active_value") or ""),
                comparator=str(ctx.get("comparator_query_en") or mapping.get("comparator_value") or ""),
                target_condition=str(ctx.get("condition_query_en") or ""),
                target_outcome=str(ctx.get("primary_outcome") or ctx.get("outcome_query_en") or ""),
                project_mode=mode,
                confirm_arm_mismatch=bool(confirm_arm_mismatch),
                confirm_protocol_mismatch=bool(confirm_protocol_mismatch),
            )
        if design_mode == "grace_period_ccw":
            protocol = json.loads(_read_repo_text("examples", "tte_grace_period_ccw_v1.json"))
            if ctx.get("search_query_en") or ctx.get("intervention_query_en"):
                protocol["study_title"] = f"宽限期 CCW 草案：{ctx.get('intervention_query_en') or ctx.get('search_query_en')}"
            normalized = validate_ccw_protocol_dict(protocol)
            return {"protocol_draft": normalized, "column_mapping": mapping, "design_mode": design_mode, "schema": "tte-ccw-v1"}
        protocol, _dgp = build_simulated_protocol_and_dgp(target_rct=study, pico=ctx, sim_config={"weight_method": mapping.get("weight_method", "overlap")})
        if design_mode == "landmark":
            protocol["time_zero"]["landmark_day"] = int(protocol["time_zero"].get("landmark_day") or 30)
        if mode == SOURCE_MODE_USER_COHORT:
            protocol["project_mode"] = SOURCE_MODE_USER_COHORT
        normalized = validate_protocol_dict(protocol)
        return {"protocol_draft": normalized, "column_mapping": mapping, "design_mode": design_mode, "schema": "tte-protocol-v1"}

    @mcp.tool(description="Validate and normalize a seven-element static or landmark TTE protocol.")
    def validate_tte_protocol(protocol: dict, session_id: str = "") -> dict:
        normalized = validate_protocol_dict(protocol)
        checked = runner.validate(normalized)
        return {
            "valid": True,
            "status": "valid",
            "violations": [],
            "source_mode": checked.get("project_mode", checked.get("source_mode", "synthetic_only")),
            "normalized_protocol": checked,
            "session_id": session_id or None,
        }

    @mcp.tool(description="Validate a tte-ccw-v1 bilateral grace-period protocol.")
    def validate_ccw_protocol(ccw_protocol: dict) -> dict:
        try:
            normalized = validate_ccw_protocol_dict(ccw_protocol)
        except ContractError as exc:
            return {"valid": False, "violations": [{"code": exc.code, "message": str(exc)}]}
        grace = float(normalized["grace_period_spec"]["grace_period_days"])
        delta = float(normalized["grace_period_spec"]["period_length_days"])
        return {
            "valid": True,
            "violations": [],
            "normalized_protocol": normalized,
            "person_period_grid": {"m_g": int(round(grace / delta)), "K": int(round(float(normalized["grace_period_spec"]["max_followup_days"]) / delta))},
            "no_treatment_confounder_feedback_boundary_zh": "期前时变协变量只使用决策前测量；存在治疗-混杂反馈时，本 P1 引擎的分臂 IPCW 不再识别边际策略效应。",
        }

    @mcp.tool(description="Run eligibility, weighting, ESS, and balance gates without the survival bootstrap.")
    def diagnose_tte_feasibility(
        protocol: dict | None = None,
        session_id: str = "",
        column_mapping: dict | None = None,
        sim_config: dict | None = None,
        project_mode: str = "",
    ) -> dict:
        cfg = sim_config or {}
        with tempfile.TemporaryDirectory(prefix="emulatrx-diagnose-") as tmp:
            work = Path(tmp)
            if session_id.strip():
                sess = runner.get_session(session_id.strip())
                mapping = {**(sess.get("default_mapping") or {}), **(column_mapping or {})}
                ingest_user_cohort(sess["cohort_path"], mapping, out_dir=work / "ingest")
                protocol_path = work / "ingest" / "normalized_protocol.json"
                cohort_path = work / "ingest" / "standardized_cohort.csv"
            else:
                draft = protocol
                if draft is None:
                    draft, dgp = build_simulated_protocol_and_dgp(sim_config=cfg)
                    (work / "dgp.json").write_text(json.dumps(dgp), encoding="utf-8")
                else:
                    dgp = None
                normalized = validate_protocol_dict(draft)
                protocol_path = work / "protocol.json"
                protocol_path.write_text(json.dumps(normalized), encoding="utf-8")
                generate_synthetic_cohort(
                    protocol_path,
                    seed=int(cfg.get("seed", 1)),
                    cases=int(cfg.get("cases", 400)),
                    out_dir=work / "synth",
                    dgp_params_path=(work / "dgp.json") if dgp is not None else None,
                    id_prefix="SIM" if project_mode == SOURCE_MODE_SIMULATED_COHORT or normalized.get("project_mode") == SOURCE_MODE_SIMULATED_COHORT else "SYN",
                )
                cohort_path = work / "synth" / "cohort.csv"
            summary, code = diagnose_cohort(
                protocol_path,
                cohort_path,
                weight_method=str((column_mapping or cfg).get("weight_method") or "overlap"),
                out_dir=work / "diag",
            )
            attrition_path = work / "diag" / "attrition_table.csv"
            balance_path = work / "diag" / "balance_diagnostics.csv"
            import csv

            waterfall = list(csv.DictReader(attrition_path.read_text(encoding="utf-8").splitlines()))
            smd_table = list(csv.DictReader(balance_path.read_text(encoding="utf-8").splitlines()))
        return {
            "gate_status": "passed" if code == 0 else "blocked",
            "waterfall": waterfall,
            "smd_table": smd_table,
            "weight_summary": {
                "weight_method": summary.get("weight_method"),
                "ess_diagnostics": summary.get("ess_diagnostics"),
                "max_abs_smd_unweighted": summary.get("max_abs_smd_unweighted"),
                "max_abs_smd_weighted": summary.get("max_abs_smd_weighted"),
                "propensity_diagnostics": summary.get("propensity_diagnostics"),
            },
            "remediation_hints": _remediation(summary.get("gate_failures") or []),
            "diagnostics": summary,
        }

    @mcp.tool(description="Run static or landmark TTE for a user cohort, a simulated cohort, or the synthetic benchmark.")
    def run_static_or_landmark_tte(
        project_mode: str = "synthetic_benchmark",
        protocol: dict | None = None,
        session_id: str = "",
        column_mapping: dict | None = None,
        nct_id: str = "",
        target_rct: dict | None = None,
        idea_text: str = "",
        condition: str = "",
        intervention: str = "",
        comparator: str = "",
        outcome: str = "",
        cases: int = 800,
        seed: int = 42,
        bootstrap_iterations: int = 20,
        weight_method: str = "overlap",
        confounding_strength: str = "medium",
        effect_direction: str = "protective_moderate",
        competing_rate_level: str = "medium",
        confirm_arm_mismatch: bool = False,
        confirm_protocol_mismatch: bool = False,
        wait_seconds: float = 5.0,
    ) -> dict:
        mode = project_mode
        if mode in ("synthetic_benchmark", "synthetic_only"):
            if protocol is None:
                raise ContractError("MISSING_PROTOCOL", "synthetic runs require protocol")
            started = runner.start(protocol, cases=int(cases), seed=int(seed), bootstrap_iterations=int(bootstrap_iterations), weight_method=weight_method)
        elif mode in (SOURCE_MODE_SIMULATED_COHORT, "simulated_cohort"):
            pico = parse_research_idea_pico(idea_text, explicit_comparator=comparator.strip()) if idea_text.strip() else None
            study = _load_target(nct_id, target_rct)
            started = runner.start_simulated_cohort(
                target_rct=study,
                pico=pico,
                idea_text=idea_text,
                condition=condition,
                intervention=intervention,
                comparator=comparator,
                outcome=outcome,
                sim_config={
                    "cases": int(cases),
                    "confounding_strength": confounding_strength,
                    "effect_direction": effect_direction,
                    "competing_rate_level": competing_rate_level,
                    "weight_method": weight_method,
                    "seed": int(seed),
                    "bootstrap_iterations": int(bootstrap_iterations),
                },
                confirm_arm_mismatch=confirm_arm_mismatch,
                confirm_protocol_mismatch=confirm_protocol_mismatch,
            )
        else:
            if not session_id.strip():
                raise ContractError("MISSING_SESSION", "user_cohort runs require session_id")
            sess = runner.get_session(session_id.strip())
            study = _load_target(nct_id, target_rct)
            mapping = column_mapping or recommend_column_mapping(sess["profile"], target_rct=study)
            started = runner.start_user_cohort(
                session_id.strip(),
                mapping,
                target_rct=study,
                seed=int(seed),
                bootstrap_iterations=int(bootstrap_iterations),
                weight_method=weight_method,
                condition_override=condition,
                intervention_override=intervention,
                comparator_override=comparator,
                outcome_override=outcome,
                confirm_arm_mismatch=confirm_arm_mismatch,
                confirm_protocol_mismatch=confirm_protocol_mismatch,
            )
        status = _wait(started["run_id"], wait_seconds)
        return {"run_id": started["run_id"], "state": status.get("state"), "gate_status": status.get("gate_status"), "project_mode": status.get("project_mode"), "status": status}

    @mcp.tool(description="Legacy alias: start a synthetic benchmark run.")
    def start_synthetic_tte(protocol: dict, cases: int = 2000, seed: int = 42, bootstrap_iterations: int = 100) -> dict:
        return runner.start(protocol, cases=cases, seed=seed, bootstrap_iterations=bootstrap_iterations)

    @mcp.tool(description="Legacy alias: start a no-data simulated target-RCT rehearsal.")
    def start_simulated_rct_tte(
        nct_id: str = "",
        idea_text: str = "",
        condition: str = "",
        intervention: str = "",
        comparator: str = "",
        outcome: str = "",
        cases: int = 1200,
        confounding_strength: str = "medium",
        effect_direction: str = "protective_moderate",
        competing_rate_level: str = "medium",
        weight_method: str = "overlap",
        seed: int = 42,
        bootstrap_iterations: int = 80,
        confirm_arm_mismatch: bool = False,
        confirm_protocol_mismatch: bool = False,
    ) -> dict:
        return run_static_or_landmark_tte(
            project_mode="simulated_cohort",
            nct_id=nct_id,
            idea_text=idea_text,
            condition=condition,
            intervention=intervention,
            comparator=comparator,
            outcome=outcome,
            cases=cases,
            confounding_strength=confounding_strength,
            effect_direction=effect_direction,
            competing_rate_level=competing_rate_level,
            weight_method=weight_method,
            seed=seed,
            bootstrap_iterations=bootstrap_iterations,
            confirm_arm_mismatch=confirm_arm_mismatch,
            confirm_protocol_mismatch=confirm_protocol_mismatch,
            wait_seconds=0,
        )

    @mcp.tool(description="Legacy alias: start an owned-cohort TTE run.")
    def start_user_cohort_tte(
        session_id: str,
        mapping: dict | None = None,
        nct_id: str = "",
        condition: str = "",
        intervention: str = "",
        comparator: str = "",
        outcome: str = "",
        weight_method: str = "overlap",
        seed: int = 42,
        bootstrap_iterations: int = 80,
        confirm_arm_mismatch: bool = False,
        confirm_protocol_mismatch: bool = False,
    ) -> dict:
        return run_static_or_landmark_tte(
            project_mode="user_cohort",
            session_id=session_id,
            column_mapping=mapping,
            nct_id=nct_id,
            condition=condition,
            intervention=intervention,
            comparator=comparator,
            outcome=outcome,
            weight_method=weight_method,
            seed=seed,
            bootstrap_iterations=bootstrap_iterations,
            confirm_arm_mismatch=confirm_arm_mismatch,
            confirm_protocol_mismatch=confirm_protocol_mismatch,
            wait_seconds=0,
        )

    @mcp.tool(description="Run a synthetic bilateral grace-period clone-censor-weight analysis and persist the run directory.")
    def run_grace_period_ccw(
        ccw_protocol: dict | None = None,
        synthetic_scenario: str = "null_effect",
        seed: int = 42,
        cases: int = 800,
        bootstrap_replicates: int = 20,
        session_id: str = "",
    ) -> dict:
        protocol = ccw_protocol or json.loads(_read_repo_text("examples", "tte_grace_period_ccw_v1.json"))
        dgp = {"true_log_hr_event1": 0.0, "true_log_hr_event2": 0.0} if synthetic_scenario == "null_effect" else (
            {"true_log_hr_event1": -0.45, "true_log_hr_event2": 0.0} if synthetic_scenario == "protective_effect" else None
        )
        if synthetic_scenario not in ("null_effect", "protective_effect", "custom") and ccw_protocol is None:
            raise ContractError("INVALID_CCW_SCENARIO", "synthetic_scenario must be null_effect, protective_effect, or custom")
        with runner.execution_slot(), tempfile.TemporaryDirectory(prefix="emulatrx-ccw-") as tmp:
            out = Path(tmp) / "out"
            result, code = run_ccw_synthetic_pipeline(
                protocol,
                seed=int(seed),
                n_cases=int(cases),
                out_dir=out,
                bootstrap_iterations=int(bootstrap_replicates),
                dgp_params=dgp,
            )
            status = runner.adopt_ccw_directory(
                out,
                gate_status="passed" if code == 0 else "blocked",
                seed=int(seed),
                case_count=int(cases),
                session_id=session_id or None,
            )
        return {"run_id": status["run_id"], "gate_status": status["gate_status"], "state": status["state"], "run_type": "grace_period_ccw", "engine_status": result.get("status")}

    def _status_payload(run_id: str) -> dict:
        status = runner.get(run_id)
        state = status.get("state")
        mapped = {"completed": "completed", "blocked": "gate_blocked", "failed": "failed", "queued": "running", "running": "running"}.get(state, state)
        return {
            "run_id": run_id,
            "user_namespace": status.get("user_namespace") or "default",
            "state": state,
            "status": mapped,
            "gate_status": status.get("gate_status"),
            "project_mode": status.get("project_mode"),
            "kind": status.get("kind"),
            "artifacts": status.get("available_artifacts", runner.artifacts(run_id) if state in ("completed", "blocked") else []),
            "error": status.get("error"),
        }

    @mcp.tool(description="Read run lifecycle, gate status, and artifact names.")
    def get_tte_run_status(run_id: str) -> dict:
        return _status_payload(run_id)

    @mcp.tool(description="Legacy alias of get_tte_run_status that also returns the bounded run payload.")
    def get_tte_run(run_id: str) -> dict:
        result = runner.get(run_id)
        if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > 32 * 1024:
            compact = {
                "run_id": result["run_id"],
                "state": result["state"],
                "project_mode": result.get("project_mode"),
                "gate_status": result.get("gate_status"),
                "cohort_rows": result.get("cohort_rows"),
                "eligible_rows": result.get("eligible_rows"),
                "available_artifacts": result.get("available_artifacts", []),
                "truncated": True,
                "hint": "Read specific artifacts with read_tte_artifact",
            }
            if isinstance(result.get("estimates"), dict):
                compact["co_presented_comparison_table"] = result["estimates"].get("co_presented_comparison_table", [])
                compact["source_mode"] = result["estimates"].get("source_mode")
                compact["report_badge"] = result["estimates"].get("report_badge")
            return compact
        return result

    def _estimates_path(run_id: str) -> tuple[Path, str]:
        root = runner._path(run_id)
        status = runner._status(root)
        kind = status.get("kind")
        if kind == "grace_period_ccw":
            return root / "ccw_estimates.json", "grace_period_ccw"
        if kind == "synthetic_tte":
            return root / "estimates" / "estimates.json", "static_or_landmark"
        return root / "bundle" / "estimates.json", "static_or_landmark"

    @mcp.tool(description="Return the clinician-facing 3-tier summary. Blocked runs omit point estimates.")
    def get_tte_result_summary(run_id: str) -> dict:
        root = runner._path(run_id)
        status = runner._status(root)
        kind = status.get("kind")
        project_mode = status.get("project_mode") or SOURCE_MODE_SYNTHETIC_ONLY
        run_type = "grace_period_ccw" if kind == "grace_period_ccw" else "static_or_landmark"
        if project_mode == SOURCE_MODE_USER_COHORT:
            notice = "user_cohort 是观察性目标试验模拟，结论受一致性、条件可交换、正值性和条件独立删失约束，不是随机对照试验证据。"
        else:
            notice = "simulated_cohort / synthetic_only 是非真实患者数据、非临床疗效证据，只用于方法学演示和方案预演。"
        gate = status.get("gate_status")
        estimate_path, _run_type = _estimates_path(run_id)
        if gate != "passed" or not estimate_path.is_file():
            diagnostics_path = (
                root / "ccw_diagnostics.json"
                if kind == "grace_period_ccw"
                else root / "diagnostics" / "diagnostics_summary.json"
            )
            if not diagnostics_path.is_file():
                diagnostics_path = root / "bundle" / "diagnostics_summary.json"
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.is_file() else {}
            failures = diagnostics.get("gate_failures") or []
            return {
                "run_id": run_id,
                "user_namespace": status.get("user_namespace") or "default",
                "run_type": run_type,
                "project_mode": project_mode,
                "evidence_boundary_notice": notice,
                "gate_status": gate or "blocked",
                "diagnostics": diagnostics,
                "remediation_hints": _remediation(failures, is_ccw=kind == "grace_period_ccw"),
                "estimates_omitted_reason": "gate_status is not passed; tier-3 and other point estimates are omitted",
                "clinical_comparison_ladder": None,
                "what_tte_controlled_beyond_psm": None,
            }
        estimates = json.loads(estimate_path.read_text(encoding="utf-8"))
        horizons = estimates.get("horizon_estimates") or estimates.get("co_presented_comparison_table") or []
        return {
            "run_id": run_id,
            "user_namespace": status.get("user_namespace") or "default",
            "run_type": run_type,
            "project_mode": project_mode,
            "evidence_boundary_notice": notice,
            "gate_status": "passed",
            "clinical_comparison_ladder": estimates.get("clinical_comparison_ladder"),
            "what_tte_controlled_beyond_psm": estimates.get("what_tte_controlled_beyond_psm"),
            "horizon_estimates": horizons,
            "hazard_ratios": estimates.get("contrasts") or estimates.get("naive_immortal_time_comparison"),
            "waterfall": estimates.get("waterfall"),
            "diagnostics": {"weight_method": estimates.get("weight_method"), "weighting_scheme": (estimates.get("clinical_comparison_ladder") or {}).get("tier_3_target_trial_emulation", {}) and (estimates.get("clinical_comparison_ladder") or {}).get("tier_3_target_trial_emulation", {}).get("weighting_scheme")},
            "causal_assumptions": estimates.get("causal_assumptions") or estimates.get("identification_assumptions"),
        }

    def _artifact_file(run_id: str, artifact_name: str) -> Path:
        aliases = {
            "report.html": "ccw_preview_report.html",
            "estimates.json": "ccw_estimates.json",
            "cif_curves.csv": "curves/ccw_cif_curves.csv",
        }
        root = runner._path(run_id)
        status = runner._status(root)
        name = artifact_name
        if status.get("kind") == "grace_period_ccw" and name in aliases and not (root / name).is_file():
            name = aliases[name]
        if status.get("kind") != "grace_period_ccw" and name in ("report.html", "audit_report.md", "estimates.json", "run_manifest.json", "cif_curves.csv"):
            bundle = root / "bundle"
            estimates = root / "estimates"
            if name == "cif_curves.csv":
                curves = sorted((bundle / "curves").glob("*.csv")) if (bundle / "curves").is_dir() else sorted((estimates / "curves").glob("*.csv"))
                if not curves:
                    raise ContractError("ARTIFACT_NOT_FOUND", "cif curves are unavailable")
                return curves[0]
            for folder in (bundle, estimates, root):
                candidate = folder / name
                if candidate.is_file():
                    return candidate
        return runner.artifact_path(run_id, name)

    @mcp.tool(description="Export a run artifact as text, parsed JSON, or a base64 chunk with its SHA-256.")
    def export_tte_artifact(run_id: str, artifact_name: str, mode: str = "text", offset: int = 0, limit: int = 16384) -> dict:
        import hashlib

        path = _artifact_file(run_id, artifact_name)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if mode == "json":
            return {"artifact_name": artifact_name, "sha256": digest, "json": json.loads(raw.decode("utf-8"))}
        if mode == "base64_chunk":
            chunk = runner.read_artifact(run_id, str(path.relative_to(runner._path(run_id))), offset=offset, limit=limit)
            chunk["sha256"] = digest
            return chunk
        text = raw.decode("utf-8")
        return {"artifact_name": artifact_name, "sha256": digest, "mode": "text", "text": text[:200000]}

    @mcp.tool(description="Legacy alias: read an artifact as a base64 chunk of at most 16 KiB.")
    def read_tte_artifact(run_id: str, name: str, offset: int = 0, limit: int = 16384) -> dict:
        return runner.read_artifact(run_id, name, offset=offset, limit=limit)

    @mcp.resource("emulatrx://schemas/tte-protocol-v1", mime_type="application/schema+json")
    def tte_protocol_schema() -> str:
        return _read_repo_text("schemas", "tte-protocol-v1.schema.json")

    @mcp.resource("emulatrx://schemas/tte-ccw-v1", mime_type="application/schema+json")
    def tte_ccw_schema() -> str:
        return _read_repo_text("schemas", "tte-ccw-v1.schema.json")

    @mcp.resource("emulatrx://samples/catalog", mime_type="application/json")
    def sample_catalog() -> str:
        from emulatrx_tte.sample_cohorts import SAMPLE_COHORT_SPECS, SAMPLE_ID_ALIASES

        catalog = {
            "samples": [
                {"sample_id": key, "title_zh": spec.get("title_zh"), "filename": spec.get("filename")}
                for key, spec in SAMPLE_COHORT_SPECS.items()
            ],
            "aliases": SAMPLE_ID_ALIASES,
            "ccw_scenarios": ["null_effect", "protective_effect"],
        }
        return json.dumps(catalog, ensure_ascii=False)

    @mcp.resource("emulatrx://guide/error-and-gate-codes", mime_type="text/plain")
    def error_guide() -> str:
        return (
            "EmulatRx gate and error codes\n"
            "HTTP 401: Web Basic and MCP Bearer are different credentials.\n"
            "HTTP 413 COHORT_FILE_TOO_LARGE: cohort upload exceeds 25 MB.\n"
            "HTTP 422 ARM_MISMATCH_CONFIRMATION_REQUIRED: arm mismatch needs confirm_arm_mismatch.\n"
            "HTTP 422 PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED: PICO-T mismatch needs confirm_protocol_mismatch.\n"
            "Exit 2: protocol contract violation. Fix the protocol and write to a new output directory.\n"
            "Exit 3 POSITIVITY_VIOLATION / ESS_RATIO_BELOW_MIN / SMD_EXCEEDS_MAX / POST_GRACE_SMD_EXCEEDS_MAX: diagnostic gate blocked. Do not report point estimates.\n"
            "PATH_TRAVERSAL_BLOCKED: HTTP file_path must stay inside EMULATRX_MCP_ALLOWED_INGEST_DIR.\n"
            "UPLOAD_TICKET_EXPIRED / UPLOAD_TICKET_INVALID: tickets are single-use and live for 600 seconds.\n"
        )
