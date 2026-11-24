"""TTE seven-element protocol schema validation, observable-rule checking, and report publishing."""

from __future__ import annotations

from pathlib import Path
import re

from ._common import (
    ALLOWED_SOURCE_MODES,
    ContractError,
    METHODOLOGICAL_DISCLAIMER,
    PROTOCOL_VERSION,
    atomic_publish,
    code_provenance,
    dependency_versions,
    is_strict_int,
    is_strict_number,
    json_bytes,
    read_strict_json,
    sha256_bytes,
    sha256_file,
    utc_now_iso,
)


REQUIRED_ROOT_FIELDS = (
    "protocol_version",
    "study_title",
    "simulation_or_data_label",
    "eligibility",
    "time_zero",
    "treatment_strategies",
    "assignment_mode",
    "followup",
    "outcomes",
    "estimand",
)
OPTIONAL_ROOT_FIELDS = ("covariates", "methodological_disclaimer", "project_mode", "target_rct")
ALLOWED_OPERATORS = frozenset(("==", "!=", "<", "<=", ">", ">=", "in", "not_in"))
REFERENCE_ROLES = frozenset(("control", "reference", "comparator"))
INVESTIGATIONAL_ROLES = frozenset(("active", "experimental"))
ALLOWED_ROLES = REFERENCE_ROLES | INVESTIGATIONAL_ROLES
ALLOWED_ASSIGNMENT_MODES = frozenset(("observational_baseline", "observational_landmark"))
ALLOWED_COMPETING_HANDLING = frozenset(("none", "explicit_cif_and_subdistribution"))
ALLOWED_ESTIMANDS = frozenset(("ATE", "ATO", "PAIRWISE_ATO", "GENERALIZED_ATO"))
ALLOWED_EFFECT_MEASURES = frozenset(("RD", "RR_CIF", "sHR", "csHR", "HR"))


def _require_nonempty_str(value: object, field_name: str, code: str = "INVALID_TYPE") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(code, f"'{field_name}' must be a non-empty string")
    return value.strip()


def _check_exact_keys(
    obj: dict,
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
    context: str,
    missing_code: str = "MISSING_FIELD",
) -> None:
    allowed = set(required) | set(optional)
    missing = sorted(set(required) - set(obj.keys()))
    if missing:
        raise ContractError(missing_code, f"Missing required field(s) in {context}: {', '.join(missing)}")
    unknown = sorted(set(obj.keys()) - allowed)
    if unknown:
        raise ContractError("UNKNOWN_FIELD", f"Unknown field(s) in {context}: {', '.join(unknown)}")


def _validate_criterion_value(value: object, operator: str, context: str) -> object:
    if operator in ("in", "not_in"):
        if not isinstance(value, list) or len(value) == 0:
            raise ContractError("INVALID_ELIGIBILITY", f"{context}.value must be a non-empty list for '{operator}'")
        out = []
        for idx, item in enumerate(value):
            if isinstance(item, bool):
                out.append(item)
            elif is_strict_number(item) or isinstance(item, str):
                out.append(item)
            else:
                raise ContractError("INVALID_ELIGIBILITY", f"{context}.value[{idx}] must be scalar")
        return out
    if isinstance(value, bool) or is_strict_number(value) or isinstance(value, str):
        return value
    raise ContractError("INVALID_ELIGIBILITY", f"{context}.value must be number, string, or boolean")


def _validate_criteria_list(
    items: object,
    *,
    list_name: str,
    landmark_day: int,
    seen_ids: set[str],
) -> list[dict]:
    if not isinstance(items, list):
        raise ContractError("INVALID_TYPE", f"eligibility.{list_name} must be a list")
    normalized_items: list[dict] = []
    for idx, raw_item in enumerate(items):
        ctx = f"eligibility.{list_name}[{idx}]"
        if not isinstance(raw_item, dict):
            raise ContractError("INVALID_TYPE", f"{ctx} must be an object")
        _check_exact_keys(
            raw_item,
            required=("id", "variable", "operator", "value", "assessed_day_offset"),
            optional=("description",),
            context=ctx,
        )
        crit_id = _require_nonempty_str(raw_item["id"], f"{ctx}.id", "INVALID_ELIGIBILITY")
        if crit_id in seen_ids:
            raise ContractError("INVALID_ELIGIBILITY", f"Duplicate eligibility criterion id: '{crit_id}'")
        seen_ids.add(crit_id)
        variable = _require_nonempty_str(raw_item["variable"], f"{ctx}.variable", "INVALID_ELIGIBILITY")
        operator = _require_nonempty_str(raw_item["operator"], f"{ctx}.operator", "INVALID_ELIGIBILITY")
        if operator not in ALLOWED_OPERATORS:
            raise ContractError(
                "INVALID_ELIGIBILITY",
                f"{ctx}.operator '{operator}' must be one of {sorted(ALLOWED_OPERATORS)}",
            )
        offset = raw_item["assessed_day_offset"]
        if not is_strict_int(offset):
            raise ContractError("INVALID_TYPE", f"{ctx}.assessed_day_offset must be an integer")
        if offset > landmark_day:
            raise ContractError(
                "POST_TIME_ZERO_ELIGIBILITY",
                f"Criterion '{crit_id}' on variable '{variable}' has assessed_day_offset={offset} "
                f"> effective baseline/landmark_day={landmark_day}. "
                "Hint: Eligibility criteria must be assessed at or before time_zero (or landmark_day) "
                "to prevent look-ahead selection / immortal-time bias; move post-baseline events to "
                "outcomes/censoring or explicitly advance time_zero.landmark_day.",
            )
        val = _validate_criterion_value(raw_item["value"], operator, ctx)
        entry = {
            "id": crit_id,
            "variable": variable,
            "operator": operator,
            "value": val,
            "assessed_day_offset": offset,
        }
        if "description" in raw_item:
            if not isinstance(raw_item["description"], str):
                raise ContractError("INVALID_TYPE", f"{ctx}.description must be a string")
            entry["description"] = raw_item["description"]
        normalized_items.append(entry)
    return normalized_items


def validate_protocol_dict(obj: dict) -> dict:
    """Validate and normalize a TTE protocol v1 dictionary, raising ContractError on violations."""
    if not isinstance(obj, dict):
        raise ContractError("INVALID_TYPE", "Protocol must be a JSON object")

    # Explicitly prioritize MISSING_TIME_ZERO before generic MISSING_FIELD when time_zero is absent/empty.
    if "time_zero" not in obj or obj["time_zero"] is None or obj["time_zero"] == {}:
        raise ContractError(
            "MISSING_TIME_ZERO",
            "Protocol is missing an explicit non-empty 'time_zero' definition. "
            "Hint: Provide time_zero={'anchor_event': '<baseline_anchor>', 'day_offset': 0}.",
        )

    _check_exact_keys(
        obj,
        required=REQUIRED_ROOT_FIELDS,
        optional=OPTIONAL_ROOT_FIELDS,
        context="protocol",
    )

    if not isinstance(obj["protocol_version"], str):
        raise ContractError("INVALID_TYPE", "protocol_version must be a string")
    if obj["protocol_version"] != PROTOCOL_VERSION:
        raise ContractError("UNKNOWN_VERSION", f"Unsupported protocol_version: {obj['protocol_version']}")

    study_title = _require_nonempty_str(obj["study_title"], "study_title")
    sim_label = _require_nonempty_str(obj["simulation_or_data_label"], "simulation_or_data_label")

    # 1. time_zero (evaluated before eligibility so landmark_day is resolved)
    tz = obj["time_zero"]
    if not isinstance(tz, dict) or len(tz) == 0:
        raise ContractError(
            "MISSING_TIME_ZERO",
            "time_zero must be a non-empty JSON object defining anchor_event and day_offset=0.",
        )
    _check_exact_keys(
        tz,
        required=("anchor_event", "day_offset"),
        optional=("landmark_day", "description"),
        context="time_zero",
        missing_code="MISSING_TIME_ZERO",
    )
    anchor_event = _require_nonempty_str(tz["anchor_event"], "time_zero.anchor_event", "MISSING_TIME_ZERO")
    if not is_strict_int(tz["day_offset"]) or tz["day_offset"] != 0:
        raise ContractError(
            "INVALID_TIME_ZERO",
            "time_zero.day_offset must be integer 0. Hint: Express all offsets relative to time_zero (day 0).",
        )
    landmark_day = tz.get("landmark_day", 0)
    if not is_strict_int(landmark_day) or landmark_day < 0:
        raise ContractError("INVALID_TIME_ZERO", "time_zero.landmark_day must be an integer >= 0")
    normalized_tz: dict = {
        "anchor_event": anchor_event,
        "day_offset": 0,
        "landmark_day": landmark_day,
    }
    if "description" in tz:
        if not isinstance(tz["description"], str):
            raise ContractError("INVALID_TYPE", "time_zero.description must be a string")
        normalized_tz["description"] = tz["description"]

    # 2. assignment_mode
    assignment_mode = obj["assignment_mode"]
    if not isinstance(assignment_mode, str) or assignment_mode not in ALLOWED_ASSIGNMENT_MODES:
        raise ContractError(
            "INVALID_ASSIGNMENT_MODE",
            f"assignment_mode must be one of {sorted(ALLOWED_ASSIGNMENT_MODES)}",
        )
    if assignment_mode == "observational_baseline" and landmark_day != 0:
        raise ContractError(
            "INVALID_TIME_ZERO",
            f"assignment_mode='observational_baseline' requires time_zero.landmark_day == 0 "
            f"(got {landmark_day}) so post-time-zero eligibility criteria and covariates cannot "
            "bypass baseline checks. Hint: Set time_zero.landmark_day=0 for baseline assignment, "
            "or set assignment_mode='observational_landmark' for an explicit landmark design.",
        )
    if assignment_mode == "observational_landmark" and landmark_day <= 0:
        raise ContractError(
            "INVALID_ASSIGNMENT_MODE",
            "assignment_mode='observational_landmark' requires time_zero.landmark_day > 0. "
            "Hint: Set time_zero.landmark_day > 0 or use 'observational_baseline'.",
        )

    # 3. eligibility
    elig = obj["eligibility"]
    if not isinstance(elig, dict):
        raise ContractError("INVALID_TYPE", "eligibility must be an object")
    _check_exact_keys(elig, required=("inclusion_criteria", "exclusion_criteria"), context="eligibility")
    seen_crit_ids: set[str] = set()
    norm_inc = _validate_criteria_list(
        elig["inclusion_criteria"],
        list_name="inclusion_criteria",
        landmark_day=landmark_day,
        seen_ids=seen_crit_ids,
    )
    norm_exc = _validate_criteria_list(
        elig["exclusion_criteria"],
        list_name="exclusion_criteria",
        landmark_day=landmark_day,
        seen_ids=seen_crit_ids,
    )

    # 4. treatment_strategies
    strategies = obj["treatment_strategies"]
    if not isinstance(strategies, list) or len(strategies) < 2:
        raise ContractError(
            "INVALID_STRATEGIES",
            "treatment_strategies must be a list of at least 2 distinct strategies. "
            "Hint: Define both an active/experimental strategy and an explicit comparator strategy.",
        )
    seen_strat_ids: set[str] = set()
    norm_strategies: list[dict] = []
    for idx, strat in enumerate(strategies):
        ctx = f"treatment_strategies[{idx}]"
        if not isinstance(strat, dict):
            raise ContractError("INVALID_STRATEGIES", f"{ctx} must be an object")
        _check_exact_keys(
            strat,
            required=("id", "label", "role"),
            optional=("description",),
            context=ctx,
            missing_code="INVALID_STRATEGIES",
        )
        sid = _require_nonempty_str(strat["id"], f"{ctx}.id", "INVALID_STRATEGIES")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sid):
            raise ContractError("INVALID_STRATEGIES", f"{ctx}.id '{sid}' contains invalid characters")
        if sid in seen_strat_ids:
            raise ContractError(
                "INVALID_STRATEGIES",
                f"Duplicate treatment strategy id '{sid}'. Hint: Assign each arm a unique id.",
            )
        seen_strat_ids.add(sid)
        slabel = _require_nonempty_str(strat["label"], f"{ctx}.label", "INVALID_STRATEGIES")
        srole = _require_nonempty_str(strat["role"], f"{ctx}.role", "INVALID_STRATEGIES")
        if srole not in ALLOWED_ROLES:
            raise ContractError("INVALID_STRATEGIES", f"{ctx}.role must be one of {sorted(ALLOWED_ROLES)}")
        norm_s = {"id": sid, "label": slabel, "role": srole}
        if "description" in strat:
            if not isinstance(strat["description"], str):
                raise ContractError("INVALID_TYPE", f"{ctx}.description must be a string")
            norm_s["description"] = strat["description"]
        norm_strategies.append(norm_s)

    ref_indices = [idx for idx, s in enumerate(norm_strategies) if s["role"] in REFERENCE_ROLES]
    if len(ref_indices) != 1:
        raise ContractError(
            "INVALID_STRATEGIES",
            f"treatment_strategies must contain exactly 1 reference/comparator strategy with role in "
            f"{sorted(REFERENCE_ROLES)} (found {len(ref_indices)}). "
            f"Hint: Designate the single reference arm as 'control'/'comparator'/'reference' at index 0 "
            f"and mark remaining arm(s) with role in {sorted(INVESTIGATIONAL_ROLES)}.",
        )
    if ref_indices[0] != 0:
        raise ContractError(
            "INVALID_STRATEGIES",
            f"The reference/comparator strategy ('{norm_strategies[ref_indices[0]]['id']}', "
            f"role='{norm_strategies[ref_indices[0]]['role']}') must be placed first "
            "(treatment_strategies[0]) to match deterministic reference arm indexing.",
        )

    # 5. followup
    followup = obj["followup"]
    if not isinstance(followup, dict):
        raise ContractError("INVALID_TYPE", "followup must be an object")
    _check_exact_keys(followup, required=("horizon_days", "administrative_end_day"), context="followup")
    raw_horizons = followup["horizon_days"]
    if not isinstance(raw_horizons, list):
        raise ContractError("INVALID_TYPE", "followup.horizon_days must be a list of positive numbers")
    horizons: list[float] = []
    for idx, h in enumerate(raw_horizons):
        if not is_strict_number(h) or float(h) <= 0.0:
            raise ContractError("INVALID_FOLLOWUP", f"followup.horizon_days[{idx}] must be > 0")
        if landmark_day > 0 and float(h) <= float(landmark_day):
            raise ContractError(
                "INVALID_FOLLOWUP",
                f"followup.horizon_days[{idx}]={h} must exceed time_zero.landmark_day={landmark_day}",
            )
        horizons.append(float(h))
    if len(set(horizons)) != len(horizons):
        raise ContractError("INVALID_FOLLOWUP", "followup.horizon_days must not contain duplicate horizons")
    horizons_sorted = sorted(horizons)

    admin_end = followup["administrative_end_day"]
    if not is_strict_number(admin_end) or float(admin_end) <= 0.0:
        raise ContractError("INVALID_FOLLOWUP", "followup.administrative_end_day must be > 0")
    admin_end_f = float(admin_end)
    if landmark_day > 0 and admin_end_f <= float(landmark_day):
        raise ContractError(
            "INVALID_FOLLOWUP",
            f"followup.administrative_end_day ({admin_end_f}) must exceed landmark_day ({landmark_day})",
        )
    if horizons_sorted and admin_end_f <= max(horizons_sorted):
        raise ContractError(
            "INVALID_FOLLOWUP",
            f"followup.administrative_end_day ({admin_end_f}) must be strictly greater than "
            f"max(horizon_days) ({max(horizons_sorted)}).",
        )

    # 6. outcomes
    outcomes = obj["outcomes"]
    if not isinstance(outcomes, dict):
        raise ContractError("INVALID_TYPE", "outcomes must be an object")
    _check_exact_keys(
        outcomes,
        required=("primary_event_code", "censor_code", "competing_event_handling"),
        optional=("competing_event_codes",),
        context="outcomes",
    )
    if not is_strict_int(outcomes["primary_event_code"]) or outcomes["primary_event_code"] != 1:
        raise ContractError("INVALID_OUTCOMES", "outcomes.primary_event_code must be integer 1")
    if not is_strict_int(outcomes["censor_code"]) or outcomes["censor_code"] != 0:
        raise ContractError("INVALID_OUTCOMES", "outcomes.censor_code must be integer 0")

    raw_competing = outcomes.get("competing_event_codes", [])
    if not isinstance(raw_competing, list):
        raise ContractError("INVALID_TYPE", "outcomes.competing_event_codes must be a list of integers")
    competing_codes: list[int] = []
    for idx, code in enumerate(raw_competing):
        if not is_strict_int(code) or code < 2:
            raise ContractError(
                "INVALID_OUTCOMES",
                f"outcomes.competing_event_codes[{idx}] must be an integer >= 2 (distinct from 0 and 1)",
            )
        if code in competing_codes:
            raise ContractError("INVALID_OUTCOMES", f"Duplicate competing event code: {code}")
        competing_codes.append(code)
    competing_codes.sort()

    comp_handling = outcomes["competing_event_handling"]
    if not isinstance(comp_handling, str) or comp_handling not in ALLOWED_COMPETING_HANDLING:
        raise ContractError(
            "INVALID_OUTCOMES",
            f"outcomes.competing_event_handling must be one of {sorted(ALLOWED_COMPETING_HANDLING)}",
        )
    if competing_codes and comp_handling == "none":
        raise ContractError(
            "UNHANDLED_COMPETING_EVENT",
            f"Protocol declares competing_event_codes={competing_codes} while "
            "competing_event_handling='none'. Hint: Set competing_event_handling="
            "'explicit_cif_and_subdistribution' so competing events are modeled via Aalen-Johansen CIF "
            "and subdistribution/cause-specific hazards rather than naive Kaplan-Meier censoring.",
        )
    if not competing_codes and comp_handling == "explicit_cif_and_subdistribution":
        raise ContractError(
            "INVALID_OUTCOMES",
            "competing_event_handling='explicit_cif_and_subdistribution' requires non-empty "
            "competing_event_codes (e.g. [2]).",
        )

    # 7. estimand & diagnostic_gates
    estimand = obj["estimand"]
    if not isinstance(estimand, dict):
        raise ContractError("INVALID_TYPE", "estimand must be an object")
    _check_exact_keys(
        estimand,
        required=("target_estimand", "effect_measures", "diagnostic_gates"),
        context="estimand",
    )
    target_estimand = estimand["target_estimand"]
    if not isinstance(target_estimand, str) or target_estimand not in ALLOWED_ESTIMANDS:
        raise ContractError(
            "INVALID_ESTIMAND",
            f"estimand.target_estimand must be one of {sorted(ALLOWED_ESTIMANDS)}",
        )
    if len(norm_strategies) >= 3 and target_estimand == "ATO":
        raise ContractError(
            "INVALID_ESTIMAND",
            "Two-arm target_estimand='ATO' is ambiguous for >=3 treatment strategies. "
            "Hint: Specify 'GENERALIZED_ATO' (common equipoise population) or 'PAIRWISE_ATO'.",
        )

    measures = estimand["effect_measures"]
    if not isinstance(measures, list) or len(measures) == 0:
        raise ContractError("INVALID_ESTIMAND", "estimand.effect_measures must be a non-empty list")
    norm_measures: list[str] = []
    for idx, m in enumerate(measures):
        if not isinstance(m, str) or m not in ALLOWED_EFFECT_MEASURES:
            raise ContractError(
                "INVALID_ESTIMAND",
                f"estimand.effect_measures[{idx}] must be one of {sorted(ALLOWED_EFFECT_MEASURES)}",
            )
        if m in norm_measures:
            raise ContractError("INVALID_ESTIMAND", f"Duplicate effect measure '{m}'")
        norm_measures.append(m)

    if any(m in ("RD", "RR_CIF") for m in norm_measures) and len(horizons_sorted) == 0:
        raise ContractError(
            "MISSING_HORIZON",
            "Effect measures 'RD' and 'RR_CIF' require at least one explicit evaluation time in "
            "followup.horizon_days. Hint: Populate followup.horizon_days (e.g. [90, 180]).",
        )
    if "sHR" in norm_measures and not competing_codes:
        raise ContractError(
            "INVALID_ESTIMAND",
            "Effect measure 'sHR' requires declared outcomes.competing_event_codes.",
        )

    gates = estimand["diagnostic_gates"]
    if not isinstance(gates, dict):
        raise ContractError("INVALID_TYPE", "estimand.diagnostic_gates must be an object")
    _check_exact_keys(
        gates,
        required=("max_abs_smd", "min_ess_ratio", "positivity_epsilon"),
        context="estimand.diagnostic_gates",
        missing_code="MISSING_DIAGNOSTIC_GATE",
    )
    max_abs_smd = gates["max_abs_smd"]
    min_ess_ratio = gates["min_ess_ratio"]
    positivity_eps = gates["positivity_epsilon"]
    if not is_strict_number(max_abs_smd) or float(max_abs_smd) <= 0.0:
        raise ContractError("INVALID_DIAGNOSTIC_GATE", "diagnostic_gates.max_abs_smd must be > 0")
    if not is_strict_number(min_ess_ratio) or not (0.0 < float(min_ess_ratio) <= 1.0):
        raise ContractError("INVALID_DIAGNOSTIC_GATE", "diagnostic_gates.min_ess_ratio must be in (0, 1]")
    if not is_strict_number(positivity_eps) or not (0.0 < float(positivity_eps) < 0.5):
        raise ContractError("INVALID_DIAGNOSTIC_GATE", "diagnostic_gates.positivity_epsilon must be in (0, 0.5)")

    normalized: dict = {
        "protocol_version": PROTOCOL_VERSION,
        "study_title": study_title,
        "simulation_or_data_label": sim_label,
        "eligibility": {
            "inclusion_criteria": norm_inc,
            "exclusion_criteria": norm_exc,
        },
        "time_zero": normalized_tz,
        "treatment_strategies": norm_strategies,
        "assignment_mode": assignment_mode,
        "followup": {
            "horizon_days": horizons_sorted,
            "administrative_end_day": admin_end_f,
        },
        "outcomes": {
            "primary_event_code": 1,
            "competing_event_codes": competing_codes,
            "censor_code": 0,
            "competing_event_handling": comp_handling,
        },
        "estimand": {
            "target_estimand": target_estimand,
            "effect_measures": norm_measures,
            "diagnostic_gates": {
                "max_abs_smd": float(max_abs_smd),
                "min_ess_ratio": float(min_ess_ratio),
                "positivity_epsilon": float(positivity_eps),
            },
        },
    }

    if "covariates" in obj:
        covs = obj["covariates"]
        if not isinstance(covs, list) or len(covs) == 0:
            raise ContractError("INVALID_COVARIATES", "covariates must be a non-empty list when provided")
        seen_covs: set[str] = set()
        norm_covs: list[dict] = []
        for idx, cov in enumerate(covs):
            ctx = f"covariates[{idx}]"
            if not isinstance(cov, dict):
                raise ContractError("INVALID_COVARIATES", f"{ctx} must be an object")
            _check_exact_keys(
                cov,
                required=("name", "type", "assessed_day_offset"),
                optional=("description",),
                context=ctx,
            )
            cname = _require_nonempty_str(cov["name"], f"{ctx}.name", "INVALID_COVARIATES")
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", cname):
                raise ContractError("INVALID_COVARIATES", f"{ctx}.name '{cname}' must be a valid identifier")
            if cname in seen_covs:
                raise ContractError("INVALID_COVARIATES", f"Duplicate covariate name '{cname}'")
            seen_covs.add(cname)
            ctype = cov["type"]
            if ctype not in ("continuous", "binary"):
                raise ContractError("INVALID_COVARIATES", f"{ctx}.type must be 'continuous' or 'binary'")
            coffset = cov["assessed_day_offset"]
            if not is_strict_int(coffset):
                raise ContractError("INVALID_TYPE", f"{ctx}.assessed_day_offset must be an integer")
            if coffset > landmark_day:
                raise ContractError(
                    "POST_LANDMARK_COVARIATE_LEAK",
                    f"Covariate '{cname}' has assessed_day_offset={coffset} > landmark_day={landmark_day}. "
                    "Hint: Propensity covariates must be measured at or before time_zero/landmark_day.",
                )
            norm_c = {"name": cname, "type": ctype, "assessed_day_offset": coffset}
            if "description" in cov:
                if not isinstance(cov["description"], str):
                    raise ContractError("INVALID_TYPE", f"{ctx}.description must be a string")
                norm_c["description"] = cov["description"]
            norm_covs.append(norm_c)
        normalized["covariates"] = norm_covs

    if "project_mode" in obj:
        pmode = obj["project_mode"]
        if not isinstance(pmode, str) or pmode not in ALLOWED_SOURCE_MODES:
            raise ContractError(
                "UNSUPPORTED_SOURCE_MODE",
                f"project_mode must be one of {sorted(ALLOWED_SOURCE_MODES)}",
            )
        normalized["project_mode"] = pmode

    if "target_rct" in obj:
        trct = obj["target_rct"]
        if not isinstance(trct, dict):
            raise ContractError("INVALID_TYPE", "target_rct must be an object")
        _check_exact_keys(
            trct,
            required=("nct_id", "brief_title"),
            optional=(
                "record_url",
                "source",
                "receipt_sha256",
                "primary_outcome_measure",
                "primary_outcome_timeframe",
                "arm_alignment_status",
                "arm_alignment_warning",
            ),
            context="target_rct",
        )
        nct_id = _require_nonempty_str(trct["nct_id"], "target_rct.nct_id")
        if not re.fullmatch(r"NCT[0-9]{8}", nct_id):
            raise ContractError("INVALID_NCT_ID", f"target_rct.nct_id '{nct_id}' must match NCT########")
        btitle = _require_nonempty_str(trct["brief_title"], "target_rct.brief_title")
        norm_rct: dict = {"nct_id": nct_id, "brief_title": btitle}
        for opt_k in (
            "record_url",
            "source",
            "receipt_sha256",
            "primary_outcome_measure",
            "primary_outcome_timeframe",
            "arm_alignment_status",
            "arm_alignment_warning",
        ):
            if opt_k in trct:
                if not isinstance(trct[opt_k], str):
                    raise ContractError("INVALID_TYPE", f"target_rct.{opt_k} must be a string")
                norm_rct[opt_k] = trct[opt_k]
        normalized["target_rct"] = norm_rct

    if "methodological_disclaimer" in obj and not isinstance(obj["methodological_disclaimer"], str):
        raise ContractError("INVALID_TYPE", "methodological_disclaimer must be a string")
    normalized["methodological_disclaimer"] = METHODOLOGICAL_DISCLAIMER
    return normalized


def load_and_validate_protocol(path: Path) -> tuple[bytes, dict]:
    raw, obj = read_strict_json(path, io_error_code="PROTOCOL_IO_ERROR")
    normalized = validate_protocol_dict(obj)
    return raw, normalized


def _render_protocol_report_md(normalized: dict) -> str:
    tz = normalized["time_zero"]
    fu = normalized["followup"]
    out = normalized["outcomes"]
    est = normalized["estimand"]
    gates = est["diagnostic_gates"]
    arms_md = "\n".join(
        f"- `{s['id']}` ({s['role']}): {s['label']}" for s in normalized["treatment_strategies"]
    )
    inc_count = len(normalized["eligibility"]["inclusion_criteria"])
    exc_count = len(normalized["eligibility"]["exclusion_criteria"])
    return (
        f"# TTE Protocol Observable-Rule Check Report\n\n"
        f"- **Study Title**: {normalized['study_title']}\n"
        f"- **Data / Simulation Label**: {normalized['simulation_or_data_label']}\n"
        f"- **Protocol Version**: `{normalized['protocol_version']}`\n"
        f"- **Status**: `PASSED_OBSERVABLE_RULE_CHECKS`\n\n"
        f"## Seven-Element Summary\n\n"
        f"1. **Eligibility**: {inc_count} inclusion criterion/criteria, {exc_count} exclusion criterion/criteria "
        f"(all `assessed_day_offset <= {tz['landmark_day']}`).\n"
        f"2. **Time Zero**: anchor `{tz['anchor_event']}`, `day_offset = 0`, `landmark_day = {tz['landmark_day']}`.\n"
        f"3. **Treatment Strategies ({len(normalized['treatment_strategies'])} arms)**:\n{arms_md}\n"
        f"4. **Assignment Mode**: `{normalized['assignment_mode']}`.\n"
        f"5. **Follow-up**: horizons `{fu['horizon_days']}` days, administrative end `{fu['administrative_end_day']}` days.\n"
        f"6. **Outcomes**: primary=`{out['primary_event_code']}`, competing=`{out['competing_event_codes']}`, "
        f"censor=`{out['censor_code']}`, handling=`{out['competing_event_handling']}`.\n"
        f"7. **Estimand & Explicit SAP Diagnostic Gates**: target=`{est['target_estimand']}`, "
        f"measures=`{est['effect_measures']}`, `max_abs_smd={gates['max_abs_smd']}`, "
        f"`min_ess_ratio={gates['min_ess_ratio']}`, `positivity_epsilon={gates['positivity_epsilon']}`.\n\n"
        f"## Methodological Boundary Disclaimer\n\n"
        f"> {METHODOLOGICAL_DISCLAIMER}\n"
    )


def validate_protocol_command(protocol_path: Path, out_dir: Path) -> dict:
    raw, normalized = load_and_validate_protocol(protocol_path)
    commit, code_dirty = code_provenance()
    deps = dependency_versions()
    report_md = _render_protocol_report_md(normalized).encode("utf-8")
    norm_bytes = json_bytes(normalized)
    output_paths = ["normalized_protocol.json", "protocol_check_report.md"]

    def _populate(stage: Path) -> None:
        (stage / "normalized_protocol.json").write_bytes(norm_bytes)
        (stage / "protocol_check_report.md").write_bytes(report_md)
        manifest = {
            "status": "protocol_validated",
            "activity": "tte_protocol_observable_rule_validation",
            "protocol_version": PROTOCOL_VERSION,
            "simulation_or_data_label": normalized["simulation_or_data_label"],
            "methodological_disclaimer": METHODOLOGICAL_DISCLAIMER,
            "input_sha256": sha256_bytes(raw),
            "code_commit": commit,
            "code_dirty": code_dirty,
            "dependency_versions": deps,
            "output_paths": output_paths,
            "output_sha256": {p: sha256_file(stage / p) for p in output_paths},
            "run_at_utc": utc_now_iso(),
        }
        (stage / "run_manifest.json").write_bytes(json_bytes(manifest))

    atomic_publish(out_dir, _populate)
    return {
        "status": "protocol_validated",
        "simulation_or_data_label": normalized["simulation_or_data_label"],
        "methodological_disclaimer": METHODOLOGICAL_DISCLAIMER,
        "output_path": str(out_dir),
    }
