"""Deterministic Target Trial Emulation (TTE) protocol checking, synthetic DGP, weighting, E-values, and competing-risks workbench."""

from ._common import (
    ALLOWED_SOURCE_MODES,
    ContractError,
    GateBlockedError,
    METHODOLOGICAL_DISCLAIMER,
    OBSERVATIONAL_CAUSAL_ASSUMPTIONS,
    PACKAGE_VERSION,
    PROHIBITED_CLAIMS,
    PROTOCOL_VERSION,
    SIMULATED_COHORT_CLAIMS,
    SOURCE_MODE_SIMULATED_COHORT,
    SOURCE_MODE_SYNTHETIC_ONLY,
    SOURCE_MODE_USER_COHORT,
    USER_COHORT_CAUSAL_CLAIMS,
)
from .cohort_ingest import (
    ingest_user_cohort,
    profile_cohort_dataframe,
    read_tabular_cohort_bytes,
    recommend_column_mapping,
)
from .competing_risks import (
    compute_reverse_km_left_limit,
    estimate_competing_risks,
    evaluate_step_curve,
    fit_fine_gray_shr,
    fit_weighted_cause_specific_cox,
    weighted_aalen_johansen_cif,
)
from .diagnostics import diagnose_cohort
from .evalue import compute_evalue
from .llm_assist import (
    assert_llm_payload_privacy_safe,
    get_llm_status,
    interpret_cohort_profile,
    normalize_llm_base_url,
    parse_llm_json_content,
    parse_research_idea_pico,
    sanitize_profile_for_llm,
)
from .pipeline import (
    build_simulated_protocol_and_dgp,
    run_simulated_cohort_pipeline,
    run_synthetic_pipeline,
    run_user_cohort_pipeline,
)
from .protocol import (
    load_and_validate_protocol,
    validate_protocol_command,
    validate_protocol_dict,
)
from .rct_matcher import (
    audit_study_arm_alignment,
    fetch_rct_by_nct,
    normalize_intervention_spec,
    parse_timeframe_to_horizons,
    search_candidate_rcts,
    translate_condition_to_english,
    translate_query_to_english,
    verify_and_enrich_target_rct,
)
from .sample_cohorts import (
    CLINICAL_IDEA_PRESETS,
    SAMPLE_COHORT_SPECS,
    generate_sample_cohort_csv_bytes,
)
from .synthetic import generate_synthetic_cohort
from .weighting import compute_weights, kish_ess

__all__ = [
    "ALLOWED_SOURCE_MODES",
    "CLINICAL_IDEA_PRESETS",
    "ContractError",
    "GateBlockedError",
    "METHODOLOGICAL_DISCLAIMER",
    "OBSERVATIONAL_CAUSAL_ASSUMPTIONS",
    "PACKAGE_VERSION",
    "PROHIBITED_CLAIMS",
    "PROTOCOL_VERSION",
    "SAMPLE_COHORT_SPECS",
    "SIMULATED_COHORT_CLAIMS",
    "SOURCE_MODE_SIMULATED_COHORT",
    "SOURCE_MODE_SYNTHETIC_ONLY",
    "SOURCE_MODE_USER_COHORT",
    "USER_COHORT_CAUSAL_CLAIMS",
    "assert_llm_payload_privacy_safe",
    "audit_study_arm_alignment",
    "build_simulated_protocol_and_dgp",
    "compute_evalue",
    "compute_reverse_km_left_limit",
    "compute_weights",
    "diagnose_cohort",
    "estimate_competing_risks",
    "evaluate_step_curve",
    "fetch_rct_by_nct",
    "fit_fine_gray_shr",
    "fit_weighted_cause_specific_cox",
    "generate_sample_cohort_csv_bytes",
    "generate_synthetic_cohort",
    "get_llm_status",
    "ingest_user_cohort",
    "interpret_cohort_profile",
    "kish_ess",
    "load_and_validate_protocol",
    "normalize_intervention_spec",
    "parse_research_idea_pico",
    "parse_timeframe_to_horizons",
    "profile_cohort_dataframe",
    "read_tabular_cohort_bytes",
    "recommend_column_mapping",
    "run_simulated_cohort_pipeline",
    "run_synthetic_pipeline",
    "run_user_cohort_pipeline",
    "sanitize_profile_for_llm",
    "search_candidate_rcts",
    "translate_condition_to_english",
    "translate_query_to_english",
    "validate_protocol_command",
    "validate_protocol_dict",
    "verify_and_enrich_target_rct",
    "weighted_aalen_johansen_cif",
]
