"""Deterministic Target Trial Emulation (TTE) protocol checking, synthetic DGP, weighting, E-values, and competing-risks workbench."""

from ._common import (
    ContractError,
    GateBlockedError,
    METHODOLOGICAL_DISCLAIMER,
    PACKAGE_VERSION,
    PROHIBITED_CLAIMS,
    PROTOCOL_VERSION,
    SOURCE_MODE_SYNTHETIC_ONLY,
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
from .pipeline import run_synthetic_pipeline
from .protocol import (
    load_and_validate_protocol,
    validate_protocol_command,
    validate_protocol_dict,
)
from .synthetic import generate_synthetic_cohort
from .weighting import compute_weights, kish_ess

__all__ = [
    "ContractError",
    "GateBlockedError",
    "METHODOLOGICAL_DISCLAIMER",
    "PACKAGE_VERSION",
    "PROHIBITED_CLAIMS",
    "PROTOCOL_VERSION",
    "SOURCE_MODE_SYNTHETIC_ONLY",
    "compute_evalue",
    "compute_reverse_km_left_limit",
    "compute_weights",
    "diagnose_cohort",
    "estimate_competing_risks",
    "evaluate_step_curve",
    "fit_fine_gray_shr",
    "fit_weighted_cause_specific_cox",
    "generate_synthetic_cohort",
    "kish_ess",
    "load_and_validate_protocol",
    "run_synthetic_pipeline",
    "validate_protocol_command",
    "validate_protocol_dict",
    "weighted_aalen_johansen_cif",
]
