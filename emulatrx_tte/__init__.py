"""Deterministic Target Trial Emulation (TTE) protocol checking, synthetic DGP, weighting, and E-values."""

from ._common import (
    ContractError,
    METHODOLOGICAL_DISCLAIMER,
    PACKAGE_VERSION,
    PROTOCOL_VERSION,
)
from .diagnostics import diagnose_cohort
from .evalue import compute_evalue
from .protocol import (
    load_and_validate_protocol,
    validate_protocol_command,
    validate_protocol_dict,
)
from .synthetic import generate_synthetic_cohort
from .weighting import compute_weights, kish_ess

__all__ = [
    "ContractError",
    "METHODOLOGICAL_DISCLAIMER",
    "PACKAGE_VERSION",
    "PROTOCOL_VERSION",
    "compute_evalue",
    "compute_weights",
    "diagnose_cohort",
    "generate_synthetic_cohort",
    "kish_ess",
    "load_and_validate_protocol",
    "validate_protocol_command",
    "validate_protocol_dict",
]
