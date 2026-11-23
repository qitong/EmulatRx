"""VanderWeele & Ding (2017) E-value sensitivity calculations with competing-risks measure boundaries."""

from __future__ import annotations

import math

import numpy as np

from ._common import ContractError, is_strict_number


EVALUE_BOUNDARY_NOTE = (
    "Sensitivity-only boundary: an E-value quantifies only the minimum strength of association "
    "(on the risk-ratio scale) that an unmeasured confounder would need with both treatment "
    "assignment and outcome, conditional on measured covariates, to shift the estimate to the "
    "null (1.0). It never proves absence of unmeasured confounding, measurement error, or "
    "selection bias."
)

RR_CIF_COMPETING_NOTE = (
    " For RR_CIF, the E-value applies to the cumulative incidence ratio at the specified follow-up "
    "horizon and reflects the total effect on primary event cumulative incidence, including indirect "
    "pathways mediated by competing events/mortality."
)


def _rr_evalue(rr: float) -> float:
    rr_star = rr if rr >= 1.0 else (1.0 / rr)
    if rr_star == 1.0:
        return 1.0
    return float(rr_star + math.sqrt(rr_star * (rr_star - 1.0)))


def _convert_to_rr(estimate: float, measure_type: str, is_rare_outcome: bool) -> float:
    if measure_type in ("RR", "RR_CIF"):
        return float(estimate)
    if is_rare_outcome:
        return float(estimate)
    if estimate == 1.0:
        return 1.0
    numerator = 1.0 - (0.5 ** math.sqrt(estimate))
    denominator = 1.0 - (0.5 ** math.sqrt(1.0 / estimate))
    return float(numerator / denominator)


def _validate_positive_scalar(name: str, value: object) -> float:
    if isinstance(value, (np.floating, np.integer)) and not isinstance(value, (bool, np.bool_)):
        value = float(value)
    if not is_strict_number(value) or float(value) <= 0.0:
        raise ContractError("INVALID_EVALUE_INPUT", f"{name} must be a finite positive number")
    return float(value)


def compute_evalue(
    estimate: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    measure_type: str = "RR",
    outcome_incidence: float | None = None,
    has_competing_risks: bool = False,
) -> dict:
    """Compute VanderWeele & Ding (2017) E-values while enforcing competing-risks validity boundaries."""
    if not isinstance(measure_type, str):
        raise ContractError("INVALID_EVALUE_INPUT", "measure_type must be a string")
    if not isinstance(has_competing_risks, bool):
        raise ContractError("INVALID_EVALUE_INPUT", "has_competing_risks must be a boolean")

    if measure_type == "sHR":
        raise ContractError(
            "UNSUPPORTED_EVALUE_MEASURE",
            "Fine-Gray subdistribution hazard ratio ('sHR') cannot be converted to a risk ratio via "
            "single-event E-value formulas. Hint: Estimate cumulative incidence ratio ('RR_CIF') at an "
            "explicit follow-up horizon and pass measure_type='RR_CIF'.",
        )
    if measure_type not in ("RR", "RR_CIF", "HR", "csHR"):
        raise ContractError(
            "UNSUPPORTED_EVALUE_MEASURE",
            f"Unsupported measure_type '{measure_type}'. Supported types: 'RR', 'RR_CIF', 'HR' "
            "(and rare-outcome 'csHR' when outcome_incidence <= 0.15).",
        )

    est = _validate_positive_scalar("estimate", estimate)
    lo = _validate_positive_scalar("ci_lower", ci_lower) if ci_lower is not None else None
    hi = _validate_positive_scalar("ci_upper", ci_upper) if ci_upper is not None else None
    if lo is not None and hi is not None and lo > hi:
        raise ContractError("INVALID_EVALUE_INPUT", "ci_lower cannot exceed ci_upper")

    incidence_val: float | None = None
    if outcome_incidence is not None:
        if isinstance(outcome_incidence, (np.floating, np.integer)) and not isinstance(
            outcome_incidence, (bool, np.bool_)
        ):
            outcome_incidence = float(outcome_incidence)
        if not is_strict_number(outcome_incidence) or not (0.0 <= float(outcome_incidence) <= 1.0):
            raise ContractError("INVALID_EVALUE_INPUT", "outcome_incidence must be a finite number in [0.0, 1.0]")
        incidence_val = float(outcome_incidence)

    effective_competing = has_competing_risks or (measure_type == "csHR")

    if measure_type in ("HR", "csHR"):
        if incidence_val is None:
            raise ContractError(
                "INVALID_EVALUE_INPUT",
                f"outcome_incidence in [0.0, 1.0] is required when measure_type='{measure_type}'",
            )
        if effective_competing and incidence_val > 0.15:
            raise ContractError(
                "UNSUPPORTED_EVALUE_MEASURE",
                f"Single-event common-outcome HR-to-RR conversion is invalid for '{measure_type}' with "
                f"competing risks when outcome_incidence={incidence_val:.4f} > 0.15. "
                "Hint: Evaluate E-values on explicit horizon cumulative incidence ratios ('RR_CIF').",
            )
        is_rare = incidence_val <= 0.15
        method = "rare_hr_direct" if is_rare else "vanderweele_ding_2017_common_hr_to_rr"
    else:
        is_rare = True
        method = "direct_rr_cif" if measure_type == "RR_CIF" else "direct_rr"

    converted_rr = _convert_to_rr(est, measure_type, is_rare)
    point_evalue = _rr_evalue(converted_rr)

    ci_evalue: float | None = None
    if lo is not None or hi is not None:
        if lo is not None and hi is not None:
            if lo <= 1.0 <= hi:
                ci_evalue = 1.0
            elif lo > 1.0:
                ci_evalue = _rr_evalue(_convert_to_rr(lo, measure_type, is_rare))
            else:
                ci_evalue = _rr_evalue(_convert_to_rr(hi, measure_type, is_rare))
        elif lo is not None:
            if est < 1.0:
                raise ContractError(
                    "INVALID_EVALUE_INPUT",
                    "ci_upper is required to compute ci_limit_closer_to_null when estimate < 1.0",
                )
            ci_evalue = 1.0 if lo <= 1.0 else _rr_evalue(_convert_to_rr(lo, measure_type, is_rare))
        else:
            assert hi is not None
            if est > 1.0:
                raise ContractError(
                    "INVALID_EVALUE_INPUT",
                    "ci_lower is required to compute ci_limit_closer_to_null when estimate > 1.0",
                )
            ci_evalue = 1.0 if hi >= 1.0 else _rr_evalue(_convert_to_rr(hi, measure_type, is_rare))

    note = EVALUE_BOUNDARY_NOTE + (RR_CIF_COMPETING_NOTE if measure_type == "RR_CIF" else "")
    return {
        "measure_type": measure_type,
        "estimate": est,
        "ci_lower": lo,
        "ci_upper": hi,
        "outcome_incidence": incidence_val,
        "has_competing_risks": effective_competing,
        "calculation_method": method,
        "approximated_rr": converted_rr,
        "point_estimate": point_evalue,
        "ci_limit_closer_to_null": ci_evalue,
        "interpretation_note": note,
    }
