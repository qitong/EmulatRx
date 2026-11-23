"""Apply only observable synthetic baseline eligibility to generated CSV rows."""

import csv
import json
from pathlib import Path

from emulatrx_tte._common import ContractError, is_strict_number


ATTRITION_FIELDS = ("step_order", "step_id", "description", "remaining_n", "excluded_n",
                    "excluded_primary_events", "excluded_competing_events", "excluded_censored", "arm_counts_json")


def validate_screening_criteria(protocol):
    """Reject unavailable or post-event variables before any work starts."""
    available = {cov["name"] for cov in protocol.get("covariates", [])}
    for kind in ("inclusion_criteria", "exclusion_criteria"):
        for criterion in protocol["eligibility"][kind]:
            variable = criterion["variable"]
            if variable not in available:
                raise ContractError("UNAVAILABLE_ELIGIBILITY_VARIABLE",
                                    f"{variable} is not a generated baseline covariate")
            value = criterion["value"]
            values = value if criterion["operator"] in ("in", "not_in") else [value]
            if any(not is_strict_number(item) for item in values):
                raise ContractError("INVALID_ELIGIBILITY", "preview eligibility values must be numeric")


def _matches(row, criterion):
    value = float(row[criterion["variable"]])
    target = criterion["value"]
    operator = criterion["operator"]
    if operator == "in":
        return value in [float(item) for item in target]
    if operator == "not_in":
        return value not in [float(item) for item in target]
    target = float(target)
    return {"==": lambda: value == target, "!=": lambda: value != target,
            "<": lambda: value < target, "<=": lambda: value <= target,
            ">": lambda: value > target, ">=": lambda: value >= target}[operator]()


def screen_cohort(protocol, source, destination):
    """Publish a filtered CSV and a row-by-row screening ledger for the TTE commands."""
    validate_screening_criteria(protocol)
    destination = Path(destination)
    destination.mkdir(mode=0o700)
    with Path(source).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        missing = sorted({c["variable"] for kind in ("inclusion_criteria", "exclusion_criteria")
                          for c in protocol["eligibility"][kind]} - set(columns))
        if missing:
            raise ContractError("UNAVAILABLE_ELIGIBILITY_VARIABLE", ", ".join(missing))
        rows = list(reader)
    arms = [strategy["id"] for strategy in protocol["treatment_strategies"]]
    event_codes = protocol["outcomes"]

    def ledger(order, step_id, description, remaining, excluded):
        return {"step_order": order, "step_id": step_id, "description": description,
                "remaining_n": len(remaining), "excluded_n": len(excluded),
                "excluded_primary_events": sum(int(r["event_code"]) == event_codes["primary_event_code"] for r in excluded),
                "excluded_competing_events": sum(int(r["event_code"]) in event_codes["competing_event_codes"] for r in excluded),
                "excluded_censored": sum(int(r["event_code"]) == event_codes["censor_code"] for r in excluded),
                "arm_counts_json": json.dumps({arm: sum(r["treatment_arm"] == arm for r in remaining) for arm in arms}, sort_keys=True)}

    attrition = [ledger(0, "generated_synthetic_cohort", "All generated synthetic rows before eligibility", rows, [])]
    order = 1
    for kind in ("inclusion_criteria", "exclusion_criteria"):
        for criterion in protocol["eligibility"][kind]:
            eligible = []
            excluded = []
            for row in rows:
                matches = _matches(row, criterion)
                (eligible if (matches if kind == "inclusion_criteria" else not matches) else excluded).append(row)
            rows = eligible
            attrition.append(ledger(order, criterion["id"],
                                    f"{'Include' if kind == 'inclusion_criteria' else 'Exclude'}: {criterion['variable']} {criterion['operator']} {criterion['value']}",
                                    rows, excluded))
            order += 1
    with (destination / "eligible_cohort.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (destination / "eligibility_attrition.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ATTRITION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(attrition)
    return {"generated_rows": attrition[0]["remaining_n"], "eligible_rows": len(rows),
            "arm_counts": json.loads(attrition[-1]["arm_counts_json"])}
