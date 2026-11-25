"""MCP tool registration for RCT benchmarking and negative-control diagnostics."""

from __future__ import annotations

from emulatrx_tte.benchmarking import (
    benchmarking_disabled_payload,
    execute_benchmark_request,
    is_benchmarking_enabled,
)


def load_run_weights(runner, run_id: str):
    """Read server-side analysis weights for a completed run. Returns row-level data only to the caller in-process."""
    import pandas as pd

    path = runner._path(run_id)
    for relative in ("diagnostics/weighted_cohort.csv", "bundle/weighted_cohort.csv"):
        candidate = path / relative
        if not candidate.is_file():
            continue
        frame = pd.read_csv(candidate)
        for column in ("analysis_weight", "normalized_weight", "raw_weight"):
            if column in frame.columns:
                return frame, frame[column].to_numpy(dtype=float)
    return None, None


def benchmark_from_payload(payload: dict, runner) -> dict:
    if not is_benchmarking_enabled():
        return benchmarking_disabled_payload()
    run_record = None
    cohort = None
    weights = None
    run_id = str(payload.get("run_id") or "").strip()
    if run_id:
        run_record = runner.get(run_id)
        run_record = {key: value for key, value in run_record.items() if key != "sample_cases"}
        if payload.get("negative_control_outcome_col"):
            cohort, weights = load_run_weights(runner, run_id)
    return execute_benchmark_request(payload, run_record=run_record, cohort=cohort, weights=weights)


def register_benchmark_tools(mcp) -> None:
    @mcp.tool(
        description=(
            "Benchmark a completed TTE run against public study-level RCT aggregates and an optional "
            "prespecified negative control. Numerical proximity never proves causality or clinical efficacy. "
            "ipd_used is always false. Returns BENCHMARKING_FEATURE_DISABLED when the rollback switch is off."
        )
    )
    def benchmark_rct_and_negative_control(
        nct_id: str = "NCT01243424",
        run_id: str = "",
        tte_result: dict | None = None,
        alignment_audit: dict | None = None,
        negative_control_outcome_col: str = "",
        negative_control_time_col: str = "",
        curated_published_summary: dict | None = None,
        synthetic_demo: str = "",
        n: int = 2500,
        seed: int = 20261125,
        confirm_protocol_mismatch: bool = False,
        confirm_arm_mismatch: bool = False,
    ) -> dict:
        if not is_benchmarking_enabled():
            return benchmarking_disabled_payload()
        from emulatrx_preview.app import runner

        payload = {
            "nct_id": nct_id,
            "run_id": run_id,
            "tte_result": tte_result,
            "alignment_audit": alignment_audit,
            "negative_control_outcome_col": negative_control_outcome_col or None,
            "negative_control_time_col": negative_control_time_col or None,
            "curated_published_summary": curated_published_summary,
            "synthetic_demo": synthetic_demo or None,
            "n": n,
            "seed": seed,
            "confirm_protocol_mismatch": confirm_protocol_mismatch,
            "confirm_arm_mismatch": confirm_arm_mismatch,
        }
        return benchmark_from_payload(payload, runner)
