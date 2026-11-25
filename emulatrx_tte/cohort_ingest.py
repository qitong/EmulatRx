"""Privacy-safe user cohort profiling, heuristic/LLM column mapping, categorical one-hot encoding, and attrition ingestion."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ._common import (
    ContractError,
    MAX_USER_COHORT_BYTES,
    MAX_USER_COHORT_ROWS,
    SOURCE_MODE_USER_COHORT,
    atomic_publish,
    code_provenance,
    dependency_versions,
    json_bytes,
    sha256_bytes,
    sha256_file,
    utc_now_iso,
)
from .llm_assist import ID_COL_RE, ID_VALUE_RE, PERSON_NAME_RE
from .protocol import validate_protocol_dict
from .rct_matcher import normalize_intervention_spec


def _finite_timeline_number(value: object, name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ContractError("INVALID_TIMELINE_WINDOW", f"{name} 必须是有限数字。")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ContractError("INVALID_TIMELINE_WINDOW", f"{name} 必须是有限数字。")
        try:
            number = float(text)
        except ValueError as exc:
            raise ContractError("INVALID_TIMELINE_WINDOW", f"{name} 必须是有限数字。") from exc
    else:
        raise ContractError("INVALID_TIMELINE_WINDOW", f"{name} 必须是有限数字。")
    if not math.isfinite(number):
        raise ContractError("INVALID_TIMELINE_WINDOW", f"{name} 不能是 NaN 或无穷大。")
    return number


def _json_timeline_number(number: float) -> int | float:
    rounded = round(number)
    if abs(number - rounded) <= 1e-9:
        return int(rounded)
    return number


def _parse_horizon_list(raw: object) -> list[float]:
    if isinstance(raw, str):
        items: list[object] = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        raise ContractError("INVALID_TIMELINE_WINDOW", "horizon 必须是大于 0 的有限天数。")
    parsed: list[float] = []
    for item in items:
        number = _finite_timeline_number(item, "horizon")
        if number <= 0:
            raise ContractError("INVALID_TIMELINE_WINDOW", "horizon 必须大于 0。")
        parsed.append(number)
    if not parsed:
        raise ContractError("INVALID_TIMELINE_WINDOW", "horizon 必须大于 0。")
    return sorted(set(parsed))


def apply_timeline_window(*sources: dict) -> tuple[int, list[int | float] | None]:
    """Validate landmark_time and horizons. Later sources override earlier ones.

    Writes ``landmark_day`` / ``landmark_time`` and, when horizons were supplied,
    ``horizon_days`` onto the first mapping or sim-config dict. ``landmark_time >= min(horizons)``,
    negative landmark, non-positive horizon, and non-finite values raise ``INVALID_TIMELINE_WINDOW``.
    """
    landmark_value = 0.0
    horizons: list[float] | None = None
    for src in sources:
        if not isinstance(src, dict):
            continue
        for key in ("landmark_day", "landmark_days", "landmark_time"):
            if key in src and src[key] not in (None, ""):
                landmark_value = _finite_timeline_number(src[key], "landmark_time")
        for key in ("horizon", "horizons", "horizon_days"):
            if key in src and src[key] not in (None, ""):
                horizons = _parse_horizon_list(src[key])
    if landmark_value < 0:
        raise ContractError("INVALID_TIMELINE_WINDOW", "landmark_time 必须大于等于 0。")
    if abs(landmark_value - round(landmark_value)) > 1e-6:
        raise ContractError("INVALID_TIMELINE_WINDOW", "landmark_time 必须是整数天。")
    landmark_day = int(round(landmark_value))
    if horizons is not None and landmark_day >= min(horizons):
        raise ContractError(
            "INVALID_TIMELINE_WINDOW",
            "landmark_time 必须小于最短 horizon，不能在图上静默夹紧。",
        )
    stored_horizons = [_json_timeline_number(item) for item in horizons] if horizons is not None else None
    primary = next((src for src in sources if isinstance(src, dict)), None)
    if primary is not None:
        primary["landmark_day"] = landmark_day
        primary["landmark_time"] = landmark_day
        if stored_horizons is not None:
            primary["horizon_days"] = stored_horizons
            primary["horizons"] = stored_horizons
    return landmark_day, stored_horizons


TIME_UNIT_FACTORS = {
    "days": 1.0,
    "months": 30.4375,
    "years": 365.25,
}

TREATMENT_NAME_HINTS = ("drug", "trt", "treat", "arm", "regimen", "therapy", "exposure", "group", "initiated", "med")
TIME_NAME_HINTS = ("followup", "follow_up", "duration", "os_days", "pfs_days", "time", "months", "days", "years", "surv")
OUTCOME_NAME_HINTS = ("outcome", "event", "status", "vital", "death", "censor", "endpoint", "relapse", "progression")

CENSOR_LABEL_HINTS = ("0", "censor", "censored", "alive", "alive_censored", "no_event", "none", "survive", "删失", "存活", "无事件")
COMPETING_LABEL_HINTS = ("2", "competing", "noncv", "non_cv", "unrelated", "other_death", "competing_death", "竞争", "非肿瘤", "非心血管")
PRIMARY_LABEL_HINTS = ("1", "event", "primary", "hf", "cv", "progression", "cancer", "stroke", "relapse", "esrd", "主要", "事件", "进展")


def _sanitize_identifier(raw: str, prefix: str = "cov") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"{prefix}_{cleaned}" if cleaned else prefix
    return cleaned[:64]


def read_tabular_cohort_bytes(raw_bytes: bytes, filename: str = "cohort.csv") -> tuple[pd.DataFrame, str, str]:
    """Parse UTF-8 / UTF-8-BOM / GB18030 CSV or TSV bytes (or .xlsx if supported) with strict size/row bounds."""
    if len(raw_bytes) == 0:
        raise ContractError("EMPTY_COHORT_FILE", "上传的数据文件为空。")
    if len(raw_bytes) > MAX_USER_COHORT_BYTES:
        raise ContractError("COHORT_FILE_TOO_LARGE", f"上传文件超过 {MAX_USER_COHORT_BYTES // (1024 * 1024)} MB 上限。")

    lower_name = (filename or "").lower()
    if lower_name.endswith((".xlsx", ".xls")):
        try:
            df_xl = pd.read_excel(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise ContractError(
                "UNSUPPORTED_EXCEL_FORMAT",
                f"当前环境无法直接解析 Excel 文件（{exc}）；请另存为 UTF-8 CSV (.csv) 或制表符分隔 (.tsv) 后上传。",
            ) from exc
        if len(df_xl) > MAX_USER_COHORT_ROWS:
            raise ContractError("TOO_MANY_COHORT_ROWS", f"队列行数超过上限 {MAX_USER_COHORT_ROWS} 行。")
        if len(df_xl) < 10 or len(df_xl.columns) < 3:
            raise ContractError("COHORT_TOO_SMALL", "队列表至少需包含 10 行记录与 3 列变量（暴露、随访时长、结局）。")
        return df_xl, "xlsx", "binary"

    text: str | None = None
    detected_enc = "utf-8"
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = raw_bytes.decode(enc)
            detected_enc = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ContractError("INVALID_FILE_ENCODING", "无法识别文件字符编码；请保存为 UTF-8 或 UTF-8-BOM 格式 CSV/TSV。")

    if "\x00" in text[:4096]:
        raise ContractError("BINARY_FILE_REJECTED", "检测到二进制文件内容；请上传标准文本 CSV 或 TSV 表格。")

    first_lines = "\n".join(text.splitlines()[:10])
    if lower_name.endswith(".tsv") or first_lines.count("\t") > first_lines.count(","):
        delimiter = "\t"
    else:
        try:
            dialect = csv.Sniffer().sniff(first_lines, delimiters=[",", "\t", ";"])
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            nrows=MAX_USER_COHORT_ROWS + 1,
            keep_default_na=True,
            na_values=["", "NA", "N/A", "null", "NULL", "NaN", "nan", "."],
        )
    except Exception as exc:
        raise ContractError("CSV_PARSE_ERROR", f"解析表格失败：{exc}") from exc

    if len(df) > MAX_USER_COHORT_ROWS:
        raise ContractError("TOO_MANY_COHORT_ROWS", f"队列行数超过上限 {MAX_USER_COHORT_ROWS} 行。")
    if len(df) < 10 or len(df.columns) < 3:
        raise ContractError("COHORT_TOO_SMALL", "队列表至少需包含 10 行记录与 3 列字段。")

    df.columns = [str(c).strip() for c in df.columns]
    if len(set(df.columns)) != len(df.columns) or any(not c for c in df.columns):
        raise ContractError("INVALID_COLUMN_HEADERS", "表头存在空列名或重复列名，请检查首行字段名。")

    return df, delimiter, detected_enc


def _classify_outcome_values(distinct_vals: list[str]) -> dict[str, list[str]]:
    censor_vals: list[str] = []
    competing_vals: list[str] = []
    primary_vals: list[str] = []

    for v in distinct_vals:
        vl = v.lower().strip()
        if vl in ("0", "0.0") or any(h in vl for h in CENSOR_LABEL_HINTS[1:]):
            censor_vals.append(v)
        elif vl in ("2", "2.0") or any(h in vl for h in COMPETING_LABEL_HINTS[1:]):
            competing_vals.append(v)
        elif vl in ("1", "1.0") or any(h in vl for h in PRIMARY_LABEL_HINTS[1:]):
            primary_vals.append(v)

    assigned = set(censor_vals) | set(competing_vals) | set(primary_vals)
    remaining = [v for v in distinct_vals if v not in assigned]
    for v in remaining:
        if not censor_vals:
            censor_vals.append(v)
        elif not primary_vals:
            primary_vals.append(v)
        elif not competing_vals:
            competing_vals.append(v)
        else:
            censor_vals.append(v)

    return {
        "primary_event_values": primary_vals,
        "competing_event_values": competing_vals,
        "censor_values": censor_vals,
    }


def profile_cohort_dataframe(df: pd.DataFrame, *, filename: str = "cohort.csv", input_sha256: str = "") -> dict:
    """Build a privacy-safe aggregate CohortProfile with zero individual patient row records."""
    n_rows = int(len(df))
    n_cols = int(len(df.columns))

    columns_info: list[dict] = []
    trt_candidates: list[tuple[int, dict]] = []
    time_candidates: list[tuple[int, dict]] = []
    outcome_candidates: list[tuple[int, dict]] = []
    id_column_guess: str | None = None

    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_rate = round(missing_count / n_rows, 6) if n_rows > 0 else 0.0
        non_null = s.dropna()
        n_unique = int(non_null.nunique(dropna=True))
        col_lower = col.lower()

        # Check numeric coercion
        num_series = pd.to_numeric(non_null, errors="coerce")
        is_numeric = len(non_null) > 0 and bool(num_series.notna().all()) and s.dtype != object
        if not is_numeric and len(non_null) > 0:
            # Handle numeric columns read as strings due to formatting
            if float(num_series.notna().mean()) >= 0.995:
                is_numeric = True

        # Detect identifier columns (unique/near-unique or ID header name or ID value pattern)
        sample_strs = [str(v).strip() for v in non_null.iloc[: min(30, len(non_null))]]
        id_val_ratio = (
            sum(1 for sv in sample_strs if ID_VALUE_RE.match(sv)) / len(sample_strs) if sample_strs else 0.0
        )
        is_id = bool(
            (ID_COL_RE.search(col) and n_unique >= max(10, int(0.5 * len(non_null))))
            or (not is_numeric and n_unique == len(non_null) and len(non_null) >= 20)
            or id_val_ratio >= 0.6
        )
        if is_id and id_column_guess is None:
            id_column_guess = col

        col_entry: dict = {
            "name": col,
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "n_unique": n_unique,
        }

        if is_id:
            col_entry["dtype_kind"] = "identifier"
            col_entry["inferred_role"] = "identifier"
            col_entry["top_categories"] = []  # Privacy lock: NEVER store ID values!
            columns_info.append(col_entry)
            continue

        if n_unique <= 1:
            col_entry["dtype_kind"] = "numeric" if is_numeric else "categorical"
            col_entry["inferred_role"] = "constant_or_unusable"
            col_entry["top_categories"] = []
            columns_info.append(col_entry)
            continue

        if is_numeric:
            arr = num_series.dropna().to_numpy(dtype=np.float64)
            q25, q50, q75 = np.quantile(arr, [0.25, 0.50, 0.75])
            num_sum = {
                "mean": round(float(np.mean(arr)), 4),
                "sd": round(float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0, 4),
                "min": round(float(np.min(arr)), 4),
                "p25": round(float(q25), 4),
                "median": round(float(q50), 4),
                "p75": round(float(q75), 4),
                "max": round(float(np.max(arr)), 4),
            }
            col_entry["dtype_kind"] = "numeric"
            col_entry["numeric_summary"] = num_sum

            uniq_vals = sorted(np.unique(arr).tolist())
            is_binary_01 = uniq_vals == [0.0, 1.0]
            col_entry["inferred_role"] = "binary_covariate" if is_binary_01 else "continuous_covariate"

            # Check if candidate follow-up time column
            if float(np.min(arr)) >= 0.0 and float(np.max(arr)) > 0.0 and n_unique >= 5:
                t_score = 10
                if any(h in col_lower for h in TIME_NAME_HINTS):
                    t_score += 50
                if "age" in col_lower or "egfr" in col_lower or "bmi" in col_lower or "hba1c" in col_lower or "pdl1" in col_lower:
                    t_score -= 40
                inferred_unit = "days"
                if "month" in col_lower or "_mo" in col_lower:
                    inferred_unit = "months"
                elif "year" in col_lower or "_yr" in col_lower:
                    inferred_unit = "years"
                elif float(np.max(arr)) <= 36.0 and "day" not in col_lower:
                    inferred_unit = "months"
                if t_score > 0:
                    time_candidates.append(
                        (
                            t_score,
                            {
                                "column": col,
                                "inferred_unit": inferred_unit,
                                "median": num_sum["median"],
                                "min": num_sum["min"],
                                "max": num_sum["max"],
                                "p25": num_sum["p25"],
                                "p75": num_sum["p75"],
                                "missing_rate": missing_rate,
                            },
                        )
                    )

            # Check if numeric outcome code column (e.g. 0/1/2 or 0/1)
            if 2 <= n_unique <= 4 and all(float(v).is_integer() and 0 <= int(v) <= 5 for v in uniq_vals):
                str_counts = non_null.astype(int).astype(str).value_counts()
                cats = [
                    {"value": str(k), "count": int(v), "rate": round(int(v) / len(non_null), 4)}
                    for k, v in str_counts.items()
                ]
                col_entry["top_categories"] = cats
                o_score = 15 + (45 if any(h in col_lower for h in OUTCOME_NAME_HINTS) else 0)
                if 2 in [int(v) for v in uniq_vals]:
                    o_score += 20
                outcome_candidates.append(
                    (
                        o_score,
                        {
                            "column": col,
                            "categories": cats,
                            "suggested_mapping": _classify_outcome_values([c["value"] for c in cats]),
                        },
                    )
                )
                if 2 == n_unique and any(h in col_lower for h in TREATMENT_NAME_HINTS):
                    trt_candidates.append((55, {"column": col, "categories": cats}))
        else:
            col_entry["dtype_kind"] = "categorical"
            col_entry["inferred_role"] = "categorical_covariate"
            if n_unique <= 20:
                vc = non_null.astype(str).str.strip().value_counts()
                cats = [
                    {"value": str(k), "count": int(v), "rate": round(int(v) / len(non_null), 4)}
                    for k, v in vc.items()
                    if not ID_VALUE_RE.match(str(k))
                ][:20]
                col_entry["top_categories"] = cats

                if 2 <= len(cats) <= 4:
                    arm_score = 20 + (55 if any(h in col_lower for h in TREATMENT_NAME_HINTS) else 0)
                    if any(w in col_lower for w in ("drug", "regimen", "therapy", "initiated", "trt", "treat", "exposure")):
                        arm_score += 20
                    if any(w in col_lower for w in ("site", "center", "centre", "hosp", "origin", "source", "age", "stage", "nyha", "ecog", "sex", "smoke", "smoking")):
                        arm_score -= 40
                    if any(h in col_lower for h in OUTCOME_NAME_HINTS):
                        arm_score -= 35
                    if all(PERSON_NAME_RE.search(c["value"]) for c in cats[:2]):
                        arm_score -= 45
                    elif sum(
                        1
                        for c in cats[:2]
                        if any(not d.startswith("token:") for d in normalize_intervention_spec(c["value"])["drug_ids"])
                    ) >= 1:
                        arm_score += 30
                    if len(cats) == 2:
                        arm_score += 12
                    if arm_score > 0:
                        trt_candidates.append((arm_score, {"column": col, "categories": cats}))

                if 2 <= len(cats) <= 6:
                    out_score = 15 + (55 if any(h in col_lower for h in OUTCOME_NAME_HINTS) else 0)
                    cat_vals_lower = " ".join(c["value"].lower() for c in cats)
                    if any(w in cat_vals_lower for w in ("censor", "alive", "death", "event", "progression")):
                        out_score += 30
                    if any(h in col_lower for h in TREATMENT_NAME_HINTS):
                        out_score -= 35
                    if out_score > 20:
                        outcome_candidates.append(
                            (
                                out_score,
                                {
                                    "column": col,
                                    "categories": cats,
                                    "suggested_mapping": _classify_outcome_values([c["value"] for c in cats]),
                                },
                            )
                        )
            else:
                col_entry["top_categories"] = []

        columns_info.append(col_entry)

    trt_candidates.sort(key=lambda x: x[0], reverse=True)
    time_candidates.sort(key=lambda x: x[0], reverse=True)
    outcome_candidates.sort(key=lambda x: x[0], reverse=True)

    return {
        "filename": filename,
        "input_sha256": input_sha256,
        "row_count": n_rows,
        "column_count": n_cols,
        "id_column_guess": id_column_guess,
        "candidate_treatment_columns": [x[1] for x in trt_candidates[:4]],
        "candidate_time_columns": [x[1] for x in time_candidates[:4]],
        "candidate_outcome_columns": [x[1] for x in outcome_candidates[:4]],
        "columns": columns_info,
    }


_PII_COLUMN_RE = re.compile(
    r"姓名|患者名|身份证|手机|电话|住址|phone|mobile|patient_name|id_card|idcard",
    re.IGNORECASE,
)
_OPAQUE_BINARY = {"0", "1", "0.0", "1.0"}


def _category_values(column_info: dict) -> set[str]:
    values: set[str] = set()
    for item in column_info.get("top_categories") or column_info.get("categories") or []:
        if isinstance(item, dict) and "value" in item:
            values.add(str(item["value"]).strip())
        else:
            values.add(str(item).strip())
    return values


def detect_missing_clinical_semantics(df: pd.DataFrame, profile: dict) -> dict:
    """Questions an agent must ask before drafting a protocol, plus quarantined PII column names.

    The returned object never contains row-level cell values.
    """
    quarantined: list[str] = []
    for column in profile.get("columns") or []:
        name = str(column.get("name", ""))
        if column.get("inferred_role") == "identifier" or _PII_COLUMN_RE.search(name):
            quarantined.append(name)
    questions: list[dict] = []
    treatment_cols = profile.get("candidate_treatment_columns") or []
    if treatment_cols:
        lead = treatment_cols[0]
        values = _category_values(lead)
        header = str(lead.get("column", ""))
        if values and values <= _OPAQUE_BINARY:
            questions.append(
                {
                    "field": "treatment",
                    "column": header,
                    "ask_zh": "治疗组列只有 0/1，没有药品通用名。请向临床专家确认 1 和 0 分别对应哪一种药物或治疗策略。",
                }
            )
    outcome_cols = profile.get("candidate_outcome_columns") or []
    if outcome_cols:
        lead_out = outcome_cols[0]
        out_values = _category_values(lead_out)
        out_header = str(lead_out.get("column", "")).lower()
        generic_header = out_header in {"event", "outcome", "status", "y", "endpoint"}
        if generic_header and out_values and out_values <= {"0", "1", "2", "0.0", "1.0", "2.0"}:
            questions.append(
                {
                    "field": "outcome",
                    "column": lead_out.get("column"),
                    "ask_zh": "结局列是匿名的 event=0/1（或含 2）。请确认 1 是哪一种疾病事件、2 是否为竞争死亡，以及计时起点。",
                }
            )
    return {
        "missing_clinical_semantics": questions,
        "pii_columns_quarantined": quarantined,
    }


def recommend_column_mapping(profile: dict, target_rct: dict | None = None) -> dict:
    """Draft default ColumnMappingSpec + protocol settings aligning a CohortProfile to an optional Target RCT."""
    trt_cands = profile.get("candidate_treatment_columns", [])
    time_cands = profile.get("candidate_time_columns", [])
    out_cands = profile.get("candidate_outcome_columns", [])

    trt_col = trt_cands[0]["column"] if trt_cands else ""
    trt_cats = [c["value"] for c in (trt_cands[0]["categories"] if trt_cands else [])]

    comparator_val = trt_cats[0] if trt_cats else ""
    active_val = trt_cats[1] if len(trt_cats) > 1 else comparator_val

    # Smart alignment with Target RCT experimental vs comparator arms (and drug-role heuristics when no RCT yet)
    if len(trt_cats) >= 2:
        exp_text = " ".join((target_rct or {}).get("experimental_arms", []) + (target_rct or {}).get("interventions", [])[:1]).lower()
        comp_text = " ".join((target_rct or {}).get("comparator_arms", [])).lower()
        scores_exp = []
        for val in trt_cats[:2]:
            vl = val.lower().replace("_", " ")
            toks = [t for t in vl.split() if len(t) >= 4]
            s = sum(10 for t in toks if t in exp_text) - sum(8 for t in toks if t in comp_text)
            if any(k in vl for k in ("placebo", "standard", "control", "comparator", "sitagliptin", "warfarin", "doublet")):
                s -= 6
            if any(k in vl for k in ("dapagliflozin", "empagliflozin", "pembrolizumab", "nivolumab", "apixaban", "finerenone", "active", "plus")):
                s += 6
            scores_exp.append((s, val))
        scores_exp.sort(key=lambda x: x[0], reverse=True)
        active_val = scores_exp[0][1]
        comparator_val = [v for v in trt_cats if v != active_val][0]

    # Pick time column not colliding with trt_col
    time_col = ""
    time_unit = "days"
    time_p75 = 365.0
    time_max = 540.0
    for tc in time_cands:
        if tc["column"] != trt_col:
            time_col = tc["column"]
            time_unit = tc["inferred_unit"]
            factor = TIME_UNIT_FACTORS.get(time_unit, 1.0)
            time_p75 = float(tc.get("p75", 365.0)) * factor
            time_max = float(tc.get("max", 540.0)) * factor
            break

    out_col = ""
    out_map = {"primary_event_values": [], "competing_event_values": [], "censor_values": []}
    for oc in out_cands:
        if oc["column"] not in {trt_col, time_col}:
            out_col = oc["column"]
            out_map = oc["suggested_mapping"]
            break

    excluded_roles = {
        profile.get("id_column_guess"),
        trt_col,
        time_col,
        out_col,
    }
    covariates: list[str] = []
    for col in profile.get("columns", []):
        cname = col["name"]
        if cname in excluded_roles:
            continue
        if col["inferred_role"] in ("identifier", "constant_or_unusable"):
            continue
        if col["dtype_kind"] == "categorical" and int(col.get("n_unique", 99)) > 12:
            continue
        covariates.append(cname)

    # Align horizons from Target RCT and clip safely within cohort's observed follow-up range
    raw_horizons = list((target_rct or {}).get("parsed_horizon_days") or [90.0, 180.0, 365.0])
    safe_cap = max(30.0, round(max(time_p75 * 1.15, time_max * 0.85), 1))
    clipped_horizons = sorted({round(float(h), 1) for h in raw_horizons if 7.0 <= float(h) < safe_cap})
    if not clipped_horizons:
        h1 = round(max(7.0, safe_cap * 0.30), 1)
        h2 = round(max(h1 + 7.0, safe_cap * 0.65), 1)
        clipped_horizons = sorted({h1, h2})
    admin_end = round(max(time_max + 1.0, clipped_horizons[-1] + 30.0), 1)

    # Suggest intuitive baseline eligibility rules if matching clinical columns exist
    suggested_eligibility: list[dict] = []
    col_map_by_name = {c["name"]: c for c in profile.get("columns", [])}
    for age_col in ("age_years", "age"):
        if age_col in col_map_by_name and col_map_by_name[age_col]["dtype_kind"] == "numeric":
            min_obs = col_map_by_name[age_col]["numeric_summary"]["min"]
            thresh = 45 if min_obs < 45 else 18
            suggested_eligibility.append(
                {
                    "id": "inc_baseline_age",
                    "kind": "inclusion",
                    "variable": age_col,
                    "operator": ">=",
                    "value": thresh,
                    "description": f"纳入基线年龄 ≥ {thresh} 岁患者",
                }
            )
            break
    if "egfr_ml_min" in col_map_by_name and col_map_by_name["egfr_ml_min"]["dtype_kind"] == "numeric":
        suggested_eligibility.append(
            {
                "id": "inc_baseline_egfr",
                "kind": "inclusion",
                "variable": "egfr_ml_min",
                "operator": ">=",
                "value": 30,
                "description": "纳入基线 eGFR ≥ 30 mL/min/1.73m²（对标 RCT 肾功能安全下限）",
            }
        )
    elif "ecog_ps" in col_map_by_name and col_map_by_name["ecog_ps"]["dtype_kind"] == "numeric":
        suggested_eligibility.append(
            {
                "id": "exc_ecog_2plus",
                "kind": "exclusion",
                "variable": "ecog_ps",
                "operator": ">=",
                "value": 2,
                "description": "排除基线 ECOG 体力评分 ≥ 2 分患者（对标 KEYNOTE 试验入组要求）",
            }
        )

    rct_title = (target_rct or {}).get("brief_title", "")
    nct_id = (target_rct or {}).get("nct_id", "")
    study_title = (
        f"对标 {nct_id} 目标试验模拟：{active_val} vs {comparator_val}"
        if nct_id
        else f"自有队列目标试验模拟：{active_val} vs {comparator_val}"
    )[:160]

    return {
        "study_title": study_title,
        "id_column": profile.get("id_column_guess") or "",
        "treatment_column": trt_col,
        "comparator_value": comparator_val,
        "active_value": active_val,
        "time_column": time_col,
        "time_unit": time_unit,
        "outcome_column": out_col,
        "primary_event_values": out_map["primary_event_values"],
        "competing_event_values": out_map["competing_event_values"],
        "censor_values": out_map["censor_values"],
        "covariate_columns": covariates,
        "eligibility_criteria": suggested_eligibility,
        "horizon_days": clipped_horizons,
        "administrative_end_day": admin_end,
        "weight_method": "overlap",
        "max_abs_smd": 0.10,
        "min_ess_ratio": 0.20,
        "positivity_epsilon": 0.01,
        "bootstrap_iterations": 100,
        "seed": 42,
        "target_rct_title": rct_title,
    }


def _evaluate_raw_criterion(series: pd.Series, operator: str, raw_val: object) -> pd.Series:
    num_s = pd.to_numeric(series, errors="coerce")
    try:
        target_f = float(raw_val)  # type: ignore[arg-type]
        is_num_cmp = True
    except (ValueError, TypeError):
        target_f = math.nan
        is_num_cmp = False

    if operator in (">=", ">", "<=", "<") and is_num_cmp:
        if operator == ">=":
            return num_s >= target_f
        if operator == ">":
            return num_s > target_f
        if operator == "<=":
            return num_s <= target_f
        return num_s < target_f

    if is_num_cmp and num_s.notna().sum() >= 0.9 * series.notna().sum():
        if operator == "==":
            return num_s == target_f
        if operator == "!=":
            return num_s.notna() & (num_s != target_f)

    s_str = series.astype(str).str.strip()
    na_mask = series.isna()
    if operator == "==":
        return (~na_mask) & (s_str == str(raw_val).strip())
    if operator == "!=":
        return (~na_mask) & (s_str != str(raw_val).strip())
    if operator == "in":
        targets = {str(x).strip() for x in (raw_val if isinstance(raw_val, list) else str(raw_val).split(","))}
        return (~na_mask) & s_str.isin(targets)
    if operator == "not_in":
        targets = {str(x).strip() for x in (raw_val if isinstance(raw_val, list) else str(raw_val).split(","))}
        return (~na_mask) & (~s_str.isin(targets))
    raise ContractError("INVALID_ELIGIBILITY", f"不支持的筛选比较符：{operator}")


def ingest_user_cohort(
    source_path: Path,
    mapping: dict,
    *,
    out_dir: Path,
    target_rct: dict | None = None,
) -> dict:
    """Ingest a user CSV/TSV according to ColumnMappingSpec, apply attrition & one-hot encoding, and publish standardized artifacts."""
    try:
        raw_bytes = Path(source_path).read_bytes()
    except OSError as exc:
        raise ContractError("COHORT_IO_ERROR", str(exc)) from exc

    input_sha = sha256_bytes(raw_bytes)
    df_raw, delimiter, encoding = read_tabular_cohort_bytes(raw_bytes, filename=Path(source_path).name)

    trt_col = str(mapping.get("treatment_column", "")).strip()
    comp_val = str(mapping.get("comparator_value", "")).strip()
    act_val = str(mapping.get("active_value", "")).strip()
    time_col = str(mapping.get("time_column", "")).strip()
    time_unit = str(mapping.get("time_unit", "days")).strip()
    out_col = str(mapping.get("outcome_column", "")).strip()
    id_col = str(mapping.get("id_column", "")).strip()

    if time_unit not in TIME_UNIT_FACTORS:
        raise ContractError("INVALID_TIME_UNIT", f"不支持的时间单位 '{time_unit}'（可选：days / months / years）")
    for req_name, col_val in (("暴露分组列", trt_col), ("随访时间列", time_col), ("结局状态列", out_col)):
        if not col_val or col_val not in df_raw.columns:
            raise ContractError("INVALID_COLUMN_MAPPING", f"{req_name} '{col_val}' 不存在于上传的数据表中。")
    if not comp_val or not act_val or comp_val == act_val:
        raise ContractError("INVALID_ARM_MAPPING", "对照臂（Comparator）与试验臂（Active）必须选择两个不同的分组取值。")

    primary_vals = {str(v).strip() for v in (mapping.get("primary_event_values") or []) if str(v).strip()}
    competing_vals = {str(v).strip() for v in (mapping.get("competing_event_values") or []) if str(v).strip()}
    censor_vals = {str(v).strip() for v in (mapping.get("censor_values") or []) if str(v).strip()}
    if not primary_vals or not censor_vals:
        raise ContractError("INVALID_OUTCOME_MAPPING", "必须至少指定 1 个主要结局事件取值（→1）与 1 个删失/存活取值（→0）。")
    if (primary_vals & competing_vals) or (primary_vals & censor_vals) or (competing_vals & censor_vals):
        raise ContractError("OVERLAPPING_OUTCOME_MAPPING", "主要事件（1）、竞争事件（2）与删失（0）的映射取值不能重叠。")

    cov_cols = [str(c).strip() for c in (mapping.get("covariate_columns") or []) if str(c).strip()]
    cov_cols = [c for c in dict.fromkeys(cov_cols) if c in df_raw.columns and c not in {trt_col, time_col, out_col, id_col}]
    if not cov_cols:
        raise ContractError("NO_COVARIATES_SELECTED", "请至少勾选 1 个基线混杂协变量用于倾向得分加权平衡。")

    comp_arm_id = "arm_comparator"
    act_arm_id = "arm_active"
    all_valid_outcomes = primary_vals | competing_vals | censor_vals

    def _map_event_series(s: pd.Series) -> pd.Series:
        def _conv(val) -> int:
            if pd.isna(val):
                return -1
            sv = str(val).strip()
            # Also handle float-formatted int strings like "1.0" -> "1"
            sv_int = sv[:-2] if sv.endswith(".0") and sv[:-2].isdigit() else sv
            if sv in primary_vals or sv_int in primary_vals:
                return 1
            if sv in competing_vals or sv_int in competing_vals:
                return 2
            if sv in censor_vals or sv_int in censor_vals:
                return 0
            return -1

        return s.map(_conv)

    def _arm_counts(sub_df: pd.DataFrame) -> dict[str, int]:
        ts = sub_df[trt_col].astype(str).str.strip()
        return {
            comp_arm_id: int((ts == comp_val).sum()),
            act_arm_id: int((ts == act_val).sum()),
        }

    def _attrition_entry(step_order: int, step_id: str, desc: str, rem_df: pd.DataFrame, exc_df: pd.DataFrame) -> dict:
        ev_exc = _map_event_series(exc_df[out_col]) if len(exc_df) > 0 else pd.Series([], dtype=int)
        return {
            "step_order": step_order,
            "step_id": step_id,
            "description": desc,
            "remaining_n": int(len(rem_df)),
            "excluded_n": int(len(exc_df)),
            "excluded_primary_events": int((ev_exc == 1).sum()),
            "excluded_competing_events": int((ev_exc == 2).sum()),
            "excluded_censored": int((ev_exc == 0).sum()),
            "arm_counts_json": json.dumps(_arm_counts(rem_df), sort_keys=True),
        }

    work_df = df_raw.copy()
    if id_col and id_col in work_df.columns:
        work_df["__subject_id"] = [
            str(v).strip() if pd.notna(v) and str(v).strip() else f"SUBJ-{idx + 1:06d}"
            for idx, v in enumerate(work_df[id_col])
        ]
    else:
        work_df["__subject_id"] = [f"SUBJ-{idx + 1:06d}" for idx in range(len(work_df))]

    attrition_steps: list[dict] = [
        _attrition_entry(0, "raw_uploaded_cohort", f"原始上传数据表总记录数（{Path(source_path).name}）", work_df, work_df.iloc[0:0])
    ]
    step_idx = 1

    # Step 1: Retain subjects in comparator or active arm
    trt_str = work_df[trt_col].astype(str).str.strip()
    trt_mask = work_df[trt_col].notna() & trt_str.isin({comp_val, act_val})
    if not bool(trt_mask.all()):
        exc = work_df.loc[~trt_mask]
        work_df = work_df.loc[trt_mask].copy().reset_index(drop=True)
        attrition_steps.append(
            _attrition_entry(
                step_idx,
                "valid_treatment_arms",
                f"保留暴露列 {trt_col} 属于对照臂 '{comp_val}' 或试验臂 '{act_val}' 的记录",
                work_df,
                exc,
            )
        )
        step_idx += 1

    # Step 2: Retain valid positive follow-up time and mapped outcome code
    time_num = pd.to_numeric(work_df[time_col], errors="coerce")
    ev_mapped = _map_event_series(work_df[out_col])
    valid_to_mask = time_num.notna() & np.isfinite(time_num) & (time_num > 0.0) & (ev_mapped >= 0)
    if not bool(valid_to_mask.all()):
        exc = work_df.loc[~valid_to_mask]
        work_df = work_df.loc[valid_to_mask].copy().reset_index(drop=True)
        attrition_steps.append(
            _attrition_entry(
                step_idx,
                "valid_time_and_outcome",
                f"剔除随访时长缺失/≤0（{time_col}）或结局状态未映射（{out_col}，合法取值 {sorted(all_valid_outcomes)}）记录",
                work_df,
                exc,
            )
        )
        step_idx += 1

    # Step 3..k: User-configured baseline eligibility criteria
    raw_criteria = mapping.get("eligibility_criteria") or []
    for c_idx, crit in enumerate(raw_criteria):
        if not isinstance(crit, dict):
            continue
        c_var = str(crit.get("variable", "")).strip()
        c_op = str(crit.get("operator", "")).strip()
        c_kind = str(crit.get("kind", "inclusion")).strip()
        c_val = crit.get("value")
        if not c_var or c_var not in work_df.columns or c_val is None or str(c_val).strip() == "":
            continue
        matched = _evaluate_raw_criterion(work_df[c_var], c_op, c_val)
        keep = matched if c_kind == "inclusion" else (~matched & work_df[c_var].notna())
        exc = work_df.loc[~keep]
        work_df = work_df.loc[keep].copy().reset_index(drop=True)
        action_zh = "纳入条件" if c_kind == "inclusion" else "排除条件"
        desc = crit.get("description") or f"{action_zh}：{c_var} {c_op} {c_val}"
        attrition_steps.append(
            _attrition_entry(
                step_idx,
                str(crit.get("id") or f"elig_{c_idx + 1}"),
                str(desc),
                work_df,
                exc,
            )
        )
        step_idx += 1

    # Step k+1: Complete-case baseline covariate check
    cov_complete_mask = work_df[cov_cols].notna().all(axis=1)
    if not bool(cov_complete_mask.all()):
        exc = work_df.loc[~cov_complete_mask]
        work_df = work_df.loc[cov_complete_mask].copy().reset_index(drop=True)
        attrition_steps.append(
            _attrition_entry(
                step_idx,
                "complete_case_covariates",
                f"完全病例筛选：剔除所选基线协变量（{', '.join(cov_cols)}）存在缺失值的个体",
                work_df,
                exc,
            )
        )
        step_idx += 1
    else:
        attrition_steps.append(
            _attrition_entry(
                step_idx,
                "complete_case_covariates",
                f"完全病例核查：所选 {len(cov_cols)} 项基线协变量均无缺失（剔除 0 例）",
                work_df,
                work_df.iloc[0:0],
            )
        )
        step_idx += 1

    if len(work_df) == 0:
        raise ContractError("EMPTY_ELIGIBLE_COHORT", "纳排条件与缺失值过滤后剩余 0 例患者，请放宽筛选条件。")

    arm_str_final = work_df[trt_col].astype(str).str.strip()
    n_comp = int((arm_str_final == comp_val).sum())
    n_act = int((arm_str_final == act_val).sum())
    if n_comp == 0 or n_act == 0:
        raise ContractError(
            "SINGLE_ARM_AFTER_SCREENING",
            f"纳排筛选后对照臂（{comp_val}: {n_comp} 例）或试验臂（{act_val}: {n_act} 例）人数为 0，无法进行比较。",
        )

    # Convert follow-up time to days and apply administrative censoring at administrative_end_day
    factor = TIME_UNIT_FACTORS[time_unit]
    raw_days = pd.to_numeric(work_df[time_col], errors="raise").to_numpy(dtype=np.float64) * factor
    raw_events = _map_event_series(work_df[out_col]).to_numpy(dtype=np.int64)

    landmark_day, parsed_horizons = apply_timeline_window(mapping)
    horizons = list(parsed_horizons or [90.0, 180.0, 365.0])
    if not horizons:
        raise ContractError("MISSING_HORIZON", "请至少指定 1 个正数评估时点（天）。")
    if landmark_day >= min(float(h) for h in horizons):
        raise ContractError("INVALID_TIMELINE_WINDOW", "landmark_time 必须小于最短 horizon。")

    admin_end_day = float(mapping.get("administrative_end_day") or max(float(np.max(raw_days)) + 1.0, float(horizons[-1]) + 30.0))
    if admin_end_day <= max(float(horizons[-1]), float(landmark_day)):
        admin_end_day = round(max(float(horizons[-1]), float(landmark_day)) + 30.0, 2)

    beyond_admin = raw_days > admin_end_day
    duration_day = np.where(beyond_admin, admin_end_day, raw_days)
    event_code = np.where(beyond_admin, 0, raw_events)

    # Encode baseline covariates (continuous / binary / categorical one-hot) + drop constant/collinear columns
    encoded_cols: dict[str, np.ndarray] = {}
    cov_protocol_specs: list[dict] = []
    cov_display_labels: dict[str, str] = {}
    dropped_covariates: list[dict] = []

    for orig_col in cov_cols:
        s_cov = work_df[orig_col]
        num_try = pd.to_numeric(s_cov, errors="coerce")
        if bool(num_try.notna().all()) and s_cov.dtype != object or float(num_try.notna().mean()) == 1.0:
            arr = num_try.to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(arr)) or float(np.std(arr)) <= 1e-12:
                dropped_covariates.append(
                    {"column": orig_col, "reason": "zero_variance_constant_in_eligible_cohort"}
                )
                continue
            uniq = sorted(np.unique(arr).tolist())
            safe_name = _sanitize_identifier(orig_col, prefix="cov")
            while safe_name in encoded_cols:
                safe_name = f"{safe_name}_n"
            ctype = "binary" if uniq == [0.0, 1.0] else "continuous"
            encoded_cols[safe_name] = arr
            cov_protocol_specs.append(
                {"name": safe_name, "type": ctype, "assessed_day_offset": 0, "description": f"Source column: {orig_col}"}
            )
            cov_display_labels[safe_name] = orig_col
        else:
            cats_series = s_cov.astype(str).str.strip()
            levels = sorted(cats_series.unique().tolist())
            if len(levels) <= 1:
                dropped_covariates.append(
                    {"column": orig_col, "reason": "single_category_constant_in_eligible_cohort"}
                )
                continue
            if len(levels) > 20:
                dropped_covariates.append(
                    {"column": orig_col, "reason": f"high_cardinality_categorical_({len(levels)}_levels)"}
                )
                continue
            ref_level = levels[0]
            for lvl in levels[1:]:
                indicator = (cats_series == lvl).to_numpy(dtype=np.float64)
                if float(np.std(indicator)) <= 1e-12:
                    dropped_covariates.append(
                        {"column": f"{orig_col}={lvl}", "reason": "zero_variance_dummy_level"}
                    )
                    continue
                dummy_name = _sanitize_identifier(f"{orig_col}__{lvl}", prefix="cat")
                while dummy_name in encoded_cols:
                    dummy_name = f"{dummy_name}_d"
                encoded_cols[dummy_name] = indicator
                cov_protocol_specs.append(
                    {
                        "name": dummy_name,
                        "type": "binary",
                        "assessed_day_offset": 0,
                        "description": f"One-hot indicator {orig_col} == '{lvl}' (ref: '{ref_level}')",
                    }
                )
                cov_display_labels[dummy_name] = f"{orig_col} = {lvl}（参照：{ref_level}）"

    # Collinearity / near-collinearity rank-deficiency guard across encoded design matrix [1, X]
    kept_specs: list[dict] = []
    kept_matrix_cols: list[np.ndarray] = [np.ones(len(work_df), dtype=np.float64)]
    current_rank = 1
    for spec in cov_protocol_specs:
        cname = spec["name"]
        col_vec = encoded_cols[cname]
        col_std = (col_vec - np.mean(col_vec)) / max(float(np.std(col_vec)), 1e-12)
        cand_mat = np.column_stack([*kept_matrix_cols, col_std])
        # Check residual variance after OLS projection onto existing columns (drops R^2 > 0.9995 near-collinear columns)
        coef, *_ = np.linalg.lstsq(np.column_stack(kept_matrix_cols), col_std, rcond=None)
        resid_sd = float(np.std(col_std - np.column_stack(kept_matrix_cols) @ coef))
        new_rank = int(np.linalg.matrix_rank(cand_mat, tol=1e-4))
        if new_rank > current_rank and resid_sd > 0.02:
            kept_specs.append(spec)
            kept_matrix_cols.append(col_std)
            current_rank = new_rank
        else:
            dropped_covariates.append({"column": cname, "reason": "collinear_with_existing_covariates"})
            encoded_cols.pop(cname, None)
            cov_display_labels.pop(cname, None)

    if not kept_specs:
        raise ContractError(
            "ALL_COVARIATES_CONSTANT",
            "所选基线协变量在纳排筛选后均为常数列或完全共线（方差为 0），请重新选择具有变异度的基线特征。",
        )

    # Construct validated protocol v1 JSON aligned with standardized cohort
    study_title = str(mapping.get("study_title") or f"自有队列目标试验模拟：{act_val} vs {comp_val}").strip()[:200]
    protocol_dict: dict = {
        "protocol_version": "1.0",
        "project_mode": SOURCE_MODE_USER_COHORT,
        "study_title": study_title,
        "simulation_or_data_label": f"自有脱敏队列真项目分析（来源文件：{Path(source_path).name}，观察性因果推断）",
        "eligibility": {
            "inclusion_criteria": [],
            "exclusion_criteria": [],
        },
        "time_zero": {
            "anchor_event": f"baseline_{_sanitize_identifier(trt_col, 'exposure')}_initiation",
            "day_offset": 0,
            "landmark_day": landmark_day,
            "description": (
                f"目标策略起始基线日（{trt_col}: {act_val} vs {comp_val}）"
                if landmark_day <= 0
                else f"地标风险集：排除事件或删失时间不超过 landmark_time={landmark_day} 天的受试者"
            ),
        },
        "treatment_strategies": [
            {"id": comp_arm_id, "label": comp_val, "role": "control", "description": f"{trt_col} == '{comp_val}'"},
            {"id": act_arm_id, "label": act_val, "role": "active", "description": f"{trt_col} == '{act_val}'"},
        ],
        "assignment_mode": "observational_landmark" if landmark_day > 0 else "observational_baseline",
        "followup": {
            "horizon_days": horizons,
            "administrative_end_day": admin_end_day,
        },
        "outcomes": {
            "primary_event_code": 1,
            "competing_event_codes": [2],
            "censor_code": 0,
            "competing_event_handling": "explicit_cif_and_subdistribution",
        },
        "estimand": {
            "target_estimand": "ATO" if mapping.get("weight_method", "overlap") == "overlap" else "ATE",
            "effect_measures": ["RD", "RR_CIF", "sHR", "csHR"],
            "diagnostic_gates": {
                "max_abs_smd": float(mapping.get("max_abs_smd", 0.10)),
                "min_ess_ratio": float(mapping.get("min_ess_ratio", 0.20)),
                "positivity_epsilon": float(mapping.get("positivity_epsilon", 0.01)),
            },
        },
        "covariates": kept_specs,
    }
    if target_rct and isinstance(target_rct, dict) and re.fullmatch(r"NCT[0-9]{8}", str(target_rct.get("nct_id", ""))):
        po_list = target_rct.get("primary_outcomes") or [{}]
        trct_entry: dict = {
            "nct_id": str(target_rct["nct_id"]),
            "brief_title": str(target_rct.get("brief_title", target_rct["nct_id"])),
            "record_url": str(target_rct.get("record_url", f"https://clinicaltrials.gov/study/{target_rct['nct_id']}")),
            "source": str(target_rct.get("source", "clinicaltrials_gov_api_v2")),
            "receipt_sha256": str(target_rct.get("receipt_sha256", "")),
            "primary_outcome_measure": str(po_list[0].get("measure", "")),
            "primary_outcome_timeframe": str(po_list[0].get("timeFrame", "")),
        }
        if target_rct.get("arm_alignment_status"):
            trct_entry["arm_alignment_status"] = str(target_rct["arm_alignment_status"])
        if target_rct.get("arm_alignment_warning"):
            trct_entry["arm_alignment_warning"] = str(target_rct["arm_alignment_warning"])
        if target_rct.get("overall_alignment_status"):
            trct_entry["overall_alignment_status"] = str(target_rct["overall_alignment_status"])
        if target_rct.get("protocol_alignment_warning"):
            trct_entry["protocol_alignment_warning"] = str(target_rct["protocol_alignment_warning"])
        protocol_dict["target_rct"] = trct_entry

    normalized_protocol = validate_protocol_dict(protocol_dict)

    n_primary_obs = int((event_code == 1).sum())
    n_competing_obs = int((event_code == 2).sum())
    n_censor_obs = int((event_code == 0).sum())
    zero_competing_notice = (
        "当前队列未映射或未观察到竞争风险事件（event_code=2 为 0 例）：加权 Aalen–Johansen 累计发生率精确退化为标准加权 Kaplan–Meier (1-S)，Fine–Gray sHR 等价于主要结局原因特异 Cox HR。"
        if n_competing_obs == 0
        else None
    )

    arm_idx_arr = (arm_str_final == act_val).to_numpy(dtype=np.int64)
    arm_id_arr = np.where(arm_idx_arr == 1, act_arm_id, comp_arm_id)

    std_df = pd.DataFrame(
        {
            "subject_id": work_df["__subject_id"].tolist(),
            "treatment_arm": arm_id_arr.tolist(),
            "treatment_idx": arm_idx_arr.tolist(),
            "duration_day": [f"{float(d):.8f}" for d in duration_day],
            "event_code": event_code.tolist(),
        }
    )
    for spec in kept_specs:
        cname = spec["name"]
        vals = encoded_cols[cname]
        if spec["type"] == "binary":
            std_df[cname] = [int(round(float(v))) for v in vals]
        else:
            std_df[cname] = [f"{float(v):.8f}" for v in vals]

    commit, code_dirty = code_provenance()
    deps = dependency_versions()
    output_paths = [
        "standardized_cohort.csv",
        "ingest_attrition.csv",
        "normalized_protocol.json",
        "ingest_manifest.json",
    ]

    def _populate(stage: Path) -> None:
        cohort_out = stage / "standardized_cohort.csv"
        std_df.to_csv(cohort_out, index=False, lineterminator="\n")

        attr_out = stage / "ingest_attrition.csv"
        fieldnames = [
            "step_order",
            "step_id",
            "description",
            "remaining_n",
            "excluded_n",
            "excluded_primary_events",
            "excluded_competing_events",
            "excluded_censored",
            "arm_counts_json",
        ]
        with attr_out.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(attrition_steps)

        (stage / "normalized_protocol.json").write_bytes(json_bytes(normalized_protocol))

        std_sha = sha256_file(cohort_out)
        ingest_manifest = {
            "status": "user_cohort_ingested",
            "source_mode": SOURCE_MODE_USER_COHORT,
            "input_filename": Path(source_path).name,
            "input_sha256": input_sha,
            "detected_delimiter": delimiter,
            "detected_encoding": encoding,
            "standardized_cohort_sha256": std_sha,
            "raw_row_count": int(len(df_raw)),
            "eligible_row_count": int(len(std_df)),
            "arm_counts": {comp_arm_id: n_comp, act_arm_id: n_act},
            "event_counts": {
                "primary_1": n_primary_obs,
                "competing_2": n_competing_obs,
                "censor_0": n_censor_obs,
            },
            "zero_competing_events_notice": zero_competing_notice,
            "time_conversion": {
                "source_column": time_col,
                "source_unit": time_unit,
                "multiplier_to_days": factor,
                "administrative_end_day": admin_end_day,
            },
            "column_mapping": {
                "id_column": id_col or None,
                "treatment_column": trt_col,
                "comparator_value": comp_val,
                "active_value": act_val,
                "time_column": time_col,
                "time_unit": time_unit,
                "outcome_column": out_col,
                "primary_event_values": sorted(primary_vals),
                "competing_event_values": sorted(competing_vals),
                "censor_values": sorted(censor_vals),
                "selected_raw_covariates": cov_cols,
                "eligibility_criteria": raw_criteria,
            },
            "covariate_display_labels": cov_display_labels,
            "dropped_constant_or_collinear_covariates": dropped_covariates,
            "code_commit": commit,
            "code_dirty": code_dirty,
            "dependency_versions": deps,
            "run_at_utc": utc_now_iso(),
        }
        (stage / "ingest_manifest.json").write_bytes(json_bytes(ingest_manifest))
        run_manifest = {
            "status": "user_cohort_ingested",
            "source_mode": SOURCE_MODE_USER_COHORT,
            "input_sha256": input_sha,
            "output_paths": output_paths,
            "output_sha256": {p: sha256_file(stage / p) for p in output_paths},
            "code_commit": commit,
            "code_dirty": code_dirty,
            "run_at_utc": utc_now_iso(),
        }
        (stage / "run_manifest.json").write_bytes(json_bytes(run_manifest))

    atomic_publish(out_dir, _populate)
    return {
        "status": "user_cohort_ingested",
        "source_mode": SOURCE_MODE_USER_COHORT,
        "input_sha256": input_sha,
        "raw_row_count": int(len(df_raw)),
        "eligible_row_count": int(len(std_df)),
        "dropped_covariates": dropped_covariates,
        "zero_competing_events_notice": zero_competing_notice,
        "output_path": str(out_dir),
    }
