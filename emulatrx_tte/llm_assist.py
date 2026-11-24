"""Privacy-guarded OpenAI-compatible LLM helper with 100% deterministic rule-engine fallback.

STRICT PRIVACY INVARIANT:
Any outbound LLM payload constructed from a user cohort MUST pass through `sanitize_profile_for_llm`
and `assert_llm_payload_privacy_safe`. It may contain ONLY table-level column names, inferred types,
missingness rates, deduplicated short category labels (<= 20 labels, never identifier columns or ID-like
tokens), and aggregate numeric quantiles/means/SDs — NEVER a single individual patient record or subject ID.
"""

from __future__ import annotations

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen as _std_urlopen

from ._common import ContractError


DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_LLM_TIMEOUT = 15

# Regexes matching identifier-like column names or subject ID values that must NEVER leave the process
ID_COL_RE = re.compile(
    r"(?:^|_)(?:id|pat|patient|subj|subject|case_no|case_id|mrn|hosp|record|ssn|uuid|name|phone|email)(?:$|_)",
    re.IGNORECASE,
)
ID_VALUE_RE = re.compile(
    r"^(?:[A-Za-z]{2,12}[-_][0-9]{3,}[-_0-9]*|[0-9]{5,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)

# Bilingual clinical term lexicon for deterministic PICO & ClinicalTrials.gov English search translation
BILINGUAL_CLINICAL_TERMS: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (("达格列净", "dapagliflozin", "sglt2"), "Dapagliflozin", "intervention", "达格列净 (Dapagliflozin)"),
    (("恩格列净", "empagliflozin"), "Empagliflozin", "intervention", "恩格列净 (Empagliflozin)"),
    (("卡格列净", "canagliflozin"), "Canagliflozin", "intervention", "卡格列净 (Canagliflozin)"),
    (("西格列汀", "sitagliptin", "dpp-4", "dpp4"), "Sitagliptin", "comparator", "西格列汀 (Sitagliptin)"),
    (("利格列汀", "linagliptin"), "Linagliptin", "comparator", "利格列汀 (Linagliptin)"),
    (("司美格鲁肽", "semaglutide", "glp-1", "glp1"), "Semaglutide", "intervention", "司美格鲁肽 (Semaglutide)"),
    (("非奈利酮", "finerenone"), "Finerenone", "intervention", "非奈利酮 (Finerenone)"),
    (("帕博利珠单抗", "pembrolizumab", "keytruda", "pd-1", "pd1"), "Pembrolizumab", "intervention", "帕博利珠单抗 (Pembrolizumab)"),
    (("纳武利尤单抗", "nivolumab"), "Nivolumab", "intervention", "纳武利尤单抗 (Nivolumab)"),
    (("阿替利珠单抗", "atezolizumab"), "Atezolizumab", "intervention", "阿替利珠单抗 (Atezolizumab)"),
    (("奥希替尼", "osimertinib"), "Osimertinib", "intervention", "奥希替尼 (Osimertinib)"),
    (("含铂", "双药化疗", "platinum", "chemo", "pemetrexed", "培美曲塞"), "Platinum Chemotherapy", "comparator", "含铂双药化疗 (Platinum Doublet Chemo)"),
    (("阿哌沙班", "apixaban", "doac", "noac"), "Apixaban", "intervention", "阿哌沙班 (Apixaban)"),
    (("利伐沙班", "rivaroxaban"), "Rivaroxaban", "intervention", "利伐沙班 (Rivaroxaban)"),
    (("华法林", "warfarin"), "Warfarin", "comparator", "华法林 (Warfarin)"),
    (("安慰剂", "标准治疗", "placebo", "standard of care"), "Placebo / Standard Care", "comparator", "标准对照治疗"),
    (("心衰", "心力衰竭", "heart failure", "hfref", "hfpef", "nyha"), "Heart Failure", "condition", "心力衰竭 (Heart Failure)"),
    (("2型糖尿病", "糖尿病", "diabetes", "t2dm", "hba1c"), "Type 2 Diabetes", "condition", "2 型糖尿病 (Type 2 Diabetes)"),
    (("慢性肾病", "慢性肾脏病", "糖尿病肾病", "肾功能", "ckd", "egfr", "kidney"), "Chronic Kidney Disease", "condition", "慢性肾脏病 (CKD)"),
    (("非小细胞肺癌", "肺癌", "nsclc", "lung cancer", "pdl1", "pd-l1"), "Non-Small Cell Lung Cancer", "condition", "晚期非小细胞肺癌 (NSCLC)"),
    (("乳腺癌", "breast cancer"), "Breast Cancer", "condition", "乳腺癌 (Breast Cancer)"),
    (("房颤", "心房颤动", "atrial fibrillation", "afib"), "Atrial Fibrillation", "condition", "心房颤动 (Atrial Fibrillation)"),
    (("脓毒症", "重症", "sepsis", "icu"), "Sepsis", "condition", "脓毒症 / 重症感染 (Sepsis)"),
    (("卒中", "脑卒中", "栓塞", "stroke", "embolism"), "Stroke or Systemic Embolism", "outcome", "卒中或体循环栓塞"),
    (("心衰恶化", "心衰住院", "心血管死亡", "心血管事件", "hf_or_cv", "cardiovascular"), "Heart Failure Hospitalization or CV Death", "outcome", "心衰住院或心血管事件"),
    (("进展", "肿瘤死亡", "无进展生存", "总生存", "progression", "cancerdeath", "os_days"), "Disease Progression or Cancer Death", "outcome", "疾病进展或肿瘤特异死亡"),
    (("肾衰竭", "终末期肾病", "esrd", "kidney failure"), "Kidney Failure or Sustained eGFR Decline", "outcome", "持续肾功能恶化或终末期肾病"),
)


def urlopen_llm(request: Request, timeout: float):
    """Module-level transport wrapper so offline unit tests can intercept and audit outbound LLM payloads."""
    return _std_urlopen(request, timeout=timeout)


def _resolve_llm_config() -> dict:
    api_key = (
        os.environ.get("EMULATRX_LLM_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    base_url = (
        os.environ.get("EMULATRX_LLM_BASE_URL", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "").strip()
        or DEFAULT_LLM_BASE_URL
    ).rstrip("/")
    model = (
        os.environ.get("EMULATRX_LLM_MODEL", "").strip()
        or os.environ.get("OPENAI_MODEL", "").strip()
        or DEFAULT_LLM_MODEL
    )
    try:
        timeout = float(os.environ.get("EMULATRX_LLM_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT)))
    except ValueError:
        timeout = float(DEFAULT_LLM_TIMEOUT)
    enabled = bool(api_key)
    return {
        "enabled": enabled,
        "api_key": api_key,
        "base_url": base_url,
        "model": model if enabled else None,
        "configured_model": model,
        "timeout": max(2.0, min(timeout, 30.0)),
    }


def get_llm_status() -> dict:
    """Return safe public status for the Web UI and MCP tools (never exposes API keys)."""
    cfg = _resolve_llm_config()
    if cfg["enabled"]:
        return {
            "enabled": True,
            "mode": "llm_assisted",
            "label_zh": f"AI 智能辅助已启用（{cfg['configured_model']}）",
            "model": cfg["configured_model"],
            "privacy_policy_zh": "严格表头与汇总统计脱敏：仅发送列名、类型、缺失率、去重短枚举（≤20项）与分位数汇总，0 行患者记录外发",
        }
    return {
        "enabled": False,
        "mode": "deterministic_rules",
        "label_zh": "本地规则引擎模式（未配置外部大模型密钥，全本地零外发运行）",
        "model": None,
        "privacy_policy_zh": "全本地确定性规则运行：无任何外部大模型网络调用，个体患者记录 100% 留存本机独立会话目录",
    }


def sanitize_profile_for_llm(profile: dict, user_note: str = "") -> dict:
    """Strip a CohortProfile down to an allowlisted aggregate schema summary with ZERO patient rows or IDs."""
    safe_columns: list[dict] = []
    for col in profile.get("columns", []):
        col_name = str(col.get("name", ""))
        role = str(col.get("inferred_role", ""))
        # Never send identifier columns or high-cardinality free text
        if role == "identifier" or ID_COL_RE.search(col_name):
            continue
        entry: dict = {
            "name": col_name[:80],
            "dtype_kind": str(col.get("dtype_kind", "unknown")),
            "inferred_role": role,
            "missing_rate": round(float(col.get("missing_rate", 0.0)), 4),
            "n_unique": int(col.get("n_unique", 0)),
        }
        if col.get("dtype_kind") == "numeric" and isinstance(col.get("numeric_summary"), dict):
            ns = col["numeric_summary"]
            entry["numeric_summary"] = {
                k: (round(float(ns[k]), 4) if ns.get(k) is not None else None)
                for k in ("mean", "sd", "p25", "median", "p75")
            }
        elif isinstance(col.get("top_categories"), list) and int(col.get("n_unique", 999)) <= 20:
            safe_labels: list[str] = []
            for item in col["top_categories"][:20]:
                lbl = str(item.get("value", "")).strip()
                if not lbl or len(lbl) > 64 or ID_VALUE_RE.match(lbl):
                    continue
                safe_labels.append(lbl)
            entry["distinct_categories"] = safe_labels
        safe_columns.append(entry)

    # Scrub any accidental ID-like tokens from optional user note
    clean_note = re.sub(
        r"\b[A-Za-z]{2,12}[-_][0-9]{3,}[-_0-9]*\b",
        "[REDACTED_ID]",
        (user_note or "").strip()[:300],
    )
    return {
        "privacy_contract": "schema_and_aggregates_only_zero_individual_rows",
        "row_count": int(profile.get("row_count", 0)),
        "column_count": int(profile.get("column_count", 0)),
        "columns": safe_columns,
        "user_domain_note": clean_note,
    }


def assert_llm_payload_privacy_safe(payload_bytes: bytes, forbidden_values: list[str] | tuple[str, ...] = ()) -> None:
    """Raise ContractError if an outbound LLM payload contains ID-like tokens or forbidden raw values."""
    text = payload_bytes.decode("utf-8", errors="replace")
    for token in forbidden_values:
        if token and len(str(token)) >= 4 and str(token) in text:
            raise ContractError(
                "PRIVACY_VIOLATION_BLOCKED",
                "Blocked outbound LLM request containing a raw patient/row value.",
            )
    # Also scan JSON string tokens for Hosp-2023-0001 / ONC-880001 / SYN-000001 style subject IDs
    for match in re.findall(r"\b(?:Hosp|ONC|PAT|MRN|SUBJ|SYN)[-_][0-9]{3,}[-_0-9]*\b", text, flags=re.IGNORECASE):
        raise ContractError(
            "PRIVACY_VIOLATION_BLOCKED",
            f"Blocked outbound LLM request containing subject identifier pattern '{match}'.",
        )


def _call_chat_json(system_prompt: str, user_payload: dict, *, forbidden_values: tuple[str, ...] = ()) -> dict | None:
    """Invoke OpenAI-compatible chat completions endpoint returning a JSON dict, or None on any failure/absence."""
    cfg = _resolve_llm_config()
    if not cfg["enabled"]:
        return None

    body_obj = {
        "model": cfg["configured_model"],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }
    raw_body = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
    # Enforce ironclad privacy check BEFORE opening any network socket!
    assert_llm_payload_privacy_safe(raw_body, forbidden_values=forbidden_values)

    url = f"{cfg['base_url']}/chat/completions"
    req = Request(
        url,
        data=raw_body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
            "User-Agent": "EmulatRx-DualMode-Workbench/1.0",
        },
    )
    try:
        with urlopen_llm(req, timeout=cfg["timeout"]) as resp:
            resp_bytes = resp.read(256 * 1024)
        parsed = json.loads(resp_bytes.decode("utf-8"))
        content = parsed["choices"][0]["message"]["content"]
        if isinstance(content, str):
            content_clean = content.strip()
            if content_clean.startswith("```"):
                content_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", content_clean, flags=re.DOTALL)
            out = json.loads(content_clean)
            if isinstance(out, dict):
                return out
    except ContractError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError, KeyError, IndexError, ValueError, TypeError):
        return None
    return None


def _extract_deterministic_terms(text: str) -> dict:
    lowered = text.lower()
    conditions_en: list[str] = []
    conditions_zh: list[str] = []
    interventions_en: list[str] = []
    interventions_zh: list[str] = []
    comparators_en: list[str] = []
    comparators_zh: list[str] = []
    outcomes_zh: list[str] = []

    for triggers, en_term, role, zh_label in BILINGUAL_CLINICAL_TERMS:
        if any(t.lower() in lowered for t in triggers):
            if role == "condition" and en_term not in conditions_en:
                conditions_en.append(en_term)
                conditions_zh.append(zh_label)
            elif role == "intervention" and en_term not in interventions_en:
                interventions_en.append(en_term)
                interventions_zh.append(zh_label)
            elif role == "comparator" and en_term not in comparators_en:
                comparators_en.append(en_term)
                comparators_zh.append(zh_label)
            elif role == "outcome" and zh_label not in outcomes_zh:
                outcomes_zh.append(zh_label)

    # Also capture explicit capitalized English drug/medical tokens from text
    ascii_tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9-]{3,24}\b", text)
    stop = {
        "with", "from", "that", "this", "plus", "versus", "study", "trial", "cohort",
        "years", "months", "days", "event", "status", "censored", "alive", "first",
        "line", "regimen", "initiated", "drug", "prior", "history", "class",
    }
    extra_en = []
    for tok in ascii_tokens:
        if tok.lower() not in stop and tok not in extra_en:
            extra_en.append(tok)

    return {
        "conditions_en": conditions_en,
        "conditions_zh": conditions_zh,
        "interventions_en": interventions_en,
        "interventions_zh": interventions_zh,
        "comparators_en": comparators_en,
        "comparators_zh": comparators_zh,
        "outcomes_zh": outcomes_zh,
        "extra_en": extra_en[:5],
    }


def parse_research_idea_pico(idea_text: str) -> dict:
    """Parse free-text clinical research idea into structured PICO + English ClinicalTrials.gov search terms."""
    clean_idea = (idea_text or "").strip()
    if not clean_idea:
        raise ContractError("EMPTY_IDEA", "请输入临床研究想法或选择上方示例。")

    det = _extract_deterministic_terms(clean_idea)
    fallback_pico = {
        "population": "、".join(det["conditions_zh"]) if det["conditions_zh"] else f"目标临床人群（源自描述：{clean_idea[:48]}）",
        "intervention": det["interventions_zh"][0] if det["interventions_zh"] else (
            det["extra_en"][0] if det["extra_en"] else "试验暴露策略（Active Arm）"
        ),
        "comparator": det["comparators_zh"][0] if det["comparators_zh"] else (
            det["interventions_zh"][1] if len(det["interventions_zh"]) > 1 else "阳性对照或标准治疗（Comparator Arm）"
        ),
        "primary_outcome": det["outcomes_zh"][0] if det["outcomes_zh"] else "主要临床终点事件（时间至事件结局）",
        "competing_event": "非目标疾病特异死亡 / 其他原因竞争死亡（编码 2）",
        "condition_query_en": " ".join(det["conditions_en"][:2]),
        "intervention_query_en": " ".join((det["interventions_en"] + det["comparators_en"])[:2]),
        "search_query_en": " ".join(
            dict.fromkeys(det["conditions_en"][:2] + det["interventions_en"][:2] + det["extra_en"][:3])
        ).strip()
        or clean_idea[:80],
        "assist_mode": "deterministic_rules",
    }

    llm_out = _call_chat_json(
        (
            "You are a clinical epidemiologist specializing in Target Trial Emulation (TTE). "
            "Extract structured PICO in Chinese and concise English search terms for ClinicalTrials.gov API v2. "
            "Return JSON with keys: population, intervention, comparator, primary_outcome, competing_event, "
            "condition_query_en, intervention_query_en, search_query_en."
        ),
        {"idea_text": clean_idea[:1500]},
    )
    if isinstance(llm_out, dict) and llm_out.get("search_query_en"):
        merged = dict(fallback_pico)
        for k in (
            "population",
            "intervention",
            "comparator",
            "primary_outcome",
            "competing_event",
            "condition_query_en",
            "intervention_query_en",
            "search_query_en",
        ):
            val = llm_out.get(k)
            if isinstance(val, str) and val.strip():
                merged[k] = val.strip()
        merged["assist_mode"] = "llm_assisted"
        return merged

    return fallback_pico


def interpret_cohort_profile(profile: dict, user_note: str = "", *, forbidden_values: tuple[str, ...] = ()) -> dict:
    """Interpret a privacy-safe CohortProfile into clinical domain hints & ClinicalTrials.gov search terms."""
    safe_summary = sanitize_profile_for_llm(profile, user_note=user_note)

    # Deterministic keyword extraction from column names, safe treatment/outcome labels, and user note
    corpus_parts = [safe_summary.get("user_domain_note", "")]
    for col in safe_summary["columns"]:
        corpus_parts.append(col["name"].replace("_", " "))
        for cat in col.get("distinct_categories", []):
            corpus_parts.append(cat.replace("_", " "))
    det = _extract_deterministic_terms(" ".join(corpus_parts))

    trt_candidates = profile.get("candidate_treatment_columns", [])
    arm_terms: list[str] = []
    if trt_candidates:
        for cat in trt_candidates[0].get("categories", [])[:2]:
            lbl = str(cat.get("value", "")).replace("_Plus_", " ").replace("_", " ").strip()
            if lbl and not ID_VALUE_RE.match(lbl):
                arm_terms.append(lbl)

    search_tokens = list(
        dict.fromkeys(
            det["interventions_en"][:2]
            + det["conditions_en"][:2]
            + [t.split()[0] for t in arm_terms if t and t[0].isascii()][:2]
        )
    )
    fallback_query = " ".join(search_tokens).strip() or " ".join(arm_terms[:2]).strip() or "Randomized Controlled Trial"

    domain_summary_zh = (
        f"识别到 {'、'.join(det['conditions_zh']) or '临床观察性专病队列'} 特征；"
        f"候选对比组包含 {' vs '.join(arm_terms[:2]) if arm_terms else '待指定暴露分组'}。"
    )
    fallback_res = {
        "domain_summary_zh": domain_summary_zh,
        "condition_query_en": " ".join(det["conditions_en"][:2]),
        "intervention_query_en": " ".join(det["interventions_en"][:2]) or (arm_terms[0] if arm_terms else ""),
        "search_query_en": fallback_query,
        "sanitized_llm_payload": safe_summary,
        "assist_mode": "deterministic_rules",
    }

    llm_out = _call_chat_json(
        (
            "You are a clinical biostatistician assisting Target Trial Emulation (TTE). "
            "Given ONLY schema-level summary statistics (never patient rows), infer the clinical disease/drug domain "
            "in Chinese and produce English search keywords for ClinicalTrials.gov API v2. "
            "Return JSON with keys: domain_summary_zh, condition_query_en, intervention_query_en, search_query_en."
        ),
        safe_summary,
        forbidden_values=forbidden_values,
    )
    if isinstance(llm_out, dict) and llm_out.get("search_query_en"):
        for k in ("domain_summary_zh", "condition_query_en", "intervention_query_en", "search_query_en"):
            val = llm_out.get(k)
            if isinstance(val, str) and val.strip():
                fallback_res[k] = val.strip()
        fallback_res["assist_mode"] = "llm_assisted"

    return fallback_res


def rerank_and_explain_rcts(candidates: list[dict], context_summary: dict) -> tuple[list[dict], str]:
    """Optionally refine candidate RCT ranking and Chinese match rationale via LLM, falling back to deterministic scores."""
    if not candidates:
        return [], get_llm_status()["mode"]

    # Ensure context_summary contains no raw rows
    compact_candidates = [
        {
            "nct_id": c["nct_id"],
            "brief_title": c["brief_title"],
            "phases": c.get("phases", []),
            "enrollment_count": c.get("enrollment_count"),
            "conditions": c.get("conditions", [])[:5],
            "experimental_arms": c.get("experimental_arms", [])[:3],
            "comparator_arms": c.get("comparator_arms", [])[:3],
            "primary_outcomes": c.get("primary_outcomes", [])[:2],
        }
        for c in candidates[:8]
    ]
    llm_out = _call_chat_json(
        (
            "You are a Target Trial Emulation trialist. Rank the candidate ClinicalTrials.gov RCTs against the "
            "study context and provide a concise Chinese match rationale for each trial. "
            "Return JSON: {\"rankings\": [{\"nct_id\": \"NCT...\", \"score\": 0-99, \"reason_zh\": \"...\"}]}"
        ),
        {"study_context": context_summary, "candidates": compact_candidates},
    )
    if isinstance(llm_out, dict) and isinstance(llm_out.get("rankings"), list):
        by_nct = {str(item.get("nct_id")): item for item in llm_out["rankings"] if isinstance(item, dict)}
        enriched = []
        for cand in candidates:
            item = dict(cand)
            hit = by_nct.get(item["nct_id"])
            if hit:
                if isinstance(hit.get("score"), (int, float)):
                    item["match_score"] = max(1, min(99, int(round(float(hit["score"])))))
                if isinstance(hit.get("reason_zh"), str) and hit["reason_zh"].strip():
                    item["match_reasons"] = [hit["reason_zh"].strip(), *item.get("match_reasons", [])[:1]]
                item["ranking_mode"] = "llm_assisted"
            enriched.append(item)
        enriched.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        return enriched, "llm_assisted"

    return candidates, "deterministic_rules"
