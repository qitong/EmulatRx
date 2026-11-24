"""Privacy-guarded OpenAI-compatible LLM helper (OpenAI / DeepSeek / DashScope Bailian) with strict whitelist & 100% deterministic fallback.

STRICT PRIVACY & SECRET INVARIANTS:
1. Secret priority:
   (a) File at `EMULATRX_LLM_API_KEY_FILE` (or default `/run/secrets/llm_api_key` if present and non-empty);
   (b) `EMULATRX_LLM_API_KEY`;
   (c) `OPENAI_API_KEY`.
   Missing or empty secret files NEVER raise on startup; the service degrades cleanly to deterministic rules.
   API keys are NEVER included in status dicts, logs, error messages, or audit artifacts.
2. Cohort & note privacy (Closed Whitelist + Anonymous Placeholders):
   Any outbound LLM payload constructed from a `user_cohort` session MUST pass through
   `sanitize_profile_for_llm`, `sanitize_study_context_for_llm`, and `assert_llm_payload_privacy_safe`:
   - Category values are ONLY sent as literal strings when they strictly match the closed clinical/status
     vocabulary whitelist AND have cell count >= 5; ALL other category strings (e.g. site names, doctor/patient
     names like 'Alice Smith', hospital codes, or rare categories) are replaced with structured anonymous
     placeholders (`CATEGORY_LEVEL_1`, `CATEGORY_LEVEL_2`, ...).
   - Free-text `user_note` in `user_cohort` mode is NEVER sent verbatim; ONLY canonical clinical terms matched
     against `BILINGUAL_CLINICAL_TERMS` / `SAFE_CONTROLLED_CLINICAL_WORDS` are extracted.
   - `assert_llm_payload_privacy_safe` enforces fail-closed blocking against session `forbidden_values`,
     subject ID patterns, person-name patterns, and standalone numeric identifiers (`>= 6` digits).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen as _std_urlopen

from ._common import ContractError


DEFAULT_LLM_SECRET_FILE = Path("/run/secrets/llm_api_key")
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_LLM_TIMEOUT = 15
MIN_K_ANONYMITY_CATEGORY_COUNT = 5

# Regexes matching identifier-like column names, subject IDs, person names, or numeric identifiers
ID_COL_RE = re.compile(
    r"(?:^|_)(?:id|pat|patient|subj|subject|case_no|case_id|mrn|hosp|record|ssn|uuid|name|phone|email|site_id)(?:$|_)",
    re.IGNORECASE,
)
ID_VALUE_RE = re.compile(
    r"^(?:[A-Za-z]{2,12}[-_][0-9]{3,}[-_0-9]*|[0-9]{5,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)
PERSON_NAME_RE = re.compile(r"\b[A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20}\b")
STANDALONE_LONG_DIGITS_RE = re.compile(r"(?<![A-Za-z0-9.])\d{6,}(?![0-9])")

# Bilingual clinical term lexicon for deterministic PICO, whitelist extraction & ClinicalTrials.gov search
BILINGUAL_CLINICAL_TERMS: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (("达格列净", "dapagliflozin", "sglt2", "sglt2i"), "Dapagliflozin", "intervention", "达格列净 (Dapagliflozin)"),
    (("恩格列净", "empagliflozin"), "Empagliflozin", "intervention", "恩格列净 (Empagliflozin)"),
    (("卡格列净", "canagliflozin"), "Canagliflozin", "intervention", "卡格列净 (Canagliflozin)"),
    (("西格列汀", "sitagliptin", "dpp-4", "dpp4", "dpp4i"), "Sitagliptin", "comparator", "西格列汀 (Sitagliptin)"),
    (("利格列汀", "linagliptin"), "Linagliptin", "comparator", "利格列汀 (Linagliptin)"),
    (("司美格鲁肽", "semaglutide", "glp-1", "glp1"), "Semaglutide", "intervention", "司美格鲁肽 (Semaglutide)"),
    (("非奈利酮", "finerenone"), "Finerenone", "intervention", "非奈利酮 (Finerenone)"),
    (("帕博利珠单抗", "pembrolizumab", "keytruda", "pd-1", "pd1"), "Pembrolizumab", "intervention", "帕博利珠单抗 (Pembrolizumab)"),
    (("纳武利尤单抗", "nivolumab"), "Nivolumab", "intervention", "纳武利尤单抗 (Nivolumab)"),
    (("阿替利珠单抗", "atezolizumab"), "Atezolizumab", "intervention", "阿替利珠单抗 (Atezolizumab)"),
    (("奥希替尼", "osimertinib"), "Osimertinib", "intervention", "奥希替尼 (Osimertinib)"),
    (("含铂", "双药化疗", "platinum", "chemo", "chemotherapy", "pemetrexed", "培美曲塞"), "Platinum Chemotherapy", "comparator", "含铂双药化疗 (Platinum Doublet Chemo)"),
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

# Closed word-level whitelist for safe clinical/statistical category labels & column names
SAFE_CONTROLLED_CLINICAL_WORDS: frozenset[str] = frozenset(
    {
        # Coding & booleans
        "0", "1", "2", "3", "4", "yes", "no", "true", "false", "y", "n", "na", "none", "unknown", "missing",
        # Sex & demographics
        "male", "female", "m", "f", "adult", "elderly",
        # Trial arms & generic treatment roles
        "active", "control", "comparator", "reference", "placebo", "experimental", "standard", "care", "soc",
        "drug", "regimen", "therapy", "arm", "group", "plus", "doublet", "monotherapy", "combination",
        # Whitelisted drugs / classes
        "dapagliflozin", "empagliflozin", "canagliflozin", "sglt2", "sglt2i",
        "sitagliptin", "linagliptin", "dpp4", "dpp4i", "semaglutide", "glp1", "finerenone", "metformin", "insulin",
        "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "osimertinib", "platinum", "chemo", "chemotherapy",
        "pemetrexed", "carboplatin", "cisplatin", "paclitaxel",
        "apixaban", "rivaroxaban", "dabigatran", "edoxaban", "warfarin", "aspirin", "statin", "acei", "arb",
        # Whitelisted outcome & survival status words
        "censor", "censored", "alive", "dead", "death", "survive", "survived", "free",
        "event", "primary", "competing", "outcome", "status", "hf", "cv", "noncv", "cardiovascular",
        "progression", "cancer", "cancerdeath", "relapse", "recurrence", "treatment", "unrelated", "treatmentunrelated",
        "stroke", "embolism", "bleeding", "esrd", "renal", "kidney", "hospitalization",
        # Whitelisted staging / functional scales / smoking
        "nyha", "class", "i", "ii", "iii", "iv", "stage", "ia", "ib", "iia", "iib", "iiia", "iiib", "iva", "ivb",
        "ecog", "ps", "g1", "g2", "g3", "g3a", "g3b", "g4", "g5",
        "never", "former", "current", "smoker", "nonsmoker", "smoking",
        "low", "medium", "moderate", "high", "positive", "negative",
        # Common Chinese controlled status labels
        "是", "否", "男", "女", "删失", "存活", "死亡", "事件", "主要事件", "竞争事件", "竞争死亡", "进展", "复发", "对照组", "试验组", "安慰剂",
    }
)

ANON_CATEGORY_PLACEHOLDER_RE = re.compile(r"^CATEGORY_LEVEL_\d+$")
SUBJECT_ID_INLINE_RE = re.compile(r"\b(?:Hosp|ONC|PAT|MRN|SUBJ|SYN)[-_][0-9]{2,}[-_0-9]*\b", re.IGNORECASE)
STRUCTURAL_ENUM_PARENT_KEYS: frozenset[str] = frozenset({"privacy_contract", "dtype_kind", "inferred_role"})
CANONICAL_CONTROLLED_TERMS: frozenset[str] = frozenset(
    {term for _, en_term, _, zh_label in BILINGUAL_CLINICAL_TERMS for term in (en_term, zh_label)}
)


def is_whitelisted_clinical_category(label: str, count: int) -> bool:
    """Return True ONLY if `label` has >= MIN_K_ANONYMITY_CATEGORY_COUNT observations AND consists 100% of controlled clinical vocabulary."""
    if count < MIN_K_ANONYMITY_CATEGORY_COUNT:
        return False
    s = (label or "").strip()
    if not s or len(s) > 64:
        return False
    if "@" in s or re.search(r"\d{3,}", s) or ID_VALUE_RE.match(s) or PERSON_NAME_RE.search(s):
        return False
    if s in SAFE_CONTROLLED_CLINICAL_WORDS:
        return True
    # Split on underscores, hyphens, slashes, spaces, plus signs; require every token to be in SAFE_CONTROLLED_CLINICAL_WORDS
    parts = [p.lower() for p in re.split(r"[\s_\-/+]+", s) if p]
    if not parts:
        return False
    return all(part in SAFE_CONTROLLED_CLINICAL_WORDS for part in parts)


def extract_whitelisted_clinical_concepts(raw_text: str) -> dict:
    """Extract ONLY canonical clinical concepts from untrusted user notes/context, discarding all numbers, names, and raw prose."""
    text = (raw_text or "").strip()
    if not text:
        return {
            "conditions_en": [],
            "conditions_zh": [],
            "interventions_en": [],
            "interventions_zh": [],
            "comparators_en": [],
            "comparators_zh": [],
            "outcomes_zh": [],
            "whitelisted_keywords_en": [],
        }

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

    # Only allow ASCII words that explicitly appear in SAFE_CONTROLLED_CLINICAL_WORDS (blocks 'Alice', 'Smith', 'Patient', 'identifier', '123456789'!)
    generic_noise = {"0", "1", "2", "3", "4", "yes", "no", "true", "false", "y", "n", "m", "f", "i", "ii", "iii", "iv", "none", "unknown", "missing"}
    whitelisted_en: list[str] = []
    for tok in re.findall(r"\b[A-Za-z][A-Za-z0-9]{1,24}\b", text):
        tl = tok.lower()
        if tl in SAFE_CONTROLLED_CLINICAL_WORDS and tl not in generic_noise and tl not in whitelisted_en:
            whitelisted_en.append(tl)

    return {
        "conditions_en": conditions_en,
        "conditions_zh": conditions_zh,
        "interventions_en": interventions_en,
        "interventions_zh": interventions_zh,
        "comparators_en": comparators_en,
        "comparators_zh": comparators_zh,
        "outcomes_zh": outcomes_zh,
        "whitelisted_keywords_en": whitelisted_en[:8],
    }


def extract_untrusted_Note_forbidden_tokens(user_note: str) -> list[str]:
    """Extract digit strings (>=3 digits) and non-whitelisted multi-char words/phrases from `user_note` for fail-closed blocking."""
    note = (user_note or "").strip()
    if not note:
        return []
    out: list[str] = []
    # Capture multi-digit identifiers (e.g. '123456789')
    for d in re.findall(r"\d{3,}", note):
        if d not in out:
            out.append(d)
    # Capture capitalized person-name phrases (e.g. 'Alice Smith')
    for nm in PERSON_NAME_RE.findall(note):
        if nm not in out:
            out.append(nm)
    # Capture ID-like tokens (e.g. 'Hosp-2023-0001')
    for id_m in re.findall(r"\b[A-Za-z]{2,12}[-_][0-9]{2,}[-_0-9]*\b", note):
        if id_m not in out:
            out.append(id_m)
    return out


def urlopen_llm(request: Request, timeout: float):
    """Module-level transport wrapper so offline unit tests can intercept and audit outbound LLM payloads."""
    return _std_urlopen(request, timeout=timeout)


def normalize_llm_base_url(raw_url: str) -> str:
    """Normalize OpenAI / DeepSeek / DashScope (百炼) compatible gateway base URLs to '.../v1' (or explicit versioned root)."""
    cleaned = (raw_url or "").strip().rstrip("/")
    if not cleaned:
        return DEFAULT_LLM_BASE_URL
    if cleaned.endswith("/chat/completions"):
        cleaned = cleaned[: -len("/chat/completions")].rstrip("/")

    parsed = urlsplit(cleaned)
    path = parsed.path.rstrip("/")
    if not path or path.endswith("/compatible-mode") or path.endswith("/openai"):
        cleaned = f"{cleaned}/v1"
    return cleaned


def _read_optional_secret_file(path_candidate: Path) -> str:
    try:
        if path_candidate.is_file():
            val = path_candidate.read_text(encoding="utf-8").strip()
            if val and val.isprintable():
                return val
    except OSError:
        return ""
    return ""


def _resolve_api_key() -> tuple[str, str]:
    """Return (api_key, key_source) following strict ymadmin priority without ever raising on missing files."""
    explicit_file = os.environ.get("EMULATRX_LLM_API_KEY_FILE", "").strip()
    if explicit_file:
        val = _read_optional_secret_file(Path(explicit_file))
        if val:
            return val, "secret_file"
    else:
        val_default = _read_optional_secret_file(DEFAULT_LLM_SECRET_FILE)
        if val_default:
            return val_default, "default_secret_file"

    env_emulatrx = os.environ.get("EMULATRX_LLM_API_KEY", "").strip()
    if env_emulatrx:
        return env_emulatrx, "env_emulatrx"

    env_openai = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_openai:
        return env_openai, "env_openai"

    return "", "none"


def _resolve_llm_config() -> dict:
    api_key, key_source = _resolve_api_key()
    raw_base = (
        os.environ.get("EMULATRX_LLM_BASE_URL", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "").strip()
        or DEFAULT_LLM_BASE_URL
    )
    base_url = normalize_llm_base_url(raw_base)
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
        "key_source": key_source,
        "base_url": base_url,
        "model": model if enabled else None,
        "configured_model": model,
        "timeout": max(2.0, min(timeout, 45.0)),
    }


def get_llm_status() -> dict:
    """Return safe public status for the Web UI and MCP tools (NEVER exposes API keys or secret file contents)."""
    cfg = _resolve_llm_config()
    parsed_host = urlsplit(cfg["base_url"]).netloc or "openai-compatible"
    if cfg["enabled"]:
        return {
            "enabled": True,
            "mode": "llm_assisted",
            "key_source": cfg["key_source"],
            "gateway_host": parsed_host,
            "label_zh": f"AI 智能辅助已启用（{cfg['configured_model']} @ {parsed_host}）",
            "model": cfg["configured_model"],
            "privacy_policy_zh": "严格临床词表白名单与匿名占位符脱敏：仅外发受控词与数值分位数，未核验标签一律替换为 CATEGORY_LEVEL_*，0 行患者记录外发",
        }
    return {
        "enabled": False,
        "mode": "deterministic_rules",
        "key_source": "none",
        "gateway_host": None,
        "label_zh": "当前未挂载模型密钥，已启用确定性规则引擎兜底（本地规则引擎模式）",
        "model": None,
        "privacy_policy_zh": "全本地确定性规则运行：无任何外部大模型网络调用，个体患者记录 100% 留存本机独立会话目录",
    }


def parse_llm_json_content(raw_content: object) -> dict | None:
    """Parse JSON dict from OpenAI / DeepSeek-R1 / Qwen / Bailian responses, stripping <think> blocks and markdown fences."""
    if isinstance(raw_content, list):
        parts = [
            str(item.get("text", ""))
            for item in raw_content
            if isinstance(item, dict) and item.get("type") in (None, "text", "output_text")
        ]
        text = "\n".join(parts).strip()
    elif isinstance(raw_content, str):
        text = raw_content.strip()
    else:
        return None

    if not text:
        return None

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = []
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    candidates.append(text)

    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            continue
    return None


def _sanitize_column_name_for_llm(col_name: str, col_idx: int) -> str:
    raw = str(col_name or "").strip()
    if not raw or "@" in raw or re.search(r"\d{4,}", raw) or PERSON_NAME_RE.search(raw):
        return f"column_{col_idx + 1}"
    return raw[:64]


def sanitize_profile_for_llm(profile: dict, user_note: str = "") -> dict:
    """Build a strictly whitelisted aggregate schema summary with anonymous placeholders for non-whitelisted categories."""
    safe_columns: list[dict] = []
    for idx, col in enumerate(profile.get("columns", [])):
        col_name = str(col.get("name", ""))
        role = str(col.get("inferred_role", ""))
        if role == "identifier" or ID_COL_RE.search(col_name):
            continue
        entry: dict = {
            "name": _sanitize_column_name_for_llm(col_name, idx),
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
            category_levels: list[dict] = []
            for cat_idx, item in enumerate(col["top_categories"][:20]):
                lbl = str(item.get("value", "")).strip()
                cnt = int(item.get("count", 0))
                rate = round(float(item.get("rate", 0.0)), 4)
                if is_whitelisted_clinical_category(lbl, cnt):
                    safe_labels.append(lbl)
                    category_levels.append({"token": lbl, "rate": rate, "whitelisted": True})
                else:
                    placeholder = f"CATEGORY_LEVEL_{cat_idx + 1}"
                    safe_labels.append(placeholder)
                    category_levels.append({"token": placeholder, "rate": rate, "whitelisted": False})
            entry["distinct_categories"] = safe_labels
            entry["category_levels"] = category_levels
        safe_columns.append(entry)

    # Strict whitelist extraction on user_note: NEVER pass raw user_note text to external LLM!
    note_concepts = extract_whitelisted_clinical_concepts(user_note)
    whitelisted_note_terms = list(
        dict.fromkeys(
            note_concepts["conditions_en"]
            + note_concepts["interventions_en"]
            + note_concepts["comparators_en"]
            + note_concepts["whitelisted_keywords_en"]
        )
    )
    return {
        "privacy_contract": "schema_and_aggregates_only_zero_individual_rows",
        "row_count": int(profile.get("row_count", 0)),
        "column_count": int(profile.get("column_count", 0)),
        "columns": safe_columns,
        "whitelisted_domain_concepts": whitelisted_note_terms,
    }


def sanitize_study_context_for_llm(context_summary: dict | None) -> dict:
    """Whitelist-only sanitizer for study_context passed into candidate RCT reranking."""
    if not isinstance(context_summary, dict):
        return {"whitelisted_concepts": []}
    combined = " ".join(
        str(context_summary.get(k, ""))
        for k in (
            "query",
            "condition",
            "intervention",
            "condition_query_en",
            "intervention_query_en",
            "search_query_en",
            "domain_summary_zh",
            "population",
            "comparator",
            "primary_outcome",
        )
    )
    concepts = extract_whitelisted_clinical_concepts(combined)
    return {
        "conditions_en": concepts["conditions_en"],
        "interventions_en": concepts["interventions_en"],
        "comparators_en": concepts["comparators_en"],
        "outcomes_zh": concepts["outcomes_zh"],
        "whitelisted_keywords_en": concepts["whitelisted_keywords_en"],
    }


def _extract_cohort_context_leaves(
    node: object,
    *,
    parent_key: str | None = None,
    in_column_entry: bool = False,
) -> tuple[list[str], list[str]]:
    """Extract (value_leaves, column_name_leaves) from structured `cohort_outbound_context`, ignoring JSON keys and static code enums."""
    value_leaves: list[str] = []
    column_name_leaves: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            key_str = str(k)
            if key_str == "columns" and isinstance(v, list):
                for col_item in v:
                    vl, cl = _extract_cohort_context_leaves(col_item, parent_key="columns", in_column_entry=True)
                    value_leaves.extend(vl)
                    column_name_leaves.extend(cl)
            else:
                vl, cl = _extract_cohort_context_leaves(v, parent_key=key_str, in_column_entry=in_column_entry)
                value_leaves.extend(vl)
                column_name_leaves.extend(cl)
    elif isinstance(node, (list, tuple)):
        for item in node:
            vl, cl = _extract_cohort_context_leaves(item, parent_key=parent_key, in_column_entry=in_column_entry)
            value_leaves.extend(vl)
            column_name_leaves.extend(cl)
    elif isinstance(node, str):
        if parent_key in STRUCTURAL_ENUM_PARENT_KEYS:
            return value_leaves, column_name_leaves
        if in_column_entry and parent_key == "name":
            column_name_leaves.append(node)
        else:
            value_leaves.append(node)
    return value_leaves, column_name_leaves


def assert_llm_payload_privacy_safe(
    cohort_outbound_context: dict | list | str | bytes,
    forbidden_values: list[str] | tuple[str, ...] = (),
) -> None:
    """Fail-closed inspector scoped exclusively to `cohort_outbound_context` non-column-name leaf values and PII patterns.

    System prompt templates (Layer A), fixed JSON schema keys (Layer A), column names, and public external
    ClinicalTrials.gov trial records (Layer B) are never matched against `forbidden_values`, eliminating
    false positives when a cohort column contains common English category values like 'patient', 'claims', or 'study'.
    """
    target_node: object = cohort_outbound_context
    if isinstance(cohort_outbound_context, (bytes, bytearray)):
        raw_text = bytes(cohort_outbound_context).decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw_text)
        except Exception:
            parsed = raw_text
        if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list) and len(parsed["messages"]) > 1:
            user_content = str(parsed["messages"][1].get("content", ""))
            try:
                user_json = json.loads(user_content)
                target_node = (
                    user_json["cohort_outbound_context"]
                    if isinstance(user_json, dict) and "cohort_outbound_context" in user_json
                    else user_json
                )
            except Exception:
                target_node = user_content
        else:
            target_node = parsed

    if (
        isinstance(target_node, dict)
        and "cohort_outbound_context" in target_node
        and set(target_node.keys()) <= {"cohort_outbound_context", "public_reference_context"}
    ):
        target_node = target_node["cohort_outbound_context"]

    if isinstance(target_node, str):
        value_leaves, column_name_leaves = [target_node], []
    else:
        value_leaves, column_name_leaves = _extract_cohort_context_leaves(target_node)

    # 1. Check PII / subject ID / standalone >=6-digit numeric identifiers across both value leaves and column names
    for leaf in [*value_leaves, *column_name_leaves]:
        id_hit = SUBJECT_ID_INLINE_RE.search(leaf)
        if id_hit:
            raise ContractError(
                "PRIVACY_VIOLATION_BLOCKED",
                f"Blocked outbound LLM context containing subject identifier pattern '{id_hit.group(0)}'.",
            )
        digit_hit = STANDALONE_LONG_DIGITS_RE.search(leaf)
        if digit_hit:
            raise ContractError(
                "PRIVACY_VIOLATION_BLOCKED",
                f"Blocked outbound LLM context containing numeric identifier '{digit_hit.group(0)}'.",
            )

    # 2. Check session `forbidden_values` strictly on non-column-name value leaves of `cohort_outbound_context`
    cleaned_forbidden: list[tuple[str, str, re.Pattern[str] | None]] = []
    for token in forbidden_values:
        tok_str = str(token).strip()
        if not tok_str or len(tok_str) < 3:
            continue
        tok_lower = tok_str.lower()
        if tok_lower in SAFE_CONTROLLED_CLINICAL_WORDS or tok_str in CANONICAL_CONTROLLED_TERMS:
            continue
        word_pat = (
            re.compile(rf"(?<![A-Za-z0-9]){re.escape(tok_lower)}(?![A-Za-z0-9])")
            if tok_str.isascii() and tok_str.isalpha()
            else None
        )
        cleaned_forbidden.append((tok_str, tok_lower, word_pat))

    if not cleaned_forbidden:
        return

    for leaf in value_leaves:
        s = leaf.strip()
        if (
            not s
            or ANON_CATEGORY_PLACEHOLDER_RE.fullmatch(s)
            or s.lower() in SAFE_CONTROLLED_CLINICAL_WORDS
            or s in CANONICAL_CONTROLLED_TERMS
        ):
            continue
        s_lower = s.lower()
        for tok_str, tok_lower, word_pat in cleaned_forbidden:
            matched = (
                s_lower == tok_lower
                or (word_pat is not None and bool(word_pat.search(s_lower)))
                or (word_pat is None and (tok_str in s or (len(tok_str) >= 4 and tok_lower in s_lower)))
            )
            if matched:
                raise ContractError(
                    "PRIVACY_VIOLATION_BLOCKED",
                    "Blocked outbound LLM cohort context containing an unverified cohort value or note identifier.",
                )


def _is_reasoning_model(model_name: str) -> bool:
    ml = (model_name or "").lower()
    return any(tok in ml for tok in ("reasoner", "deepseek-r1", "-r1", "qwq", "o1-", "o3-"))


def build_verified_llm_messages(
    system_prompt: str,
    *,
    cohort_outbound_context: dict,
    public_reference_context: dict | None = None,
    forbidden_values: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    """Assemble OpenAI-compatible `messages` from three strictly isolated payload layers:

    - Layer A (`system_prompt` & fixed JSON envelope keys): static code templates, zero user data.
    - Layer B (`public_reference_context`): public external ClinicalTrials.gov records, zero user data.
    - Layer C (`cohort_outbound_context`): sanitized cohort/study context verified by `assert_llm_payload_privacy_safe`.
    """
    if not isinstance(cohort_outbound_context, dict):
        raise ContractError("INVALID_LLM_CONTEXT", "cohort_outbound_context must be a JSON object")

    assert_llm_payload_privacy_safe(cohort_outbound_context, forbidden_values=forbidden_values)

    json_instruction = (
        f"{system_prompt}\n"
        "IMPORTANT: Output ONLY a single valid JSON object. Do not wrap in extra commentary."
    )
    user_envelope: dict[str, dict] = {"cohort_outbound_context": cohort_outbound_context}
    if public_reference_context is not None:
        if not isinstance(public_reference_context, dict):
            raise ContractError("INVALID_LLM_CONTEXT", "public_reference_context must be a JSON object")
        user_envelope["public_reference_context"] = public_reference_context

    return [
        {"role": "system", "content": json_instruction},
        {"role": "user", "content": json.dumps(user_envelope, ensure_ascii=False)},
    ]


def _call_chat_json(
    system_prompt: str,
    *,
    cohort_outbound_context: dict,
    public_reference_context: dict | None = None,
    forbidden_values: tuple[str, ...] = (),
) -> dict | None:
    """Invoke OpenAI / DeepSeek / DashScope compatible `/chat/completions` with 3-layer payload separation."""
    cfg = _resolve_llm_config()
    if not cfg["enabled"]:
        return None

    model_name = cfg["configured_model"]
    messages = build_verified_llm_messages(
        system_prompt,
        cohort_outbound_context=cohort_outbound_context,
        public_reference_context=public_reference_context,
        forbidden_values=forbidden_values,
    )
    body_obj: dict = {
        "model": model_name,
        "messages": messages,
    }
    if not _is_reasoning_model(model_name):
        body_obj["temperature"] = 0.1

    raw_body = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")

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
            resp_bytes = resp.read(512 * 1024)
        parsed = json.loads(resp_bytes.decode("utf-8"))
        message = parsed["choices"][0]["message"]
        return parse_llm_json_content(message.get("content"))
    except ContractError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError, KeyError, IndexError, ValueError, TypeError):
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

    ascii_tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9-]{3,24}\b", text)
    stop = {
        "with", "from", "that", "this", "plus", "versus", "study", "trial", "cohort",
        "years", "months", "days", "event", "status", "censored", "alive", "first",
        "line", "regimen", "initiated", "drug", "prior", "history", "class", "category", "level",
    }
    extra_en = []
    for tok in ascii_tokens:
        if tok.lower() not in stop and not tok.startswith("CATEGORY_LEVEL_") and tok not in extra_en:
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
    """Parse free-text clinical research idea (no-data simulated mode) into structured PICO + search terms."""
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
        or "Randomized Controlled Trial",
        "assist_mode": "deterministic_rules",
    }

    # Scrub any accidental standalone >=6 digit or subject IDs even in free-text research idea before LLM call
    scrubbed_idea = STANDALONE_LONG_DIGITS_RE.sub("[REDACTED_NUM]", clean_idea[:1500])
    scrubbed_idea = SUBJECT_ID_INLINE_RE.sub("[REDACTED_ID]", scrubbed_idea)
    llm_out = _call_chat_json(
        (
            "You are a clinical epidemiologist specializing in Target Trial Emulation (TTE). "
            "Extract structured PICO in Chinese and concise English search terms for ClinicalTrials.gov API v2. "
            "Return JSON with keys: population, intervention, comparator, primary_outcome, competing_event, "
            "condition_query_en, intervention_query_en, search_query_en."
        ),
        cohort_outbound_context={"idea_text": scrubbed_idea},
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
    combined_forbidden = tuple(dict.fromkeys([*forbidden_values, *extract_untrusted_Note_forbidden_tokens(user_note)]))

    corpus_parts = list(safe_summary.get("whitelisted_domain_concepts", []))
    for col in safe_summary["columns"]:
        corpus_parts.append(col["name"].replace("_", " "))
        for cat in col.get("distinct_categories", []):
            if not str(cat).startswith("CATEGORY_LEVEL_"):
                corpus_parts.append(str(cat).replace("_", " "))
    det = _extract_deterministic_terms(" ".join(corpus_parts))

    trt_candidates = profile.get("candidate_treatment_columns", [])
    whitelisted_arm_terms: list[str] = []
    local_display_arms: list[str] = []
    if trt_candidates:
        for cat in trt_candidates[0].get("categories", [])[:2]:
            raw_lbl = str(cat.get("value", "")).strip()
            cnt = int(cat.get("count", 0))
            if raw_lbl:
                local_display_arms.append(raw_lbl)
            if is_whitelisted_clinical_category(raw_lbl, cnt):
                whitelisted_arm_terms.append(raw_lbl.replace("_Plus_", " ").replace("_", " ").strip())

    search_tokens = list(
        dict.fromkeys(
            det["interventions_en"][:2]
            + det["conditions_en"][:2]
            + [t.split()[0] for t in whitelisted_arm_terms if t and t[0].isascii()][:2]
        )
    )
    fallback_query = " ".join(search_tokens).strip() or "Randomized Controlled Trial"

    domain_summary_zh = (
        f"识别到 {'、'.join(det['conditions_zh']) or '临床观察性专病队列'} 特征；"
        f"候选对比组包含 {' vs '.join(local_display_arms[:2]) if local_display_arms else '待指定暴露分组'}。"
    )
    fallback_res = {
        "domain_summary_zh": domain_summary_zh,
        "condition_query_en": " ".join(det["conditions_en"][:2]),
        "intervention_query_en": " ".join(det["interventions_en"][:2]) or (whitelisted_arm_terms[0] if whitelisted_arm_terms else ""),
        "search_query_en": fallback_query,
        "sanitized_llm_payload": safe_summary,
        "assist_mode": "deterministic_rules",
    }

    llm_out = _call_chat_json(
        (
            "You are a clinical biostatistician assisting Target Trial Emulation (TTE). "
            "Given ONLY whitelisted schema-level summary statistics and anonymous CATEGORY_LEVEL_* placeholders "
            "inside cohort_outbound_context (never patient rows), infer the clinical disease/drug domain in Chinese "
            "and produce English search keywords for ClinicalTrials.gov API v2. "
            "Return JSON with keys: domain_summary_zh, condition_query_en, intervention_query_en, search_query_en."
        ),
        cohort_outbound_context=safe_summary,
        forbidden_values=combined_forbidden,
    )
    if isinstance(llm_out, dict) and llm_out.get("search_query_en"):
        for k in ("domain_summary_zh", "condition_query_en", "intervention_query_en", "search_query_en"):
            val = llm_out.get(k)
            if isinstance(val, str) and val.strip():
                fallback_res[k] = val.strip()
        fallback_res["assist_mode"] = "llm_assisted"

    return fallback_res


def rerank_and_explain_rcts(
    candidates: list[dict],
    context_summary: dict | None,
    *,
    forbidden_values: tuple[str, ...] = (),
) -> tuple[list[dict], str]:
    """Optionally refine candidate RCT ranking and Chinese match rationale via LLM using 3-layer separated context."""
    if not candidates:
        return [], get_llm_status()["mode"]

    safe_context = sanitize_study_context_for_llm(context_summary)
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
            "You are a Target Trial Emulation trialist. Rank the candidate ClinicalTrials.gov RCTs in "
            "public_reference_context against the whitelisted clinical concepts in cohort_outbound_context "
            "and provide a concise Chinese match rationale for each trial. "
            "Return JSON: {\"rankings\": [{\"nct_id\": \"NCT...\", \"score\": 0-99, \"reason_zh\": \"...\"}]}"
        ),
        cohort_outbound_context={"study_context": safe_context},
        public_reference_context={"candidates": compact_candidates},
        forbidden_values=forbidden_values,
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
