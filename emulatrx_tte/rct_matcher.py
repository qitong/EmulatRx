"""Live ClinicalTrials.gov API v2 target RCT retrieval, relevance scoring, and explicit offline curated fallback."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen as _std_urlopen

from ._common import ContractError
from .llm_assist import rerank_and_explain_rcts


NCT_RE = re.compile(r"\b(NCT[0-9]{8})\b", re.IGNORECASE)
API_ROOT = "https://clinicaltrials.gov/api/v2"
MAX_CT_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_CT_TIMEOUT = 10

STUDY_FIELDS = (
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.identificationModule.officialTitle",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.designModule.studyType",
    "protocolSection.designModule.phases",
    "protocolSection.designModule.enrollmentInfo",
    "protocolSection.designModule.designInfo",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.armsInterventionsModule.armGroups",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.eligibilityModule",
    "protocolSection.outcomesModule.primaryOutcomes",
    "hasResults",
)


# Representative public RCT metadata catalog used ONLY when ClinicalTrials.gov is unreachable/offline.
# Every entry returned from this catalog is explicitly labeled source="offline_curated_catalog_fallback".
OFFLINE_CURATED_RCTS: tuple[dict, ...] = (
    {
        "nct_id": "NCT03036124",
        "brief_title": "Study to Evaluate the Effect of Dapagliflozin on the Incidence of Worsening Heart Failure or Cardiovascular Death in Patients With Chronic Heart Failure (DAPA-HF)",
        "official_title": "Effect of Dapagliflozin on the Incidence of Worsening Heart Failure or Cardiovascular Death in Patients With Chronic Heart Failure With Reduced Ejection Fraction",
        "phases": ["PHASE3"],
        "overall_status": "COMPLETED",
        "enrollment_count": 4744,
        "allocation": "RANDOMIZED",
        "conditions": ["Heart Failure", "Type 2 Diabetes Mellitus", "Cardiovascular Diseases"],
        "experimental_arms": ["Dapagliflozin 10 mg once daily"],
        "comparator_arms": ["Matching Placebo / Standard Oral Antidiabetic Care"],
        "interventions": ["Drug: Dapagliflozin", "Drug: Placebo"],
        "eligibility_summary": {
            "min_age": "18 Years",
            "max_age": "N/A",
            "sex": "ALL",
            "highlights": [
                "确诊慢性心力衰竭（NYHA II–IV 级）合并或不合并 2 型糖尿病",
                "排除基线估算肾小球滤过率 eGFR < 30 mL/min/1.73m² 患者",
                "排除 1 型糖尿病或近期急性失代偿低血压患者",
            ],
        },
        "primary_outcomes": [
            {
                "measure": "Time to the First Occurrence of Any of the Components of the Composite: Worsening Heart Failure Event (Hospitalization or Urgent Heart Failure Visit) or Cardiovascular Death",
                "timeFrame": "Up to approximately 18 months",
            }
        ],
        "parsed_horizon_days": [90.0, 180.0, 365.0],
        "parsed_admin_end_day": 540.0,
    },
    {
        "nct_id": "NCT01730534",
        "brief_title": "Multicenter Trial to Evaluate the Effect of Dapagliflozin on the Incidence of Cardiovascular Events (DECLARE-TIMI 58)",
        "official_title": "Dapagliflozin Effect on Cardiovascular Events — A Multicenter, Randomized, Double-Blind, Placebo-Controlled Trial in Patients With Type 2 Diabetes",
        "phases": ["PHASE3"],
        "overall_status": "COMPLETED",
        "enrollment_count": 17160,
        "allocation": "RANDOMIZED",
        "conditions": ["Type 2 Diabetes", "Cardiovascular Risk", "Heart Failure"],
        "experimental_arms": ["Dapagliflozin 10 mg daily"],
        "comparator_arms": ["Comparator Glucose-Lowering Therapy (incl. DPP-4i / Standard Care)"],
        "interventions": ["Drug: Dapagliflozin", "Drug: Placebo"],
        "eligibility_summary": {
            "min_age": "40 Years",
            "max_age": "N/A",
            "sex": "ALL",
            "highlights": [
                "年龄 ≥ 40 岁且确诊 2 型糖尿病（HbA1c 6.5%–12.0%）",
                "合并动脉粥样硬化性心血管疾病或多重心血管高危因素",
                "排除肌酐清除率 CrCl < 60 mL/min 或活动性恶性肿瘤",
            ],
        },
        "primary_outcomes": [
            {
                "measure": "Composite of Cardiovascular Death or Hospitalization for Heart Failure",
                "timeFrame": "Up to 4.2 years (median follow-up)",
            }
        ],
        "parsed_horizon_days": [180.0, 365.0, 540.0],
        "parsed_admin_end_day": 730.0,
    },
    {
        "nct_id": "NCT03594110",
        "brief_title": "The Study of Heart and Kidney Protection With Empagliflozin (EMPA-KIDNEY)",
        "official_title": "A Multicentre International Randomized Parallel Group Double-blind Placebo-controlled Clinical Trial of Empagliflozin Once Daily in Patients With Chronic Kidney Disease",
        "phases": ["PHASE3"],
        "overall_status": "COMPLETED",
        "enrollment_count": 6609,
        "allocation": "RANDOMIZED",
        "conditions": ["Chronic Kidney Disease", "Diabetic Nephropathy", "Heart Failure"],
        "experimental_arms": ["SGLT2 Inhibitor (Empagliflozin 10 mg)"],
        "comparator_arms": ["Matching Placebo + Background RAS Blockade"],
        "interventions": ["Drug: Empagliflozin", "Drug: Placebo"],
        "eligibility_summary": {
            "min_age": "18 Years",
            "max_age": "N/A",
            "sex": "ALL",
            "highlights": [
                "慢性肾脏病成人患者（eGFR ≥ 20 且 < 45，或 eGFR 45–90 伴蛋白尿）",
                "排除多囊肾或既往肾移植患者",
            ],
        },
        "primary_outcomes": [
            {
                "measure": "Time to First Occurrence of Kidney Disease Progression or Cardiovascular Death",
                "timeFrame": "Approximately 24 months",
            }
        ],
        "parsed_horizon_days": [90.0, 180.0, 365.0],
        "parsed_admin_end_day": 720.0,
    },
    {
        "nct_id": "NCT02578680",
        "brief_title": "Study of Pemetrexed+Platinum Chemotherapy With or Without Pembrolizumab (MK-3475) in Participants With First Line Metastatic Nonsquamous Non-small Cell Lung Cancer (KEYNOTE-189)",
        "official_title": "A Randomized, Double-Blind, Phase III Study of Platinum+Pemetrexed Chemotherapy With or Without Pembrolizumab in First-Line Metastatic Nonsquamous NSCLC",
        "phases": ["PHASE3"],
        "overall_status": "COMPLETED",
        "enrollment_count": 616,
        "allocation": "RANDOMIZED",
        "conditions": ["Non-Small Cell Lung Cancer", "Metastatic Nonsquamous NSCLC"],
        "experimental_arms": ["Pembrolizumab + Pemetrexed + Platinum Chemotherapy"],
        "comparator_arms": ["Saline Placebo + Pemetrexed + Platinum Doublet Chemotherapy"],
        "interventions": ["Biological: Pembrolizumab", "Drug: Pemetrexed", "Drug: Cisplatin / Carboplatin"],
        "eligibility_summary": {
            "min_age": "18 Years",
            "max_age": "N/A",
            "sex": "ALL",
            "highlights": [
                "组织学确诊 IV 期非鳞状非小细胞肺癌（NSCLC），既往未接受晚期系统化疗",
                "基线 ECOG 体力状态评分 0 或 1 分（排除 ECOG ≥ 2）",
                "无 EGFR 敏感突变或 ALK 重排",
            ],
        },
        "primary_outcomes": [
            {
                "measure": "Overall Survival (OS) and Progression-Free Survival (PFS) per RECIST 1.1",
                "timeFrame": "Up to approximately 24 months",
            }
        ],
        "parsed_horizon_days": [90.0, 180.0, 365.0],
        "parsed_admin_end_day": 720.0,
    },
    {
        "nct_id": "NCT02775435",
        "brief_title": "Study of Carboplatin-Paclitaxel/Nab-Paclitaxel Chemotherapy With or Without Pembrolizumab in Adults With First-line Metastatic Squamous Non-small Cell Lung Cancer (KEYNOTE-407)",
        "official_title": "A Randomized, Double-Blind, Phase III Study of Carboplatin-Paclitaxel/Nab-Paclitaxel With or Without Pembrolizumab in First-line Metastatic Squamous NSCLC",
        "phases": ["PHASE3"],
        "overall_status": "COMPLETED",
        "enrollment_count": 559,
        "allocation": "RANDOMIZED",
        "conditions": ["Squamous Non-Small Cell Lung Cancer", "NSCLC"],
        "experimental_arms": ["Pembrolizumab + Carboplatin + Paclitaxel/Nab-Paclitaxel"],
        "comparator_arms": ["Placebo + Carboplatin + Paclitaxel/Nab-Paclitaxel"],
        "interventions": ["Biological: Pembrolizumab", "Drug: Carboplatin", "Drug: Paclitaxel"],
        "eligibility_summary": {
            "min_age": "18 Years",
            "max_age": "N/A",
            "sex": "ALL",
            "highlights": [
                "IV 期鳞状非小细胞肺癌初治成人患者，ECOG 0–1 分",
                "提供肿瘤组织用于 PD-L1 TPS 生物标志物测定",
            ],
        },
        "primary_outcomes": [
            {
                "measure": "Progression-Free Survival (PFS) and Overall Survival (OS)",
                "timeFrame": "Up to 24 months",
            }
        ],
        "parsed_horizon_days": [90.0, 180.0, 365.0],
        "parsed_admin_end_day": 720.0,
    },
    {
        "nct_id": "NCT00412984",
        "brief_title": "Apixaban for Reduction in Stroke and Other Thromboembolic Events in Atrial Fibrillation (ARISTOTLE)",
        "official_title": "A Phase 3, Active (Warfarin) Controlled, Randomized, Double-Blind, Parallel Arm Study to Evaluate Efficacy and Safety of Apixaban in Preventing Stroke and Systemic Embolism in Subjects With Nonvalvular Atrial Fibrillation",
        "phases": ["PHASE3"],
        "overall_status": "COMPLETED",
        "enrollment_count": 18201,
        "allocation": "RANDOMIZED",
        "conditions": ["Atrial Fibrillation", "Stroke", "Systemic Embolism"],
        "experimental_arms": ["Apixaban 5 mg (or 2.5 mg) twice daily"],
        "comparator_arms": ["Dose-adjusted Warfarin (target INR 2.0–3.0)"],
        "interventions": ["Drug: Apixaban", "Drug: Warfarin"],
        "eligibility_summary": {
            "min_age": "18 Years",
            "max_age": "N/A",
            "sex": "ALL",
            "highlights": [
                "非瓣膜性心房颤动或心房扑动伴至少 1 项卒中危险因素",
                "排除严重肾功能不全（肌酐清除率 < 25 mL/min）或活动性出血患者",
            ],
        },
        "primary_outcomes": [
            {
                "measure": "Time to First Occurrence of Stroke (Ischemic or Hemorrhagic) or Systemic Embolism",
                "timeFrame": "Median 1.8 years (up to 24 months)",
            }
        ],
        "parsed_horizon_days": [90.0, 180.0, 365.0],
        "parsed_admin_end_day": 660.0,
    },
    {
        "nct_id": "NCT02540993",
        "brief_title": "Efficacy and Safety of Finerenone in Subjects With Type 2 Diabetes Mellitus and Diabetic Kidney Disease (FIDELIO-DKD)",
        "official_title": "A Randomized, Double-Blind, Placebo-Controlled, Parallel-Group, Multicenter, Event-Driven Phase III Study to Investigate Efficacy and Safety of Finerenone in Diabetic Kidney Disease",
        "phases": ["PHASE3"],
        "overall_status": "COMPLETED",
        "enrollment_count": 5734,
        "allocation": "RANDOMIZED",
        "conditions": ["Diabetic Kidney Disease", "Type 2 Diabetes", "Chronic Kidney Disease"],
        "experimental_arms": ["Finerenone 10 mg or 20 mg once daily"],
        "comparator_arms": ["Placebo + Maximally Tolerated ACEi/ARB"],
        "interventions": ["Drug: Finerenone", "Drug: Placebo"],
        "eligibility_summary": {
            "min_age": "18 Years",
            "max_age": "N/A",
            "sex": "ALL",
            "highlights": [
                "2 型糖尿病合并慢性肾脏病（eGFR ≥ 25 mL/min/1.73m²）且已接受 RAS 阻滞剂",
                "基线血钾 ≤ 4.8 mmol/L",
            ],
        },
        "primary_outcomes": [
            {
                "measure": "Time to First Occurrence of Kidney Failure, Sustained Decrease of eGFR >= 40%, or Renal Death",
                "timeFrame": "Median 2.6 years",
            }
        ],
        "parsed_horizon_days": [180.0, 365.0, 540.0],
        "parsed_admin_end_day": 780.0,
    },
)


def urlopen_ctgov(request: Request, timeout: float):
    """Module-level transport wrapper so unit tests can mock ClinicalTrials.gov HTTP calls deterministically."""
    return _std_urlopen(request, timeout=timeout)


def parse_timeframe_to_horizons(timeframe_text: str) -> tuple[list[float], float]:
    """Convert ClinicalTrials.gov primary outcome TimeFrame string into (horizon_days, admin_end_day)."""
    text = (timeframe_text or "").lower()
    total_days: float | None = None

    year_matches = [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:year|yr|年)", text)]
    month_matches = [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:month|mo|个月|月)", text)]
    week_matches = [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:week|wk|周)", text)]
    day_matches = [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:day|天|日)", text)]

    if year_matches:
        total_days = max(year_matches) * 365.0
    elif month_matches:
        total_days = max(month_matches) * 30.4375
    elif week_matches:
        total_days = max(week_matches) * 7.0
    elif day_matches:
        total_days = max(day_matches)

    if total_days is None or total_days < 30.0:
        return [90.0, 180.0, 365.0], 540.0

    capped = min(max(total_days, 45.0), 1095.0)
    if capped >= 540.0:
        return [90.0, 180.0, 365.0], round(max(capped, 540.0), 1)
    if capped >= 300.0:
        return [90.0, 180.0, 270.0], round(max(capped, 365.0), 1)
    if capped >= 120.0:
        h1 = round(capped * 0.25)
        h2 = round(capped * 0.50)
        h3 = round(capped * 0.80)
        horizons = sorted({float(max(7, h1)), float(max(14, h2)), float(max(28, h3))})
        return horizons, round(max(capped, horizons[-1] + 30.0), 1)

    h_mid = float(max(7, round(capped * 0.4)))
    h_end = float(max(h_mid + 7.0, round(capped * 0.8)))
    return [h_mid, h_end], float(round(max(capped + 14.0, h_end + 14.0), 1))


def _extract_eligibility_highlights(criteria_text: str) -> list[str]:
    if not criteria_text:
        return []
    lines = []
    for raw_line in criteria_text.splitlines():
        cleaned = re.sub(r"^[\s*•\-0-9.)]+", "", raw_line).strip()
        if len(cleaned) < 10 or cleaned.lower() in ("inclusion criteria:", "exclusion criteria:", "inclusion criteria", "exclusion criteria"):
            continue
        lines.append(cleaned[:140])
        if len(lines) >= 5:
            break
    return lines


def _normalize_api_study(study: dict, *, source: str = "clinicaltrials_gov_api_v2", raw_bytes: bytes | None = None) -> dict:
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    cond_mod = proto.get("conditionsModule", {})
    arms_mod = proto.get("armsInterventionsModule", {})
    elig_mod = proto.get("eligibilityModule", {})
    out_mod = proto.get("outcomesModule", {})

    nct_id = str(ident.get("nctId", "")).strip()
    brief_title = str(ident.get("briefTitle", "")).strip()
    official_title = str(ident.get("officialTitle", "")).strip() or brief_title
    phases = [str(p) for p in (design.get("phases") or []) if p]
    enrollment_info = design.get("enrollmentInfo") or {}
    enrollment_count = enrollment_info.get("count") if isinstance(enrollment_info.get("count"), int) else None
    design_info = design.get("designInfo") or {}
    allocation = str(design_info.get("allocation", "RANDOMIZED"))
    conditions = [str(c) for c in (cond_mod.get("conditions") or []) if c]

    exp_arms: list[str] = []
    comp_arms: list[str] = []
    other_arms: list[str] = []
    for arm in arms_mod.get("armGroups") or []:
        if not isinstance(arm, dict):
            continue
        label = str(arm.get("label", "")).strip()
        atype = str(arm.get("type", "")).upper()
        if not label:
            continue
        if "EXPERIMENTAL" in atype:
            exp_arms.append(label)
        elif any(k in atype for k in ("ACTIVE_COMPARATOR", "PLACEBO_COMPARATOR", "SHAM_COMPARATOR", "NO_INTERVENTION")):
            comp_arms.append(label)
        else:
            other_arms.append(label)

    interventions = []
    for intr in arms_mod.get("interventions") or []:
        if isinstance(intr, dict) and intr.get("name"):
            itype = str(intr.get("type", "DRUG")).title()
            interventions.append(f"{itype}: {intr['name']}")

    if not exp_arms and other_arms:
        exp_arms.append(other_arms[0])
    if not comp_arms and len(other_arms) >= 2:
        comp_arms.append(other_arms[1])
    if not exp_arms and interventions:
        exp_arms.append(interventions[0].split(": ", 1)[-1])
    if not comp_arms:
        comp_arms.append(
            interventions[1].split(": ", 1)[-1] if len(interventions) >= 2 else "Standard Comparator / Control"
        )

    primary_outcomes = []
    for po in (out_mod.get("primaryOutcomes") or [])[:3]:
        if isinstance(po, dict) and po.get("measure"):
            primary_outcomes.append(
                {
                    "measure": str(po.get("measure", "")).strip(),
                    "timeFrame": str(po.get("timeFrame", "")).strip() or "Not specified",
                }
            )

    first_tf = primary_outcomes[0]["timeFrame"] if primary_outcomes else ""
    horizons, admin_end = parse_timeframe_to_horizons(first_tf)

    canonical_bytes = raw_bytes if raw_bytes is not None else json.dumps(study, sort_keys=True).encode("utf-8")
    return {
        "nct_id": nct_id,
        "record_url": f"https://clinicaltrials.gov/study/{nct_id}",
        "brief_title": brief_title,
        "official_title": official_title,
        "phases": phases or ["INTERVENTIONAL"],
        "overall_status": str(status.get("overallStatus", "UNKNOWN")),
        "enrollment_count": enrollment_count,
        "allocation": allocation,
        "conditions": conditions,
        "experimental_arms": exp_arms,
        "comparator_arms": comp_arms,
        "interventions": interventions,
        "eligibility_summary": {
            "min_age": str(elig_mod.get("minimumAge", "18 Years")),
            "max_age": str(elig_mod.get("maximumAge", "N/A")),
            "sex": str(elig_mod.get("sex", "ALL")),
            "highlights": _extract_eligibility_highlights(str(elig_mod.get("eligibilityCriteria", ""))),
        },
        "primary_outcomes": primary_outcomes or [{"measure": "Primary Clinical Endpoint", "timeFrame": "12 months"}],
        "parsed_horizon_days": horizons,
        "parsed_admin_end_day": admin_end,
        "source": source,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "receipt_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
    }


def _score_study_deterministic(study: dict, query_tokens: list[str]) -> tuple[int, list[str]]:
    haystack_title = f"{study['brief_title']} {study['official_title']}".lower()
    haystack_arms = " ".join(study.get("experimental_arms", []) + study.get("comparator_arms", []) + study.get("interventions", [])).lower()
    haystack_cond = " ".join(study.get("conditions", [])).lower()
    haystack_out = " ".join(po.get("measure", "") for po in study.get("primary_outcomes", [])).lower()

    raw_pts = 0
    matched_arms = []
    matched_conds = []
    for tok in query_tokens:
        tl = tok.lower()
        if len(tl) < 3:
            continue
        if tl in haystack_arms:
            raw_pts += 18
            matched_arms.append(tok)
        if tl in haystack_title:
            raw_pts += 14
            matched_conds.append(tok)
        if tl in haystack_cond:
            raw_pts += 6
            if tok not in matched_conds:
                matched_conds.append(tok)
        if tl in haystack_out:
            raw_pts += 8

    phases_joined = " ".join(study.get("phases", []))
    if "PHASE3" in phases_joined or "PHASE4" in phases_joined:
        raw_pts += 10
    elif "PHASE2" in phases_joined:
        raw_pts += 5
    if (study.get("enrollment_count") or 0) >= 500:
        raw_pts += 5
    if "RANDOMIZED" in str(study.get("allocation", "")).upper():
        raw_pts += 5

    score = max(35, min(98, 42 + int(round(raw_pts * 0.52))))
    reasons: list[str] = []
    if matched_arms:
        reasons.append(f"试验干预/对照臂直接命中关键词（{', '.join(dict.fromkeys(matched_arms[:3]))}）")
    if matched_conds:
        reasons.append(f"目标适应证/疾病人群匹配（{', '.join(dict.fromkeys(matched_conds[:3]))}）")
    exp_preview = study["experimental_arms"][0] if study.get("experimental_arms") else "试验组"
    comp_preview = study["comparator_arms"][0] if study.get("comparator_arms") else "对照组"
    tf_preview = study["primary_outcomes"][0]["timeFrame"] if study.get("primary_outcomes") else "未注明"
    reasons.append(f"随机双臂对照结构清晰（{exp_preview} vs {comp_preview}），主要终点随访窗：{tf_preview}")
    return score, reasons


def fetch_rct_by_nct(nct_id: str, *, timeout: float = DEFAULT_CT_TIMEOUT) -> dict:
    """Fetch a single study by exact NCT######## from ClinicalTrials.gov API v2 (with offline fallback if matched)."""
    cleaned = (nct_id or "").strip().upper()
    if not re.fullmatch(r"NCT[0-9]{8}", cleaned):
        raise ContractError("INVALID_NCT_ID", f"NCT 编号格式须为 NCT 加 8 位数字（收到：{nct_id}）")

    if os.environ.get("EMULATRX_OFFLINE_RCT_CATALOG") != "1":
        url = f"{API_ROOT}/studies/{cleaned}?{urlencode({'fields': ','.join(STUDY_FIELDS)})}"
        req = Request(url, headers={"Accept": "application/json", "User-Agent": "EmulatRx-RCT-Matcher/1.0"})
        try:
            with urlopen_ctgov(req, timeout=timeout) as resp:
                raw = resp.read(MAX_CT_RESPONSE_BYTES + 1)
            if len(raw) <= MAX_CT_RESPONSE_BYTES:
                data = json.loads(raw.decode("utf-8"))
                if isinstance(data, dict) and "protocolSection" in data:
                    card = _normalize_api_study(data, source="clinicaltrials_gov_api_v2", raw_bytes=raw)
                    card["match_score"] = 99
                    card["match_reasons"] = [f"按用户指定的登记号 {cleaned} 精确拉取 ClinicalTrials.gov 官方记录"]
                    card["ranking_mode"] = "exact_nct_lookup"
                    return card
        except HTTPError as exc:
            if exc.code == 404:
                raise ContractError("NCT_NOT_FOUND", f"ClinicalTrials.gov 未找到编号为 {cleaned} 的注册试验。") from exc
        except (URLError, TimeoutError, OSError, ValueError):
            pass

    for item in OFFLINE_CURATED_RCTS:
        if item["nct_id"] == cleaned:
            raw_fb = json.dumps(item, sort_keys=True).encode("utf-8")
            card = {
                **item,
                "record_url": f"https://clinicaltrials.gov/study/{cleaned}",
                "source": "offline_curated_catalog_fallback",
                "fallback_reason": "clinicaltrials_gov_unreachable_or_offline_mode",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "receipt_sha256": hashlib.sha256(raw_fb).hexdigest(),
                "match_score": 99,
                "match_reasons": [f"离线公开精选目录精确匹配 {cleaned}（当前外网不可达）"],
                "ranking_mode": "exact_nct_lookup",
            }
            return card

    raise ContractError(
        "CTGOV_UNAVAILABLE",
        f"当前无法连接 ClinicalTrials.gov 拉取 {cleaned}，且该编号不在本地离线兜底目录中。",
    )


def search_candidate_rcts(
    query: str,
    *,
    condition: str = "",
    intervention: str = "",
    context_summary: dict | None = None,
    forbidden_values: tuple[str, ...] = (),
    max_results: int = 6,
    timeout: float = DEFAULT_CT_TIMEOUT,
) -> dict:
    """Search ClinicalTrials.gov API v2 for randomized interventional target trials with offline catalog fallback."""
    clean_query = (query or "").strip()
    nct_hit = NCT_RE.search(clean_query)
    if nct_hit and len(clean_query) <= 20:
        single = fetch_rct_by_nct(nct_hit.group(1), timeout=timeout)
        return {
            "source": single["source"],
            "query_used": nct_hit.group(1).upper(),
            "ranking_mode": "exact_nct_lookup",
            "candidates": [single],
        }

    raw_tokens = [
        tok
        for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", f"{clean_query} {condition} {intervention}")
        if tok.lower() not in {"and", "the", "for", "with", "versus", "randomized", "controlled", "trial"}
    ]
    seen_tok_lower: set[str] = set()
    tokens: list[str] = []
    for tok in raw_tokens:
        if tok.lower() not in seen_tok_lower:
            seen_tok_lower.add(tok.lower())
            tokens.append(tok)

    live_candidates: list[dict] = []
    live_error: str | None = None

    if os.environ.get("EMULATRX_OFFLINE_RCT_CATALOG") != "1":
        # Try targeted query first; if 0 studies returned, broaden to main query.term
        search_attempts = []
        if condition.strip() or intervention.strip():
            p1: dict[str, str | int] = {
                "filter.advanced": "AREA[StudyType]INTERVENTIONAL AND AREA[DesignAllocation]RANDOMIZED",
                "fields": ",".join(STUDY_FIELDS),
                "pageSize": 15,
            }
            if condition.strip():
                p1["query.cond"] = condition.strip()
            if intervention.strip():
                p1["query.intr"] = intervention.strip()
            elif clean_query:
                p1["query.term"] = clean_query
            search_attempts.append(p1)

        p2: dict[str, str | int] = {
            "query.term": clean_query or f"{condition} {intervention}".strip() or "randomized phase 3",
            "filter.advanced": "AREA[StudyType]INTERVENTIONAL AND AREA[DesignAllocation]RANDOMIZED",
            "fields": ",".join(STUDY_FIELDS),
            "pageSize": 15,
        }
        search_attempts.append(p2)

        for params in search_attempts:
            url = f"{API_ROOT}/studies?{urlencode(params)}"
            req = Request(url, headers={"Accept": "application/json", "User-Agent": "EmulatRx-RCT-Matcher/1.0"})
            try:
                with urlopen_ctgov(req, timeout=timeout) as resp:
                    raw = resp.read(MAX_CT_RESPONSE_BYTES + 1)
                if len(raw) > MAX_CT_RESPONSE_BYTES:
                    live_error = "response_too_large"
                    break
                payload = json.loads(raw.decode("utf-8"))
                studies = payload.get("studies") if isinstance(payload, dict) else None
                if isinstance(studies, list) and len(studies) > 0:
                    for st in studies:
                        if not isinstance(st, dict):
                            continue
                        st_bytes = json.dumps(st, sort_keys=True).encode("utf-8")
                        norm = _normalize_api_study(st, source="clinicaltrials_gov_api_v2", raw_bytes=st_bytes)
                        if not norm["nct_id"]:
                            continue
                        sc, reasons = _score_study_deterministic(norm, tokens)
                        norm["match_score"] = sc
                        norm["match_reasons"] = reasons
                        norm["ranking_mode"] = "deterministic_rules"
                        live_candidates.append(norm)
                    if live_candidates:
                        break
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                live_error = type(exc).__name__
                break

    if live_candidates:
        live_candidates.sort(key=lambda c: (c["match_score"], c.get("enrollment_count") or 0), reverse=True)
        top = live_candidates[:max_results]
        reranked, rank_mode = rerank_and_explain_rcts(
            top,
            context_summary or {"query": clean_query, "condition": condition, "intervention": intervention},
            forbidden_values=forbidden_values,
        )
        return {
            "source": "clinicaltrials_gov_api_v2",
            "source_label_zh": "实时联网检索 ClinicalTrials.gov API v2",
            "query_used": clean_query or f"{condition} {intervention}".strip(),
            "ranking_mode": rank_mode,
            "candidates": reranked[:max_results],
        }

    # Explicit offline fallback catalog (never disguised as live response)
    fallback_list: list[dict] = []
    for item in OFFLINE_CURATED_RCTS:
        raw_fb = json.dumps(item, sort_keys=True).encode("utf-8")
        entry = {
            **item,
            "record_url": f"https://clinicaltrials.gov/study/{item['nct_id']}",
            "source": "offline_curated_catalog_fallback",
            "fallback_reason": live_error or "no_live_network_or_empty_live_match",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "receipt_sha256": hashlib.sha256(raw_fb).hexdigest(),
        }
        sc, reasons = _score_study_deterministic(entry, tokens)
        entry["match_score"] = sc
        entry["match_reasons"] = [
            *reasons,
            "注：当前来自内置公开 RCT 离线精选目录兜底（source=offline_curated_catalog_fallback）",
        ]
        entry["ranking_mode"] = "deterministic_rules"
        fallback_list.append(entry)

    fallback_list.sort(key=lambda c: (c["match_score"], c.get("enrollment_count") or 0), reverse=True)
    reranked_fb, rank_mode_fb = rerank_and_explain_rcts(
        fallback_list[:max_results],
        context_summary or {"query": clean_query, "condition": condition, "intervention": intervention},
        forbidden_values=forbidden_values,
    )
    return {
        "source": "offline_curated_catalog_fallback",
        "source_label_zh": "内置代表性公开 RCT 目录兜底（offline_curated_catalog_fallback）",
        "fallback_reason": live_error or "offline_or_no_live_results",
        "query_used": clean_query or f"{condition} {intervention}".strip(),
        "ranking_mode": rank_mode_fb,
        "candidates": reranked_fb[:max_results],
    }
