"""Live ClinicalTrials.gov API v2 target RCT retrieval, bilingual drug/class dual-arm alignment auditing, deterministic score capping, and explicit offline curated fallback."""

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
    "protocolSection.outcomesModule.secondaryOutcomes",
    "hasResults",
)

# ---------------------------------------------------------------------------
# Deterministic Bilingual Drug / Mechanism-Class & Condition Lexicon
# ---------------------------------------------------------------------------
# Tuple layout: (canonical_id, en_query_term, drug_class, zh_display, is_class_generic, aliases)
DRUG_CLASS_LEXICON: tuple[tuple[str, str, str, str, bool, tuple[str, ...]], ...] = (
    # SGLT2 inhibitors
    ("dapagliflozin", "Dapagliflozin", "sglt2i", "达格列净 (Dapagliflozin)", False, ("dapagliflozin", "farxiga", "forxiga", "bms-512148", "达格列净")),
    ("empagliflozin", "Empagliflozin", "sglt2i", "恩格列净 (Empagliflozin)", False, ("empagliflozin", "jardiance", "bi 10773", "bi-10773", "恩格列净")),
    ("canagliflozin", "Canagliflozin", "sglt2i", "卡格列净 (Canagliflozin)", False, ("canagliflozin", "invokana", "卡格列净")),
    ("ertugliflozin", "Ertugliflozin", "sglt2i", "埃格列净 (Ertugliflozin)", False, ("ertugliflozin", "steglatro", "埃格列净")),
    ("sotagliflozin", "Sotagliflozin", "sglt2i", "索格列净 (Sotagliflozin)", False, ("sotagliflozin", "索格列净")),
    ("henagliflozin", "Henagliflozin", "sglt2i", "恒格列净 (Henagliflozin)", False, ("henagliflozin", "恒格列净")),
    ("sglt2i", "SGLT2 inhibitor", "sglt2i", "SGLT2 抑制剂", True, ("sglt2i", "sglt-2", "sglt2 inhibitor", "sglt2 inhibitors", "sglt2", "列净类")),
    # DPP-4 inhibitors
    ("sitagliptin", "Sitagliptin", "dpp4i", "西格列汀 (Sitagliptin)", False, ("sitagliptin", "januvia", "mk-0431", "西格列汀")),
    ("linagliptin", "Linagliptin", "dpp4i", "利格列汀 (Linagliptin)", False, ("linagliptin", "tradjenta", "trajenta", "利格列汀")),
    ("saxagliptin", "Saxagliptin", "dpp4i", "沙格列汀 (Saxagliptin)", False, ("saxagliptin", "onglyza", "沙格列汀")),
    ("vildagliptin", "Vildagliptin", "dpp4i", "维格列汀 (Vildagliptin)", False, ("vildagliptin", "galvus", "维格列汀")),
    ("alogliptin", "Alogliptin", "dpp4i", "阿格列汀 (Alogliptin)", False, ("alogliptin", "nesina", "阿格列汀")),
    ("dpp4i", "DPP-4 inhibitor", "dpp4i", "DPP-4 抑制剂", True, ("dpp4i", "dpp-4i", "dpp4 inhibitor", "dpp-4 inhibitor", "dpp-4", "dpp4", "列汀类")),
    # GLP-1 receptor agonists
    ("semaglutide", "Semaglutide", "glp1ra", "司美格鲁肽 (Semaglutide)", False, ("semaglutide", "ozempic", "wegovy", "rybelsus", "司美格鲁肽", "索马鲁肽")),
    ("liraglutide", "Liraglutide", "glp1ra", "利拉鲁肽 (Liraglutide)", False, ("liraglutide", "victoza", "利拉鲁肽")),
    ("dulaglutide", "Dulaglutide", "glp1ra", "度拉糖肽 (Dulaglutide)", False, ("dulaglutide", "trulicity", "度拉糖肽")),
    ("exenatide", "Exenatide", "glp1ra", "艾塞那肽 (Exenatide)", False, ("exenatide", "byetta", "艾塞那肽")),
    ("tirzepatide", "Tirzepatide", "glp1ra", "替尔泊肽 (Tirzepatide)", False, ("tirzepatide", "mounjaro", "替尔泊肽")),
    ("glp1ra", "GLP-1 receptor agonist", "glp1ra", "GLP-1 受体激动剂", True, ("glp1ra", "glp-1ra", "glp-1 receptor agonist", "glp-1", "glp1")),
    # Biguanides, Insulins, Sulfonylureas, MRAs
    ("metformin", "Metformin", "biguanide", "二甲双胍 (Metformin)", False, ("metformin", "glucophage", "二甲双胍", "双胍")),
    ("insulin", "Insulin", "insulin", "胰岛素 (Insulin)", False, ("insulin", "glargine", "degludec", "胰岛素", "甘精胰岛素", "德谷胰岛素")),
    ("glimepiride", "Glimepiride", "sulfonylurea", "格列美脲 (Glimepiride)", False, ("glimepiride", "格列美脲")),
    ("gliclazide", "Gliclazide", "sulfonylurea", "格列齐特 (Gliclazide)", False, ("gliclazide", "格列齐特")),
    ("glipizide", "Glipizide", "sulfonylurea", "格列吡嗪 (Glipizide)", False, ("glipizide", "格列吡嗪")),
    ("sulfonylurea", "Sulfonylurea", "sulfonylurea", "磺脲类 (Sulfonylurea)", True, ("sulfonylurea", "sulfonylureas", "磺脲", "磺酰脲")),
    ("finerenone", "Finerenone", "mra", "非奈利酮 (Finerenone)", False, ("finerenone", "kerendia", "非奈利酮")),
    ("spironolactone", "Spironolactone", "mra", "螺内酯 (Spironolactone)", False, ("spironolactone", "螺内酯")),
    ("eplerenone", "Eplerenone", "mra", "依普利酮 (Eplerenone)", False, ("eplerenone", "依普利酮")),
    # Anticoagulants (DOAC vs VKA)
    ("apixaban", "Apixaban", "doac", "阿哌沙班 (Apixaban)", False, ("apixaban", "eliquis", "阿哌沙班")),
    ("rivaroxaban", "Rivaroxaban", "doac", "利伐沙班 (Rivaroxaban)", False, ("rivaroxaban", "xarelto", "利伐沙班")),
    ("dabigatran", "Dabigatran", "doac", "达比加群 (Dabigatran)", False, ("dabigatran", "pradaxa", "达比加群")),
    ("edoxaban", "Edoxaban", "doac", "艾多沙班 (Edoxaban)", False, ("edoxaban", "savaysa", "艾多沙班")),
    ("warfarin", "Warfarin", "vka", "华法林 (Warfarin)", False, ("warfarin", "coumadin", "vitamin k antagonist", "华法林")),
    # Oncology EGFR-TKIs
    ("osimertinib", "Osimertinib", "egfr_tki", "奥希替尼 (Osimertinib)", False, ("osimertinib", "tagrisso", "azd9291", "奥希替尼")),
    ("gefitinib", "Gefitinib", "egfr_tki", "吉非替尼 (Gefitinib)", False, ("gefitinib", "iressa", "吉非替尼")),
    ("erlotinib", "Erlotinib", "egfr_tki", "厄洛替尼 (Erlotinib)", False, ("erlotinib", "tarceva", "厄洛替尼")),
    ("afatinib", "Afatinib", "egfr_tki", "阿法替尼 (Afatinib)", False, ("afatinib", "gilotrif", "阿法替尼")),
    ("aumolertinib", "Aumolertinib", "egfr_tki", "阿美替尼 (Aumolertinib)", False, ("aumolertinib", "almonertinib", "阿美替尼")),
    ("furmonertinib", "Furmonertinib", "egfr_tki", "伏美替尼 (Furmonertinib)", False, ("furmonertinib", "伏美替尼")),
    # Oncology PD-1 / PD-L1 immune checkpoint inhibitors
    ("pembrolizumab", "Pembrolizumab", "pd1_pdl1", "帕博利珠单抗 (Pembrolizumab)", False, ("pembrolizumab", "keytruda", "mk-3475", "帕博利珠单抗")),
    ("nivolumab", "Nivolumab", "pd1_pdl1", "纳武利尤单抗 (Nivolumab)", False, ("nivolumab", "opdivo", "纳武利尤单抗")),
    ("atezolizumab", "Atezolizumab", "pd1_pdl1", "阿替利珠单抗 (Atezolizumab)", False, ("atezolizumab", "tecentriq", "阿替利珠单抗")),
    ("durvalumab", "Durvalumab", "pd1_pdl1", "度伐利尤单抗 (Durvalumab)", False, ("durvalumab", "imfinzi", "度伐利尤单抗")),
    ("sintilimab", "Sintilimab", "pd1_pdl1", "信迪利单抗 (Sintilimab)", False, ("sintilimab", "信迪利单抗")),
    ("tislelizumab", "Tislelizumab", "pd1_pdl1", "替雷利珠单抗 (Tislelizumab)", False, ("tislelizumab", "替雷利珠单抗")),
    # Cytotoxic chemotherapy
    (
        "platinum_chemo",
        "Platinum Chemotherapy",
        "platinum_chemo",
        "含铂化疗 (Platinum Chemotherapy)",
        False,
        (
            "platinum",
            "carboplatin",
            "cisplatin",
            "pemetrexed",
            "paclitaxel",
            "nab-paclitaxel",
            "chemotherapy",
            "chemo",
            "doublet",
            "含铂",
            "双药化疗",
            "含铂双药",
            "铂类",
            "卡铂",
            "顺铂",
            "培美曲塞",
            "紫杉醇",
            "化疗",
        ),
    ),
    # Placebo / Standard of Care
    (
        "placebo_soc",
        "Placebo",
        "placebo_soc",
        "安慰剂 / 标准治疗 (Placebo / SoC)",
        False,
        (
            "placebo",
            "standard of care",
            "standard care",
            "usual care",
            "matching placebo",
            "saline placebo",
            "安慰剂",
            "标准治疗",
            "常规治疗",
        ),
    ),
)

CONDITION_BILINGUAL_MAP: tuple[tuple[tuple[str, ...], str], ...] = (
    (("2型糖尿病", "二型糖尿病", "糖尿病", "type 2 diabetes", "t2dm", "diabetes"), "Type 2 Diabetes"),
    (("心力衰竭", "心衰", "heart failure", "hfref", "hfpef"), "Heart Failure"),
    (("慢性肾脏病", "慢性肾病", "糖尿病肾病", "chronic kidney disease", "diabetic kidney", "ckd"), "Chronic Kidney Disease"),
    (("非小细胞肺癌", "肺癌", "non-small cell lung cancer", "nsclc", "lung cancer"), "Non-Small Cell Lung Cancer"),
    (("乳腺癌", "breast cancer"), "Breast Cancer"),
    (("心房颤动", "房颤", "atrial fibrillation", "afib"), "Atrial Fibrillation"),
    (("脓毒症", "重症感染", "sepsis"), "Sepsis"),
    (("脑卒中", "卒中", "stroke"), "Stroke"),
)

GENERIC_ARM_PLACEHOLDERS = frozenset(
    {
        "",
        "0",
        "1",
        "yes",
        "no",
        "true",
        "false",
        "active",
        "control",
        "comparator",
        "reference",
        "experimental",
        "any",
        "unspecified",
        "none",
        "arm_0",
        "arm_1",
        "arm_comparator",
        "arm_active",
        "试验组",
        "对照组",
        "试验暴露策略（active arm）",
        "阳性对照或标准治疗（comparator arm）",
        "对标试验干预臂（active）",
        "对标对照治疗臂（comparator）",
    }
)

GENERIC_DRUG_STOPWORDS = frozenset(
    {
        "drug",
        "drugs",
        "biological",
        "active",
        "comparator",
        "experimental",
        "control",
        "reference",
        "arm",
        "arms",
        "group",
        "groups",
        "plus",
        "with",
        "without",
        "added",
        "addon",
        "alone",
        "only",
        "monotherapy",
        "background",
        "standard",
        "care",
        "usual",
        "fixed",
        "combo",
        "combination",
        "single",
        "double",
        "blind",
        "open",
        "label",
        "once",
        "twice",
        "daily",
        "weekly",
        "monthly",
        "hourly",
        "every",
        "days",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
        "hours",
        "cycle",
        "cycles",
        "oral",
        "intravenous",
        "subcutaneous",
        "dose",
        "adjusted",
        "maximally",
        "tolerated",
        "continuing",
        "continued",
        "ongoing",
        "maintenance",
        "extended",
        "release",
        "sustained",
        "immediate",
        "coated",
        "film",
        "matching",
        "saline",
        "sodium",
        "potassium",
        "calcium",
        "hydrochloride",
        "phosphate",
        "succinate",
        "tartrate",
        "mesylate",
        "besylate",
        "fumarate",
        "maleate",
        "acetate",
        "citrate",
        "sulfate",
        "nitrate",
        "monohydrate",
        "dihydrate",
        "anhydrous",
        "therapy",
        "treatment",
        "regimen",
        "antidiabetic",
        "medication",
        "medications",
        "agent",
        "agents",
        "initiated",
        "first",
        "line",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "injection",
        "infusion",
        "participants",
        "patients",
        "subjects",
        "study",
        "trial",
        "randomized",
        "inhibitor",
        "inhibitors",
        "agonist",
        "agonists",
        "versus",
        "target",
    }
)

ALIGNMENT_SCORE_CAPS: dict[str, float] = {
    "exact_dual_arm": 1.0,
    "single_intervention_matched": 1.0,
    "class_dual_arm": 0.72,
    "single_arm_or_comparator_mismatch": 0.45,
    "unmatched_interventions": 0.25,
    "unspecified": 1.0,
}

ALIGNMENT_RANKS: dict[str, int] = {
    "exact_dual_arm": 4,
    "single_intervention_matched": 4,
    "class_dual_arm": 3,
    "single_arm_or_comparator_mismatch": 2,
    "unmatched_interventions": 1,
    "unspecified": 2,
}


# Representative public RCT metadata catalog used ONLY when ClinicalTrials.gov is unreachable/offline.
# Every entry returned from this catalog is explicitly labeled source="offline_curated_catalog_fallback".
OFFLINE_CURATED_RCTS: tuple[dict, ...] = (
    {
        "nct_id": "NCT01243424",
        "brief_title": "CAROLINA: Cardiovascular Outcome Study of Linagliptin Versus Glimepiride in Patients With Type 2 Diabetes",
        "official_title": "A Multicenter, International, Randomized, Parallel Group, Double Blind Study to Evaluate Cardiovascular Safety of Linagliptin Versus Glimepiride in Patients With Type 2 Diabetes Mellitus at High Cardiovascular Risk",
        "phases": ["PHASE3"],
        "overall_status": "COMPLETED",
        "enrollment_count": 6042,
        "allocation": "RANDOMIZED",
        "conditions": ["Diabetes Mellitus, Type 2", "Cardiovascular Risk", "Cardiovascular Diseases"],
        "experimental_arms": ["Linagliptin 5 mg once daily"],
        "comparator_arms": ["Glimepiride 1–4 mg once daily"],
        "interventions": ["Drug: Linagliptin", "Drug: Glimepiride"],
        "eligibility_summary": {
            "min_age": "40 Years",
            "max_age": "85 Years",
            "sex": "ALL",
            "highlights": [
                "确诊 2 型糖尿病且伴高心血管风险或既往动脉粥样硬化性心血管疾病成人患者（40–85 岁）",
                "头对头随机双盲比较利格列汀（DPP-4i）对比格列美脲（磺脲类）主动对照",
                "排除终末期肾病或严重肝功能损害患者",
            ],
        },
        "primary_outcomes": [
            {
                "measure": "Time to First Occurrence of 3P-MACE (Cardiovascular Death, Non-fatal Myocardial Infarction, or Non-fatal Stroke)",
                "timeFrame": "Up to approximately 8 years (median follow-up 6.3 years)",
            }
        ],
        "parsed_horizon_days": [180.0, 365.0, 730.0],
        "parsed_admin_end_day": 1095.0,
    },
    {
        "nct_id": "NCT02296125",
        "brief_title": "AZD9291 (Osimertinib) Versus Gefitinib or Erlotinib in Patients With Locally Advanced or Metastatic Non-small Cell Lung Cancer (FLAURA)",
        "official_title": "A Phase III, Double-blind, Randomised Study to Assess the Efficacy and Safety of AZD9291 Versus a Standard of Care Epidermal Growth Factor Receptor-Tyrosine Kinase Inhibitor in First-Line EGFR Mutation Positive Advanced NSCLC",
        "phases": ["PHASE3"],
        "overall_status": "COMPLETED",
        "enrollment_count": 556,
        "allocation": "RANDOMIZED",
        "conditions": ["Non-Small Cell Lung Cancer", "EGFR-Mutated NSCLC"],
        "experimental_arms": ["Osimertinib 80 mg once daily"],
        "comparator_arms": ["Gefitinib 250 mg once daily"],
        "interventions": ["Drug: Osimertinib", "Drug: Gefitinib"],
        "eligibility_summary": {
            "min_age": "18 Years",
            "max_age": "N/A",
            "sex": "ALL",
            "highlights": [
                "组织学确诊局部晚期或转移性 EGFR 敏感突变（Exon 19 缺失或 L858R）非小细胞肺癌初治成人患者",
                "头对头随机双盲比较奥希替尼对比一代 EGFR-TKI（吉非替尼）",
                "排除既往接受过晚期系统抗肿瘤治疗或 EGFR-TKI 治疗患者",
            ],
        },
        "primary_outcomes": [
            {
                "measure": "Progression-Free Survival (PFS) per RECIST 1.1",
                "timeFrame": "Approximately 18 months",
            }
        ],
        "secondary_outcomes": [
            {
                "measure": "Overall Survival (OS)",
                "timeFrame": "Up to 36 months",
            }
        ],
        "parsed_horizon_days": [90.0, 180.0, 365.0],
        "parsed_admin_end_day": 720.0,
    },
    {
        "nct_id": "NCT01095653",
        "brief_title": "Efficacy and Safety of Dapagliflozin Added to Metformin Compared With Sitagliptin Added to Metformin in Patients With Type 2 Diabetes Mellitus",
        "official_title": "A 24-Week With a 24-Week Extension, Multicenter, Randomized, Double-Blind, Active-Controlled Phase III Study to Evaluate Dapagliflozin Versus Sitagliptin in Type 2 Diabetes",
        "phases": ["PHASE3"],
        "overall_status": "COMPLETED",
        "enrollment_count": 1450,
        "allocation": "RANDOMIZED",
        "conditions": ["Type 2 Diabetes Mellitus"],
        "experimental_arms": ["Dapagliflozin 10 mg once daily"],
        "comparator_arms": ["Sitagliptin 100 mg once daily"],
        "interventions": ["Drug: Dapagliflozin", "Drug: Sitagliptin"],
        "eligibility_summary": {
            "min_age": "18 Years",
            "max_age": "N/A",
            "sex": "ALL",
            "highlights": [
                "确诊单纯 2 型糖尿病成人患者（年龄 ≥ 18 岁）",
                "头对头随机双盲比较达格列净对比西格列汀 24 周降糖疗效",
                "排除心力衰竭 NYHA III–IV 级、重度肾功能衰竭或 1 型糖尿病患者",
            ],
        },
        "primary_outcomes": [
            {
                "measure": "Change From Baseline in HbA1c at Week 24",
                "timeFrame": "24 weeks",
            }
        ],
        "parsed_horizon_days": [42.0, 84.0, 134.0],
        "parsed_admin_end_day": 168.0,
    },
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
        "comparator_arms": ["Matching Placebo + Standard Care"],
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
        "comparator_arms": ["Matching Placebo + Standard Care"],
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
        "comparator_arms": ["Matching Placebo + Standard Care"],
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


def normalize_intervention_spec(raw_text: str) -> dict:
    """Normalize an English/Chinese drug or regimen label into canonical drug IDs, classes, and English search terms."""
    cleaned = (raw_text or "").strip()
    if not cleaned or cleaned.lower() in GENERIC_ARM_PLACEHOLDERS:
        return {
            "raw": cleaned,
            "has_specific_agent": False,
            "drug_ids": set(),
            "class_only_ids": set(),
            "drug_classes": set(),
            "en_terms": [],
            "display_label": cleaned or "未指定",
        }

    norm_space = re.sub(r"[_/+]+", " ", cleaned.lower())
    matched_drugs: set[str] = set()
    matched_class_only: set[str] = set()
    matched_classes: set[str] = set()
    en_terms: list[str] = []
    zh_displays: list[str] = []

    for cid, en_q, dclass, zh_disp, is_cls_generic, aliases in DRUG_CLASS_LEXICON:
        hit = False
        for alias in aliases:
            al = alias.lower()
            if al.isascii() and al.replace("-", "").replace(" ", "").isalpha() and len(al) <= 5:
                if re.search(rf"(?<![a-z0-9]){re.escape(al)}(?![a-z0-9])", norm_space):
                    hit = True
                    break
            elif al in norm_space or al in cleaned.lower():
                hit = True
                break
        if hit:
            if is_cls_generic:
                matched_class_only.add(cid)
            else:
                matched_drugs.add(cid)
            matched_classes.add(dclass)
            if en_q not in en_terms:
                en_terms.append(en_q)
            if zh_disp not in zh_displays:
                zh_displays.append(zh_disp)

    # Remove generic class-only marker if a specific drug in that same class was already matched
    if matched_drugs:
        specific_classes = {
            dclass for cid, _, dclass, _, is_gen, _ in DRUG_CLASS_LEXICON if not is_cls_generic_id(cid) and cid in matched_drugs
        }
        matched_class_only = {c for c in matched_class_only if c not in specific_classes}

    # Generic fallback token extraction for unlisted drugs (e.g., 'sacubitril', 'trastuzumab', 'lapatinib')
    known_alias_words = {
        w
        for _, _, _, _, _, aliases in DRUG_CLASS_LEXICON
        for a in aliases
        for w in re.findall(r"[a-z0-9]{3,}", a.lower())
    }
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9]{3,24}", norm_space):
        tl = tok.lower()
        if tl in GENERIC_DRUG_STOPWORDS or tl in GENERIC_ARM_PLACEHOLDERS or tl in known_alias_words:
            continue
        if re.fullmatch(r"\d+(?:mg|mcg|ug|ml|kg|hr|day|wk|mo)", tl):
            continue
        matched_drugs.add(f"token:{tl}")
        matched_classes.add(f"token_class:{tl}")
        if tok not in en_terms:
            en_terms.append(tok)

    has_agent = bool(matched_drugs or matched_class_only)
    display = zh_displays[0] if len(zh_displays) == 1 else (cleaned[:60] if cleaned else "未指定")
    return {
        "raw": cleaned,
        "has_specific_agent": has_agent,
        "drug_ids": matched_drugs,
        "class_only_ids": matched_class_only,
        "drug_classes": matched_classes,
        "en_terms": en_terms,
        "display_label": display,
    }


def is_cls_generic_id(cid: str) -> bool:
    for item_cid, _, _, _, is_gen, _ in DRUG_CLASS_LEXICON:
        if item_cid == cid:
            return is_gen
    return False


def translate_condition_to_english(raw_cond: str) -> str:
    """Translate Chinese/mixed condition descriptions into concise English ClinicalTrials.gov query terms."""
    text = (raw_cond or "").strip()
    if not text:
        return ""
    lower = text.lower()
    out_en: list[str] = []
    for triggers, en_cond in CONDITION_BILINGUAL_MAP:
        if any(t in lower for t in triggers) and en_cond not in out_en:
            out_en.append(en_cond)
    for ascii_phrase in re.findall(r"[A-Za-z][A-Za-z0-9\s-]{2,40}", text):
        cl = ascii_phrase.strip()
        if cl and cl.lower() not in {x.lower() for x in out_en}:
            out_en.append(cl)
    return " ".join(out_en[:2]).strip()


def translate_query_to_english(raw_query: str) -> str:
    """Ensure Chinese drug/condition keywords in free-text queries are translated to English for ClinicalTrials.gov API v2."""
    text = (raw_query or "").strip()
    if not text:
        return ""
    intr_spec = normalize_intervention_spec(text)
    cond_en = translate_condition_to_english(text)
    ascii_words = [
        w
        for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,24}", text)
        if w.lower() not in {"and", "the", "for", "with", "versus"}
    ]
    parts: list[str] = []
    for item in [*intr_spec["en_terms"], *(cond_en.split() if cond_en else []), *ascii_words]:
        if item and item.lower() not in {p.lower() for p in parts}:
            parts.append(item)
    return " ".join(parts[:8]).strip()


def _extract_study_arm_specs(study: dict) -> tuple[list[dict], str, int]:
    """Build per-arm normalized drug/class specs strictly from declared arm groups (NEVER fabricating arms from study-level `interventions`)."""
    arm_texts: list[str] = []
    arm_labels: list[str] = []
    if "arm_groups_detail" in study:
        details = study.get("arm_groups_detail") or []
        for d in details:
            if isinstance(d, dict):
                lbl = str(d.get("label", "")).strip()
                inames = " ".join(str(x) for x in (d.get("intervention_names") or []))
                desc = str(d.get("description", "")).strip()
                combined = " ".join(p for p in (lbl, inames, desc) if p).strip()
                if combined:
                    arm_texts.append(combined)
                    arm_labels.append(lbl or inames or "Arm")
    else:
        for lbl in study.get("experimental_arms") or []:
            if str(lbl).strip():
                arm_texts.append(str(lbl).strip())
                arm_labels.append(str(lbl).strip())
        for lbl in study.get("comparator_arms") or []:
            if str(lbl).strip():
                arm_texts.append(str(lbl).strip())
                arm_labels.append(str(lbl).strip())

    verified_arm_count = len(arm_texts)
    specs = [normalize_intervention_spec(t) for t in arm_texts]
    if verified_arm_count == 0:
        pooled_intr = "、".join(str(x).split(": ", 1)[-1] for x in (study.get("interventions") or [])[:4]) or "未注明"
        summary_str = f"未声明分组结构（仅研究级干预：{pooled_intr}）"
    elif verified_arm_count == 1:
        summary_str = f"单臂研究：{arm_labels[0]}（无独立对照臂）"
    else:
        exp_disp = " / ".join((study.get("experimental_arms") or arm_labels[:1])[:2])
        comp_disp = " / ".join((study.get("comparator_arms") or arm_labels[1:2])[:2])
        summary_str = f"{exp_disp} vs {comp_disp}"
    return specs, summary_str, verified_arm_count


def audit_study_arm_alignment(
    study: dict,
    *,
    intervention: str = "",
    comparator: str = "",
    project_mode: str = "",
) -> dict:
    """Deterministically verify whether a candidate RCT's distinct arms match target `intervention` and `comparator`."""
    is_cohort_mode = str(project_mode or "").strip() == "user_cohort"
    t_spec = normalize_intervention_spec(intervention)
    c_spec = normalize_intervention_spec(comparator)
    arm_specs, trial_arms_summary, verified_arm_count = _extract_study_arm_specs(study)

    pooled_drugs: set[str] = set()
    pooled_classes: set[str] = set()
    for a in arm_specs:
        pooled_drugs |= a["drug_ids"]
        pooled_classes |= a["drug_classes"]
    for intr_raw in study.get("interventions") or []:
        ispec = normalize_intervention_spec(str(intr_raw).split(": ", 1)[-1])
        pooled_drugs |= ispec["drug_ids"]
        pooled_classes |= ispec["drug_classes"]

    # Guardrail 1B: Single-arm studies (`len(armGroups) == 1`) or missing `armGroups` (`len == 0`) can NEVER be exact_dual_arm!
    if verified_arm_count < 2:
        expected_disp = (
            f"{t_spec['display_label']} vs {c_spec['display_label']}"
            if (t_spec["has_specific_agent"] or c_spec["has_specific_agent"] or is_cohort_mode)
            else "未指定特定药物"
        )
        any_overlap = bool(
            (t_spec["drug_ids"] & pooled_drugs)
            or (c_spec["drug_ids"] & pooled_drugs)
            or (t_spec["drug_classes"] & pooled_classes)
            or (c_spec["drug_classes"] & pooled_classes)
        )
        status_code = "single_arm_or_comparator_mismatch" if (any_overlap or verified_arm_count == 1) else "unmatched_interventions"
        cap_val = ALIGNMENT_SCORE_CAPS[status_code]
        detail_msg = (
            f"该注册记录仅含单臂（{trial_arms_summary}），无随机对照臂，不可作为双臂对标试验（仅供参考）。"
            if verified_arm_count == 1
            else f"该注册记录缺少结构化 armGroups 分组声明（{trial_arms_summary}），无法核验两臂是否独立随机分配，需显式确认。"
        )
        return {
            "arm_alignment_status": status_code,
            "arm_alignment_rank": ALIGNMENT_RANKS[status_code],
            "arm_alignment_score_cap": cap_val,
            "arm_alignment_label": "对照不一致 · 仅供参考",
            "arm_alignment_detail": detail_msg,
            "expected_arms_summary": expected_disp,
            "trial_arms_summary": trial_arms_summary,
            "requires_mismatch_confirmation": True,
            "is_recommended": False,
        }

    # Guardrail 2: In `user_cohort` mode, comparative cohorts ALWAYS represent a two-arm contrast (`active` vs `comparator`).
    # Binary 0/1 or class-only unresolved cohort arms MUST NEVER enter `single_intervention_matched` or `exact_dual_arm`!
    if is_cohort_mode and (not t_spec["drug_ids"] or not c_spec["drug_ids"]):
        t_lbl = t_spec["display_label"] if t_spec["has_specific_agent"] else f"未指明药物（编码 {intervention or '1'}）"
        c_lbl = c_spec["display_label"] if c_spec["has_specific_agent"] else f"未指明对照（编码 {comparator or '0'}）"
        expected_unresolved = f"{t_lbl} vs {c_lbl}"

        # If both cohort arms at least resolved to two distinct mechanism classes (e.g. sglt2i_vs_dpp4i=1/0), allow class_dual_arm (still requires confirmation!)
        if t_spec["drug_classes"] and c_spec["drug_classes"] and t_spec["drug_classes"] != c_spec["drug_classes"]:
            t_only_c = t_spec["drug_classes"] - c_spec["drug_classes"]
            c_only_c = c_spec["drug_classes"] - t_spec["drug_classes"]
            if t_only_c and c_only_c:
                for i, arm_i in enumerate(arm_specs):
                    if not (arm_i["drug_classes"] & t_only_c):
                        continue
                    for j, arm_j in enumerate(arm_specs):
                        if i != j and (arm_j["drug_classes"] & c_only_c) and not (arm_j["drug_classes"] & t_only_c):
                            return {
                                "arm_alignment_status": "class_dual_arm",
                                "arm_alignment_rank": ALIGNMENT_RANKS["class_dual_arm"],
                                "arm_alignment_score_cap": ALIGNMENT_SCORE_CAPS["class_dual_arm"],
                                "arm_alignment_label": "同机制类别参考 · 需确认",
                                "arm_alignment_detail": (
                                    f"当前队列仅识别到药理类别「{expected_unresolved}」，试验分组为「{trial_arms_summary}」，"
                                    "请补充具体通用名或勾选不一致知情确认框。"
                                ),
                                "expected_arms_summary": expected_unresolved,
                                "trial_arms_summary": trial_arms_summary,
                                "requires_mismatch_confirmation": True,
                                "is_recommended": False,
                            }

        return {
            "arm_alignment_status": "single_arm_or_comparator_mismatch",
            "arm_alignment_rank": ALIGNMENT_RANKS["single_arm_or_comparator_mismatch"],
            "arm_alignment_score_cap": ALIGNMENT_SCORE_CAPS["single_arm_or_comparator_mismatch"],
            "arm_alignment_label": "对照不一致 · 仅供参考",
            "arm_alignment_detail": (
                f"当前自有队列分组为编码或单侧类别「{expected_unresolved}」，无法自动核验与该试验「{trial_arms_summary}」双臂一致；"
                "请在上方补充干预/对照通用名重搜，或勾选知情确认仅借用纳排与随访窗。"
            ),
            "expected_arms_summary": expected_unresolved,
            "trial_arms_summary": trial_arms_summary,
            "requires_mismatch_confirmation": True,
            "is_recommended": False,
        }

    # Case A: Idea/simulated mode where neither intervention nor comparator specifies a concrete drug/class
    if not t_spec["has_specific_agent"] and not c_spec["has_specific_agent"]:
        return {
            "arm_alignment_status": "unspecified",
            "arm_alignment_rank": ALIGNMENT_RANKS["unspecified"],
            "arm_alignment_score_cap": 1.0,
            "arm_alignment_label": "未限定双臂药物 · 参考框架",
            "arm_alignment_detail": f"该试验分组为「{trial_arms_summary}」。",
            "expected_arms_summary": "未指定特定药物",
            "trial_arms_summary": trial_arms_summary,
            "requires_mismatch_confirmation": False,
            "is_recommended": True,
        }

    # Case B: Single-intervention mode (STRICTLY restricted to non-cohort `idea` mode without explicit comparator)
    if not is_cohort_mode and t_spec["has_specific_agent"] and not c_spec["has_specific_agent"]:
        t_drugs_single = t_spec["drug_ids"]
        exact_or_class_hit = any(
            (bool(t_drugs_single) and t_drugs_single.issubset(a["drug_ids"]))
            or (not t_drugs_single and bool(t_spec["class_only_ids"]) and bool(a["drug_classes"] & t_spec["drug_classes"]))
            for a in arm_specs
        )
        if exact_or_class_hit:
            return {
                "arm_alignment_status": "single_intervention_matched",
                "arm_alignment_rank": ALIGNMENT_RANKS["single_intervention_matched"],
                "arm_alignment_score_cap": ALIGNMENT_SCORE_CAPS["single_intervention_matched"],
                "arm_alignment_label": "单臂干预匹配 · 推荐参考",
                "arm_alignment_detail": f"已匹配目标干预「{t_spec['display_label']}」；试验实际分组：{trial_arms_summary}。",
                "expected_arms_summary": f"{t_spec['display_label']}（未限定对照）",
                "trial_arms_summary": trial_arms_summary,
                "requires_mismatch_confirmation": False,
                "is_recommended": True,
            }
        return {
            "arm_alignment_status": "unmatched_interventions",
            "arm_alignment_rank": ALIGNMENT_RANKS["unmatched_interventions"],
            "arm_alignment_score_cap": ALIGNMENT_SCORE_CAPS["unmatched_interventions"],
            "arm_alignment_label": "对照不一致 · 仅供参考",
            "arm_alignment_detail": f"当前研究目标干预为「{t_spec['display_label']}」，而该试验分组为「{trial_arms_summary}」，未匹配目标干预。",
            "expected_arms_summary": t_spec["display_label"],
            "trial_arms_summary": trial_arms_summary,
            "requires_mismatch_confirmation": True,
            "is_recommended": False,
        }

    # Case C: Dual-arm contrast specified — enforce full active regimen drug-set equality for `exact_dual_arm`!
    expected_summary = f"{t_spec['display_label']} vs {c_spec['display_label']}"
    t_core_drugs = set(t_spec["drug_ids"]) - {"placebo_soc"}
    c_core_drugs = set(c_spec["drug_ids"]) - {"placebo_soc"}
    t_is_placebo = "placebo_soc" in t_spec["drug_ids"]
    c_is_placebo = "placebo_soc" in c_spec["drug_ids"]
    shared_drugs = t_core_drugs & c_core_drugs
    t_diff_drugs = t_core_drugs - shared_drugs
    c_diff_drugs = c_core_drugs - shared_drugs

    # 1. `exact_dual_arm` holds ONLY when full active drug sets match identically on both arms
    #    (no missing shared background therapy AND no extra unrequested active drugs on either arm!)
    exact_dual = False
    if not t_spec["class_only_ids"] and not c_spec["class_only_ids"]:
        for i, arm_a in enumerate(arm_specs):
            arm_a_drugs = set(arm_a["drug_ids"]) - {"placebo_soc"}
            arm_a_is_placebo = "placebo_soc" in arm_a["drug_ids"]
            for j, arm_b in enumerate(arm_specs):
                if i == j:
                    continue
                arm_b_drugs = set(arm_b["drug_ids"]) - {"placebo_soc"}
                arm_b_is_placebo = "placebo_soc" in arm_b["drug_ids"]
                if t_core_drugs and c_core_drugs and t_core_drugs != c_core_drugs:
                    if arm_a_drugs == t_core_drugs and arm_b_drugs == c_core_drugs:
                        exact_dual = True
                        break
                elif t_core_drugs and not c_core_drugs and c_is_placebo:
                    if arm_a_drugs == t_core_drugs and len(arm_b_drugs) == 0 and arm_b_is_placebo:
                        exact_dual = True
                        break
                elif not t_core_drugs and t_is_placebo and c_core_drugs:
                    if len(arm_a_drugs) == 0 and arm_a_is_placebo and arm_b_drugs == c_core_drugs:
                        exact_dual = True
                        break
            if exact_dual:
                break

    if exact_dual:
        return {
            "arm_alignment_status": "exact_dual_arm",
            "arm_alignment_rank": ALIGNMENT_RANKS["exact_dual_arm"],
            "arm_alignment_score_cap": ALIGNMENT_SCORE_CAPS["exact_dual_arm"],
            "arm_alignment_label": "双臂干预一致 · 推荐对标",
            "arm_alignment_detail": f"双臂干预严格对齐：当前研究「{expected_summary}」与该试验分组「{trial_arms_summary}」一致。",
            "expected_arms_summary": expected_summary,
            "trial_arms_summary": trial_arms_summary,
            "requires_mismatch_confirmation": False,
            "is_recommended": True,
        }

    # 1b. Symmetric background-differing core drug match: differential drugs match (`t_diff_drugs` vs `c_diff_drugs`)
    #     with NO one-sided extra active drug, but shared background/co-medication differs (e.g. Dapa+Met vs Sita+Met against Dapa vs Sita).
    bg_diff_core_match = False
    if not t_spec["class_only_ids"] and not c_spec["class_only_ids"] and t_diff_drugs and c_diff_drugs:
        for i, arm_a in enumerate(arm_specs):
            arm_a_drugs = set(arm_a["drug_ids"]) - {"placebo_soc"}
            if not t_diff_drugs.issubset(arm_a_drugs) or (arm_a_drugs & c_diff_drugs):
                continue
            for j, arm_b in enumerate(arm_specs):
                if i == j:
                    continue
                arm_b_drugs = set(arm_b["drug_ids"]) - {"placebo_soc"}
                if not c_diff_drugs.issubset(arm_b_drugs) or (arm_b_drugs & t_diff_drugs):
                    continue
                if (arm_a_drugs - t_diff_drugs) == (arm_b_drugs - c_diff_drugs):
                    bg_diff_core_match = True
                    break
            if bg_diff_core_match:
                break

    if bg_diff_core_match:
        return {
            "arm_alignment_status": "class_dual_arm",
            "arm_alignment_rank": ALIGNMENT_RANKS["class_dual_arm"],
            "arm_alignment_score_cap": ALIGNMENT_SCORE_CAPS["class_dual_arm"],
            "arm_alignment_label": "联合背景用药差异 · 需确认",
            "arm_alignment_detail": (
                f"核心差异药物一致，但联合/背景基础治疗与当前研究不同：当前研究对比「{expected_summary}」，"
                f"该试验分组为「{trial_arms_summary}」，选为参考方案前需显式确认。"
            ),
            "expected_arms_summary": expected_summary,
            "trial_arms_summary": trial_arms_summary,
            "requires_mismatch_confirmation": True,
            "is_recommended": False,
        }

    # 2. Check `class_dual_arm`: subtract shared background classes and reject asymmetric extra active classes/drugs
    t_core_classes = set(t_spec["drug_classes"]) - {"placebo_soc"}
    c_core_classes = set(c_spec["drug_classes"]) - {"placebo_soc"}
    shared_classes = t_core_classes & c_core_classes
    t_diff_classes = t_core_classes - shared_classes
    c_diff_classes = c_core_classes - shared_classes
    class_dual = False
    if t_diff_classes and c_diff_classes:
        for i, arm_i in enumerate(arm_specs):
            arm_i_classes = set(arm_i["drug_classes"]) - {"placebo_soc"}
            arm_i_drugs = set(arm_i["drug_ids"]) - {"placebo_soc"}
            if not t_diff_classes.issubset(arm_i_classes) or (arm_i_classes & c_diff_classes):
                continue
            for j, arm_j in enumerate(arm_specs):
                if i == j:
                    continue
                arm_j_classes = set(arm_j["drug_classes"]) - {"placebo_soc"}
                arm_j_drugs = set(arm_j["drug_ids"]) - {"placebo_soc"}
                if not c_diff_classes.issubset(arm_j_classes) or (arm_j_classes & t_diff_classes):
                    continue
                if (arm_i_classes - t_diff_classes) == (arm_j_classes - c_diff_classes) and len(arm_i_drugs) == len(arm_j_drugs):
                    class_dual = True
                    break
            if class_dual:
                break
    elif t_core_classes and not c_core_classes and c_is_placebo:
        for i, arm_i in enumerate(arm_specs):
            arm_i_classes = set(arm_i["drug_classes"]) - {"placebo_soc"}
            if arm_i_classes != t_core_classes:
                continue
            for j, arm_j in enumerate(arm_specs):
                if i == j:
                    continue
                arm_j_drugs = set(arm_j["drug_ids"]) - {"placebo_soc"}
                if len(arm_j_drugs) == 0 and ("placebo_soc" in arm_j["drug_ids"]):
                    class_dual = True
                    break
            if class_dual:
                break

    if class_dual:
        return {
            "arm_alignment_status": "class_dual_arm",
            "arm_alignment_rank": ALIGNMENT_RANKS["class_dual_arm"],
            "arm_alignment_score_cap": ALIGNMENT_SCORE_CAPS["class_dual_arm"],
            "arm_alignment_label": "同机制类别参考 · 需确认",
            "arm_alignment_detail": (
                f"当前研究对比「{expected_summary}」，该试验分组为「{trial_arms_summary}」："
                "两臂同属对应药理机制类别但具体通用名不同，选为参考方案前需显式确认。"
            ),
            "expected_arms_summary": expected_summary,
            "trial_arms_summary": trial_arms_summary,
            "requires_mismatch_confirmation": True,
            "is_recommended": False,
        }

    # 3. Check `single_arm_or_comparator_mismatch` vs `unmatched_interventions`
    distinct_target_drugs = t_diff_drugs | c_diff_drugs or (t_core_drugs | c_core_drugs)
    distinct_target_classes = t_diff_classes | c_diff_classes or (t_core_classes | c_core_classes)
    pooled_active_drugs = pooled_drugs - {"placebo_soc"}
    pooled_active_classes = pooled_classes - {"placebo_soc"}
    any_drug_or_class_overlap = bool(
        (distinct_target_drugs & pooled_active_drugs) or (distinct_target_classes & pooled_active_classes)
    )
    if any_drug_or_class_overlap:
        return {
            "arm_alignment_status": "single_arm_or_comparator_mismatch",
            "arm_alignment_rank": ALIGNMENT_RANKS["single_arm_or_comparator_mismatch"],
            "arm_alignment_score_cap": ALIGNMENT_SCORE_CAPS["single_arm_or_comparator_mismatch"],
            "arm_alignment_label": "对照不一致 · 仅供参考",
            "arm_alignment_detail": (
                f"当前研究对比「{expected_summary}」，而该注册试验实际分组为「{trial_arms_summary}」"
                "（缺少目标对照臂或系同类药内部互比，绝不可作为直接双臂疗效对标，仅可借用纳排标准与随访时间窗框架）。"
            ),
            "expected_arms_summary": expected_summary,
            "trial_arms_summary": trial_arms_summary,
            "requires_mismatch_confirmation": True,
            "is_recommended": False,
        }

    return {
        "arm_alignment_status": "unmatched_interventions",
        "arm_alignment_rank": ALIGNMENT_RANKS["unmatched_interventions"],
        "arm_alignment_score_cap": ALIGNMENT_SCORE_CAPS["unmatched_interventions"],
        "arm_alignment_label": "对照不一致 · 仅供参考",
        "arm_alignment_detail": (
            f"当前研究对比「{expected_summary}」，而该注册试验实际分组为「{trial_arms_summary}」，两臂干预均不一致。"
        ),
        "expected_arms_summary": expected_summary,
        "trial_arms_summary": trial_arms_summary,
        "requires_mismatch_confirmation": True,
        "is_recommended": False,
    }


# ---------------------------------------------------------------------------
# Four-Dimensional PICO-T Target Trial Alignment Engine
# (Design/PK + Population + Dual-Arm Interventions + Primary Outcome + Horizon)
# ---------------------------------------------------------------------------

EXCLUSION_HEADING_RE = re.compile(
    r"(?im)(?:^\s*[\*•\-0-9.)]*\s*(?:key\s+|main\s+|major\s+)?exclusion\s+criteria\b|exclusion\s+criteria\s*:|排除标准)",
)

HEALTHY_VOLUNTEER_RE = re.compile(
    r"(?i)\b(?:healthy\s+(?:adult\s+|male\s+|female\s+)?(?:volunteers?|subjects?|participants?|adults?|individuals?|men|women)|normal\s+healthy)\b|健康受试者|健康志愿者|健康成人",
)

PK_BIOMARKER_DESIGN_RE = re.compile(
    r"(?i)(?:\b(?:pharmacokinetic|pharmacokinetics|cmax|tmax|auc(?:0|_[0-9a-z]+|\(0|\s|$)|bioequivalence|bioavailability|drug-drug\s+interaction|drug\s+interaction|mass\s+balance|single\s+ascending\s+dose|multiple\s+ascending\s+dose)\b|药代动力学|生物等效性|药物相互作用)",
)

POPULATION_DOMAIN_TAXONOMY: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "t2dm",
        "2型糖尿病",
        (
            "type 2 diabetes",
            "type ii diabetes",
            "diabetes mellitus, type 2",
            "diabetes mellitus type 2",
            "t2dm",
            "t2d",
            "diabetic",
            "2型糖尿病",
            "2 型糖尿病",
            "二型糖尿病",
            "糖尿病",
        ),
    ),
    (
        "heart_failure",
        "心力衰竭",
        (
            "heart failure",
            "hfref",
            "hfpef",
            "cardiac failure",
            "chronic heart failure",
            "worsening heart failure",
            "心力衰竭",
            "心衰",
            "心功能不全",
        ),
    ),
    (
        "ckd",
        "慢性肾脏病",
        (
            "chronic kidney disease",
            "kidney disease",
            "renal insufficiency",
            "diabetic kidney",
            "diabetic nephropathy",
            "nephropathy",
            "albuminuria",
            "proteinuria",
            "ckd",
            "dkd",
            "慢性肾脏病",
            "慢性肾病",
            "糖尿病肾病",
            "肾病",
            "肾功能不全",
        ),
    ),
    (
        "cv_high_risk",
        "心血管高危",
        (
            "cardiovascular risk",
            "cardiovascular outcome",
            "cardiovascular outcomes",
            "cardiovascular safety",
            "cardiovascular disease",
            "cardiovascular diseases",
            "elevated cardiovascular",
            "high cardiovascular",
            "atherosclerotic",
            "ascvd",
            "coronary artery",
            "coronary heart",
            "cardiorenal",
            "心血管风险",
            "心血管高危",
            "心血管疾病",
            "动脉粥样硬化",
            "冠心病",
            "心肾",
        ),
    ),
    (
        "nsclc",
        "非小细胞肺癌",
        (
            "non-small cell lung",
            "non-small-cell lung",
            "nonsquamous non-small cell",
            "squamous non-small cell",
            "nsclc",
            "lung cancer",
            "非小细胞肺癌",
            "肺癌",
            "肺腺癌",
            "肺鳞癌",
        ),
    ),
    (
        "egfr_mut",
        "EGFR突变",
        (
            "egfr-mutated",
            "egfr mutated",
            "egfr mutation",
            "egfr mutations",
            "egfr-positive",
            "egfr positive",
            "egfr exon",
            "exon 19",
            "exon19",
            "l858r",
            "t790m",
            "azd9291",
            "egfr突变",
            "egfr 突变",
            "egfr敏感",
            "egfr 敏感",
            "egfr阳性",
        ),
    ),
    ("afib", "心房颤动", ("atrial fibrillation", "atrial flutter", "afib", "nvaf", "心房颤动", "房颤")),
    ("breast_cancer", "乳腺癌", ("breast cancer", "breast neoplasm", "乳腺癌")),
    ("copd", "慢阻肺", ("chronic obstructive pulmonary", "copd", "慢阻肺", "慢性阻塞性肺")),
    ("sepsis", "脓毒症", ("sepsis", "septic shock", "脓毒症", "感染性休克")),
)

POPULATION_DOMAIN_ZH: dict[str, str] = {dom_id: zh for dom_id, zh, _ in POPULATION_DOMAIN_TAXONOMY}

CLINICAL_OUTCOME_DOMAINS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "hf_cv_composite",
        "心衰/心血管事件",
        (
            "heart failure",
            "worsening hf",
            "hf hospitalization",
            "hospitalization for heart failure",
            "hhf",
            "cardiovascular death",
            "cv death",
            "cv mortality",
            "cardiovascular mortality",
            "cardiovascular event",
            "cardiovascular events",
            "cardiorenal",
            "mace",
            "3p-mace",
            "major adverse cardiovascular",
            "myocardial infarction",
            "stroke",
            "systemic embolism",
            "thromboembolic",
            "hf_or_cv",
            "hf_hospitalization",
            "hf event",
            "cv_event",
            "mace_cv",
            "心衰住院",
            "心衰恶化",
            "心力衰竭",
            "心血管死亡",
            "心血管事件",
            "主要不良心血管",
            "心肌梗死",
            "卒中",
            "体循环栓塞",
        ),
    ),
    (
        "mortality_os",
        "总生存/死亡",
        (
            "overall survival",
            "all-cause mortality",
            "all cause mortality",
            "total mortality",
            "time to death",
            "survival rate",
            "cancer death",
            "cancerdeath",
            "os_days",
            "总生存",
            "全因死亡",
            "肿瘤特异死亡",
            "死亡风险",
        ),
    ),
    (
        "oncology_progression",
        "无进展生存/进展",
        (
            "progression-free survival",
            "progression free survival",
            "disease-free survival",
            "event-free survival",
            "time to progression",
            "tumor progression",
            "disease progression",
            "progression_or",
            "pfs_days",
            "primary relapse",
            "relapse",
            "recurrence",
            "无进展生存",
            "无病生存",
            "疾病进展",
            "肿瘤进展",
            "复发",
        ),
    ),
    (
        "renal_hard_endpoint",
        "肾脏硬终点",
        (
            "end-stage kidney",
            "end stage kidney",
            "end-stage renal",
            "eskd",
            "esrd",
            "kidney failure",
            "renal failure",
            "dialysis",
            "kidney transplant",
            "sustained egfr",
            "sustained decrease",
            "sustained decline",
            "kidney disease progression",
            "renal death",
            "renal composite",
            "终末期肾病",
            "肾功能衰竭",
            "肾衰竭",
            "透析",
            "持续 egfr",
        ),
    ),
)

SURROGATE_OUTCOME_DOMAINS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "ketone_metabolic_lab",
        "血酮",
        (
            "ketone",
            "ketones",
            "ketonemia",
            "ketosis",
            "beta-hydroxybutyrate",
            "b-hydroxybutyrate",
            "3-hydroxybutyrate",
            "acetoacetate",
            "血酮",
            "酮体",
            "酮症",
        ),
    ),
    (
        "glycemic_lab",
        "HbA1c/血糖",
        (
            "hba1c",
            "a1c",
            "glycated hemoglobin",
            "glycosylated hemoglobin",
            "fasting plasma glucose",
            "postprandial glucose",
            "blood glucose",
            "mean amplitude of glycemic",
            "time in range",
            "insulin sensitivity",
            "homa-ir",
            "糖化血红蛋白",
            "空腹血糖",
            "血糖",
        ),
    ),
    (
        "vitals_lipids_weight",
        "体重/血压/生物标志物",
        (
            "body weight",
            "waist circumference",
            "bmi",
            "systolic blood pressure",
            "diastolic blood pressure",
            "ldl",
            "triglyceride",
            "nt-probnp",
            "lvef",
            "ejection fraction",
            "uacr",
            "albumin-to-creatinine",
            "urinary albumin",
            "体重变化",
            "血压变化",
            "射血分数",
        ),
    ),
    (
        "pk_safety_only",
        "药代/安全性",
        (
            "cmax",
            "tmax",
            "auc",
            "auc0",
            "pharmacokinetic",
            "pharmacokinetics",
            "bioequivalence",
            "drug-drug interaction",
            "mass balance",
            "clearance",
            "adverse event",
            "adverse events",
            "teae",
            "dose-limiting toxicity",
            "药代动力学",
            "生物等效性",
            "不良事件",
        ),
    ),
)

SURROGATE_OUTCOME_ZH: dict[str, str] = {sid: zh for sid, zh, _ in SURROGATE_OUTCOME_DOMAINS}
CLINICAL_OUTCOME_ZH: dict[str, str] = {cid: zh for cid, zh, _ in CLINICAL_OUTCOME_DOMAINS}

OVERALL_ALIGNMENT_RANKS: dict[str, int] = {
    "protocol_aligned": 5,
    "partial_protocol_reference": 3,
    "arm_mismatch_reference": 2,
    "non_efficacy_excluded": 1,
}


def extract_positive_inclusion_text(raw_criteria: str) -> str:
    """Return ONLY eligibility text strictly prior to 'Exclusion Criteria' / '排除标准'."""
    if not raw_criteria:
        return ""
    parts = EXCLUSION_HEADING_RE.split(str(raw_criteria), maxsplit=1)
    return parts[0].strip()


def extract_condition_domains(raw_text: str) -> set[str]:
    """Map English/Chinese disease or population text into canonical clinical indication domains."""
    if not raw_text or not raw_text.strip():
        return set()
    cleaned = re.sub(r"(?i)type\s*1\s*diabetes|1\s*型糖尿病|diabetes\s+insipidus", " ", raw_text.lower())
    cleaned_norm = re.sub(r"[_/]+", " ", cleaned)
    matched: set[str] = set()
    for dom_id, _, triggers in POPULATION_DOMAIN_TAXONOMY:
        for trig in triggers:
            tl = trig.lower()
            if tl.isascii() and tl.replace("-", "").isalpha() and len(tl) <= 5:
                if re.search(rf"(?<![a-z0-9]){re.escape(tl)}(?![a-z0-9])", cleaned_norm):
                    matched.add(dom_id)
                    break
            elif tl in cleaned_norm or tl in cleaned:
                matched.add(dom_id)
                break
    return matched


def classify_outcome_domains(raw_text: str) -> tuple[set[str], set[str]]:
    """Classify outcome measure text into (clinical_hard_domains, surrogate_biomarker_domains)."""
    if not raw_text or not raw_text.strip():
        return set(), set()
    low = re.sub(r"[_/]+", " ", raw_text.lower())
    clin: set[str] = set()
    surr: set[str] = set()
    for cid, _, triggers in CLINICAL_OUTCOME_DOMAINS:
        for trig in triggers:
            tl = trig.lower()
            if tl.isascii() and tl.isalpha() and len(tl) <= 4:
                if re.search(rf"(?<![a-z0-9]){re.escape(tl)}(?![a-z0-9])", low):
                    clin.add(cid)
                    break
            elif tl in low:
                clin.add(cid)
                break
    if re.search(r"\b(?:pfs|dfs|efs)\b", low):
        clin.add("oncology_progression")
    if re.search(r"\bos\b", low) and any(k in low for k in ("survival", "cancer", "tumor", "month", "recist")):
        clin.add("mortality_os")

    for sid, _, triggers in SURROGATE_OUTCOME_DOMAINS:
        for trig in triggers:
            tl = trig.lower()
            if tl.isascii() and tl.isalpha() and len(tl) <= 4:
                if re.search(rf"(?<![a-z0-9]){re.escape(tl)}(?![a-z0-9])", low):
                    surr.add(sid)
                    break
            elif tl in low:
                surr.add(sid)
                break
    return clin, surr


def _parse_outcome_timeframe_days(timeframes: list[str]) -> tuple[float | None, bool, str]:
    """Return (max_parsed_days_or_None, is_event_driven, concise_display_tf) from outcome timeFrame strings."""
    joined = " ; ".join(t.strip() for t in timeframes if t and t.strip() and t.strip().lower() != "not specified")
    if not joined:
        return None, False, "未注明"
    low = joined.lower()
    is_event_driven = bool(
        re.search(
            r"event[- ]driven|until\s+(?:\d+\s+)?(?:primary\s+)?(?:endpoint\s+)?events?|until\s+(?:disease\s+)?progression|until\s+death|events?\s+accrue",
            low,
        )
    )
    candidates: list[float] = []
    for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:year|years|yr|yrs|年)", low):
        candidates.append(float(x) * 365.0)
    for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:month|months|mo|mos|个月|月)", low):
        candidates.append(float(x) * 30.4375)
    for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:week|weeks|wk|wks|周)", low):
        candidates.append(float(x) * 7.0)
    for x in re.findall(r"(?:day\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:day|days|天|日)", low):
        candidates.append(float(x))
    for m in re.findall(r"\bday\s+([0-9]+(?:\.[0-9]+)?)\b", low):
        candidates.append(float(m))
    for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:hour|hours|hr|hrs|小时)", low):
        candidates.append(float(x) / 24.0)

    short_disp = timeframes[0].strip()[:32] if timeframes and timeframes[0].strip() else "未注明"
    if not candidates:
        return None, is_event_driven, short_disp
    return max(candidates), is_event_driven, short_disp


def _extract_study_positive_population_text(study: dict) -> str:
    """Gather conditions, titles, and strictly pre-exclusion inclusion criteria text for a study."""
    parts: list[str] = [
        str(study.get("brief_title", "")),
        str(study.get("official_title", "")),
        " ".join(str(c) for c in (study.get("conditions") or [])),
    ]
    has_raw_or_inclusion = False
    if "inclusion_text" in study:
        has_raw_or_inclusion = True
        if study["inclusion_text"]:
            parts.append(str(study["inclusion_text"]))
    raw_elig = (
        study.get("eligibility_criteria")
        or ((study.get("eligibilityModule") or {}).get("eligibilityCriteria") if isinstance(study.get("eligibilityModule"), dict) else "")
        or ((study.get("protocolSection") or {}).get("eligibilityModule", {}).get("eligibilityCriteria") if isinstance(study.get("protocolSection"), dict) else "")
    )
    if raw_elig:
        has_raw_or_inclusion = True
        parts.append(extract_positive_inclusion_text(str(raw_elig)))
    elig_sum = study.get("eligibility_summary")
    if not has_raw_or_inclusion and isinstance(elig_sum, dict):
        for hl in elig_sum.get("highlights") or []:
            hl_s = str(hl).strip()
            if hl_s and not hl_s.startswith("排除") and not re.match(r"(?i)^exclu", hl_s):
                parts.append(hl_s)
    return " ".join(p for p in parts if p)


def audit_study_pico4d_alignment(
    study: dict,
    *,
    intervention: str = "",
    comparator: str = "",
    target_condition: str = "",
    target_outcome: str = "",
    target_followup_days: float | None = None,
    project_mode: str = "",
) -> dict:
    """Evaluate a candidate RCT across all 5 PICO-T dimensions and compute overall alignment verdict & score cap."""
    is_cohort_mode = str(project_mode or "").strip() == "user_cohort"
    arm_audit = audit_study_arm_alignment(
        study,
        intervention=intervention,
        comparator=comparator,
        project_mode=project_mode,
    )
    arm_status = arm_audit["arm_alignment_status"]
    arm_cap = float(arm_audit["arm_alignment_score_cap"])

    # --- 1. Study Design & Healthy-Volunteer / PK-PD Suitability (`design_suitability`) ---
    elig_mod = study.get("eligibilityModule") if isinstance(study.get("eligibilityModule"), dict) else {}
    healthy_flag = bool(study.get("healthy_volunteers") is True or elig_mod.get("healthyVolunteers") is True)
    pos_pop_text = _extract_study_positive_population_text(study)
    title_cond_text = f"{study.get('brief_title', '')} {study.get('official_title', '')} {' '.join(str(c) for c in (study.get('conditions') or []))}"
    healthy_text_hit = bool(
        HEALTHY_VOLUNTEER_RE.search(pos_pop_text)
        and not re.search(r"(?i)no\s+healthy\s+volunteers|healthy\s+volunteers\s*:\s*no", pos_pop_text)
    )

    prim_list = study.get("primary_outcomes") or []
    sec_list = study.get("secondary_outcomes") or []
    prim_measures = " ; ".join(
        str(po.get("measure", po) if isinstance(po, dict) else po) for po in prim_list
    )
    sec_measures = " ; ".join(
        str(so.get("measure", so) if isinstance(so, dict) else so) for so in sec_list
    )
    prim_clin, prim_surr = classify_outcome_domains(prim_measures)
    sec_clin, _ = classify_outcome_domains(sec_measures)

    pk_primary_hit = bool(
        (PK_BIOMARKER_DESIGN_RE.search(prim_measures) and not prim_clin)
        or (PK_BIOMARKER_DESIGN_RE.search(title_cond_text) and not prim_clin)
    )
    phases_upper = {str(p).upper() for p in (study.get("phases") or []) if str(p).strip()}
    exclusively_phase1 = bool(phases_upper) and phases_upper.issubset({"EARLY_PHASE1", "PHASE1"})

    if healthy_flag or healthy_text_hit or pk_primary_hit or (exclusively_phase1 and not (prim_clin & {"mortality_os", "oncology_progression"})):
        design_suitability = "non_efficacy_pk_or_healthy"
        design_cap = 0.20
        design_detail = (
            "该研究为健康受试者或药代动力学/生物等效性/早期药理研究，不具备临床疾病结局疗效可比性。"
            if (healthy_flag or healthy_text_hit or pk_primary_hit)
            else "该研究仅为早期 I 期探索试验且主要终点非生存结局，不可作为临床疗效目标试验。"
        )
    elif exclusively_phase1:
        design_suitability = "phase1_exploratory"
        design_cap = 0.30
        design_detail = "该研究为 I 期探索性试验，样本与随访设计受限（上限 30%）。"
    else:
        design_suitability = "clinical_efficacy_rct"
        design_cap = 1.0
        design_detail = "随机对照临床研究设计。"

    # --- 2. Population / Indication Domain Alignment (`population_alignment`) ---
    target_cond_clean = (target_condition or "").strip()
    target_pop_domains = extract_condition_domains(target_cond_clean)
    study_pop_domains = extract_condition_domains(pos_pop_text)
    missing_pop_zh: list[str] = []

    if healthy_flag or healthy_text_hit:
        population_alignment = "mismatched"
        population_cap = 0.25
        pop_pill_zh = "目标人群: 健康受试者"
        pop_tone = "bad"
        pop_detail = "该试验招募健康受试者，非目标临床疾病人群。"
    elif not target_cond_clean:
        population_alignment = "unspecified"
        population_cap = 1.0
        pop_pill_zh = "目标人群: 成人临床人群"
        pop_tone = "ok"
        pop_detail = "未限定额外疾病亚型约束。"
    elif target_pop_domains:
        missing_doms = target_pop_domains - study_pop_domains
        matched_doms = target_pop_domains & study_pop_domains
        if not missing_doms:
            population_alignment = "matched"
            population_cap = 1.0
            pop_pill_zh = "目标人群: 一致"
            pop_tone = "ok"
            matched_zh = " + ".join(POPULATION_DOMAIN_ZH.get(d, d) for d in sorted(matched_doms))
            pop_detail = f"目标适应证人群一致（{matched_zh}）。"
        elif matched_doms:
            population_alignment = "partial"
            population_cap = 0.55
            missing_pop_zh = [POPULATION_DOMAIN_ZH.get(d, d) for d in sorted(missing_doms)]
            pop_pill_zh = f"目标人群: 缺关键适应证({'/'.join(missing_pop_zh)})"
            pop_tone = "warn"
            pop_detail = (
                f"人群仅部分重叠：缺少关键适应证/合并症「{'、'.join(missing_pop_zh)}」（匹配度上限 55%）。"
            )
        else:
            population_alignment = "mismatched"
            population_cap = 0.25
            missing_pop_zh = [POPULATION_DOMAIN_ZH.get(d, d) for d in sorted(target_pop_domains)]
            pop_pill_zh = "目标人群: 适应证不符"
            pop_tone = "bad"
            pop_detail = f"试验纳排人群未覆盖目标适应证「{'、'.join(missing_pop_zh)}」（上限 25%）。"
    else:
        # Generic normalized keyword fallback for unlisted rare conditions
        kw_tokens = [
            w.lower()
            for w in re.findall(r"[A-Za-z]{4,}|[\u4e00-\u9fff]{2,}", target_cond_clean)
            if w.lower() not in {"patients", "with", "adult", "adults", "disease", "syndrome", "患者", "成人", "合并"}
        ]
        pos_low = pos_pop_text.lower()
        if not kw_tokens or any(k in pos_low for k in kw_tokens):
            population_alignment = "matched"
            population_cap = 1.0
            pop_pill_zh = "目标人群: 一致"
            pop_tone = "ok"
            pop_detail = "目标疾病关键词与试验纳入人群匹配。"
        else:
            population_alignment = "mismatched"
            population_cap = 0.25
            pop_pill_zh = "目标人群: 适应证不符"
            pop_tone = "bad"
            pop_detail = f"试验纳入人群未命中目标疾病特征「{target_cond_clean[:36]}」（上限 25%）。"

    # --- 3. Primary Outcome / Endpoint Domain Alignment (`outcome_alignment`) ---
    raw_target_out = (target_outcome or "").strip()
    is_default_generic_idea_out = raw_target_out in ("", "主要临床终点事件（时间至事件结局）")
    target_clin, target_surr = classify_outcome_domains("" if is_default_generic_idea_out else raw_target_out)
    # Minimal mock dicts with generic placeholder measure ("Primary Clinical Endpoint" / "Composite") and zero surrogate terms
    prim_low = prim_measures.strip().lower()
    is_minimal_clinical_placeholder = (
        not prim_measures.strip()
        or prim_low in ("primary clinical endpoint", "composite", "clinical composite")
    ) and not prim_surr

    surr_labels_zh = [SURROGATE_OUTCOME_ZH.get(s, s) for s in sorted(prim_surr)]
    surr_short = "/".join(surr_labels_zh[:2]) or "替代指标"

    if design_suitability == "non_efficacy_pk_or_healthy" or ("pk_safety_only" in prim_surr and not prim_clin):
        outcome_alignment = "surrogate_or_mismatched"
        outcome_cap = 0.48
        out_pill_zh = "主要终点: 药代指标"
        out_tone = "bad"
        out_detail = f"该试验主要终点为药代动力学或早期安全性指标（{prim_measures[:60] or 'PK'}），非临床事件终点。"
    elif target_clin:
        if (target_clin & prim_clin) or is_minimal_clinical_placeholder:
            outcome_alignment = "matched"
            outcome_cap = 1.0
            out_pill_zh = "主要终点: 临床终点一致"
            out_tone = "ok"
            out_detail = f"主要终点与目标临床硬终点一致（{prim_measures[:72] or '临床复合终点'}）。"
        elif target_clin & sec_clin:
            outcome_alignment = "secondary_only"
            outcome_cap = 0.65
            out_pill_zh = "主要终点: 仅次要终点匹配"
            out_tone = "warn"
            out_detail = f"主要终点未直接覆盖目标结局，仅次要终点含目标临床事件（上限 65%）。"
        else:
            outcome_alignment = "surrogate_or_mismatched"
            outcome_cap = 0.48
            if prim_surr:
                out_pill_zh = f"主要终点: 替代指标({surr_short})"
                out_detail = (
                    f"主要终点为短期实验室/替代生物标志物「{surr_short}」（{prim_measures[:68]}），"
                    "与目标临床事件终点不符（硬上限 48%）。"
                )
            else:
                out_pill_zh = "主要终点: 终点领域偏离"
                out_detail = f"该试验主要终点（{prim_measures[:68]}）与目标临床结局领域不一致（硬上限 48%）。"
            out_tone = "bad"
    elif target_surr and not is_cohort_mode:
        if target_surr & prim_surr:
            outcome_alignment = "matched"
            outcome_cap = 1.0
            out_pill_zh = "主要终点: 指标一致"
            out_tone = "ok"
            out_detail = f"主要终点匹配所查询指标（{prim_measures[:68]}）。"
        else:
            outcome_alignment = "surrogate_or_mismatched"
            outcome_cap = 0.48
            out_pill_zh = "主要终点: 终点偏离"
            out_tone = "bad"
            out_detail = f"主要终点（{prim_measures[:68]}）与目标终点不一致（上限 48%）。"
    elif is_cohort_mode:
        # Opaque user_cohort survival CSV (`event=0/1`, `time_days`): still a time-to-event clinical study; NEVER pass pure PK/surrogate lab trials!
        if prim_surr and not prim_clin:
            outcome_alignment = "surrogate_or_mismatched"
            outcome_cap = 0.48
            out_pill_zh = f"主要终点: 替代指标({surr_short})"
            out_tone = "bad"
            out_detail = f"当前队列为生存时间至事件分析，而该试验主要终点仅为短期替代指标「{surr_short}」（上限 48%）。"
        else:
            outcome_alignment = "unspecified_target_outcome"
            outcome_cap = 1.0
            out_pill_zh = "主要终点: 临床事件终点"
            out_tone = "ok"
            out_detail = f"试验具备临床事件主要终点（{prim_measures[:68] or '临床事件'}）。"
    else:
        # Idea mode where user left primary outcome unspecified
        if prim_surr and not prim_clin:
            outcome_alignment = "surrogate_or_mismatched"
            outcome_cap = 0.48
            out_pill_zh = f"主要终点: 替代指标({surr_short})"
            out_tone = "bad"
            out_detail = f"该试验主要终点仅为短期实验室替代指标「{surr_short}」，非临床硬终点（上限 48%）。"
        else:
            outcome_alignment = "unspecified_clinical"
            outcome_cap = 1.0
            out_pill_zh = "主要终点: 临床事件终点"
            out_tone = "ok"
            out_detail = f"临床疗效事件终点（{prim_measures[:68] or '临床终点'}）。"

    # --- 4. Follow-Up Horizon Adequacy (`followup_alignment`) ---
    prim_tfs = [str(po.get("timeFrame", "")) for po in prim_list if isinstance(po, dict) and po.get("timeFrame")]
    sec_tfs = [str(so.get("timeFrame", "")) for so in sec_list if isinstance(so, dict) and so.get("timeFrame")]
    parsed_days, is_ev_driven, disp_tf = _parse_outcome_timeframe_days(prim_tfs or sec_tfs)
    eff_target_fu = float(target_followup_days) if (target_followup_days is not None and float(target_followup_days) > 0) else (365.0 if is_cohort_mode else 0.0)

    if is_ev_driven and (parsed_days is None or parsed_days >= 180.0):
        followup_alignment = "adequate"
        followup_cap = 1.0
        fu_pill_zh = "随访窗口: 充足"
        fu_tone = "ok"
        fu_detail = f"事件驱动长期随访窗充足（{disp_tf}）。"
    elif parsed_days is None:
        followup_alignment = "unparsed"
        followup_cap = 1.0
        fu_pill_zh = "随访窗口: 充足"
        fu_tone = "ok"
        fu_detail = f"随访时间窗：{disp_tf}。"
    elif eff_target_fu >= 365.0 and parsed_days < 90.0:
        followup_alignment = "severely_short"
        followup_cap = 0.35
        fu_pill_zh = f"随访窗口: 过短({disp_tf})"
        fu_tone = "bad"
        fu_detail = f"主要终点随访窗极短（{disp_tf} ≈ {parsed_days:.0f} 天 < 90 天，远短于目标长期随访，上限 35%）。"
    elif eff_target_fu > 0.0 and parsed_days < 0.5 * eff_target_fu:
        followup_alignment = "shorter_than_target"
        followup_cap = 0.62
        fu_pill_zh = f"随访窗口: 偏短({disp_tf})"
        fu_tone = "warn"
        fu_detail = f"主要终点随访窗偏短（{disp_tf} ≈ {parsed_days:.0f} 天，短于目标随访窗 {eff_target_fu:.0f} 天的一半，上限 62%）。"
    else:
        followup_alignment = "adequate"
        followup_cap = 1.0
        fu_pill_zh = "随访窗口: 充足"
        fu_tone = "ok"
        fu_detail = f"主要终点随访窗充足（{disp_tf}）。"

    # --- Arm pill formatting ---
    comp_arms_joined = " ".join(str(x) for x in (study.get("comparator_arms") or [])).lower()
    if arm_status in ("exact_dual_arm", "single_intervention_matched"):
        arm_pill_zh = "双臂干预: 一致"
        arm_tone = "ok"
    elif arm_status == "class_dual_arm":
        arm_pill_zh = "双臂干预: 同类/背景差异"
        arm_tone = "warn"
    elif arm_status == "single_arm_or_comparator_mismatch":
        arm_pill_zh = "双臂干预: 安慰剂参考" if "placebo" in comp_arms_joined or "安慰剂" in comp_arms_joined else "双臂干预: 错配"
        arm_tone = "bad"
    elif arm_status == "unmatched_interventions":
        arm_pill_zh = "双臂干预: 错配"
        arm_tone = "bad"
    else:
        arm_pill_zh = "双臂干预: 参考框架"
        arm_tone = "ok"

    # --- 5. Overall Protocol Alignment Verdict & Hard Score Cap ---
    arms_ok = arm_status == "exact_dual_arm" or (
        not is_cohort_mode and arm_status in ("single_intervention_matched", "unspecified")
    )
    pop_ok = population_alignment in ("matched", "unspecified")
    out_ok = outcome_alignment in ("matched", "unspecified_clinical", "unspecified_target_outcome")
    fu_ok = followup_alignment in ("adequate", "unparsed")

    diverged_dimensions: list[str] = []
    if design_suitability != "clinical_efficacy_rct":
        diverged_dimensions.append(f"design_suitability={design_suitability} ({design_detail})")
    if not arms_ok:
        diverged_dimensions.append(f"arm_alignment_status={arm_status} ({arm_audit['arm_alignment_detail']})")
    if not pop_ok:
        diverged_dimensions.append(f"population_alignment={population_alignment} ({pop_detail})")
    if not out_ok:
        diverged_dimensions.append(f"outcome_alignment={outcome_alignment} ({out_detail})")
    if not fu_ok:
        diverged_dimensions.append(f"followup_alignment={followup_alignment} ({fu_detail})")

    effective_cap = round(min(arm_cap, design_cap, population_cap, outcome_cap, followup_cap), 2)

    if design_suitability == "non_efficacy_pk_or_healthy":
        overall_status = "non_efficacy_excluded"
        overall_rank = OVERALL_ALIGNMENT_RANKS["non_efficacy_excluded"]
        overall_label = "健康受试者/药代试验 · 不可直接对标"
    elif not arms_ok:
        overall_status = "arm_mismatch_reference"
        overall_rank = 3 if arm_status == "class_dual_arm" else OVERALL_ALIGNMENT_RANKS["arm_mismatch_reference"]
        overall_label = arm_audit["arm_alignment_label"]
    elif design_suitability == "clinical_efficacy_rct" and pop_ok and out_ok and fu_ok:
        overall_status = "protocol_aligned"
        overall_rank = OVERALL_ALIGNMENT_RANKS["protocol_aligned"]
        overall_label = "四维方案一致 · 推荐对标"
    else:
        overall_status = "partial_protocol_reference"
        overall_rank = OVERALL_ALIGNMENT_RANKS["partial_protocol_reference"]
        overall_label = "双臂一致但人群/终点偏离 · 仅供参考"

    is_recommended = overall_status == "protocol_aligned"
    requires_protocol_confirmation = overall_status != "protocol_aligned"

    dimension_audit = {
        "design": {
            "status": design_suitability,
            "score_cap": design_cap,
            "pill_zh": "研究设计: 临床疗效RCT" if design_suitability == "clinical_efficacy_rct" else "研究设计: 药代/健康人/早期",
            "tone": "ok" if design_suitability == "clinical_efficacy_rct" else "bad",
            "detail_zh": design_detail,
        },
        "arms": {
            "status": arm_status,
            "score_cap": arm_cap,
            "pill_zh": arm_pill_zh,
            "tone": arm_tone,
            "detail_zh": arm_audit["arm_alignment_detail"],
        },
        "population": {
            "status": population_alignment,
            "score_cap": population_cap,
            "pill_zh": pop_pill_zh,
            "tone": pop_tone,
            "detail_zh": pop_detail,
            "missing_domains_zh": missing_pop_zh,
        },
        "outcome": {
            "status": outcome_alignment,
            "score_cap": outcome_cap,
            "pill_zh": out_pill_zh,
            "tone": out_tone,
            "detail_zh": out_detail,
        },
        "followup": {
            "status": followup_alignment,
            "score_cap": followup_cap,
            "pill_zh": fu_pill_zh,
            "tone": fu_tone,
            "detail_zh": fu_detail,
            "parsed_primary_days": parsed_days,
        },
    }

    return {
        **arm_audit,
        "design_suitability": design_suitability,
        "population_alignment": population_alignment,
        "outcome_alignment": outcome_alignment,
        "followup_alignment": followup_alignment,
        "overall_alignment_status": overall_status,
        "overall_alignment_rank": overall_rank,
        "overall_alignment_label": overall_label,
        "effective_score_cap": effective_cap,
        "dimension_audit": dimension_audit,
        "diverged_dimensions": diverged_dimensions,
        "requires_protocol_confirmation": requires_protocol_confirmation,
        "is_recommended": is_recommended,
    }


def verify_and_enrich_target_rct(
    target_rct: dict | None,
    *,
    intervention: str = "",
    comparator: str = "",
    target_condition: str = "",
    target_outcome: str = "",
    target_followup_days: float | None = None,
    project_mode: str = "",
    confirm_arm_mismatch: bool = False,
    confirm_protocol_mismatch: bool = False,
) -> dict | None:
    """Fail-closed server-side 4D PICO-T alignment gate before drafting or launching a TTE run against `target_rct`."""
    if not target_rct or not isinstance(target_rct, dict) or not target_rct.get("nct_id"):
        return target_rct

    audit = audit_study_pico4d_alignment(
        target_rct,
        intervention=intervention,
        comparator=comparator,
        target_condition=target_condition,
        target_outcome=target_outcome,
        target_followup_days=target_followup_days,
        project_mode=project_mode,
    )
    status = audit["arm_alignment_status"]
    overall_status = audit["overall_alignment_status"]
    confirmed = bool(confirm_arm_mismatch or confirm_protocol_mismatch)
    enriched = {
        **target_rct,
        **audit,
    }
    if "match_score" in enriched and isinstance(enriched["match_score"], (int, float)):
        raw_sc = float(enriched["match_score"])
        norm_sc = raw_sc / 100.0 if raw_sc > 1.0 else raw_sc
        enriched["match_score"] = round(min(norm_sc, audit["effective_score_cap"]), 2)
        enriched["match_score_pct"] = int(round(enriched["match_score"] * 100))

    if audit["requires_mismatch_confirmation"]:
        if not confirmed:
            raise ContractError(
                "ARM_MISMATCH_CONFIRMATION_REQUIRED",
                (
                    f"ARM_MISMATCH_CONFIRMATION_REQUIRED: 目标 RCT（{target_rct['nct_id']}）分组与当前研究双臂不一致（status={status}）："
                    f"当前研究对比 [{audit['expected_arms_summary']}]，而 {target_rct['nct_id']} 实际分组为 "
                    f"[{audit['trial_arms_summary']}]。Target RCT arms do not match study treatment vs comparator; "
                    "pass confirm_arm_mismatch=true only if intentionally borrowing eligibility/follow-up structure."
                ),
            )
        arm_warn = (
            f"对照臂不一致参考声明（研究者已显式确认 confirm_arm_mismatch=true）：当前研究对比「{audit['expected_arms_summary']}」，"
            f"所引用的注册试验 {target_rct['nct_id']} 实际分组为「{audit['trial_arms_summary']}」"
            f"（arm_alignment_status={status}），仅借用其基线纳排与终点随访时间窗作为方案框架，不代表两项研究干预对照相同。"
        )
        enriched["arm_alignment_warning"] = arm_warn
        enriched["protocol_alignment_warning"] = arm_warn
    elif overall_status != "protocol_aligned":
        div_summary = "；".join(audit["diverged_dimensions"])
        if not confirmed:
            raise ContractError(
                "PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED",
                (
                    f"PROTOCOL_MISMATCH_CONFIRMATION_REQUIRED: 目标 RCT（{target_rct['nct_id']}）虽干预药物匹配，"
                    f"但在 PICO-T 关键方案维度上与当前研究存在偏离（overall_alignment_status={overall_status}）：{div_summary}。"
                    "Pass confirm_protocol_mismatch=true (or confirm_arm_mismatch=true) only if intentionally using this study as a partial protocol reference."
                ),
            )
        proto_warn = (
            f"四维 PICO-T 方案偏离参考声明（研究者已显式确认）：所引用的注册试验 {target_rct['nct_id']} "
            f"（overall_alignment_status={overall_status}）在人群、研究设计、主要终点或随访窗上与当前研究存在偏离（{div_summary}），"
            "仅作为局部方案参考框架，绝不可直接等同为同适应证同终点目标试验对标。"
        )
        enriched["arm_alignment_warning"] = ""
        enriched["protocol_alignment_warning"] = proto_warn
    else:
        enriched["arm_alignment_warning"] = ""
        enriched["protocol_alignment_warning"] = ""

    return enriched


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
    in_exclusion = False
    for raw_line in criteria_text.splitlines():
        if EXCLUSION_HEADING_RE.search(raw_line):
            in_exclusion = True
            continue
        cleaned = re.sub(r"^[\s*•\-0-9.)]+", "", raw_line).strip()
        if len(cleaned) < 10 or cleaned.lower() in ("inclusion criteria:", "exclusion criteria:", "inclusion criteria", "exclusion criteria"):
            continue
        if in_exclusion and not cleaned.startswith("排除") and not re.match(r"(?i)^exclu", cleaned):
            cleaned = f"排除: {cleaned}"
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

    # Map study-level interventions back to their declared armGroupLabels when armGroups omitted interventionNames (NEVER fabricating new arms!)
    interventions = []
    linked_intr_by_arm_label: dict[str, list[str]] = {}
    for intr in arms_mod.get("interventions") or []:
        if isinstance(intr, dict) and intr.get("name"):
            itype = str(intr.get("type", "DRUG")).title()
            full_iname = f"{itype}: {intr['name']}"
            interventions.append(full_iname)
            for glbl in intr.get("armGroupLabels") or []:
                gl_key = str(glbl).strip().lower()
                if gl_key:
                    linked_intr_by_arm_label.setdefault(gl_key, []).append(full_iname)

    exp_arms: list[str] = []
    comp_arms: list[str] = []
    other_arms: list[str] = []
    arm_groups_detail: list[dict] = []
    for arm in arms_mod.get("armGroups") or []:
        if not isinstance(arm, dict):
            continue
        label = str(arm.get("label", "")).strip()
        atype = str(arm.get("type", "")).upper()
        desc = str(arm.get("description", "")).strip()
        inames = [str(x).strip() for x in (arm.get("interventionNames") or []) if str(x).strip()]
        if not inames and label.lower() in linked_intr_by_arm_label:
            inames = list(linked_intr_by_arm_label[label.lower()])
        if not label and not inames:
            continue
        disp = label
        if inames:
            clean_inames = ", ".join(x.split(": ", 1)[-1] for x in inames)
            if not disp:
                disp = clean_inames
            elif clean_inames.lower() not in disp.lower():
                disp = f"{disp} ({clean_inames})"
        arm_groups_detail.append(
            {
                "label": disp,
                "type": atype,
                "description": desc[:240],
                "intervention_names": inames,
            }
        )
        if "EXPERIMENTAL" in atype:
            exp_arms.append(disp)
        elif any(k in atype for k in ("ACTIVE_COMPARATOR", "PLACEBO_COMPARATOR", "SHAM_COMPARATOR", "NO_INTERVENTION")):
            comp_arms.append(disp)
        else:
            other_arms.append(disp)

    if not exp_arms and other_arms:
        exp_arms.append(other_arms[0])
    if not comp_arms and len(other_arms) >= 2:
        comp_arms.append(other_arms[1])
    elif not exp_arms and len(comp_arms) >= 2:
        # Handle two ACTIVE_COMPARATOR arms (e.g. head-to-head active comparator trials like NCT06642272!)
        exp_arms = [comp_arms[0]]
        comp_arms = comp_arms[1:]
    # IMPORTANT: NEVER fabricate separate experimental/comparator arms from study-level `interventions` when len(arm_groups_detail) < 2!

    primary_outcomes = []
    for po in (out_mod.get("primaryOutcomes") or [])[:3]:
        if isinstance(po, dict) and po.get("measure"):
            primary_outcomes.append(
                {
                    "measure": str(po.get("measure", "")).strip(),
                    "timeFrame": str(po.get("timeFrame", "")).strip() or "Not specified",
                }
            )
    secondary_outcomes = []
    for so in (out_mod.get("secondaryOutcomes") or [])[:5]:
        if isinstance(so, dict) and so.get("measure"):
            secondary_outcomes.append(
                {
                    "measure": str(so.get("measure", "")).strip(),
                    "timeFrame": str(so.get("timeFrame", "")).strip() or "Not specified",
                }
            )

    first_tf = primary_outcomes[0]["timeFrame"] if primary_outcomes else ""
    horizons, admin_end = parse_timeframe_to_horizons(first_tf)
    raw_criteria_str = str(elig_mod.get("eligibilityCriteria", ""))

    canonical_bytes = raw_bytes if raw_bytes is not None else json.dumps(study, sort_keys=True).encode("utf-8")
    return {
        "nct_id": nct_id,
        "record_url": f"https://clinicaltrials.gov/study/{nct_id}",
        "brief_title": brief_title,
        "official_title": official_title,
        "study_type": str(design.get("studyType", "INTERVENTIONAL")).upper(),
        "phases": phases or ["INTERVENTIONAL"],
        "overall_status": str(status.get("overallStatus", "UNKNOWN")),
        "enrollment_count": enrollment_count,
        "allocation": allocation,
        "healthy_volunteers": bool(elig_mod.get("healthyVolunteers") is True),
        "inclusion_text": extract_positive_inclusion_text(raw_criteria_str)[:3000],
        "conditions": conditions,
        "experimental_arms": exp_arms,
        "comparator_arms": comp_arms,
        "arm_groups_detail": arm_groups_detail,
        "interventions": interventions,
        "eligibility_summary": {
            "min_age": str(elig_mod.get("minimumAge", "18 Years")),
            "max_age": str(elig_mod.get("maximumAge", "N/A")),
            "sex": str(elig_mod.get("sex", "ALL")),
            "highlights": _extract_eligibility_highlights(raw_criteria_str),
        },
        "primary_outcomes": primary_outcomes or [{"measure": "Primary Clinical Endpoint", "timeFrame": "12 months"}],
        "secondary_outcomes": secondary_outcomes,
        "parsed_horizon_days": horizons,
        "parsed_admin_end_day": admin_end,
        "source": source,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "receipt_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
    }


def _score_and_audit_study(
    study: dict,
    query_tokens: list[str],
    *,
    intervention: str = "",
    comparator: str = "",
    target_condition: str = "",
    target_outcome: str = "",
    target_followup_days: float | None = None,
    project_mode: str = "",
) -> dict:
    """Compute deterministic relevance score in [0.01, 0.99], audit 4D PICO-T alignment, and apply hard score cap."""
    haystack_title = f"{study['brief_title']} {study['official_title']}".lower()
    haystack_arms = " ".join(
        study.get("experimental_arms", []) + study.get("comparator_arms", []) + study.get("interventions", [])
    ).lower()
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

    audit = audit_study_pico4d_alignment(
        study,
        intervention=intervention,
        comparator=comparator,
        target_condition=target_condition,
        target_outcome=target_outcome,
        target_followup_days=target_followup_days,
        project_mode=project_mode,
    )
    if audit["arm_alignment_status"] == "exact_dual_arm":
        raw_pts += 22
    elif audit["arm_alignment_status"] == "class_dual_arm":
        raw_pts += 6
    if audit["population_alignment"] == "matched":
        raw_pts += 8
    if audit["outcome_alignment"] == "matched":
        raw_pts += 10

    uncapped_score = min(0.98, max(0.20, (42 + raw_pts * 0.52) / 100.0))
    capped_score = round(min(uncapped_score, float(audit["effective_score_cap"])), 2)

    reasons: list[str] = []
    if audit["overall_alignment_status"] != "protocol_aligned" and audit["diverged_dimensions"]:
        reasons.append(
            f"四维 PICO-T 核验提示（{audit['overall_alignment_label']}）："
            + "；".join(d.split("(", 1)[-1].rstrip(")") for d in audit["diverged_dimensions"][:3])
        )
    reasons.append(audit["arm_alignment_detail"])
    if matched_arms:
        reasons.append(f"试验干预/对照文本命中关键词（{', '.join(dict.fromkeys(matched_arms[:3]))}）")
    if matched_conds and audit["population_alignment"] in ("matched", "unspecified"):
        reasons.append(f"目标适应证/疾病人群匹配（{', '.join(dict.fromkeys(matched_conds[:3]))}）")
    tf_preview = study["primary_outcomes"][0]["timeFrame"] if study.get("primary_outcomes") else "未注明"
    reasons.append(f"主要终点随访窗：{tf_preview}")

    return {
        **study,
        **audit,
        "match_score": capped_score,
        "match_score_pct": int(round(capped_score * 100)),
        "match_reasons": reasons,
        "ranking_mode": "deterministic_rules",
    }


def _enforce_alignment_caps_and_sort(candidates: list[dict]) -> list[dict]:
    """Post-LLM deterministic guardrail: clamp `match_score <= effective_score_cap`, lock `is_recommended`, and sort by 4D alignment rank first."""
    out: list[dict] = []
    for cand in candidates:
        item = dict(cand)
        status = str(item.get("arm_alignment_status", "unspecified"))
        arm_cap = float(item.get("arm_alignment_score_cap", ALIGNMENT_SCORE_CAPS.get(status, 1.0)))
        eff_cap = float(item.get("effective_score_cap", arm_cap))
        cap = min(arm_cap, eff_cap)
        raw_sc = float(item.get("match_score", 0.5))
        norm_sc = raw_sc / 100.0 if raw_sc > 1.0 else raw_sc
        clamped = round(max(0.01, min(norm_sc, cap)), 2)
        item["match_score"] = clamped
        item["match_score_pct"] = int(round(clamped * 100))
        item["arm_alignment_score_cap"] = arm_cap
        item["effective_score_cap"] = cap
        item["arm_alignment_rank"] = int(item.get("arm_alignment_rank", ALIGNMENT_RANKS.get(status, 2)))
        overall_st = str(item.get("overall_alignment_status", ""))
        if overall_st:
            item["overall_alignment_rank"] = int(
                item.get("overall_alignment_rank", OVERALL_ALIGNMENT_RANKS.get(overall_st, 2))
            )
            item["is_recommended"] = overall_st == "protocol_aligned"
            item["requires_protocol_confirmation"] = overall_st != "protocol_aligned"
        else:
            item["overall_alignment_rank"] = item["arm_alignment_rank"]
            item["is_recommended"] = status in ("exact_dual_arm", "single_intervention_matched", "unspecified")
            item["requires_protocol_confirmation"] = not item["is_recommended"]
        out.append(item)

    out.sort(
        key=lambda c: (
            int(c.get("overall_alignment_rank", 0)),
            int(c.get("arm_alignment_rank", 0)),
            float(c.get("match_score", 0.0)),
            int(c.get("enrollment_count") or 0),
        ),
        reverse=True,
    )
    return out


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
                    card["match_score"] = 0.99
                    card["match_score_pct"] = 99
                    card["is_recommended"] = True
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
                "match_score": 0.99,
                "match_score_pct": 99,
                "is_recommended": True,
                "match_reasons": [f"离线公开精选目录精确匹配 {cleaned}（当前外网不可达）"],
                "ranking_mode": "exact_nct_lookup",
            }
            return card

    raise ContractError(
        "CTGOV_UNAVAILABLE",
        f"当前无法连接 ClinicalTrials.gov 拉取 {cleaned}，且该编号不在本地离线兜底目录中。",
    )


def _resolve_effective_arms(
    query: str,
    intervention: str,
    comparator: str,
    context_summary: dict | None,
) -> tuple[str, str]:
    ctx = context_summary if isinstance(context_summary, dict) else {}
    eff_intr = (
        intervention.strip()
        or str(ctx.get("active_arm_label", "")).strip()
        or str(ctx.get("intervention_query_en", "")).strip()
        or str(ctx.get("intervention", "")).strip()
    )
    eff_comp = (
        comparator.strip()
        or str(ctx.get("comparator_arm_label", "")).strip()
        or str(ctx.get("comparator_query_en", "")).strip()
    )
    if not eff_comp and ctx.get("comparator_explicit") is True:
        eff_comp = str(ctx.get("comparator", "")).strip()

    # If caller passed a query containing two drugs (e.g. 'Dapagliflozin Sitagliptin Heart Failure' or '达格列净 vs 西格列汀')
    if query and not eff_comp and ctx.get("comparator_explicit") is not False:
        vs_split = re.split(r"\b(?:vs\.?|versus)\b|对比|对照", query, maxsplit=1, flags=re.IGNORECASE)
        if len(vs_split) == 2:
            if not eff_intr:
                eff_intr = vs_split[0].strip()
            eff_comp = vs_split[1].strip()
        else:
            # Extract known lexicon drugs from query so random condition nouns aren't mistaken for drugs
            known_in_q: list[str] = []
            q_low = query.lower()
            for cid, en_q, _, _, is_gen, aliases in DRUG_CLASS_LEXICON:
                if is_gen:
                    continue
                if any(a.lower() in q_low for a in aliases) and en_q not in known_in_q:
                    known_in_q.append(en_q)
            if not eff_intr and len(known_in_q) == 1:
                eff_intr = known_in_q[0]
            elif len(known_in_q) >= 2:
                if not eff_intr:
                    eff_intr = known_in_q[0]
                other_drugs = [d for d in known_in_q if d.lower() != eff_intr.lower()]
                if other_drugs:
                    eff_comp = other_drugs[0]
            elif not eff_intr:
                q_spec = normalize_intervention_spec(query)
                if q_spec["en_terms"]:
                    eff_intr = q_spec["en_terms"][0]
    return eff_intr, eff_comp


def search_candidate_rcts(
    query: str,
    *,
    condition: str = "",
    intervention: str = "",
    comparator: str = "",
    outcome: str = "",
    target_followup_days: float | None = None,
    project_mode: str = "",
    context_summary: dict | None = None,
    forbidden_values: tuple[str, ...] = (),
    max_results: int = 6,
    timeout: float = DEFAULT_CT_TIMEOUT,
) -> dict:
    """Search ClinicalTrials.gov API v2 with bilingual normalization, 4D PICO-T auditing, multi-pass landmark backfill, and hard score caps."""
    clean_query = (query or "").strip()
    ctx_d = context_summary if isinstance(context_summary, dict) else {}
    eff_intr, eff_comp = _resolve_effective_arms(clean_query, intervention, comparator, ctx_d)
    eff_target_cond = (
        condition.strip()
        or str(ctx_d.get("condition_query_en", "")).strip()
        or str(ctx_d.get("population", "")).strip()
    )
    if not eff_target_cond:
        eff_target_cond = translate_condition_to_english(clean_query)
    eff_target_out = (
        outcome.strip()
        or str(ctx_d.get("outcome_query_en", "")).strip()
        or str(ctx_d.get("primary_outcome", "")).strip()
    )
    eff_target_fu: float | None = target_followup_days
    if eff_target_fu is None and ctx_d.get("target_followup_days") is not None:
        try:
            eff_target_fu = float(ctx_d["target_followup_days"])
        except (TypeError, ValueError):
            eff_target_fu = None

    nct_hit = NCT_RE.search(clean_query)
    if nct_hit and len(clean_query) <= 20:
        single = fetch_rct_by_nct(nct_hit.group(1), timeout=timeout)
        audit = audit_study_pico4d_alignment(
            single,
            intervention=eff_intr,
            comparator=eff_comp,
            target_condition=eff_target_cond,
            target_outcome=eff_target_out,
            target_followup_days=eff_target_fu,
            project_mode=project_mode,
        )
        single = _enforce_alignment_caps_and_sort([{**single, **audit}])[0]
        return {
            "source": single["source"],
            "query_used": nct_hit.group(1).upper(),
            "ranking_mode": "exact_nct_lookup",
            "candidates": [single],
        }

    intr_spec = normalize_intervention_spec(eff_intr)
    comp_spec = normalize_intervention_spec(eff_comp)
    intr_en = intr_spec["en_terms"][0] if intr_spec["en_terms"] else translate_query_to_english(eff_intr)
    comp_en = comp_spec["en_terms"][0] if comp_spec["en_terms"] else translate_query_to_english(eff_comp)
    cond_en = translate_condition_to_english(eff_target_cond) or translate_condition_to_english(clean_query)
    en_query = translate_query_to_english(clean_query)
    out_clin_domains, _ = classify_outcome_domains(eff_target_out)
    out_query_en = ""
    if "hf_cv_composite" in out_clin_domains:
        out_query_en = "heart failure OR cardiovascular death OR MACE"
    elif "oncology_progression" in out_clin_domains or "mortality_os" in out_clin_domains:
        out_query_en = "progression-free survival OR overall survival"
    elif "renal_hard_endpoint" in out_clin_domains:
        out_query_en = "kidney failure OR eGFR OR renal death"

    raw_tokens = [
        tok
        for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", f"{en_query} {cond_en} {intr_en} {comp_en}")
        if tok.lower() not in {"and", "the", "for", "with", "versus", "randomized", "controlled", "trial", "inhibitor"}
    ]
    seen_tok_lower: set[str] = set()
    tokens: list[str] = []
    for tok in raw_tokens:
        if tok.lower() not in seen_tok_lower:
            seen_tok_lower.add(tok.lower())
            tokens.append(tok)

    live_by_nct: dict[str, dict] = {}
    live_error: str | None = None
    query_used_display = en_query or " ".join(p for p in (cond_en, intr_en, comp_en) if p).strip()

    if os.environ.get("EMULATRX_OFFLINE_RCT_CATALOG") != "1":
        base_filter = {
            "filter.advanced": "AREA[StudyType]INTERVENTIONAL AND AREA[DesignAllocation]RANDOMIZED",
            "fields": ",".join(STUDY_FIELDS),
            "pageSize": 15,
        }
        search_attempts: list[dict[str, str | int]] = []
        # 1. Dual-arm head-to-head query FIRST (`intervention AND comparator`) so exact 4D head-to-head trials surface at top
        if intr_en and comp_en and intr_en.lower() != comp_en.lower():
            dual_intr_expr = f"{intr_en} AND {comp_en}"
            query_used_display = f"{dual_intr_expr}" + (f" | {cond_en}" if cond_en else "")
            if cond_en:
                search_attempts.append({**base_filter, "query.cond": cond_en, "query.intr": dual_intr_expr})
            search_attempts.append({**base_filter, "query.intr": dual_intr_expr})

        # 2. Landmark primary-indication + clinical-outcome query (`intr + cond + outc`) so landmark trials like DAPA-HF appear alongside demoted biomarker H2H trials
        if intr_en and cond_en and out_query_en:
            search_attempts.append(
                {**base_filter, "query.cond": cond_en, "query.intr": intr_en, "query.outc": out_query_en}
            )
        if cond_en or intr_en:
            p_single: dict[str, str | int] = dict(base_filter)
            if cond_en:
                p_single["query.cond"] = cond_en
            if intr_en:
                p_single["query.intr"] = intr_en
            elif en_query:
                p_single["query.term"] = en_query
            search_attempts.append(p_single)

        # 3. Broad English term fallback
        search_attempts.append(
            {
                **base_filter,
                "query.term": " ".join(p for p in (intr_en, comp_en, cond_en) if p).strip()
                or en_query
                or "randomized phase 3",
            }
        )

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
                        if not norm["nct_id"] or norm["nct_id"] in live_by_nct:
                            continue
                        scored = _score_and_audit_study(
                            norm,
                            tokens,
                            intervention=eff_intr,
                            comparator=eff_comp,
                            target_condition=eff_target_cond,
                            target_outcome=eff_target_out,
                            target_followup_days=eff_target_fu,
                            project_mode=project_mode,
                        )
                        live_by_nct[norm["nct_id"]] = scored
                    # Stop early ONLY once we have enough full 4D `protocol_aligned` matches (so demoted ketone/PK H2H trials don't block landmark HF outcome backfill!)
                    aligned_cnt = sum(
                        1
                        for v in live_by_nct.values()
                        if v.get("overall_alignment_status") == "protocol_aligned"
                    )
                    if aligned_cnt >= max_results or (not comp_en and aligned_cnt >= 1 and len(live_by_nct) >= max_results):
                        break
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                live_error = type(exc).__name__
                break

    ctx_for_llm = dict(ctx_d)
    ctx_for_llm.setdefault("query", en_query)
    ctx_for_llm.setdefault("condition", cond_en)
    ctx_for_llm.setdefault("intervention", intr_en)
    if comp_en:
        ctx_for_llm.setdefault("comparator", comp_en)

    if live_by_nct:
        ordered_live = _enforce_alignment_caps_and_sort(list(live_by_nct.values()))
        top = ordered_live[: max(max_results, 8)]
        reranked, rank_mode = rerank_and_explain_rcts(
            top,
            ctx_for_llm,
            forbidden_values=forbidden_values,
        )
        final_live = _enforce_alignment_caps_and_sort(reranked)[:max_results]
        return {
            "source": "clinicaltrials_gov_api_v2",
            "source_label_zh": "实时联网检索 ClinicalTrials.gov API v2",
            "query_used": query_used_display or clean_query,
            "effective_intervention": eff_intr,
            "effective_comparator": eff_comp,
            "ranking_mode": rank_mode,
            "candidates": final_live,
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
        scored_fb = _score_and_audit_study(
            entry,
            tokens,
            intervention=eff_intr,
            comparator=eff_comp,
            target_condition=eff_target_cond,
            target_outcome=eff_target_out,
            target_followup_days=eff_target_fu,
            project_mode=project_mode,
        )
        scored_fb["match_reasons"] = [
            *scored_fb["match_reasons"],
            "注：当前来自内置公开 RCT 离线精选目录兜底（source=offline_curated_catalog_fallback）",
        ]
        fallback_list.append(scored_fb)

    ordered_fb = _enforce_alignment_caps_and_sort(fallback_list)
    reranked_fb, rank_mode_fb = rerank_and_explain_rcts(
        ordered_fb[: max(max_results, 8)],
        ctx_for_llm,
        forbidden_values=forbidden_values,
    )
    final_fb = _enforce_alignment_caps_and_sort(reranked_fb)[:max_results]
    return {
        "source": "offline_curated_catalog_fallback",
        "source_label_zh": "内置代表性公开 RCT 目录兜底（offline_curated_catalog_fallback）",
        "fallback_reason": live_error or "offline_or_no_live_results",
        "query_used": query_used_display or clean_query,
        "effective_intervention": eff_intr,
        "effective_comparator": eff_comp,
        "ranking_mode": rank_mode_fb,
        "candidates": final_fb,
    }
