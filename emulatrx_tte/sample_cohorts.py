"""Built-in realistic clinical header sample cohorts and clinical idea presets for interactive workbench demos.

Note: Although column headers, categorical encodings, and baseline confounding patterns mirror realistic
hospital EHR / oncology registry exports (without any `SYN-` prefix) so users can demonstrate the full
`user_cohort` ingestion, profiling, RCT matching, column mapping, and attrition pipeline out-of-the-box,
all rows are deterministically generated from fixed pseudo-random seeds and contain zero real patient records.
"""

from __future__ import annotations

import csv
import io
import math

import numpy as np
from scipy.special import expit


SAMPLE_COHORT_SPECS = {
    "cardiorenal_sglt2i": {
        "sample_id": "cardiorenal_sglt2i",
        "filename": "cardiorenal_dapagliflozin_vs_sitagliptin_1500.csv",
        "title_zh": "样表 1 · 心肾代谢住院/门诊专病队列（Dapagliflozin vs Sitagliptin，1,500 例）",
        "subtitle_zh": "真实医院导出列名风格（deid_patient_id = Hosp-2023-0001..），随访以「月」为单位，含年龄/eGFR 纳排空间、多分类心功能分级与非心血管竞争死亡",
        "domain_hint_zh": "2型糖尿病合并心肾高危/心衰人群，比较达格列净（Dapagliflozin）对比西格列汀（Sitagliptin）对心衰住院或心血管事件（主要结局）及非心血管竞争死亡的影响",
        "active_intervention_zh": "达格列净 (Dapagliflozin)",
        "active_intervention_en": "Dapagliflozin",
        "comparator_intervention_zh": "西格列汀 (Sitagliptin)",
        "comparator_intervention_en": "Sitagliptin",
        "condition_en": "Type 2 Diabetes Heart Failure",
        "row_count": 1500,
        "recommended_eligibility": [
            {"id": "inc_age_45", "kind": "inclusion", "variable": "age_years", "operator": ">=", "value": 45, "description": "纳入年龄 ≥ 45 岁成人患者"},
            {"id": "inc_egfr_25", "kind": "inclusion", "variable": "egfr_ml_min", "operator": ">=", "value": 25, "description": "纳入基线 eGFR ≥ 25 mL/min/1.73m²"},
        ],
    },
    "oncology_io_nsclc": {
        "sample_id": "oncology_io_nsclc",
        "filename": "oncology_nsclc_pembrolizumab_chemo_1200.csv",
        "title_zh": "样表 2 · 晚期非小细胞肺癌一线免疫联合队列（Pembrolizumab+Chemo vs 含铂双药，1,200 例）",
        "subtitle_zh": "真实肿瘤专病登记列名风格（case_no = ONC-880001..），随访以「天」为单位，含 ECOG / PD-L1 / 吸烟史多分类协变量与非肿瘤治疗无关竞争死亡",
        "domain_hint_zh": "晚期非小细胞肺癌（NSCLC）一线治疗，比较帕博利珠单抗联合含铂化疗（Pembrolizumab_Plus_Chemo）对比单纯含铂双药化疗（Platinum_Doublet_Chemo）的疾病进展/肿瘤死亡与非肿瘤竞争死亡",
        "active_intervention_zh": "帕博利珠单抗联合含铂化疗 (Pembrolizumab + Chemo)",
        "active_intervention_en": "Pembrolizumab",
        "comparator_intervention_zh": "含铂双药化疗 (Platinum Doublet Chemo)",
        "comparator_intervention_en": "Platinum Chemotherapy",
        "condition_en": "Non-Small Cell Lung Cancer",
        "row_count": 1200,
        "recommended_eligibility": [
            {"id": "inc_age_18", "kind": "inclusion", "variable": "age", "operator": ">=", "value": 40, "description": "纳入年龄 ≥ 40 岁患者"},
            {"id": "exc_ecog_2", "kind": "exclusion", "variable": "ecog_ps", "operator": ">=", "value": 2, "description": "排除基线 ECOG 体力评分 ≥ 2 分患者"},
        ],
    },
}


CLINICAL_IDEA_PRESETS = (
    {
        "preset_id": "idea_cardiorenal_hf",
        "domain_zh": "心肾代谢 · 心力衰竭与 2 型糖尿病",
        "title_zh": "SGLT2 抑制剂（达格列净）对比 DPP-4 抑制剂降低心衰住院与心血管死亡",
        "idea_text": (
            "我想在合并2型糖尿病与慢性肾病/心衰风险的成年住院及门诊患者中，评估新起始达格列净（Dapagliflozin）"
            "对比西格列汀（Sitagliptin）或标准治疗对12个月内心衰恶化住院或心血管死亡复合终点的影响，"
            "同时将非心血管原因死亡作为竞争风险事件进行建模。"
        ),
    },
    {
        "preset_id": "idea_nsclc_immunotherapy",
        "domain_zh": "肿瘤免疫 · 晚期非小细胞肺癌一线治疗",
        "title_zh": "帕博利珠单抗联合含铂化疗对比单纯含铂双药化疗一线治疗晚期非鳞 NSCLC",
        "idea_text": (
            "针对驱动基因阴性、ECOG 0-1分的晚期非小细胞肺癌（NSCLC）一线初治患者，模拟对标随机对照试验，"
            "比较帕博利珠单抗联合培美曲塞+铂类（Pembrolizumab plus chemotherapy）对比单纯含铂双药化疗"
            "在6个月、12个月及18个月内的肿瘤进展或肺癌特异死亡风险，并显式考虑非肿瘤竞争死亡（如感染或心血管事件）。"
        ),
    },
    {
        "preset_id": "idea_afib_doac",
        "domain_zh": "心血管抗凝 · 房颤合并高龄衰弱人群",
        "title_zh": "阿哌沙班（Apixaban）对比华法林（Warfarin）预防非瓣膜性房颤卒中与竞争死亡",
        "idea_text": (
            "在高龄非瓣膜性心房颤动（Atrial Fibrillation）患者中，比较起始阿哌沙班（Apixaban）对比华法林（Warfarin）"
            "在1年随访期内缺血性/出血性卒中或体循环栓塞（主要结局）的累计发生率，并将非卒中全因竞争死亡作为竞争风险事件。"
        ),
    },
    {
        "preset_id": "idea_ckd_finerenone",
        "domain_zh": "肾脏内分泌 · 糖尿病肾病进展防护",
        "title_zh": "非奈利酮（Finerenone）对比标准 RAS 阻滞治疗延缓糖尿病肾病肾功能衰竭",
        "idea_text": (
            "在已接受最大耐受剂量 ACEi/ARB 的2型糖尿病合并慢性肾脏病（CKD）患者中，评估加用非奈利酮（Finerenone）"
            "对比安慰剂/标准治疗对持续 eGFR 下降≥40% 或终末期肾病复合肾脏结局的影响，以肾外心血管或感染死亡作为竞争事件。"
        ),
    },
)


def generate_sample_cohort_csv_bytes(sample_id: str) -> tuple[str, bytes]:
    """Generate deterministic realistic-header CSV bytes for the requested demo sample."""
    if sample_id == "cardiorenal_sglt2i":
        return (
            SAMPLE_COHORT_SPECS["cardiorenal_sglt2i"]["filename"],
            _build_cardiorenal_csv(),
        )
    if sample_id == "oncology_io_nsclc":
        return (
            SAMPLE_COHORT_SPECS["oncology_io_nsclc"]["filename"],
            _build_oncology_csv(),
        )
    raise ValueError(f"Unknown sample_id: {sample_id}")


def _build_cardiorenal_csv() -> bytes:
    rng = np.random.default_rng(20261501)
    n = 1500
    age = np.clip(rng.normal(63.5, 11.8, size=n), 33.0, 88.0)
    female = (rng.uniform(size=n) < 0.43).astype(int)
    egfr = np.clip(rng.normal(58.0 - 0.35 * (age - 63.0), 17.5, size=n), 16.0, 112.0)
    hba1c = np.clip(rng.normal(8.1, 1.15, size=n), 5.8, 12.6)
    prior_hf = (rng.uniform(size=n) < expit(-1.15 + 0.028 * (age - 60.0) - 0.018 * (egfr - 60.0))).astype(int)
    bmi = np.clip(rng.normal(27.4, 4.1, size=n), 17.5, 43.0)

    nyha_probs = np.column_stack(
        [
            np.where(prior_hf == 1, 0.42, 0.72),
            np.where(prior_hf == 1, 0.43, 0.23),
            np.where(prior_hf == 1, 0.15, 0.05),
        ]
    )
    u_nyha = rng.uniform(size=n)
    nyha_idx = (u_nyha > nyha_probs[:, 0]).astype(int) + (
        u_nyha > (nyha_probs[:, 0] + nyha_probs[:, 1])
    ).astype(int)
    nyha_labels = ["NYHA_I_II", "NYHA_III", "NYHA_IV"]

    # Confounded treatment initiation: Dapagliflozin more likely in prior HF / higher BMI / younger-moderate eGFR
    logit_dapa = (
        -0.10
        - 0.018 * (age - 64.0)
        + 0.012 * (egfr - 58.0)
        + 0.55 * prior_hf
        + 0.045 * (bmi - 27.0)
        + 0.14 * (hba1c - 8.0)
    )
    is_dapa = (rng.uniform(size=n) < expit(logit_dapa)).astype(int)

    # Competing latent times in months (horizon up to 24 months)
    lin_primary = (
        0.026 * (age - 64.0)
        - 0.016 * (egfr - 58.0)
        + 0.52 * prior_hf
        + 0.12 * (hba1c - 8.0)
        + 0.25 * (nyha_idx >= 1)
        - 0.36 * is_dapa  # True protective log-csHR = -0.36 (csHR ≈ 0.70)
    )
    rate_primary_mo = 0.0175 * np.exp(lin_primary)
    t_primary_mo = -np.log(np.clip(rng.uniform(size=n), 1e-12, 1.0)) / rate_primary_mo

    lin_comp = (
        0.042 * (age - 64.0)
        - 0.012 * (egfr - 58.0)
        + 0.22 * prior_hf
        + 0.20 * (nyha_idx == 2)
        - 0.06 * is_dapa
    )
    rate_comp_mo = 0.0068 * np.exp(lin_comp)
    t_comp_mo = -np.log(np.clip(rng.uniform(size=n), 1e-12, 1.0)) / rate_comp_mo

    c_rand_mo = -np.log(np.clip(rng.uniform(size=n), 1e-12, 1.0)) / 0.014
    c_admin_mo = rng.uniform(14.0, 24.0, size=n)
    c_mo = np.minimum(c_rand_mo, c_admin_mo)

    obs_mo = np.maximum(0.10, np.minimum(np.minimum(t_primary_mo, t_comp_mo), c_mo))
    status = []
    for i in range(n):
        if t_primary_mo[i] <= t_comp_mo[i] and t_primary_mo[i] <= c_mo[i]:
            status.append("HF_or_CV_Event")
        elif t_comp_mo[i] < t_primary_mo[i] and t_comp_mo[i] <= c_mo[i]:
            status.append("NonCV_Competing_Death")
        else:
            status.append("Alive_Censored")

    # Introduce realistic ~1% missingness on bmi_kg_m2 so complete-case attrition is transparent
    missing_bmi_idx = set(rng.choice(n, size=18, replace=False).tolist())

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        [
            "deid_patient_id",
            "age_years",
            "female",
            "egfr_ml_min",
            "hba1c_pct",
            "prior_hf_hosp",
            "bmi_kg_m2",
            "nyha_class",
            "initiated_drug",
            "followup_months",
            "outcome_status",
        ]
    )
    for i in range(n):
        writer.writerow(
            [
                f"Hosp-2023-{i + 1:04d}",
                f"{age[i]:.1f}",
                int(female[i]),
                f"{egfr[i]:.1f}",
                f"{hba1c[i]:.2f}",
                int(prior_hf[i]),
                "" if i in missing_bmi_idx else f"{bmi[i]:.1f}",
                nyha_labels[int(nyha_idx[i])],
                "Dapagliflozin" if is_dapa[i] == 1 else "Sitagliptin",
                f"{obs_mo[i]:.2f}",
                status[i],
            ]
        )
    return buf.getvalue().encode("utf-8")


def _build_oncology_csv() -> bytes:
    rng = np.random.default_rng(20261202)
    n = 1200
    age = np.clip(rng.normal(64.8, 9.6, size=n), 36.0, 85.0)
    u_ecog = rng.uniform(size=n)
    ecog = np.where(u_ecog < 0.36, 0, np.where(u_ecog < 0.91, 1, 2))
    pdl1 = np.clip(rng.beta(1.2, 1.8, size=n) * 100.0, 0.0, 99.0)
    liver_met = (rng.uniform(size=n) < 0.21).astype(int)

    u_smoke = rng.uniform(size=n)
    smoke_idx = np.where(u_smoke < 0.28, 0, np.where(u_smoke < 0.76, 1, 2))
    smoke_labels = ["Never_Smoker", "Former_Smoker", "Current_Smoker"]

    logit_io = (
        0.05
        - 0.016 * (age - 65.0)
        - 0.38 * (ecog >= 1)
        + 0.016 * (pdl1 - 35.0)
        + 0.25 * liver_met
        + 0.20 * (smoke_idx >= 1)
    )
    is_io = (rng.uniform(size=n) < expit(logit_io)).astype(int)

    lin_prog = (
        0.015 * (age - 65.0)
        + 0.42 * (ecog >= 1)
        + 0.55 * liver_met
        - 0.006 * (pdl1 - 35.0)
        - 0.41 * is_io  # True protective log-csHR = -0.41 (csHR ≈ 0.66)
    )
    rate_prog_day = 0.0024 * np.exp(lin_prog)
    t_prog = -np.log(np.clip(rng.uniform(size=n), 1e-12, 1.0)) / rate_prog_day

    lin_other = 0.035 * (age - 65.0) + 0.35 * (ecog >= 1) + 0.15 * (smoke_idx == 2) - 0.04 * is_io
    rate_other_day = 0.00055 * np.exp(lin_other)
    t_other = -np.log(np.clip(rng.uniform(size=n), 1e-12, 1.0)) / rate_other_day

    c_rand = -np.log(np.clip(rng.uniform(size=n), 1e-12, 1.0)) / 0.0006
    c_admin = rng.uniform(420.0, 760.0, size=n)
    c_day = np.minimum(c_rand, c_admin)

    obs_days = np.maximum(3.0, np.minimum(np.minimum(t_prog, t_other), c_day))
    vital = []
    for i in range(n):
        if t_prog[i] <= t_other[i] and t_prog[i] <= c_day[i]:
            vital.append("Progression_or_CancerDeath")
        elif t_other[i] < t_prog[i] and t_other[i] <= c_day[i]:
            vital.append("TreatmentUnrelated_Death")
        else:
            vital.append("Censored")

    missing_pdl1 = set(rng.choice(n, size=12, replace=False).tolist())

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        [
            "case_no",
            "age",
            "ecog_ps",
            "pdl1_tps_pct",
            "liver_met",
            "smoking_history",
            "first_line_regimen",
            "os_days",
            "vital_event",
        ]
    )
    for i in range(n):
        writer.writerow(
            [
                f"ONC-{880001 + i}",
                f"{math.floor(age[i]):d}",
                int(ecog[i]),
                "" if i in missing_pdl1 else f"{pdl1[i]:.1f}",
                int(liver_met[i]),
                smoke_labels[int(smoke_idx[i])],
                "Pembrolizumab_Plus_Chemo" if is_io[i] == 1 else "Platinum_Doublet_Chemo",
                f"{obs_days[i]:.1f}",
                vital[i],
            ]
        )
    return buf.getvalue().encode("utf-8")
