# Blended advanced-NSCLC baseline distributions for the synthetic EmulatRx data source.
#
# Numbers are literature-typical for first-line advanced/metastatic NSCLC cohorts,
# blended across the reference trials shipped with this repo (KEYNOTE-189 /
# NEJMoa1801005, KEYNOTE-671 / NEJMoa2302983, LIBRETTO-431 / NEJMoa2309457,
# TRIDENT-1 / NEJMoa2302299). They are intentionally approximate: the synthetic
# generator only needs plausible marginal distributions plus a confounded
# treatment-assignment / outcome structure that recovers HR_TARGET.
SEED = 20260715
N_PATIENTS = 1200
HR_TARGET = 0.65               # experimental treatment vs control, on the event hazard
MEDIAN_FOLLOWUP_DAYS = 720     # ~24 months
BASELINE_MEDIAN_SURV_DAYS = 365  # control-arm median ~12 months
WEIBULL_SHAPE = 1.3

# name -> spec. ("normal",mean,sd,lo,hi) | ("lognormal",meanlog,sdlog,lo,hi) | ("bin",p) | ("cat",{level:prob})
COVARIATES = {
    "age":             ("normal", 64.0, 9.0, 30, 90),
    "sex_male":        ("bin", 0.55),
    "ecog_ps":         ("cat", {0: 0.40, 1: 0.50, 2: 0.10}),
    "smoking":         ("cat", {0: 0.20, 1: 0.50, 2: 0.30}),   # never/former/current
    "histology_adeno": ("bin", 0.70),
    "stage_iv":        ("bin", 0.70),
    "pdl1_tps":        ("normal", 35.0, 30.0, 0, 100),
    "egfr_mut":        ("bin", 0.15),
    "hemoglobin":      ("normal", 12.5, 1.6, 7, 18),
    "albumin":         ("normal", 3.8, 0.5, 2, 5),
    "ldh":             ("lognormal", 5.5, 0.4, 100, 2000),
    "neutrophils":     ("normal", 5.0, 2.0, 1, 20),
}
# Confounders driving treatment assignment (use z-scored value for names ending _z).
# Every term here also appears in OUTCOME_BETAS, so all are TRUE confounders (affect
# both treatment and outcome) that a covariate-adjusted / IPTW analysis can correctly
# remove to recover the conditional HR ~= HR_TARGET. Weights are modest so overlap
# stays reasonable. (An earlier iteration made pdl1_tps_z a pure instrument — strong
# on treatment, absent from the outcome — but adjusting for an instrument inflates
# variance / induces Z-bias, pulling the covariate-adjusted estimate off target, so
# it was made a genuine confounder below.)
#
# pdl1_tps_z is a genuine CONFOUNDER (present in both models): higher PD-L1 makes
# treatment more likely AND independently lowers the hazard. Keeping it in both
# models means adjusting for it is correct (unlike a pure instrument, which would
# inflate variance / induce Z-bias when the analyst adjusts for it), so a standard
# covariate-adjusted analysis recovers the conditional HR ~= HR_TARGET.
PROPENSITY = {"intercept": -0.2, "ecog_ps": -0.12, "stage_iv": -0.08, "pdl1_tps_z": 0.45, "age_z": -0.06}
# Covariates driving the outcome hazard (besides treatment).
OUTCOME_BETAS = {"ecog_ps": 0.5, "stage_iv": 0.4, "age_z": 0.2, "ldh_z": 0.3, "albumin": -0.3, "pdl1_tps_z": -0.2}

# ---------------------------------------------------------------------------
# Realistic-cohort mode: a raw patient POOL that includes ineligible patients,
# from which the analysis cohort is DERIVED by applying eligibility criteria.
# This makes the eligibility criteria operational (each criterion filters real
# patients), which is what the eligibility-criteria-optimization analysis needs.
# ---------------------------------------------------------------------------
N_RAW = 6000  # raw pool size; the eligible analysis cohort is a subset (~3300)

# In the raw pool ECOG spans 0-4 (the cohort keeps only 0-2 via eligibility).
# This OVERRIDES the cohort-only ecog_ps distribution in COVARIATES above.
ECOG_RAW = {0: 0.34, 1: 0.40, 2: 0.14, 3: 0.09, 4: 0.03}

# Eligibility-determining variables added to the raw pool (extra columns beyond
# the 12 analysis covariates). `stage` also DERIVES stage_iv (= 1 iff stage == 4),
# overriding the independent stage_iv draw in COVARIATES.
ELIGIBILITY_VARS = {
    "stage":                  ("cat", {1: 0.10, 2: 0.12, 3: 0.28, 4: 0.50}),  # advanced = 3 or 4
    "prior_systemic_therapy": ("bin", 0.20),
}

# Operationalized eligibility criteria. Each key is the criterion NAME, reused as
# the include/exclude .sql filename stem and the matched_patients_ec_<name>.csv stem
# (no spaces — DataframeBuilder keys criteria by filename). Predicate = (column, op,
# values); op in {"in","==","!=",">=","<=",">","<"}.
INCLUSION_CRITERIA = {
    "advanced_or_metastatic_NSCLC": ("stage", "in", [3, 4]),
    "ECOG_PS_0_to_2":               ("ecog_ps", "in", [0, 1, 2]),
}
EXCLUSION_CRITERIA = {
    "prior_systemic_therapy_for_metastatic_disease": ("prior_systemic_therapy", "==", 1),
}
