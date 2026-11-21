---
change: mimic-free-emulation
design-doc: docs/superpowers/specs/2026-07-15-mimic-free-emulation-design.md
base-ref: e353ee46e189b692a62b9cc3cba3bbc73a7ab531
archived-with: 2026-07-16-mimic-free-emulation
---

# mimic-free-emulation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement this plan task-by-task.

**Goal:** Let EmulatRx run end-to-end for an oncology (NSCLC) trial without MIMIC-IV and without OpenAI — via a confounded synthetic data source (CSV cache + optional Postgres `synthetic` schema) and DeepSeek-routed trial parsing.

**Architecture:** A deterministic generator produces a confounded NSCLC cohort in the exact shape `DataframeBuilder` consumes, written as the CSV cache the pipeline reads before touching Postgres (and optionally loaded into a `synthetic` Postgres schema). A new `dataset='synthetic'` branch guards the MIMIC-only munging. Trial parsing is re-pointed at the shared env-configured DeepSeek client.

**Tech Stack:** Python 3.11 (conda env `emulatrx` at `/Users/qitonghu/anaconda3/envs/emulatrx/bin/python`), numpy/scipy/pandas, lifelines (Cox), psycopg2/sqlalchemy (optional Postgres), DeepSeek via `OpenaiClient(base_url=…)`.

**Runtime helper:** `PY=/Users/qitonghu/anaconda3/envs/emulatrx/bin/python`

**Global constraints:** Do NOT modify `emulation_utils.py` causal methods or the LangGraph orchestration in the notebook. Quote the papers path — it contains a space and non-ASCII: `"/Users/qitonghu/Desktop/0816 长沙/papers/"`. One commit per task.

**Synthetic trial id:** `NCT00000099` (a reserved placeholder id so it never collides with a real CT.gov trial).

archived-with: 2026-07-16-mimic-free-emulation
---

### Task 0: Baseline commit (clean starting tree)

**Files:** all currently-uncommitted: `.gitignore`, `llm_zoo_rlhf.py`, `v1-codebase.ipynb`, `CLAUDE.md`, `openspec/`, `docs/`, `.claude/`.

**Step 1:** Confirm `.env` is NOT staged: `git status --porcelain | grep -F '.env'` → expect empty (it is gitignored).
**Step 2:** Stage everything except secrets: `git add -A` then `git reset -- .env 2>/dev/null || true`.
**Step 3:** Commit: `git commit -m "chore: DeepSeek backend foundation + comet scaffolding for mimic-free-emulation"`.
**Step 4:** Verify clean: `git status --short` → only `.env` may remain (untracked, ignored is fine).

> Note: the branch on which this commit lands is decided by the Comet build config step (branch vs worktree). If a branch was created first, this commit lands there.

archived-with: 2026-07-16-mimic-free-emulation
---

### Task 1: Blended NSCLC distribution config

**Files:** Create `scripts/synthetic_config.py`

**Step 1:** Attempt to extract baseline tables from the 5 papers for reference (best-effort, non-blocking):
Run: `for f in "/Users/qitonghu/Desktop/0816 长沙/papers/"*.pdf; do echo "== $f =="; "$PY" -c "import pymupdf,sys; d=pymupdf.open(sys.argv[1]); print(d[0].get_text()[:1500])" "$f"; done`
Use the trial arms / baseline characteristics tables where legible; otherwise fall back to the documented advanced-NSCLC literature defaults below.

**Step 2:** Write `synthetic_config.py` with an explicit, documented distribution spec (defaults are literature-typical advanced/metastatic NSCLC values; override with any numbers extracted in Step 1):

```python
# Blended advanced-NSCLC baseline distributions (see design doc; refine from papers/).
import numpy as np

SEED = 20260715
N_PATIENTS = 1200
HR_TARGET = 0.65          # treatment (experimental arm) vs control, on the event hazard
MEDIAN_FOLLOWUP_DAYS = 720  # ~24 months
BASELINE_MEDIAN_SURV_DAYS = 365  # control-arm median ~12 months

# name -> spec. 'normal':(mean,sd,lo,hi); 'cat':{level:prob}
COVARIATES = {
    "age":            ("normal", 64.0, 9.0, 30, 90),
    "sex_male":       ("bin", 0.55),
    "ecog_ps":        ("cat", {0: 0.40, 1: 0.50, 2: 0.10}),
    "smoking":        ("cat", {0: 0.20, 1: 0.50, 2: 0.30}),   # never/former/current
    "histology_adeno":("bin", 0.70),
    "stage_iv":       ("bin", 0.70),                          # else stage III
    "pdl1_tps":       ("normal", 35.0, 30.0, 0, 100),
    "egfr_mut":       ("bin", 0.15),
    "hemoglobin":     ("normal", 12.5, 1.6, 7, 18),
    "albumin":        ("normal", 3.8, 0.5, 2, 5),
    "ldh":            ("lognormal", 5.5, 0.4, 100, 2000),     # meanlog, sdlog on raw scale clip
    "neutrophils":    ("normal", 5.0, 2.0, 1, 20),
}

# Covariates that drive treatment assignment (confounders) and their propensity weights.
PROPENSITY = {"intercept": -0.2, "ecog_ps": -0.6, "stage_iv": -0.4, "pdl1_tps_z": 0.7, "age_z": -0.3}
# Covariates that drive the outcome hazard (besides treatment).
OUTCOME_BETAS = {"ecog_ps": 0.5, "stage_iv": 0.4, "age_z": 0.2, "ldh_z": 0.3, "albumin": -0.3}
```

**Step 3:** Commit: `git add scripts/synthetic_config.py && git commit -m "feat: blended NSCLC synthetic distribution config"`.

archived-with: 2026-07-16-mimic-free-emulation
---

### Task 2: Synthetic generator → CSV cache (the guaranteed path)

**Files:** Create `scripts/generate_synthetic_oncology.py`; Test: `scripts/test_synthetic.py`

**Step 1 (failing test):** `scripts/test_synthetic.py`:
```python
import json, subprocess, os, sys, pandas as pd
PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIAL = "NCT00000099"
CACHE = os.path.join(ROOT, "trials", TRIAL, "dataframes")

def test_generator_emits_contract():
    subprocess.run([PY, os.path.join(ROOT, "scripts", "generate_synthetic_oncology.py"),
                    "--trial-id", TRIAL, "--no-postgres"], check=True)
    df = pd.read_csv(os.path.join(CACHE, "dataframe.csv"))
    cd = json.load(open(os.path.join(CACHE, "col_dict.json")))
    lead = ["stay_id","subject_id","treatment","treatment_time_delta"]
    assert list(df.columns)[:4] == lead
    assert cd["patient_id"] == "stay_id" and cd["treatment_cols"] == "treatment"
    assert cd["outcome_cols"] == ["event"] and cd["duration_cols"] == ["duration_day"]
    assert set(df["treatment"].unique()) <= {0,1} and df["event"].isin([0,1]).all()
    assert (df["duration_day"] > 0).all()

def test_hazard_ratio_recovered():
    from lifelines import CoxPHFitter
    df = pd.read_csv(os.path.join(CACHE, "dataframe.csv"))
    cph = CoxPHFitter().fit(df[["duration_day","event","treatment"]], "duration_day", "event")
    hr = float(cph.hazard_ratios_["treatment"])
    lo, hi = cph.confidence_intervals_.loc["treatment"].apply(lambda z: pow(2.718281828, z))
    assert lo <= 0.65 <= hi, f"HR {hr:.2f} CI [{lo:.2f},{hi:.2f}] must cover target 0.65"
```

**Step 2:** Run to confirm it fails (script missing): `"$PY" -m pytest scripts/test_synthetic.py -x -q` → FAIL.

**Step 3 (implement):** `scripts/generate_synthetic_oncology.py`. Core logic:
- Seed numpy with `SEED`. Draw each covariate per its spec; compute z-scores for the `_z` names used by PROPENSITY/OUTCOME.
- Propensity: `p = sigmoid(intercept + Σ w·feature)`; `treatment ~ Bernoulli(p)`; clamp `p∈[0.05,0.95]`.
- Survival: `linpred = log(HR_TARGET)*treatment + Σ OUTCOME_BETAS·feature`; Weibull with scale set so control median = `BASELINE_MEDIAN_SURV_DAYS`; `T = weibull_scale*exp(-linpred/shape)*(-ln U)^(1/shape)`. Censor `C ~ Exponential(mean=MEDIAN_FOLLOWUP_DAYS/ln2)`. `duration_day=min(T,C)` (≥1), `event=1[T≤C]`.
- `treatment_time_delta = 0.0` where treatment==1 else NaN.
- `stay_id = 900000 + i`, `subject_id = 800000 + i`.
- Assemble `df` columns exactly: `["stay_id","subject_id","treatment","treatment_time_delta","event","duration_day"] + covariate_names`.
- `col_dict = {"covariate_cols": covariate_names, "treatment_cols":"treatment", "outcome_cols":["event"], "duration_cols":["duration_day"], "treatment_time_cols":"treatment_time_delta", "patient_id":"stay_id"}`.
- Write `trials/<trial>/dataframes/{dataframe.csv,col_dict.json}`. Also write `big_df.csv` (== dataframe pre-filter), `merged_X.csv` (stay_id+subject_id+covariates), `treatment_df.csv`, `outduration_df.csv`, and `matched_patients.csv` (all stay_ids) so every `DataframeBuilder` cache check short-circuits.
- Create sibling dirs `trials/<trial>/{include,exclude,treatment,outduration,adverseevents}` (empty) so `load_sql_files_from_folder` finds them.
- CLI: `--trial-id`, `--no-postgres` (Task 3 adds the postgres branch).

**Step 4:** Run tests: `"$PY" -m pytest scripts/test_synthetic.py -x -q` → PASS (both).

**Step 5:** Commit: `git add scripts/generate_synthetic_oncology.py scripts/test_synthetic.py trials/NCT00000099 && git commit -m "feat: confounded synthetic NSCLC generator emitting DataframeBuilder CSV cache"`.

archived-with: 2026-07-16-mimic-free-emulation
---

### Task 3: Optional Postgres load path

**Files:** Modify `scripts/generate_synthetic_oncology.py`

**Step 1:** Add `--postgres` (default off). When on, read `DB_*` from env, connect via sqlalchemy; if connect fails, print a warning and continue (non-fatal). Create schema `synthetic`; write `df` to `synthetic.cohort` (denormalized) via `df.to_sql`. Optionally also `synthetic.covariates` / `synthetic.treatment` / `synthetic.outcome` split tables.
**Step 2:** Guard test (skips cleanly with no DB): `test_postgres_optional` asserts running with `--postgres` and no reachable DB still writes the CSV cache and exits 0.
**Step 3:** Run: `"$PY" -m pytest scripts/test_synthetic.py -k postgres -q` → PASS.
**Step 4:** Commit: `git commit -am "feat: optional synthetic Postgres load (non-fatal when no DB)"`.

archived-with: 2026-07-16-mimic-free-emulation
---

### Task 4: `dataset='synthetic'` branch in DataframeBuilder

**Files:** Modify `dataframe_utils.py`

**Step 1:** In `DataframeBuilder.__init__` dataset dispatch (near line 185-192), add:
```python
elif self.dataset == 'synthetic':
    self.scheme_name = 'synthetic'
    self.X_table_names = ['cohort']
```
**Step 2:** In `_build_dataframe_X` guard the MIMIC-only munging (current lines ~283-300: `variables_keep_min`, `abs_*`, `sofa`, `_max/_min` renaming) behind `if self.dataset == 'mimic':`. For non-mimic, merge X tables on `stay_id` and skip the munging.
**Step 3:** In `_build_dataframe_treatment`, guard `clean_treatment_time` (ICU intime/outtime) behind `if self.dataset == 'mimic':`; for synthetic, return the treatment df as-is.
**Step 4:** Confirm `load_sql_files_from_folder` (line 68) tolerates empty/missing dirs (returns `{}`); if it raises on a missing dir, make it return `{}`.
**Step 5 (integration test):** `scripts/test_builder_cache.py`:
```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ensure NO DB is used: unset DB_* so connect fails fast (non-fatal in __init__)
for k in ["DB_HOST","DB_PORT","DB_USER","DB_PASSWORD","DB_NAME"]: os.environ.pop(k, None)
from dataframe_utils import DataframeBuilder
def test_cache_returns_without_db():
    b = DataframeBuilder(trialID="NCT00000099", trial_info={}, dataset="synthetic",
                         user=None, password=None, host=None, port=None, database=None,
                         root_dir="./trials")
    df, cd = b.get_dataframe()
    assert cd["patient_id"] == "stay_id" and len(df) > 0
```
Run: `"$PY" -m pytest scripts/test_builder_cache.py -x -q` → PASS.
**Step 6:** Commit: `git commit -am "feat: add dataset='synthetic' branch; guard MIMIC-specific munging behind dataset=='mimic'"`.

archived-with: 2026-07-16-mimic-free-emulation
---

### Task 5: Route trial-parsing through DeepSeek; delete hardcoded keys

**Files:** Modify `trialistUtils/openai_client.py`, `trialistUtils/main.py`, `trialistUtils/parse_info.py`

**Step 1:** `trialistUtils/openai_client.py`: remove the module-level hardcoded `client = OpenAI(...)`. Make `OpenaiClient.__init__` read from env with `base_url`/`model`/`max_tokens` support:
```python
import os
from openai import OpenAI
class OpenaiClient:
    def __init__(self, api_key=None, organization=None, project=None, base_url=None, model=None, max_tokens=8192):
        api_key  = api_key  or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, organization=organization, project=project, base_url=base_url)
    def get_chatgpt_response(self, prompt, content):
        r = self.client.chat.completions.create(model=self.model,
            messages=[{"role":"system","content":prompt},{"role":"user","content":content}],
            temperature=0, max_tokens=self.max_tokens, top_p=1)
        return r.choices[0].message.content
```
(Keep `get_chatgpt_response_with_file` but have it use `self.model`.)
**Step 2:** `trialistUtils/main.py` line ~16: replace `OpenaiClient(api_key='sk-…', organization='…', project='…')` with `OpenaiClient()` (env-driven).
**Step 3:** `trialistUtils/parse_info.py`: remove the module-level hardcoded `api_key`/`client`; construct via env inside the function (or import the env-driven `OpenaiClient`).
**Step 4 (security test):** Run `grep -rn "sk-proj-\|sk-[A-Za-z0-9]\{20\}" --include=*.py .` → expect no matches. Add `scripts/test_no_secrets.py` asserting the grep is empty.
**Step 5:** Commit: `git commit -am "fix: route trial parsing through env-configured DeepSeek client; remove committed API keys"`.

archived-with: 2026-07-16-mimic-free-emulation
---

### Task 6: Verification (maps to delta-spec scenarios)

**Files:** Create `scripts/test_integration_statistician.py`

**Step 1 (integration):** Load the synthetic `(df, col_dict)` via `Informatician.get_df_dict("NCT00000099")` with no DB; run `emulation_utils.perform_psm` (or `run_cox_model`) with `covariate_cols`/`treatment`/`event`/`duration_day` from `col_dict`. Assert: it returns estimates; post-match SMD on ≥1 key confounder (ecog_ps/stage_iv) drops below 0.1; adjusted HR CI covers ~0.65. (If a Statistician-agent column-name coupling to `hospitaldischargestatus` surfaces, add a thin alias in `col_dict`/df and note it as a Small spec increment.)
**Step 2:** Run all tests: `"$PY" -m pytest scripts/ -q` → PASS.
**Step 3 (end-to-end, manual/notebook):** Using the `verify` skill or `jupyter nbconvert --execute` on a copy with `TRIAL_ID='NCT00000099'`, `LLM_MODEL='deepseek_api'`: confirm the run reaches the Statistician stage and prints output. Record where it stops if anything downstream still assumes MIMIC.
**Step 4:** Commit any test files: `git add scripts/test_integration_statistician.py && git commit -m "test: end-to-end synthetic emulation reaches Statistician on DeepSeek"`.

archived-with: 2026-07-16-mimic-free-emulation
---

### Task 7: Docs

**Files:** Modify `CLAUDE.md`, `README.md`

**Step 1:** Add a "Running without MIMIC-IV" section: `"$PY" scripts/generate_synthetic_oncology.py --trial-id NCT00000099`, then set `TRIAL_ID='NCT00000099'`, `LLM_MODEL='deepseek_api'` and run the notebook. Note the `dataset='synthetic'` branch and that trial-parsing now uses DeepSeek.
**Step 2:** Commit: `git commit -am "docs: how to run EmulatRx on synthetic oncology data + DeepSeek"`.

archived-with: 2026-07-16-mimic-free-emulation
---

## Verification checklist (exit)
- [ ] `"$PY" -m pytest scripts/ -q` all pass (contract, HR recovery, cache-without-DB, no-secrets, integration).
- [ ] `grep -rn "sk-proj-" --include=*.py .` empty.
- [ ] Notebook reaches Statistician stage on `deepseek_api` + synthetic trial.
- [ ] `emulation_utils.py` and notebook orchestration unchanged (only guards added in `dataframe_utils.py`).
