# Design (high-level): mimic-free-emulation

> High-level architecture and approach selection only. The deep technical design (schema mapping, generator internals, delta spec) is produced in the Design phase.

## The interface we must satisfy

`DataframeBuilder`/`Informatician` expect a PostgreSQL database that can answer, per patient/episode id:

- **Eligibility** — inclusion/exclusion SQL each returning a set of ids; cohort = (∪ inclusion) − (∪ exclusion).
- **Treatment (A)** — which ids received the intervention.
- **Outcome (Y) + duration (T)** — endpoint value and time-to-event per id.
- **Covariates (X)** — baseline features per id, from a configured list of X tables joined on the id.

MIMIC wiring (for reference): id = `stay_id`; X tables = `mimiciv_derived.first_day_*`; few-shot SQL teaches the LLM the schema. A substitute source must supply the same four pieces under **one consistent id column** plus few-shot SQL examples for its own schema.

## Approach selection (two tiers)

### Option A — OMOP sample CDM → PostgreSQL (preferred)
- Source: OHDSI **Eunomia** sample datasets (Synthea-OMOP, e.g. `synthea-breast_cancer-10k` for oncology, or `GiBleed`) or **CMS SynPUF**, OMOP CDM v5.3/5.4.
- Load into Postgres via OHDSI `CommonDataModel` DDL + `ETLSyntheaBuilder` (or by importing the CDM CSVs directly).
- Add `dataset='omop'` branch to `DataframeBuilder`: schema = `cdm` (configurable); id = `person_id` (or a visit/episode id — decide in deep design); X built from OMOP domain tables (`person`, `measurement`, `observation`, `condition_occurrence`, `drug_exposure`, `visit_occurrence`, `death`).
- Provide few-shot OMOP SQL examples for eligibility/treatment/outcome so the Informatician generates CDM-correct SQL.
- Pro: real, standards-based schema; oncology sample exists. Con: OMOP↔MIMIC schema gap; id-granularity mismatch with the code's `stay_id` assumption.

### Option B — Synthetic distribution-conforming data (fallback)
- Extract per-arm distributions from the 5 oncology papers: sample sizes, baseline covariate summaries (mean/SD, proportions), treatment arms, primary outcome / hazard ratio, follow-up. Pull trial meta (eligibility, arms, outcomes) from ClinicalTrials.gov via the existing `trialistUtils/web_utils.get_trial_info` (CT.gov API v2).
- Generate a patient-level table matching those marginals (+ a plausible treatment→outcome effect), load into Postgres under a **flat schema that mirrors the interface** (one id column, covariate columns, treatment, outcome, duration) — closest to the existing `dataset='insight'` shape (`merge` schema, single `covariates` X table).
- Add `dataset='synthetic'` branch pointing at that schema.
- Pro: fully under our control, oncology-relevant, no external data license. Con: illustrative only — not real evidence.

**Decision rule:** attempt Option A first (does a sample load cleanly + map onto the interface with acceptable effort?); if the OMOP↔interface adapter is too costly or the oncology sample is unsuitable, fall back to Option B. Final choice is a Design-phase decision point.

## Data flow (unchanged downstream)

```
substitute Postgres  →  DataframeBuilder(dataset=omop|synthetic)  →  Informatician
   (cohort via include/exclude/treatment/outcome SQL, keyed on id)
   →  merged X / A / Y / T table  →  Statistician (PSM/IPTW/Cox, emulation_utils)
```

Only the *data source + its DataframeBuilder branch + few-shot SQL* are new. The agents, LangGraph orchestration, and causal-inference code are untouched.

## Task 2 — DeepSeek routing for trial-parsing

`trialistUtils/main.py::parse_clinical_trial` and `parse_info.py` build their own `OpenAI(...)` client with a hardcoded key and call `gpt-4o`. Change:
- Accept/reuse the shared env-configured client — `OpenaiClient(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, model=DEEPSEEK_MODEL)` — or an injected `llm` handle from the caller.
- Read all credentials from `.env`; delete the committed hardcoded keys from source.
- Preserve the function contract (same inputs, returns parsed text/JSON), so the rest of the Trialist path is unchanged.

Builds on the already-in-tree foundation (`OpenaiClient` now takes `base_url`/`model`/`max_tokens`; `DEEPSEEK_API_KEY` in gitignored `.env`).

## Open questions (for deep design)
- Which sample dataset, and what id granularity (person vs visit/episode) vs the code's `stay_id` assumption?
- How much of `DataframeBuilder`'s MIMIC-specific assembly (`first_day_*`, `stay_id` joins, `_get_matched_patients`) generalises vs needs an OMOP/synthetic-specific path?
- For Option B: how faithfully must synthetic data mirror the papers (marginals only, or induced treatment effect / survival curves)?
- Should trial-parsing reuse the notebook's `llm` object (dependency injection) rather than constructing its own client?
