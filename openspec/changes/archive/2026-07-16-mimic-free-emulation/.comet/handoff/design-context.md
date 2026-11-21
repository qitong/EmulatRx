# Comet Design Handoff

- Change: mimic-free-emulation
- Phase: design
- Mode: compact
- Context hash: 3ff9677c10da2a0c3f4a424209dfe9a385015e5a7101c847cfa386170121c9cd

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/mimic-free-emulation/proposal.md

- Source: openspec/changes/mimic-free-emulation/proposal.md
- Lines: 1-44
- SHA256: d001e550eba3f122794942d86261bd63187348a5450736ad2543da4e8983283d

```md
# Proposal: mimic-free-emulation

## Why

EmulatRx cannot currently run end-to-end on this machine for two independent reasons:

1. **No real-world database.** The Informatician/Statistician agents require a PostgreSQL database in **MIMIC-IV** shape (schemas `mimiciv_hosp`/`mimiciv_icu`/`mimiciv_derived`, cohort keyed on `stay_id`, covariate tables `first_day_*`). MIMIC-IV needs PhysioNet credentials + a `mimic-code` build we do not have, and its data are **ICU/critical-care only** — a poor fit for this user's domain, which is **oncology / lung cancer** (see `/Users/qitonghu/Desktop/0816 长沙/papers/`, 5 NEJM oncology trial PDFs). Additionally, the committed `trials/` cohort data are Git-LFS pointers with no retrievable objects, so there is no usable cached data either.

2. **A hardcoded OpenAI dependency.** The trial-parsing step (`trialistUtils/main.py::parse_clinical_trial`, `trialistUtils/parse_info.py`) instantiates its own OpenAI client with a **committed hardcoded API key** and calls `gpt-4o`, bypassing the configured `LLM_MODEL`/`LLMZoo`. So even after the rest of the pipeline is pointed at DeepSeek, this step still calls OpenAI on a likely-dead key.

Together these block any real run. This change makes EmulatRx runnable without MIMIC-IV and without OpenAI.

## What Changes

- Add a **substitute real-world-data source** that satisfies the Informatician's existing database interface, so a trial emulation can be built and analysed without MIMIC-IV. Two-tier strategy, evaluated in deep design:
  - **Preferred:** load a public **OHDSI/OMOP sample CDM** (Eunomia / Synthea-OMOP, e.g. `synthea-breast_cancer-10k`, or CMS SynPUF) into PostgreSQL and teach the Informatician to query it.
  - **Fallback:** **generate synthetic, distribution-conforming data** derived from the oncology reference papers (baseline covariate/arm/outcome distributions) plus trial meta pulled from ClinicalTrials.gov, and stand up a PostgreSQL database matching the interface.
- Add a new `dataset` branch in `DataframeBuilder` (schema names, X-table list, identifier column, few-shot SQL) for whichever substitute source is chosen.
- **Route trial-parsing through the configured LLM (DeepSeek).** Replace the hardcoded OpenAI clients in `trialistUtils/main.py` and `parse_info.py` with the shared, env-configured `OpenaiClient` (already supports `base_url` → DeepSeek) or an injected `llm`; delete the committed keys.

## Goals

- A trial emulation runs against the substitute database and produces a cohort table (covariates X, treatment A, outcome Y, duration T) that the Statistician can analyse — no MIMIC-IV required.
- No step in the pipeline calls OpenAI or contains a hardcoded API key; the whole run uses the `.env`-configured DeepSeek backend.

## Scope

**In scope**
- Substitute-DB provisioning (OMOP sample load *or* synthetic generator) reproducible from a script/instructions.
- New `DataframeBuilder` dataset branch + few-shot SQL for the substitute schema.
- Config/notebook wiring to select the new dataset.
- DeepSeek routing for trial-parsing + removal of committed secrets.

**Non-goals**
- Obtaining or loading real MIMIC-IV.
- Changing the causal-inference methods (`emulation_utils.py`) or the LangGraph agent orchestration.
- Clinical validity of synthetic data — the goal is a runnable substitute sufficient to exercise the pipeline, not research-grade evidence.
- Recovering the Git-LFS `trials/` data.

## Risks & Assumptions

- OMOP's schema (person_id / domain tables) differs from the code's MIMIC assumptions (`stay_id`, `first_day_*`); an interface adapter is required in either strategy.
- Synthetic data reproduces *marginal distributions* from the papers, not true causal structure — treatment-effect estimates from it are illustrative only.
- The DeepSeek routing change is small and builds on the already-in-tree `OpenaiClient(base_url=…)` foundation.
```

## openspec/changes/mimic-free-emulation/design.md

- Source: openspec/changes/mimic-free-emulation/design.md
- Lines: 1-56
- SHA256: 95eae7b6c4d400ebccc697599a08d61d3cca268d823f38a5275029dac6a3d6ea

```md
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
```

## openspec/changes/mimic-free-emulation/tasks.md

- Source: openspec/changes/mimic-free-emulation/tasks.md
- Lines: 1-25
- SHA256: 36940fed34ffc2d29fd58bbcea22662a861a76d872275b7e85a729e7c0860000

```md
# Tasks: mimic-free-emulation

> High-level checklist from the Open phase. The Design and Build phases will refine and expand these (and the delta spec).

## A. Substitute data source (Task 1)

- [ ] A1. Deep-design decision: evaluate Option A (OMOP sample → Postgres) vs Option B (synthetic generator); pick one with a documented rationale.
- [ ] A2. Stand up a PostgreSQL instance locally and record connection info in `.env` (`DB_*`).
- [ ] A3a. (If Option A) Load an OHDSI/OMOP sample CDM (e.g. Eunomia `synthea-breast_cancer-10k` or SynPUF) into Postgres; verify tables/counts.
- [ ] A3b. (If Option B) Build a synthetic-data generator that reads distributions from `/Users/qitonghu/Desktop/0816 长沙/papers/` + CT.gov meta, generates a patient-level table, and loads it into Postgres.
- [ ] A4. Add a new `dataset` branch to `DataframeBuilder` (schema name, X-table list, id column) for the chosen source.
- [ ] A5. Provide few-shot SQL examples for the Informatician matching the substitute schema (eligibility / treatment / outcome).
- [ ] A6. Wire notebook/config to select the new dataset; run one trial through DataframeBuilder → cohort table (X/A/Y/T) end-to-end.

## B. DeepSeek trial-parsing (Task 2)

- [ ] B1. Refactor `trialistUtils/main.py::parse_clinical_trial` to use the shared env-configured client (DeepSeek via `OpenaiClient(base_url=…)`) or an injected `llm`; remove the hardcoded OpenAI key.
- [ ] B2. Apply the same change to `trialistUtils/parse_info.py`; remove its hardcoded key.
- [ ] B3. Confirm no committed source file contains an API key (grep); keys come only from `.env`.
- [ ] B4. Verify parsing a trial (cached-skip path + a live parse) works through DeepSeek.

## C. Integration & verification

- [ ] C1. Run the pipeline for one oncology trial end-to-end on DeepSeek + substitute DB; confirm it reaches the Statistician stage and produces output.
- [ ] C2. Update `CLAUDE.md` / README notes: how to provision the substitute DB and that parsing now uses DeepSeek.
```

## openspec/changes/mimic-free-emulation/specs/deepseek-trial-parsing/spec.md

- Source: openspec/changes/mimic-free-emulation/specs/deepseek-trial-parsing/spec.md
- Lines: 1-18
- SHA256: 6c1cc571deb38962efb2325437e2695867d316cd7a048cd1db17501cfa02ba76

```md
# deepseek-trial-parsing (delta)

## ADDED Requirements

### Requirement: Trial parsing uses the configured LLM, not a hardcoded OpenAI client
The system SHALL route the trial-parsing step (`trialistUtils/main.py::parse_clinical_trial` and `trialistUtils/parse_info.py`) through the shared, environment-configured client (DeepSeek via `OpenaiClient` with `base_url`) or an injected `llm` handle. The system SHALL NOT construct an OpenAI client from a source-embedded API key.

#### Scenario: Parsing routes to the configured DeepSeek endpoint
- **WHEN** trial parsing runs with `DEEPSEEK_API_KEY` set in the environment
- **THEN** it issues its LLM calls through the configured client/endpoint (DeepSeek), not `api.openai.com`
- **AND** the function contract (inputs and returned parsed text/JSON) is unchanged

### Requirement: No committed API keys in source
The repository SHALL NOT contain any hardcoded API key in tracked source files; credentials SHALL come only from the environment / `.env`.

#### Scenario: Repository is free of embedded keys
- **WHEN** the source tree is scanned for embedded secrets (e.g. `grep -rn 'sk-' *.py trialistUtils/`)
- **THEN** no tracked source file contains an API key literal
```

## openspec/changes/mimic-free-emulation/specs/substitute-data-source/spec.md

- Source: openspec/changes/mimic-free-emulation/specs/substitute-data-source/spec.md
- Lines: 1-36
- SHA256: 0a29752afdca73963ebf59043bb9321df2472ae52a5ab0d3a05041041df82ed5

```md
# substitute-data-source (delta)

## ADDED Requirements

### Requirement: Synthetic oncology data source satisfies the Informatician interface
The system SHALL provide a synthetic real-world-data source for an advanced-NSCLC cohort that satisfies the existing `DataframeBuilder`/`Informatician` runtime contract without requiring MIMIC-IV. The source SHALL expose a patient table keyed on `stay_id` with columns `treatment` (0/1), `treatment_time_delta`, an outcome column and a duration column for survival, and baseline oncology covariates, plus a `col_dict` describing `patient_id`, `treatment_cols`, `outcome_cols`, `duration_cols`, `treatment_time_cols`, and `covariate_cols`.

#### Scenario: Cohort table matches the runtime contract
- **WHEN** the synthetic generator runs for the target trial id
- **THEN** it writes `trials/<id>/dataframes/dataframe.csv` whose columns are `stay_id, subject_id, treatment, treatment_time_delta, <outcome>, <duration>, <covariates...>`
- **AND** it writes `col_dict.json` whose `patient_id` is `stay_id`, `treatment_cols` is `treatment`, and `outcome_cols`/`duration_cols` reference the generated survival columns

#### Scenario: Confounded treatment effect is present
- **WHEN** the cohort is generated
- **THEN** treatment assignment depends on covariates (a covariate-dependent propensity), and survival time depends on both treatment and covariates with a target hazard ratio
- **AND** a Cox model fit on the raw generated data recovers the target hazard ratio within its confidence interval

### Requirement: Pipeline runs on the synthetic source without MIMIC-IV
The system SHALL allow `Informatician.get_df_dict` to return `(df, col_dict)` for the synthetic trial and the Statistician to run covariate balancing and survival analysis on it, with no connection to a MIMIC-IV database.

#### Scenario: End-to-end via CSV cache without a database
- **WHEN** the synthetic cache exists and `Informatician.get_df_dict(<id>)` is called with no database reachable
- **THEN** it returns `(df, col_dict)` from the cache without raising
- **AND** feeding the result to the Statistician's PSM/IPTW/Cox produces post-matching standardized mean differences below 0.1 on key covariates and an adjusted hazard ratio near the target

#### Scenario: Postgres path rebuilds the same cohort from SQL
- **WHEN** the CSV cache is cleared and `DataframeBuilder(dataset='synthetic')` runs against the loaded `synthetic` Postgres schema
- **THEN** `get_dataframe` rebuilds an equivalent `dataframe.csv` from SQL queries keyed on `stay_id`

### Requirement: MIMIC-specific assembly is guarded by dataset
The system SHALL NOT run the MIMIC-specific covariate/treatment munging when `dataset='synthetic'`.

#### Scenario: Synthetic branch bypasses MIMIC column munging
- **WHEN** `dataset='synthetic'` is selected
- **THEN** the MIMIC-only column logic in `_build_dataframe_X`/`_build_dataframe_treatment` (e.g. `abs_basophils`, `sofa`, `_max`/`_min` rules, `intime`/`outtime`) is not executed
- **AND** the synthetic covariate table loads without `KeyError`
```

