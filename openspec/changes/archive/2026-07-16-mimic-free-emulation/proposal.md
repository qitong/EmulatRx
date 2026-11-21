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
