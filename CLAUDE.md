# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**EmulatRx** is a multi-agent framework for automating **Clinical Trial Design (CTD)** and **Target Trial Emulation (TTE)**. It maps a natural-language trial protocol to structured Real-World Data (MIMIC-IV / OMOP-CDM in Postgres) and runs causal-inference analysis. Five roles collaborate over a LangGraph state machine: a **Supervisor** that routes, and four domain agents — **Trialist** (find & parse a trial to emulate), **Informatician** (build the cohort dataset), **Statistician** (causal inference), **Clinician** (literature RAG, consulted on issues).

There is no test suite and no build step. The system runs interactively from a Jupyter notebook.

## Running

```bash
conda env create -f environment.yml && conda activate emulatrx
python -m spacy download en_core_web_sm   # required by trialistUtils/parse_info.py
ollama pull phi4                           # + gemma3:27b, deepseek-r1:14b if selected
jupyter notebook v1-codebase.ipynb
```

Then in the notebook: set `TRIAL_ID` (cell 2) and `LLM_MODEL` (cell 4, one of `phi4`/`deepseek`/`gemma3` for local Ollama, or `gpt4o`/`gpt4openai`/`deepseek_api` for cloud APIs), and **run all cells top to bottom**. The final cells stream the top-level graph and print `final_summary`.

Cell 5 registers each backend only if its credentials are present, then asserts the selected `LLM_MODEL` is among the registered models — so a missing key produces a clear error naming what's available rather than a crash deep in the graph.

`llm_zoo_rlhf.py` also has a standalone `__main__` demo (`python llm_zoo_rlhf.py`) that exercises the model zoo and RLHF loop against Ollama.

### Running without MIMIC-IV (synthetic oncology data)
When no MIMIC-IV database is available, a synthetic **NSCLC** cohort can stand in. It is generated in the exact shape `DataframeBuilder` consumes and written as the CSV cache the pipeline reads *before* touching Postgres, so the emulation runs with **no database**:

```bash
python scripts/generate_synthetic_oncology.py --trial-id NCT00000099   # add --postgres to also load a `synthetic` schema
```

Then in the notebook set `TRIAL_ID = 'NCT00000099'`, `LLM_MODEL = 'deepseek_api'`, and run. The data source is wired via `dataset='synthetic'` in `DataframeBuilder` (schema `synthetic`, single `cohort` X table); the MIMIC-only covariate/treatment munging is guarded behind `if self.dataset == 'mimic'`. Distributions are configured in `scripts/synthetic_config.py` and the cohort is **confounded** (covariate-dependent treatment + a Weibull outcome with target HR 0.65), so PSM/IPTW/Cox demonstrate real adjustment. Regenerate anytime by deleting `trials/NCT00000099/dataframes/` and re-running. Note: it is illustrative data for exercising the pipeline, not clinical evidence.

Trial parsing (`trialistUtils/`) runs on the configured DeepSeek client (`DEEPSEEK_API_KEY`), not a hardcoded OpenAI key.

### Required environment (`.env` in repo root)
Loaded manually by the notebook and via `os.getenv` in modules — nothing works without it:
- LLM keys: `OPENAI_API_KEY` / `OPENAI_ORGANIZATION` / `OPENAI_PROJECT`, `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT`, `DEEPSEEK_API_KEY` (optional `DEEPSEEK_BASE_URL` default `https://api.deepseek.com`, `DEEPSEEK_MODEL` default `deepseek-chat`)
- Postgres (MIMIC-IV): `DB_USER` / `DB_PASSWORD` / `DB_HOST` / `DB_PORT` / `DB_NAME`

External services hit at runtime: Ollama (`localhost:11434`), ClinicalTrials.gov API v2, OHDSI Athena concept API, OpenAI/Azure, PubMed (via `paperscraper`).

## Architecture

### Orchestration (the notebook is the app)
`v1-codebase.ipynb` builds a **LangGraph supervisor topology**. State is a `MessagesState` subclass (`TopAgentState`) — a single dict threaded through every node. Load-bearing fields: `initial_question`, `predefined_next_agent` / `predefined_next_question` (how one agent hands off to the next), `current_information` / `all_information` (accumulated context; `all_information` is chunked into a **FAISS** index for RAG once `current_information` exceeds ~100k tokens), and `variables` (dict holding `trialid`).

- Each domain agent is its own compiled sub-graph (`medical_graph`, `statistician_graph`, `informatician_graph`, `trial_graph`), wrapped by an `*Invoker` node that copies fields in and out of `TopAgentState`.
- `MainSupervisor` is the hub: `START → MainSupervisor`, every agent edges back to `MainSupervisor`, and routing is a conditional edge on `state["next"]` (**star topology**, not a pipeline). Agents drive the sequence by setting `predefined_next_agent`.

When editing agent flow, the four per-agent `State` classes in the notebook are near-identical copies — keep them in sync.

### Module responsibilities
- **`llm_zoo_rlhf.py`** — `LLMZoo`, a model registry with JSON response caching (`llm_cache.json`, opt in per call via `use_cache=True`). Backends are callables sharing a `__call__(prompt, **kwargs)` interface: `OllamaModel`, `GPT4AzureModel`, `OpenaiClient`, `LlamafileModel`. Also the RLHF stack: `LLMwithInteractive` (collect ratings/rankings via ipywidgets → `rlhf_feedback.json`), `FeedbackDataset` + `train_reward_model` (BERT reward model → `reward_model/`), `LLMwithReward` (best-of-N over registered models), and `MoEModel` (SentenceTransformer embedding router over expert descriptions).
- **`trialistUtils/`** — the Trialist agent. `parse_clinical_trial(trial_id)` (in `main.py`) fetches the ClinicalTrials.gov API (`web_utils.get_trial_info`), extracts inclusion/exclusion/treatments/outcomes with GPT-4o prompts (`prompts.py`), and pickles a `TrialInfo` to `trialistUtils/results/trial_info/<id>/`. `web_utils.get_concept_id` maps concepts to OMOP via OHDSI Athena. `parse_info.py` is an older standalone NER-mapping script (spaCy + GPT), not on the notebook path.
- **`dataframe_utils.py`** — the Informatician agent. `Informatician` orchestrates; `DataframeBuilder` opens the Postgres connection, runs the hand-written per-criterion SQL under `trials/<id>/{include,exclude,treatment,outduration,adverseevents}/*.sql`, and assembles the cohort dataframe. `dataset='mimic'` uses the `mimiciv_derived` schema and its `first_day_*` tables; `dataset='insight'` uses the `merge` schema.
- **`emulation_utils.py`** — the Statistician agent. Covariate balancing (`perform_psm` / `matching_propensity`, `perform_iptw`, SMD checks), survival analysis (`run_cox_model`, `run_kaplan_meier`, `run_parametric_model`, `run_survival_forest`), doubly-robust ATE via **causallib**, plus `calculate_evalues`, `adjust_immortal_time_bias`, and `compute_cox_sample_size`.
- **`utils.py`** — shared helpers: PDF text/table extraction (**pymupdf**), `split_documents` for RAG chunking, ClinicalTrials.gov download helpers, `custom_python_executor`.
- **`trialist_utils.py`** — plain data classes (`TrialEligibilityCriterion`, `TrialTreatment`, `TrialOutcome`, `TrialInfo`). A second copy lives at `trialistUtils/trial_classes.py`.

### Data & caching model
`trials/<TRIAL_ID>/` is **both input and output**: you author SQL per criterion under it, and generated cohort CSVs land in `trials/<TRIAL_ID>/dataframes/` alongside pickled state (`ec_optimization.pkl`, `state_ec_optim.pkl`, `trial_info_<id>.pkl`). Caching is pervasive and file-existence-based — LLM calls (`llm_cache.json`), SQL results (CSVs), and parsed trial info (pkl/json) are all skipped if the file already exists. **To force regeneration, delete the relevant cached file.** `load_or_run_and_save(path, func, ...)` in `dataframe_utils.py` is the memoization primitive.

## Gotchas
- **`trialistUtils` uses bare intra-package imports** (`from openai_client import ...`, not `from .openai_client`). This only resolves because the notebook does `sys.path.insert(0, "./trialistUtils")` before importing. Running files inside that folder directly relies on the same cwd trick.
- `emulation_utils.py` defines `perform_balancing_method` **twice**; the second definition wins.
- **Historical committed secrets:** `trialistUtils/main.py` and `trialistUtils/parse_info.py` *used to* contain hardcoded OpenAI API keys; they now read from `.env` (`DEEPSEEK_API_KEY`, env-configured `OpenaiClient`). The old keys still exist in **git history** and should be rotated/revoked. Always source credentials from `os.getenv` / `.env` — never re-embed a key.
- `perform_psm` in `emulation_utils.py` currently raises `KeyError: 'mean'` from `check_balance_after_matching` (line 41) under recent pandas (a MultiIndex `.agg(['mean','std'])` indexing issue) — use `perform_iptw` / `run_cox_model` for balancing, or pin pandas, until that line is fixed.
