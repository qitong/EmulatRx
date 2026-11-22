# EmulatRx Framework

**EmulatRx** (EmulatRx) is an agentic framework designed to automate **Clinical Trial Design (CTD)** and **Target Trial Emulation (TTE)**. By leveraging a multi-agent architecture (Supervisor, Trialist, Informatician, Statistician, Clinician), it autonomously maps natural language trial protocols to structured Real-World Data (RWD) and performs causal inference analysis.

This repository contains the implementation of the EmulatRx framework as described in the manuscript *"EmulatRx: Empowering Clinical Trial Design with Agentic Intelligence and Real World Data"*.

This independent repository is based on [TrialLab/EmulatRx](https://github.com/TrialLab/EmulatRx), distributed under the MIT license. The original TrialLab copyright notice is retained in [LICENSE](LICENSE).

## 🚀 Features

* **Multi-Agent Workflow:** Orchestrates specialized agents via [LangGraph](https://langchain-ai.github.io/langgraph/) to handle trial parsing, cohort construction, and statistical analysis.
* **Hybrid LLM Support:** Seamlessly switch between cloud-based models (Azure OpenAI, GPT-4o) and local models (Phi-4, DeepSeek-R1, Llama 3) via **Ollama**.
* **RLHF & Optimization:** Includes a native Reinforcement Learning from Human Feedback (RLHF) loop (`llm_zoo_rlhf.py`) to train reward models and optimize agent responses based on user ratings.
* **Automated Trial Parsing:** Extracts and standardizes eligibility criteria from ClinicalTrials.gov using LLMs.
* **Real-World Evidence Generation:** Maps criteria to OMOP/MIMIC-IV schemas and performs covariate balancing (PSM/IPTW) and survival analysis.

## 🛠️ Prerequisites & Installation

### 1. System Requirements
* **Python 3.10+**
* **Ollama** (Required for local model execution)
* Access to MIMIC-IV or a compatible OMOP-CDM database.

### 2. Install Dependencies
Clone the repository and install the required Python packages:

```bash
git clone https://github.com/qitong/EmulatRx.git
cd EmulatRx
```

### 3. Environment Setup

We provide a pinned `environment.yml` file for reproducibility. The environment can be created by running:

```bash
conda env create -f environment.yml
conda activate emulatrx
```

### 4. Setting up Local Models (Ollama)

To use local models like Phi-4 or DeepSeek-R1 as described in the paper, you must install and run Ollama.

1.  **Download Ollama:** Visit ollama.com and install the version for your OS.
2.  **Pull Required Models:** Run the following commands in your terminal to download the weights:

```bash
ollama pull phi4
```

**Verify Server:** Ensure the Ollama server is running (default: localhost:11434). The `llm_zoo_rlhf.py` script connects to this local server automatically.

### 5. API Configuration
Export the provider key in your shell before running the notebook. The DeepSeek backend reads `DEEPSEEK_API_KEY`; the OpenAI backend reads `OPENAI_API_KEY`. These variables are read from the process environment, so a `.env` file alone is not loaded automatically. Do not commit real keys.

```bash
export DEEPSEEK_API_KEY="your_deepseek_key_here"
# Or, when using OpenAI:
export OPENAI_API_KEY="your_openai_key_here"
```

## 🏃 Usage
The core workflow is driven by the Jupyter Notebook `v1-codebase.ipynb`.

### Standalone synthetic teaching demo

The [v1 synthetic teaching contract](docs/synthetic-demo-contract-v1.md) has independent standard-library entry points. `validate` checks the fixed template; `demo` writes seeded, labeled, fully synthetic cases and an aggregate of arbitrary scores. These commands require no notebook, database, model, or API key. The demo has no clinical meaning.

```bash
python -m emulatrx_demo validate --template examples/synthetic_demo_v1.json --out /tmp/emulatrx-template-check
python -m emulatrx_demo demo --template examples/synthetic_demo_v1.json --out /tmp/emulatrx-teaching-demo
```

### Running an Emulation
Launch Jupyter:
```bash
jupyter notebook v1-codebase.ipynb
```

**Select LLM Backend:** In the configuration cell, you can choose between OpenAI or local Ollama models using the LLMZoo registry:

Run the cells sequentially.

### Running Without MIMIC-IV (Synthetic Data)
If you do not have a MIMIC-IV database, generate a synthetic **NSCLC** cohort that satisfies the same data interface (written as the CSV cache the pipeline reads, so no database is required):

```bash
python scripts/generate_synthetic_oncology.py --trial-id NCT00000099   # add --postgres to also load a `synthetic` Postgres schema
```

Then set `TRIAL_ID = 'NCT00000099'` and `LLM_MODEL = 'deepseek_api'` in the notebook and run the cells. The cohort is confounded (target hazard ratio 0.65), so covariate balancing and survival analysis produce a meaningful demonstration. Distributions are configured in `scripts/synthetic_config.py`. This is illustrative data for exercising the pipeline, not clinical evidence.

## 🔌 External Services

The framework will use the following external services: the ClinicalTrials.gov API, Ollama Local Server, and the OpenAI API.

## 📂 Repository Structure

* `v1-codebase.ipynb`: Main entry point controlling the LangGraph state machine and agent coordination.
* `llm_zoo_rlhf.py`: The core Model Registry.
    * Handles connections to **Ollama** (Local) and **Azure OpenAI** (Cloud).
    * Implements **RLHF** data collection and reward model training (FeedbackDataset, LLMwithReward).
    * Implements **Mixture of Experts (MoE)** routing logic.
* `trialistUtils/`:
    * `main.py`: Logic for fetching and parsing ClinicalTrials.gov data.
    * `prompts.py`: System prompts for extracting and standardizing eligibility criteria.
    * `nlp_utils.py`: NLP helper functions for text processing.
* `dataframe_utils.py`: Helper functions for the **Informatician Agent** (SQL generation, data cleaning, cohort building).
* `emulation_utils.py`: Statistical tools for the **Statistician Agent** (Propensity Score Matching, SMD calculation, Survival Analysis).
* `trials/`: Output directory where generated trial data (CSVs, JSONs) and analysis reports are cached.

The included `NCT00000099` cohort is synthetic and can be regenerated with the command above. No MIMIC-IV patient data is included; using MIMIC-IV requires your own authorized data access.
