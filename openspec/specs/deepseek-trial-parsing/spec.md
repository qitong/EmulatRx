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
