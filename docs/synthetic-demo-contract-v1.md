# Synthetic teaching template contract v1

This standalone command validates configuration only. It produces no cases, outcomes, clinical parameters, API requests, or clinical evidence. It uses only the Python standard library and Git, and does not import the notebook pipeline.

From a clean clone:

```bash
python -m emulatrx_demo validate --template examples/synthetic_demo_v1.json --out /tmp/emulatrx-template-check
```

Use a **new** output directory each time. The example is entirely synthetic. The command reads only the JSON file explicitly named by `--template`; it does not scan patient files, databases, credentials, or legacy worktrees. The input is limited to 16 KiB and must match [schema v1](../schemas/synthetic-demo-template-v1.schema.json). The fixed `scenario` is a teaching placeholder, not a disease, treatment, or trial. `case_count` is a positive integer for a future synthetic generator, not a clinical sample size claim. `seed` is an integer from 0 through 4294967295. There are no patient-level fields; unknown fields are rejected.

On success, stdout is a JSON validation result with `status: validated_only`, the simulation label, and the output path. The new output directory contains:

| File | Contents |
| --- | --- |
| `normalized_template.json` | The five accepted fields, sorted and formatted as UTF-8 JSON with a final newline. Equal valid values yield equal bytes. |
| `run_manifest.json` | `status`, fixed synthetic source type, validation activity, the simulation label, template version, seed, SHA-256 of the **original input bytes**, 40-character Git commit, `code_dirty`, output file names, and `run_at_utc`. The timestamp is a separate field; remove it when comparing reproducible manifest content. |

Both files are prepared in a temporary sibling directory and published together. The command rejects an existing destination and preserves it. Failure writes a stable error code to stderr and exits with status 2, with no published output directory. Error codes include `INVALID_JSON`, `MISSING_FIELD`, `UNKNOWN_FIELD`, `INVALID_TYPE`, `INVALID_CASE_COUNT`, `INVALID_SEED`, `UNKNOWN_VERSION`, `INVALID_SCENARIO`, `INVALID_SIMULATION_LABEL`, `OUTPUT_EXISTS`, `OUTPUT_PARENT_MISSING`, `TEMPLATE_IO_ERROR`, `CODE_VERSION_UNAVAILABLE`, and `OUTPUT_IO_ERROR`.

This first-stage manifest is a provenance outline. It records the source template hash, validation activity state, responsible code commit, and output paths. It does not claim a generated cohort or clinical result. A later stage can define generation and output hashes in a separately reviewed contract revision.

Run only the pure synthetic validator tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_demo_contract.py' -v
```

教学模拟，非真实患者、非临床疗效证据。
