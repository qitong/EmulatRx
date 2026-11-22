# Synthetic teaching template contract v1

The standalone `validate` command checks configuration only. The independent `demo` command generates labeled, fully synthetic teaching cases and an aggregate of arbitrary teaching scores. Neither command uses an API, database, notebook pipeline, patient file, or clinical trial result. Both use only the Python standard library and Git.

From a clean clone:

```bash
python -m emulatrx_demo validate --template examples/synthetic_demo_v1.json --out /tmp/emulatrx-template-check
python -m emulatrx_demo demo --template examples/synthetic_demo_v1.json --out /tmp/emulatrx-teaching-demo
```

Use a **new** output directory each time. The example is entirely synthetic. Each command reads only the JSON file explicitly named by `--template`; it does not scan patient files, databases, credentials, or legacy worktrees. The input is limited to 16 KiB and must match [schema v1](../schemas/synthetic-demo-template-v1.schema.json). The fixed `scenario` is a teaching placeholder, not a disease, treatment, or trial. `case_count` is an integer from 1 through **10,000**, not a clinical sample size claim. This cap bounds the case CSV below 2 MiB at the maximum (checked by the synthetic end-to-end test), with one row generated at a time and no in-memory cohort. `seed` is an integer from 0 through 4294967295. There are no patient-level fields; unknown fields are rejected. JSON booleans are not integers under this contract.

On success, stdout is a JSON validation result with `status: validated_only`, the simulation label, and the output path. The new output directory contains:

| File | Contents |
| --- | --- |
| `normalized_template.json` | The five accepted fields, sorted and formatted as UTF-8 JSON with a final newline. Equal valid values yield equal bytes. |
| `run_manifest.json` | `status`, fixed synthetic source type, validation activity, the simulation label, template version, seed, SHA-256 of the **original input bytes**, 40-character Git commit, `code_dirty`, output file names, and `run_at_utc`. The timestamp is a separate field; remove it when comparing reproducible manifest content. |

For `demo`, the new output directory contains:

| File | Contents |
| --- | --- |
| `synthetic/cases.csv` | Seeded case IDs (`SYN-...`), the simulation label on every row, arbitrary groups A/B, and arbitrary scores 0–100. No diagnoses, therapies, dates, or outcomes. |
| `results/summary.json` | Simulation label, case count, group counts, score sum, and a plain explanation that the score has no clinical meaning. |
| `PURPOSE.txt` | Explicit teaching use and source isolation statement. |
| `run_manifest.json` | Original template SHA-256, SHA-256 for each other output, template/generator version, seed, count, 40-character Git commit, dirty-state flag, Python version and empty external package map, paths, simulation label, and the sole timestamp field `run_at_utc`. |

The generator derives each row from SHA-256 of a fixed generator namespace, seed, and row number. It has no random global state or dependence on trial metadata. Equal template bytes, code, Python version, and seed produce byte-identical cases, summary, and purpose text; removing `run_at_utc` makes the manifests equal too. The input hash changes if template formatting changes, even when generated cases do not. Runs from dirty source trees are marked `code_dirty`; a clean committed tree is needed for a reproducibility claim tied to a commit.

Both commands prepare all files in a temporary sibling directory and publish them together. They reject an existing destination and preserve it. Failure writes a stable error code to stderr and exits with status 2, with no published output directory. Error codes include `INVALID_JSON`, `MISSING_FIELD`, `UNKNOWN_FIELD`, `INVALID_TYPE`, `INVALID_CASE_COUNT`, `INVALID_SEED`, `UNKNOWN_VERSION`, `INVALID_SCENARIO`, `INVALID_SIMULATION_LABEL`, `OUTPUT_EXISTS`, `OUTPUT_PARENT_MISSING`, `TEMPLATE_IO_ERROR`, `CODE_VERSION_UNAVAILABLE`, and `OUTPUT_IO_ERROR`.

The `validate` manifest remains a validation-only provenance outline. It records no generated cohort or clinical result. The `demo` manifest records generated teaching files and their hashes; it also makes no clinical claim. Public trial metadata, when introduced separately, must stay outside `synthetic/cases.csv` and must not determine any generated row.

Run only the pure synthetic validator tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_demo_contract.py' -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_synthetic_demo.py' -v
```

教学模拟，非真实患者、非临床疗效证据。
