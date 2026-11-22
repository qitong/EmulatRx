"""End-to-end checks for the offline, fully synthetic teaching demo."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from emulatrx_demo import SIMULATION_LABEL


ROOT = Path(__file__).resolve().parents[1]
MAX_CASES = 10000


class SyntheticDemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "template.json"
        self.template = {
            "template_version": "1.0",
            "scenario": "synthetic_tutorial_v1",
            "case_count": 12,
            "seed": 42,
            "simulation_label": SIMULATION_LABEL,
        }
        self.source.write_text(json.dumps(self.template), encoding="utf-8")

    def run_demo(self, name="output"):
        target = self.base / name
        result = subprocess.run(
            [sys.executable, "-m", "emulatrx_demo", "demo", "--template", str(self.source), "--out", str(target)],
            cwd=ROOT, text=True, capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return result, target

    def assert_rejected(self, code):
        result, target = self.run_demo()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(code, result.stderr)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.base.glob(".emulatrx-stage-*")), [])

    def test_outputs_are_labeled_hashed_and_reproducible(self):
        first, a = self.run_demo("one")
        second, b = self.run_demo("two")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        expected = {"synthetic/cases.csv", "results/summary.json", "run_manifest.json", "PURPOSE.txt"}
        for target in (a, b):
            actual = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
            self.assertEqual(actual, expected)
            manifest = json.loads((target / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "synthetic_demo_generated")
            self.assertEqual(manifest["simulation_label"], SIMULATION_LABEL)
            self.assertEqual(manifest["seed"], 42)
            self.assertEqual(manifest["case_count"], 12)
            self.assertEqual(manifest["input_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
            self.assertRegex(manifest["code_commit"], r"^[0-9a-f]{40}$")
            self.assertIn("python", manifest["dependency_versions"])
            self.assertIn("generator_version", manifest)
            self.assertIn("run_at_utc", manifest)
            self.assertEqual([k for k in manifest if k.endswith("_utc")], ["run_at_utc"])
            for path, digest in manifest["output_sha256"].items():
                self.assertEqual(digest, hashlib.sha256((target / path).read_bytes()).hexdigest())
            with (target / "synthetic/cases.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(row["simulation_label"] == SIMULATION_LABEL for row in rows))
            self.assertEqual(len({row["synthetic_case_id"] for row in rows}), 12)
            summary = json.loads((target / "results/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["simulation_label"], SIMULATION_LABEL)
            self.assertEqual(summary["case_count"], 12)
            self.assertIn(SIMULATION_LABEL, (target / "PURPOSE.txt").read_text(encoding="utf-8"))
        for path in ("synthetic/cases.csv", "results/summary.json", "PURPOSE.txt"):
            self.assertEqual((a / path).read_bytes(), (b / path).read_bytes())
        am = json.loads((a / "run_manifest.json").read_text(encoding="utf-8"))
        bm = json.loads((b / "run_manifest.json").read_text(encoding="utf-8"))
        am.pop("run_at_utc")
        bm.pop("run_at_utc")
        self.assertEqual(am, bm)

    def test_source_change_and_seed_change_are_recorded(self):
        first, a = self.run_demo("one")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.template["seed"] = 43
        self.source.write_text(json.dumps(self.template), encoding="utf-8")
        second, b = self.run_demo("two")
        self.assertEqual(second.returncode, 0, second.stderr)
        am = json.loads((a / "run_manifest.json").read_text(encoding="utf-8"))
        bm = json.loads((b / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertNotEqual(am["input_sha256"], bm["input_sha256"])
        self.assertNotEqual((a / "synthetic/cases.csv").read_bytes(), (b / "synthetic/cases.csv").read_bytes())

    def test_strict_invalid_inputs_and_case_limit(self):
        changes = [
            ({"seed": None}, "INVALID_TYPE"),
            ({"seed": True}, "INVALID_TYPE"),
            ({"seed": -1}, "INVALID_SEED"),
            ({"case_count": 0}, "INVALID_CASE_COUNT"),
            ({"case_count": MAX_CASES + 1}, "INVALID_CASE_COUNT"),
            ({"case_count": 1.5}, "INVALID_TYPE"),
            ({"patient_id": "FORBIDDEN_INPUT"}, "UNKNOWN_FIELD"),
            ({"clinical_result": "FORBIDDEN_INPUT"}, "UNKNOWN_FIELD"),
        ]
        for change, code in changes:
            with self.subTest(change=change):
                self.source.write_text(json.dumps({**self.template, **change}), encoding="utf-8")
                self.assert_rejected(code)

    def test_existing_output_is_preserved(self):
        target = self.base / "output"
        target.mkdir()
        (target / "keep.txt").write_text("keep", encoding="utf-8")
        result, _ = self.run_demo()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OUTPUT_EXISTS", result.stderr)
        self.assertEqual((target / "keep.txt").read_text(encoding="utf-8"), "keep")
        self.assertEqual(list(self.base.glob(".emulatrx-stage-*")), [])

    def test_maximum_case_count_has_bounded_output(self):
        self.template["case_count"] = MAX_CASES
        self.source.write_text(json.dumps(self.template), encoding="utf-8")
        result, target = self.run_demo()
        self.assertEqual(result.returncode, 0, result.stderr)
        cases = target / "synthetic/cases.csv"
        self.assertLess(cases.stat().st_size, 2 * 1024 * 1024)
        with cases.open(encoding="utf-8", newline="") as stream:
            self.assertEqual(sum(1 for _ in stream), MAX_CASES + 1)

    def test_unrelated_patient_file_is_never_read_or_published(self):
        decoy = self.base / "patient_records.csv"
        decoy.write_text("patient_id,marker\nSYNTHETIC_ONLY,UNIQUE_DECOY_TOKEN\n", encoding="utf-8")
        result, target = self.run_demo()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any("UNIQUE_DECOY_TOKEN" in p.read_text(encoding="utf-8")
                             for p in target.rglob("*") if p.is_file()))

    def test_generation_failure_cleans_stage(self):
        from emulatrx_demo import __main__ as entry
        with mock.patch.object(entry, "_write_cases", side_effect=OSError("injected failure")):
            with self.assertRaises(entry.ContractError) as raised:
                entry.demo(self.source, self.base / "output")
        self.assertEqual(raised.exception.code, "OUTPUT_IO_ERROR")
        self.assertFalse((self.base / "output").exists())
        self.assertEqual(list(self.base.glob(".emulatrx-stage-*")), [])


if __name__ == "__main__":
    unittest.main()
