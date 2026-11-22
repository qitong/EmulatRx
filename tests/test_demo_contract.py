"""Purely synthetic tests for the standalone template validator."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "examples" / "synthetic_demo_v1.json"
LABEL = "教学模拟，非真实患者、非临床疗效证据"


class DemoContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.template = {
            "template_version": "1.0",
            "scenario": "synthetic_tutorial_v1",
            "case_count": 12,
            "seed": 42,
            "simulation_label": LABEL,
        }

    def run_cli(self, template=None, out="output"):
        source = self.base / "template.json"
        source.write_text(json.dumps(self.template if template is None else template), encoding="utf-8")
        target = self.base / out
        result = subprocess.run(
            [sys.executable, "-m", "emulatrx_demo", "validate", "--template", str(source), "--out", str(target)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return result, source, target

    def assert_rejected(self, changed, code):
        result, _, target = self.run_cli(changed)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(code, result.stderr)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.base.glob(".emulatrx-stage-*")), [])

    def test_bundled_template_is_valid_and_outputs_contract(self):
        template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        result, source, target = self.run_cli(template)
        self.assertEqual(result.returncode, 0, result.stderr)
        normalized = json.loads((target / "normalized_template.json").read_text(encoding="utf-8"))
        manifest = json.loads((target / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(normalized, template)
        self.assertEqual(manifest["input_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(manifest["template_version"], "1.0")
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(manifest["simulation_label"], LABEL)
        self.assertRegex(manifest["code_commit"], r"^[0-9a-f]{40}$")
        self.assertIs(type(manifest["code_dirty"]), bool)
        self.assertEqual(manifest["source_type"], "fixed_synthetic_template")
        self.assertEqual(manifest["activity"], "template_validation")
        self.assertEqual(manifest["output_paths"], ["normalized_template.json", "run_manifest.json"])
        self.assertEqual(manifest["status"], "validated_only")
        self.assertIn("validated_only", result.stdout)

    def test_normalized_content_reproducible_with_time_separate(self):
        first, _, first_out = self.run_cli(out="one")
        second, _, second_out = self.run_cli(out="two")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual((first_out / "normalized_template.json").read_bytes(), (second_out / "normalized_template.json").read_bytes())
        a = json.loads((first_out / "run_manifest.json").read_text(encoding="utf-8"))
        b = json.loads((second_out / "run_manifest.json").read_text(encoding="utf-8"))
        a.pop("run_at_utc")
        b.pop("run_at_utc")
        self.assertEqual(a, b)

    def test_missing_field(self):
        changed = dict(self.template)
        del changed["seed"]
        self.assert_rejected(changed, "MISSING_FIELD")

    def test_unknown_patient_field(self):
        self.assert_rejected({**self.template, "patient_id": "synthetic-1"}, "UNKNOWN_FIELD")

    def test_wrong_type_including_bool(self):
        self.assert_rejected({**self.template, "case_count": True}, "INVALID_TYPE")

    def test_nonpositive_case_count(self):
        self.assert_rejected({**self.template, "case_count": 0}, "INVALID_CASE_COUNT")

    def test_invalid_seed(self):
        for seed in (-1, 2**32):
            with self.subTest(seed=seed):
                self.assert_rejected({**self.template, "seed": seed}, "INVALID_SEED")

    def test_unknown_version(self):
        self.assert_rejected({**self.template, "template_version": "2.0"}, "UNKNOWN_VERSION")

    def test_unknown_scenario_and_label(self):
        self.assert_rejected({**self.template, "scenario": "clinical_trial"}, "INVALID_SCENARIO")
        self.assert_rejected({**self.template, "simulation_label": "real data"}, "INVALID_SIMULATION_LABEL")

    def test_existing_output_preserved(self):
        target = self.base / "output"
        target.mkdir()
        marker = target / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        result, _, _ = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OUTPUT_EXISTS", result.stderr)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_invalid_json_and_duplicate_key(self):
        source = self.base / "bad.json"
        target = self.base / "output"
        for raw in ('{', '{"seed": 1, "seed": 2}'):
            source.write_text(raw, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "emulatrx_demo", "validate", "--template", str(source), "--out", str(target)],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("INVALID_JSON", result.stderr)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
