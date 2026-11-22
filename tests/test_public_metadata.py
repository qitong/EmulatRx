"""Offline HTTP tests for the separate public metadata receipt command."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

from emulatrx_demo import __main__ as entry
from emulatrx_demo import SIMULATION_LABEL
from emulatrx_demo import public_metadata as metadata


NCT = "NCT12345678"  # Invented test identifier; never fetched from the network.
VERSION = {"apiVersion": "v-test-2", "dataTimestamp": "2026-11-20T12:34:56Z"}
STUDY = {
    "protocolSection": {
        "identificationModule": {"nctId": NCT, "briefTitle": "SYNTHETIC_SECRET_TITLE"},
        "statusModule": {"overallStatus": "RECRUITING"},
        "designModule": {"studyType": "INTERVENTIONAL"},
        "conditionsModule": {"conditions": ["SYNTHETIC_SECRET_CONDITION"]},
        "armsInterventionsModule": {"armGroups": [{"label": "SYNTHETIC_SECRET_ARM", "type": "EXPERIMENTAL", "description": "EXTRA_SECRET"}]},
        "descriptionModule": {"detailedDescription": "EXTRA_SECRET"},
    },
    "hasResults": False,
    "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": [{"paramType": "EXTRA_SECRET"}]}},
}


class FakeResponse(io.BytesIO):
    status = 200


class PublicMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.out = self.base / "receipt"
        self.requests = []

    def run_cli(self, nct=NCT, version=None, study=None, failure=None):
        version = VERSION if version is None else version
        study = STUDY if study is None else study

        def fake_urlopen(request, timeout):
            self.assertGreater(timeout, 0)
            url = request.full_url
            self.requests.append(url)
            parsed = urlsplit(url)
            self.assertEqual((parsed.scheme, parsed.netloc), ("https", "clinicaltrials.gov"))
            if failure and "/studies/" in parsed.path:
                raise failure
            if parsed.path == "/api/v2/version":
                return FakeResponse(json.dumps(version).encode())
            self.assertEqual(parsed.path, f"/api/v2/studies/{NCT}")
            self.assertEqual(set(parse_qs(parsed.query)), {"fields"})
            requested = parse_qs(parsed.query)["fields"][0].split(",")
            self.assertEqual(requested, list(metadata.REQUESTED_LEAF_PATHS))
            self.assertTrue(all("." in path or path == "hasResults" for path in requested))
            return FakeResponse(json.dumps(study).encode())

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(metadata, "urlopen", side_effect=fake_urlopen):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = entry.main(["metadata", "--nct-id", nct, "--out", str(self.out)])
        return code, stdout.getvalue(), stderr.getvalue()

    def assert_failure(self, expected, **kwargs):
        code, stdout, stderr = self.run_cli(**kwargs)
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn(expected, stderr)
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.base.glob(".emulatrx-source-stage-*")), [])

    def test_receipt_only_with_projection_and_recomputable_hashes(self):
        code, stdout, stderr = self.run_cli()
        self.assertEqual(code, 0, stderr)
        self.assertEqual({str(p.relative_to(self.out)) for p in self.out.rglob("*") if p.is_file()}, {"source/metadata.json"})
        receipt_bytes = (self.out / "source" / "metadata.json").read_bytes()
        receipt = json.loads(receipt_bytes)
        self.assertEqual(set(receipt), {
            "nctId", "recordUrl", "requestUrls", "retrievedAtUtc", "apiVersion",
            "dataTimestamp", "requestedLeafPaths", "projectedLeafPaths",
            "rawResponseSha256", "projectionSha256", "attribution", "notice",
        })
        self.assertEqual(receipt["nctId"], NCT)
        self.assertEqual(receipt["recordUrl"], f"https://clinicaltrials.gov/study/{NCT}")
        self.assertEqual(receipt["requestUrls"], self.requests)
        self.assertEqual(receipt["apiVersion"], VERSION["apiVersion"])
        self.assertEqual(receipt["dataTimestamp"], VERSION["dataTimestamp"])
        self.assertEqual(receipt["requestedLeafPaths"], list(metadata.REQUESTED_LEAF_PATHS))
        self.assertEqual(receipt["projectedLeafPaths"], [
            "protocolSection.identificationModule.nctId",
            "protocolSection.identificationModule.briefTitle",
            "protocolSection.statusModule.overallStatus",
            "protocolSection.designModule.studyType",
            "protocolSection.conditionsModule.conditions",
            "protocolSection.armsInterventionsModule.armGroups.label",
            "protocolSection.armsInterventionsModule.armGroups.type",
            "hasResults",
        ])
        self.assertEqual(receipt["rawResponseSha256"], hashlib.sha256(json.dumps(STUDY).encode()).hexdigest())
        expected_projection = {
            "protocolSection": {
                "identificationModule": {"nctId": NCT, "briefTitle": "SYNTHETIC_SECRET_TITLE"},
                "statusModule": {"overallStatus": "RECRUITING"},
                "designModule": {"studyType": "INTERVENTIONAL"},
                "conditionsModule": {"conditions": ["SYNTHETIC_SECRET_CONDITION"]},
                "armsInterventionsModule": {"armGroups": [{"label": "SYNTHETIC_SECRET_ARM", "type": "EXPERIMENTAL"}]},
            },
            "hasResults": False,
        }
        canonical = json.dumps(expected_projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(receipt["projectionSha256"], hashlib.sha256(canonical).hexdigest())
        self.assertNotIn("SYNTHETIC_SECRET_TITLE", receipt_bytes.decode())
        self.assertNotIn("SYNTHETIC_SECRET_CONDITION", receipt_bytes.decode())
        self.assertNotIn("SYNTHETIC_SECRET_ARM", receipt_bytes.decode())
        self.assertNotIn("EXTRA_SECRET", receipt_bytes.decode())
        self.assertNotIn("SYNTHETIC_SECRET_TITLE", stdout)

    def test_invalid_input_fails_before_http(self):
        self.assert_failure("INVALID_NCT_ID", nct="NCT1234567/../x")
        self.assertEqual(self.requests, [])

    def test_http_and_network_fail_without_snapshot_fallback(self):
        self.assert_failure("HTTP_ERROR", failure=HTTPError("https://clinicaltrials.gov", 404, "missing", {}, None))
        self.assert_failure("NETWORK_ERROR", failure=URLError("offline"))

    def test_missing_required_and_mismatched_id_fail(self):
        missing = json.loads(json.dumps(STUDY))
        del missing["protocolSection"]["identificationModule"]["briefTitle"]
        self.assert_failure("MISSING_FIELD", study=missing)
        wrong = json.loads(json.dumps(STUDY))
        wrong["protocolSection"]["identificationModule"]["nctId"] = "NCT87654321"
        self.assert_failure("NCT_MISMATCH", study=wrong)
        blank = json.loads(json.dumps(STUDY))
        blank["protocolSection"]["identificationModule"]["briefTitle"] = " "
        self.assert_failure("MISSING_FIELD", study=blank)

    def test_version_and_timestamp_are_validated(self):
        self.assert_failure("INVALID_VERSION", version={"apiVersion": "", "dataTimestamp": VERSION["dataTimestamp"]})
        self.assert_failure("INVALID_TIMESTAMP", version={"apiVersion": "v-test-2", "dataTimestamp": "yesterday"})
        code, _, stderr = self.run_cli(version={"apiVersion": "2.0.5", "dataTimestamp": "2026-11-20T10:00:05"})
        self.assertEqual(code, 0, stderr)

    def test_existing_output_preserved(self):
        self.out.mkdir()
        (self.out / "keep.txt").write_text("keep")
        code, _, stderr = self.run_cli()
        self.assertNotEqual(code, 0)
        self.assertIn("OUTPUT_EXISTS", stderr)
        self.assertEqual((self.out / "keep.txt").read_text(), "keep")

    def test_oversize_response_fails_before_any_write(self):
        def fake_urlopen(request, timeout):
            if request.full_url.endswith("/version"):
                return FakeResponse(json.dumps(VERSION).encode())
            return FakeResponse(b"x" * (metadata.MAX_RESPONSE_BYTES + 1))
        with mock.patch.object(metadata, "urlopen", side_effect=fake_urlopen):
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                code = entry.main(["metadata", "--nct-id", NCT, "--out", str(self.out)])
        self.assertEqual(code, 2)
        self.assertIn("RESPONSE_TOO_LARGE", stderr.getvalue())
        self.assertFalse(self.out.exists())

    def test_receipt_does_not_affect_seeded_cases(self):
        template = self.base / "template.json"
        template.write_text(json.dumps({
            "template_version": "1.0", "scenario": "synthetic_tutorial_v1",
            "case_count": 5, "seed": 42, "simulation_label": SIMULATION_LABEL,
        }), encoding="utf-8")
        first = self.base / "before"
        second = self.base / "after"
        entry.demo(template, first)
        code, _, stderr = self.run_cli()
        self.assertEqual(code, 0, stderr)
        entry.demo(template, second)
        self.assertEqual((first / "synthetic/cases.csv").read_bytes(),
                         (second / "synthetic/cases.csv").read_bytes())
        self.assertFalse((second / "source").exists())


if __name__ == "__main__":
    unittest.main()
