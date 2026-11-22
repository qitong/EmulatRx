"""Live ClinicalTrials.gov metadata intake that persists a receipt, never study values."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ._common import ContractError, _check_destination, _json_bytes


NCT_RE = re.compile(r"NCT[0-9]{8}\Z")
API_ROOT = "https://clinicaltrials.gov/api/v2"
MAX_RESPONSE_BYTES = 1024 * 1024
TIMEOUT_SECONDS = 15

# API v2's fields parameter names nested leaves. These are the complete intake
# boundary; free text, contacts, eligibility, result numbers and descriptions
# are intentionally absent.
REQUESTED_LEAF_PATHS = (
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.designModule.studyType",
    "protocolSection.designModule.phases",
    "protocolSection.designModule.designInfo.allocation",
    "protocolSection.designModule.designInfo.interventionModel",
    "protocolSection.designModule.designInfo.primaryPurpose",
    "protocolSection.designModule.designInfo.maskingInfo.masking",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.armsInterventionsModule.armGroups.label",
    "protocolSection.armsInterventionsModule.armGroups.type",
    "protocolSection.armsInterventionsModule.interventions.name",
    "protocolSection.armsInterventionsModule.interventions.type",
    "protocolSection.outcomesModule.primaryOutcomes.measure",
    "protocolSection.outcomesModule.primaryOutcomes.timeFrame",
    "hasResults",
)
REQUIRED_LEAF_PATHS = (
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.designModule.studyType",
    "hasResults",
)
PROJECTED_LEAF_PATHS = REQUESTED_LEAF_PATHS
ARRAY_PATHS = {
    "protocolSection.designModule.phases",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.armsInterventionsModule.armGroups",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.outcomesModule.primaryOutcomes",
}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def urlopen(request, timeout):
    """Keep the transport on the fixed official host, including redirects."""
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _get_json(url):
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "EmulatRx-source-receipt/1.0"})
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise ContractError("HTTP_ERROR", f"ClinicalTrials.gov returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ContractError("NETWORK_ERROR", "ClinicalTrials.gov request failed") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ContractError("RESPONSE_TOO_LARGE", "API response exceeds 1 MiB")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError("INVALID_RESPONSE", "API returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError("INVALID_RESPONSE", "API response must be an object")
    return raw, value


def _validate_version(value):
    version = value.get("apiVersion")
    timestamp = value.get("dataTimestamp")
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version):
        raise ContractError("INVALID_VERSION", "apiVersion must be a short version token")
    if not isinstance(timestamp, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|\+00:00)?",
        timestamp,
    ):
        raise ContractError("INVALID_TIMESTAMP", "dataTimestamp must be an ISO date-time")
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("INVALID_TIMESTAMP", "dataTimestamp is not ISO formatted") from exc
    # The official /version endpoint currently emits a timezone-free ISO value.
    # Preserve that source value without assigning it a timezone; offset-bearing
    # values must explicitly be UTC.
    return version, timestamp


def _tree(paths):
    root = {}
    for path in paths:
        node = root
        for part in path.split("."):
            node = node.setdefault(part, {})
    return root


WHITELIST_TREE = _tree(REQUESTED_LEAF_PATHS)


def _project(value, schema, path=""):
    if not schema:
        if path in ARRAY_PATHS:
            if not isinstance(value, list) or any(type(item) is not str for item in value):
                raise ContractError("INVALID_FIELD", f"{path} must be an array of strings")
            return list(value)
        expected = bool if path == "hasResults" else str
        if type(value) is not expected:
            raise ContractError("INVALID_FIELD", f"{path} must be {expected.__name__}")
        return value
    if path in ARRAY_PATHS:
        if not isinstance(value, list):
            raise ContractError("INVALID_FIELD", f"{path} must be an array")
        return [_project(item, schema, path + "[]") for item in value]
    if not isinstance(value, dict):
        raise ContractError("INVALID_FIELD", f"{path or 'study'} must be an object")
    return {key: _project(value[key], child, f"{path}.{key}" if path else key)
            for key, child in schema.items() if key in value}


def _path_present(value, path):
    nodes = [value]
    for part in path.split("."):
        next_nodes = []
        for node in nodes:
            if isinstance(node, list):
                next_nodes.extend(item.get(part) for item in node if isinstance(item, dict) and part in item)
            elif isinstance(node, dict) and part in node:
                next_nodes.append(node[part])
        nodes = next_nodes
        if not nodes:
            return False
    return True


def fetch_receipt(nct_id, out):
    """Fetch both official endpoints and atomically publish only source/metadata.json."""
    if not isinstance(nct_id, str) or not NCT_RE.fullmatch(nct_id):
        raise ContractError("INVALID_NCT_ID", "expected NCT followed by eight digits")
    out = Path(out)
    _check_destination(out)
    version_url = f"{API_ROOT}/version"
    study_url = f"{API_ROOT}/studies/{nct_id}?{urlencode({'fields': ','.join(REQUESTED_LEAF_PATHS)})}"
    _, version_data = _get_json(version_url)
    api_version, data_timestamp = _validate_version(version_data)
    raw, study = _get_json(study_url)
    projection = _project(study, WHITELIST_TREE)
    for path in REQUIRED_LEAF_PATHS:
        if not _path_present(projection, path):
            raise ContractError("MISSING_FIELD", f"required study leaf missing: {path}")
    actual_nct = projection["protocolSection"]["identificationModule"]["nctId"]
    if actual_nct != nct_id:
        raise ContractError("NCT_MISMATCH", "study NCT ID differs from request")
    for module, key in (("identificationModule", "briefTitle"), ("statusModule", "overallStatus"),
                        ("designModule", "studyType")):
        if not projection["protocolSection"][module][key].strip():
            raise ContractError("MISSING_FIELD", f"required study value is blank: {module}.{key}")
    canonical = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt = {
        "nctId": nct_id,
        "recordUrl": f"https://clinicaltrials.gov/study/{nct_id}",
        "requestUrls": [version_url, study_url],
        "retrievedAtUtc": datetime.now(timezone.utc).isoformat(),
        "apiVersion": api_version,
        "dataTimestamp": data_timestamp,
        "requestedLeafPaths": list(REQUESTED_LEAF_PATHS),
        "projectedLeafPaths": [path for path in PROJECTED_LEAF_PATHS if _path_present(projection, path)],
        "rawResponseSha256": hashlib.sha256(raw).hexdigest(),
        "projectionSha256": hashlib.sha256(canonical).hexdigest(),
        "attribution": "ClinicalTrials.gov, U.S. National Library of Medicine",
        "notice": "Public trial registration metadata is not patient data or clinical efficacy evidence.",
    }
    stage = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=".emulatrx-source-stage-", dir=out.parent))
        (stage / "source").mkdir()
        (stage / "source" / "metadata.json").write_bytes(_json_bytes(receipt))
        if os.path.lexists(out):
            raise ContractError("OUTPUT_EXISTS", str(out))
        stage.rename(out)
        stage = None
    except OSError as exc:
        raise ContractError("OUTPUT_IO_ERROR", str(exc)) from exc
    finally:
        if stage is not None:
            shutil.rmtree(stage)
    return {"status": "source_receipt_created", "nctId": nct_id, "output_path": str(out)}
