"""Small shared error and atomic output helpers for independent commands."""

import json
import os


class ContractError(Exception):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _check_destination(out):
    if os.path.lexists(out):
        raise ContractError("OUTPUT_EXISTS", str(out))
    if not out.parent.is_dir():
        raise ContractError("OUTPUT_PARENT_MISSING", str(out.parent))
