"""Per-request user namespace for WorkBuddy / multi-user MCP isolation.

The active namespace is a ``ContextVar`` so HTTP requests and stdio processes
do not share mutable global state. ``default`` preserves the historical
single-user directory layout.
"""

from __future__ import annotations

from contextvars import ContextVar
import re


CURRENT_USER_NAMESPACE: ContextVar[str] = ContextVar("emulatrx_user_namespace", default="default")

# 1–63 characters, must start with an alphanumeric, no path separators.
USER_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}\Z")


class InvalidUserNamespace(ValueError):
    """Raised when a caller-supplied namespace fails the strict slug check."""

    code = "INVALID_USER_NAMESPACE"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


def coerce_user_namespace(raw: str | None) -> str:
    """Return a safe namespace. Missing or blank input is the legacy ``default`` user."""
    if raw is None:
        return "default"
    if not isinstance(raw, str):
        raise InvalidUserNamespace("INVALID_USER_NAMESPACE: user namespace must be a string")
    text = raw.strip()
    if text == "":
        return "default"
    segments = re.split(r"[/\\]", text)
    if any(part in {".", ".."} for part in segments) or not USER_NAMESPACE_RE.fullmatch(text):
        raise InvalidUserNamespace(
            "INVALID_USER_NAMESPACE: user namespace must match "
            "^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$ and must not contain path segments"
        )
    return text


def resolve_user_namespace(explicit: str | None = None) -> str:
    """Resolve an explicit namespace, otherwise the active request/process context."""
    if explicit is None:
        return coerce_user_namespace(CURRENT_USER_NAMESPACE.get())
    return coerce_user_namespace(explicit)


def namespace_from_headers(headers: dict[bytes, bytes]) -> str:
    """Read ``X-EmulatRx-User``, falling back to ``X-WorkBuddy-User`` only when the primary header is absent."""
    if b"x-emulatrx-user" in headers:
        raw = headers[b"x-emulatrx-user"].decode("latin-1")
    elif b"x-workbuddy-user" in headers:
        raw = headers[b"x-workbuddy-user"].decode("latin-1")
    else:
        raw = None
    return coerce_user_namespace(raw)
