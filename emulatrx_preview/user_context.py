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
    """Return a safe namespace for programmatic callers.

    ``None`` and a literally empty string are the legacy ``default`` user.
    HTTP header values do not use this shortcut: a header that is present but
    blank is rejected by :func:`presented_user_namespace`. Any other value is
    matched raw. Whitespace is not stripped, so padded values fail
    ``USER_NAMESPACE_RE``.
    """
    if raw is None:
        return "default"
    if not isinstance(raw, str):
        raise InvalidUserNamespace("INVALID_USER_NAMESPACE: user namespace must be a string")
    if raw == "":
        return "default"
    if not USER_NAMESPACE_RE.fullmatch(raw):
        raise InvalidUserNamespace(
            "INVALID_USER_NAMESPACE: user namespace must match "
            "^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$ and must not contain path segments"
        )
    return raw


def resolve_user_namespace(explicit: str | None = None) -> str:
    """Resolve an explicit namespace, otherwise the active request/process context."""
    if explicit is None:
        return coerce_user_namespace(CURRENT_USER_NAMESPACE.get())
    return coerce_user_namespace(explicit)


_USER_HEADER_NAMES = (b"x-emulatrx-user", b"x-workbuddy-user")


def _header_pairs(headers) -> list[tuple[bytes, bytes]]:
    if isinstance(headers, dict):
        items = headers.items()
    else:
        items = headers or []
    pairs: list[tuple[bytes, bytes]] = []
    for key, value in items:
        if isinstance(key, str):
            key = key.encode("latin-1")
        if isinstance(value, str):
            value = value.encode("latin-1")
        pairs.append((bytes(key), bytes(value)))
    return pairs


def presented_user_namespace(headers) -> str | None:
    """Return the user namespace carried by the raw ASGI header list.

    Zero ``X-EmulatRx-User`` / ``X-WorkBuddy-User`` entries means the header
    was omitted (``None``). A present empty value — including whitespace that
    HTTP/1.1 (``h11`` / ``uvicorn``) already stripped to ``b""`` — a value that
    fails the namespace pattern, a repeated ``X-EmulatRx-User``, or disagreeing
    primary and alias headers all raise :class:`InvalidUserNamespace`.
    """
    found = [
        (key.lower(), value)
        for key, value in _header_pairs(headers)
        if key.lower() in _USER_HEADER_NAMES
    ]
    if not found:
        return None
    primary = [value for name, value in found if name == b"x-emulatrx-user"]
    alias = [value for name, value in found if name == b"x-workbuddy-user"]
    if len(primary) > 1 or len(alias) > 1 or (primary and alias and primary[0] != alias[0]):
        raise InvalidUserNamespace(
            "INVALID_USER_NAMESPACE: duplicate or conflicting user namespace headers"
        )
    raw_bytes = primary[0] if primary else alias[0]
    if raw_bytes == b"":
        raise InvalidUserNamespace("INVALID_USER_NAMESPACE: user namespace header is blank")
    raw = raw_bytes.decode("latin-1")
    if not USER_NAMESPACE_RE.fullmatch(raw):
        raise InvalidUserNamespace(
            "INVALID_USER_NAMESPACE: user namespace must match "
            "^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$ and must not contain path segments"
        )
    return raw


def namespace_from_headers(headers) -> str:
    """Read a user header, or return ``default`` when the header was omitted.

    A present blank or invalid header raises. Callers that have a credential
    bound to a named user should use :func:`presented_user_namespace` so an
    omitted header can adopt that bound namespace instead of ``default``.
    """
    presented = presented_user_namespace(headers)
    if presented is None:
        return "default"
    return presented
