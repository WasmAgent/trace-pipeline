"""evomerge.sanitize.scrubber — PII and credential redaction for trace data."""
from __future__ import annotations

import re
from typing import Any

# Keys whose values should always be redacted regardless of content
_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "api-key",
    "authorization", "auth",
    "password", "passwd", "pwd",
    "token", "access_token", "refresh_token", "id_token",
    "secret", "client_secret",
    "credential", "credentials",
    "aws_access_key_id", "aws_secret_access_key", "aws_session_token",
    "private_key", "private_key_id",
    "x-api-key", "x-auth-token",
    "bearer",
})

# Patterns for PII and credentials in string values
_PATTERNS: list[re.Pattern[str]] = [
    # Email addresses
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
    # US phone numbers
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    # US SSNs
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # IPv4 addresses (not localhost/RFC1918 ranges — still redact all)
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # Bearer tokens
    re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    # AWS access key IDs
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Generic API keys: long alphanumeric strings after key= or key:
    re.compile(r"(?:key|token|secret|password)\s*[=:]\s*['\"]?([A-Za-z0-9\-._+/]{20,})['\"]?", re.IGNORECASE),
    # PEM private key blocks
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----.*?-----END (?:RSA |EC )?PRIVATE KEY-----", re.DOTALL),
    # UUIDs (commonly used as tokens/IDs)
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE),
]

_REDACTED = "[REDACTED]"


def _is_sensitive_key(key: str) -> bool:
    return key.lower().replace("-", "_") in _SENSITIVE_KEYS


def _redact_string(value: str) -> str:
    """Apply pattern-based redaction to a string."""
    for pattern in _PATTERNS:
        value = pattern.sub(_REDACTED, value)
    return value


def scrub(value: Any) -> Any:
    """Recursively scrub PII and credentials from a value.

    - Dict values whose keys match sensitive names are replaced with [REDACTED].
    - String values are pattern-scanned for PII/credentials.
    - Lists and tuples are recursively scrubbed.
    - Other types (int, float, bool, None) are returned unchanged.
    - Input is never mutated; a new object is always returned.
    """
    if isinstance(value, dict):
        return {
            k: _REDACTED if _is_sensitive_key(str(k)) else scrub(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub(item) for item in value)
    if isinstance(value, str):
        return _redact_string(value)
    return value


def scrub_message(message: dict) -> dict:
    """Scrub a trace message dict in-place of PII/credentials.

    Targets the ``tool_call``, ``model_response``, and ``response`` fields
    which are the primary carriers of sensitive data in training traces.
    Returns a new dict without mutating the input.
    """
    if not isinstance(message, dict):
        return message

    result = dict(message)
    for field in ("tool_call", "model_response", "response", "content"):
        if field in result:
            result[field] = scrub(result[field])
    return result
