"""evomerge.sanitize.scrubber — automatic PII / API-key / credential scrubbing.

Raw agent traces routinely capture secrets: a tool call passes an
``Authorization: Bearer …`` header, a model response echoes back an AWS access
key, a user pastes an email address.  If those payloads flow untouched into an
SFT/DPO corpus, the secrets are memorised by the trained model and leak on
generation.  This module removes them *before* training-record compilation.

Design
------
* Detection is **regex-based** over already-decoded text.  Every pattern is a
  :class:`ScrubPattern` (name + compiled regex + placeholder).  Matches are
  replaced with a stable, human-readable placeholder such as ``[REDACTED_JWT]``
  so downstream readers can see *that* a redaction happened without seeing the
  secret.
* :class:`Scrubber` walks arbitrarily nested payloads (dicts / lists / str) and
  scrubs every string leaf, returning a *new* structure — the input is never
  mutated.
* A :class:`ScrubReport` records, per pattern, how many replacements were made
  and how many string leaves were touched, so an export batch can be audited
  without logging the secrets themselves.

The default pattern set intentionally overlaps with, and extends, the
``BSCODE_PATTERNS`` names (``JWT``, ``API_KEY``, ``EMAIL``) declared in
:mod:`evomerge.validate.redaction`, keeping redaction-report vocabulary
consistent across the repo.  Unlike ``RedactionReport`` — which merely
*describes* an upstream redaction pass — this module actually *performs* the
substitution on data held in-process.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScrubPattern:
    """A single named detection rule.

    Attributes:
        name:        short label used in placeholders and reports, e.g. ``JWT``.
        pattern:     compiled regular expression matching the secret.
        placeholder: text substituted for each match.  Defaults to
                     ``[REDACTED_<NAME>]`` when constructed via
                     :meth:`from_regex`.
    """

    name: str
    pattern: re.Pattern[str]
    placeholder: str

    @classmethod
    def from_regex(
        cls,
        name: str,
        regex: str,
        *,
        flags: int = 0,
        placeholder: str | None = None,
    ) -> ScrubPattern:
        """Build a :class:`ScrubPattern` from a raw regex string."""
        return cls(
            name=name,
            pattern=re.compile(regex, flags),
            placeholder=placeholder or f"[REDACTED_{name}]",
        )


# ---------------------------------------------------------------------------
# Default pattern set
# ---------------------------------------------------------------------------
# Ordering matters: more specific / higher-entropy patterns run first so that,
# e.g., an AWS key or JWT is not partially eaten by the generic bearer-token or
# email rules.  Each string leaf is passed through every pattern in turn.
DEFAULT_PATTERNS: tuple[ScrubPattern, ...] = (
    # --- credentials embedded in URLs (user:pass@host) ---
    ScrubPattern.from_regex(
        "URL_CREDENTIALS",
        r"\b([a-z][a-z0-9+.\-]*://)[^\s:/@]+:[^\s:/@]+@",
        flags=re.IGNORECASE,
        # keep the scheme, drop the creds
        placeholder=r"\1[REDACTED_URL_CREDENTIALS]@",
    ),
    # --- JSON Web Tokens: three base64url segments separated by dots ---
    ScrubPattern.from_regex(
        "JWT",
        r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
    ),
    # --- AWS access key IDs ---
    ScrubPattern.from_regex(
        "AWS_ACCESS_KEY",
        r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA)[0-9A-Z]{16}\b",
    ),
    # --- GitHub personal-access / app tokens (ghp_, gho_, ghs_, ghu_, ghr_) ---
    ScrubPattern.from_regex(
        "GITHUB_TOKEN",
        r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    ),
    # --- OpenAI-style secret keys (sk-..., sk-proj-...) ---
    ScrubPattern.from_regex(
        "OPENAI_KEY",
        r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b",
    ),
    # --- Slack tokens (xoxb-, xoxp-, xoxa-, xoxr-) ---
    ScrubPattern.from_regex(
        "SLACK_TOKEN",
        r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
    ),
    # --- Google API keys ---
    ScrubPattern.from_regex(
        "GOOGLE_API_KEY",
        r"\bAIza[0-9A-Za-z_-]{35}\b",
    ),
    # --- PEM private-key blocks ---
    ScrubPattern.from_regex(
        "PRIVATE_KEY",
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
        r".*?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
        flags=re.DOTALL,
    ),
    # --- HTTP bearer / authorization header values ---
    ScrubPattern.from_regex(
        "BEARER_TOKEN",
        r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}",
        flags=re.IGNORECASE,
        placeholder="Bearer [REDACTED_BEARER_TOKEN]",
    ),
    # --- generic "key"/"token"/"secret"/"password" = value assignments ---
    # Matches shapes like  api_key="…"  password: '…'  secret=…  token => …
    ScrubPattern.from_regex(
        "API_KEY",
        r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key)\b"
        r"\s*[:=]{1,2}>?\s*"
        r"['\"]?([A-Za-z0-9._~+/=-]{6,})['\"]?",
        placeholder="[REDACTED_API_KEY]",
    ),
    # --- email addresses (PII) ---
    ScrubPattern.from_regex(
        "EMAIL",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    ),
    # --- IPv4 addresses (PII / network detail) ---
    ScrubPattern.from_regex(
        "IPV4",
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
    ),
    # --- credit-card-like 13-16 digit runs (optionally dashed/spaced) ---
    ScrubPattern.from_regex(
        "CREDIT_CARD",
        r"\b(?:\d[ -]?){13,16}\b",
    ),
    # --- US Social Security Numbers ---
    ScrubPattern.from_regex(
        "SSN",
        r"\b\d{3}-\d{2}-\d{4}\b",
    ),
)


@dataclass
class ScrubReport:
    """Audit record of a scrubbing pass.

    Records *counts only* — never the matched secrets — so a report can be
    persisted alongside an export batch without re-introducing the leak.

    Attributes:
        n_strings_scanned:  number of string leaves inspected.
        n_strings_modified: number of string leaves changed by >= 1 pattern.
        n_redactions:       total number of individual replacements made.
        per_pattern:        mapping of pattern name → replacement count.
        patterns_applied:   names of all patterns that were in the active set
                            (whether or not they matched anything).
    """

    n_strings_scanned: int = 0
    n_strings_modified: int = 0
    n_redactions: int = 0
    per_pattern: dict[str, int] = field(default_factory=dict)
    patterns_applied: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True when no redactions were required."""
        return self.n_redactions == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_strings_scanned": self.n_strings_scanned,
            "n_strings_modified": self.n_strings_modified,
            "n_redactions": self.n_redactions,
            "per_pattern": dict(self.per_pattern),
            "patterns_applied": list(self.patterns_applied),
        }


class Scrubber:
    """Removes secrets from text and nested payloads.

    Example:
        >>> s = Scrubber()
        >>> s.scrub_text("contact me at a@b.com")[0]
        'contact me at [REDACTED_EMAIL]'

    The instance is stateless across calls — each :meth:`scrub_text` /
    :meth:`scrub_payload` call produces its own :class:`ScrubReport`.
    """

    def __init__(self, patterns: Sequence[ScrubPattern] | None = None) -> None:
        self.patterns: tuple[ScrubPattern, ...] = tuple(
            patterns if patterns is not None else DEFAULT_PATTERNS
        )

    # -- text -------------------------------------------------------------
    def scrub_text(self, text: str) -> tuple[str, ScrubReport]:
        """Scrub a single string, returning ``(clean_text, report)``."""
        report = ScrubReport(
            patterns_applied=[p.name for p in self.patterns]
        )
        out = self._scrub_string(text, report)
        return out, report

    def _scrub_string(self, text: str, report: ScrubReport) -> str:
        if not isinstance(text, str) or not text:
            report.n_strings_scanned += 1
            return text
        report.n_strings_scanned += 1
        original = text
        for pat in self.patterns:
            text, n = pat.pattern.subn(pat.placeholder, text)
            if n:
                report.n_redactions += n
                report.per_pattern[pat.name] = (
                    report.per_pattern.get(pat.name, 0) + n
                )
        if text != original:
            report.n_strings_modified += 1
        return text

    # -- nested payloads --------------------------------------------------
    def scrub_payload(self, payload: Any) -> tuple[Any, ScrubReport]:
        """Recursively scrub every string leaf of a nested structure.

        Handles ``str``, ``Mapping`` (dict), and non-string ``Sequence``
        (list / tuple).  Dict *keys* are left untouched — only values are
        scrubbed — so the payload's shape is preserved.  Any other leaf type
        (int, float, bool, None) is returned unchanged.  The input object is
        never mutated; a new structure is returned.
        """
        report = ScrubReport(
            patterns_applied=[p.name for p in self.patterns]
        )
        cleaned = self._scrub_node(payload, report)
        return cleaned, report

    def _scrub_node(self, node: Any, report: ScrubReport) -> Any:
        if isinstance(node, str):
            return self._scrub_string(node, report)
        if isinstance(node, Mapping):
            return {k: self._scrub_node(v, report) for k, v in node.items()}
        # bytes/str are Sequences too; str handled above, bytes falls through
        if isinstance(node, (list, tuple)):
            scrubbed = [self._scrub_node(v, report) for v in node]
            return type(node)(scrubbed) if isinstance(node, tuple) else scrubbed
        return node


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (default pattern set)
# ---------------------------------------------------------------------------
_DEFAULT_SCRUBBER = Scrubber()


def scrub_text(text: str) -> tuple[str, ScrubReport]:
    """Scrub *text* with the default pattern set. See :meth:`Scrubber.scrub_text`."""
    return _DEFAULT_SCRUBBER.scrub_text(text)


def scrub_payload(payload: Any) -> tuple[Any, ScrubReport]:
    """Scrub *payload* with the default pattern set. See :meth:`Scrubber.scrub_payload`."""
    return _DEFAULT_SCRUBBER.scrub_payload(payload)
