"""Tests for evomerge.sanitize.scrubber — issue #70.

Covers:
  - each default pattern (JWT, AWS key, GitHub/OpenAI/Slack/Google tokens,
    PEM private key, bearer/authorization, generic key=value, email, IPv4,
    credit card, SSN, URL credentials)
  - clean text is left untouched
  - ScrubReport counts (scanned / modified / redactions / per_pattern)
  - nested payload scrubbing (dict + list) without mutating the input
  - dict keys are preserved, non-string leaves untouched
  - custom pattern sets
  - module-level convenience wrappers
"""
from __future__ import annotations

import re

from evomerge.sanitize import (
    DEFAULT_PATTERNS,
    ScrubPattern,
    ScrubReport,
    Scrubber,
    scrub_payload,
    scrub_text,
)


def _scrub(text: str) -> str:
    return Scrubber().scrub_text(text)[0]


# ---------------------------------------------------------------------------
# Individual pattern detection
# ---------------------------------------------------------------------------
class TestPatterns:
    def test_jwt(self):
        jwt = (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        out = _scrub(f"token is {jwt} ok")
        assert jwt not in out
        assert "[REDACTED_JWT]" in out

    def test_aws_access_key(self):
        out = _scrub("key AKIAIOSFODNN7EXAMPLE here")
        assert "AKIAIOSFODNN7EXAMPLE" not in out
        assert "[REDACTED_AWS_ACCESS_KEY]" in out

    def test_github_token(self):
        tok = "ghp_" + "a" * 36
        out = _scrub(f"clone with {tok}")
        assert tok not in out
        assert "[REDACTED_GITHUB_TOKEN]" in out

    def test_openai_key(self):
        tok = "sk-" + "A1b2C3d4" * 3
        out = _scrub(f"OPENAI_API_KEY={tok}")
        assert tok not in out
        assert "REDACTED" in out

    def test_slack_token(self):
        tok = "xoxb-123456789012-abcdefABCDEF"
        out = _scrub(f"slack {tok}")
        assert tok not in out
        assert "[REDACTED_SLACK_TOKEN]" in out

    def test_google_api_key(self):
        tok = "AIza" + "B" * 35
        out = _scrub(f"maps {tok}")
        assert tok not in out
        assert "[REDACTED_GOOGLE_API_KEY]" in out

    def test_private_key_block(self):
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIE...fakekeymaterial...AB\n"
            "-----END RSA PRIVATE KEY-----"
        )
        out = _scrub(f"here:\n{pem}\ndone")
        assert "fakekeymaterial" not in out
        assert "[REDACTED_PRIVATE_KEY]" in out

    def test_bearer_token(self):
        out = _scrub("Authorization: Bearer abc123DEF456ghi789")
        assert "abc123DEF456ghi789" not in out
        assert "[REDACTED_BEARER_TOKEN]" in out
        # scheme label is preserved
        assert "Bearer" in out

    def test_generic_api_key_assignment(self):
        for src in (
            'api_key="s3cr3tValue123"',
            "password: 'hunter2hunter2'",
            "access_key = MyLongAccessKey99",
        ):
            out = _scrub(src)
            assert "[REDACTED_API_KEY]" in out, src

    def test_email(self):
        out = _scrub("reach me at alice.smith@example.co.uk please")
        assert "alice.smith@example.co.uk" not in out
        assert "[REDACTED_EMAIL]" in out

    def test_ipv4(self):
        out = _scrub("server at 192.168.10.254 down")
        assert "192.168.10.254" not in out
        assert "[REDACTED_IPV4]" in out

    def test_credit_card(self):
        out = _scrub("card 4111 1111 1111 1111 expiring")
        assert "4111 1111 1111 1111" not in out
        assert "[REDACTED_CREDIT_CARD]" in out

    def test_ssn(self):
        out = _scrub("ssn 123-45-6789 on file")
        assert "123-45-6789" not in out
        assert "[REDACTED_SSN]" in out

    def test_url_credentials(self):
        out = _scrub("clone https://user:p4ssw0rd@github.com/x/y.git now")
        assert "p4ssw0rd" not in out
        assert "[REDACTED_URL_CREDENTIALS]" in out
        # scheme + host preserved
        assert "https://" in out
        assert "github.com" in out


# ---------------------------------------------------------------------------
# Clean text
# ---------------------------------------------------------------------------
class TestCleanText:
    def test_plain_text_untouched(self):
        src = "The quick brown fox writes idiomatic Python."
        out, report = Scrubber().scrub_text(src)
        assert out == src
        assert report.clean
        assert report.n_redactions == 0
        assert report.n_strings_modified == 0

    def test_empty_string(self):
        out, report = Scrubber().scrub_text("")
        assert out == ""
        assert report.clean


# ---------------------------------------------------------------------------
# ScrubReport accounting
# ---------------------------------------------------------------------------
class TestReport:
    def test_counts_single_string(self):
        out, report = Scrubber().scrub_text("a@b.com and c@d.org")
        assert report.n_strings_scanned == 1
        assert report.n_strings_modified == 1
        assert report.n_redactions == 2
        assert report.per_pattern["EMAIL"] == 2
        assert "EMAIL" in report.patterns_applied

    def test_to_dict_roundtrippable_shape(self):
        _, report = Scrubber().scrub_text("x@y.com")
        d = report.to_dict()
        assert set(d) == {
            "n_strings_scanned",
            "n_strings_modified",
            "n_redactions",
            "per_pattern",
            "patterns_applied",
        }
        assert d["per_pattern"]["EMAIL"] == 1


# ---------------------------------------------------------------------------
# Nested payloads
# ---------------------------------------------------------------------------
class TestPayload:
    def test_nested_dict_and_list(self):
        payload = {
            "tool_call": {
                "name": "http_get",
                "args": {
                    "headers": ["Authorization: Bearer sekret_TOKEN_1234567"],
                    "url": "https://api.example.com",
                },
            },
            "response": "emailed ops@example.com the AKIAIOSFODNN7EXAMPLE key",
            "retries": 3,
            "ok": True,
            "note": None,
        }
        cleaned, report = Scrubber().scrub_payload(payload)

        # secrets gone
        flat = repr(cleaned)
        assert "sekret_TOKEN_1234567" not in flat
        assert "ops@example.com" not in flat
        assert "AKIAIOSFODNN7EXAMPLE" not in flat
        # placeholders present
        assert "[REDACTED_BEARER_TOKEN]" in cleaned["tool_call"]["args"]["headers"][0]
        assert "[REDACTED_EMAIL]" in cleaned["response"]
        assert "[REDACTED_AWS_ACCESS_KEY]" in cleaned["response"]
        # non-string leaves untouched, keys preserved
        assert cleaned["retries"] == 3
        assert cleaned["ok"] is True
        assert cleaned["note"] is None
        assert set(cleaned["tool_call"]["args"]) == {"headers", "url"}
        assert report.n_redactions >= 3

    def test_input_not_mutated(self):
        payload = {"msg": "mail me at a@b.com", "nested": ["c@d.com"]}
        original = {"msg": "mail me at a@b.com", "nested": ["c@d.com"]}
        Scrubber().scrub_payload(payload)
        assert payload == original

    def test_tuple_preserved(self):
        cleaned, _ = Scrubber().scrub_payload(("a@b.com", 1))
        assert isinstance(cleaned, tuple)
        assert "[REDACTED_EMAIL]" in cleaned[0]
        assert cleaned[1] == 1


# ---------------------------------------------------------------------------
# Custom patterns + convenience wrappers
# ---------------------------------------------------------------------------
class TestCustomAndWrappers:
    def test_custom_pattern_set(self):
        pats = [ScrubPattern.from_regex("HEX_ID", r"\b[0-9a-f]{8}\b")]
        out, report = Scrubber(pats).scrub_text("id deadbeef here")
        assert "[REDACTED_HEX_ID]" in out
        assert report.patterns_applied == ["HEX_ID"]
        # default patterns are NOT applied
        assert report.per_pattern.get("EMAIL") is None

    def test_from_regex_default_placeholder(self):
        p = ScrubPattern.from_regex("FOO", r"foo")
        assert p.placeholder == "[REDACTED_FOO]"
        assert isinstance(p.pattern, re.Pattern)

    def test_custom_placeholder(self):
        p = ScrubPattern.from_regex("FOO", r"foo", placeholder="X")
        assert p.placeholder == "X"

    def test_module_wrappers(self):
        out, report = scrub_text("ping a@b.com")
        assert "[REDACTED_EMAIL]" in out
        assert isinstance(report, ScrubReport)

        cleaned, rep2 = scrub_payload({"x": "a@b.com"})
        assert "[REDACTED_EMAIL]" in cleaned["x"]
        assert rep2.n_redactions == 1

    def test_default_patterns_nonempty_and_named(self):
        names = {p.name for p in DEFAULT_PATTERNS}
        # the BSCODE vocabulary is represented
        assert {"JWT", "API_KEY", "EMAIL"} <= names
