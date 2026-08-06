"""Tests for evomerge.sanitize.scrubber."""
import pytest
from evomerge.sanitize.scrubber import scrub, scrub_message

REDACTED = "[REDACTED]"


class TestKeyBasedRedaction:
    def test_top_level_api_key(self):
        assert scrub({"api_key": "sk-abc123"})["api_key"] == REDACTED

    def test_top_level_password(self):
        assert scrub({"password": "s3cr3t"})["password"] == REDACTED

    def test_top_level_token(self):
        assert scrub({"token": "tok_xyz"})["token"] == REDACTED

    def test_authorization(self):
        assert scrub({"authorization": "Bearer abc"})["authorization"] == REDACTED

    def test_nested_secret(self):
        result = scrub({"outer": {"secret": "my_secret", "name": "keep"}})
        assert result["outer"]["secret"] == REDACTED
        assert result["outer"]["name"] == "keep"

    def test_non_sensitive_key_unchanged(self):
        assert scrub({"username": "alice"})["username"] == "alice"
        assert scrub({"count": 42})["count"] == 42

    def test_aws_keys(self):
        result = scrub({
            "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        })
        assert result["aws_access_key_id"] == REDACTED
        assert result["aws_secret_access_key"] == REDACTED


class TestPatternBasedRedaction:
    def test_email(self):
        result = scrub("contact me at alice@example.com please")
        assert "alice@example.com" not in result
        assert REDACTED in result

    def test_ssn(self):
        result = scrub("SSN: 123-45-6789")
        assert "123-45-6789" not in result

    def test_bearer_token(self):
        result = scrub("Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig")
        assert "eyJhbGciOiJSUzI1NiJ9" not in result

    def test_aws_access_key_in_string(self):
        result = scrub("key=AKIAIOSFODNN7EXAMPLE used here")
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_non_pii_unchanged(self):
        assert scrub("hello world") == "hello world"
        assert scrub("count=42") == "count=42"


class TestNonMutation:
    def test_dict_not_mutated(self):
        original = {"api_key": "secret", "name": "alice"}
        scrub(original)
        assert original["api_key"] == "secret"  # unchanged

    def test_list_not_mutated(self):
        original = ["alice@example.com", "safe"]
        scrub(original)
        assert original[0] == "alice@example.com"

    def test_message_not_mutated(self):
        msg = {"tool_call": {"api_key": "secret"}, "id": "123"}
        scrub_message(msg)
        assert msg["tool_call"]["api_key"] == "secret"


class TestScrubMessage:
    def test_scrubs_tool_call(self):
        msg = {"tool_call": {"api_key": "abc", "name": "search"}, "id": "m1"}
        result = scrub_message(msg)
        assert result["tool_call"]["api_key"] == REDACTED
        assert result["tool_call"]["name"] == "search"
        assert result["id"] == "m1"

    def test_scrubs_model_response(self):
        msg = {"model_response": "my email is alice@example.com", "id": "m2"}
        result = scrub_message(msg)
        assert "alice@example.com" not in result["model_response"]

    def test_non_dict_passthrough(self):
        assert scrub_message("string") == "string"  # type: ignore[arg-type]

    def test_deterministic(self):
        msg = {"tool_call": {"api_key": "secret123"}}
        assert scrub_message(msg) == scrub_message(msg)


class TestDeterminsticAndStructure:
    def test_list_structure_preserved(self):
        result = scrub([1, "hello", {"token": "x"}])
        assert result[0] == 1
        assert result[1] == "hello"
        assert result[2]["token"] == REDACTED

    def test_nested_list_in_dict(self):
        result = scrub({"items": [{"password": "pw"}, {"name": "ok"}]})
        assert result["items"][0]["password"] == REDACTED
        assert result["items"][1]["name"] == "ok"

    def test_none_unchanged(self):
        assert scrub(None) is None

    def test_int_unchanged(self):
        assert scrub(42) == 42
