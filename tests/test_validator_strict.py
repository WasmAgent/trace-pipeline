"""Strict tests for AEP validator hardening (P0-8).

Tests:
(a) jsonschema unavailable → validator raises ImportError
(b) Signature verification: missing signature when require_signature=True → fail
(c) Signature verification: valid Ed25519 signature → pass
(d) Signature verification: tampered payload → fail
(e) Signature verification: unknown key_id → fail
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
from typing import Any

import pytest


def _dsse_sign_record(record: dict[str, Any], private_key, key_id: str) -> dict[str, Any]:
    """Build a DSSE envelope over the record (current/only signing profile)."""
    unsigned = {
        k: v for k, v in record.items()
        if k not in ("signature", "dsse_envelope", "timestamp_proof")
    }
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": f"urn:wasmagent:run:{record['run_id']}", "digest": {"sha256": digest}}],
        "predicateType": "https://wasmagent.dev/attestations/aep/v0.4",
        "predicate": unsigned,
    }
    payload_json = json.dumps(statement, separators=(",", ":"), ensure_ascii=False).encode()
    payload_b64 = base64.b64encode(payload_json).decode()
    payload_type = "application/vnd.in-toto+json"
    # PAE covers the DECODED serialized body bytes (DSSE 1.0.2 §2).
    pt = payload_type.encode("utf-8")
    pae = (
        b"DSSEv1 "
        + str(len(pt)).encode("ascii")
        + b" "
        + pt
        + b" "
        + str(len(payload_json)).encode("ascii")
        + b" "
        + payload_json
    )
    sig_bytes = private_key.sign(pae)
    signed = dict(record)
    signed["dsse_envelope"] = {
        "payloadType": payload_type,
        "payload": payload_b64,
        "signatures": [{"keyid": key_id, "sig": base64.urlsafe_b64encode(sig_bytes).decode()}],
    }
    return signed

# ---------------------------------------------------------------------------
# Helpers to build minimal valid AEP records
# ---------------------------------------------------------------------------

def _minimal_record(**kwargs) -> dict[str, Any]:
    base = {
        "schema_version": "aep/v0.2",
        "run_id": "test-run-001",
        "created_at_ms": 1700000000000,
    }
    base.update(kwargs)
    return base


def _sign_record(record: dict[str, Any], private_key, key_id: str) -> dict[str, Any]:
    """Sign a record with an Ed25519 private key and return the record with a signature block."""
    payload_dict = {k: v for k, v in record.items() if k != "signature"}
    payload_bytes = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    sig_bytes = private_key.sign(payload_bytes)
    sig_b64 = base64.urlsafe_b64encode(sig_bytes).decode()
    signed = dict(record)
    signed["signature"] = {"alg": "ed25519", "key_id": key_id, "sig": sig_b64}
    return signed


# ---------------------------------------------------------------------------
# (a) jsonschema import failure → hard ImportError at module import time
# ---------------------------------------------------------------------------

class TestJsonschemaHardDependency:
    def test_import_error_when_jsonschema_missing(self, monkeypatch):
        """Importing evomerge.validate.aep without jsonschema must raise ImportError."""
        # Remove cached modules so we can re-import with a patched builtins
        modules_to_remove = [
            key for key in sys.modules if "evomerge.validate.aep" in key
        ]
        for mod in modules_to_remove:
            del sys.modules[mod]

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "jsonschema":
                raise ImportError("mocked: jsonschema not available")
            return original_import(name, *args, **kwargs)

        with monkeypatch.context() as m:
            # Patch sys.modules to simulate jsonschema being absent
            m.setitem(sys.modules, "jsonschema", None)  # None triggers ImportError on import
            with pytest.raises(ImportError, match="jsonschema"):
                # Remove the cached module to force re-import
                for key in list(sys.modules.keys()):
                    if "evomerge.validate.aep" in key:
                        del sys.modules[key]
                import evomerge.validate.aep  # noqa: F401


# ---------------------------------------------------------------------------
# (b)+(c)+(d)+(e) Signature verification tests
# ---------------------------------------------------------------------------

class TestAEPSignatureVerification:
    @pytest.fixture
    def ed25519_keypair(self):
        """Generate a fresh Ed25519 keypair for testing."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        pub_raw = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        pub_b64 = base64.urlsafe_b64encode(pub_raw).decode()
        return private_key, public_key, pub_b64

    def test_missing_signature_fails(self):
        """When require_signature=True and record has no signature/envelope → unsigned."""
        from evomerge.validate.aep import validate_aep_record
        record = _minimal_record()
        result = validate_aep_record(record, require_signature=True)
        assert not result.passed
        assert result.authenticity_mode == "unsigned"

    def test_valid_signature_passes(self, monkeypatch, ed25519_keypair):
        """A correctly DSSE-signed record with the public key in env → pass."""
        from evomerge.validate.aep import validate_aep_record
        private_key, public_key, pub_b64 = ed25519_keypair
        key_id = "test-key-v1"
        env_var = "WASMAGENT_AEP_PUBKEY_TEST_KEY_V1"
        record = _dsse_sign_record(_minimal_record(), private_key, key_id)
        monkeypatch.setenv(env_var, pub_b64)
        result = validate_aep_record(record, require_signature=True)
        assert result.passed, f"Expected pass but got errors: {result.errors}"
        assert result.authenticity_mode == "dsse-valid"
        assert result.authenticity_valid is True

    def test_tampered_payload_fails(self, monkeypatch, ed25519_keypair):
        """Signing a record then modifying the payload must fail verification."""
        from evomerge.validate.aep import validate_aep_record
        private_key, public_key, pub_b64 = ed25519_keypair
        key_id = "test-key-v1"
        env_var = "WASMAGENT_AEP_PUBKEY_TEST_KEY_V1"
        record = _dsse_sign_record(_minimal_record(), private_key, key_id)
        # Tamper: change run_id after signing
        record["run_id"] = "TAMPERED-run-id"
        monkeypatch.setenv(env_var, pub_b64)
        result = validate_aep_record(record, require_signature=True)
        assert not result.passed
        assert any("authenticity" in e for e in result.errors)

    def test_unknown_key_id_fails(self, monkeypatch, ed25519_keypair):
        """A key_id with no matching env var must fail."""
        from evomerge.validate.aep import validate_aep_record
        private_key, _, _ = ed25519_keypair
        key_id = "nonexistent-key-v99"
        record = _dsse_sign_record(_minimal_record(), private_key, key_id)
        # Do NOT set the env var
        monkeypatch.delenv("WASMAGENT_AEP_PUBKEY_NONEXISTENT_KEY_V99", raising=False)
        result = validate_aep_record(record, require_signature=True)
        assert not result.passed
        assert any("authenticity" in e for e in result.errors)

    def test_no_signature_required_no_check(self):
        """When require_signature=False (default), missing signature is not an error."""
        from evomerge.validate.aep import validate_aep_record
        record = _minimal_record()
        result = validate_aep_record(record, require_signature=False)
        # Should not fail due to missing signature
        sig_errors = [e for e in result.errors if "signature" in e]
        assert sig_errors == [], f"Unexpected signature errors: {sig_errors}"


class TestDSSEBase64Interop:
    """DSSE 1.0.2: "Either standard or URL-safe base64 encodings are allowed.
    Signers may use either, and verifiers MUST accept either."

    The vector uses '>'/'?' runs in the run_id so the standard payload encoding
    deterministically contains both '+' and '/', and regenerates keys until the
    signature encoding also contains an alternate-alphabet character — the
    URL-safe re-encodings then genuinely exercise the alternate alphabet
    instead of being no-ops.
    """

    KEY_ID = "test-key-v1"
    ENV_VAR = "WASMAGENT_AEP_PUBKEY_TEST_KEY_V1"

    @staticmethod
    def _make_keypair():
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        private_key = Ed25519PrivateKey.generate()
        pub_raw = (
            private_key.public_key()
            .public_bytes(Encoding.Raw, PublicFormat.Raw)
        )
        pub_b64 = base64.urlsafe_b64encode(pub_raw).decode()
        return private_key, pub_b64

    @classmethod
    def _signed_vector(cls):
        """(private_key, pub_b64, signed_record) with '+'/'/' in the payload
        standard base64 encoding and '-'/_' in the URL-safe signature encoding
        — both alternate alphabets are exercised for real.

        Note: the test signing helper emits URL-safe signatures natively, so
        the signature vector keeps that encoding and the standard-signature
        case converts it to the standard alphabet instead.
        """
        # '>' (0x3E) and '?' (0x3F) at aligned third-byte positions encode to
        # '+' and '/' respectively; long runs make the alignment a certainty.
        run_id = f"run-b64-{'>' * 9}-{'?' * 9}"
        for _ in range(64):
            private_key, pub_b64 = cls._make_keypair()
            record = _dsse_sign_record(_minimal_record(run_id=run_id), private_key, cls.KEY_ID)
            env = record["dsse_envelope"]
            payload_ok = "+" in env["payload"] and "/" in env["payload"]
            sig = env["signatures"][0]["sig"]
            sig_ok = "-" in sig or "_" in sig
            if payload_ok and sig_ok:
                return private_key, pub_b64, record
        pytest.fail("could not construct a base64 vector with alternate-alphabet characters")

    @staticmethod
    def _to_urlsafe(s: str) -> str:
        return s.replace("+", "-").replace("/", "_")

    @staticmethod
    def _to_standard(s: str) -> str:
        return s.replace("-", "+").replace("_", "/")

    def test_standard_payload_and_signature_passes(self, monkeypatch):
        """URL-safe signature converted to the standard alphabet must verify."""
        from evomerge.validate.aep import validate_aep_record
        _, pub_b64, record = self._signed_vector()
        env = record["dsse_envelope"]
        assert "+" in env["payload"] and "/" in env["payload"], "vector must exercise '+' and '/'"
        env["signatures"][0]["sig"] = self._to_standard(env["signatures"][0]["sig"])
        monkeypatch.setenv(self.ENV_VAR, pub_b64)
        result = validate_aep_record(record, require_signature=True)
        assert result.passed, f"errors: {result.errors}"

    def test_urlsafe_payload_passes(self, monkeypatch):
        from evomerge.validate.aep import validate_aep_record
        _, pub_b64, record = self._signed_vector()
        env = record["dsse_envelope"]
        env["payload"] = self._to_urlsafe(env["payload"])
        monkeypatch.setenv(self.ENV_VAR, pub_b64)
        result = validate_aep_record(record, require_signature=True)
        assert result.passed, f"URL-safe payload must be accepted; errors: {result.errors}"

    def test_urlsafe_signature_passes(self, monkeypatch):
        from evomerge.validate.aep import validate_aep_record
        _, pub_b64, record = self._signed_vector()
        env = record["dsse_envelope"]
        assert "-" in env["signatures"][0]["sig"] or "_" in env["signatures"][0]["sig"], (
            "vector signature must be URL-safe encoded"
        )
        monkeypatch.setenv(self.ENV_VAR, pub_b64)
        result = validate_aep_record(record, require_signature=True)
        assert result.passed, f"URL-safe signature must be accepted; errors: {result.errors}"

    def test_urlsafe_payload_and_signature_pass(self, monkeypatch):
        from evomerge.validate.aep import validate_aep_record
        _, pub_b64, record = self._signed_vector()
        env = record["dsse_envelope"]
        env["payload"] = self._to_urlsafe(env["payload"])
        monkeypatch.setenv(self.ENV_VAR, pub_b64)
        result = validate_aep_record(record, require_signature=True)
        assert result.passed, f"errors: {result.errors}"

    def test_mixed_alphabet_is_accepted_uniquely(self, monkeypatch):
        """A signature mixing '+' with '-' decodes uniquely under the bijection
        ('-'→'+', '_'→'/'): it must verify identically to the pure forms."""
        from evomerge.validate.aep import validate_aep_record
        _, pub_b64, record = self._signed_vector()
        env = record["dsse_envelope"]
        sig = env["signatures"][0]["sig"]
        if "-" in sig:
            env["signatures"][0]["sig"] = sig.replace("-", "+", 1)
        else:
            env["signatures"][0]["sig"] = sig.replace("_", "/", 1)
        monkeypatch.setenv(self.ENV_VAR, pub_b64)
        result = validate_aep_record(record, require_signature=True)
        assert result.passed, f"mixed alphabet decodes uniquely and must pass; errors: {result.errors}"

    def test_tampered_payload_type_fails(self, monkeypatch):
        """payloadType is inside the PAE: mutating it must invalidate the sig."""
        from evomerge.validate.aep import validate_aep_record
        private_key, pub_b64, record = self._signed_vector()
        record["dsse_envelope"]["payloadType"] = "application/json"
        monkeypatch.setenv(self.ENV_VAR, pub_b64)
        result = validate_aep_record(record, require_signature=True)
        assert not result.passed
        assert result.authenticity_mode == "invalid"
