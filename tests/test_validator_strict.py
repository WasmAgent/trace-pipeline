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
