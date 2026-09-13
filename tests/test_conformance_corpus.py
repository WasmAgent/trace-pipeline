"""Consumer-side corpus gate: trace-pipeline executes the central manifest.

When AEP_PROTOCOL_CORPUS points at the wasmagent-protocol conformance
checkout, this test enforces the manifest's structural and semantic
expectations with trace-pipeline's own validator — the shared corpus is a
live consumer-side gate, not protocol-side data. Without the variable the
module is skipped (e.g. hermetic local runs).

The authenticity test also executes the manifest's `authenticity` verdicts
with trace-pipeline's own DSSE verifier and the corpus seed key, making
trace-pipeline the third native authenticity path alongside the JS and Rust
verifiers.
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

import pytest

CORPUS = os.environ.get("AEP_PROTOCOL_CORPUS")

pytestmark = pytest.mark.skipif(
    not CORPUS or not Path(CORPUS, "manifest.json").exists(),
    reason="AEP_PROTOCOL_CORPUS not set (central corpus not checked out)",
)


def _records(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    return [json.loads(path.read_text(encoding="utf-8"))]


def _manifest() -> dict:
    """Load the corpus manifest, refusing a stale signing profile.

    This consumer implements exactly one signing construction — Ed25519 over
    PAE(payloadType, decoded serialized body bytes). A manifest describing
    anything else means the corpus and the verifier disagree; that must fail
    loudly, never silently verify under a retired profile."""
    manifest = json.loads((Path(CORPUS) / "manifest.json").read_text(encoding="utf8"))
    supported = "aep-dsse-ed25519-decoded-body-v1"
    got = manifest.get("signing_profile_id")
    if got != supported:
        pytest.fail(
            f"unsupported or stale signing_profile_id {got!r} — "
            f"this consumer implements {supported!r}"
        )
    return manifest


def test_central_corpus_structural_and_semantic() -> None:
    from evomerge.validate.aep import validate_aep_record

    manifest = _manifest()
    entries = manifest.get("conformance_target", [])
    assert entries, "manifest declares no current-target fixtures"

    executed = 0
    for entry in entries:
        fixture = Path(CORPUS) / entry["path"]
        structural = entry.get("structural")
        semantic = entry.get("semantic")
        if structural not in ("valid", "invalid") and semantic not in ("valid", "invalid"):
            continue

        for record in _records(fixture):
            result = validate_aep_record(record)
            if structural in ("valid", "invalid"):
                schema_ok = result.valid_schema
                assert schema_ok is (structural == "valid"), (
                    f"{entry['path']}: structural expected {structural}, "
                    f"errors={result.errors[:3]}"
                )
                executed += 1
            # Semantic expectations are only reachable on structurally valid
            # records — the schema enum/minimum reject unknown grades and
            # negative counts before the semantic layer runs.
            if semantic in ("valid", "invalid") and result.valid_schema:
                assert result.semantic_valid is (semantic == "valid"), (
                    f"{entry['path']}: semantic expected {semantic}, errors={result.errors[:3]}"
                )
                executed += 1

    assert executed >= 20, f"corpus executed too few checks: {executed}"


def test_central_corpus_authenticity() -> None:
    """Third native authenticity path: trace-pipeline's DSSE verifier executes
    the manifest's `authenticity` verdicts against the corpus signing key.

    The corpus ships its signing keys alongside the fixtures:
    dsse/js-verify-key.hex (key id conformance-seed-key-01, seed c0ffee00 x8)
    and dsse/rust-fixture-verify-key.hex (key id ci-sample-key, seed deadbeef
    x8). Each is registered under its envelope keyid via the env-var keystore
    for the duration of each verification."""
    from evomerge.validate.aep import verify_aep_authenticity

    manifest = _manifest()
    entries = manifest.get("conformance_target", [])
    assert entries, "manifest declares no current-target fixtures"

    def _corpus_key_b64(name: str) -> str:
        raw = bytes.fromhex((Path(CORPUS) / "dsse" / name).read_text(encoding="utf-8").strip())
        return base64.b64encode(raw).decode()

    key_by_id = {
        "conformance-seed-key-01": _corpus_key_b64("js-verify-key.hex"),
        "ci-sample-key": _corpus_key_b64("rust-fixture-verify-key.hex"),
    }
    for key_b64 in key_by_id.values():
        assert len(base64.b64decode(key_b64)) == 32, "corpus verifying keys must be raw 32-byte Ed25519"

    executed = 0
    for entry in entries:
        expected = entry.get("authenticity")
        if expected not in ("dsse-valid", "invalid"):
            continue
        fixture = Path(CORPUS) / entry["path"]
        for record in _records(fixture):
            envelope = record.get("dsse_envelope")
            if not isinstance(envelope, dict):
                continue
            sigs = envelope.get("signatures") or []
            key_id = (sigs[0] or {}).get("keyid") if sigs else None
            if not key_id:
                continue
            pub_b64 = key_by_id.get(key_id)
            if pub_b64 is None:
                continue  # unknown fixture key — not this test's subject
            env_var = "WASMAGENT_AEP_PUBKEY_" + re.sub(r"[^A-Za-z0-9]", "_", key_id).upper()
            old = os.environ.get(env_var)
            os.environ[env_var] = pub_b64
            try:
                result = verify_aep_authenticity(record)
            finally:
                if old is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = old
            assert result.valid is (expected == "dsse-valid"), (
                f"{entry['path']}: authenticity expected {expected}, "
                f"got valid={result.valid} mode={result.mode} detail={result.detail}"
            )
            executed += 1

    assert executed >= 8, f"authenticity corpus executed too few checks: {executed}"


def test_central_corpus_chain() -> None:
    """Chain assurance path: trace-pipeline's verify_aep_chain executes the
    manifest's `chain` verdicts (intact / partial / orphaned / broken /
    not-present) over every .jsonl fixture, using the same link-hash
    projection as the JS/Rust verifiers."""
    from evomerge.validate.aep import verify_aep_chain

    manifest = _manifest()
    entries = manifest.get("conformance_target", [])

    executed = 0
    for entry in entries:
        expected = entry.get("chain")
        if not expected or expected == "not-checked":
            continue
        fixture = Path(CORPUS) / entry["path"]
        if fixture.suffix != ".jsonl":
            continue
        records = _records(fixture)
        result = verify_aep_chain(records)
        assert result.status == expected, (
            f"{entry['path']}: chain expected {expected}, got {result.status}"
        )
        executed += 1

    assert executed >= 5, f"chain corpus executed too few checks: {executed}"
