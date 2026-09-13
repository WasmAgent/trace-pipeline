"""Consumer-side corpus gate: trace-pipeline executes the central manifest.

When AEP_PROTOCOL_CORPUS points at the wasmagent-protocol conformance
checkout, this test enforces the manifest's structural, semantic,
authenticity, and chain expectations with trace-pipeline's own validators —
the shared corpus is a live consumer-side gate, not protocol-side data.
Without the variable the module is skipped (e.g. hermetic local runs).

FAIL-CLOSED ACCOUNTING: every manifest target declaring a testable verdict
must be executed exactly once. A fixture that cannot be exercised (missing
envelope, unknown key id, missing key file) FAILS the suite — it is never
silently skipped, so adding an unhandled manifest fixture turns CI red.
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


def _key_b64(rel_path: str) -> str:
    raw = bytes.fromhex((Path(CORPUS) / rel_path).read_text(encoding="utf-8").strip())
    assert len(raw) == 32, f"{rel_path}: corpus verifying keys must be raw 32-byte Ed25519"
    return base64.b64encode(raw).decode()


def _manifest_key_map() -> dict[str, str]:
    """Resolve keyid → verifying-key PATH from the manifest itself.

    Honours the manifest's `verifying_keys.by_keyid` map (per-entry
    `verify_key` overrides are checked at the call site) — no per-consumer
    hardcoded key table. Values are corpus-relative paths; decoding to base64
    happens once, at the `_key_b64` call site."""
    vk = _manifest().get("verifying_keys", {})
    return {
        key_id: rel
        for key_id, rel in (vk.get("by_keyid") or {}).items()
    }


def test_central_corpus_structural_and_semantic() -> None:
    from evomerge.validate.aep import validate_aep_record

    manifest = _manifest()
    entries = manifest.get("conformance_target", [])
    assert entries, "manifest declares no current-target fixtures"

    structural_targets = {
        e["path"] for e in entries if e.get("structural") in ("valid", "invalid")
    }
    semantic_targets = {
        e["path"] for e in entries if e.get("semantic") in ("valid", "invalid")
    }
    executed_structural: set[str] = set()
    executed_semantic: set[str] = set()

    for entry in entries:
        fixture = Path(CORPUS) / entry["path"]
        records = _records(fixture)
        assert records, f"{entry['path']}: fixture contains no records"
        structural = entry.get("structural")
        semantic = entry.get("semantic")

        for record in records:
            result = validate_aep_record(record)
            if structural in ("valid", "invalid"):
                assert result.valid_schema is (structural == "valid"), (
                    f"{entry['path']}: structural expected {structural}, "
                    f"errors={result.errors[:3]}"
                )
            # Semantic expectations are only reachable on structurally valid
            # records — the schema enum/minimum reject unknown grades and
            # negative counts before the semantic layer runs.
            if semantic in ("valid", "invalid") and result.valid_schema:
                assert result.semantic_valid is (semantic == "valid"), (
                    f"{entry['path']}: semantic expected {semantic}, errors={result.errors[:3]}"
                )

        if structural in ("valid", "invalid"):
            executed_structural.add(entry["path"])
        if semantic in ("valid", "invalid"):
            executed_semantic.add(entry["path"])

    # Exact accounting: declared == executed (C-R01 analog for every layer).
    assert executed_structural == structural_targets, (
        f"structural targets not fully executed: "
        f"declared={sorted(structural_targets)} executed={sorted(executed_structural)}"
    )
    assert executed_semantic == semantic_targets, (
        f"semantic targets not fully executed: "
        f"declared={sorted(semantic_targets)} executed={sorted(executed_semantic)}"
    )


def test_central_corpus_authenticity() -> None:
    """Third native authenticity path: trace-pipeline's DSSE verifier executes
    the manifest's `authenticity` verdicts. Every declared target executes
    exactly once — a missing envelope, missing keyid, or unknown verifying key
    is a hard failure, never a silent skip."""
    from evomerge.validate.aep import verify_aep_authenticity

    manifest = _manifest()
    entries = manifest.get("conformance_target", [])
    assert entries, "manifest declares no current-target fixtures"

    by_keyid = _manifest_key_map()
    targets = [e for e in entries if e.get("authenticity") in ("dsse-valid", "invalid")]
    assert targets, "manifest declares no authenticity targets"
    executed_paths: set[str] = set()

    for entry in targets:
        expected = entry["authenticity"]
        fixture = Path(CORPUS) / entry["path"]
        records = _records(fixture)
        assert records, f"{entry['path']}: authenticity fixture contains no records"

        for record in records:
            envelope = record.get("dsse_envelope")
            assert isinstance(envelope, dict), (
                f"{entry['path']}: authenticity target lacks a DSSE envelope"
            )
            sigs = envelope.get("signatures")
            assert isinstance(sigs, list) and sigs, (
                f"{entry['path']}: envelope has no signatures"
            )
            key_id = (sigs[0] or {}).get("keyid")
            assert key_id, f"{entry['path']}: envelope signature carries no keyid"

            key_rel = entry.get("verify_key") or by_keyid.get(key_id)
            assert key_rel, (
                f"{entry['path']}: no verifying key for keyid={key_id!r} — "
                f"add it to manifest verifying_keys.by_keyid or the entry's verify_key"
            )
            pub_b64 = _key_b64(key_rel)

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
        executed_paths.add(entry["path"])

    assert executed_paths == {e["path"] for e in targets}, (
        f"authenticity targets not fully executed: "
        f"declared={sorted({e['path'] for e in targets})} "
        f"executed={sorted(executed_paths)}"
    )


def test_central_corpus_chain() -> None:
    """Chain assurance path: trace-pipeline's verify_aep_chain executes the
    manifest's `chain` verdicts (intact / partial / orphaned / broken /
    not-present) using the same link-hash projection as the JS/Rust
    verifiers. Every declared chain target executes exactly once."""
    from evomerge.validate.aep import verify_aep_chain

    manifest = _manifest()
    entries = manifest.get("conformance_target", [])

    targets = [
        e for e in entries
        if e.get("chain") and e.get("chain") != "not-checked"
    ]
    assert targets, "manifest declares no chain targets"
    executed_paths: set[str] = set()

    for entry in targets:
        expected = entry["chain"]
        fixture = Path(CORPUS) / entry["path"]
        records = _records(fixture)
        assert records, f"{entry['path']}: chain fixture contains no records"
        result = verify_aep_chain(records)
        assert result.status == expected, (
            f"{entry['path']}: chain expected {expected}, got {result.status}"
        )
        executed_paths.add(entry["path"])

    assert executed_paths == {e["path"] for e in targets}, (
        f"chain targets not fully executed: "
        f"declared={sorted({e['path'] for e in targets})} "
        f"executed={sorted(executed_paths)}"
    )
