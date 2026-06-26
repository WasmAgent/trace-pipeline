"""Tamper-evidence tests for the Registry (P0-8).

Tests:
(b) Entry with artifact_path but no digest → fail
(c) index.json modified but index.sig not updated → fail
(d) events.jsonl missing a delete record → fail
(e) Happy path: register/remove/rollback round-trips with signing
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evomerge.registry import Registry, RegistryEntry, RegistryIntegrityError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_keypair():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def _make_artifact(tmp: str, content: str = '{"ok": true}') -> Path:
    p = Path(tmp) / "artifact.json"
    p.write_text(content)
    return p


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# (b) Entry missing digest → fail
# ---------------------------------------------------------------------------

class TestMissingDigest:
    def test_register_artifact_path_without_digest_fails(self):
        """Registering an entry with artifact_path but empty digest must raise."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = Registry(Path(tmp))
            entry = RegistryEntry(
                id="bad-entry",
                entry_type="policy_bundle",
                version="1.0.0",
                artifact_path="/some/nonexistent/path.json",
                digest="",  # explicitly empty
            )
            # RegistryEntry.__post_init__ tries to compute digest from the file;
            # since the file doesn't exist it stays empty.  register() must reject it.
            with pytest.raises(RegistryIntegrityError, match="digest is empty"):
                reg.register(entry)

    def test_load_index_with_missing_digest_fails(self):
        """An index.json entry that has artifact_path but no digest must fail on load."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Write an index.json with a bad entry directly (bypassing Registry.register)
            bad_index = {
                "registry_version": "evidence-registry/v0.1",
                "updated_at": "2024-01-01T00:00:00Z",
                "entries": [
                    {
                        "id": "bad",
                        "entry_type": "verifier",
                        "version": "1.0",
                        "digest": "",               # empty — bad
                        "artifact_path": "/tmp/something.json",
                        "metadata": {},
                        "registered_at": "2024-01-01T00:00:00Z",
                    }
                ],
            }
            (tmp_path / "registry.json").write_text(json.dumps(bad_index))
            with pytest.raises(RegistryIntegrityError, match="no digest"):
                Registry(tmp_path)


# ---------------------------------------------------------------------------
# (c) index.json tampered, index.sig not updated → fail
# ---------------------------------------------------------------------------

class TestIndexSigVerification:
    def test_tampered_index_fails_sig_check(self):
        """Modifying index.json after signing must cause signature verification failure."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            private_key, public_key = _make_keypair()

            # Create a valid, signed registry
            reg = Registry(tmp_path, verify_key=None)  # no verify during setup
            artifact = _make_artifact(tmp)
            digest = _sha256_file(artifact)
            reg.register(RegistryEntry(
                id="signed-entry",
                entry_type="verifier",
                version="1.0",
                artifact_path=str(artifact),
                digest=digest,
            ))
            reg.save(signing_key=private_key)

            # Now tamper with index.json (don't update index.sig)
            index_path = tmp_path / "registry.json"
            data = json.loads(index_path.read_text())
            data["entries"][0]["version"] = "TAMPERED"
            index_path.write_text(json.dumps(data))

            # Loading with verify_key must fail
            with pytest.raises(RegistryIntegrityError, match="signature verification failed"):
                Registry(tmp_path, verify_key=public_key)

    def test_missing_sig_file_fails(self):
        """If verify_key is set but index.sig is absent, loading must fail."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            private_key, public_key = _make_keypair()

            reg = Registry(tmp_path, verify_key=None)
            reg.register(RegistryEntry(id="e1", entry_type="verifier", version="1.0"))
            reg.save(signing_key=None)  # save without signing

            # Now try to load with a verify_key — index.sig is absent
            with pytest.raises(RegistryIntegrityError, match="index.sig not found"):
                Registry(tmp_path, verify_key=public_key)

    def test_valid_signature_passes(self):
        """A correctly signed index.json must load successfully."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            private_key, public_key = _make_keypair()

            reg = Registry(tmp_path, verify_key=None)
            reg.register(RegistryEntry(id="e1", entry_type="verifier", version="1.0"))
            reg.save(signing_key=private_key)

            reg2 = Registry(tmp_path, verify_key=public_key)
            assert reg2.lookup("e1") is not None


# ---------------------------------------------------------------------------
# (d) events.jsonl missing a remove event → fail
# ---------------------------------------------------------------------------

class TestEventsJournalReplay:
    def test_missing_remove_event_fails(self):
        """If an entry was removed but events.jsonl lacks the remove event, load must fail."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Step 1: register two entries, save
            reg = Registry(tmp_path)
            reg.register(RegistryEntry(id="e1", entry_type="verifier", version="1.0"))
            reg.register(RegistryEntry(id="e2", entry_type="verifier", version="1.0"))
            reg.save()

            # Step 2: remove e2 via the API (appends event)
            reg.remove("e2")
            reg.save()

            # Step 3: manually snip the last line of events.jsonl (the remove event)
            events_path = tmp_path / "events.jsonl"
            lines = events_path.read_text().splitlines()
            # Remove the last event (the remove for e2)
            events_path.write_text("\n".join(lines[:-1]) + "\n")

            # Now load: replay says e2 should still exist, but index says it's gone → mismatch
            with pytest.raises(RegistryIntegrityError, match="mismatch"):
                Registry(tmp_path)

    def test_events_replay_consistent_passes(self):
        """A consistent events.jsonl + index.json must load without error."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reg = Registry(tmp_path)
            reg.register(RegistryEntry(id="e1", entry_type="verifier", version="1.0"))
            reg.register(RegistryEntry(id="e2", entry_type="dataset_card", version="2.0"))
            reg.save()

            reg2 = Registry(tmp_path)
            assert reg2.lookup("e1") is not None
            assert reg2.lookup("e2") is not None

    def test_remove_and_reload_consistent(self):
        """register→save→remove→save round-trip must remain consistent."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reg = Registry(tmp_path)
            reg.register(RegistryEntry(id="keep", entry_type="verifier", version="1.0"))
            reg.register(RegistryEntry(id="gone", entry_type="verifier", version="1.0"))
            reg.save()
            reg.remove("gone")
            reg.save()

            reg2 = Registry(tmp_path)
            assert reg2.lookup("keep") is not None
            assert reg2.lookup("gone") is None

    def test_rollback_event_recorded_and_replayed(self):
        """rollback() must append an event and survive a reload."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reg = Registry(tmp_path)
            original = RegistryEntry(id="e1", entry_type="verifier", version="1.0")
            reg.register(original)
            reg.save()

            # Upgrade then rollback
            upgraded = RegistryEntry(id="e1", entry_type="verifier", version="2.0")
            reg.register(upgraded)
            reg.rollback(original)
            reg.save()

            reg2 = Registry(tmp_path)
            entry = reg2.lookup("e1")
            assert entry is not None
            assert entry.version == "1.0"

    def test_extra_entry_in_index_not_in_log_fails(self):
        """An entry that appears in index.json but has no corresponding event must fail."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reg = Registry(tmp_path)
            reg.register(RegistryEntry(id="e1", entry_type="verifier", version="1.0"))
            reg.save()

            # Manually inject a phantom entry into index.json without adding an event
            index_path = tmp_path / "registry.json"
            data = json.loads(index_path.read_text())
            data["entries"].append({
                "id": "phantom",
                "entry_type": "verifier",
                "version": "1.0",
                "digest": "",
                "artifact_path": "",
                "metadata": {},
                "registered_at": "2024-01-01T00:00:00Z",
            })
            index_path.write_text(json.dumps(data))

            with pytest.raises(RegistryIntegrityError, match="mismatch"):
                Registry(tmp_path)


# ---------------------------------------------------------------------------
# Backward compatibility: entries without artifact_path still work
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_entry_without_artifact_path_allowed(self):
        """Entries with no artifact_path and no digest are still valid (metadata-only)."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = Registry(Path(tmp))
            reg.register(RegistryEntry(
                id="meta-only",
                entry_type="model_profile",
                version="1.0",
                metadata={"note": "no artifact"},
            ))
            reg.save()
            reg2 = Registry(Path(tmp))
            assert reg2.lookup("meta-only") is not None

    def test_verify_entries_no_artifact_path_skipped(self):
        """verify_entries skips entries without artifact_path (still returns True)."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = Registry(Path(tmp))
            reg.register(RegistryEntry(id="e1", entry_type="verifier", version="1.0"))
            results = reg.verify_entries()
            assert results[0][1] is True
            assert "skipped" in results[0][2]
