"""Tests for Agent Evidence Registry."""
import json
import tempfile
from pathlib import Path

from evomerge.registry import Registry, RegistryEntry, ENTRY_TYPES


def test_register_and_lookup():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(Path(tmp))
        entry = RegistryEntry(
            id="policy-default-v1",
            entry_type="policy_bundle",
            version="1.0.0",
            metadata={"description": "Default policy"},
        )
        reg.register(entry)
        found = reg.lookup("policy-default-v1")
        assert found is not None
        assert found.entry_type == "policy_bundle"


def test_save_and_reload():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(Path(tmp))
        reg.register(RegistryEntry(id="v1", entry_type="verifier", version="0.1.0"))
        reg.save()
        reg2 = Registry(Path(tmp))
        assert reg2.lookup("v1") is not None
        assert reg2.lookup("v1").entry_type == "verifier"


def test_list_by_type():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(Path(tmp))
        reg.register(RegistryEntry(id="p1", entry_type="policy_bundle", version="1.0"))
        reg.register(RegistryEntry(id="p2", entry_type="policy_bundle", version="1.1"))
        reg.register(RegistryEntry(id="v1", entry_type="verifier", version="0.1"))
        assert len(reg.list_by_type("policy_bundle")) == 2
        assert len(reg.list_by_type("verifier")) == 1


def test_invalid_entry_type():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(Path(tmp))
        import pytest
        with pytest.raises(ValueError, match="Unknown entry_type"):
            reg.register(RegistryEntry(id="x", entry_type="unknown_type", version="1.0"))


def test_digest_computed_from_file():
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "policy.json"
        artifact.write_text('{"allow": true}')
        entry = RegistryEntry(
            id="policy-with-file",
            entry_type="policy_bundle",
            version="1.0",
            artifact_path=str(artifact),
        )
        assert len(entry.digest) == 64
        assert entry.digest != ""


def test_verify_entries_no_artifact():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(Path(tmp))
        reg.register(RegistryEntry(id="e1", entry_type="verifier", version="1.0"))
        results = reg.verify_entries()
        assert results[0][1] is True


def test_index_digest_changes_after_save():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(Path(tmp))
        reg.register(RegistryEntry(id="e1", entry_type="aep_schema", version="0.1"))
        reg.save()
        d1 = reg.index_digest()
        reg.register(RegistryEntry(id="e2", entry_type="aep_schema", version="0.2"))
        reg.save()
        d2 = reg.index_digest()
        assert d1 != d2
