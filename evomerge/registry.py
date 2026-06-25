"""Agent Evidence Registry — lightweight signed-JSON artifact registry.

Stores manifests for policy bundles, verifiers, benchmark tasks, receipts,
and model/router profiles. Each entry is a JSON file with a SHA-256 digest
of the referenced artifact. The registry index (registry.json) lists all entries.

No PKI required — digests provide tamper-evidence; signing keys can be
layered on top via Sigstore or similar if needed.

Usage:
    from evomerge.registry import Registry, RegistryEntry

    reg = Registry(Path("registry/"))
    reg.register(RegistryEntry(
        id="policy-default-v1",
        entry_type="policy_bundle",
        version="1.0.0",
        artifact_path="policies/default.json",
        metadata={"description": "Default WasmAgent policy bundle"},
    ))
    reg.save()
    entry = reg.lookup("policy-default-v1")
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


ENTRY_TYPES = frozenset([
    "policy_bundle",
    "verifier",
    "benchmark_task",
    "receipt",
    "model_profile",
    "router_profile",
    "dataset_card",
    "aep_schema",
])


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class RegistryEntry:
    id: str
    entry_type: str
    version: str
    digest: str = ""
    artifact_path: str = ""
    metadata: dict = field(default_factory=dict)
    registered_at: str = ""

    def __post_init__(self) -> None:
        if not self.registered_at:
            self.registered_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.digest and self.artifact_path:
            p = Path(self.artifact_path)
            if p.is_file():
                self.digest = _sha256_file(p)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RegistryEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Registry:
    """Lightweight artifact registry backed by a directory of JSON files."""

    INDEX_FILE = "registry.json"

    def __init__(self, root: Path) -> None:
        self._root = root
        self._entries: dict[str, RegistryEntry] = {}
        if (root / self.INDEX_FILE).exists():
            self._load()

    def _load(self) -> None:
        data = json.loads((self._root / self.INDEX_FILE).read_text())
        for item in data.get("entries", []):
            e = RegistryEntry.from_dict(item)
            self._entries[e.id] = e

    def register(self, entry: RegistryEntry) -> None:
        if entry.entry_type not in ENTRY_TYPES:
            raise ValueError(f"Unknown entry_type: {entry.entry_type!r}. Valid: {sorted(ENTRY_TYPES)}")
        self._entries[entry.id] = entry

    def lookup(self, entry_id: str) -> RegistryEntry | None:
        return self._entries.get(entry_id)

    def list_by_type(self, entry_type: str) -> list[RegistryEntry]:
        return [e for e in self._entries.values() if e.entry_type == entry_type]

    def all(self) -> list[RegistryEntry]:
        return list(self._entries.values())

    def save(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        index = {
            "registry_version": "evidence-registry/v0.1",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entries": [e.to_dict() for e in self._entries.values()],
        }
        (self._root / self.INDEX_FILE).write_text(
            json.dumps(index, indent=2, ensure_ascii=False)
        )

    def index_digest(self) -> str:
        """SHA-256 of the current index JSON (for tamper-evidence)."""
        index_path = self._root / self.INDEX_FILE
        if not index_path.exists():
            return ""
        return _sha256_file(index_path)

    def verify_entries(self) -> list[tuple[str, bool, str]]:
        """Verify digest of each entry that has an artifact_path.

        Returns list of (entry_id, ok, message).
        """
        results = []
        for entry in self._entries.values():
            if not entry.artifact_path or not entry.digest:
                results.append((entry.id, True, "no artifact path — skipped"))
                continue
            actual = _sha256_file(Path(entry.artifact_path))
            if not actual:
                results.append((entry.id, False, f"artifact not found: {entry.artifact_path}"))
            elif actual == entry.digest:
                results.append((entry.id, True, "digest verified"))
            else:
                results.append((entry.id, False, f"digest mismatch: expected {entry.digest[:16]}... got {actual[:16]}..."))
        return results
