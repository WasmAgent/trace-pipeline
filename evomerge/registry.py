"""Agent Evidence Registry — tamper-evident signed-JSON artifact registry.

Stores manifests for policy bundles, verifiers, benchmark tasks, receipts,
and model/router profiles. Each entry is a JSON file with a SHA-256 digest
of the referenced artifact. The registry index (registry.json) lists all
entries.

Tamper-evidence guarantees (hardened):
- Every entry that declares an artifact_path MUST also have a non-empty digest.
  A missing digest is treated as a validation failure, never silently skipped.
- index.json is signed with Ed25519; the signature is stored in index.sig.
  Loading a registry verifies the signature before trusting the index.
- An append-only event log (events.jsonl) records every register/remove/rollback
  operation. On startup the log is replayed and compared against the current
  index; a mismatch fails hard.

Usage:
    from evomerge.registry import Registry, RegistryEntry

    reg = Registry(Path("registry/"), index_sig_key=pubkey_pem_or_none)
    reg.register(RegistryEntry(
        id="policy-default-v1",
        entry_type="policy_bundle",
        version="1.0.0",
        artifact_path="policies/default.json",
        digest="<sha256>",
        metadata={"description": "Default WasmAgent policy bundle"},
    ))
    reg.save(signing_key=private_key)
    entry = reg.lookup("policy-default-v1")
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


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

# Event types recorded in events.jsonl
_EVT_REGISTER = "register"
_EVT_REMOVE = "remove"
_EVT_ROLLBACK = "rollback"


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


class RegistryIntegrityError(Exception):
    """Raised when registry tamper-evidence checks fail."""


class Registry:
    """Tamper-evident artifact registry backed by a directory of JSON files.

    Hardening:
    - Entries with artifact_path but no digest are rejected at load time.
    - index.json must be accompanied by a valid index.sig (when a verify_key
      is provided). An index without a matching signature fails immediately.
    - An events.jsonl append-only log is maintained. At startup the log is
      replayed and the resulting state is compared to the loaded index;
      any discrepancy raises RegistryIntegrityError.
    """

    INDEX_FILE = "registry.json"
    SIG_FILE = "index.sig"
    EVENTS_FILE = "events.jsonl"

    def __init__(
        self,
        root: Path,
        verify_key=None,   # Ed25519PublicKey or None (skip sig check)
    ) -> None:
        self._root = root
        self._entries: dict[str, RegistryEntry] = {}
        self._verify_key = verify_key
        if (root / self.INDEX_FILE).exists():
            self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self._root / self.INDEX_FILE

    def _sig_path(self) -> Path:
        return self._root / self.SIG_FILE

    def _events_path(self) -> Path:
        return self._root / self.EVENTS_FILE

    def _load(self) -> None:
        index_path = self._index_path()
        index_bytes = index_path.read_bytes()

        # 1. Verify index.sig when a verify_key is provided
        if self._verify_key is not None:
            sig_path = self._sig_path()
            if not sig_path.exists():
                raise RegistryIntegrityError(
                    f"index.sig not found at {sig_path}; cannot verify index integrity"
                )
            sig_b64 = sig_path.read_text().strip()
            try:
                padding = (4 - len(sig_b64) % 4) % 4
                sig_bytes = base64.urlsafe_b64decode(sig_b64 + "=" * padding)
            except Exception as exc:
                raise RegistryIntegrityError(
                    f"Cannot decode index.sig: {exc}"
                ) from exc
            try:
                from cryptography.exceptions import InvalidSignature
                self._verify_key.verify(sig_bytes, index_bytes)
            except InvalidSignature:
                raise RegistryIntegrityError(
                    "index.json signature verification failed — file may have been tampered with"
                )

        # 2. Parse and validate entries
        data = json.loads(index_bytes)
        loaded_entries: dict[str, RegistryEntry] = {}
        for item in data.get("entries", []):
            e = RegistryEntry.from_dict(item)
            # Hard rule: artifact_path without digest is not allowed
            if e.artifact_path and not e.digest:
                raise RegistryIntegrityError(
                    f"Registry entry {e.id!r} has artifact_path but no digest — "
                    "this entry cannot be trusted"
                )
            loaded_entries[e.id] = e

        # 3. Replay events.jsonl and compare to loaded index
        events_path = self._events_path()
        if events_path.exists():
            replayed = self._replay_events(events_path)
            if set(replayed.keys()) != set(loaded_entries.keys()):
                missing = set(replayed.keys()) - set(loaded_entries.keys())
                extra = set(loaded_entries.keys()) - set(replayed.keys())
                raise RegistryIntegrityError(
                    f"Event log replay mismatch — "
                    f"missing from index: {sorted(missing)}, "
                    f"extra in index not in log: {sorted(extra)}"
                )

        self._entries = loaded_entries

    def _replay_events(self, events_path: Path) -> dict[str, RegistryEntry]:
        """Replay events.jsonl and return the expected final state."""
        state: dict[str, RegistryEntry] = {}
        with open(events_path) as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RegistryIntegrityError(
                        f"events.jsonl line {lineno}: JSON parse error — {exc}"
                    ) from exc
                evt_type = evt.get("event")
                entry_data = evt.get("entry")
                if evt_type == _EVT_REGISTER and entry_data:
                    e = RegistryEntry.from_dict(entry_data)
                    state[e.id] = e
                elif evt_type == _EVT_REMOVE and entry_data:
                    entry_id = entry_data.get("id") if isinstance(entry_data, dict) else entry_data
                    state.pop(entry_id, None)
                elif evt_type == _EVT_ROLLBACK and entry_data:
                    # Rollback restores a previous version
                    e = RegistryEntry.from_dict(entry_data)
                    state[e.id] = e
                else:
                    raise RegistryIntegrityError(
                        f"events.jsonl line {lineno}: unknown event type {evt_type!r}"
                    )
        return state

    def _append_event(self, event_type: str, entry: RegistryEntry) -> None:
        """Append one event record to events.jsonl."""
        self._root.mkdir(parents=True, exist_ok=True)
        evt = {
            "event": event_type,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entry": entry.to_dict(),
        }
        with open(self._events_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(evt, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, entry: RegistryEntry) -> None:
        if entry.entry_type not in ENTRY_TYPES:
            raise ValueError(
                f"Unknown entry_type: {entry.entry_type!r}. Valid: {sorted(ENTRY_TYPES)}"
            )
        # Hard rule: if artifact_path is given, digest must also be provided
        if entry.artifact_path and not entry.digest:
            raise RegistryIntegrityError(
                f"Entry {entry.id!r}: artifact_path is set but digest is empty. "
                "Compute the SHA-256 digest of the artifact before registering."
            )
        self._entries[entry.id] = entry
        self._append_event(_EVT_REGISTER, entry)

    def remove(self, entry_id: str) -> None:
        """Remove an entry from the registry and append a remove event."""
        entry = self._entries.pop(entry_id, None)
        if entry is not None:
            self._append_event(_EVT_REMOVE, entry)

    def rollback(self, entry: RegistryEntry) -> None:
        """Restore a previous version of an entry (e.g. on failed upgrade)."""
        self._entries[entry.id] = entry
        self._append_event(_EVT_ROLLBACK, entry)

    def lookup(self, entry_id: str) -> Optional[RegistryEntry]:
        return self._entries.get(entry_id)

    def list_by_type(self, entry_type: str) -> list[RegistryEntry]:
        return [e for e in self._entries.values() if e.entry_type == entry_type]

    def all(self) -> list[RegistryEntry]:
        return list(self._entries.values())

    def save(self, signing_key=None) -> None:
        """Write index.json (and optionally sign it into index.sig).

        Args:
            signing_key: Ed25519PrivateKey or None. When provided the
                canonical JSON bytes of the index are signed and the
                base64url signature is written to index.sig.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        index = {
            "registry_version": "evidence-registry/v0.1",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entries": [e.to_dict() for e in self._entries.values()],
        }
        index_text = json.dumps(index, indent=2, ensure_ascii=False)
        index_bytes = index_text.encode()
        self._index_path().write_bytes(index_bytes)

        if signing_key is not None:
            sig_bytes = signing_key.sign(index_bytes)
            sig_b64 = base64.urlsafe_b64encode(sig_bytes).decode()
            self._sig_path().write_text(sig_b64 + "\n")

    def index_digest(self) -> str:
        """SHA-256 of the current index JSON (for tamper-evidence)."""
        index_path = self._index_path()
        if not index_path.exists():
            return ""
        return _sha256_file(index_path)

    def verify_entries(self) -> list[tuple[str, bool, str]]:
        """Verify digest of each entry that has an artifact_path.

        Returns list of (entry_id, ok, message).

        Hard rule: an entry with artifact_path but no digest now FAILS
        (unlike the previous behaviour of being silently skipped).
        """
        results = []
        for entry in self._entries.values():
            if not entry.artifact_path:
                # No artifact: nothing to verify
                results.append((entry.id, True, "no artifact path — skipped"))
                continue
            if not entry.digest:
                # artifact_path without digest is a hard failure
                results.append((entry.id, False, "artifact_path present but digest is empty — cannot verify"))
                continue
            actual = _sha256_file(Path(entry.artifact_path))
            if not actual:
                results.append((entry.id, False, f"artifact not found: {entry.artifact_path}"))
            elif actual == entry.digest:
                results.append((entry.id, True, "digest verified"))
            else:
                results.append((entry.id, False, f"digest mismatch: expected {entry.digest[:16]}... got {actual[:16]}..."))
        return results
