"""State snapshot/restore for the reproducible trace replay framework.

A replay never touches a real external system: state-changing actions are
served from a recorded cassette (see ``engine.py``) and the *agent-visible
state* they mutate is held in a :class:`StateStore`. The store is a small
versioned key-value map that can be checkpointed and rolled back, which is
what enables "debug mode" — when a trust check fails, you can restore the
snapshot taken just before the offending action and inspect exactly the state
the agent saw at that step.

The store's content digest (sha256 of canonical JSON, matching the convention
used by :mod:`evomerge.provenance` and :mod:`evomerge.registry`) is what the
replay engine compares against an action's recorded ``pre_state_digest`` /
``post_state_digest`` to verify the recorded trace forms a reproducible causal
chain.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def canonical_digest(value: Any) -> str:
    """sha256 hex of a value's canonical JSON (sorted keys, compact separators).

    Used for state snapshots and replayed result digests so that replay
    determinism is byte-for-byte comparable across runs.
    """
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class StateSnapshot:
    """An immutable checkpoint of a :class:`StateStore`.

    ``seq`` is the monotonic sequence number of the checkpoint (0 for the
    initial empty store, incremented after each mutation batch). ``digest`` is
    the content digest of the frozen ``store``; restoring a snapshot restores
    both the contents and the sequence counter.
    """

    seq: int
    digest: str
    store: dict[str, Any] = field(default_factory=dict)


class StateStore:
    """Versioned key-value store with snapshot/restore.

    Used as the simulated agent-visible state during replay. Because the store
    is the only thing a replayed action can mutate, snapshotting it before each
    action lets ``replay_debug`` rewind to any step without re-running real
    side effects.
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._store: dict[str, Any] = dict(initial) if initial else {}
        self._seq: int = 0
        self._snapshots: list[StateSnapshot] = []

    # -- core KV operations -------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Write a key and bump the sequence counter."""
        self._store[key] = value
        self._seq += 1

    def update(self, mapping: dict[str, Any]) -> None:
        """Apply a batch of writes as a single sequence bump."""
        if not mapping:
            return
        self._store.update(mapping)
        self._seq += 1

    def as_dict(self) -> dict[str, Any]:
        return dict(self._store)

    @property
    def seq(self) -> int:
        return self._seq

    def digest(self) -> str:
        """Content digest of the current state."""
        return canonical_digest(self._store)

    # -- snapshot / restore -------------------------------------------------

    def snapshot(self) -> StateSnapshot:
        """Checkpoint the current state. The snapshot is immutable and stored
        on the history stack so it survives later mutations."""
        snap = StateSnapshot(seq=self._seq, digest=self.digest(), store=dict(self._store))
        self._snapshots.append(snap)
        return snap

    def restore(self, snapshot: StateSnapshot) -> None:
        """Roll the store back to a previously taken snapshot.

        This is the core of debug-mode replay: restore the snapshot taken
        immediately before a failing action to inspect the state at that step.
        Restoring does not clear the history — snapshots remain available for
        further rollbacks.
        """
        self._store = dict(snapshot.store)
        self._seq = snapshot.seq

    @property
    def snapshots(self) -> list[StateSnapshot]:
        """Immutable view of the checkpoint history (oldest first)."""
        return list(self._snapshots)

    def __len__(self) -> int:
        return len(self._store)
