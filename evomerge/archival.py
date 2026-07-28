"""Trace archival — retention policies and tiered cold-storage migration.

This module is the **retention / tiering** half of the Milestone-5 trace
archival and retrieval feature (issue #53). It builds on
:mod:`evomerge.storage` and provides two data-side primitives:

  1. ``RetentionPolicy`` — declares how long traces stay in the *hot* tier
     before ageing out to *cold*, and the absolute age cap after which they
     are deleted. ``plan()`` inspects a ``TraceStorage`` and returns a
     ``RetentionReport`` of planned migrate/delete actions; ``apply()``
     executes them.
  2. ``ArchivalStore``  — pairs a hot ``TraceStorage`` with a cold one (a
     different backend, e.g. S3 Glacier-class, and/or a cheaper codec, e.g.
     Parquet with zstd) and moves aged partitions hot → cold, deleting from
     hot once safely persisted.

Both primitives are backend-agnostic: the actual object-storage lifecycle
(S3 lifecycle rules, GCS object archival) can additionally enforce retention
at the bucket level; this module provides the explicit, testable in-process
equivalent so retention is reproducible and auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from evomerge.storage import _GRANULARITY, TraceStorage

#: Regex extracting the ``dt=`` bucket from a Hive-style partition directory.
_DT_RE = re.compile(r"dt=([^/]+)/?")


@dataclass(frozen=True)
class RetentionPolicy:
    """Declarative retention schedule for trace partitions.

    Attributes:
        hot_days: partitions older than this many days are migrated hot → cold.
        delete_after_days: partitions older than this are deleted outright
            (must be ``> hot_days`` when set; ``None`` = retain forever in cold).
    """

    hot_days: int = 30
    delete_after_days: int | None = None

    def __post_init__(self) -> None:
        if self.hot_days < 0:
            raise ValueError("hot_days must be non-negative")
        if self.delete_after_days is not None:
            if self.delete_after_days <= self.hot_days:
                raise ValueError("delete_after_days must be greater than hot_days")
            if self.delete_after_days < 0:
                raise ValueError("delete_after_days must be non-negative")

    def action_for(self, age_days: float) -> tuple[str, str]:
        """Return ``(action, reason)`` for a partition of the given age.

        ``action`` is ``"delete"``, ``"migrate"``, or ``"keep"``.
        """
        if self.delete_after_days is not None and age_days >= self.delete_after_days:
            return "delete", f"age {age_days:.1f}d >= delete_after_days {self.delete_after_days}d"
        if age_days >= self.hot_days:
            return "migrate", f"age {age_days:.1f}d >= hot_days {self.hot_days}d"
        return "keep", f"age {age_days:.1f}d < hot_days {self.hot_days}d"


@dataclass
class RetentionAction:
    """A single planned retention action on one partition."""

    partition: str
    action: str        # "migrate" | "delete"
    reason: str
    age_days: float
    tier: str = "hot"  # which tier the partition currently lives in


@dataclass
class RetentionReport:
    """Outcome of a retention pass — planned or executed actions."""

    actions: list[RetentionAction] = field(default_factory=list)

    @property
    def migrated(self) -> list[RetentionAction]:
        return [a for a in self.actions if a.action == "migrate"]

    @property
    def deleted(self) -> list[RetentionAction]:
        return [a for a in self.actions if a.action == "delete"]

    @property
    def kept(self) -> list[RetentionAction]:
        return [a for a in self.actions if a.action == "keep"]

    def __len__(self) -> int:
        return len(self.actions)


def partition_age_days(partition_dir: str, granularity: str, *, now: datetime | None = None) -> float:
    """Age (in days) of a partition derived from its ``dt=`` bucket.

    For ``day`` granularity the bucket is ``YYYY-MM-DD``; for ``month`` it is
    ``YYYY-MM`` (aged from the first of the month); for ``hour`` it is
    ``YYYY-MM-DDTHH``. Returns ``+inf`` if no ``dt=`` segment is present.
    """
    now = now if now is not None else datetime.now(tz=timezone.utc)
    match = _DT_RE.search(partition_dir)
    if not match:
        return float("inf")
    bucket = match.group(1)
    fmt = _GRANULARITY[granularity]
    try:
        part_dt = datetime.strptime(bucket, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
        return float("inf")
    return (now - part_dt).total_seconds() / 86400.0


class ArchivalStore:
    """Tiered trace store — migrates aged partitions hot → cold, then deletes.

    The *hot* store is the low-latency retrieval tier (default JSONL on local
    disk or S3 standard); the *cold* store is the archival tier (Parquet with
    a stronger codec, a Glacier-class bucket, or simply a different prefix).
    Records are re-encoded on migration so the two tiers may use different
    codecs — a columnar cold tier is cheaper to scan than row-based hot.
    """

    def __init__(self, hot: TraceStorage, cold: TraceStorage) -> None:
        if hot.granularity != cold.granularity:
            raise ValueError(
                "hot and cold stores must share time granularity "
                f"({hot.granularity!r} vs {cold.granularity!r})"
            )
        self.hot = hot
        self.cold = cold

    # -- planning ------------------------------------------------------------

    def plan(self, policy: RetentionPolicy, *, now: datetime | None = None) -> RetentionReport:
        """Plan retention actions for the hot tier (no side effects)."""
        now = now if now is not None else datetime.now(tz=timezone.utc)
        actions: list[RetentionAction] = []
        for pdir in self.hot.partitions():
            age = partition_age_days(pdir, self.hot.granularity, now=now)
            action, reason = policy.action_for(age)
            actions.append(RetentionAction(partition=pdir, action=action, reason=reason, age_days=age))
        # Also consider deletion of already-archived cold partitions.
        for pdir in self.cold.partitions():
            age = partition_age_days(pdir, self.cold.granularity, now=now)
            action, reason = policy.action_for(age)
            if action == "delete":
                actions.append(
                    RetentionAction(partition=pdir, action="delete", reason=reason, age_days=age, tier="cold")
                )
        return RetentionReport(actions=actions)

    # -- execution -----------------------------------------------------------

    def migrate_partition(self, partition_dir: str) -> int:
        """Move one hot partition to cold; return the record count migrated.

        Records are decoded from the hot codec and re-encoded into the cold
        codec, then removed from hot. Idempotent: re-migrating an emptied
        partition moves zero records.
        """
        records: list[dict] = []
        for key in self.hot.backend.list_keys(partition_dir):
            records.extend(self.hot.codec.loads(self.hot.backend.read_bytes(key)))
        if records:
            self.cold.write(records)
        for key in self.hot.backend.list_keys(partition_dir):
            self.hot.backend.delete(key)
        return len(records)

    def delete_partition(self, partition_dir: str, *, tier: str = "hot") -> int:
        """Delete every object in a partition from the given tier."""
        store = self.cold if tier == "cold" else self.hot
        keys = store.backend.list_keys(partition_dir)
        for key in keys:
            store.backend.delete(key)
        return len(keys)

    def apply(
        self,
        policy: RetentionPolicy,
        *,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> RetentionReport:
        """Execute the retention pass: migrate aged hot partitions, delete expired.

        With ``dry_run=True`` this is identical to :meth:`plan` (no side effects).
        Returns a report of the actions taken (or planned).
        """
        report = self.plan(policy, now=now)
        if dry_run:
            return report
        for action in report.actions:
            if action.action == "migrate":
                self.migrate_partition(action.partition)
            elif action.action == "delete":
                self.delete_partition(action.partition, tier=action.tier)
        return report


__all__ = [
    "ArchivalStore",
    "RetentionAction",
    "RetentionPolicy",
    "RetentionReport",
    "partition_age_days",
]
