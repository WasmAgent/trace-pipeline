"""Tests for evomerge.archival — Milestone 5 retention & tiered cold-storage (issue #53).

Covers the two archival sub-requirements, each in its own test class:

  - TestRetentionPolicy   — retention schedule thresholds + validation
  - TestPartitionAge      — partition age derivation from ``dt=`` buckets
  - TestArchivalStore     — hot → cold migration, deletion, dry-run planning
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evomerge.archival import (
    ArchivalStore,
    RetentionAction,
    RetentionPolicy,
    RetentionReport,
    partition_age_days,
)
from evomerge.storage import JsonLinesCodec, LocalBackend, ParquetCodec, TraceStorage

NOW = datetime(2026, 7, 28, tzinfo=timezone.utc)


def _trace(rollout_id, subject_id="s1", ts="2026-07-01T00:00:00Z", status="pass", score=1):
    return {
        "rollout_id": rollout_id,
        "subject_id": subject_id,
        "timestamp": ts,
        "objective_status": status,
        "objective_score": score,
    }


@pytest.fixture
def hot_cold(tmp_path):
    hot = TraceStorage(LocalBackend(tmp_path / "hot"), granularity="day")
    cold = TraceStorage(LocalBackend(tmp_path / "cold"), granularity="day")
    return ArchivalStore(hot, cold)


# ---------------------------------------------------------------------------
# Retention policy
# ---------------------------------------------------------------------------

class TestRetentionPolicy:
    def test_keep_below_hot_days(self):
        policy = RetentionPolicy(hot_days=30, delete_after_days=365)
        action, reason = policy.action_for(5)
        assert action == "keep"
        assert "5.0d" in reason

    def test_migrate_at_hot_days(self):
        policy = RetentionPolicy(hot_days=30, delete_after_days=365)
        action, _ = policy.action_for(40)
        assert action == "migrate"

    def test_delete_at_delete_after_days(self):
        policy = RetentionPolicy(hot_days=30, delete_after_days=365)
        action, _ = policy.action_for(400)
        assert action == "delete"

    def test_delete_takes_precedence_over_migrate(self):
        # A partition older than delete_after is deleted, not migrated.
        policy = RetentionPolicy(hot_days=30, delete_after_days=60)
        action, _ = policy.action_for(90)
        assert action == "delete"

    def test_retain_forever_when_delete_after_none(self):
        policy = RetentionPolicy(hot_days=30, delete_after_days=None)
        action, _ = policy.action_for(10_000)
        assert action == "migrate"

    def test_reject_negative_hot_days(self):
        with pytest.raises(ValueError, match="hot_days"):
            RetentionPolicy(hot_days=-1)

    def test_reject_delete_after_not_greater_than_hot(self):
        with pytest.raises(ValueError, match="greater than hot_days"):
            RetentionPolicy(hot_days=30, delete_after_days=30)
        with pytest.raises(ValueError, match="greater than hot_days"):
            RetentionPolicy(hot_days=30, delete_after_days=10)

    def test_zero_hot_days_migrates_immediately(self):
        policy = RetentionPolicy(hot_days=0)
        action, _ = policy.action_for(0.0)
        assert action == "migrate"


# ---------------------------------------------------------------------------
# Partition age
# ---------------------------------------------------------------------------

class TestPartitionAge:
    def test_day_granularity_age(self):
        age = partition_age_days("traces/subject_id=s1/dt=2026-07-01/", "day", now=NOW)
        assert age == 27.0  # 2026-07-01 → 2026-07-28

    def test_month_granularity_age_from_first_of_month(self):
        age = partition_age_days("traces/dt=2026-06/", "month", now=NOW)
        # 2026-06-01 → 2026-07-28 = 57 days
        assert age == 57.0

    def test_missing_dt_returns_inf(self):
        assert partition_age_days("traces/subject_id=s1/", "day", now=NOW) == float("inf")

    def test_malformed_dt_returns_inf(self):
        assert partition_age_days("traces/dt=not-a-date/", "day", now=NOW) == float("inf")


# ---------------------------------------------------------------------------
# Archival store
# ---------------------------------------------------------------------------

class TestArchivalStore:
    def test_plan_identifies_aged_partitions(self, hot_cold):
        hot_cold.hot.write([
            _trace("old1", ts="2026-01-01T00:00:00Z"),
            _trace("old2", ts="2026-01-02T00:00:00Z"),
            _trace("new1", ts="2026-07-27T00:00:00Z"),
        ])
        policy = RetentionPolicy(hot_days=30, delete_after_days=365)
        report = hot_cold.plan(policy, now=NOW)
        actions = {a.partition: a.action for a in report.actions}
        assert actions["traces/subject_id=s1/dt=2026-01-01/"] == "migrate"
        assert actions["traces/subject_id=s1/dt=2026-01-02/"] == "migrate"
        assert actions["traces/subject_id=s1/dt=2026-07-27/"] == "keep"

    def test_apply_migrates_aged_to_cold(self, hot_cold):
        hot_cold.hot.write([
            _trace("old1", ts="2026-01-01T00:00:00Z"),
            _trace("new1", ts="2026-07-27T00:00:00Z"),
        ])
        policy = RetentionPolicy(hot_days=30, delete_after_days=365)
        report = hot_cold.apply(policy, now=NOW)
        assert len(report.migrated) == 1
        # Hot retains only the recent partition.
        assert hot_cold.hot.count() == 1
        assert [r["rollout_id"] for r in hot_cold.hot.query()] == ["new1"]
        # Cold holds the migrated record.
        assert hot_cold.cold.count() == 1
        assert [r["rollout_id"] for r in hot_cold.cold.query()] == ["old1"]

    def test_apply_deletes_expired_from_cold(self, hot_cold):
        # 2026-04-01 at NOW (2026-07-28) = 118 days: migrates (>=30) but is not
        # yet expired (<365). At a later date it ages past delete_after → deleted.
        hot_cold.hot.write([_trace("aging", ts="2026-04-01T00:00:00Z")])
        policy = RetentionPolicy(hot_days=30, delete_after_days=365)
        # First pass: migrates to cold (age 118d is between hot and delete thresholds).
        hot_cold.apply(policy, now=NOW)
        assert hot_cold.cold.count() == 1
        # Second pass at a later date: now exceeds delete_after → deleted.
        later = datetime(2027, 7, 28, tzinfo=timezone.utc)
        report = hot_cold.apply(policy, now=later)
        assert len(report.deleted) == 1
        assert hot_cold.cold.count() == 0

    def test_dry_run_has_no_side_effects(self, hot_cold):
        hot_cold.hot.write([_trace("old1", ts="2026-01-01T00:00:00Z")])
        policy = RetentionPolicy(hot_days=30)
        report = hot_cold.apply(policy, now=NOW, dry_run=True)
        assert len(report.migrated) == 1
        # Nothing actually moved.
        assert hot_cold.hot.count() == 1
        assert hot_cold.cold.count() == 0

    def test_migrate_partition_re_encodes_across_codec(self, tmp_path):
        # Hot uses JSONL; cold uses Parquet — migration re-encodes.
        pytest.importorskip("pyarrow")
        hot = TraceStorage(LocalBackend(tmp_path / "hot"), codec=JsonLinesCodec(), granularity="day")
        cold = TraceStorage(LocalBackend(tmp_path / "cold"), codec=ParquetCodec(), granularity="day")
        arch = ArchivalStore(hot, cold)
        hot.write([_trace("old1", ts="2026-01-01T00:00:00Z")])
        moved = arch.migrate_partition("traces/subject_id=s1/dt=2026-01-01/")
        assert moved == 1
        assert hot.count() == 0
        assert cold.count() == 1
        # Cold objects are genuine Parquet.
        cold_keys = [k for p in cold.partitions() for k in cold.backend.list_keys(p)]
        assert all(k.endswith(".parquet") for k in cold_keys)

    def test_migrate_partition_idempotent(self, hot_cold):
        hot_cold.hot.write([_trace("old1", ts="2026-01-01T00:00:00Z")])
        first = hot_cold.migrate_partition("traces/subject_id=s1/dt=2026-01-01/")
        second = hot_cold.migrate_partition("traces/subject_id=s1/dt=2026-01-01/")
        assert first == 1
        assert second == 0  # already migrated, nothing to move

    def test_granularity_mismatch_rejected(self, tmp_path):
        hot = TraceStorage(LocalBackend(tmp_path / "h"), granularity="day")
        cold = TraceStorage(LocalBackend(tmp_path / "c"), granularity="hour")
        with pytest.raises(ValueError, match="granularity"):
            ArchivalStore(hot, cold)

    def test_report_action_types(self, hot_cold):
        # Structural check: RetentionAction / RetentionReport expose the API.
        hot_cold.hot.write([_trace("new1", ts="2026-07-27T00:00:00Z")])
        report = hot_cold.plan(RetentionPolicy(hot_days=30), now=NOW)
        assert isinstance(report, RetentionReport)
        assert all(isinstance(a, RetentionAction) for a in report.actions)
        assert report.kept == [a for a in report.actions if a.action == "keep"]
        assert report.migrated == []
        assert report.deleted == []
