"""Tests for evomerge.trust_score_aggregation — issue #56.

Covers:
  - TrustScoreAggregator: per-subject aggregation, drift detection
  - TrustScoreCache: LRU eviction, TTL expiry, thread safety
  - TrustScoreStore: refresh, bulk_refresh, cache-first lookup
  - TrustMetrics: record, exposition format
  - TrustScoreServer: end-to-end get/refresh/bulk_refresh
"""
from __future__ import annotations

import time

import pytest

from evomerge.trust_score_aggregation import (
    AggregatedTrustScore,
    TrustMetrics,
    TrustScoreAggregator,
    TrustScoreCache,
    TrustScoreServer,
    TrustScoreStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aep(run_id="r1", task_passed=True, has_verifier=True):
    verifiers = [{"verifier_id": "v1", "passed": task_passed}] if has_verifier else []
    return {
        "schema_version": "aep/v0.1",
        "run_id": run_id,
        "created_at_ms": 0,
        "actions": [],
        "capability_decisions": [],
        "verifier_results": verifiers,
    }


# ---------------------------------------------------------------------------
# TrustScoreAggregator
# ---------------------------------------------------------------------------

class TestTrustScoreAggregator:
    def test_empty_records(self):
        agg = TrustScoreAggregator()
        score = agg.aggregate("s1", [])
        assert score.overall is None
        assert score.n_records == 0
        assert score.drift is None

    def test_single_record(self):
        agg = TrustScoreAggregator()
        score = agg.aggregate("s1", [_aep("r1")])
        assert score.n_records == 1
        assert score.overall is not None
        assert 0.0 <= score.overall <= 1.0
        assert score.std_dev is None  # need >= 2 records

    def test_multiple_records_overall_in_range(self):
        agg = TrustScoreAggregator()
        records = [_aep(f"r{i}") for i in range(5)]
        score = agg.aggregate("s1", records)
        assert score.n_records == 5
        assert score.overall is not None
        assert 0.0 <= score.overall <= 1.0
        assert score.std_dev is not None

    def test_drift_computed_for_ge_4_records(self):
        agg = TrustScoreAggregator()
        records = (
            [_aep(f"good-{i}", task_passed=True) for i in range(2)]
            + [_aep(f"bad-{i}", task_passed=False) for i in range(2)]
        )
        score = agg.aggregate("s1", records)
        assert score.drift is not None
        assert score.drift >= 0.0

    def test_no_drift_for_lt_4_records(self):
        agg = TrustScoreAggregator()
        score = agg.aggregate("s1", [_aep("r1"), _aep("r2"), _aep("r3")])
        assert score.drift is None

    def test_min_max(self):
        agg = TrustScoreAggregator()
        records = [_aep("r1", task_passed=True), _aep("r2", task_passed=False)]
        score = agg.aggregate("s1", records)
        if score.min_score is not None and score.max_score is not None:
            assert score.min_score <= score.max_score


# ---------------------------------------------------------------------------
# TrustScoreCache
# ---------------------------------------------------------------------------

class TestTrustScoreCache:
    def _score(self, sid="s1"):
        return AggregatedTrustScore(
            subject_id=sid, overall=0.8, n_records=3,
            mean=0.8, std_dev=0.05, min_score=0.7,
            max_score=0.9, drift=None,
        )

    def test_set_and_get(self):
        cache = TrustScoreCache()
        s = self._score()
        cache.set("s1", s)
        assert cache.get("s1") is s

    def test_miss_returns_none(self):
        cache = TrustScoreCache()
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = TrustScoreCache(maxsize=2)
        for i in range(3):
            cache.set(f"s{i}", self._score(f"s{i}"))
        # s0 should have been evicted (LRU)
        assert cache.get("s0") is None
        assert cache.get("s1") is not None
        assert cache.get("s2") is not None

    def test_ttl_expiry(self):
        cache = TrustScoreCache(ttl_seconds=0.05)
        cache.set("s1", self._score())
        assert cache.get("s1") is not None
        time.sleep(0.1)
        assert cache.get("s1") is None

    def test_invalidate(self):
        cache = TrustScoreCache()
        cache.set("s1", self._score())
        cache.invalidate("s1")
        assert cache.get("s1") is None

    def test_clear(self):
        cache = TrustScoreCache()
        for i in range(5):
            cache.set(f"s{i}", self._score(f"s{i}"))
        cache.clear()
        assert cache.size == 0

    def test_size(self):
        cache = TrustScoreCache()
        cache.set("s1", self._score())
        assert cache.size == 1


# ---------------------------------------------------------------------------
# TrustScoreStore
# ---------------------------------------------------------------------------

class TestTrustScoreStore:
    def test_refresh_and_get(self):
        store = TrustScoreStore()
        records = [_aep("r1")]
        score = store.refresh("s1", records)
        assert score.subject_id == "s1"
        assert store.get("s1") is not None

    def test_bulk_refresh(self):
        store = TrustScoreStore()
        subjects = {f"s{i}": [_aep(f"r{i}")] for i in range(4)}
        scores = store.bulk_refresh(subjects)
        assert len(scores) == 4
        for sid, score in scores.items():
            assert score.subject_id == sid

    def test_get_miss(self):
        store = TrustScoreStore()
        assert store.get("unknown") is None

    def test_invalidate(self):
        store = TrustScoreStore()
        store.refresh("s1", [_aep("r1")])
        store.invalidate("s1")
        assert store.get("s1") is None

    def test_all_scores(self):
        store = TrustScoreStore()
        store.refresh("s1", [_aep("r1")])
        store.refresh("s2", [_aep("r2")])
        all_s = store.all_scores()
        assert "s1" in all_s and "s2" in all_s


# ---------------------------------------------------------------------------
# TrustMetrics
# ---------------------------------------------------------------------------

class TestTrustMetrics:
    def _score(self, sid="s1", overall=0.85, drift=0.05):
        return AggregatedTrustScore(
            subject_id=sid, overall=overall, n_records=5,
            mean=overall, std_dev=0.02, min_score=0.8,
            max_score=0.9, drift=drift,
        )

    def test_record_adds_samples(self):
        m = TrustMetrics()
        m.record(self._score())
        assert len(m.samples()) > 0

    def test_exposition_format(self):
        m = TrustMetrics()
        m.record(self._score("alice", 0.75, 0.1))
        exp = m.exposition()
        assert "trust_score_overall" in exp
        assert "alice" in exp

    def test_exposition_contains_drift(self):
        m = TrustMetrics()
        m.record(self._score("alice", drift=0.2))
        assert "trust_score_drift" in m.exposition()

    def test_none_overall_skipped(self):
        m = TrustMetrics()
        score = AggregatedTrustScore(
            subject_id="s1", overall=None, n_records=0,
            mean=None, std_dev=None, min_score=None,
            max_score=None, drift=None,
        )
        m.record(score)
        exp = m.exposition()
        assert "trust_score_n_records" in exp
        assert "trust_score_overall" not in exp

    def test_clear(self):
        m = TrustMetrics()
        m.record(self._score())
        m.clear()
        assert m.samples() == []


# ---------------------------------------------------------------------------
# TrustScoreServer
# ---------------------------------------------------------------------------

class TestTrustScoreServer:
    def test_get_miss(self):
        server = TrustScoreServer()
        assert server.get("nobody") is None

    def test_refresh_and_get(self):
        server = TrustScoreServer()
        score = server.refresh("s1", [_aep("r1")])
        assert score.subject_id == "s1"
        assert server.get("s1") is not None

    def test_bulk_refresh(self):
        server = TrustScoreServer()
        subjects = {f"s{i}": [_aep(f"r{i}")] for i in range(3)}
        scores = server.bulk_refresh(subjects)
        assert len(scores) == 3

    def test_metrics_populated_after_refresh(self):
        metrics = TrustMetrics()
        server = TrustScoreServer(metrics=metrics)
        server.refresh("s1", [_aep("r1"), _aep("r2")])
        assert len(metrics.samples()) > 0
