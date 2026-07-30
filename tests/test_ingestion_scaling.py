"""Tests for evomerge.ingestion_scaling — issue #51.

Covers:
  - ConsistentHashRing: distribution, stability, determinism
  - ShardRouter: routing-key precedence, shard assignment
  - BackpressureQueue: put/get, capacity enforcement, drain
  - ValidationWorkerPool: concurrent validation of valid/invalid records
  - IngestionPipeline: end-to-end ingest with shard distribution
"""
from __future__ import annotations

import pytest

from evomerge.ingestion_scaling import (
    BackpressureQueue,
    ConsistentHashRing,
    IngestionPipeline,
    QueueFullError,
    ShardRouter,
    ShardedBatch,
    ValidationWorkerPool,
)

# ---------------------------------------------------------------------------
# Minimal AEP-like records for testing
# ---------------------------------------------------------------------------

def _aep(run_id="r1", user_id="u1", subject_id="s1"):
    return {
        "schema_version": "aep/v0.1",
        "run_id": run_id,
        "created_at_ms": 0,
        "user_id": user_id,
        "subject_id": subject_id,
        "actions": [],
        "capability_decisions": [],
        "verifier_results": [],
    }


def _invalid_record(run_id="bad"):
    # Missing required schema fields → will fail schema validation
    return {"run_id": run_id, "junk": True}


# ---------------------------------------------------------------------------
# ConsistentHashRing
# ---------------------------------------------------------------------------

class TestConsistentHashRing:
    def test_shard_range(self):
        ring = ConsistentHashRing(n_shards=8)
        for i in range(100):
            shard = ring.shard_for(f"key-{i}")
            assert 0 <= shard < 8

    def test_deterministic(self):
        ring = ConsistentHashRing(n_shards=16)
        assert ring.shard_for("user-42") == ring.shard_for("user-42")

    def test_distribution(self):
        ring = ConsistentHashRing(n_shards=4, virtual_nodes=200)
        counts = [0] * 4
        for i in range(400):
            counts[ring.shard_for(f"user-{i}")] += 1
        # Each shard should get at least 5% of keys
        assert all(c >= 20 for c in counts), f"uneven: {counts}"

    def test_invalid_n_shards(self):
        with pytest.raises(ValueError):
            ConsistentHashRing(n_shards=0)

    def test_single_shard(self):
        ring = ConsistentHashRing(n_shards=1)
        assert ring.shard_for("anything") == 0


# ---------------------------------------------------------------------------
# ShardRouter
# ---------------------------------------------------------------------------

class TestShardRouter:
    def test_routes_by_user_id(self):
        ring = ConsistentHashRing(n_shards=4)
        router = ShardRouter(ring)
        rec = {"user_id": "alice", "subject_id": "bob", "run_id": "r1"}
        batches = router.route([rec])
        assert len(batches) == 1
        expected_shard = ring.shard_for("alice")
        assert batches[0].shard == expected_shard

    def test_falls_back_to_subject_id(self):
        ring = ConsistentHashRing(n_shards=4)
        router = ShardRouter(ring)
        rec = {"subject_id": "carol", "run_id": "r2"}
        batches = router.route([rec])
        assert batches[0].shard == ring.shard_for("carol")

    def test_falls_back_to_run_id(self):
        ring = ConsistentHashRing(n_shards=4)
        router = ShardRouter(ring)
        rec = {"run_id": "run-xyz"}
        batches = router.route([rec])
        assert batches[0].shard == ring.shard_for("run-xyz")

    def test_falls_back_to_unknown(self):
        router = ShardRouter()
        batches = router.route([{"data": 1}])
        ring = ConsistentHashRing()
        assert batches[0].shard == ring.shard_for("_unknown")

    def test_run_context_nesting(self):
        ring = ConsistentHashRing(n_shards=4)
        router = ShardRouter(ring)
        rec = {"run_context": {"user_id": "nested-user"}}
        batches = router.route([rec])
        assert batches[0].shard == ring.shard_for("nested-user")

    def test_multiple_records_grouped_per_shard(self):
        ring = ConsistentHashRing(n_shards=2)
        router = ShardRouter(ring)
        # Force two keys to the same shard by building same-shard keys
        records = [{"user_id": f"u{i}"} for i in range(20)]
        batches = router.route(records)
        total = sum(len(b.records) for b in batches)
        assert total == 20
        assert len(batches) <= 2

    def test_empty_input(self):
        router = ShardRouter()
        assert router.route([]) == []


# ---------------------------------------------------------------------------
# BackpressureQueue
# ---------------------------------------------------------------------------

class TestBackpressureQueue:
    def test_put_and_get(self):
        q = BackpressureQueue(maxsize=10)
        q.put("hello")
        assert q.get() == "hello"

    def test_blocks_when_raise_on_full(self):
        q = BackpressureQueue(maxsize=2, block_on_full=False)
        q.put("a")
        q.put("b")
        with pytest.raises(QueueFullError):
            q.put("c")

    def test_drain(self):
        q = BackpressureQueue(maxsize=5)
        for i in range(4):
            q.put(i)
        items = q.drain()
        assert sorted(items) == [0, 1, 2, 3]
        assert q.empty

    def test_qsize(self):
        q = BackpressureQueue(maxsize=10)
        assert q.qsize == 0
        q.put("x")
        assert q.qsize == 1

    def test_unbounded_queue(self):
        q = BackpressureQueue(maxsize=0)
        for i in range(10_000):
            q.put(i)
        assert q.qsize == 10_000


# ---------------------------------------------------------------------------
# ValidationWorkerPool
# ---------------------------------------------------------------------------

class TestValidationWorkerPool:
    def test_empty_batch(self):
        pool = ValidationWorkerPool()
        assert pool.validate_batch([]) == []

    def test_valid_records(self):
        pool = ValidationWorkerPool()
        records = [_aep(f"r{i}", f"u{i}", f"s{i}") for i in range(5)]
        pairs = pool.validate_batch(records)
        assert len(pairs) == 5
        for rec, result in pairs:
            assert result.valid_schema, f"Expected valid but got errors: {result.errors}"

    def test_invalid_records_caught(self):
        pool = ValidationWorkerPool()
        records = [_invalid_record("bad1"), _invalid_record("bad2")]
        pairs = pool.validate_batch(records)
        assert len(pairs) == 2
        for _, result in pairs:
            assert not result.passed

    def test_mixed_batch(self):
        pool = ValidationWorkerPool()
        records = [_aep("v1"), _invalid_record("i1"), _aep("v2")]
        pairs = pool.validate_batch(records)
        assert len(pairs) == 3
        assert pairs[0][1].valid_schema   # first is valid
        assert not pairs[1][1].passed     # second is invalid
        assert pairs[2][1].valid_schema   # third is valid

    def test_concurrent_stability(self):
        pool = ValidationWorkerPool(max_workers=4)
        records = [_aep(f"r{i}") for i in range(50)]
        pairs = pool.validate_batch(records)
        assert len(pairs) == 50


# ---------------------------------------------------------------------------
# IngestionPipeline
# ---------------------------------------------------------------------------

class TestIngestionPipeline:
    def test_basic_ingest(self):
        pipeline = IngestionPipeline()
        records = [_aep(f"r{i}", f"u{i % 3}", f"s{i % 3}") for i in range(9)]
        result = pipeline.ingest(records)
        assert result.total_records == 9
        assert result.n_shards_used >= 1
        assert result.valid_count + result.invalid_count == result.total_records

    def test_shard_distribution_populated(self):
        pipeline = IngestionPipeline(router=ShardRouter(ConsistentHashRing(n_shards=4)))
        records = [_aep(f"r{i}", f"user-{i}", f"subj-{i}") for i in range(20)]
        result = pipeline.ingest(records)
        assert sum(result.shard_distribution.values()) == 20

    def test_empty_ingest(self):
        pipeline = IngestionPipeline()
        result = pipeline.ingest([])
        assert result.total_records == 0
        assert result.n_shards_used == 0

    def test_queue_full_raises(self):
        pipeline = IngestionPipeline(queue_maxsize=1, block_on_full=False)
        records = [_aep(f"r{i}") for i in range(10)]
        with pytest.raises(QueueFullError):
            pipeline.ingest(records)
