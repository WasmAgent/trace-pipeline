"""Horizontal scaling for trace ingestion — issue #51.

Partitions AEP records by ``user_id``/``subject_id`` shards using consistent
hashing, runs concurrent validation workers, and enforces backpressure-aware
queueing to handle burst traffic from multiple agent instances.

Design
------
- ``ConsistentHashRing`` — a simple virtual-node ring that maps a ``user_id``
  or ``subject_id`` to a shard index (0 … n_shards-1), stable under node
  addition/removal as long as the virtual-node count is kept constant.
- ``ShardRouter`` — wraps the ring and routes an AEP record dict to a shard.
- ``ValidationWorkerPool`` — uses ``concurrent.futures.ThreadPoolExecutor``
  (or ``ProcessPoolExecutor``) to run ``AEPValidator.validate()`` over a batch
  of records in parallel, collecting per-record results.
- ``BackpressureQueue`` — a thin wrapper around ``queue.Queue`` with a
  configurable high-watermark; ``put()`` blocks (or raises ``QueueFullError``)
  when the queue is full, preventing runaway memory growth.
- ``IngestionPipeline`` — top-level entry point: accepts a list of raw record
  dicts, routes them to per-shard queues, runs concurrent validation, and
  returns ``IngestionResult``.

No Redis/RabbitMQ SDK is required at runtime — the backpressure queue is a
pure-Python in-process substitute that is protocol-compatible with the
interface expected by a real message-queue adapter (same ``put``/``get`` API).
A ``RedisQueue`` stub is provided as a drop-in adapter sketch; it raises
``ImportError`` when ``redis`` is not installed, so tests always use the
default in-process queue.
"""
from __future__ import annotations

import hashlib
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from evomerge.validate.aep import AEPValidationResult, validate_aep_record

__all__ = [
    "ConsistentHashRing",
    "ShardRouter",
    "BackpressureQueue",
    "QueueFullError",
    "ValidationWorkerPool",
    "IngestionPipeline",
    "IngestionResult",
    "ShardedBatch",
]


# ---------------------------------------------------------------------------
# Consistent-hash ring
# ---------------------------------------------------------------------------

class ConsistentHashRing:
    """Virtual-node consistent-hash ring mapping keys to shard indices.

    Args:
        n_shards: Number of logical shards.
        virtual_nodes: Replicas per shard on the ring (higher → more balanced).
    """

    def __init__(self, n_shards: int = 16, virtual_nodes: int = 150) -> None:
        if n_shards < 1:
            raise ValueError("n_shards must be >= 1")
        if virtual_nodes < 1:
            raise ValueError("virtual_nodes must be >= 1")
        self.n_shards = n_shards
        self.virtual_nodes = virtual_nodes
        # ring: sorted list of (hash_int, shard_index)
        self._ring: list[tuple[int, int]] = sorted(
            (
                int(hashlib.md5(f"shard-{shard}-vnode-{vnode}".encode()).hexdigest(), 16),
                shard,
            )
            for shard in range(n_shards)
            for vnode in range(virtual_nodes)
        )
        self._ring_hashes = [h for h, _ in self._ring]

    def shard_for(self, key: str) -> int:
        """Return the shard index for ``key``.

        Uses the closest clockwise node on the ring (wrap-around).
        """
        import bisect

        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        idx = bisect.bisect_left(self._ring_hashes, h) % len(self._ring)
        return self._ring[idx][1]


# ---------------------------------------------------------------------------
# Shard router
# ---------------------------------------------------------------------------

@dataclass
class ShardedBatch:
    """Per-shard partition of records."""

    shard: int
    records: list[dict[str, Any]] = field(default_factory=list)


class ShardRouter:
    """Routes AEP record dicts to shard buckets using a ``ConsistentHashRing``.

    Routing key precedence: ``user_id`` → ``subject_id`` → ``run_id`` → ``"_unknown"``.
    Both top-level keys and ``run_context`` nesting (AEP v0.3) are supported.
    """

    def __init__(self, ring: ConsistentHashRing | None = None) -> None:
        self.ring = ring or ConsistentHashRing()

    def _routing_key(self, record: dict[str, Any]) -> str:
        for field_name in ("user_id", "subject_id", "run_id"):
            val = record.get(field_name)
            if val is None:
                ctx = record.get("run_context")
                if isinstance(ctx, dict):
                    val = ctx.get(field_name)
            if val is not None:
                return str(val)
        return "_unknown"

    def route(self, records: list[dict[str, Any]]) -> list[ShardedBatch]:
        """Partition ``records`` into per-shard ``ShardedBatch`` objects."""
        batches: dict[int, ShardedBatch] = {}
        for rec in records:
            key = self._routing_key(rec)
            shard = self.ring.shard_for(key)
            if shard not in batches:
                batches[shard] = ShardedBatch(shard=shard)
            batches[shard].records.append(rec)
        return list(batches.values())


# ---------------------------------------------------------------------------
# Backpressure queue
# ---------------------------------------------------------------------------

class QueueFullError(RuntimeError):
    """Raised when a ``BackpressureQueue`` is at capacity and ``block=False``."""


class BackpressureQueue:
    """Bounded queue with backpressure.

    Acts as an in-process substitute for Redis/RabbitMQ. The same ``put``/``get``
    interface allows a ``RedisQueue`` adapter to be swapped in without changing
    the pipeline.

    Args:
        maxsize: Maximum number of items before backpressure kicks in.
            0 = unbounded (no backpressure).
        block_on_full: When ``True`` (default), ``put()`` blocks until space is
            available. When ``False``, ``put()`` raises ``QueueFullError``
            immediately.
    """

    def __init__(self, maxsize: int = 1000, block_on_full: bool = True) -> None:
        self._q: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._block = block_on_full

    def put(self, item: Any, timeout: float | None = None) -> None:
        """Enqueue an item, honouring backpressure policy."""
        try:
            self._q.put(item, block=self._block, timeout=timeout)
        except queue.Full as exc:
            raise QueueFullError(
                f"Queue is at capacity ({self._q.maxsize} items)"
            ) from exc

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Dequeue an item."""
        try:
            return self._q.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    @property
    def qsize(self) -> int:
        return self._q.qsize()

    @property
    def full(self) -> bool:
        return self._q.full()

    @property
    def empty(self) -> bool:
        return self._q.empty()

    def drain(self) -> list[Any]:
        """Return all queued items without blocking."""
        items: list[Any] = []
        while True:
            item = self.get(block=False)
            if item is None:
                break
            items.append(item)
        return items


class RedisQueue:
    """Drop-in adapter backed by Redis (requires ``redis`` package).

    Raises ``ImportError`` when the ``redis`` package is not installed, so the
    in-process ``BackpressureQueue`` remains the default in test environments.
    """

    def __init__(self, key: str, host: str = "localhost", port: int = 6379, maxsize: int = 10_000) -> None:  # noqa: E501
        try:
            import redis as _redis  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "redis package is required for RedisQueue. "
                "Install with: pip install redis"
            ) from exc
        self._r = _redis.Redis(host=host, port=port, decode_responses=False)
        self._key = key
        self._maxsize = maxsize

    def put(self, item: Any, timeout: float | None = None) -> None:  # noqa: ARG002
        import pickle

        if self._maxsize and self._r.llen(self._key) >= self._maxsize:
            raise QueueFullError(f"Redis queue '{self._key}' is at capacity ({self._maxsize})")
        self._r.rpush(self._key, pickle.dumps(item))

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        import pickle

        if block:
            result = self._r.blpop(self._key, timeout=int(timeout or 0))
            return pickle.loads(result[1]) if result else None
        result = self._r.lpop(self._key)
        return pickle.loads(result) if result else None


# ---------------------------------------------------------------------------
# Concurrent validation workers
# ---------------------------------------------------------------------------

@dataclass
class ValidationWorkerPool:
    """Runs ``validate_aep_record`` over records concurrently.

    Args:
        max_workers: Thread count (default: min(32, cpu+4)).
        require_signature: Passed through to ``validate_aep_record``.
    """

    max_workers: int | None = None
    require_signature: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def validate_batch(
        self, records: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], AEPValidationResult]]:
        """Validate ``records`` concurrently.

        Returns a list of ``(record, result)`` pairs in the same order as input.
        """
        if not records:
            return []

        futures: list[tuple[int, Future[AEPValidationResult]]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for i, rec in enumerate(records):
                future = pool.submit(validate_aep_record, rec, self.require_signature)
                futures.append((i, future))

        results: list[tuple[dict[str, Any], AEPValidationResult] | None] = [None] * len(records)
        for i, future in futures:
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                # Wrap unexpected errors as a failed validation result
                result = AEPValidationResult(
                    run_id=records[i].get("run_id", "_unknown"),
                    valid_schema=False,
                    has_model_id=False,
                    has_actions=False,
                    has_verifier_results=False,
                    state_changing_actions_with_evidence=0,
                    state_changing_actions_total=0,
                    errors=[f"Validation worker error: {exc}"],
                )
            results[i] = (records[i], result)
        return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

@dataclass
class IngestionResult:
    """Summary of a single ingestion run."""

    total_records: int
    n_shards_used: int
    valid_count: int
    invalid_count: int
    shard_distribution: dict[int, int]  # shard → record count
    validation_results: list[AEPValidationResult]


class IngestionPipeline:
    """End-to-end ingestion: shard → queue → concurrent validate.

    Args:
        router: ``ShardRouter`` (default: 16-shard ring).
        worker_pool: ``ValidationWorkerPool`` for concurrent validation.
        queue_maxsize: Per-shard queue capacity (0 = unbounded).
        block_on_full: Whether ``put`` blocks or raises on full queue.
    """

    def __init__(
        self,
        router: ShardRouter | None = None,
        worker_pool: ValidationWorkerPool | None = None,
        queue_maxsize: int = 1000,
        block_on_full: bool = True,
    ) -> None:
        self.router = router or ShardRouter()
        self.worker_pool = worker_pool or ValidationWorkerPool()
        self.queue_maxsize = queue_maxsize
        self.block_on_full = block_on_full

    def ingest(self, records: list[dict[str, Any]]) -> IngestionResult:
        """Ingest ``records``: shard, queue, validate concurrently.

        Returns an ``IngestionResult`` summarising per-shard distribution and
        validation outcomes.
        """
        batches = self.router.route(records)

        shard_queues: dict[int, BackpressureQueue] = {
            b.shard: BackpressureQueue(
                maxsize=self.queue_maxsize, block_on_full=self.block_on_full
            )
            for b in batches
        }
        for batch in batches:
            q = shard_queues[batch.shard]
            for rec in batch.records:
                q.put(rec)

        # Drain all shard queues and validate concurrently.
        all_records: list[dict[str, Any]] = []
        shard_dist: dict[int, int] = {}
        for shard, q in shard_queues.items():
            items = q.drain()
            shard_dist[shard] = len(items)
            all_records.extend(items)

        pairs = self.worker_pool.validate_batch(all_records)
        validation_results = [r for _, r in pairs]
        valid_count = sum(1 for r in validation_results if r.passed)
        invalid_count = len(validation_results) - valid_count

        return IngestionResult(
            total_records=len(all_records),
            n_shards_used=len(shard_queues),
            valid_count=valid_count,
            invalid_count=invalid_count,
            shard_distribution=shard_dist,
            validation_results=validation_results,
        )
