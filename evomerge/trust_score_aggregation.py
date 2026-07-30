"""Trust-score aggregation and serving — issue #56.

Materialises per-subject trust scores from historical traces, serves
low-latency lookups via an in-process caching layer (pluggable Redis/Memcached
adapter), and exposes Prometheus-compatible metrics for score distribution and
drift monitoring.

Design
------
- ``TrustScoreAggregator`` — consumes a sequence of AEP record dicts for a
  given ``subject_id`` and returns an ``AggregatedTrustScore``.
- ``TrustScoreCache`` — LRU in-process cache.  ``RedisCache`` / ``MemcachedCache``
  adapters are provided as drop-in replacements that raise ``ImportError`` when
  the optional client libraries are absent.
- ``TrustScoreStore`` — materialises scores into a dict-backed store (or an
  injected backend), supports point lookups and bulk refresh.
- ``TrustMetrics`` — Prometheus-compatible metric registry (plain Python,
  no ``prometheus_client`` hard-dep).  ``PrometheusMetrics`` wraps
  ``prometheus_client`` when it is installed.
- ``TrustScoreServer`` — wires together aggregator + cache + store + metrics.
"""
from __future__ import annotations

import math
import statistics
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from evomerge.trust_score import AgentTrustScoreBuilder, compute_trust_score

__all__ = [
    "AggregatedTrustScore",
    "TrustScoreAggregator",
    "TrustScoreCache",
    "RedisCache",
    "MemcachedCache",
    "TrustScoreStore",
    "TrustMetrics",
    "PrometheusMetrics",
    "TrustScoreServer",
]


# ---------------------------------------------------------------------------
# Aggregated score
# ---------------------------------------------------------------------------

@dataclass
class AggregatedTrustScore:
    """Materialised trust score for a single subject.

    Attributes:
        subject_id: The agent/subject identifier.
        overall: Geometric mean of ``overall`` across all historical records
            (``None`` if no records provided).
        n_records: Number of AEP records aggregated.
        mean: Arithmetic mean of individual overall scores.
        std_dev: Sample standard deviation (``None`` if < 2 records).
        min_score: Minimum observed overall score.
        max_score: Maximum observed overall score.
        drift: Absolute difference between the mean of the first half and the
            mean of the second half of records (chronological order). ``None``
            if fewer than 4 records.
        computed_at: Unix epoch timestamp (float) when this score was computed.
    """

    subject_id: str
    overall: float | None
    n_records: int
    mean: float | None
    std_dev: float | None
    min_score: float | None
    max_score: float | None
    drift: float | None
    computed_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class TrustScoreAggregator:
    """Compute an ``AggregatedTrustScore`` from a list of AEP record dicts."""

    def aggregate(
        self,
        subject_id: str,
        records: list[dict[str, Any]],
    ) -> AggregatedTrustScore:
        """Materialise the trust score for *subject_id* from *records*.

        Records are consumed in the order provided (callers should pass them
        in chronological order for correct drift computation).
        """
        scores: list[float] = []
        for rec in records:
            score = compute_trust_score(aep_record=rec)
            if score.overall is not None:
                scores.append(score.overall)

        if not scores:
            return AggregatedTrustScore(
                subject_id=subject_id,
                overall=None,
                n_records=len(records),
                mean=None,
                std_dev=None,
                min_score=None,
                max_score=None,
                drift=None,
            )

        # Geometric mean across runs
        log_sum = sum(math.log(s) if s > 0 else float("-inf") for s in scores)
        overall = math.exp(log_sum / len(scores)) if all(s > 0 for s in scores) else 0.0
        mean = statistics.mean(scores)
        std_dev = statistics.stdev(scores) if len(scores) >= 2 else None
        min_score = min(scores)
        max_score = max(scores)

        drift: float | None = None
        if len(scores) >= 4:
            mid = len(scores) // 2
            first_half = scores[:mid]
            second_half = scores[mid:]
            drift = abs(statistics.mean(second_half) - statistics.mean(first_half))

        return AggregatedTrustScore(
            subject_id=subject_id,
            overall=overall,
            n_records=len(records),
            mean=mean,
            std_dev=std_dev,
            min_score=min_score,
            max_score=max_score,
            drift=drift,
        )


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------

class TrustScoreCache:
    """Thread-safe in-process LRU cache for ``AggregatedTrustScore`` objects.

    Args:
        maxsize: Maximum number of entries to retain in memory.
        ttl_seconds: Cache entry time-to-live in seconds (0 = no expiry).
    """

    def __init__(self, maxsize: int = 512, ttl_seconds: float = 300.0) -> None:
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[AggregatedTrustScore, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, subject_id: str) -> AggregatedTrustScore | None:
        """Return cached score or ``None`` if absent/stale."""
        with self._lock:
            entry = self._cache.get(subject_id)
            if entry is None:
                return None
            score, stored_at = entry
            if self._ttl > 0 and (time.time() - stored_at) > self._ttl:
                del self._cache[subject_id]
                return None
            self._cache.move_to_end(subject_id)
            return score

    def set(self, subject_id: str, score: AggregatedTrustScore) -> None:
        """Store *score* for *subject_id*, evicting LRU entry if needed."""
        with self._lock:
            if subject_id in self._cache:
                self._cache.move_to_end(subject_id)
            self._cache[subject_id] = (score, time.time())
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def invalidate(self, subject_id: str) -> None:
        """Remove the cached entry for *subject_id* (no-op if absent)."""
        with self._lock:
            self._cache.pop(subject_id, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class RedisCache:
    """Drop-in cache adapter backed by Redis (requires ``redis`` package).

    Values are JSON-serialised so they survive across process restarts.
    Raises ``ImportError`` when ``redis`` is not installed.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, ttl_seconds: int = 300, prefix: str = "trust:") -> None:  # noqa: E501
        try:
            import redis as _redis  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "redis package is required for RedisCache. Install with: pip install redis"
            ) from exc
        self._r = _redis.Redis(host=host, port=port, decode_responses=True)
        self._ttl = ttl_seconds
        self._prefix = prefix

    def _key(self, subject_id: str) -> str:
        return f"{self._prefix}{subject_id}"

    def get(self, subject_id: str) -> AggregatedTrustScore | None:
        import json
        raw = self._r.get(self._key(subject_id))
        if raw is None:
            return None
        data = json.loads(raw)
        return AggregatedTrustScore(**data)

    def set(self, subject_id: str, score: AggregatedTrustScore) -> None:
        import dataclasses, json
        raw = json.dumps(dataclasses.asdict(score))
        self._r.setex(self._key(subject_id), self._ttl, raw)

    def invalidate(self, subject_id: str) -> None:
        self._r.delete(self._key(subject_id))

    def clear(self) -> None:
        for key in self._r.scan_iter(f"{self._prefix}*"):
            self._r.delete(key)


class MemcachedCache:
    """Drop-in cache adapter backed by Memcached (requires ``pymemcache``).

    Raises ``ImportError`` when ``pymemcache`` is not installed.
    """

    def __init__(self, host: str = "localhost", port: int = 11211, ttl_seconds: int = 300, prefix: str = "trust:") -> None:  # noqa: E501
        try:
            from pymemcache.client.base import Client  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "pymemcache is required for MemcachedCache. Install with: pip install pymemcache"
            ) from exc
        self._client = Client((host, port))
        self._ttl = ttl_seconds
        self._prefix = prefix

    def _key(self, subject_id: str) -> str:
        return f"{self._prefix}{subject_id}"

    def get(self, subject_id: str) -> AggregatedTrustScore | None:
        import dataclasses, json
        raw = self._client.get(self._key(subject_id))
        if raw is None:
            return None
        data = json.loads(raw)
        return AggregatedTrustScore(**data)

    def set(self, subject_id: str, score: AggregatedTrustScore) -> None:
        import dataclasses, json
        raw = json.dumps(dataclasses.asdict(score))
        self._client.set(self._key(subject_id), raw, expire=self._ttl)

    def invalidate(self, subject_id: str) -> None:
        self._client.delete(self._key(subject_id))

    def clear(self) -> None:
        self._client.flush_all()


# ---------------------------------------------------------------------------
# Materialised store
# ---------------------------------------------------------------------------

class TrustScoreStore:
    """Materialised store of ``AggregatedTrustScore`` objects.

    ``backend`` must expose ``get(subject_id)``, ``set(subject_id, score)``,
    ``invalidate(subject_id)``, and ``clear()``.  Defaults to
    ``TrustScoreCache`` when not provided.
    """

    def __init__(
        self,
        aggregator: TrustScoreAggregator | None = None,
        backend: Any = None,
    ) -> None:
        self._aggregator = aggregator or TrustScoreAggregator()
        self._backend = backend or TrustScoreCache()
        # Fallback dict for subjects not in cache
        self._store: dict[str, AggregatedTrustScore] = {}
        self._lock = threading.Lock()

    def get(self, subject_id: str) -> AggregatedTrustScore | None:
        """Look up a materialised score (cache-first, then fallback store)."""
        cached = self._backend.get(subject_id)
        if cached is not None:
            return cached
        with self._lock:
            return self._store.get(subject_id)

    def refresh(self, subject_id: str, records: list[dict[str, Any]]) -> AggregatedTrustScore:
        """Recompute and materialise the score for *subject_id* from *records*."""
        score = self._aggregator.aggregate(subject_id, records)
        self._backend.set(subject_id, score)
        with self._lock:
            self._store[subject_id] = score
        return score

    def bulk_refresh(
        self, subject_records: dict[str, list[dict[str, Any]]]
    ) -> dict[str, AggregatedTrustScore]:
        """Refresh scores for multiple subjects.

        Args:
            subject_records: Mapping of ``subject_id`` → list of AEP dicts.

        Returns:
            Mapping of ``subject_id`` → ``AggregatedTrustScore``.
        """
        return {sid: self.refresh(sid, recs) for sid, recs in subject_records.items()}

    def invalidate(self, subject_id: str) -> None:
        self._backend.invalidate(subject_id)
        with self._lock:
            self._store.pop(subject_id, None)

    def all_scores(self) -> dict[str, AggregatedTrustScore]:
        """Return all materialised scores."""
        with self._lock:
            return dict(self._store)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class _GaugeSample:
    name: str
    labels: dict[str, str]
    value: float
    timestamp: float = field(default_factory=time.time)


class TrustMetrics:
    """Prometheus-compatible metric registry (no external dep).

    Metrics exposed:
        - ``trust_score_overall``  — per-subject overall score gauge.
        - ``trust_score_drift``    — per-subject drift gauge.
        - ``trust_score_n_records``— number of records aggregated.
        - ``trust_score_distribution_bucket`` — histogram-like buckets.
    """

    _BUCKETS = (0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0)

    def __init__(self) -> None:
        self._samples: list[_GaugeSample] = []
        self._lock = threading.Lock()

    def record(self, score: AggregatedTrustScore) -> None:
        """Record metrics for *score*."""
        labels = {"subject_id": score.subject_id}
        samples: list[_GaugeSample] = []
        if score.overall is not None:
            samples.append(_GaugeSample("trust_score_overall", labels, score.overall))
            for bucket in self._BUCKETS:
                samples.append(_GaugeSample(
                    "trust_score_distribution_bucket",
                    {**labels, "le": str(bucket)},
                    float(score.overall <= bucket),
                ))
        if score.drift is not None:
            samples.append(_GaugeSample("trust_score_drift", labels, score.drift))
        samples.append(_GaugeSample("trust_score_n_records", labels, float(score.n_records)))
        with self._lock:
            self._samples.extend(samples)

    def exposition(self) -> str:
        """Return a Prometheus text-format exposition string."""
        lines: list[str] = []
        with self._lock:
            for s in self._samples:
                label_str = ",".join(f'{k}="{v}"' for k, v in s.labels.items())
                lines.append(f'{s.name}{{{label_str}}} {s.value}')
        return "\n".join(lines)

    def samples(self) -> list[_GaugeSample]:
        with self._lock:
            return list(self._samples)

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()


class PrometheusMetrics:
    """Thin wrapper that forwards to ``prometheus_client`` when installed.

    Falls back to ``TrustMetrics`` (no-op exporter) if the package is absent.
    """

    def __init__(self, registry: Any = None) -> None:
        try:
            import prometheus_client as _pc  # type: ignore[import-not-found]
            self._pc = _pc
            self._registry = registry or _pc.CollectorRegistry()
            self._overall = _pc.Gauge(
                "trust_score_overall",
                "Per-subject overall trust score",
                ["subject_id"],
                registry=self._registry,
            )
            self._drift = _pc.Gauge(
                "trust_score_drift",
                "Per-subject trust-score drift between first/second half of history",
                ["subject_id"],
                registry=self._registry,
            )
            self._n_records = _pc.Gauge(
                "trust_score_n_records",
                "Number of AEP records aggregated per subject",
                ["subject_id"],
                registry=self._registry,
            )
            self._fallback: TrustMetrics | None = None
        except ImportError:
            self._pc = None
            self._fallback = TrustMetrics()

    def record(self, score: AggregatedTrustScore) -> None:
        if self._pc is None:
            assert self._fallback is not None
            self._fallback.record(score)
            return
        sid = score.subject_id
        if score.overall is not None:
            self._overall.labels(subject_id=sid).set(score.overall)
        if score.drift is not None:
            self._drift.labels(subject_id=sid).set(score.drift)
        self._n_records.labels(subject_id=sid).set(score.n_records)

    def exposition(self) -> str:
        if self._fallback is not None:
            return self._fallback.exposition()
        return self._pc.generate_latest(self._registry).decode()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class TrustScoreServer:
    """Wires aggregator + cache + store + metrics into a single serving unit.

    This is the entry point for the score-serving path. In a production
    deployment this object would live inside a FastAPI/gRPC handler; in tests
    it is used directly.
    """

    def __init__(
        self,
        store: TrustScoreStore | None = None,
        metrics: TrustMetrics | PrometheusMetrics | None = None,
    ) -> None:
        self.store = store or TrustScoreStore()
        self.metrics = metrics or TrustMetrics()

    def get(self, subject_id: str) -> AggregatedTrustScore | None:
        """Serve the cached/materialised score for *subject_id*."""
        return self.store.get(subject_id)

    def refresh(
        self, subject_id: str, records: list[dict[str, Any]]
    ) -> AggregatedTrustScore:
        """Recompute, materialise, and return the score for *subject_id*."""
        score = self.store.refresh(subject_id, records)
        self.metrics.record(score)
        return score

    def bulk_refresh(
        self, subject_records: dict[str, list[dict[str, Any]]]
    ) -> dict[str, AggregatedTrustScore]:
        """Refresh and serve scores for multiple subjects."""
        scores = self.store.bulk_refresh(subject_records)
        for score in scores.values():
            self.metrics.record(score)
        return scores
