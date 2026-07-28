"""Trace archival storage — time-partitioned record store with indexed retrieval.

This module is the **storage layer** of the Milestone-5 trace archival and
retrieval feature (issue #53). It implements three data-side primitives:

  1. ``StorageBackend``   — a byte-store Protocol with ``LocalBackend`` (disk,
     the default) plus thin ``S3Backend`` / ``GCSBackend`` adapters that lazily
     pull in ``boto3`` / ``google-cloud-storage``. Real bucket I/O is wired
     through these; the cluster/bucket provisioning lives outside this repo,
     mirroring how ``training_pipeline.py`` treats Ray/Lightning.
  2. ``TraceCodec``       — row ⇄ bytes serialisation. ``JsonLinesCodec`` is the
     zero-dependency default; ``ParquetCodec`` writes genuine columnar Parquet
     via ``pyarrow`` (optional ``[arch]`` extra). Both are interchangeable.
  3. ``TraceStorage``     — time-partitioned record store with an indexed query
     API. Records are laid out Hive-style under partition directories
     (``subject_id=<s>/dt=YYYY-MM-DD/<batch>.<ext>``) so that audit-trail,
     reproduction, and historical-trust queries prune partitions by
     ``subject_id`` / ``user_id`` / date-range without scanning the whole store.

The Parquet + S3/GCS pieces are *opt-in*: ``TraceStorage`` defaults to the
local backend and the JSON-Lines codec so ``python -m pytest`` passes with no
extra dependencies. Install the ``[arch]`` extra to enable the Parquet codec.
"""
from __future__ import annotations

import io
import json
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ===========================================================================
# Record helpers
# ===========================================================================

#: Supported time-partition granularities → (strftime bucket, timedelta unit).
_GRANULARITY: dict[str, str] = {
    "hour": "%Y-%m-%dT%H",
    "day": "%Y-%m-%d",
    "month": "%Y-%m",
}


def _record_to_dict(rec: Any) -> dict[str, Any]:
    """Normalise a trace record (pydantic model or plain dict) to a dict."""
    if isinstance(rec, dict):
        return rec
    if hasattr(rec, "model_dump"):  # pydantic v2
        return rec.model_dump(mode="json")
    if hasattr(rec, "dict"):  # pydantic v1 fallback
        return rec.dict()
    return dict(rec)


def _extract_field(record: dict[str, Any], name: str) -> Any:
    """Read ``name`` from a record, falling back to AEP v0.3 ``run_context``.

    AEP v0.3 nests ``user_id`` / ``subject_id`` under ``run_context``; rollout /
    training records carry them at the top level. This helper covers both.
    """
    if name in record:
        return record[name]
    ctx = record.get("run_context")
    if isinstance(ctx, dict) and name in ctx:
        return ctx[name]
    return None


def _parse_time(value: Any) -> datetime | None:
    """Coerce a timestamp value to an aware ``datetime``.

    Accepts ISO-8601 strings (with or without trailing ``Z``), epoch seconds,
    and epoch milliseconds. Returns ``None`` if the value is missing/blank.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        # Heuristic: values > 1e12 are milliseconds.
        secs = value / 1000.0 if abs(value) > 1e12 else float(value)
        return datetime.fromtimestamp(secs, tz=timezone.utc)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _record_time(record: dict[str, Any], time_field: str) -> datetime:
    """Extract the record timestamp, trying ``time_field`` then common aliases."""
    for candidate in (time_field, "timestamp", "ts", "created_at", "recorded_at", "time"):
        value = _extract_field(record, candidate)
        dt = _parse_time(value)
        if dt is not None:
            return dt
    # Last resort: ingestion time, so a partition is always derivable.
    return datetime.now(tz=timezone.utc)


def _segment(value: Any) -> str:
    """Render a partition value as a path-safe segment (Hive ``key=value``)."""
    return str(value).replace("/", "_").replace(os.sep, "_")


def _join(prefix: str, child: str) -> str:
    """Join a parent key prefix (may lack a trailing '/') and a child segment.

    ``child`` is a relative path returned by ``StorageBackend.list_dirs`` (it
    carries its own trailing '/' for directories). This helper guarantees
    exactly one '/' separates the two, so a namespace like ``"traces"`` does
    not fuse with its first partition segment.
    """
    if not prefix:
        return child
    return prefix.rstrip("/") + "/" + child


def _bucket_start(dt: datetime, granularity: str) -> datetime:
    """Floor ``dt`` to the start of its time bucket."""
    if granularity == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    if granularity == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _advance(dt: datetime, granularity: str) -> datetime:
    """Advance to the start of the next bucket (boundary, not time-of-day)."""
    start = _bucket_start(dt, granularity)
    if granularity == "hour":
        return datetime.fromtimestamp(start.timestamp() + 3600, tz=start.tzinfo)
    if granularity == "day":
        return datetime.fromtimestamp(start.timestamp() + 86400, tz=start.tzinfo)
    # month: anchor on day-1 to avoid skipped-day overflow.
    if start.month == 12:
        return start.replace(year=start.year + 1, month=1)
    return start.replace(month=start.month + 1)


def _date_buckets(start: datetime | None, end: datetime | None, granularity: str) -> list[str] | None:
    """Enumerate the distinct time buckets spanning ``[start, end]``.

    Returns ``None`` when neither bound is given (meaning "all partitions");
    otherwise returns the inclusive list of bucket strings for the granularity,
    enumerated by bucket *boundary* so a range that lands mid-bucket still
    covers every bucket it touches.
    """
    fmt = _GRANULARITY[granularity]
    if start is None and end is None:
        return None
    if start is None:
        start = end  # type: ignore[assignment]
    if end is None:
        end = start  # type: ignore[assignment]
    if end < start:
        start, end = end, start

    buckets: list[str] = []
    cursor = _bucket_start(start, granularity)
    end_bucket = _bucket_start(end, granularity)
    # Cap to avoid pathological enumeration of unbounded ranges.
    max_buckets = 100_000
    while cursor <= end_bucket and len(buckets) < max_buckets:
        buckets.append(cursor.strftime(fmt))
        cursor = _advance(cursor, granularity)
    return buckets


# ===========================================================================
# Storage backends
# ===========================================================================


@runtime_checkable
class StorageBackend(Protocol):
    """Byte-store Protocol implemented by local disk, S3, and GCS backends.

    Keys are POSIX-style relative paths (``"subject_id=s/dt=2026-07-28/b.parquet"``).
    Implementations are responsible for translating keys to their physical
    location (filesystem path, S3 object key, GCS blob name).
    """

    def write_bytes(self, key: str, data: bytes) -> None: ...
    def read_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def list_keys(self, prefix: str = "") -> list[str]: ...
    def list_dirs(self, prefix: str = "") -> list[str]: ...


class LocalBackend:
    """Filesystem-backed storage (the default; used by tests and on-disk prod)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def list_keys(self, prefix: str = "") -> list[str]:
        base = self._path(prefix)
        if not base.exists():
            return []
        out: list[str] = []
        for p in sorted(base.rglob("*")):
            if p.is_file():
                out.append(str(p.relative_to(self.root)).replace(os.sep, "/"))
        return out

    def list_dirs(self, prefix: str = "") -> list[str]:
        base = self._path(prefix)
        if not base.exists():
            return []
        return [
            child.name + "/"
            for child in sorted(base.iterdir())
            if child.is_dir()
        ]


class S3Backend:
    """Amazon S3 storage adapter — lazily imports ``boto3``.

    Real bucket I/O (get/put/list/delete) is implemented; credentials, endpoint
    configuration, and bucket provisioning live with the deployment, not here.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        client: Any = None,
        endpoint_url: str | None = None,
        region: str | None = None,
        not_found_exc: type[BaseException] | tuple[type[BaseException], ...] | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        # Exception(s) ``head_object`` raises for a missing key. Defaults to
        # ``botocore.exceptions.ClientError`` (lazy); injectable so the adapter
        # is testable without the AWS SDK installed.
        self._not_found_exc = not_found_exc
        if client is not None:
            self._client = client
        else:  # pragma: no cover - exercised only when boto3 is installed
            import boto3  # type: ignore[import-not-found]

            kwargs: dict[str, Any] = {}
            if endpoint_url:
                kwargs["endpoint_url"] = endpoint_url
            if region:
                kwargs["region_name"] = region
            self._client = boto3.client("s3", **kwargs)

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def write_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def read_bytes(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()

    def exists(self, key: str) -> bool:
        if self._not_found_exc is None:
            from botocore.exceptions import ClientError  # pragma: no cover - needs AWS SDK

            not_found = ClientError
        else:
            not_found = self._not_found_exc
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except not_found:
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=self._key(key))

    def list_keys(self, prefix: str = "") -> list[str]:
        full = self._key(prefix)
        paginator = self._client.get_paginator("list_objects_v2")
        out: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []):
                if self.prefix:
                    out.append(obj["Key"][len(self.prefix) + 1 :])
                else:
                    out.append(obj["Key"])
        return out

    def list_dirs(self, prefix: str = "") -> list[str]:
        full = self._key(prefix)
        resp = self._client.list_objects_v2(
            Bucket=self.bucket, Prefix=full, Delimiter="/"
        )
        return [p["Prefix"].rstrip("/").split("/")[-1] + "/" for p in resp.get("CommonPrefixes", [])]


class GCSBackend:
    """Google Cloud Storage adapter — lazily imports ``google-cloud-storage``."""

    def __init__(self, bucket: str, prefix: str = "", *, client: Any = None) -> None:
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        if client is not None:
            self._client = client
        else:  # pragma: no cover
            from google.cloud import storage  # type: ignore[import-not-found]

            self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)

    def _name(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def write_bytes(self, key: str, data: bytes) -> None:  # pragma: no cover
        self._bucket.blob(self._name(key)).upload_from_string(data)

    def read_bytes(self, key: str) -> bytes:  # pragma: no cover
        return self._bucket.blob(self._name(key)).download_as_bytes()

    def exists(self, key: str) -> bool:  # pragma: no cover
        return self._bucket.blob(self._name(key)).exists()

    def delete(self, key: str) -> None:  # pragma: no cover
        self._bucket.blob(self._name(key)).delete()

    def list_keys(self, prefix: str = "") -> list[str]:  # pragma: no cover
        full = self._name(prefix)
        out: list[str] = []
        for blob in self._client.list_blobs(self.bucket_name, prefix=full):
            if self.prefix:
                out.append(blob.name[len(self.prefix) + 1 :])
            else:
                out.append(blob.name)
        return out

    def list_dirs(self, prefix: str = "") -> list[str]:  # pragma: no cover
        full = self._name(prefix)
        # list_blobs with delimiter yields "prefixes" for immediate sub-dirs.
        kwargs: dict[str, Any] = {"prefix": full, "delimiter": "/"}
        _iter = self._client.list_blobs(self.bucket_name, **kwargs)
        list(_iter)  # consume to populate prefixes
        return [
            p.rstrip("/").split("/")[-1] + "/"
            for p in _iter.prefixes
        ]


# ===========================================================================
# Codecs
# ===========================================================================


@runtime_checkable
class TraceCodec(Protocol):
    """Row ⇄ bytes serialisation contract for stored trace batches."""

    ext: str

    def dumps(self, rows: list[dict[str, Any]]) -> bytes: ...
    def loads(self, data: bytes) -> list[dict[str, Any]]: ...


class JsonLinesCodec:
    """Zero-dependency newline-delimited JSON codec (default)."""

    ext = "jsonl"

    def dumps(self, rows: list[dict[str, Any]]) -> bytes:
        return ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)).encode("utf-8")

    def loads(self, data: bytes) -> list[dict[str, Any]]:
        if not data:
            return []
        return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


class ParquetCodec:
    """Columnar Parquet codec backed by ``pyarrow`` (``[arch]`` extra).

    Each top-level record field becomes a Parquet column; heterogeneous batches
    (records with differing keys) are handled by ``pyarrow.Table.from_pylist``,
    which fills absent fields with nulls — exactly the Hive/Iceberg behaviour
    trace archival relies on.
    """

    ext = "parquet"

    def __init__(self, compression: str = "snappy") -> None:
        self.compression = compression

    def _import(self):  # pragma: no cover - imported lazily
        try:
            import pyarrow as pa  # type: ignore[import-not-found]
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ParquetCodec requires pyarrow. Install with: pip install 'evomerge[arch]'"
            ) from exc
        return pa, pq

    def dumps(self, rows: list[dict[str, Any]]) -> bytes:
        pa, pq = self._import()  # pragma: no cover
        if not rows:
            rows = [{"_empty": True}]  # parquet needs at least one column
        table = pa.Table.from_pylist(rows)
        sink = io.BytesIO()
        pq.write_table(table, sink, compression=self.compression)
        return sink.getvalue()

    def loads(self, data: bytes) -> list[dict[str, Any]]:  # pragma: no cover
        pa, pq = self._import()
        table = pq.read_table(io.BytesIO(data))
        rows = table.to_pylist()
        return [r for r in rows if not r.get("_empty")]


# ===========================================================================
# Trace storage
# ===========================================================================


@dataclass
class StorageManifest:
    """Summary of a ``TraceStorage`` write batch."""

    written: int                       # records persisted
    partitions: list[str]              # partition dirs touched
    objects: list[str]                 # object keys written

    @property
    def n_partitions(self) -> int:
        return len(self.partitions)


class TraceStorage:
    """Time-partitioned trace store with indexed query APIs.

    Records are persisted under Hive-style partition directories whose segments
    correspond to ``partition_by`` (default ``subject_id``) and a time bucket
    ``dt`` (granularity-controlled). The directory layout *is* the index: an
    audit-trail / reproduction / historical-trust query supplies a
    ``subject_id`` / ``user_id`` / date-range and only the matching partitions
    are read — the rest are pruned at the backend-listing level.
    """

    def __init__(
        self,
        backend: StorageBackend,
        *,
        codec: TraceCodec | None = None,
        partition_by: Sequence[str] = ("subject_id",),
        time_field: str = "timestamp",
        granularity: str = "day",
        namespace: str = "traces",
    ) -> None:
        if granularity not in _GRANULARITY:
            raise ValueError(
                f"granularity must be one of {sorted(_GRANULARITY)}, got {granularity!r}"
            )
        for dim in partition_by:
            if dim not in ("subject_id", "user_id", "tenant"):
                raise ValueError(
                    f"partition_by dimension {dim!r} not supported "
                    "(allowed: subject_id, user_id, tenant)"
                )
        self.backend = backend
        self.codec = codec if codec is not None else JsonLinesCodec()
        self.partition_dims = tuple(partition_by)
        self.time_field = time_field
        self.granularity = granularity
        self.namespace = namespace.strip("/")

    # -- partition layout ---------------------------------------------------

    def _partition_dir(self, record: dict[str, Any]) -> str:
        """Compute the Hive-style partition directory for one record."""
        dt = _record_time(record, self.time_field)
        parts = [self.namespace] if self.namespace else []
        for dim in self.partition_dims:
            value = _extract_field(record, dim)
            # Unknown shard key → "_unknown" bucket so the record is still
            # retrievable (and visibly un-partitioned) rather than dropped.
            parts.append(f"{dim}={_segment(value) if value is not None else '_unknown'}")
        parts.append(f"dt={dt.strftime(_GRANULARITY[self.granularity])}")
        return "/".join(parts)

    def _object_key(self, partition_dir: str, batch_id: str) -> str:
        return f"{partition_dir}/{batch_id}.{self.codec.ext}"

    # -- write --------------------------------------------------------------

    def write(self, records: Iterable[Any]) -> StorageManifest:
        """Persist records, grouped by partition, one object per (batch, partition).

        ``write`` is append-safe: each call emits new objects keyed by a
        monotonic batch id, so concurrent writers do not clobber one another.
        """
        buckets: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            d = _record_to_dict(rec)
            buckets.setdefault(self._partition_dir(d), []).append(d)

        partition_dirs: list[str] = []
        object_keys: list[str] = []
        total = 0
        for pdir, rows in sorted(buckets.items()):
            partition_dirs.append(pdir + "/")
            batch_id = f"batch-{int(time.time() * 1_000_000)}"
            key = self._object_key(pdir, batch_id)
            self.backend.write_bytes(key, self.codec.dumps(rows))
            object_keys.append(key)
            total += len(rows)
        return StorageManifest(written=total, partitions=partition_dirs, objects=object_keys)

    # -- indexed query ------------------------------------------------------

    def _candidate_dirs(
        self,
        subject_id: str | None,
        user_id: str | None,
        start: datetime | None,
        end: datetime | None,
    ) -> list[str]:
        """Return the partition directories a query must descend into.

        Implements partition pruning: exact-match dimensions narrow to one
        child, unspecified dimensions expand to all children (via
        ``backend.list_dirs``), and the time range narrows to the specific
        ``dt=`` buckets it spans.
        """
        bounds = {"subject_id": subject_id, "user_id": user_id}
        candidates: list[str] = [self.namespace] if self.namespace else [""]
        for dim in self.partition_dims:
            value = bounds.get(dim)
            if value is not None:
                candidates = [_join(c, f"{dim}={_segment(value)}/") for c in candidates]
            else:
                expanded: list[str] = []
                for c in candidates:
                    expanded.extend(_join(c, d) for d in self.backend.list_dirs(c))
                candidates = expanded
        # Time bucket level.
        buckets = _date_buckets(start, end, self.granularity)
        if buckets is not None:
            candidates = [_join(c, f"dt={b}/") for c in candidates for b in buckets]
        else:
            expanded = []
            for c in candidates:
                expanded.extend(
                    _join(c, d) for d in self.backend.list_dirs(c) if d.startswith("dt=")
                )
            candidates = expanded
        return candidates

    def query(
        self,
        *,
        subject_id: str | None = None,
        user_id: str | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        rollout_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Indexed retrieval of stored records.

        Partitions are pruned by ``subject_id`` / ``user_id`` / ``[start, end]``
        before any object is read; remaining records are filtered in-memory by
        the full predicate and ordered by timestamp (ascending). ``rollout_id``
        supports deterministic single-trace reproduction.
        """
        start_dt = _parse_time(start) if not isinstance(start, datetime) else start
        end_dt = _parse_time(end) if not isinstance(end, datetime) else end

        results: list[dict[str, Any]] = []
        for cdir in self._candidate_dirs(subject_id, user_id, start_dt, end_dt):
            for key in self.backend.list_keys(cdir):
                rows = self.codec.loads(self.backend.read_bytes(key))
                for row in rows:
                    if not self._matches(row, subject_id, user_id, start_dt, end_dt, rollout_id):
                        continue
                    results.append(row)

        results.sort(key=lambda r: _record_time(r, self.time_field))
        if limit is not None:
            results = results[:limit]
        return results

    @staticmethod
    def _matches(
        row: dict[str, Any],
        subject_id: str | None,
        user_id: str | None,
        start: datetime | None,
        end: datetime | None,
        rollout_id: str | None,
    ) -> bool:
        if subject_id is not None and str(_extract_field(row, "subject_id")) != str(subject_id):
            return False
        if user_id is not None and str(_extract_field(row, "user_id")) != str(user_id):
            return False
        if rollout_id is not None and str(row.get("rollout_id")) != str(rollout_id):
            return False
        if start is not None or end is not None:
            dt = _record_time(row, "timestamp")
            if start is not None and dt < start:
                return False
            if end is not None and dt > end:
                return False
        return True

    # -- introspection ------------------------------------------------------

    def partitions(self) -> list[str]:
        """List all non-empty partition directories currently in the store.

        A directory whose objects were migrated/deleted (and is now empty) is
        not reported as a live partition.
        """
        candidates: list[str] = [self.namespace] if self.namespace else [""]
        # Walk all partition levels (shard dims + dt).
        levels = len(self.partition_dims) + 1
        for _ in range(levels):
            expanded: list[str] = []
            for c in candidates:
                expanded.extend(_join(c, d) for d in self.backend.list_dirs(c))
            candidates = expanded
        return [c for c in candidates if self.backend.list_keys(c)]

    def count(self) -> int:
        """Total record count across all partitions."""
        total = 0
        for cdir in self.partitions():
            for key in self.backend.list_keys(cdir):
                total += len(self.codec.loads(self.backend.read_bytes(key)))
        return total

    def get(self, rollout_id: str) -> list[dict[str, Any]]:
        """Deterministic reproduction fetch — all records for one rollout id."""
        return self.query(rollout_id=rollout_id)


def open_trace_storage(
    root: str | Path,
    *,
    codec: str | TraceCodec = "jsonl",
    **kwargs: Any,
) -> TraceStorage:
    """Convenience factory: a ``TraceStorage`` over a local directory.

    ``codec`` may be ``"jsonl"`` (default), ``"parquet"``, or a ``TraceCodec``
    instance.
    """
    if isinstance(codec, str):
        codec = ParquetCodec() if codec == "parquet" else JsonLinesCodec()
    return TraceStorage(LocalBackend(root), codec=codec, **kwargs)


__all__ = [
    "GCSBackend",
    "JsonLinesCodec",
    "LocalBackend",
    "ParquetCodec",
    "S3Backend",
    "StorageBackend",
    "StorageManifest",
    "TraceCodec",
    "TraceStorage",
    "open_trace_storage",
]
