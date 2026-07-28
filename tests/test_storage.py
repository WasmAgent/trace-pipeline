"""Tests for evomerge.storage — Milestone 5 trace archival & retrieval (issue #53).

Covers the storage-layer sub-requirements, each in its own test class:

  - TestPartitioning   — time-partitioned layout (Hive-style subject_id=/dt= dirs)
  - TestCodecs         — JsonLinesCodec round-trip + ParquetCodec (pyarrow opt-in)
  - TestIndexedQuery   — audit-trail / reproduction / historical-trust retrieval
  - TestBackendAdapter — S3Backend routing via an injected fake client
  - TestStorageTrustLoop — end-to-end: query() feeds add_historical_traces()
"""
from __future__ import annotations

import builtins

import pytest

from evomerge.storage import (
    GCSBackend,
    JsonLinesCodec,
    LocalBackend,
    ParquetCodec,
    S3Backend,
    StorageBackend,
    StorageManifest,
    TraceStorage,
    open_trace_storage,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _trace(rollout_id, subject_id="s1", user_id="u1", ts="2026-07-01T10:00:00Z",
           status="pass", score=1):
    return {
        "rollout_id": rollout_id,
        "subject_id": subject_id,
        "user_id": user_id,
        "timestamp": ts,
        "objective_status": status,
        "objective_score": score,
    }


@pytest.fixture
def store(tmp_path):
    return TraceStorage(LocalBackend(tmp_path), granularity="day")


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

class TestPartitioning:
    def test_records_partitioned_by_subject_and_day(self, store):
        recs = [
            _trace("r1", "s1", ts="2026-07-01T10:00:00Z"),
            _trace("r2", "s1", ts="2026-07-02T10:00:00Z"),
            _trace("r3", "s2", ts="2026-07-01T11:00:00Z"),
        ]
        manifest = store.write(recs)
        assert isinstance(manifest, StorageManifest)
        assert manifest.written == 3
        assert manifest.n_partitions == 3
        parts = set(store.partitions())
        assert "traces/subject_id=s1/dt=2026-07-01/" in parts
        assert "traces/subject_id=s1/dt=2026-07-02/" in parts
        assert "traces/subject_id=s2/dt=2026-07-01/" in parts

    def test_unknown_shard_key_bucketed_not_dropped(self, store):
        # A record missing subject_id must still be retrievable.
        rec = {"rollout_id": "rx", "timestamp": "2026-07-01T00:00:00Z"}
        store.write([rec])
        out = store.query()
        assert [r["rollout_id"] for r in out] == ["rx"]
        assert any("subject_id=_unknown" in p for p in store.partitions())

    def test_granularity_variants(self, tmp_path):
        for gran, expected in [("hour", "2026-07-01T10"), ("month", "2026-07")]:
            s = TraceStorage(LocalBackend(tmp_path / gran), granularity=gran)
            s.write([_trace("r1", ts="2026-07-01T10:30:00Z")])
            assert any(f"dt={expected}" in p for p in s.partitions())

    def test_namespace_layout(self, tmp_path):
        s = TraceStorage(LocalBackend(tmp_path), namespace="archive", granularity="day")
        s.write([_trace("r1", ts="2026-07-01T00:00:00Z")])
        assert all(p.startswith("archive/") for p in s.partitions())

    def test_invalid_granularity_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="granularity"):
            TraceStorage(LocalBackend(tmp_path), granularity="week")

    def test_invalid_partition_dim_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="partition_by"):
            TraceStorage(LocalBackend(tmp_path), partition_by=("session_id",))


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------

class TestCodecs:
    def test_jsonl_codec_round_trip(self, tmp_path):
        codec = JsonLinesCodec()
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y", "c": [1, 2]}]
        assert codec.loads(codec.dumps(rows)) == rows
        assert codec.loads(b"") == []
        assert isinstance(LocalBackend(tmp_path), StorageBackend)

    def test_parquet_codec_round_trip(self):
        pytest.importorskip("pyarrow")
        codec = ParquetCodec()
        rows = [
            {"rollout_id": "r1", "subject_id": "s1", "score": 1},
            {"rollout_id": "r2", "subject_id": "s1", "score": 0},
        ]
        out = codec.loads(codec.dumps(rows))
        assert out == rows

    def test_parquet_codec_missing_dependency_message(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in ("pyarrow", "pyarrow.parquet"):
                raise ImportError("simulated missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match=r"evomerge\[arch\]"):
            ParquetCodec().dumps([{"a": 1}])

    def test_storage_with_parquet_codec(self, tmp_path):
        pytest.importorskip("pyarrow")
        s = TraceStorage(LocalBackend(tmp_path), codec=ParquetCodec(), granularity="day")
        manifest = s.write([_trace("r1", ts="2026-07-01T00:00:00Z")])
        assert all(key.endswith(".parquet") for key in manifest.objects)
        out = s.query(subject_id="s1")
        assert [r["rollout_id"] for r in out] == ["r1"]


# ---------------------------------------------------------------------------
# Indexed query
# ---------------------------------------------------------------------------

class TestIndexedQuery:
    @pytest.fixture
    def populated(self, store):
        store.write([
            _trace("r1", "s1", user_id="u1", ts="2026-07-01T10:00:00Z"),
            _trace("r2", "s1", user_id="u1", ts="2026-07-02T10:00:00Z", status="fail", score=0),
            _trace("r3", "s2", user_id="u2", ts="2026-07-01T11:00:00Z"),
        ])
        return store

    def test_audit_trail_by_subject(self, populated):
        ids = [r["rollout_id"] for r in populated.query(subject_id="s1")]
        assert ids == ["r1", "r2"]

    def test_audit_trail_by_subject_and_date_range(self, populated):
        out = populated.query(
            subject_id="s1",
            start="2026-07-02T00:00:00Z",
            end="2026-07-02T23:59:59Z",
        )
        assert [r["rollout_id"] for r in out] == ["r2"]

    def test_date_range_spanning_multiple_buckets(self, populated):
        out = populated.query(start="2026-07-01T00:00:00Z", end="2026-07-02T23:59:59Z")
        assert {r["rollout_id"] for r in out} == {"r1", "r2", "r3"}

    def test_reproduction_by_rollout_id(self, populated):
        out = populated.get("r3")
        assert [r["rollout_id"] for r in out] == ["r3"]

    def test_partition_pruning_skips_other_subjects(self, populated):
        # Querying s1 must never read s2's partition: r3 absent.
        out = populated.query(subject_id="s1")
        assert "r3" not in {r["rollout_id"] for r in out}

    def test_user_id_query(self, populated):
        out = populated.query(user_id="u1")
        assert {r["rollout_id"] for r in out} == {"r1", "r2"}

    def test_results_ordered_by_timestamp(self, populated):
        out = populated.query()
        ts = [r["timestamp"] for r in out]
        assert ts == sorted(ts)

    def test_limit_applied_after_order(self, populated):
        out = populated.query(limit=2)
        assert len(out) == 2

    def test_count(self, populated):
        assert populated.count() == 3

    def test_empty_query_returns_empty(self, store):
        assert store.query(subject_id="nobody") == []


# ---------------------------------------------------------------------------
# Backend adapter (S3 via injected fake client — no boto3 required)
# ---------------------------------------------------------------------------

class TestBackendAdapter:
    def _fake_s3(self):
        class FakeS3:
            def __init__(self):
                self.objects = {}  # key -> bytes

            def put_object(self, *, Bucket, Key, Body):
                self.objects[(Bucket, Key)] = Body

            def get_object(self, *, Bucket, Key):
                return {"Body": _Reader(self.objects[(Bucket, Key)])}

            def head_object(self, *, Bucket, Key):
                if (Bucket, Key) not in self.objects:
                    raise KeyError(Key)
                return {}

            def delete_object(self, *, Bucket, Key):
                self.objects.pop((Bucket, Key), None)

            def get_paginator(self, name):
                assert name == "list_objects_v2"
                return _Paginator(self)

            def list_objects_v2(self, *, Bucket, Prefix, Delimiter=None):
                # Only the delimiter form (CommonPrefixes) is used by list_dirs.
                prefix = Prefix
                commons = set()
                for (b, k) in self.objects:
                    if b != Bucket or not k.startswith(prefix):
                        continue
                    rest = k[len(prefix):]
                    if Delimiter == "/" and "/" in rest:
                        commons.add(prefix + rest.split("/", 1)[0] + "/")
                return {"CommonPrefixes": [{"Prefix": c} for c in sorted(commons)]}

        class _Reader:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        class _Paginator:
            def __init__(self, s3):
                self.s3 = s3

            def paginate(self, *, Bucket, Prefix):
                keys = [k for (b, k) in self.s3.objects if b == Bucket and k.startswith(Prefix)]
                yield {"Contents": [{"Key": k} for k in sorted(keys)]}

        return FakeS3()

    def test_s3_backend_routes_through_client(self):
        client = self._fake_s3()
        backend = S3Backend("mybucket", prefix="traces", client=client, not_found_exc=KeyError)
        backend.write_bytes("a/b/c.jsonl", b"hello")
        assert backend.read_bytes("a/b/c.jsonl") == b"hello"
        assert backend.exists("a/b/c.jsonl") is True
        assert backend.exists("a/b/missing.jsonl") is False
        backend.delete("a/b/c.jsonl")
        assert backend.list_keys("a/b/") == []

    def test_s3_backend_list_keys_and_dirs(self):
        client = self._fake_s3()
        backend = S3Backend("mybucket", prefix="traces", client=client)
        backend.write_bytes("subject_id=s1/dt=2026-07-01/b.jsonl", b"x")
        backend.write_bytes("subject_id=s1/dt=2026-07-02/b.jsonl", b"x")
        backend.write_bytes("subject_id=s2/dt=2026-07-01/b.jsonl", b"x")
        # list_keys returns keys relative to the prefix.
        all_keys = backend.list_keys("")
        assert len(all_keys) == 3
        assert all(k.startswith("subject_id=") for k in all_keys)
        # list_dirs at root returns the two subject partitions.
        assert set(backend.list_dirs("")) == {"subject_id=s1/", "subject_id=s2/"}
        # list_dirs under s1 returns the two date partitions.
        assert set(backend.list_dirs("subject_id=s1/")) == {
            "dt=2026-07-01/", "dt=2026-07-02/"
        }

    def test_storage_over_s3_fake_backend(self):
        client = self._fake_s3()
        # The S3 prefix is the sole top-level here (namespace="") so backend
        # prefix and storage namespace do not double up.
        backend = S3Backend("mybucket", prefix="traces", client=client)
        store = TraceStorage(backend, namespace="", granularity="day")
        store.write([_trace("r1", "s1", ts="2026-07-01T00:00:00Z")])
        out = store.query(subject_id="s1")
        assert [r["rollout_id"] for r in out] == ["r1"]

    def test_local_backend_is_storage_backend(self, tmp_path):
        assert isinstance(LocalBackend(tmp_path), StorageBackend)

    def test_gcs_backend_construct(self):
        # GCSBackend should be constructable with an injected client (no SDK).
        class FakeBucket:
            pass

        class FakeClient:
            def bucket(self, name):
                return FakeBucket()

        backend = GCSBackend("bucket", prefix="traces", client=FakeClient())
        assert backend.bucket_name == "bucket"
        assert isinstance(backend, object)


# ---------------------------------------------------------------------------
# End-to-end: retrieval → historical trust
# ---------------------------------------------------------------------------

class TestStorageTrustLoop:
    def test_query_feeds_historical_trust(self, store):
        store.write([
            _trace("r1", "s1", ts="2026-06-01T00:00:00Z", status="pass", score=1),
            _trace("r2", "s1", ts="2026-06-02T00:00:00Z", status="pass", score=1),
            _trace("r3", "s1", ts="2026-06-03T00:00:00Z", status="fail", score=0),
        ])
        from evomerge.trust_score import AgentTrustScoreBuilder

        traces = store.query(subject_id="s1", start="2026-06-01T00:00:00Z", end="2026-06-30T00:00:00Z")
        score = AgentTrustScoreBuilder().add_historical_traces(traces).build()
        # 2 pass + 1 fail → consistency 2/3.
        assert abs(score.breakdown["historical_consistency"] - 2 / 3) < 1e-9


def test_open_trace_storage_factory(tmp_path):
    s = open_trace_storage(tmp_path, codec="jsonl", granularity="day")
    assert isinstance(s, TraceStorage)
    s.write([_trace("r1", ts="2026-07-01T00:00:00Z")])
    assert s.count() == 1
