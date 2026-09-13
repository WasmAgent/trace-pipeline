"""Hostile tests for trusted tenant routing (T01–T07) and storage-byte quota
wiring (Q01–Q06).

Invariants under test:
  data-plane claim != trusted routing authority
  unknown / unauthenticated ingest must be deniable (strict mode)
  storage-byte quota is measured over actual encoded payload bytes
  failed batch admission never permanently consumes quota
"""
from __future__ import annotations

import json

import pytest

from evomerge.multi_tenant import (
    AuditLogger,
    QuotaEnforcer,
    QuotaExceededError,
    QuotaPolicy,
    TenantClaimMismatchError,
    TenantConfig,
    TenantContext,
    TenantIsolationManager,
    TenantRouter,
)


def _record(**overrides):
    base = {
        "schema_version": "aep/v0.5",
        "run_id": "run-1",
        "created_at_ms": 1_700_000_000_000,
        "subject_id": "subj-a",
    }
    base.update(overrides)
    return base


def _manager(**kwargs) -> TenantIsolationManager:
    return TenantIsolationManager(**kwargs)


CTX_A = TenantContext(tenant_id="tenant-a", actor_id="worker-7", source="mTLS")


# ---------------------------------------------------------------------------
# Trusted tenant routing — T01–T07
# ---------------------------------------------------------------------------


class TestTrustedTenantRouting:
    def test_t01_authenticated_a_with_record_tenant_id_b_is_denied(self):
        mgr = _manager()
        recs = [_record(tenant_id="tenant-b")]
        with pytest.raises(TenantClaimMismatchError):
            mgr.ingest(recs, tenant_context=CTX_A)

    def test_t02_authenticated_a_with_organization_id_b_is_denied(self):
        mgr = _manager()
        recs = [_record(organization_id="tenant-b")]
        with pytest.raises(TenantClaimMismatchError):
            mgr.ingest(recs, tenant_context=CTX_A)

    def test_t03_authenticated_a_with_run_context_org_b_is_denied(self):
        mgr = _manager()
        recs = [_record(run_context={"org": "tenant-b"})]
        with pytest.raises(TenantClaimMismatchError):
            mgr.ingest(recs, tenant_context=CTX_A)

    def test_t04_strict_mode_without_trusted_context_is_denied(self):
        mgr = _manager(require_trusted_context=True)
        with pytest.raises(TenantClaimMismatchError):
            mgr.ingest([_record()])

    def test_t04b_non_strict_mode_without_context_keeps_legacy_routing(self):
        # Backward compatibility: deployments that have not enabled strict
        # trusted routing keep the historical (untrusted) resolution order.
        mgr = _manager()
        admitted = mgr.ingest([_record(tenant_id="tenant-b")])
        assert "tenant-b" in admitted

    def test_t05_omitted_record_claim_routes_by_trusted_context(self):
        mgr = _manager()
        recs = [_record()]  # no tenant fields at all
        admitted = mgr.ingest(recs, tenant_context=CTX_A)
        assert list(admitted.keys()) == ["tenant-a"]

    def test_t06_subject_prefix_cannot_switch_tenant(self):
        cfg_b = TenantConfig(
            tenant_id="tenant-b",
            subject_id_prefixes=["subj-"],  # matches the record's subject
        )
        mgr = _manager(router=TenantRouter([cfg_b]))
        admitted = mgr.ingest([_record(subject_id="subj-a")], tenant_context=CTX_A)
        # Prefix would have routed to tenant-b in legacy mode; the trusted
        # context wins.
        assert list(admitted.keys()) == ["tenant-a"]

    def test_t07_mismatch_is_audited_with_authenticated_actor(self):
        logger = AuditLogger()
        mgr = _manager(audit_logger=logger)
        with pytest.raises(TenantClaimMismatchError):
            mgr.ingest([_record(tenant_id="tenant-b")], tenant_context=CTX_A)
        denied = logger.events(event_type="access_denied")
        assert denied, "mismatch must be audited"
        assert denied[0].tenant_id == "tenant-a"
        assert denied[0].actor == "worker-7"
        assert denied[0].detail["claimed_tenant"] == "tenant-b"

    def test_matching_claim_passes(self):
        mgr = _manager()
        admitted = mgr.ingest(
            [_record(tenant_id="tenant-a")], tenant_context=CTX_A
        )
        assert list(admitted.keys()) == ["tenant-a"]


# ---------------------------------------------------------------------------
# Storage-byte quota wiring — Q01–Q06
# ---------------------------------------------------------------------------


def _blob_records(n_bytes_target: int) -> list[dict]:
    """A single record whose encoded JSON size is ~n_bytes_target bytes."""
    payload_len = max(1, n_bytes_target - 200)
    return [_record(run_id="run-blob", blob="x" * payload_len)]


class TestStorageByteQuotaWiring:
    def test_q01_payload_below_limit_is_blocked_when_over(self):
        enforcer = QuotaEnforcer()
        quota = QuotaPolicy(max_storage_bytes=100)
        enforcer.check_and_record("t", quota, n_records=1, n_bytes=60)
        with pytest.raises(QuotaExceededError) as exc:
            enforcer.check_and_record("t", quota, n_records=1, n_bytes=60)
        assert exc.value.resource == "storage_bytes"

    def test_q02_exactly_at_limit_is_accepted(self):
        enforcer = QuotaEnforcer()
        quota = QuotaPolicy(max_storage_bytes=100)
        enforcer.check_and_record("t", quota, n_records=1, n_bytes=100)
        assert enforcer.usage("t")["storage_bytes"] == 100

    def test_q03_limit_plus_one_byte_is_blocked(self):
        enforcer = QuotaEnforcer()
        quota = QuotaPolicy(max_storage_bytes=100)
        enforcer.check_and_record("t", quota, n_records=1, n_bytes=100)
        with pytest.raises(QuotaExceededError):
            enforcer.check_and_record("t", quota, n_records=1, n_bytes=1)

    def test_q04_failed_batch_refunds_already_charged_tenants(self):
        mgr = _manager(
            router=TenantRouter([
                TenantConfig(tenant_id="tenant-a"),
                TenantConfig(
                    tenant_id="tenant-b",
                    quota=QuotaPolicy(max_records_per_day=1),
                ),
            ])
        )
        recs_a = [_record(tenant_id=None, subject_id="a1")]
        recs_b = [_record(tenant_id=None, subject_id="b1")]
        # Route everything to explicit tenants via distinct record claims is
        # unavailable without a context; instead drive the batch through two
        # tenants by relying on registered prefix configs.
        mgr2 = _manager(
            router=TenantRouter([
                TenantConfig(
                    tenant_id="tenant-a",
                    quota=QuotaPolicy(max_records_per_day=10),
                    subject_id_prefixes=["aaa-"],
                ),
                TenantConfig(
                    tenant_id="tenant-b",
                    quota=QuotaPolicy(max_records_per_day=1),
                    subject_id_prefixes=["bbb-"],
                ),
            ])
        )
        batch = [_record(subject_id="aaa-1"), _record(subject_id="bbb-1"), _record(subject_id="bbb-2")]
        with pytest.raises(QuotaExceededError):
            mgr2.ingest(batch)
        # tenant-a was charged first; the failed batch must have refunded it.
        usage_a = mgr2.enforcer.usage("tenant-a")
        assert usage_a["records_today"] == 0, "failed batch must not permanently consume quota"

    def test_q05_concurrent_ingest_cannot_oversubscribe_bytes(self):
        import threading

        enforcer = QuotaEnforcer()
        quota = QuotaPolicy(max_storage_bytes=1_000)
        blocked = []
        lock = threading.Lock()

        def worker():
            for _ in range(50):
                try:
                    enforcer.check_and_record("t", quota, n_records=1, n_bytes=30)
                except QuotaExceededError:
                    with lock:
                        blocked.append(True)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Total charged bytes must never exceed the limit.
        assert enforcer.usage("t")["storage_bytes"] <= 1_000
        assert blocked, "over-subscribing writes must be blocked"

    def test_q06_ingest_accounts_encoded_bytes_not_object_size(self):
        mgr = _manager()
        recs = [_record(blob="y" * 5_000)]
        mgr.ingest(recs, tenant_context=CTX_A)
        expected = len(json.dumps(recs, ensure_ascii=False, default=str).encode("utf-8"))
        usage = mgr.enforcer.usage("tenant-a")
        assert usage["storage_bytes"] == expected
        assert usage["storage_bytes"] > 0
