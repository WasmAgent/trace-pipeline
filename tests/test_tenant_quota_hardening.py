"""Hostile tests for trusted tenant routing (T01–T07) and storage-byte quota
wiring (Q01–Q06).

Invariants under test:
  data-plane claim != trusted routing authority
  unknown / unauthenticated ingest must be deniable (strict mode)
  storage-byte quota is measured over actual encoded payload bytes
  failed batch admission never permanently consumes quota
"""
from __future__ import annotations

import hashlib
import json

import pytest

from evomerge.multi_tenant import (
    AuditEvent,
    AuditLogger,
    QuotaCharge,
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
        # T-S05: legacy routing survives ONLY with the explicit opt-out
        # (UNTRUSTED COMPATIBILITY MODE) — never via the default constructor.
        mgr = _manager(require_trusted_context=False)
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

    def test_t_r06_denied_batch_consumes_no_quota(self):
        # T-R06: a denied (mismatched) batch must leave the enforcer's
        # counters untouched — no record quota, no byte quota.
        mgr = _manager()
        mgr.enforcer.check_and_record(
            "tenant-a", QuotaPolicy(), n_records=3, n_bytes=700
        )
        before = mgr.enforcer.usage("tenant-a")
        with pytest.raises(TenantClaimMismatchError):
            mgr.ingest([_record(tenant_id="tenant-b")], tenant_context=CTX_A)
        assert mgr.enforcer.usage("tenant-a") == before

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
            ]),
            require_trusted_context=False,
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
            ]),
            require_trusted_context=False,
        )
        batch = [_record(subject_id="aaa-1"), _record(subject_id="bbb-1"), _record(subject_id="bbb-2")]
        with pytest.raises(QuotaExceededError):
            mgr2.ingest(batch)
        # tenant-a was charged first; the failed batch must have refunded it —
        # records AND storage bytes (Q-R01, Q-R02).
        usage_a = mgr2.enforcer.usage("tenant-a")
        assert usage_a["records_today"] == 0, "failed batch must not permanently consume record quota"
        assert usage_a["storage_bytes"] == 0, "failed batch must not leave phantom storage-byte usage"

    def test_q04b_rollback_restores_pre_call_usage_not_merely_zero(self):
        """Q-R03/Q-R06: from a NON-ZERO baseline, a failed multi-tenant batch
        must restore tenant A's counters to exactly their pre-call values and
        never drive any counter negative."""
        mgr = _manager(
            router=TenantRouter([
                TenantConfig(
                    tenant_id="tenant-a",
                    quota=QuotaPolicy(max_records_per_day=100, max_storage_bytes=1_000_000),
                    subject_id_prefixes=["aaa-"],
                ),
                TenantConfig(
                    tenant_id="tenant-b",
                    quota=QuotaPolicy(max_records_per_day=1),
                    subject_id_prefixes=["bbb-"],
                ),
            ]),
            require_trusted_context=False,
        )
        # Non-zero baseline: tenant A already has prior admitted traffic.
        prior = [_record(subject_id="aaa-prior", blob="z" * 500)]
        mgr.ingest(prior)
        before = mgr.enforcer.usage("tenant-a")
        assert before["records_today"] == 1
        assert before["storage_bytes"] > 0

        # Batch: tenant A carries a heavy payload; tenant B breaches records/day.
        batch = [
            _record(subject_id="aaa-new", blob="q" * 2_000),
            _record(subject_id="bbb-1"),
            _record(subject_id="bbb-2"),
        ]
        with pytest.raises(QuotaExceededError):
            mgr.ingest(batch)

        after = mgr.enforcer.usage("tenant-a")
        assert after["records_today"] == before["records_today"], (
            "rollback must restore the record quota to the pre-call value"
        )
        assert after["storage_bytes"] == before["storage_bytes"], (
            "rollback must restore the byte quota to the pre-call value — "
            "no phantom bytes may survive a failed batch"
        )
        assert after["records_today"] >= 0 and after["storage_bytes"] >= 0

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
        expected = len(json.dumps(
            recs, ensure_ascii=False, default=str, separators=(",", ":"), sort_keys=True
        ).encode("utf-8"))
        usage = mgr.enforcer.usage("tenant-a")
        assert usage["storage_bytes"] == expected
        assert usage["storage_bytes"] > 0


# ---------------------------------------------------------------------------
# Subject-quota transactionality (QS01–QS05)
# ---------------------------------------------------------------------------


class TestSubjectQuotaRollback:
    """Failed multi-tenant batches must not consume subject quota: rollback
    decrements references instead of discarding subjects, so a poisoned batch
    can never lock a tenant out of its own future subjects."""

    def _two_tenant_manager(self, max_subjects_a: int = 1):
        return _manager(
            router=TenantRouter([
                TenantConfig(
                    tenant_id="tenant-a",
                    quota=QuotaPolicy(max_subjects=max_subjects_a),
                    subject_id_prefixes=["aaa-"],
                ),
                TenantConfig(
                    tenant_id="tenant-b",
                    quota=QuotaPolicy(max_records_per_day=1),
                    subject_id_prefixes=["bbb-"],
                ),
            ]),
            require_trusted_context=False,
        )

    def test_qs01_failed_batch_does_not_consume_subject_quota(self):
        mgr = self._two_tenant_manager(max_subjects_a=1)
        before = mgr.enforcer.usage("tenant-a")
        batch = [
            _record(subject_id="aaa-new"),
            _record(subject_id="bbb-1"),
            _record(subject_id="bbb-2"),
        ]
        with pytest.raises(QuotaExceededError):
            mgr.ingest(batch)
        assert mgr.enforcer.usage("tenant-a")["n_subjects"] == before["n_subjects"]

    def test_qs02_nonzero_baseline_restored_exactly(self):
        mgr = self._two_tenant_manager(max_subjects_a=5)
        mgr.ingest([_record(subject_id="baseline-subject")])
        before = mgr.enforcer.usage("tenant-a")

        with pytest.raises(QuotaExceededError):
            mgr.ingest([
                _record(subject_id="temporary-subject"),
                _record(subject_id="bbb-1"),
                _record(subject_id="bbb-2"),
            ])

        assert mgr.enforcer.usage("tenant-a") == before

    def test_qs03_concurrent_same_subject_commit_survives_rollback(self):
        # Transaction 2 commits subject X; transaction 1 (also X) rolls back.
        # X must survive with refcount 1 — the rollback erases only its own
        # reference, never a concurrent transaction's committed state.
        mgr = _manager(
            router=TenantRouter([TenantConfig(tenant_id="tenant-a", subject_id_prefixes=["aa-"])]),
            require_trusted_context=False,
        )
        quota = QuotaPolicy(max_subjects=1)

        # Two charges on the same subject (refcount 2).
        c1 = mgr.enforcer.check_and_record("tenant-a", quota, n_records=1, subject_ids=["aa-x"])
        c2 = mgr.enforcer.check_and_record("tenant-a", quota, n_records=1, subject_ids=["aa-x"])
        assert mgr.enforcer.usage("tenant-a")["n_subjects"] == 1

        # Roll back ONLY the first charge: the second still holds a reference.
        mgr.enforcer.refund(c1)
        assert mgr.enforcer.usage("tenant-a")["n_subjects"] == 1
        # Roll back the second: now the subject is gone.
        mgr.enforcer.refund(c2)
        assert mgr.enforcer.usage("tenant-a")["n_subjects"] == 0

    def test_qs04_rollback_does_not_erase_other_subject(self):
        # Transaction 1: subject X, later rolled back. Transaction 2: subject
        # Y, committed. End state: X absent, Y present.
        mgr = _manager(
            router=TenantRouter([TenantConfig(tenant_id="tenant-a", subject_id_prefixes=["aa-"])]),
            require_trusted_context=False,
        )
        quota = QuotaPolicy(max_subjects=5)
        c_x = mgr.enforcer.check_and_record("tenant-a", quota, n_records=1, subject_ids=["aa-x"])
        mgr.enforcer.check_and_record("tenant-a", quota, n_records=1, subject_ids=["aa-y"])
        mgr.enforcer.refund(c_x)
        usage = mgr.enforcer.usage("tenant-a")
        assert usage["n_subjects"] == 1

    def test_qs05_poisoning_loop_leaves_no_growth(self):
        # 100 failed batches, each introducing a fresh subject: the subject
        # counter must not grow, and a legitimate later ingest must still be
        # admitted.
        mgr = self._two_tenant_manager(max_subjects_a=3)
        for i in range(100):
            with pytest.raises(QuotaExceededError):
                mgr.ingest([
                    _record(subject_id=f"aaa-poison-{i}"),
                    _record(subject_id="bbb-1"),
                    _record(subject_id="bbb-2"),
                ])
        assert mgr.enforcer.usage("tenant-a")["n_subjects"] == 0

        admitted = mgr.ingest([_record(subject_id="aaa-real")])
        assert "tenant-a" in admitted


# ---------------------------------------------------------------------------
# QuotaCharge one-shot refund safety (QR01–QR05)
# ---------------------------------------------------------------------------


class TestOneShotRefund:
    """A QuotaCharge is refundable exactly once. Double refunds and forged
    charges raise RuntimeError instead of silently over-decrementing."""

    def _enforcer(self):
        return QuotaEnforcer()

    def test_qr01_same_charge_cannot_be_refunded_twice(self):
        enforcer = self._enforcer()
        quota = QuotaPolicy()
        charge = enforcer.check_and_record("t", quota, n_records=2, n_bytes=50, subject_ids=["s1"])
        enforcer.refund(charge)
        with pytest.raises(RuntimeError):
            enforcer.refund(charge)
        # Second attempt leaves all state unchanged.
        assert enforcer.usage("t") == {
            "tenant_id": "t",
            "records_today": 0,
            "storage_bytes": 0,
            "n_subjects": 0,
        }

    def test_qr02_double_refund_cannot_erase_another_transactions_subject(self):
        enforcer = self._enforcer()
        quota = QuotaPolicy(max_subjects=5)
        a = enforcer.check_and_record("t", quota, n_records=1, subject_ids=["x"])
        b = enforcer.check_and_record("t", quota, n_records=1, subject_ids=["x"])
        assert enforcer.usage("t")["n_subjects"] == 1  # refcount 2 on "x"

        enforcer.refund(a)
        assert enforcer.usage("t")["n_subjects"] == 1
        with pytest.raises(RuntimeError):
            enforcer.refund(a)  # double refund — refused
        # "x" must still be present: B's reference survives.
        assert enforcer.usage("t")["n_subjects"] == 1
        enforcer.refund(b)
        assert enforcer.usage("t")["n_subjects"] == 0

    def test_qr03_forged_charge_rejected_fail_closed(self):
        enforcer = self._enforcer()
        quota = QuotaPolicy()
        enforcer.check_and_record("t", quota, n_records=1, n_bytes=10, subject_ids=["s1"])
        before = enforcer.usage("t")

        forged = QuotaCharge(
            charge_id="fabricated-id-not-issued-by-this-enforcer",
            tenant_id="t",
            n_records=99,
            n_bytes=99_999,
            subjects=frozenset(["s1"]),
        )
        with pytest.raises(RuntimeError):
            enforcer.refund(forged)
        assert enforcer.usage("t") == before

    def test_qr04_existing_rollback_semantics_remain(self):
        # charge/rollback round trip still restores records, bytes and subjects.
        enforcer = self._enforcer()
        quota = QuotaPolicy()
        charge = enforcer.check_and_record(
            "t", quota, n_records=3, n_bytes=120, subject_ids=["s1", "s2"]
        )
        assert enforcer.usage("t")["records_today"] == 3
        enforcer.refund(charge)
        assert enforcer.usage("t")["records_today"] == 0
        assert enforcer.usage("t")["storage_bytes"] == 0
        assert enforcer.usage("t")["n_subjects"] == 0

    def test_qr05_concurrent_refund_exactly_one_succeeds(self):
        import threading

        enforcer = self._enforcer()
        quota = QuotaPolicy()
        charge = enforcer.check_and_record("t", quota, n_records=1, n_bytes=10, subject_ids=["x"])
        outcomes = {"ok": 0, "raised": 0}
        lock = threading.Lock()

        def worker():
            try:
                enforcer.refund(charge)
                with lock:
                    outcomes["ok"] += 1
            except RuntimeError:
                with lock:
                    outcomes["raised"] += 1

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert outcomes == {"ok": 1, "raised": 1}
        usage = enforcer.usage("t")
        assert usage["records_today"] == 0
        assert usage["n_subjects"] == 0


# ---------------------------------------------------------------------------
# QuotaCharge lifecycle finalization (QC01–QC08)
# ---------------------------------------------------------------------------


class TestQuotaChargeLifecycle:
    """Every charge has exactly one terminal outcome — REFUNDED or COMMITTED.
    All invalid transitions (and forged reconstructions) fail closed."""

    def _enforcer(self):
        return QuotaEnforcer()

    def _charge(self, enforcer, **kwargs):
        return enforcer.check_and_record(
            "t", QuotaPolicy(), n_records=1, n_bytes=10, subject_ids=["x"], **kwargs
        )

    def test_qc01_committed_charge_cannot_later_be_refunded(self):
        enforcer = self._enforcer()
        charge = self._charge(enforcer)
        enforcer.commit(charge)
        with pytest.raises(RuntimeError):
            enforcer.refund(charge)
        # Committed state is preserved: the quota stays charged.
        usage = enforcer.usage("t")
        assert usage["records_today"] == 1
        assert usage["storage_bytes"] == 10
        assert usage["n_subjects"] == 1

    def test_qc02_successful_transaction_leaves_no_active_charge(self):
        mgr = _manager()
        before = mgr.enforcer._active_charge_count()
        mgr.ingest([_record()], tenant_context=CTX_A)
        assert mgr.enforcer._active_charge_count() == before

    def test_qc03_repeated_successes_do_not_grow_registry(self):
        enforcer = self._enforcer()
        baseline = enforcer._active_charge_count()
        for i in range(1_000):
            charge = enforcer.check_and_record(
                "t", QuotaPolicy(), n_records=1, subject_ids=[f"s-{i}"]
            )
            enforcer.commit(charge)
        assert enforcer._active_charge_count() == baseline

    def test_qc04_committed_charge_cannot_be_committed_twice(self):
        enforcer = self._enforcer()
        charge = self._charge(enforcer)
        enforcer.commit(charge)
        with pytest.raises(RuntimeError):
            enforcer.commit(charge)

    def test_qc05_refunded_charge_cannot_be_committed(self):
        enforcer = self._enforcer()
        charge = self._charge(enforcer)
        enforcer.refund(charge)
        with pytest.raises(RuntimeError):
            enforcer.commit(charge)

    def test_qc06_unknown_charge_cannot_be_committed(self):
        enforcer = self._enforcer()
        before = enforcer.usage("t")
        forged = QuotaCharge(
            charge_id="not-issued",
            tenant_id="t",
            n_records=1,
            n_bytes=1,
            subjects=frozenset({"x"}),
        )
        with pytest.raises(RuntimeError):
            enforcer.commit(forged)
        assert enforcer.usage("t") == before

    def test_qc07_forged_reconstruction_with_valid_id_is_rejected(self):
        # This is why the registry is dict[str, QuotaCharge] and not set[str]:
        # a reconstructed charge with a REAL id but altered tenant/amount
        # fields must not validate.
        enforcer = self._enforcer()
        real = self._charge(enforcer)
        forged = QuotaCharge(
            charge_id=real.charge_id,
            tenant_id="other-tenant",
            n_records=999,
            n_bytes=999_999,
            subjects=frozenset({"evil"}),
        )
        with pytest.raises(RuntimeError):
            enforcer.refund(forged)
        with pytest.raises(RuntimeError):
            enforcer.commit(forged)
        # The real charge is untouched and still live.
        enforcer.refund(real)
        assert enforcer.usage("t")["n_subjects"] == 0

    def test_qc08_concurrent_commit_refund_race_exactly_one_winner(self):
        import threading

        enforcer = self._enforcer()
        charge = self._charge(enforcer)
        before = enforcer.usage("t")
        outcomes = {"commit": 0, "refund": 0, "raised": 0}
        lock = threading.Lock()

        def committer():
            try:
                enforcer.commit(charge)
                with lock:
                    outcomes["commit"] += 1
            except RuntimeError:
                with lock:
                    outcomes["raised"] += 1

        def refunder():
            try:
                enforcer.refund(charge)
                with lock:
                    outcomes["refund"] += 1
            except RuntimeError:
                with lock:
                    outcomes["raised"] += 1

        t1, t2 = threading.Thread(target=committer), threading.Thread(target=refunder)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert outcomes["raised"] == 1
        assert outcomes["commit"] + outcomes["refund"] == 1
        after = enforcer.usage("t")
        if outcomes["commit"] == 1:
            assert after == before, "commit wins → usage stays charged"
        else:
            assert after["records_today"] == 0, "refund wins → usage restored"

    def test_qc09_reset_refuses_tenant_with_live_charges(self):
        enforcer = self._enforcer()
        self._charge(enforcer)
        with pytest.raises(RuntimeError):
            enforcer.reset("t")


# ---------------------------------------------------------------------------
# Transaction boundary (TX01–TX08) + storage quota semantics (SQ01–SQ03)
# ---------------------------------------------------------------------------


class RecordingAuditLogger(AuditLogger):
    """AuditLogger that records the order in which events were emitted."""

    def __init__(self) -> None:
        super().__init__()
        self.emitted: list[AuditEvent] = []

    def log(self, event: AuditEvent) -> None:
        super().log(event)
        self.emitted.append(event)


class FailingStore:
    def write_batch(self, tenant_id, payload, *, payload_sha256, n_records):
        raise OSError("storage unavailable")


class MemoryStore:
    """§6 interface: writes the EXACT reserved payload bytes, hash-verified."""

    def __init__(self):
        self.stored: dict[str, bytes] = {}
        self.writes = 0

    def write_batch(self, tenant_id, payload, *, payload_sha256, n_records):
        import hashlib
        assert hashlib.sha256(payload).hexdigest() == payload_sha256, (
            "durable bytes must hash to the reserved payload digest"
        )
        self.stored[tenant_id] = bytes(payload)
        self.writes += 1


def _tx_manager(audit_logger=None, **kwargs):
    return _manager(audit_logger=audit_logger or RecordingAuditLogger(), **kwargs)


def _two_tenant_tx_manager(audit_logger=None):
    return _manager(
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
        ]),
        audit_logger=audit_logger or RecordingAuditLogger(),
        require_trusted_context=False,
    )


class TestTransactionBoundary:
    def test_tx01_prior_tenant_success_audit_does_not_survive_later_quota_failure(self):
        logger = RecordingAuditLogger()
        mgr = _two_tenant_tx_manager(audit_logger=logger)
        before_active = mgr.enforcer._active_charge_count()
        before_a = mgr.enforcer.usage("tenant-a")

        with pytest.raises(QuotaExceededError):
            mgr.ingest([
                _record(subject_id="aaa-1"),
                _record(subject_id="bbb-1"),
                _record(subject_id="bbb-2"),
            ])

        # A quota restored, B not admitted, active count back to baseline.
        assert mgr.enforcer.usage("tenant-a") == before_a
        assert mgr.enforcer._active_charge_count() == before_active
        # Zero ingest-success audits: the batch rolled back (TX01/P1-B).
        success = [e for e in logger.emitted if e.event_type == "ingest" and e.outcome == "success"]
        assert success == []

    def test_tx02_post_charge_injected_failure_rolls_back_reservation(self):
        # A custom enforcer whose check_and_record raises OSError after the
        # first tenant has been charged (post-charge pre-commit path).
        logger = RecordingAuditLogger()
        mgr = _two_tenant_tx_manager(audit_logger=logger)
        real = mgr.enforcer.check_and_record

        state = {"charged": 0}

        def flaky(tenant_id, policy, n_records, subject_ids=None, n_bytes=0, payload_sha256=None):
            if state["charged"] >= 1:
                raise OSError("disk full")
            state["charged"] += 1
            return real(tenant_id, policy, n_records=n_records, subject_ids=subject_ids, n_bytes=n_bytes, payload_sha256=payload_sha256)

        mgr.enforcer.check_and_record = flaky
        with pytest.raises(OSError):
            mgr.ingest([
                _record(subject_id="aaa-1"),
                _record(subject_id="bbb-1"),
            ])

        usage = mgr.enforcer.usage("tenant-a")
        assert usage["records_today"] == 0
        assert usage["storage_bytes"] == 0
        assert usage["n_subjects"] == 0
        assert mgr.enforcer._active_charge_count() == 0

    def test_tx03_quota_exceeded_audit_failure_does_not_strand_prior_charges(self):
        # A passes; B quota fails; the blocked-audit sink ALSO fails.
        # A's quota must still be restored and the registry must return to
        # baseline; the surfaced error may be the audit wrapper.
        class BrokenAudit(AuditLogger):
            def log(self, event):
                if event.event_type == "quota_exceeded":
                    raise OSError("audit sink down")
                super().log(event)

        audit = BrokenAudit()
        mgr = _two_tenant_tx_manager(audit_logger=audit)
        before_a = mgr.enforcer.usage("tenant-a")

        with pytest.raises((QuotaExceededError, RuntimeError)):
            mgr.ingest([
                _record(subject_id="aaa-1"),
                _record(subject_id="bbb-1"),
                _record(subject_id="bbb-2"),
            ])
        assert mgr.enforcer.usage("tenant-a") == before_a
        assert mgr.enforcer._active_charge_count() == 0

    def test_tx04_durable_write_failure_triggers_rollback(self):
        logger = RecordingAuditLogger()
        mgr = _tx_manager(audit_logger=logger)
        before = mgr.enforcer.usage("tenant-a")
        reservation = mgr.reserve_ingest([_record()], tenant_context=CTX_A)
        with pytest.raises(OSError):
            mgr.ingest_to_store([_record()], FailingStore(), tenant_context=CTX_A)
        # The failed ingest_to_store rolled back its own reservation; the
        # earlier reservation, rolled back manually, also restored quota.
        mgr.rollback(reservation)
        assert mgr.enforcer.usage("tenant-a") == before
        success = [e for e in logger.emitted if e.event_type == "ingest" and e.outcome == "success"]
        assert success == []

    def test_tx05_durable_write_success_commits_quota(self):
        logger = RecordingAuditLogger()
        mgr = _tx_manager(audit_logger=logger)
        store = MemoryStore()
        result = mgr.ingest_to_store([_record()], store, tenant_context=CTX_A)
        assert result.committed is True
        assert result.audit_status == "delivered"
        assert store.writes == 1
        assert result.batches[0].tenant_id == "tenant-a"
        usage = mgr.enforcer.usage("tenant-a")
        assert usage["records_today"] == 1
        assert mgr.enforcer._active_charge_count() == 0
        success = [e for e in logger.emitted if e.event_type == "ingest" and e.outcome == "success"]
        assert len(success) == 1

    def test_tx06_committed_charges_cannot_be_refunded_after_success(self):
        mgr = _tx_manager()
        reservation = mgr.reserve_ingest([_record()], tenant_context=CTX_A)
        store = MemoryStore()
        for batch in reservation.batches:
            store.write_batch(
                batch.tenant_id, batch.payload,
                payload_sha256=batch.payload_sha256, n_records=batch.n_records,
            )
        mgr.commit(reservation)
        for charge in reservation.charges:
            with pytest.raises(RuntimeError):
                mgr.enforcer.refund(charge)
        usage = mgr.enforcer.usage("tenant-a")
        assert usage["records_today"] == 1

    def test_tx07_abandoned_reservation_is_visible_in_registry(self):
        mgr = _tx_manager()
        before = mgr.enforcer._active_charge_count()
        reservation = mgr.reserve_ingest([_record()], tenant_context=CTX_A)
        assert mgr.enforcer._active_charge_count() == before + 1
        # No auto-GC: the abandoned reservation stays visible until resolved.
        assert mgr.enforcer._active_charge_count() > before
        mgr.rollback(reservation)
        assert mgr.enforcer._active_charge_count() == before

    def test_tx08_multi_tenant_success_emits_one_audit_per_tenant_after_storage(self):
        logger = RecordingAuditLogger()
        mgr = _two_tenant_tx_manager(audit_logger=logger)
        store = MemoryStore()
        result = mgr.ingest_to_store([
            _record(subject_id="aaa-1"),
            _record(subject_id="bbb-1"),
        ], store, tenant_context=None)
        assert {b.tenant_id for b in result.batches} == {"tenant-a", "tenant-b"}
        success = [e for e in logger.emitted if e.event_type == "ingest" and e.outcome == "success"]
        assert len(success) == 2
        assert {e.tenant_id for e in success} == {"tenant-a", "tenant-b"}


class TestStorageQuotaSemantics:
    def _mgr_with_limit(self, audit_logger=None):
        return _manager(
            router=TenantRouter([
                TenantConfig(
                    tenant_id="tenant-a",
                    quota=QuotaPolicy(max_storage_bytes=5_000),
                    subject_id_prefixes=["aaa-"],
                ),
            ]),
            audit_logger=audit_logger or RecordingAuditLogger(),
            require_trusted_context=False,
        )

    def test_sq01_failed_durable_write_does_not_consume_storage_bytes(self):
        logger = RecordingAuditLogger()
        mgr = self._mgr_with_limit(audit_logger=logger)
        before = mgr.enforcer.usage("tenant-a")
        with pytest.raises(OSError):
            mgr.ingest_to_store(
                [_record(subject_id="aaa-1", blob="x" * 1_000)],
                FailingStore(),
                tenant_context=None,
            )
        after = mgr.enforcer.usage("tenant-a")
        assert after["storage_bytes"] == before["storage_bytes"]

    def test_sq02_successful_durable_write_consumes_exactly_encoded_bytes(self):
        logger = RecordingAuditLogger()
        mgr = self._mgr_with_limit(audit_logger=logger)
        recs = [_record(subject_id="aaa-1", blob="x" * 1_000)]
        store = MemoryStore()
        mgr.ingest_to_store(recs, store, tenant_context=None)
        expected = len(json.dumps(
            recs, ensure_ascii=False, default=str, separators=(",", ":"), sort_keys=True
        ).encode("utf-8"))
        assert mgr.enforcer.usage("tenant-a")["storage_bytes"] == expected

    def test_sq03_second_write_respects_committed_first_write(self):
        mgr = self._mgr_with_limit()
        store = MemoryStore()
        mgr.ingest_to_store(
            [_record(subject_id="aaa-1", blob="x" * 2_000)], store, tenant_context=None
        )
        committed = mgr.enforcer.usage("tenant-a")["storage_bytes"]
        assert committed > 0
        # A second batch that would exceed the limit is denied; the first
        # committed usage remains intact.
        with pytest.raises(QuotaExceededError):
            mgr.ingest_to_store(
                [_record(subject_id="aaa-2", blob="x" * 4_000)], store, tenant_context=None
            )
        assert mgr.enforcer.usage("tenant-a")["storage_bytes"] == committed


# ---------------------------------------------------------------------------
# Reservation integrity (RI01–RI05)
# ---------------------------------------------------------------------------


class TestReservationIntegrity:
    def _manager(self, audit_logger=None):
        return _manager(
            router=TenantRouter([
                TenantConfig(tenant_id="tenant-a", subject_id_prefixes=["aaa-"]),
            ]),
            audit_logger=audit_logger or RecordingAuditLogger(),
            require_trusted_context=False,
        )

    def test_ri01_source_mutation_after_reserve_cannot_change_stored_bytes(self):
        mgr = self._manager()
        store = MemoryStore()
        source = [_record(subject_id="aaa-1", blob="small")]
        reservation = mgr.reserve_ingest(source, tenant_context=None)

        # Hostile caller mutates the ORIGINAL records after validation.
        source[0]["blob"] = "X" * 1_000_000
        store.write_batch(
            reservation.batches[0].tenant_id,
            reservation.batches[0].payload,
            payload_sha256=reservation.batches[0].payload_sha256,
            n_records=reservation.batches[0].n_records,
        )
        assert hashlib.sha256(store.stored["tenant-a"]).hexdigest() == (
            reservation.batches[0].payload_sha256
        )
        # The stored bytes are the RESERVED canonical bytes, not the mutated
        # source: decoded payload still reflects the validated content.
        decoded = json.loads(store.stored["tenant-a"].decode("utf-8"))
        assert decoded[0]["blob"] == "small"

    def test_ri02_tenant_mapping_cannot_be_altered_after_routing(self):
        mgr = self._manager()
        reservation = mgr.reserve_ingest(
            [_record(subject_id="aaa-1")], tenant_context=None
        )
        # Frozen reservation + immutable tuple of frozen batches: injection of
        # a different tenant is structurally impossible.
        with pytest.raises((TypeError, AttributeError)):
            reservation.batches[0].tenant_id = "tenant-b"  # type: ignore[misc]
        with pytest.raises(AttributeError):  # FrozenInstanceError
            reservation.batches = ()  # type: ignore[misc]
        assert reservation.batches[0].tenant_id == "tenant-a"

    def test_ri03_quota_bytes_equal_reserved_payload_bytes(self):
        mgr = self._manager()
        reservation = mgr.reserve_ingest(
            [_record(subject_id="aaa-1", blob="y" * 2_000)], tenant_context=None
        )
        batch = reservation.batches[0]
        charge = reservation.charges[0]
        assert charge.tenant_id == batch.tenant_id
        assert charge.n_bytes == batch.n_bytes
        assert batch.n_bytes == len(batch.payload)
        assert charge.payload_sha256 == batch.payload_sha256

    def test_ri04_payload_digest_matches_durable_bytes(self):
        mgr = self._manager()
        store = MemoryStore()
        reservation = mgr.reserve_ingest(
            [_record(subject_id="aaa-1", blob="z" * 500)], tenant_context=None
        )
        for batch in reservation.batches:
            store.write_batch(
                batch.tenant_id, batch.payload,
                payload_sha256=batch.payload_sha256, n_records=batch.n_records,
            )
        stored = store.stored["tenant-a"]
        assert hashlib.sha256(stored).hexdigest() == reservation.batches[0].payload_sha256

    def test_ri05_nested_mutation_after_reserve_has_zero_effect(self):
        mgr = self._manager()
        store = MemoryStore()
        source = [_record(subject_id="aaa-1")]
        source[0]["run_context"] = {"org": "tenant-a", "nested": {"k": [1, 2]}}
        reservation = mgr.reserve_ingest(source, tenant_context=None)
        original_sha = reservation.batches[0].payload_sha256
        original_payload = bytes(reservation.batches[0].payload)

        # Mutate the source deeply AFTER validation.
        source[0]["run_context"]["nested"]["k"].append(999)
        source[0]["run_context"]["org"] = "tenant-b"
        source[0]["metadata"] = {"injected": True}

        store.write_batch(
            reservation.batches[0].tenant_id,
            reservation.batches[0].payload,
            payload_sha256=reservation.batches[0].payload_sha256,
            n_records=reservation.batches[0].n_records,
        )
        assert store.stored["tenant-a"] == original_payload
        assert original_sha == hashlib.sha256(original_payload).hexdigest()


# ---------------------------------------------------------------------------
# Atomic reservation settlement (RT01–RT07)
# ---------------------------------------------------------------------------


class TestAtomicReservationSettlement:
    def _two_tenant_mgr(self):
        return _manager(
            router=TenantRouter([
                TenantConfig(tenant_id="tenant-a", subject_id_prefixes=["aaa-"]),
                TenantConfig(tenant_id="tenant-b", subject_id_prefixes=["bbb-"]),
            ]),
            require_trusted_context=False,
        )

    def _reservation(self, mgr):
        return mgr.reserve_ingest([
            _record(subject_id="aaa-1"),
            _record(subject_id="bbb-1"),
        ], tenant_context=None)

    def test_rt01_commit_vs_rollback_race_whole_reservation_winner(self):
        import threading

        mgr = self._two_tenant_mgr()
        before = {
            "a": mgr.enforcer.usage("tenant-a"),
            "b": mgr.enforcer.usage("tenant-b"),
        }
        reservation = self._reservation(mgr)
        outcomes = {"commit": 0, "rollback": 0, "raised": 0}
        lock = threading.Lock()

        def committer():
            try:
                mgr.commit(reservation)
                with lock:
                    outcomes["commit"] += 1
            except RuntimeError:
                with lock:
                    outcomes["raised"] += 1

        def rollbacker():
            try:
                mgr.rollback(reservation)
                with lock:
                    outcomes["rollback"] += 1
            except RuntimeError:
                with lock:
                    outcomes["raised"] += 1

        t1, t2 = threading.Thread(target=committer), threading.Thread(target=rollbacker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one terminal transition won the race...
        assert outcomes["raised"] == 1
        assert outcomes["commit"] + outcomes["rollback"] == 1
        # ...and the final state is WHOLE: every charge settled identically.
        if outcomes["commit"] == 1:
            assert mgr.enforcer.usage("tenant-a")["records_today"] == before["a"]["records_today"] + 1
            assert mgr.enforcer.usage("tenant-b")["records_today"] == before["b"]["records_today"] + 1
        else:
            assert mgr.enforcer.usage("tenant-a") == before["a"]
            assert mgr.enforcer.usage("tenant-b") == before["b"]
        assert mgr.enforcer._active_charge_count() == 0

    def test_rt02_invalid_charge_causes_zero_commit_mutation(self):
        enforcer = QuotaEnforcer()
        quota = QuotaPolicy()
        c1 = enforcer.check_and_record("t", quota, n_records=1, subject_ids=["s1"])
        forged = QuotaCharge(
            charge_id=c1.charge_id,
            tenant_id="t",
            n_records=999,  # same id, tampered amounts
            n_bytes=99_999,
            subjects=frozenset({"s1"}),
        )
        with pytest.raises(RuntimeError):
            enforcer.commit_many((c1, forged))
        # Zero mutation: c1 is STILL active (no partial commit).
        assert enforcer.usage("t")["records_today"] == 1
        assert enforcer._active_charge_count() == 1

    def test_rt03_invalid_charge_causes_zero_rollback_mutation(self):
        enforcer = QuotaEnforcer()
        quota = QuotaPolicy()
        c1 = enforcer.check_and_record("t", quota, n_records=1, n_bytes=100, subject_ids=["s1"])
        unknown = QuotaCharge(
            charge_id="never-issued",
            tenant_id="t",
            n_records=1,
            n_bytes=100,
            subjects=frozenset({"s1"}),
        )
        before = enforcer.usage("t")
        with pytest.raises(RuntimeError):
            enforcer.refund_many((c1, unknown))
        assert enforcer.usage("t") == before
        assert enforcer._active_charge_count() == 1

    def test_rt04_double_commit_rejected(self):
        mgr = self._two_tenant_mgr()
        reservation = self._reservation(mgr)
        mgr.commit(reservation)
        with pytest.raises(RuntimeError):
            mgr.commit(reservation)
        assert mgr.enforcer._active_charge_count() == 0

    def test_rt05_double_rollback_rejected(self):
        mgr = self._two_tenant_mgr()
        reservation = self._reservation(mgr)
        mgr.rollback(reservation)
        with pytest.raises(RuntimeError):
            mgr.rollback(reservation)
        assert mgr.enforcer._active_charge_count() == 0

    def test_rt06_committed_reservation_cannot_rollback(self):
        mgr = self._two_tenant_mgr()
        reservation = self._reservation(mgr)
        mgr.commit(reservation)
        committed_usage = (
            mgr.enforcer.usage("tenant-a"),
            mgr.enforcer.usage("tenant-b"),
        )
        with pytest.raises(RuntimeError):
            mgr.rollback(reservation)
        assert (
            mgr.enforcer.usage("tenant-a"),
            mgr.enforcer.usage("tenant-b"),
        ) == committed_usage

    def test_rt07_rolled_back_reservation_cannot_commit(self):
        mgr = self._two_tenant_mgr()
        baseline = (
            mgr.enforcer.usage("tenant-a"),
            mgr.enforcer.usage("tenant-b"),
        )
        reservation = self._reservation(mgr)
        mgr.rollback(reservation)
        with pytest.raises(RuntimeError):
            mgr.commit(reservation)
        assert (
            mgr.enforcer.usage("tenant-a"),
            mgr.enforcer.usage("tenant-b"),
        ) == baseline


# ---------------------------------------------------------------------------
# Post-commit audit semantics (PA01–PA04)
# ---------------------------------------------------------------------------


class FlakyAudit(AuditLogger):
    """AuditLogger whose sink fails on success-event delivery."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_next_ingest_success = False

    def log(self, event: AuditEvent) -> None:
        if (
            self.fail_next_ingest_success
            and event.event_type == "ingest"
            and event.outcome == "success"
        ):
            raise OSError("audit sink down")
        super().log(event)


class TestPostCommitAuditSemantics:
    def test_pa01_audit_failure_after_durable_commit_reports_committed(self):
        audit = FlakyAudit()
        mgr = _manager(audit_logger=audit)
        store = MemoryStore()
        audit.fail_next_ingest_success = True
        result = mgr.ingest_to_store([_record()], store, tenant_context=CTX_A)
        # Durable data present + quota committed; audit failure is REPORTED.
        assert result.committed is True
        assert result.audit_status == "failed"
        assert result.audit_error is not None
        assert store.writes == 1

    def test_pa02_audit_failure_does_not_trigger_rollback(self):
        audit = FlakyAudit()
        mgr = _manager(audit_logger=audit)
        store = MemoryStore()
        audit.fail_next_ingest_success = True
        result = mgr.ingest_to_store([_record()], store, tenant_context=CTX_A)
        usage = mgr.enforcer.usage("tenant-a")
        assert result.committed is True
        assert usage["records_today"] == 1
        assert usage["storage_bytes"] > 0
        assert "tenant-a" in store.stored

    def test_pa03_caller_can_distinguish_retry_safety(self):
        mgr = _manager()
        store = MemoryStore()
        # Pre-commit storage failure: propagates (retry-safe, nothing durable).
        with pytest.raises(OSError):
            mgr.ingest_to_store([_record()], FailingStore(), tenant_context=CTX_A)
        assert store.writes == 0
        # Post-commit audit failure: IngestResult with committed=True — NOT
        # the same undifferentiated exception class.
        audit = FlakyAudit()
        mgr2 = _manager(audit_logger=audit)
        audit.fail_next_ingest_success = True
        result = mgr2.ingest_to_store([_record()], MemoryStore(), tenant_context=CTX_A)
        assert result.committed is True
        assert result.audit_status == "failed"

    def test_pa04_repeated_retry_is_explicitly_preventable(self):
        audit = FlakyAudit()
        mgr = _manager(audit_logger=audit)
        store = MemoryStore()
        audit.fail_next_ingest_success = True
        result = mgr.ingest_to_store([_record()], store, tenant_context=CTX_A)
        assert result.committed is True
        # A caller checking result.committed knows a retry would duplicate
        # durable data — the committed flag makes blind retry preventable.
        assert store.writes == 1


# ---------------------------------------------------------------------------
# Abandoned-reservation observability (AR01–AR04)
# ---------------------------------------------------------------------------


class TestAbandonedReservationObservability:
    def test_ar01_active_reservation_visible_after_reserve(self):
        mgr = _manager()
        before = len(mgr.active_reservations())
        mgr.reserve_ingest([_record()], tenant_context=CTX_A)
        assert len(mgr.active_reservations()) == before + 1

    def test_ar02_commit_removes_reservation_from_active_registry(self):
        mgr = _manager()
        reservation = mgr.reserve_ingest([_record()], tenant_context=CTX_A)
        assert any(
            r["reservation_id"] == reservation.reservation_id
            for r in mgr.active_reservations()
        )
        mgr.commit(reservation)
        assert not any(
            r["reservation_id"] == reservation.reservation_id
            and r["state"] == "active"
            for r in mgr.active_reservations()
        )

    def test_ar03_rollback_removes_reservation_from_active_registry(self):
        mgr = _manager()
        reservation = mgr.reserve_ingest([_record()], tenant_context=CTX_A)
        mgr.rollback(reservation)
        assert not any(
            r["reservation_id"] == reservation.reservation_id
            and r["state"] == "active"
            for r in mgr.active_reservations()
        )

    def test_ar04_old_active_reservation_observable_with_age(self):
        mgr = _manager()
        reservation = mgr.reserve_ingest([_record()], tenant_context=CTX_A)
        active = [
            r for r in mgr.active_reservations()
            if r["reservation_id"] == reservation.reservation_id
        ]
        assert len(active) == 1
        assert active[0]["state"] == "active"
        assert active[0]["age_seconds"] >= 0
