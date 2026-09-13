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
        expected = len(json.dumps(recs, ensure_ascii=False, default=str).encode("utf-8"))
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
