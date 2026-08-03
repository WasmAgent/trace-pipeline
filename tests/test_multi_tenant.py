"""Tests for evomerge.multi_tenant — issue #57.

Covers:
  - TenantConfig: subject_id_prefix matching, storage_namespace default
  - QuotaPolicy: validation
  - QuotaEnforcer: per-tenant counters, limit enforcement, reset
  - TenantRouter: resolution precedence, prefix matching
  - AuditEvent: serialisation round-trip
  - AuditLogger: append, filter, NDJSON export
  - TenantIsolationManager: ingest routing+quota+audit, query audit
"""
from __future__ import annotations

import json

import pytest

from evomerge.multi_tenant import (
    AuditEvent,
    AuditLogger,
    QuotaEnforcer,
    QuotaExceededError,
    QuotaPolicy,
    TenantConfig,
    TenantIsolationManager,
    TenantRouter,
)

# ---------------------------------------------------------------------------
# TenantConfig
# ---------------------------------------------------------------------------

class TestTenantConfig:
    def test_storage_namespace_default(self):
        cfg = TenantConfig(tenant_id="acme")
        assert cfg.storage_namespace == "tenant=acme"

    def test_storage_namespace_override(self):
        cfg = TenantConfig(tenant_id="acme", storage_namespace="org=acme")
        assert cfg.storage_namespace == "org=acme"

    def test_allows_subject_empty_prefixes(self):
        cfg = TenantConfig(tenant_id="t1")
        assert cfg.allows_subject("any-subject")

    def test_allows_subject_with_prefix(self):
        cfg = TenantConfig(tenant_id="t1", subject_id_prefixes=["acme-"])
        assert cfg.allows_subject("acme-agent-1")
        assert not cfg.allows_subject("other-agent")

    def test_allows_subject_multiple_prefixes(self):
        cfg = TenantConfig(tenant_id="t1", subject_id_prefixes=["acme-", "beta-"])
        assert cfg.allows_subject("beta-bot")
        assert cfg.allows_subject("acme-bot")
        assert not cfg.allows_subject("gamma-bot")


# ---------------------------------------------------------------------------
# QuotaPolicy
# ---------------------------------------------------------------------------

class TestQuotaPolicy:
    def test_defaults_unlimited(self):
        policy = QuotaPolicy()
        assert policy.max_records_per_day is None
        assert policy.max_storage_bytes is None
        assert policy.max_subjects is None

    def test_explicit_limits(self):
        policy = QuotaPolicy(max_records_per_day=1000, max_storage_bytes=1_000_000, max_subjects=50)
        assert policy.max_records_per_day == 1000


# ---------------------------------------------------------------------------
# QuotaEnforcer
# ---------------------------------------------------------------------------

class TestQuotaEnforcer:
    def test_no_limit_passes(self):
        enforcer = QuotaEnforcer()
        enforcer.check_and_record("t1", QuotaPolicy(), n_records=9999)

    def test_records_per_day_limit(self):
        enforcer = QuotaEnforcer()
        policy = QuotaPolicy(max_records_per_day=5)
        enforcer.check_and_record("t1", policy, n_records=5)
        with pytest.raises(QuotaExceededError) as exc_info:
            enforcer.check_and_record("t1", policy, n_records=1)
        assert exc_info.value.resource == "records_per_day"
        assert exc_info.value.tenant_id == "t1"

    def test_storage_bytes_limit(self):
        enforcer = QuotaEnforcer()
        policy = QuotaPolicy(max_storage_bytes=100)
        enforcer.check_and_record("t1", policy, n_records=1, n_bytes=80)
        with pytest.raises(QuotaExceededError) as exc_info:
            enforcer.check_and_record("t1", policy, n_records=1, n_bytes=30)
        assert exc_info.value.resource == "storage_bytes"

    def test_subjects_limit(self):
        enforcer = QuotaEnforcer()
        policy = QuotaPolicy(max_subjects=2)
        enforcer.check_and_record("t1", policy, n_records=1, subject_ids=["s1", "s2"])
        with pytest.raises(QuotaExceededError) as exc_info:
            enforcer.check_and_record("t1", policy, n_records=1, subject_ids=["s3"])
        assert exc_info.value.resource == "subjects"

    def test_usage_snapshot(self):
        enforcer = QuotaEnforcer()
        enforcer.check_and_record("t1", QuotaPolicy(), n_records=3, n_bytes=50, subject_ids=["s1"])
        usage = enforcer.usage("t1")
        assert usage["records_today"] == 3
        assert usage["storage_bytes"] == 50
        assert usage["n_subjects"] == 1

    def test_reset(self):
        enforcer = QuotaEnforcer()
        policy = QuotaPolicy(max_records_per_day=5)
        enforcer.check_and_record("t1", policy, n_records=5)
        enforcer.reset("t1")
        # After reset, 5 more records should pass again
        enforcer.check_and_record("t1", policy, n_records=5)

    def test_different_tenants_isolated(self):
        enforcer = QuotaEnforcer()
        policy = QuotaPolicy(max_records_per_day=3)
        enforcer.check_and_record("t1", policy, n_records=3)
        # t2 is independent
        enforcer.check_and_record("t2", policy, n_records=3)
        # t1 should now be at limit
        with pytest.raises(QuotaExceededError):
            enforcer.check_and_record("t1", policy, n_records=1)


# ---------------------------------------------------------------------------
# TenantRouter
# ---------------------------------------------------------------------------

class TestTenantRouter:
    def _router(self):
        configs = [
            TenantConfig("acme", subject_id_prefixes=["acme-"]),
            TenantConfig("beta", subject_id_prefixes=["beta-"]),
        ]
        return TenantRouter(configs, default_tenant="_default")

    def test_explicit_tenant_id_field(self):
        router = self._router()
        rec = {"tenant_id": "acme", "subject_id": "beta-agent"}
        assert router.resolve(rec) == "acme"

    def test_organization_id_field(self):
        router = self._router()
        rec = {"organization_id": "beta"}
        assert router.resolve(rec) == "beta"

    def test_run_context_org(self):
        router = self._router()
        rec = {"run_context": {"org": "acme"}}
        assert router.resolve(rec) == "acme"

    def test_subject_id_prefix_matching(self):
        router = self._router()
        rec = {"subject_id": "acme-bot-1"}
        assert router.resolve(rec) == "acme"

    def test_default_tenant_fallback(self):
        router = self._router()
        rec = {"subject_id": "unknown-agent"}
        assert router.resolve(rec) == "_default"

    def test_config_for(self):
        router = self._router()
        cfg = router.config_for("acme")
        assert cfg is not None
        assert cfg.tenant_id == "acme"

    def test_config_for_unknown(self):
        router = self._router()
        assert router.config_for("nonexistent") is None

    def test_register(self):
        router = TenantRouter([])
        router.register(TenantConfig("new-tenant"))
        assert "new-tenant" in router.all_tenants()


# ---------------------------------------------------------------------------
# AuditEvent
# ---------------------------------------------------------------------------

class TestAuditEvent:
    def test_to_json_round_trip(self):
        event = AuditEvent(
            event_type="ingest",
            tenant_id="acme",
            actor="worker-1",
            resource="traces",
            outcome="success",
            detail={"n_records": 10},
            timestamp=1234567890.0,
        )
        line = event.to_json()
        data = json.loads(line)
        assert data["event_type"] == "ingest"
        assert data["tenant_id"] == "acme"
        assert data["detail"]["n_records"] == 10

    def test_from_json(self):
        event = AuditEvent(
            event_type="query", tenant_id="beta", actor="api",
            resource="traces/subject_id=s1", outcome="success",
            detail={}, timestamp=9999.0,
        )
        restored = AuditEvent.from_json(event.to_json())
        assert restored.event_type == "query"
        assert restored.tenant_id == "beta"
        assert restored.timestamp == 9999.0


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class TestAuditLogger:
    def _event(self, tid="t1", etype="ingest", outcome="success"):
        return AuditEvent(
            event_type=etype, tenant_id=tid, actor="test",
            resource="traces", outcome=outcome,
        )

    def test_log_and_events(self):
        logger = AuditLogger()
        logger.log(self._event())
        assert logger.count == 1

    def test_filter_by_tenant(self):
        logger = AuditLogger()
        logger.log(self._event(tid="t1"))
        logger.log(self._event(tid="t2"))
        assert len(logger.events(tenant_id="t1")) == 1
        assert len(logger.events(tenant_id="t2")) == 1

    def test_filter_by_event_type(self):
        logger = AuditLogger()
        logger.log(self._event(etype="ingest"))
        logger.log(self._event(etype="query"))
        assert len(logger.events(event_type="query")) == 1

    def test_export_ndjson(self):
        logger = AuditLogger()
        for i in range(3):
            logger.log(self._event(tid=f"t{i}"))
        ndjson = logger.export_ndjson()
        lines = ndjson.strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # each line must be valid JSON

    def test_clear(self):
        logger = AuditLogger()
        logger.log(self._event())
        logger.clear()
        assert logger.count == 0

    def test_file_sink(self, tmp_path):
        sink = str(tmp_path / "audit.ndjson")
        logger = AuditLogger(sink_path=sink)
        logger.log(self._event(tid="acme"))
        with open(sink) as fh:
            lines = fh.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["tenant_id"] == "acme"


# ---------------------------------------------------------------------------
# TenantIsolationManager
# ---------------------------------------------------------------------------

def _rec(tenant_id=None, org=None, subject_id="s1"):
    rec: dict = {"subject_id": subject_id, "run_id": "r1"}
    if tenant_id:
        rec["tenant_id"] = tenant_id
    if org:
        rec["organization_id"] = org
    return rec


class TestTenantIsolationManager:
    def _manager(self):
        configs = [
            TenantConfig("acme", subject_id_prefixes=["acme-"]),
            TenantConfig("beta", quota=QuotaPolicy(max_records_per_day=5)),
        ]
        router = TenantRouter(configs)
        return TenantIsolationManager(router=router, actor="test")

    def test_ingest_routes_by_tenant_id(self):
        mgr = self._manager()
        recs = [_rec(tenant_id="acme"), _rec(tenant_id="acme")]
        admitted = mgr.ingest(recs)
        assert "acme" in admitted
        assert len(admitted["acme"]) == 2

    def test_ingest_emits_audit_events(self):
        mgr = self._manager()
        mgr.ingest([_rec(tenant_id="acme")])
        events = mgr.audit_logger.events(tenant_id="acme", event_type="ingest")
        assert len(events) == 1
        assert events[0].outcome == "success"

    def test_ingest_quota_exceeded_emits_blocked_event(self):
        mgr = self._manager()
        # Fill up beta's quota
        recs = [_rec(tenant_id="beta") for _ in range(5)]
        mgr.ingest(recs)
        with pytest.raises(QuotaExceededError):
            mgr.ingest([_rec(tenant_id="beta")])
        events = mgr.audit_logger.events(tenant_id="beta", event_type="quota_exceeded")
        assert len(events) == 1
        assert events[0].outcome == "blocked"

    def test_ingest_multiple_tenants_segregated(self):
        mgr = self._manager()
        recs = [_rec(tenant_id="acme"), _rec(tenant_id="beta"), _rec(tenant_id="acme")]
        admitted = mgr.ingest(recs)
        assert len(admitted["acme"]) == 2
        assert len(admitted["beta"]) == 1

    def test_query_emits_audit_event(self):
        mgr = self._manager()
        mgr.query("acme", subject_id="acme-bot-1")
        events = mgr.audit_logger.events(tenant_id="acme", event_type="query")
        assert len(events) == 1
        assert events[0].outcome == "success"

    def test_query_denied_for_wrong_subject(self):
        mgr = self._manager()
        # "acme" tenant only allows "acme-" prefix subjects
        mgr.query("acme", subject_id="beta-intruder")
        events = mgr.audit_logger.events(tenant_id="acme", event_type="access_denied")
        assert len(events) == 1
        assert events[0].outcome == "failure"
