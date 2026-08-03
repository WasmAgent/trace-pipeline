"""Multi-tenant isolation — issue #57.

Introduces tenant-scoped trace segregation (by organisation/team boundaries),
per-tenant resource quotas, and audit-grade access logging for compliance
consumers.

Design
------
- ``TenantID``          — type alias (``str``) for organisation/team identifier.
- ``TenantConfig``      — declarative per-tenant configuration: allowed
  ``subject_id`` prefixes, storage namespace, and resource quotas.
- ``QuotaPolicy``       — per-tenant resource limits (max records/day, max
  storage bytes, max subjects). ``QuotaEnforcer`` tracks usage and raises
  ``QuotaExceededError`` when a limit would be breached.
- ``TenantRouter``      — resolves a record to its ``TenantID`` by inspecting
  ``organization_id``, ``tenant_id``, ``run_context.org``, and
  subject-ID-prefix matching.
- ``AuditLogger``       — append-only, thread-safe audit log. Each ``AuditEvent``
  records who accessed/wrote what and the outcome.  Exportable as NDJSON for
  compliance consumers.
- ``TenantIsolationManager`` — top-level facade: route, quota-check, validate,
  store, and audit — all in one call.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TenantID",
    "TenantConfig",
    "QuotaPolicy",
    "QuotaExceededError",
    "QuotaEnforcer",
    "TenantRouter",
    "AuditEvent",
    "AuditLogger",
    "TenantIsolationManager",
]

# A tenant identifier is just a string (organisation slug or UUID).
TenantID = str


# ---------------------------------------------------------------------------
# Tenant config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuotaPolicy:
    """Per-tenant resource limits.

    Attributes:
        max_records_per_day: Maximum number of trace records ingested per UTC day.
            ``None`` = unlimited.
        max_storage_bytes: Cumulative storage cap.  ``None`` = unlimited.
        max_subjects: Maximum number of distinct ``subject_id`` values.
            ``None`` = unlimited.
    """

    max_records_per_day: int | None = None
    max_storage_bytes: int | None = None
    max_subjects: int | None = None


@dataclass
class TenantConfig:
    """Per-tenant configuration.

    Attributes:
        tenant_id: Canonical identifier for this tenant.
        display_name: Human-readable name (for audit logs and dashboards).
        subject_id_prefixes: If non-empty, only records whose ``subject_id``
            starts with one of these prefixes are admitted to this tenant.
            Empty list = accept any subject_id.
        storage_namespace: Key prefix under which this tenant's traces are
            stored (e.g. ``"tenant=acme"``).
        quota: Resource quota policy.
    """

    tenant_id: TenantID
    display_name: str = ""
    subject_id_prefixes: list[str] = field(default_factory=list)
    storage_namespace: str = ""
    quota: QuotaPolicy = field(default_factory=QuotaPolicy)

    def __post_init__(self) -> None:
        if not self.storage_namespace:
            self.storage_namespace = f"tenant={self.tenant_id}"

    def allows_subject(self, subject_id: str) -> bool:
        """Return True if *subject_id* is admitted by this tenant's config."""
        if not self.subject_id_prefixes:
            return True
        return any(subject_id.startswith(p) for p in self.subject_id_prefixes)


# ---------------------------------------------------------------------------
# Quota enforcement
# ---------------------------------------------------------------------------

class QuotaExceededError(RuntimeError):
    """Raised when an operation would breach a tenant's quota."""

    def __init__(self, tenant_id: TenantID, resource: str, limit: int, current: int) -> None:
        self.tenant_id = tenant_id
        self.resource = resource
        self.limit = limit
        self.current = current
        super().__init__(
            f"Tenant '{tenant_id}' quota exceeded: {resource} "
            f"(limit={limit}, current={current})"
        )


@dataclass
class _DailyUsage:
    date_str: str  # UTC date "YYYY-MM-DD"
    records: int = 0


class QuotaEnforcer:
    """Tracks and enforces per-tenant quotas.

    Thread-safe. Usage counters are in-process; in production they would be
    backed by Redis atomics or a DB row.
    """

    def __init__(self) -> None:
        # tenant_id → _DailyUsage
        self._daily: dict[TenantID, _DailyUsage] = {}
        # tenant_id → total bytes written (session total, non-persistent)
        self._bytes: dict[TenantID, int] = {}
        # tenant_id → set of seen subject_ids
        self._subjects: dict[TenantID, set[str]] = {}
        self._lock = threading.Lock()

    def _today_utc(self) -> str:
        import datetime
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    def check_and_record(
        self,
        tenant_id: TenantID,
        policy: QuotaPolicy,
        n_records: int,
        n_bytes: int = 0,
        subject_ids: list[str] | None = None,
    ) -> None:
        """Verify that adding *n_records* / *n_bytes* does not breach *policy*.

        Raises ``QuotaExceededError`` on the first limit hit. On success,
        counters are updated atomically.
        """
        with self._lock:
            today = self._today_utc()
            usage = self._daily.get(tenant_id)
            if usage is None or usage.date_str != today:
                usage = _DailyUsage(date_str=today, records=0)
                self._daily[tenant_id] = usage

            if policy.max_records_per_day is not None:
                new_total = usage.records + n_records
                if new_total > policy.max_records_per_day:
                    raise QuotaExceededError(
                        tenant_id, "records_per_day", policy.max_records_per_day, usage.records
                    )

            current_bytes = self._bytes.get(tenant_id, 0)
            if policy.max_storage_bytes is not None:
                if current_bytes + n_bytes > policy.max_storage_bytes:
                    raise QuotaExceededError(
                        tenant_id, "storage_bytes", policy.max_storage_bytes, current_bytes
                    )

            seen = self._subjects.setdefault(tenant_id, set())
            new_subjects = set(subject_ids or []) - seen
            if policy.max_subjects is not None:
                if len(seen) + len(new_subjects) > policy.max_subjects:
                    raise QuotaExceededError(
                        tenant_id, "subjects", policy.max_subjects, len(seen)
                    )

            # Commit
            usage.records += n_records
            self._bytes[tenant_id] = current_bytes + n_bytes
            seen.update(new_subjects)

    def usage(self, tenant_id: TenantID) -> dict[str, Any]:
        """Return a snapshot of current usage for *tenant_id*."""
        with self._lock:
            today = self._today_utc()
            daily = self._daily.get(tenant_id)
            return {
                "tenant_id": tenant_id,
                "records_today": daily.records if daily and daily.date_str == today else 0,
                "storage_bytes": self._bytes.get(tenant_id, 0),
                "n_subjects": len(self._subjects.get(tenant_id, set())),
            }

    def reset(self, tenant_id: TenantID) -> None:
        """Reset all counters for *tenant_id* (useful in tests)."""
        with self._lock:
            self._daily.pop(tenant_id, None)
            self._bytes.pop(tenant_id, None)
            self._subjects.pop(tenant_id, None)


# ---------------------------------------------------------------------------
# Tenant router
# ---------------------------------------------------------------------------

class TenantRouter:
    """Resolves an AEP record to a ``TenantID``.

    Resolution order:
      1. ``record["tenant_id"]`` (explicit override)
      2. ``record["organization_id"]``
      3. ``record["run_context"]["org"]`` (AEP v0.3 nesting)
      4. ``subject_id`` prefix matching against registered ``TenantConfig`` objects
      5. ``default_tenant`` (fallback)

    Args:
        configs: Registered tenant configurations.
        default_tenant: Tenant ID to assign when no rule matches.
    """

    def __init__(
        self,
        configs: list[TenantConfig],
        default_tenant: TenantID = "_default",
    ) -> None:
        self._configs = {c.tenant_id: c for c in configs}
        self._default = default_tenant

    def resolve(self, record: dict[str, Any]) -> TenantID:
        """Return the ``TenantID`` for *record*."""
        # Explicit fields
        for field_name in ("tenant_id", "organization_id"):
            val = record.get(field_name)
            if val and isinstance(val, str):
                return val
        # run_context.org
        ctx = record.get("run_context")
        if isinstance(ctx, dict):
            org = ctx.get("org") or ctx.get("organization_id") or ctx.get("tenant_id")
            if org and isinstance(org, str):
                return org
        # Subject-ID prefix matching
        subject_id = record.get("subject_id") or (
            ctx.get("subject_id") if isinstance(ctx, dict) else None
        )
        if subject_id:
            for cfg in self._configs.values():
                if cfg.allows_subject(str(subject_id)) and cfg.subject_id_prefixes:
                    return cfg.tenant_id
        return self._default

    def config_for(self, tenant_id: TenantID) -> TenantConfig | None:
        return self._configs.get(tenant_id)

    def register(self, config: TenantConfig) -> None:
        self._configs[config.tenant_id] = config

    def all_tenants(self) -> list[TenantID]:
        return list(self._configs.keys())


# ---------------------------------------------------------------------------
# Audit logger
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    """A single audit-trail entry.

    Attributes:
        event_type: ``"ingest"``, ``"query"``, ``"quota_exceeded"``,
            ``"config_change"``, or ``"access_denied"``.
        tenant_id: Tenant that performed / triggered the event.
        actor: System component or user identity (e.g. ``"worker-3"``).
        resource: What was accessed/modified (e.g. ``"traces"``, ``"config"``).
        outcome: ``"success"`` | ``"failure"`` | ``"blocked"``.
        detail: Free-form detail dict for compliance consumers.
        timestamp: Unix epoch (float).
    """

    event_type: str
    tenant_id: TenantID
    actor: str
    resource: str
    outcome: str
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Serialise to a single-line JSON string (NDJSON-compatible)."""
        return json.dumps(
            {
                "event_type": self.event_type,
                "tenant_id": self.tenant_id,
                "actor": self.actor,
                "resource": self.resource,
                "outcome": self.outcome,
                "detail": self.detail,
                "timestamp": self.timestamp,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> AuditEvent:
        data = json.loads(line)
        return cls(**data)


class AuditLogger:
    """Append-only, thread-safe in-process audit log.

    In production this would write to a WORM (Write-Once-Read-Many) log sink
    (e.g. an append-only S3 key, a Kafka topic, or a ``syslog`` facility).
    Here it stores events in memory with an optional file sink.

    Args:
        sink_path: Optional filesystem path to an NDJSON audit log file.
            When provided, every ``log()`` call appends to the file.
    """

    def __init__(self, sink_path: str | None = None) -> None:
        self._events: list[AuditEvent] = []
        self._lock = threading.Lock()
        self._sink_path = sink_path

    def log(self, event: AuditEvent) -> None:
        """Append *event* to the log."""
        with self._lock:
            self._events.append(event)
            if self._sink_path:
                with open(self._sink_path, "a", encoding="utf-8") as fh:
                    fh.write(event.to_json() + "\n")

    def events(
        self,
        tenant_id: TenantID | None = None,
        event_type: str | None = None,
    ) -> list[AuditEvent]:
        """Return events, optionally filtered by tenant and/or event type."""
        with self._lock:
            return [
                e for e in self._events
                if (tenant_id is None or e.tenant_id == tenant_id)
                and (event_type is None or e.event_type == event_type)
            ]

    def export_ndjson(self) -> str:
        """Export all events as a newline-delimited JSON string."""
        with self._lock:
            return "\n".join(e.to_json() for e in self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._events)


# ---------------------------------------------------------------------------
# Top-level facade
# ---------------------------------------------------------------------------

class TenantIsolationManager:
    """Facade that enforces tenant isolation for every trace ingest/query.

    Workflow for each record batch:
      1. Route each record to a ``TenantID``.
      2. Quota-check the batch for each tenant.
      3. Segregate records into per-tenant namespaced stores.
      4. Emit an ``AuditEvent`` per tenant per operation.

    Args:
        router: ``TenantRouter`` for tenant resolution.
        enforcer: ``QuotaEnforcer`` for quota tracking.
        audit_logger: ``AuditLogger`` for compliance events.
        actor: Identity label for audit events (e.g. ``"ingestion-pipeline"``).
    """

    def __init__(
        self,
        router: TenantRouter | None = None,
        enforcer: QuotaEnforcer | None = None,
        audit_logger: AuditLogger | None = None,
        actor: str = "system",
    ) -> None:
        self.router = router or TenantRouter([])
        self.enforcer = enforcer or QuotaEnforcer()
        self.audit_logger = audit_logger or AuditLogger()
        self.actor = actor

    def ingest(
        self,
        records: list[dict[str, Any]],
    ) -> dict[TenantID, list[dict[str, Any]]]:
        """Route, quota-check, and segregate *records* by tenant.

        Returns a mapping of ``TenantID`` → list of admitted records.
        Raises ``QuotaExceededError`` if any tenant's quota would be breached.
        On quota failure, an audit event with ``outcome="blocked"`` is emitted
        before the exception propagates.
        """
        # Group records by tenant
        grouped: dict[TenantID, list[dict[str, Any]]] = {}
        for rec in records:
            tid = self.router.resolve(rec)
            grouped.setdefault(tid, []).append(rec)

        admitted: dict[TenantID, list[dict[str, Any]]] = {}
        for tid, recs in grouped.items():
            cfg = self.router.config_for(tid)
            quota = cfg.quota if cfg else QuotaPolicy()
            subject_ids = list({
                str(r.get("subject_id") or
                    (r.get("run_context") or {}).get("subject_id") or "")
                for r in recs
            } - {""})
            try:
                self.enforcer.check_and_record(
                    tid, quota, n_records=len(recs), subject_ids=subject_ids
                )
            except QuotaExceededError as exc:
                self.audit_logger.log(AuditEvent(
                    event_type="quota_exceeded",
                    tenant_id=tid,
                    actor=self.actor,
                    resource="traces",
                    outcome="blocked",
                    detail={
                        "resource": exc.resource,
                        "limit": exc.limit,
                        "current": exc.current,
                        "n_records_attempted": len(recs),
                    },
                ))
                raise
            admitted[tid] = recs
            self.audit_logger.log(AuditEvent(
                event_type="ingest",
                tenant_id=tid,
                actor=self.actor,
                resource="traces",
                outcome="success",
                detail={"n_records": len(recs), "n_subjects": len(subject_ids)},
            ))

        return admitted

    def query(
        self,
        tenant_id: TenantID,
        subject_id: str | None = None,
        *,
        actor: str | None = None,
    ) -> None:
        """Record an audit event for a query on *tenant_id*.

        This method does not perform actual data retrieval; it is called by
        the retrieval layer to ensure every data access is logged.
        """
        cfg = self.router.config_for(tenant_id)
        if cfg is not None and subject_id is not None and not cfg.allows_subject(subject_id):
            self.audit_logger.log(AuditEvent(
                event_type="access_denied",
                tenant_id=tenant_id,
                actor=actor or self.actor,
                resource=f"traces/subject_id={subject_id}",
                outcome="failure",
                detail={"reason": "subject_id not admitted by tenant config"},
            ))
            return
        self.audit_logger.log(AuditEvent(
            event_type="query",
            tenant_id=tenant_id,
            actor=actor or self.actor,
            resource="traces" if subject_id is None else f"traces/subject_id={subject_id}",
            outcome="success",
            detail={"subject_id": subject_id},
        ))
