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
from uuid import uuid4

__all__ = [
    "TenantID",
    "TenantConfig",
    "QuotaPolicy",
    "QuotaExceededError",
    "QuotaCharge",
    "IngestReservation",
    "TenantClaimMismatchError",
    "TenantContext",
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


class TenantClaimMismatchError(RuntimeError):
    """A record's self-declared tenant identity disagrees with the trusted
    (authenticated transport/session) context.

    Invariant: a data-plane claim is not a trusted routing authority. The
    record is denied before admission and the mismatch is audited."""

    def __init__(self, claimed_tenant: str, trusted_tenant: str) -> None:
        self.claimed_tenant = claimed_tenant
        self.trusted_tenant = trusted_tenant
        super().__init__(
            f"tenant claim mismatch: record claims {claimed_tenant!r} but the "
            f"trusted context is {trusted_tenant!r} — record denied"
        )


@dataclass(frozen=True)
class TenantContext:
    """Authenticated routing authority for an ingest batch.

    Attributes:
        tenant_id: The tenant the *authenticated* transport/session belongs to.
            This — never a field inside the record — determines the namespace.
        actor_id: Authenticated principal (workload / user id), for audit.
        source: Where the identity came from — ``"mTLS"``, ``"JWT"``,
            ``"gateway"``, or ``"trusted-job"``. Ad-hoc values are accepted so
            deployments can name their own trusted transport, but anything
            outside this vocabulary deserves scrutiny in audits.
    """

    tenant_id: TenantID
    actor_id: str | None = None
    source: str = "gateway"


@dataclass
class _DailyUsage:
    date_str: str  # UTC date "YYYY-MM-DD"
    records: int = 0


@dataclass(frozen=True)
class QuotaCharge:
    """Exact quota consumption of one admitted batch segment.

    Returned by :meth:`QuotaEnforcer.check_and_record` and replayed verbatim by
    :meth:`QuotaEnforcer.refund` — rollback never recomputes, it reverses
    precisely what was charged (records, storage bytes, and subject
    references).

    ``charge_id`` makes the refund ONE-SHOT: the enforcer tracks active
    charge ids, so refunding the same charge twice (or refunding a fabricated
    charge) raises instead of silently over-decrementing counters or subject
    refcounts."""

    charge_id: str
    tenant_id: TenantID
    n_records: int
    n_bytes: int
    subjects: frozenset[str]


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
        # tenant_id → subject_id → active reference count. Refcounts (not a
        # bare set) make rollback transactional: a failed batch decrements
        # only ITS references and can never erase a concurrent transaction's
        # committed subject.
        self._subjects: dict[TenantID, dict[str, int]] = {}
        # Live charges (charge_id → the exact charge as issued): a charge is
        # in exactly one of two terminal states after registration — REFUNDED
        # (rollback, counters restored) or COMMITTED (finalized). The registry
        # keeps the FULL charge object so a forged reconstruction with a valid
        # id but altered tenant/amount fields is detected (set[str] cannot).
        self._active_charges: dict[str, QuotaCharge] = {}
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
    ) -> QuotaCharge:
        """Verify that adding *n_records* / *n_bytes* / *subject_ids* does not
        breach *policy*.

        Raises ``QuotaExceededError`` on the first limit hit. On success,
        counters are updated atomically and the exact
        :class:`QuotaCharge` consumed is returned — keep it and pass it to
        :meth:`refund` to roll the whole charge back transactionally.
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

            refs = self._subjects.setdefault(tenant_id, {})
            subjects = set(subject_ids or [])
            new_subjects = {s for s in subjects if refs.get(s, 0) == 0}
            if policy.max_subjects is not None:
                if len(refs) + len(new_subjects) > policy.max_subjects:
                    raise QuotaExceededError(
                        tenant_id, "subjects", policy.max_subjects, len(refs)
                    )

            # Commit
            usage.records += n_records
            self._bytes[tenant_id] = current_bytes + n_bytes
            for s in subjects:
                refs[s] = refs.get(s, 0) + 1

            charge = QuotaCharge(
                charge_id=uuid4().hex,
                tenant_id=tenant_id,
                n_records=n_records,
                n_bytes=n_bytes,
                subjects=frozenset(subjects),
            )
            self._active_charges[charge.charge_id] = charge
            return charge

    def _active_charge_count(self) -> int:
        """Diagnostic: number of live (registered, non-terminal) charges.

        Invariant: the active count returns to its baseline after every
        completed transaction (commit or refund)."""
        with self._lock:
            return len(self._active_charges)

    def _require_active_unlocked(self, charge: QuotaCharge) -> None:
        """Validate that *charge* is live AND byte-identical to the charge
        this enforcer issued. Callers must hold ``self._lock``."""
        registered = self._active_charges.get(charge.charge_id)
        if registered is None:
            raise RuntimeError(
                f"quota charge {charge.charge_id} is not active "
                "(already committed, refunded, or unknown)"
            )
        if registered != charge:
            raise RuntimeError(
                f"quota charge {charge.charge_id} does not match "
                "the charge issued by this enforcer"
            )

    def commit(self, charge: QuotaCharge) -> None:
        """Finalize a live charge: the quota-protected operation succeeded and
        is no longer rollback-eligible.

        Deliberately infallible for a known-active charge — it only validates
        the registry entry and deletes it. No I/O, no callbacks: committing a
        batch of charges cannot partially fail."""
        with self._lock:
            self._require_active_unlocked(charge)
            del self._active_charges[charge.charge_id]

    def usage(self, tenant_id: TenantID) -> dict[str, Any]:
        """Return a snapshot of current usage for *tenant_id*."""
        with self._lock:
            today = self._today_utc()
            daily = self._daily.get(tenant_id)
            return {
                "tenant_id": tenant_id,
                "records_today": daily.records if daily and daily.date_str == today else 0,
                "storage_bytes": self._bytes.get(tenant_id, 0),
                "n_subjects": len(self._subjects.get(tenant_id, {})),
            }

    def reset(self, tenant_id: TenantID) -> None:
        """Reset all counters for *tenant_id* (test-only helper).

        Refuses to run while live charges exist for the tenant — silently
        discarding registered charges would strand lifecycle state and allow
        later refunds to decrement counters that were just zeroed."""
        with self._lock:
            if any(
                charge.tenant_id == tenant_id
                for charge in self._active_charges.values()
            ):
                raise RuntimeError(
                    f"cannot reset tenant {tenant_id!r} while active quota "
                    "charges exist — refund or commit them first"
                )
            self._daily.pop(tenant_id, None)
            self._bytes.pop(tenant_id, None)
            self._subjects.pop(tenant_id, None)

    def refund(self, charge: QuotaCharge) -> None:
        """Roll back a prior :meth:`check_and_record` charge exactly — ONCE.

        Used by batch admission: if a later tenant in the same batch breaches
        its quota, tenants already charged in the batch are refunded so a
        failed write never permanently consumes quota. Subject refs are
        DECREMENTED (not discarded): a subject survives if a concurrent
        transaction still holds a reference, so rollback can never erase
        another transaction's committed state.

        One-shot: the validation, the rollback, and the terminal state change
        all happen under the same lock. A second refund, a committed charge,
        or a forged reconstruction raises RuntimeError instead of silently
        over-decrementing."""
        with self._lock:
            self._require_active_unlocked(charge)
            today = self._today_utc()
            usage = self._daily.get(charge.tenant_id)
            if usage is not None and usage.date_str == today:
                usage.records = max(0, usage.records - charge.n_records)
            if charge.n_bytes:
                current = self._bytes.get(charge.tenant_id, 0)
                self._bytes[charge.tenant_id] = max(0, current - charge.n_bytes)
            refs = self._subjects.get(charge.tenant_id, {})
            for subject in charge.subjects:
                current_refs = refs.get(subject, 0)
                if current_refs <= 1:
                    refs.pop(subject, None)
                else:
                    refs[subject] = current_refs - 1
            # Terminal state: the charge can never be refunded or committed again.
            del self._active_charges[charge.charge_id]


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

    def resolve(
        self,
        record: dict[str, Any],
        tenant_context: TenantContext | None = None,
    ) -> TenantID:
        """Return the ``TenantID`` for *record*.

        Two authority modes:

        **Trusted-context mode** (``tenant_context`` given — the recommended
        production posture): the authenticated transport/session context is the
        ONLY routing authority. Fields inside the data record
        (``tenant_id`` / ``organization_id`` / ``run_context.org``) are
        *claims* to cross-check, not selectors: a record claim that disagrees
        with the trusted context raises :class:`TenantClaimMismatchError`
        (deny + audit upstream). Subject-prefix matching can never switch the
        tenant either.

        **Legacy mode** (no trusted context): the historical resolution order
        applies (record fields → run_context → subject prefix → default).
        Deployments should treat this as untrusted routing and migrate to
        trusted-context mode; see ``MultiTenantManager.require_trusted_context``
        for the fail-closed variant.
        """
        if tenant_context is not None:
            claimed = self.extract_claimed_tenant(record)
            if claimed is not None and claimed != tenant_context.tenant_id:
                raise TenantClaimMismatchError(
                    claimed_tenant=claimed,
                    trusted_tenant=tenant_context.tenant_id,
                )
            return tenant_context.tenant_id

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

    @staticmethod
    def extract_claimed_tenant(record: dict[str, Any]) -> str | None:
        """Return the tenant identity the RECORD itself claims (if any)."""
        for field_name in ("tenant_id", "organization_id"):
            val = record.get(field_name)
            if val and isinstance(val, str):
                return val
        ctx = record.get("run_context")
        if isinstance(ctx, dict):
            org = ctx.get("org") or ctx.get("organization_id") or ctx.get("tenant_id")
            if org and isinstance(org, str):
                return org
        return None

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


@dataclass(frozen=True)
class IngestReservation:
    """Admission state of one `reserve_ingest()` call — NOT yet committed.

    Holds everything a caller needs to finish the ingest transaction:
      * `admitted` — the per-tenant record sets that passed quota admission
        (immutable tuples; convert to lists after commit if needed);
      * `charges` — the exact QuotaCharge objects consumed by admission, to be
        finalized by :meth:`TenantIsolationManager.commit` or reversed by
        :meth:`TenantIsolationManager.rollback`;
      * `audit_events` — PREPARED success events, emitted only by
        ``commit()`` once durable storage has actually succeeded. A rolled
        back reservation therefore never leaves a false ingest-success record
        behind.

    Storage quota becomes COMMITTED only after durable storage has succeeded.
    """

    admitted: dict[TenantID, tuple[dict[str, Any], ...]]
    charges: tuple[QuotaCharge, ...]
    audit_events: tuple[AuditEvent, ...]


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
        require_trusted_context: bool = True,
    ) -> None:
        self.router = router or TenantRouter([])
        self.enforcer = enforcer or QuotaEnforcer()
        self.audit_logger = audit_logger or AuditLogger()
        self.actor = actor
        # Fail-closed by default (T-R01/T-R04): ingest() without a trusted
        # TenantContext is denied outright. Routing from record-controlled
        # fields is UNTRUSTED COMPATIBILITY MODE — never normal production
        # multi-tenancy — and requires the explicit opt-out
        # (require_trusted_context=False).
        self.require_trusted_context = require_trusted_context

    def reserve_ingest(
        self,
        records: list[dict[str, Any]],
        tenant_context: TenantContext | None = None,
    ) -> IngestReservation:
        """Adjudicate admission for *records* WITHOUT finalizing anything.

        Responsibilities (and nothing more): trusted-context validation, tenant
        routing, byte accounting, quota charging, admitted-mapping construction,
        and preparation of the success audit events.

        It must NOT emit success audits, commit quota charges, or perform
        durable storage writes — those belong to :meth:`commit` after the
        durable store has succeeded, or to :meth:`rollback` on failure.

        Transaction exception policy (§18):
          * before the first charge — propagate directly;
          * after one or more charges — refund every charge in this batch
            (reverse order) FIRST, then propagate (audit delivery failures in
            the blocked-audit path never strand quota).

        Storage quota becomes COMMITTED only after durable storage has
        succeeded — a later durable-write failure rolls the reservation back
        instead of leaving phantom storage-byte usage behind.
        """
        if self.require_trusted_context and tenant_context is None:
            self.audit_logger.log(AuditEvent(
                event_type="access_denied",
                tenant_id="_unauthenticated",
                actor=self.actor,
                resource="traces",
                outcome="blocked",
                detail={"reason": "no trusted tenant context in strict mode"},
            ))
            raise TenantClaimMismatchError(claimed_tenant="<none>", trusted_tenant="<required>")

        # Group records by tenant, cross-checking self-declared claims against
        # the trusted context. A mismatch is an active spoofing signal: the
        # whole batch is denied (fail closed) and the mismatch is audited.
        grouped: dict[TenantID, list[dict[str, Any]]] = {}
        for rec in records:
            try:
                tid = self.router.resolve(rec, tenant_context=tenant_context)
            except TenantClaimMismatchError as exc:
                self.audit_logger.log(AuditEvent(
                    event_type="access_denied",
                    tenant_id=tenant_context.tenant_id if tenant_context else "_unknown",
                    actor=tenant_context.actor_id if tenant_context else self.actor,
                    resource="traces",
                    outcome="blocked",
                    detail={
                        "reason": "tenant claim mismatch",
                        "claimed_tenant": exc.claimed_tenant,
                        "trusted_tenant": exc.trusted_tenant,
                    },
                ))
                raise
            grouped.setdefault(tid, []).append(rec)

        # Byte accounting (Q06 semantics): quota is measured over the encoded
        # JSON payload size, UTF-8, before compression — the wire size the
        # pipeline actually ingests, not Python object memory size.
        n_bytes_by_tenant: dict[TenantID, int] = {
            tid: len(json.dumps(recs, ensure_ascii=False, default=str).encode("utf-8"))
            for tid, recs in grouped.items()
        }

        admitted: dict[TenantID, tuple[dict[str, Any], ...]] = {}
        # Exact QuotaCharge objects as charged — rollback replays them
        # verbatim, in reverse order (transaction semantics).
        charged: list[QuotaCharge] = []
        pending_audit: list[AuditEvent] = []
        try:
            for tid, recs in grouped.items():
                cfg = self.router.config_for(tid)
                quota = cfg.quota if cfg else QuotaPolicy()
                subject_ids = list({
                    str(r.get("subject_id") or
                        (r.get("run_context") or {}).get("subject_id") or "")
                    for r in recs
                } - {""})
                try:
                    charge = self.enforcer.check_and_record(
                        tid, quota, n_records=len(recs), subject_ids=subject_ids,
                        n_bytes=n_bytes_by_tenant[tid],
                    )
                except QuotaExceededError as exc:
                    # §7: ROLLBACK FIRST — a quota failure must never leave
                    # earlier charges stranded just because a non-essential
                    # audit sink is about to be attempted. Clearing `charged`
                    # hands ownership to this handler so the outer envelope
                    # cannot double-refund.
                    for c in reversed(charged):
                        self.enforcer.refund(c)
                    charged.clear()
                    try:
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
                                "n_bytes_attempted": n_bytes_by_tenant[tid],
                            },
                        ))
                    except Exception as audit_exc:  # noqa: BLE001
                        # Audit delivery is observability, not transaction
                        # state — surface it chained, but the original quota
                        # failure remains the cause.
                        raise RuntimeError(
                            "quota_exceeded audit delivery failed"
                        ) from audit_exc
                    raise
                charged.append(charge)
                admitted[tid] = tuple(recs)
                # Success audit is PREPARED, not emitted: it must never claim
                # success for a batch that later fails or rolls back.
                pending_audit.append(AuditEvent(
                    event_type="ingest",
                    tenant_id=tid,
                    actor=self.actor,
                    resource="traces",
                    outcome="success",
                    detail={
                        "n_records": len(recs),
                        "n_subjects": len(subject_ids),
                        "n_bytes": n_bytes_by_tenant[tid],
                    },
                ))
        except Exception:
            # §6: ANY ordinary exception after a charge (audit sink failure,
            # serialization error, policy callback, internal bug) must strand
            # no quota state — refund everything charged in THIS batch.
            for c in reversed(charged):
                self.enforcer.refund(c)
            raise

        return IngestReservation(
            admitted=admitted,
            charges=tuple(charged),
            audit_events=tuple(pending_audit),
        )

    def commit(self, reservation: IngestReservation) -> None:
        """Finalize a reservation after durable storage has succeeded.

        Order: quota lifecycle first (charges become COMMITTED and
        irreversible), then success audit delivery. Audit delivery failure
        after commit is surfaced to the caller but must NOT roll back
        already-durable data or committed quota — audit is append-only
        observability, not part of the storage transaction (§8).
        """
        for charge in reservation.charges:
            self.enforcer.commit(charge)
        for event in reservation.audit_events:
            self.audit_logger.log(event)

    def rollback(self, reservation: IngestReservation) -> None:
        """Undo a reservation whose durable write failed: refund every charge
        in reverse order. No success audit is emitted; a single
        ``ingest_aborted`` event is recorded instead (§9)."""
        for charge in reversed(reservation.charges):
            self.enforcer.refund(charge)
        first_tenant = next(iter(reservation.admitted), "_unknown")
        self.audit_logger.log(AuditEvent(
            event_type="ingest_aborted",
            tenant_id=first_tenant,
            actor=self.actor,
            resource="traces",
            outcome="blocked",
            detail={
                "reason": "durable storage failed; reservation rolled back",
                "n_tenants": len(reservation.admitted),
            },
        ))

    def ingest_to_store(
        self,
        records: list[dict[str, Any]],
        store: Any,
        tenant_context: TenantContext | None = None,
    ) -> dict[TenantID, tuple[dict[str, Any], ...]]:
        """Load-bearing production path: reserve → durable write → commit.

        The storage quota becomes COMMITTED only after ``store.write`` has
        succeeded; a durable-write failure rolls the whole reservation back
        (quota restored, zero success audits) before the error propagates."""
        reservation = self.reserve_ingest(records, tenant_context=tenant_context)
        try:
            store.write(reservation.admitted)
        except Exception:
            self.rollback(reservation)
            raise
        self.commit(reservation)
        return reservation.admitted

    def ingest(
        self,
        records: list[dict[str, Any]],
        tenant_context: TenantContext | None = None,
    ) -> dict[TenantID, list[dict[str, Any]]]:
        """Admission-only compatibility API.

        .. warning::
            **UNTRUSTED-FOR-DURABLE-WRITES**: ``ingest()`` commits quota
            immediately and MUST NOT be used when a later durable storage
            failure should restore storage quota. Use
            :meth:`ingest_to_store` / :meth:`reserve_ingest` for production
            durable writes.

        Returns a mapping of ``TenantID`` → list of admitted records.
        Raises ``QuotaExceededError`` if any tenant's quota would be breached
        (with all-or-nothing batch refund semantics preserved).
        """
        reservation = self.reserve_ingest(records, tenant_context=tenant_context)
        self.commit(reservation)
        return {
            tid: list(recs)
            for tid, recs in reservation.admitted.items()
        }

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
