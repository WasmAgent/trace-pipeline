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

import hashlib
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

__all__ = [
    "TenantID",
    "TenantConfig",
    "QuotaPolicy",
    "QuotaExceededError",
    "QuotaCharge",
    "IngestReservation",
    "IngestResult",
    "ReservedTenantBatch",
    "ReservationState",
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
    # §5 binding: the SHA-256 of the exact immutable payload this charge
    # accounts for. Set by reservation flows; None for direct enforcer use.
    payload_sha256: str | None = None


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
        payload_sha256: str | None = None,
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
                payload_sha256=payload_sha256,
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
            self._refund_unlocked(charge)
            del self._active_charges[charge.charge_id]

    def commit_many(self, charges: tuple[QuotaCharge, ...]) -> None:
        """Atomically finalize MANY charges (§10/§11).

        Acquires ONE lock, validates EVERY charge, and performs ZERO mutation
        if any validation fails — an invalid charge can never produce a
        partially committed reservation."""
        with self._lock:
            registered: list[QuotaCharge] = []
            for charge in charges:
                self._require_active_unlocked(charge)
                registered.append(charge)
            # Validation completed — no partial commit is possible from here.
            for charge in registered:
                del self._active_charges[charge.charge_id]

    def refund_many(self, charges: tuple[QuotaCharge, ...]) -> None:
        """Atomically roll back MANY charges (§10/§12).

        Same all-or-nothing contract as :meth:`commit_many`: one lock, full
        validation before any mutation, reversed-order refund."""
        with self._lock:
            registered: list[QuotaCharge] = []
            for charge in charges:
                self._require_active_unlocked(charge)
                registered.append(charge)
            # Validation completed. No partial mutation is possible from here.
            for charge in reversed(registered):
                self._refund_unlocked(charge)
            for charge in registered:
                del self._active_charges[charge.charge_id]

    def _refund_unlocked(self, charge: QuotaCharge) -> None:
        """Refund arithmetic. Caller MUST hold ``self._lock`` (§12: the
        non-reentrant lock forbids calling the public refund() here)."""
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
class PendingAuditEvent:
    """Immutable, canonically-serialized SUCCESS audit descriptor (§5).

    Reservations hold THESE instead of mutable AuditEvent objects: the audit
    semantics of a pending event cannot be altered between reserve and
    commit. ``detail_json`` is canonical (sort_keys, compact separators).
    Delivery converts back to a regular :class:`AuditEvent`."""

    event_type: str
    tenant_id: TenantID
    actor: str
    resource: str
    outcome: str
    detail_json: str

    @classmethod
    def prepare(
        cls,
        *,
        event_type: str,
        tenant_id: TenantID,
        actor: str,
        resource: str,
        outcome: str,
        detail: dict[str, Any],
    ) -> PendingAuditEvent:
        detail_json = json.dumps(
            detail, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        return cls(
            event_type=event_type,
            tenant_id=tenant_id,
            actor=actor,
            resource=resource,
            outcome=outcome,
            detail_json=detail_json,
        )

    def to_audit_event(self) -> AuditEvent:
        return AuditEvent(
            event_type=self.event_type,
            tenant_id=self.tenant_id,
            actor=self.actor,
            resource=self.resource,
            outcome=self.outcome,
            detail=json.loads(self.detail_json),
        )


def _encode_reserved_payload(records: list[dict[str, Any]]) -> bytes:
    """ONE canonical encoding policy for reserved tenant batches (P1-A): the
    quota is charged on these bytes and these bytes are what the durable store
    writes — never two different serializations of the same records."""
    return json.dumps(
        records,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _composition_sha256(
    *,
    reservation_id: str,
    batches: tuple[ReservedTenantBatch, ...],
    charges: tuple[QuotaCharge, ...],
    audit_events: tuple[PendingAuditEvent, ...],
) -> str:
    """§6: integrity fingerprint over the reservation's canonical composition
    (id + ordered batch metadata + charge metadata + pending audit canonical
    JSON). Not a substitute for manager ownership — an additional binding for
    store idempotency, reconciliation, and debugging."""
    composition = {
        "reservation_id": reservation_id,
        "batches": [
            {
                "tenant_id": b.tenant_id,
                "payload_sha256": b.payload_sha256,
                "n_records": b.n_records,
                "n_bytes": b.n_bytes,
            }
            for b in batches
        ],
        "charges": [
            {
                "charge_id": c.charge_id,
                "tenant_id": c.tenant_id,
                "n_records": c.n_records,
                "n_bytes": c.n_bytes,
                "payload_sha256": c.payload_sha256,
            }
            for c in charges
        ],
        "audit_events": [
            {
                "event_type": e.event_type,
                "tenant_id": e.tenant_id,
                "actor": e.actor,
                "resource": e.resource,
                "outcome": e.outcome,
                "detail_json": e.detail_json,
            }
            for e in audit_events
        ],
    }
    return hashlib.sha256(
        json.dumps(composition, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ReservedTenantBatch:
    """Immutable, canonically serialized payload for ONE tenant reservation.

    P1-A: the durable payload is the frozen ``payload`` bytes — validated
    content, quota-accounted bytes, and durably stored bytes are the SAME
    bytes. ``payload_sha256`` makes the reservation self-verifying."""

    tenant_id: TenantID
    payload: bytes
    payload_sha256: str
    n_records: int
    n_bytes: int

    def as_mapping(self) -> Mapping[str, Any]:
        """Read-only view of the tenant payload (compatibility convenience —
        the bytes remain the authoritative durable content)."""
        return MappingProxyType(
            {"tenant_id": self.tenant_id, "payload": self.payload}
        )


class ReservationState(str, Enum):
    """Settlement lifecycle of a reservation (§9): exactly one terminal
    state — COMMITTED or ROLLED_BACK — with every transition fail-closed."""

    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class IngestReservation:
    """Admission state of one `reserve_ingest()` call — NOT yet committed.

    P1-A: the authoritative durable payload is a tuple of frozen
    :class:`ReservedTenantBatch` objects — there is no mutable top-level dict
    that a caller could mutate after quota validation. Tenant lookup is
    available via :meth:`batch_for`.

    Every reservation MUST be settled exactly once — via
    :meth:`TenantIsolationManager.commit` (durable write succeeded) or
    :meth:`TenantIsolationManager.rollback` (durable write failed). An
    abandoned reservation stays visible in ``active_reservations()`` for
    reconciliation; it is never silently auto-refunded (P2-D), because a
    silent refund could undo quota for a durable write that actually
    happened."""

    reservation_id: str
    created_at_ms: int
    expires_at_ms: int | None
    batches: tuple[ReservedTenantBatch, ...]
    charges: tuple[QuotaCharge, ...]
    audit_events: tuple[PendingAuditEvent, ...]
    # §6: integrity fingerprint over (reservation_id, ordered batch metadata,
    # charge metadata, pending audit canonical JSON). Storage idempotency,
    # reconciliation, and debugging bind to this digest.
    composition_sha256: str

    def batch_for(self, tenant_id: TenantID) -> ReservedTenantBatch | None:
        for batch in self.batches:
            if batch.tenant_id == tenant_id:
                return batch
        return None


@dataclass
class _ReservationEntry:
    """Manager-owned authoritative composition of an issued reservation (§3).

    The registry stores the FULL reservation — settlement always uses THIS
    object, never a caller-supplied reconstruction, so a subset/altered
    composition cannot commit under a valid reservation_id. The embedded
    reservation stays frozen; only ``state`` mutates, and only under
    ``_reservation_lock``."""

    reservation: IngestReservation
    state: ReservationState


class TraceStore:
    """Durable-store interface for the reservation path (§9).

    ``write_reservation`` publishes ALL tenant batches of a reservation
    atomically — ALL or NONE (§10 transactional backend, or §11
    staging + commit manifest for non-transactional backends).

    Idempotency (§12): same reservation_id + same composition_sha256 →
    success without duplicates; same reservation_id + DIFFERENT
    composition_sha256 → hard failure."""

    def write_reservation(
        self,
        reservation_id: str,
        batches: tuple[ReservedTenantBatch, ...],
        *,
        composition_sha256: str,
    ) -> None:
        raise NotImplementedError

    def reservation_status(
        self,
        reservation_id: str,
    ) -> Literal["absent", "staged", "committed"]:
        """Crash-recovery probe (§14): what does the STORE believe happened
        to this reservation? Divergence from the manager's own state is
        detectable and reconcilable."""
        return "absent"


@dataclass(frozen=True)
class IngestResult:
    """Outcome of a FULLY SETTLED ingest (durable write + quota settlement).

    P1-C: post-commit audit-delivery failure is NOT an ingest failure —
    ``committed`` stays True and the audit problem is reported through
    ``audit_status``/``audit_error`` so callers cannot mistake it for a
    retry-safe pre-commit error."""

    committed: bool
    batches: tuple[ReservedTenantBatch, ...]
    audit_status: Literal["delivered", "failed"]
    audit_error: str | None = None


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
        """Append *event* to the log.

        Sink authority (§20): when a file sink is configured, the FILE is the
        authoritative sink — it is written FIRST and flushed, and the in-memory
        mirror is only updated on file success. The reverse order would leave
        an in-memory record for an event the external sink never received."""
        line = event.to_json()
        with self._lock:
            if self._sink_path:
                with open(self._sink_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
            self._events.append(event)

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
        reservation_ttl_ms: int | None = None,
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
        # Reservation settlement registry (§9/§13): reservation_id → state.
        # Lock ordering (§14): reservation_lock -> enforcer._lock, held
        # briefly, with NO audit/durable/network I/O inside either lock.
        # §3: reservation_id -> AUTHORITATIVE entry (full reservation +
        # state). Settlement uses the stored composition, never a
        # caller-supplied reconstruction.
        self._reservations: dict[str, _ReservationEntry] = {}
        self._reservation_lock = threading.Lock()
        self._reservation_created_at: dict[str, int] = {}
        self._reservation_expires_at: dict[str, int] = {}
        # P2-D (§22): optional TTL for reconciliation diagnostics. Expiry
        # NEVER auto-refunds — a reservation expiring only means it requires
        # manual reconciliation, because the durable store may have committed.
        self.reservation_ttl_ms = reservation_ttl_ms

    def active_reservations(self) -> list[dict[str, Any]]:
        """Reconciliation diagnostics (§23): every ACTIVE reservation with its
        age. Long-lived ACTIVE reservations are an operational alert, never
        silently auto-refunded."""
        now = int(time.time() * 1000)
        with self._reservation_lock:
            out = []
            for rid, entry in self._reservations.items():
                if entry.state is not ReservationState.ACTIVE:
                    continue
                created = self._reservation_created_at.get(rid, 0)
                out.append({
                    "reservation_id": rid,
                    "state": entry.state.value,
                    "age_seconds": (now - created) / 1000.0,
                    "expires_at_ms": self._reservation_expires_at.get(rid),
                })
            return out

    def reserve_ingest(
        self,
        records: list[dict[str, Any]],
        tenant_context: TenantContext | None = None,
    ) -> IngestReservation:
        """Adjudicate admission for *records* WITHOUT finalizing anything.

        Responsibilities (and nothing more): trusted-context validation, tenant
        routing, CANONICAL payload serialization (P1-A — the frozen payload
        bytes are the quota-accounted bytes AND the durable bytes), quota
        charging bound to the payload digest (§5), and preparation of the
        success audit events.

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

        The returned reservation MUST be settled exactly once (P2-D):
        :meth:`commit` or :meth:`rollback`. Abandoned reservations stay
        visible via :meth:`active_reservations` — they are never silently
        auto-refunded.
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

        admitted_batches: list[ReservedTenantBatch] = []
        # Exact QuotaCharge objects as charged — rollback replays them
        # verbatim, in reverse order (transaction semantics).
        charged: list[QuotaCharge] = []
        pending_audit: list[PendingAuditEvent] = []
        try:
            for tid, recs in grouped.items():
                # P1-A: ONE canonical encoding of the tenant batch. The quota
                # is charged on these bytes and these bytes are what the
                # durable store writes — validated content == accounted bytes
                # == stored bytes. No reserialization divergence is possible.
                payload = _encode_reserved_payload(recs)
                batch = ReservedTenantBatch(
                    tenant_id=tid,
                    payload=payload,
                    payload_sha256=hashlib.sha256(payload).hexdigest(),
                    n_records=len(recs),
                    n_bytes=len(payload),
                )
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
                        n_bytes=batch.n_bytes,
                        payload_sha256=batch.payload_sha256,
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
                                "n_bytes_attempted": batch.n_bytes,
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
                admitted_batches.append(batch)
                # Success audit is PREPARED, not emitted: it must never claim
                # success for a batch that later fails or rolls back. Frozen
                # PendingAuditEvent (§5) — semantics cannot change between
                # reserve and commit.
                pending_audit.append(PendingAuditEvent.prepare(
                    event_type="ingest",
                    tenant_id=tid,
                    actor=self.actor,
                    resource="traces",
                    outcome="success",
                    detail={
                        "n_records": len(recs),
                        "n_subjects": len(subject_ids),
                        "n_bytes": batch.n_bytes,
                        "payload_sha256": batch.payload_sha256,
                    },
                ))
        except Exception:
            # §6: ANY ordinary exception after a charge (audit sink failure,
            # serialization error, policy callback, internal bug) must strand
            # no quota state — refund everything charged in THIS batch.
            for c in reversed(charged):
                self.enforcer.refund(c)
            raise

        now_ms = int(time.time() * 1000)
        reservation_id = uuid4().hex
        expires_at_ms = (
            now_ms + self.reservation_ttl_ms
            if self.reservation_ttl_ms is not None
            else None
        )
        reservation = IngestReservation(
            reservation_id=reservation_id,
            created_at_ms=now_ms,
            expires_at_ms=expires_at_ms,
            batches=tuple(admitted_batches),
            charges=tuple(charged),
            audit_events=tuple(pending_audit),
            composition_sha256=_composition_sha256(
                reservation_id=reservation_id,
                batches=tuple(admitted_batches),
                charges=tuple(charged),
                audit_events=tuple(pending_audit),
            ),
        )
        # §3: the manager owns the CANONICAL reservation composition. The
        # stored object is authoritative — settlement always uses it.
        with self._reservation_lock:
            self._reservations[reservation.reservation_id] = _ReservationEntry(
                reservation=reservation,
                state=ReservationState.ACTIVE,
            )
            self._reservation_created_at[reservation.reservation_id] = now_ms
            if expires_at_ms is not None:
                self._reservation_expires_at[reservation.reservation_id] = expires_at_ms
        return reservation

    def _resolve_settlement_target(
        self,
        reservation: IngestReservation | str,
    ) -> tuple[str, IngestReservation]:
        """§4: resolve the AUTHORITATIVE reservation for a settlement call.

        Accepts a reservation_id (preferred) or an IngestReservation object
        (compatibility). The returned composition ALWAYS comes from the
        manager's registry — a caller-supplied reconstruction with a valid id
        but altered batches/charges/audit events is detected and rejected
        (RF01–RF03 are structurally impossible on the id-based path)."""
        if isinstance(reservation, IngestReservation):
            rid = reservation.reservation_id
        else:
            rid = reservation
        entry = self._reservations.get(rid)
        if entry is None:
            raise RuntimeError(f"unknown reservation {rid}")
        if isinstance(reservation, IngestReservation) and entry.reservation != reservation:
            raise RuntimeError(
                f"reservation {rid} does not match the composition issued by "
                "this manager (RF01–RF03: subset/altered compositions denied)"
            )
        return rid, entry.reservation

    def commit(self, reservation: IngestReservation | str) -> IngestResult:
        """Atomically finalize a reservation after durable storage succeeded
        (§13): state check + quota settlement + state transition happen under
        the reservation lock with no I/O; exactly one terminal transition can
        win, and quota settlement is all-or-nothing (commit_many).

        Settles BY reservation_id — the authoritative composition comes from
        the manager's registry, never from the caller's hands. Passing the
        original IngestReservation object is accepted as a compatibility
        form and verified for exact equality (RF01–RF03).

        Success audit delivery is a post-commit obligation: a delivery failure
        is REPORTED in the returned IngestResult (audit_status="failed") and
        never rolls back committed quota or durable data (§15/§16)."""
        with self._reservation_lock:
            rid, canonical = self._resolve_settlement_target(reservation)
            entry = self._reservations[rid]
            if entry.state is not ReservationState.ACTIVE:
                raise RuntimeError(
                    f"reservation {rid} is not active "
                    f"(state: {entry.state.value})"
                )
            self.enforcer.commit_many(canonical.charges)
            entry.state = ReservationState.COMMITTED

        return self._deliver_success_audit(canonical)

    def rollback(self, reservation: IngestReservation | str) -> None:
        """Atomically undo a reservation whose durable write failed (§13):
        every charge refunded (reverse order) or none; state transitions to
        ROLLED_BACK. Settles BY reservation_id against the manager-owned
        composition. Emits a single ``ingest_aborted`` event — never per-
        tenant success events (P1-B)."""
        with self._reservation_lock:
            rid, canonical = self._resolve_settlement_target(reservation)
            entry = self._reservations[rid]
            if entry.state is not ReservationState.ACTIVE:
                raise RuntimeError(
                    f"reservation {rid} is not active "
                    f"(state: {entry.state.value})"
                )
            self.enforcer.refund_many(canonical.charges)
            entry.state = ReservationState.ROLLED_BACK

        first_tenant = (
            canonical.batches[0].tenant_id
            if canonical.batches
            else "_unknown"
        )
        self.audit_logger.log(AuditEvent(
            event_type="ingest_aborted",
            tenant_id=first_tenant,
            actor=self.actor,
            resource="traces",
            outcome="blocked",
            detail={
                "reason": "durable storage failed; reservation rolled back",
                "n_tenants": len(canonical.batches),
            },
        ))

    def _deliver_success_audit(
        self, reservation: IngestReservation
    ) -> IngestResult:
        """Post-commit success-audit delivery (§18): delivery failure is
        reported, never rolled back."""
        try:
            for pending in reservation.audit_events:
                self.audit_logger.log(pending.to_audit_event())
            return IngestResult(
                committed=True,
                batches=reservation.batches,
                audit_status="delivered",
            )
        except Exception as exc:  # noqa: BLE001
            return IngestResult(
                committed=True,
                batches=reservation.batches,
                audit_status="failed",
                audit_error=repr(exc),
            )

    def reconcile_reservation(
        self,
        reservation_id: str,
        store: TraceStore,
    ) -> None:
        """Crash-window reconciliation (§14): align the manager's settlement
        state with what the STORE believes happened.

        If the store committed → finalize the quota side. If the store has no
        committed data (absent/staged) → roll the reservation back. This is a
        recovery tool, not an automatic process — callers invoke it explicitly
        after detecting divergence.

        Note: quota counters and reservation lifecycle are in-process state; a
        process restart loses them. Production durability ultimately requires
        a durable quota/reservation backend (§14)."""
        status = store.reservation_status(reservation_id)
        if status == "committed":
            self.commit(reservation_id)
        elif status in ("absent", "staged"):
            self.rollback(reservation_id)
        else:
            raise RuntimeError(f"unknown store reservation status: {status!r}")

    def ingest_to_store(
        self,
        records: list[dict[str, Any]],
        store: TraceStore,
        tenant_context: TenantContext | None = None,
    ) -> IngestResult:
        """Load-bearing production path (§9): reserve → ATOMIC durable write
        of the whole reservation (all tenant batches or none, hash-verified,
        idempotent by reservation_id + composition digest) → atomic commit.

        The storage quota becomes COMMITTED only after ``write_reservation``
        has succeeded; a durable-write failure rolls the whole reservation
        back (quota restored, zero success audits) before the error
        propagates. Returns an :class:`IngestResult` — a post-commit audit
        failure is reported through it, not raised, so a caller can never
        mistake a committed ingest for a retry-safe failure (P1-C)."""
        reservation = self.reserve_ingest(records, tenant_context=tenant_context)
        try:
            store.write_reservation(
                reservation.reservation_id,
                reservation.batches,
                composition_sha256=reservation.composition_sha256,
            )
        except Exception:
            self.rollback(reservation.reservation_id)
            raise
        return self.commit(reservation.reservation_id)

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

        Returns a mapping of ``TenantID`` → admitted records decoded from the
        canonical reserved payloads (§7 compatibility decode — the payload
        bytes remain authoritative).
        """
        reservation = self.reserve_ingest(records, tenant_context=tenant_context)
        result = self.commit(reservation)
        del result
        return {
            batch.tenant_id: json.loads(batch.payload.decode("utf-8"))
            for batch in reservation.batches
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
