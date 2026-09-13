"""AEP (Agent Evidence Protocol) record validation.

Validates AEP records against the JSON schema and checks evidence completeness.
Optionally verifies Ed25519 signatures on records when require_signature=True.

Design rules
- jsonschema is a hard dependency; ImportError is never silently swallowed.
- Signature verification uses the keystore to load keys from env vars.
  An unknown key_id or an invalid signature is always a verification failure.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError as _exc:
    raise ImportError(
        "jsonschema is required for AEP validation. "
        "Install it with: pip install jsonschema"
    ) from _exc

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from wasmagent_protocol import get_schema

from evomerge.validate.keystore import KeyNotFoundError, load_public_key

_AEP_PAYLOAD_TYPE = "application/vnd.in-toto+json"
_AEP_PREDICATE_TYPE = "https://wasmagent.dev/attestations/aep/v0.4"
_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"


@dataclass
class AuthenticityResult:
    """Layered authenticity verdict (DSSE is the only supported profile)."""

    valid: bool
    mode: str  # dsse-valid | unsigned | unsupported-legacy | invalid | not-checked
    binding: str = "not-applicable"  # exact | invalid | not-applicable
    detail: str = ""


def _load_schema() -> dict:
    return get_schema("aep-record")


@dataclass
class AEPValidationResult:
    run_id: str
    valid_schema: bool
    has_model_id: bool
    has_actions: bool
    has_verifier_results: bool
    state_changing_actions_with_evidence: int
    state_changing_actions_total: int
    errors: list[str] = field(default_factory=list)
    has_causal_chain: bool = False
    has_run_context: bool = False
    v0_2_fields_count: int = 0
    # v0.3 top-level fields
    has_recording_mode: bool = False
    has_side_effect_class: bool = False
    v0_3_fields_count: int = 0
    # v0.5 attribution grading (canonical wasmagent-protocol 0.1.9):
    # machine-checkable statement of what backs the human attribution.
    attribution: dict[str, Any] | None = None
    v0_5_fields_count: int = 0
    # Layered verdicts (wasmagent-protocol#214): protocol validity, derived
    # semantics, local admission policy, and authenticity are distinct —
    # a boolean "valid" must never erase an assurance state.
    protocol_schema_valid: bool | None = None
    semantic_valid: bool | None = None
    security_policy_valid: bool | None = None
    authenticity_mode: str | None = None
    authenticity_valid: bool | None = None

    @property
    def evidence_completeness(self) -> float:
        if self.state_changing_actions_total == 0:
            return 1.0
        return self.state_changing_actions_with_evidence / self.state_changing_actions_total

    @property
    def passed(self) -> bool:
        return self.valid_schema and len(self.errors) == 0


def _pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE 1.0.2 §2 PAE over DECODED serialized body bytes (not base64 text)."""
    pt = payload_type.encode("utf-8")
    return (
        b"DSSEv1 "
        + str(len(pt)).encode("ascii")
        + b" "
        + pt
        + b" "
        + str(len(payload)).encode("ascii")
        + b" "
        + payload
    )


def _decode_base64_either(input_b64: str) -> bytes:
    """Decode base64 accepting both alphabets, as DSSE 1.0.2 requires ("Either
    standard or URL-safe base64 encodings are allowed ... verifiers MUST accept
    either").

    Strategy: normalize the URL-safe alphabet onto the standard one ('-'→'+',
    '_'→'/'), re-pad, and decode with the strict standard decoder. The mapping
    is a bijection, so decoding is unique for standard, URL-safe, and mixed
    inputs alike — matching the Node and Rust decoders byte-for-byte.
    """
    normalized = input_b64.replace("-", "+").replace("_", "/")
    padding = (4 - len(normalized) % 4) % 4
    return base64.b64decode(normalized + "=" * padding, validate=True)


def _decode_dsse_payload(payload_b64: str) -> bytes:
    try:
        return _decode_base64_either(payload_b64)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("invalid DSSE payload encoding") from exc


def verify_aep_authenticity(record: dict[str, Any]) -> AuthenticityResult:
    """DSSE-only authenticity dispatcher.

    The retired inline-signature construction (Ed25519 over canonical JSON)
    is unsupported: records carrying a signature block without a
    dsse_envelope report mode='unsupported-legacy', never 'valid'.
    """
    envelope = record.get("dsse_envelope")
    if not isinstance(envelope, dict):
        if record.get("signature") is not None:
            return AuthenticityResult(
                valid=False,
                mode="unsupported-legacy",
                detail="legacy inline signature construction is retired (DSSE only)",
            )
        return AuthenticityResult(valid=False, mode="unsigned")

    sigs = envelope.get("signatures")
    if not isinstance(sigs, list) or len(sigs) != 1:
        return AuthenticityResult(
            valid=False,
            mode="invalid",
            detail=f"envelope must carry exactly one signature, got {len(sigs) if isinstance(sigs, list) else 'non-list'}",
        )
    sig_entry = sigs[0] or {}
    key_id = sig_entry.get("keyid") or sig_entry.get("key_id")
    sig_b64 = sig_entry.get("sig")
    payload_type = envelope.get("payloadType")
    payload_b64 = envelope.get("payload")
    if payload_type != _AEP_PAYLOAD_TYPE:
        return AuthenticityResult(False, "invalid", "not-applicable", "payload type mismatch")
    if not key_id or not sig_b64 or not isinstance(payload_b64, str):
        return AuthenticityResult(False, "invalid", "not-applicable", "malformed envelope")

    try:
        pubkey: Ed25519PublicKey = load_public_key(key_id)
    except KeyNotFoundError as exc:
        return AuthenticityResult(False, "invalid", "not-applicable", f"keystore: {exc}")
    except ValueError as exc:
        return AuthenticityResult(False, "invalid", "not-applicable", f"key load error: {exc}")

    try:
        # Accept either alphabet, like the payload decoder (normalize URL-safe
        # onto standard; unique decoding for mixed inputs too).
        sig_bytes = _decode_base64_either(sig_b64)
    except Exception as exc:  # noqa: BLE001
        return AuthenticityResult(False, "invalid", "not-applicable", f"sig decode: {exc}")

    try:
        payload_bytes = _decode_dsse_payload(payload_b64)
    except Exception as exc:  # noqa: BLE001
        return AuthenticityResult(False, "invalid", "not-applicable", f"payload decode: {exc}")

    try:
        pubkey.verify(sig_bytes, _pae(payload_type, payload_bytes))
    except InvalidSignature:
        return AuthenticityResult(False, "invalid", "not-applicable", "PAE signature verification failed")

    try:
        statement = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return AuthenticityResult(False, "invalid", "not-applicable", f"payload not JSON: {exc}")
    if statement.get("predicateType") != _AEP_PREDICATE_TYPE:
        return AuthenticityResult(False, "invalid", "invalid", "predicateType mismatch")
    if statement.get("_type") != _STATEMENT_TYPE:
        return AuthenticityResult(False, "invalid", "invalid", "statement _type mismatch")

    run_id = record.get("run_id")
    subjects = statement.get("subject")
    if not isinstance(subjects, list) or len(subjects) != 1:
        return AuthenticityResult(False, "invalid", "invalid", "statement must carry exactly one subject")
    subject = subjects[0] or {}
    subject_name = subject.get("name")
    if subject_name != f"urn:wasmagent:run:{run_id}":
        return AuthenticityResult(False, "invalid", "invalid", "subject name does not match run_id")

    unsigned = {k: v for k, v in record.items() if k not in ("signature", "dsse_envelope", "timestamp_proof")}
    subject_digest = (subject.get("digest") or {}).get("sha256")
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if subject_digest != __import__("hashlib").sha256(canonical).hexdigest():
        return AuthenticityResult(False, "invalid", "invalid", "subject digest does not bind the record")
    if statement.get("predicate") != unsigned:
        return AuthenticityResult(False, "invalid", "invalid", "predicate does not match the record")

    return AuthenticityResult(valid=True, mode="dsse-valid", binding="exact")


def validate_aep_record(
    record: dict[str, Any],
    require_signature: bool = False,
) -> AEPValidationResult:
    run_id = record.get("run_id", "<unknown>")
    errors: list[str] = []

    # Schema validation (jsonschema is a hard dependency — no fallback).
    # Catches every exception, not just ValidationError: hostile inputs can
    # crash the validator itself (RecursionError on deeply nested structures,
    # TypeError on wrong-typed containers) — a crash here turns one bad line
    # into a whole-pipeline outage, which is exactly what an attacker wants.
    valid_schema = True
    try:
        schema = _load_schema()
        jsonschema.validate(record, schema)
    except jsonschema.ValidationError as e:
        valid_schema = False
        errors.append(f"schema: {e.message}")
    except Exception as e:  # noqa: BLE001 - validation must survive hostile input
        valid_schema = False
        errors.append(f"schema: internal validation failure ({type(e).__name__})")

    # Hostile-key admission policy: a literal "__proto__" key is inert in
    # Python but is a prototype-pollution vector for JS consumers of the
    # same record. This is a LOCAL ADMISSION decision — the canonical schema
    # deliberately keeps the field open — so it must not be reported as a
    # protocol schema violation.
    security_policy_valid = True
    if isinstance(record, dict) and "__proto__" in record:
        security_policy_valid = False
        errors.append("policy: record contains forbidden key '__proto__' (JS prototype-pollution vector)")

    # aep/v0.5 floor consistency: the floor MUST be the weakest grade in
    # `run_attribution_backing_observed` — a floor that omits or exceeds an
    # observed grade masks weak authorizations inside a strong-looking one
    # (the exact masking the floor exists to prevent).
    # `_BACKING_ORDER` is the canonical rank (weakest first); the observed
    # list is set-semantics, so ranking must never use array positions.
    # FAIL CLOSED, symmetric pair-presence (the canonical descriptions say the
    # fields ship together — "reported alongside, never instead of"):
    #   floor present + observed absent/empty  → semantic error
    #   observed present + floor absent        → semantic error (itemized list
    #     without a floor permits exactly the masking the floor exists to stop)
    #   observed present + empty (no floor)    → semantic error (empty claim)
    _BACKING_ORDER = [
        "unknown",
        "operator_asserted",
        "principal_key_signed",
        "qualified_signature",
    ]
    _BACKING_RANK = {grade: i for i, grade in enumerate(_BACKING_ORDER)}
    floor = record.get("run_attribution_backing_floor")
    observed = record.get("run_attribution_backing_observed")
    observed_nonempty = isinstance(observed, list) and len(observed) > 0
    if floor is not None and not observed_nonempty:
        errors.append(
            "attribution: floor provided without a non-empty observed set — "
            "the floor cannot be verified against the weakest-grade rule"
        )
    if observed is not None:
        if not observed_nonempty:
            errors.append(
                "attribution: run_attribution_backing_observed was provided as an "
                "empty set — an empty grading claim is not a valid record"
            )
        elif floor is None:
            errors.append(
                "attribution: run_attribution_backing_observed was provided without "
                "run_attribution_backing_floor — the pair ships together, never "
                "instead of each other"
            )
    if floor is not None and observed_nonempty:
        known = all(g in _BACKING_RANK for g in observed) and floor in _BACKING_RANK
        if not known:
            errors.append(
                "attribution: backing grade outside the canonical vocabulary"
            )
        elif floor not in observed:
            errors.append(
                "attribution: run_attribution_backing_floor is not present in "
                "run_attribution_backing_observed"
            )
        elif _BACKING_RANK[floor] != min(_BACKING_RANK[g] for g in observed):
            errors.append(
                "attribution: run_attribution_backing_floor is not the weakest "
                "grade in run_attribution_backing_observed"
            )

    # Authenticity verification — DSSE-only dispatcher (no legacy fallback).
    authenticity_mode: str | None = None
    authenticity_valid: bool | None = None
    if require_signature:
        ar = verify_aep_authenticity(record)
        authenticity_mode = ar.mode
        authenticity_valid = ar.valid
        if not ar.valid:
            errors.append(f"authenticity: {ar.mode} — {ar.detail}")

    actions = record.get("actions", [])
    sc_actions = [a for a in actions if a.get("state_changing")]
    sc_with_evidence = [a for a in sc_actions if a.get("result_digest") or a.get("evidence_refs")]

    _V0_2_CAUSAL_FIELDS = [
        "parent_action_id", "causal_chain_id", "scope_lease_id",
        "input_taint_labels", "memory_read_refs",
    ]
    v0_2_count = sum(
        1 for a in actions for f in _V0_2_CAUSAL_FIELDS if f in a
    )

    # v0.3 top-level fields
    _V0_3_FIELDS = [
        "recording_mode", "side_effect_class", "run_side_effect_class_max",
        "user_id", "subject_id",
    ]
    v0_3_count = sum(1 for f in _V0_3_FIELDS if f in record)

    # aep/v0.5 attribution grading: who the run acted for, who authorized it,
    # and how verifiably that authorization is backed. Absent = the producer
    # makes no claim (distinct from 'unknown' = observed but ungradeable).
    _V0_5_FIELDS = [
        "authorized_by", "authority_origin", "identity_source",
        "attribution_backing", "run_attribution_backing_floor",
        "run_attribution_backing_observed",
    ]
    v0_5_fields = {f: record[f] for f in _V0_5_FIELDS if f in record}
    v0_5_count = len(v0_5_fields)
    attribution: dict[str, Any] | None = None
    if v0_5_count > 0:
        attribution = {"user_id": record["user_id"]} if "user_id" in record else {}
        attribution.update(v0_5_fields)

    return AEPValidationResult(
        run_id=run_id,
        valid_schema=valid_schema,
        protocol_schema_valid=valid_schema,
        semantic_valid=not any(e.startswith("attribution:") for e in errors),
        security_policy_valid=security_policy_valid,
        authenticity_mode=authenticity_mode,
        authenticity_valid=authenticity_valid,
        has_model_id=bool(record.get("model_id")),
        has_actions=len(actions) > 0,
        has_verifier_results=len(record.get("verifier_results", [])) > 0,
        state_changing_actions_total=len(sc_actions),
        state_changing_actions_with_evidence=len(sc_with_evidence),
        errors=errors,
        has_causal_chain=any("parent_action_id" in a for a in actions),
        has_run_context="run_context" in record,
        v0_2_fields_count=v0_2_count,
        has_recording_mode="recording_mode" in record,
        has_side_effect_class="side_effect_class" in record,
        v0_3_fields_count=v0_3_count,
        attribution=attribution,
        v0_5_fields_count=v0_5_count,
    )


def validate_aep_file(
    path: Path,
    require_signature: bool = False,
) -> list[AEPValidationResult]:
    results = []
    with open(path) as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                results.append(AEPValidationResult(
                    run_id=f"line-{i}",
                    valid_schema=False,
                    has_model_id=False,
                    has_actions=False,
                    has_verifier_results=False,
                    state_changing_actions_with_evidence=0,
                    state_changing_actions_total=0,
                    errors=[f"JSON parse error: {e}"],
                ))
                continue
            except RecursionError:
                # A hostile line can nest deeply enough to blow the parser's
                # stack — treat it as one invalid record instead of crashing
                # the whole validation run.
                results.append(AEPValidationResult(
                    run_id=f"line-{i}",
                    valid_schema=False,
                    has_model_id=False,
                    has_actions=False,
                    has_verifier_results=False,
                    state_changing_actions_with_evidence=0,
                    state_changing_actions_total=0,
                    errors=["JSON parse error: nesting too deep (RecursionError)"],
                ))
                continue
            results.append(validate_aep_record(record, require_signature=require_signature))
    return results


def print_aep_report(results: list[AEPValidationResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"AEP validation: {passed}/{total} passed")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        ec = f"{r.evidence_completeness:.0%} evidence"
        print(f"  [{status}] {r.run_id} — {ec}")
        for err in r.errors:
            print(f"         error: {err}")
