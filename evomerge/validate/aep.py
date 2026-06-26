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

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

from evomerge.validate.keystore import KeyNotFoundError, load_public_key


_SCHEMA_PATH = Path(__file__).parent.parent.parent / "schemas" / "aep-record.schema.json"


def _load_schema() -> dict:
    with open(_SCHEMA_PATH) as fh:
        return json.load(fh)


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

    @property
    def evidence_completeness(self) -> float:
        if self.state_changing_actions_total == 0:
            return 1.0
        return self.state_changing_actions_with_evidence / self.state_changing_actions_total

    @property
    def passed(self) -> bool:
        return self.valid_schema and len(self.errors) == 0


def _verify_signature(record: dict[str, Any]) -> list[str]:
    """Verify the Ed25519 signature block in a record.

    Expected structure (W1 contract):
        record["signature"] = {
            "alg": "ed25519",
            "key_id": "<string>",
            "sig": "<base64-encoded 64-byte signature>",
        }

    The signed payload is the canonical JSON of the record with the
    "signature" field removed, serialized with sorted keys, no spaces.

    Returns a list of error strings (empty means verification passed).
    """
    errors: list[str] = []
    sig_block = record.get("signature")
    if sig_block is None:
        errors.append("signature: missing 'signature' field in record")
        return errors

    alg = sig_block.get("alg")
    key_id = sig_block.get("key_id")
    sig_b64 = sig_block.get("sig")

    if alg != "ed25519":
        errors.append(f"signature: unsupported algorithm {alg!r}, expected 'ed25519'")
        return errors
    if not key_id:
        errors.append("signature: missing 'key_id' in signature block")
        return errors
    if not sig_b64:
        errors.append("signature: missing 'sig' in signature block")
        return errors

    # Load public key from keystore (env var)
    try:
        pubkey: Ed25519PublicKey = load_public_key(key_id)
    except KeyNotFoundError as exc:
        errors.append(f"signature: {exc}")
        return errors
    except ValueError as exc:
        errors.append(f"signature: key load error — {exc}")
        return errors

    # Decode the signature bytes
    try:
        # Accept both standard and URL-safe base64
        padding = (4 - len(sig_b64) % 4) % 4
        sig_bytes = base64.urlsafe_b64decode(sig_b64 + "=" * padding)
    except Exception as exc:
        errors.append(f"signature: cannot base64-decode 'sig': {exc}")
        return errors

    # Reconstruct signed payload: record minus the "signature" field, sorted keys
    payload_dict = {k: v for k, v in record.items() if k != "signature"}
    payload_bytes = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

    try:
        pubkey.verify(sig_bytes, payload_bytes)
    except InvalidSignature:
        errors.append("signature: Ed25519 signature verification failed")

    return errors


def validate_aep_record(
    record: dict[str, Any],
    require_signature: bool = False,
) -> AEPValidationResult:
    run_id = record.get("run_id", "<unknown>")
    errors: list[str] = []

    # Schema validation (jsonschema is a hard dependency — no fallback)
    valid_schema = True
    try:
        schema = _load_schema()
        jsonschema.validate(record, schema)
    except jsonschema.ValidationError as e:
        valid_schema = False
        errors.append(f"schema: {e.message}")

    # Signature verification
    if require_signature:
        sig_errors = _verify_signature(record)
        if sig_errors:
            errors.extend(sig_errors)

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

    return AEPValidationResult(
        run_id=run_id,
        valid_schema=valid_schema,
        has_model_id=bool(record.get("model_id")),
        has_actions=len(actions) > 0,
        has_verifier_results=len(record.get("verifier_results", [])) > 0,
        state_changing_actions_total=len(sc_actions),
        state_changing_actions_with_evidence=len(sc_with_evidence),
        errors=errors,
        has_causal_chain=any("parent_action_id" in a for a in actions),
        has_run_context="run_context" in record,
        v0_2_fields_count=v0_2_count,
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
