# Security policy

## Reporting

This is a small audit toolkit; we don't expect security-sensitive surface
area, but we take any report seriously.

If you find a vulnerability — for example, a way to make `convert()` or
`run_audit.py` execute arbitrary code from a malformed input file — please
**open an issue** on GitHub. Choose the "API behaviour issue" template
and label it `security`.

If you'd prefer not to disclose publicly first, please reach out via
the contact in `CITATION.cff` and we'll coordinate disclosure.

## Threat model

This toolkit:

- Reads JSON / JSONL files supplied by the user.
- Computes statistics on those files.
- Optionally writes results to disk.

It does **not**:

- Execute generated code from untrusted sources.
- Make network requests at import time.
- Load model weights or run inference.
- Require root / elevated permissions.

The most realistic concern is malformed input causing an unbounded
parse / memory blowup. We treat such bugs as ordinary correctness
issues, fixed under the regular PR cycle.

## Hardened validation (since 2026-06-26)

When the toolkit is used as the **trust gate** ahead of training-data
export, three classes of attacker-supplied inputs are explicitly
defended against. These hardenings came out of the 2026-06-26
technical review (see commit `060253f` and `tests/test_validator_strict.py`,
`tests/test_registry_tamper.py`, `tests/test_trust_score_security.py`,
`tests/test_contamination_unicode.py`).

**AEP record validation (`evomerge/validate/aep.py`):**

- A missing `jsonschema` import is now a **hard error** at validator
  load time, not a silent fallback to a two-field check. Previously,
  any environment where `jsonschema` failed to import (a malformed
  venv, a PEP 668 system-Python on some distros, a deliberately
  removed dependency) would let every AEP record pass.
- When `require_signature=True`, the validator calls
  `verify_aep_signature()` and rejects records whose Ed25519 signature
  cannot be verified against the public key resolved from
  `WASMAGENT_AEP_PUBKEY_<key_id>` (see `evomerge/validate/keystore.py`).
- Records missing required path / digest fields fail closed (no
  longer treated as "no validation needed").

**Registry tamper detection (`evomerge/registry.py`):**

- `registry/index.json` is itself signed; the signature is verified
  before any entry-level check.
- An append-only `events.jsonl` log records every add / delete /
  rollback. Replay-on-start compares the events against the current
  index — a mismatch (e.g. a deleted-but-not-logged entry, or a
  rollback of an attested record) is a hard fail.

**Trust-score integrity (`evomerge/trust_score.py`):**

- An empty / unattested dimension no longer scores `1.0`; it scores
  `unknown` (excluded from the geometric mean) or `0`. Tier A/B
  grades require a minimum number of attested dimensions, so a
  record claiming "no policy decisions" cannot lift its way to a
  passing grade by virtue of absence.
- `add_receipt()` recomputes the sha256 digest and rejects records
  whose receipt is "file exists, content tampered" — previously,
  digest verification was best-effort.
- The trust-score CLI now aggregates the entire JSONL stream by
  `(trace_id, run_id)`. The earlier first-record-wins behaviour
  allowed a benign first record to mask malicious later records.

**Contamination detection (`evomerge/validate/contamination.py`):**

- Token-level 8-gram Jaccard is applied after NFKC normalisation, so
  homoglyphs (e.g. Cyrillic `а` vs Latin `a`) and zero-width-character
  obfuscation no longer slip through.
- Falls back to character-level n-gram (n=5..10) Jaccard at
  threshold ≥ 0.7 when token-level produces no overlap (defeats
  whitespace-stripping evasion).
- The DPO training gate rejects unsigned verifier evidence; an
  `objective_score` outside `[0, 1]` or `NaN`, or any output matching
  the prompt-injection key strings in the `RISK_CORPUS` subset,
  triggers an audit-log line and rejection.

These checks compose: the validator hardening protects training-data
inputs, the registry log protects historical record integrity, the
trust score protects the gate decision, and contamination defends the
DPO pair. None of them is sufficient on its own.

## Versions

The toolkit is pre-1.0; we do not currently issue security updates for
older tags. The recommended fix for any reported issue is to update to
`main`.
