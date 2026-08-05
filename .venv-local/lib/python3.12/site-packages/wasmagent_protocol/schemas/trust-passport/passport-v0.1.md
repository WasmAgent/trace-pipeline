# Trust Passport v0.1 Specification

> Status: shipped v0.1 specification. The schema and reference validator are
> published in this repository; implementation maturity remains a research
> preview.

## What is a Trust Passport?

A Trust Passport is a signed trust-state artifact for an AI agent.

It summarizes evidence quality, open risks, audit references, validity period, renewal triggers, and revocation state (including revoked timestamp, reason, and revoking authority).

It supports technical due diligence, procurement review, and internal governance workflows.

Trust Passport is not a legal certification, security certification, or ISO/EU AI Act compliance attestation.

## Trust Passport answers

- When was this agent last audited?
- What is the audit report hash?
- What is the AgentBOM hash?
- Are there open risks?
- What is the evidence quality?
- Which frameworks have selected technical evidence support?
- When does this passport expire?
- What changes trigger renewal?
- Has this passport been revoked?

## Validity model

A Trust Passport has a validity period defined at issuance. The default validity period is 90 days.

### Renewal triggers

A passport should be renewed when any of the following occur before expiry:

- AgentBOM changes (new tools, permission changes, model update)
- New high or critical risk finding
- MCP posture drift detected
- Audit report updated
- Deployment context changes

### Revocation triggers

A passport is revoked when:

- Critical security finding is discovered after issuance
- Evidence is found to be falsified
- Agent is decommissioned
- Issuer determines the trust state is no longer valid

### Expiry vs. revocation

Expiry and revocation are distinct states with different semantics and consequences:

| Aspect | Expiry | Revocation |
|--------|---------|------------|
| **Cause** | Natural passage of time beyond `expires_at` | Explicit action due to a trigger event |
| **Reversibility** | Can be resolved by renewing the passport | Irreversible — a new passport must be issued from scratch |
| **Trust signal** | Passport is stale but was never invalid | Passport was valid at issuance but is now untrusted |
| **Recommended action** | Re-evaluate agent state and renew | Investigate the revocation trigger before any new issuance |

A passport that has both expired **and** been revoked should be treated as revoked — revocation takes precedence as the stronger trust signal.

### `isExpired()` implementation

The `isExpired()` function determines whether a passport's validity window has passed. It operates solely on the `validity.expires_at` timestamp and the current system time — it does not check revocation status.

```typescript
function isExpired(passport: { validity?: { expires_at?: string } }): boolean {
  const expiresAt = passport.validity?.expires_at;
  if (!expiresAt) return false;
  return new Date(expiresAt) < new Date();
}
```

Key behaviors:

- Returns `false` if `validity` or `expires_at` is missing (defensive — a malformed passport is not expired, merely invalid).
- Compares `expires_at` as an ISO 8601 date-time string against the current wall-clock time.
- Does **not** account for revocation — use a separate `isRevoked()` check (see revocation triggers above) and always check both independently when evaluating overall passport trustworthiness.

### Trustavo integration (future)

The Trustavo product (`trustavo.com/passport`) will automate lifecycle management of Trust Passports, including:

- Automated renewal workflows triggered by AgentBOM or posture drift detection
- Revocation propagation through a distributed status channel
- Configurable validity periods per issuer policy (overriding the 90-day default)

See [Future product home](#future-product-home) for the planned CLI surface.

## Schema structure

```
TrustPassport v0.1
├── identity         — passport ID, agent ID, issuance context
├── agentbom_ref     — hash reference to AgentBOM
├── audit_ref        — hash reference to audit report
├── posture_ref      — hash reference to MCP posture snapshot
├── evidence_summary — evidence quality and framework mapping
├── risk_summary     — open risk count by severity
├── validity         — issued_at, expires_at, renewal triggers
├── revocation       — revoked flag, revoked_at, revocation_reason, revoking_authority, revocation triggers
└── attestation      — issuer, signature
```

## CLI commands

```bash
agent-trust passport validate <path>    # Validate against schema
agent-trust passport inspect <path>     # Human-readable summary
```

## Future product home

Trust Passport will be productized in `open-agent-audit / Trustavo` at `trustavo.com/passport` once the schema and workflow stabilize.

Future CLI (in open-agent-audit):
```bash
open-agent-audit passport issue --report audit-report.json --agentbom agentbom.json --out trust-passport.json
open-agent-audit passport verify trust-passport.json
```
