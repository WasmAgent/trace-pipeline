# RFC 001: Multi-Party Trust Negotiation Protocol

> **Status:** draft — Milestone 8, Production Readiness & Enterprise Integration.
> **Tracking:** WasmAgent/agent-trust-infra#216.
> **Last updated:** 2026-07-20.

## 1. Abstract

This RFC defines a protocol for federated Trust Passport exchange between
organizations. It enables cross-org AI agent deployment through **mutual
attestation** and **policy reconciliation**: two parties each present their
trust state, verify the other's evidence, and agree on shared deployment policy
before an agent is permitted to operate across organizational boundaries.

The protocol is **message-based** and **asynchronous-friendly** — parties may
exchange messages over any transport (HTTPS, message queue, git-backed channel)
without requiring synchronous handshakes.

## 2. Motivation

The Trust Passport v0.1 specification ([passport-v0.1.md](passport-v0.1.md))
defines a single-issuer trust artifact. An agent operating within one
organization can carry a passport signed by that organization's issuer. However,
real-world deployments increasingly require agents to:

- **Cross organizational boundaries** — an agent built by Org A needs to access
  tools or data hosted by Org B.
- **Present verifiable trust state** — Org B must evaluate Org A's trust
  evidence against its own governance policy before granting access.
- **Negotiate policy** — each organization may have different minimum evidence
  thresholds, blocked tool categories, or compliance framework requirements.
  Deployment proceeds only when both parties' constraints are satisfied.
- **Maintain ongoing assurance** — trust state drifts over time; both parties
  need a mechanism to re-evaluate and renegotiate after initial agreement.

Without a negotiation protocol, federated agent deployment relies on ad-hoc
email-based review, manual evidence collection, and inconsistent policy
application. This RFC standardizes the exchange.

## 3. Relationship to existing specifications

| Spec | Role in negotiation |
|---|---|
| **Trust Passport v0.1** ([passport-v0.1.md](passport-v0.1.md), [schema.json](schema.json)) | The artifact being exchanged. Each party presents a signed Trust Passport as the primary evidence. |
| **Compliance Profile v0.1** ([../compliance-profile/schema.json](../compliance-profile/schema.json)) | Defines each party's governance rules. Policy reconciliation computes the intersection of both profiles. |
| **AgentBOM** ([../agentbom/](../agentbom/)) | Referenced by `agentbom_ref` inside the Trust Passport. Provides tool inventory and permission detail for policy evaluation. |
| **MCP Posture** ([../mcp-posture/](../mcp-posture/)) | Referenced by `posture_ref` inside the Trust Passport. Provides MCP attack-surface detail for cross-org deployment decisions. |

This RFC does **not** modify any existing schema. It defines a new
`NegotiationEnvelope` message format that wraps existing artifacts.

## 4. Glossary

| Term | Definition |
|---|---|
| **Initiator** | The organization proposing a cross-org deployment. Sends the first `NegotiationOffer`. |
| **Responder** | The organization evaluating the offer. Accepts, rejects, or counters with modified policy. |
| **NegotiationEnvelope** | The top-level message container carrying one step of a negotiation. |
| **Policy reconciliation** | The process of computing the union of constraints from both parties' compliance profiles. |
| **Deployment agreement** | The final signed artifact confirming both parties accept the reconciled policy and each other's trust evidence. |

## 5. Protocol overview

The negotiation follows a three-phase lifecycle:

```
Phase 1: DISCOVERY           Phase 2: NEGOTIATION          Phase 3: AGREEMENT
─────────────────            ─────────────────             ─────────────────
Initiator                     Initiator                      Both parties
  │                             │                              │
  ├─ NegotiationOffer ──────►  │                              │
  │  (passport + policy)        │                              │
  │                             ├─ NegotiationCounter ─────►   │
  │                             │  (counter-policy)            │
  │                             │                              │
  │                             ├─ NegotiationAccept ──────►   │
  │                             │                              ├─ DeploymentAgreement
  │                             │                              │  (signed by both)
  │                             │                              │
  └─────────────────────────────┴──────────────────────────────┘
```

### Phase 1: Discovery (offer)

The Initiator sends a `NegotiationOffer` containing:

- Its own signed Trust Passport (attesting to the agent's trust state).
- Its outgoing deployment policy (what it requires from the Responder).
- A nonce and timestamp for replay protection.

### Phase 2: Negotiation (counter / accept / reject)

The Responder evaluates the offer against its own compliance profile. Three
outcomes:

- **NegotiationAccept** — the Responder's policy is compatible. It attaches
  its own Trust Passport and policy, forming a two-sided agreement.
- **NegotiationCounter** — the Responder requires additional constraints. It
  proposes a modified policy. The Initiator may accept, counter again, or
  reject. The protocol allows up to `max_rounds` (default 5) counter-offers.
- **NegotiationReject** — the Responder cannot reconcile policies. It
  includes a human-readable reason and a list of unsatisfied constraints.

### Phase 3: Agreement

Both parties sign a `DeploymentAgreement` containing:

- The reconciled policy (intersection of both parties' constraints).
- Both Trust Passports with their signatures.
- Deployment scope (which tools, data, and environments are permitted).
- Validity period and renewal mechanism.

## 6. Message format

### 6.1 NegotiationEnvelope (top-level container)

Every message in the protocol is a `NegotiationEnvelope`:

```json
{
  "envelope_version": "0.1",
  "message_type": "NegotiationOffer | NegotiationCounter | NegotiationAccept | NegotiationReject",
  "negotiation_id": "uuid-v4",
  "round": 0,
  "timestamp": "2026-07-20T12:00:00Z",
  "initiator": {
    "org_id": "did:web:org-a.example.com",
    "contact": "trust@org-a.example.com"
  },
  "responder": {
    "org_id": "did:web:org-b.example.com",
    "contact": "trust@org-b.example.com"
  },
  "payload": { },
  "attestation": {
    "issuer": "did:web:org-a.example.com",
    "signature": "<ed25519-signature-of-payload>"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `envelope_version` | string | yes | Protocol version. Must be `"0.1"`. |
| `message_type` | enum | yes | One of the four message types. |
| `negotiation_id` | string (UUIDv4) | yes | Unique identifier for this negotiation session. Stable across all rounds. |
| `round` | integer ≥ 0 | yes | Round number. `0` = offer, `1` = first counter, etc. |
| `timestamp` | ISO 8601 | yes | When this message was created. |
| `initiator` | object | yes | Initiator identity. `org_id` is a DID-style identifier. |
| `responder` | object | yes | Responder identity. May be partially populated in the initial offer. |
| `payload` | object | yes | Message-type-specific payload (see §6.2–6.5). |
| `attestation` | object | yes | Sender's signature over the canonicalized payload. |

### 6.2 NegotiationOffer payload

```json
{
  "trust_passport": { },
  "deployment_policy": {
    "required_frameworks": ["eu-ai-act-annex-iv", "soc2-2024"],
    "min_evidence_quality": "medium",
    "max_unmitigated_critical": 0,
    "max_unmitigated_high": 2,
    "blocked_tool_permissions": ["filesystem:write"],
    "allowed_deployment_contexts": ["staging", "production"],
    "renewal_interval_days": 90
  },
  "agent_scope": {
    "agent_id": "agent-abc-123",
    "intended_tools": ["mcp://org-b.example.com/filesystem", "mcp://org-b.example.com/database"],
    "intended_environments": ["production"],
    "data_classification": ["internal", "confidential"]
  },
  "nonce": "random-64-char-hex-string"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `trust_passport` | TrustPassport v0.1 | yes | Initiator's signed Trust Passport. |
| `deployment_policy` | object | yes | Initiator's policy requirements (minimum evidence, blocked permissions, etc.). |
| `agent_scope` | object | yes | What the agent intends to access at the Responder's organization. |
| `nonce` | string | yes | Cryptographic nonce for replay protection. |

### 6.3 NegotiationCounter payload

```json
{
  "counter_policy": {
    "required_frameworks": ["eu-ai-act-annex-iv", "soc2-2024", "iso27001-2022"],
    "min_evidence_quality": "high",
    "max_unmitigated_critical": 0,
    "max_unmitigated_high": 1,
    "blocked_tool_permissions": ["filesystem:write", "network:external"],
    "allowed_deployment_contexts": ["staging"],
    "renewal_interval_days": 60
  },
  "trust_passport": { },
  "counter_reason": "Org B requires ISO 27001 mapping and restricts initial deployment to staging"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `counter_policy` | object | yes | Responder's proposed policy. |
| `trust_passport` | TrustPassport v0.1 | yes | Responder's own Trust Passport (for the Initiator to verify). |
| `counter_reason` | string | no | Human-readable explanation of the counter. |

### 6.4 NegotiationAccept payload

```json
{
  "agreed_policy": {
    "required_frameworks": ["eu-ai-act-annex-iv", "soc2-2024", "iso27001-2022"],
    "min_evidence_quality": "high",
    "max_unmitigated_critical": 0,
    "max_unmitigated_high": 1,
    "blocked_tool_permissions": ["filesystem:write", "network:external"],
    "allowed_deployment_contexts": ["staging"],
    "renewal_interval_days": 60
  },
  "initiator_passport": { },
  "responder_passport": { },
  "accepted_round": 1
}
```

### 6.5 NegotiationReject payload

```json
{
  "rejection_reason": "Policy irreconcilable: Org A requires production deployment but Org B mandates staging-only for external agents",
  "unsatisfied_constraints": [
    {
      "rule": "allowed_deployment_contexts",
      "initiator_value": ["staging", "production"],
      "responder_value": ["staging"],
      "conflict": "no intersection on 'production'"
    }
  ],
  "suggested_alternatives": ["Restrict agent to staging environment", "Org A to grant Org B tool-level attestation exception"]
}
```

## 7. Policy reconciliation algorithm

Given two deployment policies (P_initiator and P_responder), reconciliation
computes P_agreed — the **intersection** of constraints. A constraint is
reconciled as follows:

### 7.1 Scalar constraints (take the stricter value)

| Constraint | Reconciliation rule |
|---|---|
| `min_evidence_quality` | Take the higher quality threshold (`high > medium > low > insufficient`). If initiator requires `medium` and responder requires `high`, agreed = `high`. |
| `max_unmitigated_critical` | Take the lower value (strictest). If either party says 0, agreed = 0. |
| `max_unmitigated_high` | Take the lower value. |
| `renewal_interval_days` | Take the shorter interval (more frequent renewal). |

### 7.2 Set constraints (take the intersection)

| Constraint | Reconciliation rule |
|---|---|
| `required_frameworks` | Take the **union** — both parties' framework requirements must be met. |
| `blocked_tool_permissions` | Take the **union** — if either party blocks a permission, it is blocked in the agreement. |
| `allowed_deployment_contexts` | Take the **intersection** — only contexts allowed by both parties are permitted. |

### 7.3 Irreconcilable constraints

A constraint is irreconcilable when:

- `allowed_deployment_contexts` intersection is empty.
- `min_evidence_quality` cannot be satisfied by one party's Trust Passport.
- `required_frameworks` includes a framework that one party's Trust Passport does not map to (`coverage: "none"`).

When any constraint is irreconcilable, the protocol produces a
`NegotiationReject` with the specific unsatisfied constraints listed.

## 8. Deployment agreement

After a `NegotiationAccept`, both parties sign a `DeploymentAgreement`:

```json
{
  "agreement_version": "0.1",
  "agreement_id": "uuid-v4",
  "negotiation_id": "uuid-v4 (from negotiation)",
  "initiator": {
    "org_id": "did:web:org-a.example.com",
    "passport_hash": "sha256:<hash of initiator's passport>",
    "signature": "<ed25519>"
  },
  "responder": {
    "org_id": "did:web:org-b.example.com",
    "passport_hash": "sha256:<hash of responder's passport>",
    "signature": "<ed25519>"
  },
  "agreed_policy": { },
  "agent_scope": { },
  "validity": {
    "agreed_at": "2026-07-20T14:00:00Z",
    "expires_at": "2026-09-18T14:00:00Z",
    "renewal_trigger": "both passports renewed or policy change"
  },
  "status": "active"
}
```

### 8.1 Renewal and drift

The agreement has a validity period tied to the **shorter** of the two Trust
Passport expiry dates. If either passport expires or is revoked, the agreement
becomes `suspended`. Either party may trigger renegotiation by sending a new
`NegotiationOffer` referencing the original `negotiation_id`.

## 9. Security considerations

### 9.1 Replay protection

Every `NegotiationOffer` contains a `nonce`. Responders must track consumed
nonces for a window equal to the negotiation timeout (recommended: 24 hours).

### 9.2 Signature verification

All messages carry an `attestation` block with an Ed25519 signature over the
canonicalized payload (JSON with sorted keys, no whitespace). Receivers must:

1. Verify the signature against the sender's known public key (resolved via
   `org_id` DID document).
2. Verify the payload matches the canonical form used for signing.
3. Reject messages with expired timestamps (recommended: 1-hour clock skew).

### 9.3 Passport freshness

Negotiation parties should verify that the presented Trust Passport:
- Is not expired (`validity.expires_at` > now).
- Is not revoked (`revocation.revoked` = false).
- Has been issued within the last `renewal_interval_days` of the agreement.

### 9.4 Minimum disclosure

The protocol is designed for **minimum disclosure** — a party need only share
its Trust Passport with organizations it is actively negotiating with. The
`agent_scope` field limits what tool and environment access is being requested,
preventing over-broad authorization.

### 9.5 Transport security

Messages must be transmitted over TLS 1.3 or equivalent. The protocol itself
does not define a transport; implementers may use HTTPS APIs, SIGNED git commits,
or message queues with server-side encryption.

## 10. Transport binding (informative)

The protocol is transport-agnostic. Recommended bindings:

| Transport | When to use | Notes |
|---|---|---|
| **HTTPS REST** | Synchronous, low-volume negotiations | `POST /negotiation/{negotiation_id}` |
| **Signed git commits** | Auditability-critical workflows | Each message is a commit to a shared negotiation repository |
| **Message queue** | High-throughput, multi-party workflows | Each message is an event on a topic keyed by `negotiation_id` |
| **Email with S/MIME** | Human-in-the-loop workflows | For organizations requiring manual review before acceptance |

## 11. Extension points

### 11.1 Multi-party (N-way) negotiation

This RFC defines the pairwise (two-party) case. N-way negotiation (e.g.,
three organizations agreeing on shared agent deployment) is a natural
extension: the reconciliation algorithm generalizes to computing the
intersection of N policies. N-way negotiation is deferred to a future RFC.

### 11.2 Automated policy agents

An organization may deploy a policy agent that evaluates incoming offers against
its compliance profile and automatically responds with `NegotiationAccept`,
`NegotiationCounter`, or `NegotiationReject` without human intervention. The
message format supports this — `counter_reason` and `suggested_alternatives`
provide human-readable context when automated decisions need review.

### 11.3 Revocation propagation

When a Trust Passport is revoked mid-agreement, the revoking party should
proactively notify the counterparty via a transport-specific mechanism (webhook,
queue message, or signed git commit). The notified party must immediately
suspend the deployment agreement pending re-evaluation. A formal revocation
notification message type is deferred to a future RFC.

## 12. Worked example

### Scenario

Org A (`acme-corp.example.com`) wants to deploy their agent `data-analyst-v3`
to access Org B's (`global-fin.example.com`) financial data API. Both
organizations have Trust Passports and compliance profiles.

### Step 1: Org A sends NegotiationOffer

```json
{
  "envelope_version": "0.1",
  "message_type": "NegotiationOffer",
  "negotiation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "round": 0,
  "timestamp": "2026-07-20T10:00:00Z",
  "initiator": {
    "org_id": "did:web:acme-corp.example.com",
    "contact": "trust@acme-corp.example.com"
  },
  "responder": {
    "org_id": "did:web:global-fin.example.com"
  },
  "payload": {
    "trust_passport": {
      "passport_version": "0.1",
      "identity": {
        "passport_id": "tp-acme-2026-007",
        "agent_id": "data-analyst-v3",
        "agent_name": "Data Analyst v3",
        "issuer": "acme-corp.example.com",
        "issuance_context": "trustavo"
      },
      "evidence_summary": {
        "evidence_quality": "high",
        "framework_mappings": [
          { "framework": "soc2-2024", "coverage": "selected_technical_evidence" },
          { "framework": "eu-ai-act-annex-iv", "coverage": "partial" }
        ]
      },
      "risk_summary": { "critical": 0, "high": 1, "medium": 3, "low": 5 },
      "validity": {
        "issued_at": "2026-06-01T00:00:00Z",
        "expires_at": "2026-08-30T00:00:00Z"
      },
      "revocation": { "revoked": false },
      "attestation": {
        "issuer": "acme-corp.example.com",
        "passport_hash": "sha256:abc123...",
        "signature": "<ed25519>"
      }
    },
    "deployment_policy": {
      "required_frameworks": ["soc2-2024"],
      "min_evidence_quality": "medium",
      "max_unmitigated_critical": 0,
      "blocked_tool_permissions": ["filesystem:write"],
      "allowed_deployment_contexts": ["staging", "production"],
      "renewal_interval_days": 90
    },
    "agent_scope": {
      "agent_id": "data-analyst-v3",
      "intended_tools": ["mcp://global-fin.example.com/financial-data-api"],
      "intended_environments": ["production"],
      "data_classification": ["confidential"]
    },
    "nonce": "f47ac10b58cc4372a5670e02b2c3d479"
  }
}
```

### Step 2: Org B responds with NegotiationCounter

Org B's compliance profile requires ISO 27001 and restricts external agents to
staging initially:

```json
{
  "envelope_version": "0.1",
  "message_type": "NegotiationCounter",
  "negotiation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "round": 1,
  "timestamp": "2026-07-20T10:15:00Z",
  "initiator": {
    "org_id": "did:web:acme-corp.example.com"
  },
  "responder": {
    "org_id": "did:web:global-fin.example.com",
    "contact": "trust@global-fin.example.com"
  },
  "payload": {
    "counter_policy": {
      "required_frameworks": ["soc2-2024", "iso27001-2022"],
      "min_evidence_quality": "high",
      "max_unmitigated_critical": 0,
      "max_unmitigated_high": 1,
      "blocked_tool_permissions": ["filesystem:write", "network:external"],
      "allowed_deployment_contexts": ["staging"],
      "renewal_interval_days": 60
    },
    "trust_passport": {
      "passport_version": "0.1",
      "identity": {
        "passport_id": "tp-gfin-2026-012",
        "agent_id": "financial-data-gateway",
        "agent_name": "Financial Data Gateway",
        "issuer": "global-fin.example.com",
        "issuance_context": "trustavo"
      },
      "evidence_summary": {
        "evidence_quality": "high",
        "framework_mappings": [
          { "framework": "iso27001-2022", "coverage": "selected_technical_evidence" },
          { "framework": "soc2-2024", "coverage": "selected_technical_evidence" }
        ]
      },
      "risk_summary": { "critical": 0, "high": 0, "medium": 2, "low": 4 },
      "validity": {
        "issued_at": "2026-05-15T00:00:00Z",
        "expires_at": "2026-08-13T00:00:00Z"
      },
      "revocation": { "revoked": false },
      "attestation": {
        "issuer": "global-fin.example.com",
        "passport_hash": "sha256:def456...",
        "signature": "<ed25519>"
      }
    },
    "counter_reason": "Global Financial requires ISO 27001 attestation and initial staging-only deployment for external agents. Production access requires 30-day staging observation period."
  }
}
```

### Step 3: Org A accepts

Org A's agent meets the `high` evidence quality threshold and has no
unmitigated critical risks. Org A agrees to staging deployment with a 60-day
renewal cycle.

### Step 4: Both parties sign DeploymentAgreement

The agreement locks in the reconciled policy and both passports. It expires on
2026-08-13 (the shorter of the two passport expiry dates). Both parties store
the agreement and begin the staging deployment.

## 13. Implementation notes

This RFC is a **specification-layer deliverable**. Reference implementation
belongs in `open-agent-audit` (`@openagentaudit/negotiation`), leveraging:

- `@wasmagent/aep` for evidence emission and signing.
- `@wasmagent/mcp-attestation` for capability attestation.
- `@wasmagent/mcp-gateway` for policy enforcement at runtime.

The `trust-passport-core` and `trust-runtime` packages in **this** repository are
frozen and will not be extended. Validators for the `NegotiationEnvelope`
message format should be added to this repository's spec layer once the schema
stabilizes.

## 14. Revision history

| Version | Date | Author | Change |
|---|---|---|---|
| 0.1-draft | 2026-07-20 | claude-bot (auto) | Initial draft — Milestone 8 deliverable |
