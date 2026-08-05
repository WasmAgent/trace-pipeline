# MCP Posture Model v0.1

> Status: shipped v0.1 specification. The schema and reference validator are
> published in this repository; implementation maturity remains a research
> preview.

## What is MCP Posture?

MCP Posture Management answers:

- Which MCP servers is this agent connected to?
- Which tools does each server expose?
- What permissions do those tools require?
- Which tools carry high-risk signals (SSRF, exfiltration, command execution, privilege escalation)?
- Has the permission surface changed since the last snapshot?
- Which findings should flow into audit reports?

MCP Posture is not a simple scan result. It is a continuous posture state with historical tracking.

## Posture pipeline

```
MCP server discovery
        ↓
Tool and permission classification
        ↓
Risk taxonomy mapping
        ↓
Permission graph
        ↓
Posture snapshot (this schema)
        ↓
Historical posture state (Trustavo)
        ↓
Audit evidence integration
```

## Schema structure

```
MCPPosture v0.1
├── posture_version        — schema version (always "0.1")
├── protocol_version       — MCP spec version (e.g. "2026-07-28"; optional, omit for pre-2026-07-28)
├── identity               — snapshot ID, agent ID, timestamp
├── servers                — connected MCP servers
│   ├── session_model      — "stateful" | "stateless-handle" | "unknown" (MCP 2026-07-28+)
│   ├── handle_expiry_policy — handle lifetime policy (stateless-handle model only)
│   └── tools              — tools per server with permissions and risk classification
├── permission_graph       — aggregate permission surface
├── risk_summary           — taxonomy-mapped risk findings (with owasp_agentic_ref)
├── drift                  — changes since previous snapshot
└── attestation            — generator, snapshot hash, and OAuth validation state
    └── auth               — audience-bound token, PKCE, per-client consent status
```

## MCP 2026-07-28 compatibility notes

The MCP 2026-07-28 specification introduced a **stateless/handle-based architecture** replacing the previous stateful session model. Key schema additions:

-  distinguishes snapshots taken under the new spec
-  records which architecture the server uses
-  captures OAuth 2.0 resource server validation state (audience-bound token, PKCE, per-client consent) as required by Microsoft MCP security guidance
-  now includes  to capture sensitive data accidentally mapped into the new  /  headers

## Risk taxonomy

| Category | Description | OWASP MCP reference | OWASP Agentic ref (2026) |
|---|---|---|---|
|  | Server-side request forgery via network tools | MCP-02 | ASI05 |
|  | Data exfiltration via output or storage tools | MCP-04 | ASI03 |
|  | Arbitrary command or code execution | MCP-01 | ASI02 |
|  | Permission scope expansion | MCP-03 | ASI06 |
|  | Tool input that can manipulate agent behavior | MCP-05 | ASI01 |
|  | Access to secrets or credentials | MCP-06 | ASI03 |
|  | Unverified MCP server provenance | MCP-07 | ASI01 |
|  | Sensitive data in MCP-Method/MCP-Name headers | — | ASI03 |

> OWASP Agentic Applications Top 10 (2026) mapping is approximate. ASI01 = Agent Goal Hijack / Tool Poisoning; ASI02 = Unsafe Code Execution; ASI03 = Data Exfiltration; ASI05 = SSRF; ASI06 = Privilege Escalation.

## CLI commands

```bash
agent-trust mcp-posture validate <path>    # Validate against schema
agent-trust mcp-posture inspect <path>     # Human-readable summary
agent-trust mcp-posture diff <old> <new>   # Show posture drift
```

## Standalone product viability

MCP Posture is designed to operate as an independent MCP security product. The
package `@wasmagent/mcp-posture-core` ships a schema, validator, inspector, and
diff engine with zero external runtime dependencies — it can be consumed outside
the agent-trust-infra monorepo by any MCP security tooling.

### What this package owns

| Capability | Export | Purpose |
|---|---|---|
| Schema validation | `validateMCPPosture()` | Structural conformance check for posture snapshots |
| Human-readable inspection | `inspectMCPPosture()` | Summarize servers, tools, risk findings, and permission surface |
| Drift detection | `diffMCPPosture()` / `formatPostureDiff()` | Compare two posture snapshots and report server, tool, permission, and risk changes |
| Risk taxonomy constants | `RISK_CATEGORIES` | Enum of 8 canonical risk categories for programmatic use |
| Type definitions | `PostureDiff`, `RiskCategory`, `SessionModel`, etc. | TypeScript types for posture data structures |

### Boundary with downstream packages

- **Runtime MCP scanning** (server discovery, live traffic analysis) →
  [`@wasmagent/mcp-attestation`](https://github.com/WasmAgent/wasmagent-js) in
  `wasmagent-js`. This package validates *snapshots*; it does not perform live
  scanning.
- **Audit report generation** (framework mapping, evidence pipelining) →
  [`open-agent-audit`](https://github.com/WasmAgent/open-agent-audit). Posture
  snapshots feed into audit reports but report rendering is downstream.
- **Trust Passport issuance / renewal** → `open-agent-audit` / Trustavo. The
  posture snapshot is referenced by a Trust Passport but passport lifecycle is
  managed separately.

### Extraction criteria

MCP Posture can be extracted into a standalone repository when:

1. The schema version (`posture_version`) has stabilized and no breaking changes
   are expected in the short term.
2. External consumers (beyond the WasmAgent ecosystem) adopt the package
   independently.
3. The risk taxonomy has coverage across OWASP Agentic Top 10 (ASI01–ASI10).

Until extraction, the canonical source is this repository. The JSON schema in
`specs/mcp-posture/schema.json` and the reference implementation in
`packages/mcp-posture-core/` are versioned together.
