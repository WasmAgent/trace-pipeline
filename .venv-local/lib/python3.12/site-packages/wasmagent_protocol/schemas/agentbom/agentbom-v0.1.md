# AgentBOM v0.1 Specification

> Status: shipped v0.1 specification. The schema and reference validator are
> published in this repository; implementation maturity remains a research
> preview.

## What is AgentBOM?

AgentBOM is a bill of materials for AI agents.

It describes the deployed composition of an agent, including model dependencies, MCP servers, tool surfaces, prompt references, permission scopes, data access boundaries, evidence references, and known risk signals.

AgentBOM is not an audit report. It is an input artifact for audit, posture analysis, procurement review, and trust passport issuance.

## Relationship to existing standards

### SBOM (CycloneDX, SPDX)

Software Bill of Materials standards such as CycloneDX and SPDX catalog the libraries, frameworks, and dependencies that make up a software artifact. They answer the question "what components were shipped?" and are essential for vulnerability tracking and supply-chain integrity.

AgentBOM does not replace an SBOM. Instead, it extends the bill-of-materials concept into the operational layer of an AI agent. Where an SBOM lists static software components, AgentBOM captures **runtime authority surfaces**: which tools the agent can invoke, what permission scopes it holds, which data sources it may access, and what prompts govern its behavior. These dimensions are outside the scope of traditional SBOM formats, which have no fields for tool registrations, permission boundaries, or prompt provenance.

### AIBOM

AIBOM (AI Bill of Materials) initiatives focus on model lineage and dataset provenance — tracking which model weights, training data, and fine-tuning steps produced a given AI capability. This is critical for understanding model-level risks such as data poisoning, bias, and license compliance.

AgentBOM builds on the same bill-of-materials philosophy but shifts focus from the model itself to the **agent wrapper around the model**. An agent that calls a well-documented model can still introduce risk through overly broad tool permissions, unbounded data access, or insufficient prompt guardrails. AgentBOM captures these agent-level concerns — tool registries, permission scopes, prompt hashes, and runtime evidence — that AIBOM alone does not address.

### OWASP LLM Top 10

The OWASP LLM Top 10 catalogs the most critical security risks specific to large language model applications, including prompt injection, excessive agency, and data leakage. It serves as a risk awareness and mitigation guide.

AgentBOM does not duplicate the OWASP LLM Top 10 taxonomy. Rather, it provides a **structured, machine-readable artifact** that records whether a given agent deployment has mitigations and findings relevant to those risk categories. The `risk_layer` and `tool_layer.risk_signals` fields can reference OWASP LLM Top 10 categories (e.g., `prompt_injection`, `excessive_agency`), enabling automated tooling to check whether a deployment has acknowledged and addressed the applicable risks.

### What AgentBOM adds

The following capabilities are not captured by SBOM, AIBOM, or OWASP LLM Top 10 alone:

- **Tool registry and permissions**: A complete inventory of tools (MCP servers, built-in functions, plugins) alongside the permission scopes each tool requires.
- **Tool skills**: Declared skills or capabilities each tool contributes, enabling capability-based analysis of agent behavior.
- **Prompt provenance**: Cryptographic hashes of system prompts and template references, enabling integrity verification of the instructions governing agent behavior. Version tracking for prompt templates supports change auditing.
- **Permission boundaries**: Declared data access scopes, credential type references, and granted authority — the "blast radius" if the agent behaves unexpectedly.
- **Workflow definitions**: Action pathway definitions describing sequences of steps, tool invocations, and decisions that the agent can execute.
- **Runtime evidence links**: References to AEP (Agent Evidence Protocol) events and evidence hashes that ground the AgentBOM in observed runtime behavior rather than declared intent alone.
- **Composability**: AgentBOM is designed to be diffed between versions, making it suitable for change-review workflows and continuous compliance monitoring.

### AgentBOM as input to audit

AgentBOM is an input artifact, not an audit report itself. In a typical audit workflow:

1. **Generation**: An AgentBOM is produced for each agent deployment, capturing the full composition at a point in time.
2. **Posture analysis**: Tools such as MCP Posture analyze the AgentBOM against policy rules and risk frameworks, producing findings.
3. **Evidence collection**: Runtime evidence (AEP events, invocation logs) is linked into the `evidence_layer`, grounding the static declaration in observed behavior.
4. **Audit review**: Auditors use the AgentBOM alongside posture findings and evidence to assess compliance, identify gaps, and verify mitigations.

### AgentBOM and the Trust Passport

The Trust Passport is a downstream artifact that summarizes the trust status of an agent for consumers such as procurement teams and platform operators. AgentBOM feeds into Trust Passport issuance by providing:

- **Identity and scope**: The agent's declared identity, version, and deployment context from `identity`.
- **Risk posture**: Aggregated risk signals from `risk_layer` and `tool_layer.risk_signals` that determine the passport's risk rating.
- **Evidence integrity**: Cryptographic hashes and AEP references from `evidence_layer` that allow the passport to make attested claims about runtime behavior.

Together, AgentBOM provides the detailed technical input while the Trust Passport provides the concise, consumer-facing trust summary.

## Schema structure

```
AgentBOM v0.1
├── identity         — agent ID, name, version, deployment context
├── model_layer      — model provider, model ID, version, capabilities
├── tool_layer       — registered tools, MCP servers, tool permissions, skills
├── prompt_layer     — system prompt references, template IDs, prompt version tracking
├── permission_layer — granted scopes, data access boundaries, credential references
├── policy_definitions — governance policies, constraints, and compliance rules
├── evidence_layer   — AEP event references, runtime evidence hashes
├── audit_log        — structured audit trail entries
├── risk_layer       — known risk signals, open findings
├── workflow_layer   — action pathway definitions (workflows, pipelines, decision sequences)
└── attestation      — generator, timestamp, hash
```

## identity

| Field | Type | Required | Description |
|---|---|---|---|
| `agentbom_version` | string | yes | Always `"0.1"` |
| `agent_id` | string | yes | Unique agent identifier |
| `agent_name` | string | yes | Human-readable name |
| `agent_version` | string | no | Semantic version |
| `deployment_context` | string | no | `development`, `staging`, `production` |
| `generated_at` | ISO 8601 | yes | Generation timestamp |

## model_layer

| Field | Type | Required | Description |
|---|---|---|---|
| `provider` | string | yes | Model provider |
| `model_id` | string | yes | Model identifier |
| `model_version` | string | no | Model version or snapshot |
| `capabilities` | string[] | no | Declared capabilities |

## tool_layer

Array of tool entries:

| Field | Type | Required | Description |
|---|---|---|---|
| `tool_id` | string | yes | Unique tool identifier |
| `tool_name` | string | yes | Tool name |
| `source` | string | yes | `mcp`, `builtin`, `plugin` |
| `mcp_server_id` | string | no | MCP server identifier if source is `mcp` |
| `skills` | string[] | no | Skills or capabilities this tool contributes to the agent |
| `permissions` | string[] | no | Permission scopes this tool requires |
| `risk_signals` | string[] | no | Known risk signals for this tool |

### skills

The `skills` field lists declarative skills or capabilities that the tool provides to the agent. Unlike `permissions` (which describe what the tool is allowed to do from an access-control perspective), `skills` describe what the tool **can do** from a functional perspective — for example `file_reading`, `code_generation`, `web_search`, `database_query`. Skills enable capability-based filtering, agent composition analysis, and automated discovery of what an agent is functionally capable of.

## prompt_layer

| Field | Type | Required | Description |
|---|---|---|---|
| `system_prompt_hash` | string | no | SHA-256 of system prompt |
| `prompt_version` | string | no | Version identifier for the active prompt set |
| `template_ids` | string[] | no | Referenced prompt template IDs |
| `templates` | object[] | no | Versioned prompt template references |

### prompt_version

The `prompt_version` field tracks the version of the active prompt configuration as a whole. This enables audit trails to answer which prompt version was active at a given point in time.

### templates

Each entry in the `templates` array provides versioned and hashed references to individual prompt templates:

| Field | Type | Required | Description |
|---|---|---|---|
| `template_id` | string | yes | Prompt template identifier |
| `version` | string | no | Template version (semver or snapshot) |
| `hash` | string | no | SHA-256 hash of the template content |

The `templates` array supplements the simpler `template_ids` array; both may be present. Consumers preferring minimal syntax can use `template_ids`, while those needing version or integrity information should use `templates`.

## permission_layer

| Field | Type | Required | Description |
|---|---|---|---|
| `granted_scopes` | string[] | no | All granted permission scopes |
| `data_access` | string[] | no | Data sources the agent can access |
| `credential_references` | string[] | no | Credential type references (no secrets) |

## policy_definitions

Array of policy definition entries:

| Field | Type | Required | Description |
|---|---|---|---|
| `policy_id` | string | yes | Unique policy identifier |
| `policy_name` | string | yes | Human-readable policy name |
| `policy_type` | string | yes | Policy type or category (e.g., `content_filter`, `rate_limit`, `data_handling`, `compliance`) |
| `version` | string | no | Policy version (semver) |
| `description` | string | no | Description of what the policy governs |
| `rules` | string[] | no | Declarative policy rules or constraint expressions |

The `policy_definitions` field captures governance policies and compliance rules that constrain or govern the agent's behavior. This is distinct from `permission_layer` (which declares what the agent *can* access) — `policy_definitions` describe the *rules* that govern *how* the agent uses those permissions, such as content filtering policies, rate limits, data handling constraints, and regulatory compliance requirements. Each policy has a type classification enabling automated policy enforcement and compliance checking.

## evidence_layer

| Field | Type | Required | Description |
|---|---|---|---|
| `aep_references` | string[] | no | AEP event IDs or hashes |
| `evidence_hashes` | object[] | no | `{type, hash, timestamp}` |

## audit_log

Array of audit trail entries:

| Field | Type | Required | Description |
|---|---|---|---|
| `timestamp` | ISO 8601 | yes | Event timestamp |
| `event_type` | string | yes | Type of audit event (e.g., `tool_call`, `permission_check`, `prompt_injection_attempt`) |
| `actor` | string | yes | Entity that performed the action (user ID, system component, or external service) |
| `resource` | string | no | Target resource identifier affected by the event |
| `outcome` | string | no | `success`, `failure`, or `partial` |
| `details` | object | no | Additional event-specific context and metadata |

## risk_layer

Array of risk entries:

| Field | Type | Required | Description |
|---|---|---|---|
| `risk_id` | string | yes | Unique risk identifier |
| `severity` | string | yes | `critical`, `high`, `medium`, `low`, `info` |
| `category` | string | yes | Risk category |
| `description` | string | yes | Risk description |
| `status` | string | yes | `open`, `mitigated`, `accepted` |

## workflow_layer

Array of workflow (action pathway) definitions:

| Field | Type | Required | Description |
|---|---|---|---|
| `workflow_id` | string | yes | Unique workflow identifier |
| `workflow_name` | string | yes | Human-readable workflow name |
| `description` | string | no | Workflow description |
| `version` | string | no | Workflow definition version (semver) |
| `steps` | object[] | yes | Ordered list of steps in this workflow |

The `workflow_layer` captures action pathway definitions — the sequences of tool calls, prompts, and decision points that define how the agent executes its tasks. Each workflow is an ordered or dependency-linked set of steps.

### steps

Each step in a workflow:

| Field | Type | Required | Description |
|---|---|---|---|
| `step_id` | string | yes | Unique step identifier within the workflow |
| `action` | string | yes | Action to perform (e.g., tool_id, prompt name, `sub_workflow`, `decision`) |
| `description` | string | no | Step description |
| `input` | string | no | Input reference or template |
| `output` | string | no | Expected output reference |
| `depends_on` | string[] | no | Step IDs this step depends on; empty means no dependency |
| `allowed_tools` | string[] | no | Tool IDs allowed at this step (subset of the agent's registered tools) |

The `action` field references what to execute: a tool ID from `tool_layer`, a prompt template from `prompt_layer`, a sub-workflow by its `workflow_id`, or a built-in decision primitive. The `depends_on` field expresses ordering constraints: if step B depends on step A, B will not start until A completes. Steps with no dependencies may execute in any order or in parallel, depending on the runtime.

## attestation

| Field | Type | Required | Description |
|---|---|---|---|
| `generator` | string | yes | Tool or process that generated this AgentBOM |
| `generator_version` | string | no | Generator version |
| `agentbom_hash` | string | no | SHA-256 of canonical AgentBOM JSON |
| `signature` | string | no | Cryptographic signature of the AgentBOM for verification |
| `timestamp` | ISO 8601 | no | Attestation timestamp |

## CLI commands

```bash
agent-trust agentbom validate <path>    # Validate against schema
agent-trust agentbom inspect <path>     # Human-readable summary
agent-trust agentbom diff <old> <new>   # Show changes between two AgentBOMs
```

## Extraction criteria

AgentBOM can be extracted into a standalone specification repository when:

1. The schema version (`agentbom_version`) has stabilized and no breaking changes
   are expected in the short term.
2. External consumers (beyond the WasmAgent ecosystem) adopt the package
   independently, or a standardization pathway (e.g., OpenSSF, OWASP CycloneDX
   extension) has been initiated — see [`docs/openssf-proposal.md`](../../docs/openssf-proposal.md).
3. The eleven-layer schema has coverage across at least one major regulatory
   framework mapping (e.g., EU AI Act Annex IV, ISO/IEC 42001, NIST AI RMF).

Until extraction, the canonical source is this repository. The JSON schema in
`specs/agentbom/schema.json` and the reference implementation in
`packages/agentbom-core/` are versioned together.
