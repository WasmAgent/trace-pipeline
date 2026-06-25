"""Tests for OpenAI Agents SDK trace to AEP import."""
from evomerge.benchmarks.openai_agents_trace import OAISpan, OAITrace, oai_trace_to_aep


def _make_trace():
    return OAITrace(
        trace_id="tr-001",
        model_id="gpt-4o-mini",
        agent_name="test-agent",
        session_id="sess-001",
        spans=[
            OAISpan(trace_id="tr-001", span_id="s1", type="agent.run", name="MyAgent",
                    started_at="2026-06-25T10:00:00Z", ended_at="2026-06-25T10:00:10Z"),
            OAISpan(trace_id="tr-001", span_id="s2", type="llm.generate", name="LLM call",
                    parent_span_id="s1",
                    started_at="2026-06-25T10:00:01Z", ended_at="2026-06-25T10:00:02Z",
                    attributes={"model": "gpt-4o-mini", "input_tokens": 100}),
            OAISpan(trace_id="tr-001", span_id="s3", type="tool.call", name="web_search",
                    parent_span_id="s1",
                    started_at="2026-06-25T10:00:03Z", ended_at="2026-06-25T10:00:05Z",
                    attributes={"tool_name": "web_search", "state_changing": False}),
            OAISpan(trace_id="tr-001", span_id="s4", type="guardrail", name="content_policy",
                    parent_span_id="s1",
                    started_at="2026-06-25T10:00:09Z", ended_at="2026-06-25T10:00:09.1Z",
                    attributes={"triggered": False}),
        ],
    )


def test_aep_schema_version():
    record = oai_trace_to_aep(_make_trace())
    assert record["schema_version"] == "aep/v0.1"


def test_tool_call_becomes_action():
    record = oai_trace_to_aep(_make_trace())
    assert len(record["actions"]) == 1
    assert record["actions"][0]["tool_name"] == "web_search"
    assert record["actions"][0]["state_changing"] is False


def test_guardrail_becomes_verifier_result():
    record = oai_trace_to_aep(_make_trace())
    vr = record["verifier_results"]
    assert len(vr) == 1
    assert vr[0]["verifier_id"] == "guardrail/content_policy"
    assert vr[0]["passed"] is True


def test_model_id_and_provider():
    record = oai_trace_to_aep(_make_trace())
    assert record["model_id"] == "gpt-4o-mini"
    assert record["model_provider"] == "openai"


def test_tool_error_adds_deny_decision():
    trace = _make_trace()
    trace.spans[2].error = "timeout"
    record = oai_trace_to_aep(trace)
    denies = [d for d in record["capability_decisions"] if d["decision"] == "deny"]
    assert len(denies) == 1
    assert denies[0]["resource"] == "web_search"


def test_run_id_format():
    record = oai_trace_to_aep(_make_trace())
    assert record["run_id"] == "oai-agents/tr-001"
