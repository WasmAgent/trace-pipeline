"""Tests for LangSmith/LangGraph trace to AEP import."""
from evomerge.benchmarks.langsmith_trace import LSRun, LSTrace, ls_trace_to_aep


def _make_trace() -> LSTrace:
    return LSTrace(
        trace_id="tr-ls-001",
        session_id="sess-ls-001",
        model_id="gpt-4o",
        runs=[
            LSRun(
                id="tr-ls-001",
                name="RootChain",
                run_type="chain",
                start_time="2026-06-25T10:00:00Z",
                end_time="2026-06-25T10:00:15Z",
            ),
            LSRun(
                id="run-llm-001",
                name="ChatOpenAI",
                run_type="llm",
                start_time="2026-06-25T10:00:01Z",
                end_time="2026-06-25T10:00:03Z",
                parent_run_id="tr-ls-001",
                extra={"invocation_params": {"model": "gpt-4o"}},
            ),
            LSRun(
                id="run-tool-001",
                name="web_search",
                run_type="tool",
                start_time="2026-06-25T10:00:04Z",
                end_time="2026-06-25T10:00:07Z",
                parent_run_id="tr-ls-001",
                outputs={"result": "some result"},
            ),
            LSRun(
                id="run-retriever-001",
                name="VectorStoreRetriever",
                run_type="retriever",
                start_time="2026-06-25T10:00:08Z",
                end_time="2026-06-25T10:00:09Z",
                parent_run_id="tr-ls-001",
                outputs={"documents": []},
            ),
        ],
    )


def test_schema_version():
    record = ls_trace_to_aep(_make_trace())
    assert record["schema_version"] == "aep/v0.1"


def test_tool_run_becomes_action():
    """Tool run with no error and non-empty outputs → state_changing=True."""
    record = ls_trace_to_aep(_make_trace())
    actions = record["actions"]
    assert len(actions) == 1
    assert actions[0]["tool_name"] == "web_search"
    assert actions[0]["state_changing"] is True


def test_tool_error_becomes_deny_decision():
    """Tool run with an error → capability_decisions deny entry."""
    trace = _make_trace()
    # Inject an error and clear outputs so state_changing logic is also exercised
    trace.runs[2].error = "connection timeout"
    trace.runs[2].outputs = {}
    record = ls_trace_to_aep(trace)
    denies = [d for d in record["capability_decisions"] if d["decision"] == "deny"]
    assert len(denies) >= 1
    tool_denies = [d for d in denies if d["resource"] == "web_search"]
    assert len(tool_denies) == 1
    assert tool_denies[0]["reason_code"] == "tool_error"


def test_llm_run_sets_model_id():
    """LLM run with extra.invocation_params.model sets model_id on AEP record."""
    trace = LSTrace(
        trace_id="tr-ls-002",
        session_id="",
        model_id="",
        runs=[
            LSRun(
                id="tr-ls-002",
                name="RootChain",
                run_type="chain",
                start_time="2026-06-25T11:00:00Z",
                end_time="2026-06-25T11:00:05Z",
            ),
            LSRun(
                id="run-llm-002",
                name="ChatOpenAI",
                run_type="llm",
                start_time="2026-06-25T11:00:01Z",
                end_time="2026-06-25T11:00:03Z",
                parent_run_id="tr-ls-002",
                extra={"invocation_params": {"model": "gpt-4o"}},
            ),
        ],
    )
    record = ls_trace_to_aep(trace)
    assert record["model_id"] == "gpt-4o"


def test_run_id_format():
    """Root run id tr-ls-001 should yield run_id 'langsmith/tr-ls-001'."""
    record = ls_trace_to_aep(_make_trace())
    assert record["run_id"] == "langsmith/tr-ls-001"


def test_retriever_not_in_actions():
    """Retriever runs must NOT appear in the actions list."""
    record = ls_trace_to_aep(_make_trace())
    action_tools = [a["tool_name"] for a in record["actions"]]
    assert "VectorStoreRetriever" not in action_tools
    # Also verify no retriever run_type leaks in
    assert all(
        name != "retriever"
        for name in action_tools
    )
