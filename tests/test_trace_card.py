from evomerge.dataset_card import TraceCard, generate_trace_card


def test_generate_trace_card_basic():
    card = TraceCard(
        dataset_name="wasmagent-v1",
        source_repo="WasmAgent/wasmagent-js",
        source_commit="abc123",
        collection_mode="evidence",
        data_subject_risk="low",
        pii_redaction_profile="bscode/pii-redact/v1",
        license="Apache-2.0",
        consent_basis="operator-consent",
        model_provider="local",
        model_id="qwen2.5-1.5b",
        policy_bundle="default",
        verifier_bundle="ifeval+deterministic",
        known_biases=["small model only", "IFEval distribution"],
        replay_command="python -m evomerge export --input data/rollouts.jsonl",
    )
    md = generate_trace_card(card, date="2026-06-25")
    assert "wasmagent-v1" in md
    assert "aep/v0.1" in md
    assert "small model only" in md
    assert "2026-06-25" in md

def test_trace_card_no_biases():
    card = TraceCard(
        dataset_name="test",
        source_repo="WasmAgent/test",
        source_commit="000",
        collection_mode="demo",
        data_subject_risk="low",
        pii_redaction_profile="none",
        license="MIT",
        consent_basis="none",
        model_provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        policy_bundle="none",
        verifier_bundle="none",
    )
    md = generate_trace_card(card)
    assert "None reported" in md
