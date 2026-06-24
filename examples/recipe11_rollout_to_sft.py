"""recipe11_rollout_to_sft.py — convert rollout JSONL to SFT training records.

Demonstrates the simplest evomerge pipeline step:
  rollout-wire/v1 JSONL  →  sft/v1 JSONL  (only passing branches)

Run:
    python examples/recipe11_rollout_to_sft.py
"""
from evomerge.io import load_rollouts, write_jsonl
from evomerge.pipeline.sft import to_sft_records

# Use the shared fixture so the example always works without external data
FIXTURE = "fixtures/data-loop/rollout-branches.v1.jsonl"

branches = load_rollouts(FIXTURE)
print(f"loaded {len(branches)} branches")
for b in branches:
    print(f"  branch {b.branch_index}: score={b.objective_score} '{b.final_answer[:60]}'")

sft = to_sft_records(branches, only_passing=True)
print(f"\nSFT records (passing only): {len(sft)}")
for r in sft:
    print(f"  output_type={r.output_type!r}  provenance={r.provenance.source!r}")
    print(f"  messages: {[m.role for m in r.messages]}")
    print(f"  target: '{r.messages[-1].content[:80]}'")

# Validate before export
from evomerge.validate.schema_check import validate_training_record
for r in sft:
    result = validate_training_record(r)
    assert result.ok, result.errors

print("\nall SFT records valid — ready to export")
