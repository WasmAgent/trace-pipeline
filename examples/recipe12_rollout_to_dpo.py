"""recipe12_rollout_to_dpo.py — convert rollout JSONL to DPO preference pairs.

Within each rollout_id the highest-ranked branch becomes 'chosen' and
the lowest-ranked becomes 'rejected'.  This recipe also shows how to
inspect the pair and run the contamination guard.

Run:
    python examples/recipe12_rollout_to_dpo.py
"""
from evomerge.io import load_rollouts
from evomerge.pipeline.dpo import to_dpo_records
from evomerge.validate.contamination import check_contamination

FIXTURE = "fixtures/data-loop/rollout-branches.v1.jsonl"

branches = load_rollouts(FIXTURE)
dpo = to_dpo_records(branches)

print(f"DPO pairs: {len(dpo)}")
for r in dpo:
    print(f"\n  rollout_id : {r.provenance.rollout_id}")
    print(f"  chosen     : '{r.chosen[:80]}'")
    print(f"  rejected   : '{r.rejected[:80]}'")
    print(f"  loss_weight: {r.loss_weight_tokens!r}")

# Contamination guard: make sure outputs don't appear in an eval set
eval_items = [
    "The nitrogen cycle describes how nitrogen moves through the atmosphere.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
]
report = check_contamination(
    training_texts=[r.chosen for r in dpo],
    eval_texts=eval_items,
    threshold=0.2,
)
print(f"\ncontamination check: {report.n_flagged}/{report.n_training} flagged "
      f"(threshold={report.threshold})")
assert report.n_flagged == 0, "unexpected contamination in fixture data"
print("contamination guard passed")
