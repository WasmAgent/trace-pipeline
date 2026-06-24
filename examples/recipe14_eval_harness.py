"""recipe14_eval_harness.py — A/B/C/D/E experiment harness.

Demonstrates how to set up the five comparison groups, run them over a
shared task list, compute metrics, and print a summary table.

The run_fn stubs here return deterministic EvalRecord objects so no model
is needed — swap in real model calls for production use.

Run:
    python examples/recipe14_eval_harness.py
"""
from evomerge.eval import EvalConfig, EvalGroup, EvalHarness, EvalRecord

# ---- task set ----
TASKS = {f"task-{i:03d}": f"Summarise document #{i} in two sentences." for i in range(30)}

# ---- deterministic stubs (replace with real model calls) ----
def _run_A(task_id: str, task: str) -> EvalRecord:
    """Base small model — fails on 70% of tasks."""
    i = int(task_id.split("-")[1])
    return EvalRecord(task_id=task_id, group="A", final_pass=(i % 10 >= 7),
                      tool_calls_total=0, latency_ms=300.0, prompt_tokens=80, generation_tokens=120)

def _run_B(task_id: str, task: str) -> EvalRecord:
    """Base small model + compliance engine — fails on 50%."""
    i = int(task_id.split("-")[1])
    rr = 1 if i % 5 == 0 else 0
    return EvalRecord(task_id=task_id, group="B", final_pass=(i % 10 >= 5),
                      repair_rounds=rr, repair_rounds_ok=rr,
                      latency_ms=450.0, prompt_tokens=100, generation_tokens=130)

def _run_C(task_id: str, task: str) -> EvalRecord:
    """Fine-tuned small model + compliance engine — fails on 15%."""
    i = int(task_id.split("-")[1])
    rr = 1 if i % 8 == 0 else 0
    return EvalRecord(task_id=task_id, group="C", final_pass=(i % 10 >= 2),
                      repair_rounds=rr, repair_rounds_ok=rr,
                      latency_ms=420.0, prompt_tokens=100, generation_tokens=125)

def _run_D(task_id: str, task: str) -> EvalRecord:
    """Large model direct prompt — fails on 5%."""
    i = int(task_id.split("-")[1])
    return EvalRecord(task_id=task_id, group="D", final_pass=(i % 20 != 0),
                      latency_ms=1200.0, prompt_tokens=200, generation_tokens=300)

# ---- harness ----
config = EvalConfig(
    task_ids=list(TASKS.keys()),
    tasks=list(TASKS.values()),
)
groups = {
    "A": EvalGroup(label="A", run_fn=_run_A, description="base small, direct"),
    "B": EvalGroup(label="B", run_fn=_run_B, description="base small + compliance"),
    "C": EvalGroup(label="C", run_fn=_run_C, description="fine-tuned + compliance"),
    "D": EvalGroup(label="D", run_fn=_run_D, description="large model, direct"),
}
harness = EvalHarness(config=config, groups=groups)
report = harness.run()

# ---- print table ----
header = f"{'Group':<8} {'n':>4} {'pass%':>7} {'repair%':>8} {'cost/task':>10} {'latency':>10}"
print(header)
print("-" * len(header))
for label in "ABCD":
    m = report.metrics[label]
    print(
        f"{label:<8} {m.n:>4} "
        f"{m.taskspec_pass_rate*100:>6.1f}% "
        f"{m.repair_success_rate*100:>7.1f}% "
        f"{m.cost_per_accepted_task:>10.0f} "
        f"{m.latency_per_accepted_ms:>9.0f}ms"
    )

print()
assert report.metrics["C"].taskspec_pass_rate > report.metrics["A"].taskspec_pass_rate
print("C outperforms A — as expected")
