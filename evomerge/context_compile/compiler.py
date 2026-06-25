"""Trace-to-Context Compiler — compile rollout traces to long-context training data.

Converts multi-step agent rollout traces into two types of training records:

  1. long-context QA (mode="long_context_qa"):
     Context = full tool call sequence + observations (the "long context")
     Question = derived from the task
     Answer = the correct final answer or decision
     Target use: small models learning to extract key signals from long contexts

  2. router / critic decisions (mode="router_critic"):
     For each step: given context so far, should the agent call another tool,
     stop, or ask for more information?
     Target use: runtime router deciding when to stop, critic judging sufficiency

Usage:
    from evomerge.context_compile.compiler import (
        compile_trace,
        ContextQaRecord,
        RouterCriticRecord,
        compile_file,
    )

    records = compile_trace(rollout_record, mode="long_context_qa")
    critic_records = compile_trace(rollout_record, mode="router_critic")
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

from evomerge.schemas.rollout import RolloutBranchRecord, ToolCallEntry

# ── Output record types ───────────────────────────────────────────────────────

CompileMode = Literal["long_context_qa", "router_critic"]


@dataclass
class ContextQaRecord:
    """Long-context QA record for small-model reading comprehension training.

    The model sees the full context (task + tool sequence + observations) and
    must produce the correct answer — without being allowed to call tools itself.
    """
    schema_version: str = "context-qa/v1"
    record_id: str = ""
    context: str = ""
    """Full serialised trace up to the final answer."""
    question: str = ""
    """Derived from the task description."""
    answer: str = ""
    """The agent's verified final answer (only from objective_status=pass branches)."""
    n_tool_calls: int = 0
    """How many tool calls were in the trace — signals long-context length."""
    objective_status: str = "unknown"
    provenance: dict = field(default_factory=dict)


@dataclass
class RouterCriticRecord:
    """Router / critic decision record.

    For each intermediate step, the model must decide:
      'continue' — more tool calls needed
      'stop'     — sufficient context, ready to answer
      'ask'      — insufficient context, need clarification

    Target: router trained to predict stop/continue/ask from partial context.
    """
    schema_version: str = "router-critic/v1"
    record_id: str = ""
    partial_context: str = ""
    """Context up to (but not including) the current step."""
    step_index: int = 0
    decision: Literal["continue", "stop", "ask"] = "continue"
    """Ground truth: 'stop' on last step of passing branch, 'continue' otherwise."""
    evidence: str = ""
    """The tool call / observation at this step."""
    n_remaining_steps: int = 0
    provenance: dict = field(default_factory=dict)


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _serialise_tool_call(entry: ToolCallEntry, index: int) -> str:
    args_str = json.dumps(entry.arguments, ensure_ascii=False)
    result_str = (
        str(entry.result)[:512] if entry.result is not None
        else f"[ERROR] {entry.error}"
    )
    return (
        f"[Step {index}] Tool: {entry.tool_name}\n"
        f"  Arguments: {args_str}\n"
        f"  Result: {result_str}"
    )


def _build_full_context(record: RolloutBranchRecord) -> str:
    parts = [f"Task: {record.task}"]
    for i, entry in enumerate(record.tool_call_sequence):
        parts.append(_serialise_tool_call(entry, i))
    parts.append(f"Final answer: {record.final_answer}")
    return "\n\n".join(parts)


def _record_hash(record_id: str, content: str) -> str:
    return hashlib.sha256(f"{record_id}:{content}".encode()).hexdigest()[:16]


# ── Core compilation ──────────────────────────────────────────────────────────

def compile_trace(
    record: RolloutBranchRecord,
    *,
    mode: CompileMode = "long_context_qa",
    min_tool_calls: int = 1,
) -> list[ContextQaRecord] | list[RouterCriticRecord]:
    """Compile one RolloutBranchRecord to context training records.

    Args:
        record: A single rollout branch.
        mode: Compilation target ('long_context_qa' or 'router_critic').
        min_tool_calls: Skip records with fewer tool calls (too short for
                        long-context training).

    Returns:
        For 'long_context_qa': one record if the branch passes, else empty.
        For 'router_critic': one record per step (all branches).
    """
    if len(record.tool_call_sequence) < min_tool_calls:
        return []

    prov = {
        "rollout_id": record.rollout_id,
        "branch_index": record.branch_index,
        "objective_status": record.objective_status,
    }

    if mode == "long_context_qa":
        # Only admitted (passing) branches produce QA records
        if record.objective_status != "pass":
            return []
        context = _build_full_context(record)
        rec = ContextQaRecord(
            record_id=_record_hash(record.rollout_id, context),
            context=context,
            question=record.task,
            answer=record.final_answer,
            n_tool_calls=len(record.tool_call_sequence),
            objective_status=record.objective_status,
            provenance=prov,
        )
        return [rec]

    else:  # router_critic
        results: list[RouterCriticRecord] = []
        steps = record.tool_call_sequence
        total = len(steps)

        context_parts = [f"Task: {record.task}"]
        for i, entry in enumerate(steps):
            partial = "\n\n".join(context_parts)
            evidence = _serialise_tool_call(entry, i)
            is_last = i == total - 1

            # Ground truth decision:
            # - stop: last step of a passing branch (agent correctly concluded)
            # - continue: any non-final step
            # - ask: last step of a failing branch with no build result
            if is_last and record.objective_status == "pass":
                decision: Literal["continue", "stop", "ask"] = "stop"
            elif is_last and record.objective_status == "fail":
                decision = "ask"
            else:
                decision = "continue"

            rec = RouterCriticRecord(
                record_id=_record_hash(f"{record.rollout_id}/{i}", partial),
                partial_context=partial,
                step_index=i,
                decision=decision,
                evidence=evidence,
                n_remaining_steps=total - i - 1,
                provenance=prov,
            )
            results.append(rec)
            context_parts.append(evidence)

        return results


def compile_file(
    rollout_jsonl: str,
    *,
    mode: CompileMode = "long_context_qa",
    min_tool_calls: int = 1,
    out: str | None = None,
) -> list[ContextQaRecord] | list[RouterCriticRecord]:
    """Load a rollout JSONL file and compile all records.

    When `out` is set, writes one record per line as JSONL.
    """
    import dataclasses
    from pathlib import Path

    from evomerge.io import load_rollouts

    records = load_rollouts(rollout_jsonl)
    all_compiled: list = []
    for r in records:
        all_compiled.extend(compile_trace(r, mode=mode, min_tool_calls=min_tool_calls))

    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for item in all_compiled:
                fh.write(json.dumps(dataclasses.asdict(item), ensure_ascii=False) + "\n")

    return all_compiled


__all__ = [
    "CompileMode",
    "ContextQaRecord",
    "RouterCriticRecord",
    "compile_trace",
    "compile_file",
]
