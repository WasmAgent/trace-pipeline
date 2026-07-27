"""Synthetic SFT/DPO sample generation driven by a teacher model.

The generator takes TaskSpec templates and asks a teacher model to produce:
  - good outputs (chosen / SFT target)
  - bad outputs with specific violation types (rejected / DPO negative)

No direct dependency on any LLM SDK: callers supply a `chat_fn` callable
so this works with Anthropic, OpenAI, local APIs, or any compatible interface.

Typical usage:

    import anthropic

    client = anthropic.Anthropic()

    def chat_fn(messages, *, model, max_tokens):
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, messages=messages
        )
        return resp.content[0].text

    cfg = GenerationConfig(teacher_model="claude-opus-4-8", n_per_template=10)
    gen = SyntheticGenerator(chat_fn=chat_fn, config=cfg)
    sft, dpo = gen.generate(templates=builtin_templates())
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from evomerge.pipeline.compliance_sft import compliance_to_sft_records
from evomerge.schemas.compliance import (
    ComplianceEvalRecord,
    TaskSpec,
)
from evomerge.schemas.training import (
    DpoTrainingRecord,
    Message,
    Provenance,
    SftTrainingRecord,
)

ChatFn = Callable[[list[dict[str, str]]], str]


@dataclass
class GenerationConfig:
    """Configuration for synthetic sample generation.

    Attributes:
        teacher_model: model identifier passed back to chat_fn.
        n_per_template: number of good outputs to generate per template.
        n_bad_per_template: number of bad outputs (for DPO negatives) per template.
        max_tokens: max output tokens per call.
        seed: random seed for reproducible shuffling.
        violation_types: list of violation descriptions to inject into bad prompts.
    """
    teacher_model: str = "claude-opus-4-8"
    n_per_template: int = 5
    n_bad_per_template: int = 5
    max_tokens: int = 2048
    seed: int = 42
    violation_types: list[str] = field(default_factory=lambda: [
        "missing required sections",
        "wrong output language",
        "no action list",
        "tool results not cited in answer",
        "invalid tool arguments",
        "evidence insufficient or fabricated",
    ])


def _task_spec_summary(spec: TaskSpec) -> str:
    lines = [f"Task ID: {spec.id}", f"Intent: {spec.intent}", f"Language: {spec.language}"]
    if spec.constraints:
        lines.append("Constraints:")
        for c in spec.constraints:
            lines.append(f"  [{c.level.value}] {c.description}")
    if spec.tools:
        lines.append(f"Allowed tools: {spec.tools.allowed}")
    return "\n".join(lines)


def _good_prompt(spec: TaskSpec) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "You are a teacher model generating high-quality training data. "
                "Produce a COMPLIANT output for the following task. "
                "The output must satisfy ALL constraints listed.\n\n"
                f"{_task_spec_summary(spec)}\n\n"
                "Respond with ONLY the task output, no preamble."
            ),
        }
    ]


def _bad_prompt(spec: TaskSpec, violation: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "You are a teacher model generating training data for preference learning. "
                "Produce a NON-COMPLIANT output for the following task. "
                f"The output must exhibit this specific violation: **{violation}**.\n\n"
                f"{_task_spec_summary(spec)}\n\n"
                "Respond with ONLY the task output, no preamble."
            ),
        }
    ]


def _repair_prompt(spec: TaskSpec, bad_output: str, violation: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "You are a teacher model generating repair training data. "
                "The following output has a constraint violation. "
                "Produce ONLY the minimal repair patch to fix it — do not rewrite unaffected sections.\n\n"
                f"Task:\n{_task_spec_summary(spec)}\n\n"
                f"Violation: {violation}\n\n"
                f"Bad output:\n{bad_output}\n\n"
                "Respond with ONLY the repair patch."
            ),
        }
    ]


class SyntheticGenerator:
    """Generate SFT and DPO records using a teacher model.

    Args:
        chat_fn: callable(messages: list[dict]) -> str.
            The caller is responsible for passing model/max_tokens to their
            SDK; use a closure or partial to bind those parameters.
        config: GenerationConfig instance.
    """

    def __init__(self, chat_fn: ChatFn, config: GenerationConfig | None = None):
        self._chat = chat_fn
        self.config = config or GenerationConfig()
        self._rng = random.Random(self.config.seed)

    def _call(self, messages: list[dict[str, str]]) -> str:
        return self._chat(messages)

    def generate(
        self,
        templates: dict[str, TaskSpec] | Sequence[TaskSpec],
    ) -> tuple[list[SftTrainingRecord], list[DpoTrainingRecord]]:
        """Generate SFT and DPO records for all templates.

        Args:
            templates: dict of {name: TaskSpec} or list of TaskSpec.

        Returns:
            (sft_records, dpo_records)
        """
        if isinstance(templates, dict):
            spec_list = list(templates.values())
        else:
            spec_list = list(templates)

        sft_records: list[SftTrainingRecord] = []
        dpo_records: list[DpoTrainingRecord] = []

        for spec in spec_list:
            s, d = self._generate_for_spec(spec)
            sft_records.extend(s)
            dpo_records.extend(d)

        return sft_records, dpo_records

    def _generate_for_spec(
        self, spec: TaskSpec
    ) -> tuple[list[SftTrainingRecord], list[DpoTrainingRecord]]:
        sft: list[SftTrainingRecord] = []
        dpo: list[DpoTrainingRecord] = []
        prov = Provenance(source="synthetic-teacher", task_id=spec.id)

        # --- good outputs → SFT records ---
        good_outputs: list[str] = []
        for _ in range(self.config.n_per_template):
            text = self._call(_good_prompt(spec))
            good_outputs.append(text)
            sft.append(
                SftTrainingRecord(
                    messages=[
                        Message(role="user", content=spec.intent),
                        Message(role="assistant", content=text),
                    ],
                    output_type="final_answer",
                    provenance=prov,
                )
            )

        # --- bad outputs → DPO pairs + repair SFT ---
        violations = self._rng.sample(
            self.config.violation_types,
            min(self.config.n_bad_per_template, len(self.config.violation_types)),
        )
        for violation in violations:
            bad_text = self._call(_bad_prompt(spec, violation))
            chosen = self._rng.choice(good_outputs) if good_outputs else ""
            if chosen:
                dpo.append(
                    DpoTrainingRecord(
                        messages=[
                            Message(role="user", content=spec.intent),
                            Message(role="assistant", content=chosen),
                        ],
                        prompt_messages=[Message(role="user", content=spec.intent)],
                        chosen=chosen,
                        rejected=bad_text,
                        provenance=prov,
                    )
                )
            # repair SFT
            repair_text = self._call(_repair_prompt(spec, bad_text, violation))
            sft.append(
                SftTrainingRecord(
                    messages=[
                        Message(
                            role="user",
                            content=(
                                f"Task: {spec.intent}\n\n"
                                f"Violation: {violation}\n\n"
                                f"Bad output:\n{bad_text}"
                            ),
                        ),
                        Message(role="assistant", content=repair_text),
                    ],
                    output_type="repair_patch",
                    loss_weight_tokens="recovery",
                    provenance=prov,
                )
            )

        return sft, dpo

    def generate_from_compliance(
        self, records: Sequence[ComplianceEvalRecord]
    ) -> list[SftTrainingRecord]:
        """Convert real compliance records to SFT, then augment failing ones
        with teacher-repaired versions.

        This is useful when you have real run data but want to fill gaps
        with synthetic repairs for failed records.
        """

        base = compliance_to_sft_records(records)
        augmented: list[SftTrainingRecord] = list(base)

        for rec in records:
            if rec.final_pass or not rec.violations:
                continue
            prov = Provenance(source="synthetic-repair", task_id=rec.task_id)
            for v in rec.violations:
                spec_stub = TaskSpec(id=rec.task_id, intent=rec.artifact[:200])
                repair_text = self._call(
                    _repair_prompt(spec_stub, rec.artifact, v.hint)
                )
                augmented.append(
                    SftTrainingRecord(
                        messages=[
                            Message(
                                role="user",
                                content=f"Task: {rec.task_id}\nViolation: {v.hint}\nBad output:\n{rec.artifact}",
                            ),
                            Message(role="assistant", content=repair_text),
                        ],
                        output_type="repair_patch",
                        loss_weight_tokens="recovery",
                        provenance=prov,
                    )
                )
        return augmented


__all__ = ["ChatFn", "GenerationConfig", "SyntheticGenerator"]
