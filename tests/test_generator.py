"""Tests for evomerge.synthesize.generator (with a mock chat_fn)."""
from __future__ import annotations

from evomerge.schemas.compliance import (
    ComplianceEvalRecord,
    ConstraintCategory,
    ConstraintLevel,
    ConstraintViolation,
    RunMode,
    ViolationStage,
)
from evomerge.schemas.training import DpoTrainingRecord, SftTrainingRecord
from evomerge.synthesize.generator import GenerationConfig, SyntheticGenerator
from evomerge.synthesize.templates import TaskType, builtin_templates, make_task_spec


def _echo_chat(messages: list[dict]) -> str:
    """Mock chat_fn: returns a fixed response based on prompt type."""
    content = messages[-1]["content"] if messages else ""
    if "NON-COMPLIANT" in content:
        return "BAD_OUTPUT: missing sections"
    if "repair patch" in content.lower():
        return "REPAIR: inserted missing section"
    return "GOOD_OUTPUT: compliant response with evidence"


_CFG = GenerationConfig(
    teacher_model="mock",
    n_per_template=2,
    n_bad_per_template=2,
    seed=0,
)


class TestSyntheticGenerator:
    def setup_method(self):
        self.gen = SyntheticGenerator(chat_fn=_echo_chat, config=_CFG)

    def test_generate_returns_sft_and_dpo(self):
        templates = {"rfi": make_task_spec(TaskType.markdown_report, intent="RFI")}
        sft, dpo = self.gen.generate(templates)
        assert len(sft) > 0
        assert len(dpo) > 0

    def test_sft_records_are_correct_type(self):
        templates = {"rfi": make_task_spec(TaskType.markdown_report, intent="RFI")}
        sft, _ = self.gen.generate(templates)
        assert all(isinstance(r, SftTrainingRecord) for r in sft)

    def test_dpo_records_are_correct_type(self):
        templates = {"t": make_task_spec(TaskType.tool_call, intent="Search")}
        _, dpo = self.gen.generate(templates)
        assert all(isinstance(r, DpoTrainingRecord) for r in dpo)

    def test_dpo_chosen_ne_rejected(self):
        templates = {"rfi": make_task_spec(TaskType.markdown_report, intent="RFI")}
        _, dpo = self.gen.generate(templates)
        for r in dpo:
            assert r.chosen != r.rejected

    def test_repair_records_have_correct_output_type(self):
        templates = {"rfi": make_task_spec(TaskType.markdown_report, intent="RFI")}
        sft, _ = self.gen.generate(templates)
        repair = [r for r in sft if r.output_type == "repair_patch"]
        assert len(repair) > 0

    def test_repair_loss_weight(self):
        templates = {"rfi": make_task_spec(TaskType.markdown_report, intent="RFI")}
        sft, _ = self.gen.generate(templates)
        repair = [r for r in sft if r.output_type == "repair_patch"]
        assert all(r.loss_weight_tokens == "recovery" for r in repair)

    def test_provenance_source(self):
        templates = {"rfi": make_task_spec(TaskType.markdown_report, intent="RFI")}
        sft, dpo = self.gen.generate(templates)
        for r in sft:
            assert r.provenance.source in ("synthetic-teacher", "synthetic-repair")

    def test_multiple_templates(self):
        templates = builtin_templates()
        sft, dpo = self.gen.generate(templates)
        n_templates = len(templates)
        assert len(sft) >= n_templates * _CFG.n_per_template

    def test_generate_from_compliance_augments_failures(self):
        violation = ConstraintViolation(
            constraint_id="c1",
            level=ConstraintLevel.hard,
            category=ConstraintCategory.content,
            hint="Missing section",
            detected_at=ViolationStage.post_decode,
        )
        rec = ComplianceEvalRecord(
            task_id="t1",
            task_spec_hash="abc",
            model="qwen-7b",
            mode=RunMode.full_pcl,
            final_pass=False,
            artifact="Incomplete output.",
            violations=[violation],
        )
        result = self.gen.generate_from_compliance([rec])
        repair_records = [r for r in result if r.output_type == "repair_patch"]
        assert len(repair_records) >= 1

    def test_generate_from_compliance_passes_unchanged(self):
        rec = ComplianceEvalRecord(
            task_id="t2",
            task_spec_hash="abc",
            model="qwen-7b",
            mode=RunMode.full_pcl,
            final_pass=True,
            artifact="Good output.",
        )
        result = self.gen.generate_from_compliance([rec])
        assert any(r.output_type == "final_answer" for r in result)
