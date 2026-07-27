"""Tests for evomerge.synthesize.templates."""
from __future__ import annotations

import pytest

from evomerge.schemas.compliance import ConstraintLevel, TaskSpec
from evomerge.synthesize.templates import (
    TaskType,
    builtin_templates,
    make_task_spec,
)


class TestBuiltinTemplates:
    def test_returns_dict(self):
        t = builtin_templates()
        assert isinstance(t, dict)
        assert len(t) >= 4

    def test_all_values_are_task_specs(self):
        for name, spec in builtin_templates().items():
            assert isinstance(spec, TaskSpec), name

    def test_rfi_report_has_required_sections(self):
        spec = builtin_templates()["rfi_report_zh"]
        section_ids = {c.id for c in spec.constraints}
        assert "c_section_0" in section_ids

    def test_rfi_report_language_zh(self):
        spec = builtin_templates()["rfi_report_zh"]
        assert spec.language == "zh-CN"

    def test_web_search_has_allowed_tools(self):
        spec = builtin_templates()["web_search_task"]
        assert spec.tools is not None
        assert "web_search" in spec.tools.allowed

    def test_repair_spec_has_patch_only_constraint(self):
        spec = builtin_templates()["repair_missing_section"]
        ids = {c.id for c in spec.constraints}
        assert "c_patch_only" in ids


class TestMakeTaskSpec:
    def test_markdown_report_defaults(self):
        spec = make_task_spec(TaskType.markdown_report, intent="Write a report")
        assert spec.intent == "Write a report"
        assert any(c.id.startswith("c_section_") for c in spec.constraints)

    def test_tool_call_defaults(self):
        spec = make_task_spec(TaskType.tool_call, intent="Search the web")
        assert spec.tools is not None
        assert "web_search" in spec.tools.allowed

    def test_repair_defaults(self):
        spec = make_task_spec(TaskType.repair, intent="Fix the draft")
        ids = {c.id for c in spec.constraints}
        assert "c_patch_only" in ids

    def test_custom_language(self):
        spec = make_task_spec(
            TaskType.markdown_report, intent="Bericht", language="de"
        )
        assert spec.language == "de"

    def test_custom_sections(self):
        spec = make_task_spec(
            TaskType.markdown_report,
            intent="Report",
            required_sections=["Alpha", "Beta"],
        )
        section_descs = [c.description for c in spec.constraints if "Alpha" in c.description or "Beta" in c.description]
        assert len(section_descs) == 2

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            make_task_spec("unknown_type")  # type: ignore

    def test_all_constraints_have_repair_policy(self):
        spec = make_task_spec(TaskType.markdown_report, intent="Test")
        for c in spec.constraints:
            assert c.repair is not None, f"No repair policy on {c.id}"

    def test_hard_constraints_only(self):
        spec = make_task_spec(TaskType.tool_call, intent="Test")
        for c in spec.constraints:
            assert c.level == ConstraintLevel.hard
