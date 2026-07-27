"""Built-in TaskSpec templates for the three MVP task types.

Task types (plan Section 7.3):
  markdown_report  — must contain required sections, be in target language,
                     include an action list, cite evidence
  tool_call        — must call allowed tools with schema-valid args, use results
  repair           — given bad_output + VerifierFeedback, produce repair_patch only
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from evomerge.schemas.compliance import (
    ConstraintCategory,
    ConstraintIR,
    ConstraintLevel,
    RepairPolicy,
    RepairStrategy,
    TaskSpec,
    TaskSpecRepairConfig,
    TaskSpecTraceConfig,
    ToolPolicy,
)


class TaskType(str, Enum):
    markdown_report = "markdown_report"
    tool_call = "tool_call"
    repair = "repair"


def _c(
    cid: str,
    desc: str,
    level: ConstraintLevel,
    category: ConstraintCategory,
    strategy: RepairStrategy = RepairStrategy.insert_section,
    priority: int = 70,
) -> ConstraintIR:
    return ConstraintIR(
        id=cid,
        description=desc,
        verify_method="section_presence" if category == ConstraintCategory.content else "schema_validate",
        level=level,
        category=category,
        priority=priority,
        repair=RepairPolicy(strategy=strategy),
    )


def _markdown_report_spec(
    task_id: str,
    intent: str,
    required_sections: list[str],
    language: str = "zh-CN",
    audience: str = "enterprise",
) -> TaskSpec:
    constraints = [
        ConstraintIR(
            id="c_language",
            description=f"Output must be written in language '{language}'",
            verify_method="language_detect",
            level=ConstraintLevel.hard,
            category=ConstraintCategory.style,
            priority=90,
            repair=RepairPolicy(strategy=RepairStrategy.full),
        ),
        ConstraintIR(
            id="c_action_list",
            description="Must include an action list or next-steps section",
            verify_method="section_presence",
            arg={"keywords": ["action", "next steps", "行动", "下一步", "建议"]},
            level=ConstraintLevel.hard,
            category=ConstraintCategory.content,
            priority=80,
            repair=RepairPolicy(strategy=RepairStrategy.insert_section, target_region="action_list"),
        ),
        ConstraintIR(
            id="c_evidence_citation",
            description="Every claim must be backed by at least one cited evidence item",
            verify_method="evidence_citation_check",
            level=ConstraintLevel.hard,
            category=ConstraintCategory.content,
            priority=80,
            repair=RepairPolicy(strategy=RepairStrategy.regenerate_region),
        ),
    ]
    for i, section in enumerate(required_sections):
        constraints.append(
            ConstraintIR(
                id=f"c_section_{i}",
                description=f"Must contain section: '{section}'",
                verify_method="section_presence",
                arg={"section": section},
                level=ConstraintLevel.hard,
                category=ConstraintCategory.content,
                priority=75,
                repair=RepairPolicy(
                    strategy=RepairStrategy.insert_section,
                    target_region=section,
                ),
            )
        )
    return TaskSpec(
        id=task_id,
        intent=intent,
        language=language,
        audience=audience,
        constraints=constraints,
        priority_hierarchy=[
            "system_policy",
            "user_explicit_constraints",
            "task_package_constraints",
        ],
        repair=TaskSpecRepairConfig(max_rounds=3, default_strategy=RepairStrategy.insert_section),
        trace=TaskSpecTraceConfig(
            record_constraint_eval=True,
            record_tool_calls=True,
            record_repairs=True,
        ),
    )


def _tool_call_spec(
    task_id: str,
    intent: str,
    allowed_tools: list[str],
    language: str = "en",
) -> TaskSpec:
    constraints = [
        ConstraintIR(
            id="c_tool_schema",
            description="All tool calls must match the declared tool schema",
            verify_method="tool_schema_validate",
            level=ConstraintLevel.hard,
            category=ConstraintCategory.tool,
            priority=90,
            repair=RepairPolicy(strategy=RepairStrategy.patch),
        ),
        ConstraintIR(
            id="c_tool_result_used",
            description="Tool results must be referenced in the final answer",
            verify_method="tool_result_in_answer",
            level=ConstraintLevel.hard,
            category=ConstraintCategory.content,
            priority=80,
            repair=RepairPolicy(strategy=RepairStrategy.regenerate_region),
        ),
        ConstraintIR(
            id="c_allowed_tools_only",
            description=f"Only these tools may be called: {allowed_tools}",
            verify_method="allowed_tools_check",
            arg={"allowed": allowed_tools},
            level=ConstraintLevel.hard,
            category=ConstraintCategory.tool,
            priority=95,
            repair=RepairPolicy(strategy=RepairStrategy.patch),
        ),
    ]
    return TaskSpec(
        id=task_id,
        intent=intent,
        language=language,
        constraints=constraints,
        priority_hierarchy=["system_policy", "user_explicit_constraints"],
        tools=ToolPolicy(allowed=allowed_tools),
        repair=TaskSpecRepairConfig(max_rounds=2, default_strategy=RepairStrategy.patch),
        trace=TaskSpecTraceConfig(
            record_constraint_eval=True,
            record_tool_calls=True,
            record_repairs=True,
        ),
    )


def _repair_spec(
    task_id: str,
    intent: str,
    language: str = "en",
) -> TaskSpec:
    constraints = [
        ConstraintIR(
            id="c_patch_only",
            description="Output must be a targeted repair patch, not a full rewrite",
            verify_method="patch_scope_check",
            level=ConstraintLevel.hard,
            category=ConstraintCategory.format,
            priority=90,
            repair=RepairPolicy(strategy=RepairStrategy.patch),
        ),
        ConstraintIR(
            id="c_violations_resolved",
            description="All flagged constraint violations must be resolved",
            verify_method="violation_resolution_check",
            level=ConstraintLevel.hard,
            category=ConstraintCategory.content,
            priority=85,
            repair=RepairPolicy(strategy=RepairStrategy.patch),
        ),
    ]
    return TaskSpec(
        id=task_id,
        intent=intent,
        language=language,
        constraints=constraints,
        priority_hierarchy=["system_policy", "user_explicit_constraints"],
        repair=TaskSpecRepairConfig(max_rounds=1, default_strategy=RepairStrategy.patch),
        trace=TaskSpecTraceConfig(
            record_constraint_eval=True,
            record_tool_calls=False,
            record_repairs=True,
        ),
    )


# ------------------------------------------------------------------
# Built-in template catalogue
# ------------------------------------------------------------------

_BUILTIN: dict[str, TaskSpec] = {
    "rfi_report_zh": _markdown_report_spec(
        task_id="rfi_report_zh",
        intent=(
            "Generate a Request for Information (RFI) response report in Chinese. "
            "The report must cover: executive summary, technical approach, "
            "compliance status, pricing overview, and next steps."
        ),
        required_sections=["执行摘要", "技术方案", "合规状态", "价格概述", "下一步"],
        language="zh-CN",
    ),
    "rfp_analysis_en": _markdown_report_spec(
        task_id="rfp_analysis_en",
        intent=(
            "Analyse an RFP document and produce a structured response covering: "
            "scope understanding, proposed solution, key risks, and action plan."
        ),
        required_sections=["Scope", "Proposed Solution", "Key Risks", "Action Plan"],
        language="en",
    ),
    "web_search_task": _tool_call_spec(
        task_id="web_search_task",
        intent=(
            "Search the web for recent information on the given topic and "
            "summarise the top three findings."
        ),
        allowed_tools=["web_search", "web_fetch"],
    ),
    "code_review_task": _tool_call_spec(
        task_id="code_review_task",
        intent=(
            "Read the specified source file, identify potential bugs or "
            "style issues, and return a structured review."
        ),
        allowed_tools=["file_read", "code_analysis"],
    ),
    "repair_missing_section": _repair_spec(
        task_id="repair_missing_section",
        intent=(
            "Given a draft report and a list of constraint violations, "
            "produce only the minimal patch needed to satisfy the violations. "
            "Do not rewrite unaffected sections."
        ),
    ),
}


def builtin_templates() -> dict[str, TaskSpec]:
    """Return a copy of the built-in TaskSpec template catalogue."""
    return dict(_BUILTIN)


def make_task_spec(task_type: TaskType, **kwargs: Any) -> TaskSpec:
    """Construct a TaskSpec from a task type with optional overrides.

    Args:
        task_type: one of TaskType.markdown_report, .tool_call, .repair
        **kwargs: passed to the corresponding factory function.
                  Required for markdown_report: intent, required_sections.
                  Required for tool_call: intent, allowed_tools.
                  Required for repair: intent.
                  Optional for all: task_id, language.

    Returns:
        A new TaskSpec instance.
    """
    try:
        task_type = TaskType(task_type)
    except ValueError as exc:
        raise ValueError(
            f"Unknown task type: {task_type!r}. Choose from {[t.value for t in TaskType]}"
        ) from exc
    task_id = kwargs.pop("task_id", f"{task_type.value}_custom")
    intent = kwargs.pop("intent", f"Custom {task_type.value} task")
    language = kwargs.pop("language", "en")

    if task_type == TaskType.markdown_report:
        required_sections = kwargs.pop("required_sections", ["Summary", "Details", "Action Plan"])
        return _markdown_report_spec(task_id, intent, required_sections, language=language, **kwargs)
    if task_type == TaskType.tool_call:
        allowed_tools = kwargs.pop("allowed_tools", ["web_search"])
        return _tool_call_spec(task_id, intent, allowed_tools, language=language)
    if task_type == TaskType.repair:
        return _repair_spec(task_id, intent, language=language)
    raise ValueError(f"Unknown task type: {task_type}")


__all__ = ["TaskType", "builtin_templates", "make_task_spec"]
