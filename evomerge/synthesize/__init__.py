"""evomerge.synthesize — TaskSpec templates and synthetic sample generation.

Provides:
  - Built-in TaskSpec templates for the three MVP task types:
      markdown_report, tool_call, repair
  - SyntheticGenerator: calls a teacher-model API to produce SFT/DPO records
    from TaskSpec templates without requiring real wasmagent-js run data.
"""
from evomerge.synthesize.generator import GenerationConfig, SyntheticGenerator
from evomerge.synthesize.templates import (
    TaskType,
    builtin_templates,
    make_task_spec,
)

__all__ = [
    "GenerationConfig",
    "SyntheticGenerator",
    "TaskType",
    "builtin_templates",
    "make_task_spec",
]
