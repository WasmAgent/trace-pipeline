"""Schema-level validation for training records.

Validates that a training record conforms to structural expectations beyond
Pydantic: non-empty messages, valid role sequences, non-empty chosen/rejected,
reward in [0, 1], etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from evomerge.schemas.training import DpoTrainingRecord, PpoTrainingRecord, SftTrainingRecord

TrainingRecord = SftTrainingRecord | DpoTrainingRecord | PpoTrainingRecord


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate_training_record(record: TrainingRecord) -> ValidationResult:
    """Run structural checks on a training record beyond Pydantic parsing.

    Checks:
      - messages list is non-empty
      - first message is role=user
      - last message is role=assistant
      - assistant content is non-empty
      - DPO: chosen != rejected and both non-empty
      - PPO: reward in [0, 1]

    Returns:
        ValidationResult with ok=True if all checks pass.
    """
    errors: list[str] = []

    if not record.messages:
        errors.append("messages is empty")
        return ValidationResult(ok=False, errors=errors)

    if record.messages[0].role != "user":
        errors.append(f"first message role is {record.messages[0].role!r}, expected 'user'")

    last = record.messages[-1]
    if last.role != "assistant":
        errors.append(f"last message role is {last.role!r}, expected 'assistant'")
    elif not last.content.strip():
        errors.append("last assistant message has empty content")

    if isinstance(record, DpoTrainingRecord):
        if not record.chosen.strip():
            errors.append("chosen is empty")
        if not record.rejected.strip():
            errors.append("rejected is empty")
        if record.chosen == record.rejected:
            errors.append("chosen and rejected are identical")

    if isinstance(record, PpoTrainingRecord):
        if not (0.0 <= record.reward <= 1.0):
            errors.append(f"reward {record.reward} is outside [0, 1]")

    return ValidationResult(ok=len(errors) == 0, errors=errors)
