"""Tests for evomerge.validate (contamination + schema_check)."""
from __future__ import annotations


from evomerge.validate.contamination import check_contamination
from evomerge.validate.schema_check import validate_training_record
from evomerge.schemas.training import (
    DpoTrainingRecord,
    Message,
    PpoTrainingRecord,
    Provenance,
    SftTrainingRecord,
)


def _prov():
    return Provenance(source="test")


def _sft(user="Task.", assistant="Answer."):
    return SftTrainingRecord(
        messages=[
            Message(role="user", content=user),
            Message(role="assistant", content=assistant),
        ],
        output_type="final_answer",
        provenance=_prov(),
    )


def _dpo(chosen="Good.", rejected="Bad."):
    return DpoTrainingRecord(
        messages=[
            Message(role="user", content="Task."),
            Message(role="assistant", content=chosen),
        ],
        chosen=chosen,
        rejected=rejected,
        provenance=_prov(),
    )


def _ppo(reward=0.8):
    return PpoTrainingRecord(
        messages=[
            Message(role="user", content="Task."),
            Message(role="assistant", content="Answer."),
        ],
        reward=reward,
        provenance=_prov(),
    )


class TestContamination:
    def test_no_overlap(self):
        report = check_contamination(
            ["The cat sat on the mat."],
            ["An entirely different sentence about dogs."],
        )
        assert report.n_flagged == 0

    def test_high_overlap_flagged(self):
        text = "The quick brown fox jumps over the lazy dog in the yard."
        report = check_contamination([text], [text], threshold=0.2)
        assert report.n_flagged == 1

    def test_threshold_respected(self):
        # Jaccard=1.0 exceeds even a very high threshold — the flag is correct.
        # Use a threshold above 1.0 for both token-level and char-level to
        # confirm nothing is flagged when both gates are disabled.
        text = "one two three four five six seven eight nine ten"
        report = check_contamination([text], [text], threshold=1.01, char_threshold=1.01)
        assert report.n_flagged == 0

    def test_empty_training(self):
        report = check_contamination([], ["some eval text here and more words"])
        assert report.n_flagged == 0
        assert report.flag_rate == 0.0

    def test_report_fields(self):
        report = check_contamination(["foo bar"], ["baz qux"], threshold=0.2)
        assert report.n_training == 1
        assert report.n_eval == 1


class TestSchemaCheck:
    def test_valid_sft(self):
        result = validate_training_record(_sft())
        assert result.ok

    def test_valid_dpo(self):
        result = validate_training_record(_dpo())
        assert result.ok

    def test_valid_ppo(self):
        result = validate_training_record(_ppo())
        assert result.ok

    def test_empty_messages(self):
        rec = _sft()
        rec.messages = []
        result = validate_training_record(rec)
        assert not result.ok
        assert any("empty" in e for e in result.errors)

    def test_last_not_assistant(self):
        rec = SftTrainingRecord(
            messages=[Message(role="user", content="Task.")],
            output_type="final_answer",
            provenance=_prov(),
        )
        result = validate_training_record(rec)
        assert not result.ok

    def test_dpo_identical_chosen_rejected(self):
        result = validate_training_record(_dpo(chosen="Same.", rejected="Same."))
        assert not result.ok
        assert any("identical" in e for e in result.errors)

    def test_ppo_reward_out_of_range(self):
        result = validate_training_record(_ppo(reward=1.5))
        assert not result.ok
        assert any("outside" in e for e in result.errors)

    def test_ppo_zero_reward_valid(self):
        result = validate_training_record(_ppo(reward=0.0))
        assert result.ok
