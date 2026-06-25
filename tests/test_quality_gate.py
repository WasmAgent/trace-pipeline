"""Tests for evomerge.validate.quality_gate."""
from __future__ import annotations

import pytest

from evomerge.schemas.training import DpoTrainingRecord, Message, Provenance, SftTrainingRecord
from evomerge.validate.quality_gate import (
    QualityReport,
    check_dpo_quality,
    check_sft_quality,
    run_quality_gate,
)


def _prov():
    return Provenance(source="test")


def _sft(output_type="final_answer", assistant_content="A good answer with enough content here."):
    return SftTrainingRecord(
        messages=[Message(role="user", content="Task"), Message(role="assistant", content=assistant_content)],
        output_type=output_type,
        provenance=_prov(),
    )


def _dpo(chosen="A good chosen answer here, long enough.", rejected="A somewhat shorter but different rejected answer."):
    return DpoTrainingRecord(
        messages=[Message(role="user", content="Task"), Message(role="assistant", content=chosen)],
        chosen=chosen, rejected=rejected,
        provenance=_prov(),
    )


class TestSftQuality:
    def test_healthy_data_no_issues(self):
        records = [_sft() for _ in range(120)] + [_sft("repair_patch") for _ in range(20)]
        issues = check_sft_quality(records, min_records=100)
        assert len(issues) == 0

    def test_too_few_records(self):
        issues = check_sft_quality([_sft() for _ in range(5)], min_records=100)
        assert any(i.check == "sft_min_records" and i.level == "error" for i in issues)

    def test_low_repair_patch_fraction(self):
        # 0 repair_patch → below 5% threshold
        issues = check_sft_quality([_sft() for _ in range(200)], min_records=100)
        assert any(i.check == "sft_repair_patch_fraction" for i in issues)

    def test_high_repair_patch_fraction(self):
        records = [_sft("repair_patch") for _ in range(200)]
        issues = check_sft_quality(records, min_records=100)
        assert any(i.check == "sft_repair_patch_fraction" for i in issues)

    def test_short_assistant_flagged(self):
        records = [_sft(assistant_content="ok")] * 5 + [_sft() for _ in range(115)]
        issues = check_sft_quality(records, min_records=100, min_assistant_chars=20)
        assert any(i.check == "sft_short_assistant" for i in issues)

    def test_empty_records_returns_error(self):
        issues = check_sft_quality([], min_records=100)
        assert any(i.level == "error" for i in issues)


class TestDpoQuality:
    def test_healthy_data_no_issues(self):
        records = [_dpo() for _ in range(50)]
        issues = check_dpo_quality(records, min_records=20)
        assert len(issues) == 0

    def test_identical_pairs_flagged(self):
        records = [_dpo(chosen="same text here.", rejected="same text here.")] * 3
        issues = check_dpo_quality(records, min_records=1)
        assert any(i.check == "dpo_identical_pairs" and i.level == "error" for i in issues)

    def test_short_chosen_flagged(self):
        records = [_dpo(chosen="ok", rejected="A much longer rejected text for comparison.")] * 5
        issues = check_dpo_quality(records, min_records=1, min_chosen_chars=30)
        assert any(i.check == "dpo_short_chosen" for i in issues)

    def test_extreme_length_ratio_flagged(self):
        records = [_dpo(chosen="A" * 500, rejected="B")] * 5
        issues = check_dpo_quality(records, min_records=1, max_length_ratio=10.0)
        assert any(i.check == "dpo_extreme_length_ratio" for i in issues)

    def test_too_few_records_warning(self):
        records = [_dpo() for _ in range(5)]
        issues = check_dpo_quality(records, min_records=20)
        assert any(i.check == "dpo_min_records" and i.level == "warning" for i in issues)


class TestRunQualityGate:
    def _healthy_sft(self, n=120):
        return [_sft() for _ in range(n)] + [_sft("repair_patch") for _ in range(20)]

    def test_healthy_data_passes(self):
        report = run_quality_gate(
            sft_records=self._healthy_sft(),
            dpo_records=[_dpo() for _ in range(30)],
        )
        assert report.ok
        assert report.n_sft == 140
        assert report.n_dpo == 30

    def test_error_fails_gate(self):
        # identical DPO pairs → error
        report = run_quality_gate(
            sft_records=self._healthy_sft(),
            dpo_records=[_dpo(chosen="same.", rejected="same.")] * 5,
        )
        assert not report.ok
        assert len(report.errors) > 0

    def test_to_dict_serialisable(self):
        import json
        report = run_quality_gate(sft_records=self._healthy_sft())
        json.dumps(report.to_dict())  # must not raise

    def test_contamination_check_clean(self):
        report = run_quality_gate(
            sft_records=self._healthy_sft(),
            eval_texts=["Completely unrelated topic about astronomy and planets."],
            contamination_threshold=0.2,
        )
        assert report.ok

    def test_contamination_flagged_as_error(self):
        # Use a long enough text to produce 8-grams (need >=8 tokens)
        text = "the quick brown fox jumps over the lazy dog near the river bank"
        sft = [_sft(assistant_content=text) for _ in range(120)] + \
              [_sft("repair_patch") for _ in range(20)]
        report = run_quality_gate(
            sft_records=sft,
            eval_texts=[text],
            contamination_threshold=0.2,
        )
        assert not report.ok

    def test_print_report_does_not_raise(self, capsys):
        report = run_quality_gate(sft_records=self._healthy_sft())
        report.print_report()
        out = capsys.readouterr().out
        assert "PASS" in out or "FAIL" in out

    def test_none_inputs_ok(self):
        report = run_quality_gate()
        assert report.n_sft == 0
        assert report.n_dpo == 0
