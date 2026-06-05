"""tests/test_lm_eval_bridge.py — lm-evaluation-harness adapter tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from eval_trust.lm_eval_bridge import _convert_sample, _extract_answer, convert, pair


def _write_jsonl(rows: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".jsonl", encoding="utf-8",
    )
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return Path(f.name)


class TestExtractAnswer:
    def test_gsm8k_style(self):
        text = "Let me think... 5 + 3 = 8.\n#### 8"
        assert _extract_answer(text) == "8"

    def test_gsm8k_with_comma(self):
        text = "Calculations done. #### 1,234"
        assert _extract_answer(text) == "1234"

    def test_negative_number(self):
        text = "After subtraction: #### -42"
        assert _extract_answer(text) == "-42"

    def test_fallback_last_number(self):
        text = "The result is 7 after several steps."
        assert _extract_answer(text) == "7"

    def test_empty_returns_none(self):
        assert _extract_answer("") is None

    def test_no_numbers_returns_none(self):
        assert _extract_answer("just text, no numerals") is None


class TestConvertSample:
    def test_minimal_gsm8k_format(self):
        sample = {
            "doc_id": 0,
            "target": "8",
            "filtered_resps": ["thinking... #### 8"],
            "acc": 1.0,
        }
        out = _convert_sample(sample)
        assert out is not None
        assert out["id"] == "0"
        assert out["expected"] == "8"
        assert out["predicted"] == "8"
        assert out["correct"] is True

    def test_missing_id_returns_none(self):
        sample = {"target": "5", "filtered_resps": ["#### 5"], "acc": 1.0}
        assert _convert_sample(sample) is None

    def test_nested_resps_list(self):
        sample = {
            "doc_id": 1,
            "target": "42",
            "resps": [["#### 42"]],
            "exact_match": 1.0,
        }
        out = _convert_sample(sample)
        assert out["correct"] is True
        assert out["gen_text"] == "#### 42"

    def test_acc_field_priority_over_exact_match(self):
        # If both are set, acc wins
        sample = {
            "doc_id": 2,
            "target": "1",
            "filtered_resps": ["wrong answer"],
            "acc": 1.0,  # claims correct
            "exact_match": 0.0,  # but exact_match disagrees
        }
        out = _convert_sample(sample)
        assert out["correct"] is True

    def test_fallback_correctness_via_substring(self):
        # No acc/exact_match field, but expected appears in text
        sample = {
            "doc_id": 3,
            "target": "100",
            "filtered_resps": ["the answer is 100, give or take"],
        }
        out = _convert_sample(sample)
        assert out["correct"] is True

    def test_doc_idx_path(self):
        sample = {
            "doc": {"idx": "abc"},
            "target": "1",
            "filtered_resps": ["#### 1"],
            "acc": 1.0,
        }
        out = _convert_sample(sample)
        assert out["id"] == "abc"


class TestConvert:
    def test_round_trip(self):
        rows = [
            {"doc_id": 0, "target": "5", "filtered_resps": ["#### 5"], "acc": 1.0},
            {"doc_id": 1, "target": "8", "filtered_resps": ["#### 7"], "acc": 0.0},
            {"doc_id": 2, "target": "3", "filtered_resps": ["#### 3"], "acc": 1.0},
        ]
        p = _write_jsonl(rows)
        try:
            out = convert(p)
            assert out["meta"]["n"] == 3
            assert out["meta"]["n_correct"] == 2
            assert out["meta"]["acc"] == 2 / 3
            assert out["meta"]["source_format"] == "lm-evaluation-harness"
            ids = [r["id"] for r in out["results"]]
            assert ids == ["0", "1", "2"]
        finally:
            p.unlink()

    def test_skips_empty_lines(self):
        # Manual JSONL with blank lines
        f = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".jsonl", encoding="utf-8",
        )
        f.write('{"doc_id": 0, "target": "1", "acc": 1.0}\n')
        f.write("\n")  # blank line
        f.write('   \n')  # whitespace-only line
        f.write('{"doc_id": 1, "target": "2", "acc": 0.0}\n')
        f.close()
        p = Path(f.name)
        try:
            out = convert(p)
            assert out["meta"]["n"] == 2
        finally:
            p.unlink()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            convert("/nonexistent/path.jsonl")

    def test_empty_file_raises_value_error(self):
        f = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".jsonl", encoding="utf-8",
        )
        f.write("")
        f.close()
        p = Path(f.name)
        try:
            with pytest.raises(ValueError):
                convert(p)
        finally:
            p.unlink()


class TestPair:
    def test_paired_audit_shape(self):
        a_rows = [
            {"doc_id": 0, "target": "1", "acc": 1.0, "filtered_resps": ["#### 1"]},
            {"doc_id": 1, "target": "2", "acc": 0.0, "filtered_resps": ["#### 9"]},
            {"doc_id": 2, "target": "3", "acc": 1.0, "filtered_resps": ["#### 3"]},
        ]
        b_rows = [
            {"doc_id": 0, "target": "1", "acc": 1.0, "filtered_resps": ["#### 1"]},
            {"doc_id": 1, "target": "2", "acc": 1.0, "filtered_resps": ["#### 2"]},
            {"doc_id": 2, "target": "3", "acc": 0.0, "filtered_resps": ["#### 7"]},
        ]
        a_path = _write_jsonl(a_rows)
        b_path = _write_jsonl(b_rows)
        try:
            out = pair(a_path, b_path)
            assert out["n_common"] == 3
            # A is correct on items 0, 2; B on 0, 1
            # discordant: A correct & B wrong on item 2 (b=1)
            #             A wrong & B correct on item 1 (c=1)
            assert out["b"] == 1  # A-only correct
            assert out["c"] == 1  # B-only correct
            assert out["a_acc"] == 2 / 3
            assert out["b_acc"] == 2 / 3
            assert out["delta_pp"] == 0.0
        finally:
            a_path.unlink()
            b_path.unlink()

    def test_pair_with_mcnemar(self):
        """End-to-end: convert + pair + McNemar."""
        from eval_trust.paired_stats import mcnemar_exact

        a_rows = [
            {"doc_id": i, "target": "1",
             "acc": 1.0 if i < 50 else 0.0,
             "filtered_resps": ["#### 1"]}
            for i in range(100)
        ]
        b_rows = [
            {"doc_id": i, "target": "1",
             "acc": 1.0 if i % 2 == 0 else 0.0,
             "filtered_resps": ["#### 1"]}
            for i in range(100)
        ]
        a_path = _write_jsonl(a_rows)
        b_path = _write_jsonl(b_rows)
        try:
            out = pair(a_path, b_path)
            p = mcnemar_exact(out["b"], out["c"])
            assert 0 <= p <= 1
        finally:
            a_path.unlink()
            b_path.unlink()
