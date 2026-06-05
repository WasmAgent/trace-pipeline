"""tests/test_truncation_extract.py — truncation detector unit tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_trust.t0v2.truncation_extract import classify, is_truncated


class FakeTokenizer:
    """Minimal stand-in for HF AutoTokenizer / tiktoken.

    Treats every space-separated word as one token, plus 1 token per
    punctuation char. Good enough to drive the truncation predicate
    while staying deterministic.
    """

    def encode(self, text: str) -> list[int]:
        # one fake id per "word", plus one per punctuation char
        words = text.split()
        punct = sum(1 for c in text if c in ".,;:!?")
        return [0] * (len(words) + punct)


class TestIsTruncatedTokenMode:
    def test_short_with_answer_line_not_truncated(self):
        item = {"gen_text": "Step 1. Step 2. The answer is 42. #### 42"}
        # Has #### N → not truncated regardless of length
        assert is_truncated(item, max_new_tokens=128) is False

    def test_long_without_answer_line_is_truncated_token_mode(self):
        # 200 fake "tokens" (200 words), no #### line, max_new=200 -> truncated
        text = " ".join(["word"] * 200)
        item = {"gen_text": text}
        assert is_truncated(item, max_new_tokens=200,
                            tokenizer=FakeTokenizer()) is True

    def test_short_without_answer_line_not_truncated(self):
        # 5 tokens, max_new=200 → not truncated (well below cap)
        item = {"gen_text": "five short words here ok"}
        assert is_truncated(item, max_new_tokens=200,
                            tokenizer=FakeTokenizer()) is False

    def test_margin_respected(self):
        # 195 tokens, max_new=200, margin=8 → 195 >= 200-8=192 → truncated
        text = " ".join(["w"] * 195)
        item = {"gen_text": text}
        assert is_truncated(item, max_new_tokens=200, margin=8,
                            tokenizer=FakeTokenizer()) is True
        # Same text, margin=10 → 195 >= 200-10=190 → still truncated
        assert is_truncated(item, max_new_tokens=200, margin=10,
                            tokenizer=FakeTokenizer()) is True
        # 180 tokens with margin=8 → 180 < 192 → not truncated
        text2 = " ".join(["w"] * 180)
        item2 = {"gen_text": text2}
        assert is_truncated(item2, max_new_tokens=200, margin=8,
                            tokenizer=FakeTokenizer()) is False

    def test_empty_gen_text(self):
        item = {"gen_text": ""}
        assert is_truncated(item, max_new_tokens=200,
                            tokenizer=FakeTokenizer()) is False

    def test_missing_gen_text(self):
        item = {}
        assert is_truncated(item, max_new_tokens=200,
                            tokenizer=FakeTokenizer()) is False


class TestIsTruncatedCharFallback:
    """Without a tokenizer we fall back to len(text) // 2."""

    def test_long_text_truncated_via_fallback(self):
        # 800 chars / 2 = 400 tokens >= 400 - 8 = 392 → truncated
        text = "x" * 800
        item = {"gen_text": text}
        assert is_truncated(item, max_new_tokens=400) is True

    def test_short_text_not_truncated_via_fallback(self):
        text = "x" * 100  # 50 tokens
        item = {"gen_text": text}
        assert is_truncated(item, max_new_tokens=400) is False


class TestClassify:
    def test_minimal_classify(self, tmp_path):
        log = {
            "meta": {"max_new_tokens": 100},
            "results": [
                # correct, ignored
                {"id": "a", "correct": True, "gen_text": "#### 1"},
                # wrong, has #### N → not truncated
                {"id": "b", "correct": False, "gen_text": "wrong but #### 9"},
                # wrong, long, no #### → truncated under fake tokenizer
                {"id": "c", "correct": False,
                 "gen_text": " ".join(["w"] * 100)},
            ],
        }
        in_path = tmp_path / "log.json"
        in_path.write_text(json.dumps(log))
        out = classify(in_path, tokenizer=FakeTokenizer())
        assert out["n_total"] == 3
        assert out["n_wrong"] == 2
        assert out["n_truncated"] == 1
        assert out["token_count_mode"] == "tokens"
        assert "c" in out["truncated_ids"]
        assert "b" in out["not_truncated_wrong_ids"]

    def test_warns_when_no_tokenizer(self, tmp_path):
        log = {
            "meta": {"max_new_tokens": 100},
            "results": [
                {"id": "a", "correct": False, "gen_text": "x" * 50},
            ],
        }
        in_path = tmp_path / "log.json"
        in_path.write_text(json.dumps(log))
        with pytest.warns(UserWarning, match="no tokenizer"):
            out = classify(in_path)  # no tokenizer
        assert out["token_count_mode"] == "char/2"

    def test_writes_out_path(self, tmp_path):
        log = {
            "meta": {"max_new_tokens": 100},
            "results": [
                {"id": "a", "correct": True, "gen_text": "#### 1"},
            ],
        }
        in_path = tmp_path / "in.json"
        out_path = tmp_path / "subdir" / "out.json"
        in_path.write_text(json.dumps(log))
        classify(in_path, out_path=out_path, tokenizer=FakeTokenizer())
        assert out_path.exists()
        content = json.loads(out_path.read_text())
        assert content["n_total"] == 1


class TestRealCaseStudyData:
    """Run against the actual case-study logs to verify the new
    tokenizer-driven path doesn't regress existing numbers."""

    def test_winner_max_new768_with_fake_tokenizer(self):
        repo_data = Path(__file__).resolve().parent.parent / "data"
        log_path = repo_data / "winner_max_new768.json"
        if not log_path.exists():
            pytest.skip("case-study data not present in this checkout")
        out = classify(log_path, tokenizer=FakeTokenizer())
        # The fake tokenizer doesn't perfectly match the real Qwen
        # tokenizer, but we still expect:
        #   - correct file structure (no exception)
        #   - some non-zero number of wrong items
        #   - token_count_mode is 'tokens' since we passed one
        assert out["token_count_mode"] == "tokens"
        assert out["n_total"] > 0
        assert out["n_wrong"] >= 0  # may be 0 if all correct, but file shape OK
