"""Contamination check: flag training records whose output n-grams overlap
with evaluation items.

Algorithm:
  1. Both reference and candidate texts are Unicode-normalised with NFKC
     before comparison, so full-width characters, homoglyphs that collapse
     under NFKC, and compatibility equivalents are treated identically.
  2. Primary check: token-level 8-gram Jaccard similarity (configurable).
  3. Char-level fallback: character n-gram Jaccard for n in [5, 10] (step 1).
     If ANY char-gram size produces Jaccard >= char_threshold (default 0.7),
     the record is flagged even if the token-level check misses it.  This
     catches zero-width-stripped, punctuation-heavy, or short texts that
     produce too few token n-grams.

Returns a report with flagged record indices so callers can filter before
exporting.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence


def _normalize(text: str) -> str:
    """Apply NFKC Unicode normalisation and sanitise invisible characters.

    NFKC folds full-width Latin letters (ｉｇｎｏｒｅ → ignore), compatibility
    ligatures, and a number of look-alike codepoints to their canonical ASCII
    equivalents.  Zero-width characters that survive NFKC (U+200B, U+200C,
    U+200D, etc.) are replaced with a regular space so that word boundaries
    are preserved (e.g. "the​quick" becomes "the quick" rather than
    "thequick").
    """
    text = unicodedata.normalize("NFKC", text)
    # Replace zero-width and invisible formatting characters with a space so
    # word boundaries are preserved after the invisible chars are removed.
    _ZERO_WIDTH = (
        "​",  # ZERO WIDTH SPACE
        "‌",  # ZERO WIDTH NON-JOINER
        "‍",  # ZERO WIDTH JOINER
        "­",  # SOFT HYPHEN
        "⁠",  # WORD JOINER
        "﻿",  # ZERO WIDTH NO-BREAK SPACE / BOM
        "⁡",  # FUNCTION APPLICATION
        "⁢",  # INVISIBLE TIMES
        "⁣",  # INVISIBLE SEPARATOR
        "⁤",  # INVISIBLE PLUS
    )
    for ch in _ZERO_WIDTH:
        text = text.replace(ch, " ")
    # Collapse multiple spaces introduced by the replacements
    return " ".join(text.split())


def _ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    tokens = re.findall(r"\w+", text.lower())
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _char_ngrams(text: str, n: int) -> set[str]:
    """Character-level n-grams over the normalised, lowercased text."""
    t = text.lower()
    return {t[i : i + n] for i in range(len(t) - n + 1)} if len(t) >= n else set()


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _char_jaccard_max(
    text_a: str,
    text_b: str,
    n_min: int = 5,
    n_max: int = 10,
) -> float:
    """Return the maximum char-level Jaccard across n in [n_min, n_max]."""
    best = 0.0
    for n in range(n_min, n_max + 1):
        score = _jaccard(_char_ngrams(text_a, n), _char_ngrams(text_b, n))
        if score > best:
            best = score
    return best


@dataclass
class ContaminationReport:
    threshold: float
    n_training: int
    n_eval: int
    flagged: list[dict] = field(default_factory=list)

    @property
    def n_flagged(self) -> int:
        return len(self.flagged)

    @property
    def flag_rate(self) -> float:
        return self.n_flagged / self.n_training if self.n_training else 0.0


def check_contamination(
    training_texts: Sequence[str],
    eval_texts: Sequence[str],
    *,
    threshold: float = 0.2,
    ngram_size: int = 8,
    char_threshold: float = 0.7,
    char_n_min: int = 5,
    char_n_max: int = 10,
) -> ContaminationReport:
    """Check training outputs against eval items for n-gram overlap.

    Both ``training_texts`` and ``eval_texts`` are NFKC-normalised (and
    zero-width characters stripped) before any comparison, ensuring that
    full-width homoglyphs and invisible-character injections are detected.

    Args:
        training_texts: output texts from training records (e.g. SftTrainingRecord
            messages[-1].content).
        eval_texts: reference eval item texts.
        threshold: token-level Jaccard similarity above which a record is flagged.
        ngram_size: token n-gram size for the primary check.
        char_threshold: char-level Jaccard threshold for the fallback check.
            A record is also flagged when any char-gram size in
            [char_n_min, char_n_max] exceeds this threshold.
        char_n_min: minimum character n-gram size for the fallback.
        char_n_max: maximum character n-gram size for the fallback.

    Returns:
        ContaminationReport with indices and scores of flagged records.
    """
    # NFKC-normalise all eval texts once
    norm_eval = [_normalize(t) for t in eval_texts]
    eval_ngrams = [_ngrams(t, ngram_size) for t in norm_eval]

    report = ContaminationReport(
        threshold=threshold,
        n_training=len(training_texts),
        n_eval=len(eval_texts),
    )
    for idx, raw_text in enumerate(training_texts):
        text = _normalize(raw_text)
        train_ng = _ngrams(text, ngram_size)

        # Primary: token-level Jaccard
        max_token_score = max(
            (_jaccard(train_ng, eng) for eng in eval_ngrams), default=0.0
        )

        # Fallback: char-level Jaccard across n in [char_n_min, char_n_max]
        max_char_score = max(
            (_char_jaccard_max(text, et, char_n_min, char_n_max) for et in norm_eval),
            default=0.0,
        )

        if max_token_score >= threshold or max_char_score >= char_threshold:
            report.flagged.append({
                "index": idx,
                "jaccard": round(max_token_score, 4),
                "char_jaccard": round(max_char_score, 4),
            })
    return report
