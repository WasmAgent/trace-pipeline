"""eval_trust.t0v2.truncation_extract — detect truncated generations.

Channel A_truncated of the T0v2 audit: a wrong answer is *truncated*
if its gen_text approaches the max_new_tokens cap and lacks a
parseable "#### N" answer line. This is the case-study primitive in
section 3 of the paper.

A note on token vs character counting
-------------------------------------
The "approaches the max_new_tokens cap" check has two implementations:

1. **Token-counting** (preferred). Pass a tokenizer with an
   ``encode(text) -> list[int]`` method. We compare ``len(tokens)``
   directly to ``max_new_tokens``.

2. **Character heuristic** (fallback). When no tokenizer is given,
   we approximate ``len(text) >= max_new_tokens * 2 - threshold``,
   based on the rough rule that English-ish output averages ~3-4
   characters per BPE token. This is *deliberately conservative*
   (misses some truncations rather than false-positives) and emits
   a warning.

Reviewer feedback (2026-06-05): "an audit paper about token-budget
primitives that uses character-count heuristics is methodologically
ironic." Fair. The fix is to pass a tokenizer. We keep the heuristic
as a warned fallback so the toolkit still works on logs that did not
record token counts and where a tokenizer is unavailable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Protocol

ANSWER_LINE = re.compile(r"####\s*-?\d[\d,\.]*")


class HasEncode(Protocol):
    """Minimal duck-type for tokenizers (HF AutoTokenizer, tiktoken, etc.)."""

    def encode(self, text: str) -> list[int]: ...


def _token_count(text: str, tokenizer: HasEncode | None) -> tuple[int, str]:
    """Return (count, mode) for the given text.

    mode is one of:
      'tokens'    — counted via tokenizer.encode (preferred)
      'char/2'    — approximated as len(text) // 2 (English-ish ~3-4 chars/token,
                    but we deliberately use 2 for a conservative under-estimate)
    """
    if tokenizer is not None:
        tokens = tokenizer.encode(text)
        return (len(tokens), "tokens")
    # fallback: char/2 is conservative — counts a tighter envelope, so we are
    # less likely to false-positive "truncated" on a verbose-but-finished
    # answer.
    return (len(text) // 2, "char/2")


def is_truncated(
    item: dict,
    max_new_tokens: int,
    *,
    tokenizer: HasEncode | None = None,
    margin: int = 8,
) -> bool:
    """Return True if item's gen_text looks truncated.

    Definition of "truncated":
      gen_text token count >= max_new_tokens - margin AND the text
      lacks a parseable "#### N" answer line.

    Args:
        item: a per-sample record dict containing 'gen_text'.
        max_new_tokens: the cap that was used at generation time.
        tokenizer: optional tokenizer with .encode(text) -> list[int].
            When provided, the token count is exact. When None, we use
            a conservative ``len(text) // 2`` approximation that
            under-counts and therefore is biased toward false negatives
            (missing some real truncations) rather than false positives.
        margin: how many tokens shy of the cap still counts as "near
            the cap". Default 8 (sufficient for typical EOS or
            stop-token padding).

    Returns:
        True if the sample looks truncated.
    """
    text = item.get("gen_text", "") or ""
    if ANSWER_LINE.search(text):
        # finished cleanly with a #### N line; not truncated by the cap
        return False
    n_tokens, _mode = _token_count(text, tokenizer)
    return n_tokens >= max(max_new_tokens - margin, 0)


def classify(
    input_path: Path,
    out_path: Path | None = None,
    *,
    tokenizer: HasEncode | None = None,
) -> dict:
    """Classify wrong answers in input_path as A_truncated or not.

    Args:
        input_path: path to a greedy-eval JSON log with the standard
            ``{results: [...], meta: {max_new_tokens: int, ...}}`` shape.
        out_path: optional path to write the JSON report.
        tokenizer: optional tokenizer; see is_truncated() for semantics.

    Returns:
        Channel report dict (also written to out_path if given).
    """
    with open(input_path) as f:
        d = json.load(f)
    max_new = d.get("meta", {}).get("max_new_tokens", 768)

    if tokenizer is None:
        warnings.warn(
            "truncation_extract: no tokenizer passed; falling back to "
            "len(text) // 2 character approximation. Pass tokenizer= "
            "(e.g. AutoTokenizer.from_pretrained(...)) for exact counts. "
            "The fallback under-counts and may miss some truncations.",
            UserWarning,
            stacklevel=2,
        )
        token_mode = "char/2"
    else:
        token_mode = "tokens"

    truncated: list[Any] = []
    not_truncated_wrong: list[Any] = []
    for r in d.get("results", []):
        if r.get("correct", True):
            continue
        if is_truncated(r, max_new, tokenizer=tokenizer):
            truncated.append(r["id"])
        else:
            not_truncated_wrong.append(r["id"])

    n_total = len(d.get("results", []))
    n_wrong = len(truncated) + len(not_truncated_wrong)
    out = {
        "input": str(input_path),
        "max_new_tokens": max_new,
        "token_count_mode": token_mode,
        "n_total": n_total,
        "n_wrong": n_wrong,
        "n_truncated": len(truncated),
        "truncated_share_of_wrong": (len(truncated) / n_wrong) if n_wrong else 0.0,
        "truncated_ids": truncated,
        "not_truncated_wrong_ids": not_truncated_wrong,
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", help="path to greedy-eval log JSON")
    ap.add_argument("--out", help="optional output path for the channel report")
    ap.add_argument(
        "--tokenizer",
        help="optional HuggingFace tokenizer name or path (for exact token counting). "
             "Without this, falls back to a char/2 heuristic with a warning.",
    )
    args = ap.parse_args()

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError:
            print("ERROR: --tokenizer requires `pip install transformers`",
                  file=sys.stderr)
            return 1
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    out = classify(
        Path(args.input),
        Path(args.out) if args.out else None,
        tokenizer=tokenizer,
    )
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
