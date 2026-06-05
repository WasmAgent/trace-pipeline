"""eval_trust.t0v2.truncation_extract — detect truncated generations.

Channel A_truncated of the T0v2 audit: a wrong answer is *truncated* if
its gen_text approaches the max_new_tokens cap and lacks a parseable
"#### N" answer line. This is the case-study primitive in section 3
of the paper.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ANSWER_LINE = re.compile(r"####\s*-?\d[\d,\.]*")


def is_truncated(item: dict, max_new_tokens: int, char_threshold: int = 20) -> bool:
    """Return True if item's gen_text looks truncated."""
    text = item.get("gen_text", "") or ""
    if ANSWER_LINE.search(text):
        return False
    # rough: if length is at least max_new_tokens * 2 chars, treat as full
    return len(text) >= max(max_new_tokens * 2 - char_threshold, 0)


def classify(input_path: Path, out_path: Path | None = None) -> dict:
    with open(input_path) as f:
        d = json.load(f)
    max_new = d.get("meta", {}).get("max_new_tokens", 768)

    truncated = []
    not_truncated_wrong = []
    for r in d.get("results", []):
        if r.get("correct", True):
            continue
        if is_truncated(r, max_new):
            truncated.append(r["id"])
        else:
            not_truncated_wrong.append(r["id"])

    n_total = len(d.get("results", []))
    n_wrong = len(truncated) + len(not_truncated_wrong)
    out = {
        "input": str(input_path),
        "max_new_tokens": max_new,
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
    args = ap.parse_args()

    out = classify(Path(args.input), Path(args.out) if args.out else None)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
