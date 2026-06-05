"""examples/recipe5_t0v2_truncation.py — A_truncated channel detector.

Run from repo root:
    python examples/recipe5_t0v2_truncation.py
    python examples/recipe5_t0v2_truncation.py --tokenizer Qwen/Qwen2.5-1.5B

By default this falls back to a ``len(text) // 2`` character heuristic
and emits a UserWarning (which is the point — see the paper's section
on measurement primitives). Pass ``--tokenizer HF_NAME`` to count
actual BPE tokens via ``transformers.AutoTokenizer``.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_trust.t0v2.truncation_extract import classify  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--tokenizer",
        help="HF tokenizer name (e.g. Qwen/Qwen2.5-1.5B). "
             "If omitted, uses char/2 fallback and prints the warning.",
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
        print(f"Using real tokenizer: {args.tokenizer}\n")
    else:
        # Suppress the warning here just so the example output stays
        # clean; in real usage you'd want to see it.
        warnings.simplefilter("ignore", UserWarning)
        print("Using char/2 fallback (pass --tokenizer HF_NAME for "
              "exact token counts).\n")

    out = classify(DATA / "winner_max_new768.json", tokenizer=tokenizer)

    print(f"Token count mode: {out['token_count_mode']}")
    print(f"Wrong items: {out['n_wrong']}")
    print(f"Truncated:   {out['n_truncated']}  "
          f"({out['truncated_share_of_wrong']*100:.1f}% of wrong)")
    print(f"First 10 truncated IDs: {out['truncated_ids'][:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
