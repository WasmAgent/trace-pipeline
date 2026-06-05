"""examples/recipe10_lm_eval_bridge.py — convert lm-eval-harness output.

Demonstrates the lm_eval_bridge with a synthetic JSONL file (so the
example runs without requiring an actual lm-evaluation-harness install).

In real use, you'd point to lm-eval-harness's samples_*.jsonl output:

    lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct \\
            --tasks gsm8k --output_path lm_out/instruct.json --log_samples

Run from repo root:
    python examples/recipe10_lm_eval_bridge.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval_trust.lm_eval_bridge import pair  # noqa: E402
from eval_trust.paired_stats import mcnemar_exact  # noqa: E402


def write_synthetic_jsonl(path: Path, accuracies: list[bool]) -> None:
    """Write a fake lm-eval-harness samples_*.jsonl with the given outcomes."""
    with open(path, "w") as f:
        for i, correct in enumerate(accuracies):
            row = {
                "doc_id": i,
                "target": "1",
                "filtered_resps": ["#### 1" if correct else "#### 9"],
                "acc": 1.0 if correct else 0.0,
            }
            f.write(json.dumps(row) + "\n")


# Make a 200-item synthetic experiment: A and B both ~50% accurate,
# correlated (rho=0.3), no real difference.
rng = np.random.default_rng(seed=0)
n = 200
z = rng.normal(size=n)
eps_a = rng.normal(size=n)
eps_b = rng.normal(size=n)
u_a = 0.55 * z + 0.83 * eps_a  # rho=0.3 with 0.55^2 + 0.83^2 ~= 1
u_b = 0.55 * z + 0.83 * eps_b
correct_a = (u_a > 0).tolist()
correct_b = (u_b > 0).tolist()

with tempfile.TemporaryDirectory() as td:
    a_path = Path(td) / "a.jsonl"
    b_path = Path(td) / "b.jsonl"
    write_synthetic_jsonl(a_path, correct_a)
    write_synthetic_jsonl(b_path, correct_b)

    out = pair(a_path, b_path)
    p = mcnemar_exact(out["b"], out["c"])
    print(f"n={out['n_common']}  delta={out['delta_pp']:+.1f}pp  "
          f"b={out['b']} c={out['c']} p={p:.4f}")
    print()
    print("On synthetic-no-signal data, McNemar p is large (no significant")
    print("difference). pair() lined up both lm-eval outputs by doc_id and")
    print("emitted the discordant counts ready for paired_stats.")
