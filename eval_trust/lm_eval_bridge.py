"""eval_trust.lm_eval_bridge — adapter from lm-evaluation-harness output to paired_stats.

The recommended workflow in the paper (sec 6.3) is:

  1. Run lm-eval-harness for the *standard* benchmark metric (gives you
     a single accuracy number on the standardised protocol).
  2. Run eval_trust on top for the *audit* layer (gives you paired
     McNemar, T0v2 channels, lottery rate).

This module is the bridge: it converts lm-eval-harness's per-sample
output JSONL into the format paired_stats and t0v2 expect, so step 2
runs on real data without re-evaluation.

Note: this is a thin compatibility shim. lm-evaluation-harness is NOT
a dependency of this package — you can install it separately and run
its CLI; we only consume its output files.

Usage:
    # 1. Run lm-eval-harness yourself
    lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-1.5B \\
            --tasks gsm8k --output_path lm_eval_out/qwen.json \\
            --log_samples

    # 2. Convert to eval_trust format
    from eval_trust.lm_eval_bridge import convert
    log = convert("lm_eval_out/qwen/samples_gsm8k_*.jsonl")

    # 3. Use with paired_stats / t0v2
    from eval_trust.paired_stats import mcnemar_exact
    ...

Reference: https://github.com/EleutherAI/lm-evaluation-harness
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def convert(samples_jsonl: str | Path) -> dict:
    """Convert lm-eval-harness samples_*.jsonl to eval_trust format.

    lm-eval-harness emits one JSON object per sample to a JSONL file.
    Each object typically includes:
      - doc_id (or similar identifier)
      - target / answer
      - resps (list of model generations)
      - filtered_resps (post-filter generations)
      - acc / exact_match (the per-sample correctness signal)

    eval_trust expects:
      {"results": [
          {"id": str, "expected": str, "predicted": str,
           "correct": bool, "gen_text": str},
          ...],
       "meta": {"n": int, "n_correct": int, ...}}

    Args:
        samples_jsonl: path to lm-eval-harness samples_*.jsonl file.
            Glob patterns are not expanded — pass a concrete file.

    Returns:
        dict in eval_trust format. Suitable to feed into
        paired_stats.mcnemar_exact (after pairing two such dicts) or
        t0v2.truncation_extract / aggregate.

    Raises:
        FileNotFoundError: if the JSONL doesn't exist.
        ValueError: if no samples are found.
    """
    p = Path(samples_jsonl)
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")

    results: list[dict[str, Any]] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            converted = _convert_sample(sample)
            if converted is not None:
                results.append(converted)

    if not results:
        raise ValueError(f"no samples found in {p}")

    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    return {
        "results": results,
        "meta": {
            "source": str(p),
            "source_format": "lm-evaluation-harness",
            "n": n,
            "n_correct": n_correct,
            "acc": n_correct / n,
        },
    }


def _convert_sample(sample: dict) -> dict | None:
    """Convert one lm-eval-harness sample to eval_trust per-item format.

    lm-eval-harness's exact field names vary by version and task. We
    handle the most common patterns; if a field is missing, return None
    rather than crashing — the caller can decide what to do.
    """
    # ID extraction (try common field names; explicit None checks because
    # doc_id == 0 is a valid ID and would be skipped by `or`-chains)
    item_id = None
    for key in ("doc_id", "id", "sample_id"):
        if key in sample and sample[key] is not None:
            item_id = sample[key]
            break
    if item_id is None:
        doc = sample.get("doc") or {}
        if doc.get("idx") is not None:
            item_id = doc["idx"]
    if item_id is None:
        return None
    item_id = str(item_id)

    # Expected answer
    expected = (
        sample.get("target")
        or sample.get("answer")
        or (sample.get("doc", {}) or {}).get("answer")
    )

    # Generated text — prefer filtered, fall back to raw
    resps = sample.get("filtered_resps") or sample.get("resps") or []
    if isinstance(resps, list) and resps:
        gen = resps[0]
        if isinstance(gen, list):  # nested [[text]] format
            gen = gen[0] if gen else ""
        gen_text = str(gen)
    else:
        gen_text = ""

    # Correctness — try the aggregator's output first
    correct = sample.get("acc")
    if correct is None:
        correct = sample.get("exact_match")
    if correct is None:
        # Fall back to: expected appears in gen_text
        if expected and gen_text:
            correct = str(expected).strip() in gen_text
        else:
            correct = False
    correct = bool(correct)

    # Predicted: best effort to extract a single answer-like token
    predicted = _extract_answer(gen_text)

    return {
        "id": item_id,
        "expected": str(expected) if expected is not None else None,
        "predicted": predicted,
        "correct": correct,
        "gen_text": gen_text,
    }


def _extract_answer(text: str) -> str | None:
    """Extract the final answer from a chain-of-thought string.

    Tries (in order): GSM8K-style ```#### N```, last number in text.
    Mirrors the heuristic in eval_trust.t0v2.truncation_extract.
    """
    import re
    if not text:
        return None
    # GSM8K-style "#### N"
    m = re.search(r"####\s*(-?\d[\d,\.]*)", text)
    if m:
        return m.group(1).replace(",", "")
    # Last numeric token
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None


def pair(
    a_jsonl: str | Path,
    b_jsonl: str | Path,
) -> dict:
    """Convenience: convert two lm-eval JSONLs and emit a paired summary.

    Returns a dict with the standard eval_trust pair-audit shape:
      {"a_results": ..., "b_results": ...,
       "n_common": int, "b": int, "c": int, "delta_pp": float}

    To go from this to a McNemar p-value, pass b/c to
    eval_trust.paired_stats.mcnemar_exact.
    """
    a = convert(a_jsonl)
    b = convert(b_jsonl)
    am = {r["id"]: r["correct"] for r in a["results"]}
    bm = {r["id"]: r["correct"] for r in b["results"]}
    common = sorted(set(am) & set(bm))
    n = len(common)
    a_correct = sum(am[i] for i in common)
    b_correct = sum(bm[i] for i in common)
    b_only = sum(1 for i in common if am[i] and not bm[i])
    c_only = sum(1 for i in common if not am[i] and bm[i])
    return {
        "a_results": a,
        "b_results": b,
        "n_common": n,
        "a_acc": a_correct / n if n else 0.0,
        "b_acc": b_correct / n if n else 0.0,
        "delta_pp": (b_correct - a_correct) / n * 100 if n else 0.0,
        "b": b_only,
        "c": c_only,
    }


__all__ = ["convert", "pair"]
