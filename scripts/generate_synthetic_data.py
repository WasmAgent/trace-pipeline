#!/usr/bin/env python3
"""Generate synthetic SFT training data via Anthropic API.

Calls SyntheticGenerator with all three MVP task types:
  markdown_report — requires sections, language, action list, evidence
  tool_call       — allowed tools, schema-valid args, results in answer
  repair          — minimal patch only, no full rewrite

Output:
  {out_dir}/sft.jsonl        (SFT records from good outputs)
  {out_dir}/dpo.jsonl        (DPO pairs from good vs bad outputs)
  {out_dir}/manifest.json    (counts + contamination report)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...

    python scripts/generate_synthetic_data.py \\
        --n-per-type 200 \\
        --out-dir data/synthetic

    # dry-run: print config only, make no API calls
    python scripts/generate_synthetic_data.py --n-per-type 5 --dry-run

Apple Silicon notes:
    API calls are local, no GPU needed. Total calls ~= n_per_type * 3 task_types
    * (1 good + n_bad_per_template violations) ≈ 200*3*7 = 4200 calls.
    At ~0.5s/call that's ~35 minutes. Use --n-per-type 50 for a quick test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _make_chat_fn(model: str, max_tokens: int):
    try:
        import anthropic
    except ImportError:
        print("[error] pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()

    def chat_fn(messages: list[dict]) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return resp.content[0].text

    return chat_fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default="data/synthetic", metavar="DIR")
    ap.add_argument("--n-per-type", type=int, default=200,
                    help="good outputs per task type (total ≈ n * 3 types)")
    ap.add_argument("--n-bad-per-type", type=int, default=5,
                    help="bad outputs per task type (for DPO rejected side)")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="Anthropic model ID (haiku is fast+cheap for bulk synthesis)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-items", metavar="FILE",
                    help="JSONL with eval items for contamination check")
    ap.add_argument("--dry-run", action="store_true",
                    help="print config and exit, make no API calls")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[error] ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    from evomerge.synthesize.generator import GenerationConfig, SyntheticGenerator
    from evomerge.synthesize.templates import TaskType, make_task_spec, builtin_templates
    from evomerge.validate.contamination import check_contamination
    from evomerge.io import write_jsonl

    cfg = GenerationConfig(
        teacher_model=args.model,
        n_per_template=args.n_per_type,
        n_bad_per_template=args.n_bad_per_type,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # Use all five built-in templates (covers both report and tool_call variants)
    templates = builtin_templates()

    est_calls = len(templates) * (args.n_per_type + args.n_bad_per_type * 2)
    print(f"config:")
    print(f"  model          : {args.model}")
    print(f"  templates      : {len(templates)} ({list(templates.keys())})")
    print(f"  n_per_type     : {args.n_per_type}")
    print(f"  n_bad_per_type : {args.n_bad_per_type}")
    print(f"  est. API calls : ~{est_calls}")
    print(f"  out_dir        : {args.out_dir}")

    if args.dry_run:
        print("\n[dry-run] no API calls made")
        return 0

    print("\ngenerating...")
    chat_fn = _make_chat_fn(args.model, args.max_tokens)
    gen = SyntheticGenerator(chat_fn=chat_fn, config=cfg)

    sft, dpo = gen.generate(templates)
    print(f"  SFT records : {len(sft)}")
    print(f"  DPO pairs   : {len(dpo)}")

    # contamination check
    n_contaminated = 0
    if args.eval_items:
        eval_texts = []
        with open(args.eval_items) as fh:
            for line in fh:
                d = json.loads(line)
                eval_texts.append(d.get("text") or d.get("task") or d.get("artifact") or "")
        eval_texts = [t for t in eval_texts if t]
        if eval_texts:
            outputs = [r.messages[-1].content for r in sft] + [r.chosen for r in dpo]
            report = check_contamination(outputs, eval_texts, threshold=0.2)
            n_contaminated = report.n_flagged
            print(f"  contaminated: {n_contaminated}/{report.n_training} (threshold=0.2)")

    # validate
    from evomerge.validate.schema_check import validate_training_record
    n_invalid = sum(
        1 for rec in sft + dpo if not validate_training_record(rec).ok
    )
    print(f"  invalid     : {n_invalid}")

    # write
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if sft:
        write_jsonl(sft, out / "sft.jsonl")
        print(f"  wrote {len(sft)} SFT → {out}/sft.jsonl")

    if dpo:
        write_jsonl(dpo, out / "dpo.jsonl")
        print(f"  wrote {len(dpo)} DPO → {out}/dpo.jsonl")

    manifest = {
        "model": args.model,
        "n_templates": len(templates),
        "template_names": list(templates.keys()),
        "n_sft": len(sft),
        "n_dpo": len(dpo),
        "n_invalid": n_invalid,
        "n_contaminated": n_contaminated,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n✓ done — {len(sft) + len(dpo)} total records in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
