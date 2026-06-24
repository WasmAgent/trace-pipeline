#!/usr/bin/env python3
"""Generate synthetic SFT/DPO data via Anthropic API with parallel calls.

Reduced-scale, low-token version:
  - n_per_template = 10  (50 total good outputs)
  - n_bad_per_template = 2  (10 bad outputs)
  - max_tokens = 300
  - 8 parallel workers
  - estimated time: ~2 min, ~800 API tokens per call

Produces ~100 SFT records + ~50 DPO pairs.
Combined with the 556 real IFEval records: ~650 total SFT.

Usage:
    python scripts/generate_synthetic_data_fast.py --out-dir data/synthetic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _make_client():
    try:
        import anthropic
    except ImportError:
        print("[error] pip install anthropic", file=sys.stderr)
        sys.exit(1)

    if os.environ.get("ANTHROPIC_AUTH_TOKEN") and os.environ.get("ANTHROPIC_BASE_URL"):
        return anthropic.Anthropic(
            auth_token=os.environ["ANTHROPIC_AUTH_TOKEN"],
            base_url=os.environ["ANTHROPIC_BASE_URL"],
        )
    elif os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    else:
        print("[error] set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN+ANTHROPIC_BASE_URL",
              file=sys.stderr)
        sys.exit(1)


def _call(client, model: str, max_tokens: int, messages: list[dict]) -> str:
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, messages=messages
    )
    return resp.content[0].text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--n-per-template", type=int, default=10,
                    help="good outputs per template (default: 10)")
    ap.add_argument("--n-bad-per-template", type=int, default=2,
                    help="bad outputs per template (default: 2)")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel API call threads")
    ap.add_argument("--model", default=None,
                    help="model ID (default: ANTHROPIC_DEFAULT_HAIKU_MODEL env var)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model = args.model or os.environ.get(
        "ANTHROPIC_DEFAULT_HAIKU_MODEL", "claude-haiku-latest"
    )

    from evomerge.synthesize.templates import builtin_templates
    from evomerge.synthesize.generator import (
        GenerationConfig, _good_prompt, _bad_prompt, _repair_prompt
    )
    from evomerge.schemas.training import (
        DpoTrainingRecord, Message, Provenance, SftTrainingRecord
    )
    import random

    templates = builtin_templates()
    rng = random.Random(42)

    violation_types = [
        "missing required sections",
        "wrong output language",
        "no action list",
        "tool results not cited in answer",
        "invalid tool arguments",
        "evidence insufficient or fabricated",
    ]

    # Build all jobs upfront
    jobs = []  # (kind, spec_name, spec, violation_or_none)
    for spec_name, spec in templates.items():
        for _ in range(args.n_per_template):
            jobs.append(("good", spec_name, spec, None))
        violations = rng.sample(violation_types, min(args.n_bad_per_template, len(violation_types)))
        for v in violations:
            jobs.append(("bad", spec_name, spec, v))

    total_calls = len(jobs) + sum(
        1 for kind, *_ in jobs if kind == "bad"
    )  # bad → also repair call
    print(f"model          : {model}")
    print(f"templates      : {len(templates)} ({list(templates.keys())})")
    print(f"good calls     : {sum(1 for j in jobs if j[0]=='good')}")
    print(f"bad+repair     : {sum(1 for j in jobs if j[0]=='bad')} × 2")
    print(f"total calls    : ~{total_calls}")
    print(f"workers        : {args.workers}")
    print(f"est. time      : ~{total_calls * 2.1 / args.workers:.0f}s")
    print(f"out_dir        : {args.out_dir}")

    if args.dry_run:
        print("\n[dry-run] no API calls")
        return 0

    client = _make_client()

    # Parallel execution
    sft_records: list[SftTrainingRecord] = []
    dpo_records: list[DpoTrainingRecord] = []

    good_outputs: dict[str, list[str]] = {n: [] for n in templates}

    print("\ngenerating good outputs...")
    good_jobs = [(spec_name, spec) for kind, spec_name, spec, _ in jobs if kind == "good"]

    def _run_good(spec_name, spec):
        msgs = _good_prompt(spec)
        text = _call(client, model, args.max_tokens, msgs)
        return spec_name, spec, text

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_good, sn, sp): (sn, sp) for sn, sp in good_jobs}
        for fut in as_completed(futures):
            spec_name, spec, text = fut.result()
            good_outputs[spec_name].append(text)
            prov = Provenance(source="synthetic-teacher", task_id=spec.id)
            sft_records.append(SftTrainingRecord(
                messages=[
                    Message(role="user", content=spec.intent),
                    Message(role="assistant", content=text),
                ],
                output_type="final_answer",
                provenance=prov,
            ))
            completed += 1
            if completed % 10 == 0:
                print(f"  good: {completed}/{len(good_jobs)}")

    print(f"  good done: {len(sft_records)} SFT records")

    print("generating bad + repair outputs...")
    bad_jobs = [
        (spec_name, spec, violation)
        for kind, spec_name, spec, violation in jobs if kind == "bad"
    ]

    def _run_bad_and_repair(spec_name, spec, violation):
        bad_text = _call(client, model, args.max_tokens, _bad_prompt(spec, violation))
        repair_text = _call(client, model, args.max_tokens,
                            _repair_prompt(spec, bad_text, violation))
        return spec_name, spec, violation, bad_text, repair_text

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_bad_and_repair, sn, sp, v): (sn, sp, v)
            for sn, sp, v in bad_jobs
        }
        for fut in as_completed(futures):
            spec_name, spec, violation, bad_text, repair_text = fut.result()
            prov = Provenance(source="synthetic-teacher", task_id=spec.id)

            # repair SFT record
            sft_records.append(SftTrainingRecord(
                messages=[
                    Message(role="user",
                            content=f"Task: {spec.intent}\nViolation: {violation}\nBad output:\n{bad_text}"),
                    Message(role="assistant", content=repair_text),
                ],
                output_type="repair_patch",
                loss_weight_tokens="recovery",
                provenance=prov,
            ))

            # DPO pair
            chosen_list = good_outputs.get(spec_name, [])
            if chosen_list:
                chosen = rng.choice(chosen_list)
                if chosen != bad_text:
                    dpo_records.append(DpoTrainingRecord(
                        messages=[
                            Message(role="user", content=spec.intent),
                            Message(role="assistant", content=chosen),
                        ],
                        prompt_messages=[Message(role="user", content=spec.intent)],
                        chosen=chosen,
                        rejected=bad_text,
                        provenance=prov,
                    ))
            completed += 1
            if completed % 5 == 0:
                print(f"  bad+repair: {completed}/{len(bad_jobs)}")

    print(f"  repair SFT: {sum(1 for r in sft_records if r.output_type=='repair_patch')}")
    print(f"  DPO pairs : {len(dpo_records)}")

    # validate
    from evomerge.validate.schema_check import validate_training_record
    n_invalid = sum(1 for r in sft_records + dpo_records if not validate_training_record(r).ok)
    print(f"  invalid   : {n_invalid}")

    # write
    from evomerge.io import write_jsonl
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_jsonl(sft_records, out / "sft.jsonl")
    write_jsonl(dpo_records, out / "dpo.jsonl")

    manifest = {
        "model": model,
        "n_templates": len(templates),
        "n_sft": len(sft_records),
        "n_dpo": len(dpo_records),
        "n_invalid": n_invalid,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n✓ {len(sft_records)} SFT + {len(dpo_records)} DPO → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
