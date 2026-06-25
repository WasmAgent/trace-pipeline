"""evomerge CLI — python -m evomerge <command> [options]

Commands:
  export          Convert rollout / compliance traces to training JSONL
  adp-export      Convert rollout-wire/v1 to ADP (Agent Data Protocol) JSONL
  rl-export       Convert rollout-wire/v1 to RL transition records JSONL
  compile-context Convert rollout traces to long-context QA or router/critic records
  router          Predict routing labels for a batch of router records
  synthesize      Generate synthetic SFT/DPO samples via a teacher model
  validate        Run contamination and schema checks on training JSONL

Run `python -m evomerge <command> --help` for per-command options.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def _cmd_export(args: argparse.Namespace) -> int:
    from evomerge.export import run_export

    eval_texts = None
    if args.eval_items:
        p = Path(args.eval_items)
        if not p.exists():
            print(f"[error] eval-items file not found: {p}", file=sys.stderr)
            return 1
        with open(p) as fh:
            eval_texts = [
                json.loads(line).get("text", json.loads(line).get("task", ""))
                for line in fh
                if line.strip() and not line.startswith("#")
            ]

    manifest = run_export(
        rollout_jsonl=args.rollout or None,
        compliance_jsonl=args.compliance or None,
        out_dir=args.out_dir,
        eval_texts=eval_texts,
        contamination_threshold=args.contamination_threshold,
        only_passing_sft=not args.include_failing,
    )
    print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# adp-export
# ---------------------------------------------------------------------------

def _cmd_adp_export(args: argparse.Namespace) -> int:
    from evomerge.adp.export import rollout_file_to_adp
    import dataclasses

    out = args.out or None
    steps = rollout_file_to_adp(args.rollout, out=out)
    if out is None:
        import json
        for step in steps:
            print(json.dumps(dataclasses.asdict(step), ensure_ascii=False))
    else:
        print(f"[ok] wrote {len(steps)} ADP steps to {out}")
    return 0


# ---------------------------------------------------------------------------
# rl-export
# ---------------------------------------------------------------------------

def _cmd_rl_export(args: argparse.Namespace) -> int:
    from evomerge.rl.export import rollout_file_to_rl_transitions
    import dataclasses

    dims = [d.strip() for d in args.reward.split(",") if d.strip()] if args.reward else None
    out = args.out or None
    transitions = rollout_file_to_rl_transitions(args.rollout, reward_dims=dims, out=out)
    if out is None:
        import json
        for t in transitions:
            print(json.dumps(dataclasses.asdict(t), ensure_ascii=False))
    else:
        print(f"[ok] wrote {len(transitions)} RL transitions to {out}")
    return 0


# ---------------------------------------------------------------------------
# compile-context
# ---------------------------------------------------------------------------

def _cmd_compile_context(args: argparse.Namespace) -> int:
    from evomerge.context_compile.compiler import compile_file
    import dataclasses
    import json

    out = args.out or None
    records = compile_file(
        args.rollout,
        mode=args.mode,
        min_tool_calls=args.min_tool_calls,
        out=out,
    )
    if out is None:
        for r in records:
            print(json.dumps(dataclasses.asdict(r), ensure_ascii=False))
    else:
        print(f"[ok] wrote {len(records)} {args.mode} records to {out}")
    return 0


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------

def _cmd_router(args: argparse.Namespace) -> int:
    from evomerge.io import load_router_records
    from evomerge.router.classifier import RouterConfig, RouterRuleClassifier

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1

    records = load_router_records(args.input)
    cfg = RouterConfig(
        max_repair_rounds=args.max_repair_rounds,
        max_violations=args.max_violations,
        min_tool_validity=args.min_tool_validity,
        max_latency_ms=args.max_latency_ms,
        hard_constraint_limit=args.hard_constraint_limit,
    )
    clf = RouterRuleClassifier(config=cfg)

    results = []
    for rec in records:
        label, reason = clf.predict_with_reason(rec.features)
        results.append({
            "task_id": rec.task_id,
            "predicted_label": label.value,
            "stored_label": rec.label.value,
            "reason": reason,
            "correct": label.value == rec.label.value,
        })

    n_correct = sum(1 for r in results if r["correct"])
    summary = {
        "n": len(results),
        "n_correct": n_correct,
        "accuracy": round(n_correct / len(results), 4) if results else 0.0,
        "predictions": results,
    }

    if args.out:
        Path(args.out).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        print(f"[ok] wrote {len(results)} predictions → {args.out}")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------

def _cmd_synthesize(args: argparse.Namespace) -> int:
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError:
        print(
            "[error] anthropic package required: pip install anthropic",
            file=sys.stderr,
        )
        return 1

    from evomerge.synthesize.generator import GenerationConfig, SyntheticGenerator
    from evomerge.synthesize.templates import builtin_templates, make_task_spec, TaskType
    from evomerge.io import write_jsonl

    client = anthropic.Anthropic()
    model = args.model

    def chat_fn(messages: list[dict]) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=args.max_tokens,
            messages=messages,
        )
        return resp.content[0].text

    cfg = GenerationConfig(
        teacher_model=model,
        n_per_template=args.n_per_template,
        n_bad_per_template=args.n_bad_per_template,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    gen = SyntheticGenerator(chat_fn=chat_fn, config=cfg)

    if args.task_type:
        try:
            tt = TaskType(args.task_type)
        except ValueError:
            print(f"[error] unknown task type: {args.task_type!r}", file=sys.stderr)
            return 1
        templates = {args.task_type: make_task_spec(tt, intent=args.intent or f"Custom {args.task_type} task")}
    else:
        templates = builtin_templates()

    sft, dpo = gen.generate(templates)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if sft:
        write_jsonl(sft, out / "sft.jsonl")
    if dpo:
        write_jsonl(dpo, out / "dpo.jsonl")

    summary = {
        "n_sft": len(sft),
        "n_dpo": len(dpo),
        "out_dir": str(out),
    }
    print(json.dumps(summary, indent=2))
    return 0


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def _cmd_validate(args: argparse.Namespace) -> int:
    from evomerge.validate.contamination import check_contamination
    from evomerge.validate.schema_check import validate_training_record
    import json

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1

    records_raw = []
    with open(args.input) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records_raw.append((lineno, json.loads(line)))
            except json.JSONDecodeError as exc:
                print(f"[error] {args.input}:{lineno}: {exc}", file=sys.stderr)
                return 1

    # schema check — try to parse as SFT/DPO/PPO based on schema_version
    from evomerge.schemas.training import SftTrainingRecord, DpoTrainingRecord, PpoTrainingRecord
    _schema_map = {"sft/v1": SftTrainingRecord, "dpo/v1": DpoTrainingRecord, "ppo/v1": PpoTrainingRecord}

    n_invalid = 0
    errors = []
    parsed = []
    for lineno, d in records_raw:
        sv = d.get("schema_version", "")
        model_cls = _schema_map.get(sv)
        if model_cls is None:
            # router records have no schema_version — skip schema check
            continue
        try:
            rec = model_cls.model_validate(d)
            result = validate_training_record(rec)
            if not result.ok:
                n_invalid += 1
                errors.append({"line": lineno, "errors": result.errors})
            else:
                parsed.append(rec)
        except Exception as exc:
            n_invalid += 1
            errors.append({"line": lineno, "errors": [str(exc)]})

    # contamination check
    n_contaminated = 0
    if args.eval_items and parsed:
        eval_texts = []
        with open(args.eval_items) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    d = json.loads(line)
                    eval_texts.append(d.get("text", d.get("task", "")))
        outputs = [r.messages[-1].content if hasattr(r, "messages") else "" for r in parsed]
        report = check_contamination(outputs, eval_texts, threshold=args.contamination_threshold)
        n_contaminated = report.n_flagged

    summary = {
        "n_records": len(records_raw),
        "n_invalid": n_invalid,
        "n_contaminated": n_contaminated,
        "errors": errors[:20],  # cap output
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if (n_invalid > 0 and args.strict) else 0


# ---------------------------------------------------------------------------
# argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m evomerge",
        description="WasmAgent trace-to-training pipeline CLI",
    )
    sub = p.add_subparsers(dest="command", metavar="command")

    # --- export ---
    ep = sub.add_parser("export", help="convert traces to training JSONL")
    ep.add_argument("--rollout", metavar="FILE", help="rollout-wire/v1 JSONL")
    ep.add_argument("--compliance", metavar="FILE", help="ComplianceEvalRecord JSONL")
    ep.add_argument("--out-dir", default="data/training", metavar="DIR")
    ep.add_argument("--eval-items", metavar="FILE", help="eval items JSONL for G3 contamination")
    ep.add_argument("--contamination-threshold", type=float, default=0.2, metavar="F")
    ep.add_argument("--include-failing", action="store_true",
                    help="include objective_score=0 branches in SFT output")

    # --- router ---
    rp = sub.add_parser("router", help="predict routing labels for a record batch")
    rp.add_argument("--input", metavar="FILE", required=True, help="router.jsonl")
    rp.add_argument("--out", metavar="FILE", help="write predictions JSON (default: stdout)")
    rp.add_argument("--max-repair-rounds", type=int, default=3)
    rp.add_argument("--max-violations", type=int, default=3)
    rp.add_argument("--min-tool-validity", type=float, default=0.8)
    rp.add_argument("--max-latency-ms", type=float, default=30000.0)
    rp.add_argument("--hard-constraint-limit", type=int, default=10)

    # --- synthesize ---
    sp = sub.add_parser("synthesize", help="generate synthetic SFT/DPO via teacher model")
    sp.add_argument("--out-dir", default="data/synthetic", metavar="DIR")
    sp.add_argument("--model", default="claude-opus-4-8", metavar="MODEL",
                    help="teacher model ID (requires ANTHROPIC_API_KEY)")
    sp.add_argument("--task-type", metavar="TYPE",
                    help="markdown_report | tool_call | repair (default: all builtins)")
    sp.add_argument("--intent", metavar="STR", help="task intent (used with --task-type)")
    sp.add_argument("--n-per-template", type=int, default=5)
    sp.add_argument("--n-bad-per-template", type=int, default=5)
    sp.add_argument("--max-tokens", type=int, default=2048)
    sp.add_argument("--seed", type=int, default=42)

    # --- validate ---
    vp = sub.add_parser("validate", help="schema + contamination check on training JSONL")
    vp.add_argument("--input", metavar="FILE", required=True)
    vp.add_argument("--eval-items", metavar="FILE")
    vp.add_argument("--contamination-threshold", type=float, default=0.2)
    vp.add_argument("--strict", action="store_true",
                    help="exit 1 if any invalid records found")

    # --- adp-export ---
    adp = sub.add_parser("adp-export", help="convert rollout-wire/v1 to ADP JSONL")
    adp.add_argument("--rollout", metavar="FILE", required=True,
                     help="rollout-wire/v1 JSONL input")
    adp.add_argument("--out", metavar="FILE",
                     help="output JSONL path (default: stdout)")

    # --- rl-export ---
    rl = sub.add_parser("rl-export", help="convert rollout-wire/v1 to RL transition JSONL")
    rl.add_argument("--rollout", metavar="FILE", required=True,
                    help="rollout-wire/v1 JSONL input")
    rl.add_argument("--reward", metavar="DIMS", default="build,policy,cost",
                    help="comma-separated reward dims: build,visual,policy,cost")
    rl.add_argument("--out", metavar="FILE",
                    help="output JSONL path (default: stdout)")

    # --- compile-context ---
    cc = sub.add_parser("compile-context",
                        help="compile rollout traces to long-context QA or router/critic records")
    cc.add_argument("--rollout", metavar="FILE", required=True,
                    help="rollout-wire/v1 JSONL input")
    cc.add_argument("--mode", choices=["long_context_qa", "router_critic"],
                    default="long_context_qa",
                    help="output format (default: long_context_qa)")
    cc.add_argument("--min-tool-calls", type=int, default=1, metavar="N",
                    help="skip traces with fewer than N tool calls (default: 1)")
    cc.add_argument("--out", metavar="FILE",
                    help="output JSONL path (default: stdout)")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    dispatch = {
        "export": _cmd_export,
        "adp-export": _cmd_adp_export,
        "rl-export": _cmd_rl_export,
        "compile-context": _cmd_compile_context,
        "router": _cmd_router,
        "synthesize": _cmd_synthesize,
        "validate": _cmd_validate,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
