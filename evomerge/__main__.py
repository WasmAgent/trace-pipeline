"""evomerge CLI — python -m evomerge <command> [options]

Commands:
  export                Convert rollout / compliance traces to training JSONL
  adp-export            Convert rollout-wire/v1 to ADP (Agent Data Protocol) JSONL
  rl-export             Convert rollout-wire/v1 to RL transition records JSONL
  compile-context       Convert rollout traces to long-context QA or router/critic records
  router                Predict routing labels for a batch of router records
  synthesize            Generate synthetic SFT/DPO samples via a teacher model
  validate              Run contamination and schema checks on training JSONL
  validate-aep          Validate AEP (Agent Evidence Protocol) records
  lint-benchmark        Check a benchmark task dir for anti-reward-hacking exploit surfaces
  receipt               Produce a run provenance receipt (RunReceipt JSON)
  import-bfcl           Convert BFCL v4 results JSONL to rollout-wire/v1 JSONL
  import-mcp-atlas      Convert MCP-Atlas results JSONL to rollout-wire/v1 or AEP JSONL
  import-oai-agents     Convert OpenAI Agents SDK trace JSONL to AEP JSONL
  import-langsmith      Convert LangSmith/LangGraph trace JSONL to AEP JSONL
  import-terminal-bench Convert Terminal-Bench results JSONL to rollout-wire/v1 or AEP JSONL
  import-tau-bench      Convert τ-bench results JSONL to rollout-wire/v1 or AEP JSONL
  import-tool-sandbox   Convert ToolSandbox results JSONL to rollout-wire/v1 or AEP JSONL
  import-agent-harm     Convert AgentHarm/OS-Harm/CUAHarm results JSONL to rollout-wire/v1 or AEP JSONL
  import-otel           Convert OTel spans JSONL to AEP JSONL
  import-ms-agent-framework Convert Microsoft Agent Framework 1.0 workflow runs to AEP JSONL
  import-adk            Convert Google ADK trace JSONL to AEP JSONL
  import-a2a-task       Convert A2A (Agent-to-Agent) task trace JSONL to AEP JSONL
  audit-report          Generate a combined AEP/lint/provenance audit report (Markdown)
  trust-score           Compute composite AgentTrustScore for an agent run
  registry-register     Register an artifact in the Agent Evidence Registry
  registry-list         List entries in the Agent Evidence Registry

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

    # Confusion matrix + failure buckets via RouterEvalReport
    eval_report = clf.evaluate(
        [rec.features for rec in records],
        [rec.label for rec in records],
    )
    n_correct = sum(1 for r in results if r["correct"])
    summary = {
        "n": len(results),
        "n_correct": n_correct,
        "accuracy": round(n_correct / len(results), 4) if results else 0.0,
        "confusion_matrix": {
            "labels": eval_report.labels,
            "matrix": eval_report.confusion_matrix,
        },
        "failure_buckets": eval_report.failure_buckets,
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
# validate-aep
# ---------------------------------------------------------------------------

def _cmd_validate_aep(args: argparse.Namespace) -> int:
    from evomerge.validate.aep import validate_aep_file, print_aep_report
    from pathlib import Path

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1

    path = Path(args.input)
    if not path.exists():
        print(f"[error] file not found: {path}", file=sys.stderr)
        return 1

    results = validate_aep_file(path)
    print_aep_report(results)

    if not results:
        return 0

    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / len(results)
    return 0 if pass_rate >= args.fail_under else 1


# ---------------------------------------------------------------------------
# lint-benchmark
# ---------------------------------------------------------------------------

def _cmd_lint_benchmark(args: argparse.Namespace) -> int:
    from evomerge.security.benchmark_linter import lint_benchmark_dir, print_lint_report

    task_dir = Path(args.task_dir)
    result = lint_benchmark_dir(task_dir)
    print_lint_report(result)
    return 0 if result.score >= args.fail_under else 1


# ---------------------------------------------------------------------------
# import-bfcl
# ---------------------------------------------------------------------------

def _cmd_import_bfcl(args: argparse.Namespace) -> int:
    """Convert BFCL v4 results JSONL to rollout-wire/v1 JSONL."""
    from evomerge.benchmarks.bfcl import BFCLAdapter

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1
    if not args.output:
        print("[error] --output is required", file=sys.stderr)
        return 1

    adapter = BFCLAdapter()
    pairs = adapter.load_jsonl(args.input)
    rollouts = adapter.to_rollouts(pairs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in rollouts:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(rollouts)} rollout-wire/v1 records to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# import-mcp-atlas
# ---------------------------------------------------------------------------

def _cmd_import_mcp_atlas(args: argparse.Namespace) -> int:
    """Convert MCP-Atlas results JSONL to rollout-wire/v1 or AEP JSONL."""
    from evomerge.benchmarks.mcp_atlas import MCPAtlasAdapter

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1
    if not args.output:
        print("[error] --output is required", file=sys.stderr)
        return 1

    adapter = MCPAtlasAdapter()
    pairs = adapter.load_jsonl(args.input)

    fmt = args.format or "rollout"
    if fmt == "aep":
        records = adapter.to_aep(pairs)
    else:
        records = adapter.to_rollouts(pairs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(records)} {fmt} records to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------

def _cmd_receipt(args: argparse.Namespace) -> int:
    from evomerge.provenance import RunReceiptBuilder

    if not args.run_id:
        print("[error] --run-id is required", file=sys.stderr)
        return 1

    builder = RunReceiptBuilder(run_id=args.run_id, operator=args.operator)

    for path in args.input or []:
        builder.add_input(path)
    for path in args.output or []:
        builder.add_output(path)
    for model in args.model or []:
        builder.add_model(model)

    receipt = builder.build()

    if args.save:
        receipt.save(Path(args.save))
        print(f"[ok] receipt saved to {args.save}")
    else:
        print(json.dumps(receipt.to_dict(), indent=2, ensure_ascii=False))

    return 0


# ---------------------------------------------------------------------------
# import-oai-agents
# ---------------------------------------------------------------------------

def _cmd_import_oai_agents(args: argparse.Namespace) -> int:
    """Convert OpenAI Agents SDK trace JSONL to AEP JSONL."""
    from evomerge.benchmarks.openai_agents_trace import load_oai_trace_jsonl, oai_trace_to_aep

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1
    if not args.output:
        print("[error] --output is required", file=sys.stderr)
        return 1

    traces = load_oai_trace_jsonl(args.input)
    records = [oai_trace_to_aep(t) for t in traces]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(records)} AEP records to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# import-langsmith
# ---------------------------------------------------------------------------

def _cmd_import_langsmith(args: argparse.Namespace) -> int:
    """Convert LangSmith/LangGraph trace JSONL to AEP JSONL."""
    from evomerge.benchmarks.langsmith_trace import load_ls_trace_jsonl, ls_trace_to_aep

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1
    if not args.output:
        print("[error] --output is required", file=sys.stderr)
        return 1

    traces = load_ls_trace_jsonl(args.input)
    records = [ls_trace_to_aep(t) for t in traces]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(records)} AEP records to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# import-terminal-bench
# ---------------------------------------------------------------------------

def _cmd_import_terminal_bench(args: argparse.Namespace) -> int:
    """Convert Terminal-Bench results JSONL to rollout-wire/v1 or AEP JSONL."""
    from evomerge.benchmarks.terminal_bench import TerminalBenchAdapter

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1
    if not args.output:
        print("[error] --output is required", file=sys.stderr)
        return 1

    adapter = TerminalBenchAdapter()
    pairs = adapter.load_jsonl(args.input)

    fmt = args.format or "rollout"
    if fmt == "aep":
        records = adapter.to_aep(pairs)
    else:
        records = adapter.to_rollouts(pairs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(records)} {fmt} records to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# import-tau-bench
# ---------------------------------------------------------------------------

def _cmd_import_tau_bench(args: argparse.Namespace) -> int:
    """Convert τ-bench results JSONL to rollout-wire/v1 or AEP JSONL."""
    from evomerge.benchmarks.tau_bench import TauBenchAdapter

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1
    if not args.output:
        print("[error] --output is required", file=sys.stderr)
        return 1

    adapter = TauBenchAdapter()
    pairs = adapter.load_jsonl(args.input)

    fmt = args.format or "rollout"
    if fmt == "aep":
        records = adapter.to_aep(pairs)
    else:
        records = adapter.to_rollouts(pairs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(records)} {fmt} records to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# import-tool-sandbox
# ---------------------------------------------------------------------------

def _cmd_import_tool_sandbox(args: argparse.Namespace) -> int:
    """Convert ToolSandbox results JSONL to rollout-wire/v1 or AEP JSONL."""
    from evomerge.benchmarks.tool_sandbox import ToolSandboxAdapter

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1
    if not args.output:
        print("[error] --output is required", file=sys.stderr)
        return 1

    adapter = ToolSandboxAdapter()
    pairs = adapter.load_jsonl(args.input)

    fmt = args.format or "rollout"
    if fmt == "aep":
        records = adapter.to_aep(pairs)
    else:
        records = adapter.to_rollouts(pairs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(records)} {fmt} records to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# import-agent-harm
# ---------------------------------------------------------------------------

def _cmd_import_agent_harm(args: argparse.Namespace) -> int:
    """Convert AgentHarm/OS-Harm/CUAHarm results JSONL to rollout-wire/v1 or AEP JSONL."""
    from evomerge.benchmarks.agent_harm import AgentHarmAdapter

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1
    if not args.output:
        print("[error] --output is required", file=sys.stderr)
        return 1

    adapter = AgentHarmAdapter()
    pairs = adapter.load_jsonl(args.input)

    fmt = args.format or "rollout"
    if fmt == "aep":
        records = adapter.to_aep(pairs)
    else:
        records = adapter.to_rollouts(pairs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(records)} {fmt} records to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# import-otel
# ---------------------------------------------------------------------------

def _cmd_import_otel(args: argparse.Namespace) -> int:
    """Convert OTel spans JSONL (grouped by trace_id) to AEP JSONL."""
    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1
    if not args.output:
        print("[error] --output is required", file=sys.stderr)
        return 1

    # Load spans and group by trace_id
    from collections import defaultdict
    spans_by_trace: dict[str, list[dict]] = defaultdict(list)
    with open(args.input) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            span = json.loads(line)
            trace_id = span.get("trace_id") or span.get("traceId") or "unknown"
            spans_by_trace[trace_id].append(span)

    def _span_name(span: dict) -> str:
        return span.get("name") or span.get("operationName") or ""

    def _span_kind(span: dict) -> str:
        return str(span.get("kind") or span.get("spanKind") or "")

    records = []
    for trace_id, spans in spans_by_trace.items():
        actions = []
        verifier_results = []
        model_id = ""

        for span in spans:
            name = _span_name(span)
            kind = _span_kind(span).lower()
            span_id = span.get("span_id") or span.get("spanId") or ""
            attrs = span.get("attributes") or span.get("tags") or {}

            # Extract model id from attributes if present
            if not model_id:
                model_id = (
                    attrs.get("model")
                    or attrs.get("llm.model")
                    or attrs.get("gen_ai.request.model")
                    or ""
                )

            name_lower = name.lower()
            is_tool = name_lower.startswith("tool") or kind == "client"
            is_verifier = name_lower == "guardrail" or "policy" in name_lower

            if is_verifier:
                triggered = attrs.get("triggered", False)
                passed = not triggered
                verifier_results.append({
                    "verifier_id": f"guardrail/{name}",
                    "passed": passed,
                    "score": 1.0 if passed else 0.0,
                    "claim_ids": [span_id] if span_id else [],
                })
            elif is_tool:
                error = span.get("error") or attrs.get("error") or None
                tool_name = attrs.get("tool_name") or attrs.get("rpc.method") or name
                state_changing = attrs.get("state_changing", error is None)
                if isinstance(state_changing, str):
                    state_changing = state_changing.lower() not in ("false", "0", "no")
                action: dict = {
                    "action_id": span_id or f"span-{len(actions)}",
                    "tool_name": tool_name,
                    "state_changing": bool(state_changing),
                    "result_digest": None,
                    "evidence_refs": [],
                    "timestamp_ms": 0.0,
                }
                actions.append(action)

        records.append({
            "schema_version": "aep/v0.1",
            "run_id": f"otel/{trace_id}",
            "model_id": model_id,
            "model_provider": "otel",
            "input_refs": [{"uri": f"otel/trace/{trace_id}"}],
            "output_refs": [],
            "capability_decisions": [],
            "actions": actions,
            "verifier_results": verifier_results,
            "created_at_ms": 0,
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(records)} AEP records to {out_path}")
    return 0



# ---------------------------------------------------------------------------
# import-ms-agent-framework
# ---------------------------------------------------------------------------

def _cmd_import_ms_agent_framework(args: argparse.Namespace) -> int:
    """Convert Microsoft Agent Framework 1.0 workflow runs to AEP JSONL."""
    from evomerge.benchmarks.ms_agent_framework_trace import load_ms_workflow_jsonl, ms_workflow_to_aep

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1
    if not args.output:
        print("[error] --output is required", file=sys.stderr)
        return 1

    runs = load_ms_workflow_jsonl(args.input)
    records = [ms_workflow_to_aep(r) for r in runs]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(records)} AEP records to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# import-adk
# ---------------------------------------------------------------------------

def _cmd_import_adk(args: argparse.Namespace) -> int:
    """Convert Google ADK trace JSONL to AEP JSONL."""
    from evomerge.benchmarks.google_adk_trace import load_adk_trace_jsonl, adk_trace_to_aep

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1
    if not args.output:
        print("[error] --output is required", file=sys.stderr)
        return 1

    traces = load_adk_trace_jsonl(args.input)
    records = [adk_trace_to_aep(t) for t in traces]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(records)} AEP records to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# import-a2a-task
# ---------------------------------------------------------------------------

def _cmd_import_a2a_task(args: argparse.Namespace) -> int:
    """Convert A2A (Agent-to-Agent) task trace JSONL to AEP JSONL."""
    from evomerge.benchmarks.a2a_task_trace import load_a2a_task_jsonl, a2a_task_to_aep

    if not args.input:
        print("[error] --input is required", file=sys.stderr)
        return 1
    if not args.output:
        print("[error] --output is required", file=sys.stderr)
        return 1

    tasks = load_a2a_task_jsonl(args.input)
    records = [a2a_task_to_aep(t) for t in tasks]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(records)} AEP records to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# audit-report
# ---------------------------------------------------------------------------

def _cmd_audit_report(args: argparse.Namespace) -> int:
    """Generate a combined AEP/lint/provenance audit report (Markdown)."""
    from evomerge.audit_report import AuditReportConfig, generate_audit_report
    from evomerge.validate.aep import validate_aep_file

    config = AuditReportConfig(
        title=args.title,
        aep_files=args.aep or [],
        task_dirs=args.task_dirs or [],
        receipt_paths=args.receipts or [],
    )
    report = generate_audit_report(config)

    if args.output and args.output != "-":
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"[ok] wrote audit report to {out_path}")
    else:
        print(report)

    # --fail-under: exit 1 if overall AEP pass rate is below threshold
    if args.fail_under > 0.0 and args.aep:
        total_pass = total = 0
        for fpath in args.aep:
            p = Path(fpath)
            if p.exists():
                results = validate_aep_file(p)
                total_pass += sum(1 for r in results if r.passed)
                total += len(results)
        pass_rate = (total_pass / total) if total else 1.0
        if pass_rate < args.fail_under:
            print(
                f"[fail] AEP pass rate {pass_rate:.1%} < --fail-under {args.fail_under:.1%}",
                flush=True,
            )
            return 1
    return 0


# ---------------------------------------------------------------------------
# trust-score
# ---------------------------------------------------------------------------

def _cmd_trust_score(args: argparse.Namespace) -> int:
    """Compute composite AgentTrustScore for an agent run.

    AEP JSONL processing: all non-comment records are grouped by trace_id /
    run_id.  Each group is scored independently so that a single malicious or
    empty record cannot silently mask problems in a multi-record file.
    """
    from evomerge.trust_score import AgentTrustScoreBuilder

    # ------------------------------------------------------------------
    # Collect AEP records grouped by run_id / trace_id
    # ------------------------------------------------------------------
    groups: dict[str, list[dict]] = {}  # group_key → list of records

    if args.aep:
        aep_path = Path(args.aep)
        if not aep_path.exists():
            print(f"[error] AEP file not found: {aep_path}", file=sys.stderr)
            return 1
        with open(aep_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[warn] skipping malformed JSON line: {exc}", file=sys.stderr)
                    continue
                # Group by trace_id, then run_id, then fall back to a single bucket
                key = (
                    record.get("trace_id")
                    or record.get("run_id")
                    or "__ungrouped__"
                )
                groups.setdefault(key, []).append(record)

    # If no AEP file provided, create one empty group to still emit a score
    if not groups:
        groups["__no_aep__"] = []

    # ------------------------------------------------------------------
    # Score each group
    # ------------------------------------------------------------------
    results: list[dict] = []
    for group_key, records in groups.items():
        builder = AgentTrustScoreBuilder()

        for record in records:
            builder.add_aep_record(record)

        if args.task_passed:
            builder.add_task_success(True)

        if args.benchmark_trust is not None:
            builder.add_benchmark_trust(args.benchmark_trust)

        if args.receipt:
            receipt_path = Path(args.receipt)
            if not receipt_path.exists():
                print(f"[error] receipt file not found: {receipt_path}", file=sys.stderr)
                return 1
            builder.add_receipt_path(receipt_path)
        else:
            builder.add_receipt(has_receipt=False)

        score = builder.build()
        entry = score.to_dict()
        entry["group_key"] = group_key
        results.append(entry)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_payload = results[0] if len(results) == 1 else {"groups": results}

    if args.output and args.output != "-":
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False))
        print(f"[ok] trust score written to {out_path}")
    else:
        print(json.dumps(output_payload, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# registry-register
# ---------------------------------------------------------------------------

def _cmd_registry_register(args: argparse.Namespace) -> int:
    """Register an artifact in the Agent Evidence Registry."""
    from evomerge.registry import Registry, RegistryEntry

    reg_dir = Path(args.registry_dir)
    meta: dict = {}
    for kv in args.meta or []:
        if "=" not in kv:
            print(f"[error] --meta must be KEY=VALUE, got: {kv!r}", file=sys.stderr)
            return 1
        k, v = kv.split("=", 1)
        meta[k] = v

    entry = RegistryEntry(
        id=args.id,
        entry_type=args.type,
        version=args.version,
        artifact_path=args.artifact or "",
        metadata=meta,
    )

    try:
        reg = Registry(reg_dir)
        reg.register(entry)
        reg.save()
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(json.dumps(entry.to_dict(), indent=2, ensure_ascii=False))
    print(f"[ok] registered {entry.id!r} in {reg_dir / Registry.INDEX_FILE}")
    return 0


# ---------------------------------------------------------------------------
# registry-list
# ---------------------------------------------------------------------------

def _cmd_registry_list(args: argparse.Namespace) -> int:
    """List entries in the Agent Evidence Registry."""
    from evomerge.registry import Registry

    reg_dir = Path(args.registry_dir)
    index_path = reg_dir / Registry.INDEX_FILE
    if not index_path.exists():
        print(f"[error] registry index not found: {index_path}", file=sys.stderr)
        return 1

    reg = Registry(reg_dir)
    entries = reg.list_by_type(args.type) if args.type else reg.all()

    result = [e.to_dict() for e in entries]
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[ok] {len(result)} entr{'y' if len(result) == 1 else 'ies'} in {reg_dir}", file=sys.stderr)
    return 0


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

    # --- validate-aep ---
    aep = sub.add_parser("validate-aep", help="validate AEP (Agent Evidence Protocol) records")
    aep.add_argument("--input", metavar="FILE", required=True,
                     help="AEP records JSONL file")
    aep.add_argument("--fail-under", type=float, default=1.0, metavar="F",
                     help="minimum pass rate (0.0–1.0) required for exit 0 (default: 1.0)")

    # --- lint-benchmark ---
    lb = sub.add_parser("lint-benchmark",
                        help="check a benchmark task dir for anti-reward-hacking exploit surfaces")
    lb.add_argument("--task-dir", metavar="PATH", required=True,
                    help="path to the benchmark task directory to lint")
    lb.add_argument("--fail-under", type=float, default=0.6, metavar="F",
                    help="minimum trust score (0.0–1.0) required for exit 0 (default: 0.6)")

    # --- import-bfcl ---
    bfcl_p = sub.add_parser("import-bfcl",
                            help="convert BFCL v4 results JSONL to rollout-wire/v1 JSONL")
    bfcl_p.add_argument("--input", metavar="FILE", required=True,
                        help="BFCL results JSONL (each line: task+result fields merged)")
    bfcl_p.add_argument("--output", metavar="FILE", required=True,
                        help="output rollout-wire/v1 JSONL path")

    # --- import-mcp-atlas ---
    mcp_p = sub.add_parser("import-mcp-atlas",
                           help="convert MCP-Atlas results JSONL to rollout-wire/v1 or AEP JSONL")
    mcp_p.add_argument("--input", metavar="FILE", required=True,
                       help="MCP-Atlas results JSONL (each line: task+result fields merged)")
    mcp_p.add_argument("--output", metavar="FILE", required=True,
                       help="output JSONL path")
    mcp_p.add_argument("--format", choices=["rollout", "aep"], default="rollout",
                       help="output format: rollout-wire/v1 or AEP (default: rollout)")

    # --- receipt ---
    rcp = sub.add_parser("receipt", help="produce a run provenance receipt (RunReceipt JSON)")
    rcp.add_argument("--run-id", metavar="STRING", required=True,
                     help="unique identifier for this pipeline run")
    rcp.add_argument("--input", metavar="FILE", action="append",
                     help="input file to record (repeatable)")
    rcp.add_argument("--output", metavar="FILE", action="append",
                     help="output file to record (repeatable)")
    rcp.add_argument("--model", metavar="STRING", action="append",
                     help="model ID used in the run (repeatable)")
    rcp.add_argument("--operator", metavar="STRING", default="ci",
                     help="operator identifier (default: ci)")
    rcp.add_argument("--save", metavar="PATH",
                     help="save receipt to this path instead of printing to stdout")

    # --- import-oai-agents ---
    oai_p = sub.add_parser("import-oai-agents",
                           help="convert OpenAI Agents SDK trace JSONL to AEP JSONL")
    oai_p.add_argument("--input", metavar="FILE", required=True,
                       help="OAI Agents spans JSONL (each line: one span)")
    oai_p.add_argument("--output", metavar="FILE", required=True,
                       help="output AEP JSONL path")

    # --- import-langsmith ---
    ls_p = sub.add_parser("import-langsmith",
                          help="convert LangSmith/LangGraph trace JSONL to AEP JSONL")
    ls_p.add_argument("--input", metavar="FILE", required=True,
                      help="LangSmith runs JSONL (each line: one Run object)")
    ls_p.add_argument("--output", metavar="FILE", required=True,
                      help="output AEP JSONL path")

    # --- import-ms-agent-framework ---
    ms_p = sub.add_parser("import-ms-agent-framework",
                          help="convert Microsoft Agent Framework 1.0 workflow runs to AEP JSONL")
    ms_p.add_argument("--input", metavar="FILE", required=True,
                      help="MS Agent Framework workflow runs JSONL (each line: one WorkflowRun)")
    ms_p.add_argument("--output", metavar="FILE", required=True,
                      help="output AEP JSONL path")

    # --- import-adk ---
    adk_p = sub.add_parser("import-adk",
                           help="convert Google ADK trace JSONL to AEP JSONL")
    adk_p.add_argument("--input", metavar="FILE", required=True,
                       help="Google ADK events JSONL (each line: one ADKEvent)")
    adk_p.add_argument("--output", metavar="FILE", required=True,
                       help="output AEP JSONL path")

    # --- import-a2a-task ---
    a2a_p = sub.add_parser("import-a2a-task",
                           help="convert A2A (Agent-to-Agent) task trace JSONL to AEP JSONL")
    a2a_p.add_argument("--input", metavar="FILE", required=True,
                       help="A2A task JSONL (each line: one A2ATask)")
    a2a_p.add_argument("--output", metavar="FILE", required=True,
                       help="output AEP JSONL path")


    # --- audit-report ---
    ar_p = sub.add_parser("audit-report",
                          help="generate a combined AEP/lint/provenance audit report (Markdown)")
    ar_p.add_argument("--title", metavar="STRING",
                      default="WasmAgent Benchmark Audit",
                      help="report title (default: 'WasmAgent Benchmark Audit')")
    ar_p.add_argument("--aep", metavar="FILE", action="append", dest="aep",
                      help="AEP records JSONL file (repeatable)")
    ar_p.add_argument("--task-dir", metavar="DIR", action="append", dest="task_dirs",
                      help="benchmark task directory to lint (repeatable)")
    ar_p.add_argument("--receipt", metavar="FILE", action="append", dest="receipts",
                      help="run receipt JSON path (repeatable)")
    ar_p.add_argument("--output", metavar="PATH", default="-",
                      help="output Markdown path (default: stdout)")
    ar_p.add_argument("--fail-under", type=float, default=0.0, metavar="F",
                      help="exit 1 if AEP pass rate < F (0.0 = never fail, default)")

    # --- trust-score ---
    ts_p = sub.add_parser("trust-score",
                          help="compute composite AgentTrustScore for an agent run")
    ts_p.add_argument("--aep", metavar="FILE",
                      help="AEP records JSONL file (uses first record)")
    ts_p.add_argument("--task-passed", action="store_true",
                      help="mark task as successfully completed")
    ts_p.add_argument("--benchmark-trust", type=float, metavar="FLOAT",
                      help="benchmark environment trust score (0.0–1.0, from lint-benchmark)")
    ts_p.add_argument("--receipt", metavar="FILE",
                      help="run receipt JSON path (presence signals supply chain integrity)")
    ts_p.add_argument("--output", metavar="PATH", default="-",
                      help="output JSON path (default: stdout)")

    # --- import-terminal-bench ---
    tb_p = sub.add_parser("import-terminal-bench",
                          help="convert Terminal-Bench results JSONL to rollout-wire/v1 or AEP JSONL")
    tb_p.add_argument("--input", metavar="FILE", required=True,
                      help="Terminal-Bench results JSONL (each line: task+result fields merged)")
    tb_p.add_argument("--output", metavar="FILE", required=True,
                      help="output JSONL path")
    tb_p.add_argument("--format", choices=["rollout", "aep"], default="rollout",
                      help="output format: rollout-wire/v1 or AEP (default: rollout)")

    # --- import-tau-bench ---
    tau_p = sub.add_parser("import-tau-bench",
                           help="convert τ-bench results JSONL to rollout-wire/v1 or AEP JSONL")
    tau_p.add_argument("--input", metavar="FILE", required=True,
                       help="τ-bench results JSONL (each line: task+result fields merged)")
    tau_p.add_argument("--output", metavar="FILE", required=True,
                       help="output JSONL path")
    tau_p.add_argument("--format", choices=["rollout", "aep"], default="rollout",
                       help="output format: rollout-wire/v1 or AEP (default: rollout)")

    # --- import-tool-sandbox ---
    ts_p = sub.add_parser("import-tool-sandbox",
                          help="convert ToolSandbox results JSONL to rollout-wire/v1 or AEP JSONL")
    ts_p.add_argument("--input", metavar="FILE", required=True,
                      help="ToolSandbox results JSONL (each line: task+result fields merged)")
    ts_p.add_argument("--output", metavar="FILE", required=True,
                      help="output JSONL path")
    ts_p.add_argument("--format", choices=["rollout", "aep"], default="rollout",
                      help="output format: rollout-wire/v1 or AEP (default: rollout)")

    # --- import-agent-harm ---
    ah_p = sub.add_parser("import-agent-harm",
                          help="convert AgentHarm/OS-Harm/CUAHarm results JSONL to rollout-wire/v1 or AEP JSONL")
    ah_p.add_argument("--input", metavar="FILE", required=True,
                      help="AgentHarm results JSONL (each line: task+result fields merged)")
    ah_p.add_argument("--output", metavar="FILE", required=True,
                      help="output JSONL path")
    ah_p.add_argument("--format", choices=["rollout", "aep"], default="rollout",
                      help="output format: rollout-wire/v1 or AEP (default: rollout)")

    # --- import-otel ---
    otel_p = sub.add_parser("import-otel",
                            help="convert OTel spans JSONL to AEP JSONL")
    otel_p.add_argument("--input", metavar="FILE", required=True,
                        help="OTel spans JSONL (each line: one span dict with trace_id)")
    otel_p.add_argument("--output", metavar="FILE", required=True,
                        help="output AEP JSONL path")

    # --- registry-register ---
    rr_p = sub.add_parser("registry-register",
                          help="register an artifact in the Agent Evidence Registry")
    rr_p.add_argument("--registry-dir", metavar="DIR", default="registry",
                      help="registry root directory (default: registry/)")
    rr_p.add_argument("--id", metavar="STRING", required=True,
                      help="unique registry entry ID")
    rr_p.add_argument("--type", metavar="TYPE", required=True,
                      help="entry type: policy_bundle | verifier | benchmark_task | receipt | "
                           "model_profile | router_profile | dataset_card | aep_schema")
    rr_p.add_argument("--version", metavar="STRING", required=True,
                      help="semver or free-form version string")
    rr_p.add_argument("--artifact", metavar="FILE",
                      help="path to the artifact file (digest computed automatically)")
    rr_p.add_argument("--meta", metavar="KEY=VALUE", action="append",
                      help="metadata key=value pair (repeatable)")

    # --- registry-list ---
    rl2_p = sub.add_parser("registry-list",
                           help="list entries in the Agent Evidence Registry")
    rl2_p.add_argument("--registry-dir", metavar="DIR", default="registry",
                       help="registry root directory (default: registry/)")
    rl2_p.add_argument("--type", metavar="TYPE", default="",
                       help="filter by entry type (default: all)")

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
        "validate-aep": _cmd_validate_aep,
        "lint-benchmark": _cmd_lint_benchmark,
        "receipt": _cmd_receipt,
        "import-bfcl": _cmd_import_bfcl,
        "import-mcp-atlas": _cmd_import_mcp_atlas,
        "import-oai-agents": _cmd_import_oai_agents,
        "import-langsmith": _cmd_import_langsmith,
        "import-terminal-bench": _cmd_import_terminal_bench,
        "import-tau-bench": _cmd_import_tau_bench,
        "import-tool-sandbox": _cmd_import_tool_sandbox,
        "import-agent-harm": _cmd_import_agent_harm,
        "import-otel": _cmd_import_otel,
        "import-ms-agent-framework": _cmd_import_ms_agent_framework,
        "import-adk": _cmd_import_adk,
        "import-a2a-task": _cmd_import_a2a_task,
        "audit-report": _cmd_audit_report,
        "trust-score": _cmd_trust_score,
        "registry-register": _cmd_registry_register,
        "registry-list": _cmd_registry_list,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
