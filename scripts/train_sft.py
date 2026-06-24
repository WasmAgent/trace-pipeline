#!/usr/bin/env python3
"""QLoRA SFT training script for WasmAgent compliance-conditioned small models.

Trains a QLoRA adapter on top of a base model using SFT training records
exported by evomerge-framework.

Hardware notes (Apple Silicon, from CLAUDE.md Phase 17 experience):
  - 1.5B / 1.7B models: use --fp32 (bfloat16 causes NaN on small models)
  - 4B+ models: use --bf16 + default use_cpu=True (22x faster than fp32)
  - Always use_cpu=True to avoid MPS wired-memory degradation
  - Set max_seq_len to ~1.2x actual data max length (dry-run prints this)
  - save_steps=50 to avoid losing progress on crash
  - Early stop when loss < 0.1

Usage:
    # combine real + synthetic SFT data and train
    python scripts/train_sft.py \\
        --train-data data/training/ifeval/compliance_sft.jsonl,data/synthetic/sft.jsonl \\
        --base-model Qwen/Qwen2.5-1.5B-Instruct \\
        --out-dir checkpoints/sft-v1 \\
        --max-steps 200

    # dry-run: print data stats, no training
    python scripts/train_sft.py \\
        --train-data data/training/ifeval/compliance_sft.jsonl \\
        --base-model Qwen/Qwen2.5-1.5B-Instruct \\
        --dry-run

    # 4B model (faster with bf16)
    python scripts/train_sft.py \\
        --train-data data/training/ifeval/compliance_sft.jsonl \\
        --base-model Qwen/Qwen2.5-3B-Instruct \\
        --bf16 --out-dir checkpoints/sft-4b-v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _load_sft_records(paths: list[str]) -> list[dict]:
    """Load and merge multiple SFT JSONL files."""
    records = []
    for path in paths:
        p = Path(path.strip())
        if not p.exists():
            print(f"[warn] not found: {p}", file=sys.stderr)
            continue
        n = 0
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("schema_version") == "sft/v1":
                    records.append(r)
                    n += 1
        print(f"  loaded {n} records ← {p}")
    return records


def _to_conversation(record: dict) -> list[dict]:
    """Convert SftTrainingRecord messages to HuggingFace chat format."""
    msgs = record.get("messages", [])
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


def _estimate_token_lengths(conversations: list[list[dict]], tokenizer) -> dict:
    lengths = []
    for conv in conversations:
        text = " ".join(m["content"] for m in conv)
        toks = tokenizer.encode(text)
        lengths.append(len(toks))
    lengths.sort()
    return {
        "min": lengths[0],
        "max": lengths[-1],
        "mean": sum(lengths) // len(lengths),
        "p95": lengths[int(len(lengths) * 0.95)],
        "recommended_max_seq_len": int(lengths[-1] * 1.2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train-data", required=True, metavar="FILES",
                    help="comma-separated SFT JSONL paths")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="HuggingFace model name or local path")
    ap.add_argument("--out-dir", default="checkpoints/sft-v1", metavar="DIR")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--max-seq-len", type=int, default=0,
                    help="0 = auto-detect from data (recommended)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="effective batch = batch_size * grad_accum")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--fp32", action="store_true",
                    help="force float32 (required for 1.5B to avoid NaN)")
    ap.add_argument("--bf16", action="store_true",
                    help="use bfloat16 (faster for 4B+, use with use_cpu=True)")
    ap.add_argument("--loss-threshold", type=float, default=0.1,
                    help="early stop when training loss falls below this")
    ap.add_argument("--dry-run", action="store_true",
                    help="load data and tokenizer, print stats, no training")
    args = ap.parse_args()

    # ── load training records ────────────────────────────────────────────────
    print("loading training data...")
    paths = [p.strip() for p in args.train_data.split(",") if p.strip()]
    records = _load_sft_records(paths)
    if not records:
        print("[error] no SFT records loaded", file=sys.stderr)
        return 1
    print(f"  total: {len(records)} SFT records")

    output_types = {}
    for r in records:
        t = r.get("output_type", "unknown")
        output_types[t] = output_types.get(t, 0) + 1
    for t, n in sorted(output_types.items()):
        print(f"  output_type={t!r}: {n}")

    conversations = [_to_conversation(r) for r in records]

    # ── load tokenizer + estimate sequence lengths ───────────────────────────
    try:
        from transformers import AutoTokenizer
    except ImportError:
        if args.dry_run:
            print("\n[dry-run] transformers not installed — skipping length estimation")
            print("  install with: pip install -e '.[train]'")
            return 0
        print("[error] pip install transformers", file=sys.stderr)
        return 1

    print(f"\nloading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("estimating sequence lengths...")
    stats = _estimate_token_lengths(conversations, tokenizer)
    for k, v in stats.items():
        print(f"  {k:<30} {v}")

    max_seq_len = args.max_seq_len or stats["recommended_max_seq_len"]
    print(f"\n  using max_seq_len = {max_seq_len}")

    if args.dry_run:
        print("\n[dry-run] no training")
        return 0

    # ── check training deps ──────────────────────────────────────────────────
    try:
        import torch
        from transformers import AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType as PeftTaskType
        from trl import SFTTrainer, SFTConfig
    except ImportError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print("  pip install -e '.[train]'  or  pip install peft trl transformers bitsandbytes",
              file=sys.stderr)
        return 1

    # dtype logic from CLAUDE.md Phase 17 experience:
    # 1.5B: fp32 (bfloat16 → NaN); 4B+: bfloat16 (22x faster, numerically stable)
    if args.bf16 and not args.fp32:
        dtype = torch.bfloat16
        dtype_str = "bfloat16"
    else:
        dtype = torch.float32
        dtype_str = "float32"

    print(f"\nloading model: {args.base_model} [{dtype_str}]")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="cpu",       # avoid MPS wired-memory degradation
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        task_type=PeftTaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # format conversations for SFT
    def format_fn(example):
        conv = example["conversations"]
        text = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    import datasets
    ds = datasets.Dataset.from_list([{"conversations": c} for c in conversations])
    ds = ds.map(format_fn, remove_columns=ds.column_names)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # EarlyStoppingCallback based on loss threshold
    from transformers import TrainerCallback

    class LossThresholdCallback(TrainerCallback):
        def __init__(self, threshold):
            self.threshold = threshold

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and logs.get("loss", float("inf")) < self.threshold:
                print(f"\n[early stop] loss {logs['loss']:.4f} < {self.threshold}")
                control.should_training_stop = True

    training_args = SFTConfig(
        output_dir=str(out),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        # Apple Silicon: use_cpu=True mandatory to avoid MPS degradation
        use_cpu=True,
        bf16=args.bf16 and not args.fp32,
        fp16=False,
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        callbacks=[LossThresholdCallback(args.loss_threshold)],
    )

    print(f"\ntraining...")
    print(f"  max_steps      : {args.max_steps}")
    print(f"  effective_batch: {args.batch_size * args.grad_accum}")
    print(f"  loss_threshold : {args.loss_threshold}")
    print(f"  out_dir        : {out}")

    trainer.train()
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))

    # save training summary
    summary = {
        "base_model": args.base_model,
        "n_train_records": len(records),
        "output_types": output_types,
        "max_seq_len": max_seq_len,
        "lora_r": args.lora_r,
        "dtype": dtype_str,
        "max_steps": args.max_steps,
        "lr": args.lr,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✓ training complete → {out}/final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
