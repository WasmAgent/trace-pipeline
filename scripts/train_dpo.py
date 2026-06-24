#!/usr/bin/env python3
"""DPO/ORPO fine-tuning on SFT checkpoint using TRL DPOTrainer.

Trains on evomerge DpoTrainingRecord JSONL (repair-trace + cross-mode pairs).

Usage:
    python scripts/train_dpo.py \
        --dpo-data data/training/v2/dpo_merged.jsonl \
        --sft-checkpoint checkpoints/sft-v2 \
        --out-dir checkpoints/dpo-v1

    # Merge all known DPO sources automatically:
    python scripts/train_dpo.py \
        --merge-all \
        --sft-checkpoint checkpoints/sft-v2 \
        --out-dir checkpoints/dpo-v1

Apple Silicon notes:
  - 1.5B: use --fp32 (bfloat16 -> NaN risk on small models)
  - Always use_cpu=True to avoid MPS wired-memory degradation
  - effective_batch = batch_size x grad_accum = 1 x 4 = 4 (DPO needs small batch)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Default DPO data sources relative to REPO_ROOT
_DEFAULT_DPO_SOURCES = [
    "data/training/ifeval/compliance_dpo.jsonl",
    "data/training/ifeval/cross_mode_dpo.jsonl",
    "data/training/v2/dpo_merged.jsonl",
]


def load_dpo_records(paths: list[str]) -> list[dict]:
    """Load and deduplicate DpoTrainingRecord (schema_version=dpo/v1) from JSONL paths.

    Deduplication is performed by SHA-256 hash of (chosen || '||' || rejected)
    so the same pair loaded from multiple files is only counted once.
    Files that do not exist are skipped with a warning.
    """
    seen: set[str] = set()
    records: list[dict] = []
    for path in paths:
        p = Path(path.strip())
        if not p.exists():
            # Try relative to REPO_ROOT
            p_rel = REPO_ROOT / path.strip()
            if p_rel.exists():
                p = p_rel
            else:
                print(f"[warn] not found: {path}", file=sys.stderr)
                continue
        n_new = 0
        n_dup = 0
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("schema_version") != "dpo/v1":
                    continue
                key = hashlib.sha256(
                    (r.get("chosen", "") + "||" + r.get("rejected", "")).encode()
                ).hexdigest()
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                records.append(r)
                n_new += 1
        dup_note = f" ({n_dup} duplicates skipped)" if n_dup else ""
        print(f"  loaded {n_new} DPO records ← {p}{dup_note}")
    return records


def load_all_dpo_records() -> list[dict]:
    """Convenience: merge all known DPO sources and deduplicate."""
    return load_dpo_records([str(REPO_ROOT / s) for s in _DEFAULT_DPO_SOURCES])


def to_hf_dataset(records: list[dict]):
    """Convert DpoTrainingRecord list to HF Dataset format."""
    import datasets

    rows = []
    for r in records:
        prompt_msgs = r.get("prompt_messages") or r.get("messages", [])[:-1]
        prompt = " ".join(m["content"] for m in prompt_msgs if m.get("content"))
        rows.append({
            "prompt":   prompt,
            "chosen":   r["chosen"],
            "rejected": r["rejected"],
        })
    return datasets.Dataset.from_list(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dpo-data", metavar="FILES",
                     help="comma-separated DPO JSONL paths")
    src.add_argument("--merge-all", action="store_true",
                     help="merge all known DPO sources and deduplicate")
    ap.add_argument("--sft-checkpoint", required=True, metavar="DIR",
                    help="SFT checkpoint directory (with adapter_config.json OR config.json)")
    ap.add_argument("--out-dir", default="checkpoints/dpo-v1", metavar="DIR")
    # Epoch-based is the primary training mode; max-steps overrides when > 0
    ap.add_argument("--epochs", type=int, default=3,
                    help="number of training epochs (default 3)")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="override epochs with a fixed step count (-1 = use --epochs)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1,
                    help="DPO temperature (default 0.1)")
    ap.add_argument("--fp32", action="store_true", default=True,
                    help="force float32 (default: True for 1.5B stability)")
    ap.add_argument("--bf16", action="store_true",
                    help="use bfloat16 (faster for 4B+)")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.merge_all:
        records = load_all_dpo_records()
    else:
        paths = [p.strip() for p in args.dpo_data.split(",") if p.strip()]
        records = load_dpo_records(paths)
    if not records:
        print("[error] no DPO records loaded", file=sys.stderr)
        return 1
    print(f"  total: {len(records)} DPO pairs")

    if args.dry_run:
        print("[dry-run] no training")
        return 0

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType as PeftTaskType, PeftModel
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        print(f"[error] {exc}\n  pip install -e '.[train]'", file=sys.stderr)
        return 1

    sft_ckpt = Path(args.sft_checkpoint)

    # Resolve base model — handle both LoRA adapter and full model dirs
    if (sft_ckpt / "adapter_config.json").exists():
        adapter_cfg = json.loads((sft_ckpt / "adapter_config.json").read_text())
        base_model_path = adapter_cfg.get("base_model_name_or_path", str(sft_ckpt))
        print(f"loading base model from {base_model_path} + LoRA from {sft_ckpt}...")
        dtype = torch.float32 if (args.fp32 or not args.bf16) else torch.bfloat16
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path, dtype=dtype, device_map="cpu", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, str(sft_ckpt))
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    else:
        base_model_path = str(sft_ckpt)
        print(f"loading full model from {base_model_path}...")
        dtype = torch.float32 if (args.fp32 or not args.bf16) else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, dtype=dtype, device_map="cpu", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add LoRA for DPO
    lora_cfg = LoraConfig(
        task_type=PeftTaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = to_hf_dataset(records)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    training_args = DPOConfig(
        output_dir=str(out),
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        max_steps=args.max_steps,  # -1 means use epochs
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=512,
        max_prompt_length=256,
        use_cpu=True,
        bf16=args.bf16 and not args.fp32,
        fp16=False,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\nDPO training...")
    print(f"  epochs       : {args.epochs if args.max_steps <= 0 else 'N/A (max_steps)'}")
    print(f"  max_steps    : {args.max_steps}")
    print(f"  beta         : {args.beta}")
    print(f"  n_dpo_pairs  : {len(records)}")
    print(f"  out_dir      : {out}")

    trainer.train()
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))

    summary = {
        "sft_checkpoint": str(sft_ckpt),
        "n_dpo_pairs": len(records),
        "beta": args.beta,
        "num_train_epochs": args.epochs if args.max_steps <= 0 else None,
        "max_steps": args.max_steps if args.max_steps > 0 else None,
        "lr": args.lr,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✓ DPO training complete → {out}/final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
