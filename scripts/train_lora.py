#!/usr/bin/env python3
"""QLoRA fine-tuning script for Qwen2.5-VL-3B-Instruct on T6 spatial instruction data.

Designed for Google Colab (T4/A100). Two modes:
  (A) text-only  — default. Renders structured text prompts from entities/scene_graph/staged_reasoning.
                    Works WITHOUT real KITTI images. Safe for first-run on Colab.
  (B) vision     — requires KITTI images downloaded to image_path. Renders image+text via processor.

Usage (Colab):
  HF_TOKEN=hf_xxx python train_lora.py --mode text-only --epochs 3
  HF_TOKEN=hf_xxx python train_lora.py --mode vision   --epochs 3   # after KITTI download

Local smoke (no GPU, no HF token):
  python train_lora.py --dataset-check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# CONFIG — edit these for Colab runs
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    # Model
    "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "max_length": 1024,           # reduce to 512 on OOM (T4)
    # LoRA / QLoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # Quantisation
    "load_in_4bit": True,
    # Training
    "epochs": 3,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "logging_steps": 5,
    "eval_steps": 50,
    "save_steps": 100,
    # Paths (Colab defaults)
    "train_path": "data/kitti_scene_train.jsonl",
    "val_path": "data/kitti_scene_val.jsonl",
    "output_dir": "outputs/lora_adapter",
    "metrics_path": "outputs/eval_metrics.json",
}

# ---------------------------------------------------------------------------
# Dataset loading (pure-Python JSONL — no torch dependency at import time)
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] skipping line {i}: {exc}")
    return rows


# ---------------------------------------------------------------------------
# Prompt formatting (text-only mode)
# ---------------------------------------------------------------------------

def render_text_only_prompt(sample: dict[str, Any]) -> str:
    """Render a structured-text training prompt from sample metadata."""
    parts: list[str] = []

    parts.append(f"Image path: {sample.get('image_path', 'N/A')}")
    parts.append(f"Image size: {sample.get('image_size', 'N/A')}")

    # Entities
    entities = sample.get("entities", [])
    if entities:
        ent_lines = []
        for e in entities:
            bbox_str = [f"{v:.4f}" for v in e.get("bbox", [])]
            ent_lines.append(f"  - {e['name']} ({e['class']}): bbox=[{', '.join(bbox_str)}]")
        parts.append("Entities:\n" + "\n".join(ent_lines))

    # Scene graph
    sg = sample.get("scene_graph", [])
    if sg:
        sg_lines = [f"  - {t['subject']} --{t['relation']}--> {t['object']}" for t in sg]
        parts.append("Scene graph:\n" + "\n".join(sg_lines))

    # Staged reasoning
    sr = sample.get("staged_reasoning", [])
    if sr:
        sr_lines = [f"  Step {s['step']}: {s['text']}" for s in sr]
        parts.append("Staged reasoning:\n" + "\n".join(sr_lines))

    context = "\n".join(parts)

    user_msg = (
        f"Given the following autonomous driving scene:\n\n{context}\n\n"
        f"Question: {sample['question']}\n"
        f"Answer:"
    )
    assistant_msg = sample["answer"]
    return user_msg, assistant_msg


def build_chat_messages(sample: dict[str, Any], mode: str) -> list[dict[str, str]]:
    """Build chat-format messages list for Qwen2.5-VL chat template."""
    user_content: list[dict[str, Any]] = []

    if mode == "vision":
        # In vision mode, include the image (requires processor on Colab with real images)
        img_path = sample.get("image_path", "")
        if img_path and Path(img_path).exists():
            user_content.append({"type": "image", "image": img_path})
        else:
            # Fallback: render as text-only even in vision mode if image missing
            pass

    # Always add structured text context
    text_ctx_parts: list[str] = []
    entities = sample.get("entities", [])
    if entities:
        ent_lines = [
            f"  - {e['name']} ({e['class']}): bbox={[round(v,4) for v in e.get('bbox',[])]}"
            for e in entities
        ]
        text_ctx_parts.append("Entities:\n" + "\n".join(ent_lines))

    sg = sample.get("scene_graph", [])
    if sg:
        sg_lines = [f"  - {t['subject']} --{t['relation']}--> {t['object']}" for t in sg]
        text_ctx_parts.append("Scene graph:\n" + "\n".join(sg_lines))

    sr = sample.get("staged_reasoning", [])
    if sr:
        sr_lines = [f"  Step {s['step']}: {s['text']}" for s in sr]
        text_ctx_parts.append("Reasoning:\n" + "\n".join(sr_lines))

    ctx_str = "\n".join(text_ctx_parts) if text_ctx_parts else "(no structured context)"
    question = sample["question"]

    prompt_text = (
        f"Autonomous driving scene:\n{ctx_str}\n\n"
        f"Question: {question}\nAnswer:"
    )
    user_content.append({"type": "text", "text": prompt_text})

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": sample["answer"]},
    ]
    return messages


# ---------------------------------------------------------------------------
# Dataset-check mode (no GPU / no HF token needed)
# ---------------------------------------------------------------------------

def dataset_check(train_path: str, val_path: str) -> bool:
    """Load JSONL files and print a sample — verifies T6 data integration."""
    for label, path in [("train", train_path), ("val", val_path)]:
        if not Path(path).exists():
            print(f"[FAIL] {label} file not found: {path}")
            return False
        rows = load_jsonl(path)
        print(f"[OK] {label}: {len(rows)} samples from {path}")
        if rows:
            sample = rows[0]
            print(f"  Sample ID: {sample['id']}")
            print(f"  Question: {sample['question'][:80]}...")
            print(f"  Answer:   {sample['answer'][:80]}...")
            print(f"  Entities: {len(sample.get('entities', []))}")
            print(f"  Scene graph triples: {len(sample.get('scene_graph', []))}")
            print(f"  Staged reasoning steps: {len(sample.get('staged_reasoning', []))}")
            # Render prompt to verify formatting
            user_msg, assistant_msg = render_text_only_prompt(sample)
            print(f"  [Prompt render OK] user_msg length: {len(user_msg)} chars")
            print(f"  [Prompt render OK] assistant_msg length: {len(assistant_msg)} chars")
    print("\n[OK] Dataset integration verified.")
    return True


# ---------------------------------------------------------------------------
# Main training (requires torch + GPU)
# ---------------------------------------------------------------------------

def train(mode: str = "text-only") -> None:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    # --- HF token ---
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[FATAL] HF_TOKEN environment variable not set.")
        print("  Get your token from https://huggingface.co/settings/tokens")
        print("  Then: export HF_TOKEN=hf_xxx  OR  set it in Colab via os.environ['HF_TOKEN'] = 'hf_xxx'")
        sys.exit(1)

    cfg = CONFIG.copy()
    cfg["mode"] = mode

    # --- Device setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    print(f"[INFO] Device: {device}, bf16: {use_bf16}")
    if device == "cpu":
        print("[FATAL] No GPU available. Use Colab with --gpu T4 or A100.")
        sys.exit(1)

    # --- Quantisation config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --- Load model + tokenizer ---
    print(f"[INFO] Loading model: {cfg['model_id']} (4-bit QLoRA)...")
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"],
            trust_remote_code=True,
            token=hf_token,
        )
    except torch.cuda.OutOfMemoryError:
        print("[WARN] OOM on initial load — retrying with reduced max_length...")
        torch.cuda.empty_cache()
        cfg["max_length"] = 512
        cfg["per_device_batch_size"] = 1
        cfg["gradient_accumulation_steps"] = 32
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"],
            trust_remote_code=True,
            token=hf_token,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_load_time = time.time() - t0
    print(f"[INFO] Model loaded in {model_load_time:.1f}s")

    # --- LoRA config ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Load data ---
    train_samples = load_jsonl(cfg["train_path"])
    val_samples = load_jsonl(cfg["val_path"])
    print(f"[INFO] Loaded {len(train_samples)} train, {len(val_samples)} val samples")

    if not train_samples:
        print("[FATAL] No training samples found.")
        sys.exit(1)

    # --- Tokenise ---
    def tokenize_fn(examples: dict) -> dict:
        """Tokenise a batch of text samples."""
        texts = examples["text"]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=cfg["max_length"],
            padding="max_length",
            return_tensors=None,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    def format_and_tokenize(samples: list[dict], split: str) -> Dataset:
        formatted: list[str] = []
        for s in samples:
            if mode == "text-only":
                user_msg, asst_msg = render_text_only_prompt(s)
                chat = f"<|user|>\n{user_msg}\n<|assistant|>\n{asst_msg}"
            else:
                messages = build_chat_messages(s, mode)
                try:
                    chat = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    user_msg, asst_msg = render_text_only_prompt(s)
                    chat = f"<|user|>\n{user_msg}\n<|assistant|>\n{asst_msg}"
            formatted.append(chat)
        ds = Dataset.from_dict({"text": formatted})
        ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        print(f"[INFO] Tokenised {split}: {len(ds)} examples")
        return ds

    train_ds = format_and_tokenize(train_samples, "train")
    val_ds = format_and_tokenize(val_samples, "val")

    # --- Training arguments ---
    batch_total = cfg["per_device_batch_size"] * cfg["gradient_accumulation_steps"]
    total_steps = (len(train_ds) // batch_total) * cfg["epochs"]
    print(f"[INFO] Effective batch size: {batch_total}, total steps ≈ {total_steps}")

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # --- Train ---
    print("[INFO] Starting training...")
    t_train_start = time.time()
    train_result = trainer.train()
    t_train_end = time.time()
    train_duration_h = (t_train_end - t_train_start) / 3600

    print(f"[INFO] Training complete in {train_duration_h:.2f} hours")
    print(f"[INFO] Train loss: {train_result.training_loss:.4f}")

    # --- Evaluate ---
    print("[INFO] Running final evaluation...")
    eval_result = trainer.evaluate()
    val_loss = eval_result.get("eval_loss", float("nan"))
    print(f"[INFO] Val loss: {val_loss:.4f}")

    # Estimate accuracy via text generation on a small val subset
    accuracy = _estimate_accuracy(model, tokenizer, val_samples[:20], mode, cfg)

    # --- Save adapter ---
    adapter_dir = Path(cfg["output_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[INFO] Adapter saved to {adapter_dir}")

    # --- Save metrics ---
    cu_estimate = train_duration_h * (13.5 if use_bf16 else 1.75)  # A100 ~13.5 CU/h, T4 ~1.75 CU/h
    metrics = {
        "model": cfg["model_id"],
        "mode": mode,
        "lora_r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "epochs": cfg["epochs"],
        "train_loss": train_result.training_loss,
        "val_loss": val_loss,
        "accuracy_keyword_overlap": accuracy,
        "train_duration_hours": round(train_duration_h, 3),
        "estimated_cu_used": round(cu_estimate, 2),
        "cu_budget_monthly": 550,
        "device": device,
        "bf16": use_bf16,
        "load_in_4bit": cfg["load_in_4bit"],
        "max_length": cfg["max_length"],
        "adapter_path": str(adapter_dir),
    }
    metrics_path = Path(cfg["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {metrics_path}")

    # --- CU budget check ---
    if cu_estimate > 550:
        print(f"[WARN] Estimated CU ({cu_estimate:.1f}) exceeds 550 monthly budget!")
    else:
        print(f"[OK] CU estimate ({cu_estimate:.1f}) within 550 monthly budget")


def _estimate_accuracy(
    model: Any,
    tokenizer: Any,
    val_samples: list[dict],
    mode: str,
    cfg: dict,
) -> float:
    """Light accuracy metric: keyword overlap between generated and reference answers."""
    import torch

    correct = 0
    total = len(val_samples)
    if total == 0:
        return 0.0

    for s in val_samples:
        if mode == "text-only":
            user_msg, _ = render_text_only_prompt(s)
            prompt = f"<|user|>\n{user_msg}\n<|assistant|>\n"
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": s["question"]}]}]
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                user_msg, _ = render_text_only_prompt(s)
                prompt = f"<|user|>\n{user_msg}\n<|assistant|>\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg["max_length"]).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        reference = s["answer"].strip()

        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        overlap = len(gen_words & ref_words) / max(len(ref_words), 1)
        if overlap > 0.3:
            correct += 1

    accuracy = correct / total
    return round(accuracy, 4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen2.5-VL-3B")
    parser.add_argument("--mode", choices=["text-only", "vision"], default="text-only",
                        help="text-only: no images needed; vision: requires KITTI images")
    parser.add_argument("--dataset-check", action="store_true",
                        help="Verify T6 dataset loading (no GPU/token needed)")
    parser.add_argument("--epochs", type=int, default=None, help="Override CONFIG epochs")
    parser.add_argument("--max-length", type=int, default=None, help="Override CONFIG max_length")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--train-path", type=str, default=None, help="Override train JSONL path")
    parser.add_argument("--val-path", type=str, default=None, help="Override val JSONL path")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    # Apply overrides
    if args.epochs is not None:
        CONFIG["epochs"] = args.epochs
    if args.max_length is not None:
        CONFIG["max_length"] = args.max_length
    if args.batch_size is not None:
        CONFIG["per_device_batch_size"] = args.batch_size
    if args.grad_accum is not None:
        CONFIG["gradient_accumulation_steps"] = args.grad_accum
    if args.train_path is not None:
        CONFIG["train_path"] = args.train_path
    if args.val_path is not None:
        CONFIG["val_path"] = args.val_path
    if args.output_dir is not None:
        CONFIG["output_dir"] = args.output_dir

    if args.dataset_check:
        ok = dataset_check(CONFIG["train_path"], CONFIG["val_path"])
        sys.exit(0 if ok else 1)

    train(mode=args.mode)


if __name__ == "__main__":
    main()
