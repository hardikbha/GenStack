"""
Stage 1: SFT (Supervised Fine-Tuning) for Qwen3-VL-8B on HydraFake.

Trains with LoRA on pattern-annotated deepfake reasoning data.
Uses HuggingFace Trainer + PEFT for robust multi-GPU training.

Usage:
    # Single-node multi-GPU with DeepSpeed ZeRO-2
    deepspeed --num_gpus 4 training/train_qwen3_sft.py \
        --train_data /home/sachin.chaudhary/hydrafake/jsons/train/qwen3_sft_data.jsonl \
        --output_dir checkpoints/qwen3_sft \
        --deepspeed configs/ds_zero2.json

    # Or with torchrun
    torchrun --nproc_per_node 4 training/train_qwen3_sft.py \
        --train_data /home/sachin.chaudhary/hydrafake/jsons/train/qwen3_sft_data.jsonl \
        --output_dir checkpoints/qwen3_sft
"""

import os, sys, json, argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from torch.utils.data import Dataset

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType


# ── Dataset ──────────────────────────────────────────────────────────────────

class Qwen3VLSFTDataset(Dataset):
    """
    Dataset for Qwen3-VL SFT training.
    Each sample has multi-modal messages (image + text) → reasoning + answer.
    """

    def __init__(self, data_path: str, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        self.data = []

        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} SFT samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        messages = sample["messages"]

        # Apply chat template to get full text
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Process vision info from user message
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Squeeze batch dim
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Create labels: mask everything before assistant response with -100
        labels = input_ids.clone()

        # Find the assistant response start token
        # In Qwen3-VL, the assistant tag appears after the user message
        assistant_content = messages[-1]["content"]
        assistant_tokens = self.processor.tokenizer.encode(
            assistant_content, add_special_tokens=False
        )
        assistant_len = len(assistant_tokens)

        # Mask input prefix (everything except the last assistant_len tokens)
        input_len = (attention_mask.sum().item()) - assistant_len
        if input_len > 0:
            labels[:int(input_len)] = -100

        # Also mask padding
        labels[attention_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Add pixel values if present
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)

        return result


# ── Custom collator ──────────────────────────────────────────────────────────

@dataclass
class VLMDataCollator:
    """Collate multi-modal samples with variable-length pixel values."""

    processor: object

    def __call__(self, features):
        # Separate text and vision features
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Handle pixel values (variable length per sample)
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.cat(
                [f["pixel_values"] for f in features], dim=0
            )
        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.cat(
                [f["image_grid_thw"] for f in features], dim=0
            )

        return batch


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    # Model
    p.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")

    # Data
    p.add_argument("--train_data", required=True)
    p.add_argument("--val_data", default=None)
    p.add_argument("--max_length", type=int, default=2048)

    # Training
    p.add_argument("--output_dir", default="checkpoints/qwen3_sft")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)

    # DeepSpeed
    p.add_argument("--deepspeed", default=None, help="Path to deepspeed config")
    p.add_argument("--local_rank", type=int, default=-1)

    return p.parse_args()


def main():
    args = parse_args()
    print(f"{'='*60}")
    print(f"Stage 1: SFT Training — Qwen3-VL-8B + LoRA")
    print(f"{'='*60}")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
    }
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"Loading {args.model_name}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name, **model_kwargs
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Freeze vision encoder
    for name, param in model.named_parameters():
        if "visual" in name and "lora" not in name:
            param.requires_grad = False

    # Datasets
    train_dataset = Qwen3VLSFTDataset(args.train_data, processor, args.max_length)

    val_dataset = None
    if args.val_data:
        val_dataset = Qwen3VLSFTDataset(args.val_data, processor, args.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        deepspeed=args.deepspeed,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=VLMDataCollator(processor=processor),
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch per GPU: {args.batch_size}")
    print(f"  Grad accum: {args.grad_accum}")
    print(f"  LR: {args.lr}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Max length: {args.max_length}")
    print(f"  DeepSpeed: {args.deepspeed or 'disabled'}")

    trainer.train()

    # Save final LoRA weights
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nTraining complete. LoRA weights saved to {final_dir}")


if __name__ == "__main__":
    main()
