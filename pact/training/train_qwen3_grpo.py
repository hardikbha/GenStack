"""
Stage 3: Part-Aware GRPO Training for Qwen3-VL-8B.

Uses trl's GRPOTrainer with custom reward functions:
  R = R_acc + λ₁·R_part + λ₂·R_cons + λ₃·R_fmt

Trains on hard samples (D4) from Stage 2 reject sampling.

Usage:
    # Multi-GPU
    deepspeed --num_gpus 4 training/train_qwen3_grpo.py \
        --model_path checkpoints/qwen3_sft/final \
        --train_data /home/sachin.chaudhary/hydrafake/jsons/train/qwen3_grpo_data.jsonl \
        --srm_evidence /home/sachin.chaudhary/hydrafake/srm_evidence/srm_evidence.json \
        --output_dir checkpoints/qwen3_grpo \
        --deepspeed configs/ds_zero2.json

    # Single GPU
    python training/train_qwen3_grpo.py \
        --model_path checkpoints/qwen3_sft/final \
        --train_data /home/sachin.chaudhary/hydrafake/jsons/train/qwen3_grpo_data.jsonl \
        --output_dir checkpoints/qwen3_grpo
"""

import os, sys, json, argparse
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset as HFDataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel, LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.rewards import compute_composite_reward, extract_answer


# ── Reward function wrapper for GRPOTrainer ──────────────────────────────────

class PartAwareRewardFn:
    """
    Custom reward function for GRPOTrainer.

    GRPOTrainer calls reward_fn(completions, prompts) → list[float].
    We intercept to extract ground truth and SRM evidence from prompts.
    """

    def __init__(
        self,
        srm_evidence: dict = None,
        lambda_part: float = 0.5,
        lambda_cons: float = 0.3,
        lambda_fmt: float = 0.2,
    ):
        self.srm_evidence = srm_evidence or {}
        self.lambda_part = lambda_part
        self.lambda_cons = lambda_cons
        self.lambda_fmt = lambda_fmt

    def __call__(self, completions: list, prompts: list = None, **kwargs) -> list[float]:
        """
        Compute rewards for a batch of completions.

        Args:
            completions: List of dicts with 'content' key (generated text)
            prompts: List of dicts with prompt content

        Returns:
            List of float rewards
        """
        rewards = []
        for i, completion in enumerate(completions):
            # Extract generated text
            if isinstance(completion, dict):
                text = completion.get("content", str(completion))
            elif isinstance(completion, list):
                text = completion[-1].get("content", "") if completion else ""
            else:
                text = str(completion)

            # Extract ground truth from the prompt metadata
            gt = "unknown"
            srm = None
            if prompts and i < len(prompts):
                prompt = prompts[i]
                if isinstance(prompt, dict):
                    gt = prompt.get("ground_truth", "unknown")
                    image_path = prompt.get("image_path", "")
                    srm = self.srm_evidence.get(image_path)
                elif isinstance(prompt, list):
                    # Look for metadata in the prompt messages
                    for msg in prompt:
                        if isinstance(msg, dict) and "ground_truth" in msg:
                            gt = msg["ground_truth"]
                            break

            result = compute_composite_reward(
                text, gt, srm,
                self.lambda_part, self.lambda_cons, self.lambda_fmt
            )
            rewards.append(result["total"])

        return rewards


# ── Dataset preparation ──────────────────────────────────────────────────────

def load_grpo_dataset(data_path: str, processor) -> HFDataset:
    """
    Load GRPO training data and format as HuggingFace Dataset.

    Each sample needs:
      - 'prompt': The formatted prompt (system + user messages)
      - 'ground_truth': The correct answer for reward computation
      - 'image_path': For SRM evidence lookup
    """
    samples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            messages = item["messages"]
            metadata = item.get("metadata", {})

            # Build prompt (system + user messages only, no assistant)
            prompt_messages = [m for m in messages if m["role"] != "assistant"]

            # Apply chat template for the prompt
            prompt_text = processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            samples.append({
                "prompt": prompt_text,
                "ground_truth": metadata.get("answer", "unknown"),
                "image_path": metadata.get("image_path", ""),
                "label": metadata.get("label", -1),
            })

    return HFDataset.from_list(samples)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    # Model
    p.add_argument("--model_path", required=True,
                   help="Path to Stage 1/2 SFT checkpoint (LoRA weights)")
    p.add_argument("--base_model", default="Qwen/Qwen3-VL-8B-Instruct",
                   help="Base model name (for loading base + LoRA)")

    # Data
    p.add_argument("--train_data", required=True,
                   help="JSONL with hard samples for GRPO")
    p.add_argument("--srm_evidence", default=None,
                   help="Path to srm_evidence.json for part-aware rewards")

    # GRPO config
    p.add_argument("--output_dir", default="checkpoints/qwen3_grpo")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--num_generations", type=int, default=4,
                   help="G: number of responses sampled per query")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--beta", type=float, default=0.0,
                   help="KL penalty coefficient (0 = disabled, like Veritas)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)

    # Reward weights
    p.add_argument("--lambda_part", type=float, default=0.5)
    p.add_argument("--lambda_cons", type=float, default=0.3)
    p.add_argument("--lambda_fmt", type=float, default=0.2)

    # DeepSpeed
    p.add_argument("--deepspeed", default=None)
    p.add_argument("--local_rank", type=int, default=-1)

    return p.parse_args()


def main():
    args = parse_args()
    print(f"{'='*60}")
    print(f"Stage 3: Part-Aware GRPO — Qwen3-VL-8B")
    print(f"{'='*60}")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True
    )

    # Load base model + LoRA weights from SFT
    print(f"Loading base model: {args.base_model}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    # Load SFT LoRA weights
    if os.path.exists(args.model_path):
        print(f"Loading SFT LoRA from: {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()  # Merge LoRA into base for GRPO
        print("LoRA merged into base model")

    # Apply fresh LoRA for GRPO (train new LoRA on top of merged SFT model)
    lora_config = LoraConfig(
        r=16,  # Smaller r for RL fine-tuning
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # Load SRM evidence for reward computation
    srm_evidence = {}
    if args.srm_evidence and os.path.exists(args.srm_evidence):
        with open(args.srm_evidence) as f:
            srm_evidence = json.load(f)
        print(f"Loaded SRM evidence for {len(srm_evidence)} images")

    # Create reward function
    reward_fn = PartAwareRewardFn(
        srm_evidence=srm_evidence,
        lambda_part=args.lambda_part,
        lambda_cons=args.lambda_cons,
        lambda_fmt=args.lambda_fmt,
    )

    # Load dataset
    train_dataset = load_grpo_dataset(args.train_data, processor)
    print(f"Loaded {len(train_dataset)} GRPO training samples")

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        bf16=args.bf16,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        seed=args.seed,
        report_to="none",
        deepspeed=args.deepspeed,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=lora_config,
    )

    # Train
    print(f"\nStarting GRPO training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch per GPU: {args.batch_size}")
    print(f"  Generations (G): {args.num_generations}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Beta (KL): {args.beta}")
    print(f"  LR: {args.lr}")
    print(f"  Reward weights: λ_part={args.lambda_part}, λ_cons={args.lambda_cons}, λ_fmt={args.lambda_fmt}")

    trainer.train()

    # Save
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nGRPO training complete. Weights saved to {final_dir}")


if __name__ == "__main__":
    main()
