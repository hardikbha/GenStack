"""
Stage 2: Hard Sample Mining + Reject Sampling.

Runs the SFT model on the training set, identifies failures,
generates multiple responses per hard sample, and keeps the best ones.

Output:
  - qwen3_grpo_data.jsonl: Hard samples for GRPO training
  - hard_sample_stats.json: Statistics on failure patterns

Usage:
    python training/hard_sample_mining.py \
        --model_path checkpoints/qwen3_sft/final \
        --train_data /home/sachin.chaudhary/hydrafake/jsons/train/qwen3_sft_data.jsonl \
        --output /home/sachin.chaudhary/hydrafake/jsons/train/qwen3_grpo_data.jsonl \
        --num_generations 8 \
        --max_hard_samples 8000
"""

import os, sys, json, argparse, re
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel


def extract_answer(text: str) -> str:
    match = re.search(r'<answer>\s*(real|fake)\s*</answer>', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    text_lower = text.lower().strip()
    if "fake" in text_lower.split()[-3:]:
        return "fake"
    if "real" in text_lower.split()[-3:]:
        return "real"
    return "unknown"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--train_data", required=True, help="JSONL SFT training data")
    p.add_argument("--output", required=True, help="Output JSONL for GRPO")
    p.add_argument("--num_generations", type=int, default=8,
                   help="Number of responses to generate per hard sample")
    p.add_argument("--max_hard_samples", type=int, default=8000)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"{'='*60}")
    print(f"Stage 2: Hard Sample Mining")
    print(f"{'='*60}")

    # Load model
    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
    )

    if os.path.exists(args.model_path):
        model = PeftModel.from_pretrained(model, args.model_path)
        print(f"Loaded LoRA from {args.model_path}")

    model.eval()

    # Load training data
    train_data = []
    with open(args.train_data) as f:
        for line in f:
            line = line.strip()
            if line:
                train_data.append(json.loads(line))
    print(f"Loaded {len(train_data)} training samples")

    # Phase 1: Identify hard samples (wrong predictions with greedy decoding)
    print(f"\nPhase 1: Identifying hard samples...")
    hard_samples = []
    correct = 0
    total = 0

    for sample in tqdm(train_data, desc="Scanning"):
        metadata = sample.get("metadata", {})
        gt_answer = metadata.get("answer", "unknown")
        if gt_answer == "unknown":
            continue

        messages = sample["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]

        # Tokenize
        text = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info
        try:
            image_inputs, video_inputs = process_vision_info(prompt_messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        except Exception:
            continue

        # Greedy decode
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = processor.tokenizer.decode(generated, skip_special_tokens=True)
        pred = extract_answer(response)

        total += 1
        if pred == gt_answer:
            correct += 1
        else:
            hard_samples.append({
                "sample": sample,
                "wrong_prediction": pred,
                "ground_truth": gt_answer,
            })

        if len(hard_samples) >= args.max_hard_samples * 2:
            break  # Enough candidates

    acc = correct / max(total, 1)
    print(f"Scanned {total} samples: {correct} correct ({acc*100:.1f}%), {len(hard_samples)} hard")

    # Trim to max
    if len(hard_samples) > args.max_hard_samples:
        hard_samples = hard_samples[:args.max_hard_samples]

    # Phase 2: Reject sampling on hard samples
    print(f"\nPhase 2: Reject sampling on {len(hard_samples)} hard samples...")
    print(f"  Generating {args.num_generations} responses per sample")

    grpo_data = []
    accepted = 0

    for hs in tqdm(hard_samples, desc="Reject sampling"):
        sample = hs["sample"]
        gt = hs["ground_truth"]
        messages = sample["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]

        text = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(prompt_messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        except Exception:
            continue

        # Generate multiple responses with sampling
        best_response = None
        any_correct = False

        for g in range(args.num_generations):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            response = processor.tokenizer.decode(generated, skip_special_tokens=True)
            pred = extract_answer(response)

            if pred == gt:
                any_correct = True
                # Keep the longest correct response (more reasoning)
                if best_response is None or len(response) > len(best_response):
                    best_response = response

        # Add to GRPO data (both accepted and hard cases)
        grpo_sample = sample.copy()
        if best_response:
            # Accepted: use the correct response as the new target
            grpo_sample["messages"][-1]["content"] = best_response
            accepted += 1

        grpo_data.append(grpo_sample)

    print(f"\nReject sampling results: {accepted}/{len(hard_samples)} accepted")
    print(f"  ({accepted/max(len(hard_samples),1)*100:.1f}% accept rate)")

    # Save GRPO data
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for item in grpo_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(grpo_data)} samples to {args.output}")

    # Save stats
    stats = {
        "total_scanned": total,
        "total_correct": correct,
        "accuracy": acc,
        "hard_samples": len(hard_samples),
        "accepted_after_reject": accepted,
        "accept_rate": accepted / max(len(hard_samples), 1),
        "grpo_data_size": len(grpo_data),
    }
    stats_path = args.output.replace(".jsonl", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
