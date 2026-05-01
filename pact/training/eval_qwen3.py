"""
Evaluate Qwen3-VL deepfake detector on HydraFake test set.

Runs the model on all 4 test splits (ID, CM, CF, CD) with per-generator
breakdown. Reports accuracy, AP, and comparison with baselines.

Usage:
    python training/eval_qwen3.py \
        --model_path checkpoints/qwen3_sft/final \
        --test_dir /home/sachin.chaudhary/hydrafake/jsons/test \
        --data_root /home/sachin.chaudhary \
        --output_dir checkpoints/qwen3_eval
"""

import os, sys, json, argparse, re, time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel


SYSTEM_PROMPT = (
    "You are a deepfake forensics expert specializing in facial image authenticity analysis. "
    "Analyze the image carefully. Think step by step in <think> tags, then provide "
    "your final verdict as either \"real\" or \"fake\" in <answer> tags."
)


def extract_answer(text: str) -> str:
    """Extract answer from model output."""
    match = re.search(r'<answer>\s*(real|fake)\s*</answer>', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    text_lower = text.lower().strip()
    if "fake" in text_lower.split()[-3:]:
        return "fake"
    if "real" in text_lower.split()[-3:]:
        return "real"
    return "unknown"


def load_model(model_path, base_model, device, bf16=True):
    """Load base model + LoRA weights."""
    processor = AutoProcessor.from_pretrained(
        base_model, trust_remote_code=True
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
    )

    if os.path.exists(model_path):
        model = PeftModel.from_pretrained(model, model_path)
        print(f"Loaded LoRA weights from {model_path}")

    model.eval()
    return model, processor


def run_inference(model, processor, image_path, max_new_tokens=512):
    """Run single-image inference."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": "Determine if this facial image is real or fake."},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.tokenizer.decode(generated, skip_special_tokens=True)

    return response


def evaluate_split(model, processor, split_dir, data_root, test_image_root):
    """Evaluate on a single test split (ID, CM, CF, CD)."""
    results = {}

    for json_file in sorted(os.listdir(split_dir)):
        if not json_file.endswith(".json"):
            continue

        gen_name = json_file.replace(".json", "")
        json_path = os.path.join(split_dir, json_file)

        with open(json_path) as f:
            test_data = json.load(f)

        if not test_data:
            continue

        preds, labels, responses = [], [], []
        for sample in tqdm(test_data, desc=f"  {gen_name}", leave=False):
            image_rel = sample["images"][0]
            label = sample["label"]

            # Try multiple path resolutions
            image_path = os.path.join(data_root, image_rel)
            if not os.path.exists(image_path):
                image_path = os.path.join(test_image_root, os.path.basename(image_rel))
            if not os.path.exists(image_path):
                continue

            response = run_inference(model, processor, image_path)
            answer = extract_answer(response)

            # Convert to binary
            pred = 1 if answer == "fake" else 0
            preds.append(pred)
            labels.append(label)
            responses.append(response)

        if not preds:
            continue

        preds_np = np.array(preds)
        labels_np = np.array(labels)

        acc = accuracy_score(labels_np, preds_np)
        ap = average_precision_score(labels_np, preds_np) if len(np.unique(labels_np)) > 1 else -1

        results[gen_name] = {
            "accuracy": acc,
            "ap": ap,
            "n": len(preds),
            "correct": int((preds_np == labels_np).sum()),
        }
        print(f"    {gen_name}: Acc={acc*100:.1f}%, AP={ap:.4f}, n={len(preds)}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to LoRA checkpoint")
    p.add_argument("--base_model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--test_dir", default="/home/sachin.chaudhary/hydrafake/jsons/test")
    p.add_argument("--data_root", default="/home/sachin.chaudhary")
    p.add_argument("--test_image_root", default="/home/sachin.chaudhary/hydrafake/test")
    p.add_argument("--output_dir", default="checkpoints/qwen3_eval")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--bf16", action="store_true", default=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(args.model_path, args.base_model, device, args.bf16)
    print(f"Model loaded on {model.device}")

    # Evaluate each split
    all_results = {}
    split_names = ["id", "cm", "cf", "cd"]

    for split in split_names:
        split_dir = os.path.join(args.test_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping {split} (directory not found)")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating split: {split.upper()}")
        print(f"{'='*50}")

        t0 = time.time()
        split_results = evaluate_split(
            model, processor, split_dir, args.data_root, args.test_image_root
        )
        elapsed = time.time() - t0

        # Aggregate split-level metrics
        all_preds = sum(r["correct"] for r in split_results.values())
        all_total = sum(r["n"] for r in split_results.values())
        split_acc = all_preds / max(all_total, 1)
        split_ap = np.mean([r["ap"] for r in split_results.values() if r["ap"] >= 0])

        all_results[split] = {
            "accuracy": split_acc,
            "ap": split_ap,
            "n": all_total,
            "per_generator": split_results,
            "time_seconds": elapsed,
        }
        print(f"\n  {split.upper()} Overall: Acc={split_acc*100:.1f}%, AP={split_ap:.4f} ({elapsed:.0f}s)")

    # Overall average
    if all_results:
        avg_acc = np.mean([r["accuracy"] for r in all_results.values()])
        avg_ap = np.mean([r["ap"] for r in all_results.values() if r["ap"] >= 0])
        all_results["average"] = {"accuracy": avg_acc, "ap": avg_ap}

        print(f"\n{'='*60}")
        print(f"OVERALL AVERAGE: Acc={avg_acc*100:.1f}%, AP={avg_ap:.4f}")
        print(f"{'='*60}")

        # Comparison table
        print(f"\n{'='*60}")
        print(f"Comparison:")
        print(f"  XGenDet ensemble v5+v14: 90.2%")
        print(f"  Veritas (InternVL3-8B):  92.1%")
        print(f"  Ours (Qwen3-VL-8B):     {avg_acc*100:.1f}%")
        print(f"{'='*60}")

    # Save results
    output_path = os.path.join(args.output_dir, "test_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
