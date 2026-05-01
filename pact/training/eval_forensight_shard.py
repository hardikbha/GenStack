"""
ForenSight Sharded Evaluation — run one shard per GPU in parallel.
Usage: python eval_forensight_shard.py --shard_id 0 --num_shards 8 --model_path ...
Each shard evaluates a subset of test generators, then results are merged.
"""

import os, sys, json, re, argparse, torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="/home/sachin.chaudhary/models/Qwen3-VL-8B-Instruct")
    p.add_argument("--adapter_path", default="checkpoints/forensight_sft/final_adapter")
    p.add_argument("--xgendet_ckpt", default="checkpoints/hydrafake_finetune/best_model.pth")
    p.add_argument("--test_dir", default="/home/sachin.chaudhary/hydrafake/jsons/test")
    p.add_argument("--output_dir", default="checkpoints/forensight_sft/eval_shards")
    p.add_argument("--shard_id", type=int, required=True)
    p.add_argument("--num_shards", type=int, default=8)
    return p.parse_args()


def extract_answer(text):
    match = re.search(r"<answer>\s*(real|fake)\s*</answer>", text.lower())
    if match:
        return 1 if match.group(1) == "fake" else 0
    text_lower = text.lower()
    if "fake" in text_lower and "real" not in text_lower:
        return 1
    if "real" in text_lower and "fake" not in text_lower:
        return 0
    return -1


def build_evidence(img_path, xgendet, transform, device):
    try:
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = xgendet(img_t, return_heatmap=False)
        conf = out["confidence"].item()
        family_idx = out["family_logit"].argmax(dim=-1).item()
        attrs = out["attr_scores"][0].cpu().tolist()
        families = ["Real", "Face Swapping", "Entire Face Gen", "Face Reenactment"]
        attr_names = ["texture", "edges", "color", "geometry", "semantics", "frequency"]
        sorted_attrs = sorted(zip(attr_names, attrs), key=lambda x: -x[1])
        verdict = "likely fake" if conf > 0.5 else "likely real"
        text = f"Detection confidence: {int(conf*100)}% ({verdict})\n"
        text += f"Generator type: {families[family_idx]}\n"
        text += "Attributes: " + ", ".join(f"{n}={v:.2f}" for n, v in sorted_attrs) + "\n"
        return text
    except:
        return "Evidence unavailable.\n"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all test JSON files across splits
    all_jobs = []
    for split in ["id", "cm", "cf", "cd"]:
        split_dir = os.path.join(args.test_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for jf in sorted(f for f in os.listdir(split_dir) if f.endswith(".json")):
            all_jobs.append((split, jf))

    # Shard assignment
    my_jobs = [j for i, j in enumerate(all_jobs) if i % args.num_shards == args.shard_id]
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(my_jobs)} generator files to evaluate")
    for s, j in my_jobs:
        print(f"  {s}/{j}")

    if not my_jobs:
        print("No jobs for this shard")
        return

    # Load VLM
    print(f"Loading VLM from {args.model_name}...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    # Load model — either full fine-tuned or base + adapter
    if args.adapter_path and args.adapter_path != "NONE" and os.path.exists(args.adapter_path):
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        print(f"Loaded base + adapter from {args.adapter_path}")
    else:
        # Full fine-tuned model saved directly
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        print(f"Loaded full model from {args.model_name}")

    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Load XGenDet on CPU
    from models.xgendet import XGenDet
    from data.augmentations import get_eval_transforms
    xgendet = XGenDet()
    xgendet.load_state_dict(
        torch.load(args.xgendet_ckpt, map_location="cpu")["model_state_dict"]
    )
    xgendet.eval()
    xgendet_transform = get_eval_transforms(224)

    SYSTEM_PROMPT = "You are a forensic image analyst. Based on the image and detector evidence, determine if this face is real or fake. Use <answer> real </answer> or <answer> fake </answer>."

    shard_results = {}

    for split, jf in my_jobs:
        gen_name = jf.replace(".json", "")
        json_path = os.path.join(args.test_dir, split, jf)

        with open(json_path) as f:
            test_data = json.load(f)

        gp, gl = [], []
        for item in tqdm(test_data, desc=f"Shard{args.shard_id} {split}/{gen_name}"):
            rel_path = item["images"][0]
            parts = rel_path.split("/")
            sub_path = "/".join(parts[2:])
            full_path = os.path.join("/home/sachin.chaudhary/hydrafake/test", sub_path)
            label = item["label"]

            if not os.path.exists(full_path):
                continue

            evidence = build_evidence(full_path, xgendet, xgendet_transform, torch.device("cpu"))

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": full_path},
                    {"type": "text", "text": f"\nDetector Evidence:\n{evidence}\nIs this face real or fake?"},
                ]},
            ]

            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                img = Image.open(full_path).convert("RGB")
                inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

                response = processor.decode(output_ids[0], skip_special_tokens=True)
                pred = extract_answer(response)

                if pred >= 0:
                    gp.append(pred)
                    gl.append(label)
            except:
                continue

        if gp:
            acc = accuracy_score(gl, gp)
            shard_results[f"{split}/{gen_name}"] = {"acc": acc, "n": len(gp), "preds": gp, "labels": gl}
            print(f"  {split}/{gen_name}: Acc={acc*100:.1f}%, n={len(gp)}")

    # Save shard results
    out_path = os.path.join(args.output_dir, f"shard_{args.shard_id}.json")
    # Convert lists to serializable format
    save_results = {}
    for k, v in shard_results.items():
        save_results[k] = {"acc": v["acc"], "n": v["n"]}
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nShard {args.shard_id} results saved to {out_path}")


if __name__ == "__main__":
    main()
