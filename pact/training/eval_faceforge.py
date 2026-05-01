"""
Evaluate FaceForge-Net on a single HydraFake test split.

Reports per-generator accuracy, overall accuracy, AP, and F1-macro.
Saves results to --output_dir/{split}_results.json.

Usage:
    python training/eval_faceforge.py \
        --model_path checkpoints/faceforge/best_model.pth \
        --test_json  /home/sachin.chaudhary/hydrafake/jsons/test/id_test.json \
        --data_root  /home/sachin.chaudhary \
        --cache_dir  /home/sachin.chaudhary/hydrafake/faceforge_crops \
        --split      id \
        --output_dir checkpoints/faceforge/eval
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.faceforge_net import FaceForgeNet
from data.faceforge_dataset import FaceForgeDataset


def collate_fn(batch):
    crops_batch = {k: torch.stack([b["crops"][k] for b in batch]) for k in batch[0]["crops"]}
    full_images = torch.stack([b["full_image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    paths = [b["image_path"] for b in batch]
    return {"crops": crops_batch, "full_image": full_images, "label": labels, "path": paths}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--test_json",  required=True)
    p.add_argument("--data_root",  default="/home/sachin.chaudhary")
    p.add_argument("--cache_dir",  required=True)
    p.add_argument("--split",      default="id")
    p.add_argument("--output_dir", default="checkpoints/faceforge/eval")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--bf16",       action="store_true", default=True)
    return p.parse_args()


def infer_generator_from_path(image_path: str) -> str:
    path_lower = image_path.lower()
    generators = [
        "faceforensics", "ff++",
        "facevid2vid", "hallo2", "stylegan",
        "midjourney", "deepfacelab", "simswap",
        "ghost", "e4s", "starganv2", "iclight",
    ]
    for g in generators:
        if g in path_lower:
            return g
    parts = Path(image_path).parts
    if len(parts) >= 3:
        return parts[-3]
    return "unknown"


def main():
    args = parse_args()

    print("=" * 60)
    print(f" FaceForge-Net Evaluation | split={args.split}")
    print(f" model: {args.model_path}")
    print(f" data:  {args.test_json}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32

    # Load model
    model = FaceForgeNet()
    ckpt = torch.load(args.model_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    # Strip DataParallel 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING missing keys: {missing[:5]}...")
    model = model.to(device).to(dtype)
    model.eval()
    print(f"  Loaded model from {args.model_path}")

    # Dataset — test split may not be in train/val cache; use split name from --split
    dataset = FaceForgeDataset(
        json_path=args.test_json,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        split=args.split,
        augment=False,
    )
    # Pre-extract test crops if needed
    dataset.pre_extract_all(num_workers=args.num_workers)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    all_probs, all_labels, all_paths = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {args.split}"):
            crops = {k: v.to(device, dtype=dtype) for k, v in batch["crops"].items()}
            full_image = batch["full_image"].to(device, dtype=dtype)
            labels = batch["label"]

            with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                out = model(crops, full_image)

            probs = out["prob"].squeeze(1).cpu().float()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
            all_paths.extend(batch["path"])

    # Metrics
    probs_t = torch.tensor(all_probs)
    labels_t = torch.tensor(all_labels)
    preds_t = (probs_t >= args.threshold).float()

    tp = ((preds_t == 1) & (labels_t == 1)).sum().item()
    tn = ((preds_t == 0) & (labels_t == 0)).sum().item()
    fp = ((preds_t == 1) & (labels_t == 0)).sum().item()
    fn = ((preds_t == 0) & (labels_t == 1)).sum().item()

    acc = (tp + tn) / max(len(all_labels), 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    # Average Precision
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        ap  = average_precision_score(all_labels, all_probs)
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        ap = auc = -1.0

    # Per-generator breakdown
    gen_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for prob, label, path in zip(all_probs, all_labels, all_paths):
        g = infer_generator_from_path(path)
        pred = 1 if prob >= args.threshold else 0
        gen_stats[g]["total"] += 1
        if pred == int(label):
            gen_stats[g]["correct"] += 1

    gen_acc = {g: v["correct"] / max(v["total"], 1) for g, v in gen_stats.items()}
    gen_acc_sorted = dict(sorted(gen_acc.items(), key=lambda x: -x[1]))

    print(f"\n{'='*60}")
    print(f" Split: {args.split.upper()}")
    print(f"  Overall Acc: {acc*100:.2f}%")
    print(f"  AP:          {ap:.4f}")
    print(f"  AUC:         {auc:.4f}")
    print(f"  F1-Macro:    {f1:.4f}")
    print(f"  Precision:   {prec:.4f}  Recall: {rec:.4f}")
    print(f"\n  Per-generator accuracy:")
    for g, a in gen_acc_sorted.items():
        n = gen_stats[g]["total"]
        print(f"    {g:<30} {a*100:.1f}%  ({gen_stats[g]['correct']}/{n})")
    print("=" * 60)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "split": args.split,
        "model_path": args.model_path,
        "num_samples": len(all_labels),
        "threshold": args.threshold,
        "accuracy": acc,
        "ap": ap,
        "auc": auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "per_generator": {g: {"acc": gen_acc[g], **gen_stats[g]} for g in gen_stats},
    }

    out_path = output_dir / f"{args.split}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
