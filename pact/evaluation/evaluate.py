"""
XGenDet Evaluation Script.

Evaluates a trained XGenDet model on OOD generators with comprehensive metrics.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import (
    average_precision_score, accuracy_score, roc_auc_score, f1_score
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xgendet import XGenDet
from data.dataset import create_dataloader
from training.calibration import TemperatureScaling, compute_ece


def parse_args():
    parser = argparse.ArgumentParser(description="XGenDet Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="/home/sachin.chaudhary/GTA")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    parser.add_argument("--eval_type", type=str, default="ood",
                       choices=["ood", "id", "both"])
    return parser.parse_args()


def find_best_threshold(y_true, y_pred):
    """Find threshold that maximizes F1 score."""
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (y_pred >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def evaluate_generator(model, loader, device):
    """Evaluate model on a single generator dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_families_pred = []
    all_families_true = []

    with torch.no_grad():
        for imgs, labels, families in loader:
            imgs = imgs.to(device)
            outputs = model(imgs, return_heatmap=False)

            confidence = outputs["confidence"].squeeze(-1).cpu().numpy()
            family_pred = outputs["family_logit"].argmax(dim=-1).cpu().numpy()

            all_preds.extend(confidence.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_families_pred.extend(family_pred.tolist())
            all_families_true.extend(families.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Metrics
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    acc_05 = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    best_thresh, best_f1 = find_best_threshold(y_true, y_pred)
    best_acc = accuracy_score(y_true, (y_pred > best_thresh).astype(int))

    # Calibration
    correct = ((y_pred > 0.5).astype(int) == y_true).astype(float)
    ece = compute_ece(y_pred, correct)

    # Per-class accuracy
    real_mask = y_true == 0
    fake_mask = y_true == 1
    r_acc = accuracy_score(y_true[real_mask], (y_pred[real_mask] > 0.5).astype(int)) if real_mask.any() else 0.0
    f_acc = accuracy_score(y_true[fake_mask], (y_pred[fake_mask] > 0.5).astype(int)) if fake_mask.any() else 0.0

    return {
        "ap": float(ap),
        "auc": float(auc),
        "acc": float(acc_05),
        "best_acc": float(best_acc),
        "best_threshold": float(best_thresh),
        "best_f1": float(best_f1),
        "ece": float(ece),
        "r_acc": float(r_acc),
        "f_acc": float(f_acc),
        "num_real": int(real_mask.sum()),
        "num_fake": int(fake_mask.sum()),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get("args", {})

    # Build model
    model = XGenDet(
        clip_model_name=model_args.get("clip_model", "ViT-L/14"),
        num_prompt_tokens=model_args.get("num_prompt_tokens", 8),
        num_prototypes=model_args.get("num_prototypes", 128),
        proto_dim=model_args.get("proto_dim", 128),
        shuffle_patch_size=model_args.get("shuffle_patch_size", 32),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully")

    # Evaluate
    results = {}
    data_root = args.data_root

    if args.eval_type in ["ood", "both"]:
        ood_generators = [
            "crn", "cyclegan", "dalle", "deepfake", "gaugan", "imle",
            "san", "seeingdark", "stargan", "stylegan", "stylegan2", "whichfaceisreal"
        ]

        print("\n=== OOD Evaluation ===")
        ood_results = {}
        for gen in ood_generators:
            real_folder = os.path.join(data_root, "OOD_GENERATORS", gen, "0_real")
            fake_folder = os.path.join(data_root, "OOD_GENERATORS", gen, "1_fake")

            if not (os.path.isdir(real_folder) and os.path.isdir(fake_folder)):
                print(f"  Skipping {gen} (not found)")
                continue

            loader = create_dataloader(
                real_folders=[real_folder],
                fake_folders=[fake_folder],
                is_train=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                generator_names=[gen],
            )

            metrics = evaluate_generator(model, loader, device)
            ood_results[gen] = metrics
            print(f"  {gen}: AP={metrics['ap']:.4f} Acc={metrics['acc']:.4f} "
                  f"BestAcc={metrics['best_acc']:.4f} ECE={metrics['ece']:.4f}")

        # Averages
        avg_metrics = {}
        for key in ["ap", "auc", "acc", "best_acc", "ece"]:
            avg_metrics[key] = float(np.mean([v[key] for v in ood_results.values()]))

        ood_results["average"] = avg_metrics
        results["ood"] = ood_results

        print(f"\n  OOD Average: AP={avg_metrics['ap']:.4f} Acc={avg_metrics['acc']:.4f} "
              f"BestAcc={avg_metrics['best_acc']:.4f} ECE={avg_metrics['ece']:.4f}")

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
