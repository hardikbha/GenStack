"""
Per-Split Threshold Tuning for XGenDet.
Grid-search optimal threshold for each split (ID/CM/CF/CD) on val set,
then apply to test set and save results.

Usage:
  python training/threshold_tuning.py \\
    --checkpoint checkpoints/hydrafake_finetune/best_model.pth \\
    --output_dir checkpoints/v5_thresholded
"""

import os, sys, json, argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgendet import XGenDet
from data.hydrafake_dataset import HydraFakeDataset, HydraFakeTestDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    p.add_argument("--val_json", default="/home/sachin.chaudhary/hydrafake/jsons/val/all.json")
    p.add_argument("--test_dir", default="/home/sachin.chaudhary/hydrafake/jsons/test")
    p.add_argument("--data_root", default="/home/sachin.chaudhary")
    p.add_argument("--image_root", default="/home/sachin.chaudhary/hydrafake/test")
    p.add_argument("--output_dir", default="checkpoints/v5_thresholded")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=224)
    return p.parse_args()


def load_model(ckpt_path, device):
    model = XGenDet()
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Run model on loader. Returns (confidences, labels)."""
    preds, labels = [], []
    for batch in tqdm(loader, leave=False, ncols=100):
        if len(batch) == 3:  # train dataset with families
            imgs, lab, _ = batch
        else:  # test dataset
            imgs, lab = batch
        imgs = imgs.to(device)
        out = model(imgs, return_heatmap=False)
        preds.extend(out["confidence"].squeeze(-1).cpu().tolist())
        labels.extend(lab.tolist())
    return np.array(preds), np.array(labels)


def find_best_threshold(confidences, labels, thresholds=None):
    """Grid-search best threshold on val set."""
    if thresholds is None:
        thresholds = np.arange(0.3, 0.75, 0.05)

    best_acc, best_thresh = 0.0, 0.5
    for t in thresholds:
        preds = (confidences > t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_thresh = acc, t

    return best_thresh, best_acc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Evaluate on val set to find per-split optimal thresholds
    print("\n" + "="*60)
    print("Tuning thresholds on val set")
    print("="*60)

    val_ds = HydraFakeDataset(args.val_json, args.data_root, is_train=False, crop_size=args.crop_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # For val set, we don't know split, so use single global threshold
    print("\nVal set (global threshold):")
    val_preds, val_labels = evaluate_model(model, val_loader, device)
    global_thresh, global_acc = find_best_threshold(val_preds, val_labels)
    print(f"  Best threshold: {global_thresh:.2f}, Acc: {global_acc*100:.2f}%")

    # Evaluate on test set per split with global threshold
    print("\n" + "="*60)
    print(f"Test evaluation with threshold={global_thresh:.2f}")
    print("="*60)

    all_results = {}
    for split in ["id", "cm", "cf", "cd"]:
        split_dir = os.path.join(args.test_dir, split)
        if not os.path.isdir(split_dir):
            continue

        print(f"\n{split.upper()} split:")
        split_preds, split_labels = [], []
        per_gen = {}

        for jf in sorted(f for f in os.listdir(split_dir) if f.endswith(".json")):
            gen = jf.replace(".json", "")
            ds = HydraFakeTestDataset(os.path.join(split_dir, jf), args.data_root,
                                      args.image_root, args.crop_size)
            if len(ds) == 0:
                continue

            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
            g_preds, g_labels = evaluate_model(model, loader, device)

            g_preds_bin = (g_preds > global_thresh).astype(int)
            g_acc = accuracy_score(g_labels, g_preds_bin)
            g_ap = average_precision_score(g_labels, g_preds) if len(np.unique(g_labels)) > 1 else -1
            per_gen[gen] = {"acc": float(g_acc), "ap": float(g_ap), "n": len(ds)}
            split_preds.extend(g_preds.tolist())
            split_labels.extend(g_labels.tolist())
            print(f"  {gen:20s}: Acc={g_acc*100:.1f}%, AP={g_ap:.4f}, n={len(ds)}")

        if split_preds:
            s_preds_bin = (np.array(split_preds) > global_thresh).astype(int)
            s_labels = np.array(split_labels)
            s_acc = accuracy_score(s_labels, s_preds_bin)
            s_ap = average_precision_score(s_labels, np.array(split_preds)) if len(np.unique(s_labels)) > 1 else -1
            all_results[split] = {"acc": float(s_acc), "ap": float(s_ap), "n": len(split_preds),
                                 "per_generator": per_gen, "threshold": float(global_thresh)}
            print(f"  {split.upper()} TOTAL: Acc={s_acc*100:.1f}%, AP={s_ap:.4f}")

    if all_results:
        avg_acc = float(np.mean([r["acc"] for r in all_results.values()]))
        avg_ap = float(np.mean([r["ap"] for r in all_results.values() if r["ap"] >= 0]))
        all_results["average"] = {"acc": avg_acc, "ap": avg_ap, "threshold": float(global_thresh)}
        print(f"\n{'='*60}")
        print(f"THRESHOLD-TUNED AVERAGE: Acc={avg_acc*100:.1f}%, AP={avg_ap:.4f}")
        print(f"{'='*60}")

    # Save
    out_path = os.path.join(args.output_dir, "test_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
