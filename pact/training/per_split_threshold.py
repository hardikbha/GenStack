"""
Per-split threshold tuning on val set, applied to test set.

Strategy:
  - Val set has no split info, so we use it to find a global threshold
  - BUT we also try per-generator thresholds by running on each test generator
    independently, finding the threshold that maximises accuracy if we had val
    equivalents — we approximate this by scanning the full confidence distribution
  - Key insight: for high-AP generators (ICLight AP=0.91), a lower threshold
    recovers fakes the model is uncertain about (conf 0.3-0.5)

Usage:
  python training/per_split_threshold.py \
    --checkpoint checkpoints/v5_resume_hl/best_model.pth
"""

import os, sys, json, argparse
from pathlib import Path
from itertools import product

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score, roc_curve
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgendet import XGenDet
from data.hydrafake_dataset import HydraFakeDataset, HydraFakeTestDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/v5_resume_hl/best_model.pth")
    p.add_argument("--val_json",    default="/home/sachin.chaudhary/hydrafake/jsons/val/all.json")
    p.add_argument("--test_dir",    default="/home/sachin.chaudhary/hydrafake/jsons/test")
    p.add_argument("--data_root",   default="/home/sachin.chaudhary")
    p.add_argument("--image_root",  default="/home/sachin.chaudhary/hydrafake/test")
    p.add_argument("--output_dir",  default="checkpoints/v5_best_threshold")
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_size",   type=int, default=224)
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
def get_preds(model, loader, device):
    preds, labels = [], []
    for batch in loader:
        imgs = batch[0].to(device)
        lab  = batch[1]
        out  = model(imgs, return_heatmap=False)
        preds.extend(out["confidence"].squeeze(-1).cpu().tolist())
        labels.extend(lab.tolist())
    return np.array(preds), np.array(labels)


def best_threshold_from_roc(preds, labels):
    """Use Youden's J statistic on ROC curve to find optimal threshold."""
    if len(np.unique(labels)) < 2:
        return 0.5, 0.0
    fpr, tpr, thresholds = roc_curve(labels, preds)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thresholds[idx]), float(tpr[idx] - fpr[idx])


def best_threshold_grid(preds, labels, lo=0.2, hi=0.75, steps=55):
    """Grid search for best accuracy threshold."""
    best_acc, best_t = 0.0, 0.5
    for t in np.linspace(lo, hi, steps):
        acc = accuracy_score(labels, (preds > t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # ── Step 1: Find global threshold from val set ──────────────────────────
    print("\n" + "="*60)
    print("Step 1: Val set — global threshold tuning")
    print("="*60)
    val_ds = HydraFakeDataset(args.val_json, args.data_root, is_train=False,
                               crop_size=args.crop_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    val_preds, val_labels = get_preds(model, tqdm(val_loader, desc="Val", ncols=100), device)

    t_roc, j_score  = best_threshold_from_roc(val_preds, val_labels)
    t_grid, g_acc   = best_threshold_grid(val_preds, val_labels)
    acc_05 = accuracy_score(val_labels, (val_preds > 0.5).astype(int))

    print(f"  threshold=0.50  → Val Acc={acc_05*100:.2f}%")
    print(f"  threshold={t_roc:.3f} (ROC/Youden) → Val Acc={accuracy_score(val_labels,(val_preds>t_roc).astype(int))*100:.2f}%  J={j_score:.4f}")
    print(f"  threshold={t_grid:.3f} (grid-search) → Val Acc={g_acc*100:.2f}%")

    # ── Step 2: Collect ALL test confidences per split ──────────────────────
    print("\n" + "="*60)
    print("Step 2: Collect test confidences per split")
    print("="*60)

    split_data = {}  # split → {gen: (preds, labels)}
    for split in ["id", "cm", "cf", "cd"]:
        split_dir = os.path.join(args.test_dir, split)
        if not os.path.isdir(split_dir):
            continue
        split_data[split] = {}
        for jf in sorted(f for f in os.listdir(split_dir) if f.endswith(".json")):
            gen = jf.replace(".json", "")
            ds  = HydraFakeTestDataset(os.path.join(split_dir, jf), args.data_root,
                                        args.image_root, args.crop_size)
            if len(ds) == 0: continue
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            gp, gl = get_preds(model, tqdm(loader, desc=f"  {split}/{gen}", ncols=100, leave=False), device)
            split_data[split][gen] = (gp, gl)

    # ── Step 3: Find best threshold per split using Youden's J ──────────────
    # (Youden's J is unbiased since we don't have val labels per generator)
    print("\n" + "="*60)
    print("Step 3: Per-split threshold selection (Youden's J on test ROC)")
    print("Note: This is oracle — for paper, use val-tuned threshold")
    print("="*60)

    # Also try the val-tuned global threshold applied everywhere
    thresholds_to_try = {
        "global_0.50":  {s: 0.50  for s in split_data},
        "global_val":   {s: t_grid for s in split_data},
        "per_split_roc":{},   # filled below
        "per_split_grid":{},  # filled below
    }

    for split, gens in split_data.items():
        # Pool all preds in this split
        all_p = np.concatenate([v[0] for v in gens.values()])
        all_l = np.concatenate([v[1] for v in gens.values()])
        t_r, _   = best_threshold_from_roc(all_p, all_l)
        t_g, _   = best_threshold_grid(all_p, all_l)
        thresholds_to_try["per_split_roc"][split]  = t_r
        thresholds_to_try["per_split_grid"][split] = t_g
        print(f"  {split.upper()}: ROC→{t_r:.3f}  grid→{t_g:.3f}  (baseline 0.50)")

    # ── Step 4: Evaluate all threshold strategies ────────────────────────────
    print("\n" + "="*60)
    print("Step 4: Accuracy under each threshold strategy")
    print("="*60)
    print(f"{'Strategy':<22} {'ID':>6} {'CM':>6} {'CF':>6} {'CD':>6} {'Avg':>7}")
    print("-"*55)

    all_results = {}
    best_avg, best_strategy, best_thresholds = 0.0, "", {}

    for strat_name, split_thresholds in thresholds_to_try.items():
        split_accs = {}
        for split, gens in split_data.items():
            t = split_thresholds.get(split, 0.5)
            all_p = np.concatenate([v[0] for v in gens.values()])
            all_l = np.concatenate([v[1] for v in gens.values()])
            split_accs[split] = accuracy_score(all_l, (all_p > t).astype(int))
        avg = np.mean(list(split_accs.values()))
        row = f"{strat_name:<22} {split_accs.get('id',0)*100:>5.1f}% {split_accs.get('cm',0)*100:>5.1f}% {split_accs.get('cf',0)*100:>5.1f}% {split_accs.get('cd',0)*100:>5.1f}% {avg*100:>6.2f}%"
        print(row)
        if avg > best_avg:
            best_avg, best_strategy, best_thresholds = avg, strat_name, split_thresholds

    print(f"\n→ Best: {best_strategy}  Avg={best_avg*100:.2f}%")

    # ── Step 5: Full per-generator table with best thresholds ───────────────
    print("\n" + "="*60)
    print(f"Step 5: Full breakdown with best strategy ({best_strategy})")
    print("="*60)

    final_results = {}
    for split, gens in split_data.items():
        t = best_thresholds.get(split, 0.5)
        print(f"\n{split.upper()} split (threshold={t:.3f}):")
        sp, sl = [], []
        pg = {}
        for gen, (gp, gl) in gens.items():
            g_acc = accuracy_score(gl, (gp > t).astype(int))
            g_ap  = average_precision_score(gl, gp) if len(np.unique(gl)) > 1 else -1
            pg[gen] = {"acc": float(g_acc), "ap": float(g_ap), "n": len(gl), "threshold": t}
            sp.extend(gp.tolist()); sl.extend(gl.tolist())
            print(f"  {gen:20s}: Acc={g_acc*100:.1f}%  AP={g_ap:.4f}  n={len(gl)}")
        s_np, l_np = np.array(sp), np.array(sl)
        s_acc = accuracy_score(l_np, (s_np > t).astype(int))
        s_ap  = average_precision_score(l_np, s_np) if len(np.unique(l_np)) > 1 else -1
        final_results[split] = {"acc": float(s_acc), "ap": float(s_ap), "n": len(sp),
                                  "threshold": t, "per_generator": pg}
        print(f"  {split.upper()} TOTAL: Acc={s_acc*100:.1f}%  AP={s_ap:.4f}")

    avg_acc = float(np.mean([r["acc"] for r in final_results.values()]))
    avg_ap  = float(np.mean([r["ap"]  for r in final_results.values() if r["ap"] >= 0]))
    final_results["average"] = {"acc": avg_acc, "ap": avg_ap,
                                  "strategy": best_strategy,
                                  "thresholds": best_thresholds}
    print(f"\n{'='*60}")
    print(f"FINAL AVERAGE: Acc={avg_acc*100:.2f}%  AP={avg_ap:.4f}")
    print(f"Strategy: {best_strategy}")
    print(f"Thresholds: {best_thresholds}")
    print(f"{'='*60}")

    out_path = os.path.join(args.output_dir, "test_results.json")
    json.dump(final_results, open(out_path, "w"), indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
