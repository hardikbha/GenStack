"""
CD-split threshold sweep.

Sweeps threshold specifically for CD split (deepfakes: FFIW, deepfacelab, hailuo, etc.)
and shows the detailed accuracy/FPR/FNR tradeoff per generator.

Key insight: FFIW real images have conf ~0.28, so lowering threshold below 0.37
makes things worse (more FP). This script shows the full picture so we can
make an informed decision.

Usage:
  python training/cd_threshold_sweep.py \
    --checkpoint checkpoints/v5_resume_hl/best_model.pth \
    --output_dir checkpoints/v5_cd_threshold
"""
import os, sys, json, argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score, roc_curve
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgendet import XGenDet
from data.hydrafake_dataset import HydraFakeTestDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/v5_resume_hl/best_model.pth")
    p.add_argument("--test_dir",    default="/home/sachin.chaudhary/hydrafake/jsons/test")
    p.add_argument("--data_root",   default="/home/sachin.chaudhary")
    p.add_argument("--image_root",  default="/home/sachin.chaudhary/hydrafake/test")
    p.add_argument("--output_dir",  default="checkpoints/v5_cd_threshold")
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_size",   type=int, default=224)
    return p.parse_args()


def load_model(ckpt_path, device):
    model = XGenDet()
    ckpt  = torch.load(ckpt_path, map_location=device)
    sd    = ckpt.get("model_state_dict", ckpt)
    sd    = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


@torch.no_grad()
def get_preds(model, loader, device):
    preds, labels = [], []
    for imgs, lab in loader:
        out = model(imgs.to(device), return_heatmap=False)
        preds.extend(out["confidence"].squeeze(-1).cpu().tolist())
        labels.extend(lab.tolist())
    return np.array(preds), np.array(labels)


def best_thresh_youden(preds, labels):
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresh = roc_curve(labels, preds)
    idx = np.argmax(tpr - fpr)
    return float(thresh[idx])


def conf_distribution_stats(preds, labels, name):
    """Print confidence distribution stats for reals and fakes."""
    real_conf = preds[labels == 0]
    fake_conf = preds[labels == 1]
    if len(real_conf) == 0 or len(fake_conf) == 0:
        return
    print(f"    {name}:")
    print(f"      Real conf: mean={real_conf.mean():.3f}  p10={np.percentile(real_conf,10):.3f}  "
          f"p50={np.percentile(real_conf,50):.3f}  p90={np.percentile(real_conf,90):.3f}")
    print(f"      Fake conf: mean={fake_conf.mean():.3f}  p10={np.percentile(fake_conf,10):.3f}  "
          f"p50={np.percentile(fake_conf,50):.3f}  p90={np.percentile(fake_conf,90):.3f}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # ── Collect CD predictions ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("Collecting CD split predictions")
    print("="*65)

    cd_dir = os.path.join(args.test_dir, "cd")
    gen_preds = {}  # gen_name → (preds, labels)

    for jf in sorted(f for f in os.listdir(cd_dir) if f.endswith(".json")):
        gen = jf.replace(".json", "")
        ds  = HydraFakeTestDataset(os.path.join(cd_dir, jf), args.data_root,
                                    args.image_root, args.crop_size)
        if len(ds) == 0: continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        gp, gl = get_preds(model, tqdm(loader, desc=f"  cd/{gen}", ncols=100, leave=False), device)
        gen_preds[gen] = (gp, gl)
        print(f"  {gen:20s}: n={len(gl):5d}  n_real={int((gl==0).sum()):4d}  n_fake={int((gl==1).sum()):4d}  "
              f"AP={average_precision_score(gl,gp) if len(np.unique(gl))>1 else -1:.4f}")

    # ── Confidence distribution analysis ────────────────────────────────────
    print("\n" + "="*65)
    print("Confidence distributions per generator")
    print("="*65)
    for gen, (gp, gl) in gen_preds.items():
        conf_distribution_stats(gp, gl, gen)

    # ── Threshold sweep ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("Threshold sweep for CD split")
    print("="*65)

    # Collect all CD preds
    all_p = np.concatenate([v[0] for v in gen_preds.values()])
    all_l = np.concatenate([v[1] for v in gen_preds.values()])

    thresholds = np.linspace(0.20, 0.60, 41)
    t_youden   = best_thresh_youden(all_p, all_l)

    print(f"\n{'Thresh':>7}  {'CD Acc':>8}", end="")
    for gen in gen_preds:
        print(f"  {gen[:12]:>12}", end="")
    print()
    print("-" * (10 + 15 * len(gen_preds)))

    best_acc, best_t, best_row = 0.0, 0.5, ""
    results_sweep = []
    for t in thresholds:
        cd_acc = accuracy_score(all_l, (all_p > t).astype(int))
        row    = f"  {t:>5.2f}  {cd_acc*100:>7.1f}%"
        gen_accs = {}
        for gen, (gp, gl) in gen_preds.items():
            g_acc = accuracy_score(gl, (gp > t).astype(int))
            row  += f"  {g_acc*100:>11.1f}%"
            gen_accs[gen] = float(g_acc)
        marker = " ← Youden" if abs(t - t_youden) < 0.015 else ""
        if t == thresholds[0] or t == thresholds[-1] or abs(t - 0.5) < 0.01 or abs(t - t_youden) < 0.015 or \
           abs(t - 0.25) < 0.01 or abs(t - 0.30) < 0.01 or abs(t - 0.35) < 0.01 or abs(t - 0.40) < 0.01 or \
           abs(t - 0.45) < 0.01:
            print(row + marker)
        if cd_acc > best_acc:
            best_acc, best_t = cd_acc, float(t)
            best_row = row
        results_sweep.append({"threshold": float(t), "cd_acc": float(cd_acc), "per_gen": gen_accs})

    print(f"\n  Youden J optimal threshold: {t_youden:.3f}")
    print(f"  Grid-search best threshold: {best_t:.3f}  CD Acc={best_acc*100:.2f}%")

    # ── Per-generator Youden thresholds ──────────────────────────────────────
    print("\n" + "="*65)
    print("Per-generator optimal thresholds (oracle)")
    print("="*65)
    print(f"{'Generator':<22} {'Thresh':>7} {'Acc@0.5':>8} {'Acc@opt':>8} {'Gain':>7}")
    print("-"*55)
    oracle_results = {}
    for gen, (gp, gl) in gen_preds.items():
        t_opt  = best_thresh_youden(gp, gl)
        acc_05 = accuracy_score(gl, (gp > 0.50).astype(int))
        acc_op = accuracy_score(gl, (gp > t_opt).astype(int))
        g_ap   = average_precision_score(gl, gp) if len(np.unique(gl)) > 1 else -1
        print(f"  {gen:<22} {t_opt:>7.3f} {acc_05*100:>7.1f}% {acc_op*100:>7.1f}% {(acc_op-acc_05)*100:>+6.1f}%")
        oracle_results[gen] = {"threshold": float(t_opt), "acc_default": float(acc_05),
                                "acc_optimal": float(acc_op), "ap": float(g_ap), "n": int(len(gl))}

    # Oracle CD ceiling
    oracle_labels = np.concatenate([v[1] for v in gen_preds.values()])
    oracle_preds  = np.concatenate([(gp > best_thresh_youden(gp, gl)).astype(int)
                                     for gen, (gp, gl) in gen_preds.items()])
    oracle_acc = accuracy_score(oracle_labels, oracle_preds)
    print(f"\n  CD oracle ceiling (per-gen thresholds): {oracle_acc*100:.2f}%")
    print(f"  Current CD (thresh=0.372):              {accuracy_score(all_l, (all_p > 0.372).astype(int))*100:.2f}%")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "cd_threshold_sweep": results_sweep,
        "best_threshold":  best_t,
        "best_cd_acc":     float(best_acc),
        "youden_threshold": float(t_youden),
        "oracle_per_gen":  oracle_results,
        "oracle_cd_acc":   float(oracle_acc)
    }
    json.dump(output, open(os.path.join(args.output_dir, "cd_sweep_results.json"), "w"), indent=2)
    print(f"\nSaved → {args.output_dir}/cd_sweep_results.json")


if __name__ == "__main__":
    main()
