"""
Temperature calibration script.

Finds optimal temperature T for binary_logit / T → confidence.
Temperature is optimized on val set to minimize NLL, then applied to test.

Usage:
  python training/temperature_calibration.py \
    --checkpoint checkpoints/v5_resume_hl/best_model.pth \
    --output_dir checkpoints/v5_temp_calibrated
"""
import os, sys, json, argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score, roc_curve
from tqdm import tqdm
from scipy.optimize import minimize_scalar

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
    p.add_argument("--output_dir",  default="checkpoints/v5_temp_calibrated")
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
def get_logits(model, loader, device):
    """Collect raw binary_logits (before temperature) and labels."""
    logits, labels = [], []
    for batch in loader:
        imgs = batch[0].to(device)
        lab  = batch[1]
        # Temporarily set temperature to 1.0 to get raw logits
        with torch.no_grad():
            orig_temp = model.module.classification_head.temperature.data.clone() \
                if hasattr(model, 'module') else model.classification_head.temperature.data.clone()
            if hasattr(model, 'module'):
                model.module.classification_head.temperature.data.fill_(1.0)
            else:
                model.classification_head.temperature.data.fill_(1.0)
            out = model(imgs, return_heatmap=False)
            logits.extend(out["binary_logit"].squeeze(-1).cpu().tolist())
            labels.extend(lab.tolist())
            # Restore temperature
            if hasattr(model, 'module'):
                model.module.classification_head.temperature.data.copy_(orig_temp)
            else:
                model.classification_head.temperature.data.copy_(orig_temp)
    return np.array(logits), np.array(labels)


@torch.no_grad()
def get_preds_with_temp(model, loader, device, temperature):
    """Get confidence predictions using a specific temperature value."""
    preds, labels = [], []
    for batch in loader:
        imgs = batch[0].to(device)
        lab  = batch[1]
        out  = model(imgs, return_heatmap=False)
        preds.extend(out["confidence"].squeeze(-1).cpu().tolist())
        labels.extend(lab.tolist())
    return np.array(preds), np.array(labels)


def nll_temperature(T, logits, labels):
    """Negative log-likelihood as a function of temperature."""
    T = max(T, 1e-3)
    probs = 1.0 / (1.0 + np.exp(-logits / T))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return float(nll)


def best_thresh_youden(preds, labels):
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresh = roc_curve(labels, preds)
    idx = np.argmax(tpr - fpr)
    return float(thresh[idx])


def set_temperature(model, T):
    if hasattr(model, 'module'):
        model.module.classification_head.temperature.data.fill_(T)
    else:
        model.classification_head.temperature.data.fill_(T)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # ── Step 1: Collect val logits ──────────────────────────────────────────
    print("\n" + "="*60)
    print("Step 1: Collecting val set logits (T=1.0)")
    print("="*60)
    val_ds = HydraFakeDataset(args.val_json, args.data_root, is_train=False,
                               crop_size=args.crop_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    val_logits, val_labels = get_logits(model, tqdm(val_loader, desc="Val logits", ncols=100), device)

    print(f"  Val samples: {len(val_labels)}, pos rate: {val_labels.mean():.3f}")
    print(f"  Logit range: [{val_logits.min():.3f}, {val_logits.max():.3f}]  mean={val_logits.mean():.3f}")

    # ── Step 2: Optimize temperature ────────────────────────────────────────
    print("\n" + "="*60)
    print("Step 2: Optimizing temperature T via NLL minimization")
    print("="*60)

    # Try a grid first to visualize
    print(f"\n{'T':>6}  {'NLL':>8}  {'Acc@0.5':>8}  {'Acc@Youden':>10}")
    print("-"*40)
    grid_results = []
    for T in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        nll = nll_temperature(T, val_logits, val_labels)
        probs = 1.0 / (1.0 + np.exp(-val_logits / T))
        acc_05 = accuracy_score(val_labels, (probs > 0.5).astype(int))
        t_opt  = best_thresh_youden(probs, val_labels)
        acc_yd = accuracy_score(val_labels, (probs > t_opt).astype(int))
        print(f"  {T:>4.1f}  {nll:>8.5f}  {acc_05*100:>7.2f}%  {acc_yd*100:>9.2f}%")
        grid_results.append((T, nll, acc_05, acc_yd))

    # Fine optimization
    result = minimize_scalar(
        lambda T: nll_temperature(T, val_logits, val_labels),
        bounds=(0.1, 10.0), method="bounded"
    )
    T_opt = result.x
    nll_opt = result.fun
    print(f"\n  Optimal T = {T_opt:.4f}  NLL = {nll_opt:.5f}")

    # Evaluate calibrated model on val
    probs_cal = 1.0 / (1.0 + np.exp(-val_logits / T_opt))
    acc_cal_05 = accuracy_score(val_labels, (probs_cal > 0.5).astype(int))
    t_cal_opt  = best_thresh_youden(probs_cal, val_labels)
    acc_cal_yd = accuracy_score(val_labels, (probs_cal > t_cal_opt).astype(int))

    probs_orig = 1.0 / (1.0 + np.exp(-val_logits))
    acc_orig_05 = accuracy_score(val_labels, (probs_orig > 0.5).astype(int))
    t_orig_opt  = best_thresh_youden(probs_orig, val_labels)
    acc_orig_yd = accuracy_score(val_labels, (probs_orig > t_orig_opt).astype(int))

    print(f"\n  Original  (T=1.00):  Acc@0.5={acc_orig_05*100:.2f}%  Acc@Youden={acc_orig_yd*100:.2f}%  YJ-thresh={t_orig_opt:.3f}")
    print(f"  Calibrated(T={T_opt:.2f}): Acc@0.5={acc_cal_05*100:.2f}%  Acc@Youden={acc_cal_yd*100:.2f}%  YJ-thresh={t_cal_opt:.3f}")

    # ── Step 3: Apply calibrated T to test ──────────────────────────────────
    print("\n" + "="*60)
    print(f"Step 3: Test evaluation with T={T_opt:.4f}")
    print("="*60)
    print(f"{'Generator':<22} {'Split':<5} {'Acc@0.5':>8} {'Acc@Youden':>11} {'AP':>7} {'n':>6}")
    print("-"*65)

    set_temperature(model, T_opt)

    results = {}
    split_preds_all = {s: {"p":[], "l":[], "t":[]} for s in ["id","cm","cf","cd"]}

    for split in ["id", "cm", "cf", "cd"]:
        split_dir = os.path.join(args.test_dir, split)
        if not os.path.isdir(split_dir):
            continue
        results[split] = {"per_generator": {}}

        for jf in sorted(f for f in os.listdir(split_dir) if f.endswith(".json")):
            gen = jf.replace(".json", "")
            ds  = HydraFakeTestDataset(os.path.join(split_dir, jf), args.data_root,
                                        args.image_root, args.crop_size)
            if len(ds) == 0: continue
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            gp, gl = get_preds_with_temp(model, loader, device, T_opt)

            t_gen_opt = best_thresh_youden(gp, gl)
            acc_05    = accuracy_score(gl, (gp > 0.50).astype(int))
            acc_yd    = accuracy_score(gl, (gp > t_gen_opt).astype(int))
            g_ap      = average_precision_score(gl, gp) if len(np.unique(gl)) > 1 else -1

            print(f"  {gen:<22} {split:<5} {acc_05*100:>7.1f}% {acc_yd*100:>10.1f}% {g_ap:>7.4f} {len(ds):>5}")

            results[split]["per_generator"][gen] = {
                "acc_default": float(acc_05), "acc_youden": float(acc_yd),
                "threshold_youden": float(t_gen_opt), "ap": float(g_ap), "n": len(ds)
            }
            split_preds_all[split]["p"].extend(gp.tolist())
            split_preds_all[split]["l"].extend(gl.tolist())
            split_preds_all[split]["t"].extend([(1 if c > t_gen_opt else 0) for c in gp])

    # Split-level summary (using 0.5 threshold on calibrated confidences)
    print("\n" + "="*60)
    print("Split-level summary (threshold=0.5 on calibrated confidences)")
    print(f"{'Split':<8} {'Acc@0.5':>10} {'Acc@Youden':>12}")
    print("-"*35)
    total_acc, total_n = 0.0, 0
    split_summary = {}
    for split in ["id", "cm", "cf", "cd"]:
        d = split_preds_all.get(split, {})
        if not d.get("l"): continue
        l_np = np.array(d["l"]); p_np = np.array(d["p"]); t_np = np.array(d["t"])
        acc_05 = accuracy_score(l_np, (p_np > 0.5).astype(int))
        acc_yd = accuracy_score(l_np, t_np)
        n = len(l_np)
        print(f"  {split.upper():<6} {acc_05*100:>9.1f}% {acc_yd*100:>11.1f}%")
        results[split]["acc_05"]    = float(acc_05)
        results[split]["acc_youden"] = float(acc_yd)
        total_acc += acc_05; total_n += 1
        split_summary[split] = acc_05

    avg_acc = total_acc / max(total_n, 1)
    print(f"  {'AVG':<6} {avg_acc*100:>9.1f}%")
    print(f"\n*** CALIBRATED (T={T_opt:.4f}) AVERAGE: {avg_acc*100:.2f}% ***")

    # Also try per-split Youden thresholds on calibrated confidences
    print("\n" + "="*60)
    print("Bonus: per-split Youden thresholds on calibrated confidences")
    print("="*60)
    total_persp = 0.0; total_persp_n = 0
    for split in ["id", "cm", "cf", "cd"]:
        d = split_preds_all.get(split, {})
        if not d.get("l"): continue
        l_np = np.array(d["l"]); p_np = np.array(d["p"])
        t_sp = best_thresh_youden(p_np, l_np)
        acc_sp = accuracy_score(l_np, (p_np > t_sp).astype(int))
        print(f"  {split.upper()}: T_youden={t_sp:.3f}  Acc={acc_sp*100:.2f}%")
        results[split]["acc_per_split_youden"] = float(acc_sp)
        results[split]["threshold_per_split"] = float(t_sp)
        total_persp += acc_sp; total_persp_n += 1
    print(f"  AVERAGE with per-split Youden: {total_persp/max(total_persp_n,1)*100:.2f}%")

    results["calibration"] = {
        "temperature": float(T_opt),
        "nll_before": float(nll_temperature(1.0, val_logits, val_labels)),
        "nll_after":  float(nll_opt),
        "avg_acc_05": float(avg_acc)
    }
    json.dump(results, open(os.path.join(args.output_dir, "test_results.json"), "w"), indent=2)
    print(f"\nSaved → {args.output_dir}/test_results.json")


if __name__ == "__main__":
    main()
