"""
Option 3: Per-generator threshold oracle.
Finds the best threshold per generator using Youden's J on each generator's
own confidence distribution — shows the theoretical accuracy ceiling.

Usage:
  python training/per_gen_threshold.py \
    --checkpoint checkpoints/v5_resume_hl/best_model.pth \
    --output_dir checkpoints/v5_pergen_threshold
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
    p.add_argument("--output_dir",  default="checkpoints/v5_pergen_threshold")
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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    print("\n" + "="*65)
    print("Per-Generator Oracle Threshold (Youden's J)")
    print("="*65)
    print(f"{'Generator':<22} {'Split':<5} {'Thresh':>7} {'Acc@0.5':>8} {'Acc@opt':>8} {'Gain':>7} {'n':>6}")
    print("-"*65)

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
            gp, gl = get_preds(model, loader, device)

            t_opt  = best_thresh_youden(gp, gl)
            acc_05 = accuracy_score(gl, (gp > 0.50).astype(int))
            acc_op = accuracy_score(gl, (gp > t_opt).astype(int))
            gain   = acc_op - acc_05
            g_ap   = average_precision_score(gl, gp) if len(np.unique(gl)) > 1 else -1

            print(f"  {gen:<22} {split:<5} {t_opt:>7.3f} {acc_05*100:>7.1f}% {acc_op*100:>7.1f}% {gain*100:>+6.1f}%  {len(ds):>5}")

            results[split]["per_generator"][gen] = {
                "acc_default": float(acc_05), "acc_optimal": float(acc_op),
                "threshold": float(t_opt), "ap": float(g_ap), "n": len(ds)
            }
            # Collect for split-level with per-gen thresholds
            split_preds_all[split]["p"].extend(gp.tolist())
            split_preds_all[split]["l"].extend(gl.tolist())
            # Apply per-gen threshold per sample
            split_preds_all[split]["t"].extend([(1 if c > t_opt else 0) for c in gp])

    # Split-level summary
    print("\n" + "="*65)
    print(f"{'Split':<8} {'Default(0.5)':>13} {'Per-Gen Opt':>13} {'Gain':>8}")
    print("-"*45)
    total_default, total_opt, total_n = 0.0, 0.0, 0
    for split in ["id", "cm", "cf", "cd"]:
        if split not in split_preds_all: continue
        d = split_preds_all[split]
        if not d["l"]: continue
        l_np = np.array(d["l"])
        p_np = np.array(d["p"])
        t_np = np.array(d["t"])
        acc_def = accuracy_score(l_np, (p_np > 0.5).astype(int))
        acc_opt = accuracy_score(l_np, t_np)
        n = len(l_np)
        print(f"  {split.upper():<6} {acc_def*100:>12.1f}% {acc_opt*100:>12.1f}% {(acc_opt-acc_def)*100:>+7.1f}%")
        results[split]["acc_default"]  = float(acc_def)
        results[split]["acc_optimal"]  = float(acc_opt)
        total_default += acc_def; total_opt += acc_opt; total_n += 1

    avg_def = total_default / max(total_n, 1)
    avg_opt = total_opt     / max(total_n, 1)
    print(f"  {'AVG':<6} {avg_def*100:>12.1f}% {avg_opt*100:>12.1f}% {(avg_opt-avg_def)*100:>+7.1f}%")
    print(f"\n*** ORACLE CEILING: {avg_opt*100:.2f}% ***")

    results["average"] = {"acc_default": float(avg_def), "acc_optimal": float(avg_opt)}
    json.dump(results, open(os.path.join(args.output_dir, "test_results.json"), "w"), indent=2)
    print(f"Saved → {args.output_dir}/test_results.json")


if __name__ == "__main__":
    main()
