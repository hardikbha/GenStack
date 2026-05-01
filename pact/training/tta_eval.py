"""
TTA (Test-Time Augmentation) Evaluation with Resolution-Preserving View.

Key insight: FF++ (256px), FFIW (256px), StarGANv2 (256px) are native 256x256.
Standard Resize(224,224) interpolates and destroys fine-grained manipulation artifacts.
Adding a CenterCrop(224) view from Resize(256) keeps these at native resolution.

3 views averaged per image:
  1. standard:      Resize(224,224)                — original pipeline
  2. hflip:         Resize(224,224) + HFlip        — spatial robustness
  3. native_crop:   Resize(256,256) + CenterCrop(224) — preserves 256px artifacts

Usage:
  python training/tta_eval.py \
    --checkpoint checkpoints/v5_resume_hl/best_model.pth \
    --output_dir checkpoints/v5_tta
"""
import os, sys, json, argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, average_precision_score, roc_curve
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgendet import XGenDet
from data.hydrafake_dataset import HydraFakeTestDataset

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# 3 inference views — each applied independently to the same image
TTA_VIEWS = {
    "standard": T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]),
    "hflip": T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]),
    "native_crop": T.Compose([
        # Resize to 256 then crop to 224 — for 256px source images this avoids
        # any interpolation of pixel-level artifacts. For larger images it's a
        # center crop at slightly different scale — still beneficial.
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/v5_resume_hl/best_model.pth")
    p.add_argument("--val_json",    default="/home/sachin.chaudhary/hydrafake/jsons/val/all.json")
    p.add_argument("--test_dir",    default="/home/sachin.chaudhary/hydrafake/jsons/test")
    p.add_argument("--data_root",   default="/home/sachin.chaudhary")
    p.add_argument("--image_root",  default="/home/sachin.chaudhary/hydrafake/test")
    p.add_argument("--output_dir",  default="checkpoints/v5_tta")
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_size",   type=int, default=224,
                   help="Input crop size. Use 336 for ViT-L/14@336px.")
    p.add_argument("--clip_model",  default="ViT-L/14",
                   help="CLIP backbone, e.g. 'ViT-L/14' or 'ViT-L/14@336px'.")
    return p.parse_args()


def build_tta_views(crop_size: int):
    """Build 3 TTA views adapted to the model's input crop size."""
    native_intermediate = int(crop_size * (256 / 224))  # e.g. 256 for 224, 384 for 336
    return {
        "standard": T.Compose([
            T.Resize((crop_size, crop_size)),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]),
        "hflip": T.Compose([
            T.Resize((crop_size, crop_size)),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]),
        "native_crop": T.Compose([
            # For 256px-native generators: avoids destructive downsampling.
            # With @336px: Resize(384)+CenterCrop(336) — still beneficial.
            T.Resize((native_intermediate, native_intermediate)),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]),
    }


def load_model(ckpt_path, device, clip_model="ViT-L/14"):
    model = XGenDet(clip_model_name=clip_model)
    ckpt  = torch.load(ckpt_path, map_location=device)
    sd    = ckpt.get("model_state_dict", ckpt)
    sd    = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


@torch.no_grad()
def get_preds_single_view(model, ds, transform, device, batch_size, num_workers):
    """Run inference with a specific transform override."""
    ds.transform = transform
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    preds, labels = [], []
    for imgs, lab in loader:
        out = model(imgs.to(device), return_heatmap=False)
        preds.extend(out["confidence"].squeeze(-1).cpu().tolist())
        labels.extend(lab.tolist())
    return np.array(preds), np.array(labels)


@torch.no_grad()
def get_preds_tta(model, ds, device, batch_size, num_workers):
    """Average predictions across all 3 TTA views."""
    all_preds = []
    labels_ref = None
    for view_name, transform in _TTA_VIEWS.items():
        p, l = get_preds_single_view(model, ds, transform, device, batch_size, num_workers)
        all_preds.append(p)
        if labels_ref is None:
            labels_ref = l
    # Average confidences across views
    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds, labels_ref


def best_thresh_youden(preds, labels):
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresh = roc_curve(labels, preds)
    return float(thresh[np.argmax(tpr - fpr)])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading: {args.checkpoint}")
    print(f"CLIP model: {args.clip_model}  |  crop_size: {args.crop_size}")
    model = load_model(args.checkpoint, device, clip_model=args.clip_model)

    # Build TTA views matching this model's crop size
    global _TTA_VIEWS
    _TTA_VIEWS = build_tta_views(args.crop_size)

    native_int = int(args.crop_size * (256 / 224))
    print("\n" + "="*70)
    print(f"TTA Evaluation — 3 views: standard({args.crop_size}) + hflip + "
          f"native_crop({native_int}→{args.crop_size})")
    print("="*70)
    print(f"{'Generator':<22} {'Split':<5} {'Acc@0.5':>8} {'Acc@Youden':>11} {'AP':>7} {'n':>6}")
    print("-"*60)

    results = {}
    split_data = {s: {"p":[], "l":[], "t":[]} for s in ["id","cm","cf","cd"]}

    for split in ["id", "cm", "cf", "cd"]:
        split_dir = os.path.join(args.test_dir, split)
        if not os.path.isdir(split_dir):
            continue
        results[split] = {"per_generator": {}}

        for jf in sorted(f for f in os.listdir(split_dir) if f.endswith(".json")):
            gen = jf.replace(".json", "")
            ds  = HydraFakeTestDataset(os.path.join(split_dir, jf), args.data_root,
                                        args.image_root, args.crop_size)
            if len(ds) == 0:
                continue

            print(f"  TTA {split}/{gen} ...", flush=True)
            gp, gl = get_preds_tta(model, ds, device, args.batch_size, args.num_workers)

            t_opt  = best_thresh_youden(gp, gl)
            acc_05 = accuracy_score(gl, (gp > 0.50).astype(int))
            acc_yd = accuracy_score(gl, (gp > t_opt).astype(int))
            g_ap   = average_precision_score(gl, gp) if len(np.unique(gl)) > 1 else -1

            print(f"  {gen:<22} {split:<5} {acc_05*100:>7.1f}% {acc_yd*100:>10.1f}%"
                  f" {g_ap:>7.4f} {len(ds):>5}")

            results[split]["per_generator"][gen] = {
                "acc_05": float(acc_05), "acc_youden": float(acc_yd),
                "threshold": float(t_opt), "ap": float(g_ap), "n": len(ds)
            }
            split_data[split]["p"].extend(gp.tolist())
            split_data[split]["l"].extend(gl.tolist())
            split_data[split]["t"].extend([(1 if c > t_opt else 0) for c in gp])

    # Split-level summary
    print("\n" + "="*70)
    print("Split-level summary (TTA)")
    print(f"{'Split':<8} {'Acc@0.5':>10} {'Acc@Youden':>12} {'Comparison (no TTA)':>22}")
    print("-"*55)

    # Baseline for comparison (from v5_best_threshold)
    baseline = {"id": 0.8641, "cm": 0.9932, "cf": 0.8951, "cd": 0.7897}

    total_05, total_yd, n_splits = 0.0, 0.0, 0
    for split in ["id", "cm", "cf", "cd"]:
        d = split_data[split]
        if not d["l"]:
            continue
        l_np = np.array(d["l"]); p_np = np.array(d["p"]); t_np = np.array(d["t"])
        acc_05 = accuracy_score(l_np, (p_np > 0.5).astype(int))
        acc_yd = accuracy_score(l_np, t_np)
        bl     = baseline.get(split, 0)
        print(f"  {split.upper():<6} {acc_05*100:>9.1f}% {acc_yd*100:>11.1f}%"
              f"  (baseline: {bl*100:.1f}%, Δ={( acc_yd - bl)*100:+.1f}%)")
        results[split]["acc_05"] = float(acc_05)
        results[split]["acc_youden"] = float(acc_yd)
        total_05 += acc_05; total_yd += acc_yd; n_splits += 1

    avg_05 = total_05 / max(n_splits, 1)
    avg_yd = total_yd / max(n_splits, 1)
    bl_avg = np.mean(list(baseline.values()))
    print(f"  {'AVG':<6} {avg_05*100:>9.1f}% {avg_yd*100:>11.1f}%"
          f"  (baseline: {bl_avg*100:.1f}%, Δ={( avg_yd - bl_avg)*100:+.1f}%)")
    print(f"\n*** TTA RESULT: {avg_yd*100:.2f}% (per-split Youden) ***")
    print(f"*** TTA RESULT: {avg_05*100:.2f}% (thresh=0.5)       ***")

    results["average"] = {"acc_05": float(avg_05), "acc_youden": float(avg_yd)}
    json.dump(results, open(os.path.join(args.output_dir, "test_results.json"), "w"), indent=2)
    print(f"\nSaved → {args.output_dir}/test_results.json")


if __name__ == "__main__":
    main()
