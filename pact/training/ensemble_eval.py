"""
Ensemble Eval + Test-Time Augmentation (TTA) for XGenDet.
Loads N model checkpoints, averages confidence scores, evaluates on HydraFake test splits.
Supports: uniform ensemble, val-tuned weighted ensemble, TTA.

Usage:
  python training/ensemble_eval.py \\
    --checkpoints checkpoints/hydrafake_scratch/best_model.pth \\
                  checkpoints/hydrafake_finetune/best_model.pth \\
                  checkpoints/v3_resaug/best_model.pth \\
    --output_dir checkpoints/ensemble_v1 \\
    --tta
"""

import os, sys, json, argparse
from pathlib import Path
from itertools import product

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score
from tqdm import tqdm
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgendet import XGenDet
from data.hydrafake_dataset import HydraFakeDataset, HydraFakeTestDataset
from data.augmentations import get_eval_transforms, CLIP_MEAN, CLIP_STD


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--val_json", default="/home/sachin.chaudhary/hydrafake/jsons/val/all.json")
    p.add_argument("--test_dir", default="/home/sachin.chaudhary/hydrafake/jsons/test")
    p.add_argument("--data_root", default="/home/sachin.chaudhary")
    p.add_argument("--image_root", default="/home/sachin.chaudhary/hydrafake/test")
    p.add_argument("--output_dir", default="checkpoints/ensemble_v1")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    p.add_argument("--tune_weights", action="store_true",
                   help="Grid-search per-model weights on val set (default: uniform)")
    return p.parse_args()


def load_model(ckpt_path, device):
    model = XGenDet()
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model


def get_tta_transforms(crop_size):
    """5 TTA augmentations covering resolution and compression variation."""
    base_norm = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    return [
        # 1. Standard eval
        T.Compose([T.Resize((crop_size, crop_size)), T.ToTensor(), base_norm]),
        # 2. Horizontal flip
        T.Compose([T.Resize((crop_size, crop_size)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), base_norm]),
        # 3. JPEG quality 60 (in-the-wild compression)
        T.Compose([T.Resize((crop_size, crop_size)),
                   T.Lambda(lambda img: _jpeg_aug(img, 60)),
                   T.ToTensor(), base_norm]),
        # 4. Upsample path: 256→224 (simulates low-res generators like deepfacelab)
        T.Compose([T.Resize(256), T.CenterCrop(crop_size), T.ToTensor(), base_norm]),
        # 5. Downsample path: 336→224 (simulates high-res generators like ICLight)
        T.Compose([T.Resize(336), T.CenterCrop(crop_size), T.ToTensor(), base_norm]),
    ]


def _jpeg_aug(img, quality):
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@torch.no_grad()
def run_models_on_loader(models, loader, device, tta_transforms=None):
    """Run all models on loader. Returns (predictions, labels) where preds = avg confidence."""
    all_preds = [[] for _ in models]
    all_labels = []

    for batch in tqdm(loader, leave=False, ncols=100):
        imgs, labels = batch[0].to(device), batch[1]
        all_labels.extend(labels.tolist())

        # Standard inference for all models
        for mi, model in enumerate(models):
            out = model(imgs, return_heatmap=False)
            all_preds[mi].extend(out["confidence"].squeeze(-1).cpu().tolist())

    return all_preds, all_labels


@torch.no_grad()
def run_models_with_tta(models, image_paths, labels, tta_transforms, device, batch_size, num_workers):
    """Run all models with TTA on a list of image paths."""
    all_preds = np.zeros((len(models), len(image_paths)))  # [n_models, n_samples]

    for ti, tfm in enumerate(tta_transforms):
        # Build a simple path-based dataset for this TTA transform
        class PathDataset(torch.utils.data.Dataset):
            def __init__(self, paths, transform):
                self.paths = paths
                self.transform = transform
            def __len__(self): return len(self.paths)
            def __getitem__(self, i):
                try:
                    img = Image.open(self.paths[i]).convert("RGB")
                    return self.transform(img)
                except:
                    return torch.zeros(3, 224, 224)

        ds = PathDataset(image_paths, tfm)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

        offset = 0
        for imgs in loader:
            imgs = imgs.to(device)
            bs = imgs.shape[0]
            for mi, model in enumerate(models):
                out = model(imgs, return_heatmap=False)
                conf = out["confidence"].squeeze(-1).cpu().numpy()
                all_preds[mi, offset:offset+bs] += conf / len(tta_transforms)
            offset += bs

    return all_preds  # [n_models, n_samples]


def tune_weights_on_val(models, val_loader, device):
    """Grid-search model weights on val set. Returns best weights."""
    all_preds, all_labels = run_models_on_loader(models, val_loader, device)
    n = len(models)
    all_preds = np.array(all_preds)  # [n_models, n_samples]
    labels = np.array(all_labels)

    best_acc, best_weights = 0.0, None
    # Grid search over weights in steps of 0.1
    steps = [w / 10 for w in range(1, 10)]  # 0.1 to 0.9
    if n == 2:
        for w1 in steps:
            w2 = round(1.0 - w1, 1)
            if w2 <= 0: continue
            avg = w1 * all_preds[0] + w2 * all_preds[1]
            acc = accuracy_score(labels, (avg > 0.5).astype(int))
            if acc > best_acc:
                best_acc, best_weights = acc, [w1, w2]
    elif n == 3:
        for w1 in steps:
            for w2 in steps:
                w3 = round(1.0 - w1 - w2, 1)
                if w3 <= 0 or w3 > 1: continue
                avg = w1 * all_preds[0] + w2 * all_preds[1] + w3 * all_preds[2]
                acc = accuracy_score(labels, (avg > 0.5).astype(int))
                if acc > best_acc:
                    best_acc, best_weights = acc, [w1, w2, w3]
    else:
        best_weights = [1.0 / n] * n  # fallback: uniform

    uniform_avg = all_preds.mean(axis=0)
    uniform_acc = accuracy_score(labels, (uniform_avg > 0.5).astype(int))
    print(f"Val: uniform weights → Acc={uniform_acc*100:.2f}%")
    if best_weights:
        print(f"Val: best weights {best_weights} → Acc={best_acc*100:.2f}%")
    return best_weights or [1.0 / n] * n


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {len(args.checkpoints)} models...")
    models = [load_model(ckpt, device) for ckpt in args.checkpoints]
    for i, ckpt in enumerate(args.checkpoints):
        print(f"  [{i}] {ckpt}")

    tta_transforms = get_tta_transforms(args.crop_size) if args.tta else None

    # Optionally tune weights on val set
    if args.tune_weights:
        print("\nTuning weights on val set...")
        val_ds = HydraFakeDataset(args.val_json, args.data_root, is_train=False, crop_size=args.crop_size)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        model_weights = tune_weights_on_val(models, val_loader, device)
    else:
        model_weights = [1.0 / len(models)] * len(models)
        print(f"Using uniform weights: {model_weights}")

    # Evaluate on test splits
    print(f"\nEvaluating on test splits (TTA={args.tta})...")
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

            if args.tta:
                # Collect image paths and labels
                img_paths = [d[0] for d in ds.data]
                g_labels = [d[1] for d in ds.data]
                preds_matrix = run_models_with_tta(
                    models, img_paths, g_labels, tta_transforms, device,
                    args.batch_size, args.num_workers
                )
                # Weighted average across models
                g_preds = sum(w * preds_matrix[mi] for mi, w in enumerate(model_weights))
            else:
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
                all_m_preds, g_labels = run_models_on_loader(models, loader, device)
                g_preds = sum(w * np.array(p) for w, p in zip(model_weights, all_m_preds))

            g_labels_np = np.array(g_labels)
            g_preds_np = np.array(g_preds)
            g_acc = accuracy_score(g_labels_np, (g_preds_np > 0.5).astype(int))
            g_ap = average_precision_score(g_labels_np, g_preds_np) if len(np.unique(g_labels_np)) > 1 else -1
            per_gen[gen] = {"acc": float(g_acc), "ap": float(g_ap), "n": len(ds)}
            split_preds.extend(g_preds_np.tolist())
            split_labels.extend(g_labels_np.tolist())
            print(f"  {gen:20s}: Acc={g_acc*100:.1f}%, AP={g_ap:.4f}, n={len(ds)}")

        if split_preds:
            s_np, l_np = np.array(split_preds), np.array(split_labels)
            s_acc = accuracy_score(l_np, (s_np > 0.5).astype(int))
            s_ap = average_precision_score(l_np, s_np) if len(np.unique(l_np)) > 1 else -1
            all_results[split] = {"acc": float(s_acc), "ap": float(s_ap), "n": len(split_preds),
                                  "per_generator": per_gen}
            print(f"  {split.upper()} TOTAL: Acc={s_acc*100:.1f}%, AP={s_ap:.4f}")

    if all_results:
        avg_acc = float(np.mean([r["acc"] for r in all_results.values()]))
        avg_ap = float(np.mean([r["ap"] for r in all_results.values() if r["ap"] >= 0]))
        all_results["average"] = {"acc": avg_acc, "ap": avg_ap}
        print(f"\n{'='*60}")
        print(f"ENSEMBLE AVERAGE: Acc={avg_acc*100:.1f}%, AP={avg_ap:.4f}")
        print(f"{'='*60}")

    # Save
    out_path = os.path.join(args.output_dir, "test_results.json")
    json.dump(all_results, open(out_path, "w"), indent=2)
    meta = {"checkpoints": args.checkpoints, "weights": model_weights, "tta": args.tta}
    json.dump(meta, open(os.path.join(args.output_dir, "config.json"), "w"), indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
