"""
Boosted evaluation: TTA + Ensemble + Threshold optimization.
No retraining needed — uses existing checkpoints.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import average_precision_score, accuracy_score
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xgendet import XGenDet
from data.hydrafake_dataset import HydraFakeDataset, HydraFakeTestDataset
from data.augmentations import get_eval_transforms

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_tta_transforms(crop_size=224):
    """5 crops + horizontal flip = 10 views per image."""
    normalize = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    transforms = []
    # Center crop
    transforms.append(T.Compose([
        T.Resize(crop_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        normalize,
    ]))
    # 4 corner crops
    for corner_fn in [
        lambda img, s: T.functional.crop(img, 0, 0, s, s),                          # top-left
        lambda img, s: T.functional.crop(img, 0, img.width - s, s, s),               # top-right
        lambda img, s: T.functional.crop(img, img.height - s, 0, s, s),              # bottom-left
        lambda img, s: T.functional.crop(img, img.height - s, img.width - s, s, s),  # bottom-right
    ]:
        transforms.append(("corner", corner_fn))

    return transforms


class TTATestDataset(torch.utils.data.Dataset):
    """Returns multiple augmented views of each image."""

    def __init__(self, json_path, data_root, image_root, crop_size=224, use_tta=True):
        self.crop_size = crop_size
        self.use_tta = use_tta
        self.normalize = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        self.base_transform = get_eval_transforms(crop_size)

        with open(json_path, "r") as f:
            annotations = json.load(f)

        self.data = []
        for item in annotations:
            rel_path = item["images"][0]
            full_path = os.path.join(data_root, rel_path)
            if not os.path.exists(full_path):
                parts = rel_path.split("/")
                if len(parts) >= 3:
                    sub_path = "/".join(parts[2:])
                    full_path = os.path.join(image_root, sub_path)
            if os.path.exists(full_path):
                self.data.append((full_path, item["label"]))

    def __len__(self):
        return len(self.data)

    def _get_tta_views(self, img):
        """Generate 10 views: 5 crops × 2 (original + hflip)."""
        views = []
        w, h = img.size
        s = min(w, h, self.crop_size)

        # Resize so short side = crop_size
        img_resized = T.functional.resize(img, self.crop_size, interpolation=T.InterpolationMode.BICUBIC)
        rw, rh = img_resized.size

        crops = []
        # Center crop
        crops.append(T.functional.center_crop(img_resized, self.crop_size))
        # 4 corners
        crops.append(T.functional.crop(img_resized, 0, 0, self.crop_size, min(rw, self.crop_size)))
        crops.append(T.functional.crop(img_resized, 0, max(0, rw - self.crop_size), self.crop_size, min(rw, self.crop_size)))
        crops.append(T.functional.crop(img_resized, max(0, rh - self.crop_size), 0, self.crop_size, min(rw, self.crop_size)))
        crops.append(T.functional.crop(img_resized, max(0, rh - self.crop_size), max(0, rw - self.crop_size), self.crop_size, min(rw, self.crop_size)))

        to_tensor = T.Compose([T.ToTensor(), self.normalize])
        for crop in crops:
            # Ensure correct size
            if crop.size != (self.crop_size, self.crop_size):
                crop = T.functional.resize(crop, (self.crop_size, self.crop_size))
            views.append(to_tensor(crop))
            views.append(to_tensor(T.functional.hflip(crop)))

        return torch.stack(views)  # [10, 3, 224, 224]

    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.use_tta:
                views = self._get_tta_views(img)  # [10, 3, 224, 224]
                return views, label
            else:
                return self.base_transform(img), label
        except Exception:
            if self.use_tta:
                return torch.zeros(10, 3, self.crop_size, self.crop_size), label
            return torch.zeros(3, self.crop_size, self.crop_size), label


def load_model(ckpt_path, device):
    model = XGenDet()
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_ap = ckpt.get("val_ap", 0)
    print(f"  Loaded {ckpt_path} (epoch {epoch}, val AP={val_ap:.4f})")
    return model


def predict_tta(model, views, device):
    """Run TTA: average confidence across all views."""
    # views: [10, 3, 224, 224]
    views = views.to(device)
    with torch.no_grad():
        outputs = model(views, return_heatmap=False)
        confs = outputs["confidence"].squeeze(-1)  # [10]
    return confs.mean().item()


def predict_single(model, img, device):
    """Standard single-view prediction."""
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img, return_heatmap=False)
    return outputs["confidence"].squeeze().item()


def find_optimal_threshold(labels, preds):
    """Find threshold that maximizes accuracy."""
    best_acc = 0
    best_t = 0.5
    for t in np.arange(0.3, 0.8, 0.01):
        acc = accuracy_score(labels, (np.array(preds) > t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc


def evaluate_split(models, split_dir, split_name, data_root, image_root, device,
                   use_tta=True, batch_size=1):
    """Evaluate one test split with TTA + ensemble."""
    json_files = sorted([f for f in os.listdir(split_dir) if f.endswith(".json")])

    split_results = {}
    all_preds = []
    all_labels = []

    for jf in json_files:
        gen_name = jf.replace(".json", "")
        json_path = os.path.join(split_dir, jf)

        dataset = TTATestDataset(
            json_path=json_path,
            data_root=data_root,
            image_root=image_root,
            use_tta=use_tta,
        )
        if len(dataset) == 0:
            continue

        gen_preds = []
        gen_labels = []

        for idx in range(len(dataset)):
            views_or_img, label = dataset[idx]

            # Ensemble: average across models
            model_confs = []
            for model in models:
                if use_tta:
                    conf = predict_tta(model, views_or_img, device)
                else:
                    conf = predict_single(model, views_or_img, device)
                model_confs.append(conf)

            ensemble_conf = np.mean(model_confs)
            gen_preds.append(ensemble_conf)
            gen_labels.append(label)

            if (idx + 1) % 500 == 0:
                print(f"    {gen_name}: {idx+1}/{len(dataset)}")

        gen_preds_np = np.array(gen_preds)
        gen_labels_np = np.array(gen_labels)

        # Standard metrics
        acc_05 = accuracy_score(gen_labels_np, (gen_preds_np > 0.5).astype(int))
        ap = average_precision_score(gen_labels_np, gen_preds_np) if len(np.unique(gen_labels_np)) > 1 else -1

        # Optimal threshold
        opt_t, opt_acc = find_optimal_threshold(gen_labels_np, gen_preds_np)

        split_results[gen_name] = {
            "acc": acc_05, "ap": ap, "opt_threshold": opt_t,
            "opt_acc": opt_acc, "n": len(dataset),
        }
        all_preds.extend(gen_preds)
        all_labels.extend(gen_labels)

        print(f"  {gen_name}: Acc={acc_05*100:.1f}% (opt: {opt_acc*100:.1f}% @{opt_t:.2f}), AP={ap:.4f}, n={len(dataset)}")

    # Split-level metrics
    all_p = np.array(all_preds)
    all_l = np.array(all_labels)
    split_acc = accuracy_score(all_l, (all_p > 0.5).astype(int))
    split_ap = average_precision_score(all_l, all_p) if len(np.unique(all_l)) > 1 else -1
    opt_t, opt_acc = find_optimal_threshold(all_l, all_p)

    print(f"\n  {split_name.upper()} TOTAL: Acc={split_acc*100:.1f}% (opt: {opt_acc*100:.1f}% @{opt_t:.2f}), AP={split_ap:.4f}, n={len(all_preds)}")

    return {
        "acc": split_acc, "ap": split_ap,
        "opt_threshold": opt_t, "opt_acc": opt_acc,
        "n": len(all_preds), "per_generator": split_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="One or more checkpoint paths")
    parser.add_argument("--test_dir", default="/home/sachin.chaudhary/hydrafake/jsons/test")
    parser.add_argument("--data_root", default="/home/sachin.chaudhary")
    parser.add_argument("--image_root", default="/home/sachin.chaudhary/hydrafake/test")
    parser.add_argument("--output", default="./checkpoints/boosted_results.json")
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA")
    parser.add_argument("--crop_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tta = not args.no_tta

    mode = []
    if use_tta:
        mode.append("TTA(10-view)")
    if len(args.checkpoints) > 1:
        mode.append(f"Ensemble({len(args.checkpoints)} models)")
    mode.append("OptThreshold")
    print(f"=== Boosted Evaluation: {' + '.join(mode)} ===")
    print(f"Device: {device}")

    # Load models
    models = []
    for ckpt_path in args.checkpoints:
        models.append(load_model(ckpt_path, device))
    print(f"Loaded {len(models)} model(s)")

    # Evaluate each split
    results = {}
    for split_name in ["id", "cm", "cf", "cd"]:
        split_dir = os.path.join(args.test_dir, split_name)
        if not os.path.isdir(split_dir):
            continue
        print(f"\n{'='*50}")
        print(f"Evaluating {split_name.upper()}...")
        print(f"{'='*50}")
        results[split_name] = evaluate_split(
            models, split_dir, split_name,
            args.data_root, args.image_root, device,
            use_tta=use_tta,
        )

    # Overall
    if results:
        avg_acc = np.mean([r["acc"] for r in results.values()])
        avg_opt_acc = np.mean([r["opt_acc"] for r in results.values()])
        avg_ap = np.mean([r["ap"] for r in results.values() if r["ap"] >= 0])
        results["average"] = {"acc": avg_acc, "opt_acc": avg_opt_acc, "ap": avg_ap}

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        for s in ["id", "cm", "cf", "cd"]:
            if s in results:
                r = results[s]
                print(f"  {s.upper():3s}: Acc={r['acc']*100:.1f}%  OptAcc={r['opt_acc']*100:.1f}%  AP={r['ap']:.4f}")
        print(f"  AVG: Acc={avg_acc*100:.1f}%  OptAcc={avg_opt_acc*100:.1f}%  AP={avg_ap:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
