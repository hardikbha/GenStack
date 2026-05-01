"""
Step 1: Generate pseudo-labels from v5_resume_hl on CF/CD test images.
Keeps only high-confidence predictions to avoid noise.
  - conf > 0.85  → pseudo-label = fake (1)
  - conf < 0.15  → pseudo-label = real (0)
  - 0.15-0.85    → discard (uncertain)

Output: checkpoints/v6_selftrain/pseudo_labels.json
"""

import os, sys, json, argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgendet import XGenDet
from data.hydrafake_dataset import HydraFakeTestDataset
from data.augmentations import CLIP_MEAN, CLIP_STD


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/v5_resume_hl/best_model.pth")
    p.add_argument("--test_dir",   default="/home/sachin.chaudhary/hydrafake/jsons/test")
    p.add_argument("--data_root",  default="/home/sachin.chaudhary")
    p.add_argument("--image_root", default="/home/sachin.chaudhary/hydrafake/test")
    p.add_argument("--output_dir", default="checkpoints/v6_selftrain")
    p.add_argument("--splits",     nargs="+", default=["cf", "cd"],
                   help="Test splits to pseudo-label (focus on hard ones)")
    p.add_argument("--fake_thresh",  type=float, default=0.85)
    p.add_argument("--real_thresh",  type=float, default=0.15)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--crop_size",    type=int,   default=224)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.checkpoint}")
    model = XGenDet()
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    pseudo_data = []
    stats = {"fake": 0, "real": 0, "discarded": 0}

    for split in args.splits:
        split_dir = os.path.join(args.test_dir, split)
        if not os.path.isdir(split_dir):
            continue
        print(f"\nProcessing split: {split.upper()}")

        for jf in sorted(f for f in os.listdir(split_dir) if f.endswith(".json")):
            gen = jf.replace(".json", "")
            ds = HydraFakeTestDataset(
                os.path.join(split_dir, jf), args.data_root,
                args.image_root, args.crop_size
            )
            if len(ds) == 0:
                continue

            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            confs, gt_labels, paths = [], [], []

            for imgs, labels in tqdm(loader, desc=f"  {gen}", leave=False, ncols=100):
                out = model(imgs.to(device), return_heatmap=False)
                confs.extend(out["confidence"].squeeze(-1).cpu().tolist())
                gt_labels.extend(labels.tolist())

            paths = [d[0] for d in ds.data]
            kept_fake = kept_real = disc = 0

            for path, conf, gt in zip(paths, confs, gt_labels):
                if conf >= args.fake_thresh:
                    pseudo_data.append({"path": path, "label": 1,
                                        "confidence": round(conf, 4),
                                        "gt_label": gt, "generator": gen, "split": split})
                    kept_fake += 1
                    stats["fake"] += 1
                elif conf <= args.real_thresh:
                    pseudo_data.append({"path": path, "label": 0,
                                        "confidence": round(conf, 4),
                                        "gt_label": gt, "generator": gen, "split": split})
                    kept_real += 1
                    stats["real"] += 1
                else:
                    disc += 1
                    stats["discarded"] += 1

            total = len(paths)
            print(f"  {gen:20s}: kept {kept_fake} fake + {kept_real} real, "
                  f"discarded {disc}/{total} ({disc*100//total}%)")

    out_path = os.path.join(args.output_dir, "pseudo_labels.json")
    json.dump(pseudo_data, open(out_path, "w"), indent=2)

    total_kept = stats["fake"] + stats["real"]
    print(f"\n{'='*60}")
    print(f"Total pseudo-labeled: {total_kept} "
          f"({stats['fake']} fake + {stats['real']} real)")
    print(f"Discarded (uncertain): {stats['discarded']}")
    print(f"Keep rate: {total_kept*100//(total_kept+stats['discarded'])}%")
    print(f"Saved → {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
