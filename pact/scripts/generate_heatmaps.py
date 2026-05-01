#!/usr/bin/env python3
"""
XGenDet Heatmap Generator: Load trained checkpoint and produce per-image
heatmap overlays, raw numpy heatmaps, and structured JSONL results.

Usage:
    python scripts/generate_heatmaps.py \
        --checkpoint checkpoints/xgendet_stage1/best_model.pth \
        --fake-dir  /path/to/fake/images \
        --real-dir  /path/to/real/images \
        --num-fake 25 --num-real 25 \
        --output-dir eval_outputs/heatmaps \
        --device cuda
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.xgendet import XGenDet
from models.prototype_module import ATTRIBUTE_BANKS

# ---------- constants ----------
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

FAMILY_NAMES = ["Real", "GAN", "Diffusion", "Autoregressive"]
ATTR_NAMES   = list(ATTRIBUTE_BANKS.keys())   # texture, edges, color, geometry, semantics, frequency

# ---------- helpers ----------

def build_transform(crop_size: int = 224):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def load_model(checkpoint_path: str, device: str = "cuda") -> XGenDet:
    """Instantiate XGenDet and load trained weights."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt.get("args", {})

    model = XGenDet(
        clip_model_name=args.get("clip_model", "ViT-L/14"),
        num_prompt_tokens=args.get("num_prompt_tokens", 8),
        num_prototypes=args.get("num_prototypes", 128),
        proto_dim=args.get("proto_dim", 128),
        shuffle_patch_size=args.get("shuffle_patch_size", 32),
        heatmap_output_size=args.get("crop_size", 224),
    )

    # Load state dict (handles exact-match and prefix-stripped cases)
    state_dict = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}  (first 5: {missing[:5]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}  (first 5: {unexpected[:5]})")

    model = model.to(device).eval()
    print(f"Model loaded from {checkpoint_path}  (epoch {ckpt.get('epoch', '?')})")
    return model


def collect_image_paths(
    fake_dir: str, real_dir: str, num_fake: int, num_real: int
):
    """Return list of (path, label) tuples. label: 1=fake, 0=real."""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    def _gather(directory, n, label):
        files = sorted([
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in exts
        ])
        return [(f, label) for f in files[:n]]

    items = _gather(fake_dir, num_fake, label=1)
    items += _gather(real_dir, num_real, label=0)
    return items


def make_heatmap_overlay(original_pil: Image.Image, heatmap_np: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
    """Blend a [H,W] float32 heatmap onto the original image (BGR output)."""
    orig = np.array(original_pil.convert("RGB"))
    h, w = orig.shape[:2]

    # Resize heatmap to match original
    hmap_resized = cv2.resize(heatmap_np, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalise to 0-255
    hmap_u8 = np.clip(hmap_resized * 255, 0, 255).astype(np.uint8)
    hmap_color = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)

    # Overlay
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_bgr, 1 - alpha, hmap_color, alpha, 0)
    return overlay


# ---------- main ----------

@torch.no_grad()
def run(args):
    device = args.device
    crop_size = 224
    transform = build_transform(crop_size)

    # 1. Load model
    model = load_model(args.checkpoint, device=device)

    # 2. Collect images
    items = collect_image_paths(args.fake_dir, args.real_dir,
                                args.num_fake, args.num_real)
    print(f"Collected {len(items)} images  ({args.num_fake} fake + {args.num_real} real)")

    # 3. Prepare output dirs
    out_root = Path(args.output_dir)
    overlay_dir = out_root / "overlays"
    raw_dir     = out_root / "raw_npy"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_root / "results.jsonl"

    total_correct = 0
    total = 0
    t0 = time.time()

    with open(jsonl_path, "w") as fout:
        for idx, (img_path, gt_label) in enumerate(items):
            # Load & preprocess
            pil_img = Image.open(img_path).convert("RGB")
            x = transform(pil_img).unsqueeze(0).to(device)  # [1,3,224,224]

            # Forward
            outputs = model.detect(x)

            pred       = outputs["prediction"].item()          # 0 or 1
            confidence = outputs["confidence"].item()          # float
            family_idx = outputs["family"].item()              # 0-3
            heatmap    = outputs["heatmap"]                    # [1,1,224,224]
            attr_raw   = outputs["attr_scores"]                # [1,6]

            # Convert heatmap to numpy [H,W]
            hmap_np = heatmap.squeeze().cpu().numpy().astype(np.float32)

            # Attribute dict
            attr_dict = {name: round(attr_raw[0, i].item(), 4)
                         for i, name in enumerate(ATTR_NAMES)}

            # Accuracy
            correct = int(pred == gt_label)
            total_correct += correct
            total += 1

            # Save raw heatmap
            stem = Path(img_path).stem
            tag  = "fake" if gt_label == 1 else "real"
            npy_name = f"{tag}_{stem}.npy"
            np.save(str(raw_dir / npy_name), hmap_np)

            # Save overlay
            overlay = make_heatmap_overlay(pil_img, hmap_np, alpha=0.5)
            overlay_name = f"{tag}_{stem}_overlay.png"
            cv2.imwrite(str(overlay_dir / overlay_name), overlay)

            # JSONL record
            record = {
                "image": img_path,
                "ground_truth": "fake" if gt_label == 1 else "real",
                "prediction": "fake" if pred == 1 else "real",
                "correct": bool(correct),
                "confidence": round(confidence, 5),
                "family": FAMILY_NAMES[family_idx],
                "attributes": attr_dict,
                "heatmap_mean": round(float(hmap_np.mean()), 5),
                "heatmap_max": round(float(hmap_np.max()), 5),
                "overlay_path": str(overlay_dir / overlay_name),
                "raw_heatmap_path": str(raw_dir / npy_name),
            }
            fout.write(json.dumps(record) + "\n")

            label_str = "FAKE" if gt_label == 1 else "REAL"
            pred_str  = "FAKE" if pred == 1 else "REAL"
            ok_str    = "OK" if correct else "WRONG"
            print(f"[{idx+1:3d}/{len(items)}] {ok_str:5s}  "
                  f"GT={label_str:4s}  Pred={pred_str:4s}  "
                  f"conf={confidence:.4f}  family={FAMILY_NAMES[family_idx]:14s}  "
                  f"hmap_mean={hmap_np.mean():.4f}  {Path(img_path).name}")

    elapsed = time.time() - t0
    acc = total_correct / total if total else 0
    print(f"\nDone in {elapsed:.1f}s  |  Accuracy: {total_correct}/{total} = {acc:.2%}")
    print(f"Overlays saved to:  {overlay_dir}")
    print(f"Raw heatmaps:       {raw_dir}")
    print(f"JSONL results:      {jsonl_path}")


def parse_args():
    p = argparse.ArgumentParser(description="XGenDet heatmap generation")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to best_model.pth")
    p.add_argument("--fake-dir", type=str, required=True,
                   help="Directory of fake images")
    p.add_argument("--real-dir", type=str, required=True,
                   help="Directory of real images")
    p.add_argument("--num-fake", type=int, default=25)
    p.add_argument("--num-real", type=int, default=25)
    p.add_argument("--output-dir", type=str, default="eval_outputs/heatmaps")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
