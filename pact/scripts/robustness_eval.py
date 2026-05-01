#!/usr/bin/env python3
"""
XGenDet Robustness Evaluation Script.

Loads the trained XGenDet model and evaluates robustness under various
image perturbations on the 210-image eval set:
  - JPEG compression: quality = 30, 50, 70, 90, 100 (100 = no compression)
  - Gaussian blur: sigma = 0.5, 1.0, 2.0
  - Resize-rescale: 75%, 50% (downscale then upscale back to 224x224)

For each perturbation, computes:
  - Average Precision (AP)
  - Accuracy at threshold 0.5

Saves full results to eval_outputs/robustness_results.json.

Usage:
    python scripts/robustness_eval.py [--checkpoint PATH] [--device cuda]
"""

import argparse
import io
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms
from sklearn.metrics import average_precision_score, accuracy_score

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CHECKPOINT = str(
    PROJECT_ROOT / "checkpoints" / "xgendet_fulldata" / "best_model.pth"
)
ORIGINALS_DIR = str(PROJECT_ROOT / "eval_outputs" / "originals")
GT_FILE = str(PROJECT_ROOT / "eval_outputs" / "ground_truth.jsonl")
OUTPUT_FILE = str(PROJECT_ROOT / "eval_outputs" / "robustness_results.json")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

# Perturbation definitions
JPEG_QUALITIES = [30, 50, 70, 90, 100]
BLUR_SIGMAS = [0.5, 1.0, 2.0]
RESIZE_SCALES = [0.75, 0.50]

# ---------------------------------------------------------------------------
# Model loading (mirrors generate_eval_outputs.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Instantiate XGenDet and load checkpoint weights."""
    from models.xgendet import XGenDet

    print(f"[Model] Instantiating XGenDet (ViT-L/14)...")
    model = XGenDet(
        clip_model_name="ViT-L/14",
        num_prompt_tokens=8,
        tune_layer_norm=True,
        num_prototypes=128,
        proto_dim=128,
        proto_heads=4,
        extract_layers=(6, 12, 18, 23),
        shuffle_patch_size=32,
        heatmap_output_size=224,
        num_families=4,
        dropout=0.2,
    )

    print(f"[Model] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    # Handle DataParallel key prefix
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[Model] Warning: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"[Model] Warning: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")

    model = model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    print(f"[Model] Loaded (epoch {epoch}), device={device}")
    return model


# ---------------------------------------------------------------------------
# Image loading & perturbation helpers
# ---------------------------------------------------------------------------

def build_clip_transform():
    """CLIP normalization transform (input: PIL Image 224x224 -> tensor)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def pil_to_tensor(pil_img, transform):
    """Convert a PIL Image (224x224, RGB) to a normalized tensor [1,3,224,224]."""
    return transform(pil_img).unsqueeze(0)


def apply_jpeg_compression(pil_img: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression in-memory and return the decoded result."""
    if quality >= 100:
        return pil_img.copy()
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_gaussian_blur(pil_img: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur to a PIL image."""
    return pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_resize_rescale(pil_img: Image.Image, scale: float) -> Image.Image:
    """Downscale to scale*224, then upscale back to 224x224."""
    orig_size = pil_img.size  # (W, H) = (224, 224)
    small_w = max(1, int(orig_size[0] * scale))
    small_h = max(1, int(orig_size[1] * scale))
    small = pil_img.resize((small_w, small_h), Image.BILINEAR)
    return small.resize(orig_size, Image.BILINEAR)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_set():
    """
    Load ground truth labels from ground_truth.jsonl.
    Build a mapping from the original-file names in eval_outputs/originals/
    to their integer labels (0=real, 1=fake).

    The predictions.jsonl stores uid = {source}_{generator}_{label}_{basename}
    and the originals folder contains files named {uid}.png.
    Ground truth stores source image paths.

    We'll use predictions.jsonl to build the mapping from original_file -> label.
    If predictions.jsonl is unavailable, we fall back to parsing the filename
    (the label is embedded as the 3rd underscore-separated token counting from
    the source prefix: e.g. OOD_biggan_0_xxx -> label=0, ID_ADM_1_xxx -> label=1).
    """
    originals_dir = ORIGINALS_DIR
    gt_file = GT_FILE

    # Try predictions.jsonl first for authoritative mapping
    pred_file = os.path.join(os.path.dirname(gt_file), "predictions.jsonl")
    file_to_label = {}

    if os.path.isfile(pred_file):
        with open(pred_file) as f:
            for line in f:
                rec = json.loads(line)
                fname = rec["original_file"]
                label = rec["ground_truth"]  # 0 or 1
                file_to_label[fname] = int(label)
        print(f"[Data] Loaded {len(file_to_label)} labels from predictions.jsonl")
    else:
        # Fallback: parse from ground_truth.jsonl
        # Map original image_path -> label
        path_to_label = {}
        with open(gt_file) as f:
            for line in f:
                rec = json.loads(line)
                lbl = 1 if rec["label"] == "fake" else 0
                path_to_label[rec["image_path"]] = lbl
        print(f"[Data] Loaded {len(path_to_label)} labels from ground_truth.jsonl")

        # For each file in originals/, try to determine label from filename pattern
        # Pattern: {source}_{generator}_{label}_{rest}.png
        for fname in sorted(os.listdir(originals_dir)):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            # Parse label from filename: split by '_' and find the label position
            # Examples: OOD_biggan_0_xxx.png, ID_ADM_1_xxx.png
            parts = fname.rsplit(".", 1)[0].split("_")
            # The label is after source and generator: parts[0]=source, parts[1]=generator, parts[2]=label
            try:
                label = int(parts[2])
                file_to_label[fname] = label
            except (IndexError, ValueError):
                print(f"[Data] Warning: cannot parse label from filename {fname}, skipping")

    # Build final list
    eval_items = []
    for fname in sorted(os.listdir(originals_dir)):
        if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
            continue
        if fname not in file_to_label:
            print(f"[Data] Warning: no label for {fname}, skipping")
            continue
        eval_items.append({
            "filename": fname,
            "path": os.path.join(originals_dir, fname),
            "label": file_to_label[fname],
        })

    print(f"[Data] Eval set: {len(eval_items)} images "
          f"({sum(1 for e in eval_items if e['label']==0)} real, "
          f"{sum(1 for e in eval_items if e['label']==1)} fake)")
    return eval_items


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_on_pil(model, pil_img, transform, device):
    """Run model inference on a single PIL image (224x224 RGB).
    Returns (prediction: int, confidence: float)."""
    x = pil_to_tensor(pil_img, transform).to(device)
    outputs = model.detect(x)
    pred = outputs["prediction"].item()
    conf = outputs["confidence"].item()
    return pred, conf


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_perturbation(model, eval_items, perturb_fn, perturb_name, transform, device):
    """
    Evaluate model under a single perturbation.

    Returns dict with AP, accuracy, per-image predictions.
    """
    labels = []
    confidences = []
    predictions = []
    per_image = []

    for item in eval_items:
        pil_img = Image.open(item["path"]).convert("RGB")
        perturbed = perturb_fn(pil_img)
        pred, conf = run_inference_on_pil(model, perturbed, transform, device)

        labels.append(item["label"])
        confidences.append(conf)
        predictions.append(pred)
        per_image.append({
            "filename": item["filename"],
            "label": item["label"],
            "prediction": pred,
            "confidence": round(conf, 5),
        })

    labels_np = np.array(labels)
    confs_np = np.array(confidences)
    preds_np = np.array(predictions)

    # AP (uses confidence as score for ranking)
    ap = float(average_precision_score(labels_np, confs_np))
    acc = float(accuracy_score(labels_np, preds_np))

    return {
        "perturbation": perturb_name,
        "ap": round(ap, 5),
        "accuracy": round(acc, 5),
        "num_images": len(labels),
        "num_correct": int((preds_np == labels_np).sum()),
        "per_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser(description="XGenDet Robustness Evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA not available, falling back to CPU")
        device = "cpu"

    print("=" * 70)
    print("XGenDet Robustness Evaluation")
    print("=" * 70)

    # 1. Load model
    model = load_model(args.checkpoint, device=device)

    # 2. Load eval set
    eval_items = load_eval_set()
    if len(eval_items) == 0:
        print("[Error] No images found. Aborting.")
        sys.exit(1)

    # 3. Build CLIP transform
    transform = build_clip_transform()

    # 4. Define perturbations
    perturbations = []

    # Clean baseline (no perturbation)
    perturbations.append(("clean", lambda img: img.copy()))

    # JPEG compression
    for q in JPEG_QUALITIES:
        perturbations.append(
            (f"jpeg_q{q}", lambda img, _q=q: apply_jpeg_compression(img, _q))
        )

    # Gaussian blur
    for sigma in BLUR_SIGMAS:
        perturbations.append(
            (f"blur_sigma{sigma}", lambda img, _s=sigma: apply_gaussian_blur(img, _s))
        )

    # Resize-rescale
    for scale in RESIZE_SCALES:
        pct = int(scale * 100)
        perturbations.append(
            (f"resize_{pct}pct", lambda img, _sc=scale: apply_resize_rescale(img, _sc))
        )

    # 5. Run evaluation
    all_results = []
    t0 = time.time()

    for i, (name, fn) in enumerate(perturbations):
        t1 = time.time()
        print(f"\n[{i+1}/{len(perturbations)}] Evaluating: {name} ...")
        result = evaluate_perturbation(model, eval_items, fn, name, transform, device)
        elapsed = time.time() - t1
        print(f"  AP={result['ap']:.4f}  Acc={result['accuracy']:.4f}  "
              f"({result['num_correct']}/{result['num_images']})  [{elapsed:.1f}s]")
        all_results.append(result)

    total_time = time.time() - t0

    # 6. Build summary
    summary = {
        "clean": None,
        "jpeg": {},
        "blur": {},
        "resize": {},
    }
    for r in all_results:
        name = r["perturbation"]
        entry = {"ap": r["ap"], "accuracy": r["accuracy"]}
        if name == "clean":
            summary["clean"] = entry
        elif name.startswith("jpeg_"):
            summary["jpeg"][name] = entry
        elif name.startswith("blur_"):
            summary["blur"][name] = entry
        elif name.startswith("resize_"):
            summary["resize"][name] = entry

    # Compute AP drop relative to clean
    clean_ap = summary["clean"]["ap"] if summary["clean"] else None
    ap_drops = {}
    if clean_ap is not None:
        for r in all_results:
            if r["perturbation"] != "clean":
                drop = clean_ap - r["ap"]
                ap_drops[r["perturbation"]] = round(drop, 5)

    output = {
        "checkpoint": args.checkpoint,
        "device": device,
        "num_images": len(eval_items),
        "total_time_sec": round(total_time, 1),
        "summary": summary,
        "ap_drops_from_clean": ap_drops,
        "detailed_results": all_results,
    }

    # 7. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Total time: {total_time:.1f}s")

    # 8. Print summary table
    print(f"\n{'Perturbation':<20s} {'AP':>8s} {'Acc':>8s} {'AP Drop':>10s}")
    print("-" * 48)
    for r in all_results:
        name = r["perturbation"]
        drop_str = f"{ap_drops.get(name, 0.0):+.4f}" if name != "clean" else "baseline"
        print(f"{name:<20s} {r['ap']:8.4f} {r['accuracy']:8.4f} {drop_str:>10s}")
    print("=" * 70)


if __name__ == "__main__":
    main()
