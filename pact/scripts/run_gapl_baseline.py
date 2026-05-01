"""
Run GAPL V2 baseline inference on XGenDet eval images.

Usage:
    conda run -n gapl_env python scripts/run_gapl_baseline.py
"""

import os
import sys
import json
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

# Add GAPL V2 code to path
GAPL_V2_DIR = "/home/sachin.chaudhary/competetion/fresh_repos/GAPL_org_clean_20260221/competition_custom_v2"
sys.path.insert(0, GAPL_V2_DIR)

# Set CLIP model path for GAPL
os.environ["GAPL_CLIP_MODEL_ID"] = "openai/clip-vit-large-patch14"

CHECKPOINT_PATH = os.path.join(
    GAPL_V2_DIR,
    "runs/v2_cv_20260306_183314/fold0/best_rocauc.pt"
)

EVAL_DIR = "/home/sachin.chaudhary/xgendet/eval_outputs/originals"
OUTPUT_PATH = "/home/sachin.chaudhary/xgendet/eval_outputs/gapl_predictions.jsonl"

# ImageNet normalization (same as GAPL inference_image.py)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_filename(fname):
    """Parse generator and label from filename.

    Format: {ID/OOD}_{generator}_{label}_{rest}.png
    where label 0 = real, 1 = fake.
    Examples:
      ID_ADM_0_ILSVRC2012_val_00003462.png  → ADM, real
      ID_ADM_1_134_adm_85.png               → ADM, fake
      OOD_crn_0_train2_LSUN_val_00006227.png → crn, real
      OOD_crn_1_crn_00044.png               → crn, fake
    """
    stem = Path(fname).stem
    parts = stem.split("_")

    if parts[0] in ("ID", "OOD") and len(parts) >= 3:
        generator = parts[1]
        label = "real" if parts[2] == "0" else "fake"
        return generator, label

    return stem, "unknown"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # Load model
    from models_v2 import GAPLModelV2

    print(f"[INFO] Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    # Create model with frozen backbone (inference only)
    model = GAPLModelV2(
        freeze_backbone=True,
        proto_delta_enabled=True,
        device=device,
    )

    # Load prototypes
    proto = ckpt.get("proVec", ckpt.get("propVec"))
    if proto is not None:
        model.load_prototype(proto)
        print(f"[INFO] Loaded prototypes: {proto.shape}")

    # Load proto_delta if present
    if "proto_delta" in ckpt and model.proto_delta is not None:
        model.proto_delta.data.copy_(ckpt["proto_delta"])
        print(f"[INFO] Loaded proto_delta")

    # Load model weights
    model_state = ckpt.get("model", {})
    # Filter to only load compatible keys
    current_state = model.state_dict()
    filtered = {}
    skipped = 0
    for k, v in model_state.items():
        if k in current_state and current_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped += 1
    msg = model.load_state_dict(filtered, strict=False)
    print(f"[INFO] Loaded {len(filtered)} params, skipped {skipped} (shape mismatch/missing)")

    model.to(device)
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    # Collect images
    images = sorted(glob.glob(os.path.join(EVAL_DIR, "*.png")))
    images += sorted(glob.glob(os.path.join(EVAL_DIR, "*.jpg")))
    images += sorted(glob.glob(os.path.join(EVAL_DIR, "*.jpeg")))
    images += sorted(glob.glob(os.path.join(EVAL_DIR, "*.JPEG")))
    print(f"[INFO] Found {len(images)} images")

    results = []
    correct = 0
    total = 0

    with torch.no_grad():
        for i, img_path in enumerate(images):
            fname = os.path.basename(img_path)
            generator, gt_label = parse_filename(fname)

            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            score = model(tensor).sigmoid().item()
            prediction = "FAKE" if score > 0.5 else "REAL"
            confidence = score if score > 0.5 else 1.0 - score

            is_correct = (prediction == "FAKE" and gt_label == "fake") or \
                         (prediction == "REAL" and gt_label == "real")
            if gt_label != "unknown":
                correct += int(is_correct)
                total += 1

            entry = {
                "image_path": img_path,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "raw_score": round(score, 4),
                "generator": generator,
                "ground_truth": gt_label,
            }
            results.append(entry)

            if (i + 1) % 20 == 0 or (i + 1) == len(images):
                acc = correct / total * 100 if total > 0 else 0
                print(f"  [{i+1}/{len(images)}] Running acc: {acc:.1f}% ({correct}/{total})")

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    acc = correct / total * 100 if total > 0 else 0
    print(f"\n[DONE] GAPL Baseline Results:")
    print(f"  Total: {len(results)}, Accuracy: {acc:.1f}% ({correct}/{total})")
    print(f"  Saved to: {OUTPUT_PATH}")

    # Per-generator breakdown
    from collections import defaultdict
    gen_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        g = r["generator"]
        gt = r["ground_truth"]
        pred = r["prediction"]
        if gt != "unknown":
            gen_stats[g]["total"] += 1
            is_correct = (pred == "FAKE" and gt == "fake") or (pred == "REAL" and gt == "real")
            gen_stats[g]["correct"] += int(is_correct)

    print(f"\n  Per-Generator:")
    for g in sorted(gen_stats.keys()):
        s = gen_stats[g]
        a = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"    {g:20s}: {a:5.1f}% ({s['correct']}/{s['total']})")


if __name__ == "__main__":
    main()
