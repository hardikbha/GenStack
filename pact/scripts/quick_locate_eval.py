"""
Quick Localization Evaluation: Score XGenDet heatmaps against GT masks.

For face-based generators (deepfake, stylegan, stargan, whichfaceisreal):
  GT mask = face region detected by OpenCV Haar cascade.

For scene-based generators (crn, cyclegan, gaugan, etc.):
  GT mask = full image (entire image is generated).
  Use Heatmap Discriminability instead of IoU.

Metrics computed:
  - Pixel-AUC: heatmap as pixel-level classifier vs GT mask
  - IoU@0.5: threshold heatmap → binary → IoU with GT mask
  - Pointing Game: argmax(heatmap) inside GT mask?
  - Heatmap Discriminability: AUC of mean(heatmap) for real vs fake
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from PIL import Image

HEATMAP_DIR = "/home/sachin.chaudhary/xgendet/eval_outputs/heatmaps"
PREDICTIONS_FILE = "/home/sachin.chaudhary/xgendet/eval_outputs/xgendet_predictions_enhanced.jsonl"
GT_FILE = "/home/sachin.chaudhary/xgendet/eval_outputs/ground_truth.jsonl"
OUTPUT_FILE = "/home/sachin.chaudhary/xgendet/eval_outputs/localization_results.json"

FACE_GENERATORS = {"deepfake", "stylegan", "stylegan2", "stargan", "whichfaceisreal"}
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def load_heatmap_grayscale(heatmap_path):
    """Load 3-panel heatmap PNG, extract just the heatmap (middle panel)."""
    img = cv2.imread(heatmap_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    # The heatmap PNG has 3 panels: Original | Heatmap | Overlay
    panel_w = w // 3
    heatmap_panel = img[:, panel_w:2*panel_w]
    # Convert to grayscale and normalize
    gray = cv2.cvtColor(heatmap_panel, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray


def create_face_mask(image_path, target_size=None):
    """Detect face(s) in image, return binary mask."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    mask = np.zeros(gray.shape, dtype=np.float32)
    if len(faces) > 0:
        for (x, y, fw, fh) in faces:
            # Expand face region by 20% for better coverage
            expand = 0.2
            x1 = max(0, int(x - fw * expand))
            y1 = max(0, int(y - fh * expand))
            x2 = min(gray.shape[1], int(x + fw * (1 + expand)))
            y2 = min(gray.shape[0], int(y + fh * (1 + expand)))
            mask[y1:y2, x1:x2] = 1.0
    else:
        # No face detected — use full image as fallback
        mask[:] = 1.0

    if target_size is not None:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return mask


def compute_pixel_auc(heatmap, gt_mask):
    """Compute pixel-level AUC: heatmap values vs binary mask labels."""
    h_flat = heatmap.flatten()
    m_flat = gt_mask.flatten()
    if m_flat.sum() == 0 or m_flat.sum() == len(m_flat):
        return None  # All same class
    try:
        return roc_auc_score(m_flat, h_flat)
    except:
        return None


def compute_pixel_ap(heatmap, gt_mask):
    """Compute pixel-level Average Precision."""
    h_flat = heatmap.flatten()
    m_flat = gt_mask.flatten()
    if m_flat.sum() == 0 or m_flat.sum() == len(m_flat):
        return None
    try:
        return average_precision_score(m_flat, h_flat)
    except:
        return None


def compute_iou(heatmap, gt_mask, threshold=0.5):
    """Compute IoU between thresholded heatmap and GT mask."""
    pred_binary = (heatmap >= threshold).astype(np.float32)
    intersection = (pred_binary * gt_mask).sum()
    union = ((pred_binary + gt_mask) > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_pointing_game(heatmap, gt_mask):
    """Does the max activation of heatmap fall inside GT mask?"""
    max_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    return float(gt_mask[max_idx] > 0.5)


def main():
    # Load predictions and GT
    preds = {}
    with open(PREDICTIONS_FILE) as f:
        for line in f:
            d = json.loads(line.strip())
            preds[d["image_path"]] = d

    gt_records = {}
    with open(GT_FILE) as f:
        for line in f:
            d = json.loads(line.strip())
            gt_records[d["image_path"]] = d

    results_per_gen = {}
    all_heatmap_means = []  # (mean_value, is_fake) for discriminability

    for img_path, pred in preds.items():
        gt = gt_records.get(img_path, {})
        generator = gt.get("generator", "unknown")
        label = gt.get("label", "unknown")
        is_fake = label == "fake"

        heatmap_path = pred.get("heatmap_path", "")
        if not heatmap_path or not os.path.exists(heatmap_path):
            continue

        heatmap = load_heatmap_grayscale(heatmap_path)
        if heatmap is None:
            continue

        # Track mean heatmap for discriminability
        all_heatmap_means.append((float(heatmap.mean()), int(is_fake)))

        if not is_fake:
            continue  # Only evaluate localization on fake images

        # Create GT mask
        if generator in FACE_GENERATORS:
            # Use face detection for face generators
            # Try original image path first
            gt_mask = create_face_mask(img_path, target_size=(heatmap.shape[1], heatmap.shape[0]))
        else:
            # Full image is fake — use uniform mask
            gt_mask = np.ones_like(heatmap)

        if gt_mask is None:
            continue

        # Resize heatmap to match mask if needed
        if heatmap.shape != gt_mask.shape:
            gt_mask = cv2.resize(gt_mask, (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Compute metrics
        pixel_auc = compute_pixel_auc(heatmap, gt_mask)
        pixel_ap = compute_pixel_ap(heatmap, gt_mask)
        iou = compute_iou(heatmap, gt_mask, threshold=0.5)
        pointing = compute_pointing_game(heatmap, gt_mask)

        if generator not in results_per_gen:
            results_per_gen[generator] = {"pixel_auc": [], "pixel_ap": [], "iou": [], "pointing": [], "n": 0}

        if pixel_auc is not None:
            results_per_gen[generator]["pixel_auc"].append(pixel_auc)
        if pixel_ap is not None:
            results_per_gen[generator]["pixel_ap"].append(pixel_ap)
        results_per_gen[generator]["iou"].append(iou)
        results_per_gen[generator]["pointing"].append(pointing)
        results_per_gen[generator]["n"] += 1

    # Compute heatmap discriminability
    means = np.array([x[0] for x in all_heatmap_means])
    labels = np.array([x[1] for x in all_heatmap_means])
    try:
        discrim_auc = roc_auc_score(labels, means)
    except:
        discrim_auc = 0.0

    # Print results
    print("=" * 80)
    print("  LOCALIZATION EVALUATION RESULTS")
    print("=" * 80)
    print(f"\n  Heatmap Discriminability (AUC of mean intensity): {discrim_auc:.4f}")
    print(f"  (Can heatmap intensity alone distinguish real from fake?)\n")

    print(f"  {'Generator':<20s} {'Pixel-AUC':>10s} {'Pixel-AP':>10s} {'IoU@0.5':>10s} {'Pointing':>10s} {'N':>5s}")
    print(f"  {'-'*65}")

    all_aucs, all_aps, all_ious, all_points = [], [], [], []
    for gen in sorted(results_per_gen.keys()):
        r = results_per_gen[gen]
        auc_mean = np.mean(r["pixel_auc"]) if r["pixel_auc"] else 0
        ap_mean = np.mean(r["pixel_ap"]) if r["pixel_ap"] else 0
        iou_mean = np.mean(r["iou"])
        point_mean = np.mean(r["pointing"])
        n = r["n"]

        face_tag = " [face]" if gen in FACE_GENERATORS else ""
        print(f"  {gen + face_tag:<20s} {auc_mean:>10.4f} {ap_mean:>10.4f} {iou_mean:>10.4f} {point_mean:>10.4f} {n:>5d}")

        all_aucs.extend(r["pixel_auc"])
        all_aps.extend(r["pixel_ap"])
        all_ious.extend(r["iou"])
        all_points.extend(r["pointing"])

    print(f"  {'-'*65}")
    print(f"  {'AVERAGE':<20s} {np.mean(all_aucs) if all_aucs else 0:>10.4f} {np.mean(all_aps) if all_aps else 0:>10.4f} {np.mean(all_ious):>10.4f} {np.mean(all_points):>10.4f} {len(all_ious):>5d}")

    # Save results
    output = {
        "heatmap_discriminability_auc": discrim_auc,
        "overall": {
            "pixel_auc": float(np.mean(all_aucs)) if all_aucs else 0,
            "pixel_ap": float(np.mean(all_aps)) if all_aps else 0,
            "iou_at_05": float(np.mean(all_ious)),
            "pointing_game": float(np.mean(all_points)),
            "n_fake_images": len(all_ious),
        },
        "per_generator": {}
    }
    for gen, r in results_per_gen.items():
        output["per_generator"][gen] = {
            "pixel_auc": float(np.mean(r["pixel_auc"])) if r["pixel_auc"] else 0,
            "pixel_ap": float(np.mean(r["pixel_ap"])) if r["pixel_ap"] else 0,
            "iou_at_05": float(np.mean(r["iou"])),
            "pointing_game": float(np.mean(r["pointing"])),
            "n": r["n"],
            "is_face": gen in FACE_GENERATORS,
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
