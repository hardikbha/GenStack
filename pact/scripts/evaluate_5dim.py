#!/usr/bin/env python3
"""
XGenBench 5-Dimension Evaluation Script.

Computes metrics across all 5 dimensions of the XGenBench evaluation framework:
  1. DETECT  - Binary classification (AP, Accuracy, F1)
  2. LOCATE  - Spatial localization quality (IoU, energy concentration, pointing game)
  3. ATTRIBUTE - Quality of 6 artifact attribute scores (MAE, Spearman correlation)
  4. EXPLAIN - Quality of NL explanations (BLEU-4, ROUGE-L, faithfulness)
  5. ROBUST  - Performance under degradation (AP drop under JPEG, blur, resize)

Overall Score = 0.25*Detect + 0.20*Locate + 0.20*Attribute + 0.20*Explain + 0.15*Robust

Usage:
  conda run -n D3 python scripts/evaluate_5dim.py \\
      --predictions predictions.jsonl \\
      --ground_truth gt.jsonl \\
      [--annotations gemini_annotations.jsonl] \\
      [--degraded_predictions degraded_jpeg50.jsonl degraded_jpeg30.jsonl ...] \\
      [--output results_5dim.json] \\
      [--compare method1_preds.jsonl method2_preds.jsonl ...]

Input Formats:
  predictions.jsonl:
    {"image_path": "...", "prediction": 0/1, "confidence": 0.85,
     "family_pred": "GAN", "heatmap_path": "heatmaps/img.npy",
     "attributes": {"Texture": 0.7, ...}, "explanation": "This image shows..."}

  ground_truth.jsonl:
    {"image_path": "...", "label": "fake", "generator": "stylegan", "family_id": 1}

  gemini_annotations.jsonl (optional):
    {"image_path": "...", "attributes": {"Texture": 0.8, ...},
     "explanation": "The image exhibits..."}
"""

import os
import sys
import json
import argparse
import warnings
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# Constants
# ============================================================================
ATTRIBUTE_NAMES = ["Texture", "Edges", "Color", "Geometry", "Semantics", "Frequency"]

DIMENSION_WEIGHTS = {
    "detect": 0.25,
    "locate": 0.20,
    "attribute": 0.20,
    "explain": 0.20,
    "robust": 0.15,
}

DEGRADATION_TYPES = [
    "jpeg_q50",
    "jpeg_q30",
    "gaussian_blur_sigma2",
    "resize_0.5x",
]


# ============================================================================
# I/O Helpers
# ============================================================================
def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping malformed line {line_num} in {path}: {e}")
    return records


def index_by_image(records, key="image_path"):
    """Build a dict mapping image_path -> record."""
    idx = {}
    for r in records:
        img_key = normalize_path(r.get(key, ""))
        if img_key:
            idx[img_key] = r
    return idx


def normalize_path(p):
    """Normalize a path for consistent matching."""
    if not p:
        return ""
    return os.path.normpath(os.path.expanduser(p))


# ============================================================================
# BLEU-4 Implementation (no nltk dependency)
# ============================================================================
def _get_ngrams(tokens, n):
    """Extract n-grams from a token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _count_ngrams(tokens, n):
    """Count n-grams in a token list."""
    counts = Counter()
    for ng in _get_ngrams(tokens, n):
        counts[ng] += 1
    return counts


def compute_bleu4(reference, hypothesis, max_n=4):
    """
    Compute BLEU-4 score between a reference and hypothesis string.

    Uses modified precision with brevity penalty, following the original
    BLEU paper (Papineni et al., 2002).

    Args:
        reference: Ground-truth text string.
        hypothesis: Predicted text string.
        max_n: Maximum n-gram order (default 4 for BLEU-4).

    Returns:
        BLEU-4 score in [0, 1].
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1.0 - len(ref_tokens) / len(hyp_tokens))) if len(hyp_tokens) > 0 else 0.0

    log_precisions = []
    for n in range(1, max_n + 1):
        ref_counts = _count_ngrams(ref_tokens, n)
        hyp_counts = _count_ngrams(hyp_tokens, n)

        # Clipped counts
        clipped = 0
        total = 0
        for ng, count in hyp_counts.items():
            clipped += min(count, ref_counts.get(ng, 0))
            total += count

        if total == 0:
            # No n-grams of this order in hypothesis -> BLEU = 0
            return 0.0

        precision = clipped / total
        if precision == 0:
            return 0.0
        log_precisions.append(np.log(precision))

    # Geometric mean of precisions with uniform weights
    avg_log_precision = np.mean(log_precisions)
    bleu = bp * np.exp(avg_log_precision)
    return float(bleu)


def compute_bleu4_corpus(references, hypotheses, max_n=4):
    """
    Compute corpus-level BLEU-4 by aggregating n-gram counts across all pairs.

    This is the standard corpus BLEU computation (not just averaging sentence BLEU).
    """
    if not references or not hypotheses:
        return 0.0

    total_hyp_len = 0
    total_ref_len = 0
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        total_hyp_len += len(hyp_tokens)
        total_ref_len += len(ref_tokens)

        for n in range(1, max_n + 1):
            ref_ng = _count_ngrams(ref_tokens, n)
            hyp_ng = _count_ngrams(hyp_tokens, n)
            for ng, count in hyp_ng.items():
                clipped_counts[n - 1] += min(count, ref_ng.get(ng, 0))
                total_counts[n - 1] += count

    if total_hyp_len == 0:
        return 0.0

    bp = min(1.0, np.exp(1.0 - total_ref_len / total_hyp_len))

    log_precisions = []
    for n in range(max_n):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            return 0.0
        log_precisions.append(np.log(clipped_counts[n] / total_counts[n]))

    bleu = bp * np.exp(np.mean(log_precisions))
    return float(bleu)


# ============================================================================
# ROUGE-L Implementation (no nltk dependency)
# ============================================================================
def _lcs_length(x, y):
    """Compute length of the longest common subsequence between two sequences."""
    m, n = len(x), len(y)
    # Use O(n) space
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(reference, hypothesis):
    """
    Compute ROUGE-L F1 score between a reference and hypothesis string.

    ROUGE-L uses longest common subsequence (LCS) for scoring.

    Args:
        reference: Ground-truth text string.
        hypothesis: Predicted text string.

    Returns:
        ROUGE-L F1 score in [0, 1].
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0

    lcs_len = _lcs_length(ref_tokens, hyp_tokens)

    if lcs_len == 0:
        return 0.0

    precision = lcs_len / len(hyp_tokens)
    recall = lcs_len / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


# ============================================================================
# Heatmap Utilities
# ============================================================================
def load_heatmap(heatmap_path):
    """
    Load a heatmap from a .npy or image file (.png, .jpg).

    Returns a 2D numpy array normalized to [0, 1], or None if loading fails.
    """
    if not heatmap_path or not os.path.exists(heatmap_path):
        return None
    try:
        ext = os.path.splitext(heatmap_path)[1].lower()
        if ext == ".npy":
            hmap = np.load(heatmap_path)
        elif ext in (".png", ".jpg", ".jpeg"):
            from PIL import Image
            img = Image.open(heatmap_path).convert("L")  # grayscale
            hmap = np.array(img, dtype=np.float32) / 255.0
            return hmap
        else:
            return None
        # Handle potential extra dimensions
        if hmap.ndim == 3:
            hmap = hmap.squeeze(0) if hmap.shape[0] == 1 else hmap.mean(axis=-1)
        if hmap.ndim == 4:
            hmap = hmap.squeeze(0).squeeze(0)
        # Normalize to [0, 1]
        hmin, hmax = hmap.min(), hmap.max()
        if hmax > hmin:
            hmap = (hmap - hmin) / (hmax - hmin)
        else:
            hmap = np.zeros_like(hmap)
        return hmap.astype(np.float32)
    except Exception:
        return None


def compute_iou_heatmap(heatmap, is_fake, threshold=0.5):
    """
    Compute IoU between a thresholded heatmap and a pseudo-GT mask.

    For fully-generated fake images, the pseudo-GT mask is the entire image.
    For real images, the pseudo-GT mask is empty (no artifacts).

    Args:
        heatmap: 2D numpy array in [0, 1].
        is_fake: Whether the image is fake (True) or real (False).
        threshold: Binarization threshold for the heatmap.

    Returns:
        IoU score in [0, 1].
    """
    pred_mask = (heatmap >= threshold).astype(np.float32)

    if is_fake:
        # Pseudo-GT: entire image is the artifact region
        gt_mask = np.ones_like(heatmap, dtype=np.float32)
    else:
        # Real image: no artifacts. Perfect prediction = no highlighted region.
        gt_mask = np.zeros_like(heatmap, dtype=np.float32)

    intersection = (pred_mask * gt_mask).sum()
    union = ((pred_mask + gt_mask) > 0).astype(np.float32).sum()

    if union == 0:
        # Both masks are empty -> perfect match for real images
        return 1.0
    return float(intersection / union)


def compute_energy_concentration(heatmap, top_k_percent=10):
    """
    Compute energy concentration ratio: fraction of total heatmap energy
    falling in the top-K% of pixels.

    Higher values indicate more focused/localized detections.

    Args:
        heatmap: 2D numpy array in [0, 1].
        top_k_percent: Percentage of top pixels to consider.

    Returns:
        Energy concentration ratio in [0, 1].
    """
    flat = heatmap.flatten()
    total_energy = flat.sum()
    if total_energy == 0:
        return 0.0

    k = max(1, int(len(flat) * top_k_percent / 100))
    # Get the top-k pixel values
    top_k_values = np.partition(flat, -k)[-k:]
    top_k_energy = top_k_values.sum()
    return float(top_k_energy / total_energy)


def compute_pointing_game(heatmap, is_fake):
    """
    Pointing game accuracy: does the heatmap's maximum activation
    fall within an artifact region?

    For fully-generated fakes: always 1 (the whole image is artifact).
    For real images: 1 if max activation is low (< 0.3), else 0.

    Args:
        heatmap: 2D numpy array in [0, 1].
        is_fake: Whether the image is fake.

    Returns:
        1 if pointing is correct, 0 otherwise.
    """
    max_val = heatmap.max()
    if is_fake:
        # For fully-generated fakes, any strong activation is correct
        return 1.0 if max_val > 0.3 else 0.0
    else:
        # For real images, the heatmap should have low activation everywhere
        return 1.0 if max_val < 0.3 else 0.0


# ============================================================================
# Dimension 1: DETECT
# ============================================================================
def evaluate_detect(predictions_idx, gt_idx):
    """
    Evaluate binary detection performance.

    Returns dict with AP, accuracy, F1, and per-generator breakdown.
    """
    results = {"available": False, "score": 0.0}

    # Collect matched pairs
    y_true = []
    y_conf = []
    per_gen = defaultdict(lambda: {"y_true": [], "y_conf": []})

    for img_path, pred in predictions_idx.items():
        gt = gt_idx.get(img_path)
        if gt is None:
            continue

        label_str = str(gt.get("label", "")).lower()
        if label_str in ("fake", "1", "ai", "generated"):
            label = 1
        elif label_str in ("real", "0", "authentic"):
            label = 0
        else:
            continue

        conf = pred.get("confidence")
        if conf is None:
            continue

        conf = float(conf)
        y_true.append(label)
        y_conf.append(conf)

        generator = gt.get("generator", "unknown")
        per_gen[generator]["y_true"].append(label)
        per_gen[generator]["y_conf"].append(conf)

    if len(y_true) < 2:
        print("  [DETECT] WARNING: Too few matched samples ({})".format(len(y_true)))
        return results

    y_true = np.array(y_true)
    y_conf = np.array(y_conf)
    y_pred = (y_conf >= 0.5).astype(int)

    # Check that both classes are present
    if len(np.unique(y_true)) < 2:
        print("  [DETECT] WARNING: Only one class present in ground truth")
        ap = 0.0
    else:
        ap = average_precision_score(y_true, y_conf)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Per-generator AP
    gen_metrics = {}
    for gen, data in sorted(per_gen.items()):
        gt_arr = np.array(data["y_true"])
        conf_arr = np.array(data["y_conf"])
        pred_arr = (conf_arr >= 0.5).astype(int)

        gen_result = {
            "n_samples": len(gt_arr),
            "n_real": int((gt_arr == 0).sum()),
            "n_fake": int((gt_arr == 1).sum()),
        }

        if len(np.unique(gt_arr)) >= 2:
            gen_result["ap"] = float(average_precision_score(gt_arr, conf_arr))
            gen_result["acc"] = float(accuracy_score(gt_arr, pred_arr))
            gen_result["f1"] = float(f1_score(gt_arr, pred_arr, zero_division=0))
        else:
            # Only one class: report accuracy only
            gen_result["ap"] = float("nan")
            gen_result["acc"] = float(accuracy_score(gt_arr, pred_arr))
            gen_result["f1"] = float("nan")

        gen_metrics[gen] = gen_result

    # Normalized score for overall computation: use AP as the primary metric
    score = ap

    results = {
        "available": True,
        "score": float(score),
        "ap": float(ap),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "n_samples": len(y_true),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "per_generator": gen_metrics,
    }

    return results


# ============================================================================
# Dimension 2: LOCATE
# ============================================================================
def evaluate_locate(predictions_idx, gt_idx):
    """
    Evaluate spatial localization quality using heatmaps.

    Returns dict with mean IoU, energy concentration, and pointing game accuracy.
    """
    results = {"available": False, "score": 0.0}

    ious = []
    energy_ratios = []
    pointing_scores = []
    n_heatmaps = 0
    n_missing = 0

    for img_path, pred in predictions_idx.items():
        gt = gt_idx.get(img_path)
        if gt is None:
            continue

        label_str = str(gt.get("label", "")).lower()
        is_fake = label_str in ("fake", "1", "ai", "generated")

        heatmap_path = pred.get("heatmap_path")
        heatmap = load_heatmap(heatmap_path)

        if heatmap is None:
            n_missing += 1
            # No heatmap -> score 0 for this sample
            ious.append(0.0)
            energy_ratios.append(0.0)
            pointing_scores.append(0.0)
            continue

        n_heatmaps += 1
        ious.append(compute_iou_heatmap(heatmap, is_fake))
        energy_ratios.append(compute_energy_concentration(heatmap, top_k_percent=10))
        pointing_scores.append(compute_pointing_game(heatmap, is_fake))

    total = n_heatmaps + n_missing
    if total == 0:
        print("  [LOCATE] WARNING: No samples to evaluate")
        return results

    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_energy = float(np.mean(energy_ratios)) if energy_ratios else 0.0
    mean_pointing = float(np.mean(pointing_scores)) if pointing_scores else 0.0

    # Composite localization score (weighted average)
    score = 0.5 * mean_iou + 0.25 * mean_energy + 0.25 * mean_pointing

    results = {
        "available": True,
        "score": float(score),
        "mean_iou": mean_iou,
        "energy_concentration_10pct": mean_energy,
        "pointing_game_accuracy": mean_pointing,
        "n_with_heatmap": n_heatmaps,
        "n_without_heatmap": n_missing,
        "n_total": total,
        "heatmap_coverage": float(n_heatmaps / total) if total > 0 else 0.0,
    }

    return results


# ============================================================================
# Dimension 3: ATTRIBUTE
# ============================================================================
def evaluate_attribute(predictions_idx, gt_idx, annotations_idx):
    """
    Evaluate quality of 6 artifact attribute scores against Gemini GT.

    Returns dict with MAE, Spearman correlation, and per-attribute breakdown.
    """
    results = {"available": False, "score": 0.0}

    if not annotations_idx:
        print("  [ATTRIBUTE] No annotation file provided. Score = 0.")
        return results

    per_attr_pred = defaultdict(list)
    per_attr_gt = defaultdict(list)
    n_matched = 0
    n_missing_pred = 0
    n_missing_gt = 0

    for img_path, pred in predictions_idx.items():
        gt_ann = annotations_idx.get(img_path)
        if gt_ann is None:
            n_missing_gt += 1
            continue

        pred_attrs = pred.get("attributes")
        gt_attrs = gt_ann.get("attributes")

        if not pred_attrs:
            n_missing_pred += 1
            continue
        if not gt_attrs:
            n_missing_gt += 1
            continue

        n_matched += 1
        # Build case-insensitive lookup for both pred and gt attributes
        pred_lower = {k.lower(): v for k, v in pred_attrs.items()}
        gt_lower = {k.lower(): v for k, v in gt_attrs.items()}
        for attr_name in ATTRIBUTE_NAMES:
            pred_val = pred_attrs.get(attr_name) or pred_lower.get(attr_name.lower())
            gt_val = gt_attrs.get(attr_name) or gt_lower.get(attr_name.lower())
            if pred_val is not None and gt_val is not None:
                per_attr_pred[attr_name].append(float(pred_val))
                per_attr_gt[attr_name].append(float(gt_val))

    if n_matched == 0:
        print("  [ATTRIBUTE] WARNING: No matched samples with attributes")
        return results

    # Per-attribute metrics
    attr_metrics = {}
    maes = []
    spearmans = []

    for attr_name in ATTRIBUTE_NAMES:
        preds = np.array(per_attr_pred.get(attr_name, []))
        gts = np.array(per_attr_gt.get(attr_name, []))

        if len(preds) < 2:
            attr_metrics[attr_name] = {"mae": float("nan"), "spearman": float("nan"), "n": len(preds)}
            continue

        mae = float(np.mean(np.abs(preds - gts)))
        maes.append(mae)

        # Spearman rank correlation
        if np.std(preds) > 0 and np.std(gts) > 0:
            rho, pval = scipy_stats.spearmanr(preds, gts)
            rho = float(rho) if not np.isnan(rho) else 0.0
        else:
            rho = 0.0
            pval = 1.0
        spearmans.append(rho)

        attr_metrics[attr_name] = {
            "mae": mae,
            "spearman": rho,
            "spearman_pvalue": float(pval),
            "n": len(preds),
        }

    mean_mae = float(np.mean(maes)) if maes else 1.0
    mean_spearman = float(np.mean(spearmans)) if spearmans else 0.0

    # Score: combine MAE (inverted, lower is better) and Spearman
    # MAE is in [0, 1] for normalized attributes; convert to quality score
    mae_score = max(0.0, 1.0 - mean_mae)
    spearman_score = max(0.0, (mean_spearman + 1.0) / 2.0)  # map [-1,1] -> [0,1]
    score = 0.5 * mae_score + 0.5 * spearman_score

    results = {
        "available": True,
        "score": float(score),
        "mean_mae": mean_mae,
        "mean_spearman": mean_spearman,
        "mae_quality_score": float(mae_score),
        "spearman_quality_score": float(spearman_score),
        "per_attribute": attr_metrics,
        "n_matched": n_matched,
        "n_missing_pred": n_missing_pred,
        "n_missing_gt": n_missing_gt,
    }

    return results


# ============================================================================
# Dimension 4: EXPLAIN
# ============================================================================
def evaluate_explain(predictions_idx, gt_idx, annotations_idx):
    """
    Evaluate quality of natural language explanations.

    Returns dict with BLEU-4, ROUGE-L, and faithfulness scores.
    """
    results = {"available": False, "score": 0.0}

    if not annotations_idx:
        print("  [EXPLAIN] No annotation file provided. Score = 0.")
        return results

    references = []
    hypotheses = []
    faithfulness_scores = []
    n_matched = 0
    n_missing = 0

    for img_path, pred in predictions_idx.items():
        gt_ann = annotations_idx.get(img_path)
        if gt_ann is None:
            continue

        pred_explanation = pred.get("explanation", "")
        gt_explanation = gt_ann.get("explanation", "")

        if not pred_explanation:
            n_missing += 1
            continue
        if not gt_explanation:
            continue

        n_matched += 1
        references.append(gt_explanation)
        hypotheses.append(pred_explanation)

        # Faithfulness: correlation between mentioned attributes and scores
        faith = _compute_faithfulness(pred_explanation, pred.get("attributes", {}))
        faithfulness_scores.append(faith)

    if n_matched == 0:
        print("  [EXPLAIN] WARNING: No matched explanations")
        return results

    # Corpus-level BLEU-4
    bleu4_corpus = compute_bleu4_corpus(references, hypotheses)

    # Average sentence-level BLEU-4 and ROUGE-L
    bleu4_scores = []
    rouge_l_scores = []
    for ref, hyp in zip(references, hypotheses):
        bleu4_scores.append(compute_bleu4(ref, hyp))
        rouge_l_scores.append(compute_rouge_l(ref, hyp))

    mean_bleu4 = float(np.mean(bleu4_scores)) if bleu4_scores else 0.0
    mean_rouge_l = float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0
    mean_faithfulness = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0

    # Composite explanation score
    score = (0.30 * mean_bleu4 + 0.40 * mean_rouge_l + 0.30 * mean_faithfulness)

    results = {
        "available": True,
        "score": float(score),
        "bleu4_corpus": float(bleu4_corpus),
        "bleu4_sentence_avg": mean_bleu4,
        "rouge_l": mean_rouge_l,
        "faithfulness": mean_faithfulness,
        "n_matched": n_matched,
        "n_missing_explanation": n_missing,
    }

    return results


def _compute_faithfulness(explanation, attributes):
    """
    Compute faithfulness: how well does the explanation align with
    the attribute scores?

    If an attribute is mentioned in the explanation, its score should be
    high. If not mentioned, its score should be low. We measure this
    as the Spearman correlation between mention indicators and scores.

    Returns a score in [0, 1].
    """
    if not attributes:
        return 0.0

    explanation_lower = explanation.lower()

    # Check which attributes are mentioned
    attr_keywords = {
        "Texture": ["texture", "textur", "smooth", "rough", "grain", "noise pattern"],
        "Edges": ["edge", "boundary", "border", "contour", "sharp", "blurr"],
        "Color": ["color", "colour", "saturation", "hue", "chromatic", "palette"],
        "Geometry": ["geometry", "geometric", "shape", "distort", "proportion", "symmetr"],
        "Semantics": ["semantic", "meaning", "context", "object", "scene", "unnatural", "anomal"],
        "Frequency": ["frequency", "spectral", "fourier", "artifact", "periodic", "pattern"],
    }

    mentions = []
    scores = []
    for attr_name in ATTRIBUTE_NAMES:
        attr_score = attributes.get(attr_name)
        if attr_score is None:
            continue

        # Check if any keyword for this attribute is mentioned
        keywords = attr_keywords.get(attr_name, [attr_name.lower()])
        mentioned = any(kw in explanation_lower for kw in keywords)
        mentions.append(1.0 if mentioned else 0.0)
        scores.append(float(attr_score))

    if len(mentions) < 3 or np.std(mentions) == 0 or np.std(scores) == 0:
        # Not enough variation to compute meaningful correlation
        # Fall back to: if high-score attributes are mentioned, that's faithful
        if not mentions:
            return 0.0
        # Simple agreement: fraction of high-score attributes (>0.5) that are mentioned
        high_attrs = [(m, s) for m, s in zip(mentions, scores) if s > 0.5]
        if high_attrs:
            return float(np.mean([m for m, s in high_attrs]))
        return 0.5  # neutral

    rho, _ = scipy_stats.spearmanr(mentions, scores)
    if np.isnan(rho):
        return 0.0
    # Map from [-1, 1] to [0, 1]
    return float(max(0.0, (rho + 1.0) / 2.0))


# ============================================================================
# Dimension 5: ROBUST
# ============================================================================
def evaluate_robust(detect_results, degraded_predictions_list, gt_idx, degradation_labels):
    """
    Evaluate robustness under degradation.

    Takes original AP and computes AP drop for each degradation type.

    Args:
        detect_results: Results from evaluate_detect on clean images.
        degraded_predictions_list: List of prediction dicts (one per degradation).
        gt_idx: Ground truth index.
        degradation_labels: List of degradation type labels.

    Returns dict with per-degradation AP drop and average robustness score.
    """
    results = {"available": False, "score": 0.0}

    if not detect_results.get("available"):
        print("  [ROBUST] Original detection results not available. Score = 0.")
        return results

    original_ap = detect_results.get("ap", 0.0)
    if original_ap == 0.0:
        print("  [ROBUST] Original AP is 0. Cannot compute robustness.")
        return results

    if not degraded_predictions_list:
        print("  [ROBUST] No degraded prediction files provided. Score = 0.")
        return results

    degradation_results = {}
    ap_drops = []

    for deg_preds, deg_label in zip(degraded_predictions_list, degradation_labels):
        deg_idx = index_by_image(deg_preds)
        deg_detect = evaluate_detect(deg_idx, gt_idx)

        if not deg_detect.get("available"):
            degradation_results[deg_label] = {
                "ap": float("nan"),
                "ap_drop": float("nan"),
                "ap_retention": float("nan"),
            }
            continue

        deg_ap = deg_detect["ap"]
        ap_drop = (original_ap - deg_ap) / original_ap if original_ap > 0 else 0.0
        ap_retention = 1.0 - ap_drop  # Higher is better

        degradation_results[deg_label] = {
            "ap": float(deg_ap),
            "ap_drop": float(ap_drop),
            "ap_retention": float(ap_retention),
            "accuracy": float(deg_detect.get("accuracy", 0.0)),
            "f1": float(deg_detect.get("f1", 0.0)),
        }
        ap_drops.append(ap_drop)

    if not ap_drops:
        return results

    mean_ap_drop = float(np.mean(ap_drops))
    mean_retention = float(1.0 - mean_ap_drop)
    # Score = retention (higher is more robust)
    score = max(0.0, mean_retention)

    results = {
        "available": True,
        "score": float(score),
        "original_ap": float(original_ap),
        "mean_ap_drop": mean_ap_drop,
        "mean_ap_retention": mean_retention,
        "per_degradation": degradation_results,
    }

    return results


# ============================================================================
# Overall Score
# ============================================================================
def compute_overall_score(dim_results):
    """
    Compute weighted overall score across 5 dimensions.

    Overall = 0.25*Detect + 0.20*Locate + 0.20*Attribute + 0.20*Explain + 0.15*Robust

    Dimensions that are not available get score=0.
    """
    weighted_sum = 0.0
    weight_sum = 0.0

    for dim_name, weight in DIMENSION_WEIGHTS.items():
        dim_score = dim_results.get(dim_name, {}).get("score", 0.0)
        weighted_sum += weight * dim_score
        weight_sum += weight

    overall = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    return float(overall)


# ============================================================================
# Display / Reporting
# ============================================================================
def print_results_table(all_results, method_name="XGenDet"):
    """Print a formatted console table of all metrics."""
    print()
    print("=" * 78)
    print(f"  XGenBench 5-Dimension Evaluation Results: {method_name}")
    print("=" * 78)

    # ---- Dimension 1: DETECT ----
    det = all_results.get("detect", {})
    print()
    print("-" * 78)
    print("  [1] DETECT - Binary Classification")
    print("-" * 78)
    if det.get("available"):
        print(f"    Average Precision (AP): {det['ap']:.4f}")
        print(f"    Accuracy @0.5:          {det['accuracy']:.4f}")
        print(f"    F1 Score:               {det['f1']:.4f}")
        print(f"    Precision:              {det['precision']:.4f}")
        print(f"    Recall:                 {det['recall']:.4f}")
        print(f"    Samples:                {det['n_samples']} ({det['n_real']} real, {det['n_fake']} fake)")

        if det.get("per_generator"):
            print()
            print(f"    {'Generator':<22s} {'AP':>8s} {'Acc':>8s} {'F1':>8s} {'N':>6s}")
            print(f"    {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
            for gen, gm in sorted(det["per_generator"].items()):
                ap_str = f"{gm['ap']:.4f}" if not (isinstance(gm['ap'], float) and np.isnan(gm['ap'])) else "  N/A "
                f1_str = f"{gm['f1']:.4f}" if not (isinstance(gm['f1'], float) and np.isnan(gm['f1'])) else "  N/A "
                print(f"    {gen:<22s} {ap_str:>8s} {gm['acc']:.4f} {f1_str:>8s} {gm['n_samples']:>6d}")
    else:
        print("    NOT AVAILABLE (missing confidence scores or ground truth)")
    print(f"    >> Dimension Score: {det.get('score', 0.0):.4f}")

    # ---- Dimension 2: LOCATE ----
    loc = all_results.get("locate", {})
    print()
    print("-" * 78)
    print("  [2] LOCATE - Spatial Localization")
    print("-" * 78)
    if loc.get("available"):
        print(f"    Mean IoU:                    {loc['mean_iou']:.4f}")
        print(f"    Energy Concentration (10%):  {loc['energy_concentration_10pct']:.4f}")
        print(f"    Pointing Game Accuracy:      {loc['pointing_game_accuracy']:.4f}")
        print(f"    Heatmap Coverage:            {loc['heatmap_coverage']:.1%} "
              f"({loc['n_with_heatmap']}/{loc['n_total']})")
    else:
        print("    NOT AVAILABLE (no heatmaps provided)")
    print(f"    >> Dimension Score: {loc.get('score', 0.0):.4f}")

    # ---- Dimension 3: ATTRIBUTE ----
    attr = all_results.get("attribute", {})
    print()
    print("-" * 78)
    print("  [3] ATTRIBUTE - Artifact Attribute Scores")
    print("-" * 78)
    if attr.get("available"):
        print(f"    Mean MAE:          {attr['mean_mae']:.4f}")
        print(f"    Mean Spearman rho: {attr['mean_spearman']:.4f}")
        print(f"    Matched Samples:   {attr['n_matched']}")

        if attr.get("per_attribute"):
            print()
            print(f"    {'Attribute':<14s} {'MAE':>8s} {'Spearman':>10s} {'N':>6s}")
            print(f"    {'-'*14} {'-'*8} {'-'*10} {'-'*6}")
            for aname in ATTRIBUTE_NAMES:
                am = attr["per_attribute"].get(aname, {})
                mae_str = f"{am.get('mae', float('nan')):.4f}"
                sp_str = f"{am.get('spearman', float('nan')):.4f}"
                n = am.get("n", 0)
                print(f"    {aname:<14s} {mae_str:>8s} {sp_str:>10s} {n:>6d}")
    else:
        print("    NOT AVAILABLE (no attribute data or annotations)")
    print(f"    >> Dimension Score: {attr.get('score', 0.0):.4f}")

    # ---- Dimension 4: EXPLAIN ----
    exp = all_results.get("explain", {})
    print()
    print("-" * 78)
    print("  [4] EXPLAIN - Natural Language Explanations")
    print("-" * 78)
    if exp.get("available"):
        print(f"    BLEU-4 (corpus):    {exp['bleu4_corpus']:.4f}")
        print(f"    BLEU-4 (sentence):  {exp['bleu4_sentence_avg']:.4f}")
        print(f"    ROUGE-L:            {exp['rouge_l']:.4f}")
        print(f"    Faithfulness:       {exp['faithfulness']:.4f}")
        print(f"    Matched:            {exp['n_matched']} "
              f"(missing: {exp['n_missing_explanation']})")
    else:
        print("    NOT AVAILABLE (no explanations or annotations)")
    print(f"    >> Dimension Score: {exp.get('score', 0.0):.4f}")

    # ---- Dimension 5: ROBUST ----
    rob = all_results.get("robust", {})
    print()
    print("-" * 78)
    print("  [5] ROBUST - Degradation Robustness")
    print("-" * 78)
    if rob.get("available"):
        print(f"    Original AP:        {rob['original_ap']:.4f}")
        print(f"    Mean AP Drop:       {rob['mean_ap_drop']:.4f}")
        print(f"    Mean AP Retention:  {rob['mean_ap_retention']:.4f}")

        if rob.get("per_degradation"):
            print()
            print(f"    {'Degradation':<28s} {'AP':>8s} {'AP Drop':>10s} {'Retention':>10s}")
            print(f"    {'-'*28} {'-'*8} {'-'*10} {'-'*10}")
            for deg, dm in rob["per_degradation"].items():
                ap_str = f"{dm['ap']:.4f}" if not np.isnan(dm.get('ap', float('nan'))) else "  N/A "
                drop_str = f"{dm['ap_drop']:.4f}" if not np.isnan(dm.get('ap_drop', float('nan'))) else "  N/A "
                ret_str = f"{dm['ap_retention']:.4f}" if not np.isnan(dm.get('ap_retention', float('nan'))) else "  N/A "
                print(f"    {deg:<28s} {ap_str:>8s} {drop_str:>10s} {ret_str:>10s}")
    else:
        print("    NOT AVAILABLE (no degraded predictions provided)")
    print(f"    >> Dimension Score: {rob.get('score', 0.0):.4f}")

    # ---- Overall ----
    overall = all_results.get("overall_score", 0.0)
    print()
    print("=" * 78)
    print("  OVERALL SCORE (weighted average)")
    print("=" * 78)
    print()
    print(f"    {'Dimension':<14s} {'Weight':>8s} {'Score':>8s} {'Weighted':>10s}")
    print(f"    {'-'*14} {'-'*8} {'-'*8} {'-'*10}")
    for dim_name, weight in DIMENSION_WEIGHTS.items():
        dim_score = all_results.get(dim_name, {}).get("score", 0.0)
        weighted = weight * dim_score
        avail = "  " if all_results.get(dim_name, {}).get("available") else " *"
        print(f"    {dim_name.upper():<14s} {weight:>7.0%} {dim_score:>8.4f} {weighted:>10.4f}{avail}")
    print(f"    {'-'*14} {'-'*8} {'-'*8} {'-'*10}")
    print(f"    {'OVERALL':<14s} {'100%':>8s} {'':>8s} {overall:>10.4f}")
    print()
    n_avail = sum(1 for d in DIMENSION_WEIGHTS if all_results.get(d, {}).get("available"))
    if n_avail < 5:
        print(f"    * = dimension not available (scored as 0)")
        print(f"    Available dimensions: {n_avail}/5")
    print("=" * 78)
    print()


def print_comparison_table(all_method_results):
    """Print a comparison table across multiple methods."""
    if len(all_method_results) < 2:
        return

    print()
    print("=" * 90)
    print("  METHOD COMPARISON")
    print("=" * 90)
    print()

    methods = list(all_method_results.keys())

    # Header
    header = f"  {'Metric':<24s}"
    for m in methods:
        header += f" {m:>14s}"
    print(header)
    print("  " + "-" * (24 + 15 * len(methods)))

    # Rows
    rows = [
        ("DETECT AP", lambda r: r.get("detect", {}).get("ap")),
        ("DETECT Acc", lambda r: r.get("detect", {}).get("accuracy")),
        ("DETECT F1", lambda r: r.get("detect", {}).get("f1")),
        ("LOCATE IoU", lambda r: r.get("locate", {}).get("mean_iou")),
        ("LOCATE Energy", lambda r: r.get("locate", {}).get("energy_concentration_10pct")),
        ("LOCATE Pointing", lambda r: r.get("locate", {}).get("pointing_game_accuracy")),
        ("ATTR MAE", lambda r: r.get("attribute", {}).get("mean_mae")),
        ("ATTR Spearman", lambda r: r.get("attribute", {}).get("mean_spearman")),
        ("EXPLAIN BLEU-4", lambda r: r.get("explain", {}).get("bleu4_corpus")),
        ("EXPLAIN ROUGE-L", lambda r: r.get("explain", {}).get("rouge_l")),
        ("EXPLAIN Faithful", lambda r: r.get("explain", {}).get("faithfulness")),
        ("ROBUST Retention", lambda r: r.get("robust", {}).get("mean_ap_retention")),
        ("", None),  # separator
        ("Detect Score", lambda r: r.get("detect", {}).get("score")),
        ("Locate Score", lambda r: r.get("locate", {}).get("score")),
        ("Attribute Score", lambda r: r.get("attribute", {}).get("score")),
        ("Explain Score", lambda r: r.get("explain", {}).get("score")),
        ("Robust Score", lambda r: r.get("robust", {}).get("score")),
        ("", None),  # separator
        ("OVERALL", lambda r: r.get("overall_score")),
    ]

    for row_name, getter in rows:
        if getter is None:
            print("  " + "-" * (24 + 15 * len(methods)))
            continue

        line = f"  {row_name:<24s}"
        values = []
        for m in methods:
            val = getter(all_method_results[m])
            values.append(val)

        # Find best value (highest for most metrics, lowest for MAE)
        is_lower_better = "MAE" in row_name
        valid_vals = [v for v in values if v is not None and not np.isnan(v)]
        best_val = min(valid_vals) if is_lower_better and valid_vals else (max(valid_vals) if valid_vals else None)

        for val in values:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                line += f" {'N/A':>14s}"
            else:
                marker = " *" if val == best_val and len(valid_vals) > 1 else "  "
                line += f" {val:>12.4f}{marker}"

        print(line)

    print()
    print("  * = best among compared methods")
    print("=" * 90)
    print()


# ============================================================================
# Main Entry Point
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="XGenBench 5-Dimension Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection-only evaluation:
  python scripts/evaluate_5dim.py \\
      --predictions predictions.jsonl \\
      --ground_truth gt.jsonl

  # Full 5-dimension evaluation with annotations:
  python scripts/evaluate_5dim.py \\
      --predictions predictions.jsonl \\
      --ground_truth gt.jsonl \\
      --annotations gemini_annotations.jsonl

  # With robustness evaluation:
  python scripts/evaluate_5dim.py \\
      --predictions predictions.jsonl \\
      --ground_truth gt.jsonl \\
      --degraded_predictions deg_jpeg50.jsonl deg_jpeg30.jsonl deg_blur2.jsonl deg_resize05.jsonl \\
      --degradation_labels jpeg_q50 jpeg_q30 gaussian_blur_sigma2 resize_0.5x

  # Compare multiple methods:
  python scripts/evaluate_5dim.py \\
      --predictions xgendet_preds.jsonl \\
      --ground_truth gt.jsonl \\
      --compare d3_preds.jsonl cnndect_preds.jsonl \\
      --compare_names D3 CNNDetect
        """,
    )

    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to predictions JSONL file",
    )
    parser.add_argument(
        "--ground_truth", type=str, required=True,
        help="Path to ground truth JSONL file",
    )
    parser.add_argument(
        "--annotations", type=str, default=None,
        help="Path to Gemini annotations JSONL file (GT attributes + explanations)",
    )
    parser.add_argument(
        "--degraded_predictions", type=str, nargs="*", default=None,
        help="Paths to degraded prediction JSONL files (for robustness eval)",
    )
    parser.add_argument(
        "--degradation_labels", type=str, nargs="*", default=None,
        help="Labels for each degradation type (must match --degraded_predictions count)",
    )
    parser.add_argument(
        "--compare", type=str, nargs="*", default=None,
        help="Additional prediction JSONL files for method comparison",
    )
    parser.add_argument(
        "--compare_names", type=str, nargs="*", default=None,
        help="Names for comparison methods (must match --compare count)",
    )
    parser.add_argument(
        "--method_name", type=str, default="XGenDet",
        help="Name for the primary method (default: XGenDet)",
    )
    parser.add_argument(
        "--output", type=str, default="results_5dim.json",
        help="Path for output JSON file (default: results_5dim.json)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress console output (only write JSON)",
    )

    return parser.parse_args()


def evaluate_single_method(predictions, gt_idx, annotations_idx,
                           degraded_list=None, degradation_labels=None):
    """
    Run all 5 dimensions of evaluation for a single method.

    Returns a dict with results for each dimension and the overall score.
    """
    pred_idx = index_by_image(predictions)

    # Dimension 1: DETECT
    detect_results = evaluate_detect(pred_idx, gt_idx)

    # Dimension 2: LOCATE
    locate_results = evaluate_locate(pred_idx, gt_idx)

    # Dimension 3: ATTRIBUTE
    attribute_results = evaluate_attribute(pred_idx, gt_idx, annotations_idx)

    # Dimension 4: EXPLAIN
    explain_results = evaluate_explain(pred_idx, gt_idx, annotations_idx)

    # Dimension 5: ROBUST
    robust_results = evaluate_robust(
        detect_results, degraded_list or [], gt_idx, degradation_labels or []
    )

    all_results = {
        "detect": detect_results,
        "locate": locate_results,
        "attribute": attribute_results,
        "explain": explain_results,
        "robust": robust_results,
    }

    all_results["overall_score"] = compute_overall_score(all_results)

    return all_results


def main():
    args = parse_args()

    # ---- Load data ----
    print("Loading data...")
    predictions = load_jsonl(args.predictions)
    print(f"  Predictions:    {len(predictions)} records from {args.predictions}")

    ground_truth = load_jsonl(args.ground_truth)
    print(f"  Ground truth:   {len(ground_truth)} records from {args.ground_truth}")

    gt_idx = index_by_image(ground_truth)

    annotations_idx = None
    if args.annotations:
        annotations = load_jsonl(args.annotations)
        annotations_idx = index_by_image(annotations)
        print(f"  Annotations:    {len(annotations)} records from {args.annotations}")

    degraded_list = []
    degradation_labels = []
    if args.degraded_predictions:
        if args.degradation_labels and len(args.degradation_labels) == len(args.degraded_predictions):
            degradation_labels = args.degradation_labels
        else:
            degradation_labels = [f"degradation_{i}" for i in range(len(args.degraded_predictions))]
            if args.degradation_labels:
                print("  WARNING: --degradation_labels count does not match --degraded_predictions count")
                print("           Using auto-generated labels.")

        for dp_path, dl in zip(args.degraded_predictions, degradation_labels):
            dp = load_jsonl(dp_path)
            degraded_list.append(dp)
            print(f"  Degraded ({dl}): {len(dp)} records from {dp_path}")

    # ---- Evaluate primary method ----
    print(f"\nEvaluating {args.method_name}...")
    primary_results = evaluate_single_method(
        predictions, gt_idx, annotations_idx, degraded_list, degradation_labels
    )

    if not args.quiet:
        print_results_table(primary_results, method_name=args.method_name)

    # ---- Evaluate comparison methods ----
    all_method_results = {args.method_name: primary_results}

    if args.compare:
        compare_names = args.compare_names or []
        for i, comp_path in enumerate(args.compare):
            if i < len(compare_names):
                comp_name = compare_names[i]
            else:
                comp_name = Path(comp_path).stem

            print(f"\nEvaluating comparison method: {comp_name}...")
            comp_preds = load_jsonl(comp_path)
            print(f"  Loaded {len(comp_preds)} predictions from {comp_path}")

            comp_results = evaluate_single_method(
                comp_preds, gt_idx, annotations_idx, degraded_list, degradation_labels
            )

            if not args.quiet:
                print_results_table(comp_results, method_name=comp_name)

            all_method_results[comp_name] = comp_results

        if not args.quiet:
            print_comparison_table(all_method_results)

    # ---- Save JSON output ----
    output_data = {
        "methods": {},
        "comparison_summary": {},
    }

    for method_name, results in all_method_results.items():
        # Convert numpy types to Python types for JSON serialization
        output_data["methods"][method_name] = _sanitize_for_json(results)

    # Build comparison summary
    if len(all_method_results) > 1:
        summary = {}
        for dim_name in list(DIMENSION_WEIGHTS.keys()) + ["overall_score"]:
            if dim_name == "overall_score":
                scores = {m: r.get("overall_score", 0.0) for m, r in all_method_results.items()}
            else:
                scores = {m: r.get(dim_name, {}).get("score", 0.0) for m, r in all_method_results.items()}
            summary[dim_name] = scores
        output_data["comparison_summary"] = summary

    # Write output
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")

    # Return overall score (useful for scripting)
    overall = primary_results.get("overall_score", 0.0)
    print(f"\n{args.method_name} Overall Score: {overall:.4f}")
    return overall


def _sanitize_for_json(obj):
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val):
            return None
        if np.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    elif isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return None
        return obj
    return obj


# ============================================================================
# Self-test: create synthetic data and validate metrics
# ============================================================================
def _self_test():
    """
    Run a quick self-test with synthetic data to verify all metrics compute
    without errors. Invoked via --self_test flag.
    """
    import tempfile

    print("=" * 60)
    print("  Running self-test with synthetic data")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n_samples = 100

    # Create synthetic GT
    gt_records = []
    generators = ["stylegan", "dalle", "midjourney", "real_camera"]
    for i in range(n_samples):
        is_fake = i < 60  # 60 fake, 40 real
        gen = generators[i % len(generators)] if is_fake else "real_camera"
        gt_records.append({
            "image_path": f"/data/images/img_{i:04d}.png",
            "label": "fake" if is_fake else "real",
            "generator": gen,
            "family_id": (i % 3) + 1 if is_fake else 0,
        })

    # Create synthetic predictions
    pred_records = []
    for i, gt in enumerate(gt_records):
        is_fake = gt["label"] == "fake"
        # Simulate noisy confidence: biased toward correct
        if is_fake:
            conf = float(np.clip(rng.normal(0.75, 0.15), 0, 1))
        else:
            conf = float(np.clip(rng.normal(0.25, 0.15), 0, 1))

        attrs = {}
        for attr in ATTRIBUTE_NAMES:
            attrs[attr] = float(np.clip(rng.uniform(0.1, 0.9), 0, 1))

        pred_records.append({
            "image_path": gt["image_path"],
            "prediction": 1 if conf >= 0.5 else 0,
            "confidence": conf,
            "family_pred": "GAN" if conf >= 0.5 else "Real",
            "heatmap_path": None,  # No heatmaps for self-test
            "attributes": attrs,
            "explanation": "The image shows synthetic texture patterns "
                           "with color inconsistencies and edge artifacts."
                           if is_fake else "The image appears authentic with natural texture.",
        })

    # Create synthetic annotations
    ann_records = []
    for i, gt in enumerate(gt_records):
        is_fake = gt["label"] == "fake"
        attrs = {}
        for attr in ATTRIBUTE_NAMES:
            if is_fake:
                attrs[attr] = float(np.clip(rng.uniform(0.4, 0.95), 0, 1))
            else:
                attrs[attr] = float(np.clip(rng.uniform(0.05, 0.3), 0, 1))

        ann_records.append({
            "image_path": gt["image_path"],
            "attributes": attrs,
            "explanation": "This AI-generated image exhibits noticeable texture anomalies "
                           "and color distribution shifts with edge artifacts."
                           if is_fake else "This photograph shows natural characteristics "
                           "with consistent texture and color.",
        })

    # Create synthetic degraded predictions (lower accuracy)
    deg_records = []
    for pred in pred_records:
        deg_conf = float(np.clip(pred["confidence"] + rng.normal(0, 0.1), 0, 1))
        deg_records.append({
            "image_path": pred["image_path"],
            "prediction": 1 if deg_conf >= 0.5 else 0,
            "confidence": deg_conf,
        })

    # Write to temp files
    tmp_dir = tempfile.mkdtemp(prefix="xgenbench_selftest_")

    def write_jsonl(records, name):
        path = os.path.join(tmp_dir, name)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    gt_path = write_jsonl(gt_records, "gt.jsonl")
    pred_path = write_jsonl(pred_records, "pred.jsonl")
    ann_path = write_jsonl(ann_records, "ann.jsonl")
    deg_path = write_jsonl(deg_records, "deg.jsonl")

    # Run evaluation
    gt_idx = index_by_image(gt_records)
    ann_idx = index_by_image(ann_records)

    results = evaluate_single_method(
        pred_records, gt_idx, ann_idx,
        degraded_list=[deg_records],
        degradation_labels=["jpeg_q50"],
    )

    print_results_table(results, method_name="SelfTest")

    # Validate
    checks_passed = 0
    checks_total = 0

    def check(name, condition):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name}")

    check("Detect available", results["detect"]["available"])
    check("Detect AP in [0,1]", 0.0 <= results["detect"].get("ap", -1) <= 1.0)
    check("Detect Acc in [0,1]", 0.0 <= results["detect"].get("accuracy", -1) <= 1.0)
    check("Detect F1 in [0,1]", 0.0 <= results["detect"].get("f1", -1) <= 1.0)
    check("Detect has per-gen", len(results["detect"].get("per_generator", {})) > 0)

    check("Locate available", results["locate"]["available"])
    check("Locate IoU in [0,1]", 0.0 <= results["locate"].get("mean_iou", -1) <= 1.0)

    check("Attribute available", results["attribute"]["available"])
    check("Attribute MAE in [0,1]", 0.0 <= results["attribute"].get("mean_mae", -1) <= 1.0)

    check("Explain available", results["explain"]["available"])
    check("Explain ROUGE-L in [0,1]", 0.0 <= results["explain"].get("rouge_l", -1) <= 1.0)

    check("Robust available", results["robust"]["available"])
    check("Robust AP retention in [0,1]",
          0.0 <= results["robust"].get("mean_ap_retention", -1) <= 1.0)

    check("Overall score in [0,1]", 0.0 <= results["overall_score"] <= 1.0)
    check("Overall score > 0", results["overall_score"] > 0.0)

    print()
    print(f"  Self-test: {checks_passed}/{checks_total} checks passed")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if checks_passed == checks_total:
        print("  ALL CHECKS PASSED")
    else:
        print(f"  WARNING: {checks_total - checks_passed} checks FAILED")
        sys.exit(1)

    print("=" * 60)
    return checks_passed == checks_total


if __name__ == "__main__":
    # Check for special self-test mode
    if "--self_test" in sys.argv:
        sys.argv.remove("--self_test")
        success = _self_test()
        sys.exit(0 if success else 1)

    main()
