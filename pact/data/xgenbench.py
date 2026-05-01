"""
XGenBench: Benchmark for Explainable Generalized AI-Generated Image Detection.

5 evaluation dimensions:
1. Detect: Binary classification accuracy (AP, AUC, Acc)
2. Locate: Spatial heatmap quality (IoU, Pixel-AP, Pointing Game)
3. Explain: NL explanation quality (GPT-4 judge, ROUGE-L)
4. Attribute: Attribute score accuracy (MAE, Spearman correlation)
5. Robust: Performance under perturbations (JPEG, blur, resize)
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional

from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from scipy.stats import spearmanr


class XGenBench:
    """XGenBench evaluation suite."""

    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        heatmap_root: str = None,
        mask_root: str = None,
    ):
        """
        Args:
            annotation_file: JSONL with ground truth annotations
            image_root: Root for images
            heatmap_root: Root for ground truth heatmaps/masks
            mask_root: Root for pixel-level ground truth masks
        """
        self.image_root = image_root
        self.heatmap_root = heatmap_root
        self.mask_root = mask_root

        self.data = []
        with open(annotation_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

        print(f"XGenBench: Loaded {len(self.data)} evaluation samples")

    def evaluate_detection(
        self,
        predictions: List[Dict],
    ) -> Dict[str, float]:
        """
        Evaluate binary detection performance.

        Args:
            predictions: list of dicts with 'image_path', 'confidence' (0-1)

        Returns:
            dict with AP, AUC, Acc, BestAcc, F1
        """
        pred_map = {p["image_path"]: p["confidence"] for p in predictions}

        labels = []
        scores = []
        for entry in self.data:
            path = entry["image_path"]
            if path in pred_map:
                labels.append(entry.get("label", 0))
                scores.append(pred_map[path])

        labels = np.array(labels)
        scores = np.array(scores)

        if len(labels) == 0:
            return {}

        results = {
            "AP": float(average_precision_score(labels, scores)),
            "AUC": float(roc_auc_score(labels, scores)),
            "Acc@0.5": float(accuracy_score(labels, (scores > 0.5).astype(int))),
        }

        # Best threshold accuracy
        best_acc = 0.0
        best_thresh = 0.5
        for t in np.arange(0.1, 0.95, 0.05):
            acc = accuracy_score(labels, (scores > t).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_thresh = t
        results["BestAcc"] = float(best_acc)
        results["BestThresh"] = float(best_thresh)

        # F1
        preds = (scores > 0.5).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        results["F1"] = float(2 * precision * recall / max(precision + recall, 1e-8))

        return results

    def evaluate_localization(
        self,
        predictions: List[Dict],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Evaluate spatial heatmap quality against ground truth masks.

        Args:
            predictions: list of dicts with 'image_path', 'heatmap' (numpy H x W)
            iou_threshold: IoU threshold for binary evaluation

        Returns:
            dict with IoU, PixelAP, PointingGame
        """
        if not self.mask_root:
            return {"warning": "No mask_root provided for localization evaluation"}

        ious = []
        pixel_aps = []
        pointing_correct = 0
        pointing_total = 0

        pred_map = {p["image_path"]: p["heatmap"] for p in predictions}

        for entry in self.data:
            path = entry["image_path"]
            if path not in pred_map:
                continue

            mask_path = os.path.join(self.mask_root, entry.get("mask_path", ""))
            if not os.path.exists(mask_path):
                continue

            pred_heatmap = pred_map[path]
            gt_mask = np.load(mask_path) if mask_path.endswith(".npy") else \
                np.array(Image.open(mask_path).convert("L")) / 255.0

            # Resize pred to match gt
            if pred_heatmap.shape != gt_mask.shape:
                from PIL import Image as PILImage
                pred_pil = PILImage.fromarray((pred_heatmap.squeeze() * 255).astype(np.uint8))
                pred_pil = pred_pil.resize((gt_mask.shape[1], gt_mask.shape[0]), PILImage.BILINEAR)
                pred_heatmap = np.array(pred_pil).astype(np.float32) / 255.0

            pred_heatmap = pred_heatmap.squeeze()

            # IoU (binarized)
            pred_bin = (pred_heatmap > 0.5).astype(float)
            gt_bin = (gt_mask > 0.5).astype(float)
            intersection = (pred_bin * gt_bin).sum()
            union = pred_bin.sum() + gt_bin.sum() - intersection
            iou = intersection / max(union, 1e-8)
            ious.append(iou)

            # Pixel-AP
            pixel_ap = average_precision_score(gt_bin.flatten(), pred_heatmap.flatten())
            pixel_aps.append(pixel_ap)

            # Pointing Game: does max activation fall within GT mask?
            max_idx = np.unravel_index(pred_heatmap.argmax(), pred_heatmap.shape)
            if gt_bin[max_idx] > 0.5:
                pointing_correct += 1
            pointing_total += 1

        results = {}
        if ious:
            results["IoU"] = float(np.mean(ious))
        if pixel_aps:
            results["PixelAP"] = float(np.mean(pixel_aps))
        if pointing_total > 0:
            results["PointingGame"] = float(pointing_correct / pointing_total)

        return results

    def evaluate_attributes(
        self,
        predictions: List[Dict],
    ) -> Dict[str, float]:
        """
        Evaluate attribute score accuracy.

        Args:
            predictions: list of dicts with 'image_path', 'attributes' (dict of 6 scores)

        Returns:
            dict with MAE per attribute, overall MAE, Spearman correlation
        """
        attr_names = [
            "texture_consistency", "edge_quality", "color_distribution",
            "geometric_coherence", "semantic_plausibility", "frequency_artifacts"
        ]

        pred_map = {p["image_path"]: p["attributes"] for p in predictions}

        gt_all = {name: [] for name in attr_names}
        pred_all = {name: [] for name in attr_names}

        for entry in self.data:
            path = entry["image_path"]
            if path not in pred_map or "attributes" not in entry:
                continue

            gt_attrs = entry["attributes"]
            pred_attrs = pred_map[path]

            for name in attr_names:
                if name in gt_attrs and name in pred_attrs:
                    gt_all[name].append(gt_attrs[name])
                    pred_all[name].append(pred_attrs[name])

        results = {}
        all_maes = []

        for name in attr_names:
            if gt_all[name]:
                gt = np.array(gt_all[name])
                pred = np.array(pred_all[name])
                mae = np.mean(np.abs(gt - pred))
                results[f"MAE_{name}"] = float(mae)
                all_maes.append(mae)

                if len(gt) > 2:
                    corr, pval = spearmanr(gt, pred)
                    results[f"Spearman_{name}"] = float(corr)

        if all_maes:
            results["MAE_overall"] = float(np.mean(all_maes))

        return results

    def evaluate_explanation(
        self,
        predictions: List[Dict],
        api_key: str = None,
        use_rouge: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate NL explanation quality.

        Args:
            predictions: list of dicts with 'image_path', 'explanation' (text)
            api_key: OpenAI API key for GPT-4 judge (optional)
            use_rouge: Whether to compute ROUGE-L

        Returns:
            dict with ROUGE-L and optionally GPT-4 judge score
        """
        pred_map = {p["image_path"]: p["explanation"] for p in predictions}

        results = {}

        if use_rouge:
            try:
                from rouge_score import rouge_scorer
                scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

                rouge_scores = []
                for entry in self.data:
                    path = entry["image_path"]
                    if path in pred_map and "explanation" in entry:
                        score = scorer.score(entry["explanation"], pred_map[path])
                        rouge_scores.append(score["rougeL"].fmeasure)

                if rouge_scores:
                    results["ROUGE-L"] = float(np.mean(rouge_scores))
            except ImportError:
                results["ROUGE-L"] = "rouge_score not installed"

        return results

    def evaluate_robustness(
        self,
        model_fn,
        perturbation: str = "jpeg",
        levels: list = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate detection robustness under perturbations.

        Args:
            model_fn: callable that takes PIL Image and returns confidence score
            perturbation: 'jpeg', 'blur', or 'resize'
            levels: perturbation levels to test

        Returns:
            dict mapping level -> metrics
        """
        import io
        from PIL import ImageFilter

        if levels is None:
            if perturbation == "jpeg":
                levels = [30, 50, 70, 90, 100]
            elif perturbation == "blur":
                levels = [0.5, 1.0, 1.5, 2.0]
            elif perturbation == "resize":
                levels = [0.5, 0.75, 1.0, 1.5]

        results = {}

        for level in levels:
            labels = []
            scores = []

            for entry in self.data:
                img_path = os.path.join(self.image_root, entry["image_path"])
                if not os.path.exists(img_path):
                    continue

                img = Image.open(img_path).convert("RGB")

                # Apply perturbation
                if perturbation == "jpeg":
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=int(level))
                    buf.seek(0)
                    img = Image.open(buf).convert("RGB")
                elif perturbation == "blur":
                    img = img.filter(ImageFilter.GaussianBlur(radius=level))
                elif perturbation == "resize":
                    w, h = img.size
                    new_w, new_h = int(w * level), int(h * level)
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                    img = img.resize((w, h), Image.LANCZOS)

                try:
                    confidence = model_fn(img)
                    labels.append(entry.get("label", 0))
                    scores.append(confidence)
                except Exception:
                    continue

            if labels:
                labels = np.array(labels)
                scores = np.array(scores)
                results[str(level)] = {
                    "AP": float(average_precision_score(labels, scores)),
                    "Acc": float(accuracy_score(labels, (scores > 0.5).astype(int))),
                    "n_samples": len(labels),
                }

        return results

    def full_evaluation(
        self,
        predictions: List[Dict],
        model_fn=None,
        api_key: str = None,
    ) -> Dict[str, Dict]:
        """Run all evaluation dimensions."""
        results = {}

        print("Evaluating detection...")
        results["detection"] = self.evaluate_detection(predictions)

        print("Evaluating localization...")
        results["localization"] = self.evaluate_localization(predictions)

        print("Evaluating attributes...")
        results["attributes"] = self.evaluate_attributes(predictions)

        print("Evaluating explanations...")
        results["explanation"] = self.evaluate_explanation(predictions)

        if model_fn:
            print("Evaluating robustness...")
            for pert in ["jpeg", "blur", "resize"]:
                results[f"robust_{pert}"] = self.evaluate_robustness(model_fn, pert)

        return results
