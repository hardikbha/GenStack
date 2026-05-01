"""
XGenDet Visualization utilities.

Functions for creating heatmap overlays, prototype visualizations,
t-SNE plots, and confidence calibration diagrams.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torchvision.transforms as transforms


CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711])


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor back to displayable image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * CLIP_STD + CLIP_MEAN
    img = np.clip(img, 0, 1)
    return img


def create_heatmap_overlay(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Create heatmap overlay on image.

    Args:
        image: [3, H, W] normalized tensor
        heatmap: [1, H, W] or [H, W] heatmap tensor
        alpha: overlay transparency
        colormap: matplotlib colormap name

    Returns:
        overlay: [H, W, 3] numpy array (0-1 range)
    """
    img = denormalize_image(image)

    hmap = heatmap.squeeze().cpu().numpy()
    # Normalize heatmap to [0, 1]
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)

    # Resize heatmap to image size if needed
    if hmap.shape != img.shape[:2]:
        from PIL import Image as PILImage
        hmap_pil = PILImage.fromarray((hmap * 255).astype(np.uint8))
        hmap_pil = hmap_pil.resize((img.shape[1], img.shape[0]), PILImage.BILINEAR)
        hmap = np.array(hmap_pil).astype(np.float32) / 255.0

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(hmap)[:, :, :3]  # [H, W, 3]

    # Blend
    overlay = alpha * heatmap_colored + (1 - alpha) * img
    overlay = np.clip(overlay, 0, 1)

    return overlay


def plot_attribute_radar(
    attr_scores: np.ndarray,
    attr_names: list = None,
    title: str = "Attribute Analysis",
    save_path: str = None,
):
    """Create radar chart of attribute scores."""
    if attr_names is None:
        attr_names = ["Texture", "Edges", "Color", "Geometry", "Semantics", "Frequency"]

    N = len(attr_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    scores = attr_scores.tolist()
    scores += scores[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, alpha=0.25, color="red")
    ax.plot(angles, scores, "o-", linewidth=2, color="red")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attr_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, pad=20)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_calibration_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
    title: str = "Calibration Diagram",
    save_path: str = None,
):
    """Plot reliability diagram for confidence calibration."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accs.append(accuracies[mask].mean())
            bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(bin_centers, bin_accs, width=1.0/n_bins, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()

    # Histogram
    ax2.bar(bin_centers, bin_counts, width=1.0/n_bins, alpha=0.7, color="coral", edgecolor="black")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def visualize_full_output(
    image: torch.Tensor,
    outputs: dict,
    save_path: str = None,
    title: str = "",
):
    """
    Create a comprehensive visualization of all XGenDet outputs.

    Shows: original image, heatmap overlay, attribute radar, prediction text.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    img = denormalize_image(image)
    axes[0].imshow(img)
    pred = "FAKE" if outputs["confidence"].item() > 0.5 else "REAL"
    conf = outputs["confidence"].item()
    axes[0].set_title(f"Prediction: {pred} ({conf:.1%})", fontsize=14)
    axes[0].axis("off")

    # Heatmap overlay
    if outputs["heatmap"] is not None:
        overlay = create_heatmap_overlay(image, outputs["heatmap"].squeeze(0))
        axes[1].imshow(overlay)
    axes[1].set_title("Suspicion Heatmap", fontsize=14)
    axes[1].axis("off")

    # Attribute radar
    attr_names = ["Texture", "Edges", "Color", "Geometry", "Semantics", "Frequency"]
    attr_scores = outputs["attr_scores"].cpu().numpy()

    N = len(attr_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    scores = attr_scores.tolist()
    scores += scores[:1]

    ax_radar = fig.add_subplot(1, 3, 3, polar=True)
    axes[2].set_visible(False)
    ax_radar.fill(angles, scores, alpha=0.25, color="red")
    ax_radar.plot(angles, scores, "o-", linewidth=2, color="red")
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(attr_names, fontsize=9)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title("Attribute Analysis", fontsize=14, pad=20)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
