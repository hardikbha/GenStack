"""
XGenDet Interactive Demo with Gradio.

Provides a web interface for:
- Upload image -> detect real/fake
- Visualize heatmap overlay
- Show attribute radar chart
- Display prototype activations
- (Optional) Generate NL explanation via MLLM
"""

import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xgendet import XGenDet
from models.prototype_module import ATTRIBUTE_BANKS
from data.augmentations import get_eval_transforms

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711])
FAMILY_NAMES = ["Real", "GAN", "Diffusion", "Autoregressive"]


def load_model(checkpoint_path, device="cuda"):
    """Load XGenDet Stage 1 model."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint.get("args", {})

    model = XGenDet(
        clip_model_name=model_args.get("clip_model", "ViT-L/14"),
        num_prompt_tokens=model_args.get("num_prompt_tokens", 8),
        num_prototypes=model_args.get("num_prototypes", 128),
        proto_dim=model_args.get("proto_dim", 128),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def create_overlay(image_np, heatmap, alpha=0.5):
    """Create heatmap overlay."""
    hmap = heatmap.squeeze()
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)

    if hmap.shape != image_np.shape[:2]:
        hmap_pil = Image.fromarray((hmap * 255).astype(np.uint8))
        hmap_pil = hmap_pil.resize((image_np.shape[1], image_np.shape[0]), Image.BILINEAR)
        hmap = np.array(hmap_pil).astype(np.float32) / 255.0

    cmap = cm.get_cmap("jet")
    heatmap_colored = cmap(hmap)[:, :, :3]
    img_float = image_np.astype(np.float32) / 255.0
    overlay = alpha * heatmap_colored + (1 - alpha) * img_float
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


def create_radar_chart(attr_scores):
    """Create attribute radar chart as numpy image."""
    attr_names = list(ATTRIBUTE_BANKS.keys())
    N = len(attr_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    scores = attr_scores.tolist()
    scores += scores[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, alpha=0.25, color="red")
    ax.plot(angles, scores, "o-", linewidth=2, color="red")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([n.capitalize() for n in attr_names], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Attribute Analysis", fontsize=12, pad=15)
    plt.tight_layout()

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img


def detect(image, model, device):
    """Run detection on a single image."""
    transform = get_eval_transforms(crop_size=224)
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor, return_heatmap=True)

    confidence = outputs["confidence"].item()
    prediction = "FAKE" if confidence > 0.5 else "REAL"
    family_idx = outputs["family_logit"].argmax(dim=-1).item()
    family = FAMILY_NAMES[family_idx]
    attr_scores = outputs["attr_scores"][0].cpu().numpy()
    heatmap = outputs["heatmap"][0, 0].cpu().numpy()

    # Get top prototypes
    top_protos = model.prototype_module.get_top_prototypes(
        outputs["proto_activations"], top_k=5
    )[0]

    return {
        "prediction": prediction,
        "confidence": confidence,
        "family": family,
        "attr_scores": attr_scores,
        "heatmap": heatmap,
        "top_protos": top_protos,
    }


def build_app(checkpoint_path, device="cuda"):
    """Build Gradio app."""
    import gradio as gr

    model, dev = load_model(checkpoint_path, device)

    def process_image(img):
        if img is None:
            return None, None, "Upload an image to analyze."

        pil_img = Image.fromarray(img).convert("RGB")
        results = detect(pil_img, model, dev)

        # Heatmap overlay
        overlay = create_overlay(np.array(pil_img), results["heatmap"])

        # Radar chart
        radar = create_radar_chart(results["attr_scores"])

        # Text summary
        attr_names = list(ATTRIBUTE_BANKS.keys())
        proto_lines = "\n".join(
            f"  Proto #{p['prototype_id']:3d} ({p['attribute']:10s}): {p['activation']:.3f}"
            for p in results["top_protos"]
        )

        summary = (
            f"Prediction: {results['prediction']}\n"
            f"Confidence: {results['confidence']:.1%}\n"
            f"Generator Family: {results['family']}\n\n"
            f"Attribute Scores:\n"
        )
        for name, score in zip(attr_names, results["attr_scores"]):
            bar = "=" * int(score * 20) + " " * (20 - int(score * 20))
            summary += f"  {name:12s}: [{bar}] {score:.3f}\n"
        summary += f"\nTop-5 Prototypes:\n{proto_lines}"

        return overlay, radar, summary

    with gr.Blocks(title="XGenDet: Explainable AI-Generated Image Detector") as app:
        gr.Markdown("# XGenDet: Explainable Generalized AI-Generated Image Detection")
        gr.Markdown("Upload an image to detect if it's real or AI-generated, with full explainability.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Image", type="numpy")
                detect_btn = gr.Button("Analyze", variant="primary")

            with gr.Column(scale=1):
                overlay_output = gr.Image(label="Suspicion Heatmap")

        with gr.Row():
            with gr.Column(scale=1):
                radar_output = gr.Image(label="Attribute Analysis")
            with gr.Column(scale=1):
                text_output = gr.Textbox(label="Detection Results", lines=15)

        detect_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[overlay_output, radar_output, text_output],
        )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args.checkpoint, args.device)
    app.launch(server_port=args.port, share=args.share)
