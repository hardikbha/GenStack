"""
XGenDet Demo: Single image inference with full output visualization.

Usage:
    python demo/demo.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
"""

import sys
import argparse
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xgendet import XGenDet
from data.augmentations import get_eval_transforms
from evaluation.visualize import visualize_full_output, create_heatmap_overlay
from models.prototype_module import ATTRIBUTE_BANKS


def parse_args():
    parser = argparse.ArgumentParser(description="XGenDet Demo")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="xgendet_output.png")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get("args", {})

    model = XGenDet(
        clip_model_name=model_args.get("clip_model", "ViT-L/14"),
        num_prompt_tokens=model_args.get("num_prompt_tokens", 8),
        num_prototypes=model_args.get("num_prototypes", 128),
        proto_dim=model_args.get("proto_dim", 128),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load and preprocess image
    print(f"Processing image: {args.image}")
    img = Image.open(args.image).convert("RGB")
    transform = get_eval_transforms(crop_size=224)
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor, return_heatmap=True)

    # Extract results
    confidence = outputs["confidence"].item()
    prediction = "FAKE" if confidence > 0.5 else "REAL"
    family_idx = outputs["family_logit"].argmax(dim=-1).item()
    family_names = ["Real", "GAN", "Diffusion", "Autoregressive"]
    family = family_names[family_idx]

    attr_scores = outputs["attr_scores"][0].cpu().numpy()
    attr_names = list(ATTRIBUTE_BANKS.keys())

    # Print results
    print(f"\n{'='*50}")
    print(f"  XGenDet Detection Results")
    print(f"{'='*50}")
    print(f"  Prediction:  {prediction}")
    print(f"  Confidence:  {confidence:.1%}")
    print(f"  Gen Family:  {family}")
    print(f"")
    print(f"  Attribute Scores:")
    for name, score in zip(attr_names, attr_scores):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {name:12s}: {bar} {score:.3f}")

    # Get top prototypes
    top_protos = model.prototype_module.get_top_prototypes(
        outputs["proto_activations"], top_k=5
    )[0]
    print(f"\n  Top-5 Prototype Activations:")
    for p in top_protos:
        print(f"    Proto #{p['prototype_id']:3d} ({p['attribute']:10s}): {p['activation']:.3f}")

    print(f"{'='*50}")

    # Visualize
    outputs_single = {
        "confidence": outputs["confidence"][0],
        "heatmap": outputs["heatmap"][0] if outputs["heatmap"] is not None else None,
        "attr_scores": outputs["attr_scores"][0],
    }
    visualize_full_output(
        img_tensor[0],
        outputs_single,
        save_path=args.output,
        title=f"XGenDet: {prediction} ({confidence:.1%}) | Family: {family}",
    )
    print(f"\nVisualization saved to: {args.output}")


if __name__ == "__main__":
    main()
