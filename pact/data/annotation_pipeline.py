"""
XGenDet Annotation Pipeline.

Generates training annotations for Stage 2 MLLM fine-tuning using GPT-4o API.
Pipeline: Image + Stage 1 heatmap -> GPT-4o -> structured attributes + explanation.
"""

import os
import sys
import json
import argparse
import base64
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


GPT4O_ANNOTATION_PROMPT = """You are an expert forensic analyst for AI-generated images. You will be shown an image and a heatmap overlay highlighting suspicious regions (red = more suspicious).

Detection results from an automated system:
- Prediction: {prediction} (confidence: {confidence:.1%})
- Generator family: {family}

Analyze the image and heatmap carefully. Provide your assessment:

1. Rate each attribute on a scale of 0.0 to 1.0 (higher = more indicative of AI generation):
   - texture_consistency: How consistent are textures? (1.0 = very inconsistent/artificial)
   - edge_quality: How natural are edges? (1.0 = very unnatural/blurry/sharp)
   - color_distribution: How natural is the color? (1.0 = very unnatural distribution)
   - geometric_coherence: How coherent is geometry? (1.0 = very incoherent)
   - semantic_plausibility: How plausible is the content? (1.0 = very implausible)
   - frequency_artifacts: Are there frequency artifacts? (1.0 = strong artifacts)

2. Write a 2-4 sentence forensic explanation referencing specific visual evidence.

Respond in this exact JSON format:
{{
    "attributes": {{
        "texture_consistency": 0.0,
        "edge_quality": 0.0,
        "color_distribution": 0.0,
        "geometric_coherence": 0.0,
        "semantic_plausibility": 0.0,
        "frequency_artifacts": 0.0
    }},
    "explanation": "Your explanation here."
}}"""


def image_to_base64(img: Image.Image, max_size: int = 512) -> str:
    """Convert PIL image to base64 string, resizing if needed."""
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Create heatmap overlay on image."""
    import matplotlib.cm as cm

    img_array = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    hmap = heatmap.squeeze()
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)

    if hmap.shape != img_array.shape[:2]:
        hmap_pil = Image.fromarray((hmap * 255).astype(np.uint8))
        hmap_pil = hmap_pil.resize((img_array.shape[1], img_array.shape[0]), Image.BILINEAR)
        hmap = np.array(hmap_pil).astype(np.float32) / 255.0

    cmap = cm.get_cmap("jet")
    heatmap_colored = cmap(hmap)[:, :, :3]
    overlay = alpha * heatmap_colored + (1 - alpha) * img_array
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def annotate_single_gpt4o(
    image: Image.Image,
    overlay: Image.Image,
    stage1_outputs: dict,
    api_key: str,
    model: str = "gpt-4o",
) -> dict:
    """Call GPT-4o API for a single image annotation."""
    import openai

    client = openai.OpenAI(api_key=api_key)

    confidence = stage1_outputs.get("confidence", 0.5)
    prediction = "FAKE" if confidence > 0.5 else "REAL"
    family_names = ["Real", "GAN", "Diffusion", "Autoregressive"]
    family_idx = stage1_outputs.get("family", 0)
    family = family_names[family_idx] if family_idx < len(family_names) else "Unknown"

    prompt = GPT4O_ANNOTATION_PROMPT.format(
        prediction=prediction,
        confidence=confidence,
        family=family,
    )

    img_b64 = image_to_base64(image)
    overlay_b64 = image_to_base64(overlay)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{overlay_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=500,
        temperature=0.3,
    )

    text = response.choices[0].message.content

    # Parse JSON response
    try:
        # Find JSON block in response
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(text[json_start:json_end])
            return result
    except json.JSONDecodeError:
        pass

    return {"attributes": {}, "explanation": text, "parse_error": True}


def generate_annotations(
    image_list: str,
    stage1_checkpoint: str,
    output_file: str,
    api_key: str,
    image_root: str = "",
    heatmap_output_dir: str = None,
    max_images: int = None,
    num_threads: int = 4,
    gpt_model: str = "gpt-4o",
    device: str = "cuda",
):
    """
    Main annotation pipeline.

    Args:
        image_list: Text file with image paths (one per line)
        stage1_checkpoint: Path to Stage 1 model checkpoint
        output_file: Output JSONL file path
        api_key: OpenAI API key
        image_root: Root directory for image paths
        heatmap_output_dir: Directory to save heatmap .npy files
        max_images: Max images to annotate
        num_threads: Number of parallel GPT-4o API calls
        gpt_model: GPT model to use
        device: Device for Stage 1 model
    """
    import torch
    from models.xgendet import XGenDet
    from data.augmentations import get_eval_transforms

    # Load image list
    with open(image_list, "r") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    if max_images:
        image_paths = image_paths[:max_images]

    print(f"Annotating {len(image_paths)} images...")

    # Load Stage 1 model
    print(f"Loading Stage 1 model from {stage1_checkpoint}...")
    checkpoint = torch.load(stage1_checkpoint, map_location=device)
    model_args = checkpoint.get("args", {})

    model = XGenDet(
        clip_model_name=model_args.get("clip_model", "ViT-L/14"),
        num_prompt_tokens=model_args.get("num_prompt_tokens", 8),
        num_prototypes=model_args.get("num_prototypes", 128),
        proto_dim=model_args.get("proto_dim", 128),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = get_eval_transforms(crop_size=224)

    if heatmap_output_dir:
        os.makedirs(heatmap_output_dir, exist_ok=True)

    # Process images through Stage 1
    print("Running Stage 1 inference...")
    stage1_results = []

    for img_path in tqdm(image_paths, desc="Stage 1"):
        full_path = os.path.join(image_root, img_path) if image_root else img_path
        try:
            img = Image.open(full_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor, return_heatmap=True)

            confidence = outputs["confidence"].item()
            family = outputs["family_logit"].argmax(dim=-1).item()
            attr_scores = outputs["attr_scores"][0].cpu().numpy().tolist()
            heatmap = outputs["heatmap"][0].cpu().numpy()

            # Save heatmap
            heatmap_path = None
            if heatmap_output_dir:
                heatmap_filename = Path(img_path).stem + ".npy"
                heatmap_path = os.path.join(heatmap_output_dir, heatmap_filename)
                np.save(heatmap_path, heatmap)

            stage1_results.append({
                "image_path": img_path,
                "image": img,
                "heatmap": heatmap,
                "heatmap_path": heatmap_filename if heatmap_path else None,
                "stage1_outputs": {
                    "confidence": confidence,
                    "family": family,
                    "attr_scores": attr_scores,
                },
            })
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")

    # Generate GPT-4o annotations in parallel
    print(f"\nGenerating GPT-4o annotations ({num_threads} threads)...")
    annotations = []

    def process_single(item):
        overlay = create_heatmap_overlay(item["image"], item["heatmap"])
        result = annotate_single_gpt4o(
            image=item["image"],
            overlay=overlay,
            stage1_outputs=item["stage1_outputs"],
            api_key=api_key,
            model=gpt_model,
        )
        return {
            "image_path": item["image_path"],
            "heatmap_path": item.get("heatmap_path"),
            "stage1_outputs": item["stage1_outputs"],
            "attributes": result.get("attributes", {}),
            "explanation": result.get("explanation", ""),
        }

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_single, item): i for i, item in enumerate(stage1_results)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="GPT-4o"):
            try:
                annotation = future.result()
                annotations.append(annotation)
            except Exception as e:
                idx = futures[future]
                print(f"  Error annotating image {idx}: {e}")

    # Save annotations
    with open(output_file, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")

    print(f"\nSaved {len(annotations)} annotations to {output_file}")
    return annotations


def parse_args():
    parser = argparse.ArgumentParser(description="XGenDet Annotation Pipeline")
    parser.add_argument("--image_list", type=str, required=True)
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--heatmap_dir", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--gpt_model", type=str, default="gpt-4o")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set --api_key or OPENAI_API_KEY environment variable")
        sys.exit(1)

    generate_annotations(
        image_list=args.image_list,
        stage1_checkpoint=args.stage1_checkpoint,
        output_file=args.output_file,
        api_key=api_key,
        image_root=args.image_root,
        heatmap_output_dir=args.heatmap_dir,
        max_images=args.max_images,
        num_threads=args.num_threads,
        gpt_model=args.gpt_model,
        device=args.device,
    )
