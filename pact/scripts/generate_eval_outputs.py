"""
Generate heatmaps and predictions from best XGenDet checkpoint on a stratified eval set.

Samples 5 real + 5 fake from each OOD generator (12 generators)
and 5 real + 5 fake from each ID generator (8 generators).
Total: ~200 images.

Outputs:
- eval_outputs/predictions.jsonl  (one JSON line per image)
- eval_outputs/heatmaps/          (heatmap overlays as PNG)
- eval_outputs/originals/         (resized 224x224 originals)
"""

import os
import sys
import json
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/home/sachin.chaudhary/xgendet"
DATA_ROOT = "/home/sachin.chaudhary/GTA"
CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "xgendet_stage1", "best_model.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "eval_outputs")
HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps")
ORIGINAL_DIR = os.path.join(OUTPUT_DIR, "originals")
PRED_FILE = os.path.join(OUTPUT_DIR, "predictions.jsonl")

os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(ORIGINAL_DIR, exist_ok=True)

# Add project root to path so we can import the model package
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
SAMPLES_PER_CLASS = 5  # 5 real + 5 fake per generator

FAMILY_NAMES = {0: "Real", 1: "GAN", 2: "Diffusion", 3: "Autoregressive"}
ATTRIBUTE_NAMES = ["texture", "edges", "color", "geometry", "semantics", "frequency"]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# OOD generators:  OOD_GENERATORS/<name>/0_real  and  1_fake
# ---------------------------------------------------------------------------
OOD_GENERATORS = [
    "biggan", "crn", "cyclegan", "dalle", "deepfake", "gaugan",
    "imle", "san", "seeingdark", "stargan", "stylegan", "stylegan2",
    "whichfaceisreal",
]

# ---------------------------------------------------------------------------
# ID generators with their specific subdirectory layouts
# Most follow:  ID_GENERATORS/<Name>/<subdir>/val/ai  and  val/nature
# progan is special: ID_GENERATORS/progan/<class>/0_real  and  1_fake
# ---------------------------------------------------------------------------
ID_GENERATOR_SUBDIRS = {
    "ADM": "imagenet_ai_0508_adm",
    "BigGAN": "imagenet_ai_0419_biggan",
    "glide": "imagenet_glide",
    "LDM": "imagenet_ai_0419_sdv4",
    "Midjourney": "imagenet_midjourney",
    "VQDM": "imagenet_ai_0419_vqdm",
    "wukong": "imagenet_ai_0424_wukong",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scan_images(folder, max_depth=1):
    """Collect image paths from a folder (non-recursive by default)."""
    images = []
    if not os.path.isdir(folder):
        return images
    for entry in os.listdir(folder):
        full = os.path.join(folder, entry)
        if os.path.isfile(full) and os.path.splitext(entry)[1].lower() in IMAGE_EXTENSIONS:
            images.append(full)
    return sorted(images)


def sample_paths(folder, n):
    """Return up to n randomly-sampled image paths from folder."""
    imgs = scan_images(folder)
    if len(imgs) == 0:
        return []
    random.shuffle(imgs)
    return imgs[:n]


def build_eval_set():
    """
    Build the stratified evaluation image list.
    Returns list of dicts: {path, label (0/1), generator, source ('ID'/'OOD')}.
    """
    eval_set = []

    # --- OOD generators ---
    for gen in OOD_GENERATORS:
        real_dir = os.path.join(DATA_ROOT, "OOD_GENERATORS", gen, "0_real")
        fake_dir = os.path.join(DATA_ROOT, "OOD_GENERATORS", gen, "1_fake")

        for path in sample_paths(real_dir, SAMPLES_PER_CLASS):
            eval_set.append({"path": path, "label": 0, "generator": gen, "source": "OOD"})
        for path in sample_paths(fake_dir, SAMPLES_PER_CLASS):
            eval_set.append({"path": path, "label": 1, "generator": gen, "source": "OOD"})

    # --- ID generators (standard layout) ---
    for gen_name, subdir in ID_GENERATOR_SUBDIRS.items():
        real_dir = os.path.join(DATA_ROOT, "ID_GENERATORS", gen_name, subdir, "val", "nature")
        fake_dir = os.path.join(DATA_ROOT, "ID_GENERATORS", gen_name, subdir, "val", "ai")

        for path in sample_paths(real_dir, SAMPLES_PER_CLASS):
            eval_set.append({"path": path, "label": 0, "generator": gen_name, "source": "ID"})
        for path in sample_paths(fake_dir, SAMPLES_PER_CLASS):
            eval_set.append({"path": path, "label": 1, "generator": gen_name, "source": "ID"})

    # --- ID: progan (class-based layout) ---
    progan_root = os.path.join(DATA_ROOT, "ID_GENERATORS", "progan")
    if os.path.isdir(progan_root):
        # Collect all real and all fake images across classes, then sample
        all_real, all_fake = [], []
        for cls_name in sorted(os.listdir(progan_root)):
            cls_dir = os.path.join(progan_root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            all_real.extend(scan_images(os.path.join(cls_dir, "0_real")))
            all_fake.extend(scan_images(os.path.join(cls_dir, "1_fake")))
        random.shuffle(all_real)
        random.shuffle(all_fake)
        for path in all_real[:SAMPLES_PER_CLASS]:
            eval_set.append({"path": path, "label": 0, "generator": "progan", "source": "ID"})
        for path in all_fake[:SAMPLES_PER_CLASS]:
            eval_set.append({"path": path, "label": 1, "generator": "progan", "source": "ID"})

    print(f"Eval set: {len(eval_set)} images "
          f"({sum(1 for e in eval_set if e['source']=='OOD')} OOD, "
          f"{sum(1 for e in eval_set if e['source']=='ID')} ID)")
    return eval_set


def load_model():
    """Instantiate XGenDet and load checkpoint weights."""
    from models.xgendet import XGenDet

    print("Instantiating XGenDet model...")
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

    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    # Handle potential key mismatches from DataParallel wrapping
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")
        cleaned[new_k] = v

    model.load_state_dict(cleaned, strict=False)
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')})")
    return model


def load_and_preprocess(path):
    """Load an image, convert to RGB, resize to 224x224, return (tensor, pil_image)."""
    img = Image.open(path)
    # Handle RGBA, palette, etc.
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Keep a resized copy for visualization
    img_resized = img.resize((224, 224), Image.BILINEAR)

    # Build tensor with CLIP normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    tensor = transform(img)  # [3, 224, 224]
    return tensor, img_resized


def save_heatmap_overlay(heatmap_np, original_pil, save_path):
    """
    Save a heatmap overlay on the original image.
    heatmap_np: 2-D numpy array [H, W] in [0, 1] range.
    original_pil: PIL Image (224x224).
    Red = suspicious (high values), Blue = normal (low values).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: original
    axes[0].imshow(original_pil)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    # Middle: heatmap only
    im = axes[1].imshow(heatmap_np, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Heatmap", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Right: overlay
    orig_arr = np.array(original_pil).astype(np.float32) / 255.0
    # Apply jet colormap to heatmap
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_np)[:, :, :3]  # [H, W, 3], drop alpha
    # Blend: 60% original + 40% heatmap
    overlay = 0.6 * orig_arr + 0.4 * heatmap_colored.astype(np.float32)
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (red=suspicious)", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_inference(model, eval_set):
    """Run model inference on all images and save outputs."""
    predictions = []

    total = len(eval_set)
    for idx, item in enumerate(eval_set):
        path = item["path"]
        label = item["label"]
        generator = item["generator"]
        source = item["source"]

        # Unique filename for outputs
        basename = os.path.splitext(os.path.basename(path))[0]
        uid = f"{source}_{generator}_{label}_{basename}"
        # Sanitize
        uid = uid.replace("/", "_").replace(" ", "_")

        try:
            tensor, img_pil = load_and_preprocess(path)
        except Exception as e:
            print(f"  [{idx+1}/{total}] SKIP (load error): {path} - {e}")
            continue

        # Add batch dimension
        x = tensor.unsqueeze(0)  # [1, 3, 224, 224]

        with torch.no_grad():
            outputs = model.forward(x, return_heatmap=True)

        # Extract outputs
        confidence = outputs["confidence"].squeeze().item()
        binary_logit = outputs["binary_logit"].squeeze().item()
        prediction = 1 if confidence > 0.5 else 0
        family_logits = outputs["family_logit"].squeeze()  # [4]
        family_pred = family_logits.argmax().item()
        family_probs = F.softmax(family_logits, dim=0).tolist()
        attr_scores = outputs["attr_scores"].squeeze().tolist()  # [6]
        proto_activations = outputs["proto_activations"].squeeze().tolist()  # [128]

        # Heatmap: [1, 1, 224, 224] -> [224, 224] numpy
        heatmap = outputs["heatmap"].squeeze().cpu().numpy()  # [224, 224]
        # Normalize to [0, 1]
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max - h_min > 1e-8:
            heatmap_norm = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap_norm = np.zeros_like(heatmap)

        # Save heatmap overlay
        heatmap_path = os.path.join(HEATMAP_DIR, f"{uid}.png")
        save_heatmap_overlay(heatmap_norm, img_pil, heatmap_path)

        # Save original (resized)
        original_path = os.path.join(ORIGINAL_DIR, f"{uid}.png")
        img_pil.save(original_path)

        # Build prediction record
        record = {
            "uid": uid,
            "source_path": path,
            "source": source,
            "generator": generator,
            "ground_truth": label,
            "prediction": prediction,
            "confidence": round(confidence, 5),
            "binary_logit": round(binary_logit, 5),
            "family_pred": family_pred,
            "family_name": FAMILY_NAMES.get(family_pred, "Unknown"),
            "family_probs": [round(p, 5) for p in family_probs],
            "attr_scores": {
                name: round(score, 5)
                for name, score in zip(ATTRIBUTE_NAMES, attr_scores)
            },
            "proto_activations_top10": _top_k_protos(proto_activations, k=10),
            "heatmap_file": os.path.basename(heatmap_path),
            "original_file": os.path.basename(original_path),
            "heatmap_mean": round(float(heatmap_norm.mean()), 5),
            "heatmap_max": round(float(heatmap_norm.max()), 5),
        }
        predictions.append(record)

        correct = "OK" if prediction == label else "WRONG"
        label_str = "FAKE" if label == 1 else "REAL"
        pred_str = "FAKE" if prediction == 1 else "REAL"
        print(f"  [{idx+1}/{total}] {source}/{generator} | GT={label_str} Pred={pred_str} "
              f"conf={confidence:.3f} family={FAMILY_NAMES.get(family_pred, '?')} [{correct}]")

    return predictions


def _top_k_protos(activations, k=10):
    """Return top-k prototype activations with bank labels."""
    from models.prototype_module import ATTRIBUTE_BANKS
    indexed = [(i, v) for i, v in enumerate(activations)]
    indexed.sort(key=lambda x: x[1], reverse=True)
    results = []
    for proto_id, val in indexed[:k]:
        bank = "unknown"
        for name, (start, end) in ATTRIBUTE_BANKS.items():
            if start <= proto_id < end:
                bank = name
                break
        results.append({"id": proto_id, "activation": round(val, 5), "bank": bank})
    return results


def print_summary(predictions):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    total = len(predictions)
    correct = sum(1 for p in predictions if p["prediction"] == p["ground_truth"])
    print(f"Overall accuracy: {correct}/{total} = {100*correct/total:.1f}%")

    # Per-source
    for src in ["OOD", "ID"]:
        subset = [p for p in predictions if p["source"] == src]
        if not subset:
            continue
        c = sum(1 for p in subset if p["prediction"] == p["ground_truth"])
        print(f"  {src}: {c}/{len(subset)} = {100*c/len(subset):.1f}%")

    # Per-generator
    print("\nPer-generator accuracy:")
    generators = sorted(set(p["generator"] for p in predictions))
    for gen in generators:
        subset = [p for p in predictions if p["generator"] == gen]
        c = sum(1 for p in subset if p["prediction"] == p["ground_truth"])
        src = subset[0]["source"]
        print(f"  [{src}] {gen:20s}: {c}/{len(subset)} = {100*c/len(subset):.1f}%")

    # Per-class
    for lbl, name in [(0, "Real"), (1, "Fake")]:
        subset = [p for p in predictions if p["ground_truth"] == lbl]
        if not subset:
            continue
        c = sum(1 for p in subset if p["prediction"] == p["ground_truth"])
        print(f"\n  {name} images: {c}/{len(subset)} = {100*c/len(subset):.1f}%")

    # Mean confidence for correct/incorrect
    correct_confs = [p["confidence"] for p in predictions if p["prediction"] == p["ground_truth"]]
    wrong_confs = [p["confidence"] for p in predictions if p["prediction"] != p["ground_truth"]]
    if correct_confs:
        print(f"\n  Mean confidence (correct): {np.mean(correct_confs):.4f}")
    if wrong_confs:
        print(f"  Mean confidence (wrong):   {np.mean(wrong_confs):.4f}")

    print(f"\nPredictions saved to: {PRED_FILE}")
    print(f"Heatmaps saved to:   {HEATMAP_DIR}/")
    print(f"Originals saved to:  {ORIGINAL_DIR}/")
    print("=" * 70)


def main():
    print("=" * 70)
    print("XGenDet Evaluation Output Generator")
    print("=" * 70)

    # 1. Build stratified eval set
    print("\n[Step 1] Building stratified evaluation set...")
    eval_set = build_eval_set()

    # 2. Load model
    print("\n[Step 2] Loading model...")
    model = load_model()

    # 3. Run inference
    print(f"\n[Step 3] Running inference on {len(eval_set)} images (CPU)...")
    predictions = run_inference(model, eval_set)

    # 4. Save predictions as JSONL
    print(f"\n[Step 4] Saving predictions to {PRED_FILE}...")
    with open(PRED_FILE, "w") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")

    # 5. Summary
    print_summary(predictions)


if __name__ == "__main__":
    main()
