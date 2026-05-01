#!/usr/bin/env python3
"""
XGenDet End-to-End Inference Smoke Test on REAL Images.

Demonstrates the FULL inference pipeline from raw image to all 6 outputs:
  1. Binary prediction (real/fake)
  2. Confidence score (0-1)
  3. Generator family prediction
  4. Heatmap (shape + statistics)
  5. Top-5 activated prototypes (with attribute bank labels)
  6. MLLM prompt (what Stage 2 would receive -- not executed, just shown)

Uses RANDOM weights (no trained checkpoint), so predictions are meaningless.
The point is to verify the pipeline runs end-to-end without errors.
"""

import os
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Test images: 1 REAL, 2 FAKE from different generators
# ---------------------------------------------------------------------------
TEST_IMAGES = [
    {
        "path": "/home/sachin.chaudhary/GTA/OOD_GENERATORS/stylegan/0_real/00000.png",
        "label": "REAL",
        "source": "StyleGAN dataset (real partition)",
    },
    {
        "path": "/home/sachin.chaudhary/GTA/OOD_GENERATORS/stylegan/1_fake/001024.png",
        "label": "FAKE",
        "source": "StyleGAN (GAN family)",
    },
    {
        "path": "/home/sachin.chaudhary/GTA/OOD_GENERATORS/biggan/1_fake/00000701.png",
        "label": "FAKE",
        "source": "BigGAN (GAN family, different architecture)",
    },
]

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FAMILY_NAMES = ["Real", "GAN", "Diffusion", "Autoregressive"]
ATTR_NAMES = ["Texture", "Edges", "Color", "Geometry", "Semantics", "Frequency"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def divider(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def section(title: str) -> None:
    print(f"\n  --- {title} ---")


# ---------------------------------------------------------------------------
# Step 1: Verify test images are accessible
# ---------------------------------------------------------------------------
def verify_images():
    divider("Step 1: Verifying test images")
    from PIL import Image

    valid = []
    for entry in TEST_IMAGES:
        path = entry["path"]
        try:
            img = Image.open(path).convert("RGB")
            print(f"  OK: {os.path.basename(path):30s}  size={img.size}  mode={img.mode}")
            print(f"      Source: {entry['source']}")
            print(f"      Ground truth: {entry['label']}")
            valid.append(entry)
        except Exception as e:
            print(f"  SKIP: {path} -- {e}")

    if len(valid) == 0:
        raise RuntimeError("No valid test images found!")

    print(f"\n  {len(valid)}/{len(TEST_IMAGES)} images verified")
    return valid


# ---------------------------------------------------------------------------
# Step 2: Build model with random weights
# ---------------------------------------------------------------------------
def build_model():
    divider("Step 2: Building XGenDet model (random weights)")
    from models.xgendet import XGenDet

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
    model = model.to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model built on {DEVICE}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  NOTE: Using RANDOM weights (no checkpoint loaded)")

    return model


# ---------------------------------------------------------------------------
# Step 3: Run inference on each image
# ---------------------------------------------------------------------------
def run_inference(model, test_images):
    divider("Step 3: Running inference on each image")

    from PIL import Image
    from data.augmentations import get_eval_transforms

    transform = get_eval_transforms(crop_size=224)

    results = []
    for idx, entry in enumerate(test_images):
        section(f"Image {idx+1}/{len(test_images)}: {os.path.basename(entry['path'])}")
        print(f"  Path: {entry['path']}")
        print(f"  Ground truth: {entry['label']} ({entry['source']})")

        # Load and preprocess
        img_pil = Image.open(entry["path"]).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]
        print(f"  Input tensor shape: {list(img_tensor.shape)}")

        # Run model
        t0 = time.time()
        with torch.no_grad():
            outputs = model.detect(img_tensor)
        elapsed = time.time() - t0
        print(f"  Inference time: {elapsed:.3f}s")

        # --- Output 1: Binary prediction ---
        prediction = outputs["prediction"].item()
        pred_label = "FAKE" if prediction == 1 else "REAL"
        print(f"\n  [1] Binary Prediction: {pred_label}")

        # --- Output 2: Confidence score ---
        confidence = outputs["confidence"].item()
        print(f"  [2] Confidence Score:  {confidence:.4f} (0=real, 1=fake)")

        # --- Output 3: Generator family ---
        family_idx = outputs["family"].item()
        family_name = FAMILY_NAMES[family_idx] if family_idx < len(FAMILY_NAMES) else f"Unknown({family_idx})"
        print(f"  [3] Generator Family:  {family_name} (index={family_idx})")

        # --- Output 4: Heatmap shape + statistics ---
        heatmap = outputs["heatmap"]  # [1, 1, 224, 224]
        hmap_np = heatmap.squeeze().cpu().numpy()
        print(f"  [4] Heatmap:")
        print(f"       Shape: {list(heatmap.shape)}")
        print(f"       Min:   {hmap_np.min():.6f}")
        print(f"       Max:   {hmap_np.max():.6f}")
        print(f"       Mean:  {hmap_np.mean():.6f}")
        print(f"       Std:   {hmap_np.std():.6f}")

        # --- Output 5: Top-5 activated prototypes ---
        proto_acts = outputs["proto_activations"]  # [1, 128]
        top5_vals, top5_idxs = proto_acts.topk(5, dim=-1)

        # Map prototype index to attribute bank
        from models.prototype_module import ATTRIBUTE_BANKS
        def get_bank_name(proto_idx):
            for name, (start, end) in ATTRIBUTE_BANKS.items():
                if start <= proto_idx < end:
                    return name
            return "unknown"

        print(f"  [5] Top-5 Activated Prototypes:")
        for k in range(5):
            pidx = top5_idxs[0, k].item()
            pval = top5_vals[0, k].item()
            bank = get_bank_name(pidx)
            print(f"       #{k+1}: prototype[{pidx:3d}]  activation={pval:.4f}  bank={bank}")

        # --- Output 6: Attribute scores ---
        attr_scores = outputs["attr_scores"]  # [1, 6]
        print(f"  [6] Attribute Scores:")
        for i, name in enumerate(ATTR_NAMES):
            print(f"       {name:12s}: {attr_scores[0, i].item():.4f}")

        # Store result
        results.append({
            "entry": entry,
            "img_pil": img_pil,
            "img_tensor": img_tensor,
            "outputs": outputs,
            "pred_label": pred_label,
            "confidence": confidence,
            "family_name": family_name,
            "heatmap_np": hmap_np,
        })

    return results


# ---------------------------------------------------------------------------
# Step 4: Show what MLLM prompt would look like
# ---------------------------------------------------------------------------
def show_mllm_prompt(results):
    divider("Step 4: MLLM Stage 2 Prompt Preview (NOT executed)")
    from models.mllm_module import MLLMExplainer, FORENSIC_PROMPT

    print("  The MLLM module (Qwen2.5-VL-7B) is NOT loaded in this smoke test.")
    print("  Below is what the prompt would look like for each image:\n")

    for idx, result in enumerate(results):
        entry = result["entry"]
        section(f"MLLM Prompt for Image {idx+1}: {os.path.basename(entry['path'])}")

        stage1_outputs = {
            "confidence": result["confidence"],
            "family": result["outputs"]["family"].item(),
            "attr_scores": result["outputs"]["attr_scores"][0].cpu().tolist(),
        }

        explainer = MLLMExplainer.__new__(MLLMExplainer)
        prompt = explainer._build_prompt(stage1_outputs)
        # Print prompt (indented)
        for line in prompt.split("\n"):
            print(f"    {line}")
        print()


# ---------------------------------------------------------------------------
# Step 5: Save heatmap overlays
# ---------------------------------------------------------------------------
def save_visualizations(results):
    divider("Step 5: Saving heatmap overlay visualizations")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from evaluation.visualize import denormalize_image, create_heatmap_overlay

    saved_paths = []

    for idx, result in enumerate(results):
        entry = result["entry"]
        outputs = result["outputs"]
        img_tensor = result["img_tensor"][0]  # [3, 224, 224]
        heatmap_tensor = outputs["heatmap"][0]  # [1, 224, 224]
        attr_scores = outputs["attr_scores"][0].cpu().numpy()

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 6))

        # Panel 1: Original image
        ax1 = fig.add_subplot(1, 4, 1)
        img_np = denormalize_image(img_tensor)
        ax1.imshow(img_np)
        ax1.set_title(f"Input: {entry['label']}\n({entry['source']})", fontsize=10)
        ax1.axis("off")

        # Panel 2: Heatmap overlay
        ax2 = fig.add_subplot(1, 4, 2)
        overlay = create_heatmap_overlay(img_tensor, heatmap_tensor, alpha=0.5)
        ax2.imshow(overlay)
        hmap_np = result["heatmap_np"]
        ax2.set_title(
            f"Suspicion Heatmap\n"
            f"min={hmap_np.min():.3f} max={hmap_np.max():.3f} mean={hmap_np.mean():.3f}",
            fontsize=10,
        )
        ax2.axis("off")

        # Panel 3: Prediction summary
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.axis("off")
        summary_text = (
            f"Prediction: {result['pred_label']}\n"
            f"Confidence: {result['confidence']:.4f}\n"
            f"Family: {result['family_name']}\n"
            f"\nAttribute Scores:\n"
        )
        for i, name in enumerate(ATTR_NAMES):
            summary_text += f"  {name}: {attr_scores[i]:.3f}\n"

        # Top-5 prototypes
        proto_acts = outputs["proto_activations"]
        top5_vals, top5_idxs = proto_acts.topk(5, dim=-1)
        from models.prototype_module import ATTRIBUTE_BANKS
        def get_bank(pidx):
            for n, (s, e) in ATTRIBUTE_BANKS.items():
                if s <= pidx < e:
                    return n
            return "?"
        summary_text += f"\nTop-5 Prototypes:\n"
        for k in range(5):
            pidx = top5_idxs[0, k].item()
            pval = top5_vals[0, k].item()
            summary_text += f"  [{pidx:3d}] {get_bank(pidx):10s} {pval:.3f}\n"

        ax3.text(
            0.05, 0.95, summary_text,
            transform=ax3.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )
        ax3.set_title("Prediction Summary", fontsize=10)

        # Panel 4: Attribute radar chart
        ax4 = fig.add_subplot(1, 4, 4, polar=True)
        N = len(ATTR_NAMES)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        scores = attr_scores.tolist()
        scores += scores[:1]
        ax4.fill(angles, scores, alpha=0.25, color="red")
        ax4.plot(angles, scores, "o-", linewidth=2, color="red")
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(ATTR_NAMES, fontsize=8)
        ax4.set_ylim(0, 1)
        ax4.set_title("Attribute Analysis", fontsize=10, pad=15)

        fig.suptitle(
            f"XGenDet E2E Smoke Test -- Image {idx+1}: "
            f"{os.path.basename(entry['path'])} (random weights)",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()

        fname = f"e2e_smoke_{idx+1}_{entry['label'].lower()}_{os.path.splitext(os.path.basename(entry['path']))[0]}.png"
        out_path = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")
        saved_paths.append(out_path)

    return saved_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  XGenDet End-to-End Inference Smoke Test on REAL Images")
    print("=" * 72)
    print(f"  Python:   {sys.executable}")
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CUDA:     {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
    print(f"  Device:   {DEVICE}")
    print(f"  Outputs:  {OUTPUT_DIR}")

    t_total = time.time()
    all_passed = True

    try:
        # Step 1: Verify images
        test_images = verify_images()

        # Step 2: Build model
        model = build_model()

        # Step 3: Run inference
        results = run_inference(model, test_images)

        # Step 4: Show MLLM prompt preview
        show_mllm_prompt(results)

        # Step 5: Save visualizations
        saved_paths = save_visualizations(results)

    except Exception:
        divider("FATAL ERROR")
        traceback.print_exc()
        all_passed = False

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    elapsed = time.time() - t_total
    divider("FINAL SUMMARY")

    if all_passed:
        print(f"  PIPELINE STATUS: ALL 6 OUTPUTS PRODUCED SUCCESSFULLY")
        print(f"  Total time: {elapsed:.1f}s")
        print()
        print(f"  Outputs demonstrated:")
        print(f"    [1] Binary prediction (real/fake)")
        print(f"    [2] Confidence score (0-1, calibrated via temperature scaling)")
        print(f"    [3] Generator family prediction (Real/GAN/Diffusion/Autoregressive)")
        print(f"    [4] Pixel-level suspicion heatmap (224x224)")
        print(f"    [5] Top-5 prototype activations with attribute bank labels")
        print(f"    [6] MLLM prompt preview (Stage 2, not executed)")
        print()
        print(f"  Visualization PNGs saved to: {OUTPUT_DIR}/")
        print()
        print(f"  NOTE: All predictions are from RANDOM weights.")
        print(f"  After training, these outputs will be meaningful.")
    else:
        print(f"  PIPELINE FAILED -- see error above")
        print(f"  Total time: {elapsed:.1f}s")

    print("=" * 72)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
