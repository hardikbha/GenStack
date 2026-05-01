#!/usr/bin/env python3
"""
XGenDet Stage 1 Pipeline — End-to-End Smoke Test.

Verifies:
  1. Model construction (XGenDet with CLIP ViT-L/14 backbone)
  2. Data loading from a real generator directory (ProGAN real+fake)
  3. Forward pass produces all expected output tensors
  4. Loss computation (all sub-losses)
  5. Backward pass with gradient flow to every trainable parameter
  6. Heatmap visualization saved to disk
  7. Parameter count report per component
"""

import os
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# 0. Ensure project root is on sys.path so `models`, `training`, `data`
#    are importable as packages.  Must happen BEFORE any project imports.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GENERATOR_DIR = "/home/sachin.chaudhary/GTA/final_GENERATORS/progan"
REAL_DIR = os.path.join(GENERATOR_DIR, "real")
FAKE_DIR = os.path.join(GENERATOR_DIR, "fake")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # small batch for smoke test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def divider(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n:>12,}  ({n/1_000_000:.2f}M)"
    elif n >= 1_000:
        return f"{n:>12,}  ({n/1_000:.1f}K)"
    return f"{n:>12,}"


# ---------------------------------------------------------------------------
# 1. Build model
# ---------------------------------------------------------------------------
def build_model():
    divider("1. Building XGenDet model")
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
    model.train()
    print(f"  Model built and moved to {DEVICE}")
    return model


# ---------------------------------------------------------------------------
# 2. Load a small batch of images
# ---------------------------------------------------------------------------
def load_batch():
    divider("2. Loading images from ProGAN directory")
    from data.augmentations import get_eval_transforms
    from PIL import Image

    transform = get_eval_transforms(crop_size=224)

    def load_images(folder, n):
        paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ])[:n]
        tensors = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))
        return torch.stack(tensors), paths

    half = BATCH_SIZE // 2
    real_tensors, real_paths = load_images(REAL_DIR, half)
    fake_tensors, fake_paths = load_images(FAKE_DIR, half)

    images = torch.cat([real_tensors, fake_tensors], dim=0).to(DEVICE)
    labels = torch.tensor([0]*half + [1]*half, dtype=torch.long, device=DEVICE)
    family_labels = torch.tensor([0]*half + [1]*half, dtype=torch.long, device=DEVICE)  # Real=0, GAN=1

    print(f"  Loaded {half} real + {half} fake = {BATCH_SIZE} images")
    print(f"  Image tensor shape: {images.shape}  dtype: {images.dtype}")
    print(f"  Labels: {labels.tolist()}")
    print(f"  Family labels: {family_labels.tolist()}")
    print(f"  Sample real path: {real_paths[0]}")
    print(f"  Sample fake path: {fake_paths[0]}")

    return images, labels, family_labels


# ---------------------------------------------------------------------------
# 3. Forward pass
# ---------------------------------------------------------------------------
def run_forward(model, images):
    divider("3. Running forward pass")
    t0 = time.time()
    outputs = model(images, return_heatmap=True)
    elapsed = time.time() - t0

    print(f"  Forward pass completed in {elapsed:.2f}s")
    print(f"  Output keys: {sorted(outputs.keys())}")
    for k, v in sorted(outputs.items()):
        if isinstance(v, torch.Tensor):
            print(f"    {k:25s}  shape={str(list(v.shape)):20s}  dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"    {k:25s}  dict with {len(v)} entries")
        else:
            print(f"    {k:25s}  {type(v).__name__}")

    # Basic shape assertions
    B = images.shape[0]
    assert outputs["binary_logit"].shape == (B, 1), f"binary_logit shape mismatch: {outputs['binary_logit'].shape}"
    assert outputs["confidence"].shape == (B, 1), f"confidence shape mismatch: {outputs['confidence'].shape}"
    assert outputs["family_logit"].shape == (B, 4), f"family_logit shape mismatch: {outputs['family_logit'].shape}"
    assert outputs["heatmap"].shape == (B, 1, 224, 224), f"heatmap shape mismatch: {outputs['heatmap'].shape}"
    assert outputs["proto_activations"].shape == (B, 128), f"proto_activations shape mismatch"
    assert outputs["proto_spatial_maps"].shape == (B, 128, 16, 16), f"proto_spatial_maps shape mismatch"
    assert outputs["attr_scores"].shape == (B, 6), f"attr_scores shape mismatch"
    assert outputs["proto_features"].shape == (B, 128, 128), f"proto_features shape mismatch"

    print("  All output shape assertions PASSED")
    return outputs


# ---------------------------------------------------------------------------
# 4. Compute loss
# ---------------------------------------------------------------------------
def compute_loss(model, outputs, labels, family_labels):
    divider("4. Computing loss")
    from training.losses import XGenDetLoss

    criterion = XGenDetLoss()
    losses = criterion(
        outputs=outputs,
        labels=labels,
        family_labels=family_labels,
        prototype_module=model.prototype_module,
    )

    print(f"  Loss breakdown:")
    for k, v in sorted(losses.items()):
        print(f"    {k:20s}  {v.item():.6f}")
    print(f"  Total loss: {losses['total'].item():.6f}")

    assert torch.isfinite(losses["total"]), "Total loss is not finite!"
    print("  Loss finiteness check PASSED")
    return losses


# ---------------------------------------------------------------------------
# 5. Backward pass + gradient check
# ---------------------------------------------------------------------------
def run_backward(model, losses):
    divider("5. Running backward pass + gradient check")
    t0 = time.time()
    losses["total"].backward()
    elapsed = time.time() - t0
    print(f"  Backward pass completed in {elapsed:.2f}s")

    # Collect all trainable parameters with names
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total_trainable = len(trainable)
    has_grad = 0
    no_grad = 0
    no_grad_names = []

    for name, param in trainable:
        if param.grad is not None and param.grad.abs().sum().item() > 0:
            has_grad += 1
        else:
            no_grad += 1
            no_grad_names.append(name)

    print(f"  Trainable parameters:      {total_trainable}")
    print(f"  With non-zero gradient:    {has_grad}")
    print(f"  With zero/no gradient:     {no_grad}")

    if no_grad_names:
        print(f"  Parameters without gradient flow:")
        for n in no_grad_names:
            print(f"    - {n}")
    else:
        print("  ALL trainable parameters received gradients")

    if no_grad > 0:
        print(f"  WARNING: {no_grad} trainable parameters did not receive gradients")
    else:
        print("  Gradient flow check PASSED")

    return has_grad, no_grad, no_grad_names


# ---------------------------------------------------------------------------
# 6. Save heatmap visualization
# ---------------------------------------------------------------------------
def save_heatmap(outputs, images):
    divider("6. Saving heatmap visualization")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        heatmaps = outputs["heatmap"].detach().cpu().numpy()  # [B, 1, 224, 224]
        B = heatmaps.shape[0]

        # Denormalize images for display
        clip_mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1)
        clip_std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)
        imgs_np = images.detach().cpu().numpy()
        imgs_np = imgs_np * clip_std + clip_mean
        imgs_np = np.clip(imgs_np, 0, 1).transpose(0, 2, 3, 1)  # [B, H, W, 3]

        fig, axes = plt.subplots(2, B, figsize=(4 * B, 8))
        if B == 1:
            axes = axes.reshape(2, 1)

        label_names = ["REAL", "REAL", "FAKE", "FAKE"][:B]
        for i in range(B):
            # Original image
            axes[0, i].imshow(imgs_np[i])
            conf = outputs["confidence"][i, 0].item()
            axes[0, i].set_title(f"{label_names[i]} (conf={conf:.3f})")
            axes[0, i].axis("off")

            # Heatmap overlay
            axes[1, i].imshow(imgs_np[i])
            axes[1, i].imshow(heatmaps[i, 0], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            axes[1, i].set_title(f"Heatmap")
            axes[1, i].axis("off")

        plt.suptitle("XGenDet Stage 1 Smoke Test — Heatmap Visualization", fontsize=14)
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, "smoke_test_heatmap.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Heatmap saved to: {out_path}")
        return True
    except Exception as e:
        print(f"  WARNING: Could not save heatmap visualization: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 7. Parameter count report
# ---------------------------------------------------------------------------
def report_params(model):
    divider("7. Parameter count per component")
    counts = model.count_trainable_params()

    print(f"  {'Component':<25s}  {'Trainable Params':>25s}")
    print(f"  {'-'*25}  {'-'*25}")
    for k, v in counts.items():
        print(f"  {k:<25s}  {fmt_count(v)}")

    # Also report total (trainable vs frozen) for the entire model
    total_all = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_frozen = total_all - total_trainable

    print()
    print(f"  {'TOTAL (all params)':<25s}  {fmt_count(total_all)}")
    print(f"  {'TOTAL (trainable)':<25s}  {fmt_count(total_trainable)}")
    print(f"  {'TOTAL (frozen)':<25s}  {fmt_count(total_frozen)}")
    print(f"  Trainable ratio: {100.0*total_trainable/total_all:.2f}%")

    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  XGenDet Stage 1 Pipeline — End-to-End Smoke Test")
    print("=" * 70)
    print(f"  Python:  {sys.executable}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
    print(f"  Device:  {DEVICE}")

    all_passed = True
    t_total = time.time()

    try:
        # 1. Build model
        model = build_model()

        # 2. Load images
        images, labels, family_labels = load_batch()

        # 3. Forward pass
        outputs = run_forward(model, images)

        # 4. Compute loss
        losses = compute_loss(model, outputs, labels, family_labels)

        # 5. Backward pass + gradient check
        has_grad, no_grad, no_grad_names = run_backward(model, losses)
        if no_grad > 0:
            all_passed = False

        # 6. Save heatmap
        heatmap_ok = save_heatmap(outputs, images)
        if not heatmap_ok:
            all_passed = False

        # 7. Parameter counts
        counts = report_params(model)

    except Exception as e:
        divider("FATAL ERROR")
        traceback.print_exc()
        all_passed = False

    # ---------------------------------------------------------------------------
    # Final verdict
    # ---------------------------------------------------------------------------
    elapsed_total = time.time() - t_total
    divider("RESULT")
    if all_passed:
        print(f"  SMOKE TEST PASSED  (total time: {elapsed_total:.1f}s)")
    else:
        print(f"  SMOKE TEST FAILED  (total time: {elapsed_total:.1f}s)")
    print("=" * 70)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
