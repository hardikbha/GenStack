#!/usr/bin/env python3
"""
D3 Attention Rollout + GradCAM Heatmap Generator for 10 Demo Images.

Generates heatmap visualizations from the D3 model (CLIP ViT-L/14 backbone)
for side-by-side comparison with XGenDet heatmaps.

Approach:
  1. Attention Rollout: Extracts attention weights from all 24 ViT transformer
     layers and multiplies them (with identity residual) to compute how each
     patch attends to the CLS token through the full network.
  2. GradCAM on penultimate features: Uses gradients of the D3 output w.r.t.
     the penultimate layer features to weight spatial token contributions.
  3. Combined: Average of both for a robust heatmap.

Output format matches XGenDet: 3-panel image (Original | Heatmap | Overlay).

Usage:
    conda activate D3
    python scripts/d3_gradcam_demo.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from pathlib import Path

# ============================================================================
# Config
# ============================================================================
D3_ROOT = "/home/sachin.chaudhary/GTA/D3"
CHECKPOINT = "/home/sachin.chaudhary/GTA/D3/checkpoints/train_d3/model_epoch_best.pth"

ORIGINAL_DIR = "/home/sachin.chaudhary/xgendet/eval_outputs/originals"
PREDICTIONS_FILE = "/home/sachin.chaudhary/xgendet/eval_outputs/xgendet_predictions_enhanced.jsonl"
OUTPUT_DIR = "/home/sachin.chaudhary/xgendet/demo_outputs_d3"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# The 10 demo image selections (must match create_demo_v2.py)
DEMO_SELECTIONS = [
    ("deepfake", "fake"),
    ("stylegan", "fake"),
    ("stargan", "fake"),
    ("dalle", "fake"),
    ("cyclegan", "fake"),
    ("gaugan", "fake"),
    ("ADM", "fake"),
    ("BigGAN", "fake"),
    ("ADM", "real"),
    ("stylegan2", "real"),
]


# ============================================================================
# Model Loading
# ============================================================================
def load_d3_model(checkpoint_path, device):
    """Load D3 model with CLIP ViT-L/14 backbone + attention head."""
    sys.path.insert(0, D3_ROOT)
    from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer

    print(f"  Building D3 model (CLIP ViT-L/14)...")
    model = CLIPModelShuffleAttentionPenultimateLayer(
        "ViT-L/14",
        shuffle_times=1,
        original_times=1,
        patch_size=[14],
    )

    print(f"  Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.attention_head.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    print(f"  Model loaded successfully on {device}")
    return model


def get_transform():
    """CLIP-compatible transform for D3."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


# ============================================================================
# Attention Rollout
# ============================================================================
class AttentionExtractor:
    """Hooks into CLIP ViT to extract attention weights from all layers."""

    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self._hooks = []

    def _hook_fn(self, module, input_args, output):
        """Capture attention weights from MultiheadAttention.

        nn.MultiheadAttention returns (attn_output, attn_weights) when
        need_weights=True. We monkey-patch the forward to get weights.
        """
        # output is a tuple: (attn_output, attn_weights_or_None)
        # But since need_weights=False in original code, we need a different approach
        pass

    def register_hooks(self):
        """Register forward hooks on all attention layers to capture attention maps."""
        self.attention_weights = []
        self._hooks = []

        vit = self.model.model.visual
        blocks = list(vit.transformer.resblocks.children())

        for block in blocks:
            # Replace the attention method temporarily to get weights
            hook = block.attn.register_forward_hook(self._attn_hook)
            self._hooks.append(hook)

    def _attn_hook(self, module, inputs, output):
        """Hook that re-runs attention with need_weights=True to capture them."""
        # inputs to MultiheadAttention.forward: (query, key, value, ...)
        # We need to re-run to get weights
        # Instead, we store the inputs and compute attention ourselves
        pass

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def extract_attention_rollout(model, img_tensor, device):
    """Extract attention rollout map from CLIP ViT-L/14.

    We temporarily modify the ResidualAttentionBlock.attention method to
    capture attention weights, then compute attention rollout.

    Returns:
        rollout: np.ndarray of shape [16, 16] (spatial attention map)
    """
    vit = model.model.visual
    blocks = list(vit.transformer.resblocks.children())
    num_blocks = len(blocks)

    # Storage for attention weights
    attn_weights_list = []

    # Monkey-patch each block's attention method to capture weights
    original_methods = []
    for block in blocks:
        original_methods.append(block.attention)

        def make_new_attention(blk):
            def new_attention(x):
                blk.attn_mask = (
                    blk.attn_mask.to(dtype=x.dtype, device=x.device)
                    if blk.attn_mask is not None else None
                )
                attn_output, attn_weight = blk.attn(
                    x, x, x, need_weights=True, attn_mask=blk.attn_mask
                )
                attn_weights_list.append(attn_weight.detach().cpu())
                return attn_output
            return new_attention

        block.attention = make_new_attention(block)

    try:
        # Run forward pass (the model's forward uses torch.no_grad for CLIP)
        with torch.no_grad():
            model.model.encode_image(img_tensor.to(device))
    finally:
        # Restore original methods
        for block, orig_method in zip(blocks, original_methods):
            block.attention = orig_method

    if not attn_weights_list:
        print("  WARNING: No attention weights captured!")
        return np.ones((16, 16), dtype=np.float32) * 0.5

    # Compute attention rollout
    # Each attn_weight has shape [B, num_tokens, num_tokens] = [1, 257, 257]
    # (averaged over heads by PyTorch's MultiheadAttention)
    rollout = None
    for attn_w in attn_weights_list:
        # attn_w shape: [B, 257, 257]
        attn = attn_w[0].float()  # [257, 257]

        # Add identity (residual connections)
        eye = torch.eye(attn.size(0))
        attn = 0.5 * attn + 0.5 * eye

        # Renormalize rows
        attn = attn / attn.sum(dim=-1, keepdim=True)

        if rollout is None:
            rollout = attn
        else:
            rollout = torch.matmul(attn, rollout)

    # Extract CLS token's attention to all patch tokens
    # rollout[0, :] = how CLS attends to each token after propagation
    # We want rollout[0, 1:] (skip CLS token itself)
    cls_attn = rollout[0, 1:]  # [256]

    # Reshape to 16x16 grid
    grid_size = int(cls_attn.shape[0] ** 0.5)
    spatial_attn = cls_attn.reshape(grid_size, grid_size).numpy()

    # Normalize to [0, 1]
    spatial_attn = (spatial_attn - spatial_attn.min()) / (spatial_attn.max() - spatial_attn.min() + 1e-8)

    return spatial_attn


# ============================================================================
# GradCAM on penultimate features
# ============================================================================
def extract_gradcam(model, img_tensor, device):
    """Extract GradCAM-like heatmap using gradients of D3 output w.r.t.
    the penultimate layer features (before ln_post).

    D3's ln_post hook captures features of shape [B, 1024] (CLS token only).
    For spatial GradCAM, we need to hook into the transformer output BEFORE
    the CLS token is extracted, to get [B, 257, 1024].
    """
    vit = model.model.visual

    # We'll hook the last transformer block's output to get full sequence
    spatial_features = {}

    def hook_fn(module, input, output):
        # output shape from Transformer: (out_dict, x) where x is [L, N, D]
        # But the sequential resblocks return just x
        # Actually, the Transformer.forward returns (out, x), but individual
        # ResidualAttentionBlock returns x directly
        spatial_features['output'] = output  # [L, N, D] = [257, B, 1024]

    blocks = list(vit.transformer.resblocks.children())
    last_block = blocks[-1]
    handle = last_block.register_forward_hook(hook_fn)

    # Need gradients for this pass
    img_t = img_tensor.clone().to(device)

    # Temporarily enable gradients for the attention head
    for p in model.attention_head.parameters():
        p.requires_grad_(True)

    try:
        # We need to modify the forward to NOT use torch.no_grad on CLIP
        # Instead, manually run the CLIP encoder with grads on the last block output

        # Run CLIP encoder manually to get features with grad
        x = vit.conv1(img_t.type(vit.conv1.weight.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid**2]
        x = x.permute(0, 2, 1)  # [B, grid**2, width]
        x = torch.cat([
            vit.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x
        ], dim=1)  # [B, grid**2+1, width]
        x = x + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Run all blocks except last without grad, last with grad
        with torch.no_grad():
            for block in blocks[:-1]:
                x = block(x)

        # Last block WITH gradients
        x_with_grad = x.detach().requires_grad_(True)
        x_out = last_block(x_with_grad)

        x_out_perm = x_out.permute(1, 0, 2)  # LND -> NLD
        cls_feat = vit.ln_post(x_out_perm[:, 0, :])  # [B, 1024]

        # Now get the full features for shuffled + original through attention head
        # The original D3 forward stacks shuffled + original features
        # For GradCAM, we just use the original image features
        # Simulate what the attention head would do with just original features
        # (In D3, features = [shuffled_feat, original_feat] stacked along dim=-2)

        # For simplicity, compute output using only the CLS features
        # through a simplified path
        feat_shuffled = cls_feat.detach()  # Use detached as proxy for shuffled
        feat_original = cls_feat  # This has gradients

        stacked = torch.stack([feat_shuffled, feat_original], dim=-2)  # [B, 2, 1024]
        output = model.attention_head(stacked)

        if output.shape[-1] == 2:
            output = output[:, 0]

        # Compute gradients w.r.t. the pre-CLS spatial features
        output.backward(retain_graph=True)

        if x_with_grad.grad is not None:
            # x_with_grad.grad shape: [257, B, 1024] (LND format)
            grads = x_with_grad.grad.permute(1, 0, 2)  # [B, 257, 1024]
            feats = x_out.detach().permute(1, 0, 2)  # [B, 257, 1024]

            # GradCAM: weight each channel by its gradient, take spatial tokens
            weights = grads[:, 1:, :].mean(dim=1, keepdim=True)  # [B, 1, 1024]
            cam = (feats[:, 1:, :].float() * weights.float()).sum(dim=-1)  # [B, 256]
            cam = F.relu(cam)  # Only positive contributions

            cam = cam[0].detach().cpu().numpy()  # [256]
            grid_size = int(cam.shape[0] ** 0.5)
            cam = cam.reshape(grid_size, grid_size)

            # Normalize
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            print("  WARNING: No gradients computed for GradCAM, falling back to uniform")
            cam = np.ones((16, 16), dtype=np.float32) * 0.5

    finally:
        handle.remove()
        # Restore requires_grad state
        for p in model.attention_head.parameters():
            p.requires_grad_(True)  # Keep trainable (default state)
        model.zero_grad()

    return cam


# ============================================================================
# Heatmap Visualization
# ============================================================================
def create_heatmap_image(heatmap_np, size=(224, 224)):
    """Convert a [H,W] float heatmap to a JET colormap image."""
    h, w = size
    hmap_resized = cv2.resize(heatmap_np, (w, h), interpolation=cv2.INTER_LINEAR)
    hmap_u8 = np.clip(hmap_resized * 255, 0, 255).astype(np.uint8)
    hmap_color = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    return hmap_color  # BGR


def create_overlay(original_bgr, heatmap_color, alpha=0.5):
    """Blend heatmap colormap onto original image."""
    h, w = original_bgr.shape[:2]
    hmap_resized = cv2.resize(heatmap_color, (w, h), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, hmap_resized, alpha, 0)
    return overlay


def create_3panel(original_pil, heatmap_np, label_text="", confidence=None):
    """Create a 3-panel image: Original | Heatmap | Overlay.

    Matches the format used by XGenDet heatmap outputs.
    """
    orig = np.array(original_pil.convert("RGB"))
    h, w = orig.shape[:2]

    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

    hmap_color = create_heatmap_image(heatmap_np, size=(w, h))
    overlay = create_overlay(orig_bgr, hmap_color, alpha=0.45)

    # Create 3-panel with labels
    gap = 5
    label_h = 30
    panel_w = w
    canvas_w = panel_w * 3 + gap * 2
    canvas_h = h + label_h

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Panel 1: Original
    canvas[label_h:label_h + h, 0:panel_w] = orig_bgr
    # Panel 2: Heatmap
    canvas[label_h:label_h + h, panel_w + gap:2 * panel_w + gap] = hmap_color
    # Panel 3: Overlay
    canvas[label_h:label_h + h, 2 * panel_w + 2 * gap:3 * panel_w + 2 * gap] = overlay

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    # Title with prediction info (top-left area, over the Original panel)
    if confidence is not None:
        pred_label = "FAKE" if confidence >= 0.5 else "REAL"
        color = (0, 0, 200) if confidence >= 0.5 else (0, 150, 0)
        title = f"D3: {pred_label} ({confidence * 100:.1f}%)  [{label_text}]"
    else:
        title = label_text if label_text else "D3 Attention Map"
        color = (0, 0, 0)

    cv2.putText(canvas, title, (5, 12), font, 0.4, color, thickness)

    # Panel labels (bottom line of label area)
    cv2.putText(canvas, "Original", (panel_w // 2 - 22, 26), font, 0.35, (100, 100, 100), 1)
    cv2.putText(canvas, "Attention Rollout", (panel_w + gap + panel_w // 2 - 48, 26),
                font, 0.35, (100, 100, 100), 1)
    cv2.putText(canvas, "Overlay", (2 * panel_w + 2 * gap + panel_w // 2 - 20, 26),
                font, 0.35, (100, 100, 100), 1)

    return canvas


# ============================================================================
# Demo Image Selection (matching create_demo_v2.py logic)
# ============================================================================
def find_demo_images():
    """Find the same 10 demo images as used by XGenDet demo_v2."""
    preds = []
    with open(PREDICTIONS_FILE) as f:
        for line in f:
            preds.append(json.loads(line))

    demo_images = []
    for gen, label in DEMO_SELECTIONS:
        best = None
        best_conf = -1 if label == "fake" else 2.0

        for p in preds:
            path = p["image_path"]
            if gen.lower() not in path.lower():
                continue
            if label == "fake" and p["prediction"] != "fake":
                continue
            if label == "real" and p["prediction"] != "real":
                continue

            conf = p["confidence"]
            if label == "fake" and conf > best_conf:
                best = p
                best_conf = conf
            elif label == "real" and conf < best_conf:
                best = p
                best_conf = conf

        if best:
            hm_name = os.path.basename(best.get("heatmap_path", ""))
            orig_path = os.path.join(ORIGINAL_DIR, hm_name)
            if os.path.exists(orig_path):
                demo_images.append({
                    "original_path": orig_path,
                    "gen": gen,
                    "label": label,
                    "filename": hm_name,
                    "xgendet_conf": best["confidence"],
                })
                print(f"  [OK] {gen} ({label}): {hm_name}")
            else:
                print(f"  [SKIP] {gen} ({label}): {hm_name} not found in originals/")
        else:
            print(f"  [SKIP] {gen} ({label}): no matching prediction found")

    return demo_images


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 72)
    print("  D3 Attention Rollout Heatmap Generator")
    print("  For comparison with XGenDet heatmaps")
    print("=" * 72)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load D3 model
    print("\n  Loading D3 model...")
    model = load_d3_model(CHECKPOINT, device)

    # Get transform
    transform = get_transform()

    # Find demo images
    print("\n  Finding demo images...")
    demo_images = find_demo_images()
    print(f"\n  Found {len(demo_images)} demo images")

    if not demo_images:
        print("  ERROR: No demo images found!")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each image
    print(f"\n  Generating heatmaps...")
    print("-" * 72)

    for i, item in enumerate(demo_images):
        print(f"\n  [{i + 1}/{len(demo_images)}] {item['gen']} ({item['label']})")
        print(f"    Image: {item['filename']}")

        # Load image
        pil_img = Image.open(item["original_path"]).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0)  # [1, 3, 224, 224]

        # ---- Attention Rollout ----
        print(f"    Computing attention rollout...", end="", flush=True)
        rollout_map = extract_attention_rollout(model, img_tensor, device)
        print(f" done (mean={rollout_map.mean():.3f}, max={rollout_map.max():.3f})")

        # ---- GradCAM ----
        print(f"    Computing GradCAM...", end="", flush=True)
        try:
            gradcam_map = extract_gradcam(model, img_tensor, device)
            print(f" done (mean={gradcam_map.mean():.3f}, max={gradcam_map.max():.3f})")
        except Exception as e:
            print(f" failed ({e}), using rollout only")
            gradcam_map = rollout_map

        # ---- Combined heatmap ----
        combined = 0.6 * rollout_map + 0.4 * gradcam_map
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

        # ---- D3 inference for confidence ----
        with torch.no_grad():
            output = model(img_tensor.to(device))
            if output.shape[-1] == 2:
                output = output[:, 0]
            d3_conf = output.sigmoid().item()

        pred_label = "FAKE" if d3_conf >= 0.5 else "REAL"
        gt_label = "FAKE" if item["label"] == "fake" else "REAL"
        correct = pred_label == gt_label
        check = "OK" if correct else "WRONG"
        print(f"    D3 prediction: {pred_label} ({d3_conf:.4f}) GT={gt_label} [{check}]")

        # ---- Create visualizations ----
        # Resize original to 224x224 (matching model input)
        orig_resized = pil_img.resize((224, 224), Image.LANCZOS)

        # Create 3-panel images
        gen_label = f"{item['gen']} ({item['label']})"

        # 1. Attention Rollout panel
        rollout_panel = create_3panel(orig_resized, rollout_map,
                                      label_text=gen_label, confidence=d3_conf)
        rollout_path = os.path.join(
            OUTPUT_DIR, f"demo_{i + 1:02d}_{item['gen']}_{item['label']}_rollout.png"
        )
        cv2.imwrite(rollout_path, rollout_panel)

        # 2. Combined (rollout + gradcam) panel — this is the main output
        combined_panel = create_3panel(orig_resized, combined,
                                       label_text=gen_label, confidence=d3_conf)
        combined_path = os.path.join(
            OUTPUT_DIR, f"demo_{i + 1:02d}_{item['gen']}_{item['label']}.png"
        )
        cv2.imwrite(combined_path, combined_panel)

        print(f"    Saved: {os.path.basename(combined_path)}")

    # ---- Summary ----
    print("\n" + "=" * 72)
    print(f"  DONE: {len(demo_images)} heatmaps saved to {OUTPUT_DIR}/")
    print(f"\n  Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {f}  ({size_kb:.0f} KB)")
    print("=" * 72)


if __name__ == "__main__":
    main()
