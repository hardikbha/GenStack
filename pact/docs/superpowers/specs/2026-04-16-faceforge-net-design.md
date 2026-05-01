# FaceForge-Net Design Spec
**Date:** 2026-04-16  
**Problem:** HydraFake ID accuracy stuck at 85% because FaceForensics++ (80.8%, 65% of ID split) is failing. CLIP-based models see semantics, not local forgery artifacts.  
**Goal:** Single model, no ensemble. Fix FF++ → ID accuracy 93%+. Beat Veritas (97.3% ID) on HydraFake.

---

## Root Cause Analysis

| Generator | Acc | Images | Share of ID split |
|-----------|-----|--------|------------------|
| **FaceForensics++** | **80.8%** | **8,959** | **64.8%** ← bottleneck |
| facevid2vid | 88.2% | 2,000 | 14.5% |
| Hallo2 | 97.9% | 1,660 | 12.0% |
| StyleGAN | 99.2% | 600 | 4.3% |
| Midjourney | 99.5% | 600 | 4.3% |

**Why FF++ is hard for CLIP**: FF++ pastes a fake face onto a real video. CLIP sees "a face" semantically and misses the local blending artifacts at the face-background boundary. Artifacts live at: (1) face contour seam, (2) skin texture discontinuities, (3) compression blocking around the paste region.

---

## Architecture: FaceForge-Net

### Overview
```
Input Image (full resolution)
       │
       ├─ MediaPipe FaceMesh (468 landmarks)
       │         │
       │         └─ Extract 8 region crops (landmark-guided, 224×224 each):
       │              [L_eye] [R_eye] [Nose] [Mouth] [L_ear] [R_ear]
       │              [Boundary_strip] [Full_face]
       │
       Each crop → DINOv2 ViT-L/14 (shared, LoRA r=16)
                 → 1024-d CLS token per region
       
       Full image (336px) → SRM 30 fixed filters → noise_cnn → 256-d
       
       ─── Cross-Region Attention Fusion ───
       Q = T_boundary (boundary token queries all regions)
       K,V = [T₁...T₈]
       → attended features (8 × 1024-d)
       → concat + LayerNorm → 1024-d
       + [256-d SRM]
       → MLP: Linear(1280→256)→GELU→Linear(256→64)→GELU→Linear(64→1) → sigmoid
       → confidence score (0–1)
```

### 8 Region Crops

| Region | What it captures | Crop size | Augmentation |
|--------|-----------------|-----------|--------------|
| Left eye | Iris, corner geometry, blink artifacts | 112×112 | ColorJitter, HFlip, rotation ±5° |
| Right eye | Same | 112×112 | Same |
| Nose | Blending seam, skin texture under nose | 80×80 | ColorJitter, HFlip |
| Mouth | Expression reenactment artifacts, lip seam | 112×80 | ColorJitter, HFlip |
| Left ear | Ear boundary — often sloppily pasted in swaps | 80×80 | HFlip, brightness |
| Right ear | Same | 80×80 | Same |
| **Boundary strip** | **Face contour seam — primary FF++ signal** | 224×224 (masked) | **JPEG q=30-80, blur σ=0.5-1.5, H.264 block noise, HFlip** |
| Full face | Global context (for StyleGAN/Midjourney) | 224×224 | Standard v4 augmentation |

**Boundary strip extraction**: Dilate face contour mask by 12px → erode by 12px → thin ring. This captures the exact paste boundary for FF++ face swaps.

### Backbone: DINOv2 ViT-L/14 (not CLIP)

Why DINOv2 over CLIP:
- CLIP: trained for global image-text alignment → learns semantics
- DINOv2: trained with self-supervised patch objectives → learns local texture/structure
- DINOv2 attention maps highlight local discriminative regions naturally
- Already cached at `~/.cache/huggingface/hub/facebook--dinov2-large` ✅

LoRA configuration: r=16, α=32, target=qkv projections in last 8 transformer blocks.

### Cross-Attention Fusion

The boundary strip token is used as the **query** — it interrogates other regions:
- "Does the eye region confirm the seam artifacts I see?"
- "Does the skin texture elsewhere support my suspicion?"

This lets the model weight each region's evidence dynamically per image. For full-generation fakes (StyleGAN), full_face token dominates. For face swaps (FF++), boundary + ear tokens dominate.

---

## Training Configuration

### Loss Function
```
L = L_BCE + 0.5·L_focal + 0.1·L_region_div

L_BCE:        BCE with label smoothing 0.1
L_focal:      Focal loss (γ=2) — focuses training on hard FF++ samples
L_region_div: Penalizes if all 8 regions agree unanimously (prevents shortcut)
```

### Optimizer
```
DINOv2 LoRA layers:    LR = 1e-5
Fusion head (MLP):     LR = 1e-4
SRM noise CNN:         LR = 5e-5
Optimizer: AdamW, weight_decay=0.01
Schedule: CosineAnnealingLR, T_max=total_steps, eta_min=1e-7
```

### Training Config
- Epochs: 30, patience: 10 (early stop on val AP)
- Batch: 16/GPU × 4 GPUs = 64 effective
- Gradient clipping: 1.0
- ~12-15 hours on 4× A100/H100

### Trainable Parameters
- DINOv2 LoRA (last 8 blocks): ~8.0M
- Cross-attention fusion: ~0.3M
- SRM noise CNN: ~0.2M
- MLP head: ~0.08M
- **Total: ~8.6M** (0.4% of DINOv2's 307M total)

---

## Expected Results

| Generator | v5 (CLIP, current) | FaceForge-Net (expected) |
|-----------|--------------------|--------------------------|
| **FF++** | **80.8%** | **90-92%** |
| facevid2vid | 88.2% | 92-94% |
| Hallo2 | 97.9% | 97-98% |
| StyleGAN | 99.2% | 98-99% |
| Midjourney | 99.5% | 98-99% |
| **ID Overall** | **85%** | **93-95%** |
| Veritas ID | — | 97.3% (target) |

---

## Files to Create

| File | Lines (est.) | Purpose |
|------|-------------|---------|
| `models/dino_backbone.py` | ~120 | DINOv2 ViT-L/14 + LoRA wrapper |
| `models/region_extractor.py` | ~200 | MediaPipe landmark → 8 region crops |
| `models/faceforge_net.py` | ~220 | Full architecture |
| `data/faceforge_dataset.py` | ~250 | Dataset with offline region crop caching |
| `training/train_faceforge.py` | ~350 | Training script |
| `pbs/faceforge.pbs` | ~100 | PBS job: mediapipe install + train |

**Existing code reused:**
- `models/srm_branch.py:1-120` — SRM 30 filters (unchanged)
- `data/hydrafake_dataset.py` — base data loading patterns
- `training/train_hydrafake_v3.py` — training loop structure

---

## Verification Plan

1. **Quick sanity check** (before full training): Run on 100 FF++ samples → should see boundary strip attention activate at face boundary
2. **After epoch 5**: FF++ accuracy > 83% (better than baseline)
3. **After full training**: ID accuracy > 90% (conservative), > 93% (target)
4. **Per-generator breakdown**: FF++ must be ≥ 88% for the ID improvement to hold
5. **Cross-split**: CF and CD should not degrade vs v5 (check ICLight still works)

---

## Key Dependencies

- `mediapipe` — NOT installed. Install: `pip install mediapipe` in Qwen2.5 env
- `timm` — for DINOv2 (check: `timm--eva02_large` cached, timm likely installed)
- `peft` 0.16.0 ✅
- DINOv2 large: `facebook--dinov2-large` cached ✅
- SRM branch: `models/srm_branch.py` ✅
