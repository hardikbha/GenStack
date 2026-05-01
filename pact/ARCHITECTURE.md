# XGenDet: End-to-End Architecture Diagram

## Complete Pipeline Overview

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                              XGenDet: End-to-End Pipeline                          ║
║                 Explainable Generalized AI-Generated Image Detection               ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

                                 Input Image (224x224)
                                         │
                          ┌──────────────┼──────────────┐
                          │              │              │
                          ▼              │              ▼
                   ┌─────────────┐       │       ┌─────────────┐
                   │  Original   │       │       │   Shuffled   │
                   │   Image     │       │       │   (D3 patch  │
                   │             │       │       │   shuffle)   │
                   └──────┬──────┘       │       └──────┬──────┘
                          │              │              │
╔═════════════════════════╪══════════════╪══════════════╪════════════════════════════╗
║  STAGE 1: Detection     │    FROZEN CLIP ViT-L/14    │              ~1.2M params  ║
║  (~20ms inference)      │    (304M params, frozen)    │                            ║
╠═════════════════════════╪══════════════╪══════════════╪════════════════════════════╣
║                         │              │              │                            ║
║                         ▼              │              ▼                            ║
║              ┌───────────────────┐     │   ┌───────────────────┐                  ║
║              │ Patch Embedding   │     │   │ Patch Embedding   │                  ║
║              │ Conv(3,1024,14,14)│     │   │ Conv(3,1024,14,14)│                  ║
║              └────────┬──────────┘     │   └────────┬──────────┘                  ║
║                       │                │            │                              ║
║                       ▼                │            ▼                              ║
║              ┌─────────────────────────┤   ┌──────────────────┐                   ║
║              │ [CLS] + 256 patches     │   │ [CLS] + 256 pats │                   ║
║              │ + positional embed      │   │ + positional emb  │                   ║
║              └────────┬────────────────┤   └────────┬─────────┘                   ║
║                       │                │            │                              ║
║   ┌───────────────────▼───┐            │            │                              ║
║   │  ★ Forgery Prompt     │            │            │                              ║
║   │  Tokens (8 tokens)    │◄── Learnable (8K params)│                              ║
║   │  Injected after CLS   │            │            │                              ║
║   └───────────────────┬───┘            │            │                              ║
║                       │                │            │                              ║
║                       ▼                │            ▼                              ║
║              ┌──────────────────┐      │   ┌──────────────────┐                   ║
║              │   ViT Transformer │      │   │  ViT Transformer │                   ║
║              │   24 layers       │      │   │  24 layers       │                   ║
║              │   ★ LN tuned     │◄──────┤   │  ★ LN tuned     │                   ║
║              │   (110K params)   │      │   │                  │                   ║
║              │                  │      │   │                  │                   ║
║              │  Attention hooks │      │   │  Attention hooks │                   ║
║              │  @layers 6,12,   │      │   │  @layers 6,12,   │                   ║
║              │   18,23          │      │   │   18,23          │                   ║
║              └──┬──────────┬────┘      │   └──┬──────────┬────┘                   ║
║                 │          │           │      │          │                         ║
║          ┌──────┘    ┌─────┘           │  ┌───┘    ┌─────┘                         ║
║          ▼           ▼                 │  ▼        ▼                               ║
║     ┌─────────┐ ┌───────────┐          │ ┌────────┐┌───────────┐                   ║
║     │CLS_orig │ │256 Patch  │          │ │CLS_shuf││256 Patch  │                   ║
║     │ (768d)  │ │Tokens     │          │ │ (768d) ││Tokens     │                   ║
║     └────┬────┘ │(256x1024) │          │ └───┬────┘│(256x1024) │                   ║
║          │      └─────┬─────┘          │     │     └─────┬─────┘                   ║
║          │            │                │     │           │                         ║
║          │            │   ┌────────────┘     │           │                         ║
║          │            │   │                  │           │                         ║
║          │            ▼   │                  │           │                         ║
║          │  ╔══════════════════════════════╗  │           │                         ║
║          │  ║  ★ PGAD Module (214K params) ║  │           │                         ║
║          │  ╠══════════════════════════════╣  │           │                         ║
║          │  ║                              ║  │           │                         ║
║          │  ║  128 Prototypes (6 banks):   ║  │           │                         ║
║          │  ║  ┌─────────────────────────┐ ║  │           │                         ║
║          │  ║  │ Texture   (proto 0-21)  │ ║  │           │                         ║
║          │  ║  │ Edges     (proto 21-43) │ ║  │           │                         ║
║          │  ║  │ Color     (proto 43-64) │ ║  │           │                         ║
║          │  ║  │ Geometry  (proto 64-86) │ ║  │           │                         ║
║          │  ║  │ Semantics (proto 86-107)│ ║  │           │                         ║
║          │  ║  │ Frequency (proto107-128)│ ║  │           │                         ║
║          │  ║  └─────────────────────────┘ ║  │           │                         ║
║          │  ║                              ║  │           │                         ║
║          │  ║  Cross-Attention (4 heads):  ║  │           │                         ║
║          │  ║  Q = Prototypes (128x128)    ║  │           │                         ║
║          │  ║  K,V = Patches (256x1024     ║  │           │                         ║
║          │  ║         -> projected 128)    ║  │           │                         ║
║          │  ║                              ║  │           │                         ║
║          │  ╚══════╤══════════╤════════╤═══╝  │           │                         ║
║          │         │          │        │      │           │                         ║
║          │         ▼          ▼        ▼      │           │                         ║
║          │    ┌─────────┐┌────────┐┌──────┐   │           │                         ║
║          │    │Proto Act.││Spatial ││Attr  │   │           │                         ║
║          │    │[B, 128]  ││Maps   ││Scores│   │           │                         ║
║          │    │          ││[B,128,││[B, 6]│   │           │                         ║
║          │    │          ││16,16] ││      │   │           │                         ║
║          │    └────┬─────┘└───┬───┘└──┬───┘   │           │                         ║
║          │         │          │       │       │           │                         ║
║          │         │          │       │       │           │                         ║
║          │         │     ┌────┘       │       │           │                         ║
║          │         │     │            │       │           │                         ║
║          │         │     ▼            │       │           │                         ║
║          │         │  ╔════════════════════════════════╗  │                         ║
║          │         │  ║  Heatmap Generator (1.3K)      ║  │                         ║
║          │         │  ╠════════════════════════════════╣  │                         ║
║          │         │  ║  Three-Source Fusion:           ║  │                         ║
║          │         │  ║                                ║  │                         ║
║          │         │  ║  1. Attention Rollout           ║◄─┤ Attn maps              ║
║          │         │  ║     (forgery prompt -> patches) ║  │ from hooks              ║
║          │         │  ║                                ║  │                         ║
║          │         │  ║  2. Shuffle Difference          ║◄─┤ Orig vs shuf           ║
║          │         │  ║     |attn_orig - attn_shuf|    ║  │ attention               ║
║          │         │  ║                                ║  │                         ║
║          │         │  ║  3. Proto Spatial Maps          ║◄─┘                        ║
║          │         │  ║     (weighted by activations)   ║                           ║
║          │         │  ║                                ║                           ║
║          │         │  ║  Conv2d(3->16->1) fusion       ║                           ║
║          │         │  ║  Upsample to 224x224           ║                           ║
║          │         │  ╚══════════════╤═════════════════╝                           ║
║          │         │                 │                                              ║
║          │         │                 ▼                                              ║
║          │         │          ┌─────────────┐                                      ║
║          │         │          │  Heatmap     │                                      ║
║          │         │          │ [B,1,224,224]│                                      ║
║          │         │          └──────┬──────┘                                      ║
║          │         │                 │                                              ║
║          │         │                 ▼                                              ║
║          │         │          ┌──────────────┐                                     ║
║          │         │          │Heatmap Stats │                                     ║
║          │         │          │  (32-dim)    │                                     ║
║          │         │          └──────┬───────┘                                     ║
║          │         │                 │                                              ║
║          ▼         ▼                 ▼                ▼                             ║
║     ╔══════════════════════════════════════════════════════════╗                    ║
║     ║  ★ Classification Head (907K params)                     ║                    ║
║     ╠══════════════════════════════════════════════════════════╣                    ║
║     ║                                                          ║                    ║
║     ║  Input = concat[CLS_orig, CLS_shuf, Proto_pool,         ║                    ║
║     ║                  Heatmap_stats, Attr_scores]             ║                    ║
║     ║                                                          ║                    ║
║     ║  ┌────────────────────┐    ┌────────────────────┐        ║                    ║
║     ║  │  Binary Head       │    │  Family Head        │        ║                    ║
║     ║  │  Linear->GELU->    │    │  Linear->GELU->     │        ║                    ║
║     ║  │  Dropout->LN->     │    │  Dropout->LN->      │        ║                    ║
║     ║  │  Linear->GELU->    │    │  Linear (->4)       │        ║                    ║
║     ║  │  Linear (->1)      │    │                     │        ║                    ║
║     ║  └─────────┬──────────┘    └──────────┬──────────┘        ║                    ║
║     ║            │                           │                  ║                    ║
║     ║            ▼                           ▼                  ║                    ║
║     ║     ┌──────────────┐           ┌──────────────┐           ║                    ║
║     ║     │binary_logit  │           │family_logit  │           ║                    ║
║     ║     │  [B, 1]      │           │  [B, 4]      │           ║                    ║
║     ║     └──────┬───────┘           └──────────────┘           ║                    ║
║     ║            │                                              ║                    ║
║     ║            ▼ ÷ temperature                                ║                    ║
║     ║     ┌──────────────┐                                      ║                    ║
║     ║     │ Calibrated   │                                      ║                    ║
║     ║     │ Confidence   │                                      ║                    ║
║     ║     │ sigmoid(l/T) │                                      ║                    ║
║     ║     │  [B, 1]      │                                      ║                    ║
║     ║     └──────────────┘                                      ║                    ║
║     ╚══════════════════════════════════════════════════════════╝                    ║
║                                                                                     ║
╠═════════════════════════════════════════════════════════════════════════════════════╣
║                           STAGE 1 OUTPUTS (6 outputs)                              ║
╠═════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                     ║
║  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐  ║
║  │Real/Fake │ │Calibrated│ │Generator │ │ Spatial  │ │Attribute │ │  Prototype  │  ║
║  │Binary    │ │Confidence│ │Family    │ │ Heatmap  │ │ Scores   │ │ Activations │  ║
║  │Prediction│ │  0.0-1.0 │ │GAN/Diff/ │ │ 224x224  │ │  6 dims  │ │  128 dims   │  ║
║  │          │ │          │ │AR/Real   │ │ (WHERE)  │ │  (WHY)   │ │  (WHAT)     │  ║
║  └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬──────┘  ║
║        │           │            │             │            │              │         ║
╚════════╪═══════════╪════════════╪═════════════╪════════════╪══════════════╪═════════╝
         │           │            │             │            │              │
         └─────┬─────┴────────────┴──────┬──────┘            │              │
               │                         │                   │              │
               ▼                         ▼                   │              │
         ┌───────────────┐        ┌──────────────┐           │              │
         │ Structured    │        │ Heatmap      │           │              │
         │ Prefix for    │        │ Overlay on   │           │              │
         │ MLLM Prompt   │        │ Original Img │           │              │
         └───────┬───────┘        └──────┬───────┘           │              │
                 │                       │                   │              │
╔════════════════╪═══════════════════════╪═══════════════════╪══════════════╪═════════╗
║  STAGE 2:      │  MLLM Explainability  │                   │              │         ║
║  (Optional)    │  (~2-5s inference)    │                   │              │         ║
╠════════════════╪═══════════════════════╪═══════════════════╧══════════════╧═════════╣
║                │                       │                                            ║
║                ▼                       ▼                                            ║
║       ┌─────────────────────────────────────────────┐                              ║
║       │  Qwen2.5-VL-7B-Instruct + LoRA (r=16)      │                              ║
║       │                                             │                              ║
║       │  Input:                                     │                              ║
║       │  ┌─────────┐ ┌─────────┐ ┌───────────────┐  │                              ║
║       │  │Original │ │Heatmap  │ │ Forensic      │  │                              ║
║       │  │Image    │ │Overlay  │ │ Prompt +      │  │                              ║
║       │  │         │ │         │ │ Stage1 Output │  │                              ║
║       │  └─────────┘ └─────────┘ └───────────────┘  │                              ║
║       │                                             │                              ║
║       │  LoRA targets: q_proj, v_proj               │                              ║
║       │  Only ~8M params trained (0.1% of 7B)       │                              ║
║       └──────────────────┬──────────────────────────┘                              ║
║                          │                                                          ║
║                    ┌─────┴─────┐                                                   ║
║                    │           │                                                    ║
║                    ▼           ▼                                                    ║
║            ┌──────────────┐ ┌──────────────────────────────────┐                   ║
║            │ Refined      │ │ Natural Language Explanation     │                   ║
║            │ Attribute    │ │                                  │                   ║
║            │ Scores       │ │ "This image exhibits strong     │                   ║
║            │              │ │  texture inconsistencies in the │                   ║
║            │ texture: 0.82│ │  upper-right quadrant, where    │                   ║
║            │ edges:   0.45│ │  the heatmap shows high         │                   ║
║            │ color:   0.31│ │  activation. The frequency      │                   ║
║            │ geometry:0.67│ │  domain reveals periodic        │                   ║
║            │ semantic:0.23│ │  artifacts consistent with      │                   ║
║            │ frequency:0.9│ │  diffusion model generation."   │                   ║
║            └──────────────┘ └──────────────────────────────────┘                   ║
║                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════╝


         ╔══════════════════════════════════════════════╗
         ║           FINAL OUTPUT (8 outputs)           ║
         ╠══════════════════════════════════════════════╣
         ║                                              ║
         ║  1. Prediction:     FAKE                     ║
         ║  2. Confidence:     92.3% (calibrated)       ║
         ║  3. Generator:      Diffusion                ║
         ║  4. Heatmap:        224x224 spatial map       ║
         ║  5. Attr Scores:    6 interpretable dims      ║
         ║  6. Proto Activs:   128 prototype scores      ║
         ║  7. Refined Attrs:  6 MLLM-refined scores     ║
         ║  8. Explanation:    NL forensic reasoning      ║
         ║                                              ║
         ╚══════════════════════════════════════════════╝
```

---

## Training Pipeline

```
╔═══════════════════════════════════════════════════════════════════╗
║                    TRAINING PIPELINE                              ║
╚═══════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│                     STAGE 1 TRAINING                            │
│                   (~2-4 hours on H100)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Training Data (8 ID generators):                               │
│  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌─────┐┌────┐│
│  │ ADM  ││BigGAN││GLIDE ││ LDM  ││Midj. ││ProGAN││VQDM ││Wuk.││
│  └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘└─────┘└────┘│
│  + Corresponding REAL images for each generator                 │
│                                                                 │
│  Augmentation: JPEG(p=0.5) + Blur(p=0.5) + HFlip               │
│                                                                 │
│  Loss = L_cls + 0.5*L_family + 0.3*L_div + 0.2*L_compact       │
│         + 0.3*L_heatmap + 0.1*L_attr + 0.2*L_calib             │
│                                                                 │
│  AdamW: lr=1e-4 (new) / 1e-6 (LN) | CosineAnnealing | WD=0.01 │
│                                                                 │
│  Validation: 12 OOD generators (never seen during training)     │
│  Early stopping: patience=3 on OOD average AP                   │
│                                                                 │
│         ┌─────────────────────────────────┐                     │
│         │  Output: best_model.pth         │                     │
│         │  (Stage 1 checkpoint)           │                     │
│         └──────────────┬──────────────────┘                     │
│                        │                                        │
└────────────────────────┼────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ANNOTATION PIPELINE                            │
│                  (~$300-500 API cost)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Run Stage 1 on 50K diverse images                           │
│     └─> Heatmaps + Stage 1 outputs saved                       │
│                                                                 │
│  2. GPT-4o API: (image + heatmap + S1 output) -> annotation    │
│     └─> 50K structured annotations (attributes + explanation)   │
│                                                                 │
│  3. Human review: 5K samples (10%) quality check                │
│     └─> Corrections fed back to improve prompt                  │
│                                                                 │
│         ┌─────────────────────────────────┐                     │
│         │  Output: train.jsonl            │                     │
│         │  (50K annotated triplets)       │                     │
│         └──────────────┬──────────────────┘                     │
│                        │                                        │
└────────────────────────┼────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STAGE 2 TRAINING                            │
│                  (~6-8 hours on H100)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Qwen2.5-VL-7B-Instruct + LoRA (r=16, alpha=32)                │
│  Only q_proj, v_proj trainable (~8M LoRA params)                │
│                                                                 │
│  Input: [image, heatmap_overlay, forensic_prompt]               │
│  Target: structured attributes + NL explanation                 │
│                                                                 │
│  AdamW: lr=2e-5 | 3 epochs | grad_accum=8                      │
│  Causal LM loss on target tokens only                           │
│                                                                 │
│         ┌─────────────────────────────────┐                     │
│         │  Output: lora_weights/          │                     │
│         │  (LoRA adapter checkpoint)      │                     │
│         └─────────────────────────────────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prototype-Guided Attention Decomposition (PGAD) Detail

```
                    Patch Tokens from CLIP ViT
                    [B, 256, 1024]
                          │
                          ▼
                 ┌────────────────┐
                 │  Projection    │
                 │ Linear(1024,128)│
                 │ + LayerNorm    │
                 │ + GELU         │
                 └────────┬───────┘
                          │
                    [B, 256, 128]
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │  Keys (K) │   │Values (V) │   │           │
    │Linear(128)│   │Linear(128)│   │           │
    └─────┬─────┘   └─────┬─────┘   │           │
          │               │         │           │
          │               │         │           │
          │    Prototype Bank       │           │
          │    [128, 128]           │           │
          │         │               │           │
          │         ▼               │           │
          │   ┌───────────┐         │           │
          │   │Queries (Q)│         │           │
          │   │Linear(128)│         │           │
          │   └─────┬─────┘         │           │
          │         │               │           │
          ▼         ▼               ▼           │
    ┌─────────────────────────────────┐         │
    │  Multi-Head Cross-Attention     │         │
    │  (4 heads, 32-dim per head)     │         │
    │                                 │         │
    │  Attn = softmax(Q @ K^T / √d)  │         │
    │  Out = Attn @ V                 │         │
    └──────────┬──────────────────────┘         │
               │                                │
         ┌─────┴─────┐                          │
         │           │                          │
         ▼           ▼                          │
   ┌──────────┐ ┌───────────────┐               │
   │ Proto    │ │ Spatial Maps  │               │
   │ Features │ │ [B,128,16,16] │               │
   │[B,128,128│ │               │               │
   └────┬─────┘ │ Reshape attn  │               │
        │       │ weights to    │               │
        │       │ grid layout   │               │
        ▼       └───────┬───────┘               │
   ┌──────────┐         │                       │
   │ Proto    │         │                       │
   │Activation│         │                       │
   │ [B, 128] │         │                       │
   │ (norm)   │         │                       │
   └────┬─────┘         │                       │
        │               │                       │
        ▼               │                       │
   ┌──────────────────────────┐                 │
   │  Attribute Aggregation    │                 │
   │                          │                 │
   │  For each bank:          │                 │
   │  texture  = mean(act[0:21])               │
   │  edges    = mean(act[21:43])              │
   │  color    = mean(act[43:64])              │
   │  geometry = mean(act[64:86])              │
   │  semantics= mean(act[86:107])             │
   │  frequency= mean(act[107:128])            │
   └────────────┬─────────────┘                 │
                │                               │
                ▼                               │
         ┌──────────┐                           │
         │Attr Scores│                          │
         │  [B, 6]   │                          │
         └───────────┘                          │
```

---

## Output Interpretation Guide

```
╔═══════════════════════════════════════════════════════════════╗
║                  WHAT EACH OUTPUT MEANS                       ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  1. BINARY PREDICTION                                         ║
║     "Is this image AI-generated?"                             ║
║     → REAL or FAKE                                            ║
║                                                               ║
║  2. CALIBRATED CONFIDENCE                                     ║
║     "How certain is the detector?"                            ║
║     → 0.0 (definitely real) to 1.0 (definitely fake)         ║
║     → Temperature-scaled for reliable probability estimates   ║
║                                                               ║
║  3. GENERATOR FAMILY                                          ║
║     "What type of generator made this?"                       ║
║     → Real / GAN / Diffusion / Autoregressive                 ║
║                                                               ║
║  4. SPATIAL HEATMAP                                           ║
║     "WHERE are the forgery artifacts?"                        ║
║     → 224x224 pixel-level suspicion map                       ║
║     → Red = more suspicious, Blue = less suspicious           ║
║                                                               ║
║  5. ATTRIBUTE SCORES                                          ║
║     "WHY does it look fake?"                                  ║
║     → Texture:   Surface consistency / micro-patterns         ║
║     → Edges:     Boundary quality / sharpness artifacts       ║
║     → Color:     Distribution naturalness / banding           ║
║     → Geometry:  Structural coherence / perspective errors    ║
║     → Semantics: Content plausibility / impossible elements   ║
║     → Frequency: Spectral artifacts / aliasing patterns       ║
║                                                               ║
║  6. PROTOTYPE ACTIVATIONS                                     ║
║     "WHAT specific forgery patterns were found?"              ║
║     → 128 prototype scores showing which learned patterns     ║
║       were detected and how strongly                          ║
║                                                               ║
║  7. NL EXPLANATION (Stage 2)                                  ║
║     "Human-readable forensic reasoning"                       ║
║     → 2-4 sentences referencing specific visual evidence      ║
║     → Grounded in heatmap regions and attribute analysis      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```
