# XGenDet: Two-Paper Publication Roadmap

**Generated**: March 13, 2026
**Strategy**: Two-paper split for maximum publication probability (~80%)

---

## Publication Targets

| Paper | Venue | Deadline | Track | Probability |
|-------|-------|----------|-------|-------------|
| **Paper 1: XGenBench** | NeurIPS 2026 | May 7, 2026 | Datasets & Benchmarks | ~75% |
| **Paper 2: XGenDet** | ECCV 2026 or IEEE TIFS | ~July 2026 / Rolling | Main conference / Journal | ~70% |

---

## What We're Building

### Paper 1: XGenBench
**"The First Multi-Dimensional Benchmark for Explainable Generalized AI-Generated Image Detection"**

A benchmark that evaluates detectors across **5 dimensions** (not just binary accuracy):

| Dimension | Question | Metrics |
|-----------|----------|---------|
| **Detect** | Is it real or fake? | Accuracy, AP, AUC-ROC, F1 |
| **Locate** | Where are the artifacts? | IoU, Pixel-AP, Pointing Game |
| **Attribute** | What type of artifacts? | MAE, Spearman correlation |
| **Explain** | Why is it fake? | ROUGE-L, GPT-4 judge score |
| **Robust** | Does it survive compression? | Metrics under JPEG/blur/resize |

**Contributions:**
1. First benchmark evaluating 5 complementary dimensions of explainable detection
2. Standardized evaluation protocol + open-source code
3. 8+ baseline results (D3, GAPL, NPR, LNCLIP-DF, GradCAM, zero-shot MLLM, XGenDet-Lite)
4. Analysis of where current detectors fail on explainability

### Paper 2: XGenDet (Full Method)
**"Explainable Generalized AI-Generated Image Detection via Prototype-Guided Attention Decomposition"**

| Feature | Description |
|---------|-------------|
| Architecture | Frozen CLIP ViT-L/14 + VPT-Deep + 128 prototypes in 6 semantic banks |
| Trainable params | ~1.2M (0.4% of backbone) |
| Outputs | Binary prediction, calibrated confidence, generator family, artifact heatmap, 6 attribute scores, NL explanation |
| Stage 1 | Detection + heatmap (~20ms inference) |
| Stage 2 | MLLM explanation via Qwen2.5-VL-7B + LoRA (~2-5s inference) |

**Contributions:**
1. Prototype-Guided Attention Decomposition (PGAD) — prototypes as spatial forensic primitives
2. Forgery-Aware VPT-Deep + true multiplicative attention rollout
3. Detection-conditioned MLLM reasoning
4. SOTA on XGenBench + standard benchmarks

---

## Current State (March 13, 2026)

### Code Complete
```
xgendet/
├── models/
│   ├── backbone.py           ✅ CLIP ViT-L/14 + prompts + pos_embed (FIXED)
│   ├── prototype_module.py   ✅ 128 prototypes, 6 banks, cross-attention
│   ├── heatmap_generator.py  ✅ Three-source fusion (weighted sum, needs rollout for Paper 2)
│   ├── classification_head.py ✅ Binary + family heads
│   ├── mllm_module.py        ✅ Qwen2.5-VL-7B + LoRA wrapper
│   └── xgendet.py            ✅ Full pipeline
├── data/
│   ├── dataset.py            ✅ RealFakeDataset (Midjourney family FIXED)
│   ├── augmentations.py      ✅ Blur, JPEG, transforms
│   ├── annotation_pipeline.py ✅ GPT-4.1 annotation generation
│   └── xgenbench.py          ✅ XGenBench evaluation dataset
├── training/
│   ├── train_stage1.py       ✅ Stage 1 training loop
│   ├── train_stage2.py       ✅ Stage 2 MLLM fine-tuning
│   ├── calibration.py        ✅ Temperature scaling + ECE
│   └── losses.py             ✅ All loss functions
├── evaluation/
│   ├── evaluate.py           ✅ Metrics (AP, AUC, Acc, F1, ECE)
│   └── visualize.py          ✅ Heatmap overlay, t-SNE, confidence plots
├── demo/
│   ├── demo.py               ✅ Single-image inference
│   └── gradio_app.py         ✅ Interactive web demo
└── configs/                  ✅ All YAML configs
```

### Bugs Fixed (Today)
| Bug | File | Fix |
|-----|------|-----|
| Midjourney labeled as Autoregressive (3) | dataset.py:36, id_generators.yaml:32 | Changed to Diffusion (2) |
| Prompt tokens had no positional embeddings | backbone.py | Added `prompt_pos_embed` parameter |

### Bugs Deferred to Paper 2
| Bug | File | Why Deferred |
|-----|------|-------------|
| VPT-Shallow → VPT-Deep | backbone.py | Acceptable for Paper 1 baseline; upgrade for Paper 2 |
| Weighted sum → true attention rollout | heatmap_generator.py | Acceptable design choice for Paper 1; improve for Paper 2 |

### Training Status
- **Job 14796.mgmt01**: RUNNING on GPU queue (2 CPUs, 1 GPU)
- Training 8 ID generators, validating on 12 OOD generators
- Target: >88% OOD accuracy, >0.90 AP
- Expected time: ~2-4 hours on H100

### Data Available
| Dataset | Location | Size | Status |
|---------|----------|------|--------|
| 8 ID generators | `GTA/final_GENERATORS/` | 3.1M images | ✅ Ready |
| 12 OOD generators | `GTA/OOD_GENERATORS/` | 100K images | ✅ Ready |
| Community Forensics Small | HuggingFace `OwensLab/CommunityForensics-Small` | 278 GB | ❌ Not downloaded |
| X-AIGD | HuggingFace `Coxy7/X-AIGD` | 3.3K samples | ❌ Not downloaded |
| DF40 | GitHub `YZY-stack/DF40` | 143 GB | ❌ Not downloaded (face-only) |

---

## Week-by-Week Execution Plan

### PAPER 1: XGenBench (Weeks 1-8)

---

#### Week 1: Mar 13-19 — Foundation + Training ← WE ARE HERE

**You (lead):**
- [x] Fix Midjourney family label (dataset.py + id_generators.yaml)
- [x] Add positional embeddings for prompt tokens (backbone.py)
- [x] Smoke test: backbone + full model forward pass
- [x] Submit Stage 1 training job (PBS 14796)
- [ ] Monitor training — check loss curves after epoch 1
- [ ] If training converges: save checkpoint, run OOD evaluation
- [ ] Begin preparing XGenBench JSONL annotation schema

**Collaborators:**
- [ ] Download X-AIGD from HuggingFace: `load_dataset("Coxy7/X-AIGD")`
- [ ] Start literature review: list all existing detection benchmarks and their limitations
- [ ] Set up LaTeX template (NeurIPS D&B format)

**Milestone:** Stage 1 model trained and evaluated. OOD accuracy >85%.

---

#### Week 2: Mar 20-26 — Heatmaps + Annotation Pipeline

**You:**
- [ ] Run temperature scaling calibration on validation set
- [ ] Generate heatmaps on 15K diverse images (stratified across all 20 generators)
- [ ] Set up GPT-4.1-mini annotation pipeline
- [ ] Test on 100 images, review quality
- [ ] Tune annotation prompt template based on quality review

**Collaborators:**
- [ ] Curate 15K image selection (stratified sampling: ~750 per generator)
- [ ] Clone D3 repo, reproduce D3 baseline on our evaluation set
- [ ] Collect published numbers from GAPL, NPR, LNCLIP-DF, UnivFD papers
- [ ] Organize X-AIGD data into XGenBench format

**Milestone:** Annotation pipeline tested and ready. D3 baseline numbers obtained.

---

#### Week 3: Mar 27 - Apr 2 — Full Annotations + Baselines

**You:**
- [ ] Run full annotation pipeline: 10K images through GPT-4.1-mini (~$50-80)
- [ ] Polish XGenBench evaluation harness:
  - Per-generator stratified evaluation
  - Cross-generator family analysis
  - Social media perturbation simulation (JPEG 30-100, blur 0.5-2.0, resize 50%-150%)
- [ ] Run XGenDet-Lite (our Stage 1 model) through full XGenBench evaluation
- [ ] Implement explanation faithfulness metric (insertion/deletion)

**Collaborators:**
- [ ] Run D3 baseline through XGenBench evaluation
- [ ] Generate GradCAM baseline heatmaps (zero-shot CLIP + GradCAM++)
- [ ] Run zero-shot Qwen2.5-VL-7B explanation on 500 images
- [ ] Begin human review of GPT-4.1-mini annotations (target: 2K samples)

**Milestone:** All annotations generated. 3+ baselines evaluated.

---

#### Week 4: Apr 3-9 — Complete Baselines + Analysis

**You:**
- [ ] Run remaining baselines through XGenBench (FakeShield if code available)
- [ ] Compute all 5-dimension metrics for every baseline
- [ ] Run heatmap quality ablation: prototype-only vs attention-only vs fusion
- [ ] Generate all quantitative results tables
- [ ] Statistical significance tests (bootstrap confidence intervals)

**Collaborators:**
- [ ] Complete 2K human annotation reviews
- [ ] Compute inter-annotator agreement (Cohen's kappa or Fleiss' kappa)
- [ ] Begin writing: Introduction + Related Work
- [ ] Create Datasheets for Datasets card

**Milestone:** All experiment results ready. Tables populated.

---

#### Week 5: Apr 10-16 — Robustness + Qualitative Figures

**You:**
- [ ] Full robustness evaluation:
  - JPEG compression: quality 30, 50, 70, 90, 100
  - Gaussian blur: sigma 0.5, 1.0, 1.5, 2.0
  - Resize: 50%, 75%, 100%, 150%
  - Social media simulation: Instagram, Twitter, WeChat
- [ ] Generate qualitative figures:
  - Heatmap comparison across methods (ours vs GradCAM vs FakeShield)
  - Failure case analysis (what fools every detector?)
  - Per-generator performance radar charts
- [ ] Create benchmark framework architecture diagram

**Collaborators:**
- [ ] Draft "Benchmark Design" section (data composition, annotation process, quality control)
- [ ] Draft "Evaluation Protocol" section (5 dimensions, metrics, splits)
- [ ] Create all tables and figures (initial versions)
- [ ] Write data collection methodology

**Milestone:** All experiments complete. Figures drafted.

---

#### Week 6: Apr 17-23 — Paper Writing Sprint

**You:**
- [ ] Write "Baseline Results & Analysis" section
- [ ] Write "Robustness Analysis" section
- [ ] Write "Discussion & Open Challenges" section
- [ ] Create supplementary materials (full per-generator tables, additional heatmap examples)

**Collaborators:**
- [ ] Polish Introduction + Related Work
- [ ] Create framework overview figure (the 5-dimension evaluation pipeline)
- [ ] Proofread all sections
- [ ] Clean up evaluation code for open-source release

**Milestone:** Full draft complete.

---

#### Week 7: Apr 24-30 — Internal Review + Revisions

**You:**
- [ ] Internal review cycle 1: read full draft, identify weak points
- [ ] Fill any missing experimental results
- [ ] Rerun experiments if review reveals issues
- [ ] Finalize supplementary materials

**Collaborators:**
- [ ] Full proofreading pass (grammar, consistency, notation)
- [ ] NeurIPS D&B formatting compliance check (9 pages + unlimited appendix)
- [ ] Prepare HuggingFace dataset card for XGenBench
- [ ] Prepare GitHub README for benchmark code release

**Milestone:** Reviewed draft, all issues addressed.

---

#### Week 8: May 1-7 — Final Polish + Submit

**You:**
- [ ] Final edits from review cycle 2
- [ ] Verify every number in every table matches code outputs
- [ ] Prepare author response template for common reviewer questions
- [ ] **Submit to NeurIPS 2026 D&B by May 7**

**Collaborators:**
- [ ] Final LaTeX formatting and figure quality check
- [ ] Upload supplementary materials
- [ ] Prepare dataset hosting on HuggingFace
- [ ] Tag GitHub release

**Milestone:** Paper 1 submitted!

---

### PAPER 2: XGenDet Method (Weeks 9-16)

---

#### Week 9: May 8-14 — VPT-Deep Implementation

- [ ] Modify backbone.py: inject fresh prompt tokens at ViT layers 0, 6, 12, 18
- [ ] Requires replacing monolithic `visual.transformer(x)` with per-block forward loop
- [ ] Adds ~32K params (4 layers × 8 tokens × 1024 dim)
- [ ] Retrain Stage 1 with VPT-Deep
- [ ] Compare: VPT-Shallow vs VPT-Deep (accuracy, heatmap quality, training speed)

---

#### Week 10: May 15-21 — True Attention Rollout + Enhanced Heatmaps

- [ ] Implement multiplicative rollout in heatmap_generator.py:
  ```
  R = Π_{l=1}^{L} (0.5·I + 0.5·A_l)
  ```
- [ ] Compare heatmap quality: weighted sum vs true rollout vs hybrid
- [ ] Run localization evaluation: IoU, Pixel-AP, Pointing Game
- [ ] Add prototype specialization loss (inter-bank orthogonality)

---

#### Week 11: May 22-28 — MLLM Stage 2 Full Pipeline

- [ ] Scale annotation pipeline to 50K images (~$145 with GPT-4.1-mini Batch API)
- [ ] Fine-tune Qwen2.5-VL-7B on 50K annotated triplets
- [ ] Benchmark InternVL3-8B as alternative
- [ ] Add constrained decoding to force structured JSON output
- [ ] Optional: DPO on 2K human-curated preference pairs

---

#### Week 12: May 29 - Jun 4 — Comprehensive Ablations (7+)

1. Component ablation: w/o prompts, w/o prototypes, w/o LayerNorm tuning, w/o MLLM
2. Prototype count: 32 vs 64 vs 128 vs 256
3. Prompt token count: 2 vs 4 vs 8 vs 16
4. Backbone: CLIP ViT-L/14 vs DINOv2 ViT-L vs CLIP ViT-B/16
5. MLLM: InternVL3-8B vs Qwen2.5-VL-7B vs LLaVA-NeXT-7B
6. Training data scaling: 1 gen vs 4 vs 8 vs Community Forensics
7. Calibration: before vs after temperature scaling (ECE)

---

#### Week 13: Jun 5-11 — Additional Experiments

- [ ] Download and train on Community Forensics Small (278 GB, 4700+ generators)
- [ ] Cross-dataset generalization (train GenImage → test Community Forensics)
- [ ] Full robustness evaluation with VPT-Deep model
- [ ] t-SNE of prototype space showing generator clustering
- [ ] Confidence calibration reliability diagram

---

#### Week 14: Jun 12-18 — Paper Writing Sprint

- [ ] Write Method section (PGAD architecture, VPT-Deep, rollout, losses)
- [ ] Write complete Experiments section (main results, ablations, robustness, qualitative)
- [ ] Create architecture diagram (Figure 1)
- [ ] Create all figures: heatmaps, t-SNE, calibration, failure cases

---

#### Week 15: Jun 19-25 — Internal Review

- [ ] Internal review cycle
- [ ] Rerun necessary experiments
- [ ] Finalize supplementary materials
- [ ] Code cleanup for open-source release

---

#### Week 16: Jun 26 - Jul 10 — Submit

- [ ] Final edits
- [ ] **Submit to ECCV 2026** (~July deadline) or **IEEE TIFS** (rolling)
- [ ] Prepare GitHub repo with pretrained weights

---

## Key Technical Decisions

### Novelty Defense (Against M2F2-Det Overlap)

M2F2-Det (CVPR 2025 Oral) uses CLIP + forgery prompts + prototypes + LLM explanations — similar surface-level architecture. Our differentiation:

| Aspect | M2F2-Det | XGenDet (Ours) |
|--------|----------|----------------|
| Prototypes for | Classification only | **Spatial interpretability** — each prototype activates on specific image regions |
| Heatmaps | No | **Yes** — prototype activation maps + attention rollout fusion |
| Attribute banks | No | **Yes** — 6 semantic banks (texture, edge, color, geometry, semantics, frequency) |
| Calibration | No | **Yes** — temperature scaling with ECE reporting |
| Generator family | No | **Yes** — 4-way classification |
| Benchmark | No | **Yes** — XGenBench (Paper 1) |

**Key phrase for rebuttals:** "While M2F2-Det uses prototypes as classification features, XGenDet treats prototypes as spatially-grounded forensic primitives that provide interpretable artifact localization."

### Why Two Papers > One Paper

1. Paper 1 (benchmark) answers "**How should we evaluate?**" — lower novelty bar at D&B track
2. Paper 2 (method) answers "**How should we detect?**" — cleaner architectural story
3. Each paper stands alone with focused contributions
4. Papers cross-cite for mutual reinforcement
5. Combined success probability: ~80% (vs ~15-20% for single paper at NeurIPS main)

### Generator Family Taxonomy

| Family ID | Label | Generators |
|-----------|-------|------------|
| 0 | Real | All real images |
| 1 | GAN | ProGAN, BigGAN, StyleGAN, StyleGAN2, CycleGAN, StarGAN, GauGAN, CRN, IMLE, SAN, Deepfake |
| 2 | Diffusion | ADM, GLIDE, LDM, Midjourney, Wukong, SeeingDark, VQDM |
| 3 | Autoregressive | (reserved for DALL-E 1 if confirmed) |

**Edge cases documented:**
- Midjourney → Diffusion (latent diffusion architecture, FIXED from Autoregressive)
- VQDM → Diffusion (discrete diffusion / masked token prediction, hybrid but closer to diffusion)
- CRN → GAN (not technically adversarial, but convention in the literature)
- SeeingDark → Diffusion (enhancement network, debatable)

---

## Resource Requirements

### Compute
| Resource | Paper 1 | Paper 2 |
|----------|---------|---------|
| GPU hours (H100) | ~10-20h | ~50-100h |
| Training runs | 1 Stage 1 + baselines | 7+ ablations + Stage 2 MLLM |
| PBS queue | gpu queue, 2 CPUs + 1 GPU | Same |

### Budget
| Item | Cost | For |
|------|------|-----|
| GPT-4.1-mini annotations (10K) | ~$50-80 | Paper 1 |
| GPT-4.1-mini annotations (50K) | ~$145 | Paper 2 |
| Human annotation review | Collaborator time | Both |
| **Total** | **~$200-225** | |

### Storage
| Dataset | Size | Status |
|---------|------|--------|
| Local GenImage (ID + OOD) | ~300 GB | ✅ Available |
| Community Forensics Small | 278 GB | Download for Paper 2 |
| X-AIGD | ~1 GB | Download for Paper 1 |
| XGenBench annotations | ~5 GB | Generate |
| Model checkpoints | ~10 GB | Generate |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Stage 1 accuracy < 85% OOD | Medium | High | Ablate backbone (add DINOv2), increase prototypes, add LoRA as fallback |
| Heatmaps look random | Low | High | GradCAM++ as backup; combine with shuffle-diff maps |
| MLLM hallucinations | Medium | Medium | Constrained decoding; attribute-first prompting; DPO |
| Community Forensics download fails | Low | Medium | Use local 3.1M data only — still larger than most papers |
| NeurIPS D&B reviewers reject benchmark scope | Medium | High | Emphasize: first 5-dimension evaluation; show no existing benchmark covers even 2 |
| M2F2-Det overlap flagged | High | Medium | Prototype spatial grounding + benchmark are genuine differentiators |
| PBS queue delays | Medium | Low | Use 2 CPUs to get faster scheduling; run ablations sequentially |

---

## Monitoring & Checkpoints

### How to Monitor Training
```bash
# Check job status
qstat -u sachin.chaudhary

# Watch training logs
tail -f /home/sachin.chaudhary/xgendet/logs/train_stage1_14796.mgmt01.out

# Check TensorBoard
tensorboard --logdir /home/sachin.chaudhary/xgendet/checkpoints/xgendet_stage1/tensorboard
```

### Decision Gates

| Gate | When | Pass Criteria | If Fail |
|------|------|---------------|---------|
| Stage 1 convergence | End of Week 1 | Loss < 0.4, OOD acc > 70% | Debug architecture, check data loading |
| OOD accuracy | End of Week 2 | > 85% accuracy, > 0.90 AP | Add LoRA, increase prototypes |
| Heatmap quality | End of Week 2 | Visual inspection passes on 100 images | Add GradCAM++, adjust fusion weights |
| Annotation quality | End of Week 3 | Human review agrees >80% of the time | Improve prompt template, increase manual review |
| Paper 1 draft quality | End of Week 7 | Internal review score > 6/10 | Extra writing sprint in Week 8 |

---

## Contact & Resources

| Resource | Location |
|----------|----------|
| Code | `/home/sachin.chaudhary/xgendet/` |
| Data (ID) | `/home/sachin.chaudhary/GTA/final_GENERATORS/` |
| Data (OOD) | `/home/sachin.chaudhary/GTA/OOD_GENERATORS/` |
| D3 baseline | `/home/sachin.chaudhary/GTA/D3/` |
| Qwen2.5-VL | `/home/sachin.chaudhary/Qwen2.5-VL/` |
| Plan file | `/home/sachin.chaudhary/.claude/plans/peaceful-prancing-kurzweil.md` |
| This roadmap | `/home/sachin.chaudhary/xgendet/ROADMAP.md` |
