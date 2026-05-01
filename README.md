<div align="center">

# GenStack

### Dual-Branch Mixture of Experts for Generalizable Face Forgery Detection

*IJCB 2026 submission*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HydraFake](https://img.shields.io/badge/Benchmark-HydraFake-9cf.svg)](#dataset)
[![CLIP ViT-L/14](https://img.shields.io/badge/Backbone-CLIP%20ViT--L%2F14-success.svg)](https://github.com/openai/CLIP)
[![InternVL3-8B](https://img.shields.io/badge/Reasoner-InternVL3--8B-orange.svg)](https://github.com/OpenGVLab/InternVL)

</div>

> **TL;DR.** GenStack pairs a frozen-CLIP discriminative head (**Pact**) with an InternVL3-8B reasoner over RGB + frequency + noise-residual views (**Prism-SFT**), then fuses them with a $K=15$ KMeans-cluster Mixture-of-Experts gate. On HydraFake (23 generators, 4 splits) it sets a new state of the art at **92.83 % avg accuracy**, **+2.13 pp over the prior best**.

<div align="center">

| | **ID** | **CM** | **CF** | **CD** | **Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|
| Prior SoTA (Veritas) | 94.16 | 99.20 | 92.62 | 81.62 | 90.70 |
| **GenStack ($K{=}15$, ours)** | **94.95** | **99.40** | **94.96** | **84.53** | **92.83** |
| $\Delta$ over SoTA | +0.79 | +0.20 | +2.34 | +2.91 | **+2.13** |

</div>

---

## Architecture at a Glance

```
                        ┌──────────────────────────────────┐
                        │  Branch 1 — Pact (discriminative)│
            ┌──────────▶│  CLIP ViT-L/14 (frozen)          │──▶ p_v ∈ [0,1]
            │           │   + 8 Forgery Prompt Tokens      │
            │           │   + PGAD (128 prototypes, 6 banks)│
            │           └──────────────────────────────────┘
            │                                                  ┐
   image x ─┤                                                  ├─▶ MoE soft gate
            │                                                  │     (K=15 KMeans)
            │           ┌──────────────────────────────────┐   │
            │           │  Branch 2 — Prism-SFT (reasoner) │   │
            └──────────▶│  InternVL3-8B + LoRA (r=128)    │──▶ b_p ∈ {0,1}
                        │  Multi-view: RGB ⊕ FFT ⊕ noise   │      ▼
                        │  → "<region-grounded reasoning>" │   ŷ(x) = Σ π_c(x)·f_c(p_v, b_p)
                        │  → "Real" / "Fake"               │      ▼
                        └──────────────────────────────────┘   verdict
```

**Three small ideas working together:**
1. **Pact** — CLIP ViT-L/14 stays frozen; 8 *forgery prompt tokens* and a 128-prototype attention head (PGAD) are the only learned params on the discriminative side.
2. **Prism-SFT** — InternVL3-8B fine-tuned with LoRA (r=128, α=256) on a tri-view input (RGB + FFT + noise residual). Outputs short region-grounded reasoning before its verdict.
3. **MoE meta-learner** — 15 Gradient-Boosting experts over (Pact score, Prism verdict), routed by a multinomial-LR gate on $\ell_2$-normalised CLIP CLS. Soft-mixed: `ŷ(x) = Σₖ πₖ(x) · fₖ(p_v, b_p)`. Adds <100 K params, trains in <1 minute on a CPU, <0.1 ms per image at inference.

---

## Repository Layout

```
GenStack/
├── pact/                    Discriminative branch (CLIP + FPT + PGAD) and shared infra
│   ├── models/              Backbone, prototype module, multi-branch heads, fusion
│   ├── training/            Training loops (XGenDet v5+, Pact)
│   ├── data/                HydraFake loaders, augmentations, view extractors
│   ├── evaluation/          Per-split eval, robustness sweep, calibration
│   ├── scripts/             End-to-end runners + utilities
│   ├── configs/             Yaml configs (model size, LR, schedules)
│   ├── pbs/                 PBS / SLURM job templates
│   ├── tests/               Smoke tests (forward pass, dataloaders, training)
│   └── requirements.txt
├── prism_sft_ablation/      Prism-SFT InternVL3-8B recipe + view-masking ablation
│   └── multiview/           RGB / RGB+FFT / RGB+FFT+noise SFT data + scripts
├── reproduce/
│   └── cached_predictions/  All cached (Pact, Prism) tuples + CLIP CLS chunks
│                            needed to reproduce every MoE table in the paper
├── paper/                   IJCB 2026 paper source (main + supplementary, figures, bib)
├── AGENTS.md                How to drive this repo with an LLM agent (Claude/Codex)
├── README.md                You are here
└── LICENSE                  MIT
```

---

## Quick Start

### 1. Environment

```bash
git clone https://github.com/hardikbha/GenStack.git
cd GenStack
conda create -n genstack python=3.10 -y
conda activate genstack
pip install -r pact/requirements.txt
```

Tested with PyTorch 2.1, CUDA 12.1, single A100 / H100. Multi-GPU works for SFT only — Pact fits on a single 24 GB card.

### 2. Reproduce the headline number (no GPU, <2 min)

The full MoE meta-learner can be retrained from cached `(p_v, b_p, CLIP-CLS, label)` tuples shipped under `reproduce/cached_predictions/`. No image data, no backbone weights needed.

```bash
python pact/scripts/reproduce_genstack_k15.py \
    --cache reproduce/cached_predictions \
    --K 15 \
    --seed 42
```

Expected:

```
GenStack-15  ID=94.95  CM=99.40  CF=94.96  CD=84.53  Avg=92.83
```

(Cross-seed std `±0.24 pp`; see `paper/supplementary.tex` §M.)

### 3. Train Pact from scratch (1 × A100, ~6 h)

```bash
python pact/training/train_v5plus.py \
    --config pact/configs/v5plus_hydrafake.yaml \
    --data /path/to/hydrafake \
    --out runs/pact_v5plus
```

Drops a `pact.pt` checkpoint and a per-image score JSON.

### 4. Train Prism-SFT (4 × A100 80 GB, ~12 h)

```bash
bash prism_sft_ablation/multiview/train_full.sh \
    --base internvl3-8b \
    --views rgb,fft,noise \
    --lora-rank 128 --lora-alpha 256 \
    --out runs/prism_sft
```

For the small view-masking ablation (Table S6):

```bash
bash prism_sft_ablation/multiview/run_inference_views.sh rgb_only,rgb_fft,full
```

### 5. Glue them together

```bash
python pact/scripts/build_genstack.py \
    --pact runs/pact_v5plus/pact.pt \
    --prism runs/prism_sft \
    --K 15 \
    --out runs/genstack_k15
```

---

## What's *not* in this repo (and where to find it)

| Artifact | Size | Where |
|---|---|---|
| Pact checkpoint | ~600 MB | _link TBD — see [Releases](https://github.com/hardikbha/GenStack/releases)_ |
| Prism-SFT merged 8B weights | ~16 GB | _Hugging Face: `hardikbha/genstack-prism-sft`_ |
| HydraFake image data (52 K test images) | ~30 GB | Original benchmark — request from authors of HydraFake |

Please **do not** redistribute HydraFake images; we only release derived per-image scores under `reproduce/cached_predictions/`.

---

## Reproducing paper numbers

The headline number is reproduced by `pact/scripts/reproduce_genstack_k15.py`. To
re-derive any other table/figure that is built from cached predictions (every
ablation in the supplementary), pass different arguments to that script — its
`run()` function takes `K` and `seed` as arguments and returns the per-split
dict. Wrap it in a small driver to sweep `K`, seeds, etc. Pact training and
Prism-SFT training do require GPUs; see Sections 3 & 4 of this README.

The cached predictions under `reproduce/cached_predictions/` are sufficient for
**every** ablation that does not change the per-image `(p_v, b_p)` tuples — i.e.
all of Tables 1, 3, 4, 5, S1–S5, S8–S11, and Figures 3, S1, S2, S4. Robustness
(Figure 3, Table S7) needs Pact re-evaluated on degraded inputs and is not
reproducible from the cache alone.

---

## Citing

```bibtex
@inproceedings{genstack2026,
  title     = {GenStack: Dual-Branch Mixture of Experts for Generalizable Face Forgery Detection},
  author    = {Anonymous},
  booktitle = {IJCB},
  year      = {2026}
}
```

---

## Acknowledgements

Built on top of [CLIP ViT-L/14](https://github.com/openai/CLIP), [InternVL3](https://github.com/OpenGVLab/InternVL), the [HydraFake](https://github.com/) benchmark, and [`ms-swift`](https://github.com/modelscope/ms-swift) for VLM SFT. Pact's prototype-guided attention head extends the design from [ProtoPNet](https://arxiv.org/abs/1806.10574) into the forgery-detection setting.

See [AGENTS.md](AGENTS.md) for instructions on driving this repo with Claude / Codex / Cursor / similar.
