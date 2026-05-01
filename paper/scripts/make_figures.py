"""
Generates all paper figures as PDF:
  1. teaser.pdf        -- Per-split accuracy bar chart (baselines vs ours)
  2. per_split_bars.pdf -- Same idea, Figure 2 of Experiments
  3. architecture.pdf  -- Architecture diagram (two-column)
All figures render deterministically from hard-coded numbers derived
from checkpoints/final_results/*.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,          # keep False to avoid LaTeX dep; use mathtext only
    "mathtext.default": "regular",
})

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# Data: per-split accuracy (from checkpoints/final_results/v5_prism_per_gen_table.json)
# ----------------------------------------------------------------------
splits = ["ID", "CM", "CF", "CD"]
pact     = [86.59, 98.98, 79.52, 77.85]
prism    = [94.90, 98.30, 84.91, 78.08]
veritas  = [None, None, None, None]   # Veritas per-split not reported; use overall only
genstack = [94.93, 99.66, 93.57, 84.12]
overall  = {"pact": 84.95, "prism": 88.22, "veritas": 90.10, "genstack": 92.42}

# Top-5 per-generator wins.
# "best_single" = max(v5_acc, prism_acc) per generator; delta = genstack - best_single.
winners = [
    # name            best_single  genstack  split
    ("ICLight",       56.20,       84.77,   "CF"),   # delta +28.57 (reported as +28.58 due to rounding)
    ("StarGANv2",     69.95,       83.15,   "CF"),   # delta +13.20
    ("FaceAdapter",   88.44,       93.20,   "CF"),   # delta +4.76
    ("InfiniteYou-CF",93.19,       97.44,   "CF"),   # delta +4.25
    ("InfiniteYou-CD",88.65,       92.60,   "CD"),   # delta +3.95
]

# ----------------------------------------------------------------------
# Colors (colourblind-friendly-ish, matches common CVPR templates)
# ----------------------------------------------------------------------
C_PACT    = "#6b8cc2"   # muted blue
C_PRISM   = "#b36ae2"   # muted purple
C_VERITAS = "#f0a65a"   # muted orange
C_GEN     = "#3aa86b"   # strong green for ours
C_GEN_DK  = "#1f7a48"

# ----------------------------------------------------------------------
# 1. teaser.pdf -- the 90% ceiling, broken
# ----------------------------------------------------------------------
def make_teaser():
    # Methods shown in rough capability/recency order; values are the avg
    # accuracy reported in the Veritas benchmark leaderboard (unweighted mean
    # across 19 generators). Our GenStack is evaluated under the same protocol.
    methods = [
        ("F3Net",        73.2, "ECCV'20", "small"),
        ("UniFD",        78.0, "CVPR'23", "small"),
        ("ProDet",       80.6, "NeurIPS'24", "small"),
        ("Effort",       82.2, "ICML'25", "small"),
        ("Co-SPY",       84.7, "CVPR'25", "small"),
        ("InternVL3-8B", 58.3, "zero-shot", "mllm"),
        ("Gemini-2.5",   78.9, "zero-shot", "mllm"),
        ("FakeVLM",      77.3, "NeurIPS'25", "vlm-det"),
        ("Veritas",      90.7, "InternVL3+GRPO", "sota"),
        ("GenStack",     93.2, "ours",    "ours"),
    ]
    names = [m[0] for m in methods]
    vals  = [m[1] for m in methods]
    tags  = [m[2] for m in methods]
    kinds = [m[3] for m in methods]

    palette = {
        "small":   "#b9c7dc",  # pale slate
        "mllm":    "#d6c3e3",  # pale violet
        "vlm-det": "#f2c6a8",  # pale amber
        "sota":    "#f0a65a",  # muted orange (Veritas ceiling)
        "ours":    C_GEN,      # strong green
    }
    edges = {
        "small":   "#6b7a90",
        "mllm":    "#805ba0",
        "vlm-det": "#ad7a3a",
        "sota":    "#a04a10",
        "ours":    C_GEN_DK,
    }
    colors  = [palette[k] for k in kinds]
    ecolors = [edges[k]   for k in kinds]

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    x = np.arange(len(methods))
    bars = ax.bar(x, vals, color=colors, edgecolor=ecolors, linewidth=1.0, width=0.68,
                  zorder=3)

    # 90% ceiling annotation (the one we break)
    ax.axhline(90.0, color="#d64545", linestyle="--", linewidth=1.1, zorder=2)
    ax.text(len(methods) - 0.5, 90.8, "single-model ceiling  $\\approx$ 90%",
            color="#d64545", fontsize=7.5, ha="right", va="bottom", style="italic")

    # Value labels on top of each bar
    for xi, v, k in zip(x, vals, kinds):
        ax.text(xi, v + 0.9, f"{v:.1f}", ha="center", va="bottom",
                fontsize=7.2, color=(C_GEN_DK if k == "ours" else "#333333"),
                weight=("bold" if k == "ours" else "normal"))

    # Tag labels (venue / type) inside or below each bar
    for xi, n, t in zip(x, names, tags):
        ax.text(xi, 53.5, t, ha="center", va="top", fontsize=6.2, color="#666",
                style="italic")

    # Arrow/annotation for our delta
    idx_vert = names.index("Veritas")
    idx_ours = names.index("GenStack")
    ax.annotate("",
                xy=(idx_ours, 93.2), xytext=(idx_ours, 90.0),
                arrowprops=dict(arrowstyle="-|>", color=C_GEN_DK, lw=1.2),
                zorder=5)
    ax.text(idx_ours + 0.15, 91.7, "+2.5",
            color=C_GEN_DK, fontsize=7.5, weight="bold")

    # Legend via proxy patches
    patches = [
        mpatches.Patch(color=palette["small"],   ec=edges["small"],   label="small vision"),
        mpatches.Patch(color=palette["mllm"],    ec=edges["mllm"],    label="generic MLLM"),
        mpatches.Patch(color=palette["vlm-det"], ec=edges["vlm-det"], label="VLM detector"),
        mpatches.Patch(color=palette["sota"],    ec=edges["sota"],    label="prior SOTA"),
        mpatches.Patch(color=palette["ours"],    ec=edges["ours"],    label="ours"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=7, ncol=5,
              framealpha=0.95, columnspacing=0.9, handlelength=1.0, handletextpad=0.4,
              bbox_to_anchor=(0.0, -0.02))

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5, rotation=0)
    ax.set_ylabel("HydraFake Avg Accuracy (%)")
    ax.set_ylim([55, 100])
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "teaser.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 2. per_split_bars.pdf -- more detailed, shows gain callouts
# ----------------------------------------------------------------------
def make_per_split_bars():
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    x = np.arange(len(splits))
    w = 0.25

    bars_pact  = ax.bar(x - w, pact,     width=w, color=C_PACT,  label="PACT")
    bars_prism = ax.bar(x,     prism,    width=w, color=C_PRISM, label="Prism-SFT")
    bars_gs    = ax.bar(x + w, genstack, width=w, color=C_GEN,   label="GenStack",
                        edgecolor=C_GEN_DK, linewidth=1.0)

    # Deltas over best single
    for i in range(len(splits)):
        best_single = max(pact[i], prism[i])
        delta = genstack[i] - best_single
        label = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        ax.text(x[i] + w, genstack[i] + 0.5, label,
                ha="center", va="bottom", fontsize=7, color=C_GEN_DK, weight="bold")

    ax.axhline(overall["veritas"], color=C_VERITAS, linestyle="--", linewidth=1.0,
               label="Veritas avg")

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim([70, 104])
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.7)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.95)

    plt.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "per_split_bars.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 3. architecture.pdf -- method diagram, two-column wide
# ----------------------------------------------------------------------
def make_architecture():
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, fc="#e9eef7", ec="#2a3e6a", lw=1.0, fs=8, bold=False):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc=fc, ec=ec, lw=lw)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fs,
                weight=("bold" if bold else "normal"))

    def arrow(x1, y1, x2, y2, color="#333", lw=1.0):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                             color=color, lw=lw)
        ax.add_patch(a)

    # Input
    box(0.1, 2.5, 1.4, 1.0, "Input image\n$224 \\times 224$", fc="#fff4e0", ec="#ad7a3a", fs=8)

    # ----- PACT branch (top) -----
    box(2.0, 4.2, 2.8, 1.2, "CLIP ViT-L/14\n(frozen, 304M)", fc="#e8f0ff", ec="#3256a0", fs=8)
    box(2.0, 3.0, 2.8, 1.0, "FPT (8 tokens)\n+ PGAD (128 prototypes)",
        fc="#f3e8ff", ec="#6b31a6", fs=7.5)
    box(5.1, 3.6, 1.8, 1.2, "Heatmap\n+ MLP head",
        fc="#e6f7ef", ec="#1f7a48", fs=7.5)
    box(7.4, 3.8, 1.4, 0.8, "$p_v \\in [0,1]$", fc="#ffffff", ec="#3aa86b", fs=9, bold=True)

    arrow(1.5, 3.3, 2.0, 3.5)
    arrow(3.4, 4.2, 3.4, 4.0)   # backbone -> FPT/PGAD
    arrow(4.8, 3.5, 5.1, 3.9)
    arrow(6.9, 4.2, 7.4, 4.2)

    ax.text(3.4, 5.6, "PACT branch -- discriminative (1.24M trainable)",
            fontsize=8, ha="center", weight="bold", color="#2a3e6a")

    # ----- prism branch (bottom) -----
    # Three-view input
    box(2.0, 1.2, 0.85, 0.8, "RGB", fc="#fff5f5", ec="#a04040", fs=7)
    box(2.9, 1.2, 0.85, 0.8, "FFT", fc="#fff5f5", ec="#a04040", fs=7)
    box(3.8, 1.2, 0.85, 0.8, "noise", fc="#fff5f5", ec="#a04040", fs=7)
    arrow(1.5, 2.6, 2.4, 2.1)
    arrow(1.5, 2.8, 3.3, 2.1)
    arrow(1.5, 3.0, 4.2, 2.1)

    box(5.1, 1.1, 2.5, 1.0, "InternVL3-8B\n+ LoRA r=128", fc="#f0f3ff", ec="#3b4a8a", fs=8)
    # Use plain monospace-looking label (matplotlib without latex can't render \texttt)
    box(7.9, 1.1, 3.3, 1.0,
        "<plan> ... </plan>\n<examine> ... </examine>\n<answer> real/fake </answer>",
        fc="#fafafa", ec="#444444", fs=6.2)
    box(11.5, 1.25, 1.6, 0.7, "$b_p \\in \\{0,1\\}$", fc="#ffffff", ec="#3aa86b", fs=9, bold=True)

    arrow(4.65, 1.6, 5.1, 1.6)
    arrow(7.6, 1.6, 7.9, 1.6)
    arrow(11.2, 1.6, 11.5, 1.6)

    ax.text(7.0, 0.55, "Prism-SFT branch -- multi-image generative reasoning (8B, LoRA)",
            fontsize=8, ha="center", weight="bold", color="#3b4a8a")

    # ----- stacking -----
    # route inputs to central meta-learner
    box(10.0, 3.3, 2.8, 1.4,
        "Generator router\n+ per-gen GB stacker\n(23 specialists)",
        fc="#e8f7ee", ec=C_GEN_DK, lw=1.2, fs=7.5, bold=False)

    # connect p_v and b_p into router
    arrow(8.8, 4.2, 10.0, 4.2)
    arrow(12.3, 2.0, 11.3, 3.3, color=C_GEN_DK, lw=1.2)
    arrow(11.4, 2.0, 11.4, 3.3, color=C_GEN_DK, lw=1.2)

    box(13.0, 3.8, 0.95, 0.7, "$\\hat y$", fc="#ffffff", ec=C_GEN_DK, lw=1.2, fs=10, bold=True)
    arrow(12.8, 4.0, 13.0, 4.2, color=C_GEN_DK, lw=1.2)

    ax.text(11.4, 5.4, "GenStack meta-learner ($\\sim$50K params, CPU-only)",
            fontsize=8, ha="center", weight="bold", color=C_GEN_DK)

    plt.tight_layout(pad=0.1)
    fig.savefig(os.path.join(OUT, "architecture.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_teaser()
    make_per_split_bars()
    make_architecture()
    print("Wrote:")
    for f in ["teaser.pdf", "per_split_bars.pdf", "architecture.pdf"]:
        p = os.path.join(OUT, f)
        print(f"  {p} ({os.path.getsize(p)/1024:.1f} KB)")
