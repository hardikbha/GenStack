"""Compute the GenStack vs Veritas average with the top-2 highest-gain
generators (ICLight, StarGANv2) removed, as a robustness sanity check.

Defangs the "your headline depends on 2 generators" reviewer attack.
"""
import json
import numpy as np
from pathlib import Path

R = Path(
    "/home/sachin.chaudhary/xgendet/checkpoints/final_results/full_per_gen_breakdown.json"
)
data = json.loads(R.read_text())

# Veritas per-generator values from Table 1 (paper)
VERITAS = {
    "ff": 97.30,
    "Hallo2": 99.70,
    "Midjourney": 100.0,
    "StyleGAN": 100.0,
    "facevid2vid": 99.90,
    "AdobeFirefly": 94.8,
    "Flux11Pro": 99.8,
    "StarryAI": 97.0,
    "MAGI": 99.9,
    "hart": 99.9,
    "Infinity": 99.9,
    "starganv2": 90.3,
    "iclight": 75.7,
    "codeformer": 97.0,
    "infiniteyou": 91.8,
    "PuLID": 95.1,
    "FaceAdapter": 91.7,
    "deepfacelab": 58.6,
    "infiniteyou_cd": 84.1,
    "dreamina": 92.3,
    "hailuo": 90.2,
    "GPT4o": 89.2,
    "FFIW": 78.5,
}

DROP_GENS = {"iclight", "starganv2"}

per_gen = data["per_generator"]
gs_with, gs_drop = [], []
v_with, v_drop = [], []
matched_gens = []


# Build canonical name -> Veritas value lookup (lowercase + alias map)
def canon(g):
    g = g.lower()
    if g.startswith(("id_", "cm_", "cf_", "cd_")):
        g = g[3:]
    g = g.replace("face++", "ff").replace("faceforensics", "ff")
    g = g.replace("flux1.1pro", "flux11pro")
    return g


for row in per_gen:
    name = row["generator"]
    can = canon(name)
    # find matching Veritas entry
    v_val = None
    for vk, vv in VERITAS.items():
        if canon(vk) == can:
            v_val = vv
            break
    if v_val is None:
        continue
    matched_gens.append((name, can))
    gs_with.append(row["ensemble"] * 100)
    v_with.append(v_val)
    if can not in DROP_GENS:
        gs_drop.append(row["ensemble"] * 100)
        v_drop.append(v_val)

print(f"Matched generators: {len(matched_gens)}")
print(f"  GenStack avg (all 19):              {np.mean(gs_with):.2f}")
print(f"  Veritas avg (all 19):               {np.mean(v_with):.2f}")
print(
    f"  Δ (all 19):                         {np.mean(gs_with) - np.mean(v_with):+.2f}"
)
print()
print(f"Drop ICLight + StarGANv2:")
print(f"  GenStack avg (17 generators):       {np.mean(gs_drop):.2f}")
print(f"  Veritas avg (17 generators):        {np.mean(v_drop):.2f}")
print(
    f"  Δ without top-2 gain generators:    {np.mean(gs_drop) - np.mean(v_drop):+.2f}"
)
