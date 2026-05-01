"""Robustness smoke test — apply degradation, run PACT (v5+) inference on small subset.

This validates that:
  1. We can load HydraFake test JSONs and build a stratified subset
  2. We can apply JPEG / Gaussian blur degradations
  3. We can load + run PACT (v5+) on degraded images and get sensible probs

Usage: python robustness_smoke.py
"""
import json
import sys
import io
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageFilter
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
DATA_ROOT = "/home/sachin.chaudhary"
TEST_DIR = Path("/home/sachin.chaudhary/hydrafake/jsons/test")
V5_CKPT = "/home/sachin.chaudhary/xgendet/checkpoints/v5_resume_hl/best_model.pth"
V5PLUS_CKPT = "/home/sachin.chaudhary/xgendet/checkpoints/v5plus/best_branches.pth"


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def build_stratified_subset(per_gen: int = 10) -> list:
    """Pick `per_gen` images per generator across all splits."""
    samples = []
    for split_dir in ["id", "cm", "cf", "cd"]:
        for jpath in sorted((TEST_DIR / split_dir).glob("*.json")):
            try:
                d = json.load(open(jpath))
            except Exception as e:
                print(f"  skip {jpath.name}: {e}", flush=True)
                continue
            gen_name = jpath.stem
            # take per_gen items, balanced real/fake if possible
            real = [x for x in d if x["label"] == 0][:per_gen // 2]
            fake = [x for x in d if x["label"] == 1][:per_gen - len(real)]
            for x in real + fake:
                img_path = x["images"][0]
                if not img_path.startswith("/"):
                    img_path = str(Path(DATA_ROOT) / img_path)
                samples.append({
                    "split": split_dir,
                    "generator": gen_name,
                    "label": x["label"],
                    "img_path": img_path,
                })
    return samples


def load_pact():
    """Load PACT (v5+) model."""
    from models.xgendet_v5plus import XGenDetV5Plus
    model = XGenDetV5Plus(v5_checkpoint_path=V5_CKPT)
    model.load_branches_checkpoint(V5PLUS_CKPT)
    return model.cuda().eval()  # keep float; some submodules don't accept bf16


def preprocess(img: Image.Image, size: int = 224) -> torch.Tensor:
    tfm = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    return tfm(img)


def main():
    print("[1/5] Building stratified subset (10 per gen)...", flush=True)
    samples = build_stratified_subset(per_gen=10)
    print(
        f"  -> {len(samples)} samples across {len({s['generator'] for s in samples})} generators",
        flush=True)
    if not samples:
        print("ERROR: no samples", flush=True)
        return 1

    # Verify a few image paths exist
    n_missing = sum(1 for s in samples[:20]
                    if not Path(s["img_path"]).exists())
    print(f"  -> {n_missing}/20 sampled paths missing on disk", flush=True)
    if n_missing > 5:
        print("ERROR: many missing image paths; aborting", flush=True)
        for s in samples[:5]:
            print("    sample:", s["img_path"], "exists?",
                  Path(s["img_path"]).exists())
        return 1

    print("[2/5] Loading PACT (v5+) model...", flush=True)
    try:
        model = load_pact()
        print("  -> loaded ok", flush=True)
    except Exception as e:
        print(f"ERROR loading PACT: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1

    print("[3/5] Running PACT on ORIGINAL images (warmup)...", flush=True)
    results_orig = []
    n_test = min(20, len(samples))
    with torch.no_grad():
        for s in samples[:n_test]:
            try:
                img = Image.open(s["img_path"]).convert("RGB")
                x = preprocess(img).unsqueeze(0).cuda().float()
                out = model(x)
                p = out["prob"].squeeze().float().item()
                results_orig.append({
                    "gen": s["generator"],
                    "label": s["label"],
                    "p_v": p
                })
            except Exception as e:
                print(f"  fail on {s['img_path']}: {e}", flush=True)
    accuracies_orig = sum(
        int((r["p_v"] >= 0.5) == r["label"])
        for r in results_orig) / max(len(results_orig), 1)
    print(
        f"  -> PACT acc on {len(results_orig)} originals: {accuracies_orig*100:.1f}%",
        flush=True)

    print("[4/5] Running PACT on JPEG QF=70 degraded images...", flush=True)
    results_jpeg = []
    with torch.no_grad():
        for s in samples[:n_test]:
            try:
                img = Image.open(s["img_path"]).convert("RGB")
                img_deg = jpeg_compress(img, quality=70)
                x = preprocess(img_deg).unsqueeze(0).cuda().float()
                out = model(x)
                p = out["prob"].squeeze().float().item()
                results_jpeg.append({
                    "gen": s["generator"],
                    "label": s["label"],
                    "p_v": p
                })
            except Exception as e:
                print(f"  fail: {e}", flush=True)
    accuracies_jpeg = sum(
        int((r["p_v"] >= 0.5) == r["label"])
        for r in results_jpeg) / max(len(results_jpeg), 1)
    print(
        f"  -> PACT acc on {len(results_jpeg)} JPEG-QF70: {accuracies_jpeg*100:.1f}%",
        flush=True)

    print("[5/5] Running PACT on Gaussian blur sigma=2 degraded images...",
          flush=True)
    results_blur = []
    with torch.no_grad():
        for s in samples[:n_test]:
            try:
                img = Image.open(s["img_path"]).convert("RGB")
                img_deg = gaussian_blur(img, sigma=2.0)
                x = preprocess(img_deg).unsqueeze(0).cuda().float()
                out = model(x)
                p = out["prob"].squeeze().float().item()
                results_blur.append({
                    "gen": s["generator"],
                    "label": s["label"],
                    "p_v": p
                })
            except Exception as e:
                print(f"  fail: {e}", flush=True)
    accuracies_blur = sum(
        int((r["p_v"] >= 0.5) == r["label"])
        for r in results_blur) / max(len(results_blur), 1)
    print(
        f"  -> PACT acc on {len(results_blur)} blur-sigma2: {accuracies_blur*100:.1f}%",
        flush=True)

    print("\n=== SMOKE SUMMARY ===")
    print(
        f"  Original    : {accuracies_orig*100:.1f}%  ({len(results_orig)} samples)"
    )
    print(
        f"  JPEG QF=70  : {accuracies_jpeg*100:.1f}%  ({len(results_jpeg)} samples)"
    )
    print(
        f"  Blur sigma=2: {accuracies_blur*100:.1f}%  ({len(results_blur)} samples)"
    )
    print("=== SMOKE PASSED ===" if results_orig and results_jpeg
          and results_blur else "=== SMOKE INCOMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
