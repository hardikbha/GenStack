#!/usr/bin/env python3
"""
D3 Baseline Evaluation on OOD Generator Images.

Loads the trained D3 (CVPR 2025) checkpoint and runs inference on test images
from OOD_GENERATORS to establish baseline numbers for comparison with XGenDet.

D3 Architecture:
  - Backbone: CLIP ViT-L/14 (frozen, 427.6M params)
  - Head: TransformerAttention (Q/K/V + FC, 3.15M trainable params)
  - Dual branch: original image + patch-shuffled image -> attention fusion
  - Total: 430.8M params, 0.73% trainable

Usage:
  conda activate D3
  python scripts/d3_baseline_eval.py

If the D3 conda env is not available, run with any env that has torch + CLIP:
  python scripts/d3_baseline_eval.py --checkpoint /path/to/model_epoch_best.pth
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
D3_ROOT = "/home/sachin.chaudhary/GTA/D3"
CHECKPOINT_PATHS = [
    "/home/sachin.chaudhary/GTA/D3/checkpoints/train_d3/model_epoch_best.pth",
    "/home/sachin.chaudhary/GTA/D3/ckpt/train_d3/model_epoch_best.pth",
]
OOD_BASE = "/home/sachin.chaudhary/GTA/OOD_GENERATORS"

# OOD generators with available images
OOD_GENERATORS = [
    "dalle", "gaugan", "stargan", "san", "deepfake",
    "whichfaceisreal", "cyclegan", "imle", "biggan", "crn", "seeingdark",
]

# Test images: pick 1 real + 1 fake from 3 different generators
TEST_GENERATORS = ["dalle", "gaugan", "deepfake"]
IMAGES_PER_GENERATOR = 3  # real + fake pairs


def parse_args():
    parser = argparse.ArgumentParser(description="D3 Baseline Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--full-eval", action="store_true",
                        help="Run full evaluation on all OOD generators (not just 3 test images)")
    parser.add_argument("--max-per-generator", type=int, default=500,
                        help="Max images per class (real/fake) for full eval")
    return parser.parse_args()


def divider(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def find_checkpoint(override=None):
    """Find a valid D3 checkpoint."""
    if override and os.path.exists(override):
        return override
    for path in CHECKPOINT_PATHS:
        if os.path.exists(path):
            return path
    return None


def find_test_images(generators, base_dir, n_per_class=3):
    """Find n real + n fake images for each generator."""
    test_images = []
    for gen in generators:
        real_dir = os.path.join(base_dir, gen, "0_real")
        fake_dir = os.path.join(base_dir, gen, "1_fake")
        if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
            print(f"  WARNING: {gen} missing real/fake dirs, skipping")
            continue

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        real_imgs = sorted([
            os.path.join(real_dir, f) for f in os.listdir(real_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])[:n_per_class]
        fake_imgs = sorted([
            os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])[:n_per_class]

        for p in real_imgs:
            test_images.append({"path": p, "label": 0, "generator": gen, "class": "real"})
        for p in fake_imgs:
            test_images.append({"path": p, "label": 1, "generator": gen, "class": "fake"})

    return test_images


def load_d3_model(checkpoint_path, device):
    """Load D3 model with CLIP ViT-L/14 backbone + attention head."""
    sys.path.insert(0, D3_ROOT)
    from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer

    print(f"  Building CLIPModelShuffleAttentionPenultimateLayer (ViT-L/14)...")
    model = CLIPModelShuffleAttentionPenultimateLayer(
        "ViT-L/14",
        shuffle_times=1,
        original_times=1,
        patch_size=[14],
    )

    print(f"  Loading checkpoint: {checkpoint_path}")
    import torch
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.attention_head.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"  Model loaded successfully")
    print(f"  Total parameters:     {total_params:>12,}  ({total_params/1e6:.1f}M)")
    print(f"  Trainable parameters: {trainable_params:>12,}  ({trainable_params/1e6:.2f}M)")
    print(f"  Frozen parameters:    {frozen_params:>12,}  ({frozen_params/1e6:.1f}M)")
    print(f"  Trainable ratio:      {100*trainable_params/total_params:.2f}%")

    return model, trainable_params, total_params


def get_transform():
    """CLIP-compatible transform for D3."""
    import torchvision.transforms as transforms
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def run_inference_single(model, image_path, transform, device):
    """Run D3 inference on a single image. Returns sigmoid confidence."""
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        if output.shape[-1] == 2:
            output = output[:, 0]
        confidence = output.sigmoid().item()

    return confidence


def run_batch_inference(model, image_paths, labels, transform, device, batch_size=64):
    """Run D3 inference on a batch of images."""
    import torch
    from PIL import Image
    import numpy as np

    all_confidences = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
            except Exception:
                # Use zero tensor as placeholder for corrupt images
                tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            output = model(batch)
            if output.shape[-1] == 2:
                output = output[:, 0]
            confidences = output.sigmoid().flatten().tolist()
        all_confidences.extend(confidences)

    return np.array(all_confidences)


def print_published_results():
    """Print D3's published OOD results from the validation logs."""
    divider("D3 Published OOD Results (from validation logs)")

    # Results from the first validation run (using ckpt/classifier.pth - original pretrained)
    pretrained_results = {
        "dalle":            {"acc_05": 0.9090, "ap": 0.9786, "best_acc": 0.9255},
        "gaugan":           {"acc_05": 0.9857, "ap": 0.9995, "best_acc": 0.9905},
        "stargan":          {"acc_05": 0.9390, "ap": 0.9905, "best_acc": 0.9542},
        "san":              {"acc_05": 0.6347, "ap": 0.6901, "best_acc": 0.6530},
        "stylegan":         {"acc_05": 0.9234, "ap": 0.9805, "best_acc": 0.9236},
        "whichfaceisreal":  {"acc_05": 0.7690, "ap": 0.8940, "best_acc": 0.7995},
        "stylegan2":        {"acc_05": 0.9060, "ap": 0.9730, "best_acc": 0.9091},
        "cyclegan":         {"acc_05": 0.9319, "ap": 0.9851, "best_acc": 0.9330},
        "imle":             {"acc_05": 0.8856, "ap": 0.9836, "best_acc": 0.9408},
        "biggan":           {"acc_05": 0.9932, "ap": 0.9998, "best_acc": 0.9952},
        "crn":              {"acc_05": 0.8313, "ap": 0.9238, "best_acc": 0.8360},
        "seeingdark":       {"acc_05": 0.7333, "ap": 0.8363, "best_acc": 0.7528},
        "deepfake":         {"acc_05": 0.7361, "ap": 0.8747, "best_acc": 0.7993},
    }

    # Results from second run (using checkpoints/train_d3/model_epoch_best.pth - retrained)
    retrained_results = {
        "dalle":            {"acc_05": 0.9050, "ap": 0.9778, "best_acc": 0.9170},
        "gaugan":           {"acc_05": 0.9784, "ap": 0.9989, "best_acc": 0.9870},
        "stargan":          {"acc_05": 0.9257, "ap": 0.9826, "best_acc": 0.9320},
        "san":              {"acc_05": 0.6575, "ap": 0.6805, "best_acc": 0.6621},
        "stylegan":         {"acc_05": 0.9140, "ap": 0.9769, "best_acc": 0.9155},
        "whichfaceisreal":  {"acc_05": 0.8240, "ap": 0.9038, "best_acc": 0.8245},
        "stylegan2":        {"acc_05": 0.9049, "ap": 0.9724, "best_acc": 0.9085},
        "cyclegan":         {"acc_05": 0.9103, "ap": 0.9772, "best_acc": 0.9262},
        "imle":             {"acc_05": 0.8573, "ap": 0.9799, "best_acc": 0.9447},
        "biggan":           {"acc_05": 0.9905, "ap": 0.9996, "best_acc": 0.9930},
        "crn":              {"acc_05": 0.8415, "ap": 0.9370, "best_acc": 0.8720},
        "seeingdark":       {"acc_05": 0.7222, "ap": 0.8249, "best_acc": 0.7306},
        "deepfake":         {"acc_05": 0.7500, "ap": 0.8577, "best_acc": 0.7822},
    }

    # Published paper results (from README.MD)
    print("\n  D3 Paper Results (CVPR 2025):")
    print(f"    ID Accuracy:    96.6%")
    print(f"    OOD Accuracy:   86.7%")
    print(f"    Total Accuracy: 90.7%")

    print("\n  D3 Logged OOD Results (pretrained checkpoint, classifier.pth):")
    print(f"    {'Generator':<20s}  {'Acc@0.5':>8s}  {'AP':>8s}  {'Best Acc':>8s}")
    print(f"    {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}")
    for gen, r in pretrained_results.items():
        print(f"    {gen:<20s}  {r['acc_05']*100:>7.1f}%  {r['ap']*100:>7.1f}%  {r['best_acc']*100:>7.1f}%")
    avg_acc = sum(r["acc_05"] for r in pretrained_results.values()) / len(pretrained_results)
    avg_ap = sum(r["ap"] for r in pretrained_results.values()) / len(pretrained_results)
    avg_best = sum(r["best_acc"] for r in pretrained_results.values()) / len(pretrained_results)
    print(f"    {'AVERAGE':<20s}  {avg_acc*100:>7.1f}%  {avg_ap*100:>7.1f}%  {avg_best*100:>7.1f}%")

    print("\n  D3 Logged OOD Results (retrained checkpoint, model_epoch_best.pth):")
    print(f"    {'Generator':<20s}  {'Acc@0.5':>8s}  {'AP':>8s}  {'Best Acc':>8s}")
    print(f"    {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}")
    for gen, r in retrained_results.items():
        print(f"    {gen:<20s}  {r['acc_05']*100:>7.1f}%  {r['ap']*100:>7.1f}%  {r['best_acc']*100:>7.1f}%")
    avg_acc = sum(r["acc_05"] for r in retrained_results.values()) / len(retrained_results)
    avg_ap = sum(r["ap"] for r in retrained_results.values()) / len(retrained_results)
    avg_best = sum(r["best_acc"] for r in retrained_results.values()) / len(retrained_results)
    print(f"    {'AVERAGE':<20s}  {avg_acc*100:>7.1f}%  {avg_ap*100:>7.1f}%  {avg_best*100:>7.1f}%")

    return pretrained_results, retrained_results


def main():
    args = parse_args()

    print("=" * 72)
    print("  D3 Baseline Evaluation for XGenDet Comparison")
    print("=" * 72)

    # ---------------------------------------------------------------------------
    # 0. Print published results (always available, even without GPU)
    # ---------------------------------------------------------------------------
    pretrained_results, retrained_results = print_published_results()

    # ---------------------------------------------------------------------------
    # 1. Find checkpoint
    # ---------------------------------------------------------------------------
    divider("1. Locating D3 Checkpoint")
    ckpt_path = find_checkpoint(args.checkpoint)
    if ckpt_path is None:
        print("  ERROR: No D3 checkpoint found!")
        print("  Searched:")
        for p in CHECKPOINT_PATHS:
            print(f"    {p}")
        print("\n  Cannot run live inference. Published results above are still valid.")
        return
    print(f"  Found checkpoint: {ckpt_path}")
    print(f"  Size: {os.path.getsize(ckpt_path) / 1e6:.1f} MB")

    # ---------------------------------------------------------------------------
    # 2. Setup device
    # ---------------------------------------------------------------------------
    import torch
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------------------------
    # 3. Load model
    # ---------------------------------------------------------------------------
    divider("2. Loading D3 Model")
    try:
        model, trainable_params, total_params = load_d3_model(ckpt_path, device)
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        traceback.print_exc()
        print("\n  Cannot run live inference. Published results above are still valid.")
        return

    # ---------------------------------------------------------------------------
    # 4. Find test images
    # ---------------------------------------------------------------------------
    divider("3. Selecting Test Images")
    test_images = find_test_images(TEST_GENERATORS, OOD_BASE, n_per_class=IMAGES_PER_GENERATOR)
    if not test_images:
        print("  ERROR: No test images found!")
        return

    print(f"  Found {len(test_images)} test images from {len(TEST_GENERATORS)} generators:")
    for gen in TEST_GENERATORS:
        gen_imgs = [t for t in test_images if t["generator"] == gen]
        real_count = sum(1 for t in gen_imgs if t["label"] == 0)
        fake_count = sum(1 for t in gen_imgs if t["label"] == 1)
        print(f"    {gen}: {real_count} real + {fake_count} fake")

    # ---------------------------------------------------------------------------
    # 5. Run inference on test images
    # ---------------------------------------------------------------------------
    divider("4. Running D3 Inference on Test Images")
    transform = get_transform()

    results = []
    correct = 0
    total = 0
    t0 = time.time()

    for item in test_images:
        try:
            conf = run_inference_single(model, item["path"], transform, device)
            pred_label = 1 if conf >= 0.5 else 0
            is_correct = pred_label == item["label"]
            if is_correct:
                correct += 1
            total += 1

            verdict = "FAKE" if pred_label == 1 else "REAL"
            gt = "FAKE" if item["label"] == 1 else "REAL"
            check = "OK" if is_correct else "WRONG"

            results.append({
                "generator": item["generator"],
                "class": item["class"],
                "gt": gt,
                "pred": verdict,
                "confidence": conf,
                "correct": is_correct,
                "path": item["path"],
            })

            print(f"  [{check:>5s}] {item['generator']:<18s} GT={gt:<4s}  Pred={verdict:<4s}  "
                  f"Conf={conf:.4f}  {os.path.basename(item['path'])}")
        except Exception as e:
            print(f"  [ERROR] {item['path']}: {e}")

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0

    print(f"\n  Inference time: {elapsed:.2f}s ({elapsed/total:.3f}s per image)")
    print(f"  Overall accuracy: {correct}/{total} = {acc*100:.1f}%")

    # Per-generator accuracy
    gen_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        gen_results[r["generator"]]["total"] += 1
        if r["correct"]:
            gen_results[r["generator"]]["correct"] += 1

    print(f"\n  Per-generator accuracy:")
    for gen in TEST_GENERATORS:
        gr = gen_results[gen]
        g_acc = gr["correct"] / gr["total"] if gr["total"] > 0 else 0
        print(f"    {gen:<18s}: {gr['correct']}/{gr['total']} = {g_acc*100:.1f}%")

    # ---------------------------------------------------------------------------
    # 6. Full evaluation (optional)
    # ---------------------------------------------------------------------------
    if args.full_eval:
        divider("5. Full OOD Evaluation")
        import numpy as np
        from sklearn.metrics import average_precision_score, accuracy_score

        full_results = {}
        for gen in OOD_GENERATORS:
            real_dir = os.path.join(OOD_BASE, gen, "0_real")
            fake_dir = os.path.join(OOD_BASE, gen, "1_fake")
            if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
                print(f"  Skipping {gen}: directory not found")
                continue

            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            real_imgs = sorted([
                os.path.join(real_dir, f) for f in os.listdir(real_dir)
                if os.path.splitext(f)[1].lower() in exts
            ])[:args.max_per_generator]
            fake_imgs = sorted([
                os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                if os.path.splitext(f)[1].lower() in exts
            ])[:args.max_per_generator]

            if not real_imgs or not fake_imgs:
                print(f"  Skipping {gen}: no images found")
                continue

            n = min(len(real_imgs), len(fake_imgs))
            real_imgs = real_imgs[:n]
            fake_imgs = fake_imgs[:n]

            all_paths = real_imgs + fake_imgs
            all_labels = np.array([0]*len(real_imgs) + [1]*len(fake_imgs))

            print(f"  Evaluating {gen}: {n} real + {n} fake images...", end="", flush=True)
            t1 = time.time()
            confs = run_batch_inference(model, all_paths, all_labels, transform, device)
            elapsed_gen = time.time() - t1

            preds = (confs >= 0.5).astype(int)
            acc_gen = accuracy_score(all_labels, preds)
            try:
                ap_gen = average_precision_score(all_labels, confs)
            except Exception:
                ap_gen = float("nan")

            real_acc = accuracy_score(all_labels[all_labels==0], preds[all_labels==0])
            fake_acc = accuracy_score(all_labels[all_labels==1], preds[all_labels==1])

            full_results[gen] = {
                "acc": acc_gen, "ap": ap_gen,
                "real_acc": real_acc, "fake_acc": fake_acc,
                "n": n, "time": elapsed_gen,
            }
            print(f" Acc={acc_gen*100:.1f}% AP={ap_gen*100:.1f}% ({elapsed_gen:.1f}s)")

        if full_results:
            print(f"\n  {'Generator':<18s}  {'Acc':>8s}  {'AP':>8s}  {'Real Acc':>8s}  {'Fake Acc':>8s}  {'N':>6s}")
            print(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
            for gen, r in full_results.items():
                print(f"  {gen:<18s}  {r['acc']*100:>7.1f}%  {r['ap']*100:>7.1f}%  "
                      f"{r['real_acc']*100:>7.1f}%  {r['fake_acc']*100:>7.1f}%  {r['n']:>6d}")
            avg_acc = sum(r["acc"] for r in full_results.values()) / len(full_results)
            avg_ap = sum(r["ap"] for r in full_results.values()) / len(full_results)
            print(f"  {'AVERAGE':<18s}  {avg_acc*100:>7.1f}%  {avg_ap*100:>7.1f}%")

    # ---------------------------------------------------------------------------
    # 7. Summary / Comparison
    # ---------------------------------------------------------------------------
    divider("SUMMARY: D3 vs XGenDet Comparison Framework")

    print("""
  D3 (CVPR 2025) Architecture:
    Backbone:        CLIP ViT-L/14 (frozen)
    Head:            TransformerAttention (Q/K/V linear projections + FC)
    Dual Branch:     Original image + patch-shuffled image -> attention fusion
    Total params:    430.8M
    Trainable:       3.15M (0.73% of total)
    Input:           224x224 (CenterCrop from 256)
    Training:        BCEWithLogitsLoss, AdamW, lr=1e-4, batch_size=512

  D3 Published Performance:
    Paper (CVPR 2025 Table):
      ID Accuracy:    96.6%
      OOD Accuracy:   86.7%
      Total Accuracy: 90.7%

    Logged OOD results (pretrained ckpt):
      Average Acc@0.5:  86.0%
      Average AP:       93.2%
      Average Best Acc: 87.8%

    Weak spots (below 80%):
      - SAN:            63.5% (acc@0.5)
      - SeeingDark:     73.3%
      - DeepFake:       73.6%
      - WhichFaceIsReal: 76.9%

  XGenDet Target:
    Same backbone (CLIP ViT-L/14, frozen) PLUS:
      - Learnable prompt tokens
      - Multi-layer feature extraction
      - Prototype-based Guided Attention (PGAD)
      - Heatmap generator for explainability
      - Family classification head
      - Attribute scoring (6 forensic attributes)

    Goal: Exceed D3's 86.7% OOD accuracy with:
      - Similar or fewer trainable parameters
      - Built-in explainability (heatmaps, prototypes, attributes)
      - No additional inference overhead from patch shuffling
""")

    # Print live inference results summary if we ran them
    if total > 0:
        print(f"  Live Inference Results ({total} images, {len(TEST_GENERATORS)} generators):")
        print(f"    D3 accuracy: {acc*100:.1f}%")
        for gen in TEST_GENERATORS:
            gr = gen_results[gen]
            g_acc = gr["correct"] / gr["total"] if gr["total"] > 0 else 0
            print(f"    {gen:<18s}: {g_acc*100:.1f}%")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
