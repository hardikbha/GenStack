"""
Extract SRM noise evidence for all HydraFake images.

Computes per-region noise scores using 30 SRM high-pass filters.
Output: JSON file mapping image_path → region noise scores.

Usage:
    python training/extract_srm_evidence.py \
        --data_json /home/sachin.chaudhary/hydrafake/jsons/train/sft_36k.json \
        --data_root /home/sachin.chaudhary \
        --output_dir /home/sachin.chaudhary/hydrafake/srm_evidence \
        --batch_size 64
"""

import os, sys, json, argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.srm_branch import _SRM_KERNELS_3x3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_json", required=True, help="Path to sft_36k.json or all.json")
    p.add_argument("--data_root", default="/home/sachin.chaudhary")
    p.add_argument("--output_dir", default="/home/sachin.chaudhary/hydrafake/srm_evidence")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grid_size", type=int, default=4, help="NxN grid for region scores")
    p.add_argument("--image_size", type=int, default=336)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


class ImagePathDataset(torch.utils.data.Dataset):
    """Simple dataset that loads images by path."""

    def __init__(self, image_paths, data_root, image_size):
        self.paths = image_paths
        self.data_root = data_root
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel_path = self.paths[idx]
        full_path = os.path.join(self.data_root, rel_path)
        try:
            img = Image.open(full_path).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, 336, 336)
        return img, idx


def extract_srm_regions(images, srm_weight, grid_size=4, T=3.0):
    """
    Extract per-region SRM noise scores.

    Args:
        images: [B, 3, H, W] tensor
        srm_weight: [30, 1, 3, 3] SRM filter kernels
        grid_size: NxN grid for region decomposition
        T: truncation threshold

    Returns:
        region_scores: [B, grid_size, grid_size] mean absolute noise per region
        global_score: [B] mean absolute noise across entire image
    """
    # Convert to grayscale
    gray = (0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]).unsqueeze(1)

    # Apply all 30 SRM filters
    residuals = F.conv2d(gray, srm_weight, padding=1)  # [B, 30, H, W]

    # Truncate
    residuals = torch.clamp(residuals / T, -1.0, 1.0)

    # Mean absolute residual across all 30 filters
    abs_residual = residuals.abs().mean(dim=1)  # [B, H, W]

    B, H, W = abs_residual.shape
    global_score = abs_residual.mean(dim=(1, 2))  # [B]

    # Split into grid regions
    rh, rw = H // grid_size, W // grid_size
    region_scores = torch.zeros(B, grid_size, grid_size, device=images.device)
    for gi in range(grid_size):
        for gj in range(grid_size):
            region = abs_residual[:, gi*rh:(gi+1)*rh, gj*rw:(gj+1)*rw]
            region_scores[:, gi, gj] = region.mean(dim=(1, 2))

    return region_scores, global_score


# Map grid positions to approximate face regions
FACE_REGION_NAMES = {
    (0, 0): "forehead_left",    (0, 1): "forehead_center_left",
    (0, 2): "forehead_center_right", (0, 3): "forehead_right",
    (1, 0): "eye_left",         (1, 1): "nose_bridge_left",
    (1, 2): "nose_bridge_right", (1, 3): "eye_right",
    (2, 0): "cheek_left",       (2, 1): "nose_left",
    (2, 2): "nose_right",       (2, 3): "cheek_right",
    (3, 0): "jaw_left",         (3, 1): "mouth_left",
    (3, 2): "mouth_right",      (3, 3): "jaw_right",
}


def format_evidence(region_scores_np, global_score):
    """
    Format SRM evidence as structured text for VLM injection.

    Returns:
        evidence_text: str with region-level noise analysis
    """
    grid_size = region_scores_np.shape[0]
    threshold = np.mean(region_scores_np) + np.std(region_scores_np)

    # Find high-noise regions
    high_regions = []
    low_regions = []
    for i in range(grid_size):
        for j in range(grid_size):
            name = FACE_REGION_NAMES.get((i, j), f"region_{i}_{j}")
            score = float(region_scores_np[i, j])
            if score > threshold:
                high_regions.append((name, score))
            else:
                low_regions.append((name, score))

    high_regions.sort(key=lambda x: -x[1])

    lines = [f"Global noise level: {global_score:.3f}"]
    if high_regions:
        lines.append("High-noise regions: " + ", ".join(
            f"{name} ({score:.3f})" for name, score in high_regions[:4]
        ))
    if low_regions:
        lines.append("Low-noise regions: " + ", ".join(
            f"{name} ({score:.3f})" for name, score in low_regions[:3]
        ))

    # Summary
    if len(high_regions) >= 3:
        lines.append("Assessment: widespread noise inconsistency detected")
    elif len(high_regions) >= 1:
        lines.append(f"Assessment: localized noise anomaly in {high_regions[0][0]}")
    else:
        lines.append("Assessment: noise distribution appears uniform")

    return "\n".join(lines)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    with open(args.data_json) as f:
        data = json.load(f)
    image_paths = [d["images"][0] for d in data]
    print(f"Processing {len(image_paths)} images...")

    # Build SRM kernels
    kernels = torch.from_numpy(_SRM_KERNELS_3x3).unsqueeze(1).float()  # [30, 1, 3, 3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernels = kernels.to(device)

    # Dataset and loader
    dataset = ImagePathDataset(image_paths, args.data_root, args.image_size)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Extract
    all_evidence = {}
    for imgs, indices in tqdm(loader, desc="Extracting SRM evidence"):
        imgs = imgs.to(device)
        with torch.no_grad():
            region_scores, global_scores = extract_srm_regions(
                imgs, kernels, grid_size=args.grid_size
            )

        region_np = region_scores.cpu().numpy()
        global_np = global_scores.cpu().numpy()

        for i, idx in enumerate(indices):
            idx = idx.item()
            path = image_paths[idx]
            evidence_text = format_evidence(region_np[i], global_np[i])
            all_evidence[path] = {
                "region_scores": region_np[i].tolist(),
                "global_score": float(global_np[i]),
                "evidence_text": evidence_text,
            }

    # Save
    output_path = os.path.join(args.output_dir, "srm_evidence.json")
    with open(output_path, "w") as f:
        json.dump(all_evidence, f)
    print(f"Saved SRM evidence for {len(all_evidence)} images to {output_path}")

    # Also save a compact version with just text evidence (for SFT data formatting)
    text_evidence = {k: v["evidence_text"] for k, v in all_evidence.items()}
    text_path = os.path.join(args.output_dir, "srm_evidence_text.json")
    with open(text_path, "w") as f:
        json.dump(text_evidence, f)
    print(f"Saved text evidence to {text_path}")


if __name__ == "__main__":
    main()
