"""
Step 1: Cache XGenDet v1_finetune inference on all HydraFake training images.
Saves: confidence, family, attr_scores per image.
"""

import os, sys, json, torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xgendet import XGenDet
from data.augmentations import get_eval_transforms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load best v1_finetune model
    ckpt_path = "checkpoints/hydrafake_finetune/best_model.pth"
    print(f"Loading: {ckpt_path}")
    model = XGenDet()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    transform = get_eval_transforms(224)

    # Load SFT JSON
    sft_path = "/home/sachin.chaudhary/hydrafake/jsons/train/sft_36k.json"
    with open(sft_path) as f:
        sft_data = json.load(f)
    print(f"Processing {len(sft_data)} images...")

    data_root = "/home/sachin.chaudhary"
    results = {}
    attr_names = ["texture", "edges", "color", "geometry", "semantics", "frequency"]

    for item in tqdm(sft_data, desc="Caching XGenDet features"):
        rel_path = item["images"][0]
        full_path = os.path.join(data_root, rel_path)

        if not os.path.exists(full_path):
            continue

        try:
            img = Image.open(full_path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(img_t, return_heatmap=False)
                conf = out["confidence"].item()
                family_idx = out["family_logit"].argmax(dim=-1).item()
                attrs = out["attr_scores"][0].cpu().tolist()

            families = ["Real", "Face Swapping", "Entire Face Gen", "Face Reenactment"]
            results[rel_path] = {
                "confidence": round(conf, 4),
                "family": families[family_idx],
                "family_idx": family_idx,
                "attr_scores": {name: round(v, 4) for name, v in zip(attr_names, attrs)},
            }
        except Exception as e:
            continue

    # Save cached features
    out_path = "checkpoints/xgendet_cached_features.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCached {len(results)} image features to {out_path}")


if __name__ == "__main__":
    main()
