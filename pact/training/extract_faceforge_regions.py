"""
Standalone pre-extraction script: FaceForge-Net region crops.

Runs FaceRegionExtractor on all train + val images in parallel and writes
cached .pkl files to --cache_dir.  Called once from faceforge.pbs before
training.  Subsequent runs skip already-cached images.

Usage:
    python training/extract_faceforge_regions.py \
        --train_json /home/sachin.chaudhary/hydrafake/jsons/train/train.json \
        --val_json   /home/sachin.chaudhary/hydrafake/jsons/train/val.json \
        --data_root  /home/sachin.chaudhary \
        --cache_dir  /home/sachin.chaudhary/hydrafake/faceforge_crops \
        --num_workers 8
"""

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.region_extractor import FaceRegionExtractor


def _cache_key(cache_dir: Path, split: str, image_path: str) -> Path:
    stem = Path(image_path)
    parent_part = stem.parent.name
    cache_name = f"{parent_part}__{stem.stem}.pkl"
    return cache_dir / split / cache_name


def _extract_one(args_tuple):
    image_full_path, cache_path = args_tuple
    if cache_path.exists():
        return "skip"
    extractor = FaceRegionExtractor()
    try:
        crops = extractor.extract(image_full_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(crops, f, protocol=4)
        return "ok"
    except Exception as e:
        return f"error: {e}"


def load_json_paths(json_path: str, data_root: str):
    with open(json_path) as f:
        annotations = json.load(f)
    paths = []
    for item in annotations:
        rel = item["images"][0]
        full = os.path.join(data_root, rel)
        if os.path.exists(full):
            paths.append((rel, full))
    return paths


def extract_split(split_name: str, json_path: str, data_root: str,
                  cache_dir: Path, num_workers: int):
    print(f"\n>>> Processing split: {split_name} ({json_path})")
    paths = load_json_paths(json_path, data_root)
    print(f"    Found {len(paths)} images on disk")

    tasks = []
    for rel, full in paths:
        cp = _cache_key(cache_dir, split_name, rel)
        tasks.append((full, cp))

    already_cached = sum(1 for _, cp in tasks if cp.exists())
    todo = [(full, cp) for full, cp in tasks if not cp.exists()]
    print(f"    Already cached: {already_cached}  |  To extract: {len(todo)}")

    if not todo:
        print(f"    All cached. Skipping.")
        return

    ok = skip = errors = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_extract_one, t): t for t in todo}
        for fut in tqdm(as_completed(futures), total=len(todo),
                        desc=f"  {split_name}", unit="img"):
            result = fut.result()
            if result == "ok":
                ok += 1
            elif result == "skip":
                skip += 1
            else:
                errors += 1

    print(f"    Done: {ok} extracted, {skip} skipped, {errors} errors")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_json", required=True)
    p.add_argument("--val_json", required=True)
    p.add_argument("--data_root", default="/home/sachin.chaudhary")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" FaceForge-Net: Pre-extracting region crops")
    print(f" cache_dir:   {cache_dir}")
    print(f" num_workers: {args.num_workers}")
    print("=" * 60)

    extract_split("train", args.train_json, args.data_root, cache_dir, args.num_workers)
    extract_split("val",   args.val_json,   args.data_root, cache_dir, args.num_workers)

    print("\n>>> Extraction complete.")


if __name__ == "__main__":
    main()
