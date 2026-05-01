"""
Extract face part crops (eyes, nose, mouth) from HydraFake images.

Uses OpenCV Haar cascades for face detection, then crops approximate
facial regions based on proportional geometry.

Output: Saves 4 part crops per image to output_dir/
  - {image_hash}_left_eye.jpg
  - {image_hash}_right_eye.jpg
  - {image_hash}_nose.jpg
  - {image_hash}_mouth.jpg

Also saves a JSON mapping image_path → part crop paths.

Usage:
    python training/extract_face_parts.py \
        --data_json /home/sachin.chaudhary/hydrafake/jsons/train/sft_36k.json \
        --data_root /home/sachin.chaudhary \
        --output_dir /home/sachin.chaudhary/hydrafake/face_parts \
        --num_workers 4
"""

import os, sys, json, argparse, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_json", required=True)
    p.add_argument("--data_root", default="/home/sachin.chaudhary")
    p.add_argument("--output_dir", default="/home/sachin.chaudhary/hydrafake/face_parts")
    p.add_argument("--crop_size", type=int, default=112, help="Size of each part crop")
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def get_face_bbox(img_bgr):
    """Detect face using Haar cascade. Returns (x, y, w, h) or None."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        # Fallback: use entire image as face region
        h, w = img_bgr.shape[:2]
        margin = int(min(h, w) * 0.1)
        return (margin, margin, w - 2*margin, h - 2*margin)

    # Take largest face
    areas = [w * h for (x, y, w, h) in faces]
    idx = np.argmax(areas)
    return tuple(faces[idx])


def crop_face_parts(img_bgr, face_bbox, crop_size=112):
    """
    Given a face bounding box, extract 4 part crops using proportional geometry.

    Face proportions (approximate):
      - Eyes: 25-45% height, 10-90% width
      - Nose: 40-65% height, 30-70% width
      - Mouth: 60-85% height, 20-80% width

    Returns dict of part_name → PIL Image crops.
    """
    fx, fy, fw, fh = face_bbox
    h, w = img_bgr.shape[:2]

    # Clamp face bbox to image
    fx, fy = max(0, fx), max(0, fy)
    fw = min(fw, w - fx)
    fh = min(fh, h - fy)

    def safe_crop(y1_pct, y2_pct, x1_pct, x2_pct):
        y1 = fy + int(fh * y1_pct)
        y2 = fy + int(fh * y2_pct)
        x1 = fx + int(fw * x1_pct)
        x2 = fx + int(fw * x2_pct)
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)
        if y2 <= y1 or x2 <= x1:
            return None
        crop = img_bgr[y1:y2, x1:x2]
        crop = cv2.resize(crop, (crop_size, crop_size))
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    parts = {}
    # Left eye (viewer's left = face's right)
    parts["left_eye"] = safe_crop(0.20, 0.45, 0.05, 0.45)
    # Right eye
    parts["right_eye"] = safe_crop(0.20, 0.45, 0.55, 0.95)
    # Nose
    parts["nose"] = safe_crop(0.35, 0.65, 0.25, 0.75)
    # Mouth
    parts["mouth"] = safe_crop(0.60, 0.85, 0.15, 0.85)

    return parts


def path_hash(path):
    """Short hash for unique filename."""
    return hashlib.md5(path.encode()).hexdigest()[:12]


def process_single_image(args_tuple):
    """Process a single image. Returns (rel_path, part_paths_dict) or (rel_path, None)."""
    rel_path, data_root, output_dir, crop_size = args_tuple
    full_path = os.path.join(data_root, rel_path)

    try:
        img_bgr = cv2.imread(full_path)
        if img_bgr is None:
            return rel_path, None

        face_bbox = get_face_bbox(img_bgr)
        parts = crop_face_parts(img_bgr, face_bbox, crop_size)

        img_hash = path_hash(rel_path)
        part_paths = {}
        for part_name, crop in parts.items():
            if crop is None:
                continue
            fname = f"{img_hash}_{part_name}.jpg"
            save_path = os.path.join(output_dir, fname)
            crop.save(save_path, quality=95)
            part_paths[part_name] = fname

        return rel_path, part_paths

    except Exception as e:
        return rel_path, None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    with open(args.data_json) as f:
        data = json.load(f)
    image_paths = [d["images"][0] for d in data]
    print(f"Processing {len(image_paths)} images for face part extraction...")

    # Process in parallel
    work_items = [
        (path, args.data_root, args.output_dir, args.crop_size)
        for path in image_paths
    ]

    results = {}
    failed = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(process_single_image, item): item[0] for item in work_items}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting face parts"):
            rel_path, part_paths = future.result()
            if part_paths is not None:
                results[rel_path] = part_paths
            else:
                failed += 1

    # Save mapping
    output_path = os.path.join(args.output_dir, "face_parts_map.json")
    with open(output_path, "w") as f:
        json.dump(results, f)

    print(f"Done. {len(results)} images processed, {failed} failed.")
    print(f"Face parts mapping saved to {output_path}")

    # Print stats
    part_counts = {}
    for parts in results.values():
        for part_name in parts:
            part_counts[part_name] = part_counts.get(part_name, 0) + 1
    for part, count in sorted(part_counts.items()):
        print(f"  {part}: {count} crops")


if __name__ == "__main__":
    main()
