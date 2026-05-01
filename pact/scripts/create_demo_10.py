"""
Create 10-image demo for presentation: original + bounding box overlay + reasons
Draws bounding boxes (not heatmaps) around detected anomaly regions.
"""
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys

OUTPUT_DIR = "/home/sachin.chaudhary/xgendet/demo_outputs"
HEATMAP_DIR = "/home/sachin.chaudhary/xgendet/eval_outputs/heatmaps"
ORIGINAL_DIR = "/home/sachin.chaudhary/xgendet/eval_outputs/originals"
PREDICTIONS_FILE = "/home/sachin.chaudhary/xgendet/eval_outputs/xgendet_predictions_enhanced.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load predictions
preds = []
with open(PREDICTIONS_FILE) as f:
    for line in f:
        preds.append(json.loads(line))

# Build filename -> prediction map
pred_map = {}
for p in preds:
    orig_path = p["image_path"]
    # Find the corresponding file in originals/
    basename = os.path.basename(p.get("heatmap_path", ""))
    if basename:
        pred_map[basename] = p

# Select 10 diverse demo images
# Pick: 2 real (correctly classified), 8 fake from different generators
DEMO_SELECTIONS = [
    # (generator, label, description for selection)
    ("deepfake", "fake", "Face deepfake - face artifacts"),
    ("stylegan", "fake", "StyleGAN face - texture artifacts"),
    ("stargan", "fake", "StarGAN face - hair/skin artifacts"),
    ("dalle", "fake", "DALL-E scene - semantic artifacts"),
    ("cyclegan", "fake", "CycleGAN - style transfer artifacts"),
    ("gaugan", "fake", "GauGAN scene - layout artifacts"),
    ("ADM", "fake", "ADM diffusion - texture artifacts"),
    ("BigGAN", "fake", "BigGAN - class-conditional artifacts"),
    ("ADM", "real", "Real photo - correctly identified"),
    ("stylegan2", "real", "Real photo - correctly identified"),
]


def extract_bboxes_from_heatmap(heatmap_img, threshold_pct=0.6, min_area=100):
    """Convert a heatmap image to bounding boxes.

    Args:
        heatmap_img: PIL Image of the 3-panel heatmap (original | heatmap | overlay)
        threshold_pct: percentile threshold for bounding box extraction
        min_area: minimum area for a bounding box

    Returns:
        list of (x1, y1, x2, y2) bounding boxes
    """
    arr = np.array(heatmap_img)
    h, w = arr.shape[:2]

    # The heatmap image is a 3-panel: original | heatmap | overlay
    # Each panel is w/3 wide
    panel_w = w // 3

    # Extract the heatmap panel (middle)
    heatmap_panel = arr[:, panel_w:2*panel_w, :]

    # Convert to grayscale-like intensity (red channel is strongest in jet colormap)
    # For red-based heatmaps: high red = hot
    red = heatmap_panel[:, :, 0].astype(float)
    green = heatmap_panel[:, :, 1].astype(float)
    blue = heatmap_panel[:, :, 2].astype(float)

    # Heatmap intensity: red regions are suspicious
    # Jet colormap: hot = red (R high, G low, B low)
    intensity = red - 0.5 * green - 0.5 * blue
    intensity = np.clip(intensity, 0, 255)

    if intensity.max() == 0:
        return []

    # Normalize
    intensity = intensity / intensity.max()

    # Threshold
    thresh = np.percentile(intensity[intensity > 0], threshold_pct * 100) if np.any(intensity > 0) else 0.5
    binary = (intensity > thresh).astype(np.uint8)

    # Find connected components using simple flood fill
    bboxes = []
    visited = np.zeros_like(binary, dtype=bool)

    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            if binary[y, x] == 1 and not visited[y, x]:
                # BFS to find connected component
                queue = [(y, x)]
                visited[y, x] = True
                min_y, max_y, min_x, max_x = y, y, x, x

                while queue:
                    cy, cx = queue.pop(0)
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)

                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < binary.shape[0] and 0 <= nx < binary.shape[1]:
                            if binary[ny, nx] == 1 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                area = (max_y - min_y) * (max_x - min_x)
                if area >= min_area:
                    # Add padding
                    pad = 3
                    min_y = max(0, min_y - pad)
                    min_x = max(0, min_x - pad)
                    max_y = min(binary.shape[0]-1, max_y + pad)
                    max_x = min(binary.shape[1]-1, max_x + pad)
                    bboxes.append((min_x, min_y, max_x, max_y))

    # Merge overlapping boxes
    if len(bboxes) > 5:
        # Keep top 3 by area
        bboxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        bboxes = bboxes[:3]

    return bboxes


def create_demo_image(original_path, heatmap_path, prediction, output_path):
    """Create a demo image with bounding boxes and annotation text."""

    # Load original image
    orig = Image.open(original_path).convert("RGB")
    orig_w, orig_h = orig.size

    # Load heatmap panel image
    heatmap_full = Image.open(heatmap_path).convert("RGB")

    # Extract bounding boxes from heatmap
    bboxes = extract_bboxes_from_heatmap(heatmap_full, threshold_pct=0.55)

    # Create annotated image with bounding boxes
    annotated = orig.copy()
    draw = ImageDraw.Draw(annotated)

    is_fake = prediction["prediction"] == "fake"
    box_color = (255, 50, 50) if is_fake else (50, 200, 50)  # Red for fake, green for real

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        # Scale bbox from heatmap panel size to original image size
        hm_w = heatmap_full.size[0] // 3
        hm_h = heatmap_full.size[1]
        sx = orig_w / hm_w
        sy = orig_h / hm_h
        x1s, y1s = int(x1 * sx), int(y1 * sy)
        x2s, y2s = int(x2 * sx), int(y2 * sy)

        # Draw bounding box (thick)
        for t in range(3):
            draw.rectangle([x1s-t, y1s-t, x2s+t, y2s+t], outline=box_color)

    # Create the final composite: original | annotated with bbox | info panel
    panel_w = orig_w
    info_w = 280
    canvas_w = panel_w * 2 + info_w + 20  # 10px gap each
    canvas_h = max(orig_h, 224)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    # Paste original
    canvas.paste(orig, (0, 0))

    # Paste annotated
    canvas.paste(annotated, (panel_w + 10, 0))

    # Draw info panel
    info_draw = ImageDraw.Draw(canvas)
    x_info = panel_w * 2 + 20
    y_pos = 5

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 13)
            font_small = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()
            font_small = font

    # Prediction label
    label = f"{'FAKE' if is_fake else 'REAL'} ({prediction['confidence']*100:.1f}%)"
    label_color = (200, 30, 30) if is_fake else (30, 150, 30)
    info_draw.text((x_info, y_pos), label, fill=label_color, font=font)
    y_pos += 18

    # Generator family
    family = prediction.get("family_pred", "Unknown")
    info_draw.text((x_info, y_pos), f"Family: {family}", fill=(60, 60, 60), font=font_small)
    y_pos += 15

    # Bounding boxes count
    info_draw.text((x_info, y_pos), f"Anomaly regions: {len(bboxes)}", fill=(60, 60, 60), font=font_small)
    y_pos += 18

    # Attribute scores
    attrs = prediction.get("attributes", {})
    info_draw.text((x_info, y_pos), "Artifact Scores:", fill=(0, 0, 0), font=font)
    y_pos += 16

    attr_names = ["texture", "edges", "color", "geometry", "semantics", "frequency"]
    for attr in attr_names:
        val = attrs.get(attr, attrs.get(attr.capitalize(), 0))
        # Color based on score: higher = more artifacts
        r = int(min(255, val * 2 * 255))
        g = int(min(255, (1 - val) * 2 * 255))
        bar_color = (r, g, 50)

        info_draw.text((x_info, y_pos), f"{attr[:7]:>7}: {val:.2f}", fill=(40, 40, 40), font=font_small)
        # Draw mini bar
        bar_x = x_info + 100
        bar_w = int(val * 120)
        info_draw.rectangle([bar_x, y_pos+2, bar_x + bar_w, y_pos + 10], fill=bar_color)
        info_draw.rectangle([bar_x, y_pos+2, bar_x + 120, y_pos + 10], outline=(180, 180, 180))
        y_pos += 13

    y_pos += 5

    # Explanation (wrap text)
    explanation = prediction.get("explanation", "No explanation available")
    info_draw.text((x_info, y_pos), "Why:", fill=(0, 0, 0), font=font)
    y_pos += 15

    # Word wrap explanation
    words = explanation.split()
    line = ""
    max_chars = 32
    for word in words:
        if len(line) + len(word) + 1 > max_chars:
            info_draw.text((x_info, y_pos), line.strip(), fill=(80, 80, 80), font=font_small)
            y_pos += 12
            line = word + " "
            if y_pos > canvas_h - 15:
                break
        else:
            line += word + " "
    if line.strip() and y_pos <= canvas_h - 15:
        info_draw.text((x_info, y_pos), line.strip(), fill=(80, 80, 80), font=font_small)

    # Add labels at top
    header_draw = ImageDraw.Draw(canvas)

    canvas.save(output_path, quality=95)
    return len(bboxes)


def main():
    # Find matching predictions for our selected generators
    demo_images = []

    for gen, label, desc in DEMO_SELECTIONS:
        target_label = 1 if label == "fake" else 0
        best = None
        best_conf = -1 if label == "fake" else 2.0

        for p in preds:
            path = p["image_path"]
            fname = os.path.basename(p.get("heatmap_path", ""))

            # Check generator match
            gen_lower = gen.lower()
            if gen_lower not in path.lower():
                continue

            # Check label match
            if label == "fake" and p["prediction"] != "fake":
                continue
            if label == "real" and p["prediction"] != "real":
                continue

            conf = p["confidence"]
            if label == "fake" and conf > best_conf:
                best = p
                best_conf = conf
            elif label == "real" and conf < best_conf:
                best = p
                best_conf = conf

        if best:
            demo_images.append((best, gen, label, desc))
            print(f"[OK] {gen} ({label}): conf={best['confidence']:.3f}")
        else:
            print(f"[MISS] {gen} ({label}): no match found")

    print(f"\nCreating {len(demo_images)} demo images...")

    # Create demo outputs
    summary = []
    for i, (pred, gen, label, desc) in enumerate(demo_images):
        heatmap_path = pred.get("heatmap_path", "")
        heatmap_fname = os.path.basename(heatmap_path)

        # Find original in originals/
        orig_fname = heatmap_fname  # Same filename
        orig_path = os.path.join(ORIGINAL_DIR, orig_fname)
        hm_path = os.path.join(HEATMAP_DIR, heatmap_fname)

        if not os.path.exists(orig_path):
            print(f"  [SKIP] Original not found: {orig_path}")
            continue
        if not os.path.exists(hm_path):
            print(f"  [SKIP] Heatmap not found: {hm_path}")
            continue

        output_path = os.path.join(OUTPUT_DIR, f"demo_{i+1:02d}_{gen}_{label}.png")
        n_boxes = create_demo_image(orig_path, hm_path, pred, output_path)

        print(f"  [{i+1}] {gen} ({label}): {n_boxes} boxes -> {output_path}")

        summary.append({
            "index": i + 1,
            "generator": gen,
            "label": label,
            "description": desc,
            "prediction": pred["prediction"],
            "confidence": pred["confidence"],
            "family": pred.get("family_pred", "Unknown"),
            "n_bboxes": n_boxes,
            "attributes": pred.get("attributes", {}),
            "explanation": pred.get("explanation", ""),
            "output_path": output_path,
            "original_path": orig_path,
            "heatmap_path": hm_path,
        })

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "demo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Demo images saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
