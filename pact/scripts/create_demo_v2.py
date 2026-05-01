"""
Demo v2: Honest visualization for presentation.
Shows: Original | Heatmap Overlay | Attribute Radar + Explanation

Instead of bounding boxes (which imply false precision), shows:
- Smooth heatmap overlay (what the model actually attends to)
- 6-attribute radar chart (what types of artifacts detected)
- Natural language explanation (why it's fake)
"""
import json
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = "/home/sachin.chaudhary/xgendet/demo_outputs_v2"
HEATMAP_DIR = "/home/sachin.chaudhary/xgendet/eval_outputs/heatmaps"
ORIGINAL_DIR = "/home/sachin.chaudhary/xgendet/eval_outputs/originals"
PREDICTIONS_FILE = "/home/sachin.chaudhary/xgendet/eval_outputs/xgendet_predictions_enhanced.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load predictions
preds = []
with open(PREDICTIONS_FILE) as f:
    for line in f:
        preds.append(json.loads(line))


def get_font(size, bold=False):
    """Try to load a good font, fallback to default."""
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Bold.ttf" if bold else "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except:
            continue
    return ImageFont.load_default()


def extract_overlay(heatmap_path, original_path):
    """Extract the heatmap overlay panel from the 3-panel image,
    or create one from heatmap + original."""
    hm_full = Image.open(heatmap_path).convert("RGB")
    orig = Image.open(original_path).convert("RGB")

    w, h = hm_full.size
    panel_w = w // 3

    # The overlay is the 3rd panel (rightmost)
    overlay = hm_full.crop((2 * panel_w, 0, 3 * panel_w, h))

    # Resize to match original
    overlay = overlay.resize(orig.size, Image.LANCZOS)

    # Also extract pure heatmap (middle panel)
    heatmap = hm_full.crop((panel_w, 0, 2 * panel_w, h))
    heatmap = heatmap.resize(orig.size, Image.LANCZOS)

    return overlay, heatmap


def draw_radar_chart(draw, cx, cy, radius, scores, labels, colors):
    """Draw a simple radar/spider chart."""
    n = len(scores)
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]

    # Draw grid circles
    for r_frac in [0.25, 0.5, 0.75, 1.0]:
        r = radius * r_frac
        points = []
        for a in angles:
            x = cx + r * math.cos(a)
            y = cy + r * math.sin(a)
            points.append((x, y))
        points.append(points[0])
        draw.line(points, fill=(200, 200, 200), width=1)

    # Draw axes
    for a in angles:
        x = cx + radius * math.cos(a)
        y = cy + radius * math.sin(a)
        draw.line([(cx, cy), (x, y)], fill=(180, 180, 180), width=1)

    # Draw filled polygon for scores
    polygon_points = []
    for i, (score, a) in enumerate(zip(scores, angles)):
        r = radius * score
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        polygon_points.append((x, y))

    # Fill
    draw.polygon(polygon_points, fill=(255, 80, 80, 60), outline=(220, 50, 50, 200))

    # Draw score points
    for i, (score, a) in enumerate(zip(scores, angles)):
        r = radius * score
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(220, 50, 50))

    # Draw labels
    font_label = get_font(9)
    for i, (label, a) in enumerate(zip(labels, angles)):
        x = cx + (radius + 14) * math.cos(a)
        y = cy + (radius + 14) * math.sin(a)
        # Center text
        try:
            tw = draw.textlength(label, font=font_label)
        except:
            tw = len(label) * 5
        draw.text((x - tw / 2, y - 5), label, fill=(60, 60, 60), font=font_label)


def word_wrap(text, max_chars=38):
    """Wrap text to lines."""
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > max_chars:
            lines.append(line.strip())
            line = word + " "
        else:
            line += word + " "
    if line.strip():
        lines.append(line.strip())
    return lines


def create_demo_v2(original_path, heatmap_path, prediction, output_path, gen_name):
    """Create presentation-ready demo image."""
    orig = Image.open(original_path).convert("RGB")
    orig_w, orig_h = orig.size

    overlay, heatmap = extract_overlay(heatmap_path, original_path)

    is_fake = prediction["prediction"] == "fake"

    # Canvas: Original(224) + gap + Overlay(224) + gap + Info panel(260)
    info_w = 260
    gap = 8
    canvas_w = orig_w + gap + orig_w + gap + info_w
    canvas_h = max(orig_h + 30, 260)  # Extra 30 for title bar

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    # Title bar
    title_draw = ImageDraw.Draw(canvas)
    font_title = get_font(13, bold=True)
    font_small = get_font(10)
    font_tiny = get_font(9)

    title_color = (200, 30, 30) if is_fake else (30, 140, 30)
    conf_pct = prediction["confidence"] * 100
    title_text = f"{gen_name} — {'FAKE' if is_fake else 'REAL'} ({conf_pct:.1f}%)"
    family = prediction.get("family_pred", "Unknown")
    title_text += f"  [{family}]"
    title_draw.text((5, 3), title_text, fill=title_color, font=font_title)

    y_start = 22

    # Column labels
    title_draw.text((orig_w // 2 - 25, y_start), "Original", fill=(100, 100, 100), font=font_tiny)
    title_draw.text((orig_w + gap + orig_w // 2 - 35, y_start), "Attention Map", fill=(100, 100, 100), font=font_tiny)

    y_img = y_start + 12

    # Paste original with thin border
    canvas.paste(orig, (0, y_img))
    # Draw border
    border_color = (200, 200, 200)
    title_draw.rectangle([0, y_img, orig_w - 1, y_img + orig_h - 1], outline=border_color)

    # Paste overlay
    canvas.paste(overlay, (orig_w + gap, y_img))
    title_draw.rectangle([orig_w + gap, y_img, orig_w + gap + orig_w - 1, y_img + orig_h - 1], outline=border_color)

    # Info panel
    x_info = orig_w * 2 + gap * 2
    y_pos = y_img

    # Attribute radar chart
    attrs = prediction.get("attributes", {})
    attr_names = ["texture", "edges", "color", "geometry", "semantics", "frequency"]
    attr_labels = ["Texture", "Edges", "Color", "Geom.", "Semantic", "Freq."]
    scores = [attrs.get(a, attrs.get(a.capitalize(), 0.5)) for a in attr_names]

    # Draw radar chart
    radar_cx = x_info + 70
    radar_cy = y_pos + 65
    radar_r = 48

    # Need RGBA for transparency
    draw_rgba = ImageDraw.Draw(canvas, "RGBA")
    draw_radar_chart(draw_rgba, radar_cx, radar_cy, radar_r, scores, attr_labels, None)

    title_draw.text((x_info + 20, y_pos - 2), "Artifact Profile", fill=(40, 40, 40), font=get_font(11, bold=True))

    y_pos = radar_cy + radar_r + 20

    # Separator line
    title_draw.line([(x_info, y_pos), (x_info + info_w - 10, y_pos)], fill=(200, 200, 200), width=1)
    y_pos += 5

    # Explanation
    title_draw.text((x_info, y_pos), "Forensic Analysis:", fill=(40, 40, 40), font=get_font(10, bold=True))
    y_pos += 14

    explanation = prediction.get("explanation", "No explanation available.")
    lines = word_wrap(explanation, max_chars=36)
    for line in lines[:8]:  # Max 8 lines
        title_draw.text((x_info, y_pos), line, fill=(80, 80, 80), font=font_tiny)
        y_pos += 12

    # Convert to RGB for saving
    canvas_rgb = Image.new("RGB", canvas.size, (255, 255, 255))
    canvas_rgb.paste(canvas, mask=canvas.split()[3])
    canvas_rgb.save(output_path, quality=95)


def main():
    # Select 10 diverse demo images
    selections = [
        ("deepfake", "fake"),
        ("stylegan", "fake"),
        ("stargan", "fake"),
        ("dalle", "fake"),
        ("cyclegan", "fake"),
        ("gaugan", "fake"),
        ("ADM", "fake"),
        ("BigGAN", "fake"),
        ("ADM", "real"),
        ("stylegan2", "real"),
    ]

    demo_images = []
    for gen, label in selections:
        best = None
        best_conf = -1 if label == "fake" else 2.0

        for p in preds:
            path = p["image_path"]
            if gen.lower() not in path.lower():
                continue
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
            demo_images.append((best, gen, label))
            print(f"[OK] {gen} ({label}): conf={best['confidence']:.3f}")

    print(f"\nCreating {len(demo_images)} demo images (v2 — overlay + radar)...")

    for i, (pred, gen, label) in enumerate(demo_images):
        heatmap_fname = os.path.basename(pred.get("heatmap_path", ""))
        orig_path = os.path.join(ORIGINAL_DIR, heatmap_fname)
        hm_path = os.path.join(HEATMAP_DIR, heatmap_fname)

        if not os.path.exists(orig_path) or not os.path.exists(hm_path):
            print(f"  [SKIP] Missing files for {gen}")
            continue

        output_path = os.path.join(OUTPUT_DIR, f"demo_{i+1:02d}_{gen}_{label}.png")
        create_demo_v2(orig_path, hm_path, pred, output_path, gen)
        print(f"  [{i+1}] {gen} ({label}) -> {output_path}")

    print(f"\nDone! Images at: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
