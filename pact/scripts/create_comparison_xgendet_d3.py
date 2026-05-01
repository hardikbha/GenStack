#!/usr/bin/env python3
"""
Generate side-by-side comparison images: XGenDet heatmaps vs D3 GradCAM heatmaps.

Layout per image:
  [Title Bar: Generator Name | Ground Truth Label]
  [Original 224x224 | XGenDet Overlay 224x224 | D3 Overlay 224x224 | Verdict Panel]

Inputs:
  - XGenDet 3-panel heatmaps: eval_outputs/heatmaps/  (crop 3rd panel = overlay)
  - D3 3-panel rollout images: demo_outputs_d3/demo_XX_gen_label_rollout.png (crop 3rd panel = overlay)
  - Originals: eval_outputs/originals/
  - Analysis JSON: demo_outputs_detailed/all_detailed_analyses.json

Output: demo_outputs_comparison/
"""

import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# Paths
# ==============================================================================
BASE = "/home/sachin.chaudhary/xgendet"
HEATMAP_DIR = os.path.join(BASE, "eval_outputs", "heatmaps")
ORIGINAL_DIR = os.path.join(BASE, "eval_outputs", "originals")
D3_DIR = os.path.join(BASE, "demo_outputs_d3")
JSON_PATH = os.path.join(BASE, "demo_outputs_detailed", "all_detailed_analyses.json")
OUTPUT_DIR = os.path.join(BASE, "demo_outputs_comparison")

# Font paths
FONT_DIR = "/usr/share/fonts/dejavu-sans-fonts"
FONT_REGULAR = os.path.join(FONT_DIR, "DejaVuSans.ttf")
FONT_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")

# Layout constants
PANEL_SIZE = 224
PADDING = 8
TITLE_HEIGHT = 40
LABEL_HEIGHT = 24
VERDICT_WIDTH = 200
SEPARATOR_WIDTH = 3
BG_COLOR = (30, 30, 30)
TITLE_BG = (20, 20, 45)
LABEL_BG = (45, 45, 55)
SEPARATOR_COLOR = (60, 60, 70)
WHITE = (255, 255, 255)
LIGHT_GRAY = (200, 200, 200)
GREEN = (80, 220, 100)
RED = (240, 80, 80)
YELLOW = (240, 220, 80)
BLUE_ACCENT = (100, 150, 255)


def load_fonts():
    """Load DejaVu fonts at various sizes."""
    fonts = {}
    try:
        fonts["title"] = ImageFont.truetype(FONT_BOLD, 18)
        fonts["label"] = ImageFont.truetype(FONT_REGULAR, 13)
        fonts["verdict_header"] = ImageFont.truetype(FONT_BOLD, 14)
        fonts["verdict_text"] = ImageFont.truetype(FONT_REGULAR, 12)
        fonts["verdict_small"] = ImageFont.truetype(FONT_REGULAR, 11)
        fonts["tag"] = ImageFont.truetype(FONT_BOLD, 11)
    except Exception as e:
        print(f"Warning: Could not load DejaVu fonts: {e}")
        print("Falling back to default font.")
        default = ImageFont.load_default()
        for key in ["title", "label", "verdict_header", "verdict_text", "verdict_small", "tag"]:
            fonts[key] = default
    return fonts


def crop_xgendet_overlay(heatmap_path):
    """
    Crop the 3rd panel (overlay) from XGenDet 3-panel heatmap.

    XGenDet heatmap layout (~1489x501 RGBA):
      Border (10px) | Panel1 (459px) | Sep | Panel2 (420px) | Sep | Colorbar (~28px) | Gap | Panel3 (459px) | Border
      Panel 3 (overlay): x=1021:1480, y=32:491 -> 459x459
    """
    img = Image.open(heatmap_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Find panel 3 boundaries dynamically
    # Scan from right to find rightmost content boundary
    right_border = w
    for x in range(w - 1, w - 20, -1):
        col = arr[h // 2, x, :]
        if col.mean() < 240:
            right_border = x + 1
            break

    # Scan leftward from right_border to find the start of panel 3
    panel3_end = right_border
    panel3_start = None
    in_white = False
    for x in range(panel3_end - 1, 0, -1):
        col_slice = arr[h // 4 : 3 * h // 4, x, :]
        is_white = col_slice.mean() > 248 and col_slice.std() < 10
        if is_white and not in_white:
            in_white = True
        if not is_white and in_white:
            panel3_start = x + 1
            # Step over the whitespace to find actual content start
            # Look for the first content column after whitespace
            break

    if panel3_start is None:
        # Fallback: use known offsets
        panel3_start = 1021
        panel3_end = 1480

    # Find vertical content bounds
    top_y = 0
    for y in range(h):
        row = arr[y, panel3_start : panel3_start + 50, :]
        if row.mean() < 240:
            top_y = y
            break

    bottom_y = h
    for y in range(h - 1, 0, -1):
        row = arr[y, panel3_start : panel3_start + 50, :]
        if row.mean() < 240:
            bottom_y = y + 1
            break

    overlay = img.crop((panel3_start, top_y, panel3_end, bottom_y))
    overlay = overlay.resize((PANEL_SIZE, PANEL_SIZE), Image.LANCZOS)
    return overlay


def crop_d3_overlay(rollout_path):
    """
    Crop the 3rd panel (overlay) from D3 rollout 3-panel image.

    D3 rollout layout (682x254):
      Rows 0-29: title bars and separators
      Rows 30-253: image content (224 rows)
      Cols 0-223: original (224px)
      Cols 224-228: separator (5px white)
      Cols 229-452: raw heatmap (224px)
      Cols 453-457: separator (5px white)
      Cols 458-681: overlay (224px)
    """
    img = Image.open(rollout_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Find the top of image content by looking for the last white separator
    # row before actual image pixels. The layout has title rows + subtitle rows
    # + a white separator row, then image content starts.
    # Scan from bottom up to find where content ends, then from there find start.
    content_top = 30  # safe default: after title + subtitle + separator
    # Find the last fully-white row before the image content region
    last_white = 0
    for y in range(min(50, h)):
        row = arr[y, :, :]
        if row.mean() > 252 and row.std() < 5:
            last_white = y
    content_top = last_white + 1

    # Find column separators dynamically in the image content area
    sep_groups = []
    in_sep = False
    sep_start = 0
    for x in range(w):
        col = arr[content_top : content_top + 100, x, :]
        is_white = col.mean() > 248 and col.std() < 5
        if is_white and not in_sep:
            in_sep = True
            sep_start = x
        elif not is_white and in_sep:
            in_sep = False
            sep_groups.append((sep_start, x - 1))

    # Panel 3 starts after the second separator group
    if len(sep_groups) >= 2:
        panel3_start = sep_groups[1][1] + 1
    else:
        panel3_start = 458  # fallback

    content_bottom = min(content_top + PANEL_SIZE, h)
    panel3_end = min(panel3_start + PANEL_SIZE, w)

    overlay = img.crop((panel3_start, content_top, panel3_end, content_bottom))
    if overlay.size != (PANEL_SIZE, PANEL_SIZE):
        overlay = overlay.resize((PANEL_SIZE, PANEL_SIZE), Image.LANCZOS)
    return overlay


def draw_rounded_rect(draw, xy, fill, radius=6):
    """Draw a rounded rectangle."""
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def draw_verdict_panel(draw, x_start, y_start, panel_height, entry, fonts):
    """
    Draw the verdict panel showing model predictions and agreement.
    """
    gt_label = entry["label"]
    xg_pred = entry["xgendet_prediction"]
    xg_conf = entry["xgendet_confidence"]
    d3_verdict = entry["gemini_detailed"]["verdict"].lower()
    d3_conf = entry["gemini_detailed"]["confidence"]

    xg_correct = xg_pred == gt_label
    d3_correct = d3_verdict == gt_label
    models_agree = xg_pred == d3_verdict

    x = x_start + 10
    y = y_start + 8
    vw = VERDICT_WIDTH - 20

    # --- XGenDet Section ---
    draw.text((x, y), "XGenDet", fill=BLUE_ACCENT, font=fonts["verdict_header"])
    y += 20
    xg_color = GREEN if xg_correct else RED
    xg_label = xg_pred.upper()
    draw.text((x + 4, y), xg_label, fill=xg_color, font=fonts["verdict_text"])
    conf_text = f"({xg_conf:.1%})"
    draw.text((x + 52, y), conf_text, fill=LIGHT_GRAY, font=fonts["verdict_small"])
    y += 16
    # Correctness tag
    tag = "CORRECT" if xg_correct else "WRONG"
    tag_color = GREEN if xg_correct else RED
    draw.text((x + 4, y), tag, fill=tag_color, font=fonts["tag"])
    y += 22

    # Separator line
    draw.line([(x, y), (x + vw, y)], fill=SEPARATOR_COLOR, width=1)
    y += 8

    # --- D3 Section ---
    draw.text((x, y), "D3 (GradCAM)", fill=BLUE_ACCENT, font=fonts["verdict_header"])
    y += 20
    d3_color = GREEN if d3_correct else RED
    d3_label = d3_verdict.upper()
    draw.text((x + 4, y), d3_label, fill=d3_color, font=fonts["verdict_text"])
    conf_text = f"({d3_conf:.0%})"
    draw.text((x + 52, y), conf_text, fill=LIGHT_GRAY, font=fonts["verdict_small"])
    y += 16
    tag = "CORRECT" if d3_correct else "WRONG"
    tag_color = GREEN if d3_correct else RED
    draw.text((x + 4, y), tag, fill=tag_color, font=fonts["tag"])
    y += 22

    # Separator line
    draw.line([(x, y), (x + vw, y)], fill=SEPARATOR_COLOR, width=1)
    y += 8

    # --- Agreement ---
    if models_agree:
        agree_text = "AGREE"
        agree_color = GREEN
    else:
        agree_text = "DISAGREE"
        agree_color = YELLOW

    draw.text((x, y), "Agreement:", fill=LIGHT_GRAY, font=fonts["verdict_text"])
    y += 16
    draw.text((x + 4, y), agree_text, fill=agree_color, font=fonts["verdict_header"])


def create_comparison(entry, fonts):
    """
    Create a single comparison image for one demo entry.

    Returns the PIL Image.
    """
    idx = entry["index"]
    generator = entry["generator"]
    label = entry["label"]
    filename = entry["filename"]

    # --- Load images ---
    # Original
    orig_path = os.path.join(ORIGINAL_DIR, filename)
    if not os.path.exists(orig_path):
        print(f"  WARNING: Original not found: {orig_path}")
        return None
    original = Image.open(orig_path).convert("RGB").resize(
        (PANEL_SIZE, PANEL_SIZE), Image.LANCZOS
    )

    # XGenDet overlay (3rd panel of heatmap)
    hm_path = os.path.join(HEATMAP_DIR, filename)
    if not os.path.exists(hm_path):
        print(f"  WARNING: XGenDet heatmap not found: {hm_path}")
        return None
    xgendet_overlay = crop_xgendet_overlay(hm_path)

    # D3 overlay (3rd panel of rollout image)
    d3_rollout_name = f"demo_{idx:02d}_{generator}_{label}_rollout.png"
    d3_rollout_path = os.path.join(D3_DIR, d3_rollout_name)
    if not os.path.exists(d3_rollout_path):
        print(f"  WARNING: D3 rollout not found: {d3_rollout_path}")
        return None
    d3_overlay = crop_d3_overlay(d3_rollout_path)

    # --- Compute canvas dimensions ---
    # 3 image panels + verdict panel + separators + padding
    num_panels = 3
    total_img_width = num_panels * PANEL_SIZE + (num_panels - 1) * SEPARATOR_WIDTH
    canvas_width = PADDING + total_img_width + SEPARATOR_WIDTH + VERDICT_WIDTH + PADDING
    canvas_height = TITLE_HEIGHT + LABEL_HEIGHT + PANEL_SIZE + PADDING

    # --- Build canvas ---
    canvas = Image.new("RGB", (canvas_width, canvas_height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # Title bar
    draw.rectangle(
        [(0, 0), (canvas_width, TITLE_HEIGHT)],
        fill=TITLE_BG,
    )
    gt_tag = "REAL" if label == "real" else "FAKE"
    gt_color = GREEN if label == "real" else RED
    title_text = f"Demo {idx:02d}  |  {generator.upper()}"
    draw.text((PADDING + 4, 10), title_text, fill=WHITE, font=fonts["title"])

    # Ground truth tag on the right of title
    gt_display = f"Ground Truth: {gt_tag}"
    bbox = draw.textbbox((0, 0), gt_display, font=fonts["verdict_header"])
    gt_text_w = bbox[2] - bbox[0]
    draw.text(
        (canvas_width - PADDING - gt_text_w - 4, 13),
        gt_display,
        fill=gt_color,
        font=fonts["verdict_header"],
    )

    # Column labels
    y_label = TITLE_HEIGHT
    draw.rectangle(
        [(0, y_label), (canvas_width, y_label + LABEL_HEIGHT)],
        fill=LABEL_BG,
    )
    col_labels = ["Original", "XGenDet Attention", "D3 GradCAM Attention"]
    for i, lbl in enumerate(col_labels):
        lx = PADDING + i * (PANEL_SIZE + SEPARATOR_WIDTH)
        bbox = draw.textbbox((0, 0), lbl, font=fonts["label"])
        lbl_w = bbox[2] - bbox[0]
        cx = lx + (PANEL_SIZE - lbl_w) // 2
        draw.text((cx, y_label + 5), lbl, fill=LIGHT_GRAY, font=fonts["label"])

    # Verdict column label
    vx = PADDING + num_panels * PANEL_SIZE + (num_panels - 1) * SEPARATOR_WIDTH + SEPARATOR_WIDTH
    verdict_label = "Verdict"
    bbox = draw.textbbox((0, 0), verdict_label, font=fonts["label"])
    vlw = bbox[2] - bbox[0]
    draw.text(
        (vx + (VERDICT_WIDTH - vlw) // 2, y_label + 5),
        verdict_label,
        fill=LIGHT_GRAY,
        font=fonts["label"],
    )

    # Image panels
    y_img = TITLE_HEIGHT + LABEL_HEIGHT
    panels = [original, xgendet_overlay, d3_overlay]
    for i, panel in enumerate(panels):
        px = PADDING + i * (PANEL_SIZE + SEPARATOR_WIDTH)
        canvas.paste(panel, (px, y_img))

        # Draw separator after each panel (except after the last before verdict)
        if i < num_panels - 1:
            sx = px + PANEL_SIZE
            draw.rectangle(
                [(sx, y_img), (sx + SEPARATOR_WIDTH, y_img + PANEL_SIZE)],
                fill=SEPARATOR_COLOR,
            )

    # Separator before verdict panel
    sep_x = PADDING + num_panels * PANEL_SIZE + (num_panels - 1) * SEPARATOR_WIDTH
    draw.rectangle(
        [(sep_x, y_img), (sep_x + SEPARATOR_WIDTH, y_img + PANEL_SIZE)],
        fill=SEPARATOR_COLOR,
    )

    # Verdict panel
    verdict_x = sep_x + SEPARATOR_WIDTH
    draw_verdict_panel(draw, verdict_x, y_img, PANEL_SIZE, entry, fonts)

    return canvas


def main():
    # Load analysis data
    print(f"Loading analysis data from {JSON_PATH}")
    with open(JSON_PATH) as f:
        analyses = json.load(f)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fonts = load_fonts()

    generated = []
    for entry in analyses:
        idx = entry["index"]
        generator = entry["generator"]
        label = entry["label"]
        print(f"Processing Demo {idx:02d}: {generator} ({label})")

        img = create_comparison(entry, fonts)
        if img is None:
            print(f"  SKIPPED (missing input files)")
            continue

        out_name = f"comparison_{idx:02d}_{generator}_{label}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        img.save(out_path, "PNG")
        print(f"  Saved: {out_path} ({img.size[0]}x{img.size[1]})")
        generated.append(out_name)

    # Also create a combined grid (2 columns x 5 rows)
    if generated:
        print("\nCreating combined grid...")
        single_imgs = []
        for name in generated:
            single_imgs.append(Image.open(os.path.join(OUTPUT_DIR, name)))

        if single_imgs:
            cols = 2
            rows = (len(single_imgs) + cols - 1) // cols
            iw, ih = single_imgs[0].size
            grid_gap = 4
            grid_w = cols * iw + (cols - 1) * grid_gap
            grid_h = rows * ih + (rows - 1) * grid_gap
            grid = Image.new("RGB", (grid_w, grid_h), (15, 15, 15))

            for i, simg in enumerate(single_imgs):
                r, c = divmod(i, cols)
                gx = c * (iw + grid_gap)
                gy = r * (ih + grid_gap)
                grid.paste(simg, (gx, gy))

            grid_path = os.path.join(OUTPUT_DIR, "comparison_grid_all.png")
            grid.save(grid_path, "PNG")
            print(f"  Saved grid: {grid_path} ({grid.size[0]}x{grid.size[1]})")

    print(f"\nDone! Generated {len(generated)} comparison images in {OUTPUT_DIR}")
    return generated


if __name__ == "__main__":
    generated = main()
    if not generated:
        print("ERROR: No images generated!", file=sys.stderr)
        sys.exit(1)
