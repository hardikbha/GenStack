"""
Demo v3: Detailed forensic visualization using Gemini detailed analysis.
Shows: Original | Heatmap Overlay | Detailed Forensic Report
"""
import json
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "/home/sachin.chaudhary/xgendet/demo_outputs_v3"
HEATMAP_DIR = "/home/sachin.chaudhary/xgendet/eval_outputs/heatmaps"
ORIGINAL_DIR = "/home/sachin.chaudhary/xgendet/eval_outputs/originals"
DETAILED_DIR = "/home/sachin.chaudhary/xgendet/demo_outputs_detailed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_font(size, bold=False):
    paths = [
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except:
            continue
    return ImageFont.load_default()


def draw_radar(draw, cx, cy, radius, scores, labels):
    n = len(scores)
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]

    for r_frac in [0.25, 0.5, 0.75, 1.0]:
        r = radius * r_frac
        pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
        pts.append(pts[0])
        draw.line(pts, fill=(210, 210, 210), width=1)

    for a in angles:
        draw.line([(cx, cy), (cx + radius * math.cos(a), cy + radius * math.sin(a))],
                  fill=(210, 210, 210), width=1)

    poly = [(cx + radius * s * math.cos(a), cy + radius * s * math.sin(a))
            for s, a in zip(scores, angles)]
    draw.polygon(poly, fill=(255, 80, 80, 50), outline=(220, 50, 50))

    for s, a in zip(scores, angles):
        x = cx + radius * s * math.cos(a)
        y = cy + radius * s * math.sin(a)
        draw.ellipse([x-2, y-2, x+2, y+2], fill=(200, 40, 40))

    font = get_font(8)
    for label, a in zip(labels, angles):
        x = cx + (radius + 12) * math.cos(a)
        y = cy + (radius + 12) * math.sin(a)
        try:
            tw = draw.textlength(label, font=font)
        except:
            tw = len(label) * 5
        draw.text((x - tw/2, y - 5), label, fill=(80, 80, 80), font=font)


def wrap_text(text, max_chars=50):
    words = text.split()
    lines, line = [], ""
    for w in words:
        if len(line) + len(w) + 1 > max_chars:
            lines.append(line.strip())
            line = w + " "
        else:
            line += w + " "
    if line.strip():
        lines.append(line.strip())
    return lines


def create_demo_v3(orig_path, hm_path, analysis, output_path):
    orig = Image.open(orig_path).convert("RGB")
    hm_full = Image.open(hm_path).convert("RGB")
    ow, oh = orig.size

    # Extract overlay (3rd panel)
    pw = hm_full.size[0] // 3
    overlay = hm_full.crop((2*pw, 0, 3*pw, hm_full.size[1])).resize((ow, oh), Image.LANCZOS)

    gemini = analysis["gemini_detailed"]
    is_fake = gemini.get("verdict", "FAKE") == "FAKE"

    # Layout: wider info panel for detailed text
    info_w = 380
    gap = 6
    canvas_w = ow + gap + ow + gap + info_w
    canvas_h = max(oh + 28, 380)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Title bar
    gen = analysis["generator"]
    conf = gemini.get("confidence", 0)
    verdict = gemini.get("verdict", "?")
    gen_guess = gemini.get("likely_generator", "Unknown")
    title_color = (200, 30, 30) if is_fake else (30, 140, 30)

    font_title = get_font(12, bold=True)
    font_section = get_font(10, bold=True)
    font_body = get_font(9)
    font_small = get_font(8)

    title = f"{gen.upper()} — {verdict} ({conf*100:.0f}%) — Likely: {gen_guess}"
    draw.text((4, 2), title, fill=title_color, font=font_title)

    y_img = 20

    # Labels
    draw.text((ow//2 - 20, y_img - 2), "Original", fill=(130,130,130), font=font_small)
    draw.text((ow + gap + ow//2 - 30, y_img - 2), "Attention Map", fill=(130,130,130), font=font_small)

    y_img += 10
    canvas.paste(orig, (0, y_img))
    canvas.paste(overlay, (ow + gap, y_img))

    # Draw borders
    draw.rectangle([0, y_img, ow-1, y_img+oh-1], outline=(200,200,200))
    draw.rectangle([ow+gap, y_img, ow+gap+ow-1, y_img+oh-1], outline=(200,200,200))

    # Info panel
    x = ow * 2 + gap * 2 + 4
    y = y_img - 4

    # Radar chart (compact)
    attrs = gemini.get("attributes", {})
    attr_keys = ["texture_consistency", "edge_quality", "color_distribution",
                 "geometric_coherence", "semantic_plausibility", "frequency_artifacts"]
    attr_labels = ["Texture", "Edges", "Color", "Geom.", "Semantic", "Freq."]
    scores = []
    for k in attr_keys:
        v = attrs.get(k, {})
        if isinstance(v, dict):
            scores.append(v.get("score", 0.5))
        else:
            scores.append(float(v) if v else 0.5)

    draw_radar(draw, x + 55, y + 48, 36, scores, attr_labels)
    y += 100

    # Primary artifacts
    draw.line([(x-2, y), (x + info_w - 10, y)], fill=(200,200,200))
    y += 4
    draw.text((x, y), "Primary Artifacts:", fill=(40,40,40), font=font_section)
    y += 14

    artifacts = gemini.get("primary_artifacts", [])
    for art in artifacts[:4]:
        atype = art.get("artifact_type", "?")
        loc = art.get("location", "?")
        sev = art.get("severity", "?")
        cause = art.get("technical_cause", "")

        sev_color = {"severe": (200,30,30), "moderate": (200,130,30), "mild": (100,160,50)}.get(sev, (100,100,100))

        draw.text((x, y), f"• {atype}", fill=(50,50,50), font=font_body)
        y += 12
        draw.text((x + 10, y), f"@ {loc}", fill=(100,100,100), font=font_small)
        # Severity badge
        draw.text((x + 10 + len(loc)*5 + 10, y), f"[{sev}]", fill=sev_color, font=font_small)
        y += 11

        # Technical cause (1 line)
        if cause:
            cause_lines = wrap_text(cause, max_chars=52)
            for cl in cause_lines[:2]:
                draw.text((x + 10, y), cl, fill=(120,120,120), font=font_small)
                y += 10
        y += 3

    # Key evidence
    draw.line([(x-2, y), (x + info_w - 10, y)], fill=(200,200,200))
    y += 4
    draw.text((x, y), "Key Evidence:", fill=(40,40,40), font=font_section)
    y += 14

    evidence = gemini.get("key_evidence", [])
    for ev in evidence[:4]:
        ev_lines = wrap_text(f"→ {ev}", max_chars=52)
        for el in ev_lines[:2]:
            if y < canvas_h - 12:
                draw.text((x, y), el, fill=(70,70,70), font=font_small)
                y += 10
        y += 2

    # Forensic summary at bottom
    if y < canvas_h - 40:
        draw.line([(x-2, y), (x + info_w - 10, y)], fill=(200,200,200))
        y += 4
        draw.text((x, y), "Forensic Summary:", fill=(40,40,40), font=font_section)
        y += 14
        summary = gemini.get("forensic_summary", "")
        for sl in wrap_text(summary, max_chars=52):
            if y < canvas_h - 10:
                draw.text((x, y), sl, fill=(80,80,80), font=font_small)
                y += 10

    # Save
    canvas_rgb = Image.new("RGB", canvas.size, (255,255,255))
    canvas_rgb.paste(canvas, mask=canvas.split()[3])
    canvas_rgb.save(output_path, quality=95)


def main():
    # Load all detailed analyses
    all_path = os.path.join(DETAILED_DIR, "all_detailed_analyses.json")
    with open(all_path) as f:
        analyses = json.load(f)

    print(f"Creating v3 demo images for {len(analyses)} analyses...")

    for analysis in analyses:
        fname = analysis["filename"]
        gen = analysis["generator"]
        label = analysis["label"]
        idx = analysis["index"]

        orig_path = os.path.join(ORIGINAL_DIR, fname)
        hm_path = os.path.join(HEATMAP_DIR, fname)

        if not os.path.exists(orig_path) or not os.path.exists(hm_path):
            print(f"  [{idx}] SKIP: files missing for {gen}")
            continue

        output_path = os.path.join(OUTPUT_DIR, f"demo_{idx:02d}_{gen}_{label}.png")
        create_demo_v3(orig_path, hm_path, analysis, output_path)
        print(f"  [{idx}] {gen} ({label}) -> {output_path}")

    print(f"\nDone! Images at: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
