"""
XGenDet: Gemini-based Annotation Pipeline for Explainability.

Uses Gemini 2.0 Flash (free tier: 1500 req/day) to generate:
  - 6 attribute scores (texture, edges, color, geometry, semantics, frequency)
  - 2-4 sentence forensic natural language explanation

Usage:
  python scripts/gemini_annotate.py --input_dir /path/to/images --output annotations.jsonl
  python scripts/gemini_annotate.py --input_dir /path/to/images --heatmap_dir /path/to/heatmaps --output annotations.jsonl
  python scripts/gemini_annotate.py --resume annotations.jsonl  # Resume from partial run
"""

import os
import sys
import json
import time
import base64
import argparse
import glob
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load API key
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_URL = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}'

# Rate limiting: Gemini free tier = 15 RPM, 1500 RPD
REQUESTS_PER_MINUTE = 14  # Stay slightly under limit
REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE  # ~4.3 seconds

FORENSIC_PROMPT_WITH_HEATMAP = """You are an expert forensic image analyst specializing in AI-generated image detection.

You are given two images:
1. IMAGE 1: The original image to analyze
2. IMAGE 2: An artifact heatmap overlay (red/warm = suspicious regions, blue/cool = normal regions)

Analyze the image for signs of AI generation. For each of the following 6 attributes, provide a score from 0.0 (no artifacts detected) to 1.0 (severe, obvious artifacts):

1. texture_consistency: Surface texture naturalness (smoothness, repetition, plastic-like appearance)
2. edge_quality: Boundary sharpness and coherence (blurring, ringing, ghosting at edges)
3. color_distribution: Color naturalness (banding, oversaturation, impossible color combinations)
4. geometric_coherence: Structural and perspective consistency (warped geometry, impossible angles)
5. semantic_plausibility: Content-level plausibility (impossible objects, physics violations)
6. frequency_artifacts: Spectral domain artifacts (aliasing, periodic noise, grid patterns)

Then provide a 2-4 sentence forensic explanation that:
(a) References specific regions highlighted in the heatmap
(b) Describes the types of artifacts observed
(c) Provides a confidence assessment

You MUST respond with ONLY valid JSON in this exact format:
{
  "prediction": "REAL" or "FAKE",
  "confidence": 0.0 to 1.0,
  "attributes": {
    "texture_consistency": 0.0,
    "edge_quality": 0.0,
    "color_distribution": 0.0,
    "geometric_coherence": 0.0,
    "semantic_plausibility": 0.0,
    "frequency_artifacts": 0.0
  },
  "explanation": "Your 2-4 sentence forensic explanation here."
}"""

FORENSIC_PROMPT_NO_HEATMAP = """You are an expert forensic image analyst specializing in AI-generated image detection.

Analyze this image for signs of AI generation. For each of the following 6 attributes, provide a score from 0.0 (no artifacts detected) to 1.0 (severe, obvious artifacts):

1. texture_consistency: Surface texture naturalness (smoothness, repetition, plastic-like appearance)
2. edge_quality: Boundary sharpness and coherence (blurring, ringing, ghosting at edges)
3. color_distribution: Color naturalness (banding, oversaturation, impossible color combinations)
4. geometric_coherence: Structural and perspective consistency (warped geometry, impossible angles)
5. semantic_plausibility: Content-level plausibility (impossible objects, physics violations)
6. frequency_artifacts: Spectral domain artifacts (aliasing, periodic noise, grid patterns)

Then provide a 2-4 sentence forensic explanation that:
(a) Identifies the most suspicious regions in the image
(b) Describes the types of artifacts observed
(c) Provides a confidence assessment

You MUST respond with ONLY valid JSON in this exact format:
{
  "prediction": "REAL" or "FAKE",
  "confidence": 0.0 to 1.0,
  "attributes": {
    "texture_consistency": 0.0,
    "edge_quality": 0.0,
    "color_distribution": 0.0,
    "geometric_coherence": 0.0,
    "semantic_plausibility": 0.0,
    "frequency_artifacts": 0.0
  },
  "explanation": "Your 2-4 sentence forensic explanation here."
}"""


def encode_image(path):
    """Read and base64-encode an image file."""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_mime_type(path):
    """Get MIME type from file extension."""
    ext = Path(path).suffix.lower()
    return {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.gif': 'image/gif',
    }.get(ext, 'image/jpeg')


def call_gemini(image_path, heatmap_path=None, max_retries=3):
    """Call Gemini API with image (and optional heatmap) for forensic analysis."""
    parts = []

    # Add original image
    img_b64 = encode_image(image_path)
    parts.append({
        'inline_data': {
            'mime_type': get_mime_type(image_path),
            'data': img_b64
        }
    })

    # Add heatmap if available
    if heatmap_path and os.path.exists(heatmap_path):
        hmap_b64 = encode_image(heatmap_path)
        parts.append({
            'inline_data': {
                'mime_type': get_mime_type(heatmap_path),
                'data': hmap_b64
            }
        })
        parts.append({'text': FORENSIC_PROMPT_WITH_HEATMAP})
    else:
        parts.append({'text': FORENSIC_PROMPT_NO_HEATMAP})

    payload = {
        'contents': [{'parts': parts}],
        'generationConfig': {
            'temperature': 0.1,  # Low temperature for consistent annotations
            'maxOutputTokens': 512,
        }
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(GEMINI_URL, json=payload, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                text = data['candidates'][0]['content']['parts'][0]['text']
                # Clean up markdown code fences if present
                text = text.strip()
                if text.startswith('```json'):
                    text = text[7:]
                if text.startswith('```'):
                    text = text[3:]
                if text.endswith('```'):
                    text = text[:-3]
                text = text.strip()

                result = json.loads(text)
                return result

            elif resp.status_code == 429:
                # Rate limited — wait and retry
                wait = 60 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            elif resp.status_code == 503:
                # Service unavailable — brief retry
                time.sleep(10 * (attempt + 1))
                continue

            else:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                return None

        except json.JSONDecodeError as e:
            print(f"  JSON parse error on attempt {attempt+1}: {e}")
            print(f"  Raw text: {text[:200]}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None

        except requests.exceptions.Timeout:
            print(f"  Timeout on attempt {attempt+1}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None

        except Exception as e:
            print(f"  Error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None

    return None


def collect_images(input_dir, extensions=('.png', '.jpg', '.jpeg', '.webp', '.JPEG', '.JPG', '.PNG')):
    """Collect all image files from a directory (recursively)."""
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(input_dir, '**', f'*{ext}'), recursive=True))
    return sorted(images)


def load_existing_annotations(output_path):
    """Load already-annotated image paths for resume support."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    done.add(entry.get('image_path', ''))
                except json.JSONDecodeError:
                    continue
    return done


def main():
    parser = argparse.ArgumentParser(description='XGenDet Gemini Annotation Pipeline')
    parser.add_argument('--input_dir', type=str, help='Directory containing images to annotate')
    parser.add_argument('--heatmap_dir', type=str, default=None,
                        help='Directory containing heatmap images (matched by filename)')
    parser.add_argument('--output', type=str, default='annotations/gemini_annotations.jsonl',
                        help='Output JSONL file path')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to annotate')
    parser.add_argument('--label', type=str, choices=['real', 'fake', 'unknown'], default='unknown',
                        help='Ground truth label for all images in this batch')
    parser.add_argument('--generator', type=str, default='unknown',
                        help='Generator name (e.g., StyleGAN, Midjourney)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file')
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found. Set it in xgendet/.env")
        sys.exit(1)

    # Collect images
    images = collect_images(args.input_dir)
    print(f"Found {len(images)} images in {args.input_dir}")

    if args.max_images:
        images = images[:args.max_images]
        print(f"Limited to {args.max_images} images")

    # Resume support
    done = set()
    if args.resume:
        done = load_existing_annotations(args.output)
        print(f"Resuming: {len(done)} already annotated, {len(images) - len(done)} remaining")

    # Create output directory
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    # Annotate
    success = 0
    failed = 0
    skipped = 0

    with open(args.output, 'a') as fout:
        for i, img_path in enumerate(images):
            if img_path in done:
                skipped += 1
                continue

            # Find matching heatmap
            heatmap_path = None
            if args.heatmap_dir:
                stem = Path(img_path).stem
                for ext in ['.png', '.jpg']:
                    candidate = os.path.join(args.heatmap_dir, stem + ext)
                    if os.path.exists(candidate):
                        heatmap_path = candidate
                        break

            print(f"[{i+1}/{len(images)}] Annotating: {os.path.basename(img_path)}", end='')
            if heatmap_path:
                print(f" (+ heatmap)", end='')

            result = call_gemini(img_path, heatmap_path)

            if result:
                entry = {
                    'image_path': img_path,
                    'heatmap_path': heatmap_path,
                    'ground_truth': args.label,
                    'generator': args.generator,
                    'gemini_prediction': result.get('prediction', 'UNKNOWN'),
                    'gemini_confidence': result.get('confidence', 0.0),
                    'attributes': result.get('attributes', {}),
                    'explanation': result.get('explanation', ''),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                fout.write(json.dumps(entry) + '\n')
                fout.flush()
                success += 1
                print(f" -> {result.get('prediction', '?')} ({result.get('confidence', 0):.2f})")
            else:
                failed += 1
                print(f" -> FAILED")

            # Rate limiting
            time.sleep(REQUEST_INTERVAL)

    print(f"\nDone! Success: {success}, Failed: {failed}, Skipped: {skipped}")
    print(f"Annotations saved to: {args.output}")


if __name__ == '__main__':
    main()
