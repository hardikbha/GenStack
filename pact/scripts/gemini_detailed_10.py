"""
Detailed Gemini forensic analysis for 10 demo images.
Returns structured JSON with very specific artifact descriptions.
"""
import os
import json
import base64
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_URL = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}'

ORIGINAL_DIR = "/home/sachin.chaudhary/xgendet/eval_outputs/originals"
HEATMAP_DIR = "/home/sachin.chaudhary/xgendet/eval_outputs/heatmaps"
DEMO_V2_DIR = "/home/sachin.chaudhary/xgendet/demo_outputs_v2"
OUTPUT_DIR = "/home/sachin.chaudhary/xgendet/demo_outputs_detailed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Much more detailed forensic prompt
DETAILED_PROMPT = """You are a world-class forensic image analyst with 15 years of experience detecting AI-generated images. You specialize in identifying artifacts from GANs (ProGAN, StyleGAN, BigGAN, CycleGAN, StarGAN), diffusion models (DALL-E, Stable Diffusion, ADM, GLIDE, Midjourney), and deepfake systems.

You are given:
1. IMAGE 1: The original image to analyze
2. IMAGE 2: An artifact attention heatmap overlay where RED/WARM regions = areas the detection model finds most suspicious, BLUE/COOL = less suspicious

Perform an exhaustive forensic analysis. Be EXTREMELY specific — reference exact spatial locations (e.g., "upper-left quadrant", "around the subject's left eye", "boundary between foreground shirt and background"), name exact artifact types, and explain the underlying technical cause.

Return a JSON response with this EXACT structure:
{
  "verdict": "REAL" or "FAKE",
  "confidence": 0.0 to 1.0,
  "likely_generator_family": "GAN" or "Diffusion" or "Autoregressive" or "Unknown",
  "likely_generator": "best guess of specific generator (e.g., StyleGAN2, DALL-E, Midjourney)",

  "attributes": {
    "texture_consistency": {
      "score": 0.0 to 1.0,
      "finding": "One specific sentence describing what you observe about surface textures",
      "locations": ["list of specific image regions where texture artifacts appear"]
    },
    "edge_quality": {
      "score": 0.0 to 1.0,
      "finding": "One specific sentence about boundary/edge quality",
      "locations": ["specific regions"]
    },
    "color_distribution": {
      "score": 0.0 to 1.0,
      "finding": "One specific sentence about color artifacts",
      "locations": ["specific regions"]
    },
    "geometric_coherence": {
      "score": 0.0 to 1.0,
      "finding": "One specific sentence about geometric/structural issues",
      "locations": ["specific regions"]
    },
    "semantic_plausibility": {
      "score": 0.0 to 1.0,
      "finding": "One specific sentence about content-level plausibility",
      "locations": ["specific regions"]
    },
    "frequency_artifacts": {
      "score": 0.0 to 1.0,
      "finding": "One specific sentence about spectral/frequency-domain artifacts",
      "locations": ["specific regions"]
    }
  },

  "primary_artifacts": [
    {
      "artifact_type": "Name of the artifact (e.g., 'skin texture smoothing', 'checkerboard pattern', 'color banding', 'perspective inconsistency', 'boundary blending error')",
      "location": "Exact location in the image (e.g., 'around the nose bridge and under-eye area')",
      "severity": "mild/moderate/severe",
      "technical_cause": "Why this artifact occurs (e.g., 'GAN upsampling creates periodic patterns in skin regions', 'Diffusion denoising leaves characteristic smoothing in high-frequency areas')"
    }
  ],

  "heatmap_analysis": "2-3 sentences explaining what the attention heatmap reveals. Reference specific hot regions and explain why the detection model flags them.",

  "forensic_summary": "3-5 sentence comprehensive forensic report. Start with the verdict, then describe the strongest evidence, reference specific locations, and conclude with confidence level. This should read like an expert witness report.",

  "key_evidence": ["List 3-5 bullet points of the most compelling evidence, each referencing a specific region and artifact"]
}

Be precise. Do not use vague language like "some areas show anomalies". Instead say exactly WHERE, WHAT artifact, and WHY it's suspicious. If the image is REAL, explain what makes it convincing and note any regions that might superficially resemble artifacts but are natural."""


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_gemini(original_path, heatmap_path):
    """Call Gemini with both original and heatmap overlay."""
    orig_b64 = encode_image(original_path)
    hm_b64 = encode_image(heatmap_path)

    payload = {
        "contents": [{
            "parts": [
                {"text": DETAILED_PROMPT},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": orig_b64
                    }
                },
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": hm_b64
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json"
        }
    }

    response = requests.post(GEMINI_URL, json=payload, timeout=60)
    if response.status_code != 200:
        print(f"  API error {response.status_code}: {response.text[:200]}")
        return None

    data = response.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        # Parse JSON
        result = json.loads(text)
        return result
    except (KeyError, json.JSONDecodeError) as e:
        print(f"  Parse error: {e}")
        # Try to extract JSON from text
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            # Find JSON in text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except:
            pass
        return None


def main():
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set in .env")
        return

    # Get the 10 demo images
    demo_files = sorted([f for f in os.listdir(DEMO_V2_DIR) if f.endswith(".png")])
    print(f"Found {len(demo_files)} demo images")

    # Load predictions to get file paths
    preds = []
    with open("/home/sachin.chaudhary/xgendet/eval_outputs/xgendet_predictions_enhanced.jsonl") as f:
        for line in f:
            preds.append(json.loads(line))

    # Map heatmap basename -> prediction
    pred_map = {}
    for p in preds:
        hm = os.path.basename(p.get("heatmap_path", ""))
        if hm:
            pred_map[hm] = p

    # Match demo files to originals/heatmaps
    # Demo filenames: demo_01_deepfake_fake.png -> need to find the actual heatmap file
    # Read demo_summary from v1 to get mappings
    demo_summary_v1 = "/home/sachin.chaudhary/xgendet/demo_outputs/demo_summary.json"
    if os.path.exists(demo_summary_v1):
        with open(demo_summary_v1) as f:
            v1_summary = json.load(f)
    else:
        v1_summary = []

    # Also try to match by generator name from predictions
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

    results = []

    for i, (gen, label) in enumerate(selections):
        # Find best matching prediction
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

        if not best:
            print(f"[{i+1}] SKIP: No match for {gen} ({label})")
            continue

        hm_basename = os.path.basename(best.get("heatmap_path", ""))
        orig_path = os.path.join(ORIGINAL_DIR, hm_basename)
        hm_path = os.path.join(HEATMAP_DIR, hm_basename)

        if not os.path.exists(orig_path) or not os.path.exists(hm_path):
            print(f"[{i+1}] SKIP: Files missing for {gen}")
            continue

        print(f"[{i+1}/10] Analyzing {gen} ({label}) — {hm_basename}...")

        # Call Gemini
        gemini_result = call_gemini(orig_path, hm_path)

        if gemini_result:
            result_entry = {
                "index": i + 1,
                "generator": gen,
                "label": label,
                "filename": hm_basename,
                "xgendet_prediction": best["prediction"],
                "xgendet_confidence": best["confidence"],
                "xgendet_family": best.get("family_pred", "Unknown"),
                "xgendet_attributes": best.get("attributes", {}),
                "gemini_detailed": gemini_result
            }
            results.append(result_entry)

            # Save individual result
            individual_path = os.path.join(OUTPUT_DIR, f"analysis_{i+1:02d}_{gen}_{label}.json")
            with open(individual_path, "w") as f:
                json.dump(result_entry, f, indent=2)

            # Print summary
            summary = gemini_result.get("forensic_summary", "No summary")
            print(f"  Verdict: {gemini_result.get('verdict', 'N/A')} (conf={gemini_result.get('confidence', 0):.2f})")
            print(f"  Generator guess: {gemini_result.get('likely_generator', 'N/A')}")
            artifacts = gemini_result.get("primary_artifacts", [])
            print(f"  Primary artifacts: {len(artifacts)}")
            for art in artifacts[:3]:
                print(f"    - {art.get('artifact_type', 'N/A')} @ {art.get('location', 'N/A')} ({art.get('severity', 'N/A')})")
            print(f"  Summary: {summary[:120]}...")
        else:
            print(f"  FAILED: No result from Gemini")

        # Rate limit: ~4 seconds between requests
        time.sleep(4.5)

    # Save all results
    all_path = os.path.join(OUTPUT_DIR, "all_detailed_analyses.json")
    with open(all_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! {len(results)}/10 images analyzed")
    print(f"Individual results: {OUTPUT_DIR}/analysis_*.json")
    print(f"Combined results: {all_path}")


if __name__ == "__main__":
    main()
