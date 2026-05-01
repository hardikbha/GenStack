"""
Step 2: Build ForenSight SFT training data.
Combines HydraFake SFT reasoning traces with XGenDet evidence in the prompt.
Output: JSON in Qwen3-VL chat format for LoRA fine-tuning.
"""

import os, json, sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_evidence_text(feat):
    """Build structured evidence text from XGenDet features."""
    conf = feat["confidence"]
    family = feat["family"]
    attrs = feat["attr_scores"]

    # Sort attributes by score (descending)
    sorted_attrs = sorted(attrs.items(), key=lambda x: -x[1])
    top_attrs = ", ".join(f"{k}({v:.2f})" for k, v in sorted_attrs[:3])

    verdict = "likely fake" if conf > 0.5 else "likely real"
    conf_pct = int(conf * 100)

    text = (
        f"A forensic detection system analyzed this face image and found:\n"
        f"- Detection confidence: {conf_pct}% ({verdict})\n"
        f"- Predicted generator type: {family}\n"
        f"- Artifact attribute scores (0-1 scale):\n"
    )
    for name, score in sorted_attrs:
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        text += f"  {name:12s}: {score:.2f} [{bar}]\n"
    text += f"- Strongest signals: {top_attrs}\n"

    return text


SYSTEM_PROMPT = """You are a forensic image analyst specializing in deepfake detection. A detection system has already analyzed this image and provided structured evidence (attribute scores, confidence, generator type).

Your task:
1. Consider BOTH the visual content AND the detector's evidence
2. Use structured reasoning with XML tags
3. Reference specific visual observations that support or contradict the detector's findings
4. Give your final answer as <answer> real </answer> or <answer> fake </answer>

Reasoning format:
<fast> Quick initial assessment based on detector evidence and first impression </fast>
<reasoning> Detailed analysis referencing specific visual features and detector scores </reasoning>
<conclusion> Synthesize all evidence into a final judgment </conclusion>
<answer> real/fake </answer>"""


def main():
    # Load XGenDet cached features
    cache_path = "checkpoints/xgendet_cached_features.json"
    if not os.path.exists(cache_path):
        print(f"ERROR: Run cache_xgendet_features.py first! Missing: {cache_path}")
        sys.exit(1)

    with open(cache_path) as f:
        xgendet_cache = json.load(f)
    print(f"Loaded {len(xgendet_cache)} cached XGenDet features")

    # Load original SFT data (has reasoning traces)
    with open("/home/sachin.chaudhary/hydrafake/jsons/train/sft_36k.json") as f:
        sft_data = json.load(f)
    print(f"Loaded {len(sft_data)} SFT samples")

    # Build ForenSight training data
    forensight_data = []
    skipped = 0

    for item in tqdm(sft_data, desc="Building ForenSight SFT data"):
        rel_path = item["images"][0]

        # Get XGenDet evidence
        if rel_path not in xgendet_cache:
            skipped += 1
            continue

        feat = xgendet_cache[rel_path]
        evidence_text = build_evidence_text(feat)

        # Get original reasoning trace
        assistant_msg = None
        for msg in item["messages"]:
            if msg["role"] == "assistant":
                assistant_msg = msg["content"]
                break

        if not assistant_msg:
            skipped += 1
            continue

        # Build full path for image
        full_path = os.path.join("/home/sachin.chaudhary", rel_path)

        # Build Qwen3-VL format
        forensight_item = {
            "image": full_path,
            "label": item["label"],
            "type": item.get("type", "unknown"),
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"<image>\n\nDetector Evidence:\n{evidence_text}\n\nBased on the image and the detector's analysis, determine if this face is real or fake.",
                },
                {
                    "role": "assistant",
                    "content": assistant_msg,
                },
            ],
            "xgendet_evidence": feat,
        }
        forensight_data.append(forensight_item)

    # Save
    out_path = "checkpoints/forensight_sft_data.json"
    with open(out_path, "w") as f:
        json.dump(forensight_data, f, indent=2)

    print(f"\nBuilt {len(forensight_data)} ForenSight SFT samples (skipped {skipped})")
    print(f"Saved to: {out_path}")

    # Also create a smaller val set
    val_data = forensight_data[:2000]  # First 2000 as val
    train_data = forensight_data[2000:]  # Rest as train
    with open("checkpoints/forensight_sft_train.json", "w") as f:
        json.dump(train_data, f)
    with open("checkpoints/forensight_sft_val.json", "w") as f:
        json.dump(val_data, f)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")


if __name__ == "__main__":
    main()
