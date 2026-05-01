"""
Reformat HydraFake SFT data for Qwen3-VL-8B fine-tuning.

Takes the existing sft_36k.json (Veritas/InternVL format) and converts to
Qwen3-VL chat format, injecting SRM forensic evidence as structured text.

Input:
  - sft_36k.json: Original SFT data with pattern-annotated reasoning
  - srm_evidence_text.json: Pre-computed SRM noise evidence per image
  - face_parts_map.json: Face part crop paths per image

Output:
  - qwen3_sft_data.jsonl: Qwen3-VL formatted training data

Usage:
    python training/prepare_qwen3_sft.py \
        --sft_json /home/sachin.chaudhary/hydrafake/jsons/train/sft_36k.json \
        --srm_evidence /home/sachin.chaudhary/hydrafake/srm_evidence/srm_evidence_text.json \
        --face_parts /home/sachin.chaudhary/hydrafake/face_parts/face_parts_map.json \
        --data_root /home/sachin.chaudhary \
        --face_parts_dir /home/sachin.chaudhary/hydrafake/face_parts \
        --output /home/sachin.chaudhary/hydrafake/jsons/train/qwen3_sft_data.jsonl
"""

import os, sys, json, argparse, re


SYSTEM_PROMPT = (
    "You are a deepfake forensics expert specializing in facial image authenticity analysis. "
    "You have access to the original image, close-up crops of key facial regions "
    "(eyes, nose, mouth), and forensic noise analysis from SRM (Steganalysis Rich Model) filters.\n\n"
    "Analyze all available evidence carefully. Think step by step in <think> tags, then provide "
    "your final verdict as either \"real\" or \"fake\" in <answer> tags."
)


def convert_reasoning_to_think(assistant_content):
    """
    Convert Veritas-style pattern tags to Qwen3-VL native <think> format.

    Input patterns: <fast>...</fast>, <planning>...</planning>,
                   <reasoning>...</reasoning>, <reflection>...</reflection>,
                   <conclusion>...</conclusion>
    Output: <think>unified reasoning</think><answer>real/fake</answer>
    """
    # Extract the answer
    answer_match = re.search(r'<answer>\s*(real|fake)\s*</answer>', assistant_content, re.IGNORECASE)
    answer = answer_match.group(1).lower() if answer_match else "fake"

    # Extract reasoning from all pattern tags
    reasoning_parts = []
    for tag in ["fast", "planning", "reasoning", "reflection", "conclusion"]:
        pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
        matches = re.findall(pattern, assistant_content, re.DOTALL)
        for match in matches:
            clean = match.strip()
            if clean:
                reasoning_parts.append(clean)

    # If no pattern tags found, use entire content minus answer
    if not reasoning_parts:
        text = re.sub(r'<answer>.*?</answer>', '', assistant_content, flags=re.DOTALL).strip()
        if text:
            reasoning_parts.append(text)

    reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else "Analyzing the image for signs of manipulation."

    return reasoning, answer


def build_user_message(image_path, srm_text, face_parts, data_root, face_parts_dir):
    """
    Build the multi-modal user message with image, face parts, and SRM evidence.

    Qwen3-VL format for multi-image:
    [
        {"type": "image", "image": "file:///path/to/image.png"},
        {"type": "text", "text": "..."}
    ]
    """
    content = []

    # Main image
    full_image_path = os.path.join(data_root, image_path)
    content.append({"type": "image", "image": f"file://{full_image_path}"})

    # Face part crops (if available)
    part_labels = []
    if face_parts:
        for part_name in ["left_eye", "right_eye", "nose", "mouth"]:
            if part_name in face_parts:
                crop_path = os.path.join(face_parts_dir, face_parts[part_name])
                if os.path.exists(crop_path):
                    content.append({"type": "image", "image": f"file://{crop_path}"})
                    part_labels.append(part_name.replace("_", " "))

    # Text instruction with SRM evidence
    text_parts = []
    text_parts.append("Analyze this facial image for authenticity.")

    if part_labels:
        text_parts.append(f"\nAdditional close-up crops provided: {', '.join(part_labels)}.")

    if srm_text:
        text_parts.append(f"\nForensic noise analysis (SRM):\n{srm_text}")

    text_parts.append("\nDetermine if this image is real or fake.")

    content.append({"type": "text", "text": "\n".join(text_parts)})

    return content


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_json", required=True)
    p.add_argument("--srm_evidence", default=None, help="Path to srm_evidence_text.json")
    p.add_argument("--face_parts", default=None, help="Path to face_parts_map.json")
    p.add_argument("--data_root", default="/home/sachin.chaudhary")
    p.add_argument("--face_parts_dir", default="/home/sachin.chaudhary/hydrafake/face_parts")
    p.add_argument("--output", required=True)
    p.add_argument("--skip_no_reasoning", action="store_true",
                   help="Skip samples without reasoning (short answers only)")
    args = p.parse_args()

    # Load data
    with open(args.sft_json) as f:
        sft_data = json.load(f)
    print(f"Loaded {len(sft_data)} SFT samples")

    # Load SRM evidence (optional)
    srm_evidence = {}
    if args.srm_evidence and os.path.exists(args.srm_evidence):
        with open(args.srm_evidence) as f:
            srm_evidence = json.load(f)
        print(f"Loaded SRM evidence for {len(srm_evidence)} images")

    # Load face parts (optional)
    face_parts_map = {}
    if args.face_parts and os.path.exists(args.face_parts):
        with open(args.face_parts) as f:
            face_parts_map = json.load(f)
        print(f"Loaded face parts for {len(face_parts_map)} images")

    # Convert
    output_data = []
    skipped = 0
    for sample in sft_data:
        image_path = sample["images"][0]
        label = sample["label"]
        forgery_type = sample.get("type", "unknown")

        # Get assistant response and convert reasoning
        assistant_content = sample["messages"][2]["content"]
        reasoning, answer = convert_reasoning_to_think(assistant_content)

        # Skip short answers if requested
        if args.skip_no_reasoning and len(reasoning) < 20:
            skipped += 1
            continue

        # Get SRM evidence
        srm_text = srm_evidence.get(image_path, "")

        # Get face parts
        face_parts = face_parts_map.get(image_path, {})

        # Build user message
        user_content = build_user_message(
            image_path, srm_text, face_parts,
            args.data_root, args.face_parts_dir
        )

        # Build Qwen3-VL format
        qwen3_sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": f"<think>\n{reasoning}\n</think>\n<answer> {answer} </answer>"
                },
            ],
            # Metadata for evaluation
            "metadata": {
                "image_path": image_path,
                "label": label,
                "forgery_type": forgery_type,
                "answer": answer,
            }
        }
        output_data.append(qwen3_sample)

    # Save as JSONL
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nConverted {len(output_data)} samples ({skipped} skipped)")
    print(f"Output: {args.output}")

    # Stats
    real_count = sum(1 for d in output_data if d["metadata"]["label"] == 0)
    fake_count = sum(1 for d in output_data if d["metadata"]["label"] == 1)
    with_srm = sum(1 for d in output_data
                   if any("SRM" in str(m.get("content", "")) for m in d["messages"]))
    with_parts = sum(1 for d in output_data
                     if any(isinstance(m.get("content"), list) and len(m["content"]) > 2
                            for m in d["messages"]))

    print(f"  Real: {real_count}, Fake: {fake_count}")
    print(f"  With SRM evidence: {with_srm}")
    print(f"  With face parts: {with_parts}")


if __name__ == "__main__":
    main()
