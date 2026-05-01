"""
Convert ForenSight SFT data to the qwen-vl-finetune JSONL format.
Matches the format used in ostf-train-cot.jsonl.
"""

import os, json, sys
from tqdm import tqdm


def main():
    # Load cached XGenDet features
    cache = json.load(open("checkpoints/xgendet_cached_features.json"))
    print(f"XGenDet cache: {len(cache)} images")

    # Load SFT data with reasoning traces
    sft = json.load(open("/home/sachin.chaudhary/hydrafake/jsons/train/sft_36k.json"))
    print(f"SFT data: {len(sft)} samples")

    attr_names = ["texture", "edges", "color", "geometry", "semantics", "frequency"]

    output_lines = []
    skipped = 0

    for item in tqdm(sft, desc="Building JSONL"):
        rel_path = item["images"][0]
        full_path = os.path.join("/home/sachin.chaudhary", rel_path)

        if not os.path.exists(full_path) or rel_path not in cache:
            skipped += 1
            continue

        feat = cache[rel_path]
        conf = feat["confidence"]
        family = feat["family"]
        attrs = feat["attr_scores"]

        # Sort attributes by score
        sorted_attrs = sorted(attrs.items(), key=lambda x: -x[1])
        top_attrs = ", ".join(f"{k}({v:.2f})" for k, v in sorted_attrs[:3])

        verdict = "likely fake" if conf > 0.5 else "likely real"

        # Build evidence text
        evidence = (
            f"Forensic evidence from automated detector (XGenDet):\n"
            f"- Detection confidence: {int(conf*100)}% ({verdict})\n"
            f"- Predicted generator type: {family}\n"
            f"- Artifact attribute scores (0=clean, 1=highly suspicious):\n"
        )
        for name, score in sorted_attrs:
            evidence += f"  {name:12s}: {score:.2f}\n"
        evidence += f"- Strongest signals: {top_attrs}\n"
        evidence += (
            f"\nExamine facial features including skin texture, edge consistency, "
            f"color distribution, geometric coherence, and lighting patterns. "
            f"Provide structured forensic analysis, then classify as real or fake."
        )

        # Get assistant reasoning
        assistant_msg = None
        for msg in item["messages"]:
            if msg["role"] == "assistant":
                assistant_msg = msg["content"]
                break

        if not assistant_msg:
            skipped += 1
            continue

        # Build JSONL entry matching qwen-vl-finetune format
        entry = {
            "image": full_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{evidence}",
                },
                {
                    "from": "gpt",
                    "value": assistant_msg,
                },
            ],
        }
        output_lines.append(json.dumps(entry))

    # Write JSONL
    out_path = "/home/sachin.chaudhary/Qwen2.5-VL/qwen-vl-finetune/qwenvl/data/dataset/forensight-hydrafake-train.jsonl"
    with open(out_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"\nWritten {len(output_lines)} samples to {out_path}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
