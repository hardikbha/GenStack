"""Merge sharded evaluation results into final test_results.json."""

import os, json, sys, numpy as np
from sklearn.metrics import accuracy_score


def main():
    shard_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/forensight_sft/eval_shards"
    output = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/forensight_sft/test_results.json"

    all_results = {}
    for f in sorted(os.listdir(shard_dir)):
        if f.startswith("shard_") and f.endswith(".json"):
            with open(os.path.join(shard_dir, f)) as fh:
                shard = json.load(fh)
                all_results.update(shard)

    # Aggregate by split
    splits = {}
    for key, val in all_results.items():
        split, gen = key.split("/")
        if split not in splits:
            splits[split] = {"per_generator": {}, "total_correct": 0, "total_n": 0}
        splits[split]["per_generator"][gen] = val
        splits[split]["total_correct"] += int(val["acc"] * val["n"])
        splits[split]["total_n"] += val["n"]

    final = {}
    for split, data in splits.items():
        acc = data["total_correct"] / data["total_n"] if data["total_n"] > 0 else 0
        final[split] = {"acc": acc, "n": data["total_n"], "per_generator": data["per_generator"]}
        print(f"{split.upper()}: Acc={acc*100:.1f}%, n={data['total_n']}")

    if final:
        avg = np.mean([v["acc"] for v in final.values()])
        final["average"] = {"acc": avg}
        print(f"AVERAGE: Acc={avg*100:.1f}%")

    with open(output, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
