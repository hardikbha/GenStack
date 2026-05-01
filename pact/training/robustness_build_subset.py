"""Build the stratified test subset for robustness experiments.

Picks 220 samples per generator across 23 HydraFake generators (≈5060 total).
Saves to checkpoints/final_results/robustness_subset.json.
"""
import json
from pathlib import Path
from collections import defaultdict
import random

random.seed(42)

DATA_ROOT = Path("/home/sachin.chaudhary")
TEST_DIR = Path("/home/sachin.chaudhary/hydrafake/jsons/test")
OUT = Path(
    "/home/sachin.chaudhary/xgendet/checkpoints/final_results/robustness_subset.json"
)
PER_GEN = 220

# Map filename → generator name as used by the rest of the pipeline
ALIAS = {
    "FaceForensics++": "ff",
    "infiniteyou": "infiniteyou",
    "FFIW": "FFIW",
}


def main():
    samples = []
    counts = defaultdict(int)
    for split_dir in ["id", "cm", "cf", "cd"]:
        for jpath in sorted((TEST_DIR / split_dir).glob("*.json")):
            if "_part" in jpath.stem:
                continue  # skip FFIW shards
            try:
                items = json.load(open(jpath))
            except Exception:
                continue
            gen_name = jpath.stem
            # Balance real/fake within each generator
            real = [x for x in items if x["label"] == 0]
            fake = [x for x in items if x["label"] == 1]
            random.shuffle(real)
            random.shuffle(fake)
            n_each = PER_GEN // 2
            picks = real[:n_each] + fake[:PER_GEN - len(real[:n_each])]
            for x in picks:
                p = x["images"][0]
                if not p.startswith("/"):
                    p = str(DATA_ROOT / p)
                if not Path(p).exists():
                    continue
                samples.append({
                    "split":
                    split_dir,
                    "generator":
                    gen_name,
                    "label":
                    int(x["label"]),
                    "img_path":
                    p,
                    "id":
                    f"{split_dir}__{gen_name}__{x.get('video_id', counts[gen_name])}__{counts[gen_name]:04d}",
                })
                counts[gen_name] += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(samples, open(OUT, "w"))
    print(f"Wrote {len(samples)} samples to {OUT}")
    print("Per-generator counts:")
    for g, n in sorted(counts.items()):
        print(f"  {g:25s} {n}")


if __name__ == "__main__":
    main()
