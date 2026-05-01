"""PACT-only robustness sweep — runs all 7 conditions sequentially on one GPU.

Output: 7 JSONL files in checkpoints/final_results/robustness_out/{condition}.jsonl
Plus: a summary table robustness_pact_summary.{json,md}.

Reasoning: full GenStack robustness requires re-running both branches under each
degradation. The Prism (Qwen3-VL SFT) branch needs detector-evidence inputs from
the base XGenDet that we don't have wired into a degradation-aware pipeline. We
therefore report the discriminative branch (PACT) under degradations as a
defensible, self-contained robustness story.
"""
import argparse, io, json, sys, time
from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image, ImageFilter
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
SUBSET_JSON = "/home/sachin.chaudhary/xgendet/checkpoints/final_results/robustness_subset.json"
V5_CKPT = "/home/sachin.chaudhary/xgendet/checkpoints/v5_resume_hl/best_model.pth"
V5PLUS_CKPT = "/home/sachin.chaudhary/xgendet/checkpoints/v5plus/best_branches.pth"
OUT_DIR = Path(
    "/home/sachin.chaudhary/xgendet/checkpoints/final_results/robustness_out")

CONDITIONS = [
    "orig", "jpeg_90", "jpeg_70", "jpeg_60", "blur_1", "blur_2", "blur_4"
]


def degrade(im, cond):
    if cond == "orig": return im
    if cond.startswith("jpeg_"):
        q = int(cond.split("_")[1])
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if cond.startswith("blur_"):
        s = float(cond.split("_")[1])
        return im.filter(ImageFilter.GaussianBlur(radius=s))
    raise ValueError(cond)


def preprocess(img, size=224):
    tfm = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    return tfm(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = json.load(open(SUBSET_JSON))
    if args.limit: samples = samples[:args.limit]
    print(f"[init] {len(samples)} samples × {len(CONDITIONS)} conditions",
          flush=True)

    print("[init] loading PACT (v5+)...", flush=True)
    from models.xgendet_v5plus import XGenDetV5Plus
    model = XGenDetV5Plus(v5_checkpoint_path=V5_CKPT)
    model.load_branches_checkpoint(V5PLUS_CKPT)
    model = model.cuda().eval()
    print("[init] loaded.", flush=True)

    summary = {}
    for cond in CONDITIONS:
        out_path = OUT_DIR / f"pact_{cond}.jsonl"
        t0 = time.time()
        f_out = open(out_path, "w")
        n_correct = 0
        n_total = 0
        per_split = defaultdict(lambda: [0, 0])  # split -> [correct, total]
        per_gen = defaultdict(lambda: [0, 0])

        # Batched inference
        i = 0
        while i < len(samples):
            batch = samples[i:i + args.batch]
            imgs = []
            metas = []
            for s in batch:
                try:
                    im = Image.open(s["img_path"]).convert("RGB")
                    im = degrade(im, cond)
                    imgs.append(preprocess(im))
                    metas.append(s)
                except Exception:
                    pass
            if not imgs:
                i += args.batch
                continue
            x = torch.stack(imgs).cuda().float()
            with torch.no_grad():
                p_v = model(x)["prob"].squeeze(1).float().cpu().tolist()
            for s, p in zip(metas, p_v):
                pred = int(p >= 0.5)
                ok = int(pred == s["label"])
                n_correct += ok
                n_total += 1
                per_split[s["split"]][0] += ok
                per_split[s["split"]][1] += 1
                per_gen[s["generator"]][0] += ok
                per_gen[s["generator"]][1] += 1
                f_out.write(
                    json.dumps({
                        "id": s["id"],
                        "split": s["split"],
                        "generator": s["generator"],
                        "label": s["label"],
                        "p_v": p,
                    }) + "\n")
            i += args.batch
            if (i // args.batch) % 10 == 0:
                rate = i / max(time.time() - t0, 1e-3)
                print(
                    f"  [{cond}] {i}/{len(samples)} acc={n_correct/max(n_total,1)*100:.2f}% rate={rate:.0f}/s",
                    flush=True)
        f_out.close()
        elapsed = time.time() - t0
        acc = n_correct / max(n_total, 1)
        split_accs = {k: v[0] / max(v[1], 1) for k, v in per_split.items()}
        summary[cond] = {
            "n": n_total,
            "acc": acc,
            "elapsed_s": elapsed,
            "by_split": split_accs,
            "by_generator": {
                k: v[0] / max(v[1], 1)
                for k, v in per_gen.items()
            },
        }
        print(
            f"[{cond}] DONE  n={n_total}  acc={acc*100:.2f}%  "
            f"ID={split_accs.get('id',0)*100:.2f}  CM={split_accs.get('cm',0)*100:.2f}  "
            f"CF={split_accs.get('cf',0)*100:.2f}  CD={split_accs.get('cd',0)*100:.2f}  "
            f"({elapsed:.1f}s)",
            flush=True)

    out_json = OUT_DIR / "pact_robustness_summary.json"
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"\n[done] wrote {out_json}", flush=True)

    # Markdown table
    md = ["# PACT Robustness Summary", ""]
    md += [
        f"Subset: stratified ~{len(samples)} samples across 22 generators × 4 splits.",
        ""
    ]
    md += [
        "| Condition | ID | CM | CF | CD | **Avg** |",
        "|---|---:|---:|---:|---:|---:|"
    ]
    for cond in CONDITIONS:
        s = summary[cond]
        md.append(
            f"| {cond} | {s['by_split'].get('id',0)*100:.2f} | {s['by_split'].get('cm',0)*100:.2f} | "
            f"{s['by_split'].get('cf',0)*100:.2f} | {s['by_split'].get('cd',0)*100:.2f} | "
            f"**{s['acc']*100:.2f}** |")
    open(OUT_DIR / "pact_robustness_summary.md", "w").write("\n".join(md))
    print(f"[done] wrote {OUT_DIR/'pact_robustness_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
