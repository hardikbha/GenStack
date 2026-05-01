"""Production robustness inference: run PACT + Prism on subset under one degradation condition.

Usage:
    python training/robustness_run.py --condition orig --gpu 0 --out_dir robustness_out
    python training/robustness_run.py --condition jpeg_70 --gpu 1 --out_dir robustness_out
    python training/robustness_run.py --condition blur_2 --gpu 2 --out_dir robustness_out

Conditions: orig, jpeg_90, jpeg_70, jpeg_60, blur_1, blur_2, blur_4
"""
import argparse, io, json, os, re, sys, time
from pathlib import Path
import torch
from PIL import Image, ImageFilter
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
SUBSET_JSON = "/home/sachin.chaudhary/xgendet/checkpoints/final_results/robustness_subset.json"
V5_CKPT = "/home/sachin.chaudhary/xgendet/checkpoints/v5_resume_hl/best_model.pth"
V5PLUS_CKPT = "/home/sachin.chaudhary/xgendet/checkpoints/v5plus/best_branches.pth"
PRISM_CKPT = "/home/sachin.chaudhary/xgendet/checkpoints/forensight_qwen3/checkpoint-3939"
PRISM_BASE = "/home/sachin.chaudhary/models/Qwen3-VL-8B-Instruct"

CONDITIONS = {
    "orig": lambda im: im,
    "jpeg_90": lambda im: _jpeg(im, 90),
    "jpeg_70": lambda im: _jpeg(im, 70),
    "jpeg_60": lambda im: _jpeg(im, 60),
    "blur_1": lambda im: im.filter(ImageFilter.GaussianBlur(radius=1)),
    "blur_2": lambda im: im.filter(ImageFilter.GaussianBlur(radius=2)),
    "blur_4": lambda im: im.filter(ImageFilter.GaussianBlur(radius=4)),
}


def _jpeg(im, q):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def preprocess_pact(img, size=224):
    tfm = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    return tfm(img)


SYS_PRISM = (
    "You are a deepfake forensics expert. Analyze the image carefully. "
    "Provide your reasoning, then output your verdict as "
    "<answer>real</answer> or <answer>fake</answer>.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition",
                    required=True,
                    choices=list(CONDITIONS.keys()))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument(
        "--out_dir",
        default=
        "/home/sachin.chaudhary/xgendet/checkpoints/final_results/robustness_out"
    )
    ap.add_argument("--limit", type=int, default=0, help="Cap samples (debug)")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.condition}.jsonl"
    if out_path.exists():
        n_done = sum(1 for _ in open(out_path))
    else:
        n_done = 0

    samples = json.load(open(SUBSET_JSON))
    if args.limit:
        samples = samples[:args.limit]
    print(
        f"[{args.condition}] gpu={args.gpu} total={len(samples)} resume_from={n_done}",
        flush=True)

    print(f"[{args.condition}] loading PACT...", flush=True)
    from models.xgendet_v5plus import XGenDetV5Plus
    pact = XGenDetV5Plus(v5_checkpoint_path=V5_CKPT)
    pact.load_branches_checkpoint(V5PLUS_CKPT)
    pact = pact.cuda().eval()

    print(f"[{args.condition}] loading Prism (Qwen3-VL-SFT)...", flush=True)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(PRISM_BASE,
                                              trust_remote_code=True)
    prism = Qwen3VLForConditionalGeneration.from_pretrained(
        PRISM_CKPT,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()

    degrade = CONDITIONS[args.condition]
    f_out = open(out_path, "a")
    t_start = time.time()
    for i, s in enumerate(samples):
        if i < n_done:
            continue
        try:
            img = Image.open(s["img_path"]).convert("RGB")
            img = degrade(img)

            # PACT
            with torch.no_grad():
                x = preprocess_pact(img).unsqueeze(0).cuda().float()
                p_v = pact(x)["prob"].squeeze().float().item()

            # Prism
            messages = [
                {
                    "role": "system",
                    "content": SYS_PRISM
                },
                {
                    "role":
                    "user",
                    "content": [{
                        "type": "image",
                        "image": img
                    }, {
                        "type":
                        "text",
                        "text":
                        "Determine if this face is real or fake."
                    }]
                },
            ]
            text = processor.apply_chat_template(messages,
                                                 tokenize=False,
                                                 add_generation_prompt=True)
            inputs = processor(text=[text],
                               images=[img],
                               padding=True,
                               return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                gen = prism.generate(**inputs,
                                     max_new_tokens=256,
                                     do_sample=False,
                                     temperature=None,
                                     top_p=None,
                                     top_k=None)
            out_ids = gen[0][inputs["input_ids"].shape[1]:]
            rationale = processor.decode(out_ids,
                                         skip_special_tokens=True).strip()
            m = re.search(r"<answer>\s*(real|fake)\s*</answer>", rationale,
                          re.IGNORECASE)
            b_p = 1 if (m
                        and m.group(1).lower() == "fake") else (0 if m else -1)

            f_out.write(
                json.dumps({
                    "id": s["id"],
                    "split": s["split"],
                    "generator": s["generator"],
                    "label": s["label"],
                    "p_v": p_v,
                    "b_p": b_p,
                    "rationale": rationale,
                }) + "\n")
            f_out.flush()
        except Exception as e:
            f_out.write(
                json.dumps({
                    "id": s["id"],
                    "error": str(e),
                    "label": s.get("label", -1),
                    "p_v": -1,
                    "b_p": -1,
                    "rationale": "",
                }) + "\n")
            f_out.flush()

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1 - n_done) / max(elapsed, 1e-3)
            eta = (len(samples) - i - 1) / max(rate, 1e-3) / 60
            print(
                f"[{args.condition}] {i+1}/{len(samples)} rate={rate:.2f}/s eta={eta:.1f}min",
                flush=True)

    f_out.close()
    print(f"[{args.condition}] done. wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
