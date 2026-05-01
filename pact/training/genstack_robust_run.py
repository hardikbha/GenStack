"""Full GenStack robustness — XGenDet → evidence → Prism + PACT, one condition per process.

Usage:
    CUDA_VISIBLE_DEVICES=0 python genstack_robust_run.py --condition orig
    CUDA_VISIBLE_DEVICES=1 python genstack_robust_run.py --condition jpeg_70
    ...

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
XGENDET_CKPT = "/home/sachin.chaudhary/xgendet/checkpoints/hydrafake_finetune/best_model.pth"
PRISM_CKPT = "/home/sachin.chaudhary/xgendet/checkpoints/forensight_qwen3/checkpoint-3939"
PRISM_BASE = "/home/sachin.chaudhary/models/Qwen3-VL-8B-Instruct"
OUT_DIR = Path(
    "/home/sachin.chaudhary/xgendet/checkpoints/final_results/genstack_robust_out"
)

FAMILIES = ["Real", "Face Swapping", "Entire Face Gen", "Face Reenactment"]
ATTR_NAMES = [
    "texture", "edges", "color", "geometry", "semantics", "frequency"
]


def degrade(im, cond):
    if cond == "orig":
        return im
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


def progress_bar(score, width=10):
    filled = int(round(max(0, min(1, score)) * width))
    return "█" * filled + "░" * (width - filled)


def build_evidence(out):
    conf = out["confidence"].item()
    family_idx = out["family_logit"].argmax(dim=-1).item()
    attrs = out["attr_scores"][0].cpu().tolist()
    sorted_attrs = sorted(zip(ATTR_NAMES, attrs), key=lambda x: -x[1])
    verdict = "likely fake" if conf > 0.5 else "likely real"
    lines = [
        "A forensic detection system analyzed this face image and found:",
        f"- Detection confidence: {int(conf*100)}% ({verdict})",
        f"- Predicted generator type: {FAMILIES[family_idx]}",
        "- Artifact attribute scores (0-1 scale):",
    ]
    for n, v in sorted_attrs:
        lines.append(f"  {n:<11s} : {v:.2f} [{progress_bar(v)}]")
    top3 = sorted_attrs[:3]
    lines.append("- Strongest signals: " + ", ".join(f"{n}({v:.2f})"
                                                     for n, v in top3))
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a forensic image analyst specializing in deepfake detection. "
    "A detection system has already analyzed this image and provided structured "
    "evidence (attribute scores, confidence, generator type).\n\n"
    "Your task:\n"
    "1. Consider BOTH the visual content AND the detector's evidence\n"
    "2. Use structured reasoning with XML tags\n"
    "3. Reference specific visual observations that support or contradict the detector's findings\n"
    "4. Give your final answer as <answer> real </answer> or <answer> fake </answer>\n\n"
    "Reasoning format:\n"
    "<fast> Quick initial assessment based on detector evidence and first impression </fast>\n"
    "<reasoning> Detailed analysis referencing specific visual features and detector scores </reasoning>\n"
    "<conclusion> Synthesize all evidence into a final judgment </conclusion>\n"
    "<answer> real/fake </answer>")


def preprocess_pact(img):
    tfm = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    return tfm(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition",
                    required=True,
                    choices=[
                        "orig", "jpeg_90", "jpeg_70", "jpeg_60", "blur_1",
                        "blur_2", "blur_4"
                    ])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.condition}.jsonl"
    n_done = sum(1 for _ in open(out_path)) if out_path.exists() else 0

    samples = json.load(open(SUBSET_JSON))
    if args.limit:
        samples = samples[:args.limit]
    print(f"[{args.condition}] N={len(samples)} resume_from={n_done}",
          flush=True)

    print(f"[{args.condition}] loading XGenDet base...", flush=True)
    from models.xgendet import XGenDet
    xgendet = XGenDet()
    xgendet.load_state_dict(
        torch.load(XGENDET_CKPT, map_location="cpu",
                   weights_only=False)["model_state_dict"])
    xgendet = xgendet.cuda().eval()
    xt = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    print(f"[{args.condition}] loading PACT (v5+)...", flush=True)
    from models.xgendet_v5plus import XGenDetV5Plus
    pact = XGenDetV5Plus(v5_checkpoint_path=V5_CKPT)
    pact.load_branches_checkpoint(V5PLUS_CKPT)
    pact = pact.cuda().eval()

    print(f"[{args.condition}] loading Prism (Qwen3-VL SFT)...", flush=True)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(PRISM_BASE,
                                              trust_remote_code=True)
    prism = Qwen3VLForConditionalGeneration.from_pretrained(
        PRISM_CKPT,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0").eval()

    f_out = open(out_path, "a")
    t0 = time.time()
    for i, s in enumerate(samples):
        if i < n_done:
            continue
        try:
            img = Image.open(s["img_path"]).convert("RGB")
            img = degrade(img, args.condition)

            with torch.no_grad():
                xg_in = xt(img).unsqueeze(0).cuda()
                xg_out = xgendet(xg_in, return_heatmap=False)
            evidence = build_evidence(xg_out)

            with torch.no_grad():
                pact_x = preprocess_pact(img).unsqueeze(0).cuda().float()
                p_v = pact(pact_x)["prob"].squeeze().float().item()

            user_text = (f"\n\nDetector Evidence:\n{evidence}\n\n"
                         "Based on the image and the detector's analysis, "
                         "determine if this face is real or fake.")
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role":
                    "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img
                        },
                        {
                            "type": "text",
                            "text": user_text
                        },
                    ]
                },
            ]
            chat = processor.apply_chat_template(messages,
                                                 tokenize=False,
                                                 add_generation_prompt=True)
            inputs = processor(text=[chat],
                               images=[img],
                               padding=True,
                               return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                gen = prism.generate(
                    **inputs,
                    max_new_tokens=384,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            out_ids = gen[0][inputs["input_ids"].shape[1]:]
            rationale = processor.decode(out_ids,
                                         skip_special_tokens=True).strip()
            m = re.search(r"<answer>\s*(real|fake)\s*</answer>", rationale,
                          re.IGNORECASE)
            b_p = 1 if (m
                        and m.group(1).lower() == "fake") else (0 if m else -1)

            f_out.write(
                json.dumps({
                    "id":
                    s["id"],
                    "split":
                    s["split"],
                    "generator":
                    s["generator"],
                    "label":
                    s["label"],
                    "p_v":
                    p_v,
                    "b_p":
                    b_p,
                    "xg_conf":
                    float(xg_out["confidence"].item()),
                    "xg_family":
                    int(xg_out["family_logit"].argmax(dim=-1).item()),
                }) + "\n")
            f_out.flush()
        except Exception as e:
            f_out.write(
                json.dumps({
                    "id": s.get("id", "?"),
                    "error": str(e),
                    "label": s.get("label", -1),
                    "p_v": -1,
                    "b_p": -1,
                }) + "\n")
            f_out.flush()

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = max(i + 1 - n_done, 1) / max(elapsed, 1)
            eta = (len(samples) - i - 1) / max(rate, 1e-3) / 60
            print(
                f"[{args.condition}] {i+1}/{len(samples)} rate={rate:.2f}/s eta={eta:.1f}min",
                flush=True)

    f_out.close()
    print(f"[{args.condition}] done. wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
