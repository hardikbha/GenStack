"""GenStack full pipeline smoke test: XGenDet → evidence → Prism on small mixed sample.

Verifies that with XGenDet evidence in the prompt (matching training format), Prism
recovers from the 6.9% smoke-test accuracy back to 70%+. If yes, we can run the full
robustness sweep.
"""
import sys, json, re, time
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
DATA_ROOT = "/home/sachin.chaudhary"
SUBSET_JSON = "/home/sachin.chaudhary/xgendet/checkpoints/final_results/robustness_subset.json"
XGENDET_CKPT = "/home/sachin.chaudhary/xgendet/checkpoints/hydrafake_finetune/best_model.pth"
PRISM_CKPT = "/home/sachin.chaudhary/xgendet/checkpoints/forensight_qwen3/checkpoint-3939"
PRISM_BASE = "/home/sachin.chaudhary/models/Qwen3-VL-8B-Instruct"

FAMILIES = ["Real", "Face Swapping", "Entire Face Gen", "Face Reenactment"]
ATTR_NAMES = [
    "texture", "edges", "color", "geometry", "semantics", "frequency"
]


def progress_bar(score, width=10):
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


def build_evidence(out):
    """Format XGenDet output to match Prism's SFT training prompt."""
    conf = out["confidence"].item()
    family_idx = out["family_logit"].argmax(dim=-1).item()
    attrs = out["attr_scores"][0].cpu().tolist()
    sorted_attrs = sorted(zip(ATTR_NAMES, attrs), key=lambda x: -x[1])
    verdict = "likely fake" if conf > 0.5 else "likely real"

    lines = [
        f"A forensic detection system analyzed this face image and found:",
        f"- Detection confidence: {int(conf*100)}% ({verdict})",
        f"- Predicted generator type: {FAMILIES[family_idx]}",
        f"- Artifact attribute scores (0-1 scale):",
    ]
    for n, v in sorted(zip(ATTR_NAMES, attrs), key=lambda x: -x[1]):
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


def main():
    print("[1/5] loading XGenDet base ...", flush=True)
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
    print("  -> loaded", flush=True)

    print("[2/5] loading Prism (Qwen3-VL SFT) ...", flush=True)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(PRISM_BASE,
                                              trust_remote_code=True)
    prism = Qwen3VLForConditionalGeneration.from_pretrained(
        PRISM_CKPT,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0").eval()
    print("  -> loaded", flush=True)

    print("[3/5] picking 12 mixed samples ...", flush=True)
    samples = json.load(open(SUBSET_JSON))
    # balanced: 3 real + 3 fake per split = 24 samples
    by_split_label = {
        (sp, lbl): []
        for sp in ["id", "cm", "cf", "cd"]
        for lbl in (0, 1)
    }
    for s in samples:
        key = (s["split"], s["label"])
        if key in by_split_label and len(by_split_label[key]) < 3:
            by_split_label[key].append(s)
    picks = sum(by_split_label.values(), [])
    print(f"  -> {len(picks)} samples", flush=True)

    print("[4/5] running XGenDet → evidence → Prism on each ...", flush=True)
    n_correct = 0
    for s in picks:
        try:
            img = Image.open(s["img_path"]).convert("RGB")
            with torch.no_grad():
                xg_in = xt(img).unsqueeze(0).cuda()
                xg_out = xgendet(xg_in, return_heatmap=False)
            evidence = build_evidence(xg_out)
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
            if m:
                pred = 1 if m.group(1).lower() == "fake" else 0
            else:
                pred = -1
            ok = (pred == s["label"])
            if ok and pred >= 0:
                n_correct += 1
            head = rationale[:80].replace("\n", " ")
            print(
                f"  [{s['split']}/{s['generator']:<20s}] gt={s['label']} pred={pred:>2d} ok={ok}  raw={head!r}",
                flush=True)
        except Exception as e:
            print(f"  fail on {s['img_path']}: {e}", flush=True)

    acc = n_correct / max(len(picks), 1)
    print(f"\n[5/5] SMOKE ACCURACY: {n_correct}/{len(picks)} = {acc*100:.1f}%",
          flush=True)
    if acc >= 0.6:
        print("=== SMOKE PASSED ===", flush=True)
    else:
        print(
            "=== SMOKE FAILED — Prism still misclassifying. Aborting full sweep. ===",
            flush=True)


if __name__ == "__main__":
    main()
