"""Verify the Forensight Qwen3-VL SFT checkpoint loads and produces an answer."""
import os, sys, json, re, time
from pathlib import Path
import torch
from PIL import Image

CKPT = "/home/sachin.chaudhary/xgendet/checkpoints/forensight_qwen3/checkpoint-3939"
TEST_JSON = "/home/sachin.chaudhary/hydrafake/jsons/test/cf/iclight.json"
DATA_ROOT = "/home/sachin.chaudhary"

print("[1/4] Loading Qwen3-VL processor + model from", CKPT, flush=True)
t0 = time.time()
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

BASE = "/home/sachin.chaudhary/models/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    CKPT,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0")
model.eval()
print(f"  -> loaded in {time.time()-t0:.1f}s", flush=True)

print(
    "[2/4] Building 3 sample inputs (1 fake + 1 real, plus 1 with degradation)...",
    flush=True)
items = json.load(open(TEST_JSON))
sample_paths = []
for it in items:
    p = it["images"][0]
    if not p.startswith("/"): p = str(Path(DATA_ROOT) / p)
    if Path(p).exists():
        sample_paths.append((p, it["label"]))
    if len(sample_paths) >= 3: break
print(f"  -> {len(sample_paths)} samples", flush=True)

SYS = (
    "You are a deepfake forensics expert. Analyze the image. "
    "Provide reasoning then output your verdict as <answer>real</answer> or <answer>fake</answer>."
)


def run(img_path):
    messages = [
        {
            "role": "system",
            "content": SYS
        },
        {
            "role":
            "user",
            "content": [{
                "type": "image",
                "image": f"file://{img_path}"
            }, {
                "type": "text",
                "text": "Determine if this face is real or fake."
            }]
        },
    ]
    text = processor.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    image = Image.open(img_path).convert("RGB")
    inputs = processor(text=[text],
                       images=[image],
                       padding=True,
                       return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    out_ids = gen[0][inputs["input_ids"].shape[1]:]
    return processor.decode(out_ids, skip_special_tokens=True).strip()


print("[3/4] Running 3 inferences...", flush=True)
for path, label in sample_paths:
    t0 = time.time()
    txt = run(path)
    m = re.search(r"<answer>\s*(real|fake)\s*</answer>", txt, re.IGNORECASE)
    pred = m.group(1).lower() if m else "?"
    correct = (pred == ("fake" if label == 1 else "real"))
    print(
        f"  [{time.time()-t0:.1f}s] gt={label} pred={pred} ok={correct}  output_head={txt[:120]!r}",
        flush=True)

print("[4/4] Smoke complete.", flush=True)
