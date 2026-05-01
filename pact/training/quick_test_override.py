"""Test if VLM ever overrides XGenDet's wrong predictions."""
import torch, sys, os, re
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from models.xgendet import XGenDet
from data.augmentations import get_eval_transforms

MODEL = "checkpoints/forensight_qwen3"
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
processor = AutoProcessor.from_pretrained(MODEL)

xgendet = XGenDet()
xgendet.load_state_dict(torch.load("checkpoints/hydrafake_finetune/best_model.pth", map_location="cpu")["model_state_dict"])
xgendet.eval()
transform = get_eval_transforms(224)
attr_names = ["texture", "edges", "color", "geometry", "semantics", "frequency"]
families = ["Real", "Face Swapping", "Entire Face Gen", "Face Reenactment"]

# Test cases where XGenDet is WRONG — does VLM override?
tests = [
    # XGenDet misses these fakes (conf < 0.5)
    ("/home/sachin.chaudhary/hydrafake/test/ICLight/1_fake/00740_ppt1_seed12345_left.png", 1, "ICLight fake"),
    ("/home/sachin.chaudhary/hydrafake/test/ICLight/1_fake/00790_ppt1_seed12345_bottom.png", 1, "ICLight fake2"),
    ("/home/sachin.chaudhary/hydrafake/test/StarGANv2/1_fake/000001.png", 1, "StarGANv2 fake"),
    ("/home/sachin.chaudhary/hydrafake/test/FFIW/1_fake/000001.png", 1, "FFIW fake"),
    # XGenDet falsely flags these reals
    ("/home/sachin.chaudhary/hydrafake/test/ICLight/0_real/01891.png", 0, "ICLight real"),
    ("/home/sachin.chaudhary/hydrafake/test/FFIW/0_real/000000.png", 0, "FFIW real"),
]

overrides = 0
total = 0

for img_path, label, desc in tests:
    if not os.path.exists(img_path):
        # Try finding the file
        parts = img_path.split("/")
        alt = os.path.join("/home/sachin.chaudhary/hydrafake/test", parts[-3], parts[-2])
        if os.path.isdir(alt):
            files = sorted(os.listdir(alt))[:1]
            if files:
                img_path = os.path.join(alt, files[0])
    if not os.path.exists(img_path):
        print(f"SKIP: {desc} — file not found")
        continue

    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        xout = xgendet(img_t, return_heatmap=False)
    conf = xout["confidence"].item()
    fam = families[xout["family_logit"].argmax().item()]
    attrs = xout["attr_scores"][0].tolist()
    sorted_attrs = sorted(zip(attr_names, attrs), key=lambda x: -x[1])
    xgendet_pred = 1 if conf > 0.5 else 0

    verdict = "likely fake" if conf > 0.5 else "likely real"
    evidence = f"Detection confidence: {int(conf*100)}% ({verdict})\n"
    evidence += f"Generator type: {fam}\n"
    evidence += "Attributes: " + ", ".join(f"{n}={v:.2f}" for n, v in sorted_attrs) + "\n"

    messages = [
        {"role": "system", "content": "You are a forensic image analyst. Based on the image and detector evidence, determine if this face is real or fake. The detector may be wrong — use your own visual judgment too. Use <answer> real </answer> or <answer> fake </answer>."},
        {"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": f"\nDetector Evidence:\n{evidence}\nIs this face real or fake? The detector may be incorrect — examine the image carefully."},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    response = processor.decode(out[0], skip_special_tokens=True)
    gen = response.split("examine the image carefully.")[-1].strip()

    match = re.search(r"<answer>\s*(real|fake)\s*</answer>", gen.lower())
    vlm_pred = -1
    if match:
        vlm_pred = 1 if match.group(1) == "fake" else 0

    xgendet_correct = xgendet_pred == label
    vlm_correct = vlm_pred == label
    did_override = vlm_pred != xgendet_pred

    total += 1
    if did_override:
        overrides += 1

    status = ""
    if did_override and vlm_correct:
        status = "OVERRIDE (VLM fixed XGenDet!)"
    elif did_override and not vlm_correct:
        status = "OVERRIDE (VLM made it worse)"
    elif not did_override and xgendet_correct:
        status = "AGREE (both correct)"
    else:
        status = "AGREE (both wrong)"

    gt = "FAKE" if label == 1 else "REAL"
    xp = "FAKE" if xgendet_pred == 1 else "REAL"
    vp = "FAKE" if vlm_pred == 1 else ("REAL" if vlm_pred == 0 else "???")
    print(f"{desc:20s} | GT={gt:4s} | XGenDet={xp:4s}(conf={conf:.2f}) | VLM={vp:4s} | {status}")

print(f"\nOverrides: {overrides}/{total}")
