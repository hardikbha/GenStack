"""Quick test: run ForenSight on 3 images and print full output."""
import torch, sys, json, os
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from models.xgendet import XGenDet
from data.augmentations import get_eval_transforms

MODEL = "checkpoints/forensight_qwen3"

print("Loading ForenSight VLM...")
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
processor = AutoProcessor.from_pretrained(MODEL)

print("Loading XGenDet...")
xgendet = XGenDet()
xgendet.load_state_dict(torch.load("checkpoints/hydrafake_finetune/best_model.pth", map_location="cpu")["model_state_dict"])
xgendet.eval()
transform = get_eval_transforms(224)

attr_names = ["texture", "edges", "color", "geometry", "semantics", "frequency"]
families = ["Real", "Face Swapping", "Entire Face Gen", "Face Reenactment"]

tests = [
    ("/home/sachin.chaudhary/hydrafake/test/AdobeFirefly/1_fake/img_459.png", 1, "FAKE AdobeFirefly"),
    ("/home/sachin.chaudhary/hydrafake/test/FaceForensics++/0_real/044_215.png", 0, "REAL FaceForensics++"),
    ("/home/sachin.chaudhary/hydrafake/test/ICLight/1_fake/00722_ppt0_seed12345_right.png", 1, "FAKE ICLight"),
]

for img_path, label, desc in tests:
    print("\n" + "=" * 60)
    print(f"Image: {desc}")
    print(f"Ground Truth: {'FAKE' if label == 1 else 'REAL'}")
    print("=" * 60)

    # XGenDet evidence
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        xout = xgendet(img_t, return_heatmap=False)
    conf = xout["confidence"].item()
    fam = families[xout["family_logit"].argmax().item()]
    attrs = xout["attr_scores"][0].tolist()
    sorted_attrs = sorted(zip(attr_names, attrs), key=lambda x: -x[1])

    verdict = "likely fake" if conf > 0.5 else "likely real"
    evidence = f"Detection confidence: {int(conf*100)}% ({verdict})\n"
    evidence += f"Generator type: {fam}\n"
    evidence += "Attributes: " + ", ".join(f"{n}={v:.2f}" for n, v in sorted_attrs) + "\n"

    print(f"\nXGenDet Evidence:")
    print(evidence)

    # VLM inference
    messages = [
        {"role": "system", "content": "You are a forensic image analyst. Based on the image and detector evidence, determine if this face is real or fake. Use <answer> real </answer> or <answer> fake </answer>."},
        {"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": f"\nDetector Evidence:\n{evidence}\nIs this face real or fake?"},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    response = processor.decode(out[0], skip_special_tokens=True)
    # Extract generated part
    gen = response.split("Is this face real or fake?")[-1].strip()
    print(f"ForenSight Response:\n{gen}")

    # Check answer
    import re
    match = re.search(r"<answer>\s*(real|fake)\s*</answer>", gen.lower())
    if match:
        pred = match.group(1)
        correct = (pred == "fake" and label == 1) or (pred == "real" and label == 0)
        print(f"\nPrediction: {pred.upper()} ({'CORRECT' if correct else 'WRONG'})")
    else:
        print(f"\nCould not extract answer from response")
