"""
ForenSight: Fine-tune Qwen3-VL-8B with LoRA on HydraFake.
Input: image + XGenDet evidence. Output: structured reasoning + real/fake answer.
After training → auto eval on 52K test images.
"""

import os, sys, json, re, argparse, torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import average_precision_score, accuracy_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="/home/sachin.chaudhary/models/Qwen3-VL-8B-Instruct")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--train_data", default="checkpoints/forensight_sft_train.json")
    p.add_argument("--val_data", default="checkpoints/forensight_sft_val.json")
    p.add_argument("--output_dir", default="checkpoints/forensight_sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--test_dir", default="/home/sachin.chaudhary/hydrafake/jsons/test")
    p.add_argument("--xgendet_cache", default="checkpoints/xgendet_cached_features.json")
    p.add_argument("--xgendet_ckpt", default="checkpoints/hydrafake_finetune/best_model.pth")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def extract_answer(text):
    """Extract real/fake from <answer> tags."""
    match = re.search(r"<answer>\s*(real|fake)\s*</answer>", text.lower())
    if match:
        return 1 if match.group(1) == "fake" else 0
    # Fallback: check for keywords
    text_lower = text.lower()
    if "fake" in text_lower and "real" not in text_lower:
        return 1
    if "real" in text_lower and "fake" not in text_lower:
        return 0
    return -1  # Unclear


def build_evidence_for_image(img_path, xgendet_model, transform, device):
    """Run XGenDet on a single image and return evidence text."""
    try:
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = xgendet_model(img_t, return_heatmap=False)
        conf = out["confidence"].item()
        family_idx = out["family_logit"].argmax(dim=-1).item()
        attrs = out["attr_scores"][0].cpu().tolist()

        families = ["Real", "Face Swapping", "Entire Face Gen", "Face Reenactment"]
        attr_names = ["texture", "edges", "color", "geometry", "semantics", "frequency"]
        sorted_attrs = sorted(zip(attr_names, attrs), key=lambda x: -x[1])

        verdict = "likely fake" if conf > 0.5 else "likely real"
        text = f"Detection confidence: {int(conf*100)}% ({verdict})\n"
        text += f"Generator type: {families[family_idx]}\n"
        text += "Attribute scores: " + ", ".join(f"{n}={v:.2f}" for n, v in sorted_attrs) + "\n"
        return text
    except:
        return "Detection evidence unavailable.\n"


def train(args):
    from transformers import (
        Qwen3VLForConditionalGeneration,
        AutoProcessor,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"Loading model: {args.model_name}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load training data
    with open(args.train_data) as f:
        train_data = json.load(f)
    print(f"Training samples: {len(train_data)}")

    # Build dataset
    class ForenSightDataset(torch.utils.data.Dataset):
        def __init__(self, data, processor, max_len):
            self.data = data
            self.processor = processor
            self.max_len = max_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            image_path = item["image"]

            # Build messages for processor
            messages = []
            for msg in item["messages"]:
                if msg["role"] == "user":
                    # User message contains <image> tag
                    content_parts = []
                    if "<image>" in msg["content"]:
                        content_parts.append({"type": "image", "image": image_path})
                        text_part = msg["content"].replace("<image>", "").strip()
                    else:
                        text_part = msg["content"]
                    content_parts.append({"type": "text", "text": text_part})
                    messages.append({"role": "user", "content": content_parts})
                elif msg["role"] == "system":
                    messages.append({"role": "system", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})

            # Process with Qwen3-VL processor
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                inputs = self.processor(
                    text=[text],
                    images=[Image.open(image_path).convert("RGB")],
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                # Create labels (mask everything except assistant response)
                input_ids = inputs["input_ids"].squeeze(0)
                labels = input_ids.clone()

                # Simple approach: use full sequence as labels
                # The model learns to predict the assistant's response
                return {
                    "input_ids": input_ids,
                    "attention_mask": inputs["attention_mask"].squeeze(0),
                    "labels": labels,
                    "pixel_values": inputs.get("pixel_values", torch.zeros(1)),
                    "image_grid_thw": inputs.get("image_grid_thw", torch.zeros(1)),
                }
            except Exception as e:
                # Return dummy on error
                return {
                    "input_ids": torch.zeros(self.max_len, dtype=torch.long),
                    "attention_mask": torch.zeros(self.max_len, dtype=torch.long),
                    "labels": torch.full((self.max_len,), -100, dtype=torch.long),
                    "pixel_values": torch.zeros(1),
                    "image_grid_thw": torch.zeros(1),
                }

    train_dataset = ForenSightDataset(train_data, processor, args.max_len)

    # Training arguments
    os.makedirs(args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Save final adapter
    model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    print(f"Saved adapter to {args.output_dir}/final_adapter")


def evaluate_test(args):
    """Evaluate on HydraFake test splits using the trained model."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    print("Loading model for evaluation...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    adapter_path = os.path.join(args.output_dir, "final_adapter")
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"Loaded LoRA adapter from {adapter_path}")
    else:
        model = base_model
        print("WARNING: No adapter found, using base model")

    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Load XGenDet for evidence generation during eval
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.xgendet import XGenDet
    from data.augmentations import get_eval_transforms

    xgendet = XGenDet()
    xgendet_ckpt = torch.load(args.xgendet_ckpt, map_location="cpu")
    xgendet.load_state_dict(xgendet_ckpt["model_state_dict"])
    xgendet.eval()
    # Put xgendet on CPU to save GPU memory
    xgendet_device = torch.device("cpu")
    xgendet_transform = get_eval_transforms(224)

    SYSTEM_PROMPT = """You are a forensic image analyst. A detection system has analyzed this image. Based on the image and detector evidence, determine if this face is real or fake. Use <fast>, <reasoning>, <conclusion>, <answer> tags. Answer with <answer> real </answer> or <answer> fake </answer>."""

    test_dir = args.test_dir
    results = {}

    for split_name in ["id", "cm", "cf", "cd"]:
        split_dir = os.path.join(test_dir, split_name)
        if not os.path.isdir(split_dir):
            continue

        json_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".json"))
        split_preds, split_labels, per_gen = [], [], {}

        for jf in json_files:
            gen_name = jf.replace(".json", "")
            json_path = os.path.join(split_dir, jf)

            with open(json_path) as f:
                test_data = json.load(f)

            gp, gl = [], []
            for item in tqdm(test_data, desc=f"{split_name}/{gen_name}", leave=False):
                rel_path = item["images"][0]
                parts = rel_path.split("/")
                sub_path = "/".join(parts[2:])
                full_path = os.path.join("/home/sachin.chaudhary/hydrafake/test", sub_path)
                label = item["label"]

                if not os.path.exists(full_path):
                    continue

                # Get XGenDet evidence
                evidence = build_evidence_for_image(full_path, xgendet, xgendet_transform, xgendet_device)

                # Build prompt
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image", "image": full_path},
                        {"type": "text", "text": f"\nDetector Evidence:\n{evidence}\nIs this face real or fake?"},
                    ]},
                ]

                try:
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    img = Image.open(full_path).convert("RGB")
                    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
                    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.1,
                            do_sample=False,
                        )

                    response = processor.decode(output_ids[0], skip_special_tokens=True)
                    pred = extract_answer(response)

                    if pred >= 0:
                        gp.append(pred)
                        gl.append(label)
                except Exception as e:
                    continue

            if gp:
                gp_np, gl_np = np.array(gp), np.array(gl)
                gen_acc = accuracy_score(gl_np, gp_np)
                per_gen[gen_name] = {"acc": gen_acc, "n": len(gp)}
                split_preds.extend(gp)
                split_labels.extend(gl)
                print(f"  {gen_name}: Acc={gen_acc*100:.1f}%, n={len(gp)}")

        if split_preds:
            sp, sl = np.array(split_preds), np.array(split_labels)
            split_acc = accuracy_score(sl, sp)
            results[split_name] = {"acc": split_acc, "n": len(sp), "per_generator": per_gen}
            print(f"  {split_name.upper()}: Acc={split_acc*100:.1f}%")

    if results:
        avg_acc = np.mean([r["acc"] for r in results.values()])
        results["average"] = {"acc": avg_acc}
        print(f"\n  AVERAGE: Acc={avg_acc*100:.1f}%")

    out_path = os.path.join(args.output_dir, "test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


def main():
    args = parse_args()

    if not args.eval_only:
        train(args)

    print("\n" + "=" * 60)
    print("Running test evaluation on HydraFake...")
    print("=" * 60)
    evaluate_test(args)


if __name__ == "__main__":
    main()
