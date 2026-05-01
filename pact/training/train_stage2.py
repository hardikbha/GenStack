"""
XGenDet Stage 2 Training: Fine-tune Qwen2.5-VL-7B with LoRA
for forensic explanation generation.

Uses annotated data: (image, heatmap, stage1_output) -> (attributes, explanation)
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


class ExplanationDataset(Dataset):
    """Dataset for Stage 2 MLLM fine-tuning."""

    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        heatmap_root: str = None,
    ):
        """
        Args:
            annotation_file: JSONL file with annotations
            image_root: Root directory for images
            heatmap_root: Root directory for saved heatmaps (numpy .npy files)
        """
        self.image_root = image_root
        self.heatmap_root = heatmap_root
        self.data = []

        with open(annotation_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                self.data.append(entry)

        print(f"Loaded {len(self.data)} explanation annotations")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Load image
        img_path = os.path.join(self.image_root, entry["image_path"])
        image = Image.open(img_path).convert("RGB")

        # Load heatmap if available
        heatmap = None
        if self.heatmap_root and "heatmap_path" in entry:
            hmap_path = os.path.join(self.heatmap_root, entry["heatmap_path"])
            if os.path.exists(hmap_path):
                heatmap = np.load(hmap_path)

        # Stage 1 outputs
        stage1 = entry.get("stage1_outputs", {
            "confidence": 0.5,
            "family": 0,
            "attr_scores": [0.5] * 6,
        })

        # Target annotations
        target_attrs = entry.get("attributes", {})
        target_explanation = entry.get("explanation", "")

        return {
            "image": image,
            "heatmap": heatmap,
            "stage1_outputs": stage1,
            "target_attributes": target_attrs,
            "target_explanation": target_explanation,
        }


def collate_fn(batch):
    """Custom collate - keeps PIL images and dicts as-is."""
    return batch


def parse_args():
    parser = argparse.ArgumentParser(description="XGenDet Stage 2 Training")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Data
    parser.add_argument("--annotation_file", type=str, required=True,
                        help="JSONL file with training annotations")
    parser.add_argument("--val_annotation_file", type=str, default=None)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--heatmap_root", type=str, default=None)

    # Training
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage2")
    parser.add_argument("--exp_name", type=str, default="xgendet_stage2")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def train():
    args = parse_args()
    torch.manual_seed(args.seed)

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load MLLM
    from models.mllm_module import MLLMExplainer, ATTRIBUTE_NAMES, parse_mllm_output

    explainer = MLLMExplainer(
        model_name=args.model_name,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_new_tokens=args.max_new_tokens,
    )
    explainer.load_model()
    model = explainer.model
    processor = explainer.processor

    # Dataset
    train_dataset = ExplanationDataset(
        annotation_file=args.annotation_file,
        image_root=args.image_root,
        heatmap_root=args.heatmap_root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # PIL images can't be pickled easily
    )

    # Optimizer (only LoRA params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)

    # Linear warmup + cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print(f"\nStage 2 Training: {args.epochs} epochs, {len(train_loader)} steps/epoch")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"Total optimization steps: {total_steps}")

    global_step = 0
    model.train()

    from qwen_vl_utils import process_vision_info

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            batch_loss = 0.0

            for sample in batch:
                # Prepare training example
                training_data = explainer.prepare_training_data(
                    image=sample["image"],
                    heatmap=sample["heatmap"],
                    stage1_outputs=sample["stage1_outputs"],
                    target_attributes=sample["target_attributes"],
                    target_explanation=sample["target_explanation"],
                )

                messages = training_data["messages"]

                # Tokenize with chat template
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                image_inputs, video_inputs = process_vision_info(
                    [messages[0]]  # Only user message has images
                )

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Create labels (mask input tokens with -100)
                labels = inputs["input_ids"].clone()

                # Find where assistant response starts
                # Mask everything before the assistant's response
                target_text = training_data["target"]
                target_ids = processor.tokenizer.encode(target_text, add_special_tokens=False)
                target_len = len(target_ids)
                input_len = labels.shape[1] - target_len

                if input_len > 0:
                    labels[:, :input_len] = -100

                inputs["labels"] = labels

                # Forward pass
                outputs = model(**inputs)
                loss = outputs.loss / args.grad_accum
                loss.backward()
                batch_loss += loss.item()

            num_batches += 1
            epoch_loss += batch_loss

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch} step {global_step} "
                          f"loss={avg_loss:.4f} lr={lr:.2e}")

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(exp_dir, f"step_{global_step}")
                    explainer.save_lora_weights(save_path)

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch}/{args.epochs} - Loss: {avg_epoch_loss:.4f}")

        # Save epoch checkpoint
        save_path = os.path.join(exp_dir, f"epoch_{epoch}")
        explainer.save_lora_weights(save_path)

    # Save final
    final_path = os.path.join(exp_dir, "final")
    explainer.save_lora_weights(final_path)
    print(f"\nTraining complete. Final weights saved to {final_path}")


if __name__ == "__main__":
    train()
