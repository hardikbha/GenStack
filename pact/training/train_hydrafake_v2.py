"""
XGenDet v2 Training on HydraFake — Improved with OHEM + Label Smoothing.
Automatically runs test evaluation after training completes.
Supports multi-GPU DataParallel (heatmap disabled during training, enabled for eval).
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from collections import defaultdict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False
from sklearn.metrics import average_precision_score, accuracy_score
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xgendet import XGenDet
from data.hydrafake_dataset import (
    create_hydrafake_dataloader,
    HydraFakeDataset,
    HydraFakeTestDataset,
)
from training.losses import XGenDetLoss


# ─── OHEM: Online Hard Example Mining ───────────────────────────────────

class OHEMSampler:
    """Track per-sample losses and resample hard examples."""

    def __init__(self, dataset_size, hard_ratio=0.5, warmup_epochs=3):
        self.dataset_size = dataset_size
        self.hard_ratio = hard_ratio
        self.warmup_epochs = warmup_epochs
        self.sample_losses = np.zeros(dataset_size)
        self.sample_counts = np.zeros(dataset_size)

    def update(self, indices, losses):
        """Update loss history for sampled indices."""
        for idx, loss in zip(indices, losses):
            if idx < self.dataset_size:
                self.sample_losses[idx] = 0.9 * self.sample_losses[idx] + 0.1 * loss
                self.sample_counts[idx] += 1

    def get_sampler(self, epoch, labels):
        """Get weighted sampler emphasizing hard examples."""
        if epoch < self.warmup_epochs:
            # During warmup: balanced sampling only
            class_counts = [labels.count(0), labels.count(1)]
            weights = [1.0 / class_counts[l] for l in labels]
            return WeightedRandomSampler(weights, len(labels), replacement=True)

        # After warmup: combine class balance + hard example emphasis
        class_counts = [labels.count(0), labels.count(1)]
        balance_weights = np.array([1.0 / class_counts[l] for l in labels])

        # Hard example weights: higher loss → higher weight
        loss_weights = np.ones(len(labels))
        if self.sample_counts.sum() > 0:
            valid = self.sample_counts > 0
            if valid.sum() > 0:
                normalized_losses = np.zeros(len(labels))
                normalized_losses[valid] = self.sample_losses[valid]
                # Top hard_ratio fraction gets 3x weight
                threshold = np.percentile(normalized_losses[valid], (1 - self.hard_ratio) * 100)
                loss_weights[normalized_losses >= threshold] = 3.0

        combined = balance_weights * loss_weights
        combined = combined / combined.sum() * len(labels)
        return WeightedRandomSampler(combined.tolist(), len(labels), replacement=True)


# ─── Label Smoothing BCE ────────────────────────────────────────────────

class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logit, target):
        target_smooth = target.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logit.squeeze(-1), target_smooth)


# ─── Main ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="XGenDet v2 HydraFake Training")

    # Model
    parser.add_argument("--clip_model", type=str, default="ViT-L/14")
    parser.add_argument("--num_prompt_tokens", type=int, default=8)
    parser.add_argument("--num_prototypes", type=int, default=128)
    parser.add_argument("--proto_dim", type=int, default=128)
    parser.add_argument("--shuffle_patch_size", type=int, default=32)

    # Training
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_ln", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)

    # OHEM
    parser.add_argument("--ohem", action="store_true", default=True)
    parser.add_argument("--ohem_hard_ratio", type=float, default=0.5)
    parser.add_argument("--ohem_warmup", type=int, default=3)

    # Label smoothing
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    # Data
    parser.add_argument("--train_json", type=str,
                        default="/home/sachin.chaudhary/hydrafake/jsons/train/all.json")
    parser.add_argument("--val_json", type=str,
                        default="/home/sachin.chaudhary/hydrafake/jsons/val/all.json")
    parser.add_argument("--data_root", type=str, default="/home/sachin.chaudhary")
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--jpeg_prob", type=float, default=0.5)
    parser.add_argument("--blur_prob", type=float, default=0.5)

    # Loss weights
    parser.add_argument("--w_family", type=float, default=0.5)
    parser.add_argument("--w_proto_div", type=float, default=0.3)
    parser.add_argument("--w_proto_compact", type=float, default=0.2)
    parser.add_argument("--w_heatmap", type=float, default=0.3)
    parser.add_argument("--w_attr", type=float, default=0.1)
    parser.add_argument("--w_calib", type=float, default=0.2)

    # Logging
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--exp_name", type=str, default="hydrafake_v2")
    parser.add_argument("--log_freq", type=int, default=25)
    parser.add_argument("--save_freq", type=int, default=1)

    # Resume / fine-tune
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)

    # Test
    parser.add_argument("--test_dir", type=str,
                        default="/home/sachin.chaudhary/hydrafake/jsons/test")

    args = parser.parse_args()
    return args


def validate(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, families in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs, return_heatmap=False)
            probs = outputs["confidence"].squeeze(-1).cpu().numpy()
            all_preds.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    ap = average_precision_score(all_labels, np.array(all_preds))
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    model.train()
    return {"ap": ap, "acc": acc}


def evaluate_test_splits(model, args, device):
    test_dir = args.test_dir
    results = {}
    model.eval()
    for split_name in ["id", "cm", "cf", "cd"]:
        split_dir = os.path.join(test_dir, split_name)
        if not os.path.isdir(split_dir):
            continue
        json_files = sorted([f for f in os.listdir(split_dir) if f.endswith(".json")])
        split_preds, split_labels, per_gen = [], [], {}
        for jf in json_files:
            gen_name = jf.replace(".json", "")
            dataset = HydraFakeTestDataset(
                json_path=os.path.join(split_dir, jf),
                data_root=args.data_root,
                image_root="/home/sachin.chaudhary/hydrafake/test",
                crop_size=args.crop_size,
            )
            if len(dataset) == 0:
                continue
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            gp, gl = [], []
            with torch.no_grad():
                for imgs, labels in loader:
                    imgs = imgs.to(device)
                    outputs = model(imgs, return_heatmap=False)
                    probs = outputs["confidence"].squeeze(-1).cpu().numpy()
                    gp.extend(probs.tolist())
                    gl.extend(labels.numpy().tolist())
            gp_np, gl_np = np.array(gp), np.array(gl)
            gen_ap = average_precision_score(gl_np, gp_np) if len(np.unique(gl_np)) > 1 else -1
            gen_acc = accuracy_score(gl_np, (gp_np > 0.5).astype(int))
            per_gen[gen_name] = {"ap": gen_ap, "acc": gen_acc, "n": len(dataset)}
            split_preds.extend(gp)
            split_labels.extend(gl)
            print(f"    {gen_name}: Acc={gen_acc*100:.1f}%, AP={gen_ap:.4f}, n={len(dataset)}")

        sp, sl = np.array(split_preds), np.array(split_labels)
        split_ap = average_precision_score(sl, sp) if len(np.unique(sl)) > 1 else -1
        split_acc = accuracy_score(sl, (sp > 0.5).astype(int))
        results[split_name] = {"ap": split_ap, "acc": split_acc, "n": len(split_preds), "per_generator": per_gen}
        print(f"  {split_name.upper()}: Acc={split_acc*100:.1f}%, AP={split_ap:.4f}")

    if results:
        avg_acc = np.mean([r["acc"] for r in results.values()])
        avg_ap = np.mean([r["ap"] for r in results.values() if r["ap"] >= 0])
        results["average"] = {"ap": avg_ap, "acc": avg_acc}
        print(f"\n  AVERAGE: Acc={avg_acc*100:.1f}%, AP={avg_ap:.4f}")
    return results


def train():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Available GPUs: {n_gpus}")

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Build model
    print("Building XGenDet model...")
    model = XGenDet(
        clip_model_name=args.clip_model,
        num_prompt_tokens=args.num_prompt_tokens,
        num_prototypes=args.num_prototypes,
        proto_dim=args.proto_dim,
        shuffle_patch_size=args.shuffle_patch_size,
    ).to(device)
    print(f"Trainable params: {model.count_trainable_params()['total']:,}")

    # Load pretrained
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print("  Pretrained weights loaded")

    # Multi-GPU
    raw_model = model
    use_multigpu = n_gpus > 1
    if use_multigpu:
        print(f"Using DataParallel on {n_gpus} GPUs (heatmap disabled during training)")
        model = nn.DataParallel(model)
        args.batch_size = args.batch_size * n_gpus
        print(f"  Effective batch size: {args.batch_size}")

    # Data — initial loader (OHEM will rebuild each epoch after warmup)
    print("Loading data...")
    train_dataset = HydraFakeDataset(
        json_path=args.train_json,
        data_root=args.data_root,
        is_train=True,
        crop_size=args.crop_size,
        jpeg_prob=args.jpeg_prob,
        blur_prob=args.blur_prob,
    )
    val_loader = create_hydrafake_dataloader(
        json_path=args.val_json,
        data_root=args.data_root,
        is_train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
    )

    # OHEM sampler
    ohem = OHEMSampler(
        dataset_size=len(train_dataset),
        hard_ratio=args.ohem_hard_ratio,
        warmup_epochs=args.ohem_warmup,
    ) if args.ohem else None

    # Loss with label smoothing
    criterion = XGenDetLoss(
        w_family=args.w_family,
        w_proto_div=args.w_proto_div,
        w_proto_compact=args.w_proto_compact,
        w_heatmap=args.w_heatmap,
        w_attr=args.w_attr,
        w_calib=args.w_calib,
    )
    ls_bce = LabelSmoothingBCELoss(smoothing=args.label_smoothing)

    # Optimizer
    param_groups = raw_model.get_trainable_params()
    base_lr, ln_lr = float(args.lr), float(args.lr_ln)
    optimizer_params = []
    for group in param_groups:
        lr_scale = float(group.get("lr_scale", 1.0))
        optimizer_params.append({
            "params": group["params"],
            "lr": base_lr * lr_scale if "layer_norm" not in group.get("name", "") else ln_lr,
            "name": group.get("name", "unnamed"),
        })
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=args.weight_decay)

    # Build initial train loader
    labels_list = [d[1] for d in train_dataset.data]

    def build_train_loader(epoch):
        if ohem:
            sampler = ohem.get_sampler(epoch, labels_list)
        else:
            class_counts = [labels_list.count(0), labels_list.count(1)]
            weights = [1.0 / class_counts[l] for l in labels_list]
            sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
        return DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )

    total_steps = (len(train_dataset) // args.batch_size) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    # Resume
    start_epoch = 1
    best_val_ap = 0.0
    patience_counter = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt["model_state_dict"].items()}
        raw_model.load_state_dict(sd)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_ap = ckpt.get("val_ap", 0.0)
        global_step = ckpt.get("global_step", 0)

    # Training
    print(f"\n{'='*60}")
    print(f"XGenDet v2 on HydraFake — OHEM={args.ohem}, LabelSmooth={args.label_smoothing}")
    print(f"  Exp: {args.exp_name}, Epochs: {start_epoch}-{args.epochs}")
    print(f"  Batch: {args.batch_size}, LR: {args.lr}, Seed: {args.seed}")
    print(f"  GPUs: {n_gpus}, Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loader = build_train_loader(epoch)
        epoch_losses = defaultdict(float)
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, (imgs, labels, families) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            families = families.to(device)

            outputs = model(imgs, return_heatmap=not use_multigpu)

            # Standard losses
            losses = criterion(outputs, labels, families, prototype_module=raw_model.prototype_module)

            # Replace cls loss with label-smoothed version
            losses["cls_smooth"] = ls_bce(outputs["binary_logit"], labels)
            losses["total"] = (
                losses["cls_smooth"]
                + criterion.w_family * losses["family"]
                + criterion.w_proto_div * losses["proto_div"]
                + criterion.w_proto_compact * losses["proto_compact"]
                + criterion.w_heatmap * losses["heatmap"]
                + criterion.w_attr * losses["attr"]
                + criterion.w_calib * losses["calib"]
            )

            loss = losses["total"] / args.grad_accum
            loss.backward()

            # OHEM: track per-sample losses
            if ohem:
                with torch.no_grad():
                    per_sample = F.binary_cross_entropy_with_logits(
                        outputs["binary_logit"].squeeze(-1), labels.float(), reduction='none'
                    ).cpu().numpy()
                    # Use batch indices as proxy (sequential within epoch)
                    start_idx = batch_idx * args.batch_size
                    indices = range(start_idx, min(start_idx + len(per_sample), len(train_dataset)))
                    ohem.update(list(indices), per_sample[:len(list(indices))])

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            for k in losses:
                epoch_losses[k] += losses[k].item()
            num_batches += 1

            if (batch_idx + 1) % args.log_freq == 0:
                avg_loss = epoch_losses["total"] / num_batches
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] loss={avg_loss:.4f} lr={lr:.2e}")

        epoch_time = time.time() - epoch_start
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.0f}s) Loss: {epoch_losses['total']:.4f}")

        # Validate
        print("Validating...")
        val_metrics = validate(raw_model if use_multigpu else model, val_loader, device)
        val_ap, val_acc = val_metrics["ap"], val_metrics["acc"]
        print(f"  Val: AP={val_ap:.4f}, Acc={val_acc:.4f}")

        # Save
        if epoch % args.save_freq == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "val_ap": val_ap, "val_acc": val_acc,
                "args": vars(args),
            }, os.path.join(exp_dir, f"epoch_{epoch}.pth"))

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "val_ap": val_ap, "val_acc": val_acc,
                "args": vars(args),
            }, os.path.join(exp_dir, "best_model.pth"))
            print(f"  *** New best! AP: {best_val_ap:.4f}, Acc: {val_acc:.4f} ***")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # ─── AUTO TEST EVAL ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training complete. Running test evaluation...")
    print(f"{'='*60}")

    best_ckpt = os.path.join(exp_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model (epoch {ckpt.get('epoch','?')}, AP={ckpt.get('val_ap',0):.4f})")

    raw_model.eval()
    test_results = evaluate_test_splits(raw_model, args, device)

    results_path = os.path.join(exp_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nTest results saved to: {results_path}")
    print(f"Best Val AP: {best_val_ap:.4f}")


if __name__ == "__main__":
    train()
