"""
train_physfreq.py — Train PhysFreqNet on HydraFake.

Configurable branches via --branches (e.g., "fft", "fft,retinex", "fft,retinex,ca").
Ablation-friendly: each branch can be turned on/off independently.

Example:
    python training/train_physfreq.py --branches fft,retinex,ca --bf16

Loss   : BCEWithLogitsLoss (no label smoothing)
Metrics: accuracy @ 0.5, Average Precision (sklearn), F1
Sched  : CosineAnnealingLR(T_max = steps_per_epoch * epochs, eta_min=1e-7)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.physfreq_net import PhysFreqNet
from data.hydrafake_dataset import HydraFakeDataset


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Train PhysFreqNet on HydraFake")
    p.add_argument("--branches", type=str, default="fft,retinex,ca",
                   help="comma-separated branch names: fft / retinex / ca")
    p.add_argument("--train_json", type=str,
                   default="/home/sachin.chaudhary/hydrafake/jsons/train/all.json")
    p.add_argument("--val_json", type=str,
                   default="/home/sachin.chaudhary/hydrafake/jsons/val/all.json")
    p.add_argument("--data_root", type=str, default="/home/sachin.chaudhary")
    p.add_argument("--output_dir", type=str, default=None,
                   help="default: checkpoints/physfreq_{branches}")
    p.add_argument("--log_file", type=str, default=None,
                   help="default: /home/sachin.chaudhary/xgendet/logs/physfreq_{branches}_live.log")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--bf16", action="store_true",
                   help="cast model + inputs to bfloat16 (CUDA only)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Parse branches -> tuple
    args.branches = tuple(b.strip() for b in args.branches.split(",") if b.strip())
    assert len(args.branches) > 0, "Need at least one branch in --branches"
    for b in args.branches:
        assert b in ("fft", "retinex", "ca"), f"Unknown branch: {b}"

    branches_tag = "_".join(args.branches)
    if args.output_dir is None:
        args.output_dir = f"checkpoints/physfreq_{branches_tag}"
    if args.log_file is None:
        args.log_file = f"/home/sachin.chaudhary/xgendet/logs/physfreq_{branches_tag}_live.log"
    args.branches_tag = branches_tag
    return args


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(args):
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("physfreq")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- #
# Build model / data / optim
# --------------------------------------------------------------------------- #
def build_model(args, device, logger):
    model = PhysFreqNet(branches=args.branches)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        logger.info(f"[{_ts()}] [branches={args.branches_tag}] "
                    f"Using DataParallel over {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    if args.bf16 and device.type == "cuda":
        model = model.to(torch.bfloat16)
        logger.info(f"[{_ts()}] [branches={args.branches_tag}] Cast model to bfloat16")
    return model


def build_dataloaders(args, logger):
    train_ds = HydraFakeDataset(
        json_path=args.train_json,
        data_root=args.data_root,
        is_train=True,
        crop_size=224,
    )
    val_ds = HydraFakeDataset(
        json_path=args.val_json,
        data_root=args.data_root,
        is_train=False,
        crop_size=224,
    )
    logger.info(f"[{_ts()}] [branches={args.branches_tag}] "
                f"Dataset sizes: train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )
    return train_loader, val_loader


def build_optimizer(model, args, logger):
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    param_groups = base_model.get_trainable_params()

    logger.info(f"[{_ts()}] [branches={args.branches_tag}] Optimizer param groups:")
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        logger.info(f"  {g['name']}: {n:,} params @ lr={g['lr']}")

    optimizer = AdamW(param_groups, weight_decay=0.01)
    # Placeholder; real T_max set in train() once steps_per_epoch is known.
    scheduler = CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-7)
    return optimizer, scheduler


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, val_loader, device, args):
    model.eval()
    all_probs = []
    all_labels = []
    dtype_in = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    for batch in val_loader:
        imgs, labels, _ = batch
        imgs = imgs.to(device, dtype=dtype_in, non_blocking=True)
        labels = labels.to(device, dtype=torch.float32).unsqueeze(1)  # [B,1]

        out = model(imgs)
        logit = out["logit"]
        if logit.dim() == 2:
            logit = logit.squeeze(1)
        logit = logit.float()
        prob = torch.sigmoid(logit)

        all_probs.append(prob.detach().cpu().numpy())
        all_labels.append(labels.squeeze(1).detach().cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(np.int64)
    preds = (probs >= 0.5).astype(np.int64)

    acc = float((preds == labels).mean())
    try:
        ap = float(average_precision_score(labels, probs))
    except Exception:
        ap = 0.0
    try:
        f1 = float(f1_score(labels, preds, zero_division=0))
    except Exception:
        f1 = 0.0
    return {"acc": acc, "ap": ap, "f1": f1}


# --------------------------------------------------------------------------- #
# Train loop
# --------------------------------------------------------------------------- #
def train(model, train_loader, val_loader, optimizer, scheduler, args, device, logger):
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    # Rebuild scheduler now that we know total_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    base_model = model.module if isinstance(model, nn.DataParallel) else model
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best_model.pth"

    dtype_in = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    best_ap = -1.0
    patience_left = args.patience
    tag = args.branches_tag

    logger.info(f"[{_ts()}] [branches={tag}] "
                f"Begin training: epochs={args.epochs} steps/epoch={steps_per_epoch} "
                f"batch_size={args.batch_size} bf16={args.bf16}")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0
        t_epoch = time.time()

        for step, batch in enumerate(train_loader, start=1):
            imgs, labels, _ = batch
            imgs = imgs.to(device, dtype=dtype_in, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32).unsqueeze(1)  # [B,1]

            out = model(imgs)
            logit = out["logit"]
            if logit.dim() == 2:
                logit = logit.squeeze(1)
            logit = logit.float()                # stabilize BCE under bf16
            labels_flat = labels.squeeze(1)

            loss = F.binary_cross_entropy_with_logits(logit, labels_flat)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_n += bs
            global_step += 1

            if step % args.log_steps == 0 or step == steps_per_epoch:
                avg_loss = running_loss / max(running_n, 1)
                cur_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[{_ts()}] [branches={tag}] "
                    f"Epoch {epoch}/{args.epochs} Step {step}/{steps_per_epoch} "
                    f"| Loss: {avg_loss:.3f} | LR: {cur_lr:.2e}"
                )

        epoch_time = time.time() - t_epoch
        logger.info(f"[{_ts()}] [branches={tag}] "
                    f"Epoch {epoch}/{args.epochs} done in {epoch_time:.1f}s "
                    f"| avg train loss: {running_loss / max(running_n,1):.4f}")

        # Validation
        metrics = validate(model, val_loader, device, args)
        ap = metrics["ap"]
        is_best = ap > best_ap
        star = " *" if is_best else ""
        logger.info(
            f"[{_ts()}] [branches={tag}] "
            f"Epoch {epoch}/{args.epochs} Val | AP: {ap:.4f} | "
            f"Acc: {metrics['acc']*100:.1f}% | F1: {metrics['f1']:.3f} | "
            f"Best: {max(best_ap, ap):.4f}{star}"
        )

        if is_best:
            best_ap = ap
            patience_left = args.patience
            base_model.save_checkpoint(str(best_ckpt))
        else:
            patience_left -= 1
            logger.info(f"[{_ts()}] [branches={tag}] "
                        f"No improvement. Patience left: {patience_left}/{args.patience}")
            if patience_left <= 0:
                logger.info(f"[{_ts()}] [branches={tag}] Early stopping triggered.")
                break

    logger.info(f"[{_ts()}] [branches={tag}] "
                f"Training complete. Best val AP: {best_ap:.4f} | ckpt: {best_ckpt}")
    return best_ap


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger = setup_logging(args)
    logger.info(f"[{_ts()}] [branches={args.branches_tag}] Args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{_ts()}] [branches={args.branches_tag}] Device: {device} "
                f"(cuda_count={torch.cuda.device_count()})")

    model = build_model(args, device, logger)
    train_loader, val_loader = build_dataloaders(args, logger)
    optimizer, scheduler = build_optimizer(model, args, logger)

    best_ap = train(model, train_loader, val_loader, optimizer, scheduler,
                    args, device, logger)
    logger.info(f"[{_ts()}] [branches={args.branches_tag}] FINAL best AP = {best_ap:.4f}")


if __name__ == "__main__":
    main()
