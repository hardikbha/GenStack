"""
XGenDet Training on HydraFake Dataset.

Trains the 6-bank prototype model on HydraFake face forensics data.
Supports both from-scratch training and fine-tuning from GenImage checkpoint.
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path

import yaml
import torch
import torch.nn as nn
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False
from sklearn.metrics import average_precision_score, accuracy_score
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xgendet import XGenDet
from data.hydrafake_dataset import (
    create_hydrafake_dataloader,
    HydraFakeTestDataset,
)
from training.losses import XGenDetLoss


def parse_args():
    parser = argparse.ArgumentParser(description="XGenDet HydraFake Training")
    parser.add_argument("--config", type=str, default=None)

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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)

    # Data
    parser.add_argument("--train_json", type=str,
                        default="/home/sachin.chaudhary/hydrafake/jsons/train/all.json")
    parser.add_argument("--val_json", type=str,
                        default="/home/sachin.chaudhary/hydrafake/jsons/val/all.json")
    parser.add_argument("--data_root", type=str, default="/home/sachin.chaudhary")
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
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
    parser.add_argument("--exp_name", type=str, default="hydrafake_scratch")
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=1)

    # Resume / fine-tune
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume training from")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Pretrained checkpoint to fine-tune from (loads weights only, resets optimizer)")

    # Test splits for evaluation
    parser.add_argument("--test_dir", type=str,
                        default="/home/sachin.chaudhary/hydrafake/jsons/test")

    args = parser.parse_args()
    return args


def setup_data(args):
    """Set up HydraFake train and validation dataloaders."""
    print(f"Loading training data from: {args.train_json}")
    train_loader = create_hydrafake_dataloader(
        json_path=args.train_json,
        data_root=args.data_root,
        is_train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples_per_class=args.max_samples,
        crop_size=args.crop_size,
        jpeg_prob=args.jpeg_prob,
        blur_prob=args.blur_prob,
    )

    print(f"Loading validation data from: {args.val_json}")
    val_loader = create_hydrafake_dataloader(
        json_path=args.val_json,
        data_root=args.data_root,
        is_train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
    )

    return train_loader, val_loader


def validate(model, val_loader, device):
    """Validate on HydraFake val set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, families in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs, return_heatmap=False)
            probs = outputs["confidence"].squeeze(-1).cpu().numpy()
            all_preds.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    ap = average_precision_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

    model.train()
    return {"ap": ap, "acc": acc}


def evaluate_test_splits(model, args, device):
    """Evaluate on all 4 HydraFake test splits."""
    from torch.utils.data import DataLoader

    test_dir = args.test_dir
    splits = {"id": [], "cm": [], "cf": [], "cd": []}

    results = {}
    for split_name in splits:
        split_dir = os.path.join(test_dir, split_name)
        if not os.path.isdir(split_dir):
            print(f"  Test split '{split_name}' not found, skipping")
            continue

        json_files = [f for f in os.listdir(split_dir) if f.endswith(".json")]
        split_preds = []
        split_labels = []
        per_gen = {}

        for jf in sorted(json_files):
            gen_name = jf.replace(".json", "")
            json_path = os.path.join(split_dir, jf)

            dataset = HydraFakeTestDataset(
                json_path=json_path,
                data_root=args.data_root,
                image_root="/home/sachin.chaudhary/hydrafake/test",
                crop_size=args.crop_size,
            )

            if len(dataset) == 0:
                print(f"    {gen_name}: 0 images found, skipping")
                continue

            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )

            gen_preds = []
            gen_labels = []
            with torch.no_grad():
                for imgs, labels in loader:
                    imgs = imgs.to(device)
                    outputs = model(imgs, return_heatmap=False)
                    probs = outputs["confidence"].squeeze(-1).cpu().numpy()
                    gen_preds.extend(probs.tolist())
                    gen_labels.extend(labels.numpy().tolist())

            gen_preds_np = np.array(gen_preds)
            gen_labels_np = np.array(gen_labels)

            if len(np.unique(gen_labels_np)) > 1:
                gen_ap = average_precision_score(gen_labels_np, gen_preds_np)
            else:
                gen_ap = -1.0
            gen_acc = accuracy_score(gen_labels_np, (gen_preds_np > 0.5).astype(int))

            per_gen[gen_name] = {"ap": gen_ap, "acc": gen_acc, "n": len(dataset)}
            split_preds.extend(gen_preds)
            split_labels.extend(gen_labels)

        if split_preds:
            sp = np.array(split_preds)
            sl = np.array(split_labels)
            split_ap = average_precision_score(sl, sp) if len(np.unique(sl)) > 1 else -1.0
            split_acc = accuracy_score(sl, (sp > 0.5).astype(int))
        else:
            split_ap = 0.0
            split_acc = 0.0

        results[split_name] = {
            "ap": split_ap,
            "acc": split_acc,
            "n": len(split_preds),
            "per_generator": per_gen,
        }

        print(f"  {split_name.upper()}: AP={split_ap:.4f}, Acc={split_acc:.4f} ({len(split_preds)} samples)")
        for gen, gm in per_gen.items():
            print(f"    {gen}: AP={gm['ap']:.4f}, Acc={gm['acc']:.4f} (n={gm['n']})")

    # Overall average
    if results:
        avg_acc = np.mean([r["acc"] for r in results.values()])
        avg_ap = np.mean([r["ap"] for r in results.values() if r["ap"] >= 0])
        results["average"] = {"ap": avg_ap, "acc": avg_acc}
        print(f"\n  AVERAGE: AP={avg_ap:.4f}, Acc={avg_acc:.4f}")

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

    if _HAS_TENSORBOARD:
        writer = SummaryWriter(os.path.join(exp_dir, "logs"))
    else:
        writer = None

    # Build model
    print("Building XGenDet model...")
    model = XGenDet(
        clip_model_name=args.clip_model,
        num_prompt_tokens=args.num_prompt_tokens,
        num_prototypes=args.num_prototypes,
        proto_dim=args.proto_dim,
        shuffle_patch_size=args.shuffle_patch_size,
    ).to(device)

    param_counts = model.count_trainable_params()
    print(f"Trainable parameters: {param_counts['total']:,}")

    # Load pretrained weights (fine-tune mode)
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained weights from: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Strip "module." prefix if saved from DataParallel
        state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}...")
        print("  Pretrained weights loaded (optimizer reset)")

    # Multi-GPU with DataParallel
    raw_model = model  # keep reference to unwrapped model for param groups / saving
    use_multigpu = n_gpus > 1
    if use_multigpu:
        print(f"Using DataParallel on {n_gpus} GPUs")
        print(f"  NOTE: Heatmap disabled during training (hook incompatibility with DataParallel)")
        print(f"  Heatmap loss will be skipped; heatmaps computed at eval time (single GPU)")
        model = nn.DataParallel(model)
        args.batch_size = args.batch_size * n_gpus
        print(f"  Effective batch size: {args.batch_size}")

    # Data
    print("Setting up HydraFake data...")
    train_loader, val_loader = setup_data(args)

    # Loss
    criterion = XGenDetLoss(
        w_family=args.w_family,
        w_proto_div=args.w_proto_div,
        w_proto_compact=args.w_proto_compact,
        w_heatmap=args.w_heatmap,
        w_attr=args.w_attr,
        w_calib=args.w_calib,
    )

    # Optimizer (always use raw_model for param groups)
    param_groups = raw_model.get_trainable_params()
    base_lr = float(args.lr)
    ln_lr = float(args.lr_ln)
    optimizer_params = []
    for group in param_groups:
        lr_scale = float(group.get("lr_scale", 1.0))
        optimizer_params.append({
            "params": group["params"],
            "lr": base_lr * lr_scale if "layer_norm" not in group.get("name", "") else ln_lr,
            "name": group.get("name", "unnamed"),
        })

    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )

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
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_ap = ckpt.get("val_ap", 0.0)
        global_step = ckpt.get("global_step", 0)
        print(f"  Resumed at epoch {start_epoch}, best AP: {best_val_ap:.4f}")

    # Training
    print(f"\n{'='*60}")
    print(f"Training XGenDet on HydraFake")
    print(f"  Experiment: {args.exp_name}")
    print(f"  Epochs: {start_epoch}-{args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}, LR_LN: {args.lr_ln}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses = {k: 0.0 for k in ["total", "cls", "family", "proto_div",
                                           "proto_compact", "heatmap", "attr", "calib"]}
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, (imgs, labels, families) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            families = families.to(device)

            outputs = model(imgs, return_heatmap=not use_multigpu)
            losses = criterion(
                outputs, labels, families,
                prototype_module=raw_model.prototype_module,
            )

            loss = losses["total"] / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k] += losses[k].item()
            num_batches += 1

            if (batch_idx + 1) % args.log_freq == 0:
                avg_loss = epoch_losses["total"] / num_batches
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={avg_loss:.4f} lr={lr:.2e}")

                if writer is not None:
                    for k, v in losses.items():
                        writer.add_scalar(f"train/{k}", v.item(), global_step)

        epoch_time = time.time() - epoch_start
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.0f}s) "
              f"Loss: {epoch_losses['total']:.4f}")
        print(f"  cls={epoch_losses['cls']:.4f} family={epoch_losses['family']:.4f} "
              f"heatmap={epoch_losses['heatmap']:.4f} attr={epoch_losses['attr']:.4f}")

        # Validation
        print("Validating...")
        val_metrics = validate(model, val_loader, device)
        val_ap = val_metrics["ap"]
        val_acc = val_metrics["acc"]
        print(f"  Val: AP={val_ap:.4f}, Acc={val_acc:.4f}")

        if writer is not None:
            writer.add_scalar("val/ap", val_ap, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)
            writer.add_scalar("train/epoch_loss", epoch_losses["total"], epoch)

        # Save checkpoint
        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(exp_dir, f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "val_ap": val_ap,
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

        # Best model
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            patience_counter = 0
            best_path = os.path.join(exp_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "val_ap": val_ap,
                "val_acc": val_acc,
                "args": vars(args),
            }, best_path)
            print(f"  *** New best model! AP: {best_val_ap:.4f} ***")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Final test evaluation
    print(f"\n{'='*60}")
    print("Running final evaluation on HydraFake test splits...")
    print(f"{'='*60}")

    # Load best model
    best_ckpt = os.path.join(exp_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt["model_state_dict"].items()}
        raw_model.load_state_dict(sd)
        print(f"Loaded best model (epoch {ckpt.get('epoch', '?')}, AP={ckpt.get('val_ap', 0):.4f})")

    raw_model.eval()
    test_results = evaluate_test_splits(raw_model, args, device)

    # Save test results
    results_path = os.path.join(exp_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nTest results saved to: {results_path}")

    if writer is not None:
        writer.close()

    print(f"\nTraining complete. Best Val AP: {best_val_ap:.4f}")
    print(f"Checkpoints: {exp_dir}")


if __name__ == "__main__":
    train()
