"""
XGenDet Stage 1 Training Script.

Trains the detection backbone (CLIP + forgery prompts + prototypes + heatmap + classifier)
on multi-generator real/fake image data.
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
from data.dataset import create_dataloader
from training.losses import XGenDetLoss


def parse_args():
    parser = argparse.ArgumentParser(description="XGenDet Stage 1 Training")

    # Config file (optional, overrides defaults; CLI args override config)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    # Model
    parser.add_argument("--clip_model", type=str, default=None)
    parser.add_argument("--num_prompt_tokens", type=int, default=None)
    parser.add_argument("--num_prototypes", type=int, default=None)
    parser.add_argument("--proto_dim", type=int, default=None)
    parser.add_argument("--shuffle_patch_size", type=int, default=None)

    # Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_ln", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)

    # Data
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--crop_size", type=int, default=None)

    # Loss weights
    parser.add_argument("--w_family", type=float, default=None)
    parser.add_argument("--w_proto_div", type=float, default=None)
    parser.add_argument("--w_proto_compact", type=float, default=None)
    parser.add_argument("--w_heatmap", type=float, default=None)
    parser.add_argument("--w_attr", type=float, default=None)
    parser.add_argument("--w_calib", type=float, default=None)

    # Logging
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--log_freq", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # Early stopping
    parser.add_argument("--patience", type=int, default=None)

    # Resume from checkpoint
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pth to resume from")

    args = parser.parse_args()

    # Hardcoded defaults (used when neither config nor CLI provides a value)
    defaults = {
        "clip_model": "ViT-L/14",
        "num_prompt_tokens": 8,
        "num_prototypes": 128,
        "proto_dim": 128,
        "shuffle_patch_size": 32,
        "lr": 1e-4,
        "lr_ln": 1e-6,
        "weight_decay": 0.01,
        "batch_size": 64,
        "epochs": 15,
        "warmup_steps": 500,
        "grad_accum": 1,
        "data_root": "/home/sachin.chaudhary/GTA",
        "max_samples": None,
        "num_workers": 8,
        "crop_size": 224,
        "w_family": 0.5,
        "w_proto_div": 0.3,
        "w_proto_compact": 0.2,
        "w_heatmap": 0.3,
        "w_attr": 0.1,
        "w_calib": 0.2,
        "jpeg_prob": 0.5,
        "blur_prob": 0.5,
        "output_dir": "./checkpoints",
        "exp_name": "xgendet_stage1",
        "log_freq": 100,
        "save_freq": 1,
        "seed": 42,
        "patience": 3,
    }

    # Load YAML config if provided
    yaml_values = {}
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

        # Flatten nested YAML structure into flat dict
        yaml_map = {
            "model": {
                "clip_model": "clip_model",
                "num_prompt_tokens": "num_prompt_tokens",
                "num_prototypes": "num_prototypes",
                "proto_dim": "proto_dim",
                "shuffle_patch_size": "shuffle_patch_size",
            },
            "training": {
                "lr": "lr",
                "lr_ln": "lr_ln",
                "weight_decay": "weight_decay",
                "batch_size": "batch_size",
                "epochs": "epochs",
                "warmup_steps": "warmup_steps",
                "grad_accum": "grad_accum",
                "seed": "seed",
                "patience": "patience",
            },
            "loss": {
                "w_family": "w_family",
                "w_proto_div": "w_proto_div",
                "w_proto_compact": "w_proto_compact",
                "w_heatmap": "w_heatmap",
                "w_attr": "w_attr",
                "w_calib": "w_calib",
            },
            "data": {
                "data_root": "data_root",
                "crop_size": "crop_size",
                "num_workers": "num_workers",
                "max_samples_per_class": "max_samples",
                "jpeg_prob": "jpeg_prob",
                "blur_prob": "blur_prob",
            },
            "logging": {
                "output_dir": "output_dir",
                "exp_name": "exp_name",
                "log_freq": "log_freq",
                "save_freq": "save_freq",
            },
        }

        for section, mapping in yaml_map.items():
            if section in cfg and isinstance(cfg[section], dict):
                for yaml_key, arg_key in mapping.items():
                    if yaml_key in cfg[section] and cfg[section][yaml_key] is not None:
                        yaml_values[arg_key] = cfg[section][yaml_key]

        # Also store list-type config values that argparse doesn't handle
        if "data" in cfg:
            if "id_generators" in cfg["data"]:
                yaml_values["id_generators"] = cfg["data"]["id_generators"]
            if "ood_generators" in cfg["data"]:
                yaml_values["ood_generators"] = cfg["data"]["ood_generators"]

    # Merge: CLI args > YAML config > hardcoded defaults
    for key, default_val in defaults.items():
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            # CLI explicitly set -- keep it
            pass
        elif key in yaml_values:
            val = yaml_values[key]
            # Coerce type to match the default (YAML may parse e.g. 1e-4 as str)
            if default_val is not None and val is not None:
                try:
                    val = type(default_val)(val)
                except (ValueError, TypeError):
                    pass
            setattr(args, key, val)
        else:
            setattr(args, key, default_val)

    # Store list configs from YAML (not available via CLI)
    if "id_generators" in yaml_values:
        args.id_generators = yaml_values["id_generators"]
    else:
        args.id_generators = None  # Will use hardcoded list in setup_data

    if "ood_generators" in yaml_values:
        args.ood_generators = yaml_values["ood_generators"]
    else:
        args.ood_generators = None  # Will use hardcoded list in setup_data

    return args


def setup_data(args):
    """Set up training and validation dataloaders."""
    data_root = args.data_root

    # In-domain generators for training (from config or hardcoded default)
    if getattr(args, "id_generators", None):
        id_generators = args.id_generators
    else:
        id_generators = ["ADM", "BigGAN", "glide", "LDM", "Midjourney", "progan", "VQDM", "wukong"]

    train_real_folders = [os.path.join(data_root, "final_GENERATORS", g, "real") for g in id_generators]
    train_fake_folders = [os.path.join(data_root, "final_GENERATORS", g, "fake") for g in id_generators]

    # Filter to existing directories and report missing ones
    for g, f in zip(id_generators, train_real_folders):
        if not os.path.isdir(f):
            print(f"WARNING: Training real folder not found: {f}")
    for g, f in zip(id_generators, train_fake_folders):
        if not os.path.isdir(f):
            print(f"WARNING: Training fake folder not found: {f}")

    train_real_folders = [f for f in train_real_folders if os.path.isdir(f)]
    train_fake_folders = [f for f in train_fake_folders if os.path.isdir(f)]

    print(f"Training real folders: {len(train_real_folders)}")
    print(f"Training fake folders: {len(train_fake_folders)}")

    if len(train_real_folders) == 0 or len(train_fake_folders) == 0:
        raise RuntimeError(
            f"No training data found! Check data_root={data_root} and "
            f"that final_GENERATORS/<gen>/real and final_GENERATORS/<gen>/fake exist."
        )

    train_loader = create_dataloader(
        real_folders=train_real_folders,
        fake_folders=train_fake_folders,
        is_train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples_per_class=args.max_samples,
        generator_names=id_generators,
        crop_size=args.crop_size,
        jpeg_prob=args.jpeg_prob,
        blur_prob=args.blur_prob,
    )

    # OOD generators for validation (from config or hardcoded default)
    if getattr(args, "ood_generators", None):
        ood_generators = args.ood_generators
    else:
        ood_generators = [
            "crn", "cyclegan", "dalle", "deepfake", "gaugan", "imle",
            "san", "seeingdark", "stargan", "stylegan", "stylegan2", "whichfaceisreal"
        ]

    val_loaders = {}
    for gen in ood_generators:
        real_folder = os.path.join(data_root, "OOD_GENERATORS", gen, "0_real")
        fake_folder = os.path.join(data_root, "OOD_GENERATORS", gen, "1_fake")

        if os.path.isdir(real_folder) and os.path.isdir(fake_folder):
            val_loaders[gen] = create_dataloader(
                real_folders=[real_folder],
                fake_folders=[fake_folder],
                is_train=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                generator_names=[gen],
                crop_size=args.crop_size,
            )
        else:
            print(f"WARNING: OOD generator '{gen}' not found at expected path, skipping.")

    print(f"Validation generators: {list(val_loaders.keys())}")
    return train_loader, val_loaders


def validate(model, val_loaders, device):
    """Validate on all OOD generators."""
    model.eval()
    results = {}

    with torch.no_grad():
        for gen_name, loader in val_loaders.items():
            all_preds = []
            all_labels = []

            for imgs, labels, families in loader:
                imgs = imgs.to(device)
                outputs = model(imgs, return_heatmap=False)
                probs = outputs["confidence"].squeeze(-1).cpu().numpy()
                all_preds.extend(probs.tolist())
                all_labels.extend(labels.numpy().tolist())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            # Compute metrics
            ap = average_precision_score(all_labels, all_preds)
            acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

            results[gen_name] = {"ap": ap, "acc": acc}

    # Compute averages
    if results:
        avg_ap = np.mean([v["ap"] for v in results.values()])
        avg_acc = np.mean([v["acc"] for v in results.values()])
    else:
        avg_ap = 0.0
        avg_acc = 0.0
    results["average"] = {"ap": avg_ap, "acc": avg_acc}

    model.train()
    return results


def train():
    args = parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # TensorBoard (optional)
    if _HAS_TENSORBOARD:
        writer = SummaryWriter(os.path.join(exp_dir, "logs"))
    else:
        writer = None
        print("WARNING: tensorboard not installed. Logging to stdout only.")

    # Model
    print("Building XGenDet model...")
    model = XGenDet(
        clip_model_name=args.clip_model,
        num_prompt_tokens=args.num_prompt_tokens,
        num_prototypes=args.num_prototypes,
        proto_dim=args.proto_dim,
        shuffle_patch_size=args.shuffle_patch_size,
    ).to(device)

    # Count parameters
    param_counts = model.count_trainable_params()
    print(f"Trainable parameters: {param_counts}")
    print(f"Total trainable: {param_counts['total']:,}")

    # Data
    print("Setting up data...")
    train_loader, val_loaders = setup_data(args)

    # Loss
    criterion = XGenDetLoss(
        w_family=args.w_family,
        w_proto_div=args.w_proto_div,
        w_proto_compact=args.w_proto_compact,
        w_heatmap=args.w_heatmap,
        w_attr=args.w_attr,
        w_calib=args.w_calib,
    )

    # Optimizer with parameter groups
    param_groups = model.get_trainable_params()
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

    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )

    # Resume from checkpoint
    start_epoch = 1
    best_val_ap = 0.0
    patience_counter = 0
    global_step = 0

    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_ap = ckpt.get("val_results", {}).get("average", {}).get("ap", 0.0)
        global_step = ckpt.get("global_step", 0)
        print(f"  Resumed at epoch {start_epoch}, best AP so far: {best_val_ap:.4f}")

    # Training loop
    print(f"\nStarting training for epochs {start_epoch}-{args.epochs}...")
    print(f"Total steps: {total_steps}")
    print(f"Device: {device}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses = {k: 0.0 for k in ["total", "cls", "family", "proto_div", "proto_compact", "heatmap", "attr", "calib"]}
        num_batches = 0

        for batch_idx, (imgs, labels, families) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            families = families.to(device)

            # Forward
            outputs = model(imgs, return_heatmap=True)

            # Loss
            losses = criterion(
                outputs, labels, families,
                prototype_module=model.prototype_module,
            )

            # Backward
            loss = losses["total"] / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Accumulate losses
            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k] += losses[k].item()
            num_batches += 1

            # Logging
            if (batch_idx + 1) % args.log_freq == 0:
                avg_loss = epoch_losses["total"] / num_batches
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={avg_loss:.4f} lr={lr:.2e}")

                if writer is not None:
                    for k, v in losses.items():
                        writer.add_scalar(f"train/{k}", v.item(), global_step)
                    writer.add_scalar("train/lr", lr, global_step)

        # Epoch summary
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)
        print(f"\nEpoch {epoch}/{args.epochs} - Avg Loss: {epoch_losses['total']:.4f}")
        print(f"  cls={epoch_losses['cls']:.4f} family={epoch_losses['family']:.4f} "
              f"proto_div={epoch_losses['proto_div']:.4f} heatmap={epoch_losses['heatmap']:.4f}")

        # Validation
        print(f"\nValidating on OOD generators...")
        val_results = validate(model, val_loaders, device)
        avg_ap = val_results["average"]["ap"]
        avg_acc = val_results["average"]["acc"]

        print(f"  OOD Average - AP: {avg_ap:.4f}, Acc: {avg_acc:.4f}")
        for gen, metrics in val_results.items():
            if gen != "average":
                print(f"    {gen}: AP={metrics['ap']:.4f}, Acc={metrics['acc']:.4f}")

        if writer is not None:
            writer.add_scalar("val/avg_ap", avg_ap, epoch)
            writer.add_scalar("val/avg_acc", avg_acc, epoch)

        # Save checkpoint
        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(exp_dir, f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "val_results": val_results,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Best model
        if avg_ap > best_val_ap:
            best_val_ap = avg_ap
            patience_counter = 0
            best_path = os.path.join(exp_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_results": val_results,
                "args": vars(args),
            }, best_path)
            print(f"  New best model! AP: {best_val_ap:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if writer is not None:
        writer.close()
    print(f"\nTraining complete. Best OOD AP: {best_val_ap:.4f}")
    print(f"Checkpoints saved to: {exp_dir}")


if __name__ == "__main__":
    train()
