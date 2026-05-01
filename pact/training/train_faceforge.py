"""
FaceForge-Net Training Script.

Trains FaceForgeNet (8.6M params) on HydraFake dataset using DataParallel
for multi-GPU support (NOT DeepSpeed — model is small).

Loss: L = L_BCE + 0.5 * L_focal + 0.1 * L_region_div
  - L_BCE:       BCEWithLogitsLoss with label smoothing 0.1
  - L_focal:     Focal loss gamma=2 on logits
  - L_region_div: -std(region_weights) — penalizes uniform region attention

Optimizer: AdamW with differential LR via param groups:
  backbone_lora → 1e-5
  fusion        → 1e-4
  srm           → 5e-5

Scheduler: CosineAnnealingLR, T_max=total_steps, eta_min=1e-7

Usage (single GPU):
    python training/train_faceforge.py --train_json ... --val_json ...

Usage (multi-GPU via DataParallel, controlled by CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python training/train_faceforge.py ...
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.faceforge_net import FaceForgeNet
from data.faceforge_dataset import FaceForgeDataset


# ── Loss functions ────────────────────────────────────────────────────────────

def focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss to down-weight easy examples."""
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


def bce_with_label_smoothing(logits: torch.Tensor, labels: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """BCEWithLogitsLoss with label smoothing."""
    smooth_labels = labels * (1.0 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, smooth_labels)


def region_div_loss(region_weights: torch.Tensor) -> torch.Tensor:
    """
    Penalizes uniform region attention weights.
    region_weights: [B, 8]
    Returns: -std(region_weights) averaged over batch — we SUBTRACT std to
             encourage specialization (higher std = more specialized = lower loss).
    """
    # std per sample, then mean over batch
    std_per_sample = region_weights.std(dim=1)          # [B]
    return -std_per_sample.mean()


def compute_loss(outputs: dict, labels: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """
    Compute combined FaceForge loss.

    Args:
        outputs: dict with keys 'logit' [B], 'prob' [B], 'region_weights' [B,8]
        labels:  float tensor [B]

    Returns:
        (total_loss, component_dict)
    """
    logits = outputs['logit'].squeeze(1)  # [B,1] → [B]
    region_weights = outputs['region_weights']

    l_bce   = bce_with_label_smoothing(logits, labels, smoothing=0.1)
    l_focal = focal_loss(logits, labels, gamma=2.0)
    l_reg   = region_div_loss(region_weights)

    total = l_bce + 0.5 * l_focal + 0.1 * l_reg

    return total, {
        'loss':    total.item(),
        'bce':     l_bce.item(),
        'focal':   l_focal.item(),
        'reg_div': l_reg.item(),
    }


# ── DataLoader collate_fn ─────────────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    """
    Collate a list of FaceForgeDataset samples into a batch.

    Each sample is a dict with:
      'crops':      dict of {region_name: tensor}
      'full_image': tensor [C, H, W]
      'label':      int (0 or 1)
    """
    crops_batch = {
        k: torch.stack([b['crops'][k] for b in batch])
        for k in batch[0]['crops']
    }
    full_images = torch.stack([b['full_image'] for b in batch])
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.float32)
    return {'crops': crops_batch, 'full_image': full_images, 'label': labels}


# ── Logging ───────────────────────────────────────────────────────────────────

LOG_FILE = '/home/sachin.chaudhary/xgendet/logs/faceforge_live.log'


def setup_logging(args: argparse.Namespace) -> None:
    """Ensure log directory exists."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)


def log(msg: str) -> None:
    """Write timestamped message to both stdout and live log file."""
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


# ── Arg parsing ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train FaceForge-Net on HydraFake')

    # Data
    p.add_argument('--train_json',  required=True,
                   help='Path to train split JSON')
    p.add_argument('--val_json',    required=True,
                   help='Path to val split JSON')
    p.add_argument('--data_root',   default='/home/sachin.chaudhary',
                   help='Root directory for image paths')
    p.add_argument('--cache_dir',   default='/home/sachin.chaudhary/hydrafake/faceforge_crops',
                   help='Cache directory for pre-extracted region crops')

    # Output
    p.add_argument('--output_dir',  default='checkpoints/faceforge',
                   help='Where to save checkpoints')

    # Training hyperparams
    p.add_argument('--epochs',      type=int,   default=30)
    p.add_argument('--batch_size',  type=int,   default=16,
                   help='Per-GPU batch size')
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--grad_clip',   type=float, default=1.0)
    p.add_argument('--patience',    type=int,   default=10,
                   help='Early stopping patience on val AP')
    p.add_argument('--log_steps',   type=int,   default=50)
    p.add_argument('--save_steps',  type=int,   default=500)
    p.add_argument('--bf16',        action='store_true', default=False,
                   help='Use bfloat16 precision')

    # Model / LoRA
    p.add_argument('--lora_r',      type=int,   default=16)
    p.add_argument('--lora_alpha',  type=int,   default=32)
    p.add_argument('--lora_blocks', type=int,   default=8)

    return p.parse_args()


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(args: argparse.Namespace) -> nn.Module:
    """Instantiate FaceForgeNet and wrap in DataParallel if multiple GPUs."""
    model = FaceForgeNet(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_blocks=args.lora_blocks,
    )

    if args.bf16:
        model = model.to(torch.bfloat16)

    # Detect GPUs from CUDA_VISIBLE_DEVICES (set externally by PBS)
    n_gpus = torch.cuda.device_count()
    print(f'>>> CUDA device count: {n_gpus}')

    if n_gpus > 1:
        print(f'>>> Using DataParallel across {n_gpus} GPUs')
        model = nn.DataParallel(model)
    elif n_gpus == 1:
        print('>>> Using single GPU')

    if n_gpus >= 1:
        model = model.cuda()

    return model


# ── DataLoaders ───────────────────────────────────────────────────────────────

def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders."""
    train_dataset = FaceForgeDataset(
        json_path=args.train_json,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        split='train',
    )
    val_dataset = FaceForgeDataset(
        json_path=args.val_json,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        split='val',
    )

    # Pre-extract all region crops to cache before training begins
    print('>>> Pre-extracting train region crops to cache...')
    train_dataset.pre_extract_all(num_workers=args.num_workers)
    print('>>> Pre-extracting val region crops to cache...')
    val_dataset.pre_extract_all(num_workers=args.num_workers)
    print('>>> Pre-extraction complete.')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    print(f'>>> Train samples: {len(train_dataset)} | batches: {len(train_loader)}')
    print(f'>>> Val samples:   {len(val_dataset)} | batches: {len(val_loader)}')

    return train_loader, val_loader


# ── Optimizer & scheduler ─────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
) -> tuple[AdamW, CosineAnnealingLR]:
    """
    Build AdamW with differential LRs from FaceForgeNet.get_trainable_params().

    Expected groups returned by get_trainable_params():
      {'backbone_lora': [...], 'fusion': [...], 'srm': [...]}

    LR map:
      backbone_lora → 1e-5
      fusion        → 1e-4
      srm           → 5e-5
    """
    lr_map = {
        'backbone_lora': 1e-5,
        'fusion':        1e-4,
        'srm':           5e-5,
    }

    # Unwrap DataParallel if needed to access get_trainable_params
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    param_groups = base_model.get_trainable_params()

    for g in param_groups:
        name = g.get('name', '?')
        if name in lr_map:
            g['lr'] = lr_map[name]
        n_params = sum(p.numel() for p in g['params'])
        print(f'  Param group "{name}": {n_params:,} params @ lr={g["lr"]}')

    optimizer = AdamW(param_groups, weight_decay=0.01)

    # Total steps for cosine scheduler (computed after dataloaders are built,
    # but we calculate it here from train_loader length × epochs)
    # Note: train_loader is not passed here — total_steps is computed in train()
    # We return a placeholder scheduler that gets re-created in train() once
    # total_steps is known.  To avoid two-pass construction, we pass a sentinel.
    # CosineAnnealingLR requires T_max at construction; we set it to 1 and
    # re-build in train() once total_steps is known.
    scheduler = CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-7)

    return optimizer, scheduler


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    args: argparse.Namespace,
) -> dict:
    """
    Run validation and return metrics dict:
      {'ap': float, 'acc': float, 'f1': float}
    """
    model.eval()
    all_probs  = []
    all_preds  = []
    all_labels = []

    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32

    for batch in val_loader:
        crops      = {k: v.cuda() for k, v in batch['crops'].items()} if torch.cuda.is_available() else batch['crops']
        full_image = batch['full_image'].cuda() if torch.cuda.is_available() else batch['full_image']
        labels     = batch['label']

        if args.bf16:
            crops      = {k: v.to(torch.bfloat16) for k, v in crops.items()}
            full_image = full_image.to(torch.bfloat16)

        with torch.autocast('cuda', dtype=autocast_dtype, enabled=torch.cuda.is_available()):
            outputs = model(crops, full_image)

        probs  = outputs['prob'].float().cpu()
        preds  = (probs >= 0.5).long()

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.long().tolist())

    ap  = average_precision_score(all_labels, all_probs)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    model.train()
    return {'ap': ap, 'acc': acc, 'f1': f1}


# ── Checkpoint saving ─────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    best_ap: float,
    path: str,
) -> None:
    """Save trainable model params, optimizer state, epoch, and best AP."""
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    # Only save trainable parameters
    trainable_state = {
        k: v for k, v in base_model.state_dict().items()
        if any(p is base_model.state_dict()[k]
               for p in base_model.parameters()
               if p.requires_grad)
    }
    # Fallback: save full state_dict filtered by requires_grad
    trainable_keys = {
        name for name, param in base_model.named_parameters()
        if param.requires_grad
    }
    trainable_state = {
        k: v for k, v in base_model.state_dict().items()
        if k in trainable_keys
    }

    torch.save({
        'model_state_dict': trainable_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_ap': best_ap,
    }, path)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    args: argparse.Namespace,
) -> None:
    """Main training loop with early stopping on val AP."""

    total_steps = len(train_loader) * args.epochs

    # Re-build scheduler now that total_steps is known
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = param_group['lr']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7,
                                   last_epoch=-1)

    best_ap          = 0.0
    best_path        = os.path.join(args.output_dir, 'best_model.pth')
    patience_counter = 0
    global_step      = 0

    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32

    log(f'Starting training | epochs={args.epochs} | steps_per_epoch={len(train_loader)} | total_steps={total_steps}')

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Running accumulators for smooth loss reporting
        running_loss    = 0.0
        running_bce     = 0.0
        running_focal   = 0.0
        running_reg_div = 0.0
        steps_since_log = 0

        for step, batch in enumerate(train_loader, start=1):
            global_step += 1

            # Move to GPU
            if torch.cuda.is_available():
                crops      = {k: v.cuda() for k, v in batch['crops'].items()}
                full_image = batch['full_image'].cuda()
                labels     = batch['label'].cuda()
            else:
                crops      = batch['crops']
                full_image = batch['full_image']
                labels     = batch['label']

            if args.bf16:
                crops      = {k: v.to(torch.bfloat16) for k, v in crops.items()}
                full_image = full_image.to(torch.bfloat16)

            # Forward + loss
            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=autocast_dtype, enabled=torch.cuda.is_available()):
                outputs = model(crops, full_image)
                loss, components = compute_loss(outputs, labels)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            # Accumulate for logging
            running_loss    += components['loss']
            running_bce     += components['bce']
            running_focal   += components['focal']
            running_reg_div += components['reg_div']
            steps_since_log += 1

            # Periodic log
            if global_step % args.log_steps == 0:
                avg_loss    = running_loss    / steps_since_log
                avg_bce     = running_bce     / steps_since_log
                avg_focal   = running_focal   / steps_since_log
                avg_reg_div = running_reg_div / steps_since_log
                current_lr  = scheduler.get_last_lr()[0]

                log(
                    f'Epoch {epoch}/{args.epochs} Step {step}/{len(train_loader)} | '
                    f'Loss: {avg_loss:.4f} | BCE: {avg_bce:.4f} | '
                    f'Focal: {avg_focal:.4f} | RegDiv: {avg_reg_div:.4f} | '
                    f'LR: {current_lr:.2e}'
                )

                running_loss    = 0.0
                running_bce     = 0.0
                running_focal   = 0.0
                running_reg_div = 0.0
                steps_since_log = 0

            # Periodic checkpoint
            if global_step % args.save_steps == 0:
                ckpt_path = os.path.join(args.output_dir, f'checkpoint_step{global_step}.pth')
                save_checkpoint(model, optimizer, epoch, best_ap, ckpt_path)
                print(f'  Saved checkpoint: {ckpt_path}')

        # ── End of epoch: validation ─────────────────────────────────────────
        metrics = validate(model, val_loader, args)
        ap      = metrics['ap']
        acc     = metrics['acc']
        f1      = metrics['f1']

        is_best = ap > best_ap
        star    = ' ★' if is_best else ''

        log(
            f'Epoch {epoch}/{args.epochs} Val | '
            f'AP: {ap:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | '
            f'Best: {max(ap, best_ap):.4f}{star}'
        )

        if is_best:
            best_ap = ap
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_ap, best_path)
            print(f'  New best model saved → {best_path}  (AP={best_ap:.4f})')
        else:
            patience_counter += 1
            print(f'  No improvement. Patience: {patience_counter}/{args.patience}')

        if patience_counter >= args.patience:
            log(f'Early stopping triggered at epoch {epoch} (patience={args.patience})')
            break

    log(f'Training complete. Best val AP: {best_ap:.4f} | Checkpoint: {best_path}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_logging(args)

    print('=' * 60)
    print(' FaceForge-Net Training')
    print(f'  Train JSON:  {args.train_json}')
    print(f'  Val JSON:    {args.val_json}')
    print(f'  Output dir:  {args.output_dir}')
    print(f'  Epochs:      {args.epochs}')
    print(f'  Batch size:  {args.batch_size}')
    print(f'  BF16:        {args.bf16}')
    print(f'  LoRA r/α:    {args.lora_r}/{args.lora_alpha}  blocks={args.lora_blocks}')
    print(f'  Patience:    {args.patience}')
    print('=' * 60)

    model = build_model(args)
    train_loader, val_loader = build_dataloaders(args)
    optimizer, scheduler = build_optimizer(model, args)
    train(model, train_loader, val_loader, optimizer, scheduler, args)


if __name__ == '__main__':
    main()
