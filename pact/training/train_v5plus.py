"""
XGenDetV5Plus Training Script.

Trains only the 3 small branches + fusion head (~2M params) grafted on top of a
frozen XGenDet v5 backbone.

Loss:
    L = BCEWithLogitsLoss(logit, y) + 0.3 * BCEWithLogitsLoss(logit_v5, y)
    The auxiliary v5 term keeps the frozen backbone's calibration signal from
    drifting while the new branches learn to complement it.

Optimizer: AdamW with two param groups from model.get_trainable_params():
    branches  → lr=1e-4
    head      → lr=5e-4
Scheduler: CosineAnnealingLR, T_max=total_steps, eta_min=1e-7

Usage (single GPU):
    python training/train_v5plus.py \\
        --v5_checkpoint checkpoints/v5_resume_hl/best_model.pth \\
        --train_json /home/sachin.chaudhary/hydrafake/jsons/train/all.json \\
        --val_json   /home/sachin.chaudhary/hydrafake/jsons/val/all.json \\
        --data_root  /home/sachin.chaudhary \\
        --output_dir checkpoints/v5plus \\
        --epochs 20 --batch_size 32 --num_workers 2 --patience 7 --bf16

Usage (multi-GPU via DataParallel, controlled by CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=0,1 python training/train_v5plus.py ...
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
from models.xgendet_v5plus import XGenDetV5Plus
from data.hydrafake_dataset import HydraFakeDataset


# ── Live log ──────────────────────────────────────────────────────────────────

LOG_FILE = '/home/sachin.chaudhary/xgendet/logs/v5plus_live.log'


def setup_logging(args: argparse.Namespace) -> None:
    """Ensure log directory and output directory exist."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)


def log(msg: str) -> None:
    """Write timestamped message to both stdout and the live log file."""
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


# ── Arg parsing ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train XGenDetV5Plus branches+head on HydraFake')

    # Checkpoints
    p.add_argument('--v5_checkpoint', default='checkpoints/v5_resume_hl/best_model.pth',
                   help='Path to frozen XGenDet v5 checkpoint')

    # Data
    p.add_argument('--train_json', default='/home/sachin.chaudhary/hydrafake/jsons/train/all.json',
                   help='Path to train split JSON')
    p.add_argument('--val_json', default='/home/sachin.chaudhary/hydrafake/jsons/val/all.json',
                   help='Path to val split JSON')
    p.add_argument('--data_root', default='/home/sachin.chaudhary',
                   help='Root directory for relative image paths')

    # Output
    p.add_argument('--output_dir', default='checkpoints/v5plus',
                   help='Directory to save branch+head checkpoints')

    # Training hyperparams
    p.add_argument('--epochs',      type=int,   default=20)
    p.add_argument('--batch_size',  type=int,   default=32,
                   help='Per-GPU batch size')
    p.add_argument('--num_workers', type=int,   default=2)
    p.add_argument('--patience',    type=int,   default=7,
                   help='Early-stopping patience on val AP')
    p.add_argument('--log_steps',   type=int,   default=50,
                   help='Log every N global steps')
    p.add_argument('--bf16',        action='store_true', default=False,
                   help='Use bfloat16 mixed precision')

    return p.parse_args()


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(args: argparse.Namespace) -> nn.Module:
    """
    Instantiate XGenDetV5Plus, load the frozen v5 checkpoint, optionally
    cast to bfloat16, and wrap in DataParallel if multiple GPUs are available.
    """
    model = XGenDetV5Plus(v5_checkpoint_path=args.v5_checkpoint)

    if args.bf16:
        model = model.to(torch.bfloat16)

    n_gpus = torch.cuda.device_count()
    print(f'>>> CUDA device count: {n_gpus}')

    if n_gpus > 1:
        print(f'>>> Using DataParallel across {n_gpus} GPUs')
        model = nn.DataParallel(model)
    elif n_gpus == 1:
        print('>>> Using single GPU')

    if n_gpus >= 1:
        model = model.cuda()

    # Report trainable parameter count
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    total_trainable = sum(
        p.numel() for p in base_model.parameters() if p.requires_grad
    )
    print(f'>>> Trainable params: {total_trainable:,}  (~{total_trainable/1e6:.1f}M)')

    return model


# ── DataLoaders ───────────────────────────────────────────────────────────────

def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders using the existing HydraFakeDataset."""
    train_dataset = HydraFakeDataset(
        json_path=args.train_json,
        data_root=args.data_root,
        is_train=True,
        crop_size=224,
    )
    val_dataset = HydraFakeDataset(
        json_path=args.val_json,
        data_root=args.data_root,
        is_train=False,
        crop_size=224,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f'>>> Train samples: {len(train_dataset):,} | batches: {len(train_loader)}')
    print(f'>>> Val samples:   {len(val_dataset):,} | batches: {len(val_loader)}')

    return train_loader, val_loader


# ── Optimizer & scheduler ─────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
) -> tuple[AdamW, CosineAnnealingLR]:
    """
    Build AdamW with two param groups from XGenDetV5Plus.get_trainable_params():
        branches → lr=1e-4
        head     → lr=5e-4

    The scheduler is initialised with T_max=1 here; it is re-built in train()
    once total_steps is known.
    """
    lr_map = {
        'branches': 1e-4,
        'head':     5e-4,
    }

    base_model = model.module if isinstance(model, nn.DataParallel) else model
    param_groups = base_model.get_trainable_params()

    for g in param_groups:
        name = g.get('name', '?')
        if name in lr_map:
            g['lr'] = lr_map[name]
        n_params = sum(p.numel() for p in g['params'])
        print(f'  Param group "{name}": {n_params:,} params @ lr={g["lr"]:.1e}')

    optimizer = AdamW(param_groups, weight_decay=0.01)

    # Placeholder scheduler — re-built with correct T_max in train()
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
    Run full validation pass and return:
        {'ap': float, 'acc': float, 'f1': float}
    """
    model.eval()
    all_probs  = []
    all_preds  = []
    all_labels = []

    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32

    for imgs, labels, _families in val_loader:
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        if args.bf16:
            imgs = imgs.to(torch.bfloat16)

        with torch.autocast('cuda', dtype=autocast_dtype, enabled=torch.cuda.is_available()):
            outputs = model(imgs)

        probs = outputs['prob'].float().squeeze(1).cpu()
        preds = (probs >= 0.5).long()

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.long().tolist())

    ap  = average_precision_score(all_labels, all_probs)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    model.train()
    return {'ap': ap, 'acc': acc, 'f1': f1}


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    args: argparse.Namespace,
) -> None:
    """Main training loop with cosine LR schedule and early stopping on val AP."""

    total_steps = len(train_loader) * args.epochs

    # Re-build scheduler now that total_steps is known
    for pg in optimizer.param_groups:
        pg['initial_lr'] = pg['lr']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7, last_epoch=-1)

    best_ap          = 0.0
    best_path        = os.path.join(args.output_dir, 'best_branches.pth')
    patience_counter = 0
    global_step      = 0

    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Convenience: access base model for saving
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    log(
        f'Starting training | epochs={args.epochs} | '
        f'steps_per_epoch={len(train_loader)} | total_steps={total_steps}'
    )

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss    = 0.0
        running_main    = 0.0
        running_aux     = 0.0
        steps_since_log = 0

        for step, (imgs, labels, _families) in enumerate(train_loader, start=1):
            global_step += 1

            if torch.cuda.is_available():
                imgs   = imgs.cuda()
                labels = labels.cuda()
            if args.bf16:
                imgs = imgs.to(torch.bfloat16)

            labels_f = labels.float()

            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=autocast_dtype, enabled=torch.cuda.is_available()):
                outputs = model(imgs)

                logit    = outputs['logit'].squeeze(1)      # [B]
                logit_v5 = outputs['logit_v5'].squeeze(1)   # [B]

                loss_main = F.binary_cross_entropy_with_logits(logit, labels_f)
                loss_aux  = F.binary_cross_entropy_with_logits(logit_v5, labels_f)
                loss      = loss_main + 0.3 * loss_aux

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss    += loss.item()
            running_main    += loss_main.item()
            running_aux     += loss_aux.item()
            steps_since_log += 1

            if global_step % args.log_steps == 0:
                avg_loss = running_loss    / steps_since_log
                avg_main = running_main    / steps_since_log
                avg_aux  = running_aux     / steps_since_log

                # Read per-group LRs directly from optimizer param_groups
                # (names were set during build_optimizer)
                branch_lr = None
                head_lr   = None
                for pg in optimizer.param_groups:
                    if pg.get('name') == 'branches':
                        branch_lr = pg['lr']
                    elif pg.get('name') == 'head':
                        head_lr = pg['lr']
                # Fallback if names not present
                if branch_lr is None:
                    branch_lr = optimizer.param_groups[0]['lr']
                if head_lr is None:
                    head_lr = optimizer.param_groups[-1]['lr']

                log(
                    f'Epoch {epoch}/{args.epochs} Step {step}/{len(train_loader)} | '
                    f'Loss: {avg_loss:.3f} | Main: {avg_main:.3f} | Aux: {avg_aux:.3f} | '
                    f'BranchLR: {branch_lr:.1e} | HeadLR: {head_lr:.1e}'
                )

                running_loss    = 0.0
                running_main    = 0.0
                running_aux     = 0.0
                steps_since_log = 0

        # ── End-of-epoch validation ───────────────────────────────────────────
        metrics = validate(model, val_loader, args)
        ap  = metrics['ap']
        acc = metrics['acc']
        f1  = metrics['f1']

        is_best = ap > best_ap
        star    = ' \u2605' if is_best else ''

        log(
            f'Epoch {epoch}/{args.epochs} Val | '
            f'AP: {ap:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | '
            f'Best: {max(ap, best_ap):.4f}{star}'
        )

        if is_best:
            best_ap          = ap
            patience_counter = 0
            # Save only branch+head params (~30 MB, not 1.6 GB)
            base_model.save_branches_checkpoint(best_path)
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
    print(' XGenDetV5Plus Training  (branches + head only)')
    print(f'  v5 checkpoint: {args.v5_checkpoint}')
    print(f'  Train JSON:    {args.train_json}')
    print(f'  Val JSON:      {args.val_json}')
    print(f'  Output dir:    {args.output_dir}')
    print(f'  Epochs:        {args.epochs}')
    print(f'  Batch size:    {args.batch_size}')
    print(f'  BF16:          {args.bf16}')
    print(f'  Patience:      {args.patience}')
    print('=' * 60)

    model                    = build_model(args)
    train_loader, val_loader = build_dataloaders(args)
    optimizer, scheduler     = build_optimizer(model, args)
    train(model, train_loader, val_loader, optimizer, scheduler, args)


if __name__ == '__main__':
    main()
