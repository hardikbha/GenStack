"""
XGenDet v3 Training — supports resolution-aware aug and frequency branch.
Train → auto test eval → save results. DataParallel safe (heatmap off for multi-GPU).
"""

import os, sys, time, argparse, json
from pathlib import Path
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import average_precision_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.hydrafake_dataset import HydraFakeDataset, HydraFakeTestDataset
from training.losses import XGenDetLoss


def parse_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--model", choices=["base", "freq", "v2", "srm"], default="base")
    p.add_argument("--clip_model", default="ViT-L/14")
    p.add_argument("--num_prompt_tokens", type=int, default=8)
    p.add_argument("--num_prototypes", type=int, default=128)
    p.add_argument("--proto_dim", type=int, default=128)
    p.add_argument("--shuffle_patch_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=3)
    # Training
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_ln", type=float, default=2e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    # Augmentation
    p.add_argument("--aug", choices=["v1", "v2", "v3", "v4"], default="v1",
                   help="v1=basic, v2=resolution-aware, v3=hard-generator targeted, v4=native-crop consistency")
    p.add_argument("--jpeg_prob", type=float, default=0.5)
    p.add_argument("--blur_prob", type=float, default=0.5)
    p.add_argument("--resolution_aug_prob", type=float, default=0.4)
    # Family weighting (0=Real,1=FS,2=EFG,3=FR) — boost families similar to weak test generators
    p.add_argument("--family_weights", default="0:1.0,1:1.5,2:0.8,3:2.0",
                   help="Per-family sampling weights. FR×2 helps StarGANv2/hailuo, FS×1.5 helps FFIW.")
    # Data
    p.add_argument("--train_json", default="/home/sachin.chaudhary/hydrafake/jsons/train/all.json")
    p.add_argument("--val_json", default="/home/sachin.chaudhary/hydrafake/jsons/val/all.json")
    p.add_argument("--data_root", default="/home/sachin.chaudhary")
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    # Loss
    p.add_argument("--w_family", type=float, default=0.5)
    p.add_argument("--w_proto_div", type=float, default=0.3)
    p.add_argument("--w_proto_compact", type=float, default=0.2)
    p.add_argument("--w_heatmap", type=float, default=0.3)
    p.add_argument("--w_attr", type=float, default=0.1)
    p.add_argument("--w_calib", type=float, default=0.2)
    # Logging
    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--exp_name", default="v3_exp")
    p.add_argument("--log_freq", type=int, default=25)
    p.add_argument("--save_freq", type=int, default=1)
    p.add_argument("--test_interval", type=int, default=10,
                   help="Run test evaluation every N epochs to monitor progress")
    # Resume
    p.add_argument("--pretrained", default=None)
    p.add_argument("--test_dir", default="/home/sachin.chaudhary/hydrafake/jsons/test")
    # Generator oversampling (e.g. "FaceForensics++:3.0" triples FF++ samples)
    p.add_argument("--gen_oversample", default="",
                   help="Comma-separated generator:factor pairs, e.g. 'FaceForensics++:3.0'")
    # Backbone partial unfreeze
    p.add_argument("--unfreeze_blocks", type=int, default=0,
                   help="Unfreeze last N CLIP transformer blocks (0=fully frozen). "
                        "LR for these blocks = lr * 0.005 (very conservative).")
    # ViT Adapter blocks
    p.add_argument("--adapter_blocks", default="",
                   help="Comma-separated block indices for ViT adapters, e.g. '12,13,...,23'. "
                        "Use 'last12' for blocks 12-23, 'last6' for 18-23.")
    p.add_argument("--adapter_bottleneck", type=int, default=64,
                   help="Bottleneck dim for each ViT adapter (default 64).")
    return p.parse_args()


def parse_adapter_blocks(spec: str) -> list:
    """Parse adapter block spec: 'last12' → [12..23], 'last6' → [18..23], or '12,13,14' → [12,13,14]."""
    if not spec:
        return []
    spec = spec.strip()
    if spec == "last12":
        return list(range(12, 24))
    if spec == "last6":
        return list(range(18, 24))
    if spec == "last8":
        return list(range(16, 24))
    try:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    except ValueError:
        return []


def build_model(args, device):
    adapter_blocks = parse_adapter_blocks(getattr(args, "adapter_blocks", ""))
    adapter_bn     = getattr(args, "adapter_bottleneck", 64)

    if args.model == "v2":
        from models.xgendet_v2 import XGenDetV2
        model = XGenDetV2(
            clip_model_name=args.clip_model,
            num_prompt_tokens=args.num_prompt_tokens,
            num_prototypes=args.num_prototypes,
            proto_dim=args.proto_dim,
            shuffle_patch_size=args.shuffle_patch_size,
            top_k=args.top_k,
        )
    elif args.model == "freq":
        from models.xgendet_freq import XGenDetFreq
        kwargs = dict(clip_model_name=args.clip_model, num_prompt_tokens=args.num_prompt_tokens,
                      num_prototypes=args.num_prototypes, proto_dim=args.proto_dim,
                      shuffle_patch_size=args.shuffle_patch_size)
        model = XGenDetFreq(spectral_dim=128, num_spectral_filters=8, **kwargs)
    elif args.model == "srm":
        from models.xgendet_srm import XGenDetSRM
        model = XGenDetSRM(
            srm_dim=256,
            clip_model=args.clip_model,
            num_prompt_tokens=args.num_prompt_tokens,
            num_prototypes=args.num_prototypes,
            proto_dim=args.proto_dim,
            shuffle_patch_size=args.shuffle_patch_size,
        )
    else:
        from models.xgendet import XGenDet
        model = XGenDet(
            clip_model_name=args.clip_model,
            num_prompt_tokens=args.num_prompt_tokens,
            num_prototypes=args.num_prototypes,
            proto_dim=args.proto_dim,
            shuffle_patch_size=args.shuffle_patch_size,
            adapter_blocks=adapter_blocks,
            adapter_bottleneck=adapter_bn,
        )
    return model.to(device)


def build_dataset(args, is_train=True):
    if is_train and args.aug == "v2":
        from data.augmentations_v2 import get_train_transforms_v2
        transform = get_train_transforms_v2(
            crop_size=args.crop_size,
            jpeg_prob=args.jpeg_prob,
            blur_prob=args.blur_prob,
            resolution_aug_prob=args.resolution_aug_prob,
        )
    elif is_train and args.aug == "v3":
        from data.augmentations_v3 import get_train_transforms_v3
        transform = get_train_transforms_v3(
            crop_size=args.crop_size,
            jpeg_prob=args.jpeg_prob,
            blur_prob=args.blur_prob,
            resolution_aug_prob=args.resolution_aug_prob,
        )
    elif is_train and args.aug == "v4":
        from data.augmentations_v4 import get_train_transforms_v4
        transform = get_train_transforms_v4(
            crop_size=args.crop_size,
            jpeg_prob=args.jpeg_prob,
            blur_prob=args.blur_prob,
        )
    elif is_train:
        from data.augmentations import get_train_transforms
        transform = get_train_transforms(args.crop_size, args.jpeg_prob, args.blur_prob)
    else:
        from data.augmentations import get_eval_transforms
        transform = get_eval_transforms(args.crop_size)

    json_path = args.train_json if is_train else args.val_json
    ds = HydraFakeDataset(
        json_path=json_path, data_root=args.data_root,
        is_train=is_train, crop_size=args.crop_size,
        jpeg_prob=args.jpeg_prob, blur_prob=args.blur_prob,
    )
    # Generator oversampling: inject before data is used
    if is_train and hasattr(args, 'gen_oversample') and args.gen_oversample:
        gen_map = {}
        for kv in args.gen_oversample.split(","):
            if ":" in kv:
                gen, factor = kv.strip().split(":")
                gen_map[gen.strip()] = float(factor)
        if gen_map:
            ds._gen_oversample = gen_map
            # Re-trigger oversampling (data already built, apply now)
            extra = []
            for path, label, family in ds.data:
                for pattern, factor in gen_map.items():
                    if pattern in path:
                        extra.extend([(path, label, family)] * int(factor - 1))
            ds.data.extend(extra)
            from random import shuffle as random_shuffle
            random_shuffle(ds.data)
            n_ff = sum(1 for d in ds.data if any(p in d[0] for p in gen_map))
            print(f"  Gen oversample {gen_map}: total={len(ds.data)}, matched={n_ff}")
    # Override transform if v2, v3 or v4
    if is_train and args.aug in ("v2", "v3", "v4"):
        ds.transform = transform
    return ds


def validate(model, val_loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lab, _ in val_loader:
            imgs = imgs.to(device)
            out = model(imgs, return_heatmap=False)
            preds.extend(out["confidence"].squeeze(-1).cpu().tolist())
            labels.extend(lab.tolist())
    ap = average_precision_score(labels, preds)
    acc = accuracy_score(labels, (np.array(preds) > 0.5).astype(int))
    model.train()
    return {"ap": ap, "acc": acc}


def evaluate_test(model, args, device):
    model.eval()
    results = {}
    for split in ["id", "cm", "cf", "cd"]:
        split_dir = os.path.join(args.test_dir, split)
        if not os.path.isdir(split_dir):
            continue
        sp, sl, pg = [], [], {}
        for jf in sorted(f for f in os.listdir(split_dir) if f.endswith(".json")):
            gen = jf.replace(".json", "")
            ds = HydraFakeTestDataset(
                os.path.join(split_dir, jf), args.data_root,
                "/home/sachin.chaudhary/hydrafake/test", args.crop_size)
            if len(ds) == 0: continue
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            gp, gl = [], []
            with torch.no_grad():
                for imgs, lab in loader:
                    imgs = imgs.to(device)
                    out = model(imgs, return_heatmap=False)
                    gp.extend(out["confidence"].squeeze(-1).cpu().tolist())
                    gl.extend(lab.tolist())
            gp_np, gl_np = np.array(gp), np.array(gl)
            g_ap = average_precision_score(gl_np, gp_np) if len(np.unique(gl_np)) > 1 else -1
            g_acc = accuracy_score(gl_np, (gp_np > 0.5).astype(int))
            pg[gen] = {"ap": g_ap, "acc": g_acc, "n": len(ds)}
            sp.extend(gp); sl.extend(gl)
            print(f"    {gen}: Acc={g_acc*100:.1f}%, AP={g_ap:.4f}, n={len(ds)}")
        s_np, l_np = np.array(sp), np.array(sl)
        s_ap = average_precision_score(l_np, s_np) if len(np.unique(l_np)) > 1 else -1
        s_acc = accuracy_score(l_np, (s_np > 0.5).astype(int))
        results[split] = {"ap": s_ap, "acc": s_acc, "n": len(sp), "per_generator": pg}
        print(f"  {split.upper()}: Acc={s_acc*100:.1f}%, AP={s_ap:.4f}")
    if results:
        avg_acc = np.mean([r["acc"] for r in results.values()])
        avg_ap = np.mean([r["ap"] for r in results.values() if r["ap"] >= 0])
        results["average"] = {"ap": avg_ap, "acc": avg_acc}
        print(f"\n  AVERAGE: Acc={avg_acc*100:.1f}%, AP={avg_ap:.4f}")
    return results


class LabelSmoothBCE(nn.Module):
    def __init__(self, s=0.1):
        super().__init__()
        self.s = s
    def forward(self, logit, target):
        t = target.float() * (1 - self.s) + 0.5 * self.s
        return F.binary_cross_entropy_with_logits(logit.squeeze(-1), t)


def train():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(exp_dir, "config.json"), "w"), indent=2)

    print(f"Building model: {args.model}, Aug: {args.aug}, GPUs: {n_gpus}")
    model = build_model(args, device)
    raw_model = model
    print(f"Trainable params: {raw_model.count_trainable_params()}")

    if args.pretrained and os.path.exists(args.pretrained):
        sd = torch.load(args.pretrained, map_location=device).get("model_state_dict", {})
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        raw_model.load_state_dict(sd, strict=False)
        print(f"Loaded pretrained: {args.pretrained}")

    # Partial backbone unfreeze (AFTER loading pretrained weights)
    if args.unfreeze_blocks > 0:
        raw_model.backbone.unfreeze_last_blocks(args.unfreeze_blocks)
        tp = raw_model.count_trainable_params()
        total = tp.get("total", sum(tp.values())) if isinstance(tp, dict) else tp
        print(f"Trainable params after backbone unfreeze: {total:,}")

    use_dp = n_gpus > 1
    if use_dp:
        model = nn.DataParallel(model)
        args.batch_size *= n_gpus
        print(f"DataParallel on {n_gpus} GPUs, batch={args.batch_size}")

    # Parse family weights
    fam_w = {int(k): float(v) for k, v in (kv.split(":") for kv in args.family_weights.split(","))}
    print(f"Family weights: {fam_w}")

    # Data
    train_ds = build_dataset(args, is_train=True)
    val_ds = build_dataset(args, is_train=False)
    labels_list = [d[1] for d in train_ds.data]
    families_list = [d[2] for d in train_ds.data]
    cc = [labels_list.count(0), labels_list.count(1)]
    # Combine class-balance weight with per-family weight
    weights = [fam_w.get(f, 1.0) / cc[l] for l, f in zip(labels_list, families_list)]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=WeightedRandomSampler(weights, len(train_ds), replacement=True),
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Loss + optimizer
    criterion = XGenDetLoss(w_family=args.w_family, w_proto_div=args.w_proto_div,
                            w_proto_compact=args.w_proto_compact, w_heatmap=args.w_heatmap,
                            w_attr=args.w_attr, w_calib=args.w_calib)
    ls_bce = LabelSmoothBCE(args.label_smoothing)

    param_groups = raw_model.get_trainable_params()
    opt_params = []
    for g in param_groups:
        lr_s = float(g.get("lr_scale", 1.0))
        opt_params.append({"params": g["params"],
                           "lr": float(args.lr) * lr_s if "layer_norm" not in g.get("name", "") else float(args.lr_ln),
                           "name": g.get("name", "")})
    optimizer = torch.optim.AdamW(opt_params, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    proto_mod = raw_model.prototype_module if hasattr(raw_model, 'prototype_module') else (raw_model.base.prototype_module if hasattr(raw_model, 'base') else None)

    best_ap, patience_ctr, g_step = 0.0, 0, 0
    print(f"\n{'='*60}")
    print(f"Exp: {args.exp_name} | Model: {args.model} | Aug: {args.aug}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses_acc = defaultdict(float)
        n_batch = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True, ncols=120)
        for bi, (imgs, labels, families) in enumerate(pbar):
            imgs, labels, families = imgs.to(device), labels.to(device), families.to(device)
            out = model(imgs, return_heatmap=not use_dp)
            losses = criterion(out, labels, families, prototype_module=proto_mod)
            losses["cls_smooth"] = ls_bce(out["binary_logit"], labels)
            losses["total"] = (losses["cls_smooth"] + criterion.w_family * losses["family"]
                               + criterion.w_proto_div * losses["proto_div"]
                               + criterion.w_proto_compact * losses["proto_compact"]
                               + criterion.w_heatmap * losses["heatmap"]
                               + criterion.w_attr * losses["attr"]
                               + criterion.w_calib * losses["calib"])
            (losses["total"] / args.grad_accum).backward()
            if (bi + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(); g_step += 1
            for k in losses: losses_acc[k] += losses[k].item()
            n_batch += 1
            pbar.set_postfix(loss=f"{losses_acc['total']/n_batch:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        et = time.time() - t0
        for k in losses_acc: losses_acc[k] /= max(n_batch, 1)
        print(f"\nEpoch {epoch}/{args.epochs} ({et:.0f}s) Loss={losses_acc['total']:.4f}")

        vm = validate(raw_model, val_loader, device)
        print(f"  Val: AP={vm['ap']:.4f}, Acc={vm['acc']:.4f}")

        # Intermediate test evaluation every N epochs
        if epoch % args.test_interval == 0:
            print(f"\n  Running test evaluation at epoch {epoch}...")
            test_results = evaluate_test(raw_model, args, device)
            test_avg_acc = test_results.get("average", {}).get("acc", 0.0)
            print(f"  Test Avg Acc @ Epoch {epoch}: {test_avg_acc*100:.2f}%")
            # Save intermediate results
            json.dump(test_results, open(os.path.join(exp_dir, f"test_results_epoch_{epoch}.json"), "w"), indent=2)
            print()

        if epoch % args.save_freq == 0:
            torch.save({"epoch": epoch, "model_state_dict": raw_model.state_dict(),
                         "val_ap": vm["ap"], "val_acc": vm["acc"], "args": vars(args)},
                       os.path.join(exp_dir, f"epoch_{epoch}.pth"))

        if vm["ap"] > best_ap:
            best_ap = vm["ap"]; patience_ctr = 0
            torch.save({"epoch": epoch, "model_state_dict": raw_model.state_dict(),
                         "val_ap": vm["ap"], "val_acc": vm["acc"], "args": vars(args)},
                       os.path.join(exp_dir, "best_model.pth"))
            print(f"  *** New best! AP={best_ap:.4f} Acc={vm['acc']:.4f} ***")
        else:
            patience_ctr += 1
            print(f"  No improvement. Patience: {patience_ctr}/{args.patience}")
        if patience_ctr >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}"); break

    # Auto test eval
    print(f"\n{'='*60}\nRunning test evaluation...\n{'='*60}")
    ckpt = torch.load(os.path.join(exp_dir, "best_model.pth"), map_location=device)
    raw_model.load_state_dict(ckpt["model_state_dict"])
    raw_model.eval()
    results = evaluate_test(raw_model, args, device)
    json.dump(results, open(os.path.join(exp_dir, "test_results.json"), "w"), indent=2)
    print(f"\nResults saved to {exp_dir}/test_results.json")
    print(f"Best Val AP: {best_ap:.4f}")


if __name__ == "__main__":
    train()
