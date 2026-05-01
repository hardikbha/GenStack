"""
Train CNN Detector (EfficientNet-B4 + SRM) — the "different model" for ensemble with XGenDet.
Fully fine-tuned, 19M params. Designed to make different errors than CLIP-based XGenDet.
"""

import os, sys, time, json, argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import average_precision_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.hydrafake_dataset import HydraFakeDataset, HydraFakeTestDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name",    default="cnn_v1")
    p.add_argument("--crop_size",   type=int, default=336)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--patience",    type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--test_interval", type=int, default=5)
    p.add_argument("--train_json",  default="/home/sachin.chaudhary/hydrafake/jsons/train/all.json")
    p.add_argument("--val_json",    default="/home/sachin.chaudhary/hydrafake/jsons/val/all.json")
    p.add_argument("--test_dir",    default="/home/sachin.chaudhary/hydrafake/jsons/test")
    p.add_argument("--data_root",   default="/home/sachin.chaudhary")
    p.add_argument("--output_dir",  default="checkpoints")
    p.add_argument("--pretrained",  default=None)
    return p.parse_args()


def get_transforms(crop_size, is_train):
    import torchvision.transforms as T
    CLIP_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization for EfficientNet
    CLIP_STD  = [0.229, 0.224, 0.225]
    if is_train:
        from data.augmentations_v4 import NativeCropResize
        return T.Compose([
            NativeCropResize(crop_size=crop_size, p_native=0.5),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    else:
        return T.Compose([
            T.Resize((crop_size, crop_size)),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])


def evaluate_test(model, args, device):
    model.eval()
    results = {}
    for split in ["id", "cm", "cf", "cd"]:
        split_dir = os.path.join(args.test_dir, split)
        if not os.path.isdir(split_dir): continue
        sp, sl, pg = [], [], {}
        for jf in sorted(f for f in os.listdir(split_dir) if f.endswith(".json")):
            gen = jf.replace(".json", "")
            ds = HydraFakeTestDataset(
                os.path.join(split_dir, jf), args.data_root,
                "/home/sachin.chaudhary/hydrafake/test", args.crop_size)
            if len(ds) == 0: continue
            loader = DataLoader(ds, batch_size=args.batch_size*2, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            gp, gl = [], []
            with torch.no_grad():
                for imgs, lab in loader:
                    out = model(imgs.to(device))
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


def train():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(exp_dir, "config.json"), "w"), indent=2)

    # Model
    from models.cnn_detector import CNNDetector
    model = CNNDetector(srm_dim=128, dropout=0.3)
    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNNDetector: {total_p/1e6:.1f}M trainable params, {n_gpus} GPUs")

    if args.pretrained and os.path.exists(args.pretrained):
        sd = torch.load(args.pretrained, map_location=device).get("model_state_dict", {})
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print(f"Loaded pretrained: {args.pretrained}")

    model = model.to(device)
    raw_model = model
    if n_gpus > 1:
        model = nn.DataParallel(model)
        args.batch_size *= n_gpus
        print(f"DataParallel on {n_gpus} GPUs, batch={args.batch_size}")

    # Data
    train_ds = HydraFakeDataset(args.train_json, args.data_root, is_train=True, crop_size=args.crop_size)
    val_ds   = HydraFakeDataset(args.val_json,   args.data_root, is_train=False, crop_size=args.crop_size)
    train_ds.transform = get_transforms(args.crop_size, is_train=True)
    val_ds.transform   = get_transforms(args.crop_size, is_train=False)

    labels_list = [d[1] for d in train_ds.data]
    cc = [labels_list.count(0), labels_list.count(1)]
    weights = [1.0/cc[l] for l in labels_list]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=WeightedRandomSampler(weights, len(train_ds), replacement=True),
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Optimizer with differential LR: backbone lower, head higher
    backbone_params = list(raw_model.backbone.parameters())
    head_params = (list(raw_model.noise_cnn.parameters()) +
                   list(raw_model.classifier.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},  # 1e-5 for backbone
        {"params": head_params,     "lr": args.lr},         # 1e-4 for head + SRM
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    # Label smoothing BCE
    smooth = args.label_smoothing
    def ls_bce(logit, target):
        t = target.float() * (1 - smooth) + 0.5 * smooth
        return F.binary_cross_entropy_with_logits(logit.squeeze(-1), t)

    best_ap, patience_ctr = 0.0, 0
    print(f"\n{'='*60}")
    print(f"Exp: {args.exp_name} | EfficientNet-B4 + SRM")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Crop: {args.crop_size} | Label smooth: {args.label_smoothing}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batch = 0.0, 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True, ncols=100)
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = ls_bce(out["binary_logit"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item(); n_batch += 1
            pbar.set_postfix(loss=f"{total_loss/n_batch:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        avg_loss = total_loss / max(n_batch, 1)
        et = time.time() - t0
        print(f"\nEpoch {epoch}/{args.epochs} ({et:.0f}s) Loss={avg_loss:.4f}")

        # Validate
        raw_model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for imgs, lab, _ in val_loader:
                out = raw_model(imgs.to(device))
                preds.extend(out["confidence"].squeeze(-1).cpu().tolist())
                labs.extend(lab.tolist())
        val_ap = average_precision_score(labs, preds)
        val_acc = accuracy_score(labs, (np.array(preds) > 0.5).astype(int))
        print(f"  Val: AP={val_ap:.4f}, Acc={val_acc:.4f}")

        # Test eval
        if epoch % args.test_interval == 0:
            print(f"\n  Test evaluation @ epoch {epoch}:")
            test_r = evaluate_test(raw_model, args, device)
            test_avg = test_r.get("average", {}).get("acc", 0)
            print(f"  Test Avg Acc @ Epoch {epoch}: {test_avg*100:.2f}%")
            json.dump(test_r, open(os.path.join(exp_dir, f"test_results_epoch_{epoch}.json"), "w"), indent=2)

        # Save
        torch.save({"epoch": epoch, "model_state_dict": raw_model.state_dict(),
                     "val_ap": val_ap}, os.path.join(exp_dir, f"epoch_{epoch}.pth"))
        if val_ap > best_ap:
            best_ap = val_ap; patience_ctr = 0
            torch.save({"epoch": epoch, "model_state_dict": raw_model.state_dict(),
                         "val_ap": val_ap}, os.path.join(exp_dir, "best_model.pth"))
            print(f"  *** New best! AP={best_ap:.4f} ***")
        else:
            patience_ctr += 1
            print(f"  No improvement. Patience: {patience_ctr}/{args.patience}")
        if patience_ctr >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}"); break

    # Final test
    print(f"\n{'='*60}\nFinal test evaluation (best model)...\n{'='*60}")
    ckpt = torch.load(os.path.join(exp_dir, "best_model.pth"), map_location=device)
    raw_model.load_state_dict(ckpt["model_state_dict"])
    results = evaluate_test(raw_model, args, device)
    json.dump(results, open(os.path.join(exp_dir, "test_results.json"), "w"), indent=2)
    print(f"\nResults saved to {exp_dir}/test_results.json")


if __name__ == "__main__":
    train()
