"""
Step 2: Self-training — fine-tune v5_resume_hl on pseudo-labeled CF/CD images
mixed with original training data.

Training mix:
  - Original 48K train images (real labels)
  - Pseudo-labeled CF/CD test images (high-confidence predictions)
  - Weight pseudo-labeled samples 2× to give them extra emphasis

Usage:
  python training/self_train.py \
    --pretrained checkpoints/v5_resume_hl/best_model.pth \
    --pseudo_json checkpoints/v6_selftrain/pseudo_labels.json \
    --exp_name v6_selftrain
"""

import os, sys, time, json, argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from sklearn.metrics import average_precision_score, accuracy_score
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgendet import XGenDet
from data.hydrafake_dataset import HydraFakeDataset, HydraFakeTestDataset, HYDRAFAKE_FAMILIES
from data.augmentations import get_train_transforms, get_eval_transforms
from training.losses import XGenDetLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained",   default="checkpoints/v5_resume_hl/best_model.pth")
    p.add_argument("--train_json",   default="/home/sachin.chaudhary/hydrafake/jsons/train/all.json")
    p.add_argument("--val_json",     default="/home/sachin.chaudhary/hydrafake/jsons/val/all.json")
    p.add_argument("--pseudo_json",  default="checkpoints/v6_selftrain/pseudo_labels.json")
    p.add_argument("--data_root",    default="/home/sachin.chaudhary")
    p.add_argument("--test_dir",     default="/home/sachin.chaudhary/hydrafake/jsons/test")
    p.add_argument("--image_root",   default="/home/sachin.chaudhary/hydrafake/test")
    p.add_argument("--output_dir",   default="checkpoints")
    p.add_argument("--exp_name",     default="v6_selftrain")
    # Training
    p.add_argument("--lr",           type=float, default=5e-6)
    p.add_argument("--lr_ln",        type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=8)
    p.add_argument("--patience",     type=int,   default=4)
    p.add_argument("--pseudo_weight",type=float, default=2.0,
                   help="Sampling weight for pseudo-labeled samples vs original")
    p.add_argument("--crop_size",    type=int,   default=224)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--test_interval",   type=int,   default=4)
    # Loss
    p.add_argument("--w_family",      type=float, default=0.5)
    p.add_argument("--w_proto_div",   type=float, default=0.3)
    p.add_argument("--w_proto_compact",type=float,default=0.2)
    p.add_argument("--w_heatmap",     type=float, default=0.3)
    p.add_argument("--w_attr",        type=float, default=0.1)
    p.add_argument("--w_calib",       type=float, default=0.2)
    return p.parse_args()


class PseudoLabelDataset(Dataset):
    """Dataset for pseudo-labeled test images (no family label — use 1=FS as default fake)."""

    def __init__(self, pseudo_json, crop_size=224):
        self.transform = get_train_transforms(crop_size, jpeg_prob=0.3, blur_prob=0.3)
        with open(pseudo_json) as f:
            items = json.load(f)
        self.data = []
        for item in items:
            path = item["path"]
            if os.path.exists(path):
                label = item["label"]
                # Use family: real=0, fake→FS family=1 (conservative)
                family = 0 if label == 0 else 1
                self.data.append((path, label, family))
        n_fake = sum(1 for d in self.data if d[1] == 1)
        n_real = sum(1 for d in self.data if d[1] == 0)
        print(f"PseudoDataset: {n_fake} fake + {n_real} real = {len(self.data)} total")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        path, label, family = self.data[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), label, family
        except:
            return self.__getitem__((idx + 1) % len(self.data))


class LabelSmoothBCE(nn.Module):
    def __init__(self, s=0.05):
        super().__init__()
        self.s = s
    def forward(self, logit, target):
        t = target.float() * (1 - self.s) + 0.5 * self.s
        return F.binary_cross_entropy_with_logits(logit.squeeze(-1), t)


def validate(model, val_loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lab, _ in val_loader:
            out = model(imgs.to(device), return_heatmap=False)
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
        if not os.path.isdir(split_dir): continue
        sp, sl = [], []
        pg = {}
        for jf in sorted(f for f in os.listdir(split_dir) if f.endswith(".json")):
            gen = jf.replace(".json", "")
            ds = HydraFakeTestDataset(os.path.join(split_dir, jf), args.data_root,
                                      args.image_root, args.crop_size)
            if len(ds) == 0: continue
            loader = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            gp, gl = [], []
            with torch.no_grad():
                for imgs, lab in loader:
                    out = model(imgs.to(device), return_heatmap=False)
                    gp.extend(out["confidence"].squeeze(-1).cpu().tolist())
                    gl.extend(lab.tolist())
            gp_np, gl_np = np.array(gp), np.array(gl)
            g_acc = accuracy_score(gl_np, (gp_np > 0.5).astype(int))
            g_ap  = average_precision_score(gl_np, gp_np) if len(np.unique(gl_np)) > 1 else -1
            pg[gen] = {"acc": float(g_acc), "ap": float(g_ap), "n": len(ds)}
            sp.extend(gp); sl.extend(gl)
            print(f"    {gen:20s}: Acc={g_acc*100:.1f}%  AP={g_ap:.4f}  n={len(ds)}")
        if sp:
            s_np, l_np = np.array(sp), np.array(sl)
            s_acc = accuracy_score(l_np, (s_np > 0.5).astype(int))
            s_ap  = average_precision_score(l_np, s_np) if len(np.unique(l_np)) > 1 else -1
            results[split] = {"acc": float(s_acc), "ap": float(s_ap),
                               "n": len(sp), "per_generator": pg}
            print(f"  {split.upper()}: Acc={s_acc*100:.1f}%  AP={s_ap:.4f}")
    if results:
        avg_acc = float(np.mean([r["acc"] for r in results.values()]))
        avg_ap  = float(np.mean([r["ap"]  for r in results.values() if r["ap"] >= 0]))
        results["average"] = {"acc": avg_acc, "ap": avg_ap}
        print(f"\n  *** AVERAGE: Acc={avg_acc*100:.2f}%  AP={avg_ap:.4f} ***")
    return results


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(exp_dir, "config.json"), "w"), indent=2)

    # Model
    print(f"Loading model from: {args.pretrained}")
    model = XGenDet()
    ckpt = torch.load(args.pretrained, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    raw_model = model.to(device)

    use_dp = n_gpus > 1
    if use_dp:
        model = nn.DataParallel(model)
        args.batch_size *= n_gpus
        print(f"DataParallel on {n_gpus} GPUs, effective batch={args.batch_size}")

    # Datasets
    orig_ds = HydraFakeDataset(args.train_json, args.data_root, is_train=True,
                                crop_size=args.crop_size)
    pseudo_ds = PseudoLabelDataset(args.pseudo_json, args.crop_size)
    val_ds   = HydraFakeDataset(args.val_json, args.data_root, is_train=False,
                                 crop_size=args.crop_size)

    # Build weighted sampler: orig=1.0, pseudo=pseudo_weight
    orig_weights   = [1.0] * len(orig_ds)
    pseudo_weights = [args.pseudo_weight] * len(pseudo_ds)
    all_weights = orig_weights + pseudo_weights
    combined_ds = ConcatDataset([orig_ds, pseudo_ds])
    sampler = WeightedRandomSampler(all_weights, len(combined_ds), replacement=True)

    train_loader = DataLoader(combined_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"Combined dataset: {len(orig_ds)} orig + {len(pseudo_ds)} pseudo "
          f"= {len(combined_ds)} total")

    # Loss + optimizer (very low LR — fine-grained adaptation)
    criterion = XGenDetLoss(w_family=args.w_family, w_proto_div=args.w_proto_div,
                             w_proto_compact=args.w_proto_compact, w_heatmap=args.w_heatmap,
                             w_attr=args.w_attr, w_calib=args.w_calib)
    ls_bce = LabelSmoothBCE(args.label_smoothing)

    param_groups = raw_model.get_trainable_params()
    opt_params = []
    for g in param_groups:
        lr_s = float(g.get("lr_scale", 1.0))
        name = g.get("name", "")
        lr = float(args.lr_ln) if "layer_norm" in name else float(args.lr) * lr_s
        opt_params.append({"params": g["params"], "lr": lr, "name": name})
    optimizer = torch.optim.AdamW(opt_params, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-9)
    proto_mod = raw_model.prototype_module if hasattr(raw_model, "prototype_module") else None

    print(f"\n{'='*60}")
    print(f"Self-Training | LR={args.lr} | Epochs={args.epochs} | Pseudo×{args.pseudo_weight}")
    print(f"{'='*60}\n")

    best_ap, patience_ctr = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses_acc = defaultdict(float)
        n_batch = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True, ncols=120)
        for imgs, labels, families in pbar:
            imgs, labels, families = imgs.to(device), labels.to(device), families.to(device)
            out = model(imgs, return_heatmap=not use_dp)
            losses = criterion(out, labels, families, prototype_module=proto_mod)
            losses["cls_smooth"] = ls_bce(out["binary_logit"], labels)
            losses["total"] = (losses["cls_smooth"]
                               + criterion.w_family       * losses["family"]
                               + criterion.w_proto_div    * losses["proto_div"]
                               + criterion.w_proto_compact* losses["proto_compact"]
                               + criterion.w_heatmap      * losses["heatmap"]
                               + criterion.w_attr         * losses["attr"]
                               + criterion.w_calib        * losses["calib"])
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            for k in losses: losses_acc[k] += losses[k].item()
            n_batch += 1
            pbar.set_postfix(loss=f"{losses_acc['total']/n_batch:.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        et = time.time() - t0
        for k in losses_acc: losses_acc[k] /= max(n_batch, 1)
        print(f"\nEpoch {epoch}/{args.epochs} ({et:.0f}s) Loss={losses_acc['total']:.4f}")

        vm = validate(raw_model, val_loader, device)
        print(f"  Val: AP={vm['ap']:.4f}  Acc={vm['acc']*100:.2f}%")

        # Intermediate test eval
        if epoch % args.test_interval == 0:
            print(f"\n  Running test eval @ epoch {epoch}...")
            tr = evaluate_test(raw_model, args, device)
            json.dump(tr, open(os.path.join(exp_dir, f"test_results_epoch_{epoch}.json"), "w"), indent=2)

        if vm["ap"] > best_ap:
            best_ap = vm["ap"]; patience_ctr = 0
            torch.save({"epoch": epoch, "model_state_dict": raw_model.state_dict(),
                        "val_ap": vm["ap"]},
                       os.path.join(exp_dir, "best_model.pth"))
            print(f"  *** New best AP={best_ap:.4f} ***")
        else:
            patience_ctr += 1
            print(f"  Patience: {patience_ctr}/{args.patience}")
        if patience_ctr >= args.patience:
            print(f"Early stopping at epoch {epoch}"); break

    # Final test eval on best model
    print(f"\n{'='*60}\nFinal test evaluation on best model\n{'='*60}")
    ck = torch.load(os.path.join(exp_dir, "best_model.pth"), map_location=device)
    raw_model.load_state_dict(ck["model_state_dict"])
    results = evaluate_test(raw_model, args, device)
    json.dump(results, open(os.path.join(exp_dir, "test_results.json"), "w"), indent=2)
    print(f"Saved → {exp_dir}/test_results.json")


if __name__ == "__main__":
    main()
