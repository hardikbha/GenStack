"""
Evaluate XGenDetV5Plus on a single HydraFake test split / generator.

Usage:
    python training/eval_v5plus.py \
        --model_path     checkpoints/v5plus/best_branches.pth \
        --v5_checkpoint  checkpoints/v5_resume_hl/best_model.pth \
        --test_json      /home/sachin.chaudhary/hydrafake/jsons/test/id/FaceForensics++.json \
        --data_root      /home/sachin.chaudhary \
        --split          id_FaceForensics++
"""

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgendet_v5plus import XGenDetV5Plus
from data.hydrafake_dataset import HydraFakeDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",    required=True)
    p.add_argument("--v5_checkpoint", required=True)
    p.add_argument("--test_json",     required=True)
    p.add_argument("--data_root",     default="/home/sachin.chaudhary")
    p.add_argument("--split",         default="test")
    p.add_argument("--output_dir",    default="checkpoints/v5plus/eval")
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--num_workers",   type=int, default=2)
    p.add_argument("--threshold",     type=float, default=0.5)
    p.add_argument("--bf16",          action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32

    print(f"Evaluating split={args.split} | {args.test_json}")

    model = XGenDetV5Plus(v5_checkpoint_path=args.v5_checkpoint)
    model.load_branches_checkpoint(args.model_path)
    model = model.to(device).to(dtype).eval()

    dataset = HydraFakeDataset(
        json_path=args.test_json, data_root=args.data_root,
        is_train=False, crop_size=224,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=args.split):
            imgs, labels, _ = batch
            imgs = imgs.to(device, dtype=dtype)
            out = model(imgs)
            all_probs.extend(out["prob"].squeeze(1).cpu().float().tolist())
            all_labels.extend(labels.tolist())

    probs_t = torch.tensor(all_probs)
    labels_t = torch.tensor(all_labels, dtype=torch.float32)
    preds_t  = (probs_t >= args.threshold).float()

    tp = ((preds_t==1)&(labels_t==1)).sum().item()
    tn = ((preds_t==0)&(labels_t==0)).sum().item()
    fp = ((preds_t==1)&(labels_t==0)).sum().item()
    fn = ((preds_t==0)&(labels_t==1)).sum().item()

    acc  = (tp+tn) / max(len(all_labels), 1)
    prec = tp / max(tp+fp, 1)
    rec  = tp / max(tp+fn, 1)
    f1   = 2*prec*rec / max(prec+rec, 1e-9)

    try:
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(all_labels, all_probs)
    except Exception:
        ap = -1.0

    print(f"  {args.split:<40} Acc={acc*100:.2f}%  AP={ap:.4f}  F1={f1:.4f}  n={len(all_labels)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = dict(split=args.split, acc=acc, ap=ap, f1=f1, n=len(all_labels),
                  precision=prec, recall=rec)
    with open(out_dir / f"{args.split}_results.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
