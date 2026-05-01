"""Evaluate a trained PhysFreqNet model on a single HydraFake test JSON.

Reports accuracy (threshold=0.5), Average Precision, F1, and sample count.
Writes a JSON file with the results to <output_dir>/<split>_results.json.
"""

import sys
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.physfreq_net import PhysFreqNet
from data.hydrafake_dataset import HydraFakeDataset


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PhysFreqNet on one test JSON.")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to trained PhysFreqNet checkpoint (best_model.pth).")
    p.add_argument("--branches", type=str, required=True,
                   help="Comma separated branches, e.g. 'fft,retinex,ca'.")
    p.add_argument("--test_json", type=str, required=True)
    p.add_argument("--data_root", type=str, default="/home/sachin.chaudhary")
    p.add_argument("--split", type=str, required=True,
                   help="Tag for this split, e.g. 'id_FaceForensics++'.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def _strip_module_prefix(state_dict):
    """Strip DataParallel 'module.' prefix from state_dict keys if present."""
    if not state_dict:
        return state_dict
    needs_strip = any(k.startswith("module.") for k in state_dict.keys())
    if not needs_strip:
        return state_dict
    return {k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state_dict.items()}


def _load_physfreq(model_path: str, branches):
    """Build PhysFreqNet and try model.load_checkpoint first, fall back to manual load."""
    model = PhysFreqNet(branches=branches)

    # Try the model's own loader if provided.
    loaded = False
    if hasattr(model, "load_checkpoint"):
        try:
            model.load_checkpoint(model_path)
            loaded = True
        except Exception as e:
            print(f"[eval_physfreq] load_checkpoint failed ({e}); falling back to manual load.")

    if not loaded:
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("model_state_dict",
                                  ckpt.get("state_dict", ckpt))
        else:
            state_dict = ckpt
        state_dict = _strip_module_prefix(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[eval_physfreq] missing keys (first 5): {list(missing)[:5]}")
        if unexpected:
            print(f"[eval_physfreq] unexpected keys (first 5): {list(unexpected)[:5]}")

    return model


def main():
    args = parse_args()
    branches = tuple(b.strip() for b in args.branches.split(",") if b.strip())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    print(f"[eval_physfreq] split={args.split}  branches={branches}  "
          f"device={device}  dtype={dtype}")

    model = _load_physfreq(args.model_path, branches)
    model = model.to(device).to(dtype).eval()

    dataset = HydraFakeDataset(
        args.test_json, args.data_root,
        is_train=False, crop_size=args.crop_size,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    probs, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=args.split):
            # HydraFakeDataset returns (img, label, family)
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                imgs = batch[0]
                lbls = batch[1]
            else:
                raise RuntimeError(f"Unexpected batch structure: {type(batch)}")

            imgs = imgs.to(device, dtype=dtype, non_blocking=True)
            out = model(imgs)
            if isinstance(out, dict) and "prob" in out:
                p = out["prob"]
            elif isinstance(out, dict) and "logit" in out:
                p = torch.sigmoid(out["logit"])
            elif torch.is_tensor(out):
                p = torch.sigmoid(out) if out.dtype.is_floating_point else out.float()
            else:
                raise RuntimeError(f"Unrecognised model output: {type(out)}")

            if p.dim() > 1:
                p = p.squeeze(-1)
            probs.extend(p.detach().cpu().float().tolist())
            labels.extend(lbls.tolist())

    probs_t = torch.tensor(probs)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    preds_t = (probs_t >= 0.5).float()

    tp = ((preds_t == 1) & (labels_t == 1)).sum().item()
    tn = ((preds_t == 0) & (labels_t == 0)).sum().item()
    fp = ((preds_t == 1) & (labels_t == 0)).sum().item()
    fn = ((preds_t == 0) & (labels_t == 1)).sum().item()

    n = len(labels)
    acc = (tp + tn) / max(n, 1)
    f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1)

    try:
        from sklearn.metrics import average_precision_score
        # AP requires both classes present.
        if len(set(labels)) >= 2:
            ap = float(average_precision_score(labels, probs))
        else:
            ap = -1.0
    except Exception as e:
        print(f"[eval_physfreq] AP computation failed: {e}")
        ap = -1.0

    result = {
        "split": args.split,
        "branches": list(branches),
        "acc": acc,
        "ap": ap,
        "f1": f1,
        "n": n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "model_path": args.model_path,
        "test_json": args.test_json,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.split}_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  {args.split:<40} Acc={acc*100:.2f}%  AP={ap:.4f}  "
          f"F1={f1:.4f}  n={n}")
    print(f"[eval_physfreq] wrote {out_file}")
    return result


if __name__ == "__main__":
    main()
