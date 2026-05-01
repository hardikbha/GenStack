"""Ensemble evaluation: XGenDet v5  +  PhysFreqNet (full / configurable).

For each batch:
    prob_v5 = sigmoid(v5(x)['binary_logit'])
    prob_pf = physfreq(x)['prob']
    final   = w_v5 * prob_v5 + w_pf * prob_pf
Metrics at threshold 0.5: acc, AP, F1.
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
from models.xgendet import XGenDet
from data.hydrafake_dataset import HydraFakeDataset


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble v5 + PhysFreqNet on one test JSON.")
    p.add_argument("--v5_checkpoint", type=str, required=True)
    p.add_argument("--physfreq_checkpoint", type=str, required=True)
    p.add_argument("--branches", type=str, default="fft,retinex,ca")
    p.add_argument("--test_json", type=str, required=True)
    p.add_argument("--data_root", type=str, default="/home/sachin.chaudhary")
    p.add_argument("--split", type=str, required=True)
    p.add_argument("--weight_v5", type=float, default=0.55)
    p.add_argument("--weight_physfreq", type=float, default=0.45)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--bf16", action="store_true")
    # v5 construction hyper-params (kept at v5 defaults)
    p.add_argument("--clip_model_name", type=str, default="ViT-L/14")
    p.add_argument("--num_prompt_tokens", type=int, default=8)
    p.add_argument("--num_prototypes", type=int, default=128)
    p.add_argument("--proto_dim", type=int, default=128)
    p.add_argument("--shuffle_patch_size", type=int, default=32)
    return p.parse_args()


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state_dict.items()}


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        return ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    return ckpt


def _build_v5(args, device, dtype):
    v5 = XGenDet(
        clip_model_name=args.clip_model_name,
        num_prompt_tokens=args.num_prompt_tokens,
        num_prototypes=args.num_prototypes,
        proto_dim=args.proto_dim,
        shuffle_patch_size=args.shuffle_patch_size,
    )
    ckpt = torch.load(args.v5_checkpoint, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    state_dict = _strip_module_prefix(state_dict)
    missing, unexpected = v5.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[ensemble] v5 missing keys (first 5): {list(missing)[:5]}")
    if unexpected:
        print(f"[ensemble] v5 unexpected keys (first 5): {list(unexpected)[:5]}")
    # v5 (CLIP ViT) breaks in BFloat16 due to layer_norm type checks — keep in float32
    v5 = v5.to(device).to(torch.float32).eval()
    for p in v5.parameters():
        p.requires_grad = False
    return v5


def _build_physfreq(args, branches, device, dtype):
    model = PhysFreqNet(branches=branches)
    loaded = False
    if hasattr(model, "load_checkpoint"):
        try:
            model.load_checkpoint(args.physfreq_checkpoint)
            loaded = True
        except Exception as e:
            print(f"[ensemble] physfreq load_checkpoint failed ({e}); manual load.")
    if not loaded:
        ckpt = torch.load(args.physfreq_checkpoint, map_location="cpu")
        state_dict = _extract_state_dict(ckpt)
        state_dict = _strip_module_prefix(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[ensemble] physfreq missing keys (first 5): {list(missing)[:5]}")
        if unexpected:
            print(f"[ensemble] physfreq unexpected keys (first 5): {list(unexpected)[:5]}")
    model = model.to(device).to(dtype).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _v5_prob(v5_out):
    """Extract binary prob from XGenDet forward output."""
    if isinstance(v5_out, dict):
        if "binary_prob" in v5_out:
            return v5_out["binary_prob"]
        if "binary_logit" in v5_out:
            return torch.sigmoid(v5_out["binary_logit"])
        if "logit" in v5_out:
            return torch.sigmoid(v5_out["logit"])
        if "prob" in v5_out:
            return v5_out["prob"]
    if torch.is_tensor(v5_out):
        return torch.sigmoid(v5_out)
    raise RuntimeError(f"Unrecognised v5 output: {type(v5_out)}")


def _pf_prob(pf_out):
    if isinstance(pf_out, dict):
        if "prob" in pf_out:
            return pf_out["prob"]
        if "logit" in pf_out:
            return torch.sigmoid(pf_out["logit"])
    if torch.is_tensor(pf_out):
        return torch.sigmoid(pf_out)
    raise RuntimeError(f"Unrecognised physfreq output: {type(pf_out)}")


def main():
    args = parse_args()
    branches = tuple(b.strip() for b in args.branches.split(",") if b.strip())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    w_v5 = float(args.weight_v5)
    w_pf = float(args.weight_physfreq)
    total_w = w_v5 + w_pf
    if total_w <= 0:
        raise ValueError("weight_v5 + weight_physfreq must be > 0")
    # Normalise (safety; spec defaults already sum to 1.0).
    w_v5, w_pf = w_v5 / total_w, w_pf / total_w

    print(f"[ensemble] split={args.split}  branches={branches}  "
          f"w_v5={w_v5:.3f}  w_pf={w_pf:.3f}  dtype={dtype}")

    v5 = _build_v5(args, device, dtype)
    pf = _build_physfreq(args, branches, device, dtype)

    dataset = HydraFakeDataset(
        args.test_json, args.data_root,
        is_train=False, crop_size=args.crop_size,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    probs_v5, probs_pf, probs_final, labels = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=args.split):
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                imgs = batch[0]
                lbls = batch[1]
            else:
                raise RuntimeError(f"Unexpected batch structure: {type(batch)}")

            imgs = imgs.to(device, non_blocking=True)

            # v5 always in float32 (CLIP LayerNorm needs it)
            v5_out = v5(imgs.float())
            # physfreq uses requested dtype (bf16 or float32)
            pf_out = pf(imgs.to(dtype=dtype))

            pv = _v5_prob(v5_out).float()
            pp = _pf_prob(pf_out).float()
            if pv.dim() > 1:
                pv = pv.squeeze(-1)
            if pp.dim() > 1:
                pp = pp.squeeze(-1)
            pv = pv.clamp(0.0, 1.0)
            pp = pp.clamp(0.0, 1.0)

            final = w_v5 * pv + w_pf * pp

            probs_v5.extend(pv.detach().cpu().tolist())
            probs_pf.extend(pp.detach().cpu().tolist())
            probs_final.extend(final.detach().cpu().tolist())
            labels.extend(lbls.tolist())

    probs_t = torch.tensor(probs_final)
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
        if len(set(labels)) >= 2:
            ap = float(average_precision_score(labels, probs_final))
            ap_v5 = float(average_precision_score(labels, probs_v5))
            ap_pf = float(average_precision_score(labels, probs_pf))
        else:
            ap = ap_v5 = ap_pf = -1.0
    except Exception as e:
        print(f"[ensemble] AP computation failed: {e}")
        ap = ap_v5 = ap_pf = -1.0

    result = {
        "split": args.split,
        "branches": list(branches),
        "weight_v5": w_v5,
        "weight_physfreq": w_pf,
        "acc": acc,
        "ap": ap,
        "ap_v5_only": ap_v5,
        "ap_physfreq_only": ap_pf,
        "f1": f1,
        "n": n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "v5_checkpoint": args.v5_checkpoint,
        "physfreq_checkpoint": args.physfreq_checkpoint,
        "test_json": args.test_json,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.split}_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  {args.split:<40} Acc={acc*100:.2f}%  AP={ap:.4f}  "
          f"F1={f1:.4f}  n={n}  (v5 AP={ap_v5:.4f}  pf AP={ap_pf:.4f})")
    print(f"[ensemble] wrote {out_file}")
    return result


if __name__ == "__main__":
    main()
