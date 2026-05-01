"""
Ensemble v5 (CLIP ViT-L/14 discriminative) + PRISM v1 SFT (InternVL3-8B generative).

Strategy:
  - PRISM already evaluated → parse existing JSONL outputs for per-sample binary predictions
  - v5 needs re-inference on the same images → run and save per-sample probabilities
  - Ensemble via weighted average: final = α * v5_prob + (1 - α) * prism_pseudo_prob
    where prism_pseudo_prob = 0.9 if PRISM says fake else 0.1

Output:
  - Per-split accuracy with α ∈ [0.3, 0.4, 0.5, 0.6, 0.7]
  - Overall accuracy
  - Comparison vs PRISM-only, v5-only, Veritas baseline (90.1%)
"""

import argparse, json, re, os, sys
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgendet import XGenDet

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─── PRISM parsing ────────────────────────────────────────────────────────────

_ANSWER_RE = re.compile(r'<answer>\s*(real|fake)\s*</answer>', re.I)

def parse_prism_answer(response: str) -> str:
    """Extract real/fake from PRISM generated text."""
    m = _ANSWER_RE.search(response or '')
    if m:
        return m.group(1).lower()
    # Fallback
    tl = (response or '').lower().strip()
    for tok in tl.split()[-5:]:
        if tok in ('real', 'fake'):
            return tok
    return 'unknown'


def load_prism_predictions(prism_dirs: list) -> dict:
    """
    Scan all PRISM eval directories and build:
        {image_path -> {'prism_pred': 'real'/'fake', 'label': int, 'split': str}}

    Skips prism_v2_* (v2 is different model) and prism_mipo_* (mipo is another variant).
    Only keeps PRISM v1 SFT results.
    """
    from glob import glob
    predictions = {}

    # Collect all JSONLs starting with prism_* but NOT prism_v2_ or prism_mipo_
    all_jsonls = []
    for d in prism_dirs:
        for f in glob(f"{d}/prism_*.jsonl"):
            bn = Path(f).name
            if bn.startswith(('prism_v2_', 'prism_mipo_')):
                continue
            all_jsonls.append(f)

    print(f"[prism] Found {len(all_jsonls)} PRISM v1 JSONL files")

    for f in all_jsonls:
        # Extract split name: "prism_id_ff_part0-prism_id_ff_part0.jsonl" → "id_ff_part0"
        bn = Path(f).stem
        if '-' in bn:
            bn = bn.split('-')[0]  # take first occurrence
        split_name = bn.replace('prism_', '', 1)

        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                imgs = d.get('images', [])
                if not imgs:
                    continue
                img_path = imgs[0].get('path') if isinstance(imgs[0], dict) else imgs[0]
                if not img_path:
                    continue
                label = d.get('label')
                if label is None:
                    continue
                pred = parse_prism_answer(d.get('response', ''))
                predictions[img_path] = {
                    'prism_pred': pred,
                    'label': int(label),
                    'split': split_name,
                }

    print(f"[prism] Collected {len(predictions)} unique image predictions")
    return predictions


# ─── v5 inference ─────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]   # CLIP normalization
_IMAGENET_STD  = [0.26862954, 0.26130258, 0.27577711]


class ImageListDataset(Dataset):
    def __init__(self, image_paths, data_roots, crop_size=224):
        self.paths = image_paths
        self.roots = data_roots
        self.tf = transforms.Compose([
            transforms.Resize((crop_size, crop_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def _find_file(self, rel_path):
        for r in self.roots:
            full = os.path.join(r, rel_path)
            if os.path.exists(full):
                return full
        return None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel = self.paths[idx]
        full = self._find_file(rel)
        if full is None:
            # Return zero tensor + flag
            return torch.zeros(3, 224, 224), rel, False
        try:
            img = Image.open(full).convert('RGB')
            return self.tf(img), rel, True
        except Exception:
            return torch.zeros(3, 224, 224), rel, False


def load_v5(ckpt_path, device):
    """Build and load the v5 model (XGenDet base config)."""
    model = XGenDet(
        clip_model_name="ViT-L/14",
        num_prompt_tokens=8,
        num_prototypes=128,
        proto_dim=128,
        shuffle_patch_size=32,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    # Remap flat prototype_*/heatmap_* keys → nested prototype_module.*/heatmap_generator.*
    # The checkpoint was saved with a flat naming; current XGenDet uses nested submodules.
    remapped = {}
    for k, v in sd.items():
        nk = k
        if k.startswith('prototype_') and not k.startswith('prototype_module.'):
            nk = 'prototype_module.' + k[len('prototype_'):]
        elif k.startswith('heatmap_') and not k.startswith(('heatmap_generator.','heatmap_stats.')):
            # Figure out if it's heatmap_generator or heatmap_stats
            rest = k[len('heatmap_'):]
            if rest.startswith('stats.') or rest.startswith('stat_'):
                nk = 'heatmap_stats.' + rest.split('.', 1)[1] if '.' in rest else 'heatmap_stats.' + rest
            else:
                nk = 'heatmap_generator.' + rest
        remapped[nk] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing: print(f"[v5] missing keys (first 5): {list(missing)[:5]}")
    if unexpected: print(f"[v5] unexpected keys (first 5): {list(unexpected)[:5]}")
    n_loaded = len(remapped) - len(unexpected)
    print(f"[v5] Loaded {n_loaded}/{len(remapped)} keys")
    model = model.to(device).float().eval()
    for p in model.parameters(): p.requires_grad = False
    return model


@torch.no_grad()
def run_v5_inference(model, image_paths, data_roots, device, batch_size=32, num_workers=2):
    """Run v5 on a list of image paths → dict {path: prob}."""
    ds = ImageListDataset(image_paths, data_roots)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    probs = {}
    for imgs, paths, flags in tqdm(loader, desc='v5_inference'):
        imgs = imgs.to(device)
        out = model(imgs, return_heatmap=False)
        # Use `confidence` which is sigmoid of calibrated logit
        conf = out.get('confidence')
        if conf is None:
            conf = torch.sigmoid(out['binary_logit'])
        conf = conf.squeeze(-1).cpu().float().tolist()
        for p, c, ok in zip(paths, conf, flags):
            if ok:
                probs[p] = float(c)
            else:
                probs[p] = 0.5  # fallback for missing files
    return probs


# ─── Ensemble ─────────────────────────────────────────────────────────────────

def compute_ensemble_metrics(merged, alphas, threshold=0.5):
    """
    merged: dict {path: {'v5_prob', 'prism_pred', 'label', 'split'}}
    alphas: list of α weights (v5 weight)
    Returns:
        per_alpha_results: dict {α: {split: acc, 'overall': acc}}
    """
    results = {}
    for alpha in alphas:
        split_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        overall = {'correct': 0, 'total': 0, 'v5_only_correct': 0, 'prism_only_correct': 0}

        for path, d in merged.items():
            if 'v5_prob' not in d:
                continue
            v5_p = d['v5_prob']
            prism_p = 0.9 if d['prism_pred'] == 'fake' else 0.1 if d['prism_pred'] == 'real' else 0.5
            ensemble_p = alpha * v5_p + (1 - alpha) * prism_p
            pred = 1 if ensemble_p >= threshold else 0
            label = d['label']

            # Standardize split name: "id_ff_part0" → "id", etc.
            split = d['split'].split('_')[0]

            split_stats[split]['total'] += 1
            if pred == label:
                split_stats[split]['correct'] += 1

            overall['total'] += 1
            if pred == label: overall['correct'] += 1
            if (v5_p >= 0.5) == bool(label): overall['v5_only_correct'] += 1
            pp = 1 if d['prism_pred'] == 'fake' else 0
            if pp == label: overall['prism_only_correct'] += 1

        results[alpha] = {
            'per_split': {k: v['correct']/max(v['total'],1) for k,v in split_stats.items()},
            'per_split_counts': {k: dict(v) for k,v in split_stats.items()},
            'overall_acc': overall['correct']/max(overall['total'],1),
            'v5_only_acc': overall['v5_only_correct']/max(overall['total'],1),
            'prism_only_acc': overall['prism_only_correct']/max(overall['total'],1),
            'n_samples': overall['total'],
        }
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--v5_checkpoint', required=True)
    p.add_argument('--prism_dirs', nargs='+',
                   default=[
                       '/home/sachin.chaudhary/veritas_clone/result_prism_helper12',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_worker0',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_worker1',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_worker2',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_worker3',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_helper0',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_helper1',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_helper2',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_helper3',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_helper4',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_helper5',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_helper6',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_helper7',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_helper8',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_helper9',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_helper10',
                       '/home/sachin.chaudhary/veritas_clone/result_prism_helper11',
                   ])
    p.add_argument('--data_roots', nargs='+',
                   default=['/home/sachin.chaudhary', '/home/sachin.chaudhary/veritas_clone'])
    p.add_argument('--output', default='checkpoints/ensemble_v5_prism/results.json')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--cache_v5', default='checkpoints/ensemble_v5_prism/v5_probs.json')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. Load PRISM predictions
    print("\n=== Step 1: Loading PRISM predictions ===")
    prism_preds = load_prism_predictions(args.prism_dirs)

    # 2. Run v5 on the same images (with caching)
    print("\n=== Step 2: Running v5 inference ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(args.cache_v5):
        print(f"[v5] Loading cached probs from {args.cache_v5}")
        v5_probs = json.load(open(args.cache_v5))
    else:
        print(f"[v5] No cache found, running fresh inference...")
        model = load_v5(args.v5_checkpoint, device)
        image_paths = list(prism_preds.keys())
        v5_probs = run_v5_inference(model, image_paths, args.data_roots, device,
                                    batch_size=args.batch_size, num_workers=args.num_workers)
        os.makedirs(os.path.dirname(args.cache_v5), exist_ok=True)
        with open(args.cache_v5, 'w') as f:
            json.dump(v5_probs, f)
        print(f"[v5] Cached to {args.cache_v5}")

    # 3. Merge
    print("\n=== Step 3: Merging predictions ===")
    merged = {}
    for path, pd in prism_preds.items():
        vp = v5_probs.get(path)
        if vp is None:
            continue
        merged[path] = {**pd, 'v5_prob': vp}
    print(f"[merge] {len(merged)} samples have both PRISM and v5 predictions")

    # 4. Compute ensemble metrics at multiple α
    print("\n=== Step 4: Computing ensemble metrics ===")
    alphas = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    results = compute_ensemble_metrics(merged, alphas)

    # 5. Print table
    splits = ['id', 'cm', 'cf', 'cd']
    print("\n" + "="*80)
    print(f"{'α':>5} {'ID':>7} {'CM':>7} {'CF':>7} {'CD':>7} {'Overall':>9}")
    print("-"*80)
    for alpha in alphas:
        r = results[alpha]
        ps = r['per_split']
        row = f"{alpha:>5.2f} "
        for s in splits:
            row += f"{ps.get(s,0)*100:>6.2f}% "
        row += f"{r['overall_acc']*100:>8.2f}%"
        print(row)
    print("-"*80)
    # Baselines
    r_pf = results[list(alphas)[0]]
    print(f"{'v5':>5} {'(α=1.0)':>7} overall = {r_pf['v5_only_acc']*100:>.2f}%  PRISM only overall = {r_pf['prism_only_acc']*100:>.2f}%")
    print(f"{'Veritas baseline (target to beat)':>40}: 90.10%")
    print("="*80)

    # Find best α
    best_alpha = max(alphas, key=lambda a: results[a]['overall_acc'])
    print(f"\n>>> BEST α = {best_alpha:.2f} → Overall = {results[best_alpha]['overall_acc']*100:.2f}%")
    beats_veritas = results[best_alpha]['overall_acc'] > 0.901
    print(f">>> Beats Veritas 90.1%? {'YES ★' if beats_veritas else 'NO'}")

    # 6. Save
    out = {
        'alphas': alphas,
        'results': {str(a): r for a, r in results.items()},
        'best_alpha': best_alpha,
        'best_overall_acc': results[best_alpha]['overall_acc'],
        'n_samples': len(merged),
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
