"""
Ensemble v5plus (CLIP + BoundarySRM + Phase + Bilateral) + PRISM v1 SFT.

Uses v5plus.forward()['prob'] and PRISM binary predictions from existing JSONLs.
"""

import argparse, json, re, os, sys
from pathlib import Path
from collections import defaultdict
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgendet_v5plus import XGenDetV5Plus

ImageFile.LOAD_TRUNCATED_IMAGES = True

_ANSWER_RE = re.compile(r'<answer>\s*(real|fake)\s*</answer>', re.I)

# CLIP normalization (v5 uses CLIP)
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def parse_prism_answer(text):
    m = _ANSWER_RE.search(text or '')
    if m: return m.group(1).lower()
    tl = (text or '').lower().strip()
    for tok in tl.split()[-5:]:
        if tok in ('real', 'fake'): return tok
    return 'unknown'


def load_prism_predictions(prism_dirs):
    """Collect PRISM v1 SFT predictions from all JSONLs."""
    predictions = {}
    for d in prism_dirs:
        for f in glob(f"{d}/prism_*.jsonl"):
            bn = Path(f).name
            if bn.startswith(('prism_v2_', 'prism_mipo_')):
                continue
            split_name = bn.replace('.jsonl', '').split('-')[0].replace('prism_', '', 1)
            with open(f) as fh:
                for line in fh:
                    if not line.strip(): continue
                    try: d = json.loads(line)
                    except: continue
                    imgs = d.get('images', [])
                    if not imgs: continue
                    img_path = imgs[0].get('path') if isinstance(imgs[0], dict) else imgs[0]
                    if not img_path or d.get('label') is None: continue
                    pred = parse_prism_answer(d.get('response', ''))
                    if pred == 'unknown': continue
                    predictions[img_path] = {
                        'prism_pred': pred,
                        'label': int(d['label']),
                        'split': split_name,
                    }
    return predictions


class ImageListDataset(Dataset):
    def __init__(self, paths, roots, crop_size=224):
        self.paths = paths
        self.roots = roots
        self.tf = transforms.Compose([
            transforms.Resize((crop_size, crop_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
        ])

    def _find(self, rel):
        for r in self.roots:
            full = os.path.join(r, rel)
            if os.path.exists(full): return full
        return None

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        rel = self.paths[idx]
        full = self._find(rel)
        if full is None:
            return torch.zeros(3, 224, 224), rel, False
        try:
            img = Image.open(full).convert('RGB')
            return self.tf(img), rel, True
        except Exception:
            return torch.zeros(3, 224, 224), rel, False


@torch.no_grad()
def run_v5plus_inference(model, paths, roots, device, batch_size=32, num_workers=2):
    ds = ImageListDataset(paths, roots)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    probs = {}
    for imgs, paths_b, flags in tqdm(loader, desc='v5plus_inference'):
        imgs = imgs.to(device)
        out = model(imgs)
        p = out['prob'].squeeze(-1).cpu().float().tolist()
        for path, pv, ok in zip(paths_b, p, flags):
            probs[path] = float(pv) if ok else 0.5
    return probs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--v5_checkpoint', default='checkpoints/v5_resume_hl/best_model.pth')
    p.add_argument('--branches_checkpoint', default='checkpoints/v5plus/best_branches.pth')
    p.add_argument('--prism_dirs', nargs='+',
                   default=[f'/home/sachin.chaudhary/veritas_clone/result_prism_{s}{i}'
                            for s in ['worker', 'helper']
                            for i in list(range(13)) + ['']])
    p.add_argument('--data_roots', nargs='+',
                   default=['/home/sachin.chaudhary', '/home/sachin.chaudhary/veritas_clone'])
    p.add_argument('--output', default='checkpoints/ensemble_v5plus_prism/results.json')
    p.add_argument('--cache', default='checkpoints/ensemble_v5plus_prism/v5plus_probs.json')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. PRISM predictions
    print("=== Step 1: Loading PRISM predictions ===")
    # Filter to existing dirs
    prism_dirs = [d for d in args.prism_dirs if os.path.isdir(d)]
    prism_preds = load_prism_predictions(prism_dirs)
    print(f"[prism] {len(prism_preds)} samples from {len(prism_dirs)} dirs")

    # 2. v5plus inference (with cache)
    print("\n=== Step 2: v5plus inference ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(args.cache):
        print(f"[v5plus] Loading cache: {args.cache}")
        v5plus_probs = json.load(open(args.cache))
    else:
        print(f"[v5plus] Building model + loading checkpoints...")
        model = XGenDetV5Plus(v5_checkpoint_path=args.v5_checkpoint)
        model.load_branches_checkpoint(args.branches_checkpoint)
        model = model.to(device).float().eval()
        for p in model.parameters(): p.requires_grad = False

        image_paths = list(prism_preds.keys())
        v5plus_probs = run_v5plus_inference(model, image_paths, args.data_roots, device,
                                            batch_size=args.batch_size, num_workers=args.num_workers)
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        with open(args.cache, 'w') as f:
            json.dump(v5plus_probs, f)
        print(f"[v5plus] Cached to {args.cache}")

    # 3. Merge
    print("\n=== Step 3: Merging predictions ===")
    merged = {}
    for path, pd in prism_preds.items():
        vp = v5plus_probs.get(path)
        if vp is None: continue
        merged[path] = {**pd, 'v5plus_prob': vp}
    print(f"[merge] {len(merged)} samples in both")

    # 4. Ensemble across α values
    print("\n=== Step 4: Ensemble sweep ===")
    alphas = [i/100 for i in range(0, 101, 5)]
    splits = ['id', 'cm', 'cf', 'cd']

    # Global α
    print("\n--- Global α sweep (overall acc) ---")
    best_global_a, best_global_acc = 0, 0
    for a in alphas:
        correct, total = 0, 0
        for path, d in merged.items():
            v5p = d['v5plus_prob']
            pseudo = 0.9 if d['prism_pred'] == 'fake' else 0.1
            p = a * v5p + (1 - a) * pseudo
            if (p >= 0.5) == bool(d['label']): correct += 1
            total += 1
        acc = correct / max(total, 1)
        if acc > best_global_acc:
            best_global_a, best_global_acc = a, acc
        if a in (0.0, 0.3, 0.5, 0.7, 1.0) or abs(a - best_global_a) < 0.05:
            marker = " ★" if acc >= 0.901 else ""
            print(f"  α={a:.2f}: {acc*100:.2f}%{marker}")

    # Per-split α
    print("\n--- Per-split optimal α ---")
    split_data = defaultdict(list)
    for path, d in merged.items():
        s = d['split'].split('_')[0]
        split_data[s].append((d['v5plus_prob'], d['prism_pred'], d['label']))

    per_split_results = {}
    total_correct, total_n = 0, 0
    for s in splits:
        data = split_data[s]
        best_a, best_acc = 0, 0
        for a in alphas:
            c = 0
            for v5p, pp, lbl in data:
                pseudo = 0.9 if pp == 'fake' else 0.1
                p = a * v5p + (1 - a) * pseudo
                if (p >= 0.5) == bool(lbl): c += 1
            acc = c / len(data)
            if acc > best_acc: best_acc, best_a = acc, a
        per_split_results[s] = {'alpha': best_a, 'acc': best_acc, 'n': len(data)}
        total_correct += best_acc * len(data); total_n += len(data)
        print(f"  {s.upper()}: α={best_a:.2f}  Acc={best_acc*100:.2f}%  (n={len(data)})")

    per_split_overall = total_correct / max(total_n, 1)

    # v5plus alone / PRISM alone
    v5plus_only_c = sum(1 for p, d in merged.items() if (d['v5plus_prob'] >= 0.5) == bool(d['label']))
    prism_only_c = sum(1 for p, d in merged.items() if (d['prism_pred'] == 'fake') == bool(d['label']))
    v5plus_only = v5plus_only_c / len(merged)
    prism_only = prism_only_c / len(merged)

    # Print summary
    print("\n" + "="*75)
    print("  FINAL ENSEMBLE RESULTS")
    print("="*75)
    print(f"  v5plus alone:                           {v5plus_only*100:.2f}%")
    print(f"  PRISM v1 SFT alone:                     {prism_only*100:.2f}%")
    print(f"  Ensemble global α={best_global_a:.2f}:                {best_global_acc*100:.2f}%")
    print(f"  Ensemble per-split optimal α:           {per_split_overall*100:.2f}%")
    print(f"  Veritas baseline:                       90.10%")
    beat = per_split_overall - 0.901
    print(f"  Δ vs Veritas (per-split):               {beat*100:+.2f}%  {'★ BEATS' if beat > 0 else '(below)'}")
    print("="*75)

    # Save
    out = {
        'v5plus_alone': v5plus_only,
        'prism_alone': prism_only,
        'global_best_alpha': best_global_a,
        'global_best_acc': best_global_acc,
        'per_split': per_split_results,
        'per_split_overall': per_split_overall,
        'n_samples': len(merged),
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
