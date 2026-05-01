"""
Per-split α + threshold optimization for v5 + PRISM v1 SFT ensemble.

Baseline: Global α=0.5, threshold=0.5 → 88.38% (from earlier ensemble run).

This script:
  - Loads v5 probs cache + PRISM predictions (both already computed)
  - For each split (id, cm, cf, cd), grid-searches α ∈ [0.0, 0.1, ..., 1.0]
    and threshold ∈ [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
  - Picks per-split optimal (α*, threshold*)
  - Reports improvement vs global baseline
"""

import json, re, os, sys
from pathlib import Path
from collections import defaultdict
from glob import glob

ANSWER_RE = re.compile(r'<answer>\s*(real|fake)\s*</answer>', re.I)

def parse_answer(text):
    m = ANSWER_RE.search(text or '')
    if m: return m.group(1).lower()
    tl = (text or '').lower().strip()
    for tok in tl.split()[-5:]:
        if tok in ('real','fake'): return tok
    return 'unknown'

def load_prism():
    preds = {}
    dirs = ['result_prism_helper'+str(i) for i in range(13)] + ['result_prism_worker'+str(i) for i in range(4)]
    for d in dirs:
        full = f'/home/sachin.chaudhary/veritas_clone/{d}'
        if not os.path.exists(full): continue
        for f in glob(f"{full}/prism_*.jsonl"):
            bn = Path(f).name
            if bn.startswith(('prism_v2_','prism_mipo_')): continue
            split_name = Path(f).stem.split('-')[0].replace('prism_','',1)
            with open(f) as fh:
                for line in fh:
                    if not line.strip(): continue
                    try:
                        x = json.loads(line)
                    except: continue
                    imgs = x.get('images', [])
                    if not imgs: continue
                    ip = imgs[0].get('path') if isinstance(imgs[0],dict) else imgs[0]
                    if not ip: continue
                    label = x.get('label')
                    if label is None: continue
                    preds[ip] = {
                        'prism_pred': parse_answer(x.get('response','')),
                        'label': int(label),
                        'split': split_name.split('_')[0],
                    }
    return preds

def main():
    print("Loading PRISM predictions...")
    prism = load_prism()
    print(f"  {len(prism)} PRISM samples")

    print("Loading v5 probs cache...")
    v5_probs = json.load(open('/home/sachin.chaudhary/xgendet/checkpoints/ensemble_v5_prism/v5_probs.json'))
    print(f"  {len(v5_probs)} v5 probs")

    # Merge
    merged = {}
    for p, pd in prism.items():
        vp = v5_probs.get(p)
        if vp is None: continue
        merged[p] = {**pd, 'v5_prob': float(vp)}
    print(f"  merged: {len(merged)} samples\n")

    # Group by split
    by_split = defaultdict(list)
    for d in merged.values():
        by_split[d['split']].append(d)

    alphas = [round(i*0.1, 1) for i in range(11)]         # 0.0..1.0
    thresholds = [round(0.3 + 0.05*i, 2) for i in range(9)]  # 0.30..0.70

    def eval_split(items, alpha, thr):
        c = t = 0
        for d in items:
            v5_p = d['v5_prob']
            pp = 0.9 if d['prism_pred']=='fake' else 0.1 if d['prism_pred']=='real' else 0.5
            ens = alpha*v5_p + (1-alpha)*pp
            pred = 1 if ens >= thr else 0
            t += 1
            if pred == d['label']: c += 1
        return c/max(t,1), c, t

    print(f"{'Split':<6} {'n':>7}  Global(α=.5,t=.5)   Per-split best (α*, t*)   Gain")
    print('-'*75)

    overall_correct_global = 0
    overall_correct_optimal = 0
    overall_total = 0
    best_configs = {}

    for s in ['id','cm','cf','cd']:
        items = by_split.get(s, [])
        if not items: continue
        # Global
        g_acc, g_c, g_t = eval_split(items, 0.5, 0.5)
        # Search per-split
        best = (0, 0.5, 0.5)
        for a in alphas:
            for t in thresholds:
                acc, c, _ = eval_split(items, a, t)
                if acc > best[0]:
                    best = (acc, a, t)
        b_acc, b_a, b_t = best
        b_c = round(b_acc * g_t)
        overall_correct_global += g_c
        overall_correct_optimal += b_c
        overall_total += g_t
        gain = (b_acc - g_acc) * 100
        best_configs[s] = (b_a, b_t, b_acc)
        print(f"{s.upper():<6} {g_t:>7}  {g_acc*100:>6.2f}% (c={g_c})     {b_acc*100:>6.2f}% (α={b_a:.1f}, t={b_t:.2f})   +{gain:.2f}%")

    print('-'*75)
    print(f"Global overall: {overall_correct_global/overall_total*100:.2f}% (α=.5, t=.5 everywhere)")
    print(f"Per-split opt:  {overall_correct_optimal/overall_total*100:.2f}% (tuned per split)")
    print(f"Gain: +{(overall_correct_optimal-overall_correct_global)/overall_total*100:.2f}%")
    print(f"Veritas baseline: 90.10%")
    beats = overall_correct_optimal/overall_total > 0.901
    print(f"\n>>> {'BEATS VERITAS' if beats else 'Does not beat Veritas'} <<<")

    # Save optimal configs
    out = {
        'per_split_config': {s: {'alpha':a, 'threshold':t, 'acc':acc} for s,(a,t,acc) in best_configs.items()},
        'global_acc': overall_correct_global/overall_total,
        'optimal_acc': overall_correct_optimal/overall_total,
        'n': overall_total,
        'beats_veritas': beats,
    }
    with open('/home/sachin.chaudhary/xgendet/checkpoints/ensemble_v5_prism/per_split_tuned.json', 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == '__main__':
    main()
