"""
K-fold cross-validation of per-split tuning.

For each of 4 folds:
  - Tune (α, threshold) per split on 3 folds
  - Evaluate on held-out fold
Report mean ± std across folds.

This is the HONEST generalization number for the paper.
Only 8 hyperparameters total (α + threshold per 4 splits) → low overfit risk,
but formal validation is still required.
"""

import json, re, os, random
from collections import defaultdict
from glob import glob
from pathlib import Path

random.seed(42)

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
            if Path(f).name.startswith(('prism_v2_','prism_mipo_')): continue
            split_name = Path(f).stem.split('-')[0].replace('prism_','',1)
            with open(f) as fh:
                for line in fh:
                    if not line.strip(): continue
                    try: x = json.loads(line)
                    except: continue
                    imgs = x.get('images',[])
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
    prism = load_prism()
    v5_probs = json.load(open('/home/sachin.chaudhary/xgendet/checkpoints/ensemble_v5_prism/v5_probs.json'))

    merged = []
    for p, pd in prism.items():
        vp = v5_probs.get(p)
        if vp is None: continue
        merged.append({**pd, 'path': p, 'v5_prob': float(vp)})
    print(f"Total merged samples: {len(merged)}")

    # Group by split for stratified folding
    by_split = defaultdict(list)
    for d in merged: by_split[d['split']].append(d)

    K = 4
    for s in by_split: random.shuffle(by_split[s])

    # Build K folds, stratified by split
    folds = [[] for _ in range(K)]
    for s, items in by_split.items():
        for i, d in enumerate(items):
            folds[i % K].append(d)

    alphas = [round(i*0.1,1) for i in range(11)]
    thresholds = [round(0.3+0.05*i,2) for i in range(9)]

    def compute_split_acc(items, alpha, thr):
        c = t = 0
        for d in items:
            v5p = d['v5_prob']
            pp = 0.9 if d['prism_pred']=='fake' else 0.1 if d['prism_pred']=='real' else 0.5
            ens = alpha*v5p + (1-alpha)*pp
            pred = 1 if ens >= thr else 0
            t += 1
            if pred == d['label']: c += 1
        return c, t

    fold_results = []
    for k in range(K):
        val_fold = folds[k]
        train_folds = [d for j in range(K) if j != k for d in folds[j]]

        # Tune on train_folds, per split
        train_by_split = defaultdict(list)
        for d in train_folds: train_by_split[d['split']].append(d)

        best_cfg = {}
        for s in ['id','cm','cf','cd']:
            items = train_by_split.get(s, [])
            if not items: continue
            best = (0, 0.5, 0.5)
            for a in alphas:
                for t in thresholds:
                    c, tot = compute_split_acc(items, a, t)
                    acc = c/tot
                    if acc > best[0]:
                        best = (acc, a, t)
            best_cfg[s] = (best[1], best[2])

        # Apply to held-out fold
        correct = total = 0
        per_split = {}
        val_by_split = defaultdict(list)
        for d in val_fold: val_by_split[d['split']].append(d)
        for s, items in val_by_split.items():
            a, t = best_cfg.get(s, (0.5, 0.5))
            c, tot = compute_split_acc(items, a, t)
            per_split[s] = (c, tot, c/tot)
            correct += c; total += tot
        fold_acc = correct / total
        fold_results.append(fold_acc)
        print(f"\nFold {k+1}/{K}: overall={fold_acc*100:.2f}%")
        for s in ['id','cm','cf','cd']:
            if s in per_split:
                c, tot, acc = per_split[s]
                a, t = best_cfg[s]
                print(f"  {s.upper()}: acc={acc*100:.2f}% (α={a}, t={t}, {c}/{tot})")

    import statistics
    mean_acc = statistics.mean(fold_results)
    std_acc = statistics.stdev(fold_results) if len(fold_results) > 1 else 0
    print(f"\n{'='*60}")
    print(f"K-fold CV ({K} folds): Mean = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Veritas baseline: 90.10%")
    if mean_acc > 0.901:
        print(f">>> BEATS Veritas by {(mean_acc-0.901)*100:.2f}% (within ±{std_acc*100:.2f}%)")
    else:
        print(f">>> {(0.901-mean_acc)*100:.2f}% below Veritas (need this much more)")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
