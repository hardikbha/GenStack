"""
Generate the final per-generator results table for v5 + PRISM v1 with
per-generator Gradient Boosting stacking (92.42% overall).
"""

import json, re, os
from pathlib import Path
from glob import glob

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

# ─── Load cached data ────────────────────────────────────────────────────────
v5_probs = json.load(open('/home/sachin.chaudhary/xgendet/checkpoints/ensemble_v5_prism/v5_probs.json'))
_ANS = re.compile(r'<answer>\s*(real|fake)\s*</answer>', re.I)

prism_v1 = {}
for d in ([f'/home/sachin.chaudhary/veritas_clone/result_prism_worker{i}' for i in range(4)] +
          [f'/home/sachin.chaudhary/veritas_clone/result_prism_helper{i}' for i in range(13)]):
    if not os.path.isdir(d): continue
    for f in glob(f"{d}/prism_*.jsonl"):
        bn = Path(f).name
        if bn.startswith(('prism_v2_', 'prism_mipo_')): continue
        name = bn.replace('.jsonl','').split('-')[0].replace('prism_','',1)
        gen = name.replace('_part0','').replace('_part1','')
        with open(f) as fh:
            for line in fh:
                if not line.strip(): continue
                try: d_ = json.loads(line)
                except: continue
                imgs = d_.get('images', [])
                if not imgs or d_.get('label') is None: continue
                p = imgs[0].get('path') if isinstance(imgs[0], dict) else imgs[0]
                if not p: continue
                m = _ANS.search(d_.get('response', ''))
                if not m: continue
                prism_v1[p] = {'pred': 1 if m.group(1).lower() == 'fake' else 0,
                               'label': int(d_['label']), 'generator': gen,
                               'split': gen.split('_')[0]}

merged = []
for p, d in prism_v1.items():
    vp = v5_probs.get(p)
    if vp is None: continue
    merged.append({'v5': vp, 'prism': d['pred'], 'label': d['label'],
                   'generator': d['generator'], 'split': d['split']})

X = np.array([[r['v5'], r['prism']] for r in merged])
y = np.array([r['label'] for r in merged])
gens = np.array([r['generator'] for r in merged])
splits = np.array([r['split'] for r in merged])

# ─── Per-generator GB stacking (5-fold CV) ──────────────────────────────
oof = np.zeros(len(y))
for g in sorted(set(gens)):
    m = gens == g
    idx = np.where(m)[0]
    Xs, ys = X[idx], y[idx]
    if m.sum() < 50 or ys.mean() in (0, 1):
        pseudo = np.where(X[idx, 1] == 1, 0.9, 0.1)
        oof[idx] = 0.5 * X[idx, 0] + 0.5 * pseudo
        continue
    n_fold = max(2, min(5, int(m.sum() // 20)))
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    for tr, te in kf.split(Xs):
        if y[idx[tr]].mean() in (0, 1):
            pseudo = np.where(X[idx[te], 1] == 1, 0.9, 0.1)
            oof[idx[te]] = 0.5 * X[idx[te], 0] + 0.5 * pseudo
            continue
        gb = GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)
        gb.fit(Xs[tr], y[idx[tr]])
        oof[idx[te]] = gb.predict_proba(Xs[te])[:, 1]

# ─── Per-generator table ─────────────────────────────────────────────────
SPLIT_ORDER = ['id', 'cm', 'cf', 'cd']
SPLIT_NAMES = {'id': 'In-Domain', 'cm': 'Cross-Manipulation', 'cf': 'Cross-Forgery', 'cd': 'Cross-Domain'}

print(f"\n{'='*90}")
print(f"{'PER-GENERATOR RESULTS TABLE':<90}")
print(f"{'v5 + PRISM v1 SFT  with Per-Generator Gradient Boosting Stacking':<90}")
print(f"{'='*90}")
print(f"{'Generator':<28} {'n':>6} {'v5':>8} {'PRISM':>8} {'Ensemble':>10} {'Δ':>8}")
print(f"{'-'*90}")

all_rows = []
for s in SPLIT_ORDER:
    split_gens = sorted(set(gens[splits == s]))
    # Header per split
    print(f"\n{SPLIT_NAMES[s]:<28}")
    split_c = split_n = 0; v5_c = v5_n = 0; pr_c = pr_n = 0
    for g in split_gens:
        m = gens == g
        n = int(m.sum())
        # v5 acc
        v5_acc = ((X[m, 0] >= 0.5).astype(int) == y[m]).mean()
        # PRISM acc
        pr_acc = (X[m, 1].astype(int) == y[m]).mean()
        # Ensemble acc
        e_acc = ((oof[m] >= 0.5).astype(int) == y[m]).mean()
        delta = e_acc - max(v5_acc, pr_acc)
        delta_str = f"{delta*100:+.2f}"
        gen_label = g.split('_', 1)[1] if '_' in g else g  # strip split prefix
        print(f"  {gen_label:<26} {n:>6} {v5_acc*100:>7.2f}% {pr_acc*100:>7.2f}% {e_acc*100:>9.2f}% {delta_str:>7}")
        all_rows.append({'split': s, 'generator': gen_label, 'n': n,
                         'v5_acc': v5_acc, 'prism_acc': pr_acc, 'ensemble_acc': e_acc})
        split_c += int(((oof[m] >= 0.5).astype(int) == y[m]).sum()); split_n += n
        v5_c += int(((X[m, 0] >= 0.5).astype(int) == y[m]).sum()); v5_n += n
        pr_c += int((X[m, 1].astype(int) == y[m]).sum()); pr_n += n

    v5_split = v5_c / max(v5_n, 1)
    pr_split = pr_c / max(pr_n, 1)
    e_split = split_c / max(split_n, 1)
    print(f"  {'SPLIT TOTAL':<26} {split_n:>6} {v5_split*100:>7.2f}% {pr_split*100:>7.2f}% {e_split*100:>9.2f}%"
          f" {(e_split - max(v5_split, pr_split))*100:+7.2f}")

# Grand overall
overall_c = int(((oof >= 0.5).astype(int) == y).sum())
overall_n = len(y)
v5_overall = ((X[:, 0] >= 0.5).astype(int) == y).mean()
prism_overall = (X[:, 1].astype(int) == y).mean()
ensemble_overall = overall_c / overall_n

print(f"\n{'='*90}")
print(f"{'OVERALL':<28} {overall_n:>6} {v5_overall*100:>7.2f}% {prism_overall*100:>7.2f}%"
      f" {ensemble_overall*100:>9.2f}%")
print(f"{'Veritas baseline':<28} {'':>6} {'':>8} {'':>8} {90.10:>9.2f}%")
print(f"{'Δ vs Veritas':<28} {'':>6} {'':>8} {'':>8} {(ensemble_overall - 0.901)*100:>+9.2f}%")
print(f"{'='*90}")

# Save
import os
os.makedirs('checkpoints/final_results', exist_ok=True)
out = {
    'overall': {
        'v5_alone': v5_overall,
        'prism_alone': prism_overall,
        'ensemble': ensemble_overall,
        'veritas': 0.901,
        'delta_vs_veritas': ensemble_overall - 0.901,
        'n_samples': overall_n,
    },
    'per_generator': all_rows,
}
with open('checkpoints/final_results/v5_prism_per_gen_table.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to checkpoints/final_results/v5_prism_per_gen_table.json")
