"""
Full per-generator breakdown across all 4 models + ensemble.
Shows: v5, v5plus, PRISM v1 SFT, PRISM MiPO, and Ensemble (per-gen GB) accuracy.
"""

import json, re, os
from pathlib import Path
from glob import glob

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

# ─── Load cached probs ───────────────────────────────────────────────────────
v5_probs     = json.load(open('/home/sachin.chaudhary/xgendet/checkpoints/ensemble_v5_prism/v5_probs.json'))
v5plus_probs = json.load(open('/home/sachin.chaudhary/xgendet/checkpoints/ensemble_v5plus_prism/v5plus_probs.json'))

_ANS = re.compile(r'<answer>\s*(real|fake)\s*</answer>', re.I)

def parse_dir(d):
    out = {}
    for f in glob(f"{d}/prism_*.jsonl"):
        bn = Path(f).name
        if bn.startswith(('prism_v2_',)): continue
        name = bn.replace('.jsonl','').split('-')[0].replace('prism_','',1)
        if name.startswith('mipo_'): name = name[5:]
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
                out[p] = {'pred': 1 if m.group(1).lower() == 'fake' else 0,
                          'label': int(d_['label']), 'generator': gen,
                          'split': gen.split('_')[0]}
    return out

# PRISM v1
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

# PRISM MiPO
prism_mipo = {}
for d in [f'/home/sachin.chaudhary/veritas_clone/result_prism_mipo_helper{i}' for i in range(11)]:
    if not os.path.isdir(d): continue
    prism_mipo.update(parse_dir(d))

# Merge — only samples in all 4
merged = []
for p, d in prism_v1.items():
    vp = v5_probs.get(p); vpp = v5plus_probs.get(p); mipo = prism_mipo.get(p)
    if vp is None or vpp is None or mipo is None: continue
    merged.append({
        'path': p,
        'v5': vp, 'v5plus': vpp,
        'prism_v1': d['pred'],
        'prism_mipo': mipo['pred'],
        'label': d['label'],
        'generator': d['generator'],
        'split': d['split'],
    })

X = np.array([[r['v5'], r['prism_v1']] for r in merged])  # for ensemble (v5+PRISM only)
y = np.array([r['label'] for r in merged])
gens = np.array([r['generator'] for r in merged])
splits = np.array([r['split'] for r in merged])
v5_arr = np.array([r['v5'] for r in merged])
v5plus_arr = np.array([r['v5plus'] for r in merged])
prism_v1_arr = np.array([r['prism_v1'] for r in merged])
prism_mipo_arr = np.array([r['prism_mipo'] for r in merged])

# ─── Per-generator GB ensemble (v5 + PRISM v1) — our 92.42% model ─────────
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

# ─── Accuracy helpers ─────────────────────────────────────────────────────
def acc_prob(p, l):  return ((p >= 0.5).astype(int) == l).mean()
def acc_bin(p, l):   return (p.astype(int) == l).mean()

# ─── Print the full table ─────────────────────────────────────────────────
SPLIT_ORDER = ['id', 'cm', 'cf', 'cd']
SPLIT_NAMES = {'id': 'In-Domain', 'cm': 'Cross-Manipulation', 'cf': 'Cross-Forgery', 'cd': 'Cross-Domain'}

print("=" * 112)
print(f"{'FULL PER-GENERATOR BREAKDOWN (all 4 models + final ensemble)':<112}")
print("=" * 112)
print(f"{'Split / Generator':<25} {'n':>6} {'v5':>8} {'v5plus':>8} {'PRISM v1':>9} {'PRISM MiPO':>11} {'Ensemble':>10} {'Δ':>7}")
print("-" * 112)

all_rows = []
for s in SPLIT_ORDER:
    split_gens = sorted(set(gens[splits == s]))
    # split totals accumulator
    tot = {'n': 0, 'v5': 0, 'v5p': 0, 'pv1': 0, 'pmipo': 0, 'ens': 0}

    print(f"\n{SPLIT_NAMES[s]}")
    for g in split_gens:
        m = gens == g
        n = int(m.sum())
        a_v5  = acc_prob(v5_arr[m], y[m])
        a_v5p = acc_prob(v5plus_arr[m], y[m])
        a_pv1 = acc_bin(prism_v1_arr[m], y[m])
        a_pm  = acc_bin(prism_mipo_arr[m], y[m])
        a_ens = acc_prob(oof[m], y[m])
        best_single = max(a_v5, a_v5p, a_pv1, a_pm)
        delta = a_ens - best_single
        gen_label = g.split('_', 1)[1] if '_' in g else g
        print(f"  {gen_label:<23} {n:>6} {a_v5*100:>7.2f}% {a_v5p*100:>7.2f}% {a_pv1*100:>8.2f}% "
              f"{a_pm*100:>10.2f}% {a_ens*100:>9.2f}% {delta*100:>+6.2f}")
        all_rows.append({'split': s, 'generator': gen_label, 'n': n,
                         'v5': a_v5, 'v5plus': a_v5p, 'prism_v1': a_pv1,
                         'prism_mipo': a_pm, 'ensemble': a_ens})
        tot['n'] += n
        tot['v5']    += int(((v5_arr[m] >= 0.5).astype(int) == y[m]).sum())
        tot['v5p']   += int(((v5plus_arr[m] >= 0.5).astype(int) == y[m]).sum())
        tot['pv1']   += int((prism_v1_arr[m].astype(int) == y[m]).sum())
        tot['pmipo'] += int((prism_mipo_arr[m].astype(int) == y[m]).sum())
        tot['ens']   += int(((oof[m] >= 0.5).astype(int) == y[m]).sum())

    # split summary
    n = tot['n']
    s_v5 = tot['v5']/n; s_v5p = tot['v5p']/n
    s_pv1 = tot['pv1']/n; s_pm = tot['pmipo']/n
    s_ens = tot['ens']/n
    s_best = max(s_v5, s_v5p, s_pv1, s_pm)
    print(f"  {'SPLIT TOTAL':<23} {n:>6} {s_v5*100:>7.2f}% {s_v5p*100:>7.2f}% {s_pv1*100:>8.2f}% "
          f"{s_pm*100:>10.2f}% {s_ens*100:>9.2f}% {(s_ens-s_best)*100:>+6.2f}")

# Overall
ov = {
    'n': len(y),
    'v5': acc_prob(v5_arr, y),
    'v5p': acc_prob(v5plus_arr, y),
    'pv1': acc_bin(prism_v1_arr, y),
    'pmipo': acc_bin(prism_mipo_arr, y),
    'ens': acc_prob(oof, y),
}
print("\n" + "=" * 112)
print(f"{'OVERALL':<25} {ov['n']:>6} {ov['v5']*100:>7.2f}% {ov['v5p']*100:>7.2f}% {ov['pv1']*100:>8.2f}% "
      f"{ov['pmipo']*100:>10.2f}% {ov['ens']*100:>9.2f}%")
print(f"{'Veritas baseline':<25}                                                                    {'90.10%':>9}")
print(f"{'Δ vs Veritas':<25}                                                                    {(ov['ens']-0.901)*100:+8.2f}%")
print("=" * 112)

# Save
os.makedirs('checkpoints/final_results', exist_ok=True)
with open('checkpoints/final_results/full_per_gen_breakdown.json', 'w') as f:
    json.dump({'overall': ov, 'per_generator': all_rows}, f, indent=2)
print("\nSaved to checkpoints/final_results/full_per_gen_breakdown.json")
