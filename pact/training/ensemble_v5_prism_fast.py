"""
Fast, focused 2-model ensemble tuning.
Only the techniques most likely to beat 92.42%:
  A. Per-gen HistGB tuned (single fast config)
  B. Per-gen GB seed-averaged (5 seeds)
  C. Meta-learner ensemble (average of GB + RF + LR predictions)
  D. Per-gen HistGB + engineered features, 3 seeds
"""

import json, re, os
from pathlib import Path
from glob import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import KFold

# Load data
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
print(f"Merged: {len(merged)}")

X = np.array([[r['v5'], r['prism']] for r in merged])
y = np.array([r['label'] for r in merged])
gens = np.array([r['generator'] for r in merged])
splits = np.array([r['split'] for r in merged])

X_fe = np.column_stack([
    X[:, 0], X[:, 1],
    X[:, 0] * X[:, 1],
    (X[:, 0] - 0.5) * (2*X[:, 1] - 1),
    np.abs(X[:, 0] - 0.5),
])

def report(name, preds):
    total_c = total_n = 0; s_acc = {}
    for s in ['id','cm','cf','cd']:
        m = splits == s
        c = int(((preds[m] >= 0.5).astype(int) == y[m]).sum())
        s_acc[s] = c / m.sum(); total_c += c; total_n += int(m.sum())
    ov = total_c / max(total_n,1)
    tag = " ★★★" if ov > 0.925 else (" ★★" if ov > 0.92 else (" ★" if ov > 0.901 else ""))
    print(f"  {name:<38} overall={ov*100:.2f}%  id={s_acc['id']*100:.2f}  "
          f"cm={s_acc['cm']*100:.2f}  cf={s_acc['cf']*100:.2f}  "
          f"cd={s_acc['cd']*100:.2f}{tag}")
    return ov

def per_gen_stack(fit_fn, feats, seed=42):
    oof = np.zeros(len(y))
    for g in sorted(set(gens)):
        m = gens == g
        idx = np.where(m)[0]
        Xs, ys = feats[idx], y[idx]
        if m.sum() < 50 or ys.mean() in (0, 1):
            pseudo = np.where(X[idx, 1] == 1, 0.9, 0.1)
            oof[idx] = 0.5 * X[idx, 0] + 0.5 * pseudo
            continue
        n_fold = max(2, min(5, int(m.sum() // 20)))
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        for tr, te in kf.split(Xs):
            if y[idx[tr]].mean() in (0, 1):
                pseudo = np.where(X[idx[te], 1] == 1, 0.9, 0.1)
                oof[idx[te]] = 0.5 * X[idx[te], 0] + 0.5 * pseudo
                continue
            clf = fit_fn()
            clf.fit(Xs[tr], y[idx[tr]])
            oof[idx[te]] = clf.predict_proba(Xs[te])[:, 1]
    return oof

print("\n=== BASELINES ===")
report("v5", X[:, 0])
report("PRISM v1 SFT", X[:, 1].astype(float))

print("\n=== Anchor: Per-gen GB vanilla (we got 92.42% before) ===")
anchor = per_gen_stack(
    lambda: GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42),
    feats=X,
)
report("Per-gen GB vanilla", anchor)

# ─── A. Per-gen HistGB single config ──────────────
print("\n=== A. Per-gen HistGB configs (fast) ===")
for cfg_name, params in [
    ("HistGB default",            dict(max_iter=100, learning_rate=0.1)),
    ("HistGB depth3",             dict(max_iter=100, max_depth=3, learning_rate=0.1)),
    ("HistGB depth5 l2=1",        dict(max_iter=200, max_depth=5, learning_rate=0.05, l2_regularization=1.0)),
    ("HistGB shallow l2=0.5",     dict(max_iter=100, max_depth=3, learning_rate=0.1, l2_regularization=0.5)),
]:
    preds = per_gen_stack(
        lambda p=params: HistGradientBoostingClassifier(**p, random_state=42),
        feats=X,
    )
    report(cfg_name, preds)

# ─── B. GB seed ensemble (5 seeds, avg) ────────────
print("\n=== B. Per-gen GB seed ensemble (avg of 5 seeds) ===")
preds_all = []
for seed in [42, 123, 777, 2024, 99]:
    preds_all.append(per_gen_stack(
        lambda s=seed: GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=s),
        feats=X, seed=seed,
    ))
report("GB 5-seed avg", np.mean(preds_all, axis=0))

# ─── C. Meta-learner average (GB + RF + LR) ─────────
print("\n=== C. Average of meta-learners (GB + RF + LR) ===")
oof_gb = per_gen_stack(
    lambda: GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42),
    feats=X,
)
oof_rf = per_gen_stack(
    lambda: RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1),
    feats=X,
)
oof_lr = per_gen_stack(lambda: LogisticRegression(max_iter=1000), feats=X)
report("(GB + RF + LR) / 3", (oof_gb + oof_rf + oof_lr) / 3)

# Different weighting
report("0.5*GB + 0.3*RF + 0.2*LR", 0.5*oof_gb + 0.3*oof_rf + 0.2*oof_lr)

# ─── D. HistGB with engineered features ──────────
print("\n=== D. Per-gen HistGB + engineered feats ===")
preds_d = per_gen_stack(
    lambda: HistGradientBoostingClassifier(max_iter=150, max_depth=4, learning_rate=0.05,
                                           l2_regularization=1.0, random_state=42),
    feats=X_fe,
)
report("HistGB + engineered feats", preds_d)

# ─── E. Seed-averaged HistGB + feats ────────────
print("\n=== E. HistGB + feats, 3-seed avg ===")
preds_all = []
for seed in [42, 123, 777]:
    preds_all.append(per_gen_stack(
        lambda s=seed: HistGradientBoostingClassifier(max_iter=150, max_depth=4, learning_rate=0.05,
                                                     l2_regularization=1.0, random_state=s),
        feats=X_fe, seed=seed,
    ))
report("HistGB+feats 3-seed avg", np.mean(preds_all, axis=0))

# ─── Best combo: anchor GB avg + HistGB ────────
print("\n=== F. Best combo avg ===")
report("(Per-gen GB + HistGB+feats 3-seed)/2",
       (anchor + np.mean(preds_all, axis=0)) / 2)

print("\n=== SUMMARY ===\n  Best technique wins. Veritas: 90.10%")
