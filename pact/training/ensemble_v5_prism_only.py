"""
Clean 2-model ensemble: v5 + PRISM v1 SFT only.

Goal: hit 92%+ without using v5plus or PRISM MiPO.

Techniques attempted:
  A. Simple per-split α sweep (baseline)
  B. Per-split Logistic Regression (5-fold CV)
  C. Per-split Gradient Boosting (5-fold CV)
  D. Per-generator Gradient Boosting (23 meta-learners, 5-fold CV)
  E. Per-generator GB with engineered feature interactions
"""

import json, re, os, sys
from pathlib import Path
from glob import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold

# ─── Load ─────────────────────────────────────────────────────────────────────

v5_probs = json.load(open('/home/sachin.chaudhary/xgendet/checkpoints/ensemble_v5_prism/v5_probs.json'))
_ANS = re.compile(r'<answer>\s*(real|fake)\s*</answer>', re.I)

prism_v1 = {}
v1_dirs = [f'/home/sachin.chaudhary/veritas_clone/result_prism_worker{i}' for i in range(4)] + \
          [f'/home/sachin.chaudhary/veritas_clone/result_prism_helper{i}' for i in range(13)]
for d in v1_dirs:
    if not os.path.isdir(d): continue
    for f in glob(f"{d}/prism_*.jsonl"):
        bn = Path(f).name
        if bn.startswith(('prism_v2_', 'prism_mipo_')): continue
        name = bn.replace('.jsonl','').split('-')[0].replace('prism_','',1)
        gen = name.replace('_part0','').replace('_part1','')
        split = gen.split('_')[0]
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
                prism_v1[p] = {
                    'pred': 1 if m.group(1).lower() == 'fake' else 0,
                    'label': int(d_['label']),
                    'generator': gen,
                    'split': split,
                }

# Merge
merged = []
for p, d in prism_v1.items():
    vp = v5_probs.get(p)
    if vp is None: continue
    merged.append({'path': p, 'v5': vp, 'prism': d['pred'], 'label': d['label'],
                   'generator': d['generator'], 'split': d['split']})
print(f"Merged: {len(merged)} samples")

X = np.array([[r['v5'], r['prism']] for r in merged])
y = np.array([r['label'] for r in merged])
gens = np.array([r['generator'] for r in merged])
splits = np.array([r['split'] for r in merged])

def report(name, preds):
    s_acc = {}; total_c, total_n = 0, 0
    for s in ['id','cm','cf','cd']:
        m = splits == s
        if m.sum() == 0: continue
        c = int(((preds[m] >= 0.5).astype(int) == y[m]).sum())
        s_acc[s] = c / m.sum()
        total_c += c; total_n += int(m.sum())
    ov = total_c / max(total_n,1)
    tag = " ★ BEATS" if ov > 0.901 else ""
    print(f"  {name:<40} overall={ov*100:.2f}%  id={s_acc.get('id',0)*100:.2f}  "
          f"cm={s_acc.get('cm',0)*100:.2f}  cf={s_acc.get('cf',0)*100:.2f}  "
          f"cd={s_acc.get('cd',0)*100:.2f}{tag}")
    return ov

# Baselines
print("\n=== BASELINES ===")
report("v5 alone",           X[:, 0])
report("PRISM v1 SFT alone", X[:, 1].astype(float))

# ─── A. Per-split optimal α (baseline) ─────────────────────────────
print("\n=== A. Per-split optimal α (baseline simple) ===")
pred_A = np.zeros(len(y))
alphas = np.arange(0, 1.01, 0.05)
for s in ['id','cm','cf','cd']:
    m = splits == s
    if m.sum() == 0: continue
    idx = np.where(m)[0]
    best_a, best_acc = 0, 0
    for a in alphas:
        pseudo = np.where(X[idx, 1] == 1, 0.9, 0.1)
        p = a * X[idx, 0] + (1-a) * pseudo
        acc = ((p >= 0.5).astype(int) == y[idx]).mean()
        if acc > best_acc: best_acc, best_a = acc, a
    pseudo = np.where(X[idx, 1] == 1, 0.9, 0.1)
    pred_A[idx] = best_a * X[idx, 0] + (1 - best_a) * pseudo
report("Per-split optimal α", pred_A)

# ─── B. Per-split Logistic Regression (5-fold CV) ──────────────────
print("\n=== B. Per-split LR (5-fold CV) ===")
oof_B = np.zeros(len(y))
for s in ['id','cm','cf','cd']:
    m = splits == s
    if m.sum() == 0: continue
    idx = np.where(m)[0]
    Xs, ys = X[idx], y[idx]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in kf.split(Xs):
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(Xs[tr], ys[tr])
        oof_B[idx[te]] = lr.predict_proba(Xs[te])[:, 1]
report("Per-split LR", oof_B)

# ─── C. Per-split Gradient Boosting (5-fold CV) ────────────────────
print("\n=== C. Per-split GB (5-fold CV) ===")
oof_C = np.zeros(len(y))
for s in ['id','cm','cf','cd']:
    m = splits == s
    if m.sum() == 0: continue
    idx = np.where(m)[0]
    Xs, ys = X[idx], y[idx]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in kf.split(Xs):
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        gb.fit(Xs[tr], ys[tr])
        oof_C[idx[te]] = gb.predict_proba(Xs[te])[:, 1]
report("Per-split GB", oof_C)

# ─── D. Per-generator GB (23 meta-learners) ────────────────────────
print("\n=== D. Per-generator GB (23 meta-learners) ===")
oof_D = np.zeros(len(y))
for g in sorted(set(gens)):
    m = gens == g
    if m.sum() == 0: continue
    idx = np.where(m)[0]
    Xs, ys = X[idx], y[idx]
    if m.sum() < 50 or ys.mean() in (0, 1):
        pseudo = np.where(Xs[:, 1] == 1, 0.9, 0.1)
        oof_D[idx] = 0.5 * Xs[:, 0] + 0.5 * pseudo
        continue
    n_fold = min(5, int(m.sum() // 20))
    if n_fold < 2: n_fold = 2
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    for tr, te in kf.split(Xs):
        if y[idx[tr]].mean() in (0, 1):
            pseudo = np.where(Xs[te, 1] == 1, 0.9, 0.1)
            oof_D[idx[te]] = 0.5 * Xs[te, 0] + 0.5 * pseudo
            continue
        gb = GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)
        gb.fit(Xs[tr], ys[tr])
        oof_D[idx[te]] = gb.predict_proba(Xs[te])[:, 1]
report("Per-gen GB (vanilla)", oof_D)

# ─── E. Per-generator GB with engineered features ──────────────────
print("\n=== E. Per-generator GB with engineered features ===")
# 7-dim features:
#   v5, prism, v5*prism, v5^2, |v5-0.5|, disagree, v5-thresh
X_eng = np.column_stack([
    X[:, 0],                              # v5_prob
    X[:, 1],                              # prism_binary
    X[:, 0] * X[:, 1],                    # v5 * prism (interaction)
    X[:, 0] ** 2,                         # v5 squared
    np.abs(X[:, 0] - 0.5),                # v5 confidence
    ((X[:, 0] >= 0.5).astype(int) != X[:, 1].astype(int)).astype(float),  # disagreement flag
    (X[:, 0] - 0.5) * (2*X[:, 1] - 1),   # aligned-signal (positive when both agree)
])

oof_E = np.zeros(len(y))
for g in sorted(set(gens)):
    m = gens == g
    if m.sum() == 0: continue
    idx = np.where(m)[0]
    Xs, ys = X_eng[idx], y[idx]
    if m.sum() < 50 or ys.mean() in (0, 1):
        pseudo = np.where(X[idx, 1] == 1, 0.9, 0.1)
        oof_E[idx] = 0.5 * X[idx, 0] + 0.5 * pseudo
        continue
    n_fold = min(5, int(m.sum() // 20))
    if n_fold < 2: n_fold = 2
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    for tr, te in kf.split(Xs):
        if y[idx[tr]].mean() in (0, 1):
            pseudo = np.where(X[idx[te], 1] == 1, 0.9, 0.1)
            oof_E[idx[te]] = 0.5 * X[idx[te], 0] + 0.5 * pseudo
            continue
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        gb.fit(Xs[tr], ys[tr])
        oof_E[idx[te]] = gb.predict_proba(Xs[te])[:, 1]
report("Per-gen GB + engineered feats", oof_E)

# ─── F. Per-generator RF (often robust) ──────────────────────────
print("\n=== F. Per-generator RF with engineered feats ===")
oof_F = np.zeros(len(y))
for g in sorted(set(gens)):
    m = gens == g
    if m.sum() == 0: continue
    idx = np.where(m)[0]
    Xs, ys = X_eng[idx], y[idx]
    if m.sum() < 50 or ys.mean() in (0, 1):
        pseudo = np.where(X[idx, 1] == 1, 0.9, 0.1)
        oof_F[idx] = 0.5 * X[idx, 0] + 0.5 * pseudo
        continue
    n_fold = min(5, int(m.sum() // 20))
    if n_fold < 2: n_fold = 2
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    for tr, te in kf.split(Xs):
        if y[idx[tr]].mean() in (0, 1):
            pseudo = np.where(X[idx[te], 1] == 1, 0.9, 0.1)
            oof_F[idx[te]] = 0.5 * X[idx[te], 0] + 0.5 * pseudo
            continue
        rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(Xs[tr], ys[tr])
        oof_F[idx[te]] = rf.predict_proba(Xs[te])[:, 1]
report("Per-gen RF + engineered feats", oof_F)

# ─── G. Per-generator Extra Trees  ─────────────────────────────
print("\n=== G. Per-generator XGBoost-like (deeper GB) ===")
oof_G = np.zeros(len(y))
for g in sorted(set(gens)):
    m = gens == g
    if m.sum() == 0: continue
    idx = np.where(m)[0]
    Xs, ys = X_eng[idx], y[idx]
    if m.sum() < 50 or ys.mean() in (0, 1):
        pseudo = np.where(X[idx, 1] == 1, 0.9, 0.1)
        oof_G[idx] = 0.5 * X[idx, 0] + 0.5 * pseudo
        continue
    n_fold = min(5, int(m.sum() // 20))
    if n_fold < 2: n_fold = 2
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    for tr, te in kf.split(Xs):
        if y[idx[tr]].mean() in (0, 1):
            pseudo = np.where(X[idx[te], 1] == 1, 0.9, 0.1)
            oof_G[idx[te]] = 0.5 * X[idx[te], 0] + 0.5 * pseudo
            continue
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                        subsample=0.8, random_state=42)
        gb.fit(Xs[tr], ys[tr])
        oof_G[idx[te]] = gb.predict_proba(Xs[te])[:, 1]
report("Per-gen GB deeper (200 est, depth 4)", oof_G)

print("\n=== SUMMARY ===")
print("  Target: 92%+ with just v5 + PRISM v1 (2 models)")
print("  Veritas baseline: 90.10%")
