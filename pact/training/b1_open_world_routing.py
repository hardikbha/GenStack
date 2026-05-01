"""
B1 — Open-world routing without generator manifest.

Replaces the oracle gate `g(x)` with a UNSUPERVISED clustering router that
groups test samples by their (p_v, b_p) feature vector. We train one Gradient
Boosting expert per cluster and report the accuracy under different cluster
counts K.

This directly addresses the reviewer concern that the per-generator MoE only
works because it uses dataset-manifest information.

Outputs to /home/sachin.chaudhary/xgendet/checkpoints/final_results/b1_open_world.{json,md}
"""
import json
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

# ─── load cached predictions (same merge as ablations_tier_a_b.py) ───────────
print("[1/4] Loading cached predictions...")
v5_probs = json.load(
    open(
        '/home/sachin.chaudhary/xgendet/checkpoints/ensemble_v5_prism/v5_probs.json'
    ))
_ANS = re.compile(r'<answer>\s*(real|fake)\s*</answer>', re.I)

prism_v1 = {}
for d in ([
        f'/home/sachin.chaudhary/veritas_clone/result_prism_worker{i}'
        for i in range(4)
] + [
        f'/home/sachin.chaudhary/veritas_clone/result_prism_helper{i}'
        for i in range(13)
]):
    if not os.path.isdir(d):
        continue
    for f in glob(f"{d}/prism_*.jsonl"):
        bn = Path(f).name
        if bn.startswith(('prism_v2_', 'prism_mipo_')):
            continue
        name = bn.replace('.jsonl', '').split('-')[0].replace('prism_', '', 1)
        gen = name.replace('_part0', '').replace('_part1', '')
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    d_ = json.loads(line)
                except Exception:
                    continue
                imgs = d_.get('images', [])
                if not imgs or d_.get('label') is None:
                    continue
                p = imgs[0].get('path') if isinstance(imgs[0],
                                                      dict) else imgs[0]
                if not p:
                    continue
                m = _ANS.search(d_.get('response', ''))
                if not m:
                    continue
                prism_v1[p] = {
                    'pred': 1 if m.group(1).lower() == 'fake' else 0,
                    'label': int(d_['label']),
                    'generator': gen,
                    'split': gen.split('_')[0],
                }

merged = []
for p, d in prism_v1.items():
    vp = v5_probs.get(p)
    if vp is None:
        continue
    merged.append({
        'path': p,
        'v5': vp,
        'prism': d['pred'],
        'label': d['label'],
        'generator': d['generator'],
        'split': d['split']
    })
print(f"   merged samples: {len(merged)}")

X = np.array([[r['v5'], r['prism']] for r in merged])
y = np.array([r['label'] for r in merged])
gens = np.array([r['generator'] for r in merged])
splits = np.array([r['split'] for r in merged])


def acc_prob(p, l):
    return float(((p >= 0.5).astype(int) == l).mean())


def per_split_acc(pred, threshold=0.5):
    out = {}
    for s in ('id', 'cm', 'cf', 'cd'):
        m = splits == s
        out[s] = float(((pred[m] >= threshold).astype(int) == y[m]).mean())
    out['overall'] = float(((pred >= threshold).astype(int) == y).mean())
    return out


def per_cluster_kfold_oof(cluster_ids, K_folds=5):
    oof = np.zeros(len(y))
    for c in sorted(set(cluster_ids)):
        m = cluster_ids == c
        idx = np.where(m)[0]
        Xs, ys = X[idx], y[idx]
        if m.sum() < 50 or ys.mean() in (0, 1):
            pseudo = np.where(X[idx, 1] == 1, 0.9, 0.1)
            oof[idx] = 0.5 * X[idx, 0] + 0.5 * pseudo
            continue
        n_fold = max(2, min(K_folds, int(m.sum() // 20)))
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
        for tr, te in kf.split(Xs):
            if y[idx[tr]].mean() in (0, 1):
                pseudo = np.where(X[idx[te], 1] == 1, 0.9, 0.1)
                oof[idx[te]] = 0.5 * X[idx[te], 0] + 0.5 * pseudo
                continue
            mdl = GradientBoostingClassifier(n_estimators=50,
                                             max_depth=2,
                                             learning_rate=0.1,
                                             random_state=42)
            mdl.fit(Xs[tr], y[idx[tr]])
            oof[idx[te]] = mdl.predict_proba(Xs[te])[:, 1]
    return oof


# ─── Reference: oracle per-generator routing ─────────────────────────────────
print("[2/4] Reference (oracle per-generator) ...")
oracle_oof = per_cluster_kfold_oof(gens)
oracle_acc = per_split_acc(oracle_oof)
print(f"   oracle (23 generators) overall = {oracle_acc['overall']*100:.2f}%")

# ─── B1 main: KMeans clustering on (p_v, b_p) ───────────────────────────────
results = {'oracle_per_generator': oracle_acc}
print("[3/4] Open-world KMeans routing...")
for K in (1, 4, 8, 16, 23):
    if K == 1:
        cluster_ids = np.zeros(len(y), dtype=int)
    else:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(X)
    oof = per_cluster_kfold_oof(cluster_ids)
    acc = per_split_acc(oof)
    results[f'kmeans_K={K}'] = acc
    print(f"   KMeans K={K:>2}  overall = {acc['overall']*100:5.2f}%")

# ─── B1 alt: split-aware K-means (cluster within each split) ─────────────────
print("[4/4] Open-world split-aware KMeans (4 splits × K_per_split)...")
for K_per in (1, 2, 4, 6):
    cluster_ids = np.full(len(y), -1, dtype=int)
    next_id = 0
    for s in ('id', 'cm', 'cf', 'cd'):
        mask = splits == s
        Xs = X[mask]
        if K_per == 1 or Xs.shape[0] < K_per * 20:
            cluster_ids[mask] = next_id
            next_id += 1
        else:
            km = KMeans(n_clusters=K_per, random_state=42, n_init=10)
            local = km.fit_predict(Xs)
            cluster_ids[mask] = local + next_id
            next_id += K_per
    oof = per_cluster_kfold_oof(cluster_ids)
    acc = per_split_acc(oof)
    n_clusters = len(set(cluster_ids))
    results[f'split_aware_K_per={K_per}_total={n_clusters}'] = acc
    print(
        f"   split×{K_per}={n_clusters:>2} clusters overall = {acc['overall']*100:5.2f}%"
    )

# ─── Save ────────────────────────────────────────────────────────────────────
out = Path('/home/sachin.chaudhary/xgendet/checkpoints/final_results')
(out / 'b1_open_world.json').write_text(json.dumps(results, indent=2))

lines = [
    "# B1 — Open-world routing (no generator manifest)", "",
    "Replace the oracle gate `g(x)` with an unsupervised KMeans router",
    "over the 2-d feature `(p_v, b_p)`. One GB expert per cluster.", "",
    "## Plain KMeans", "", "| Variant | ID | CM | CF | CD | Overall |",
    "|---|---:|---:|---:|---:|---:|"
]
for K in (1, 4, 8, 16, 23):
    v = results[f'kmeans_K={K}']
    lines.append(f"| KMeans K={K} | {v['id']*100:.2f} | {v['cm']*100:.2f} | "
                 f"{v['cf']*100:.2f} | {v['cd']*100:.2f} | "
                 f"**{v['overall']*100:.2f}** |")
lines.append("")

lines.append("## Split-aware KMeans (cluster within each ID/CM/CF/CD split)")
lines.append("")
lines.append("| Variant | ID | CM | CF | CD | Overall |")
lines.append("|---|---:|---:|---:|---:|---:|")
for K_per in (1, 2, 4, 6):
    keys = [k for k in results if k.startswith(f'split_aware_K_per={K_per}')]
    if not keys:
        continue
    v = results[keys[0]]
    nclu = keys[0].split('=')[-1]
    lines.append(f"| split-aware K={K_per} (total {nclu}) | "
                 f"{v['id']*100:.2f} | {v['cm']*100:.2f} | "
                 f"{v['cf']*100:.2f} | {v['cd']*100:.2f} | "
                 f"**{v['overall']*100:.2f}** |")
lines.append("")
lines.append("## Reference (oracle, generator manifest known)")
lines.append("")
lines.append("| Variant | ID | CM | CF | CD | Overall |")
lines.append("|---|---:|---:|---:|---:|---:|")
v = results['oracle_per_generator']
lines.append(f"| oracle 23 gen | {v['id']*100:.2f} | {v['cm']*100:.2f} | "
             f"{v['cf']*100:.2f} | {v['cd']*100:.2f} | "
             f"**{v['overall']*100:.2f}** |")

(out / 'b1_open_world.md').write_text('\n'.join(lines))
print(f"\nSaved {out / 'b1_open_world.md'}")
print("Done.")
