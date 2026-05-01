"""C1-E gate eval — same outer-5-fold routing protocol as c1_predicted_gate.py
but feed the gate classifier the CLIP-CLS embedding (768-d) instead of the
raw 2-d (p_v, b_p). Experts still consume (p_v, b_p) for the binary decision.

Reports oracle, predicted-gate (hard/soft) for both 23-gen and 4-split gates,
plus gate top-1/top-3 diagnostics.
"""
import json, os, glob
from pathlib import Path
import numpy as np
from sklearn.ensemble import (GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

ROOT = Path('/home/sachin.chaudhary/xgendet/checkpoints/final_results')
PATHS_JSON = ROOT / 'c1e_merged_paths.json'
CHUNK_DIR = ROOT / 'c1e_clip_chunks'
OUT_JSON = ROOT / 'c1e_clip_gate.json'
OUT_MD = ROOT / 'c1e_clip_gate.md'

# ── load merged metadata ─────────────────────────────────────────────────────
print('[1/5] loading merged paths + chunks')
merged = json.load(open(PATHS_JSON))
N = len(merged)
print(f'   {N} merged samples')

# ── load chunks ──────────────────────────────────────────────────────────────
chunks = sorted(glob.glob(str(CHUNK_DIR / 'feat_*.npy')))
assert len(chunks) == 8, f'expected 8 chunks, got {len(chunks)}'
clip_feats = np.full((N, 768), np.nan, dtype=np.float32)
bad_total = 0
for npy_path in chunks:
    idx_path = npy_path.replace('.npy', '.json')
    feat = np.load(npy_path)
    info = json.load(open(idx_path))
    s = info['slice_start']
    for j, kept in enumerate(info['kept_idx_in_slice']):
        clip_feats[s + kept] = feat[j]
    bad_total += info['bad']
mask_ok = ~np.isnan(clip_feats[:, 0])
print(
    f'   clip_feats {clip_feats.shape}  bad/missing={(~mask_ok).sum()} (worker bad={bad_total})'
)

# ── filter to rows with clip features ────────────────────────────────────────
keep = np.where(mask_ok)[0]
merged_ok = [merged[i] for i in keep]
clip_ok = clip_feats[keep]
print(f'   keeping {len(keep)} samples after clip filter')

X2 = np.array([[r['v5'], r['prism']] for r in merged_ok], dtype=np.float32)
y = np.array([r['label'] for r in merged_ok], dtype=np.int64)
gens = np.array([r['generator'] for r in merged_ok])
splits = np.array([r['split'] for r in merged_ok])
gen_list = sorted(set(gens.tolist()))
gen2id = {g: i for i, g in enumerate(gen_list)}
gen_y = np.array([gen2id[g] for g in gens])
N_GEN = len(gen_list)
split_list = ['id', 'cm', 'cf', 'cd']
split_y = np.array([split_list.index(s) for s in splits])
N_SPLIT = 4

print(f'   {N_GEN} gens, {N_SPLIT} splits')

# l2-normalize CLIP features (standard)
clip_n = clip_ok / (np.linalg.norm(clip_ok, axis=1, keepdims=True) + 1e-8)


def per_split_acc(pred, threshold=0.5):
    out = {}
    for s in split_list:
        m = splits == s
        out[s] = float(((pred[m] >= threshold).astype(int) == y[m]).mean())
    out['overall'] = float(((pred >= threshold).astype(int) == y).mean())
    return out


def fit_expert(X_tr, y_tr):
    if len(y_tr) < 50 or y_tr.mean() in (0, 1):
        return None
    e = GradientBoostingClassifier(n_estimators=50,
                                   max_depth=2,
                                   learning_rate=0.1,
                                   random_state=42)
    e.fit(X_tr, y_tr)
    return e


def expert_predict(e, Xs):
    if e is None:
        return 0.5 * Xs[:, 0] + 0.5 * np.where(Xs[:, 1] == 1, 0.9, 0.1)
    return e.predict_proba(Xs)[:, 1]


# ── outer 5-fold ─────────────────────────────────────────────────────────────
print('[2/5] outer 5-fold over binary label')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_oracle = np.zeros(len(y))
oof_p23h = np.zeros(len(y))
oof_p23s = np.zeros(len(y))
oof_p4h = np.zeros(len(y))
oof_p4s = np.zeros(len(y))
gate23_top1 = np.zeros(len(y), dtype=bool)
gate23_top3 = np.zeros(len(y), dtype=bool)
gate4_top1 = np.zeros(len(y), dtype=bool)

for fold, (tr, te) in enumerate(skf.split(X2, y), 1):
    print(f'   fold {fold}/5  train={len(tr)}  test={len(te)}')

    experts_g = {}
    for gi in range(N_GEN):
        m = gen_y[tr] == gi
        experts_g[gi] = fit_expert(X2[tr][m], y[tr][m])
    experts_s = {}
    for si in range(N_SPLIT):
        m = split_y[tr] == si
        experts_s[si] = fit_expert(X2[tr][m], y[tr][m])

    # gate classifiers on CLIP features (standardised + LR multinomial — fast & strong)
    sc23 = StandardScaler().fit(clip_n[tr])
    Xc_tr = sc23.transform(clip_n[tr])
    Xc_te = sc23.transform(clip_n[te])
    gate23 = LogisticRegression(max_iter=1500,
                                C=1.0,
                                n_jobs=-1,
                                solver='lbfgs')
    gate23.fit(Xc_tr, gen_y[tr])
    gate4 = LogisticRegression(max_iter=1500, C=1.0, n_jobs=-1, solver='lbfgs')
    gate4.fit(Xc_tr, split_y[tr])

    Xte2 = X2[te]

    # oracle route
    for j, i in enumerate(te):
        gi = gen_y[i]
        oof_oracle[i] = expert_predict(experts_g[gi], Xte2[j:j + 1])[0]

    # precompute expert outputs per-test-row for soft mixing
    eg = np.zeros((len(te), N_GEN))
    for gi in range(N_GEN):
        eg[:, gi] = expert_predict(experts_g[gi], Xte2)
    es = np.zeros((len(te), N_SPLIT))
    for si in range(N_SPLIT):
        es[:, si] = expert_predict(experts_s[si], Xte2)

    proba23 = gate23.predict_proba(Xc_te)
    classes23 = gate23.classes_
    full23 = np.zeros((len(te), N_GEN))
    for ci, c in enumerate(classes23):
        full23[:, c] = proba23[:, ci]
    pred23 = classes23[proba23.argmax(axis=1)]
    oof_p23h[te] = eg[np.arange(len(te)), pred23]
    oof_p23s[te] = (full23 * eg).sum(axis=1)
    gate23_top1[te] = pred23 == gen_y[te]
    top3 = np.argsort(-proba23, axis=1)[:, :3]
    top3c = classes23[top3]
    gate23_top3[te] = (top3c == gen_y[te][:, None]).any(axis=1)

    proba4 = gate4.predict_proba(Xc_te)
    classes4 = gate4.classes_
    full4 = np.zeros((len(te), N_SPLIT))
    for ci, c in enumerate(classes4):
        full4[:, c] = proba4[:, ci]
    pred4 = classes4[proba4.argmax(axis=1)]
    oof_p4h[te] = es[np.arange(len(te)), pred4]
    oof_p4s[te] = (full4 * es).sum(axis=1)
    gate4_top1[te] = pred4 == split_y[te]

# ── report ───────────────────────────────────────────────────────────────────
print('[3/5] routing accuracy:')
results = {
    'oracle_23gen': per_split_acc(oof_oracle),
    'pred23gen_hard': per_split_acc(oof_p23h),
    'pred23gen_soft': per_split_acc(oof_p23s),
    'pred4split_hard': per_split_acc(oof_p4h),
    'pred4split_soft': per_split_acc(oof_p4s),
}
for n, r in results.items():
    print(f"   {n:18s} ID {r['id']*100:5.2f}  CM {r['cm']*100:5.2f}  "
          f"CF {r['cf']*100:5.2f}  CD {r['cd']*100:5.2f}  "
          f"OVR {r['overall']*100:5.2f}")

print('[4/5] gate diagnostics:')
gate_diag = {
    'gate23_top1_acc': float(gate23_top1.mean()),
    'gate23_top3_acc': float(gate23_top3.mean()),
    'gate4_top1_acc': float(gate4_top1.mean()),
}
for k, v in gate_diag.items():
    print(f'   {k:18s} {v*100:5.2f}%')

print('[5/5] saving outputs')
json.dump(
    {
        'routing': results,
        'gate_diagnostics': gate_diag,
        'gate_features': 'CLIP ViT-L/14 CLS, l2-normalised, multinomial LR',
        'n_samples': int(len(y))
    },
    open(OUT_JSON, 'w'),
    indent=2)

lines = [
    '# C1-E — Predicted-gate with CLIP features (deployable)',
    '',
    'Same protocol as C1 but the gate classifier consumes frozen CLIP ViT-L/14',
    'CLS embeddings (768-d, l2-normalised, multinomial LR). Experts are unchanged',
    '(GradientBoosting on (p_v, b_p)). Outer 5-fold OOF.',
    '',
    '## Routing accuracy',
    '',
    '| Variant | ID | CM | CF | CD | Overall |',
    '|---|---:|---:|---:|---:|---:|',
]


def row(name, r):
    return (f"| {name} | {r['id']*100:.2f} | {r['cm']*100:.2f} | "
            f"{r['cf']*100:.2f} | {r['cd']*100:.2f} | "
            f"**{r['overall']*100:.2f}** |")


lines += [
    row('Oracle 23-gen (manifest)', results['oracle_23gen']),
    row('Pred 23-gen gate (hard)', results['pred23gen_hard']),
    row('Pred 23-gen gate (soft)', results['pred23gen_soft']),
    row('Pred 4-split gate (hard)', results['pred4split_hard']),
    row('Pred 4-split gate (soft)',
        results['pred4split_soft']), '', '## Gate diagnostics', '',
    f"- 23-way gate top-1 acc: **{gate_diag['gate23_top1_acc']*100:.2f}%** (chance 4.35%)",
    f"- 23-way gate top-3 acc: **{gate_diag['gate23_top3_acc']*100:.2f}%**",
    f"-  4-way gate top-1 acc: **{gate_diag['gate4_top1_acc']*100:.2f}%** (chance 25.00%)"
]
open(OUT_MD, 'w').write('\n'.join(lines))
print(f'   saved {OUT_MD}')
print('Done.')
