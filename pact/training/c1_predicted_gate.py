"""C1 — Predicted-gate routing (deployable open-world variant).

Reviewer concern: the headline 92.42% uses a manifest-aware oracle gate. We
defend with three deployable variants — train a learned gate classifier on
(p_v, b_p) and route to the matching expert. Outputs go in
/home/sachin.chaudhary/xgendet/checkpoints/final_results/c1_predicted_gate.{json,md}.

Reports:
  - Oracle 23-way (manifest, current headline)
  - Predicted 23-gen gate (hard argmax)        ← deployable
  - Predicted 23-gen gate (soft mixture)       ← deployable
  - Predicted 4-split gate (hard argmax)       ← deployable
  - Predicted 4-split gate (soft mixture)      ← deployable
  - Gate top-1 / top-3 accuracy (diagnostic: how learnable is the gate?)

Outer 5-fold StratifiedKFold over the binary label, single seed.
Experts: GradientBoosting depth-2, 50 trees (matches existing setup).
Gate classifier: HistGradientBoosting (handles multiclass natively, fast).
"""
import json
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
from sklearn.ensemble import (GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.model_selection import StratifiedKFold

# ─── load + merge cached predictions (same as b1_open_world_routing.py) ─────
print("[1/5] Loading cached predictions...")
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

gen_list = sorted(set(gens.tolist()))
gen2id = {g: i for i, g in enumerate(gen_list)}
gen_y = np.array([gen2id[g] for g in gens])
N_GEN = len(gen_list)

split_list = ['id', 'cm', 'cf', 'cd']
split_y = np.array([split_list.index(s) for s in splits])
N_SPLIT = 4
print(f"   {N_GEN} generators, {N_SPLIT} splits")


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


def expert_predict(e, Xs, Xs_pseudo):
    if e is None:
        return 0.5 * Xs[:, 0] + 0.5 * np.where(Xs[:, 1] == 1, 0.9, 0.1)
    return e.predict_proba(Xs)[:, 1]


# ─── Outer 5-fold over binary label, build OOF predictions ──────────────────
print("[2/5] Outer 5-fold OOF over binary label...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_oracle = np.zeros(len(y))
oof_pred23h = np.zeros(len(y))  # predicted 23-gate, HARD
oof_pred23s = np.zeros(len(y))  # predicted 23-gate, SOFT
oof_pred4h = np.zeros(len(y))  # predicted 4-split gate, HARD
oof_pred4s = np.zeros(len(y))  # predicted 4-split gate, SOFT

gate23_top1 = np.zeros(len(y), dtype=bool)
gate23_top3 = np.zeros(len(y), dtype=bool)
gate4_top1 = np.zeros(len(y), dtype=bool)

for fold, (tr, te) in enumerate(skf.split(X, y), 1):
    print(f"   fold {fold}/5  train={len(tr)}  test={len(te)}")

    # ── train per-generator experts on train portion ──
    experts_g = {}
    for gi in range(N_GEN):
        mask = gen_y[tr] == gi
        if mask.sum() == 0:
            experts_g[gi] = None
            continue
        experts_g[gi] = fit_expert(X[tr][mask], y[tr][mask])

    # ── train per-split experts on train portion ──
    experts_s = {}
    for si in range(N_SPLIT):
        mask = split_y[tr] == si
        experts_s[si] = fit_expert(X[tr][mask], y[tr][mask])

    # ── train gate classifiers on train portion ──
    gate23 = HistGradientBoostingClassifier(max_iter=120,
                                            max_depth=4,
                                            learning_rate=0.1,
                                            random_state=42)
    gate23.fit(X[tr], gen_y[tr])
    gate4 = HistGradientBoostingClassifier(max_iter=80,
                                           max_depth=3,
                                           learning_rate=0.1,
                                           random_state=42)
    gate4.fit(X[tr], split_y[tr])

    Xte = X[te]
    # ── ORACLE: route by true generator ──
    for j, i in enumerate(te):
        gi = gen_y[i]
        oof_oracle[i] = expert_predict(experts_g[gi], Xte[j:j + 1],
                                       Xte[j:j + 1])[0]

    # ── PRECOMPUTE expert outputs on test portion ──
    # for soft mixture, we need expert_g(x) and expert_s(x) for every x in te
    expert_g_pred = np.zeros((len(te), N_GEN))
    for gi in range(N_GEN):
        expert_g_pred[:, gi] = expert_predict(experts_g[gi], Xte, Xte)
    expert_s_pred = np.zeros((len(te), N_SPLIT))
    for si in range(N_SPLIT):
        expert_s_pred[:, si] = expert_predict(experts_s[si], Xte, Xte)

    # ── PRED 23 (hard + soft) ──
    proba23 = gate23.predict_proba(Xte)  # (n_te, N_GEN_seen)
    # Map gate23 classes back to gen_y space (HistGB exposes classes_)
    classes23 = gate23.classes_
    # build full-dim proba (some classes may be missing in train)
    full_proba23 = np.zeros((len(te), N_GEN))
    for ci, c in enumerate(classes23):
        full_proba23[:, c] = proba23[:, ci]
    pred23 = classes23[proba23.argmax(axis=1)]
    oof_pred23h[te] = expert_g_pred[np.arange(len(te)), pred23]
    oof_pred23s[te] = (full_proba23 * expert_g_pred).sum(axis=1)

    # gate top-1 / top-3 accuracy diagnostic
    gate23_top1[te] = pred23 == gen_y[te]
    top3 = np.argsort(-proba23, axis=1)[:, :3]
    top3_classes = classes23[top3]
    gate23_top3[te] = (top3_classes == gen_y[te][:, None]).any(axis=1)

    # ── PRED 4 (hard + soft) ──
    proba4 = gate4.predict_proba(Xte)
    classes4 = gate4.classes_
    full_proba4 = np.zeros((len(te), N_SPLIT))
    for ci, c in enumerate(classes4):
        full_proba4[:, c] = proba4[:, ci]
    pred4 = classes4[proba4.argmax(axis=1)]
    oof_pred4h[te] = expert_s_pred[np.arange(len(te)), pred4]
    oof_pred4s[te] = (full_proba4 * expert_s_pred).sum(axis=1)
    gate4_top1[te] = pred4 == split_y[te]

# ─── report ─────────────────────────────────────────────────────────────────
print("\n[3/5] Routing accuracy (binary, threshold 0.5):")
results = {
    'oracle_23gen': per_split_acc(oof_oracle),
    'pred23gen_hard': per_split_acc(oof_pred23h),
    'pred23gen_soft': per_split_acc(oof_pred23s),
    'pred4split_hard': per_split_acc(oof_pred4h),
    'pred4split_soft': per_split_acc(oof_pred4s),
}
for name, r in results.items():
    print(f"   {name:18s} ID {r['id']*100:5.2f}  CM {r['cm']*100:5.2f}  "
          f"CF {r['cf']*100:5.2f}  CD {r['cd']*100:5.2f}  "
          f"OVR {r['overall']*100:5.2f}")

print("\n[4/5] Gate diagnostics:")
gate_diag = {
    'gate23_top1_acc': float(gate23_top1.mean()),
    'gate23_top3_acc': float(gate23_top3.mean()),
    'gate4_top1_acc': float(gate4_top1.mean()),
    'random_chance_23': 1.0 / N_GEN,
    'random_chance_4': 1.0 / N_SPLIT,
}
for k, v in gate_diag.items():
    print(f"   {k:18s} {v*100:5.2f}%")

# ─── save ────────────────────────────────────────────────────────────────────
print("\n[5/5] Writing outputs...")
out = Path('/home/sachin.chaudhary/xgendet/checkpoints/final_results')
(out / 'c1_predicted_gate.json').write_text(
    json.dumps({
        'routing': results,
        'gate_diagnostics': gate_diag
    }, indent=2))

lines = [
    "# C1 — Predicted-gate (deployable open-world routing)",
    "",
    "Defends the oracle MoE: train a learned gate classifier on `(p_v, b_p)`",
    "and route to the matching expert. Outer 5-fold OOF over binary label.",
    "",
    "## Routing accuracy",
    "",
    "| Variant | ID | CM | CF | CD | Overall |",
    "|---|---:|---:|---:|---:|---:|",
]


def row(name, r):
    return (f"| {name} | {r['id']*100:.2f} | {r['cm']*100:.2f} | "
            f"{r['cf']*100:.2f} | {r['cd']*100:.2f} | "
            f"**{r['overall']*100:.2f}** |")


lines.append(row("Oracle 23-gen (manifest)", results['oracle_23gen']))
lines.append(row("Pred 23-gen gate (hard)", results['pred23gen_hard']))
lines.append(row("Pred 23-gen gate (soft)", results['pred23gen_soft']))
lines.append(row("Pred 4-split gate (hard)", results['pred4split_hard']))
lines.append(row("Pred 4-split gate (soft)", results['pred4split_soft']))
lines.append("")
lines.append("## Gate diagnostics")
lines.append("")
lines.append(
    f"- 23-way gate top-1 acc: **{gate_diag['gate23_top1_acc']*100:.2f}%** "
    f"(chance {gate_diag['random_chance_23']*100:.2f}%)")
lines.append(
    f"- 23-way gate top-3 acc: **{gate_diag['gate23_top3_acc']*100:.2f}%**")
lines.append(
    f"-  4-way gate top-1 acc: **{gate_diag['gate4_top1_acc']*100:.2f}%** "
    f"(chance {gate_diag['random_chance_4']*100:.2f}%)")

(out / 'c1_predicted_gate.md').write_text('\n'.join(lines))
print(f"   saved {out / 'c1_predicted_gate.md'}")
print("Done.")
