"""
GenStack ablations Tier A + Tier B (cached-prediction-only versions).

Reads the same cached prediction sources used by full_per_gen_breakdown.py and
runs every ablation that does NOT require re-training the base detectors.

Outputs:
  /home/sachin.chaudhary/xgendet/checkpoints/final_results/ablation_results.json
  /home/sachin.chaudhary/xgendet/checkpoints/final_results/ablation_summary.md

Ablations covered:
  A1  branch removal: PACT only / Prism only / both no-MoE / both with MoE
  A2  number of experts: 1 (global) / 4 (per-split) / 23 (per-generator)
  A3  threshold sensitivity at t in [0.30, 0.35, ..., 0.70]
  A4  K-fold sensitivity: K = 3 / 5 / 10 (per-generator GB)
  A5  meta-learner choice: Logistic Regression vs Random Forest vs Gradient Boosting
  B2  leave-one-generator-out: train MoE on 22 generators, eval on the held-out one
"""

import json
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# ─── 1. Load cached probs (same sources as full_per_gen_breakdown.py) ────────
print("[1/8] Loading cached predictions...")
v5_probs = json.load(
    open(
        '/home/sachin.chaudhary/xgendet/checkpoints/ensemble_v5_prism/v5_probs.json'
    ))

_ANS = re.compile(r'<answer>\s*(real|fake)\s*</answer>', re.I)


def parse_dir(d):
    out = {}
    for f in glob(f"{d}/prism_*.jsonl"):
        bn = Path(f).name
        if bn.startswith(('prism_v2_', )):
            continue
        name = bn.replace('.jsonl', '').split('-')[0].replace('prism_', '', 1)
        if name.startswith('mipo_'):
            name = name[5:]
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
                out[p] = {
                    'pred': 1 if m.group(1).lower() == 'fake' else 0,
                    'label': int(d_['label']),
                    'generator': gen,
                    'split': gen.split('_')[0],
                }
    return out


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

# Merge — only samples in both PACT and Prism
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
        'split': d['split'],
    })

print(f"   merged samples: {len(merged)}")

X = np.array([[r['v5'], r['prism']] for r in merged])
y = np.array([r['label'] for r in merged])
gens = np.array([r['generator'] for r in merged])
splits = np.array([r['split'] for r in merged])
v5_arr = X[:, 0]
prism_arr = X[:, 1]


# ─── helpers ──────────────────────────────────────────────────────────────
def acc_prob(p, l):
    return float(((p >= 0.5).astype(int) == l).mean())


def acc_bin(p, l):
    return float((p.astype(int) == l).mean())


def per_split_acc(pred_continuous, threshold=0.5):
    out = {}
    for s in ('id', 'cm', 'cf', 'cd'):
        m = splits == s
        if m.sum() == 0:
            out[s] = None
        else:
            out[s] = float(((pred_continuous[m]
                             >= threshold).astype(int) == y[m]).mean())
    out['overall'] = float(((pred_continuous
                             >= threshold).astype(int) == y).mean())
    return out


def per_gen_kfold_oof(
    X_,
    y_,
    gens_,
    K=5,
    model_factory=lambda: GradientBoostingClassifier(
        n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)):
    """Per-generator K-fold out-of-fold predictions."""
    oof = np.zeros(len(y_))
    for g in sorted(set(gens_)):
        m = gens_ == g
        idx = np.where(m)[0]
        Xs, ys = X_[idx], y_[idx]
        if m.sum() < 50 or ys.mean() in (0, 1):
            pseudo = np.where(X_[idx, 1] == 1, 0.9, 0.1)
            oof[idx] = 0.5 * X_[idx, 0] + 0.5 * pseudo
            continue
        n_fold = max(2, min(K, int(m.sum() // 20)))
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
        for tr, te in kf.split(Xs):
            if y_[idx[tr]].mean() in (0, 1):
                pseudo = np.where(X_[idx[te], 1] == 1, 0.9, 0.1)
                oof[idx[te]] = 0.5 * X_[idx[te], 0] + 0.5 * pseudo
                continue
            mdl = model_factory()
            mdl.fit(Xs[tr], y_[idx[tr]])
            oof[idx[te]] = mdl.predict_proba(Xs[te])[:, 1]
    return oof


def per_split_kfold_oof(X_, y_, splits_, K=5):
    """Per-split K-fold out-of-fold predictions (only 4 experts: ID/CM/CF/CD)."""
    oof = np.zeros(len(y_))
    for s in ('id', 'cm', 'cf', 'cd'):
        m = splits_ == s
        idx = np.where(m)[0]
        Xs, ys = X_[idx], y_[idx]
        if m.sum() < 50 or ys.mean() in (0, 1):
            pseudo = np.where(X_[idx, 1] == 1, 0.9, 0.1)
            oof[idx] = 0.5 * X_[idx, 0] + 0.5 * pseudo
            continue
        kf = KFold(n_splits=K, shuffle=True, random_state=42)
        for tr, te in kf.split(Xs):
            mdl = GradientBoostingClassifier(n_estimators=50,
                                             max_depth=2,
                                             learning_rate=0.1,
                                             random_state=42)
            mdl.fit(Xs[tr], y_[idx[tr]])
            oof[idx[te]] = mdl.predict_proba(Xs[te])[:, 1]
    return oof


def global_kfold_oof(X_, y_, K=5):
    """Single global K-fold out-of-fold predictions (1 expert)."""
    oof = np.zeros(len(y_))
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    for tr, te in kf.split(X_):
        mdl = GradientBoostingClassifier(n_estimators=50,
                                         max_depth=2,
                                         learning_rate=0.1,
                                         random_state=42)
        mdl.fit(X_[tr], y_[tr])
        oof[te] = mdl.predict_proba(X_[te])[:, 1]
    return oof


# ─── Reference: per-generator GB OOF (the headline GenStack number) ─────────
print("[2/8] Computing reference per-generator GB OOF...")
oof_pergen = per_gen_kfold_oof(X, y, gens, K=5)
ref_acc = per_split_acc(oof_pergen)
print(f"   reference GenStack overall accuracy: {ref_acc['overall']*100:.2f}%")

results = {}

# ─── A1. Branch removal ─────────────────────────────────────────────────────
print("[3/8] A1 branch-removal ablation...")
results['A1_branch_removal'] = {
    'pact_only': per_split_acc(v5_arr),
    'prism_only': {
        **{
            s: acc_bin(prism_arr[splits == s], y[splits == s])
            for s in ('id', 'cm', 'cf', 'cd')
        }, 'overall': acc_bin(prism_arr, y)
    },
    'simple_average': per_split_acc(0.5 * v5_arr + 0.5 * prism_arr),
    'genstack_pergen': ref_acc,
}
for k, v in results['A1_branch_removal'].items():
    print(f"   {k:<22} overall={v['overall']*100:5.2f}%")

# ─── A2. Number of experts ─────────────────────────────────────────────────
print("[4/8] A2 routing-granularity ablation...")
oof_global = global_kfold_oof(X, y, K=5)
oof_persplit = per_split_kfold_oof(X, y, splits, K=5)
results['A2_routing_granularity'] = {
    '1_global': per_split_acc(oof_global),
    '4_per_split': per_split_acc(oof_persplit),
    '23_per_generator': ref_acc,
}
for k, v in results['A2_routing_granularity'].items():
    print(f"   {k:<22} overall={v['overall']*100:5.2f}%")

# ─── A3. Threshold sensitivity ────────────────────────────────────────────
print("[5/8] A3 threshold sensitivity...")
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
results['A3_threshold'] = {
    f'{t:.2f}': per_split_acc(oof_pergen, threshold=t)
    for t in thresholds
}
for t, v in results['A3_threshold'].items():
    print(f"   t={t}                overall={v['overall']*100:5.2f}%")

# ─── A4. Number of CV folds ────────────────────────────────────────────────
print("[6/8] A4 CV fold sensitivity...")
results['A4_cv_folds'] = {}
for K in (3, 5, 10):
    oof_K = per_gen_kfold_oof(X, y, gens, K=K)
    results['A4_cv_folds'][f'K={K}'] = per_split_acc(oof_K)
    print(
        f"   K={K}                  overall={results['A4_cv_folds'][f'K={K}']['overall']*100:5.2f}%"
    )

# ─── A5. Meta-learner choice ───────────────────────────────────────────────
print("[7/8] A5 meta-learner choice...")
results['A5_meta_learner'] = {}
for name, factory in (
    ('LogisticRegression',
     lambda: LogisticRegression(max_iter=200, random_state=42)),
    ('RandomForest', lambda: RandomForestClassifier(
        n_estimators=50, max_depth=4, random_state=42)),
    ('GradientBoosting', lambda: GradientBoostingClassifier(
        n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42)),
):
    oof_m = per_gen_kfold_oof(X, y, gens, K=5, model_factory=factory)
    results['A5_meta_learner'][name] = per_split_acc(oof_m)
    print(
        f"   {name:<22} overall={results['A5_meta_learner'][name]['overall']*100:5.2f}%"
    )

# ─── B2. Leave-one-generator-out ────────────────────────────────────────────
print("[8/8] B2 leave-one-generator-out...")
unique_gens = sorted(set(gens))
results['B2_leave_one_gen_out'] = {}
held_out_accs = []
for g in unique_gens:
    train_mask = gens != g
    test_mask = gens == g
    if test_mask.sum() < 5:
        continue
    if y[train_mask].mean() in (0, 1):
        continue
    # Single global GB trained on the other 22 generators
    mdl = GradientBoostingClassifier(n_estimators=50,
                                     max_depth=2,
                                     learning_rate=0.1,
                                     random_state=42)
    mdl.fit(X[train_mask], y[train_mask])
    pred = mdl.predict_proba(X[test_mask])[:, 1]
    a = acc_prob(pred, y[test_mask])
    pergen_a = ref_acc.get(g, None)
    # also compute the per-generator OOF accuracy on this generator (for context)
    in_gen_acc = float(((oof_pergen[test_mask]
                         >= 0.5).astype(int) == y[test_mask]).mean())
    results['B2_leave_one_gen_out'][g] = {
        'lo_gen_out_acc': a,
        'in_distribution_acc': in_gen_acc,
        'n': int(test_mask.sum()),
    }
    held_out_accs.append(a)

mean_lo = float(np.mean(held_out_accs))
results['B2_leave_one_gen_out']['_mean_held_out'] = mean_lo
results['B2_leave_one_gen_out']['_mean_in_distribution'] = ref_acc['overall']
print(f"   mean held-out acc:        {mean_lo*100:5.2f}%")
print(f"   mean in-distribution acc: {ref_acc['overall']*100:5.2f}%")

# ─── Save ──────────────────────────────────────────────────────────────────
out_dir = Path('/home/sachin.chaudhary/xgendet/checkpoints/final_results')
out_dir.mkdir(parents=True, exist_ok=True)
out_json = out_dir / 'ablation_results.json'
out_json.write_text(json.dumps(results, indent=2))
print(f"\nSaved JSON: {out_json}")


# ─── Summary markdown ──────────────────────────────────────────────────────
def fmt_split(d):
    return (f"ID={d['id']*100:5.2f}  CM={d['cm']*100:5.2f}  "
            f"CF={d['cf']*100:5.2f}  CD={d['cd']*100:5.2f}  "
            f"Avg={d['overall']*100:5.2f}")


lines = [
    "# GenStack ablations — Tier A + B2", "",
    f"Total merged samples: **{len(merged)}**", ""
]

lines.append("## A1 — Branch removal")
lines.append("")
lines.append("| Variant | ID | CM | CF | CD | Overall |")
lines.append("|---|---:|---:|---:|---:|---:|")
for k in ('pact_only', 'prism_only', 'simple_average', 'genstack_pergen'):
    v = results['A1_branch_removal'][k]
    lines.append(f"| {k} | {v['id']*100:.2f} | {v['cm']*100:.2f} | "
                 f"{v['cf']*100:.2f} | {v['cd']*100:.2f} | "
                 f"**{v['overall']*100:.2f}** |")
lines.append("")

lines.append("## A2 — Routing granularity (number of experts)")
lines.append("")
lines.append("| Variant | ID | CM | CF | CD | Overall |")
lines.append("|---|---:|---:|---:|---:|---:|")
for k in ('1_global', '4_per_split', '23_per_generator'):
    v = results['A2_routing_granularity'][k]
    lines.append(f"| {k} | {v['id']*100:.2f} | {v['cm']*100:.2f} | "
                 f"{v['cf']*100:.2f} | {v['cd']*100:.2f} | "
                 f"**{v['overall']*100:.2f}** |")
lines.append("")

lines.append("## A3 — Threshold sensitivity (per-generator GB)")
lines.append("")
lines.append("| t | ID | CM | CF | CD | Overall |")
lines.append("|---|---:|---:|---:|---:|---:|")
for t in thresholds:
    v = results['A3_threshold'][f'{t:.2f}']
    lines.append(f"| {t:.2f} | {v['id']*100:.2f} | {v['cm']*100:.2f} | "
                 f"{v['cf']*100:.2f} | {v['cd']*100:.2f} | "
                 f"**{v['overall']*100:.2f}** |")
lines.append("")

lines.append("## A4 — CV-fold sensitivity")
lines.append("")
lines.append("| K | ID | CM | CF | CD | Overall |")
lines.append("|---|---:|---:|---:|---:|---:|")
for K in (3, 5, 10):
    v = results['A4_cv_folds'][f'K={K}']
    lines.append(f"| {K} | {v['id']*100:.2f} | {v['cm']*100:.2f} | "
                 f"{v['cf']*100:.2f} | {v['cd']*100:.2f} | "
                 f"**{v['overall']*100:.2f}** |")
lines.append("")

lines.append("## A5 — Meta-learner choice")
lines.append("")
lines.append("| Meta-learner | ID | CM | CF | CD | Overall |")
lines.append("|---|---:|---:|---:|---:|---:|")
for name in ('LogisticRegression', 'RandomForest', 'GradientBoosting'):
    v = results['A5_meta_learner'][name]
    lines.append(f"| {name} | {v['id']*100:.2f} | {v['cm']*100:.2f} | "
                 f"{v['cf']*100:.2f} | {v['cd']*100:.2f} | "
                 f"**{v['overall']*100:.2f}** |")
lines.append("")

lines.append("## B2 — Leave-one-generator-out")
lines.append("")
lines.append(
    "Train MoE on the other 22 generators, evaluate on the held-out one.")
lines.append(
    "Compares against the in-distribution per-generator GB OOF accuracy on the same images."
)
lines.append("")
lines.append(f"- **Mean held-out accuracy:** {mean_lo*100:.2f}%")
lines.append(
    f"- **Mean in-distribution accuracy:** {ref_acc['overall']*100:.2f}%")
lines.append("")
lines.append("| Generator | n | LO-gen-out acc | In-dist acc | Δ |")
lines.append("|---|---:|---:|---:|---:|")
for g in sorted(unique_gens):
    if g not in results['B2_leave_one_gen_out']:
        continue
    rec = results['B2_leave_one_gen_out'][g]
    delta = (rec['lo_gen_out_acc'] - rec['in_distribution_acc']) * 100
    lines.append(f"| {g} | {rec['n']} | {rec['lo_gen_out_acc']*100:.2f} | "
                 f"{rec['in_distribution_acc']*100:.2f} | {delta:+.2f} |")

out_md = out_dir / 'ablation_summary.md'
out_md.write_text('\n'.join(lines))
print(f"Saved Markdown: {out_md}")
print("\nDone.")
