"""Reproduce the GenStack-15 headline (Avg = 92.83) from cached predictions.

Run:
    python pact/scripts/reproduce_genstack_k15.py --cache reproduce/cached_predictions

Inputs (under --cache):
    c1e_merged_paths.json  : 52,266 rows of {path, v5, prism, label, generator, split}
    c1e_clip_chunks/*.npy  : per-sample CLIP-CLS features (768-d)

Outputs:
    Per-split + Avg accuracy printed to stdout.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def load_cached(cache_dir: Path):
    merged = json.load(open(cache_dir / "c1e_merged_paths.json"))
    chunks = sorted(
        glob.glob(str(cache_dir / "c1e_clip_chunks" / "feat_*.npy")))
    N = len(merged)
    clip = np.full((N, 768), np.nan, dtype=np.float32)
    for npy in chunks:
        info = json.load(open(npy.replace(".npy", ".json")))
        feat = np.load(npy)
        s = info["slice_start"]
        for j, k in enumerate(info["kept_idx_in_slice"]):
            clip[s + k] = feat[j]
    keep = np.where(~np.isnan(clip[:, 0]))[0]
    merged_ok = [merged[i] for i in keep]
    clip = clip[keep]
    clip = clip / (np.linalg.norm(clip, axis=1, keepdims=True) + 1e-8)
    return merged_ok, clip


def run(cache_dir: Path, K: int, seed: int) -> dict:
    rows, clip = load_cached(cache_dir)
    gens = np.array([r["generator"] for r in rows])
    gen_list = sorted(set(gens.tolist()))
    gen2id = {g: i for i, g in enumerate(gen_list)}
    y = np.array([r["label"] for r in rows])
    pv = np.array([r["v5"] for r in rows], dtype=np.float32)
    bp = np.array([r["prism"] for r in rows], dtype=np.int64)
    splits = np.array([r["split"] for r in rows])
    X2 = np.stack([pv, bp.astype(np.float32)], axis=1)

    gen_mean = np.array([clip[gens == g].mean(axis=0) for g in gen_list])
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    cluster_of_gen = km.fit_predict(gen_mean)
    sample_cluster = np.array([cluster_of_gen[gen2id[g]] for g in gens])

    NM = len(y)
    oof = np.zeros(NM)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X2, y):
        scl = StandardScaler().fit(clip[tr])
        gate = LogisticRegression(max_iter=1500, C=1.0,
                                  n_jobs=-1).fit(scl.transform(clip[tr]),
                                                 sample_cluster[tr])
        experts = [None] * K
        for c in range(K):
            mc = sample_cluster[tr] == c
            if mc.sum() < 50 or y[tr][mc].mean() in (0, 1):
                continue
            experts[c] = GradientBoostingClassifier(n_estimators=50,
                                                    max_depth=2,
                                                    learning_rate=0.1,
                                                    random_state=seed).fit(
                                                        X2[tr][mc], y[tr][mc])
        eg = np.zeros((len(te), K))
        for c in range(K):
            eg[:, c] = experts[c].predict_proba(
                X2[te])[:, 1] if experts[c] else X2[te, 0]
        proba = gate.predict_proba(scl.transform(clip[te]))
        full = np.zeros((len(te), K))
        for ci, c in enumerate(gate.classes_):
            full[:, c] = proba[:, ci]
        oof[te] = (full * eg).sum(axis=1)
    pred = (oof >= 0.5).astype(int)
    out = {
        s: float((pred[splits == s] == y[splits == s]).mean()) * 100
        for s in ("id", "cm", "cf", "cd")
    }
    out["avg"] = float((pred == y).mean()) * 100
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",
                    default="reproduce/cached_predictions",
                    type=Path)
    ap.add_argument("--K", default=15, type=int)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    res = run(args.cache, args.K, args.seed)
    print(f"GenStack-{args.K}  ID={res['id']:.2f}  CM={res['cm']:.2f}  "
          f"CF={res['cf']:.2f}  CD={res['cd']:.2f}  Avg={res['avg']:.2f}")
