"""Reproduce Table 5 / Supp §E: per-generator accuracy of Pact, Prism, GenStack-15.

Also prints lift = GenStack-15 - max(Pact, Prism) per generator.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from _repro_common import load_cached, split_arrays


def main(cache_dir: Path, K: int, seed: int):
    rows, clip = load_cached(cache_dir)
    gens, y, pv, bp, splits = split_arrays(rows)
    gen_list = sorted(set(gens.tolist()))
    gen2id = {g: i for i, g in enumerate(gen_list)}
    X2 = np.stack([pv, bp.astype(np.float32)], axis=1)

    gen_mean = np.array([clip[gens == g].mean(axis=0) for g in gen_list])
    cluster_of_gen = KMeans(n_clusters=K, random_state=seed,
                            n_init=10).fit_predict(gen_mean)
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
    gs = (oof >= 0.5).astype(int)
    pv_pred = (pv >= 0.5).astype(int)

    print(
        f"{'generator':<22}  {'n':>5}  {'Pact':>6}  {'Prism':>6}  {'GS-15':>6}  {'Lift':>6}"
    )
    for g in gen_list:
        m = gens == g
        n = int(m.sum())
        a_pv = (pv_pred[m] == y[m]).mean() * 100
        a_bp = (bp[m] == y[m]).mean() * 100
        a_gs = (gs[m] == y[m]).mean() * 100
        lift = a_gs - max(a_pv, a_bp)
        print(
            f"{g:<22}  {n:>5}  {a_pv:>6.2f}  {a_bp:>6.2f}  {a_gs:>6.2f}  {lift:>+6.2f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",
                    default="reproduce/cached_predictions",
                    type=Path)
    ap.add_argument("--K", default=15, type=int)
    ap.add_argument("--seed", default=42, type=int)
    a = ap.parse_args()
    main(a.cache, a.K, a.seed)
