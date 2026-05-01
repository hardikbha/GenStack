"""Reproduce Supp §M: GenStack-15 cross-seed stability over seeds {42, 0, 7}."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from reproduce_genstack_k15 import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",
                    default="reproduce/cached_predictions",
                    type=Path)
    ap.add_argument("--seeds", default="42,0,7")
    ap.add_argument("--K", default=15, type=int)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rs = [run(args.cache, args.K, s) for s in seeds]
    print(
        f"{'seed':>5}  {'ID':>6}  {'CM':>6}  {'CF':>6}  {'CD':>6}  {'Avg':>6}")
    for s, r in zip(seeds, rs):
        print(
            f"{s:>5}  {r['id']:>6.2f}  {r['cm']:>6.2f}  {r['cf']:>6.2f}  {r['cd']:>6.2f}  {r['avg']:>6.2f}"
        )
    print()
    for k in ("id", "cm", "cf", "cd", "avg"):
        vals = [r[k] for r in rs]
        print(f"  {k:>5}: mean={np.mean(vals):.2f}  std={np.std(vals):.2f}")
