"""Reproduce Table 3 / Supp §A: routing-granularity sweep over K.

Runs GenStack with K in {1, 4, 10, 12, 15, 18, 20, 23} from cached predictions.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from reproduce_genstack_k15 import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",
                    default="reproduce/cached_predictions",
                    type=Path)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--Ks", default="1,4,10,12,15,18,20,23")
    args = ap.parse_args()

    print(f"{'K':>3}  {'ID':>6}  {'CM':>6}  {'CF':>6}  {'CD':>6}  {'Avg':>6}")
    for K in [int(k) for k in args.Ks.split(",")]:
        r = run(args.cache, K, args.seed)
        print(
            f"{K:>3}  {r['id']:>6.2f}  {r['cm']:>6.2f}  {r['cf']:>6.2f}  {r['cd']:>6.2f}  {r['avg']:>6.2f}"
        )
