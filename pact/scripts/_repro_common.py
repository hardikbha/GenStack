"""Shared cached-prediction loader for reproduction scripts."""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np


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
    rows = [merged[i] for i in keep]
    clip = clip[keep]
    clip = clip / (np.linalg.norm(clip, axis=1, keepdims=True) + 1e-8)
    return rows, clip


def split_arrays(rows):
    gens = np.array([r["generator"] for r in rows])
    y = np.array([r["label"] for r in rows])
    pv = np.array([r["v5"] for r in rows], dtype=np.float32)
    bp = np.array([r["prism"] for r in rows], dtype=np.int64)
    splits = np.array([r["split"] for r in rows])
    return gens, y, pv, bp, splits


def per_split_acc(pred, y, splits):
    out = {
        s: float((pred[splits == s] == y[splits == s]).mean()) * 100
        for s in ("id", "cm", "cf", "cd")
    }
    out["avg"] = float((pred == y).mean()) * 100
    return out


def fmt(res, prefix=""):
    return (f"{prefix}ID={res['id']:.2f}  CM={res['cm']:.2f}  "
            f"CF={res['cf']:.2f}  CD={res['cd']:.2f}  Avg={res['avg']:.2f}")
