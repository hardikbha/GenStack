"""Aggregate GenStack robustness JSONLs into a summary table."""
import json
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path(
    "/home/sachin.chaudhary/xgendet/checkpoints/final_results/genstack_robust_out"
)
CONDITIONS = [
    "orig", "jpeg_90", "jpeg_70", "jpeg_60", "blur_1", "blur_2", "blur_4"
]

THRESHOLDS = {"pact": 0.5, "prism": 0.5}


def calc_accuracy(records, scorer):
    by_split = defaultdict(lambda: [0, 0])
    overall = [0, 0]
    for r in records:
        if r.get("error") or r.get("p_v", -1) < 0:
            continue
        pred = scorer(r)
        if pred is None:
            continue
        ok = int(pred == r["label"])
        overall[0] += ok
        overall[1] += 1
        by_split[r["split"]][0] += ok
        by_split[r["split"]][1] += 1
    return {
        "overall": overall[0] / max(overall[1], 1),
        "n": overall[1],
        "by_split": {
            k: v[0] / max(v[1], 1)
            for k, v in by_split.items()
        },
        "n_by_split": {
            k: v[1]
            for k, v in by_split.items()
        },
    }


def pact_score(r):
    return int(r["p_v"] >= 0.5)


def prism_score(r):
    if r.get("b_p", -1) < 0:
        return None
    return int(r["b_p"])


def avg_score(r):
    if r.get("b_p", -1) < 0:
        return None
    avg = 0.5 * r["p_v"] + 0.5 * r["b_p"]
    return int(avg >= 0.5)


def main():
    summary = {}
    for cond in CONDITIONS:
        path = OUT_DIR / f"{cond}.jsonl"
        if not path.exists():
            print(f"[skip] {cond} (no file)")
            continue
        with open(path) as f:
            records = [json.loads(line) for line in f]
        if not records:
            continue
        summary[cond] = {
            "pact": calc_accuracy(records, pact_score),
            "prism": calc_accuracy(records, prism_score),
            "avg": calc_accuracy(records, avg_score),
            "n_total": len(records),
            "n_errors": sum(1 for r in records if r.get("error")),
            "n_prism_failed": sum(1 for r in records if r.get("b_p", 0) == -1),
        }

    json.dump(summary,
              open(OUT_DIR / "genstack_robust_summary.json", "w"),
              indent=2)

    # Markdown
    md = ["# Full GenStack Robustness Summary", ""]
    md.append(
        f"Subset: 1500 stratified samples across 23 generators × 4 splits.")
    md.append("")
    md.append("## PACT-only (p_v ≥ 0.5)")
    md.append("| Condition | ID | CM | CF | CD | **Avg** | n |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for cond in CONDITIONS:
        if cond not in summary:
            continue
        s = summary[cond]["pact"]
        bs = s["by_split"]
        md.append(
            f"| {cond} | {bs.get('id',0)*100:.2f} | {bs.get('cm',0)*100:.2f} | "
            f"{bs.get('cf',0)*100:.2f} | {bs.get('cd',0)*100:.2f} | "
            f"**{s['overall']*100:.2f}** | {s['n']} |")

    md.append("")
    md.append("## Prism-only (b_p; verdict from <answer> tag)")
    md.append("| Condition | ID | CM | CF | CD | **Avg** | n | n_failed |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cond in CONDITIONS:
        if cond not in summary:
            continue
        s = summary[cond]["prism"]
        bs = s["by_split"]
        md.append(
            f"| {cond} | {bs.get('id',0)*100:.2f} | {bs.get('cm',0)*100:.2f} | "
            f"{bs.get('cf',0)*100:.2f} | {bs.get('cd',0)*100:.2f} | "
            f"**{s['overall']*100:.2f}** | {s['n']} | "
            f"{summary[cond]['n_prism_failed']} |")

    md.append("")
    md.append("## GenStack (simple avg of p_v and b_p)")
    md.append("| Condition | ID | CM | CF | CD | **Avg** |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for cond in CONDITIONS:
        if cond not in summary:
            continue
        s = summary[cond]["avg"]
        bs = s["by_split"]
        md.append(
            f"| {cond} | {bs.get('id',0)*100:.2f} | {bs.get('cm',0)*100:.2f} | "
            f"{bs.get('cf',0)*100:.2f} | {bs.get('cd',0)*100:.2f} | "
            f"**{s['overall']*100:.2f}** |")

    open(OUT_DIR / "genstack_robust_summary.md", "w").write("\n".join(md))
    print(f"Wrote {OUT_DIR/'genstack_robust_summary.json'}")
    print(f"Wrote {OUT_DIR/'genstack_robust_summary.md'}")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
