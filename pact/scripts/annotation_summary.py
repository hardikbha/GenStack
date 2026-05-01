"""
XGenDet: Annotation Summary Analysis

Computes:
  1. Average attribute scores for real vs fake across generators
  2. Gemini prediction accuracy (does it correctly identify real/fake?)
  3. Summary table printed to stdout

Usage:
  python scripts/annotation_summary.py
  python scripts/annotation_summary.py --annotations_dir annotations/
"""

import os
import json
import argparse
from collections import defaultdict


ATTRIBUTES = [
    'texture_consistency',
    'edge_quality',
    'color_distribution',
    'geometric_coherence',
    'semantic_plausibility',
    'frequency_artifacts',
]

GENERATORS = ['BigGAN', 'ADM', 'LDM', 'Midjourney']


def load_annotations(annotations_dir):
    """Load all JSONL annotation files from a directory."""
    entries = []
    for fname in sorted(os.listdir(annotations_dir)):
        if not fname.endswith('.jsonl'):
            continue
        # Skip test files
        if 'test' in fname.lower():
            continue
        fpath = os.path.join(annotations_dir, fname)
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    return entries


def compute_summary(entries):
    """Compute summary statistics from annotation entries."""
    # Group by generator and ground_truth
    groups = defaultdict(list)
    for e in entries:
        gen = e.get('generator', 'unknown')
        gt = e.get('ground_truth', 'unknown')
        groups[(gen, gt)].append(e)

    return groups


def print_summary_table(groups):
    """Print a formatted summary table."""
    print("=" * 120)
    print("XGENDET GEMINI ANNOTATION SUMMARY")
    print("=" * 120)

    # ---- Section 1: Gemini Prediction Accuracy ----
    print("\n--- Gemini Prediction Accuracy (per generator) ---\n")
    print(f"{'Generator':<12} {'Label':<6} {'N':>4} {'Correct':>8} {'Accuracy':>10} {'Avg Conf':>10}")
    print("-" * 60)

    overall_correct = 0
    overall_total = 0

    for gen in GENERATORS:
        for gt in ['fake', 'real']:
            key = (gen, gt)
            if key not in groups:
                continue
            entries = groups[key]
            n = len(entries)

            # Determine correctness
            correct = 0
            total_conf = 0.0
            for e in entries:
                pred = e.get('gemini_prediction', '').upper()
                conf = e.get('gemini_confidence', 0.0)
                total_conf += conf

                if gt == 'fake' and pred == 'FAKE':
                    correct += 1
                elif gt == 'real' and pred == 'REAL':
                    correct += 1

            acc = correct / n if n > 0 else 0.0
            avg_conf = total_conf / n if n > 0 else 0.0
            overall_correct += correct
            overall_total += n

            print(f"{gen:<12} {gt:<6} {n:>4} {correct:>8} {acc:>9.1%} {avg_conf:>10.2f}")

    if overall_total > 0:
        print("-" * 60)
        print(f"{'OVERALL':<12} {'':>6} {overall_total:>4} {overall_correct:>8} {overall_correct/overall_total:>9.1%}")

    # ---- Section 2: Average Attribute Scores ----
    print("\n\n--- Average Attribute Scores: FAKE images (per generator) ---\n")
    header = f"{'Generator':<12}"
    for attr in ATTRIBUTES:
        short = attr[:10]
        header += f" {short:>10}"
    print(header)
    print("-" * (12 + 11 * len(ATTRIBUTES)))

    for gen in GENERATORS:
        key = (gen, 'fake')
        if key not in groups:
            continue
        entries = groups[key]
        row = f"{gen:<12}"
        for attr in ATTRIBUTES:
            vals = [e.get('attributes', {}).get(attr, 0.0) for e in entries]
            avg = sum(vals) / len(vals) if vals else 0.0
            row += f" {avg:>10.3f}"
        print(row)

    print("\n\n--- Average Attribute Scores: REAL images (per generator) ---\n")
    print(header)
    print("-" * (12 + 11 * len(ATTRIBUTES)))

    for gen in GENERATORS:
        key = (gen, 'real')
        if key not in groups:
            continue
        entries = groups[key]
        row = f"{gen:<12}"
        for attr in ATTRIBUTES:
            vals = [e.get('attributes', {}).get(attr, 0.0) for e in entries]
            avg = sum(vals) / len(vals) if vals else 0.0
            row += f" {avg:>10.3f}"
        print(row)

    # ---- Section 3: Aggregate comparison ----
    print("\n\n--- Aggregate: FAKE vs REAL attribute scores (all generators) ---\n")
    print(f"{'Attribute':<25} {'Fake (avg)':>12} {'Real (avg)':>12} {'Diff (F-R)':>12}")
    print("-" * 65)

    all_fake = [e for e in sum([groups.get((g, 'fake'), []) for g in GENERATORS], []) ]
    all_real = [e for e in sum([groups.get((g, 'real'), []) for g in GENERATORS], []) ]

    for attr in ATTRIBUTES:
        fake_vals = [e.get('attributes', {}).get(attr, 0.0) for e in all_fake]
        real_vals = [e.get('attributes', {}).get(attr, 0.0) for e in all_real]
        fake_avg = sum(fake_vals) / len(fake_vals) if fake_vals else 0.0
        real_avg = sum(real_vals) / len(real_vals) if real_vals else 0.0
        diff = fake_avg - real_avg
        print(f"{attr:<25} {fake_avg:>12.3f} {real_avg:>12.3f} {diff:>+12.3f}")

    # ---- Section 4: Per-generator detection rate breakdown ----
    print("\n\n--- Detection Rate Breakdown ---\n")
    print(f"{'Generator':<12} {'Fake->FAKE':>12} {'Fake->REAL':>12} {'Real->REAL':>12} {'Real->FAKE':>12}")
    print("-" * 64)

    for gen in GENERATORS:
        fake_entries = groups.get((gen, 'fake'), [])
        real_entries = groups.get((gen, 'real'), [])

        ff = sum(1 for e in fake_entries if e.get('gemini_prediction', '').upper() == 'FAKE')
        fr = sum(1 for e in fake_entries if e.get('gemini_prediction', '').upper() == 'REAL')
        rr = sum(1 for e in real_entries if e.get('gemini_prediction', '').upper() == 'REAL')
        rf = sum(1 for e in real_entries if e.get('gemini_prediction', '').upper() == 'FAKE')

        print(f"{gen:<12} {ff:>12} {fr:>12} {rr:>12} {rf:>12}")

    print("\n" + "=" * 120)
    print("NOTE: Attribute scores range 0.0 (no artifacts) to 1.0 (severe artifacts).")
    print("Higher scores for FAKE images suggest Gemini detects more artifacts in AI-generated content.")
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description='XGenDet Annotation Summary')
    parser.add_argument('--annotations_dir', type=str, default='annotations/',
                        help='Directory containing JSONL annotation files')
    args = parser.parse_args()

    entries = load_annotations(args.annotations_dir)
    print(f"Loaded {len(entries)} annotations from {args.annotations_dir}\n")

    groups = compute_summary(entries)
    print_summary_table(groups)


if __name__ == '__main__':
    main()
