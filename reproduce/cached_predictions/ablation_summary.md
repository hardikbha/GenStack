# GenStack ablations — Tier A + B2

Total merged samples: **52266**

## A1 — Branch removal

| Variant | ID | CM | CF | CD | Overall |
|---|---:|---:|---:|---:|---:|
| pact_only | 86.59 | 98.98 | 79.52 | 77.85 | **84.95** |
| prism_only | 94.90 | 98.30 | 84.91 | 78.08 | **88.22** |
| simple_average | 94.90 | 98.30 | 84.91 | 78.08 | **88.22** |
| genstack_pergen | 94.93 | 99.66 | 93.57 | 84.12 | **92.42** |

## A2 — Routing granularity (number of experts)

| Variant | ID | CM | CF | CD | Overall |
|---|---:|---:|---:|---:|---:|
| 1_global | 94.16 | 99.20 | 85.74 | 78.91 | **88.68** |
| 4_per_split | 94.95 | 99.44 | 91.63 | 80.56 | **90.85** |
| 23_per_generator | 94.93 | 99.66 | 93.57 | 84.12 | **92.42** |

## A3 — Threshold sensitivity (per-generator GB)

| t | ID | CM | CF | CD | Overall |
|---|---:|---:|---:|---:|---:|
| 0.30 | 94.90 | 99.62 | 92.68 | 82.71 | **91.77** |
| 0.35 | 94.87 | 99.63 | 93.13 | 83.60 | **92.13** |
| 0.40 | 94.92 | 99.64 | 93.32 | 84.12 | **92.35** |
| 0.45 | 95.02 | 99.67 | 93.46 | 84.26 | **92.46** |
| 0.50 | 94.93 | 99.66 | 93.57 | 84.12 | **92.42** |
| 0.55 | 94.92 | 99.68 | 93.57 | 84.10 | **92.42** |
| 0.60 | 94.86 | 99.69 | 93.52 | 83.72 | **92.28** |
| 0.65 | 94.43 | 99.67 | 93.21 | 82.18 | **91.64** |
| 0.70 | 94.31 | 99.66 | 92.91 | 79.96 | **90.88** |

## A4 — CV-fold sensitivity

| K | ID | CM | CF | CD | Overall |
|---|---:|---:|---:|---:|---:|
| 3 | 95.06 | 99.70 | 93.42 | 84.25 | **92.46** |
| 5 | 94.93 | 99.66 | 93.57 | 84.12 | **92.42** |
| 10 | 95.00 | 99.69 | 93.50 | 84.22 | **92.45** |

## A5 — Meta-learner choice

| Meta-learner | ID | CM | CF | CD | Overall |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 94.91 | 99.64 | 91.59 | 84.21 | **91.95** |
| RandomForest | 94.97 | 99.69 | 93.48 | 84.16 | **92.42** |
| GradientBoosting | 94.93 | 99.66 | 93.57 | 84.12 | **92.42** |

## B2 — Leave-one-generator-out

Train MoE on the other 22 generators, evaluate on the held-out one.
Compares against the in-distribution per-generator GB OOF accuracy on the same images.

- **Mean held-out accuracy:** 89.54%
- **Mean in-distribution accuracy:** 92.42%

| Generator | n | LO-gen-out acc | In-dist acc | Δ |
|---|---:|---:|---:|---:|
| cd_FFIW | 6832 | 78.95 | 79.33 | -0.38 |
| cd_deepfacelab | 3094 | 57.47 | 79.70 | -22.24 |
| cd_dreamina | 952 | 97.69 | 98.21 | -0.53 |
| cd_gpt4o | 630 | 84.44 | 86.67 | -2.22 |
| cd_hailuo | 1000 | 89.50 | 90.30 | -0.80 |
| cd_infiniteyou | 2960 | 89.83 | 92.60 | -2.77 |
| cf_codeformer | 1750 | 99.83 | 99.94 | -0.11 |
| cf_faceadapter | 294 | 80.61 | 93.20 | -12.59 |
| cf_iclight | 2082 | 56.15 | 84.77 | -28.63 |
| cf_infiniteyou | 3244 | 93.74 | 97.44 | -3.70 |
| cf_pulid | 3360 | 97.71 | 98.18 | -0.48 |
| cf_starganv2 | 2000 | 66.30 | 83.15 | -16.85 |
| cm_AdobeFirefly | 600 | 93.67 | 97.83 | -4.17 |
| cm_Flux11Pro | 600 | 99.67 | 99.50 | +0.17 |
| cm_Infinity | 4200 | 99.95 | 99.95 | +0.00 |
| cm_MAGI | 1048 | 99.90 | 99.90 | +0.00 |
| cm_StarryAI | 600 | 92.67 | 97.33 | -4.67 |
| cm_hart | 4201 | 99.93 | 99.93 | +0.00 |
| id_Hallo2 | 1660 | 99.76 | 99.64 | +0.12 |
| id_Midjourney | 600 | 99.83 | 100.00 | -0.17 |
| id_StyleGAN | 600 | 100.00 | 100.00 | +0.00 |
| id_facevid2vid | 1000 | 99.90 | 99.90 | +0.00 |
| id_ff | 8959 | 81.95 | 92.82 | -10.87 |