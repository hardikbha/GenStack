# C1-E — Predicted-gate with CLIP features (deployable)

Same protocol as C1 but the gate classifier consumes frozen CLIP ViT-L/14
CLS embeddings (768-d, l2-normalised, multinomial LR). Experts are unchanged
(GradientBoosting on (p_v, b_p)). Outer 5-fold OOF.

## Routing accuracy

| Variant | ID | CM | CF | CD | Overall |
|---|---:|---:|---:|---:|---:|
| Oracle 23-gen (manifest) | 94.98 | 99.70 | 93.49 | 84.38 | **92.50** |
| Pred 23-gen gate (hard) | 94.70 | 98.50 | 94.66 | 84.36 | **92.45** |
| Pred 23-gen gate (soft) | 94.91 | 99.43 | 94.96 | 84.57 | **92.83** |
| Pred 4-split gate (hard) | 94.58 | 98.06 | 92.26 | 80.97 | **90.74** |
| Pred 4-split gate (soft) | 94.85 | 99.03 | 92.62 | 81.14 | **91.15** |

## Gate diagnostics

- 23-way gate top-1 acc: **71.01%** (chance 4.35%)
- 23-way gate top-3 acc: **81.29%**
-  4-way gate top-1 acc: **82.24%** (chance 25.00%)