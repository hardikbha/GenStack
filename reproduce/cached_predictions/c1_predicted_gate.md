# C1 — Predicted-gate (deployable open-world routing)

Defends the oracle MoE: train a learned gate classifier on `(p_v, b_p)`
and route to the matching expert. Outer 5-fold OOF over binary label.

## Routing accuracy

| Variant | ID | CM | CF | CD | Overall |
|---|---:|---:|---:|---:|---:|
| Oracle 23-gen (manifest) | 94.91 | 99.72 | 93.52 | 84.24 | **92.45** |
| Pred 23-gen gate (hard) | 93.88 | 98.55 | 80.55 | 79.18 | **87.29** |
| Pred 23-gen gate (soft) | 94.34 | 99.16 | 85.57 | 78.84 | **88.65** |
| Pred 4-split gate (hard) | 94.48 | 98.93 | 82.25 | 80.30 | **88.26** |
| Pred 4-split gate (soft) | 94.17 | 99.19 | 85.82 | 78.93 | **88.71** |

## Gate diagnostics

- 23-way gate top-1 acc: **21.91%** (chance 4.35%)
- 23-way gate top-3 acc: **52.05%**
-  4-way gate top-1 acc: **44.18%** (chance 25.00%)