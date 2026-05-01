# B1 — Open-world routing (no generator manifest)

Replace the oracle gate `g(x)` with an unsupervised KMeans router
over the 2-d feature `(p_v, b_p)`. One GB expert per cluster.

## Plain KMeans

| Variant | ID | CM | CF | CD | Overall |
|---|---:|---:|---:|---:|---:|
| KMeans K=1 | 94.16 | 99.20 | 85.74 | 78.91 | **88.68** |
| KMeans K=4 | 94.06 | 99.14 | 85.73 | 78.90 | **88.64** |
| KMeans K=8 | 93.81 | 99.13 | 85.73 | 78.83 | **88.56** |
| KMeans K=16 | 93.46 | 99.06 | 85.52 | 78.90 | **88.42** |
| KMeans K=23 | 93.28 | 99.01 | 85.25 | 78.94 | **88.31** |

## Split-aware KMeans (cluster within each ID/CM/CF/CD split)

| Variant | ID | CM | CF | CD | Overall |
|---|---:|---:|---:|---:|---:|
| split-aware K=1 (total 4) | 94.95 | 99.44 | 91.63 | 80.56 | **90.85** |
| split-aware K=2 (total 8) | 94.83 | 99.41 | 91.41 | 80.66 | **90.79** |
| split-aware K=4 (total 16) | 94.59 | 99.39 | 91.36 | 80.39 | **90.63** |
| split-aware K=6 (total 24) | 94.43 | 99.26 | 91.35 | 80.54 | **90.61** |

## Reference (oracle, generator manifest known)

| Variant | ID | CM | CF | CD | Overall |
|---|---:|---:|---:|---:|---:|
| oracle 23 gen | 94.93 | 99.66 | 93.57 | 84.12 | **92.42** |