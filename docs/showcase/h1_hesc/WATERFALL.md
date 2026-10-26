# WATERFALL (cells)

| step | cells | removed |
|---|---|---|
| in | 221,273 | |
| QC: After MT filter | 221,273 | 0 |
| QC: After min genes | 221,273 | 0 |
| QC: After max genes | 212,036 | 9,237 |
| QC: After min counts | 212,036 | 0 |
| QC: After max counts | 194,348 | 17,688 |
| escaper filter | 185,173 | 9,175 |
| undersized perturbations dropped (2 perturbations) | 185,117 | 56 |
| survivors to Phase 7 | 185,117 | |

cells in = survivors + removed: 221,273 = 185,117 + 26,925 (QC) + 9,175 (escapers) + 56 (undersized) -> RECONCILES

metacell units: 106,119 (79,779 in train_hybrid)
perturbation groups using the fallback embedding: 0
