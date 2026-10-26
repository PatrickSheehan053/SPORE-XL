# WATERFALL (cells)

| step | cells | removed |
|---|---|---|
| in | 247,914 | |
| QC: After MT filter | 247,914 | 0 |
| QC: After min genes | 247,914 | 0 |
| QC: After max genes | 247,914 | 0 |
| QC: After min counts | 247,914 | 0 |
| QC: After max counts | 247,827 | 87 |
| escaper filter | 230,181 | 17,646 |
| undersized perturbations dropped (879 perturbations) | 205,015 | 25,166 |
| survivors to Phase 7 | 205,015 | |

cells in = survivors + removed: 247,914 = 205,015 + 87 (QC) + 17,646 (escapers) + 25,166 (undersized) -> RECONCILES

metacell units: 103,681 (74,814 in train_hybrid)
perturbation groups using the fallback embedding: 0
