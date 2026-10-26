# WATERFALL (cells)

| step | cells | removed |
|---|---|---|
| in | 1,989,578 | |
| QC: After MT filter | 1,977,979 | 11,599 |
| QC: After min genes | 1,977,979 | 0 |
| QC: After max genes | 1,977,979 | 0 |
| QC: After min counts | 1,977,979 | 0 |
| QC: After max counts | 1,977,979 | 0 |
| escaper filter | 1,808,039 | 169,940 |
| undersized perturbations dropped (748 perturbations) | 1,783,992 | 24,047 |
| survivors to Phase 7 | 1,783,992 | |

cells in = survivors + removed: 1,989,578 = 1,783,992 + 11,599 (QC) + 169,940 (escapers) + 24,047 (undersized) -> RECONCILES

metacell units: 893,620 (680,190 in train_hybrid)
perturbation groups using the fallback embedding: 0
