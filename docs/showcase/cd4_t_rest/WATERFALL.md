# WATERFALL (cells)

| step | cells | removed |
|---|---|---|
| in | 7,679,247 | |
| QC: After MT filter | 7,675,720 | 3,527 |
| QC: After min genes | 7,675,720 | 0 |
| QC: After max genes | 7,675,720 | 0 |
| QC: After min counts | 7,674,841 | 879 |
| QC: After max counts | 7,673,217 | 1,624 |
| escaper filter | 7,103,935 | 569,282 |
| undersized perturbations dropped (1025 perturbations) | 7,085,197 | 18,738 |
| survivors to Phase 7 | 7,085,197 | |

cells in = survivors + removed: 7,679,247 = 7,085,197 + 6,030 (QC) + 569,282 (escapers) + 18,738 (undersized) -> RECONCILES

metacell units: 3,551,703 (2,594,453 in train_hybrid)
perturbation groups using the fallback embedding: 0
