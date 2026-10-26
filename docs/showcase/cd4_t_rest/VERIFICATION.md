# VERIFICATION PASS

| check | result | detail |
|---|---|---|
| every unit has a .done marker and its files match their checksums | PASS | 586 units |
| survivors in global_obs_final equal reduce2's count | PASS | 7,085,197 |
| every perturbation is in exactly one block | PASS | 11700 perturbations; split across blocks: [] |
| (a) label sets of the train/val/test parts equal the split JSON | PASS | {'train': (8252, 8252), 'val': (1148, 1148), 'test': (2300, 2300)} |
| (b) train/val/test label sets are pairwise disjoint and cover the surviving set | PASS | 11700 labels vs 11700 surviving |
| (c) for every perturbation, cells after the escaper filter equal cells in its split | PASS | mismatches: [] |
| (d) every control cell has exactly one split | PASS | 329,288 controls assigned |
| (e) no xl_global_index appears twice | PASS | 7,085,197 cells in the parts, 7,085,197 survivors |
| (f) train_hybrid contains no val or test units | PASS | split values ['control', 'train'] |
| (g) every metacell is built from one perturbation, hence one split; controls are single-cell | PASS | 11701 groups |
| var names identical across all parts | PASS | 1 distinct var lists over hvgraw/metacell/hybrid parts |
| p9 parts hold exactly the cells of the hvgraw parts | PASS |  |
| train_hybrid_lognorm has the units of train_hybrid, finite, rows sum to target_sum after expm1 | PASS | 2594453 units |
| v3.0-name aliases (_p9) exist and match their _norm targets | PASS | 393 aliases; bad: [] |
| GRAPH-READY in COVERAGE_REPORT | PASS |  |
