# Known issues (SPORE-XL v4.0)

Last reviewed October 2026. Items 1 to 6 are design limits you should know before interpreting results; 7 to 12 are issues of the v3.0 code
that v4.0 inherits or works around; 13 onwards are housekeeping.

## 1. The escaper rule is blind for low-expressed targets (design limit)

Rule 1 (phase 4, as in v3.0) calls a cell an escaper when its target gene is above the 10th percentile of that gene in the control cells. If the
control 10th percentile is 0, which is the case for every target expressed below roughly one count per cell, the rule keeps every cell with a zero
count, and an escaper with a zero count looks exactly like a knocked-down cell. On the stress dataset `02_lowexpr` (designed escapers 20%): escapers of
high-expressed targets were removed at 86%, of mid-expressed targets at 39%, of low-expressed targets at 0%. The same rule also removes about 27% of
cells that are truly knocked down (their count exceeds the control 10th percentile by chance). The optional response score
(`phase5_escaper_filtering.escaper_score`) looks at the transcriptional response instead of the target; `report` mode prints how many cells each rule
flags so the two can be compared on your data before you choose. On the synthetic 60k the score rule flags far more cells than rule 1 (weak designed
responses); it is experimental and not the default.

## 2. Perturbation x batch confounding is reported, not corrected

If a perturbation's cells sit in one batch, a model can learn the batch. v4.0 reports it (`reports/REPORT.md`, `stats/reduce2/all/confounding.tsv`)
with a warning when the batch mix of a perturbation differs from the controls beyond random sampling (total variation distance >= 0.5 AND a Monte Carlo p-value <= 0.001 against random draws of the same number of control cells; before build r3 the fixed 0.5 alone flagged most perturbations of a screen with 150 to 200 cells over 267 batches). Nothing is removed or corrected. With very few control cells the reference mix is itself noisy and the check loses power (the summary lists the reference cell count).

## 3. Gene symbols shared by two features resolve to the LAST copy (inherited)

When two features have the same symbol after ID translation (for example a PAR_Y duplicate), SPORE names the later copy `SYMBOL_1`, and a
perturbation label `SYMBOL` is matched to the later copy, because `src/phase05_escaper_filtering.py` builds its alias map from `var_names` and then
from every `var` column, and later entries overwrite earlier ones. On the stress dataset `09_dup_symbols` all 10 duplicated targets used the `_1` copy
(for 5 of 10 the designed knockdown was on the first copy). Nothing is logged by the core. `spore_xl.py convert` reports duplicated symbols in
`intake_report.md`; decide per gene before running. v4.0 does not change this behaviour (SPORE v3.0 equivalence).

## 4. Labels are taken as given by the core

The core treats `perturbation_col` literally. Padded, lower-case, Excel-damaged (`1-Sep`) and differently spelled control labels (`NT`, `Non-Targeting`)
become separate "perturbations", and the control variants become a large group with a split. Repair them in intake (`spore_xl.py convert` does:
whitespace, case, Excel dates, control spellings, with refusals for anything ambiguous). Stress dataset `08_messy_labels` shows the effect.

## 5. Multi-target perturbations and CRISPRa are refused

Labels containing `+`, `perturbation_structure: combinatorial` and an activation escaper direction stop the run in `plan` before anything is written.

## 6. Normalized input is refused

`plan` samples 2,000 rows and refuses non-integer or negative values (log1p, CPM, scaled data) before it writes a shard. Point `xl.source` at the raw
counts (`spore_xl.py convert` with `x_key: layers/counts`).

## 7. Phase 5 alias map uses every var column (inherited, not patched in the core)

`src/phase05_escaper_filtering.py` lines 84 to 87 map the string of every `adata.var` column value to a gene index, including the QC columns added in
phase 3 (`mt`, `n_cells_by_counts`, `mean_counts`, `pct_dropout_by_counts`, `total_counts`). A perturbation label equal to such a string (for example `True`
or `12`) would be tested against the wrong gene. XL builds the alias map from the harmonized source `var` only and refuses to run when a label collides
with a QC value.

## 8. v3.0 sums `total_counts` and `mean_counts` in float32 (inherited)

For genes with more than 2^24 total counts (RPE1: 12 genes, GAPDH 38,874,522 vs 38,874,844 in v3.0) the v3.0 column is rounded; XL is exact. No decision
depends on these columns. The compare tool holds them to a per-gene relative 1e-4.

## 9. SPORE v3.0 above 1,000,000 cells uses a chunked float64 path (inherited)

Phase 6 and 7 statistics switch to chunked float64 above 1M cells, so exact equivalence between v3.0 and v4.0 is claimed only below 1M cells
(K562: panel 7,087 for XL vs 7,090 for v3.0 under policy `all`).

## 10. v3.0 writes next to a backed raw input (inherited; the XL path does not)

Phase 0 of the SPORE v3.0 runner (`spore_local.py`, tag `v3.0`) treats a backed sparse `X` as dense and writes `<raw>_sparse.h5ad` next to the raw file. XL never writes next to its input.

## 11. The v3.0 runner is untouched

The SPORE v3.0 runners (`spore_local.py` and `spore_cluster.py`, tag `v3.0`) still carry the v3.0 issues listed in the v3.0 file (large-dataset worker path untested at scale, ghost excision
skipped on the in-process path, thesis-era names in log lines). They are kept for the compatibility tests and for datasets small enough for one process.

## 12. Hash seed

Compat mode (`xl.compat: v3`) reproduces v3.0's set-order tie-breaks and therefore needs `PYTHONHASHSEED=0`. Production mode does not.

## 13. Phase 10 (PCA, Harmony, cell-cycle regression) is not in v4.0

None of its outputs fed the split, the gene panel or the metacells. If a downstream tool needs the `_p10` files, phase 10 can return as a module; the v3.0 code is `src/phase10_*.py` on the tag `v3.0` and the branch `v3-archive`.

## 14. Notes

* The SPORE v3.0 phase modules that SPORE-XL does not use (ambient RNA, doublets, phase 10, cell-line separation) and its diagnostics, reporting and clean-up helpers are not shipped; they are on the tag `v3.0` and the branch `v3-archive`.
* `src/intake/R/export_se_tables_v2.R` has been tested on a mock R object, not yet on a real Bioconductor `SummarizedExperiment`.
* `finalize` normalizes the training blocks one after another (about 15 of its 20 minutes on the 7.7-million-cell CD4+ T screen); parallelizing it is planned.
* Merged split files are skipped above `xl.merge_cap_gb` (the CD4+ T screen would have needed 167 GB). The per-block parts under `processed/` are then the output; the user guide shows how to load them.
* The report key `target_nodes_in_panel` counts the training perturbations whose target is a panel gene; the total number of node perturbations (held-out plus training) is in `manifest/perturbation_nodes.tsv`. The key will be renamed in a later release.
* The metacells differ by about 0.1% between operating systems, because the k-means step rounds differently on different processors; every cleaning and splitting result is identical.
