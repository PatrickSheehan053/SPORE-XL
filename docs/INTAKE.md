# Intake (phase 1): `spore_xl.py convert`

```
python -B spore_xl.py convert --recipe src/intake/recipes/<dataset>.convert.yaml --out <NEW folder> [--window 20000]
```

Reads one raw format in windows of rows, applies the recipe, checks that the matrix is raw counts, and writes
`<out>/<dataset>.h5ad` (sparse CSR, float32, `obs.gene` = perturbation label as a categorical with sorted categories, control label normalized,
`obs.batch` / `obs.donor` when the recipe names them), `provenance.json` and `intake_report.md`. Point `xl.source.h5ad` of the run config at the result. The output folder must not exist
(nothing is overwritten); the file is written as `.h5ad.partial` and renamed when complete, so an interrupted conversion never looks finished.

## Formats (`source.format`)

| format | path is | cells x genes read as | notes |
|---|---|---|---|
| `h5ad` | `.h5ad`, sparse CSR or dense | `obs` / `var` of the file | `x_key`: `X` (default), `layers/counts`, `raw/X`. A csc-stored X is refused (convert once with anndata) |
| `10x_h5` | CellRanger v3+ `.h5` | barcodes are the stored columns, so a cell is read without a transpose | only features of type Gene Expression become the matrix; CRISPR guide-capture features feed `obs.label_from_guides` |
| `10x_mtx` | folder with `matrix.mtx(.gz)`, `barcodes.tsv(.gz)`, `features.tsv(.gz)` or `genes.tsv(.gz)` | `source.orientation` (default `genes_by_cells`, the 10x convention) | read whole |
| `loom` | `.loom` | `/matrix` is genes x cells; `col_attrs` become obs, `row_attrs` var | |
| `csv`, `tsv` | table with a header row and an index column | `source.orientation` is REQUIRED (`genes_by_cells` or `cells_by_genes`) | read whole; refused above 2 GB |
| `dense_h5_tables` | HDF5 with one dense 2-D dataset (`source.h5_dataset`) | `source.cells_table` and `source.genes_table` (TSV) give the labels; the orientation is checked against their lengths | first case: DLD-1 GSE337988, tables exported from the `.rds` by `src/intake/R/export_se_tables_v2.R` |

## Recipe keys

```yaml
dataset: <name, also the output file name>
source: {format: ..., path: ..., x_key / orientation / h5_dataset / cells_table / genes_table / cell_index_column / gene_id_column: ...}
obs:
  label_column: <cell-table column with the perturbation label>      # or label_from_guides (below)
  keep: "<pandas query on the cell table, e.g. demux_type_v2 == 'singlet'>"
  drop_labels: [nan, '', None, <labels to drop>]
  batch_column: <column>        # written as obs.batch
  donor_column: <column>        # written as obs.donor
  extra_columns: [<columns copied to obs>]
  label_from_guides: {min_umi: 3, target_regex: '-\d+$', control_regex: '^(non-?targeting|NT|control)'}     # 10x_h5 with CRISPR features
control_labels: [<spellings of the control label in this dataset>]
control_label_out: non-targeting
label_fixes: {strip: true, excel_dates: true, case_fold_to_genes: true, map: {"1-Jun": JUN}}
subsample: {per_label_max: 100, seed: 42}      # cap per perturbation (controls are all kept), seeded
genes: {symbol_column: <column of the gene table>, duplicate_symbols: report | error}
perturbation_col: gene
```

## What it repairs and what it refuses

Repairs: surrounding whitespace; case (`stat1` -> `STAT1`, only when exactly one gene matches); Excel-damaged gene symbols (`1-Sep` -> `SEPT1`, `1-Mar` -> `MARCH1`, `1-Dec` -> `DEC1`, and `1-Jun` / `2-Jun` -> `JUN` for the guide names `JUN-1` / `JUN-2`) when exactly one gene in the matrix could have produced the label; control spellings listed in `control_labels` (case-insensitive) -> `control_label_out`; an explicit `label_fixes.map`.

Refuses (exit code 2, message says what to change, nothing is written): non-integer or negative values ("looks normalized: use the raw counts", checked on a 2,000-cell sample before writing and on every window while writing); a label that looks like a control but is not in `control_labels` (`scramble`, `ntc`, ...); an Excel-date label that two or more genes could have produced (the message lists them: resolve in `label_fixes.map`) or none; a multi-target label (`A+B`); no control cells left; a gene axis that does not match the gene table (the message names the DLD-1 cause); a bad column name; an existing output folder.

Reported, not changed: labels that are not a gene in the matrix (kept: their target is not measured), duplicate gene symbols (the core will resolve a label to the LAST copy, see `KNOWN_ISSUES.md` 3), guide-assignment counts.

## DLD-1 (GSE337988, Low MOI): the R-object case

The `*_se.rds` holds only metadata; the counts are a dense HDF5 (`/assay001`, 1,196,592 x 36,601, int32). In a first export the gene names had 23,480 rows: they were the guide library
(`A1CF_P1_1` ...), not the 36,601 genes. `src/intake/R/export_se_tables_v2.R <se.rds> <new out dir>` walks the object without Bioconductor, lists every character vector with its length (`candidates.tsv`),
ignores alternative experiments, and writes `genes.tsv` from the unique vector whose length equals the assay's gene axis; with several candidates it stops and prints them (rerun with `--gene-path`).
Run it with an R installation that can read the object (`Rscript`; the script has only been tested on a mock S4 object), copy `genes.tsv` and `cells.tsv.gz` back,
fill `genes_table:` in `src/intake/recipes/GSE337988_DLD1_Low.convert.yaml` and run `convert`.
