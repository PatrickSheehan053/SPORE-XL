# H1 hESC: interactive session against SLURM job

The same configuration ran twice on the same cluster, once in an interactive session and once as a SLURM batch job. Both runs passed all 15 verification checks, and every result below is identical.

| result | interactive | SLURM | match |
|---|---|---|---|
| surviving cells | 185,117 | 185,117 | identical cell for cell |
| split of every cell | | | identical |
| control assignment | 33,321 controls | 33,321 controls | identical |
| split file (train / val / test perturbations) | 88 / 20 / 40 | 88 / 20 / 40 | identical |
| knockdown (mean-shift) scores | | | identical |
| escaper thresholds | | | identical |
| QC variables | | | identical |
| genes kept by gene QC | 14,145 | 14,145 | identical, 0 genes differ |
| gene panel | 5,000 genes | 5,000 genes | identical, 0 genes differ |
| raw counts in the training parts | | | identical |

A file-by-file comparison of the two run folders found 567 identical files and 27 that differ only in the run's own folder path. The remaining 77 are the run logs, the timing and memory tables, the record of the run label, and `.h5ad` files whose stored data is identical but whose HDF5 container bytes differ between writes.
