# RPE1: interactive session, SLURM job and laptop

## Interactive session against SLURM job

The same configuration ran on the same cluster in an interactive session and as a SLURM batch job. Both runs passed all 15 verification checks, and the comparison found every result identical.

| result | match |
|---|---|
| surviving cells (205,015) | identical cell for cell |
| split of every cell | identical |
| control assignment | identical |
| split file (1,058 / 152 / 304 perturbations) | identical |
| knockdown (mean-shift) scores | identical |
| escaper thresholds | identical |
| QC variables | identical |
| genes kept by gene QC (8,739) | identical, 0 genes differ |
| gene panel (5,000 genes) | identical, 0 genes differ |
| raw counts in the training parts | identical |

## Windows laptop against Linux cluster

The same configuration ran on a Windows laptop (6 cores) and as a SLURM job on a Linux cluster node. Both passed all 15 verification checks. Every cleaning and splitting result matches.

| result | laptop | cluster | match |
|---|---|---|---|
| surviving cells | 205,015 | 205,015 | identical |
| split (train / val / test perturbations) | 1,058 / 152 / 304 | 1,058 / 152 / 304 | identical split file |
| control cells | 11,483 | 11,483 | identical |
| genes kept by gene QC | 8,739 | 8,739 | identical gene list |
| gene panel | 5,000 genes | 5,000 genes | identical panel |
| escaper thresholds, knockdown table, perturbation nodes | | | identical |
| metacell units | 103,569 | 103,681 | differ by 0.1% |
| training metacell set (metacells plus single control cells) | 74,679 | 74,814 | differ by 0.2% |

Text files written on Windows end their lines differently, so they were compared by content. The metacell step groups the cells of each perturbation with k-means, and k-means arithmetic rounds slightly differently on different processors and math libraries. An interactive run and a SLURM run on the same cluster produced identical metacells, so the difference comes from the platform and not from how a run is launched. Everything upstream of the metacells, including which cells survive, the split and the gene panel, is identical across platforms.
