"""XL clone of the decision part of SPORE src/phase06_gene_triage.py::filter_genes (SPORE v3.0, GitHub 5015dc6).

Reproduces lines 82-155 (ambient filter, scale-adaptive filter, target rescue, cell-cycle ghost rescue) and line 163
(penetrance) from merged per-gene statistics (detection count = getnnz, sum, number of cells) instead of the matrix.
The statistics are taken over the cells SPORE's Phase 6 sees: survivors of the escaper filter AND the
undersized-perturbation drop (phase05:239-264), accumulated per block in `gather`.

mean_expr follows the branch SPORE would take: in-memory (n_cells <= 1,000,000): float32 column sums divided by n_cells
(line 77); large path: float64 sums divided, then cast to float32 (line 71). For integer-valued counts below 2**24 per
gene the float32 sums are exact, so the exact float64 sum cast to float32 is identical.
"""
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.phases.phase06_gene_triage import _get_perturbation_targets
from src.phases.utils import get_cell_cycle_genes


MEAN_UMI_THRESHOLD_DEFAULT = 0.25  # v4 (P8): the one default. SPORE v3.0 phase06:103 said 0.05, config_v3._DEFAULTS said 0.25 and won


def keep_mask_from_stats(gene_sums, var_names, surviving_labels, cfg, logger=None, ghost_rescue=True):
    """ghost_rescue=False (v4 production, P3): cell-cycle genes that Phase 6 would remove stay removed."""
    p6 = cfg.get("phase6_gene_triage", {})
    waterfall = OrderedDict()
    n_genes = len(var_names)
    waterfall["Starting genes"] = n_genes

    n_cells = gene_sums.n_cells
    cells_per_gene = gene_sums.nnz
    if n_cells > 1000000:
        mean_expr = (gene_sums.total / n_cells).astype(np.float32)
    else:
        mean_expr = gene_sums.total.astype(np.float32) / n_cells

    keep_genes = np.ones(n_genes, dtype=bool)
    min_cells = p6.get("min_cells_expressing", 10)
    ambient_fail = cells_per_gene < min_cells
    keep_genes &= ~ambient_fail
    waterfall["After ambient filter"] = int(keep_genes.sum())

    pct_thresh = p6.get("pct_filter", 0.01)
    min_cells_adaptive = max(min_cells, int(n_cells * pct_thresh))
    umi_thresh = p6.get("mean_umi_threshold", MEAN_UMI_THRESHOLD_DEFAULT)
    adaptive_fail = (cells_per_gene < min_cells_adaptive) & (mean_expr < umi_thresh)

    would_remove = ~keep_genes | adaptive_fail
    waterfall["After adaptive (pre-rescue)"] = int((~would_remove).sum())

    rescued = []
    if p6.get("rescue_perturbation_targets", True):
        stub = SimpleNamespace(obs=pd.DataFrame({cfg["dataset"]["perturbation_col"]: list(surviving_labels)}))
        targets = _get_perturbation_targets(stub, cfg)
        var_set = set(var_names)
        targets_in_features = targets & var_set
        gene_name_to_idx = {g: i for i, g in enumerate(var_names)}
        targets_to_rescue = {t for t in targets_in_features if would_remove[gene_name_to_idx[t]]}
        if targets_to_rescue:
            rescued = sorted(targets_to_rescue)
            for t in targets_to_rescue:
                would_remove[gene_name_to_idx[t]] = False

    cc_s, cc_g2m = get_cell_cycle_genes()
    cc_genes = set(cc_s + cc_g2m) if ghost_rescue else set()
    var_upper_map = {v.upper(): i for i, v in enumerate(var_names)}
    ghosts_rescued = []
    for g in cc_genes:
        if g in var_upper_map:
            idx = var_upper_map[g]
            if would_remove[idx]:
                would_remove[idx] = False
                ghosts_rescued.append(var_names[idx])

    final_keep = ~would_remove
    waterfall["After adaptive (post-rescue)"] = int(final_keep.sum())
    penetrance = np.float32((cells_per_gene / n_cells) * 100)
    return final_keep, waterfall, rescued, sorted(ghosts_rescued), penetrance
