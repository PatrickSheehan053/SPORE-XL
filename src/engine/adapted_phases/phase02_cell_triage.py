"""XL clone of SPORE src/phase02_cell_triage.py (SPORE v3.0, GitHub 5015dc6), split into a per-cell mask and a
gene mask that uses a FIXED ribosomal-exemption set computed once by `plan` (blueprint Amendment 1 item 1).

Reproduces: phase02_cell_triage.py lines 49-98 (cell thresholds, verbatim) and lines 101-127 (ribosomal mask,
with `exempt` passed in instead of rebuilt from the labels of the cells at hand).
QC metrics themselves come from SPORE's own `_ensure_qc_metrics` (imported in stage_map1).
"""
from collections import OrderedDict

import numpy as np


def qc_cell_mask(adata, cfg, logger):
    p2 = cfg.get("phase2_cell_triage", {})
    overrides = cfg.get("_detection_overrides", {})
    waterfall = OrderedDict()
    waterfall["Starting cells"] = adata.n_obs

    keep_cells = np.ones(adata.n_obs, dtype=bool)

    disable_mt = overrides.get("disable_mt_filter", False)
    if not disable_mt:
        mt_thresh = p2.get("mt_threshold", 0.20)
        keep_cells &= (adata.obs["pct_counts_mt"].values <= mt_thresh * 100)
    waterfall["After MT filter"] = int(keep_cells.sum())

    keep_cells &= (adata.obs["n_genes_by_counts"].values >= p2["min_genes_per_cell"])
    waterfall["After min genes"] = int(keep_cells.sum())

    keep_cells &= (adata.obs["n_genes_by_counts"].values <= p2["max_genes_per_cell"])
    waterfall["After max genes"] = int(keep_cells.sum())

    keep_cells &= (adata.obs["total_counts"].values >= p2["min_counts_per_cell"])
    waterfall["After min counts"] = int(keep_cells.sum())

    keep_cells &= (adata.obs["total_counts"].values <= p2["max_counts_per_cell"])
    waterfall["After max counts"] = int(keep_cells.sum())
    return keep_cells, waterfall


def ribo_gene_mask(var_names, cfg, exempt):
    p2 = cfg.get("phase2_cell_triage", {})
    raw_prefixes = p2.get("ribo_prefixes", ["RPL", "RPS"])
    upper_prefixes = tuple(prefix.upper() for prefix in raw_prefixes)
    is_ribo = var_names.str.upper().str.startswith(upper_prefixes)
    is_exempt = var_names.isin(exempt)
    return np.asarray((~is_ribo) | is_exempt)
