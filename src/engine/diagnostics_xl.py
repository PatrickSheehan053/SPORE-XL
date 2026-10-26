"""P5 diagnostics (Amendment 2): no data is changed, the numbers go into the report.

 confounding_table   perturbation x batch table from metadata, with a warning when a perturbation's batch mix differs from the
                     controls' (total variation distance) beyond a threshold
 variance_explained  share of the (log-normalized) variance on the HVG subsample that a factor (batch, donor, perturbation) explains,
                     raw and adjusted for the number of groups
 cell_cycle_diagnostic  S / G2M scores on the subsample: score variance, phase fractions, and how much of it is perturbation / batch
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

BATCH_CANDIDATES = ("gem_group", "batch", "lane", "sample", "sample_id", "library", "library_id", "run", "channel")
DONOR_CANDIDATES = ("donor", "donor_id", "patient", "patient_id", "individual", "subject")


def diagnostic_columns(obs_columns, cfg):
    """(batch column or None, donor column or None): explicit diagnostics.batch_col / donor_col first, then the usual names."""
    settings = cfg.get("diagnostics", {}) or {}
    columns = set(obs_columns)

    def pick(explicit, candidates):
        if explicit:
            if explicit not in columns:
                raise ValueError(f"diagnostics column '{explicit}' is not in the source obs (columns: {sorted(columns)[:15]})")
            return explicit
        return next((name for name in candidates if name in columns), None)

    return pick(settings.get("batch_col"), BATCH_CANDIDATES), pick(settings.get("donor_col"), DONOR_CANDIDATES)


NULL_DRAWS = 2000      # Monte Carlo draws per distinct cell count (resolution of the p-value: 1 / 2001)
NULL_SEED = 20261022   # fixed: the table is reproducible


def sampling_null_tvd(n_cells, reference, draws=NULL_DRAWS, seed=NULL_SEED):
    """TVD between a batch mix of n_cells cells drawn at random from `reference` (the controls' batch shares) and `reference`: the distance
    that sampling noise ALONE produces. Returns the sorted draws. With ~200 cells over 267 batches most batches get 0 or 1 cell, so this
    is already about 0.5 and a fixed threshold of 0.5 flags almost every perturbation (K562, 22 Oct)."""
    rng = np.random.default_rng([seed, int(n_cells)])
    sampled = rng.multinomial(int(n_cells), reference / reference.sum(), size=draws) / float(n_cells)
    return np.sort(0.5 * np.abs(sampled - reference[None, :]).sum(axis=1))


def confounding_table(labels, batches, control_label, tvd_warn=0.5, min_cells=1, alpha=0.001):
    """labels, batches: one entry per surviving cell. Returns (table per perturbation, summary dict).
    tvd = total variation distance between the perturbation's batch distribution and the controls' (0 = same mix, 1 = disjoint).
    CALIBRATED: a perturbation is flagged only if tvd >= tvd_warn AND its tvd is beyond what random sampling of the same number of cells from the
    controls' batch mix would give (Monte Carlo p-value <= alpha). Columns tvd_null_mean / tvd_null_p99 / p_value / flagged_uncalibrated show the
    comparison; `flagged_uncalibrated` is the old fixed-threshold rule."""
    frame = pd.DataFrame({"label": np.asarray(labels).astype(str), "batch": np.asarray(batches).astype(str)})
    controls = frame[frame["label"] == control_label]
    reference = (controls if len(controls) else frame)["batch"].value_counts(normalize=True)
    crosstab = pd.crosstab(frame.loc[frame["label"] != control_label, "label"], frame.loc[frame["label"] != control_label, "batch"])
    all_batches = sorted(set(frame["batch"]))
    crosstab = crosstab.reindex(columns=all_batches, fill_value=0)
    reference = reference.reindex(all_batches, fill_value=0.0).to_numpy()
    counts = crosstab.to_numpy(dtype=np.float64)
    totals = counts.sum(axis=1)
    shares = counts / np.maximum(totals[:, None], 1)
    tvd = 0.5 * np.abs(shares - reference[None, :]).sum(axis=1)
    null_mean, null_p99, p_value = np.full(len(totals), np.nan), np.full(len(totals), np.nan), np.full(len(totals), np.nan)
    if len(all_batches) > 1:
        for n_cells in np.unique(totals[totals >= 1]):
            draws = sampling_null_tvd(n_cells, reference)
            rows = totals == n_cells
            null_mean[rows], null_p99[rows] = draws.mean(), draws[int(0.99 * (len(draws) - 1))]
            exceed = len(draws) - np.searchsorted(draws, tvd[rows] - 1e-12, side="left")   # draws >= observed
            p_value[rows] = (1.0 + exceed) / (len(draws) + 1.0)
    table = pd.DataFrame({"label": crosstab.index.astype(str), "n_cells": totals.astype(int), "n_batches_present": (counts > 0).sum(axis=1),
                          "dominant_batch": np.array(all_batches)[counts.argmax(axis=1)] if len(counts) else [],
                          "max_batch_share": shares.max(axis=1) if len(counts) else [], "tvd_vs_controls": tvd,
                          "tvd_null_mean": null_mean, "tvd_null_p99": null_p99, "p_value": p_value})
    eligible = (table["n_cells"] >= min_cells) & (len(all_batches) > 1)
    table["flagged_uncalibrated"] = (table["tvd_vs_controls"] >= tvd_warn) & eligible
    table["flagged"] = table["flagged_uncalibrated"] & (table["p_value"] <= alpha)
    big = counts[totals >= 20]
    cramers = float("nan")
    if len(all_batches) > 1 and len(big) > 1:
        expected = big.sum(axis=1)[:, None] * big.sum(axis=0)[None, :] / big.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = float(np.nansum((big - expected) ** 2 / np.where(expected > 0, expected, np.nan)))
        cramers = float(np.sqrt(chi2 / (big.sum() * (min(big.shape) - 1))))
    flagged = table["flagged"]
    summary = {"n_batches": len(all_batches), "n_perturbations": int(len(table)), "n_flagged": int(flagged.sum()),
               "n_flagged_uncalibrated": int(table["flagged_uncalibrated"].sum()), "alpha": float(alpha), "null_draws": NULL_DRAWS,
               "cells_in_flagged": int(table.loc[flagged, "n_cells"].sum()), "cells_total": int(totals.sum()),
               "tvd_warn": float(tvd_warn), "cramers_v_perturbation_x_batch": cramers,
               "n_fully_in_one_batch": int((table["max_batch_share"] >= 0.999).sum()) if len(table) else 0,
               "n_reference_cells": int(len(controls) if len(controls) else len(frame)),
               "reference": "controls" if len(controls) else "all cells"}
    return table.sort_values(["flagged", "p_value", "tvd_vs_controls"], ascending=[False, True, False]).reset_index(drop=True), summary


def variance_explained(matrix, groups, gene_chunk=2000):
    """Fraction of total variance (summed over genes) explained by a grouping of the cells, one-way: SS_between / SS_total.
    Returns (r2, adjusted_r2, n_groups). matrix: cells x genes (dense or sparse, log-normalized)."""
    groups = pd.Categorical(np.asarray(groups).astype(str))
    n_cells, n_genes = matrix.shape
    codes = groups.codes
    n_groups = len(groups.categories)
    if n_groups < 2 or n_groups >= n_cells:
        return float("nan"), float("nan"), n_groups
    indicator = sp.csr_matrix((np.ones(n_cells), (codes, np.arange(n_cells))), shape=(n_groups, n_cells))
    sizes = np.asarray(indicator.sum(axis=1)).ravel()
    matrix = sp.csr_matrix(matrix) if sp.issparse(matrix) else matrix
    ss_between = ss_total = 0.0
    for start in range(0, n_genes, gene_chunk):
        block = matrix[:, start:start + gene_chunk]
        block = block.toarray() if sp.issparse(block) else np.asarray(block)
        block = block.astype(np.float64, copy=False)
        grand = block.mean(axis=0)
        ss_total += float(((block - grand) ** 2).sum())
        group_means = (indicator @ block) / sizes[:, None]
        ss_between += float((sizes[:, None] * (group_means - grand) ** 2).sum())
    r2 = ss_between / ss_total if ss_total > 0 else float("nan")
    adjusted = 1 - (1 - r2) * (n_cells - 1) / (n_cells - n_groups) if np.isfinite(r2) else float("nan")
    return r2, adjusted, n_groups


def lognorm_copy(counts, target_sum=1e4):
    """normalize_total + log1p of a CSR count matrix (float32), without scanpy."""
    counts = sp.csr_matrix(counts, dtype=np.float32)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.where(totals > 0, target_sum / np.maximum(totals, 1e-12), 0.0).astype(np.float32)
    scaled = sp.diags(scale) @ counts
    scaled.data = np.log1p(scaled.data)
    return sp.csr_matrix(scaled)


def cell_cycle_diagnostic(lognorm, var_names, s_genes, g2m_genes, perturbation, batch=None, min_genes=5):
    """S and G2M scores (scanpy score_genes_cell_cycle on the log-normalized subsample, fixed seed). Returns a dict (all values plain)."""
    import anndata as ad
    import scanpy as sc
    names = pd.Index(var_names)
    upper = {name.upper(): name for name in names}
    s_present = [upper[g] for g in s_genes if g in upper]
    g2m_present = [upper[g] for g in g2m_genes if g in upper]
    result = {"n_s_genes": len(s_present), "n_g2m_genes": len(g2m_present)}
    if len(s_present) < min_genes or len(g2m_present) < min_genes:
        result["status"] = f"not computed: fewer than {min_genes} S or G2M genes in the gene set"
        return result
    adata = ad.AnnData(X=sp.csr_matrix(lognorm), var=pd.DataFrame(index=names.astype(str)))
    try:
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_present, g2m_genes=g2m_present, random_state=0, copy=False)
    except Exception as error:  # noqa: BLE001
        result["status"] = f"not computed: {type(error).__name__}: {error}"
        return result
    s_score, g2m_score = adata.obs["S_score"].to_numpy(), adata.obs["G2M_score"].to_numpy()
    phases = adata.obs["phase"].astype(str).value_counts(normalize=True).to_dict()
    result.update({"status": "ok", "s_score_variance": float(np.var(s_score)), "g2m_score_variance": float(np.var(g2m_score)),
                   "phase_fractions": {key: round(float(value), 4) for key, value in phases.items()}})
    for name, values in (("s_score", s_score), ("g2m_score", g2m_score)):
        column = values.reshape(-1, 1)
        r2, adjusted, _ = variance_explained(column, perturbation)
        result[f"{name}_r2_by_perturbation_adjusted"] = None if not np.isfinite(adjusted) else float(adjusted)
        if batch is not None:
            r2, adjusted, _ = variance_explained(column, batch)
            result[f"{name}_r2_by_batch_adjusted"] = None if not np.isfinite(adjusted) else float(adjusted)
    return result
