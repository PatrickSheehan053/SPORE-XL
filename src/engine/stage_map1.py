"""map1 (one task per shard): SPORE Phase 2 on one shard (adapted_phases/phase02), then the Phase 5 statistics.

Per shard it writes (in stats/map1/shard_NNNN/):
  bank.npz          IntHistogramBank: counts of each integer value of every target gene among QC-passing control cells
  rawsums.npz       GeneSums over ALL cells before QC (source of the five QC var columns, phase02:33 in SPORE)
  pertsums.parquet  label, gene_index, n_cells, sum of the target gene over all QC-passing cells of the perturbation
  labelcounts.parquet  label, n_qc_pass
  waterfall.json, meta.json
and the QC shard _xl_work/map1/shard_NNNN/qc.h5ad (QC-passing cells, ribosomal columns dropped).
"""
import json

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from xl_common import (Run, alias_map, require_done, shard_unit, task, write_new_json, write_new_parquet,
                       write_new_text)
import escaper_score
from stats import GeneSums, IntHistogramBank, Waterfall

from adapted_phases.phase02_cell_triage import qc_cell_mask, ribo_gene_mask
from src.phases.phase02_cell_triage import _ensure_qc_metrics
from src.phases.phase05_escaper_filtering import _parse_target_genes


def target_genes_by_label(run, labels, gene_to_idx):
    """label -> list of gene indices (constituents that exist in the matrix), as phase05:117-118."""
    mapping = {}
    for label in labels:
        if label == run.ctrl_label:
            continue
        constituents = _parse_target_genes(label, run.cfg)
        mapping[label] = [gene_to_idx[gene] for gene in constituents if gene in gene_to_idx]
    return mapping


def run_map1(run: Run, index: int):
    unit = shard_unit(index)
    with task(run, "stats", "map1", unit) as stats_unit:
        if stats_unit is None:
            return
        with task(run, "work", "map1", unit, log=False) as work_unit:
            if work_unit is None:
                raise RuntimeError("map1 work unit finished but stats unit is not; inconsistent state")
            _map1(run, index, stats_unit, work_unit)


def _map1(run, index, stats_unit, work_unit):
    cfg, logger = run.cfg, run.logger
    shards = run.shards()
    row = shards.iloc[index]
    shard = ad.read_h5ad(run.run_dir / row["file"])
    if "xl_global_index" not in shard.obs.columns:  # recipe (intake) shards: position in the manifest-order concatenation
        shard.obs["xl_global_index"] = int(row["row_start"]) + np.arange(shard.n_obs, dtype=np.int64)
        shard.obs["xl_shard"] = np.full(shard.n_obs, int(index), dtype=np.int64)
    source_info = run.source_info()
    categories = source_info.get("perturbation_categories")
    if categories is not None:  # blueprint Amendment 1 item 5: refuse a shard whose category order differs
        if [str(value) for value in shard.obs[run.pert_col].cat.categories] != categories:
            raise RuntimeError(f"{row['file']}: perturbation category order differs from manifest/source.json")

    var_table = run.var_table()
    source_names = (run.manifest_dir / "var_source_names.txt").read_text(encoding="utf-8").split("\n")[:-1]
    if [str(name) for name in shard.var_names] != source_names:
        raise RuntimeError(f"{row['file']}: var_names differ from the planned source var")
    shard.var = var_table.copy()  # harmonized names + columns, identical for every shard
    if not sp.issparse(shard.X):
        shard.X = sp.csr_matrix(shard.X)
    shard.X = shard.X.tocsr()

    raw_sums = GeneSums(shard.n_vars).add_matrix(shard.X)  # all cells, before QC
    _ensure_qc_metrics(shard, cfg, logger)  # per-cell total_counts / n_genes_by_counts / pct_counts_mt, as SPORE
    keep_cells, waterfall_steps = qc_cell_mask(shard, cfg, logger)
    # per-cell QC metrics of ALL cells (pre-filter), for the report stage's QC figures (about 17 bytes per cell)
    write_new_parquet(stats_unit / "qc_cells.parquet", pd.DataFrame({
        "xl_global_index": shard.obs["xl_global_index"].to_numpy().astype(np.int64),
        "total_counts": shard.obs["total_counts"].to_numpy().astype(np.float32),
        "n_genes_by_counts": shard.obs["n_genes_by_counts"].to_numpy().astype(np.int32),
        "pct_counts_mt": shard.obs["pct_counts_mt"].to_numpy().astype(np.float32),
        "qc_pass": keep_cells}))
    keep_genes = ribo_gene_mask(shard.var_names, cfg, run.ribo_exempt())

    matrix = shard.X
    if not keep_cells.all():
        matrix = matrix[keep_cells]
    if not keep_genes.all():
        matrix = matrix[:, keep_genes]
    matrix = sp.csr_matrix(matrix)
    var_reduced = var_table.loc[keep_genes].copy()
    obs = shard.obs.loc[keep_cells].copy()
    qc_shard = ad.AnnData(X=matrix, obs=obs, var=var_reduced)
    qc_shard.write_h5ad(work_unit / "qc.h5ad", compression="lzf")

    # ---- Phase 5 statistics, on the post-Phase-2 gene space (alias map excludes QC var columns)
    labels = obs[run.pert_col].astype(str).to_numpy()
    gene_to_idx = alias_map(var_reduced)
    all_labels = run.labels_table()["label"].tolist()
    gene_lists = target_genes_by_label(run, all_labels, gene_to_idx)
    target_genes = sorted({gene for genes in gene_lists.values() for gene in genes})

    control_rows = np.flatnonzero(labels == run.ctrl_label)
    bank = IntHistogramBank()
    n_controls = int(len(control_rows))
    if n_controls:
        control_csc = matrix[control_rows].tocsc()
        for gene in target_genes:
            start, stop = control_csc.indptr[gene], control_csc.indptr[gene + 1]
            bank.add_column(str(gene), control_csc.data[start:stop], n_cells=n_controls)
    bank.save(stats_unit / "bank.npz")
    raw_sums.save(stats_unit / "rawsums.npz")
    if run.escaper_mode != "off":  # P4: this shard's share of the control reference sample for the score rule
        hashes, global_index, rows = escaper_score.shard_reference(matrix, obs["xl_global_index"].to_numpy(), labels == run.ctrl_label)
        np.savez(stats_unit / "control_reference.npz", hashes=hashes, global_index=global_index, rows=rows)

    present = pd.Series(labels).value_counts()
    write_new_parquet(stats_unit / "labelcounts.parquet",
                      pd.DataFrame({"label": present.index.astype(str), "n_qc_pass": present.to_numpy()}))
    sums_rows = []
    row_groups = pd.Series(np.arange(len(labels))).groupby(labels).apply(lambda rows: rows.to_numpy())
    for label, rows in row_groups.items():
        if label == run.ctrl_label:
            continue
        for gene in gene_lists.get(label, []):
            column = matrix[rows][:, [gene]]
            sums_rows.append({"label": label, "gene_index": int(gene), "n_cells": int(len(rows)),
                              "sum": float(np.asarray(column.astype(np.float64).sum()).item())})
    write_new_parquet(stats_unit / "pertsums.parquet", pd.DataFrame(
        sums_rows, columns=["label", "gene_index", "n_cells", "sum"]))
    waterfall = Waterfall()
    for step, number in waterfall_steps.items():
        waterfall.count(step, number)
    waterfall.save(stats_unit / "waterfall.json")
    write_new_json(stats_unit / "meta.json", {
        "shard": int(index), "x_dtype": str(matrix.dtype), "n_cells_in": int(shard.n_obs),
        "n_cells_qc_pass": int(matrix.shape[0]), "n_controls_qc_pass": n_controls,
        "n_genes_in": int(shard.n_vars), "n_genes_after_ribo": int(matrix.shape[1])})
