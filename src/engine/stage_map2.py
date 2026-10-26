"""map2 (one task per shard): apply the escaper filter (adapted_phases/phase05) to the QC shard and route the kept cells
into perturbation-block buckets (_xl_work/map2/shard_NNNN/block_BBBB.h5ad).

stats/map2/shard_NNNN/counts.parquet: label, n_total (QC-passing cells), n_kept, bypassed. Nothing about gene
statistics or undersized perturbations is decided here (blueprint Amendment 1 item 2).
"""
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from xl_common import (Run, alias_map, block_of_label, control_block_of, require_done, shard_unit, task,
                       write_new_json, write_new_parquet)
from stage_map1 import target_genes_by_label
from stats import Waterfall

from adapted_phases.phase05_escaper_filtering import gene_pass
from src.phases.phase05_escaper_filtering import _get_escaper_direction


def load_thresholds(run):
    path = require_done(run, "stats", "reduce1", "all") / "escaper_thresholds.tsv"
    frame = pd.read_csv(path, sep="\t")
    return {int(row.gene_index): np.dtype(row.threshold_dtype).type(float(row.threshold)) for row in frame.itertuples()}


def run_map2(run: Run, index: int):
    unit = shard_unit(index)
    with task(run, "stats", "map2", unit) as stats_unit:
        if stats_unit is None:
            return
        with task(run, "work", "map2", unit, log=False) as work_unit:
            _map2(run, index, stats_unit, work_unit)


def _map2(run, index, stats_unit, work_unit):
    unit = shard_unit(index)
    qc_path = require_done(run, "work", "map1", unit) / "qc.h5ad"
    qc = ad.read_h5ad(qc_path)
    thresholds = load_thresholds(run)
    direction = _get_escaper_direction(run.cfg, run.logger)
    layout = run.layout()
    matrix = sp.csr_matrix(qc.X)
    labels = qc.obs[run.pert_col].astype(str).to_numpy()

    gene_to_idx = alias_map(qc.var)
    all_labels = run.labels_table()["label"].tolist()
    gene_lists = target_genes_by_label(run, all_labels, gene_to_idx)

    keep = np.ones(len(labels), dtype=bool)  # controls are never filtered (phase05:109)
    rows_by_label = pd.Series(np.arange(len(labels))).groupby(labels).apply(lambda rows: rows.to_numpy())
    records = []
    for label, rows in rows_by_label.items():
        if label == run.ctrl_label:
            continue
        genes = gene_lists.get(label, [])
        if not genes:  # phase05:120-130
            records.append({"label": label, "n_total": int(len(rows)), "n_kept": int(len(rows)), "bypassed": True})
            continue
        pass_mask = np.ones(len(rows), dtype=bool)
        block = matrix[rows]
        for gene in genes:
            values = block[:, gene].toarray().flatten()
            pass_mask &= gene_pass(values, thresholds[gene], direction)
        keep[rows] = pass_mask
        records.append({"label": label, "n_total": int(len(rows)), "n_kept": int(pass_mask.sum()), "bypassed": False})
    write_new_parquet(stats_unit / "counts.parquet", pd.DataFrame(
        records, columns=["label", "n_total", "n_kept", "bypassed"]))

    waterfall = Waterfall()
    waterfall.count("QC-passing cells", len(labels)).count("Escapers removed", int((~keep).sum()))
    waterfall.save(stats_unit / "waterfall.json")

    # ---- bucket the kept cells by block
    global_index = qc.obs["xl_global_index"].to_numpy()
    n_blocks, cuts = layout["n_blocks"], layout["control_cuts"]
    block_cache, block_of_row = {}, np.empty(len(labels), dtype=np.int64)
    for row_index, (label, gidx) in enumerate(zip(labels, global_index)):
        if label == run.ctrl_label:
            block_of_row[row_index] = control_block_of(gidx, cuts, n_blocks)
        else:
            if label not in block_cache:
                block_cache[label] = block_of_label(label, n_blocks)
            block_of_row[row_index] = block_cache[label]
    written = []
    carry = run.escaper_mode != "off"  # P4: rule-1 escapers travel to gather flagged, so rule 2 sees every QC-passing cell of the perturbation
    bucketed = np.ones(len(labels), dtype=bool) if carry else keep
    for block in np.unique(block_of_row[bucketed]):
        rows = np.flatnonzero(bucketed & (block_of_row == block))
        piece_obs = qc.obs.iloc[rows].copy()
        if carry:
            piece_obs["escaper_rule1"] = ~keep[rows]
        piece = ad.AnnData(X=matrix[rows], obs=piece_obs, var=qc.var.copy())
        piece.write_h5ad(work_unit / f"block_{int(block):04d}.h5ad", compression=run.xl.get("bucket_compression", "lzf"))
        written.append(int(block))
    write_new_json(stats_unit / "buckets.json", {"blocks": written, "n_cells_kept": int(keep.sum())})
