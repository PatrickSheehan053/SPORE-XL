"""map3 (one task per block): gene mask + HVG panel, routing by the split lookup, then SPORE's hvgraw and p9 files.

For every split present in the block it writes
  processed/{name}_{split}_hvgraw/part_BBBB.h5ad   raw counts on the panel (SPORE: emit_hvgraw_splits, spore_local.py:655-673)
  processed/{name}_{split}_p9/part_BBBB.h5ad       normalize_split(hvgraw)  (SPORE Phase 9, run on the panel genes only)
Routing: perturbation cells by the split JSON, control cells by the control_assignment sidecar (both written once, in
reduce2). Nothing here decides a split.
"""
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from xl_common import (Run, block_unit, quarantine_incomplete, register_external, require_done, task)

from src.phases.phase09_normalization import normalize_split
from src.phases.phase12_metacell import _load_split_map

SPLITS = ("train", "val", "test")


def part_path(run, split, kind, block):
    return run.processed_dir / f"{run.name}_{split}_{run.kind_name(kind)}" / f"part_{block:04d}.h5ad"


def load_panel_tables(run):
    """(panel var table with the recomputed QC columns, boolean mask over the post-ribo genes, mask over Phase 6 genes)."""
    from adapted_phases.phase02_cell_triage import ribo_gene_mask
    var_table = run.var_table()
    var_reduced = var_table.loc[ribo_gene_mask(var_table.index, run.cfg, run.ribo_exempt())]
    kept_names = [n for n in (require_done(run, "stats", "reduce2", "all") / "gene_keep_mask.txt").read_text(encoding="utf-8").split("\n") if n]
    panel_names = [n for n in (run.splits_dir / f"{run.name}_hvg_panel.txt").read_text(encoding="utf-8").split("\n") if n]
    panel_mask = var_reduced.index.isin(panel_names)
    # SPORE's panel order is the var order of the post-Phase-6 genes; assert that equals the file order
    ordered = [name for name in var_reduced.index if name in set(panel_names)]
    if ordered != panel_names:
        raise RuntimeError("panel file is not in var order")
    qc = pd.read_csv(require_done(run, "stats", "reduce2", "all") / "qc_var_columns_post_ribo.tsv", sep="\t", index_col=0)
    panel_var = var_reduced.loc[panel_mask].copy()
    for column in qc.columns:
        panel_var[column] = qc.loc[panel_var.index, column].to_numpy()
    if run.layout()["x_dtype"] == "float32":
        panel_var["mean_counts"] = panel_var["mean_counts"].astype(np.float32)
    return panel_var, panel_mask, set(kept_names)


def run_map3(run: Run, block: int):
    unit = block_unit(block)
    expected = [part_path(run, split, kind, block) for split in SPLITS for kind in ("hvgraw", "p9")]
    marker = run.marker("work", "map3", unit)
    if not marker.exists():
        quarantine_incomplete(run, "map3", unit, expected)
    with task(run, "work", "map3", unit) as directory:
        if directory is None:
            return
        written = _map3(run, block)
        register_external(directory, written)


def _map3(run, block):
    cfg, logger = run.cfg, run.logger
    block_file = require_done(run, "work", "gather", block_unit(block)) / "block.h5ad"
    if not block_file.exists():
        return []
    adata = ad.read_h5ad(block_file)
    panel_var, panel_mask, _ = load_panel_tables(run)
    matrix = sp.csr_matrix(adata.X)[:, np.flatnonzero(panel_mask)]
    obs = adata.obs.copy()
    obs["xl_block"] = np.full(adata.n_obs, block, dtype=np.int64)

    pert_to_split, _ = _load_split_map(run.splits_dir / f"{run.name}_split_indices.json", run.ctrl_label, logger)
    control_split = pd.read_parquet(run.splits_dir / f"{run.name}_control_assignment.parquet").set_index("xl_global_index")["split"]
    labels = obs[run.pert_col].astype(str).to_numpy()
    gidx = obs["xl_global_index"].to_numpy()
    split_of = np.empty(len(labels), dtype=object)
    is_control = labels == run.ctrl_label
    split_of[is_control] = control_split.loc[gidx[is_control]].to_numpy()
    for index in np.flatnonzero(~is_control):
        split_of[index] = pert_to_split[labels[index]]  # KeyError here = a surviving label missing from the JSON

    written = []
    for split in SPLITS:
        rows = np.flatnonzero(split_of == split)
        if not len(rows):
            continue
        raw = ad.AnnData(X=matrix[rows].copy(), obs=obs.iloc[rows].copy(), var=panel_var.copy())
        raw_path = part_path(run, split, "hvgraw", block)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.write_h5ad(raw_path, compression=run.xl.get("bucket_compression", "lzf"))
        written.append(raw_path)
        normalized = normalize_split(raw, cfg, logger, label=f"{split}/b{block}")
        normalized.uns.pop("p9_raw_sample", None)  # SPORE's 50k-value plotting cache; not needed in the parts
        p9_path = part_path(run, split, "p9", block)
        p9_path.parent.mkdir(parents=True, exist_ok=True)
        normalized.write_h5ad(p9_path, compression=run.xl.get("bucket_compression", "lzf"))
        written.append(p9_path)
    return written
