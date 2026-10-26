"""metacell_prep (one job) and metacell (one task per block): SPORE Phase 12 per perturbation group.

metacell_prep fits the fallback embedding model (the one allowed deviation) on the HVG subsample, in panel space.
metacell writes processed/{name}_allsplits_metacell_ctrlpreserved/part_BBBB.h5ad: perturbation groups -> MiniBatchKMeans
k metacells (mean of raw counts), control cells pass through single-cell. obs: gene, n_cells_in_metacell, split, plus
xl_block, xl_unit (cluster order inside the group), xl_global_index and xl_control_split (controls only) so `finalize`
can restore SPORE's unit order. stats/metacell/block_BBBB/groups.parquet records every group (n cells, units, whether the
fallback embedding was used).
"""
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from xl_common import (Run, block_unit, quarantine_incomplete, register_external, require_done, task, write_new_json,
                       write_new_parquet)
from stage_gather import total_blocks
from stage_map3 import SPLITS, part_path

from adapted_phases.phase12_metacell import aggregate_group, fit_fallback, group_embedding
from src.phases.phase12_metacell import _load_split_map


def metacell_part_path(run, block):
    return run.processed_dir / f"{run.name}_allsplits_metacell_ctrlpreserved" / f"part_{block:04d}.h5ad"


def run_metacell_prep(run: Run):
    with task(run, "stats", "metacell_prep", "all") as out:
        if out is None:
            return
        p12 = run.cfg.get("phase12_metacell", {})
        seed = int(p12.get("seed", run.cfg.get("phase7_splits", {}).get("random_seed", 42)))
        sample = pd.read_parquet(require_done(run, "stats", "hvg", "all") / "hvg_subsample_index.parquet")
        wanted = sample["xl_global_index"].to_numpy()
        pieces, indexes = [], []
        for block in range(total_blocks(run)):
            path = part_path(run, "train", "hvgraw", block)
            if not path.exists():
                continue
            part = ad.read_h5ad(path)
            gidx = part.obs["xl_global_index"].to_numpy()
            rows = np.flatnonzero(np.isin(gidx, wanted))
            if len(rows):
                pieces.append(sp.csr_matrix(part.X[rows]))
                indexes.append(gidx[rows])
        matrix = sp.vstack(pieces, format="csr")[np.argsort(np.concatenate(indexes), kind="stable")]
        model = fit_fallback(matrix, seed)
        np.savez(out / "fallback_model.npz", target=np.array([model["target"]]), mean=model["mean"],
                 components=model["components"])
        write_new_json(out / "summary.json", {"n_subsample": int(matrix.shape[0]), "n_genes": int(matrix.shape[1]),
                                              "n_components": int(model["components"].shape[0]), "seed": seed})
        run.logger.info(f"[metacell_prep] fallback PCA fitted on {matrix.shape[0]:,} subsample cells")


def load_fallback_model(run):
    path = require_done(run, "stats", "metacell_prep", "all") / "fallback_model.npz"
    with np.load(path) as data:
        return {"target": float(data["target"][0]), "mean": data["mean"].copy(), "components": data["components"].copy()}


def run_metacell(run: Run, block: int):
    unit = block_unit(block)
    expected = [metacell_part_path(run, block)]
    if not run.marker("stats", "metacell", unit).exists():
        quarantine_incomplete(run, "metacell", unit, expected)
    with task(run, "stats", "metacell", unit) as directory:
        if directory is None:
            return
        written = _metacell(run, block, directory)
        register_external(directory, written)


def _metacell(run, block, directory):
    cfg, logger = run.cfg, run.logger
    p12 = cfg.get("phase12_metacell", {})
    k = int(p12.get("mbk_k", 2))
    seed = int(p12.get("seed", cfg.get("phase7_splits", {}).get("random_seed", 42)))
    if not p12.get("enabled", True):
        return []
    pieces = []
    for split in SPLITS:
        path = part_path(run, split, "hvgraw", block)
        if path.exists():
            pieces.append(ad.read_h5ad(path))
    if not pieces:
        write_new_parquet(directory / "groups.parquet", pd.DataFrame(
            columns=["label", "split", "n_cells", "n_units", "used_fallback", "has_local_embedding"]))
        return []
    matrix = sp.vstack([piece.X for piece in pieces], format="csr")
    obs = pd.concat([piece.obs for piece in pieces])
    var = pieces[0].var
    order = np.argsort(obs["xl_global_index"].to_numpy(), kind="stable")  # groups keep SPORE's row order
    matrix, obs = matrix[order], obs.iloc[order]
    pert_to_split, _ = _load_split_map(run.splits_dir / f"{run.name}_split_indices.json", run.ctrl_label, logger)
    control_split = pd.read_parquet(run.splits_dir / f"{run.name}_control_assignment.parquet").set_index("xl_global_index")["split"]
    labels = obs[run.pert_col].astype(str).to_numpy()
    model = None

    rows_out, obs_out, group_records = [], [], []
    for label in np.unique(labels):
        mask = labels == label
        X_sub = matrix[mask]
        if label == run.ctrl_label:
            dense = X_sub.toarray()
            gidx = obs["xl_global_index"].to_numpy()[mask]
            for position in range(dense.shape[0]):
                rows_out.append(dense[position].astype(np.float32))
                obs_out.append({run.pert_col: label, "n_cells_in_metacell": 1, "split": "control", "xl_unit": position,
                                "xl_global_index": int(gidx[position]),
                                "xl_control_split": control_split.loc[gidx[position]]})
            group_records.append({"label": label, "split": "control", "n_cells": int(mask.sum()),
                                  "n_units": int(mask.sum()), "used_fallback": False, "has_local_embedding": False})
            continue
        local = group_embedding(X_sub, seed)
        need_model = local is None and max(1, X_sub.shape[0] // k) > 1
        if need_model and model is None:
            model = load_fallback_model(run)
        exprs, ncells, used_fallback = aggregate_group(X_sub, local, model if need_model else None, k, seed)
        for unit_index, (expr, count) in enumerate(zip(exprs, ncells)):
            rows_out.append(expr)
            obs_out.append({run.pert_col: label, "n_cells_in_metacell": int(count), "split": pert_to_split[label],
                            "xl_unit": unit_index, "xl_global_index": -1, "xl_control_split": ""})
        group_records.append({"label": label, "split": pert_to_split[label], "n_cells": int(X_sub.shape[0]),
                              "n_units": len(exprs), "used_fallback": bool(used_fallback),
                              "has_local_embedding": local is not None})
    write_new_parquet(directory / "groups.parquet", pd.DataFrame(group_records))
    unit_obs = pd.DataFrame(obs_out)
    unit_obs["xl_block"] = block
    unit_obs.index = [f"b{block:04d}_u{index:06d}" for index in range(len(unit_obs))]
    units = ad.AnnData(X=sp.csr_matrix(np.vstack(rows_out).astype(np.float32)), obs=unit_obs, var=var.copy())
    path = metacell_part_path(run, block)
    path.parent.mkdir(parents=True, exist_ok=True)
    units.write_h5ad(path, compression=run.xl.get("bucket_compression", "lzf"))
    return [path]
