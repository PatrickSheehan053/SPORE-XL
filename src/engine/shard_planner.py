"""Shard planner for single large .h5ad sources (blueprint component 2): h5ad_rows and h5ad_obs.

Every shard is a small .h5ad holding the source var, a slice of obs and X, plus two obs columns:
  xl_global_index  the row position of the cell in the SOURCE file (this is SPORE's row index, which
                   is what makes exact equivalence possible; in both modes it is the source row)
  xl_shard         shard number
Writes manifest/shards.tsv and manifest/source.json. Idempotent: a finished shard has a .done file
(sha256 + sizes); re-running verifies and skips. Nothing is ever deleted or overwritten: a shard file
without a matching .done is reported and the run stops.
"""
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

SHARD_COLUMNS = ["shard", "file", "kind", "n_cells", "row_start", "row_stop", "obs_value",
                 "first_global_index", "last_global_index", "size_bytes", "sha256"]


def sha256_file(path, block_bytes=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def measure_h5ad(path):
    """Header-only facts: shape and stored nonzeros. No matrix values are read for CSR/CSC files."""
    with h5py.File(path, "r") as handle:
        matrix = handle["X"]
        if isinstance(matrix, h5py.Group):
            shape = tuple(int(value) for value in matrix.attrs["shape"])
            nnz = int(matrix["indptr"][-1])
            fmt = matrix.attrs.get("encoding-type", "csr_matrix")
        else:
            shape = tuple(int(value) for value in matrix.shape)
            sample = np.asarray(matrix[:min(2000, shape[0])])
            nnz = int(round((sample != 0).sum() / sample.shape[0] * shape[0]))
            fmt = "dense"
    return {"n_obs": shape[0], "n_vars": shape[1], "nnz": nnz, "nnz_per_cell": nnz / max(shape[0], 1),
            "format": str(fmt)}


def derive_shard_rows(memory_budget_gb, nnz_per_cell, working_copy_factor=3, bytes_per_value=12):
    """Blueprint: shard_rows = budget / (nnz per cell x 12 bytes x working-copy factor)."""
    return max(1, int(memory_budget_gb * 1e9 / (nnz_per_cell * bytes_per_value * working_copy_factor)))


def _source_summary(path, backed, pert_col):
    summary = {
        "path": str(path), "size_bytes": os.path.getsize(path), "n_obs": int(backed.n_obs),
        "n_vars": int(backed.n_vars),
        "var_names_sha256": hashlib.sha256("\n".join(map(str, backed.var_names)).encode("utf-8")).hexdigest(),
        "obs_columns": {column: str(backed.obs[column].dtype) for column in backed.obs.columns},
    }
    if pert_col is not None:
        column = backed.obs[pert_col]
        summary["perturbation_col"] = pert_col
        summary["perturbation_categorical"] = bool(isinstance(column.dtype, pd.CategoricalDtype))
        if summary["perturbation_categorical"]:
            # order matters: SPORE's HVG subsample draw iterates groups in category order
            summary["perturbation_categories"] = [str(value) for value in column.cat.categories]
    return summary


def _write_shard(backed, row_index, shard_number, path, compression):
    obs = backed.obs.iloc[row_index].copy()
    obs["xl_global_index"] = row_index.astype(np.int64)
    obs["xl_shard"] = np.full(len(row_index), shard_number, dtype=np.int64)
    if len(row_index) and np.array_equal(row_index, np.arange(row_index[0], row_index[-1] + 1)):
        matrix = backed.X[int(row_index[0]):int(row_index[-1]) + 1]
    else:
        matrix = backed.X[row_index]
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    shard = ad.AnnData(X=matrix.tocsr(), obs=obs, var=backed.var.copy())
    temp_path = path.with_name(path.stem + ".tmp.h5ad")
    if temp_path.exists():
        raise FileExistsError(f"leftover temp shard {temp_path}; Patrick decides whether to remove it")
    shard.write_h5ad(temp_path, compression=compression)
    os.replace(temp_path, path)


def _done_path(path):
    return path.with_name(path.name + ".done")


def _finish_or_verify(path, record_extra):
    done = _done_path(path)
    if done.exists():
        info = json.loads(done.read_text(encoding="utf-8"))
        if os.path.getsize(path) != info["size_bytes"] or sha256_file(path) != info["sha256"]:
            raise RuntimeError(f"{path} does not match its .done checksum")
        return info
    info = {"size_bytes": os.path.getsize(path), "sha256": sha256_file(path), **record_extra}
    done.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def _shard_task(arguments):
    """One shard written by a pool worker (P10): opens its own backed handle, writes, checksums, marks done. Same bytes as the serial path."""
    h5ad_path, row_index, shard_number, path_text, compression = arguments
    path = Path(path_text)
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if path.exists() and not _done_path(path).exists():
            raise FileExistsError(f"{path} exists without a .done marker (interrupted write); Patrick decides")
        if not path.exists():
            _write_shard(backed, row_index, shard_number, path, compression)
        info = _finish_or_verify(path, {"n_cells": int(len(row_index))})
    finally:
        backed.file.close()
    return shard_number, info


def _plan(h5ad_path, out_dir, groups, kind, pert_col, compression, workers=1):
    """groups: list of (row_index array, row_start, row_stop, obs_value). workers > 1: shards are written by a spawn pool (P10)."""
    out_dir = Path(out_dir)
    shard_dir = out_dir / "shards"
    manifest_dir = out_dir / "manifest"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        summary = _source_summary(h5ad_path, backed, pert_col)
        source_path = manifest_dir / "source.json"
        if source_path.exists():
            previous = json.loads(source_path.read_text(encoding="utf-8"))
            for key in ("n_obs", "n_vars", "var_names_sha256", "perturbation_categories"):
                if previous.get(key) != summary.get(key):
                    raise RuntimeError(f"existing plan in {out_dir} was made from a different source ({key} differs)")
        else:
            source_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        rows = []
        infos = {}
        if workers > 1 and len(groups) > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            jobs = [(str(h5ad_path), row_index, number, str(shard_dir / f"shard_{number:04d}.h5ad"), compression)
                    for number, (row_index, _, _, _) in enumerate(groups)]
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as pool:
                for number, info in pool.map(_shard_task, jobs):
                    infos[number] = info
        for shard_number, (row_index, row_start, row_stop, obs_value) in enumerate(groups):
            path = shard_dir / f"shard_{shard_number:04d}.h5ad"
            if shard_number in infos:
                info = infos[shard_number]
            else:
                if path.exists() and not _done_path(path).exists():
                    raise FileExistsError(f"{path} exists without a .done marker (interrupted write); Patrick decides")
                if not path.exists():
                    _write_shard(backed, row_index, shard_number, path, compression)
                info = _finish_or_verify(path, {"n_cells": int(len(row_index))})
            if info["n_cells"] != len(row_index):
                raise RuntimeError(f"{path} was written for a different plan ({info['n_cells']} cells, now {len(row_index)})")
            rows.append({
                "shard": shard_number, "file": str(path.relative_to(out_dir)).replace("\\", "/"), "kind": kind,
                "n_cells": int(len(row_index)), "row_start": row_start, "row_stop": row_stop,
                "obs_value": obs_value, "first_global_index": int(row_index[0]) if len(row_index) else -1,
                "last_global_index": int(row_index[-1]) if len(row_index) else -1,
                "size_bytes": info["size_bytes"], "sha256": info["sha256"],
            })
    finally:
        backed.file.close()
    table = pd.DataFrame(rows, columns=SHARD_COLUMNS)
    table_path = manifest_dir / "shards.tsv"
    if table_path.exists():
        existing = pd.read_csv(table_path, sep="\t", dtype=str, keep_default_na=False)
        if not existing.equals(table.astype(str)):
            raise RuntimeError(f"{table_path} exists and differs from this plan")
    else:
        table.to_csv(table_path, sep="\t", index=False)
    return table


def plan_h5ad_rows(h5ad_path, out_dir, shard_rows=None, memory_budget_gb=None, pert_col=None,
                   compression="lzf", workers=1):
    """Contiguous row ranges of at most shard_rows cells (derived from memory_budget_gb if not given)."""
    facts = measure_h5ad(h5ad_path)
    if shard_rows is None:
        if memory_budget_gb is None:
            raise ValueError("give shard_rows or memory_budget_gb")
        shard_rows = derive_shard_rows(memory_budget_gb, facts["nnz_per_cell"])
    groups = []
    for start in range(0, facts["n_obs"], shard_rows):
        stop = min(start + shard_rows, facts["n_obs"])
        groups.append((np.arange(start, stop), start, stop, ""))
    return _plan(h5ad_path, out_dir, groups, "h5ad_rows", pert_col, compression, workers)


def plan_h5ad_obs(h5ad_path, out_dir, obs_key, pert_col=None, compression="lzf", workers=1):
    """One shard per distinct value of obs[obs_key], cells in source order (shards are not contiguous;
    xl_global_index is still the source row, so ordering by it restores the source order)."""
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        values = backed.obs[obs_key].astype(str).to_numpy()
    finally:
        backed.file.close()
    groups = []
    for value in sorted(set(values)):
        row_index = np.flatnonzero(values == value)
        groups.append((row_index, "", "", value))
    return _plan(h5ad_path, out_dir, groups, "h5ad_obs", pert_col, compression, workers)


def load_plan(out_dir):
    out_dir = Path(out_dir)
    table = pd.read_csv(out_dir / "manifest" / "shards.tsv", sep="\t", keep_default_na=False)
    source = json.loads((out_dir / "manifest" / "source.json").read_text(encoding="utf-8"))
    return table, source


def read_shard(out_dir, file):
    return ad.read_h5ad(Path(out_dir) / file)
