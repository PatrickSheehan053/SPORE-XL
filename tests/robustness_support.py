"""Helpers for the T4/T4b robustness tests: crafted variants of synthetic_60k, XL configs for them, and an INDEPENDENT numpy
re-derivation of the survivors (SPORE Phases 2 and 5 plus the undersized drop) that does not use XL or SPORE code."""
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import BASE_CONFIG, BUILD, SOURCE  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import yaml  # noqa: E402

CONTROL = "non-targeting"


def scratch_dir(prefix):
    base = os.environ.get("XL_TEST_TMP", tempfile.gettempdir())
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))  # not cleaned up (no deletion without Patrick)


def load_source():
    adata = ad.read_h5ad(SOURCE)
    adata.X = sp.csr_matrix(adata.X)
    adata.obs["gene"] = adata.obs["gene"].astype(str)
    return adata


def finish(adata, label):
    """Categorical pert column with SORTED categories (the intake convention), written to scratch."""
    adata = adata.copy()
    adata.obs["gene"] = pd.Categorical(adata.obs["gene"].astype(str))
    adata.obs["gem_group"] = pd.Categorical(adata.obs["gem_group"].astype(str))
    path = scratch_dir(f"xl_variant_{label}_") / f"{label}.h5ad"
    adata.write_h5ad(path, compression="lzf")
    return path, adata


def small_subset(adata, n_perts=120):
    sizes = adata.obs["gene"].value_counts()
    keep = list(sizes[sizes.index != CONTROL].index[:n_perts]) + [CONTROL]
    return adata[adata.obs["gene"].isin(keep).to_numpy()].copy()


def variant_config(label, source_path, shard_rows=None, kind="h5ad_rows", obs_key=None, overrides=None,
                   n_blocks=8, control_block_rows=1000, test_n=15, val_n=8, memory_budget_gb=24):
    run_dir = scratch_dir(f"xl_run_{label}_")
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["paths"]["project_root"] = str(run_dir).replace("\\", "/")
    cfg["paths"]["raw_h5ad"] = ""
    cfg["phase7_splits"].update({"test_n": test_n, "val_n": val_n})
    source = {"kind": kind, "h5ad": str(source_path).replace("\\", "/")}
    if kind == "h5ad_rows":
        source["shard_rows"] = int(shard_rows)
    else:
        source["obs_key"] = obs_key
    cfg["xl"] = {"compat": "v3", "source": source, "work_dir": "_xl_work", "n_blocks": n_blocks, "control_block_rows": control_block_rows,
                 "hvg_subsample_cells": 100000, "bucket_compression": "lzf", "memory_budget_gb": memory_budget_gb}
    for section, values in (overrides or {}).items():
        if isinstance(values, dict):
            cfg.setdefault(section, {}).update(values)
        else:
            cfg[section] = values
    config_path = run_dir.parent / f"{run_dir.name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


def run_through(config_path, through="finalize"):
    from xl_common import Run
    import cli as spore_xl
    run = Run(config_path)
    order = ["plan", "map1", "reduce1", "map2", "gather", "reduce2", "hvg", "map3", "metacell_prep", "metacell", "finalize"]
    for stage in order[:order.index(through) + 1]:
        spore_xl.run_stage(run, stage)
    return run


def independent_survivors(adata, min_cells=50, percentile=10):
    """Phases 2 and 5 from first principles. Returns the set of surviving cell names."""
    matrix = sp.csr_matrix(adata.X)
    labels = adata.obs["gene"].astype(str).to_numpy()
    total = np.asarray(matrix.sum(axis=1)).ravel()
    n_genes = matrix.getnnz(axis=1)
    mt_columns = np.flatnonzero(adata.var_names.str.startswith("MT-"))
    percent_mt = 100 * np.asarray(matrix[:, mt_columns].sum(axis=1)).ravel() / np.maximum(total, 1e-12)
    qc_pass = (percent_mt <= 20) & (n_genes >= 200) & (n_genes <= 10000) & (total >= 500) & (total <= 80000)
    keep = qc_pass.copy()
    control_rows = np.flatnonzero(qc_pass & (labels == CONTROL))
    column_of = {gene: index for index, gene in enumerate(adata.var_names)}
    csc = matrix.tocsc()
    for label in np.unique(labels[qc_pass]):
        if label == CONTROL or label not in column_of:  # a label that is not a gene keeps all its cells (bypass)
            continue
        rows = np.flatnonzero(qc_pass & (labels == label))
        column = column_of[label]
        threshold = np.percentile(csc[:, column][control_rows].toarray().ravel(), percentile)
        values = csc[:, column][rows].toarray().ravel()
        keep[rows[values > threshold]] = False
    sizes = pd.Series(labels[keep]).value_counts()
    undersized = set(sizes[(sizes < min_cells) & (sizes.index != CONTROL)].index)
    keep &= ~np.isin(labels, list(undersized))
    return set(adata.obs_names[keep])


def xl_survivors(run, adata):
    final = pd.read_parquet(run.manifest_dir / "global_obs_final.parquet")
    names = np.array(adata.obs_names)
    return set(names[final.loc[final["survived_to_phase7"], "xl_global_index"].to_numpy()])
