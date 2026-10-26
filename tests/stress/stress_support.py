"""Support for the stress-suite runner: run configs, an INDEPENDENT numpy derivation of the surviving cells (SPORE Phases 2 and 5 from
first principles, no XL or SPORE code), and readers for the XL outputs."""
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent.parent
BUILD = REPO
# v2: the code under test is chosen with STRESS_CODE_DIR (the frozen 21 Oct src\ by default, or release_v4\SPORE_v4\xl) and STRESS_CLI
CODE_DIR = Path(os.environ.get("STRESS_CODE_DIR", str(REPO / "src" / "engine")))
sys.path.insert(0, str(CODE_DIR))

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import yaml  # noqa: E402

BASE_CONFIG = Path(os.environ.get("XL_TEST_DATA", str(REPO / "tests" / "ref"))) / "data" / "spore_config_synthetic60k_baseline_v1.yaml"
CONTROL = "non-targeting"


def make_config(key, name, arm, source_path, run_dir, config_path, protected_path, source_overrides=None, config_overrides=None,
                xl_overrides=None):
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["paths"]["project_root"] = str(run_dir).replace("\\", "/")
    cfg["paths"]["raw_h5ad"] = ""
    cfg["dataset"]["name"] = f"stress_{name.split('_', 1)[1]}"
    cfg["dataset"]["perturbation_col"] = "gene"
    cfg["dataset"]["control_label"] = CONTROL
    cfg["dataset"]["gene_id_format"] = key.get("gene_id_format", "symbol")
    cfg["phase7_splits"].update({"test_n": key["test_n"], "val_n": key["val_n"]})
    cfg["phase8_hvg"]["n_top_genes"] = key.get("n_top_genes", 500)
    cfg["phase8_hvg"]["protected_gene_list"] = str(protected_path).replace("\\", "/")
    n_blocks = 16 if key["n_cells"] > 85000 and key["n_perturbations"] > 500 else 8
    control_block_rows = 15000 if key["n_controls"] > 30000 else 1000
    shard_rows = int(math.ceil(key["n_cells"] / 4))
    cfg["xl"] = {"source": {"kind": "h5ad_rows", "h5ad": str(source_path).replace("\\", "/"), "shard_rows": shard_rows},
                 "work_dir": "_xl_work", "n_blocks": n_blocks, "control_block_rows": control_block_rows, "hvg_subsample_cells": 100000,
                 "bucket_compression": "lzf", "memory_budget_gb": 24, "merge_split_outputs": "never"}
    cfg["xl"]["source"].update(source_overrides or {})
    cfg["xl"].update(xl_overrides or {})
    for section, values in (config_overrides or {}).items():
        if isinstance(values, dict):
            cfg.setdefault(section, {}).update(values)
        else:
            cfg[section] = values
    Path(config_path).write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg


def dedup_like_spore(symbols):
    """phase01_detection.py:246-258: the second and later copies of a symbol get _1, _2 ... (then make_unique)."""
    seen, out = {}, []
    for name in symbols:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def independent_derivation(h5ad_path, label_column="gene", symbols=None, min_cells=50, percentile=10, ctrl=CONTROL,
                           qc=(0.2, 200, 10000, 500, 80000)):
    """Returns dict: names (obs), qc_pass, survivor, escaper_removed, dropped_labels. Gene lookup = phase05's alias map semantics
    (var names, then every var column value, later copies overwrite earlier ones)."""
    adata = ad.read_h5ad(h5ad_path)
    matrix = sp.csr_matrix(adata.X)
    labels = adata.obs[label_column].astype(str).to_numpy()
    if symbols is None:
        symbols = list(adata.var_names)
        var_names = symbols
    else:
        var_names = dedup_like_spore(symbols)
    total = np.asarray(matrix.sum(axis=1)).ravel()
    n_genes = matrix.getnnz(axis=1)
    mt_columns = np.flatnonzero(pd.Index(var_names).str.startswith("MT-"))
    percent_mt = 100 * np.asarray(matrix[:, mt_columns].sum(axis=1)).ravel() / np.maximum(total, 1e-12)
    mt_threshold, min_genes, max_genes, min_counts, max_counts = qc
    qc_pass = (percent_mt <= mt_threshold * 100) & (n_genes >= min_genes) & (n_genes <= max_genes) & (total >= min_counts) & (total <= max_counts)
    gene_to_idx = {}
    for index, name in enumerate(var_names):
        gene_to_idx[str(name)] = index
    for index, name in enumerate(symbols):  # the gene_name column of the harmonized var overwrites (last copy wins)
        gene_to_idx[str(name)] = index
    keep = qc_pass.copy()
    control_rows = np.flatnonzero(qc_pass & (labels == ctrl))
    csc = matrix.tocsc()
    for label in np.unique(labels[qc_pass]):
        if label == ctrl or label not in gene_to_idx:
            continue
        rows = np.flatnonzero(qc_pass & (labels == label))
        column = csc[:, gene_to_idx[label]]
        threshold = np.percentile(column[control_rows].toarray().ravel(), percentile)
        values = column[rows].toarray().ravel()
        keep[rows[values > threshold]] = False
    escaper_removed = qc_pass & ~keep
    sizes = pd.Series(labels[keep]).value_counts()
    dropped = set(sizes[(sizes < min_cells) & (sizes.index != ctrl)].index)
    survivor = keep & ~np.isin(labels, list(dropped))
    return {"names": np.array(adata.obs_names), "labels": labels, "qc_pass": qc_pass, "escaper_removed": escaper_removed,
            "survivor": survivor, "dropped_labels": dropped, "gene_to_idx": gene_to_idx, "var_names": var_names}


def xl_survivor_names(run, names):
    final = pd.read_parquet(run.manifest_dir / "global_obs_final.parquet")
    return set(np.asarray(names)[final.loc[final["survived_to_phase7"], "xl_global_index"].to_numpy()])


def part_files(run, split, kind):
    return sorted((run.processed_dir / f"{run.name}_{split}_{kind}").glob("part_*.h5ad"))


def read_obs(path):
    part = ad.read_h5ad(path, backed="r")
    try:
        return part.obs.copy(), list(part.var_names)
    finally:
        part.file.close()
