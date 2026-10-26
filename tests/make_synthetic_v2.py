"""Parametrized synthetic CRISPRi generator for the memory-flatness proof (task 7): same biology as make_synthetic.py
(NB counts, CRISPRi knockdown plus downstream effects, escapers, 2 batches, three kinds of QC failure, ribosomal / cell-cycle /
low-expressed targets), scaled to n cells. Different seed and fresh output names; refuses to overwrite.

Run: python -B tests\\make_synthetic_v2.py --out data\\synthetic_1m.h5ad --cells 1000000 --perts 1000 --controls 40000 --small 50
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from make_synthetic import build_gene_table

CHUNK = 5_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--perts", type=int, required=True)
    parser.add_argument("--controls", type=int, required=True)
    parser.add_argument("--small", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--size-sigma", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=70)
    args = parser.parse_args()
    out_path = Path(args.out)
    design_path = out_path.with_name(out_path.stem + "_design.json")
    for path in (out_path, design_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    n_cells, n_perts, n_controls, n_small = args.cells, args.perts, args.controls, args.small
    n_genes = 2000

    rng = np.random.default_rng(args.seed)
    gene_names, gene_kind, mean_expr, low_positions = build_gene_table(rng)
    mt_positions = np.flatnonzero(gene_kind == "mt")
    ribo_pool, cc_pool = np.flatnonzero(gene_kind == "ribo"), np.flatnonzero(gene_kind == "cc")
    ordinary_pool = np.flatnonzero((gene_kind == "plain") & (mean_expr >= 0.25) & (mean_expr <= 40))
    fixed = np.concatenate([rng.choice(ribo_pool, 5, replace=False), rng.choice(cc_pool, 8, replace=False),
                            rng.choice(low_positions, 15, replace=False)])
    remaining = n_perts - len(fixed)
    pool = np.setdiff1d(ordinary_pool, fixed)
    if remaining > len(pool):
        raise ValueError(f"only {len(pool)} ordinary target genes available for {remaining} perturbations")
    target_positions = np.concatenate([fixed, rng.choice(pool, remaining, replace=False)])
    target_positions = target_positions[rng.permutation(n_perts)]
    pert_labels = gene_names[target_positions]
    knockdown_factor = rng.uniform(0.2, 0.4, size=n_perts)

    effect_log = np.zeros((n_perts, n_genes), dtype=np.float32)
    strength = rng.lognormal(mean=np.log(0.3), sigma=0.7, size=n_perts)
    for index in range(n_perts):
        downstream = rng.choice(np.setdiff1d(np.arange(n_genes), [target_positions[index]]), size=int(rng.integers(10, 31)), replace=False)
        effect_log[index, downstream] = rng.normal(0, strength[index], size=len(downstream))
    batch_effect = np.zeros((2, n_genes))
    shifted = rng.choice(n_genes, size=100, replace=False)
    batch_effect[1, shifted] = rng.normal(0, 0.3, size=100)

    small_sizes = rng.integers(12, 47, size=n_small)
    n_big = n_perts - n_small
    raw_sizes = rng.lognormal(mean=0.0, sigma=args.size_sigma, size=n_big)
    big_total = n_cells - n_controls - int(small_sizes.sum())
    big_sizes = np.maximum(args.min_size, np.floor(raw_sizes / raw_sizes.sum() * big_total)).astype(int)
    big_sizes[0] += big_total - big_sizes.sum()
    sizes = np.concatenate([small_sizes, big_sizes])
    sizes = sizes[rng.permutation(n_perts)]
    assert sizes.sum() + n_controls == n_cells, (sizes.sum(), n_cells)

    cell_pert = np.concatenate([np.repeat(np.arange(n_perts), sizes), np.full(n_controls, -1)])
    cell_pert = cell_pert[rng.permutation(n_cells)]
    batch = rng.integers(0, 2, size=n_cells)
    is_perturbed = cell_pert >= 0
    is_escaper = is_perturbed & (rng.random(n_cells) < 0.10)

    n_low, n_mt, n_high = int(0.0025 * n_cells), int(0.0025 * n_cells), int(0.00067 * n_cells)
    chosen = rng.permutation(n_cells)
    low_count_cells, high_mt_cells, high_count_cells = chosen[:n_low], chosen[n_low:n_low + n_mt], chosen[n_low + n_mt:n_low + n_mt + n_high]
    size_factor = rng.lognormal(mean=0.0, sigma=0.35, size=n_cells) * np.where(batch == 1, 1.15, 1.0)
    size_factor[low_count_cells] *= 0.05
    size_factor[high_count_cells] *= 50.0
    mt_boost = np.ones(n_cells)
    mt_boost[high_mt_cells] = 40.0
    scale = 2200.0 / mean_expr.sum()

    data_parts, index_parts, row_nnz = [], [], []
    for start in range(0, n_cells, CHUNK):
        rows = np.arange(start, min(start + CHUNK, n_cells))
        log_mu = np.log(mean_expr * scale)[None, :] + batch_effect[batch[rows]]
        perturbed_rows = is_perturbed[rows] & ~is_escaper[rows]
        pert_index = cell_pert[rows]
        log_mu[perturbed_rows] += effect_log[pert_index[perturbed_rows]]
        target_cols = target_positions[pert_index[perturbed_rows]]
        log_mu[np.flatnonzero(perturbed_rows), target_cols] += np.log(knockdown_factor[pert_index[perturbed_rows]])
        log_mu[:, mt_positions] += np.log(mt_boost[rows])[:, None]
        mu = np.exp(log_mu) * size_factor[rows][:, None]
        counts = rng.poisson(rng.gamma(shape=5.0, scale=mu / 5.0))
        block = sp.csr_matrix(counts.astype(np.float32))
        data_parts.append(block.data)
        index_parts.append(block.indices)
        row_nnz.append(np.diff(block.indptr))
        if (start // CHUNK) % 40 == 0:
            print(f"  generated {start + len(rows):,}/{n_cells:,} cells", flush=True)
    indptr = np.concatenate([[0], np.cumsum(np.concatenate(row_nnz))]).astype(np.int64)
    data = np.concatenate(data_parts)
    del data_parts
    indices = np.concatenate(index_parts)
    del index_parts
    matrix = sp.csr_matrix((data, indices, indptr), shape=(n_cells, n_genes))

    labels = np.where(is_perturbed, pert_labels[np.maximum(cell_pert, 0)], "non-targeting")
    categories = sorted(set(labels))
    obs = pd.DataFrame({"gene": pd.Categorical(labels, categories=categories),
                        "gem_group": pd.Categorical((batch + 1).astype(str), categories=["1", "2"])},
                       index=pd.Index([f"cell_{index:07d}" for index in range(n_cells)], name="cell"))
    adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=pd.Index(gene_names, name="gene_symbol")))
    adata.write_h5ad(out_path, compression="lzf")
    design_path.write_text(json.dumps({
        "seed": args.seed, "n_cells": n_cells, "n_genes": n_genes, "n_perturbations": n_perts, "n_controls": n_controls,
        "n_small": n_small, "n_escapers": int(is_escaper.sum()), "nnz": int(matrix.nnz), "x_dtype": "float32 integer-valued CSR",
        "perturbation_col": "gene", "control_label": "non-targeting", "batch_col": "gem_group"}, indent=2), encoding="utf-8")
    print(f"wrote {out_path}  {out_path.stat().st_size / 1e9:.2f} GB  nnz={matrix.nnz:,}  cells={n_cells:,}", flush=True)


if __name__ == "__main__":
    main()
