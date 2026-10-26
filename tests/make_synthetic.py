"""Synthetic CRISPRi Perturb-seq dataset for SPORE / SPORE-XL equivalence tests (blueprint step 2).

60,000 cells x 2,000 genes, 300 perturbations (~20 with under 50 cells), 4,000 non-targeting controls,
10 percent escaper cells, 2 batches, QC failures of three kinds, gene symbols as var_names,
raw integer counts stored as float32 CSR. Refuses to overwrite an existing output.

Run: python -B tests\\make_synthetic.py --out data\\synthetic_60k.h5ad
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 20261018
N_CELLS, N_GENES, N_PERTS, N_CONTROLS, N_SMALL = 60_000, 2_000, 300, 4_000, 20
CHUNK = 5_000

S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL",
           "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN"]
G2M_GENES = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B",
             "MKI67", "TMPO", "CENPF", "TACC3", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB"]
MT_GENES = ["MT-ND1", "MT-ND2", "MT-ND3", "MT-ND4", "MT-ND4L", "MT-ND5", "MT-ND6", "MT-CO1", "MT-CO2",
            "MT-CO3", "MT-ATP6", "MT-ATP8", "MT-CYB"]
RIBO_GENES = [f"RPL{index}" for index in (3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22)] + \
             [f"RPS{index}" for index in (2, 3, 4, 5, 6, 8, 9, 12, 13, 15, 16, 18)]


def build_gene_table(rng):
    special = MT_GENES + RIBO_GENES + S_GENES + G2M_GENES
    plain = [f"GENE{index:04d}" for index in range(N_GENES - len(special))]
    names = special + plain
    kind = (["mt"] * len(MT_GENES) + ["ribo"] * len(RIBO_GENES) + ["cc"] * (len(S_GENES) + len(G2M_GENES)) +
            ["plain"] * len(plain))
    order = rng.permutation(N_GENES)
    names = np.array(names)[order]
    kind = np.array(kind)[order]
    mean_expr = np.clip(rng.lognormal(mean=np.log(0.5), sigma=1.5, size=N_GENES), 0.02, 40.0)
    plain_positions = np.flatnonzero(kind == "plain")
    low_positions = rng.choice(plain_positions, size=220, replace=False)  # filtered by Phase 6 unless rescued
    mean_expr[low_positions] = rng.uniform(0.0008, 0.008, size=len(low_positions))
    mean_expr[kind == "mt"] = 8.0
    mean_expr[kind == "ribo"] = rng.uniform(4.0, 25.0, size=int((kind == "ribo").sum()))
    mean_expr[kind == "cc"] = rng.uniform(0.3, 3.0, size=int((kind == "cc").sum()))
    return names, kind, mean_expr, low_positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_path = Path(args.out)
    truth_path = out_path.with_name(out_path.stem + "_truth.parquet")
    design_path = out_path.with_name(out_path.stem + "_design.json")
    for path in (out_path, truth_path, design_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    rng = np.random.default_rng(SEED)
    gene_names, gene_kind, mean_expr, low_positions = build_gene_table(rng)
    mt_positions = np.flatnonzero(gene_kind == "mt")

    # --- perturbations: 300 targets (5 ribosomal, 8 cell-cycle, 15 very low expressed, rest ordinary)
    ribo_pool = np.flatnonzero(gene_kind == "ribo")
    cc_pool = np.flatnonzero(gene_kind == "cc")
    ordinary_pool = np.flatnonzero((gene_kind == "plain") & (mean_expr >= 0.8) & (mean_expr <= 40))
    target_positions = np.concatenate([
        rng.choice(ribo_pool, 5, replace=False),
        rng.choice(cc_pool, 8, replace=False),
        rng.choice(low_positions, 15, replace=False),
    ])
    remaining = N_PERTS - len(target_positions)
    target_positions = np.concatenate([target_positions, rng.choice(
        np.setdiff1d(ordinary_pool, target_positions), remaining, replace=False)])
    target_positions = target_positions[rng.permutation(N_PERTS)]
    pert_labels = gene_names[target_positions]
    knockdown_factor = rng.uniform(0.2, 0.4, size=N_PERTS)

    # downstream effects: 10-30 genes per perturbation, effect strength varies a lot (good score spread)
    effect_log = np.zeros((N_PERTS, N_GENES))
    strength = rng.lognormal(mean=np.log(0.3), sigma=0.7, size=N_PERTS)
    for index in range(N_PERTS):
        downstream = rng.choice(np.setdiff1d(np.arange(N_GENES), [target_positions[index]]),
                                size=int(rng.integers(10, 31)), replace=False)
        effect_log[index, downstream] = rng.normal(0, strength[index], size=len(downstream))
    batch_effect = np.zeros((2, N_GENES))
    shifted = rng.choice(N_GENES, size=100, replace=False)
    batch_effect[1, shifted] = rng.normal(0, 0.3, size=100)

    # --- cells per perturbation: 20 small (12-46 cells), the rest lognormal rescaled to the total
    small_sizes = rng.integers(12, 47, size=N_SMALL)
    n_big = N_PERTS - N_SMALL
    raw_sizes = rng.lognormal(mean=0.0, sigma=0.5, size=n_big)
    big_total = N_CELLS - N_CONTROLS - int(small_sizes.sum())
    big_sizes = np.maximum(70, np.floor(raw_sizes / raw_sizes.sum() * big_total)).astype(int)
    big_sizes[0] += big_total - big_sizes.sum()
    sizes = np.concatenate([small_sizes, big_sizes])
    sizes = sizes[rng.permutation(N_PERTS)]
    assert sizes.sum() + N_CONTROLS == N_CELLS, (sizes.sum(), N_CELLS)

    cell_pert = np.concatenate([np.repeat(np.arange(N_PERTS), sizes), np.full(N_CONTROLS, -1)])
    cell_order = rng.permutation(N_CELLS)  # shuffle rows so shards mix perturbations and batches
    cell_pert = cell_pert[cell_order]
    batch = rng.integers(0, 2, size=N_CELLS)
    is_perturbed = cell_pert >= 0
    is_escaper = is_perturbed & (rng.random(N_CELLS) < 0.10)

    # --- QC failures, disjoint random cells
    chosen = rng.permutation(N_CELLS)
    low_count_cells, high_mt_cells, high_count_cells = chosen[:150], chosen[150:300], chosen[300:340]
    size_factor = rng.lognormal(mean=0.0, sigma=0.35, size=N_CELLS) * np.where(batch == 1, 1.15, 1.0)
    size_factor[low_count_cells] *= 0.05
    size_factor[high_count_cells] *= 50.0
    mt_boost = np.ones(N_CELLS)
    mt_boost[high_mt_cells] = 40.0
    library_target = 2200.0
    scale = library_target / mean_expr.sum()

    # --- counts, chunked
    blocks = []
    dispersion = 5.0
    for start in range(0, N_CELLS, CHUNK):
        rows = np.arange(start, min(start + CHUNK, N_CELLS))
        log_mu = np.log(mean_expr * scale)[None, :] + batch_effect[batch[rows]]
        perturbed_rows = is_perturbed[rows] & ~is_escaper[rows]
        pert_index = cell_pert[rows]
        log_mu[perturbed_rows] += effect_log[pert_index[perturbed_rows]]
        target_cols = target_positions[pert_index[perturbed_rows]]
        log_mu[np.flatnonzero(perturbed_rows), target_cols] += np.log(knockdown_factor[pert_index[perturbed_rows]])
        log_mu[:, mt_positions] += np.log(mt_boost[rows])[:, None]
        mu = np.exp(log_mu) * size_factor[rows][:, None]
        counts = rng.poisson(rng.gamma(shape=dispersion, scale=mu / dispersion))
        blocks.append(sp.csr_matrix(counts.astype(np.float32)))
    matrix = sp.vstack(blocks, format="csr")

    labels = np.where(is_perturbed, pert_labels[np.maximum(cell_pert, 0)], "non-targeting")
    categories = sorted(set(labels))
    obs = pd.DataFrame({
        "gene": pd.Categorical(labels, categories=categories),
        "gem_group": pd.Categorical((batch + 1).astype(str), categories=["1", "2"]),
    }, index=pd.Index([f"cell_{index:06d}" for index in range(N_CELLS)], name="cell"))
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene_symbol"))
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.write_h5ad(out_path, compression="gzip")

    truth = pd.DataFrame({
        "cell": obs.index, "row": np.arange(N_CELLS), "label": labels, "batch": batch + 1,
        "is_escaper": is_escaper,
        "qc_type": np.select([np.isin(np.arange(N_CELLS), low_count_cells),
                              np.isin(np.arange(N_CELLS), high_mt_cells),
                              np.isin(np.arange(N_CELLS), high_count_cells)],
                             ["low_count", "high_mt", "high_count"], default="ok"),
    })
    truth.to_parquet(truth_path)
    small_labels = sorted(pert_labels[np.argsort(sizes)[:N_SMALL]])
    design = {
        "seed": SEED, "n_cells": N_CELLS, "n_genes": N_GENES, "n_perturbations": N_PERTS,
        "n_controls": N_CONTROLS, "n_escapers": int(is_escaper.sum()),
        "qc_failures": {"low_count": 150, "high_mt": 150, "high_count": 40},
        "small_perturbations_under_50_cells": small_labels,
        "ribosomal_targets": sorted(set(pert_labels) & set(gene_names[gene_kind == "ribo"])),
        "cell_cycle_targets": sorted(set(pert_labels) & set(gene_names[gene_kind == "cc"])),
        "low_expressed_targets": sorted(set(pert_labels) & set(gene_names[low_positions])),
        "x_dtype": "float32 integer-valued CSR", "perturbation_col": "gene", "control_label": "non-targeting",
        "batch_col": "gem_group",
    }
    design_path.write_text(json.dumps(design, indent=2), encoding="utf-8")
    print(f"wrote {out_path}  {out_path.stat().st_size / 1e6:.1f} MB  nnz={matrix.nnz:,}  "
          f"density={matrix.nnz / (N_CELLS * N_GENES):.3f}  mean library={matrix.sum() / N_CELLS:.0f}")


if __name__ == "__main__":
    main()
