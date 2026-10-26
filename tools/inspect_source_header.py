"""Read-only look at an .h5ad before planning a run: matrix format and dtype, obs/var columns, label and control counts, and a
line per fact that differs from what the config expects. Writes nothing (safe on another user's file).

  python -B src/inspect_source_header.py <file.h5ad> [--pert-col gene] [--control non-targeting] [--batch-col gem_group]
          [--expect-cells 1989578] [--expect-genes 8248] [--sample-values 2000000]
"""
import argparse
import sys

import h5py
import numpy as np
import pandas as pd


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--pert-col", default="gene")
    parser.add_argument("--control", default="non-targeting")
    parser.add_argument("--batch-col", default="gem_group")
    parser.add_argument("--expect-cells", type=int, default=None)
    parser.add_argument("--expect-genes", type=int, default=None)
    parser.add_argument("--sample-values", type=int, default=2_000_000)
    args = parser.parse_args(argv)
    differences = []

    with h5py.File(args.path, "r") as handle:
        x_node = handle["X"]
        if isinstance(x_node, h5py.Group):
            encoding = x_node.attrs.get("encoding-type", b"")
            encoding = encoding.decode() if isinstance(encoding, bytes) else str(encoding)
            shape = tuple(int(value) for value in x_node.attrs["shape"])
            nnz = int(x_node["indptr"][-1])
            data = x_node["data"]
            sample = np.asarray(data[: min(args.sample_values, data.shape[0])])
            print(f"X: group, encoding {encoding}, shape {shape}, stored values {nnz:,} ({nnz / shape[0]:,.0f} per cell), data dtype {data.dtype}, "
                  f"compression {data.compression}, indices dtype {x_node['indices'].dtype}")
            print(f"X: first {len(sample):,} stored values: min {sample.min()}, max {sample.max()}, all integer-valued {bool(np.all(sample == np.round(sample)))}, "
                  f"all positive {bool((sample > 0).all())}")
            if encoding != "csr_matrix":
                differences.append(f"X is {encoding}, not csr_matrix (h5ad_rows reads CSR row slices)")
            if not np.all(sample == np.round(sample)):
                differences.append("stored values are not integer-valued: SPORE-XL needs raw counts (the exact percentile histograms refuse non-integers)")
        else:
            shape = tuple(x_node.shape)
            differences.append(f"X is a dense dataset of shape {shape}: SPORE Phase 0 would convert it; plan this with the sparse cache instead")
            print(f"X: dense dataset, shape {shape}, dtype {x_node.dtype}")

    adata = __import__("anndata").read_h5ad(args.path, backed="r")
    print(f"shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    if args.expect_cells and adata.n_obs != args.expect_cells:
        differences.append(f"cells {adata.n_obs:,} != expected {args.expect_cells:,}")
    if args.expect_genes and adata.n_vars != args.expect_genes:
        differences.append(f"genes {adata.n_vars:,} != expected {args.expect_genes:,}")
    print("obs columns: " + ", ".join(f"{name} ({adata.obs[name].dtype})" for name in adata.obs.columns))
    print("var columns: " + ", ".join(f"{name} ({adata.var[name].dtype})" for name in adata.var.columns))
    print(f"var_names: first {list(adata.var_names[:3])}; Ensembl-style (start with ENSG): {bool(pd.Index(adata.var_names).str.startswith('ENSG').all())}")
    if "gene_name" not in adata.var.columns:
        differences.append("var has no 'gene_name' column (gene_id_format: ensembl needs it)")
    if args.pert_col not in adata.obs.columns:
        differences.append(f"obs has no perturbation column '{args.pert_col}'")
    else:
        column = adata.obs[args.pert_col]
        counts = column.astype(str).value_counts()
        n_control = int(counts.get(args.control, 0))
        print(f"obs['{args.pert_col}']: dtype {column.dtype}, {len(counts):,} distinct labels, control '{args.control}' cells {n_control:,}; "
              f"smallest group {int(counts.min())} cells, largest non-control {int(counts.drop(args.control, errors='ignore').max()):,}")
        print(f"   labels containing '+': {int(counts.index.str.contains('+', regex=False).sum())} (v1.0 refuses any); NaN labels: {int(column.isna().sum())}")
        if n_control == 0:
            differences.append(f"no cells labelled '{args.control}' in obs['{args.pert_col}']")
        if column.dtype.name != "category":
            print("   note: the perturbation column is not categorical; plan records the sorted unique labels as the category order")
    if args.batch_col not in adata.obs.columns:
        differences.append(f"obs has no batch column '{args.batch_col}' (only Phase 10, inert in XL, uses it)")
    else:
        print(f"obs['{args.batch_col}']: {adata.obs[args.batch_col].nunique()} distinct values")
    print("DIFFERENCES FROM THE EXPECTED FILE:" if differences else "No difference from the expected file.")
    for line in differences:
        print("  - " + line)
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
