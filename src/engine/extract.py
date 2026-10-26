"""extract: nested cell-count ladder and gene micro-panels from a finished XL run (blueprint component 9).

Every surviving cell gets a FIXED priority u = splitmix64(xl_global_index, seed) / 2**64 that never changes, so rung c is a
subset of rung 2c by construction. Rung c keeps the c lowest-priority cells of each perturbation (all of them when a
perturbation has fewer than c cells; those are listed in the provenance). Controls follow the dataset's control fraction PER
SPLIT: the number of controls kept in a split is round(pert cells kept in that split x (controls / pert cells of that split in
the full run)), taking the lowest-priority controls of that split. The split of every cell is the run's fixed split; extract
never re-splits. Output: <out>/<name>.h5ad (raw counts of the requested genes, in the requested order) and
<out>/<name>.provenance.json (no timestamps, so repeated extractions are byte-identical).

  python -B src\\extract.py --config run.yaml --genes genes.txt --cells-per-pert 16 --seed 7 --out DIR [--perts perts.txt]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from xl_common import Run, block_unit, environment_text, require_done
from stage_gather import total_blocks

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def hash64(index, seed):
    """splitmix64 of (index, seed); vectorized, platform independent (unsigned 64-bit wraparound)."""
    with np.errstate(over="ignore"):
        z = np.asarray(index, dtype=np.uint64) + (np.uint64(seed) + np.uint64(1)) * GOLDEN
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def priority(index, seed):
    return hash64(index, seed).astype(np.float64) / 18446744073709551616.0  # 2**64


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_cells(final, cells_per_pert, seed, perturbations=None, ctrl_label="non-targeting", control_policy="match_split_share"):
    """final: global_obs_final rows of the SURVIVING cells (xl_global_index, label, split). Returns (selected frame, short labels, summary)."""
    survivors = final.copy()
    survivors["label"] = survivors["label"].astype(str)
    survivors["split"] = survivors["split"].astype(str)
    survivors["priority"] = priority(survivors["xl_global_index"].to_numpy(), seed)
    is_control = survivors["label"] == ctrl_label
    perts = survivors[~is_control]
    if perturbations is not None:
        unknown = sorted(set(perturbations) - set(perts["label"]))
        if unknown:
            raise ValueError(f"requested perturbations are not in the run's surviving set: {unknown[:5]}")
        perts = perts[perts["label"].isin(perturbations)]
    ordered = perts.sort_values(["label", "priority", "xl_global_index"], kind="stable")
    rank = ordered.groupby("label", sort=False).cumcount()
    chosen = ordered[rank < cells_per_pert]
    sizes = perts["label"].value_counts()
    short = sorted(sizes[sizes < cells_per_pert].index)

    controls_all = survivors[is_control]
    pert_all = survivors[~is_control]
    selected_controls = []
    per_split = {}
    for split in ("train", "val", "test"):
        full_pert = int((pert_all["split"] == split).sum())
        full_ctrl = int((controls_all["split"] == split).sum())
        kept_pert = int((chosen["split"] == split).sum())
        if control_policy == "none" or full_pert == 0:
            n_ctrl = 0
        else:
            n_ctrl = min(full_ctrl, int(round(kept_pert * full_ctrl / full_pert)))
        pool = controls_all[controls_all["split"] == split].sort_values(["priority", "xl_global_index"], kind="stable")
        selected_controls.append(pool.head(n_ctrl))
        per_split[split] = {"pert_cells": kept_pert, "control_cells": int(n_ctrl), "dataset_control_ratio": (full_ctrl / full_pert) if full_pert else None}
    selected = pd.concat([chosen] + selected_controls).sort_values("xl_global_index", kind="stable").reset_index(drop=True)
    return selected, short, per_split


def extract(run: Run, genes, cells_per_pert, seed, out_dir, name=None, perturbations=None, control_policy="match_split_share"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"{run.name}_c{cells_per_pert}_seed{seed}"
    h5ad_path, provenance_path = out_dir / f"{name}.h5ad", out_dir / f"{name}.provenance.json"
    for path in (h5ad_path, provenance_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    final = pd.read_parquet(run.manifest_dir / "global_obs_final.parquet")
    final = final[final["survived_to_phase7"]][["xl_global_index", "label", "split"]]
    selected, short, per_split = select_cells(final, cells_per_pert, seed, perturbations, run.ctrl_label, control_policy)
    wanted = selected["xl_global_index"].to_numpy()

    # ---- raw counts of the requested genes from the blocks (Phase 6 space before the gene mask)
    pieces, indices = [], []
    gene_columns = None
    for block in range(total_blocks(run)):
        path = require_done(run, "work", "gather", block_unit(block)) / "block.h5ad"
        if not path.exists():
            continue
        block_adata = ad.read_h5ad(path)
        if gene_columns is None:
            position = {gene: index for index, gene in enumerate(block_adata.var_names)}
            missing = [gene for gene in genes if gene not in position]
            if missing:
                raise ValueError(f"{len(missing)} requested genes are not in the run's gene space (examples: {missing[:5]})")
            gene_columns = np.array([position[gene] for gene in genes])
        gidx = block_adata.obs["xl_global_index"].to_numpy()
        rows = np.flatnonzero(np.isin(gidx, wanted))
        if len(rows):
            pieces.append(sp.csr_matrix(block_adata.X[rows])[:, gene_columns])
            indices.append(gidx[rows])
    matrix = sp.vstack(pieces, format="csr")
    fetched = np.concatenate(indices)
    order = np.argsort(fetched, kind="stable")
    matrix, fetched = matrix[order], fetched[order]
    if not np.array_equal(fetched, wanted):
        raise RuntimeError("cells fetched from the blocks do not match the selection")

    obs = pd.DataFrame({
        "gene": pd.Categorical(selected["label"].to_numpy()),
        "split": selected["split"].to_numpy(),
        "xl_global_index": wanted.astype(np.int64),
        "priority": selected["priority"].to_numpy(),
        "is_control": (selected["label"] == run.ctrl_label).to_numpy(),
    }, index=pd.Index([f"cell_{index}" for index in wanted], name="cell"))
    adata = ad.AnnData(X=matrix.astype(np.float32), obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    adata.write_h5ad(h5ad_path)

    split_json = run.splits_dir / f"{run.name}_split_indices.json"
    provenance = {
        "tool": "spore_xl extract (blueprint component 9)",
        "run_dir": str(run.run_dir), "run_name": run.name,
        "priority_function": "splitmix64(xl_global_index, seed) / 2**64; rung c = the c lowest-priority cells per perturbation",
        "seed": int(seed), "cells_per_perturbation": int(cells_per_pert), "control_policy": control_policy,
        "genes": {"n": len(genes), "sha256_of_ordered_list": _sha256_text("\n".join(genes)), "first": list(genes[:5])},
        "perturbations": "all surviving" if perturbations is None else {"n": len(perturbations), "sha256": _sha256_text("\n".join(sorted(perturbations)))},
        "n_cells": int(adata.n_obs), "n_perturbations": int(obs.loc[~obs["is_control"], "gene"].nunique()),
        "per_split": per_split, "perturbations_with_fewer_cells_than_rung": short,
        "split_json_sha256": _file_sha256(split_json),
        "control_assignment_sha256": _file_sha256(run.splits_dir / f"{run.name}_control_assignment.parquet"),
        "global_obs_final_sha256": _file_sha256(run.manifest_dir / "global_obs_final.parquet"),
        "xl_environment": (run.config_dir / "environment.txt").read_text(encoding="utf-8").strip().split("\n"),
        "x_dtype": "float32 raw counts", "gene_space": "Phase 6 input space (ribosomal genes already dropped, gene mask not applied)",
        "output_h5ad": h5ad_path.name,
        "output_x_sha256": hashlib.sha256(sp.csr_matrix(adata.X).data.tobytes() + sp.csr_matrix(adata.X).indices.tobytes()
                                          + sp.csr_matrix(adata.X).indptr.tobytes()).hexdigest(),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return h5ad_path, provenance_path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="extract.py")
    parser.add_argument("--config", required=True)
    parser.add_argument("--genes", required=True, help="text file, one gene per line")
    parser.add_argument("--cells-per-pert", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--perts", default=None, help="text file, one perturbation per line (default: all surviving)")
    parser.add_argument("--control-policy", default="match_split_share", choices=["match_split_share", "none"])
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    genes = [line.strip() for line in Path(args.genes).read_text(encoding="utf-8").split("\n") if line.strip()]
    perts = [line.strip() for line in Path(args.perts).read_text(encoding="utf-8").split("\n") if line.strip()] if args.perts else None
    paths = extract(Run(args.config), genes, args.cells_per_pert, args.seed, args.out, args.name, perts, args.control_policy)
    print("\n".join(map(str, paths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
