"""Runs SPORE v3.0's own Phases 1, 2, 5, 6, 7 in-process on synthetic_60k and saves what SPORE itself
does not keep: Phase 2 cells, escaper thresholds (SPORE's literal lines), post-escaper cells, knockdown
records, post-undersized cells, Phase 6 gene mask, mean-shift scores, split labels and split cell sets.
These are the reference the XL stage tests compare against.

Run (fresh output dir each time):
  set PYTHONHASHSEED=0 ; python -B tests\\build_spore_reference.py --out intermediate\\spore_reference_data
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import os
REPO = Path(__file__).resolve().parent.parent
BUILD = Path(os.environ.get("XL_TEST_DATA", str(REPO / "tests" / "ref")))
sys.path.insert(0, str(REPO / "src" / "engine"))
import paths  # noqa: E402

paths.add_spore_to_path()

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import config_v3  # noqa: E402
from src.phases.phase01_detection import run_phase1  # noqa: E402
from src.phases.phase02_cell_triage import filter_cells  # noqa: E402
from src.phases.phase05_escaper_filtering import (_get_escaper_direction, _parse_target_genes,  # noqa: E402
                                            filter_escapers, filter_undersized_perturbations)
from src.phases.phase06_gene_triage import filter_genes  # noqa: E402
from src.phases.phase07_splits import split_zero_shot  # noqa: E402


def write_lines(path, items):
    Path(path).write_text("\n".join(map(str, items)) + "\n", encoding="utf-8")


def spore_thresholds(adata, cfg, logger):
    """phase05_escaper_filtering.py:63-87 and 156-179 reproduced literally, thresholds only."""
    p5 = cfg.get("phase5_escaper_filtering", {})
    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]
    percentile = p5.get("escaper_percentile", 10)
    direction = _get_escaper_direction(cfg, logger)
    pert_values = adata.obs[pert_col].values
    ctrl_indices = np.where(pert_values == ctrl_label)[0]
    gene_to_idx = {}
    for index, name in enumerate(adata.var_names):
        gene_to_idx[str(name)] = index
    for column in adata.var.columns:
        for index, value in enumerate(adata.var[column]):
            if pd.notna(value):
                gene_to_idx[str(value)] = index
    ctrl_csc = adata.X[ctrl_indices, :].tocsc()
    rows = []
    for target in [p for p in np.unique(pert_values) if p != ctrl_label]:
        for gene in [g for g in _parse_target_genes(target, cfg) if g in gene_to_idx]:
            ctrl_expr = ctrl_csc[:, gene_to_idx[gene]].toarray().flatten()
            if direction == "knockdown":
                threshold = np.percentile(ctrl_expr, percentile)
            else:
                threshold = np.percentile(ctrl_expr, 100 - percentile)
                if threshold == 0:
                    threshold = 1e-5
            rows.append({"gene": gene, "gene_index": gene_to_idx[gene], "threshold": repr(float(threshold)),
                         "threshold_dtype": str(np.asarray(threshold).dtype), "n_ctrl": len(ctrl_expr),
                         "ctrl_mean": repr(float(ctrl_expr.mean()))})
    return pd.DataFrame(rows).drop_duplicates("gene").sort_values("gene")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(BUILD / "data" / "spore_config_synthetic60k_reference_v1.yaml"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; use a fresh directory")
    out.mkdir(parents=True, exist_ok=True)

    cfg = config_v3.load_config(args.config)
    logger = logging.getLogger("spore_reference")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)

    adata = ad.read_h5ad(cfg["paths"]["_raw_h5ad"])
    adata, detection = run_phase1(adata, cfg, logger)
    adata, wf2 = filter_cells(adata, cfg, logger)
    write_lines(out / "p2_cells.txt", adata.obs_names)
    write_lines(out / "p2_var.txt", adata.var_names)
    adata.var.to_csv(out / "p2_var_table.tsv", sep="\t")
    adata.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].to_parquet(out / "p2_obs_qc.parquet")
    json.dump({k: int(v) for k, v in wf2.items()}, open(out / "p2_waterfall.json", "w"), indent=2)

    thresholds = spore_thresholds(adata, cfg, logger)
    thresholds.to_csv(out / "p5_thresholds.tsv", sep="\t", index=False)

    adata, escaper_stats = filter_escapers(adata, cfg, logger)
    write_lines(out / "p5_escaper_kept_cells.txt", adata.obs_names)
    escaper_stats.to_csv(out / "p5_escaper_stats.tsv", sep="\t", index=False)
    knock = pd.DataFrame.from_dict(adata.uns["knockdown_efficiency"], orient="index")
    knock.index.name = "label"
    knock.assign(knockdown_depth=knock["knockdown_depth"].map(repr)).to_csv(out / "p5_knockdown.tsv", sep="\t")
    adata.obs[["escaper_efficiency", "escaper_knockdown_depth"]].to_parquet(out / "p5_obs_eff.parquet")

    adata, pert_sizes = filter_undersized_perturbations(adata, cfg, logger)
    write_lines(out / "p5b_cells.txt", adata.obs_names)

    adata, wf6, rescued = filter_genes(adata, cfg, logger)
    write_lines(out / "p6_var.txt", adata.var_names)
    write_lines(out / "p6_rescued.txt", rescued)
    np.save(out / "p6_penetrance.npy", adata.uns["phase6_gene_penetrance"])
    json.dump({k: int(v) for k, v in wf6.items()}, open(out / "p6_waterfall.json", "w"), indent=2)

    result = split_zero_shot(adata, cfg, logger, seed=cfg["phase7_splits"]["random_seed"])
    result["deg_counts"].rename("score").map(repr).to_frame().to_csv(out / "p7_mean_shift.tsv", sep="\t", index_label="label")
    json.dump({"train_labels": list(result["train_labels"]), "val_labels": list(result["val_labels"]),
               "test_labels": list(result["test_labels"]), "seed": int(result["seed"])},
              open(out / "p7_labels.json", "w"), indent=2)
    for split in ("train", "val", "test"):
        write_lines(out / f"p7_{split}_cells.txt", result[split].obs_names)
    print(f"reference written to {out}: P2 {len(open(out / 'p2_cells.txt').read().split())} cells, "
          f"{len(thresholds)} target genes, P5b {len(open(out / 'p5b_cells.txt').read().split())} cells, "
          f"P6 {len(open(out / 'p6_var.txt').read().split())} genes")


if __name__ == "__main__":
    main()
