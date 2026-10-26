"""Check a SPORE run on synthetic_60k against (a) the KNOWN_ISSUES section 1 items and (b) an
independent re-derivation of Phases 2 and 5 straight from the raw file and the truth table.
Run: python -B tests\\check_baseline.py <run_dir>     (run_dir contains processed\\ and splits\\)
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import os
BUILD = Path(os.environ.get("XL_TEST_DATA", str(Path(__file__).resolve().parent.parent / "tests" / "ref")))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

NAME = "synth60k"
SOURCE = BUILD / "data" / "synthetic_60k.h5ad"
TRUTH = BUILD / "data" / "synthetic_60k_truth.parquet"
DESIGN = BUILD / "data" / "synthetic_60k_design.json"


def independent_expectation():
    """Phase 2 (QC thresholds, prefix MT-) and Phase 5 (10th percentile of controls, per target gene, then min 50)
    recomputed with plain numpy from the raw matrix. Returns the expected surviving cell names."""
    adata = ad.read_h5ad(SOURCE)
    matrix = sp.csr_matrix(adata.X)
    labels = adata.obs["gene"].astype(str).to_numpy()
    total = np.asarray(matrix.sum(axis=1)).ravel()
    n_genes = matrix.getnnz(axis=1)
    mt_columns = np.flatnonzero(adata.var_names.str.startswith("MT-"))
    percent_mt = 100 * np.asarray(matrix[:, mt_columns].sum(axis=1)).ravel() / total
    qc_pass = ((percent_mt <= 20) & (n_genes >= 200) & (n_genes <= 10000) &
               (total >= 500) & (total <= 80000))
    keep = qc_pass.copy()
    control = qc_pass & (labels == "non-targeting")
    control_rows = np.flatnonzero(control)
    column_of = {gene: index for index, gene in enumerate(adata.var_names)}
    csc = matrix.tocsc()
    for label in np.unique(labels[qc_pass]):
        if label == "non-targeting":
            continue
        rows = np.flatnonzero(qc_pass & (labels == label))
        column = column_of[label]
        threshold = np.percentile(csc[:, column][control_rows].toarray().ravel(), 10)
        values = csc[:, column][rows].toarray().ravel()
        keep[rows[values > threshold]] = False
    sizes = pd.Series(labels[keep]).value_counts()
    undersized = set(sizes[(sizes < 50) & (sizes.index != "non-targeting")].index)
    keep &= ~np.isin(labels, list(undersized))
    return adata.obs_names[keep], undersized, qc_pass


def main(run_dir):
    run_dir = Path(run_dir)
    splits, processed = run_dir / "splits", run_dir / "processed"
    failures = []

    report = (processed / "COVERAGE_REPORT.md").read_text(encoding="utf-8")
    print("COVERAGE_REPORT:", [line for line in report.splitlines() if "READY" in line])
    if "GRAPH-READY" not in report or "NOT READY" in report:
        failures.append("COVERAGE_REPORT is not GRAPH-READY")

    hybrid = ad.read_h5ad(processed / f"{NAME}_train_hybrid.h5ad")
    leaked = int(hybrid.obs["split"].astype(str).isin(["val", "test"]).sum())
    print(f"train_hybrid: {hybrid.n_obs} units, split values {sorted(hybrid.obs['split'].astype(str).unique())}, val/test units {leaked}")
    if leaked:
        failures.append(f"{leaked} val/test units in train_hybrid")

    raw = {split: ad.read_h5ad(splits / f"{NAME}_{split}_hvgraw.h5ad") for split in ("train", "val", "test")}
    kept_names = pd.Index(np.concatenate([raw[split].obs_names for split in raw]))
    assert kept_names.is_unique
    allsplits = ad.read_h5ad(processed / f"{NAME}_allsplits_metacell_ctrlpreserved.h5ad")
    control_cells = int(sum((raw[split].obs["gene"].astype(str) == "non-targeting").sum() for split in raw))
    control_units = allsplits.obs[allsplits.obs["gene"].astype(str) == "non-targeting"]
    print(f"Phase 12: {control_cells} control cells -> {len(control_units)} control units, "
          f"all single-cell: {bool((control_units['n_cells_in_metacell'] == 1).all())}")
    if len(control_units) != control_cells or not (control_units["n_cells_in_metacell"] == 1).all():
        failures.append("Phase 12 control units are not one per control cell")

    expected_names, expected_undersized, qc_pass = independent_expectation()
    truth = pd.read_parquet(TRUTH).set_index("cell")
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    print(f"independent re-derivation: expected survivors {len(expected_names)}; SPORE survivors {len(kept_names)}")
    only_expected = expected_names.difference(kept_names)
    only_spore = kept_names.difference(expected_names)
    print(f"  expected-but-missing {len(only_expected)}, present-but-unexpected {len(only_spore)}")
    if len(only_expected) or len(only_spore):
        failures.append("survivor set differs from the independent re-derivation")

    present = truth.loc[kept_names]
    print("truth table check:")
    print(f"  QC-failed cells present: {int((present.qc_type != 'ok').sum())} (of 340 designed)")
    print(f"  designed escapers surviving: {int(present.is_escaper.sum())} of {int(truth.is_escaper.sum())} "
          f"({100 * present.is_escaper.sum() / truth.is_escaper.sum():.1f}%)")
    survivors_by_label = present.label.value_counts()
    small = set(design["small_perturbations_under_50_cells"])
    print(f"  designed small (<50 cell) perturbations still present: {sorted(small & set(survivors_by_label.index))}")
    print(f"  perturbations dropped as undersized by the re-derivation: {len(expected_undersized)} "
          f"(designed small ones among them: {len(small & expected_undersized)} of {len(small)})")
    if (present.qc_type != "ok").any():
        failures.append("a designed QC-failure cell survived")
    if small & set(survivors_by_label.index):
        failures.append("a designed small perturbation survived")

    labels_out = set(survivors_by_label.index) - {"non-targeting"}
    split_json = json.loads((splits / f"{NAME}_split_indices.json").read_text(encoding="utf-8"))
    in_json = set(split_json["train_labels"]) | set(split_json["val_labels"]) | set(split_json["test_labels"])
    print(f"  labels in output {len(labels_out)}; labels in split JSON {len(in_json)}; equal: {labels_out == in_json}")
    if labels_out != in_json:
        failures.append("output labels differ from split JSON labels")

    print("PASS" if not failures else "FAIL: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
