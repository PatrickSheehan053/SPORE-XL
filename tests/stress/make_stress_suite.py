"""Stress-suite generators (Amendment 2, P11): 11 deliberately difficult synthetic datasets, 50-100k cells each, each with a truth table
(per cell) and an answer key (design facts + the outcomes a correct pipeline should give). Refuses to overwrite.

  python -B tests\\stress\\make_stress_suite.py --out data\\stress                (all)
  python -B tests\\stress\\make_stress_suite.py --out data\\stress --only 06_tiny_groups

Files per dataset <name>: <name>.h5ad, <name>_truth.parquet, <name>_key.json, <name>_protected.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from stress_core import CONTROL, build_gene_table, lognormal_sizes, make_obs, pert_table, simulate  # noqa: E402

SEED_BASE = 20261021


def pick_targets(rng, kind, mean_expr, count, low=0.8, high=40.0, exclude=()):
    pool = np.flatnonzero((kind == "plain") & (mean_expr >= low) & (mean_expr <= high))
    pool = np.setdiff1d(pool, np.asarray(list(exclude), dtype=np.int64))
    if count > len(pool):
        raise ValueError(f"only {len(pool)} target genes available for {count} perturbations")
    return rng.choice(pool, count, replace=False)


def protected_list(rng, names, kind, target_positions, count=12):
    pool = np.setdiff1d(np.flatnonzero(kind == "plain"), target_positions)
    genes = list(names[rng.choice(pool, count, replace=False)]) + ["RPL5", "TOP2A", "NOT_A_GENE_IN_PANEL"]
    return {"metadata": {"name": "stress list", "dataset_gene_id_format": "symbol"},
            "protected_genes": {"tier1": {"tier": 1, "genes": genes[:8]}, "tier2": {"tier": 2, "genes": genes[8:]}}}


def expectations(*pairs):
    return [{"id": key, "expectation": text} for key, text in pairs]


UNIVERSAL = expectations(
    ("completes", "all stages run to the end without an exception and COVERAGE_REPORT says GRAPH-READY"),
    ("verify", "VERIFICATION: every check PASS"),
    ("survivors", "the cells that reach the outputs equal an independent numpy derivation of SPORE Phases 2 and 5 (QC, escaper filter, undersized drop)"),
    ("obs_fidelity", "every output cell carries the label, batch and donor the generator gave it (no cell is mislabelled by sharding or block routing)"),
    ("x_fidelity", "hvgraw rows equal the raw rows restricted to the panel genes (300 sampled cells, exact)"),
    ("p9_sanity", "p9 rows are finite and expm1 of each row sums to target_sum (10000) wherever the raw panel row is non-empty"),
    ("mass_conservation", "per perturbation group, sum of n_cells_in_metacell equals the surviving cells; control units equal surviving controls"),
)


# ---------------------------------------------------------------------------------------------------- 01 genome-wide-like
def build_genomewide(rng):
    names, kind, mean_expr, low_positions = build_gene_table(rng, 3000, low_fraction=0.1)
    targets = pick_targets(rng, kind, mean_expr, 1000, low=0.3)
    perts = pert_table(names[targets], targets, lognormal_sizes(rng, 1000, 85, 0.3, 70), rng.uniform(0.2, 0.4, 1000))
    matrix, truth = simulate(rng, names, kind, mean_expr, perts, 4000, 2)
    key = {"n_top_genes": 300, "test_n": 200, "val_n": 100, "n_targets": 1000, "protected_targets_excluded": True}
    exp = expectations(
        ("panel_stacked", "policy all: more targets than panel slots, so the panel is stacked above n_top_genes (SPORE behaviour); under the production budget policy the panel equals n_top_genes exactly"),
        ("heldout_in_panel", "every surviving held-out target gene that passed Phase 6 is a node of the panel"))
    return matrix, truth, names, kind, key, exp, targets


# ---------------------------------------------------------------------------------------------------- 02 low-expressed targets
def build_lowexpr(rng):
    names, kind, mean_expr, low_positions = build_gene_table(rng, 2000, low_fraction=0.12)
    plain = np.flatnonzero(kind == "plain")
    high_pos = rng.choice(plain[(mean_expr[plain] > 0.5)], 100, replace=False)
    mean_expr[high_pos] = rng.uniform(8, 25, 100)
    rest = np.setdiff1d(plain, high_pos)
    mid_pos = rng.choice(rest[mean_expr[rest] > 0.03], 100, replace=False)
    mean_expr[mid_pos] = rng.uniform(0.6, 1.5, 100)
    low_pos = rng.choice(np.setdiff1d(low_positions, np.concatenate([high_pos, mid_pos])), 100, replace=False)
    targets = np.concatenate([high_pos, mid_pos, low_pos])
    cls = ["high"] * 100 + ["mid"] * 100 + ["low"] * 100
    perts = pert_table(names[targets], targets, lognormal_sizes(rng, 300, 160, 0.25, 110), np.full(300, 0.25), escaper_frac=0.20, cls=cls)
    matrix, truth = simulate(rng, names, kind, mean_expr, perts, 4000, 2)
    key = {"test_n": 50, "val_n": 25, "escaper_fraction": 0.20,
           "classes": {"high": sorted(names[high_pos]), "mid": sorted(names[mid_pos]), "low": sorted(names[low_pos])}}
    exp = expectations(
        ("escaper_removal_high", "designed escapers of HIGH-expressed targets (control p10 above 0) are removed at a rate of at least 70% of the QC-passing escapers"),
        ("escaper_removal_blind", "designed escapers of MID and LOW expressed targets (control p10 = 0, the blind spot) are removed at a rate of at least 50%; SPORE's rule cannot do that (KNOWN_ISSUES 5: design limitation, P4 reports the comparison)"))
    return matrix, truth, names, kind, key, exp, targets


# ---------------------------------------------------------------------------------------------------- 03 batch confounding
def build_confounded(rng):
    names, kind, mean_expr, low_positions = build_gene_table(rng, 2000)
    targets = pick_targets(rng, kind, mean_expr, 300)
    batch_sets = [[index % 4] if index < 150 else None for index in range(300)]
    cls = ["confounded" if index < 150 else "balanced" for index in range(300)]
    perts = pert_table(names[targets], targets, lognormal_sizes(rng, 300, 190, 0.3, 120), rng.uniform(0.2, 0.4, 300), batch_sets=batch_sets, cls=cls)
    matrix, truth = simulate(rng, names, kind, mean_expr, perts, 6000, 4, batch_sigma=0.5, n_batch_genes=300)
    key = {"test_n": 50, "val_n": 25, "confounded_labels": sorted(names[targets[:150]]), "n_batches": 4}
    exp = expectations(
        ("confounding_flagged", "the report contains a perturbation x batch confounding table and warns about the 150 perturbations whose cells all come from one batch (P5; absent in the 21 Oct code)"),
        ("batch_variance_reported", "the report states the variance explained by batch on the HVG subsample (P5)"))
    return matrix, truth, names, kind, key, exp, targets


# ---------------------------------------------------------------------------------------------------- 04 extreme depth variation
def build_depth(rng):
    names, kind, mean_expr, low_positions = build_gene_table(rng, 2000)
    targets = pick_targets(rng, kind, mean_expr, 300)
    mult = np.ones(300)
    mult[:30], mult[30:60] = 4.0, 0.25
    cls = ["deep"] * 30 + ["shallow"] * 30 + ["ordinary"] * 240
    perts = pert_table(names[targets], targets, lognormal_sizes(rng, 300, 200, 0.25, 130), rng.uniform(0.2, 0.4, 300), depth_mult=mult, cls=cls)
    matrix, truth = simulate(rng, names, kind, mean_expr, perts, 4000, 2, depth_sigma=1.0)
    key = {"test_n": 50, "val_n": 25, "deep_labels": sorted(names[targets[:30]]), "shallow_labels": sorted(names[targets[30:60]])}
    exp = expectations(
        ("depth_dropout", "shallow perturbations lose cells to the min-counts / min-genes filters and some fall under 50 cells (dropped as undersized); the independent derivation says exactly which"),)
    return matrix, truth, names, kind, key, exp, targets


# ---------------------------------------------------------------------------------------------------- 05 multiple donors
def build_donors(rng):
    names, kind, mean_expr, low_positions = build_gene_table(rng, 2000)
    targets = pick_targets(rng, kind, mean_expr, 300)
    donor_sets = [sorted(rng.choice(6, size=int(rng.integers(3, 7)), replace=False).tolist()) for _ in range(300)]
    perts = pert_table(names[targets], targets, lognormal_sizes(rng, 300, 190, 0.3, 120), rng.uniform(0.2, 0.4, 300), donor_sets=donor_sets)
    matrix, truth = simulate(rng, names, kind, mean_expr, perts, 4000, 4, n_donors=6, donor_sigma=0.25)
    key = {"test_n": 50, "val_n": 25, "n_donors": 6, "donor_column": "donor"}
    exp = expectations(
        ("donor_variance_reported", "the report states the variance explained by donor on the HVG subsample (P5; absent in the 21 Oct code)"),)
    return matrix, truth, names, kind, key, exp, targets


# ---------------------------------------------------------------------------------------------------- 06 tiny groups
def build_tiny(rng):
    names, kind, mean_expr, low_positions = build_gene_table(rng, 2000)
    targets = pick_targets(rng, kind, mean_expr, 310)
    sizes = np.concatenate([lognormal_sizes(rng, 120, 400, 0.2, 300), rng.integers(2, 11, 150), rng.integers(12, 46, 40)])
    cls = ["big"] * 120 + ["tiny"] * 150 + ["mid"] * 40
    perts = pert_table(names[targets], targets, sizes, rng.uniform(0.2, 0.4, 310), cls=cls)
    matrix, truth = simulate(rng, names, kind, mean_expr, perts, 4000, 2)
    key = {"test_n": 30, "val_n": 15, "tiny_labels": sorted(names[targets[120:270]]), "mid_labels": sorted(names[targets[270:]]),
           "big_labels": sorted(names[targets[:120]]),
           "arms": {"min50": {"overrides": {}}, "min2": {"overrides": {"phase5_escaper_filtering": {"min_cells_per_perturbation": 2}}}}}
    exp = expectations(
        ("tiny_dropped", "arm min50: all 150 tiny (2-10 cells) and 40 mid (12-45 cells) perturbations are dropped as undersized; only the 120 big ones reach the outputs"),
        ("tiny_kept_survive", "arm min2: groups of 2-10 cells go through metacell construction without error, each yields at least one unit, and their cells are conserved"))
    return matrix, truth, names, kind, key, exp, targets


# ---------------------------------------------------------------------------------------------------- 07 control-poor / control-heavy
def build_ctrl_poor(rng):
    names, kind, mean_expr, low_positions = build_gene_table(rng, 2000)
    targets = pick_targets(rng, kind, mean_expr, 300)
    perts = pert_table(names[targets], targets, lognormal_sizes(rng, 300, 200, 0.3, 130), rng.uniform(0.2, 0.4, 300))
    matrix, truth = simulate(rng, names, kind, mean_expr, perts, 40, 3, control_batches=[0], batch_sorted=True, qc_fail=(100, 100, 30))
    key = {"test_n": 50, "val_n": 25, "n_controls": 40, "controls_only_in_batch": "1", "batch_sorted": True}
    exp = expectations(
        ("few_controls_ok", "40 controls (all in one batch, row-sorted by batch so most shards have none) still give thresholds, splits and outputs equal to the independent derivation; a control-poor warning is not required by the blueprint"),)
    return matrix, truth, names, kind, key, exp, targets


def build_ctrl_heavy(rng):
    names, kind, mean_expr, low_positions = build_gene_table(rng, 2000)
    targets = pick_targets(rng, kind, mean_expr, 300)
    perts = pert_table(names[targets], targets, lognormal_sizes(rng, 300, 100, 0.2, 80), rng.uniform(0.2, 0.4, 300))
    matrix, truth = simulate(rng, names, kind, mean_expr, perts, 60000, 2)
    key = {"test_n": 50, "val_n": 25, "n_controls": 60000}
    exp = expectations(
        ("many_controls_ok", "60,000 controls (two thirds of the data) split into several control blocks; outputs equal to the independent derivation"),)
    return matrix, truth, names, kind, key, exp, targets


# ---------------------------------------------------------------------------------------------------- 08 messy labels
EXCEL_GENES = ["SEPT1", "MARCH1", "DEC1", "SEPT2", "MARCH2"]
EXCEL_DAMAGED = {"SEPT1": "1-Sep", "MARCH1": "1-Mar", "DEC1": "1-Dec", "SEPT2": "2-Sep", "MARCH2": "2-Mar"}


def build_messy(rng):
    names, kind, mean_expr, low_positions = build_gene_table(rng, 2000, extra_names=EXCEL_GENES)
    extra_pos = np.flatnonzero(kind == "extra")
    mean_expr[extra_pos] = rng.uniform(1.5, 4.0, len(extra_pos))
    ordinary = pick_targets(rng, kind, mean_expr, 295)
    targets = np.concatenate([extra_pos, ordinary])
    perts = pert_table(names[targets], targets, lognormal_sizes(rng, 300, 190, 0.3, 120), rng.uniform(0.2, 0.4, 300))
    matrix, truth = simulate(rng, names, kind, mean_expr, perts, 4000, 2)
    observed = truth["label"].to_numpy().copy()
    ordinary_labels = list(names[ordinary])
    lower, spaced, split_case = ordinary_labels[:6], ordinary_labels[6:12], ordinary_labels[12:15]
    for gene in EXCEL_GENES:
        observed[truth["label"].to_numpy() == gene] = EXCEL_DAMAGED[gene]
    for gene in lower:
        observed[truth["label"].to_numpy() == gene] = gene.lower()
    for gene in spaced:
        observed[truth["label"].to_numpy() == gene] = f" {gene} "
    for gene in split_case:
        rows = np.flatnonzero(truth["label"].to_numpy() == gene)
        observed[rows[: len(rows) // 2]] = gene.lower()
    control_rows = np.flatnonzero(truth["label"].to_numpy() == CONTROL)
    shuffled = rng.permutation(control_rows)
    cut_nt, cut_cap = int(0.4 * len(shuffled)), int(0.5 * len(shuffled))
    observed[shuffled[:cut_nt]] = "NT"
    observed[shuffled[cut_nt:cut_cap]] = "Non-Targeting"
    truth["obs_label"] = observed
    key = {"test_n": 50, "val_n": 25, "excel": EXCEL_DAMAGED, "lowercase": lower, "padded": spaced, "split_by_case": split_case,
           "control_variants": {"NT": cut_nt, "Non-Targeting": cut_cap - cut_nt, "non-targeting": len(shuffled) - cut_cap},
           "label_column": "obs_label"}
    exp = expectations(
        ("labels_not_silently_accepted", "Excel-damaged (1-Sep), lower-case, padded and split-by-case labels and the control variants NT / Non-Targeting are either repaired by the intake or refused; they must not run through the core as separate 'perturbations' (the control variants would become a 40% 'perturbation' with a split)"),)
    return matrix, truth, names, kind, key, exp, targets


# ---------------------------------------------------------------------------------------------------- 09 duplicate symbols after ID translation
def build_dup(rng):
    names, kind, mean_expr, low_positions = build_gene_table(rng, 2000)
    plain = np.flatnonzero(kind == "plain")
    chosen = rng.choice(plain[(mean_expr[plain] >= 0.8) & (mean_expr[plain] <= 40)], 100, replace=False)
    symbols = names.copy().astype(object)
    first_copies, second_copies = chosen[:20], chosen[20:40]
    for first, second in zip(first_copies, second_copies):  # the SECOND copy gets the first copy's symbol: pairs share a symbol
        symbols[second] = symbols[first]
    # duplicated symbols (20 pairs); targets: 10 pairs use the lower-index copy as the true target, 10 pairs the higher-index copy
    target_symbol_pairs = list(zip(first_copies, second_copies))
    true_copy = []
    for index, (first, second) in enumerate(target_symbol_pairs[:10]):
        true_copy.append(min(first, second) if index < 5 else max(first, second))
    ordinary = pick_targets(rng, kind, mean_expr, 290, exclude=np.concatenate([first_copies, second_copies]))
    label_symbols = [symbols[position] for position in true_copy] + list(names[ordinary])
    gene_pos = np.array(true_copy + list(ordinary))
    perts = pert_table(label_symbols, gene_pos, lognormal_sizes(rng, 300, 190, 0.3, 120), rng.uniform(0.2, 0.4, 300))
    matrix, truth = simulate(rng, names, kind, mean_expr, perts, 4000, 2)
    ensembl = np.array([f"ENSG{number + 100000:011d}" for number in range(len(names))])
    key = {"test_n": 50, "val_n": 25, "gene_id_format": "ensembl", "n_duplicate_symbol_pairs": 20, "duplicate_target_labels": sorted(map(str, label_symbols[:10])),
           "true_copy_is_first": [int(position == min(pair)) for position, pair in zip(true_copy, target_symbol_pairs[:10])],
           "ensembl_of_duplicate_pairs": [[str(ensembl[first]), str(ensembl[second])] for first, second in target_symbol_pairs],
           "label_column": "label"}
    exp = expectations(
        ("var_unique", "after ID translation the var names are unique and the run completes"),
        ("duplicates_flagged", "a target symbol that maps to two Ensembl IDs is reported as ambiguous (log / report) instead of silently using one copy (feature gap; the ID-translation step belongs to intake, P7)"))
    return matrix, truth, (ensembl, symbols), kind, key, exp, gene_pos


# ---------------------------------------------------------------------------------------------------- 10 dense input / 11 normalized input
def build_dense(rng):
    names, kind, mean_expr, low_positions = build_gene_table(rng, 1500)
    targets = pick_targets(rng, kind, mean_expr, 330)
    perts = pert_table(names[targets], targets, lognormal_sizes(rng, 330, 150, 0.3, 110), rng.uniform(0.2, 0.4, 330))
    matrix, truth = simulate(rng, names, kind, mean_expr, perts, 4000, 2)
    key = {"test_n": 50, "val_n": 25, "storage": "dense float32 array"}
    exp = expectations(("dense_equals_sparse", "the same counts stored dense give outputs identical to the sparse copy of the file"),)
    return matrix, truth, names, kind, key, exp, targets


def build_normalized(rng):
    names, kind, mean_expr, low_positions = build_gene_table(rng, 2000)
    targets = pick_targets(rng, kind, mean_expr, 300)
    perts = pert_table(names[targets], targets, lognormal_sizes(rng, 300, 170, 0.3, 120), rng.uniform(0.2, 0.4, 300))
    matrix, truth = simulate(rng, names, kind, mean_expr, perts, 4000, 2)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scaled = sp.diags(1e4 / np.maximum(totals, 1)) @ matrix
    scaled.data = np.log1p(scaled.data).astype(np.float32)
    key = {"test_n": 50, "val_n": 25, "storage": "log1p(CP10k) float32 CSR"}
    exp = expectations(("refused_before_writing", "plan refuses non-integer (normalized) input with a message that says the matrix must be raw counts, before any shard or manifest file is written"),)
    return scaled.tocsr().astype(np.float32), truth, names, kind, key, exp, targets


BUILDERS = {
    "01_genomewide": build_genomewide, "02_lowexpr": build_lowexpr, "03_confounded": build_confounded, "04_depth": build_depth,
    "05_donors": build_donors, "06_tiny_groups": build_tiny, "07a_ctrl_poor": build_ctrl_poor, "07b_ctrl_heavy": build_ctrl_heavy,
    "08_messy_labels": build_messy, "09_dup_symbols": build_dup, "10_dense": build_dense, "11_normalized": build_normalized,
}


def write_dataset(out_dir, name, index):
    rng = np.random.default_rng(SEED_BASE + index)
    matrix, truth, names, kind, key, exp, targets = BUILDERS[name](rng)
    paths = {suffix: out_dir / f"{name}{suffix}" for suffix in (".h5ad", "_truth.parquet", "_key.json", "_protected.json")}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if name == "09_dup_symbols":
        ensembl, symbols = names
        var = pd.DataFrame({"gene_name": symbols.astype(str)}, index=pd.Index(ensembl, name="gene_id"))
        gene_names = symbols.astype(str)
    else:
        var = pd.DataFrame(index=pd.Index(names, name="gene_symbol"))
        gene_names = names
    label_column = key.get("label_column", "label")
    obs = make_obs(truth, label_column=label_column, with_donor=(name == "05_donors"))
    if name == "10_dense":
        matrix = matrix.toarray().astype(np.float32)
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.write_h5ad(paths[".h5ad"], compression="lzf")
    truth.to_parquet(paths["_truth.parquet"])
    protected = protected_list(np.random.default_rng(SEED_BASE + 500 + index), gene_names, kind, targets)
    paths["_protected.json"].write_text(json.dumps(protected, indent=2), encoding="utf-8")
    nnz = int(sp.csr_matrix(matrix).nnz)
    key_full = {"name": name, "seed": SEED_BASE + index, "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars), "nnz": nnz,
                "n_perturbations": int(truth.loc[truth["label"] != CONTROL, "label"].nunique()),
                "n_controls": int((truth["label"] == CONTROL).sum()), "n_designed_escapers": int(truth["is_escaper"].sum()),
                "qc_failures": truth["qc_type"].value_counts().to_dict(), "perturbation_col": "gene", "control_label": CONTROL,
                **key, "expected_outcomes": UNIVERSAL + exp}
    paths["_key.json"].write_text(json.dumps(key_full, indent=2, default=str), encoding="utf-8")
    print(f"{name}: {adata.n_obs:,} cells x {adata.n_vars:,} genes, {key_full['n_perturbations']} perturbations, "
          f"{key_full['n_controls']:,} controls, nnz {nnz:,}, {paths['.h5ad'].stat().st_size / 1e6:.0f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default=None, help="one dataset name; default all")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(BUILDERS):
        if args.only and name != args.only:
            continue
        write_dataset(out_dir, name, index)


if __name__ == "__main__":
    main()
