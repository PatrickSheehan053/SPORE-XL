"""Synthetic 10x-Flex-style filtered_feature_bc_matrix.h5 members (h5py) and a tar holding them, for the intake / measure /
gate tests. Layout as in CellRanger: /matrix/{barcodes,data,indices,indptr,shape} and /matrix/features/{id,name,feature_type}.
The matrix is stored CSC over barcodes (indptr has n_cells + 1 entries; indices are feature rows), i.e. CSR over cells.

Guide library: GENE targets T001..T040 with two guides each (T001-1, T001-2), NTC-1..NTC-3, CTRL-1, and the Excel-damaged JUN
guides stored as the literal feature names "1-Jun" and "2-Jun". Cell scenarios cover every assignment case the real data has:
no guide, one strong guide, two guides of one target, weak guide only (below min_umi), two targets, three targets, JUN guide 1 only,
JUN guide 2 only, both JUN guides, a JUN guide plus another target, NTC only, CTRL only, NTC and CTRL together.
The ground truth is computed independently, cell by cell, in plain python (truth_for_member).
"""
import re
import sys
import tarfile
from pathlib import Path

sys.dont_write_bytecode = True

import h5py
import numpy as np

N_GENES = 200
N_TARGETS = 40
GENE_TYPE = "Gene Expression"
GUIDE_TYPE = "CRISPR Guide Capture"

SCENARIOS = ["none", "one", "one_both_guides", "weak_only", "two_targets", "three_targets", "jun1", "jun2", "jun_both",
             "jun_plus_other", "ntc", "ctrl", "ntc_ctrl"]
PROBABILITIES = [0.08, 0.40, 0.10, 0.05, 0.08, 0.02, 0.03, 0.03, 0.02, 0.02, 0.10, 0.04, 0.03]


def library():
    gene_names = [f"GENE{index:03d}" for index in range(N_GENES)]
    gene_ids = [f"ENSG{index:011d}" for index in range(N_GENES)]
    guides = []
    for index in range(1, N_TARGETS + 1):
        guides += [f"T{index:03d}-1", f"T{index:03d}-2"]
    guides += ["NTC-1", "NTC-2", "NTC-3", "CTRL-1", "1-Jun", "2-Jun"]
    return gene_ids, gene_names, guides


def _umi(rng, strong=True):
    return int(rng.integers(3, 12)) if strong else int(rng.integers(1, 3))


def write_flex_h5(path, n_cells, seed, barcode_prefix="AAAC", gene_order=None):
    """Writes one member. Returns {cell index: {guide name: UMI}} for the guide features (the independent truth input).
    gene_order: optional list of ORIGINAL gene indices in the order the member stores them (a permutation = same genes in another order;
    a shorter list = genes this member never measured). The cells and the truth are the same either way."""
    rng = np.random.default_rng(seed)
    gene_ids, gene_names, guides = library()
    n_features = N_GENES + len(guides)
    guide_row = {name: N_GENES + index for index, name in enumerate(guides)}
    scenario_of = rng.choice(len(SCENARIOS), size=n_cells, p=PROBABILITIES)
    truth = {}
    data, indices, indptr = [], [], [0]
    for cell in range(n_cells):
        scenario = SCENARIOS[scenario_of[cell]]
        picked = {}
        targets = [f"T{int(i):03d}" for i in rng.choice(np.arange(1, N_TARGETS + 1), size=3, replace=False)]
        if scenario == "one":
            picked[f"{targets[0]}-{int(rng.integers(1, 3))}"] = _umi(rng)
        elif scenario == "one_both_guides":
            picked[f"{targets[0]}-1"], picked[f"{targets[0]}-2"] = _umi(rng), _umi(rng)
        elif scenario == "weak_only":
            picked[f"{targets[0]}-1"] = _umi(rng, strong=False)
        elif scenario == "two_targets":
            picked[f"{targets[0]}-1"], picked[f"{targets[1]}-2"] = _umi(rng), _umi(rng)
        elif scenario == "three_targets":
            for target in targets:
                picked[f"{target}-1"] = _umi(rng)
        elif scenario == "jun1":
            picked["1-Jun"] = _umi(rng)
        elif scenario == "jun2":
            picked["2-Jun"] = _umi(rng)
        elif scenario == "jun_both":
            picked["1-Jun"], picked["2-Jun"] = _umi(rng), _umi(rng)
        elif scenario == "jun_plus_other":
            picked[["1-Jun", "2-Jun"][int(rng.integers(0, 2))]], picked[f"{targets[0]}-1"] = _umi(rng), _umi(rng)
        elif scenario == "ntc":
            picked[f"NTC-{int(rng.integers(1, 4))}"] = _umi(rng)
        elif scenario == "ctrl":
            picked["CTRL-1"] = _umi(rng)
        elif scenario == "ntc_ctrl":
            picked["NTC-1"], picked["CTRL-1"] = _umi(rng), _umi(rng)
        if rng.random() < 0.15:  # a stray weak guide below min_umi on top of the scenario (ambient guide RNA)
            stray = guides[int(rng.integers(0, len(guides)))]
            picked.setdefault(stray, _umi(rng, strong=False))
        gene_rows = np.flatnonzero(rng.random(N_GENES) < 0.35)
        rows = list(gene_rows) + [guide_row[name] for name in picked]
        values = list(rng.poisson(3, size=len(gene_rows)) + 1) + list(picked.values())
        order = np.argsort(rows)
        indices.extend(np.asarray(rows)[order].tolist())
        data.extend(np.asarray(values)[order].tolist())
        indptr.append(len(indices))
        truth[cell] = dict(picked)
    n_genes_out = N_GENES
    if gene_order is not None:  # a member with another gene order (a second probe-set batch) and/or fewer genes: positions = original gene indices
        gene_order = np.asarray(gene_order, dtype=np.int64)
        n_genes_out = len(gene_order)
        mapping = np.full(n_features, -1, dtype=np.int64)
        mapping[gene_order] = np.arange(n_genes_out)
        mapping[N_GENES:] = n_genes_out + np.arange(len(guides))
        counts = np.diff(np.asarray(indptr))
        cell_of = np.repeat(np.arange(n_cells), counts)
        new_index = mapping[np.asarray(indices, dtype=np.int64)]
        keep = new_index >= 0
        cell_kept, index_kept, data_kept = cell_of[keep], new_index[keep], np.asarray(data)[keep]
        order_kept = np.lexsort((index_kept, cell_kept))
        indices, data = index_kept[order_kept].tolist(), data_kept[order_kept].tolist()
        indptr = [0] + np.cumsum(np.bincount(cell_kept, minlength=n_cells)).tolist()
        gene_ids, gene_names = [gene_ids[i] for i in gene_order], [gene_names[i] for i in gene_order]
        n_features = n_genes_out + len(guides)
    barcodes = np.array([f"{barcode_prefix}{index:08d}-1".encode() for index in range(n_cells)])
    feature_type = np.array([GENE_TYPE.encode()] * n_genes_out + [GUIDE_TYPE.encode()] * len(guides))
    names = np.array([name.encode() for name in gene_names + guides])
    ids = np.array([value.encode() for value in gene_ids + guides])
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("barcodes", data=barcodes)
        group.create_dataset("data", data=np.asarray(data, dtype=np.int32), compression="gzip")
        group.create_dataset("indices", data=np.asarray(indices, dtype=np.int64), compression="gzip")
        group.create_dataset("indptr", data=np.asarray(indptr, dtype=np.int64))
        group.create_dataset("shape", data=np.array([n_features, n_cells], dtype=np.int32))
        features = group.create_group("features")
        features.create_dataset("id", data=ids)
        features.create_dataset("name", data=names)
        features.create_dataset("feature_type", data=feature_type)
        features.create_dataset("genome", data=np.array([b"GRCh38"] * n_features))
    return truth


def member_name(gsm, replicate, lane, donor, arm):
    return f"GSM{gsm}_CD4i_R{replicate}L{lane:02d}_D{donor}_{arm}_sample_filtered_feature_bc_matrix.h5"


def truth_for_member(truth, min_umi=3, target_pattern=r"-\d+$", fixes=None, controls=("NTC", "CTRL"), control_label="non-targeting"):
    """Plain-python assignment, independent of the vectorized implementation.
    Returns (per-cell label or None after normalization, per-cell raw target set, counters)."""
    fixes = fixes or {"1-Jun": "JUN", "2-Jun": "JUN"}
    labels, raw_sets = {}, {}
    for cell, picked in truth.items():
        raw = {re.sub(target_pattern, "", name) for name, umi in picked.items() if umi >= min_umi}
        raw_sets[cell] = raw
        normalized = set()
        for target in raw:
            target = fixes.get(target, target)
            target = control_label if target in controls else target
            normalized.add(target)
        labels[cell] = next(iter(normalized)) if len(normalized) == 1 else None
    return labels, raw_sets


def build_tar(directory, tar_path, members, seed0=100, gene_orders=None):
    """members: list of (name, n_cells). Writes the h5 files to `directory`, adds them to `tar_path`. Returns {name: truth}.
    gene_orders: optional {position in `members`: gene_order list} (see write_flex_h5)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    truths = {}
    with tarfile.open(tar_path, "w") as archive:
        for index, (name, n_cells) in enumerate(members):
            path = directory / name
            truths[name] = write_flex_h5(path, n_cells, seed0 + index, barcode_prefix=f"BC{index:02d}", gene_order=(gene_orders or {}).get(index))
            archive.add(path, arcname=name)
    return truths


if __name__ == "__main__":
    out = Path(sys.argv[1])
    members = [(member_name(9393800 + i, 1 + i % 2, 9 + i, 1 + i % 4, "Rest"), 2500) for i in range(3)]
    members.append((member_name(9393850, 1, 14, 1, "Stim8hr"), 1500))
    build_tar(out / "h5", out / "GSE_synthetic_RAW.tar", members)
    print(f"wrote {out / 'GSE_synthetic_RAW.tar'} with {len(members)} members")
