"""Intake for 10x Flex filtered_feature_bc_matrix.h5 members (blueprint component 1): reads one member DIRECTLY from a tar
(tarfile.extractfile + h5py, nothing is extracted to disk) or from a folder of .h5 files, assigns the target of every cell with
the verified tally rule, normalizes labels BEFORE counting targets, and writes one SPORE-ready shard (.h5ad, raw counts, gene
features only) plus the counters needed to reconcile with the verified tally.

Pattern proven in thesis\\hpc\\perturbation_library\\scripts\\assign_guides_h5tar.py (same vectorized distinct-(cell, target) logic).
Rules (recipe src/intake/recipes/GSE314342_CD4T-Rest.yaml):
  * features split by feature_type: "Gene Expression" rows become the matrix, guide-capture rows only decide the target
  * a guide counts at >= min_umi UMI; target = guide name minus the -N suffix; 1-Jun / 2-Jun -> JUN and NTC / CTRL -> control_label
    are applied BEFORE counting the distinct targets of a cell (two guides of one target count once)
  * a cell is WRITTEN only with exactly one (normalized) target; cells with 0 or 2+ targets are counted, not written
  * obs: gene (categorical, SORTED categories = every normalized label of the guide library plus the control label), sample (GSM),
    replicate, lane, donor, arm (parsed from the member name), gem_group (R<rep>L<lane>), guide_umi_max, n_guides_assigned
  * the tally-comparable RAW counts (no label normalization) are written per member so the result can be checked against the tally
"""
import hashlib
import json
import re
import tarfile
from contextlib import contextmanager
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml


def load_recipe(path):
    recipe = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("member_match", "member_suffix", "member_name_regex", "gene_feature_type", "guide_feature_regex", "min_umi",
                "target_pattern", "label_fixes", "control_targets", "control_label", "perturbation_col"):
        if key not in recipe:
            raise ValueError(f"recipe {path} lacks '{key}'")
    return recipe


def list_members(source, recipe):
    """[(member name, size)] sorted by name. `source` is a .tar or a folder of .h5 files."""
    source = Path(source)
    match, suffix = recipe["member_match"], recipe["member_suffix"]
    if source.is_dir():
        found = [(path.name, path.stat().st_size) for path in source.iterdir() if path.is_file() and path.name.endswith(suffix) and match in path.name]
    else:
        with tarfile.open(source) as archive:
            found = [(member.name, member.size) for member in archive if member.isfile() and member.name.endswith(suffix) and match in member.name]
    return sorted(found)


@contextmanager
def open_member(source, name):
    source = Path(source)
    if source.is_dir():
        with h5py.File(source / name, "r") as handle:
            yield handle
        return
    with tarfile.open(source) as archive:
        member = archive.getmember(name)
        with archive.extractfile(member) as raw, h5py.File(raw, "r") as handle:
            yield handle


def parse_member_name(name, recipe):
    found = re.match(recipe["member_name_regex"], Path(name).name)
    if not found:
        raise ValueError(f"member name does not match the recipe pattern: {name}")
    fields = found.groupdict()
    return {"sample": fields["gsm"], "replicate": int(fields["replicate"]), "lane": int(fields["lane"]),
            "donor": int(fields["donor"]), "arm": fields["arm"]}


def _decode(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _digest(items):
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def guide_library(feature_names, feature_types, recipe):
    """slot of each feature (-1 for non-guides), raw target per guide slot, normalized label per raw target."""
    guide_regex = re.compile(recipe["guide_feature_regex"], re.I)
    guide_rows = [i for i, kind in enumerate(feature_types) if guide_regex.search(kind)]
    if not guide_rows:
        raise ValueError("no guide-capture features in this member")
    raw_targets = [re.sub(recipe["target_pattern"], "", feature_names[i]) for i in guide_rows]
    raw_names = sorted(set(raw_targets))
    raw_index = {name: k for k, name in enumerate(raw_names)}
    raw_of_slot = np.array([raw_index[name] for name in raw_targets], dtype=np.int64)
    fixes, controls, control_label = recipe["label_fixes"], set(recipe["control_targets"]), recipe["control_label"]

    def normalize(target):
        target = fixes.get(target, target)
        return control_label if target in controls else target

    normalized = [normalize(name) for name in raw_names]
    labels = sorted(set(normalized) | {control_label})
    label_index = {name: k for k, name in enumerate(labels)}
    norm_of_raw = np.array([label_index[name] for name in normalized], dtype=np.int64)
    slot_of_feature = np.full(len(feature_names), -1, dtype=np.int32)
    slot_of_feature[guide_rows] = np.arange(len(guide_rows), dtype=np.int32)
    return {"slot_of_feature": slot_of_feature, "raw_of_slot": raw_of_slot, "raw_names": raw_names, "norm_of_raw": norm_of_raw,
            "labels": labels, "n_guides": len(guide_rows)}


def _merge_groups(recipe, raw_names):
    """Groups of RAW targets that normalization merges into one label (>= 2 members), for the reconciliation counters."""
    groups = {}
    for raw in raw_names:
        target = recipe["label_fixes"].get(raw, raw)
        target = recipe["control_label"] if target in set(recipe["control_targets"]) else target
        groups.setdefault(target, []).append(raw)
    return {label: members for label, members in groups.items() if len(members) >= 2}


def var_names_for(gene_names, gene_ids):
    """var index of a shard: the gene name, or name|id when a name occurs more than once."""
    duplicated = pd.Series(gene_names).duplicated(keep=False).to_numpy()
    return [f"{name}|{identifier}" if dup else name for name, identifier, dup in zip(gene_names, gene_ids, duplicated)]


def gene_features(handle, recipe):
    """(ids, names) of the gene-expression features of one member, in the member's own order."""
    features = handle["matrix"]["features"]
    kinds = _decode(features["feature_type"][...])
    ids, names = _decode(features["id"][...]), _decode(features["name"][...])
    rows = [i for i, kind in enumerate(kinds) if kind == recipe["gene_feature_type"]]
    if not rows:
        raise ValueError(f"no '{recipe['gene_feature_type']}' features")
    return [ids[i] for i in rows], [names[i] for i in rows]


def order_status(member_ids, canonical_ids):
    """How a member's gene list relates to the run's canonical list (which is always a subset of it)."""
    if member_ids == canonical_ids:
        return "same_as_canonical"
    wanted = set(canonical_ids)
    kept = [identifier for identifier in member_ids if identifier in wanted]
    if len(kept) == len(member_ids):
        return "reordered"
    return "dropped_genes" if kept == canonical_ids else "reordered_and_dropped_genes"


def plan_gene_harmonization(member_genes, max_dropped_fraction=0.02):
    """ONE gene list for the whole run. member_genes: {member name: (ids, names)} in sorted member order.
    Same gene SET in every member (only the order differs): canonical order = the first member's order (lossless reorder of the others).
    Different sets: the INTERSECTION, in the first member's order (a union would invent zeros for genes a sample never measured and bias QC,
    HVG and co-expression); refused if it drops more than `max_dropped_fraction` of the union (recipe key max_dropped_gene_fraction).
    Returns (canonical dict {policy, ids, names, reference_member, n_union, n_dropped, dropped_fraction}, per-member report DataFrame,
    dropped-genes DataFrame, refusal message or None)."""
    members = list(member_genes)
    first = members[0]
    for name in members:
        ids = member_genes[name][0]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name}: gene ids are not unique ({len(ids) - len(set(ids))} repeated): cannot harmonize")
    sets = {name: set(member_genes[name][0]) for name in members}
    union, common = set.union(*sets.values()), set.intersection(*sets.values())
    policy = "reorder" if common == union else "intersection"
    first_ids, first_names = member_genes[first]
    keep = [(identifier, label) for identifier, label in zip(first_ids, first_names) if identifier in common]
    canonical = {"policy": policy, "ids": [identifier for identifier, _ in keep], "names": [label for _, label in keep], "reference_member": first,
                 "n_union": len(union), "n_dropped": len(union) - len(common), "dropped_fraction": (len(union) - len(common)) / max(len(union), 1)}
    rows = []
    for name in members:
        ids = member_genes[name][0]
        rows.append({"member": name, "n_genes": len(ids), "n_genes_written": len(canonical["ids"]), "n_dropped_here": len(ids) - len(canonical["ids"]),
                     "order_sha256": _digest(ids)[:8], "sorted_sha256": _digest(sorted(ids))[:8], "status": order_status(ids, canonical["ids"])})
    dropped_rows = []
    for identifier in sorted(union - common):
        present = [name for name in members if identifier in sets[name]]
        absent = [name for name in members if identifier not in sets[name]]
        label = next(member_genes[name][1][member_genes[name][0].index(identifier)] for name in present)
        dropped_rows.append({"gene_id": identifier, "name": label, "n_members_with": len(present), "n_members_without": len(absent),
                             "members_with": ";".join(present), "members_without": ";".join(absent)})
    refusal = None
    if canonical["dropped_fraction"] > max_dropped_fraction:
        refusal = (f"the members do not share one gene set: the intersection would drop {canonical['n_dropped']:,} of {canonical['n_union']:,} genes "
                   f"({canonical['dropped_fraction']:.1%} > the allowance of {max_dropped_fraction:.1%}; recipe key max_dropped_gene_fraction)")
    return canonical, pd.DataFrame(rows), pd.DataFrame(dropped_rows, columns=["gene_id", "name", "n_members_with", "n_members_without", "members_with", "members_without"]), refusal


def intake_member(handle, member_name, recipe, out_path, canonical=None):
    """Writes the shard (.h5ad) and returns (meta dict, raw-target counts DataFrame, label counts DataFrame).
    canonical: the run's one gene list from plan_gene_harmonization ({ids, names}); the shard is written already in that order (and without
    the genes outside it), so every shard of a run has the same var. None = the member's own genes in its own order (old behaviour)."""
    group = handle["matrix"]
    feature_type = _decode(group["features"]["feature_type"][...])
    feature_name = _decode(group["features"]["name"][...])
    feature_id = _decode(group["features"]["id"][...])
    gene_rows = np.array([i for i, kind in enumerate(feature_type) if kind == recipe["gene_feature_type"]], dtype=np.int64)
    if len(gene_rows) == 0:
        raise ValueError(f"no '{recipe['gene_feature_type']}' features")
    n_genes_in_member = len(gene_rows)
    gene_order_status = "member_order"
    if canonical is not None:
        row_of_id = {feature_id[i]: int(i) for i in gene_rows}
        if len(row_of_id) != n_genes_in_member:
            raise ValueError(f"{member_name}: gene ids are not unique")
        missing = [identifier for identifier in canonical["ids"] if identifier not in row_of_id]
        if missing:
            raise ValueError(f"{member_name} lacks {len(missing)} genes of the run's gene list (e.g. {missing[:3]}): run `intake_list` again on this tar")
        ordered = np.array([row_of_id[identifier] for identifier in canonical["ids"]], dtype=np.int64)
        gene_order_status = order_status([feature_id[i] for i in gene_rows], canonical["ids"])
        gene_rows = ordered
    library = guide_library(feature_name, feature_type, recipe)
    raw_names, labels, control_label = library["raw_names"], library["labels"], recipe["control_label"]
    n_raw, n_norm = len(raw_names), len(labels)
    merge_groups = _merge_groups(recipe, raw_names)
    group_raw_index = {label: np.array([raw_names.index(m) for m in members]) for label, members in merge_groups.items()}

    indptr = group["indptr"][...].astype(np.int64)
    n_cells = len(indptr) - 1
    n_features = len(feature_name)
    barcodes = _decode(group["barcodes"][...])
    min_umi = int(recipe["min_umi"])
    chunk_cells = int(recipe.get("chunk_cells", 20000))

    raw_one_target = np.zeros(n_raw, dtype=np.int64)
    raw_any = np.zeros(n_raw, dtype=np.int64)
    raw_distinct_hist = np.zeros(4, dtype=np.int64)  # 0, 1, 2, 3+ raw targets
    pair_counts = {label: 0 for label in merge_groups}
    cells_with_any_guide = 0
    written_label, written_cell, written_umi_max, written_n_guides = [], [], [], []
    n_cells_norm_hist = np.zeros(4, dtype=np.int64)
    gene_chunks = []

    for start in range(0, n_cells, chunk_cells):
        stop = min(start + chunk_cells, n_cells)
        count = stop - start
        first, last = int(indptr[start]), int(indptr[stop])
        indices = group["indices"][first:last]
        data = group["data"][first:last]
        local_indptr = indptr[start:stop + 1] - first
        entry_cell = np.repeat(np.arange(count, dtype=np.int64), np.diff(local_indptr))
        slot = library["slot_of_feature"][indices]
        is_guide = slot >= 0
        cells_with_any_guide += len(np.unique(entry_cell[is_guide]))
        strong = is_guide & (data >= min_umi)
        cells_s, slots_s, umi_s = entry_cell[strong], slot[strong].astype(np.int64), data[strong].astype(np.int64)

        raw_target = library["raw_of_slot"][slots_s]
        raw_pairs = np.unique(cells_s * n_raw + raw_target)
        raw_pair_cell, raw_pair_target = raw_pairs // n_raw, raw_pairs % n_raw
        raw_per_cell = np.bincount(raw_pair_cell, minlength=count)
        raw_distinct_hist += np.bincount(np.minimum(raw_per_cell, 3), minlength=4)
        np.add.at(raw_any, raw_pair_target, 1)
        single_raw = raw_per_cell[raw_pair_cell] == 1
        np.add.at(raw_one_target, raw_pair_target[single_raw], 1)
        for label, members in group_raw_index.items():  # cells whose raw target set is EXACTLY the merged group
            in_group = np.isin(raw_pair_target, members)
            per_cell_in_group = np.bincount(raw_pair_cell[in_group], minlength=count)
            pair_counts[label] += int(((per_cell_in_group == len(members)) & (raw_per_cell == len(members))).sum())

        norm_target = library["norm_of_raw"][raw_target]
        norm_pairs = np.unique(cells_s * n_norm + norm_target)
        norm_pair_cell, norm_pair_label = norm_pairs // n_norm, norm_pairs % n_norm
        norm_per_cell = np.bincount(norm_pair_cell, minlength=count)
        n_cells_norm_hist += np.bincount(np.minimum(norm_per_cell, 3), minlength=4)
        label_of_cell = np.full(count, -1, dtype=np.int64)
        label_of_cell[norm_pair_cell] = norm_pair_label  # only read where norm_per_cell == 1
        rows = np.flatnonzero(norm_per_cell == 1)
        if len(rows):
            umi_max = np.zeros(count, dtype=np.int64)
            np.maximum.at(umi_max, cells_s, umi_s)
            n_guides = np.bincount(cells_s, minlength=count)
            chunk = sp.csr_matrix((data, indices, local_indptr), shape=(count, n_features))
            gene_chunks.append(chunk[rows][:, gene_rows].astype(np.float32))
            written_cell.append(rows + start)
            written_label.append(label_of_cell[rows])
            written_umi_max.append(umi_max[rows])
            written_n_guides.append(n_guides[rows])

    cell_index = np.concatenate(written_cell) if written_cell else np.zeros(0, dtype=np.int64)
    label_index = np.concatenate(written_label) if written_label else np.zeros(0, dtype=np.int64)
    matrix = sp.vstack(gene_chunks, format="csr") if gene_chunks else sp.csr_matrix((0, len(gene_rows)), dtype=np.float32)
    design = parse_member_name(member_name, recipe)
    gene_names = list(canonical["names"]) if canonical is not None else [feature_name[i] for i in gene_rows]
    gene_ids = list(canonical["ids"]) if canonical is not None else [feature_id[i] for i in gene_rows]
    var_names = var_names_for(gene_names, gene_ids)
    var = pd.DataFrame({"gene_ids": gene_ids, "feature_types": recipe["gene_feature_type"]}, index=pd.Index(var_names, name=None))

    obs = pd.DataFrame({
        recipe["perturbation_col"]: pd.Categorical([labels[i] for i in label_index], categories=labels),  # labels is sorted
        "sample": design["sample"], "replicate": design["replicate"], "lane": design["lane"], "donor": design["donor"],
        "arm": design["arm"], "gem_group": f"R{design['replicate']}L{design['lane']}",
        "guide_umi_max": np.concatenate(written_umi_max) if written_umi_max else np.zeros(0, dtype=np.int64),
        "n_guides_assigned": np.concatenate(written_n_guides) if written_n_guides else np.zeros(0, dtype=np.int64),
    }, index=pd.Index([f"{design['sample']}_{barcodes[i]}" for i in cell_index], name="cell"))
    obs["replicate"] = obs["replicate"].astype(np.int16)
    obs["lane"] = obs["lane"].astype(np.int16)
    obs["donor"] = obs["donor"].astype(np.int16)
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.write_h5ad(out_path, compression="lzf")

    written_counts = obs[recipe["perturbation_col"]].astype(str).value_counts()
    label_table = pd.DataFrame({"label": written_counts.index.astype(str), "cells_written": written_counts.to_numpy()})
    raw_table = pd.DataFrame({"target_raw": raw_names, "cells_one_target": raw_one_target, "cells_any": raw_any})
    raw_table = raw_table[raw_table["cells_any"] > 0]
    meta = {
        "member": member_name, **design, "n_cells": int(n_cells), "n_cells_with_any_guide": int(cells_with_any_guide),
        "n_features": int(n_features), "n_gene_features": int(n_genes_in_member), "n_genes_written": int(len(gene_rows)), "gene_order_status": gene_order_status, "n_guide_features": int(library["n_guides"]),
        "min_umi": min_umi, "nnz_input": int(indptr[-1]), "nnz_written": int(matrix.nnz),
        "raw_cells_0_targets": int(raw_distinct_hist[0]), "raw_cells_1_target": int(raw_distinct_hist[1]),
        "raw_cells_2_targets": int(raw_distinct_hist[2]), "raw_cells_3plus_targets": int(raw_distinct_hist[3]),
        "cells_0_targets": int(n_cells_norm_hist[0]), "cells_1_target": int(n_cells_norm_hist[1]),
        "cells_2_targets": int(n_cells_norm_hist[2]), "cells_3plus_targets": int(n_cells_norm_hist[3]),
        "cells_written": int(adata.n_obs), "cells_written_control": int((obs[recipe["perturbation_col"]].astype(str) == control_label).sum()),
        "merged_pair_cells": pair_counts, "features_sha256": _digest(gene_ids), "genes_sorted_sha256": _digest(sorted(gene_ids)),
        "guide_library_sha256": _digest(sorted(feature_name[i] for i in range(n_features) if library["slot_of_feature"][i] >= 0)),
        "labels_sha256": _digest(labels), "n_labels": len(labels), "x_dtype": "float32",
    }
    return meta, raw_table, label_table
