"""intake_list (one job) and intake (one task per tar member): the recipe source of SPORE-XL (CD4 Rest).

  xl.source: {kind: recipe, recipe: <path to recipe yaml>, tar: <TAR_PATH or folder of .h5 files>}

intake_list  scans the tar once, writes manifest/intake_members.tsv (index, name, size, cells, nnz from the headers) in SORTED member
             name order. That order is the global cell order (xl_global_index = position in the concatenation of the intake shards).
intake       one task per member: stats/intake/shard_NNNN/{meta.json, targets_raw.tsv, labels.tsv} and the shard
             _xl_work/intake/shard_NNNN/shard.h5ad (see src/intake/intake_10x_flex_tar.py).
recipe_inputs(run)  is what `plan` reads afterwards: shards table, labels, var, category order (identical across members, checked).
"""
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

import diagnostics_xl
import measure
import preflight
from xl_common import (Run, quarantine_incomplete, require_done, shard_unit, task, write_new_json, write_new_text, write_new_tsv)

from src.intake import intake_10x_flex_tar as intake_core


def source_settings(run):
    source = run.xl.get("source", {})
    if source.get("kind") != "recipe":
        raise ValueError("intake needs xl.source.kind: recipe")
    recipe = intake_core.load_recipe(source["recipe"])
    tar = source.get("tar")
    if not tar or "<TAR_PATH>" in str(tar):
        raise ValueError("xl.source.tar is not set (still the <TAR_PATH> placeholder?)")
    return recipe, tar


def members_table(run):
    path = run.manifest_dir / "intake_members.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `intake_list` first")
    return pd.read_csv(path, sep="\t")


def n_members(run):
    return len(members_table(run))


def genes_manifest_paths(run):
    return {"genes": run.manifest_dir / "intake_genes.tsv", "summary": run.manifest_dir / "intake_genes.json",
            "members": run.reports_dir / "intake_genes_members.tsv", "dropped": run.reports_dir / "intake_genes_dropped.tsv"}


def canonical_genes(run):
    """The run's ONE gene list written by intake_list: {ids, names, var_names, policy}."""
    paths = genes_manifest_paths(run)
    if not paths["genes"].exists():
        raise FileNotFoundError(f"{paths['genes']} is missing: run `intake_list` first (a run whose intake_list ran before the gene harmonization "
                                f"of build r3 must redo intake_list and intake in a fresh run folder / work dir)")
    table = pd.read_csv(paths["genes"], sep="\t", keep_default_na=False)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    return {"ids": table["gene_id"].tolist(), "names": table["name"].tolist(), "var_names": table["var_name"].tolist(), "policy": summary["policy"]}


def run_intake_list(run: Run):
    marker = run.manifest_dir / "intake_list.done"
    if marker.exists():
        run.logger.info("[intake_list] already done, skipped")
        return
    out = run.manifest_dir / "intake_members.tsv"
    gene_paths = genes_manifest_paths(run)
    quarantine_incomplete(run, "intake_list", "all", [out, *gene_paths.values()])
    recipe, tar = source_settings(run)
    members = intake_core.list_members(tar, recipe)
    if not members:
        raise ValueError(f"no members matching '{recipe['member_match']}' (*{recipe['member_suffix']}) in {tar}")
    rows, member_genes = [], {}
    for index, (name, size) in enumerate(members):
        intake_core.parse_member_name(name, recipe)  # refuse unexpected names now, not after hours of work
        with intake_core.open_member(tar, name) as handle:
            facts = measure.measure_10x_handle(handle)
            member_genes[name] = intake_core.gene_features(handle, recipe)
        rows.append({"index": index, "name": name, "size": size, "cells": facts["cells"], "nnz": facts["nnz"],
                     "gene_features": facts["gene_features"], "guide_features": facts["guide_features"]})
    table = pd.DataFrame(rows)
    # ONE gene list for the whole run: the same set in another order is reordered (lossless), different sets give the intersection (see
    # intake_10x_flex_tar.plan_gene_harmonization); the report is written even when the allowance is exceeded and the run is refused
    allowance = float(recipe.get("max_dropped_gene_fraction", 0.02))
    canonical, member_report, dropped, refusal = intake_core.plan_gene_harmonization(member_genes, allowance)
    write_new_tsv(gene_paths["members"], member_report)
    write_new_tsv(gene_paths["dropped"], dropped)
    if refusal:
        raise ValueError(f"{refusal}. Per-member fingerprints: {gene_paths['members']}; the dropped genes and the members they are missing from: "
                         f"{gene_paths['dropped']}. Set max_dropped_gene_fraction in the recipe only if you accept dropping them.")
    write_new_tsv(gene_paths["genes"], pd.DataFrame({"position": range(len(canonical["ids"])), "gene_id": canonical["ids"], "name": canonical["names"],
                                                     "var_name": intake_core.var_names_for(canonical["names"], canonical["ids"])}))
    write_new_json(gene_paths["summary"], {key: canonical[key] for key in ("policy", "reference_member", "n_union", "n_dropped", "dropped_fraction")}
                   | {"n_genes": len(canonical["ids"]), "max_dropped_gene_fraction": allowance, "n_members": len(members),
                      "statuses": member_report["status"].value_counts().to_dict()})
    write_new_tsv(out, table)
    write_new_text(marker, "intake_list finished\n")
    run.logger.info(f"[intake_list] {len(table)} members, {int(table['cells'].sum()):,} barcodes, {int(table['nnz'].sum()):,} stored values")
    run.logger.info(f"[intake_list] gene list: policy {canonical['policy']}, {len(canonical['ids']):,} genes (union {canonical['n_union']:,}, dropped "
                    f"{canonical['n_dropped']:,}), members {member_report['status'].value_counts().to_dict()}; canonical order = {canonical['reference_member']}")


def run_intake(run: Run, index: int):
    unit = shard_unit(index)
    with task(run, "stats", "intake", unit) as stats_unit:
        if stats_unit is None:
            return
        with task(run, "work", "intake", unit, log=False) as work_unit:
            recipe, tar = source_settings(run)
            table = members_table(run)
            name = table.loc[table["index"] == index, "name"].iloc[0]
            estimate = preflight.estimate_stage_ram_gb(run, "intake")
            preflight.check_task_memory(run, estimate, f"intake of {name}")
            canonical = canonical_genes(run)
            with intake_core.open_member(tar, name) as handle:
                meta, raw_table, label_table = intake_core.intake_member(handle, name, recipe, work_unit / "shard.h5ad", canonical=canonical)
            write_new_json(stats_unit / "meta.json", meta)
            write_new_tsv(stats_unit / "targets_raw.tsv", raw_table)
            write_new_tsv(stats_unit / "labels.tsv", label_table)


def shard_var_names(path):
    """var index of a shard .h5ad read with h5py (no obs, no matrix)."""
    with h5py.File(path, "r") as handle:
        group = handle["var"]
        index_name = group.attrs["_index"]
        index_name = index_name.decode() if isinstance(index_name, bytes) else str(index_name)
        return [value.decode() if isinstance(value, bytes) else str(value) for value in group[index_name][...]]


def assert_uniform_var(run, require_all=True):
    """Every intake shard has EXACTLY the run's gene list (same genes, same order). Returns the number of shards checked."""
    expected = canonical_genes(run)["var_names"]
    members = members_table(run)
    checked = 0
    for index in range(len(members)):
        unit = shard_unit(index)
        if not run.marker("work", "intake", unit).exists():
            if require_all:
                raise RuntimeError(f"intake unit {unit} is not finished: the gene-list check needs every shard")
            continue
        names = shard_var_names(run.work_dir / "intake" / unit / "shard.h5ad")
        if names != expected:
            differing = sum(1 for left, right in zip(names, expected) if left != right) + abs(len(names) - len(expected))
            raise RuntimeError(f"intake shard {unit} ({members.loc[index, 'name']}) does not have the run's gene list: {len(names):,} genes against "
                               f"{len(expected):,}, {differing:,} positions differ. All shards of one run must share ONE gene list: redo `intake_list` "
                               f"and `intake` with this version in a fresh run folder and work dir (move the old ones aside)")
        checked += 1
    return checked


def recipe_inputs(run):
    """Everything `plan` needs from the finished intake shards (and the cross-member consistency checks)."""
    members = members_table(run)
    n = len(members)
    rows, metas, labels_parts = [], [], []
    batch_parts, donor_parts, batch_col, donor_col = [], [], None, None
    first_var, categories, reference_hashes = None, None, None
    offset = 0
    for index in range(n):
        stats_unit = require_done(run, "stats", "intake", shard_unit(index))
        work_unit = require_done(run, "work", "intake", shard_unit(index))
        meta = json.loads((stats_unit / "meta.json").read_text(encoding="utf-8"))
        shard_path = work_unit / "shard.h5ad"
        hashes = (meta["features_sha256"], meta["guide_library_sha256"], meta["labels_sha256"])
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise RuntimeError(f"member {meta['member']} has a different feature list, guide library or label set than the first member")
        backed = ad.read_h5ad(shard_path, backed="r")
        try:
            if first_var is None:
                first_var = backed.var.copy()
                if list(first_var.index) != canonical_genes(run)["var_names"]:
                    raise RuntimeError(f"{shard_path}: the first shard does not have the run's gene list (intake_genes.tsv)")
                categories = [str(c) for c in backed.obs[run.pert_col].cat.categories]
                batch_col, donor_col = diagnostics_xl.diagnostic_columns(backed.obs.columns, run.cfg)
            elif list(backed.var.index) != list(first_var.index):
                raise RuntimeError(f"{shard_path}: gene list differs from the first shard ({backed.n_vars:,} against {first_var.shape[0]:,} genes); "
                                   f"all intake shards must share ONE gene list")
            elif [str(c) for c in backed.obs[run.pert_col].cat.categories] != categories:
                raise RuntimeError(f"{shard_path}: category order differs from the first shard")
            labels_parts.append(backed.obs[run.pert_col].astype(str).to_numpy())
            if batch_col:
                batch_parts.append(backed.obs[batch_col].astype(str).to_numpy())
            if donor_col:
                donor_parts.append(backed.obs[donor_col].astype(str).to_numpy())
        finally:
            backed.file.close()
        rows.append({"shard": index, "file": run.relpath(shard_path), "kind": "recipe",
                     "n_cells": meta["cells_written"], "row_start": offset, "row_stop": offset + meta["cells_written"], "obs_value": "",
                     "first_global_index": offset, "last_global_index": offset + meta["cells_written"] - 1,
                     "size_bytes": shard_path.stat().st_size, "sha256": ""})
        offset += meta["cells_written"]
        metas.append(meta)
    labels = np.concatenate(labels_parts)
    table = pd.DataFrame(rows)
    total_nnz = int(sum(m["nnz_written"] for m in metas))
    planner_source = {"path": str(run.xl["source"].get("tar")), "n_obs": int(offset), "n_vars": int(first_var.shape[0]),
                      "perturbation_col": run.pert_col, "perturbation_categorical": True, "perturbation_categories": categories,
                      "members_in_order": members["name"].tolist(), "member_order_note": "sorted by raw member name; "
                      "xl_global_index = position in the concatenation of the intake shards in this order"}
    return {"n_obs": int(offset), "n_vars": int(first_var.shape[0]), "x_dtype": "float32", "labels": labels, "category_order": categories,
            "source_var": first_var, "table": table, "batch_col": batch_col, "donor_col": donor_col,
            "batch_values": np.concatenate(batch_parts) if batch_parts else None,
            "donor_values": np.concatenate(donor_parts) if donor_parts else None, "planner_source": planner_source, "total_nnz": total_nnz,
            "nnz_per_cell": total_nnz / max(offset, 1), "largest_cells": int(table["n_cells"].max())}
