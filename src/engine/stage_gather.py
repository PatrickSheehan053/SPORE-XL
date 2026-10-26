"""gather (one task per block): concatenate the bucket pieces of a block in xl_global_index order, drop undersized
perturbations (phase05:239-258; every perturbation is whole in its block, controls never dropped), attach SPORE's
escaper columns, and emit the statistics SPORE's Phases 6 and 7 read:

stats/gather/block_BBBB/
  genesums.npz        GeneSums over the surviving cells of this block (Phase 6 input)
  survivors.parquet   xl_global_index, label of every surviving cell
  labelcounts.parquet label, n_cells
  dropped.parquet     undersized perturbations dropped here (label, n_kept)
  permean.npz         per-perturbation float32 column means over all genes, bit-for-bit scipy's X[pert].mean(axis=0)
_xl_work/gather/block_BBBB/block.h5ad  the block matrix (surviving cells, global order)
"""
import json

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import preflight
from xl_common import (Run, alias_map, block_unit, require_done, shard_unit, task, write_new_parquet, write_new_tsv)
from stats import F32MeanAccumulator, GeneSums

from adapted_phases.phase05_escaper_filtering import efficiency, per_cell_efficiency_columns, undersized_labels


FINAL_COUNT_COLUMNS = ["label", "n_total", "n_final", "bypassed"]


def apply_escaper_rules(run, mode, matrix, obs, labels, var):
    """P4. obs carries escaper_rule1 (rule 1 flag). Computes the rule-2 flag per perturbation and removes cells by mode:
    report = rule 1 only (rule 2 is only counted), union = rule 1 or rule 2, replace = rule 2 (rule 1 where rule 2 cannot be computed).
    Controls are never removed. Returns (matrix, obs, labels, per-perturbation records)."""
    import escaper_score
    from stage_map1 import target_genes_by_label
    z = float(run.cfg.get("phase5_escaper_filtering", {}).get("escaper_score_z", 2.0))
    with np.load(require_done(run, "stats", "reduce1", "all") / "control_reference.npz") as data:
        reference = data["rows"].copy() if len(data["rows"]) else None
    mu_control = reference.mean(axis=0) if reference is not None else None
    gene_to_idx = alias_map(var)
    labels_all = run.labels_table()["label"].tolist()
    gene_lists = target_genes_by_label(run, labels_all, gene_to_idx)
    rule1 = obs["escaper_rule1"].to_numpy(dtype=bool)
    rule2 = np.zeros(len(labels), dtype=bool)
    has_rule2 = {}
    records = []
    rows_by_label = pd.Series(np.arange(len(labels))).groupby(labels).apply(lambda rows: rows.to_numpy())
    for label in sorted(rows_by_label.index):
        if label == run.ctrl_label:
            continue
        rows = rows_by_label[label]
        flags, info = escaper_score.score_perturbation(escaper_score.lognorm_rows(matrix[rows]), reference, gene_lists.get(label, []), z=z, mu_control=mu_control)
        has_rule2[label] = flags is not None
        if flags is not None:
            rule2[rows] = flags
        both = int((rule1[rows] & rule2[rows]).sum())
        records.append({"label": label, "n_total": int(len(rows)), "n_rule1": int(rule1[rows].sum()),
                        "n_rule2": int(rule2[rows].sum()) if flags is not None else -1, "n_both": both if flags is not None else -1,
                        "rule2_status": info["status"], "control_score_mean": info.get("control_score_mean", float("nan")),
                        "control_score_sd": info.get("control_score_sd", float("nan")), "median_cell_score": info.get("median_cell_score", float("nan"))})
    is_control = labels == run.ctrl_label
    if mode == "report":
        remove = rule1.copy()
    elif mode == "union":
        remove = rule1 | rule2
    else:  # replace: rule 2 where it could be computed, otherwise rule 1
        usable = np.array([has_rule2.get(label, False) for label in labels])
        remove = np.where(usable, rule2, rule1)
    remove &= ~is_control
    kept_records = pd.Series(~remove).groupby(labels).sum()
    for record in records:
        record["n_removed"] = int(record["n_total"] - kept_records.get(record["label"], 0))
        record["mode"] = mode
    keep = ~remove
    obs = obs.loc[keep].drop(columns=["escaper_rule1"])
    return matrix[keep], obs, labels[keep], records


def assigned_rows(run, block, counts, already):
    """final_counts rows for the perturbations that belong to this block but sent no cell to it (every QC-passing cell was removed as an
    escaper in map2): n_final 0. Without them the waterfall would lose their cells (found by the stress suite, dataset 06)."""
    from xl_common import block_of_label
    n_blocks = run.layout()["n_blocks"]
    if block >= n_blocks:
        return []
    return [{"label": label, "n_total": int(row.n_total), "n_final": 0, "bypassed": bool(row.bypassed)}
            for label, row in counts.iterrows() if label != run.ctrl_label and label not in already and block_of_label(label, n_blocks) == block]


def escaper_counts(run):
    """Per perturbation: n_total (QC-passing cells), n_kept (cells left after the escaper rules, before the undersized drop), bypassed.
    From the gather units when they wrote final_counts.parquet (every v4 run), else from map2 (older runs)."""
    frames = []
    for block in range(total_blocks(run)):
        path = run.stats_dir / "gather" / block_unit(block) / "final_counts.parquet"
        if not path.exists():
            return merged_counts(run)
        frames.append(pd.read_parquet(path))
    merged = pd.concat(frames).set_index("label").rename(columns={"n_final": "n_kept"})
    return merged[["n_total", "n_kept", "bypassed"]]


def total_blocks(run):
    layout = run.layout()
    return layout["n_blocks"] + layout["n_control_blocks"]


def merged_counts(run):
    frames = []
    for index in range(run.n_shards()):
        unit = require_done(run, "stats", "map2", shard_unit(index))
        frames.append(pd.read_parquet(unit / "counts.parquet"))
    merged = pd.concat(frames)
    bypassed = merged.groupby("label")["bypassed"].max()
    totals = merged.groupby("label")[["n_total", "n_kept"]].sum()
    totals["bypassed"] = bypassed
    return totals


def run_gather(run: Run, block):
    unit = block_unit(block)
    with task(run, "stats", "gather", unit) as stats_unit:
        if stats_unit is None:
            return
        with task(run, "work", "gather", unit, log=False) as work_unit:
            _gather(run, block, stats_unit, work_unit)


def _gather(run, block, stats_unit, work_unit):
    cfg = run.cfg
    pieces = []
    for index in range(run.n_shards()):
        directory = require_done(run, "work", "map2", shard_unit(index))
        path = directory / f"block_{block:04d}.h5ad"
        if path.exists():
            pieces.append(ad.read_h5ad(path))
    if pieces:  # RAM pre-flight from the on-disk size of the pieces (lzf pieces are about 1:1 with RAM x working copies)
        piece_gb = sum((require_done(run, "work", "map2", shard_unit(i)) / f"block_{block:04d}.h5ad").stat().st_size
                       for i in range(run.n_shards())
                       if (require_done(run, "work", "map2", shard_unit(i)) / f"block_{block:04d}.h5ad").exists()) / 1e9
        preflight.check_task_memory(run, piece_gb * 3, f"gather block {block}")
    if not pieces:
        meta = json.loads((require_done(run, "stats", "map1", shard_unit(0)) / "meta.json").read_text(encoding="utf-8"))
        n_genes = int(meta["n_genes_after_ribo"])
        GeneSums(n_genes).save(stats_unit / "genesums.npz")
        np.savez(stats_unit / "permean.npz", labels=np.array([], dtype=str), means=np.zeros((0, n_genes), dtype=np.float32))
        write_new_parquet(stats_unit / "survivors.parquet", pd.DataFrame(
            {"xl_global_index": np.zeros(0, dtype=np.int64), "label": np.zeros(0, dtype=str)}))
        write_new_parquet(stats_unit / "labelcounts.parquet", pd.DataFrame({"label": [], "n_cells": []}))
        write_new_parquet(stats_unit / "dropped.parquet", pd.DataFrame({"label": [], "n_kept": []}))
        write_new_parquet(stats_unit / "final_counts.parquet", pd.DataFrame(assigned_rows(run, block, merged_counts(run), set()), columns=FINAL_COUNT_COLUMNS))
        return
    matrix = sp.vstack([piece.X for piece in pieces], format="csr")
    obs = pd.concat([piece.obs for piece in pieces])
    var = pieces[0].var
    order = np.argsort(obs["xl_global_index"].to_numpy(), kind="stable")
    matrix, obs = matrix[order], obs.iloc[order].copy()
    labels = obs[run.pert_col].astype(str).to_numpy()

    counts = merged_counts(run)
    mode = run.escaper_mode
    arrived = pd.Series(labels).value_counts()
    # off: map2 already dropped the rule-1 escapers; any other mode: map2 sent every QC-passing cell, flagged (escaper_rule1)
    expected_column = "n_kept" if mode == "off" else "n_total"
    for label, size in arrived.items():
        if label != run.ctrl_label and int(counts.loc[label, expected_column]) != size:
            raise RuntimeError(f"block {block}: perturbation {label} has {size} cells here but map2 sent "
                               f"{int(counts.loc[label, expected_column])}; a perturbation is split across blocks")
    rule_records = []
    if mode != "off":
        matrix, obs, labels, rule_records = apply_escaper_rules(run, mode, matrix, obs, labels, var)
    block_counts = pd.Series(labels).value_counts()
    final_rows = [{"label": label, "n_total": int(counts.loc[label, "n_total"]), "n_final": int(block_counts.get(label, 0)),
                   "bypassed": bool(counts.loc[label, "bypassed"])} for label in arrived.index if label != run.ctrl_label]
    final_rows += assigned_rows(run, block, counts, {row["label"] for row in final_rows})  # labels whose cells ALL escaped send nothing here
    write_new_parquet(stats_unit / "final_counts.parquet", pd.DataFrame(final_rows, columns=FINAL_COUNT_COLUMNS))
    if rule_records:
        write_new_parquet(stats_unit / "rule_counts.parquet", pd.DataFrame(rule_records))

    min_cells = cfg.get("phase5_escaper_filtering", {}).get("min_cells_per_perturbation", 50)
    drop = undersized_labels(block_counts.to_dict(), min_cells, run.ctrl_label)
    keep = ~np.isin(labels, list(drop))
    write_new_parquet(stats_unit / "dropped.parquet", pd.DataFrame(
        {"label": sorted(drop), "n_kept": [int(block_counts[label]) for label in sorted(drop)]}))

    knockdown = pd.read_csv(require_done(run, "stats", "reduce1", "all") / "knockdown.tsv", sep="\t",
                            keep_default_na=False).set_index("label")["knockdown_depth"].astype(float).to_dict()
    efficiency_by_label = {row["label"]: efficiency(row["n_final"], row["n_total"]) for row in final_rows}
    eff_scores, kd_depths = per_cell_efficiency_columns(labels, efficiency_by_label, knockdown)
    obs["escaper_efficiency"] = eff_scores
    obs["escaper_knockdown_depth"] = kd_depths

    matrix, obs, labels = matrix[keep], obs.loc[keep], labels[keep]
    block_adata = ad.AnnData(X=sp.csr_matrix(matrix), obs=obs, var=var)
    block_adata.write_h5ad(work_unit / "block.h5ad", compression=run.xl.get("bucket_compression", "lzf"))

    GeneSums(matrix.shape[1]).add_matrix(matrix).save(stats_unit / "genesums.npz")
    write_new_parquet(stats_unit / "survivors.parquet", pd.DataFrame(
        {"xl_global_index": obs["xl_global_index"].to_numpy(), "label": labels}))
    sizes = pd.Series(labels).value_counts()
    write_new_parquet(stats_unit / "labelcounts.parquet", pd.DataFrame(
        {"label": sizes.index.astype(str), "n_cells": sizes.to_numpy()}))

    # per-perturbation float32 means (controls are handled serially in reduce2)
    mean_labels, mean_rows = [], []
    rows_by_label = pd.Series(np.arange(len(labels))).groupby(labels).apply(lambda rows: rows.to_numpy())
    for label in sorted(rows_by_label.index):
        if label == run.ctrl_label:
            continue
        rows = rows_by_label[label]
        accumulator = F32MeanAccumulator(matrix.shape[1], len(rows)).add_block(matrix[rows])
        mean_labels.append(label)
        mean_rows.append(accumulator.result())
    np.savez(stats_unit / "permean.npz", labels=np.array(mean_labels, dtype=str),
             means=np.vstack(mean_rows) if mean_rows else np.zeros((0, matrix.shape[1]), dtype=np.float32))
