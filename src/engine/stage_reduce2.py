"""reduce2 (one job, serial): every global decision after the escaper filter.

 1. merge GeneSums of all blocks -> Phase 6 keep mask, rescues, penetrance (adapted_phases/phase06)
 2. control mean: one carried float32 accumulator over the control blocks in global order (phase07:160)
 3. mean-shift score per perturbation from the per-perturbation float32 means, masked to the kept genes, summed in
    float32 and ranked exactly like phase07:164-173
 4. label hygiene check, then the zero-shot split (adapted_phases/phase07: same RNG order as SPORE) and the control assignment
 5. write split JSON (SPORE schema, byte-identical), control_assignment.parquet, global_obs_final.parquet
"""
import json
from collections import OrderedDict

import numpy as np
import pandas as pd

from xl_common import (Run, alias_map, block_unit, quarantine_incomplete, require_done, task, write_new_json, write_new_parquet,
                       write_new_text, write_new_tsv)
import diagnostics_xl
from stage_gather import total_blocks
from stats import F32MeanAccumulator, GeneSums

from adapted_phases.phase02_cell_triage import ribo_gene_mask
from adapted_phases.phase06_gene_triage import keep_mask_from_stats
from adapted_phases.phase07_splits import split_from_scores


def check_label_hygiene(run, labels, var_reduced):
    """Split integrity item 2: two different labels must not point at the same gene (a gene under two labels could
    land in two splits)."""
    gene_to_idx = alias_map(var_reduced)
    from src.phases.phase05_escaper_filtering import _parse_target_genes
    seen = {}
    for label in labels:
        genes = tuple(sorted(gene_to_idx[g] for g in _parse_target_genes(label, run.cfg) if g in gene_to_idx))
        if genes and len(genes) == 1:
            if genes in seen and seen[genes] != label:
                raise ValueError(f"labels '{seen[genes]}' and '{label}' map to the same gene "
                                 f"{var_reduced.index[genes[0]]}; fix the labels in intake (split integrity rule)")
            seen[genes] = label


def run_reduce2(run: Run):
    cfg, logger = run.cfg, run.logger
    split_json = run.splits_dir / f"{run.name}_split_indices.json"
    control_path = run.splits_dir / f"{run.name}_control_assignment.parquet"
    final_obs_path = run.manifest_dir / "global_obs_final.parquet"
    marker = run.marker("stats", "reduce2", "all")
    if not marker.exists():
        quarantine_incomplete(run, "reduce2", "all", [split_json, control_path, final_obs_path])
    with task(run, "stats", "reduce2", "all") as out:
        if out is None:
            for path in (split_json, control_path, final_obs_path):
                if not path.exists():
                    raise FileNotFoundError(f"reduce2 is marked done but {path} is missing")
            return
        n_total = total_blocks(run)
        layout = run.layout()
        var_table = run.var_table()
        var_reduced = var_table.loc[ribo_gene_mask(var_table.index, cfg, run.ribo_exempt())]
        var_names = [str(name) for name in var_reduced.index]

        # ---- 1. merge block statistics
        sums, survivors, dropped = None, [], []
        for block in range(n_total):
            unit = require_done(run, "stats", "gather", block_unit(block))
            block_sums = GeneSums.load(unit / "genesums.npz")
            sums = block_sums if sums is None else sums.add(block_sums)
            survivors.append(pd.read_parquet(unit / "survivors.parquet"))
            dropped.append(pd.read_parquet(unit / "dropped.parquet"))
        survivors = pd.concat(survivors).sort_values("xl_global_index").reset_index(drop=True)
        dropped = pd.concat(dropped)
        n_survivors = len(survivors)
        if sums.n_cells != n_survivors:
            raise RuntimeError(f"GeneSums cells {sums.n_cells} != survivors {n_survivors}")
        surviving_labels = sorted(set(survivors["label"]) - {run.ctrl_label})
        if not sums.exact_in_float32():
            logger.warning("  gene sums exceed 2**24: SPORE's float32 accumulation would round; XL uses the exact sum")

        # ---- Phase 6 keep mask
        final_keep, waterfall6, rescued, ghosts, penetrance = keep_mask_from_stats(
            sums, var_names, surviving_labels, cfg, logger,
            ghost_rescue=bool(cfg.get("phase6_gene_triage", {}).get("rescue_cell_cycle_genes", False)))
        write_new_tsv(out / "gene_stats.tsv", pd.DataFrame({
            "gene": var_names, "cells_per_gene": sums.nnz, "sum": sums.total, "keep": final_keep}))
        write_new_text(out / "gene_keep_mask.txt", "\n".join(np.array(var_names)[final_keep]) + "\n")
        write_new_text(out / "phase6_rescued.txt", "\n".join(rescued) + "\n")
        write_new_json(out / "phase6_waterfall.json", {key: int(value) for key, value in waterfall6.items()})
        np.save(out / "phase6_penetrance.npy", penetrance)

        # ---- QC var columns, subset to the SPORE gene space (kept in stats for hvgraw var later)
        qc = pd.read_csv(require_done(run, "stats", "reduce1", "all") / "qc_var_columns.tsv", sep="\t", index_col=0)
        write_new_tsv(out / "qc_var_columns_post_ribo.tsv", qc.loc[var_reduced.index], index=True)

        # ---- 2. control mean: serial carried float32 accumulator in global order
        control_rows = survivors[survivors["label"] == run.ctrl_label]
        n_controls = len(control_rows)
        control_blocks = range(layout["n_blocks"], n_total)
        import anndata as ad
        accumulator = F32MeanAccumulator(len(var_names), n_controls)
        for block in control_blocks:
            path = require_done(run, "work", "gather", block_unit(block)) / "block.h5ad"
            if path.exists():
                accumulator.add_block(ad.read_h5ad(path).X)
        control_mean = accumulator.result()
        np.save(out / "control_mean.npy", control_mean)  # float32 over the post-ribosomal genes; used by the hvg target-node ranking

        # ---- 3. mean shift (phase07:164-173)
        permean = {}
        for block in range(layout["n_blocks"]):
            with np.load(require_done(run, "stats", "gather", block_unit(block)) / "permean.npz") as data:
                for label, mean in zip(data["labels"].tolist(), data["means"]):
                    permean[label] = mean
        perturbations = [label for label in np.unique(survivors["label"].to_numpy()) if label != run.ctrl_label]
        if sorted(perturbations) != sorted(permean):
            raise RuntimeError("per-perturbation means do not cover the surviving perturbations")
        shifts = {}
        control_kept = control_mean[final_keep]
        for target in perturbations:
            shifts[target] = float(np.abs(permean[target][final_keep] - control_kept).sum())
        deg_counts = pd.Series(shifts).sort_values(ascending=False)
        write_new_tsv(out / "mean_shift.tsv", deg_counts.rename("score").map(repr).rename_axis("label").to_frame(), index=True)

        # ---- 4. labels, split, control assignment
        check_label_hygiene(run, perturbations, var_reduced)
        result = split_from_scores(deg_counts, n_controls, cfg, seed=cfg["phase7_splits"].get("random_seed", 42))
        meta = {"train_labels": list(result["train_labels"]), "val_labels": list(result["val_labels"]),
                "test_labels": list(result["test_labels"]), "seed": int(result["seed"])}
        with open(split_json, "w") as handle:  # same call as phase07:328-330
            json.dump(meta, handle, indent=2)
        control_global = control_rows["xl_global_index"].to_numpy()
        assignment = pd.DataFrame({
            "xl_global_index": np.concatenate([control_global[result["control_train_positions"]],
                                               control_global[result["control_val_positions"]],
                                               control_global[result["control_test_positions"]]]),
            "split": np.repeat(["train", "val", "test"], [len(result["control_train_positions"]),
                                                          len(result["control_val_positions"]),
                                                          len(result["control_test_positions"])])})
        assignment = assignment.sort_values("xl_global_index").reset_index(drop=True)
        write_new_parquet(control_path, assignment)

        # ---- 5. global_obs_final
        global_obs = pd.read_parquet(run.manifest_dir / "global_obs.parquet")
        split_of_label = {**{label: "train" for label in meta["train_labels"]},
                          **{label: "val" for label in meta["val_labels"]},
                          **{label: "test" for label in meta["test_labels"]}}
        survived = np.zeros(len(global_obs), dtype=bool)
        survived[survivors["xl_global_index"].to_numpy()] = True
        split = np.full(len(global_obs), "", dtype=object)
        labels = global_obs["label"].astype(str).to_numpy()
        for index in np.flatnonzero(survived & (labels != run.ctrl_label)):
            split[index] = split_of_label[labels[index]]
        split[assignment["xl_global_index"].to_numpy()] = assignment["split"].to_numpy()
        final = global_obs.assign(survived_to_phase7=survived, split=pd.Categorical(split, categories=["", "train", "val", "test"]))
        write_new_parquet(final_obs_path, final)
        write_new_tsv(out / "dropped_undersized.tsv", dropped.sort_values("label"))
        if "batch" in final.columns and cfg.get("diagnostics", {}).get("enabled", True):  # P5: metadata only, nothing is changed
            kept = final[final["survived_to_phase7"]]
            tvd_warn = float((cfg.get("diagnostics", {}) or {}).get("confound_tvd_warn", 0.5))
            alpha = float((cfg.get("diagnostics", {}) or {}).get("confound_alpha", 0.001))
            confounding, confounding_summary = diagnostics_xl.confounding_table(kept["label"], kept["batch"], run.ctrl_label, tvd_warn=tvd_warn, alpha=alpha)
            write_new_tsv(out / "confounding.tsv", confounding)
            write_new_json(out / "confounding_summary.json", confounding_summary)
            if confounding_summary["n_flagged"]:
                logger.warning(f"[reduce2] BATCH CONFOUNDING: {confounding_summary['n_flagged']} of {confounding_summary['n_perturbations']} perturbations "
                               f"({confounding_summary['cells_in_flagged']:,} cells) have a batch mix that differs from the controls beyond sampling noise "
                               f"(TVD >= {tvd_warn} and p <= {alpha} against a random draw of the same number of control cells; the fixed-threshold rule alone would "
                               f"flag {confounding_summary['n_flagged_uncalibrated']}); {confounding_summary['n_fully_in_one_batch']} sit entirely in one batch; "
                               f"see stats/reduce2/all/confounding.tsv")
        write_new_json(out / "summary.json", {
            "n_survivors": int(n_survivors), "n_controls": int(n_controls), "n_perturbations": len(perturbations),
            "n_genes_kept": int(final_keep.sum()), "n_train": len(meta["train_labels"]), "n_val": len(meta["val_labels"]),
            "n_test": len(meta["test_labels"]), "split_json": str(split_json), "control_assignment": str(control_path)})
        logger.info(f"[reduce2] {n_survivors:,} survivors, {int(final_keep.sum())} genes kept, split "
                    f"{len(meta['train_labels'])}/{len(meta['val_labels'])}/{len(meta['test_labels'])}")
