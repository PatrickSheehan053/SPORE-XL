"""hvg (one job): draw the stratified train subsample exactly as SPORE (default_rng(42), recorded category order,
train-local index), fetch those rows from the blocks, and run SPORE's Phase 8 selection and rescues unchanged.

Writes splits/{name}_hvg_panel.txt (panel genes in var order) and stats/hvg/all/{hvg_stats.parquet, core_features.txt,
hvg_subsample_index.parquet, summary.json}.
"""
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from xl_common import (Run, block_unit, quarantine_incomplete, require_done, task, write_new_json, write_new_parquet, write_new_text,
                       write_new_tsv)
import diagnostics_xl
from stage_gather import total_blocks

from adapted_phases.phase02_cell_triage import ribo_gene_mask
from adapted_phases.phase08_hvg import deg_proxy_counts, draw_subsample, select_panel_with_policy


def kept_gene_table(run):
    """var table of the genes that survived Phase 6, in var order."""
    var_table = run.var_table()
    var_reduced = var_table.loc[ribo_gene_mask(var_table.index, run.cfg, run.ribo_exempt())]
    kept_names = (require_done(run, "stats", "reduce2", "all") / "gene_keep_mask.txt").read_text(encoding="utf-8").split("\n")
    kept_names = [name for name in kept_names if name]
    keep = var_reduced.index.isin(kept_names)
    return var_reduced, keep


def run_hvg(run: Run):
    panel_path = run.splits_dir / f"{run.name}_hvg_panel.txt"
    marker = run.marker("stats", "hvg", "all")
    if not marker.exists():
        quarantine_incomplete(run, "hvg", "all", [panel_path, run.manifest_dir / "perturbation_nodes.tsv"])
    with task(run, "stats", "hvg", "all") as out:
        if out is None:
            return
        cfg, logger = run.cfg, run.logger
        if not cfg.get("phase8_hvg", {}).get("enabled", True):
            raise NotImplementedError("phase8_hvg.enabled: false is not supported by XL v1.0")
        var_reduced, keep = kept_gene_table(run)
        var_kept = var_reduced.loc[keep].copy()

        final = pd.read_parquet(run.manifest_dir / "global_obs_final.parquet")
        train = final[final["split"] == "train"].sort_values("xl_global_index")
        train_labels = train["label"].astype(str).to_numpy()
        train_global = train["xl_global_index"].to_numpy()
        category_order = run.source_info()["perturbation_category_order"]
        cap = int(run.xl.get("hvg_subsample_cells", 100000))
        idx_sorted = draw_subsample(train_labels, category_order, cap)
        sample_global = train_global[idx_sorted]
        write_new_parquet(out / "hvg_subsample_index.parquet", pd.DataFrame(
            {"train_position": idx_sorted.astype(np.int64), "xl_global_index": sample_global.astype(np.int64)}))

        # ---- fetch the sampled rows from the blocks, restricted to the Phase 6 genes
        wanted = set(sample_global.tolist())
        pieces, piece_index = [], []
        for block in range(total_blocks(run)):
            path = require_done(run, "work", "gather", block_unit(block)) / "block.h5ad"
            if not path.exists():
                continue
            block_adata = ad.read_h5ad(path)
            gidx = block_adata.obs["xl_global_index"].to_numpy()
            rows = np.flatnonzero(np.isin(gidx, sample_global))
            if len(rows):
                pieces.append(sp.csr_matrix(block_adata.X[rows])[:, np.flatnonzero(keep)])
                piece_index.append(gidx[rows])
        matrix = sp.vstack(pieces, format="csr")
        index = np.concatenate(piece_index)
        if len(index) != len(sample_global) or set(index.tolist()) != wanted:
            raise RuntimeError("subsample rows fetched from blocks do not match the drawn indices")
        matrix = matrix[np.argsort(index, kind="stable")]  # train order == global order

        calc_adata = ad.AnnData(X=matrix, var=var_kept.copy())
        stub = ad.AnnData(X=sp.csr_matrix((0, var_kept.shape[0]), dtype=np.float32), var=var_kept.copy())
        present = set(train_labels)
        ranking_inputs = None
        if cfg.get("phase8_hvg", {}).get("target_node_policy", "budget") == "budget":
            ranking_inputs = _ranking_inputs(run, train_labels, keep)
        keep_mask, hvg_stats, core_features, hvg_names, ghosts, policy_report = select_panel_with_policy(
            calc_adata, stub, present, cfg, logger, ranking_inputs)
        details = policy_report.pop("_details", None)
        logger.info(f"[hvg] policy report: {policy_report}")

        panel = [name for name, flag in zip(var_kept.index, keep_mask) if flag]
        write_new_text(panel_path, "\n".join(panel) + "\n")
        write_node_table(run, final, set(panel), set(var_kept.index), hvg_stats, details, out)
        write_new_parquet(out / "hvg_stats.parquet", hvg_stats)
        write_new_text(out / "core_features.txt", "\n".join(core_features) + "\n")
        write_new_text(out / "panel_mask_over_kept_genes.txt", "\n".join(map(str, keep_mask.astype(int))) + "\n")
        write_new_json(out / "summary.json", {
            "n_subsample": int(len(sample_global)), "n_train": int(len(train)), "n_panel": int(keep_mask.sum()),
            "n_genes_phase6": int(var_kept.shape[0]), "n_ghosts_stacked": len(ghosts),
            "rescued_target": int(hvg_stats["rescued_target"].sum()),
            "rescued_protected": int(hvg_stats["rescued_protected"].sum()),
            "rescued_heldout": int(hvg_stats["rescued_heldout"].sum()), **policy_report})
        logger.info(f"[hvg] subsample {len(sample_global):,}/{len(train):,} train cells, panel {int(keep_mask.sum())} genes")
        if cfg.get("diagnostics", {}).get("enabled", True):
            diagnostics = subsample_diagnostics(matrix, var_kept.index, keep_mask, sample_global, final, train_labels, idx_sorted)
            write_new_json(out / "diagnostics.json", diagnostics)
            logger.info(f"[hvg] diagnostics (train subsample): {diagnostics.get('variance_explained', 'n/a')}")


def subsample_diagnostics(matrix, var_names, panel_mask, sample_global, final, train_labels, idx_sorted):
    """P5: variance explained by batch / donor / perturbation on the log-normalized train subsample (panel genes), and the
    cell-cycle score diagnostic. Train cells only; nothing is changed."""
    from src.phases.utils import get_cell_cycle_genes
    lognorm = diagnostics_xl.lognorm_copy(matrix)
    panel_columns = np.flatnonzero(panel_mask)
    sample_obs = final.set_index("xl_global_index").loc[sample_global]
    labels = np.asarray(train_labels)[idx_sorted].astype(str)
    result = {"n_cells": int(lognorm.shape[0]), "n_panel_genes": int(len(panel_columns)), "variance_explained": {}}
    factors = {"perturbation": labels}
    if "batch" in sample_obs.columns:
        factors["batch"] = sample_obs["batch"].astype(str).to_numpy()
    if "donor" in sample_obs.columns:
        factors["donor"] = sample_obs["donor"].astype(str).to_numpy()
    for name, groups in factors.items():
        r2, adjusted, n_groups = diagnostics_xl.variance_explained(lognorm[:, panel_columns], groups)
        result["variance_explained"][name] = {"groups": int(n_groups), "r2": None if not np.isfinite(r2) else round(float(r2), 6),
                                              "r2_adjusted": None if not np.isfinite(adjusted) else round(float(adjusted), 6)}
    s_genes, g2m_genes = get_cell_cycle_genes()
    result["cell_cycle"] = diagnostics_xl.cell_cycle_diagnostic(lognorm, var_names, s_genes, g2m_genes, labels, factors.get("batch"))
    return result


def write_node_table(run, final, panel, phase6_genes, hvg_stats, details, out):
    """P2: manifest/perturbation_nodes.tsv = every surviving perturbation, its split, its cells, and whether its target gene is a NODE of
    the panel (a column of the outputs) or only ROWS (cells kept, target not a node), with the reason."""
    survivors = final[final["survived_to_phase7"] & (final["label"].astype(str) != run.ctrl_label)]
    counts = survivors.groupby(survivors["label"].astype(str), observed=True).agg(n_cells=("xl_global_index", "size"), split=("split", "first"))
    rows = []
    for label, row in counts.iterrows():
        in_panel = label in panel
        if label not in phase6_genes:
            reason = "target_not_measured_or_removed_by_gene_qc"
        elif details is not None:
            if label in details["heldout"]:
                reason = "heldout_forced" if in_panel else "heldout_missing"
            elif label in details["protected"]:
                reason = "protected"
            elif label in details["train_added"]:
                reason = "train_ranked_in_budget"
            elif in_panel:
                reason = "hvg_variance_rank"
            else:
                reason = "train_over_budget"
        else:
            reason = "all_targets_policy" if in_panel else "not_in_panel"
        rows.append({"label": label, "split": str(row["split"]), "n_cells": int(row["n_cells"]), "is_node": bool(in_panel), "reason": reason,
                     "budget_rank_score": (details or {}).get("train_rank", {}).get(label, "")})
    table = pd.DataFrame(rows, columns=["label", "split", "n_cells", "is_node", "reason", "budget_rank_score"])
    write_new_tsv(run.manifest_dir / "perturbation_nodes.tsv", table.sort_values("label"))
    summary = table.groupby("reason").agg(perturbations=("label", "size"), cells=("n_cells", "sum"))
    write_new_tsv(out / "node_summary.tsv", summary, index=True)
    run.logger.info(f"[hvg] perturbation nodes: {int(table['is_node'].sum())} node perturbations, {int((~table['is_node']).sum())} row-only "
                    f"({int(table.loc[~table['is_node'], 'n_cells'].sum()):,} cells); manifest/perturbation_nodes.tsv")


def _ranking_inputs(run, train_labels, keep):
    """cells after the escaper filter, knockdown depth and a DEG-count proxy for every train target (blueprint task 6)."""
    import numpy as np
    reduce2 = require_done(run, "stats", "reduce2", "all")
    control_path = reduce2 / "control_mean.npy"
    if not control_path.exists():
        raise FileNotFoundError(f"{control_path} is missing: this run's reduce2 pre-dates the budget policy; re-run reduce2 with the current code")
    control_mean = np.load(control_path)
    train_set = set(train_labels) - {run.ctrl_label}
    cells = pd.Series(train_labels).value_counts().to_dict()
    knockdown = pd.read_csv(require_done(run, "stats", "reduce1", "all") / "knockdown.tsv", sep="\t", keep_default_na=False)
    depth = knockdown.set_index("label")["knockdown_depth"].astype(float).to_dict()
    permeans = {}
    n_blocks = run.layout()["n_blocks"]
    for block in range(n_blocks):
        with np.load(require_done(run, "stats", "gather", block_unit(block)) / "permean.npz") as data:
            for label, mean in zip(data["labels"].tolist(), data["means"]):
                if label in train_set:
                    permeans[label] = mean
    cfg8 = run.cfg.get("phase8_hvg", {})
    degs = deg_proxy_counts(permeans, control_mean, keep, cfg8.get("deg_log2fc", 0.5), cfg8.get("deg_min_mean", 0.1))
    return {"cells": cells, "depth": depth, "degs": degs}
