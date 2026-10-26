"""XL clone of SPORE src/phase08_hvg.py::select_hvgs (SPORE v3.0, GitHub 5015dc6), split into
  draw_subsample    lines 506-522 (stratified subsample indices, default_rng(42), group order = category order)
  select_panel      lines 533-647 (Seurat v3 HVG on the subsample, target rescue, ghost rescue, protected rescue,
                    held-out force-carry), run UNCHANGED on the subsample
so that the full train matrix is never needed. Row positions are train-local, exactly SPORE's `orig_idx`.
"""
import numpy as np
import pandas as pd
import scanpy as sc

from src.phases.phase08_hvg import force_include_into_hvg, load_heldout_target_ids
from src.phases.utils import get_cell_cycle_genes, load_protected_genes


def _tie_break(p8):
    """v4 (P8): 'stable' = ties in the swap ranking are broken by gene name (no dependence on PYTHONHASHSEED); 'spore_set' = SPORE v3.0's
    list(set) + unstable sort (compat tests; needs PYTHONHASHSEED=0 to be repeatable)."""
    mode = p8.get("tie_break", "stable")
    if mode not in ("stable", "spore_set"):
        raise ValueError(f"phase8_hvg.tie_break must be stable or spore_set, got {mode}")
    return mode


def _cell_cycle_in_matrix(var_names, p8):
    """Cell-cycle genes present in the matrix, or [] when cell-cycle rescue is off (v4 default, P3: it only served Phase 10)."""
    if not p8.get("rescue_cell_cycle_genes", False):
        return []
    cc_s, cc_g2m = get_cell_cycle_genes()
    var_upper = {v.upper(): v for v in var_names}
    return [var_upper[g] for g in set(cc_s + cc_g2m) if g in var_upper]


def _lowest(hvg_stats, rank_col, names, count, tie_break):
    """The `count` genes of `names` with the lowest rank score."""
    if tie_break == "spore_set":
        return hvg_stats.loc[list(names), rank_col].sort_values().head(count).index.tolist()
    return hvg_stats.loc[sorted(names), rank_col].sort_values(kind="stable").head(count).index.tolist()


def force_include(force_ids, hvg_names, hvg_stats, var_names, rank_col, do_not_drop, flag_col, tag, logger, tie_break):
    """SPORE's force_include_into_hvg (phase08:89-121); with tie_break 'stable' the genes swapped out are the lowest-ranked with ties
    broken by gene name instead of by set order."""
    if tie_break == "spore_set":
        return force_include_into_hvg(force_ids, hvg_names, hvg_stats, var_names, rank_col, do_not_drop, flag_col, tag, logger)
    if flag_col not in hvg_stats.columns:
        hvg_stats[flag_col] = False
    to_add = (set(map(str, force_ids)) & set(map(str, var_names))) - set(hvg_names)
    if not to_add:
        return hvg_names
    safe_to_drop = set(hvg_names) - set(map(str, force_ids)) - set(do_not_drop)
    if rank_col in hvg_stats.columns and len(safe_to_drop) >= len(to_add):
        genes_to_drop = _lowest(hvg_stats, rank_col, safe_to_drop, len(to_add), tie_break)
        hvg_names.difference_update(genes_to_drop)
        hvg_names.update(to_add)
        hvg_stats.loc[genes_to_drop, "highly_variable"] = False
        hvg_stats.loc[sorted(to_add), "highly_variable"] = True
        hvg_stats.loc[sorted(to_add), flag_col] = True
        if logger:
            logger.info(f"  {tag}: swapped {len(to_add)} genes into HVG (panel size stable)")
    else:
        hvg_names.update(to_add)
        hvg_stats.loc[sorted(to_add), flag_col] = True
        if logger:
            logger.info(f"  {tag}: stacked {len(to_add)} genes into HVG")
    return hvg_names


def draw_subsample(train_labels, category_order, n_sample_cap=100_000):
    """train_labels: label of every TRAIN cell in train order (global order restricted to train cells).
    Returns idx_sorted (train-local positions)."""
    n_obs = len(train_labels)
    n_sample = min(n_sample_cap, n_obs)
    frac = n_sample / n_obs
    df = pd.DataFrame({"pert": pd.Categorical(train_labels, categories=category_order)})
    df["orig_idx"] = np.arange(n_obs)
    rng = np.random.default_rng(42)
    sampled = []
    for _, group in df.groupby("pert", observed=True):
        if len(group) == 0:
            continue
        n_take = max(1, min(int(np.round(len(group) * frac)), len(group)))
        sampled.extend(
            rng.choice(group["orig_idx"].values, size=n_take, replace=False))
    return np.sort(np.array(sampled))


def select_panel(calc_adata, train_adata_stub, train_labels_present, cfg, logger):
    """calc_adata: the subsample AnnData (X float32 CSR, var = kept genes).
    train_adata_stub: zero-row AnnData carrying the kept-gene var table (load_protected_genes reads var only).
    train_labels_present: set of perturbation labels present in the train split (controls included or not).
    Returns (keep_mask over var order, hvg_stats DataFrame, core_features list, hvg_names set, ghosts list)."""
    p8 = cfg.get("phase8_hvg", {})
    n_top = p8.get("n_top_genes", 5000)
    method = p8.get("method", "seurat_v3")
    pert_col = cfg.get("dataset", {}).get("perturbation_col", "perturbation")
    var_names = train_adata_stub.var_names

    tie_break = _tie_break(p8)
    cc_in_matrix = _cell_cycle_in_matrix(var_names, p8)

    calc_adata.X.data = calc_adata.X.data.astype(np.float32, copy=False)
    sc.pp.highly_variable_genes(
        calc_adata, flavor=method, n_top_genes=n_top, subset=False)

    stat_cols = [c for c in calc_adata.var.columns
                 if c in ("highly_variable", "means",
                          "dispersions", "dispersions_norm",
                          "variances", "variances_norm")]
    hvg_stats = calc_adata.var[stat_cols].copy()
    hvg_names = set(calc_adata.var_names[calc_adata.var["highly_variable"]])
    rank_col = ("variances_norm" if "variances_norm" in hvg_stats.columns
                else "dispersions_norm")

    hvg_stats["rescued_target"] = False
    hvg_stats["rescued_ghost"] = False

    if p8.get("rescue_perturbation_targets", True):
        ctrl_label = cfg["dataset"]["control_label"]
        is_comb = (cfg.get("dataset", {}).get("perturbation_structure") == "combinatorial")
        sep = cfg.get("dataset", {}).get("perturbation_separator", "+")
        var_set = set(var_names)
        raw_labels = set(train_labels_present) - {ctrl_label}
        targets = set()
        for lbl in raw_labels:
            if is_comb and sep and sep in str(lbl):
                for part in str(lbl).split(sep):
                    targets.add(part.strip())
            else:
                targets.add(str(lbl))

        target_rescue = (targets & var_set) - hvg_names
        if target_rescue:
            n_rescue = len(target_rescue)
            safe_to_drop = list(hvg_names - targets - set(cc_in_matrix))
            if rank_col in hvg_stats.columns and len(safe_to_drop) >= n_rescue:
                genes_to_drop = _lowest(hvg_stats, rank_col, safe_to_drop, n_rescue, tie_break)
                hvg_names.difference_update(genes_to_drop)
                hvg_names.update(target_rescue)
                hvg_stats.loc[genes_to_drop, "highly_variable"] = False
                hvg_stats.loc[list(target_rescue), "highly_variable"] = True
                hvg_stats.loc[list(target_rescue), "rescued_target"] = True
                logger.info(f"  Target rescue: swapped {n_rescue} targets")
            else:
                hvg_names.update(target_rescue)
                hvg_stats.loc[list(target_rescue), "rescued_target"] = True

    ghosts = [g for g in cc_in_matrix if g not in hvg_names]
    if ghosts:
        logger.info(f"  Ghost rescue: {len(ghosts)} cell-cycle genes stacked")
        hvg_names.update(ghosts)
        hvg_stats.loc[ghosts, "rescued_ghost"] = True

    hvg_stats["rescued_protected"] = False
    protected_path = p8.get("protected_gene_list")
    if protected_path:
        protected_ids, prot_report = load_protected_genes(protected_path, train_adata_stub, logger)
        protected_set = set(protected_ids)
        protected_rescue = (protected_set & set(var_names)) - hvg_names
        if protected_rescue:
            n_rescue = len(protected_rescue)
            already_rescued = set(hvg_stats.index[hvg_stats["rescued_target"]]) | set(ghosts)
            safe_to_drop = list(hvg_names - protected_set - set(cc_in_matrix) - already_rescued)
            if rank_col in hvg_stats.columns and len(safe_to_drop) >= n_rescue:
                genes_to_drop = _lowest(hvg_stats, rank_col, safe_to_drop, n_rescue, tie_break)
                hvg_names.difference_update(genes_to_drop)
                hvg_names.update(protected_rescue)
                hvg_stats.loc[genes_to_drop, "highly_variable"] = False
                hvg_stats.loc[list(protected_rescue), "highly_variable"] = True
                hvg_stats.loc[list(protected_rescue), "rescued_protected"] = True
            else:
                hvg_names.update(protected_rescue)
                hvg_stats.loc[list(protected_rescue), "rescued_protected"] = True

    hvg_stats["rescued_heldout"] = False
    ctrl_label_h = cfg["dataset"]["control_label"]
    sep_h = cfg.get("dataset", {}).get("perturbation_separator", "+")
    is_comb_h = cfg.get("dataset", {}).get("perturbation_structure") == "combinatorial"
    heldout_ids = load_heldout_target_ids(
        cfg["paths"]["_splits"], cfg["dataset"]["name"], var_names,
        ctrl_label_h, sep=sep_h, is_combinatorial=is_comb_h, logger=logger)
    _already_forced = set(ghosts)
    if "rescued_target" in hvg_stats.columns:
        _already_forced |= set(hvg_stats.index[hvg_stats["rescued_target"]])
    if "rescued_protected" in hvg_stats.columns:
        _already_forced |= set(hvg_stats.index[hvg_stats["rescued_protected"]])
    force_include(
        heldout_ids, hvg_names, hvg_stats, var_names, rank_col,
        do_not_drop=(set(cc_in_matrix) | _already_forced | set(heldout_ids)),
        flag_col="rescued_heldout", tag="Held-out target force-carry", logger=logger, tie_break=tie_break)

    keep_mask = np.array([g in hvg_names for g in var_names])
    core_features = [g for g in var_names if g in hvg_names and g not in set(ghosts)]
    return keep_mask, hvg_stats, core_features, hvg_names, ghosts


# ---------------------------------------------------------------------------------------------- target-node policy (XL option)
def deg_proxy_counts(permean_by_label, control_mean, keep_mask, log2fc=0.5, min_mean=0.1):
    """Number of Phase-6 genes whose mean differs from the control mean by at least `log2fc` (log2 ratio with a 0.05 pseudocount)
    and whose larger mean is at least `min_mean`. A transparent stand-in for 'number of DEGs' (no per-gene variance is stored)."""
    control = control_mean[keep_mask].astype(np.float64)
    counts = {}
    for label, mean in permean_by_label.items():
        pert = mean[keep_mask].astype(np.float64)
        fold = np.abs(np.log2((pert + 0.05) / (control + 0.05)))
        counts[label] = int(((fold >= log2fc) & (np.maximum(pert, control) >= min_mean)).sum())
    return counts


def rank_train_targets(cells_by_label, depth_by_label, degs_by_label, candidates):
    """Average of three ranks: more cells after the escaper filter, deeper knockdown (smaller fold change), more DEGs. Ties by label."""
    frame = pd.DataFrame({"label": list(candidates)})
    frame["cells"] = frame["label"].map(cells_by_label).fillna(0)
    frame["depth"] = frame["label"].map(depth_by_label)
    frame["degs"] = frame["label"].map(degs_by_label).fillna(0)
    rank_cells = frame["cells"].rank(ascending=False, method="min")
    rank_depth = frame["depth"].rank(ascending=True, method="min", na_option="bottom")
    rank_degs = frame["degs"].rank(ascending=False, method="min")
    frame["score"] = (rank_cells + rank_depth + rank_degs) / 3
    return frame.sort_values(["score", "label"], kind="stable").reset_index(drop=True)


def select_panel_with_policy(calc_adata, train_adata_stub, train_labels_present, cfg, logger, ranking_inputs=None):
    """target_node_policy all (default = SPORE, exactly select_panel) or budget. Returns select_panel's 5 outputs plus a report dict."""
    p8 = cfg.get("phase8_hvg", {})
    policy = p8.get("target_node_policy", "budget")  # v4 default (P2); SPORE v3.0 behaviour is `all`
    n_top = p8.get("n_top_genes", 5000)
    if policy not in ("all", "budget"):
        raise ValueError(f"phase8_hvg.target_node_policy must be all or budget, got {policy}")
    var_names = list(train_adata_stub.var_names)
    ctrl = cfg["dataset"]["control_label"]
    targets_present = {str(l) for l in train_labels_present if str(l) != ctrl} & set(var_names)

    if policy == "all":
        keep_mask, hvg_stats, core_features, hvg_names, ghosts = select_panel(calc_adata, train_adata_stub, train_labels_present, cfg, logger)
        report = {"target_node_policy": "all", "n_top_genes": n_top, "panel_size": int(keep_mask.sum()),
                  "panel_stacked_above_n_top": int(keep_mask.sum()) - n_top,
                  "train_targets_in_panel": len(targets_present & hvg_names)}
        return keep_mask, hvg_stats, core_features, hvg_names, ghosts, report

    # ---------------- budget
    max_nodes = p8.get("max_target_nodes")
    if max_nodes is None:
        max_nodes = int(0.7 * n_top)  # v4 default (P2): 0.7 x n_top_genes
    if ranking_inputs is None:
        raise ValueError("budget policy needs ranking inputs (cells, knockdown depth, DEG counts per train target)")
    calc_adata.X.data = calc_adata.X.data.astype(np.float32, copy=False)
    sc.pp.highly_variable_genes(calc_adata, flavor=p8.get("method", "seurat_v3"), n_top_genes=n_top, subset=False)
    stat_cols = [c for c in calc_adata.var.columns
                 if c in ("highly_variable", "means", "dispersions", "dispersions_norm", "variances", "variances_norm")]
    hvg_stats = calc_adata.var[stat_cols].copy()
    original_hvg = set(calc_adata.var_names[calc_adata.var["highly_variable"]])
    rank_col = "variances_norm" if "variances_norm" in hvg_stats.columns else "dispersions_norm"

    cc_in_matrix = _cell_cycle_in_matrix(var_names, p8)  # [] unless phase8_hvg.rescue_cell_cycle_genes (P3)
    heldout = set(load_heldout_target_ids(cfg["paths"]["_splits"], cfg["dataset"]["name"], train_adata_stub.var_names, ctrl,
                                          sep=cfg.get("dataset", {}).get("perturbation_separator", "+"),
                                          is_combinatorial=False, logger=logger))
    protected = set()
    if p8.get("protected_gene_list"):
        protected_ids, _ = load_protected_genes(p8["protected_gene_list"], train_adata_stub, logger)
        protected = set(protected_ids) & set(var_names)

    forced = set(heldout) | protected | set(cc_in_matrix)
    forced_targets = (heldout | protected) & targets_present | heldout  # held-out targets are targets by definition
    panel = set(forced)
    train_candidates = rank_train_targets(ranking_inputs["cells"], ranking_inputs["depth"], ranking_inputs["degs"],
                                          sorted(targets_present - forced))
    allowed_train = max(0, int(max_nodes) - len(forced_targets))
    room = max(0, n_top - len(panel))
    n_train_added = min(allowed_train, room, len(train_candidates))
    train_added = list(train_candidates["label"][:n_train_added])
    panel |= set(train_added)
    ranked_genes = hvg_stats[rank_col].sort_values(ascending=False, na_position="last", kind="stable").index  # ties: var order (stable)
    for gene in ranked_genes:
        if len(panel) >= n_top:
            break
        panel.add(gene)
    if len(forced) > n_top:
        logger.warning(f"  budget policy: forced genes ({len(forced)}) exceed n_top_genes ({n_top}); panel = forced genes only")

    hvg_stats["highly_variable"] = [g in panel for g in hvg_stats.index]
    hvg_stats["rescued_target"] = [g in set(train_added) and g not in original_hvg for g in hvg_stats.index]
    ghosts = [g for g in cc_in_matrix if g not in original_hvg]
    hvg_stats["rescued_ghost"] = [g in set(ghosts) for g in hvg_stats.index]
    hvg_stats["rescued_protected"] = [g in protected and g not in original_hvg for g in hvg_stats.index]
    hvg_stats["rescued_heldout"] = [g in heldout and g not in original_hvg for g in hvg_stats.index]
    keep_mask = np.array([g in panel for g in var_names])
    core_features = [g for g in var_names if g in panel and g not in set(ghosts)]
    report = {"target_node_policy": "budget", "n_top_genes": n_top, "max_target_nodes": int(max_nodes), "panel_size": int(keep_mask.sum()),
              "panel_stacked_above_n_top": int(keep_mask.sum()) - n_top, "forced_heldout_targets": len(heldout),
              "forced_protected_genes": len(protected), "forced_cell_cycle_genes": len(cc_in_matrix),
              "train_targets_candidates": int(len(train_candidates)), "train_targets_added": int(n_train_added),
              "train_targets_dropped_by_budget": int(len(train_candidates) - n_train_added),
              "target_nodes_in_panel": len(set(targets_present) & panel), "hvgs_from_variance_rank": int(len(panel) - len(forced) - n_train_added)}
    report["_details"] = {"heldout": heldout, "protected": protected, "train_added": set(train_added),
                          "train_rank": dict(zip(train_candidates["label"], train_candidates["score"]))}
    return keep_mask, hvg_stats, core_features, panel, ghosts, report
