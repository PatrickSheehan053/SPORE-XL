"""SPORE v3.0 config defaults and loader, copied VERBATIM from spore_local.py at SPORE commit 5015dc6 (tag v3.0).

The engine needs three things from spore_local.py and nothing else: the import-time thread / allocator environment defaults, load_config (the config
merged over _DEFAULTS, paths resolved, run folders created) and report_heldout_coverage. spore_local.py itself (the v3.0 runner) is not part of
SPORE-XL. Every piece below is a character-for-character copy; tests/test_spore_api.py hashes each one against tests/spore_5015dc6_config_segments.json
and, when XL_SPORE_V3_DIR points at a v3.0 checkout, against the real file. Do not edit: re-copy from the v3.0 tag instead.
"""
import os
from pathlib import Path

import yaml


# ══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT SETUP  (spore_local.py lines 55-61; set at import, before the numeric libraries are used)
# ══════════════════════════════════════════════════════════════════════════════

os.environ.setdefault("OMP_NUM_THREADS",          "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",     "1")
os.environ.setdefault("MKL_NUM_THREADS",          "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS",   "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",      "1")
os.environ.setdefault("MALLOC_ARENA_MAX",         "2")
os.environ.setdefault("MALLOC_MMAP_THRESHOLD_",   "134217728")


# ---- spore_local.py: _DEFAULTS
_DEFAULTS: dict = {
    "runtime": {
        "n_jobs":                   6,
        "sparse_on_load":           True,
        "memory_monitor":           True,
        "large_dataset_mode":       "auto",
        "large_dataset_threshold":  500_000,  # cells above this use subprocess paths
        "checkpointing":            True,
    },
    "phase0_ingestion":  {"chunk_size": 50_000},
    "phase1_detection": {
        "enabled":                  True,
        "modality_detection":       False,
        "gene_id_harmonization":    True,
        "combinatorial_detection":  False,
        "cell_line_detection":      False,
        "ensembl_prefixes": {
            "human":     "ENSG",
            "mouse":     "ENSMUSG",
            "zebrafish": "ENSDARG",
        },
        "combinatorial_min_fraction": 0.05,
    },
    "phase2_cell_triage": {
        "mt_method":          "explicit_list",
        "mt_threshold":       0.20,
        "mt_genes": [
            "MT-ND1","MT-ND2","MT-ND3","MT-ND4","MT-ND4L","MT-ND5","MT-ND6",
            "MT-CO1","MT-CO2","MT-CO3","MT-ATP6","MT-ATP8","MT-CYB",
        ],
        "ribo_prefixes":       ["RPL", "RPS"],
        "min_genes_per_cell":  200,
        "max_genes_per_cell":  10000,
        "min_counts_per_cell": 500,
        "max_counts_per_cell": 80000,
    },
    "phase3_ambient":  {"enabled": False},
    "phase4_doublets": {"enabled": False},
    "phase5_escaper_filtering": {
        "efficiency_threshold":        0.50,
        "escaper_percentile":          10,
        "min_cells_per_perturbation":  50,
        "direction":                   "auto",
    },
    "phase6_gene_triage": {
        "min_cells_expressing":              10,
        "pct_filter":                        0.01,
        "mean_umi_threshold":                0.25,
        "rescue_perturbation_targets":       True,
        "rescue_combinatorial_constituents": True,
    },
    "phase7_splits": {
        "mode":          "zero_shot",
        "random_seed":   42,
        "test_mode":     False,
        "stratify_bins": 4,
        "test_n":        20,
        "val_n":         20,
        "test_ratio":    0.15,
        "val_ratio":     0.10,
    },
    "phase8_hvg": {
        "enabled":                    True,
        "n_top_genes":                1000,
        "method":                     "seurat_v3",
        "rescue_perturbation_targets": True,
        "compute_aware_budget":        True,
        "protected_gene_list":         None,
    },
    "phase9_normalization": {
        "target_sum":    10000,
        "log_transform": True,
        "method":        "lognorm",
    },
    "phase10_confounders": {
        "cell_cycle_regression": True,
        "s_genes":               "tirosh2016",
        "g2m_genes":             "tirosh2016",
        "batch_correction": {
            "enabled":                   True,
            "method":                    "harmony",
            "batch_key":                 "batch",
            "confounding_ari_threshold": 0.5,
            "min_cells_per_batch":       50,
        },
        "imputation_prohibited": True,
        "fit_on_train_only":     True,
    },
    "phase11_cell_line": {
        "enabled":               "false",
        "cell_line_col":         None,
        "expected_n_cell_lines": None,
        "auto_detect": {
            "enabled":            True,
            "max_k_to_try":       15,
            "n_bootstrap":        50,
            "bootstrap_frac":     0.80,
            "min_stability_ari":  0.85,
            "min_bic_improvement": 50.0,
        },
        "min_cells_per_cell_line":    500,
        "output_single_cell_splits":  False,
    },
    "phase12_metacell": {
        "enabled":                     True,
        "target_cells_per_metacell":   10,
        "min_metacells_per_pert":      5,
        "min_cells_per_metacell":      3,
        "systema_calibration":         True,
        "compute_quality_metrics":     True,
        "inner_variance_warn_threshold": 2.0,
        "suffix":                      "_metacell",
    },
    "plotting": {
        "style":         "dark",
        "dpi":           150,
        "save_figures":  False,
        "figure_format": "png",
    },
    "output": {
        "cleanup_intermediates": False,
        "generate_report":       True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_paths(cfg: dict) -> dict:
    root = Path(cfg["paths"].get("project_root") or ".")
    cfg["paths"]["_root"]          = root
    raw = cfg["paths"].get("raw_h5ad") or ""
    cfg["paths"]["_raw_h5ad"]      = root / raw if raw else None
    cfg["paths"]["_processed"]     = root / cfg["paths"]["processed_dir"]
    cfg["paths"]["_splits"]        = root / cfg["paths"]["splits_dir"]
    cfg["paths"]["_figures"]       = root / cfg["paths"].get("figures_dir",       "figures")
    cfg["paths"]["_logs"]          = root / cfg["paths"].get("log_dir",           "logs")
    # CHITIN (Phase 13) removed — no chitin_output_dir.
    for key in ["_processed", "_splits", "_figures", "_logs"]:
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    return cfg


def _apply_light_overrides(cfg: dict) -> dict:
    # save_figures is controlled by the YAML, not forced off here.
    # Express mode forced it off (no display on HPC), but on the 2070
    # we want plots saved to the figures directory.
    cfg.setdefault("runtime", {}).setdefault("checkpointing", True)
    # Phase 13 section intentionally absent — CHITIN is a separate tool
    return cfg


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _apply_light_overrides(_resolve_paths(_deep_merge(_DEFAULTS, raw)))
    return cfg


# ---- spore_local.py: held-out coverage
def _split_targets(labels, ctrl, sep, is_comb):
    out = set()
    for lbl in labels:
        if str(lbl) == str(ctrl):
            continue
        if is_comb and sep and sep in str(lbl):
            out.update(p.strip() for p in str(lbl).split(sep) if p.strip())
        else:
            out.add(str(lbl))
    return out


def report_heldout_coverage(splits: dict, cfg: dict, logger) -> dict:
    """
    Defect 2/7: report how many held-out (val+test) perturbation-target genes are present as NODES in the
    HVG panel, plus the perturbed-but-never-measured ceiling. missing_measured should be 0 once the Phase-8
    force-carry works (that is the whole point). Loud, not silent.
    """
    import json as _json
    splits_dir = cfg["paths"]["_splits"]
    name       = cfg["dataset"]["name"]
    ds         = cfg["dataset"]
    ctrl       = ds["control_label"]
    sep        = ds.get("perturbation_separator", "+")
    is_comb    = ds.get("perturbation_structure") == "combinatorial"

    train = splits.get("train")
    panel = set(map(str, train.var_names)) if train is not None else set()

    sj = splits_dir / f"{name}_split_indices.json"
    meta = _json.load(open(sj))
    va = _split_targets(meta.get("val_labels", []),  ctrl, sep, is_comb)
    te = _split_targets(meta.get("test_labels", []), ctrl, sep, is_comb)
    heldout = va | te
    present = {g for g in heldout if g in panel}
    missing = sorted(heldout - present)

    measured        = cfg.get("_measured_symbols", set())
    never_measured  = sorted(g for g in missing if measured and g not in measured)
    missing_measured = sorted(g for g in missing if measured and g in measured)

    cov    = 100.0 * len(present) / max(1, len(heldout))
    va_cov = 100.0 * len({g for g in va if g in panel}) / max(1, len(va))
    te_cov = 100.0 * len({g for g in te if g in panel}) / max(1, len(te))

    logger.info("  ── HELD-OUT TARGET COVERAGE (zero-shot node coverage) ──")
    logger.info(f"    val+test targets present as nodes: {len(present)}/{len(heldout)} ({cov:.1f}%)")
    logger.info(f"    val {va_cov:.1f}% ({len(va)} targets) · test {te_cov:.1f}% ({len(te)} targets)")
    logger.info(f"    missing: {len(missing)}  (never-measured ceiling: {len(never_measured)}; "
                f"measured-but-filtered: {len(missing_measured)})")
    if missing_measured:
        logger.warning(f"    ⚠ {len(missing_measured)} MEASURED held-out targets are NOT nodes "
                       f"(force-carry should make this 0): {missing_measured[:15]}")

    return dict(
        val_test_targets=len(heldout), present_as_nodes=len(present), missing=missing,
        coverage_pct=round(cov, 2), val_coverage_pct=round(va_cov, 2), test_coverage_pct=round(te_cov, 2),
        never_measured=never_measured, missing_measured=missing_measured,
        n_val_targets=len(va), n_test_targets=len(te))
