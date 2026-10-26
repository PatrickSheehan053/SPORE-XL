"""plan: shards, harmonized var table (SPORE Phase 1, once), label union and ribosomal exemption (Phase 2),
control-block cut points, global_obs skeleton, environment record. Reads obs only; no matrix values.
"""
import json
import shutil
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import diagnostics_xl
import shard_planner
import preflight
from xl_common import (_PeakSampler, Run, environment_text, quarantine_incomplete, write_new_json, write_new_parquet, write_new_text,
                       write_new_tsv)

from src.phases.phase01_detection import run_phase1
from src.phases.phase05_escaper_filtering import _get_escaper_direction
from src.phases.utils import load_protected_genes


UNSUPPORTED_COMBINATORIAL = ("combinatorial / multi-target perturbations are not supported in SPORE-XL v1.0 (blueprint: out of scope). "
                             "Remove those cells in intake or run SPORE v3.0 on this dataset.")
UNSUPPORTED_ACTIVATION = ("CRISPRa / activation escaper direction is not supported in SPORE-XL v1.0 (only CRISPRi knockdown has been "
                          "validated against SPORE). Set dataset.perturbation_type to CRISPRi or run SPORE v3.0.")


def check_supported(run, labels):
    """v1.0 scope: single-target CRISPRi. Multi-target labels and the activation direction are refused loudly."""
    separator = run.cfg["dataset"].get("perturbation_separator", "+") or "+"
    multi = sorted({label for label in set(labels) if separator in label and label != run.ctrl_label})
    if run.cfg["dataset"].get("perturbation_structure") == "combinatorial" or multi:
        raise NotImplementedError(f"{UNSUPPORTED_COMBINATORIAL} Labels containing '{separator}': {multi[:5]}")
    if _get_escaper_direction(run.cfg, run.logger) == "activation":
        raise NotImplementedError(UNSUPPORTED_ACTIVATION)


def _plan_marker(run):
    return run.manifest_dir / "plan.done"


def run_plan(run: Run):
    with _PeakSampler() as sampler:
        started = time.time()
        _run_plan(run)
        elapsed = time.time() - started
    if elapsed < 1.0 and (run.manifest_dir / "plan.done").exists():
        return  # skipped re-run: nothing to log
    log_path = run.reports_dir / "memory_and_time.tsv"
    already_logged = log_path.exists() and any(
        line.startswith("plan\t") for line in log_path.read_text(encoding="utf-8").split("\n"))
    if not already_logged:
        with open(log_path, "a", encoding="utf-8") as handle:
            if log_path.stat().st_size == 0:
                handle.write("stage\tunit\tseconds\tpeak_rss_gb\n")
            handle.write(f"plan\tall\t{elapsed:.2f}\t{sampler.peak:.3f}\n")


def _run_plan(run: Run):
    logger = run.logger
    marker = _plan_marker(run)
    if (run.cfg.get("phase8_hvg", {}) or {}).get("protected_expression_gate") and not (run.manifest_dir / "protected_genes_gated.json").exists() and not marker.exists():
        raise RuntimeError("phase8_hvg.protected_expression_gate is on but the gate has not run (manifest/protected_genes_gated.json is missing): use `spore_xl.py run`, "
                           "which runs the gate after the intake and before plan")
    if marker.exists():
        logger.info("[plan] already done, skipped")
        return
    # an interrupted plan leaves some manifest files, config/environment.txt and maybe half-written source shards: move them aside
    plan_files = ("shards.tsv", "layout.json", "var.parquet", "var_source_names.txt", "cfg_runtime.json", "labels.tsv", "ribo_exempt.txt",
                  "global_obs.parquet", "source.json")  # not intake_members.tsv / intake_list.done: those belong to the intake stages
    leftovers = [run.manifest_dir / name for name in plan_files if (run.manifest_dir / name).exists()]
    leftovers += [path for path in [run.config_dir / "environment.txt"] if path.exists()]
    source_shards = run.work_dir / "source_shards"
    if source_shards.exists():
        leftovers += sorted(source_shards.glob("*.tmp.h5ad"))
        leftovers += sorted(path for path in source_shards.glob("shard_*.h5ad") if not path.with_name(path.name + ".done").exists())
    quarantine_incomplete(run, "plan", "all", leftovers)

    source = run.xl.get("source", {})
    kind = source.get("kind")
    if kind not in ("h5ad_rows", "h5ad_obs", "recipe"):
        raise NotImplementedError(f"xl.source.kind '{kind}' is not supported")
    obs_key_values = None
    if kind == "recipe":
        import stage_intake
        info = stage_intake.recipe_inputs(run)  # intake must have finished: shards, labels, var, category order
        n_obs, n_vars, x_dtype, labels = info["n_obs"], info["n_vars"], info["x_dtype"], info["labels"]
        category_order, categorical, source_var = info["category_order"], True, info["source_var"]
        batch_col, donor_col, batch_values, donor_values = info["batch_col"], info["donor_col"], info["batch_values"], info["donor_values"]
        h5ad_path = None
    else:
        h5ad_path = Path(source["h5ad"])
        backed = ad.read_h5ad(h5ad_path, backed="r")
        try:
            n_obs, n_vars = int(backed.n_obs), int(backed.n_vars)
            x_dtype = str(backed.X.dtype)
            obs_labels = backed.obs[run.pert_col]
            if isinstance(obs_labels.dtype, pd.CategoricalDtype):
                category_order = [str(value) for value in obs_labels.cat.categories]
                categorical = True
            else:
                category_order = sorted(set(obs_labels.astype(str)))
                categorical = False
            labels = obs_labels.astype(str).to_numpy()
            obs_key_values = backed.obs[source["obs_key"]].astype(str).to_numpy() if kind == "h5ad_obs" else None
            source_var = backed.var.copy()
            batch_col, donor_col = diagnostics_xl.diagnostic_columns(backed.obs.columns, run.cfg)
            batch_values = backed.obs[batch_col].astype(str).to_numpy() if batch_col else None
            donor_values = backed.obs[donor_col].astype(str).to_numpy() if donor_col else None
        finally:
            backed.file.close()

    # --- refuse what v1.0 does not support, BEFORE anything is written (out of scope, never silently mishandled)
    check_supported(run, labels)
    if h5ad_path is not None:  # refuse normalized / log-transformed / scaled input now, before a single shard is written (stress suite 11)
        from src.intake import intake_adapters
        problem = intake_adapters.sample_rows_check_h5ad(h5ad_path)
        if problem:
            raise intake_adapters.NotRawCountsError(
                problem + f". Source: {h5ad_path}. SPORE needs raw integer counts (recipe source.x_key can point at layers/counts or raw/X; "
                          f"`spore_xl.py convert` does this for you).")

    # --- Phase 1 once, on a zero-row AnnData that carries the source var (in memory; written after the checks)
    var_adata = ad.AnnData(X=sp.csr_matrix((0, n_vars), dtype=np.float32), var=source_var)
    run.cfg["dataset"].setdefault("perturbation_col", run.pert_col)
    var_adata, detection = run_phase1(var_adata, run.cfg, logger)
    if run.cfg["dataset"].get("perturbation_structure") == "combinatorial":
        raise NotImplementedError(UNSUPPORTED_COMBINATORIAL)

    # --- pre-flight: disk for the working files, RAM for one shard (never delete, never start what cannot finish)
    if kind == "recipe":
        preflight.check_disk_for_nnz(run, info["total_nnz"])
        preflight.check_shard_memory(run, info["largest_cells"], info["nnz_per_cell"])
        table, planner_source = info["table"], info["planner_source"]
    else:
        preflight.check_disk(run, h5ad_path)
        facts = shard_planner.measure_h5ad(h5ad_path)
        if kind == "h5ad_rows":
            planned_rows = source.get("shard_rows") or shard_planner.derive_shard_rows(
                run.xl.get("memory_budget_gb", 24), facts["nnz_per_cell"])
            largest = min(int(planned_rows), n_obs)
        else:
            largest = int(pd.Series(obs_key_values).value_counts().max())
        preflight.check_shard_memory(run, largest, facts["nnz_per_cell"])

        shard_root = run.work_dir / "source_shards"
        if kind == "h5ad_rows":
            table = shard_planner.plan_h5ad_rows(
                h5ad_path, shard_root, shard_rows=source.get("shard_rows"),
                memory_budget_gb=run.xl.get("memory_budget_gb"), pert_col=run.pert_col, workers=int(run.xl.get("plan_workers", 1)))
        else:
            table = shard_planner.plan_h5ad_obs(h5ad_path, shard_root, obs_key=source["obs_key"], pert_col=run.pert_col,
                                                workers=int(run.xl.get("plan_workers", 1)))
        table = table.copy()
        table["file"] = [run.relpath(shard_root / name) for name in table["file"]]
        planner_source = json.loads((shard_root / "manifest" / "source.json").read_text(encoding="utf-8"))
    write_new_tsv(run.manifest_dir / "shards.tsv", table)

    var_table = var_adata.var.copy()
    var_table.index = var_table.index.astype(str)
    write_new_parquet(run.manifest_dir / "var.parquet", var_table)
    write_new_text(run.manifest_dir / "var_source_names.txt", "\n".join(map(str, source_var.index)) + "\n")
    runtime = {"dataset": {key: run.cfg["dataset"][key] for key in
                           ("gene_id_format", "perturbation_structure", "perturbation_separator")
                           if key in run.cfg["dataset"]},
               "_detection_overrides": run.cfg.get("_detection_overrides", {})}
    write_new_json(run.manifest_dir / "cfg_runtime.json", runtime)
    run.reload_runtime_updates()

    # --- label union over ALL cells before QC; ribosomal exemption exactly as phase02:111-127
    counts = pd.Series(labels).value_counts()
    labels_frame = pd.DataFrame({"label": counts.index.astype(str), "n_cells_raw": counts.to_numpy()})
    labels_frame = labels_frame.sort_values("label").reset_index(drop=True)
    write_new_tsv(run.manifest_dir / "labels.tsv", labels_frame)
    exempt = set()
    for label in pd.unique(pd.Series(labels).astype(str)):
        if label != run.ctrl_label:
            exempt.update(str(label).split("+"))
    protected_path = run.cfg.get("phase8_hvg", {}).get("protected_gene_list")
    if protected_path:
        try:
            protected_ids, _ = load_protected_genes(protected_path, var_adata, logger)
            exempt.update(protected_ids)
        except Exception as error:  # noqa: BLE001  (SPORE only warns here)
            logger.warning(f"  Ribosomal exempt: could not load protected list ({type(error).__name__}: {error})")
    write_new_text(run.manifest_dir / "ribo_exempt.txt", "\n".join(sorted(exempt)) + "\n")

    # --- control blocks: cut points on the global index so a control block is a contiguous stretch of controls
    control_index = np.flatnonzero(labels == run.ctrl_label)
    cap = run.control_block_rows
    cuts = [int(value) for value in control_index[cap::cap]] if len(control_index) > cap else []
    n_blocks = run.n_blocks

    # --- global_obs skeleton
    shard_of = np.full(n_obs, -1, dtype=np.int32)
    for row in table.itertuples():
        if kind in ("h5ad_rows", "recipe"):
            shard_of[int(row.row_start):int(row.row_stop)] = row.shard
        else:
            shard_of[obs_key_values == str(row.obs_value)] = row.shard
    assert (shard_of >= 0).all()
    global_obs = pd.DataFrame({
        "xl_global_index": np.arange(n_obs, dtype=np.int64), "xl_shard": shard_of,
        "label": pd.Categorical(labels, categories=category_order)})
    if batch_values is not None:  # P5: metadata for the confounding table and variance-explained diagnostics
        global_obs["batch"] = pd.Categorical(batch_values)
    if donor_values is not None:
        global_obs["donor"] = pd.Categorical(donor_values)
    write_new_parquet(run.manifest_dir / "global_obs.parquet", global_obs)

    write_new_json(run.manifest_dir / "layout.json", {
        "kind": kind, "n_obs": n_obs, "n_vars": n_vars, "x_dtype": x_dtype, "n_blocks": n_blocks,
        "control_block_rows": cap, "control_cuts": cuts, "n_control_blocks": len(cuts) + 1,
        "n_control_cells_raw": int(len(control_index)), "pert_col": run.pert_col, "ctrl_label": run.ctrl_label,
        "gene_id_format": run.cfg["dataset"].get("gene_id_format"), "n_labels": int(len(labels_frame)),
        "batch_col": batch_col, "donor_col": donor_col,
        "perturbation_categorical": categorical})
    write_new_json(run.manifest_dir / "source.json", {
        **planner_source, "xl_kind": kind, "x_dtype": x_dtype, "perturbation_category_order": category_order,
        "category_order_note": "source categories if categorical, else sorted unique labels"})
    write_new_text(run.config_dir / "environment.txt", environment_text())
    if not (run.config_dir / "run_config.yaml").exists():
        shutil.copyfile(run.config_path, run.config_dir / "run_config.yaml")
    write_new_text(marker, "plan finished\n")
    logger.info(f"[plan] {len(table)} shards, {n_obs:,} cells, {n_vars:,} genes, {len(labels_frame):,} labels, "
                f"{len(cuts) + 1} control block(s), {len(exempt):,} ribosomal-exempt names")

