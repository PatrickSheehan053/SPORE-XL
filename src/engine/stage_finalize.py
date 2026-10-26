"""finalize (one job): train_hybrid parts, optional merges, SPORE-format COVERAGE_REPORT, WATERFALL, aligned split JSON.

Merging (xl.merge_split_outputs: auto | always | never; xl.merge_cap_gb): the merged files are built in memory in SPORE's
order when the projected size is under the cap; parts remain the product either way. COVERAGE_REPORT.md reproduces
spore_local.final_verification_gate's text (spore_local.py:796-810); its three checks are re-implemented here because
SPORE's version reads the whole substrate into RAM.
"""
import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from xl_common import (Run, block_unit, quarantine_incomplete, register_external, require_done, task, write_new_json,
                       write_new_text)
from stage_gather import escaper_counts, total_blocks
from stage_map3 import SPLITS, load_panel_tables, part_path
from stage_metacell import metacell_part_path

import config_v3
from src.phases.phase09_normalization import normalize_split

SPLIT_RANK = {"train": 0, "val": 1, "test": 2}


def hybrid_part_path(run, block):
    return run.processed_dir / f"{run.name}_train_hybrid" / f"part_{block:04d}.h5ad"


def make_aliases(run, n_total):
    """P1: v3.0 names of renamed outputs (`_p9` for `_norm`), kept for one release. Hard links (no extra disk); symbolic link if the file
    system refuses; otherwise a warning (a copy of GBs is never made silently). manifest/aliases.tsv lists every alias."""
    import os
    if run.mode != "production" or not run.xl.get("output_aliases", True):
        return []
    pairs = []
    for split in SPLITS:
        for block in range(n_total):
            target = part_path(run, split, "p9", block)
            if target.exists():
                pairs.append((target.parent.parent / f"{run.name}_{split}_p9" / target.name, target))
        merged = run.processed_dir / f"{run.name}_{split}_norm.h5ad"
        if merged.exists():
            pairs.append((run.processed_dir / f"{run.name}_{split}_p9.h5ad", merged))
    rows = []
    for alias, target in pairs:
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.exists():
            method = "exists"
        else:
            try:
                os.link(target, alias)
                method = "hardlink"
            except OSError:
                try:
                    os.symlink(target, alias)
                    method = "symlink"
                except OSError:
                    method = "NOT CREATED (file system refuses links)"
                    run.logger.warning(f"[finalize] alias {alias.name} not created: {method}; use the _norm name")
        rows.append({"alias": run.relpath(alias), "target": run.relpath(target), "method": method})
    if rows:
        import pandas as pd
        path = run.manifest_dir / "aliases.tsv"
        if path.exists():  # aliases of merged files made later by `spore_xl.py merge`
            path = run.manifest_dir / "aliases_merged.tsv"
        if not path.exists():
            pd.DataFrame(rows).to_csv(path, sep="	", index=False)
        run.logger.info(f"[finalize] {len(rows)} v3.0-name aliases (_p9 -> _norm): {pd.Series([r['method'] for r in rows]).value_counts().to_dict()}")
    return rows


def hybrid_lognorm_path(run, block):
    return run.processed_dir / f"{run.name}_train_hybrid_lognorm" / f"part_{block:04d}.h5ad"


def lognorm_enabled(run):
    return bool(run.xl.get("write_train_hybrid_lognorm", False))


def measured_symbols(run):
    """spore_local.validate_config_against_data lines 297-304, on the SOURCE var (before harmonization)."""
    var_table = run.var_table()
    for column in ("gene_name", "gene_symbol", "gene_symbols", "feature_name", "symbol", "hgnc_symbol"):
        if column in var_table.columns:
            return set(map(str, var_table[column].astype(str)))
    source_names = (run.manifest_dir / "var_source_names.txt").read_text(encoding="utf-8").split("\n")[:-1]
    return set(map(str, source_names))


def read_parts(paths):
    pieces = [ad.read_h5ad(path) for path in paths]
    return pieces


def merged_projection_gb(run):
    total = 0
    for directory in run.processed_dir.glob(f"{run.name}_*"):
        if directory.is_dir():
            total += sum(path.stat().st_size for path in directory.glob("part_*.h5ad"))
    return total * 3 / 1e9  # lzf parts expand roughly 3x in memory


def run_finalize(run: Run):
    n_total = total_blocks(run)
    out_files = [run.reports_dir / name for name in ("COVERAGE_REPORT.md", "COVERAGE_REPORT.json", "WATERFALL.md")] + [run.manifest_dir / "aliases.tsv"]
    hybrid_files = [hybrid_part_path(run, block) for block in range(n_total)]
    lognorm_files = [hybrid_lognorm_path(run, block) for block in range(n_total)] if lognorm_enabled(run) else []
    aligned_json = run.processed_dir / f"{run.name}_allsplits_metacell_split_indices.json"
    merged_names = ([f"{run.name}_{split}_{run.kind_name(kind)}.h5ad" for split in SPLITS for kind in ("hvgraw", "p9")] +
                    [f"{run.name}_allsplits_metacell_ctrlpreserved.h5ad", f"{run.name}_train_hybrid.h5ad"] +
                    ([f"{run.name}_train_hybrid_lognorm.h5ad"] if lognorm_files else []))
    marker = run.marker("stats", "finalize", "all")
    if not marker.exists():
        quarantine_incomplete(run, "finalize", "all", out_files + hybrid_files + lognorm_files + [aligned_json]
                              + [run.processed_dir / name for name in merged_names])
    with task(run, "stats", "finalize", "all") as out:
        if out is None:
            return
        written = _finalize(run, out, n_total, hybrid_files, aligned_json, merged_names)
        register_external(out, written)


def _finalize(run, out, n_total, hybrid_files, aligned_json, merged_names):
    cfg, logger = run.cfg, run.logger
    written = []
    for block in range(n_total):
        require_done(run, "stats", "metacell", block_unit(block))
    cfg["_measured_symbols"] = measured_symbols(run)

    # ---- train_hybrid parts: split in {train, control}, exactly SPORE's rule (phase12:303)
    n_hybrid_units = 0
    for block in range(n_total):
        path = metacell_part_path(run, block)
        if not path.exists():
            continue
        part = ad.read_h5ad(path)
        keep = np.isin(part.obs["split"].astype(str).to_numpy(), ["train", "control"])
        hybrid = part[keep].copy()
        if int(hybrid.obs["split"].isin(["val", "test"]).sum()):
            raise RuntimeError("FIREWALL BREACH: val/test units in the train substrate")
        target = hybrid_part_path(run, block)
        target.parent.mkdir(parents=True, exist_ok=True)
        hybrid.write_h5ad(target, compression=run.xl.get("bucket_compression", "lzf"))
        written.append(target)
        n_hybrid_units += hybrid.n_obs
        if lognorm_enabled(run):  # P6: normalize_total(target_sum) + log1p of every unit, as FUNGI's sc_data_path expects
            lognorm = normalize_split(hybrid.copy(), cfg, logger, label=f"hybrid/b{block}")
            lognorm.uns.pop("p9_raw_sample", None)
            lognorm_target = hybrid_lognorm_path(run, block)
            lognorm_target.parent.mkdir(parents=True, exist_ok=True)
            lognorm.write_h5ad(lognorm_target, compression=run.xl.get("bucket_compression", "lzf"))
            written.append(lognorm_target)

    # ---- aligned split JSON (phase12:290-300)
    split_json = run.splits_dir / f"{run.name}_split_indices.json"
    src_meta = json.loads(split_json.read_text(encoding="utf-8"))
    groups = pd.concat([pd.read_parquet(require_done(run, "stats", "metacell", block_unit(block)) / "groups.parquet")
                        for block in range(n_total)])
    out_perturbed = set(groups.loc[groups["label"] != run.ctrl_label, "label"].astype(str))
    seed = int(run.cfg.get("phase12_metacell", {}).get("seed", cfg.get("phase7_splits", {}).get("random_seed", 42)))
    aligned = {
        "train_labels": [p for p in src_meta.get("train_labels", []) if str(p) in out_perturbed],
        "val_labels": [p for p in src_meta.get("val_labels", []) if str(p) in out_perturbed],
        "test_labels": [p for p in src_meta.get("test_labels", []) if str(p) in out_perturbed],
        "seed": int(src_meta.get("seed", seed)),
        "_source_split_json": str(split_json),
        "_note": ("Labels UNCHANGED from source: metacells keep their perturbation label. Control cells are "
                  "single-cell here and re-split proportionally by the downstream dataloader."),
    }
    with open(aligned_json, "w") as handle:
        json.dump(aligned, handle, indent=2)
    written.append(aligned_json)

    # ---- merges: xl.merge_mode ondisk (default; anndata concat_on_disk, bounded memory, rows in block order) or memory (SPORE
    #      row order, needs RAM); only when the merged file is under xl.merge_cap_gb, otherwise the parts are the product
    mode = run.xl.get("merge_split_outputs", "auto")
    merge_mode = run.xl.get("merge_mode", "ondisk")
    if merge_mode not in ("ondisk", "memory"):
        raise ValueError(f"xl.merge_mode must be ondisk or memory, got {merge_mode}")
    cap = float(run.xl.get("merge_cap_gb", 40))
    parts_gb = merged_projection_gb(run) / 3
    projected = parts_gb * (1.0 if merge_mode == "ondisk" else 3.0)
    merged = mode == "always" or (mode == "auto" and projected < cap)
    merge_line = ""
    if merged and merge_mode == "ondisk":
        written += _merge_outputs_ondisk(run, n_total, hybrid_files, [hybrid_lognorm_path(run, b) for b in range(n_total)] if lognorm_enabled(run) else [])
        merge_line = (f"merged files written with concat_on_disk (bounded memory); rows are in block order, each row carries "
                      f"xl_global_index (and xl_unit for metacell units) so consumers can restore SPORE's order")
    elif merged:
        written += _merge_outputs(run, n_total, out_perturbed)
        merge_line = "merged files written in memory in SPORE's row order"
    else:
        merge_line = (f"merged files NOT written: projected {projected:.1f} GB vs xl.merge_cap_gb {cap:.0f} GB (xl.merge_split_outputs: {mode}); "
                      f"the per-block parts under processed/ are the output")
    logger.info(f"[finalize] {merge_line}")
    make_aliases(run, n_total)

    # ---- coverage + gate (SPORE report format)
    panel_var, _, _ = load_panel_tables(run)
    stub = {"train": SimpleNamespace(var_names=pd.Index(panel_var.index))}
    coverage = config_v3.report_heldout_coverage(stub, cfg, logger)
    checks = _gate_checks(run, coverage, n_total)
    passed = all(ok for _, ok, _ in checks)
    lines = [f"# {run.name} — SPORE substrate coverage + firewall verification",
             f"_(generated by the SPORE-XL final gate)_\n",
             f"- **held-out (val+test) target node coverage: {coverage['coverage_pct']}%** "
             f"({coverage['present_as_nodes']}/{coverage['val_test_targets']})",
             f"  - val {coverage['val_coverage_pct']}% ({coverage['n_val_targets']} targets) · "
             f"test {coverage['test_coverage_pct']}% ({coverage['n_test_targets']} targets)",
             f"- missing held-out targets: {len(coverage['missing'])} "
             f"(never-measured ceiling: {len(coverage['never_measured'])}; "
             f"measured-but-filtered: {len(coverage['missing_measured'])})", ""]
    for label, ok, detail in checks:
        lines.append(f"- **{'PASS ✓' if ok else 'FAIL ✗'}** — {label}  ({detail})")
    lines += ["", f"- merge: {merge_line}", "", f"## {'✅ GRAPH-READY' if passed else '❌ NOT READY'}"]
    (run.reports_dir / "COVERAGE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    json.dump({"passed": passed, "coverage": {k: v for k, v in coverage.items() if k != "missing"},
               "checks": [{"check": label, "pass": ok, "detail": detail} for label, ok, detail in checks]},
              open(run.reports_dir / "COVERAGE_REPORT.json", "w"), indent=2)
    (run.reports_dir / "WATERFALL.md").write_text(waterfall_text(run, groups, n_hybrid_units), encoding="utf-8")
    written += [run.reports_dir / "COVERAGE_REPORT.md", run.reports_dir / "COVERAGE_REPORT.json", run.reports_dir / "WATERFALL.md"]
    write_new_json(out / "summary.json", {"graph_ready": bool(passed), "merged": bool(merged), "projected_gb": projected,
                                          "n_hybrid_units": int(n_hybrid_units)})
    logger.info(f"[finalize] {'GRAPH-READY' if passed else 'NOT READY'}")
    return written


def _gate_checks(run, coverage, n_total):
    checks = []
    missing_measured = coverage.get("missing_measured", [])
    checks.append(("coverage at measurable ceiling (missing ⊆ never-measured)", len(missing_measured) == 0,
                   f"{len(missing_measured)} measured held-out targets missing"))
    leak, n_units = 0, 0
    for block in range(n_total):
        path = hybrid_part_path(run, block)
        if path.exists():
            part = ad.read_h5ad(path, backed="r")
            leak += int(part.obs["split"].astype(str).isin(["val", "test"]).sum())
            n_units += part.n_obs
            part.file.close()
    checks.append(("train-hybrid firewall (0 val/test units)", leak == 0, f"{leak} val/test units in train hybrid"))
    ctrl_rows, n_genes = [], 0
    for block in range(run.layout()["n_blocks"], n_total):
        path = metacell_part_path(run, block)
        if path.exists() and sum(len(r) for r in ctrl_rows) < 200:
            part = ad.read_h5ad(path)
            is_ctrl = (part.obs[run.pert_col].astype(str) == str(run.ctrl_label)).to_numpy()
            if is_ctrl.any():
                ctrl_rows.append(part.X[is_ctrl][:200].toarray() if sp.issparse(part.X) else np.asarray(part.X[is_ctrl][:200]))
            n_genes = part.n_vars
    if ctrl_rows:
        block_x = np.vstack(ctrl_rows)[:200]
        nz = block_x[block_x > 0]
        near_int = bool(nz.size == 0 or np.mean(np.abs(nz - np.round(nz)) < 1e-4) > 0.98)
        maxval = float(nz.max()) if nz.size else 0.0
        ok = near_int and (maxval > 10.0 or nz.size == 0)
        detail = f"{n_genes} genes; control units integer={near_int}; max={maxval:.1f}"
    else:
        ok, detail = False, "allsplits substrate not found"
    checks.append(("substrate is HVG-panel RAW counts", ok, detail))
    return checks


def waterfall_text(run, groups, n_hybrid_units):
    reduce1 = require_done(run, "stats", "reduce1", "all")
    reduce2 = require_done(run, "stats", "reduce2", "all")
    qc = json.loads((reduce1 / "waterfall_phase2.json").read_text(encoding="utf-8"))
    counts = escaper_counts(run)
    n_qc = int(counts["n_total"].sum()) + int(pd.read_csv(reduce1 / "labelcounts_qc.tsv", sep="\t", keep_default_na=False)
                                              .query("label == @run.ctrl_label")["n_qc_pass"].sum())
    escapers = int((counts["n_total"] - counts["n_kept"]).sum())
    dropped = pd.read_csv(reduce2 / "dropped_undersized.tsv", sep="\t", keep_default_na=False)
    summary = json.loads((reduce2 / "summary.json").read_text(encoding="utf-8"))
    start = int(qc["Starting cells"])
    removed_qc = start - int(qc["After max counts"])
    after_escape = n_qc - escapers
    after_undersized = after_escape - int(dropped["n_kept"].sum())
    reconciles = (after_undersized == summary["n_survivors"]) and (n_qc == int(qc["After max counts"]))
    lines = ["# WATERFALL (cells)", "", "| step | cells | removed |", "|---|---|---|",
             f"| in | {start:,} | |"]
    previous = start
    for step in ("After MT filter", "After min genes", "After max genes", "After min counts", "After max counts"):
        lines.append(f"| QC: {step} | {int(qc[step]):,} | {previous - int(qc[step]):,} |")
        previous = int(qc[step])
    lines += [f"| escaper filter | {after_escape:,} | {escapers:,} |",
              f"| undersized perturbations dropped ({len(dropped)} perturbations) | {after_undersized:,} | {int(dropped['n_kept'].sum()):,} |",
              f"| survivors to Phase 7 | {summary['n_survivors']:,} | |", "",
              f"cells in = survivors + removed: {start:,} = {summary['n_survivors']:,} + {removed_qc:,} (QC) + {escapers:,} "
              f"(escapers) + {int(dropped['n_kept'].sum()):,} (undersized) -> {'RECONCILES' if reconciles else 'DOES NOT RECONCILE'}",
              "", f"metacell units: {int(groups['n_units'].sum()):,} ({n_hybrid_units:,} in train_hybrid)",
              f"perturbation groups using the fallback embedding: {int(groups['used_fallback'].sum())}"]
    if not reconciles:
        raise RuntimeError("WATERFALL does not reconcile: " + "\n".join(lines))
    return "\n".join(lines) + "\n"


def _merge_outputs(run, n_total, out_perturbed):
    written = []
    hvg_stats = pd.read_parquet(require_done(run, "stats", "hvg", "all") / "hvg_stats.parquet")
    core = [n for n in (require_done(run, "stats", "hvg", "all") / "core_features.txt").read_text(encoding="utf-8").split("\n") if n]
    for split in SPLITS:
        for kind in ("hvgraw", "p9"):
            parts = [ad.read_h5ad(part_path(run, split, kind, block)) for block in range(n_total)
                     if part_path(run, split, kind, block).exists()]
            matrix = sp.vstack([p.X for p in parts], format="csr")
            obs = pd.concat([p.obs for p in parts])
            order = np.argsort(obs["xl_global_index"].to_numpy(), kind="stable")
            merged = ad.AnnData(X=matrix[order], obs=obs.iloc[order], var=parts[0].var.copy())
            if split == "train" and kind == "hvgraw":
                merged.uns["hvg_stats"] = hvg_stats.to_dict("index")
                merged.uns["spore_core_features"] = core
            path = run.processed_dir / f"{run.name}_{split}_{run.kind_name(kind)}.h5ad"
            merged.write_h5ad(path)
            written.append(path)
    # allsplits metacell in SPORE's order: sorted labels; controls at their sorted slot, train then val then test controls
    parts = [ad.read_h5ad(metacell_part_path(run, block)) for block in range(n_total) if metacell_part_path(run, block).exists()]
    matrix = sp.vstack([p.X for p in parts], format="csr")
    obs = pd.concat([p.obs for p in parts])
    obs["_label"] = obs[run.pert_col].astype(str)
    obs["_rank"] = obs["xl_control_split"].astype(str).map(SPLIT_RANK).fillna(0).astype(int)
    obs["_row"] = np.arange(len(obs))
    ordered = obs.sort_values(["_label", "_rank", "xl_global_index", "xl_unit"], kind="stable")
    all_obs = ordered.drop(columns=["_label", "_rank", "_row"])
    all_obs[run.pert_col] = all_obs[run.pert_col].astype(str)
    allsplits = ad.AnnData(X=matrix[ordered["_row"].to_numpy()], obs=all_obs, var=parts[0].var.copy())
    n_ctrl = int((all_obs[run.pert_col] == run.ctrl_label).sum())
    allsplits.uns["metacell_metadata"] = {
        "recipe": "control_preserving_mbk", "k_perturbed": int(run.cfg.get("phase12_metacell", {}).get("mbk_k", 2)),
        "control_single_cell": True, "splits": "all (train+val+test), within-perturbation",
        "n_units": int(allsplits.n_obs), "n_ctrl_units": n_ctrl, "n_genes": int(allsplits.n_vars),
        "built_by": "SPORE-XL (parts merged in SPORE order)"}
    path = run.processed_dir / f"{run.name}_allsplits_metacell_ctrlpreserved.h5ad"
    allsplits.write_h5ad(path)
    written.append(path)
    hybrid = allsplits[np.isin(allsplits.obs["split"].astype(str).to_numpy(), ["train", "control"])].copy()
    if int(hybrid.obs["split"].isin(["val", "test"]).sum()):
        raise RuntimeError("FIREWALL BREACH in merged train_hybrid")
    path = run.processed_dir / f"{run.name}_train_hybrid.h5ad"
    hybrid.write_h5ad(path)
    written.append(path)
    if lognorm_enabled(run):
        lognorm = normalize_split(hybrid.copy(), run.cfg, run.logger, label="hybrid")
        lognorm.uns.pop("p9_raw_sample", None)
        path = run.processed_dir / f"{run.name}_train_hybrid_lognorm.h5ad"
        lognorm.write_h5ad(path)
        written.append(path)
    return written


def _merge_outputs_ondisk(run, n_total, hybrid_files, lognorm_files=()):
    """anndata.experimental.concat_on_disk over the parts (files in block order); peak RAM is bounded by max_loaded_elems and the
    obs tables, not by the matrix size."""
    from anndata.experimental import concat_on_disk
    import anndata.io as anndata_io
    import h5py
    elements = int(run.xl.get("merge_max_loaded_elems", 20000000))
    jobs = []
    for split in SPLITS:
        for kind in ("hvgraw", "p9"):
            files = [part_path(run, split, kind, block) for block in range(n_total) if part_path(run, split, kind, block).exists()]
            jobs.append((files, run.processed_dir / f"{run.name}_{split}_{run.kind_name(kind)}.h5ad"))
    jobs.append(([metacell_part_path(run, b) for b in range(n_total) if metacell_part_path(run, b).exists()],
                 run.processed_dir / f"{run.name}_allsplits_metacell_ctrlpreserved.h5ad"))
    jobs.append(([path for path in hybrid_files if path.exists()], run.processed_dir / f"{run.name}_train_hybrid.h5ad"))
    if lognorm_files:
        jobs.append(([path for path in lognorm_files if path.exists()], run.processed_dir / f"{run.name}_train_hybrid_lognorm.h5ad"))
    hvg_stats = pd.read_parquet(require_done(run, "stats", "hvg", "all") / "hvg_stats.parquet")
    core = [n for n in (require_done(run, "stats", "hvg", "all") / "core_features.txt").read_text(encoding="utf-8").split(chr(10)) if n]
    written = []
    for files, out_path in jobs:
        concat_on_disk([str(path) for path in files], str(out_path), axis=0, join="inner", max_loaded_elems=elements)
        written.append(out_path)
        if out_path.name.endswith("_train_hvgraw.h5ad"):  # SPORE keeps these in the train hvgraw uns
            with h5py.File(out_path, "r+") as handle:
                anndata_io.write_elem(handle, "uns", {"hvg_stats": hvg_stats.to_dict("index"), "spore_core_features": core})
    return written
