"""report (one job, after finalize): SPORE's figure set drawn post hoc from XL's saved statistics, plus reports/REPORT.md.

Figures (same file names as SPORE's run folder): sparsity_saturation, p02_qc_violin, p02_mt_scatter, p05_efficiency_scatter,
p06_triage_dashboard, p07_splits_dashboard, p08_hvg_dashboard, p09_normalization. Phase 10 (p10_*) is NOT produced: Phase 10 is
out of scope in v1.0. Every figure carries a corner label saying whether it is drawn from all cells or from a subsample of N.
SPORE's own plotting functions are called on small stand-in objects (obs table, uns dict); only the sparsity overview is cloned
(adapted_phases/plotting_xl.py) because SPORE derives its arrays from the matrix. Figures are saved into <run>/figures/.
"""
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xl_common import Run, block_unit, quarantine_incomplete, require_done, shard_unit, task, write_new_text  # noqa: E402
from stage_gather import escaper_counts, total_blocks  # noqa: E402
from stage_map3 import SPLITS, part_path  # noqa: E402
from stats import GeneSums  # noqa: E402

from adapted_phases.plotting_xl import plot_sparsity_overview_arrays  # noqa: E402
import src.report.plotting as spore_plotting  # noqa: E402

# (v4 content name, v3.0 name kept as an alias for one release, what). Phase numbers are the v4 numbers (see PHASE_MAP).
FIGURES = [("sparsity_saturation", "sparsity_saturation", "Input sparsity overview (phase 1-2)"),
           ("cell_qc_violin", "p02_qc_violin", "Phase 3 cell QC violins"),
           ("cell_qc_mt_scatter", "p02_mt_scatter", "Phase 3 cell QC MT scatter"),
           ("escaper_efficiency_scatter", "p05_efficiency_scatter", "Phase 4 escaper filter / perturbation efficiency"),
           ("gene_qc_dashboard", "p06_triage_dashboard", "Phase 5 gene QC"),
           ("split_dashboard", "p07_splits_dashboard", "Phase 6 split"),
           ("hvg_dashboard", "p08_hvg_dashboard", "Phase 7 HVG panel"),
           ("normalization", "p09_normalization", "Phase 8 normalization")]
OLD_NAME = {new: old for new, old, _ in FIGURES}

# v4 phase number -> (name, v3.0 phase number, config section kept from v3.0)
PHASE_MAP = [(1, "Intake / formatting", "new", "xl.source / recipe"), (2, "Gene-ID harmonization", "1", "phase1_detection"),
             (3, "Cell QC", "2", "phase2_cell_triage"), (4, "Escaper filter + perturbation triage", "5", "phase5_escaper_filtering"),
             (5, "Gene QC", "6", "phase6_gene_triage"), (6, "Split", "7", "phase7_splits"), (7, "HVG panel", "8", "phase8_hvg"),
             (8, "Normalization", "9", "phase9_normalization"), (9, "Metacell substrate", "12", "phase12_metacell"),
             (10, "Verification + report", "new", "-")]


def _stamp(fig, text):
    fig.text(0.995, 0.003, text, ha="right", va="bottom", fontsize=8, color="#555555", style="italic")


def _save(fig, run, stem, label):
    _stamp(fig, label)
    path = run.run_dir / "figures" / f"{stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    return path


def _alias_copy(run, stem):
    """v3.0 file name of a figure (P1: kept for one release); a copy, figures are small."""
    import shutil
    old = OLD_NAME[stem]
    source, target = run.run_dir / "figures" / f"{stem}.png", run.run_dir / "figures" / f"{old}.png"
    if old != stem and source.exists() and not target.exists():
        shutil.copyfile(source, target)
        return target
    return None


def _qc_table(run):
    frames = [pd.read_parquet(require_done(run, "stats", "map1", shard_unit(i)) / "qc_cells.parquet") for i in range(run.n_shards())]
    return pd.concat(frames, ignore_index=True)


def _phase9_cache(run, per_split=50000, max_parts=20, seed=42):
    rng = np.random.default_rng(seed)
    raw, norm = [], []
    n_parts_used = 0
    for split in SPLITS:
        candidates = [b for b in range(total_blocks(run)) if part_path(run, split, "p9", b).exists()]
        if not candidates:
            continue
        picked = rng.choice(candidates, size=min(max_parts, len(candidates)), replace=False)
        raw_values, norm_values = [], []
        for block in sorted(picked):
            p9 = ad.read_h5ad(part_path(run, split, "p9", block))
            hvgraw = ad.read_h5ad(part_path(run, split, "hvgraw", block))
            p9_data, raw_data = sp.csr_matrix(p9.X).data, sp.csr_matrix(hvgraw.X).data
            take = rng.choice(len(p9_data), size=min(per_split // len(picked) + 1, len(p9_data)), replace=False)
            raw_values.append(raw_data[take])
            norm_values.append(p9_data[take])
            n_parts_used += 1
        raw.extend(np.concatenate(raw_values)[:per_split])
        norm.extend(np.concatenate(norm_values)[:per_split])
    return {"raw": raw, "norm": norm}, len(raw), n_parts_used


def _escaper_lines(run):
    """P4: how many cells each escaper rule flags (rule 1 target-expression percentile, rule 2 response score), side by side."""
    mode = run.escaper_mode
    lines = [f"## Escaper rules (phase 4; escaper_score = {mode})", ""]
    if mode == "off":
        return lines + ["Rule 2 (response score) was not computed in this mode; only rule 1 (SPORE v3.0 target-expression percentile) is applied.", ""]
    frames = []
    for block in range(total_blocks(run)):
        path = run.stats_dir / "gather" / f"block_{block:04d}" / "rule_counts.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return lines + ["no rule counts found.", ""]
    table = pd.concat(frames).sort_values("label").reset_index(drop=True)
    path = run.reports_dir / "escaper_rules.tsv"
    if not path.exists():
        table.to_csv(path, sep="	", index=False)
    usable = table[table["n_rule2"] >= 0]
    cells = int(table["n_total"].sum())
    applied = {"report": "rule 1 only", "union": "rule 1 or rule 2", "replace": "rule 2 (rule 1 where rule 2 could not be computed)"}[mode]
    lines += [f"Applied: {applied}. Per perturbation in reports/escaper_rules.tsv. Counts are over the QC-passing cells of the {len(table):,} perturbations.", "",
              "| | cells | share of perturbation cells |", "|---|---|---|",
              f"| rule 1 flags (target above the control 10th percentile) | {int(table['n_rule1'].sum()):,} | {table['n_rule1'].sum() / max(cells, 1):.1%} |",
              f"| rule 2 flags (response score, {len(usable):,} perturbations where it could be computed) | {int(usable['n_rule2'].sum()):,} | {usable['n_rule2'].sum() / max(int(usable['n_total'].sum()), 1):.1%} |",
              f"| flagged by both | {int(usable['n_both'].sum()):,} | |",
              f"| rule 1 only | {int((usable['n_rule1'] - usable['n_both']).sum()):,} | |",
              f"| rule 2 only | {int((usable['n_rule2'] - usable['n_both']).sum()):,} | |",
              f"| removed by the applied rule | {int(table['n_removed'].sum()):,} | {table['n_removed'].sum() / max(cells, 1):.1%} |", ""]
    unusable = table[table["n_rule2"] < 0]
    if len(unusable):
        lines += [f"Rule 2 not computable for {len(unusable)} perturbations: " + ", ".join(f"{k} {v}" for k, v in unusable['rule2_status'].value_counts().items()) + ".", ""]
    gain = usable.assign(extra=usable["n_rule2"] - usable["n_both"]).sort_values("extra", ascending=False).head(10)
    if len(gain) and int(gain["extra"].max()) > 0:
        lines += ["Perturbations where rule 2 flags the most cells that rule 1 misses (candidates for the low-expressed-target blind spot):", "",
                  "| perturbation | cells | rule 1 | rule 2 | both |", "|---|---|---|---|---|"]
        lines += [f"| {row.label} | {row.n_total} | {row.n_rule1} | {row.n_rule2} | {row.n_both} |" for row in gain.itertuples()]
        lines += [""]
    return lines


def _phase_map_lines(run):
    lines = ["## Phase numbering (v4.0; old SPORE v3.0 numbers in the second column)", "", "| v4 phase | name | v3.0 phase | config section |", "|---|---|---|---|"]
    lines += [f"| {number} | {name} | {old} | {section} |" for number, name, old, section in PHASE_MAP]
    lines += ["", "Not in the core any more (future modules): v3.0 phases 3 ambient RNA, 4 doublets, 10 PCA / Harmony / cell-cycle regression, "
              "11 cell-line separation. v3.0 output names are kept as aliases for one release "
              f"({'ON' if run.xl.get('output_aliases', True) and run.mode == 'production' else 'off in this mode'}).", ""]
    return lines


def _diagnostics_lines(run, reduce2, hvg_dir):
    """P5: confounding table, variance explained by batch / donor / perturbation, cell-cycle score diagnostic. Metadata only."""
    lines = ["## Diagnostics (nothing is changed by them)", ""]
    summary_path = reduce2 / "confounding_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        table = pd.read_csv(reduce2 / "confounding.tsv", sep="\t", keep_default_na=False)
        lines += ["### Perturbation x batch confounding", "",
                  f"- {summary['n_batches']} batches; reference batch mix = {summary['reference']} ({summary.get('n_reference_cells', 0):,} cells); a perturbation is flagged when its total "
                  f"variation distance (TVD) from the reference is >= {summary['tvd_warn']} AND beyond random sampling of the same number of reference cells "
                  f"(Monte Carlo p <= {summary.get('alpha', 0.001)}); the fixed-threshold rule alone would flag {summary.get('n_flagged_uncalibrated', summary['n_flagged'])}",
                  f"- Cramer's V between perturbation and batch (perturbations with >= 20 cells): {summary['cramers_v_perturbation_x_batch']:.3f}",
                  f"- perturbations entirely in one batch: {summary['n_fully_in_one_batch']}"]
        if summary["n_flagged"]:
            lines += [f"- **WARNING: {summary['n_flagged']} of {summary['n_perturbations']} perturbations ({summary['cells_in_flagged']:,} of "
                      f"{summary['cells_total']:,} cells) have a batch mix unlike the controls.** A model can learn the batch instead of the perturbation "
                      f"for these (worst 10 below; all in stats/reduce2/all/confounding.tsv)", "", "| perturbation | cells | dominant batch | max share | TVD | TVD expected by chance | p |", "|---|---|---|---|---|---|---|"]
            lines += [f"| {row.label} | {row.n_cells} | {row.dominant_batch} | {row.max_batch_share:.2f} | {row.tvd_vs_controls:.2f} | {row.tvd_null_mean:.2f} | {row.p_value:.4f} |"
                      for row in table.head(10).itertuples()]
        else:
            lines += ["- no perturbation is flagged"]
        lines += [""]
    else:
        lines += ["### Perturbation x batch confounding", "", "not computed: no batch column found in the source obs (set diagnostics.batch_col).", ""]
    diag_path = hvg_dir / "diagnostics.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        lines += ["### Variance explained on the HVG subsample (train cells only, log-normalized panel genes)", "",
                  f"{diag['n_cells']:,} cells x {diag['n_panel_genes']:,} genes. R2 = share of total variance; adjusted R2 corrects for the number of groups "
                  f"(perturbation has many groups, so only the adjusted value is comparable).", "", "| factor | groups | R2 | adjusted R2 |", "|---|---|---|---|"]
        for name, values in diag["variance_explained"].items():
            lines.append(f"| {name} | {values['groups']} | {values['r2']} | {values['r2_adjusted']} |")
        if "batch" not in diag["variance_explained"]:
            lines.append("| batch | - | not available (no batch column) | |")
        if "donor" not in diag["variance_explained"]:
            lines.append("| donor | - | not available (no donor column) | |")
        cycle = diag["cell_cycle"]
        lines += ["", "### Cell-cycle score diagnostic", ""]
        if cycle.get("status") == "ok":
            lines += [f"- {cycle['n_s_genes']} S and {cycle['n_g2m_genes']} G2M genes present; score variance S {cycle['s_score_variance']:.4f}, G2M {cycle['g2m_score_variance']:.4f}",
                      f"- phase fractions: {cycle['phase_fractions']}",
                      f"- adjusted R2 of the scores by perturbation: S {cycle.get('s_score_r2_by_perturbation_adjusted')}, G2M {cycle.get('g2m_score_r2_by_perturbation_adjusted')}"
                      + (f"; by batch: S {cycle.get('s_score_r2_by_batch_adjusted')}, G2M {cycle.get('g2m_score_r2_by_batch_adjusted')}" if "s_score_r2_by_batch_adjusted" in cycle else "")]
        else:
            lines += [f"- {cycle.get('status')}"]
        lines += [""]
    return lines


def _intake_lines(run):
    """Generic intake summary of EVERY run, so a broken intake is visible without any tally: the source, the cells and genes read, the perturbations and controls,
    and (recipe sources) per member the cells with 0 / 1 / 2 / 3+ targets, i.e. which cells were dropped at the intake and why."""
    layout = run.layout()
    source = run.xl.get("source", {}) or {}
    lines = ["## Intake summary", "", f"- source: kind `{source.get('kind', '?')}`, `{source.get('h5ad') or source.get('tar') or source.get('recipe') or '?'}`",
             f"- {layout['n_obs']:,} cells and {layout['n_vars']:,} genes enter the pipeline"]
    labels_path = run.manifest_dir / "labels.tsv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path, sep="\t", keep_default_na=False)
        controls = int(labels.loc[labels["label"] == run.ctrl_label, "n_cells_raw"].sum())
        lines.append(f"- {len(labels) - (1 if controls else 0):,} perturbation labels and {controls:,} control cells (`{run.ctrl_label}`) before any QC")
    members_path = run.manifest_dir / "intake_members.tsv"
    if source.get("kind") == "recipe" and members_path.exists():
        members = pd.read_csv(members_path, sep="\t")
        rows = []
        for row in members.itertuples():
            meta_path = run.stats_dir / "intake" / f"shard_{int(row.index):04d}" / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows.append({"member": meta["member"], "barcodes": meta["n_cells"], "cells_with_any_guide": meta["n_cells_with_any_guide"], "cells_0_targets": meta["cells_0_targets"],
                         "cells_1_target_written": meta["cells_written"], "cells_2_targets": meta["cells_2_targets"], "cells_3plus_targets": meta["cells_3plus_targets"],
                         "control_cells_written": meta["cells_written_control"], "labels": meta["n_labels"], "genes_written": meta.get("n_genes_written", meta.get("n_gene_features")),
                         "gene_order": meta.get("gene_order_status", "")})
        if rows:
            frame = pd.DataFrame(rows)
            summary_path = run.reports_dir / "intake_members_summary.tsv"
            if not summary_path.exists():
                frame.to_csv(summary_path, sep="\t", index=False)
            total = frame[["barcodes", "cells_with_any_guide", "cells_0_targets", "cells_1_target_written", "cells_2_targets", "cells_3plus_targets", "control_cells_written"]].sum()
            lines += [f"- {len(frame)} members, {int(total['barcodes']):,} barcodes (per member {int(frame['barcodes'].min()):,} to {int(frame['barcodes'].max()):,}), "
                      f"{int(total['cells_with_any_guide']):,} with any guide",
                      f"- cells written (exactly one target): {int(total['cells_1_target_written']):,} of which control {int(total['control_cells_written']):,}",
                      f"- cells dropped at the intake: {int(total['cells_0_targets']):,} with no target (no guide at or above the UMI threshold), "
                      f"{int(total['cells_2_targets']):,} with two targets, {int(total['cells_3plus_targets']):,} with three or more",
                      f"- per member table: reports/intake_members_summary.tsv"]
        genes_json = run.manifest_dir / "intake_genes.json"
        if genes_json.exists():
            info = json.loads(genes_json.read_text(encoding="utf-8"))
            lines.append(f"- ONE gene list for all members: policy `{info['policy']}`, {info['n_genes']:,} genes (union {info['n_union']:,}, dropped {info['n_dropped']:,}); "
                         f"members {info.get('statuses', {})}")
    waterfall = run.reports_dir / "WATERFALL.md"
    if waterfall.exists():
        lines += ["", "Cells dropped after the intake, and why (waterfall):", "", waterfall.read_text(encoding="utf-8").strip()]
    return lines + [""]


def _protected_lines(run):
    """Which protected-gene list was used: its path and version, and what the optional expression gate did to it."""
    p8 = run.cfg.get("phase8_hvg", {}) or {}
    gated = run.manifest_dir / "protected_genes_gated.json"
    lines = ["## Protected genes", ""]
    if gated.exists():
        meta = json.loads(gated.read_text(encoding="utf-8")).get("metadata", {})
        gate = meta.get("expression_gate", {})
        lines += [f"- input list: `{gate.get('input_list')}` (name {gate.get('input_name')}, version {gate.get('input_version')}, sha256 {str(gate.get('input_sha256'))[:12]})",
                  f"- expression gate: ON; genes detected in < {gate.get('min_fraction', 0):.0%} of {gate.get('n_control_cells', 0):,} control cells are removed from the tahoe_menu tier "
                  f"and flagged for review in tiers 1-2: {gate.get('n_genes_before')} -> {gate.get('n_genes_after')} genes, removed {len(gate.get('removed_from_tahoe_menu', []))}, "
                  f"flagged {len(gate.get('flagged_for_review', []))}",
                  f"- list used by plan, hvg and the report (version {meta.get('version')}): `{gated}`; gate report: `{run.manifest_dir / 'protected_gate_report.tsv'}`"]
    elif p8.get("protected_gene_list"):
        path = Path(p8["protected_gene_list"])
        version = name = "?"
        if path.exists():
            info = json.loads(path.read_text(encoding="utf-8")).get("metadata", {})
            version, name = info.get("version", "unversioned"), info.get("name")
        lines += [f"- list: `{path}` (name {name}, version {version}); expression gate: off"]
    else:
        lines += ["- no protected-gene list in this run"]
    return lines + [""]


def run_report(run: Run):
    figure_names = [name for name, _, _ in FIGURES]
    report_path = run.reports_dir / "REPORT.md"
    marker = run.marker("stats", "report", "all")
    if not marker.exists():
        quarantine_incomplete(run, "report", "all", [report_path] + [run.run_dir / "figures" / f"{name}.png" for name in figure_names])
    with task(run, "stats", "report", "all") as out:
        if out is None:
            return
        outcome, aliases = _report(run, out)
        # register the figures, their v3.0-name aliases and REPORT.md in the unit
        from xl_common import register_external
        register_external(out, [report_path] + [path for _, path, _, _ in outcome if path is not None] + aliases)


def _report(run, out):
    cfg, logger = run.cfg, run.logger
    cfg_plot = {**cfg, "paths": {**cfg["paths"], "_figures": None}}  # SPORE's _save then returns None: we save (and label) ourselves
    pert_col, ctrl = run.pert_col, run.ctrl_label
    layout = run.layout()
    outcome, aliases = [], []
    compat = run.mode == "compat-v3"

    def draw(stem, label, builder):
        try:
            fig = builder()
            if isinstance(fig, tuple):
                fig = fig[0]
            if fig is None:
                outcome.append((stem, None, label, "SPORE function returned no figure"))
                return
            file_stem = OLD_NAME[stem] if compat else stem  # compat-v3: the v3.0 file names are the primary names
            saved = _save(fig, run, file_stem, label)
            outcome.append((stem, saved, label, "ok"))
            if not compat and run.xl.get("output_aliases", True):
                alias = _alias_copy(run, stem)
                if alias is not None:
                    aliases.append(alias)
        except Exception as error:  # noqa: BLE001  (one bad figure must not stop the rest)
            logger.warning(f"[report] {stem} failed: {type(error).__name__}: {error}")
            outcome.append((stem, None, label, f"FAILED {type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}"))

    # ---- per-cell QC of ALL cells
    qc = _qc_table(run)
    raw_sums = GeneSums.load(require_done(run, "stats", "reduce1", "all") / "rawsums_all.npz")
    n_cells = len(qc)
    obs_stub = SimpleNamespace(obs=qc[["total_counts", "n_genes_by_counts", "pct_counts_mt"]], n_obs=n_cells,
                               shape=(n_cells, len(raw_sums.nnz)))
    draw("sparsity_saturation", f"all {n_cells:,} cells (scatter: random 50,000)",
         lambda: plot_sparsity_overview_arrays(raw_sums.nnz, qc["n_genes_by_counts"].to_numpy(), qc["total_counts"].to_numpy(),
                                               n_cells, len(raw_sums.nnz)))
    draw("cell_qc_violin", f"all {n_cells:,} cells (pre-filter)", lambda: spore_plotting.plot_qc_violin(obs_stub, cfg_plot))
    draw("cell_qc_mt_scatter", f"all {n_cells:,} cells (scatter: random 30,000)", lambda: spore_plotting.plot_mt_scatter(obs_stub, cfg_plot))

    # ---- Phase 5 efficiency (all perturbations)
    reduce1 = require_done(run, "stats", "reduce1", "all")
    counts = escaper_counts(run)
    knockdown = pd.read_csv(reduce1 / "knockdown.tsv", sep="\t", keep_default_na=False).set_index("label")["knockdown_depth"].astype(float)
    threshold = cfg.get("phase5_escaper_filtering", {}).get("efficiency_threshold", 0.5)
    efficiency = {}
    for label, row in counts.iterrows():
        score = row["n_kept"] / max(row["n_total"], 1)
        efficiency[label] = {"efficiency_score": float(score), "knockdown_depth": float(knockdown.get(label, float("nan"))),
                             "low_efficiency": bool(score < threshold)}
    draw("escaper_efficiency_scatter", f"all {len(efficiency):,} perturbations",
         lambda: spore_plotting.plot_perturbation_efficiency(SimpleNamespace(uns={"knockdown_efficiency": efficiency}), cfg_plot))

    # ---- Phase 6
    reduce2 = require_done(run, "stats", "reduce2", "all")
    waterfall6 = json.loads((reduce2 / "phase6_waterfall.json").read_text(encoding="utf-8"))
    penetrance = np.load(reduce2 / "phase6_penetrance.npy")
    draw("gene_qc_dashboard", f"all {len(penetrance):,} genes; cells = survivors of QC, escaper and undersized filters",
         lambda: spore_plotting.plot_phase6_diagnostics(SimpleNamespace(uns={"phase6_waterfall": waterfall6, "phase6_gene_penetrance": penetrance}), cfg_plot))

    # ---- Phase 7
    final = pd.read_parquet(run.manifest_dir / "global_obs_final.parquet")
    survivors = final[final["survived_to_phase7"]]
    meta = json.loads((run.splits_dir / f"{run.name}_split_indices.json").read_text(encoding="utf-8"))
    deg = pd.read_csv(reduce2 / "mean_shift.tsv", sep="\t", keep_default_na=False)
    deg_counts = pd.Series(deg["score"].astype(float).to_numpy(), index=deg["label"].astype(str))
    stubs, info = {}, {}
    for split in SPLITS:
        labels = survivors.loc[survivors["split"].astype(str) == split, "label"].astype(str)
        stubs[split] = SimpleNamespace(obs=pd.DataFrame({pert_col: labels.to_numpy()}))
        info[split] = int(len(labels))
    result = [{"split_info": info, "deg_counts": deg_counts, "train_labels": meta["train_labels"], "val_labels": meta["val_labels"],
               "test_labels": meta["test_labels"], **stubs}]
    draw("split_dashboard", f"all {len(survivors):,} surviving cells", lambda: spore_plotting.plot_phase7_splits(result, cfg_plot))

    # ---- Phase 8 (statistics come from the stratified HVG subsample by SPORE's design)
    hvg_dir = require_done(run, "stats", "hvg", "all")
    hvg_stats = pd.read_parquet(hvg_dir / "hvg_stats.parquet")
    hvg_summary = json.loads((hvg_dir / "summary.json").read_text(encoding="utf-8"))
    draw("hvg_dashboard", f"gene statistics from a stratified subsample of {hvg_summary['n_subsample']:,} of "
                              f"{hvg_summary['n_train']:,} train cells",
         lambda: spore_plotting.plot_phase8_hvg(SimpleNamespace(uns={"hvg_stats": hvg_stats.to_dict("index")}), cfg_plot))

    # ---- Phase 9
    cache, n_values, n_parts = _phase9_cache(run)
    draw("normalization", f"subsample of {len(cache['raw']):,} nonzero values from {n_parts} parts (random parts per split)",
         lambda: spore_plotting.plot_phase9_normalization(cache, cfg_plot))

    # ---- REPORT.md
    lines = [f"# SPORE-XL report: {run.name}", "", f"Run directory: `{run.run_dir}`", ""]
    lines += ["## Figures", "", "| figure | what | drawn from | status |", "|---|---|---|---|"]
    what = {new: text for new, _, text in FIGURES}
    for stem, path, label, status in outcome:
        shown = f"{OLD_NAME[stem]}.png" if compat else f"{stem}.png (alias {OLD_NAME[stem]}.png)" if stem != OLD_NAME[stem] else f"{stem}.png"
        lines.append(f"| {shown} | {what[stem]} | {label} | {status.splitlines()[0]} |")
    lines += ["", "No PCA / Harmony / cell-cycle-regression figures (old phase 10): that phase is not in the v4.0 core.", ""]
    coverage = (run.reports_dir / "COVERAGE_REPORT.md")
    summary2 = json.loads((reduce2 / "summary.json").read_text(encoding="utf-8"))
    lines += ["## Run summary", "", f"- cells in {layout['n_obs']:,}; survivors to Phase 7 {summary2['n_survivors']:,}; controls {summary2['n_controls']:,}; "
              f"perturbations {summary2['n_perturbations']:,}; genes kept by Phase 6 {summary2['n_genes_kept']:,}",
              f"- split (perturbations): train {summary2['n_train']}, val {summary2['n_val']}, test {summary2['n_test']}",
              f"- HVG panel {hvg_summary['n_panel']} genes (policy and counts below)", ""]
    lines += _intake_lines(run)
    lines += _protected_lines(run)
    lines += [f"## Perturbation-target node policy (phase 7, HVG panel; mode {run.mode})", ""]
    lines += [f"- {key}: {value}" for key, value in hvg_summary.items()]
    node_summary = hvg_dir / "node_summary.tsv"
    if node_summary.exists():
        table = pd.read_csv(node_summary, sep="\t")
        lines += ["", "Node perturbations (target gene is a column of the outputs) versus row-only perturbations (cells kept, target not a node); "
                  "per perturbation in manifest/perturbation_nodes.tsv:", "", "| reason | perturbations | cells |", "|---|---|---|"]
        lines += [f"| {row.reason} | {int(row.perturbations):,} | {int(row.cells):,} |" for row in table.itertuples()]
    lines += [""]
    lines += _escaper_lines(run)
    lines += _diagnostics_lines(run, reduce2, hvg_dir)
    lines += _phase_map_lines(run)
    if coverage.exists():
        lines += ["## Coverage and firewall", "", coverage.read_text(encoding="utf-8"), ""]
    timing_path = run.reports_dir / "memory_and_time.tsv"
    if timing_path.exists():
        timing = pd.read_csv(timing_path, sep="\t")
        grouped = timing.groupby("stage").agg(units=("unit", "count"), peak_rss_gb_max=("peak_rss_gb", "max"), seconds_sum=("seconds", "sum"),
                                               seconds_max=("seconds", "max")).round(2)
        lines += ["## Time and peak RSS per stage", "", "| stage | units | peak RSS max (GB) | seconds sum | seconds max |", "|---|---|---|---|---|"]
        lines += [f"| {stage} | {int(row.units)} | {row.peak_rss_gb_max} | {row.seconds_sum} | {row.seconds_max} |" for stage, row in grouped.iterrows()]
        lines += [""]
    cleanup_path = run.reports_dir / "cleanup_log.tsv"
    if cleanup_path.exists():
        cleanup = pd.read_csv(cleanup_path, sep="\t")
        lines += ["## Clean-up (Patrick-approved, opt-in)", "", f"{len(cleanup)} files removed, {cleanup['size_bytes'].sum() / 1e9:.2f} GB; "
                  f"full list in reports/cleanup_log.tsv", ""]
    else:
        lines += ["## Clean-up", "", "not active (default): no intermediate file was removed.", ""]
    failed = [stem for stem, path, _, status in outcome if path is None]
    if failed:
        lines += ["## Figure failures", ""] + [f"- {stem}: {status}" for stem, path, _, status in outcome if path is None]
    write_new_text(run.reports_dir / "REPORT.md", "\n".join(lines) + "\n")
    logger.info(f"[report] {len(outcome) - len(failed)}/{len(outcome)} figures; {run.reports_dir / 'REPORT.md'}")
    return outcome, aliases
