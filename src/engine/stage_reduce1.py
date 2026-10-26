"""reduce1 (one job): exact escaper thresholds and knockdown depth per target gene, QC var columns, QC waterfall.

Reproduces phase05_escaper_filtering.py:158-179 (threshold = np.percentile of the control counts, zero-inflation
guard for activation) from the merged integer histograms; the percentile is returned in the dtype of X, as
np.percentile would on the dense column. Knockdown depth (phase05:164-168, 186) is report-only.
"""
import numpy as np
import pandas as pd

from xl_common import (Run, require_done, shard_unit, task, write_new_json, write_new_parquet, write_new_tsv,
                       alias_map)
import escaper_score
from stats import GeneSums, IntHistogramBank, Waterfall
from stage_map1 import target_genes_by_label
from adapted_phases.phase02_cell_triage import ribo_gene_mask

from src.phases.phase05_escaper_filtering import _get_escaper_direction

QC_VAR_COLUMNS = ["mt", "n_cells_by_counts", "mean_counts", "pct_dropout_by_counts", "total_counts"]


def qc_var_columns(raw_sums, var_names, organism="human", x_dtype=np.float64):
    """scanpy.pp.calculate_qc_metrics var columns, recomputed from global gene sums (blueprint Amendment 1 item 6)."""
    n_cells = raw_sums.n_cells
    prefix = "MT-" if organism == "human" else "mt-"
    frame = pd.DataFrame(index=pd.Index(var_names, name=None))
    frame["mt"] = pd.Index(var_names).str.startswith(prefix)
    frame["n_cells_by_counts"] = raw_sums.nnz
    frame["mean_counts"] = raw_sums.total / n_cells
    if np.dtype(x_dtype) == np.float32:  # scanpy keeps mean_counts in the matrix dtype (verified bit-equal after the cast)
        frame["mean_counts"] = frame["mean_counts"].astype(np.float32)
    frame["pct_dropout_by_counts"] = (1 - raw_sums.nnz / n_cells) * 100
    frame["total_counts"] = raw_sums.total
    return frame


def check_label_collisions(labels, qc_columns):
    """Latent SPORE issue (phase05:84-87 maps str(value) of every var column): fail loudly if a perturbation
    label could be hit by a stringified QC value."""
    problems = []
    label_set = set(labels)
    if label_set & {"True", "False"}:
        problems.append(f"label equals a stringified bool: {sorted(label_set & {'True', 'False'})}")
    numeric_labels = {}
    for label in label_set:
        try:
            numeric_labels[label] = float(label)
        except ValueError:
            continue
    if numeric_labels:
        for column in ("n_cells_by_counts", "mean_counts", "pct_dropout_by_counts", "total_counts"):
            values = qc_columns[column].to_numpy(dtype=np.float64)
            for label, number in numeric_labels.items():
                if np.any(np.isclose(values, number, rtol=1e-5, atol=0)):
                    problems.append(f"label '{label}' equals a value of QC var column {column}")
    if problems:
        raise ValueError("perturbation labels collide with stringified QC var values (SPORE phase05:84-87): "
                         + "; ".join(problems))


def run_reduce1(run: Run):
    with task(run, "stats", "reduce1", "all") as out:
        if out is None:
            return
        cfg = run.cfg
        n_shards = run.n_shards()
        raw_sums, waterfall = None, Waterfall()
        label_counts, pert_sums, dtypes = [], [], set()
        n_controls = 0
        for index in range(n_shards):
            unit = require_done(run, "stats", "map1", shard_unit(index))
            shard_sums = GeneSums.load(unit / "rawsums.npz")
            raw_sums = shard_sums if raw_sums is None else raw_sums.add(shard_sums)
            waterfall.add(Waterfall.load(unit / "waterfall.json"))
            label_counts.append(pd.read_parquet(unit / "labelcounts.parquet"))
            pert_sums.append(pd.read_parquet(unit / "pertsums.parquet"))
            meta = pd.read_json(unit / "meta.json", typ="series")
            dtypes.add(str(meta["x_dtype"]))
            n_controls += int(meta["n_controls_qc_pass"])
        bank = IntHistogramBank.merge_files([run.unit_dir("stats", "map1", shard_unit(index)) / "bank.npz" for index in range(n_shards)])  # P10
        if len(dtypes) != 1:
            raise RuntimeError(f"shards disagree on the X dtype: {dtypes}")
        x_dtype = np.dtype(dtypes.pop())

        var_table = run.var_table()
        qc_columns = qc_var_columns(raw_sums, var_table.index, cfg["dataset"].get("organism", "human"), x_dtype)
        labels = run.labels_table()["label"].tolist()
        check_label_collisions([label for label in labels if label != run.ctrl_label], qc_columns)
        write_new_tsv(out / "qc_var_columns.tsv", qc_columns, index=True)
        if run.escaper_mode != "off":  # P4: merge the per-shard control reference samples into the 2,000-cell reference
            parts = []
            for index in range(n_shards):
                with np.load(require_done(run, "stats", "map1", shard_unit(index)) / "control_reference.npz") as data:
                    parts.append((data["hashes"].copy(), data["global_index"].copy(), data["rows"].copy()))
            merged = escaper_score.merge_references(parts)
            if merged is None:
                np.savez(out / "control_reference.npz", hashes=np.zeros(0, np.uint64), global_index=np.zeros(0, np.int64),
                         rows=np.zeros((0, parts[0][2].shape[1]), np.float32))
            else:
                np.savez(out / "control_reference.npz", hashes=merged[0], global_index=merged[1], rows=merged[2])
        raw_sums.save(out / "rawsums_all.npz")
        waterfall.save(out / "waterfall_phase2.json")

        label_counts = pd.concat(label_counts).groupby("label", sort=True)["n_qc_pass"].sum().reset_index()
        write_new_tsv(out / "labelcounts_qc.tsv", label_counts)

        # ---- thresholds
        p5 = cfg.get("phase5_escaper_filtering", {})
        percentile = p5.get("escaper_percentile", 10)
        direction = _get_escaper_direction(cfg, run.logger)
        ribo_keep = ribo_gene_mask(var_table.index, cfg, run.ribo_exempt())
        var_reduced = var_table.loc[ribo_keep]
        gene_to_idx = alias_map(var_reduced)
        gene_lists = target_genes_by_label(run, labels, gene_to_idx)
        target_genes = sorted({gene for genes in gene_lists.values() for gene in genes})
        rows = []
        for gene in target_genes:
            key = str(gene)
            if bank.n_cells(key) == 0:
                raise ValueError(f"no control cells to compute the escaper threshold for gene index {gene}")
            if direction == "knockdown":
                threshold = bank.percentile(key, percentile, x_dtype)
            else:
                threshold = bank.percentile(key, 100 - percentile, x_dtype)
                if threshold == 0:
                    threshold = 1e-5
            values, counts = bank.table[key]
            n_ctrl = int(counts.sum())
            rows.append({"gene_index": gene, "gene": str(var_reduced.index[gene]), "threshold": repr(float(threshold)),
                         "threshold_dtype": str(np.asarray(threshold).dtype), "n_ctrl": n_ctrl,
                         "ctrl_mean": repr(float((values * counts).sum() / n_ctrl))})
        thresholds = pd.DataFrame(rows, columns=["gene_index", "gene", "threshold", "threshold_dtype", "n_ctrl", "ctrl_mean"])
        write_new_tsv(out / "escaper_thresholds.tsv", thresholds)

        # ---- knockdown depth per perturbation (report only)
        ctrl_mean = dict(zip(thresholds["gene_index"], thresholds["ctrl_mean"].map(float)))
        sums = pd.concat(pert_sums).groupby(["label", "gene_index"], sort=True)[["n_cells", "sum"]].sum().reset_index()
        depth_rows = []
        for label, group in sums.groupby("label", sort=True):
            depths = [row["sum"] / row["n_cells"] / ctrl_mean[row["gene_index"]]
                      for _, row in group.iterrows() if ctrl_mean[row["gene_index"]] > 1e-10]
            depth_rows.append({"label": label, "knockdown_depth": repr(float(np.mean(depths))) if depths else "nan",
                               "n_genes_checked": int(len(gene_lists.get(label, [])))})
        write_new_tsv(out / "knockdown.tsv", pd.DataFrame(depth_rows, columns=["label", "knockdown_depth", "n_genes_checked"]))
        write_new_json(out / "summary.json", {
            "x_dtype": str(x_dtype), "direction": direction, "percentile": percentile, "n_target_genes": len(target_genes),
            "n_controls_qc_pass": n_controls, "n_labels": len(labels), "n_cells_raw": int(raw_sums.n_cells),
            "n_genes_after_ribo": int(var_reduced.shape[0])})
        run.logger.info(f"[reduce1] {len(target_genes)} target genes, {n_controls:,} QC-passing controls")
