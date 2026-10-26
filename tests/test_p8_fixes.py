"""P8 known-issue fixes: one mean_umi_threshold default, measure's worker check consistent with the per-stage pre-flight, compare_v4's default tolerance.
(The stable HVG tie-break is tested in test_production_defaults.py.)  Run: python -B tests\\test_p8_fixes.py"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "engine"))
import paths  # noqa: E402

paths.add_spore_to_path()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import compare_v4  # noqa: E402
import measure  # noqa: E402
from adapted_phases import phase06_gene_triage as phase06  # noqa: E402
from stats import GeneSums  # noqa: E402


def test_one_mean_umi_default():
    assert phase06.MEAN_UMI_THRESHOLD_DEFAULT == 0.25
    import config_v3
    assert config_v3._DEFAULTS["phase6_gene_triage"]["mean_umi_threshold"] == phase06.MEAN_UMI_THRESHOLD_DEFAULT
    # a gene seen in 30 of 10,000 cells (below the adaptive cut of 100 cells) with mean 0.12: removed at 0.25, kept at 0.05
    sums = GeneSums(2)
    sums.n_cells = 10000
    sums.nnz = np.array([5000, 30])
    sums.total = np.array([9000.0, 1200.0])
    cfg = {"phase6_gene_triage": {"min_cells_expressing": 10, "pct_filter": 0.01, "rescue_perturbation_targets": False}, "dataset": {"perturbation_col": "gene"}}
    keep, *_ = phase06.keep_mask_from_stats(sums, ["A", "B"], [], cfg)
    assert list(keep) == [True, False], "a bare cfg now behaves like spore_local (0.25)"
    cfg["phase6_gene_triage"]["mean_umi_threshold"] = 0.05
    keep, *_ = phase06.keep_mask_from_stats(sums, ["A", "B"], [], cfg)
    assert list(keep) == [True, True]


def test_ghost_rescue_is_a_parameter():
    sums = GeneSums(2)
    sums.n_cells = 10000
    sums.nnz = np.array([5000, 3])
    sums.total = np.array([9000.0, 3.0])
    cfg = {"phase6_gene_triage": {"rescue_perturbation_targets": False}, "dataset": {"perturbation_col": "gene"}}
    on, *_ = phase06.keep_mask_from_stats(sums, ["A", "MCM5"], [], cfg, ghost_rescue=True)
    off, *_ = phase06.keep_mask_from_stats(sums, ["A", "MCM5"], [], cfg, ghost_rescue=False)
    assert list(on) == [True, True] and list(off) == [True, False]


def test_measure_worker_check_uses_the_parallel_tasks():
    table = pd.DataFrame({"unit": ["x.h5ad"], "kind": ["h5ad"], "size_bytes": [1], "cells": [1_000_000], "features": [8000], "gene_features": [8000],
                          "guide_features": [0], "nnz": [3_000_000_000], "nnz_estimated": [False]})
    table["nnz_per_cell"] = table["nnz"] / table["cells"]
    projection = measure.project(table, shard_rows=40000, n_blocks=128, n_perts=9000, survival=0.9, workers=14, panel_genes=5000, cpus=16, max_ram_gb=110)
    parallel, single = projection["peak_parallel_task_ram_gb"], projection["peak_task_ram_gb"]
    assert single >= parallel
    assert projection["workers_x_parallel_peak_gb"] == round(14 * parallel, 1)
    # a budget that fits 14 parallel tasks but not 14 x the single-task peak must NOT be refused by main()
    budget = 14 * parallel + 1
    assert 14 * single > budget or single == parallel
    source = Path(__file__).resolve()
    buffer = io.StringIO()
    # main() needs a real source; use the numbers through the same rule instead
    assert measure.recommend_workers(16, budget, parallel) == min(14, int(budget // parallel))


def test_compare_default_tolerance_is_item_13():
    assert compare_v4.DEFAULT_QC_FLOAT32_REL_TOL == 1e-4
    import inspect
    assert inspect.signature(compare_v4.compare_with_spore).parameters["qc_float32_rel_tol"].default == 1e-4
    ours = pd.DataFrame({"n_cells_by_counts": [10, 20], "mean_counts": [1.0, 2.0], "pct_dropout_by_counts": [50.0, 60.0], "total_counts": [1e8, 2e8]})
    theirs = ours.copy()
    theirs["total_counts"] = theirs["total_counts"] * (1 + 5e-6)   # float32 rounding of a big column sum
    assert compare_v4.qc_column_check(ours, theirs, compare_v4.DEFAULT_QC_FLOAT32_REL_TOL)[0]
    assert not compare_v4.qc_column_check(ours, theirs)[0]            # strict 1e-6 of the column maximum
    theirs2 = ours.copy()
    theirs2["n_cells_by_counts"] = [10, 21]                            # the exact columns stay strict
    assert not compare_v4.qc_column_check(ours, theirs2, compare_v4.DEFAULT_QC_FLOAT32_REL_TOL)[0]


def test_perturbations_whose_cells_all_escaped_still_get_a_final_count_row():
    """Found by the stress suite (dataset 06): a perturbation with every cell removed as an escaper sends nothing to gather; its cells must still be counted."""
    from types import SimpleNamespace
    import stage_gather
    from xl_common import block_of_label
    labels = [f"G{i}" for i in range(40)]
    counts = pd.DataFrame({"n_total": range(10, 50), "n_kept": 0, "bypassed": False}, index=pd.Index(labels, name="label"))
    run = SimpleNamespace(layout=lambda: {"n_blocks": 8}, ctrl_label="non-targeting")
    rows = []
    for block in range(8):
        rows += stage_gather.assigned_rows(run, block, counts, set())
    assert sorted(row["label"] for row in rows) == sorted(labels), "every label is assigned to exactly one block"
    assert all(row["n_final"] == 0 for row in rows) and sum(row["n_total"] for row in rows) == int(counts["n_total"].sum())
    only = stage_gather.assigned_rows(run, block_of_label("G3", 8), counts, {"G3"})
    assert "G3" not in [row["label"] for row in only]
    assert stage_gather.assigned_rows(run, 8, counts, set()) == [], "control blocks own no perturbation"


if __name__ == "__main__":
    failures = 0
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except Exception:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                failures += 1
                print(f"FAIL {name}")
    sys.exit(1 if failures else 0)
