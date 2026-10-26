"""plan + map1 + reduce1 at 1, 3 and 7 row shards vs the in-process SPORE reference.
Run: set PYTHONHASHSEED=0 ; python -B tests\\test_stage_map1_reduce1.py   (XL_TEST_TMP = scratch folder)
Gate: QC-passing cells, gene space, QC obs columns and QC var columns equal SPORE Phase 2; escaper thresholds equal
SPORE's np.percentile values (exact); knockdown depth within 1e-5 relative (report-only column)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import REFERENCE, fresh_run_config, read_lines, run_all, shard_rows_for  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xl_common import Run  # noqa: E402
import stage_map1  # noqa: E402
import stage_plan  # noqa: E402
import stage_reduce1  # noqa: E402


def build(n_shards):
    config = fresh_run_config(f"m1r1_{n_shards}", shard_rows=shard_rows_for(n_shards))
    run = Run(config)
    stage_plan.run_plan(run)
    for index in range(run.n_shards()):
        stage_map1.run_map1(run, index)
    stage_reduce1.run_reduce1(run)
    return run


def check(n_shards):
    run = build(n_shards)
    assert run.n_shards() == n_shards
    # idempotent: a second pass is a no-op
    stage_plan.run_plan(run)
    stage_map1.run_map1(run, 0)
    stage_reduce1.run_reduce1(run)

    # ---- Phase 2 equivalence: cells, order, gene space, QC columns
    ref_cells = read_lines(REFERENCE / "p2_cells.txt")
    ref_var = read_lines(REFERENCE / "p2_var.txt")
    pieces = [ad.read_h5ad(run.work_dir / "map1" / f"shard_{i:04d}" / "qc.h5ad") for i in range(n_shards)]
    names = [name for piece in pieces for name in piece.obs_names]
    assert names == ref_cells, (len(names), len(ref_cells))
    for piece in pieces:
        assert list(piece.var_names) == ref_var
    obs = pd.concat([piece.obs for piece in pieces])
    ref_qc = pd.read_parquet(REFERENCE / "p2_obs_qc.parquet")
    for column in ("total_counts", "n_genes_by_counts", "pct_counts_mt"):
        assert obs[column].to_numpy().tobytes() == ref_qc[column].to_numpy().tobytes(), column
    assert (obs["xl_global_index"].to_numpy() == np.sort(obs["xl_global_index"].to_numpy())).all()

    # ---- QC var columns (recomputed from global gene sums) vs SPORE's pre-filter columns
    reduce1 = run.stats_dir / "reduce1" / "all"
    qc = pd.read_csv(reduce1 / "qc_var_columns.tsv", sep="\t", index_col=0)
    ref_var_table = pd.read_csv(REFERENCE / "p2_var_table.tsv", sep="\t", index_col=0)
    for column in ("n_cells_by_counts", "mean_counts", "pct_dropout_by_counts", "total_counts"):
        reference = ref_var_table.loc[ref_var, column].to_numpy(float)
        ours = qc.loc[ref_var, column].to_numpy(float)
        assert np.max(np.abs(reference - ours)) <= 1e-6 * max(1.0, np.max(np.abs(reference))), column
    assert (qc.loc[ref_var, "mt"].astype(bool).to_numpy() == ref_var_table.loc[ref_var, "mt"].astype(bool).to_numpy()).all()

    # ---- Phase 2 waterfall and label counts
    waterfall = pd.read_json(reduce1 / "waterfall_phase2.json", typ="series")
    ref_waterfall = pd.read_json(REFERENCE / "p2_waterfall.json", typ="series")
    assert (waterfall.to_numpy() == ref_waterfall.to_numpy()).all(), (waterfall.to_dict(), ref_waterfall.to_dict())
    counts = pd.read_csv(reduce1 / "labelcounts_qc.tsv", sep="\t", keep_default_na=False).set_index("label")["n_qc_pass"]
    reference_labels = pd.read_parquet(run.manifest_dir / "global_obs.parquet").set_index("xl_global_index")
    truth_labels = pd.read_parquet(Path(REFERENCE).parent.parent / "data" / "synthetic_60k_truth.parquet").set_index("cell")["label"]
    expected = truth_labels.loc[ref_cells].value_counts()
    assert counts.sort_index().to_dict() == expected.sort_index().to_dict()

    # ---- thresholds: exact
    ours = pd.read_csv(reduce1 / "escaper_thresholds.tsv", sep="\t").set_index("gene")
    reference = pd.read_csv(REFERENCE / "p5_thresholds.tsv", sep="\t").set_index("gene")
    assert sorted(ours.index) == sorted(reference.index), "target gene sets differ"
    mismatches = [gene for gene in reference.index
                  if float(ours.loc[gene, "threshold"]) != float(reference.loc[gene, "threshold"])
                  or ours.loc[gene, "threshold_dtype"] != reference.loc[gene, "threshold_dtype"]
                  or int(ours.loc[gene, "n_ctrl"]) != int(reference.loc[gene, "n_ctrl"])]
    assert not mismatches, mismatches[:10]

    # ---- knockdown depth (report-only): 1e-5 relative
    depth = pd.read_csv(reduce1 / "knockdown.tsv", sep="\t", keep_default_na=False).set_index("label")["knockdown_depth"]
    ref_depth = pd.read_csv(REFERENCE / "p5_knockdown.tsv", sep="\t", keep_default_na=False).set_index("label")["knockdown_depth"]
    shared = ref_depth.index
    ours_values = depth.reindex(shared).astype(float).to_numpy()
    ref_values = ref_depth.astype(float).to_numpy()
    both_nan = np.isnan(ours_values) & np.isnan(ref_values)
    close = np.isclose(ours_values, ref_values, rtol=1e-5, atol=0) | both_nan
    assert close.all(), list(shared[~close][:5])
    print(f"    {n_shards} shard(s): {len(names)} QC cells, {len(ours)} thresholds exact, depth max rel diff "
          f"{np.nanmax(np.abs(ours_values - ref_values) / np.maximum(np.abs(ref_values), 1e-12)):.2e}")


def test_map1_reduce1(n_shards):
    check(n_shards)


if __name__ == "__main__":
    sys.exit(1 if run_all(test_map1_reduce1) else 0)
