"""map2 + gather at 1, 3 and 7 row shards vs the in-process SPORE reference and the raw matrix.
Run: set PYTHONHASHSEED=0 ; python -B tests\\test_stage_map2_gather.py   (XL_TEST_TMP = scratch folder)
Gate: kept cells (escaper filter) and survivors (after the undersized drop) equal SPORE's; every perturbation in exactly
one block; blocks in global order; block matrices equal the raw matrix restricted to those cells and genes; Phase 6
GeneSums equal an independent computation on SPORE's survivors; per-perturbation float32 means bit-equal to scipy."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import REFERENCE, SOURCE, fresh_run_config, read_lines, run_all, shard_rows_for  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from xl_common import Run  # noqa: E402
import stage_gather  # noqa: E402
import stage_map1  # noqa: E402
import stage_map2  # noqa: E402
import stage_plan  # noqa: E402
import stage_reduce1  # noqa: E402
from stats import GeneSums  # noqa: E402

_SOURCE_CACHE = {}


def source_matrix():
    if "x" not in _SOURCE_CACHE:
        adata = ad.read_h5ad(SOURCE)
        _SOURCE_CACHE.update(x=sp.csr_matrix(adata.X), names=np.array(adata.obs_names), var=np.array(adata.var_names),
                             labels=adata.obs["gene"].astype(str).to_numpy())
    return _SOURCE_CACHE


def build(n_shards):
    config = fresh_run_config(f"m2g_{n_shards}", shard_rows=shard_rows_for(n_shards))
    run = Run(config)
    stage_plan.run_plan(run)
    for index in range(run.n_shards()):
        stage_map1.run_map1(run, index)
    stage_reduce1.run_reduce1(run)
    for index in range(run.n_shards()):
        stage_map2.run_map2(run, index)
    for block in range(stage_gather.total_blocks(run)):
        stage_gather.run_gather(run, block)
    return run


def check(n_shards):
    run = build(n_shards)
    source = source_matrix()
    name_to_row = {name: row for row, name in enumerate(source["names"])}
    layout = run.layout()
    n_blocks_total = stage_gather.total_blocks(run)

    # idempotent re-run
    stage_map2.run_map2(run, 0)
    stage_gather.run_gather(run, 0)

    # ---- map2: kept cells after the escaper filter, one block per perturbation
    kept = []
    label_block = {}
    for index in range(run.n_shards()):
        directory = run.work_dir / "map2" / f"shard_{index:04d}"
        for path in sorted(directory.glob("block_*.h5ad")):
            piece = ad.read_h5ad(path)
            block = int(path.stem.split("_")[1])
            kept.extend(zip(piece.obs["xl_global_index"], piece.obs_names))
            for label in set(piece.obs["gene"].astype(str)):
                if label != run.ctrl_label:
                    assert label_block.setdefault(label, block) == block, f"{label} in two blocks"
    kept.sort()
    assert [name for _, name in kept] == read_lines(REFERENCE / "p5_escaper_kept_cells.txt")
    ref_stats = pd.read_csv(REFERENCE / "p5_escaper_stats.tsv", sep="\t", keep_default_na=False).set_index("perturbation")
    counts = stage_gather.merged_counts(run)
    assert sorted(counts.index) == sorted(ref_stats.index)
    assert (counts.loc[ref_stats.index, "n_total"].to_numpy() == ref_stats["n_total"].to_numpy()).all()
    assert (counts.loc[ref_stats.index, "n_kept"].to_numpy() == ref_stats["n_kept"].to_numpy()).all()
    assert (counts.loc[ref_stats.index, "bypassed"].to_numpy() == (ref_stats["status"] != "filtered").to_numpy()).all()

    # ---- gather: survivors, order, matrices, dropped labels
    survivors_ref = read_lines(REFERENCE / "p5b_cells.txt")
    ref_var = read_lines(REFERENCE / "p2_var.txt")
    var_columns = [int(np.flatnonzero(source["var"] == gene)[0]) for gene in ref_var]
    all_survivors, genesums, dropped = [], None, []
    permeans = {}
    for block in range(n_blocks_total):
        stats_unit = run.stats_dir / "gather" / f"block_{block:04d}"
        survivors = pd.read_parquet(stats_unit / "survivors.parquet")
        assert np.all(np.diff(survivors["xl_global_index"].to_numpy()) > 0), "block not in global order"
        all_survivors.append(survivors)
        dropped.extend(pd.read_parquet(stats_unit / "dropped.parquet")["label"].tolist())
        sums = GeneSums.load(stats_unit / "genesums.npz")
        genesums = sums if genesums is None else genesums.add(sums)
        with np.load(stats_unit / "permean.npz") as data:
            for label, mean in zip(data["labels"].tolist(), data["means"]):
                permeans[label] = mean
        block_file = run.work_dir / "gather" / f"block_{block:04d}" / "block.h5ad"
        if survivors.shape[0]:
            block_adata = ad.read_h5ad(block_file)
            rows = [name_to_row[name] for name in block_adata.obs_names]
            assert rows == survivors["xl_global_index"].tolist()
            expected = source["x"][rows][:, var_columns]
            assert list(block_adata.var_names) == ref_var
            assert (sp.csr_matrix(block_adata.X) != expected).nnz == 0, f"block {block} matrix differs from the raw data"
    merged = pd.concat(all_survivors).sort_values("xl_global_index")
    assert [source["names"][i] for i in merged["xl_global_index"]] == survivors_ref
    expected_dropped = sorted(ref_stats.index[ref_stats["n_kept"] < 50])
    assert sorted(dropped) == expected_dropped, (sorted(dropped)[:5], expected_dropped[:5])

    # ---- Phase 6 GeneSums vs an independent computation on SPORE's survivors
    rows = [name_to_row[name] for name in survivors_ref]
    reference_matrix = source["x"][rows][:, var_columns]
    assert genesums.n_cells == len(survivors_ref)
    assert np.array_equal(genesums.nnz, reference_matrix.getnnz(axis=0))
    assert np.array_equal(genesums.total, np.asarray(reference_matrix.astype(np.float64).sum(axis=0)).ravel())

    # ---- escaper obs columns
    ref_eff = pd.read_parquet(REFERENCE / "p5_obs_eff.parquet").loc[survivors_ref]
    pieces = [ad.read_h5ad(run.work_dir / "gather" / f"block_{b:04d}" / "block.h5ad")
              for b in range(n_blocks_total) if (run.work_dir / "gather" / f"block_{b:04d}" / "block.h5ad").exists()]
    obs = pd.concat([piece.obs for piece in pieces]).sort_values("xl_global_index")
    assert obs["escaper_efficiency"].to_numpy().tobytes() == ref_eff["escaper_efficiency"].to_numpy().tobytes()
    ours, theirs = obs["escaper_knockdown_depth"].to_numpy(), ref_eff["escaper_knockdown_depth"].to_numpy()
    assert (np.isnan(ours) == np.isnan(theirs)).all()
    assert np.allclose(ours, theirs, rtol=1e-5, equal_nan=True)

    # ---- per-perturbation float32 means bit-equal to scipy on SPORE's matrix
    survivor_labels = source["labels"][rows]
    n_checked = 0
    for label in sorted(set(survivor_labels) - {run.ctrl_label}):
        scipy_mean = np.asarray(reference_matrix[survivor_labels == label].mean(axis=0)).ravel()
        assert permeans[label].tobytes() == scipy_mean.tobytes(), label
        n_checked += 1
    print(f"    {n_shards} shard(s): {len(survivors_ref)} survivors, {len(expected_dropped)} undersized dropped, "
          f"{n_checked} per-perturbation means bit-equal, {n_blocks_total} blocks ({layout['n_control_blocks']} control)")


def test_map2_gather(n_shards):
    check(n_shards)


if __name__ == "__main__":
    sys.exit(1 if run_all(test_map2_gather) else 0)
