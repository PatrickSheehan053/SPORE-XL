"""Step 4 gate: 1, 3 and 7 shards of synthetic_60k concatenate back to the exact original.
Run: python -B tests\\test_shard_planner.py     (output goes under XL_TEST_TMP, never cleaned up)
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
BUILD = Path(os.environ.get("XL_TEST_DATA", str(REPO / "tests" / "ref")))
sys.path.insert(0, str(REPO / "src" / "engine"))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import shard_planner

PERT_COL = "gene"


def find_source():
    """synthetic_60k.h5ad: XL_TEST_DATA/data (reference folder), tests/ref/data (prepare_references.py), or the cluster bundle's
    synthetic/ folder next to the repository (docs/HPC_SETUP.md section 2). None when it is nowhere."""
    candidates = [BUILD / "data" / "synthetic_60k.h5ad", REPO / "tests" / "ref" / "data" / "synthetic_60k.h5ad",
                  REPO.parent / "synthetic" / "synthetic_60k.h5ad"]
    return next((path for path in candidates if path.exists()), None), candidates


SOURCE, _CANDIDATES = find_source()


def _skip_without_source(name):
    if SOURCE is not None:
        return False
    print(f"  SKIP {name}: synthetic_60k.h5ad not found (looked in {[str(path) for path in _CANDIDATES]}); set XL_TEST_DATA to a reference "
          f"folder (tests/prepare_references.py) or unpack the cluster bundle so that synthetic/synthetic_60k.h5ad sits next to the repository")
    return True


def _work_dir(label):
    base = os.environ.get("XL_TEST_TMP", tempfile.gettempdir())
    return Path(tempfile.mkdtemp(prefix=f"xl_shards_{label}_", dir=base))  # not cleaned up


def _assert_same_matrix(left, right):
    left, right = sp.csr_matrix(left), sp.csr_matrix(right)
    assert left.shape == right.shape
    assert left.dtype == right.dtype, (left.dtype, right.dtype)
    assert np.array_equal(left.indptr, right.indptr)
    assert np.array_equal(left.indices, right.indices)
    assert left.data.tobytes() == right.data.tobytes()


def _assert_same_obs(left, right):
    assert list(left.columns) == list(right.columns)
    for column in left.columns:
        assert left[column].dtype == right[column].dtype, column
        assert left[column].equals(right[column]), column
    assert left.index.equals(right.index)


def _check_roundtrip(out_dir, original, sort_by_global_index):
    table, source = shard_planner.load_plan(out_dir)
    assert source["n_obs"] == original.n_obs and source["n_vars"] == original.n_vars
    shards = [shard_planner.read_shard(out_dir, name) for name in table["file"]]
    assert sum(shard.n_obs for shard in shards) == original.n_obs
    for shard in shards:
        assert list(shard.var_names) == list(original.var_names)
    matrix = sp.vstack([shard.X for shard in shards], format="csr")
    obs = pd.concat([shard.obs for shard in shards])
    global_index = obs["xl_global_index"].to_numpy()
    assert len(np.unique(global_index)) == original.n_obs
    if sort_by_global_index:
        order = np.argsort(global_index, kind="stable")
        matrix, obs, global_index = matrix[order], obs.iloc[order], global_index[order]
    assert np.array_equal(global_index, np.arange(original.n_obs))
    for shard in shards:  # within a shard, cells keep the source order
        assert np.all(np.diff(shard.obs["xl_global_index"].to_numpy()) > 0)
    _assert_same_matrix(matrix, original.X)
    obs = obs.drop(columns=["xl_global_index", "xl_shard"])
    # concat of categoricals with identical categories keeps the dtype; compare the category lists too
    assert list(obs[PERT_COL].cat.categories) == list(original.obs[PERT_COL].cat.categories)
    _assert_same_obs(obs, original.obs)
    return table, source


def test_rows_1_3_7_shards_reproduce_original():
    if _skip_without_source("test_rows_1_3_7_shards_reproduce_original"):
        return
    original = ad.read_h5ad(SOURCE)
    n_cells = original.n_obs
    for n_shards in (1, 3, 7):
        shard_rows = -(-n_cells // n_shards)
        out_dir = _work_dir(f"rows{n_shards}")
        table = shard_planner.plan_h5ad_rows(SOURCE, out_dir, shard_rows=shard_rows, pert_col=PERT_COL)
        source = shard_planner.load_plan(out_dir)[1]
        assert len(table) == n_shards, (len(table), n_shards)
        assert source["perturbation_categories"] == list(original.obs[PERT_COL].cat.categories)
        _check_roundtrip(out_dir, original, sort_by_global_index=False)
        print(f"  h5ad_rows: {n_shards} shard(s) concatenate to the original matrix, obs and var exactly")


def test_obs_shards_reproduce_original_after_global_index_sort():
    if _skip_without_source("test_obs_shards_reproduce_original_after_global_index_sort"):
        return
    original = ad.read_h5ad(SOURCE)
    out_dir = _work_dir("obs")
    table = shard_planner.plan_h5ad_obs(SOURCE, out_dir, obs_key="gem_group", pert_col=PERT_COL)
    assert list(table["obs_value"]) == ["1", "2"]
    _check_roundtrip(out_dir, original, sort_by_global_index=True)
    print("  h5ad_obs (gem_group): 2 shards restore the original exactly after sorting on xl_global_index")


def test_rerun_is_noop_and_partial_write_stops():
    if _skip_without_source("test_rerun_is_noop_and_partial_write_stops"):
        return
    out_dir = _work_dir("idem")
    shard_planner.plan_h5ad_rows(SOURCE, out_dir, shard_rows=25000, pert_col=PERT_COL)
    shard_file = out_dir / "shards" / "shard_0000.h5ad"
    stamp = shard_file.stat().st_mtime_ns
    time.sleep(0.05)
    shard_planner.plan_h5ad_rows(SOURCE, out_dir, shard_rows=25000, pert_col=PERT_COL)
    assert shard_file.stat().st_mtime_ns == stamp, "re-run rewrote a finished shard"
    # a different plan on the same directory must be refused, not merged
    try:
        shard_planner.plan_h5ad_rows(SOURCE, out_dir, shard_rows=20000, pert_col=PERT_COL)
        raise AssertionError("different plan over the same directory must fail")
    except (RuntimeError, FileExistsError):
        pass
    # an interrupted shard (file without .done) stops the run and is not deleted
    broken_dir = _work_dir("broken")
    shard_planner.plan_h5ad_rows(SOURCE, broken_dir, shard_rows=40000, pert_col=PERT_COL)
    (broken_dir / "shards" / "shard_0001.h5ad.done").rename(broken_dir / "shards" / "moved.done.txt")
    try:
        shard_planner.plan_h5ad_rows(SOURCE, broken_dir, shard_rows=40000, pert_col=PERT_COL)
        raise AssertionError("shard without .done must stop the run")
    except FileExistsError:
        pass
    assert (broken_dir / "shards" / "shard_0001.h5ad").exists()
    print("  idempotent re-run leaves finished shards untouched; mismatched plan and partial shard are refused")


def test_measure_and_shard_rows_rule():
    if not _skip_without_source("test_measure_and_shard_rows_rule (measure part)"):
        facts = shard_planner.measure_h5ad(SOURCE)
        assert facts["n_obs"] == 60000 and facts["n_vars"] == 2000
        assert abs(facts["nnz_per_cell"] - 44615573 / 60000) < 1e-6
    assert shard_planner.derive_shard_rows(20, 3250) == int(20e9 / (3250 * 12 * 3))  # about 170k cells (blueprint)
    assert 160000 < shard_planner.derive_shard_rows(20, 3250) < 180000
    assert 80000 < shard_planner.derive_shard_rows(10, 3250) < 90000


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except Exception as error:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                failures += 1
                print(f"FAIL {name}: {type(error).__name__}: {error}")
    sys.exit(1 if failures else 0)
