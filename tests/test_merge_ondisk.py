"""On-disk merge (anndata concat_on_disk) equals the in-memory merge on the 60k synthetic (up to row order, which is the block
order on disk), carries the train hvg_stats uns, and the cap rule keeps the parts when the projection exceeds the cap.
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; set XL_TEST_TMP=<scratch> ; python -B tests\test_merge_ondisk.py"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import get_run, scratch_for  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import yaml  # noqa: E402

import branch  # noqa: E402
from xl_common import Run  # noqa: E402
import cli as spore_xl  # noqa: E402

SPLIT_RANK = {"train": 0, "val": 1, "test": 2}


def make_branch(run_a, label, xl_updates):
    cfg = yaml.safe_load(Path(run_a.config_path).read_text(encoding="utf-8"))
    run_dir = scratch_for(f"merge_{label}")
    cfg["paths"]["project_root"] = str(run_dir).replace("\\", "/")
    cfg["xl"].update(xl_updates)
    config_path = run_dir.parent / f"{run_dir.name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    branch.branch(run_a.config_path, config_path, "finalize")
    return Run(config_path)


def canonical(adata, kind):
    obs = adata.obs
    if kind == "cells":
        order = np.argsort(obs["xl_global_index"].to_numpy(), kind="stable")
    else:
        rank = obs["xl_control_split"].astype(str).map(SPLIT_RANK).fillna(0).astype(int).to_numpy()
        order = np.lexsort((obs["xl_unit"].to_numpy(), obs["xl_global_index"].to_numpy(), rank, obs["gene"].astype(str).to_numpy()))
    return adata[order]


def same(left, right):
    left, right = sp.csr_matrix(left.X), sp.csr_matrix(right.X)
    return (left.shape == right.shape and np.array_equal(left.indptr, right.indptr) and np.array_equal(left.indices, right.indices)
            and left.data.tobytes() == right.data.tobytes())


def test_ondisk_equals_in_memory_merge():
    run_a = get_run(3, through="finalize")  # xl.merge_mode: memory (SPORE row order)
    run_b = make_branch(run_a, "ondisk", {"merge_mode": "ondisk"})
    spore_xl.run_stage(run_b, "finalize")
    names = [f"{split}_{kind}" for split in ("train", "val", "test") for kind in ("hvgraw", "p9")]
    for name in names:
        a = ad.read_h5ad(run_a.processed_dir / f"{run_a.name}_{name}.h5ad")
        b = ad.read_h5ad(run_b.processed_dir / f"{run_b.name}_{name}.h5ad")
        a, b = canonical(a, "cells"), canonical(b, "cells")
        assert list(a.obs_names) == list(b.obs_names) and list(a.var_names) == list(b.var_names), name
        assert same(a, b), name
        assert list(a.obs.columns) == list(b.obs.columns)
    for name in ("allsplits_metacell_ctrlpreserved", "train_hybrid"):
        a = ad.read_h5ad(run_a.processed_dir / f"{run_a.name}_{name}.h5ad")
        b = ad.read_h5ad(run_b.processed_dir / f"{run_b.name}_{name}.h5ad")
        a, b = canonical(a, "units"), canonical(b, "units")
        assert same(a, b), name
        assert list(a.obs["gene"].astype(str)) == list(b.obs["gene"].astype(str))
        assert list(a.obs["n_cells_in_metacell"]) == list(b.obs["n_cells_in_metacell"])
    train_a = ad.read_h5ad(run_a.processed_dir / f"{run_a.name}_train_hvgraw.h5ad")
    train_b = ad.read_h5ad(run_b.processed_dir / f"{run_b.name}_train_hvgraw.h5ad")
    assert sorted(train_a.uns["hvg_stats"]) == sorted(train_b.uns["hvg_stats"])
    assert list(train_a.uns["spore_core_features"]) == list(train_b.uns["spore_core_features"])
    report = (run_b.reports_dir / "COVERAGE_REPORT.md").read_text(encoding="utf-8")
    assert "concat_on_disk" in report and "GRAPH-READY" in report
    peak_a = pd.read_json(run_a.stats_dir / "finalize" / "all.done", typ="series")["peak_rss_gb"]
    peak_b = pd.read_json(run_b.stats_dir / "finalize" / "all.done", typ="series")["peak_rss_gb"]
    print(f"    on-disk merged files equal the in-memory merge (canonical row order); finalize peak RSS {peak_b} GB (memory mode {peak_a} GB)")


def test_cap_keeps_the_parts_and_says_so():
    run_a = get_run(3, through="finalize")
    run_c = make_branch(run_a, "cap", {"merge_mode": "ondisk", "merge_cap_gb": 0.0001})
    spore_xl.run_stage(run_c, "finalize")
    assert not list(run_c.processed_dir.glob("*.h5ad")), "merged files must not be written over the cap"
    assert list(run_c.processed_dir.glob("*_hvgraw/part_*.h5ad"))
    report = (run_c.reports_dir / "COVERAGE_REPORT.md").read_text(encoding="utf-8")
    assert "merged files NOT written" in report and "GRAPH-READY" in report
    print("    over the cap: no merged files, parts kept, the report says so")


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
