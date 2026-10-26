"""P10 speed changes give the same results: vectorized histogram merge == sequential merge; parallel plan sharding == serial sharding;
merged files made by `merge` equal the merge made inside finalize.
Run: set PYTHONHASHSEED=0 ; python -B tests\\test_speed_changes.py   (XL_TEST_TMP, XL_TEST_DATA)"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import file_digest, fresh_run_config, scratch_for, shard_rows_for  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cli as spore_xl  # noqa: E402
from stats import IntHistogramBank  # noqa: E402
from xl_common import Run  # noqa: E402


def test_histogram_merge_vectorized_equals_sequential():
    rng = np.random.default_rng(11)
    folder = scratch_for("histogram_merge")
    paths, banks = [], []
    for index in range(7):
        bank = IntHistogramBank()
        for gene in rng.choice(300, size=int(rng.integers(0, 120)), replace=False):
            n_cells = int(rng.integers(5, 400))
            values = rng.poisson(rng.uniform(0.1, 30), size=n_cells)
            bank.add_column(str(int(gene)), values[values > 0], n_cells=n_cells)
        path = folder / f"bank_{index}.npz"
        bank.save(path)
        paths.append(path)
        banks.append(bank)
    sequential = IntHistogramBank()
    for path in paths:
        sequential.add(IntHistogramBank.load(path))
    merged = IntHistogramBank.merge_files(paths)
    assert merged == sequential
    for gene in sequential.genes():
        for quantile in (1, 10, 50, 90):
            assert merged.percentile(gene, quantile, np.float32) == sequential.percentile(gene, quantile, np.float32)
    assert IntHistogramBank.merge_files([]) == IntHistogramBank()
    empty = IntHistogramBank()
    empty.save(folder / "empty.npz")
    assert IntHistogramBank.merge_files([folder / "empty.npz"]) == IntHistogramBank()
    print(f"    {len(sequential.genes())} genes, 7 files: vectorized merge == sequential merge (tables and percentiles)")


def test_parallel_plan_equals_serial_plan():
    serial_config = fresh_run_config("plan_serial", shard_rows=shard_rows_for(3))
    parallel_config = fresh_run_config("plan_parallel", shard_rows=shard_rows_for(3))
    runs = []
    for config, workers in ((serial_config, 1), (parallel_config, 3)):
        run = Run(config)
        run.xl["plan_workers"] = workers
        spore_xl.run_stage(run, "plan")
        runs.append(run)
    left, right = runs
    for column in ("shard", "n_cells", "row_start", "row_stop"):
        assert left.shards()[column].tolist() == right.shards()[column].tolist()
    for index in range(left.n_shards()):
        a, b = left.run_dir / left.shards()["file"][index], right.run_dir / right.shards()["file"][index]
        assert file_digest(a) == file_digest(b), f"shard {index} differs"
    for name in ("labels.tsv", "ribo_exempt.txt", "layout.json"):
        assert (left.manifest_dir / name).read_bytes() == (right.manifest_dir / name).read_bytes(), name
    print(f"    {left.n_shards()} shards written by 1 and by 3 workers: identical content, manifest files identical")


def test_merge_command_equals_inline_merge():
    inline_config = fresh_run_config("merge_inline", shard_rows=shard_rows_for(2))
    later_config = fresh_run_config("merge_later", shard_rows=shard_rows_for(2))
    import yaml
    inline_cfg = yaml.safe_load(Path(inline_config).read_text(encoding="utf-8"))
    inline_cfg["xl"].update({"merge_split_outputs": "always", "merge_mode": "ondisk"})
    Path(inline_config).write_text(yaml.safe_dump(inline_cfg), encoding="utf-8")
    later_cfg = yaml.safe_load(Path(later_config).read_text(encoding="utf-8"))
    later_cfg["xl"].update({"merge_split_outputs": "never", "merge_mode": "ondisk"})
    Path(later_config).write_text(yaml.safe_dump(later_cfg), encoding="utf-8")
    runs = [Run(inline_config), Run(later_config)]
    for run in runs:
        for stage in ["plan", "map1", "reduce1", "map2", "gather", "reduce2", "hvg", "map3", "metacell_prep", "metacell", "finalize"]:
            spore_xl.run_stage(run, stage)
    inline, later = runs
    assert not list(later.processed_dir.glob("*.h5ad")), "finalize with merge_split_outputs: never writes no merged file"
    spore_xl.run_stage(later, "merge")
    names = sorted(path.name for path in inline.processed_dir.glob("*.h5ad"))
    assert names and names == sorted(path.name for path in later.processed_dir.glob("*.h5ad"))
    for name in names:
        assert file_digest(inline.processed_dir / name) == file_digest(later.processed_dir / name), name
    print(f"    {len(names)} merged files: `merge` after finalize == merge inside finalize")


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
