"""measure: header-only facts and projections for an .h5ad, 10x members inside a .tar, and loose .h5 files.
Run: python -B tests\test_measure.py   (XL_TEST_TMP = scratch folder)"""
import sys
import tarfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import SOURCE, scratch_for  # noqa: E402
import make_flex_synthetic as flex  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import measure  # noqa: E402
import cli as spore_xl  # noqa: E402


def test_tar_members_header_only():
    work = scratch_for("measure_tar")
    members = [(flex.member_name(9393800 + i, 1, 9 + i, 1 + i, "Rest"), 800 + 100 * i) for i in range(3)]
    members.append((flex.member_name(9393900, 1, 14, 1, "Stim8hr"), 500))
    flex.build_tar(work / "h5", work / "x.tar", members)
    table = measure.measure_source(work / "x.tar", "_Rest_")
    assert list(table["cells"]) == [800, 900, 1000] and (table["gene_features"] == 200).all() and (table["guide_features"] == 86).all()
    # header-only proof: a member WITHOUT its data and indices datasets still measures (only indptr, shape, feature types are read)
    stripped = work / "stripped.h5"
    with h5py.File(work / "h5" / members[0][0], "r") as source, h5py.File(stripped, "w") as target:
        group = target.create_group("matrix")
        for name in ("indptr", "shape"):
            group.create_dataset(name, data=source["matrix"][name][...])
        source.copy("matrix/features", group)
    with h5py.File(stripped, "r") as handle:
        facts = measure.measure_10x_handle(handle)
    assert facts["cells"] == 800 and facts["nnz"] > 0
    projection = measure.project(table, n_blocks=8, workers=4)
    assert projection["total_cells"] == 2700 and projection["peak_task_ram_gb"] > 0.7
    print("    tar members measured from headers only (a member without data/indices still measures)")


def test_h5ad_and_cli_refusal():
    table = measure.measure_source(SOURCE)
    assert int(table["cells"].iloc[0]) == 60000 and int(table["nnz"].iloc[0]) == 44615573
    out = scratch_for("measure_out") / "m"
    code = spore_xl.main(["measure", "--source", str(SOURCE), "--out", str(out), "--shard-rows", "20000", "--n-blocks", "8"])
    assert code == 0 and (out / "measure.tsv").exists() and (out / "measure_projection.md").exists()
    projection = pd.read_json(out / "measure_projection.json", typ="series")
    assert projection["disk_gb_with_cleanup_peak"] < projection["disk_gb_without_cleanup"]
    code = spore_xl.main(["measure", "--source", str(SOURCE), "--shard-rows", "20000", "--workers", "50", "--max-ram-gb", "10"])
    assert code == 2, "workers x RAM over the budget must be refused"
    print("    h5ad measured (60,000 cells, 44,615,573 nonzeros); projection with and without clean-up; --max-ram-gb refusal")


def test_loose_h5_folder():
    work = scratch_for("measure_dir")
    flex.write_flex_h5(work / "a_Rest_x.h5", 300, 1)
    table = measure.measure_source(work, "_Rest_")
    assert len(table) == 1 and int(table["cells"].iloc[0]) == 300


def test_disk_model_reproduces_the_rpe1_run():
    # session 22: the finished RPE1 run measured 34.49 GB for 867,313,640 source values (survival 0.827, panel share 0.548, merged files)
    work, outputs = measure.disk_projection(867_313_640, 0.8269, 0.548, merged=True)
    total = (sum(work.values()) + sum(outputs.values())) * 1.01
    assert 33.0 <= total <= 36.0, total
    no_merge = sum(measure.disk_projection(867_313_640, 0.8269, 0.548, merged=False)[1].values())
    assert no_merge < sum(outputs.values())
    assert abs(measure.panel_fraction_estimate(5000, 8739) - 0.572) < 0.001 and measure.panel_fraction_estimate(5000, 36000) == 0.25
    print(f"    disk model on the RPE1 inputs: {total:.1f} GB (measured 34.5)")


def test_recommended_workers_rule():
    assert measure.recommend_workers(16, 110, 17.6) == 6            # RAM-bound (CD4-like)
    assert measure.recommend_workers(16, 110, 5.3) == 14            # CPU-bound: CPUs - 2
    assert measure.recommend_workers(2, 110, 5.3) == 1              # never below 1
    table = measure.measure_source(SOURCE)
    projection = measure.project(table, shard_rows=20000, n_blocks=8, cpus=8, max_ram_gb=3.0)
    assert 1 <= projection["recommended_workers"] <= 6 and projection["cpus"] == 8 and projection["max_ram_gb"] == 3.0
    assert projection["disk_gb_without_cleanup_with_margin"] > projection["disk_gb_without_cleanup"]
    print(f"    rule min(CPUs-2, floor(RAM/peak)): 16/110/17.6 -> 6; 60k with 8 CPUs and 3 GB -> {projection['recommended_workers']}")


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
