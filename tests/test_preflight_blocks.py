"""Block-stage RAM pre-flight budgets the LARGEST block (CD4 Rest, 22 Oct, 10 workers of 16 CPUs; r4 raises the workers to the CPUs, so an underestimate is a safety issue):
 metacell: the estimate was 3.73 GB (file-size rule) while the control blocks 0128 / 0129 / 0130 peaked at 16.51 / 16.70 / 12.29 GB (the other blocks 3.1 to 5.1 GB);
           now the largest block's dense working set sets it: cells x panel genes x 4 bytes x 12 (about 110,000 cells x 5,000 genes -> 26.9 GB)
 gather:   the estimate was 10.75 GB while blocks 0128 / 0129 peaked at 13.4 / 13.6 GB; the expansion is now 7.7 (5.14 measured x 1.5): 20.0 GB for the same input
The test pins the formulas and that each estimate covers the measured worst peak with at least 1.4x, and that the typical blocks do not raise it.
Run: python -B tests\\test_preflight_blocks.py   (no pipeline run)"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "engine"))

import preflight  # noqa: E402


def fake_run():
    folder = Path(tempfile.mkdtemp(prefix="preflight_blocks_", dir=__import__("os").environ.get("XL_TEST_TMP", tempfile.gettempdir())))
    return SimpleNamespace(name="cd4", work_dir=folder / "work", processed_dir=folder / "processed"), folder


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_metacell_budgets_the_largest_block():
    run, _ = fake_run()
    cells = {}
    for split in ("train", "val", "test"):
        for block, n_cells in ((3, 4000), (127, 3000), (128, 109_700), (129, 109_800), (130, 80_000)):
            path = touch(run.processed_dir / f"cd4_{split}_hvgraw" / f"part_{block:04d}.h5ad")
            cells[(path.parent.name, path.name)] = n_cells if block >= 127 and split == "train" else (n_cells // 100 if block >= 127 else n_cells // 3)
    originals = (preflight._size_gb, preflight._h5ad_shape)
    preflight._size_gb = lambda path: 0.25                                    # the file-size rule alone would say 0.7 + 3 x 0.25 x 4 = 3.7 GB (the 22 Oct estimate)
    preflight._h5ad_shape = lambda path: (cells[(Path(path).parent.name, Path(path).name)], 5000)
    try:
        estimate = preflight.estimate_stage_ram_gb(run, "metacell")
    finally:
        preflight._size_gb, preflight._h5ad_shape = originals
    largest = max(sum(cells[(f"cd4_{split}_hvgraw", f"part_{block:04d}.h5ad")] for split in ("train", "val", "test")) for block in (3, 127, 128, 129, 130))
    assert abs(estimate - (0.7 + largest * 5000 * 4 * preflight.METACELL_DENSE_FACTOR / 1e9)) < 1e-6, estimate
    assert estimate >= 1.4 * 16.70, f"must cover the measured worst control-block peak (16.70 GB) with 1.4x: {estimate:.1f} GB"
    assert estimate > 3.73 * 5, "the old file-size estimate (3.73 GB) is gone"
    print(f"    metacell: the largest block ({largest:,} cells x 5,000 genes) sets {estimate:.1f} GB (was 3.73 GB; measured worst 16.70 GB)")


def test_typical_blocks_alone_do_not_raise_the_metacell_estimate():
    run, _ = fake_run()
    for block in (3, 4, 5):
        touch(run.processed_dir / "cd4_train_hvgraw" / f"part_{block:04d}.h5ad")
    originals = (preflight._size_gb, preflight._h5ad_shape)
    preflight._size_gb = lambda path: 0.25
    preflight._h5ad_shape = lambda path: (1500, 5000)
    try:
        estimate = preflight.estimate_stage_ram_gb(run, "metacell")
    finally:
        preflight._size_gb, preflight._h5ad_shape = originals
    assert abs(estimate - (0.7 + max(0.25 * 4.0, 1500 * 5000 * 4 * preflight.METACELL_DENSE_FACTOR / 1e9))) < 1e-6 and estimate < 3.0, estimate
    print(f"    metacell: blocks of 1,500 cells give {estimate:.2f} GB (the file-size rule), nothing inflated")


def test_gather_covers_the_measured_peaks():
    run, _ = fake_run()
    for shard in (0, 1):
        touch(run.work_dir / "map2" / f"shard_{shard:04d}" / "block_0128.h5ad")
        touch(run.work_dir / "map2" / f"shard_{shard:04d}" / "block_0003.h5ad")
    originals = preflight._size_gb
    preflight._size_gb = lambda path: 1.255 if path.name == "block_0128.h5ad" else 0.3      # two pieces of 1.255 GB = the 2.51 GB block of 22 Oct
    try:
        estimate = preflight.estimate_stage_ram_gb(run, "gather")
    finally:
        preflight._size_gb = originals
    assert abs(estimate - (0.7 + 2.51 * 7.83)) < 1e-9, estimate
    assert estimate >= 1.45 * 13.6 and estimate > 10.75, f"the old estimate (10.75 GB) was below the measured 13.6 GB: {estimate:.1f}"
    print(f"    gather: the 2.51 GB block gives {estimate:.1f} GB (was 10.75 GB; measured worst 13.6 GB)")


def test_constants_cover_the_three_calibration_runs():
    """tools/memory_calibration.py on CD4 / K562 compat / K562 production: worst measured (peak - 0.7 GB) / input per stage; each constant is that x 1.5 (max over the runs)."""
    measured = {"map1": [3.60, 4.09, 4.11], "map2": [1.44, 1.12, 1.14], "gather": [5.22, 4.69, 4.69], "map3": [2.84, 4.37, 3.41]}
    for stage, ratios in measured.items():
        constant = preflight.STAGE_EXPANSION[stage]
        assert constant >= 1.49 * max(ratios), (stage, constant, max(ratios))
        assert constant < 1.6 * max(ratios), (stage, "not an inflated constant")
    assert preflight.STAGE_EXPANSION == {"map1": 6.16, "map2": 2.17, "gather": 7.83, "map3": 6.56} and preflight.INTAKE_BYTES_PER_VALUE == 22.2
    # metacell (largest-block rule): K562 control blocks of 40,000 rows (control_block_rows) x the panel (7,087 genes in compat, 5,000 in production) against the measured 9.57 / 6.80 GB
    for panel, measured_peak in ((7087, 9.57), (5000, 6.80)):
        estimate = 0.7 + 37_600 * panel * 4 * preflight.METACELL_DENSE_FACTOR / 1e9
        assert estimate >= 1.4 * measured_peak, (panel, estimate, measured_peak)
    print("    constants: map1 6.16, map2 2.17, gather 7.83, map3 6.56, intake 22.2 B per value cover the worst measured point of the three runs x 1.5; metacell largest-block rule covers K562 (>= 1.4x)")


if __name__ == "__main__":
    test_constants_cover_the_three_calibration_runs()
    test_metacell_budgets_the_largest_block()
    test_typical_blocks_alone_do_not_raise_the_metacell_estimate()
    test_gather_covers_the_measured_peaks()
    print("PASS test_preflight_blocks")
