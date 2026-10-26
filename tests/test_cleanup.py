"""Opt-in clean-up: OFF by default; refused without BOTH approved_by: Patrick and approved_on; when approved, verified predecessor
files are removed, final outputs are identical to a run without clean-up, peak disk drops, and reports/cleanup_log.tsv is complete.
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; set XL_TEST_TMP=<scratch> ; python -B tests\\test_cleanup.py"""
import hashlib
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import scratch_for, fresh_run_config, shard_rows_for, file_digest  # noqa: E402

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import compare_v4 as compare  # noqa: E402
import cli as spore_xl  # noqa: E402
import verify  # noqa: E402
from xl_common import Run  # noqa: E402

STAGES = ["plan", "map1", "reduce1", "map2", "gather", "reduce2", "hvg", "map3", "metacell_prep", "metacell", "finalize", "report"]


def dir_bytes(root):
    total = 0
    for path in Path(root).rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            pass
    return total


def run_with_peak_disk(config_path):
    run = Run(config_path)
    peak = {"bytes": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak["bytes"] = max(peak["bytes"], dir_bytes(run.run_dir))
            time.sleep(0.05)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        for stage in STAGES:
            spore_xl.run_stage(run, stage)
    finally:
        stop.set()
        thread.join()
    peak["bytes"] = max(peak["bytes"], dir_bytes(run.run_dir))
    return run, peak["bytes"]


def config_with(label, cleanup=None):
    path = fresh_run_config(label, shard_rows=shard_rows_for(3))
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cleanup is not None:
        cfg["xl"]["cleanup"] = cleanup
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def final_bytes(run):
    digest = {}
    for folder in (run.processed_dir, run.splits_dir):
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file():
                digest[str(path.relative_to(run.run_dir))] = file_digest(path)
    return digest


def test_incomplete_approval_is_refused_before_anything_happens():
    for section in ({}, {"approved_by": "Patrick"}, {"approved_on": "2026-10-19"}, {"approved_by": "Someone", "approved_on": "2026-10-19"}):
        config = config_with("cleanup_bad", cleanup=section)
        run = Run(config)
        try:
            spore_xl.run_stage(run, "plan")
            spore_xl.run_stage(run, "map1")
            raise AssertionError(f"section {section} must be refused")
        except ValueError as error:
            assert "approved_by: Patrick" in str(error)
        assert not (run.reports_dir / "cleanup_log.tsv").exists()
    print("    xl.cleanup without both approved_by: Patrick and approved_on is refused; nothing removed")


def test_cleanup_on_equals_off_and_lowers_peak_disk():
    off_run, off_peak = run_with_peak_disk(config_with("cleanup_off"))
    on_run, on_peak = run_with_peak_disk(config_with("cleanup_on", {"approved_by": "Patrick", "approved_on": "2026-10-19"}))
    assert not (off_run.reports_dir / "cleanup_log.tsv").exists(), "default must not clean anything"
    log = pd.read_csv(on_run.reports_dir / "cleanup_log.tsv", sep="\t")
    assert set(log["stage"]) == {"source_shards", "qc_shards", "buckets"}, set(log["stage"])
    assert all(not Path(path).exists() for path in log["path"])
    assert all(str(on_run.work_dir.resolve()) in path for path in log["path"])
    n_shards = on_run.n_shards()
    assert (log["stage"] == "source_shards").sum() == n_shards and (log["stage"] == "qc_shards").sum() == n_shards
    bucket_files_off = len(list(off_run.work_dir.glob("map2/shard_*/block_*.h5ad")))
    assert (log["stage"] == "buckets").sum() == bucket_files_off
    removed_bytes = int(log["size_bytes"].sum())
    assert (log["approved_by"] == "Patrick").all()
    # blocks (needed for re-runs, branch and extract) stay unless include_blocks
    assert len(list(on_run.work_dir.glob("gather/block_*/block.h5ad"))) == len(list(off_run.work_dir.glob("gather/block_*/block.h5ad")))
    off_files, on_files = final_bytes(off_run), final_bytes(on_run)
    assert set(off_files) == set(on_files)
    assert off_files == on_files, "final outputs differ with clean-up on"
    assert compare.xl_signature(off_run) == compare.xl_signature(on_run)
    failed = [name for name, ok, _ in verify.verify_run(on_run, write_report=False) if not ok]
    assert not failed, failed
    for stage in STAGES:  # re-running is still a no-op
        spore_xl.run_stage(on_run, stage)
    assert on_peak < off_peak, (on_peak, off_peak)
    print(f"    clean-up ON == OFF on all final outputs; {len(log)} files ({removed_bytes / 1e6:.0f} MB) removed and logged; "
          f"peak run-dir size {on_peak / 1e6:.0f} MB vs {off_peak / 1e6:.0f} MB without clean-up")
    assert "Clean-up (Patrick-approved" in (on_run.reports_dir / "REPORT.md").read_text(encoding="utf-8")


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
