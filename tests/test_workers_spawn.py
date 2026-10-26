"""Regression for the Linux crash "Cannot set NUMBA_NUM_THREADS to a different value once the threads have been launched" (HPC1, 19 Oct):
a forked pool inherited a Numba runtime the parent had already started, and the workers then set the environment variable.
Fixed in cli: spawn pools on every platform, Numba limited with numba.set_num_threads(1), NUMBA_NUM_THREADS never touched.

1. NUMBA_NUM_THREADS=4 is set before Numba is imported, and Numba is started in the parent (sc.pp.normalize_total on a tiny AnnData):
   the situation the old code could not handle (workers that set the variable to another value than the running runtime has).
2. every pool worker reports numba.get_num_threads() == 1 and an unchanged NUMBA_NUM_THREADS.
3. all stages of the 60k synthetic in this process with run_stage(workers=3) == the same run with workers=1 (content signatures), and
   the parent's NUMBA_NUM_THREADS is still "4" afterwards (the old code flipped it between stages).
4. the real CLI `all-local --workers 3` in a fresh process with NUMBA_NUM_THREADS unset finishes GRAPH-READY with the same signature.
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; set XL_TEST_TMP=<scratch> ; python -B tests\\test_workers_spawn.py"""
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["NUMBA_NUM_THREADS"] = "4"   # BEFORE Numba is imported: parent and spawned children start with 4 threads (never change it later)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xl_test_support import fresh_run_config, shard_rows_for  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import scanpy as sc  # noqa: E402

import compare_v4 as compare  # noqa: E402
import cli as spore_xl  # noqa: E402
from xl_common import Run  # noqa: E402

STAGES = [stage for stage in spore_xl.ORDER if stage not in ("intake_list", "intake")]


def warm_up_numba_in_parent():
    tiny = ad.AnnData(np.random.default_rng(0).poisson(3.0, size=(50, 30)).astype(np.float32))
    sc.pp.normalize_total(tiny)
    import numba
    return numba.get_num_threads()


def numba_probe(_):
    import numba
    return os.getpid(), numba.get_num_threads(), os.environ.get("NUMBA_NUM_THREADS")


def test_worker_numba_threads():
    assert spore_xl.POOL_START_METHOD == "spawn" and spore_xl.pool_context().get_start_method() == "spawn"
    assert "NUMBA_NUM_THREADS" not in spore_xl.THREAD_VARIABLES
    with ProcessPoolExecutor(max_workers=3, mp_context=spore_xl.pool_context(), initializer=spore_xl._init_worker) as pool:
        results = list(pool.map(numba_probe, range(6)))
    for pid, threads, variable in results:
        assert threads == 1 and variable == "4", (pid, threads, variable)
    print(f"    {len({pid for pid, _, _ in results})} spawned worker(s): numba threads = 1, NUMBA_NUM_THREADS left at {results[0][2]}")


def run_all_stages(config, workers):
    run = Run(config)
    for stage in STAGES:
        spore_xl.run_stage(run, stage, workers=workers, max_ram_gb=20)
    return run


def test_pool_after_numba_matches_serial():
    rows = shard_rows_for(6)
    serial = run_all_stages(fresh_run_config("spawn_serial", shard_rows=rows), 1)
    pooled = run_all_stages(fresh_run_config("spawn_pool3", shard_rows=rows), 3)
    assert os.environ["NUMBA_NUM_THREADS"] == "4", "the parent's NUMBA_NUM_THREADS must not be flipped by run_stage"
    assert compare.xl_signature(serial) == compare.xl_signature(pooled), "workers=3 differs from workers=1"
    print("    run_stage workers=3 after Numba started in the parent: content signatures equal to workers=1")
    return serial


def test_cli_all_local_workers3(serial):
    config = fresh_run_config("spawn_cli", shard_rows=shard_rows_for(6))
    environment = {key: value for key, value in os.environ.items() if key != "NUMBA_NUM_THREADS"}
    result = subprocess.run([sys.executable, "-B", str(HERE.parent / "spore_xl.py"), "all-local", "--config", str(config),
                             "--workers", "3", "--max-ram-gb", "20"], capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-2500:]
    assert "GRAPH-READY" in result.stdout and "NUMBA_NUM_THREADS" not in result.stderr
    assert result.stdout.count("spore fact:") == 1, "the start-up banner must print once in the main process, never in spawned workers"
    assert compare.xl_signature(Run(config)) == compare.xl_signature(serial)
    print("    CLI all-local --workers 3 (fresh process): GRAPH-READY, signatures equal to workers=1")


if __name__ == "__main__":
    print(f"    parent numba threads at start: {warm_up_numba_in_parent()}")
    test_worker_numba_threads()
    reference = test_pool_after_numba_matches_serial()
    test_cli_all_local_workers3(reference)
    print("PASS test_workers_spawn")
