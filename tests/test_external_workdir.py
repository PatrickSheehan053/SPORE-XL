"""xl.work_dir as an ABSOLUTE path outside the run directory (HPC1: outputs in /group, working files in /scratch).
Two fresh 60k runs through the real CLI, one with the default _xl_work inside the run directory, one with the work dir elsewhere:
all content signatures equal, verify.py passes, no work files inside the external run directory, manifest holds absolute shard paths.
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; set XL_TEST_TMP=<scratch> ; python -B tests\\test_external_workdir.py"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xl_test_support import fresh_run_config, shard_rows_for  # noqa: E402

import yaml  # noqa: E402

import compare_v4 as compare  # noqa: E402
from xl_common import Run  # noqa: E402

PY = sys.executable
CLI = str(HERE.parent / "spore_xl.py")


def run_cli(*arguments):
    result = subprocess.run([PY, "-B", CLI, *arguments], capture_output=True, text=True, env=dict(os.environ))
    if result.returncode != 0:
        raise RuntimeError(result.stdout[-1500:] + result.stderr[-2500:])
    return result.stdout


def test_external_work_dir():
    inside = fresh_run_config("workdir_inside", shard_rows=shard_rows_for(3))
    outside = fresh_run_config("workdir_outside", shard_rows=shard_rows_for(3))
    external = Path(tempfile.mkdtemp(prefix="xl_external_work_", dir=os.environ.get("XL_TEST_TMP", tempfile.gettempdir())))
    cfg = yaml.safe_load(Path(outside).read_text(encoding="utf-8"))
    cfg["xl"]["work_dir"] = str(external).replace("\\", "/")
    Path(outside).write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run_cli("all-local", "--config", str(inside))
    log = run_cli("all-local", "--config", str(outside))
    assert "GRAPH-READY" in log and "disk pre-flight" in log, "plan must log its disk pre-flight"
    run_in, run_out = Run(inside), Run(outside)
    assert not (run_out.run_dir / "_xl_work").exists() and (external / "map1").exists() and (external / "gather").exists()
    shards = run_out.shards()
    assert all(Path(name).is_absolute() and str(external).replace("\\", "/") in name for name in shards["file"]), list(shards["file"])
    assert compare.xl_signature(run_in) == compare.xl_signature(run_out), "outputs differ with an external work dir"
    verify_out = subprocess.run([PY, "-B", str(HERE.parent / "src" / "engine" / "verify.py"), str(outside)], capture_output=True, text=True, env=dict(os.environ))
    assert verify_out.returncode == 0, verify_out.stdout[-1500:] + verify_out.stderr[-1500:]
    rerun = run_cli("all-local", "--config", str(outside))
    assert rerun.count("already done") >= 5 and " done in " not in rerun, "a finished run must re-run as a no-op (the pooled stages with nothing pending print nothing per unit)"
    print(f"    external work dir {external.name}: content signatures equal to the in-run work dir, verify.py PASS, re-run is a no-op")


if __name__ == "__main__":
    test_external_work_dir()
    print("PASS test_external_workdir")
