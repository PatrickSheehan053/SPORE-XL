"""Runs the test files one after another as separate processes and writes summary.tsv (file, seconds, exit code, last line).
Resumable: a test that already has a line with exit code 0 in the summary of the same --tag is skipped.

  python -B tests\\run_all_tests.py --tag v4a --logs <new folder> --scratch <folder for test runs> [--data <reference folder>] [--only name,name] [--quick]

--quick runs the tests that need no pipeline run (about a minute). The full list takes about an hour on a laptop (one test at a time).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

QUICK = ["test_stats", "test_banner", "test_shipped_configs", "test_confounding_null", "test_preflight_intake", "test_preflight_blocks", "test_spore_api", "test_compare_env_v4", "test_p8_fixes", "test_intake_adapters", "test_gate_protected", "test_shard_planner", "test_layout"]
FULL = QUICK + ["test_measure", "test_stage_map1_reduce1", "test_stage_map2_gather", "test_stage_reduce2", "test_stage_hvg", "test_stage_hvg_subsample",
                "test_stage_map3", "test_stage_metacell", "test_equivalence", "test_robustness", "test_extract", "test_merge_ondisk", "test_report",
                "test_cleanup", "test_target_policy", "test_intake", "test_intake_genes", "test_run_command", "test_spore_xl_sbatch", "test_external_workdir", "test_production_defaults", "test_escaper_modes", "test_speed_changes",
                "test_workers_spawn", "test_resume_incomplete"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    summary = logs / f"summary_{args.tag}.tsv"
    done = set()
    if summary.exists():
        for line in summary.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2] == "0":
                done.add(parts[0])
    else:
        summary.write_text("test\tseconds\texit\tlast_line\n", encoding="utf-8")
    names = args.only.split(",") if args.only else (QUICK if args.quick else FULL)
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", PYTHONUTF8="1",
               XL_TEST_TMP=str(args.scratch), XL_RUN_TAG=args.tag)
    if args.data:
        env["XL_TEST_DATA"] = str(args.data)
    Path(args.scratch).mkdir(parents=True, exist_ok=True)
    for name in names:
        script = HERE / f"{name}.py"
        if name in done or not script.exists():
            print(f"skip {name}" + ("" if script.exists() else " (no such file)"))
            continue
        started = time.time()
        log_path = logs / f"{name}_{args.tag}.log"
        with open(log_path, "w", encoding="utf-8") as handle:
            code = subprocess.run([sys.executable, "-B", str(script)], stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=str(HERE.parent)).returncode
        last = [line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()][-1:] or [""]
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"{name}\t{time.time() - started:.0f}\t{code}\t{last[0][:160]}\n")
        print(f"{name}: exit {code} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
