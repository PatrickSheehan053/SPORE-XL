"""SPORE-XL v4.0 command line (the module behind spore_xl.py).

ONE command runs everything:   python spore_xl.py run --config my_run.yaml
  checks that every input path exists, does the intake (recipe / tar sources), the optional protected-list expression gate, plan ... report, then the verification
  (reports/VERIFICATION.md; exit code 1 on any FAIL, 2 when an input path is missing). Workers per stage = min(CPUs, units, RAM budget / per-task estimate), lowered
  and logged when the RAM budget cannot carry them (--workers and --max-ram-gb are optional limits). `all-local` is a hidden alias of `run`.
Stage-level commands for developers (plan map1 reduce1 ... report, merge; --local runs every unit in this process, --workers N a pool, --index N one unit),
and measure, convert (intake adapters), branch, cleanup. Worker pools always use the SPAWN start method and never touch NUMBA_NUM_THREADS (on Linux a fork pool inherited
a Numba runtime the parent had already started with 16 threads, and setting the variable in the child raised "Cannot set NUMBA_NUM_THREADS to a different value once
the threads have been launched").
Set PYTHONHASHSEED=0 for compat-mode runs (recorded in config/environment.txt by `plan`).
"""
import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stage_gather  # noqa: E402
import stage_map1  # noqa: E402
import stage_map2  # noqa: E402
import stage_plan  # noqa: E402
import stage_reduce1  # noqa: E402
import stage_reduce2  # noqa: E402
from xl_common import Run  # noqa: E402

import stage_hvg  # noqa: E402
import stage_map3  # noqa: E402
import stage_metacell  # noqa: E402
import stage_finalize  # noqa: E402
import stage_report  # noqa: E402
import stage_intake  # noqa: E402
import stage_merge  # noqa: E402

SHARD_STAGES = {"map1": stage_map1.run_map1, "map2": stage_map2.run_map2, "intake": stage_intake.run_intake}
BLOCK_STAGES = {"gather": stage_gather.run_gather, "map3": stage_map3.run_map3, "metacell": stage_metacell.run_metacell}
SINGLE_STAGES = {"plan": stage_plan.run_plan, "reduce1": stage_reduce1.run_reduce1, "reduce2": stage_reduce2.run_reduce2,
                 "hvg": stage_hvg.run_hvg, "metacell_prep": stage_metacell.run_metacell_prep,
                 "finalize": stage_finalize.run_finalize, "report": stage_report.run_report,
                 "intake_list": stage_intake.run_intake_list, "merge": stage_merge.run_merge}
ORDER = ["intake_list", "intake", "plan", "map1", "reduce1", "map2", "gather", "reduce2", "hvg", "map3", "metacell_prep", "metacell", "finalize", "report"]
OPTIONAL = ["merge"]  # not part of all-local: `spore_xl.py merge --config run.yaml` after finalize


THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")  # NOT Numba: see _init_worker
VERSION = "4.0"
POOL_START_METHOD = "spawn"   # the same on every platform, so Linux runs exactly the path that is tested on Windows


def pool_context():
    import multiprocessing
    return multiprocessing.get_context(POOL_START_METHOD)


def _init_worker():
    """One BLAS/OpenMP/Numba thread per worker, so N workers do not oversubscribe the CPUs. BLAS libraries read the variables when
    they load (the parent sets them around the pool, spawned children inherit them); threadpoolctl covers libraries loaded earlier.
    Numba is limited through its runtime API: its environment variable cannot be changed once threads were launched, and
    set_num_threads(1) is always allowed (1 is not above the number of threads the runtime started with)."""
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass
    try:
        import numba
        numba.set_num_threads(1)
    except ImportError:
        pass


def _unit_worker(config_path, stage, index):
    """Runs one unit in a child process (each child builds its own Run from the config; units share nothing but files)."""
    run = Run(config_path)
    function = SHARD_STAGES.get(stage) or BLOCK_STAGES[stage]
    function(run, index)
    return index


def _pending(run, stage, count):
    root = "stats" if stage in ("map1", "map2", "gather", "metacell", "intake") else "work"
    from xl_common import block_unit, shard_unit
    name = shard_unit if stage in SHARD_STAGES else block_unit
    return [index for index in range(count) if not run.marker(root, stage, name(index)).exists()]


def run_stage(run, stage, workers=1, max_ram_gb=None):
    """Run every unit of one stage (_run_stage); after the whole intake, every shard must have the run's one gene list."""
    _run_stage(run, stage, workers, max_ram_gb)
    if stage == "intake":
        checked = stage_intake.assert_uniform_var(run)
        run.logger.info(f"[intake] gene list is uniform: {checked} shards have exactly the {len(stage_intake.canonical_genes(run)['ids']):,} genes of intake_genes.tsv")


def _n_units(run, stage):
    if stage in SHARD_STAGES:
        return stage_intake.n_members(run) if stage == "intake" else run.n_shards()
    return stage_gather.total_blocks(run)


def _run_stage(run, stage, workers=1, max_ram_gb=None):
    """Run every unit of one stage: serially, or in a process pool of `workers` (outputs are identical either way). The pre-flight lowers the workers to what the RAM budget carries."""
    import cleanup
    if stage in SINGLE_STAGES:
        SINGLE_STAGES[stage](run)
        cleanup.after_stage(run, stage)
        return
    count = _n_units(run, stage)
    function = SHARD_STAGES.get(stage) or BLOCK_STAGES[stage]
    if workers > 1:
        import preflight
        workers = preflight.check_workers(run, stage, workers, max_ram_gb)
    if workers <= 1:
        for index in range(count):
            function(run, index)
        cleanup.after_stage(run, stage)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed
    log_path = run.reports_dir / "memory_and_time.tsv"
    if not log_path.exists():
        log_path.write_text("stage\tunit\tseconds\tpeak_rss_gb\n", encoding="utf-8")  # header before the children append rows
    todo = _pending(run, stage, count)
    if not todo:
        cleanup.after_stage(run, stage)
        return
    saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    os.environ.update({name: "1" for name in THREAD_VARIABLES})
    try:
        with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=pool_context(), initializer=_init_worker) as pool:
            futures = [pool.submit(_unit_worker, str(run.config_path), stage, index) for index in todo]
            for future in as_completed(futures):
                future.result()  # re-raises the first failure; the pool cancels what has not started
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    cleanup.after_stage(run, stage)


def show_banner(run=None, workers=None):
    """Start-up banner, once per command, from the main process only (spawned workers never run _main)."""
    from src.report import banner
    import run_command
    banner.print_banner(VERSION, run=run.name if run else None, mode=run.mode if run else None, workers=workers, context=run_command.run_context())


def _run_command(args, console):
    """`spore_xl.py run`: input check, intake, gate, plan ... report, verification. Exit codes: 0 all PASS, 1 a verification FAIL, 2 an input path is missing."""
    import yaml
    import run_command
    config_path = Path(args.config)
    context, cpus, budget = run_command.run_context(), run_command.available_cpus(), run_command.ram_budget_gb(args.max_ram_gb)
    cap = args.workers or cpus
    if not config_path.exists():
        show_banner(None, cap)
        print(f"STOP: the config {config_path} does not exist. Nothing was run.")
        return 2
    problems = run_command.input_problems(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}, Path(__file__).resolve().parents[2])
    if problems:
        show_banner(None, cap)
        for what, missing in problems:
            print(f"STOP: {what} = {missing} does not exist.")
        print("Nothing was run (check every input path with `ls -l` before launching).")
        return 2
    run = Run(args.config)
    console.attach(run.logs_dir, "run")
    show_banner(run, cap)
    print(f"command: spore_xl.py {' '.join(sys.argv[1:])}   (PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', '<unset>')})")
    print(f"[run] {context}: {cpus} CPUs, RAM budget {budget:.1f} GB; workers per stage = min({'--workers ' + str(args.workers) if args.workers else 'CPUs'}, units, "
          f"RAM budget / per-task estimate), lowered and logged when the RAM budget cannot carry them")
    if args.workers:
        run.xl.setdefault("plan_workers", args.workers)
    else:
        run.xl.setdefault("plan_workers", max(1, min(cpus, int(budget // float(run.xl.get("memory_budget_gb", 24))))))
    import gate_protected
    import preflight
    stages = ORDER if (run.xl.get("source") or {}).get("kind") == "recipe" else [stage for stage in ORDER if stage not in ("intake_list", "intake")]
    for stage in stages:
        if stage == "plan" and gate_protected.gate_enabled(run.cfg):
            gate_protected.run_expression_gate(run)
            run._apply_runtime_updates()   # plan, hvg and the workers read the gated list from here on
        if stage in SINGLE_STAGES:
            SINGLE_STAGES[stage](run)
            continue
        units = _n_units(run, stage)
        workers, note = run_command.auto_workers(preflight.estimate_stage_ram_gb(run, stage), units, cpus, budget, args.workers)
        print(f"[{stage}] workers: {note}")
        run_stage(run, stage, workers, budget)
    import verify
    print("[run] verification (reports/VERIFICATION.md)")
    results = verify.verify_run(run, write_report=True)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"[run] VERIFICATION FAILED: {len(failed)} of {len(results)} checks (see reports/VERIFICATION*.md)")
        return 1
    print(f"[run] DONE: {len(results)} verification checks PASS ({context})")
    return 0


def main(argv=None):
    """Every invocation copies its console output to <run_dir>/logs/spore_xl_<stage-or-all>_<timestamp>.log (console_log.py)."""
    from console_log import ConsoleLog
    argv = sys.argv[1:] if argv is None else argv
    console = ConsoleLog()
    try:
        return _main(argv, console)
    except BaseException:
        import traceback
        console.record(traceback.format_exc())  # Python prints the traceback itself; this puts it in the log file too
        raise
    finally:
        if console.path:
            print(f"console output written to {console.path}")
        console.close()


def _main(argv, console):
    if not argv or "-h" in argv or "--help" in argv or argv[0] in ("measure", "convert"):
        show_banner()  # commands without a run (help, measure, convert): banner without run name
    if argv and argv[0] == "measure":  # header-only; needs no run config
        import measure
        if "--out" in argv and argv.index("--out") + 1 < len(argv):
            console.attach(argv[argv.index("--out") + 1], "measure")
        return measure.main(argv[1:])
    if argv and argv[0] == "convert":  # intake adapters (P7): any supported raw format + a recipe -> a SPORE-ready sparse .h5ad
        from src.intake import intake_adapters
        return intake_adapters.main(argv[1:])
    if argv and argv[0] == "submit":  # withdrawn from v4.0
        print("`spore_xl.py submit` was withdrawn from SPORE-XL v4.0: SPORE-XL bounds the memory of every task, so a cluster run is ONE job on one node:\n"
              "    sbatch slurm/spore_xl.sbatch my_run.yaml")
        return 2
    if argv and argv[0] == "branch":  # new run dir reusing the finished stages of another run
        import branch
        branch_parser = argparse.ArgumentParser(prog="spore_xl.py branch")
        branch_parser.add_argument("--from-config", required=True)
        branch_parser.add_argument("--config", required=True)
        branch_parser.add_argument("--from-stage", required=True)
        branch_args = branch_parser.parse_args(argv[1:])
        branch_run = Run(branch_args.config)
        console.attach(branch_run.logs_dir, "branch")
        show_banner(branch_run)
        branch.branch(branch_args.from_config, branch_args.config, branch_args.from_stage)
        return 0
    if argv and argv[0] == "cleanup":  # for SLURM arrays: run after a whole stage finished and verified
        import cleanup
        cleanup_parser = argparse.ArgumentParser(prog="spore_xl.py cleanup")
        cleanup_parser.add_argument("--config", required=True)
        cleanup_parser.add_argument("--after", required=True, choices=list(cleanup.PLAN))
        cleanup_args = cleanup_parser.parse_args(argv[1:])
        cleanup_run = Run(cleanup_args.config)
        console.attach(cleanup_run.logs_dir, f"cleanup_{cleanup_args.after}")
        show_banner(cleanup_run)
        removed = cleanup.after_stage(cleanup_run, cleanup_args.after)
        print(f"removed {removed} files (0 when clean-up is not approved in the config)")
        return 0
    parser = argparse.ArgumentParser(prog="spore_xl.py", description="SPORE-XL v4.0. One command: python spore_xl.py run --config my_run.yaml (intake, every stage, verification). "
                                     "Stage-level commands (plan map1 reduce1 map2 gather reduce2 hvg map3 metacell_prep metacell finalize report merge) are for developers; "
                                     "also: convert (raw formats to a sparse .h5ad), measure, branch, cleanup.")
    parser.add_argument("stage", metavar="command", choices=["run"] + ORDER + OPTIONAL + ["all-local"], help="run, or one stage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--workers", type=int, default=None, help="upper limit of the process pool (default: chosen per stage from the CPUs and the RAM budget)")
    parser.add_argument("--max-ram-gb", type=float, default=None, help="RAM budget (default: 90 percent of the SLURM job's memory, else 80 percent of this machine)")
    args = parser.parse_args(argv)
    if args.stage in ("run", "all-local"):
        return _run_command(args, console)
    workers = args.workers or 1
    run = Run(args.config)
    if workers > 1:  # P10: the serial plan stage shards the input with a pool as well
        run.xl.setdefault("plan_workers", workers)
    console.attach(run.logs_dir, args.stage + (f"_{args.index}" if args.index is not None else ""))
    pooled = args.stage == "plan" or (args.local and (args.stage in SHARD_STAGES or args.stage in BLOCK_STAGES))  # plan shards with a pool (P10)
    show_banner(run, max(workers, 1) if pooled else 1)
    print(f"command: spore_xl.py {' '.join(sys.argv[1:])}   (PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', '<unset>')})")
    stage = args.stage
    if stage in SINGLE_STAGES:
        SINGLE_STAGES[stage](run)
    elif args.local:
        run_stage(run, stage, workers, args.max_ram_gb)
    elif args.index is None:
        parser.error(f"{stage} needs --index N or --local")
    else:
        (SHARD_STAGES.get(stage) or BLOCK_STAGES[stage])(run, args.index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
