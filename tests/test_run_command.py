"""`spore_xl.py run --config my.yaml`: the one command of a v4.0 run (intake when the source is a recipe, every stage up to the report, then the verification).
 1 helpers: context ("interactive" / "SLURM job <id>"), RAM budget (90 percent of SLURM_MEM_PER_NODE, --max-ram-gb overrides), workers = min(CPUs or --workers, units,
   RAM budget / per-task estimate) and never below 1 (the pre-flight LOWERS, it does not refuse)
 2 a run on the 60k synthetic: exit 0, GRAPH-READY, reports/VERIFICATION.md says PASS, one banner that says "interactive"; the hidden alias `all-local` resumes the
   same run (finished units skipped) and, in a SLURM environment, the banner and the log say "SLURM job <id>" with the CPUs and the RAM budget of the job
 3 --max-ram-gb 1 with --workers 3: the worker count is LOWERED (and logged), the run still finishes
 4 a missing input path stops with exit 2 before any stage, naming the path
 5 a verification FAIL makes `run` exit 1; every `run` writes its own VERIFICATION report (VERIFICATION.md, VERIFICATION_2.md, ...)
 6 phase8_hvg.protected_expression_gate: true (CD4-like recipe source): the gate runs inside `run` after the intake, the gated list is written into the run folder
   (manifest/protected_genes_gated.json with manifest/protected_gate_report.tsv beside it; never into the repository or over the input list) and is the list that plan and hvg use
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; set XL_TEST_TMP=<scratch> ; set XL_TEST_DATA=<reference folder> ; python -B tests\\test_run_command.py"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import BASE_CONFIG, REPO, fresh_run_config, scratch_for, shard_rows_for  # noqa: E402
import make_flex_synthetic as flex  # noqa: E402

import yaml  # noqa: E402

import run_command  # noqa: E402
from xl_common import Run  # noqa: E402

SPORE = REPO / "spore_xl.py"
RECIPE = REPO / "src" / "intake" / "recipes" / "GSE314342_CD4T-Rest.yaml"


def run_cli(*arguments, environment=None):
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_") and key != "SPORE_NO_BANNER"}
    env.update(PYTHONHASHSEED="0", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    env.update(environment or {})
    return subprocess.run([sys.executable, "-B", str(SPORE), *arguments], capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")


def test_helpers():
    saved = {key: os.environ.pop(key, None) for key in ("SLURM_JOB_ID", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPUS_PER_TASK")}
    try:
        assert run_command.run_context() == "interactive"
        os.environ["SLURM_JOB_ID"] = "44987070"
        assert run_command.run_context() == "SLURM job 44987070"
        os.environ["SLURM_MEM_PER_NODE"] = str(400 * 1024)
        assert abs(run_command.ram_budget_gb() - 360.0) < 1e-6, "90 percent of the job's memory"
        assert run_command.ram_budget_gb(override=50) == 50.0
        del os.environ["SLURM_MEM_PER_NODE"]
        os.environ.update(SLURM_MEM_PER_CPU="16384", SLURM_CPUS_PER_TASK="8")
        assert abs(run_command.ram_budget_gb() - 0.9 * 16 * 8) < 1e-6
        os.environ["SLURM_CPUS_PER_TASK"] = "12"
        assert run_command.available_cpus() == 12
    finally:
        for key, value in saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value
    workers, note = run_command.auto_workers(estimate_gb=11.5, n_units=94, cpus=16, budget_gb=360.0)
    assert workers == 16 and "LOWERED" not in note, note                                  # CPUs are the limit
    workers, note = run_command.auto_workers(estimate_gb=20.23, n_units=94, cpus=64, budget_gb=230.0)
    assert workers == 11 and "LOWERED" in note, note                                      # the 22 Oct CD4 case: 230 / 20.23
    workers, note = run_command.auto_workers(estimate_gb=0.8, n_units=3, cpus=16, budget_gb=360.0)
    assert workers == 3, note                                                             # never more workers than units
    workers, note = run_command.auto_workers(estimate_gb=5.0, n_units=100, cpus=16, budget_gb=360.0, requested=4)
    assert workers == 4, note                                                             # --workers is an upper bound
    workers, note = run_command.auto_workers(estimate_gb=50.0, n_units=100, cpus=16, budget_gb=20.0, requested=8)
    assert workers == 1 and "WARNING" in note, note                                       # lowered to 1, never a refusal
    print("    helpers: context, RAM budget (90 percent of the job's memory), workers = min(CPUs, units, RAM / task), at least 1")


def test_run_alias_slurm_banner_and_lowering():
    config = fresh_run_config("run_a", shard_rows=shard_rows_for(3))
    first = run_cli("run", "--config", str(config), "--workers", "3")
    assert first.returncode == 0, first.stdout[-2500:] + first.stderr[-2500:]
    assert "GRAPH-READY" in first.stdout and first.stdout.count("spore fact:") == 1 and "interactive" in first.stdout, first.stdout[:1500]
    run = Run(config)
    report = (run.reports_dir / "VERIFICATION.md").read_text(encoding="utf-8")
    assert report.startswith("# VERIFICATION PASS") and "FAIL" not in report, report[:300]
    assert "[run]" in first.stdout and "workers" in first.stdout
    again = run_cli("all-local", "--config", str(config), environment={"SLURM_JOB_ID": "4242", "SLURM_CPUS_PER_TASK": "2", "SLURM_MEM_PER_NODE": "2048"})
    assert again.returncode == 0, again.stdout[-2500:] + again.stderr[-2500:]
    assert "SLURM job 4242" in again.stdout and "2 CPUs" in again.stdout and "RAM budget 1.8 GB" in again.stdout, again.stdout[:2500]
    assert "already done, skipped" in again.stdout + again.stderr
    assert "SLURM job 4242" in next(run.logs_dir.glob("spore_xl_run_*.log")).read_text(encoding="utf-8") or any(
        "SLURM job 4242" in path.read_text(encoding="utf-8") for path in run.logs_dir.glob("spore_xl_*.log")), "the log file must say where the run ran"
    config_b = fresh_run_config("run_b", shard_rows=shard_rows_for(3))
    lowered = run_cli("run", "--config", str(config_b), "--workers", "3", "--max-ram-gb", "1")
    assert lowered.returncode == 0, lowered.stdout[-2500:] + lowered.stderr[-2500:]
    assert "LOWERED" in lowered.stdout + lowered.stderr and "GRAPH-READY" in lowered.stdout, "the pre-flight lowers the workers and logs it; it does not refuse"
    # a verification FAIL makes `run` exit 1 (in process: the finished run is resumed, every stage is skipped, and one check of the verification is made to fail;
    # tampering with a finished file instead is caught EARLIER by the checksum guard of the resume, which also stops the command with a non-zero exit)
    import argparse
    import cli
    import console_log
    import verify
    real = verify.verify_run

    def planted_failure(run_, write_report=True):
        return real(run_, write_report=write_report) + [("planted failure (test)", False, "this check always fails")]
    verify.verify_run = planted_failure
    console = console_log.ConsoleLog()
    try:
        code = cli._run_command(argparse.Namespace(config=str(config), workers=None, max_ram_gb=None), console)
    finally:
        console.close()
        verify.verify_run = real
    assert code == 1, code
    assert (run.reports_dir / "VERIFICATION_2.md").exists() or (run.reports_dir / "VERIFICATION_3.md").exists(), "each `run` writes its own VERIFICATION report"
    print("    run: exit 0 + VERIFICATION.md PASS + one banner; alias resumes; SLURM banner/log; workers lowered not refused; a verification FAIL exits 1")


def test_missing_input_stops_before_any_stage():
    work = scratch_for("run_missing")
    config = fresh_run_config("run_missing", shard_rows=shard_rows_for(3))
    broken = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    broken["xl"]["source"]["h5ad"] = str(work / "no_such_source.h5ad").replace("\\", "/")
    path = work / "broken.yaml"
    path.write_text(yaml.safe_dump(broken), encoding="utf-8")
    result = run_cli("run", "--config", str(path))
    assert result.returncode == 2 and "no_such_source.h5ad" in result.stdout + result.stderr and "does not exist" in result.stdout + result.stderr, result.stdout[-800:]
    run = Run(path)
    assert not (run.manifest_dir / "plan.done").exists() and not list(run.manifest_dir.glob("*.tsv")), "no stage may have started"
    print("    a missing input path stops `run` with exit 2 before any stage, naming the path")


def test_protected_expression_gate_inside_run():
    work = scratch_for("run_gate")
    members = [(flex.member_name(9393800 + i, 1 + i % 2, 9 + i, 1 + i, "Rest"), 2500 + 100 * i) for i in range(3)]
    flex.build_tar(work / "h5", work / "x.tar", members, gene_orders={1: list(reversed(range(flex.N_GENES)))})
    gene_names = flex.library()[1]
    protected_in = work / "protected_in.json"
    original = {"metadata": {"version": "1.0", "name": "synthetic"}, "protected_genes": {
        "identity": {"tier": 1, "genes": list(gene_names[:3])}, "tahoe_menu": {"tier": 3, "genes": list(gene_names[3:6]) + ["NOT_IN_MATRIX"]}}}
    protected_in.write_text(json.dumps(original), encoding="utf-8")
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["paths"].update({"project_root": str(work / "run").replace("\\", "/"), "raw_h5ad": ""})
    cfg["phase8_hvg"].update({"protected_gene_list": str(protected_in).replace("\\", "/"), "protected_expression_gate": True, "n_top_genes": 60})
    cfg["phase5_escaper_filtering"]["min_cells_per_perturbation"] = 10
    cfg["phase2_cell_triage"].update({"min_genes_per_cell": 10, "min_counts_per_cell": 20})
    cfg["phase7_splits"].update({"test_n": 6, "val_n": 3})
    cfg["xl"] = {"source": {"kind": "recipe", "recipe": str(RECIPE).replace("\\", "/"), "tar": str(work / "x.tar").replace("\\", "/")},
                 "work_dir": "_xl_work", "n_blocks": 4, "control_block_rows": 1000, "bucket_compression": "lzf", "memory_budget_gb": 24, "merge_mode": "memory"}
    config = work / "run.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = run_cli("run", "--config", str(config), "--workers", "2")
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    run = Run(config)
    gated = run.manifest_dir / "protected_genes_gated.json"
    assert gated.exists() and (run.manifest_dir / "protected_gate_report.tsv").exists()
    new = json.loads(gated.read_text(encoding="utf-8"))
    assert "NOT_IN_MATRIX" not in new["protected_genes"]["tahoe_menu"]["genes"] and "expression_gate" in new["metadata"], "a menu gene that is not expressed is removed"
    assert json.loads(protected_in.read_text(encoding="utf-8")) == original, "the input list is never touched"
    assert run.cfg["phase8_hvg"]["protected_gene_list"] == str(gated), "plan, hvg and every worker must read the gated list"
    assert (run.reports_dir / "VERIFICATION.md").read_text(encoding="utf-8").startswith("# VERIFICATION PASS")
    again = run_cli("run", "--config", str(config), "--workers", "2")
    assert again.returncode == 0 and "protected_genes_gated" in again.stdout + again.stderr and "already done" in again.stdout + again.stderr
    print("    gate: runs inside `run` after the intake, writes the gated list into the run folder, the input list is untouched, plan/hvg read the gated list; resume skips it")


if __name__ == "__main__":
    test_helpers()
    test_missing_input_stops_before_any_stage()
    test_run_alias_slurm_banner_and_lowering()
    test_protected_expression_gate_inside_run()
    print("PASS test_run_command")
