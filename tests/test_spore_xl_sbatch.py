"""slurm/spore_xl.sbatch (`sbatch slurm/spore_xl.sbatch my_run.yaml`: the one cluster command; it calls `spore_xl.py run`).
 1 static: bash -n, LF, #SBATCH header with SET BY PATRICK, `spore_xl.py run` is the only call, no submit / sbatch / deletion commands, no placeholders
 2 FUNCTIONAL (POSIX bash needed) on the 60k synthetic with a fake venv: no argument and a missing config stop with exit 2 before any work; a RELATIVE config path is taken
   relative to the folder the job was submitted from (SLURM_SUBMIT_DIR); the run finishes with exit 0, the banner and the [run] line say "SLURM job <id>", the job's CPUs
   and 90 percent of the job's memory (SLURM_CPUS_PER_TASK, SLURM_MEM_PER_NODE), and VERIFICATION.md says PASS; a second submission resumes (finished units skipped)
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; set XL_TEST_TMP=<scratch> ; set XL_TEST_DATA=<reference folder> ; python -B tests\\test_spore_xl_sbatch.py"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import REPO, fresh_run_config, scratch_for, shard_rows_for  # noqa: E402

from xl_common import Run  # noqa: E402

SCRIPT = REPO / "slurm" / "spore_xl.sbatch"


def posix_bash():
    bash = shutil.which("bash")
    if not bash:
        return None
    probe = subprocess.run([bash, "-c", "uname -o"], capture_output=True, text=True)
    return bash if probe.returncode == 0 and probe.stdout.strip() in ("Msys", "Cygwin", "GNU/Linux") else None


def test_static_checks():
    raw = SCRIPT.read_bytes()
    text = raw.decode("utf-8")
    assert b"\r" not in raw and text.startswith("#!/bin/bash") and "set -euo pipefail" in text and "SET BY PATRICK" in text
    for key in ("--job-name", "--partition=cpuq", "--cpus-per-task=32", "--mem=400G", "--time"):
        assert f"#SBATCH {key}" in text, key
    body = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert not [line for line in body if re.search(r"<[A-Z_0-9]+>", line)] and not any(re.search(r"^\s*(rm|sbatch|srun)\b|spore\.py submit|--dependency", line) for line in body)   # messages may say "sbatch"; no command may
    calls = [line for line in body if "spore_xl.py" in line and "python" in line]
    assert len(calls) == 1 and "spore_xl.py run --config" in calls[0] and "--max-ram-gb" in calls[0], calls
    if shutil.which("bash"):
        result = subprocess.run([shutil.which("bash"), "-n", str(SCRIPT)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    print("    spore_xl.sbatch: syntax, LF, header (32 CPUs, 400 GB proposed), one call `spore_xl.py run`, no placeholders, no submit mode, no deletions")


def run_script(arguments, environment_extra, workdir):
    environment = {"PATH": os.environ["PATH"], "PYTHONHASHSEED": "0", "PYTHONUTF8": "1", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    environment.update(environment_extra)
    return subprocess.run([posix_bash(), str(SCRIPT), *arguments], capture_output=True, text=True, env=environment, encoding="utf-8", errors="replace", cwd=str(workdir))


def test_functional_arguments_slurm_environment_and_resume():
    if not posix_bash():
        print("    SKIP functional part: no POSIX bash on this machine")
        return
    posix = lambda path: str(path).replace("\\", "/")  # noqa: E731
    work = scratch_for("spore_xl_sbatch")
    fake_venv = work / "venv"
    (fake_venv / "bin").mkdir(parents=True)
    (fake_venv / "bin" / "activate").write_text(f'python() {{ "{sys.executable.replace(chr(92), "/")}" "$@"; }}\n', encoding="utf-8", newline="\n")
    base = {"VENV": posix(fake_venv), "REPO": posix(REPO), "SLURM_SUBMIT_DIR": posix(work)}
    none = run_script([], base, work)
    assert none.returncode == 2 and "give the config file" in none.stdout, none.stdout
    missing = run_script(["no_such_config.yaml"], base, work)
    assert missing.returncode == 2 and "does not exist" in missing.stdout and posix(work) in missing.stdout, "a relative config is taken from the submit folder: " + missing.stdout
    no_repo = run_script([posix(work / "x.yaml")], {"VENV": posix(fake_venv), "SLURM_SUBMIT_DIR": posix(work)}, work)
    assert no_repo.returncode == 2 and "does not exist" in no_repo.stdout, "without REPO and without spore_xl.py in the submit folder the cluster default applies (absent here)"
    config = fresh_run_config("sbatch_run", shard_rows=shard_rows_for(3))
    relative = Path(config).name
    job = {"SLURM_JOB_ID": "777", "SLURM_CPUS_PER_TASK": "2", "SLURM_MEM_PER_NODE": str(2 * 1024)}
    done = run_script([relative], {**base, **job, "SLURM_SUBMIT_DIR": posix(Path(config).parent)}, work)
    assert done.returncode == 0, done.stdout[-2500:] + done.stderr[-2500:]
    assert "SLURM job 777" in done.stdout and "2 CPUs" in done.stdout and "RAM budget 1.8 GB" in done.stdout and "GRAPH-READY" in done.stdout, done.stdout[:2500]
    assert done.stdout.count("spore fact:") == 1
    run = Run(config)
    assert (run.reports_dir / "VERIFICATION.md").read_text(encoding="utf-8").startswith("# VERIFICATION PASS")
    again = run_script([posix(config)], {**base, **job}, work)
    assert again.returncode == 0 and "already done, skipped" in again.stdout + again.stderr, f"exit {again.returncode}\n" + again.stdout[-1200:] + again.stderr[-1200:]
    print("    functional: no argument / missing config -> exit 2; a relative config is read from the submit folder; the run finishes (exit 0) with `SLURM job 777`, 2 CPUs, "
          "RAM budget 1.8 GB (90 percent of the job's memory) and VERIFICATION PASS; a second submission resumes")


if __name__ == "__main__":
    test_static_checks()
    test_functional_arguments_slurm_environment_and_resume()
    print("PASS test_spore_xl_sbatch")
