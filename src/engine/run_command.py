"""Helpers of `spore_xl.py run` (the one command of a SPORE-XL v4.0 run): where it runs, how many workers each stage gets, which input paths must exist.

  run_context()        "interactive" or "SLURM job <id>"
  available_cpus()     SLURM_CPUS_PER_TASK, else the CPUs this process may use
  ram_budget_gb()      90 percent of SLURM_MEM_PER_NODE (or SLURM_MEM_PER_CPU x CPUs) in a SLURM job, else 80 percent of this machine; --max-ram-gb overrides
  auto_workers()       workers = min(CPUs, units of the stage, floor(RAM budget / per-task estimate)); never below 1; --workers is an upper bound
  input_problems()     every input path the config names that does not exist (checked before any work starts)
"""
import os
from pathlib import Path


def run_context():
    job = os.environ.get("SLURM_JOB_ID", "").strip()
    return f"SLURM job {job}" if job else "interactive"


def available_cpus():
    from src.report import banner
    return banner.available_cpus()


def ram_budget_gb(override=None):
    if override is not None:
        return float(override)
    per_node = os.environ.get("SLURM_MEM_PER_NODE", "").strip()
    if per_node.isdigit():
        return 0.9 * int(per_node) / 1024.0
    per_cpu, cpus = os.environ.get("SLURM_MEM_PER_CPU", "").strip(), os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if per_cpu.isdigit() and cpus.isdigit():
        return 0.9 * int(per_cpu) * int(cpus) / 1024.0
    import psutil
    return 0.8 * psutil.virtual_memory().total / 1e9


def auto_workers(estimate_gb, n_units, cpus, budget_gb, requested=None):
    """(workers, note). `requested` (--workers) is an upper bound, never a demand: when the RAM budget cannot carry it the count is LOWERED and the note says so."""
    cap = int(requested) if requested else int(cpus)
    by_ram = max(1, int(budget_gb // max(estimate_gb, 1e-9)))
    workers = max(1, min(cap, int(n_units), by_ram))
    note = f"{workers} worker(s): min(CPUs/requested {cap}, units {int(n_units)}, RAM budget {budget_gb:.1f} GB / {estimate_gb:.2f} GB per task = {by_ram})"
    if by_ram < min(cap, int(n_units)):
        note += f"; LOWERED from {min(cap, int(n_units))} by the RAM budget"
    if estimate_gb > budget_gb:
        note += f"; WARNING: even one task is estimated above the budget ({estimate_gb:.2f} GB > {budget_gb:.1f} GB), running with 1 worker"
    return workers, note


def input_problems(cfg, repo_root):
    """[(what, path)] for every input path of the config (a dict, read straight from the yaml, so nothing has been created yet) that does not exist:
    source file / tar / recipe of xl.source and the protected-gene list."""
    repo_root = Path(repo_root)
    source = (cfg.get("xl", {}) or {}).get("source", {}) or {}
    wanted = [(f"xl.source.{key}", source[key]) for key in ("h5ad", "tar", "recipe") if source.get(key)]
    protected = (cfg.get("phase8_hvg", {}) or {}).get("protected_gene_list")
    if protected:
        wanted.append(("phase8_hvg.protected_gene_list", protected))
    missing = []
    for what, value in wanted:
        path = Path(str(value))
        if not (path.exists() or (not path.is_absolute() and (repo_root / path).exists())):
            missing.append((what, str(value)))
    return missing
