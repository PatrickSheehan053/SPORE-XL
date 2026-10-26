"""Pre-flight checks (blueprint component 10 and the --local disk rule): refuse to start what cannot finish.

disk   projected working + output bytes (recalibrated on the finished RPE1 run: 39.8 bytes per source stored value including
       merged outputs, was 20 from the synthetic 60k run) against free space on the run drive, keeping a reserve
memory rows x nonzeros per row x 12 bytes x working-copy factor 3 against xl.memory_budget_gb (blueprint shard rule);
       the same estimate guards a block in gather
Nothing is deleted or modified; the message says what to change.
"""
import os
import shutil

import shard_planner

RESERVE_GB = 5.0
# Per-stage constants from tools/memory_calibration.py on three finished runs (CD4 Rest, K562 compat, K562 production; HPC1, 22 Oct): the worst measured (peak - 0.7 GB) / input size of any unit,
# times a safety factor of 1.5, the maximum over the runs. Before: 4.0 for every stage (below the measured peak for K562 map1 and map3 and for gather on CD4 and K562).
STAGE_EXPANSION = {"map1": 6.16,      # K562 production (CD4 5.40, K562 compat 6.13)
                   "map2": 2.17,      # CD4 (K562 1.68 / 1.71)
                   "gather": 7.83,    # CD4 (K562 7.03)
                   "map3": 6.56}      # K562 compat (K562 production 5.12, CD4 4.26)
GATHER_EXPANSION = STAGE_EXPANSION["gather"]
METACELL_DENSE_FACTOR = 12    # CD4 Rest, 22 Oct: control blocks of about 110,000 cells x 5,000 genes (2.2 GB dense float32) peaked at 16.5-16.7 GB = about 8 x, times a safety factor of 1.5
INTAKE_BYTES_PER_VALUE = 22.2  # CD4 Rest, 22 Oct: worst measured 14.78 bytes per stored value above the 0.7 GB base, x 1.5 (members of 320-542 M values peaked at 4.7-8.0 GB); was 36, then 20
PLAN_TIME_SURVIVAL = 0.9   # unknown at plan time; measured 0.827 (RPE1), 0.897 (K562 per Michele's SPORE log), 0.65 (synthetic)


def projected_disk_gb(run, nnz, n_features):
    """(work GB, output GB) of the whole run without clean-up, from the model calibrated on the finished RPE1 run (measure.BYTES_PER_NNZ,
    39.8 bytes per source value in total), each with the 20 percent margin. Merged split files are counted unless
    xl.merge_split_outputs is never. Work files go to xl.work_dir, outputs to the run directory (different drives are allowed)."""
    import measure
    panel = int(run.cfg.get("phase8_hvg", {}).get("n_top_genes", 5000))
    fraction = measure.panel_fraction_estimate(panel, n_features) if n_features else 0.5
    merged = run.xl.get("merge_split_outputs", "auto") != "never"
    work, outputs = measure.disk_projection(nnz, PLAN_TIME_SURVIVAL, fraction, merged)
    return sum(work.values()) * 1.01 * measure.DISK_MARGIN, sum(outputs.values()) * 1.01 * measure.DISK_MARGIN


def _check_disk(run, nnz, n_features):
    work_gb, output_gb = projected_disk_gb(run, nnz, n_features)
    run_free = shutil.disk_usage(run.run_dir).free / 1e9
    work_free = shutil.disk_usage(run.work_dir).free / 1e9
    same_drive = os.stat(run.run_dir).st_dev == os.stat(run.work_dir).st_dev
    if same_drive:
        projected_gb, free_gb = work_gb + output_gb, run_free
        run.logger.info(f"[plan] disk pre-flight: projected {projected_gb:.1f} GB (incl. 20% margin), free {free_gb:.1f} GB")
        needs = [("the run drive", projected_gb, free_gb)]
    else:
        projected_gb, free_gb = work_gb + output_gb, min(run_free, work_free)
        run.logger.info(f"[plan] disk pre-flight: work files {work_gb:.1f} GB on {run.work_dir} (free {work_free:.1f} GB); outputs "
                        f"{output_gb:.1f} GB in {run.run_dir} (free {run_free:.1f} GB); both incl. 20% margin")
        needs = [("the work drive", work_gb, work_free), ("the run drive", output_gb, run_free)]
    for label, need, free in needs:
        if need + RESERVE_GB > free:
            raise OSError(f"disk pre-flight failed: this run needs about {need:.1f} GB (+{RESERVE_GB:.0f} GB reserve) on {label} but "
                          f"{free:.1f} GB are free. Free space or choose another directory; nothing was written.")
    return projected_gb, free_gb


def check_disk(run, h5ad_path):
    facts = shard_planner.measure_h5ad(h5ad_path)
    return _check_disk(run, facts["nnz"], facts["n_vars"])


def check_disk_for_nnz(run, nnz, n_features=None):
    """Same rule for sources that are not one .h5ad (the intake shards of a recipe run)."""
    return _check_disk(run, nnz, n_features)


def estimate_ram_gb(n_cells, nnz_per_cell, working_copy_factor=3, bytes_per_value=12):
    return n_cells * nnz_per_cell * bytes_per_value * working_copy_factor / 1e9


def check_task_memory(run, estimated_gb, what):
    budget = float(run.xl.get("memory_budget_gb", 24))
    if estimated_gb > budget:
        raise MemoryError(f"memory pre-flight failed for {what}: estimated {estimated_gb:.1f} GB > xl.memory_budget_gb {budget:.1f} GB. "
                          f"Use smaller shards (xl.source.shard_rows) or more blocks (xl.n_blocks).")
    return estimated_gb


def check_shard_memory(run, largest_cells, nnz_per_cell):
    estimate = estimate_ram_gb(largest_cells, nnz_per_cell)
    check_task_memory(run, estimate, f"the largest shard ({largest_cells:,} cells)")
    run.logger.info(f"[plan] memory pre-flight: largest shard {largest_cells:,} cells -> about {estimate:.2f} GB (budget "
                    f"{run.xl.get('memory_budget_gb', 24)} GB)")
    return estimate


# ---------------------------------------------------------------------------------------------- per-stage RAM and --workers
def _size_gb(path):
    from pathlib import Path
    path = Path(path)
    return path.stat().st_size / 1e9 if path.exists() else 0.0


def _h5ad_shape(path):
    """(cells, genes) of an .h5ad from its header (h5py; no matrix is read)."""
    import h5py
    with h5py.File(path, "r") as handle:
        node = handle["X"]
        shape = node.attrs["shape"] if isinstance(node, h5py.Group) else node.shape
        return int(shape[0]), int(shape[1])


def estimate_stage_ram_gb(run, stage):
    """Peak RAM of ONE task of this stage, from the on-disk size of its inputs (calibrated: about 1.2 x file size per working
    copy x 3-4 copies; measured task peaks were 0.4 to 0.85 GB for 20k-cell shards of 53 MB). Returns the max over the stage's units."""
    base = 0.7
    expansion = 4.0   # stages without a calibrated constant
    work = run.work_dir
    if stage == "intake":
        members = run.manifest_dir / "intake_members.tsv"
        if not members.exists():
            return base
        import pandas as pd
        table = pd.read_csv(members, sep="\t")
        if "nnz" in table.columns:
            return base + float(table["nnz"].max()) * INTAKE_BYTES_PER_VALUE / 1e9
        return base + float(table["size"].max()) / 1e9 * 8
    if stage == "map1":
        shards = run.shards()
        return base + max((_size_gb(run.run_dir / name) for name in shards["file"]), default=0.0) * STAGE_EXPANSION["map1"]
    if stage == "map2":
        return base + max((_size_gb(path) for path in (work / "map1").glob("shard_*/qc.h5ad")), default=0.0) * STAGE_EXPANSION["map2"]
    if stage == "gather":
        sizes = {}
        for path in (work / "map2").glob("shard_*/block_*.h5ad"):
            sizes[path.name] = sizes.get(path.name, 0.0) + _size_gb(path)
        return base + max(sizes.values(), default=0.0) * GATHER_EXPANSION
    if stage == "map3":
        return base + max((_size_gb(path) for path in (work / "gather").glob("block_*/block.h5ad")), default=0.0) * STAGE_EXPANSION["map3"]
    if stage == "metacell":
        sizes, cells, genes = {}, {}, 0
        for path in run.processed_dir.glob(f"{run.name}_*_hvgraw/part_*.h5ad"):
            sizes[path.name] = sizes.get(path.name, 0.0) + _size_gb(path)
            n_obs, n_vars = _h5ad_shape(path)
            cells[path.name] = cells.get(path.name, 0) + n_obs
            genes = max(genes, n_vars)
        # the file size of a block says little about the dense working set of its k-means / PCA: the CONTROL blocks (up to xl.control_block_rows cells) peaked at 16.5 GB
        # while the size rule gave 3.7 GB. The largest block (most cells) therefore sets the estimate: cells x panel genes x 4 bytes x METACELL_DENSE_FACTOR
        dense = max(cells.values(), default=0) * genes * 4 * METACELL_DENSE_FACTOR / 1e9
        return base + max(max(sizes.values(), default=0.0) * expansion, dense)
    return base


def check_workers(run, stage, workers, max_ram_gb=None):
    """Returns the number of workers this stage can run with: `workers` lowered, and logged, when workers x per-task peak RAM would exceed the budget
    (default: 80 percent of this machine's RAM). It never refuses: one task above the budget gives a warning and 1 worker."""
    if workers <= 1:
        return 1
    import psutil
    budget = float(max_ram_gb) if max_ram_gb is not None else 0.8 * psutil.virtual_memory().total / 1e9
    estimate = estimate_stage_ram_gb(run, stage)
    need = workers * estimate
    run.logger.info(f"[{stage}] workers pre-flight: {workers} x {estimate:.2f} GB = {need:.1f} GB (budget {budget:.1f} GB)")
    if need <= budget:
        return workers
    allowed = max(1, int(budget // estimate))
    note = f"; one task alone ({estimate:.2f} GB) is above the budget" if estimate > budget else ""
    run.logger.warning(f"[{stage}] workers LOWERED from {workers} to {allowed}: {workers} x {estimate:.2f} GB per task = {need:.1f} GB > budget {budget:.1f} GB{note}")
    return allowed
