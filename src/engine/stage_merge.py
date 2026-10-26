"""merge (optional, one job; P10): the merged split files (concat_on_disk) of a finished run, made on demand instead of inside finalize.
The per-block parts under processed/ stay the product; extract, HYPHAE readers and FUNGI tools read them directly."""
from xl_common import Run, register_external, require_done, task
from stage_finalize import _merge_outputs_ondisk, hybrid_lognorm_path, hybrid_part_path, lognorm_enabled, make_aliases
from stage_gather import total_blocks


def run_merge(run: Run):
    require_done(run, "stats", "finalize", "all")
    n_total = total_blocks(run)
    with task(run, "stats", "merge", "all") as out:
        if out is None:
            return
        hybrid_files = [hybrid_part_path(run, block) for block in range(n_total)]
        lognorm_files = [hybrid_lognorm_path(run, block) for block in range(n_total)] if lognorm_enabled(run) else []
        written = _merge_outputs_ondisk(run, n_total, hybrid_files, lognorm_files)
        register_external(out, written)
        make_aliases(run, n_total)
        run.logger.info(f"[merge] {len(written)} merged files written with concat_on_disk (rows in block order, each row carries xl_global_index)")
