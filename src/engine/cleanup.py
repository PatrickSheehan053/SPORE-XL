"""Opt-in clean-up of VERIFIED intermediate files (keeps peak disk to about 2 to 2.5 copies of the data).

DEFAULT OFF. Active only when the run config holds, under xl.cleanup, BOTH
    approved_by: Patrick
    approved_on: <date>
(Patrick's standing rule: nothing is deleted without his explicit approval; that config line is his per-run approval.) A cleanup
section with anything else is a configuration error and stops the run before anything is touched.

When a stage has finished and its units verify (marker + sha256 of every file), the files of its PREDECESSOR that no later stage
needs are removed:
    after map1      the planned/intake source shards            (_xl_work/source_shards/shards/*.h5ad, _xl_work/intake/*/shard.h5ad)
    after map2      the QC shards                               (_xl_work/map1/shard_*/qc.h5ad)
    after gather    the bucket files                            (_xl_work/map2/shard_*/block_*.h5ad)
    after finalize  the gather blocks, ONLY with xl.cleanup.include_blocks: true (blocks are needed to re-run hvg onward, to
                    branch, and by extract)
Only files that a finished unit recorded in its own .done marker are removed, only below <run>/_xl_work, never raw input, never a
final output, nothing outside the run directory. Every removal is appended to reports/cleanup_log.tsv (time, stage, path, size,
sha256 from the marker). verify and the unit checks accept files listed in that log as removed.
"""
import json
import os
import time
from pathlib import Path

from xl_common import verify_marker

PLAN = {  # stage that just finished -> (label, glob patterns relative to <run>/_xl_work)
    "map1": ("source_shards", ["source_shards/shards/*.h5ad", "intake/shard_*/shard.h5ad"]),
    "map2": ("qc_shards", ["map1/shard_*/qc.h5ad"]),
    "gather": ("buckets", ["map2/shard_*/block_*.h5ad"]),
    "finalize": ("blocks", ["gather/block_*/block.h5ad"]),
}


def approval(run):
    """None when clean-up is not requested; the approval record when it is; ValueError for a half-filled section."""
    section = (run.xl or {}).get("cleanup")
    if section is None:
        return None
    approved_by, approved_on = section.get("approved_by"), section.get("approved_on")
    if approved_by != "Patrick" or not approved_on:
        raise ValueError("xl.cleanup is present but not approved: it needs approved_by: Patrick AND approved_on: <date>. "
                         "Nothing was removed. Remove the section to keep every intermediate file.")
    return {"approved_by": approved_by, "approved_on": str(approved_on), "include_blocks": bool(section.get("include_blocks", False))}


def removed_set(run_dir):
    log = Path(run_dir) / "reports" / "cleanup_log.tsv"
    if not log.exists():
        return set()
    lines = log.read_text(encoding="utf-8").split("\n")[1:]
    return {line.split("\t")[2] for line in lines if line.count("\t") >= 3}


def _unit_files_recorded(work_dir):
    """absolute path -> sha256 for every file recorded in a finished unit marker under <run>/_xl_work."""
    recorded = {}
    for marker in Path(work_dir).glob("*/*.done"):
        unit_dir = marker.with_name(marker.stem)
        info = json.loads(marker.read_text(encoding="utf-8"))
        for relative, expected in info.get("files", {}).items():
            recorded[str((unit_dir / relative).resolve())] = expected["sha256"]
    return recorded


def verify_stage(run, stage):
    """Every unit marker of the stage exists and its files match their checksums."""
    roots = [run.stats_dir / stage, run.work_dir / stage]
    n_checked = 0
    for base in roots:
        if not base.exists():
            continue
        for marker in sorted(base.glob("*.done")):
            verify_marker(marker, marker.with_name(marker.stem), run.run_dir)
            n_checked += 1
    if n_checked == 0:
        raise RuntimeError(f"cleanup: no finished units found for {stage}")
    return n_checked


def after_stage(run, stage):
    """Called after a whole stage finished. Returns the number of files removed (0 when inactive or nothing to remove)."""
    record = approval(run)
    if record is None or stage not in PLAN:
        return 0
    label, patterns = PLAN[stage]
    if label == "blocks" and not record["include_blocks"]:
        return 0
    verify_stage(run, stage)
    work_root = run.work_dir.resolve()
    recorded = _unit_files_recorded(run.work_dir)
    already = removed_set(run.run_dir)
    log_path = run.reports_dir / "cleanup_log.tsv"
    if not log_path.exists():
        log_path.write_text("time\tstage\tpath\tsize_bytes\tsha256\tapproved_by\tapproved_on\n", encoding="utf-8")
    removed = 0
    for pattern in patterns:
        for path in sorted(run.work_dir.glob(pattern)):
            resolved = path.resolve()
            if work_root not in resolved.parents or path.suffix != ".h5ad":
                raise RuntimeError(f"cleanup refuses {path}: not an .h5ad file below {work_root}")
            key = str(resolved)
            if key in already or not path.exists():
                continue
            if key not in recorded and label != "source_shards":
                continue  # not a file that a finished unit hashed: never touched
            size = path.stat().st_size
            sha = recorded.get(key, "")
            os.remove(path)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{label}\t{key}\t{size}\t{sha}\t{record['approved_by']}\t{record['approved_on']}\n")
                handle.flush()
                os.fsync(handle.fileno())
            removed += 1
    run.logger.info(f"[cleanup after {stage}] removed {removed} {label} files (approved by {record['approved_by']} on {record['approved_on']})")
    return removed
