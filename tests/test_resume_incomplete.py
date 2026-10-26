"""A run killed in the middle of a unit must resume: leftovers of the unit are MOVED (never deleted) to <work>/_incomplete/<stage>_<unit>_<time>/,
logged (console, log file, reports/incomplete_moves.tsv), the unit is recomputed, and the final outputs equal an uninterrupted run.
On the 60k synthetic (3 shards, 8 blocks + 4 control blocks):
  A. a real kill: `map3 --index 2` is started as a process and killed as soon as its first part file appears;
  B. simulated leftovers: a half-written part file + unit folder of map3 block 3, and half-written split JSON / global_obs_final of reduce2;
  C. nested work unit finished but its enclosing stats unit not (killed between the two markers) -> both recomputed;
then every remaining stage is resumed and the content digests of processed/ and splits/ plus the signatures equal a fresh uninterrupted run.
(.h5ad files embed HDF5 timestamps, so 'identical' means identical content digests, as in the other tests.)
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; set XL_TEST_TMP=<scratch> ; python -B tests\\test_resume_incomplete.py"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xl_test_support import fresh_run_config, shard_rows_for, file_digest  # noqa: E402

import pandas as pd  # noqa: E402

import compare_v4 as compare  # noqa: E402
import cli as spore_xl  # noqa: E402
from xl_common import Run, task  # noqa: E402

PY = sys.executable
CLI = str(HERE.parent / "spore_xl.py")
STAGES = [stage for stage in spore_xl.ORDER if stage not in ("intake_list", "intake")]


def final_digest(run):
    digest = {}
    for folder in (run.processed_dir, run.splits_dir):
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file():
                digest[str(path.relative_to(run.run_dir))] = file_digest(path)
    return digest


def part_files(run, block):
    return sorted(run.processed_dir.glob(f"*/part_{block:04d}.h5ad"))


def test_nested_unit_finished_but_parent_not(run):
    attempts = []

    def body(fail):
        with task(run, "stats", "fakestage", "u0") as outer:
            if outer is None:
                return "skipped"
            with task(run, "work", "fakestage", "u0", log=False) as inner:
                attempts.append(inner is not None)
                (inner / "data.txt").write_text("inner", encoding="utf-8")
            if fail:
                raise KeyboardInterrupt("killed between the two markers")
            (outer / "result.txt").write_text("outer", encoding="utf-8")
        return "done"

    try:
        body(True)
    except KeyboardInterrupt:
        pass
    assert run.marker("work", "fakestage", "u0").exists() and not run.marker("stats", "fakestage", "u0").exists()
    assert body(False) == "done" and attempts == [True, True]          # the inner unit ran again with its parent
    assert run.marker("stats", "fakestage", "u0").exists() and (run.unit_dir("work", "fakestage", "u0") / "data.txt").exists()
    moved = sorted((run.work_dir / "_incomplete").glob("fakestage_u0_*"))
    names = [item.name for folder in moved for item in folder.iterdir()]
    assert len(moved) == 2 and "01_u0.done" in names and names.count("00_u0") == 2, names   # outer dir; inner dir + inner marker
    assert body(False) == "skipped"
    print("    C. nested unit finished, parent not: both recomputed, old copies moved to _incomplete")


def main():
    rows = shard_rows_for(3)
    reference = Run(fresh_run_config("resume_reference", shard_rows=rows))
    for stage in STAGES:
        spore_xl.run_stage(reference, stage)
    interrupted_config = fresh_run_config("resume_interrupted", shard_rows=rows)
    run = Run(interrupted_config)
    for stage in STAGES[:STAGES.index("reduce2")]:
        spore_xl.run_stage(run, stage)

    # B (reduce2 part): leftovers of an interrupted reduce2 in splits/ and manifest/
    junk = [run.splits_dir / f"{run.name}_split_indices.json", run.manifest_dir / "global_obs_final.parquet"]
    for path in junk:
        path.write_bytes(b"half-written")
    spore_xl.run_stage(run, "reduce2")
    assert all(path.exists() and path.read_bytes() != b"half-written" for path in junk)
    moved_reduce2 = list((run.work_dir / "_incomplete").glob("reduce2_all_*/*"))
    assert len(moved_reduce2) == 2 and all(path.read_bytes() == b"half-written" for path in moved_reduce2)
    print("    B. reduce2: half-written split JSON and global_obs_final moved aside, stage recomputed")

    for stage in ("hvg",):
        spore_xl.run_stage(run, stage)

    # A: really kill map3 block 2 once its first part file exists
    block = 2
    process = subprocess.Popen([PY, "-B", CLI, "map3", "--config", str(interrupted_config), "--index", str(block)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ))
    deadline = time.time() + 300
    while time.time() < deadline and process.poll() is None and not part_files(run, block):
        time.sleep(0.05)
    killed = process.poll() is None
    if killed:
        process.kill()
    process.wait()
    marker = run.marker("work", "map3", f"block_{block:04d}")
    real_kill = killed and not marker.exists()
    print(f"    A. map3 block {block}: {'killed with ' + str(len(part_files(run, block))) + ' part file(s) on disk, no marker' if real_kill else 'finished before the kill (race); the simulated case B covers it'}")

    # B (map3 part): half-written part file and unit folder of block 3
    block_b = 3
    directory = run.unit_dir("work", "map3", f"block_{block_b:04d}")
    directory.mkdir(parents=True)
    (directory / "partial.txt").write_text("half", encoding="utf-8")
    victims = [path for path in (run.processed_dir / f"{run.name}_train_hvgraw" / f"part_{block_b:04d}.h5ad",)]
    victims[0].parent.mkdir(parents=True, exist_ok=True)
    victims[0].write_bytes(b"half-written part")

    test_nested_unit_finished_but_parent_not(run)

    # resume: the same stage commands again, then everything after
    for stage in STAGES[STAGES.index("map3"):]:
        spore_xl.run_stage(run, stage)

    incomplete = list((run.work_dir / "_incomplete").rglob("*"))
    moves = pd.read_csv(run.reports_dir / "incomplete_moves.tsv", sep="\t")
    assert len(moves) >= 2 + 2 + 1 and (moves["stage"] == "map3").any(), moves
    assert all(Path(path).exists() for path in moves["to"]), "a moved path is missing: nothing may be deleted"
    assert any(path.name == "part_0003.h5ad" or path.name.endswith("part_0003.h5ad") for path in incomplete)
    print(f"    {len(moves)} moves logged in reports/incomplete_moves.tsv; every moved path still exists ({len(incomplete)} files/folders under _incomplete)")

    ours, theirs = final_digest(run), final_digest(reference)
    different = [key for key in theirs if ours.get(key) != theirs[key]]
    assert not different and set(ours) == set(theirs), f"resumed run differs from the uninterrupted run: {different[:5]}"
    assert compare.xl_signature(run) == compare.xl_signature(reference)
    print(f"    resumed run == uninterrupted run: {len(ours)} final files content-identical, signatures equal")


if __name__ == "__main__":
    main()
    print("PASS test_resume_incomplete")
