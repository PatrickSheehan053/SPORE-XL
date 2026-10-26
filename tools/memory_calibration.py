"""Developer tool (read-only): calibrate the per-stage RAM estimates of the pre-flight from a FINISHED (or running) run.

  python -B tools/memory_calibration.py --config <the run's config yaml> [--repo <SPORE repository>] [--out per_unit.tsv] [--safety 1.5]

For every unit of the pooled stages (intake, map1, map2, gather, map3, metacell) it pairs the measured peak RSS (<run>/reports/memory_and_time.tsv) with the size of the unit's
input as the pre-flight sees it (stored values for the intake; file bytes for the other stages), and prints per stage: the number of units, the input and peak ranges,
what the CURRENT estimate (0.7 GB base + expansion x input size; intake 0.7 GB + 36 or 20 bytes per stored value) says for the largest unit, how many times it overestimates
the worst measured point, and the expansion that would cover every measured point with the safety factor. Nothing is written except the optional --out table.
Send the printed table (small) of each run: it is what the next calibration of src/engine/preflight.py is made from.
"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(sys.argv[sys.argv.index("--repo") + 1]) if "--repo" in sys.argv else Path(__file__).resolve().parent.parent   # --repo: use the xl modules of that SPORE repository
sys.path.insert(0, str(REPO / "src" / "engine"))

import pandas as pd  # noqa: E402

from xl_common import Run  # noqa: E402

BASE_GB = 0.7
CURRENT_EXPANSION = 4.0


def size_gb(path):
    path = Path(path)
    return path.stat().st_size / 1e9 if path.exists() else float("nan")


def unit_inputs(run):
    """{(stage, unit): (input size in GB, or stored values for the intake, what it is)} with the same inputs the pre-flight uses."""
    inputs = {}
    members = run.manifest_dir / "intake_members.tsv"
    if members.exists():
        table = pd.read_csv(members, sep="\t")
        for row in table.itertuples():
            inputs[("intake", f"shard_{int(row.index):04d}")] = (float(row.nnz), "stored values")
    shards = run.manifest_dir / "shards.tsv"
    if shards.exists():
        for index, name in enumerate(run.shards()["file"]):
            inputs[("map1", f"shard_{index:04d}")] = (size_gb(run.run_dir / name), "GB source shard")
    for path in sorted((run.work_dir / "map1").glob("shard_*/qc.h5ad")):
        inputs[("map2", path.parent.name)] = (size_gb(path), "GB qc.h5ad")
    pieces = {}
    for path in (run.work_dir / "map2").glob("shard_*/block_*.h5ad"):
        pieces[path.name[:-5]] = pieces.get(path.name[:-5], 0.0) + size_gb(path)
    for unit, total in pieces.items():
        inputs[("gather", unit)] = (total, "GB map2 pieces of the block")
    for path in sorted((run.work_dir / "gather").glob("block_*/block.h5ad")):
        inputs[("map3", path.parent.name)] = (size_gb(path), "GB block.h5ad")
    parts = {}
    for path in run.processed_dir.glob(f"{run.name}_*_hvgraw/part_*.h5ad"):
        parts[path.name] = parts.get(path.name, 0.0) + size_gb(path)
    for name, total in parts.items():
        inputs[("metacell", "block_" + name[len("part_"):-len(".h5ad")])] = (total, "GB hvgraw parts of the block")
    return inputs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo", default=None, help="SPORE-XL repository whose src/engine/ modules are used (default: the one this file sits in)")
    parser.add_argument("--out", default=None, help="per-unit table (a new file; refuses to overwrite)")
    parser.add_argument("--safety", type=float, default=1.5)
    args = parser.parse_args(argv)
    run = Run(args.config)
    measured = pd.read_csv(run.reports_dir / "memory_and_time.tsv", sep="\t")
    inputs = unit_inputs(run)
    rows = []
    for row in measured.itertuples():
        key = (row.stage, row.unit)
        if key not in inputs:
            continue
        size, kind = inputs[key]
        peak = float(row.peak_rss_gb)
        current = BASE_GB + (size * 36 / 1e9 if row.stage == "intake" else size * CURRENT_EXPANSION)
        measure_per_input = max(0.0, (peak - BASE_GB) / (size / 1e9 if row.stage == "intake" else size)) if size and size == size else float("nan")   # a peak below the 0.7 GB base counts as 0
        rows.append({"stage": row.stage, "unit": row.unit, "input": size, "input_kind": kind, "peak_rss_gb": peak, "seconds": float(row.seconds),
                     "current_estimate_gb": current, "peak_per_input": measure_per_input})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("no unit of a pooled stage has both a measured peak and an input file (was the clean-up switched on? is memory_and_time.tsv empty?)")
    if args.out:
        out = Path(args.out)
        if out.exists():
            raise FileExistsError(f"refusing to overwrite {out}")
        frame.to_csv(out, sep="\t", index=False)
    print(f"run {run.name}: {len(frame)} units with a measured peak and an input")
    print(f"{'stage':9s} {'units':>5s} {'input min..max':>26s} {'peak GB min..max':>18s} {'current est. (largest)':>23s} {'overest.':>9s} {'worst peak/input':>17s} {'proposed':>26s}")
    for stage, part in frame.groupby("stage", sort=False):
        largest = part.loc[part["input"].idxmax()]
        worst = part["peak_per_input"].max()
        overestimate = part["current_estimate_gb"].max() / part["peak_rss_gb"].max()
        if stage == "intake":
            proposed = f"{args.safety * worst:.1f} bytes per stored value"
            span = f"{part['input'].min() / 1e6:.0f}..{part['input'].max() / 1e6:.0f} M values"
        else:
            proposed = f"expansion {args.safety * worst:.2f} (now {CURRENT_EXPANSION})"
            span = f"{part['input'].min():.3f}..{part['input'].max():.3f} GB"
        print(f"{stage:9s} {len(part):5d} {span:>26s} {part['peak_rss_gb'].min():8.2f}..{part['peak_rss_gb'].max():<8.2f} {largest['current_estimate_gb']:23.2f} "
              f"{overestimate:8.1f}x {worst:17.2f} {proposed:>26s}")
    others = measured[~measured.set_index(["stage", "unit"]).index.isin(list(inputs))]
    singles = others[others["unit"] == "all"]
    if len(singles):
        print("single-task stages (not pooled, shown for the record): " + "; ".join(f"{r.stage} {r.peak_rss_gb:.2f} GB in {r.seconds:.0f} s" for r in singles.itertuples()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
