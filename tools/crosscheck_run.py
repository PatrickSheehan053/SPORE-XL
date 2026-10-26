"""Prints the headline numbers of a finished SPORE-XL run next to the numbers of another SPORE run's log (read-only).
Defaults are Michele's K562 GWPS SPORE log (its run took SPORE's >1M-cell chunked path, so NEAR agreement is expected, not exact).

  python -B src/crosscheck_run.py --config <xl_config.yaml> [--expect-mt-removed 11599 --expect-escapers 169940 --expect-bypassed 2182
          --expect-undersized-groups 748 --expect-undersized-cells 24047 --expect-split 6720/800/1598 --expect-panel 7090]
EXACT = same number; NEAR = within 1 percent; CHECK = further apart (report it, do not guess why).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "engine"))   # works from src/engine/, tests/ and tools/ of a repository

from xl_common import Run


def read_numbers(run):
    reduce1 = json.loads((run.stats_dir / "reduce1" / "all" / "waterfall_phase2.json").read_text(encoding="utf-8"))
    counts = pd.concat([pd.read_parquet(path) for path in sorted(glob.glob(str(run.stats_dir / "map2" / "shard_*" / "counts.parquet")))])
    totals = counts.groupby("label").agg(n_total=("n_total", "sum"), n_kept=("n_kept", "sum"), bypassed=("bypassed", "max"))
    dropped = pd.read_csv(run.stats_dir / "reduce2" / "all" / "dropped_undersized.tsv", sep="\t")
    summary = json.loads((run.stats_dir / "reduce2" / "all" / "summary.json").read_text(encoding="utf-8"))
    hvg = json.loads((run.stats_dir / "hvg" / "all" / "summary.json").read_text(encoding="utf-8"))
    return {
        "MT filter removed": reduce1["Starting cells"] - reduce1["After MT filter"],
        "escapers removed": int((totals["n_total"] - totals["n_kept"]).sum()),
        "bypassed perturbations (target not measured)": int(totals["bypassed"].sum()),
        "undersized groups": int(len(dropped)),
        "undersized cells": int(dropped["n_kept"].sum()),
        "split train/val/test": f"{summary['n_train']}/{summary['n_val']}/{summary['n_test']}",
        "HVG panel size": int(hvg["panel_size"]),
        "(info) QC-passing cells": int(reduce1["After max counts"]),
        "(info) survivors to Phase 7": int(summary["n_survivors"]),
        "(info) perturbation groups surviving": int(summary["n_perturbations"]),
    }


def verdict(ours, theirs):
    if theirs is None:
        return ""
    if str(ours) == str(theirs):
        return "EXACT"
    if isinstance(ours, int) and isinstance(theirs, int):
        relative = abs(ours - theirs) / max(abs(theirs), 1)
        return f"NEAR ({ours - theirs:+,}; {relative:.2%})" if relative <= 0.01 else f"CHECK ({ours - theirs:+,}; {relative:.2%})"
    return "CHECK"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expect-mt-removed", type=int, default=11599)
    parser.add_argument("--expect-escapers", type=int, default=169940)
    parser.add_argument("--expect-bypassed", type=int, default=2182)
    parser.add_argument("--expect-undersized-groups", type=int, default=748)
    parser.add_argument("--expect-undersized-cells", type=int, default=24047)
    parser.add_argument("--expect-split", default="6720/800/1598")
    parser.add_argument("--expect-panel", type=int, default=7090)
    args = parser.parse_args(argv)
    run = Run(args.config)
    ours = read_numbers(run)
    expected = {"MT filter removed": args.expect_mt_removed, "escapers removed": args.expect_escapers,
                "bypassed perturbations (target not measured)": args.expect_bypassed, "undersized groups": args.expect_undersized_groups,
                "undersized cells": args.expect_undersized_cells, "split train/val/test": args.expect_split, "HVG panel size": args.expect_panel}
    print(f"{'item':48s} {'XL run':>12s} {'expected':>12s}  verdict")
    worst = 0
    for name, value in ours.items():
        theirs = expected.get(name)
        text = verdict(value, theirs)
        worst = max(worst, 2 if text.startswith("CHECK") else 1 if text.startswith("NEAR") else 0)
        print(f"{name:48s} {value if isinstance(value, str) else format(value, ','):>12s} {'' if theirs is None else (theirs if isinstance(theirs, str) else format(theirs, ',')):>12}  {text}")
    print(["ALL EXACT", "NEAR AGREEMENT (expected above 1M cells)", "CHECK: at least one number is more than 1 percent away"][worst])
    return 0


if __name__ == "__main__":
    sys.exit(main())
