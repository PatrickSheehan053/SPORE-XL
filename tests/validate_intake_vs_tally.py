"""Validate the intake of a finished recipe run against the VERIFIED tally of GSE314342 (min_umi 3), to be run ON THE CLUSTER over all
Rest shards (and in the unit test on synthetic data).

  python -B tests\\validate_intake_vs_tally.py --config run.yaml --tally gse314342_merged_targets.tsv --group Rest --min-umi 3
         [--summary gse314342_merged_summary.tsv] --out intake_vs_tally.tsv

Checks (all must PASS):
 1 RAW per-target counts (the tally rule, no label normalization): cells_one_target and cells_any per target equal the tally, for
   every target (1-Jun and 2-Jun still separate here)
 2 NORMALIZED written cells per label: every target except the merged ones equals the tally's single-target count; JUN equals
   tally(1-Jun) + tally(2-Jun) + (cells carrying exactly both JUN guides); the control label equals tally(NTC) + tally(CTRL) +
   (cells carrying exactly NTC and CTRL). The pair counts come from the intake waterfall, so the formula is exact, not approximate
 3 (with --summary) totals: samples, cells, cells with any guide, cells with 0 / 1 / 2 / 3+ raw targets equal the tally summary row
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "engine"))

import pandas as pd

from xl_common import Run, shard_unit


def load_intake(run):
    members = pd.read_csv(run.manifest_dir / "intake_members.tsv", sep="\t")
    metas, raw_frames, label_frames = [], [], []
    for index in range(len(members)):
        unit = run.stats_dir / "intake" / shard_unit(index)
        if not run.marker("stats", "intake", shard_unit(index)).exists():
            raise FileNotFoundError(f"intake unit {index} is not finished")
        metas.append(json.loads((unit / "meta.json").read_text(encoding="utf-8")))
        raw_frames.append(pd.read_csv(unit / "targets_raw.tsv", sep="\t", keep_default_na=False))
        label_frames.append(pd.read_csv(unit / "labels.tsv", sep="\t", keep_default_na=False))
    raw = pd.concat(raw_frames).groupby("target_raw")[["cells_one_target", "cells_any"]].sum()
    labels = pd.concat(label_frames).groupby("label")["cells_written"].sum()
    return metas, raw, labels


def validate(run, tally, summary=None, group="Rest", min_umi=3, merged=None, control_targets=("NTC", "CTRL"), control_label="non-targeting"):
    merged = merged or {"JUN": ["1-Jun", "2-Jun"]}
    metas, raw, labels = load_intake(run)
    table = tally[(tally["group"] == group) & (tally["min_umi"] == min_umi)].set_index("target")
    results = []

    def record(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    # 1 raw
    only_tally = sorted(set(table.index) - set(raw.index))
    only_intake = sorted(set(raw.index) - set(table.index))
    both = sorted(set(table.index) & set(raw.index))
    bad_one = [t for t in both if int(raw.loc[t, "cells_one_target"]) != int(table.loc[t, "cells_one_target"])]
    bad_any = [t for t in both if int(raw.loc[t, "cells_any"]) != int(table.loc[t, "cells_any"])]
    record("raw single-target cells per target equal the tally", not bad_one and not only_tally and not only_intake,
           f"{len(both)} targets compared; only in tally {only_tally[:3]}; only in intake {only_intake[:3]}; differing {bad_one[:3]}")
    record("raw cells carrying the target (any) per target equal the tally", not bad_any, f"differing {bad_any[:3]}")

    # 2 normalized
    pairs = {}
    for meta in metas:
        for label, count in meta["merged_pair_cells"].items():
            pairs[label] = pairs.get(label, 0) + int(count)
    expected = {}
    merged_members = {t for members in merged.values() for t in members} | set(control_targets)
    for target in table.index:
        if target not in merged_members:
            expected[target] = int(table.loc[target, "cells_one_target"])
    for label, members in merged.items():
        expected[label] = sum(int(table.loc[m, "cells_one_target"]) for m in members if m in table.index) + pairs.get(label, 0)
    expected[control_label] = sum(int(table.loc[t, "cells_one_target"]) for t in control_targets if t in table.index) + pairs.get(control_label, 0)
    mismatched = [label for label in expected if int(labels.get(label, 0)) != expected[label]]
    extra = sorted(set(labels.index) - set(expected))
    record("normalized written cells per label equal the tally (JUN and control by the pair formula)", not mismatched and not extra,
           f"{len(expected)} labels; differing {[(l, int(labels.get(l, 0)), expected[l]) for l in mismatched[:3]]}; unexpected labels {extra[:3]}")
    record("JUN pair cells (both guides, no other target)", True, f"{pairs}")

    # 3 totals
    if summary is not None:
        row = summary[(summary["group"] == group) & (summary["min_umi"] == min_umi)].iloc[0]
        totals = {"samples": len(metas), "cells": sum(m["n_cells"] for m in metas),
                  "cells_with_any_guide": sum(m["n_cells_with_any_guide"] for m in metas),
                  "cells_0_targets": sum(m["raw_cells_0_targets"] for m in metas), "cells_1_target": sum(m["raw_cells_1_target"] for m in metas),
                  "cells_2_targets": sum(m["raw_cells_2_targets"] for m in metas), "cells_3plus_targets": sum(m["raw_cells_3plus_targets"] for m in metas)}
        differing = {key: (value, int(row[key])) for key, value in totals.items() if int(row[key]) != value}
        record("totals equal the tally summary (samples, cells, cells with any guide, 0/1/2/3+ targets)", not differing, f"differing {differing}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tally", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--group", default="Rest")
    parser.add_argument("--min-umi", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run = Run(args.config)
    tally = pd.read_csv(args.tally, sep="\t", keep_default_na=False, dtype={"target": str})
    summary = pd.read_csv(args.summary, sep="\t") if args.summary else None
    results = validate(run, tally, summary, args.group, args.min_umi, control_label=run.ctrl_label)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    pd.DataFrame(results, columns=["check", "pass", "detail"]).to_csv(out, sep="\t", index=False)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
