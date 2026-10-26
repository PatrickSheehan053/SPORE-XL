"""verify: hard gates after a run. Writes reports/VERIFICATION.md and returns [(check, ok, detail), ...].

Split-integrity checks (a)-(g) are those of the blueprint's "Split integrity (leakage firewall)" section, item 5:
 (a) label sets of the train/val/test parts equal the split JSON exactly
 (b) the three sets are pairwise disjoint and their union is the full surviving set
 (c) for every perturbation, cells after the escaper filter equal cells in its split (nothing lost or duplicated)
 (d) every control cell has exactly one split
 (e) no xl_global_index appears twice anywhere
 (f) train_hybrid contains no val or test units
 (g) every metacell is built from one perturbation and therefore one split
plus unit/checksum accounting, cell accounting, one-block-per-perturbation, identical var across all parts.
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from xl_common import Run, block_unit, shard_unit, verify_marker
from stage_finalize import hybrid_part_path
from stage_gather import escaper_counts, total_blocks
from stage_map3 import SPLITS, part_path
from stage_metacell import metacell_part_path


def _obs_of(path):
    part = ad.read_h5ad(path, backed="r")
    try:
        return part.obs.copy(), list(part.var_names)
    finally:
        part.file.close()


def verify_run(run: Run, write_report=True):
    results = []

    def record(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    n_total = total_blocks(run)
    layout = run.layout()

    # ---- units and checksums
    expected = {("stats", "map1"): run.n_shards(), ("stats", "map2"): run.n_shards(), ("stats", "gather"): n_total,
                ("stats", "reduce1"): 1, ("stats", "reduce2"): 1, ("stats", "hvg"): 1, ("work", "map3"): n_total,
                ("stats", "metacell_prep"): 1, ("stats", "metacell"): n_total, ("stats", "finalize"): 1}
    problems = []
    for (root, stage), count in expected.items():
        base = (run.stats_dir if root == "stats" else run.work_dir) / stage
        markers = sorted(base.glob("*.done")) if base.exists() else []
        if len(markers) != count:
            problems.append(f"{root}/{stage}: {len(markers)} of {count} units done")
        for marker in markers:
            try:
                verify_marker(marker, marker.with_name(marker.stem), run.run_dir)
            except Exception as error:  # noqa: BLE001
                problems.append(f"{marker}: {error}")
    record("every unit has a .done marker and its files match their checksums", not problems, "; ".join(problems[:3]) or f"{sum(expected.values())} units")

    # ---- survivors and cell accounting
    final = pd.read_parquet(run.manifest_dir / "global_obs_final.parquet")
    survivors = final[final["survived_to_phase7"]]
    surviving_labels = set(survivors["label"].astype(str)) - {run.ctrl_label}
    summary2 = json.loads((run.stats_dir / "reduce2" / "all" / "summary.json").read_text(encoding="utf-8"))
    record("survivors in global_obs_final equal reduce2's count", len(survivors) == summary2["n_survivors"], f"{len(survivors):,}")
    gather_labels, duplicated_blocks = {}, []
    for block in range(n_total):
        frame = pd.read_parquet(run.stats_dir / "gather" / block_unit(block) / "survivors.parquet")
        for label in set(frame["label"].astype(str)) - {run.ctrl_label}:
            if gather_labels.setdefault(label, block) != block:
                duplicated_blocks.append(label)
    record("every perturbation is in exactly one block", not duplicated_blocks and set(gather_labels) == surviving_labels,
           f"{len(gather_labels)} perturbations; split across blocks: {duplicated_blocks[:3]}")

    # ---- collect the hvgraw parts
    split_obs, var_sets, all_gidx = {}, set(), []
    for split in SPLITS:
        frames = []
        for block in range(n_total):
            path = part_path(run, split, "hvgraw", block)
            if path.exists():
                obs, var = _obs_of(path)
                frames.append(obs)
                var_sets.add(tuple(var))
        split_obs[split] = pd.concat(frames) if frames else pd.DataFrame(columns=[run.pert_col, "xl_global_index"])
        all_gidx.append(split_obs[split]["xl_global_index"].to_numpy())
    split_json = json.loads((run.splits_dir / f"{run.name}_split_indices.json").read_text(encoding="utf-8"))
    json_sets = {split: set(split_json[f"{split}_labels"]) for split in SPLITS}

    # (a)
    part_sets = {split: set(split_obs[split][run.pert_col].astype(str)) - {run.ctrl_label} for split in SPLITS}
    record("(a) label sets of the train/val/test parts equal the split JSON", part_sets == json_sets,
           {split: (len(part_sets[split]), len(json_sets[split])) for split in SPLITS})
    # (b)
    disjoint = not (json_sets["train"] & json_sets["val"]) and not (json_sets["train"] & json_sets["test"]) and not (json_sets["val"] & json_sets["test"])
    union = json_sets["train"] | json_sets["val"] | json_sets["test"]
    record("(b) train/val/test label sets are pairwise disjoint and cover the surviving set", disjoint and union == surviving_labels,
           f"{len(union)} labels vs {len(surviving_labels)} surviving")
    # (c)
    counts = escaper_counts(run)
    n_cells_by_label = pd.concat([split_obs[split][run.pert_col].astype(str).value_counts() for split in SPLITS])
    bad = [label for label in surviving_labels if int(counts.loc[label, "n_kept"]) != int(n_cells_by_label.get(label, 0))]
    record("(c) for every perturbation, cells after the escaper filter equal cells in its split", not bad, f"mismatches: {bad[:3]}")
    # (d)
    assignment = pd.read_parquet(run.splits_dir / f"{run.name}_control_assignment.parquet")
    survivor_controls = set(survivors.loc[survivors["label"].astype(str) == run.ctrl_label, "xl_global_index"])
    per_split_ok = all(
        set(split_obs[split].loc[split_obs[split][run.pert_col].astype(str) == run.ctrl_label, "xl_global_index"])
        == set(assignment.loc[assignment["split"] == split, "xl_global_index"]) for split in SPLITS)
    record("(d) every control cell has exactly one split",
           assignment["xl_global_index"].is_unique and set(assignment["xl_global_index"]) == survivor_controls and per_split_ok,
           f"{len(assignment):,} controls assigned")
    # (e)
    merged = np.concatenate(all_gidx)
    record("(e) no xl_global_index appears twice", len(merged) == len(np.unique(merged)) and len(merged) == len(survivors),
           f"{len(merged):,} cells in the parts, {len(survivors):,} survivors")
    # (f)
    hybrid_splits, hybrid_labels = set(), set()
    for block in range(n_total):
        path = hybrid_part_path(run, block)
        if path.exists():
            obs, var = _obs_of(path)
            hybrid_splits |= set(obs["split"].astype(str))
            hybrid_labels |= set(obs[run.pert_col].astype(str)) - {run.ctrl_label}
            var_sets.add(tuple(var))
    record("(f) train_hybrid contains no val or test units", hybrid_splits <= {"train", "control"} and hybrid_labels <= json_sets["train"],
           f"split values {sorted(hybrid_splits)}")
    # (g)
    unit_splits, ctrl_ok = {}, True
    for block in range(n_total):
        path = metacell_part_path(run, block)
        if path.exists():
            obs, var = _obs_of(path)
            var_sets.add(tuple(var))
            for label, group in obs.groupby(obs[run.pert_col].astype(str)):
                unit_splits.setdefault(label, set()).update(group["split"].astype(str))
                if label == run.ctrl_label and not (group["n_cells_in_metacell"] == 1).all():
                    ctrl_ok = False
    expected_split = {**{l: "train" for l in json_sets["train"]}, **{l: "val" for l in json_sets["val"]}, **{l: "test" for l in json_sets["test"]}}
    one_split = all(len(values) == 1 for values in unit_splits.values())
    right_split = all(unit_splits[label] == {expected_split[label]} for label in expected_split if label in unit_splits)
    record("(g) every metacell is built from one perturbation, hence one split; controls are single-cell",
           one_split and right_split and ctrl_ok and unit_splits.get(run.ctrl_label) == {"control"}, f"{len(unit_splits)} groups")
    # var names identical everywhere
    record("var names identical across all parts", len(var_sets) == 1, f"{len(var_sets)} distinct var lists over hvgraw/metacell/hybrid parts")
    # p9 has the same cells as hvgraw
    p9_ok = True
    for split in SPLITS:
        for block in range(n_total):
            raw_path, p9_path = part_path(run, split, "hvgraw", block), part_path(run, split, "p9", block)
            if raw_path.exists() != p9_path.exists():
                p9_ok = False
            elif raw_path.exists():
                if list(_obs_of(raw_path)[0].index) != list(_obs_of(p9_path)[0].index):
                    p9_ok = False
    record("p9 parts hold exactly the cells of the hvgraw parts", p9_ok)
    # P6: the log-normalized training hybrid has exactly the units of train_hybrid, finite, rows summing to target_sum after expm1
    from stage_finalize import hybrid_lognorm_path, lognorm_enabled
    if lognorm_enabled(run):
        problems, n_checked = [], 0
        target = float(run.cfg.get("phase9_normalization", {}).get("target_sum", 10000))
        for block in range(n_total):
            hybrid_path, lognorm_path = hybrid_part_path(run, block), hybrid_lognorm_path(run, block)
            if hybrid_path.exists() != lognorm_path.exists():
                problems.append(f"block {block}: only one of train_hybrid / train_hybrid_lognorm exists")
            elif hybrid_path.exists():
                raw_part, norm_part = ad.read_h5ad(hybrid_path), ad.read_h5ad(lognorm_path)
                if list(raw_part.obs.index) != list(norm_part.obs.index):
                    problems.append(f"block {block}: different units")
                dense = np.asarray(norm_part.X.toarray() if hasattr(norm_part.X, "toarray") else norm_part.X)
                raw_dense = np.asarray(raw_part.X.toarray() if hasattr(raw_part.X, "toarray") else raw_part.X)
                nonempty = raw_dense.sum(axis=1) > 0
                sums = np.expm1(dense.astype(np.float64)).sum(axis=1)
                if not np.isfinite(dense).all() or (np.abs(sums[nonempty] - target) > 0.002 * target).any():
                    problems.append(f"block {block}: values not finite or rows do not sum to {target:g}")
                n_checked += int(nonempty.sum())
        record("train_hybrid_lognorm has the units of train_hybrid, finite, rows sum to target_sum after expm1", not problems,
               "; ".join(problems[:3]) or f"{n_checked} units")
    aliases_path = run.manifest_dir / "aliases.tsv"
    if aliases_path.exists():
        aliases = pd.read_csv(aliases_path, sep="	")
        bad = [row.alias for row in aliases.itertuples()
               if "NOT CREATED" not in str(row.method)
               and (not (run.run_dir / row.alias).exists() or (run.run_dir / row.alias).stat().st_size != (run.run_dir / row.target).stat().st_size)]
        record("v3.0-name aliases (_p9) exist and match their _norm targets", not bad, f"{len(aliases)} aliases; bad: {bad[:3]}")
    record("GRAPH-READY in COVERAGE_REPORT", "GRAPH-READY" in (run.reports_dir / "COVERAGE_REPORT.md").read_text(encoding="utf-8")
           and "NOT READY" not in (run.reports_dir / "COVERAGE_REPORT.md").read_text(encoding="utf-8"))

    if write_report:
        path = run.reports_dir / "VERIFICATION.md"
        if path.exists():
            path = run.reports_dir / f"VERIFICATION_{len(list(run.reports_dir.glob('VERIFICATION*.md'))) + 1}.md"
        passed = all(ok for _, ok, _ in results)
        lines = [f"# VERIFICATION {'PASS' if passed else 'FAIL'}", "", "| check | result | detail |", "|---|---|---|"]
        lines += [f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |" for name, ok, detail in results]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    outcome = verify_run(Run(sys.argv[1]))
    for name, ok, detail in outcome:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    sys.exit(0 if all(ok for _, ok, _ in outcome) else 1)
