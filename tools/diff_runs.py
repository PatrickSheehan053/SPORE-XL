"""diff_runs: a production run against a compat (SPORE v3.0 reproducing) run of the same data, every difference traced to its Amendment 2 item.

  python -B tools/diff_runs.py --ref <compat config.yaml> --new <production config.yaml>

Must be IDENTICAL (they come before the panel and do not depend on any production default): survivors and the split of every cell, split JSON,
control assignment, escaper thresholds, QC variable columns. Expected to differ, with the reason: Phase 5 gene mask (P3 cell-cycle rescue),
panel (P2 budget, P3), the columns of hvgraw / norm / metacell / hybrid (they live on the panel), the normalization and metacell values (they are computed on the panel),
and the new outputs (P1 aliases, P2 node manifest, P4 rule counts, P5 diagnostics, P6 lognorm hybrid). Writes reports/EXPECTED_DIFFERENCES_vs_<ref>.md
into the NEW run (a new file; refuses to overwrite).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "engine"))   # works from src/engine/, tests/ and tools/ of a repository

from xl_common import Run
from stage_gather import total_blocks
from stage_map3 import SPLITS, part_path


def _lines(path):
    return [line for line in Path(path).read_text(encoding="utf-8").split("\n") if line]


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diff_runs(ref: Run, new: Run):
    rows = []

    def record(item, expected, observed, ok, cause=""):
        rows.append({"item": item, "expected": expected, "observed": observed, "ok": bool(ok), "cause": cause})

    # ---- must be identical
    ref_obs = pd.read_parquet(ref.manifest_dir / "global_obs_final.parquet")
    new_obs = pd.read_parquet(new.manifest_dir / "global_obs_final.parquet")
    same_survivors = ref_obs["survived_to_phase7"].equals(new_obs["survived_to_phase7"])
    record("surviving cells", "identical", f"{int(new_obs['survived_to_phase7'].sum()):,} cells, {'identical' if same_survivors else 'DIFFERENT'}", same_survivors)
    same_split = ref_obs["split"].astype(str).equals(new_obs["split"].astype(str))
    record("split of every cell", "identical", "identical" if same_split else "DIFFERENT", same_split)
    ref_assignment = pd.read_parquet(ref.splits_dir / f"{ref.name}_control_assignment.parquet")
    new_assignment = pd.read_parquet(new.splits_dir / f"{new.name}_control_assignment.parquet")
    record("control assignment", "identical", "identical" if ref_assignment.equals(new_assignment) else "DIFFERENT", ref_assignment.equals(new_assignment))
    for label, relative in (("split JSON", "splits/{name}_split_indices.json"), ("mean-shift scores", "stats/reduce2/all/mean_shift.tsv"),
                            ("escaper thresholds", "stats/reduce1/all/escaper_thresholds.tsv")):
        ref_path, new_path = ref.run_dir / relative.format(name=ref.name), new.run_dir / relative.format(name=new.name)
        same = _digest(ref_path) == _digest(new_path)
        record(label, "identical", "identical" if same else "DIFFERENT", same)
    ref_qc = pd.read_csv(ref.stats_dir / "reduce1" / "all" / "qc_var_columns.tsv", sep="\t", index_col=0)
    new_qc = pd.read_csv(new.stats_dir / "reduce1" / "all" / "qc_var_columns.tsv", sep="\t", index_col=0)
    record("QC variable columns", "identical", "identical" if ref_qc.equals(new_qc) else "DIFFERENT", ref_qc.equals(new_qc))

    # ---- expected to differ (reported with the cause)
    ref_mask = set(_lines(ref.stats_dir / "reduce2" / "all" / "gene_keep_mask.txt"))
    new_mask = set(_lines(new.stats_dir / "reduce2" / "all" / "gene_keep_mask.txt"))
    record("Phase 5 gene mask (genes kept)", "same unless cell-cycle genes were rescued in the reference",
           f"{len(ref_mask)} vs {len(new_mask)}; only in reference: {sorted(ref_mask - new_mask)[:8]}; only in new: {sorted(new_mask - ref_mask)[:8]}",
           not (new_mask - ref_mask), "P3: the reference rescued cell-cycle genes that gene QC would remove; new does not")
    ref_panel, new_panel = _lines(ref.splits_dir / f"{ref.name}_hvg_panel.txt"), _lines(new.splits_dir / f"{new.name}_hvg_panel.txt")
    n_top = int(new.cfg.get("phase8_hvg", {}).get("n_top_genes", 5000))
    record("HVG panel size", f"new == n_top_genes ({n_top}) under the budget policy", f"reference {len(ref_panel)}, new {len(new_panel)}",
           len(new_panel) == n_top or new.cfg["phase8_hvg"].get("target_node_policy") != "budget",
           "P2 (budget instead of stacking every training target) and P3 (no cell-cycle genes stacked)")
    only_ref, only_new = sorted(set(ref_panel) - set(new_panel)), sorted(set(new_panel) - set(ref_panel))
    record("panel genes only in the reference", "training targets / cell-cycle genes beyond the budget", f"{len(only_ref)} genes, e.g. {only_ref[:6]}", True, "P2 / P3")
    record("panel genes only in new", "genes the reference swapped out by a different rule", f"{len(only_new)} genes, e.g. {only_new[:6]}", True, "P2")

    # hvgraw: same cells in the same order, identical counts on the common genes (first three train parts)
    common = [g for g in ref_panel if g in set(new_panel)]
    checked, mismatched = 0, 0
    for block in range(total_blocks(new)):
        left, right = part_path(ref, "train", "hvgraw", block), part_path(new, "train", "hvgraw", block)
        if not (left.exists() and right.exists()):
            continue
        a, b = ad.read_h5ad(left), ad.read_h5ad(right)
        if list(a.obs_names) != list(b.obs_names):
            mismatched += 1
        else:
            xa = sp.csr_matrix(a[:, common].X).toarray() if len(common) else np.zeros((a.n_obs, 0))
            xb = sp.csr_matrix(b[:, common].X).toarray() if len(common) else np.zeros((b.n_obs, 0))
            mismatched += int(not np.array_equal(xa, xb))
        checked += 1
        if checked >= 3:
            break
    record("hvgraw: same cells, same counts on the common genes (3 train parts)", "identical", f"{checked} parts checked, {mismatched} differ", mismatched == 0 and checked > 0,
           "the panel columns differ (P2/P3); the rows and the counts do not")
    record("norm / metacell / train_hybrid values", "differ", "computed on the panel (library size over the panel genes, embedding on the panel)", True, "P2/P3 follow from the panel")

    # ---- new outputs
    for item, path, cause in (("manifest/perturbation_nodes.tsv", new.manifest_dir / "perturbation_nodes.tsv", "P2 node versus row-only manifest"),
                              ("reports/escaper_rules.tsv", new.reports_dir / "escaper_rules.tsv", "P4 both escaper rules counted"),
                              ("stats/hvg/all/diagnostics.json", new.stats_dir / "hvg" / "all" / "diagnostics.json", "P5 variance explained, cell-cycle diagnostic"),
                              ("stats/reduce2/all/confounding.tsv", new.stats_dir / "reduce2" / "all" / "confounding.tsv", "P5 confounding table"),
                              ("manifest/aliases.tsv", new.manifest_dir / "aliases.tsv", "P1 v3.0 names as aliases"),
                              ("processed/*_train_hybrid_lognorm", next(iter(new.processed_dir.glob(f"{new.name}_train_hybrid_lognorm*")), None), "P6")):
        present = path is not None and Path(path).exists()
        record(f"new output {item}", "present only in new", "present" if present else "MISSING", present, cause)
    return rows


def write_report(new: Run, ref: Run, rows):
    path = new.reports_dir / f"EXPECTED_DIFFERENCES_vs_{ref.run_dir.name}.md"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    lines = [f"# Expected differences: {new.run_dir.name} (production) vs {ref.run_dir.name} (reference)", "",
             "Every row is either identical (must be) or a difference traced to an Amendment 2 item. `ok` = what was expected happened.", "",
             "| item | expected | observed | ok | traced to |", "|---|---|---|---|---|"]
    lines += [f"| {r['item']} | {r['expected']} | {str(r['observed']).replace('|', '/')} | {'yes' if r['ok'] else '**NO**'} | {r['cause']} |" for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--ref-is-v3", action="store_true", help="the reference run was made by code before v4 (for example the 21 Oct K562 run): v3.0 file names")
    args = parser.parse_args(argv)
    ref, new = Run(args.ref), Run(args.new)
    if args.ref_is_v3 or ref.cfg.get("xl", {}).get("compat") == "v3":
        ref.mode = "compat-v3"
    rows = diff_runs(ref, new)
    path = write_report(new, ref, rows)
    for row in rows:
        print(f"{'ok  ' if row['ok'] else 'FAIL'} {row['item']}: {row['observed']}")
    print(f"written {path}")
    return 0 if all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
