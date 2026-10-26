"""phase8_hvg.target_node_policy on a genome-wide-like synthetic (1,000 perturbations over about 1,800 kept genes: most genes are targets).
 - all (default, SPORE): the panel is STACKED above n_top_genes (the K562 / CD4 effect)
 - budget: the panel is exactly n_top_genes; held-out targets and protected genes always in; train targets limited by max_target_nodes
 - the re-run path: a branch run reuses plan..reduce2 of the finished run (hard links) and re-runs hvg onward; the original run is untouched
 - policy `all` re-run in a branch reproduces the original hvg outputs exactly
(`all` equal to SPORE itself is covered by test_stage_hvg and test_stage_hvg_subsample.)
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; set MPLBACKEND=Agg ; set XL_TEST_TMP=<scratch> ; python -B tests\\test_target_policy.py"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_support import variant_config  # noqa: E402
from xl_test_support import BUILD, scratch_for  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import branch  # noqa: E402
import cli as spore_xl  # noqa: E402
import verify  # noqa: E402
from xl_common import Run  # noqa: E402

SOURCE = BUILD / "data" / "synthetic_gw_90k.h5ad"
LATER = ["hvg", "map3", "metacell_prep", "metacell", "finalize", "report"]
EARLY = ["plan", "map1", "reduce1", "map2", "gather", "reduce2"]
_STATE = {}


def read_lines(path):
    return [line for line in Path(path).read_text(encoding="utf-8").split("\n") if line]


def base_run():
    if "a" in _STATE:
        return _STATE["a"]
    config = variant_config("gw_all", SOURCE, shard_rows=30000, test_n=200, val_n=100, n_blocks=16,
                            overrides={"phase5_escaper_filtering": {"min_cells_per_perturbation": 10},
                                       "phase8_hvg": {"n_top_genes": 500}})
    run = Run(config)
    for stage in EARLY + ["hvg"]:
        spore_xl.run_stage(run, stage)
    _STATE["a"] = run
    return run


def make_branch(run_a, label, phase8_updates):
    cfg = yaml.safe_load(Path(run_a.config_path).read_text(encoding="utf-8"))
    run_dir = scratch_for(f"policy_{label}")
    cfg["paths"]["project_root"] = str(run_dir).replace("\\", "/")
    cfg["phase8_hvg"].update(phase8_updates)
    path = run_dir.parent / f"{run_dir.name}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    branch.branch(run_a.config_path, path, "hvg")
    return Run(path)


def tree_digest(run):
    digest = {}
    for path in sorted(run.run_dir.rglob("*")):
        if path.is_file() and path.suffix not in (".done",) and "reports" not in path.parts:
            digest[str(path.relative_to(run.run_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def test_all_policy_stacks_the_panel_on_genome_wide_data():
    run = base_run()
    panel = read_lines(run.splits_dir / f"{run.name}_hvg_panel.txt")
    summary = json.loads((run.stats_dir / "hvg" / "all" / "summary.json").read_text(encoding="utf-8"))
    assert summary["target_node_policy"] == "all" and len(panel) == summary["panel_size"]
    assert len(panel) > 500 + 100, f"expected a stacked panel, got {len(panel)}"
    print(f"    policy all: panel {len(panel)} genes for n_top_genes 500 (stacked by {summary['panel_stacked_above_n_top']}), "
          f"{summary['train_targets_in_panel']} train targets in the panel")


def test_budget_policy_keeps_n_top_and_rerun_path_works():
    run_a = base_run()
    before = tree_digest(run_a)
    run_b = make_branch(run_a, "budget", {"target_node_policy": "budget", "max_target_nodes": 350})
    for stage in LATER:
        spore_xl.run_stage(run_b, stage)
    panel = read_lines(run_b.splits_dir / f"{run_b.name}_hvg_panel.txt")
    summary = json.loads((run_b.stats_dir / "hvg" / "all" / "summary.json").read_text(encoding="utf-8"))
    assert len(panel) == 500, len(panel)
    assert summary["target_node_policy"] == "budget" and summary["panel_stacked_above_n_top"] == 0
    split = json.loads((run_b.splits_dir / f"{run_b.name}_split_indices.json").read_text(encoding="utf-8"))
    var = set(pd.read_parquet(run_b.stats_dir / "hvg" / "all" / "hvg_stats.parquet").index)  # genes that survived Phase 6
    heldout = (set(split["val_labels"]) | set(split["test_labels"])) & var
    assert heldout <= set(panel), "held-out targets must always be nodes"
    assert summary["forced_heldout_targets"] == len(heldout)
    train_targets = (set(split["train_labels"]) & var) & set(panel)
    assert summary["train_targets_added"] <= 350 - len(heldout) and summary["train_targets_added"] > 0
    assert summary["target_nodes_in_panel"] <= 350 + summary["forced_protected_genes"] + 120, summary
    protected = json.loads((BUILD / "data" / "protected_synthetic_60k.json").read_text(encoding="utf-8"))
    wanted = {g for cat in protected["protected_genes"].values() for g in cat["genes"]} & var
    assert wanted <= set(panel), sorted(wanted - set(panel))[:5]
    failed = [name for name, ok, _ in verify.verify_run(run_b, write_report=False) if not ok]
    assert not failed, failed
    assert "GRAPH-READY" in (run_b.reports_dir / "COVERAGE_REPORT.md").read_text(encoding="utf-8")
    report = (run_b.reports_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "target_node_policy: budget" in report and "train_targets_added" in report, "policy and counts must be in the report"
    assert tree_digest(run_a) == before, "branching changed the original run"
    print(f"    policy budget (branch from hvg, max_target_nodes 350): panel exactly {len(panel)}; held-out {len(heldout)} all in; "
          f"{summary['train_targets_added']} of {summary['train_targets_candidates']} train targets added; verify PASS, GRAPH-READY; "
          f"original run untouched; policy and counts in REPORT.md")


def test_all_policy_rerun_in_a_branch_reproduces_the_original():
    run_a = base_run()
    run_c = make_branch(run_a, "again", {})
    spore_xl.run_stage(run_c, "hvg")
    assert read_lines(run_c.splits_dir / f"{run_c.name}_hvg_panel.txt") == read_lines(run_a.splits_dir / f"{run_a.name}_hvg_panel.txt")
    stats_a = pd.read_parquet(run_a.stats_dir / "hvg" / "all" / "hvg_stats.parquet")
    stats_c = pd.read_parquet(run_c.stats_dir / "hvg" / "all" / "hvg_stats.parquet")
    assert stats_a.equals(stats_c)
    print("    branch + policy all reproduces the original hvg panel and hvg_stats exactly")


def test_budget_without_max_target_nodes_uses_the_default_of_0_7_n_top():
    """v4 (P2): max_target_nodes defaults to 0.7 x n_top_genes (v3.0-era XL refused the missing key)."""
    run_a = base_run()
    run_d = make_branch(run_a, "default_nodes", {"target_node_policy": "budget"})
    spore_xl.run_stage(run_d, "hvg")
    summary = json.loads((run_d.stats_dir / "hvg" / "all" / "summary.json").read_text(encoding="utf-8"))
    assert summary["target_node_policy"] == "budget" and summary["max_target_nodes"] == int(0.7 * summary["n_top_genes"])


if __name__ == "__main__":
    failures = 0
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except Exception:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                failures += 1
                print(f"FAIL {name}")
    sys.exit(1 if failures else 0)
