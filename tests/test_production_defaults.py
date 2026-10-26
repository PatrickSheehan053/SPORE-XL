"""v4.0 production defaults (Amendment 2 P1-P6, P8) on the synthetic 60k, against the compat run that reproduces SPORE v3.0.
  P2  budget policy: panel == n_top_genes exactly, max_target_nodes = 0.7 x n_top, node / row-only manifest
  P3  no cell-cycle (ghost) rescue
  P4  escaper score counted in the report, rule 1 still the one applied (so the survivors equal the compat run)
  P5  confounding table, variance explained, cell-cycle diagnostic in the report
  P6  train_hybrid_lognorm written and checked by verify
  P1  content names (_norm) with the v3.0 names (_p9) as aliases; phase map in the report
  P8  stable HVG tie-break independent of PYTHONHASHSEED
Run: set PYTHONHASHSEED=0 ; python -B tests\\test_production_defaults.py   (XL_TEST_TMP, XL_TEST_DATA set; about 4 minutes)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import BASE_CONFIG, SOURCE, get_run, scratch_for, shard_rows_for  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import cli as spore_xl  # noqa: E402
import verify  # noqa: E402
from xl_common import Run  # noqa: E402

_STATE = {}


def production_run():
    if "run" in _STATE:
        return _STATE["run"]
    run_dir = scratch_for("production")
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["paths"]["project_root"] = str(run_dir).replace("\\", "/")
    cfg["paths"]["raw_h5ad"] = ""
    cfg["xl"] = {"source": {"kind": "h5ad_rows", "h5ad": str(SOURCE).replace("\\", "/"), "shard_rows": shard_rows_for(3)}, "work_dir": "_xl_work",
                 "n_blocks": 8, "control_block_rows": 1000, "hvg_subsample_cells": 100000, "bucket_compression": "lzf", "memory_budget_gb": 24}
    config_path = run_dir.parent / f"{run_dir.name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run = Run(config_path)
    assert run.mode == "production"
    for stage in ["plan", "map1", "reduce1", "map2", "gather", "reduce2", "hvg", "map3", "metacell_prep", "metacell", "finalize", "report"]:
        spore_xl.run_stage(run, stage)
    _STATE["run"] = run
    return run


def test_budget_panel_and_node_manifest():
    run = production_run()
    summary = json.loads((run.stats_dir / "hvg" / "all" / "summary.json").read_text(encoding="utf-8"))
    assert summary["target_node_policy"] == "budget" and summary["max_target_nodes"] == 350 and summary["n_top_genes"] == 500
    panel = [line for line in (run.splits_dir / f"{run.name}_hvg_panel.txt").read_text(encoding="utf-8").split("\n") if line]
    assert len(panel) == 500, len(panel)
    nodes = pd.read_csv(run.manifest_dir / "perturbation_nodes.tsv", sep="\t", keep_default_na=False)
    final = pd.read_parquet(run.manifest_dir / "global_obs_final.parquet")
    surviving = set(final.loc[final["survived_to_phase7"], "label"].astype(str)) - {run.ctrl_label}
    assert set(nodes["label"]) == surviving, "every surviving perturbation is in the node manifest"
    assert set(nodes.loc[nodes["is_node"].astype(str) == "True", "label"]) == set(nodes["label"]) & set(panel)
    assert nodes.loc[nodes["reason"] == "heldout_forced", "is_node"].astype(str).eq("True").all()
    print(f"    panel {len(panel)}; nodes {int((nodes['is_node'].astype(str) == 'True').sum())}, row-only {int((nodes['is_node'].astype(str) != 'True').sum())}; reasons {nodes['reason'].value_counts().to_dict()}")


def test_no_ghost_rescue():
    run = production_run()
    summary = json.loads((run.stats_dir / "hvg" / "all" / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_ghosts_stacked"] == 0 and summary["forced_cell_cycle_genes"] == 0
    stats = pd.read_parquet(run.stats_dir / "hvg" / "all" / "hvg_stats.parquet")
    assert not stats["rescued_ghost"].any()


def test_outputs_equal_compat_run_except_panel_policy():
    run = production_run()
    compat = get_run(3, through="finalize")
    assert compat.mode == "compat-v3"
    assert (run.splits_dir / f"{run.name}_split_indices.json").read_bytes() == (compat.splits_dir / f"{compat.name}_split_indices.json").read_bytes()
    left = pd.read_parquet(run.manifest_dir / "global_obs_final.parquet")
    right = pd.read_parquet(compat.manifest_dir / "global_obs_final.parquet")
    assert left["survived_to_phase7"].equals(right["survived_to_phase7"]) and left["split"].astype(str).equals(right["split"].astype(str))
    assert (run.stats_dir / "reduce2" / "all" / "gene_keep_mask.txt").read_bytes() == (compat.stats_dir / "reduce2" / "all" / "gene_keep_mask.txt").read_bytes()
    print("    split JSON, survivors, split of every cell and the Phase 6 gene mask are identical to the compat run")


def test_escaper_rules_counted_not_applied():
    run = production_run()
    rules = pd.read_csv(run.reports_dir / "escaper_rules.tsv", sep="\t")
    escapers = int((rules["n_total"] - (rules["n_total"] - rules["n_removed"])).sum())
    assert int(rules["n_removed"].sum()) == int(rules["n_rule1"].sum()), "report mode removes exactly the rule-1 cells"
    usable = rules[rules["n_rule2"] >= 0]
    assert len(usable) > 100 and (usable["n_both"] <= np.minimum(usable["n_rule1"], usable["n_rule2"])).all()
    report = (run.reports_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "Escaper rules" in report and "rule 2" in report
    print(f"    rule 1 flags {int(rules['n_rule1'].sum()):,} cells, rule 2 flags {int(usable['n_rule2'].sum()):,} (both {int(usable['n_both'].sum()):,}) over {len(usable)} perturbations")


def test_diagnostics_in_report():
    run = production_run()
    report = (run.reports_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "Perturbation x batch confounding" in report and "Variance explained on the HVG subsample" in report and "Cell-cycle score diagnostic" in report
    diag = json.loads((run.stats_dir / "hvg" / "all" / "diagnostics.json").read_text(encoding="utf-8"))
    assert diag["variance_explained"]["batch"]["groups"] == 2 and 0 <= diag["variance_explained"]["batch"]["r2"] < 1
    assert diag["cell_cycle"]["status"] == "ok", diag["cell_cycle"]
    summary = json.loads((run.stats_dir / "reduce2" / "all" / "confounding_summary.json").read_text(encoding="utf-8"))
    assert summary["n_batches"] == 2 and summary["n_perturbations"] > 100
    print(f"    batch R2 {diag['variance_explained']['batch']['r2']}, perturbation adjusted R2 {diag['variance_explained']['perturbation']['r2_adjusted']}; "
          f"{summary['n_flagged']} of {summary['n_perturbations']} perturbations flagged for batch confounding")


def test_names_aliases_lognorm_and_verify():
    run = production_run()
    for split in ("train", "val", "test"):
        norm_dir, alias_dir = run.processed_dir / f"{run.name}_{split}_norm", run.processed_dir / f"{run.name}_{split}_p9"
        assert norm_dir.is_dir() and alias_dir.is_dir()
        for part in norm_dir.glob("part_*.h5ad"):
            assert (alias_dir / part.name).exists() and os.path.samefile(part, alias_dir / part.name) or (alias_dir / part.name).is_symlink()
    assert (run.processed_dir / f"{run.name}_train_hybrid_lognorm").is_dir()
    hybrid = ad.read_h5ad(sorted((run.processed_dir / f"{run.name}_train_hybrid_lognorm").glob("part_*.h5ad"))[0])
    assert float(np.expm1(hybrid.X.toarray() if hasattr(hybrid.X, "toarray") else hybrid.X).sum(axis=1).max()) < 10000 * 1.01
    figures = sorted(path.name for path in (run.run_dir / "figures").glob("*.png"))
    assert "cell_qc_violin.png" in figures and "p02_qc_violin.png" in figures, figures
    report = (run.reports_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "Phase numbering" in report and "Cell QC" in report
    results = verify.verify_run(run, write_report=False)
    failed = [name for name, ok, _ in results if not ok]
    assert not failed, failed
    assert any("lognorm" in name for name, _, _ in results) and any("aliases" in name for name, _, _ in results)
    print(f"    figures {len(figures)} (content names + v3.0 aliases); verify {len(results)}/{len(results)} PASS")


def test_stable_tie_break_does_not_depend_on_hash_seed():
    script = scratch_for("tiebreak") / "ties.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\nsys.dont_write_bytecode = True\n"
        f"sys.path.insert(0, r'{Path(__file__).resolve().parent.parent / 'src' / 'engine'}')\n"
        "import paths\npaths.add_spore_to_path()\nimport pandas as pd\nfrom adapted_phases.phase08_hvg import _lowest\n"
        "names = [f'G{i:03d}' for i in range(60)]\n"
        "stats = pd.DataFrame({'variances_norm': [1.0] * 40 + [2.0] * 20}, index=names)\n"
        "print(','.join(_lowest(stats, 'variances_norm', set(names), 7, sys.argv[1])))\n", encoding="utf-8")
    outputs = {}
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
        outputs[seed] = subprocess.run([sys.executable, "-B", str(script), "stable"], capture_output=True, text=True, env=env, check=True).stdout.strip()
    assert len(set(outputs.values())) == 1, outputs
    assert outputs["0"] == "G000,G001,G002,G003,G004,G005,G006"
    print("    stable tie-break: identical under PYTHONHASHSEED 0, 1, 12345 (ties broken by gene name)")


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
