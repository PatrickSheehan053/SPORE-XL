"""P4 escaper_score modes on a 60-perturbation subset of the synthetic 60k (production defaults otherwise):
  off      no carried cells, no rule counts (what compat mode does)
  report   rule 2 counted, rule 1 applied: survivors equal the off run
  union    cells flagged by rule 1 OR rule 2 are removed
  replace  rule 2 decides where it could be computed, rule 1 elsewhere
Every mode passes verify and the waterfall reconciles. Run: python -B tests\\test_escaper_modes.py   (XL_TEST_TMP, XL_TEST_DATA)"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_support import finish, load_source, run_through, small_subset, variant_config  # noqa: E402

import pandas as pd  # noqa: E402

import verify  # noqa: E402

_STATE = {}


def subset_path():
    if "path" not in _STATE:
        _STATE["path"], _ = finish(small_subset(load_source(), 60), "escmodes")
    return _STATE["path"]


def run_mode(mode):
    if mode not in _STATE:
        overrides = {"phase5_escaper_filtering": {"escaper_score": mode}, "xl": {"compat": None, "merge_split_outputs": "never"}}
        config = variant_config(f"esc_{mode}", subset_path(), shard_rows=7000, test_n=10, val_n=5, n_blocks=6, overrides=overrides)
        run = run_through(config, "finalize")
        results = verify.verify_run(run, write_report=False)
        failed = [name for name, ok, _ in results if not ok]
        assert not failed, (mode, failed)
        waterfall = (run.reports_dir / "WATERFALL.md").read_text(encoding="utf-8")
        assert "RECONCILES" in waterfall and "DOES NOT" not in waterfall, (mode, waterfall)
        _STATE[mode] = run
    return _STATE[mode]


def rule_table(run):
    frames = [pd.read_parquet(path) for path in sorted((run.stats_dir / "gather").glob("block_*/rule_counts.parquet"))]
    return pd.concat(frames).set_index("label")


def survivors(run):
    final = pd.read_parquet(run.manifest_dir / "global_obs_final.parquet")
    return set(final.loc[final["survived_to_phase7"], "xl_global_index"])


def test_off_carries_nothing():
    run = run_mode("off")
    assert not list((run.stats_dir / "gather").glob("block_*/rule_counts.parquet"))
    assert run.escaper_mode == "off"


def test_report_mode_applies_rule_1_only():
    off, report = run_mode("off"), run_mode("report")
    assert survivors(off) == survivors(report)
    table = rule_table(report)
    assert (table["n_removed"] == table["n_rule1"]).all() and (table["n_rule2"] >= -1).all()
    assert int(table["n_rule2"].clip(lower=0).sum()) > 0, "rule 2 flagged nothing: suspicious"


def test_union_and_replace_follow_their_definitions():
    report, union, replace = run_mode("report"), run_mode("union"), run_mode("replace")
    table_report, table_union, table_replace = rule_table(report), rule_table(union), rule_table(replace)
    assert table_union.index.equals(table_report.index)
    usable = table_union["n_rule2"] >= 0
    expected_union = (table_union["n_rule1"] + table_union["n_rule2"] - table_union["n_both"]).where(usable, table_union["n_rule1"])
    assert (table_union["n_removed"] == expected_union).all()
    expected_replace = table_replace["n_rule2"].where(table_replace["n_rule2"] >= 0, table_replace["n_rule1"])
    assert (table_replace["n_removed"] == expected_replace).all()
    assert survivors(union) <= survivors(report), "union can only remove more cells than rule 1"
    print(f"    removed cells: report {int(table_report['n_removed'].sum()):,}, union {int(table_union['n_removed'].sum()):,}, replace {int(table_replace['n_removed'].sum()):,}")


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
