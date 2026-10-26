"""CD4 intake (src/intake/intake_10x_flex_tar.py + src/intake/recipes/GSE314342_CD4T-Rest.yaml) on a synthetic 10x-Flex tar built with h5py.
 - only `_Rest_` members, sorted by name; read directly from the tar (nothing extracted)
 - every cell's written label == an INDEPENDENT plain-python assignment (min_umi 3, -N suffix removed, 1-Jun/2-Jun -> JUN BEFORE counting,
   NTC/CTRL -> control label, exactly one target); cells with 0 or 2+ targets are counted not written; X equals the member's gene counts
 - raw tally counts per target equal an independent tally; validate_intake_vs_tally.py PASSES on it (JUN and control pair formulas)
 - sorted categories identical across shards; design fields parsed from the member name
 - the whole XL pipeline then runs on the intake shards (recipe source): verify PASS, GRAPH-READY
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; set MPLBACKEND=Agg ; set XL_TEST_TMP=<scratch> ; python -B tests\\test_intake.py"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import BASE_CONFIG, BUILD, REPO, scratch_for  # noqa: E402
import make_flex_synthetic as flex  # noqa: E402
import validate_intake_vs_tally as validator  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import yaml  # noqa: E402

from src.intake import intake_10x_flex_tar as core  # noqa: E402
import cli as spore_xl  # noqa: E402
import stage_intake  # noqa: E402
import verify  # noqa: E402
from xl_common import Run  # noqa: E402

RECIPE = REPO / "src" / "intake" / "recipes" / "GSE314342_CD4T-Rest.yaml"
_STATE = {}


def build():
    if "run" in _STATE:
        return _STATE
    work = scratch_for("intake")
    members = [(flex.member_name(9393800 + i, 1 + i % 2, 9 + i, 1 + i, "Rest"), 2500 + 100 * i) for i in range(3)]
    members.append((flex.member_name(9393900, 1, 14, 1, "Stim8hr"), 1500))
    # shuffled order in the tar: the intake must sort by name
    order = [2, 0, 3, 1]
    truths = flex.build_tar(work / "h5", work / "x.tar", [members[i] for i in order])
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    run_dir = work / "run"
    cfg["paths"].update({"project_root": str(run_dir).replace("\\", "/"), "raw_h5ad": ""})
    cfg["phase8_hvg"].update({"protected_gene_list": None, "n_top_genes": 60})
    cfg["phase5_escaper_filtering"]["min_cells_per_perturbation"] = 10
    cfg["phase2_cell_triage"].update({"min_genes_per_cell": 10, "min_counts_per_cell": 20})  # the synthetic cells are tiny
    cfg["phase7_splits"].update({"test_n": 6, "val_n": 3})
    cfg["xl"] = {"source": {"kind": "recipe", "recipe": str(RECIPE).replace("\\", "/"), "tar": str(work / "x.tar").replace("\\", "/")},
                 "work_dir": (str(work / "external_work").replace(chr(92), "/") if __import__("os").environ.get("XL_TEST_EXTERNAL_WORKDIR") else "_xl_work"), "n_blocks": 4, "control_block_rows": 1000, "bucket_compression": "lzf", "memory_budget_gb": 24,
                 "merge_mode": "memory"}
    config = work / "run.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run = Run(config)
    spore_xl.run_stage(run, "intake_list")
    spore_xl.run_stage(run, "intake")
    _STATE.update(run=run, truths=truths, members=members, work=work)
    return _STATE


def test_members_sorted_and_only_rest():
    state = build()
    run = state["run"]
    table = stage_intake.members_table(run)
    names = list(table["name"])
    assert names == sorted(names) and len(names) == 3 and all("_Rest_" in n for n in names) and not any("Stim8hr" in n for n in names)
    assert table["cells"].tolist() == [2500, 2600, 2700]
    print(f"    {len(names)} Rest members found in the tar (Stim8hr excluded), in sorted name order")


def test_every_cell_matches_the_independent_assignment():
    state = build()
    run, truths = state["run"], state["truths"]
    recipe = core.load_recipe(RECIPE)
    members = stage_intake.members_table(run)
    categories_seen = set()
    tally_rows = []
    for index, name in enumerate(members["name"]):
        truth = truths[name]
        labels, raw_sets = flex.truth_for_member(truth, min_umi=3)
        shard = __import__("anndata").read_h5ad(run.work_dir / "intake" / f"shard_{index:04d}" / "shard.h5ad")
        meta = pd.read_json(run.stats_dir / "intake" / f"shard_{index:04d}" / "meta.json", typ="series")
        design = core.parse_member_name(name, recipe)
        sample = design["sample"]
        expected_cells = {f"{sample}_BC{ordering(state, name):02d}{cell:08d}-1": label for cell, label in labels.items() if label is not None}
        written = dict(zip(shard.obs_names, shard.obs["gene"].astype(str)))
        assert written == expected_cells, f"{name}: written cells/labels differ from the independent assignment"
        # not-written accounting
        n_none = sum(1 for label in labels.values() if label is None)
        assert shard.n_obs + n_none == len(labels) == int(meta["n_cells"])
        # JUN merge happened BEFORE counting: no 1-Jun / 2-Jun labels, JUN present; controls collapsed
        assert not ({"1-Jun", "2-Jun", "NTC", "CTRL"} & set(written.values())) and {"JUN", "non-targeting"} <= set(written.values())
        # raw tally counts
        one, any_count = {}, {}
        for cell, raw in raw_sets.items():
            for target in raw:
                any_count[target] = any_count.get(target, 0) + 1
                if len(raw) == 1:
                    one[target] = one.get(target, 0) + 1
        raw_table = pd.read_csv(run.stats_dir / "intake" / f"shard_{index:04d}" / "targets_raw.tsv", sep="\t").set_index("target_raw")
        assert {t: int(v) for t, v in raw_table["cells_one_target"].items() if v} == {t: v for t, v in one.items() if v}
        assert {t: int(v) for t, v in raw_table["cells_any"].items()} == any_count
        pair_jun = sum(1 for raw in raw_sets.values() if raw == {"1-Jun", "2-Jun"})
        pair_ctrl = sum(1 for raw in raw_sets.values() if raw == {"NTC", "CTRL"})
        assert meta["merged_pair_cells"] == {"JUN": pair_jun, "non-targeting": pair_ctrl}, (meta["merged_pair_cells"], pair_jun, pair_ctrl)
        for target in set(one) | set(any_count):
            tally_rows.append((name, target, one.get(target, 0), any_count.get(target, 0)))
        # X equals the member's gene counts for the written cells
        with h5py.File(state["work"] / "h5" / name, "r") as handle:
            group = handle["matrix"]
            matrix = sp.csr_matrix((group["data"][...], group["indices"][...], group["indptr"][...]), shape=(len(labels), int(group["shape"][0])))
        rows = [int(cell_name.split("-")[0][-8:]) for cell_name in shard.obs_names]
        assert (sp.csr_matrix(shard.X) != matrix[rows][:, :flex.N_GENES].astype(np.float32)).nnz == 0
        assert shard.X.dtype == np.float32 and shard.n_vars == flex.N_GENES and not any(v.startswith(("T0", "NTC", "CTRL")) for v in shard.var_names)
        # design fields and categories
        assert (shard.obs["replicate"] == design["replicate"]).all() and (shard.obs["lane"] == design["lane"]).all()
        assert (shard.obs["donor"] == design["donor"]).all() and (shard.obs["arm"] == "Rest").all()
        cats = list(shard.obs["gene"].cat.categories)
        assert cats == sorted(cats) and "JUN" in cats and "1-Jun" not in cats
        categories_seen.add(tuple(cats))
    assert len(categories_seen) == 1, "category order must be identical across shards"
    _STATE["tally_rows"] = tally_rows
    print(f"    {len(members)} members: written labels == independent assignment for every cell; raw tally counts, JUN/control pair "
          f"counters, X values, design fields and sorted categories all match")


def ordering(state, name):
    """the member's position in the tar (barcode prefix BCxx in make_flex_synthetic.build_tar)."""
    order = [2, 0, 3, 1]
    original = [m[0] for m in state["members"]]
    in_tar = [original[i] for i in order]
    return in_tar.index(name)


def test_validate_against_a_tally_and_detects_errors():
    state = build()
    test_every_cell_matches_the_independent_assignment()
    rows = state["tally_rows"]
    frame = pd.DataFrame(rows, columns=["member", "target", "one", "any"]).groupby("target")[["one", "any"]].sum().reset_index()
    tally = pd.DataFrame({"group": "Rest", "min_umi": 3, "target": frame["target"], "looks_like_control": frame["target"].isin(["NTC", "CTRL"]),
                          "cells_one_target": frame["one"], "cells_one_guide": frame["one"], "cells_any": frame["any"]})
    results = validator.validate(state["run"], tally)
    assert all(ok for _, ok, _ in results), results
    broken = tally.copy()
    broken.loc[broken["target"] == "1-Jun", "cells_one_target"] += 1
    results = validator.validate(state["run"], broken)
    assert not results[0][1], "a changed tally count must be detected"
    print("    validate_intake_vs_tally PASSES on the independent tally and FAILS when one count is changed")


def test_whole_pipeline_runs_on_the_recipe_source():
    state = build()
    run = state["run"]
    for stage in ["plan", "map1", "reduce1", "map2", "gather", "reduce2", "hvg", "map3", "metacell_prep", "metacell", "finalize"]:
        spore_xl.run_stage(run, stage)
    layout = run.layout()
    assert layout["kind"] == "recipe" and layout["n_obs"] == sum(
        int(pd.read_json(run.stats_dir / "intake" / f"shard_{i:04d}" / "meta.json", typ="series")["cells_written"]) for i in range(3))
    failed = [name for name, ok, _ in verify.verify_run(run, write_report=False) if not ok]
    assert not failed, failed
    assert "GRAPH-READY" in (run.reports_dir / "COVERAGE_REPORT.md").read_text(encoding="utf-8")
    source = run.source_info()
    assert source["members_in_order"] == sorted(source["members_in_order"]) and source["perturbation_category_order"] == sorted(source["perturbation_category_order"])
    print(f"    XL pipeline on the recipe source: {layout['n_obs']:,} cells, verify PASS, GRAPH-READY")


def test_placeholder_tar_and_bad_names_are_refused():
    state = build()
    cfg = yaml.safe_load(Path(state["run"].config_path).read_text(encoding="utf-8"))
    cfg["paths"]["project_root"] = str(scratch_for("intake_bad")).replace("\\", "/")
    cfg["xl"]["source"]["tar"] = "<TAR_PATH>"
    path = Path(cfg["paths"]["project_root"]).parent / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    try:
        spore_xl.run_stage(Run(path), "intake_list")
    except ValueError as error:
        assert "TAR_PATH" in str(error)
    else:
        raise AssertionError("placeholder tar must be refused")
    recipe = core.load_recipe(RECIPE)
    try:
        core.parse_member_name("GSM1_CD4i_weird_name.h5", recipe)
    except ValueError:
        return
    raise AssertionError("a member name outside the recipe pattern must be refused")


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
