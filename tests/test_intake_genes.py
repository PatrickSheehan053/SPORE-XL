"""ONE gene list for a multi-member intake (the CD4 failure of 22 Oct: 30 of 94 shards stored the same 18,130 genes in another order and the gate stopped at
"gene list differs from the first shard"). Synthetic Flex members (tests/make_flex_synthetic.py) with other gene orders / fewer genes:
 1 same gene SET, other order (case a): intake_list picks the first member's order as canonical, writes it to manifest/intake_genes.tsv, the per-member
   fingerprints and statuses to reports/intake_genes_members.tsv; every shard is written already in that order; each shard's counts per GENE ID equal the
   member file's (lossless reorder); the end-of-intake assertion passes; gate_protected and `plan` accept the shards
 2 different sets within the allowance: the INTERSECTION is kept (no zeros invented), the dropped genes and the members they are missing from are in
   reports/intake_genes_dropped.tsv, the shards carry exactly the intersection
 3 a set difference above the allowance (default 2 percent) is REFUSED by intake_list (report written first); the recipe key max_dropped_gene_fraction
   lifts the refusal explicitly
 4 the end-of-intake assertion names the offending shard when a shard no longer has the run's gene list
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; set XL_TEST_TMP=<scratch> ; set XL_TEST_DATA=<reference folder> ; python -B tests\\test_intake_genes.py"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import BASE_CONFIG, REPO, scratch_for  # noqa: E402
import make_flex_synthetic as flex  # noqa: E402

import anndata as ad  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import yaml  # noqa: E402

import cli as spore_xl  # noqa: E402
import stage_intake  # noqa: E402
from xl_common import Run  # noqa: E402

RECIPE = REPO / "src" / "intake" / "recipes" / "GSE314342_CD4T-Rest.yaml"
N = flex.N_GENES


def make_run(label, orders, recipe_changes=None):
    """A run on a tar whose members (3 Rest + 1 Stim8hr) have the gene orders in `orders` ({member position: gene_order})."""
    work = scratch_for(label)
    members = [(flex.member_name(9393800 + i, 1 + i % 2, 9 + i, 1 + i, "Rest"), 2000 + 100 * i) for i in range(3)]
    members.append((flex.member_name(9393900, 1, 14, 1, "Stim8hr"), 1500))
    flex.build_tar(work / "h5", work / "x.tar", members, gene_orders=orders)
    recipe_path = RECIPE
    if recipe_changes:
        recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
        recipe.update(recipe_changes)
        recipe_path = work / "recipe.yaml"
        recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["paths"].update({"project_root": str(work / "run").replace("\\", "/"), "raw_h5ad": ""})
    cfg["phase8_hvg"].update({"protected_gene_list": None, "n_top_genes": 60})
    cfg["phase5_escaper_filtering"]["min_cells_per_perturbation"] = 10
    cfg["phase2_cell_triage"].update({"min_genes_per_cell": 10, "min_counts_per_cell": 20})
    cfg["phase7_splits"].update({"test_n": 6, "val_n": 3})
    cfg["xl"] = {"source": {"kind": "recipe", "recipe": str(recipe_path).replace("\\", "/"), "tar": str(work / "x.tar").replace("\\", "/")},
                 "work_dir": "_xl_work", "n_blocks": 4, "control_block_rows": 1000, "bucket_compression": "lzf", "memory_budget_gb": 24, "merge_mode": "memory"}
    config = work / "run.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return work, members, Run(config)


def shard_matrix_by_id(run, index):
    shard = ad.read_h5ad(run.work_dir / "intake" / f"shard_{index:04d}" / "shard.h5ad")
    return shard, sp.csr_matrix(shard.X)


def member_counts_by_gene(path, cells, gene_id):
    """Counts of one gene id for the given cells, read from the member file itself (independent of the intake code)."""
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        ids = [value.decode() for value in group["features"]["id"][...]]
        row = ids.index(gene_id)
        indptr, indices, data = group["indptr"][...], group["indices"][...], group["data"][...]
        values = []
        for cell in cells:
            start, stop = indptr[cell], indptr[cell + 1]
            hit = np.flatnonzero(indices[start:stop] == row)
            values.append(float(data[start + hit[0]]) if len(hit) else 0.0)
        return np.array(values)


def test_same_set_other_order_is_reordered_losslessly():
    rng = np.random.default_rng(7)
    orders = {1: rng.permutation(N), 2: rng.permutation(N)}     # members 1 and 2 (sorted positions) store the 200 genes in another order
    work, members, run = make_run("genes_reorder", orders)
    spore_xl.run_stage(run, "intake_list")
    paths = stage_intake.genes_manifest_paths(run)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["policy"] == "reorder" and summary["n_genes"] == N and summary["n_dropped"] == 0
    report = pd.read_csv(paths["members"], sep="\t").set_index("member")
    names = sorted(name for name, _ in members if "_Rest_" in name)
    assert report.loc[names[0], "status"] == "same_as_canonical" and set(report["status"]) == {"same_as_canonical", "reordered"}
    assert report["sorted_sha256"].nunique() == 1 and report["order_sha256"].nunique() == 3, "same set, three orders (the 22 Oct fingerprints)"
    canonical = stage_intake.canonical_genes(run)
    assert canonical["ids"] == [f"ENSG{index:011d}" for index in range(N)], "canonical order = the first member (sorted by name)"
    spore_xl.run_stage(run, "intake")                      # includes the end-of-intake uniform-gene-list assertion
    assert stage_intake.assert_uniform_var(run) == 3
    table = stage_intake.members_table(run)
    for index, name in enumerate(table["name"]):
        shard, matrix = shard_matrix_by_id(run, index)
        assert list(shard.var_names) == canonical["var_names"] and list(shard.var["gene_ids"]) == canonical["ids"]
        cells = [int(cell.split("-")[0][-8:]) for cell in shard.obs_names]
        for position in (0, 5, 77, 123, N - 1):          # per GENE ID, not per position: the reorder must not move a count to another gene
            expected = member_counts_by_gene(work / "h5" / name, cells, canonical["ids"][position])
            assert np.array_equal(matrix[:, position].toarray().ravel(), expected), (name, position)
        meta = json.loads((run.stats_dir / "intake" / f"shard_{index:04d}" / "meta.json").read_text(encoding="utf-8"))
        assert meta["gene_order_status"] == report.loc[name, "status"] and meta["n_genes_written"] == N
    assert len({json.loads((run.stats_dir / "intake" / f"shard_{i:04d}" / "meta.json").read_text(encoding="utf-8"))["features_sha256"] for i in range(3)}) == 1
    import gate_protected
    gene_names = canonical["names"]
    protected = work / "protected.json"
    protected.write_text(json.dumps({"metadata": {"version": "1.0", "name": "synthetic"}, "protected_genes": {"identity": {"tier": 1, "genes": gene_names[:3]}}}), encoding="utf-8")
    assert gate_protected.main(["--config", str(run.config_path), "--protected", str(protected), "--out-json", str(work / "v11.json"), "--out-tsv", str(work / "gate.tsv")]) == 0
    spore_xl.run_stage(run, "plan")
    assert (run.manifest_dir / "plan.done").exists()
    print(f"    reorder: 3 members, 3 gene orders, 1 gene set: canonical list from the first member, shards identical in var, counts equal per gene id; gate and plan accept")


def test_intersection_within_the_allowance_reports_the_dropped_genes():
    first_three_missing = [index for index in range(N) if index not in (10, 11, 12)]          # member 1 never measured 3 genes (1.5 percent of 200)
    work, members, run = make_run("genes_intersection", {1: first_three_missing})
    spore_xl.run_stage(run, "intake_list")
    paths = stage_intake.genes_manifest_paths(run)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["policy"] == "intersection" and summary["n_genes"] == N - 3 and summary["n_dropped"] == 3 and summary["n_union"] == N
    dropped = pd.read_csv(paths["dropped"], sep="\t")
    names = sorted(name for name, _ in members if "_Rest_" in name)
    assert sorted(dropped["gene_id"]) == [f"ENSG{index:011d}" for index in (10, 11, 12)]
    assert (dropped["n_members_with"] == 2).all() and (dropped["members_without"] == names[1]).all() and dropped["members_with"].str.contains(names[0]).all()
    spore_xl.run_stage(run, "intake")
    canonical = stage_intake.canonical_genes(run)
    assert len(canonical["ids"]) == N - 3 and not ({f"ENSG{index:011d}" for index in (10, 11, 12)} & set(canonical["ids"]))
    for index in range(3):
        shard, _ = shard_matrix_by_id(run, index)
        assert shard.n_vars == N - 3 and list(shard.var_names) == canonical["var_names"]
    meta = json.loads((run.stats_dir / "intake" / "shard_0000" / "meta.json").read_text(encoding="utf-8"))
    assert meta["gene_order_status"] == "dropped_genes" and meta["n_gene_features"] == N and meta["n_genes_written"] == N - 3
    print("    intersection: 3 genes missing from one member are dropped everywhere (no zeros invented) and listed with the members they are missing from")


def test_set_difference_above_the_allowance_is_refused_unless_the_recipe_allows_it():
    many_missing = [index for index in range(N) if index >= 10]                                  # member 1 has only 190 of 200 genes: 5 percent
    work, members, run = make_run("genes_refused", {1: many_missing})
    try:
        spore_xl.run_stage(run, "intake_list")
    except ValueError as error:
        assert "max_dropped_gene_fraction" in str(error) and "5.0%" in str(error), str(error)
    else:
        raise AssertionError("a 5 percent gene-set difference must be refused with the default allowance of 2 percent")
    paths = stage_intake.genes_manifest_paths(run)
    assert paths["members"].exists() and paths["dropped"].exists() and not paths["genes"].exists(), "the report is written before the refusal; no gene list, no intake_list.done"
    assert not (run.manifest_dir / "intake_list.done").exists()
    _, _, allowed = make_run("genes_allowed", {1: many_missing}, recipe_changes={"max_dropped_gene_fraction": 0.10})
    spore_xl.run_stage(allowed, "intake_list")
    assert stage_intake.canonical_genes(allowed)["policy"] == "intersection" and len(stage_intake.canonical_genes(allowed)["ids"]) == N - 10
    print("    allowance: 5 percent refused by default (report written), accepted with max_dropped_gene_fraction 0.10")


def test_end_of_intake_assertion_names_the_offending_shard():
    work, members, run = make_run("genes_assert", {})
    spore_xl.run_stage(run, "intake_list")
    spore_xl.run_stage(run, "intake")
    shard_path = run.work_dir / "intake" / "shard_0001" / "shard.h5ad"
    shard = ad.read_h5ad(shard_path)
    renamed = list(shard.var_names)
    renamed[0], renamed[1] = renamed[1], renamed[0]
    shard.var_names = renamed
    shard.write_h5ad(shard_path)
    try:
        stage_intake.assert_uniform_var(run)
    except RuntimeError as error:
        assert "shard_0001" in str(error) and "ONE gene list" in str(error), str(error)
    else:
        raise AssertionError("a shard with another gene order must be reported")
    print("    assertion: a shard with two genes swapped is reported by name")


if __name__ == "__main__":
    test_same_set_other_order_is_reordered_losslessly()
    test_intersection_within_the_allowance_reports_the_dropped_genes()
    test_set_difference_above_the_allowance_is_refused_unless_the_recipe_allows_it()
    test_end_of_intake_assertion_names_the_offending_shard()
    print("PASS test_intake_genes")
