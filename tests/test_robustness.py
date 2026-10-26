"""T4 / T4b robustness tests (XL only, plus an independent numpy re-derivation; the SPORE comparisons are in test_equivalence).
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; python -B tests\\test_robustness.py   (XL_TEST_TMP set)

 1 ordering variant: a shard with zero cells after QC, shards without controls, perturbations present in one shard only,
   a label that is not a gene (bypass), a target gene that is all-zero in controls; survivors equal the independent
   derivation and the outputs are identical to a 1-shard run
 2 label colliding with a QC var value is refused ("True", and a number equal to a gene's detection count)
 3 two labels aliasing one gene (duplicated alias) is refused by reduce2
 4 multi-target / combinatorial labels, combinatorial structure, and the activation direction are refused with a clear
   error before anything is written
 5 disk and memory pre-flight refuse before anything is written
 6 re-running every finished stage is a no-op (no file changed, none added)
 7 an h5ad_obs source gives outputs identical to h5ad_rows (and compares PASS against SPORE)
"""
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_support import (CONTROL, BUILD, finish, independent_survivors, load_source, run_through, small_subset,  # noqa: E402
                                variant_config, xl_survivors)
from xl_test_support import get_run  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import compare_v4 as compare  # noqa: E402
import cli as spore_xl  # noqa: E402
import verify  # noqa: E402
from xl_common import Run  # noqa: E402

SPORE_RUN = BUILD / "intermediate" / "spore_baseline" / "run1"
_STATE = {}


def raises(function, exception, text):
    try:
        function()
    except exception as error:
        assert text.lower() in str(error).lower(), f"expected '{text}' in: {error}"
        return str(error)
    raise AssertionError(f"expected {exception.__name__} containing '{text}'")


def snapshot(root):
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in Path(root).rglob("*") if path.is_file()}


# ------------------------------------------------------------------------------------------------ 1
def build_ordering_variant():
    if "ordering" in _STATE:
        return _STATE["ordering"]
    adata = small_subset(load_source(), 120)
    sizes = adata.obs["gene"].value_counts()
    perts = [label for label in sizes.index if label != CONTROL]
    bypass_from, zero_gene = perts[0], perts[3]
    labels = adata.obs["gene"].astype(str).to_numpy().copy()
    labels[labels == bypass_from] = "NOT_A_GENE1"
    adata.obs["gene"] = labels
    # the zero-histogram gene: every control cell has 0 counts of zero_gene
    column = int(np.flatnonzero(adata.var_names == zero_gene)[0])
    csc = sp.csc_matrix(adata.X)
    start, stop = csc.indptr[column], csc.indptr[column + 1]
    control_row = np.zeros(adata.n_obs, dtype=bool)
    control_row[labels == CONTROL] = True
    csc.data[start:stop][control_row[csc.indices[start:stop]]] = 0
    csc.eliminate_zeros()
    adata.X = sp.csr_matrix(csc)
    # order: perturbations sorted by label, controls last
    order = np.lexsort((np.arange(adata.n_obs), labels, (labels == CONTROL).astype(int)))
    adata = adata[order].copy()
    # 1,500 junk cells in front: nearly empty, they fail Phase 2 (a shard with zero cells after QC)
    rng = np.random.default_rng(5)
    donors = rng.choice(adata.n_obs, size=1500, replace=False)
    junk_x = sp.csr_matrix((np.ones(1500, dtype=np.float32), (np.arange(1500), rng.integers(0, adata.n_vars, 1500))),
                           shape=(1500, adata.n_vars))
    junk = ad.AnnData(X=junk_x, obs=adata.obs.iloc[donors].copy(), var=adata.var.copy())
    junk.obs_names = [f"junk_{i}" for i in range(1500)]
    combined = ad.concat([junk, adata], join="inner", merge="same")
    combined.obs["gene"] = combined.obs["gene"].astype(str)
    path, combined = finish(combined, "ordering")
    _STATE["ordering"] = (path, combined, zero_gene)
    return _STATE["ordering"]


def test_ordering_variant_invariance_and_edge_cases():
    path, adata, zero_gene = build_ordering_variant()
    expected = independent_survivors(adata)
    assert len(expected) > 5000
    run_a = run_through(variant_config("ord_many", path, shard_rows=1500))
    run_b = run_through(variant_config("ord_one", path, shard_rows=adata.n_obs + 1))
    for run in (run_a, run_b):
        assert xl_survivors(run, adata) == expected, "survivors differ from the independent derivation"
        failed = [name for name, ok, _ in verify.verify_run(run) if not ok]
        assert not failed, failed
    # edge cases really occurred in the many-shard run
    n_shards = run_a.n_shards()
    qc_cells = [ad.read_h5ad(run_a.work_dir / "map1" / f"shard_{i:04d}" / "qc.h5ad", backed="r").n_obs for i in range(n_shards)]
    no_controls = [i for i in range(n_shards)
                   if pd.read_json(run_a.stats_dir / "map1" / f"shard_{i:04d}" / "meta.json", typ="series")["n_controls_qc_pass"] == 0]
    assert qc_cells[0] == 0, "shard 0 should be empty after QC"
    assert len(no_controls) >= 5, no_controls
    one_shard = 0
    presence = {}
    for i in range(n_shards):
        for label in pd.read_parquet(run_a.stats_dir / "map1" / f"shard_{i:04d}" / "labelcounts.parquet")["label"]:
            presence.setdefault(label, set()).add(i)
    one_shard = sum(1 for label, shards in presence.items() if len(shards) == 1 and label != CONTROL)
    assert one_shard >= 20, one_shard
    counts = pd.concat([pd.read_parquet(run_a.stats_dir / "map2" / f"shard_{i:04d}" / "counts.parquet") for i in range(n_shards)]) \
        .groupby("label")[["n_total", "n_kept"]].sum()
    assert counts.loc["NOT_A_GENE1", "n_total"] == counts.loc["NOT_A_GENE1", "n_kept"] > 0, "bypassed label must keep all its cells"
    thresholds = pd.read_csv(run_a.stats_dir / "reduce1" / "all" / "escaper_thresholds.tsv", sep="\t").set_index("gene")
    assert float(thresholds.loc[zero_gene, "threshold"]) == 0.0, "all-zero control histogram must give threshold 0"
    assert compare.xl_signature(run_a) == compare.xl_signature(run_b), "outputs depend on the shard layout"
    print(f"    ordering variant: {adata.n_obs:,} cells, {n_shards} shards (shard 0 empty after QC, {len(no_controls)} without controls, "
          f"{one_shard} perturbations in one shard only); survivors == independent derivation ({len(expected):,}); "
          f"verify PASS; outputs identical to the 1-shard run")


# ------------------------------------------------------------------------------------------------ 2, 3
def small_variant(label, mutate):
    adata = small_subset(load_source(), 120)
    labels = adata.obs["gene"].astype(str).to_numpy().copy()
    var = adata.var.copy()
    labels, var = mutate(adata, labels, var)
    adata.obs["gene"] = labels
    adata.var = var
    return finish(adata, label)


def test_label_colliding_with_qc_value_is_refused():
    def to_true(adata, labels, var):
        labels[labels == adata.obs["gene"].value_counts().index[1]] = "True"
        return labels, var
    path, _ = small_variant("collide_true", to_true)
    run = Run(variant_config("collide_true", path, shard_rows=30000))
    message = raises(lambda: run_through(run.config_path, "reduce1"), ValueError, "collide")
    assert "True" in message

    def to_number(adata, labels, var):
        count = int(sp.csr_matrix(adata.X).getnnz(axis=0)[5])
        labels[labels == adata.obs["gene"].value_counts().index[1]] = str(count)
        return labels, var
    path, _ = small_variant("collide_number", to_number)
    raises(lambda: run_through(variant_config("collide_number", path, shard_rows=30000), "reduce1"), ValueError, "collide")


def test_duplicated_alias_label_is_refused_by_reduce2():
    def add_alias(adata, labels, var):
        big = adata.obs["gene"].value_counts().index[1]
        cells = np.flatnonzero(labels == big)
        labels[cells[: len(cells) // 2]] = big + "_ALT"
        var["alias"] = np.nan
        var["alias"] = var["alias"].astype(object)
        var.loc[big, "alias"] = big + "_ALT"
        return labels, var
    path, _ = small_variant("alias", add_alias)
    config = variant_config("alias", path, shard_rows=30000)
    message = raises(lambda: run_through(config, "reduce2"), ValueError, "same gene")
    print(f"    refused: {message[:110]}...")


# ------------------------------------------------------------------------------------------------ 4, 5
def test_unsupported_inputs_are_refused_before_anything_is_written():
    def two_target(adata, labels, var):
        small = adata.obs["gene"].value_counts().index[2]
        other = adata.obs["gene"].value_counts().index[3]
        cells = np.flatnonzero(labels == small)[:3]
        labels[cells] = f"{small}+{other}"
        return labels, var
    path, _ = small_variant("two_target", two_target)
    config = variant_config("two_target", path, shard_rows=30000)
    raises(lambda: run_through(config, "plan"), NotImplementedError, "multi-target")
    run = Run(config)
    assert not (run.manifest_dir / "shards.tsv").exists() and not (run.work_dir / "source_shards").exists()
    assert not list(run.manifest_dir.iterdir()), "plan wrote files before refusing"

    clean_path, _ = small_variant("clean", lambda adata, labels, var: (labels, var))
    raises(lambda: run_through(variant_config("crispra", clean_path, shard_rows=30000,
                                              overrides={"dataset": {"perturbation_type": "CRISPRa"}}), "plan"),
           NotImplementedError, "CRISPRa")
    raises(lambda: run_through(variant_config("comb", clean_path, shard_rows=30000,
                                              overrides={"dataset": {"perturbation_structure": "combinatorial"}}), "plan"),
           NotImplementedError, "combinatorial")
    raises(lambda: run_through(variant_config("direction", clean_path, shard_rows=30000,
                                              overrides={"phase5_escaper_filtering": {"direction": "activation"}}), "plan"),
           NotImplementedError, "activation")
    _STATE["clean"] = clean_path


def test_preflights_refuse_before_writing():
    if "clean" not in _STATE:
        _STATE["clean"], _ = small_variant("clean", lambda adata, labels, var: (labels, var))
    clean_path = _STATE["clean"]
    config = variant_config("tinymem", clean_path, shard_rows=30000, memory_budget_gb=0.001)
    message = raises(lambda: run_through(config, "plan"), MemoryError, "memory pre-flight")
    run = Run(config)
    assert not list(run.manifest_dir.iterdir()) and not (run.work_dir / "source_shards").exists()
    import preflight
    real = preflight.shutil.disk_usage
    preflight.shutil.disk_usage = lambda path: shutil._ntuple_diskusage(10**12, 10**12 - 10**6, 10**6)  # 1 MB free
    try:
        config = variant_config("nodisk", clean_path, shard_rows=30000)
        raises(lambda: run_through(config, "plan"), OSError, "disk pre-flight")
        run = Run(config)
        assert not list(run.manifest_dir.iterdir()) and not (run.work_dir / "source_shards").exists()
    finally:
        preflight.shutil.disk_usage = real


# ------------------------------------------------------------------------------------------------ 6
def test_rerunning_finished_stages_is_a_noop():
    run = get_run(3, through="finalize")
    before = snapshot(run.run_dir)
    for stage in ["plan", "map1", "reduce1", "map2", "gather", "reduce2", "hvg", "map3", "metacell_prep", "metacell", "finalize"]:
        spore_xl.run_stage(run, stage)
    after = snapshot(run.run_dir)
    assert set(before) == set(after), (set(after) ^ set(before))
    changed = [name for name in before if before[name] != after[name]]
    assert not changed, changed[:5]
    print(f"    re-ran all 11 stages: {len(before)} files, none changed, none added")


# ------------------------------------------------------------------------------------------------ 7
def test_h5ad_obs_source_equals_h5ad_rows():
    run_obs = get_run(2, through="finalize", kind="h5ad_obs")
    run_rows = get_run(3, through="finalize")
    assert run_obs.layout()["kind"] == "h5ad_obs"
    failed = [name for name, ok, _ in verify.verify_run(run_obs, write_report=False) if not ok]
    assert not failed, failed
    failed = [label for label, ok, _ in compare.compare_with_spore(run_obs, SPORE_RUN, write_report=False) if not ok]
    assert not failed, failed
    assert compare.xl_signature(run_obs) == compare.xl_signature(run_rows), "h5ad_obs output differs from h5ad_rows"
    print("    h5ad_obs (gem_group, 2 shards) == h5ad_rows (3 shards) on all 14 output signatures, verify PASS, compare vs SPORE PASS")


# ------------------------------------------------------------------------------------------------ 8
MIN10_SPORE = BUILD / "intermediate" / "spore_min10" / "run1"


def test_fallback_embedding_deviation_is_measured_against_real_spore():
    """SPORE v3.0 was run (2026-10-18) with min_cells_per_perturbation 10 so that groups under 50 cells exist; XL runs the same
    config. Everything outside the deviation must still be identical; the deviation itself is counted."""
    from robustness_support import SOURCE
    from xl_test_support import shard_rows_for
    config = variant_config("min10", SOURCE, shard_rows=shard_rows_for(3), test_n=50, val_n=25,
                            overrides={"phase5_escaper_filtering": {"min_cells_per_perturbation": 10}})
    run = run_through(config, "finalize")
    assert not [n for n, ok, _ in verify.verify_run(run) if not ok]
    results = compare.compare_with_spore(run, MIN10_SPORE, write_report=False)
    failed = [label for label, ok, _ in results if not ok]
    assert not failed, failed
    detail = [d for label, _, d in results if "deviation report" in label][0]
    groups = pd.concat([pd.read_parquet(p) for p in sorted((run.stats_dir / "metacell").glob("block_*/groups.parquet"))])
    small = groups[(groups["label"] != CONTROL) & (groups["n_cells"] < 50)]
    assert len(small) > 20 and int(small["used_fallback"].sum()) > 20, "the fallback path was not exercised"
    print(f"    {len(small)} groups under 50 cells (fallback used for {int(small['used_fallback'].sum())}); {detail}")


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
