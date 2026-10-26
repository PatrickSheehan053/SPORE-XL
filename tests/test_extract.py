"""T8 ladder: extract rungs c=16 and c=32 (and 64) on a 1,000-gene micro-panel from finished XL runs.
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; python -B tests\\test_extract.py   (XL_TEST_TMP set)
Gate: rungs nested (16 in 32 in 64), repeated extraction byte-identical, identical across shard counts, controls follow the
dataset's control fraction per split, splits fixed (no cell leaves its split), values equal the raw matrix, provenance complete."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import SOURCE, file_digest, get_run, scratch_for  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import extract  # noqa: E402

REQUIRED = ["run_dir", "run_name", "priority_function", "seed", "cells_per_perturbation", "control_policy", "genes", "perturbations",
            "n_cells", "n_perturbations", "per_split", "perturbations_with_fewer_cells_than_rung", "split_json_sha256",
            "control_assignment_sha256", "global_obs_final_sha256", "xl_environment", "x_dtype", "gene_space", "output_h5ad",
            "output_x_sha256"]


def micro_panel(run, n=1000, seed=3):
    var = run.var_table().index
    kept = run.var_table().loc[~var.str.upper().str.startswith(("RPL", "RPS"))].index
    rng = np.random.default_rng(seed)
    return [str(g) for g in rng.choice(np.array(kept), size=n, replace=False)]


def load(path):
    adata = ad.read_h5ad(path)
    return adata


def test_splitmix_matches_a_python_reference():
    def reference(value, seed):
        mask = (1 << 64) - 1
        z = (value + (seed + 1) * 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return z ^ (z >> 31)
    values = np.array([0, 1, 2, 12345, 59999, 11_000_000, 2**40 + 7], dtype=np.int64)
    for seed in (0, 7, 2026):
        assert [int(v) for v in extract.hash64(values, seed)] == [reference(int(v), seed) for v in values]


def test_ladder_nested_deterministic_and_fixed_split():
    run = get_run(3, through="finalize")
    genes = micro_panel(run)
    out = scratch_for("extract_a")
    rungs = {}
    for c in (16, 32, 64):
        h5ad_path, provenance_path = extract.extract(run, genes, c, seed=7, out_dir=out)
        rungs[c] = load(h5ad_path)
        provenance = pd.read_json(provenance_path, typ="series")
        assert not [key for key in REQUIRED if key not in provenance.index], [key for key in REQUIRED if key not in provenance.index]
    names = {c: set(rungs[c].obs["xl_global_index"]) for c in rungs}
    assert names[16] < names[32] < names[64], "rungs are not nested"

    final = pd.read_parquet(run.manifest_dir / "global_obs_final.parquet").set_index("xl_global_index")
    source = ad.read_h5ad(SOURCE)
    source_x = sp.csr_matrix(source.X)
    gene_index = [int(np.flatnonzero(source.var_names == g)[0]) for g in genes]
    for c, adata in rungs.items():
        obs = adata.obs
        # splits are the run's fixed splits
        assert (obs["split"].astype(str).to_numpy() == final.loc[obs["xl_global_index"], "split"].astype(str).to_numpy()).all()
        assert final.loc[obs["xl_global_index"], "survived_to_phase7"].all()
        # c lowest-priority cells per perturbation
        perts = obs[~obs["is_control"]]
        counts = perts["gene"].astype(str).value_counts()
        available = final[final["survived_to_phase7"] & (final["label"].astype(str) != run.ctrl_label)]["label"].astype(str).value_counts()
        assert (counts == available.reindex(counts.index).clip(upper=c)).all(), f"rung {c}: counts are not min(c, available)"
        expected = extract.priority(final.index.to_numpy(), 7)
        priority = pd.Series(expected, index=final.index)
        for label in list(counts.index[:15]):
            pool = final[(final["label"].astype(str) == label) & final["survived_to_phase7"]].index
            lowest = set(priority.loc[pool].sort_values(kind="stable").index[:c])
            assert set(perts.loc[perts["gene"].astype(str) == label, "xl_global_index"]) == lowest, label
        # controls follow the dataset's fraction per split
        for split in ("train", "val", "test"):
            n_pert = int((perts["split"].astype(str) == split).sum())
            n_ctrl = int(((obs["split"].astype(str) == split) & obs["is_control"]).sum())
            full = final[final["survived_to_phase7"]]
            ratio = ((full["label"].astype(str) == run.ctrl_label) & (full["split"].astype(str) == split)).sum() / (
                (full["label"].astype(str) != run.ctrl_label) & (full["split"].astype(str) == split)).sum()
            assert abs(n_ctrl - n_pert * ratio) <= 1.0, (c, split, n_ctrl, n_pert * ratio)
        # values equal the raw matrix
        rows = obs["xl_global_index"].to_numpy()
        reference = source_x[rows][:, gene_index]
        assert (sp.csr_matrix(adata.X) != reference).nnz == 0
        assert list(adata.var_names) == genes
    assert len(set(rungs[16].obs["split"].astype(str))) == 3
    print(f"    rungs 16/32/64: {[rungs[c].n_obs for c in (16, 32, 64)]} cells, nested; per perturbation = min(c, available); "
          f"lowest-priority property checked on 15 perturbations per rung; controls within 1 cell of the dataset ratio per split; "
          f"splits fixed; values equal the raw matrix")

    # repeated extraction: byte-identical
    out_b = scratch_for("extract_b")
    for c in (16, 32):
        again, again_prov = extract.extract(run, genes, c, seed=7, out_dir=out_b)
        first = out / f"{run.name}_c{c}_seed7.h5ad"
        assert file_digest(again) == file_digest(first), f"rung {c}: repeated extraction differs in content"
        assert again_prov.read_bytes() == (out / f"{run.name}_c{c}_seed7.provenance.json").read_bytes()
    # another seed selects different cells, same sizes
    other, _ = extract.extract(run, genes, 16, seed=8, out_dir=out_b)
    other_adata = load(other)
    assert set(other_adata.obs["xl_global_index"]) != names[16] and other_adata.n_obs == rungs[16].n_obs
    print("    repeated extraction identical (h5ad content and provenance bytes); a different seed changes the cells, not the sizes")


def test_ladder_is_identical_across_shard_counts():
    genes = micro_panel(get_run(3, through="finalize"))
    out = scratch_for("extract_shards")
    results = {}
    for n_shards in (1, 7):
        run = get_run(n_shards, through="finalize")
        path, _ = extract.extract(run, genes, 16, seed=7, out_dir=out, name=f"shards{n_shards}")
        adata = load(path)
        results[n_shards] = (list(adata.obs_names), sp.csr_matrix(adata.X).data.tobytes(), sp.csr_matrix(adata.X).indices.tobytes(),
                             list(adata.obs["split"].astype(str)))
    assert results[1] == results[7], "extraction depends on the shard layout"


def test_requested_gene_outside_the_gene_space_is_refused():
    run = get_run(3, through="finalize")
    try:
        extract.extract(run, ["GENE0001", "NOT_A_GENE_XYZ"], 16, seed=1, out_dir=scratch_for("extract_bad"))
    except ValueError as error:
        assert "not in the run's gene space" in str(error)
        return
    raise AssertionError("expected ValueError")


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
