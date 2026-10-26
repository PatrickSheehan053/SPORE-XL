"""metacell at 1, 3 and 7 row shards vs SPORE's baseline run1 allsplits metacell file.
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; python -B tests\test_stage_metacell.py   (XL_TEST_TMP set)
Gate: every unit of every group with 50+ cells (expression exact, n_cells_in_metacell, split, unit order) and every control
unit equal to SPORE's; the number of groups that used the fallback embedding (the allowed deviation) is reported."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import BUILD, get_run, run_all  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

BASELINE = BUILD / "intermediate" / "spore_baseline" / "run1" / "processed" / "synth60k_allsplits_metacell_ctrlpreserved.h5ad"
SPLIT_RANK = {"train": 0, "val": 1, "test": 2}


def xl_units(run):
    directory = run.processed_dir / f"{run.name}_allsplits_metacell_ctrlpreserved"
    pieces = [ad.read_h5ad(path) for path in sorted(directory.glob("part_*.h5ad"))]
    return sp.vstack([piece.X for piece in pieces], format="csr"), pd.concat([piece.obs for piece in pieces]), pieces[0].var


def check(n_shards):
    run = get_run(n_shards, through="metacell")
    matrix, obs, var = xl_units(run)
    reference = ad.read_h5ad(BASELINE)
    ref_x = reference.X.toarray() if sp.issparse(reference.X) else np.asarray(reference.X)
    ref_obs = reference.obs.copy()
    ref_obs["gene"] = ref_obs["gene"].astype(str)
    assert list(var.index) == list(reference.var_names)
    assert matrix.shape[0] == reference.n_obs, (matrix.shape[0], reference.n_obs)

    groups = pd.concat([pd.read_parquet(path) for path in sorted((run.stats_dir / "metacell").glob("block_*/groups.parquet"))])
    big = groups[(groups["label"] != run.ctrl_label)]
    n_fallback = int(big["used_fallback"].sum())
    n_small = int((big["n_cells"] < 50).sum())
    n_compared_groups = n_compared_units = 0
    for label, group in obs[obs["gene"].astype(str) != run.ctrl_label].groupby(obs["gene"].astype(str), sort=True):
        ordered = group.sort_values("xl_unit")
        ours = matrix[[obs.index.get_loc(name) for name in ordered.index]].toarray()
        wanted = np.flatnonzero(ref_obs["gene"].to_numpy() == label)
        assert len(wanted) == len(ordered), label
        n_cells_group = int(ordered["n_cells_in_metacell"].sum())
        if n_cells_group >= 50:
            assert ours.tobytes() == ref_x[wanted].tobytes(), f"units differ for group {label}"
            assert list(ordered["n_cells_in_metacell"]) == list(ref_obs["n_cells_in_metacell"].to_numpy()[wanted])
            assert list(ordered["split"].astype(str)) == list(ref_obs["split"].astype(str).to_numpy()[wanted])
            n_compared_groups += 1
            n_compared_units += len(ordered)
    assert n_compared_groups == int((big["n_cells"] >= 50).sum())

    controls = obs[obs["gene"].astype(str) == run.ctrl_label].copy()
    controls["rank"] = controls["xl_control_split"].astype(str).map(SPLIT_RANK)
    controls = controls.sort_values(["rank", "xl_global_index"])
    ours = matrix[[obs.index.get_loc(name) for name in controls.index]].toarray()
    wanted = np.flatnonzero(ref_obs["gene"].to_numpy() == run.ctrl_label)
    assert len(wanted) == len(controls)
    assert ours.tobytes() == ref_x[wanted].tobytes(), "control units differ"
    assert (controls["n_cells_in_metacell"] == 1).all() and (controls["split"].astype(str) == "control").all()
    print(f"    {n_shards} shard(s): {matrix.shape[0]} units; {n_compared_groups} groups with 50+ cells ({n_compared_units} units) "
          f"and {len(controls)} control units equal SPORE's; groups under 50 cells: {n_small}; fallback embedding used for {n_fallback}")


def test_metacell(n_shards):
    check(n_shards)


if __name__ == "__main__":
    sys.exit(1 if run_all(test_metacell) else 0)
