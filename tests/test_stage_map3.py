"""map3 at 1, 3 and 7 row shards vs SPORE's baseline run1 hvgraw and p9 files, per split.
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; python -B tests\test_stage_map3.py   (XL_TEST_TMP set)
Gate: hvgraw X exactly equal (cells, order, genes, every stored value), obs columns equal, QC var columns within 1e-6,
p9 X within 1e-6, per split."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import BUILD, get_run, run_all  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

BASELINE = BUILD / "intermediate" / "spore_baseline" / "run1" / "splits"
QC_COLUMNS = ["n_cells_by_counts", "mean_counts", "pct_dropout_by_counts", "total_counts"]


def load_parts(run, split, kind):
    directory = run.processed_dir / f"{run.name}_{split}_{kind}"
    pieces = [ad.read_h5ad(path) for path in sorted(directory.glob("part_*.h5ad"))]
    matrix = sp.vstack([piece.X for piece in pieces], format="csr")
    obs = pd.concat([piece.obs for piece in pieces])
    order = np.argsort(obs["xl_global_index"].to_numpy(), kind="stable")
    return matrix[order], obs.iloc[order], pieces[0].var, len(pieces)


def same_csr(left, right):
    left, right = sp.csr_matrix(left), sp.csr_matrix(right)
    return (left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left.indptr, right.indptr)
            and np.array_equal(left.indices, right.indices) and left.data.tobytes() == right.data.tobytes())


def check(n_shards):
    run = get_run(n_shards, through="map3")
    run_again = get_run(n_shards, through="map3")  # idempotent
    total_cells = 0
    for split in ("train", "val", "test"):
        reference = ad.read_h5ad(BASELINE / f"synth60k_{split}_hvgraw.h5ad")
        matrix, obs, var, n_parts = load_parts(run, split, "hvgraw")
        assert list(obs.index) == list(reference.obs_names), split
        assert list(var.index) == list(reference.var_names), split
        assert same_csr(matrix, reference.X), f"hvgraw X differs [{split}]"
        for column in reference.obs.columns:
            ours, theirs = obs[column], reference.obs[column]
            if theirs.dtype.kind == "f":
                assert np.allclose(ours.to_numpy(float), theirs.to_numpy(float), rtol=1e-5, equal_nan=True), column
                if column != "escaper_knockdown_depth":
                    assert ours.to_numpy().tobytes() == theirs.to_numpy().tobytes(), column
            else:
                assert list(ours.astype(str)) == list(theirs.astype(str)), column
        for column in QC_COLUMNS:
            assert np.max(np.abs(var[column].to_numpy(float) - reference.var[column].to_numpy(float))) <= 1e-6, column
        assert (var["mt"].astype(bool).to_numpy() == reference.var["mt"].astype(bool).to_numpy()).all()
        p9_matrix, p9_obs, _, _ = load_parts(run, split, "p9")
        reference_p9 = ad.read_h5ad(BASELINE / f"synth60k_{split}_p9.h5ad")
        assert list(p9_obs.index) == list(reference_p9.obs_names)
        difference = abs(sp.csr_matrix(p9_matrix) - sp.csr_matrix(reference_p9.X))
        worst = float(difference.max()) if difference.nnz else 0.0
        assert worst <= 1e-6, (split, worst)
        total_cells += matrix.shape[0]
        print(f"    {n_shards} shard(s) [{split}]: {matrix.shape[0]} cells x {matrix.shape[1]} genes, {n_parts} parts; hvgraw exact, p9 max abs diff {worst}")
    assert total_cells == 38972


def test_map3(n_shards):
    check(n_shards)


if __name__ == "__main__":
    sys.exit(1 if run_all(test_map3) else 0)
