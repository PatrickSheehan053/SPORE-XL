"""T2/T3 equivalence milestone: finalize + verify + compare at 1, 3 and 7 row shards.
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; python -B tests\test_equivalence.py   (XL_TEST_TMP set)
Gate: compare XL vs spore_baseline\run1 returns PASS; verify returns all PASS; merged finalize outputs equal SPORE's
merged substrate files (order included); the XL outputs are identical across 1, 3 and 7 shards."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import BUILD, get_run, run_all  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import compare_v4 as compare  # noqa: E402
import verify  # noqa: E402

SPORE_RUN = BUILD / "intermediate" / "spore_baseline" / "run1"
SIGNATURES = {}


def dense(matrix):
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def check(n_shards):
    run = get_run(n_shards, through="finalize")
    results = verify.verify_run(run, write_report=not (run.reports_dir / "VERIFICATION.md").exists())
    failed = [name for name, ok, _ in results if not ok]
    assert not failed, failed
    comparison = compare.compare_with_spore(run, SPORE_RUN)
    failed = [label for label, ok, _ in comparison if not ok]
    assert not failed, failed

    # merged substrate files (SPORE order) vs SPORE's
    for name in ("allsplits_metacell_ctrlpreserved", "train_hybrid"):
        ours = ad.read_h5ad(run.processed_dir / f"{run.name}_{name}.h5ad")
        reference = ad.read_h5ad(SPORE_RUN / "processed" / f"synth60k_{name}.h5ad")
        assert ours.n_obs == reference.n_obs and list(ours.var_names) == list(reference.var_names), name
        assert list(ours.obs["gene"].astype(str)) == list(reference.obs["gene"].astype(str)), name
        assert list(ours.obs["n_cells_in_metacell"]) == list(reference.obs["n_cells_in_metacell"]), name
        assert list(ours.obs["split"].astype(str)) == list(reference.obs["split"].astype(str)), name
        assert dense(ours.X).tobytes() == dense(reference.X).tobytes(), f"merged {name} X differs from SPORE's"
    for split in ("train", "val", "test"):
        ours = ad.read_h5ad(run.processed_dir / f"{run.name}_{split}_hvgraw.h5ad")
        reference = ad.read_h5ad(SPORE_RUN / "splits" / f"synth60k_{split}_hvgraw.h5ad")
        assert list(ours.obs_names) == list(reference.obs_names)
        assert dense(ours.X).tobytes() == dense(reference.X).tobytes()
    assert (run.processed_dir / f"{run.name}_allsplits_metacell_split_indices.json").exists()

    signature = compare.xl_signature(run)
    if SIGNATURES:
        differing = [key for key in signature if signature[key] != SIGNATURES["first"][key]]
        assert not differing, f"outputs differ between shard counts: {differing}"
    else:
        SIGNATURES["first"] = signature
    print(f"    {n_shards} shard(s): verify {len(results)}/{len(results)} PASS, compare {len(comparison)}/{len(comparison)} PASS "
          f"({[detail for label, _, detail in comparison if 'deviation' in label][0]}), merged files equal SPORE's, "
          f"signature of {len(signature)} outputs identical to the first shard count")


def test_equivalence(n_shards):
    check(n_shards)


if __name__ == "__main__":
    sys.exit(1 if run_all(test_equivalence) else 0)
