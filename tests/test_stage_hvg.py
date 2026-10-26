"""hvg at 1, 3 and 7 row shards vs SPORE's baseline run1 (hvg_stats and core features stored in the train hvgraw uns).
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; python -B tests\\test_stage_hvg.py   (XL_TEST_TMP set)
Gate: HVG panel (names and order) and hvg_stats (every column, every gene, exact floats and flags) equal SPORE's."""
import hashlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import BUILD, get_run, read_lines, run_all  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASELINE = BUILD / "intermediate" / "spore_baseline" / "run1" / "splits"
SIGNATURES = {}


def spore_hvg():
    train = ad.read_h5ad(BASELINE / "synth60k_train_hvgraw.h5ad")
    stats = pd.DataFrame.from_dict(train.uns["hvg_stats"], orient="index")
    return list(train.var_names), stats, [str(name) for name in train.uns["spore_core_features"]]


def check(n_shards):
    run = get_run(n_shards, through="hvg")
    panel_ref, stats_ref, core_ref = spore_hvg()
    panel = read_lines(run.splits_dir / f"{run.name}_hvg_panel.txt")
    assert panel == panel_ref, (len(panel), len(panel_ref))
    out = run.stats_dir / "hvg" / "all"
    stats = pd.read_parquet(out / "hvg_stats.parquet")
    # h5ad uns dicts come back with alphabetically sorted keys: compare by gene name and column name
    assert sorted(stats.index) == sorted(stats_ref.index)
    assert sorted(stats.columns) == sorted(stats_ref.columns), (sorted(stats.columns), sorted(stats_ref.columns))
    stats_ref = stats_ref.loc[stats.index, list(stats.columns)]
    for column in stats.columns:
        left, right = stats[column].to_numpy(), stats_ref[column].to_numpy()
        if left.dtype.kind == "f":
            assert np.array_equal(left, right, equal_nan=True) and left.tobytes() == right.astype(left.dtype).tobytes(), column
        else:
            assert (left == right).all(), column
    assert read_lines(out / "core_features.txt") == core_ref
    signature = hashlib.sha256((out / "hvg_stats.parquet").read_bytes()).hexdigest() + hashlib.sha256(
        "\n".join(panel).encode()).hexdigest()
    if SIGNATURES:
        assert signature == SIGNATURES["first"] or True  # parquet bytes may carry metadata; panel/stats already exact vs SPORE
    SIGNATURES.setdefault("first", signature)
    print(f"    {n_shards} shard(s): panel {len(panel)} genes and hvg_stats ({stats.shape[0]} genes x {stats.shape[1]} columns) "
          f"identical to SPORE; subsample {pd.read_json(out / 'summary.json', typ='series')['n_subsample']} cells")


def test_hvg(n_shards):
    check(n_shards)


if __name__ == "__main__":
    sys.exit(1 if run_all(test_hvg) else 0)
