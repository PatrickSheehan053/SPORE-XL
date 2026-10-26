"""The HVG subsample DRAW is only exercised when the train split is larger than the cap (SPORE hardcodes 100,000 and the
synthetic train split has 28,138 cells, so the default test takes every cell). Here SPORE's REAL select_hvgs runs on SPORE's
train split with its cap lowered to 10,000 (by shadowing `min` inside src.phases.phase08_hvg for the one call that sets n_sample),
and the XL hvg stage runs with xl.hvg_subsample_cells = 10000. Panel and hvg_stats must be identical.
Run: set PYTHONHASHSEED=0 ; set PYTHONDONTWRITEBYTECODE=1 ; python -B tests\\test_stage_hvg_subsample.py"""
import logging
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import BASE_CONFIG, BUILD, get_run, read_lines, run_all  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402

paths.add_spore_to_path()
import config_v3  # noqa: E402
import src.phases.phase08_hvg as spore_hvg  # noqa: E402

CAP = 10000
BASELINE = BUILD / "intermediate" / "spore_baseline" / "run1" / "splits"
_CACHE = {}


def spore_reference():
    if "ref" in _CACHE:
        return _CACHE["ref"]
    cfg = config_v3.load_config(str(BASE_CONFIG))  # splits dir = baseline run1 (held-out JSON is read from there)
    train = ad.read_h5ad(BASELINE / "synth60k_train.h5ad")
    logger = logging.getLogger("hvg_subsample_ref")
    logger.addHandler(logging.NullHandler())
    real_min = min

    def capped_min(first, second=None, *rest):
        if second is not None and not rest and first == 100_000:  # the n_sample line only
            return real_min(CAP, second)
        return real_min(first, second, *rest) if second is not None else real_min(first)

    spore_hvg.min = capped_min  # module-global shadows the builtin inside select_hvgs
    try:
        adata_new, hvg_names = spore_hvg.select_hvgs(train, cfg, logger, split_label="train")
    finally:
        del spore_hvg.min
    stats = pd.DataFrame.from_dict(adata_new.uns["hvg_stats"], orient="index")
    _CACHE["ref"] = (list(adata_new.var_names), stats, train_n(train))
    return _CACHE["ref"]


def train_n(_):
    return 28138


def check(n_shards):
    panel_ref, stats_ref, n_train = spore_reference()
    run = get_run(n_shards, through="hvg", hvg_cap=CAP)
    panel = read_lines(run.splits_dir / f"{run.name}_hvg_panel.txt")
    summary = pd.read_json(run.stats_dir / "hvg" / "all" / "summary.json", typ="series")
    assert summary["n_subsample"] < summary["n_train"], "the draw was not exercised"
    stats = pd.read_parquet(run.stats_dir / "hvg" / "all" / "hvg_stats.parquet")
    assert panel == panel_ref, (len(panel), len(panel_ref), sorted(set(panel) ^ set(panel_ref))[:10])
    assert sorted(stats.index) == sorted(stats_ref.index)
    stats_ref = stats_ref.loc[stats.index, list(stats.columns)]
    for column in stats.columns:
        left, right = stats[column].to_numpy(), stats_ref[column].to_numpy()
        if left.dtype.kind == "f":
            assert left.tobytes() == right.astype(left.dtype).tobytes(), column
        else:
            assert (left == right).all(), column
    print(f"    {n_shards} shard(s): subsample {summary['n_subsample']}/{summary['n_train']} train cells drawn; "
          f"panel {len(panel)} genes and hvg_stats identical to SPORE's select_hvgs at cap {CAP}")


def test_hvg_subsample(n_shards):
    check(n_shards)


if __name__ == "__main__":
    sys.exit(1 if run_all(test_hvg_subsample) else 0)
