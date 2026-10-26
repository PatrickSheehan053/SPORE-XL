"""reduce2 at 1, 3 and 7 row shards vs SPORE (in-process reference + the baseline run's split JSON).
Run: set PYTHONHASHSEED=0 ; python -B tests\\test_stage_reduce2.py   (XL_TEST_TMP = scratch folder)
Gate: Phase 6 gene mask / rescues / waterfall / penetrance equal; mean_shift.tsv equal to SPORE's scores (same repr of
every float, same order); split JSON BYTE-identical to the baseline run's; control assignment equal to SPORE's three
split cell sets; survived_to_phase7 equals SPORE's survivors; QC var columns within 1e-6; outputs identical across
shard counts."""
import hashlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import BUILD, REFERENCE, fresh_run_config, read_lines, run_all, shard_rows_for  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xl_common import Run  # noqa: E402
import stage_gather  # noqa: E402
import stage_map1  # noqa: E402
import stage_map2  # noqa: E402
import stage_plan  # noqa: E402
import stage_reduce1  # noqa: E402
import stage_reduce2  # noqa: E402

BASELINE_JSON = BUILD / "intermediate" / "spore_baseline" / "run1" / "splits" / "synth60k_split_indices.json"
SIGNATURES = {}


def build(n_shards):
    config = fresh_run_config(f"r2_{n_shards}", shard_rows=shard_rows_for(n_shards))
    run = Run(config)
    stage_plan.run_plan(run)
    for index in range(run.n_shards()):
        stage_map1.run_map1(run, index)
    stage_reduce1.run_reduce1(run)
    for index in range(run.n_shards()):
        stage_map2.run_map2(run, index)
    for block in range(stage_gather.total_blocks(run)):
        stage_gather.run_gather(run, block)
    stage_reduce2.run_reduce2(run)
    return run


def check(n_shards):
    run = build(n_shards)
    stage_reduce2.run_reduce2(run)  # idempotent
    out = run.stats_dir / "reduce2" / "all"

    # ---- Phase 6
    ref_var = read_lines(REFERENCE / "p6_var.txt")
    assert read_lines(out / "gene_keep_mask.txt") == ref_var
    assert read_lines(out / "phase6_rescued.txt") == read_lines(REFERENCE / "p6_rescued.txt")
    ref_waterfall = pd.read_json(REFERENCE / "p6_waterfall.json", typ="series")
    ours_waterfall = pd.read_json(out / "phase6_waterfall.json", typ="series")
    assert ours_waterfall.to_dict() == ref_waterfall.to_dict(), (ours_waterfall.to_dict(), ref_waterfall.to_dict())
    assert np.load(out / "phase6_penetrance.npy").tobytes() == np.load(REFERENCE / "p6_penetrance.npy").tobytes()

    # ---- mean shift: same scores (exact repr) in the same order
    ours = pd.read_csv(out / "mean_shift.tsv", sep="\t", keep_default_na=False)
    reference = pd.read_csv(REFERENCE / "p7_mean_shift.tsv", sep="\t", keep_default_na=False)
    assert list(ours["label"]) == list(reference["label"]), "ranking differs"
    assert list(ours["score"]) == list(reference["score"]), "scores differ in the last digits"

    # ---- split JSON byte-identical to the SPORE baseline run
    split_json = run.splits_dir / f"{run.name}_split_indices.json"
    assert split_json.read_bytes() == BASELINE_JSON.read_bytes(), "split JSON differs from SPORE's"

    # ---- control assignment equals SPORE's split cell sets
    layout = run.layout()
    names = np.array(ad.read_h5ad(BUILD / "data" / "synthetic_60k.h5ad", backed="r").obs_names)
    assignment = pd.read_parquet(run.splits_dir / f"{run.name}_control_assignment.parquet")
    for split in ("train", "val", "test"):
        reference_controls = {name for name in read_lines(REFERENCE / f"p7_{split}_cells.txt")
                              if name in set(names[assignment["xl_global_index"]])}
        ours_controls = set(names[assignment.loc[assignment["split"] == split, "xl_global_index"]])
        assert ours_controls == reference_controls, split
    final = pd.read_parquet(run.manifest_dir / "global_obs_final.parquet")
    survivors = set(names[final.loc[final["survived_to_phase7"], "xl_global_index"]])
    assert survivors == set(read_lines(REFERENCE / "p5b_cells.txt"))
    for split in ("train", "val", "test"):
        ours_cells = set(names[final.loc[final["split"] == split, "xl_global_index"]])
        assert ours_cells == set(read_lines(REFERENCE / f"p7_{split}_cells.txt")), split

    # ---- QC var columns vs SPORE (after the ribosomal drop)
    qc = pd.read_csv(out / "qc_var_columns_post_ribo.tsv", sep="\t", index_col=0)
    ref_table = pd.read_csv(REFERENCE / "p2_var_table.tsv", sep="\t", index_col=0)
    for column in ("n_cells_by_counts", "mean_counts", "pct_dropout_by_counts", "total_counts"):
        assert np.max(np.abs(qc[column].to_numpy(float) - ref_table.loc[qc.index, column].to_numpy(float))) <= 1e-6 * max(
            1.0, float(np.max(np.abs(qc[column].to_numpy(float))))), column

    # ---- identical across shard counts
    signature = {
        "mean_shift": hashlib.sha256((out / "mean_shift.tsv").read_bytes()).hexdigest(),
        "split": hashlib.sha256(split_json.read_bytes()).hexdigest(),
        "mask": hashlib.sha256((out / "gene_keep_mask.txt").read_bytes()).hexdigest(),
        "controls": hashlib.sha256(assignment.to_csv(index=False).encode()).hexdigest(),
    }
    if SIGNATURES:
        assert signature == SIGNATURES["first"], "outputs differ between shard counts"
    else:
        SIGNATURES["first"] = signature
    print(f"    {n_shards} shard(s): {len(ref_var)} genes kept, mean shift identical for {len(ours)} perturbations, "
          f"split JSON byte-identical, {len(assignment)} controls assigned")


def test_reduce2(n_shards):
    check(n_shards)


if __name__ == "__main__":
    sys.exit(1 if run_all(test_reduce2) else 0)
