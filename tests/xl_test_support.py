"""Shared helpers for the stage tests: build a fresh XL run directory + config, load the SPORE reference."""
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent                    # the SPORE v4 repository
sys.path.insert(0, str(REPO / "src" / "engine"))
# BUILD = the folder that holds the test reference material: data/ (synthetic_60k.h5ad, base config, protected list) and intermediate/
# (SPORE v3.0 reference runs). `python -B tests/prepare_references.py` builds it under tests/ref; XL_TEST_DATA points elsewhere.
BUILD = Path(os.environ.get("XL_TEST_DATA", str(REPO / "tests" / "ref")))

import yaml  # noqa: E402

REFERENCE = BUILD / "intermediate" / "spore_reference_data"
BASE_CONFIG = BUILD / "data" / "spore_config_synthetic60k_baseline_v1.yaml"
SOURCE = BUILD / "data" / "synthetic_60k.h5ad"


def fresh_run_config(label, shard_rows=None, kind="h5ad_rows", obs_key="gem_group", n_blocks=8, control_block_rows=1000,
                     hvg_cap=100000):
    """Writes a config whose run dir is a brand-new folder under XL_TEST_TMP; returns the config path."""
    base = os.environ.get("XL_TEST_TMP", tempfile.gettempdir())
    run_dir = Path(tempfile.mkdtemp(prefix=f"xl_run_{label}_", dir=base))
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["paths"]["project_root"] = str(run_dir).replace("\\", "/")
    cfg["paths"]["raw_h5ad"] = ""
    source = {"kind": kind, "h5ad": str(SOURCE).replace("\\", "/")}
    if kind == "h5ad_rows":
        source["shard_rows"] = int(shard_rows)
    else:
        source["obs_key"] = obs_key
    cfg["xl"] = {"compat": "v3", "source": source, "work_dir": "_xl_work", "n_blocks": n_blocks,
                 "control_block_rows": control_block_rows, "hvg_subsample_cells": int(hvg_cap),
                 "bucket_compression": "lzf", "memory_budget_gb": 24, "merge_mode": "memory"}
    config_path = run_dir.parent / f"{run_dir.name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


def read_lines(path):
    return [line for line in Path(path).read_text(encoding="utf-8").split("\n") if line]


def shard_rows_for(n_shards, n_cells=60000):
    return -(-n_cells // n_shards)


def run_all(test_function, shard_counts=(1, 3, 7)):
    failures = 0
    for count in shard_counts:
        try:
            test_function(count)
            print(f"PASS {test_function.__name__} with {count} shard(s)")
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            failures += 1
            print(f"FAIL {test_function.__name__} with {count} shard(s)")
    return failures


# ---- persistent test runs: stages are idempotent, so a run dir is reused and only missing stages execute.
# Key includes XL_RUN_TAG (default "v1"); bump the tag when code of an EARLIER stage changes.
def get_run(n_shards, through="reduce2", kind="h5ad_rows", hvg_cap=100000):
    import json
    base = Path(os.environ.get("XL_TEST_TMP", tempfile.gettempdir()))
    registry_path = base / "xl_runs_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    key = f"{kind}_{n_shards}_{os.environ.get('XL_RUN_TAG', 'v4')}" + (f"_cap{hvg_cap}" if hvg_cap != 100000 else "")
    if key not in registry:
        shard_rows = shard_rows_for(n_shards) if kind == "h5ad_rows" else None
        registry[key] = str(fresh_run_config(f"reg_{kind}_{n_shards}", shard_rows=shard_rows, kind=kind, hvg_cap=hvg_cap))
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    from xl_common import Run
    import cli as spore_xl
    run = Run(registry[key])
    order = ["plan", "map1", "reduce1", "map2", "gather", "reduce2", "hvg", "map3", "metacell_prep", "metacell", "finalize", "report"]
    for stage in order[:order.index(through) + 1]:
        spore_xl.run_stage(run, stage)
    return run


def scratch_for(prefix):
    base = os.environ.get("XL_TEST_TMP", tempfile.gettempdir())
    return Path(tempfile.mkdtemp(prefix=f"xl_{prefix}_", dir=base))  # not cleaned up


def file_digest(path):
    """Content digest. .h5ad files embed HDF5 object timestamps, so whole-file bytes can differ by a second between two identical
    runs: hash X (indptr, indices, data bytes), obs and var instead. JSON: parsed, without the run-specific _source_split_json path."""
    import hashlib
    import json
    path = Path(path)
    if path.suffix == ".h5ad":
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sp
        adata = ad.read_h5ad(path)
        matrix = sp.csr_matrix(adata.X)
        digest = hashlib.sha256()
        for array in (matrix.indptr, matrix.indices, matrix.data):
            digest.update(array.tobytes())
        digest.update(pd.util.hash_pandas_object(adata.obs.astype(str), index=True).to_numpy().tobytes())
        digest.update(pd.util.hash_pandas_object(adata.var.astype(str), index=True).to_numpy().tobytes())
        return digest.hexdigest()
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("_source_split_json", None)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()
