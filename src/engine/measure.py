"""measure: header-only facts about the raw input, and the RAM / disk projection for a SPORE-XL run. No matrix values are read.

  python -B src\\spore_xl.py measure --source <file.h5ad | file.tar | folder of .h5> [--match _Rest_] [--out DIR]
         [--shard-rows N] [--n-blocks N] [--n-perts N] [--survival 0.65] [--workers N] [--max-ram-gb G]

Source units: an .h5ad (one unit), 10x .h5 members inside a .tar (members whose name contains --match and ends in .h5; read
with tarfile.extractfile + h5py, only indptr[-1], the shape and the feature types are touched), or loose .h5 files.
Writes measure.tsv (cells, features, stored nonzeros, nonzeros per cell per unit) and measure_projection.json/.md.
The projection constants were calibrated on the synthetic 60k and 1M runs (2026-10-18); they are estimates, not guarantees.
"""
import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import psutil

import shard_planner

# Disk model recalibrated 2026-10-19 (session 22) on the finished RPE1 graduation run (867,313,640 stored values in the source,
# 247,914 cells, survival 0.827, panel 5,000 of 8,739 genes = 0.548 of the surviving stored values, merged files written):
# run directory 34.49 GB = 39.76 bytes per source value (tests/diag_run_disk_breakdown.py). The earlier constants came from the
# synthetic 60k/1M runs and gave 18.5 GB (real: 34.5 GB). Work files scale with the source values; outputs scale with the
# surviving PANEL values (source values x survival x panel fraction).
BYTES_PER_NNZ = {"intake_or_source_shards": 5.45, "qc_shards": 5.44, "buckets": 5.25, "blocks": 4.49}   # per source stored value
OUTPUT_BYTES_PER_PANEL_VALUE = {"parts": 16.5, "merged": 25.6}  # per surviving panel stored value; merged = concat_on_disk files beside the parts
DISK_MARGIN = 1.2            # safety factor applied by the pre-flight and the "with margin" figure
MIN_PANEL_FRACTION = 0.25    # floor for the panel share of stored values when the genome is much larger than the panel
RAM_BYTES_PER_VALUE = 12     # CSR value + index + pointer share
WORKING_COPY_FACTOR = 3      # blueprint shard rule
BASE_PROCESS_GB = 0.7        # python + scanpy + scipy imports (measured 0.5 to 0.85 GB)
METADATA_BYTES_PER_CELL = 150


def _decode(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def measure_10x_handle(handle):
    group = handle["matrix"]
    shape = [int(value) for value in group["shape"][...]]
    indptr = group["indptr"]
    n_cells = int(indptr.shape[0]) - 1
    nnz = int(indptr[-1])  # one element, not the array
    types = _decode(group["features"]["feature_type"][...])
    gene_features = sum(1 for kind in types if kind == "Gene Expression")
    guide_features = sum(1 for kind in types if any(token in kind.lower() for token in ("crispr", "guide", "protospacer")))
    return {"cells": n_cells, "features": shape[0], "gene_features": gene_features, "guide_features": guide_features,
            "nnz": nnz, "nnz_estimated": False}


def list_h5_members(source, match):
    source = Path(source)
    if source.is_dir():
        return [(str(path), path.stat().st_size) for path in sorted(source.glob("*.h5")) if match in path.name]
    with tarfile.open(source) as archive:
        return sorted((member.name, member.size) for member in archive if member.isfile() and member.name.endswith(".h5") and match in member.name)


def measure_source(source, match=""):
    source = Path(source)
    rows = []
    if source.suffix == ".h5ad":
        facts = shard_planner.measure_h5ad(source)
        rows.append({"unit": source.name, "kind": "h5ad", "size_bytes": source.stat().st_size, "cells": facts["n_obs"],
                     "features": facts["n_vars"], "gene_features": facts["n_vars"], "guide_features": 0, "nnz": facts["nnz"],
                     "nnz_estimated": facts["format"] == "dense"})
    elif source.suffix == ".tar":
        with tarfile.open(source) as archive:
            members = sorted((m for m in archive if m.isfile() and m.name.endswith(".h5") and match in m.name), key=lambda m: m.name)
            for member in members:
                with archive.extractfile(member) as raw, h5py.File(raw, "r") as handle:
                    facts = measure_10x_handle(handle)
                rows.append({"unit": member.name.split("/")[-1], "kind": "10x_h5_in_tar", "size_bytes": member.size, **facts})
    elif source.is_dir():
        for path in sorted(source.glob("*.h5")):
            if match not in path.name:
                continue
            with h5py.File(path, "r") as handle:
                facts = measure_10x_handle(handle)
            rows.append({"unit": path.name, "kind": "10x_h5", "size_bytes": path.stat().st_size, **facts})
    else:
        raise ValueError(f"unsupported source {source}: expected .h5ad, .tar or a folder of .h5 files")
    if not rows:
        raise ValueError(f"no units found in {source} (match '{match}')")
    table = pd.DataFrame(rows)
    table["nnz_per_cell"] = table["nnz"] / table["cells"].clip(lower=1)
    return table


def panel_fraction_estimate(panel_genes, gene_features):
    """Share of the surviving stored values that sit in panel genes: RPE1 measured 0.548 for 5,000 of 8,739 genes (proxy 0.572)."""
    return float(min(1.0, max(MIN_PANEL_FRACTION, panel_genes / max(gene_features, 1))))


def disk_projection(total_nnz, survival, panel_fraction, merged=True):
    """GB by part for the whole run directory when nothing is cleaned up. `survival` = cells reaching the outputs / cells in."""
    work = {name: total_nnz * value / 1e9 for name, value in BYTES_PER_NNZ.items()}
    panel_values = total_nnz * survival * panel_fraction
    outputs = {"outputs_parts": panel_values * OUTPUT_BYTES_PER_PANEL_VALUE["parts"] / 1e9}
    if merged:
        outputs["outputs_merged"] = panel_values * OUTPUT_BYTES_PER_PANEL_VALUE["merged"] / 1e9
    return work, outputs


def recommend_workers(cpus, max_ram_gb, peak_task_gb):
    """min(CPUs - 2, floor(max_ram_gb / per-task peak)), at least 1."""
    return int(max(1, min(cpus - 2, int(max_ram_gb // max(peak_task_gb, 0.1)))))


PARALLEL_TASKS = ("intake / map1 / map2 (one shard)", "gather / map3 / metacell (largest block)")  # the stages --workers can spread


def project(table, shard_rows=None, n_blocks=64, n_perts=None, survival=0.65, workers=1, panel_genes=5000, hvg_cap=100000,
            control_block_rows=120000, cpus=None, max_ram_gb=None, merged=True):
    """RAM per task and disk for the whole run. Units that are tar members are shards themselves (one member = one shard)."""
    total_cells = int(table["cells"].sum())
    total_nnz = int(table["nnz"].sum())
    nnz_per_cell = total_nnz / max(total_cells, 1)
    gene_fraction = (table["gene_features"] / table["features"].clip(lower=1)).mean()
    if table["kind"].iloc[0].startswith("10x"):
        largest_shard = int(table["cells"].max())
        n_shards = len(table)
    else:
        largest_shard = int(min(shard_rows or table["cells"].max(), table["cells"].max()))
        n_shards = int(np.ceil(total_cells / largest_shard))
    surviving_cells = total_cells * survival
    pert_cells = surviving_cells * 0.96
    per_cell_gb = nnz_per_cell * RAM_BYTES_PER_VALUE * WORKING_COPY_FACTOR / 1e9
    largest_block_cells = pert_cells / n_blocks * 3.0  # a block cannot be smaller than its largest perturbation; skew factor 3
    perts = n_perts or 0
    n_sub = min(hvg_cap, pert_cells * 0.6)
    ram = {
        "intake / map1 / map2 (one shard)": BASE_PROCESS_GB + largest_shard * per_cell_gb,
        "gather / map3 / metacell (largest block)": BASE_PROCESS_GB + largest_block_cells * per_cell_gb,
        "hvg (subsample rows + selection)": BASE_PROCESS_GB + 1.0 + n_sub * nnz_per_cell * RAM_BYTES_PER_VALUE * 2 / 1e9,
        "metacell_prep (dense PCA fit on the subsample)": BASE_PROCESS_GB + n_sub * panel_genes * 4 * 1.5 / 1e9,
        "reduce2 (per-cell metadata + per-perturbation means)": BASE_PROCESS_GB + surviving_cells * METADATA_BYTES_PER_CELL / 1e9
                                                               + perts * 20000 * 4 / 1e9,
        "plan": BASE_PROCESS_GB + total_cells * METADATA_BYTES_PER_CELL / 1e9,
    }
    peak_task = max(ram.values())
    panel_fraction = panel_fraction_estimate(panel_genes, table["gene_features"].mean())
    work, outputs = disk_projection(total_nnz, survival, panel_fraction, merged)
    output_gb = sum(outputs.values())
    disk = {**work, **outputs}
    without = (sum(work.values()) + output_gb) * 1.01          # + stats / manifest / reports (0.1 bytes per value on RPE1)
    # with clean-up (source shards after map1, QC shards after map2, buckets after gather; blocks kept): live set per stage
    live = {"map1 (source shards + QC shards)": work["intake_or_source_shards"] + work["qc_shards"],
            "map2 (QC shards + buckets)": work["qc_shards"] + work["buckets"],
            "gather (buckets + blocks)": work["buckets"] + work["blocks"],
            "map3 / metacell / finalize (blocks + outputs)": work["blocks"] + output_gb}
    with_cleanup = max(live.values())
    cpus = int(cpus or os.cpu_count() or 2)
    budget = float(max_ram_gb) if max_ram_gb is not None else 0.8 * psutil.virtual_memory().total / 1e9
    parallel_peak = max(ram[name] for name in PARALLEL_TASKS)
    return {"total_cells": total_cells, "total_nnz": total_nnz, "nnz_per_cell": nnz_per_cell, "gene_feature_fraction": float(gene_fraction),
            "n_shards": n_shards, "largest_shard_cells": largest_shard, "n_blocks": n_blocks, "survival_assumed": survival,
            "ram_gb_per_task": {key: round(value, 2) for key, value in ram.items()}, "peak_task_ram_gb": round(peak_task, 2),
            "workers": workers, "workers_x_peak_gb": round(workers * peak_task, 1), "workers_x_parallel_peak_gb": round(workers * max(ram[name] for name in PARALLEL_TASKS), 1),
            "cpus": cpus, "max_ram_gb": round(budget, 1), "peak_parallel_task_ram_gb": round(parallel_peak, 2),
            "recommended_workers": recommend_workers(cpus, budget, parallel_peak),
            "recommended_workers_rule": "min(CPUs - 2, floor(max_ram_gb / peak RAM of the parallel tasks)); hvg, reduce and plan are single tasks",
            "panel_fraction_assumed": round(panel_fraction, 3), "merged_outputs_assumed": bool(merged),
            "disk_gb_per_part": {key: round(value, 1) for key, value in disk.items()},
            "disk_gb_without_cleanup": round(without, 1), "disk_gb_without_cleanup_with_margin": round(without * DISK_MARGIN, 1),
            "disk_gb_with_cleanup_peak": round(with_cleanup * 1.01, 1),
            "disk_live_set_gb_by_stage": {key: round(value * 1.01, 1) for key, value in live.items()},
            "note": ("disk constants calibrated on the finished RPE1 run (39.8 bytes per source value in total, see BYTES_PER_NNZ); "
                     "the input data itself is not included; the panel share of stored values is a proxy")}


def write_outputs(table, projection, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("measure.tsv", "measure_projection.json", "measure_projection.md"):
        if (out_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite {out_dir / name}")
    table.to_csv(out_dir / "measure.tsv", sep="\t", index=False)
    (out_dir / "measure_projection.json").write_text(json.dumps(projection, indent=2), encoding="utf-8")
    lines = ["# measure projection", "",
             f"- units: {len(table)}; cells {projection['total_cells']:,}; stored nonzeros {projection['total_nnz']:,}; "
             f"nonzeros per cell {projection['nnz_per_cell']:.0f}",
             f"- shards {projection['n_shards']} (largest {projection['largest_shard_cells']:,} cells); blocks {projection['n_blocks']}",
             "", "| task | projected peak RAM (GB) |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in projection["ram_gb_per_task"].items()]
    lines += ["", f"peak task RAM {projection['peak_task_ram_gb']} GB; with {projection['workers']} worker(s) in the parallel stages: {projection['workers_x_parallel_peak_gb']} GB",
              f"recommended --workers {projection['recommended_workers']} ({projection['recommended_workers_rule']}; CPUs {projection['cpus']}, "
              f"max RAM {projection['max_ram_gb']} GB, parallel-task peak {projection['peak_parallel_task_ram_gb']} GB)",
              "", f"disk without clean-up: about {projection['disk_gb_without_cleanup']} GB ({projection['disk_gb_without_cleanup_with_margin']} GB with the "
              f"{DISK_MARGIN} margin); with clean-up (peak live set): about {projection['disk_gb_with_cleanup_peak']} GB", "", "| stage | live set (GB) |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in projection["disk_live_set_gb_by_stage"].items()]
    lines += ["", projection["note"]]
    (out_dir / "measure_projection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="spore_xl.py measure")
    parser.add_argument("--source", required=True)
    parser.add_argument("--match", default="")
    parser.add_argument("--out", default=None)
    parser.add_argument("--shard-rows", type=int, default=None)
    parser.add_argument("--n-blocks", type=int, default=64)
    parser.add_argument("--n-perts", type=int, default=None)
    parser.add_argument("--survival", type=float, default=0.65)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-ram-gb", type=float, default=None)
    parser.add_argument("--panel-genes", type=int, default=5000)
    parser.add_argument("--cpus", type=int, default=None, help="CPUs of the machine that will run it (default: this machine)")
    parser.add_argument("--no-merged-outputs", action="store_true", help="project without the merged split files (xl.merge_split_outputs: never)")
    args = parser.parse_args(argv)
    table = measure_source(args.source, args.match)
    projection = project(table, args.shard_rows, args.n_blocks, args.n_perts, args.survival, args.workers, args.panel_genes,
                         cpus=args.cpus, max_ram_gb=args.max_ram_gb, merged=not args.no_merged_outputs)
    if args.out:
        write_outputs(table, projection, args.out)
    print(table.to_string(index=False, max_rows=12))
    print(json.dumps(projection, indent=2))
    # P8: the same rule as preflight.check_workers and recommended_workers: workers only multiply the PARALLEL stages (intake / map1 / map2 /
    # gather / map3 / metacell); hvg, reduce and plan are single tasks and never run in the pool
    need = args.workers * projection["peak_parallel_task_ram_gb"]
    if args.max_ram_gb is not None and need > args.max_ram_gb:
        print(f"REFUSED: {args.workers} worker(s) x {projection['peak_parallel_task_ram_gb']} GB (largest parallel task) = {need:.1f} GB exceeds "
              f"--max-ram-gb {args.max_ram_gb}; use --workers {projection['recommended_workers']} or less")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
