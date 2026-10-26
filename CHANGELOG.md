# Changelog

## v4.0 (October 2026): SPORE-XL

SPORE-XL is SPORE rebuilt as a sharded, memory-bounded pipeline. It applies the same cleaning rules as SPORE v3.0 (the SPORE phase files are byte-identical), but no task ever holds more than one shard, one perturbation block or the 100,000-cell gene-panel subsample in memory. A screen's size now sets how long a run takes, not how much memory it needs: the 7.7-million-cell CD4+ T screen needed 17.5 GB for its largest task, and compute grows by about 1.4 core-hours per million cells. SPORE v3.0 stays available as the tag `v3.0` and the branch `v3-archive`.

### New

* **One command per run.** `python spore_xl.py run --config my_run.yaml`, or `sbatch slurm/spore_xl.sbatch my_run.yaml` on SLURM, runs intake, every stage, the report and the verification. It checks every input path first, finds the CPUs and memory available, lowers the number of parallel tasks when memory is short, and resumes where it stopped after an interruption.
* **Verification on every run.** 15 checks prove that every perturbation and every control cell is in exactly one split, that no cell appears twice, that the training set holds no validation or test unit, and that every written file matches its recorded checksum. The run exits with an error naming any failed check.
* **Identical across environments.** The same config gives identical cells, splits, gene panel and counts in an interactive session and in a SLURM job. Between a Windows laptop and a Linux cluster every cleaning and splitting result is identical, and the metacells differ by about 0.1% because k-means rounding depends on the processor.
* **A log-normalized training set**, `{name}_train_hybrid_lognorm`, next to the raw one.
* **A gene panel of exactly `n_top_genes` genes.** Perturbation targets take at most `max_target_nodes` of it (default 70%), so highly variable genes always keep room; held-out targets and protected genes are always included. `manifest/perturbation_nodes.tsv` lists which perturbations became panel genes and why. SPORE v3.0 grew the panel above `n_top_genes` instead (`target_node_policy: all` restores that).
* **A batch-confounding check calibrated against sampling noise.** A perturbation is flagged only when its batch mix differs from the controls' beyond what random draws of the same number of control cells produce. On the K562 genome-wide screen the uncalibrated rule flagged 5,092 of 9,118 perturbations; the calibrated one flags 22.
* **A response-based escaper score** (`phase5_escaper_filtering.escaper_score`). It scores each cell by how far its whole transcriptome moved, so it can see escapers of targets too weakly expressed for the target-expression rule. By default it is counted in the report without removing cells; `union` applies both rules.
* **Diagnostics in the report**: the batch-confounding table, the share of variance explained by perturbation, batch and donor, and a cell-cycle diagnostic. None of them changes the data.
* **Intake for more formats.** `spore_xl.py convert` reads h5ad (sparse or dense, including `layers/counts` and `raw/X`), 10x CellRanger `.h5` with guide assignment, 10x `.mtx` folders, loom, csv / tsv and HDF5 with R-exported tables, and repairs common label problems. A summary of the intake is in every report.
* **Normalized input is refused** before anything is written.
* **A protected-gene expression gate** (`phase8_hvg.protected_expression_gate`): protected genes barely detected in the controls can be left out of the panel. The gated list is written into the run folder; your list is never changed.
* **Memory pre-flight.** Before each stage the run estimates the memory of its largest task and chooses how many tasks to run at once.
* **A stable gene-panel tie-break** that no longer depends on Python's hash seed.
* **A start-up banner** with the version, run, mode, workers and host (`SPORE_NO_BANNER=1` turns it off).

### Compatibility with SPORE v3.0

* `xl.compat: v3` in a config reproduces SPORE v3.0 exactly: the same cells, gene mask, split file, gene panel, raw counts, normalized values and metacells. The test suite checks this on synthetic data at 1, 3 and 7 shards, and it was confirmed on the RPE1 and K562 screens.
* Config sections keep their names, so SPORE v3.0 configs still load.
* The v3.0 names of the normalized outputs and figures stay available as aliases (hard links, no extra disk) for this release:

| v4.0 name | v3.0 name |
|---|---|
| `processed/{name}_{split}_norm/` | `{name}_{split}_p9/` |
| `figures/cell_qc_violin.png`, `cell_qc_mt_scatter.png`, `escaper_efficiency_scatter.png`, `gene_qc_dashboard.png`, `split_dashboard.png`, `hvg_dashboard.png`, `normalization.png` | `p02_qc_violin.png`, `p02_mt_scatter.png`, `p05_efficiency_scatter.png`, `p06_triage_dashboard.png`, `p07_splits_dashboard.png`, `p08_hvg_dashboard.png`, `p09_normalization.png` |

* The phases are renumbered to the order they run in:

| v4.0 phase | v3.0 phase | config section |
|---|---|---|
| 1 Intake | 0 (in part) | `xl.source`, `spore_xl.py convert` |
| 2 Gene IDs | 1 | `phase1_detection` |
| 3 Cell QC | 2 | `phase2_cell_triage` |
| 4 Escaper filter | 5 | `phase5_escaper_filtering` |
| 5 Gene QC | 6 | `phase6_gene_triage` |
| 6 Split | 7 | `phase7_splits` |
| 7 Gene panel | 8 | `phase8_hvg` |
| 8 Normalization | 9 | `phase9_normalization` |
| 9 Metacells | 12 | `phase12_metacell` |
| 10 Verification and report | new | |

* Ambient RNA removal (v3.0 phase 3), doublet removal (4), PCA, Harmony and cell-cycle regression (10) and cell-line separation (11) are not part of SPORE-XL. None of their outputs fed the split, the gene panel or the metacells. They remain available on the tag `v3.0`.

### Repository

* The repository is renamed `SPORE-XL`. `main` holds SPORE-XL; SPORE v3.0 is the tag `v3.0` and the branch `v3-archive`, and v1 is the branch `v1-archive`.
* One entry point, `spore_xl.py`. The code is one package: `src/phases/` (the SPORE cleaning rules), `src/engine/` (sharding, stages, verification), `src/intake/` and `src/report/`. Example configs are in `configs/`, developer tools in `tools/` and `tests/`.

### Fixes

* One `mean_umi_threshold` default (0.25); SPORE v3.0 had two conflicting values.
* Multi-file 10x Flex inputs whose files list the same genes in different orders now produce one consistent gene list.
* The memory estimate of the block stages now budgets the largest block, which is usually a control block.

## v3.0 (October 2026), standalone release

One shared `src/` and two runners, replacing the separate light and plus packages: `spore_local.py` (workstation, `--resume`) and `spore_cluster.py` (one dataset per SLURM array task). Stale-checkpoint fix in Phase 9, protected list found from any folder, fresh-folder guard, configs from the runs that worked, K562 protected list v2.0 (312 genes), `scikit-misc` in `requirements.txt`. Tag `v3.0`, branch `v3-archive`.

## v2.1 (8 July 2026), one-stop rebuild

Raw HVG panel on disk (`_hvgraw`), automatic held-out coverage in Phase 8, native Phase 12 (control-preserving metacell substrate and train hybrid), packaging fixes, config validation against the data, removal of CHITIN, and a verification gate that writes `COVERAGE_REPORT.md`.

## v2.0.1 (early July 2026), coverage recovery

Ribosomal perturbation targets and protected genes exempt from the ribosomal filter; split-independent protected list; held-out targets present in the final RPE1 graph rose from 35/152 to 119/152 (validation) and from 79/304 to 249/304 (test).

## v2.0, SPORE+

Fourteen phases (0 to 13): detection, separate ambient RNA and doublet phases, cell-line separation, protected-gene tiers, worker subprocesses for the largest phases.
