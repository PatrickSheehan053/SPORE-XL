# SPORE-XL v4.0 developer guide

A user needs none of this: `python spore_xl.py run --config my_run.yaml`, or `sbatch slurm/spore_xl.sbatch my_run.yaml` on a cluster (README, `docs/HPC_SETUP.md`). This guide is for the
people who change SPORE-XL or who have to show that a change, a new cluster or a new dataset behaves: the test suite, the comparisons with SPORE v3.0, the acceptance run on synthetic
data, the K562 comparison of the two final runs, the CD4 check against the verified tally, and the measuring tools. All paths are relative to the repository folder.

## 1. Test suite

`python -B tests/run_all_tests.py --tag <name> --logs <new folder> --scratch <folder> [--data <reference folder>] [--quick]` runs the test files one at a time (resumable). `--quick` (13 tests, about two
minutes) needs no reference data. The full list needs the reference material built once by `tests/prepare_references.py` (synthetic data and three SPORE v3.0 runs); see `tests/README_TESTS.md`.
`tests/stress/` holds eleven kinds of deliberately difficult synthetic datasets with answer keys and a runner.

## 1b. SPORE v3.0 is not in this repository: how the tests get it

SPORE v3.0 (commit `5015dc6`) is the tag `v3.0` and the branch `v3-archive` of this GitHub repository. Its runners (`spore_local.py`, `spore_cluster.py`) and its v3-format configs are not part of the
tree. Two things need it, and both use a checkout OUTSIDE the repository (same Python environment as the run that is compared: `tests/compare_v4.py` refuses otherwise):

```
git clone https://github.com/PatrickSheehan053/SPORE-XL <folder outside this repository>/spore_v3_checkout
git -C <folder outside this repository>/spore_v3_checkout checkout v3.0
set XL_SPORE_V3_DIR=<folder outside this repository>/spore_v3_checkout          (Linux: export XL_SPORE_V3_DIR=...)
```
* `tests/prepare_references.py --spore-v3-dir <checkout>` (or `XL_SPORE_V3_DIR`) runs the v3.0 runner `<checkout>/spore_local.py` on the synthetic 60k to build the reference runs that the equivalence, stage and
  robustness tests compare with (it refuses a checkout whose `spore_local.py` differs from the recorded hash). The quick tests need no checkout.
* The provenance proof. `tests/spore_5015dc6_hashes.json` records the sha256 of the 12 SPORE v3.0 files the engine reads or clones. **10 of them are in this repository, in `src/phases/`, byte-identical to
  v3.0 (moved, never edited). 2 are not shipped and are verified only in the v3.0 checkout:** `spore_local.py` (the runner) and `phase00_sparse_convert.py` (called only by the runner).
  `python -B tests/test_spore_api.py` checks the 10 always, and all 12 against the checkout when `XL_SPORE_V3_DIR` is set (without it the output says the 2 were not checked).
* What the engine takes from `spore_local.py`: the thread / allocator environment defaults it sets at import (`OMP_NUM_THREADS` ... `MALLOC_MMAP_THRESHOLD_`, all `setdefault`), `load_config` with the SPORE v3.0
  `_DEFAULTS`, and `report_heldout_coverage`. They are copied character for character into `src/engine/config_v3.py` (imported at the same place and time as `spore_local` was). Do not edit that file: each copied
  piece is hashed against `tests/spore_5015dc6_config_segments.json` (recorded from `spore_local.py` at `5015dc6`) by `test_spore_api.py`, and with a checkout `load_config` is compared with the real one on every shipped config.
* `docs/PROVENANCE.csv` lists where the SPORE v3.0 files came from, by their v3.0 paths; modules that are not in this tree are on the tag.

## 1c. The pairs: `src/phases/` and `src/engine/adapted_phases/`

`src/phases/` holds SPORE's own phase modules (v3.0, unchanged). They work on a whole in-memory AnnData. `src/engine/adapted_phases/` holds adapted copies of the DECISION parts of six of them, rewritten
to work from shard statistics instead of the whole matrix; each file starts with a docstring naming the SPORE function it reproduces, and the file names are the SPORE names so a pair is easy to see:

| `src/phases/` (SPORE v3.0) | `src/engine/adapted_phases/` (SPORE-XL) | used by |
|---|---|---|
| `phase02_cell_triage.py` | `phase02_cell_triage.py` (per-cell QC mask) | map1, map3, hvg, reduce1, reduce2 |
| `phase05_escaper_filtering.py` | `phase05_escaper_filtering.py` (efficiency and undersized decisions) | map2, gather |
| `phase06_gene_triage.py` | `phase06_gene_triage.py` (keep mask from merged statistics) | reduce2 |
| `phase07_splits.py` | `phase07_splits.py` (`split_zero_shot` fed a precomputed mean shift) | reduce2 |
| `phase08_hvg.py` | `phase08_hvg.py` (`select_hvgs` split into steps) | hvg |
| `phase12_metacell.py` | `phase12_metacell.py` (per-block driver) | metacell |
| `src/report/plotting.py` | `plotting_xl.py` (`plot_sparsity_overview` from arrays) | report |

The adapted copies are separate modules on purpose and are never merged into the originals (the originals are the provenance guarantee). Where an adapted copy still calls the original (`_get_perturbation_targets`,
`force_include_into_hvg`, ...), `tests/test_spore_api.py` pins the original's signature; if SPORE changes, re-read the original and update the copy and the test together.

## 2. Comparing a run with SPORE v3.0 (or with another SPORE-XL run)

`tests/compare_v4.py` is the equivalence tool of the compat setting (`xl.compat: v3`, which reproduces SPORE v3.0 exactly):

```
python -B tests/compare_v4.py spore --xl <v4 run config> --spore-dir <SPORE v3.0 run folder>
python -B tests/compare_v4.py xl --a <config of run A> [--a-is-v3] --b <config of run B>
```
Both runs must have been made in ONE Python environment (the tool refuses otherwise). `tools/write_environment.py <run folder>` records the environment of a SPORE v3.0 run.

## 3. Acceptance run on the synthetic 60k (about 10 minutes, under 2 GB)

Proves that this Python 3.12 environment reproduces the laptop result (v4 compat equals SPORE v3.0) before real data are touched. The commands assume the unpacked bundle (`synthetic/synthetic_60k.h5ad` next to the repository folder) and a v3.0 checkout (section 1b); `<SCRATCH_WORK_DIR>` below is `<scratch folder>`.

CLUSTER (venv active, repository folder)

```
python -B tests/test_spore_api.py
```
Expected: `10 SPORE files byte-identical to the recorded hashes`, `14 pieces of config_v3.py are verbatim copies`, `29 imported/cloned SPORE names match their signatures`.

```
python -B tests/run_all_tests.py --tag install --logs <scratch folder>/test_logs --scratch <scratch folder>/test_tmp --quick
```
Expected: 13 lines `exit 0` (the start-up banner, shipped-config, confounding-null and layout tests are four of them).

```
mkdir -p <scratch folder>/synthetic_60k
```
```
cp <data folder>/synthetic_60k.h5ad <scratch folder>/synthetic_60k/synthetic_60k.h5ad
```
```
sed -e 's|<SCRATCH_WORK_DIR>|<scratch folder>|g' -e 's|<SPORE_ROOT>|<repo folder>|g' tests/configs/synthetic_60k_spore_v3.yaml > <scratch folder>/configs/synthetic_60k_spore_v3.yaml
```
```
sed -e 's|<SCRATCH_WORK_DIR>|<scratch folder>|g' -e 's|<SPORE_ROOT>|<repo folder>|g' tests/configs/synthetic_60k_compat.yaml > <scratch folder>/configs/synthetic_60k_compat.yaml
```
```
sed -e 's|<SCRATCH_WORK_DIR>|<scratch folder>|g' -e 's|<SPORE_ROOT>|<repo folder>|g' tests/configs/synthetic_60k_production.yaml > <scratch folder>/configs/synthetic_60k_production.yaml
```
SPORE v3.0 on the 60k (the reference, about 2 minutes), from the v3.0 checkout of section 1b:
```
git clone https://github.com/PatrickSheehan053/SPORE-XL <scratch folder>/spore_v3_checkout
```
```
git -C <scratch folder>/spore_v3_checkout checkout v3.0
```
```
export XL_SPORE_V3_DIR=<scratch folder>/spore_v3_checkout
```
```
python -B tests/test_spore_api.py
```
Expected now: also `all 12 recorded files verified in the v3.0 checkout`.
```
python -B $XL_SPORE_V3_DIR/spore_local.py --config <scratch folder>/configs/synthetic_60k_spore_v3.yaml
```
Expected end: `SPORE COMPLETE (GRAPH-READY)`.
```
python -B tools/write_environment.py <scratch folder>/synthetic_60k/spore_v3
```
SPORE-XL v4.0 in compat mode on the same data:
```
python -B spore_xl.py run --config <scratch folder>/configs/synthetic_60k_compat.yaml --workers 3 --max-ram-gb 20
```
Expected end: `[finalize] GRAPH-READY`, `[report] 8/8 figures`.
```
python -B src/engine/verify.py <scratch folder>/configs/synthetic_60k_compat.yaml
```
Expected: every line `PASS`.
```
python -B tests/compare_v4.py spore --xl <scratch folder>/configs/synthetic_60k_compat.yaml --spore-dir <scratch folder>/synthetic_60k/spore_v3
```
Expected: 26 lines `PASS` and a last line `PASS`; no `WARN` (a refusal naming numpy/scipy/... means SPORE and XL ran in different environments).

SPORE-XL v4.0 with the production defaults, and its differences traced to node budget to log-normalized training set:
```
python -B spore_xl.py run --config <scratch folder>/configs/synthetic_60k_production.yaml --workers 3 --max-ram-gb 20
```
```
python -B tools/diff_runs.py --ref <scratch folder>/configs/synthetic_60k_compat.yaml --new <scratch folder>/configs/synthetic_60k_production.yaml
```
Expected: every line `ok`, `HVG panel size: reference 5xx, new 500`.

If any line is FAIL, or a test fails, STOP and send the full output: do not start K562.


## 4. K562: comparing the two runs (after both jobs finished)

CLUSTER (venv active, repository folder; HPC_SETUP.md section 3)

(a) The compat run must be identical to the 21 Oct run:
```
python -B tests/compare_v4.py xl --a <reference runs>/spore_20261021_p5000_prottahoe_v800t1600_seed42_tnall_xl/config/run_config.yaml --a-is-v3 --b <scratch folder>/configs/k562_compat.yaml
```
Expected: `IDENTICAL`. Anything else is a finding: send the whole output. (The `--a` file is the config copy that the 21 Oct run stored in its own folder; its paths point at that run.)
```
python -B tools/crosscheck_run.py --config <scratch folder>/configs/k562_compat.yaml
```
Cross-checks run (a) against the SPORE v3.0 log of the same screen as on 21 Oct: MT 11,599, escapers 169,940, bypassed 2,182, undersized 748 / 24,047, split 6720/800/1598 EXACT; panel 7,087 vs 7,090 NEAR.

(b) The production run against the compat run:
```
python -B tools/diff_runs.py --ref <scratch folder>/configs/k562_compat.yaml --new <scratch folder>/configs/k562_production.yaml
```
Writes `reports/EXPECTED_DIFFERENCES_vs_<compat run>.md` into the production run. The table says what to expect; any row marked `FAIL` is a finding.

| item | compat run (a) = 21 Oct | production run (b) | traced to |
|---|---|---|---|
| cells, survivors, split of every cell, split JSON, control assignment, escaper thresholds, QC variable columns | 1,989,578 in; 1,783,992 survivors; train 6720 / val 800 / test 1598 | identical | none: they come before any production default |
| Phase 5 gene mask | 8,247 genes | the same unless cell-cycle genes had been rescued by gene QC | no cell-cycle rescue |
| panel size | 7,087 (stacked above 5,000) | 5,000 exactly | node budget (budget), no cell-cycle rescue |
| training targets as panel nodes | 4,557 | up to 3,500 target nodes in all (held-out measurable about 1,846 and protected genes always nodes) | node budget |
| row-only perturbations (cells kept, target not a node) | none (policy all) | listed in `manifest/perturbation_nodes.tsv` with the reason | node budget |
| stacked ghosts | 0 (97 cell-cycle genes were forced in the budget variant) | 0; no cell-cycle special case | no cell-cycle rescue |
| hvgraw rows | | the same cells and counts on the common genes; fewer columns | node budget |
| norm, metacell, train_hybrid values | | differ (library size and embedding are computed on the panel) | node budget |
| `train_hybrid_lognorm` | not written | written (parts) | log-normalized training set |
| names | `_p9` | `_norm` with `_p9` as hard-link aliases (`manifest/aliases.tsv`) | output names |
| merged split files | written (cap 300 GB) | not written; `python -B spore_xl.py merge --config ...` when wanted | optional merge |
| report | v3.0 sections | adds the escaper rule counts (escaper response score), the confounding table, variance explained by batch, donor and perturbation, the cell-cycle diagnostic (diagnostics), the node table (node budget) and the phase map (output names) | output names-diagnostics |
| time | plan 812 s, reduce1 605 s, finalize 1,679 s | plan faster with `plan_workers`, reduce1 faster (vectorized merge), finalize without the merge | optional merge |

Disk of each run: `du -sh` of the run folder and of `<scratch folder>/k562_gwps_v4compat_work` (and `k562_gwps_v4_work`).


## 5. CD4 Rest: the intake against the verified tally (CD4-specific)

The verified tally (`gse314342_merged_targets.tsv` and `gse314342_merged_summary.tsv`, made from the raw files on 7 Oct, min_umi 3) is the reference for the CD4 intake. Every run reports a generic
intake summary (members, barcodes, cells with 0 / 1 / 2 / 3+ targets, controls, labels; `reports/REPORT.md`, `reports/intake_members_summary.tsv`), which shows a broken intake without the tally.
The tally check itself is a developer check, run after the intake:

```
python -B tests/validate_intake_vs_tally.py --config <cd4 run config> --tally <gse314342_merged_targets.tsv> --summary <gse314342_merged_summary.tsv> --group Rest --min-umi 3 --out <new tsv>
```
Expected: five lines `PASS` (raw single-target cells per target, cells carrying each target, normalized labels with the JUN and control pair formulas, JUN pair cells, the totals row: 94 samples,
11,491,456 cells, 11,303,972 with any guide, 1,409,317 / 7,679,211 / 1,803,088 / 599,840 with 0 / 1 / 2 / 3+ targets).

## 6. Measuring and inspecting

* `python -B spore_xl.py measure --source <file.h5ad or tar> [--match _Rest_] --out <new folder> --n-blocks 128 --n-perts 11500 --panel-genes 5000 --cpus 16 --max-ram-gb 110 --workers 1`: header-only
  projection of RAM and disk per stage and the recommended workers.
* `python -B tools/inspect_source_header.py <file.h5ad> [--pert-col gene] [--control non-targeting] [--expect-cells N] [--expect-genes N]`: a read-only look at an `.h5ad` (matrix format, obs and
  var columns, label and control counts).
* `python -B tools/memory_calibration.py --config <run config> [--repo <SPORE-XL repository>] [--out <new tsv>]`: pairs the measured peak RSS of every unit (`reports/memory_and_time.tsv`) with its input
  size and prints, per stage, how much the pre-flight estimate overestimates and the expansion that would cover every measured point with a safety factor of 1.5. The pre-flight (`src/engine/preflight.py`)
  is calibrated from this table.
* `python -B tools/identity_compare.py --a <run folder A> --b <run folder B> --out <new prefix>`: sha256 of every file of two run folders (standard library only; also run the folders of `xl.work_dir` with `--pair <A> <B>` when they sit outside the run folders). Writes `<prefix>_files.tsv` (identical / differs / only_a / only_b per file) and `<prefix>_diffs.txt` (first differing lines of each differing text file; `path_only` = identical once the two run folder paths are normalized). Exit code 1 unless every file is identical or path_only. Use it to compare two builds of the repository or two runs of the same config (a changed layout, a new cluster, a new Python environment): everything that must not change is `identical`. It is how the SPORE-XL layout was proved to change no output.
* `python -B tools/diff_runs.py --ref <config of run A> --new <config of run B>` and `python -B tools/crosscheck_run.py --config <run config>`: see section 4.

## 7. Stage-level commands

`python -B spore_xl.py <stage> --config <yaml> [--local --workers N | --index N]` runs one stage (`plan`, `map1`, `reduce1`, `map2`, `gather`, `reduce2`, `hvg`, `map3`, `metacell_prep`, `metacell`, `finalize`,
`report`, `merge`); `--index N` runs one unit. `spore_xl.py branch --from-config <A> --config <B> --from-stage <stage>` starts a new run folder that reuses the finished stages of another run;
`spore_xl.py cleanup --config <yaml> --after <stage>` is the opt-in removal of intermediates. `python -B src/engine/verify.py <config>` is the verification that `run` performs at its end.

## 8. A multi-job mode was prototyped and withdrawn

`spore_xl.py submit` (one SLURM job per stage, chained with `afterok`) is not part of v4.0. SPORE-XL bounds the memory of every task, so one node suffices, and the chain cost more than it saved: queue waits
between stages, one resource guess per stage, and on the CD4 run a reduce1 job killed with its proposed 4 GB stalled the whole chain overnight. The prototype code is not in this repository.
