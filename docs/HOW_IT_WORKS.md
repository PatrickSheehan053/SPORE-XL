# How SPORE-XL v4.0 works

SPORE-XL v4.0 turns a raw Perturb-seq screen into a clean, leakage-free dataset by the same rules as SPORE v3.0, but never holds the
matrix in memory. The v3.0 explanation of what each cleaning step does and why still holds (see the `v3-archive` branch, `docs/HOW_IT_WORKS.md`);
this file describes what is new: how the work is cut up, what is written where, the production defaults and the diagnostics.

## 1. The one idea: map, reduce, map

Every global decision SPORE makes is a function of a small set of statistics that add up across cells:

* fixed cell QC thresholds need nothing global;
* the escaper threshold of a target gene is a percentile of that gene's counts in control cells, which is exactly recoverable from the summed
  integer histogram of those counts (`src/engine/exact_percentile.py` reproduces `numpy.percentile` bit for bit);
* gene QC needs per-gene detection counts and sums;
* the split needs per-perturbation mean-shift scores (a float32 carried accumulator reproduces scipy's mean bit for bit, so no rank changes);
* the HVG panel is selected on a stratified 100,000-cell train subsample, as in v3.0.

So the pipeline is per-unit jobs that emit statistics, a small job that makes the decision, then per-unit jobs that apply it. The only places
that see many cells at once are one shard, one perturbation block, or the subsample.

| stage (CLI name) | unit | phase | what it does |
|---|---|---|---|
| `intake_list`, `intake` | tar member | 1 | (10x Flex tar recipe only) convert each raw member to a shard |
| `plan` | whole input | 1-2 | shards the input (`xl.plan_workers` in parallel), harmonizes gene IDs once, label table, ribosomal exemption list, control-block cut points; **refuses** multi-target, CRISPRa and normalized input before writing |
| `map1` | shard | 3 | cell QC, per-gene sums, control histograms for the target genes, control reference sample (escaper score) |
| `reduce1` | one job | 4 | exact escaper thresholds and knockdown depth per target gene (vectorized merge of all shard histograms) |
| `map2` | shard | 4 | applies the escaper filter, writes the kept cells into perturbation blocks (flagged cells travel too when `escaper_score` is on) |
| `gather` | block | 4-5 | concatenates a block in global order, applies the rule(s), drops undersized perturbations, gene statistics, per-perturbation means |
| `reduce2` | one job | 5-6 | gene mask, mean shift, **the** gene-disjoint split (made once, here), control assignment, confounding table |
| `hvg` | one job | 7 | subsample draw, HVG selection with the node policy, node manifest, diagnostics |
| `map3` | block | 8 | gene mask + panel, routes cells by the split lookup, writes `hvgraw` and `norm` parts |
| `metacell_prep`, `metacell` | job, block | 9 | control-preserving MiniBatchKMeans metacells per perturbation |
| `finalize` | one job | 9-10 | train hybrid (+ lognorm), aliases, coverage report, waterfall, optional merge |
| `report` | one job | 10 | figures and `REPORT.md` |
| `verify` | one job | 10 | hard gates (below) |
| `merge` (optional) | one job | | merged split files with `concat_on_disk` |

`run` executes every stage on one machine (`--workers N` for a process pool, always spawn, one BLAS thread per worker); `spore_xl.py <stage>
--index N` runs one unit of a stage. A finished unit has a `.done` marker with the checksum of its files and is
skipped on re-run; a unit cut off in the middle is moved aside (never deleted) into `_xl_work/_incomplete/` and recomputed, so the same command
always resumes.

## 2. Two data layouts

1. **Cell shards** (one raw sample file, a row range, or an obs value). Natural for intake and QC; they carry the batch structure.
2. **Perturbation blocks** (`block = blake2b(label) mod n_blocks`; controls are cut into row-capped control blocks). Every cell of a perturbation
   lives in exactly one block, so mean shift, normalization per split and metacells never read across blocks. Splits are decided by perturbation, so each
   block's cells go wholly to train, val or test.

`xl_global_index` is the cell's row in the source (or, for tar sources, in the sorted-member concatenation); every output row carries it, so SPORE's
row order can be restored from the parts.

## 3. Leakage firewall

The split is computed in exactly one place (`reduce2`) from the final perturbation list. Later stages only look it up. Intake keeps only cells with exactly
one target. Train-only statistics (HVG subsample, diagnostics) use train cells. `verify` checks: label sets of the parts equal the split JSON, the three
sets are disjoint and cover the survivors, cell counts per perturbation match, every control has exactly one split, no cell index appears twice,
`train_hybrid` has no val/test unit, every metacell comes from one perturbation, variable names are identical everywhere, `p9`/`norm` parts hold the
cells of `hvgraw`, aliases and the lognorm hybrid are consistent.

## 4. Outputs

```
<run>/
  config/      run yaml, environment.txt (python and library versions), resolved settings
  manifest/    shards.tsv, global_obs.parquet (+ _final: survived, split), labels.tsv, perturbation_nodes.tsv, aliases.tsv
  stats/       per-unit statistics, reduce outputs (escaper_thresholds.tsv, gene_stats.tsv, mean_shift.tsv, confounding.tsv, hvg diagnostics.json)
  splits/      {name}_split_indices.json (SPORE format), {name}_control_assignment.parquet, {name}_hvg_panel.txt
  processed/   {name}_{train,val,test}_hvgraw/  (raw counts on the panel)       part_NNNN.h5ad per block
               {name}_{train,val,test}_norm/    (normalize_total + log1p; alias dir _p9)
               {name}_allsplits_metacell_ctrlpreserved/, {name}_train_hybrid/, {name}_train_hybrid_lognorm/
  reports/     REPORT.md, COVERAGE_REPORT.md (GRAPH-READY), WATERFALL.md, VERIFICATION.md, memory_and_time.tsv, escaper_rules.tsv
  figures/     content-named PNGs + v3.0-named copies
  logs/        one console log per command
  _xl_work/    shards, QC shards, buckets, blocks (kept; removal is your decision)
```

## 5. Production defaults and the compatibility setting

`xl.compat: v3` reproduces SPORE v3.0; leaving it out gives the production defaults. Only keys missing from your config are filled in.

| key | production | `xl.compat: v3` | why |
|---|---|---|---|
| `phase8_hvg.target_node_policy` | `budget` | `all` | panel = `n_top_genes` exactly |
| `phase8_hvg.max_target_nodes` | 0.7 x `n_top_genes` | n/a | node budget |
| `phase8_hvg.rescue_cell_cycle_genes`, `phase6_gene_triage.rescue_cell_cycle_genes` | false | true | ghost rescue served only the removed phase 10 |
| `phase8_hvg.tie_break` | `stable` | `spore_set` | no `PYTHONHASHSEED` dependence |
| `phase5_escaper_filtering.escaper_score` | `report` | `off` | count the response-score rule, apply rule 1 |
| `xl.output_aliases` | true | false | v3.0 names as hard links |
| `xl.write_train_hybrid_lognorm` | true | false | |
| `xl.merge_split_outputs` | `never` | `auto` | merged files on demand |

### The node budget

Under `all`, every training target gene becomes a panel gene, so a genome-wide screen's panel grows far above `n_top_genes` (K562: 7,087 for 5,000). Under
`budget`: held-out (val + test) targets and protected genes are always panel genes; training targets are added by rank (more cells after the escaper
filter, deeper knockdown, more DEGs; average of the three ranks, ties by name) until `max_target_nodes` target nodes or `n_top_genes` genes; the rest of the slots
are the most variable genes. A perturbation whose target is not a node keeps its cells as rows: `manifest/perturbation_nodes.tsv` lists, for every perturbation, the
split, the cell count, node or row-only, and why.

## 6. Diagnostics in the report

* **Confounding:** for each perturbation the total variation distance between its batch mix and the controls' (0 same, 1 disjoint); a warning when the distance is at least `diagnostics.confound_tvd_warn` (0.5), Cramer's V for the whole table.
* **Variance explained** by batch, donor and perturbation on the train HVG subsample (log-normalized panel genes): one-way R2 and R2 adjusted for the number of groups. Batch and donor columns are found by name (`gem_group`, `batch`, `donor`, ...) or set with `diagnostics.batch_col` / `donor_col`.
* **Cell-cycle score** (S and G2M, scanpy, fixed seed): score variance, phase fractions, and how much perturbation and batch explain.

## 7. Intake

`spore_xl.py convert --recipe src/intake/recipes/<dataset>.convert.yaml --out <new folder>` reads a raw format in windows of rows, applies the recipe (filter, label column,
control spellings, label repairs, batch / donor columns, per-perturbation cap with a seed), checks that the matrix is raw counts, and writes a sparse CSR `.h5ad`
that `xl.source.kind: h5ad_rows` reads, plus `provenance.json` and `intake_report.md`. See `src/intake/recipes/` for a worked example per format and `src/intake/R/` for the R export.

## 8. Cluster use: one job per run

Every task of SPORE-XL has bounded memory (one shard, one perturbation block or the 100,000-cell HVG subsample), so a single node always suffices. A cluster run is one sbatch job that runs `spore_xl.py all-local --workers N --max-ram-gb <90 percent of the job's memory>` and then `src/engine/verify.py` (`slurm/run_all_local.sbatch`; `slurm/cd4_run_hpc1.sbatch` adds the CD4 intake validation and the protected-list gate in front). Stages run one after the other inside the job, the workers of a stage share the node, and the per-stage pre-flight caps the concurrency to the RAM budget. Resubmitting the same job resumes. A multi-job mode (one SLURM job per stage, chained) was prototyped and withdrawn: it added queue waits between stages and one resource guess per stage, and a stage that was given too little memory stalled the whole chain.

## 9. What is not here

Phase 3 (ambient), 4 (doublets), 10 (PCA, Harmony, cell-cycle regression) and 11 (cell-line separation) of v3.0 are not in the core; their modules are not shipped here; they are on the tag `v3.0` and the branch `v3-archive`.
Combinatorial perturbations and CRISPRa are refused. The response-universe curves are deferred to Experiment C.
