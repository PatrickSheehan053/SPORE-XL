# Config reference

A SPORE-XL run is described by one YAML file. This page lists every setting SPORE-XL reads, its default, and what changing it does. Start from `configs/template.yaml`, which sets the values most runs use; the files in `configs/datasets/` are the configs of the four showcase runs.

Keys you leave out take the default in the **default** column. Two settings have a recommended value that differs from the built-in default (`phase8_hvg.n_top_genes` and the split sizes), and the template already sets them.

A config may hold placeholders written as `<NAME>`. SPORE-XL refuses to start while any placeholder remains outside a comment, so a half-edited config cannot run by accident.

## paths

| key | default | what it does |
|---|---|---|
| `project_root` | required | the run folder. Every output goes here. Use a new folder per run; SPORE-XL never overwrites a finished file. |
| `processed_dir` | `processed` | subfolder for the train, validation and test data |
| `splits_dir` | `splits` | subfolder for the split file, control assignment and gene panel |
| `figures_dir` | `figures` | subfolder for the figures |
| `log_dir` | `logs` | subfolder for the console logs |
| `raw_h5ad` | `''` | not used by SPORE-XL; the input is `xl.source` |

## dataset

| key | default | what it does |
|---|---|---|
| `name` | required | prefix of every output file |
| `organism` | `human` | used to recognize Ensembl IDs |
| `perturbation_col` | required | the `obs` column holding each cell's perturbation label (the target gene symbol, controls as `control_label`) |
| `control_label` | required | the label of control cells, for example `non-targeting` |
| `perturbation_type` | `CRISPRi` | sets the escaper direction to knockdown. Activation screens (`CRISPRa`) and multi-target labels are refused in this version. |
| `gene_id_format` | detected | `symbol`, `ensembl` or `entrez`. Ensembl and Entrez IDs are translated to symbols once, at the start of the run. |

## phase2_cell_triage (cell QC)

A cell is removed when any of these limits fails. The defaults suit standard 10x Perturb-seq data.

| key | default | what it does |
|---|---|---|
| `mt_threshold` | `0.20` | maximum fraction of a cell's counts from mitochondrial genes |
| `mt_genes` | the 13 human `MT-` genes | the genes counted as mitochondrial |
| `ribo_prefixes` | `[RPL, RPS]` | ribosomal gene prefixes; ribosomal perturbation targets are exempt from ribosomal filtering |
| `min_genes_per_cell` | `200` | minimum number of genes detected |
| `max_genes_per_cell` | `10000` | maximum number of genes detected |
| `min_counts_per_cell` | `500` | minimum total UMI count |
| `max_counts_per_cell` | `80000` | maximum total UMI count |

## phase5_escaper_filtering (escaper filter)

An escaper is a cell that carries a guide but whose target gene was not knocked down. Rule 1 removes a cell when its target gene's expression is above the `escaper_percentile` of that gene in the control cells.

| key | default | what it does |
|---|---|---|
| `escaper_percentile` | `10` | the control percentile a knocked-down cell must fall to or below |
| `min_cells_per_perturbation` | `50` | perturbations with fewer cells left after the escaper filter are dropped |
| `efficiency_threshold` | `0.50` | perturbations whose knockdown efficiency is below this are flagged in the report; they are not removed |
| `direction` | `auto` | `auto` derives the direction from `perturbation_type` |
| `escaper_score` | `report` | a second, response-based escaper rule (rule 2, in the style of Mixscape). `off` skips it, `report` counts it in the report without removing cells, `union` removes cells flagged by either rule, `replace` uses rule 2 only. |

Rule 1 cannot see a knockdown of a gene that is not expressed in controls. The report lists the perturbations where rule 2 flags the most cells that rule 1 misses, so you can judge whether to switch to `union`.

## phase6_gene_triage (gene QC)

| key | default | what it does |
|---|---|---|
| `min_cells_expressing` | `10` | genes detected in fewer cells are removed |
| `pct_filter` | `0.01` | a gene detected in fewer than this fraction of cells is removed when its mean count is also below `mean_umi_threshold` |
| `mean_umi_threshold` | `0.25` | see `pct_filter` |
| `rescue_perturbation_targets` | `true` | keeps every gene that is a perturbation target, even when it fails the filters above |

## phase7_splits (train, validation and test)

Splits are made by perturbation. All cells of a perturbation go to one split, and a held-out perturbation's target gene is never the target of a training perturbation. Perturbations are ranked by how strongly they shift expression away from the controls, grouped into `stratify_bins` bins, and each split samples from every bin, so the three splits contain the same mix of strong and weak perturbations.

| key | default | what it does |
|---|---|---|
| `mode` | `zero_shot` | the only mode in this version |
| `test_n` | `20` | number of test perturbations. Set it for your screen; the showcase runs hold out 20 to 27% of perturbations for testing. |
| `val_n` | `20` | number of validation perturbations, 10 to 14% in the showcase runs |
| `test_ratio`, `val_ratio` | `0.15`, `0.10` | used only when `test_n` / `val_n` are absent |
| `random_seed` | `42` | changes which perturbations land in which split; the same seed gives the same split everywhere |
| `stratify_bins` | `4` | number of response-strength bins |

Control cells are divided across all three splits. The control assignment is written to `splits/<name>_control_assignment.parquet`.

## phase8_hvg (gene panel)

| key | default | what it does |
|---|---|---|
| `n_top_genes` | `1000` (template: `5000`) | size of the gene panel |
| `method` | `seurat_v3` | highly variable gene method (scanpy), computed on training cells only |
| `protected_gene_list` | none | a JSON list of genes always included in the panel when they are measured, for example transcription factors of interest. Files for the four showcase cell types are in `protected_genes/`. |
| `protected_expression_gate` | `false` | when `true`, protected genes detected in fewer than `protected_expression_min_fraction` of control cells are left out; the gated list is written to the run folder and your file is never changed |
| `protected_expression_min_fraction` | `0.01` | see above |
| `target_node_policy` | `budget` | `budget` keeps the panel at exactly `n_top_genes` and gives perturbation targets at most `max_target_nodes` of it; `all` adds every training target on top of the panel (the SPORE v3.0 behavior) |
| `max_target_nodes` | `0.7 × n_top_genes` | the most panel genes perturbation targets may take under `budget` |

The targets of every validation and test perturbation are always in the panel when they were measured, so a model can be scored on them. The report's node table states which perturbations became panel genes and why.

## phase9_normalization

| key | default | what it does |
|---|---|---|
| `target_sum` | `10000` | each cell is scaled to this total count |
| `log_transform` | `true` | applies `log1p` after scaling |
| `method` | `lognorm` | the only method in this version |

The raw counts are always kept alongside (`hvgraw` outputs), so you can normalize differently later.

## phase12_metacell

| key | default | what it does |
|---|---|---|
| `enabled` | `true` | builds metacells |
| `mbk_k` | `2` | each perturbation's cells are grouped into about `cells / mbk_k` metacells with MiniBatchKMeans; a metacell is the mean of the raw counts of its cells |
| `seed` | `42` | k-means seed |

Control cells stay single cells. Every metacell comes from one perturbation and so from one split.

## diagnostics

These change nothing in the output; they add tables to the report.

| key | default | what it does |
|---|---|---|
| `batch_col` | found by name (`gem_group`, `batch`, ...) | the batch column for the confounding check |
| `donor_col` | found by name | the donor column for the variance diagnostic |
| `confound_tvd_warn` | `0.5` | a perturbation is flagged as confounded with batch when its batch mix differs from the controls' by at least this total variation distance... |
| `confound_alpha` | `0.001` | ...and the difference is beyond what random sampling of the same number of control cells produces |

## xl (the engine)

| key | default | what it does |
|---|---|---|
| `source.kind` | required | `h5ad_rows` (a sparse CSR or dense `.h5ad`, cut into shards of consecutive rows), `h5ad_obs` (one shard per value of the `obs` column named in `source.obs_key`) or `recipe` (a 10x Flex tar) |
| `source.h5ad` | required for `h5ad_*` | the input file with raw integer counts. It is read in place and never changed. |
| `source.shard_rows` | `40000` | cells per shard; about 5 GB of memory per task at 3,000 stored values per cell |
| `work_dir` | `_xl_work` | working files (shards, blocks). Put it on fast scratch storage. The run checks free disk space before it starts and logs how much it projects to need. |
| `n_blocks` | `64` | number of perturbation blocks. Every perturbation is in exactly one block. |
| `control_block_rows` | `120000` | control cells per control block |
| `hvg_subsample_cells` | `100000` | training cells sampled (stratified) for the gene panel |
| `memory_budget_gb` | `24` | a task whose projected memory exceeds this is refused before it starts |
| `plan_workers` | `1` | parallel workers for cutting the input into shards |
| `merge_split_outputs` | `never` | `never` writes per-block parts only, `auto` merges when the merged file stays under `merge_cap_gb`, `always` merges. `python spore_xl.py merge --config <yaml>` merges later. |
| `merge_cap_gb` | `40` | size limit for `auto` |
| `write_train_hybrid_lognorm` | `true` | also writes the log-normalized training metacell set |
| `output_aliases` | `true` | also writes SPORE v3.0 file names for the normalized outputs |
| `compat` | absent | `v3` reproduces SPORE v3.0 exactly (used by the tests); leave it out for the defaults described on this page |

## Command line

| command | what it does |
|---|---|
| `python spore_xl.py run --config my.yaml` | the whole run, then verification |
| `--workers N` | the most tasks run at once (default: all CPUs) |
| `--max-ram-gb G` | the memory budget for the run (default: set from the machine's total memory). Set it when other programs share the machine, because SPORE-XL cannot see memory they will use later. |
| `python spore_xl.py convert --recipe <recipe.yaml> --out <folder>` | turns h5ad, 10x h5 or mtx, loom, csv / tsv or R-exported HDF5 into the sparse `.h5ad` SPORE-XL reads |
| `python spore_xl.py merge --config my.yaml` | writes merged train, validation and test files from the parts |
| `sbatch --output=<log folder>/slurm_%j.out slurm/spore_xl.sbatch my.yaml` | the whole run as one SLURM job |
| `SPORE_NO_BANNER=1` | environment variable that turns off the start-up banner |
