# FAQ

## Getting started

### What does SPORE-XL need as input?

A single-cell CRISPR screen with raw integer UMI counts, one column in the cell metadata that names each cell's perturbation (the target gene symbol), and a label for control cells such as `non-targeting`. SPORE-XL reads a sparse or dense `.h5ad` directly. For 10x h5 or mtx, loom, csv / tsv or an HDF5 file exported from R, `python spore_xl.py convert` writes the `.h5ad` first ([config reference](CONFIG_REFERENCE.md#command-line)).

### My matrix is already normalized. Can I use it?

No. Every decision SPORE-XL makes, from cell QC to the escaper filter to the gene panel, is made on raw counts, and normalized values would silently change all of them. SPORE-XL checks the input and refuses a normalized matrix before it writes anything. Go back to the raw counts from the original data release.

### Do I need a computing cluster?

No. SPORE-XL holds one shard or one perturbation block in memory at a time, so a screen's size sets how long a run takes, not how much memory it needs. The RPE1 showcase screen (247,914 cells) ran on a six-core laptop with the same result as on the cluster. The memory a run needs is set by its largest single task, which was 13.8 to 26.9 GB across the four showcase screens; the seven-million-cell CD4 screen needed 17.5 GB.

### How long does a run take?

The RPE1 and H1 screens (about 220,000 to 250,000 cells) took 8 to 10 minutes as a SLURM job on one 16-CPU node. Time grows with the number of cells; the [README](../README.md#four-screens-four-cell-types) lists all four screens.

### How much disk space does a run need?

The run estimates its working space before it starts, logs the estimate, and stops if the disk is too small. The RPE1 run projected 31 GB of working files for a 1.2 GB input. Point `xl.work_dir` at fast scratch storage.

### My laptop ran out of memory. What happened?

SPORE-XL sets its memory budget from the machine's total memory when the run starts. It cannot know that you will open another memory-heavy program while it runs. Pass `--max-ram-gb` with the memory you can actually spare, and `--workers` to limit how many tasks run at once. An interrupted run resumes where it stopped when you rerun the same command, and the result is the same as an uninterrupted run.

## Splits and leakage

### How do I know the test set does not leak into training?

SPORE-XL splits by perturbation. Every cell of a perturbation goes to exactly one of train, validation and test, so a model never sees a cell of a test perturbation during training. The run then proves it. The verification step checks, among other things, that the three label sets are disjoint, that every perturbation and every control cell sits in exactly one split, that no cell appears twice, and that the training metacell set contains no validation or test unit. A failed check stops the run with an error naming the check, and `reports/VERIFICATION.md` records the result of every check.

### Can I choose the split sizes?

Yes. Set the number of test and validation perturbations with `phase7_splits.test_n` and `val_n`, and the seed with `random_seed`. The same seed gives the same split on any machine. Perturbations are stratified by response strength, so strong and weak perturbations are spread evenly across the three sets.

### Why are control cells in all three splits?

Every split needs controls as its reference, for example to measure how far a test perturbation moved expression. SPORE-XL divides the control cells between the splits so each control cell belongs to exactly one split, and the assignment is written to `splits/<name>_control_assignment.parquet`.

### Why are the targets of my test perturbations in the gene panel?

A model is scored on how it predicts the response to a held-out perturbation, and that requires the targeted gene to be measured in the output. SPORE-XL always carries the targets of validation and test perturbations into the panel when they were measured. This uses no information from the test cells; only the list of held-out target genes.

### Some perturbations are listed as "row-only". What does that mean?

Under the default `budget` policy, perturbation targets may take at most 70% of the gene panel, so the panel keeps room for highly variable genes. A training perturbation whose target did not fit is kept with all its cells (a row-only perturbation), and only its target gene is missing from the panel's columns. `manifest/perturbation_nodes.tsv` lists every perturbation with the reason.

## Cleaning

### What is an escaper, and why remove it?

An escaper is a cell that received a guide but whose target gene was not knocked down. Its label says one perturbation, its expression says control, and a model trained on it learns a weaker effect than the real one. SPORE-XL removes a cell when its target gene's expression is above the 10th percentile of that gene in the control cells. Across the four showcase screens this removed 6 to 9% of perturbed cells.

### Can the escaper filter miss anything?

Yes, for perturbations whose target gene is barely expressed in controls. If the 10th percentile of the target in controls is zero, a knocked-down cell and an escaper look the same. SPORE-XL computes a second, response-based score for every cell and reports, per perturbation, how many cells it would flag that the first rule missed. Set `phase5_escaper_filtering.escaper_score: union` to remove cells flagged by either rule.

### Why did a perturbation disappear?

A perturbation with fewer than 50 cells left after the escaper filter is dropped (`min_cells_per_perturbation`). The cell waterfall in the report counts every cell removed at each step, and the per-perturbation table records why each perturbation left.

### Does SPORE-XL correct batch effects?

No. It measures them. The report flags perturbations whose cells come from a different mix of batches than the controls, using a test calibrated against random sampling so that small perturbations are not flagged by chance, and it reports the share of variance explained by perturbation, batch and donor. Batch correction, cell-cycle regression, ambient RNA removal and doublet removal are not part of SPORE-XL; SPORE v3.0 (tag `v3.0`) still has them.

### What does SPORE-XL do with screens that break other pipelines?

The test suite includes eleven synthetic screens built to break preprocessing, each with an answer key a correct run must reproduce:

| synthetic screen | what is difficult about it |
|---|---|
| genome-wide | thousands of perturbations with few cells each |
| low-expressed targets | knockdowns invisible to the target-expression rule |
| batch-confounded | perturbations recorded in batches the controls are not |
| variable depth | sequencing depth that differs strongly between cells |
| multiple donors | donor structure across all perturbations |
| tiny groups | perturbations at the edge of the minimum cell count |
| few controls / many controls | control fractions far below and far above normal |
| messy labels | inconsistent perturbation and control labels |
| duplicate symbols | gene symbols that repeat in the matrix |
| dense matrix | a dense instead of a sparse count matrix |
| normalized matrix | an input that is not raw counts |

The final code passed 131 of 135 answer-key checks. The normalized screen was refused before anything was written, the dense screen gave the same result as its sparse copy, and the batch-confounded perturbations were flagged. Two of the four remaining checks concern messy labels and duplicate symbols, which `convert` repairs at intake rather than the core pipeline; one is the low-expressed-target limit described above; and one was an error in the test's own expectation.

## Results and reproducibility

### Will I get the same result on my machine?

On the same type of machine, yes, bit for bit: two runs of the same configuration on one cluster, one interactive and one as a SLURM job, gave identical outputs. Between operating systems, every cleaning and splitting result is identical, and the metacells differ by about 0.1% because the k-means step that builds them rounds differently on different processors ([RPE1 comparison](showcase/rpe1/IDENTITY.md)). The pinned requirements file reproduces the environment of the showcase runs exactly.

### Why does H1 need more memory than the much larger CD4 screen?

The largest task in a run is the gene-panel step, which works on a stratified subsample of 100,000 training cells across every gene that passed gene QC. H1 kept 14,145 genes and CD4 kept 12,499, so the H1 step was larger, regardless of how many cells either screen has in total.

### Where are my train, validation and test files?

Under `processed/`, as one `.h5ad` per perturbation block in one folder per split and output type, for example `processed/<name>_test_hvgraw/` for raw counts of the test cells on the gene panel. Each part loads with `anndata.read_h5ad` and carries every cell's perturbation in `obs`. The [user guide](USER_GUIDE.md#load-the-outputs) shows how to load a whole split. `python spore_xl.py merge --config <yaml>` writes single merged files when you want them, and `xl.merge_split_outputs: auto` merges automatically when the result stays under `xl.merge_cap_gb`.

### What does "GRAPH-READY" mean?

It is the last line of `reports/COVERAGE_REPORT.md` when every held-out perturbation that was measured has its target in the panel and the training set passes the leakage checks, so the output can go straight into network inference or model training.
