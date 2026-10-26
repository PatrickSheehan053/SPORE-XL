# User guide

This guide takes a screen from raw counts to a verified, split dataset. It assumes you can run Python from a terminal. [Cluster setup](HPC_SETUP.md) covers the SLURM-specific steps.

## 1. Install

SPORE-XL needs Python 3.12. Create a fresh virtual environment and install the pinned requirements, which reproduce the environment of the showcase runs exactly.

```
python3.12 -m venv spore_xl_env
source spore_xl_env/bin/activate          # Windows: spore_xl_env\Scripts\activate
pip install --only-binary=:all: -r requirements_pinned.txt
```

Check the installation with the quick tests, which take a few minutes:

```
python -B tests/run_all_tests.py --tag quick --logs <log folder> --scratch <scratch folder> --quick
```

Every line should end in `exit 0`.

## 2. Prepare the input

SPORE-XL reads one `.h5ad` file with raw integer UMI counts (cells as rows) and a column in `obs` that names each cell's perturbation. The label of a perturbed cell is its target gene symbol; control cells share one label such as `non-targeting`. The matrix can be sparse (CSR) or dense.

If your data is in another format, `convert` writes the `.h5ad` and repairs common label problems:

```
python spore_xl.py convert --recipe <recipe.yaml> --out <new folder>
```

Example recipes are in `src/intake/recipes/`, and [INTAKE.md](INTAKE.md) describes every format.

## 3. Write the config

Copy `configs/template.yaml` to a new file and set these values. Everything else can stay at its default, and the [config reference](CONFIG_REFERENCE.md) explains each one.

| key | set it to |
|---|---|
| `paths.project_root` | a new folder for this run's outputs |
| `dataset.name` | a short name; it prefixes every output file |
| `dataset.perturbation_col` | the `obs` column with the perturbation labels |
| `dataset.control_label` | the label of your control cells |
| `dataset.gene_id_format` | `symbol`, `ensembl` or `entrez`, whichever your gene names are |
| `xl.source.h5ad` | your input file |
| `xl.work_dir` | a folder on fast storage for working files |
| `phase7_splits.test_n`, `val_n` | how many perturbations to hold out for testing and validation |
| `phase8_hvg.protected_gene_list` | optional: a list of genes to keep in the panel, such as the files in `protected_genes/` |

The configs of the four showcase screens are in `configs/datasets/` and make good starting points for similar screens.

## 4. Run

On a workstation or laptop:

```
python spore_xl.py run --config my_run.yaml
```

SPORE-XL finds the CPUs and memory available and decides how many tasks to run at once. If other programs will share the machine during the run, give it limits, because it cannot see memory that another program claims later:

```
python spore_xl.py run --config my_run.yaml --workers 6 --max-ram-gb 32
```

On a SLURM cluster, submit the same run as one job. Always pass `--output` with a log folder, or SLURM writes its log into the folder you submitted from:

```
sbatch --output=<log folder>/slurm_%j.out slurm/spore_xl.sbatch my_run.yaml
```

If a run stops for any reason (a time limit, a crash, Ctrl+C), run the same command again. Finished work is skipped, half-written work is moved aside and redone, and nothing is deleted. The result is the same as an uninterrupted run.

A healthy run ends with:

```
[run] DONE: 15 verification checks PASS
```

## 5. Read the results

The run folder holds:

| folder or file | contents |
|---|---|
| `reports/REPORT.md` | the run summary, the cell waterfall, the gene panel, escaper statistics, batch and donor diagnostics, timings and the figure list |
| `reports/VERIFICATION.md` | the result of every verification check |
| `reports/COVERAGE_REPORT.md` | held-out target coverage; ends with GRAPH-READY when the output is ready for inference |
| `reports/WATERFALL.md` | every cell that entered and why each removed cell left |
| `figures/` | eight figures, from input sparsity to normalization |
| `splits/` | the split file, the control assignment and the gene panel |
| `manifest/perturbation_nodes.tsv` | every perturbation, its cell count, and whether its target is a panel gene |
| `processed/` | the data |

Before using a run, check three things. `VERIFICATION.md` must show every check as PASS. The waterfall must end with RECONCILES, which means the cells kept plus the cells removed equal the cells that entered. And the report's batch confounding section names any perturbation whose cells came from different batches than the controls; consider leaving those out of an evaluation.

## 6. Load the outputs

The data is written as one `.h5ad` per perturbation block, in one folder per split and output type:

| folder | contents |
|---|---|
| `processed/<name>_<split>_hvgraw/` | raw counts on the gene panel, for `train`, `val` and `test` |
| `processed/<name>_<split>_norm/` | the same cells, library-size normalized and log-transformed |
| `processed/<name>_allsplits_metacell_ctrlpreserved/` | metacells for every split, with control cells kept as single cells |
| `processed/<name>_train_hybrid/` | the training set of metacells plus single control cells, raw |
| `processed/<name>_train_hybrid_lognorm/` | the same, log-normalized |

To load one split as a single AnnData:

```python
import glob
import anndata as ad

parts = sorted(glob.glob("processed/<name>_test_hvgraw/part_*.h5ad"))
test = ad.concat([ad.read_h5ad(path) for path in parts])
```

`python spore_xl.py merge --config my_run.yaml` writes merged files instead, when they fit in memory.

## 7. Change the defaults

The changes users make most often:

| goal | setting |
|---|---|
| a larger or smaller held-out set | `phase7_splits.test_n`, `val_n` |
| a different split | `phase7_splits.random_seed` |
| a larger gene panel | `phase8_hvg.n_top_genes` |
| always keep certain genes | `phase8_hvg.protected_gene_list` |
| catch escapers of low-expressed targets | `phase5_escaper_filtering.escaper_score: union` |
| keep perturbations with fewer cells | `phase5_escaper_filtering.min_cells_per_perturbation` |
| stricter or looser cell QC | `phase2_cell_triage` limits |

Use a new `project_root` for every changed run. SPORE-XL never overwrites a finished run, so a changed config pointed at an old folder stops with an error rather than mixing two runs.
