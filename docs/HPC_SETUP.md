# Running SPORE-XL on a cluster

SPORE-XL runs a whole screen as one job on one node. Every task holds at most one shard, one perturbation block or the 100,000-cell gene-panel subsample, so a single node is always enough, and SLURM only has to provide CPUs, memory and time.

## 1. Requirements

- Linux x86_64 with glibc 2.28 or newer (`ldd --version | head -1`)
- Python 3.12. If your cluster has no Python 3.12 module, `uv` installs one without root rights or conda (step 2).
- fast scratch storage for the working files; the run estimates the space it needs before it starts and stops if the disk is too small

## 2. Install

Choose two folders on your cluster: one for the code (`<repo folder>`, shared storage) and one for the Python environment (`<env folder>`, scratch is fine).

```
git clone https://github.com/PatrickSheehan053/SPORE-XL.git <repo folder>
```

If Python 3.12 is not available, install `uv` and let it fetch Python:

```
python -m pip install --user uv
~/.local/bin/uv python install 3.12.9
~/.local/bin/uv venv --python 3.12.9 <env folder>
~/.local/bin/uv pip install --python <env folder>/bin/python -r <repo folder>/requirements_pinned.txt
```

With Python 3.12 already available, a plain virtual environment works the same way:

```
python3.12 -m venv <env folder>
<env folder>/bin/pip install --only-binary=:all: -r <repo folder>/requirements_pinned.txt
```

The pinned versions are the ones every showcase run in the README used.

## 3. Check the installation

```
source <env folder>/bin/activate
cd <repo folder>
python -B tests/run_all_tests.py --tag install --logs <log folder> --scratch <scratch folder> --quick
```

Expected: 13 lines ending in `exit 0`, in a few minutes. `python -B tests/test_spore_api.py` additionally confirms that the SPORE phase files are byte-identical to the recorded originals.

## 4. Write the config

Start from `configs/template.yaml` or from the config of a similar screen in `configs/datasets/`. On a cluster, two settings matter most:

- `xl.work_dir`: put the working files on scratch storage. A 2-million-cell screen needs on the order of 200 GB of working space.
- `paths.project_root`: the run folder, on storage you keep. Use a new folder for every run.

## 5. Run as a SLURM job

`slurm/spore_xl.sbatch` runs one config as one job. Submit it from the repository folder, give the config as its argument, and always pass `--output`; without it SLURM writes the job log into the folder you submit from:

```
cd <repo folder>
sbatch --cpus-per-task=16 --mem=64G --output=<log folder>/slurm_%j.out slurm/spore_xl.sbatch my_run.yaml
```

The script activates the environment, sets the environment variables every run needs, gives `run` the job's CPUs and 90% of its memory, and runs the 15 verification checks at the end. Before the first job, open the script and edit the **site settings** block at the top: the default environment folder (`VENV`) and the partition (`#SBATCH --partition`). Both can also be set per job:

```
sbatch --export=ALL,VENV=<env folder> --partition=<partition> --output=<log folder>/slurm_%j.out slurm/spore_xl.sbatch my_run.yaml
```

## 6. Choose the resources

| | how to choose |
|---|---|
| CPUs | more CPUs shorten the run until the number of shards and blocks is reached. The showcase runs used 16 CPUs (up to 250,000 cells and the 7.7-million-cell CD4+ T screen) and 32 CPUs (2 million cells). |
| memory | at least the memory of the largest single task plus room for parallel tasks. The largest task was 13.8 to 26.9 GB in the showcase runs; 64 GB suits screens up to a few hundred thousand cells, 200 GB suits the large ones. When memory is short, the run lowers the number of parallel tasks itself and logs it. |
| time | about 1.4 core-hours of compute per million cells: divide by the number of CPUs and add a margin. The K562 screen (2 million cells) took 38 minutes on 32 CPUs. |

## 7. Run interactively

In an interactive session the same run is one command. Set the variables the job script sets for itself first:

```
source <env folder>/bin/activate
cd <repo folder>
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg HDF5_USE_FILE_LOCKING=FALSE
python -B spore_xl.py run --config my_run.yaml
```

`run` finds the CPUs and memory of the session itself. On a shared node, add `--workers` and `--max-ram-gb`.

## 8. Resume and healthy logs

If a job stops (time limit, node failure, cancellation), submit the same line again. Finished work is skipped, work cut off in the middle is moved aside and redone, nothing is deleted, and the result equals that of an uninterrupted run.

A healthy log shows one banner, one `workers:` line per stage, `[finalize] GRAPH-READY`, `[report] 8/8 figures`, every verification line as `PASS`, and ends with `[run] DONE: 15 verification checks PASS`.
