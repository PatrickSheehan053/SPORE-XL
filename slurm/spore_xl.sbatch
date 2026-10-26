#!/bin/bash
# SPORE-XL v4.0 on a cluster, ONE command:
#
#     sbatch slurm/spore_xl.sbatch my_config.yaml
#
# One job, one node, the whole run: intake (when the config's source is a recipe / tar), every stage up to the report, then the verification
# (reports/VERIFICATION.md; this job exits non-zero on any FAIL). The script sets the environment and the venv, gives `spore_xl.py run` the job's CPUs as the
# worker limit and 90 percent of the job's memory as the RAM budget, and `run` lowers the worker count per stage when a stage's tasks would not fit.
# RESUMABLE: submit the same line again after any stop (time limit, node failure): finished units are skipped. Nothing is deleted or overwritten.
# Submit it from the repository folder and nothing else has to be set. Optional, in the environment (sbatch --export=ALL,NAME=value,...):
#   REPO   the SPORE repository (default: the folder you submit from if it holds spore_xl.py, else the cluster default below)
#   VENV   the Python 3.12 venv (default /scratch/patrick.sheehan/envs/spore_xl; docs/HPC_SETUP.md builds it; HPC1 and HPC2 each have their own /scratch)
# The job log: give sbatch --output=<existing folder outside the repository>/slurm_%j.out; without it SLURM writes slurm-<job id>.out into the folder you submit from.
# (There is no #SBATCH --output line here on purpose: SLURM does not create the folder, and no path is right for every user.)
#
# RESOURCES: SET BY PATRICK. PROPOSED: 32 CPUs, 400 GB, 24 h (one HPC1 node has 32 cores and about 580 GB; K562, 2 million cells: 72 minutes with 14 workers and 110 GB; CD4 Rest, 8 million cells: 400 GB).
# Override on the command line, for example: sbatch --cpus-per-task=16 --mem=200G --time=12:00:00 slurm/spore_xl.sbatch my_config.yaml
# ============================ SITE SETTINGS: edit these for your installation ============================
# The defaults point to the Sottoriva lab deployment on HPC1. Three settings are site-specific:
#   - the fallback REPO (line marked SITE SETTING below; used when you do not submit from the repository folder and REPO is not set),
#   - the VENV (line marked SITE SETTING below),
#   - the partition (#SBATCH --partition below).
# REPO and VENV can also be given per job from the environment: sbatch --export=ALL,REPO=...,VENV=... 
# ==========================================================================================================
#SBATCH --job-name=spore_xl
#SBATCH --partition=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG="${1:-}"
if [ -z "$CONFIG" ]; then
    echo "STOP: give the config file: sbatch slurm/spore_xl.sbatch my_config.yaml"
    exit 2
fi
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
case "$CONFIG" in /*|[A-Za-z]:*) ;; *) CONFIG="$SUBMIT_DIR/$CONFIG" ;; esac   # an absolute path (also C:/... under Git Bash) is kept, a relative one is taken from the submit folder

REPO="${REPO:-}"
if [ -z "$REPO" ]; then
    if [ -f "$SUBMIT_DIR/spore_xl.py" ]; then
        REPO="$SUBMIT_DIR"
    else
        REPO="/group/sottoriva/patrick.sheehan/spore_xl_r5/SPORE-XL"   # SITE SETTING: edit for your installation (Sottoriva lab, HPC1)
    fi
fi
VENV="${VENV:-/scratch/patrick.sheehan/envs/spore_xl}"   # SITE SETTING: edit for your installation (Sottoriva lab, HPC1)

for needed in "$CONFIG" "$REPO/spore_xl.py" "$VENV/bin/activate"; do
    if [ ! -e "$needed" ]; then
        echo "STOP: $needed does not exist (docs/HPC_SETUP.md: the config, the repository, the venv). Nothing was run."
        exit 2
    fi
done

# the RAM budget is 90 percent of what SLURM granted; the workers never exceed the CPUs of the job
if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
    MAX_RAM_GB=$(awk -v mb="$SLURM_MEM_PER_NODE" 'BEGIN { printf "%.1f", mb * 0.9 / 1024 }')
else
    MAX_RAM_GB="${MAX_RAM_GB:-90}"
fi
WORKERS="${SLURM_CPUS_PER_TASK:-${WORKERS:-4}}"

source "$VENV/bin/activate"
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export HDF5_USE_FILE_LOCKING=FALSE
export XL_SPORE_ROOT="$REPO"
cd "$REPO"

python -B spore_xl.py run --config "$CONFIG" --workers "$WORKERS" --max-ram-gb "$MAX_RAM_GB"
