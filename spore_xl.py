"""SPORE-XL v4.0 entry point.

  python spore_xl.py run --config my_run.yaml                 the whole run: input check, intake (recipe sources), every stage, then the verification (exit code 1 on a FAIL)
  sbatch --output=<log folder>/slurm_%j.out slurm/spore_xl.sbatch my_run.yaml     the same run as one SLURM job
  python spore_xl.py convert --recipe src/intake/recipes/<dataset>.convert.yaml --out <dir>     raw formats -> SPORE-ready sparse .h5ad
Developers: stage-level commands (plan ... report, --index N), measure, branch, cleanup: docs/DEVELOPER.md. Set PYTHONHASHSEED=0 for compat-mode runs.
SPORE v3.0 (the single-process runner) is the tag v3.0 and the branch v3-archive of this repository; it is not part of this tree (docs/DEVELOPER.md section 1b).
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "engine"))

import cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli.main())
