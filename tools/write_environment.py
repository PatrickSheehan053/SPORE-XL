"""Writes environment.txt (python, platform, PYTHONHASHSEED, numpy, scipy, pandas, scanpy, anndata, scikit-learn) into a folder, for
runs that do not write it themselves (SPORE v3.0 runs): compare_v3.py reads it to check that SPORE and XL ran in one environment.
Plain UTF-8 without a byte-order mark; refuses to overwrite.

  python -B src/write_environment.py <spore run folder>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "engine"))   # works from src/engine/, tests/ and tools/ of a repository

from xl_common import environment_text


def main(argv=None):
    folder = Path((argv or sys.argv[1:])[0])
    target = folder / "environment.txt"
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a folder")
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    target.write_text(environment_text(), encoding="utf-8")
    print(f"wrote {target}")
    print(target.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
