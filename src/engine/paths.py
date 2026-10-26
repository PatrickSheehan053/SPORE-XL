"""The one place that makes the SPORE v3.0 phase modules importable. Import this module before any SPORE module.
In the v4 repository layout SPORE_ROOT is the repository root (two levels above src/engine/); XL_SPORE_ROOT overrides it."""
import os
import sys
from pathlib import Path

# must happen before any SPORE import: no __pycache__ inside src, headless matplotlib
sys.dont_write_bytecode = True
os.environ["MPLBACKEND"] = "Agg"

BUILD_ROOT = Path(__file__).resolve().parent                      # src/engine/ (holds the `adapted_phases` package)
SPORE_ROOT = Path(os.environ.get("XL_SPORE_ROOT", str(BUILD_ROOT.parents[1])))


def add_spore_to_path():
    if not (SPORE_ROOT / "src").is_dir():
        raise FileNotFoundError(f"SPORE not found at {SPORE_ROOT} (set XL_SPORE_ROOT)")
    spore_text = str(SPORE_ROOT)
    if spore_text not in sys.path:
        sys.path.insert(0, spore_text)
    return SPORE_ROOT
