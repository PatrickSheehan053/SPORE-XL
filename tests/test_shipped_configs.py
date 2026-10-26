"""The SPORE-XL configs: configs/template.yaml, configs/datasets/ (K562 compat and production, CD4 Rest, RPE1) and the synthetic test configs in tests/configs/ (compat, production):
 - none carries the SPORE v3.0-only sections that v4.0 does not read (runtime, phase0_ingestion, phase1_detection, phase3_ambient, phase4_doublets,
   phase10_confounders = cell-cycle regression and Harmony, phase11_cell_line, plotting), so nobody can think they run
 - no <RUN_DATE> placeholder (the run date in the folder name is fixed to 20261022); the placeholders left are the ones listed in PLACEHOLDERS
 - the dataset configs and the template carry no personal cluster path (/group/sottoriva/<person>, /scratch/<person>)
 - with the placeholders substituted, Run(config) builds, in the intended mode (compat configs: compat-v3; the others: production), and the merged config still
   has SPORE's defaults for the removed sections
Run: set PYTHONDONTWRITEBYTECODE=1 ; set XL_TEST_TMP=<scratch> ; python -B tests\\test_shipped_configs.py   (needs no reference data)"""
import os
import re
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "engine"))

import yaml  # noqa: E402

from xl_common import Run  # noqa: E402

V3_ONLY = ["runtime", "phase0_ingestion", "phase1_detection", "phase3_ambient", "phase4_doublets", "phase10_confounders", "phase11_cell_line", "plotting"]
EXPECTED_MODE = {"configs/template.yaml": "production", "configs/datasets/k562_compat.yaml": "compat-v3", "configs/datasets/k562_production.yaml": "production",
                 "configs/datasets/cd4_rest.yaml": "production", "configs/datasets/rpe1.yaml": "production",
                 "tests/configs/synthetic_60k_compat.yaml": "compat-v3", "tests/configs/synthetic_60k_production.yaml": "production"}
PLACEHOLDERS = {"<SPORE_ROOT>", "<SCRATCH_WORK_DIR>", "<OUTPUT_DIR>", "<K562_H5AD>", "<CD4_TAR>", "<RPE1_H5AD>", "<RUN_LABEL>"}
NO_PERSONAL_PATHS = ["configs/template.yaml", "configs/datasets/k562_compat.yaml", "configs/datasets/k562_production.yaml", "configs/datasets/cd4_rest.yaml", "configs/datasets/rpe1.yaml"]


def code_lines(text):
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


def test_no_v3_only_sections_and_only_known_placeholders():
    for name in EXPECTED_MODE:
        text = (REPO / name).read_text(encoding="utf-8")
        loaded = yaml.safe_load(text)
        present = [section for section in V3_ONLY if section in loaded]
        assert not present, f"{name} still carries v3-only sections {present}"
        assert "<RUN_DATE>" not in text, f"{name}: <RUN_DATE> placeholder"
        left = sorted({token for line in code_lines(text) for token in re.findall(r"<[A-Z_0-9]+>", line)})
        assert set(left) <= PLACEHOLDERS, (name, left)
    for name in NO_PERSONAL_PATHS:
        body = "\n".join(code_lines((REPO / name).read_text(encoding="utf-8")))
        assert not re.search(r"/(group/sottoriva|scratch|home)/[a-z]+\.[a-z]+", body), f"{name}: a personal cluster path"
        assert "patrick.sheehan" not in body and "michele.calabro" not in body, name
    print(f"    {len(EXPECTED_MODE)} configs: no v3-only section, no <RUN_DATE>, known placeholders only, no personal paths in the dataset configs")


def test_configs_build_a_run_in_the_intended_mode():
    base = Path(tempfile.mkdtemp(prefix="cfgload_", dir=os.environ.get("XL_TEST_TMP", tempfile.gettempdir())))
    fill = {"<SPORE_ROOT>": REPO.as_posix(), "<SCRATCH_WORK_DIR>": (base / "scratch").as_posix(), "<OUTPUT_DIR>": (base / "library").as_posix(),
            "<K562_H5AD>": (base / "k562.h5ad").as_posix(), "<CD4_TAR>": (base / "cd4.tar").as_posix(), "<RPE1_H5AD>": (base / "rpe1.h5ad").as_posix(), "<RUN_LABEL>": "test"}
    previous = os.getcwd()
    os.chdir(base)   # the template config has a relative project_root
    try:
        for name, mode in EXPECTED_MODE.items():
            text = (REPO / name).read_text(encoding="utf-8")
            for token, value in fill.items():
                text = text.replace(token, value)
            path = base / Path(name).name
            path.write_text(text, encoding="utf-8")
            run = Run(path)
            assert run.mode == mode, (name, run.mode)
            assert run.cfg["phase10_confounders"]["cell_cycle_regression"] is True, "SPORE's default fills the removed section (and v4.0 never reads it)"
    finally:
        os.chdir(previous)
    print(f"    Run() builds for all {len(EXPECTED_MODE)} configs, modes as intended")


if __name__ == "__main__":
    test_no_v3_only_sections_and_only_known_placeholders()
    test_configs_build_a_run_in_the_intended_mode()
    print("PASS test_shipped_configs")
