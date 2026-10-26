"""Builds the reference material the equivalence tests compare against, in a NEW folder (default tests/ref):
  data/       synthetic_60k.h5ad (+ truth, design), protected list, the five SPORE v3.0 test configs with this folder filled in
  intermediate/spore_baseline/run1, run2     SPORE v3.0 (spore_local.py) runs on the synthetic 60k
  intermediate/spore_min10/run1              the same with min_cells_per_perturbation 10 (groups under 50 cells exist)
  intermediate/spore_baseline_figs/run2      the same, figures saved (test_report compares figure names)
  intermediate/spore_reference_data          SPORE's own Phase 1-7 intermediates (build_spore_reference.py)
Each step is skipped when its output exists; nothing is overwritten. About 15 minutes on a laptop (SPORE v3.0 runs take about 2 minutes each).

  python -B tests/prepare_references.py --spore-v3-dir <checkout of the tag v3.0, outside this repository> [--out tests/ref]
Then:  set XL_TEST_DATA=<that folder> ; python -B tests/run_all_tests.py --tag t1 --logs <new folder> --scratch <folder>
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
TEMPLATES = Path(__file__).resolve().parent / "ref_templates"
CONFIGS = ["spore_config_synthetic60k_baseline_v1", "spore_config_synthetic60k_baseline_v1_run2", "spore_config_synthetic60k_min10_v1",
           "spore_config_synthetic60k_baseline_figs_v1", "spore_config_synthetic60k_reference_v1"]
RUNS = {"spore_config_synthetic60k_baseline_v1": "intermediate/spore_baseline/run1", "spore_config_synthetic60k_baseline_v1_run2": "intermediate/spore_baseline/run2",
        "spore_config_synthetic60k_min10_v1": "intermediate/spore_min10/run1", "spore_config_synthetic60k_baseline_figs_v1": "intermediate/spore_baseline_figs/run2"}


def run(command, env, cwd=None):
    print("   $", " ".join(map(str, command)))
    subprocess.run([str(part) for part in command], check=True, env=env, cwd=str(cwd or REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "tests" / "ref"))
    parser.add_argument("--spore-v3-dir", default=os.environ.get("XL_SPORE_V3_DIR"), help="a checkout of the tag v3.0 (outside this repository)")
    args = parser.parse_args()
    v3_dir = Path(args.spore_v3_dir).resolve() if args.spore_v3_dir else None
    if v3_dir is None or not (v3_dir / "spore_local.py").exists():
        raise SystemExit("SPORE v3.0 is not part of this repository: pass --spore-v3-dir <checkout of the tag v3.0> (or set XL_SPORE_V3_DIR); docs/DEVELOPER.md section 1b")
    recorded = json.loads((REPO / "tests" / "spore_5015dc6_hashes.json").read_text(encoding="utf-8"))["sha256"]["spore_local.py"]
    if hashlib.sha256((v3_dir / "spore_local.py").read_bytes()).hexdigest() != recorded:
        raise SystemExit(f"{v3_dir / 'spore_local.py'} is not SPORE v3.0 (5015dc6): its sha256 differs from tests/spore_5015dc6_hashes.json")
    out = Path(args.out).resolve()
    ref_text = str(out).replace("\\", "/")
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", PYTHONUTF8="1", XL_TEST_DATA=str(out))
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    if not (data / "synthetic_60k.h5ad").exists():
        run([sys.executable, "-B", REPO / "tests" / "make_synthetic.py", "--out", data / "synthetic_60k.h5ad"], env)
    if not (data / "protected_synthetic_60k.json").exists():
        (data / "protected_synthetic_60k.json").write_text((TEMPLATES / "protected_synthetic_60k.json").read_text(encoding="utf-8"), encoding="utf-8")
    for name in CONFIGS:
        target = data / f"{name}.yaml"
        if not target.exists():
            target.write_text((TEMPLATES / f"{name}.yaml").read_text(encoding="utf-8").replace("{REF}", ref_text), encoding="utf-8")
    for name, relative in RUNS.items():
        run_dir = out / relative
        if (run_dir / "environment.txt").exists():
            print(f"skip {relative}")
            continue
        if not (run_dir / "processed").exists():
            run([sys.executable, "-B", v3_dir / "spore_local.py", "--config", data / f"{name}.yaml"], env, cwd=v3_dir)
        run([sys.executable, "-B", REPO / "tools" / "write_environment.py", run_dir], env)
    reference = out / "intermediate" / "spore_reference_data"
    if not reference.exists():
        run([sys.executable, "-B", REPO / "tests" / "build_spore_reference.py", "--out", reference, "--config", data / "spore_config_synthetic60k_reference_v1.yaml"], env)
    print(f"reference material ready in {out}")


if __name__ == "__main__":
    main()
