"""report stage on the 60k synthetic: the XL figure folder holds the same figure names as SPORE's (run with save_figures: true,
intermediate\\spore_baseline_figs\\run1\\figures; Phase 10's p10_* excluded and stated), per-cell QC plots drawn from ALL cells, every
figure labelled all cells / subsample of N, re-run is a no-op. Run: python -B tests\\test_report.py  (XL_TEST_TMP set)"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import BUILD, get_run  # noqa: E402

import cli as spore_xl  # noqa: E402

SPORE_FIGURES = BUILD / "intermediate" / "spore_baseline_figs" / "run2" / "figures"
EXPECTED = {"sparsity_saturation", "p02_qc_violin", "p02_mt_scatter", "p05_efficiency_scatter", "p06_triage_dashboard",
            "p07_splits_dashboard", "p08_hvg_dashboard", "p09_normalization"}


def test_figures_match_spore_names_and_labels():
    run = get_run(3, through="report")
    ours = {path.stem for path in (run.run_dir / "figures").glob("*.png")}
    assert ours == EXPECTED, (sorted(ours ^ EXPECTED))
    if SPORE_FIGURES.exists() and list(SPORE_FIGURES.glob("*.png")):
        theirs = {path.stem for path in SPORE_FIGURES.glob("*.png")}
        assert ours == {name for name in theirs if not name.startswith("p10")}, sorted(ours ^ theirs)
        print(f"    figure names equal SPORE's ({len(theirs)} in SPORE: {sorted(theirs - ours)} not drawn by XL)")
    for path in (run.run_dir / "figures").glob("*.png"):
        assert path.stat().st_size > 20000, f"{path.name} looks empty"
    report = (run.reports_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "all 60,000 cells" in report, "per-cell QC figures must be drawn from all cells"
    for stem in ("p02_qc_violin", "p02_mt_scatter", "sparsity_saturation"):
        line = [row for row in report.split("\n") if row.startswith(f"| {stem}.png")][0]
        assert "all 60,000 cells" in line and line.rstrip().endswith("ok |"), line
    assert "subsample of" in report and "No PCA / Harmony / cell-cycle-regression figures" in report and "Phase numbering" in report
    assert "FAILED" not in report
    spore_xl.run_stage(run, "report")  # no-op
    print("    8 figures drawn, QC plots from all 60,000 cells, labels present, phase-10 note and phase map stated, re-run is a no-op")


if __name__ == "__main__":
    try:
        test_figures_match_spore_names_and_labels()
        print("PASS test_report")
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("FAIL test_report")
        sys.exit(1)
