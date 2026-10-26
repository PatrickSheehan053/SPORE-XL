"""Intake RAM pre-flight (preflight.estimate_stage_ram_gb, stage intake) calibrated on the CD4 Rest run of 22 Oct (94 members, 11 workers at once, HPC1):
the estimate is 0.7 GB + 22.2 bytes per stored value of the largest member (before build r4: 36 bytes, 20.23 GB for the CD4 member of 542 million values; now 12.7 GB, against
measured peaks of 4.7 to 8.0 GB). The test pins (a) the formula, (b) that it covers every measured (stored values, peak RSS) point of that run with a margin of at
least 1.25, and (c) the old overestimate is gone (about 2x the worst measured point per stored value is the ceiling).
Run: python -B tests\\test_preflight_intake.py   (no pipeline run)"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "engine"))

import pandas as pd  # noqa: E402

import preflight  # noqa: E402

# (stored values of the member, measured peak RSS in GB), read from intake_members.tsv and reports/memory_and_time.tsv of the CD4 Rest run (22 Oct)
MEASURED = [(319_923_328, 4.714), (349_762_008, 5.128), (403_041_932, 6.321), (416_133_311, 6.690), (450_582_767, 5.500), (457_975_783, 5.361),
            (480_195_391, 5.965), (487_222_361, 5.904), (497_636_274, 7.929), (500_792_351, 7.970), (513_932_936, 6.100), (517_461_736, 7.080),
            (518_094_054, 7.450), (524_867_810, 6.761), (527_202_844, 7.348), (530_049_690, 7.235), (538_266_190, 7.177), (542_477_833, 6.793)]


def fake_run(nnz_values):
    folder = Path(tempfile.mkdtemp(prefix="preflight_intake_", dir=__import__("os").environ.get("XL_TEST_TMP", tempfile.gettempdir())))
    manifest = folder / "manifest"
    manifest.mkdir()
    pd.DataFrame({"index": range(len(nnz_values)), "name": [f"m{i}.h5" for i in range(len(nnz_values))], "size": [10 * value for value in nnz_values],
                  "cells": [100000] * len(nnz_values), "nnz": nnz_values}).to_csv(manifest / "intake_members.tsv", sep="\t", index=False)
    return SimpleNamespace(manifest_dir=manifest, work_dir=folder / "work")


def test_formula_and_margin_over_the_measured_peaks():
    estimate = preflight.estimate_stage_ram_gb(fake_run([542_477_833]), "intake")
    assert abs(estimate - (0.7 + 542_477_833 * 22.2 / 1e9)) < 1e-9, estimate
    assert 12.0 < estimate < 13.5, f"CD4's largest member must now be about 12.7 GB (was 20.23), got {estimate:.2f}"
    for nnz, peak in MEASURED:
        value = preflight.estimate_stage_ram_gb(fake_run([nnz]), "intake")
        assert value >= 1.25 * peak, f"estimate {value:.2f} GB does not cover the measured {peak} GB of a member of {nnz:,} values with a margin of 1.25"
    worst_ratio = max((peak - 0.7) / (nnz / 1e9) for nnz, peak in MEASURED)
    assert 13 < worst_ratio < 16, f"worst measured bytes per stored value: {worst_ratio:.1f}"
    assert 22.2 / worst_ratio < 2.0, "the estimate must stay within about 2x of the worst measured point per stored value"
    print(f"    estimate 0.7 GB + 22.2 B per stored value: {estimate:.2f} GB for the CD4 member of 542 M values (before 20.23); covers all {len(MEASURED)} measured points "
          f"with >= 1.25x; worst measured {worst_ratio:.1f} B per stored value")


def test_the_largest_member_decides_and_the_size_fallback_is_unchanged():
    assert preflight.estimate_stage_ram_gb(fake_run([100_000_000, 300_000_000, 200_000_000]), "intake") == preflight.estimate_stage_ram_gb(fake_run([300_000_000]), "intake")
    run = fake_run([1])
    table = pd.read_csv(run.manifest_dir / "intake_members.tsv", sep="\t").drop(columns=["nnz"])
    table.to_csv(run.manifest_dir / "intake_members.tsv", sep="\t", index=False)
    assert abs(preflight.estimate_stage_ram_gb(run, "intake") - (0.7 + 10 / 1e9 * 8)) < 1e-9
    print("    the largest member decides; without an nnz column the file-size rule of before is kept")


if __name__ == "__main__":
    test_formula_and_margin_over_the_measured_peaks()
    test_the_largest_member_decides_and_the_size_fallback_is_unchanged()
    print("PASS test_preflight_intake")
