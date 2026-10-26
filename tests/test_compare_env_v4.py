"""compare_v4: environment check (BOM / UTF-16 files, key missing on one side = WARN, differing key refuses) and the QC var rule.
Run: python -B tests\\test_compare_env_v3.py"""
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "engine"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import compare_v4  # noqa: E402

FULL = "python 3.12.9\nplatform x\nPYTHONHASHSEED 0\nnumpy 2.5.4\nscipy 1.18.1\npandas 2.3.1\nscanpy 1.12.4\nanndata 0.13.4\nscikit-learn 1.9.1\n"


def test_bom_and_crlf_are_ignored():
    parsed = compare_v4.parse_environment("﻿" + FULL.replace("\n", "\r\n"))
    assert parsed["python"] == "3.12.9" and parsed["scikit-learn"] == "1.9.1" and parsed["PYTHONHASHSEED"] == "0"


def test_files_with_bom_and_utf16_are_read():
    with tempfile.TemporaryDirectory() as folder:
        bom_file, utf16_file = Path(folder) / "a.txt", Path(folder) / "b.txt"
        bom_file.write_bytes(b"\xef\xbb\xbf" + FULL.replace("\n", "\r\n").encode("utf-8"))
        utf16_file.write_bytes(FULL.encode("utf-16"))
        for path in (bom_file, utf16_file):
            assert compare_v4.parse_environment(compare_v4.read_text_any(path))["python"] == "3.12.9"


def test_missing_key_is_a_warning_not_a_refusal():
    recorded = compare_v4.parse_environment(FULL.replace("python 3.12.9\n", ""))
    refusals, warnings = compare_v4.compare_environments("the other run", recorded, compare_v4.parse_environment(FULL))
    assert refusals == [] and len(warnings) == 1 and "'python' not recorded" in warnings[0]


def test_differing_key_refuses():
    recorded = compare_v4.parse_environment(FULL.replace("numpy 2.5.4", "numpy 2.5.3"))
    refusals, warnings = compare_v4.compare_environments("the other run", recorded, compare_v4.parse_environment(FULL))
    assert len(refusals) == 1 and "numpy" in refusals[0] and warnings == []


def test_identical_is_silent():
    here = compare_v4.parse_environment(FULL)
    assert compare_v4.compare_environments("x", dict(here), here) == ([], [])


def _frames(shift):
    base = {"n_cells_by_counts": [10.0, 20.0], "mean_counts": [1.5, 2.5], "pct_dropout_by_counts": [50.0, 40.0],
            "total_counts": [40_000_000.0, 100.0]}
    other = dict(base)
    other["total_counts"] = [40_000_000.0 + shift, 100.0]
    return pd.DataFrame(other), pd.DataFrame(base)


def test_qc_rule_default_is_strict_and_optin_is_reported():
    ours, theirs = _frames(0.0)
    assert compare_v4.qc_column_check(ours, theirs)[0]
    ours, theirs = _frames(322.0)  # 8e-6 relative on one gene, 322 > 1e-6 x 4e7
    ok, detail = compare_v4.qc_column_check(ours, theirs)
    assert not ok and "total_counts: FAIL" in detail
    ok, detail = compare_v4.qc_column_check(ours, theirs, float32_rel_tol=1e-4)
    assert ok and "[default, Amendment 1 item 13]" in detail
    ours, theirs = _frames(10_000.0)  # 2.5e-4 relative: even the opt-in refuses
    assert not compare_v4.qc_column_check(ours, theirs, float32_rel_tol=1e-4)[0]
    ours["n_cells_by_counts"] = ours["n_cells_by_counts"] + 5  # opt-in never relaxes the integer columns
    assert not compare_v4.qc_column_check(ours, theirs, float32_rel_tol=1.0)[0]


if __name__ == "__main__":
    failures = 0
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except Exception:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                failures += 1
                print(f"FAIL {name}")
    sys.exit(1 if failures else 0)
