"""compare_v4 (session 23, P8): same as compare_v3 except that the DEFAULT for mean_counts / total_counts is a per-gene relative tolerance of 1e-4
(Amendment 1 item 13: SPORE accumulates them in float32, XL is exact); n_cells_by_counts and pct_dropout_by_counts stay at 1e-6.
--strict-qc restores the 1e-6 rule for all four columns.

compare_v3 (session 22; replaces compare.py and the unverified compare_v2.py, both kept unchanged).
Fixes: (1) environment.txt files written by PowerShell start with a UTF-8 byte-order mark (or are UTF-16), which made the
first key read "<BOM>python" and the python version parse as None: the real cause of the false refusal on RPE1 r3.
(2) a key missing on ONE side is a WARN ("not recorded"); a key present on both sides with different values still refuses.
(3) QC var columns keep the Amendment 1 item 9 tolerance (1e-6 of the column maximum) by default; the detail line says how many
genes differ and by how much. --qc-float32-rel-tol is an explicit, reported opt-in for the float32 columns (mean_counts,
total_counts), because scanpy/SPORE accumulate those in float32 and XL is exact (tests\\diag_total_counts_independent.py).
(4) the console output is also written to <xl run>\\logs\\compare_<mode>_<timestamp>.log.
compare: equivalence of an XL run with a SPORE run directory (blueprint Amendment 1 item 9), and invariance of XL runs.

  python -B src\\compare_v3.py spore --xl <xl_run_config.yaml> --spore-dir <spore run dir> [--qc-float32-rel-tol 1e-4]
  python -B src\\compare_v3.py xl --a <config> --b <config>

Exact: kept cells (and order), perturbation survivors, gene mask, split JSON bytes, control assignment, HVG panel and
hvg_stats, hvgraw X. 1e-6: p9 X, QC var columns. Phase 12 units: exact for groups of 50+ cells; groups under 50 cells
are counted as the allowed deviation (how many differ). Ignored: uns plotting caches and file bytes.
Refuses to compare when a recorded environment key differs (python/numpy/scipy/pandas/scanpy/anndata/scikit-learn/PYTHONHASHSEED).
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "engine"))   # works from src/engine/, tests/ and tools/ of a repository

from xl_common import Run, environment_text
from stage_gather import total_blocks
from stage_map3 import SPLITS, part_path
from stage_metacell import metacell_part_path

QC_COLUMNS = ["n_cells_by_counts", "mean_counts", "pct_dropout_by_counts", "total_counts"]
FLOAT32_QC_COLUMNS = ("mean_counts", "total_counts")
ENV_KEYS = ("python", "PYTHONHASHSEED", "numpy", "scipy", "pandas", "scanpy", "anndata", "scikit-learn")


def read_text_any(path):
    """Text of a small file written by Python (utf-8), PowerShell 5 (utf-8 with BOM) or `>` redirection (utf-16)."""
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def parse_environment(text):
    """name -> value for the keys in ENV_KEYS; a key that is not in the text is None. A leading byte-order mark and
    carriage returns (files written by PowerShell) are ignored."""
    pairs = {}
    for line in text.lstrip("﻿").strip().split("\n"):
        name, _, value = line.strip().partition(" ")
        pairs[name] = value.strip()
    return {key: pairs.get(key) for key in ENV_KEYS}


def compare_environments(label, recorded, here):
    """(refusals, warnings): a key recorded on both sides with different values refuses; a key missing on ONE side
    (None) is only a warning ("not recorded")."""
    refusals, warnings = [], []
    for key in ENV_KEYS:
        if recorded[key] is None or here[key] is None:
            warnings.append(f"environment key '{key}' not recorded in {label} ({recorded[key]!r} vs {here[key]!r}); not compared")
        elif recorded[key] != here[key]:
            refusals.append(f"{key}: {label} has {recorded[key]!r}, this interpreter {here[key]!r}")
    return refusals, warnings


def check_environment(run, other_text=None):
    """Returns (environment of this interpreter, warnings). Raises only when a key present on both sides differs."""
    here = parse_environment(environment_text())
    refusals, warnings = [], []
    sides = [("the XL run", read_text_any(run.config_dir / "environment.txt"))]
    if other_text is not None:
        sides.append(("the other run", other_text))
    for label, text in sides:
        bad, warn = compare_environments(label, parse_environment(text), here)
        refusals += bad
        warnings += warn
    if refusals:
        raise RuntimeError("environments differ: " + "; ".join(refusals))
    return here, warnings


def load_parts(run, kind, split):
    n_total = total_blocks(run)
    pieces = [ad.read_h5ad(part_path(run, split, kind, block)) for block in range(n_total) if part_path(run, split, kind, block).exists()]
    matrix = sp.vstack([piece.X for piece in pieces], format="csr")
    obs = pd.concat([piece.obs for piece in pieces])
    order = np.argsort(obs["xl_global_index"].to_numpy(), kind="stable")
    return matrix[order], obs.iloc[order], pieces[0].var


def load_units(run):
    n_total = total_blocks(run)
    pieces = [ad.read_h5ad(metacell_part_path(run, block)) for block in range(n_total) if metacell_part_path(run, block).exists()]
    return sp.vstack([piece.X for piece in pieces], format="csr"), pd.concat([piece.obs for piece in pieces])


def _same_csr(left, right):
    left, right = sp.csr_matrix(left), sp.csr_matrix(right)
    return (left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left.indptr, right.indptr)
            and np.array_equal(left.indices, right.indices) and left.data.tobytes() == right.data.tobytes())


def _max_abs_diff(left, right):
    left, right = sp.csr_matrix(left), sp.csr_matrix(right)
    if left.shape != right.shape:
        return float("inf")
    difference = abs(left - right)
    return float(difference.max()) if difference.nnz else 0.0


def qc_column_check(ours_var, reference_var, float32_rel_tol=None):
    """(ok, detail) for the QC var columns. Default: every column within 1e-6 of the column maximum (Amendment 1 item 9).
    With float32_rel_tol, mean_counts and total_counts are instead held to that per-gene relative tolerance (opt-in)."""
    ok, parts = True, []
    for column in QC_COLUMNS:
        ours, theirs = ours_var[column].to_numpy(float), reference_var[column].to_numpy(float)
        difference = np.abs(ours - theirs)
        relative = difference / np.maximum(np.abs(theirs), 1.0)
        if float32_rel_tol is not None and column in FLOAT32_QC_COLUMNS:
            column_ok = bool(relative.max() <= float32_rel_tol)
            rule = f"relative <= {float32_rel_tol:g} [default, Amendment 1 item 13]"
        else:
            column_ok = bool(difference.max() <= 1e-6 * max(1.0, float(np.abs(theirs).max())))
            rule = "<= 1e-6 of column max"
        ok &= column_ok
        parts.append(f"{column}: {'ok' if column_ok else 'FAIL'} ({rule}; {int((difference > 0).sum())} genes differ, "
                     f"{int((relative > 1e-6).sum())} by more than 1e-6 relative, max abs {difference.max():.6g}, max relative {relative.max():.3g})")
    return ok, "; ".join(parts)


DEFAULT_QC_FLOAT32_REL_TOL = 1e-4


def compare_with_spore(run: Run, spore_dir, name=None, write_report=True, qc_float32_rel_tol=DEFAULT_QC_FLOAT32_REL_TOL):
    spore_dir = Path(spore_dir)
    name = name or run.name
    env_file = spore_dir / "environment.txt"
    _, env_warnings = check_environment(run, read_text_any(env_file) if env_file.exists() else None)
    if not env_file.exists():
        env_warnings.append(f"{env_file} does not exist: environment of the SPORE run not recorded; not compared")
    results = []

    def record(label, ok, detail=""):
        results.append((label, bool(ok), detail))

    for warning in env_warnings:
        record("environment check", True, f"WARN {warning}")

    split_dir, processed = spore_dir / "splits", spore_dir / "processed"
    record("split JSON byte-identical",
           (run.splits_dir / f"{run.name}_split_indices.json").read_bytes() == (split_dir / f"{name}_split_indices.json").read_bytes())
    kept_names = [n for n in (run.stats_dir / "reduce2" / "all" / "gene_keep_mask.txt").read_text(encoding="utf-8").split("\n") if n]
    spore_train_raw = ad.read_h5ad(split_dir / f"{name}_train_hvgraw.h5ad")
    spore_stats = pd.DataFrame.from_dict(spore_train_raw.uns["hvg_stats"], orient="index")
    record("gene mask (genes surviving Phase 6)", sorted(kept_names) == sorted(spore_stats.index), f"{len(kept_names)} genes")
    panel = [n for n in (run.splits_dir / f"{run.name}_hvg_panel.txt").read_text(encoding="utf-8").split("\n") if n]
    record("HVG panel (names and order)", panel == list(spore_train_raw.var_names), f"{len(panel)} genes")
    xl_stats = pd.read_parquet(run.stats_dir / "hvg" / "all" / "hvg_stats.parquet")
    stats_ok = sorted(xl_stats.columns) == sorted(spore_stats.columns) and sorted(xl_stats.index) == sorted(spore_stats.index)
    if stats_ok:
        reference = spore_stats.loc[xl_stats.index, list(xl_stats.columns)]
        for column in xl_stats.columns:
            left, right = xl_stats[column].to_numpy(), reference[column].to_numpy()
            stats_ok &= (left.tobytes() == right.astype(left.dtype).tobytes()) if left.dtype.kind == "f" else bool((left == right).all())
    record("hvg_stats identical", stats_ok)
    assignment = pd.read_parquet(run.splits_dir / f"{run.name}_control_assignment.parquet")
    names_by_gidx = None
    total_cells = 0
    for split in SPLITS:
        reference = ad.read_h5ad(split_dir / f"{name}_{split}_hvgraw.h5ad")
        matrix, obs, var = load_parts(run, "hvgraw", split)
        record(f"kept cells + order [{split}]", list(obs.index) == list(reference.obs_names), f"{matrix.shape[0]:,} cells")
        record(f"perturbation survivors [{split}]",
               set(obs[run.pert_col].astype(str)) - {run.ctrl_label} == set(reference.obs[run.pert_col].astype(str)) - {run.ctrl_label})
        ours_controls = list(obs.index[(obs[run.pert_col].astype(str) == run.ctrl_label).to_numpy()])
        ref_controls = list(reference.obs_names[(reference.obs[run.pert_col].astype(str) == run.ctrl_label).to_numpy()])
        record(f"control assignment [{split}]", ours_controls == ref_controls and
               set(obs.loc[obs.index.isin(ours_controls), "xl_global_index"]) == set(assignment.loc[assignment["split"] == split, "xl_global_index"]),
               f"{len(ours_controls):,} controls")
        record(f"hvgraw X exact [{split}]", _same_csr(matrix, reference.X) and list(var.index) == list(reference.var_names))
        qc_ok, qc_detail = qc_column_check(var, reference.var, qc_float32_rel_tol)
        record(f"QC var columns [{split}]", qc_ok, qc_detail)
        p9_matrix, _, _ = load_parts(run, "p9", split)
        reference_p9 = ad.read_h5ad(split_dir / f"{name}_{split}_p9.h5ad")
        worst = _max_abs_diff(p9_matrix, reference_p9.X)
        record(f"p9 X <= 1e-6 [{split}]", worst <= 1e-6, f"max abs diff {worst}")
        total_cells += matrix.shape[0]

    # ---- Phase 12 units
    reference = ad.read_h5ad(processed / f"{name}_allsplits_metacell_ctrlpreserved.h5ad")
    ref_x = reference.X.toarray() if sp.issparse(reference.X) else np.asarray(reference.X)
    ref_labels = reference.obs[run.pert_col].astype(str).to_numpy()
    matrix, obs = load_units(run)
    groups = pd.concat([pd.read_parquet(p) for p in sorted((run.stats_dir / "metacell").glob("block_*/groups.parquet"))])
    sizes = groups.set_index("label")["n_cells"].to_dict()
    differing_small = compared_big = 0
    big_ok = True
    small_total = int(((groups["label"] != run.ctrl_label) & (groups["n_cells"] < 50)).sum())
    labels_xl = obs[run.pert_col].astype(str)
    for label, group in obs[labels_xl != run.ctrl_label].groupby(labels_xl[labels_xl != run.ctrl_label], sort=True):
        ordered = group.sort_values("xl_unit")
        ours = matrix[[obs.index.get_loc(n) for n in ordered.index]].toarray()
        wanted = np.flatnonzero(ref_labels == label)
        same = (len(wanted) == len(ordered) and ours.tobytes() == ref_x[wanted].tobytes() and
                list(ordered["n_cells_in_metacell"]) == list(reference.obs["n_cells_in_metacell"].to_numpy()[wanted]))
        if sizes[label] >= 50:
            compared_big += 1
            big_ok &= same
        elif not same:
            differing_small += 1
    record("Phase 12 units: every group of 50+ cells identical", big_ok, f"{compared_big} groups compared")
    controls = obs[labels_xl == run.ctrl_label].copy()
    controls["rank"] = controls["xl_control_split"].astype(str).map({"train": 0, "val": 1, "test": 2})
    controls = controls.sort_values(["rank", "xl_global_index"])
    ours = matrix[[obs.index.get_loc(n) for n in controls.index]].toarray()
    wanted = np.flatnonzero(ref_labels == run.ctrl_label)
    record("Phase 12: control units identical", len(wanted) == len(controls) and ours.tobytes() == ref_x[wanted].tobytes(), f"{len(controls):,} units")
    # units of groups under 50 cells may differ in content AND (through empty clusters) in number: the allowed deviation
    small_labels = set(groups.loc[(groups["label"] != run.ctrl_label) & (groups["n_cells"] < 50), "label"].astype(str))
    ref_counts = pd.Series(ref_labels).value_counts()
    xl_counts = labels_xl.value_counts()
    unit_count_diff = sum(1 for label in small_labels if int(ref_counts.get(label, 0)) != int(xl_counts.get(label, 0)))
    other_units_equal = all(int(ref_counts.get(label, 0)) == int(xl_counts.get(label, 0))
                            for label in set(ref_counts.index) | set(xl_counts.index) if label not in small_labels)
    record("Phase 12: unit counts equal for every group outside the deviation (50+ cells and controls)", other_units_equal,
           f"{matrix.shape[0]:,} vs {reference.n_obs:,} units in total")
    deviation = (f"{differing_small} of {small_total} groups under 50 cells differ in content, {unit_count_diff} in unit count "
                 f"(allowed deviation: fallback embedding fitted on the HVG subsample)")
    record("Phase 12 deviation report (informational, groups under 50 cells)", True, deviation)

    if write_report:
        path = run.reports_dir / f"EQUIVALENCE_vs_{spore_dir.name}.md"
        passed = all(ok for _, ok, _ in results)
        tolerance = ("Amendment 1 item 9 tolerances (QC var columns within 1e-6 of the column maximum)" if qc_float32_rel_tol is None else
                     f"OPT-IN relaxed tolerance for mean_counts/total_counts: relative <= {qc_float32_rel_tol:g} (not in Amendment 1)")
        lines = [f"# Equivalence: XL run {run.run_dir.name} vs SPORE {spore_dir}", "",
                 f"Result: **{'PASS' if passed else 'FAIL'}**  ({sum(ok for _, ok, _ in results)}/{len(results)} checks)", "",
                 f"Tolerances: {tolerance}", "",
                 "Exact equivalence is claimed below 1M cells only (SPORE switches numerical path above it).", "",
                 "| check | result | detail |", "|---|---|---|"]
        lines += [f"| {label} | {'PASS' if ok else 'FAIL'} | {detail} |" for label, ok, detail in results]
        counter = 1
        while path.exists():  # never overwrite an earlier report: EQUIVALENCE_vs_<run>_2.md, _3.md, ...
            counter += 1
            path = run.reports_dir / f"EQUIVALENCE_vs_{spore_dir.name}_{counter}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def xl_signature(run: Run):
    """Content hashes of the outputs that must not depend on the shard count."""
    def digest(data):
        return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode("utf-8")).hexdigest()

    stats = run.stats_dir
    signature = {
        "split_json": digest((run.splits_dir / f"{run.name}_split_indices.json").read_bytes()),
        "control_assignment": digest(pd.read_parquet(run.splits_dir / f"{run.name}_control_assignment.parquet").to_csv(index=False)),
        "panel": digest((run.splits_dir / f"{run.name}_hvg_panel.txt").read_bytes()),
        "gene_keep_mask": digest((stats / "reduce2" / "all" / "gene_keep_mask.txt").read_bytes()),
        "mean_shift": digest((stats / "reduce2" / "all" / "mean_shift.tsv").read_bytes()),
        "escaper_thresholds": digest((stats / "reduce1" / "all" / "escaper_thresholds.tsv").read_bytes()),
        "hvg_stats": digest(pd.read_parquet(stats / "hvg" / "all" / "hvg_stats.parquet").sort_index().to_csv()),
    }
    for kind in ("hvgraw", "p9"):
        for split in SPLITS:
            matrix, obs, var = load_parts(run, kind, split)
            signature[f"{kind}_{split}"] = digest(
                digest(matrix.indptr.tobytes()) + digest(matrix.indices.tobytes()) + digest(matrix.data.tobytes()) +
                digest("\n".join(obs.index)) + digest(obs[run.pert_col].astype(str).str.cat(sep="\n")))
    matrix, obs = load_units(run)
    labels = obs[run.pert_col].astype(str)
    canonical = obs.assign(_label=labels, _rank=obs["xl_control_split"].astype(str).map({"train": 0, "val": 1, "test": 2}).fillna(0)) \
        .sort_values(["_label", "_rank", "xl_global_index", "xl_unit"], kind="stable")
    rows = matrix[[obs.index.get_loc(n) for n in canonical.index]]
    signature["metacell_units"] = digest(digest(sp.csr_matrix(rows).data.tobytes()) + digest(sp.csr_matrix(rows).indices.tobytes()) +
                                         digest(canonical["n_cells_in_metacell"].to_numpy().tobytes()))
    return signature


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    one = sub.add_parser("spore")
    one.add_argument("--xl", required=True)
    one.add_argument("--spore-dir", required=True)
    one.add_argument("--qc-float32-rel-tol", type=float, default=DEFAULT_QC_FLOAT32_REL_TOL,
                     help="per-gene relative tolerance for mean_counts/total_counts (default 1e-4: SPORE accumulates them in float32)")
    one.add_argument("--strict-qc", action="store_true", help="hold all four QC var columns to 1e-6 of the column maximum")
    two = sub.add_parser("xl")
    two.add_argument("--a", required=True)
    two.add_argument("--b", required=True)
    two.add_argument("--a-is-v3", action="store_true", help="run A was made by code before v4 (for example the 21 Oct K562 run): v3.0 file names")
    two.add_argument("--b-is-v3", action="store_true", help="the same for run B")
    args = parser.parse_args(argv)
    run = Run(args.xl if args.mode == "spore" else args.a)
    if args.mode == "xl" and args.a_is_v3:
        run.mode = "compat-v3"
    lines = []

    def say(text):
        print(text)
        lines.append(text)

    try:
        if args.mode == "spore":
            outcome = compare_with_spore(run, args.spore_dir, qc_float32_rel_tol=None if args.strict_qc else args.qc_float32_rel_tol)
            for label, ok, detail in outcome:
                say(f"{'WARN' if detail.startswith('WARN') else 'PASS' if ok else 'FAIL'}  {label}  {detail}")
            passed = all(ok for _, ok, _ in outcome)
            say("PASS" if passed else "FAIL")
            code = 0 if passed else 1
        else:
            run_b = Run(args.b)
            if args.b_is_v3:
                run_b.mode = "compat-v3"
            first, second = xl_signature(run), xl_signature(run_b)
            differing = [key for key in first if first[key] != second[key]]
            say("IDENTICAL" if not differing else f"DIFFERENT: {differing}")
            code = 0 if not differing else 1
    except RuntimeError as refusal:  # environment mismatch: say so in the log as well
        say(f"REFUSED  {refusal}")
        code = 2
    log_path = run.logs_dir / f"compare_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"console output also written to {log_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
