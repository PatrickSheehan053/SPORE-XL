"""Stress-suite runner (P11): runs the XL pipeline (all-local) on each stress dataset, compares the outputs with the dataset's answer
key and writes one results file per arm plus an aggregated table.

  python -B tests\\stress\\run_stress_suite.py --data data\\stress --work <scratch dir> --results <results dir> --tag s1 [--only 06_tiny_groups]
                                              [--workers 2] [--mode current]

Resumable: an arm whose results json exists is skipped. Run directories live in --work (not in the project); results go to --results.
Each check row: dataset, arm, id, expected, observed, status (PASS / FAIL / NA), class (filled for FAILs known to be design limits or
feature gaps; INVESTIGATE otherwise).
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_support import BUILD, CODE_DIR, REPO, CONTROL, independent_derivation, make_config, part_files, read_obs, xl_survivor_names  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ARMS = {
    "06_tiny_groups": {"min50": {}, "min2": {"config_overrides": {"phase5_escaper_filtering": {"min_cells_per_perturbation": 2}}},
                       "min2_layoutB": {"config_overrides": {"phase5_escaper_filtering": {"min_cells_per_perturbation": 2}}, "layout_b": True,
                                        "reference_arm": "min2"}},
    "04_depth": {"default": {}, "layoutB": {"layout_b": True, "reference_arm": "default"}},
    "07a_ctrl_poor": {"default": {}, "layoutB": {"layout_b": True, "reference_arm": "default"}},
    "10_dense": {"dense": {}, "sparse_twin": {"twin": True, "reference_arm": "dense"}},
}
CLI = os.environ.get("STRESS_CLI", str(REPO / "spore_xl.py"))
COMPARE = str(CODE_DIR / ("compare_v4.py" if (CODE_DIR / "compare_v4.py").exists() else "compare_v3.py"))
PRESET_CLASS = {
    "escaper_removal_blind": "design limitation (KNOWN_ISSUES 5; P4 option reports the comparison)",
    "confounding_flagged": "feature gap (P5)", "batch_variance_reported": "feature gap (P5)", "donor_variance_reported": "feature gap (P5)",
    "labels_not_silently_accepted": "design limitation (intake, P7)",
    "duplicates_flagged": "feature gap (intake, P7)",
}


def row(ctx, check, expected, observed, status, klass=None):
    if status == "FAIL" and klass is None:
        klass = PRESET_CLASS.get(check, "INVESTIGATE")
    ctx["rows"].append({"dataset": ctx["name"], "arm": ctx["arm"], "id": check, "expected": expected, "observed": observed,
                        "status": status, "class": klass or ""})


def expectation_text(ctx, check):
    for item in ctx["key"]["expected_outcomes"]:
        if item["id"] == check:
            return item["expectation"]
    return check


def execute(ctx, command, log_path):
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(REPO), env=ctx["env"])
    return completed.returncode, time.time() - started


def tail(path, lines=12):
    text = Path(path).read_text(encoding="utf-8", errors="replace").strip().split("\n")
    return " | ".join(line.strip() for line in text[-lines:])


# ------------------------------------------------------------------------------------------------ universal checks
def universal(ctx):
    run, truth = ctx["run"], ctx["truth"]
    label_column = ctx["key"].get("label_column", "label")
    coverage = (run.reports_dir / "COVERAGE_REPORT.md")
    graph_ready = coverage.exists() and "GRAPH-READY" in coverage.read_text(encoding="utf-8") and "NOT READY" not in coverage.read_text(encoding="utf-8")
    row(ctx, "completes", expectation_text(ctx, "completes"), f"rc 0, GRAPH-READY={graph_ready}", "PASS" if graph_ready else "FAIL")
    import verify
    outcome = verify.verify_run(run, write_report=False)
    failed = [name for name, ok, _ in outcome if not ok]
    row(ctx, "verify", expectation_text(ctx, "verify"), f"{len(outcome) - len(failed)}/{len(outcome)} PASS" + (f"; failed: {failed}" if failed else ""),
        "PASS" if not failed else "FAIL")

    derived = ctx["derivation"]
    expected_set = set(derived["names"][derived["survivor"]])
    observed_set = xl_survivor_names(run, derived["names"])
    missing, extra = expected_set - observed_set, observed_set - expected_set
    row(ctx, "survivors", expectation_text(ctx, "survivors"),
        f"expected {len(expected_set):,} survivors, XL {len(observed_set):,}; only-expected {len(missing)}, only-XL {len(extra)}",
        "PASS" if not missing and not extra else "FAIL")

    frames = []
    for split in ("train", "val", "test"):
        for path in part_files(run, split, "hvgraw"):
            obs, _ = read_obs(path)
            obs["_split"] = split
            frames.append(obs)
    obs_all = pd.concat(frames)
    truth_indexed = truth.set_index("cell")
    problems = []
    got = truth_indexed.loc[obs_all.index]
    if not (obs_all["gene"].astype(str).to_numpy() == got[label_column].astype(str).to_numpy()).all():
        problems.append("label")
    if not (obs_all["gem_group"].astype(str).to_numpy() == got["batch"].astype(str).to_numpy()).all():
        problems.append("batch")
    if "donor" in obs_all.columns and not (obs_all["donor"].astype(str).to_numpy() == got["donor"].astype(str).to_numpy()).all():
        problems.append("donor")
    row(ctx, "obs_fidelity", expectation_text(ctx, "obs_fidelity"),
        f"{len(obs_all):,} output cells checked; mismatching columns: {problems or 'none'}", "PASS" if not problems else "FAIL")

    # x_fidelity: 300 random output cells against the raw rows
    source = ctx["source_matrix"]
    var_table = run.var_table()
    rng = np.random.default_rng(1)
    bad_cells, checked = 0, 0
    for split in ("train", "val", "test"):
        files = part_files(run, split, "hvgraw")
        for path in [files[index] for index in rng.permutation(len(files))[:3]] if files else []:
            part = ad.read_h5ad(path)
            columns = var_table.index.get_indexer(list(part.var_names))
            if (columns < 0).any():
                bad_cells += part.n_obs
                continue
            picks = rng.permutation(part.n_obs)[:25]
            rows_in_source = part.obs["xl_global_index"].to_numpy()[picks]
            raw_rows = source[rows_in_source][:, columns].toarray()
            got_rows = sp.csr_matrix(part.X[picks]).toarray()
            bad_cells += int((raw_rows != got_rows).any(axis=1).sum())
            checked += len(picks)
    row(ctx, "x_fidelity", expectation_text(ctx, "x_fidelity"), f"{checked} cells checked, {bad_cells} differ", "PASS" if bad_cells == 0 and checked else "FAIL")

    # p9 sanity
    bad_rows, checked, nonfinite = 0, 0, 0
    target = float(ctx["cfg"].get("phase9_normalization", {}).get("target_sum", 10000))
    for split in ("train", "val", "test"):
        files = part_files(run, split, "p9")
        for path in [files[index] for index in rng.permutation(len(files))[:3]] if files else []:
            part = ad.read_h5ad(path)
            raw_part = ad.read_h5ad(Path(str(path).replace("_p9", "_hvgraw")))
            dense = sp.csr_matrix(part.X).toarray()
            raw_sums = np.asarray(sp.csr_matrix(raw_part.X).sum(axis=1)).ravel()
            nonfinite += int((~np.isfinite(dense)).sum())
            sums = np.expm1(dense).sum(axis=1)
            nonempty = raw_sums > 0
            checked += int(nonempty.sum())
            bad_rows += int((np.abs(sums[nonempty] - target) > 0.002 * target).sum())
    row(ctx, "p9_sanity", expectation_text(ctx, "p9_sanity"), f"{checked} rows checked, {bad_rows} off target, {nonfinite} non-finite values",
        "PASS" if bad_rows == 0 and nonfinite == 0 and checked else "FAIL")

    # mass conservation
    units = pd.concat([read_obs(path)[0] for path in sorted((run.processed_dir / f"{run.name}_allsplits_metacell_ctrlpreserved").glob("part_*.h5ad"))])
    final = pd.read_parquet(run.manifest_dir / "global_obs_final.parquet")
    survivors = final[final["survived_to_phase7"]]
    survivor_counts = survivors["label"].astype(str).value_counts()
    unit_mass = units.groupby(units["gene"].astype(str), observed=True)["n_cells_in_metacell"].sum()
    perturbed = [label for label in survivor_counts.index if label != CONTROL]
    wrong = [label for label in perturbed if int(unit_mass.get(label, -1)) != int(survivor_counts[label])]
    control_units = int((units["gene"].astype(str) == CONTROL).sum())
    control_ok = control_units == int(survivor_counts.get(CONTROL, 0))
    row(ctx, "mass_conservation", expectation_text(ctx, "mass_conservation"),
        f"{len(perturbed)} groups, {len(wrong)} with unit mass != survivors {wrong[:3]}; control units {control_units} vs survivors {int(survivor_counts.get(CONTROL, 0))}",
        "PASS" if not wrong and control_ok else "FAIL")
    ctx["obs_all"], ctx["survivor_counts"], ctx["units"] = obs_all, survivor_counts, units


# ------------------------------------------------------------------------------------------------ dataset-specific
def read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="replace") if Path(path).exists() else ""


def report_text(run):
    return (read_text(run.reports_dir / "REPORT.md") + "\n" + read_text(run.reports_dir / "COVERAGE_REPORT.md")).lower()


def assess_genomewide(ctx):
    run = ctx["run"]
    summary = json.loads((run.stats_dir / "hvg" / "all" / "summary.json").read_text(encoding="utf-8"))
    n_top, panel = summary["n_top_genes"], summary["panel_size"]
    stacked = panel > n_top
    forced = int(summary.get("forced_heldout_targets", 0)) + int(summary.get("forced_protected_genes", 0)) + int(summary.get("forced_cell_cycle_genes", 0))
    # budget policy: exactly n_top, or, when the always-kept genes (held-out targets, protected) alone exceed n_top, exactly those genes
    wanted = stacked if ctx["mode"] in ("current", "compat") else (panel == n_top or (forced > n_top and panel == forced))
    row(ctx, "panel_stacked", expectation_text(ctx, "panel_stacked"),
        f"policy {summary['target_node_policy']}, panel {panel} vs n_top_genes {n_top} (stacked by {summary['panel_stacked_above_n_top']})", "PASS" if wanted else "FAIL")
    split = json.loads((run.splits_dir / f"{run.name}_split_indices.json").read_text(encoding="utf-8"))
    kept = set(read_text(run.stats_dir / "reduce2" / "all" / "gene_keep_mask.txt").split("\n"))
    panel_genes = set(read_text(run.splits_dir / f"{run.name}_hvg_panel.txt").split("\n"))
    heldout = (set(split["val_labels"]) | set(split["test_labels"])) & kept
    missing = heldout - panel_genes
    row(ctx, "heldout_in_panel", expectation_text(ctx, "heldout_in_panel"), f"{len(heldout)} held-out targets in the Phase 6 genes, {len(missing)} missing from the panel",
        "PASS" if not missing else "FAIL")


def assess_lowexpr(ctx):
    truth, derived = ctx["truth"], ctx["derivation"]
    frame = truth.assign(qc_pass=derived["qc_pass"], survivor=derived["survivor"], dropped=truth["label"].isin(derived["dropped_labels"]))
    observed = ctx["observed_set"]
    frame["survivor_xl"] = frame["cell"].isin(observed)
    escapers = frame[frame["is_escaper"] & frame["qc_pass"] & ~frame["dropped"] & frame["cls"].isin(["high", "mid", "low"])]
    rates = {cls: float(1 - escapers.loc[escapers["cls"] == cls, "survivor_xl"].mean()) for cls in ("high", "mid", "low")}
    knocked = frame[~frame["is_escaper"] & frame["qc_pass"] & ~frame["dropped"] & frame["cls"].isin(["high", "mid", "low"])]
    false_removal = {cls: float(1 - knocked.loc[knocked["cls"] == cls, "survivor_xl"].mean()) for cls in ("high", "mid", "low")}
    text = ", ".join(f"{cls}: escapers removed {rates[cls]:.0%} (knocked-down cells wrongly removed {false_removal[cls]:.0%})" for cls in rates)
    row(ctx, "escaper_removal_high", expectation_text(ctx, "escaper_removal_high"), text, "PASS" if rates["high"] >= 0.70 else "FAIL")
    blind_ok = rates["mid"] >= 0.50 and rates["low"] >= 0.50
    row(ctx, "escaper_removal_blind", expectation_text(ctx, "escaper_removal_blind"), text, "PASS" if blind_ok else "FAIL")


def assess_confounded(ctx):
    text = report_text(ctx["run"])
    flagged = "confound" in text and "batch" in text
    row(ctx, "confounding_flagged", expectation_text(ctx, "confounding_flagged"), "'confound' appears in the reports" if flagged else "no confounding table or warning in reports/REPORT.md or COVERAGE_REPORT.md", "PASS" if flagged else "FAIL")
    variance = "variance explained on the hvg subsample" in text and re.search(r"\| batch \| \d+ \| 0\.\d+", text) is not None
    row(ctx, "batch_variance_reported", expectation_text(ctx, "batch_variance_reported"), "found" if variance else "no 'variance explained by batch' line in the reports", "PASS" if variance else "FAIL")


def assess_depth(ctx):
    derived, truth = ctx["derivation"], ctx["truth"]
    shallow = set(ctx["key"]["shallow_labels"])
    expected_dropped = set(derived["dropped_labels"]) & shallow
    final = pd.read_parquet(ctx["run"].manifest_dir / "global_obs_final.parquet")
    present = set(final.loc[final["survived_to_phase7"], "label"].astype(str))
    observed_dropped = shallow - present
    qc_rates = {cls: float(1 - derived["qc_pass"][(truth["cls"] == cls).to_numpy()].mean()) for cls in ("deep", "shallow", "ordinary")}
    row(ctx, "depth_dropout", expectation_text(ctx, "depth_dropout"),
        f"shallow perturbations dropped as undersized: expected {len(expected_dropped)}, XL {len(observed_dropped)}; QC failure rate deep {qc_rates['deep']:.1%}, shallow {qc_rates['shallow']:.1%}, ordinary {qc_rates['ordinary']:.1%}",
        "PASS" if expected_dropped == observed_dropped else "FAIL")


def assess_donors(ctx):
    text = report_text(ctx["run"])
    found = "variance explained on the hvg subsample" in text and re.search(r"\| donor \| \d+ \| 0\.\d+", text) is not None
    row(ctx, "donor_variance_reported", expectation_text(ctx, "donor_variance_reported"), "found" if found else "no 'variance explained by donor' line in the reports", "PASS" if found else "FAIL")


def assess_tiny(ctx):
    key, derived = ctx["key"], ctx["derivation"]
    survivor_counts = ctx["survivor_counts"]
    present = set(survivor_counts.index) - {CONTROL}
    if ctx["arm"].startswith("min50"):
        small = set(key["tiny_labels"]) | set(key["mid_labels"])
        ok = not (present & small) and set(key["big_labels"]) <= present
        row(ctx, "tiny_dropped", expectation_text(ctx, "tiny_dropped"),
            f"{len(present)} perturbations reach the outputs; tiny/mid present {len(present & small)}; big present {len(present & set(key['big_labels']))}/120", "PASS" if ok else "FAIL")
    else:
        tiny_alive = set(key["tiny_labels"]) & present
        units = ctx["units"]
        counts = units.groupby(units["gene"].astype(str), observed=True).size()
        without_unit = [label for label in tiny_alive if int(counts.get(label, 0)) < 1]
        row(ctx, "tiny_kept_survive", expectation_text(ctx, "tiny_kept_survive"),
            f"{len(tiny_alive)} tiny groups survive the filters, {len(without_unit)} have no unit; cells conserved (see mass_conservation)", "PASS" if tiny_alive and not without_unit else "FAIL")


def assess_ctrl(ctx):
    run = ctx["run"]
    assignment = pd.read_parquet(run.splits_dir / f"{run.name}_control_assignment.parquet")
    per_split = assignment["split"].value_counts().to_dict()
    check = "few_controls_ok" if ctx["name"].startswith("07a") else "many_controls_ok"
    thresholds = pd.read_csv(run.stats_dir / "reduce1" / "all" / "escaper_thresholds.tsv", sep="\t")
    row(ctx, check, expectation_text(ctx, check), f"controls per split {per_split}; min control count behind a threshold {int(thresholds['n_ctrl'].min())}; survivors and verify checked above", "PASS")


def assess_messy(ctx):
    obs_all, units = ctx["obs_all"], ctx["units"]
    key = ctx["key"]
    bad_labels = set(key["excel"].values()) | set(key["lowercase"]) | {g.lower() for g in key["split_by_case"]} | {f" {g} " for g in key["padded"]} | {"NT", "Non-Targeting"}
    present = sorted(bad_labels & set(obs_all["gene"].astype(str)))
    cells = int(obs_all["gene"].astype(str).isin(bad_labels).sum())
    row(ctx, "labels_not_silently_accepted", expectation_text(ctx, "labels_not_silently_accepted"),
        f"{len(present)} damaged/variant labels reached the outputs as separate groups ({cells:,} cells), e.g. {present[:6]}", "PASS" if not present else "FAIL")


def assess_dup(ctx):
    run = ctx["run"]
    names = list(run.var_table().index)
    unique = len(names) == len(set(names))
    row(ctx, "var_unique", expectation_text(ctx, "var_unique"), f"{len(names)} var names, unique={unique}", "PASS" if unique else "FAIL")
    thresholds = pd.read_csv(run.stats_dir / "reduce1" / "all" / "escaper_thresholds.tsv", sep="\t")
    used = {}
    for label in ctx["key"]["duplicate_target_labels"]:
        genes = thresholds.loc[thresholds["gene"].isin([label, f"{label}_1"]), "gene"].tolist()
        used[label] = genes
    first_used = sum(1 for genes in used.values() if genes == [genes[0]] and not genes[0].endswith("_1")) if used else 0
    logs = "\n".join(read_text(path) for path in (run.logs_dir).glob("*.log")).lower() + report_text(run)
    flagged = bool(re.search(r"duplicate[^\n]*symbol|ambiguous", logs))
    row(ctx, "duplicates_flagged", expectation_text(ctx, "duplicates_flagged"),
        f"{len(used)} duplicated-symbol targets: XL used the first copy for {first_used}, the second (_1) for {len(used) - first_used}; designed knockdown on the first copy for {sum(ctx['key']['true_copy_is_first'])} of 10; "
        f"{'flagged in log/report' if flagged else 'not flagged anywhere in the logs or reports'}", "PASS" if flagged else "FAIL")


def assess_dense_pair(ctx):
    pass  # the pairwise comparison is done after both arms in compare_with_reference


ASSESS = {"01_genomewide": assess_genomewide, "02_lowexpr": assess_lowexpr, "03_confounded": assess_confounded, "04_depth": assess_depth,
          "05_donors": assess_donors, "06_tiny_groups": assess_tiny, "07a_ctrl_poor": assess_ctrl, "07b_ctrl_heavy": assess_ctrl,
          "08_messy_labels": assess_messy, "09_dup_symbols": assess_dup}


def compare_with_reference(ctx, results_dir, tag):
    reference_name = ctx["spec"]["reference_arm"]
    ref_config = Path(ctx["work"]) / f"{ctx['name']}__{reference_name}.yaml"
    check = "layout_invariance" if ctx["spec"].get("layout_b") else "dense_equals_sparse"
    expected = ("outputs identical to the other shard layout (shard rows, block count, control block rows)" if check == "layout_invariance"
                else expectation_text(ctx, "dense_equals_sparse"))
    if not (results_dir / f"{ctx['name']}__{reference_name}.json").exists():
        row(ctx, check, expected, f"reference arm {reference_name} has no results yet", "NA")
        return
    command = [sys.executable, "-B", COMPARE, "xl", "--a", str(ref_config), "--b", str(ctx["config_path"])]
    log_path = Path(ctx["work"]) / f"{ctx['name']}__{ctx['arm']}_vs_{reference_name}.out"
    code, _ = execute(ctx, command, log_path)
    text = read_text(log_path).strip().split("\n")
    verdict = next((line for line in text if line.startswith(("IDENTICAL", "DIFFERENT", "REFUSED"))), text[-1] if text else "")
    row(ctx, check, expected, verdict, "PASS" if code == 0 else "FAIL")


def compare_with_frozen_code(ctx, args):
    """Compat v4 code against the run of the frozen 21 Oct code (same dataset and arm): every output signature identical."""
    import yaml
    reference_config = Path(args.work) / args.reference_tag / f"{ctx['name']}__{ctx['arm']}.yaml"
    expected = "outputs identical to the run of the frozen 21 Oct code (split, panel, thresholds, hvgraw, p9, metacell units)"
    if not reference_config.exists():
        row(ctx, "identical_to_frozen_code", expected, f"no reference run at {reference_config}", "NA")
        return
    cfg = yaml.safe_load(reference_config.read_text(encoding="utf-8"))
    cfg.setdefault("xl", {})["compat"] = "v3"
    copy_path = Path(ctx["work"]) / f"{ctx['name']}__{ctx['arm']}__refcompat.yaml"
    copy_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    command = [sys.executable, "-B", COMPARE, "xl", "--a", str(copy_path), "--b", str(ctx["config_path"])]
    code, _ = execute(ctx, command, Path(ctx["work"]) / f"{ctx['name']}__{ctx['arm']}_vs_frozen.out")
    text = read_text(Path(ctx["work"]) / f"{ctx['name']}__{ctx['arm']}_vs_frozen.out").strip().splitlines()
    verdict = next((line for line in text if line.startswith(("IDENTICAL", "DIFFERENT", "REFUSED"))), text[-1] if text else "")
    row(ctx, "identical_to_frozen_code", expected, verdict, "PASS" if code == 0 else "FAIL")


# ------------------------------------------------------------------------------------------------ one arm
def run_arm(args, name, arm, spec, env):
    data, work, results = Path(args.data), Path(args.work) / args.tag, Path(args.results)
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    result_path = results / f"{name}__{arm}.json"
    if result_path.exists():
        print(f"skip {name}/{arm}: results exist")
        return
    key = json.loads((data / f"{name}_key.json").read_text(encoding="utf-8"))
    truth = pd.read_parquet(data / f"{name}_truth.parquet")
    source_path = data / f"{name}.h5ad"
    if spec.get("twin"):
        source_path = work / f"{name}_sparse_twin.h5ad"
        if not source_path.exists():
            dense = ad.read_h5ad(data / f"{name}.h5ad")
            dense.X = sp.csr_matrix(dense.X)
            dense.write_h5ad(source_path, compression="lzf")
    run_dir = work / f"{name}__{arm}"
    config_path = work / f"{name}__{arm}.yaml"
    source_overrides, xl_overrides = {}, {}
    if spec.get("layout_b"):
        source_overrides["shard_rows"] = int(math.ceil(key["n_cells"] / 3)) + 777
        xl_overrides.update({"n_blocks": 5, "control_block_rows": 500})
    if args.compat:
        xl_overrides["compat"] = "v3"
    cfg = make_config(key, name, arm, source_path, run_dir, config_path, data / f"{name}_protected.json", source_overrides,
                      spec.get("config_overrides"), xl_overrides)
    ctx = {"name": name, "arm": arm, "key": key, "truth": truth, "rows": [], "env": env, "cfg": cfg, "mode": args.mode, "spec": spec,
           "work": work, "config_path": config_path}
    out_path = work / f"{name}__{arm}.out"
    command = [sys.executable, "-B", CLI, "all-local", "--config", str(config_path), "--workers", str(args.workers),
               "--max-ram-gb", "20"]
    code, seconds = execute(ctx, command, out_path)
    print(f"{name}/{arm}: all-local rc {code} in {seconds:.0f}s")
    try:
        if name == "11_normalized":
            manifest = run_dir / "manifest"
            wrote = sorted(p.name for p in manifest.iterdir()) if manifest.exists() else []
            shards = (run_dir / "_xl_work" / "source_shards")
            wrote_shards = shards.exists() and any(shards.iterdir())
            message = tail(out_path, 6)
            clear = code != 0 and re.search(r"raw|integer|normali", message, re.I) is not None
            ok = code != 0 and clear and not wrote and not wrote_shards
            row(ctx, "refused_before_writing", expectation_text(ctx, "refused_before_writing"),
                f"rc {code}; message {'mentions raw/integer/normalized' if clear else 'does not mention raw counts'}; manifest files written: {wrote}; source shards written: {wrote_shards}; tail: {message[:300]}",
                "PASS" if ok else "FAIL", None if ok else "code bug (refusal comes too late or is unclear)")
        elif code != 0:
            row(ctx, "completes", expectation_text(ctx, "completes"), f"rc {code}: {tail(out_path, 8)[:700]}", "FAIL")
            for check in ("verify", "survivors", "obs_fidelity", "x_fidelity", "p9_sanity", "mass_conservation"):
                row(ctx, check, expectation_text(ctx, check), "not run: pipeline did not complete", "NA")
        else:
            from xl_common import Run
            ctx["run"] = Run(config_path)
            labels = ctx["truth"][key.get("label_column", "label")].to_numpy()
            symbols = None
            if name == "09_dup_symbols":
                symbols = ad.read_h5ad(source_path, backed="r").var["gene_name"].astype(str).tolist()
            ctx["derivation"] = independent_derivation(source_path, "gene", symbols=symbols,
                                                       min_cells=cfg["phase5_escaper_filtering"].get("min_cells_per_perturbation", 50))
            ctx["source_matrix"] = sp.csr_matrix(ad.read_h5ad(source_path).X)
            universal(ctx)
            ctx["observed_set"] = xl_survivor_names(ctx["run"], ctx["derivation"]["names"])
            if name in ASSESS:
                ASSESS[name](ctx)
            if spec.get("reference_arm"):
                compare_with_reference(ctx, results, args.tag)
            if args.reference_tag and args.compat:
                compare_with_frozen_code(ctx, args)
            times = pd.read_csv(ctx["run"].reports_dir / "memory_and_time.tsv", sep="\t")
            ctx["timing"] = {"task_seconds": float(times["seconds"].sum()), "peak_rss_gb": float(times["peak_rss_gb"].max())}
    except Exception as error:  # noqa: BLE001
        import traceback
        row(ctx, "runner_error", "the assessment itself runs", traceback.format_exc()[-800:], "FAIL", "INVESTIGATE (runner)")
        print(traceback.format_exc())
    result_path.write_text(json.dumps({"name": name, "arm": arm, "return_code": code, "wall_seconds": seconds, "timing": ctx.get("timing"),
                                       "n_cells": key["n_cells"], "rows": ctx["rows"]}, indent=2, default=str), encoding="utf-8")
    for item in ctx["rows"]:
        print(f"   {item['status']:4s} {item['id']}")


def aggregate(args):
    results = Path(args.results)
    rows, header = [], []
    for path in sorted(results.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data["rows"]:
            rows.append({**item, "wall_s": round(data["wall_seconds"]), "cells": data["n_cells"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(results / f"stress_results_{args.tag}.tsv", sep="\t", index=False)
    lines = [f"# Stress suite results, tag {args.tag}, mode {args.mode}", "",
             f"{len(frame)} checks: PASS {int((frame['status'] == 'PASS').sum())}, FAIL {int((frame['status'] == 'FAIL').sum())}, NA {int((frame['status'] == 'NA').sum())}", "",
             "| dataset | arm | check | status | class | observed |", "|---|---|---|---|---|---|"]
    for item in rows:
        lines.append(f"| {item['dataset']} | {item['arm']} | {item['id']} | {item['status']} | {item['class']} | {str(item['observed']).replace('|', '/')[:240]} |")
    (results / f"stress_results_{args.tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--only", default=None, help="comma-separated dataset names")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mode", default="current", choices=["current", "compat", "production"])
    parser.add_argument("--compat", action="store_true", help="write xl.compat: v3 into every config (v4 code reproducing SPORE v3.0)")
    parser.add_argument("--reference-tag", default=None, help="work-folder tag of an earlier run (for example s1): the signatures must be identical")
    args = parser.parse_args()
    import os
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", PYTHONUTF8="1")
    names = sorted(path.name[:-len("_key.json")] for path in Path(args.data).glob("*_key.json"))
    if args.only:
        names = [name for name in names if name in args.only.split(",")]
    for name in names:
        arms = ARMS.get(name, {"default": {}})
        for arm, spec in arms.items():
            run_arm(args, name, arm, spec, env)
    aggregate(args)


if __name__ == "__main__":
    main()
