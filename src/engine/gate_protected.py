"""Protected-list expression gate (CD4 Rest): detection fraction of every listed gene in CONTROL cells, from the intake shards.

  python -B src\\gate_protected.py --config run.yaml --protected SPORE\\protected_genes\\protected_genes_CD4_T_Rest.json
         --out-json SPORE\\protected_genes\\protected_genes_CD4_T_Rest_v1.1.json --out-tsv protected_gate_report.tsv
  python -B src\\gate_protected.py --shards "<dir>\\shard_*\\shard.h5ad" ...        (explicit shard files instead of a run config)

Rule (Director, task 8): genes detected in < 1 percent of control cells (--min-fraction 0.01) are REMOVED from the tahoe_menu tier
(the drug-target menu: no expression, no node) and only FLAGGED for review in tiers 1 and 2 (they stay in the list; a human decides).
Genes that are not in the matrix at all count as 0 percent. Other tier-3 categories are reported, not changed.
The output JSON is a NEW file (a copy of the input with metadata.expression_gate added); the tool refuses to overwrite anything and
refuses an --out-json that is the input file. Controls are read row by row from the shards (backed), never the whole matrix.
"""
import argparse
import copy
import glob
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def control_detection(shard_paths, control_label, pert_col="gene", chunk_rows=5000):
    """(gene names, n control cells, detected-cell count per gene) over all shards (gene list must be identical across shards)."""
    genes, n_controls, detected = None, 0, None
    for path in shard_paths:
        backed = ad.read_h5ad(path, backed="r")
        try:
            names = list(map(str, backed.var_names))
            if genes is None:
                genes, detected = names, np.zeros(len(names), dtype=np.int64)
            elif names != genes:
                raise ValueError(f"{path}: gene list differs from the first shard; all shards of one run must share ONE gene list (the intake of build r3 "
                                 f"harmonizes it: redo `intake_list` and `intake` in a fresh work dir)")
            rows = np.flatnonzero((backed.obs[pert_col].astype(str) == control_label).to_numpy())
            for start in range(0, len(rows), chunk_rows):
                part = rows[start:start + chunk_rows]
                block = backed.X[part]
                block = block.tocsc() if hasattr(block, "tocsc") else block
                detected += np.asarray((block > 0).sum(axis=0)).ravel().astype(np.int64)
            n_controls += len(rows)
        finally:
            backed.file.close()
    if genes is None or n_controls == 0:
        raise ValueError("no control cells found in the shards")
    return genes, n_controls, detected


def gate(protected, genes, n_controls, detected, min_fraction=0.01):
    """Returns (new protected dict, report DataFrame)."""
    index = {name: i for i, name in enumerate(genes)}
    new = copy.deepcopy(protected)
    rows, removed, flagged = [], [], []
    for category, spec in protected["protected_genes"].items():
        tier = spec.get("tier")
        kept = []
        for gene in spec.get("genes", []):
            in_matrix = gene in index
            fraction = float(detected[index[gene]]) / n_controls if in_matrix else 0.0
            low = fraction < min_fraction
            if low and category == "tahoe_menu":
                action = "removed_from_tahoe_menu"
                removed.append(gene)
            else:
                kept.append(gene)
                if low and tier in (1, 2):
                    action = "flagged_review_not_in_matrix" if not in_matrix else "flagged_review_low_detection"
                    flagged.append({"gene": gene, "category": category, "tier": tier, "detection_fraction": round(fraction, 6)})
                elif low:
                    action = "low_detection_kept_other_tier3"
                else:
                    action = "kept"
            rows.append({"gene": gene, "category": category, "tier": tier, "in_matrix": in_matrix,
                         "n_control_cells": n_controls, "n_detected": int(detected[index[gene]]) if in_matrix else 0,
                         "detection_fraction": round(fraction, 6), "action": action})
        new["protected_genes"][category]["genes"] = kept
    metadata = new.setdefault("metadata", {})
    metadata["version"] = "1.1"
    metadata["expression_gate"] = {
        "rule": f"genes detected in < {min_fraction:.0%} of control cells: removed from tahoe_menu, flagged for review in tiers 1-2",
        "min_fraction": min_fraction, "n_control_cells": int(n_controls), "n_shards_note": "intake shards of the Rest arm",
        "removed_from_tahoe_menu": sorted(removed), "flagged_for_review": flagged,
        "n_genes_before": int(sum(len(s.get("genes", [])) for s in protected["protected_genes"].values())),
        "n_genes_after": int(sum(len(s.get("genes", [])) for s in new["protected_genes"].values()))}
    return new, pd.DataFrame(rows)


def gate_enabled(cfg):
    return bool((cfg.get("phase8_hvg", {}) or {}).get("protected_expression_gate", False))


def gated_paths(run):
    """(gated list, gate report): both in the RUN folder, never in the repository, never over the input list."""
    return run.manifest_dir / "protected_genes_gated.json", run.manifest_dir / "protected_gate_report.tsv"


def run_expression_gate(run):
    """The optional protected-list expression gate (phase8_hvg.protected_expression_gate: true) as a step of `spore_xl.py run`, after the intake and before plan.
    Genes detected in < protected_expression_min_fraction (default 0.01) of the CONTROL cells are removed from the tahoe_menu tier and only flagged in tiers 1-2.
    Reads the intake shards (recipe sources) or the source .h5ad (h5ad sources; var names must be gene symbols). Writes manifest/protected_genes_gated.json and
    manifest/protected_gate_report.tsv; Run then points phase8_hvg.protected_gene_list at the gated file for plan, hvg and every worker. Skipped when it exists."""
    import hashlib
    from xl_common import write_new_json, write_new_tsv
    out_json, out_tsv = gated_paths(run)
    if out_json.exists():
        run.logger.info(f"[gate] already done, skipped: {out_json}")
        return out_json
    p8 = run.cfg.get("phase8_hvg", {}) or {}
    original = p8.get("protected_gene_list")
    if not original or not Path(original).exists():
        raise ValueError("phase8_hvg.protected_expression_gate needs phase8_hvg.protected_gene_list to name an existing list (the INPUT of the gate)")
    source = (run.xl.get("source") or {})
    if source.get("kind") == "recipe":
        import stage_intake
        stage_intake.assert_uniform_var(run, require_all=True)
        shard_paths = sorted(run.work_dir.glob("intake/shard_*/shard.h5ad"))
    else:
        if str(run.cfg.get("dataset", {}).get("gene_id_format", "symbol")) != "symbol":
            raise ValueError("the expression gate reads gene symbols from the source: set dataset.gene_id_format: symbol (the source's var names must be symbols) "
                             "or switch phase8_hvg.protected_expression_gate off")
        shard_paths = [Path(source["h5ad"])]
    if not shard_paths:
        raise FileNotFoundError("the expression gate found no intake shards / source file to read the controls from")
    min_fraction = float(p8.get("protected_expression_min_fraction", 0.01))
    run.logger.info(f"[gate] reading the control cells of {len(shard_paths)} file(s) (genes detected in < {min_fraction:.0%} of controls are removed from the tahoe_menu tier)")
    genes, n_controls, detected = control_detection(shard_paths, run.ctrl_label, run.pert_col)
    protected = json.loads(Path(original).read_text(encoding="utf-8"))
    new, report = gate(protected, genes, n_controls, detected, min_fraction)
    version = (protected.get("metadata", {}) or {}).get("version", "unversioned")
    new["metadata"]["version"] = f"{version}+expression_gate"
    new["metadata"]["expression_gate"].update({"n_shards_note": f"{len(shard_paths)} file(s) read", "input_list": str(Path(original).resolve()), "input_version": version,
                                               "input_name": (protected.get("metadata", {}) or {}).get("name"),
                                               "input_sha256": hashlib.sha256(Path(original).read_bytes()).hexdigest()})
    write_new_json(out_json, new)
    write_new_tsv(out_tsv, report)
    gate_meta = new["metadata"]["expression_gate"]
    run.logger.info(f"[gate] done: {gate_meta['n_genes_before']} genes in the input list (version {version}) -> {gate_meta['n_genes_after']}; removed from tahoe_menu "
                    f"{len(gate_meta['removed_from_tahoe_menu'])}, flagged for review {len(gate_meta['flagged_for_review'])}; {out_json}")
    return out_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gate_protected.py")
    parser.add_argument("--config", default=None, help="XL run config whose intake shards (_xl_work/intake/shard_*/shard.h5ad) are used")
    parser.add_argument("--shards", default=None, help="glob of shard .h5ad files (alternative to --config)")
    parser.add_argument("--protected", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--min-fraction", type=float, default=0.01)
    parser.add_argument("--control-label", default=None)
    parser.add_argument("--pert-col", default=None)
    args = parser.parse_args(argv)
    control_label, pert_col = args.control_label, args.pert_col
    if args.config:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from xl_common import Run
        run = Run(args.config)
        shard_paths = sorted(run.work_dir.glob("intake/shard_*/shard.h5ad"))
        import stage_intake
        stage_intake.assert_uniform_var(run, require_all=False)   # every finished shard must have the run's one gene list (see intake_genes.tsv)
        control_label, pert_col = control_label or run.ctrl_label, pert_col or run.pert_col
    elif args.shards:
        shard_paths = sorted(Path(p) for p in glob.glob(args.shards))
        control_label, pert_col = control_label or "non-targeting", pert_col or "gene"
    else:
        parser.error("give --config or --shards")
    if not shard_paths:
        raise FileNotFoundError("no intake shards found (were they removed by the opt-in clean-up? run the gate before `map1` finishes)")
    out_json, out_tsv = Path(args.out_json), Path(args.out_tsv)
    if out_json.resolve() == Path(args.protected).resolve():
        raise ValueError("--out-json is the input file: the gate writes a NEW file and never overwrites the list it read")
    for path in (out_json, out_tsv):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    protected = json.loads(Path(args.protected).read_text(encoding="utf-8"))
    genes, n_controls, detected = control_detection(shard_paths, control_label, pert_col)
    new, report = gate(protected, genes, n_controls, detected, args.min_fraction)
    new["metadata"]["expression_gate"]["n_shards"] = len(shard_paths)
    out_json.write_text(json.dumps(new, indent=2), encoding="utf-8")
    report.to_csv(out_tsv, sep="\t", index=False)
    summary = report["action"].value_counts().to_dict()
    print(f"{len(shard_paths)} shards, {n_controls:,} control cells; {summary}")
    print(f"wrote {out_json}\nwrote {out_tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
