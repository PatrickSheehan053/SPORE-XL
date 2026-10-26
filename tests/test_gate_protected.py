"""gate_protected on synthetic shards with KNOWN control detection fractions.
Run: python -B tests\\test_gate_protected.py   (XL_TEST_TMP = scratch folder)"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_test_support import scratch_for  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import gate_protected  # noqa: E402

GENES = ["TCF7", "LEF1", "CD3E", "IL2", "LOWT1", "TAHOE_HI", "TAHOE_LO", "TAHOE_ZERO", "QUIES_LO", "OTHER"]
# detection fraction among controls per gene (of 1,000 controls in total over 2 shards)
FRACTION = {"TCF7": 0.50, "LEF1": 0.004, "CD3E": 0.30, "IL2": 0.0, "LOWT1": 0.008, "TAHOE_HI": 0.05, "TAHOE_LO": 0.008,
            "TAHOE_ZERO": 0.0, "QUIES_LO": 0.002, "OTHER": 0.9}


def make_shards(work):
    rng = np.random.default_rng(0)
    paths = []
    for shard in range(2):
        n_ctrl, n_pert = 500, 400
        rows = []
        for gene in GENES:
            detected = int(round(FRACTION[gene] * 1000 / 2))
            column = np.zeros(n_ctrl + n_pert, dtype=np.float32)
            column[rng.choice(n_ctrl, size=detected, replace=False)] = 3
            column[n_ctrl:] = rng.integers(0, 3, size=n_pert)  # perturbed cells must NOT count
            rows.append(column)
        matrix = sp.csr_matrix(np.vstack(rows).T)
        labels = ["non-targeting"] * n_ctrl + [f"T{i % 7}" for i in range(n_pert)]
        obs = pd.DataFrame({"gene": pd.Categorical(labels)}, index=[f"s{shard}_{i}" for i in range(n_ctrl + n_pert)])
        path = work / f"shard_{shard}.h5ad"
        ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=GENES)).write_h5ad(path)
        paths.append(path)
    return paths


def protected_list():
    return {"metadata": {"version": "1.0", "name": "synthetic"}, "protected_genes": {
        "tcell_identity_tfs": {"tier": 1, "genes": ["TCF7", "LEF1", "CD3E"]},
        "node_only": {"tier": 2, "genes": ["IL2", "LOWT1", "NOT_IN_MATRIX"]},
        "tahoe_menu": {"tier": 3, "genes": ["TAHOE_HI", "TAHOE_LO", "TAHOE_ZERO", "NOT_IN_MATRIX_TAHOE"]},
        "quiescence_cell_cycle": {"tier": 3, "genes": ["QUIES_LO", "OTHER"]}}}


def test_gate_rules_and_outputs():
    work = scratch_for("gate")
    paths = make_shards(work)
    protected_path = work / "protected_v1.0.json"
    protected_path.write_text(json.dumps(protected_list()), encoding="utf-8")
    out_json, out_tsv = work / "protected_v1.1.json", work / "gate.tsv"
    code = gate_protected.main(["--shards", str(work / "shard_*.h5ad"), "--protected", str(protected_path),
                                "--out-json", str(out_json), "--out-tsv", str(out_tsv)])
    assert code == 0
    new = json.loads(out_json.read_text(encoding="utf-8"))
    report = pd.read_csv(out_tsv, sep="\t").set_index("gene")
    # detection fractions are exact and exclude perturbed cells
    for gene in GENES:
        assert abs(report.loc[gene, "detection_fraction"] - FRACTION[gene]) < 1e-9, gene
    assert (report["n_control_cells"] == 1000).all()
    tahoe = new["protected_genes"]["tahoe_menu"]["genes"]
    assert tahoe == ["TAHOE_HI"], tahoe  # LO (0.99%), ZERO and the gene absent from the matrix are removed
    assert new["protected_genes"]["tcell_identity_tfs"]["genes"] == ["TCF7", "LEF1", "CD3E"], "tiers 1-2 are flagged, never removed"
    assert new["protected_genes"]["node_only"]["genes"] == ["IL2", "LOWT1", "NOT_IN_MATRIX"]
    assert new["protected_genes"]["quiescence_cell_cycle"]["genes"] == ["QUIES_LO", "OTHER"], "other tier-3 categories are reported, not changed"
    gate_meta = new["metadata"]["expression_gate"]
    assert new["metadata"]["version"] == "1.1"
    assert gate_meta["removed_from_tahoe_menu"] == ["NOT_IN_MATRIX_TAHOE", "TAHOE_LO", "TAHOE_ZERO"]
    assert sorted(item["gene"] for item in gate_meta["flagged_for_review"]) == ["IL2", "LEF1", "LOWT1", "NOT_IN_MATRIX"]
    assert gate_meta["n_genes_before"] == 12 + 0 and gate_meta["n_genes_after"] == 9, gate_meta
    assert report.loc["LEF1", "action"] == "flagged_review_low_detection" and report.loc["NOT_IN_MATRIX", "action"] == "flagged_review_not_in_matrix"
    assert report.loc["QUIES_LO", "action"] == "low_detection_kept_other_tier3" and report.loc["OTHER", "action"] == "kept"
    print("    exact control detection fractions; tahoe_menu low genes removed (incl. absent); tiers 1-2 flagged and kept; report and v1.1 json written")


def test_never_overwrites_and_never_writes_over_the_input():
    work = scratch_for("gate2")
    paths = make_shards(work)
    protected_path = work / "p.json"
    protected_path.write_text(json.dumps(protected_list()), encoding="utf-8")
    for args in (["--out-json", str(protected_path), "--out-tsv", str(work / "a.tsv")],):
        try:
            gate_protected.main(["--shards", str(work / "shard_*.h5ad"), "--protected", str(protected_path), *args])
        except ValueError as error:
            assert "NEW file" in str(error)
        else:
            raise AssertionError("writing over the input list must be refused")
    existing = work / "exists.json"
    existing.write_text("{}", encoding="utf-8")
    try:
        gate_protected.main(["--shards", str(work / "shard_*.h5ad"), "--protected", str(protected_path), "--out-json", str(existing), "--out-tsv", str(work / "b.tsv")])
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing output must be refused")
    assert json.loads(protected_path.read_text(encoding="utf-8"))["metadata"]["version"] == "1.0"


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
