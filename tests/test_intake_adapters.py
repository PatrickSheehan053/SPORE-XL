"""P7: every intake adapter on a synthetic file, plus label repair and the refusals.
Run: set PYTHONHASHSEED=0 ; python -B tests\\test_intake_adapters.py   (XL_TEST_TMP = scratch folder, default: system temp)
"""
import gzip
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "engine"))
sys.path.insert(0, str(ROOT))

import anndata as ad  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import yaml  # noqa: E402

from src.intake import intake_adapters as adapters  # noqa: E402

N_CELLS, N_GENES = 900, 60
GENES = ["SEPT1", "MARCH1", "JUN", "MARC1"] + [f"GENE{number:03d}" for number in range(N_GENES - 4)]
PERTS = ["GENE001", "GENE002", "GENE003", "JUN"]


def scratch(prefix):
    return Path(tempfile.mkdtemp(prefix=prefix, dir=os.environ.get("XL_TEST_TMP", tempfile.gettempdir())))


def make_data(seed=1):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rng.uniform(0.2, 4, N_GENES), size=(N_CELLS, N_GENES)).astype(np.float32)
    labels = np.array((PERTS * 150) + ["NT"] * 300)
    rng.shuffle(labels)
    barcodes = np.array([f"CELL{index:05d}" for index in range(N_CELLS)])
    batch = rng.choice(["L1", "L2", "L3"], size=N_CELLS)
    return counts, labels, barcodes, batch


def recipe_for(name, source, label_column="target", extra=None):
    recipe = {"dataset": name, "source": source, "obs": {"label_column": label_column, "batch_column": "lane"},
              "control_labels": ["NT"], "control_label_out": "non-targeting", "label_fixes": {"strip": True, "excel_dates": True, "case_fold_to_genes": True}}
    recipe.update(extra or {})
    return recipe


def run_convert(recipe, folder):
    recipe_path = folder / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    out = folder / "out"
    adapters.convert(recipe_path, out, window=250, log=lambda text: None)
    return ad.read_h5ad(out / f"{recipe['dataset']}.h5ad"), out


def check_output(result, counts, labels, barcodes, expect_cells=None):
    assert result.n_vars == counts.shape[1]
    order = {name: index for index, name in enumerate(barcodes)}
    rows = np.array([order[name] for name in result.obs_names])
    assert np.array_equal(sp.csr_matrix(result.X).toarray(), counts[rows]), "counts differ from the source"
    expected_labels = np.where(labels[rows] == "NT", "non-targeting", labels[rows])
    assert np.array_equal(result.obs["gene"].astype(str).to_numpy(), expected_labels)
    assert list(result.obs["gene"].cat.categories) == sorted(set(expected_labels)), "categories must be sorted (Amendment 1 item 5)"
    assert result.n_obs == (expect_cells or len(rows))


def base_obs(labels, barcodes, batch):
    return pd.DataFrame({"target": labels, "lane": batch}, index=pd.Index(barcodes, name="barcode"))


# ------------------------------------------------------------------------------------------------ one test per format
def test_h5ad_sparse_and_dense_and_layer():
    counts, labels, barcodes, batch = make_data()
    folder = scratch("intake_h5ad_")
    var = pd.DataFrame(index=pd.Index(GENES, name="gene"))
    for kind, matrix in (("sparse", sp.csr_matrix(counts)), ("dense", counts)):
        path = folder / f"{kind}.h5ad"
        ad.AnnData(X=matrix, obs=base_obs(labels, barcodes, batch), var=var).write_h5ad(path)
        result, _ = run_convert(recipe_for(f"h5ad_{kind}", {"format": "h5ad", "path": str(path)}), scratch(f"intake_{kind}_"))
        check_output(result, counts, labels, barcodes)
    # normalized X with raw counts in layers/counts
    normalized = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4).astype(np.float32)
    path = folder / "layer.h5ad"
    adata = ad.AnnData(X=sp.csr_matrix(normalized), obs=base_obs(labels, barcodes, batch), var=var)
    adata.layers["counts"] = sp.csr_matrix(counts)
    adata.write_h5ad(path)
    try:
        run_convert(recipe_for("h5ad_norm", {"format": "h5ad", "path": str(path)}), scratch("intake_norm_"))
        raise AssertionError("normalized X must be refused")
    except adapters.NotRawCountsError as error:
        assert "raw" in str(error).lower()
    result, _ = run_convert(recipe_for("h5ad_layer", {"format": "h5ad", "path": str(path), "x_key": "layers/counts"}), scratch("intake_layer_"))
    check_output(result, counts, labels, barcodes)


def write_10x_h5(path, counts, barcodes, guides=None):
    names = [f"ENSG{index:08d}" for index in range(counts.shape[1])]
    all_counts = counts if guides is None else np.hstack([counts, guides[0]])
    feature_names = GENES + ([] if guides is None else guides[1])
    types = ["Gene Expression"] * counts.shape[1] + ([] if guides is None else ["CRISPR Guide Capture"] * len(guides[1]))
    ids = names + ([] if guides is None else guides[1])
    matrix = sp.csc_matrix(all_counts.T)  # features x barcodes
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=matrix.data.astype(np.int32))
        group.create_dataset("indices", data=matrix.indices.astype(np.int64))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
        group.create_dataset("shape", data=np.array(matrix.shape, dtype=np.int32))
        group.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        features = group.create_group("features")
        features.create_dataset("id", data=np.array(ids, dtype="S"))
        features.create_dataset("name", data=np.array(feature_names, dtype="S"))
        features.create_dataset("feature_type", data=np.array(types, dtype="S"))


def test_10x_h5_and_guide_assignment():
    counts, labels, barcodes, batch = make_data()
    folder = scratch("intake_10x_")
    path = folder / "sample.h5"
    write_10x_h5(path, counts, barcodes)
    obs_table = base_obs(labels, barcodes, batch)
    # a 10x file carries no obs table: the recipe merges one from a TSV
    result = adapters.Tenx10xH5Reader(path)
    assert result.n_vars == N_GENES and result.n_obs == N_CELLS and list(result.var["gene_name"]) == GENES
    assert np.array_equal(result.read_block(100, 250).toarray(), counts[100:250])
    # guide counts: 3 targets x 2 guides + 1 control guide pair; every cell has one target or control at >= 3 UMI
    guide_names = ["GENE001-1", "GENE001-2", "GENE002-1", "GENE002-2", "NT-1", "NT-2"]
    guide_counts = np.zeros((N_CELLS, len(guide_names)), dtype=np.float32)
    rng = np.random.default_rng(3)
    truth = np.empty(N_CELLS, dtype=object)
    for cell in range(N_CELLS):
        pick = int(rng.integers(0, 3))
        pair = [0, 1] if pick == 0 else [2, 3] if pick == 1 else [4, 5]
        guide_counts[cell, rng.choice(pair)] = rng.integers(3, 20)
        truth[cell] = ["GENE001", "GENE002", "non-targeting"][pick]
    guide_counts[0, 0] = 5
    guide_counts[0, 2] = 5   # two targets: dropped
    guide_counts[1, :] = 0   # no guide: dropped
    guide_counts[2, :] = 0
    guide_counts[2, 1] = 2   # below min_umi: dropped
    write_10x_h5(path.with_name("crispr.h5"), counts, barcodes, (guide_counts, guide_names))
    reader = adapters.Tenx10xH5Reader(path.with_name("crispr.h5"))
    matrix, names = reader.load_guides()
    assigned, report = adapters.assign_from_guides(matrix, names, 3)
    assert report == {"no_guide": 2, "multi_target": 1, "assigned": N_CELLS - 3}, report
    assert (assigned[3:] == truth[3:]).all() and (assigned[:3] == "").all()


def test_10x_mtx_loom_csv_tsv_dense_tables():
    counts, labels, barcodes, batch = make_data()
    folder = scratch("intake_formats_")
    obs_table = base_obs(labels, barcodes, batch)
    # mtx folder (genes x cells, gz)
    from scipy.io import mmwrite
    mtx = folder / "mtx"
    mtx.mkdir()
    mmwrite(str(mtx / "matrix.mtx"), sp.coo_matrix(counts.T.astype(np.int64)))
    with open(mtx / "matrix.mtx", "rb") as source, gzip.open(mtx / "matrix.mtx.gz", "wb") as target:
        target.write(source.read())
    (mtx / "matrix.mtx").rename(mtx / "matrix.mtx.plain")
    pd.Series(barcodes).to_csv(mtx / "barcodes.tsv.gz", sep="\t", index=False, header=False)
    pd.DataFrame({0: GENES, 1: GENES, 2: ["Gene Expression"] * N_GENES}).to_csv(mtx / "features.tsv.gz", sep="\t", index=False, header=False)
    reader = adapters.Tenx10xMtxReader(mtx)
    assert np.array_equal(reader.read_block(0, N_CELLS).toarray(), counts)
    # loom
    loom = folder / "data.loom"
    with h5py.File(loom, "w") as handle:
        handle.create_dataset("matrix", data=counts.T)
        handle.create_group("col_attrs").create_dataset("CellID", data=np.array(barcodes, dtype="S"))
        handle["col_attrs"].create_dataset("target", data=np.array(labels, dtype="S"))
        handle["col_attrs"].create_dataset("lane", data=np.array(batch, dtype="S"))
        handle.create_group("row_attrs").create_dataset("Gene", data=np.array(GENES, dtype="S"))
    result, _ = run_convert(recipe_for("loom_case", {"format": "loom", "path": str(loom)}), scratch("intake_loom_"))
    check_output(result, counts, labels, barcodes)
    # csv genes x cells, tsv cells x genes (labels come from a cell table merged as obs: recipe uses the label table through extra obs file)
    frame = pd.DataFrame(counts.T, index=GENES, columns=barcodes)
    csv_path, tsv_path = folder / "g_by_c.csv", folder / "c_by_g.tsv"
    frame.to_csv(csv_path)
    frame.T.to_csv(tsv_path, sep="\t")
    for path, fmt, orientation in ((csv_path, "csv", "genes_by_cells"), (tsv_path, "tsv", "cells_by_genes")):
        reader = adapters.open_reader({"source": {"format": fmt, "path": str(path), "orientation": orientation}})
        assert np.array_equal(reader.read_block(0, N_CELLS).toarray(), counts), fmt
    try:
        adapters.open_reader({"source": {"format": "csv", "path": str(csv_path)}})
        raise AssertionError("csv without orientation must be refused")
    except adapters.IntakeError as error:
        assert "orientation" in str(error)
    # dense HDF5 + R-exported tables (the DLD-1 route), cells x genes and genes x cells, and the gene-axis mismatch
    h5_path = folder / "assay.h5"
    cells = pd.DataFrame({"cell_index": range(N_CELLS), "barcode": barcodes, "target": labels, "lane": batch})
    genes = pd.DataFrame({"gene_index": range(N_GENES), "gene_id": GENES})
    cells.to_csv(folder / "cells.tsv.gz", sep="\t", index=False)
    genes.to_csv(folder / "genes.tsv", sep="\t", index=False)
    for orientation, array in (("cells_by_genes", counts.astype(np.int32)), ("genes_by_cells", counts.T.astype(np.int32))):
        with h5py.File(h5_path, "w") as handle:
            handle.create_dataset("assay001", data=array, chunks=(min(100, array.shape[0]), min(20, array.shape[1])), compression="gzip")
        source = {"format": "dense_h5_tables", "path": str(h5_path), "h5_dataset": "assay001", "cells_table": str(folder / "cells.tsv.gz"),
                  "genes_table": str(folder / "genes.tsv")}
        result, _ = run_convert(recipe_for("dense_" + orientation, source), scratch("intake_dense_"))
        check_output(result, counts, labels, barcodes)
        h5_path.rename(folder / f"assay_{orientation}.h5")
    short = genes.iloc[:23].copy()  # 23 names for 60 gene rows: the DLD-1 discrepancy
    short.to_csv(folder / "genes_short.tsv", sep="\t", index=False)
    with h5py.File(folder / "assay_bad.h5", "w") as handle:
        handle.create_dataset("assay001", data=counts.astype(np.int32))
    try:
        adapters.open_reader({"source": {"format": "dense_h5_tables", "path": str(folder / "assay_bad.h5"), "h5_dataset": "assay001",
                                         "cells_table": str(folder / "cells.tsv.gz"), "genes_table": str(folder / "genes_short.tsv")}})
        raise AssertionError("a gene axis that does not match the table must be refused")
    except adapters.IntakeError as error:
        assert "gene axis" in str(error) and "23" in str(error)


# ------------------------------------------------------------------------------------------------ labels
def label_recipe(extra=None):
    recipe = {"control_labels": ["NT", "ntc"], "control_label_out": "non-targeting", "label_fixes": {"strip": True, "excel_dates": True, "case_fold_to_genes": True}}
    recipe.update(extra or {})
    return recipe


def test_label_repairs():
    symbols = ["SEPT1", "MARCH1", "JUN", "DEC1", "GENE001", "STAT1"]
    labels = np.array(["1-Sep", "1-Mar", "1-Jun", "2-Jun", "1-Dec", "gene001", " STAT1 ", "NT", "Ntc", "non-targeting", "GENE001", "NOT_A_GENE"])
    fixed, report = adapters.fix_labels(labels, symbols, label_recipe())
    assert list(fixed) == ["SEPT1", "MARCH1", "JUN", "JUN", "DEC1", "GENE001", "STAT1", "non-targeting", "non-targeting", "non-targeting", "GENE001", "NOT_A_GENE"], list(fixed)
    assert report["excel_repaired"] == 5 and report["case_repaired"] == 1 and report["stripped"] == 1
    assert report["unresolved_not_a_gene"] == {"NOT_A_GENE": 1}
    # an Excel date two genes could have come from is refused, not guessed; the recipe map resolves it
    try:
        adapters.fix_labels(np.array(["1-Mar"]), ["MARCH1", "MARC1"], label_recipe())
        raise AssertionError("ambiguous Excel date must be refused")
    except adapters.IntakeError as error:
        assert "MARC1" in str(error) and "MARCH1" in str(error)
    fixed, _ = adapters.fix_labels(np.array(["1-Mar"]), ["MARCH1", "MARC1"], label_recipe({"label_fixes": {"map": {"1-Mar": "MARCH1"}}}))
    assert list(fixed) == ["MARCH1"]
    # a control-like name that the recipe does not list is refused (it would become a giant 'perturbation')
    try:
        adapters.fix_labels(np.array(["GENE001", "scramble"]), symbols, label_recipe())
        raise AssertionError("unlisted control-like label must be refused")
    except adapters.IntakeError as error:
        assert "scramble" in str(error)
    try:
        adapters.fix_labels(np.array(["GENE001+STAT1"]), symbols, label_recipe())
        raise AssertionError("multi-target label must be refused")
    except adapters.IntakeError as error:
        assert "multi-target" in str(error)


def test_normalized_negative_and_plan_sample_check():
    assert adapters.raw_counts_problem(np.array([1.0, 2.0, 7.0])) is None
    assert "non-integer" in adapters.raw_counts_problem(np.array([0.5, 1.7, 2.0]))
    assert "negative" in adapters.raw_counts_problem(np.array([-1.0, 2.0]))
    counts, labels, barcodes, batch = make_data()
    folder = scratch("intake_plan_check_")
    var = pd.DataFrame(index=pd.Index(GENES, name="gene"))
    ad.AnnData(X=sp.csr_matrix(counts), obs=base_obs(labels, barcodes, batch), var=var).write_h5ad(folder / "raw.h5ad")
    ad.AnnData(X=sp.csr_matrix(np.log1p(counts)), obs=base_obs(labels, barcodes, batch), var=var).write_h5ad(folder / "norm.h5ad")
    ad.AnnData(X=np.log1p(counts), obs=base_obs(labels, barcodes, batch), var=var).write_h5ad(folder / "norm_dense.h5ad")
    assert adapters.sample_rows_check_h5ad(folder / "raw.h5ad") is None
    assert adapters.sample_rows_check_h5ad(folder / "norm.h5ad") is not None
    assert adapters.sample_rows_check_h5ad(folder / "norm_dense.h5ad") is not None


def test_convert_filters_subsample_and_refuses_existing_output():
    counts, labels, barcodes, batch = make_data()
    folder = scratch("intake_filters_")
    obs = base_obs(labels, barcodes, batch)
    obs["demux"] = np.where(np.arange(N_CELLS) % 5 == 0, "doublet", "singlet")
    path = folder / "raw.h5ad"
    ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=pd.Index(GENES, name="gene"))).write_h5ad(path)
    recipe = recipe_for("filtered", {"format": "h5ad", "path": str(path)}, extra={"subsample": {"per_label_max": 40, "seed": 7}})
    recipe["obs"]["keep"] = "demux == 'singlet'"
    result, out = run_convert(recipe, folder)
    sizes = result.obs["gene"].value_counts()
    assert (result.obs_names.isin(barcodes[obs["demux"].to_numpy() == "singlet"])).all()
    assert all(sizes[label] == 40 for label in PERTS) and sizes["non-targeting"] > 100, sizes.to_dict()
    again, _ = run_convert(recipe, scratch("intake_again_"))  # deterministic given the seed
    dropping = dict(recipe, obs=dict(recipe["obs"], drop_labels=["GENE003"]))
    result_drop, out_drop = run_convert(dropping, scratch("intake_drop_"))
    provenance = json.loads((out_drop / "provenance.json").read_text(encoding="utf-8"))
    assert "GENE003" not in set(result_drop.obs["gene"].astype(str)) and provenance["dropped_by_drop_labels"] == {"GENE003": int(((labels == "GENE003") & (obs["demux"].to_numpy() == "singlet")).sum())}, provenance["dropped_by_drop_labels"]
    assert "GENE003" in (out_drop / "intake_report.md").read_text(encoding="utf-8")
    assert list(again.obs_names) == list(result.obs_names)
    try:
        adapters.convert(folder / "recipe.yaml", out, log=lambda text: None)
        raise AssertionError("an existing output folder must be refused")
    except FileExistsError:
        pass
    assert (out / "provenance.json").exists() and (out / "intake_report.md").exists()


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
