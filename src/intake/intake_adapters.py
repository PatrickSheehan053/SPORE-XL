"""P7: intake adapters. Any supported raw format + one recipe YAML per dataset -> a SPORE-ready sparse CSR .h5ad (raw counts, one row per kept
cell, `gene` = perturbation label, control label normalized) that `xl.source.kind: h5ad_rows` reads. Streams in windows of rows, so inputs
larger than RAM are fine for h5ad, 10x .h5, loom and dense HDF5; .mtx and CSV/TSV are read whole (small datasets).

  python -B spore_xl.py convert --recipe src/intake/recipes/<dataset>.yaml --out <new folder>

Formats (recipe `source.format`): h5ad | 10x_h5 | 10x_mtx | loom | csv | tsv | dense_h5_tables (a dense HDF5 matrix + cell and gene tables exported from R,
first case: DLD-1 GSE337988, see src/intake/R/). Refuses: normalized or negative input (before writing anything), a gene axis that does not
match the gene table, labels it cannot repair (Excel dates with several candidates, control-like names that are not listed, multi-target
labels). Never overwrites: the output folder must be new.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
CONTROL_LIKE = {"nt", "ntc", "non-targeting", "nontargeting", "non targeting", "control", "ctrl", "negative control", "neg", "scramble",
                "scrambled", "non_targeting", "safe", "nt_control", "negctrl", "neg_ctrl"}
SOURCE_FORMATS = ("h5ad", "10x_h5", "10x_mtx", "loom", "csv", "tsv", "dense_h5_tables")


class IntakeError(ValueError):
    pass


class NotRawCountsError(IntakeError):
    pass


# ------------------------------------------------------------------------------------------------ raw-count check (also used by `plan`)
def raw_counts_problem(values, label="matrix"):
    """None when `values` (the stored entries of a sample of rows) look like raw UMI counts, else a message."""
    values = np.asarray(values)
    if values.size == 0:
        return None
    if np.any(values < 0):
        return f"{label} has negative values: it is scaled/centered data, not raw counts"
    fractional = np.abs(values - np.rint(values)) > 1e-6
    if fractional.any():
        share = fractional.mean()
        return (f"{label} has non-integer values ({share:.1%} of the stored entries, e.g. {values[fractional][:3].tolist()}): it looks normalized "
                f"or log-transformed. SPORE needs RAW integer UMI counts")
    return None


def assert_raw_counts(values, label="matrix"):
    problem = raw_counts_problem(values, label)
    if problem:
        raise NotRawCountsError(problem + ". Use the raw count layer (recipe source.x_key, for example layers/counts or raw/X) or export the raw counts.")


def sample_rows_check_h5ad(h5ad_path, n_rows=2000, seed=0):
    """Called by plan before it writes anything: raw-count check on a sample of rows of an .h5ad (sparse or dense X)."""
    with h5py.File(h5ad_path, "r") as handle:
        node = handle["X"]
        rng = np.random.default_rng(seed)
        if isinstance(node, h5py.Dataset):
            n_obs = node.shape[0]
            rows = np.sort(rng.choice(n_obs, size=min(n_rows, n_obs), replace=False))
            sample = np.vstack([node[start:start + 1] for start in rows[:: max(1, len(rows) // 400)]])  # ~400 single-row reads
            return raw_counts_problem(sample[sample != 0], "X")
        encoding = node.attrs.get("encoding-type", "csr_matrix")
        indptr = node["indptr"][...]
        n_obs = len(indptr) - 1
        rows = np.sort(rng.choice(n_obs, size=min(n_rows, n_obs), replace=False))
        picked = []
        step = max(1, len(rows) // 400)
        if encoding == "csr_matrix":
            for row in rows[::step]:
                picked.append(node["data"][indptr[row]:indptr[row + 1]])
        else:  # csc: sample the first stored values of random columns instead
            picked.append(node["data"][: min(200000, node["data"].shape[0])])
        return raw_counts_problem(np.concatenate(picked) if picked else np.zeros(0), "X")


# ------------------------------------------------------------------------------------------------ readers
def _text_open(path):
    path = str(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "rt", encoding="utf-8")


def _read_table(path, index_col=None):
    return pd.read_csv(path, sep="\t", compression="infer", index_col=index_col, low_memory=False, dtype={"barcode": str})


class Reader:
    """obs / var tables plus read_block(start, stop) -> CSR (cells x genes) of the genes in self.var."""
    n_obs = 0
    n_vars = 0
    guides = None  # (CSR cells x guides, guide names) for formats that carry a guide-count matrix

    def read_block(self, start, stop):
        raise NotImplementedError

    def close(self):
        pass


def _csr_from_indptr(indptr, data_node, indices_node, start, stop, n_cols):
    first, last = int(indptr[start]), int(indptr[stop])
    pointer = (indptr[start:stop + 1] - first).astype(np.int64)
    return sp.csr_matrix((data_node[first:last], indices_node[first:last], pointer), shape=(stop - start, n_cols))


class H5adReader(Reader):
    def __init__(self, path, x_key="X"):
        import anndata.io as anndata_io
        self.handle = h5py.File(path, "r")
        base = "raw" if x_key.startswith("raw") else ""
        self.node = self.handle[x_key if x_key != "raw.X" else "raw/X"]
        prefix = "raw/" if x_key in ("raw.X", "raw/X") else ""
        self.obs = anndata_io.read_elem(self.handle["obs"])
        self.var = anndata_io.read_elem(self.handle[prefix + "var"])
        if isinstance(self.node, h5py.Dataset):
            self.kind, self.n_obs, self.n_vars = "dense", self.node.shape[0], self.node.shape[1]
        else:
            encoding = self.node.attrs.get("encoding-type", "csr_matrix")
            if encoding != "csr_matrix":
                raise IntakeError(f"{path}: X is stored as {encoding}; only csr_matrix and dense can be streamed. Convert it once with anndata "
                                  f"(adata.X = adata.X.tocsr()) or use a csr copy.")
            self.kind = "csr"
            self.indptr = self.node["indptr"][...]
            self.n_obs, self.n_vars = len(self.indptr) - 1, int(self.node.attrs["shape"][1])
        if len(self.obs) != self.n_obs or len(self.var) != self.n_vars:
            raise IntakeError(f"{path}: matrix {self.n_obs} x {self.n_vars} does not match obs ({len(self.obs)}) / var ({len(self.var)})")

    def read_block(self, start, stop):
        if self.kind == "dense":
            return sp.csr_matrix(self.node[start:stop])
        return _csr_from_indptr(self.indptr, self.node["data"], self.node["indices"], start, stop, self.n_vars)

    def close(self):
        self.handle.close()


class Tenx10xH5Reader(Reader):
    """10x CellRanger v3+ .h5: matrix/{data,indices,indptr,shape,barcodes,features/*}; CSC over barcodes, so a cell is a stored column."""

    def __init__(self, path, feature_type="Gene Expression", guide_types=("crispr", "guide", "protospacer")):
        self.handle = h5py.File(path, "r")
        if "matrix" not in self.handle:
            raise IntakeError(f"{path}: no 'matrix' group; only the CellRanger v3+ .h5 layout is supported (v2 files: convert with CellRanger or scanpy)")
        group = self.handle["matrix"]
        decode = lambda values: [v.decode() if isinstance(v, bytes) else str(v) for v in values]  # noqa: E731
        features = pd.DataFrame({"gene_id": decode(group["features"]["id"][...]), "gene_name": decode(group["features"]["name"][...]),
                                 "feature_type": decode(group["features"]["feature_type"][...])})
        shape = [int(v) for v in group["shape"][...]]
        self.n_features, self.n_obs = shape
        self.indptr = group["indptr"][...]
        self.data, self.indices = group["data"], group["indices"]
        self.gene_cols = np.flatnonzero((features["feature_type"] == feature_type).to_numpy())
        if not len(self.gene_cols):
            raise IntakeError(f"{path}: no features of type '{feature_type}' (found {sorted(set(features['feature_type']))})")
        guide_mask = features["feature_type"].str.lower().apply(lambda text: any(token in text for token in guide_types)).to_numpy()
        self.guide_cols = np.flatnonzero(guide_mask)
        self.guide_names = list(features["gene_name"][guide_mask]) if guide_mask.any() else []
        self.var = features.iloc[self.gene_cols].set_index("gene_id")
        self.n_vars = len(self.gene_cols)
        self.obs = pd.DataFrame(index=pd.Index(decode(group["barcodes"][...]), name="barcode"))

    def _block_all(self, start, stop):
        return _csr_from_indptr(self.indptr, self.data, self.indices, start, stop, self.n_features)

    def read_block(self, start, stop):
        return self._block_all(start, stop)[:, self.gene_cols]

    def load_guides(self, window=50000):
        if not len(self.guide_cols):
            return None
        parts = [self._block_all(start, min(start + window, self.n_obs))[:, self.guide_cols] for start in range(0, self.n_obs, window)]
        return sp.vstack(parts, format="csr"), self.guide_names

    def close(self):
        self.handle.close()


class Tenx10xMtxReader(Reader):
    """CellRanger matrix folder: matrix.mtx(.gz) (features x barcodes), barcodes.tsv(.gz), features.tsv(.gz) or genes.tsv(.gz). Read whole."""

    def __init__(self, folder, orientation="genes_by_cells"):
        from scipy.io import mmread
        folder = Path(folder)

        def find(*names):
            for name in names:
                for suffix in ("", ".gz"):
                    if (folder / (name + suffix)).exists():
                        return folder / (name + suffix)
            raise IntakeError(f"{folder}: none of {names} found")

        matrix = sp.csr_matrix(mmread(find("matrix.mtx")))
        barcodes = pd.read_csv(find("barcodes.tsv"), sep="\t", header=None)[0].astype(str)
        features = pd.read_csv(find("features.tsv", "genes.tsv"), sep="\t", header=None)
        if orientation == "genes_by_cells":
            matrix = matrix.T.tocsr()
        if matrix.shape[0] != len(barcodes) or matrix.shape[1] != len(features):
            raise IntakeError(f"{folder}: matrix {matrix.shape} does not match {len(barcodes)} barcodes x {len(features)} features; "
                              f"check source.orientation (genes_by_cells is the 10x convention)")
        columns = ["gene_id", "gene_name", "feature_type"][: features.shape[1]]
        features.columns = columns
        gene_rows = np.flatnonzero((features["feature_type"] == "Gene Expression").to_numpy()) if "feature_type" in features else np.arange(len(features))
        self.matrix = matrix[:, gene_rows].tocsr()
        self.var = features.iloc[gene_rows].set_index("gene_id")
        self.obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
        self.n_obs, self.n_vars = self.matrix.shape

    def read_block(self, start, stop):
        return self.matrix[start:stop]


class LoomReader(Reader):
    def __init__(self, path):
        self.handle = h5py.File(path, "r")
        self.dataset = self.handle["matrix"]  # loom: genes x cells
        n_genes, n_cells = self.dataset.shape

        def attributes(group, length):
            frame = {}
            for key in self.handle[group]:
                values = self.handle[group][key][...]
                if values.ndim == 1 and len(values) == length:
                    frame[key] = [v.decode() if isinstance(v, bytes) else v for v in values]
            return pd.DataFrame(frame)

        cols, rows = attributes("col_attrs", n_cells), attributes("row_attrs", n_genes)
        cell_ids = cols["CellID"] if "CellID" in cols else pd.Series([f"cell_{i}" for i in range(n_cells)])
        gene_ids = rows["Accession"] if "Accession" in rows else rows["Gene"] if "Gene" in rows else pd.Series([f"gene_{i}" for i in range(n_genes)])
        self.obs = cols.set_index(cell_ids.astype(str))
        self.var = rows.set_index(gene_ids.astype(str))
        if "Gene" in rows.columns and "gene_name" not in self.var.columns:
            self.var["gene_name"] = rows["Gene"].astype(str).to_numpy()
        self.n_obs, self.n_vars = n_cells, n_genes

    def read_block(self, start, stop):
        return sp.csr_matrix(self.dataset[:, start:stop].T)

    def close(self):
        self.handle.close()


class DelimitedReader(Reader):
    """CSV / TSV of counts, read whole (refuses files over 2 GB). orientation: genes_by_cells (rows are genes) or cells_by_genes."""

    def __init__(self, path, separator, orientation, max_bytes=2 * 10 ** 9):
        if Path(path).stat().st_size > max_bytes:
            raise IntakeError(f"{path} is larger than {max_bytes / 1e9:.0f} GB: use an h5ad, 10x .h5 or loom file for large data")
        frame = pd.read_csv(path, sep=separator, index_col=0, compression="infer")
        if orientation == "genes_by_cells":
            frame = frame.T
        elif orientation != "cells_by_genes":
            raise IntakeError("csv/tsv needs source.orientation: genes_by_cells or cells_by_genes (a table cannot be guessed safely)")
        self.matrix = sp.csr_matrix(frame.to_numpy())
        self.obs = pd.DataFrame(index=pd.Index(frame.index.astype(str), name="barcode"))
        self.var = pd.DataFrame(index=pd.Index(frame.columns.astype(str), name="gene_id"))
        self.var["gene_name"] = self.var.index
        self.n_obs, self.n_vars = self.matrix.shape

    def read_block(self, start, stop):
        return self.matrix[start:stop]


class DenseH5TablesReader(Reader):
    """A dense HDF5 matrix plus cell and gene tables (TSV) exported from R. The orientation is checked against the table lengths; a gene
    axis that matches neither orientation is the DLD-1 situation (rds row names that belong to another slot) and is refused with numbers."""

    def __init__(self, path, dataset, cells_table, genes_table, orientation="auto", cell_index_column=None, gene_id_column=None):
        self.handle = h5py.File(path, "r")
        if dataset not in self.handle:
            raise IntakeError(f"{path}: no dataset {dataset}; found {list(self.handle)}")
        self.dataset = self.handle[dataset]
        cells = _read_table(cells_table)
        genes = _read_table(genes_table)
        n_first, n_second = self.dataset.shape
        n_cells, n_genes = len(cells), len(genes)
        if orientation == "auto":
            if (n_first, n_second) == (n_cells, n_genes):
                orientation = "cells_by_genes"
            elif (n_first, n_second) == (n_genes, n_cells):
                orientation = "genes_by_cells"
        mismatch = (orientation == "auto" or (orientation == "cells_by_genes" and (n_first, n_second) != (n_cells, n_genes))
                    or (orientation == "genes_by_cells" and (n_first, n_second) != (n_genes, n_cells)))
        if mismatch:
            raise IntakeError(
                f"{path}:{dataset} has shape {self.dataset.shape} but the cell table has {n_cells:,} rows and the gene table {n_genes:,} rows, so the gene axis "
                f"does not match. This is the DLD-1 SummarizedExperiment problem: the exported names came from another slot (for example the guide library) "
                f"instead of the rows of the experiment. Re-export the row names of the main experiment with src/intake/R/export_se_tables_v2.R and use that gene table.")
        self.orientation = orientation
        index_column = cell_index_column or ("barcode" if "barcode" in cells.columns else cells.columns[0])
        self.obs = cells.set_index(index_column)
        self.obs.index = self.obs.index.astype(str)
        id_column = gene_id_column or ("gene_id" if "gene_id" in genes.columns else genes.columns[-1])
        self.var = genes.set_index(id_column)
        self.var.index = self.var.index.astype(str)
        self.n_obs, self.n_vars = n_cells, n_genes

    def read_block(self, start, stop):
        if self.orientation == "cells_by_genes":
            return sp.csr_matrix(self.dataset[start:stop])
        return sp.csr_matrix(self.dataset[:, start:stop].T)

    def close(self):
        self.handle.close()


def open_reader(recipe):
    source = recipe["source"]
    fmt, path = source["format"], source["path"]
    if fmt not in SOURCE_FORMATS:
        raise IntakeError(f"source.format must be one of {SOURCE_FORMATS}, got {fmt}")
    if fmt == "h5ad":
        return H5adReader(path, source.get("x_key", "X"))
    if fmt == "10x_h5":
        return Tenx10xH5Reader(path)
    if fmt == "10x_mtx":
        return Tenx10xMtxReader(path, source.get("orientation", "genes_by_cells"))
    if fmt == "loom":
        return LoomReader(path)
    if fmt in ("csv", "tsv"):
        return DelimitedReader(path, "," if fmt == "csv" else "\t", source.get("orientation", ""))
    return DenseH5TablesReader(path, source.get("h5_dataset", "/assay001"), source["cells_table"], source["genes_table"],
                               source.get("orientation", "auto"), source.get("cell_index_column"), source.get("gene_id_column"))


# ------------------------------------------------------------------------------------------------ labels
def excel_damage_forms(symbol):
    """The forms Excel turns a gene symbol into (SEPT1 -> 1-Sep, MARCH1 -> 1-Mar, DEC1 -> 1-Dec, OCT4 -> 4-Oct); [] when it would not be changed."""
    match = re.fullmatch(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*?(\d{1,2})", str(symbol).upper())
    if not match:
        return []
    month = match.group(1).capitalize()
    return [f"{int(match.group(2))}-{month}", f"{month}-{int(match.group(2))}"]


def looks_like_excel_date(label):
    match = re.fullmatch(r"(?:(\d{1,2})-([A-Za-z]{3})|([A-Za-z]{3})-(\d{1,2}))", label)
    if not match:
        return False
    month = (match.group(2) or match.group(3)).capitalize()
    return month in MONTHS


def fix_labels(labels, symbols, recipe):
    """labels: array of str. symbols: gene symbols of the matrix. Returns (new labels, report). Raises IntakeError for what cannot be repaired."""
    rules = recipe.get("label_fixes", {}) or {}
    control_in = [str(v) for v in recipe.get("control_labels", [])]
    control_out = recipe.get("control_label_out", "non-targeting")
    explicit = {str(k): str(v) for k, v in (rules.get("map") or {}).items()}
    symbol_set = set(map(str, symbols))
    upper_to_symbol = {}
    for symbol in symbol_set:
        upper_to_symbol.setdefault(symbol.upper(), set()).add(symbol)
    damaged = {}
    for symbol in symbol_set:
        for form in excel_damage_forms(symbol):
            damaged.setdefault(form.upper(), set()).add(symbol)
    for symbol in symbol_set:  # "1-Jun" can also be the guide name JUN-1 of the gene JUN
        if symbol.upper() in {m.upper() for m in MONTHS}:
            for number in range(1, 100):
                damaged.setdefault(f"{number}-{symbol.capitalize()}".upper(), set()).add(symbol)
    report = {"stripped": 0, "mapped": 0, "controls_normalized": 0, "excel_repaired": 0, "case_repaired": 0, "unresolved_not_a_gene": {},
              "ambiguous": {}, "control_like_unlisted": {}, "repairs": {}}
    control_lower = {value.strip().lower() for value in control_in}
    values = pd.Series(labels.astype(str))
    unique = pd.unique(values)
    translation = {}
    for original in unique:
        label = original
        if rules.get("strip", True):
            label = label.strip()
        if label.lower() in control_lower:
            translation[original] = control_out
            continue
        if label in explicit:
            translation[original] = explicit[label]
            continue
        if label in symbol_set or label == control_out:
            translation[original] = label
            continue
        if "+" in label:
            raise IntakeError(f"multi-target label '{original}': combinatorial perturbations are not supported (remove those cells with obs.keep)")
        candidates = set()
        reason = ""
        if rules.get("excel_dates", True) and looks_like_excel_date(label):
            candidates = damaged.get(label.upper(), set())
            reason = "excel_repaired"
        if not candidates and rules.get("case_fold_to_genes", True) and label.upper() in upper_to_symbol:
            candidates = upper_to_symbol[label.upper()]
            reason = "case_repaired"
        if len(candidates) == 1:
            translation[original] = next(iter(candidates))
            report[reason] += 1
            report["repairs"][original] = translation[original]
        elif len(candidates) > 1:
            report["ambiguous"][original] = sorted(candidates)
            translation[original] = label
        else:
            if label.lower() in CONTROL_LIKE and label.lower() not in control_lower:
                report["control_like_unlisted"][original] = int((values == original).sum())
            elif looks_like_excel_date(label):
                report["ambiguous"][original] = []  # a date-like label with no gene it could have come from
            else:
                report["unresolved_not_a_gene"][original] = int((values == original).sum())
            translation[original] = label
    if report["control_like_unlisted"]:
        raise IntakeError(f"labels that look like controls but are not in control_labels: {report['control_like_unlisted']}. Add them to the recipe's control_labels "
                          f"(or to label_fixes.map if they are real targets); they would otherwise become a huge 'perturbation'.")
    if report["ambiguous"]:
        raise IntakeError(f"labels that look like Excel-damaged gene symbols and cannot be repaired uniquely: {report['ambiguous']}. "
                          f"Resolve them in label_fixes.map (damaged label -> gene symbol).")
    new = values.map(translation)
    report["stripped"] = int((values != values.str.strip()).sum())
    report["controls_normalized"] = int(((new == control_out) & (values != control_out)).sum())
    report["mapped"] = int(sum(1 for key, value in translation.items() if key in explicit.values() or key in explicit))
    report["n_unresolved_labels"] = len(report["unresolved_not_a_gene"])
    report["n_unresolved_cells"] = int(sum(report["unresolved_not_a_gene"].values()))
    return new.to_numpy(), report


def assign_from_guides(guide_matrix, guide_names, min_umi, target_regex=r"-\d+$", control_regex=r"^(non-?targeting|NT|control)", control_out="non-targeting"):
    """Guide-count matrix -> one label per cell: the single target with a guide at >= min_umi UMI (two guides of one gene count once); controls by
    regex; cells with no or several targets get '' (dropped). Returns (labels, counts dict)."""
    names = np.asarray(guide_names, dtype=str)
    is_control = np.array([re.search(control_regex, name, re.I) is not None for name in names])
    targets = np.array([control_out if control else re.sub(target_regex, "", name) for name, control in zip(names, is_control)])
    unique_targets, inverse = np.unique(targets, return_inverse=True)
    present = sp.csr_matrix(guide_matrix >= min_umi)
    membership = sp.csr_matrix((np.ones(len(names)), (np.arange(len(names)), inverse)), shape=(len(names), len(unique_targets)))
    target_hits = (present @ membership).tocsr()
    target_hits.data = (target_hits.data > 0).astype(np.int64)
    n_targets = np.asarray(target_hits.sum(axis=1)).ravel()
    best = np.asarray(target_hits.argmax(axis=1)).ravel()
    labels = np.where(n_targets == 1, unique_targets[best], "")
    return labels, {"no_guide": int((n_targets == 0).sum()), "multi_target": int((n_targets > 1).sum()), "assigned": int((n_targets == 1).sum())}


# ------------------------------------------------------------------------------------------------ streaming CSR h5ad writer
class CsrH5adWriter:
    def __init__(self, path, n_vars, dtype=np.float32):
        self.path, self.n_vars, self.dtype = Path(path), n_vars, np.dtype(dtype)
        self.handle = h5py.File(self.path, "w")
        self.handle.attrs["encoding-type"] = "anndata"
        self.handle.attrs["encoding-version"] = "0.1.0"
        self.group = self.handle.create_group("X")
        self.group.attrs["encoding-type"] = "csr_matrix"
        self.group.attrs["encoding-version"] = "0.1.0"
        self.data = self.group.create_dataset("data", shape=(0,), maxshape=(None,), dtype=self.dtype, chunks=(1 << 20,), compression="lzf")
        self.indices = self.group.create_dataset("indices", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1 << 20,), compression="lzf")
        self.indptr_parts, self.offset, self.n_obs = [np.zeros(1, dtype=np.int64)], 0, 0

    def append(self, block):
        block = sp.csr_matrix(block, dtype=self.dtype)
        block.sort_indices()
        size = len(block.data)
        self.data.resize(self.offset + size, axis=0)
        self.indices.resize(self.offset + size, axis=0)
        self.data[self.offset:self.offset + size] = block.data
        self.indices[self.offset:self.offset + size] = block.indices.astype(np.int32)
        self.indptr_parts.append(block.indptr[1:].astype(np.int64) + self.offset)
        self.offset += size
        self.n_obs += block.shape[0]

    def finish(self, obs, var, uns=None):
        import anndata.io as anndata_io
        self.group.create_dataset("indptr", data=np.concatenate(self.indptr_parts), dtype=np.int64)
        self.group.attrs["shape"] = np.array([self.n_obs, self.n_vars], dtype=np.int64)
        anndata_io.write_elem(self.handle, "obs", obs)
        anndata_io.write_elem(self.handle, "var", var)
        anndata_io.write_elem(self.handle, "uns", uns or {})
        self.handle.close()


# ------------------------------------------------------------------------------------------------ the conversion
def load_recipe(path):
    recipe = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("dataset", "source", "obs"):
        if key not in recipe:
            raise IntakeError(f"recipe {path}: missing '{key}'")
    return recipe


def _symbols(var, recipe):
    column = (recipe.get("genes") or {}).get("symbol_column")
    if column:
        if column not in var.columns:
            raise IntakeError(f"genes.symbol_column '{column}' is not in the gene table (columns: {list(var.columns)})")
        return var[column].astype(str).to_numpy()
    for candidate in ("gene_name", "gene_symbol", "gene_symbols", "feature_name", "symbol", "hgnc_symbol"):
        if candidate in var.columns:
            return var[candidate].astype(str).to_numpy()
    return var.index.astype(str).to_numpy()


def convert(recipe_path, out_dir, window=20000, log=print):
    recipe = load_recipe(recipe_path)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already exists and is not empty; choose a new folder (nothing is overwritten)")
    reader = open_reader(recipe)
    try:
        return _convert(recipe, recipe_path, out_dir, reader, window, log)
    finally:
        reader.close()


def _convert(recipe, recipe_path, out_dir, reader, window, log):
    started = time.time()
    obs, var = reader.obs.copy(), reader.var.copy()
    spec = recipe["obs"]
    steps = [("cells in the source", int(reader.n_obs))]
    keep = np.ones(reader.n_obs, dtype=bool)
    if spec.get("keep"):
        keep &= obs.eval(spec["keep"]).to_numpy(dtype=bool)
        steps.append((f"after obs.keep ({spec['keep']})", int(keep.sum())))

    symbols = _symbols(var, recipe)
    guide_report = None
    if spec.get("label_from_guides"):
        loaded = reader.load_guides() if hasattr(reader, "load_guides") else None
        if loaded is None:
            raise IntakeError("obs.label_from_guides needs a guide-count matrix; this source has none")
        options = spec["label_from_guides"]
        labels, guide_report = assign_from_guides(loaded[0], loaded[1], options.get("min_umi", 3), options.get("target_regex", r"-\d+$"),
                                                  options.get("control_regex", r"^(non-?targeting|NT|control)"), recipe.get("control_label_out", "non-targeting"))
        keep &= labels != ""
        steps.append(("after guide assignment (single target or control)", int(keep.sum())))
    else:
        column = spec["label_column"]
        if column not in obs.columns:
            raise IntakeError(f"obs.label_column '{column}' is not in the cell table (columns: {list(obs.columns)[:20]})")
        labels = obs[column].astype(str).to_numpy()
    drop = {str(v) for v in spec.get("drop_labels", ["nan", "", "None"])}
    dropped_label_counts = {label: int((keep & (labels == label)).sum()) for label in sorted(drop) if (keep & (labels == label)).any()}
    keep &= ~np.isin(labels, list(drop))
    steps.append(("after dropping empty labels", int(keep.sum())))
    for label, count in dropped_label_counts.items():
        steps.append((f"  dropped by obs.drop_labels: '{label}'", count))
        log(f"[convert] drop_labels: {count:,} cells with label '{label}' dropped (not merged into the controls)")

    labels_fixed, label_report = fix_labels(labels[keep], symbols, recipe)
    control_out = recipe.get("control_label_out", "non-targeting")
    final_labels = np.full(reader.n_obs, "", dtype=object)
    final_labels[np.flatnonzero(keep)] = labels_fixed
    if not (final_labels == control_out).any():
        raise IntakeError(f"no control cells after label fixes (control_labels {recipe.get('control_labels')}); SPORE needs controls")

    selected = np.flatnonzero(keep)
    subsample = recipe.get("subsample") or {}
    if subsample.get("per_label_max"):
        rng = np.random.default_rng(int(subsample.get("seed", 42)))
        chosen = []
        series = pd.Series(np.arange(len(selected)), index=final_labels[selected])
        for label, positions in series.groupby(level=0, sort=True):
            rows = np.sort(positions.to_numpy())
            cap = None if label == control_out else int(subsample["per_label_max"])
            chosen.append(rows if cap is None or len(rows) <= cap else np.sort(rng.choice(rows, size=cap, replace=False)))
        selected = selected[np.sort(np.concatenate(chosen))]
        steps.append((f"after subsample (max {subsample['per_label_max']} cells per perturbation, controls kept)", int(len(selected))))
    if not len(selected):
        raise IntakeError("no cells left after the recipe's filters")

    # raw-count check on a sample BEFORE anything is written
    rng = np.random.default_rng(0)
    probe = np.sort(rng.choice(selected, size=min(2000, len(selected)), replace=False))
    sample_values = []
    for first in range(0, len(probe), 500):
        rows = probe[first:first + 500]
        block = reader.read_block(int(rows[0]), int(rows[-1]) + 1)
        sample_values.append(block[rows - rows[0]].data)
    assert_raw_counts(np.concatenate(sample_values), "the count matrix")

    duplicate_note = None
    duplicated = pd.Series(symbols).duplicated(keep=False)
    if duplicated.any():
        duplicate_note = sorted(set(pd.Series(symbols)[duplicated]))
        policy = (recipe.get("genes") or {}).get("duplicate_symbols", "report")
        if policy == "error":
            raise IntakeError(f"{len(duplicate_note)} gene symbols map to more than one feature (e.g. {duplicate_note[:5]}); set genes.duplicate_symbols to report")
    n_dupes = len(duplicate_note) if duplicate_note else 0

    out_dir.mkdir(parents=True)
    name = recipe["dataset"]
    partial = out_dir / f"{name}.h5ad.partial"
    final_path = out_dir / f"{name}.h5ad"
    writer = CsrH5adWriter(partial, reader.n_vars)
    selected_set = selected
    position = 0
    for first in range(0, reader.n_obs, window):
        stop = min(first + window, reader.n_obs)
        left, right = np.searchsorted(selected_set, first), np.searchsorted(selected_set, stop)
        if right == left:
            continue
        block = reader.read_block(first, stop)
        rows = selected_set[left:right] - first
        piece = block[rows]
        assert_raw_counts(piece.data, f"rows {first}-{stop}")
        writer.append(piece)
    perturbation_col = recipe.get("perturbation_col", "gene")
    out_labels = final_labels[selected].astype(str)
    out_obs = pd.DataFrame({perturbation_col: pd.Categorical(out_labels, categories=sorted(set(out_labels)))}, index=obs.index[selected].astype(str))
    for target, source in (("batch", spec.get("batch_column")), ("donor", spec.get("donor_column"))):
        if source:
            if source not in obs.columns:
                raise IntakeError(f"obs.{target}_column '{source}' is not in the cell table (columns: {list(obs.columns)[:20]})")
            out_obs[target] = pd.Categorical(obs[source].astype(str).to_numpy()[selected])
    for extra in spec.get("extra_columns", []) or []:
        out_obs[extra] = obs[extra].to_numpy()[selected]
    out_var = var.copy()
    out_var.index = out_var.index.astype(str)
    if "gene_name" not in out_var.columns:
        out_var["gene_name"] = symbols
    uns = {"intake": {"recipe": str(recipe_path), "dataset": name, "source_format": recipe["source"]["format"], "created": time.strftime("%Y-%m-%d %H:%M:%S")}}
    writer.finish(out_obs, out_var, uns)
    if final_path.exists():
        raise FileExistsError(final_path)
    os.replace(partial, final_path)  # the partial file is our own new file: renamed when complete, never an overwrite
    steps.append(("cells written", int(len(selected))))

    provenance = {"recipe": recipe, "recipe_file": str(recipe_path), "steps": steps, "label_report": label_report, "guide_report": guide_report, "dropped_by_drop_labels": dropped_label_counts,
                  "duplicate_gene_symbols": n_dupes, "duplicate_symbol_examples": (duplicate_note or [])[:20], "n_genes": int(reader.n_vars),
                  "seconds": round(time.time() - started, 1), "output": str(final_path),
                  "source_size_bytes": Path(recipe["source"]["path"]).stat().st_size if Path(recipe["source"]["path"]).is_file() else None}
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    lines = [f"# Intake report: {name}", "", f"source: {recipe['source']['format']} {recipe['source']['path']}", "", "| step | cells |", "|---|---|"]
    lines += [f"| {text} | {count:,} |" for text, count in steps]
    lines += ["", "## Labels", "", f"- controls normalized to `{control_out}`: {label_report['controls_normalized']:,} cells",
              f"- whitespace stripped: {label_report['stripped']:,}; case repaired: {label_report['case_repaired']}; Excel dates repaired: {label_report['excel_repaired']}",
              f"- repairs: {label_report['repairs'] or 'none'}",
              f"- labels that are not a gene in the matrix (kept, their target is not measured): {label_report['n_unresolved_labels']} labels, {label_report['n_unresolved_cells']:,} cells"]
    if guide_report:
        lines += ["", f"## Guide assignment", "", f"- {guide_report}"]
    if n_dupes:
        lines += ["", "## Duplicate gene symbols", "", f"- {n_dupes} symbols map to more than one feature (examples {(duplicate_note or [])[:8]}). The core pipeline suffixes the later "
                  f"copies (_1) and a perturbation label resolves to the LAST copy; check that these genes are not perturbation targets."]
    (out_dir / "intake_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[convert] {name}: {len(selected):,} cells x {reader.n_vars:,} genes -> {final_path}")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(prog="spore_xl.py convert")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--out", required=True, help="a NEW folder")
    parser.add_argument("--window", type=int, default=20000, help="rows read at a time")
    args = parser.parse_args(argv)
    try:
        convert(args.recipe, args.out, args.window)
    except IntakeError as error:
        print(f"[convert] REFUSED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
