"""Mergeable statistic containers. Every container has add(other), save(path), load(path).

IntHistogramBank   per-gene histogram of integer counts among control cells (Phase 5 percentile)
GeneSums           per-gene detection count, sum, cell count (Phase 6 triage)
F32MeanAccumulator exact float32 column mean of SPORE's scipy path (Phase 7 mean shift, Phase 12 metacells)
Waterfall          ordered reason -> cell count (accounting)
"""
import json
import os
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp

from exact_percentile import percentile_from_histogram


def _atomic_savez(path, **arrays):
    path = str(path)
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    temp_path = path + ".tmp.npz"
    np.savez(temp_path, **arrays)
    os.replace(temp_path, path)


class IntHistogramBank:
    def __init__(self):
        self.table = {}  # gene -> (sorted int64 values, int64 counts)

    def add_histogram(self, gene, values, counts):
        values = np.asarray(values, dtype=np.int64)
        counts = np.asarray(counts, dtype=np.int64)
        if gene in self.table:
            old_values, old_counts = self.table[gene]
            values = np.concatenate([old_values, values])
            counts = np.concatenate([old_counts, counts])
        unique_values, inverse = np.unique(values, return_inverse=True)
        merged = np.zeros(len(unique_values), dtype=np.int64)
        np.add.at(merged, inverse, counts)
        keep = merged > 0
        self.table[gene] = (unique_values[keep], merged[keep])

    def add_column(self, gene, column_values, n_cells=None):
        """column_values: the nonzero entries of one gene (or the whole dense column).
        n_cells: total number of cells; the missing ones count as zeros."""
        column_values = np.asarray(column_values)
        rounded = np.rint(column_values)
        if not np.array_equal(rounded, column_values):
            raise ValueError(f"non-integer counts for gene {gene}")
        values, counts = np.unique(rounded.astype(np.int64), return_counts=True)
        if n_cells is not None:
            n_zero = int(n_cells) - int(counts.sum()) + int(counts[values == 0].sum())
            keep = values != 0
            values, counts = values[keep], counts[keep]
            if n_zero > 0:
                values = np.concatenate([[0], values])
                counts = np.concatenate([[n_zero], counts])
        self.add_histogram(gene, values, counts)

    def add(self, other):
        for gene, (values, counts) in other.table.items():
            self.add_histogram(gene, values, counts)
        return self

    @classmethod
    def merge_files(cls, paths):
        """P10: the same result as loading every file and calling add() one by one (integer sums, so exactly equal), in one vectorized
        group-by over all (gene, value) entries instead of a Python loop with an np.unique per gene and file."""
        genes, values, counts = [], [], []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                genes.append(np.repeat(data["genes"], data["lengths"]))
                values.append(data["values"].astype(np.int64))
                counts.append(data["counts"].astype(np.int64))
        bank = cls()
        if not genes:
            return bank
        genes, values, counts = np.concatenate(genes), np.concatenate(values), np.concatenate(counts)
        if not len(genes):
            return bank
        names, code = np.unique(genes, return_inverse=True)
        order = np.lexsort((values, code))
        code, values, counts = code[order], values[order], counts[order]
        starts = np.flatnonzero(np.r_[True, (code[1:] != code[:-1]) | (values[1:] != values[:-1])])
        merged_counts = np.add.reduceat(counts, starts)
        merged_values, merged_code = values[starts], code[starts]
        bounds = np.flatnonzero(np.r_[True, merged_code[1:] != merged_code[:-1], True])
        for left, right in zip(bounds[:-1], bounds[1:]):
            keep = merged_counts[left:right] > 0
            bank.table[str(names[merged_code[left]])] = (merged_values[left:right][keep], merged_counts[left:right][keep])
        return bank

    def n_cells(self, gene):
        return int(self.table[gene][1].sum()) if gene in self.table else 0

    def percentile(self, gene, quantile, dtype=np.float64):
        values, counts = self.table[gene]
        return percentile_from_histogram(values, counts, quantile, dtype)

    def genes(self):
        return sorted(self.table)

    def save(self, path):
        genes = self.genes()
        lengths = np.array([len(self.table[gene][0]) for gene in genes], dtype=np.int64)
        all_values = np.concatenate([self.table[gene][0] for gene in genes]) if genes else np.zeros(0, np.int64)
        all_counts = np.concatenate([self.table[gene][1] for gene in genes]) if genes else np.zeros(0, np.int64)
        _atomic_savez(path, genes=np.array(genes, dtype=str), lengths=lengths, values=all_values, counts=all_counts)

    @classmethod
    def load(cls, path):
        bank = cls()
        with np.load(path, allow_pickle=False) as data:
            offset = 0
            for gene, length in zip(data["genes"].tolist(), data["lengths"].tolist()):
                bank.table[gene] = (data["values"][offset:offset + length].copy(),
                                    data["counts"][offset:offset + length].copy())
                offset += length
        return bank

    def __eq__(self, other):
        if self.genes() != other.genes():
            return False
        return all(np.array_equal(self.table[gene][0], other.table[gene][0]) and
                   np.array_equal(self.table[gene][1], other.table[gene][1]) for gene in self.table)


class GeneSums:
    """Per-gene detection count (stored entries, like getnnz), sum, and total cells."""

    def __init__(self, n_genes):
        self.n_genes = n_genes
        self.n_cells = 0
        self.nnz = np.zeros(n_genes, dtype=np.int64)
        self.total = np.zeros(n_genes, dtype=np.float64)

    def add_matrix(self, matrix):
        matrix = sp.csr_matrix(matrix)
        self.nnz += matrix.getnnz(axis=0)
        self.total += np.asarray(matrix.astype(np.float64).sum(axis=0)).ravel()
        self.n_cells += matrix.shape[0]
        return self

    def add(self, other):
        if other.n_genes != self.n_genes:
            raise ValueError("gene count mismatch")
        self.nnz += other.nnz
        self.total += other.total
        self.n_cells += other.n_cells
        return self

    def exact_in_float32(self):
        # float32 accumulation of integer counts is exact below 2**24 per gene
        return bool(self.total.max(initial=0) < 2 ** 24)

    def mean_like_spore(self):
        """SPORE phase06:77 in float32 (sum cast to float32, divided by n_cells in float32)."""
        return self.total.astype(np.float32) / self.n_cells

    def save(self, path):
        _atomic_savez(path, n_genes=np.array([self.n_genes]), n_cells=np.array([self.n_cells]),
                      nnz=self.nnz, total=self.total)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            sums = cls(int(data["n_genes"][0]))
            sums.n_cells = int(data["n_cells"][0])
            sums.nnz = data["nnz"].copy()
            sums.total = data["total"].copy()
        return sums

    def __eq__(self, other):
        return (self.n_genes == other.n_genes and self.n_cells == other.n_cells and
                np.array_equal(self.nnz, other.nnz) and np.array_equal(self.total, other.total))


class F32MeanAccumulator:
    """Reproduces scipy's float32 CSR column mean, X.mean(axis=0) == (X * (1.0 / n)).sum(axis=0),
    block by block. Blocks must arrive in the original row order and n_total (the full group size)
    must be known first. The carried partial sum is stacked as the first row of the next block, so
    scipy's own summation order is kept (see tests/test_stats.py for the bit-for-bit check)."""

    def __init__(self, n_genes, n_total):
        self.n_genes = n_genes
        self.n_total = int(n_total)
        self.n_seen = 0
        self.carry = np.zeros(n_genes, dtype=np.float32)

    def add_block(self, block):
        block = sp.csr_matrix(block, dtype=np.float32)
        scaled = block * (1.0 / self.n_total)
        carry_row = sp.csr_matrix(self.carry.reshape(1, -1))
        stacked = sp.vstack([carry_row, scaled], format="csr")
        self.carry = np.asarray(stacked.sum(axis=0)).ravel().astype(np.float32, copy=False)
        self.n_seen += block.shape[0]
        return self

    def result(self):
        if self.n_seen != self.n_total:
            raise ValueError(f"saw {self.n_seen} rows, expected {self.n_total}")
        return self.carry.copy()

    def save(self, path):
        _atomic_savez(path, n_total=np.array([self.n_total]), n_seen=np.array([self.n_seen]), carry=self.carry)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            carry = data["carry"].copy()
            accumulator = cls(len(carry), int(data["n_total"][0]))
            accumulator.n_seen = int(data["n_seen"][0])
            accumulator.carry = carry
        return accumulator


class Waterfall:
    def __init__(self):
        self.steps = OrderedDict()

    def count(self, reason, number):
        self.steps[reason] = self.steps.get(reason, 0) + int(number)
        return self

    def add(self, other):
        for reason, number in other.steps.items():
            self.count(reason, number)
        return self

    def save(self, path):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.steps, handle, indent=2)

    @classmethod
    def load(cls, path):
        waterfall = cls()
        with open(path, encoding="utf-8") as handle:
            waterfall.steps = OrderedDict(json.load(handle))
        return waterfall

    def __eq__(self, other):
        return dict(self.steps) == dict(other.steps)
