"""T1 statistics. Run: python -B tests\\test_stats.py   (also collectable by pytest)."""
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "engine"))

import numpy as np
import scipy.sparse as sp

from exact_percentile import percentile_from_histogram
from stats import F32MeanAccumulator, GeneSums, IntHistogramBank, Waterfall

QUANTILES = (1, 5, 10, 50, 90, 95, 99)


def _scratch_dir():
    base = os.environ.get("XL_TEST_TMP", tempfile.gettempdir())
    return Path(tempfile.mkdtemp(prefix="xl_test_", dir=base))  # not cleaned up: removal is Patrick's call


def _random_vector(rng):
    length = int(rng.choice([1, 2, 3, 7, 50, 333, 1000, 4000, 20000]))
    mode = int(rng.integers(0, 3))
    if mode == 0:
        vector = rng.negative_binomial(2, 0.4, size=length)
    elif mode == 1:
        vector = np.where(rng.random(length) < 0.9, 0, rng.integers(1, 60, size=length))
    else:
        vector = rng.poisson(rng.uniform(0.05, 30), size=length)
    return vector.astype(np.int64)


def test_percentile_bit_identical_1000_vectors():
    rng = np.random.default_rng(2026)
    checked = 0
    for dtype in (np.float64, np.float32, np.int64):
        for _ in range(1000 if dtype is np.float64 else 300):
            vector = _random_vector(rng)
            values, counts = np.unique(vector, return_counts=True)
            typed = vector.astype(dtype)
            for quantile in QUANTILES:
                expected = np.percentile(typed, quantile)
                actual = percentile_from_histogram(values, counts, quantile, dtype)
                assert np.asarray(expected).dtype == np.asarray(actual).dtype, (dtype, quantile)
                assert np.asarray(expected).tobytes() == np.asarray(actual).tobytes(), \
                    (dtype, quantile, expected, actual, len(vector))
                checked += 1
    print(f"  percentile: {checked} comparisons bit-identical (float64, float32, int64)")


def test_percentile_through_bank_with_shards_and_zeros():
    rng = np.random.default_rng(7)
    for _ in range(200):
        vector = _random_vector(rng)
        n_shards = int(rng.integers(1, 8))
        cuts = np.sort(rng.integers(0, len(vector) + 1, size=n_shards - 1))
        bank = IntHistogramBank()
        for piece in np.split(vector, cuts):
            nonzero = piece[piece != 0]
            shard_bank = IntHistogramBank()
            shard_bank.add_column("G", nonzero, n_cells=len(piece))
            bank.add(shard_bank)
        assert bank.n_cells("G") == len(vector)
        for quantile in QUANTILES:
            expected = np.percentile(vector.astype(np.float32), quantile)
            assert bank.percentile("G", quantile, np.float32).tobytes() == expected.tobytes()


def test_histogram_merge_associative_commutative_and_roundtrip():
    rng = np.random.default_rng(11)
    banks = []
    for _ in range(3):
        bank = IntHistogramBank()
        for gene in ("A", "B", "C"):
            bank.add_column(gene, rng.poisson(3, size=500), n_cells=500)
        banks.append(bank)

    def merged(order, grouping):
        first, second, third = (IntHistogramBank().add(banks[index]) for index in order)
        if grouping == "left":
            return first.add(second).add(third)
        return first.add(second.add(third))

    reference = merged((0, 1, 2), "left")
    assert reference == merged((0, 1, 2), "right")
    assert reference == merged((2, 0, 1), "left")
    assert reference == merged((1, 2, 0), "right")
    path = _scratch_dir() / "bank.npz"
    reference.save(path)
    assert IntHistogramBank.load(path) == reference
    try:
        reference.save(path)
        raise AssertionError("save must refuse to overwrite")
    except FileExistsError:
        pass


def test_gene_sums_shard_invariance_and_float32_mean():
    rng = np.random.default_rng(5)
    dense = rng.negative_binomial(1, 0.2, size=(5000, 120)).astype(np.float32)
    dense[rng.random(dense.shape) < 0.6] = 0
    matrix = sp.csr_matrix(dense)
    whole = GeneSums(120).add_matrix(matrix)
    assert np.array_equal(whole.nnz, matrix.getnnz(axis=0))
    for n_shards in (3, 7):
        pieces = np.array_split(np.arange(5000), n_shards)
        merged = GeneSums(120)
        for rows in reversed(pieces):  # merge order must not matter
            merged.add(GeneSums(120).add_matrix(matrix[rows]))
        assert merged == whole
    assert whole.exact_in_float32()
    spore_mean = np.array(matrix.sum(axis=0)).flatten() / matrix.shape[0]  # phase06:77
    assert whole.mean_like_spore().tobytes() == spore_mean.tobytes()
    path = _scratch_dir() / "sums.npz"
    whole.save(path)
    assert GeneSums.load(path) == whole


def test_f32_mean_accumulator_bit_for_bit_vs_scipy():
    rng = np.random.default_rng(3)
    for trial in range(6):
        n_cells = int(rng.choice([1, 2, 17, 4000, 30000]))
        n_genes = int(rng.choice([5, 400]))
        dense = rng.negative_binomial(1, 0.05, size=(n_cells, n_genes)).astype(np.float32)
        dense[rng.random(dense.shape) < rng.uniform(0.3, 0.95)] = 0
        matrix = sp.csr_matrix(dense)
        reference = np.asarray(matrix.mean(axis=0)).ravel()
        for n_blocks in (1, 3, 11):
            cuts = np.sort(rng.integers(0, n_cells + 1, size=n_blocks - 1))
            accumulator = F32MeanAccumulator(n_genes, n_cells)
            start = 0
            for stop in list(cuts) + [n_cells]:
                accumulator.add_block(matrix[start:stop])  # includes empty blocks
                start = stop
            assert accumulator.result().tobytes() == reference.tobytes(), (trial, n_blocks, n_cells)
    # the failure mode this class exists to avoid: naive partial sums are NOT identical
    dense = rng.negative_binomial(1, 0.05, size=(30000, 400)).astype(np.float32)
    dense[rng.random(dense.shape) < 0.7] = 0
    matrix = sp.csr_matrix(dense)
    reference = np.asarray(matrix.mean(axis=0)).ravel()
    naive = np.zeros(400, dtype=np.float32)
    for start in range(0, 30000, 7000):
        naive = naive + np.asarray((matrix[start:start + 7000] * (1.0 / 30000)).sum(axis=0)).ravel()
    assert naive.tobytes() != reference.tobytes()
    path = _scratch_dir() / "acc.npz"
    accumulator = F32MeanAccumulator(400, 30000).add_block(matrix[:100])
    accumulator.save(path)
    assert np.array_equal(F32MeanAccumulator.load(path).carry, accumulator.carry)
    print("  f32 mean accumulator: bit-for-bit vs scipy for 1/3/11 blocks; naive partial sums differ (as expected)")


def test_waterfall_merge_and_roundtrip():
    first, second = Waterfall(), Waterfall()
    first.count("start", 10).count("qc_fail", 2)
    second.count("start", 5).count("escaper", 1)
    assert Waterfall().add(first).add(second) == Waterfall().add(second).add(first)
    assert dict(Waterfall().add(first).add(second).steps) == {"start": 15, "qc_fail": 2, "escaper": 1}
    path = _scratch_dir() / "waterfall.json"
    first.save(path)
    assert Waterfall.load(path) == first


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(error).__name__}: {error}")
    sys.exit(1 if failures else 0)
