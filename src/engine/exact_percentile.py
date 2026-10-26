"""numpy.percentile (default 'linear' method) recomputed from an integer histogram.

Mirrors numpy._lerp operation for operation (same weak-scalar float handling), so the
result is bit-identical to np.percentile(vector, q) for a vector of the same dtype.
Checked against numpy 2.5.4; tests/test_stats.py re-verifies on every run.
"""
import numpy as np


def _order_statistic(cumulative, values, rank):
    # value at 0-based sorted position `rank`
    slot = np.searchsorted(cumulative, rank, side="right")
    return values[slot]


def percentile_from_histogram(values, counts, quantile, dtype=np.float64):
    """values: sorted distinct integers; counts: how many vector entries have each value.

    dtype is the dtype of the vector SPORE would have passed to np.percentile
    (float32 for a float32 matrix column). Returns a numpy scalar.
    """
    values = np.asarray(values)
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        raise ValueError("empty histogram")
    cumulative = np.cumsum(counts)
    virtual_index = (total - 1) * (quantile / 100)
    previous_index = int(np.floor(virtual_index))
    next_index = min(previous_index + 1, total - 1)
    gamma = float(virtual_index - previous_index)
    lower = _order_statistic(cumulative, values, previous_index).astype(dtype)
    upper = _order_statistic(cumulative, values, next_index).astype(dtype)
    diff = upper - lower
    result = lower + diff * gamma
    if gamma >= 0.5:
        result = upper - diff * (1 - gamma)
    return result
