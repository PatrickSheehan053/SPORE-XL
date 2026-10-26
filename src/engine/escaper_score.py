"""P4: Mixscape-style per-cell escaper score, an OPTION (phase5_escaper_filtering.escaper_score: off | report | union | replace).

Rule 1 (SPORE v3.0): a cell escapes when its target gene is above the 10th percentile of that gene in controls. It needs the
target to be expressed in controls; for a low-expressed target (control p10 = 0) it sees nothing (KNOWN_ISSUES 5).
Rule 2 (this module): look at the RESPONSE instead of the target.
  1. every cell is normalized to 1e4 over the post-ribosomal genes and log1p'd
  2. a fixed random sample of up to 2,000 QC-passing control cells is the reference (mean mu_c, rows R); the sample is the cells with the
     smallest blake2b hash of xl_global_index, so it does not depend on the shard layout
  3. for perturbation p the signature d = mean(p) - mu_c over the TOP_GENES genes with the largest |d| (the target gene itself is excluded,
     so the score is independent of rule 1)
  4. score(cell) = (x - mu_c) . d / (d . d) over those genes: about 1 for an average perturbed cell, about 0 for a control-like cell
  5. the reference cells get scores too (mean m0, sd s0); a cell is a rule-2 escaper when score <= m0 + z * s0 (default z = 2)
Known bias: the signature is estimated from the same cells (as in Mixscape), so scores of a perturbation with no real effect are pushed
up a little. The option is for evaluation (report mode counts both rules); nothing changes by default.
"""
import hashlib

import numpy as np
import scipy.sparse as sp

TOP_GENES = 100
REFERENCE_PER_SHARD = 300
REFERENCE_TOTAL = 2000
MIN_REFERENCE = 20
MIN_CELLS = 20
MODES = ("off", "report", "union", "replace")


def normalize_mode(value):
    if value in (False, None, "off"):
        return "off"
    if value is True:
        return "union"
    if value not in MODES:
        raise ValueError(f"phase5_escaper_filtering.escaper_score must be one of {MODES} (or true/false), got {value!r}")
    return value


def lognorm_rows(matrix, target_sum=1e4):
    """CSR log1p(CP10k) of the given rows (float32)."""
    matrix = sp.csr_matrix(matrix, dtype=np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.where(totals > 0, target_sum / np.maximum(totals, 1e-12), 0.0).astype(np.float32)
    scaled = sp.diags(scale) @ matrix
    scaled.data = np.log1p(scaled.data)
    return sp.csr_matrix(scaled)


def row_hash(global_index):
    digest = hashlib.blake2b(int(global_index).to_bytes(8, "little", signed=False), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def shard_reference(matrix, global_index, is_control, per_shard=REFERENCE_PER_SHARD):
    """The (at most `per_shard`) control rows of this shard with the smallest hash: (hashes, global_index, dense log-norm rows)."""
    rows = np.flatnonzero(is_control)
    if not len(rows):
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64), np.zeros((0, matrix.shape[1]), dtype=np.float32)
    hashes = np.array([row_hash(global_index[row]) for row in rows], dtype=np.uint64)
    keep = np.argsort(hashes, kind="stable")[:per_shard]
    chosen = rows[keep]
    return hashes[keep], np.asarray(global_index)[chosen].astype(np.int64), lognorm_rows(matrix[chosen]).toarray()


def merge_references(parts, total=REFERENCE_TOTAL):
    """parts: iterable of (hashes, global_index, rows). Keeps the `total` smallest hashes over all shards; shard order does not matter."""
    parts = [part for part in parts if len(part[0])]
    if not parts:
        return None
    hashes = np.concatenate([part[0] for part in parts])
    index = np.concatenate([part[1] for part in parts])
    rows = np.vstack([part[2] for part in parts])
    order = np.lexsort((index, hashes))[:total]
    order = order[np.argsort(index[order], kind="stable")]  # store in global order
    return hashes[order], index[order], rows[order]


def score_perturbation(cell_lognorm, reference_rows, target_columns, z=2.0, top_genes=TOP_GENES, mu_control=None):
    """cell_lognorm: sparse log-norm rows of ONE perturbation; reference_rows: dense control reference; mu_control: its column mean (pass it
    when scoring many perturbations: it is the same for all). Returns (flags, info) with flags True where the cell looks like a control
    (rule-2 escaper); None when the rule cannot be applied."""
    n_cells = cell_lognorm.shape[0]
    if reference_rows is None or len(reference_rows) < MIN_REFERENCE:
        return None, {"status": "no control reference"}
    if n_cells < MIN_CELLS:
        return None, {"status": f"fewer than {MIN_CELLS} cells"}
    if mu_control is None:
        mu_control = reference_rows.mean(axis=0)
    signature = np.asarray(cell_lognorm.mean(axis=0)).ravel() - mu_control
    if len(target_columns):
        signature[list(target_columns)] = 0.0
    n_top = min(top_genes, signature.shape[0])
    selected = np.argpartition(-np.abs(signature), n_top - 1)[:n_top]
    selected = np.sort(selected)  # fixed order, so ties cannot depend on partition internals
    direction = signature[selected]
    denominator = float(direction @ direction)
    if denominator <= 1e-12:
        return None, {"status": "no response signature"}
    cell_scores = ((cell_lognorm[:, selected].toarray() - mu_control[selected]) @ direction) / denominator
    reference_scores = ((reference_rows[:, selected] - mu_control[selected]) @ direction) / denominator
    center, spread = float(reference_scores.mean()), float(reference_scores.std(ddof=1))
    flags = cell_scores <= center + z * spread
    return flags, {"status": "ok", "control_score_mean": center, "control_score_sd": spread, "median_cell_score": float(np.median(cell_scores))}
