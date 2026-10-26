"""XL clone of SPORE src/phase12_metacell.py::run_phase12 (SPORE v3.0, GitHub 5015dc6), as a per-block driver.

Reproduced from SPORE (called, not copied): _compute_group_pca (per-group TruncatedSVD-10 on RAW counts for groups of 50+
cells), _aggregate_group_mbk (MiniBatchKMeans, n_metacells = n_cells // k, MEAN of raw counts), _mean_rows.
Reproduced here: the grouping rule of lines 90-95 and 231-249 (controls single-cell, perturbed -> MBK k).

The ONE allowed deviation (blueprint Amendment 1): SPORE fits the global fallback embedding (phase12:75-87,
normalize_total(target_sum=None) -> log1p -> PCA-50) on ALL cells and uses it for perturbation groups with fewer than 50
cells. XL fits the same recipe on the HVG subsample (panel genes) and projects each small group through the fitted model.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from src.phases.phase12_metacell import _PER_GROUP_PCA_THRESHOLD, _aggregate_group_mbk, _compute_group_pca

PER_GROUP_PCA_THRESHOLD = _PER_GROUP_PCA_THRESHOLD


def group_embedding(X_sub, seed):
    # phase12:90-95
    if X_sub.shape[0] >= PER_GROUP_PCA_THRESHOLD:
        with threadpool_limits(limits=1):
            return _compute_group_pca(X_sub, n_comps=10, seed=seed)
    return None


def fit_fallback(subsample, seed, n_comps=50, max_dense_gb=6.0):
    """subsample: CSR raw counts (cells x panel genes). Returns a dict with the fitted model."""
    subsample = sp.csr_matrix(subsample, dtype=np.float32)
    if subsample.shape[0] * subsample.shape[1] * 4 > max_dense_gb * 1e9:
        raise MemoryError("fallback-embedding subsample would need more than the dense budget; lower hvg_subsample_cells")
    totals = np.asarray(subsample.sum(axis=1)).ravel()
    target = float(np.median(totals[totals > 0]))  # scanpy normalize_total(target_sum=None): median of non-zero totals
    dense = _normalize_log(subsample, target).toarray() if sp.issparse(subsample) else subsample
    n_actual = min(n_comps, dense.shape[0] - 1, dense.shape[1] - 1)
    with threadpool_limits(limits=1):
        pca = PCA(n_components=n_actual, svd_solver="randomized", random_state=seed).fit(dense)
    return {"target": target, "mean": pca.mean_.astype(np.float32), "components": pca.components_.astype(np.float32)}


def _normalize_log(matrix, target):
    matrix = sp.csr_matrix(matrix, dtype=np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.where(totals > 0, target / np.maximum(totals, 1e-12), 0.0).astype(np.float32)
    scaled = sp.diags(scale) @ matrix
    scaled.data = np.log1p(scaled.data)
    return sp.csr_matrix(scaled)


def project_fallback(model, X_sub):
    dense = _normalize_log(X_sub, model["target"]).toarray()
    return ((dense - model["mean"]) @ model["components"].T).astype(np.float32)


def aggregate_group(X_sub, local_embedding, model, k, seed):
    """Returns (expr list, ncells list, used_fallback bool)."""
    n_cells = X_sub.shape[0]
    n_metacells = max(1, n_cells // k)
    embedding = local_embedding
    used_fallback = False
    if embedding is None and n_metacells > 1 and model is not None:
        embedding = project_fallback(model, X_sub)
        used_fallback = True
    exprs, ncells = _aggregate_group_mbk(X_sub, embedding, k, seed)
    return exprs, ncells, used_fallback
