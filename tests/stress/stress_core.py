"""Shared generator for the stress suite (blueprint Amendment 2, P11): negative-binomial Perturb-seq counts with CRISPRi knockdown,
downstream effects, escapers, batches, donors, depth variation and QC failures. Each dataset builder in make_stress_suite.py
fills in a perturbation table and calls simulate(); the truth table keeps every designed fact per cell.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

CONTROL = "non-targeting"
S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL",
           "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN"]
G2M_GENES = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B",
             "MKI67", "TMPO", "CENPF", "TACC3", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB"]
MT_GENES = ["MT-ND1", "MT-ND2", "MT-ND3", "MT-ND4", "MT-ND4L", "MT-ND5", "MT-ND6", "MT-CO1", "MT-CO2",
            "MT-CO3", "MT-ATP6", "MT-ATP8", "MT-CYB"]
RIBO_GENES = [f"RPL{number}" for number in (3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22)] + \
             [f"RPS{number}" for number in (2, 3, 4, 5, 6, 8, 9, 12, 13, 15, 16, 18)]


def build_gene_table(rng, n_genes, low_fraction=0.1, extra_names=()):
    """Gene names, kind (mt / ribo / cc / plain / extra), mean expression and the positions of the very-low-expressed genes."""
    special = MT_GENES + RIBO_GENES + S_GENES + G2M_GENES + list(extra_names)
    plain = [f"GENE{number:04d}" for number in range(n_genes - len(special))]
    names = np.array(special + plain)
    kind = np.array(["mt"] * len(MT_GENES) + ["ribo"] * len(RIBO_GENES) + ["cc"] * (len(S_GENES) + len(G2M_GENES)) +
                    ["extra"] * len(extra_names) + ["plain"] * len(plain))
    order = rng.permutation(n_genes)
    names, kind = names[order], kind[order]
    mean_expr = np.clip(rng.lognormal(mean=np.log(0.5), sigma=1.5, size=n_genes), 0.02, 40.0)
    plain_positions = np.flatnonzero(kind == "plain")
    low_positions = rng.choice(plain_positions, size=int(low_fraction * n_genes), replace=False)
    mean_expr[low_positions] = rng.uniform(0.0008, 0.008, size=len(low_positions))
    mean_expr[kind == "mt"] = 8.0
    mean_expr[kind == "ribo"] = rng.uniform(4.0, 25.0, size=int((kind == "ribo").sum()))
    mean_expr[kind == "cc"] = rng.uniform(0.3, 3.0, size=int((kind == "cc").sum()))
    mean_expr[kind == "extra"] = 1.0
    return names, kind, mean_expr, low_positions


def pert_table(labels, gene_pos, sizes, kd, batch_sets=None, donor_sets=None, depth_mult=None, escaper_frac=0.10, cls=None):
    count = len(labels)
    return pd.DataFrame({
        "label": list(labels), "gene_pos": np.asarray(gene_pos, dtype=np.int64), "size": np.asarray(sizes, dtype=np.int64),
        "kd": np.asarray(kd, dtype=np.float64),
        "batch_set": batch_sets if batch_sets is not None else [None] * count,
        "donor_set": donor_sets if donor_sets is not None else [None] * count,
        "depth_mult": np.ones(count) if depth_mult is None else np.asarray(depth_mult, dtype=np.float64),
        "escaper_frac": np.full(count, escaper_frac) if np.isscalar(escaper_frac) else np.asarray(escaper_frac, dtype=np.float64),
        "cls": ["ordinary"] * count if cls is None else list(cls)})


def lognormal_sizes(rng, count, mean, sigma, minimum):
    raw = rng.lognormal(mean=0.0, sigma=sigma, size=count)
    return np.maximum(minimum, np.round(raw / raw.mean() * mean)).astype(np.int64)


def simulate(rng, names, kind, mean_expr, perts, n_controls, n_batches, n_donors=1, control_batches=None, control_donors=None,
             depth_sigma=0.35, qc_fail=(150, 150, 40), batch_sigma=0.3, n_batch_genes=100, donor_sigma=0.0, library_target=2200.0,
             dispersion=5.0, batch_sorted=False, downstream=(10, 30), effect_scale=0.3, chunk=5000):
    """Returns (sparse count matrix float32, truth DataFrame). Rows are shuffled (or batch-sorted); QC failures are random cells."""
    n_genes = len(names)
    n_perts = len(perts)
    sizes = perts["size"].to_numpy()
    n_cells = int(sizes.sum()) + n_controls
    cell_pert = np.concatenate([np.repeat(np.arange(n_perts), sizes), np.full(n_controls, -1)])
    cell_batch = np.zeros(n_cells, dtype=np.int64)
    cell_donor = np.zeros(n_cells, dtype=np.int64)
    start = 0
    for index in range(n_perts):
        stop = start + int(sizes[index])
        allowed_batches = perts["batch_set"].iloc[index]
        allowed_donors = perts["donor_set"].iloc[index]
        cell_batch[start:stop] = rng.choice(allowed_batches if allowed_batches is not None else np.arange(n_batches), size=stop - start)
        cell_donor[start:stop] = rng.choice(allowed_donors if allowed_donors is not None else np.arange(n_donors), size=stop - start)
        start = stop
    if n_controls:
        cell_batch[start:] = rng.choice(control_batches if control_batches is not None else np.arange(n_batches), size=n_controls)
        cell_donor[start:] = rng.choice(control_donors if control_donors is not None else np.arange(n_donors), size=n_controls)

    order = rng.permutation(n_cells)
    cell_pert, cell_batch, cell_donor = cell_pert[order], cell_batch[order], cell_donor[order]
    if batch_sorted:
        order = np.argsort(cell_batch, kind="stable")
        cell_pert, cell_batch, cell_donor = cell_pert[order], cell_batch[order], cell_donor[order]
    is_perturbed = cell_pert >= 0
    escaper_rate = np.where(is_perturbed, perts["escaper_frac"].to_numpy()[np.maximum(cell_pert, 0)], 0.0)
    is_escaper = is_perturbed & (rng.random(n_cells) < escaper_rate)

    chosen = rng.permutation(n_cells)
    n_low, n_mt, n_high = qc_fail
    low_cells, mt_cells, high_cells = chosen[:n_low], chosen[n_low:n_low + n_mt], chosen[n_low + n_mt:n_low + n_mt + n_high]
    depth_mult = np.where(is_perturbed, perts["depth_mult"].to_numpy()[np.maximum(cell_pert, 0)], 1.0)
    size_factor = rng.lognormal(mean=0.0, sigma=depth_sigma, size=n_cells) * depth_mult
    size_factor[low_cells] *= 0.05
    size_factor[high_cells] *= 50.0
    mt_boost = np.ones(n_cells)
    mt_boost[mt_cells] = 40.0
    mt_positions = np.flatnonzero(kind == "mt")

    effect_log = np.zeros((n_perts, n_genes), dtype=np.float32)
    strength = rng.lognormal(mean=np.log(effect_scale), sigma=0.7, size=n_perts)
    target_positions = perts["gene_pos"].to_numpy()
    for index in range(n_perts):
        pool = np.setdiff1d(np.arange(n_genes), [target_positions[index]])
        chosen_genes = rng.choice(pool, size=int(rng.integers(downstream[0], downstream[1] + 1)), replace=False)
        effect_log[index, chosen_genes] = rng.normal(0, strength[index], size=len(chosen_genes))
    batch_effect = np.zeros((n_batches, n_genes))
    for batch_index in range(1, n_batches):
        shifted = rng.choice(n_genes, size=n_batch_genes, replace=False)
        batch_effect[batch_index, shifted] = rng.normal(0, batch_sigma, size=n_batch_genes)
    donor_effect = np.zeros((n_donors, n_genes))
    if donor_sigma > 0:
        for donor_index in range(n_donors):
            shifted = rng.choice(n_genes, size=n_batch_genes, replace=False)
            donor_effect[donor_index, shifted] = rng.normal(0, donor_sigma, size=n_batch_genes)

    scale = library_target / mean_expr.sum()
    base_log = np.log(mean_expr * scale)
    knockdown_log = np.log(perts["kd"].to_numpy())
    blocks = []
    for first in range(0, n_cells, chunk):
        rows = np.arange(first, min(first + chunk, n_cells))
        log_mu = base_log[None, :] + batch_effect[cell_batch[rows]] + donor_effect[cell_donor[rows]]
        active = is_perturbed[rows] & ~is_escaper[rows]
        pert_rows = cell_pert[rows]
        log_mu[active] += effect_log[pert_rows[active]]
        log_mu[np.flatnonzero(active), target_positions[pert_rows[active]]] += knockdown_log[pert_rows[active]]
        log_mu[:, mt_positions] += np.log(mt_boost[rows])[:, None]
        mu = np.exp(log_mu) * size_factor[rows][:, None]
        counts = rng.poisson(rng.gamma(shape=dispersion, scale=mu / dispersion))
        blocks.append(sp.csr_matrix(counts.astype(np.float32)))
    matrix = sp.vstack(blocks, format="csr")

    labels = np.where(is_perturbed, perts["label"].to_numpy()[np.maximum(cell_pert, 0)], CONTROL)
    qc_type = np.full(n_cells, "ok", dtype=object)
    qc_type[low_cells], qc_type[mt_cells], qc_type[high_cells] = "low_count", "high_mt", "high_count"
    cls = np.where(is_perturbed, perts["cls"].to_numpy()[np.maximum(cell_pert, 0)], "control")
    truth = pd.DataFrame({
        "cell": [f"cell_{number:06d}" for number in range(n_cells)], "row": np.arange(n_cells), "label": labels,
        "batch": (cell_batch + 1).astype(str), "donor": [f"D{number + 1}" for number in cell_donor], "is_escaper": is_escaper,
        "qc_type": qc_type.astype(str), "depth_factor": size_factor.astype(np.float32), "cls": cls})
    return matrix, truth


def make_obs(truth, label_column="label", batch_name="gem_group", with_donor=False):
    obs = pd.DataFrame({
        "gene": pd.Categorical(truth[label_column].astype(str)),  # sorted categories (intake convention)
        batch_name: pd.Categorical(truth["batch"].astype(str)),
    }, index=pd.Index(truth["cell"].to_numpy(), name="cell"))
    if with_donor:
        obs["donor"] = pd.Categorical(truth["donor"].astype(str))
    return obs
