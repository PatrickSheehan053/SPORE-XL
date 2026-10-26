"""XL clone of the DECISION parts of SPORE src/phase05_escaper_filtering.py (SPORE v3.0, GitHub 5015dc6).

Reproduces: lines 170-181 (per-gene pass test against the control percentile threshold, ANDed over constituent
genes), lines 120-130 (perturbations with no gene in the matrix are bypassed, all cells kept), lines 185-197
(efficiency), lines 239-258 (undersized perturbations). The threshold itself is computed in reduce1 from the
merged control histogram (== np.percentile on the dense control column).
"""
import numpy as np


def gene_pass(values, threshold, direction):
    # phase05:171-179. `threshold` is a numpy scalar in the dtype SPORE's np.percentile returned.
    if direction == "knockdown":
        return values <= threshold
    return values >= threshold


def efficiency(n_kept, n_total):
    # phase05:185
    return n_kept / max(n_total, 1)


def undersized_labels(sizes, min_cells, ctrl_label):
    """sizes: dict label -> cells left AFTER the escaper filter (value_counts at phase05:245).
    phase05:246: undersized = sizes[(sizes < min_cells) & (sizes.index != ctrl_label)]."""
    return {label for label, size in sizes.items() if size < min_cells and label != ctrl_label}


def per_cell_efficiency_columns(labels, efficiency_by_label, depth_by_label):
    """phase05:205-212: defaults 1.0 / nan for controls and unrecorded labels, float32 columns."""
    eff_scores = np.ones(len(labels), dtype=np.float32)
    kd_depths = np.full(len(labels), float("nan"), dtype=np.float32)
    for index, label in enumerate(labels):
        if label in efficiency_by_label:
            eff_scores[index] = efficiency_by_label[label]
            kd_value = depth_by_label.get(label, float("nan"))
            if not np.isnan(kd_value):
                kd_depths[index] = kd_value
    return eff_scores, kd_depths
