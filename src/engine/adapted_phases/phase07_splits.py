"""XL clone of SPORE src/phase07_splits.py::split_zero_shot (SPORE v3.0, GitHub 5015dc6), fed a precomputed mean-shift
Series instead of an AnnData, with the destructive split replaced by label lists + an explicit control assignment.

Lines 183-277 are reproduced in the same order, so the RNG call sequence is identical:
  1. rng = np.random.default_rng(seed)
  2. one rng.integers(1e9) per stratification bin, in ascending bin order (line 239)
  3. one rng.shuffle of the control positions (line 257)
The mean shift (`deg_counts`: pd.Series, descending, index = perturbation label) is computed in reduce2 with the
float32 carried-accumulator method that reproduces phase07:156-173 bit for bit.
"""
import numpy as np
import pandas as pd


def split_from_scores(deg_counts, n_controls, cfg, seed=None):
    zs = cfg.get("phase7_splits", {})
    if seed is None:
        seed = zs.get("random_seed", 42)
    rng = np.random.default_rng(seed)

    n_perts = len(deg_counts)
    test_n_req = zs.get("test_n")
    val_n_req = zs.get("val_n")
    if test_n_req is not None and val_n_req is not None:
        if int(test_n_req) + int(val_n_req) >= n_perts:
            raise ValueError(
                f"[Phase 7] test_n ({test_n_req}) + val_n ({val_n_req}) = "
                f"{int(test_n_req)+int(val_n_req)} >= {n_perts} surviving perturbations after triage — "
                f"no perturbations left to train on. Lower test_n/val_n in the config.")

    test_n_cfg = zs.get("test_n", None)
    val_n_cfg = zs.get("val_n", None)
    if test_n_cfg is not None:
        n_test = max(1, min(int(test_n_cfg), n_perts - 2))
    else:
        test_ratio = zs.get("test_ratio", 0.25)
        n_test = max(1, int(n_perts * test_ratio))
    n_remain = n_perts - n_test
    if val_n_cfg is not None:
        n_val = max(1, min(int(val_n_cfg), n_remain - 1))
    else:
        val_ratio = zs.get("val_ratio", 0.10)
        n_val = max(1, int(n_remain * val_ratio))

    test_ratio_eff = n_test / n_perts
    val_ratio_eff = n_val / max(n_remain, 1)

    df = deg_counts.reset_index()
    df.columns = ["perturbation", "score"]
    df["bin"] = pd.qcut(df["score"], q=zs.get("stratify_bins", 4), labels=False, duplicates="drop")

    train_labels, val_labels, test_labels = [], [], []
    for _, group in df.groupby("bin"):
        shuffled = group.sample(frac=1, random_state=rng.integers(1e9))
        bin_test = max(1, int(len(shuffled) * test_ratio_eff)) if len(shuffled) > 3 else 1
        bin_remain = len(shuffled) - bin_test
        bin_val = max(1, int(bin_remain * val_ratio_eff)) if bin_remain > 2 else 1
        test_labels.extend(shuffled.iloc[:bin_test]["perturbation"])
        val_labels.extend(shuffled.iloc[bin_test:bin_test + bin_val]["perturbation"])
        train_labels.extend(shuffled.iloc[bin_test + bin_val:]["perturbation"])

    # control cells: positions 0..K-1 stand for the K surviving controls in global (= SPORE row) order
    control_positions = np.arange(n_controls, dtype=np.int64)
    rng.shuffle(control_positions)
    frac_test = len(test_labels) / n_perts
    frac_val = len(val_labels) / n_perts
    n_ctrl_test = int(len(control_positions) * frac_test)
    n_ctrl_val = int(len(control_positions) * frac_val)
    return {
        "train_labels": train_labels, "val_labels": val_labels, "test_labels": test_labels, "seed": seed,
        "control_test_positions": control_positions[:n_ctrl_test],
        "control_val_positions": control_positions[n_ctrl_test:n_ctrl_test + n_ctrl_val],
        "control_train_positions": control_positions[n_ctrl_test + n_ctrl_val:],
    }
