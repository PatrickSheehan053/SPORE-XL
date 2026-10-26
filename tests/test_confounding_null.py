"""P5 batch-confounding check calibrated against a sampling null (K562 on 22 Oct: "5092 of 9118 perturbations TVD >= 0.5" over 267 gem groups with ~200
cells per perturbation: random sampling alone puts most batches at 0 or 1 cell, so the TVD from the controls is already about 0.5).
 1 267 batches, 60,000 controls, 300 perturbations of 200 cells drawn at random from the control mix: the OLD fixed-threshold rule flags almost all of
   them, the calibrated rule flags (almost) none (expected 0.3 at alpha 0.001; the test allows 3)
 2 real confounding is still found: perturbations whose cells all sit in 2 batches, and ones with 60 percent in one batch
 3 few batches (3) with a mild imbalance is not flagged (TVD below the 0.5 floor); one batch flags nothing; the table is reproducible
Run: python -B tests\\test_confounding_null.py   (no pipeline run, a few seconds)"""
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "engine"))

import numpy as np  # noqa: E402

import diagnostics_xl  # noqa: E402


def build(n_batches, n_controls, perturbations, seed=3, weights=None):
    """perturbations: {label: callable(rng) -> array of batch indices}. Controls follow `weights` over the batches."""
    rng = np.random.default_rng(seed)
    weights = np.ones(n_batches) / n_batches if weights is None else np.asarray(weights) / np.sum(weights)
    labels, batches = ["non-targeting"] * n_controls, list(rng.choice(n_batches, size=n_controls, p=weights))
    for label, draw in perturbations.items():
        drawn = draw(rng)
        labels += [label] * len(drawn)
        batches += list(drawn)
    return np.array(labels), np.array([f"b{value:03d}" for value in batches])


WEIGHTS = np.random.default_rng(1).gamma(4.0, size=267)      # uneven batch sizes, as in a real screen
SHARES = WEIGHTS / WEIGHTS.sum()


def from_controls(size):
    return lambda rng: rng.choice(267, size=size, p=SHARES)


def test_random_sampling_is_not_flagged_but_the_old_rule_flags_it():
    perturbations = {f"G{index:03d}": from_controls(150) for index in range(300)}
    labels, batches = build(267, 60000, perturbations, weights=WEIGHTS)
    started = time.time()
    table, summary = diagnostics_xl.confounding_table(labels, batches, "non-targeting")
    assert summary["n_flagged_uncalibrated"] >= 150, f"the old rule must reproduce the false positives: {summary['n_flagged_uncalibrated']}"
    assert summary["n_flagged"] <= 3, f"calibrated rule flagged {summary['n_flagged']} of 300 random perturbations"
    assert 0.40 < table["tvd_null_mean"].mean() < 0.75 and (table["p_value"] > 0).all()
    print(f"    random: old rule flags {summary['n_flagged_uncalibrated']} of 300, calibrated rule {summary['n_flagged']} "
          f"(null TVD for 150 cells over 267 batches = {table['tvd_null_mean'].mean():.2f}; {time.time() - started:.1f} s)")


def test_real_confounding_is_still_found():
    perturbations = {f"R{index:03d}": from_controls(200) for index in range(100)}
    perturbations |= {f"ONE{index}": (lambda rng, index=index: np.array([index] * 150 + [index + 1] * 50)) for index in range(5)}          # all cells in 2 batches
    perturbations |= {f"HALF{index}": (lambda rng, index=index: np.concatenate([np.full(120, 20 + index), rng.choice(267, size=80, p=SHARES)])) for index in range(5)}
    labels, batches = build(267, 60000, perturbations, weights=WEIGHTS)
    table, summary = diagnostics_xl.confounding_table(labels, batches, "non-targeting")
    flagged = set(table.loc[table["flagged"], "label"])
    assert {f"ONE{index}" for index in range(5)} <= flagged and {f"HALF{index}" for index in range(5)} <= flagged, sorted(flagged)
    assert len(flagged - {f"ONE{i}" for i in range(5)} - {f"HALF{i}" for i in range(5)}) <= 1, sorted(flagged)
    assert summary["n_fully_in_one_batch"] == 0 and (table.set_index("label").loc["ONE0", "p_value"] < 0.001)
    print(f"    confounded: all 10 planted perturbations flagged, {summary['n_flagged'] - 10} extra of 100 random")


def test_few_batches_one_batch_and_reproducibility():
    perturbations = {f"M{index}": (lambda rng: rng.choice(3, size=500, p=[0.45, 0.35, 0.20])) for index in range(10)}
    labels, batches = build(3, 30000, perturbations, weights=[1, 1, 1])
    table, summary = diagnostics_xl.confounding_table(labels, batches, "non-targeting")
    assert summary["n_flagged"] == 0 and table["tvd_vs_controls"].max() < 0.5, "mild imbalance over 3 batches is below the 0.5 floor"
    labels, batches = build(1, 1000, {f"S{index}": (lambda rng: np.zeros(50, dtype=int)) for index in range(4)})
    table, summary = diagnostics_xl.confounding_table(labels, batches, "non-targeting")
    assert summary["n_flagged"] == 0 and not table["flagged"].any()
    labels, batches = build(267, 20000, {f"G{index}": (lambda rng: rng.integers(0, 267, size=100)) for index in range(20)})
    first, _ = diagnostics_xl.confounding_table(labels, batches, "non-targeting")
    second, _ = diagnostics_xl.confounding_table(labels, batches, "non-targeting")
    assert first.equals(second), "the null uses a fixed seed: the table must be reproducible"
    print("    3 batches: not flagged; 1 batch: nothing flagged; two calls give an identical table")


if __name__ == "__main__":
    test_random_sampling_is_not_flagged_but_the_old_rule_flags_it()
    test_real_confounding_is_still_found()
    test_few_batches_one_batch_and_reproducibility()
    print("PASS test_confounding_null")
