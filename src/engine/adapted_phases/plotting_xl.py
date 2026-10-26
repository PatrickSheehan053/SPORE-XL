"""XL clone of SPORE src/plotting.py::plot_sparsity_overview (SPORE v3.0, GitHub 5015dc6, lines 107-241).

SPORE computes cells-per-gene, genes-per-cell and total counts from the matrix; XL has them as summary arrays (gene detection counts
from the merged raw gene sums, per-cell metrics from the map1 QC tables), so the drawing code (lines 154-241) is reproduced here on
arrays. All other report figures call SPORE's own plotting functions on small stand-in objects (see src/stage_report.py).
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from src.report.plotting import _THEME, _apply_theme


def plot_sparsity_overview_arrays(cells_per_gene, genes_per_cell, total_counts, n_cells, n_genes):
    _apply_theme()
    cpg, gpc = np.asarray(cells_per_gene), np.asarray(genes_per_cell)
    total_counts = np.asarray(total_counts, dtype=np.float32)

    fig = plt.figure(figsize=(15, 6.5))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.25)
    inner_left = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], width_ratios=[4, 1], height_ratios=[1, 4],
                                                  wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(inner_left[1, 0])
    ax_top = fig.add_subplot(inner_left[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(inner_left[1, 1], sharey=ax_main)

    log_counts = np.log10(total_counts + 1)
    if n_cells > 50000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_cells, 50000, replace=False)
        plot_x, plot_y = log_counts[idx], gpc[idx]
    else:
        plot_x, plot_y = log_counts, gpc

    ax_main.scatter(plot_x, plot_y, color=_THEME["low"], s=5, alpha=0.6, edgecolor="None", rasterized=True)
    sns.kdeplot(x=plot_x, ax=ax_top, color=_THEME["high"], fill=True, alpha=0.8, lw=1.5)
    sns.kdeplot(y=plot_y, ax=ax_right, color=_THEME["high"], fill=True, alpha=0.8, lw=1.5)
    ax_top.set_ylabel("Cell\nDensity", fontsize=10, fontweight="bold", rotation=0, labelpad=25, va="center")
    ax_right.set_xlabel("Cell\nDensity", fontsize=10, fontweight="bold")
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
    ax_right.xaxis.set_major_locator(ticker.MaxNLocator(nbins=1))
    ax_right.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '0' if x == 0 else f"{x:g}"))
    ax_main.set_xlabel("Total UMI Counts (log10)", fontweight="bold")
    ax_main.set_ylabel("Genes Detected per Cell", fontweight="bold")
    ax_top.set_title("A  Sequencing Saturation", loc="left", fontweight="bold", pad=15)

    inner_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.45)
    ax_r1 = fig.add_subplot(inner_right[0])
    ax_r2 = fig.add_subplot(inner_right[1])
    p_low = max(cpg.min(), 1)
    p_high = max(cpg.max(), 2)
    log_bins = np.logspace(np.log10(p_low * 0.8), np.log10(p_high * 1.2), 50)
    ax_r1.hist(cpg, bins=log_bins, color=_THEME["mid"], edgecolor="white", lw=0.5)
    ax_r1.set_xscale("log")
    ax_r1.set_xlim(p_low * 0.5, p_high * 1.5)
    ax_r1.axvline(np.median(cpg), color=_THEME["accent"], ls="--", lw=1.5, label=f"Median: {int(np.median(cpg)):,}")
    ax_r1.set_xlabel("Cells expressing gene (log scale)")
    ax_r1.set_ylabel("Number of Genes")
    ax_r1.set_title("B  Gene Detection Frequency", loc="left", fontweight="bold")
    ax_r1.legend(loc="upper left")

    sorted_g = np.sort(gpc)
    ecdf = np.arange(1, len(sorted_g) + 1) / len(sorted_g)
    ax_r2.plot(sorted_g, ecdf, color=_THEME["low"], lw=2.5)
    ax_r2.axvline(np.median(gpc), color=_THEME["accent"], ls="--", lw=1.5, label=f"Median: {int(np.median(gpc)):,}")
    ax_r2.set_xlabel("Genes Detected per Cell")
    ax_r2.set_ylabel("Cumulative Fraction")
    ax_r2.set_title("C  Genes per Cell (ECDF)", loc="left", fontweight="bold")
    ax_r2.legend(loc="lower right")

    fig.suptitle(f"Data Sparsity & Saturation  |  {n_cells:,} cells × {n_genes:,} genes", fontsize=15, fontweight="bold", y=1.04)
    sns.despine()
    return fig
