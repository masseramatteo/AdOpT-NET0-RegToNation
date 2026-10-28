"""
Figures for the near-optimal comparison between the first-best 3000-run batch and the
"small electrolyzer excluded" re-optimisation batch.

Reads reopt_comparison_with_LCOH.xlsx (produced by build_lcoh_comparison.py), merges the
scenario parameters from the first-best batch summary, and writes PNG figures into
<REOPT_DIR>/figures_LCOH/.
"""

from __future__ import annotations

import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

BASE = r"X:/3000_simulations_24_may_2026"
ORIG_DIR = os.path.join(BASE, "Snellius_3000_simulations")
REOPT_DIR = os.path.join(
    BASE, "reopt_comparison_Snellius_3000_simulations_small_electrolyzer_only"
)
COMPARISON_FILE = os.path.join(REOPT_DIR, "reopt_comparison_with_LCOH.xlsx")
FIG_DIR = os.path.join(REOPT_DIR, "figures_LCOH")

# Runs whose re-optimised cost falls below the first-best cost by more than this are
# infeasible by construction and indicate an unconverged first-best solution.
CONVERGENCE_TOLERANCE_PCT = -0.1

# Reference palette (validated light mode, surface #fcfcfb).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"
# Sequential blue ramp, light -> dark (steps 100..700).
BLUE_RAMP = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#0d366b",
]

DRIVERS = [
    ("total_demand_TWh", "Total H2 demand [TWh]"),
    ("electricity_price_avg", "Average electricity price [EUR/MWh]"),
    ("hydrogen_import_price", "H2 import price [EUR/MWh]"),
    ("solar_cf_realized_mean", "Solar mean capacity factor [-]"),
    ("unbalance_ratio", "Unbalance ratio [-]"),
    ("import_availability_ratio", "Import availability ratio [-]"),
]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  {name}")


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def parse_removed_capacity(text: object) -> float:
    """Sum the removed capacities listed in 'Tech: Node=12.3, Node=4.5' strings."""
    if not isinstance(text, str):
        return np.nan
    return float(sum(float(v) for v in re.findall(r"=\s*(-?\d+(?:\.\d+)?)", text)))


def load_data() -> pd.DataFrame:
    df = pd.read_excel(COMPARISON_FILE)

    params = pd.read_csv(
        os.path.join(ORIG_DIR, "parallel_results_summary.csv"), sep=";"
    )
    keep = ["run_id"] + [c for c, _ in DRIVERS if c in params.columns]
    df = df.merge(params[keep], on="run_id", how="left")

    df["removed_capacity_MW"] = df["removed_tech_sizes"].apply(parse_removed_capacity)
    df["solved"] = df["delta_cost_pct"].notna()
    df["converged"] = df["delta_cost_pct"] >= CONVERGENCE_TOLERANCE_PCT
    return df


def binned_median(x: pd.Series, y: pd.Series, n_bins: int = 12):
    """Median of y within equal-count bins of x, with the interquartile band.

    Returns (bin centre, median, q25, q75).
    """
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    empty = (np.array([]),) * 4
    if len(x) < n_bins * 3:
        return empty
    bins = pd.qcut(x, n_bins, duplicates="drop")
    grouped = pd.DataFrame({"x": x, "y": y}).groupby(bins, observed=True)
    return (
        grouped["x"].median().to_numpy(),
        grouped["y"].median().to_numpy(),
        grouped["y"].quantile(0.25).to_numpy(),
        grouped["y"].quantile(0.75).to_numpy(),
    )


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def fig_distribution(df: pd.DataFrame) -> None:
    """Distribution of the LCOH penalty, with the infeasible-negative tail called out."""
    d = df.loc[df["solved"], "delta_lcoh_eur_per_kg"]
    fig, ax = plt.subplots(figsize=(8, 4.6))

    bins = np.linspace(d.min(), np.percentile(d, 99.5), 70)
    counts, edges = np.histogram(d, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    colors = [RED if c < 0 else BLUE for c in centres]
    ax.bar(centres, counts, width=np.diff(edges) * 0.9, color=colors, linewidth=0)

    median = d.median()
    ax.axvline(0, color=INK_MUTED, linewidth=1.2, zorder=0)
    ax.axvline(median, color=ORANGE, linewidth=2, linestyle="--", zorder=3)

    # Stats block in the sparse upper-right corner, clear of the bars near zero.
    n_neg = int((d < 0).sum())
    n_unconverged = int((df.loc[df["solved"], "delta_cost_pct"] < CONVERGENCE_TOLERANCE_PCT).sum())
    ax.text(
        0.98,
        0.95,
        f"median  +{median:.3f} EUR/kg\n"
        f"mean    +{d.mean():.3f} EUR/kg\n"
        f"p95     +{np.percentile(d, 95):.3f} EUR/kg\n"
        f"{n_neg} runs below zero, of which\n"
        f"{n_unconverged} beyond the MIP gap",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=INK_2,
        linespacing=1.6,
    )
    ax.text(
        0.03,
        0.32,
        "below zero:\nfirst best\nnot converged",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=RED,
        linespacing=1.5,
    )

    ax.set_xlabel("LCOH penalty of excluding Electrolyzer_small [EUR/kg H2]")
    ax.set_ylabel("Number of runs")
    ax.set_title("Most scenarios lose little by dropping the small electrolyzer")
    ax.grid(axis="x", visible=False)
    save(fig, "01_distribution_delta_lcoh.png")


def fig_ecdf(df: pd.DataFrame) -> None:
    """Share of scenarios below a given cost penalty."""
    d = df.loc[df["converged"], "delta_cost_pct"].sort_values()
    y = 100 * np.arange(1, len(d) + 1) / len(d)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(d, y, color=BLUE)
    ax.fill_between(d, 0, y, color=BLUE, alpha=0.10, linewidth=0)

    for threshold in (0.5, 1.0, 2.0, 5.0):
        share = 100 * (d <= threshold).mean()
        ax.plot([threshold], [share], "o", color=ORANGE, markersize=8, zorder=5)
        ax.annotate(
            f"{share:.0f}% below {threshold:g}%",
            xy=(threshold, share),
            xytext=(10, -12),
            textcoords="offset points",
            color=INK_2,
            fontsize=9,
        )

    ax.set_xlim(left=0)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Total system cost penalty [%]")
    ax.set_ylabel("Share of scenarios [%]")
    ax.set_title("Cumulative distribution of the near-optimal cost penalty")
    save(fig, "02_ecdf_cost_penalty.png")


def fig_level_scatter(df: pd.DataFrame) -> None:
    """First-best LCOH against re-optimised LCOH, with the identity line."""
    d = df[df["converged"]]
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    ax.scatter(
        d["lcoh_first_best_eur_per_kg"],
        d["lcoh_reopt_eur_per_kg"],
        s=12,
        color=BLUE,
        alpha=0.30,
        linewidth=0,
    )
    lims = [
        min(d["lcoh_first_best_eur_per_kg"].min(), d["lcoh_reopt_eur_per_kg"].min()),
        max(d["lcoh_first_best_eur_per_kg"].max(), d["lcoh_reopt_eur_per_kg"].max()),
    ]
    ax.plot(lims, lims, color=INK_MUTED, linewidth=1.4, linestyle="--", zorder=0)
    ax.annotate(
        "no penalty (y = x)",
        xy=(lims[0] + 0.75 * (lims[1] - lims[0]),) * 2,
        xytext=(6, -16),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=9,
    )
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("LCOH first best [EUR/kg H2]")
    ax.set_ylabel("LCOH without small electrolyzer [EUR/kg H2]")
    ax.set_title("Points sit on or just above the identity line")
    save(fig, "03_lcoh_levels_scatter.png")


def fig_drivers(df: pd.DataFrame) -> None:
    """Small multiples: LCOH penalty against each scenario driver."""
    d = df[df["converged"]]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharey=True)

    for ax, (col, label) in zip(axes.ravel(), DRIVERS):
        if col not in d.columns:
            ax.set_visible(False)
            continue
        ax.scatter(
            d[col], d["delta_lcoh_eur_per_kg"], s=7, color=BLUE, alpha=0.18, linewidth=0
        )
        bx, by, lo, hi = binned_median(d[col], d["delta_lcoh_eur_per_kg"], n_bins=10)
        if len(bx):
            ax.fill_between(bx, lo, hi, color=ORANGE, alpha=0.15, linewidth=0, zorder=4)
            ax.plot(bx, by, color=ORANGE, linewidth=2.4, zorder=5)
        ax.axhline(0, color=INK_MUTED, linewidth=1, zorder=0)
        ax.set_xlabel(label)
        ax.set_ylim(-0.05, np.percentile(d["delta_lcoh_eur_per_kg"], 99.5))

    for ax in axes[:, 0]:
        ax.set_ylabel("LCOH penalty [EUR/kg H2]")
    axes[0, 0].plot([], [], color=ORANGE, label="median per decile")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(
        "What drives the value of the small electrolyzer",
        fontsize=13,
        fontweight="bold",
        color=INK,
        x=0.09,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "04_penalty_vs_drivers.png")


def fig_removed_capacity(df: pd.DataFrame) -> None:
    """The penalty against how much small-electrolyzer capacity the first best had built."""
    d = df[df["converged"] & df["removed_capacity_MW"].notna()]
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.scatter(
        d["removed_capacity_MW"],
        d["delta_lcoh_eur_per_kg"],
        s=10,
        color=BLUE,
        alpha=0.22,
        linewidth=0,
    )
    bx, by, lo, hi = binned_median(
        d["removed_capacity_MW"], d["delta_lcoh_eur_per_kg"], n_bins=8
    )
    ax.fill_between(
        bx, lo, hi, color=ORANGE, alpha=0.15, linewidth=0, zorder=4, label="interquartile range"
    )
    ax.plot(bx, by, color=ORANGE, linewidth=2.4, zorder=5, label="median per bin")
    ax.axhline(0, color=INK_MUTED, linewidth=1, zorder=0)
    ax.set_ylim(-0.05, np.percentile(d["delta_lcoh_eur_per_kg"], 99.5))
    ax.set_xlabel("Small electrolyzer capacity removed [MW]")
    ax.set_ylabel("LCOH penalty [EUR/kg H2]")
    ax.set_title("The more small electrolyzer the first best builds, the more it costs to lose it")
    ax.legend(loc="upper left")
    save(fig, "05_penalty_vs_removed_capacity.png")


def fig_heatmap(df: pd.DataFrame) -> None:
    """Median penalty over the two strongest price drivers."""
    d = df[df["converged"]]
    x_col, x_label = "electricity_price_avg", "Average electricity price [EUR/MWh]"
    y_col, y_label = "hydrogen_import_price", "H2 import price [EUR/MWh]"
    if x_col not in d.columns or y_col not in d.columns:
        return

    # The scenario sample is not a full factorial, so many price combinations are thin.
    # Cells backed by fewer runs than this carry no signal and are left blank.
    min_runs = 8

    n = 6
    x_bins = pd.qcut(d[x_col], n, duplicates="drop")
    y_bins = pd.qcut(d[y_col], n, duplicates="drop")
    grouped = d.groupby([y_bins, x_bins], observed=True)["delta_lcoh_eur_per_kg"]
    grid = grouped.median().unstack()
    counts = grouped.size().unstack().reindex_like(grid)
    grid = grid.where(counts >= min_runs)

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    cmap.set_bad(GRID)
    im = ax.imshow(
        np.ma.masked_invalid(grid.to_numpy()), cmap=cmap, origin="lower", aspect="auto"
    )

    vmax = np.nanmax(grid.to_numpy())
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.to_numpy()[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j,
                i,
                f"{value:.3f}\nn={int(counts.to_numpy()[i, j])}",
                ha="center",
                va="center",
                fontsize=8,
                linespacing=1.4,
                color="#ffffff" if value > 0.55 * vmax else INK,
            )

    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xticklabels([f"{iv.mid:.0f}" for iv in grid.columns], fontsize=8)
    ax.set_yticklabels([f"{iv.mid:.0f}" for iv in grid.index], fontsize=8)
    ax.set_xlabel(x_label + "  (bin centre)")
    ax.set_ylabel(y_label + "  (bin centre)")
    ax.set_title("Median LCOH penalty [EUR/kg H2] by price regime", pad=26)
    ax.text(
        0.0,
        1.015,
        f"grey cells: fewer than {min_runs} runs in that price combination",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_2,
    )
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, label="Median LCOH penalty [EUR/kg H2]", shrink=0.85)
    save(fig, "06_penalty_heatmap_prices.png")


def fig_lcoh_levels_distribution(df: pd.DataFrame) -> None:
    """Where the LCOH itself sits, first best vs re-optimised."""
    d = df[df["converged"]]
    fig, ax = plt.subplots(figsize=(8, 4.6))

    # The two distributions nearly coincide, so overlaid histograms are unreadable;
    # cumulative curves show the shift cleanly.
    for column, color, label, anchor in (
        ("lcoh_first_best_eur_per_kg", BLUE, "first best", 0.22),
        ("lcoh_reopt_eur_per_kg", ORANGE, "no small electrolyzer", 0.80),
    ):
        values = d[column].sort_values()
        share = 100 * np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, share, color=color, label=label)
        ax.annotate(
            label,
            xy=(values.iloc[int(anchor * len(values))], 100 * anchor),
            xytext=(10, -6),
            textcoords="offset points",
            color=color,
            fontweight="bold",
            fontsize=9,
        )

    ax.set_ylim(0, 101)
    ax.set_xlabel("LCOH [EUR/kg H2]")
    ax.set_ylabel("Share of scenarios [%]")
    ax.set_title("LCOH levels barely shift when the small electrolyzer is removed")
    ax.legend(loc="lower right")
    save(fig, "07_lcoh_level_distributions.png")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    df = load_data()
    print(
        f"runs: {len(df)} | solved: {int(df['solved'].sum())} | "
        f"converged: {int(df['converged'].sum())}"
    )
    print(f"Writing figures to {FIG_DIR}")

    fig_distribution(df)
    fig_ecdf(df)
    fig_level_scatter(df)
    fig_drivers(df)
    fig_removed_capacity(df)
    fig_heatmap(df)
    fig_lcoh_levels_distribution(df)

    df.to_excel(os.path.join(FIG_DIR, "plot_data.xlsx"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()