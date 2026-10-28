"""
Do the scenarios with a large near-optimal penalty share a common input pattern?

IMPORTANT — the unit of analysis. The batch is a full cross of a small set of
uncertainty draws with the 40 node topologies: each draw of the ten uncertainty
parameters is re-solved on every topology. The ~2160 converged runs therefore carry
only ~66 independent parameter samples. Clustering or fitting rules at run level
would treat the same sample up to 40 times and manufacture significance, so all
statistics here are computed at DRAW level, with topology treated as a within-draw
replicate.

Three analyses:
  A. Replication structure — how much of the penalty spread is between draws versus
     between topologies within a draw.
  B. k-means over the 66 draws in standardised uncertainty space, with the penalty
     of each cluster. This is the "which input patterns go together" question.
  C. Univariate ranking (Spearman) of each parameter against the draw-level penalty,
     plus an explicit table of the draws that exceed the 5% threshold.

Writes figures and tables into <REOPT_DIR>/figures_clusters/.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

BASE = r"X:/3000_simulations_24_may_2026"
ORIG_DIR = os.path.join(BASE, "Snellius_3000_simulations")
REOPT_DIR = os.path.join(
    BASE, "reopt_comparison_Snellius_3000_simulations_small_electrolyzer_only"
)
COMPARISON_FILE = os.path.join(REOPT_DIR, "reopt_comparison_with_LCOH.xlsx")
OUT_DIR = os.path.join(REOPT_DIR, "figures_clusters")

HIGH_PENALTY_PCT = 5.0
CONVERGENCE_TOLERANCE_PCT = -0.1
K_RANGE = range(2, 9)
RANDOM_STATE = 0

# The ten sampled uncertainty parameters. These define a draw.
UNCERTAINTY = {
    "total_demand_TWh": "Total H2 demand [TWh]",
    "demand_level_ratio": "Demand level ratio",
    "unbalance_ratio": "Unbalance ratio",
    "import_availability_ratio": "Import availability ratio",
    "electricity_price_avg": "Electricity price avg [EUR/MWh]",
    "electricity_standard_dev": "Electricity price std [EUR/MWh]",
    "solar_cf_realized_mean": "Solar mean capacity factor",
    "electricity_availability_small": "El. availability small [MW]",
    "electricity_availability_large": "El. availability large [MW]",
    "hydrogen_import_price": "H2 import price [EUR/MWh]",
    # sampled since the v5 campaign (Sept 2026)
    "capex_ratio_small_big": "Small/large electrolyzer capex ratio",
    "pipeline_cost_multiplier": "Pipeline cost multiplier",
}

BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"
DIVERGING = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_red",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#8f2120"],
)


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
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(os.path.join(OUT_DIR, name))
    plt.close(fig)
    print(f"  {name}")


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def load_runs() -> pd.DataFrame:
    df = pd.read_excel(COMPARISON_FILE)
    params = pd.read_csv(os.path.join(ORIG_DIR, "parallel_results_summary.csv"), sep=";")
    df = df.merge(
        params[["run_id", "scenario"] + list(UNCERTAINTY)], on="run_id", how="left"
    )

    df = df[df["delta_cost_pct"] >= CONVERGENCE_TOLERANCE_PCT].copy()
    df = df.dropna(subset=list(UNCERTAINTY))

    # A draw is one distinct combination of the ten uncertainty parameters.
    df["draw_key"] = df[list(UNCERTAINTY)].round(6).astype(str).agg("|".join, axis=1)
    codes = {key: i + 1 for i, key in enumerate(sorted(df["draw_key"].unique()))}
    df["draw"] = df["draw_key"].map(codes)
    df["is_high"] = (df["delta_cost_pct"] > HIGH_PENALTY_PCT).astype(int)
    return df


def aggregate_draws(runs: pd.DataFrame) -> pd.DataFrame:
    agg = {c: (c, "first") for c in UNCERTAINTY}
    agg.update(
        n_topologies=("delta_cost_pct", "size"),
        penalty_median_pct=("delta_cost_pct", "median"),
        penalty_min_pct=("delta_cost_pct", "min"),
        penalty_max_pct=("delta_cost_pct", "max"),
        delta_lcoh_median_eur_per_kg=("delta_lcoh_eur_per_kg", "median"),
        share_topologies_high_pct=("is_high", lambda s: 100 * s.mean()),
    )
    draws = runs.groupby("draw").agg(**agg).reset_index()
    draws["is_high"] = (draws["penalty_median_pct"] > HIGH_PENALTY_PCT).astype(int)
    return draws


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def fig_replication(runs: pd.DataFrame, draws: pd.DataFrame) -> None:
    """Penalty per draw, showing the spread contributed by topology within each draw."""
    order = draws.sort_values("penalty_median_pct")["draw"].tolist()
    position = {d: i for i, d in enumerate(order)}

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.scatter(
        runs["draw"].map(position),
        runs["delta_cost_pct"],
        s=7,
        color=BLUE,
        alpha=0.30,
        linewidth=0,
        label="one run (draw x topology)",
    )
    medians = draws.set_index("draw").loc[order, "penalty_median_pct"]
    ax.plot(
        range(len(order)),
        medians.to_numpy(),
        color=ORANGE,
        linewidth=2.2,
        label="draw median over the 40 topologies",
    )
    ax.axhline(HIGH_PENALTY_PCT, color=RED, linestyle="--", linewidth=1.6)
    ax.annotate(
        f"{HIGH_PENALTY_PCT:g}% threshold",
        xy=(1, HIGH_PENALTY_PCT),
        xytext=(4, 6),
        textcoords="offset points",
        color=RED,
        fontsize=9,
    )
    n_high = int(draws["is_high"].sum())
    ax.annotate(
        f"only {n_high} of {len(draws)} draws sit above the threshold",
        xy=(len(order) - 1, HIGH_PENALTY_PCT),
        xytext=(-8, 22),
        textcoords="offset points",
        ha="right",
        color=RED,
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("Uncertainty draw, sorted by median penalty")
    ax.set_ylabel("Total system cost penalty [%]")
    ax.set_title("Each draw is re-solved on all 40 topologies; topology moves it little")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "08_replication_structure.png")


def fig_variance_share(runs: pd.DataFrame) -> None:
    """How much of the penalty variance is between draws versus within a draw."""
    overall = runs["delta_cost_pct"].mean()
    group_means = runs.groupby("draw")["delta_cost_pct"].transform("mean")
    between = ((group_means - overall) ** 2).sum()
    within = ((runs["delta_cost_pct"] - group_means) ** 2).sum()
    shares = 100 * np.array([between, within]) / (between + within)

    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    left = 0.0
    for value, color, label in zip(
        shares, (BLUE, ORANGE), ("between draws (uncertainty parameters)", "within a draw (topology)")
    ):
        ax.barh([0], [value], left=left, color=color, height=0.55)
        ax.annotate(
            f"{label}\n{value:.1f}%",
            xy=(left + value / 2, 0),
            ha="center",
            va="center",
            color="#ffffff" if value > 18 else INK_2,
            fontsize=9,
            fontweight="bold",
        )
        left += value + 0.4

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Share of total penalty variance [%]")
    ax.set_title("The uncertainty draw decides the penalty; topology barely matters")
    ax.grid(visible=False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(False)
    save(fig, "09_variance_decomposition.png")


def fig_k_selection(scores: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(scores["k"], scores["silhouette"], color=BLUE, marker="o", markersize=7)
    best = scores.loc[scores["silhouette"].idxmax()]
    ax.plot([best["k"]], [best["silhouette"]], "o", color=ORANGE, markersize=12, zorder=5)
    ax.annotate(
        f"k = {int(best['k'])}",
        xy=(best["k"], best["silhouette"]),
        xytext=(8, 6),
        textcoords="offset points",
        color=ORANGE,
        fontweight="bold",
    )
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Silhouette score")
    ax.set_ylim(0, max(0.5, scores["silhouette"].max() * 1.4))
    ax.set_title("Cluster-count selection over the 66 uncertainty draws")
    ax.annotate(
        "Every score is far below 0.25: the draws come from a space-filling\n"
        "Latin hypercube, so there are no natural clusters to find.\n"
        "The partition below is a convenience grouping, not structure in the data.",
        xy=(0.03, 0.95),
        xycoords="axes fraction",
        va="top",
        fontsize=9,
        color=RED,
        linespacing=1.5,
    )
    save(fig, "10_cluster_k_selection.png")


def fig_cluster_penalty(summary: pd.DataFrame) -> None:
    """Penalty of each draw cluster, with the individual draws overlaid."""
    s = summary.sort_values("median_penalty_pct")
    fig, ax = plt.subplots(figsize=(8.8, 0.62 * len(s) + 2.4))

    labels = [f"{c}  (n={int(n)} draws)" for c, n in zip(s["cluster"], s["n_draws"])]
    colors = [RED if v > 0 else BLUE for v in s["n_high_draws"]]
    ax.barh(labels, s["median_penalty_pct"], color=colors, height=0.62)
    for y, (value, n_high) in enumerate(zip(s["median_penalty_pct"], s["n_high_draws"])):
        text = f"{value:.2f}%"
        if n_high:
            text += f"   contains {int(n_high)} of the {int(summary['n_high_draws'].sum())} >5% draws"
        ax.annotate(
            text,
            xy=(value, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=RED if n_high else INK_2,
            fontweight="bold" if n_high else "normal",
        )
    ax.set_xlim(0, s["median_penalty_pct"].max() * 1.9)
    ax.set_xlabel("Median cost penalty of the cluster [%]")
    ax.set_title("Median penalty by cluster of uncertainty draws")
    ax.grid(axis="y", visible=False)
    save(fig, "11_cluster_penalty.png")


def fig_cluster_profiles(draws: pd.DataFrame, order: list) -> None:
    """Cluster centroids as z-scores over the 66 draws."""
    scaler = StandardScaler().fit(draws[list(UNCERTAINTY)])
    z = pd.DataFrame(
        scaler.transform(draws[list(UNCERTAINTY)]),
        columns=list(UNCERTAINTY),
        index=draws.index,
    )
    z["cluster"] = draws["cluster"]
    centroids = z.groupby("cluster").mean().loc[order]

    fig, ax = plt.subplots(figsize=(1.25 * len(centroids) + 6.4, 5.6))
    limit = float(np.abs(centroids.to_numpy()).max())
    im = ax.imshow(
        centroids.T.to_numpy(), cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto"
    )
    for i in range(centroids.shape[1]):
        for j in range(centroids.shape[0]):
            value = centroids.to_numpy()[j, i]
            ax.text(
                j,
                i,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if abs(value) > 0.62 * limit else INK,
            )
    ax.set_xticks(range(centroids.shape[0]))
    ax.set_xticklabels(centroids.index, fontsize=9)
    ax.set_yticks(range(len(UNCERTAINTY)))
    ax.set_yticklabels(list(UNCERTAINTY.values()), fontsize=9)
    ax.set_title("Input profile of each draw cluster (ordered by median penalty)", pad=14)
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, label="Standard deviations across the 66 draws", shrink=0.85)
    save(fig, "12_cluster_profiles.png")


def fig_spearman(draws: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation of each parameter with the draw-level penalty."""
    rows = []
    for column, label in UNCERTAINTY.items():
        rho, p = spearmanr(draws[column], draws["penalty_median_pct"])
        rows.append({"parameter": label, "column": column, "spearman_rho": rho, "p_value": p})
    corr = pd.DataFrame(rows).sort_values("spearman_rho")

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    colors = [
        RED if (r < 0 and p < 0.05) else BLUE if p < 0.05 else INK_MUTED
        for r, p in zip(corr["spearman_rho"], corr["p_value"])
    ]
    ax.barh(corr["parameter"], corr["spearman_rho"], color=colors, height=0.62)
    ax.axvline(0, color=INK_2, linewidth=1.2)
    for y, (rho, p) in enumerate(zip(corr["spearman_rho"], corr["p_value"])):
        ax.annotate(
            f"{rho:+.2f}" + ("*" if p < 0.05 else ""),
            xy=(rho, y),
            xytext=(6 if rho >= 0 else -6, 0),
            textcoords="offset points",
            va="center",
            ha="left" if rho >= 0 else "right",
            fontsize=9,
            color=INK_2,
        )
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Spearman rho against the draw median penalty   (* p < 0.05, n = 66)")
    ax.set_title("Large-node power availability is the strongest monotone driver")
    ax.text(
        0.0,
        1.045,
        "The electricity price ranks weakly here because it acts as a threshold, not a gradient — see figure 15",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_2,
    )
    ax.grid(axis="y", visible=False)
    save(fig, "13_spearman_ranking.png")
    return corr


def fig_pattern_scatter(draws: pd.DataFrame) -> None:
    """The hypothesised pattern: cheap power plus costly imports, at draw level."""
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    low = draws[draws["is_high"] == 0]
    high = draws[draws["is_high"] == 1]

    ax.scatter(
        low["electricity_price_avg"],
        low["hydrogen_import_price"],
        s=60,
        color=BLUE,
        alpha=0.55,
        linewidth=0,
        label="draw with median penalty 0-5%",
    )
    ax.scatter(
        high["electricity_price_avg"],
        high["hydrogen_import_price"],
        s=180,
        color=RED,
        marker="D",
        linewidth=1.2,
        edgecolor=SURFACE,
        label="draw with median penalty > 5%",
        zorder=5,
    )
    for _, row in high.iterrows():
        ax.annotate(
            f"draw {int(row['draw'])}\n{row['penalty_median_pct']:.1f}%",
            xy=(row["electricity_price_avg"], row["hydrogen_import_price"]),
            xytext=(12, -4),
            textcoords="offset points",
            color=RED,
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel(UNCERTAINTY["electricity_price_avg"])
    ax.set_ylabel(UNCERTAINTY["hydrogen_import_price"])
    ax.set_title("All four high-penalty draws sit at the cheapest electricity prices")
    ax.legend(loc="upper right")
    save(fig, "14_pattern_scatter_draws.png")


def fig_gate(draws: pd.DataFrame, price_gate: float) -> pd.DataFrame:
    """The two-stage condition: a cheap-electricity gate, then local power at the small nodes.

    Only the draws that pass the price gate are plotted, because no draw above the gate
    reaches the threshold at all.
    """
    cheap = draws[draws["electricity_price_avg"] <= price_gate].sort_values(
        "penalty_median_pct"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # Left: the gate itself.
    ax = axes[0]
    ax.scatter(
        draws["electricity_price_avg"],
        draws["penalty_median_pct"],
        s=55,
        color=BLUE,
        alpha=0.55,
        linewidth=0,
    )
    ax.axvline(price_gate, color=RED, linestyle="--", linewidth=1.6)
    ax.axhline(HIGH_PENALTY_PCT, color=INK_MUTED, linestyle=":", linewidth=1.4)
    ax.annotate(
        f"no draw above {price_gate:g} EUR/MWh\nreaches {HIGH_PENALTY_PCT:g}%",
        xy=(price_gate, draws["penalty_median_pct"].max()),
        xytext=(12, -6),
        textcoords="offset points",
        color=RED,
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel(UNCERTAINTY["electricity_price_avg"])
    ax.set_ylabel("Draw median penalty [%]")
    ax.set_title(f"Stage 1: cheap-power gate (all {len(draws)} draws)")

    # Right: what separates the draws that pass the gate.
    ax = axes[1]
    colors = [RED if v else BLUE for v in cheap["is_high"]]
    ax.scatter(
        cheap["electricity_availability_small"],
        cheap["penalty_median_pct"],
        s=150,
        color=colors,
        linewidth=1.0,
        edgecolor=SURFACE,
        zorder=5,
    )
    for _, row in cheap.iterrows():
        ax.annotate(
            f"draw {int(row['draw'])}",
            xy=(row["electricity_availability_small"], row["penalty_median_pct"]),
            xytext=(9, -3),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
    ax.axhline(HIGH_PENALTY_PCT, color=INK_MUTED, linestyle=":", linewidth=1.4)
    ax.set_xlabel(UNCERTAINTY["electricity_availability_small"])
    ax.set_ylabel("Draw median penalty [%]")
    ax.set_title(f"Stage 2: only the {len(cheap)} draws that pass the gate")
    ax.annotate(
        f"{int(cheap['is_high'].sum())} of {len(cheap)} pass both conditions\n"
        "evidence base for the whole pattern",
        xy=(0.03, 0.95),
        xycoords="axes fraction",
        va="top",
        fontsize=9,
        color=RED,
        fontweight="bold",
    )

    fig.suptitle(
        "The pattern is a two-stage condition, resting on very few draws",
        fontsize=13,
        fontweight="bold",
        color=INK,
        x=0.065,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "15_two_stage_condition.png")
    return cheap


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    set_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    runs = load_runs()
    draws = aggregate_draws(runs)
    print(
        f"converged runs: {len(runs)} | distinct uncertainty draws: {len(draws)} | "
        f"topologies: {runs['scenario'].nunique()}"
    )
    print(
        f"runs with penalty > {HIGH_PENALTY_PCT}%: {int(runs['is_high'].sum())} "
        f"-> but only {int(draws['is_high'].sum())} distinct draws"
    )

    x = StandardScaler().fit_transform(draws[list(UNCERTAINTY)])
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit(x)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(x, km.labels_)})
    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    print(f"best k over draws: {best_k} (silhouette {scores['silhouette'].max():.3f})")

    labels = KMeans(n_clusters=best_k, n_init=50, random_state=RANDOM_STATE).fit_predict(x)
    draws["cluster"] = ["C" + str(i + 1) for i in labels]

    summary = (
        draws.groupby("cluster")
        .agg(
            n_draws=("is_high", "size"),
            n_high_draws=("is_high", "sum"),
            median_penalty_pct=("penalty_median_pct", "median"),
            max_penalty_pct=("penalty_median_pct", "max"),
            median_delta_lcoh_eur_per_kg=("delta_lcoh_median_eur_per_kg", "median"),
        )
        .reset_index()
    )
    for column in UNCERTAINTY:
        summary[f"median_{column}"] = draws.groupby("cluster")[column].median().to_numpy()

    print(f"Writing to {OUT_DIR}")
    fig_replication(runs, draws)
    fig_variance_share(runs)
    fig_k_selection(scores)
    fig_cluster_penalty(summary)
    order = summary.sort_values("median_penalty_pct", ascending=False)["cluster"].tolist()
    fig_cluster_profiles(draws, order)
    corr = fig_spearman(draws)
    fig_pattern_scatter(draws)

    # The gate is the highest electricity price among the draws above the threshold,
    # rounded up, rather than a value tuned for separation.
    price_gate = float(np.ceil(draws.loc[draws["is_high"] == 1, "electricity_price_avg"].max()))
    cheap = fig_gate(draws, price_gate)
    print(
        f"\nPrice gate at {price_gate:g} EUR/MWh: {len(cheap)} draws pass, "
        f"{int(cheap['is_high'].sum())} of them exceed {HIGH_PENALTY_PCT:g}%"
    )
    print(
        cheap[
            [
                "draw",
                "cluster",
                "penalty_median_pct",
                "electricity_price_avg",
                "electricity_availability_small",
                "electricity_availability_large",
                "import_availability_ratio",
                "hydrogen_import_price",
            ]
        ]
        .sort_values("penalty_median_pct", ascending=False)
        .round(2)
        .to_string(index=False)
    )

    print("\nCluster summary (draw level):")
    print(
        summary.sort_values("median_penalty_pct", ascending=False)[
            [
                "cluster",
                "n_draws",
                "n_high_draws",
                "median_penalty_pct",
                "max_penalty_pct",
                "median_electricity_price_avg",
                "median_hydrogen_import_price",
                "median_solar_cf_realized_mean",
            ]
        ].to_string(index=False)
    )

    print("\nThe draws above the threshold:")
    print(
        draws[draws["is_high"] == 1][
            ["draw", "cluster", "penalty_median_pct", "penalty_min_pct", "penalty_max_pct"]
            + list(UNCERTAINTY)
        ].to_string(index=False)
    )

    print("\nSpearman ranking against the draw median penalty:")
    print(corr[["parameter", "spearman_rho", "p_value"]].to_string(index=False))

    output_path = os.path.join(OUT_DIR, "cluster_analysis.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.sort_values("median_penalty_pct", ascending=False).to_excel(
            writer, sheet_name="clusters", index=False
        )
        draws.to_excel(writer, sheet_name="draws", index=False)
        corr.to_excel(writer, sheet_name="spearman", index=False)
        runs.drop(columns=["draw_key"]).to_excel(
            writer, sheet_name="runs", index=False
        )
    print(f"\nWritten: {output_path}")


if __name__ == "__main__":
    main()