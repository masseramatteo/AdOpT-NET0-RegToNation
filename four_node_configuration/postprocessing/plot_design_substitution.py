"""
Figures for the design substitution: what the system builds instead of the small
electrolyzer, in the four uncertainty draws whose cost penalty exceeds 5%.

Reads design_substitution.xlsx (from extract_design_substitution.py). Every quantity is
aggregated over the 40 topologies of a draw, so each bar is a draw-level median and the
spread across topologies is shown separately.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = r"X:/3000_simulations_24_may_2026"
REOPT_DIR = os.path.join(
    BASE, "reopt_comparison_Snellius_3000_simulations_small_electrolyzer_only"
)
OUT_DIR = os.path.join(REOPT_DIR, "figures_design")

# Penalty band to plot; the extraction script writes one file per band.
PENALTY_BAND = (5.0, float("inf"))


def band_tag(band: tuple[float, float]) -> str:
    low, high = band
    return f"{low:g}plus" if high == float("inf") else f"{low:g}to{high:g}"


def data_file(band: tuple[float, float]) -> str:
    return os.path.join(OUT_DIR, f"design_substitution_{band_tag(band)}.xlsx")

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"
VIOLET = "#4a3aa7"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

MILLION = 1e6


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


def sum_columns(df: pd.DataFrame, prefix: str, contains: list[str]) -> pd.Series:
    """Sum every column that starts with prefix and contains all the given fragments."""
    cols = [
        c
        for c in df.columns
        if c.startswith(prefix) and all(fragment in c for fragment in contains)
    ]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].sum(axis=1)


def build_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the raw per-node columns into the handful of quantities that matter."""
    q = df[["run_id", "draw", "scenario", "delta_cost_pct", "delta_lcoh_eur_per_kg"]].copy()

    for tag, prefix in (("fb", "fb__"), ("ro", "ro__")):
        q[f"{tag}_elec_big_MW"] = sum_columns(df, prefix, ["size|Electrolyzer_big|"])
        q[f"{tag}_elec_small_MW"] = sum_columns(df, prefix, ["size|Electrolyzer_small|"])
        q[f"{tag}_pv_MW"] = sum_columns(df, prefix, ["size|Photovoltaic|"])
        q[f"{tag}_stor_highP_MWh"] = sum_columns(df, prefix, ["size|Storage_H2_highP|"])
        q[f"{tag}_stor_lowP_MWh"] = sum_columns(df, prefix, ["size|Storage_H2_lowP|"])
        q[f"{tag}_stor_cavern_MWh"] = sum_columns(df, prefix, ["size|Storage_H2_Cavern"])

        q[f"{tag}_pipe_MW"] = sum_columns(df, prefix, ["netw_size|"])
        q[f"{tag}_pipe_small_large_MW"] = sum_columns(
            df, prefix, ["netw_size|", "small_large"]
        )
        q[f"{tag}_pipe_large_large_MW"] = sum_columns(
            df, prefix, ["netw_size|", "large_large"]
        )
        q[f"{tag}_pipe_small_small_MW"] = sum_columns(
            df, prefix, ["netw_size|", "small_small"]
        )
        q[f"{tag}_pipe_flow_TWh"] = sum_columns(df, prefix, ["netw_flow|"]) / 1e6

        q[f"{tag}_h2_import_TWh"] = (
            sum_columns(df, prefix, ["annual|hydrogen_import|"]) / 1e6
        )
        q[f"{tag}_el_import_large_TWh"] = (
            sum_columns(df, prefix, ["annual|electricity_import|Large"]) / 1e6
        )
        q[f"{tag}_el_import_small_TWh"] = (
            sum_columns(df, prefix, ["annual|electricity_import|Small"]) / 1e6
        )
        q[f"{tag}_h2_from_big_TWh"] = (
            sum_columns(df, prefix, ["annual|h2_out|Electrolyzer_big|"]) / 1e6
        )
        q[f"{tag}_h2_from_small_TWh"] = (
            sum_columns(df, prefix, ["annual|h2_out|Electrolyzer_small|"]) / 1e6
        )

        for cost in (
            "cost_capex_tecs",
            "cost_capex_netws",
            "cost_opex_tecs",
            "cost_opex_netws",
            "cost_imports",
        ):
            q[f"{tag}_{cost}"] = df[f"{prefix}{cost}"]

    for column in [c for c in q.columns if c.startswith("fb_")]:
        key = column[3:]
        q[f"d_{key}"] = q[f"ro_{key}"] - q[column]
    return q


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def fig_cost_decomposition(q: pd.DataFrame, draws: list) -> None:
    """Where the extra money goes, per draw."""
    components = [
        ("d_cost_capex_tecs", "CAPEX technologies", BLUE),
        ("d_cost_capex_netws", "CAPEX pipelines", ORANGE),
        ("d_cost_opex_tecs", "OPEX technologies", AQUA),
        ("d_cost_opex_netws", "OPEX pipelines", YELLOW),
        ("d_cost_imports", "Import cost", VIOLET),
    ]

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    width = 0.62
    x = np.arange(len(draws))
    pos_base = np.zeros(len(draws))
    neg_base = np.zeros(len(draws))

    for column, label, color in components:
        values = np.array(
            [q.loc[q["draw"] == d, column].median() / MILLION for d in draws]
        )
        base = np.where(values >= 0, pos_base, neg_base)
        ax.bar(x, values, width, bottom=base, color=color, label=label)
        pos_base = pos_base + np.where(values >= 0, values, 0)
        neg_base = neg_base + np.where(values < 0, values, 0)

    total = np.array(
        [
            sum(q.loc[q["draw"] == d, c].median() for c, _, _ in components) / MILLION
            for d in draws
        ]
    )
    ax.plot(x, total, "D", color=INK, markersize=9, zorder=6, label="net change")
    for xi, value in zip(x, total):
        ax.annotate(
            f"{value:+,.0f} M EUR",
            xy=(xi, value),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=9,
            color=INK,
        )

    ax.axhline(0, color=INK_2, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"draw {int(d)}\n(+{q.loc[q['draw'] == d, 'delta_cost_pct'].median():.1f}%)"
            for d in draws
        ]
    )
    ax.set_ylabel("Change in NPV component [million EUR]")
    ax.set_title("How the extra cost is paid when the small electrolyzer is removed", pad=34)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        fontsize=9,
        handlelength=1.4,
        columnspacing=1.2,
    )
    ax.grid(axis="x", visible=False)
    save(fig, "16_cost_decomposition.png")


def fig_capacity_substitution(q: pd.DataFrame, draws: list) -> None:
    """Which capacities grow to replace the removed electrolyzer."""
    items = [
        ("d_elec_small_MW", "Electrolyzer small [MW]"),
        ("d_elec_big_MW", "Electrolyzer big [MW]"),
        ("d_pipe_small_large_MW", "Pipeline small-large [MW]"),
        ("d_pipe_large_large_MW", "Pipeline large-large [MW]"),
        ("d_stor_lowP_MWh", "Storage low P [MWh]"),
        ("d_stor_highP_MWh", "Storage high P [MWh]"),
        ("d_pv_MW", "Photovoltaic [MW]"),
    ]

    fig, axes = plt.subplots(1, len(draws), figsize=(3.5 * len(draws) + 1.5, 5.6), sharey=True)
    axes = np.atleast_1d(axes)
    labels = [label for _, label in items]
    y = np.arange(len(items))

    for ax, draw in zip(axes, draws):
        sub = q[q["draw"] == draw]
        values = [sub[column].median() for column, _ in items]
        colors = [RED if v < 0 else BLUE for v in values]
        ax.barh(y, values, color=colors, height=0.62)
        ax.axvline(0, color=INK_2, linewidth=1.2)
        for yi, value in zip(y, values):
            if abs(value) < 1e-6:
                continue
            ax.annotate(
                f"{value:+,.0f}",
                xy=(value, yi),
                xytext=(6 if value >= 0 else -6, 0),
                textcoords="offset points",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
                color=INK_2,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(
            f"draw {int(draw)}  (+{sub['delta_cost_pct'].median():.1f}%)", fontsize=11
        )
        ax.set_xlabel("Median change over 40 topologies")
        ax.grid(axis="y", visible=False)

    fig.suptitle(
        "What is built instead", fontsize=13, fontweight="bold", color=INK, x=0.06, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "17_capacity_substitution.png")


def fig_supply_mix(q: pd.DataFrame, draws: list) -> None:
    """Where the hydrogen comes from, before and after."""
    sources = [
        ("h2_from_small_TWh", "Small electrolysis", ORANGE),
        ("h2_from_big_TWh", "Big electrolysis", BLUE),
        ("h2_import_TWh", "H2 import", AQUA),
    ]

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    width = 0.36
    x = np.arange(len(draws))

    offsets = []
    for offset, tag, hatch in (
        (-width / 2 - 0.02, "fb", None),
        (+width / 2 + 0.02, "ro", "//"),
    ):
        offsets.append(offset)
        base = np.zeros(len(draws))
        for key, label, color in sources:
            values = np.array(
                [q.loc[q["draw"] == d, f"{tag}_{key}"].median() for d in draws]
            )
            ax.bar(
                x + offset,
                values,
                width,
                bottom=base,
                color=color,
                hatch=hatch,
                edgecolor=SURFACE,
                linewidth=1.2,
                label=label if tag == "fb" else None,
            )
            base = base + values

    # Which bar is which goes on the x axis, so nothing has to be written over the bars.
    ax.set_xticks(np.concatenate([x + offsets[0], x + offsets[1]]))
    ax.set_xticklabels(
        ["first\nbest"] * len(draws) + ["no small\nelectrolyzer"] * len(draws), fontsize=8
    )
    for xi, draw in zip(x, draws):
        ax.annotate(
            f"draw {int(draw)}",
            xy=(xi, -0.14),
            xycoords=("data", "axes fraction"),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )
    ax.set_ylabel("Annual hydrogen supply [TWh]")
    ax.set_title("Hydrogen supply mix before and after removing the small electrolyzer")
    ax.legend(loc="upper left", ncol=3)
    ax.grid(axis="x", visible=False)
    save(fig, "18_supply_mix.png")


def fig_topology_spread(q: pd.DataFrame, draws: list) -> None:
    """Is the substitution the same on every topology within a draw?"""
    keys = [
        ("d_elec_big_MW", "Electrolyzer big [MW]"),
        ("d_h2_import_TWh", "H2 import [TWh]"),
        ("d_pipe_small_large_MW", "Pipeline small-large [MW]"),
    ]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.8))

    for ax, (column, label) in zip(axes, keys):
        data = [q.loc[q["draw"] == d, column].to_numpy() for d in draws]
        parts = ax.boxplot(
            data,
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color=ORANGE, linewidth=2),
            flierprops=dict(
                marker="o", markersize=3, markerfacecolor=INK_MUTED, markeredgecolor="none"
            ),
        )
        for patch in parts["boxes"]:
            patch.set_facecolor(BLUE)
            patch.set_alpha(0.30)
            patch.set_edgecolor(BLUE)
        for whisker in parts["whiskers"] + parts["caps"]:
            whisker.set_color(BLUE)
        ax.axhline(0, color=INK_2, linewidth=1.1)
        ax.set_xticklabels([f"draw {int(d)}" for d in draws], fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Change, one point per topology")
    fig.suptitle(
        "Spread of the substitution across the 40 topologies of each draw",
        fontsize=13,
        fontweight="bold",
        color=INK,
        x=0.06,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "19_topology_spread.png")


def fig_band_comparison() -> None:
    """Is the substitution mechanism the same in the 1-5% band as above 5%?

    Everything is normalised by the hydrogen the small electrolyzer used to produce, so
    draws of very different size can be put on the same axis.
    """
    bands = [((1.0, 5.0), "penalty 1-5%", BLUE), ((5.0, float("inf")), "penalty > 5%", ORANGE)]
    frames = []
    for band, label, _ in bands:
        path = data_file(band)
        if not os.path.exists(path):
            print(f"  (skipping comparison: {os.path.basename(path)} not found)")
            return
        q = build_quantities(pd.read_excel(path, sheet_name="pairs"))
        q["band"] = label
        frames.append(q)
    data = pd.concat(frames, ignore_index=True)

    lost = data["fb_h2_from_small_TWh"].replace(0, np.nan)
    data["share_by_import"] = 100 * data["d_h2_import_TWh"] / lost
    data["share_by_big"] = 100 * data["d_h2_from_big_TWh"] / lost
    data["share_demand_drop"] = 100 * (
        -(data["d_h2_import_TWh"] + data["d_h2_from_big_TWh"]) / lost + 1
    )
    data["orphaned_el_TWh"] = -data["d_el_import_small_TWh"]

    metrics = [
        ("share_by_import", "Lost H2 replaced\nby import [%]"),
        ("share_by_big", "Lost H2 replaced by\nbig electrolysis [%]"),
        ("orphaned_el_TWh", "Cheap electricity left\nunused at small nodes [TWh]"),
        ("d_pipe_small_large_MW", "Extra pipeline\nsmall-large [MW]"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.8))
    for ax, (column, label) in zip(axes, metrics):
        groups = [data.loc[data["band"] == lab, column].dropna() for _, lab, _ in bands]
        parts = ax.boxplot(
            groups,
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color=INK, linewidth=2),
            flierprops=dict(
                marker="o", markersize=3, markerfacecolor=INK_MUTED, markeredgecolor="none"
            ),
        )
        for patch, (_, _, color) in zip(parts["boxes"], bands):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for i, group in enumerate(groups, start=1):
            ax.annotate(
                f"med {group.median():,.0f}",
                xy=(i, group.median()),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=INK,
            )
        ax.axhline(0, color=INK_2, linewidth=1.1)
        ax.set_xticklabels([lab.replace("penalty ", "") for _, lab, _ in bands], fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        "The substitution route itself changes: electrolysis below 5%, imports above",
        fontsize=13,
        fontweight="bold",
        color=INK,
        x=0.06,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "20_band_comparison.png")


def fig_spare_capacity() -> None:
    """The penalty against the spare electricity capacity at the large nodes.

    One point per draw, pooling both bands, because the band boundary turns out not to be
    the thing that separates the two substitution routes.
    """
    cluster_file = os.path.join(REOPT_DIR, "figures_clusters", "cluster_analysis.xlsx")
    draws_info = pd.read_excel(cluster_file, sheet_name="draws").set_index("draw")

    rows = []
    for band in ((1.0, 5.0), (5.0, float("inf"))):
        path = data_file(band)
        if not os.path.exists(path):
            return
        q = build_quantities(pd.read_excel(path, sheet_name="pairs"))
        for draw in sorted(q["draw"].unique()):
            sub = q[q["draw"] == draw]
            cap = draws_info.loc[draw, "electricity_availability_large"] * 2 * 8760 / 1e6
            lost = sub["fb_h2_from_small_TWh"].median()
            rows.append(
                {
                    "draw": draw,
                    "penalty_pct": sub["delta_cost_pct"].median(),
                    "spare_TWh": cap - sub["fb_el_import_large_TWh"].median(),
                    "share_import": 100 * sub["d_h2_import_TWh"].median() / lost,
                }
            )
    t = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    scatter = ax.scatter(
        t["spare_TWh"],
        t["penalty_pct"],
        c=t["share_import"],
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "b2r", ["#2a78d6", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948"]
        ),
        s=190,
        linewidth=1.0,
        edgecolor=SURFACE,
        vmin=0,
        vmax=100,
        zorder=5,
    )
    for _, row in t.iterrows():
        ax.annotate(
            f"{int(row['draw'])}",
            xy=(row["spare_TWh"], row["penalty_pct"]),
            xytext=(13, -4),
            textcoords="offset points",
            fontsize=9,
            color=INK_2,
        )
    ax.axhline(5, color=INK_MUTED, linestyle=":", linewidth=1.4)
    ax.annotate(
        "5% band boundary",
        xy=(t["spare_TWh"].max(), 5),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK_MUTED,
    )
    ax.set_xlabel("Spare electricity capacity at the large nodes, first best [TWh/year]")
    ax.set_ylabel("Cost penalty [%]")
    ax.set_title("Spare capacity at the large nodes sets both the route and the price")
    fig.colorbar(scatter, ax=ax, label="Share of lost H2 replaced by import [%]", shrink=0.85)
    save(fig, "21_spare_capacity.png")


def main(band: tuple[float, float] = PENALTY_BAND) -> None:
    set_style()
    path = data_file(band)
    df = pd.read_excel(path, sheet_name="pairs")
    q = build_quantities(df)
    draws = sorted(q["draw"].unique())
    tag = band_tag(band)
    print(f"band {tag}: draws {draws} | run pairs: {len(q)}")

    global save
    original_save = save

    def save_with_tag(fig, name):  # per-band files, except the cross-band figures
        cross_band = name.startswith(("20_", "21_"))
        original_save(fig, name if cross_band else f"{tag}_{name}")

    save = save_with_tag
    try:
        fig_cost_decomposition(q, draws)
        fig_capacity_substitution(q, draws)
        fig_supply_mix(q, draws)
        fig_topology_spread(q, draws)
        fig_band_comparison()
        fig_spare_capacity()
    finally:
        save = original_save

    summary = q.groupby("draw").median(numeric_only=True).T
    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        q.to_excel(writer, sheet_name="quantities", index=False)
        summary.to_excel(writer, sheet_name="median_by_draw")
    print(f"Updated: {path}")

    show = [
        "delta_cost_pct",
        "fb_elec_small_MW",
        "d_elec_big_MW",
        "d_h2_import_TWh",
        "d_el_import_large_TWh",
        "d_el_import_small_TWh",
        "d_pipe_small_large_MW",
        "d_stor_highP_MWh",
        "d_cost_capex_tecs",
        "d_cost_imports",
    ]
    print("\nMedian by draw:")
    print(summary.loc[[r for r in show if r in summary.index]].round(2).to_string())


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        main((float(sys.argv[1]), float(sys.argv[2])))
    else:
        main()