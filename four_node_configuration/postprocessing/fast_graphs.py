r"""
Fast graphs / quick analysis on the 3000-simulation extracted results.

Loads the aggregated Excel produced on Snellius into a pandas DataFrame and
provides duration-curve / stacked-area plots of the hydrogen supply mix.

Network drive X: maps to
    \\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\
    AdOpT-NET0-RegToNation\four_node_configuration\results
Always use X: to avoid UNC-path issues.

Units note: all h5 operational sums are over 20 typical days (480 h) in MWh.
Scale to annual TWh with (365/20)/1e6  ->  see _annual_twh().

Plot catalogue (call any with df; pass save=Path(...) to write a PNG):
    supply_stack(df)                     absolute TWh, demand line + stacked sources
    demand_vs_component(df, which)       absolute TWh, demand line + one source
    share_curve(df, which)               duration curve, one source % of demand
    combined_share_curves(df)            3 duration curves overlaid (indep. sort)
    local_self_sufficiency_curve(df)     small production / small demand (%)
    supply_stack_pct(df)                 per-run stacked % of demand, sorted by demand
    supply_duration_pct(df, sort_by)     stacked % duration curve, sorted by a share
`which` / `sort_by` in {"centralized", "local", "import"}.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
RESULTS_ROOT = Path(r"X:\3000_simulations_24_may_2026\Snellius_3000_simulations")
INPUT_XLSX = RESULTS_ROOT / "extracted_results.xlsx"

# --------------------------------------------------------------------------- #
# Supply-source definitions
# --------------------------------------------------------------------------- #
# component name -> h5 columns (MWh over 20 typical days) that make it up
COMPONENTS = {
    "centralized": [  # large-cluster (big electrolyzer) production
        "Large_cluster1_Electrolyzer_big_H2_output_sum",
        "Large_cluster2_Electrolyzer_big_H2_output_sum",
    ],
    "local": [  # small-cluster (small electrolyzer) production
        "Small_cluster1_Electrolyzer_small_H2_output_sum",
        "Small_cluster2_Electrolyzer_small_H2_output_sum",
    ],
    "import": [  # all imported hydrogen (small-cluster import is ~0)
        "Large_cluster1_hydrogen_import_sum",
        "Large_cluster2_hydrogen_import_sum",
        "Small_cluster1_hydrogen_import_sum",
        "Small_cluster2_hydrogen_import_sum",
    ],
}

_LABELS = {
    "centralized": "Centralized production (large clusters)",
    "local": "Local production (small clusters)",
    "import": "Total imported hydrogen",
}
_COLORS = {"centralized": "#1f77b4", "local": "#2ca02c", "import": "#ff7f0e"}


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def load_results(path: Path = INPUT_XLSX, sheet: str = "Results") -> pd.DataFrame:
    """Read the extracted results Excel into a DataFrame."""
    df = pd.read_excel(path, sheet_name=sheet)
    print(f"Loaded {len(df)} runs x {df.shape[1]} columns from {path}")
    return df


def _annual_twh(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Sum given columns (MWh over 20 typical days) -> annual TWh."""
    to_twh = (365 / 20) / 1e6
    total = df[cols[0]].copy()
    for c in cols[1:]:
        total = total + df[c]
    return total * to_twh


def decentralization(df: pd.DataFrame, mode: str = "production") -> pd.Series:
    """
    Decentralization level (%).
        mode="production" -> local / (local + centralized)  = production mix
        mode="supply"     -> local / (local + centralized + import) = self-supply
    Values in %.
    """
    loc = _annual_twh(df, COMPONENTS["local"])
    cen = _annual_twh(df, COMPONENTS["centralized"])
    if mode == "production":
        return loc / (loc + cen) * 100
    elif mode == "supply":
        imp = _annual_twh(df, COMPONENTS["import"])
        return loc / (loc + cen + imp) * 100
    raise ValueError("mode must be 'production' or 'supply'")


def small_cluster_demand_twh(df: pd.DataFrame) -> pd.Series:
    """
    Small-cluster (both nodes) H2 demand from run params:
        small_demand = total_demand / (1 + demand_level_ratio)
    (see Change_data_excel.py). Returns TWh/yr.
    """
    return df["total_demand_TWh"] / (1 + df["demand_level_ratio"])


def _finish(fig, save):
    """Save PNG if a path is given, else show interactively."""
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
        print(f"Wrote {save}")
    else:
        plt.show()
    return fig


# --------------------------------------------------------------------------- #
# Absolute plots (TWh/yr) — demand line + sources
# --------------------------------------------------------------------------- #
def supply_stack(df: pd.DataFrame, save: Path | None = None):
    """
    Demand line (TWh/yr) + stacked supply sources underneath, runs sorted by
    total demand. Shows how demand is met across the ensemble.
    """
    centralized = _annual_twh(df, COMPONENTS["centralized"])
    local = _annual_twh(df, COMPONENTS["local"])
    imp = _annual_twh(df, COMPONENTS["import"])
    demand = df["total_demand_TWh"]

    order = demand.sort_values().index
    x = range(len(order))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(
        x,
        centralized[order],
        local[order],
        imp[order],
        labels=[_LABELS["centralized"], _LABELS["local"], _LABELS["import"]],
        colors=[_COLORS["centralized"], _COLORS["local"], _COLORS["import"]],
        alpha=0.9,
    )
    ax.plot(x, demand[order], color="black", lw=1.8, label="Total demand")

    ax.set_xlabel("Runs (sorted by total demand)")
    ax.set_ylabel("Hydrogen [TWh/yr]")
    ax.set_title("How H2 demand is met across runs")
    ax.set_xlim(0, len(order) - 1)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    return _finish(fig, save)


def demand_vs_component(df: pd.DataFrame, which: str, save: Path | None = None):
    """Demand line + one supply component (filled area, TWh/yr), sorted by demand."""
    comp = _annual_twh(df, COMPONENTS[which])
    demand = df["total_demand_TWh"]

    order = demand.sort_values().index
    x = range(len(order))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(x, comp[order], color=_COLORS[which], alpha=0.85, label=_LABELS[which])
    ax.plot(x, demand[order], color="black", lw=1.8, label="Total demand")

    ax.set_xlabel("Runs (sorted by total demand)")
    ax.set_ylabel("Hydrogen [TWh/yr]")
    ax.set_title(f"Total demand vs {_LABELS[which].lower()}")
    ax.set_xlim(0, len(order) - 1)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    return _finish(fig, save)


# --------------------------------------------------------------------------- #
# Share duration curves (% of demand)
# --------------------------------------------------------------------------- #
def share_curve(df: pd.DataFrame, which: str, save: Path | None = None):
    """Duration curve of one component's share of total demand (%), sorted desc."""
    share = _annual_twh(df, COMPONENTS[which]) / df["total_demand_TWh"] * 100.0
    y = share.sort_values(ascending=False).to_numpy()
    x = range(len(y))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(x, y, color=_COLORS[which], alpha=0.85, label=_LABELS[which])
    ax.plot(x, y, color=_COLORS[which], lw=1.2)

    ax.set_xlabel("Runs (sorted by share, descending)")
    ax.set_ylabel("Share of total demand [%]")
    ax.set_title(f"{_LABELS[which]} — share of demand")
    ax.set_xlim(0, len(y) - 1)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    return _finish(fig, save)


def combined_share_curves(df: pd.DataFrame, save: Path | None = None):
    """
    3 supply shares of total demand overlaid, each a duration curve sorted
    descending independently. Shows each source's achievable range (curves do
    NOT sum to 100% at a given x — independent sort).
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for which in ("centralized", "local", "import"):
        share = _annual_twh(df, COMPONENTS[which]) / df["total_demand_TWh"] * 100.0
        y = share.sort_values(ascending=False).to_numpy()
        ax.plot(range(len(y)), y, color=_COLORS[which], lw=1.8, label=_LABELS[which])

    ax.set_xlabel("Runs (sorted by share, descending — independent per curve)")
    ax.set_ylabel("Share of total demand [%]")
    ax.set_title("Supply-source shares of total demand — duration curves")
    ax.set_xlim(0, len(df) - 1)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    return _finish(fig, save)


def local_self_sufficiency_curve(df: pd.DataFrame, save: Path | None = None):
    """
    Duration curve: SMALL-cluster production / SMALL-cluster demand (%).
    Values > 100% = small clusters overproduce and export via network.
    """
    share = _annual_twh(df, COMPONENTS["local"]) / small_cluster_demand_twh(df) * 100.0
    y = share.sort_values(ascending=False).to_numpy()
    x = range(len(y))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(x, y, color=_COLORS["local"], alpha=0.85,
                    label="Small production / small demand")
    ax.plot(x, y, color=_COLORS["local"], lw=1.2)
    ax.axhline(100, color="black", lw=1.0, ls="--", label="100% (self-sufficient)")

    ax.set_xlabel("Runs (sorted by share, descending)")
    ax.set_ylabel("Small-cluster demand met locally [%]")
    ax.set_title("Small-cluster self-sufficiency (local production / local demand)")
    ax.set_xlim(0, len(y) - 1)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    return _finish(fig, save)


# --------------------------------------------------------------------------- #
# Stacked composition (% of demand)
# --------------------------------------------------------------------------- #
def supply_stack_pct(df: pd.DataFrame, save: Path | None = None):
    """Per-run stacked supply composition (% of demand), runs sorted by demand."""
    return _stacked_pct(df, sort_by=None, save=save,
                        title="Per-run supply composition (% of demand)",
                        xlabel="Runs (sorted by total demand)")


def supply_duration_pct(df: pd.DataFrame, sort_by: str = "centralized",
                        save: Path | None = None):
    """
    Stacked supply-mix duration curve (% of demand), runs sorted by `sort_by`
    share (descending). Stack can exceed 100% (surplus = storage/network losses).
    `sort_by` in {"centralized", "local", "import"}.
    """
    return _stacked_pct(df, sort_by=sort_by, save=save,
                        title="Supply-mix duration curve (% of demand)",
                        xlabel=f"Runs (sorted by {sort_by} share, descending)")


def _stacked_pct(df, sort_by, save, title, xlabel):
    """Shared stacked-% plot. sort_by=None -> sort by total demand ascending."""
    demand = df["total_demand_TWh"]
    shares = {k: _annual_twh(df, COMPONENTS[k]) / demand * 100 for k in COMPONENTS}

    if sort_by is None:
        order = demand.sort_values().index
    else:
        order = shares[sort_by].sort_values(ascending=False).index
    x = range(len(order))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(
        x,
        shares["centralized"][order],
        shares["local"][order],
        shares["import"][order],
        labels=[_LABELS["centralized"], _LABELS["local"], _LABELS["import"]],
        colors=[_COLORS["centralized"], _COLORS["local"], _COLORS["import"]],
        alpha=0.9,
    )
    ax.axhline(100, color="black", lw=1.0, ls="--", label="100% of demand")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of total demand [%]")
    ax.set_title(title)
    ax.set_xlim(0, len(order) - 1)
    ax.set_ylim(0, None)
    ax.legend(loc="lower left")
    return _finish(fig, save)


def decentralization_overlay(df: pd.DataFrame, save: Path | None = None):
    """
    Overlay the two decentralization metrics as duration curves (runs sorted by
    the production-mix metric). The shaded gap between them = import dependence:
        production mix  = local / (local + centralized)
        self-supply     = local / (local + centralized + import)
    """
    d_prod = decentralization(df, "production")
    d_supp = decentralization(df, "supply")
    order = d_prod.sort_values(ascending=False).index
    x = range(len(order))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, d_prod[order].to_numpy(), color="#2ca02c", lw=1.6,
            label="local / total_production (production mix)")
    ax.plot(x, d_supp[order].to_numpy(), color="#9467bd", lw=1.4,
            label="local / total_supply (self-supply)")
    ax.fill_between(x, d_supp[order].to_numpy(), d_prod[order].to_numpy(),
                    color="#ff7f0e", alpha=0.25, label="gap (= import dependence)")

    ax.set_xlabel("Runs (sorted by production-mix decentralization, desc)")
    ax.set_ylabel("Decentralization [%]")
    ax.set_title("Decentralization: production mix vs self-supply")
    ax.set_xlim(0, len(order) - 1)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    return _finish(fig, save)


def decentralization_triptych(df: pd.DataFrame, save: Path | None = None):
    """
    Three panels in a row, shared y-axis:
        left   = production-mix decentralization curve (local / total_production)
        middle = both curves overlaid + gap (import dependence)
        right  = self-supply decentralization curve (local / total_supply)
    Each curve is a duration curve sorted by its own values, except the middle
    which is sorted by the production-mix metric.
    """
    d_prod = decentralization(df, "production")
    d_supp = decentralization(df, "supply")

    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)

    # left: production mix alone
    yL = d_prod.sort_values(ascending=False).to_numpy()
    axL.fill_between(range(len(yL)), yL, color="#2ca02c", alpha=0.85)
    axL.plot(range(len(yL)), yL, color="#2ca02c", lw=1.2)
    axL.set_title("local / total_production\n(production mix)")
    axL.set_xlim(0, len(yL) - 1)

    # middle: overlay, sorted by production mix
    order = d_prod.sort_values(ascending=False).index
    x = range(len(order))
    axM.plot(x, d_prod[order].to_numpy(), color="#2ca02c", lw=1.6, label="production mix")
    axM.plot(x, d_supp[order].to_numpy(), color="#9467bd", lw=1.4, label="self-supply")
    axM.fill_between(x, d_supp[order].to_numpy(), d_prod[order].to_numpy(),
                     color="#ff7f0e", alpha=0.25, label="gap (import dependence)")
    axM.set_title("overlay\n(sorted by production mix)")
    axM.set_xlim(0, len(order) - 1)
    axM.legend(loc="upper right", fontsize=8)

    # right: self-supply alone
    yR = d_supp.sort_values(ascending=False).to_numpy()
    axR.fill_between(range(len(yR)), yR, color="#9467bd", alpha=0.85)
    axR.plot(range(len(yR)), yR, color="#9467bd", lw=1.2)
    axR.set_title("local / total_supply\n(self-supply)")
    axR.set_xlim(0, len(yR) - 1)

    for ax in (axL, axM, axR):
        ax.set_xlabel("Runs (sorted, desc)")
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
    axL.set_ylabel("Decentralization [%]")
    fig.suptitle("Decentralization: production mix vs self-supply")
    return _finish(fig, save)


def decentralization_gap(df: pd.DataFrame, save: Path | None = None):
    """
    Where the two decentralization metrics differ, and why. The gap between
    production-mix and self-supply is exactly the import share:
        d_supp = d_prod * (1 - import_share),  import_share = import/total_supply

    Left  : both curves + gap, runs sorted by IMPORT share (gap fans out
            monotonically -> difference lives in high-import runs).
    Right : the gap (d_prod - d_supp) vs import share -> collapses onto a line,
            proving the difference is import dependence.
    """
    d_prod = decentralization(df, "production")
    d_supp = decentralization(df, "supply")
    imp = _annual_twh(df, COMPONENTS["import"])
    tot_supply = (_annual_twh(df, COMPONENTS["centralized"])
                  + _annual_twh(df, COMPONENTS["local"]) + imp)
    import_share = imp / tot_supply * 100
    gap = d_prod - d_supp

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))

    order = import_share.sort_values().index          # sort by import share
    x = range(len(order))
    axL.plot(x, d_prod[order].to_numpy(), color="#2ca02c", lw=1.5, label="production mix")
    axL.plot(x, d_supp[order].to_numpy(), color="#9467bd", lw=1.5, label="self-supply")
    axL.fill_between(x, d_supp[order].to_numpy(), d_prod[order].to_numpy(),
                     color="#ff7f0e", alpha=0.3, label="gap")
    axL.set_xlabel("Runs (sorted by import share, ascending)")
    axL.set_ylabel("Decentralization [%]")
    axL.set_title("Curves sorted by import share")
    axL.set_xlim(0, len(order) - 1)
    axL.set_ylim(0, None)
    axL.grid(True, alpha=0.3)
    axL.legend(loc="upper left")

    axR.scatter(import_share, gap, s=8, alpha=0.35, color="#ff7f0e")
    axR.set_xlabel("Import share [% of total supply]")
    axR.set_ylabel("Gap = production_mix - self_supply [pp]")
    axR.set_title("Gap is explained by import share")
    axR.grid(True, alpha=0.3)

    fig.suptitle("Where the two decentralization metrics differ (= import dependence)")
    return _finish(fig, save)


def share_curve_grid(df: pd.DataFrame, which=("centralized", "local", "import"),
                     save: Path | None = None):
    """
    Side-by-side power/duration curves: each source's share of total demand (%),
    sorted descending independently. One panel per source.
    """
    fig, axes = plt.subplots(1, len(which), figsize=(6 * len(which), 5.5),
                             sharey=True)
    if len(which) == 1:
        axes = [axes]
    for ax, w in zip(axes, which):
        share = _annual_twh(df, COMPONENTS[w]) / df["total_demand_TWh"] * 100.0
        y = share.sort_values(ascending=False).to_numpy()
        ax.fill_between(range(len(y)), y, color=_COLORS[w], alpha=0.85)
        ax.plot(range(len(y)), y, color=_COLORS[w], lw=1.2)
        ax.set_xlabel("Runs (sorted, descending)")
        ax.set_title(_LABELS[w])
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Share of total demand [%]")
    return _finish(fig, save)


# --------------------------------------------------------------------------- #
# Quick / scratch plotting — try any per-row operation between columns
# --------------------------------------------------------------------------- #
def quick(y, kind="duration", x=None, label=None, save: Path | None = None):
    """
    Fast one-liner plotting of any per-row Series (one value per run).

    y     : a pandas Series, e.g. df["var_npv"], df["A"] / df["B"], df["A"] - df["B"].
    kind  :
        "duration" -> sort y descending, plot as power/duration curve (default)
        "sorted"   -> plot y sorted ascending
        "raw"      -> plot y in run order
        "hist"     -> histogram of y
        "scatter"  -> y vs x (x is another Series; falls back to run index)
    x     : Series for "scatter" (or a custom sort key for "duration"/"sorted").
    label : legend/axis label (defaults to the Series name).

    Examples:
        quick(df["var_npv"])
        quick(df["Large_cluster1_Electrolyzer_big"] / df["total_demand_TWh"])
        quick(df["npv_over_demand"], kind="hist")
        quick(df["var_npv"], kind="scatter", x=df["total_demand_TWh"])
    """
    y = pd.Series(y)
    label = label or (y.name if y.name is not None else "value")

    fig, ax = plt.subplots(figsize=(11, 6))

    if kind == "hist":
        ax.hist(y.dropna(), bins=50, color="#1f77b4", alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label}")
    elif kind == "scatter":
        xs = pd.Series(x) if x is not None else pd.Series(range(len(y)))
        ax.scatter(xs, y, s=8, alpha=0.4, color="#1f77b4")
        ax.set_xlabel(x.name if (x is not None and x.name) else "run index")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs {ax.get_xlabel()}")
        ax.grid(True, alpha=0.3)
    else:
        if kind == "duration":
            vals = (y.reindex(x.sort_values(ascending=False).index) if x is not None
                    else y.sort_values(ascending=False)).to_numpy()
        elif kind == "sorted":
            vals = (y.reindex(x.sort_values().index) if x is not None
                    else y.sort_values()).to_numpy()
        else:  # raw
            vals = y.to_numpy()
        ax.plot(range(len(vals)), vals, color="#1f77b4", lw=1.5, label=label)
        ax.fill_between(range(len(vals)), vals, color="#1f77b4", alpha=0.25)
        ax.set_xlabel("Runs" + ("" if kind == "raw" else f" (sorted, {kind})"))
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xlim(0, len(vals) - 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    return _finish(fig, save)


def quick_dual(y, y2, kind="duration", label=None, label2=None,
               save: Path | None = None):
    """
    Duration curve of `y` (line, left axis) with `y2` overlaid as points on a
    second right axis, using the SAME run ordering (runs sorted by y).

    y     : Series -> the power/duration curve (e.g. small_prod / total_prod).
    y2    : Series -> shown as scatter for the same runs (e.g. demand_level_ratio).
    kind  : "duration" (sort y desc, default) or "sorted" (sort y asc).

    Example:
        quick_dual(small_prod / total_prod, df["demand_level_ratio"])
    """
    y = pd.Series(y)
    y2 = pd.Series(y2)
    label = label or (y.name if y.name is not None else "value")
    label2 = label2 or (y2.name if y2.name is not None else "value2")

    order = y.sort_values(ascending=(kind == "sorted")).index
    x = range(len(order))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, y[order].to_numpy(), color="#1f77b4", lw=1.6, label=label, zorder=3)
    ax.fill_between(x, y[order].to_numpy(), color="#1f77b4", alpha=0.2, zorder=2)
    ax.set_xlabel(f"Runs (sorted by {label}, {kind})")
    ax.set_ylabel(label, color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_xlim(0, len(order) - 1)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.scatter(x, y2[order].to_numpy(), s=10, alpha=0.4, color="#ff7f0e",
                label=label2, zorder=1)
    ax2.set_ylabel(label2, color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    ax.set_title(f"{label} (curve) + {label2} (points)")
    return _finish(fig, save)


def relation(x, y, c=None, bins=15, label_x=None, label_y=None, label_c=None,
             cmap="viridis", save: Path | None = None):
    """
    Direct relationship between two per-row quantities: scatter of y vs x plus a
    binned mean +/- std trend line. Clearer than a rank axis when x is a
    continuous parameter (e.g. demand_level_ratio).

    c : optional Series -> colors each point by a third parameter (adds colorbar),
        e.g. c=df["electricity_price_avg"] to see what explains the scatter.

    Example:
        relation(df["demand_level_ratio"], small_prod / total_prod)
        relation(df["demand_level_ratio"], small_prod / total_prod,
                 c=df["electricity_price_avg"])
    """
    import numpy as np

    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    cc = pd.Series(c).reset_index(drop=True) if c is not None else None
    m = x.notna() & y.notna()
    if cc is not None:
        m = m & cc.notna()
    x, y = x[m].to_numpy(), y[m].to_numpy()
    cc = cc[m].to_numpy() if cc is not None else None
    label_x = label_x or "x"
    label_y = label_y or "y"

    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.digitize(x, edges[1:-1])
    centers, means, stds = [], [], []
    for b in range(bins):
        sel = idx == b
        if sel.sum():
            centers.append(x[sel].mean())
            means.append(y[sel].mean())
            stds.append(y[sel].std())
    centers, means, stds = map(np.array, (centers, means, stds))

    fig, ax = plt.subplots(figsize=(11, 6))
    if cc is not None:
        sc = ax.scatter(x, y, s=12, alpha=0.6, c=cc, cmap=cmap, label="runs")
        fig.colorbar(sc, ax=ax, label=label_c or "color")
    else:
        ax.scatter(x, y, s=8, alpha=0.25, color="#1f77b4", label="runs")
    ax.plot(centers, means, color="#d62728", lw=2.2, marker="o", label="binned mean")
    ax.fill_between(centers, means - stds, means + stds, color="#d62728",
                    alpha=0.15, label="+/- 1 std")

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(f"{label_y} vs {label_x}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return _finish(fig, save)


def relation_color_grid(x, y, ccols, df=None, label_x=None, label_y=None,
                        cmap="viridis", ncols=2, save: Path | None = None):
    """
    Same scatter (y vs x) repeated in a grid of panels, each colored by a
    different variable in `ccols`. Reveals which parameter explains the spread.
    Panels wrap over `ncols` columns (compact 2xN layout by default).

    x, y  : Series. ccols : list of column names (resolved from `df`) or Series.
    """
    import math

    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    label_x = label_x or (x.name if x.name is not None else "x")
    label_y = label_y or (y.name if y.name is not None else "y")

    n = len(ccols)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.flat

    for ax, col in zip(flat, ccols):
        cser = pd.Series(df[col] if isinstance(col, str) else col).reset_index(drop=True)
        name = col if isinstance(col, str) else (cser.name or "color")
        sc = ax.scatter(x, y, s=8, alpha=0.6, c=cser, cmap=cmap)
        fig.colorbar(sc, ax=ax, label=name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    for ax in list(flat)[n:]:      # hide unused panels
        ax.set_visible(False)

    for r in range(nrows):
        axes[r, 0].set_ylabel(label_y)
    for c in range(ncols):
        axes[nrows - 1, c].set_xlabel(label_x)

    fig.suptitle(f"{label_y} vs {label_x}")
    return _finish(fig, save)


def _binned_mean(x, y, bins):
    """Return (centers, means, stds) of y binned over x into `bins` equal bins."""
    import numpy as np
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.digitize(x, edges[1:-1])
    centers, means, stds = [], [], []
    for b in range(bins):
        sel = idx == b
        if sel.sum():
            centers.append(x[sel].mean())
            means.append(y[sel].mean())
            stds.append(y[sel].std())
    return map(np.array, (centers, means, stds))


def relation_grid(df, y, xcols, bins=15, label_y=None, save: Path | None = None):
    """
    One row of scatter+binned-mean panels sharing the same y, one panel per x
    column in `xcols`. Fast way to compare several drivers side by side.

    Example:
        relation_grid(df, small_prod / total_prod,
                      ["electricity_price_avg", "total_demand_TWh",
                       "el_import_limit_large"])
    """
    y = pd.Series(y).reset_index(drop=True)
    label_y = label_y or (y.name if y.name is not None else "y")

    fig, axes = plt.subplots(1, len(xcols), figsize=(6 * len(xcols), 5.5),
                             sharey=True)
    if len(xcols) == 1:
        axes = [axes]

    for ax, col in zip(axes, xcols):
        x = pd.Series(df[col]).reset_index(drop=True)
        m = x.notna() & y.notna()
        xv, yv = x[m].to_numpy(), y[m].to_numpy()
        ax.scatter(xv, yv, s=8, alpha=0.25, color="#1f77b4")
        centers, means, stds = _binned_mean(xv, yv, bins)
        ax.plot(centers, means, color="#d62728", lw=2.2, marker="o",
                label="binned mean")
        ax.fill_between(centers, means - stds, means + stds, color="#d62728",
                        alpha=0.15)
        ax.set_xlabel(col)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(label_y)
    axes[0].legend(loc="best")
    fig.suptitle(f"{label_y} vs drivers")
    return _finish(fig, save)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    df = load_results()

    # --- predefined supply-mix plots (uncomment one) ---
    # supply_stack(df)
    # demand_vs_component(df, "centralized")
    # share_curve(df, "import")
    # combined_share_curves(df)
    # share_curve_grid(df, which=("centralized", "local", "import"))
    # local_self_sufficiency_curve(df)
    # supply_stack_pct(df)
    # supply_duration_pct(df, sort_by="centralized")

    # --- decentralization ---
    decentralization_gap(df)                          # where/why the two differ
    # decentralization_triptych(df)                   # prod-mix | overlay | self-supply
    # decentralization_overlay(df)                    # just the overlay

    # decentr = decentralization(df, "production")    # or "supply"
    # for ycol in ("var_npv", "npv_over_demand"):
    #     print(f"corr(decentr, {ycol}) = {decentr.corr(df[ycol]):.3f}")
    # relation_color_grid(decentr, df["npv_over_demand"],
    #                     ["electricity_price_avg", "electricity_availability_small",
    #                      "electricity_availability_large", "hydrogen_import_price"],
    #                     df=df, label_x="Decentralization [%]",
    #                     label_y="npv_over_demand")