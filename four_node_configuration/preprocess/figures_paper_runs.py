"""
Paper figure - from the two samples to the optimisation runs (v5 design).

Run matrix: one row per techno-economic scenario (glyph of its 12 sampled
values, normalised to their ranges), one column per topology slot, one cell
per optimisation run showing the actual node layout of that run. Rows shown:
the first few economies, an ellipsis row and the last economy.

Reads the real design: results/sampling_design_dry_run/economies.csv (same
seed as the campaign) and preprocess/generated_topology_v5/
{design_manifest,topology_parameters}.csv. SVG (editable text) + PDF + PNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import figures_paper_topology as topo

HERE = Path(__file__).resolve().parent
ECON_CSV = HERE.parent / "results" / "sampling_design_dry_run" / "economies.csv"
TOPO_DIR = HERE / "generated_topology_v5"
OUT_DIR = topo.OUT_DIR.parent / "run_matrix"

# ------------------------------------------------------------------- config --

ROWS = [1, 2, 3, None, 750]          # economies shown, None = ellipsis row
SLOTS = ["A_near", "A_far", "B_near", "B_far",
         "C_near", "C_far", "D_near", "D_far"]
SLOT_LABEL = {s: s.replace("_", " ") for s in SLOTS}
SLOT_GROUP = {"A": "AB", "B": "AB", "C": "C", "D": "D"}

WIDTH_MM = topo.FULL_WIDTH_MM
ROW_MM = 17
GLYPH_W = 1.5                        # width of the economy column, in cell widths

NODE = {"large": "#161D41", "small": "#FFCD00"}     # as archetype_schematics
PARAM_RANGES = {                     # campaign ranges, for the glyph
    "total_demand_TWh": (5, 15), "demand_level_ratio": (8, 30),
    "unbalance_ratio": (2, 8), "import_availability_ratio": (0, 0.7),
    "hydrogen_import_price": (150, 300), "electricity_price_avg": (20, 250),
    "electricity_standard_dev": (10, 100), "solar_cf_mean": (0.09, 0.17),
    "electricity_availability_large": (500, 2000),
    "electricity_availability_small": (20, 120),
    "capex_ratio_small_big": (1.0, 2.0), "pipeline_cost_multiplier": (0.5, 2.0),
}

# -------------------------------------------------------------------- data --


def load():
    econ = pd.read_csv(ECON_CSV).set_index("economy")
    man = pd.read_csv(TOPO_DIR / "design_manifest.csv", sep=";", dtype=str)
    man["economy"] = man["economy"].astype(int)
    man = man.set_index("economy")
    tp = pd.read_csv(TOPO_DIR / "topology_parameters.csv", sep=";",
                     dtype={"topology": str}).set_index("topology")
    return econ, man, tp


def positions(tp, name):
    r = tp.loc[name]
    big = [(r["Large_cluster1_x"], r["Large_cluster1_y"]),
           (r["Large_cluster2_x"], r["Large_cluster2_y"])]
    small = [(r["Small_cluster1_x"], r["Small_cluster1_y"]),
             (r["Small_cluster2_x"], r["Small_cluster2_y"])]
    big, small = np.array(big), np.array(small)
    allp = np.vstack([big, small])                  # centre the layout's bbox
    c = (allp.min(0) + allp.max(0)) / 2
    return big - c, small - c


# ----------------------------------------------------------------- drawing --


def draw_layout(ax, big, small, lim):
    ax.plot(big[:, 0], big[:, 1], color="#BBBBBB", lw=0.5, zorder=1)
    for s in small:                                  # link to nearest hub
        h = big[np.argmin(np.hypot(*(big - s).T))]
        ax.plot([s[0], h[0]], [s[1], h[1]], color="#BBBBBB", lw=0.4, zorder=1)
    ax.scatter(big[:, 0], big[:, 1], s=14, c=NODE["large"], lw=0, zorder=3)
    ax.scatter(small[:, 0], small[:, 1], s=9, marker="s", c=NODE["small"],
               edgecolors=NODE["large"], lw=0.3, zorder=3)
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])
    ax.set_aspect("equal")
    ax.axis("off")


def cell_frame(ax, color):
    ax.add_patch(FancyBboxPatch((0.03, 0.05), 0.94, 0.9,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                transform=ax.transAxes, fc="white",
                                ec=color, lw=0.6, zorder=0, clip_on=False))


def draw_glyph(ax, row):
    keys = list(PARAM_RANGES)
    v = [(row[k] - lo) / (hi - lo) for k, (lo, hi) in PARAM_RANGES.items()]
    x = np.arange(len(keys))
    ax.bar(x, [1] * len(x), width=0.7, color="#EEEEEE", lw=0)
    ax.bar(x, v, width=0.7, color="#555555", lw=0)
    ax.set_xlim(-0.8, len(x) - 0.2)
    ax.set_ylim(0, 1.05)
    ax.axis("off")


def main():
    topo.set_style()
    topo.OUT_DIR = OUT_DIR
    econ, man, tp = load()

    # common km window over every layout shown, so near/far read at one scale
    names = [man.loc[e, s] for e in ROWS if e is not None for s in SLOTS]
    P = np.vstack([np.vstack(positions(tp, n)) for n in names])
    xh = abs(P[:, 0]).max() * 1.22                 # half-window, km
    yh = max(abs(P[:, 1]).max() * 1.2, xh * 0.6)
    lim = ((-xh, xh), (-yh, yh))

    n_cols = 1 + len(SLOTS)
    fig = plt.figure(figsize=(WIDTH_MM * topo.MM,
                              (len(ROWS) * ROW_MM + 8) * topo.MM))
    gs = fig.add_gridspec(len(ROWS) + 1, n_cols,
                          width_ratios=[GLYPH_W] + [1] * len(SLOTS),
                          height_ratios=[0.35] + [1] * len(ROWS),
                          left=0.035, right=0.995, top=0.99, bottom=0.01,
                          wspace=0.06, hspace=0.12)

    for c, s in enumerate(SLOTS, start=1):           # column headers
        ax = fig.add_subplot(gs[0, c])
        ax.axis("off")
        ax.text(0.5, 0.3, SLOT_LABEL[s], ha="center", va="center",
                color=topo.COLORS[SLOT_GROUP[s[0]]], fontweight="bold",
                transform=ax.transAxes)

    for r, e in enumerate(ROWS, start=1):
        if e is None:                                # ellipsis row
            for c in range(n_cols):
                ax = fig.add_subplot(gs[r, c])
                ax.axis("off")
                ax.text(0.5, 0.5, r"$\vdots$", ha="center", va="center",
                        fontsize=topo.FONT_SIZE + 3, transform=ax.transAxes)
            continue
        ax = fig.add_subplot(gs[r, 0])
        draw_glyph(ax, econ.loc[e])
        fig.text(0.004, ax.get_position().y0 + ax.get_position().height / 2,
                 f"{e}", ha="left", va="center")
        for c, s in enumerate(SLOTS, start=1):
            ax = fig.add_subplot(gs[r, c])
            cell_frame(ax, topo.COLORS[SLOT_GROUP[s[0]]])
            big, small = positions(tp, man.loc[e, s])
            draw_layout(ax, big, small, lim)

    topo.save(fig, "run_matrix")
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
