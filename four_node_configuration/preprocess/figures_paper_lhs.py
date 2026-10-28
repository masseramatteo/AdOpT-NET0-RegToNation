"""
Paper figure - constrained Latin hypercube of the techno-economic scenarios.

Illustrative two-dimensional version of sampling_design.constrained_lhs_economies:
a small Latin hypercube over demand D and total grid capacity C = 2 C_L + 2 C_S
(real ranges), the infeasible corner g < margin drawn for r_imp = 0, and the
Petelet-style repair used by the sampler: a coordinate of an infeasible point is
exchanged with the same coordinate of a feasible point (D first, then C), and
the exchange is kept only if both points end up feasible. Exchanging a
coordinate exchanges the stratum, so both marginals stay Latin.

    constrained_lhs         left: design as drawn, infeasible points in red;
                            right: repaired design, arrows old -> new position
    constrained_lhs_single  LAYOUT = "single": both in one panel

SVG (editable text) + PDF + PNG into OUT_DIR. Style shared with
figures_paper_topology.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sampling_design import ETA_ELECTROLYZER

import figures_paper_topology as topo

# ------------------------------------------------------------------- config --

OUT_DIR = topo.OUT_DIR.parent / "constrained_lhs"
PANEL_MM = 62

N = 12                               # points of the illustrative hypercube
MARGIN = 1.02                        # g + r_imp >= MARGIN, here r_imp = 0
D_RANGE = (5.0, 15.0)                # TWh/yr
C_RANGE = (2 * 500 + 2 * 20, 2 * 2000 + 2 * 120)   # MW, 2 C_L + 2 C_S
SEED = None                          # None = search for a readable seed
WANT_INFEASIBLE = (2, 3)             # accepted number of infeasible points
CLEAR = 1.05                         # after repair every point has g >= CLEAR * MARGIN
LAYOUT = "two"                       # "two" = before | after, "single" = one panel
SWAP_ORDER = (0, 1)                  # axes tried in the repair: 0 = D, 1 = C (sampler: D first)

COLORS = {
    "point": "#0072B2",
    "infeasible": "#D55E00",
    "region": "#D55E00",
    "grid": "#D9D9D9",
    "old": "#B0B0B0",
}

# ---------------------------------------------------------------- sampling --


def p_req(d_twh):
    return d_twh * 1e6 / 8760.0 / ETA_ELECTROLYZER


def feasible(d, c):
    return c / p_req(d) >= MARGIN


def to_values(U):
    return (D_RANGE[0] + U[:, 0] * (D_RANGE[1] - D_RANGE[0]),
            C_RANGE[0] + U[:, 1] * (C_RANGE[1] - C_RANGE[0]))


def repair(U, rng):
    """Same rule as sampling_design.constrained_lhs_economies, two axes."""
    U = U.copy()
    swaps = []                                   # (i, j, axis)
    for _ in range(50):
        D, C = to_values(U)
        ok = feasible(D, C)
        bad = np.flatnonzero(~ok)
        if bad.size == 0:
            break
        good = np.flatnonzero(ok)
        for i in bad:
            done = False
            for axis in SWAP_ORDER:
                for j in rng.permutation(good):
                    U[i, axis], U[j, axis] = U[j, axis], U[i, axis]
                    D, C = to_values(U)
                    if feasible(D[i], C[i]) and feasible(D[j], C[j]):
                        swaps.append((i, j, axis))
                        done = True
                        break
                    U[i, axis], U[j, axis] = U[j, axis], U[i, axis]
                if done:
                    break
    D, C = to_values(U)
    if not feasible(D, C).all():
        return None, swaps
    return U, swaps


def design(seed):
    U0 = qmc.LatinHypercube(d=2, optimization="random-cd", seed=seed).random(N)
    D, C = to_values(U0)
    n_bad = int((~feasible(D, C)).sum())
    U1, swaps = repair(U0, np.random.default_rng(seed))
    return U0, U1, swaps, n_bad


def pick_seed():
    for s in range(1000):
        U0, U1, swaps, n_bad = design(s)
        if U1 is None or not WANT_INFEASIBLE[0] <= n_bad <= WANT_INFEASIBLE[1] \
                or len(swaps) != n_bad:
            continue
        D, C = to_values(U1)
        if (C / p_req(D)).min() >= CLEAR * MARGIN:
            return s
    raise RuntimeError("no seed found")


# ----------------------------------------------------------------- drawing --


def style_axes(ax):
    ax.set_xlim(*D_RANGE)
    ax.set_ylim(*C_RANGE)
    for x in np.linspace(*D_RANGE, N + 1)[1:-1]:
        ax.axvline(x, color=COLORS["grid"], lw=0.4, zorder=0)
    for y in np.linspace(*C_RANGE, N + 1)[1:-1]:
        ax.axhline(y, color=COLORS["grid"], lw=0.4, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("demand")
    ax.set_ylabel("grid capacity")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_box_aspect(1)
    # infeasible corner: C < MARGIN * P_req(D)
    d = np.linspace(*D_RANGE, 200)
    c_min = MARGIN * p_req(d)
    on = c_min > C_RANGE[0]
    ax.fill_between(d, C_RANGE[0], np.clip(c_min, *C_RANGE), where=on,
                    facecolor="none", edgecolor=COLORS["region"],
                    hatch="/////", lw=0, alpha=0.5, zorder=1)
    ax.plot(d[on], c_min[on], color=COLORS["region"], lw=0.8, zorder=2)


def draw_single(ax, U0, U1):
    """Final design in blue; each infeasible start point (red, in the hatched
    zone) with a solid arrow to where the exchange puts it; its feasible
    partner's move as a light arrow from its old position (grey)."""
    D0, C0 = to_values(U0)
    D1, C1 = to_values(U1)
    bad = ~feasible(D0, C0)
    moved = (D0 != D1) | (C0 != C1)
    partner = moved & ~bad
    for k in np.flatnonzero(moved):
        solid = bad[k]
        ax.annotate("", xy=(D1[k], C1[k]), xytext=(D0[k], C0[k]),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7 if solid else 0.5,
                                    color=COLORS["infeasible"] if solid else COLORS["old"],
                                    shrinkA=3, shrinkB=3, mutation_scale=6),
                    zorder=4)
    ax.scatter(D0[partner], C0[partner], s=12, facecolors="white",
               edgecolors=COLORS["old"], lw=0.6, zorder=3, clip_on=False)
    ax.scatter(D0[bad], C0[bad], s=12, facecolors="white",
               edgecolors=COLORS["infeasible"], lw=0.9, zorder=3, clip_on=False)
    ax.scatter(D1, C1, s=12, c=COLORS["point"], lw=0, zorder=5, clip_on=False)


def draw_before(ax, U0):
    D, C = to_values(U0)
    ok = feasible(D, C)
    ax.scatter(D[ok], C[ok], s=12, c=COLORS["point"], lw=0, zorder=3, clip_on=False)
    ax.scatter(D[~ok], C[~ok], s=12, facecolors="white",
               edgecolors=COLORS["infeasible"], lw=0.9, zorder=3, clip_on=False)


def draw_after(ax, U0, U1):
    D0, C0 = to_values(U0)
    D1, C1 = to_values(U1)
    moved = np.flatnonzero((D0 != D1) | (C0 != C1))
    ax.scatter(D0[moved], C0[moved], s=12, facecolors="white",
               edgecolors=COLORS["old"], lw=0.6, zorder=3, clip_on=False)
    for k in moved:
        ax.annotate("", xy=(D1[k], C1[k]), xytext=(D0[k], C0[k]),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.6,
                                    shrinkA=3, shrinkB=3, mutation_scale=6),
                    zorder=4)
    ax.scatter(D1, C1, s=12, c=COLORS["point"], lw=0, zorder=5, clip_on=False)


def figure(U0, U1):
    if LAYOUT == "single":
        fig, ax = plt.subplots(figsize=(PANEL_MM * topo.MM, PANEL_MM * topo.MM))
        style_axes(ax)
        draw_single(ax, U0, U1)
        fig.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.97)
        topo.save(fig, "constrained_lhs_single")
        return
    fig, axs = plt.subplots(1, 2, figsize=(2 * PANEL_MM * topo.MM,
                                           PANEL_MM * 0.95 * topo.MM))
    for ax in axs:
        style_axes(ax)
    draw_before(axs[0], U0)
    draw_after(axs[1], U0, U1)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.1, top=0.98, wspace=0.25)
    topo.save(fig, "constrained_lhs")


def main():
    topo.set_style()
    topo.OUT_DIR = OUT_DIR                       # save() writes to topo.OUT_DIR
    seed = SEED if SEED is not None else pick_seed()
    U0, U1, swaps, n_bad = design(seed)
    print(f"seed {seed}: {n_bad} infeasible of {N}, swaps "
          + ", ".join(f"{i}<->{j} on {'DC'[a]}" for i, j, a in swaps))
    figure(U0, U1)
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
