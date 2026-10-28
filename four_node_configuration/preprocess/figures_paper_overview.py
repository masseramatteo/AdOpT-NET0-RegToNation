"""
Paper figure - scenario generation overview (v5): two sampling streams that
merge into the optimisation runs.

    top lane      spatial stream: per archetype group a Sobol box (accepted /
                  rejected candidates, feasible volume) -> d_SL strip split at
                  the median into near / far, rank pairing drawn as nested arcs
    bottom lane   techno-economic stream: constrained Latin hypercube (swap
                  repair) -> row of scenario glyphs te 1 ... te N
    right         scenario te k fans out to its 8 configurations (near / far x
                  A, B, C, D, real layouts); coloured threads carry pair k of
                  every group into its columns (A and B share the A/B pair)

Real data: Sobol streams of generate_node_topology_paired.py, the v5 design
files and the dry-run economies. Reuses the drawing helpers of the other
figures_paper_* scripts. SVG (editable text) + PDF + PNG.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Arc

import figures_paper_topology as topo
import figures_paper_lhs as lhs
import figures_paper_runs as runs

OUT_DIR = topo.OUT_DIR.parent / "overview"

# ------------------------------------------------------------------- config --

K = 3                                # highlighted economy (te k)
TE_SHOWN = [1, 2, 3, None, 750]      # scenario glyphs in the bottom lane
GROUPS = ["AB", "D", "C"]            # top-lane order
GROUP_SLOTS = {"AB": ["A", "B"], "D": ["D"], "C": ["C"]}
CELL_COLS = ["A", "B", "C", "D"]
STRIP_POINTS = 40                    # points drawn per d_SL strip
N_ARCS = 5                           # rank-pairing arcs drawn per strip
N_BOX_CAND = 250                     # Sobol candidates in the mini boxes
FIG_MM = (topo.FULL_WIDTH_MM, 105)
DARK = "#333333"
STRIP_Y = [0.84, 0.66, 0.48]         # bottom of each group strip (AB, D, C)
STRIP_H = 0.11
CELL_X0, CELL_W, CELL_H, CELL_GAP = 0.63, 0.075, 0.13, 0.008

# ----------------------------------------------------------------- helpers --


def fig_axes(fig, rect, **kw):
    ax = fig.add_axes(rect, **kw)
    return ax


def arrow(fig, p0, p1, color=DARK, lw=0.8, rad=0.0):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=7,
                                   lw=lw, color=color,
                                   connectionstyle=f"arc3,rad={rad}"))


def thread(fig, ax_a, xy_a, ax_b, xy_b, color, lw=0.6, rad=0.0, ls="-"):
    fig.add_artist(ConnectionPatch(xy_a, xy_b, "data", "data", axesA=ax_a,
                                   axesB=ax_b, color=color, lw=lw, ls=ls,
                                   connectionstyle=f"arc3,rad={rad}",
                                   zorder=0))


# ------------------------------------------------------------- top: Sobol --


def mini_box(fig, rect, g, U, ok, surf, vol):
    ax = fig.add_axes(rect, projection="3d", computed_zorder=False)
    ax.view_init(**topo.VIEW)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(*topo.D_LL_RANGE); ax.set_ylim(*topo.D_SL_RANGE)
    ax.set_zlim(*topo.D_SS_RANGE)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(topo.COLORS["box"])
        axis.pane.set_linewidth(0.3)
        axis.line.set_linewidth(0.4)
    ax.grid(False)
    topo.draw_box_faces(ax, vol, topo.COLORS[g])
    topo.draw_surface(ax, surf, topo.COLORS[g], alpha=topo.CUT_ALPHA)
    P, m = topo.to_km(U[:N_BOX_CAND]), ok[:N_BOX_CAND]
    ax.scatter(*P[~m].T, marker="x", s=2, lw=0.3, c=topo.COLORS["rejected"],
               depthshade=False, zorder=2)
    ax.scatter(*P[m].T, s=2.5, lw=0, c=topo.COLORS[g], depthshade=False, zorder=3)
    return ax


def strip(fig, rect, g, d_sl_all, d_near_k, d_far_k, rng):
    """d_SL number line: sample of the group's accepted points, median split,
    nested rank-pairing arcs, pair k highlighted. Returns (ax, x_near, x_far)."""
    ax = fig.add_axes(rect)
    lo, hi = topo.D_SL_RANGE
    med = np.median(d_sl_all)
    sub = np.sort(rng.choice(d_sl_all, STRIP_POINTS, replace=False))
    near, far = sub[sub < med], sub[sub >= med]
    col = topo.COLORS[g]
    ax.axhline(0, color="#999999", lw=0.4, zorder=0)
    ax.scatter(near, np.zeros_like(near), s=4, c=col, alpha=0.45, lw=0, zorder=2)
    ax.scatter(far, np.zeros_like(far), s=4, c=col, lw=0, zorder=2)
    ax.plot([med, med], [-0.35, 0.35], color=DARK, lw=0.6, ls=(0, (2, 1.5)))
    # nested arcs: i-th near <-> i-th far
    n = min(len(near), len(far))
    for i in np.linspace(0, n - 1, N_ARCS).astype(int):
        a, b = near[i], far[i]
        ax.add_patch(Arc(((a + b) / 2, 0), b - a, 1.4 * (b - a) / (hi - lo),
                         theta1=0, theta2=180, color=col, lw=0.35, alpha=0.6))
    a, b = d_near_k, d_far_k                         # pair k
    ax.add_patch(Arc(((a + b) / 2, 0), b - a, 1.4 * (b - a) / (hi - lo),
                     theta1=0, theta2=180, color=DARK, lw=0.8))
    ax.scatter([a, b], [0, 0], s=14, facecolors="white", edgecolors=DARK,
               lw=0.8, zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.25, 0.95)
    ax.axis("off")
    return ax


# -------------------------------------------------------- bottom: LHS + te --


def mini_lhs(fig, rect):
    ax = fig.add_axes(rect)
    seed = lhs.SEED if lhs.SEED is not None else lhs.pick_seed()
    U0, U1, _, _ = lhs.design(seed)
    lhs.style_axes(ax)
    ax.set_xlabel(""); ax.set_ylabel("")
    lhs.draw_single(ax, U0, U1)
    return ax


def glyph(fig, rect, row, highlight=False):
    ax = fig.add_axes(rect)
    runs.draw_glyph(ax, row)
    if highlight:
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(DARK); sp.set_linewidth(0.9)
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    return ax


# ------------------------------------------------------------------- main --


def main():
    topo.set_style()
    topo.OUT_DIR = OUT_DIR
    rng = np.random.default_rng(7)
    econ, man, tp = runs.load()
    W, H = FIG_MM
    fig = plt.figure(figsize=(W * topo.MM, H * topo.MM))

    # ---------------- top lane: Sobol boxes and near / far strips
    strip_axes, line_y = {}, {}
    for i, g in enumerate(GROUPS):
        U, ok = topo.candidate_stream(g)
        surf, vol = topo.boundary_surfaces(g)
        mini_box(fig, [0.005 + 0.105 * i, 0.60, 0.125, 0.36], g, U, ok, surf, vol)
        acc = topo.to_km(U[ok])
        slot = {"AB": "A", "D": "D", "C": "C"}[g]
        tn, tf = man.loc[K, f"{slot}_near"], man.loc[K, f"{slot}_far"]
        y0 = STRIP_Y[i]
        ax = strip(fig, [0.345, y0, 0.2, STRIP_H], g, acc[:, 1],
                   tp.loc[tn, "d_SL_target"], tp.loc[tf, "d_SL_target"], rng)
        ax.pair = (tp.loc[tn, "d_SL_target"], tp.loc[tf, "d_SL_target"])
        strip_axes[g] = ax
        line_y[g] = y0 + STRIP_H * 0.25 / 1.2        # data y = 0 in fig coords
    arrow(fig, (0.325, 0.78), (0.345, 0.78))

    # ---------------- bottom lane: constrained LHS and scenario glyphs
    mini_lhs(fig, [0.04, 0.07, 0.19, 0.36])
    arrow(fig, (0.255, 0.25), (0.295, 0.25))
    te_axes = {}
    x = 0.305
    for e in TE_SHOWN:
        if e is None:
            fig.text(x + 0.012, 0.25, r"$\cdots$", ha="center", va="center",
                     fontsize=topo.FONT_SIZE + 2)
            x += 0.03
            continue
        te_axes[e] = glyph(fig, [x, 0.20, 0.05, 0.10], econ.loc[e],
                           highlight=(e == K))
        fig.text(x + 0.025, 0.175, f"te {e}" if e != 750 else r"te $N_e$",
                 ha="center", va="top", fontsize=topo.FONT_SIZE - 1,
                 fontweight="bold" if e == K else "normal")
        x += 0.06

    # ---------------- right: the 8 configurations of te k, one row per group
    rows = {"AB": ["A_near", "A_far", "B_near", "B_far"],
            "D": ["D_near", "D_far"], "C": ["C_near", "C_far"]}
    names = [man.loc[K, s_] for r_ in rows.values() for s_ in r_]
    P = np.vstack([np.vstack(runs.positions(tp, n)) for n in names])
    xh = abs(P[:, 0]).max() * 1.22
    yh = max(abs(P[:, 1]).max() * 1.2, xh * 0.6)
    lim = ((-xh, xh), (-yh, yh))
    cell_axes = {}
    for g, slots in rows.items():
        for j, s_ in enumerate(slots):
            ax = fig.add_axes([CELL_X0 + j * (CELL_W + CELL_GAP),
                               line_y[g] - CELL_H / 2, CELL_W, CELL_H])
            runs.cell_frame(ax, topo.COLORS[g])
            big, small = runs.positions(tp, man.loc[K, s_])
            runs.draw_layout(ax, big, small, lim)
            cell_axes[s_] = ax
            fig.text(CELL_X0 + j * (CELL_W + CELL_GAP) + CELL_W / 2,
                     line_y[g] + CELL_H / 2 + 0.004, s_.replace("_", " "),
                     ha="center", va="bottom", fontsize=topo.FONT_SIZE - 1,
                     color=topo.COLORS[g])
        a_, b_ = strip_axes[g].pair
        for s_ in slots:                               # pair k -> its cells
            src = a_ if s_.endswith("near") else b_
            fig.add_artist(ConnectionPatch(
                (src, 0), (0.03, 0.5), "data", "axes fraction",
                axesA=strip_axes[g], axesB=cell_axes[s_], color=topo.COLORS[g],
                lw=0.6, ls="-" if s_.endswith("near") else (0, (2, 1)),
                connectionstyle="arc3,rad=-0.12", zorder=-0.5))

    # dashed frame around the block + ghost frames = repetition over scenarios
    fx0 = CELL_X0 - 0.012
    fx1 = CELL_X0 + 4 * CELL_W + 3 * CELL_GAP + 0.012
    fy0 = line_y["C"] - CELL_H / 2 - 0.02
    fy1 = line_y["AB"] + CELL_H / 2 + 0.035
    from matplotlib.patches import FancyBboxPatch
    for k_, (dx, dy) in enumerate([(0.016, -0.024), (0.008, -0.012), (0, 0)]):
        last = k_ == 2
        fig.add_artist(FancyBboxPatch(
            (fx0 + dx, fy0 + dy), fx1 - fx0, fy1 - fy0,
            boxstyle="round,pad=0,rounding_size=0.01", transform=fig.transFigure,
            fc="white", ec=DARK if last else "#BBBBBB", lw=0.7 if last else 0.5,
            ls=(0, (3, 2)) if last else "-", zorder=-1 - (2 - k_)))
    fig.text(fx1 + 0.016, fy0 - 0.03, r"$	imes\,N_e$", ha="right", va="top",
             fontsize=topo.FONT_SIZE)
    # te k feeds the block
    fig.add_artist(ConnectionPatch(
        (0.5, 1.0), ((fx0 + fx1) / 2, fy0 - 0.024), "axes fraction",
        "figure fraction", axesA=te_axes[K], color=DARK, lw=0.8,
        arrowstyle="-|>", mutation_scale=7, connectionstyle="arc3,rad=-0.25"))

    topo.save(fig, "scenario_generation_overview")
    print(f"written to {OUT_DIR}")




# ------------------------------------------------------------ simple layout --

SIMPLE_FIG_MM = (topo.FULL_WIDTH_MM, 58)
SIMPLE_SLOTS = ["A_near", "A_far", "B_near", "B_far",
                "C_near", "C_far", "D_near", "D_far"]


def main_simple():
    """Two sources, one merge: Sobol boxes (top left) and constrained LHS
    (bottom left) -> block te k + its 8 configurations, stacked x N_e."""
    from matplotlib.patches import FancyBboxPatch
    topo.set_style()
    topo.OUT_DIR = OUT_DIR
    econ, man, tp = runs.load()
    W, H = SIMPLE_FIG_MM
    fig = plt.figure(figsize=(W * topo.MM, H * topo.MM))

    # sources
    for i, g in enumerate(GROUPS):
        U, ok = topo.candidate_stream(g)
        surf, vol = topo.boundary_surfaces(g)
        mini_box(fig, [-0.01 + 0.105 * i, 0.47, 0.125, 0.53], g, U, ok, surf, vol)
    mini_lhs(fig, [0.105, 0.02, 0.13, 0.42])

    # block: te k glyph + 8 cells in one row, cells shaped like the layouts
    names = [man.loc[K, s_] for s_ in SIMPLE_SLOTS]
    P = np.vstack([np.vstack(runs.positions(tp, n)) for n in names])
    xh = abs(P[:, 0]).max() * 1.15
    yh = max(abs(P[:, 1]).max() * 1.2, xh * 0.5)
    lim = ((-xh, xh), (-yh, yh))
    bx0, gw, cw, gap = 0.365, 0.034, 0.064, 0.005
    bh = cw * W * (yh / xh) / H                     # fig fraction
    by = 0.5 - bh / 2
    glyph(fig, [bx0, by, gw, bh], econ.loc[K], highlight=True)
    fig.text(bx0 + gw / 2, by - 0.02, r"$te_k$", ha="center", va="top")
    x = bx0 + gw + 0.012
    for s_ in SIMPLE_SLOTS:
        g = "AB" if s_[0] in "AB" else s_[0]
        ax = fig.add_axes([x, by, cw, bh])
        runs.cell_frame(ax, topo.COLORS[g])
        big, small = runs.positions(tp, man.loc[K, s_])
        runs.draw_layout(ax, big, small, lim)
        fig.text(x + cw / 2, by + bh + 0.015, s_.replace("_", " "), ha="center",
                 va="bottom", fontsize=topo.FONT_SIZE - 1, color=topo.COLORS[g])
        x += cw + gap
    fx0, fx1 = bx0 - 0.008, x - gap + 0.006
    fy0, fy1 = by - 0.11, by + bh + 0.1
    for k_, d in enumerate([0.012, 0.006, 0.0]):
        last = k_ == 2
        fig.add_artist(FancyBboxPatch(
            (fx0 + d, fy0 - 2 * d), fx1 - fx0, fy1 - fy0,
            boxstyle="round,pad=0,rounding_size=0.01", transform=fig.transFigure,
            fc="white", ec=DARK if last else "#BBBBBB", lw=0.7 if last else 0.5,
            zorder=-3 + k_))
    fig.text(fx1 + 0.012, fy0 - 0.035, r"$\times\,N_e$", ha="right", va="top")

    # the two merging arrows
    arrow(fig, (0.315, 0.72), (fx0 - 0.004, 0.56), rad=-0.2)
    arrow(fig, (0.245, 0.22), (fx0 - 0.004, 0.44), rad=0.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):                # cropped to content
        try:
            fig.savefig(OUT_DIR / f"scenario_generation_simple.{ext}",
                        dpi=topo.DPI_PNG, bbox_inches="tight", pad_inches=0.01)
        except OSError as e:
            print(f"  not written, file locked: {ext} ({e.strerror})")
    plt.close(fig)
    print(f"written to {OUT_DIR}")


# -------------------------------------------------------- sequential layout --

SEQ_FIG_MM = (topo.FULL_WIDTH_MM, 118)
TILE_ORDER = ["A_near", "A_far", "B_near", "B_far",
              "C_near", "C_far", "D_near", "D_far"]
CHIPS = [("AB", "near", ["A_near", "B_near"]), ("AB", "far", ["A_far", "B_far"]),
         ("C", "near", ["C_near"]), ("C", "far", ["C_far"]),
         ("D", "near", ["D_near"]), ("D", "far", ["D_far"])]


def split_panel(fig, rect, U, ok, pair):
    """d_SL x d_SS of the A/B sample: median split (far half shaded), pair k."""
    ax = fig.add_axes(rect)
    acc = topo.to_km(U[ok])
    sub = acc[np.random.default_rng(3).choice(len(acc), 220, replace=False)]
    med = np.median(acc[:, 1])
    col = topo.COLORS["AB"]
    ax.axvspan(med, topo.D_SL_RANGE[1], color="#F0F0F0", lw=0, zorder=0)
    ax.axvline(med, color=DARK, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
    near = sub[:, 1] < med
    ax.scatter(sub[near, 1], sub[near, 2], s=3, c=col, alpha=0.45, lw=0, zorder=2)
    ax.scatter(sub[~near, 1], sub[~near, 2], s=3, c=col, lw=0, zorder=2)
    (a1, a2), (b1, b2) = pair
    ax.plot([a1, b1], [a2, b2], color=DARK, lw=0.8, zorder=3)
    ax.scatter([a1, b1], [a2, b2], s=16, facecolors="white", edgecolors=DARK,
               lw=0.8, zorder=4)
    ax.set_xlim(*topo.D_SL_RANGE)
    ax.set_ylim(*topo.D_SS_RANGE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$d_{SL}$", labelpad=1)
    ax.set_ylabel(r"$d_{SS}$", labelpad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax


def chip(fig, rect, g, d):
    """Distance triple as three bars (d_LL, d_SL, d_SS scaled to their ranges)."""
    ax = fig.add_axes(rect)
    v = [(d[0] - topo.D_LL_RANGE[0]) / np.ptp(topo.D_LL_RANGE),
         (d[1] - topo.D_SL_RANGE[0]) / np.ptp(topo.D_SL_RANGE),
         (d[2] - topo.D_SS_RANGE[0]) / np.ptp(topo.D_SS_RANGE)]
    ax.bar(range(3), [1] * 3, width=0.6, color="#EEEEEE", lw=0)
    ax.bar(range(3), v, width=0.6, color=topo.COLORS[g], lw=0)
    ax.set_xlim(-0.7, 2.7)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(topo.COLORS[g])
        sp.set_linewidth(0.6)
    return ax


def line(fig, p0, p1, color, lw=0.6):
    fig.add_artist(plt.Line2D([p0[0], p1[0]], [p0[1], p1[1]],
                              transform=fig.transFigure, color=color, lw=lw,
                              zorder=-0.5))


def main_seq():
    from matplotlib.patches import FancyBboxPatch
    topo.set_style()
    topo.OUT_DIR = OUT_DIR
    econ, man, tp = runs.load()
    W, H = SEQ_FIG_MM
    fig = plt.figure(figsize=(W * topo.MM, H * topo.MM))

    def trip(name):
        r = tp.loc[name]
        return (r["d_LL_target"], r["d_SL_target"], r["d_SS_target"])

    # ---- (a) Sobol boxes -> near/far split (A/B shown)
    for i, g in enumerate(GROUPS):
        U, ok = topo.candidate_stream(g)
        surf, vol = topo.boundary_surfaces(g)
        mini_box(fig, [-0.01 + 0.1 * i, 0.64, 0.12, 0.35], g, U, ok, surf, vol)
        if g == "AB":
            U_ab, ok_ab = U, ok
    tn, tf = trip(man.loc[K, "A_near"]), trip(man.loc[K, "A_far"])
    split_panel(fig, [0.36, 0.68, 0.12, 0.26], U_ab, ok_ab,
                ((tn[1], tn[2]), (tf[1], tf[2])))
    arrow(fig, (0.302, 0.81), (0.33, 0.81))

    # ---- (b) constrained LHS: before -> after
    seed = lhs.SEED if lhs.SEED is not None else lhs.pick_seed()
    U0, U1, _, _ = lhs.design(seed)
    for x0, which in [(0.585, "before"), (0.80, "after")]:
        ax = fig.add_axes([x0, 0.66, 0.16, 0.30])
        lhs.style_axes(ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if which == "before":
            lhs.draw_before(ax, U0)
        else:
            lhs.draw_after(ax, U0, U1)
    arrow(fig, (0.765, 0.81), (0.795, 0.81))
    fig.add_artist(plt.Line2D([0.535, 0.535], [0.64, 0.98],
                              transform=fig.transFigure, color="#DDDDDD", lw=0.6))

    # ---- (c) scenario row
    te_y, te_h, te_w = 0.47, 0.10, 0.045
    te_x = {}
    x = 0.42
    for e in TE_SHOWN:
        if e is None:
            fig.text(x + 0.015, te_y + te_h / 2, r"$\cdots$", ha="center",
                     va="center", fontsize=topo.FONT_SIZE + 2)
            x += 0.035
            continue
        glyph(fig, [x, te_y, te_w, te_h], econ.loc[e], highlight=(e == K))
        lab = r"$te_{N_e}$" if e == 750 else r"$te_{%d}$" % e
        fig.text(x + te_w / 2, te_y - 0.012, lab, ha="center", va="top",
                 fontsize=topo.FONT_SIZE - 1)
        te_x[e] = x + te_w / 2
        x += te_w + 0.02
    arrow(fig, (0.88, 0.645), (x - 0.01, te_y + te_h + 0.01), rad=-0.2)

    # ---- (c) pairing block
    fx0, fx1, fy0, fy1 = 0.04, 0.98, 0.02, 0.37
    fig.add_artist(FancyBboxPatch((fx0, fy0), fx1 - fx0, fy1 - fy0,
                                  boxstyle="round,pad=0,rounding_size=0.01",
                                  transform=fig.transFigure, fc="none", ec=DARK,
                                  lw=0.7, ls=(0, (3, 2))))
    arrow(fig, (te_x[K], te_y - 0.035), (te_x[K], fy1 + 0.003))

    names = [man.loc[K, s_] for s_ in TILE_ORDER]
    P = np.vstack([np.vstack(runs.positions(tp, n)) for n in names])
    xh = abs(P[:, 0]).max() * 1.15
    yh = max(abs(P[:, 1]).max() * 1.2, xh * 0.5)
    lim = ((-xh, xh), (-yh, yh))
    tw, gap = 0.098, 0.014
    th = tw * W * (yh / xh) / H
    ty = 0.05
    tile_top = {}
    x = fx0 + 0.02
    for idx, s_ in enumerate(TILE_ORDER):
        if idx in (4, 6):
            x += 0.012
        g = "AB" if s_[0] in "AB" else s_[0]
        ax = fig.add_axes([x, ty, tw, th])
        runs.cell_frame(ax, topo.COLORS[g])
        big, small = runs.positions(tp, man.loc[K, s_])
        runs.draw_layout(ax, big, small, lim)
        fig.text(x + tw / 2, ty - 0.008, s_.replace("_", " "), ha="center",
                 va="top", fontsize=topo.FONT_SIZE - 1, color=topo.COLORS[g])
        tile_top[s_] = (x + tw / 2, ty + th * 0.95)
        x += tw + gap

    cw_, ch_, cy = 0.04, 0.075, 0.26
    for g, h, targets in CHIPS:
        if g == "AB":
            mid = (tile_top["A_far"][0] + tile_top["B_near"][0]) / 2
            cx = mid + (-0.03 if h == "near" else 0.03)
        else:
            cx = tile_top[targets[0]][0]
        slot = {"AB": "A", "C": "C", "D": "D"}[g]
        chip(fig, [cx - cw_ / 2, cy, cw_, ch_], g, trip(man.loc[K, f"{slot}_{h}"]))
        fig.text(cx, cy + ch_ + 0.008, h, ha="center", va="bottom",
                 fontsize=topo.FONT_SIZE - 2, color=topo.COLORS[g])
        for t in targets:
            line(fig, (cx, cy), tile_top[t], topo.COLORS[g])
    arrow(fig, (0.39, 0.64), (0.3, fy1 + 0.003), color=topo.COLORS["AB"],
          lw=0.7, rad=0.25)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        try:
            fig.savefig(OUT_DIR / f"scenario_generation_sequential.{ext}",
                        dpi=topo.DPI_PNG, bbox_inches="tight", pad_inches=0.01)
        except OSError as e:
            print(f"  not written, file locked: {ext} ({e.strerror})")
    plt.close(fig)
    print(f"written to {OUT_DIR}")


LAYOUT = "sequential"                # "sequential", "simple" or "detailed"

if __name__ == "__main__":
    {"sequential": main_seq, "simple": main_simple, "detailed": main}[LAYOUT]()
