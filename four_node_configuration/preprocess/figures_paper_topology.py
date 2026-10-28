"""
Paper figures - topology generation with a per-archetype Sobol sequence (v5).

Reproduces exactly the candidate streams of generate_node_topology_paired.py
(same seeds, same pool, same feasibility test) and draws, per archetype group
(A/B, D, C):

    feasible_<g>          3-D box of the stated ranges, feasible-region
                          boundary surface, the 2*N_ECON accepted points
    accept_reject_<g>     first N_CAND Sobol candidates over the full box,
                          accepted (group colour) vs rejected (grey x)
    fill_<g>_n<n>         first n accepted points, progressive fill

Every panel is written as SVG (editable text) + PDF + PNG into OUT_DIR, plus
one combined preview PNG. All style choices sit in the STYLE block below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

import generate_node_topology_paired as v5
from generate_node_topology_pooled import (
    D_LL_RANGE, D_SL_RANGE, D_SS_RANGE, build_topology, unit_to_params,
)

# ------------------------------------------------------------------- style --

OUT_DIR = Path(r"C:\Users\Masse007\OneDrive - Universiteit Utrecht\Documents"
               r"\Research\FirstWork\Images\Methodology\Figures_paper"
               r"\topology_sobol")

MM = 1 / 25.4
FULL_WIDTH_MM = 190                  # Elsevier double column
PANEL_MM = 58                        # one 3-D box panel (3 fit a full width)
FONT = "Arial"
FONT_SIZE = 7
DPI_PNG = 600

COLORS = {                           # Okabe-Ito, colourblind safe
    "AB": "#0072B2",
    "D": "#E69F00",
    "C": "#CC79A7",
    "rejected": "#9A9A9A",
    "box": "#4D4D4D",
    "shadow": "#C8C8C8",
}
GROUP_TITLE = {"AB": "A / B", "D": "D", "C": "C"}
VIEW = dict(elev=22, azim=-58)       # camera, same for every panel
SURFACE_ALPHA = 0.18

N_CAND = 400                         # candidates shown in the 2-D accept/reject
GRID_N = 80                          # boundary-surface resolution per axis
SS_SCAN = 200                        # d_SS resolution of the column scan
PLANE_MM = (FULL_WIDTH_MM, 58)       # 2-D row of three projections

# --------------------------------------------------------------- sampling --

GROUP_SEED = {g: v5.SEED + 100 * i for i, g in enumerate(v5.GROUPS)}


def feasible(label, u):
    return build_topology(label, *unit_to_params(u)) is not None


def candidate_stream(group):
    """Sobol pool of the group, acceptance mask up to the 2*N_ECON-th accept."""
    label = v5.GROUPS[group]
    U = qmc.Sobol(d=3, scramble=True, seed=GROUP_SEED[group]).random(v5.POOL_SIZE)
    need = 2 * v5.N_ECON
    ok, n_acc = [], 0
    for u in U:
        f = feasible(label, u)
        ok.append(f)
        n_acc += f
        if n_acc == need:
            break
    ok = np.array(ok)
    return U[: len(ok)], ok


def to_km(U):
    lo = np.array([D_LL_RANGE[0], D_SL_RANGE[0], D_SS_RANGE[0]])
    hi = np.array([D_LL_RANGE[1], D_SL_RANGE[1], D_SS_RANGE[1]])
    return lo + U * (hi - lo)


def boundary_surfaces(group):
    """Per (d_LL, d_SL) column, lowest and highest feasible d_SS. Cells where
    the bound coincides with the box face are masked, so only the cut through
    the box is drawn."""
    label = v5.GROUPS[group]
    ll = np.linspace(*D_LL_RANGE, GRID_N)
    sl = np.linspace(*D_SL_RANGE, GRID_N)
    ss = np.linspace(*D_SS_RANGE, SS_SCAN)
    LL, SL = np.meshgrid(ll, sl, indexing="ij")
    lo = np.full(LL.shape, np.nan)
    hi = np.full(LL.shape, np.nan)
    tol = 1.5 * (ss[1] - ss[0])
    F = np.zeros((GRID_N, GRID_N, SS_SCAN), dtype=bool)
    for i, a in enumerate(ll):
        for j, b in enumerate(sl):
            f = [build_topology(label, a, b, c) is not None for c in ss]
            F[i, j] = f
            idx = np.flatnonzero(f)
            if idx.size == 0:
                continue
            if ss[idx[0]] > D_SS_RANGE[0] + tol:
                lo[i, j] = ss[idx[0]]
            if ss[idx[-1]] < D_SS_RANGE[1] - tol:
                hi[i, j] = ss[idx[-1]]
    return (LL, SL, lo, hi), (ll, sl, ss, F)


# ---------------------------------------------------------------- drawing --

def set_style():
    plt.rcParams.update({
        "font.family": FONT,
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
    })


def new_box(title=None):
    fig = plt.figure(figsize=(PANEL_MM * MM, PANEL_MM * MM))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.view_init(**VIEW)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlim(*D_LL_RANGE)
    ax.set_ylim(*D_SL_RANGE)
    ax.set_zlim(*D_SS_RANGE)
    ax.set_xticks([400, 950, 1500])
    ax.set_yticks([200, 350, 500])
    ax.set_zticks([100, 550, 1000])
    ax.set_xlabel(r"$d_{LL}$ [km]", labelpad=-6)
    ax.set_ylabel(r"$d_{SL}$ [km]", labelpad=-6)
    ax.set_zlabel(r"$d_{SS}$ [km]", labelpad=-7)
    ax.tick_params(pad=-3, length=2, width=0.4)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(COLORS["box"])
        axis.pane.set_linewidth(0.4)
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis.line.set_linewidth(0.5)
    ax.grid(False)
    if title:
        ax.set_title(title, pad=-2)
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.04, top=0.96)
    return fig, ax


def draw_surface(ax, surf, color, alpha=SURFACE_ALPHA):
    LL, SL, lo, hi = surf
    for Z in (lo, hi):
        if np.all(np.isnan(Z)):
            continue
        ax.plot_surface(LL, SL, np.ma.masked_invalid(Z), color=color,
                        alpha=alpha, linewidth=0, antialiased=False,
                        shade=False, rstride=1, cstride=1, zorder=1)


def draw_points(ax, P, color, size=2.0, alpha=0.9, shadow=False):
    if shadow:
        ax.scatter(P[:, 0], P[:, 1], np.full(len(P), D_SS_RANGE[0]),
                   s=size * 0.6, c=COLORS["shadow"], lw=0, depthshade=False,
                   zorder=2)
        for p in P:
            ax.plot([p[0], p[0]], [p[1], p[1]], [D_SS_RANGE[0], p[2]],
                    color=COLORS["shadow"], lw=0.3, zorder=2)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=size, c=color, lw=0,
               alpha=alpha, depthshade=False, zorder=3)


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        try:
            fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=DPI_PNG)
        except OSError as e:                   # file open/locked (OneDrive, viewer)
            print(f"  not written, file locked: {name}.{ext} ({e.strerror})")
    plt.close(fig)


# ------------------------------------------------------------------ panels --

N_SHOW = 400                         # Sobol candidates shown per group
FACE_ALPHA = 0.10                    # feasible region on the box faces
CUT_ALPHA = 0.30                     # boundary surface inside the box
SHOW_TEXT = False                    # rule, region labels, legend (else: caption)
SHOW_CUT_LABELS = False              # boundary equations on the surfaces

# Boundary labels (text, point on the surface) and region labels (text, point).
# Positions in km, tuned by eye for VIEW.
ANNOT = {
    "AB": {
        "cuts": [(r"$d_{SS} = 2\,d_{SL}$", (1500, 330, 660))],
        "regions": [("feasible\naccept", (1000, 480, 200)),
                    ("infeasible\nreject", (450, 110, 950))],
        "rule": r"accept if $d_{SS} \leq 2\,d_{SL}$",
    },
    "D": {
        "cuts": [(r"$d_{SS} = d_{LL} - 2\,d_{SL}$", (1450, 380, 690)),
                 (r"$d_{SS} = d_{LL} + 2\,d_{SL}$", (420, 290, 1000))],
        "regions": [("feasible\naccept", (750, 400, 620)),
                    ("infeasible\nreject", (1400, 110, 150))],
        "rule": r"accept if $|d_{LL} - d_{SS}| \leq 2\,d_{SL}$",
    },
    "C": {
        "cuts": [],
        "regions": [("feasible\naccept", None),   # None = centre of accepted points
                    ("infeasible\nreject", (1400, 300, 150))],
        "rule": r"accept if $d_{near}/d_{far} \geq 0.9$ and $d_{SS} \leq 2\,d_{SL}$",
    },
}


def draw_box_faces(ax, vol, color, step=4):
    """Feasible region where it meets the six faces of the box."""
    ll, sl, ss, F = vol
    ssd, Fd = ss[::step], F[:, :, ::step]
    faces = [
        (np.meshgrid(ll, sl, indexing="ij"), ssd[0], Fd[:, :, 0], "z"),
        (np.meshgrid(ll, sl, indexing="ij"), ssd[-1], Fd[:, :, -1], "z"),
        (np.meshgrid(sl, ssd, indexing="ij"), ll[0], Fd[0], "x"),
        (np.meshgrid(sl, ssd, indexing="ij"), ll[-1], Fd[-1], "x"),
        (np.meshgrid(ll, ssd, indexing="ij"), sl[0], Fd[:, 0, :], "y"),
        (np.meshgrid(ll, ssd, indexing="ij"), sl[-1], Fd[:, -1, :], "y"),
    ]
    for (A, B), c, M, normal in faces:
        if not M.any():
            continue
        C = np.full(A.shape, float(c))
        if normal == "z":
            X, Y, Z = A, B, np.where(M, C, np.nan)
        elif normal == "x":
            X, Y, Z = C, A, np.where(M, B, np.nan)
        else:
            X, Y, Z = A, C, np.where(M, B, np.nan)
        ax.plot_surface(X, Y, Z, color=color, alpha=FACE_ALPHA, linewidth=0,
                        antialiased=False, shade=False, zorder=1)


def panel_sampling(g, P_cand, ok, surf, vol, share):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    P, m = P_cand[:N_SHOW], ok[:N_SHOW]
    a = ANNOT[g]
    fig, ax = new_box()
    fig.set_size_inches(PANEL_MM * MM, PANEL_MM * (1.3 if SHOW_TEXT else 1.05) * MM)
    fig.text(0.5, 0.99, GROUP_TITLE[g], ha="center", va="top",
             fontsize=FONT_SIZE + 1, fontweight="bold")
    if SHOW_TEXT:
        fig.text(0.5, 0.95, a["rule"], ha="center", va="top", fontsize=FONT_SIZE - 1)
    draw_box_faces(ax, vol, COLORS[g])
    draw_surface(ax, surf, COLORS[g], alpha=CUT_ALPHA)
    ax.scatter(P[~m, 0], P[~m, 1], P[~m, 2], marker="x", s=4, lw=0.4,
               c=COLORS["rejected"], depthshade=False, zorder=2)
    ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=5, lw=0.2, edgecolors="white",
               c=COLORS[g], depthshade=False, zorder=3)
    box = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)
    for txt, (x, y, z) in (a["cuts"] if SHOW_CUT_LABELS else []):
        ax.text(x, y, z, txt, color=COLORS[g], fontsize=FONT_SIZE - 1,
                ha="center", va="center", zorder=5, bbox=box)
    if not SHOW_TEXT:
        fig.subplots_adjust(left=0.0, right=0.9, bottom=0.03, top=0.95)
        save(fig, f"sampling_3d_{g}")
        return
    for txt, xyz in a["regions"]:
        if xyz is None:
            xyz = np.median(P_cand[ok], axis=0)
        col = COLORS[g] if txt.startswith("feasible") else "#555555"
        ax.text(*xyz, txt, color=col, fontsize=FONT_SIZE - 1, linespacing=0.9,
                fontweight="bold", ha="center", va="center", zorder=5, bbox=box)
    handles = [
        Patch(facecolor=COLORS[g], alpha=0.35, edgecolor="none",
              label=f"feasible region ({share:.1f}% of box)"),
        Line2D([], [], ls="", marker="o", ms=3, mfc=COLORS[g], mec="white",
               mew=0.2, label=f"accepted ({m.sum()})"),
        Line2D([], [], ls="", marker="x", ms=3, mec=COLORS["rejected"],
               mew=0.5, label=f"rejected ({(~m).sum()})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
               fontsize=FONT_SIZE - 1, handlelength=1.2, borderaxespad=0.2)
    fig.subplots_adjust(left=0.0, right=0.9, bottom=0.2, top=0.92)
    save(fig, f"sampling_3d_{g}")


def preview(names, out):
    import matplotlib.image as mpimg
    groups = list(v5.GROUPS)
    fig, axs = plt.subplots(len(names), len(groups),
                            figsize=(len(groups) * 3.0, len(names) * 3.9),
                            squeeze=False)
    for r, pat in enumerate(names):
        for c, g in enumerate(groups):
            axs[r, c].imshow(mpimg.imread(OUT_DIR / f"{pat.format(g=g)}.png"))
            axs[r, c].axis("off")
    fig.tight_layout(pad=0.2)
    try:
        fig.savefig(OUT_DIR / out, dpi=200)
    except OSError as e:                       # file open/locked (OneDrive)
        print(f"preview not written: {e}")
    plt.close(fig)


# ------------------------------------------------------------- 2-D panels --

PLANES = [  # (x index, y index, collapsed axis of F)
    (0, 1, 2),
    (0, 2, 1),
    (1, 2, 0),
]
AX_LABEL = [r"$d_{LL}$ [km]", r"$d_{SL}$ [km]", r"$d_{SS}$ [km]"]
AX_RANGE = [D_LL_RANGE, D_SL_RANGE, D_SS_RANGE]
AX_TICKS = [[400, 950, 1500], [100, 300, 500], [100, 550, 1000]]


def new_planes(title=None):
    fig, axs = plt.subplots(1, 3, figsize=(PLANE_MM[0] * MM, PLANE_MM[1] * MM))
    for ax, (xi, yi, _) in zip(axs, PLANES):
        ax.set_xlim(*AX_RANGE[xi])
        ax.set_ylim(*AX_RANGE[yi])
        ax.set_xticks(AX_TICKS[xi])
        ax.set_yticks(AX_TICKS[yi])
        ax.set_xlabel(AX_LABEL[xi])
        ax.set_ylabel(AX_LABEL[yi], labelpad=1)
        ax.set_box_aspect(1)
        ax.tick_params(length=2, width=0.4, pad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    if title:
        fig.suptitle(title, y=0.97, fontsize=FONT_SIZE + 1)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.86, wspace=0.45)
    return fig, axs


def draw_density(axs, vol, color, alpha=1.0):
    """Feasible fraction along the collapsed axis = projected density of a
    sample uniform over the feasible region (up to a constant)."""
    from matplotlib.colors import LinearSegmentedColormap
    ll, sl, ss, F = vol
    grids = [ll, sl, ss]
    cmap = LinearSegmentedColormap.from_list("g", ["#FFFFFF", color])
    for ax, (xi, yi, ci) in zip(axs, PLANES):
        frac = F.mean(axis=ci)                     # indexed [x, y] after collapse
        if xi > yi:
            frac = frac.T
        frac = frac / frac.max()                   # relative density per plane
        X, Y = np.meshgrid(grids[xi], grids[yi], indexing="ij")
        ax.contourf(X, Y, frac, levels=np.linspace(0.02, 1, 6), cmap=cmap,
                    alpha=alpha * 0.6, antialiased=True, zorder=0)
        ax.contour(X, Y, frac, levels=[0.02], colors=[color],
                   linewidths=0.6, zorder=1)


def scatter_planes(axs, P, **kw):
    for ax, (xi, yi, _) in zip(axs, PLANES):
        ax.scatter(P[:, xi], P[:, yi], **kw)


def panel2d_feasible(g, P_acc, vol, share):
    fig, axs = new_planes(f"{GROUP_TITLE[g]}: {len(P_acc)} accepted points "
                          f"({share:.1f}% of box)")
    draw_density(axs, vol, COLORS[g])
    scatter_planes(axs, P_acc, s=0.6, lw=0, c=COLORS[g], alpha=0.7, zorder=3)
    save(fig, f"2d_feasible_{g}")


def panel2d_accept_reject(g, P_cand, ok, vol):
    P, m = P_cand[:N_CAND], ok[:N_CAND]
    fig, axs = new_planes(f"{GROUP_TITLE[g]}: first {N_CAND} Sobol candidates, "
                          f"accepted {m.sum()} / rejected {(~m).sum()}")
    draw_density(axs, vol, COLORS[g], alpha=0.6)
    scatter_planes(axs, P[~m], marker="x", s=5, lw=0.4, c=COLORS["rejected"], zorder=2)
    scatter_planes(axs, P[m], s=5, lw=0, c=COLORS[g], zorder=3)
    save(fig, f"2d_accept_reject_{g}")


def main():
    set_style()
    for g in v5.GROUPS:
        U, ok = candidate_stream(g)
        P_all = to_km(U)
        share = 100 * ok.mean()
        print(f"{g}: {ok.sum()} accepted of {len(ok)} candidates ({share:.1f}%)")
        surf, vol = boundary_surfaces(g)
        panel_sampling(g, P_all, ok, surf, vol, share)
        panel2d_accept_reject(g, P_all, ok, vol)
    preview(["sampling_3d_{g}"], "_preview_3d.png")
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
