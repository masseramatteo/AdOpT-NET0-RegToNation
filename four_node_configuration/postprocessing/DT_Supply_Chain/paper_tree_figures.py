"""
paper_tree_figures.py
=====================
Step 2 of the paper decision-tree figures: draw publication figures from the JSON
written by paper_tree_data.py. Needs only matplotlib + numpy (no network drive, no refit).

Eight styles per tree, all at full page width (7.1 in), fonts 5.5-7 pt at final size:

    A_tree_vertical     classic top-down tree, leaf cards with network + supply mix
    B_tree_horizontal   root on the left, leaves as rows of a small table
    C_tree_flow         alluvial tree: ribbon width = number of runs, colour = class,
                        so the reader sees how each split sorts the scenarios
    D_rule_table        one row per leaf, one column per driver: the allowed range of
                        each driver as a strip (rows sorted by class)
    F_tree_cards        A in the look of tree.html: curved branches, pill labels, leaf cards
    G_tree_cards_weighted   F + branch width = runs, type bar in each split
    H_tree_cards_staggered  F with leaves on two rows (wide cards, labelled clusters)
    E_partition_map     the runs in the plane of the two main drivers, faceted by the
                        root split, with the leaf regions drawn on top

Colour language (supply-mix trees): green = self-served, blue = transported,
orange = imported; a class takes the colour of its dominant supply mode, so class
colours and mix bars read the same way. Clusters as in the archetype figure of the
paper: large = navy circle, small = yellow square.

Usage
-----
    python paper_tree_figures.py                       # every JSON in <PAPER_DIR>/_data
    python paper_tree_figures.py sysmix_sysmix_clf_k03_supply_L8
    python paper_tree_figures.py <case> --only A C     # some styles only
    python paper_tree_figures.py <case> --rename "Self + transport=Hub-supplied"

Output: <PAPER_DIR>/<case>/<style>.png (400 dpi) / .pdf / .svg and caption.txt
(numbers to quote in the caption: runs, accuracy, CI, kappa, baseline).
"""
import argparse
import glob
import json
import os
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path

PAPER_DIR = (r"C:\Users\Masse007\OneDrive - Universiteit Utrecht\Documents\Research\FirstWork"
             r"\Images\Results\Decision_trees")
DATA_DIR = os.path.join(PAPER_DIR, "_data")

# ---------------------------------------------------------------- style -----------------------------------------
FULL_W = 7.1  # page width, inches
NAVY, YELLOW = "#161D41", "#FFCD00"  # UU colours of the archetype figure (large / small cluster)
INK, GREY, LINE, TRACK = NAVY, "#6b7280", "#aab1bd", "#e6e8ec"
SELF, TRANSP, IMPORT, HIGHP = "#3a9b7b", "#3d6fb0", "#e0902c", "#b91c1c"
MIX_COLOR = {"self_served": SELF, "transported": TRANSP, "imported": IMPORT}
MIX_LABEL = {"self_served": "self-served", "transported": "transported", "imported": "imported"}
CLASS_COLOR = {"Self-sufficient": SELF, "Self + transport": TRANSP, "Import-reliant": IMPORT,
               "Import-dominated": "#b3541e", "Import + transport": "#7b6aa8", "Mixed supply": "#8d96a3"}
PALETTE = ["#3d6fb0", "#e0902c", "#3a9b7b", "#7b6aa8", "#b3541e", "#8d96a3", "#c0507a", "#2f3d7a"]
FREQ_SOLID, FREQ_DASH = 0.6, 0.3  # pipeline drawn solid / dashed if built in at least this share of the leaf's runs
FREQ_RARE = 0.1  # card styles (F, G, H) also draw a faint dotted line down to this share, as tree.html does
CARD_EDGE, SHADOW = "#d9dce3", "#0b1030"
LINKS = {"L1L2": ("L1", "L2"), "L1S1": ("L1", "S1"), "L1S2": ("L1", "S2"),
         "L2S1": ("L2", "S1"), "L2S2": ("L2", "S2"), "S1S2": ("S1", "S2")}
TAG = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88)  # label tag on top of the ribbons

for _f in glob.glob(r"C:\Windows\Fonts\OpenSans-*.ttf"):
    font_manager.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Open Sans", "Arial", "DejaVu Sans"],
                     "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.6,
                     "mathtext.default": "regular"})


# ---------------------------------------------------------------- data ------------------------------------------
def load(case, rename=None):
    path = case if os.path.exists(case) else os.path.join(DATA_DIR, case + ".json")
    d = json.load(open(path, encoding="utf-8"))
    rename = rename or {}
    for i, c in enumerate(d["classes"]):
        c["name"] = rename.get(c["name"], c["name"])
        c["color"] = CLASS_COLOR.get(c["name"], PALETTE[i % len(PALETTE)])
    for F in d["features"].values():
        F["label"] = F["label"].replace("₂", "$_2$")  # Open Sans has no subscript-two glyph
    leaves, splits = [], []

    def walk(n, depth, bounds):
        n["depth"], n["bounds"] = depth, dict(bounds)
        n["cls"] = int(np.argmax(n["counts"]))
        n["purity"] = max(n["counts"]) / n["n"]
        if n["type"] == "leaf":
            n["pos"] = len(leaves)
            leaves.append(n)
            return
        splits.append(n)
        lo, hi = bounds.get(n["feature"], (-np.inf, np.inf))
        walk(n["left"], depth + 1, {**bounds, n["feature"]: (lo, n["thr"])})
        walk(n["right"], depth + 1, {**bounds, n["feature"]: (n["thr"], hi)})
        n["pos"] = (n["left"]["pos"] + n["right"]["pos"]) / 2

    walk(d["tree"], 0, {})
    d["leaves"], d["splits"] = leaves, splits
    d["levels"] = max(s["depth"] for s in splits) + 1
    return d


def fmt_thr(v, unit=""):
    s = f"{v:,.0f}" if abs(v) >= 100 else (f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:.1f}")
    return f"{s} {unit}".strip()


def edge_text(d, n, side):
    f = d["features"][n["feature"]]
    return ("≤ " if side == "left" else "> ") + fmt_thr(n["thr"], f["unit"]), f["lo_word" if side == "left" else "hi_word"]


def wrap(s, width):
    """Greedy wrap on spaces (and after hyphens) to at most `width` characters per line."""
    words, lines, cur = s.replace("-", "- ").split(" "), [], ""
    ln = lambda t: len(t.replace("$_2$", "2"))
    for w in words:
        glue = "" if cur.endswith("-") or not cur else " "
        if cur and ln(cur) + len(glue) + ln(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur += glue + w
    return "\n".join(lines + [cur])


def split_labels(d, width):
    """Wrapped driver label of every split and the half height of a box that fits the tallest one."""
    labs = {n["id"]: wrap(d["features"][n["feature"]]["label"], width) for n in d["splits"]}
    return labs, 0.05 + 0.045 * max(t.count("\n") + 1 for t in labs.values())


def leaf_sub(d, n, sep=" · "):
    if d["meta"]["regression"]:
        return f"n = {n['n']:,}{sep}± {n['info']['spread']:.2f}"
    return f"n = {n['n']:,}{sep}{n['purity']:.0%} pure"


# ---------------------------------------------------------------- drawing helpers -------------------------------
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def slug(t):
    return re.sub(r"[^A-Za-z0-9]+", "_", t.replace("$_2$", "2")).strip("_")


def name_new(ax, name):
    """Name (svg id) every object drawn since the last call, so it shows under this name in the Layers panel of
    Illustrator / Inkscape. Does not change how the figure looks."""
    k = 0
    for a in ax.get_children():
        if a.get_gid() is None:
            k += 1
            a.set_gid(f"{name}__{k:02d}")


def name_rest(fig):
    """Fallback names for whatever a style did not name itself: texts by their content, shapes by their kind."""
    count = {}
    for ax in fig.axes:
        for a in ax.get_children():
            if a.get_gid() is not None:
                continue
            if isinstance(a, matplotlib.text.Text):
                if not a.get_text().strip():
                    continue
                base = "text_" + slug(a.get_text())[:40]
            else:
                base = type(a).__name__.lower()
            count[base] = count.get(base, 0) + 1
            a.set_gid(f"{base}__{count[base]:02d}")


def save(fig, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    name_rest(fig)
    for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
        for attempt in range(5):  # OneDrive locks a file for a moment while it syncs
            try:
                fig.savefig(out + ext, **kw)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2)
    plt.close(fig)
    print("saved", out + ".png")


def draw_network(ax, info, sc, cx, cy, r, z=5):
    """Typical network of a leaf. L1 top, L2 bottom, S1 left, S2 right; r = half extent in inches."""
    k = r / 0.30
    P = {"L1": (cx, cy + r), "L2": (cx, cy - r), "S1": (cx - r, cy), "S2": (cx + r, cy)}
    for key, (a, b) in LINKS.items():
        freq, cap, hp = info[key]
        if freq < FREQ_DASH:
            continue
        ax.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]], color=HIGHP if hp > 0.5 else NAVY,
                lw=(0.6 + 2.4 * cap / sc["pipe_max"]) * max(k, 0.6), zorder=z,
                ls="-" if freq >= FREQ_SOLID else (0, (1.6, 1.2)), solid_capstyle="round")
    for n in ("L1", "L2"):
        rad = r * (0.17 + 0.27 * info[f"cap_{n}"] / sc["large_elzr_max"])
        imp = info[f"imp_{n}"] / sc["import_max"]
        if imp > 0.03:
            sgn = 1 if n == "L1" else -1
            ax.add_patch(FancyArrowPatch((P[n][0] - 1.05 * r, P[n][1] + sgn * 0.42 * r), (P[n][0] - rad, P[n][1]),
                                         arrowstyle="-|>", mutation_scale=(3.5 + 4.5 * imp) * max(k, 0.6),
                                         lw=(0.5 + 1.7 * imp) * max(k, 0.6), color=IMPORT, zorder=z + 1,
                                         shrinkA=0, shrinkB=0.5))
        ax.add_patch(Circle(P[n], rad, fc=NAVY, ec="white", lw=0.5, zorder=z + 2))
    for n in ("S1", "S2"):
        s = r * (0.13 + 0.19 * info[f"cap_{n}"] / sc["small_elzr_max"])
        ax.add_patch(Rectangle((P[n][0] - s, P[n][1] - s), 2 * s, 2 * s, fc=YELLOW, ec=NAVY, lw=0.5, zorder=z + 2))


def draw_mix(ax, d, mix, x, y, w, h, fs=5.0, z=4):
    """Stacked supply-mix bar; percentages inside the segments that are wide enough."""
    x0 = x
    for name, share in zip(d["meta"]["mix_order"], mix):
        ax.add_patch(Rectangle((x0, y), w * share, h, fc=MIX_COLOR[name], ec="white", lw=0.4, zorder=z))
        if w * share > 0.012 * fs * 2.4:
            ax.text(x0 + w * share / 2, y + h / 2 - 0.004, f"{100 * share:.0f}", ha="center", va="center",
                    fontsize=fs, color="white", weight="bold", zorder=z + 1)
        x0 += w * share


def chip(ax, x, y, w, h, text, color, fs=6.3, z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={min(h, w) * 0.22}",
                                fc=color, ec="none", zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="white", weight="bold",
            zorder=z + 1, linespacing=1.0)


def split_box(ax, x, y, label, hw, hh, fs=6.3, z=6):
    ax.add_patch(FancyBboxPatch((x - hw, y - hh), 2 * hw, 2 * hh, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="white", ec=NAVY, lw=0.8, zorder=z))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=INK, weight="bold", zorder=z + 1,
            linespacing=1.0)


def legend(fig, ax, d, x, y, width, classes=False, network=True, mix=True, fs=5.8, row=0.15, rare=False,
           clusters=True, freq=None):
    """Flowing legend, top-left at (x, y); returns the y of its bottom."""
    items = []
    if classes:
        items += [("patch", c["color"], c["name"]) for c in d["classes"]] + [("gap",)]
    if mix:
        items += [("text", None, "Supply mix (% of demand):")]
        items += [("patch", MIX_COLOR[m], MIX_LABEL[m]) for m in d["meta"]["mix_order"]] + [("gap",)]
    if network:
        fs_, fd_, fr_ = freq or (FREQ_SOLID, FREQ_DASH, FREQ_RARE if rare else None)
        drawn = [lf["info"][k] for lf in d["leaves"] for k in LINKS if lf["info"][k][0] >= fd_]
        if clusters:
            items += [("hub", None, "large cluster"), ("small", None, "small cluster (size = electrolyzer capacity)"),
                      ("gap",)]
        items += [("solid", NAVY, f"pipeline in ≥{fs_:.0%} of runs"),
                  ("dash", NAVY, f"{fd_:.0%}–{fs_:.0%} (width = capacity)")]
        if rare and fr_ is not None:
            items += [("rare", "#9aa1ae", f"{fr_:.0%}–{fd_:.0%}")]
        if any(v[2] > 0.5 for v in drawn):
            items += [("solid", HIGHP, "high pressure")]
        items += [("arrow", IMPORT, "H$_2$ import")]
    rend = fig.canvas.get_renderer()
    cx, cy, sym = x, y - row / 2, 0.17
    for it in items:
        if it[0] == "gap":
            cx += 0.10
            continue
        kind, col, lab = it
        t = ax.text(0, 0, lab, fontsize=fs, color=GREY if kind != "text" else INK, va="center")
        tw = t.get_window_extent(rend).width / fig.dpi
        need = tw + (0 if kind == "text" else sym + 0.05) + 0.12
        if cx + need > x + width + 0.12 and cx > x:
            cx, cy = x, cy - row
        tx = cx
        if kind == "patch":
            ax.add_patch(Rectangle((cx, cy - 0.035), 0.13, 0.07, fc=col, ec="none"))
        elif kind == "hub":
            ax.add_patch(Circle((cx + 0.07, cy), 0.04, fc=NAVY, ec="none"))
        elif kind == "small":
            ax.add_patch(Rectangle((cx + 0.035, cy - 0.035), 0.07, 0.07, fc=YELLOW, ec=NAVY, lw=0.5))
        elif kind == "rare":
            ax.plot([cx, cx + sym], [cy, cy], color=col, lw=0.6, ls=(0, (1, 1.6)))
        elif kind in ("solid", "dash"):
            ax.plot([cx, cx + sym], [cy, cy], color=col, lw=1.6, ls="-" if kind == "solid" else (0, (1.6, 1.2)))
        elif kind == "arrow":
            ax.add_patch(FancyArrowPatch((cx, cy), (cx + sym, cy), arrowstyle="-|>", mutation_scale=6, lw=1.2,
                                         color=col, shrinkA=0, shrinkB=0))
        if kind != "text":
            tx = cx + sym + 0.05
        t.set_position((tx, cy - 0.003))
        cx += need
    return cy - row / 2


# ---------------------------------------------------------------- A: vertical tree ------------------------------
def fig_vertical(d, out):
    nl, levels = len(d["leaves"]), d["levels"]
    card_h, leg_h = 1.78, 0.42
    dy = 0.60 + 2 * (split_labels(d, 14)[1] - 0.14)  # taller boxes -> more room between levels
    leaf_top = leg_h + card_h
    H = leaf_top + 0.50 + (levels - 1) * dy + 0.24
    fig, ax = canvas(FULL_W, H)
    x0, x1 = 0.06, FULL_W - 0.06
    slot = (x1 - x0) / nl
    xs = lambda p: x0 + slot * (p + 0.5)
    ys = lambda dep: H - 0.24 - dep * dy
    labs, hh = split_labels(d, 14)
    hw = min(0.46, slot * 0.53)

    for n in d["splits"]:
        x, y = xs(n["pos"]), ys(n["depth"])
        ybar = y - hh - 0.07
        for side in ("left", "right"):
            c = n[side]
            cx = xs(c["pos"])
            cy = ys(c["depth"]) + hh if c["type"] == "split" else leaf_top
            ax.plot([x, x, cx, cx], [y - hh, ybar, ybar, cy], color=LINE, lw=0.9, zorder=1,
                    solid_joinstyle="round")
            thr, word = edge_text(d, n, side)
            ax.text(cx, ybar - 0.035, thr, ha="center", va="top", fontsize=5.8, color=INK, zorder=3,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
            if word:
                ax.text(cx, ybar - 0.135, word, ha="center", va="top", fontsize=5.4, color=GREY, style="italic",
                        zorder=3, bbox=dict(boxstyle="square,pad=0.10", fc="white", ec="none"))
        split_box(ax, x, y, labs[n["id"]], hw, hh)

    w = slot - 0.08
    for lf in d["leaves"]:
        c = d["classes"][lf["cls"]]
        cx = xs(lf["pos"])
        lx = cx - w / 2
        ax.add_patch(FancyBboxPatch((lx, leaf_top - card_h), w, card_h, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="white", ec=c["color"], lw=1.0, zorder=2))
        chip(ax, lx, leaf_top - 0.32, w, 0.32, wrap(c["name"], 11), c["color"], fs=6.3, z=3)
        draw_network(ax, lf["info"], d["scale"], cx, leaf_top - 0.84, 0.27)
        draw_mix(ax, d, lf["mix"], lx + 0.06, leaf_top - 1.45, w - 0.12, 0.12)
        ax.text(cx, leaf_top - 1.55, leaf_sub(d, lf, "\n"), ha="center", va="top", fontsize=5.6, color=GREY,
                linespacing=1.15, zorder=4)
    legend(fig, ax, d, 0.08, leg_h - 0.06, FULL_W - 0.16)
    save(fig, out)


# ---------------------------------------------------------------- B: horizontal tree ----------------------------
def table_columns(ax, d, x, y, fs=5.8):
    """Column headers of the leaf table used by B, C and D; x = left of the name column."""
    for dx, lab in ((0.0, "Supply type" if d["meta"]["group"] == "sysmix" else "Type"), (0.93, "Network"),
                    (1.42, "Supply mix (%)"), (2.28, "Runs")):
        ax.text(x + dx, y, lab, fontsize=fs, color=GREY, va="center")


def table_row(ax, d, lf, x, cy, h, with_chip=True):
    c = d["classes"][lf["cls"]]
    if with_chip:
        chip(ax, x, cy - min(h, 0.30) / 2, 0.86, min(h, 0.30), c["name"], c["color"], fs=6.0)
    draw_network(ax, lf["info"], d["scale"], x + 1.12, cy, min(0.15, h * 0.31))
    draw_mix(ax, d, lf["mix"], x + 1.42, cy - 0.06, 0.78, 0.12)
    ax.text(x + 2.28, cy, leaf_sub(d, lf, "\n"), fontsize=5.5, color=GREY, va="center", linespacing=1.15)


def fig_horizontal(d, out):
    nl, levels = len(d["leaves"]), d["levels"]
    hr, leg_h, head = 0.48, 0.42, 0.26
    H = leg_h + nl * hr + head
    fig, ax = canvas(FULL_W, H)
    ys = lambda p: H - head - hr * (p + 0.5)
    tx = FULL_W - 2.78  # left edge of the table
    hw = 0.31
    labs, hh = split_labels(d, 11)
    step = (tx - 0.45 - 0.06 - 2 * hw) / max(levels - 1, 1)  # 0.45: room for the labels after the last split
    xs = lambda dep: 0.06 + hw + dep * step

    for n in d["splits"]:
        x, y = xs(n["depth"]), ys(n["pos"])
        xe = x + hw + 0.06
        for side in ("left", "right"):
            c = n[side]
            cy = ys(c["pos"])
            cx = xs(c["depth"]) - hw if c["type"] == "split" else tx - 0.05
            ax.plot([x + hw, xe, xe, cx], [y, y, cy, cy], color=LINE, lw=0.9, zorder=1, solid_joinstyle="round")
            thr, word = edge_text(d, n, side)
            ax.text(xe + 0.03, cy + 0.025, thr, ha="left", va="bottom", fontsize=5.6, color=INK, zorder=3)
            if word:
                room = cx - xe > 0.9
                ax.text(xe + 0.03, cy - 0.03, word if room else wrap(word, 8), ha="left", va="top", fontsize=5.3,
                        color=GREY, style="italic", zorder=3, linespacing=0.95)
        split_box(ax, x, y, labs[n["id"]], hw, hh, fs=6.0)

    table_columns(ax, d, tx, H - head / 2)
    for i, lf in enumerate(d["leaves"]):
        cy = ys(lf["pos"])
        if i:
            ax.plot([tx, FULL_W - 0.06], [cy + hr / 2] * 2, color=TRACK, lw=0.6, zorder=0)
        table_row(ax, d, lf, tx, cy, hr - 0.1)
    legend(fig, ax, d, 0.08, leg_h - 0.06, FULL_W - 0.16)
    save(fig, out)


# ---------------------------------------------------------------- C: flow (alluvial) tree -----------------------
def band(ax, x0, x1, a0, b0, a1, b1, color, curve=0.85, z=2):
    """Ribbon from the vertical interval [a0, b0] at x0 to [a1, b1] at x1 (curved part `curve` inches long)."""
    xc = min(x1, x0 + curve)
    xm = (x0 + xc) / 2
    verts = [(x0, b0), (xm, b0), (xm, b1), (xc, b1), (x1, b1), (x1, a1), (xc, a1), (xm, a1), (xm, a0), (x0, a0),
             (x0, b0)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.LINETO, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), fc=color, ec="none", alpha=0.88, zorder=z))


def fig_flow(d, out):
    leaves, levels, N = d["leaves"], d["levels"], d["tree"]["n"]
    s = 2.7 / N  # inches per run
    hmin, leg_h, head = 0.40, 0.42, 0.26
    # leaf slots: at least hmin tall; wider gap between leaves that do not share a parent
    parent = {}
    for sp in d["splits"]:
        parent[sp["left"]["id"]] = parent[sp["right"]["id"]] = sp["id"]
    gaps = [0.07 if parent[a["id"]] == parent[b["id"]] else 0.15 for a, b in zip(leaves, leaves[1:])]
    slots = [max(lf["n"] * s, hmin) for lf in leaves]
    H = leg_h + sum(slots) + sum(gaps) + head + 0.04
    fig, ax = canvas(FULL_W, H)
    y = H - head - 0.04
    for lf, h, g in zip(leaves, slots, gaps + [0]):
        lf["yc"] = y - h / 2
        y -= h + g
    tx = FULL_W - 2.78
    xl = tx - 0.16  # leaf bars
    hw = 0.36
    labs, hh = split_labels(d, 11)
    step = (xl - 0.06 - hw) / levels
    xs = lambda n: xl if n["type"] == "leaf" else 0.06 + hw + n["depth"] * step
    bw = 0.035

    def centre(n):
        if n["type"] == "split":
            n["yc"] = (centre(n["left"]) * n["left"]["n"] + centre(n["right"]) * n["right"]["n"]) / n["n"]
        return n["yc"]

    centre(d["tree"])
    for n in d["splits"]:
        x, top = xs(n), n["yc"] + n["n"] * s / 2
        y0 = top
        for side in ("left", "right"):
            c = n[side]
            cx = xs(c)
            y1 = c["yc"] + c["n"] * s / 2
            pa, ca = y0, y1
            for k, cnt in enumerate(c["counts"]):
                if cnt:
                    band(ax, x, cx, pa - cnt * s, pa, ca - cnt * s, ca, d["classes"][k]["color"])
                pa, ca = pa - cnt * s, ca - cnt * s
            thr, word = edge_text(d, n, side)
            lab = thr + (f" · {word}" if word else "")
            if c["type"] == "leaf":  # on the ribbon, just before the leaf
                ly = c["yc"]
                hidden = xl - 0.09 - 0.047 * len(lab) < x + hw + 0.04 and abs(ly - n["yc"]) < hh + 0.07
                if hidden:  # the parent's box would cover it: move it just above / below the box
                    ly = n["yc"] + (hh + 0.09) * (1 if side == "left" else -1)
                ax.text(xl - 0.09, ly, lab, ha="right", va="center", fontsize=5.6, color=INK, zorder=7, bbox=TAG)
            else:  # above the box of the next split
                ax.text(cx, c["yc"] + hh + 0.05, lab, ha="right", va="bottom", fontsize=5.6, color=INK,
                        zorder=9, bbox=TAG)
            y0 -= c["n"] * s
        ax.add_patch(Rectangle((x - bw / 2, n["yc"] - n["n"] * s / 2), bw, n["n"] * s, fc=NAVY, ec="none", zorder=5))
        split_box(ax, x, n["yc"], labs[n["id"]], hw, hh, fs=6.0, z=8)

    table_columns(ax, d, tx, H - head / 2)
    for lf in leaves:
        c = d["classes"][lf["cls"]]
        ax.add_patch(Rectangle((xl - bw / 2, lf["yc"] - lf["n"] * s / 2), bw, lf["n"] * s, fc=NAVY, ec="none",
                               zorder=5))
        ax.text(tx, lf["yc"], wrap(c["name"], 16), fontsize=6.3, color=c["color"], weight="bold", va="center",
                linespacing=1.0)
        table_row(ax, d, lf, tx, lf["yc"], hmin - 0.04, with_chip=False)
    yb = legend(fig, ax, d, 0.08, leg_h - 0.06, FULL_W - 0.16)
    ax.text(FULL_W - 0.08, max(yb, 0.02) + 0.07, "ribbon width = number of runs, colour = type of the run",
            fontsize=5.8, color=GREY, ha="right", va="center")
    save(fig, out)


# ---------------------------------------------------------------- D: rule table ---------------------------------
def fig_rules(d, out):
    feats = sorted(d["features"], key=lambda f: -d["features"][f]["importance"])
    leaves = sorted(d["leaves"], key=lambda lf: (-lf["cls"], -lf["n"]))
    hr, leg_h, head = 0.40, 0.42, 0.50
    H = leg_h + len(leaves) * hr + head
    fig, ax = canvas(FULL_W, H)
    tx = FULL_W - 2.78 + 0.93  # network / mix / runs columns of the shared table
    cx0, cx1 = 1.08, tx - 0.12
    cw = (cx1 - cx0) / len(feats)
    top = H - head
    ax.text(0.08, H - 0.14, "Supply type" if d["meta"]["group"] == "sysmix" else "Type", fontsize=5.8, color=GREY,
            va="center")
    for dx, lab in ((0.0, "Network"), (0.49, "Supply mix (%)"), (1.35, "Runs")):
        ax.text(tx + dx, H - 0.14, lab, fontsize=5.8, color=GREY, va="center")
    span = {}
    for j, f in enumerate(feats):
        F = d["features"][f]
        a, b = cx0 + j * cw + 0.07, cx0 + (j + 1) * cw - 0.07
        span[f] = (a, b, F["min"], F["max"])
        ax.text((a + b) / 2, H - 0.06, wrap(F["label"], 13), ha="center", va="top", fontsize=6.2, color=INK,
                weight="bold", linespacing=1.0)
        ax.text(a, top + 0.07, fmt_thr(F["min"]), ha="left", va="center", fontsize=5.0, color=GREY)
        ax.text(b, top + 0.07, fmt_thr(F["max"]), ha="right", va="center", fontsize=5.0, color=GREY)
        if F["unit"]:
            ax.text((a + b) / 2, top + 0.07, F["unit"], ha="center", va="center", fontsize=5.0, color=GREY)

    prev = None
    for i, lf in enumerate(leaves):
        c = d["classes"][lf["cls"]]
        cy = top - hr * (i + 0.5)
        if prev is not None:
            ax.plot([0.08, FULL_W - 0.06], [cy + hr / 2] * 2, color=LINE if prev != lf["cls"] else TRACK,
                    lw=0.8 if prev != lf["cls"] else 0.5, zorder=0)
        prev = lf["cls"]
        chip(ax, 0.08, cy - 0.13, 0.9, 0.26, c["name"], c["color"], fs=6.0)
        for f in feats:
            a, b, lo, hi = span[f]
            px = lambda v: a + (b - a) * (min(max(v, lo), hi) - lo) / (hi - lo)
            ax.plot([a, b], [cy - 0.05] * 2, color=TRACK, lw=3.2, solid_capstyle="round", zorder=1)
            if f not in lf["bounds"]:
                continue
            blo, bhi = lf["bounds"][f]
            ax.plot([px(blo) + 0.02, px(bhi) - 0.02], [cy - 0.05] * 2, color=c["color"], lw=3.2,
                    solid_capstyle="round", zorder=2)
            both = np.isfinite(blo) and np.isfinite(bhi)  # two cuts: labels outwards, else centred on the cut
            for v, sign, ha in ((blo, ">", "right" if both else "center"), (bhi, "≤", "left" if both else "center")):
                if np.isfinite(v):
                    ax.plot([px(v)] * 2, [cy - 0.11, cy + 0.01], color=INK, lw=0.7, zorder=3)
                    ax.text(px(v) if both else min(max(px(v), a + 0.12), b - 0.12), cy + 0.08,
                            f"{sign} {fmt_thr(v)}", ha=ha, va="center", fontsize=5.4, color=INK, zorder=4)
        draw_network(ax, lf["info"], d["scale"], tx + 0.19, cy, 0.105)
        draw_mix(ax, d, lf["mix"], tx + 0.49, cy - 0.06, 0.78, 0.12)
        ax.text(tx + 1.35, cy, leaf_sub(d, lf, "\n"), fontsize=5.5, color=GREY, va="center", linespacing=1.15)
    yb = legend(fig, ax, d, 0.08, leg_h - 0.06, FULL_W - 0.16)
    ax.text(FULL_W - 0.08, max(yb, 0.02) + 0.07, "coloured strip = range of the driver in the leaf; grey = any value",
            fontsize=5.8, color=GREY, ha="right", va="center")
    save(fig, out)


# ---------------------------------------------------------------- E: partition map ------------------------------
def cond_text(d, f, lo, hi):
    lab, u = d["features"][f]["label"], d["features"][f]["unit"]
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{fmt_thr(lo)} < {lab} ≤ {fmt_thr(hi, u)}"
    return f"{lab} ≤ {fmt_thr(hi, u)}" if np.isfinite(hi) else f"{lab} > {fmt_thr(lo, u)}"


def fig_partition(d, out):
    F, root = d["features"], d["tree"]
    if len(F) < 2:
        print("partition map skipped: the tree uses a single driver")
        return
    facet = root["feature"] if len(F) > 2 else None
    fx, fy = sorted((f for f in F if f != facet), key=lambda f: -F[f]["importance"])[:2]
    P = d["points"]
    X, Y, cls = np.array(P[fx]), np.array(P[fy]), np.array(P["class"])
    Z = np.array(P[facet]) if facet else np.zeros(len(X))
    # the 8 runs of an economy share the driver values: one dot per distinct point, most frequent class
    _, first, inv = np.unique(np.c_[X, Y, Z], axis=0, return_index=True, return_inverse=True)
    inv = np.ravel(inv)
    maj = np.array([np.bincount(cls[inv == i], minlength=len(d["classes"])).argmax() for i in range(len(first))])
    X, Y, Z, cls = X[first], Y[first], Z[first], maj
    # log axis when the cuts of a driver sit in the lowest fifth of its range (they would be squeezed)
    cut = {f: np.mean([s["thr"] for s in d["splits"] if s["feature"] == f]) for f in (fx, fy)}
    logs = {f: F[f]["min"] > 0 and (cut[f] - F[f]["min"]) / (F[f]["max"] - F[f]["min"]) < 0.2 for f in (fx, fy)}
    mid = lambda f, a, b: float(np.sqrt(a * b)) if logs[f] else (a + b) / 2
    panels = [("left", "≤"), ("right", ">")] if facet else [(None, "")]
    fig, axes = plt.subplots(1, len(panels), figsize=(FULL_W, 3.55), sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.235, wspace=0.05)
    xlim, ylim = (F[fx]["min"], F[fx]["max"]), (F[fy]["min"], F[fy]["max"])
    rng = np.random.default_rng(0)
    for ax, (side, sign) in zip(axes, panels):
        sub_root = root[side] if facet else root
        sel = (Z <= root["thr"]) == (side == "left") if facet else np.ones(len(X), bool)
        idx = rng.permutation(np.where(sel)[0])
        ax.scatter(X[idx], Y[idx], s=4.5, c=[d["classes"][k]["color"] for k in cls[idx]], alpha=0.8, lw=0,
                   rasterized=True, zorder=1)
        for f, axis, setscale in ((fx, ax.xaxis, ax.set_xscale), (fy, ax.yaxis, ax.set_yscale)):
            if logs[f]:
                setscale("log")
                axis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
                axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                axis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        groups = {}
        for lf in d["leaves"]:
            node, ok = lf, False  # is the leaf below this panel's subtree?
            stack = [sub_root]
            while stack:
                n = stack.pop()
                ok |= n["id"] == lf["id"]
                if n["type"] == "split":
                    stack += [n["left"], n["right"]]
            if not ok:
                continue
            bx, by = lf["bounds"].get(fx, (-np.inf, np.inf)), lf["bounds"].get(fy, (-np.inf, np.inf))
            box = (max(bx[0], xlim[0]), min(bx[1], xlim[1]), max(by[0], ylim[0]), min(by[1], ylim[1]))
            groups.setdefault(box, []).append(lf)
        for (a, b, lo, hi), lfs in groups.items():
            counts = np.sum([lf["counts"] for lf in lfs], axis=0)
            ax.add_patch(Rectangle((a, lo), b - a, hi - lo, fc=d["classes"][int(counts.argmax())]["color"],
                                   alpha=0.10, ec="none", zorder=0))
            ax.add_patch(Rectangle((a, lo), b - a, hi - lo, fc="none", ec=NAVY, lw=0.9, zorder=3))
            # one label per region; leaves that share it (split on a driver off the axes) get one line each
            rows = []
            for lf in lfs:
                extra = [cond_text(d, f, *bd) for f, bd in lf["bounds"].items()
                         if f not in (fx, fy) and not (f == facet and bd == sub_root["bounds"][f])]
                name = d["classes"][lf["cls"]]["name"]
                rows.append(f"{name}\n{leaf_sub(d, lf)}" if len(lfs) == 1 else
                            f"{', '.join(extra)}:  {name}  (n = {lf['n']:,})")
            ax.text(mid(fx, a, b), mid(fy, lo, hi), "\n".join(rows), ha="center", va="center", fontsize=5.6,
                    color=INK, zorder=5, linespacing=1.2,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", lw=0.9, alpha=0.93,
                              ec=d["classes"][lfs[0]["cls"]]["color"] if len(lfs) == 1 else NAVY))
        ax.set_xlabel(F[fx]["label"] + (f" ({F[fx]['unit']})" if F[fx]["unit"] else ""), fontsize=6.5, color=INK)
        ax.tick_params(labelsize=5.8, length=2.5, width=0.6, colors=INK)
        for sp in ax.spines.values():
            sp.set_color(INK)
        if facet:
            word = F[facet]["lo_word" if side == "left" else "hi_word"]
            ax.set_title(f"{F[facet]['label']} {sign} {fmt_thr(root['thr'], F[facet]['unit'])}"
                         + (f"  ({word})" if word else ""), fontsize=6.8, color=INK, weight="bold", pad=4)
    axes[0].set_ylabel(F[fy]["label"] + (f" ({F[fy]['unit']})" if F[fy]["unit"] else ""), fontsize=6.5, color=INK)
    lax = fig.add_axes([0, 0, 1, 0.12])
    lax.set_xlim(0, FULL_W)
    lax.set_ylim(0, 3.55 * 0.12)
    lax.axis("off")
    legend(fig, lax, d, 0.5, 0.40, FULL_W - 0.6, classes=True, network=False, mix=False, fs=6.0)
    lax.text(FULL_W - 0.08, 0.12, "dot = one economy (8 topologies), coloured by its most frequent type · box = leaf of the tree",
             fontsize=5.8, color=GREY, ha="right", va="center")
    save(fig, out)


# ---------------------------------------------------------------- F, G, H: card trees ---------------------------
# Paper-size take on tree.html: curved branches, pill labels on the branches, light leaf cards with a soft
# shadow and a coloured top edge, labelled clusters, vertical import arrows, square-root size scales.
def shadow_box(ax, x, y, w, h, rs, fc="white", ec=CARD_EDGE, lw=0.5, z=2):
    """Rounded box with a soft drop shadow; returns the box patch (to clip a coloured top edge on it)."""
    for off, grow, a in ((0.016, 0.012, 0.035), (0.010, 0.004, 0.06)):
        ax.add_patch(FancyBboxPatch((x - grow, y - off - grow), w + 2 * grow, h + 2 * grow,
                                    boxstyle=f"round,pad=0,rounding_size={rs + grow}", fc=SHADOW, ec="none",
                                    alpha=a, zorder=z - 0.5))
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}", fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(box)
    return box


def draw_network_card(ax, info, sc, cx, cy, r, labels=False, z=5):
    """Leaf network as in tree.html: sizes on a square-root scale, import arrows vertical, rare links faint."""
    k = r / 0.30
    sq = lambda v, vmax: float(np.sqrt(max(v, 0) / vmax))
    P = {"L1": (cx, cy + r), "L2": (cx, cy - r), "S1": (cx - 1.05 * r, cy), "S2": (cx + 1.05 * r, cy)}
    for key, (a, b) in LINKS.items():
        freq, cap, hp = info[key]
        if freq < FREQ_RARE:
            continue
        xy = ([P[a][0], P[b][0]], [P[a][1], P[b][1]])
        if freq < FREQ_DASH:
            ax.plot(*xy, color="#9aa1ae", lw=0.5 * max(k, 0.8), ls=(0, (1, 1.8)), zorder=z)
            continue
        ax.plot(*xy, color=HIGHP if hp > 0.5 else NAVY, lw=max(0.7, sq(cap, sc["pipe_max"]) * 3.6 * k), zorder=z + 1,
                solid_capstyle="round", **({} if freq >= FREQ_SOLID else {"ls": (0, (1.5, 1.2)), "alpha": 0.55}))
    for n, sgn in (("L1", 1), ("L2", -1)):
        rad = r * (0.17 + 0.23 * sq(info[f"cap_{n}"], sc["large_elzr_max"]))
        f = sq(info[f"imp_{n}"], sc["import_max"])
        if info[f"imp_{n}"] / sc["import_max"] > 0.03:
            tip = P[n][1] + sgn * (rad + 0.035 * k)
            ax.add_patch(FancyArrowPatch((cx, tip + sgn * 0.42 * r), (cx, tip), arrowstyle="-|>", color=IMPORT,
                                         mutation_scale=(3.5 + 4.5 * f) * max(k, 0.7), lw=(0.7 + 1.9 * f) * max(k, 0.7),
                                         shrinkA=0, shrinkB=0, zorder=z + 2, capstyle="butt"))
        ax.add_patch(Circle(P[n], rad, fc=NAVY, ec="white", lw=0.5, zorder=z + 3))
        if labels:
            ax.text(P[n][0], P[n][1] - 0.004, n, ha="center", va="center", fontsize=4.6 * min(k, 1.15), color="white",
                    weight="bold", zorder=z + 4)
    for n in ("S1", "S2"):
        s = r * (0.13 + 0.17 * sq(info[f"cap_{n}"], sc["small_elzr_max"]))
        ax.add_patch(FancyBboxPatch((P[n][0] - s, P[n][1] - s), 2 * s, 2 * s, fc=YELLOW, ec=NAVY, lw=0.6, zorder=z + 3,
                                    boxstyle=f"round,pad=0,rounding_size={0.25 * s}"))
        if labels:
            ax.text(P[n][0], P[n][1] - 0.004, n, ha="center", va="center", fontsize=4.2 * min(k, 1.15), color=NAVY,
                    weight="bold", zorder=z + 4)


ETA = 0.655  # electrolyzer efficiency used in dt_v5_derived_ratios (MWh H2 per MWh electricity)
OUT_L, OUT_S = "#dfe3ee", "#fff4c2"  # fill of the demand outline of large / small clusters


def node_values(info, mode):
    """(demand, supply) of each cluster in one common unit.
    capacity:   electrolyzer capacity [MW] vs the capacity that would cover the cluster's demand at full load
    production: local hydrogen production vs demand [TWh/y]"""
    out = {}
    for n in ("L1", "L2", "S1", "S2"):
        if mode == "capacity":
            out[n] = (info[f"dem_{n}"] * 1e6 / 8760 / ETA, info[f"cap_{n}"])
        else:
            out[n] = (info[f"dem_{n}"], info[f"prod_{n}"])
    return out


def node_symbol(ax, shape, xy, ro, ri, frac, look, lw, z):
    """One cluster symbol. ro / ri = size of the demand / of the electrolyzer (radius or half side), frac = ri
    value over ro value. look: nested (pale outline shape, cluster colour inside) | dark (cluster colour = demand,
    navy inside = electrolyzer, for both shapes) | gauge (shape = demand, filled from the bottom up to frac;
    a ring outside when the cluster produces more than its demand)."""
    circ = shape == "circle"
    mk = lambda s_, **kw: (Circle(xy, s_, **kw) if circ else
                           Rectangle((xy[0] - s_, xy[1] - s_), 2 * s_, 2 * s_, **kw))
    if look == "gauge":
        bg = mk(ro, fc=OUT_L if circ else "white", ec="none", zorder=z)
        ax.add_patch(bg)
        lvl = Rectangle((xy[0] - ro, xy[1] - ro), 2 * ro, 2 * ro * min(frac, 1.0), fc=NAVY if circ else YELLOW,
                        ec="none", zorder=z + 1)
        ax.add_patch(lvl)
        lvl.set_clip_path(bg)
        ax.add_patch(mk(ro, fc="none", ec=NAVY, lw=lw, zorder=z + 2))
        if frac > 1.05:
            ax.add_patch(mk(ri, fc="none", ec=NAVY, lw=0.8 * lw, zorder=z + 2))
        return
    fo = OUT_L if circ else (OUT_S if look == "nested" else YELLOW)
    fi = NAVY if (circ or look == "dark") else YELLOW
    ax.add_patch(mk(ro, fc=fo, ec="none", zorder=z))
    ax.add_patch(mk(ri, fc=fi, ec="none", zorder=z + 1))
    over = ri > ro
    # demand outline on top: navy around a smaller electrolyzer, light inside a larger one
    ec = NAVY if not (over and fi == NAVY) else ("white" if circ else YELLOW)
    ax.add_patch(mk(ro, fc="none", ec=ec, lw=lw, zorder=z + 2))
    if over and fi != NAVY:
        ax.add_patch(mk(ri, fc="none", ec=NAVY, lw=0.5 * lw, zorder=z + 1))


def draw_network_nested(ax, info, sc, cx, cy, r, mode="capacity", power=0.5, vmax=1.0, z=5, look="nested",
                        arrows="vertical", freq=None, labels=False):
    """Leaf network with nested cluster symbols on one scale for large and small clusters: the outline is
    the demand, the shape inside is the electrolyzer (capacity or production, see node_values; look: see
    node_symbol). power 0.5 = areas proportional to the values; lower = compressed, small clusters readable.
    arrows: vertical | side (import arrow enters the large cluster from the left, its line stays visible).
    freq = (solid, dashed, dotted) shares of runs from which a pipeline is drawn; dotted None = no dotted class.
    labels: print local production (or capacity) as % of the demand next to each cluster."""
    k = r / 0.30
    fs_, fd_, fr_ = freq or (FREQ_SOLID, FREQ_DASH, FREQ_RARE)
    sq = lambda v, m: float(np.sqrt(max(v, 0) / m))
    size = lambda v: max(0.46 * r * (max(v, 0) / vmax) ** power, 0.012)  # circle radius
    # dark look: below 100% the inside is linear in the share (a 70-90% share leaves a visible rim)
    inner = lambda dem, el: (size(dem), size(dem) * el / dem if look == "dark" and el < dem else size(el))
    P = {"L1": (cx, cy + r), "L2": (cx, cy - r), "S1": (cx - 1.05 * r, cy), "S2": (cx + 1.05 * r, cy)}
    for key, (a, b) in LINKS.items():
        f, cap, hp = info[key]
        if f < (fd_ if fr_ is None else fr_):
            continue
        xy = ([P[a][0], P[b][0]], [P[a][1], P[b][1]])
        if f < fd_:
            ax.plot(*xy, color="#9aa1ae", lw=0.5 * max(k, 0.8), ls=(0, (1, 1.8)), zorder=z)
            continue
        ax.plot(*xy, color=HIGHP if hp > 0.5 else NAVY, lw=max(0.7, sq(cap, sc["pipe_max"]) * 3.2 * k), zorder=z + 1,
                solid_capstyle="round", **({} if f >= fs_ else {"ls": (0, (1.5, 1.2)), "alpha": 0.55}))
    vals = node_values(info, mode)
    pct = lambda n: f"{100 * vals[n][1] / max(vals[n][0], 1e-9):.0f}%"
    for n, sgn in (("L1", 1), ("L2", -1)):
        ro, ri = inner(*vals[n])
        ext = max(ro, ri)
        f = sq(info[f"imp_{n}"], sc["import_max"])
        if info[f"imp_{n}"] / sc["import_max"] > 0.03:
            if arrows == "side":
                tip = P[n][0] - ext - 0.025 * k
                ax.add_patch(FancyArrowPatch((tip - 0.80 * r, P[n][1]), (tip, P[n][1]), color=IMPORT,
                                             arrowstyle="-|>,head_length=0.45,head_width=0.26",
                                             mutation_scale=(4.0 + 3.0 * f) * max(k, 0.7),
                                             lw=(0.7 + 2.0 * f) * max(k, 0.7), shrinkA=0, shrinkB=0, zorder=z + 2,
                                             capstyle="butt", joinstyle="miter"))
            else:
                tip = P[n][1] + sgn * (ext + 0.03 * k)
                ax.add_patch(FancyArrowPatch((cx, tip + sgn * 0.36 * r), (cx, tip), arrowstyle="-|>", color=IMPORT,
                                             mutation_scale=(3.5 + 4.5 * f) * max(k, 0.7),
                                             lw=(0.7 + 1.9 * f) * max(k, 0.7), shrinkA=0, shrinkB=0, zorder=z + 2,
                                             capstyle="butt"))
        node_symbol(ax, "circle", P[n], ro, ri, vals[n][1] / max(vals[n][0], 1e-9), look, 0.7 * max(k, 0.8), z + 2)
        if labels:
            ax.text(P[n][0] + ext + 0.03, P[n][1] - 0.003, pct(n), ha="left", va="center", fontsize=4.6, color=GREY,
                    zorder=z + 5)
    for n in ("S1", "S2"):
        so, si = (v * 0.886 for v in inner(*vals[n]))  # half side of the square with the area of the circle
        node_symbol(ax, "square", P[n], so, si, vals[n][1] / max(vals[n][0], 1e-9), look, 0.6 * max(k, 0.8), z + 2)
        if labels:
            out_ = -1 if n == "S1" else 1
            ax.text(P[n][0] - out_ * max(so, si), P[n][1] - max(so, si) - 0.022, pct(n),
                    ha="left" if out_ > 0 else "right", va="top", fontsize=4.6, color=GREY, zorder=z + 5)


def nested_legend(fig, ax, x, y, mode, fs=5.8, look="nested", labels=False):
    """One legend line for the nested cluster symbols; returns the x after it."""
    inner = "electrolyzer capacity" if mode == "capacity" else "local H$_2$ production"
    node_symbol(ax, "circle", (x + 0.06, y), 0.055, 0.036, 0.55, look, 0.6, 3)
    node_symbol(ax, "square", (x + 0.205, y), 0.042, 0.026, 0.55, look, 0.6, 3)
    txt = (f"large / small cluster:  size = demand,  fill level = {inner} ÷ demand  (outer ring: surplus)"
           if look == "gauge" else
           f"large / small cluster:  outline = demand,  inside = {inner}  (same scale)")
    if labels:
        txt += ",  % = the same share"
    t = ax.text(x + 0.29, y - 0.003, txt, fontsize=fs, color=GREY, va="center")
    return x + 0.29 + t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi


def split_strip(ax, d, n, x, y, w, style, z=7):
    """Range of the split's parameter inside its box: light track = sampled range, dark = values that reach
    this split, tick = where the tree cuts. style: plain | number (cut value on the tick) | hist (runs)."""
    f, F = n["feature"], d["features"][n["feature"]]
    lo, hi = F["min"], F["max"]
    cuts = [s["thr"] for s in d["splits"] if s["feature"] == f]
    log = lo > 0 and (np.mean(cuts) - lo) / (hi - lo) < 0.2  # cuts squeezed at the low end -> log scale
    tr = (lambda v: np.log(v)) if log else (lambda v: v)
    px = lambda v: x - w / 2 + w * (tr(min(max(v, lo), hi)) - tr(lo)) / (tr(hi) - tr(lo))
    blo, bhi = n["bounds"].get(f, (-np.inf, np.inf))
    a, b, c = px(max(blo, lo)), px(min(bhi, hi)), px(n["thr"])
    if style.startswith("hist"):
        P = d["points"]
        ids = set()
        stack = [n]
        while stack:
            m = stack.pop()
            if m["type"] == "leaf":
                ids.add(m["id"])
            else:
                stack += [m["left"], m["right"]]
        v = np.array(P[f])[np.isin(P["leaf"], list(ids))]
        edges = np.linspace(x - w / 2, x + w / 2, 19)
        cnt, _ = np.histogram([px(t) for t in v], bins=edges)
        for e0, e1, k in zip(edges[:-1], edges[1:], cnt):
            if k:
                ax.add_patch(Rectangle((e0, y), e1 - e0, 0.085 * k / cnt.max(), ec="white", lw=0.2, zorder=z,
                                       fc="#aab1bd" if (e0 + e1) / 2 <= c else "#5b6686"))
        ax.plot([x - w / 2, x + w / 2], [y, y], color=LINE, lw=0.5, zorder=z)
        ax.plot([c, c], [y - 0.012, y + 0.10], color=NAVY, lw=0.9, zorder=z + 1)
        ytxt = y + 0.105
    else:
        ax.plot([x - w / 2, x + w / 2], [y, y], color=TRACK, lw=2.6, solid_capstyle="round", zorder=z)
        # values reaching the split: lighter below the cut (left branch), darker above it (right branch)
        ax.plot([a, c], [y, y], color="#aab1bd", lw=2.6, solid_capstyle="butt", zorder=z + 0.1)
        ax.plot([c, b], [y, y], color="#5b6686", lw=2.6, solid_capstyle="butt", zorder=z + 0.1)
        ax.plot([c, c], [y - 0.035, y + 0.035], color=NAVY, lw=1.0, zorder=z + 1)
        ax.add_patch(plt.Polygon([(c, y + 0.03), (c - 0.022, y + 0.065), (c + 0.022, y + 0.065)], fc=NAVY, ec="none",
                                 zorder=z + 1))
        ytxt = y + 0.07
    if style in ("number", "hist"):
        ax.text(min(max(c, x - w / 2 + 0.1), x + w / 2 - 0.1), ytxt, fmt_thr(n["thr"], F["unit"]), ha="center",
                va="bottom", fontsize=5.4, color=INK, weight="bold", zorder=z + 1)
    for v, xx, ha in ((lo, x - w / 2, "left"), (hi, x + w / 2, "right")):
        ax.text(xx, y - 0.028, fmt_thr(v), ha=ha, va="top", fontsize=4.6, color=GREY, zorder=z)
    if log:
        ax.text(x, y - 0.028, "log", ha="center", va="top", fontsize=4.4, color=GREY, style="italic", zorder=z)


def leaf_card(ax, d, lf, cx, top, w, h, r, title_h, wide, net=None):
    c = d["classes"][lf["cls"]]
    x, y = cx - w / 2, top - h
    box = shadow_box(ax, x, y, w, h, 0.06)
    cap = Rectangle((x, top - 0.045), w, 0.045, fc=c["color"], ec="none", zorder=2.2)
    ax.add_patch(cap)
    cap.set_clip_path(box)
    ax.text(cx, top - 0.065 - title_h / 2, c["name"] if wide else wrap(c["name"], 11), ha="center", va="center",
            fontsize=6.8 if wide else 6.2, color=c["color"], weight="bold", linespacing=0.98, zorder=4)
    (net or (lambda *a: draw_network_card(*a, labels=wide)))(ax, lf["info"], d["scale"], cx,
                                                              top - 0.065 - title_h - 1.95 * r, r)
    by = y + 0.19
    draw_mix(ax, d, lf["mix"], x + 0.07, by, w - 0.14, 0.11, fs=5.0 if not wide else 5.4)
    meta = (f"± {lf['info']['spread']:.2f}" if d["meta"]["regression"] else f"{lf['purity']:.0%} pure")
    fs = 5.6 if wide else 5.0
    ax.text(x + 0.07, y + 0.10, f"n = {lf['n']:,}", ha="left", va="center", fontsize=fs, color=GREY, zorder=4)
    ax.text(x + w - 0.07, y + 0.10, meta, ha="right", va="center", fontsize=fs, color=GREY, zorder=4)


def fig_cards(d, out, weighted=False, staggered=False, nodes=None, power=0.5, strip=None, pills="full",
              look="nested", arrows="vertical", freq=None, labels=False):
    """F: card tree. G (weighted): branch width = share of the runs, each split shows the mix of types that
    reaches it. H (staggered): leaves on two rows, so each card is twice as wide and the networks are large.
    G_V*: nodes = capacity | production -> nested cluster symbols (draw_network_nested, size exponent `power`);
    strip = plain | number | hist -> parameter range with the cut inside each split (split_strip) instead of
    the type bar (histbare = histogram without the cut value); pills = full (cut value + word) | word (word
    only, the cut value is on the strip). look, arrows, freq, labels: see draw_network_nested."""
    leaves, levels, N = d["leaves"], d["levels"], d["tree"]["n"]
    nl = len(leaves)
    x0, x1 = 0.06, FULL_W - 0.06
    # staggered cards span two slots, so the outer leaves need half a slot of margin on each side
    slot = (x1 - x0) / (nl + 1 if staggered else nl)
    xs = lambda p: x0 + slot * (p + (1.0 if staggered else 0.5))
    labs, hh = split_labels(d, 14)
    extra = {None: 0.035 if weighted else 0.0, "plain": 0.09, "number": 0.12, "hist": 0.14, "histbare": 0.115}[strip]
    hh += extra
    hw = min(0.47, slot * 0.54)
    curve, pill_h, leg_h, gap = 0.20, 0.215 if pills == "full" else 0.125, 0.44 + (0.30 if nodes else 0), 0.10
    net = None
    if nodes:
        vmax = max(max(v) for lf in leaves for v in node_values(lf["info"], nodes).values())
        net = lambda ax_, info, sc, cx, cy, r_: draw_network_nested(
            ax_, info, sc, cx, cy, r_, nodes, power, vmax, look=look, arrows=arrows, freq=freq, labels=labels)
    dy = 2 * hh + curve + pill_h + 0.10
    rows = 2 if staggered else 1
    w = 2 * slot - 0.10 if staggered else slot - 0.07
    r = 0.30 if staggered else 0.215
    title_h = 0.14 if staggered else 0.23
    card_h = 0.065 + title_h + 3.9 * r + 0.04 + 0.11 + 0.19
    leaf_top = leg_h + rows * card_h + (rows - 1) * gap
    H = leaf_top + 0.16 + 2 * hh + (levels - 1) * dy + curve + pill_h
    fig, ax = canvas(FULL_W, H)
    ys = lambda dep: H - 0.09 - hh - dep * dy
    card_top = lambda lf: leaf_top - ((lf["pos"] % 2) * (card_h + gap) if staggered else 0)

    name_new(ax, "background")
    for n in d["splits"]:
        x, y = xs(n["pos"]), ys(n["depth"])
        py = y - hh
        for side in ("left", "right"):
            c = n[side]
            cx = xs(c["pos"])
            cy = ys(c["depth"]) + hh if c["type"] == "split" else card_top(c)
            path = Path([(x, py), (x, py - 0.6 * curve), (cx, py - 0.4 * curve), (cx, py - curve), (cx, cy)],
                        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO])
            ax.add_patch(PathPatch(path, fc="none", ec="#b7bcc9", lw=max(0.8, 7.0 * c["n"] / N) if weighted else 1.1,
                                   capstyle="round", zorder=1))
            thr, word = edge_text(d, n, side)
            lab = thr + (f"\n{word}" if word else "") if pills == "full" else (word or thr)
            ax.text(cx, py - curve - 0.035, lab, ha="center", va="top", fontsize=5.5,
                    color=INK, linespacing=1.05, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.6", fc="white", ec="#c3c8d2", lw=0.5))
        name_new(ax, f"branches_{n['id']}")
        shadow_box(ax, x - hw, y - hh, 2 * hw, 2 * hh, 0.06, ec=NAVY, lw=0.8, z=6)
        ax.text(x, y + extra, labs[n["id"]], ha="center", va="center", fontsize=6.3, color=INK,
                weight="bold", linespacing=1.0, zorder=7)
        if strip:
            split_strip(ax, d, n, x, y - hh + 0.115, 2 * hw - 0.2, strip)
        elif weighted:  # types of the runs that reach this split
            bx, bw = x - hw + 0.08, 2 * hw - 0.16
            for k, cnt in enumerate(n["counts"]):
                ax.add_patch(Rectangle((bx, y - hh + 0.05), bw * cnt / n["n"], 0.045, fc=d["classes"][k]["color"],
                                       ec="white", lw=0.3, zorder=7))
                bx += bw * cnt / n["n"]

        name_new(ax, "split_" + slug(d["features"][n["feature"]]["label"]) + f"_{n['id']}")
    for lf in leaves:
        leaf_card(ax, d, lf, xs(lf["pos"]), card_top(lf), w, card_h, r, title_h, staggered, net)
        name_new(ax, f"leaf{lf['pos'] + 1}_" + slug(d["classes"][lf["cls"]]["name"]))
    yb = legend(fig, ax, d, 0.08, leg_h - 0.07, FULL_W - 0.16, rare=True, clusters=not nodes, freq=freq)
    note = {None: "bar in a split = types of the runs reaching it",
            "plain": "strip in a split = sampled range of the parameter, tick = cut (light: left branch, dark: right)",
            "number": "strip in a split = sampled range of the parameter, tick = cut (light: left branch, dark: right)",
            "hist": "bars in a split = runs reaching it along the sampled range, line = cut",
            "histbare": "bars in a split = runs reaching it along the sampled range, line = cut"}[strip]
    if nodes:
        xe = nested_legend(fig, ax, 0.08, yb - 0.075, nodes, look=look, labels=labels)
        ax.text(0.08, yb - 0.228, "branch width = share of the runs  ·  " + note, fontsize=5.8, color=GREY,
                ha="left", va="center")
    elif weighted:
        ax.text(FULL_W - 0.08, max(yb, 0.02) + 0.07, "branch width = share of the runs · " + note,
                fontsize=5.8, color=GREY, ha="right", va="center")
    name_new(ax, "legend")
    save(fig, out)


# ---------------------------------------------------------------- main ------------------------------------------
STYLES = {"A": ("A_tree_vertical", fig_vertical), "B": ("B_tree_horizontal", fig_horizontal),
          "C": ("C_tree_flow", fig_flow), "D": ("D_rule_table", fig_rules), "E": ("E_partition_map", fig_partition),
          "F": ("F_tree_cards", fig_cards),
          "G": ("G_tree_cards_weighted", lambda d, out: fig_cards(d, out, weighted=True)),
          "H": ("H_tree_cards_staggered", lambda d, out: fig_cards(d, out, staggered=True)),
          # G variants: nested cluster symbols (demand outline + electrolyzer inside) and parameter-range strips
          "G1": ("G_V1_nested_strip", lambda d, out: fig_cards(d, out, weighted=True, nodes="capacity", power=0.33,
                                                                strip="plain")),
          "G2": ("G_V2_true_area_cut_value", lambda d, out: fig_cards(d, out, weighted=True, nodes="capacity",
                                                                       power=0.5, strip="number", pills="word")),
          "G3": ("G_V3_production_histogram", lambda d, out: fig_cards(d, out, weighted=True, nodes="production",
                                                                        power=0.33, strip="hist", pills="word")),
          # V3 variants: cut value back on the branches, import arrows from the side, pipelines from 20% of runs
          "G3a": ("G_V3_a_dark_inside", lambda d, out: fig_cards(
              d, out, weighted=True, nodes="production", power=0.25, strip="histbare", look="dark", arrows="side",
              freq=(0.6, 0.3, 0.2))),
          "G3b": ("G_V3_b_fill_level", lambda d, out: fig_cards(
              d, out, weighted=True, nodes="production", power=0.30, strip="histbare", look="gauge", arrows="side",
              freq=(0.6, 0.3, 0.2))),
          "G3c": ("G_V3_c_percent_labels", lambda d, out: fig_cards(
              d, out, weighted=True, nodes="production", power=0.30, strip="histbare", look="gauge", arrows="side",
              freq=(0.6, 0.2, None), labels=True))}


def caption(d):
    m, s = d["meta"], d["stats"]
    kind = "regression" if m["regression"] else "classification"
    drv = "sampled parameters and supply ratios derived from them" if m["derived"] else "sampled parameters"
    lines = [f"case: {m['case']}",
             f"{m['leaves']}-leaf {kind} tree on the {drv}; {m['runs']:,} optimizations, {m['economies']} economies.",
             f"grouped 5-fold CV by economy: accuracy {s['accuracy']:.1%} (95% CI {s['accuracy_ci'][0]:.1%}–"
             f"{s['accuracy_ci'][1]:.1%}), balanced accuracy {s['balanced_accuracy']:.1%}, Cohen's kappa "
             f"{s['kappa']:.2f} (95% CI {s['kappa_ci'][0]:.2f}–{s['kappa_ci'][1]:.2f}), majority-class baseline "
             f"{s['baseline']:.1%}."]
    if m["regression"]:
        lines.append(f"grouped R2 {s['r2_grouped']:.2f}; accuracy read on k = {s['k_reference']} classes "
                     f"(predicted mix -> nearest k-Means centroid). ± = mean std of the shares in the leaf.")
    lines += ["", "classes (n, mean supply mix " + " / ".join(m["mix_order"]) + "):"]
    lines += [f"  {c['name']}: n = {c['n']:,}, " + " / ".join(f"{100 * v:.0f}%" for v in c["mix"]) for c in d["classes"]]
    lines += ["", "driver importance in the tree:"]
    lines += [f"  {F['label'].replace('$_2$', '2')}: {100 * F['importance']:.1f}%" for F in
              sorted(d["features"].values(), key=lambda F: -F["importance"])]
    lines += ["", f"network glyph: pipeline solid if built in >= {FREQ_SOLID:.0%} of the leaf's runs, dashed "
                  f"{FREQ_DASH:.0%}-{FREQ_SOLID:.0%}, not drawn below; width = mean capacity when built; marker size = "
                  "mean electrolyzer capacity; arrow = hydrogen import at the large cluster."]
    return "\n".join(lines) + "\n"


def render(case, only=None, rename=None, out_dir=None):
    d = load(case, rename)
    folder = os.path.join(out_dir or PAPER_DIR, d["meta"]["case"])
    os.makedirs(folder, exist_ok=True)
    for key, (name, fn) in STYLES.items():
        if not only or key in only:
            fn(d, os.path.join(folder, name))
    open(os.path.join(folder, "caption.txt"), "w", encoding="utf-8").write(caption(d))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="case names or JSON paths (default: all in <PAPER_DIR>/_data)")
    ap.add_argument("--only", nargs="*", default=None, help="styles to draw: A B C D E")
    ap.add_argument("--rename", nargs="*", default=[], help='"old class name=new name" pairs')
    ap.add_argument("--out", default=None, help="output root (default PAPER_DIR)")
    a = ap.parse_args()
    cases = a.cases or sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    for case in cases:
        render(case, a.only, dict(r.split("=", 1) for r in a.rename), a.out)
