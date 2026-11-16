"""
render_tree_slide.py
====================
16:9 slide figure of a driver tree, redrawn from the ``tree.html`` that
``dt_v5_uncertain.py`` writes in every case folder (DATA object embedded in the
html), so splits, leaves and leaf networks are exactly the ones in the html.

Each leaf shows the class name, n and purity, the representative 4-node network
(hub square / small circle size = electrolyzer capacity, line width = pipeline
capacity, solid >=60% of runs, dashed 30-60%, grey 10-30%, red = high pressure,
yellow arrow = import) and a local-vs-import bar. The legend gives the cluster
profile on the clustering features (cluster_profiles.csv).

Two layouts: vertical (root on top, leaves in a bottom row) and horizontal
(root on the left, leaves stacked on the right).

Usage
-----
    python render_tree_slide.py <case_folder> [--horizontal]
    python render_tree_slide.py <case_folder> --title "What drives the supply mix?" \
        --rename "Low self-served=Import-reliant" "Medium self-served=Self + transport" \
                 "High self-served=Self-sufficient"

Writes driver_tree_slide[_horizontal].png (300 dpi), .pdf and .svg into the case folder.
"""
import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

PALETTE = ["#d1495b", "#00798c", "#edae49", "#66a182", "#2e4057", "#8d96a3", "#b56576", "#6d597a",
           "#e07a5f", "#3d405b"]
INK, GREY, EDGE = "#1f2937", "#6b7280", "#9ca3af"
LOCAL, TRANSP, IMPORT = "#63A593", "#4a6fa5", "#d4a017"
SHORT = {"self_served": "self", "imported": "imp", "transported": "transp", "small_selfsuff": "small self-suff",
         "hub_dependence": "hub dep", "small_prod_share": "small prod", "n_pipelines": "pipes",
         "n_LS_arcs": "L-S arcs", "LL_pipe": "L-L pipe", "SS_pipe": "S-S pipe", "highP_share": "high-P"}
COUNTS = {"n_pipelines", "n_LS_arcs"}  # profile columns that are counts, not shares
BINARY = {"LL_pipe", "SS_pipe"}  # 0/1 per run: profile mean = share of runs
LINKS = {"L1L2": ("L1", "L2"), "L1S1": ("L1", "S1"), "L1S2": ("L1", "S2"),
         "L2S1": ("L2", "S1"), "L2S2": ("L2", "S2"), "S1S2": ("S1", "S2")}
NETWORK_NOTE = ("Network: ■ hub / ● small cluster (size = electrolyzer) · pipeline: solid ≥60% of runs, "
                "dashed 30–60%, grey 10–30%, width = capacity, red = high pressure · yellow arrow = import · "
                "bar: local (green) vs import (yellow)")
W, H = 13.333, 7.5


# ---------------------------------------------------------------- input -----------------------------------------
def read_html(case):
    html = open(os.path.join(case, "tree.html"), encoding="utf-8").read()
    grab = lambda pat: json.loads(re.search(pat, html, re.S).group(1))
    data = grab(r"const DATA=(\{.*?\}), FEAT=")
    feat = grab(r"FEAT=(\{.*?\}), SUB=")
    pairs = grab(r"PAIRS=(\{.*?\});")
    text = re.sub(r"<[^>]+>", " ", html.split("<script")[0])
    head = re.search(r"(\d+) optimizations, (\d+) economies\. (\d+)-leaf tree, grouped &kappa; ([\d.]+) "
                     r"\(baseline ([\d.]+)\)\. Clustered on ([\w, ]+)\.", re.sub(r"\s+", " ", text))
    return data, feat, pairs, head


def profiles(case, clustered_on):
    p = os.path.join(case, "cluster_profiles.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    cols = [c for c in clustered_on if c in df.columns]
    out = {}
    for _, r in df.iterrows():
        parts = [f"{SHORT.get(c, c)} {r[c]:.1f}" if c in COUNTS else
                 f"{SHORT.get(c, c)} in {100 * r[c]:.0f}% of runs" if c in BINARY else
                 f"{SHORT.get(c, c)} {100 * r[c]:.0f}%" for c in cols]
        out[r["name"]] = " / ".join(parts)
    return out


def fmt_thr(feat_meta, feat, thr):
    unit = feat_meta.get(feat, [feat, ""])[1]
    v = f"{thr:,.0f}" if thr >= 100 else (f"{thr:.2f}".rstrip("0").rstrip(".") if thr < 10 else f"{thr:.1f}")
    return f"{v} {unit}".strip() if unit not in ("ratio", "-", "") else v


# ---------------------------------------------------------------- drawing helpers -------------------------------
def draw_network(ax, info, sc, cx, cy, sx, sy, k=1.0):
    """4-node network: L1 top, L2 bottom, S1 left, S2 right, centred on (cx, cy). k scales marks."""
    P = {"L1": (cx, cy + sy), "L2": (cx, cy - sy), "S1": (cx - sx, cy), "S2": (cx + sx, cy)}
    for key, (a, b) in LINKS.items():
        freq, cap, hp = info[key]
        if freq < 0.10:
            continue
        style = dict(color=INK, ls="-") if freq >= 0.6 else \
            dict(color=INK, ls=(0, (3, 2)), alpha=0.8) if freq >= 0.3 else dict(color="#c7cbd1", ls="-")
        if hp > 0.5:
            style["color"] = "#b91c1c"
        lw = (0.8 + 4.2 * cap / sc["pipe_max"]) * max(k, 0.55) if freq >= 0.3 else 1.0 * max(k, 0.7)
        (xa, ya), (xb, yb) = P[a], P[b]
        ax.plot([xa, xb], [ya, yb], lw=lw, zorder=3, solid_capstyle="round", **style)
    for n in ("L1", "L2"):
        s = (0.07 + 0.17 * info[f"cap_{n}"] / sc["large_elzr_max"]) * k
        ax.add_patch(Rectangle((P[n][0] - s, P[n][1] - s), 2 * s, 2 * s, fc="#374151", ec="white", lw=0.8 * k,
                               zorder=5))
        imp = info[f"imp_{n}"] / sc["import_max"]
        if imp > 0.03:
            ax.add_patch(FancyArrowPatch((P[n][0] - 0.55 * k, P[n][1] + (0.22 if n == "L1" else -0.22) * k),
                                         (P[n][0] - s - 0.02 * k, P[n][1]), arrowstyle="-|>",
                                         mutation_scale=(6 + 10 * imp) * max(k, 0.6), lw=(0.6 + 3 * imp) * max(k, 0.6),
                                         color=IMPORT, zorder=4))
    for n in ("S1", "S2"):
        r = (0.04 + 0.11 * info[f"cap_{n}"] / sc["small_elzr_max"]) * k
        ax.add_patch(Circle(P[n], r, fc="#9ca3af", ec="white", lw=0.8 * k, zorder=5))


def draw_bar(ax, x, y, w, h, info):
    """Leaf bar: supply mix self / transp / imp when the leaf carries `mix` (regression tree),
    else local vs import. Returns the caption under the bar."""
    if "mix" in info:
        x0 = x
        for share, c in zip(info["mix"], (LOCAL, TRANSP, IMPORT)):
            ax.add_patch(Rectangle((x0, y), w * share, h, fc=c, ec="none", zorder=3))
            x0 += w * share
        s, t, i = (round(100 * v) for v in info["mix"])
        return f"self {s} · transp {t} · imp {i}"
    imp = info["impfrac"]
    ax.add_patch(Rectangle((x, y), w * (1 - imp), h, fc=LOCAL, ec="none", zorder=3))
    ax.add_patch(Rectangle((x + w * (1 - imp), y), w * imp, h, fc=IMPORT, ec="none", zorder=3))
    return f"import {imp:.0%}"


def leaf_meta(info):
    if "spread" in info:  # regression leaf: mean std of the targets instead of a class purity
        return f"n = {info['n']} · ± {info['spread']:.2f}"
    return f"n = {info['n']} · pure {info['purity']:.0%}"


def split_box(ax, x, y, label, fs, hw=0.85, hh=0.2):
    ax.add_patch(FancyBboxPatch((x - hw, y - hh), 2 * hw, 2 * hh, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#f3f4f6", ec="#4b5563", lw=1, zorder=2))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=INK, weight="bold", zorder=4,
            linespacing=1.1)


def edge_label(feat_meta, pairs, n, side, sep=" · "):
    sign = "≤" if side == "left" else ">"
    word = pairs.get(n["feature"], ["", ""])[0 if side == "left" else 1]
    return f"{sign} {fmt_thr(feat_meta, n['feature'], n['thr'])}" + (f"{sep}{word}" if word else "")


def header_and_legend(ax, title, sub, order, col, name, prof, note=NETWORK_NOTE):
    ax.text(0.4, 7.2, title, fontsize=17, weight="bold", color=INK, va="center")
    if sub:
        ax.text(0.4, 6.88, sub, fontsize=10, color=GREY, va="center")
    lx, ly = 0.55, 0.3
    step = (W - 1.1) / max(len(order), 1)
    for lab in order:
        ax.add_patch(Rectangle((lx, ly), 0.18, 0.13, fc=col[lab], ec="none"))
        txt = f"{name(lab)}" + (f": {prof[lab]}" if lab in prof else "")
        if len(order) > 3 and lab in prof:  # narrow slots: name on top, profile below
            txt = f"{name(lab)}\n{prof[lab]}"
        ax.text(lx + 0.25, ly + 0.065, txt, fontsize=8.2 if len(order) <= 3 else 7.2, va="center", color=INK,
                linespacing=1.05)
        lx += step
    ax.text(0.55, ly - 0.17, note, fontsize=7.4, color=GREY, ha="left", va="center")


# ---------------------------------------------------------------- layouts ---------------------------------------
def layout_vertical(ax, tree, leaves, dmax, sc, feat_meta, pairs, col, name):
    nl = len(leaves)
    x0, x1 = 0.55, W - 0.55
    xs = lambda i: x0 + (x1 - x0) * (i + 0.5) / nl
    leaf_top = 3.35
    ys = lambda d: 6.45 - d * (6.45 - leaf_top - 0.85) / max(dmax - 1, 1)
    fs = 9 if nl <= 8 else 7.5

    def draw_split(n):
        x, y = xs(n["pos"]), ys(n["depth"])
        for side in ("left", "right"):
            c = n[side]
            cx = xs(c["pos"])
            cy = ys(c["depth"]) if c["type"] == "split" else leaf_top
            ax.plot([x, cx], [y - 0.2, cy + (0.2 if c["type"] == "split" else 0)], color=EDGE, lw=1.1, zorder=1)
            t = 0.35
            mx, my = x + (cx - x) * t, (y - 0.2) + ((cy + 0.2) - (y - 0.2)) * t
            ax.text(mx + (-0.06 if side == "left" else 0.06), my, edge_label(feat_meta, pairs, n, side),
                    ha="right" if side == "left" else "left", va="center", fontsize=fs - 0.8, color=INK, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
            if c["type"] == "split":
                draw_split(c)
        split_box(ax, x, y, feat_meta.get(n["feature"], [n["feature"]])[0], fs)

    def draw_leaf(n):
        info = n["info"]
        c = col[info["label"]]
        cx = xs(n["pos"])
        w, h = (x1 - x0) / nl - 0.12, 2.75
        lx, y1 = cx - w / 2, leaf_top
        ax.add_patch(FancyBboxPatch((lx, y1 - h), w, h, boxstyle="round,pad=0.0,rounding_size=0.08",
                                    fc="white", ec=c, lw=2, zorder=2))
        ax.add_patch(Rectangle((lx, y1 - 0.5), w, 0.5, fc=c, ec="none", zorder=3))
        ax.text(cx, y1 - 0.25, name(info["label"]), ha="center", va="center", fontsize=fs + 0.5, color="white",
                weight="bold", zorder=4)
        ax.text(cx, y1 - 0.68, leaf_meta(info), ha="center", va="center", fontsize=fs - 1, color=INK, zorder=4)
        draw_network(ax, info, sc, cx, y1 - 1.55, w * 0.36, 0.5)
        by = y1 - h + 0.36
        cap = draw_bar(ax, lx + 0.12, by, w - 0.24, 0.14, info)
        ax.text(cx, by - 0.17, cap, ha="center", va="center", fontsize=fs - 1.8, color=GREY, zorder=4)

    draw_split(tree)
    for lf in leaves:
        draw_leaf(lf)


def layout_horizontal(ax, tree, leaves, dmax, sc, feat_meta, pairs, col, name):
    nl = len(leaves)
    top, bot = 6.55, 0.72
    hr = (top - bot) / nl
    ys = lambda i: top - hr * (i + 0.5)
    card_x0, card_x1 = 9.05, W - 0.4
    hw, hh = 0.62, 0.25
    x_first, x_last = 0.35 + hw, card_x0 - 0.95 - hw
    xs = lambda d: x_first + d * (x_last - x_first) / max(dmax - 1, 1)
    fs = 9 if nl <= 8 else 7.5

    def wrap(s, width=17):
        if len(s) <= width or " " not in s:
            return s
        cut = s.rfind(" ", 0, width + 1)
        cut = cut if cut > 0 else s.find(" ")
        return s[:cut] + "\n" + s[cut + 1:]

    def draw_split(n):
        x, y = xs(n["depth"]), ys(n["pos"])
        for side in ("left", "right"):
            c = n[side]
            cy = ys(c["pos"])
            cx = xs(c["depth"]) - hw if c["type"] == "split" else card_x0
            # elbow connector: out of the box, vertical, into the child; label on the last segment
            xm = x + hw + 0.08
            ax.plot([x + hw, xm, xm, cx], [y, y, cy, cy], color=EDGE, lw=1.1, zorder=1)
            ax.text(xm + 0.04, cy + 0.04, edge_label(feat_meta, pairs, n, side, sep="\n"), ha="left",
                    va="bottom", fontsize=fs - 1.9, color=INK, zorder=3, linespacing=1.05)
            if c["type"] == "split":
                draw_split(c)
        split_box(ax, x, y, wrap(feat_meta.get(n["feature"], [n["feature"]])[0]), fs - 0.8, hw, hh)

    def draw_leaf(n):
        info = n["info"]
        c = col[info["label"]]
        cy = ys(n["pos"])
        h = hr - 0.08
        w = card_x1 - card_x0
        y0 = cy - h / 2
        ax.add_patch(FancyBboxPatch((card_x0, y0), w, h, boxstyle="round,pad=0.0,rounding_size=0.06",
                                    fc="white", ec=c, lw=1.8, zorder=2))
        tag = 1.45
        ax.add_patch(Rectangle((card_x0, y0), tag, h, fc=c, ec="none", zorder=3))
        ax.text(card_x0 + tag / 2, cy + h * 0.14, name(info["label"]), ha="center", va="center", fontsize=fs,
                color="white", weight="bold", zorder=4)
        ax.text(card_x0 + tag / 2, cy - h * 0.22, leaf_meta(info), ha="center", va="center", fontsize=fs - 1.8,
                color="white", zorder=4)
        k = min(1.0, h / 1.5)
        draw_network(ax, info, sc, card_x0 + tag + 0.62, cy, 0.36, h * 0.36, k=k)
        bx = card_x0 + tag + 1.22
        bw = card_x1 - 0.12 - bx
        cap = draw_bar(ax, bx, cy + 0.02, bw, 0.13, info)
        ax.text(bx + bw / 2, cy - 0.12, cap, ha="center", va="center", fontsize=fs - 2.4, color=GREY, zorder=4)

    draw_split(tree)
    for lf in leaves:
        draw_leaf(lf)


# ---------------------------------------------------------------- main ------------------------------------------
def render(case, title=None, rename=None, horizontal=False):
    """Slide figure for a DT_uncert case folder (reads its tree.html)."""
    data, feat_meta, pairs, head = read_html(case)
    clustered_on = [c.strip() for c in head.group(6).split(",")] if head else []
    prof = profiles(case, clustered_on)
    sub = (f"{head.group(3)}-leaf decision tree · {int(head.group(1))} runs, {head.group(2)} economies · "
           f"grouped κ {head.group(4)} (baseline {head.group(5)}) · clustered on {', '.join(clustered_on)}"
           if head else "")
    if not title:
        title = os.path.relpath(case, os.path.dirname(os.path.dirname(os.path.dirname(case)))).replace(os.sep, " / ")
    out = os.path.join(case, "driver_tree_slide" + ("_horizontal" if horizontal else ""))
    return render_data(data, feat_meta, pairs, out, title, sub, prof, rename, horizontal)


def render_data(data, feat_meta, pairs, out, title, sub, prof=None, rename=None, horizontal=False,
                note=NETWORK_NOTE, colors=None):
    """Draw a tree given as the DATA dict of tree.html ({"tree": nested split/leaf dicts, "scale": ...}).
    Leaves may carry "mix" [self, transp, imp] and "spread" instead of "impfrac" / "purity"."""
    tree, sc = data["tree"], data["scale"]
    rename = rename or {}
    prof = prof or {}

    # class order = level order of the profiles, else order of first appearance
    order = list(prof)
    leaves, depth = [], [0]

    def walk(n, d):
        depth[0] = max(depth[0], d)
        n["depth"] = d
        if n["type"] == "leaf":
            n["pos"] = len(leaves)
            leaves.append(n)
            if n["info"]["label"] not in order:
                order.append(n["info"]["label"])
        else:
            walk(n["left"], d + 1)
            walk(n["right"], d + 1)
            n["pos"] = (n["left"]["pos"] + n["right"]["pos"]) / 2

    walk(tree, 0)
    col = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(order)}
    col.update(colors or {})
    name = lambda lab: rename.get(lab, lab)

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    layout = layout_horizontal if horizontal else layout_vertical
    layout(ax, tree, leaves, depth[0], sc, feat_meta, pairs, col, name)
    header_and_legend(ax, title, sub, order, col, name, prof, note)

    if os.name == "nt" and len(out) > 200 and out.startswith("\\\\") and not out.startswith("\\\\?\\"):
        out = "\\\\?\\UNC\\" + out[2:]  # network paths beyond MAX_PATH
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".svg")
    plt.close(fig)
    return out + ".png"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", help="case folder containing tree.html (e.g. ...\\DT_uncert\\sysmix\\sysmix\\all\\k03)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--rename", nargs="*", default=[], help='"old name=new name" pairs')
    ap.add_argument("--horizontal", action="store_true", help="root on the left, leaves stacked on the right")
    a = ap.parse_args()
    ren = dict(r.split("=", 1) for r in a.rename)
    print("saved", render(a.case, a.title, ren, a.horizontal))
