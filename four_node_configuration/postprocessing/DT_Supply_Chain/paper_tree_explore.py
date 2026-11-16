"""
Explorative decision-tree figures (ideas to discuss, not paper candidates yet).

Reads the same JSON as paper_tree_figures.py (step 1 = paper_tree_data.py) and reuses its drawing helpers.
Output: <PAPER_DIR>/<case>/esplorative/<name>.png / .pdf / .svg

    python paper_tree_explore.py                                  # all ideas, default case
    python paper_tree_explore.py sysmix_sysmix_clf_k03_supply_L8 --only X2

Ideas
-----
X1  class histograms   G_V3_b, but the histogram in each split is stacked by type: shows how the cut separates them
X2  radial tree        root in the centre, leaves on a circle as medallions (donut = supply mix, network inside)
X3  medallion tree     top-down tree, leaves as medallions, branches coloured by the type they mostly lead to
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle, Wedge
from matplotlib.path import Path

import paper_tree_figures as P

FREQ = (0.6, 0.3, 0.2)  # pipelines: solid / dashed / dotted from these shares of the runs
NET = dict(mode="production", power=0.30, look="gauge", arrows="side", freq=FREQ)


def tint(color, a):
    """Colour blended with white (a = share of the colour)."""
    import matplotlib.colors as mc
    return tuple(a * c + (1 - a) for c in mc.to_rgb(color))


slug, name_new = P.slug, P.name_new  # object names for Illustrator, see paper_tree_figures.py


def subtree_leaves(n):
    return [n] if n["type"] == "leaf" else subtree_leaves(n["left"]) + subtree_leaves(n["right"])


def vmax_of(d):
    return max(max(v) for lf in d["leaves"] for v in P.node_values(lf["info"], NET["mode"]).values())


# ---------------------------------------------------------------- X1 --------------------------------------------
def strip_by_class(ax, d, n, x, y, w, style, z=7):
    """Histogram of the runs reaching the split along the sampled range of its parameter, stacked by type."""
    f, F = n["feature"], d["features"][n["feature"]]
    lo, hi = F["min"], F["max"]
    cuts = [s["thr"] for s in d["splits"] if s["feature"] == f]
    log = lo > 0 and (np.mean(cuts) - lo) / (hi - lo) < 0.2
    tr = np.log if log else (lambda v: v)
    px = lambda v: x - w / 2 + w * (tr(np.clip(v, lo, hi)) - tr(lo)) / (tr(hi) - tr(lo))
    pts = d["points"]
    sel = np.isin(pts["leaf"], [lf["id"] for lf in subtree_leaves(n)])
    v, cls = px(np.array(pts[f])[sel]), np.array(pts["class"])[sel]
    edges = np.linspace(x - w / 2, x + w / 2, 21)
    cnt = np.array([np.histogram(v[cls == k], bins=edges)[0] for k in range(len(d["classes"]))])
    top = cnt.sum(0).max()
    base = np.zeros(len(edges) - 1)
    for k, c in enumerate(d["classes"]):
        h = 0.105 * cnt[k] / top
        for e0, e1, b0, hk in zip(edges[:-1], edges[1:], base, h):
            if hk > 0:
                ax.add_patch(Rectangle((e0, y + b0), e1 - e0, hk, fc=c["color"], ec="white", lw=0.15, zorder=z))
        base += h
    c = float(px(n["thr"]))
    ax.plot([x - w / 2, x + w / 2], [y, y], color=P.LINE, lw=0.5, zorder=z)
    ax.plot([c, c], [y - 0.012, y + 0.12], color=P.NAVY, lw=1.0, zorder=z + 1)
    for val, xx, ha in ((lo, x - w / 2, "left"), (hi, x + w / 2, "right")):
        ax.text(xx, y - 0.028, P.fmt_thr(val), ha=ha, va="top", fontsize=4.6, color=P.GREY, zorder=z)
    if log:
        ax.text(x, y - 0.028, "log", ha="center", va="top", fontsize=4.4, color=P.GREY, style="italic", zorder=z)


def fig_class_hist(d, out):
    keep = P.split_strip
    P.split_strip = strip_by_class
    try:
        P.fig_cards(d, out, weighted=True, nodes="production", power=0.30, strip="hist", look="gauge", arrows="side",
                    freq=FREQ)
    finally:
        P.split_strip = keep


# ---------------------------------------------------------------- medallion leaf --------------------------------
def medallion(ax, d, lf, cx, cy, R, vmax, z=4):
    """Leaf as a disc: donut ring = supply mix (% of demand), typical network inside."""
    for off, grow, a in ((0.018, 0.016, 0.04), (0.010, 0.006, 0.07)):
        ax.add_patch(Circle((cx, cy - off), R + grow, fc=P.SHADOW, ec="none", alpha=a, zorder=z - 1))
    ax.add_patch(Circle((cx, cy), R, fc="white", ec="none", zorder=z))
    ring, a0 = 0.20 * R, 90.0
    for name, share in zip(d["meta"]["mix_order"], lf["mix"]):
        a1 = a0 - 360 * share
        ax.add_patch(Wedge((cx, cy), R, a1, a0, width=ring, fc=P.MIX_COLOR[name], ec="white", lw=0.5, zorder=z + 1))
        if share > 0.12:
            am = np.radians((a0 + a1) / 2)
            ax.text(cx + (R - ring / 2) * np.cos(am), cy + (R - ring / 2) * np.sin(am) - 0.003, f"{100 * share:.0f}",
                    ha="center", va="center", fontsize=4.6, color="white", weight="bold", zorder=z + 2)
        a0 = a1
    ax.add_patch(Circle((cx, cy), R - ring, fc="#f7f8fa", ec="none", zorder=z + 1))
    P.draw_network_nested(ax, lf["info"], d["scale"], cx + 0.04 * R, cy, 0.44 * R, vmax=vmax, z=z + 2, **NET)


def leaf_caption(ax, d, lf, x, y, z=6):
    c = d["classes"][lf["cls"]]
    ax.text(x, y + 0.045, c["name"], ha="center", va="center", fontsize=6.2, color=c["color"], weight="bold", zorder=z)
    ax.text(x, y - 0.06, P.leaf_sub(d, lf), ha="center", va="center", fontsize=4.9, color=P.GREY, zorder=z)


def legend_block(fig, ax, d, y, extra):
    yb = P.legend(fig, ax, d, 0.08, y, P.FULL_W - 0.16, rare=True, clusters=False, freq=FREQ)
    P.nested_legend(fig, ax, 0.08, yb - 0.075, NET["mode"], look=NET["look"])
    ax.text(0.08, yb - 0.228, extra, fontsize=5.8, color=P.GREY, ha="left", va="center")


def pill(ax, x, y, d, n, side, z=5, narrow=False):
    thr, word = P.edge_text(d, n, side)
    if narrow and word:  # one word per line, for the short gaps of the horizontal tree
        word = word.replace(" ", chr(10))
    ax.text(x, y, thr + (f"\n{word}" if word else ""), ha="center", va="center", fontsize=5.3, color=P.INK,
            linespacing=1.05, zorder=z,
            bbox=dict(boxstyle="round,pad=0.26,rounding_size=0.6", fc="white", ec="#c3c8d2", lw=0.5))


# ---------------------------------------------------------------- X2: radial tree -------------------------------
def fig_radial(d, out):
    leaves, N, vmax = d["leaves"], d["tree"]["n"], vmax_of(d)
    nl = len(leaves)
    leg_h, R, Rm = 0.80, 2.62, 0.38
    radii = [0.0, 0.88, 1.47, 1.93, 2.1]
    H = leg_h + 2 * (R * 0.93 + Rm + 0.34)
    fig, ax = P.canvas(P.FULL_W, H)
    ox, oy = P.FULL_W / 2, leg_h + R * 0.93 + Rm + 0.34
    xy = lambda r, a: (ox + r * np.cos(a), oy + r * np.sin(a))
    ang = lambda n: np.mean([np.radians(-90 - (lf["pos"] + 0.5) * 360 / nl) for lf in
                             (subtree_leaves(n)[0], subtree_leaves(n)[-1])])
    for g in (radii[1], radii[2], radii[3]):  # faint depth rings
        ax.add_patch(Circle((ox, oy), g, fc="none", ec="#eceef2", lw=0.6, zorder=0))

    for n in d["splits"]:
        r0, a0 = radii[n["depth"]], ang(n)
        for side in ("left", "right"):
            c = n[side]
            a1 = ang(c)
            r1 = radii[c["depth"]] if c["type"] == "split" else R - Rm
            rm = (r0 + r1) / 2
            start = a1 if n["depth"] == 0 else a0
            pts = [xy(r0, start), xy(rm, start), xy(rm, a1), xy(r1, a1)]
            col = d["classes"][c["cls"]]["color"]
            ax.add_patch(PathPatch(Path(pts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]), fc="none",
                                   ec=tint(col, 0.45), lw=max(1.0, 9.0 * c["n"] / N), capstyle="round", zorder=1))
            t = 0.60 if c["type"] == "split" else 0.58
            b = np.array(pts)
            m = (1 - t) ** 3 * b[0] + 3 * (1 - t) ** 2 * t * b[1] + 3 * (1 - t) * t ** 2 * b[2] + t ** 3 * b[3]
            pill(ax, m[0], m[1], d, n, side, z=9)
    for n in d["splits"]:
        x, y = xy(radii[n["depth"]], ang(n))
        lab = P.wrap(d["features"][n["feature"]]["label"], 13)
        hw, hh = 0.34, 0.045 + 0.042 * (lab.count("\n") + 1)
        ax.add_patch(FancyBboxPatch((x - hw, y - hh), 2 * hw, 2 * hh, boxstyle=f"round,pad=0,rounding_size={hh}",
                                    fc=P.NAVY, ec="white", lw=0.8, zorder=6))
        ax.text(x, y - 0.004, lab, ha="center", va="center", fontsize=5.4, color="white", weight="bold",
                linespacing=1.0, zorder=7)
    for lf in leaves:
        a = ang(lf)
        mx, my = xy(R, a)
        medallion(ax, d, lf, mx, my, Rm, vmax)
        leaf_caption(ax, d, lf, mx, my + (Rm + 0.15) * (1 if np.sin(a) > 0.5 else -1))
    legend_block(fig, ax, d, leg_h - 0.07, "ring around a leaf = supply mix  ·  branch width = share of the runs, "
                                           "branch colour = most frequent type downstream")
    P.save(fig, out)


# ---------------------------------------------------------------- X3: medallion tree ----------------------------
def fig_medallions(d, out):
    leaves, levels, N, vmax = d["leaves"], d["levels"], d["tree"]["n"], vmax_of(d)
    nl = len(leaves)
    x0, x1 = 0.06, P.FULL_W - 0.06
    slot = (x1 - x0) / nl
    xs = lambda p: x0 + slot * (p + 0.5)
    labs, hh = P.split_labels(d, 14)
    hw, curve, pill_h, leg_h, Rm = 0.44, 0.22, 0.215, 0.80, 0.405
    dy = 2 * hh + curve + pill_h + 0.12
    cy_leaf = leg_h + 0.30 + Rm
    H = cy_leaf + Rm + 0.20 + 2 * hh + (levels - 1) * dy + curve + pill_h + 0.09
    fig, ax = P.canvas(P.FULL_W, H)
    ys = lambda dep: H - 0.09 - hh - dep * dy

    for n in d["splits"]:
        x, y = xs(n["pos"]), ys(n["depth"])
        py = y - hh
        for side in ("left", "right"):
            c = n[side]
            cx = xs(c["pos"])
            cy = ys(c["depth"]) + hh if c["type"] == "split" else cy_leaf + Rm
            path = Path([(x, py), (x, py - 0.6 * curve), (cx, py - 0.4 * curve), (cx, py - curve), (cx, cy)],
                        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO])
            ax.add_patch(PathPatch(path, fc="none", ec=tint(d["classes"][c["cls"]]["color"], 0.45),
                                   lw=max(1.0, 9.0 * c["n"] / N), capstyle="round", zorder=1))
            pill(ax, cx, py - curve - 0.035 - pill_h / 2 + 0.02, d, n, side)
        # capsule-shaped split node
        for off, grow, a in ((0.016, 0.012, 0.035), (0.010, 0.004, 0.06)):
            ax.add_patch(FancyBboxPatch((x - hw - grow, y - hh - off - grow), 2 * hw + 2 * grow, 2 * hh + 2 * grow,
                                        boxstyle=f"round,pad=0,rounding_size={hh + grow}", fc=P.SHADOW, ec="none",
                                        alpha=a, zorder=5.5))
        ax.add_patch(FancyBboxPatch((x - hw, y - hh), 2 * hw, 2 * hh, boxstyle=f"round,pad=0,rounding_size={hh}",
                                    fc=P.NAVY, ec="none", zorder=6))
        ax.text(x, y - 0.004, labs[n["id"]], ha="center", va="center", fontsize=6.3, color="white", weight="bold",
                linespacing=1.0, zorder=7)
    for lf in leaves:
        medallion(ax, d, lf, xs(lf["pos"]), cy_leaf, Rm, vmax)
        leaf_caption(ax, d, lf, xs(lf["pos"]), cy_leaf - Rm - 0.15)
    legend_block(fig, ax, d, leg_h - 0.07, "ring around a leaf = supply mix  ·  branch width = share of the runs, "
                                           "branch colour = most frequent type downstream")
    P.save(fig, out)


# ---------------------------------------------------------------- X4: smooth densities (scipy) ------------------
def strip_density(ax, d, n, x, y, w, style, z=7):
    """As strip_by_class, with one smooth density curve per type (overlapping) instead of stacked bars."""
    from scipy.stats import gaussian_kde
    f, F = n["feature"], d["features"][n["feature"]]
    lo, hi = F["min"], F["max"]
    cuts = [s["thr"] for s in d["splits"] if s["feature"] == f]
    log = lo > 0 and (np.mean(cuts) - lo) / (hi - lo) < 0.2
    tr = np.log if log else (lambda v: v)
    px = lambda v: x - w / 2 + w * (tr(np.clip(v, lo, hi)) - tr(lo)) / (tr(hi) - tr(lo))
    pts = d["points"]
    sel = np.isin(pts["leaf"], [lf["id"] for lf in subtree_leaves(n)])
    v, cls = px(np.array(pts[f])[sel]), np.array(pts["class"])[sel]
    grid = np.linspace(x - w / 2, x + w / 2, 140)
    dens = []
    for k in range(len(d["classes"])):
        vk = v[cls == k]
        dens.append(gaussian_kde(vk, bw_method=0.22)(grid) * len(vk) if len(vk) > 5 and np.ptp(vk) > 0
                    else np.zeros_like(grid))
    top = max(dk.max() for dk in dens)
    for k in np.argsort([-dk.max() for dk in dens]):  # tallest first, so small types stay visible on top
        col = d["classes"][k]["color"]
        ax.fill_between(grid, y, y + 0.115 * dens[k] / top, fc=tint(col, 0.55), ec=col, lw=0.6, alpha=0.8, zorder=z)
    c = float(px(n["thr"]))
    ax.plot([x - w / 2, x + w / 2], [y, y], color=P.LINE, lw=0.5, zorder=z)
    ax.plot([c, c], [y - 0.012, y + 0.125], color=P.NAVY, lw=1.0, zorder=z + 1)
    for val, xx, ha in ((lo, x - w / 2, "left"), (hi, x + w / 2, "right")):
        ax.text(xx, y - 0.028, P.fmt_thr(val), ha=ha, va="top", fontsize=4.6, color=P.GREY, zorder=z)
    if log:
        ax.text(x, y - 0.028, "log", ha="center", va="top", fontsize=4.4, color=P.GREY, style="italic", zorder=z)


def fig_class_density(d, out):
    keep = P.split_strip
    P.split_strip = strip_density
    try:
        P.fig_cards(d, out, weighted=True, nodes="production", power=0.30, strip="hist", look="gauge", arrows="side",
                    freq=FREQ)
    finally:
        P.split_strip = keep


# ---------------------------------------------------------------- X5: dtreeviz (reference package) --------------
def refit(d):
    """Same tree refitted offline on the drivers stored in the JSON (checked: identical leaf sizes)."""
    from sklearn.tree import DecisionTreeClassifier
    feats = [k for k in d["points"] if k not in ("class", "leaf")]
    X, y = np.column_stack([d["points"][f] for f in feats]), np.array(d["points"]["class"])
    m = DecisionTreeClassifier(max_leaf_nodes=len(d["leaves"]), min_samples_leaf=d["stats"]["min_samples_leaf"],
                               random_state=0).fit(X, y)
    t = m.tree_
    assert sorted(t.n_node_samples[t.children_left == -1]) == sorted(lf["n"] for lf in d["leaves"]), "tree differs"
    return m, X, y, feats


def fig_dtreeviz(d, out):
    os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
    import dtreeviz
    import vl_convert as vlc
    m, X, y, feats = refit(d)
    cols = [c["color"] for c in d["classes"]]
    viz = dtreeviz.model(m, X_train=X, y_train=y, target_name="supply type",
                         feature_names=[d["features"][f]["label"].replace("$_2$", "2") for f in feats],
                         class_names=[c["name"] for c in d["classes"]])
    for name, kw in (("", {}), ("_horizontal", {"orientation": "LR"})):
        v = viz.view(colors={"classes": [None] * len(cols) + [cols]}, fontname="Open Sans", label_fontsize=11,
                     ticks_fontsize=8, **kw)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        v.save(out + name + ".svg")
        open(out + name + ".png", "wb").write(vlc.svg_to_png(open(out + name + ".svg", encoding="utf-8").read(),
                                                               scale=3))
        print("saved", out + name + ".png")


# ---------------------------------------------------------------- X6: plotly icicle / sunburst ------------------
def fig_plotly(d, out):
    import plotly.graph_objects as go
    ids, parents, values, labels, colors = [], [], [], [], []

    def walk(n, parent, edge):
        ids.append(str(n["id"]))
        parents.append(parent)
        values.append(n["n"])
        c = d["classes"][n["cls"]]
        if n["type"] == "leaf":
            labels.append(f"{edge}<b>{c['name']}</b><br>n = {n['n']:,} · {n['purity']:.0%} pure")
            colors.append(c["color"])
            return
        F = d["features"][n["feature"]]
        labels.append(f"{edge}<b>{F['label'].replace('$_2$', '₂')}</b>")
        colors.append("rgb({},{},{})".format(*(int(255 * v) for v in tint(c["color"], 0.28))))
        for side in ("left", "right"):
            thr, word = P.edge_text(d, n, side)
            walk(n[side], str(n["id"]), f"<i>{word} ({thr})</i><br>")

    walk(d["tree"], "", "")
    common = dict(ids=ids, parents=parents, values=values, labels=labels, branchvalues="total", sort=False,
                  marker=dict(colors=colors, line=dict(color="white", width=2)), textinfo="label")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for name, trace, size in (("icicle", go.Icicle(tiling=dict(orientation="v"), **common), (1000, 620)),
                              ("sunburst", go.Sunburst(insidetextorientation="radial", **common), (900, 900))):
        fig = go.Figure(trace)
        fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), font=dict(family="Open Sans", size=13, color=P.NAVY),
                          width=size[0], height=size[1], paper_bgcolor="white", uniformtext=dict(minsize=9))
        fig.write_image(f"{out}_{name}.png", scale=3)
        fig.write_image(f"{out}_{name}.pdf")
        fig.write_image(f"{out}_{name}.svg")
        fig.write_html(f"{out}_{name}.html", include_plotlyjs="cdn")
        print("saved", f"{out}_{name}.png")


# ---------------------------------------------------------------- pictograms, sorting, rules in words -----------
font_manager.fontManager.addfont(r"C:\Windows\Fonts\seguisym.ttf")
ICONS = {"k_large": 0x26A1, "k_small": 0x26A1, "g": 0x1F50C, "supply_margin": 0x1F50C,  # bolt, plug
         "import_availability_ratio": 0x1F6A2, "h2_vs_elec": 0x20AC, "hydrogen_import_price": 0x20AC,  # ship, euro
         "electricity_price_avg": 0x20AC, "unbalance_ratio": 0x2696, "demand_level_ratio": 0x2696,  # scales
         "total_demand_TWh": 0x1F3ED, "solar_cf_mean": 0x2600}  # factory, sun


def badge(ax, t, key, on=True, z=6):
    """Round pictogram of the driver, left of its label t (glyphs of Segoe UI Symbol, live text in the svg)."""
    if not on or key not in ICONS:
        return
    fig = ax.figure
    bb = t.get_window_extent(fig.canvas.get_renderer())
    x, y = bb.x0 / fig.dpi - 0.105, (bb.y0 + bb.y1) / 2 / fig.dpi
    ax.add_patch(Circle((x, y), 0.078, fc=P.NAVY, ec="none", zorder=z))
    ax.text(x, y - 0.004, chr(ICONS[key]), ha="center", va="center", fontsize=6.4, color="white",
            fontfamily="Segoe UI Symbol", zorder=z + 1)


def resort(d):
    """Order the leaves from the least to the most import-reliant (children of every split drawn in that order;
    the branch labels still carry <= and >)."""
    imp = d["meta"]["mix_order"].index("imported")
    share = lambda n: (sum(lf["mix"][imp] * lf["n"] for lf in subtree_leaves(n))
                       / sum(lf["n"] for lf in subtree_leaves(n)))
    order = []

    def walk(n):
        if n["type"] == "leaf":
            n["pos"] = len(order)
            order.append(n)
            return
        for c in sorted((n["left"], n["right"]), key=share):
            walk(c)
        n["pos"] = (n["left"]["pos"] + n["right"]["pos"]) / 2

    walk(d["tree"])
    d["leaves"] = order
    return d


def leaf_paths(d):
    """{leaf id: [(split, side), ...]} from the root down."""
    out = {}

    def walk(n, path):
        if n["type"] == "leaf":
            out[n["id"]] = path
            return
        for side in ("left", "right"):
            walk(n[side], path + [(n, side)])

    walk(d["tree"], [])
    return out


def cond_words(d, n, side, thr=False):
    F = d["features"][n["feature"]]
    t, word = P.edge_text(d, n, side)
    txt = word if " " in word else f"{F['label']}: {word}"
    return f"{txt} ({t})" if thr else txt


# ---------------------------------------------------------------- X7: dtreeviz look, our content ----------------
def open_hist(ax, d, n, xc, yb, w, h, kind, z=5, compact=False, icon=False):
    """dtreeviz-style split node: open histogram of the runs reaching the split, by type (kind = bars | density),
    wedge under the axis at the cut, driver name below. (xc, yb) = centre of the baseline."""
    from scipy.stats import gaussian_kde
    f, F = n["feature"], d["features"][n["feature"]]
    lo, hi = F["min"], F["max"]
    cuts = [s["thr"] for s in d["splits"] if s["feature"] == f]
    log = lo > 0 and (np.mean(cuts) - lo) / (hi - lo) < 0.2
    tr = np.log if log else (lambda v: v)
    px = lambda v: xc - w / 2 + w * (tr(np.clip(v, lo, hi)) - tr(lo)) / (tr(hi) - tr(lo))
    pts = d["points"]
    sel = np.isin(pts["leaf"], [lf["id"] for lf in subtree_leaves(n)])
    v, cls = px(np.array(pts[f])[sel]), np.array(pts["class"])[sel]
    K = range(len(d["classes"]))
    if kind == "bars":
        edges = np.linspace(xc - w / 2, xc + w / 2, 21)
        cnt = np.array([np.histogram(v[cls == k], bins=edges)[0] for k in K])
        top, base = cnt.sum(0).max(), np.zeros(len(edges) - 1)
        for k in K:
            hk = h * cnt[k] / top
            for e0, e1, b0, hh in zip(edges[:-1], edges[1:], base, hk):
                if hh > 0:
                    ax.add_patch(Rectangle((e0, yb + b0), e1 - e0, hh, fc=d["classes"][k]["color"], ec="#2b2f3a",
                                           lw=0.2, zorder=z))
            base += hk
    else:
        grid = np.linspace(xc - w / 2, xc + w / 2, 160)
        dens = [gaussian_kde(v[cls == k], bw_method=0.22)(grid) * (cls == k).sum()
                if (cls == k).sum() > 5 and np.ptp(v[cls == k]) > 0 else np.zeros_like(grid) for k in K]
        top = max(dk.max() for dk in dens)
        for k in np.argsort([-dk.max() for dk in dens]):
            col = d["classes"][k]["color"]
            ax.fill_between(grid, yb, yb + h * dens[k] / top, fc=tint(col, 0.55), ec=col, lw=0.7, alpha=0.8, zorder=z)
    c = float(px(n["thr"]))
    ax.plot([xc - w / 2 - 0.04, xc + w / 2 + 0.04], [yb, yb], color="#444", lw=0.5, zorder=z + 1)
    ax.add_patch(plt.Polygon([(c, yb - 0.008), (c - 0.03, yb - 0.065), (c + 0.03, yb - 0.065)], fc="#333", ec="none",
                             zorder=z + 2))
    ax.plot([c, c], [yb, yb + h + 0.02], color="#333", lw=0.5, ls=(0, (2, 1.5)), zorder=z + 2)
    ax.text(c, yb - 0.075, P.fmt_thr(n["thr"], F["unit"]), ha="center", va="top", fontsize=5.4, color=P.INK,
            weight="bold", zorder=z + 2)
    for val, xx in ((lo, xc - w / 2), (hi, xc + w / 2)):
        ax.plot([xx, xx], [yb, yb - 0.02], color="#444", lw=0.5, zorder=z + 1)
        if abs(xx - c) > 0.20:
            ax.text(xx, yb - 0.035, P.fmt_thr(val), ha="center", va="top", fontsize=4.8, color=P.GREY, zorder=z)
    if compact:  # horizontal layout: narrow node, label on two lines, no run count
        t = ax.text(xc + (0.09 if icon else 0), yb - 0.185, P.wrap(F["label"], 15), ha="center", va="top",
                    fontsize=6.0, color=P.INK, weight="bold", linespacing=1.0, zorder=z)
        badge(ax, t, f, icon)
        if log:
            ax.text(xc + w / 2, yb - 0.105, "log", ha="center", va="top", fontsize=4.4, color=P.GREY,
                    style="italic", zorder=z)
        return
    t = ax.text(xc + (0.09 if icon else 0), yb - 0.185, F["label"] + ("  (log)" if log else ""), ha="center",
                va="top", fontsize=6.4, color=P.INK, weight="bold", zorder=z)
    badge(ax, t, f, icon)
    ax.text(xc - w / 2 - 0.05, yb + h, f"{sel.sum():,}", ha="right", va="top", fontsize=4.6, color=P.GREY, zorder=z)


def fig_dtv(d, out, orient="TD", kind="bars"):
    """Layout and look of dtreeviz (open histograms, thin arrows, leaves at their own depth, type legend box),
    content of G_V3_b / X4 (cut words on the branches, leaf = supply-mix ring + typical network)."""
    from matplotlib.patches import FancyArrowPatch
    leaves, vmax = d["leaves"], vmax_of(d)
    nl, depth_max = len(leaves), max(lf["depth"] for lf in leaves)
    leg_h, Rm = 0.80, 0.37
    if orient == "TD":
        w, h, dy, slot = 1.12, 0.40, 1.36, (P.FULL_W - 0.12) / nl
        H = leg_h + 0.62 + depth_max * dy + 0.52
        pos = lambda n: (0.06 + slot * (n["pos"] + 0.5), H - 0.52 - n["depth"] * dy)
    else:
        w, h, dx, slot = 0.86, 0.36, 1.44, 1.0
        H = leg_h + 0.62 + (nl - 1) * slot + 0.52
        pos = lambda n: (0.56 + n["depth"] * dx, H - 0.52 - n["pos"] * slot)
    fig, ax = P.canvas(P.FULL_W, H)
    for n in d["splits"]:
        x, y = pos(n)
        open_hist(ax, d, n, x, y - 0.06 + (0.10 if orient == "LR" else 0), w, h, kind, compact=orient == "LR")
        name_new(ax, "split_" + slug(d["features"][n["feature"]]["label"]) + f"_{n['id']}")
        for side, sg in (("left", -1), ("right", 1)):
            c = n[side]
            cx, cy = pos(c)
            leaf = c["type"] == "leaf"
            if orient == "TD":
                a, b = (x + sg * 0.20, y - 0.40), (cx, cy + (0.50 if leaf else 0.42))
            else:
                a = (x + w / 2 + 0.10, y + 0.22 - sg * 0.12)
                b = (cx - (Rm + 0.05 if leaf else w / 2 + 0.10), cy + (0.10 if leaf else 0.22))
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>,head_length=0.5,head_width=0.22", mutation_scale=7,
                                         lw=0.6, color="#444", shrinkA=0, shrinkB=0, zorder=2))
            t = 0.5
            pill(ax, a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), d, n, side, z=8)
            name_new(ax, f"branch_{n['id']}_{side}")
    for lf in leaves:
        x, y = pos(lf)
        medallion(ax, d, lf, x, y + 0.10, Rm, vmax)
        leaf_caption(ax, d, lf, x, y + 0.10 - Rm - 0.14)
        name_new(ax, f"leaf{lf['pos'] + 1}_" + slug(d["classes"][lf["cls"]]["name"]))
    # type legend box, as in dtreeviz
    bw, bh = 1.18, 0.16 + 0.15 * len(d["classes"]) + 0.12
    bx, by = (P.FULL_W - bw - 0.10, H - bh - 0.10) if orient == "TD" else (P.FULL_W - bw - 0.10, leg_h + 0.15)
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.03", fc="white", ec="#9aa1ae",
                                lw=0.5, zorder=3))
    ax.text(bx + bw / 2, by + bh - 0.10, "supply type", ha="center", va="center", fontsize=6.0, color=P.INK,
            weight="bold", zorder=4)
    for i, c in enumerate(d["classes"]):
        yy = by + bh - 0.27 - 0.15 * i
        ax.add_patch(Rectangle((bx + 0.10, yy - 0.035), 0.15, 0.07, fc=c["color"], ec="#2b2f3a", lw=0.2, zorder=4))
        ax.text(bx + 0.31, yy - 0.003, c["name"], ha="left", va="center", fontsize=5.8, color=P.INK, zorder=4)
    legend_block(fig, ax, d, leg_h - 0.07, "ring around a leaf = supply mix  ·  histogram in a split = runs reaching "
                                           "it along the sampled range of the driver, by type; wedge = cut")
    P.save(fig, out)


# ---------------------------------------------------------------- X8: open histograms + leaf cards --------------
def type_box(ax, d, bx, by_top):
    """Type legend box (as dtreeviz), top-left corner at (bx, by_top)."""
    bw, bh = 1.18, 0.16 + 0.15 * len(d["classes"]) + 0.12
    by = by_top - bh
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.03", fc="white", ec="#9aa1ae",
                                lw=0.5, zorder=3))
    ax.text(bx + bw / 2, by + bh - 0.10, "supply type", ha="center", va="center", fontsize=6.0, color=P.INK,
            weight="bold", zorder=4)
    for i, c in enumerate(d["classes"]):
        yy = by + bh - 0.27 - 0.15 * i
        ax.add_patch(Rectangle((bx + 0.10, yy - 0.035), 0.15, 0.07, fc=c["color"], ec="#2b2f3a", lw=0.2, zorder=4))
        ax.text(bx + 0.31, yy - 0.003, c["name"], ha="left", va="center", fontsize=5.8, color=P.INK, zorder=4)


def row_card(ax, d, lf, x, cy, w, h, vmax):
    """Leaf as a low wide card (horizontal tree): network left; type, supply-mix bar and runs right."""
    c = d["classes"][lf["cls"]]
    box = P.shadow_box(ax, x, cy - h / 2, w, h, 0.05)
    edge = Rectangle((x, cy - h / 2), 0.045, h, fc=c["color"], ec="none", zorder=2.2)
    ax.add_patch(edge)
    edge.set_clip_path(box)
    P.draw_network_nested(ax, lf["info"], d["scale"], x + 0.47, cy, 0.185, vmax=vmax, **NET)
    tx = x + 0.84
    ax.text(tx, cy + 0.165, c["name"], ha="left", va="center", fontsize=6.2, color=c["color"], weight="bold", zorder=4)
    P.draw_mix(ax, d, lf["mix"], tx, cy - 0.055, x + w - 0.07 - tx, 0.10, fs=5.0)
    ax.text(tx, cy - 0.175, P.leaf_sub(d, lf), ha="left", va="center", fontsize=4.9, color=P.GREY, zorder=4)


def fig_open_cards(d, out, orient="TD", kind="density", icons=False, sort=False, rules=False):
    """dtreeviz-style open histograms in the splits, leaves as cards aligned on one side (as G_V3_b / X4),
    branch width = share of the runs. LR: landscape, leaves as rows on the right."""
    if sort:
        d = resort(P.load(d["meta"]["case"]))
    leaves, N, vmax = d["leaves"], d["tree"]["n"], vmax_of(d)
    nl, leg_h = len(leaves), 0.80
    grey = "#b7bcc9"
    paths = leaf_paths(d)
    rule_h = 0.075 * max(len(v) for v in paths.values()) + 0.08 if rules and orient == "TD" else 0.0
    lw = lambda c: max(0.8, 7.0 * c["n"] / N)
    net = lambda ax_, info, sc, cx, cy, r_: P.draw_network_nested(ax_, info, sc, cx, cy, r_, vmax=vmax, **NET)
    if orient == "TD":
        slot = (P.FULL_W - 0.12) / nl
        xs = lambda p: 0.06 + slot * (p + 0.5)
        w, h, curve, pill_h, r, title_h = 1.00, 0.34, 0.20, 0.215, 0.215, 0.23
        card_h = 0.065 + title_h + 3.9 * r + 0.04 + 0.11 + 0.19
        dy = h + 0.06 + 0.34 + curve + pill_h + 0.10
        leaf_top = leg_h + rule_h + card_h
        H = leaf_top + 0.16 + 0.34 + curve + pill_h + (d["levels"] - 1) * dy + h + 0.12
        fig, ax = P.canvas(P.FULL_W, H)
        yb = lambda dep: H - 0.12 - h - dep * dy
        for n in d["splits"]:
            x, y = xs(n["pos"]), yb(n["depth"])
            open_hist(ax, d, n, x, y, w, h, kind, icon=icons)
            name_new(ax, "split_" + slug(d["features"][n["feature"]]["label"]) + f"_{n['id']}")
            py = y - 0.34
            for side in ("left", "right"):
                c = n[side]
                cx = xs(c["pos"])
                cy = yb(c["depth"]) + h + 0.06 if c["type"] == "split" else leaf_top
                path = Path([(x, py), (x, py - 0.6 * curve), (cx, py - 0.4 * curve), (cx, py - curve), (cx, cy)],
                            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO])
                ax.add_patch(PathPatch(path, fc="none", ec=grey, lw=lw(c), capstyle="round", zorder=1))
                pill(ax, cx, py - curve - 0.035 - pill_h / 2 + 0.02, d, n, side)
                name_new(ax, f"branch_{n['id']}_{side}")
        for lf in leaves:
            P.leaf_card(ax, d, lf, xs(lf["pos"]), leaf_top, slot - 0.07, card_h, r, title_h, False, net)
            if rule_h:  # the rule of the leaf in words
                ax.text(xs(lf["pos"]), leaf_top - card_h - 0.06, chr(10).join(cond_words(d, *c) for c in paths[lf["id"]]),
                        ha="center", va="top", fontsize=4.5, color=P.INK, linespacing=1.18, zorder=4)
            name_new(ax, f"leaf{lf['pos'] + 1}_" + slug(d["classes"][lf["cls"]]["name"]))
        type_box(ax, d, P.FULL_W - 1.28, H - 0.08)
        name_new(ax, "type_box")
    else:
        pitch, rh, cw = 0.66, 0.58, 1.80
        w, h, dx, curve = 0.66, 0.32, 1.22, 0.25
        H = leg_h + 0.12 + nl * pitch + 0.06
        fig, ax = P.canvas(P.FULL_W, H)
        ys = lambda p: H - 0.08 - pitch * (p + 0.5)
        xc = lambda dep: 0.45 + dep * dx
        card_x = P.FULL_W - 0.06 - cw
        for n in d["splits"]:
            x, y = xc(n["depth"]), ys(n["pos"])
            open_hist(ax, d, n, x, y - 0.06, w, h, kind, compact=True, icon=icons)
            name_new(ax, "split_" + slug(d["features"][n["feature"]]["label"]) + f"_{n['id']}")
            a = (x + w / 2 + 0.07, y + 0.08)
            for side in ("left", "right"):
                c = n[side]
                cy = ys(c["pos"])
                if c["type"] == "split":
                    b = (xc(c["depth"]) - w / 2 - 0.09, cy + 0.08)
                    mid = (a[0] + b[0]) / 2
                    path = Path([a, (mid, a[1]), (mid, b[1]), b], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
                    px_, py_ = mid, (a[1] + b[1]) / 2
                else:
                    b = (card_x, cy)
                    path = Path([a, (a[0] + 0.6 * curve, a[1]), (a[0] + 0.4 * curve, cy), (a[0] + curve, cy), b],
                                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO])
                    px_, py_ = min(a[0] + curve + 0.40, (a[0] + curve + b[0]) / 2), cy
                ax.add_patch(PathPatch(path, fc="none", ec=grey, lw=lw(c), capstyle="round", zorder=1))
                pill(ax, px_, py_, d, n, side, z=8, narrow=c["type"] == "split")
                name_new(ax, f"branch_{n['id']}_{side}")
        for lf in leaves:
            row_card(ax, d, lf, card_x, ys(lf["pos"]), cw, rh, vmax)
            name_new(ax, f"leaf{lf['pos'] + 1}_" + slug(d["classes"][lf["cls"]]["name"]))
        type_box(ax, d, 0.08, H - 0.08)
        name_new(ax, "type_box")
    legend_block(fig, ax, d, leg_h - 0.07, "branch width = share of the runs  ·  curves in a split = runs reaching it "
                                           "along the sampled range of the driver, by type; wedge = cut")
    name_new(ax, "legend")
    P.save(fig, out)


# ---------------------------------------------------------------- X10: type portraits ---------------------------
def aggregate(leaves):
    """Run-weighted mean leaf of a group of leaves: (info, mix, runs)."""
    W = np.array([lf["n"] for lf in leaves], float)
    info = {}
    for k, v0 in leaves[0]["info"].items():
        vals = [lf["info"][k] for lf in leaves]
        if k in P.LINKS:
            fr = W * np.array([v[0] for v in vals])
            info[k] = [fr.sum() / W.sum()] + [float((fr * np.array([v[i] for v in vals])).sum() / max(fr.sum(), 1e-9))
                                              for i in (1, 2)]
        elif isinstance(v0, (int, float)):
            info[k] = float((W * np.array(vals)).sum() / W.sum())
    mix = (W[:, None] * np.array([lf["mix"] for lf in leaves])).sum(0) / W.sum()
    return info, mix, int(W.sum())


def fig_portraits(d, out):
    """One panel per supply type: its typical system, large, and the conditions that lead to it. The answer
    first, the rule second (no tree drawn)."""
    N, paths, vmax = d["tree"]["n"], leaf_paths(d), vmax_of(d)
    imp = d["meta"]["mix_order"].index("imported")
    groups = [(c, [lf for lf in d["leaves"] if lf["cls"] == k]) for k, c in enumerate(d["classes"])]
    groups = [(c, ls, aggregate(ls)) for c, ls in groups if ls]
    groups.sort(key=lambda g: g[2][1][imp])
    leg_h, gap, lh = 0.80, 0.10, 0.128
    pw = (P.FULL_W - 0.12 - gap * (len(groups) - 1)) / len(groups)
    rows = max(sum(len(paths[lf["id"]]) + 1.4 for lf in ls) for _, ls, _ in groups)
    ph = 0.62 + 1.70 + 0.42 + 0.30 + rows * lh + 0.06
    H = leg_h + ph + 0.14
    fig, ax = P.canvas(P.FULL_W, H)
    for i, (c, ls, (info, mix, n)) in enumerate(groups):
        x, top = 0.06 + i * (pw + gap), H - 0.08
        box = P.shadow_box(ax, x, top - ph, pw, ph, 0.07)
        cap = Rectangle((x, top - 0.06), pw, 0.06, fc=c["color"], ec="none", zorder=2.2)
        ax.add_patch(cap)
        cap.set_clip_path(box)
        cx = x + pw / 2
        ax.text(cx, top - 0.27, c["name"], ha="center", va="center", fontsize=10, color=c["color"], weight="bold",
                zorder=4)
        ax.text(cx, top - 0.47, f"{n / N:.0%} of the runs  ·  n = {n:,}", ha="center", va="center", fontsize=5.8,
                color=P.GREY, zorder=4)
        P.draw_network_nested(ax, info, d["scale"], cx + 0.05, top - 0.62 - 0.85, 0.50, vmax=vmax, **NET)
        yb_ = top - 0.62 - 1.70 - 0.30
        P.draw_mix(ax, d, mix, x + 0.14, yb_, pw - 0.28, 0.15, fs=6.2)
        y = yb_ - 0.22
        ax.text(x + 0.14, y, "Reached when", ha="left", va="center", fontsize=6.2, color=P.INK, weight="bold", zorder=4)
        y -= 0.06
        for lf in sorted(ls, key=lambda q: -q["n"]):
            y -= lh * 1.25
            ax.add_patch(Circle((x + 0.18, y), 0.028, fc=c["color"], ec="none", zorder=4))
            ax.text(x + 0.26, y - 0.003, f"{lf['n'] / N:.0%} of the runs", ha="left", va="center", fontsize=5.4,
                    color=c["color"], weight="bold", zorder=4)
            for nd, side in paths[lf["id"]]:
                y -= lh
                F, (thr, word) = d["features"][nd["feature"]], P.edge_text(d, nd, side)
                t = ax.text(x + 0.26, y - 0.003, F["label"], ha="left", va="center", fontsize=5.4, color=P.GREY,
                            zorder=4)
                ax.text(x + pw - 0.12, y - 0.003, f"{word}  {thr}", ha="right", va="center", fontsize=5.4,
                        color=P.INK, zorder=4)
            y -= lh * 0.15
        name_new(ax, "portrait_" + slug(c["name"]))
    legend_block(fig, ax, d, leg_h - 0.07, "each panel = all leaves of one type (run-weighted mean of the leaves)  ·  "
                                           "conditions: one block per leaf of the tree")
    name_new(ax, "legend")
    P.save(fig, out)


# ---------------------------------------------------------------- X11: phase map --------------------------------
def fig_phase(d, out):
    """Phase diagram: plane of the first two drivers of the tree, cut lines at the tree's thresholds, one cell
    per leaf (cells inside a quadrant = the deeper splits), typical network in every cell."""
    root = d["tree"]
    kids = [root["left"], root["right"]]
    if any(k["type"] != "split" for k in kids) or kids[0]["feature"] != kids[1]["feature"]:
        print("X11 skipped: needs a tree whose two first-level branches split on the same driver")
        return
    vmax, paths = vmax_of(d), leaf_paths(d)
    leg_h = 0.80
    X0, X1, Y0, Y1 = 0.78, P.FULL_W - 0.10, leg_h + 0.66, leg_h + 0.66 + 5.25
    H = Y1 + 0.12
    fig, ax = P.canvas(P.FULL_W, H)

    def scale(f, thr):  # linear or log, whichever puts the cut nearer the middle
        F = d["features"][f]
        lo, hi = F["min"], F["max"]
        lin = (thr - lo) / (hi - lo)
        lg = np.log(thr / lo) / np.log(hi / lo) if lo > 0 else -9
        log = abs(lg - 0.5) < abs(lin - 0.5)
        return (lambda v: np.log(v / lo) / np.log(hi / lo)) if log else (lambda v: (v - lo) / (hi - lo)), log, lo, hi

    fx, fy = root["feature"], kids[0]["feature"]
    sx, logx, xlo, xhi = scale(fx, root["thr"])
    xc_ = X0 + (X1 - X0) * sx(root["thr"])
    g = 0.035
    for kid, (xa, xb) in zip(kids, ((X0, xc_), (xc_, X1))):
        sy, logy, ylo, yhi = scale(fy, kid["thr"])
        yc_ = Y0 + (Y1 - Y0) * sy(kid["thr"])
        for sub, (ya, yb_) in ((kid["left"], (Y0, yc_)), (kid["right"], (yc_, Y1))):
            ls = subtree_leaves(sub)
            cw = (xb - xa) / len(ls)
            for j, lf in enumerate(ls):
                c = d["classes"][lf["cls"]]
                cx0, cx1 = xa + j * cw, xa + (j + 1) * cw
                ax.add_patch(FancyBboxPatch((cx0 + g, ya + g), cw - 2 * g, yb_ - ya - 2 * g,
                                            boxstyle="round,pad=0,rounding_size=0.05", fc=tint(c["color"], 0.13),
                                            ec=tint(c["color"], 0.55), lw=0.6, zorder=1))
                cx, cy = (cx0 + cx1) / 2, (ya + yb_) / 2
                extra = paths[lf["id"]][2:]
                if extra:
                    ax.text(cx, yb_ - g - 0.07, chr(10).join((cond_words(d, *e).replace(": ", ":" + chr(10)) if cw < 1.3 else cond_words(d, *e)) + "  " + P.edge_text(d, *e)[0] for e in extra),
                            ha="center", va="top", fontsize=5.0, color=P.INK, linespacing=1.12, zorder=4)
                ax.text(cx, cy + 0.60, c["name"], ha="center", va="center", fontsize=6.6, color=c["color"],
                        weight="bold", zorder=4)
                P.draw_network_nested(ax, lf["info"], d["scale"], cx + 0.03, cy + 0.10, 0.225, vmax=vmax, **NET)
                bw = min(cw - 0.22, 1.15)
                P.draw_mix(ax, d, lf["mix"], cx - bw / 2, cy - 0.44, bw, 0.11, fs=5.0)
                ax.text(cx, cy - 0.56, P.leaf_sub(d, lf), ha="center", va="center", fontsize=5.0, color=P.GREY, zorder=4)
                name_new(ax, f"cell_{lf['pos'] + 1}_" + slug(c["name"]))
        # y cut of this half, with its value
        ax.plot([xa + g, xb - g], [yc_, yc_], color=P.NAVY, lw=1.1, zorder=3)
        if kid is kids[0]:
            Fy = d["features"][fy]
            for v, yy in ((ylo, Y0), (kid["thr"], yc_), (yhi, Y1)):
                ax.text(X0 - 0.06, yy, P.fmt_thr(v), ha="right", va="center", fontsize=5.6,
                        color=P.INK if yy == yc_ else P.GREY, weight="bold" if yy == yc_ else "normal")
            for word, yy in ((Fy["lo_word"], (Y0 + yc_) / 2), (Fy["hi_word"], (yc_ + Y1) / 2)):
                ax.text(X0 - 0.30, yy, word, ha="center", va="center", fontsize=6.4, color=P.INK, rotation=90)
            ax.text(X0 - 0.55, (Y0 + Y1) / 2, Fy["label"] + ("  (log scale)" if logy else ""), ha="center",
                    va="center", fontsize=7.0, color=P.INK, weight="bold", rotation=90)
    ax.plot([xc_, xc_], [Y0 + g, Y1 - g], color=P.NAVY, lw=1.1, zorder=3)
    Fx = d["features"][fx]
    for v, xx in ((xlo, X0), (root["thr"], xc_), (xhi, X1)):
        ax.text(xx, Y0 - 0.06, P.fmt_thr(v), ha="center", va="top", fontsize=5.6,
                color=P.INK if xx == xc_ else P.GREY, weight="bold" if xx == xc_ else "normal")
    for word, xx in ((Fx["lo_word"], (X0 + xc_) / 2), (Fx["hi_word"], (xc_ + X1) / 2)):
        ax.text(xx, Y0 - 0.25, word, ha="center", va="center", fontsize=6.4, color=P.INK)
    ax.text((X0 + X1) / 2, Y0 - 0.46, Fx["label"] + ("  (log scale)" if logx else ""), ha="center", va="center",
            fontsize=7.0, color=P.INK, weight="bold")
    name_new(ax, "axes_labels")
    legend_block(fig, ax, d, leg_h - 0.07, "dark lines = cuts of the tree on the two main drivers  ·  cells side by "
                                           "side in a quadrant = further splits (condition on top), not to scale")
    name_new(ax, "legend")
    P.save(fig, out)


IDEAS = {"X9a": ("X9a_icons_sorted_rules_vertical",
                 lambda d, out: fig_open_cards(d, out, "TD", "density", icons=True, sort=True, rules=True)),
         "X9c": ("X9c_icons_sorted_horizontal",
                 lambda d, out: fig_open_cards(d, out, "LR", "density", icons=True, sort=True)),
         "X10": ("X10_type_portraits", fig_portraits), "X11": ("X11_phase_map", fig_phase),
         "X8a": ("X8a_open_hist_cards_density", lambda d, out: fig_open_cards(d, out, "TD", "density")),
         "X8b": ("X8b_open_hist_cards_bars", lambda d, out: fig_open_cards(d, out, "TD", "bars")),
         "X8c": ("X8c_open_hist_cards_horizontal_density", lambda d, out: fig_open_cards(d, out, "LR", "density")),
         "X8d": ("X8d_open_hist_cards_horizontal_bars", lambda d, out: fig_open_cards(d, out, "LR", "bars"))}
IDEAS = {**IDEAS, "X7a":("X7a_dtreeviz_style_bars", lambda d, out: fig_dtv(d, out, "TD", "bars")),
         "X7b": ("X7b_dtreeviz_style_density", lambda d, out: fig_dtv(d, out, "TD", "density")),
         "X7c": ("X7c_dtreeviz_style_horizontal_bars", lambda d, out: fig_dtv(d, out, "LR", "bars")),
         "X7d": ("X7d_dtreeviz_style_horizontal_density", lambda d, out: fig_dtv(d, out, "LR", "density"))}
IDEAS = {**{"X1": ("X1_class_histograms", fig_class_hist), "X2": ("X2_radial_tree", fig_radial),
         "X3": ("X3_medallion_tree", fig_medallions), "X4": ("X4_class_densities", fig_class_density),
         "X5": ("X5_dtreeviz", fig_dtreeviz), "X6": ("X6_plotly", fig_plotly)}, **IDEAS}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="sysmix_sysmix_clf_k03_supply_L8")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    d = P.load(a.case)
    for key, (name, fn) in IDEAS.items():
        if not a.only or key in a.only:
            fn(d, os.path.join(P.PAPER_DIR, a.case, "esplorative", name))
