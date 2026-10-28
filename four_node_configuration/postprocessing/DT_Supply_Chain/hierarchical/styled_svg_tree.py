"""
Render the styled schematic driver tree as ONE native SVG (no browser, no pagination).

The existing tree.html already embeds everything we need as JSON:
  const DATA = {tree, scale}   -- every split + every leaf's aggregated network info
  const FEAT, SUB, COLOR, PAIRS
We parse those and re-emit the exact layout (split cards, curved edges, branch labels
and the 4-node hydrogen-network leaf schematics) as a single standalone SVG. Text stays
real <text> (editable in Illustrator); shapes stay vector. Nothing is clipped.

Geometry ports render_driver_tree_html.py's JS (LEAFW/VGAP/TOP, node positions, leafSVG).

Run:  python styled_svg_tree.py
Out:  DT_supplychain/vector/<tag>_styled.svg  (+ tree_styled.svg next to each tree.html)
"""

import html as htmllib
import json
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

# --- layout constants (must match render_driver_tree_html.py) ----------------
LEAFW, VGAP, TOP = 216, 136, 42
CARD_W, CARD_H = 196, 300
# node positions inside a 196x196 leaf schematic
NPOS = {"L1": (98, 64), "L2": (98, 132), "S1": (36, 98), "S2": (160, 98)}
LK = {"L1L2": ("L1", "L2"), "L1S1": ("L1", "S1"), "L1S2": ("L1", "S2"),
      "L2S1": ("L2", "S1"), "L2S2": ("L2", "S2"), "S1S2": ("S1", "S2")}

BASE = C.BASE_FIG_DIR
VECTOR_DIR = os.path.join(BASE, "vector")
CASES = [
    (os.path.join(BASE, "strat", "supply", "all", "k03", "tree.html"), "supply_all_k3"),
    (os.path.join(BASE, "decentr", "A", "robust", "k04", "tree.html"), "decentrA_robust_k4"),
]


def sq(v, m):
    return math.sqrt(max(v, 0) / m) if m else 0.0


def esc(s):
    return htmllib.escape(str(s), quote=True)


def _extract(html, start, end):
    """Slice the JSON literal between two unique markers and parse it."""
    seg = html.split(start, 1)[1]
    seg = seg.split(end, 1)[0]
    return json.loads(seg.strip())


def parse_html(path):
    html = open(path, encoding="utf-8").read()
    data = _extract(html, "const DATA=", ", FEAT=")
    feat = _extract(html, ", FEAT=", ", SUB=")
    sub = _extract(html, ", SUB=", ";")
    color = _extract(html, "const COLOR=", ";")
    pairs = _extract(html, "const PAIRS=", ";")
    return data, feat, sub, color, pairs


# --- layout (in-order leaf x; depth-based y) ---------------------------------
def layout(node, depth, ctr):
    if node["type"] == "leaf":
        node["x"] = ctr[0] * LEAFW + LEAFW / 2
        node["y"] = TOP + depth * VGAP
        ctr[0] += 1
        return node["x"]
    a = layout(node["left"], depth + 1, ctr)
    b = layout(node["right"], depth + 1, ctr)
    node["x"] = (a + b) / 2
    node["y"] = TOP + depth * VGAP
    return node["x"]


def n_leaves(n):
    return 1 if n["type"] == "leaf" else n_leaves(n["left"]) + n_leaves(n["right"])


def max_depth(n):
    return 0 if n["type"] == "leaf" else 1 + max(max_depth(n["left"]), max_depth(n["right"]))


# --- leaf 4-node schematic (port of JS leafSVG), drawn in 196x196 space -------
def schematic(i, sc):
    s = []
    T_MAIN, T_OCC, T_RARE = 0.60, 0.30, 0.10
    for k, (a, b) in LK.items():
        built, capv, hpv = i[k]
        if built < T_RARE:
            continue
        x1, y1 = NPOS[a]
        x2, y2 = NPOS[b]
        col = "#C00935" if hpv > 0.5 else "#161D41"
        if built >= T_MAIN:
            w = max(1.2, sq(capv, sc["pipe_max"]) * 11)
            s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
                     f'stroke-width="{w:.1f}" stroke-linecap="round"/>')
        elif built >= T_OCC:
            w = max(1.2, sq(capv, sc["pipe_max"]) * 11)
            s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
                     f'stroke-width="{w:.1f}" stroke-opacity="0.45" stroke-dasharray="4 4" '
                     f'stroke-linecap="round"/>')
        else:
            s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#8a8f9c" '
                     f'stroke-width="1.4" stroke-opacity="0.55" stroke-dasharray="2.5 3.5" '
                     f'stroke-linecap="round"/>')
    for n, key, dr in (("L1", "imp_L1", -1), ("L2", "imp_L2", 1)):
        v = i.get(key, 0)
        if not v or v < 8000:
            continue
        x, y = NPOS[n]
        f = sq(v, sc["import_max"])
        boxH = max(15, sq(i["cap_" + n], sc["large_elzr_max"]) * 34)
        gap, LEN = 6, 20
        w = round(2.4 + f * 2.6, 1)
        yTip = y + dr * (boxH / 2 + gap)
        yTail = yTip + dr * LEN
        headLen = 3.6 * w
        yShaftEnd = yTip + dr * (headLen * 0.55)
        s.append(f'<line x1="{x}" y1="{yTail:.1f}" x2="{x}" y2="{yShaftEnd:.1f}" '
                 f'stroke="#b8900a" stroke-width="{w}" stroke-linecap="butt"/>')
        s.append(f'<path d="M {x} {yTip:.1f} L {x-headLen*0.34:.1f} {yTip+dr*headLen:.1f} '
                 f'L {x+headLen*0.34:.1f} {yTip+dr*headLen:.1f} Z" fill="#b8900a"/>')
    for n, (x, y) in NPOS.items():
        if n[0] == "L":
            f = sq(i["cap_" + n], sc["large_elzr_max"])
            w, h = max(20, f * 46), max(15, f * 34)
            s.append(f'<rect x="{x-w/2:.1f}" y="{y-h/2:.1f}" width="{w:.1f}" height="{h:.1f}" '
                     f'rx="6" fill="#161D41"/>')
            s.append(f'<text x="{x}" y="{y+4}" font-size="11" font-weight="bold" '
                     f'text-anchor="middle" fill="#fff">{n}</text>')
        else:
            f = sq(i["cap_" + n], sc["small_elzr_max"])
            r = max(6, f * 17)
            s.append(f'<circle cx="{x}" cy="{y}" r="{r:.1f}" fill="#63A593" '
                     f'stroke="#161D41" stroke-width="1.6"/>')
            s.append(f'<text x="{x}" y="{y+3.5}" font-size="9" font-weight="bold" '
                     f'text-anchor="middle" fill="#161D41">{n}</text>')
    return "".join(s)


# --- leaf card ---------------------------------------------------------------
def leaf_card(n, sc, sub, color):
    i = n["info"]
    x, y = n["x"], n["y"]
    lab = i["label"]
    c = color.get(lab, "#8a8f9c")
    imp = round(i["impfrac"] * 100)
    loc = 100 - imp
    left = x - CARD_W / 2
    s = [f'<g>']
    # card body + coloured top border
    s.append(f'<rect x="{left:.1f}" y="{y:.1f}" width="{CARD_W}" height="{CARD_H}" rx="12" '
             f'fill="#ffffff" stroke="#b7bcc9" stroke-width="1"/>')
    s.append(f'<rect x="{left:.1f}" y="{y:.1f}" width="{CARD_W}" height="4" rx="2" fill="{c}"/>')
    s.append(f'<text x="{x}" y="{y+18:.1f}" font-size="12.5" font-weight="bold" '
             f'text-anchor="middle" fill="{c}">{esc(lab)}</text>')
    desc = sub.get(lab, "")
    if desc:
        s.append(f'<text x="{x}" y="{y+30:.1f}" font-size="9" text-anchor="middle" '
                 f'fill="#8a8f9c">{esc(desc)}</text>')
    # schematic (196 space) offset inside the card
    s.append(f'<g transform="translate({left+5:.1f},{y+36:.1f})">{schematic(i, sc)}</g>')
    ss = round(i["S1S2"][0] * 100)
    s.append(f'<text x="{x}" y="{y+232:.1f}" font-size="8" font-style="italic" '
             f'text-anchor="middle" fill="#8a8f9c">small-small link in {ss}% of runs</text>')
    # supply-mix bar (local teal | import amber)
    bx, bw = left + 6, CARD_W - 12
    s.append(f'<rect x="{bx:.1f}" y="{y+238:.1f}" width="{bw*loc/100:.1f}" height="9" fill="#63A593"/>')
    s.append(f'<rect x="{bx+bw*loc/100:.1f}" y="{y+238:.1f}" width="{bw*imp/100:.1f}" height="9" fill="#b8900a"/>')
    s.append(f'<rect x="{bx:.1f}" y="{y+238:.1f}" width="{bw:.1f}" height="9" fill="none" '
             f'stroke="#dfe2e8" stroke-width="1"/>')
    s.append(f'<text x="{x}" y="{y+256:.1f}" font-size="8" text-anchor="middle" '
             f'fill="#8a8f9c">local {loc}%  ·  import {imp}%</text>')
    s.append(f'<text x="{left+8:.1f}" y="{y+270:.1f}" font-size="9" fill="#8a8f9c">N={i["n"]}</text>')
    s.append(f'<text x="{left+CARD_W-8:.1f}" y="{y+270:.1f}" font-size="9" text-anchor="end" '
             f'fill="#8a8f9c">pure {round(i["purity"]*100)}%</text>')
    s.append(f'<text x="{x}" y="{y+282:.1f}" font-size="8.5" text-anchor="middle" '
             f'fill="#8a8f9c">price {i["avg_price"]} EUR/MWh · demand {i["avg_demand"]} TWh</text>')
    # flags
    flags = []
    if i.get("dual", 0) >= 0.10:
        flags.append(("dual %d%%" % round(i["dual"] * 100), "#6b4e9e", "#efeaf7"))
    if i.get("mesh", 0) >= 0.10:
        flags.append(("mesh %d%%" % round(i["mesh"] * 100), "#C00935", "#fdeff2"))
    fx = x - sum(len(t) * 5.4 + 12 for t, _, _ in flags) / 2
    for t, fg, bg in flags:
        w = len(t) * 5.4 + 12
        s.append(f'<rect x="{fx:.1f}" y="{y+288:.1f}" width="{w:.1f}" height="13" rx="6" fill="{bg}"/>')
        s.append(f'<text x="{fx+w/2:.1f}" y="{y+297:.1f}" font-size="8.5" font-weight="bold" '
                 f'text-anchor="middle" fill="{fg}">{esc(t)}</text>')
        fx += w + 4
    s.append("</g>")
    return "".join(s)


# --- split card + edges + branch labels --------------------------------------
def split_card(n, feat):
    x, y = n["x"], n["y"]
    lab, unit = feat.get(n["feature"], [n["feature"], ""])
    thr = n["thr"]
    thr_s = f"≤ {thr}" + (f" {unit}" if unit else "")
    w, h = 128, 40
    s = [f'<g>']
    s.append(f'<rect x="{x-w/2:.1f}" y="{y-h/2:.1f}" width="{w}" height="{h}" rx="10" '
             f'fill="#ffffff" stroke="#161D41" stroke-width="1.6"/>')
    s.append(f'<text x="{x}" y="{y-2:.1f}" font-size="12" font-weight="bold" '
             f'text-anchor="middle" fill="#161D41">{esc(lab)}</text>')
    s.append(f'<rect x="{x-len(thr_s)*3.4-6:.1f}" y="{y+3:.1f}" width="{len(thr_s)*6.8+12:.1f}" '
             f'height="14" rx="5" fill="#eef0f4"/>')
    s.append(f'<text x="{x}" y="{y+13:.1f}" font-size="10.5" text-anchor="middle" '
             f'fill="#161D41" font-family="monospace">{esc(thr_s)}</text>')
    s.append("</g>")
    return "".join(s)


def edges_and_labels(n, pairs):
    if n["type"] == "leaf":
        return ""
    out = []
    py = n["y"] + 20
    p = pairs.get(n["feature"], ["≤", ">"])
    for c, txt in ((n["left"], p[0]), (n["right"], p[1])):
        cy = c["y"] if c["type"] == "leaf" else c["y"] - 20
        my = (py + cy) / 2
        out.append(f'<path d="M{n["x"]:.1f},{py:.1f} C{n["x"]:.1f},{my:.1f} '
                   f'{c["x"]:.1f},{my:.1f} {c["x"]:.1f},{cy:.1f}" fill="none" '
                   f'stroke="#b7bcc9" stroke-width="1.7"/>')
        # branch label pill
        lx, ly = (n["x"] + c["x"]) / 2, (n["y"] + 20 + cy) / 2
        lw = len(txt) * 5.6 + 12
        out.append(f'<rect x="{lx-lw/2:.1f}" y="{ly-8:.1f}" width="{lw:.1f}" height="16" rx="8" '
                   f'fill="#fbfbfa" stroke="#b7bcc9" stroke-width="1"/>')
        out.append(f'<text x="{lx:.1f}" y="{ly+3:.1f}" font-size="10" font-weight="600" '
                   f'text-anchor="middle" fill="#161D41">{esc(txt)}</text>')
    out.append(edges_and_labels(n["left"], pairs))
    out.append(edges_and_labels(n["right"], pairs))
    return "".join(out)


def nodes(n, sc, sub, color, feat):
    if n["type"] == "leaf":
        return leaf_card(n, sc, sub, color)
    return split_card(n, feat) + nodes(n["left"], sc, sub, color, feat) + \
        nodes(n["right"], sc, sub, color, feat)


def build_svg(path):
    data, feat, sub, color, pairs = parse_html(path)
    tree, sc = data["tree"], data["scale"]
    layout(tree, 0, [0])
    W = n_leaves(tree) * LEAFW
    H = TOP + max_depth(tree) * VGAP + CARD_H + 20
    PAD = 14
    body = edges_and_labels(tree, pairs) + nodes(tree, sc, sub, color, feat)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W+2*PAD}" height="{H+2*PAD}" '
           f'viewBox="0 0 {W+2*PAD} {H+2*PAD}" font-family="Arial, Helvetica, sans-serif">'
           f'<rect width="100%" height="100%" fill="#fbfbfa"/>'
           f'<g transform="translate({PAD},{PAD})">{body}</g></svg>')
    return svg, W + 2 * PAD, H + 2 * PAD


def main():
    os.makedirs(VECTOR_DIR, exist_ok=True)
    for src, tag in CASES:
        if not os.path.exists(src):
            print(f"  MISSING: {src}"); continue
        svg, w, h = build_svg(src)
        local = os.path.join(os.path.dirname(src), "tree_styled.svg")
        dst = os.path.join(VECTOR_DIR, f"{tag}_styled.svg")
        for p in (local, dst):
            with open(p, "w", encoding="utf-8") as f:
                f.write(svg)
        print(f"  {tag}: {w}x{h}px  -> {dst} ({len(svg)//1024} KB)")
    print(f"\nNative styled SVGs -> {VECTOR_DIR}")


if __name__ == "__main__":
    main()