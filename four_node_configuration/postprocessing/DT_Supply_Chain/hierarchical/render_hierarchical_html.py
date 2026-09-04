"""
Schematic-leaf HTML for the SINGLE hierarchical tree (same representation as the
k=4 driver_tree_magnitude.html): splits test sampled inputs, each leaf is a drawn
4-node network aggregated over its runs, labelled by the majority "Origin >
Structure" hierarchical group.

Reuses the renderer/aggregation from render_driver_tree_html.py.

Configure via env:  STAGE_K1 (2), STAGE_K2 (3), TREE_LEAVES (12)
Run:  python render_hierarchical_html.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import render_driver_tree_html as R
from hierarchical_clustering import LEVEL1, LEVEL2, name_l1, name_l2

K1 = int(os.environ.get("STAGE_K1", 2))
K2 = int(os.environ.get("STAGE_K2", 3))
LEAVES = int(os.environ.get("TREE_LEAVES", 12))
OUT_DIR = os.path.join(C.BASE_FIG_DIR, "hierarchical", "single")


def _z(df, cols):
    return StandardScaler().fit_transform(df[cols].fillna(0.0))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    feats = pd.read_csv(C.FEATURES_CSV)
    raw = pd.read_excel(C.EXTRACTED_RESULTS)
    df = feats.merge(raw, on="run", how="left", suffixes=("", "_raw"))

    # hierarchical combined label + nested names
    df["L1"] = KMeans(K1, random_state=C.RANDOM_STATE, n_init=10).fit_predict(_z(df, LEVEL1))
    l1_names = {g: name_l1(df.loc[df["L1"] == g, LEVEL1].mean()) for g in range(K1)}
    df["L2"] = -1
    for g in range(K1):
        m = df["L1"] == g
        df.loc[m, "L2"] = KMeans(K2, random_state=C.RANDOM_STATE, n_init=10).fit_predict(_z(df[m], LEVEL2))
    code, names = {}, {}
    c = 0
    for g in range(K1):
        for h in range(K2):
            code[(g, h)] = c
            mm = (df["L1"] == g) & (df["L2"] == h)
            names[c] = f"{l1_names[g]} > {name_l2(df.loc[mm, LEVEL2].mean())}"
            c += 1
    df["cluster_final"] = df.apply(lambda r: code[(r["L1"], r["L2"])], axis=1).astype(int)

    inputs = [col for col in C.DRIVER_INPUTS if col in df.columns]
    X = df[inputs].fillna(0.0)
    clf = DecisionTreeClassifier(max_leaf_nodes=LEAVES, min_samples_leaf=20,
                                 min_samples_split=20, class_weight="balanced",
                                 random_state=C.RANDOM_STATE).fit(X, df["cluster_final"])
    df["_leaf"] = clf.apply(X)
    leaf_rows = {lid: df[df["_leaf"] == lid] for lid in np.unique(df["_leaf"])}
    name_by_cluster = {i: names[i] for i in range(K1 * K2)}
    tree_dict = R._build_node(clf.tree_, 0, inputs, leaf_rows, name_by_cluster)

    scale = {
        "pipe_max": round(max(max(nd["info"][k][1] for k in R.ARC_COLS)
                              for nd in R._leaves(tree_dict)), 0) or 1.0,
        "large_elzr_max": round(max(max(nd["info"]["cap_L1"], nd["info"]["cap_L2"])
                                    for nd in R._leaves(tree_dict)), 0),
        "small_elzr_max": round(max(max(nd["info"]["cap_S1"], nd["info"]["cap_S2"])
                                    for nd in R._leaves(tree_dict)), 0),
        "import_max": round(max(max(nd["info"]["imp_L1"], nd["info"]["imp_L2"])
                                for nd in R._leaves(tree_dict)), 0),
    }
    data = {"tree": tree_dict, "scale": scale}

    # colours: base per supply origin, shaded per structure
    base = {0: C.UU_NAVY, 1: C.UU_TEAL, 2: C.UU_YELLOW}
    R.COLOR = {}
    R.SUB = {}
    for g in range(K1):
        for h in range(K2):
            nm = names[code[(g, h)]]
            R.COLOR[nm] = base.get(g, "#8a8f9c") if h == 0 else _shade(base.get(g, "#8a8f9c"), 0.25 + 0.2 * h)
            R.SUB[nm] = nm.split(" > ")[1]

    print(f"Hierarchical HTML  K1={K1} K2={K2} leaves={LEAVES}, {len(leaf_rows)} leaves, "
          f"train acc {clf.score(X, df['cluster_final']):.3f}")
    for nd in R._leaves(tree_dict):
        i = nd["info"]
        print(f"  {i['label']:<34} N={i['n']:<5} pure {i['purity']*100:.0f}%")

    _render(data)


def _shade(hex_color, amt):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amt); g = int(g + (255 - g) * amt); b = int(b + (255 - b) * amt)
    return f"#{r:02x}{g:02x}{b:02x}"


def _render(data):
    labels = [nd["info"]["label"] for nd in R._leaves(data["tree"])]
    cbar = "\n".join(
        f' <span class="cchip"><span class="dot" style="background:{R.COLOR[l]}"></span>{l}</span>'
        for l in R.COLOR if l in labels)
    html = R._TEMPLATE
    html = html.replace("/*__CBAR__*/", cbar)
    html = html.replace("/*__DATA__*/", json.dumps(data))
    html = html.replace("/*__FEAT__*/", json.dumps(R.FEAT))
    html = html.replace("/*__SUB__*/", json.dumps(R.SUB))
    html = html.replace("/*__COLOR__*/", json.dumps(R.COLOR))
    html = html.replace("/*__PAIRS__*/", json.dumps(R.PAIRS))
    out = os.path.join(OUT_DIR, "hierarchical_tree_magnitude.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()