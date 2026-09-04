"""
ONE single tree for scheme H (decentralisation x Minimal/Satellite/Meshed) on ROBUST
runs, rendered as the schematic-leaf HTML (same style as driver_tree_magnitude.html):
splits test sampled inputs, each leaf drawn as a 4-node network aggregated over its
runs, labelled by the majority "decentralisation > structure" group.

Robust = small-cluster reopt infeasible OR delta_npv_pct > 1% (1 MW install threshold).

Outputs: DT_supplychain/robust/H/robust_H_tree_magnitude.html

Run:  python robust_H_html.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import render_driver_tree_html as R
from explore_level_schemes import decentr_bins, struct_bins3, name_decentr, name_struct3
from robust_trees import robust_run_ids, ROBUST_SCENARIO

OUT = os.path.join(C.BASE_FIG_DIR, "robust", "H")
LEAVES = 12
DEC_BASE = {0: C.UU_NAVY, 1: C.UU_TEAL, 2: C.UU_YELLOW}   # None / Low / High


def _shade(hex_color, amt):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amt); g = int(g + (255 - g) * amt); b = int(b + (255 - b) * amt)
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    os.makedirs(OUT, exist_ok=True)
    feats = pd.read_csv(C.FEATURES_CSV)
    raw = pd.read_excel(C.EXTRACTED_RESULTS)
    rob_ids, *_ = robust_run_ids(ROBUST_SCENARIO)
    feats = feats[feats["run"].isin(rob_ids)].copy()
    df = feats.merge(raw, on="run", how="left", suffixes=("", "_raw"))
    print(f"Robust H HTML: {len(df)} robust runs")

    # scheme H labels: decentralisation (self-suff bins) x structure (Min/Sat/Meshed)
    dec = decentr_bins(df); st = struct_bins3(df)
    code, names, dec_of = {}, {}, {}
    c = 0
    for d in sorted(set(dec)):
        for s in sorted(set(st[dec == d])):
            m = (dec == d) & (st == s)
            code[(d, s)] = c
            names[c] = f"{name_decentr(df.loc[m, C.OUTPUTS_OF_INTEREST].mean())} > " \
                       f"{name_struct3(df.loc[m, C.OUTPUTS_OF_INTEREST].mean())}"
            dec_of[c] = int(d)
            c += 1
    df["cluster_final"] = [code[(d, s)] for d, s in zip(dec, st)]

    inputs = [col for col in C.DRIVER_INPUTS if col in df.columns]
    X = df[inputs].fillna(0.0)
    clf = DecisionTreeClassifier(max_leaf_nodes=LEAVES, min_samples_leaf=20,
                                 min_samples_split=20, class_weight="balanced",
                                 random_state=C.RANDOM_STATE).fit(X, df["cluster_final"])
    print(f"  single tree {LEAVES} leaves, train acc {clf.score(X, df['cluster_final']):.3f}")
    df["_leaf"] = clf.apply(X)
    leaf_rows = {lid: df[df["_leaf"] == lid] for lid in np.unique(df["_leaf"])}
    name_by_cluster = {i: names[i] for i in names}
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

    # colours: base by decentralisation level, shaded by structure
    R.COLOR, R.SUB = {}, {}
    for cid, nm in names.items():
        R.COLOR[nm] = _shade(DEC_BASE.get(dec_of[cid], "#8a8f9c"), 0.15 * (cid % 3))
        R.SUB[nm] = nm.split(" > ")[1]

    labels = [nd["info"]["label"] for nd in R._leaves(tree_dict)]
    cbar = "\n".join(
        f' <span class="cchip"><span class="dot" style="background:{R.COLOR[l]}"></span>{l}</span>'
        for l in dict.fromkeys(labels))
    html = R._TEMPLATE
    html = html.replace("/*__CBAR__*/", cbar)
    html = html.replace("/*__DATA__*/", json.dumps(data))
    html = html.replace("/*__FEAT__*/", json.dumps(R.FEAT))
    html = html.replace("/*__SUB__*/", json.dumps(R.SUB))
    html = html.replace("/*__COLOR__*/", json.dumps(R.COLOR))
    html = html.replace("/*__PAIRS__*/", json.dumps(R.PAIRS))
    out = os.path.join(OUT, "robust_H_tree_magnitude.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    for nd in R._leaves(tree_dict):
        i = nd["info"]
        print(f"  {i['label']:<28} N={i['n']:<4} pure {i['purity']*100:.0f}%")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()