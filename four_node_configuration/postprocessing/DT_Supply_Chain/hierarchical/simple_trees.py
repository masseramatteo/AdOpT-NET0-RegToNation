"""
Small, simple, narrative decision trees (few leaves) for two typologies:
  (a) k=4 flat clusters (the original 4 archetypes)
  (b) scheme H typology (decentralisation x Minimal/Satellite/Meshed)

For each we sweep tiny leaf budgets and save a compact tree (accuracy in title).
Goal: a figure a reader grasps at a glance, still reasonably accurate.

Outputs: DT_supplychain/hierarchical/simple/

Run:  python simple_trees.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from explore_level_schemes import decentr_bins, struct_bins3, name_struct3, name_decentr

OUT = os.path.join(C.BASE_FIG_DIR, "hierarchical", "simple")
LEAF_GRID = [3, 4, 5, 6, 8]


def _z(df, cols):
    return StandardScaler().fit_transform(df[cols].fillna(0.0))


def _tree(leaves):
    return DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=20,
                                  min_samples_split=20, class_weight="balanced",
                                  random_state=C.RANDOM_STATE)


def labels_k4(df):
    y = KMeans(4, random_state=C.RANDOM_STATE, n_init=10).fit_predict(_z(df, C.OUTPUTS_OF_INTEREST))
    df = df.assign(grp=y)
    names = {g: C.name_cluster(df.loc[df.grp == g, C.OUTPUTS_OF_INTEREST].mean()) for g in range(4)}
    return df, [names[g] for g in range(4)]


def labels_H(df):
    dec = decentr_bins(df)
    st = struct_bins3(df)
    dname = {0: "No local", 1: "Low local", 2: "High local"}
    combo, names, code = [], {}, {}
    c = 0
    key = list(zip(dec, st))
    for d, s in sorted(set(key)):
        m = (dec == d) & (st == s)
        code[(d, s)] = c
        names[c] = f"{dname[d]} > {name_struct3(df.loc[m, C.OUTPUTS_OF_INTEREST].mean())}"
        c += 1
    grp = np.array([code[(d, s)] for d, s in key])
    return df.assign(grp=grp), [names[i] for i in range(len(names))]


def make(df, name, labeller):
    dfx, class_names = labeller(df)
    inputs = [c for c in C.DRIVER_INPUTS if c in dfx.columns]
    X = dfx[inputs].fillna(0.0)
    y = dfx["grp"]
    print(f"\n{name}: {len(class_names)} classes -> {class_names}")
    print("  leaves : CV acc")
    for lv in LEAF_GRID:
        acc = cross_val_score(_tree(lv), X, y, cv=5).mean()
        print(f"   {lv:>4}  : {acc:.3f}")
    # save a compact tree (6 leaves, or #classes+2 if small)
    lv = min(6, max(4, len(class_names)))
    clf = _tree(lv).fit(X, y)
    acc = cross_val_score(_tree(lv), X, y, cv=5).mean()
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(clf, feature_names=inputs, class_names=class_names, filled=True, rounded=True,
              proportion=True, impurity=False, precision=1, fontsize=10, ax=ax)
    ax.set_title(f"{name} — simple tree ({lv} leaves, CV {acc:.2f})", fontsize=15)
    fig.savefig(os.path.join(OUT, f"simple_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return name, lv, acc, len(class_names)


def main():
    C.apply_joule_style()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(C.FEATURES_CSV)
    res = [make(df, "k4_flat", labels_k4), make(df, "H_typology", labels_H)]
    print("\nSaved compact trees:")
    for n, lv, acc, k in res:
        print(f"  simple_{n}.png  ({k} classes, {lv} leaves, CV {acc:.2f})")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()