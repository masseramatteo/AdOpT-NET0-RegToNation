"""
Single decision-tree view of the hierarchical typology.

Instead of two stage panels, this trains ONE DecisionTreeClassifier on the combined
hierarchical groups (supply origin x transport structure) and plots it as a single
tree — the same look as the earlier driver_tree.png. Class names are the nested
"Origin > Structure" labels. Readability controlled by a leaf budget.

Configure via env:  STAGE_K1 (2), STAGE_K2 (3), TREE_LEAVES (16)

Run:  python single_hierarchical_tree.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from hierarchical_clustering import LEVEL1, LEVEL2, name_l1, name_l2

K1 = int(os.environ.get("STAGE_K1", 2))
K2 = int(os.environ.get("STAGE_K2", 3))
LEAVES = int(os.environ.get("TREE_LEAVES", 16))
OUT_DIR = os.path.join(C.BASE_FIG_DIR, "hierarchical", "single")


def _z(df, cols):
    return StandardScaler().fit_transform(df[cols].fillna(0.0))


def main():
    C.apply_joule_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(C.FEATURES_CSV)
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    X = df[inputs].fillna(0.0)

    # hierarchical ground-truth -> combined 0..K1*K2-1 label + nested names
    df["L1"] = KMeans(K1, random_state=C.RANDOM_STATE, n_init=10).fit_predict(_z(df, LEVEL1))
    l1_names = {g: name_l1(df.loc[df["L1"] == g, LEVEL1].mean()) for g in range(K1)}
    df["L2"] = -1
    for g in range(K1):
        m = df["L1"] == g
        df.loc[m, "L2"] = KMeans(K2, random_state=C.RANDOM_STATE, n_init=10).fit_predict(_z(df[m], LEVEL2))
    combined, names = {}, {}
    code = {}
    c = 0
    for g in range(K1):
        for h in range(K2):
            code[(g, h)] = c
            mm = (df["L1"] == g) & (df["L2"] == h)
            names[c] = f"{l1_names[g]} > {name_l2(df.loc[mm, LEVEL2].mean())}"
            c += 1
    y = df.apply(lambda r: code[(r["L1"], r["L2"])], axis=1).astype(int)
    class_names = [names[i] for i in range(K1 * K2)]

    clf = DecisionTreeClassifier(max_leaf_nodes=LEAVES, min_samples_leaf=20,
                                 min_samples_split=20, class_weight="balanced",
                                 random_state=C.RANDOM_STATE)
    clf.fit(X, y)
    cv = cross_val_score(clf, X, y, cv=5)
    print(f"Single hierarchical tree  K1={K1} K2={K2}  {K1*K2} groups, {LEAVES} leaves")
    print("Classes:", class_names)
    print(f"Train acc {accuracy_score(y, clf.predict(X)):.3f}  CV {cv.mean():.3f} +/- {cv.std():.3f}")
    print("\n" + export_text(clf, feature_names=inputs, class_names=class_names))

    fig, ax = plt.subplots(figsize=(26, 13))
    plot_tree(clf, feature_names=inputs, class_names=class_names, filled=True,
              rounded=True, proportion=True, impurity=False, precision=1, fontsize=9, ax=ax)
    ax.set_title(f"Hierarchical typology as one tree — inputs → Origin > Structure "
                 f"({K1*K2} groups, {LEAVES} leaves, CV {cv.mean():.2f})", fontsize=15)
    fig.savefig(os.path.join(OUT_DIR, "single_hierarchical_tree.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {OUT_DIR}\\single_hierarchical_tree.png")


if __name__ == "__main__":
    main()