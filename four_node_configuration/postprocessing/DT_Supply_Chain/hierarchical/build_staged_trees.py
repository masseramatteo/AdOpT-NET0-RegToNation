"""
Build and save the STAGED driver trees:
  - Level 1 tree: sampled inputs -> supply origin (K1 classes)
  - Level 2 trees: within each L1 branch, inputs -> transport structure (K2 classes)

Ground-truth labels from the manual level-wise hierarchy. Trees use the standard
DecisionTreeClassifier knobs. Figures saved to DT_supplychain/hierarchical/staged/.

Configure via env vars:  STAGE_K1 (default 2), STAGE_K2 (3), STAGE_DEPTH (5)

Run:  python build_staged_trees.py
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
from hierarchical_clustering import LEVEL1, LEVEL2, name_l1, name_l2

K1 = int(os.environ.get("STAGE_K1", 2))
K2 = int(os.environ.get("STAGE_K2", 3))
DEPTH = int(os.environ.get("STAGE_DEPTH", 5))
MIN_LEAF, MIN_SPLIT = 20, 20
OUT_DIR = os.path.join(C.BASE_FIG_DIR, "hierarchical", "staged")


def _z(df, cols):
    return StandardScaler().fit_transform(df[cols].fillna(0.0))


def _tree():
    return DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=MIN_LEAF,
                                  min_samples_split=MIN_SPLIT, class_weight="balanced",
                                  random_state=C.RANDOM_STATE)


def main():
    C.apply_joule_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(C.FEATURES_CSV)
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    X = df[inputs].fillna(0.0)

    # ground-truth hierarchy
    df["L1"] = KMeans(K1, random_state=C.RANDOM_STATE, n_init=10).fit_predict(_z(df, LEVEL1))
    l1_names = {g: name_l1(df.loc[df["L1"] == g, LEVEL1].mean()) for g in range(K1)}
    df["L2"] = -1
    l2_names = {}
    for g in range(K1):
        m = df["L1"] == g
        df.loc[m, "L2"] = KMeans(K2, random_state=C.RANDOM_STATE, n_init=10).fit_predict(_z(df[m], LEVEL2))
        for h in range(K2):
            mm = m & (df["L2"] == h)
            l2_names[(g, h)] = name_l2(df.loc[mm, LEVEL2].mean())

    print(f"Staged trees  K1={K1} K2={K2} depth={DEPTH}  -> {OUT_DIR}")
    print("Supply-origin (L1) groups:", {g: f"{l1_names[g]} (n={(df['L1']==g).sum()})" for g in range(K1)})

    # ---- Level 1 tree -------------------------------------------------------
    t1 = _tree().fit(X, df["L1"])
    cv1 = cross_val_score(_tree(), X, df["L1"], cv=5)
    print(f"\nLevel-1 (supply origin): train {t1.score(X, df['L1']):.3f}  CV {cv1.mean():.3f} +/- {cv1.std():.3f}")
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(t1, feature_names=inputs, class_names=[l1_names[g] for g in range(K1)],
              filled=True, rounded=True, proportion=True, impurity=False,
              precision=1, fontsize=9, ax=ax)
    ax.set_title(f"Level 1 driver tree — inputs → supply origin "
                 f"(K1={K1}, depth {DEPTH}, CV {cv1.mean():.2f})", fontsize=15)
    fig.savefig(os.path.join(OUT_DIR, "staged_L1_supply_origin.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Level 2 trees (one per branch) ------------------------------------
    fig, axes = plt.subplots(K1, 1, figsize=(20, 9 * K1))
    axes = np.atleast_1d(axes)
    for g in range(K1):
        m = df["L1"] == g
        Xg, yg = X[m], df.loc[m, "L2"]
        tg = _tree().fit(Xg, yg)
        cvg = cross_val_score(_tree(), Xg, yg, cv=5)
        cls = [l2_names[(g, h)] for h in range(K2)]
        print(f"Level-2 within '{l1_names[g]}': train {tg.score(Xg, yg):.3f}  "
              f"CV {cvg.mean():.3f}  (n={m.sum()})  structures={cls}")
        plot_tree(tg, feature_names=inputs, class_names=cls, filled=True, rounded=True,
                  proportion=True, impurity=False, precision=1, fontsize=8, ax=axes[g])
        axes[g].set_title(f"Level 2 — within '{l1_names[g]}' → transport structure "
                          f"(CV {cvg.mean():.2f}, n={m.sum()})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "staged_L2_structure_per_branch.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    df[["run", "L1", "L2"]].to_csv(os.path.join(OUT_DIR, "staged_labels.csv"), index=False)
    print(f"\nSaved trees + staged_labels.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()