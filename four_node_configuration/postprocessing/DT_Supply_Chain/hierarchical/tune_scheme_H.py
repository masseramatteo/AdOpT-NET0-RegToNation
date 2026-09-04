"""
Tune the driver decision-tree for SCHEME H (decentralisation -> Minimal/Satellite/
Meshed). Sweep leaf budget / min_samples_leaf / depth for both the single tree and
the staged trees, and report the RandomForest ceiling. Find the best accuracy that
stays readable.

Run:  python tune_scheme_H.py
"""

import os
import sys
import itertools

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from explore_level_schemes import SCHEMES, build_labels

SID = "H_decentr_then_struct3"


def _tree(leaves=None, depth=None, min_leaf=20, min_split=20):
    return DecisionTreeClassifier(max_leaf_nodes=leaves, max_depth=depth,
                                  min_samples_leaf=min_leaf, min_samples_split=min_split,
                                  class_weight="balanced", random_state=C.RANDOM_STATE)


def staged_cv(X, L1, L2, k1, leaves, min_leaf):
    combined = L1 * 10 + L2
    skf = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    e2e = []
    for tr, te in skf.split(X, combined):
        t1 = _tree(leaves=leaves, min_leaf=min_leaf).fit(X.iloc[tr], L1[tr])
        l1p = t1.predict(X.iloc[te])
        l2p = np.zeros(len(te), dtype=int)
        for g in range(k1):
            mg = L1[tr] == g
            if len(np.unique(L2[tr][mg])) > 1:
                tg = _tree(leaves=leaves, min_leaf=min_leaf).fit(X.iloc[tr][mg], L2[tr][mg])
                sel = l1p == g
                if sel.any():
                    l2p[sel] = tg.predict(X.iloc[te][sel])
        e2e.append(((l1p == L1[te]) & (l2p == L2[te])).mean())
    return np.mean(e2e)


def main():
    df = pd.read_csv(C.FEATURES_CSV)
    df, code, names, lv1_names, (k1, k2) = build_labels(df, SCHEMES[SID])
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    X = df[inputs].fillna(0.0)
    y = df["grp"]
    base = y.value_counts(normalize=True).max()
    print(f"SCHEME {SID}: {len(names)} groups, baseline {base:.3f}\n")

    # ---- single tree: leaf budget x min_leaf --------------------------------
    print("SINGLE TREE  (max_leaf_nodes x min_leaf) -> CV acc [n_leaves]")
    print(f"{'leaves':>7}" + "".join(f"{'ml='+str(m):>12}" for m in [10, 20, 40]))
    for ml_nodes in [8, 10, 12, 14, 16, 20, 25, 30, 40]:
        row = f"{ml_nodes:>7}"
        for mleaf in [10, 20, 40]:
            clf = _tree(leaves=ml_nodes, min_leaf=mleaf)
            acc = cross_val_score(clf, X, y, cv=5).mean()
            nlv = clf.fit(X, y).get_n_leaves()
            row += f"{acc:>7.3f}[{nlv:>2}]"
        print(row)

    # ---- staged: leaf budget x min_leaf -------------------------------------
    print("\nSTAGED (per-stage max_leaf_nodes x min_leaf) -> end-to-end CV acc")
    print(f"{'leaves':>7}" + "".join(f"{'ml='+str(m):>10}" for m in [10, 20, 40]))
    L1, L2 = df["Lv1"].values, df["Lv2"].values
    for ml_nodes in [6, 8, 10, 12, 16, 20]:
        row = f"{ml_nodes:>7}"
        for mleaf in [10, 20, 40]:
            row += f"{staged_cv(X, L1, L2, k1, ml_nodes, mleaf):>10.3f}"
        print(row)

    # ---- ceiling ------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                class_weight="balanced_subsample", random_state=0, n_jobs=-1)
    print(f"\nRandomForest ceiling (single flat): {cross_val_score(rf, X, y, cv=5).mean():.3f}")


if __name__ == "__main__":
    main()