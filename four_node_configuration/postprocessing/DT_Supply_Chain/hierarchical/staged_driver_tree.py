"""
Staged (hierarchical) driver tree vs flat driver tree.

Idea: predicting a coarse Level-1 label (supply origin) from the sampled inputs
should be much easier than predicting the full fine typology at once. So we train:
  - a Level-1 driver tree  (inputs -> supply origin, K1 classes)
  - within each L1 branch, a Level-2 driver tree (inputs -> transport structure, K2)
and evaluate END-TO-END path accuracy, comparing to a FLAT driver tree that predicts
the K1*K2 combined typology directly.

Ground-truth clusters come from the manual level-wise hierarchy (k-Means on the
level features). Trees use the standard DecisionTreeClassifier knobs
(max_depth, min_samples_leaf, min_samples_split, class_weight='balanced').

Run:  python staged_driver_tree.py
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from hierarchical_clustering import LEVEL1, LEVEL2

# sweeps
K1S = [2, 3]
K2S = [2, 3]
PARAMS = [   # (max_depth, min_samples_leaf, min_samples_split)
    (3, 40, 30),
    (4, 40, 30),
    (5, 20, 20),
    (6, 20, 20),
]


def _z(df, cols):
    return StandardScaler().fit_transform(df[cols].fillna(0.0))


def _tree(md, ml, ms):
    return DecisionTreeClassifier(max_depth=md, min_samples_leaf=ml,
                                  min_samples_split=ms, class_weight="balanced",
                                  random_state=C.RANDOM_STATE)


def build_hierarchy(df, k1, k2):
    """Ground-truth L1/L2 labels from k-Means on the level features."""
    L1 = KMeans(k1, random_state=C.RANDOM_STATE, n_init=10).fit_predict(_z(df, LEVEL1))
    L2 = np.full(len(df), -1)
    for g in range(k1):
        m = L1 == g
        if m.sum() >= k2:
            L2[m] = KMeans(k2, random_state=C.RANDOM_STATE, n_init=10).fit_predict(_z(df[m], LEVEL2))
        else:
            L2[m] = 0
    return L1, L2


def staged_cv(X, L1, L2, k1, params):
    """End-to-end 5-fold CV: predict L1, then L2 within predicted branch."""
    combined = L1 * 10 + L2
    skf = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    l1_acc, e2e_acc = [], []
    for tr, te in skf.split(X, combined):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        l1_tr, l1_te = L1[tr], L1[te]
        l2_tr, l2_te = L2[tr], L2[te]

        t1 = _tree(*params).fit(Xtr, l1_tr)
        l1p = t1.predict(Xte)
        l1_acc.append((l1p == l1_te).mean())

        # per-branch L2 trees
        branch = {}
        for g in range(k1):
            mg = l1_tr == g
            if len(np.unique(l2_tr[mg])) > 1:
                branch[g] = _tree(*params).fit(Xtr[mg], l2_tr[mg])
            else:
                branch[g] = None
                branch[f"const{g}"] = l2_tr[mg][0] if mg.any() else 0
        l2p = np.zeros(len(Xte), dtype=int)
        for i, g in enumerate(l1p):
            if branch.get(g) is not None:
                l2p[i] = branch[g].predict(Xte.iloc[[i]])[0]
            else:
                l2p[i] = branch.get(f"const{g}", 0)
        e2e_acc.append(((l1p == l1_te) & (l2p == l2_te)).mean())
    return np.mean(l1_acc), np.mean(e2e_acc)


def flat_cv(df, X, k, params):
    """Flat driver tree predicting a k-way k-Means typology on ALL outputs."""
    y = KMeans(k, random_state=C.RANDOM_STATE, n_init=10).fit_predict(_z(df, C.OUTPUTS_OF_INTEREST))
    base = pd.Series(y).value_counts(normalize=True).max()
    acc = cross_val_score(_tree(*params), X, y, cv=5).mean()
    return acc, base


def main():
    df = pd.read_csv(C.FEATURES_CSV)
    X = df[[c for c in C.DRIVER_INPUTS if c in df.columns]].fillna(0.0)

    print(f"{'K1':>3}{'K2':>3}{'tot':>4}  {'params(d,leaf,split)':<22}"
          f"{'L1_acc':>8}{'staged_e2e':>12}{'flat_acc':>10}{'flat_base':>11}")
    print("-" * 86)
    for k1, k2 in itertools.product(K1S, K2S):
        L1, L2 = build_hierarchy(df, k1, k2)
        tot = k1 * k2
        for p in PARAMS:
            l1a, e2e = staged_cv(X, L1, L2, k1, p)
            flat, base = flat_cv(df, X, tot, p)
            tag = f"({p[0]},{p[1]},{p[2]})"
            print(f"{k1:>3}{k2:>3}{tot:>4}  {tag:<22}{l1a:>8.3f}{e2e:>12.3f}"
                  f"{flat:>10.3f}{base:>11.3f}")
    print("\nL1_acc  = accuracy of the coarse supply-origin tree (K1 classes)")
    print("staged_e2e = end-to-end: both L1 and L2 correct")
    print("flat_acc = single tree predicting the K1*K2 typology directly")


if __name__ == "__main__":
    main()