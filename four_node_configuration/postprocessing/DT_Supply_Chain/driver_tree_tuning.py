"""
Grid-search the DRIVER decision-tree hyperparameters to raise accuracy while
keeping the tree readable (few leaves).

For k in {4,6,8}, clusters are built on the revised outputs of interest; then a
single DecisionTreeClassifier (class_weight='balanced') predicts the cluster from
the sampled inputs. We sweep:
    max_depth          in {3,4,5,6}
    min_samples_leaf   in {10,20,40,60}
    min_samples_split  in {20,40}
and report 5-fold CV accuracy + balanced accuracy + the actual leaf count (a
readability proxy). "Readable" = <= READABLE_LEAVES leaves.

Run:  python driver_tree_tuning.py
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

import config as C

KS = [4, 6, 8]
DEPTHS = [3, 4, 5, 6]
LEAVES = [10, 20, 40, 60]
SPLITS = [20, 40]
READABLE_LEAVES = 16       # trees with <= this many leaves stay legible


def main():
    df = pd.read_csv(C.FEATURES_CSV)
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    Xd = df[inputs].fillna(0.0)
    Xooi = StandardScaler().fit_transform(df[C.OUTPUTS_OF_INTEREST].fillna(0.0))

    for k in KS:
        y = KMeans(n_clusters=k, random_state=C.RANDOM_STATE, n_init=10).fit_predict(Xooi)
        base = pd.Series(y).value_counts(normalize=True).max()
        rows = []
        for d, leaf, split in itertools.product(DEPTHS, LEAVES, SPLITS):
            clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=leaf,
                                         min_samples_split=split,
                                         class_weight="balanced", random_state=0)
            acc = cross_val_score(clf, Xd, y, cv=5, scoring="accuracy").mean()
            bal = cross_val_score(clf, Xd, y, cv=5, scoring="balanced_accuracy").mean()
            n_leaves = clf.fit(Xd, y).get_n_leaves()
            rows.append((d, leaf, split, n_leaves, acc, bal))
        res = pd.DataFrame(rows, columns=["depth", "min_leaf", "min_split",
                                          "leaves", "cv_acc", "cv_bal_acc"])

        print("\n" + "=" * 78)
        print(f"k={k}   baseline acc={base:.3f}   (readable = <= {READABLE_LEAVES} leaves)")
        print("=" * 78)
        readable = res[res["leaves"] <= READABLE_LEAVES].sort_values("cv_acc", ascending=False)
        print("Top 6 READABLE configs by CV accuracy:")
        print(readable.head(6).to_string(index=False,
              formatters={"cv_acc": lambda v: f"{v:.3f}", "cv_bal_acc": lambda v: f"{v:.3f}"}))
        best_any = res.sort_values("cv_acc", ascending=False).iloc[0]
        print(f"Best overall (any size): depth={int(best_any.depth)} "
              f"min_leaf={int(best_any.min_leaf)} min_split={int(best_any.min_split)} "
              f"leaves={int(best_any.leaves)} cv_acc={best_any.cv_acc:.3f}")


if __name__ == "__main__":
    main()