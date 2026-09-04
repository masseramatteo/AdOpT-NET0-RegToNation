"""
Accuracy sweep for the driver tree.

Explores how well the sampled inputs predict the infrastructure type as a function
of:
  - number of clusters k (fewer clusters = coarser, easier target)
  - decision-tree depth
  - model (DecisionTree vs RandomForest = predictability ceiling)

Reports 5-fold CV accuracy against the majority-class baseline, so we can see where
real signal lives. Nothing is saved; this is a diagnostic.

Run:  python experiments.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import config as C

DEPTHS = [3, 4, 5, 6, 8, None]
KS = [2, 3, 4, 5, 6]


def main():
    df = pd.read_csv(C.FEATURES_CSV)
    Xd = df[[c for c in C.DRIVER_INPUTS if c in df.columns]].fillna(0.0)
    Xooi = StandardScaler().fit_transform(df[C.OUTPUTS_OF_INTEREST].fillna(0.0))

    print(f"{'k':>2} {'baseline':>9} " + " ".join(f"d{str(d):>5}" for d in DEPTHS) + f" {'RF':>6}")
    print("-" * 70)
    for k in KS:
        km = KMeans(n_clusters=k, random_state=C.RANDOM_STATE, n_init=10)
        y = km.fit_predict(Xooi)
        baseline = pd.Series(y).value_counts(normalize=True).max()

        row = []
        for d in DEPTHS:
            clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=C.DRIVER_MIN_SAMPLES_LEAF,
                                         min_samples_split=C.DRIVER_MIN_SAMPLES_SPLIT,
                                         class_weight="balanced", random_state=C.RANDOM_STATE)
            acc = cross_val_score(clf, Xd, y, cv=5).mean()
            row.append(acc)
        rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=3,
                                    class_weight="balanced", random_state=C.RANDOM_STATE, n_jobs=-1)
        rf_acc = cross_val_score(rf, Xd, y, cv=5).mean()

        print(f"{k:>2} {baseline:>9.3f} " + " ".join(f"{a:>6.3f}" for a in row) + f" {rf_acc:>6.3f}")

    print("\nbaseline = majority-class share (accuracy of always predicting the biggest cluster)")
    print("RF = RandomForest (depth unlimited) — the predictability ceiling from these inputs")


if __name__ == "__main__":
    main()