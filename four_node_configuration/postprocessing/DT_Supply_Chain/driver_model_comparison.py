"""
Driver-model comparison: how high can input -> infrastructure-type accuracy go?

For k in {4,6,8}, clusters are built on the revised outputs of interest, then several
models predict the cluster from the sampled inputs. Reports 5-fold CV:
  - accuracy          (raw; inflated by class imbalance)
  - balanced accuracy (mean per-class recall; fair under imbalance)
  - macro F1
plus the RandomForest per-feature importances.

Models:
  DT_readable  single tree used in the pipeline (depth 4, <=10 leaves)  -> the figure
  DT_full      single tree, unconstrained depth (min_leaf 40)
  RF           RandomForest with the user's params
  RF_big       RandomForest, more/deeper trees
  HGB          Histogram Gradient Boosting (usually the strongest)

Run:  python driver_model_comparison.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

import config as C

KS = [4, 6, 8]
SCORERS = {"acc": "accuracy", "bal_acc": "balanced_accuracy", "macroF1": "f1_macro"}


def models():
    return {
        "DT_readable": DecisionTreeClassifier(
            max_depth=4, max_leaf_nodes=10, min_samples_leaf=40,
            min_samples_split=30, class_weight="balanced", random_state=0),
        "DT_full": DecisionTreeClassifier(
            max_depth=None, min_samples_leaf=40, class_weight="balanced", random_state=0),
        "RF": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=3, min_samples_split=6,
            random_state=0, class_weight="balanced", n_jobs=-1),
        "RF_big": RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            random_state=0, class_weight="balanced_subsample", n_jobs=-1),
        "HGB": HistGradientBoostingClassifier(
            max_depth=None, learning_rate=0.1, max_iter=400, random_state=0),
    }


def main():
    df = pd.read_csv(C.FEATURES_CSV)
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    Xd = df[inputs].fillna(0.0)
    Xooi = StandardScaler().fit_transform(df[C.OUTPUTS_OF_INTEREST].fillna(0.0))

    for k in KS:
        y = KMeans(n_clusters=k, random_state=C.RANDOM_STATE, n_init=10).fit_predict(Xooi)
        base = pd.Series(y).value_counts(normalize=True).max()
        print("\n" + "=" * 74)
        print(f"k={k}   (majority-class baseline accuracy = {base:.3f})")
        print("=" * 74)
        print(f"{'model':<12}{'acc':>8}{'bal_acc':>9}{'macroF1':>9}")
        for name, mdl in models().items():
            row = {s: cross_val_score(mdl, Xd, y, cv=5, scoring=sc).mean()
                   for s, sc in SCORERS.items()}
            print(f"{name:<12}{row['acc']:>8.3f}{row['bal_acc']:>9.3f}{row['macroF1']:>9.3f}")

        # RF feature importances (fit on full data)
        rf = models()["RF"].fit(Xd, y)
        imp = sorted(zip(inputs, rf.feature_importances_), key=lambda t: -t[1])
        print("  RF importances:", ", ".join(f"{f}={v:.2f}" for f, v in imp if v > 0.03))


if __name__ == "__main__":
    main()