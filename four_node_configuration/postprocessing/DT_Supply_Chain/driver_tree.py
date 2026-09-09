"""
Step 4 - Driver tree (scenario discovery).

Trains a DecisionTreeClassifier that predicts the infrastructure cluster
(cluster_final, from explain_tree.py) FROM the sampled input parameters. This is
the "which sampled conditions lead to which infrastructure type" analysis
(Wiest et al. 2026). class_weight='balanced', depth 3-4.

Run:  python driver_tree.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

import config as C


def run():
    C.apply_joule_style()
    os.makedirs(C.FIG_DIR, exist_ok=True)

    df = pd.read_csv(C.CLUSTER_LABELS_CSV)
    if "cluster_final" not in df.columns:
        raise RuntimeError("cluster_final missing — run explain_tree.py first.")

    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    X = df[inputs].fillna(0.0)
    y = df["cluster_final"]
    k = y.nunique()
    print(f"Driver tree: {len(inputs)} input params -> {k} infrastructure clusters")
    print(f"Inputs: {inputs}")

    # robustness params mirror decision_tree_local_production.py (stable leaves)
    clf = DecisionTreeClassifier(max_depth=C.DRIVER_MAX_DEPTH,
                                 max_leaf_nodes=C.DRIVER_MAX_LEAF_NODES,
                                 min_samples_leaf=C.DRIVER_MIN_SAMPLES_LEAF,
                                 min_samples_split=C.DRIVER_MIN_SAMPLES_SPLIT,
                                 class_weight="balanced",
                                 random_state=C.RANDOM_STATE)
    clf.fit(X, y)
    pred = clf.predict(X)

    # Group folds by economy when the campaign has one (v5 paired design:
    # 8 runs share every economic input), otherwise plain 5-fold.
    if C.GROUP_COL in df.columns:
        from sklearn.model_selection import StratifiedGroupKFold
        cv = cross_val_score(clf, X, y, groups=df[C.GROUP_COL],
                             cv=StratifiedGroupKFold(5, shuffle=True, random_state=C.RANDOM_STATE))
        print(f"\n  CV grouped by '{C.GROUP_COL}' ({df[C.GROUP_COL].nunique()} groups)")
    else:
        cv = cross_val_score(clf, X, y, cv=5)
    print(f"\n  Train accuracy: {accuracy_score(y, pred):.4f}")
    print(f"  5-fold CV accuracy: {cv.mean():.4f} +/- {cv.std():.4f}")
    print("\nClassification report (train):")
    print(classification_report(y, pred, zero_division=0))

    print("Feature importances:")
    imp = sorted(zip(inputs, clf.feature_importances_), key=lambda t: -t[1])
    for f, v in imp:
        if v > 0:
            print(f"  {f}: {v:.4f}")

    txt = export_text(clf, feature_names=inputs)
    print("\nDriver tree structure:\n" + txt)

    class_names = [C.CLUSTER_NAMES.get(c, f"C{c}") for c in range(k)]
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(clf, feature_names=inputs, class_names=class_names,
              filled=True, rounded=True, proportion=True, impurity=False,
              precision=1, fontsize=9, ax=ax)
    ax.set_title(f"Driver tree — sampled inputs -> infrastructure type "
                 f"(depth {C.DRIVER_MAX_DEPTH}, min_leaf {C.DRIVER_MIN_SAMPLES_LEAF})",
                 fontsize=15)
    fig.savefig(os.path.join(C.FIG_DIR, "driver_tree.png"), dpi=300)
    plt.close(fig)

    # feature-importance bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    names = [f for f, v in imp if v > 0]
    vals = [v for f, v in imp if v > 0]
    ax.barh(range(len(names)), vals, color=C.UU_TEAL)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("importance")
    ax.set_title("Driver tree — feature importances")
    fig.savefig(os.path.join(C.FIG_DIR, "driver_importances.png"))
    plt.close(fig)

    print(f"\nFigures saved to {C.FIG_DIR}")
    return clf


if __name__ == "__main__":
    run()