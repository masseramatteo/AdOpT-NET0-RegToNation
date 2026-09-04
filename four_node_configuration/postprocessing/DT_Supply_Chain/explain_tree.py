"""
Step 3 - Interpretable decision tree over the clusters (Baader steps 2 & 3).

Trains a DecisionTreeClassifier on the (raw) outputs of interest to reproduce the
k-Means labels, capping leaves at k so each cluster becomes a set of threshold
rules. Then re-orders: the few misclassified points are re-assigned to the tree's
leaf label (Baader step 3), so clusters == tree paths.

Outputs:
  - text + graphical tree
  - radar plot per final cluster (leaf), Baader/Wiest style
  - cluster_labels.csv updated with 'cluster_final' (post-reorder label)

Run:  python explain_tree.py            # uses whatever k is in cluster_labels.csv
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score

import config as C


def run():
    C.apply_joule_style()
    os.makedirs(C.FIG_DIR, exist_ok=True)

    df = pd.read_csv(C.CLUSTER_LABELS_CSV)
    k = df["cluster_kmeans"].nunique()
    max_leaves = max(C.EXPLAIN_MAX_LEAF_NODES, k)
    X = df[C.OUTPUTS_OF_INTEREST].fillna(0.0)
    y = df["cluster_kmeans"]

    print(f"Explaining k={k} clusters with a decision tree (max_leaf_nodes={max_leaves})")

    # --- Step 2: tree, leaves capped (>= k) ---------------------------------
    tree = DecisionTreeClassifier(max_leaf_nodes=max_leaves, random_state=C.RANDOM_STATE)
    tree.fit(X, y)
    pred = tree.predict(X)
    acc = accuracy_score(y, pred)
    n_reassigned = int((pred != y).sum())
    print(f"  Train accuracy: {acc:.4f}")
    print(f"  Points re-assigned by tree (Step 3 re-ordering): {n_reassigned} "
          f"({n_reassigned/len(df)*100:.2f}%)")

    # --- Step 3: re-order — cluster == tree leaf ----------------------------
    df["cluster_final"] = pred

    # check for repeated splits on the same feature (Baader's k-reduction criterion)
    feats_used = [C.OUTPUTS_OF_INTEREST[i] for i in tree.tree_.feature if i >= 0]
    from collections import Counter
    split_counts = Counter(feats_used)
    print(f"  Features used in splits: {dict(split_counts)}")
    repeated = {f: n for f, n in split_counts.items() if n > 1}
    if repeated:
        print(f"  NOTE: tree splits repeatedly on {repeated} -> consider reducing k")
    else:
        print("  No feature is split on more than once — k looks well chosen.")

    # --- text + graphical tree ----------------------------------------------
    txt = export_text(tree, feature_names=list(C.OUTPUTS_OF_INTEREST))
    print("\nDecision tree structure:\n" + txt)

    class_names = [C.CLUSTER_NAMES.get(c, f"C{c}") for c in range(k)]
    fig, ax = plt.subplots(figsize=(22, 12))
    plot_tree(tree, feature_names=list(C.OUTPUTS_OF_INTEREST),
              class_names=class_names, filled=True, rounded=True,
              proportion=True, impurity=False, precision=2, fontsize=11, ax=ax)
    ax.set_title(f"Infrastructure typology — explaining tree "
                 f"({max_leaves} leaves, train acc {acc:.0%})", fontsize=15)
    fig.savefig(os.path.join(C.FIG_DIR, "explain_tree.png"), dpi=300)
    plt.close(fig)

    # --- per-cluster profiles (final labels) --------------------------------
    prof = df.groupby("cluster_final")[C.OUTPUTS_OF_INTEREST].mean()
    print("\nFinal cluster profiles (mean of each output of interest):")
    with pd.option_context("display.float_format", lambda v: f"{v:7.3f}",
                           "display.width", 200):
        print(prof.T)

    print("\nAuto-generated archetype names:")
    for c in sorted(prof.index):
        size = (df["cluster_final"] == c).sum()
        print(f"  C{c} (n={size}): {C.name_cluster(prof.loc[c])}")

    _radar_grid(df, prof)

    df.to_csv(C.CLUSTER_LABELS_CSV, index=False)
    print(f"\nUpdated labels (with cluster_final) -> {C.CLUSTER_LABELS_CSV}")
    return df, tree


def _radar_grid(df, prof):
    """One radar per final cluster; axes = outputs of interest, min-max normalized
    across all clusters so shapes are comparable (Baader/Wiest style)."""
    oois = C.OUTPUTS_OF_INTEREST
    # normalize each axis to [0,1] over the global data range for comparability
    gmin = df[oois].min()
    gmax = df[oois].max()
    norm = (prof[oois] - gmin) / (gmax - gmin).replace(0, 1)

    clusters = sorted(prof.index)
    n = len(clusters)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    angles = np.linspace(0, 2 * np.pi, len(oois), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow),
                             subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes).ravel()

    short = [o.replace("_", "\n") for o in oois]
    for idx, c in enumerate(clusters):
        ax = axes[idx]
        vals = norm.loc[c].tolist()
        vals += vals[:1]
        color = C.CLUSTER_PALETTE[idx % len(C.CLUSTER_PALETTE)]
        ax.plot(angles, vals, color=color, linewidth=2)
        ax.fill(angles, vals, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short, fontsize=6)
        ax.set_yticklabels([])
        ax.set_ylim(0, 1)
        size = (df["cluster_final"] == c).sum()
        name = C.name_cluster(prof.loc[c])
        ax.set_title(f"{name}\n(C{c}, n={size})", fontsize=9, color=color, pad=12)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Infrastructure archetypes — outputs-of-interest profiles "
                 "(min-max normalized)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "radar_clusters.png"))
    plt.close(fig)


if __name__ == "__main__":
    run()