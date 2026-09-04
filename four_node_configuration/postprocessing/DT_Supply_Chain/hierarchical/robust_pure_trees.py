"""
Robust + pure simple trees: the easy, low-gini trees, on ROBUST runs only, with a
90/10 train/test split. For the decisions that were low-accuracy on the full set but
have potential, robustness filtering removes the degenerate ties and sharpens them.

Targets: has_local (2cl), decentralisation (3cl), k4 archetypes (4cl).
For each: FULL vs ROBUST train/test accuracy, and the high-purity (low-gini) rules.

Outputs: DT_supplychain/robust/pure/

Run:  python robust_pure_trees.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from explore_level_schemes import decentr_bins
from pure_leaves import leaf_rules
from robust_trees import robust_run_ids, ROBUST_SCENARIO

OUT = os.path.join(C.BASE_FIG_DIR, "robust", "pure")
PURITY = 0.70


def _tree(leaves):
    return DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=20,
                                  min_samples_split=20, class_weight="balanced",
                                  random_state=C.RANDOM_STATE)


def fit_report(df, name, y, class_names, leaves, tag):
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    X = df[inputs].fillna(0).reset_index(drop=True)
    y = np.asarray(y)
    tr, te = train_test_split(np.arange(len(df)), test_size=0.1, random_state=42, stratify=y)
    clf = _tree(leaves).fit(X.iloc[tr], y[tr])
    a_tr, a_te = clf.score(X.iloc[tr], y[tr]), clf.score(X.iloc[te], y[te])
    rules = leaf_rules(clf, inputs, class_names)
    pure = [r for r in rules if r["purity"] >= PURITY]
    cov = sum(r["n"] for r in pure) / len(df)

    if tag == "ROBUST":
        fig, ax = plt.subplots(figsize=(15, 8))
        plot_tree(clf, feature_names=inputs, class_names=class_names, filled=True, rounded=True,
                  proportion=True, impurity=True, precision=1, fontsize=10, ax=ax)
        ax.set_title(f"ROBUST {name} — {leaves} leaves, train {a_tr:.2f}/test {a_te:.2f}, "
                     f"{len(pure)} pure leaves cover {cov*100:.0f}%", fontsize=14)
        fig.savefig(os.path.join(OUT, f"robust_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        with open(os.path.join(OUT, f"robust_{name}_rules.txt"), "w", encoding="utf-8") as f:
            f.write(f"ROBUST {name}: train {a_tr:.3f} test {a_te:.3f} n={len(df)}\n")
            for r in sorted(pure, key=lambda r: -r["purity"]):
                f.write(f"  [{r['purity']:.0%}, n={r['n']}] {r['class']}  IF {r['rule']}\n")
    return a_tr, a_te, len(pure), len(rules), cov


def targets(df):
    dec = decentr_bins(df)
    k4 = KMeans(4, random_state=C.RANDOM_STATE, n_init=10).fit_predict(
        StandardScaler().fit_transform(df[C.OUTPUTS_OF_INTEREST].fillna(0)))
    k4n = [C.name_cluster(df.loc[k4 == g, C.OUTPUTS_OF_INTEREST].mean()) for g in range(4)]
    return [
        ("has_local", (dec > 0).astype(int), ["No local", "Local"], 3),
        ("decentralisation", dec, ["No local", "Low local", "High local"], 5),
        ("k4_archetypes", k4, k4n, 6),
    ]


def main():
    C.apply_joule_style()
    os.makedirs(OUT, exist_ok=True)
    feat = pd.read_csv(C.FEATURES_CSV)
    rob_ids, *_ = robust_run_ids(ROBUST_SCENARIO)
    robust = feat[feat["run"].isin(rob_ids)].copy()
    print(f"full {len(feat)}  robust {len(robust)}\n")

    print(f"{'target':<18}{'FULL tr/te':>16}{'ROBUST tr/te':>16}{'robust pure/cov':>18}")
    for name, y, cn, lv in targets(feat):
        ftr, fte, *_ = fit_report(feat, name, y, cn, lv, "FULL")
        # robust: recompute target on robust subset
        for rname, ry, rcn, rlv in targets(robust):
            if rname == name:
                rtr, rte, npure, ntot, cov = fit_report(robust, name, ry, rcn, rlv, "ROBUST")
                break
        print(f"{name:<18}{ftr:.2f}/{fte:.2f}{'':>6}{rtr:.2f}/{rte:.2f}{'':>6}"
              f"{npure}/{ntot}, {cov*100:.0f}%")
    print(f"\nSaved robust pure trees -> {OUT}")


if __name__ == "__main__":
    main()