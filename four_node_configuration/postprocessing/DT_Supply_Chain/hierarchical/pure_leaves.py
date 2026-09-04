"""
Simple, narrative trees keeping only HIGH-PURITY leaves (low gini = confident rules).

For each target we fit a small DecisionTree, then report only the leaves whose
majority-class share >= PURITY_THRESHOLD as clean "if ... then archetype" rules
(the confident regions). Low-purity leaves are flagged as mixed/uncertain — the
honest 'we can't call it' zones. Saves the tree figure + a rules.txt per target.

Targets (predict the PREDICTABLE decisions, not the degenerate topology):
  - has_local   : is there local electrolysis?            (2 classes)
  - decentr     : decentralisation level None/Low/High     (3 classes)
  - k4          : the 4 flat archetypes                     (4 classes)

Outputs: DT_supplychain/hierarchical/pure/

Run:  python pure_leaves.py
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
from explore_level_schemes import decentr_bins

OUT = os.path.join(C.BASE_FIG_DIR, "hierarchical", "pure")
PURITY_THRESHOLD = 0.70


def _z(df, cols):
    return StandardScaler().fit_transform(df[cols].fillna(0.0))


def _tree(leaves):
    return DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=20,
                                  min_samples_split=20, class_weight="balanced",
                                  random_state=C.RANDOM_STATE)


def leaf_rules(clf, inputs, class_names):
    """Return per-leaf: rule string, majority class, purity, gini, n."""
    t = clf.tree_
    rules = []

    def recurse(node, conds):
        if t.children_left[node] == t.children_right[node]:   # leaf
            counts = t.value[node][0]
            n = counts.sum()
            p = counts / n
            purity = p.max()
            gini = 1 - (p ** 2).sum()
            maj = class_names[int(np.argmax(counts))]
            rules.append({"rule": " AND ".join(conds) if conds else "(root)",
                          "class": maj, "purity": purity, "gini": gini,
                          "n": int(t.n_node_samples[node])})
            return
        f = inputs[t.feature[node]]
        thr = t.threshold[node]
        recurse(t.children_left[node], conds + [f"{f} <= {thr:.1f}"])
        recurse(t.children_right[node], conds + [f"{f} > {thr:.1f}"])

    recurse(0, [])
    return rules


def run(df, name, y, class_names, leaves):
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    X = df[inputs].fillna(0.0)
    acc = cross_val_score(_tree(leaves), X, y, cv=5).mean()
    clf = _tree(leaves).fit(X, y)
    rules = leaf_rules(clf, inputs, class_names)
    pure = [r for r in rules if r["purity"] >= PURITY_THRESHOLD]
    covered = sum(r["n"] for r in pure)

    lines = [f"{name}: {len(class_names)} classes, {leaves} leaves, CV {acc:.3f}",
             f"  pure leaves (purity >= {PURITY_THRESHOLD:.0%}): "
             f"{len(pure)}/{len(rules)}, covering {covered}/{len(df)} runs "
             f"({covered/len(df)*100:.0f}%)", "", "  CONFIDENT RULES:"]
    for r in sorted(pure, key=lambda r: -r["purity"]):
        lines.append(f"   [{r['purity']:.0%}, n={r['n']:>4}] {r['class']}")
        lines.append(f"        IF {r['rule']}")
    mixed = [r for r in rules if r["purity"] < PURITY_THRESHOLD]
    if mixed:
        lines.append("\n  MIXED / uncertain leaves (no confident call):")
        for r in mixed:
            lines.append(f"   [{r['purity']:.0%}, n={r['n']:>4}] ~{r['class']}  (gini {r['gini']:.2f})")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(OUT, f"rules_{name}.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    # figure
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_tree(clf, feature_names=inputs, class_names=class_names, filled=True, rounded=True,
              proportion=True, impurity=True, precision=1, fontsize=10, ax=ax)
    ax.set_title(f"{name} — {leaves} leaves, CV {acc:.2f}, "
                 f"{len(pure)} pure leaves cover {covered/len(df)*100:.0f}% of runs", fontsize=14)
    fig.savefig(os.path.join(OUT, f"tree_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return name, acc, len(pure), len(rules), covered / len(df)


def main():
    C.apply_joule_style()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(C.FEATURES_CSV)
    dec = decentr_bins(df)

    res = []
    res.append(run(df, "has_local", (dec > 0).astype(int),
                   ["No local", "Local"], leaves=3))
    res.append(run(df, "decentralisation", dec,
                   ["No local", "Low local", "High local"], leaves=5))
    k4 = KMeans(4, random_state=C.RANDOM_STATE, n_init=10).fit_predict(_z(df, C.OUTPUTS_OF_INTEREST))
    k4_names = [C.name_cluster(df.loc[k4 == g, C.OUTPUTS_OF_INTEREST].mean()) for g in range(4)]
    res.append(run(df.assign(g=k4), "k4_archetypes", pd.Series(k4),
                   k4_names, leaves=6))

    print("\n" + "=" * 60)
    print("SUMMARY (CV acc / pure leaves / coverage of confident rules)")
    for n, acc, npure, ntot, cov in res:
        print(f"  {n:<18} CV {acc:.2f}  pure {npure}/{ntot}  covers {cov*100:.0f}%")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()