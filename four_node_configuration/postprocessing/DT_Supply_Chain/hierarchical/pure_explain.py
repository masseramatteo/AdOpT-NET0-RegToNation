"""
Paper-style EXPLAIN tree (Baader/Wiest): predict the k-Means infrastructure cluster
from the OUTPUTS OF INTEREST (not the scenario inputs). High accuracy (~0.90-0.98),
because the tree recovers the threshold rules that define each archetype.

Reports TRAIN and TEST accuracy (90/10 stratified), plots the tree, and writes the
per-leaf defining rules (high-purity leaves only). This is the descriptive typology
figure — pair it with the driver tree / confident rules for 'which conditions'.

Outputs: DT_supplychain/hierarchical/explain/

Run:  python pure_explain.py
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
from pure_leaves import leaf_rules

OUT = os.path.join(C.BASE_FIG_DIR, "hierarchical", "explain")
PURITY = 0.70
CONFIGS = [(4, 7), (6, 9)]     # (k clusters, leaf budget)


def run(df, k, leaves):
    Xooi = df[C.OUTPUTS_OF_INTEREST].fillna(0.0)
    y = KMeans(k, random_state=C.RANDOM_STATE, n_init=10).fit_predict(
        StandardScaler().fit_transform(Xooi))
    names = [C.name_cluster(df.loc[y == g, C.OUTPUTS_OF_INTEREST].mean()) for g in range(k)]

    tr, te = train_test_split(np.arange(len(df)), test_size=0.10,
                              random_state=42, stratify=y)
    clf = DecisionTreeClassifier(max_leaf_nodes=leaves, class_weight="balanced",
                                 random_state=C.RANDOM_STATE)
    clf.fit(Xooi.iloc[tr], y[tr])
    a_tr, a_te = clf.score(Xooi.iloc[tr], y[tr]), clf.score(Xooi.iloc[te], y[te])

    rules = leaf_rules(clf, list(C.OUTPUTS_OF_INTEREST), names)
    pure = [r for r in rules if r["purity"] >= PURITY]
    lines = [f"EXPLAIN tree  k={k}, {leaves} leaves   train {a_tr:.3f}  test {a_te:.3f}",
             f"  {len(pure)}/{len(rules)} pure leaves (>= {PURITY:.0%})", "  DEFINING RULES:"]
    for r in sorted(pure, key=lambda r: -r["purity"]):
        lines.append(f"   [{r['purity']:.0%}, n={r['n']:>4}] {r['class']}  IF {r['rule']}")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(OUT, f"rules_explain_k{k}.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    fig, ax = plt.subplots(figsize=(20, 11))
    plot_tree(clf, feature_names=list(C.OUTPUTS_OF_INTEREST), class_names=names,
              filled=True, rounded=True, proportion=True, impurity=True,
              precision=2, fontsize=9, ax=ax)
    ax.set_title(f"Explain tree — outputs of interest → infrastructure archetype "
                 f"(k={k}, {leaves} leaves, train {a_tr:.2f} / test {a_te:.2f})", fontsize=15)
    fig.savefig(os.path.join(OUT, f"explain_tree_k{k}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return k, leaves, a_tr, a_te


def main():
    C.apply_joule_style()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(C.FEATURES_CSV)
    res = [run(df, k, lv) for k, lv in CONFIGS]
    print("\nSUMMARY (train / test):")
    for k, lv, atr, ate in res:
        print(f"  k={k} ({lv} leaves): train {atr:.3f}  test {ate:.3f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()