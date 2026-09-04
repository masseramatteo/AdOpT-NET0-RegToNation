"""
Pure-leaf STAGED tree for scheme H (decentralisation -> structure), with a real
90/10 train/test split.

  Level 1: inputs -> decentralisation  (No local / Low local / High local)
  Level 2: within each decentralisation branch, inputs -> structure
           (Minimal / Satellite / Meshed)

For every tree we report TRAIN and TEST accuracy (fit on the 90% train split,
evaluated on the held-out 10%), and extract only the HIGH-PURITY leaves as confident
"IF ... THEN" rules (purity computed on the training leaves). End-to-end test
accuracy = both decentralisation and structure correct on the held-out set.

Outputs: DT_supplychain/hierarchical/pure_H/

Run:  python pure_staged_H.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from explore_level_schemes import decentr_bins, struct_bins3, name_struct3
from pure_leaves import leaf_rules

OUT = os.path.join(C.BASE_FIG_DIR, "hierarchical", "pure_H")
PURITY = 0.70
L1_LEAVES, L2_LEAVES = 5, 6
DEC_NAME = {0: "No local", 1: "Low local", 2: "High local"}


def _tree(leaves):
    return DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=20,
                                  min_samples_split=20, class_weight="balanced",
                                  random_state=C.RANDOM_STATE)


def _rules_block(clf, inputs, class_names, title):
    rules = leaf_rules(clf, inputs, class_names)
    pure = [r for r in rules if r["purity"] >= PURITY]
    lines = [f"  {title}: {len(pure)}/{len(rules)} pure leaves (>= {PURITY:.0%})"]
    for r in sorted(pure, key=lambda r: -r["purity"]):
        lines.append(f"    [{r['purity']:.0%}, n={r['n']:>4}] {r['class']}  IF {r['rule']}")
    for r in [r for r in rules if r["purity"] < PURITY]:
        lines.append(f"    [mixed {r['purity']:.0%}, n={r['n']:>4}] ~{r['class']} (gini {r['gini']:.2f})")
    return "\n".join(lines)


def main():
    C.apply_joule_style()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(C.FEATURES_CSV)
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    df["dec"] = decentr_bins(df)
    df["st"] = struct_bins3(df)
    df["combo"] = df["dec"] * 10 + df["st"]

    tr, te = train_test_split(df, test_size=0.10, random_state=42, stratify=df["combo"])
    Xtr, Xte = tr[inputs].fillna(0), te[inputs].fillna(0)
    print(f"train {len(tr)}  test {len(te)}  (90/10 stratified)")

    report = [f"STAGED PURE-LEAF TREE — scheme H (decentralisation -> structure)",
              f"train {len(tr)} / test {len(te)}  (90/10 stratified split)", ""]

    # ---- Level 1: decentralisation -----------------------------------------
    t1 = _tree(L1_LEAVES).fit(Xtr, tr["dec"])
    a1_tr, a1_te = t1.score(Xtr, tr["dec"]), t1.score(Xte, te["dec"])
    l1_names = [DEC_NAME[d] for d in [0, 1, 2]]
    report += [f"LEVEL 1 — decentralisation   train {a1_tr:.3f}  test {a1_te:.3f}",
               _rules_block(t1, inputs, l1_names, "L1 leaves"), ""]

    # ---- Level 2: structure within each decentralisation level -------------
    l2_trees = {}
    for d in [0, 1, 2]:
        mtr = tr["dec"] == d
        mte = te["dec"] == d
        if tr.loc[mtr, "st"].nunique() < 2:
            l2_trees[d] = ("const", int(tr.loc[mtr, "st"].iloc[0]))
            report += [f"LEVEL 2 within '{DEC_NAME[d]}': single structure -> "
                       f"{name_struct3(tr.loc[mtr, C.OUTPUTS_OF_INTEREST].mean())}", ""]
            continue
        present = sorted(tr.loc[mtr, "st"].unique())
        cls = [name_struct3(tr.loc[mtr & (tr["st"] == h), C.OUTPUTS_OF_INTEREST].mean()) for h in present]
        tg = _tree(L2_LEAVES).fit(Xtr[mtr], tr.loc[mtr, "st"])
        l2_trees[d] = tg
        a_tr = tg.score(Xtr[mtr], tr.loc[mtr, "st"])
        a_te = tg.score(Xte[mte], te.loc[mte, "st"]) if mte.sum() else float("nan")
        report += [f"LEVEL 2 within '{DEC_NAME[d]}'  train {a_tr:.3f}  test {a_te:.3f}  "
                   f"(n_train={mtr.sum()})",
                   _rules_block(tg, inputs, cls, f"L2/{DEC_NAME[d]} leaves"), ""]

    # ---- end-to-end on held-out test ---------------------------------------
    dpred = t1.predict(Xte)
    spred = np.zeros(len(te), dtype=int)
    for i, d in enumerate(dpred):
        node = l2_trees[d]
        spred[i] = node[1] if isinstance(node, tuple) else node.predict(Xte.iloc[[i]])[0]
    e2e = float(((dpred == te["dec"].values) & (spred == te["st"].values)).mean())
    report += [f"END-TO-END on held-out test: {e2e:.3f} "
               f"(both decentralisation and structure correct)"]

    txt = "\n".join(report)
    print("\n" + txt)
    with open(os.path.join(OUT, "rules_staged_H.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    # ---- figure: L1 + per-branch L2 ----------------------------------------
    branches = [d for d in [0, 1, 2] if not isinstance(l2_trees[d], tuple)]
    fig, axes = plt.subplots(1 + len(branches), 1, figsize=(16, 5.5 * (1 + len(branches))))
    axes = np.atleast_1d(axes)
    plot_tree(t1, feature_names=inputs, class_names=l1_names, filled=True, rounded=True,
              proportion=True, impurity=True, precision=1, fontsize=9, ax=axes[0])
    axes[0].set_title(f"Level 1 — decentralisation  (train {a1_tr:.2f} / test {a1_te:.2f})", fontsize=13)
    for ax, d in zip(axes[1:], branches):
        mtr = tr["dec"] == d
        present = sorted(tr.loc[mtr, "st"].unique())
        cls = [name_struct3(tr.loc[mtr & (tr["st"] == h), C.OUTPUTS_OF_INTEREST].mean()) for h in present]
        plot_tree(l2_trees[d], feature_names=inputs, class_names=cls, filled=True, rounded=True,
                  proportion=True, impurity=True, precision=1, fontsize=9, ax=ax)
        mte = te["dec"] == d
        a_te = l2_trees[d].score(Xte[mte], te.loc[mte, "st"]) if mte.sum() else float("nan")
        ax.set_title(f"Level 2 — structure within '{DEC_NAME[d]}'  (test {a_te:.2f})", fontsize=12)
    fig.suptitle(f"Staged pure-leaf tree — H  |  end-to-end test {e2e:.2f}", fontsize=15, y=1.001)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "staged_pure_H.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()