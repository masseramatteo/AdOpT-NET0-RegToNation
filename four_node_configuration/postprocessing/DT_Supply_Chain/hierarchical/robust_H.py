"""
Scheme H (decentralisation -> Minimal/Satellite/Meshed) on ROBUST runs only.

Robust = small-cluster reopt infeasible OR delta_npv_pct > DELTA_PCT (design genuinely
cost-determined, not a near-cost-neutral tie). Rebuilds H staged (L1 decentralisation
+ per-branch L2 structure) and the single H tree, with 90/10 train/test + CV, pure-leaf
rules. Compares to full-data H.

Outputs: DT_supplychain/robust/H/

Run:  python robust_H.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from explore_level_schemes import decentr_bins, struct_bins3, name_struct3
from pure_leaves import leaf_rules
from robust_trees import robust_run_ids, ROBUST_SCENARIO, DELTA_PCT

OUT = os.path.join(C.BASE_FIG_DIR, "robust", "H")
LEAVES, MIN_LEAF = 12, 10
DEC = {0: "No local", 1: "Low local", 2: "High local"}


def _tree():
    return DecisionTreeClassifier(max_leaf_nodes=LEAVES, min_samples_leaf=MIN_LEAF,
                                  min_samples_split=MIN_LEAF, class_weight="balanced",
                                  random_state=C.RANDOM_STATE)


def _staged_e2e(X, dec, st, k1=3):
    combined = dec * 10 + st
    skf = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    acc = []
    for tr, te in skf.split(X, combined):
        t1 = _tree().fit(X.iloc[tr], dec[tr])
        dp = t1.predict(X.iloc[te])
        sp = np.zeros(len(te), dtype=int)
        for d in range(k1):
            md = dec[tr] == d
            if len(np.unique(st[tr][md])) > 1:
                tg = _tree().fit(X.iloc[tr][md], st[tr][md])
                sel = dp == d
                if sel.any():
                    sp[sel] = tg.predict(X.iloc[te][sel])
        acc.append(((dp == dec[te]) & (sp == st[te])).mean())
    return float(np.mean(acc))


def run(df, tag, out):
    os.makedirs(out, exist_ok=True)
    df = df.reset_index(drop=True)
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    X = df[inputs].fillna(0)
    dec = decentr_bins(df); st = struct_bins3(df)

    a_dec = cross_val_score(_tree(), X, dec, cv=5).mean()
    e2e = _staged_e2e(X, dec, st)
    print(f"[{tag}] n={len(df)}  L1 decentr CV {a_dec:.3f}  staged end-to-end CV {e2e:.3f}")

    # staged figure: L1 + per-branch L2, with 90/10 test acc in titles
    tr, te = train_test_split(np.arange(len(df)), test_size=0.1, random_state=42,
                              stratify=dec * 10 + st)
    branches = [d for d in [0, 1, 2] if pd.Series(st[dec == d]).nunique() > 1]
    fig, axes = plt.subplots(1 + len(branches), 1, figsize=(16, 5.5 * (1 + len(branches))))
    axes = np.atleast_1d(axes)
    t1 = _tree().fit(X.iloc[tr], dec[tr])
    a1_te = t1.score(X.iloc[te], dec[te])
    plot_tree(t1, feature_names=inputs, class_names=[DEC[d] for d in [0, 1, 2]], filled=True,
              rounded=True, proportion=True, impurity=True, precision=1, fontsize=9, ax=axes[0])
    axes[0].set_title(f"L1 decentralisation (test {a1_te:.2f})", fontsize=13)
    rules_txt = [f"{tag}: n={len(df)}  L1 decentr CV {a_dec:.3f}  staged e2e CV {e2e:.3f}", ""]
    for ax, d in zip(axes[1:], branches):
        md = dec == d
        present = sorted(np.unique(st[md]))
        cls = [name_struct3(df.loc[md & (st == h), C.OUTPUTS_OF_INTEREST].mean()) for h in present]
        idx = np.where(md)[0]
        tr_d = [i for i in tr if md[i]]; te_d = [i for i in te if md[i]]
        tg = _tree().fit(X.iloc[tr_d], st[tr_d])
        a_te = tg.score(X.iloc[te_d], st[te_d]) if te_d else float("nan")
        plot_tree(tg, feature_names=inputs, class_names=cls, filled=True, rounded=True,
                  proportion=True, impurity=True, precision=1, fontsize=8, ax=ax)
        ax.set_title(f"L2 structure within '{DEC[d]}' (test {a_te:.2f}, n={md.sum()})", fontsize=12)
        rules_txt.append(f"L2 within {DEC[d]} (test {a_te:.2f}):")
        for r in leaf_rules(tg, inputs, cls):
            mark = "" if r["purity"] >= 0.70 else "  [mixed]"
            rules_txt.append(f"   [{r['purity']:.0%}, n={r['n']}] {r['class']}{mark}  IF {r['rule']}")
    fig.suptitle(f"Scheme H — {tag} runs | staged end-to-end CV {e2e:.2f}", fontsize=15, y=1.001)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"H_staged_{tag}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(out, f"H_rules_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rules_txt) + "\n")
    return {"tag": tag, "n": len(df), "L1_decentr_CV": a_dec, "staged_e2e_CV": e2e}


def main():
    C.apply_joule_style()
    os.makedirs(OUT, exist_ok=True)
    feat = pd.read_csv(C.FEATURES_CSV)
    rob_ids, *_ = robust_run_ids(ROBUST_SCENARIO)
    robust = feat[feat["run"].isin(rob_ids)].copy()
    print(f"Robust filter: {ROBUST_SCENARIO} (infeasible OR delta_npv>{DELTA_PCT}%) "
          f"-> {len(robust)}/{len(feat)} runs\n")
    rows = [run(feat, "FULL", OUT), run(robust, "ROBUST", OUT)]
    res = pd.DataFrame(rows).set_index("tag")
    print("\n" + "=" * 56)
    print(res.T.to_string())
    res.T.to_csv(os.path.join(OUT, "H_full_vs_robust.csv"))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()