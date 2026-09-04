"""
Rebuild the key trees on ROBUST runs only.

A run is 'robust/strong' if its cost-optimal design is genuinely determined — i.e. in
the near-optimal re-optimisation (reopt) that excludes the small-cluster investment,
the run becomes INFEASIBLE or loses > DELTA_PCT % of NPV. Degenerate runs (removing
the tech barely changes NPV) are dropped, as their design is a near-cost-neutral tie.

Compares FULL vs ROBUST accuracy for the trees we care about:
  - explain tree (k-Means clusters -> archetype, from outputs of interest)
  - has_local driver (inputs -> local electrolysis? )
  - decentralisation driver (inputs -> None/Low/High)
  - three-level staged end-to-end

Saves the robust-subset figures + a comparison to DT_supplychain/robust/.

Run:  python robust_trees.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from explore_level_schemes import decentr_bins, struct_bins3, name_struct3
from pure_leaves import leaf_rules

REOPT_BASE = r"X:\3000_simulations_24_may_2026"
SCENARIOS = {
    "small_components": "reopt_comparison_Snellius_3000_simulations_small_clusters_components",
    "small_electrolyzer": "reopt_comparison_Snellius_3000_simulations_small_electrolyzer_only",
    "no_networks": "reopt_comparison_Snellius_3000_simulations_no_networks",
}
ROBUST_SCENARIO = "small_components"   # the discriminating filter for decentralisation
DELTA_PCT = 1.0
OUT = os.path.join(C.BASE_FIG_DIR, "robust")


def robust_run_ids(scenario):
    f = os.path.join(REOPT_BASE, SCENARIOS[scenario], "reopt_comparison.xlsx")
    df = pd.read_excel(f, sheet_name=0)
    dpct = pd.to_numeric(df["delta_npv_pct"], errors="coerce")
    true_infeas = df["npv_reopt"].isna() & df["error"].isna()
    robust = df.loc[(dpct > DELTA_PCT) | true_infeas, "run_id"]
    return set(robust), df, dpct, true_infeas


def _tree(leaves):
    return DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=20,
                                  min_samples_split=20, class_weight="balanced",
                                  random_state=C.RANDOM_STATE)


def cv(X, y, leaves=6):
    return cross_val_score(_tree(leaves), X, y, cv=5).mean()


def explain_cv(df):
    Xooi = df[C.OUTPUTS_OF_INTEREST].fillna(0)
    y = KMeans(4, random_state=C.RANDOM_STATE, n_init=10).fit_predict(
        StandardScaler().fit_transform(Xooi))
    return cross_val_score(_tree(7), Xooi, y, cv=5).mean(), y


def staged_e2e(df, X):
    dec = decentr_bins(df); st = struct_bins3(df)
    combined = dec * 10 + st
    skf = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    acc = []
    for tr, te in skf.split(X, combined):
        t1 = _tree(6).fit(X.iloc[tr], dec[tr])
        dp = t1.predict(X.iloc[te])
        sp = np.zeros(len(te), dtype=int)
        for d in [0, 1, 2]:
            md = dec[tr] == d
            if len(np.unique(st[tr][md])) > 1:
                tg = _tree(6).fit(X.iloc[tr][md], st[tr][md])
                sel = dp == d
                if sel.any():
                    sp[sel] = tg.predict(X.iloc[te][sel])
        acc.append(((dp == dec[te]) & (sp == st[te])).mean())
    return np.mean(acc)


def block(df, tag):
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    X = df[inputs].fillna(0)
    dec = decentr_bins(df)
    haslocal = (dec > 0).astype(int)
    exp, _ = explain_cv(df)
    return {
        "tag": tag, "n": len(df),
        "explain_k4": exp,
        "has_local": cv(X, haslocal, 3),
        "decentr": cv(X, dec, 5),
        "staged_e2e": staged_e2e(df, X),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    feat = pd.read_csv(C.FEATURES_CSV)

    rob_ids, rdf, dpct, infeas = robust_run_ids(ROBUST_SCENARIO)
    feat["robust"] = feat["run"].isin(rob_ids)
    full = feat
    robust = feat[feat["robust"]].copy()
    print(f"Robustness filter = {ROBUST_SCENARIO} (infeasible OR delta_npv > {DELTA_PCT}%)")
    print(f"  full: {len(full)} runs   robust: {len(robust)} runs "
          f"({len(robust)/len(full)*100:.0f}%)")

    # robustness by scenario, for reference
    for name in SCENARIOS:
        ids, _, _, _ = robust_run_ids(name)
        print(f"  [{name}] robust runs matched in features: {feat['run'].isin(ids).sum()}")

    rows = [block(full, "FULL"), block(robust, "ROBUST")]
    res = pd.DataFrame(rows).set_index("tag")
    print("\n" + "=" * 66)
    print("ACCURACY: FULL vs ROBUST (5-fold CV)")
    print("=" * 66)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(res.T)
    res.T.to_csv(os.path.join(OUT, "full_vs_robust_accuracy.csv"))

    # save the robust explain tree + has_local tree as figures
    _fig_explain(robust)
    _fig_haslocal(robust)
    robust[["run"]].to_csv(os.path.join(OUT, "robust_run_ids.csv"), index=False)
    print(f"\nSaved -> {OUT}")


def _fig_explain(df):
    C.apply_joule_style()
    Xooi = df[C.OUTPUTS_OF_INTEREST].fillna(0)
    y = KMeans(4, random_state=C.RANDOM_STATE, n_init=10).fit_predict(
        StandardScaler().fit_transform(Xooi))
    names = [C.name_cluster(df.loc[y == g, C.OUTPUTS_OF_INTEREST].mean()) for g in range(4)]
    tr, te = train_test_split(np.arange(len(df)), test_size=0.1, random_state=42, stratify=y)
    clf = _tree(7).fit(Xooi.iloc[tr], y[tr])
    a_tr, a_te = clf.score(Xooi.iloc[tr], y[tr]), clf.score(Xooi.iloc[te], y[te])
    fig, ax = plt.subplots(figsize=(20, 11))
    plot_tree(clf, feature_names=list(C.OUTPUTS_OF_INTEREST), class_names=names,
              filled=True, rounded=True, proportion=True, impurity=True, precision=2,
              fontsize=9, ax=ax)
    ax.set_title(f"ROBUST explain tree (k=4, outputs → archetype, n={len(df)}, "
                 f"train {a_tr:.2f}/test {a_te:.2f})", fontsize=15)
    fig.savefig(os.path.join(OUT, "robust_explain_tree_k4.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _fig_haslocal(df):
    C.apply_joule_style()
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    X = df[inputs].fillna(0).reset_index(drop=True)
    y = (decentr_bins(df) > 0).astype(int)   # ndarray
    tr, te = train_test_split(np.arange(len(df)), test_size=0.1, random_state=42, stratify=y)
    clf = _tree(3).fit(X.iloc[tr], y[tr])
    a_tr, a_te = clf.score(X.iloc[tr], y[tr]), clf.score(X.iloc[te], y[te])
    rules = leaf_rules(clf, inputs, ["No local", "Local"])
    with open(os.path.join(OUT, "robust_has_local_rules.txt"), "w", encoding="utf-8") as f:
        f.write(f"ROBUST has_local  train {a_tr:.3f} test {a_te:.3f}  n={len(df)}\n")
        for r in sorted(rules, key=lambda r: -r["purity"]):
            f.write(f"  [{r['purity']:.0%}, n={r['n']}] {r['class']}  IF {r['rule']}\n")
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(clf, feature_names=inputs, class_names=["No local", "Local"], filled=True,
              rounded=True, proportion=True, impurity=True, precision=1, fontsize=10, ax=ax)
    ax.set_title(f"ROBUST has_local (inputs → local electrolysis?, n={len(df)}, "
                 f"train {a_tr:.2f}/test {a_te:.2f})", fontsize=14)
    fig.savefig(os.path.join(OUT, "robust_has_local_tree.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()