"""
v5 DT_uncert: what does a 90-95% accurate driver tree cost?

Takes the meaningful DT_uncert cases (same k-Means labels, same 15 uncertain drivers)
and sweeps decision-tree hyperparameters far beyond the readable range used in
dt_v5_uncertain.py (max_depth, max_leaf_nodes, min_samples_leaf, min_samples_split,
ccp_alpha, criterion, class_weight). For each config it reports:
    train accuracy      fit and scored on all runs (resubstitution)
    ungrouped CV        5-fold StratifiedKFold (siblings of the same economy leak)
    grouped CV          5-fold StratifiedGroupKFold by economy (the honest number)
plus tree size (leaves, depth). RF and gradient boosting give the model ceiling.

Output: <RESULTS_FOLDER>/DT_uncert_accuracy/
    sweep_all.csv           every config x case
    summary.csv             per case: best grouped, smallest tree reaching 90/95% per scale
    pareto_<case>.png       accuracy vs leaves, three scales
    pareto_all.png          grouped + train frontiers of every case

Run:  python dt_v5_accuracy_sweep.py
"""
import itertools
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config as C
import dt_v5 as V

RS = V.RS
OUT = os.path.join(C.RESULTS_FOLDER, "DT_uncert_accuracy")
DRIVERS = [d for d in V.DRIVERS if d != "k_small"]

# (group, definition, k): headline + useful cases of DT_uncert
CASES = [("sysmix", "cent_imp", 3), ("sysmix", "sysmix", 3), ("sysmix", "cent_imp", 4),
         ("combo", "mix_topo", 3), ("decentr", "A_share", 3), ("decentr", "A_share", 5),
         ("topo", "backbone", 3), ("topo", "meshsat", 3)]

DEPTHS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]
LEAVES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, None]
MIN_LEAF = [1, 2, 5, 10, 20, 40]
MIN_SPLIT = [2, 20]
CCP = [0.0, 0.0005, 0.002]
CRIT = ["gini", "entropy"]
CW = ["balanced", None]


def labels(f, group, dname, k):
    """Same k-Means labels as dt_v5_uncertain.py (ordered by first feature mean)."""
    feats = dict((n, ft) for n, ft, _ in V.GROUPS[group])[dname]
    Xc = StandardScaler().fit_transform(f[feats].fillna(0))
    y = KMeans(k, random_state=RS, n_init=10).fit(Xc).labels_
    order = np.argsort([f[feats[0]].values[y == c].mean() for c in range(k)])
    remap = {o: i for i, o in enumerate(order)}
    return np.array([remap[c] for c in y])


def grid():
    out = []
    # depth-limited trees
    for d, ml, ms, a, cr, cw in itertools.product(DEPTHS, MIN_LEAF, MIN_SPLIT, CCP, CRIT, CW):
        out.append(dict(max_depth=d, max_leaf_nodes=None, min_samples_leaf=ml,
                        min_samples_split=ms, ccp_alpha=a, criterion=cr, class_weight=cw))
    # leaf-budget trees (best-first growth, what dt_v5 uses)
    for lv, ml, a, cr, cw in itertools.product(LEAVES, MIN_LEAF, CCP, CRIT, CW):
        if lv is None:
            continue
        out.append(dict(max_depth=None, max_leaf_nodes=lv, min_samples_leaf=ml,
                        min_samples_split=2, ccp_alpha=a, criterion=cr, class_weight=cw))
    return out


CV_G = StratifiedGroupKFold(5, shuffle=True, random_state=RS)
CV_U = StratifiedKFold(5, shuffle=True, random_state=RS)


def score(clf, X, y, g):
    pg = cross_val_predict(clf, X, y, cv=CV_G, groups=g)
    pu = cross_val_predict(clf, X, y, cv=CV_U)
    clf.fit(X, y)
    pt = clf.predict(X)
    return dict(acc_train=accuracy_score(y, pt), kappa_train=cohen_kappa_score(y, pt),
                acc_ungrouped=accuracy_score(y, pu), kappa_ungrouped=cohen_kappa_score(y, pu),
                acc_grouped=accuracy_score(y, pg), kappa_grouped=cohen_kappa_score(y, pg))


def run_cfg(p, X, y, g):
    clf = DecisionTreeClassifier(random_state=RS, **p)
    r = score(clf, X, y, g)
    r.update(p)
    r["leaves"] = clf.get_n_leaves()
    r["depth"] = clf.get_depth()
    return r


def smallest(df, col, thr):
    hit = df[df[col] >= thr]
    if hit.empty:
        return "", ""
    best = hit.sort_values(["leaves", col], ascending=[True, False]).iloc[0]
    return int(best.leaves), int(best.depth)


def frontier(df, col):
    """Best accuracy reachable at <= n leaves (monotone envelope)."""
    s = df.groupby("leaves")[col].max().sort_index()
    return s.index.values, np.maximum.accumulate(s.values)


def main():
    os.makedirs(OUT, exist_ok=True)
    f = V.load(C.EXTRACTED_RESULTS)
    X = f[DRIVERS].fillna(0.0).values
    g = f.economy.values
    cfgs = grid()
    print(f"{len(f)} runs, {f.economy.nunique()} economies, {len(cfgs)} configs x {len(CASES)} cases")

    allrows, summ = [], []
    for group, dname, k in CASES:
        case = f"{group}/{dname}/k{k:02d}"
        y = labels(f, group, dname, k)
        base = pd.Series(y).value_counts(normalize=True).max()
        rows = Parallel(n_jobs=-1)(delayed(run_cfg)(p, X, y, g) for p in cfgs)
        df = pd.DataFrame(rows)
        df.insert(0, "case", case)
        df["baseline"] = base
        allrows.append(df)

        rf = score(RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                          class_weight="balanced", random_state=0, n_jobs=-1),
                   X, y, g)
        hgb = score(HistGradientBoostingClassifier(random_state=0), X, y, g)
        bg = df.sort_values("kappa_grouped", ascending=False).iloc[0]
        row = dict(case=case, baseline=round(base, 3),
                   best_grouped_acc=round(bg.acc_grouped, 3),
                   best_grouped_kappa=round(bg.kappa_grouped, 3),
                   best_leaves=int(bg.leaves), best_depth=int(bg.depth),
                   best_cfg=f"depth={bg.max_depth} leaves={bg.max_leaf_nodes} "
                            f"min_leaf={bg.min_samples_leaf} ccp={bg.ccp_alpha} "
                            f"{bg.criterion} cw={bg.class_weight}",
                   best_train_acc_same_tree=round(bg.acc_train, 3),
                   max_ungrouped_acc=round(df.acc_ungrouped.max(), 3),
                   max_train_acc=round(df.acc_train.max(), 3),
                   rf_grouped_acc=round(rf["acc_grouped"], 3),
                   rf_grouped_kappa=round(rf["kappa_grouped"], 3),
                   hgb_grouped_acc=round(hgb["acc_grouped"], 3),
                   hgb_grouped_kappa=round(hgb["kappa_grouped"], 3))
        for scale in ("train", "ungrouped", "grouped"):
            for thr in (0.90, 0.95):
                lv, dp = smallest(df, f"acc_{scale}", thr)
                row[f"{scale}_{int(thr * 100)}_leaves"] = lv
                row[f"{scale}_{int(thr * 100)}_depth"] = dp
        summ.append(row)
        print(f"{case}: grouped best {row['best_grouped_acc']} ({row['best_leaves']} leaves) | "
              f"ungrouped max {row['max_ungrouped_acc']} | RF {row['rf_grouped_acc']} | "
              f"HGB {row['hgb_grouped_acc']} | train90 leaves {row['train_90_leaves']} "
              f"train95 {row['train_95_leaves']}")

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for col, lab, c in [("acc_train", "train (resubstitution)", C.UU_AMBER),
                            ("acc_ungrouped", "CV ungrouped", C.UU_TEAL),
                            ("acc_grouped", "CV grouped by economy", C.UU_NAVY)]:
            ax.scatter(df.leaves, df[col], s=4, alpha=0.15, color=c)
            xs, ys = frontier(df, col)
            ax.step(xs, ys, where="post", color=c, lw=2, label=lab)
        ax.axhline(rf["acc_grouped"], ls="--", color="grey", lw=1, label="RF grouped")
        ax.axhline(base, ls=":", color="grey", lw=1, label="baseline")
        for t in (0.90, 0.95):
            ax.axhline(t, color="#C00935", lw=0.6, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("tree leaves")
        ax.set_ylabel("accuracy")
        ax.set_ylim(max(0, base - 0.05), 1.01)
        ax.set_title(case, color=C.UU_NAVY)
        ax.legend(fontsize=8, loc="lower right")
        fig.savefig(os.path.join(OUT, f"pareto_{group}_{dname}_k{k:02d}.png"), dpi=160,
                    bbox_inches="tight")
        plt.close(fig)

    big = pd.concat(allrows, ignore_index=True)
    big.round(4).to_csv(os.path.join(OUT, "sweep_all.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "summary.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for case, df in big.groupby("case", sort=False):
        for ax, col in zip(axs, ("acc_grouped", "acc_train")):
            xs, ys = frontier(df, col)
            ax.step(xs, ys, where="post", lw=1.6, label=case)
    for ax, t in zip(axs, ("CV grouped by economy", "train (resubstitution)")):
        ax.set_xscale("log")
        ax.set_xlabel("tree leaves")
        ax.set_title(t, color=C.UU_NAVY)
        ax.axhline(0.90, color="#C00935", lw=0.6)
        ax.axhline(0.95, color="#C00935", lw=0.6)
    axs[0].set_ylabel("best accuracy at <= n leaves")
    axs[1].legend(fontsize=7)
    fig.savefig(os.path.join(OUT, "pareto_all.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
