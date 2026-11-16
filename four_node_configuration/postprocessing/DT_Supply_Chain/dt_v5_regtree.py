"""
v5 DT_uncert: let the tree make the types — multi-output regression tree
(predictive clustering tree) vs k-Means + classification tree.

For every DT_uncert definition the clustering features become continuous TARGETS
(standardized). Two routes are compared on the same scale, grouped-CV R^2 by economy:

  regtree   DecisionTreeRegressor on all targets jointly; each leaf = one type
  kmeans    k-Means (k = 3..10) labels -> DecisionTreeClassifier -> predicted class
            replaced by its centroid in target space
  oracle_k  R^2 of k-Means if the class were predicted perfectly (quantisation cap)
  ceilings  RandomForestRegressor, HistGradientBoostingRegressor (one per target)

R^2 is pooled over standardized targets: 1 - SSE / SST. Leaf budget is the shared axis.

Output: <RESULTS_FOLDER>/DT_uncert_accuracy/regtree/
    curves.csv              definition x method x leaves (x k): grouped R^2, train R^2
    summary.csv             per definition: best of each route, plateau leaves, ceilings
    r2_<group>_<def>.png    grouped R^2 vs leaves: regtree, kmeans per k, oracle, ceilings
    trees/<group>_<def>/L<n>/{rules.txt, leaf_profiles.csv, tree.pdf, tree.png}
                            for the 8-leaf tree and the plateau tree

Run:  python dt_v5_regtree.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

import config as C
import dt_v5 as V
import dt_v5_accuracy_sweep as S

RS = V.RS
OUT = os.path.join(S.OUT, "regtree")
DRIVERS = S.DRIVERS
DERIVED = False         # --derived: base drivers + supply ratios, output to regtree_derived/


def use_derived():
    """Switch to the supply-ratio driver set of dt_v5_derived_ratios.py."""
    global OUT, DRIVERS, DERIVED
    import dt_v5_derived_ratios as D
    OUT = os.path.join(S.OUT, "regtree_derived")
    DRIVERS = D.SETS["supply"]
    DERIVED = True


def load():
    f = V.load(C.EXTRACTED_RESULTS)
    if DERIVED:
        import dt_v5_derived_ratios as D
        f = D.add_ratios(f)
    return f
LEAVES = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
MIN_LEAF = [5, 20, 40]
KS = range(3, 11)
PLATEAU_TOL = 0.01      # plateau = smallest tree within 0.01 R^2 of the best
MERGE_SD = 0.25         # leaves whose mean profiles are closer than this (in SD) = same type
CV = GroupKFold(5)      # targets are continuous; grouped by economy


def r2(Y, P):
    P = np.asarray(P).reshape(Y.shape)      # single-target regressors return 1-D
    return 1.0 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def regtree_cfg(X, Y, g, lv, ml):
    m = DecisionTreeRegressor(max_leaf_nodes=lv, min_samples_leaf=ml, random_state=RS)
    P = cross_val_predict(m, X, Y, cv=CV, groups=g)
    m.fit(X, Y)
    return dict(method="regtree", k=np.nan, leaves_budget=lv, min_leaf=ml,
                leaves=m.get_n_leaves(), r2_grouped=r2(Y, P), r2_train=r2(Y, m.predict(X)))


def kmeans_cfg(X, Y, g, y, cent, k, lv, ml):
    if lv < k:
        return None
    m = DecisionTreeClassifier(max_leaf_nodes=lv, min_samples_leaf=ml, random_state=RS)
    p = cross_val_predict(m, X, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=RS),
                          groups=g)
    m.fit(X, y)
    return dict(method="kmeans", k=k, leaves_budget=lv, min_leaf=ml, leaves=m.get_n_leaves(),
                r2_grouped=r2(Y, cent[p]), r2_train=r2(Y, cent[m.predict(X)]),
                acc_grouped=(p == y).mean())


def best_per_leaves(df):
    return df.sort_values("r2_grouped", ascending=False).groupby("leaves_budget").head(1) \
        .sort_values("leaves_budget")


def leaf_types(tree, X, Yraw, feats):
    """Leaf mean profiles, plus a merge of leaves with near-identical profiles."""
    leaf = tree.apply(X)
    prof = pd.DataFrame(Yraw, columns=feats).groupby(leaf).agg(["mean", "std"])
    n = pd.Series(leaf).value_counts()
    means = pd.DataFrame(Yraw, columns=feats).groupby(leaf).mean()
    sd = Yraw.std(0)
    sd[sd == 0] = 1
    Z = means.values / sd
    if len(Z) > 1:
        grp = fcluster(linkage(Z, "average"), MERGE_SD * np.sqrt(len(feats)), "distance")
    else:
        grp = np.array([1])
    out = means.copy()
    out.columns = [f"{c}_mean" for c in feats]
    for c in feats:
        out[f"{c}_std"] = prof[(c, "std")].values
    out.insert(0, "n", n.reindex(out.index).values)
    out.insert(1, "type", grp)
    return out, len(set(grp))


def write_tree(tree, X, Yraw, feats, d, title):
    os.makedirs(d, exist_ok=True)
    prof, ntypes = leaf_types(tree, X, Yraw, feats)
    prof.round(4).to_csv(os.path.join(d, "leaf_profiles.csv"))
    t = tree.tree_
    lines = []

    def walk(i, cond):
        if t.children_left[i] == -1:
            r = prof.loc[i]
            pr = " / ".join(f"{c} {r[f'{c}_mean']:.2f}±{r[f'{c}_std']:.2f}" for c in feats)
            lines.append((int(r.n), f"[n={int(r.n)}, type {int(r.type)}] {pr}\n"
                                    f"    IF {' AND '.join(cond) if cond else '(root)'}"))
            return
        f, th = DRIVERS[t.feature[i]], t.threshold[i]
        walk(t.children_left[i], cond + [f"{f} <= {th:.1f}"])
        walk(t.children_right[i], cond + [f"{f} > {th:.1f}"])

    walk(0, [])
    with open(os.path.join(d, "rules.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n# targets: {', '.join(feats)} (leaf mean ± std, raw units)\n"
                 f"# {len(prof)} leaves -> {ntypes} distinct types (merge < {MERGE_SD} SD)\n\n")
        for _, s in sorted(lines, key=lambda q: -q[0]):
            fh.write(s + "\n")
    nl = tree.get_n_leaves()
    fig, ax = plt.subplots(figsize=(min(max(14, nl * 1.2), 160), min(max(7, tree.get_depth() * 1.6), 40)))
    plot_tree(tree, feature_names=DRIVERS, filled=True, rounded=True, precision=2,
              fontsize=7 if nl <= 16 else 5, ax=ax, impurity=False)
    ax.set_title(title, fontsize=12, color=C.UU_NAVY)
    for ext, dpi in (("png", 110 if nl > 40 else 160), ("pdf", None)):
        fig.savefig(os.path.join(d, f"tree.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return ntypes


def main():
    os.makedirs(OUT, exist_ok=True)
    f = load()
    X = f[DRIVERS].fillna(0.0).values
    g = f.economy.values
    curves, summ = [], []
    for group, defs in V.GROUPS.items():
        for dname, feats, desc in defs:
            tag = f"{group}_{dname}"
            Yraw = f[feats].fillna(0).values.astype(float)
            Y = StandardScaler().fit_transform(Yraw)

            rows = Parallel(n_jobs=-1)(delayed(regtree_cfg)(X, Y, g, lv, ml)
                                       for lv in LEAVES for ml in MIN_LEAF)
            oracle = {}
            nun = f[feats].round(3).drop_duplicates().shape[0]
            for k in [k for k in KS if k <= nun]:
                y = S.labels(f, group, dname, k)
                cent = np.vstack([Y[y == c].mean(0) for c in range(k)])
                oracle[k] = r2(Y, cent[y])
                rows += [r for r in Parallel(n_jobs=-1)(
                    delayed(kmeans_cfg)(X, Y, g, y, cent, k, lv, ml)
                    for lv in LEAVES for ml in MIN_LEAF) if r is not None]
            df = pd.DataFrame(rows)
            df.insert(0, "definition", f"{group}/{dname}")
            curves.append(df)

            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0,
                                       n_jobs=-1)
            rf_r2 = r2(Y, cross_val_predict(rf, X, Y, cv=CV, groups=g))
            hgb = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=0))
            hgb_r2 = r2(Y, cross_val_predict(hgb, X, Y, cv=CV, groups=g))

            rt = best_per_leaves(df[df.method == "regtree"])
            best = rt.r2_grouped.max()
            plat = rt[rt.r2_grouped >= best - PLATEAU_TOL].iloc[0]
            r8 = rt[rt.leaves_budget == 8].iloc[0]
            km = df[df.method == "kmeans"]
            kbest = km.sort_values("r2_grouped", ascending=False).iloc[0]
            # k-Means at the same leaf budget as the regtree plateau / at 8 leaves
            km8 = km[km.leaves_budget == 8].sort_values("r2_grouped", ascending=False)
            kmp = km[km.leaves_budget == plat.leaves_budget].sort_values("r2_grouped",
                                                                         ascending=False)

            tdir = os.path.join(OUT, "trees", tag)
            fitted = {}
            for lab, r in (("L8", r8), (f"L{int(plat.leaves_budget)}", plat)):
                m = DecisionTreeRegressor(max_leaf_nodes=int(r.leaves_budget),
                                          min_samples_leaf=int(r.min_leaf),
                                          random_state=RS).fit(X, Y)
                fitted[lab] = write_tree(
                    m, X, Yraw, feats, os.path.join(tdir, lab),
                    f"{group}/{dname} regression tree, {m.get_n_leaves()} leaves | grouped R2 "
                    f"{r.r2_grouped:.3f} | train R2 {r.r2_train:.3f}")

            summ.append(dict(
                definition=f"{group}/{dname}", n_targets=len(feats),
                regtree_r2_L8=round(r8.r2_grouped, 3), regtree_types_L8=fitted["L8"],
                regtree_r2_best=round(best, 3),
                regtree_plateau_leaves=int(plat.leaves_budget),
                regtree_r2_plateau=round(plat.r2_grouped, 3),
                regtree_types_plateau=fitted[f"L{int(plat.leaves_budget)}"],
                regtree_train_r2_plateau=round(plat.r2_train, 3),
                kmeans_r2_L8=round(km8.iloc[0].r2_grouped, 3) if len(km8) else np.nan,
                kmeans_k_L8=int(km8.iloc[0].k) if len(km8) else np.nan,
                kmeans_r2_at_plateau_leaves=round(kmp.iloc[0].r2_grouped, 3) if len(kmp) else np.nan,
                kmeans_r2_best=round(kbest.r2_grouped, 3), kmeans_best_k=int(kbest.k),
                kmeans_best_leaves=int(kbest.leaves_budget),
                oracle_r2_k3=round(oracle.get(3, np.nan), 3),
                oracle_r2_k10=round(oracle.get(max(oracle), np.nan), 3) if oracle else np.nan,
                rf_r2=round(rf_r2, 3), hgb_r2=round(hgb_r2, 3)))
            s = summ[-1]
            print(f"{group}/{dname:14s} regtree L8 {s['regtree_r2_L8']:.3f} "
                  f"best {s['regtree_r2_best']:.3f} (plateau L{s['regtree_plateau_leaves']}, "
                  f"{s['regtree_types_plateau']} types) | kmeans L8 {s['kmeans_r2_L8']} "
                  f"best {s['kmeans_r2_best']} (k{s['kmeans_best_k']}, L{s['kmeans_best_leaves']}) "
                  f"| RF {s['rf_r2']} HGB {s['hgb_r2']}")

            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            ax.plot(rt.leaves_budget, rt.r2_grouped, "-o", color=C.UU_NAVY, lw=2.2,
                    label="regression tree (grouped)")
            ax.plot(rt.leaves_budget, rt.r2_train, ":", color=C.UU_NAVY, lw=1,
                    label="regression tree (train)")
            cmap = plt.get_cmap("viridis")
            for i, k in enumerate(sorted(oracle)):
                kk = best_per_leaves(km[km.k == k])
                ax.plot(kk.leaves_budget, kk.r2_grouped, "-", lw=1,
                        color=cmap(i / max(1, len(oracle) - 1)), label=f"k-Means k={k}")
            ax.axhline(hgb_r2, ls="--", color="grey", lw=1, label="HGB ceiling")
            ax.axhline(rf_r2, ls="-.", color="grey", lw=1, label="RF")
            ax.set_xscale("log")
            ax.set_xlabel("tree leaves")
            ax.set_ylabel("R² on outputs (pooled, standardized)")
            ax.set_ylim(0, 1)
            ax.set_title(f"{group}/{dname} — targets: {', '.join(feats)}", fontsize=9,
                         color=C.UU_NAVY)
            ax.legend(fontsize=6, ncol=2, loc="lower right")
            fig.savefig(os.path.join(OUT, f"r2_{tag}.png"), dpi=160, bbox_inches="tight")
            plt.close(fig)

    pd.concat(curves, ignore_index=True).round(4).to_csv(os.path.join(OUT, "curves.csv"),
                                                         index=False)
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print("wrote", OUT)


def accuracy():
    """Classification-style numbers for the regression trees, so they can be quoted next to
    the k-Means trees: out-of-fold predicted profile -> nearest k-Means centroid (k = 3),
    accuracy / kappa vs the true k-Means class, plus MAE per target in raw units.
    Done for the 8-leaf and the plateau tree; k-Means classifier at k = 3 and the same leaf
    budget alongside."""
    from sklearn.metrics import cohen_kappa_score
    f = load()
    X = f[DRIVERS].fillna(0.0).values
    g = f.economy.values
    cur = pd.read_csv(os.path.join(OUT, "curves.csv"))
    summ = pd.read_csv(os.path.join(OUT, "summary.csv"))
    rows = []
    for group, defs in V.GROUPS.items():
        for dname, feats, _ in defs:
            d = f"{group}/{dname}"
            Yraw = f[feats].fillna(0).values.astype(float)
            sc = StandardScaler().fit(Yraw)
            Y = sc.transform(Yraw)
            y = S.labels(f, group, dname, 3)
            cent = np.vstack([Y[y == c].mean(0) for c in range(3)])
            base = pd.Series(y).value_counts(normalize=True).max()
            rt = best_per_leaves(cur[(cur.definition == d) & (cur.method == "regtree")])
            km = cur[(cur.definition == d) & (cur.method == "kmeans") & (cur.k == 3)]
            plat = int(summ.loc[summ.definition == d, "regtree_plateau_leaves"].iloc[0])
            for lab, lv in (("L8", 8), ("plateau", plat)):
                r = rt[rt.leaves_budget == lv].iloc[0]
                m = DecisionTreeRegressor(max_leaf_nodes=lv, min_samples_leaf=int(r.min_leaf),
                                          random_state=RS)
                P = np.asarray(cross_val_predict(m, X, Y, cv=CV, groups=g)).reshape(Y.shape)
                p = ((P[:, None, :] - cent[None]) ** 2).sum(2).argmin(1)
                mae = np.abs(sc.inverse_transform(P) - Yraw).mean(0)
                kk = km[km.leaves_budget == lv].sort_values("r2_grouped", ascending=False)
                rows.append(dict(
                    definition=d, tree=lab, leaves=lv, r2_grouped=round(r.r2_grouped, 3),
                    baseline_k3=round(base, 3),
                    acc_k3_grouped=round((p == y).mean(), 3),
                    kappa_k3_grouped=round(cohen_kappa_score(y, p), 3),
                    kmeans_clf_acc_k3=round(kk.iloc[0].acc_grouped, 3) if len(kk) else np.nan,
                    kmeans_clf_r2_k3=round(kk.iloc[0].r2_grouped, 3) if len(kk) else np.nan,
                    mae=" / ".join(f"{c} {v:.3f}" for c, v in zip(feats, mae))))
                print(rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "accuracy.csv"), index=False)
    print("wrote", os.path.join(OUT, "accuracy.csv"))


if __name__ == "__main__":
    import sys
    if "--derived" in sys.argv:
        use_derived()
    if "--accuracy" in sys.argv:
        accuracy()
    else:
        main()
        accuracy()
