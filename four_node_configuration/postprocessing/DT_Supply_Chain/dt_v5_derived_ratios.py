"""
v5 DT_uncert: do derived physical ratios raise driver-tree accuracy?

Axis-aligned trees cannot build ratios of two inputs. This test adds ratios, all
computed from the sampled parameters only (definitions follow sampling_design.py):

    k_small        small-cluster grid connection / electricity need of one small node
    k_large        same for one large node
    g              grid adequacy: (2 avail_large + 2 avail_small) / total electricity need
    supply_margin  g + import_availability_ratio  (feasibility margin of the design)
    h2_vs_elec     hydrogen_import_price / (electricity_price_avg / eta)  (import vs
                   electricity cost of one MWh H2)
    pipe_SL, pipe_SS, pipe_LL   pipeline_cost_multiplier x distance (pipe cost proxy)

Feature sets: base (15 uncertain), +k_small, +supply ratios, +all ratios.
Per case and set, grouped CV by economy (5 folds):
    classification tree on k-Means labels (DT_uncert cases): 8 leaves and best <= 64
    regression tree on the clustering features (R^2, + accuracy via nearest k=3 centroid)
    HistGradientBoosting ceiling for both

Output: <RESULTS_FOLDER>/DT_uncert_accuracy/derived/
    classification.csv, regression.csv, importance.csv, index.html
    trees/<case>/<set>_L8/{rules.txt, tree.png, tree.pdf}  (classification, base vs all)

Run:  python dt_v5_derived_ratios.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

import config as C
import dt_v5 as V
import dt_v5_accuracy_sweep as S
import dt_v5_regtree as RT

RS = V.RS
OUT = os.path.join(S.OUT, "derived")
ETA = 0.655
BASE = S.DRIVERS
SUPPLY = ["k_small", "k_large", "g", "supply_margin", "h2_vs_elec"]
PIPE = ["pipe_SL", "pipe_SS", "pipe_LL"]
SETS = {"base": BASE, "k_small": BASE + ["k_small"], "supply": BASE + SUPPLY,
        "all": BASE + SUPPLY + PIPE}
LEAVES = [8, 12, 16, 24, 32, 48, 64]
MIN_LEAF = [10, 20, 40]
CV_C = StratifiedGroupKFold(5, shuffle=True, random_state=RS)
CV_R = GroupKFold(5)


def add_ratios(f):
    p_req = f.total_demand_TWh * 1e6 / 8760.0 / ETA                     # MW, whole system
    r = f.demand_level_ratio
    f["k_small"] = f.electricity_availability_small / (p_req / (1 + r) / 2)
    f["k_large"] = f.electricity_availability_large / (p_req * r / (1 + r) / 2)
    f["g"] = (2 * f.electricity_availability_large + 2 * f.electricity_availability_small) / p_req
    f["supply_margin"] = f.g + f.import_availability_ratio
    f["h2_vs_elec"] = f.hydrogen_import_price / (f.electricity_price_avg / ETA)
    f["pipe_SL"] = f.pipeline_cost_multiplier * f.d_SL_mean
    f["pipe_SS"] = f.pipeline_cost_multiplier * f.d_SS
    f["pipe_LL"] = f.pipeline_cost_multiplier * f.d_LL
    return f


def clf_cfg(X, y, g, lv, ml):
    m = DecisionTreeClassifier(max_leaf_nodes=lv, min_samples_leaf=ml, random_state=RS)
    p = cross_val_predict(m, X, y, cv=CV_C, groups=g)
    return lv, ml, accuracy_score(y, p), cohen_kappa_score(y, p)


def reg_cfg(X, Y, g, lv, ml, cent, y3):
    m = DecisionTreeRegressor(max_leaf_nodes=lv, min_samples_leaf=ml, random_state=RS)
    P = np.asarray(cross_val_predict(m, X, Y, cv=CV_R, groups=g)).reshape(Y.shape)
    p = ((P[:, None, :] - cent[None]) ** 2).sum(2).argmin(1)
    return lv, ml, RT.r2(Y, P), accuracy_score(y3, p)


def write_clf_tree(m, names, cnames, d, title):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "rules.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        for r in V.leaf_rules(m, names, cnames):
            mark = "" if r["purity"] >= 0.70 else "  [mixed]"
            fh.write(f"[{r['purity']:.0%}, n={r['n']}] {r['cls']}{mark}\n    IF {r['rule']}\n")
    fig, ax = plt.subplots(figsize=(17, 8))
    plot_tree(m, feature_names=names, class_names=cnames, filled=True, rounded=True,
              proportion=True, precision=2, fontsize=7, ax=ax)
    ax.set_title(title, fontsize=11, color=C.UU_NAVY)
    for ext, dpi in (("png", 160), ("pdf", None)):
        fig.savefig(os.path.join(d, f"tree.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    f = add_ratios(V.load(C.EXTRACTED_RESULTS))
    g = f.economy.values
    crow, rrow, irow = [], [], []

    # ---- classification on the DT_uncert k-Means labels -----------------------------
    for group, dname, k in S.CASES:
        case = f"{group}/{dname}/k{k:02d}"
        y = S.labels(f, group, dname, k)
        base_acc = pd.Series(y).value_counts(normalize=True).max()
        cnames = [f"L{i}" for i in range(k)]
        for sname, feats in SETS.items():
            X = f[feats].fillna(0.0).values
            res = Parallel(n_jobs=-1)(delayed(clf_cfg)(X, y, g, lv, ml)
                                      for lv in LEAVES for ml in MIN_LEAF)
            res = pd.DataFrame(res, columns=["leaves", "min_leaf", "acc", "kappa"])
            r8 = res[res.leaves == 8].sort_values("acc", ascending=False).iloc[0]
            rb = res.sort_values("acc", ascending=False).iloc[0]
            ph = cross_val_predict(HistGradientBoostingClassifier(random_state=0), X, y,
                                   cv=CV_C, groups=g)
            crow.append(dict(case=case, set=sname, n_feat=len(feats), baseline=round(base_acc, 3),
                             acc_L8=round(r8.acc, 3), kappa_L8=round(r8.kappa, 3),
                             acc_best=round(rb.acc, 3), kappa_best=round(rb.kappa, 3),
                             best_leaves=int(rb.leaves),
                             hgb_acc=round(accuracy_score(y, ph), 3),
                             hgb_kappa=round(cohen_kappa_score(y, ph), 3)))
            m = DecisionTreeClassifier(max_leaf_nodes=8, min_samples_leaf=int(r8.min_leaf),
                                       random_state=RS).fit(X, y)
            for fn, v in zip(feats, m.feature_importances_):
                irow.append(dict(case=case, set=sname, model="clf_L8", feature=fn,
                                 importance_pct=round(100 * v, 1)))
            if sname in ("base", "all"):
                write_clf_tree(m, feats, cnames,
                               os.path.join(OUT, "trees", case.replace("/", "_"), f"{sname}_L8"),
                               f"{case} | {sname} features | 8 leaves | grouped acc {r8.acc:.3f} "
                               f"kappa {r8.kappa:.3f} | baseline {base_acc:.3f}")
            print(crow[-1])

    # ---- regression tree on the clustering features ---------------------------------
    done = set()
    for group, dname, _ in S.CASES:
        d = f"{group}/{dname}"
        if d in done:
            continue
        done.add(d)
        feats_t = dict((n, ft) for n, ft, _ in V.GROUPS[group])[dname]
        Y = StandardScaler().fit_transform(f[feats_t].fillna(0).values.astype(float))
        y3 = S.labels(f, group, dname, 3)
        cent = np.vstack([Y[y3 == c].mean(0) for c in range(3)])
        for sname, feats in SETS.items():
            X = f[feats].fillna(0.0).values
            res = Parallel(n_jobs=-1)(delayed(reg_cfg)(X, Y, g, lv, ml, cent, y3)
                                      for lv in LEAVES for ml in MIN_LEAF)
            res = pd.DataFrame(res, columns=["leaves", "min_leaf", "r2", "acc_k3"])
            r8 = res[res.leaves == 8].sort_values("r2", ascending=False).iloc[0]
            rb = res.sort_values("r2", ascending=False).iloc[0]
            hgb = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=0))
            hr2 = RT.r2(Y, cross_val_predict(hgb, X, Y, cv=CV_R, groups=g))
            rrow.append(dict(definition=d, set=sname, r2_L8=round(r8.r2, 3),
                             acc_k3_L8=round(r8.acc_k3, 3), r2_best=round(rb.r2, 3),
                             acc_k3_best=round(rb.acc_k3, 3), best_leaves=int(rb.leaves),
                             hgb_r2=round(hr2, 3)))
            m = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=int(r8.min_leaf),
                                      random_state=RS).fit(X, Y)
            for fn, v in zip(feats, m.feature_importances_):
                irow.append(dict(case=d, set=sname, model="reg_L8", feature=fn,
                                 importance_pct=round(100 * v, 1)))
            print(rrow[-1])

    cl, rg, im = pd.DataFrame(crow), pd.DataFrame(rrow), pd.DataFrame(irow)
    cl.to_csv(os.path.join(OUT, "classification.csv"), index=False)
    rg.to_csv(os.path.join(OUT, "regression.csv"), index=False)
    im.to_csv(os.path.join(OUT, "importance.csv"), index=False)
    write_html(cl, rg, im)
    print("wrote", OUT)


def pivot_html(df, idx, val, fmt="{:.3f}"):
    p = df.pivot(index=idx, columns="set", values=val)[list(SETS)]
    p.insert(len(SETS), "Δ all−base", p["all"] - p["base"])
    rows = "".join("<tr><td>" + i + "</td>" + "".join(
        f"<td class=num>{('<b>' if c == 'Δ all−base' else '')}{fmt.format(v)}"
        f"{('</b>' if c == 'Δ all−base' else '')}</td>" for c, v in r.items()) + "</tr>"
        for i, r in p.iterrows())
    return ("<table><tr><th></th>" + "".join(f"<th>{c}</th>" for c in p.columns) + "</tr>"
            + rows + "</table>")


def write_html(cl, rg, im):
    top = (im[(im.set == "all")].sort_values("importance_pct", ascending=False)
           .groupby(["case", "model"]).head(3)
           .groupby(["case", "model"]).apply(
               lambda d: ", ".join(f"{a} {b:.0f}%" for a, b in zip(d.feature, d.importance_pct)))
           .reset_index(name="top3"))
    trows = "".join(f"<tr><td>{r.case}</td><td>{r.model}</td><td>{r.top3}</td></tr>"
                    for _, r in top.iterrows())
    links = "".join(
        f"<li>{c}: " + " · ".join(
            f"<a href='trees/{c.replace('/', '_')}/{s}_L8/tree.png'>{s} png</a> "
            f"<a href='trees/{c.replace('/', '_')}/{s}_L8/rules.txt'>rules</a>"
            for s in ("base", "all")) + "</li>" for c in cl.case.unique())
    css = "<style>td.num{text-align:right}</style>"
    html = (
        f"<html><head><meta charset='utf-8'><title>Derived ratios</title>{V.CSS}{css}</head>"
        f"<body><h1>Derived physical ratios as drivers — v5, DT_uncert</h1>"
        f"<div class='note'>All metrics grouped CV by economy. Sets: <b>base</b> = 15 uncertain "
        f"parameters; <b>k_small</b> = base + k_small; <b>supply</b> = base + k_small, k_large, "
        f"g, supply_margin, h2_vs_elec; <b>all</b> = supply + pipe cost proxies "
        f"(pipeline_cost_multiplier × d_SL, d_SS, d_LL). Ratios are computed from the sampled "
        f"parameters only. <a href='../index.html'>accuracy study</a> · "
        f"<a href='../regtree/index.html'>regression tree</a></div>"
        f"<h2>Classification tree on k-Means labels — accuracy, 8 leaves</h2>"
        + pivot_html(cl, "case", "acc_L8") +
        f"<h2>Classification — κ, 8 leaves</h2>" + pivot_html(cl, "case", "kappa_L8") +
        f"<h2>Classification — best accuracy (≤64 leaves)</h2>" + pivot_html(cl, "case", "acc_best") +
        f"<h2>Classification — HGB ceiling accuracy</h2>" + pivot_html(cl, "case", "hgb_acc") +
        f"<h2>Regression tree — R², 8 leaves</h2>" + pivot_html(rg, "definition", "r2_L8") +
        f"<h2>Regression tree — best R² (≤64 leaves)</h2>" + pivot_html(rg, "definition", "r2_best") +
        f"<h2>Regression tree — accuracy at k = 3 (nearest centroid), best tree</h2>"
        + pivot_html(rg, "definition", "acc_k3_best") +
        f"<h2>Regression — HGB ceiling R²</h2>" + pivot_html(rg, "definition", "hgb_r2") +
        f"<h2>Top-3 drivers, 8-leaf tree, set 'all'</h2><table><tr><th>case</th><th>model</th>"
        f"<th>top 3</th></tr>{trows}</table>"
        f"<h2>Trees (8 leaves, base vs all)</h2><ul>{links}</ul>"
        f"<p><a href='classification.csv'>classification.csv</a> · "
        f"<a href='regression.csv'>regression.csv</a> · "
        f"<a href='importance.csv'>importance.csv</a></p></body></html>")
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html)


if __name__ == "__main__":
    main()
