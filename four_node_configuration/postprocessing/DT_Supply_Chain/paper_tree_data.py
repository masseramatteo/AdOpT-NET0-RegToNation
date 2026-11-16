"""
paper_tree_data.py
==================
Step 1 of the paper decision-tree figures: refit one tree of the accuracy study and
export everything the figures need to a single JSON file. The drawing script
(paper_tree_figures.py) reads only that JSON, so figures can be restyled without the
network drive or a refit.

The tree is refit exactly as in render_regtree_slide.py (same drivers, labels, leaf
budget, best min_samples_leaf on grouped CV), so splits and leaves are the ones of the
slide figures in DT_uncert_accuracy/.

    classifier (default)  DecisionTreeClassifier on the DT_uncert k-Means labels
    --regression          multi-output DecisionTreeRegressor on the clustering features;
                          leaves with near-identical profiles share a type name
    --derived             sampled drivers + supply ratios (k_small, k_large, g, ...)

Usage
-----
    python paper_tree_data.py sysmix sysmix --derived               # the paper example
    python paper_tree_data.py sysmix sysmix --regression
    python paper_tree_data.py topo meshsat --derived --leaves 8 --k 3

Output: <PAPER_DIR>/_data/<case>.json, case = <group>_<def>_<clf_k03|reg>_<supply|base>_L<leaves>

JSON content
------------
    meta      case description, counts, drivers
    classes   [{name, color, n, mix}]            class order = low -> high self-served
    tree      nested nodes; every node: id, n, counts (per class), mix [self, transp, imp];
              splits: feature, thr, left, right; leaves: info (network of the leaf, purity, ...)
    scale     normalisation of the network glyphs (max capacities over the leaves)
    features  {name: {label, unit, lo_word, hi_word, min, max, q01, q99, importance}}
    stats     grouped-CV accuracy, balanced accuracy, kappa, CIs, baseline, confusion matrix
    points    per run: values of the features used by the tree, class, leaf id
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import config as C
import dt_v5 as V
import dt_v5_accuracy_sweep as SW
import dt_v5_derived_ratios as D
import dt_v5_regtree as RT
import render_driver_tree_html as H
import render_regtree_slide as RR  # also registers the labels of the derived ratios in H.FEAT / H.PAIRS

PAPER_DIR = (r"C:\Users\Masse007\OneDrive - Universiteit Utrecht\Documents\Research\FirstWork"
             r"\Images\Results\Decision_trees")
MIX = ["self_served", "transported", "imported"]  # order of the mix bars everywhere
NODES = {"L1": "Large_cluster1", "L2": "Large_cluster2", "S1": "Small_cluster1", "S2": "Small_cluster2"}


def fit_classifier(f, drivers, group, dname, leaves, k):
    X, g = f[drivers].fillna(0.0).values, f.economy.values
    y = SW.labels(f, group, dname, k)
    best = None
    for ml in D.MIN_LEAF:
        m = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=ml, random_state=V.RS)
        p = cross_val_predict(m, X, y, cv=RR.CV_CLF, groups=g)
        acc = accuracy_score(y, p)
        if best is None or acc > best[0]:
            best = (acc, ml, p)
    _, ml, p = best
    tree = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=ml, random_state=V.RS).fit(X, y)

    case_dir = os.path.join(C.RESULTS_FOLDER, "DT_uncert", group, dname, "all", f"k{k:02d}")
    lv = pd.read_csv(os.path.join(case_dir, "cluster_levels.csv"))
    names = [dict(zip(lv.level, lv.name))[i] for i in range(k)]
    if (group, dname, k) == ("sysmix", "sysmix", 3):
        names = [RR.SYSMIX_K3.get(n, n) for n in names]
    sc = RR.scores(y, p, g)
    stats = {"kind": "classification", "min_samples_leaf": ml, "train_accuracy": accuracy_score(y, tree.predict(X)),
             "confusion": confusion_matrix(y, p).tolist()}
    return tree, X, y, names, sc, stats


def fit_regression(f, drivers, group, dname, leaves, k):
    feats = next(ft for n, ft, _ in V.GROUPS[group] if n == dname)
    X, g = f[drivers].fillna(0.0).values, f.economy.values
    Yraw = f[feats].fillna(0).values.astype(float)
    Y = StandardScaler().fit_transform(Yraw)
    best = None
    for ml in RT.MIN_LEAF:
        m = DecisionTreeRegressor(max_leaf_nodes=leaves, min_samples_leaf=ml, random_state=RT.RS)
        P = cross_val_predict(m, X, Y, cv=RT.CV, groups=g)
        r2 = RT.r2(Y, P)
        if best is None or r2 > best[0]:
            best = (r2, ml, P)
    r2g, ml, P = best
    tree = DecisionTreeRegressor(max_leaf_nodes=leaves, min_samples_leaf=ml, random_state=RT.RS).fit(X, Y)
    prof, _ = RT.leaf_types(tree, X, Yraw, feats)

    # type name per leaf (leaves of the same type share it); classes = distinct names
    tname = {}
    for tp, grp in prof.groupby("type"):
        w = grp["n"] / grp["n"].sum()
        mix = {c: float((grp[f"{c}_mean"] * w).sum()) for c in feats}
        tname[tp] = RR.sysmix_name(mix) if (group, dname) == ("sysmix", "sysmix") else f"Type {tp}"
    leaf_name = prof.type.map(tname)
    leaf_id = tree.apply(X)
    run_name = leaf_name.reindex(leaf_id).values
    names = sorted(set(run_name), key=lambda nm: f.loc[run_name == nm, feats[0]].mean())
    y = np.array([names.index(nm) for nm in run_name])

    # read as a k-class prediction: predicted mix -> nearest k-Means centroid
    yk = SW.labels(f, group, dname, k)
    cent = np.vstack([Y[yk == c].mean(0) for c in range(k)])
    pk = ((P[:, None, :] - cent[None]) ** 2).sum(2).argmin(1)
    sc = RR.scores(yk, pk, g)
    spread = {int(i): float(np.mean([prof.loc[i, f"{c}_std"] for c in feats])) for i in prof.index}
    stats = {"kind": "regression", "min_samples_leaf": ml, "r2_grouped": float(r2g), "k_reference": k,
             "leaf_spread": spread}
    return tree, X, y, names, sc, stats


def export(group, dname, leaves=8, k=3, derived=False, regression=False, out_dir=None):
    f, raw, drivers = RR.load(derived)
    fit = fit_regression if regression else fit_classifier
    tree, X, y, names, sc, stats = fit(f, drivers, group, dname, leaves, k)
    t = tree.tree_
    leaf_id = tree.apply(X)
    path = tree.decision_path(X).toarray().astype(bool)  # runs x nodes
    mixcols = [c for c in MIX if c in f.columns]
    M = f[mixcols].fillna(0).values
    raw = raw.copy()
    raw["cluster_final"] = y
    cname = dict(enumerate(names))
    # demand of each cluster from the sampled parameters (large : small = demand_level_ratio, L1 : L2 =
    # unbalance_ratio, S1 = S2); agrees with the hydrogen balance of every cluster within a few %
    T, lev, unb = (raw[c].values for c in ("total_demand_TWh", "demand_level_ratio", "unbalance_ratio"))
    large = T * lev / (1 + lev)
    demand = {"L1": large * unb / (1 + unb), "L2": large / (1 + unb), "S1": T / (1 + lev) / 2, "S2": T / (1 + lev) / 2}
    production = {k: raw[f"{n}_TOTAL_H2_production"].fillna(0).values / 1e6 for k, n in NODES.items()}

    def node(i):
        rows = path[:, i]
        out = {"id": int(i), "n": int(rows.sum()),
               "counts": np.bincount(y[rows], minlength=len(names)).tolist(),
               "mix": [round(float(v), 4) for v in M[rows].mean(0)]}
        if t.children_left[i] == -1:
            info = H._leaf_info(raw[leaf_id == i], cname)
            for key in NODES:  # mean demand and local production of each cluster, TWh/y
                info[f"dem_{key}"] = round(float(demand[key][rows].mean()), 4)
                info[f"prod_{key}"] = round(float(production[key][rows].mean()), 4)
            if regression:
                info["spread"] = round(stats["leaf_spread"][int(i)], 3)
            out.update(type="leaf", info=info)
        else:
            out.update(type="split", feature=drivers[t.feature[i]], thr=H._round(float(t.threshold[i])),
                       left=node(t.children_left[i]), right=node(t.children_right[i]))
        return out

    root = node(0)
    infos, stack = [], [root]
    while stack:
        n = stack.pop()
        if n["type"] == "leaf":
            infos.append(n["info"])
        else:
            stack += [n["left"], n["right"]]
    scale = {"pipe_max": max(max(i[a][1] for a in H.ARC_COLS) for i in infos) or 1.0,
             "large_elzr_max": max(max(i["cap_L1"], i["cap_L2"]) for i in infos) or 1.0,
             "small_elzr_max": max(max(i["cap_S1"], i["cap_S2"]) for i in infos) or 1.0,
             "import_max": max(max(i["imp_L1"], i["imp_L2"]) for i in infos) or 1.0}

    used = [drivers[j] for j in sorted(set(t.feature[t.feature >= 0]))]
    imp = dict(zip(drivers, tree.feature_importances_))
    features = {}
    for d in used:
        v = f[d].astype(float)
        lab = H.FEAT.get(d, [d, ""])
        words = H.PAIRS.get(d, ["", ""])
        features[d] = {"label": lab[0], "unit": "" if lab[1] in ("-", "ratio") else lab[1],
                       "lo_word": words[0], "hi_word": words[1],
                       "min": float(v.min()), "max": float(v.max()),
                       "q01": float(v.quantile(.01)), "q99": float(v.quantile(.99)),
                       "importance": round(float(imp[d]), 4)}

    colors = {**RR.COLORS, **{a: RR.COLORS[b] for a, b in RR.SYSMIX_K3.items()}}
    classes = [{"name": nm, "color": colors.get(nm), "n": int((y == i).sum()),
                "mix": [round(float(v), 4) for v in M[y == i].mean(0)]} for i, nm in enumerate(names)]

    case = (f"{group}_{dname}_{'reg' if regression else f'clf_k{k:02d}'}_{'supply' if derived else 'base'}"
            f"_L{leaves}")
    stats.pop("leaf_spread", None)
    stats.update(accuracy=sc["acc"], accuracy_ci=list(sc["acc_ci"]), balanced_accuracy=sc["bal"],
                 kappa=sc["kappa"], kappa_ci=list(sc["kappa_ci"]), baseline=sc["baseline"])
    data = {"meta": {"case": case, "group": group, "definition": dname, "leaves": leaves, "k": k,
                     "derived": derived, "regression": regression, "runs": int(len(f)),
                     "economies": int(f.economy.nunique()), "drivers": drivers, "mix_order": mixcols,
                     "source": C.EXTRACTED_RESULTS},
            "classes": classes, "tree": root, "scale": scale, "features": features, "stats": stats,
            "points": {"class": y.tolist(), "leaf": leaf_id.tolist(),
                       **{d: np.round(f[d].astype(float).values, 4).tolist() for d in used}}}

    out_dir = out_dir or os.path.join(PAPER_DIR, "_data")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, case + ".json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, default=float)
    return out, data


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group")
    ap.add_argument("definition")
    ap.add_argument("--leaves", type=int, default=8)
    ap.add_argument("--k", type=int, default=3, help="k-Means classes (labels / accuracy reference)")
    ap.add_argument("--regression", action="store_true", help="regression tree instead of classification tree")
    ap.add_argument("--derived", action="store_true", help="sampled drivers + supply ratios")
    ap.add_argument("--out", default=None, help="folder for the JSON (default <PAPER_DIR>/_data)")
    a = ap.parse_args()
    out, data = export(a.group, a.definition, a.leaves, a.k, a.derived, a.regression, a.out)
    s = data["stats"]
    print(f"accuracy {s['accuracy']:.3f}  balanced {s['balanced_accuracy']:.3f}  kappa {s['kappa']:.3f}  "
          f"baseline {s['baseline']:.3f}")
    print("saved", out)
