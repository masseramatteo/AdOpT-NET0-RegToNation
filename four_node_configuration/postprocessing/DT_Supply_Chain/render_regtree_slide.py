"""
render_regtree_slide.py
=======================
Slide figures (style of render_tree_slide.py) for the trees of the accuracy study
that have no tree.html:

* regression tree (dt_v5_regtree.py): multi-output DecisionTreeRegressor on the
  clustering features; leaves show a type name, n, the spread (mean std of the
  targets in the leaf), the network and the predicted mix as a stacked bar.
  Leaves with near-identical profiles share a type (dt_v5_regtree.leaf_types).
* --classifier: classification tree on the DT_uncert k-Means labels
  (dt_v5_derived_ratios.py), leaves named as in DT_uncert.

Both refit the tree with the settings of the script that produced it (leaf budget,
best min_samples_leaf on grouped CV); --derived adds the supply ratios
(k_small, k_large, g, supply_margin, h2_vs_elec) to the 15 sampled drivers.

Each run also writes stats.txt: accuracy / balanced accuracy / Cohen's kappa on the k
classes (regression tree: predicted mix -> nearest k-Means centroid), 95% CI by
bootstrap over economies, and R^2 for the regression tree. All grouped CV by economy.

Usage
-----
    python render_regtree_slide.py sysmix sysmix [--leaves 8] [--derived] [--horizontal]
    python render_regtree_slide.py sysmix sysmix --classifier --derived [--k 3]
        [--title "..."] [--names "old name=new name" ...]

Output: DT_uncert_accuracy/regtree[_derived]/trees/<group>_<def>/L<leaves>/ (regression) or
DT_uncert_accuracy/derived/trees/<group>_<def>_k<k>/<base|supply>_L<leaves>/ (classifier):
driver_tree_slide[_horizontal].png/.pdf/.svg + stats.txt.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import config as C
import dt_v5 as V
import dt_v5_accuracy_sweep as SW
import dt_v5_derived_ratios as D
import dt_v5_regtree as RT
import dt_v5_uncertain  # noqa: F401  (updates the renderer labels to the v5 column names)
import render_driver_tree_html as H
import render_tree_slide as S

# labels for the derived supply ratios (dt_v5_derived_ratios.add_ratios)
H.FEAT.update({"k_small": ["Small grid ÷ need", "-"], "k_large": ["Hub grid ÷ need", "-"],
               "g": ["Grid adequacy g", "-"], "supply_margin": ["Supply margin", "-"],
               "h2_vs_elec": ["H₂ import ÷ elec. cost", "-"]})
H.PAIRS.update({"k_small": ["short", "ample"], "k_large": ["short", "ample"], "g": ["short", "ample"],
                "supply_margin": ["tight", "ample"], "h2_vs_elec": ["import cheaper", "import dearer"]})
NOTE_CLF = ("■ hub / ● small cluster (size = electrolyzer) · pipeline solid ≥60% of runs, dashed 30–60%, "
            "width = capacity · yellow arrow = import · bar: local (green) vs import (yellow)")
NOTE_REG = ("■ hub / ● small cluster (size = electrolyzer) · pipeline solid ≥60% of runs, dashed 30–60% · "
            "bar = predicted mix in %: self (green) / transported (blue) / imported (yellow) · "
            "± = spread of the shares in the leaf")
# same colours as the classification tree slide for the shared names
COLORS = {"Import-dominated": "#8c1c2b", "Import-reliant": "#d1495b", "Import + transport": "#b56576",
          "Self + transport": "#00798c", "Self-sufficient": "#edae49", "Mixed supply": "#8d96a3"}
SYSMIX_K3 = {"Low self-served": "Import-reliant", "Medium self-served": "Self + transport",
             "High self-served": "Self-sufficient"}
CV_CLF = StratifiedGroupKFold(5, shuffle=True, random_state=V.RS)


def sysmix_name(m):
    """Automatic type name from a supply-mix profile {self_served, imported, transported}."""
    if m["imported"] >= 0.5:
        return "Import-dominated"
    if m["imported"] >= 0.3 and m["transported"] >= 0.1:
        return "Import + transport"
    if m["imported"] >= 0.3:
        return "Import-reliant"
    if m["transported"] >= 0.15:
        return "Self + transport"
    if m["self_served"] >= 0.8:
        return "Self-sufficient"
    return "Mixed supply"


def load(derived):
    f = V.load(C.EXTRACTED_RESULTS)
    if derived:
        f = D.add_ratios(f)
    raw = pd.read_excel(C.EXTRACTED_RESULTS, sheet_name="Results")
    raw["import_intensity"] = f["imported"].values
    return f, raw, (D.SETS["supply"] if derived else SW.DRIVERS)


def scores(y, p, g, n_boot=1000):
    """acc, balanced acc, kappa + 95% CI of acc and kappa by bootstrap over economies."""
    rng = np.random.default_rng(0)
    idx = {e: np.where(g == e)[0] for e in np.unique(g)}
    ue = np.array(list(idx))
    bs = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[e] for e in rng.choice(ue, len(ue))])
        bs.append((accuracy_score(y[ii], p[ii]), cohen_kappa_score(y[ii], p[ii])))
    bs = np.array(bs)
    q = lambda j: (np.quantile(bs[:, j], .025), np.quantile(bs[:, j], .975))
    return dict(acc=accuracy_score(y, p), bal=balanced_accuracy_score(y, p), kappa=cohen_kappa_score(y, p),
                acc_ci=q(0), kappa_ci=q(1), baseline=np.bincount(y).max() / len(y))


def stats_lines(sc):
    return [f"baseline (majority class)  {sc['baseline']:.3f}",
            f"accuracy {sc['acc']:.3f}  (95% CI {sc['acc_ci'][0]:.3f}–{sc['acc_ci'][1]:.3f})",
            f"balanced accuracy {sc['bal']:.3f}",
            f"Cohen's kappa {sc['kappa']:.3f}  (95% CI {sc['kappa_ci'][0]:.3f}–{sc['kappa_ci'][1]:.3f})"]


def tree_dict(tree, drivers, raw, leaf_id, names, extra=None):
    """Nested split/leaf dict of tree.html; extra(leaf_node_id) -> dict merged into the leaf info."""
    t = tree.tree_

    def node(i):
        if t.children_left[i] == -1:
            info = H._leaf_info(raw[leaf_id == i], names)
            info.update(extra(i) if extra else {})
            return {"type": "leaf", "info": info}
        return {"type": "split", "feature": drivers[t.feature[i]], "thr": H._round(float(t.threshold[i])),
                "left": node(t.children_left[i]), "right": node(t.children_right[i])}

    tr = node(0)
    lv, stack = [], [tr]
    while stack:
        n = stack.pop()
        if n["type"] == "leaf":
            lv.append(n["info"])
        else:
            stack += [n["left"], n["right"]]
    scale = {"pipe_max": max(max(i[a][1] for a in H.ARC_COLS) for i in lv) or 1.0,
             "large_elzr_max": max(max(i["cap_L1"], i["cap_L2"]) for i in lv) or 1.0,
             "small_elzr_max": max(max(i["cap_S1"], i["cap_S2"]) for i in lv) or 1.0,
             "import_max": max(max(i["imp_L1"], i["imp_L2"]) for i in lv) or 1.0}
    return {"tree": tr, "scale": scale}


# ---------------------------------------------------------------- regression tree ---------------------------------
def build_regression(group, dname, leaves, derived, k=3):
    f, raw, drivers = load(derived)
    feats = next(ft for n, ft, _ in V.GROUPS[group] if n == dname)
    X, g = f[drivers].fillna(0.0).values, f.economy.values
    Yraw = f[feats].fillna(0).values.astype(float)
    Y = StandardScaler().fit_transform(Yraw)

    # same choice as dt_v5_regtree: best min_samples_leaf on grouped R^2 at this leaf budget
    best = None
    for ml in RT.MIN_LEAF:
        m = DecisionTreeRegressor(max_leaf_nodes=leaves, min_samples_leaf=ml, random_state=RT.RS)
        P = cross_val_predict(m, X, Y, cv=RT.CV, groups=g)
        r2 = RT.r2(Y, P)
        if best is None or r2 > best[0]:
            best = (r2, ml, P)
    r2g, ml, P = best
    tree = DecisionTreeRegressor(max_leaf_nodes=leaves, min_samples_leaf=ml, random_state=RT.RS).fit(X, Y)
    old = RT.DRIVERS
    RT.DRIVERS = drivers
    prof, ntypes = RT.leaf_types(tree, X, Yraw, feats)
    RT.DRIVERS = old

    # read as a k-class prediction: predicted mix -> nearest k-Means centroid
    y = SW.labels(f, group, dname, k)
    cent = np.vstack([Y[y == c].mean(0) for c in range(k)])
    p = ((P[:, None, :] - cent[None]) ** 2).sum(2).argmin(1)
    sc = scores(y, p, g)

    leaf_id = tree.apply(X)
    raw = raw.copy()
    raw["cluster_final"] = [prof.type[i] for i in leaf_id]
    names = {}
    for tp, grp in prof.groupby("type"):
        w = grp["n"] / grp["n"].sum()
        mix = {c: float((grp[f"{c}_mean"] * w).sum()) for c in feats}
        names[tp] = sysmix_name(mix) if (group, dname) == ("sysmix", "sysmix") else f"Type {tp}"
    # types with the same name share colour and legend entry (each leaf still shows its own mix)
    prof["name"] = prof.type.map(names)
    order = [c for c in ("self_served", "transported", "imported") if c in feats]

    def extra(i):
        r = prof.loc[i]
        out = {"spread": float(np.mean([r[f"{c}_std"] for c in feats])), "type": int(r.type)}
        if len(order) == 3:
            out["mix"] = [round(float(r[f"{c}_mean"]), 3) for c in order]
        return out

    data = tree_dict(tree, drivers, raw, leaf_id, names, extra)
    legend = {}
    pooled = {nm: grp for nm, grp in prof.groupby("name")}
    for nm in sorted(pooled, key=lambda nm: -(pooled[nm]["self_served_mean"] * pooled[nm]["n"]).sum()
                     / pooled[nm]["n"].sum() if "self_served" in feats else nm):
        grp = pooled[nm]
        w = grp["n"] / grp["n"].sum()
        legend[nm] = " / ".join(f"{S.SHORT.get(c, c)} {100 * (grp[f'{c}_mean'] * w).sum():.0f}"
                                for c in (order or feats)) + " %"
    ntypes = len(legend)
    out_dir = os.path.join(SW.OUT, "regtree_derived" if derived else "regtree", "trees", f"{group}_{dname}",
                           f"L{leaves}")
    drv = "sampled parameters + supply ratios" if derived else "sampled parameters"
    sub = (f"{leaves}-leaf regression tree · {drv} · {ntypes} types · grouped R² {r2g:.2f} · "
           f"accuracy on k={k} classes {sc['acc']:.0%} · κ {sc['kappa']:.2f} (baseline {sc['baseline']:.0%})")
    lines = [f"{group}/{dname} regression tree, {leaves} leaves, min_samples_leaf {ml}, drivers: {', '.join(drivers)}",
             f"targets: {', '.join(feats)} · {len(f)} runs, {f.economy.nunique()} economies · grouped CV by economy",
             f"grouped R2 {r2g:.3f}", f"read as k={k} classes (predicted mix -> nearest k-Means centroid):"]
    lines += stats_lines(sc)
    return data, out_dir, sub, legend, lines, NOTE_REG, {}


# ---------------------------------------------------------------- classification tree -----------------------------
def build_classifier(group, dname, leaves, derived, k=3):
    f, raw, drivers = load(derived)
    X, g = f[drivers].fillna(0.0).values, f.economy.values
    y = SW.labels(f, group, dname, k)

    # same choice as dt_v5_derived_ratios: best min_samples_leaf on grouped accuracy at this leaf budget
    best = None
    for ml in D.MIN_LEAF:
        m = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=ml, random_state=V.RS)
        p = cross_val_predict(m, X, y, cv=CV_CLF, groups=g)
        acc = accuracy_score(y, p)
        if best is None or acc > best[0]:
            best = (acc, ml, p)
    _, ml, p = best
    sc = scores(y, p, g)
    tree = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=ml, random_state=V.RS).fit(X, y)
    train_acc = accuracy_score(y, tree.predict(X))

    case_dir = os.path.join(os.path.dirname(C.EXTRACTED_RESULTS), "DT_uncert", group, dname, "all", f"k{k:02d}")
    lv = pd.read_csv(os.path.join(case_dir, "cluster_levels.csv"))
    names = dict(zip(lv.level, lv.name))
    feats = next(ft for n, ft, _ in V.GROUPS[group] if n == dname)
    legend = S.profiles(case_dir, feats)

    raw = raw.copy()
    raw["cluster_final"] = y
    data = tree_dict(tree, drivers, raw, tree.apply(X), names)
    out_dir = os.path.join(SW.OUT, "derived", "trees", f"{group}_{dname}_k{k:02d}",
                           f"{'supply' if derived else 'base'}_L{leaves}")
    drv = "sampled parameters + supply ratios" if derived else "sampled parameters"
    sub = (f"{leaves}-leaf decision tree · {drv} · {len(f)} runs, {f.economy.nunique()} economies · "
           f"accuracy {sc['acc']:.0%} (CI {sc['acc_ci'][0]:.0%}–{sc['acc_ci'][1]:.0%}) · κ {sc['kappa']:.2f} "
           f"(baseline {sc['baseline']:.0%})")
    imp = pd.Series(tree.feature_importances_ * 100, index=drivers).sort_values(ascending=False)
    lines = [f"{group}/{dname} k={k} classification tree, {leaves} leaves, min_samples_leaf {ml}",
             f"drivers: {', '.join(drivers)}", f"{len(f)} runs, {f.economy.nunique()} economies · grouped CV by economy"]
    lines += stats_lines(sc) + [f"train accuracy {train_acc:.3f}", "", "importance (%):",
                                imp[imp > 0].round(1).to_string()]
    return data, out_dir, sub, legend, lines, NOTE_CLF, {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group")
    ap.add_argument("definition")
    ap.add_argument("--leaves", type=int, default=8)
    ap.add_argument("--k", type=int, default=3, help="k-Means classes (labels / accuracy reference)")
    ap.add_argument("--classifier", action="store_true", help="classification tree instead of regression tree")
    ap.add_argument("--derived", action="store_true", help="sampled drivers + supply ratios")
    ap.add_argument("--horizontal", action="store_true")
    ap.add_argument("--title", default=None)
    ap.add_argument("--names", nargs="*", default=[], help='"old name=new name" pairs')
    a = ap.parse_args()
    build = build_classifier if a.classifier else build_regression
    data, out_dir, sub, legend, lines, note, _ = build(a.group, a.definition, a.leaves, a.derived, a.k)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "stats.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    ren = dict(r.split("=", 1) for r in a.names)
    if (a.group, a.definition, a.k) == ("sysmix", "sysmix", 3):
        ren = {**SYSMIX_K3, **ren}
    out = os.path.join(out_dir, "driver_tree_slide" + ("_horizontal" if a.horizontal else ""))
    title = a.title or f"{a.group}/{a.definition} — {'decision' if a.classifier else 'regression'} tree"
    print("saved", S.render_data(data, H.FEAT, H.PAIRS, out, title, sub, legend, ren, a.horizontal, note=note,
                                 colors={**COLORS, **{k_: COLORS[v] for k_, v in SYSMIX_K3.items()}}))
