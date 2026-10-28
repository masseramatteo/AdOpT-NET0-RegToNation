"""
v5 DT_supplychain, uncertain-parameter edition -- POOLED + ONE TREE PER ARCHETYPE.

Implements section 6 of the methodology note
"2026-09-20 Treating of economies and topologies in analysis":
a stratified rebuild of the best DT_uncert cases as five models each --
one pooled tree over all 5992 runs and one tree per spatial archetype
(A, B, C, D; 1498 runs / 749 economies each) -- plus the pre-registered
cross-archetype comparison of section 6.3.

NOTHING of the earlier run is modified. `dt_v5.py` and `dt_v5_uncertain.py`
are imported and reused unchanged; the output goes to a new folder
`DT_uncert_archetype/` next to the existing `DT_uncert/`.

What is held fixed so the five models are comparable
----------------------------------------------------
1. The k-Means typology is fitted ONCE on the pooled 5992 runs and its labels
   are reused by all five models. A per-archetype clustering would give each
   archetype a different target and no comparison would be possible.
2. The economy-grouped CV folds are built ONCE on the pooled data
   (StratifiedGroupKFold(5), groups = economy, same seed as dt_v5) and each
   archetype model is evaluated on the RESTRICTION of those folds to its own
   rows. The pooled model is scored on an archetype by subsetting its
   out-of-fold predictions to that archetype's rows -- identical test rows,
   identical folds, so the kappa difference is a model effect only.
3. The tuning grid (max_leaf_nodes x min_samples_leaf) and its tie-break are
   the ones of `dt_v5.tune`, used unchanged for all five models, so a kappa
   difference is not a tree-size artefact. The size confound is additionally
   removed by re-fitting each archetype at the POOLED model's (leaves,
   min_leaf) -- reported as `kappa_fixed_size`.
4. Cluster names come from the pooled profile, so level i means the same thing
   in all five folders.

Because of (1)-(3) the `pooled/` folder reproduces the numbers already in
`DT_uncert/` for the same (group, definition, k) case.

Comparison, pre-registered in section 6.3 (decided before looking)
------------------------------------------------------------------
  measure 1  does stratifying buy accuracy?  d_kappa = kappa(archetype model)
             - kappa(pooled model on the same rows), with a cluster bootstrap
             over economies.  Interval covering zero => pooled tree is the
             right headline.
  measure 2  do the drivers differ?  Spearman rank correlation between the
             importance vectors (6 archetype pairs + 4 vs pooled), bootstrapped
             over economies, plus the root split and top-3 of every model.
  measure 3  formal interaction test: the pooled tree refitted WITH the
             archetype as a one-hot feature -- where it is used and what it is
             worth in kappa.  Reported next to the A/B twin-disagreement floor
             (section 4.5): the 1498 A/B twins share all 15 numeric axes, so
             their disagreement rate on the target is a lower bound on the
             error of any model that does not see the archetype.

Layout
------
    <model>/<group>/<definition>/all/k<kk>/{cluster_profiles.csv,
        cluster_levels.csv, feature_importance.csv, rules.txt, driver_tree.png,
        tree.html, details.html}                    model in pooled, A, B, C, D
    comparison/{delta_kappa.csv, importance_rank_corr.csv, top_splits.csv,
        archetype_feature_test.csv, delta_kappa_heatmap.png,
        importance_corr_heatmap.png, comparison.html}
    comparison_all.csv / overview.html

Run:  python dt_v5_uncertain_archetype.py
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, silhouette_score)
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

import config as C
import dt_v5 as V
import dt_v5_uncertain as U
import render_driver_tree_html as R

warnings.filterwarnings("ignore")

RS = V.RS
DRIVERS = U.DRIVERS          # sampled uncertain axes only (k_small dropped)
DIST = V.DIST
OUT = os.path.join(C.RESULTS_FOLDER, "DT_uncert_archetype")

ARCHS = ["A", "B", "C", "D"]
MODELS = ["pooled"] + ARCHS

# The cases rebuilt here: the 7 PASS cases of the DT_uncert run of 2026-09-17
# (grouped kappa >= 0.60 and baseline <= 0.80), plus the best case of the two
# groups that PASS does not reach, so all four approach groups are covered.
# (group, definition, k, kappa of the pooled DT_uncert run -- for the record)
CASES = [
    ("sysmix", "cent_imp", 3, 0.685),
    ("sysmix", "sysmix", 3, 0.680),
    ("sysmix", "sysmix", 4, 0.650),
    ("combo", "mix_topo", 3, 0.644),
    ("sysmix", "cent_imp", 4, 0.632),
    ("combo", "supply_struct", 3, 0.613),
    ("sysmix", "selfserved", 3, 0.609),
    ("decentr", "B_plus_mix", 4, 0.590),   # best decentr case
    ("topo", "backbone", 3, 0.511),        # best topo case
]

N_BOOT_KAPPA = 500      # cluster bootstrap over economies for d_kappa
N_BOOT_IMP = 100        # cluster bootstrap over economies for importance ranks

DEFS = {(gn, dn): (ft, ds) for gn, defs in V.GROUPS.items() for dn, ft, ds in defs}


# ------------------------------------------------------------------ folds --
def make_folds(X, y, g):
    """The economy-grouped folds of dt_v5, materialised once so every model can
    be scored on exactly the same test rows."""
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=RS)
    return [(tr, te) for tr, te in cv.split(X, y, groups=g)]


def oof(clf, X, y, folds, m):
    """Out-of-fold predictions on the pooled folds restricted to mask `m`.
    Rows outside `m` keep -1: the model never saw them and is never scored on
    them."""
    pred = np.full(len(y), -1, dtype=int)
    for tr, te in folds:
        tr_m, te_m = tr[m[tr]], te[m[te]]
        if len(tr_m) == 0 or len(te_m) == 0:
            continue
        c = clone(clf).fit(X.iloc[tr_m], y[tr_m])
        pred[te_m] = c.predict(X.iloc[te_m])
    return pred


def score(y, pred, m):
    yy, pp = y[m], pred[m]
    return (round(accuracy_score(yy, pp), 3),
            round(balanced_accuracy_score(yy, pp), 3),
            round(cohen_kappa_score(yy, pp), 3))


def tune(X, y, folds, m, k):
    """dt_v5.tune, on the fixed folds and restricted to `m`.

    Same grid and same tie-break (fewest leaves among candidates within 0.01 of
    the best), so the five models are tuned by an identical procedure."""
    cand = []
    for lv in [k, k + 2, 8, 12, 16, 20]:
        if lv < k:
            continue
        for ml in [10, 20, 40]:
            a, b, kp = score(y, oof(V.mk_tree(lv, ml), X, y, folds, m), m)
            cand.append((kp, lv, ml, a, b))
    top = max(c[0] for c in cand)
    good = [c for c in cand if c[0] >= top - 0.01]
    good.sort(key=lambda c: (c[1], -c[2]))
    return good[0]


def ungrouped_kappa(clf, X, y, m):
    """The same model scored WITHOUT economy grouping, kept beside the grouped
    value exactly as dt_v5 does, to keep the clustering leak visible."""
    idx = np.where(m)[0]
    p = cross_val_predict(clf, X.iloc[idx], y[idx],
                          cv=StratifiedKFold(5, shuffle=True, random_state=RS))
    return round(cohen_kappa_score(y[idx], p), 3)


# ------------------------------------------------------------- bootstraps --
def econ_positions(econ, m):
    """economy -> array of row positions of that economy inside mask `m`."""
    idx = np.where(m)[0]
    out = {}
    for p in idx:
        out.setdefault(econ[p], []).append(p)
    return {e: np.asarray(v) for e, v in out.items()}


def boot_delta_kappa(y, pred_a, pred_p, pos, econs, nboot=N_BOOT_KAPPA):
    """Cluster bootstrap over economies of kappa(archetype) - kappa(pooled),
    both evaluated on the same resampled rows of one archetype."""
    rng = np.random.default_rng(RS)
    ka, kp, dk = [], [], []
    for _ in range(nboot):
        samp = rng.choice(econs, size=len(econs), replace=True)
        idx = np.concatenate([pos[e] for e in samp])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        a = cohen_kappa_score(yy, pred_a[idx])
        p = cohen_kappa_score(yy, pred_p[idx])
        ka.append(a)
        kp.append(p)
        dk.append(a - p)

    def q(v, x):
        return float(np.percentile(v, x)) if v else np.nan

    return dict(d_lo=q(dk, 2.5), d_hi=q(dk, 97.5),
                ka_lo=q(ka, 2.5), ka_hi=q(ka, 97.5),
                kp_lo=q(kp, 2.5), kp_hi=q(kp, 97.5))


def boot_importances(X, y, econ, masks, params, nboot=N_BOOT_IMP):
    """Economy-resampled importance vectors, the SAME economy sample used for
    all five models in a replicate so the rank correlations are paired."""
    rng = np.random.default_rng(RS)
    pos = {mo: econ_positions(econ, masks[mo]) for mo in MODELS}
    econs = np.unique(econ)
    out = {mo: [] for mo in MODELS}
    for _ in range(nboot):
        samp = rng.choice(econs, size=len(econs), replace=True)
        for mo in MODELS:
            pm = pos[mo]
            idx = np.concatenate([pm[e] for e in samp if e in pm])
            lv, ml = params[mo]
            t = V.mk_tree(lv, ml).fit(X.iloc[idx], y[idx])
            out[mo].append(t.feature_importances_)
    return {mo: np.asarray(v) for mo, v in out.items()}


# ------------------------------------------------------------- tree facts --
def root_split(t):
    """(feature, threshold) of the root split of a fitted tree."""
    if t.tree_.feature[0] < 0:
        return "(no split)", np.nan
    return DRIVERS[t.tree_.feature[0]], float(t.tree_.threshold[0])


def first_depth_of(t, names, wanted):
    """Shallowest depth at which any feature in `wanted` is tested. inf if never."""
    best = [np.inf]

    def walk(i, d):
        if t.tree_.children_left[i] == -1:
            return
        if names[t.tree_.feature[i]] in wanted:
            best[0] = min(best[0], d)
        walk(t.tree_.children_left[i], d + 1)
        walk(t.tree_.children_right[i], d + 1)

    walk(0, 0)
    return best[0]


# --------------------------------------------------------------- one case --
def profile_and_range(f, y, m, k, axis_col, prof_pooled, rng_pooled):
    """Per-model cluster profile and axis range; falls back to the pooled value
    for a cluster that is empty inside this archetype."""
    sub = f[m]
    ysub = y[m]
    prof = sub.groupby(ysub)[V.PROFILE].mean().reindex(range(k))
    prof = prof.fillna(prof_pooled)
    prof["n"] = pd.Series(ysub).value_counts().reindex(range(k)).fillna(0).astype(int).values
    av = sub[axis_col].values
    rng = []
    for i in range(k):
        v = av[ysub == i]
        rng.append((v.min(), v.max(), v.mean()) if len(v) else rng_pooled[i])
    return prof, rng


def write_case_artifacts(kdir, group, dname, k, feats, desc, model, f, dfm, X, y, m,
                         dt, rf, lv, ml, acc, bal, kp, ung, rfk, base, sil,
                         cnames, rtxt, subs, hkp):
    """The exact artefact set of dt_v5_uncertain, written for one model."""
    os.makedirs(kdir, exist_ok=True)
    axis_col = U.AXIS[(group, dname)][0]
    sub = f[m]
    ysub = y[m]

    prof = sub.groupby(ysub)[V.PROFILE].mean().reindex(range(k))
    prof["n"] = pd.Series(ysub).value_counts().reindex(range(k)).fillna(0).astype(int).values
    prof.insert(0, "name", cnames)
    prof.round(3).to_csv(os.path.join(kdir, "cluster_profiles.csv"))

    av = sub[axis_col].values
    lo = [av[ysub == i].min() if (ysub == i).any() else np.nan for i in range(k)]
    hi = [av[ysub == i].max() if (ysub == i).any() else np.nan for i in range(k)]
    mu = [av[ysub == i].mean() if (ysub == i).any() else np.nan for i in range(k)]
    pd.DataFrame({"level": range(k), "name": cnames, "axis": axis_col,
                  "range_lo": lo, "range_hi": hi, "mean": mu,
                  "n": [int((ysub == i).sum()) for i in range(k)]}
                 ).round(4).to_csv(os.path.join(kdir, "cluster_levels.csv"), index=False)

    imp = pd.Series(dt.feature_importances_ * 100, index=DRIVERS)
    rimp = pd.Series(rf.feature_importances_ * 100, index=DRIVERS)
    pd.DataFrame({"feature": DRIVERS, "DT_importance_pct": imp.round(1).values,
                  "RF_importance_pct": rimp.round(1).values}
                 ).sort_values("DT_importance_pct", ascending=False).to_csv(
        os.path.join(kdir, "feature_importance.csv"), index=False)

    rnames = [f"{n} ({r})" for n, r in zip(cnames, rtxt)]
    rules = V.leaf_rules(dt, DRIVERS, rnames)
    with open(os.path.join(kdir, "rules.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# {model} | {group}/{dname} k={k} | grouped kappa {kp:.3f} "
                 f"| acc {acc:.3f} | baseline {base:.3f} | n={int(m.sum())}\n")
        fh.write(f"# clustered on: {', '.join(feats)} (k-Means fitted on the pooled "
                 f"campaign, labels shared by all five models)\n")
        fh.write(f"# drivers: uncertain sampled parameters only (no k_small)\n")
        fh.write(f"# folds: economy-grouped, pooled folds restricted to this model's rows\n\n")
        for r in rules:
            mark = "" if r["purity"] >= 0.70 else "  [mixed]"
            fh.write(f"[{r['purity']:.0%}, n={r['n']}] {r['cls']}{mark}\n    IF {r['rule']}\n")

    fig, ax = plt.subplots(figsize=(17, 9))
    plot_tree(dt, feature_names=DRIVERS, class_names=rnames, filled=True, rounded=True,
              proportion=True, precision=1, fontsize=7, ax=ax)
    ax.set_title(f"[{model}] {group}/{dname}  k={k}   n={int(m.sum())} | baseline {base:.2f} | "
                 f"acc {acc:.2f} | grouped kappa {kp:.2f} | RF kappa {rfk:.2f}",
                 fontsize=12, color=C.UU_NAVY)
    fig.savefig(os.path.join(kdir, "driver_tree.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    n_ec = int(pd.Series(f.economy.values[m]).nunique())
    sub_text = (f"{int(m.sum())} optimizations, {n_ec} economies "
                f"({'all archetypes' if model == 'pooled' else 'archetype ' + model}). "
                f"{U.HTML_LEAVES}-leaf tree, grouped &kappa; {hkp:.2f} (baseline {base:.2f}). "
                f"Clustered on {', '.join(feats)}.")
    U.schematic_html(dfm[m], y[m], cnames, subs, os.path.join(kdir, "tree.html"),
                     f"{model} &mdash; {group} / {dname}, k={k}: from uncertain parameters "
                     f"to hydrogen infrastructure", sub_text)

    top3 = imp.sort_values(ascending=False).head(3)
    with open(os.path.join(kdir, "details.html"), "w", encoding="utf-8") as fh:
        fh.write(f"<html><head><meta charset='utf-8'><title>{model} {group}/{dname} k{k}</title>"
                 f"{V.CSS}</head><body><h1>{model} &mdash; {group} / {dname}, k={k}</h1>"
                 f"<div class='note'>{desc}<br>Clustered on: <b>{', '.join(feats)}</b> "
                 f"(k-Means fitted once on the pooled campaign; labels shared by all five "
                 f"models).<br>Rows: <b>{int(m.sum())}</b> runs / {n_ec} economies. "
                 f"Drivers: uncertain sampled parameters only (k_small dropped).<br>"
                 f"grouped &kappa; <b>{kp:.2f}</b> {V.verdict(kp)} &middot; accuracy {acc:.2f} "
                 f"&middot; balanced {bal:.2f} &middot; baseline {base:.2f} &middot; "
                 f"RF &kappa; {rfk:.2f} &middot; ungrouped &kappa; {ung:.2f} &middot; "
                 f"silhouette {sil:.2f} &middot; tree {lv} leaves / min_leaf {ml}<br>"
                 f"<a href='tree.html'>schematic driver tree</a> ({U.HTML_LEAVES} leaves, "
                 f"grouped &kappa; {hkp:.2f})</div>"
                 f"<h2>Cluster profiles</h2>{prof.round(3).to_html()}"
                 f"<h2>Drivers</h2><p>" +
                 ", ".join(f"{a} {b:.0f}%" for a, b in top3.items()) +
                 f"</p><h2>Leaf rules</h2><pre>" +
                 "\n".join(f"[{r['purity']:.0%}, n={r['n']}] {r['cls']}\n    IF {r['rule']}"
                           for r in rules) +
                 f"</pre><h2>Tree</h2><img src='driver_tree.png' style='width:100%'>"
                 f"</body></html>")
    return imp, rimp


# -------------------------------------------------------------------- main --
def main():
    f = V.load(C.EXTRACTED_RESULTS).reset_index(drop=True)
    raw = pd.read_excel(C.EXTRACTED_RESULTS, sheet_name="Results").reset_index(drop=True)
    dfm = raw.copy()
    dfm["import_intensity"] = f["imported"].values
    X = f[DRIVERS].fillna(0.0)
    econ = f.economy.values
    arch = f.archetype.values
    masks = {"pooled": np.ones(len(f), bool)}
    for a in ARCHS:
        masks[a] = (arch == a)
    os.makedirs(OUT, exist_ok=True)
    print(f"{len(f)} runs, {f.economy.nunique()} economies, {len(DRIVERS)} uncertain drivers")
    print("rows per model:", {mo: int(masks[mo].sum()) for mo in MODELS})
    print(f"{len(CASES)} cases x {len(MODELS)} models = {len(CASES) * len(MODELS)} trees\n")

    # A/B twins: same economy, same distance triple, only the hub attachment differs.
    twins = []
    for tag in ("near", "far"):
        sel = f.slot.str.endswith("_" + tag)
        ia = f.index[sel & (f.archetype == "A")]
        ib = f.index[sel & (f.archetype == "B")]
        ea = pd.Series(np.arange(len(ia)), index=f.economy.values[ia])
        eb = pd.Series(np.arange(len(ib)), index=f.economy.values[ib])
        common = ea.index.intersection(eb.index)
        twins.append((tag, np.asarray(ia)[ea.loc[common].values],
                      np.asarray(ib)[eb.loc[common].values]))

    allrows, dk_rows, corr_rows, split_rows, arch_rows = [], [], [], [], []

    for group, dname, k, kappa_ref in CASES:
        feats, desc = DEFS[(group, dname)]
        print(f"=== {group}/{dname} k={k}  (DT_uncert pooled kappa {kappa_ref:.3f}) ===")

        # --- typology: fitted ONCE on the pooled campaign, exactly as DT_uncert
        Xc = StandardScaler().fit_transform(f[feats].fillna(0))
        km = KMeans(k, random_state=RS, n_init=10).fit(Xc)
        y = km.labels_
        order = np.argsort([f[feats[0]].values[y == c].mean() for c in range(k)])
        remap = {o: i for i, o in enumerate(order)}
        y = np.array([remap[c] for c in y])

        folds = make_folds(X, y, econ)

        axis_col = U.AXIS[(group, dname)][0]
        prof_pooled = f.groupby(y)[V.PROFILE].mean().reindex(range(k))
        av = f[axis_col].values
        rng_pooled = [(av[y == i].min(), av[y == i].max(), av[y == i].mean()) for i in range(k)]
        # cluster names come from the POOLED profile for all five models
        cnames, rtxt, _ = U.cluster_naming(group, dname, k, prof_pooled, rng_pooled)

        preds, params, imps, rimps, trees = {}, {}, {}, {}, {}
        for model in MODELS:
            m = masks[model]
            base = pd.Series(y[m]).value_counts(normalize=True).max()
            sil = silhouette_score(Xc[m], y[m])

            kp, lv, ml, acc, bal = tune(X, y, folds, m, k)
            dt = V.mk_tree(lv, ml)
            pred = oof(dt, X, y, folds, m)
            ung = ungrouped_kappa(V.mk_tree(lv, ml), X, y, m)
            rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                        class_weight="balanced", random_state=0, n_jobs=-1)
            _, _, rfk = score(y, oof(rf, X, y, folds, m), m)

            dtf = V.mk_tree(lv, ml).fit(X.iloc[np.where(m)[0]], y[m])
            rff = rf.fit(X.iloc[np.where(m)[0]], y[m])

            hclf = DecisionTreeClassifier(max_leaf_nodes=U.HTML_LEAVES,
                                          min_samples_leaf=U.HTML_MIN_LEAF,
                                          min_samples_split=U.HTML_MIN_LEAF,
                                          class_weight="balanced", random_state=RS)
            _, _, hkp = score(y, oof(hclf, X, y, folds, m), m)

            prof_m, rng_m = profile_and_range(f, y, m, k, axis_col, prof_pooled, rng_pooled)
            _, _, subs = U.cluster_naming(group, dname, k, prof_m, rng_m)

            kdir = os.path.join(OUT, model, group, dname, "all", f"k{k:02d}")
            imp, rimp = write_case_artifacts(kdir, group, dname, k, feats, desc, model, f, dfm,
                                             X, y, m, dtf, rff, lv, ml, acc, bal, kp, ung, rfk,
                                             base, sil, cnames, rtxt, subs, hkp)

            preds[model], params[model], trees[model] = pred, (lv, ml), dtf
            imps[model], rimps[model] = imp, rimp
            rfeat, rthr = root_split(dtf)
            allrows.append(dict(model=model, group=group, definition=dname, k=k,
                                n=int(m.sum()), n_economies=int(pd.Series(econ[m]).nunique()),
                                silhouette=round(sil, 3), baseline=round(base, 3),
                                accuracy=acc, bal_accuracy=bal, kappa=kp,
                                kappa_ungrouped=ung, kappa_inflation=round(ung - kp, 3),
                                rf_kappa=rfk, dt_vs_rf=round(kp - rfk, 3), kappa_html8=hkp,
                                leaves=lv, min_leaf=ml, root_split=rfeat,
                                root_threshold=round(rthr, 2) if rthr == rthr else np.nan,
                                top_driver=imp.idxmax(), top_pct=round(imp.max(), 1),
                                dist_pct=round(imp[DIST].sum(), 1),
                                degenerate=int(base > 0.80),
                                PASS=int(kp >= 0.60 and base <= 0.80),
                                cluster_names=" | ".join(cnames), desc=desc))
            split_rows.append(dict(group=group, definition=dname, k=k, model=model,
                                   root_feature=rfeat, root_threshold=rthr,
                                   **{f"top{i + 1}": f"{a} {b:.0f}%" for i, (a, b) in
                                      enumerate(imp.sort_values(ascending=False).head(3).items())}))
            print(f"  {model:7s} n={int(m.sum()):5d}  kappa {kp:5.2f}  ung {ung:5.2f}  "
                  f"RF {rfk:5.2f}  base {base:.2f}  leaves {lv:2d}  top {imp.idxmax()}")

        # ---- measure 1: does stratifying buy accuracy? -----------------------
        for a in ARCHS:
            m = masks[a]
            _, _, ka = score(y, preds[a], m)
            _, _, kpool = score(y, preds["pooled"], m)
            lvp, mlp = params["pooled"]
            _, _, kfix = score(y, oof(V.mk_tree(lvp, mlp), X, y, folds, m), m)
            pos = econ_positions(econ, m)
            bt = boot_delta_kappa(y, preds[a], preds["pooled"], pos, np.asarray(list(pos)))
            sig = int(not (bt["d_lo"] <= 0 <= bt["d_hi"]))
            dk_rows.append(dict(group=group, definition=dname, k=k, archetype=a,
                                n=int(m.sum()), n_economies=len(pos),
                                kappa_archetype=ka, kappa_pooled_here=kpool,
                                d_kappa=round(ka - kpool, 3),
                                d_lo=round(bt["d_lo"], 3), d_hi=round(bt["d_hi"], 3),
                                significant=sig,
                                kappa_fixed_size=kfix,
                                d_kappa_fixed_size=round(kfix - kpool, 3),
                                leaves_archetype=params[a][0], leaves_pooled=lvp))
            print(f"  d_kappa {a}: {ka - kpool:+.3f}  [{bt['d_lo']:+.3f}, {bt['d_hi']:+.3f}]"
                  f"{'  *' if sig else ''}   (fixed size {kfix - kpool:+.3f})")

        # ---- measure 2: do the drivers themselves differ? --------------------
        B = boot_importances(X, y, econ, masks, params)
        pairs = [(ARCHS[i], ARCHS[j]) for i in range(4) for j in range(i + 1, 4)]
        pairs += [("pooled", a) for a in ARCHS]
        for p, q in pairs:
            rho = spearmanr(imps[p].values, imps[q].values).statistic
            # The DT importance vector is mostly zeros (a tuned tree uses 4-6 of the 15
            # drivers), so its rank correlation is partly carried by the tied zero block.
            # The RF importance vector has no zeros and separates the drivers the tree
            # never reaches -- it is the discriminating version of the same question.
            rho_rf = spearmanr(rimps[p].values, rimps[q].values).statistic
            tied0 = int(((imps[p].values == 0) & (imps[q].values == 0)).sum())
            bs = [spearmanr(B[p][b], B[q][b]).statistic for b in range(len(B[p]))]
            bs = [v for v in bs if v == v]
            corr_rows.append(dict(group=group, definition=dname, k=k, pair=f"{p} vs {q}",
                                  spearman=round(float(rho), 3),
                                  lo=round(float(np.percentile(bs, 2.5)), 3) if bs else np.nan,
                                  hi=round(float(np.percentile(bs, 97.5)), 3) if bs else np.nan,
                                  spearman_RF=round(float(rho_rf), 3),
                                  tied_zero_drivers=tied0))
        print("  importance rank corr DT (A-B, A-C, A-D, B-C, B-D, C-D): " +
              ", ".join(f"{r['spearman']:.2f}" for r in corr_rows[-10:-4]))
        print("  importance rank corr RF (A-B, A-C, A-D, B-C, B-D, C-D): " +
              ", ".join(f"{r['spearman_RF']:.2f}" for r in corr_rows[-10:-4]))

        # ---- measure 3: archetype as a feature, and the A/B twin floor -------
        Xa = X.copy()
        for a in ARCHS:
            Xa[f"arch_{a}"] = (arch == a).astype(int)
        acols = [f"arch_{a}" for a in ARCHS]
        names_a = list(Xa.columns)
        mp = masks["pooled"]
        cand = []
        for lv in [k, k + 2, 8, 12, 16, 20]:
            if lv < k:
                continue
            for ml in [10, 20, 40]:
                _, _, kk = score(y, oof(V.mk_tree(lv, ml), Xa, y, folds, mp), mp)
                cand.append((kk, lv, ml))
        top = max(c[0] for c in cand)
        good = sorted([c for c in cand if c[0] >= top - 0.01], key=lambda c: (c[1], -c[2]))
        ka_kappa, lva, mla = good[0]
        ta = V.mk_tree(lva, mla).fit(Xa, y)
        a_imp = float(pd.Series(ta.feature_importances_ * 100, index=names_a)[acols].sum())
        depth = first_depth_of(ta, names_a, set(acols))
        _, _, kpool_all = score(y, preds["pooled"], mp)

        twin_dis = float(np.mean([(y[ia] != y[ib]).mean() for _, ia, ib in twins]))
        arch_rows.append(dict(group=group, definition=dname, k=k,
                              kappa_pooled=kpool_all, kappa_pooled_with_archetype=ka_kappa,
                              d_kappa=round(ka_kappa - kpool_all, 3),
                              archetype_importance_pct=round(a_imp, 1),
                              first_depth_used=(int(depth) if np.isfinite(depth) else -1),
                              leaves=lva, min_leaf=mla,
                              AB_twin_disagreement=round(twin_dis, 3)))
        print(f"  archetype-as-feature: kappa {ka_kappa:.3f} vs pooled {kpool_all:.3f} "
              f"({ka_kappa - kpool_all:+.3f}), importance {a_imp:.1f}%, "
              f"first used at depth {depth}; A/B twin disagreement {twin_dis:.3f}\n")

    # --------------------------------------------------------- comparison --
    cdir = os.path.join(OUT, "comparison")
    os.makedirs(cdir, exist_ok=True)
    A = pd.DataFrame(allrows)
    DK = pd.DataFrame(dk_rows)
    CO = pd.DataFrame(corr_rows)
    SP = pd.DataFrame(split_rows)
    AR = pd.DataFrame(arch_rows)
    A.to_csv(os.path.join(OUT, "comparison_all.csv"), index=False)
    DK.to_csv(os.path.join(cdir, "delta_kappa.csv"), index=False)
    CO.to_csv(os.path.join(cdir, "importance_rank_corr.csv"), index=False)
    SP.to_csv(os.path.join(cdir, "top_splits.csv"), index=False)
    AR.to_csv(os.path.join(cdir, "archetype_feature_test.csv"), index=False)

    case_lbl = [f"{g}/{d} k{k}" for g, d, k, _ in CASES]
    key = DK.group + "/" + DK.definition + " k" + DK.k.astype(str)
    piv = DK.assign(_k=key).pivot_table(index="_k", columns="archetype",
                                        values="d_kappa").reindex(case_lbl)
    sigp = DK.assign(_k=key).pivot_table(index="_k", columns="archetype",
                                         values="significant").reindex(case_lbl)
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(piv) + 2))
    vmax = float(np.nanmax(np.abs(piv.values))) or 0.05
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v == v:
                star = "*" if sigp.values[i, j] == 1 else ""
                ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=8)
    ax.set_title("kappa(archetype tree) - kappa(pooled tree on the same rows)\n"
                 "* = cluster-bootstrap 95% interval over economies excludes 0",
                 color=C.UU_NAVY, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(cdir, "delta_kappa_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    ckey = CO.group + "/" + CO.definition + " k" + CO.k.astype(str)
    cp = CO.assign(_k=ckey).pivot_table(index="_k", columns="pair",
                                        values="spearman").reindex(case_lbl)
    fig, ax = plt.subplots(figsize=(1.0 * len(cp.columns) + 3, 0.5 * len(cp) + 2))
    im = ax.imshow(cp.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cp.columns)))
    ax.set_xticklabels(cp.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cp.index)))
    ax.set_yticklabels(cp.index, fontsize=8)
    for i in range(cp.shape[0]):
        for j in range(cp.shape[1]):
            v = cp.values[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if v < 0.6 else "k")
    ax.set_title("Spearman rank correlation of the driver-importance vectors",
                 color=C.UU_NAVY, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(cdir, "importance_corr_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    nsig = int(DK.significant.sum())
    med_rho = float(CO[~CO.pair.str.contains("pooled")].spearman.median())
    with open(os.path.join(cdir, "comparison.html"), "w", encoding="utf-8") as fh:
        fh.write(f"<html><head><meta charset='utf-8'><title>pooled vs per-archetype</title>"
                 f"{V.CSS}</head><body>"
                 f"<h1>One tree or four? &mdash; pooled vs per-archetype driver trees</h1>"
                 f"<div class='note'>Same typology (k-Means fitted once on the pooled "
                 f"campaign), same economy-grouped folds, same tuning grid. The pooled model "
                 f"is scored on an archetype by restricting its out-of-fold predictions to "
                 f"that archetype's rows, so <b>d_kappa is a model effect, not a test-set "
                 f"effect</b>. Intervals are cluster bootstraps over economies "
                 f"({N_BOOT_KAPPA} replicates for kappa, {N_BOOT_IMP} for the importance "
                 f"ranks).<br><b>{nsig} of {len(DK)}</b> archetype/case combinations have a "
                 f"d_kappa interval that excludes zero; median rank correlation between "
                 f"archetype importance vectors is <b>{med_rho:.2f}</b>.</div>"
                 f"<h2>1 &mdash; Does stratifying buy accuracy?</h2>"
                 f"<p><img src='delta_kappa_heatmap.png' style='max-width:900px'></p>"
                 f"{V.table_html(DK)}"
                 f"<h2>2 &mdash; Do the drivers themselves differ?</h2>"
                 f"<div class='note'><code>spearman</code> is the rank correlation of the "
                 f"decision-tree importance vectors, with a cluster-bootstrap interval over "
                 f"economies; a tuned tree uses only 4-6 of the 15 drivers, so "
                 f"<code>tied_zero_drivers</code> of them are tied at zero in both members of "
                 f"the pair and the correlation is partly carried by that tie block. "
                 f"<code>spearman_RF</code> repeats the measure on the random-forest "
                 f"importances, which have no zeros and therefore also rank the drivers the "
                 f"tree never reaches. Read the two together.</div>"
                 f"<p><img src='importance_corr_heatmap.png' style='max-width:1100px'></p>"
                 f"{V.table_html(CO)}"
                 f"<h3>Root split and top-3 drivers of every model</h3>{V.table_html(SP)}"
                 f"<h2>3 &mdash; Archetype as a feature, and the A/B twin floor</h2>"
                 f"<div class='note'>The pooled tree refitted with the archetype as four "
                 f"one-hot columns. If it is never used near the root and buys less kappa "
                 f"than the bootstrap noise, the stratification is descriptive rather than "
                 f"necessary. <code>AB_twin_disagreement</code> is the share of the A/B "
                 f"twin pairs (identical economics, identical distance triple) that fall in "
                 f"different clusters &mdash; an irreducible error floor for any model blind "
                 f"to the archetype.</div>{V.table_html(AR)}"
                 f"</body></html>")

    cols = ["model", "group", "definition", "k", "n", "n_economies", "silhouette", "baseline",
            "accuracy", "bal_accuracy", "kappa", "kappa_html8", "kappa_ungrouped",
            "kappa_inflation", "rf_kappa", "leaves", "root_split", "root_threshold",
            "top_driver", "top_pct", "dist_pct", "PASS"]
    with open(os.path.join(OUT, "overview.html"), "w", encoding="utf-8") as fh:
        fh.write(f"<html><head><meta charset='utf-8'><title>DT_uncert_archetype</title>{V.CSS}"
                 f"</head><body><h1>DT_supplychain &mdash; v5, uncertain parameters, "
                 f"pooled + one tree per archetype</h1>"
                 f"<div class='note'>{len(CASES)} cases &times; {len(MODELS)} models = "
                 f"{len(A)} trees. Cases are the seven PASS cases of the "
                 f"<code>DT_uncert</code> run of 2026-09-17 plus the best case of "
                 f"<code>decentr</code> and <code>topo</code>.<br>"
                 f"<b>Drivers ({len(DRIVERS)}):</b> {', '.join(DRIVERS)}.<br>"
                 f"The typology is fitted once on the pooled campaign and its labels are "
                 f"shared by all five models; cross-validation is economy-grouped and the "
                 f"archetype models use the restriction of the pooled folds, so all five "
                 f"models are scored on identical test rows. The <code>pooled</code> folder "
                 f"therefore reproduces the corresponding rows of <code>DT_uncert</code>."
                 f"</div>"
                 f"<h2>Models</h2><ul>" +
                 "".join(f"<li><b>{mo}</b> &mdash; " +
                         " &middot; ".join(
                             f"<a href='{mo}/{g}/{d}/all/k{k:02d}/tree.html'>{g}/{d} k{k}</a>"
                             for g, d, k, _ in CASES) + "</li>"
                         for mo in MODELS) +
                 f"</ul><h2><a href='comparison/comparison.html'>Comparison: one tree vs four"
                 f"</a></h2>"
                 f"<h2>All models, ranked by grouped &kappa;</h2>"
                 f"{V.table_html(A.sort_values('kappa', ascending=False)[cols])}"
                 f"</body></html>")

    print(f"\n{len(A)} models written to {OUT}")
    print(DK[["group", "definition", "k", "archetype", "kappa_archetype", "kappa_pooled_here",
              "d_kappa", "d_lo", "d_hi", "significant"]].to_string(index=False))
    print("\narchetype-as-feature test:")
    print(AR.to_string(index=False))


if __name__ == "__main__":
    main()
