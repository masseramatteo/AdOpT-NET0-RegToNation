"""
v5 DT_supplychain, uncertain-parameter edition.

Same cluster-then-explain sweep as dt_v5.py, with two deliberate changes asked for
after the first v5 run:

  1. DRIVERS contain ONLY the sampled uncertain parameters of the campaign — the
     12 economy axes plus the three sampled distances. The derived measure
     `k_small` (small-cluster grid self-sufficiency, avail_small / demand-implied
     need) is DROPPED: it is a diagnostic computed from two drivers already in the
     set, and it took the splits away from the parameters an analyst can actually
     set. See the campaign decision of 2026-09-09 (absolute MW kept as the sampled
     axes; g and k live only inside sampling_design.py).

  2. Every case gets the SCHEMATIC leaf HTML of the May 2026 (3000-run) study as
     `tree.html`: each leaf draws the representative 4-node network of its runs
     (pipe width = capacity, red = high pressure, node size = electrolyzer
     capacity, gold arrow = import volume, bar = local vs import). The tabular
     view dt_v5 used to write as tree.html is kept next to it as `details.html`.

Layout (mirrors the May deliverable):
    <group>/<definition>/all/k03..k10/{cluster_profiles.csv, driver_tree.png,
                                       feature_importance.csv, rules.txt,
                                       tree.html, details.html}
    <group>/{comparison.csv, comparison.html, definitions.csv, kappa_heatmap.png}
    comparison_all.csv / overview.html

Cross-validation stays grouped by economy (8 runs share one economy in the v5
paired design); the ungrouped kappa is reported alongside to show the inflation.

Run:  python dt_v5_uncertain.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

import config as C
import dt_v5 as V
import render_driver_tree_html as R

RS = V.RS
KS = V.KS
OUT = os.path.join(C.RESULTS_FOLDER, "DT_uncert")

# --- drivers: sampled uncertain parameters only (k_small dropped) -------------
DRIVERS = [d for d in V.DRIVERS if d != "k_small"]
DIST = V.DIST

# schematic HTML tree: fixed small leaf budget, the May study's readability choice
HTML_LEAVES = 8
HTML_MIN_LEAF = 20

# --- labels for the schematic renderer, updated to the v5 column names --------
R.FEAT.update({
    "electricity_availability_small": ["Elec. availability (small)", "MW"],
    "electricity_availability_large": ["Elec. availability (large)", "MW"],
    "solar_cf_mean": ["Solar capacity factor", "-"],
    "d_SL_mean": ["Distance small→large", "km"],
    "d_LL": ["Distance large→large", "km"],
    "d_SS": ["Distance small→small", "km"],
})
R.PAIRS.update({
    "solar_cf_mean": ["low", "high"],
    "d_SL_mean": ["near", "far"],
    "d_LL": ["near", "far"],
    "d_SS": ["near", "far"],
})
GRAD = [C.UU_NAVY, "#2f3d7a", C.UU_TEAL, "#9bc4b3", C.UU_YELLOW, C.UU_AMBER,
        C.UU_RED, "#6b4e9e", "#e08a2b", "#8a8f9c"]

# --- cluster naming: ordered levels along the definition's own axis ----------
# Same scheme as the May 2026 decentralisation study: clusters are k-Means groups
# ordered low->high on the leading clustering feature, so they get level names
# ("Low decentr.", "High decentr.", ...) plus the numeric range they cover,
# instead of a profile string.
LEVELS = {
    2: ["Low", "High"],
    3: ["Low", "Medium", "High"],
    4: ["Very low", "Low", "High", "Very high"],
    5: ["Very low", "Low", "Medium", "High", "Very high"],
    6: ["Very low", "Low", "Med-low", "Med-high", "High", "Very high"],
    7: ["Very low", "Low", "Med-low", "Medium", "Med-high", "High", "Very high"],
}

# (group, definition) -> (axis column, kind, label). The axis is the feature the
# clusters are ordered on in dt_v5 (feats[0]).
AXIS = {
    ("sysmix", "sysmix"): ("self_served", "pct", "self-served"),
    ("sysmix", "cent_imp"): ("self_served", "pct", "self-served"),
    ("sysmix", "selfserved"): ("self_served", "pct", "self-served"),
    ("sysmix", "transport"): ("transported", "pct", "transport"),
    ("decentr", "A_share"): ("small_prod_share", "pct", "decentr."),
    ("decentr", "B_selfsuff"): ("small_selfsuff", "pct", "small self-suff."),
    ("decentr", "hubdep"): ("hub_dependence", "pct", "hub dependence"),
    ("decentr", "B_plus_mix"): ("small_selfsuff", "pct", "small self-suff."),
    ("topo", "count"): ("n_pipelines", "count", "pipelines"),
    ("topo", "LS"): ("n_LS_arcs", "count", "hub feeds"),
    ("topo", "backbone"): ("LL_pipe", "backbone", "backbone"),
    ("topo", "meshsat"): ("SS_pipe", "mesh", "small-small link"),
    ("topo", "structure"): ("LL_pipe", "backbone", "backbone"),
    ("combo", "mix_topo"): ("self_served", "pct", "self-served"),
    ("combo", "supply_struct"): ("self_served", "pct", "self-served"),
    ("combo", "full"): ("self_served", "pct", "self-served"),
}


def level_words(k):
    return LEVELS.get(k, [f"Level {i + 1}/{k}" for i in range(k)])


def _count_txt(lo, hi, label):
    return f"{lo:.0f} {label}" if round(lo) == round(hi) else f"{lo:.0f}–{hi:.0f} {label}"


def cluster_naming(group, dname, k, prof, rng):
    """Ordered level names + numeric range per cluster.

    rng[i] = (min, max, mean) of the ordering axis inside cluster i.
    Returns (names, ranges_txt, subtitles): the name goes on the tree/leaf card,
    the range is appended in rules.txt and cluster_levels.csv, the subtitle is the
    small grey line under the leaf title in the schematic HTML.
    """
    axis_col, kind, label = AXIS[(group, dname)]
    words = level_words(k)
    names, rtxt, subs = [], [], []
    for i in range(k):
        lo, hi, mu = rng[i]
        p = prof.loc[i]
        if kind == "pct":
            names.append(f"{words[i]} {label}")
            rtxt.append(f"{lo:.0%}-{hi:.0%}")
            sub = f"{lo:.0%}–{hi:.0%} {label}"
        elif kind == "count":
            names.append(_count_txt(lo, hi, label))
            rtxt.append(f"{lo:.0f}-{hi:.0f}")
            sub = f"{_count_txt(lo, hi, label)} built"
        elif kind == "backbone":
            base = "Backbone" if p["LL_pipe"] > 0.5 else "No backbone"
            names.append(f"{base}, {p['n_pipelines']:.0f} pipes")
            rtxt.append(f"LL {p['LL_pipe']:.0%}")
            sub = (f"large–large link in {p['LL_pipe']:.0%} of runs · "
                   f"{p['n_pipelines']:.1f} pipes · {p['n_LS_arcs']:.1f} hub feeds")
        else:  # mesh
            base = "Meshed smalls" if p["SS_pipe"] > 0.5 else "Radial smalls"
            names.append(f"{base}, {p['n_LS_arcs']:.0f} hub feeds")
            rtxt.append(f"SS {p['SS_pipe']:.0%}")
            sub = (f"small–small link in {p['SS_pipe']:.0%} of runs · "
                   f"{p['n_LS_arcs']:.1f} hub feeds")
        # second line of context, always: where the hydrogen comes from
        subs.append(f"{sub} · self {p['self_served']:.0%} / imp {p['imported']:.0%} / "
                    f"transp {p['transported']:.0%} · {p['n_pipelines']:.1f} pipes")
    names = V.uniq(names)
    return names, rtxt, subs


def schematic_html(dfm, y, names, subs, out_html, title, sub_text):
    """Schematic-leaf driver tree (the May 2026 HTML design), fitted on DRIVERS."""
    dfm = dfm.copy()
    dfm["cluster_final"] = y
    X = dfm[DRIVERS].fillna(0.0)
    clf = DecisionTreeClassifier(max_leaf_nodes=HTML_LEAVES, min_samples_leaf=HTML_MIN_LEAF,
                                 min_samples_split=HTML_MIN_LEAF, class_weight="balanced",
                                 random_state=RS).fit(X, y)
    dfm["_leaf"] = clf.apply(X)
    leaf_rows = {lid: dfm[dfm["_leaf"] == lid] for lid in np.unique(dfm["_leaf"])}
    name_by = {i: names[i] for i in range(len(names))}
    tree = R._build_node(clf.tree_, 0, DRIVERS, leaf_rows, name_by)
    leaves = R._leaves(tree)
    scale = {
        "pipe_max": max(max(nd["info"][a][1] for a in R.ARC_COLS) for nd in leaves) or 1.0,
        "large_elzr_max": max(max(nd["info"]["cap_L1"], nd["info"]["cap_L2"]) for nd in leaves) or 1.0,
        "small_elzr_max": max(max(nd["info"]["cap_S1"], nd["info"]["cap_S2"]) for nd in leaves) or 1.0,
        "import_max": max(max(nd["info"]["imp_L1"], nd["info"]["imp_L2"]) for nd in leaves) or 1.0,
    }
    color = {names[i]: GRAD[i % len(GRAD)] for i in range(len(names))}
    sub = {names[i]: subs[i] for i in range(len(names))}
    present = list(dict.fromkeys(nd["info"]["label"] for nd in leaves))
    cbar = "\n".join(f' <span class="cchip"><span class="dot" style="background:{color[l]}"></span>{l}</span>'
                     for l in present)
    html = R._TEMPLATE
    for tok, val in [("/*__CBAR__*/", cbar),
                     ("/*__DATA__*/", json.dumps({"tree": tree, "scale": scale})),
                     ("/*__FEAT__*/", json.dumps(R.FEAT)), ("/*__SUB__*/", json.dumps(sub)),
                     ("/*__COLOR__*/", json.dumps(color)), ("/*__PAIRS__*/", json.dumps(R.PAIRS))]:
        html = html.replace(tok, val)
    html = html.replace("<h1>From input parameters to hydrogen infrastructure design</h1>",
                        f"<h1>{title}</h1>")
    old_sub = ("Driver tree over 2719 optimizations. Each diamond tests one sampled input "
               "parameter;")
    html = html.replace(old_sub, sub_text + " Each diamond tests one sampled input parameter;")
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    return clf


def cluster_subtitle(p):
    """One-line output profile of a cluster, shown under its name on each leaf card."""
    return (f"self {p['self_served']:.0%} · import {p['imported']:.0%} · "
            f"transp {p['transported']:.0%} · small self-suff {p['small_selfsuff']:.0%} "
            f"· {p['n_pipelines']:.1f} pipes")


def main():
    f = V.load(C.EXTRACTED_RESULTS)
    raw = pd.read_excel(C.EXTRACTED_RESULTS, sheet_name="Results")
    # rows needed by the schematic renderer: raw design sizes + the import share
    dfm = raw.copy()
    dfm["import_intensity"] = f["imported"].values
    X = f[DRIVERS].fillna(0.0)
    g = f.economy.values
    os.makedirs(OUT, exist_ok=True)
    print(f"{len(f)} runs, {f.economy.nunique()} economies, {len(DRIVERS)} uncertain drivers")
    print("drivers:", ", ".join(DRIVERS))

    allrows = []
    for group, defs in V.GROUPS.items():
        gdir = os.path.join(OUT, group)
        os.makedirs(gdir, exist_ok=True)
        pd.DataFrame([{"definition": n, "features": " + ".join(ft), "description": ds}
                      for n, ft, ds in defs]).to_csv(
            os.path.join(gdir, "definitions.csv"), index=False)
        grows = []
        for dname, feats, desc in defs:
            Xc = StandardScaler().fit_transform(f[feats].fillna(0))
            nun = f[feats].round(3).drop_duplicates().shape[0]
            sil_curve = []
            for k in [k for k in KS if k <= nun]:
                km = KMeans(k, random_state=RS, n_init=10).fit(Xc)
                y = km.labels_
                order = np.argsort([f[feats[0]].values[y == c].mean() for c in range(k)])
                remap = {o: i for i, o in enumerate(order)}
                y = np.array([remap[c] for c in y])
                sil = silhouette_score(Xc, y)
                sil_curve.append((k, sil, km.inertia_))
                base = pd.Series(y).value_counts(normalize=True).max()

                kp, lv, ml, acc, bal = V.tune(X, y, g, k)
                dt = V.mk_tree(lv, ml)
                _, _, ung = V.metrics(dt, X, y, g, grouped=False)
                rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                            class_weight="balanced", random_state=0, n_jobs=-1)
                _, _, rfk = V.metrics(rf, X, y, g)
                dt.fit(X, y)
                imp = pd.Series(dt.feature_importances_ * 100, index=DRIVERS)
                rff = rf.fit(X, y)
                rimp = pd.Series(rff.feature_importances_ * 100, index=DRIVERS)

                prof = f.groupby(y)[V.PROFILE].mean()
                prof["n"] = pd.Series(y).value_counts().sort_index()
                axis_col = AXIS[(group, dname)][0]
                av = f[axis_col].values
                rng = [(av[y == i].min(), av[y == i].max(), av[y == i].mean()) for i in range(k)]
                cnames, rtxt, subs = cluster_naming(group, dname, k, prof, rng)
                # names used in rules.txt / plot_tree carry the range, as in the May study
                rnames = [f"{n} ({r})" for n, r in zip(cnames, rtxt)]
                prof.insert(0, "name", cnames)

                kdir = os.path.join(gdir, dname, "all", f"k{k:02d}")
                os.makedirs(kdir, exist_ok=True)
                prof.round(3).to_csv(os.path.join(kdir, "cluster_profiles.csv"))
                pd.DataFrame({"level": range(k), "name": cnames, "axis": axis_col,
                              "range_lo": [r[0] for r in rng], "range_hi": [r[1] for r in rng],
                              "mean": [r[2] for r in rng],
                              "n": [int((y == i).sum()) for i in range(k)]}
                             ).round(4).to_csv(os.path.join(kdir, "cluster_levels.csv"),
                                               index=False)
                pd.DataFrame({"feature": DRIVERS,
                              "DT_importance_pct": imp.round(1).values,
                              "RF_importance_pct": rimp.round(1).values}
                             ).sort_values("DT_importance_pct", ascending=False).to_csv(
                    os.path.join(kdir, "feature_importance.csv"), index=False)
                rules = V.leaf_rules(dt, DRIVERS, rnames)
                with open(os.path.join(kdir, "rules.txt"), "w", encoding="utf-8") as fh:
                    fh.write(f"# {group}/{dname} k={k} | grouped kappa {kp:.3f} "
                             f"| acc {acc:.3f} | baseline {base:.3f}\n")
                    fh.write(f"# clustered on: {', '.join(feats)}\n")
                    fh.write(f"# drivers: uncertain sampled parameters only (no k_small)\n\n")
                    for r in rules:
                        mark = "" if r["purity"] >= 0.70 else "  [mixed]"
                        fh.write(f"[{r['purity']:.0%}, n={r['n']}] {r['cls']}{mark}\n"
                                 f"    IF {r['rule']}\n")

                fig, ax = plt.subplots(figsize=(17, 9))
                plot_tree(dt, feature_names=DRIVERS, class_names=rnames, filled=True,
                          rounded=True, proportion=True, precision=1, fontsize=7, ax=ax)
                ax.set_title(f"{group}/{dname}  k={k}   baseline {base:.2f} | "
                             f"acc {acc:.2f} | grouped kappa {kp:.2f} | RF kappa {rfk:.2f}",
                             fontsize=12, color=C.UU_NAVY)
                fig.savefig(os.path.join(kdir, "driver_tree.png"), dpi=170, bbox_inches="tight")
                plt.close(fig)

                # schematic-leaf HTML (May 2026 design), plus its own grouped kappa
                hclf = DecisionTreeClassifier(max_leaf_nodes=HTML_LEAVES,
                                              min_samples_leaf=HTML_MIN_LEAF,
                                              min_samples_split=HTML_MIN_LEAF,
                                              class_weight="balanced", random_state=RS)
                _, _, hkp = V.metrics(hclf, X, y, g)
                sub_text = (f"{len(f)} optimizations, {f.economy.nunique()} economies. "
                            f"{HTML_LEAVES}-leaf tree, grouped &kappa; {hkp:.2f} "
                            f"(baseline {base:.2f}). Clustered on {', '.join(feats)}.")
                schematic_html(dfm, y, cnames, subs, os.path.join(kdir, "tree.html"),
                               f"{group} / {dname} &mdash; k={k}: from uncertain parameters "
                               f"to hydrogen infrastructure", sub_text)

                top3 = imp.sort_values(ascending=False).head(3)
                with open(os.path.join(kdir, "details.html"), "w", encoding="utf-8") as fh:
                    fh.write(f"<html><head><meta charset='utf-8'><title>{group}/{dname} k{k}</title>"
                             f"{V.CSS}</head><body><h1>{group} / {dname} &mdash; k={k}</h1>"
                             f"<div class='note'>{desc}<br>Clustered on: <b>{', '.join(feats)}</b><br>"
                             f"Drivers: uncertain sampled parameters only (k_small dropped).<br>"
                             f"grouped &kappa; <b>{kp:.2f}</b> {V.verdict(kp)} &middot; accuracy {acc:.2f} "
                             f"&middot; balanced {bal:.2f} &middot; baseline {base:.2f} "
                             f"&middot; RF &kappa; {rfk:.2f} &middot; ungrouped &kappa; {ung:.2f}<br>"
                             f"<a href='tree.html'>schematic driver tree</a> "
                             f"({HTML_LEAVES} leaves, grouped &kappa; {hkp:.2f})</div>"
                             f"<h2>Cluster profiles</h2>{prof.round(3).to_html()}"
                             f"<h2>Drivers</h2><p>" +
                             ", ".join(f"{a} {b:.0f}%" for a, b in top3.items()) +
                             f"</p><h2>Leaf rules</h2><pre>" +
                             "\n".join(f"[{r['purity']:.0%}, n={r['n']}] {r['cls']}\n    IF {r['rule']}"
                                       for r in rules) +
                             f"</pre><h2>Tree</h2><img src='driver_tree.png' style='width:100%'>"
                             f"</body></html>")

                row = dict(group=group, definition=dname, k=k, n=len(f),
                           silhouette=round(sil, 3), baseline=round(base, 3),
                           accuracy=acc, bal_accuracy=bal, kappa=kp,
                           kappa_ungrouped=ung, kappa_inflation=round(ung - kp, 3),
                           rf_kappa=rfk, dt_vs_rf=round(kp - rfk, 3),
                           kappa_html8=hkp, leaves=lv, min_leaf=ml,
                           top_driver=imp.idxmax(), top_pct=round(imp.max(), 1),
                           dist_pct=round(imp[DIST].sum(), 1),
                           degenerate=int(base > 0.80),
                           PASS=int(kp >= 0.60 and base <= 0.80),
                           cluster_names=" | ".join(cnames), desc=desc)
                grows.append(row)
                allrows.append(row)
                print(f"  {group:8s} {dname:14s} k={k:2d}  kappa {kp:5.2f}  html8 {hkp:5.2f}  "
                      f"base {base:.2f}  sil {sil:.2f}  RF {rfk:5.2f}  "
                      f"top {imp.idxmax():32s} dist {imp[DIST].sum():4.1f}%")

            if sil_curve:
                fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
                ks = [c[0] for c in sil_curve]
                ax[0].plot(ks, [c[2] for c in sil_curve], "o-", color=C.UU_NAVY)
                ax[0].set_xlabel("k"); ax[0].set_ylabel("inertia"); ax[0].set_title("elbow")
                ax[1].plot(ks, [c[1] for c in sil_curve], "o-", color=C.UU_TEAL)
                ax[1].set_xlabel("k"); ax[1].set_ylabel("silhouette"); ax[1].set_title("silhouette")
                fig.suptitle(f"{group}/{dname}", color=C.UU_NAVY)
                fig.tight_layout()
                fig.savefig(os.path.join(gdir, dname, "elbow_silhouette.png"),
                            dpi=150, bbox_inches="tight")
                plt.close(fig)

        G = pd.DataFrame(grows).sort_values("kappa", ascending=False)
        G.to_csv(os.path.join(gdir, "comparison.csv"), index=False)
        cols = ["definition", "k", "silhouette", "baseline", "accuracy", "bal_accuracy",
                "kappa", "kappa_html8", "kappa_ungrouped", "rf_kappa", "top_driver", "top_pct",
                "dist_pct", "degenerate", "PASS"]
        with open(os.path.join(gdir, "comparison.html"), "w", encoding="utf-8") as fh:
            fh.write(f"<html><head><meta charset='utf-8'><title>{group}</title>{V.CSS}</head><body>"
                     f"<h1>{group} &mdash; all cases (uncertain drivers)</h1>"
                     f"<div class='note'>Splits use the sampled uncertain parameters only; the "
                     f"derived <code>k_small</code> is not a feature. <b>PASS</b> = grouped "
                     f"&kappa; &ge; 0.60 and baseline &le; 0.80; a flag, not a filter. "
                     f"<code>kappa_html8</code> is the {HTML_LEAVES}-leaf tree drawn in "
                     f"<code>tree.html</code>.</div>{V.table_html(G[cols])}</body></html>")

        piv = G.pivot_table(index="definition", columns="k", values="kappa")
        fig, ax = plt.subplots(figsize=(1.0 * len(piv.columns) + 3, 0.5 * len(piv) + 2))
        im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="w" if v < 0.6 else "k", fontsize=8)
        ax.set_title(f"{group}: grouped kappa by definition and k", color=C.UU_NAVY)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(os.path.join(gdir, "kappa_heatmap.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    A = pd.DataFrame(allrows).sort_values("kappa", ascending=False)
    A.to_csv(os.path.join(OUT, "comparison_all.csv"), index=False)
    cols = ["group", "definition", "k", "silhouette", "baseline", "accuracy", "bal_accuracy",
            "kappa", "kappa_html8", "kappa_ungrouped", "kappa_inflation", "rf_kappa", "dt_vs_rf",
            "top_driver", "top_pct", "dist_pct", "degenerate", "PASS"]
    npass = int(A.PASS.sum())
    with open(os.path.join(OUT, "overview.html"), "w", encoding="utf-8") as fh:
        fh.write(f"<html><head><meta charset='utf-8'><title>DT_uncert v5</title>{V.CSS}</head><body>"
                 f"<h1>DT_supplychain &mdash; v5 campaign, uncertain parameters only</h1>"
                 f"<div class='note'><b>{len(A)} cases</b> across {len(V.GROUPS)} approach groups, "
                 f"k = 3..10, all {len(f)} runs / {f.economy.nunique()} economies. "
                 f"{npass} cases meet &kappa; &ge; 0.60 with baseline &le; 0.80 (PASS flag, "
                 f"nothing is dropped).<br>"
                 f"<b>Drivers ({len(DRIVERS)}):</b> {', '.join(DRIVERS)} &mdash; the sampled "
                 f"uncertain axes of the campaign. The derived <code>k_small</code> used in the "
                 f"first v5 run is excluded on purpose: it is a function of two drivers already "
                 f"present and it pulled the splits onto a quantity that is not a scenario knob."
                 f"<br>Each case's <code>tree.html</code> is the schematic-leaf design of the May "
                 f"2026 study: every leaf draws the representative 4-node network of its runs. "
                 f"<code>details.html</code> holds the profile tables and the plain tree figure."
                 f"<br>Cross-validation is <b>grouped by economy</b> (8 topologies per economy); "
                 f"<code>kappa_inflation</code> = ungrouped &minus; grouped is the leak that "
                 f"ignoring the paired design would have bought.</div>"
                 f"<h2>Groups</h2><ul>" +
                 "".join(f"<li><a href='{gp}/comparison.html'>{gp}</a> &mdash; "
                         f"{len(V.GROUPS[gp])} definitions</li>" for gp in V.GROUPS) +
                 f"</ul><h2>All cases, ranked by grouped &kappa;</h2>{V.table_html(A[cols])}"
                 f"</body></html>")
    print(f"\n{len(A)} cases written to {OUT}")
    print(f"PASS (kappa>=0.60 and baseline<=0.80): {npass}")
    print(A[["group", "definition", "k", "baseline", "kappa", "kappa_html8", "rf_kappa",
             "top_driver", "dist_pct", "PASS"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
