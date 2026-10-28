"""
v5 DT_supplychain: cluster-then-explain over the 6000-run campaign.

Mirrors the May 2026 deliverable layout:
    <group>/<definition>/all/k03..k10/{cluster_profiles.csv, driver_tree.png,
                                       feature_importance.csv, rules.txt, tree.html}
    <group>/{comparison.csv, comparison.html, drivers.html, definitions.csv}
    comparison_all.csv / overview.html   (every case, all groups)

Differences vs May, all deliberate:
  - k starts at 3 (k=2 excluded).
  - Every case is reported; the quality criteria are FLAGS, never a filter.
  - Cross-validation is grouped by economy (8 runs share one economy in the v5
    paired design); ungrouped kappa is reported alongside to show the inflation.
  - No robustness subset (near-optimal re-optimisation does not exist for v5 yet).
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, silhouette_score)
from sklearn.model_selection import (cross_val_predict, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

RS = 42
KS = range(3, 11)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DT_supplychain_v5")
UU_NAVY, UU_TEAL, UU_YELLOW, UU_RED, UU_AMBER = "#161D41", "#63A593", "#FFCD00", "#C00935", "#b8900a"

N = ["Large_cluster1", "Large_cluster2", "Small_cluster1", "Small_cluster2"]
LARGE, SMALL = N[:2], N[2:]

DRIVERS = ["k_small", "electricity_availability_small", "electricity_availability_large",
           "total_demand_TWh", "demand_level_ratio", "unbalance_ratio",
           "import_availability_ratio", "electricity_price_avg", "electricity_standard_dev",
           "hydrogen_import_price", "solar_cf_mean", "capex_ratio_small_big",
           "pipeline_cost_multiplier", "d_SL_mean", "d_LL", "d_SS"]
DIST = ["d_SL_mean", "d_LL", "d_SS"]

PROFILE = ["self_served", "imported", "transported", "small_selfsuff", "hub_dependence",
           "small_prod_share", "n_pipelines", "n_LS_arcs", "LL_pipe", "SS_pipe", "highP_share"]

# group -> list of (definition name, clustering features, description)
GROUPS = {
    "sysmix": [
        ("sysmix", ["self_served", "imported", "transported"],
         "System supply mix: shares of total demand that are self-served, imported, transported"),
        ("cent_imp", ["self_served", "imported"],
         "Self-produced vs imported (transport folded out)"),
        ("selfserved", ["self_served"],
         "Degree of self-served demand: system decentralisation on one axis"),
        ("transport", ["transported"],
         "Transport intensity: how much H2 must be moved over the network"),
    ],
    "decentr": [
        ("A_share", ["small_prod_share"],
         "Def A: small-cluster production / total system demand"),
        ("B_selfsuff", ["small_selfsuff"],
         "Def B: small-cluster production / small-cluster demand (self-sufficiency)"),
        ("hubdep", ["hub_dependence"],
         "Small-cluster dependence on hub deliveries (1 - self-sufficiency)"),
        ("B_plus_mix", ["small_selfsuff", "imported"],
         "Small-cluster autonomy combined with system import reliance"),
    ],
    "topo": [
        ("count", ["n_pipelines"], "Network scale: number of pipelines built (of 6 arcs)"),
        ("LS", ["n_LS_arcs"], "Number of large-to-small corridors built (of 4)"),
        ("backbone", ["LL_pipe", "n_pipelines"], "Hub backbone presence plus network scale"),
        ("meshsat", ["SS_pipe", "n_LS_arcs"], "Small-small link vs radial hub feeds"),
        ("structure", ["LL_pipe", "SS_pipe", "n_LS_arcs", "highP_share"],
         "Full network structure: which corridors and at which pressure"),
    ],
    "combo": [
        ("mix_topo", ["self_served", "imported", "transported", "n_pipelines"],
         "System supply mix plus network scale"),
        ("supply_struct", ["self_served", "imported", "n_LS_arcs", "SS_pipe"],
         "Supply origin plus corridor structure"),
        ("full", ["self_served", "imported", "transported", "small_selfsuff",
                  "n_pipelines", "LL_pipe", "SS_pipe", "highP_share"],
         "All outputs of interest together (the May 'full' definition)"),
    ],
}


# --------------------------------------------------------------- features --
def load(xlsx="extracted_results.xlsx"):
    d = pd.read_excel(xlsx, sheet_name="Results")
    cols = [f"{n}{s}" for n in N for s in
            ["_TOTAL_H2_production", "_hydrogen_import_sum",
             "_hydrogen_network_inflow_sum", "_hydrogen_network_outflow_sum"]]
    d[cols] = d[cols].fillna(0.0)
    P = {n: d[f"{n}_TOTAL_H2_production"] for n in N}
    I = {n: d[f"{n}_hydrogen_import_sum"] for n in N}
    FI = {n: d[f"{n}_hydrogen_network_inflow_sum"] for n in N}
    FO = {n: d[f"{n}_hydrogen_network_outflow_sum"] for n in N}
    D = {n: P[n] + I[n] + FI[n] - FO[n] for n in N}
    tot = sum(D[n] for n in N)

    f = pd.DataFrame(index=d.index)
    for c in ("run", "economy", "slot", "archetype"):
        f[c] = d[c]

    s_self = sum(np.minimum(P[n], D[n]) for n in N)
    s_imp = sum(I[n] for n in N)
    s_pipe = sum((FI[n] - FO[n]).clip(lower=0) for n in N)
    tt = (s_self + s_imp + s_pipe).replace(0, np.nan)
    f["self_served"] = (s_self / tt).fillna(0)
    f["imported"] = (s_imp / tt).fillna(0)
    f["transported"] = (s_pipe / tt).fillna(0)

    sm_p, sm_d = sum(P[n] for n in SMALL), sum(D[n] for n in SMALL)
    f["small_selfsuff"] = (sm_p / sm_d).clip(0, 1)
    f["hub_dependence"] = (sum(FI[n] - FO[n] for n in SMALL) / sm_d).clip(0, 1)
    f["small_prod_share"] = sm_p / tot

    arcs = [("Large_cluster1", "Large_cluster2"), ("Large_cluster1", "Small_cluster1"),
            ("Large_cluster1", "Small_cluster2"), ("Large_cluster2", "Small_cluster1"),
            ("Large_cluster2", "Small_cluster2"), ("Small_cluster1", "Small_cluster2")]
    built, hp = {}, {}
    for a, b in arcs:
        h = d.get(f"{a}_to_{b}_hydrogenPipelineOnshore_highP",
                  pd.Series(0., index=d.index)).fillna(0)
        lo = d.get(f"{a}_to_{b}_hydrogenPipelineOnshore_lowP",
                   pd.Series(0., index=d.index)).fillna(0)
        built[(a, b)] = ((h > 1) | (lo > 1)).astype(int)
        hp[(a, b)] = (h > 1).astype(int)
    LS = [(a, b) for a, b in arcs if a.startswith("Large") and b.startswith("Small")]
    f["n_pipelines"] = sum(built[k] for k in built)
    f["n_LS_arcs"] = sum(built[k] for k in LS)
    f["LL_pipe"] = built[("Large_cluster1", "Large_cluster2")]
    f["SS_pipe"] = built[("Small_cluster1", "Small_cluster2")]
    f["highP_share"] = (sum(hp[k] for k in hp) / f.n_pipelines.replace(0, np.nan)).fillna(0)

    f["k_small"] = d.electricity_availability_small / (
        (d.total_demand_TWh / (1 + d.demand_level_ratio) / 2) * 1e6 / 8760 / 0.655)
    for c in ["total_demand_TWh", "demand_level_ratio", "unbalance_ratio",
              "import_availability_ratio", "electricity_price_avg", "electricity_standard_dev",
              "hydrogen_import_price", "electricity_availability_small",
              "electricity_availability_large", "solar_cf_mean", "capex_ratio_small_big",
              "pipeline_cost_multiplier", "d_LL", "d_SS", "d_SL_mean"]:
        f[c] = d[c]
    return f


# ---------------------------------------------------------------- scoring --
def metrics(clf, X, y, groups, grouped=True):
    cv = (StratifiedGroupKFold(5, shuffle=True, random_state=RS) if grouped
          else StratifiedKFold(5, shuffle=True, random_state=RS))
    p = cross_val_predict(clf, X, y, cv=cv, groups=groups if grouped else None)
    return (round(accuracy_score(y, p), 3), round(balanced_accuracy_score(y, p), 3),
            round(cohen_kappa_score(y, p), 3))


def mk_tree(lv, ml):
    return DecisionTreeClassifier(max_leaf_nodes=lv, min_samples_leaf=ml, min_samples_split=ml,
                                  class_weight="balanced", random_state=RS)


def tune(X, y, groups, k):
    """Pick (max_leaf_nodes, min_samples_leaf) on grouped-CV kappa; prefer fewer leaves
    among candidates within 0.01 of the best (readability)."""
    cand = []
    for lv in [k, k + 2, 8, 12, 16, 20]:
        if lv < k:
            continue
        for ml in [10, 20, 40]:
            a, b, kp = metrics(mk_tree(lv, ml), X, y, groups)
            cand.append((kp, lv, ml, a, b))
    top = max(c[0] for c in cand)
    good = [c for c in cand if c[0] >= top - 0.01]
    good.sort(key=lambda c: (c[1], -c[2]))
    return good[0]


def leaf_rules(tree, names, classes):
    """One readable rule per leaf: purity, n, class, condition path."""
    t = tree.tree_
    out = []

    def walk(i, cond):
        if t.children_left[i] == -1:
            v = t.value[i][0]
            out.append(dict(rule=" AND ".join(cond) if cond else "(root)",
                            cls=classes[int(np.argmax(v))],
                            purity=float(v.max() / v.sum()),
                            n=int(t.n_node_samples[i])))
            return
        f, th = names[t.feature[i]], t.threshold[i]
        walk(t.children_left[i], cond + [f"{f} <= {th:.1f}"])
        walk(t.children_right[i], cond + [f"{f} > {th:.1f}"])

    walk(0, [])
    return sorted(out, key=lambda r: -r["n"])


def name_cluster(p):
    """Descriptive label from a cluster's mean output profile."""
    ss, imp, tr = p["self_served"], p["imported"], p["transported"]
    sd, npi = p["small_selfsuff"], p["n_pipelines"]
    if npi < 0.5:
        net = "islanded"
    elif p["SS_pipe"] > 0.5:
        net = "S-S linked"
    elif p["LL_pipe"] > 0.5 and p["n_LS_arcs"] < 1.0:
        net = "backbone"
    elif p["n_LS_arcs"] >= 1.5:
        net = "radial"
    else:
        net = "sparse"
    if imp > 0.35:
        sup = "import-reliant"
    elif tr > 0.20:
        sup = "transport-heavy"
    elif ss > 0.85:
        sup = "self-served"
    else:
        sup = "mixed-supply"
    aut = "autonomous" if sd > 0.9 else ("hub-fed" if sd < 0.25 else "partial")
    return f"{sup} / {aut} / {net}"


def uniq(names):
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} #{seen[n]}")
    return out


# ------------------------------------------------------------------- html --
CSS = """<style>
body{font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#161D41;margin:24px;max-width:1500px}
h1{font-size:20px;border-bottom:3px solid #161D41;padding-bottom:6px}
h2{font-size:16px;margin-top:26px}
table{border-collapse:collapse;margin:10px 0;font-size:12px}
th,td{border:1px solid #ccd;padding:4px 8px;text-align:right}
th{background:#161D41;color:#fff;text-align:center;position:sticky;top:0}
td.l,th.l{text-align:left}
tr.pass td{background:#e8f3ef}
tr.fail td{color:#8a8a8a}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;color:#fff;font-size:11px}
.strong{background:#63A593}.mod{background:#FFCD00;color:#161D41}.weak{background:#C00935}
.note{background:#f6f6f9;border-left:4px solid #63A593;padding:8px 12px;margin:12px 0}
a{color:#161D41}
</style>"""


def verdict(kappa):
    if kappa >= 0.60:
        return '<span class="tag strong">strong</span>'
    if kappa >= 0.40:
        return '<span class="tag mod">moderate</span>'
    return '<span class="tag weak">weak</span>'


def table_html(df, link_col=None):
    head = "".join(f'<th class="{"l" if c in ("group","definition","top_driver","cluster_names","desc") else ""}">{c}</th>'
                   for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cls = "pass" if r.get("PASS", 0) == 1 else "fail"
        tds = []
        for c in df.columns:
            v = r[c]
            al = "l" if c in ("group", "definition", "top_driver", "cluster_names", "desc") else ""
            if c == "kappa":
                v = f"{v:.2f} {verdict(v)}"
            tds.append(f'<td class="{al}">{v}</td>')
        rows.append(f'<tr class="{cls}">' + "".join(tds) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


# ------------------------------------------------------------------- main --
def main():
    f = load()
    X = f[DRIVERS].fillna(0.0)
    g = f.economy.values
    os.makedirs(OUT, exist_ok=True)
    print(f"{len(f)} runs, {f.economy.nunique()} economies, {len(DRIVERS)} drivers")

    allrows = []
    for group, defs in GROUPS.items():
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

                kp, lv, ml, acc, bal = tune(X, y, g, k)
                dt = mk_tree(lv, ml)
                _, _, ung = metrics(dt, X, y, g, grouped=False)
                rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                            class_weight="balanced", random_state=0, n_jobs=-1)
                _, _, rfk = metrics(rf, X, y, g)
                dt.fit(X, y)
                imp = pd.Series(dt.feature_importances_ * 100, index=DRIVERS)
                rff = rf.fit(X, y)
                rimp = pd.Series(rff.feature_importances_ * 100, index=DRIVERS)

                prof = f.groupby(y)[PROFILE].mean()
                prof["n"] = pd.Series(y).value_counts().sort_index()
                prof["share_k_lt_1"] = f.groupby(y).k_small.apply(lambda s: (s < 1).mean())
                cnames = uniq([name_cluster(prof.loc[i]) for i in range(k)])
                prof.insert(0, "name", cnames)

                kdir = os.path.join(gdir, dname, "all", f"k{k:02d}")
                os.makedirs(kdir, exist_ok=True)
                prof.round(3).to_csv(os.path.join(kdir, "cluster_profiles.csv"))
                pd.DataFrame({"feature": DRIVERS,
                              "DT_importance_pct": imp.round(1).values,
                              "RF_importance_pct": rimp.round(1).values}
                             ).sort_values("DT_importance_pct", ascending=False).to_csv(
                    os.path.join(kdir, "feature_importance.csv"), index=False)
                rules = leaf_rules(dt, DRIVERS, cnames)
                with open(os.path.join(kdir, "rules.txt"), "w", encoding="utf-8") as fh:
                    fh.write(f"# {group}/{dname} k={k} | grouped kappa {kp:.3f} "
                             f"| acc {acc:.3f} | baseline {base:.3f}\n")
                    fh.write(f"# clustered on: {', '.join(feats)}\n\n")
                    for r in rules:
                        mark = "" if r["purity"] >= 0.70 else "  [mixed]"
                        fh.write(f"[{r['purity']:.0%}, n={r['n']}] {r['cls']}{mark}\n"
                                 f"    IF {r['rule']}\n")
                fig, ax = plt.subplots(figsize=(17, 9))
                plot_tree(dt, feature_names=DRIVERS, class_names=cnames, filled=True,
                          rounded=True, proportion=True, precision=1, fontsize=7, ax=ax)
                ax.set_title(f"{group}/{dname}  k={k}   baseline {base:.2f} | "
                             f"acc {acc:.2f} | grouped kappa {kp:.2f} | RF kappa {rfk:.2f}",
                             fontsize=12, color=UU_NAVY)
                fig.savefig(os.path.join(kdir, "driver_tree.png"), dpi=170, bbox_inches="tight")
                plt.close(fig)

                top3 = imp.sort_values(ascending=False).head(3)
                with open(os.path.join(kdir, "tree.html"), "w", encoding="utf-8") as fh:
                    fh.write(f"<html><head><meta charset='utf-8'><title>{group}/{dname} k{k}</title>"
                             f"{CSS}</head><body><h1>{group} / {dname} &mdash; k={k}</h1>"
                             f"<div class='note'>{desc}<br>Clustered on: <b>{', '.join(feats)}</b><br>"
                             f"grouped &kappa; <b>{kp:.2f}</b> {verdict(kp)} &middot; accuracy {acc:.2f} "
                             f"&middot; balanced {bal:.2f} &middot; baseline {base:.2f} "
                             f"&middot; RF &kappa; {rfk:.2f} &middot; ungrouped &kappa; {ung:.2f}</div>"
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
                           leaves=lv, min_leaf=ml,
                           top_driver=imp.idxmax(), top_pct=round(imp.max(), 1),
                           dist_pct=round(imp[DIST].sum(), 1),
                           k_small_pct=round(imp["k_small"], 1),
                           degenerate=int(base > 0.80),
                           PASS=int(kp >= 0.60 and base <= 0.80),
                           cluster_names=" | ".join(cnames), desc=desc)
                grows.append(row)
                allrows.append(row)
                print(f"  {group:8s} {dname:14s} k={k:2d}  kappa {kp:5.2f}  base {base:.2f}  "
                      f"sil {sil:.2f}  RF {rfk:5.2f}  top {imp.idxmax():28s} dist {imp[DIST].sum():4.1f}%")

            if sil_curve:
                fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
                ks = [c[0] for c in sil_curve]
                ax[0].plot(ks, [c[2] for c in sil_curve], "o-", color=UU_NAVY)
                ax[0].set_xlabel("k"); ax[0].set_ylabel("inertia"); ax[0].set_title("elbow")
                ax[1].plot(ks, [c[1] for c in sil_curve], "o-", color=UU_TEAL)
                ax[1].set_xlabel("k"); ax[1].set_ylabel("silhouette"); ax[1].set_title("silhouette")
                fig.suptitle(f"{group}/{dname}", color=UU_NAVY)
                fig.tight_layout()
                fig.savefig(os.path.join(gdir, dname, "elbow_silhouette.png"),
                            dpi=150, bbox_inches="tight")
                plt.close(fig)

        G = pd.DataFrame(grows).sort_values("kappa", ascending=False)
        G.to_csv(os.path.join(gdir, "comparison.csv"), index=False)
        cols = ["definition", "k", "silhouette", "baseline", "accuracy", "bal_accuracy",
                "kappa", "kappa_ungrouped", "rf_kappa", "top_driver", "top_pct",
                "dist_pct", "degenerate", "PASS"]
        with open(os.path.join(gdir, "comparison.html"), "w", encoding="utf-8") as fh:
            fh.write(f"<html><head><meta charset='utf-8'><title>{group}</title>{CSS}</head><body>"
                     f"<h1>{group} &mdash; all cases</h1>"
                     f"<div class='note'>Every case is listed. <b>PASS</b> = grouped &kappa; &ge; 0.60 "
                     f"and baseline &le; 0.80; it is a flag, not a filter. &kappa; is grouped by economy; "
                     f"<code>kappa_ungrouped</code> shows what ignoring the paired design would have "
                     f"reported.</div>{table_html(G[cols])}</body></html>")

        # drivers heatmap for the group
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
        ax.set_title(f"{group}: grouped kappa by definition and k", color=UU_NAVY)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(os.path.join(gdir, "kappa_heatmap.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    A = pd.DataFrame(allrows).sort_values("kappa", ascending=False)
    A.to_csv(os.path.join(OUT, "comparison_all.csv"), index=False)
    cols = ["group", "definition", "k", "silhouette", "baseline", "accuracy", "bal_accuracy",
            "kappa", "kappa_ungrouped", "kappa_inflation", "rf_kappa", "dt_vs_rf",
            "top_driver", "top_pct", "k_small_pct", "dist_pct", "degenerate", "PASS"]
    npass = int(A.PASS.sum())
    with open(os.path.join(OUT, "overview.html"), "w", encoding="utf-8") as fh:
        fh.write(f"<html><head><meta charset='utf-8'><title>DT_supplychain v5</title>{CSS}</head><body>"
                 f"<h1>DT_supplychain &mdash; v5 campaign (6000 runs)</h1>"
                 f"<div class='note'><b>{len(A)} cases</b> across {len(GROUPS)} approach groups, "
                 f"k = 3..10, all {len(f)} runs / {f.economy.nunique()} economies. "
                 f"{npass} cases meet &kappa; &ge; 0.60 with baseline &le; 0.80 &mdash; shown as the "
                 f"<b>PASS</b> flag, not used to drop anything.<br>"
                 f"Cross-validation is <b>grouped by economy</b>: the v5 design gives each economy 8 "
                 f"topologies, so ungrouped folds leak. <code>kappa_inflation</code> = ungrouped &minus; "
                 f"grouped is the size of that leak.<br>"
                 f"<code>dist_pct</code> = combined decision-tree importance of d_SL, d_LL, d_SS. "
                 f"Near-zero in supply axes does <b>not</b> mean distance is inert &mdash; within economies "
                 f"its effect is t &asymp; 10; tree importance cannot resolve an effect worth ~2% of "
                 f"within-economy variance.</div>"
                 f"<h2>Groups</h2><ul>" +
                 "".join(f"<li><a href='{gp}/comparison.html'>{gp}</a> &mdash; "
                         f"{len(GROUPS[gp])} definitions</li>" for gp in GROUPS) +
                 f"</ul><h2>All cases, ranked by grouped &kappa;</h2>{table_html(A[cols])}"
                 f"</body></html>")
    print(f"\n{len(A)} cases written to {OUT}")
    print(f"PASS (kappa>=0.60 and baseline<=0.80): {npass}")
    print(A[["group", "definition", "k", "baseline", "kappa", "rf_kappa",
             "top_driver", "dist_pct", "PASS"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
