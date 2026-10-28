"""
Two-stage decentralisation trees for the v5 campaign (DT_uncert edition).

Why two stages: 63% of small-cluster cases are islanded (no pipeline), and an
islanded cluster has self-sufficiency exactly 1 by its hydrogen balance. The
pooled, clipped `small_selfsuff` of dt_v5 therefore mixes a structural yes/no
decision (connect or not) with a continuous operational one (how much to produce
once connected), and k-Means on it separates the mass point at 1 from the tail
rather than economic regimes. Here the two are explained separately:

  Stage 1 (structural, run level): n_SC_connected in {0, 1, 2}, the number of
      small clusters with any pipeline capacity (> 1 MW, same threshold as dt_v5).
  Stage 2 (operational, node level, connected small clusters only): unclipped
      self-sufficiency SS = P / D per small cluster, in rule-based classes
      hub-fed (< 0.05) / partial (0.05-0.99) / self-covering or exporter (>= 0.99).
      Two node-specific drivers are added: distance to the nearest large cluster
      and the realised solar capacity factor of that node.

Classes are rule-based (no k-Means). Drivers are the sampled uncertain parameters
of dt_v5_uncertain; CV is grouped by economy; tuning, scoring and the schematic
leaf HTML are reused from dt_v5 / dt_v5_uncertain.

Output: <RESULTS_FOLDER>/DT_uncert/decentr/C_two_stage/
    definitions.csv, comparison.csv, comparison.html, node_states.csv
    stage1_connectivity/        {cluster_levels.csv, cluster_profiles.csv, details.html,
    stage2_selfsuff_connected/   driver_tree.png, feature_importance.csv, rules.txt, tree.html}

Run:  python dt_v5_decentr_two_stage.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

import config as C
import dt_v5 as V
import dt_v5_uncertain as U
import render_driver_tree_html as R

OUT = os.path.join(U.OUT, "decentr", "C_two_stage")
DRIVERS = U.DRIVERS
NODE_DRIVERS = DRIVERS + ["d_SL_node_min", "solar_cf_node"]
SS_BINS = [-np.inf, 0.05, 0.99, np.inf]

R.FEAT.update({
    "d_SL_node_min": ["Distance to nearest large cluster (node)", "km"],
    "solar_cf_node": ["Solar capacity factor (node)", "-"],
})
R.PAIRS.update({"d_SL_node_min": ["near", "far"], "solar_cf_node": ["low", "high"]})

STAGES = [
    ("stage1_connectivity", "n_SC_connected",
     "Stage 1 (structural, run level): number of small clusters connected to the pipeline "
     "network; islanded clusters have SS = 1 by their hydrogen balance",
     ["0 connected", "1 connected", "2 connected"],
     ["0 of 2 small clusters connected (both islanded)",
      "1 of 2 small clusters connected",
      "2 of 2 small clusters connected"]),
    ("stage2_selfsuff_connected", "SS (connected small clusters, unclipped)",
     "Stage 2 (operational, node level): self-sufficiency P/D of connected small clusters "
     "only, unclipped, rule-based classes",
     ["Hub-fed", "Partial", "Self-covering/exporter"],
     ["SS < 0.05", "0.05 <= SS < 0.99", "SS >= 0.99"]),
]


def node_table(raw):
    """One row per (run, small cluster): connectivity, unclipped SS, node drivers."""
    arcs = [c for c in raw.columns if "_to_" in c and "hydrogenPipeline" in c
            and not c.endswith(("_sum", "_mean", "cost"))]
    rows = []
    for n in V.SMALL:
        P = raw[f"{n}_TOTAL_H2_production"].fillna(0)
        D = (P + raw[f"{n}_hydrogen_import_sum"].fillna(0)
             + raw[f"{n}_hydrogen_network_inflow_sum"].fillna(0)
             - raw[f"{n}_hydrogen_network_outflow_sum"].fillna(0))
        cap = raw[[a for a in arcs if n in a]].fillna(0)
        rows.append(pd.DataFrame({
            "run": raw.run.values, "node": n,
            "connected": (cap > 1).any(axis=1).astype(int).values,
            "pipe_capacity_MW": cap.sum(axis=1).values,
            "SS": (P / D).values,
            "d_SL_node_min": raw[[f"distance_Large_cluster1_to_{n}_km",
                                  f"distance_Large_cluster2_to_{n}_km"]].min(axis=1).values,
            "solar_cf_node": raw[f"solar_cf_realized_{n}"].values,
        }))
    return pd.concat(rows, ignore_index=True)


def write_stage(sdir, name, desc, X, y, groups, drivers, names, ranges, prof, dfm, axis_vals,
                axis_label):
    """Same file set as a dt_v5_uncertain k-folder, for a rule-based target."""
    os.makedirs(sdir, exist_ok=True)
    k = len(names)
    base = pd.Series(y).value_counts(normalize=True).max()
    kp, lv, ml, acc, bal = V.tune(X, y, groups, k)
    dt = V.mk_tree(lv, ml)
    _, _, ung = V.metrics(dt, X, y, groups, grouped=False)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                class_weight="balanced", random_state=0, n_jobs=-1)
    _, _, rfk = V.metrics(rf, X, y, groups)
    dt.fit(X, y)
    rf.fit(X, y)
    imp = pd.Series(dt.feature_importances_ * 100, index=drivers)
    rimp = pd.Series(rf.feature_importances_ * 100, index=drivers)
    rnames = [f"{n} ({r})" for n, r in zip(names, ranges)]

    prof.insert(0, "name", names)
    prof.round(3).to_csv(os.path.join(sdir, "cluster_profiles.csv"))
    pd.DataFrame({"level": range(k), "name": names, "rule": ranges, "axis": axis_label,
                  "range_lo": [axis_vals[y == i].min() for i in range(k)],
                  "range_hi": [axis_vals[y == i].max() for i in range(k)],
                  "mean": [axis_vals[y == i].mean() for i in range(k)],
                  "n": [int((y == i).sum()) for i in range(k)]}
                 ).round(4).to_csv(os.path.join(sdir, "cluster_levels.csv"), index=False)
    pd.DataFrame({"feature": drivers, "DT_importance_pct": imp.round(1).values,
                  "RF_importance_pct": rimp.round(1).values}
                 ).sort_values("DT_importance_pct", ascending=False).to_csv(
        os.path.join(sdir, "feature_importance.csv"), index=False)

    rules = V.leaf_rules(dt, drivers, rnames)
    with open(os.path.join(sdir, "rules.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# decentr/C_two_stage/{name} | grouped kappa {kp:.3f} | acc {acc:.3f} "
                 f"| bal acc {bal:.3f} | baseline {base:.3f} | RF kappa {rfk:.3f} | n={len(y)}\n")
        fh.write(f"# target: {desc}\n")
        fh.write("# drivers: uncertain sampled parameters only (no k_small)"
                 + (" + node distance / node solar CF\n\n" if len(drivers) > len(DRIVERS)
                    else "\n\n"))
        for r in rules:
            mark = "" if r["purity"] >= 0.70 else "  [mixed]"
            fh.write(f"[{r['purity']:.0%}, n={r['n']}] {r['cls']}{mark}\n    IF {r['rule']}\n")

    fig, ax = plt.subplots(figsize=(17, 9))
    plot_tree(dt, feature_names=drivers, class_names=rnames, filled=True, rounded=True,
              proportion=True, precision=2, fontsize=7, ax=ax)
    ax.set_title(f"decentr/C_two_stage/{name}   n={len(y)} | baseline {base:.2f} | "
                 f"acc {acc:.2f} | grouped kappa {kp:.2f} | RF kappa {rfk:.2f}",
                 fontsize=12, color=C.UU_NAVY)
    fig.savefig(os.path.join(sdir, "driver_tree.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # schematic-leaf HTML: dt_v5_uncertain renderer, which reads U.DRIVERS
    hclf = DecisionTreeClassifier(max_leaf_nodes=U.HTML_LEAVES, min_samples_leaf=U.HTML_MIN_LEAF,
                                  min_samples_split=U.HTML_MIN_LEAF, class_weight="balanced",
                                  random_state=V.RS)
    _, _, hkp = V.metrics(hclf, X, y, groups)
    subs = [f"{r} · " + U.cluster_subtitle(prof.loc[i]) for i, r in enumerate(ranges)]
    unit = "small-cluster cases" if len(drivers) > len(DRIVERS) else "optimizations"
    sub_text = (f"{len(y)} {unit}, {len(np.unique(groups))} economies. {U.HTML_LEAVES}-leaf "
                f"tree, grouped &kappa; {hkp:.2f} (baseline {base:.2f}). {desc}.")
    saved = U.DRIVERS
    U.DRIVERS = drivers
    try:
        U.schematic_html(dfm, y, names, subs, os.path.join(sdir, "tree.html"),
                         f"decentr / C_two_stage / {name}: from uncertain parameters to "
                         f"hydrogen infrastructure", sub_text)
    finally:
        U.DRIVERS = saved

    top3 = imp.sort_values(ascending=False).head(3)
    with open(os.path.join(sdir, "details.html"), "w", encoding="utf-8") as fh:
        fh.write(f"<html><head><meta charset='utf-8'><title>C_two_stage {name}</title>"
                 f"{V.CSS}</head><body><h1>decentr / C_two_stage / {name}</h1>"
                 f"<div class='note'>{desc}<br>Classes are rule-based, not k-Means.<br>"
                 f"Drivers: {', '.join(drivers)}.<br>"
                 f"grouped &kappa; <b>{kp:.2f}</b> {V.verdict(kp)} &middot; accuracy {acc:.2f} "
                 f"&middot; balanced {bal:.2f} &middot; baseline {base:.2f} "
                 f"&middot; RF &kappa; {rfk:.2f} &middot; ungrouped &kappa; {ung:.2f}<br>"
                 f"<a href='tree.html'>schematic driver tree</a> "
                 f"({U.HTML_LEAVES} leaves, grouped &kappa; {hkp:.2f})</div>"
                 f"<h2>Class profiles</h2>{prof.round(3).to_html()}"
                 f"<h2>Drivers</h2><p>" +
                 ", ".join(f"{a} {b:.0f}%" for a, b in top3.items()) +
                 f"</p><h2>Leaf rules</h2><pre>" +
                 "\n".join(f"[{r['purity']:.0%}, n={r['n']}] {r['cls']}\n    IF {r['rule']}"
                           for r in rules) +
                 f"</pre><h2>Tree</h2><img src='driver_tree.png' style='width:100%'>"
                 f"</body></html>")

    print(f"\n== {name}  n={len(y)}  leaves {lv} min_leaf {ml}  baseline {base:.3f}  "
          f"acc {acc:.3f}  bal {bal:.3f}  kappa {kp:.3f} (ungrouped {ung:.3f})  "
          f"RF {rfk:.3f}  html8 {hkp:.3f}")
    print("   classes: " + ", ".join(f"{rnames[i]} {int((y == i).sum())}" for i in range(k)))
    print("   top: " + ", ".join(f"{a} {b:.0f}%" for a, b in
                                 imp.sort_values(ascending=False).head(5).items()))
    for r in rules:
        print(f"   [{r['purity']:.0%}, n={r['n']}] {r['cls']}  IF {r['rule']}")

    return dict(group="decentr", definition=f"C_two_stage/{name}", k=k, n=len(y),
                silhouette=np.nan, baseline=round(base, 3), accuracy=acc, bal_accuracy=bal,
                kappa=kp, kappa_ungrouped=ung, kappa_inflation=round(ung - kp, 3),
                rf_kappa=rfk, dt_vs_rf=round(kp - rfk, 3), kappa_html8=hkp,
                leaves=lv, min_leaf=ml, top_driver=imp.idxmax(), top_pct=round(imp.max(), 1),
                dist_pct=round(imp[[c for c in V.DIST + ["d_SL_node_min"] if c in imp]].sum(), 1),
                degenerate=int(base > 0.80), PASS=int(kp >= 0.60 and base <= 0.80),
                cluster_names=" | ".join(names), desc=desc)


def main():
    f = V.load(C.EXTRACTED_RESULTS)
    raw = pd.read_excel(C.EXTRACTED_RESULTS, sheet_name="Results")
    dfm = raw.copy()
    dfm["import_intensity"] = f["imported"].values
    nodes = node_table(raw)
    os.makedirs(OUT, exist_ok=True)
    nodes.to_csv(os.path.join(OUT, "node_states.csv"), index=False)
    print(f"{len(f)} runs, {f.economy.nunique()} economies -> {OUT}")

    rows = []
    # ---- stage 1: connectivity, run level
    name, axis, desc, names, ranges = STAGES[0]
    nconn = nodes.groupby("run").connected.sum().reindex(f.run).values.astype(int)
    prof = f.groupby(nconn)[V.PROFILE].mean()
    prof["n"] = pd.Series(nconn).value_counts().sort_index()
    rows.append(write_stage(os.path.join(OUT, name), name, desc, f[DRIVERS].fillna(0.0), nconn,
                            f.economy.values, DRIVERS, names, ranges, prof, dfm,
                            nconn.astype(float), axis))

    # ---- stage 2: unclipped SS of connected small clusters, node level
    name, axis, desc, names, ranges = STAGES[1]
    nd = nodes.merge(f[["run", "economy"] + DRIVERS + V.PROFILE], on="run")
    keep = (nd.connected == 1).values
    nd = nd[keep].reset_index(drop=True)
    y2 = pd.cut(nd.SS, SS_BINS, labels=False, right=False).astype(int).values
    prof = nd.groupby(y2)[["SS", "pipe_capacity_MW", "d_SL_node_min", "solar_cf_node"]
                          + V.PROFILE].mean()
    prof["n"] = pd.Series(y2).value_counts().sort_index()
    dfm2 = dfm.set_index("run").loc[nd.run].reset_index()   # full network of each node's run
    for c in ["d_SL_node_min", "solar_cf_node"]:
        dfm2[c] = nd[c].values
    rows.append(write_stage(os.path.join(OUT, name), name, desc, nd[NODE_DRIVERS].fillna(0.0), y2,
                            nd.economy.values, NODE_DRIVERS, names, ranges, prof, dfm2,
                            nd.SS.values, axis))

    pd.DataFrame([{"definition": f"C_two_stage/{s[0]}", "features": s[1], "description": s[2]}
                  for s in STAGES]).to_csv(os.path.join(OUT, "definitions.csv"), index=False)
    G = pd.DataFrame(rows)
    G.to_csv(os.path.join(OUT, "comparison.csv"), index=False)
    cols = ["definition", "k", "n", "baseline", "accuracy", "bal_accuracy", "kappa",
            "kappa_html8", "kappa_ungrouped", "rf_kappa", "top_driver", "top_pct",
            "dist_pct", "degenerate", "PASS"]
    links = "".join(f"<li><a href='{s[0]}/details.html'>{s[0]}</a> &middot; "
                    f"<a href='{s[0]}/tree.html'>schematic tree</a> &mdash; {s[2]}</li>"
                    for s in STAGES)
    with open(os.path.join(OUT, "comparison.html"), "w", encoding="utf-8") as fh:
        fh.write(f"<html><head><meta charset='utf-8'><title>decentr C_two_stage</title>{V.CSS}"
                 f"</head><body><h1>decentr / C_two_stage &mdash; connectivity, then "
                 f"self-sufficiency</h1><div class='note'>An islanded small cluster has "
                 f"SS = 1 by its hydrogen balance, so the pooled clipped "
                 f"<code>small_selfsuff</code> mixes the connect-or-not decision with the "
                 f"production level. Stage 1 explains connectivity (run level); stage 2 "
                 f"explains unclipped SS of connected small clusters only (node level). "
                 f"Classes are rule-based. CV grouped by economy. <b>PASS</b> = grouped "
                 f"&kappa; &ge; 0.60 and baseline &le; 0.80.</div><ul>{links}</ul>"
                 f"{V.table_html(G[cols])}</body></html>")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
