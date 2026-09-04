"""
System-level supply-mix clustering (shares of TOTAL demand).

Unlike the small-cluster self-sufficiency view (decentr/supply), here every source is a
share of total system demand, so they are mutually exclusive and sum to ~1:
    small_share  = small-cluster production / total demand   (decentralised)
    large_share  = large-cluster production / total demand   (centralised)
    import_share = imports / total demand
plus transport_intensity = H2 moved over the network / total demand.

NOTE: small_share is tiny (<=~16%) because the small clusters are small — so at the
system scale the meaningful axis is central-production vs import, not decentralisation.

Same pipeline as strat: k-Means (k=2..10, capped), DT tuned (leaf + min_leaf) with Cohen
kappa + balanced accuracy + RandomForest, degenerate flag, schematic HTML named by system
supply mix, feature_importance, cluster_profiles, comparison.html (legend + verdict).

Outputs: DT_supplychain/sysmix/     (short path)
Run:  python system_supply_analysis.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from pure_leaves import leaf_rules
from robust_trees import robust_run_ids, ROBUST_SCENARIO
from combined_strategies import best_tree2, _tree2, unique
from decentralization_full_analysis import cv_metrics, schematic_html

OUT = r"X:\3000_simulations_24_may_2026\Snellius_3000_simulations\DT_supplychain\sysmix"
KS = range(2, 11)
PURITY = 0.70
NODES = ["Large_cluster1", "Large_cluster2", "Small_cluster1", "Small_cluster2"]
PROFILE = ["small_share", "large_share", "import_share", "transport_intensity"]

STRATEGIES = [
    ("sysmix",    ["small_share", "large_share", "import_share"],
     "System supply mix: small vs large production vs import (shares of total demand)", "core"),
    ("cent_imp",  ["large_share", "import_share"],
     "Central production vs import (small share is negligible at system scale)", "core"),
    ("transport", ["transport_intensity"],
     "How much H2 is transported / total demand", "easy"),
    ("mix_trans", ["small_share", "import_share", "transport_intensity"],
     "Supply mix + transport intensity", "core+"),
]


def add_shares(feat, raw):
    for n in NODES:
        for s in ["_TOTAL_H2_production", "_hydrogen_import_sum"]:
            raw[n + s] = raw[n + s].fillna(0)
    m = feat.merge(raw, on="run", how="left", suffixes=("", "_r"))
    sp = m["Small_cluster1_TOTAL_H2_production"] + m["Small_cluster2_TOTAL_H2_production"]
    lp = m["Large_cluster1_TOTAL_H2_production"] + m["Large_cluster2_TOTAL_H2_production"]
    imp = sum(m[n + "_hydrogen_import_sum"] for n in NODES)
    tot = (sp + lp + imp).replace(0, np.nan)
    feat = feat.copy()
    feat["small_share"] = (sp / tot).fillna(0).values
    feat["large_share"] = (lp / tot).fillna(0).values
    feat["import_share"] = (imp / tot).fillna(0).values
    return feat


def name_sys(p):
    imp, sm, lg = p.get("import_share", 0), p.get("small_share", 0), p.get("large_share", 0)
    base = ("Import-heavy" if imp > 0.30 else "Fully-central" if lg > 0.88
            else "Central+import" if imp > 0.12 else "Central")
    return base + (" +local" if sm > 0.07 else "")


def cluster_name(p, feats):
    parts = [f"L{p.get('large_share',0)*100:.0f}%", f"I{p.get('import_share',0)*100:.0f}%",
             f"S{p.get('small_share',0)*100:.0f}%"]
    if "transport_intensity" in feats:
        parts.append(f"tr{p.get('transport_intensity',0)*100:.0f}%")
    return f"{name_sys(p)} · " + " ".join(parts)


def cluster(df, feats, k):
    Xc = StandardScaler().fit_transform(df[feats].fillna(0))
    lab = KMeans(k, random_state=C.RANDOM_STATE, n_init=10).fit_predict(Xc)
    order = np.argsort([df["large_share"].values[lab == c].mean() for c in range(k)])
    remap = {o: n for n, o in enumerate(order)}
    y = np.array([remap[c] for c in lab])
    sil = silhouette_score(Xc, lab) if k > 1 else 0.0
    profs = [df.loc[y == i, PROFILE].mean() for i in range(k)]
    return y, sil, profs


def run_case(feat, raw, name, feats, tag):
    dfm = feat.merge(raw, on="run", how="left", suffixes=("", "_raw"))
    inputs = [c for c in C.DRIVER_INPUTS if c in feat.columns]
    X = feat[inputs].fillna(0).reset_index(drop=True)
    nun = feat[feats].round(3).drop_duplicates().shape[0]
    rows = []
    for k in [k for k in KS if k <= nun]:
        y, sil, profs = cluster(feat, feats, k)
        names = unique([cluster_name(p, feats) for p in profs])
        subs = {names[i]: name_sys(profs[i]) for i in range(k)}
        base = pd.Series(y).value_counts(normalize=True).max()
        lv, ml, _ = best_tree2(X, y, k)
        dtcv, dtbal, dtkap = cv_metrics(_tree2(lv, ml), X, y)
        rfcv = cross_val_score(RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                               class_weight="balanced", random_state=0, n_jobs=-1), X, y, cv=5).mean()
        rows.append({"def": name, "subset": tag, "k": k, "n": len(feat), "silhouette": round(sil, 3),
                     "baseline": round(base, 3), "DT_leaves": (lv or 0), "DT_minleaf": ml,
                     "DT_CV": dtcv, "DT_bal_acc": dtbal, "DT_kappa": dtkap,
                     "RF_CV": round(rfcv, 3), "degenerate": int(base > 0.80)})
        kd = os.path.join(OUT, name, tag, f"k{k:02d}"); os.makedirs(kd, exist_ok=True)
        full = _tree2(lv, ml).fit(X, y)
        fig, ax = plt.subplots(figsize=(16, 9))
        plot_tree(full, feature_names=inputs, class_names=names, filled=True, rounded=True,
                  proportion=True, impurity=True, precision=1, fontsize=8, ax=ax)
        ax.set_title(f"{name} [{tag}] k={k} | base {base:.2f} acc {dtcv:.2f} κ {dtkap:.2f} RF {rfcv:.2f}",
                     fontsize=12)
        fig.savefig(os.path.join(kd, "driver_tree.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
        with open(os.path.join(kd, "rules.txt"), "w", encoding="utf-8") as f:
            for r in leaf_rules(full, inputs, names):
                mk = "" if r["purity"] >= PURITY else " [mixed]"
                f.write(f"[{r['purity']:.0%}, n={r['n']}] {r['class']}{mk} IF {r['rule']}\n")
        rf_full = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced",
                                         random_state=0, n_jobs=-1).fit(X, y)
        pd.DataFrame({"feature": inputs, "DT_importance_pct": (full.feature_importances_*100).round(1),
                      "RF_importance_pct": (rf_full.feature_importances_*100).round(1)}
                     ).sort_values("DT_importance_pct", ascending=False).to_csv(
            os.path.join(kd, "feature_importance.csv"), index=False)
        pd.DataFrame([dict(cluster=names[i], n=int((y == i).sum()),
                     **{c: round(float(profs[i][c]), 3) for c in PROFILE}) for i in range(k)]
                     ).to_csv(os.path.join(kd, "cluster_profiles.csv"), index=False)
        schematic_html(dfm, y, names, os.path.join(kd, "tree.html"), f"{name} {tag} k={k}", subs=subs)
    print(f"  {name} [{tag}]: done")
    return rows


def comparison_html(allrows, defs):
    df = pd.DataFrame(allrows); df.to_csv(os.path.join(OUT, "comparison.csv"), index=False)
    leg = "".join(f"<tr><td><b>{c}</b></td><td>{x}</td><td>{cc}</td><td><code>{', '.join(fs)}</code></td></tr>"
                  for c, fs, cc, x in defs)
    def col(v): t=max(0,min(1,(v-0.2)/0.75)); return f"background:rgb({int(255-t*156)},{int(255-t*90)},{int(255-t*108)})"
    def verdict(kp): return ('strong','#8fd19e') if kp>=0.4 else ('moderate','#ffe08a') if kp>=0.2 else ('weak','#f2c6a0') if kp>=0.05 else ('no skill','#f2b6b6')
    body = []
    for _, r in df.iterrows():
        deg = int(r.degenerate) == 1
        nd = df[(df["def"] == r["def"]) & (df.subset == r.subset) & (df.degenerate == 0)]
        best = (not deg) and len(nd) and r.DT_kappa >= nd.DT_kappa.max() - 1e-9
        st = " style='opacity:.4;background:#f2f2f2'" if deg else ""
        vt, vc = ("degenerate", "#f2b6b6") if deg else verdict(r.DT_kappa)
        body.append(f"<tr{st}><td>{r['def']}</td><td>{r['subset']}</td><td>{r['k']}</td><td>{r['n']}</td>"
                    f"<td>{r['silhouette']}</td><td>{r['baseline']}{' ⚠' if deg else ''}</td>"
                    f"<td style='{col(r.DT_CV)}'>{r.DT_CV}{' ★' if best else ''}</td><td>{r.DT_bal_acc}</td>"
                    f"<td style='{col(r.DT_kappa)}'>{r.DT_kappa}</td><td style='{col(r.RF_CV)}'>{r.RF_CV}</td>"
                    f"<td style='background:{vc}'>{vt}</td></tr>")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>System supply mix — comparison</title>
<style>body{{font-family:Inter,Calibri,sans-serif;margin:26px;color:#161D41}}h1{{font-size:20px}}h2{{font-size:15px}}
.sub{{color:#8a8f9c;font-size:13px;max-width:960px;margin-bottom:12px}}table{{border-collapse:collapse;font-size:12px;margin-bottom:16px}}
th,td{{border:1px solid #d4d8e0;padding:4px 9px;text-align:center}}th{{background:#161D41;color:#fff}}code{{font-size:11px}}</style></head><body>
<h1>System-level supply-mix clustering</h1>
<div class="sub">Shares of <b>total demand</b> (mutually exclusive, sum≈1): small production (decentralised),
large production (central), import — plus transport intensity. NB small_share is &le;~16% (small clusters are
small), so at system scale the axis is <b>central vs import</b>, not decentralisation. Cohen&nbsp;κ = skill above
baseline. ★ = best κ per def/subset. Greyed (⚠) = degenerate (&gt;80% one cluster).</div>
<h2>Clustering ideas</h2>
<table><thead><tr><th>folder</th><th>type</th><th>idea</th><th>features</th></tr></thead><tbody>{leg}</tbody></table>
<h2>Accuracy (Cohen κ)</h2>
<table><thead><tr><th>def</th><th>subset</th><th>k</th><th>n</th><th>silhouette</th><th>baseline</th>
<th>DT acc</th><th>bal.acc</th><th>Cohen&nbsp;κ</th><th>RF CV</th><th>verdict</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></body></html>"""
    open(os.path.join(OUT, "comparison.html"), "w", encoding="utf-8").write(html)
    pd.DataFrame([{"folder": c, "type": x, "concept": cc, "features": ";".join(fs)} for c, fs, cc, x in defs]
                 ).to_csv(os.path.join(OUT, "definitions.csv"), index=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    feat0 = pd.read_csv(C.FEATURES_CSV); raw = pd.read_excel(C.EXTRACTED_RESULTS)
    feat = add_shares(feat0, raw)
    rob, *_ = robust_run_ids(ROBUST_SCENARIO)
    print("share ranges:", {c: (round(feat[c].min(), 2), round(feat[c].max(), 2)) for c in
                             ["small_share", "large_share", "import_share"]})
    allrows = []
    for name, feats, _c, _x in STRATEGIES:
        for tag, mask in [("all", feat["run"].notna()), ("robust", feat["run"].isin(rob))]:
            allrows += run_case(feat[mask].reset_index(drop=True), raw, name, feats, tag)
    comparison_html(allrows, STRATEGIES)
    df = pd.DataFrame(allrows); inf = df[df.degenerate == 0]
    print("\nBEST by kappa per strategy/subset:")
    for (nm, sub), g in inf.groupby(["def", "subset"]):
        r = g.loc[g.DT_kappa.idxmax()]
        print(f"  {nm:<10} {sub:<7} k={int(r.k)} acc={r.DT_CV} κ={r.DT_kappa}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
