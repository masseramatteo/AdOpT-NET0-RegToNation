"""
Full TOPOLOGY-only analysis (mirror of the decentralisation one, but multi-D).

Cluster on the pipeline-topology feature set (k-Means, k=2..10), for two definitions:
  T_A "connectivity/scale"  = [n_pipelines, degree_small1, degree_small2, LL_pipe]
  T_B "structure/character" = [LL_pipe, SS_pipe, meshed]
and two subsets (all, robust). Clusters are NAMED by their topology character
(backbone / regional-meshed / satellite / sparse). Then predict the topology cluster
from the sampled inputs with a decision tree (leaf tuned for accuracy+readability) and
a RandomForest for the ceiling. Saves per case: driver_tree.png, tree.html (schematic,
named by topology), rules.txt, feature_importance.csv/png, cluster_profiles.csv.
Master pages: comparison.html (accuracy) + drivers.html (feature-importance heatmap of
the >90% cases).

Outputs: DT_supplychain/topo/     (short path -> stays under Windows MAX_PATH)
Run:  python topology_full_analysis.py
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
from sklearn.model_selection import cross_val_score, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import render_driver_tree_html as R
from pure_leaves import leaf_rules
from robust_trees import robust_run_ids, ROBUST_SCENARIO
from decentralization_full_analysis import best_tree, _tree, schematic_html, cv_metrics

OUT = r"X:\3000_simulations_24_may_2026\Snellius_3000_simulations\DT_supplychain\topo"
KS = range(2, 11)
PURITY = 0.70
PROFILE_COLS = ["n_pipelines", "LL_pipe", "SS_pipe", "meshed", "highP_share",
                "degree_small1", "degree_small2"]
# topology clustering ideas, from easy (1-D) to complex (full topology).
# (folder, features clustered on, concept, complexity)
TOPO_DEFS = [
    ("count",     ["n_pipelines"],
     "How many pipelines are built (1-D)", "easy"),
    ("backbone",  ["LL_pipe"],
     "Is a large-large backbone present?", "easy"),
    ("meshsat",   ["SS_pipe", "meshed"],
     "Satellite vs regional/meshed (small-small link / meshed)", "easy"),
    ("connect",   ["n_pipelines", "degree_small1", "degree_small2", "LL_pipe"],
     "Connectivity & scale: #pipelines + small-cluster degrees + backbone", "medium"),
    ("structure", ["LL_pipe", "SS_pipe", "meshed"],
     "Structural character: backbone / regional / meshed", "medium"),
    ("full",      ["n_pipelines", "LL_pipe", "SS_pipe", "meshed", "highP_share",
                   "degree_small1", "degree_small2"],
     "Full topology: counts + structure + pressure class", "complex"),
]


def name_topo(p):
    mesh = p.get("SS_pipe", 0) > 0.5 or p.get("meshed", 0) > 0.5
    ll = p.get("LL_pipe", 0) > 0.5
    npc = p.get("n_pipelines", 0)
    if mesh and ll: return "Meshed+backbone"
    if mesh: return "Regional/meshed"
    if ll and npc > 2.5: return "Backbone+satellites"
    if ll: return "Backbone only"
    if npc < 2.2: return "Sparse"
    return "Satellite"


def unique(names):
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


def cluster_topo(df, feats, k):
    Xc = StandardScaler().fit_transform(df[feats].fillna(0))
    km = KMeans(k, random_state=C.RANDOM_STATE, n_init=10).fit(Xc)
    lab = km.labels_
    order = np.argsort([df["n_pipelines"].values[lab == c].mean() for c in range(k)])
    remap = {old: new for new, old in enumerate(order)}
    y = np.array([remap[c] for c in lab])
    sil = silhouette_score(Xc, lab) if k > 1 else 0.0
    profs = [df.loc[y == i, PROFILE_COLS].mean() for i in range(k)]
    return y, sil, profs


def run_case(feat, raw, name, feats, tag):
    dfm = feat.merge(raw, on="run", how="left", suffixes=("", "_raw"))
    inputs = [c for c in C.DRIVER_INPUTS if c in feat.columns]
    Xdrv = feat[inputs].fillna(0).reset_index(drop=True)
    # cap k to the number of distinct feature combinations (avoid empty clusters)
    n_unique = feat[feats].round(3).drop_duplicates().shape[0]
    ks = [k for k in KS if k <= n_unique]
    rows = []
    for k in ks:
        y, sil, profs = cluster_topo(feat, feats, k)
        names = unique([name_topo(p) for p in profs])
        subs = {names[i]: f"{profs[i]['n_pipelines']:.1f} pipes · "
                          f"{'mesh' if (profs[i]['meshed']>0.5 or profs[i]['SS_pipe']>0.5) else 'radial'}"
                          f"{' · LL' if profs[i]['LL_pipe']>0.5 else ''}" for i in range(k)}
        base = pd.Series(y).value_counts(normalize=True).max()
        lv, _ = best_tree(Xdrv, y, k)
        dtcv, dtbal, dtkap = cv_metrics(_tree(lv), Xdrv, y)   # acc / balanced acc / Cohen kappa
        tr, te = train_test_split(np.arange(len(feat)), test_size=0.1, random_state=42, stratify=y)
        clf = _tree(lv).fit(Xdrv.iloc[tr], y[tr]); dttest = clf.score(Xdrv.iloc[te], y[te])
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced",
                                    random_state=0, n_jobs=-1)
        rfcv = cross_val_score(rf, Xdrv, y, cv=5).mean()
        rows.append({"def": name, "subset": tag, "k": k, "n": len(feat), "silhouette": round(sil, 3),
                     "baseline": round(base, 3), "DT_leaves": lv, "DT_CV": dtcv, "DT_bal_acc": dtbal,
                     "DT_kappa": dtkap, "DT_test": round(dttest, 3), "DT_lift": round(dtcv - base, 3),
                     "RF_CV": round(rfcv, 3),
                     "degenerate": int(base > 0.80)})   # dominant cluster >80% -> not informative

        kd = os.path.join(OUT, name, tag, f"k{k:02d}"); os.makedirs(kd, exist_ok=True)
        cls = [f"{names[i]}" for i in range(k)]
        full = _tree(lv).fit(Xdrv, y)
        fig, ax = plt.subplots(figsize=(16, 9))
        plot_tree(full, feature_names=inputs, class_names=cls, filled=True, rounded=True,
                  proportion=True, impurity=True, precision=1, fontsize=8, ax=ax)
        ax.set_title(f"topology {name} [{tag}] k={k} leaves={lv} | base {base:.2f} DT_CV {dtcv:.2f} "
                     f"test {dttest:.2f} RF {rfcv:.2f}", fontsize=12)
        fig.savefig(os.path.join(kd, "driver_tree.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
        with open(os.path.join(kd, "rules.txt"), "w", encoding="utf-8") as f:
            for r in leaf_rules(full, inputs, cls):
                mk = "" if r["purity"] >= PURITY else " [mixed]"
                f.write(f"[{r['purity']:.0%}, n={r['n']}] {r['class']}{mk} IF {r['rule']}\n")
        # importance
        rf_full = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced",
                                         random_state=0, n_jobs=-1).fit(Xdrv, y)
        pd.DataFrame({"feature": inputs, "DT_importance_pct": (full.feature_importances_*100).round(1),
                      "RF_importance_pct": (rf_full.feature_importances_*100).round(1)}
                     ).sort_values("DT_importance_pct", ascending=False).to_csv(
            os.path.join(kd, "feature_importance.csv"), index=False)
        # cluster profiles
        pd.DataFrame([dict(cluster=names[i], n=int((y == i).sum()), **{c: round(float(profs[i][c]), 3)
                     for c in PROFILE_COLS}) for i in range(k)]).to_csv(
            os.path.join(kd, "cluster_profiles.csv"), index=False)
        # schematic html named by topology
        schematic_html(dfm, y, names, os.path.join(kd, "tree.html"),
                       f"topology {name} {tag} k={k}", subs=subs)
    print(f"  topology {name} [{tag}]: done k=2..10")
    return rows


def comparison_html(allrows, defs):
    df = pd.DataFrame(allrows); df.to_csv(os.path.join(OUT, "comparison.csv"), index=False)
    # legend: which clustering idea each folder encodes
    legrows = "".join(
        f"<tr><td><b>{code}</b></td><td>{complx}</td><td>{concept}</td>"
        f"<td><code>{', '.join(feats)}</code></td></tr>"
        for code, feats, concept, complx in defs)
    legend = ("<h2 style='font-size:15px'>Clustering ideas per folder</h2>"
              "<table><thead><tr><th>folder</th><th>complexity</th><th>idea</th>"
              "<th>features clustered on</th></tr></thead><tbody>" + legrows + "</tbody></table><br>")
    pd.DataFrame([{"folder": c, "complexity": x, "concept": cc, "features": ";".join(fs)}
                 for c, fs, cc, x in defs]).to_csv(os.path.join(OUT, "definitions.csv"), index=False)
    def col(v, lo=0.2, hi=0.9):
        t = max(0, min(1, (v-lo)/(hi-lo))); r=int(255-t*156); g=int(255-t*90); b=int(255-t*108)
        return f"background:rgb({r},{g},{b})"
    body = []
    for _, r in df.iterrows():
        deg = int(r.get("degenerate", 0)) == 1
        nd = df[(df["def"] == r["def"]) & (df.subset == r.subset) & (df.degenerate == 0)]
        best = (not deg) and len(nd) and r["DT_CV"] >= nd["DT_CV"].max() - 1e-9
        rowstyle = " style='opacity:.4;background:#f2f2f2' title='degenerate: dominant cluster >80% — not informative'" if deg else ""
        body.append(f"<tr{rowstyle}><td>{r['def']}</td><td>{r['subset']}</td><td>{r['k']}</td><td>{r['n']}</td>"
                    f"<td>{r['silhouette']}</td><td>{r['baseline']}{' ⚠' if deg else ''}</td><td>{r['DT_leaves']}</td>"
                    f"<td style='{col(r['DT_CV'])}'>{r['DT_CV']}{' ★' if best else ''}</td><td>{r['DT_bal_acc']}</td>"
                    f"<td style='{col(r['DT_kappa'], 0, 1)}'>{r['DT_kappa']}</td>"
                    f"<td>{r['DT_test']}</td><td>{r['DT_lift']}</td><td style='{col(r['RF_CV'])}'>{r['RF_CV']}</td></tr>")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Topology — DT comparison</title>
<style>body{{font-family:Inter,Calibri,sans-serif;margin:26px;color:#161D41}}h1{{font-size:20px}}
.sub{{color:#8a8f9c;font-size:13px;max-width:940px;margin-bottom:14px}}table{{border-collapse:collapse;font-size:12px}}
th,td{{border:1px solid #d4d8e0;padding:4px 9px;text-align:center}}th{{background:#161D41;color:#fff}}
tr:hover{{outline:2px solid #63A593}}a{{color:#63A593}}</style></head><body>
<h1>Topology-only decision trees — comparison</h1>
<p><a href="drivers.html">&rarr; Driver consistency across &gt;90% cases</a></p>
<div class="sub">Cluster on pipeline topology (several ideas, easy&rarr;complex, see legend below), k=2..10, subsets
all / robust. Decision tree predicts the topology cluster from sampled inputs; leaf tuned for accuracy+readability.
RandomForest = ceiling. ★ = best DT_CV per def/subset (informative cases only). Green = higher acc.
<b>Greyed rows (⚠) = degenerate</b>: one cluster holds &gt;80% of runs (e.g. robust designs converge on ~3
pipelines), so the label is near-constant and the tree is uninformative. k is capped per idea to its number of
distinct feature combinations (no empty clusters). Per-k folders: driver_tree.png, tree.html (schematic, named by
topology), rules.txt, feature_importance.csv, cluster_profiles.csv.</div>
{legend}
<table><thead><tr><th>def</th><th>subset</th><th>k</th><th>n</th><th>silhouette</th><th>baseline</th>
<th>DT leaves</th><th>DT acc</th><th>bal.acc</th><th>Cohen&nbsp;&kappa;</th><th>DT test</th><th>DT lift</th><th>RF CV</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></body></html>"""
    open(os.path.join(OUT, "comparison.html"), "w", encoding="utf-8").write(html)
    return df


def drivers_html():
    d = pd.read_csv(os.path.join(OUT, "comparison.csv"))
    hi = d[(d.DT_CV > 0.90) & (d.degenerate == 0)].sort_values(["def", "subset", "k"])
    if len(hi) == 0:  # fall back to best informative cases
        hi = d[d.degenerate == 0].sort_values("DT_CV", ascending=False).head(8)
    cases, imps = [], {}
    for _, r in hi.iterrows():
        imp = pd.read_csv(os.path.join(OUT, r["def"], r["subset"], f"k{int(r['k']):02d}",
                          "feature_importance.csv")).set_index("feature")["DT_importance_pct"]
        tag = f"{r['def']}/{r['subset']}/k{int(r['k'])}"; cases.append((tag, round(r["DT_CV"], 2))); imps[tag] = imp
    M = pd.DataFrame({t: imps[t] for t, _ in cases}).fillna(0.0); M["MEAN"] = M.mean(axis=1)
    M = M.sort_values("MEAN", ascending=False)
    def cc(v): t=max(0,min(1,v/60)); r=int(255-t*156);g=int(255-t*90);b=int(255-t*108); return f"background:rgb({r},{g},{b})"
    head = "".join(f"<th>{t}<br><span style='font-weight:400;font-size:10px'>{cv:.2f}</span></th>" for t,cv in cases)+"<th>MEAN</th>"
    body = []
    for feat, row in M.iterrows():
        tds = "".join(f"<td style='{cc(row[t])}'>{row[t]:.0f}</td>" for t,_ in cases)
        tds += f"<td style='{cc(row['MEAN'])};font-weight:700'>{row['MEAN']:.0f}</td>"
        hl = " style='background:#FFCD00'" if ("distance" in feat) else ""
        body.append(f"<tr><th{hl}>{feat}</th>{tds}</tr>")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Topology drivers</title>
<style>body{{font-family:Inter,Calibri,sans-serif;margin:26px;color:#161D41}}h1{{font-size:19px}}
table{{border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #d4d8e0;padding:4px 8px;text-align:center}}
thead th{{background:#161D41;color:#fff}}tbody th{{background:#eef0f4;text-align:left;white-space:nowrap}}a{{color:#63A593}}
.sub{{color:#8a8f9c;font-size:13px;max-width:940px}}</style></head><body>
<p><a href="comparison.html">&larr; back to accuracy comparison</a></p>
<h1>Which inputs drive the topology? (cases with DT CV &gt; 0.90; distance rows highlighted)</h1>
<div class="sub">Feature importance (%) of the topology driver tree per high-accuracy case. Contrast with the
decentralisation analysis, where distances were ~0 — here we expect inter-cluster <b>distances</b> to matter.</div>
<table><thead><tr><th>feature \\ case</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></body></html>"""
    open(os.path.join(OUT, "drivers.html"), "w", encoding="utf-8").write(html)
    print(f"drivers.html: {len(cases)} cases")


def main():
    os.makedirs(OUT, exist_ok=True)
    feat = pd.read_csv(C.FEATURES_CSV)
    raw = pd.read_excel(C.EXTRACTED_RESULTS)
    rob, *_ = robust_run_ids(ROBUST_SCENARIO)
    allrows = []
    for name, feats, _concept, _cx in TOPO_DEFS:
        for tag, mask in [("all", feat["run"].notna()), ("robust", feat["run"].isin(rob))]:
            fsub = feat[mask].reset_index(drop=True)
            allrows += run_case(fsub, raw, name, feats, tag)
    comparison_html(allrows, TOPO_DEFS)
    drivers_html()
    print(f"\n-> {OUT}")
    print(pd.DataFrame(allrows).groupby(["def", "subset"]).apply(
        lambda g: g.loc[g.DT_CV.idxmax(), ["k", "DT_CV", "DT_test", "DT_lift", "RF_CV"]]).to_string())


if __name__ == "__main__":
    main()
