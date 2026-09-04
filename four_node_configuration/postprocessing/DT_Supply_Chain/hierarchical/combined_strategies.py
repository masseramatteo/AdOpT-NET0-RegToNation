"""
Combined clustering strategies — try many MEANINGFUL clustering ideas that connect the
predictable dimensions (decentralisation + import + network scale), and find which
ones give a cluster typology that a decision tree can predict from inputs with high
accuracy and lift.

Each strategy = a feature set with a real interpretation. Same pipeline as decentr/topo:
k-Means (k=2..10, capped to distinct combos), DT (leaf tuned) + RandomForest, degenerate
flag (dominant cluster >80%), per-case tree.html schematic (named by supply x topology),
feature_importance, cluster_profiles. Master comparison.html (legend + accuracy) +
drivers.html.

Outputs: DT_supplychain/strat/       (short path)
Run:  python combined_strategies.py
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

OUT = r"X:\3000_simulations_24_may_2026\Snellius_3000_simulations\DT_supplychain\strat"
KS = range(2, 11)
PURITY = 0.70
PROFILE = ["local_self_sufficiency", "import_intensity", "transport_intensity",
           "storage_large_share", "n_pipelines", "LL_pipe", "SS_pipe", "meshed"]

# meaningful clustering strategies (folder, features, concept, complexity)
STRATEGIES = [
    ("supply",       ["local_self_sufficiency", "import_intensity"],
     "Supply origin: local vs central vs import", "core"),
    ("supply_scale", ["local_self_sufficiency", "import_intensity", "n_pipelines"],
     "Supply origin + network scale (#pipelines)", "core+"),
    ("decentr_imp",  ["local_self_sufficiency", "import_intensity", "storage_large_share"],
     "Supply origin + storage siting", "core+"),
    ("import_only",  ["import_intensity"],
     "Import intensity alone (import / total demand)", "easy"),
    ("decentr_bb",   ["local_self_sufficiency", "LL_pipe"],
     "Local production + backbone presence", "core+"),
    ("decentr_trans",["local_self_sufficiency", "transport_intensity"],
     "Local production + transport volume", "core+"),
    ("full",         ["local_self_sufficiency", "import_intensity", "transport_intensity",
                      "storage_large_share", "n_pipelines", "LL_pipe", "SS_pipe", "meshed"],
     "Full infrastructure typology (all outputs of interest)", "complex"),
]


def _fmt(f, v):
    """Compact, meaningful label piece for a feature's cluster-mean value."""
    m = {
        "local_self_sufficiency": lambda v: f"loc{v*100:.0f}%",
        "import_intensity":       lambda v: f"imp{v*100:.0f}%",
        "transport_intensity":    lambda v: f"tr{v*100:.0f}%",
        "storage_large_share":    lambda v: f"stoL{v*100:.0f}%",
        "highP_share":            lambda v: f"hiP{v*100:.0f}%",
        "n_pipelines":            lambda v: f"{v:.1f}pipes",
        "LL_pipe":                lambda v: ("BB" if v > 0.5 else "noBB"),
        "SS_pipe":                lambda v: ("SS" if v > 0.5 else ""),
        "meshed":                 lambda v: ("mesh" if v > 0.5 else ""),
        "degree_small1":          lambda v: f"dS1={v:.1f}",
        "degree_small2":          lambda v: f"dS2={v:.1f}",
    }
    return m.get(f, lambda v: f"{v:.2f}")(v)


def name_from_feats(p, feats):
    """Cluster name from the mean values of the strategy's (first ~3) clustered features.
    Value-based -> always distinct and directly interpretable (no coarse-bucket collisions)."""
    parts = [_fmt(f, p.get(f, 0)) for f in feats[:3]]
    return " · ".join(x for x in parts if x)


def supply_word(p):
    """Human supply-origin label from local production + import (small clusters don't import
    externally, so low-local & low-import = centrally piped from the large clusters)."""
    if p.get("import_intensity", 0) > 0.30:
        return "Import-reliant"
    if p.get("local_self_sufficiency", 0) > 0.55:
        return "Local"
    if p.get("local_self_sufficiency", 0) < 0.35:
        return "Central"
    return "Mixed"


def cluster_name(p, feats):
    """For supply-type strategies (local + import in the feature set) prepend the human
    supply-origin word; always append the value string so clusters stay distinct."""
    vals = name_from_feats(p, feats)
    if "local_self_sufficiency" in feats and "import_intensity" in feats:
        return f"{supply_word(p)} · {vals}"
    return vals


def unique(names):
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


def _tree2(lv, ml):
    return DecisionTreeClassifier(max_leaf_nodes=lv, min_samples_leaf=ml, min_samples_split=ml,
                                  class_weight="balanced", random_state=C.RANDOM_STATE)


def best_tree2(X, y, k):
    """Tune (max_leaf_nodes, min_samples_leaf) for accuracy+readability: best CV, then
    among cases within 0.01 of it prefer fewer leaves and larger min_leaf (cleaner tree)."""
    cand = []
    for lv in [None, k, 6, 8, 10, 12, 16, 20, 25]:
        if lv is not None and lv < k:
            continue
        for ml in [5, 10, 20, 40]:
            acc = cross_val_score(_tree2(lv, ml), X, y, cv=5).mean()
            cand.append((acc, lv, ml))
    top = max(c[0] for c in cand)
    good = [c for c in cand if c[0] >= top - 0.01]
    good.sort(key=lambda c: (c[1] if c[1] is not None else 999, -c[2]))
    acc, lv, ml = good[0]
    return lv, ml, acc


def cluster(df, feats, k):
    Xc = StandardScaler().fit_transform(df[feats].fillna(0))
    km = KMeans(k, random_state=C.RANDOM_STATE, n_init=10).fit(Xc)
    lab = km.labels_
    order = np.argsort([df["local_self_sufficiency"].values[lab == c].mean() for c in range(k)])
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
        subs = {names[i]: f"loc {profs[i]['local_self_sufficiency']*100:.0f}% · "
                          f"imp {profs[i]['import_intensity']*100:.0f}% · {profs[i]['n_pipelines']:.1f} pipes"
                for i in range(k)}
        base = pd.Series(y).value_counts(normalize=True).max()
        lv, ml, _ = best_tree2(X, y, k)   # tuned leaf budget + min_samples_leaf
        dtcv, dtbal, dtkap = cv_metrics(_tree2(lv, ml), X, y)  # acc / balanced acc / Cohen kappa
        tr, te = train_test_split(np.arange(len(feat)), test_size=0.1, random_state=42, stratify=y)
        dttest = _tree2(lv, ml).fit(X.iloc[tr], y[tr]).score(X.iloc[te], y[te])
        rfcv = cross_val_score(RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                               class_weight="balanced", random_state=0, n_jobs=-1), X, y, cv=5).mean()
        rows.append({"def": name, "subset": tag, "k": k, "n": len(feat), "silhouette": round(sil, 3),
                     "baseline": round(base, 3), "DT_leaves": (lv or 0), "DT_minleaf": ml,
                     "DT_CV": dtcv, "DT_bal_acc": dtbal, "DT_kappa": dtkap,
                     "DT_test": round(dttest, 3), "DT_lift": round(dtcv - base, 3),
                     "RF_CV": round(rfcv, 3), "degenerate": int(base > 0.80)})
        kd = os.path.join(OUT, name, tag, f"k{k:02d}"); os.makedirs(kd, exist_ok=True)
        full = _tree2(lv, ml).fit(X, y)
        fig, ax = plt.subplots(figsize=(16, 9))
        plot_tree(full, feature_names=inputs, class_names=names, filled=True, rounded=True,
                  proportion=True, impurity=True, precision=1, fontsize=8, ax=ax)
        ax.set_title(f"{name} [{tag}] k={k} lv={lv} | base {base:.2f} DT_CV {dtcv:.2f} "
                     f"test {dttest:.2f} RF {rfcv:.2f}", fontsize=12)
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
    print(f"  {name} [{tag}]: k<= {nun} done")
    return rows


def comparison_html(allrows, defs):
    df = pd.DataFrame(allrows); df.to_csv(os.path.join(OUT, "comparison.csv"), index=False)
    leg = "".join(f"<tr><td><b>{c}</b></td><td>{x}</td><td>{cc}</td><td><code>{', '.join(fs)}</code></td></tr>"
                  for c, fs, cc, x in defs)
    def col(v, lo=0.2, hi=0.95):
        t = max(0, min(1, (v-lo)/(hi-lo))); return f"background:rgb({int(255-t*156)},{int(255-t*90)},{int(255-t*108)})"
    body = []
    for _, r in df.iterrows():
        deg = int(r.degenerate) == 1
        nd = df[(df["def"] == r["def"]) & (df.subset == r.subset) & (df.degenerate == 0)]
        best = (not deg) and len(nd) and r.DT_CV >= nd.DT_CV.max() - 1e-9
        st = " style='opacity:.4;background:#f2f2f2'" if deg else ""
        leaves_disp = "none" if int(r.DT_leaves) == 0 else int(r.DT_leaves)
        body.append(f"<tr{st}><td>{r['def']}</td><td>{r['subset']}</td><td>{r['k']}</td><td>{r['n']}</td>"
                    f"<td>{r['silhouette']}</td><td>{r['baseline']}{' ⚠' if deg else ''}</td>"
                    f"<td>{leaves_disp}</td><td>{r['DT_minleaf']}</td>"
                    f"<td style='{col(r.DT_CV)}'>{r.DT_CV}{' ★' if best else ''}</td>"
                    f"<td>{r.DT_bal_acc}</td><td style='{col(r.DT_kappa, 0, 1)}'>{r.DT_kappa}</td>"
                    f"<td>{r.DT_test}</td><td>{r.DT_lift}</td><td style='{col(r.RF_CV)}'>{r.RF_CV}</td></tr>")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Combined strategies — comparison</title>
<style>body{{font-family:Inter,Calibri,sans-serif;margin:26px;color:#161D41}}h1{{font-size:20px}}h2{{font-size:15px}}
.sub{{color:#8a8f9c;font-size:13px;max-width:960px;margin-bottom:12px}}table{{border-collapse:collapse;font-size:12px;margin-bottom:16px}}
th,td{{border:1px solid #d4d8e0;padding:4px 9px;text-align:center}}th{{background:#161D41;color:#fff}}tr:hover{{outline:2px solid #63A593}}a{{color:#63A593}}</style></head><body>
<h1>Combined clustering strategies — decision-tree predictability</h1>
<p><a href="drivers.html">&rarr; Driver consistency across &gt;90% cases</a></p>
<div class="sub">Meaningful clustering ideas combining the predictable dimensions (local production, import, scale).
DT predicts the cluster from sampled inputs; leaf tuned for accuracy+readability. RF = ceiling. ★ = best DT_CV per
def/subset (informative only). <b>Greyed (⚠) = degenerate</b> (one cluster &gt;80%). k capped to distinct feature
combos. Look for strategies with a real meaning AND high DT lift = clusters the model can actually explain.</div>
<h2>Strategy legend</h2>
<table><thead><tr><th>folder</th><th>type</th><th>idea</th><th>features</th></tr></thead><tbody>{leg}</tbody></table>
<h2>Accuracy</h2>
<table><thead><tr><th>def</th><th>subset</th><th>k</th><th>n</th><th>silhouette</th><th>baseline</th>
<th>DT leaves</th><th>min_leaf</th><th>DT acc</th><th>bal.acc</th><th>Cohen&nbsp;&kappa;</th>
<th>DT test</th><th>DT lift</th><th>RF CV</th></tr></thead>
<tbody>{''.join(body)}</tbody></table>
<div class="sub">DT tuned per case: leaf budget &amp; min_samples_leaf swept for best CV (then fewest leaves / most
robust among near-best). Cluster names are value-based (mean of the clustered features) so every cluster is distinct
and directly readable.</div></body></html>"""
    open(os.path.join(OUT, "comparison.html"), "w", encoding="utf-8").write(html)
    pd.DataFrame([{"folder": c, "type": x, "concept": cc, "features": ";".join(fs)} for c, fs, cc, x in defs]
                 ).to_csv(os.path.join(OUT, "definitions.csv"), index=False)
    return df


def drivers_html():
    d = pd.read_csv(os.path.join(OUT, "comparison.csv"))
    hi = d[(d.DT_CV > 0.90) & (d.degenerate == 0)].sort_values(["def", "subset", "k"])
    if len(hi) == 0:
        hi = d[d.degenerate == 0].sort_values("DT_CV", ascending=False).head(10)
    cases, imps = [], {}
    for _, r in hi.iterrows():
        imp = pd.read_csv(os.path.join(OUT, r["def"], r["subset"], f"k{int(r['k']):02d}",
                          "feature_importance.csv")).set_index("feature")["DT_importance_pct"]
        t = f"{r['def']}/{r['subset']}/k{int(r['k'])}"; cases.append((t, round(r.DT_CV, 2))); imps[t] = imp
    M = pd.DataFrame({t: imps[t] for t, _ in cases}).fillna(0.0); M["MEAN"] = M.mean(axis=1)
    M = M.sort_values("MEAN", ascending=False)
    def cc(v): t = max(0, min(1, v/60)); return f"background:rgb({int(255-t*156)},{int(255-t*90)},{int(255-t*108)})"
    head = "".join(f"<th>{t}<br><span style='font-weight:400;font-size:10px'>{cv}</span></th>" for t, cv in cases)+"<th>MEAN</th>"
    body = "".join("<tr><th style='background:#eef0f4;text-align:left'>"+f+"</th>"
                   + "".join(f"<td style='{cc(M.loc[f,t])}'>{M.loc[f,t]:.0f}</td>" for t, _ in cases)
                   + f"<td style='{cc(M.loc[f,'MEAN'])};font-weight:700'>{M.loc[f,'MEAN']:.0f}</td></tr>" for f in M.index)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Combined drivers</title>
<style>body{{font-family:Inter,Calibri,sans-serif;margin:26px;color:#161D41}}table{{border-collapse:collapse;font-size:12px}}
th,td{{border:1px solid #d4d8e0;padding:4px 8px;text-align:center}}thead th{{background:#161D41;color:#fff}}a{{color:#63A593}}</style></head><body>
<p><a href="comparison.html">&larr; back</a></p><h1 style='font-size:19px'>Drivers of the combined typologies (&gt;90% cases)</h1>
<table><thead><tr><th>feature \\ case</th>{head}</tr></thead><tbody>{body}</tbody></table></body></html>"""
    open(os.path.join(OUT, "drivers.html"), "w", encoding="utf-8").write(html)
    print(f"drivers.html: {len(cases)} cases")


def main():
    os.makedirs(OUT, exist_ok=True)
    feat = pd.read_csv(C.FEATURES_CSV); raw = pd.read_excel(C.EXTRACTED_RESULTS)
    rob, *_ = robust_run_ids(ROBUST_SCENARIO)
    allrows = []
    for name, feats, _c, _x in STRATEGIES:
        for tag, mask in [("all", feat["run"].notna()), ("robust", feat["run"].isin(rob))]:
            allrows += run_case(feat[mask].reset_index(drop=True), raw, name, feats, tag)
    comparison_html(allrows, STRATEGIES)
    drivers_html()
    df = pd.DataFrame(allrows); inf = df[df.degenerate == 0]
    print("\nBEST informative per strategy/subset:")
    for (nm, sub), g in inf.groupby(["def", "subset"]):
        r = g.loc[g.DT_CV.idxmax()]
        print(f"  {nm:<13} {sub:<7} k={int(r.k)} base={r.baseline} DT_CV={r.DT_CV} lift={r.DT_lift} RF={r.RF_CV}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
