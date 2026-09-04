"""
Full decentralisation-only analysis.

For each definition (A: small_prod/total_demand, B: small_prod/small_demand=self-suff),
each subset (all, robust) and each k = 2..10:
  - k-Means cluster on the 1-D decentralisation metric (ordered low->high, named by level)
  - decision tree (inputs -> cluster), leaf budget tuned for accuracy+readability
  - RandomForest comparison (accuracy ceiling)
  - schematic-leaf HTML tree (leaves named by decentralisation level)
  - per-k driver_tree.png + rules.txt
Then a master comparison HTML with accuracy tables (DT vs RF, per def/subset/k).

Outputs: DT_supplychain/decentralization_only/
Run:  python decentralization_full_analysis.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe in background threads)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score


def cv_metrics(clf, X, y):
    """5-fold CV accuracy, balanced accuracy and Cohen's kappa in one pass."""
    p = cross_val_predict(clf, X, y, cv=5)
    return (round(accuracy_score(y, p), 3), round(balanced_accuracy_score(y, p), 3),
            round(cohen_kappa_score(y, p), 3))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import render_driver_tree_html as R
from pure_leaves import leaf_rules
from robust_trees import robust_run_ids, ROBUST_SCENARIO
from decentralization_clustering import load_metrics, NODES, SMALL

# short path (folder/file names kept brief to stay under Windows MAX_PATH 260 chars,
# otherwise files land under the \\?\UNC\ long-path form and won't open in a browser)
OUT = r"X:\3000_simulations_24_may_2026\Snellius_3000_simulations\DT_supplychain\decentr"
KS = range(2, 11)
LEAF_GRID = [None, 4, 5, 6, 8, 10, 12, 16, 20]   # None handled -> k as min
DEFS = [("A", "defA", "small_prod / total_demand"),
        ("B", "defB", "small_prod / small_demand = self-suff")]
PURITY = 0.70


def level_names(k):
    presets = {
        2: ["Low", "High"],
        3: ["Low", "Medium", "High"],
        4: ["Very low", "Low", "High", "Very high"],
        5: ["Very low", "Low", "Medium", "High", "Very high"],
        6: ["Very low", "Low", "Med-low", "Med-high", "High", "Very high"],
        7: ["Very low", "Low", "Med-low", "Medium", "Med-high", "High", "Very high"],
    }
    if k in presets:
        return [f"{n} decentr." for n in presets[k]]
    return [f"Decentr L{i+1}/{k}" for i in range(k)]


def cluster_1d(x, k):
    xs = StandardScaler().fit_transform(x.reshape(-1, 1))
    km = KMeans(k, random_state=C.RANDOM_STATE, n_init=10).fit(xs)
    order = np.argsort([x[km.labels_ == c].mean() for c in range(k)])
    remap = {old: new for new, old in enumerate(order)}
    y = np.array([remap[c] for c in km.labels_])
    sil = silhouette_score(xs, km.labels_) if k > 1 else 0
    ranges = [(x[y == i].min(), x[y == i].max(), (y == i).sum()) for i in range(k)]
    return y, sil, ranges


def best_tree(X, y, k):
    """Pick smallest leaf budget whose CV is within 0.01 of the best (readable+accurate)."""
    cand = sorted({(lv if lv else k) for lv in LEAF_GRID if (lv or k) >= k})
    best = []
    for lv in cand:
        clf = DecisionTreeClassifier(max_leaf_nodes=lv, min_samples_leaf=20, min_samples_split=20,
                                     class_weight="balanced", random_state=C.RANDOM_STATE)
        best.append((lv, cross_val_score(clf, X, y, cv=5).mean()))
    top = max(b[1] for b in best)
    lv = min(l for l, a in best if a >= top - 0.01)
    return lv, dict(best)[lv]


def _tree(lv):
    return DecisionTreeClassifier(max_leaf_nodes=lv, min_samples_leaf=20, min_samples_split=20,
                                  class_weight="balanced", random_state=C.RANDOM_STATE)


def schematic_html(dfm, y, names, out_html, title, ranges=None, subs=None):
    """Reuse the schematic-leaf renderer with custom cluster names.
    ranges = list of (lo, hi, n) per cluster -> shown as the metric % on each card;
    subs   = optional dict {name: subtitle} overriding the card subtitle (e.g. topology)."""
    dfm = dfm.copy()
    dfm["cluster_final"] = y
    inputs = [c for c in C.DRIVER_INPUTS if c in dfm.columns]
    X = dfm[inputs].fillna(0)
    k = len(names)
    lv, _ = best_tree(X, y, k)
    clf = _tree(lv).fit(X, y)
    dfm["_leaf"] = clf.apply(X)
    leaf_rows = {lid: dfm[dfm["_leaf"] == lid] for lid in np.unique(dfm["_leaf"])}
    name_by = {i: names[i] for i in range(k)}
    tree_dict = R._build_node(clf.tree_, 0, inputs, leaf_rows, name_by)
    scale = {
        "pipe_max": round(max(max(nd["info"][a][1] for a in R.ARC_COLS) for nd in R._leaves(tree_dict)), 0) or 1.0,
        "large_elzr_max": round(max(max(nd["info"]["cap_L1"], nd["info"]["cap_L2"]) for nd in R._leaves(tree_dict)), 0) or 1,
        "small_elzr_max": round(max(max(nd["info"]["cap_S1"], nd["info"]["cap_S2"]) for nd in R._leaves(tree_dict)), 0) or 1,
        "import_max": round(max(max(nd["info"]["imp_L1"], nd["info"]["imp_L2"]) for nd in R._leaves(tree_dict)), 0) or 1,
    }
    grad = [C.UU_NAVY, "#2f3d7a", C.UU_TEAL, "#9bc4b3", C.UU_YELLOW, C.UU_AMBER,
            C.UU_RED, "#6b4e9e", "#8a8f9c", "#3a6ea5"]
    R.COLOR = {names[i]: grad[i % len(grad)] for i in range(k)}
    if subs is not None:
        R.SUB = {names[i]: subs.get(names[i], "") for i in range(k)}
    elif ranges is not None:
        R.SUB = {names[i]: f"{ranges[i][0]*100:.0f}–{ranges[i][1]*100:.0f}% local" for i in range(k)}
    else:
        R.SUB = {names[i]: f"level {i+1}/{k}" for i in range(k)}
    labels = [nd["info"]["label"] for nd in R._leaves(tree_dict)]
    cbar = "\n".join(f' <span class="cchip"><span class="dot" style="background:{R.COLOR[l]}"></span>{l}</span>'
                     for l in dict.fromkeys(labels))
    html = R._TEMPLATE
    for tok, val in [("/*__CBAR__*/", cbar), ("/*__DATA__*/", json.dumps({"tree": tree_dict, "scale": scale})),
                     ("/*__FEAT__*/", json.dumps(R.FEAT)), ("/*__SUB__*/", json.dumps(R.SUB)),
                     ("/*__COLOR__*/", json.dumps(R.COLOR)), ("/*__PAIRS__*/", json.dumps(R.PAIRS))]:
        html = html.replace(tok, val)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def run_case(feat, dm, raw, name, col, unit, tag, rob_ids):
    x = dm[col].values
    # keep feature column names; raw duplicates get _raw (schematic needs raw caps/arcs/prod)
    dfm = feat.merge(raw, on="run", how="left", suffixes=("", "_raw"))
    inputs = [c for c in C.DRIVER_INPUTS if c in feat.columns]
    Xdrv = feat[inputs].fillna(0).reset_index(drop=True)
    rows = []
    best_sil, best_k = -1, 2
    for k in KS:
        y, sil, ranges = cluster_1d(x, k)
        if sil > best_sil:
            best_sil, best_k = sil, k
        base = pd.Series(y).value_counts(normalize=True).max()
        lv, _ = best_tree(Xdrv, y, k)
        dtcv, dtbal, dtkap = cv_metrics(_tree(lv), Xdrv, y)   # acc / balanced acc / Cohen kappa
        # test acc on 90/10
        tr, te = train_test_split(np.arange(len(feat)), test_size=0.1, random_state=42, stratify=y)
        clf = _tree(lv).fit(Xdrv.iloc[tr], y[tr]); dttest = clf.score(Xdrv.iloc[te], y[te])
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced",
                                    random_state=0, n_jobs=-1)
        rfcv = cross_val_score(rf, Xdrv, y, cv=5).mean()
        rows.append({"def": name, "unit": unit, "subset": tag, "k": k, "n": len(feat),
                     "silhouette": round(sil, 3), "baseline": round(base, 3),
                     "DT_leaves": lv, "DT_CV": dtcv, "DT_bal_acc": dtbal, "DT_kappa": dtkap,
                     "DT_test": round(dttest, 3), "DT_lift": round(dtcv - base, 3), "RF_CV": round(rfcv, 3)})
        # per-k folder: tree png + rules
        kd = os.path.join(OUT, name, tag, f"k{k:02d}")
        os.makedirs(kd, exist_ok=True)
        names_k = level_names(k)
        cls = [f"{names_k[i]} ({ranges[i][0]:.2f}-{ranges[i][1]:.2f})" for i in range(k)]
        fig, ax = plt.subplots(figsize=(16, 9))
        plot_tree(clf.fit(Xdrv, y), feature_names=inputs, class_names=cls, filled=True,
                  rounded=True, proportion=True, impurity=True, precision=1, fontsize=8, ax=ax)
        ax.set_title(f"{name} [{tag}] k={k} leaves={lv} | baseline {base:.2f} DT_CV {dtcv:.2f} "
                     f"test {dttest:.2f} RF {rfcv:.2f}", fontsize=13)
        fig.savefig(os.path.join(kd, "driver_tree.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        with open(os.path.join(kd, "rules.txt"), "w", encoding="utf-8") as f:
            for r in leaf_rules(_tree(lv).fit(Xdrv, y), inputs, cls):
                mk = "" if r["purity"] >= PURITY else " [mixed]"
                f.write(f"[{r['purity']:.0%}, n={r['n']}] {r['class']}{mk} IF {r['rule']}\n")

        # --- feature importance (%) : DT and RF ---
        dt_full = _tree(lv).fit(Xdrv, y)
        rf_full = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                         class_weight="balanced", random_state=0, n_jobs=-1).fit(Xdrv, y)
        imp = pd.DataFrame({"feature": inputs,
                            "DT_importance_pct": (dt_full.feature_importances_ * 100).round(1),
                            "RF_importance_pct": (rf_full.feature_importances_ * 100).round(1)}
                           ).sort_values("DT_importance_pct", ascending=False)
        imp.to_csv(os.path.join(kd, "feature_importance.csv"), index=False)
        top = imp[imp["DT_importance_pct"] > 0]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(range(len(top)), top["DT_importance_pct"], color=C.UU_TEAL, label="DT")
        ax.barh(range(len(top)), top["RF_importance_pct"], left=0, color="none",
                edgecolor=C.UU_NAVY, label="RF (outline)")
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top["feature"]); ax.invert_yaxis()
        ax.set_xlabel("importance (%)"); ax.legend(fontsize=8)
        ax.set_title(f"{name} [{tag}] k={k} — feature importance %", fontsize=11)
        fig.savefig(os.path.join(kd, "feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- cluster levels (ranges of the decentralisation metric) ---
        pd.DataFrame([{"level": i, "name": names_k[i], "range_lo": round(float(ranges[i][0]), 4),
                       "range_hi": round(float(ranges[i][1]), 4), "n": int(ranges[i][2]),
                       "mean": round(float(x[y == i].mean()), 4)} for i in range(k)]
                     ).to_csv(os.path.join(kd, "cluster_levels.csv"), index=False)
        # schematic HTML per k (clusters named by decentralisation level + range)
        schematic_html(dfm, y, names_k, os.path.join(kd, "tree.html"),
                       f"{name} {tag} k={k}", ranges)
    print(f"  {name} [{tag}]: done k=2..10, best silhouette k={best_k} ({best_sil:.2f})")
    return rows, best_k


def comparison_html(allrows):
    df = pd.DataFrame(allrows)
    df.to_csv(os.path.join(OUT, "comparison.csv"), index=False)
    # build HTML table
    def color(v, lo=0.3, hi=0.9):
        t = max(0, min(1, (v - lo) / (hi - lo)))
        r = int(255 - t * (255 - 99)); g = int(255 - t * (255 - 165)); b = int(255 - t * (255 - 147))
        return f"background:rgb({r},{g},{b})"
    rowshtml = []
    for _, r in df.iterrows():
        best = r["DT_CV"] >= df[(df["def"] == r["def"]) & (df.subset == r.subset)]["DT_CV"].max() - 1e-9
        star = " ★" if best else ""
        rowshtml.append(
            f"<tr><td>{r['def']}</td><td>{r['subset']}</td><td>{r['k']}</td><td>{r['n']}</td>"
            f"<td>{r['silhouette']}</td><td>{r['baseline']}</td><td>{r['DT_leaves']}</td>"
            f"<td style='{color(r['DT_CV'])}'>{r['DT_CV']}{star}</td><td>{r['DT_bal_acc']}</td>"
            f"<td style='{color(r['DT_kappa'], 0, 1)}'>{r['DT_kappa']}</td><td>{r['DT_test']}</td>"
            f"<td>{r['DT_lift']}</td><td style='{color(r['RF_CV'])}'>{r['RF_CV']}</td></tr>")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Decentralisation — DT comparison</title>
<style>body{{font-family:Inter,Calibri,sans-serif;margin:26px;color:#161D41}}
h1{{font-size:20px}} .sub{{color:#8a8f9c;font-size:13px;margin-bottom:16px;max-width:900px}}
table{{border-collapse:collapse;font-size:12px;margin-bottom:24px}}
th,td{{border:1px solid #d4d8e0;padding:4px 9px;text-align:center}}
th{{background:#161D41;color:#fff;position:sticky;top:0}} tr:hover{{outline:2px solid #63A593}}
.leg{{font-size:11px;color:#8a8f9c}}</style></head><body>
<h1>Decentralisation-only decision trees — comparison</h1>
<div class="sub">Cluster on the decentralisation metric (1-D) with k-Means, k=2..10, for two definitions
(A = small production / total demand; B = small production / small demand = self-sufficiency) and two
subsets (all 2719 runs; robust = design cost-determined, reopt infeasible or &Delta;NPV&gt;1%).
Decision tree predicts the decentralisation cluster from the sampled inputs; leaf budget tuned for
accuracy+readability. RandomForest = accuracy ceiling. ★ = best DT_CV within each def/subset.
Green = higher accuracy. Per-k folders hold driver_tree.png, rules.txt and tree_magnitude.html.</div>
<table><thead><tr><th>definition</th><th>subset</th><th>k</th><th>n</th><th>silhouette</th>
<th>baseline</th><th>DT leaves</th><th>DT acc</th><th>bal.acc</th><th>Cohen&nbsp;&kappa;</th><th>DT test</th><th>DT lift</th><th>RF CV</th></tr></thead>
<tbody>{''.join(rowshtml)}</tbody></table>
<div class="leg">baseline = majority-class share · DT lift = DT_CV &minus; baseline (real skill) · RF = RandomForest 200 trees</div>
</body></html>"""
    out = os.path.join(OUT, "comparison.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nComparison HTML -> {out}")
    print(df.to_string(index=False))


def main():
    C.apply_joule_style()
    os.makedirs(OUT, exist_ok=True)
    feat = pd.read_csv(C.FEATURES_CSV)
    dm = load_metrics()
    raw = pd.read_excel(C.EXTRACTED_RESULTS)
    rob_ids, *_ = robust_run_ids(ROBUST_SCENARIO)

    allrows = []
    for name, col, unit in DEFS:
        for tag, mask in [("all", feat["run"].notna()), ("robust", feat["run"].isin(rob_ids))]:
            fsub = feat[mask].reset_index(drop=True)
            dsub = dm[dm["run"].isin(fsub["run"])].set_index("run").loc[fsub["run"]].reset_index()
            rows, _ = run_case(fsub, dsub, raw, name, col, unit, tag, rob_ids)
            allrows += rows
    comparison_html(allrows)


if __name__ == "__main__":
    main()
