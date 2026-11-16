"""
One HTML page that puts the DT accuracy study together, with links to every tree.

Reads the outputs of dt_v5_accuracy_sweep.py, dt_v5_accuracy_ladder.py and
dt_v5_regtree.py and writes <RESULTS_FOLDER>/DT_uncert_accuracy/index.html.
All links are relative, so the page works from X:, the UNC path or a download.

Run:  python dt_v5_accuracy_overview.py
"""
import os

import pandas as pd

import dt_v5 as V
import dt_v5_accuracy_sweep as S

OUT = S.OUT

EXTRA_CSS = """<style>
.grid{display:flex;flex-wrap:wrap;gap:14px}
.grid figure{margin:0;width:420px}
.grid img{width:100%;border:1px solid #ddd}
figcaption{font-size:12px;color:#555}
td.num{text-align:right}
.toc a{margin-right:14px}
</style>"""


def rel(*p):
    return "/".join(p).replace(os.sep, "/")


def a(href, txt):
    return f"<a href='{href}'>{txt}</a>"


def section_sweep():
    s = pd.read_csv(os.path.join(OUT, "summary.csv"))
    rows = []
    for _, r in s.iterrows():
        tag = r.case.replace("/", "_")
        rows.append(
            f"<tr><td>{r.case}</td><td class=num>{r.baseline:.2f}</td>"
            f"<td class=num><b>{r.best_grouped_acc:.3f}</b> ({r.best_grouped_kappa:.2f})</td>"
            f"<td class=num>{r.best_leaves}</td><td class=num>{r.rf_grouped_acc:.3f}</td>"
            f"<td class=num>{r.hgb_grouped_acc:.3f}</td><td class=num>{r.max_ungrouped_acc:.3f}</td>"
            f"<td class=num>{r.max_train_acc:.3f}</td>"
            f"<td class=num>{r.train_90_leaves} / {r.train_95_leaves}</td>"
            f"<td>{a(rel('pareto_' + tag + '.png'), 'curve')}</td></tr>")
    figs = "".join(
        f"<figure><a href='pareto_{c.replace('/', '_')}.png'><img src='pareto_{c.replace('/', '_')}.png'></a>"
        f"<figcaption>{c}</figcaption></figure>" for c in s.case)
    return (
        "<h2 id='sweep'>1. Hyperparameter sweep — classification trees on k-Means labels</h2>"
        "<div class='note'>2448 configs per case (max_depth, max_leaf_nodes, min_samples_leaf, "
        "min_samples_split, ccp_alpha, criterion, class_weight). Three scales: train "
        "(resubstitution), ungrouped CV (sibling runs leak), <b>grouped CV by economy</b> (honest). "
        "<b>No tree reaches 90% grouped.</b> 90–95% exists only on train / ungrouped. "
        "max_depth is the only lever that matters (plateau at depth 6–8); min_samples_leaf "
        "±0.02, min_samples_split none.</div>"
        "<table><tr><th>case</th><th>baseline</th><th>best grouped acc (&kappa;)</th>"
        "<th>leaves</th><th>RF grouped</th><th>HGB grouped</th><th>max ungrouped</th>"
        "<th>max train</th><th>leaves train 90/95%</th><th></th></tr>"
        + "".join(rows) + "</table>"
        f"<p>{a('pareto_all.png', 'all frontiers in one figure')} &middot; "
        f"{a('sweep_all.csv', 'sweep_all.csv')} &middot; {a('summary.csv', 'summary.csv')}</p>"
        f"<div class='grid'>{figs}</div>")


def section_ladder():
    lad = pd.read_csv(os.path.join(OUT, "trees", "ladder_all.csv"))
    body = []
    for case, d in lad.groupby("case", sort=False):
        body.append(f"<h3>{case}</h3><table><tr><th>step</th><th>leaves</th><th>depth</th>"
                    f"<th>grouped acc</th><th>&kappa;</th><th>ungrouped</th><th>train</th>"
                    f"<th>leaves &ge;90% pure</th><th>files</th></tr>")
        for _, r in d.iterrows():
            fo = rel("trees", r.folder)
            body.append(
                f"<tr><td>{r.step}</td><td class=num>{r.leaves}</td><td class=num>{r.depth}</td>"
                f"<td class=num><b>{r.acc_grouped:.3f}</b></td><td class=num>{r.kappa_grouped:.3f}</td>"
                f"<td class=num>{r.acc_ungrouped:.3f}</td><td class=num>{r.acc_train:.3f}</td>"
                f"<td class=num>{r.leaves_pure90_pct}%</td>"
                f"<td>{a(fo + '/tree.html', 'schematic')} &middot; "
                f"{a(fo + '/driver_tree.png', 'png')} &middot; {a(fo + '/driver_tree.pdf', 'pdf')} "
                f"&middot; {a(fo + '/rules.txt', 'rules')} &middot; {a(fo + '/tree.txt', 'tree.txt')}"
                f"</td></tr>")
        body.append("</table>")
    return (
        "<h2 id='ladder'>2. Accuracy ladder — complete trees</h2>"
        "<div class='note'>Per case, on grouped acc: <b>base</b> = best &le;8 leaves (&ge;0.65); "
        "<b>plus5</b> = +5 pt; <b>mid</b> = 1/3, 2/3 of the gap; <b>plateau</b> = smallest tree "
        "within 0.005 of the best; steps &lt;0.01 apart dropped. Plateau mostly ~24 leaves, "
        "depth 6–7, for +2–7 pt; +5 pt only reachable in cent_imp k3/k4.</div>"
        + "".join(body))


BASE_FINDINGS = (
    "<b>Regression tree beats k-Means + classifier in all 16 definitions</b> (+0.04 to +0.19 "
    "best vs best). k-Means loses on predictability, not resolution (oracle caps high). "
    "Single trees stay 0.15–0.30 below boosting. Supply mix well explained; decentralisation "
    "degree and topology weak (topology largely geometry-driven: SS_pipe 74% of variance "
    "within economy).")

BASE_ACC_NOTE = ("On 3-level accuracy the dedicated classifier is equal or slightly better "
                 "(0–0.08), as expected — it optimises exactly that target. The regression tree "
                 "wins on R&sup2;, i.e. on the actual output values.")

# from regtree_derived/summary.csv and accuracy.csv (2026-11-04)
DERIVED_FINDINGS = (
    "<b>Ratios lift the regression tree as much as the classifier</b>: R&sup2; at 8 leaves "
    "+0.11 to +0.41 over the base drivers (cent_imp 0.63 &rarr; 0.83, selfserved 0.57 &rarr; "
    "0.80, topo/count 0.29 &rarr; 0.71, backbone 0.21 &rarr; 0.57); best tree 0.38–0.91. "
    "<b>Regression tree still beats k-Means + classifier in all 16 definitions</b> (+0.02 to "
    "+0.13 best vs best). Gap to boosting shrinks to 0.06–0.18 (was 0.15–0.30); the cent_imp "
    "plateau tree (0.91, 32 leaves) equals the old base-driver boosting ceiling (0.905). "
    "Plateaus are small for decentralisation and topology (8–16 leaves), 32–64 for supply mix. "
    "Weakest still: decentralisation degree (A_share 0.50, HGB 0.64) and full structure (0.38).")
DERIVED_ACC_NOTE = ("With ratios the regression tree is almost as good on 3-level accuracy as "
                    "the dedicated classifier (&minus;0.04 to +0.03 at 8 leaves; better for "
                    "transport, meshsat, structure) while giving the quantitative profile. "
                    "8-leaf accuracy 0.78–0.88, up from 0.58–0.79 with base drivers.")


def section_regtree(folder="regtree", findings=None, acc_note=None):
    rd = os.path.join(OUT, folder)
    s = pd.read_csv(os.path.join(rd, "summary.csv"))
    rows, figs = [], []
    for _, r in s.iterrows():
        tag = r.definition.replace("/", "_")
        t8 = rel("trees", tag, "L8")
        tp = rel("trees", tag, f"L{int(r.regtree_plateau_leaves)}")
        links = (f"L8: {a(t8 + '/tree.png', 'png')} {a(t8 + '/tree.pdf', 'pdf')} "
                 f"{a(t8 + '/rules.txt', 'rules')} {a(t8 + '/leaf_profiles.csv', 'profiles')}")
        if int(r.regtree_plateau_leaves) != 8:
            links += (f"<br>L{int(r.regtree_plateau_leaves)}: {a(tp + '/tree.png', 'png')} "
                      f"{a(tp + '/tree.pdf', 'pdf')} {a(tp + '/rules.txt', 'rules')} "
                      f"{a(tp + '/leaf_profiles.csv', 'profiles')}")
        rows.append(
            f"<tr><td>{r.definition}</td><td class=num>{r.n_targets}</td>"
            f"<td class=num>{r.regtree_r2_L8:.2f} ({r.regtree_types_L8})</td>"
            f"<td class=num><b>{r.regtree_r2_best:.2f}</b></td>"
            f"<td class=num>{int(r.regtree_plateau_leaves)} / {r.regtree_types_plateau}</td>"
            f"<td class=num>{r.regtree_train_r2_plateau:.2f}</td>"
            f"<td class=num>{r.kmeans_r2_L8:.2f}</td>"
            f"<td class=num>{r.kmeans_r2_best:.2f} (k{r.kmeans_best_k})</td>"
            f"<td class=num>{r.oracle_r2_k3:.2f} / {r.oracle_r2_k10:.2f}</td>"
            f"<td class=num>{r.rf_r2:.2f} / {r.hgb_r2:.2f}</td>"
            f"<td>{a('r2_' + tag + '.png', 'curve')}</td><td>{links}</td></tr>")
        figs.append(f"<figure><a href='r2_{tag}.png'><img src='r2_{tag}.png'></a>"
                    f"<figcaption>{r.definition}</figcaption></figure>")
    acc = pd.read_csv(os.path.join(rd, "accuracy.csv"))
    arows = "".join(
        f"<tr><td>{r.definition}</td><td>{r.tree}</td><td class=num>{r.leaves}</td>"
        f"<td class=num>{r.r2_grouped:.3f}</td><td class=num>{r.baseline_k3:.3f}</td>"
        f"<td class=num><b>{r.acc_k3_grouped:.3f}</b></td><td class=num>{r.kappa_k3_grouped:.3f}</td>"
        f"<td class=num>{r.kmeans_clf_acc_k3:.3f}</td><td class=num>{r.kmeans_clf_r2_k3:.3f}</td>"
        f"<td>{r.mae}</td></tr>" for _, r in acc.iterrows())
    acc_html = (
        "<h3 id='accuracy'>Accuracy of the regression trees</h3>"
        "<div class='note'>To quote next to the classification trees. Each run's out-of-fold "
        "predicted profile (grouped CV by economy) is assigned to the nearest k-Means centroid "
        "at <b>k = 3</b> and compared with its true k-Means class. <i>k-Means clf</i> = "
        "classification tree trained directly on the k = 3 labels, same leaf budget. "
        "MAE = mean absolute error per target in raw units (shares 0–1, pipes in count).<br>"
        + (acc_note if acc_note is not None else BASE_ACC_NOTE) + "</div>"
        "<table><tr><th>definition</th><th>tree</th><th>leaves</th><th>R&sup2; grouped</th>"
        "<th>baseline k3</th><th>accuracy k3 (grouped)</th><th>&kappa; k3</th>"
        "<th>k-Means clf acc k3</th><th>k-Means clf R&sup2; k3</th><th>MAE per target</th></tr>"
        + arows + f"</table><p>{a('accuracy.csv', 'accuracy.csv')}</p>")
    return (
        "<h2 id='regtree'>Alternative to k — multi-output regression tree</h2>"
        "<div class='note'>Predictive clustering tree: the clustering features become continuous "
        "targets (standardized); each leaf is a type, no k. Compared with k-Means (k 3–10) + "
        "classification tree, predicted class replaced by its centroid. Metric: <b>pooled R&sup2; "
        "on standardized outputs, grouped CV by economy</b> (1 &minus; SSE/SST; 0 = predicting the "
        "overall mean). <i>Oracle</i> = k-Means R&sup2; if the class were predicted perfectly. "
        "<i>Types</i> = leaves after merging profiles closer than 0.25 SD.<br>"
        + (findings if findings is not None else BASE_FINDINGS) + "</div>"
        "<table><tr><th>definition</th><th>targets</th><th>regtree L8 R&sup2; (types)</th>"
        "<th>regtree best</th><th>plateau leaves / types</th><th>train R&sup2; plateau</th>"
        "<th>k-Means L8</th><th>k-Means best</th><th>oracle k3 / k10</th><th>RF / HGB</th>"
        "<th></th><th>trees</th></tr>" + "".join(rows) + "</table>"
        f"<p>{a('summary.csv', 'summary.csv')} &middot; "
        f"{a('curves.csv', 'curves.csv')}</p>"
        + acc_html +
        f"<h3>R&sup2; vs leaves</h3><div class='grid'>{''.join(figs)}</div>")


def main():
    # DT_uncert_accuracy/index.html: sweep + ladder (classification trees on k-Means labels)
    html = (
        f"<html><head><meta charset='utf-8'><title>DT accuracy study</title>{V.CSS}{EXTRA_CSS}"
        f"</head><body><h1>Driver-tree accuracy study — v5, DT_uncert</h1>"
        f"<div class='note'>5992 runs, 749 economies, 15 uncertain drivers (no k_small). "
        f"Question: what does a 90–95% accurate tree cost? Obsidian: "
        f"<i>Analysis/2026-11-03 Increasing DT accuracy</i>.</div>"
        f"<p class='toc'>{a('#sweep', '1. Sweep')}{a('#ladder', '2. Ladder')}"
        f"{a('trees/index.html', 'ladder index')}"
        f"{a('regtree/index.html', 'alternative to k: regression tree')}"
        f"{a('derived/index.html', 'derived ratios as drivers')}"
        f"{a('regtree_derived/index.html', 'regression tree with ratios')}</p>"
        + section_sweep() + section_ladder() + "</body></html>")
    # DT_uncert_accuracy/regtree/index.html: regression tree vs k-Means
    rhtml = (
        f"<html><head><meta charset='utf-8'><title>Regression tree vs k-Means</title>{V.CSS}"
        f"{EXTRA_CSS}</head><body><h1>Alternative to k — v5, DT_uncert</h1>"
        f"<div class='note'>5992 runs, 749 economies, 15 uncertain drivers (no k_small). "
        f"Obsidian: <i>Analysis/2026-11-03 Increasing DT accuracy</i>. "
        f"{a('../index.html', 'back to the accuracy study')}</div>"
        f"<p class='toc'>{a('#regtree', 'R&sup2; comparison')}{a('#accuracy', 'accuracy')}</p>"
        + section_regtree() + "</body></html>")
    pages = [(os.path.join(OUT, "index.html"), html),
             (os.path.join(OUT, "regtree", "index.html"), rhtml)]
    # DT_uncert_accuracy/regtree_derived/index.html: same, drivers = base + supply ratios
    if os.path.exists(os.path.join(OUT, "regtree_derived", "summary.csv")):
        dhtml = (
            f"<html><head><meta charset='utf-8'><title>Regression tree with ratios</title>{V.CSS}"
            f"{EXTRA_CSS}</head><body><h1>Alternative to k, with derived ratios — v5, DT_uncert</h1>"
            f"<div class='note'>5992 runs, 749 economies. Drivers: 15 uncertain parameters + "
            f"supply ratios <b>k_small, k_large, g, supply_margin, h2_vs_elec</b> (computed from "
            f"the sampled parameters, see <a href='../derived/index.html'>derived ratios</a>). "
            f"Compare with the {a('../regtree/index.html', 'base-driver regression tree')}. "
            f"Obsidian: <i>Analysis/2026-11-03 Increasing DT accuracy</i>. "
            f"{a('../index.html', 'back to the accuracy study')}</div>"
            f"<p class='toc'>{a('#regtree', 'R&sup2; comparison')}{a('#accuracy', 'accuracy')}</p>"
            + section_regtree("regtree_derived", DERIVED_FINDINGS, DERIVED_ACC_NOTE)
            + "</body></html>")
        pages.append((os.path.join(OUT, "regtree_derived", "index.html"), dhtml))
    for path, h in pages:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(h)
        print("wrote", path)


if __name__ == "__main__":
    main()
