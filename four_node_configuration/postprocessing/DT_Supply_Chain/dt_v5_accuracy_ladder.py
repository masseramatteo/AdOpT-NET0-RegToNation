"""
v5 DT_uncert: accuracy ladder of complete driver trees, picked from the sweep.

Reads DT_uncert_accuracy/sweep_all.csv (dt_v5_accuracy_sweep.py) and, per case,
picks a ladder of trees on GROUPED-CV accuracy (folds grouped by economy):
    base      best tree with <= BASE_LEAVES leaves (must be >= MIN_ACC, else the
              smallest tree reaching MIN_ACC)
    +5pt      smallest tree reaching base + 0.05 (if below the plateau)
    mid       smallest trees reaching base + 1/3 and 2/3 of the gap to the plateau
    plateau   smallest tree within PLATEAU_TOL of the best grouped accuracy
Steps closer than MIN_STEP in accuracy to the previous one are dropped (CV noise).

Each selected tree is refit on all runs and written in full:
    trees/<case>/<n>_<step>_L<leaves>/{tree.html (schematic), driver_tree.png/.pdf,
                                        rules.txt, tree.txt}
    trees/<case>/ladder.csv, trees/ladder_all.csv, trees/index.html

Run:  python dt_v5_accuracy_ladder.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

import config as C
import dt_v5 as V
import dt_v5_uncertain as U
import dt_v5_accuracy_sweep as S
import render_driver_tree_html as R

OUT = os.path.join(S.OUT, "trees")
DRIVERS = S.DRIVERS
BASE_LEAVES = 8
MIN_ACC = 0.65
PLATEAU_TOL = 0.005
MIN_STEP = 0.01


def cfg_of(r):
    def num(v):
        return None if pd.isna(v) else int(v)
    return dict(max_depth=num(r.max_depth), max_leaf_nodes=num(r.max_leaf_nodes),
                min_samples_leaf=int(r.min_samples_leaf),
                min_samples_split=int(r.min_samples_split), ccp_alpha=float(r.ccp_alpha),
                criterion=r.criterion,
                class_weight=None if pd.isna(r.class_weight) else r.class_weight)


def smallest_reaching(df, thr):
    hit = df[df.acc_grouped >= thr]
    if hit.empty:
        return None
    return hit.sort_values(["leaves", "acc_grouped"], ascending=[True, False]).iloc[0]


def pick_ladder(df):
    small = df[df.leaves <= BASE_LEAVES]
    base = small.sort_values(["acc_grouped", "leaves"], ascending=[False, True]).iloc[0] \
        if not small.empty else None
    if base is None or base.acc_grouped < MIN_ACC:
        base = smallest_reaching(df, MIN_ACC)
    top = df.acc_grouped.max()
    plat = smallest_reaching(df, top - PLATEAU_TOL)
    gap = plat.acc_grouped - base.acc_grouped
    cand = [("base", base),
            ("plus5", smallest_reaching(df, base.acc_grouped + 0.05)),
            ("mid1", smallest_reaching(df, base.acc_grouped + gap / 3)),
            ("mid2", smallest_reaching(df, base.acc_grouped + 2 * gap / 3)),
            ("plateau", plat)]
    cand = [(s, r) for s, r in cand if r is not None and r.leaves <= plat.leaves]
    cand.sort(key=lambda t: (t[1].leaves, t[1].acc_grouped))
    ladder = []
    for s, r in cand:
        if ladder and r.acc_grouped < ladder[-1][1].acc_grouped + MIN_STEP:
            if s == "plateau":            # plateau wins over a near-equal step
                ladder[-1] = (s, r)
            continue
        ladder.append((s, r))
    if ladder[-1][0] != "plateau" and ladder[-1][1].leaves != plat.leaves:
        ladder.append(("plateau", plat))
    return ladder


def schematic_html(dfm, clf, X, y, names, subs, out_html, title, sub_text):
    """dt_v5_uncertain.schematic_html, but for an already fitted tree of any shape."""
    dfm = dfm.copy()
    dfm["cluster_final"] = y
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
    color = {names[i]: U.GRAD[i % len(U.GRAD)] for i in range(len(names))}
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


def main():
    sweep = pd.read_csv(os.path.join(S.OUT, "sweep_all.csv"))
    f = V.load(C.EXTRACTED_RESULTS)
    raw = pd.read_excel(C.EXTRACTED_RESULTS, sheet_name="Results")
    dfm = raw.copy()
    dfm["import_intensity"] = f["imported"].values
    X = f[DRIVERS].fillna(0.0)
    g = f.economy.values
    os.makedirs(OUT, exist_ok=True)

    allrows = []
    for group, dname, k in S.CASES:
        case = f"{group}/{dname}/k{k:02d}"
        tag = f"{group}_{dname}_k{k:02d}"
        y = S.labels(f, group, dname, k)
        base_acc = pd.Series(y).value_counts(normalize=True).max()
        feats = dict((n, ft) for n, ft, _ in V.GROUPS[group])[dname]

        prof = f.groupby(y)[V.PROFILE].mean()
        prof["n"] = pd.Series(y).value_counts().sort_index()
        av = f[U.AXIS[(group, dname)][0]].values
        rng = [(av[y == i].min(), av[y == i].max(), av[y == i].mean()) for i in range(k)]
        cnames, rtxt, subs = U.cluster_naming(group, dname, k, prof, rng)
        rnames = [f"{n} ({r})" for n, r in zip(cnames, rtxt)]

        df = sweep[sweep.case == case]
        ladder = pick_ladder(df)
        cdir = os.path.join(OUT, tag)
        os.makedirs(cdir, exist_ok=True)
        rows = []
        for i, (step, r) in enumerate(ladder):
            p = cfg_of(r)
            clf = DecisionTreeClassifier(random_state=S.RS, **p)
            m = S.score(clf, X.values, y, g)            # re-check; refits on all runs
            clf.fit(X, y)
            nl, dp = clf.get_n_leaves(), clf.get_depth()
            sdir = os.path.join(cdir, f"{i}_{step}_L{nl}")
            os.makedirs(sdir, exist_ok=True)
            head = (f"{case} | {step} | {nl} leaves, depth {dp} | grouped acc "
                    f"{m['acc_grouped']:.3f} kappa {m['kappa_grouped']:.3f} | ungrouped acc "
                    f"{m['acc_ungrouped']:.3f} | train acc {m['acc_train']:.3f} | "
                    f"baseline {base_acc:.3f}")
            cfg_txt = ", ".join(f"{a}={b}" for a, b in p.items())

            rules = V.leaf_rules(clf, DRIVERS, rnames)
            with open(os.path.join(sdir, "rules.txt"), "w", encoding="utf-8") as fh:
                fh.write(f"# {head}\n# config: {cfg_txt}\n# clustered on: {', '.join(feats)}\n\n")
                for rr in sorted(rules, key=lambda q: -q["n"]):
                    mark = "" if rr["purity"] >= 0.70 else "  [mixed]"
                    fh.write(f"[{rr['purity']:.0%}, n={rr['n']}] {rr['cls']}{mark}\n"
                             f"    IF {rr['rule']}\n")
            with open(os.path.join(sdir, "tree.txt"), "w", encoding="utf-8") as fh:
                fh.write(f"# {head}\n# config: {cfg_txt}\n\n")
                fh.write(export_text(clf, feature_names=DRIVERS, class_names=rnames,
                                     show_weights=True, decimals=2, max_depth=100))

            w = min(max(14, nl * 1.1), 160)
            h = min(max(7, dp * 1.6), 40)
            fig, ax = plt.subplots(figsize=(w, h))
            plot_tree(clf, feature_names=DRIVERS, class_names=cnames, filled=True,
                      rounded=True, proportion=True, precision=1,
                      fontsize=7 if nl <= 16 else 5, ax=ax)
            ax.set_title(head, fontsize=12, color=C.UU_NAVY)
            for ext, dpi in (("png", 110 if nl > 40 else 160), ("pdf", None)):
                fig.savefig(os.path.join(sdir, f"driver_tree.{ext}"), dpi=dpi,
                            bbox_inches="tight")
            plt.close(fig)

            sub_text = (f"{len(f)} optimizations, {f.economy.nunique()} economies. "
                        f"{step}: {nl}-leaf tree (depth {dp}), grouped accuracy "
                        f"{m['acc_grouped']:.2f} / &kappa; {m['kappa_grouped']:.2f} "
                        f"(baseline {base_acc:.2f}). Clustered on {', '.join(feats)}.")
            schematic_html(dfm, clf, X, y, cnames, subs, os.path.join(sdir, "tree.html"),
                           f"{group} / {dname} &mdash; k={k} &mdash; {step} ({nl} leaves)",
                           sub_text)

            pure = np.mean([rr["purity"] >= 0.90 for rr in rules])
            rows.append(dict(case=case, step_no=i, step=step, leaves=nl, depth=dp,
                             acc_grouped=round(m["acc_grouped"], 3),
                             kappa_grouped=round(m["kappa_grouped"], 3),
                             acc_ungrouped=round(m["acc_ungrouped"], 3),
                             acc_train=round(m["acc_train"], 3), baseline=round(base_acc, 3),
                             leaves_pure90_pct=round(100 * pure), config=cfg_txt,
                             folder=os.path.relpath(sdir, OUT)))
            print(f"{case:22s} {step:8s} L{nl:<4d} d{dp:<3d} grouped {m['acc_grouped']:.3f} "
                  f"(sweep {r.acc_grouped:.3f}) train {m['acc_train']:.3f}")
        pd.DataFrame(rows).to_csv(os.path.join(cdir, "ladder.csv"), index=False)
        allrows += rows

    lad = pd.DataFrame(allrows)
    lad.to_csv(os.path.join(OUT, "ladder_all.csv"), index=False)
    body = []
    for case, d in lad.groupby("case", sort=False):
        body.append(f"<h2>{case}</h2><table><tr><th>step</th><th>leaves</th><th>depth</th>"
                    f"<th>grouped acc</th><th>&kappa;</th><th>ungrouped</th><th>train</th>"
                    f"<th>leaves &ge;90% pure</th><th>files</th></tr>")
        for _, r in d.iterrows():
            fo = r.folder.replace(os.sep, "/")
            body.append(f"<tr><td>{r.step}</td><td>{r.leaves}</td><td>{r.depth}</td>"
                        f"<td><b>{r.acc_grouped:.3f}</b></td><td>{r.kappa_grouped:.3f}</td>"
                        f"<td>{r.acc_ungrouped:.3f}</td><td>{r.acc_train:.3f}</td>"
                        f"<td>{r.leaves_pure90_pct}%</td>"
                        f"<td><a href='{fo}/tree.html'>schematic</a> · "
                        f"<a href='{fo}/driver_tree.pdf'>pdf</a> · "
                        f"<a href='{fo}/rules.txt'>rules</a></td></tr>")
        body.append("</table>")
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(f"<html><head><meta charset='utf-8'><title>DT accuracy ladder</title>{V.CSS}"
                 f"</head><body><h1>Driver-tree accuracy ladder (grouped CV by economy)</h1>"
                 f"<div class='note'>base = best tree &le;{BASE_LEAVES} leaves (&ge;{MIN_ACC}); "
                 f"plus5 = +5 pt; mid = 1/3, 2/3 of the gap; plateau = smallest tree within "
                 f"{PLATEAU_TOL} of the best. Steps &lt;{MIN_STEP} apart dropped.</div>"
                 + "".join(body) + "</body></html>")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
