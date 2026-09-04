"""
Overview page for the combined strategies: (1) which values each strategy clusters on,
(2) accuracy + meaningfulness verdict per strategy, (3) overall parameter influence.

Reads strat/comparison.csv, strat/definitions.csv and per-case feature_importance.csv.
Outputs: DT_supplychain/strat/overview.html  (linked from comparison.html)

Run:  python strat_overview.py
"""

import os

import pandas as pd

OUT = r"X:\3000_simulations_24_may_2026\Snellius_3000_simulations\DT_supplychain\strat"


def verdict(best_lift, any_informative):
    if not any_informative:
        return ("degenerate", "#f2b6b6")
    if best_lift >= 0.40:            # best_lift now carries Cohen's kappa
        return ("strong (κ≥0.4)", "#8fd19e")
    if best_lift >= 0.20:
        return ("moderate (κ≥0.2)", "#ffe08a")
    if best_lift >= 0.05:
        return ("weak", "#f2c6a0")
    return ("no skill (κ≈0)", "#f2b6b6")


def bar(v, mx):
    w = 0 if mx == 0 else int(100 * v / mx)
    return f"<div style='background:#63A593;height:10px;width:{w}px;display:inline-block'></div> {v:.0f}%"


def main():
    d = pd.read_csv(os.path.join(OUT, "comparison.csv"))
    defs = pd.read_csv(os.path.join(OUT, "definitions.csv"))
    concept = dict(zip(defs.folder, defs.concept))
    feats_used = dict(zip(defs.folder, defs.features))
    typ = dict(zip(defs.folder, defs.type))

    # (2) best informative per strategy (across subsets)
    rows2, imp_cases = [], []
    for name in defs.folder:
        g = d[d["def"] == name]
        inf = g[g.degenerate == 0]
        any_inf = len(inf) > 0
        if any_inf:
            r = inf.loc[inf.DT_kappa.idxmax()]
        else:
            r = g.loc[g.DT_kappa.idxmax()]
        vtxt, vcol = verdict(r.DT_kappa if any_inf else -1, any_inf)
        rows2.append((name, r.subset, int(r.k), r.silhouette, r.baseline, r.DT_CV,
                      r.DT_bal_acc, r.DT_kappa, r.DT_test, r.RF_CV, vtxt, vcol))
        if any_inf and r.DT_kappa >= 0.20:
            imp_cases.append((name, r.subset, int(r.k)))

    # (2b) best informative per strategy restricted to k>=3 (>=3 clusters)
    rows2b = []
    for name in defs.folder:
        g = d[(d["def"] == name) & (d.degenerate == 0) & (d.k >= 3)]
        if len(g) == 0:
            rows2b.append((name, "—", "—", "—", "—", "—", "—", "—", "—", "—", "no informative k≥3", "#f2b6b6"))
            continue
        r = g.loc[g.DT_kappa.idxmax()]
        vtxt, vcol = verdict(r.DT_kappa, True)
        rows2b.append((name, r.subset, int(r.k), r.silhouette, r.baseline, r.DT_CV,
                       r.DT_bal_acc, r.DT_kappa, r.DT_test, r.RF_CV, vtxt, vcol))

    # (3) overall parameter influence: mean DT importance across informative winners
    agg = {}
    for name, sub, k in imp_cases:
        fp = os.path.join(OUT, name, sub, f"k{k:02d}", "feature_importance.csv")
        if not os.path.isfile(fp):
            continue
        imp = pd.read_csv(fp).set_index("feature")["DT_importance_pct"]
        for f, v in imp.items():
            agg.setdefault(f, []).append(v)
    infl = sorted(((f, sum(v) / len(imp_cases)) for f, v in agg.items()), key=lambda t: -t[1])
    mx = infl[0][1] if infl else 1

    # HTML
    t1 = "".join(f"<tr><td><b>{f}</b></td><td>{typ[f]}</td><td>{concept[f]}</td>"
                 f"<td><code>{feats_used[f].replace(';', ', ')}</code></td></tr>" for f in defs.folder)

    def cvcol(v):
        t = max(0, min(1, (v-0.2)/0.75)); return f"background:rgb({int(255-t*156)},{int(255-t*90)},{int(255-t*108)})"

    def mkrows(rws):
        out = []
        for (n, sub, k, sil, base, acc, bal, kappa, test, rf, vt, vc) in rws:
            acccell = f"<td style='{cvcol(acc)}'>{acc}</td>" if isinstance(acc, float) else f"<td>{acc}</td>"
            kcell = f"<td style='{cvcol(kappa)}'>{kappa}</td>" if isinstance(kappa, float) else f"<td>{kappa}</td>"
            out.append(f"<tr><td><b>{n}</b></td><td>{sub}</td><td>{k}</td><td>{sil}</td><td>{base}</td>"
                       f"{acccell}<td>{bal}</td>{kcell}<td>{test}</td><td>{rf}</td>"
                       f"<td style='background:{vc}'>{vt}</td></tr>")
        return "".join(out)
    t2 = mkrows(rows2)
    t2b = mkrows(rows2b)
    t3 = "".join(f"<tr><td>{f}</td><td>{bar(v, mx)}</td></tr>" for f, v in infl[:12])

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Combined strategies — overview</title>
<style>body{{font-family:Inter,Calibri,sans-serif;margin:26px;color:#161D41}}h1{{font-size:21px}}h2{{font-size:15px;margin-top:26px}}
.sub{{color:#8a8f9c;font-size:13px;max-width:980px}}table{{border-collapse:collapse;font-size:12px}}
th,td{{border:1px solid #d4d8e0;padding:4px 9px;text-align:center}}th{{background:#161D41;color:#fff}}
td:first-child,th:first-child{{text-align:left}}a{{color:#63A593}}code{{font-size:11px}}</style></head><body>
<p><a href="comparison.html">&larr; full accuracy table</a> &nbsp;·&nbsp; <a href="drivers.html">driver heatmap &rarr;</a></p>
<h1>Combined clustering strategies — overview</h1>
<div class="sub">One page: what each strategy clusters on, whether a decision tree can predict its typology from the
sampled inputs (accuracy + lift over baseline = meaningfulness), and which input parameters drive the predictable
strategies overall.</div>

<h2>1 · Values used for clustering</h2>
<table><thead><tr><th>strategy</th><th>type</th><th>idea</th><th>clustered on</th></tr></thead><tbody>{t1}</tbody></table>

<h2>2 · Accuracy &amp; meaningfulness (best informative case per strategy)</h2>
<table><thead><tr><th>strategy</th><th>subset</th><th>k</th><th>silh.</th><th>baseline</th><th>DT&nbsp;acc</th>
<th>bal.acc</th><th>Cohen&nbsp;&kappa;</th><th>DT&nbsp;test</th><th>RF&nbsp;CV</th><th>verdict</th></tr></thead><tbody>{t2}</tbody></table>
<div class="sub"><b>Cohen's &kappa;</b> = (accuracy &minus; chance) / (1 &minus; chance): skill above the majority-class
baseline, corrected for imbalance (κ=0 no skill, κ=1 perfect). <b>bal.acc</b> = mean per-class recall. Verdict by κ:
strong &ge;0.40 · moderate &ge;0.20 · weak &ge;0.05 · no skill &lt;0.05 · degenerate = every k has a &gt;80% cluster.
Best case selected by κ (not raw accuracy).</div>

<h2>2b · Same, restricted to k &ge; 3 (at least 3 clusters)</h2>
<table><thead><tr><th>strategy</th><th>subset</th><th>k</th><th>silh.</th><th>baseline</th><th>DT&nbsp;acc</th>
<th>bal.acc</th><th>Cohen&nbsp;&kappa;</th><th>DT&nbsp;test</th><th>RF&nbsp;CV</th><th>verdict</th></tr></thead><tbody>{t2b}</tbody></table>
<div class="sub">Best informative case with &ge;3 clusters per strategy — the typologies rich enough to be worth naming.</div>

<h2>3 · Overall parameter influence (mean DT importance across predictable strategies)</h2>
<table><thead><tr><th>input parameter</th><th>mean importance</th></tr></thead><tbody>{t3}</tbody></table>
</body></html>"""
    open(os.path.join(OUT, "overview.html"), "w", encoding="utf-8").write(html)

    # link from comparison.html
    cp = os.path.join(OUT, "comparison.html")
    c = open(cp, encoding="utf-8").read()
    if "overview.html" not in c:
        c = c.replace('<p><a href="drivers.html">',
                      '<p><a href="overview.html">&rarr; Overview (values · accuracy · influence)</a></p>\n<p><a href="drivers.html">', 1)
        open(cp, "w", encoding="utf-8").write(c)
    print("overview.html written; top drivers:", [f"{f} {v:.0f}%" for f, v in infl[:5]])


if __name__ == "__main__":
    main()
