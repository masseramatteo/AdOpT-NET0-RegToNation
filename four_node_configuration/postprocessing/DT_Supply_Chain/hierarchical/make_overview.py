"""
Generic overview page for a clustering-analysis folder (strat / sysmix / ...).
Reads <folder>/comparison.csv + definitions.csv + per-case feature_importance.csv and
writes <folder>/overview.html with: (1) values clustered on, (2) best case per strategy
by Cohen kappa + verdict, (2b) same restricted to k>=3, (3) overall parameter influence.

Run:  python make_overview.py <folder-name-under-DT_supplychain>
      e.g. python make_overview.py sysmix
"""

import os
import sys

import pandas as pd

BASE = r"X:\3000_simulations_24_may_2026\Snellius_3000_simulations\DT_supplychain"


def verdict(kp, informative=True):
    if not informative:
        return ("degenerate", "#f2b6b6")
    if kp >= 0.40:
        return ("strong (κ≥0.4)", "#8fd19e")
    if kp >= 0.20:
        return ("moderate (κ≥0.2)", "#ffe08a")
    if kp >= 0.05:
        return ("weak", "#f2c6a0")
    return ("no skill", "#f2b6b6")


def cvcol(v):
    if not isinstance(v, float):
        return ""
    t = max(0, min(1, (v - 0.2) / 0.75))
    return f"background:rgb({int(255-t*156)},{int(255-t*90)},{int(255-t*108)})"


def best_row(g):
    inf = g[g.degenerate == 0] if "degenerate" in g else g
    if len(inf) == 0:
        r = g.loc[g.DT_kappa.idxmax()]; return r, False
    return inf.loc[inf.DT_kappa.idxmax()], True


def mkrows(rws):
    out = []
    for (n, sub, k, sil, base, acc, bal, kp, rf, vt, vc) in rws:
        acccell = f"<td style='{cvcol(acc)}'>{acc}</td>" if isinstance(acc, float) else f"<td>{acc}</td>"
        kcell = f"<td style='{cvcol(kp)}'>{kp}</td>" if isinstance(kp, float) else f"<td>{kp}</td>"
        out.append(f"<tr><td><b>{n}</b></td><td>{sub}</td><td>{k}</td><td>{sil}</td><td>{base}</td>"
                   f"{acccell}<td>{bal}</td>{kcell}<td>{rf}</td><td style='background:{vc}'>{vt}</td></tr>")
    return "".join(out)


def main(folder):
    OUT = os.path.join(BASE, folder)
    d = pd.read_csv(os.path.join(OUT, "comparison.csv"))
    defs = pd.read_csv(os.path.join(OUT, "definitions.csv"))
    concept = dict(zip(defs.folder, defs.concept)); feats = dict(zip(defs.folder, defs.features))
    typ = dict(zip(defs.folder, defs.type))

    rows2, rows2b, imp_cases = [], [], []
    for name in defs.folder:
        g = d[d["def"] == name]
        r, inf = best_row(g)
        vt, vc = verdict(r.DT_kappa if inf else -1, inf)
        rows2.append((name, r.subset, int(r.k), r.silhouette, r.baseline, r.DT_CV, r.DT_bal_acc,
                      r.DT_kappa, r.RF_CV, vt, vc))
        if inf and r.DT_kappa >= 0.20:
            imp_cases.append((name, r.subset, int(r.k)))
        g3 = g[(g.degenerate == 0) & (g.k >= 3)] if "degenerate" in g else g[g.k >= 3]
        if len(g3):
            r3 = g3.loc[g3.DT_kappa.idxmax()]; vt3, vc3 = verdict(r3.DT_kappa)
            rows2b.append((name, r3.subset, int(r3.k), r3.silhouette, r3.baseline, r3.DT_CV,
                           r3.DT_bal_acc, r3.DT_kappa, r3.RF_CV, vt3, vc3))
        else:
            rows2b.append((name, "—", "—", "—", "—", "—", "—", "—", "—", "no k≥3", "#f2b6b6"))

    agg = {}
    for name, sub, k in imp_cases:
        fp = os.path.join(OUT, name, sub, f"k{k:02d}", "feature_importance.csv")
        if os.path.isfile(fp):
            for f, v in pd.read_csv(fp).set_index("feature")["DT_importance_pct"].items():
                agg.setdefault(f, []).append(v)
    infl = sorted(((f, sum(v) / max(len(imp_cases), 1)) for f, v in agg.items()), key=lambda t: -t[1])
    mx = infl[0][1] if infl else 1
    t1 = "".join(f"<tr><td><b>{f}</b></td><td>{typ[f]}</td><td>{concept[f]}</td>"
                 f"<td><code>{feats[f].replace(';', ', ')}</code></td></tr>" for f in defs.folder)
    t3 = "".join(f"<tr><td>{f}</td><td><div style='background:#63A593;height:10px;width:{int(100*v/mx)}px;"
                 f"display:inline-block'></div> {v:.0f}%</td></tr>" for f, v in infl[:12])

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{folder} — overview</title>
<style>body{{font-family:Inter,Calibri,sans-serif;margin:26px;color:#161D41}}h1{{font-size:21px}}h2{{font-size:15px;margin-top:24px}}
.sub{{color:#8a8f9c;font-size:13px;max-width:980px}}table{{border-collapse:collapse;font-size:12px}}
th,td{{border:1px solid #d4d8e0;padding:4px 9px;text-align:center}}th{{background:#161D41;color:#fff}}
td:first-child,th:first-child{{text-align:left}}a{{color:#63A593}}code{{font-size:11px}}</style></head><body>
<p><a href="comparison.html">&larr; full comparison table</a></p>
<h1>{folder} — overview</h1>
<div class="sub">What each strategy clusters on, whether a decision tree can predict the typology from sampled inputs
(Cohen κ = skill above the majority baseline, imbalance-corrected), and which inputs drive the predictable ones.</div>
<h2>1 · Values used for clustering</h2>
<table><thead><tr><th>strategy</th><th>type</th><th>idea</th><th>clustered on</th></tr></thead><tbody>{t1}</tbody></table>
<h2>2 · Accuracy &amp; meaningfulness (best by κ per strategy)</h2>
<table><thead><tr><th>strategy</th><th>subset</th><th>k</th><th>silh.</th><th>baseline</th><th>DT&nbsp;acc</th>
<th>bal.acc</th><th>Cohen&nbsp;κ</th><th>RF&nbsp;CV</th><th>verdict</th></tr></thead><tbody>{mkrows(rows2)}</tbody></table>
<div class="sub">κ verdict: strong ≥0.40 · moderate ≥0.20 · weak ≥0.05 · degenerate = every k has a &gt;80% cluster.</div>
<h2>2b · Same, restricted to k ≥ 3</h2>
<table><thead><tr><th>strategy</th><th>subset</th><th>k</th><th>silh.</th><th>baseline</th><th>DT&nbsp;acc</th>
<th>bal.acc</th><th>Cohen&nbsp;κ</th><th>RF&nbsp;CV</th><th>verdict</th></tr></thead><tbody>{mkrows(rows2b)}</tbody></table>
<h2>3 · Overall parameter influence (mean DT importance across κ≥0.2 strategies)</h2>
<table><thead><tr><th>input parameter</th><th>mean importance</th></tr></thead><tbody>{t3}</tbody></table>
</body></html>"""
    open(os.path.join(OUT, "overview.html"), "w", encoding="utf-8").write(html)
    print(f"overview.html -> {OUT}; top drivers:", [f"{f} {v:.0f}%" for f, v in infl[:5]])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sysmix")
