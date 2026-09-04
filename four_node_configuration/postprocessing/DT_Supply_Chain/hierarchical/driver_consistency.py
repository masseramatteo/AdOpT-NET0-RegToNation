"""
Second comparison page: are the parameters that drive central/decentral the same
across the high-accuracy (DT_CV > THRESHOLD) cases?

Builds a feature-importance heatmap (feature x case, DT importance %) for the >90%
cases, a mean-importance ranking, and a short narrative. Links to/from comparison.html.

Outputs: DT_supplychain/decentr/drivers.html
Run:  python driver_consistency.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = r"X:\3000_simulations_24_may_2026\Snellius_3000_simulations\DT_supplychain\decentr"
THRESH = 0.90


def cell_color(v):
    t = max(0.0, min(1.0, v / 60.0))   # 0..60% -> white..teal
    r = int(255 - t * (255 - 99)); g = int(255 - t * (255 - 165)); b = int(255 - t * (255 - 147))
    return f"background:rgb({r},{g},{b})"


def main():
    d = pd.read_csv(os.path.join(BASE, "comparison.csv"))
    hi = d[d.DT_CV > THRESH].copy().sort_values(["def", "subset", "k"])
    cases, imps = [], {}
    for _, r in hi.iterrows():
        fp = os.path.join(BASE, r["def"], r["subset"], f"k{int(r['k']):02d}", "feature_importance.csv")
        imp = pd.read_csv(fp).set_index("feature")["DT_importance_pct"]
        tag = f"{r['def']}/{r['subset']}/k{int(r['k'])}"
        cases.append((tag, round(r["DT_CV"], 2)))
        imps[tag] = imp
    M = pd.DataFrame({t: imps[t] for t, _ in cases}).fillna(0.0)
    M["MEAN"] = M.mean(axis=1)
    M = M.sort_values("MEAN", ascending=False)

    # header row
    head = "".join(f"<th>{t}<br><span class='cv'>{cv:.2f}</span></th>" for t, cv in cases) + "<th>MEAN</th>"
    body = []
    for feat, row in M.iterrows():
        tds = "".join(f"<td style='{cell_color(row[t])}'>{row[t]:.0f}</td>" for t, _ in cases)
        tds += f"<td style='{cell_color(row['MEAN'])};font-weight:700'>{row['MEAN']:.0f}</td>"
        hl = " class='keyfeat'" if "electricity_availability" in feat else ""
        body.append(f"<tr><th{hl}>{feat}</th>{tds}</tr>")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Driver consistency (&gt;90% cases)</title>
<style>body{{font-family:Inter,Calibri,sans-serif;margin:26px;color:#161D41}}
h1{{font-size:20px}} .sub{{color:#8a8f9c;font-size:13px;max-width:960px;margin-bottom:14px}}
table{{border-collapse:collapse;font-size:12px}} th,td{{border:1px solid #d4d8e0;padding:4px 8px;text-align:center}}
thead th{{background:#161D41;color:#fff}} tbody th{{background:#eef0f4;text-align:left;white-space:nowrap}}
tbody th.keyfeat{{background:#FFCD00;font-weight:700}} .cv{{font-weight:400;color:#9fb0d0;font-size:10px}}
a{{color:#63A593}} .note{{margin-top:18px;font-size:13px;max-width:960px;line-height:1.5}}</style></head><body>
<p><a href="comparison.html">&larr; back to accuracy comparison</a></p>
<h1>Do the same parameters drive central vs decentral? (cases with DT&nbsp;CV&nbsp;&gt;&nbsp;{THRESH:.0%})</h1>
<div class="sub">Feature importance (%) of the driver decision tree, for every case above {THRESH:.0%} accuracy.
Columns = case (def/subset/k, with CV); rows = input parameter (sorted by mean importance). Greener = more
important. <b>Electricity-availability rows highlighted.</b></div>
<table><thead><tr><th>feature \\ case</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>
<div class="note"><b>Finding.</b> The central/decentral decision is <b>consistently driven by electricity
availability</b> across all &gt;90% cases — <code>electricity_availability_large</code> for definition A
(local share of <i>total</i> demand) and <code>electricity_availability_small</code> for definition B
(small-cluster self-sufficiency). Same physics: where electricity is available decides where H&#8322; is produced.
At low k (2&ndash;3, the clean central-vs-decentral split) this single driver dominates; only at finer k do
secondary factors enter (solar availability, total demand, demand-level/unbalance ratios, H&#8322; import price),
and these vary by case.</div>
</body></html>"""
    out = os.path.join(BASE, "drivers.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved -> {out}\n{len(cases)} cases, {len(M)} features")

    # add a link at the top of comparison.html (once)
    cpath = os.path.join(BASE, "comparison.html")
    c = open(cpath, encoding="utf-8").read()
    link = '<p><a href="drivers.html">&rarr; Driver consistency across &gt;90% cases</a></p>'
    if "drivers.html" not in c:
        c = c.replace("</h1>", "</h1>\n" + link, 1)
        open(cpath, "w", encoding="utf-8").write(c)
        print("linked from comparison.html")


if __name__ == "__main__":
    main()
