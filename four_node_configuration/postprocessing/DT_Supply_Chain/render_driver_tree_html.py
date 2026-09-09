"""
Driver tree rendered as an interactive HTML, in the style of the user's
v3_leafs/decision_tree_magnitude.html.

Splits test sampled input parameters (human labels: near/far, cheap/dear, ...).
Each LEAF is drawn as a 4-node hydrogen network schematic aggregated over the runs
that fall in that leaf:
  - large clusters L1/L2 = navy boxes, size = mean electrolyzer capacity
  - small clusters S1/S2 = teal circles, radius = mean electrolyzer capacity
  - pipelines: width = mean capacity, red = high-pressure; solid/dashed/faint by
    how often the link is built (>=60% / 30-60% / 10-30%)
  - gold arrows = import volume at large clusters
  - supply-mix bar (local vs import), dual/mesh flags, N and leaf purity
Leaf label = majority infrastructure archetype (cluster_final from explain_tree.py).

Run:  python render_driver_tree_html.py
Out:  <FIG_DIR>/driver_tree_magnitude.html
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import config as C

# tree size for readability (depth 4 = driver-accuracy plateau, ~8 leaves)
HTML_MAX_DEPTH = C.DRIVER_MAX_DEPTH
HTML_MAX_LEAF_NODES = 8

# arc key -> column stem
ARC_COLS = {
    "L1L2": "Large_cluster1_to_Large_cluster2",
    "L1S1": "Large_cluster1_to_Small_cluster1",
    "L1S2": "Large_cluster1_to_Small_cluster2",
    "L2S1": "Large_cluster2_to_Small_cluster1",
    "L2S2": "Large_cluster2_to_Small_cluster2",
    "S1S2": "Small_cluster1_to_Small_cluster2",
}
ELZR_COL = {
    "L1": "Large_cluster1_Electrolyzer_big",
    "L2": "Large_cluster2_Electrolyzer_big",
    "S1": "Small_cluster1_Electrolyzer_small",
    "S2": "Small_cluster2_Electrolyzer_small",
}
IMP_COL = {"L1": "Large_cluster1_hydrogen_import_sum",
           "L2": "Large_cluster2_hydrogen_import_sum"}
PROD_COLS = [f"{n}_TOTAL_H2_production" for n in
             ["Large_cluster1", "Large_cluster2", "Small_cluster1", "Small_cluster2"]]

FEAT = {
    "total_demand_TWh": ["Total demand", "TWh/y"],
    "demand_level_ratio": ["Demand level ratio", ""],
    "unbalance_ratio": ["Unbalance ratio", ""],
    "electricity_availability_small": ["Elec. availability (small)", "GWh"],
    "electricity_availability_large": ["Elec. availability (large)", "GWh"],
    "import_availability_ratio": ["Import availability", "ratio"],
    "electricity_price_avg": ["Electricity price", "€/MWh"],
    "electricity_standard_dev": ["Price volatility σ", "€/MWh"],
    "hydrogen_import_price": ["H₂ import price", "€/MWh"],
    "solar_cf_realized_mean": ["Solar mean capacity factor", "-"],
    "distance_Large_cluster1_to_Large_cluster2_km": ["Distance large→large", "km"],
    "distance_Small_cluster1_to_Small_cluster2_km": ["Distance small→small", "km"],
    "distance_from_large_cluster": ["Distance small→large", "km"],
}
PAIRS = {
    "total_demand_TWh": ["low", "high"],
    "demand_level_ratio": ["balanced", "skewed"],
    "unbalance_ratio": ["even", "uneven"],
    "electricity_availability_small": ["scarce", "abundant"],
    "electricity_availability_large": ["scarce", "abundant"],
    "import_availability_ratio": ["low", "high"],
    "electricity_price_avg": ["cheap", "dear"],
    "electricity_standard_dev": ["stable", "volatile"],
    "hydrogen_import_price": ["cheap", "dear"],
    "solar_cf_realized_mean": ["low", "high"],
    "distance_Large_cluster1_to_Large_cluster2_km": ["near", "far"],
    "distance_Small_cluster1_to_Small_cluster2_km": ["near", "far"],
    "distance_from_large_cluster": ["near", "far"],
}
COLOR = {
    "Minimal network": "#8a8f9c",
    "Local self-sufficient": "#63A593",
    "Centralized backbone": "#C00935",
    "Import-reliant": "#b8900a",
    "Small-small meshed": "#6b4e9e",
    "Dual-supply meshed": "#e08a2b",
}
SUB = {
    "Minimal network": "little to no transport",
    "Local self-sufficient": "smalls cover own demand; larges linked",
    "Centralized backbone": "larges linked; central production",
    "Import-reliant": "demand largely met by import",
    "Small-small meshed": "small–small link present",
    "Dual-supply meshed": "smalls attach to both larges",
}


def _round(thr):
    return round(thr, 0) if abs(thr) >= 100 else round(thr, 2)


def _leaf_info(sub, name_by_cluster):
    """Aggregate one leaf's runs into the info dict the JS renderer expects."""
    n = len(sub)
    maj = sub["cluster_final"].value_counts()
    maj_cluster = int(maj.index[0])
    purity = maj.iloc[0] / n
    info = {
        "n": int(n),
        "label": name_by_cluster.get(maj_cluster, f"C{maj_cluster}"),
        "purity": round(float(purity), 2),
        "impfrac": round(float(sub["import_intensity"].mean()), 3),
        "avg_price": int(round(sub["electricity_price_avg"].mean())),
        "avg_demand": round(float(sub["total_demand_TWh"].mean()), 1),
        "prod": round(float(sub[PROD_COLS].sum(axis=1).mean()), 0),
    }
    # arcs: [built_frac, mean capacity when built, highP fraction when built]
    dual_small = np.zeros(n, dtype=bool)
    for key, stem in ARC_COLS.items():
        hp = sub.get(f"{stem}_hydrogenPipelineOnshore_highP", pd.Series(0, index=sub.index)).fillna(0)
        lp = sub.get(f"{stem}_hydrogenPipelineOnshore_lowP", pd.Series(0, index=sub.index)).fillna(0)
        tot = hp + lp
        built = tot > C.BUILD_EPS
        bf = built.mean()
        cap = float(tot[built].mean()) if built.any() else 0.0
        hpf = float((hp[built] > C.BUILD_EPS).mean()) if built.any() else 0.0
        info[key] = [round(float(bf), 2), round(cap, 1), round(hpf, 2)]

    # dual = a small cluster links to BOTH large clusters
    b = {k: (sub.get(f"{ARC_COLS[k]}_hydrogenPipelineOnshore_highP", 0).fillna(0)
             + sub.get(f"{ARC_COLS[k]}_hydrogenPipelineOnshore_lowP", 0).fillna(0)) > C.BUILD_EPS
         for k in ARC_COLS}
    dual_small = (b["L1S1"] & b["L2S1"]) | (b["L1S2"] & b["L2S2"])
    info["dual"] = round(float(dual_small.mean()), 2)
    info["mesh"] = round(float(b["S1S2"].mean()), 2)

    for n_key, col in ELZR_COL.items():
        info[f"cap_{n_key}"] = round(float(sub[col].fillna(0).mean()), 1)
    for n_key, col in IMP_COL.items():
        info[f"imp_{n_key}"] = round(float(sub[col].fillna(0).mean()), 0)
    return info


def _build_node(tree, node_id, feat_names, leaf_rows, name_by_cluster):
    left = tree.children_left[node_id]
    right = tree.children_right[node_id]
    if left == -1:  # leaf
        sub = leaf_rows[node_id]
        return {"type": "leaf", "info": _leaf_info(sub, name_by_cluster)}
    return {
        "type": "split",
        "feature": feat_names[tree.feature[node_id]],
        "thr": _round(float(tree.threshold[node_id])),
        "left": _build_node(tree, left, feat_names, leaf_rows, name_by_cluster),
        "right": _build_node(tree, right, feat_names, leaf_rows, name_by_cluster),
    }


def build_data():
    labels = pd.read_csv(C.CLUSTER_LABELS_CSV)
    if "cluster_final" not in labels.columns:
        raise RuntimeError("cluster_final missing — run explain_tree.py first.")
    raw = pd.read_excel(C.EXTRACTED_RESULTS)

    df = labels.merge(raw, on="run", how="left", suffixes=("", "_raw"))
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    X = df[inputs].fillna(0.0)
    y = df["cluster_final"]

    clf = DecisionTreeClassifier(max_depth=HTML_MAX_DEPTH,
                                 max_leaf_nodes=HTML_MAX_LEAF_NODES,
                                 min_samples_leaf=C.DRIVER_MIN_SAMPLES_LEAF,
                                 min_samples_split=C.DRIVER_MIN_SAMPLES_SPLIT,
                                 class_weight="balanced",
                                 random_state=C.RANDOM_STATE)
    clf.fit(X, y)
    acc = clf.score(X, y)
    leaf_id = clf.apply(X)
    df["_leaf"] = leaf_id
    leaf_rows = {lid: df[df["_leaf"] == lid] for lid in np.unique(leaf_id)}

    name_by_cluster = {c: C.CLUSTER_NAMES.get(c, f"C{c}") for c in sorted(y.unique())}
    tree_dict = _build_node(clf.tree_, 0, inputs, leaf_rows, name_by_cluster)

    # global scales for sqrt sizing
    def _leaf_vals(fn):
        out = []
        _walk(tree_dict, lambda nd: out.append(fn(nd["info"])) if nd["type"] == "leaf" else None)
        return out

    pipe_max = max([max(nd["info"][k][1] for k in ARC_COLS)
                    for nd in _leaves(tree_dict)] + [1.0])
    scale = {
        "pipe_max": round(pipe_max, 0),
        "large_elzr_max": round(max(max(nd["info"]["cap_L1"], nd["info"]["cap_L2"])
                                    for nd in _leaves(tree_dict)), 0),
        "small_elzr_max": round(max(max(nd["info"]["cap_S1"], nd["info"]["cap_S2"])
                                    for nd in _leaves(tree_dict)), 0),
        "import_max": round(max(max(nd["info"]["imp_L1"], nd["info"]["imp_L2"])
                                for nd in _leaves(tree_dict)), 0),
    }
    print(f"Driver tree for HTML: {len(leaf_rows)} leaves, train acc {acc:.3f}")
    for nd in _leaves(tree_dict):
        i = nd["info"]
        print(f"  {i['label']:<24} N={i['n']:<5} pure {i['purity']*100:.0f}%  "
              f"import {i['impfrac']*100:.0f}%")
    return {"tree": tree_dict, "scale": scale}


def _leaves(node):
    if node["type"] == "leaf":
        return [node]
    return _leaves(node["left"]) + _leaves(node["right"])


def _walk(node, fn):
    fn(node)
    if node["type"] == "split":
        _walk(node["left"], fn)
        _walk(node["right"], fn)


def render_html(data):
    labels_present = [nd["info"]["label"] for nd in _leaves(data["tree"])]
    cbar = "\n".join(
        f' <span class="cchip"><span class="dot" style="background:{COLOR[l]}"></span>{l}</span>'
        for l in COLOR if l in labels_present)
    html = _TEMPLATE
    html = html.replace("/*__CBAR__*/", cbar)
    html = html.replace("/*__DATA__*/", json.dumps(data))
    html = html.replace("/*__FEAT__*/", json.dumps(FEAT))
    html = html.replace("/*__SUB__*/", json.dumps(SUB))
    html = html.replace("/*__COLOR__*/", json.dumps(COLOR))
    html = html.replace("/*__PAIRS__*/", json.dumps(PAIRS))
    os.makedirs(C.FIG_DIR, exist_ok=True)
    out = os.path.join(C.FIG_DIR, "driver_tree_magnitude.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nSaved -> {out}")
    return out


# HTML/CSS/JS template (renderer adapted from the user's v3_leafs prototype).
_TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>H2 infrastructure driver tree</title>
<style>
:root{--navy:#161D41;--teal:#63A593;--red:#C00935;--amber:#b8900a;--orange:#e08a2b;
 --grey:#8a8f9c;--light:#eef0f4;--line:#b7bcc9;--purple:#6b4e9e;}
*{box-sizing:border-box;}
body{margin:0;background:#fbfbfa;color:var(--navy);
 font-family:"Inter","Calibri",system-ui,sans-serif;padding:26px 20px 46px;}
h1{font-size:19px;font-weight:700;margin:0 0 3px;}
.sub{color:var(--grey);font-size:12px;margin-bottom:12px;max-width:960px;}
.cbar{display:flex;gap:15px;margin:2px 0 12px;font-size:11.5px;flex-wrap:wrap;align-items:center;}
.cchip{display:flex;align-items:center;gap:5px;}
.dot{width:11px;height:11px;border-radius:3px;}
#stage{position:relative;overflow-x:auto;}
svg.edges{position:absolute;top:0;left:0;pointer-events:none;}
.split{position:absolute;transform:translate(-50%,-50%);background:#fff;
 border:1.6px solid var(--navy);border-radius:10px;padding:7px 12px;text-align:center;
 min-width:122px;box-shadow:0 2px 5px rgba(0,0,0,.06);z-index:3;}
.split .feat{font-weight:700;font-size:12px;line-height:1.15;}
.split .thr{font-size:10.5px;margin-top:3px;font-family:"JetBrains Mono",monospace;
 background:var(--light);border-radius:5px;padding:1px 6px;display:inline-block;}
.blabel{position:absolute;transform:translate(-50%,-50%);font-size:10px;font-weight:600;
 background:#fbfbfa;color:var(--navy);padding:1px 7px;border:1px solid var(--line);
 border-radius:10px;white-space:nowrap;z-index:4;}
.leaf{position:absolute;transform:translate(-50%,0);background:#fff;border-radius:12px;
 padding:7px 6px 6px;box-shadow:0 2px 9px rgba(0,0,0,.09);width:196px;z-index:2;}
.leaf .title{font-weight:700;font-size:12.5px;text-align:center;margin:0 0 1px;}
.leaf .desc{font-size:9px;color:var(--grey);text-align:center;line-height:1.2;margin-bottom:2px;min-height:22px;}
.leaf .meta{display:flex;justify-content:space-between;font-size:9px;color:var(--grey);
 font-family:"JetBrains Mono",monospace;margin-top:2px;padding:0 4px;}
.leaf .stat{font-size:8.5px;color:var(--grey);text-align:center;margin-top:1px;}
.magbar{margin:4px 3px 0;height:9px;border-radius:3px;overflow:hidden;display:flex;
 border:1px solid #dfe2e8;}
.magbar i{display:block;height:100%;}
.magcap{font-size:8px;color:var(--grey);text-align:center;margin-top:1px;}
.ssnote{font-size:8px;color:var(--grey);text-align:center;margin-top:3px;font-style:italic;}
.flags{text-align:center;margin-top:2px;min-height:11px;}
.flag{display:inline-block;padding:0 5px;border-radius:6px;margin:0 2px;font-size:8.5px;font-weight:600;}
.flag.dual{background:#efeaf7;color:var(--purple);}
.flag.mesh{background:#fdeff2;color:var(--red);}
.legend{margin:22px 0 0;background:var(--light);border:1px solid var(--line);border-radius:12px;
 padding:13px 18px;display:grid;grid-template-columns:repeat(3,1fr);gap:8px 20px;font-size:11px;max-width:900px;}
.legend .row{display:flex;align-items:center;gap:9px;}
</style></head><body>
<h1>From input parameters to hydrogen infrastructure design</h1>
<div class="sub">Driver tree over 2719 optimizations. Each diamond tests one sampled input parameter; each leaf shows the representative network of the runs it contains. <b>Line thickness = pipeline capacity, node size = electrolyzer capacity, arrow size = import volume.</b> Solid links are built in &ge;60% of a leaf's runs, dashed 30&ndash;60%, faint grey 10&ndash;30%.</div>
<div class="cbar">
/*__CBAR__*/
</div>
<div id="stage"></div>

<div class="legend">
 <div class="row"><svg width="40" height="22"><rect x="3" y="7" width="14" height="9" rx="2" fill="#161D41"/><rect x="21" y="3" width="17" height="16" rx="3" fill="#161D41"/></svg>large cluster — box size = electrolyzer capacity</div>
 <div class="row"><svg width="40" height="22"><circle cx="10" cy="11" r="4" fill="#63A593" stroke="#161D41" stroke-width="1.4"/><circle cx="28" cy="11" r="8" fill="#63A593" stroke="#161D41" stroke-width="1.4"/></svg>small cluster — radius = electrolyzer capacity</div>
 <div class="row"><svg width="40" height="22"><line x1="3" y1="7" x2="37" y2="7" stroke="#161D41" stroke-width="2"/><line x1="3" y1="16" x2="37" y2="16" stroke="#161D41" stroke-width="8"/></svg>built (&ge;60% of runs) — width = capacity</div>
 <div class="row"><svg width="40" height="22"><line x1="3" y1="11" x2="37" y2="11" stroke="#161D41" stroke-width="5" stroke-opacity=".45" stroke-dasharray="4 4"/></svg>occasional (30–60% of runs)</div>
 <div class="row"><svg width="40" height="22"><line x1="3" y1="11" x2="37" y2="11" stroke="#8a8f9c" stroke-width="1.4" stroke-opacity=".55" stroke-dasharray="2.5 3.5"/></svg>rare (10–30%) — width not to scale</div>
 <div class="row"><svg width="40" height="22"><line x1="3" y1="11" x2="37" y2="11" stroke="#C00935" stroke-width="7"/></svg>high-pressure pipeline</div>
 <div class="row"><svg width="46" height="28"><line x1="13" y1="4" x2="13" y2="15" stroke="#b8900a" stroke-width="2.6" stroke-linecap="butt"/><path d="M 13 22 L 9.6 13.5 L 16.4 13.5 Z" fill="#b8900a"/><line x1="33" y1="4" x2="33" y2="14" stroke="#b8900a" stroke-width="5" stroke-linecap="butt"/><path d="M 33 24 L 27.5 11.5 L 38.5 11.5 Z" fill="#b8900a"/></svg>import — thickness = volume</div>
 <div class="row"><svg width="40" height="22"><rect x="3" y="7" width="22" height="8" fill="#63A593"/><rect x="25" y="7" width="12" height="8" fill="#b8900a"/></svg>supply mix bar: local vs import</div>
 <div class="row"><span class="flag dual">dual</span>a small links to both larges</div>
 <div class="row"><span class="flag mesh">mesh</span>small–small link present</div>
 <div class="row" style="color:#8a8f9c">N = runs · pure = leaf purity</div>
</div>

<script>
const DATA=/*__DATA__*/, FEAT=/*__FEAT__*/, SUB=/*__SUB__*/;
const TREE=DATA.tree, SC=DATA.scale;
const COLOR=/*__COLOR__*/;
const N={L1:[98,64],L2:[98,132],S1:[36,98],S2:[160,98]};
const LK={L1L2:['L1','L2'],L1S1:['L1','S1'],L1S2:['L1','S2'],L2S1:['L2','S1'],L2S2:['L2','S2'],S1S2:['S1','S2']};
const PAIRS=/*__PAIRS__*/;

const sq=(v,max)=>Math.sqrt(Math.max(v,0)/max);

function leafSVG(i,uid){
 let s=`<svg viewBox="0 0 196 196" width="186" height="186">`;
 s+=`<defs><marker id="${uid}a" markerWidth="3.6" markerHeight="3.6" refX="3.2" refY="1.8" orient="auto"><path d="M0,0 L3.4,1.8 L0,3.6 Z" fill="#b8900a"/></marker></defs>`;
 const T_MAIN=0.60, T_OCC=0.30, T_RARE=0.10;
 for(const k in LK){
  const [a,b]=LK[k]; const built=i[k][0], capv=i[k][1], hpv=i[k][2];
  if(built<T_RARE) continue;
  const [x1,y1]=N[a],[x2,y2]=N[b];
  const col=(hpv>0.5)?'#C00935':'#161D41';
  if(built>=T_MAIN){
    const w=Math.max(1.2, sq(capv,SC.pipe_max)*11);
    s+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="${w.toFixed(1)}" stroke-linecap="round"/>`;
  }else if(built>=T_OCC){
    const w=Math.max(1.2, sq(capv,SC.pipe_max)*11);
    s+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="${w.toFixed(1)}" stroke-opacity="0.45" stroke-dasharray="4 4" stroke-linecap="round"/>`;
  }else{
    s+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#8a8f9c" stroke-width="1.4" stroke-opacity="0.55" stroke-dasharray="2.5 3.5" stroke-linecap="round"/>`;
  }
 }
 [['L1','imp_L1',-1],['L2','imp_L2',1]].forEach(([n,key,dir])=>{
   const v=i[key]; if(!v||v<8000) return;
   const [x,y]=N[n];
   const f=sq(v,SC.import_max);
   const boxH=Math.max(15, sq(i['cap_'+n],SC.large_elzr_max)*34);
   const gap=6, LEN=20;
   const w=+(2.4+f*2.6).toFixed(1);
   const yTip=y+dir*(boxH/2+gap);
   const yTail=yTip+dir*LEN;
   const headLen=3.6*w;
   const yShaftEnd=yTip+dir*(headLen*0.55);
   s+=`<line x1="${x}" y1="${yTail}" x2="${x}" y2="${yShaftEnd}" stroke="#b8900a" stroke-width="${w}" stroke-linecap="butt"/>`;
   s+=`<path d="M ${x} ${yTip} L ${x-headLen*0.34} ${yTip+dir*headLen} L ${x+headLen*0.34} ${yTip+dir*headLen} Z" fill="#b8900a"/>`;
 });
 for(const n in N){
  const [x,y]=N[n];
  if(n[0]==='L'){
    const f=sq(i['cap_'+n],SC.large_elzr_max);
    const w=Math.max(20,f*46), h=Math.max(15,f*34);
    s+=`<rect x="${x-w/2}" y="${y-h/2}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="6" fill="#161D41"/>`;
    s+=`<text x="${x}" y="${y+4}" font-size="11" font-weight="bold" text-anchor="middle" fill="#fff" font-family="Calibri,sans-serif">${n}</text>`;
  }else{
    const f=sq(i['cap_'+n],SC.small_elzr_max);
    const r=Math.max(6,f*17);
    s+=`<circle cx="${x}" cy="${y}" r="${r.toFixed(1)}" fill="#63A593" stroke="#161D41" stroke-width="1.6"/>`;
    s+=`<text x="${x}" y="${y+3.5}" font-size="9" font-weight="bold" text-anchor="middle" fill="#161D41" font-family="Calibri,sans-serif">${n}</text>`;
  }
 }
 s+='</svg>';
 return s;
}

const LEAFW=216, VGAP=136, TOP=42;
function layout(n,d){
 if(n.type==='leaf'){ n.x=layout.i*LEAFW+LEAFW/2; layout.i++; n.y=TOP+d*VGAP; return n.x; }
 const a=layout(n.left,d+1), b=layout(n.right,d+1);
 n.x=(a+b)/2; n.y=TOP+d*VGAP; return n.x;
}
function render(){
 const stage=document.getElementById('stage'); stage.innerHTML='';
 layout.i=0; layout(TREE,0);
 const totalW=layout.i*LEAFW;
 const maxD=(function d(n){return n.type==='leaf'?0:1+Math.max(d(n.left),d(n.right));})(TREE);
 const totalH=TOP+maxD*VGAP+285;
 stage.style.width=totalW+'px'; stage.style.height=totalH+'px';
 const NS='http://www.w3.org/2000/svg';
 const svg=document.createElementNS(NS,'svg');
 svg.setAttribute('class','edges'); svg.setAttribute('width',totalW); svg.setAttribute('height',totalH);
 (function edges(n){
   if(n.type==='leaf')return;
   const py=n.y+20;
   [n.left,n.right].forEach(c=>{
     const cy=(c.type==='leaf')?c.y:c.y-20;
     const p=document.createElementNS(NS,'path'); const my=(py+cy)/2;
     p.setAttribute('d',`M${n.x},${py} C${n.x},${my} ${c.x},${my} ${c.x},${cy}`);
     p.setAttribute('fill','none'); p.setAttribute('stroke','#b7bcc9'); p.setAttribute('stroke-width','1.7');
     svg.appendChild(p);
   });
   edges(n.left); edges(n.right);
 })(TREE);
 stage.appendChild(svg);
 let uid=0;
 (function place(n){
   if(n.type==='leaf'){
     const i=n.info, c=COLOR[i.label]||'#888';
     let flags='';
     if(i.dual>=0.10) flags+=`<span class="flag dual">dual ${Math.round(i.dual*100)}%</span>`;
     if(i.mesh>=0.10) flags+=`<span class="flag mesh">mesh ${Math.round(i.mesh*100)}%</span>`;
     const impPct=Math.round(i.impfrac*100), locPct=100-impPct;
     const d=document.createElement('div'); d.className='leaf';
     d.style.left=n.x+'px'; d.style.top=n.y+'px'; d.style.borderTop=`4px solid ${c}`;
     d.innerHTML=`<div class="title" style="color:${c}">${i.label}</div>
       <div class="desc">${SUB[i.label]||''}</div>
       ${leafSVG(i,'u'+(uid++))}
       <div class="ssnote">small–small link in ${Math.round(i.S1S2[0]*100)}% of runs</div>
       <div class="magbar"><i style="width:${locPct}%;background:#63A593"></i><i style="width:${impPct}%;background:#b8900a"></i></div>
       <div class="magcap">local ${locPct}%  ·  import ${impPct}%</div>
       <div class="meta"><span>N=${i.n}</span><span>pure ${Math.round(i.purity*100)}%</span></div>
       <div class="stat">price ${i.avg_price} €/MWh · demand ${i.avg_demand} TWh</div>
       <div class="flags">${flags}</div>`;
     stage.appendChild(d); return;
   }
   const [lab,unit]=FEAT[n.feature]||[n.feature,''];
   const box=document.createElement('div'); box.className='split';
   box.style.left=n.x+'px'; box.style.top=n.y+'px';
   box.innerHTML=`<div class="feat">${lab}</div><div class="thr">≤ ${n.thr}${unit?' '+unit:''}</div>`;
   stage.appendChild(box);
   const p=PAIRS[n.feature]||['≤','>'];
   [[n.left,p[0]],[n.right,p[1]]].forEach(([c,txt])=>{
     const cy=(c.type==='leaf')?c.y:c.y-20;
     const l=document.createElement('div'); l.className='blabel';
     l.style.left=((n.x+c.x)/2)+'px'; l.style.top=((n.y+20+cy)/2)+'px'; l.textContent=txt;
     stage.appendChild(l);
   });
   place(n.left); place(n.right);
 })(TREE);
}
render();
</script></body></html>"""


if __name__ == "__main__":
    data = build_data()
    render_html(data)