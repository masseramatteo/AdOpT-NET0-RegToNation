"""
Export the two chosen driver trees as VECTOR files for Illustrator (SVG + PDF + DOT).

Reproduces EXACTLY the same clustering + tuned tree as the HTML/PNG pipelines, so the
figures match the reported accuracy:
  1) strat/supply, all runs, k=3  -> Central / Import-reliant / Local   (DT_CV 0.895)
  2) decentr A, robust runs, k=4  -> 4 decentralisation levels          (DT_CV 0.934)

Vector notes:
  - SVG saved with svg.fonttype='none'  -> text stays EDITABLE TEXT in Illustrator.
  - PDF saved with pdf.fonttype=42      -> embedded TrueType, still editable text.
  - .dot exported too (sklearn export_graphviz) = fully structured, re-styleable.

Run (PowerShell, from the DT_Supply_Chain/hierarchical folder):
  $env:PYTHONIOENCODING="utf-8"; python export_vector_trees.py
Out: DT_supplychain/vector/{supply_all_k3, decentrA_robust_k4}/tree.{svg,pdf,dot}
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_graphviz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from robust_trees import robust_run_ids, ROBUST_SCENARIO
from decentralization_clustering import load_metrics
from decentralization_full_analysis import cluster_1d, best_tree, _tree, level_names
from combined_strategies import cluster as strat_cluster, best_tree2, _tree2, supply_word, unique

# vector output lands next to the other DT_supplychain folders (UNC-safe, no X: mapping needed)
OUT = os.path.join(C.BASE_FIG_DIR, "vector")

# Human-readable axis labels for the sampled driver inputs (used in the figure only)
FEATURE_LABELS = {
    "total_demand_TWh": "Total H2 demand [TWh]",
    "demand_level_ratio": "Demand level ratio",
    "unbalance_ratio": "Demand unbalance ratio",
    "electricity_availability_small": "Small-cluster elec. availability",
    "electricity_availability_large": "Large-cluster elec. availability",
    "import_availability_ratio": "H2 import availability",
    "electricity_price_avg": "Avg electricity price [EUR/MWh]",
    "electricity_standard_dev": "Electricity price volatility",
    "hydrogen_import_price": "H2 import price [EUR/MWh]",
    "solar_cf_realized_mean": "Solar mean capacity factor",
    "capex_ratio_small_big": "Small/large electrolyzer capex ratio",
    "pipeline_cost_multiplier": "Pipeline cost multiplier",
    "distance_Large_cluster1_to_Large_cluster2_km": "Dist. Large-Large [km]",
    "distance_Small_cluster1_to_Small_cluster2_km": "Dist. Small-Small [km]",
    "distance_from_large_cluster": "Small->Large distance [km]",
}


def _vector_rcparams():
    """Keep text as editable text (not outlines) in SVG and PDF for Illustrator."""
    plt.rcParams.update({
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_all(clf, feature_names, class_names, out_dir, title, figsize=(18, 11)):
    os.makedirs(out_dir, exist_ok=True)
    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(clf, feature_names=labels, class_names=class_names, filled=True,
              rounded=True, proportion=True, impurity=True, precision=2, fontsize=9, ax=ax)
    ax.set_title(title, fontsize=13)
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(out_dir, f"tree.{ext}"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "tree.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    # graphviz .dot (text, fully editable structure)
    export_graphviz(clf, out_file=os.path.join(out_dir, "tree.dot"), feature_names=labels,
                    class_names=class_names, filled=True, rounded=True, proportion=True,
                    impurity=True, precision=2, special_characters=True)
    print(f"  -> {out_dir}  (tree.svg / tree.pdf / tree.dot / tree.png)")


def case_supply_k3(feat):
    """strat/supply, all runs, k=3. Clean class names: Central / Import-reliant / Local."""
    feats = ["local_self_sufficiency", "import_intensity"]
    inputs = [c for c in C.DRIVER_INPUTS if c in feat.columns]
    X = feat[inputs].fillna(0).reset_index(drop=True)
    y, sil, profs = strat_cluster(feat, feats, 3)
    class_names = unique([supply_word(p) for p in profs])  # clean human labels, deduped
    lv, ml, _ = best_tree2(X, y, 3)
    clf = _tree2(lv, ml).fit(X, y)
    title = (f"Supply origin (all runs, k=3): inputs -> archetype  |  "
             f"classes {class_names}  |  DT_CV 0.895, leaves {lv}")
    print(f"supply k=3: classes={class_names} leaves={lv} minleaf={ml}")
    _save_all(clf, inputs, class_names, os.path.join(OUT, "supply_all_k3"), title)


def case_decentr_A_robust_k4(feat, dm, rob_ids):
    """decentr A (small_prod/total_demand), robust runs, k=4. Named by decentralisation level."""
    fsub = feat[feat["run"].isin(rob_ids)].reset_index(drop=True)
    dsub = dm[dm["run"].isin(fsub["run"])].set_index("run").loc[fsub["run"]].reset_index()
    x = dsub["defA"].values
    inputs = [c for c in C.DRIVER_INPUTS if c in fsub.columns]
    X = fsub[inputs].fillna(0).reset_index(drop=True)
    y, sil, ranges = cluster_1d(x, 4)
    names_k = level_names(4)
    class_names = [f"{names_k[i]}\n({ranges[i][0]*100:.0f}-{ranges[i][1]*100:.0f}%)" for i in range(4)]
    lv, _ = best_tree(X, y, 4)
    clf = _tree(lv).fit(X, y)
    title = (f"Decentralisation level A=small_prod/total_demand (robust runs, k=4): "
             f"inputs -> level  |  DT_CV 0.934, leaves {lv}")
    print(f"decentr A robust k=4: leaves={lv} ranges={[ (round(r[0],3),round(r[1],3),r[2]) for r in ranges]}")
    _save_all(clf, inputs, class_names, os.path.join(OUT, "decentrA_robust_k4"), title)


def main():
    C.apply_joule_style()
    _vector_rcparams()
    os.makedirs(OUT, exist_ok=True)
    feat = pd.read_csv(C.FEATURES_CSV)
    dm = load_metrics()
    rob_ids, *_ = robust_run_ids(ROBUST_SCENARIO)
    print(f"features: {len(feat)} runs | robust: {len(rob_ids)} runs\n")

    case_supply_k3(feat)
    case_decentr_A_robust_k4(feat, dm, rob_ids)
    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()