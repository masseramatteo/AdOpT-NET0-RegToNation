"""
One-shot orchestrator: build features (if needed) -> cluster (k=N_CLUSTERS) ->
explain tree -> driver tree -> driver-tree HTML. All outputs for a given k go to
DT_supplychain/k{N}/ .

Choose k via the DT_K environment variable, e.g. in PowerShell:
    $env:DT_K=9; python run_pipeline.py
    $env:DT_K=4; python run_pipeline.py
"""

import os

import config as C
import build_features
import cluster_infrastructure as ci
import explain_tree as et
import driver_tree as dt
import render_driver_tree_html as rh


def main():
    print("=" * 70)
    print(f"PIPELINE  k={C.N_CLUSTERS}   ->  {C.FIG_DIR}")
    print("=" * 70)

    if not os.path.isfile(C.FEATURES_CSV):
        build_features.build()

    ci.fit(C.N_CLUSTERS)     # writes cluster_labels_k{N}.csv
    et.run()                 # explain tree + radar + cluster_final
    dt.run()                 # driver tree png + importances
    rh.render_html(rh.build_data())   # driver_tree_magnitude.html

    print("\n" + "=" * 70)
    print(f"DONE k={C.N_CLUSTERS}. Figures in {C.FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()