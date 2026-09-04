"""
Cluster-number justification (elbow + silhouette vs k) for the strat and sysmix folders.
For each strategy (all-data) plots within-cluster inertia (elbow) and silhouette across
k, marking the silhouette-optimal k. Saves <folder>/cluster_number_choice.png.

Run:  python mk_elbow.py
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from combined_strategies import STRATEGIES as STRAT_DEFS
from system_supply_analysis import STRATEGIES as SYS_DEFS, add_shares

BASE = r"X:\3000_simulations_24_may_2026\Snellius_3000_simulations\DT_supplychain"


def elbow_fig(folder, defs, df):
    fig, axes = plt.subplots(len(defs), 2, figsize=(11, 3.1 * len(defs)))
    axes = np.atleast_2d(axes)
    for i, (name, feats, _c, _x) in enumerate(defs):
        X = StandardScaler().fit_transform(df[feats].fillna(0))
        nun = df[feats].round(3).drop_duplicates().shape[0]
        ks = [k for k in range(2, 11) if k <= nun]
        inert, sil = [], []
        for k in ks:
            km = KMeans(k, random_state=C.RANDOM_STATE, n_init=10).fit(X)
            inert.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
        kbest = ks[int(np.argmax(sil))]
        a1, a2 = axes[i, 0], axes[i, 1]
        a1.plot(ks, inert, "o-", color=C.UU_NAVY); a1.set_title(f"{name} — elbow (inertia)", fontsize=10)
        a1.set_xlabel("k"); a1.grid(alpha=.3)
        a2.plot(ks, sil, "o-", color=C.UU_TEAL); a2.axvline(kbest, color=C.UU_RED, ls="--")
        a2.set_title(f"{name} — silhouette (k*={kbest})", fontsize=10); a2.set_xlabel("k"); a2.grid(alpha=.3)
    fig.suptitle(f"{folder}: cluster-number choice (all runs)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(BASE, folder, "cluster_number_choice.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def main():
    feat = pd.read_csv(C.FEATURES_CSV)
    raw = pd.read_excel(C.EXTRACTED_RESULTS)
    elbow_fig("strat", STRAT_DEFS, feat)
    elbow_fig("sysmix", SYS_DEFS, add_shares(feat, raw))


if __name__ == "__main__":
    main()
