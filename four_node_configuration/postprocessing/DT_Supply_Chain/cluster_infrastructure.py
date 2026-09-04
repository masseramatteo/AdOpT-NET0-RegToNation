"""
Step 2 - Cluster scenarios on the outputs of interest (Baader-faithful).

Two entry points:
  diagnostics()  -> correlation matrix, elbow (inertia) and silhouette vs k.
                    Run this FIRST, inspect, then choose k.
  fit(k)         -> final k-Means with chosen k; saves cluster_labels.csv and
                    prints per-cluster profiles.

All features are standardized (z-score) before k-Means, per the locked decision
(standardize all, incl. binary topology features).

Run:  python cluster_infrastructure.py            # diagnostics
      python cluster_infrastructure.py --fit K     # final fit with k=K
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config as C


def _load_X():
    df = pd.read_csv(C.FEATURES_CSV)
    X = df[C.OUTPUTS_OF_INTEREST].copy()
    # clip the rare local_share_small > 1 tail? No - keep, it is meaningful (net export).
    X = X.fillna(0.0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return df, X, Xs, scaler


def diagnostics(k_range=range(2, 11)):
    os.makedirs(C.FIG_DIR, exist_ok=True)
    df, X, Xs, _ = _load_X()
    print(f"Clustering matrix: {Xs.shape[0]} runs x {Xs.shape[1]} features")
    print(f"Features: {C.OUTPUTS_OF_INTEREST}")

    # --- correlation among outputs of interest ------------------------------
    corr = X.corr()
    print("\nCorrelation matrix of outputs of interest:")
    with pd.option_context("display.float_format", lambda v: f"{v:6.2f}",
                           "display.width", 200):
        print(corr)
    _plot_corr(corr)

    # --- elbow + silhouette --------------------------------------------------
    inertias, sils = [], []
    ks = list(k_range)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=C.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, labels))
        print(f"  k={k}: inertia={km.inertia_:10.1f}  silhouette={sils[-1]:.4f}")

    _plot_elbow_sil(ks, inertias, sils)
    print(f"\nDiagnostic figures saved to {C.FIG_DIR}")
    print("Inspect them, then run:  python cluster_infrastructure.py --fit K")


def fit(k):
    os.makedirs(C.FIG_DIR, exist_ok=True)
    df, X, Xs, _ = _load_X()
    km = KMeans(n_clusters=k, random_state=C.RANDOM_STATE, n_init=10)
    labels = km.fit_predict(Xs)
    df["cluster_kmeans"] = labels

    print(f"k-Means with k={k}  (silhouette={silhouette_score(Xs, labels):.4f})")
    print("\nCluster sizes:")
    print(df["cluster_kmeans"].value_counts().sort_index().to_dict())

    # per-cluster profile (mean of each output of interest, original units)
    print("\nCluster profiles (mean of each output of interest):")
    prof = df.groupby("cluster_kmeans")[C.OUTPUTS_OF_INTEREST].mean()
    with pd.option_context("display.float_format", lambda v: f"{v:7.3f}",
                           "display.width", 200):
        print(prof.T)

    # archetype composition of each cluster
    print("\nArchetype composition per cluster (row % ):")
    comp = pd.crosstab(df["cluster_kmeans"], df["archetype"], normalize="index") * 100
    with pd.option_context("display.float_format", lambda v: f"{v:5.1f}"):
        print(comp)

    keep = ["run", "archetype", "cluster_kmeans"] + C.OUTPUTS_OF_INTEREST + C.DRIVER_INPUTS
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(C.CLUSTER_LABELS_CSV, index=False)
    print(f"\nSaved labels -> {C.CLUSTER_LABELS_CSV}")
    return df


def _plot_corr(corr):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation of outputs of interest")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "diag_correlation.png"))
    plt.close(fig)


def _plot_elbow_sil(ks, inertias, sils):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ks, inertias, "o-", color=C.UU_NAVY)
    ax1.set_xlabel("k"); ax1.set_ylabel("inertia (within-cluster SSE)")
    ax1.set_title("Elbow method"); ax1.grid(alpha=0.3)
    ax2.plot(ks, sils, "o-", color=C.UU_TEAL)
    ax2.set_xlabel("k"); ax2.set_ylabel("silhouette score")
    ax2.set_title("Silhouette"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "diag_elbow_silhouette.png"))
    plt.close(fig)


if __name__ == "__main__":
    C.apply_joule_style()
    if len(sys.argv) >= 3 and sys.argv[1] == "--fit":
        fit(int(sys.argv[2]))
    else:
        diagnostics()