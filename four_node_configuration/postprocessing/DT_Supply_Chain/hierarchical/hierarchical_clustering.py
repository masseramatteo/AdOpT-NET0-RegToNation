"""
Hierarchical / progressive clustering of the infrastructure typology.

Three views of the same idea — build the typology in LEVELS instead of one flat
k-Means:

  1. MANUAL LEVEL-WISE (the requested idea): cluster on 'supply origin' features
     first (level 1: local self-sufficiency / import / transport), then WITHIN each
     branch cluster on 'transport structure' (level 2: #pipelines, L-L, S-S, meshed).
     Optional level 3 on tech detail. Produces nested, human-named labels and a
     sunburst.

  2. AGGLOMERATIVE (Ward) dendrogram — bottom-up hierarchy; cut at any k. The classic
     "levels" view; shows how clusters merge.

  3. BISECTING k-MEANS — progressive/divisive k-means (repeatedly split the largest
     cluster). This is the "progressive k-means" analogue.

Outputs go to DT_supplychain/hierarchical/ on the network drive.

Run:  python hierarchical_clustering.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, BisectingKMeans
from scipy.cluster.hierarchy import dendrogram, linkage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

OUT_DIR = os.path.join(C.BASE_FIG_DIR, "hierarchical")

# feature levels (coarse -> fine)
LEVEL1 = ["local_self_sufficiency", "import_intensity", "transport_intensity"]   # supply origin
LEVEL2 = ["n_pipelines", "LL_pipe", "SS_pipe", "meshed"]                          # transport structure
LEVEL3 = ["highP_share", "storage_large_share"]                                  # tech detail

K1 = 3      # supply-origin groups
K2 = 2      # structure sub-groups within each origin group


def _z(df, cols):
    return StandardScaler().fit_transform(df[cols].fillna(0.0))


def _km(X, k):
    return KMeans(n_clusters=k, random_state=C.RANDOM_STATE, n_init=10).fit_predict(X)


def name_l1(p):
    """Name a supply-origin group from its level-1 means."""
    if p["import_intensity"] > 0.25:
        return "Import-sourced"
    if p["local_self_sufficiency"] > 0.60:
        return "Locally-sourced"
    return "Centrally-sourced"


def name_l2(p):
    """Name a transport-structure sub-group from its level-2 means."""
    if p["SS_pipe"] > 0.5 or p["meshed"] > 0.5:
        return "Meshed"
    if p["LL_pipe"] > 0.5:
        return "Trunk"
    if p["n_pipelines"] < 2.2:
        return "Sparse"
    return "Radial"


def manual_hierarchy(df):
    print("\n" + "=" * 74)
    print("1) MANUAL LEVEL-WISE HIERARCHY")
    print("=" * 74)
    # level 1
    df["L1"] = _km(_z(df, LEVEL1), K1)
    l1_names = {}
    for g in sorted(df["L1"].unique()):
        l1_names[g] = name_l1(df.loc[df["L1"] == g, LEVEL1].mean())
    # level 2 within each L1
    df["L2"] = -1
    for g in sorted(df["L1"].unique()):
        m = df["L1"] == g
        df.loc[m, "L2"] = _km(_z(df[m], LEVEL2), K2)
    # names + report
    rows = []
    for g in sorted(df["L1"].unique()):
        for h in sorted(df.loc[df["L1"] == g, "L2"].unique()):
            m = (df["L1"] == g) & (df["L2"] == h)
            l2n = name_l2(df.loc[m, LEVEL2].mean())
            rows.append((g, h, l1_names[g], l2n, int(m.sum())))
    hier = pd.DataFrame(rows, columns=["L1", "L2", "origin", "structure", "n"])
    df["hier_label"] = df.apply(lambda r: f"{l1_names[r['L1']]} > "
                                f"{name_l2(df.loc[(df['L1']==r['L1'])&(df['L2']==r['L2']), LEVEL2].mean())}",
                                axis=1)
    print(hier.to_string(index=False))
    print(f"\nTotal final groups: {len(hier)}")

    _sunburst(hier, l1_names)
    df[["run", "L1", "L2", "hier_label"]].to_csv(
        os.path.join(OUT_DIR, "hierarchical_labels.csv"), index=False)
    return df, hier, l1_names


def _sunburst(hier, l1_names):
    """Two-ring sunburst: inner = supply origin (L1), outer = structure (L2)."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))
    base = {g: C.CLUSTER_PALETTE[i % len(C.CLUSTER_PALETTE)]
            for i, g in enumerate(sorted(hier["L1"].unique()))}
    # inner ring
    l1 = hier.groupby("L1")["n"].sum().reindex(sorted(hier["L1"].unique()))
    ax.pie(l1.values, radius=0.7, labels=[l1_names[g] for g in l1.index],
           labeldistance=0.35, colors=[base[g] for g in l1.index],
           wedgeprops=dict(width=0.7, edgecolor="w"), textprops=dict(fontsize=9, color="w", weight="bold"))
    # outer ring
    outer = hier.sort_values(["L1", "L2"])
    ax.pie(outer["n"].values, radius=1.0,
           labels=[f"{s}\n{n}" for s, n in zip(outer["structure"], outer["n"])],
           labeldistance=0.82, colors=[_shade(base[g], 0.45) for g in outer["L1"]],
           wedgeprops=dict(width=0.3, edgecolor="w"), textprops=dict(fontsize=7.5))
    ax.set_title("Hierarchical infrastructure typology\ninner = supply origin · outer = transport structure",
                 fontsize=12)
    fig.savefig(os.path.join(OUT_DIR, "sunburst_hierarchy.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _shade(hex_color, amt):
    """Lighten a hex color by amt (0..1)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amt); g = int(g + (255 - g) * amt); b = int(b + (255 - b) * amt)
    return f"#{r:02x}{g:02x}{b:02x}"


def ward_dendrogram(df):
    print("\n" + "=" * 74)
    print("2) AGGLOMERATIVE (Ward) DENDROGRAM")
    print("=" * 74)
    X = _z(df, C.OUTPUTS_OF_INTEREST)
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, truncate_mode="lastp", p=30, no_labels=True,
               color_threshold=0.7 * max(Z[:, 2]), ax=ax)
    ax.set_title("Ward dendrogram over outputs of interest (progressive merging)")
    ax.set_ylabel("merge distance")
    fig.savefig(os.path.join(OUT_DIR, "ward_dendrogram.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    # report cluster sizes at cuts k=2..6
    for k in range(2, 7):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        sizes = pd.Series(lab).value_counts().sort_index().to_dict()
        print(f"  cut k={k}: sizes {sizes}")


def bisecting(df):
    print("\n" + "=" * 74)
    print("3) BISECTING k-MEANS (progressive / divisive)")
    print("=" * 74)
    X = _z(df, C.OUTPUTS_OF_INTEREST)
    for k in [4, 6]:
        bk = BisectingKMeans(n_clusters=k, random_state=C.RANDOM_STATE,
                             bisecting_strategy="largest_cluster").fit(X)
        sizes = pd.Series(bk.labels_).value_counts().sort_index().to_dict()
        print(f"  BisectingKMeans k={k}: sizes {sizes}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(C.FEATURES_CSV)
    print(f"Loaded {len(df)} runs; outputs -> {OUT_DIR}")
    df, hier, l1_names = manual_hierarchy(df)
    ward_dendrogram(df)
    bisecting(df)
    print(f"\nDone. Figures in {OUT_DIR}")


if __name__ == "__main__":
    main()