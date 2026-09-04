"""
Explore different LEVEL definitions for the hierarchical infrastructure typology.

The earlier scheme put 'supply origin' (import vs central) at level 1 — but both make
H2 at the large clusters, so the physical infrastructure is similar. The axis that
actually changes the infrastructure is DECENTRALISATION (where electrolysis sits),
then the pipeline layout. This script tries several level schemes and, for each,
saves a paper-ready single decision tree (CV accuracy in the title), the staged
per-level trees, a sunburst, and a summary.

Each scheme = ordered list of levels; each level = (name, feature_list, k, namer).
Nested k-Means defines the ground-truth groups; a DecisionTree then predicts them
from the sampled inputs.

Outputs: DT_supplychain/hierarchical/schemes/<scheme>/

Run:  python explore_level_schemes.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

SCHEME_DIR = os.path.join(C.BASE_FIG_DIR, "hierarchical", "schemes")
LEAVES = 14            # readable single-tree leaf budget
MIN_LEAF, MIN_SPLIT = 20, 20
STAGED_LEAVES, STAGED_MINLEAF = 12, 10   # per-stage budget: cleaner figure (~0.61), still near ceiling


# ---------- level namers -----------------------------------------------------
def name_decentr(p):
    s = p["local_self_sufficiency"]
    return "No local prod." if s < 0.05 else ("Low local" if s < 0.50 else "High local")


def decentr_bins(df):
    """Explicit decentralisation levels by fixed threshold on local self-sufficiency:
    0 = None (~0), 1 = Low (<0.5), 2 = High (>=0.5). Guarantees a 'No local prod.' level."""
    s = df["local_self_sufficiency"].fillna(0.0).values
    return np.where(s < 0.05, 0, np.where(s < 0.50, 1, 2))


def name_struct(p):
    if p["SS_pipe"] > 0.5 or p["meshed"] > 0.5:
        return "Meshed"
    if p["LL_pipe"] > 0.5:
        return "Trunk"
    if p["n_pipelines"] < 2.2:
        return "Sparse"
    return "Radial"


def name_transport(p):
    if p["highP_share"] > 0.30:
        return "High-pressure"
    if p["transport_intensity"] > 0.22:
        return "Heavy transport"
    if p["transport_intensity"] < 0.10:
        return "Minimal transport"
    return "Low-pressure"


def name_storage(p):
    return "Large storage" if p["storage_large_share"] > 0.5 else "Small storage"


def name_struct3(p):
    """Three explicit transport structures."""
    if p["SS_pipe"] > 0.5 or p["meshed"] > 0.5:
        return "Meshed"
    if p["n_pipelines"] < 1.5:
        return "Minimal transport"
    return "Satellite"


def struct_bins3(df):
    """Rule-based structure: 0 = Minimal (almost no pipelines), 2 = Meshed (small-small
    or dual link), 1 = Satellite (radial: smalls attach to larges, no mesh)."""
    meshed = (df["SS_pipe"].fillna(0) > 0.5) | (df["meshed"].fillna(0) > 0.5)
    minimal = df["n_pipelines"].fillna(0) < 1.5
    return np.where(meshed, 2, np.where(minimal, 0, 1))


def decentr_bins2(df):
    """Two-level decentralisation for use as a within-branch (level-2) split:
    0 = Low local (<0.5), 1 = High local (>=0.5). No k-Means -> no name collisions."""
    s = df["local_self_sufficiency"].fillna(0.0).values
    return np.where(s < 0.50, 0, 1)


# ---------- schemes ----------------------------------------------------------
# ALL decentralisation levels use fixed threshold bins (never k-Means) so the level
# names can never collide (fixes the duplicate "Low local" issue). k-Means is only
# used where the level is genuinely multi-feature (structure / transport / storage).
SCHEMES = {
    # WINNER: how much is made locally (None/Low/High) -> how it's transported
    "E_decentr_then_transport": [
        ("Decentralisation", ["local_self_sufficiency"], decentr_bins, name_decentr),
        ("Transport", ["transport_intensity", "highP_share", "LL_pipe"], 2, name_transport),
    ],
    # decentralisation -> pipeline structure
    "F_decentr_then_structure": [
        ("Decentralisation", ["local_self_sufficiency"], decentr_bins, name_decentr),
        ("Structure", ["n_pipelines", "LL_pipe", "SS_pipe", "meshed"], 2, name_struct),
    ],
    # decentralisation -> storage siting
    "G_decentr_then_storage": [
        ("Decentralisation", ["local_self_sufficiency"], decentr_bins, name_decentr),
        ("Storage", ["storage_large_share", "transport_intensity"], 2, name_storage),
    ],
    # opposite ordering: pipeline structure first, then decentralisation (bins, 2-way)
    "C_structure_then_decentr": [
        ("Structure", ["n_pipelines", "LL_pipe", "SS_pipe", "meshed"], 3, name_struct),
        ("Decentralisation", ["local_self_sufficiency"], decentr_bins2, name_decentr),
    ],
    # decentralisation -> explicit 3-way structure (Minimal / Satellite / Meshed) = 9 groups
    "H_decentr_then_struct3": [
        ("Decentralisation", ["local_self_sufficiency"], decentr_bins, name_decentr),
        ("Structure", ["n_pipelines", "SS_pipe", "meshed", "LL_pipe"], struct_bins3, name_struct3),
    ],
}


def _z(df, cols):
    return StandardScaler().fit_transform(df[cols].fillna(0.0))


def _tree(leaves=LEAVES):
    return DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=MIN_LEAF,
                                  min_samples_split=MIN_SPLIT, class_weight="balanced",
                                  random_state=C.RANDOM_STATE)


def build_labels(df, scheme):
    """Nested k-Means for a 2-level scheme; returns combined code, names, and the
    per-level label columns."""
    (n1, f1, k1, nm1), (n2, f2, k2, nm2) = scheme
    # level 1: k-Means (int k) OR an explicit binner (callable) for fixed levels
    if callable(k1):
        df["Lv1"] = k1(df)
        k1 = int(df["Lv1"].nunique())
    else:
        df["Lv1"] = KMeans(k1, random_state=C.RANDOM_STATE, n_init=10).fit_predict(_z(df, f1))
    lv1_names = {g: nm1(df.loc[df["Lv1"] == g, C.OUTPUTS_OF_INTEREST].mean()) for g in range(k1)}
    df["Lv2"] = -1
    for g in range(k1):
        m = df["Lv1"] == g
        if callable(k2):
            df.loc[m, "Lv2"] = k2(df[m])
        else:
            df.loc[m, "Lv2"] = KMeans(k2, random_state=C.RANDOM_STATE, n_init=10).fit_predict(_z(df[m], f2))
    k2 = int(df["Lv2"].max()) + 1
    code, names = {}, {}
    c = 0
    for g in range(k1):
        for h in range(k2):
            mm = (df["Lv1"] == g) & (df["Lv2"] == h)
            if mm.sum() == 0:            # skip empty combos (avoids phantom groups)
                continue
            code[(g, h)] = c
            names[c] = f"{lv1_names[g]} > {nm2(df.loc[mm, C.OUTPUTS_OF_INTEREST].mean())}"
            c += 1
    df["grp"] = df.apply(lambda r: code[(r["Lv1"], r["Lv2"])], axis=1).astype(int)
    return df, code, names, lv1_names, (k1, k2)


def staged_e2e_cv(X, L1, L2, k1):
    combined = L1 * 10 + L2
    skf = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    e2e = []
    for tr, te in skf.split(X, combined):
        t1 = _tree_staged().fit(X.iloc[tr], L1[tr])
        l1p = t1.predict(X.iloc[te])
        l2p = np.zeros(len(te), dtype=int)
        for g in range(k1):
            mg = L1[tr] == g
            if len(np.unique(L2[tr][mg])) > 1:
                tg = _tree_staged().fit(X.iloc[tr][mg], L2[tr][mg])
                sel = l1p == g
                if sel.any():
                    l2p[sel] = tg.predict(X.iloc[te][sel])
        e2e.append(((l1p == L1[te]) & (l2p == L2[te])).mean())
    return np.mean(e2e)


def run_scheme(df0, sid, scheme):
    out = os.path.join(SCHEME_DIR, sid)
    os.makedirs(out, exist_ok=True)
    df = df0.copy()
    df, code, names, lv1_names, (k1, k2) = build_labels(df, scheme)
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    X = df[inputs].fillna(0.0)
    n_groups = len(names)
    class_names = [names[i] for i in range(n_groups)]

    # single tree (paper figure)
    clf = _tree().fit(X, df["grp"])
    cv = cross_val_score(clf, X, df["grp"], cv=5)
    e2e = staged_e2e_cv(X, df["Lv1"].values, df["Lv2"].values, k1)

    lines = [f"SCHEME {sid}",
             f"  level 1: {scheme[0][0]} on {scheme[0][1]} (k={k1})",
             f"  level 2: {scheme[1][0]} on {scheme[1][1]} (k={k2})",
             f"  groups ({n_groups}):"]
    for i in range(n_groups):
        lines.append(f"    {i}: {class_names[i]}  (n={(df['grp']==i).sum()})")
    lines.append(f"  single-tree CV acc = {cv.mean():.3f} +/- {cv.std():.3f}")
    lines.append(f"  staged end-to-end CV acc = {e2e:.3f}")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # paper-ready single tree
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(clf, feature_names=inputs, class_names=class_names, filled=True,
              rounded=True, proportion=True, impurity=False, precision=1, fontsize=9, ax=ax)
    ax.set_title(f"{sid} — inputs → {scheme[0][0]} > {scheme[1][0]}   "
                 f"(CV {cv.mean():.2f}, staged {e2e:.2f})", fontsize=15)
    fig.savefig(os.path.join(out, "decision_tree.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # sunburst + staged trees (L1 + per-branch L2) — the accurate/interpretable view
    _sunburst(df, lv1_names, scheme, k1, k2, out)
    _staged_fig(df, scheme, inputs, X, lv1_names, k1, e2e, out)
    df[["run", "Lv1", "Lv2", "grp"]].to_csv(os.path.join(out, "labels.csv"), index=False)
    return sid, cv.mean(), e2e, len(names)


def _tree_staged():
    return DecisionTreeClassifier(max_leaf_nodes=STAGED_LEAVES, min_samples_leaf=STAGED_MINLEAF,
                                  min_samples_split=STAGED_MINLEAF, class_weight="balanced",
                                  random_state=C.RANDOM_STATE)


def _staged_fig(df, scheme, inputs, X, lv1_names, k1, e2e, out):
    """Level-1 tree + one Level-2 tree per branch, stacked in one figure."""
    nm2 = scheme[1][3]
    ooi = C.OUTPUTS_OF_INTEREST
    branches = [g for g in range(k1) if df.loc[df["Lv1"] == g, "Lv2"].nunique() > 1]
    nrows = 1 + len(branches)
    fig, axes = plt.subplots(nrows, 1, figsize=(22, 6.5 * nrows))
    axes = np.atleast_1d(axes)

    t1 = _tree_staged().fit(X, df["Lv1"])
    cv1 = cross_val_score(_tree_staged(), X, df["Lv1"], cv=5).mean()
    plot_tree(t1, feature_names=inputs, class_names=[lv1_names[g] for g in range(k1)],
              filled=True, rounded=True, proportion=True, impurity=False,
              precision=1, fontsize=8, ax=axes[0])
    axes[0].set_title(f"Level 1 — {scheme[0][0]} (CV {cv1:.2f})", fontsize=13)

    for i, g in enumerate(branches, 1):
        m = df["Lv1"] == g
        present = sorted(df.loc[m, "Lv2"].unique())
        tg = _tree_staged().fit(X[m], df.loc[m, "Lv2"])
        cvg = cross_val_score(_tree_staged(), X[m], df.loc[m, "Lv2"], cv=5).mean()
        cls = [nm2(df.loc[m & (df["Lv2"] == h), ooi].mean()) for h in present]
        plot_tree(tg, feature_names=inputs, class_names=cls, filled=True, rounded=True,
                  proportion=True, impurity=False, precision=1, fontsize=8, ax=axes[i])
        axes[i].set_title(f"Level 2 — within '{lv1_names[g]}' → {scheme[1][0]} "
                          f"(CV {cvg:.2f}, n={m.sum()})", fontsize=12)
    fig.suptitle(f"{scheme[0][0]} → {scheme[1][0]}  |  staged end-to-end CV {e2e:.2f}",
                 fontsize=15, y=1.002)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "staged_trees.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _sunburst(df, lv1_names, scheme, k1, k2, out):
    (n1, f1, _, _), (n2, f2, _, nm2) = scheme
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))
    base = {g: C.CLUSTER_PALETTE[g % len(C.CLUSTER_PALETTE)] for g in range(k1)}
    inner = df.groupby("Lv1").size().reindex(range(k1))
    ax.pie(inner.values, radius=0.7, labels=[lv1_names[g] for g in range(k1)],
           labeldistance=0.3, colors=[base[g] for g in range(k1)],
           wedgeprops=dict(width=0.7, edgecolor="w"),
           textprops=dict(fontsize=8, color="w", weight="bold"))
    outer_sizes, outer_lbl, outer_col = [], [], []
    for g in range(k1):
        for h in range(k2):
            mm = (df["Lv1"] == g) & (df["Lv2"] == h)
            if mm.sum() == 0:            # skip empty combos
                continue
            outer_sizes.append(mm.sum())
            outer_lbl.append(f"{nm2(df.loc[mm, C.OUTPUTS_OF_INTEREST].mean())}\n{mm.sum()}")
            outer_col.append(_shade(base[g], 0.4))
    ax.pie(outer_sizes, radius=1.0, labels=outer_lbl, labeldistance=0.8,
           colors=outer_col, wedgeprops=dict(width=0.3, edgecolor="w"),
           textprops=dict(fontsize=7))
    ax.set_title(f"{scheme[0][0]} (inner) → {scheme[1][0]} (outer)", fontsize=12)
    fig.savefig(os.path.join(out, "sunburst.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _shade(hex_color, amt):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amt); g = int(g + (255 - g) * amt); b = int(b + (255 - b) * amt)
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    C.apply_joule_style()
    os.makedirs(SCHEME_DIR, exist_ok=True)
    df0 = pd.read_csv(C.FEATURES_CSV)
    results = []
    for sid, scheme in SCHEMES.items():
        results.append(run_scheme(df0, sid, scheme))

    print("\n" + "=" * 70)
    print("SCHEME COMPARISON (single-tree CV / staged CV / #groups)")
    print("=" * 70)
    for sid, cv, e2e, g in sorted(results, key=lambda t: -t[2]):
        print(f"  {sid:<28} single {cv:.3f}   staged {e2e:.3f}   groups {g}")
    with open(os.path.join(SCHEME_DIR, "COMPARISON.txt"), "w", encoding="utf-8") as f:
        for sid, cv, e2e, g in sorted(results, key=lambda t: -t[2]):
            f.write(f"{sid}: single {cv:.3f}  staged {e2e:.3f}  groups {g}\n")
    print(f"\nAll schemes saved under {SCHEME_DIR}")


if __name__ == "__main__":
    main()