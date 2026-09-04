"""
Three-level ASYMMETRIC hierarchy (user's idea):

  Level 1  — is there local electrolysis in the small clusters?   (No / Yes)
  Level 2  — (only if Yes) degree of decentralisation             (Low / High)
  Level 3  — transport structure                                  (Minimal / Satellite / Meshed)

The 'No local' branch skips level 2 (decentralisation degree is undefined there) and
goes straight to structure. Final groups are the same typology as scheme H, but the
DRIVER prediction is staged into three interpretable decisions — the first (build
local electrolysis or not) is the most fundamental and the most predictable.

Reports per-stage CV accuracy + end-to-end, saves the staged trees, a 3-ring
sunburst and labels.

Outputs: DT_supplychain/hierarchical/three_level/

Run:  python three_level_scheme.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from explore_level_schemes import struct_bins3, name_struct3, name_decentr

OUT = os.path.join(C.BASE_FIG_DIR, "hierarchical", "three_level")
LEAVES, MIN_LEAF = 12, 10
DECENTR_NAME = {0: "No local prod.", 1: "Low local", 2: "High local"}


def _tree():
    return DecisionTreeClassifier(max_leaf_nodes=LEAVES, min_samples_leaf=MIN_LEAF,
                                  min_samples_split=MIN_LEAF, class_weight="balanced",
                                  random_state=C.RANDOM_STATE)


def build(df):
    s = df["local_self_sufficiency"].fillna(0.0).values
    df["decentr"] = np.where(s < 0.05, 0, np.where(s < 0.50, 1, 2))   # None/Low/High
    df["has_local"] = (df["decentr"] > 0).astype(int)                  # L1
    df["hi_local"] = (df["decentr"] == 2).astype(int)                  # L2 (within has_local)
    df["struct"] = struct_bins3(df)                                    # L3
    return df


def stage_accuracies(df, X):
    L1 = df["has_local"].values
    dec = df["decentr"].values
    st = df["struct"].values

    a1 = cross_val_score(_tree(), X, L1, cv=5).mean()
    m = df["has_local"] == 1
    a2 = cross_val_score(_tree(), X[m], df.loc[m, "hi_local"], cv=5).mean()
    # structure per decentralisation level (weighted)
    a3s, ws = [], []
    for d in [0, 1, 2]:
        md = df["decentr"] == d
        if df.loc[md, "struct"].nunique() > 1:
            a3s.append(cross_val_score(_tree(), X[md], df.loc[md, "struct"], cv=5).mean())
            ws.append(md.sum())
    a3 = float(np.average(a3s, weights=ws))

    # end-to-end
    combined = dec * 10 + st
    skf = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    e2e = []
    for tr, te in skf.split(X, combined):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        t1 = _tree().fit(Xtr, L1[tr])
        mtr = L1[tr] == 1
        t2 = _tree().fit(Xtr[mtr], df["hi_local"].values[tr][mtr])
        stree = {}
        for d in [0, 1, 2]:
            md = dec[tr] == d
            if len(np.unique(st[tr][md])) > 1:
                stree[d] = _tree().fit(Xtr[md], st[tr][md])
            else:
                stree[d] = ("const", int(st[tr][md][0]) if md.any() else 0)
        # predict
        hl = t1.predict(Xte)
        dpred = np.where(hl == 0, 0, np.where(t2.predict(Xte) == 1, 2, 1))
        spred = np.zeros(len(te), dtype=int)
        for i, d in enumerate(dpred):
            node = stree[d]
            spred[i] = node.predict(Xte.iloc[[i]])[0] if not isinstance(node, tuple) else node[1]
        e2e.append(((dpred == dec[te]) & (spred == st[te])).mean())
    return a1, a2, a3, float(np.mean(e2e))


def figures(df, X, a1, a2, a3, e2e):
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    # staged trees: L1 (binary), L2 (Low/High), L3 per decentralisation level
    panels = [("Level 1 — local electrolysis?", df["has_local"], ["No local", "Local"], df.index)]
    m = df["has_local"] == 1
    panels.append(("Level 2 — decentralisation degree (within Local)", df.loc[m, "hi_local"],
                   ["Low local", "High local"], df[m].index))
    for d in [0, 1, 2]:
        md = df["decentr"] == d
        if df.loc[md, "struct"].nunique() > 1:
            present = sorted(df.loc[md, "struct"].unique())
            cls = [name_struct3(df.loc[md & (df["struct"] == h), C.OUTPUTS_OF_INTEREST].mean())
                   for h in present]
            panels.append((f"Level 3 — structure within '{DECENTR_NAME[d]}'",
                           df.loc[md, "struct"], cls, df[md].index))

    fig, axes = plt.subplots(len(panels), 1, figsize=(20, 6 * len(panels)))
    axes = np.atleast_1d(axes)
    for ax, (title, y, cls, idx) in zip(axes, panels):
        t = _tree().fit(X.loc[idx], y)
        plot_tree(t, feature_names=inputs, class_names=cls, filled=True, rounded=True,
                  proportion=True, impurity=False, precision=1, fontsize=8, ax=ax)
        ax.set_title(title, fontsize=13)
    fig.suptitle(f"Three-level driver tree | L1 {a1:.2f} · L2 {a2:.2f} · L3 {a3:.2f} · "
                 f"end-to-end {e2e:.2f}", fontsize=15, y=1.001)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "three_level_trees.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def sunburst(df):
    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(aspect="equal"))
    base = {0: C.UU_NAVY, 1: C.UU_TEAL, 2: C.UU_YELLOW}
    inner = df.groupby("decentr").size().reindex([0, 1, 2]).fillna(0)
    ax.pie(inner.values, radius=0.7, labels=[DECENTR_NAME[d] for d in [0, 1, 2]],
           labeldistance=0.3, colors=[base[d] for d in [0, 1, 2]],
           wedgeprops=dict(width=0.7, edgecolor="w"),
           textprops=dict(fontsize=8, color="w", weight="bold"))
    sizes, lbl, col = [], [], []
    for d in [0, 1, 2]:
        for h in sorted(df.loc[df["decentr"] == d, "struct"].unique()):
            mm = (df["decentr"] == d) & (df["struct"] == h)
            if mm.sum() == 0:
                continue
            sizes.append(mm.sum())
            lbl.append(f"{name_struct3(df.loc[mm, C.OUTPUTS_OF_INTEREST].mean())}\n{mm.sum()}")
            col.append(_shade(base[d], 0.4))
    ax.pie(sizes, radius=1.0, labels=lbl, labeldistance=0.8, colors=col,
           wedgeprops=dict(width=0.3, edgecolor="w"), textprops=dict(fontsize=7))
    ax.set_title("Decentralisation (inner) → structure (outer)", fontsize=12)
    fig.savefig(os.path.join(OUT, "sunburst.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _shade(hex_color, amt):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amt); g = int(g + (255 - g) * amt); b = int(b + (255 - b) * amt)
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    C.apply_joule_style()
    os.makedirs(OUT, exist_ok=True)
    df = build(pd.read_csv(C.FEATURES_CSV))
    inputs = [c for c in C.DRIVER_INPUTS if c in df.columns]
    X = df[inputs].fillna(0.0)

    a1, a2, a3, e2e = stage_accuracies(df, X)
    print("THREE-LEVEL ASYMMETRIC SCHEME")
    print(f"  L1 local electrolysis? (No/Yes) : CV {a1:.3f}")
    print(f"  L2 decentralisation (Low/High)  : CV {a2:.3f}")
    print(f"  L3 structure (Min/Sat/Meshed)   : CV {a3:.3f} (weighted)")
    print(f"  end-to-end (all three correct)  : CV {e2e:.3f}")
    print("\n  group sizes:")
    for d in [0, 1, 2]:
        for h in sorted(df.loc[df["decentr"] == d, "struct"].unique()):
            mm = (df["decentr"] == d) & (df["struct"] == h)
            if mm.sum():
                print(f"    {DECENTR_NAME[d]} > {name_struct3(df.loc[mm, C.OUTPUTS_OF_INTEREST].mean())}: {mm.sum()}")

    figures(df, X, a1, a2, a3, e2e)
    sunburst(df)
    df[["run", "decentr", "has_local", "hi_local", "struct"]].to_csv(
        os.path.join(OUT, "labels.csv"), index=False)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"3-level asymmetric\nL1 {a1:.3f}  L2 {a2:.3f}  L3 {a3:.3f}  e2e {e2e:.3f}\n")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()