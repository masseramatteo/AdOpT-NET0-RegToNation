"""
Test 1 on the v5 campaign: within-economy effect of d_SS on the network layout, by archetype.

d_SS is the top driver of the network-layout trees in DT_uncert, but it is confounded with the
A/B-vs-D contrast (eta^2 0.179). Each economy has two topologies per archetype (near/far) with
identical economics, so first differences far - near within economy x archetype hold the economy
fixed (= economy fixed effects) and isolate geometry:

    dy = b0 + b1 * dd_SS + b2 * dd_SL + b3 * dd_LL      (per +100 km)

on the network variables built by dt_v5.load(). 95% CIs by bootstrap over economies, paired
across archetypes so the A/B - D difference is valid. A and B have identical distances and are
averaged into one contrast per economy. C is reported but not interpretable for d_SS alone:
within C pairs dd_SS ~ -dd_LL (corr -0.88).

Outputs (into <results>/various_graphs):
    dss_within_economy_effects.png / .csv
    dss_within_economy_ss_link_transitions.png
    dss_level_ss_link_by_archetype.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import dt_v5 as V

OUT = os.path.join(C.RESULTS_FOLDER, "various_graphs")
NAVY, TEAL, RED, AMBER = V.UU_NAVY, V.UU_TEAL, V.UU_RED, V.UU_AMBER
COL = {"AB": NAVY, "D": TEAL, "C": AMBER}
LAB = {"AB": "A/B (both small on one hub)", "D": "D (one small per hub)",
       "C": "C (midway; d_SS ~ -d_LL)"}
ARCH = ["AB", "D", "C"]
Y = ["SS_pipe", "n_LS_arcs", "LL_pipe", "n_pipelines", "highP_share"]
YL = {"SS_pipe": "Small-small link built", "n_LS_arcs": "Hub-small corridors (count)",
      "LL_pipe": "Hub-hub backbone built", "n_pipelines": "Pipelines built (count)",
      "highP_share": "High-pressure share"}
D = ["d_SS", "d_SL_mean", "d_LL"]
CS_LO, CS_HI = 271, 721   # common d_SS support of A/B and D (D p10 .. A p90)
NB = 2000
RS = 42

plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": NAVY, "axes.labelcolor": NAVY,
                     "xtick.color": NAVY, "ytick.color": NAVY})


def pairs(f):
    """far - near differences per economy x archetype; A and B averaged into AB."""
    f = f.copy()
    f["arch"] = f["slot"].str[0]
    f["side"] = f["slot"].str.split("_").str[1]
    w = f.pivot_table(index=["economy", "arch"], columns="side", values=Y + D).dropna()
    dl = pd.DataFrame({c: w[(c, "far")] - w[(c, "near")] for c in Y + D})
    dl["dss_lo"] = np.minimum(w[("d_SS", "near")], w[("d_SS", "far")])
    dl["dss_hi"] = np.maximum(w[("d_SS", "near")], w[("d_SS", "far")])
    dl = dl.reset_index()
    ab = dl[dl.arch.isin(["A", "B"])].groupby("economy").filter(lambda g: len(g) == 2)
    ab = ab.groupby("economy").mean(numeric_only=True).reset_index()
    ab["arch"] = "AB"
    return pd.concat([dl[dl.arch.isin(["C", "D"])], ab], ignore_index=True)


def design_check(dl):
    """Spread and co-movement of the distance contrasts: can d_SS be separated at all?"""
    print("Within-pair contrast far - near (km): median |dd_SS|, corr with dd_SL / dd_LL, "
          "unique share of dd_SS")
    for a in ARCH:
        g = dl[dl.arch == a]
        X = np.column_stack([np.ones(len(g)), g.d_SL_mean, g.d_LL])
        res = g.d_SS - X @ np.linalg.lstsq(X, g.d_SS, rcond=None)[0]
        uniq = (res ** 2).sum() / ((g.d_SS - g.d_SS.mean()) ** 2).sum()
        c = g[D].corr()
        print(f"  {a:2s} n={len(g)}  |dd_SS| {g.d_SS.abs().median():4.0f}  "
              f"corr dSL {c.loc['d_SS', 'd_SL_mean']:+.2f}  dLL {c.loc['d_SS', 'd_LL']:+.2f}  "
              f"unique {uniq:.2f}")


def b1(g, y):
    X = np.column_stack([np.ones(len(g)), g.d_SS / 100, g.d_SL_mean / 100, g.d_LL / 100])
    return np.linalg.lstsq(X, g[y].to_numpy(), rcond=None)[0][1]


def estimate(dl, mask, rng):
    idx = {a: dl[(dl.arch == a) & mask].set_index("economy") for a in ARCH}
    econ = dl.economy.unique()
    boots = {(a, y): [] for a in ARCH for y in Y}
    for _ in range(NB):
        cnt = pd.Series(rng.choice(econ, len(econ))).value_counts()
        for a in ARCH:
            g = idx[a]
            k = g.index.intersection(cnt.index)
            gi = g.loc[np.repeat(k.to_numpy(), cnt.reindex(k).to_numpy())]
            for y in Y:
                boots[(a, y)].append(b1(gi, y))
    rows = []
    for a in ARCH:
        for y in Y:
            lo, hi = np.percentile(boots[(a, y)], [2.5, 97.5])
            rows.append(dict(arch=a, y=y, b=b1(idx[a], y), lo=lo, hi=hi, n=len(idx[a]),
                             base_changed=(idx[a][y] != 0).mean()))
    for y in Y:
        d = np.array(boots[("AB", y)]) - np.array(boots[("D", y)])
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(arch="AB-D", y=y, b=b1(idx["AB"], y) - b1(idx["D"], y), lo=lo, hi=hi,
                         n=np.nan, base_changed=np.nan))
    return pd.DataFrame(rows)


def fig_effects(R):
    ys = Y[:4]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (title, r) in zip(axes, R.items()):
        for i, y in enumerate(ys):
            for j, a in enumerate(ARCH):
                q = r[(r.arch == a) & (r.y == y)].iloc[0]
                yy = i + (j - 1) * 0.22
                ax.plot([q.lo, q.hi], [yy, yy], color=COL[a], lw=2)
                ax.plot(q.b, yy, "o", color=COL[a], ms=6, mfc=COL[a] if a != "C" else "white",
                        label=LAB[a] if i == 0 else None)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        n = {a: int(r[r.arch == a].n.iloc[0]) for a in ARCH}
        ax.set_title(f"{title}\n(n = {n['AB']} AB, {n['D']} D, {n['C']} C economies)",
                     fontsize=10, color=NAVY)
        ax.set_xlabel("Change per +100 km of d_SS (economy, d_SL, d_LL held fixed)")
    axes[0].set_yticks(range(len(ys)))
    axes[0].set_yticklabels([YL[y] for y in ys])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Within-economy effect of the small-small distance on network layout, by archetype",
                 color=NAVY, fontsize=12, x=0.02, ha="left")
    fig.text(0.02, -0.04, "First differences far - near within economy x archetype (= economy fixed "
             "effects); linear probability / linear model; 95% CI from 2000 bootstrap draws over "
             "economies.\nA/B averaged (identical geometry). C hollow: d_SS and d_LL move together "
             "(corr -0.88), joint effect only.", fontsize=7.5, color="grey")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dss_within_economy_effects.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_transitions(dl):
    bins = np.arange(-750, 751, 150)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, a in zip(axes, ARCH):
        g = dl[dl.arch == a].copy()
        g["bin"] = pd.cut(g.d_SS, bins)
        s = g.groupby("bin", observed=True).agg(gain=("SS_pipe", lambda x: (x > 0).mean()),
                                                lose=("SS_pipe", lambda x: (x < 0).mean()),
                                                n=("SS_pipe", "size"))
        s = s[s.n >= 15]
        x = [iv.mid for iv in s.index]
        ax.bar(x, s.gain, width=120, color=TEAL, label="link gained")
        ax.bar(x, -s.lose, width=120, color=RED, label="link lost")
        ax.plot(x, s.gain - s.lose, "o-", color=NAVY, ms=4, label="net")
        for xi, n in zip(x, s.n):
            ax.text(xi, -0.47, f"{n}", ha="center", fontsize=7, color="grey")
        ax.axhline(0, color=NAVY, lw=0.6)
        ax.axvline(0, color="grey", lw=0.6, ls="--")
        ax.set_title(LAB[a], fontsize=10, color=COL[a])
        ax.set_xlabel("Change in d_SS, far - near run (km)")
        ax.set_ylim(-0.5, 0.5)
    axes[0].set_ylabel("Share of economy pairs")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Small-small link: gained or lost between the two runs of the same economy and "
                 "archetype", color=NAVY, fontsize=12, x=0.02, ha="left")
    fig.text(0.02, -0.04, "Raw transitions, no controls (d_SL always rises near->far). Bins with <15 "
             "pairs hidden; grey numbers = pairs per bin. A/B: average of the A and B twin "
             "(half-steps possible).", fontsize=7.5, color="grey")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dss_within_economy_ss_link_transitions.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def fig_levels(f):
    f = f.copy()
    f["A3"] = f["slot"].str[0].replace({"A": "AB", "B": "AB"})
    lb = np.arange(100, 1001, 75)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvspan(CS_LO, CS_HI, color="0.93", zorder=0)
    ax.text((CS_LO + CS_HI) / 2, 0.57, "common support", ha="center", fontsize=8, color="grey")
    for a in ARCH:
        g = f[f.A3 == a].copy()
        g["bin"] = pd.cut(g.d_SS, lb)
        s = g.groupby("bin", observed=True).agg(p=("SS_pipe", "mean"), n=("SS_pipe", "size"),
                                                e=("economy", "nunique"))
        s = s[s.n >= 30]
        ci = 1.96 * np.sqrt(s.p * (1 - s.p) / s.e)  # economies as effective n (conservative)
        x = np.array([iv.mid for iv in s.index])
        ax.plot(x, s.p, "o-", color=COL[a], ms=4, label=LAB[a], mfc=COL[a] if a != "C" else "white")
        ax.fill_between(x, (s.p - ci).clip(0), s.p + ci, color=COL[a], alpha=0.15)
    ax.set_xlabel("d_SS (km)")
    ax.set_ylabel("Share of runs with small-small link")
    ax.set_title("Small-small link vs distance, all runs (pooled over economies)", fontsize=11,
                 color=NAVY, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.text(0.02, -0.04, "Bins with <30 runs hidden; band = 95% CI with economies as effective n. "
             "D sits at higher d_SS (eta^2 0.179) - compare archetypes inside the grey band.",
             fontsize=7.5, color="grey")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dss_level_ss_link_by_archetype.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    f = V.load(C.EXTRACTED_RESULTS)
    print(f"rows {len(f)}  economies {f.economy.nunique()}")
    dl = pairs(f)
    design_check(dl)
    rng = np.random.default_rng(RS)
    cs = (dl.dss_lo >= CS_LO) & (dl.dss_hi <= CS_HI)
    R = {"All pairs": estimate(dl, pd.Series(True, index=dl.index), rng),
         f"Common support (both runs d_SS {CS_LO}-{CS_HI} km)": estimate(dl, cs, rng)}
    for title, r in R.items():
        print(f"\n=== {title}: b1 per +100 km d_SS [95% CI] ===")
        for _, q in r.iterrows():
            sig = "*" if q.lo > 0 or q.hi < 0 else " "
            print(f"  {q.y:12s} {q.arch:5s} {q.b:+.3f} [{q.lo:+.3f}, {q.hi:+.3f}] {sig}")
    pd.concat(R).to_csv(os.path.join(OUT, "dss_within_economy_effects.csv"))
    fig_effects(R)
    fig_transitions(dl)
    fig_levels(f)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
