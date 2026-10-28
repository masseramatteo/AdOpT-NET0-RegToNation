"""
Diagnostic figures for the v5 paired topology design.

Answers, in five figures, the questions raised on the scenario generation:

    Fig 1  feasibility_map.png      do the archetypes occupy disjoint regions?
    Fig 2  design_coverage.png      is the sample uniform, and how much variance
                                    does each archetype retain?
    Fig 3  variance_confounding.png how much distance variance does the
                                    archetype label explain, now and under a
                                    fully crossed A/B/D alternative?
    Fig 4  within_economy.png       does an economy see a narrow slice of the
                                    distance space?
    Fig 5  common_support.png       where can archetype C be compared with the
                                    others?

Every panel reports a variance-based statistic, since that is the axis the
design is judged on:

    Var[d] / Var_unif[d]   variance retention, Var_unif = (hi-lo)^2/12
    eta^2                  share of the variance of a distance explained by the
                           archetype label (between-group SS / total SS)
    w                      share of the variance of a distance occurring WITHIN
                           an economy (1 - between-economy SS / total SS)

Run from the preprocess folder:  python make_design_figures.py
Outputs go to generated_topology_v5/ (PNG + figure_stats.txt).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.gridspec import GridSpec
from scipy.stats import qmc

from generate_node_topology_pooled import (
    build_topology, unit_to_params, D_LL_RANGE, D_SL_RANGE, D_SS_RANGE,
)

# ------------------------------------------------------------------ config --

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated_topology_v5"
PARAMS = OUT / "topology_parameters.csv"

RANGES = {"d_LL": D_LL_RANGE, "d_SL": D_SL_RANGE, "d_SS": D_SS_RANGE}
COLS = {"d_LL": "d_LL_target", "d_SL": "d_SL_target", "d_SS": "d_SS_target"}
LABEL = {"d_LL": r"$d_{LL}$  hub-to-hub  [km]",
         "d_SL": r"$d_{SL}$  small-to-hub (mean)  [km]",
         "d_SS": r"$d_{SS}$  small-to-small  [km]"}

ARCH = ["A", "B", "C", "D"]
ACOL = {"A": "#4C72B0", "B": "#55A868", "C": "#C44E52", "D": "#DD8452"}
ANAME = {"A": "A - both smalls on the larger hub",
         "B": "B - both smalls on the smaller hub",
         "C": "C - both smalls midway",
         "D": "D - one small per hub"}

N_STRATA = 60
N_SAMPLE = 1500          # rows per archetype in the real design
SLICES = [200.0, 500.0, 850.0]     # d_SS values for the feasibility maps
GRID = (320, 240)        # d_LL x d_SL resolution of the maps
SEED = 20260920

PATTERNS = [("ABCD", "feasible as A, B, C and D", "#6A3D9A"),
            ("ABD",  "feasible as A, B and D",    "#4C72B0"),
            ("AB",   "feasible as A and B only",  "#55A868"),
            ("D",    "feasible as D only",        "#DD8452"),
            ("none", "infeasible",                "#D9D9D9")]
PCOL = {k: c for k, _, c in PATTERNS}
PIDX = {k: i for i, (k, _, _) in enumerate(PATTERNS)}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "figure.dpi": 160, "savefig.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})

LOG: list[str] = []


def save(fig, name, tries=5):
    """Write the PNG, retrying: on Windows a sync/indexing service occasionally
    holds the handle and savefig raises OSError 22."""
    import time
    for k in range(tries):
        try:
            fig.savefig(OUT / name)
            plt.close(fig)
            log(f"  wrote {name}")
            return
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(1.5)


def log(s=""):
    print(s)
    LOG.append(s)


# ------------------------------------------------------------- statistics --

def var_ratio(v, par):
    """Var[v] divided by the variance of a uniform over the full stated range."""
    lo, hi = RANGES[par]
    return float(np.var(v, ddof=1) / ((hi - lo) ** 2 / 12))


def eta2(groups):
    """Share of variance explained by the grouping. groups = list of arrays."""
    v = np.concatenate(groups)
    grand = v.mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_t = ((v - grand) ** 2).sum()
    return float(ss_b / ss_t)


def within_share(df, col, by="economy"):
    """1 - between-group SS / total SS."""
    v = df[col].to_numpy()
    grand = v.mean()
    ss_b = sum(len(g) * (g[col].mean() - grand) ** 2 for _, g in df.groupby(by))
    ss_t = ((v - grand) ** 2).sum()
    return float(1 - ss_b / ss_t)


def coverage(v, par):
    lo, hi = RANGES[par]
    u = np.sort((v - lo) / (hi - lo))
    filled = len(np.unique(np.clip((u * N_STRATA).astype(int), 0, N_STRATA - 1)))
    gap = max(u[0], 1 - u[-1], float(np.max(np.diff(u))) if len(u) > 1 else 0.0)
    return dict(span=100 * (v.max() - v.min()) / (hi - lo), strata=filled,
                gap=gap, varr=var_ratio(v, par))


def pattern_of(d):
    ok = {a: build_topology("C1" if a == "C" else a, *d) is not None for a in ARCH}
    if ok["C"]:
        return "ABCD"
    if ok["A"] and ok["D"]:
        return "ABD"
    if ok["A"]:
        return "AB"
    if ok["D"]:
        return "D"
    return "none"


def feasible_stream(pred, n_needed, seed, pool=2 ** 18):
    U = qmc.Sobol(d=3, scramble=True, seed=seed).random(pool)
    keep = []
    for k in range(len(U)):
        p = unit_to_params(U[k])
        if pred(p):
            keep.append(p)
            if len(keep) == n_needed:
                return np.array(keep), len(keep) / (k + 1)
    raise RuntimeError(f"only {len(keep)} feasible of {pool}")


# ------------------------------------------------------------------- data --

def load_design():
    df = pd.read_csv(PARAMS, sep=";")
    for p, c in COLS.items():
        df[p] = df[c]
    return df


# ------------------------------------------------- Fig 1 - feasibility map --

def fig_feasibility(shares):
    fig = plt.figure(figsize=(11.0, 6.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.35, 1.0], hspace=0.38,
                  wspace=0.26, left=0.07, right=0.985, top=0.90, bottom=0.09)

    ll = np.linspace(*D_LL_RANGE, GRID[0])
    sl = np.linspace(*D_SL_RANGE, GRID[1])
    LL, SL = np.meshgrid(ll, sl)
    cmap = matplotlib.colors.ListedColormap([c for _, _, c in PATTERNS])

    for j, dss in enumerate(SLICES):
        ax = fig.add_subplot(gs[0, j])
        M = np.empty(LL.shape, dtype=int)
        for r in range(LL.shape[0]):
            for c in range(LL.shape[1]):
                M[r, c] = PIDX[pattern_of((LL[r, c], SL[r, c], dss))]
        ax.imshow(M, origin="lower", aspect="auto", cmap=cmap,
                  vmin=-0.5, vmax=len(PATTERNS) - 0.5,
                  extent=[ll[0], ll[-1], sl[0], sl[-1]], interpolation="nearest")
        frac_c = float((M == PIDX["ABCD"]).mean())
        ax.set_title(f"slice $d_{{SS}}$ = {dss:.0f} km\nC-feasible area of the slice "
                     f"= {100*frac_c:.1f} %")
        ax.set_xlabel(LABEL["d_LL"])
        if j == 0:
            ax.set_ylabel(LABEL["d_SL"])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)

    # legend
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.legend(handles=[Patch(facecolor=c, label=lab) for _, lab, c in PATTERNS],
              loc="upper left", frameon=False, fontsize=8,
              title="feasibility pattern of a triple", title_fontsize=8.5)
    ax.text(0.0, 0.14, "archetype = a construction rule, not a region:\n"
                       "the same triple can be built as A, B, C or D.\n"
                       r"What partitions is centrality $c=d_{near}/d_{far}$"
                       "\n(an output), at the threshold 0.90.",
            transform=ax.transAxes, va="top", fontsize=7.6, color="#333333")

    # stacked share bar
    ax = fig.add_subplot(gs[1, 1])
    left = 0.0
    for key, lab, col in PATTERNS:
        w = shares[key]
        ax.barh([0], [w], left=left, color=col, edgecolor="white", linewidth=0.6)
        if w > 3:
            ax.text(left + w / 2, 0, f"{w:.1f}%", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")
        left += w
    ax.annotate(f"{shares['ABCD']:.2f}%", xy=(shares["ABCD"] / 2, 0.42),
                xytext=(14, 24), textcoords="offset points", fontsize=7.5,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#6A3D9A"),
                color="#6A3D9A", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("share of the parameter box  [%]")
    ax.set_title("volume of the box by feasibility pattern")

    # marginal acceptance
    ax = fig.add_subplot(gs[1, 2])
    marg = {"A": shares["ABCD"] + shares["ABD"] + shares["AB"],
            "B": shares["ABCD"] + shares["ABD"] + shares["AB"],
            "C": shares["ABCD"],
            "D": shares["ABCD"] + shares["ABD"] + shares["D"]}
    ax.bar(range(4), [marg[a] for a in ARCH],
           color=[ACOL[a] for a in ARCH], width=0.62)
    for i, a in enumerate(ARCH):
        ax.text(i, marg[a] + 1.5, f"{marg[a]:.1f}%", ha="center", fontsize=7.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(ARCH)
    ax.set_ylim(0, 78)
    ax.set_ylabel("acceptance rate  [% of box]")
    ax.set_title(r"$|\mathcal{F}_a| / |\mathcal{B}|$  per archetype")

    fig.suptitle("Fig 1 - The archetype feasible sets overlap; C is a strict subset "
                 r"($\mathcal{F}_C \subset \mathcal{F}_A \cap \mathcal{F}_B \cap "
                 r"\mathcal{F}_D$)", fontsize=10.5, y=0.975)
    save(fig, "feasibility_map.png")


# --------------------------------------------------- Fig 2 - design coverage --

def fig_coverage(df):
    fig = plt.figure(figsize=(11.0, 7.4))
    gs = GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 1.18], hspace=0.30,
                  wspace=0.30, left=0.065, right=0.985, top=0.90, bottom=0.075)
    pars = list(RANGES)
    rng = np.random.default_rng(SEED)

    for i, pi in enumerate(pars):
        for j, pj in enumerate(pars):
            ax = fig.add_subplot(gs[i, j])
            if i == j:
                lo, hi = RANGES[pi]
                bins = np.linspace(lo, hi, N_STRATA + 1)
                ax.hist([df[df.archetype == a][pi] for a in ARCH], bins=bins,
                        stacked=True, color=[ACOL[a] for a in ARCH], linewidth=0)
                ax.axhline(len(df) / N_STRATA, color="black", lw=0.9, ls="--")
                ax.text(0.98, 0.94, "uniform", transform=ax.transAxes, ha="right",
                        va="top", fontsize=6.2, color="black")
                ax.set_yticks([])
                ax.set_title(f"{pi}   pooled fill "
                             f"{coverage(df[pi].to_numpy(), pi)['strata']}/{N_STRATA}"
                             f" strata", fontsize=7.8)
                ax.set_xlim(lo, hi)
            elif i > j:
                for a in ARCH:
                    sub = df[df.archetype == a]
                    k = min(350, len(sub))
                    sub = sub.iloc[rng.choice(len(sub), k, replace=False)]
                    ax.scatter(sub[pj], sub[pi], s=2.0, alpha=0.45,
                               color=ACOL[a], linewidths=0)
                ax.set_xlim(*RANGES[pj])
                ax.set_ylim(*RANGES[pi])
            else:
                # upper triangle: support-induced correlation per archetype
                ax.axis("off")
                rows = [r"$\rho$ within archetype", ""]
                for a in ARCH:
                    sub = df[df.archetype == a]
                    r = np.corrcoef(sub[pi], sub[pj])[0, 1]
                    rows.append(f"{a}   {r:+.2f}")
                ax.text(0.5, 0.92, f"{pi} vs {pj}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=7.6, fontweight="bold")
                for k_, a in enumerate(ARCH):
                    sub = df[df.archetype == a]
                    r = np.corrcoef(sub[pi], sub[pj])[0, 1]
                    ax.text(0.5, 0.70 - 0.15 * k_, f"{a}   {r:+.2f}",
                            transform=ax.transAxes, ha="center", va="top",
                            fontsize=8.2, color=ACOL[a],
                            family="DejaVu Sans Mono")
                ax.text(0.5, 0.06, "geometric, not forced", transform=ax.transAxes,
                        ha="center", fontsize=6.4, color="#777777")
                continue
            if i == 2:
                ax.set_xlabel(LABEL[pj], fontsize=7.4)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != j:
                ax.set_ylabel(LABEL[pi], fontsize=7.4)
            ax.tick_params(labelsize=6.6)

    # variance / coverage table
    ax = fig.add_subplot(gs[:, 3])
    ax.axis("off")
    lines = [r"$\bf{variance\ retention}$   Var$[d]$ / Var$_{unif}[d]$",
             "  1.00 = as spread as a uniform", "  over the full stated range", ""]
    hdr = f"{'':<3}{'d_LL':>9}{'d_SL':>9}{'d_SS':>9}"
    lines.append(hdr)
    for a in ARCH:
        sub = df[df.archetype == a]
        lines.append(f"{a:<3}" + "".join(
            f"{var_ratio(sub[p].to_numpy(), p):>9.2f}" for p in pars))
    lines += ["", r"$\bf{range\ spanned}$  [% of stated range]", hdr]
    for a in ARCH:
        sub = df[df.archetype == a]
        lines.append(f"{a:<3}" + "".join(
            f"{coverage(sub[p].to_numpy(), p)['span']:>9.1f}" for p in pars))
    lines += ["", rf"$\bf{{strata\ filled}}$  (of {N_STRATA})", hdr]
    for a in ARCH:
        sub = df[df.archetype == a]
        lines.append(f"{a:<3}" + "".join(
            f"{coverage(sub[p].to_numpy(), p)['strata']:>9d}" for p in pars))
    lines += ["", r"$\bf{max\ normalised\ gap}$", hdr]
    for a in ARCH:
        sub = df[df.archetype == a]
        lines.append(f"{a:<3}" + "".join(
            f"{coverage(sub[p].to_numpy(), p)['gap']:>9.3f}" for p in pars))
    lines += ["", r"$\bf{pooled}$ variance retention", hdr]
    lines.append("   " + "".join(
        f"{coverage(df[p].to_numpy(), p)['varr']:>9.2f}" for p in pars))
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top",
            family="DejaVu Sans Mono", fontsize=7.0)
    ax.legend(handles=[Patch(facecolor=ACOL[a], label=ANAME[a]) for a in ARCH],
              loc="lower left", bbox_to_anchor=(0.0, -0.02), frameon=False,
              fontsize=7.2)

    fig.suptitle("Fig 2 - Each archetype is low-discrepancy over its OWN feasible "
                 "set; C retains far less variance than A/B/D", fontsize=10.5, y=0.972)
    save(fig, "design_coverage.png")


# ----------------------------------------- Fig 3 - variance confounding --

def fig_confounding(df, P3):
    pars = list(RANGES)
    cur_all = {p: eta2([df[df.archetype == a][p].to_numpy() for a in ARCH]) for p in pars}
    cur_abd = {p: eta2([df[df.archetype == a][p].to_numpy() for a in "ABD"]) for p in pars}
    # crossed design: A, B, D share one triple drawn from the 3-way intersection
    cro_all = {p: eta2([P3[:, k], P3[:, k], P3[:, k],
                        df[df.archetype == "C"][p].to_numpy()])
               for k, p in enumerate(pars)}

    fig = plt.figure(figsize=(11.6, 5.6))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[2.15, 1, 1, 1],
                  height_ratios=[1, 1.12], hspace=0.62, wspace=0.40,
                  left=0.062, right=0.965, top=0.855, bottom=0.10)

    ax = fig.add_subplot(gs[:, 0])
    x = np.arange(3)
    w = 0.26
    series = [("current, all four archetypes", cur_all, "#6A3D9A"),
              ("current, A/B/D only (C excluded)", cur_abd, "#4C72B0"),
              ("crossed A/B/D, all four", cro_all, "#BBBBBB")]
    for i, (lab, d, col) in enumerate(series):
        vals = [d[p] for p in pars]
        ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col, edgecolor="white")
        for xx, v in zip(x + (i - 1) * w, vals):
            ax.text(xx, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.6,
                    rotation=90 if v < 0.02 else 0,
                    va="bottom")
    for y, lab in [(0.01, "small"), (0.06, "medium"), (0.14, "large")]:
        ax.axhline(y, color="#999999", lw=0.6, ls=":")
        ax.text(2.52, y, lab, fontsize=6.6, color="#777777", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[p].split("  ")[0] for p in pars])
    ax.set_ylabel(r"$\eta^2$ = share of the distance variance"
                  "\nexplained by the archetype label")
    ax.set_ylim(0, 0.215)
    ax.set_xlim(-0.55, 2.75)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    ax.set_title(r"(a) confounding: how much of each distance's variance"
                 "\nthe archetype label already tells you", fontsize=8.6)

    heat = [("current A/B", df[df.archetype == "A"][pars].to_numpy()),
            ("current D", df[df.archetype == "D"][pars].to_numpy()),
            ("crossed A/B/D", P3)]
    for j, (nm, M) in enumerate(heat):
        ax = fig.add_subplot(gs[0, j + 1])
        C = np.corrcoef(M.T)
        im = ax.imshow(C, vmin=-0.6, vmax=0.6, cmap="RdBu_r")
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{C[r, c]:+.2f}", ha="center", va="center",
                        fontsize=6.4,
                        color="white" if abs(C[r, c]) > 0.42 else "black")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(pars, fontsize=6.2, rotation=45)
        ax.set_yticklabels(pars, fontsize=6.2)
        ax.set_title(nm, fontsize=7.6)
        if j == 2:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

    # (b) which geometries a crossed design would have to drop
    ax = fig.add_subplot(gs[1, 1])
    Pa = df[df.archetype == "A"][pars].to_numpy()
    keep = np.array([build_topology("D", *p) is not None for p in Pa])
    kept, lost = Pa[keep], Pa[~keep]
    x = np.arange(3)
    for i, (M, lab, col) in enumerate([(kept, "kept (also D-feasible)", "#4C72B0"),
                                       (lost, "dropped", "#C44E52")]):
        med = [(np.median(M[:, k]) - RANGES[p][0]) / (RANGES[p][1] - RANGES[p][0])
               for k, p in enumerate(pars)]
        ax.bar(x + (i - 0.5) * 0.34, med, 0.34, color=col, label=lab)
        for xx, v, k in zip(x + (i - 0.5) * 0.34, med, range(3)):
            ax.text(xx, v - 0.03, f"{np.median(M[:, k]):.0f}", ha="center",
                    va="top", fontsize=6.4, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pars, fontsize=6.8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("median, as fraction\nof the stated range", fontsize=7)
    ax.legend(frameon=False, fontsize=6.4, loc="upper left")
    ax.set_title(f"(b) {100*(~keep).mean():.0f} % of the A/B region\nis not "
                 "D-feasible: what crossing drops", fontsize=7.8)

    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    mean_off = lambda M: np.mean(np.abs(np.corrcoef(M.T)[np.triu_indices(3, 1)]))
    txt = (
        "(c) the price of crossing\n\n"
        r"Forcing A, B and D onto one shared triple sends $\eta^2\to0$ for"
        "\nthose three. But the 3-way intersection is a diagonal region, so\n"
        "the three distances become mutually correlated inside it: mean\n"
        f"$|\\rho|$ rises from {mean_off(Pa):.2f} (A/B) and "
        f"{mean_off(df[df.archetype=='D'][pars].to_numpy()):.2f} (D) to "
        f"{mean_off(P3):.2f} (crossed).\n\n"
        "Between-archetype confounding is TRANSFERRED into within-\n"
        "archetype collinearity - which is what limits a tree's ability to\n"
        r"attribute an effect to $d_{LL}$ rather than $d_{SL}$." "\n\n"
        "What is dropped (b) is far-apart hubs with tightly attached smalls,\n"
        "i.e. exactly where 'which hub' should matter most. C keeps its own\n"
        r"stream either way, so the all-four $\eta^2$ never reaches zero.")
    ax.text(0.0, 1.02, txt, transform=ax.transAxes, va="top", fontsize=7.2,
            color="#333333")

    fig.suptitle(r"Fig 3 - Variance accounting: $\eta^2$ of the archetype label, "
                 "current design vs a fully crossed A/B/D alternative",
                 fontsize=10.5, y=0.965)
    save(fig, "variance_confounding.png")
    return cur_all, cur_abd, cro_all


# ------------------------------------------------ Fig 4 - within economy --

def fig_within(df):
    pars = list(RANGES)
    w = {p: within_share(df, p) for p in pars}
    spread = {p: df.groupby("economy")[p].agg(
        lambda v, lo=RANGES[p][0], hi=RANGES[p][1]: (v.max() - v.min()) / (hi - lo))
        for p in pars}

    rng = np.random.default_rng(SEED)
    econs = rng.choice(np.sort(df.economy.unique()), 40, replace=False)
    order = df[df.economy.isin(econs)].groupby("economy")["d_LL"].agg(
        lambda v: v.max() - v.min()).sort_values().index.to_numpy()

    fig = plt.figure(figsize=(12.6, 6.0))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 0.92, 0.92],
                  height_ratios=[1, 0.95], hspace=0.45, wspace=0.36,
                  left=0.050, right=0.992, top=0.850, bottom=0.115)

    MARK = {"A": ("o", "A/B (shared triple)", "#4C72B0"),
            "C": ("^", "C", ACOL["C"]), "D": ("s", "D", ACOL["D"])}
    for j, p in enumerate(pars):
        ax = fig.add_subplot(gs[:, j])
        for i, e in enumerate(order):
            sub = df[df.economy == e]
            ax.plot([sub[p].min(), sub[p].max()], [i, i], color="#CCCCCC", lw=0.8,
                    zorder=1)
            for a, (mk, _, col) in MARK.items():
                v = sub[sub.archetype == a][p]
                ax.scatter(v, [i] * len(v), s=9, color=col, zorder=2,
                           marker=mk, linewidths=0, alpha=0.9)
        if j == 0:
            ax.legend(handles=[plt.Line2D([], [], ls="", marker=mk, color=col,
                                          markersize=4, label=lab)
                               for mk, lab, col in MARK.values()],
                      loc="lower left", frameon=False, fontsize=6.6,
                      bbox_to_anchor=(-0.02, -0.135), ncol=3,
                      handletextpad=0.25, columnspacing=0.9)
        ax.set_xlim(*RANGES[p])
        ax.set_ylim(-1, len(order))
        ax.set_yticks([])
        ax.set_xlabel(LABEL[p], fontsize=7.4)
        ax.tick_params(labelsize=6.6)
        ax.set_title(f"{p}:  median economy spans\n"
                     f"{spread[p].median()*100:.0f} % of the range", fontsize=8.0)
        if j == 0:
            ax.set_ylabel("40 random economies, sorted by spread", fontsize=7.4)

    ax = fig.add_subplot(gs[0, 3])
    bottom = [w[p] for p in pars]
    ax.bar(range(3), bottom, 0.6, color="#4C72B0", label="within economy")
    ax.bar(range(3), [1 - b for b in bottom], 0.6, bottom=bottom,
           color="#D9D9D9", label="between economies")
    for i, b in enumerate(bottom):
        ax.text(i, b / 2, f"{b:.3f}", ha="center", va="center", fontsize=7.4,
                color="white", fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(pars, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of total variance")
    ax.legend(frameon=False, fontsize=6.8, loc="lower center",
              bbox_to_anchor=(0.5, -0.42), ncol=1)
    ax.set_title("(a) variance decomposition", fontsize=8.2)

    ax = fig.add_subplot(gs[0, 4])
    for p in pars:
        s = np.sort(spread[p].to_numpy())
        ax.plot(s, np.linspace(0, 1, len(s)), lw=1.4, label=p)
    ax.axvline(0.5, color="#999999", lw=0.7, ls=":")
    ax.set_xlabel("within-economy spread\n(max-min) / full range", fontsize=7.4)
    ax.set_ylabel("cumulative share of economies", fontsize=7.4)
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("(b) spread, 750 economies", fontsize=8.2)

    ax = fig.add_subplot(gs[1, 3:])
    ax.axis("off")
    piv = {g: df[df.slot == s].set_index("economy")[pars]
           for g, s in [("AB", "A_near"), ("D", "D_near"), ("C", "C_near")]}
    rows = [f"{'':<6}{'(AB,D)':>10}{'(AB,C)':>10}{'(D,C)':>10}"]
    for p in pars:
        rows.append(f"{p:<6}"
                    f"{np.corrcoef(piv['AB'][p], piv['D'][p])[0,1]:>+10.3f}"
                    f"{np.corrcoef(piv['AB'][p], piv['C'][p])[0,1]:>+10.3f}"
                    f"{np.corrcoef(piv['D'][p], piv['C'][p])[0,1]:>+10.3f}")
    ax.text(0.0, 1.02, "(c) within an economy, the three groups'\n"
                       "     triples are independent draws    "
                       r"($\rho$ below)",
            transform=ax.transAxes, va="top", fontsize=8.0)
    ax.text(0.0, 0.66, "\n".join(rows), transform=ax.transAxes, va="top",
            family="DejaVu Sans Mono", fontsize=7.2)
    ax.text(0.0, 0.16, "Each economy is paired with 8 topologies =\n"
                       "6 distinct triples: A/B near+far (shared),\n"
                       "D near+far, C near+far.",
            transform=ax.transAxes, va="top", fontsize=7.3, color="#333333")

    fig.suptitle("Fig 4 - 81-90 % of the distance variance sits WITHIN an economy: "
                 "pairing does not narrow the space an economy explores",
                 fontsize=10.5, y=0.965)
    save(fig, "within_economy.png")
    return w, spread


# ------------------------------------------------- Fig 5 - common support --

def fig_common_support(df):
    c = df[df.archetype == "C"]
    box = dict(ll=(c.d_LL.min(), c.d_LL.max()), sl=(c.d_SL.min(), c.d_SL.max()),
               ss=(c.d_SS.min(), c.d_SS.max()))
    inside = ((df.d_LL <= box["ll"][1]) & (df.d_SL >= box["sl"][0])
              & (df.d_SS <= box["ss"][1]))
    cnt = df[inside].groupby("archetype").size()
    ec = df[inside].groupby("archetype")["economy"].nunique()

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0),
                             gridspec_kw=dict(width_ratios=[1, 1, 0.95],
                                              wspace=0.30, left=0.06, right=0.985,
                                              top=0.83, bottom=0.14))
    for ax, (px, py) in zip(axes[:2], [("d_LL", "d_SL"), ("d_LL", "d_SS")]):
        out = df[~inside]
        ax.scatter(out[px], out[py], s=2.2, color="#DDDDDD", linewidths=0,
                   label="outside C's box")
        for a in ARCH:
            sub = df[inside & (df.archetype == a)]
            ax.scatter(sub[px], sub[py], s=2.6, color=ACOL[a], linewidths=0,
                       alpha=0.65)
        lox, hix = (box["ll"] if px == "d_LL" else RANGES[px])
        loy, hiy = (box["sl"] if py == "d_SL" else box["ss"])
        ax.add_patch(Rectangle((lox, loy), hix - lox, hiy - loy, fill=False,
                               edgecolor="#C44E52", lw=1.3, ls="--", zorder=5))
        ax.set_xlim(*RANGES[px]); ax.set_ylim(*RANGES[py])
        ax.set_xlabel(LABEL[px], fontsize=7.6)
        ax.set_ylabel(LABEL[py], fontsize=7.6)
        ax.tick_params(labelsize=6.8)
    axes[0].set_title("C's box in $(d_{LL}, d_{SL})$", fontsize=8.4)
    axes[0].legend(handles=[Patch(facecolor=ACOL[a], label=a) for a in ARCH]
                           + [Patch(facecolor="#DDDDDD", label="outside C's box")],
                   loc="lower right", frameon=True, framealpha=0.9, fontsize=6.6,
                   ncol=2, handlelength=1.0, edgecolor="none")
    axes[1].set_title("C's box in $(d_{LL}, d_{SS})$", fontsize=8.4)

    ax = axes[2]
    ax.bar(range(4), [cnt.get(a, 0) for a in ARCH],
           color=[ACOL[a] for a in ARCH], width=0.62)
    for i, a in enumerate(ARCH):
        ax.text(i, cnt.get(a, 0) + 25, f"{cnt.get(a,0)}\n({ec.get(a,0)} econ.)",
                ha="center", fontsize=7.0)
    ax.set_xticks(range(4)); ax.set_xticklabels(ARCH)
    ax.set_ylim(0, 1750)
    ax.set_ylabel("rows inside C's box (of 1500 per archetype)")
    ax.set_title("(c) available counterfactuals for C", fontsize=8.4)

    fig.suptitle("Fig 5 - C is restricted by geometry, not by sampling; compare it "
                 "only on the common support", fontsize=10.5, y=0.955)
    fig.text(0.06, 0.015,
             rf"C's box: $d_{{LL}} \leq$ {box['ll'][1]:.0f} km, "
             rf"$d_{{SL}} \geq$ {box['sl'][0]:.0f} km, "
             rf"$d_{{SS}} \leq$ {box['ss'][1]:.0f} km. "
             "Midway placement forces each small cluster ~$d_{LL}/2$ from each hub, "
             r"so $d_{SL} \leq 500$ caps $d_{LL}$ near 1030 km.", fontsize=7.2,
             color="#333333")
    save(fig, "common_support.png")
    return box, cnt, ec


# ------------------------------------------------------------------- main --

def main():
    log("=" * 74)
    log("v5 paired topology design - diagnostic figures")
    log("=" * 74)
    df = load_design()
    log(f"design: {len(df)} topologies, {df.economy.nunique()} economies, "
        f"{df.groupby(list(COLS.values())).ngroups} distinct triples")

    log("\nMonte Carlo over the parameter box (Sobol, 2^16 points) ...")
    U = qmc.Sobol(d=3, scramble=True, seed=SEED).random(2 ** 16)
    pat = [pattern_of(unit_to_params(u)) for u in U]
    shares = {k: 100 * pat.count(k) / len(pat) for k, _, _ in PATTERNS}
    for k, lab, _ in PATTERNS:
        log(f"   {lab:<28}{shares[k]:6.2f} %")

    log("\ndrawing the 3-way intersection sample (crossed A/B/D alternative) ...")
    P3, f3 = feasible_stream(
        lambda p: all(build_topology(l, *p) is not None for l in ("A", "B", "D")),
        N_SAMPLE, SEED + 7)
    log(f"   acceptance {100*f3:.2f} % of the box")

    log("\nfigures:")
    fig_feasibility(shares)
    fig_coverage(df)
    cur_all, cur_abd, cro_all = fig_confounding(df, P3)
    w, spread = fig_within(df)
    box, cnt, ec = fig_common_support(df)

    log("\n" + "-" * 74)
    log("VARIANCE SUMMARY")
    log("-" * 74)
    log(f"{'':<8}{'d_LL':>10}{'d_SL':>10}{'d_SS':>10}")
    for nm, d in [("eta2 all4", cur_all), ("eta2 ABD", cur_abd),
                  ("eta2 cross", cro_all)]:
        log(f"{nm:<8}" + "".join(f"{d[p]:>10.3f}" for p in RANGES))
    log(f"{'w(within)':<8}" + "".join(f"{w[p]:>10.3f}" for p in RANGES))
    log(f"{'varretain':<8}" + "".join(
        f"{var_ratio(df[p].to_numpy(), p):>10.3f}" for p in RANGES)
        + "   (pooled)")
    for a in ARCH:
        sub = df[df.archetype == a]
        log(f"  {a:<6}" + "".join(
            f"{var_ratio(sub[p].to_numpy(), p):>10.3f}" for p in RANGES))
    log(f"{'spread':<8}" + "".join(
        f"{spread[p].median():>10.3f}" for p in RANGES) + "   (median economy)")
    log("\nC box: d_LL <= %.0f, d_SL >= %.0f, d_SS <= %.0f"
        % (box["ll"][1], box["sl"][0], box["ss"][1]))
    log("rows inside C box: " + ", ".join(f"{a} {cnt.get(a,0)} ({ec.get(a,0)} econ.)"
                                          for a in ARCH))
    (OUT / "figure_stats.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\nall figures + figure_stats.txt written to {OUT}")


if __name__ == "__main__":
    main()
