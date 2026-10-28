"""
Does the two-stage design fill the joint 15-D parameter space, or does it lie on
a lattice of rows?

Compares the actual campaign design (constrained LHS over 12 techno-economic
axes x 8 own topologies per economy) against a flat 15-D Latin hypercube with no
archetypes and no feasibility constraint, an i.i.d. uniform sample, and the old
v4-style design (40 shared topologies crossed with every economy), at equal
total number of scenarios.

    Fig A  cloud_or_rows.png        the structure itself, at n = 1000/3000/6000
    Fig B  projection_types.png     the three kinds of 2-D projection, n = 6000
    Fig C  coverage_vs_n.png        coverage statistics as n grows
    Fig D  constrained_lhs.png      what the flat LHS costs once the real
                                    constraints (feasibility, archetypes) are
                                    imposed on it
    +      coverage_stats.txt       every number printed

Statistics (all on the unit cube, each axis scaled by its own sampled range):

    2-D / 3-D fill   share of k^2 (k=8) or k^3 (k=4) cells occupied, averaged
                     over all 105 pairs / 455 triples, plus the worst one
    orthants         distinct sign patterns around the mid-point of every axis,
                     out of 2^15 = 32768 - the strictest "is it a cloud" test
    distinct rows    rows differing in at least one of the 15 numeric axes
    CD               centred L2 discrepancy, lower = more uniform
    NN               median nearest-neighbour distance in the 15-D cube
    eff. dim         PCA participation ratio, 15 = isotropic

Run from the preprocess folder:  python make_joint_coverage_figures.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sampling_design as sd
from generate_node_topology_pooled import build_topology

OUT = Path(r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M"
           r"\AdOpT-NET0-RegToNation\four_node_configuration\results"
           r"\Snellius_simulations_10_09\6000_snellius_runs"
           r"\Scenario_combination_coverage")

SLOTS = 8                      # topologies per economy
N_SHOW = [1000, 3000, 6000]    # total scenarios for Fig A
N_CURVE = [800, 1600, 2400, 3200, 4800, 6000]
K2, K3 = 8, 4
SEED, MARGIN = 42, 1.02

COL = {"paired": "#4C72B0", "lhs": "#55A868", "iid": "#999999", "v4": "#DD8452"}
NAME = {"paired": "paired design (actual campaign)",
        "lhs": "flat 15-D LHS (no archetype, no constraint)",
        "iid": "i.i.d. uniform",
        "v4": "v4-style: 40 shared topologies x economies"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "figure.dpi": 160, "savefig.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})

LOG: list[str] = []


def log(s=""):
    print(s)
    LOG.append(s)


# --------------------------------------------------------------- the design --

cfg = sd.load_campaign_definition()
GRID = cfg["PARAM_GRID_FOR_SAMPLING"]
ECON = list(GRID)
SPAT = {"d_LL": (400.0, 1500.0), "d_SL_mean": (100.0, 500.0), "d_SS": (100.0, 1000.0)}
KEYS = ECON + list(SPAT)
D = len(KEYS)
BOUNDS = {k: (float(min(GRID[k])), float(max(GRID[k]))) for k in ECON} | SPAT

MANIFEST, TOPO = sd.load_manifest()
_cache: dict[int, np.ndarray] = {}


def paired_design(n_runs):
    """Unit-cube matrix of the real design with n_runs = economies x 8."""
    if n_runs in _cache:
        return _cache[n_runs]
    n_econ = n_runs // SLOTS
    economies, _, _ = sd.constrained_lhs_economies(GRID, n_econ, SEED, MARGIN)
    rows = sd.build_run_table(economies, MANIFEST, TOPO, {}, n_economies_to_run=n_econ)
    df = pd.DataFrame(rows)
    for c in SPAT:
        df[c] = [TOPO.at[s, c] for s in df.scenario]
    X = np.empty((len(df), D))
    for j, k in enumerate(KEYS):
        lo, hi = BOUNDS[k]
        X[:, j] = (df[k].to_numpy(float) - lo) / (hi - lo)
    X = np.clip(X, 0, 1)
    _cache[n_runs] = X
    return X


def flat_lhs(n_runs, seed=11):
    return qmc.LatinHypercube(d=D, seed=seed).random(n_runs)


def iid(n_runs, seed=7):
    return np.random.default_rng(seed).random((n_runs, D))


def v4_grid(n_runs, seed=5):
    """40 topologies shared by every economy - the design v5 replaced."""
    rng = np.random.default_rng(seed)
    n_econ = max(n_runs // 40, 1)
    economies, _, _ = sd.constrained_lhs_economies(GRID, n_econ, SEED, MARGIN)
    T = TOPO.iloc[rng.choice(len(TOPO), 40, replace=False)][list(SPAT)].to_numpy(float)
    X = np.empty((n_econ * 40, D))
    for i, e in enumerate(economies):
        ue = [(e[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - BOUNDS[k][0]) for k in ECON]
        for j in range(40):
            X[i * 40 + j, :len(ECON)] = ue
            X[i * 40 + j, len(ECON):] = [
                (T[j, m] - SPAT[k][0]) / (SPAT[k][1] - SPAT[k][0])
                for m, k in enumerate(SPAT)]
    return np.clip(X, 0, 1)


BUILD = {"paired": paired_design, "lhs": flat_lhs, "iid": iid, "v4": v4_grid}

# --------------------------------------------- the constrained flat designs --
# What Jan's proposal costs once the real constraints are imposed on it:
#   feas  keep the draw only if the economics are adequate (g + import >= 1.02)
#         AND the distance triple is realizable by at least one archetype
#   arch  as feas, but the archetype is a 16th LHS dimension and the triple must
#         be realizable BY THAT archetype
ARCH_LABELS = {"A": "A", "B": "B", "C": "C1", "D": "D"}


def _econ_params(u12):
    return {k: sd.unit_to_value(u12[i], GRID[k]) for i, k in enumerate(ECON)}


def _triple(u3):
    return tuple(lo + u3[i] * (hi - lo) for i, (lo, hi) in enumerate(SPAT.values()))


def constrained_lhs(n_runs, mode, seed=11, over=12):
    """Return (X, archetype list or None, acceptance rate)."""
    d = D + (1 if mode == "arch" else 0)
    U = qmc.LatinHypercube(d=d, seed=seed).random(n_runs * over)
    keep, arch = [], []
    i = 0
    for i in range(len(U)):
        if not sd.is_feasible(_econ_params(U[i, :len(ECON)]), MARGIN):
            continue
        t = _triple(U[i, len(ECON):D])
        if mode == "feas":
            if not any(build_topology(ARCH_LABELS[a], *t) is not None
                       for a in ("A", "B", "C", "D")):
                continue
        else:
            a = ("A", "B", "C", "D")[min(int(U[i, D] * 4), 3)]
            if build_topology(ARCH_LABELS[a], *t) is None:
                continue
            arch.append(a)
        keep.append(U[i, :D])
        if len(keep) == n_runs:
            break
    if len(keep) < n_runs:
        raise RuntimeError(f"only {len(keep)} feasible of {len(U)}; raise 'over'")
    return np.array(keep), (arch or None), len(keep) / (i + 1)


# --------------------------------------------------------------- statistics --

def fill2(X, k=K2):
    B = np.clip((X * k).astype(int), 0, k - 1)
    fr = [len(np.unique(B[:, a] * k + B[:, b])) / k ** 2
          for a in range(D) for b in range(a + 1, D)]
    return float(np.mean(fr)), float(np.min(fr)), np.array(fr)


def fill3(X, k=K3):
    B = np.clip((X * k).astype(int), 0, k - 1)
    fr = []
    for a in range(D):
        for b in range(a + 1, D):
            for c in range(b + 1, D):
                fr.append(len(np.unique((B[:, a] * k + B[:, b]) * k + B[:, c])) / k ** 3)
    return float(np.mean(fr)), float(np.min(fr))


def orthants(X):
    code = (X > 0.5).astype(np.int64) @ (2 ** np.arange(D))
    return len(np.unique(code))


def distinct_rows(X):
    return len(np.unique(np.round(X, 9), axis=0))


def nn_median(X):
    """Median nearest-neighbour distance among the DISTINCT rows (exact, KD-tree).

    Taken on distinct rows on purpose: the A/B twins are exact duplicates, so on
    the raw matrix this statistic would only re-count the duplicates that the
    'distinct rows' column already reports.
    """
    from scipy.spatial import cKDTree
    U = np.unique(np.round(X, 9), axis=0)
    d, _ = cKDTree(U).query(U, k=2)
    return float(np.median(d[:, 1]))


def eff_dim(X):
    lam = np.clip(np.linalg.eigvalsh(np.cov(X.T)), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def stats(X):
    m2, w2, _ = fill2(X)
    m3, w3 = fill3(X)
    C = np.corrcoef(X.T)
    iu = np.triu_indices(D, 1)
    return dict(n=len(X), fill2=m2, worst2=w2, fill3=m3, worst3=w3,
                orth=orthants(X), distinct=distinct_rows(X),
                triples=distinct_rows(X[:, len(ECON):]),
                cd=float(qmc.discrepancy(X, method="CD")), nn=nn_median(X),
                eff=eff_dim(X), maxcorr=float(np.max(np.abs(C[iu]))))


# ------------------------------------------------------- Fig A - cloud/rows --

PX, PY = "electricity_price_avg", "d_LL"
JX, JY = KEYS.index(PX), KEYS.index(PY)


def unscale(u, key):
    lo, hi = BOUNDS[key]
    return lo + u * (hi - lo)


def fig_cloud_or_rows(S):
    fig = plt.figure(figsize=(11.6, 7.0))
    gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.22,
                  left=0.065, right=0.99, top=0.865, bottom=0.145)
    for r, key in enumerate(["paired", "lhs"]):
        for c, n in enumerate(N_SHOW):
            # window holding ~15 economies, so the comb stays legible as n grows;
            # both rows use the same window, so both show the same point count
            lo_p, hi_p = BOUNDS[PX]
            width = (hi_p - lo_p) * 15.0 / (n / SLOTS)
            zoom = (70.0, 70.0 + width)
            ax = fig.add_subplot(gs[r, c])
            X = BUILD[key](n)
            x, y = unscale(X[:, JX], PX), unscale(X[:, JY], PY)
            ax.scatter(x, y, s=2.2, color=COL[key], alpha=0.45, linewidths=0)
            ax.set_xlim(*BOUNDS[PX])
            ax.set_ylim(*BOUNDS[PY])
            st = S[key][n]
            ax.set_title(f"{NAME[key].split(' (')[0]}   n = {n}\n"
                         f"{st['distinct']} distinct 15-D rows, "
                         f"this pair fills {100*fill_pair(X):.0f} % of 8x8",
                         fontsize=8.0)
            if r == 1:
                ax.set_xlabel("electricity_price_avg  [EUR/MWh]")
            if c == 0:
                ax.set_ylabel(r"$d_{LL}$  [km]")
            # inset: a narrow price window, where the structure is legible
            ins = ax.inset_axes([0.545, 0.05, 0.44, 0.40])
            m = (x >= zoom[0]) & (x <= zoom[1])
            if key == "paired":
                for xv in np.unique(np.round(x[m], 6)):
                    yy = y[np.isclose(x, xv)]
                    ins.plot([xv, xv], [yy.min(), yy.max()], color=COL[key],
                             lw=0.25, alpha=0.18, zorder=1)
            ins.scatter(x[m], y[m], s=3.0, color=COL[key], zorder=2, linewidths=0)
            ins.set_xlim(*zoom)
            ins.set_ylim(*BOUNDS[PY])
            ins.set_xticks([])
            ins.set_yticks([])
            ins.set_title(f"zoom {zoom[0]:.0f}-{zoom[1]:.0f} EUR/MWh "
                          f"({int(m.sum())} points)", fontsize=6.0, pad=1.5)
            for sp in ins.spines.values():
                sp.set_visible(True)
                sp.set_color("#888888")
            ax.indicate_inset_zoom(ins, edgecolor="#888888", alpha=0.6)
    fig.suptitle("Fig A - The paired design is a set of vertical combs, the flat LHS "
                 "a cloud; both cover the plane", fontsize=10.5, y=0.965)
    fig.text(0.065, 0.075,
             "Each economy contributes 8 scenarios that share all 12 economic "
             "coordinates and differ only in geometry, so in any economic x spatial "
             "projection it is a vertical segment of 8 points (inset).\nThat is "
             "replication, not a gap: no cell of the 8x8 grid is empty in either "
             "design, at any n. Both insets use the same price window and therefore "
             "show the same number of points -\nthe difference is how those points "
             "are arranged. The cost of replication is quantified in Fig C.",
             fontsize=7.4, color="#333333", va="top")
    save(fig, "cloud_or_rows.png")


def fill_pair(X, k=K2):
    B = np.clip((X * k).astype(int), 0, k - 1)
    return len(np.unique(B[:, JX] * k + B[:, JY])) / k ** 2


# -------------------------------------------------- Fig B - projection types --

def fig_projection_types(n=6000):
    pairs = [(ECON.index("electricity_price_avg"), ECON.index("hydrogen_import_price"),
              "economic x economic", "elec. price  vs  H2 import price"),
             (JX, JY, "economic x spatial", r"elec. price  vs  $d_{LL}$"),
             (KEYS.index("d_SL_mean"), KEYS.index("d_SS"), "spatial x spatial",
              r"$d_{SL}$  vs  $d_{SS}$")]
    fig = plt.figure(figsize=(11.0, 7.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.24,
                  left=0.07, right=0.99, top=0.865, bottom=0.135)
    for r, key in enumerate(["paired", "lhs"]):
        X = BUILD[key](n)
        B = np.clip((X * K2).astype(int), 0, K2 - 1)
        for c, (a, b, title, short) in enumerate(pairs):
            ax = fig.add_subplot(gs[r, c])
            ax.scatter(X[:, a], X[:, b], s=2.0, color=COL[key], alpha=0.40,
                       linewidths=0)
            occ = set(zip(B[:, a], B[:, b]))
            for i in range(K2):
                for j in range(K2):
                    if (i, j) not in occ:
                        ax.add_patch(Rectangle((i / K2, j / K2), 1 / K2, 1 / K2,
                                               color="#C44E52", alpha=0.35, zorder=3))
            for t in np.linspace(0, 1, K2 + 1):
                ax.axvline(t, color="#BBBBBB", lw=0.4, zorder=0)
                ax.axhline(t, color="#BBBBBB", lw=0.4, zorder=0)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
            npts = len(np.unique(np.round(X[:, [a, b]], 9), axis=0))
            if r == 0:
                ax.set_title(f"{title}\n{short}", fontsize=8.4, pad=6)
            ax.text(0.5, -0.045, f"cells filled {100*len(occ)/K2**2:.0f} %   -   "
                                 f"{npts} distinct points of {len(X)}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7.2,
                    color="#333333")
            if c == 0:
                ax.set_ylabel(NAME[key].split(" (")[0], fontsize=8.4)
    fig.suptitle(f"Fig B - All three kinds of 2-D projection, n = {n}: no empty cell "
                 "in either design (empty cells would be red)", fontsize=10.5, y=0.962)
    fig.text(0.07, 0.068,
             "Left: the 8 scenarios of an economy coincide exactly, so the paired "
             "panel shows 750 points where the flat LHS shows 6000 - the same plane, "
             "sampled at fewer places.\nMiddle: the paired structure shows as columns "
             "of 8 at each economy's price. Right: the paired panel carries the "
             "geometric support constraints; the flat LHS fills the\nrectangle "
             "uniformly, which means 14.8 % of its triples correspond to no "
             "realizable four-node topology at all.  The grid test asks what a "
             "decision tree asks -\nis there a region of the plane the design never "
             "visits - and the answer is no, for all 105 pairs, in both designs.",
             fontsize=7.4, color="#333333", va="top")
    save(fig, "projection_types.png")


# ------------------------------------------------------ Fig C - coverage v n --

def fig_coverage(S):
    panels = [
        ("distinct", "distinct 15-D rows, as a share of n",
         lambda s: s["distinct"] / s["n"], False),
        ("triples", "distinct distance triples", lambda s: s["triples"], True),
        ("worst2", "WORST of the 105 2-D projections\n(share of 8x8 cells filled)",
         lambda s: s["worst2"], False),
        ("fill2", "MEAN of the 105 2-D projections\n(share of 8x8 cells filled)",
         lambda s: s["fill2"], False),
        ("worst3", "WORST of the 455 3-D projections\n(share of 4x4x4 cells filled)",
         lambda s: s["worst3"], False),
        ("orth", f"orthants reached (of {2**D})", lambda s: s["orth"], False),
        ("nn", "median nearest-neighbour distance\nbetween distinct 15-D rows",
         lambda s: s["nn"], False),
        ("cd", "centred L2 discrepancy\n(lower = more uniform)",
         lambda s: s["cd"], True)]
    fig = plt.figure(figsize=(13.2, 6.6))
    gs = GridSpec(2, 4, figure=fig, hspace=0.52, wspace=0.30,
                  left=0.062, right=0.99, top=0.855, bottom=0.155)
    for i, (k, title, get, logy) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        for key in ["paired", "lhs", "iid", "v4"]:
            ns = sorted(S[key])
            ax.plot(ns, [get(S[key][n]) for n in ns], "o-", ms=3.4, lw=1.3,
                    color=COL[key], label=NAME[key])
        if k == "distinct":
            ax.set_ylim(0, 1.08)
            ax.axhline(1.0, ls=":", color="black", lw=0.8)
            ax.text(0.03, 0.10, "1.0 = every scenario is a\ndifferent point in 15-D."
                                "\nHigh here is necessary,\nnot sufficient - see the\n"
                                "panel on its right.",
                    transform=ax.transAxes, fontsize=6.2, color="#555555")
        if k == "triples":
            ax.text(0.03, 0.36, "v4 is stuck at 40 geometries\nat every campaign size",
                    transform=ax.transAxes, fontsize=6.2, color=COL["v4"])
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=8.0)
        ax.set_xlabel("total scenarios n", fontsize=7.4)
        ax.tick_params(labelsize=7)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.6,
               bbox_to_anchor=(0.5, 0.072))
    fig.suptitle("Fig C - Coverage of the joint 15-D space as the campaign grows: "
                 "replication costs distinct points, a lattice costs coverage",
                 fontsize=10.5, y=0.962)
    fig.text(0.062, 0.048,
             "Read the first two panels together. The v4 design never repeats an "
             "(economy, topology) combination, so every one of its rows is distinct "
             "in 15-D - yet its three spatial axes\nonly ever take 40 values, and its "
             "worst 2-D projection is stuck at 0.44 however large the campaign gets. "
             "The paired design repeats (0.75, the A/B twins)\nbut reaches full 2-D "
             "coverage at n = 3200.  Distinctness counts points; the projection panels "
             "count places visited - only the second is what a tree consumes.",
             fontsize=7.2, color="#333333", va="top")
    save(fig, "coverage_vs_n.png")


# --------------------------------------------- Fig D - constraining the LHS --

CCOL = {"paired": "#4C72B0", "free": "#55A868", "feas": "#8172B2", "arch": "#CCB974"}
CNAME = {"paired": "paired\n(actual)",
         "free": "flat LHS\nfree",
         "feas": "flat LHS\n+ feas.",
         "arch": "flat LHS\n+ arch."}
CLONG = {"paired": "paired design (actual campaign)",
         "free": "flat LHS, no constraint",
         "feas": "flat LHS + feasibility",
         "arch": "flat LHS + archetype sampled"}
ORDER = ["free", "feas", "arch", "paired"]


def fig_constrained(n=6000):
    log("\n" + "=" * 96)
    log("Constraining the flat LHS (n = 6000)")
    log("=" * 96)
    X_feas, _, acc_f = constrained_lhs(n, "feas")
    X_arch, arch, acc_a = constrained_lhs(n, "arch")
    Xs = {"paired": paired_design(n), "free": flat_lhs(n),
          "feas": X_feas, "arch": X_arch}
    ACC = {"paired": 1.0, "free": 1.0, "feas": acc_f, "arch": acc_a}
    S = {k: stats(v) for k, v in Xs.items()}
    shares = {a: 100 * arch.count(a) / len(arch) for a in "ABCD"}
    for k in ORDER:
        s = S[k]
        log(f"{CLONG[k]:<40}accept {100*ACC[k]:5.1f} %  "
            f"2D {s['fill2']:.3f}/{s['worst2']:.3f}  3D worst {s['worst3']:.3f}  "
            f"orth {s['orth']:5d}  NN {s['nn']:.3f}  CD {s['cd']:.2e}  "
            f"max|r| {s['maxcorr']:.2f}")
    log("archetype shares with the archetype sampled as a 16th LHS axis: "
        + "  ".join(f"{a} {shares[a]:.1f}%" for a in "ABCD"))
    need = 1500 / (shares["C"] / 100)
    log(f"scenarios needed for 1500 archetype-C runs that way: {need:,.0f}")

    fig = plt.figure(figsize=(12.4, 6.9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.62, wspace=0.30,
                  left=0.065, right=0.985, top=0.835, bottom=0.175)
    x = np.arange(4)

    # (a) archetype shares
    ax = fig.add_subplot(gs[0, 0])
    w = 0.38
    ax.bar(x - w / 2, [25] * 4, w, color=CCOL["paired"], label="paired design")
    ax.bar(x + w / 2, [shares[a] for a in "ABCD"], w, color=CCOL["arch"],
           label="flat LHS, archetype sampled")
    for i, a in enumerate("ABCD"):
        ax.text(i + w / 2, shares[a] + 1.0, f"{shares[a]:.1f}", ha="center",
                fontsize=6.8)
    ax.annotate(f"C collapses to {shares['C']:.1f} %\n"
                f"= {int(round(shares['C']/100*n))} runs of {n}",
                xy=(2 + w / 2, shares["C"] + 0.6), xytext=(2.05, 45),
                fontsize=6.8, color="#B03A3A", ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#B03A3A",
                                connectionstyle="arc3,rad=0.25"))
    ax.set_xticks(x); ax.set_xticklabels(list("ABCD"))
    ax.set_ylabel("share of scenarios  [%]")
    ax.set_ylim(0, 54)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")
    ax.set_title("(a) archetype balance", fontsize=8.6)

    # (b) acceptance
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(4), [100 * ACC[k] for k in ORDER], 0.6,
           color=[CCOL[k] for k in ORDER])
    for i, k in enumerate(ORDER):
        ax.text(i, 100 * ACC[k] + 2, f"{100*ACC[k]:.0f} %", ha="center", fontsize=7)
    ax.set_xticks(range(4))
    ax.set_xticklabels([CNAME[k] for k in ORDER], fontsize=6.2)
    ax.set_ylim(0, 118)
    ax.set_ylabel("draws kept  [%]")
    ax.set_title("(b) cost of rejection", fontsize=8.6)

    panels = [("cd", "centred L2 discrepancy\n(log, lower = more uniform)", True,
               "constraint-driven"),
              ("maxcorr", r"largest pairwise $|\rho|$", False, "constraint-driven"),
              ("orth", f"orthants reached (of {2**D})", False, "replication-driven"),
              ("nn", "median nearest-neighbour\ndistance between distinct rows",
               False, "replication-driven")]
    pos = [(0, 2), (1, 0), (1, 1), (1, 2)]
    for (k, title, logy, tag), (r, c) in zip(panels, pos):
        ax = fig.add_subplot(gs[r, c])
        vals = [S[key][k] for key in ORDER]
        ax.bar(range(4), vals, 0.6, color=[CCOL[key] for key in ORDER])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}" if not logy else f"{v:.1e}", ha="center",
                    va="bottom", fontsize=6.6)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(min(vals) * 0.4, max(vals) * 4)
        else:
            ax.set_ylim(0, max(vals) * 1.22)
        ax.set_xticks(range(4))
        ax.set_xticklabels([CNAME[key] for key in ORDER], fontsize=6.2)
        ax.set_title(f"{title}\n[{tag}]", fontsize=8.0,
                     color="#333333" if tag.startswith("constraint") else "#333333")

    fig.suptitle("Fig D - Constraining the flat LHS: the uniformity advantage is "
                 "mostly spent on the constraints, and archetype C disappears",
                 fontsize=10.5, y=0.955)
    fig.text(0.065, 0.115,
             "Bars ordered free LHS -> + feasibility (economics adequate and the "
             "triple realizable by SOME archetype) -> + archetype sampled as a 16th "
             "LHS axis and the triple realizable\nBY THAT archetype -> the paired "
             "design. Discrepancy and correlation are almost entirely spent by the "
             "constraints: CD goes 2.9e-03 -> 2.8e-02 -> 8.8e-02 against the paired "
             "design's 1.6e-01,\nand max |rho| 0.04 -> 0.13 -> 0.19 against 0.21. "
             "Orthants and nearest-neighbour distance barely move under the "
             "constraints and only fall with the pairing - so constraints explain\n"
             "the uniformity penalty and replication explains the distinct-point "
             "penalty. Panel (a) is the decisive one: sampling the archetype "
             "uniformly and rejecting infeasible draws yields C in proportion to "
             f"its\nfeasible volume (1.95 % of the box), so 1500 archetype-C runs "
             f"would need about {need:,.0f} scenarios. Keeping the four archetypes "
             "balanced requires sampling each over its own region - the current "
             "design.",
             fontsize=7.1, color="#333333", va="top")
    save(fig, "constrained_lhs.png")


# --------------------------------------------------------------------- misc --

def save(fig, name, tries=5):
    for k in range(tries):
        try:
            fig.savefig(OUT / name)
            plt.close(fig)
            log(f"  wrote {name}")
            return
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(2.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("=" * 96)
    log("Joint 15-D coverage: paired campaign design vs a flat LHS at equal n")
    log("=" * 96)
    log(f"{D} axes = {len(ECON)} techno-economic + 3 spatial; "
        f"paired design keeps {SLOTS} topologies per economy")
    log(f"output: {OUT}\n")

    S = {k: {} for k in BUILD}
    ns = sorted(set(N_CURVE) | set(N_SHOW))
    for n in ns:
        for key in BUILD:
            t0 = time.time()
            S[key][n] = stats(BUILD[key](n))
            print(f"   n={n:5d}  {key:<7} {time.time()-t0:5.1f}s", end="\r")
    print(" " * 60, end="\r")

    hdr = (f"{'design':<44}{'n':>6}{'distinct':>10}{'orth':>8}{'2D':>7}{'w2D':>7}"
           f"{'3D':>7}{'w3D':>7}{'NN':>7}{'CD':>10}{'effdim':>8}{'max|r|':>8}")
    log(hdr)
    log("-" * len(hdr))
    for n in ns:
        for key in ["paired", "lhs", "iid", "v4"]:
            s = S[key][n]
            log(f"{NAME[key]:<44}{s['n']:>6}{s['distinct']:>10}{s['orth']:>8}"
                f"{s['fill2']:>7.3f}{s['worst2']:>7.3f}{s['fill3']:>7.3f}"
                f"{s['worst3']:>7.3f}{s['nn']:>7.3f}{s['cd']:>10.2e}"
                f"{s['eff']:>8.2f}{s['maxcorr']:>8.2f}")
        log("")

    log("figures:")
    fig_cloud_or_rows(S)
    fig_projection_types()
    fig_coverage(S)
    fig_constrained()

    (OUT / "coverage_stats.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\ndone -> {OUT}")


if __name__ == "__main__":
    main()
