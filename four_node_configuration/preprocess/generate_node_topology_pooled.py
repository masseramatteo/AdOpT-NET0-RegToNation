"""
Four-node spatial topology design (v4) - pooled Latin hypercube.

How this differs from v3 (generate_node_topology_design.py)
-----------------------------------------------------------
v3 asks every archetype to be, on its own, a Latin hypercube of the three
distance ranges. That is impossible to do well: each archetype can realize only
a subset of the parameter box (56% / 56% / 3% / 64%), so an archetype whose
region is thin has to trade marginal coverage against parameter correlation,
and the equidistant archetype ends up with 5 of 10 strata on d_LL and
max |correlation| 0.88.

v4 targets homogeneity of the design AS A WHOLE, which is what the study needs:

    the N topologies TOGETHER form one Latin hypercube of the three ranges,
    and the archetype label is assigned per point, subject to geometric
    feasibility and equal counts per archetype.

This works because a thin feasible region can still have broad marginal
projections. The equidistant archetype covers only ~3% of the box by volume,
but its projections span d_LL 400-1030, d_SL 211-500 and d_SS 100-943 km, so
its share of the design can occupy the lower d_LL strata while the other
archetypes supply the upper ones. The pooled marginals come out uniform even
though no single archetype reaches every combination.

Consequences:
  - The stated ranges are covered exactly, with no clamping, no rescaling of
    infeasible draws, and no widening of any range.
  - "A midway cluster cannot exist when the hubs are 1500 km apart and the
    small-large distance is capped at 500 km" stops being a defect of the
    sampler. It is a geometric fact - a cluster midway between hubs 1400 km
    apart is ~700 km from each - and those strata are simply served by the
    archetypes that can reach them.
  - Per-archetype coverage is reported, not forced. Comparisons WITHIN one
    archetype are still valid over the range that archetype actually spans.

Archetype C
-----------
C_MODE selects how the midway archetype is built.

  "balanced" (default) - both small clusters symmetric about the L1-L2 axis at
      S = (x, +-d_SS/2), x = sqrt(d_SL^2 - (d_SS/2)^2), each at exactly d_SL
      from the nearer hub, required to be roughly balanced rather than exactly
      equidistant: centrality = d_near / d_far >= C_MIN.
      The two small clusters are exchangeable, as they are in every other
      archetype, and d_SL is a genuine per-node distance inside its stated
      range. Setting C_MIN equal to NEARNESS_RATIO makes the archetypes
      PARTITION the centrality axis: A, B and D take centrality <= the
      threshold, C takes centrality >= it, with no overlap and no gap.
      The pair leans marginally toward one hub; since the two hubs differ in
      demand, half of C's points lean each way (internal labels C1 / C2).

  "equidistant" - the v3 construction: both small clusters on the perpendicular
      bisector, exactly equidistant from both hubs, with d_SL realized as the
      MEAN of the two. Kept for comparison. Note that it makes the two small
      clusters non-exchangeable (they sit at different distances from the hubs,
      211-449 vs 253-760 km in the v3 design) and lets individual node
      distances leave the stated d_SL range, because d_SL is only their mean.

Geometry (identical to v3 for A, B, D)
--------------------------------------
L1 = (0, 0), L2 = (d_LL, 0), L1 the larger hub. Four coplanar nodes have five
shape degrees of freedom; the three parameters constrain four of the six
pairwise distances, leaving one, which each archetype removes with an explicit
symmetry convention. No free angle is sampled.

  A  both smalls attached to L1, symmetric about the axis:
     S = (+-x, +-d_SS/2), x = sqrt(d_SL^2 - (d_SS/2)^2); inner side when that
     keeps both nearer L1, outer otherwise. Feasible iff d_SS <= 2*d_SL.
  B  mirror of A about x = d_LL/2.
  C  see C_MODE above.
  D  one small per hub, symmetric about the perpendicular bisector, both on the
     same side of the axis:
     S1 = ((d_LL-d_SS)/2, y0), S2 = ((d_LL+d_SS)/2, y0),
     y0 = sqrt(d_SL^2 - ((d_LL-d_SS)/2)^2). Feasible iff d_SL >= |d_LL-d_SS|/2.

Coordinates
-----------
Azimuthal-equidistant inverse projection, then a least-squares refinement so the
HAVERSINE distances used downstream by
utilities.calculate_distance_between_coordinates reproduce the planar targets.

Outputs (in OUTPUT_FOLDER)
--------------------------
    NodeLocations_XXXX.csv      one per topology, same format as before
    topology_parameters.csv     one row per topology, targets + realized
    topology_nodes_long.csv     one row per (topology, small cluster)
    topology_coverage.txt       pooled and per-archetype coverage report
    methodology_note.txt        paper-ready description of what was generated
    design_diagnostics.png      pooled marginals and pairwise scatter
    archetype_schematics.png    one representative layout per archetype
    topology_XXXX.png           individual layouts (every PLOT_EVERY)

Note: four_node_configuration/utilities.py:223 reads the scenario files from
preprocess/generated_topology/. Point it at OUTPUT_FOLDER or rename the folder
before launching a batch.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares
from scipy.stats import qmc

# ------------------------------------------------------------------ config --

BIG_NODES = ["Large_cluster1", "Large_cluster2"]      # Large_cluster1 is bigger
SMALL_NODES = ["Small_cluster1", "Small_cluster2"]
NODES = BIG_NODES + SMALL_NODES

# Parameter ranges [km] - realized exactly, never clamped or rescaled.
D_LL_RANGE = (400.0, 1500.0)
D_SL_RANGE = (100.0, 500.0)
D_SS_RANGE = (100.0, 1000.0)

ARCHETYPES = ["A", "B", "C", "D"]
ARCHETYPE_NAME = {
    "A": "both small near Large_cluster1 (the larger hub)",
    "B": "both small near Large_cluster2 (the smaller hub)",
    "C": "both small midway between the hubs",
    "D": "one small cluster per hub",
}

N_PER_ARCHETYPE = 10            # 4 archetypes x 10 = 40 topologies
POOL_SIZE = 2 ** 16             # low-discrepancy candidate pool over the box
SEED = 42

# A small cluster counts as attached to a hub only if it is this much closer to
# it than to the other hub. C_MIN is the same number, so the archetypes
# partition the centrality axis: A/B/D below it, C above it.
NEARNESS_RATIO = 0.90
C_MIN = NEARNESS_RATIO
C_MODE = "balanced"             # "balanced" | "equidistant"
HUB_SIDE = "auto"               # "auto" | "inner" | "outer"  (archetypes A/B)
SPLIT_INSIDE_ONLY = False       # archetype D: require both small in the corridor

# Pooled annealing. The objective is evaluated on all N points at once, so the
# budget scales with N rather than with the per-archetype count.
POOLED_ITER = 300000
POOLED_RESTARTS = 6
POOLED_W_CORR = 5.0
QC_MAX_CORR = 0.30              # orthogonality treated as a constraint

# Pooled uniformity alone is not enough. Left to itself the annealer buys it by
# parking each archetype in a slice of the box - archetype A at high d_LL,
# archetype D at high d_SS - which CONFOUNDS the archetype with the distances
# and makes the decision tree read "one-each topology" and "wide small-small
# separation" as the same signal. This second term asks each archetype to also
# spread over the marginal range its own geometry can reach. Pooled coverage
# stays the primary objective; raise this to trade some of it for a cleaner
# archetype contrast.
W_ARCH_SPREAD = 0.5

# Geographic anchor and projection.
LON0, LAT0 = 4.4777, 51.9244
EARTH_R = 6371.0                # must match utilities.calculate_distance_...
REFINE_LONLAT = True

OUTPUT_FOLDER = Path(__file__).resolve().parent / "generated_topology_v4"
PLOT_EVERY = 1

RANGES = {"d_LL": D_LL_RANGE, "d_SL": D_SL_RANGE, "d_SS": D_SS_RANGE}
_UU_DARK, _UU_YELLOW = "#161D41", "#FFCD00"

# Internal labels. C is split into two mirror variants so that the midway pair
# leans toward the larger and the smaller hub equally often; both are reported
# as archetype C.
def slot_labels():
    out = []
    for a in ARCHETYPES:
        if a == "C" and C_MODE == "balanced":
            half = N_PER_ARCHETYPE // 2
            out += ["C1"] * half + ["C2"] * (N_PER_ARCHETYPE - half)
        else:
            out += [a] * N_PER_ARCHETYPE
    return out


def public_archetype(label):
    return "C" if label in ("C1", "C2") else label


# ---------------------------------------------------------------- geometry --

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _hub_candidates(d_ll, d_sl, d_ss, hub):
    """A / B: both smalls at d_sl from one hub, symmetric about the L1-L2 axis.
    Inner side first, then outer."""
    if d_ss > 2.0 * d_sl:
        return []
    x = math.sqrt(max(0.0, d_sl ** 2 - (d_ss / 2.0) ** 2))
    sides = {"auto": (x, -x), "inner": (x,), "outer": (-x,)}[HUB_SIDE]
    anchor_x = 0.0 if hub == 1 else d_ll
    sign = 1.0 if hub == 1 else -1.0
    return [((anchor_x + sign * xs, d_ss / 2.0),
             (anchor_x + sign * xs, -d_ss / 2.0)) for xs in sides]


def _midway_balanced_candidates(d_ll, d_sl, d_ss, lean):
    """C, balanced mode: both smalls symmetric about the axis, each at exactly
    d_sl from the nearer hub. lean = 1 places them nearer L1, 2 nearer L2. The
    centrality test in archetype_holds enforces that they are roughly midway."""
    if d_ss > 2.0 * d_sl:
        return []
    x = math.sqrt(max(0.0, d_sl ** 2 - (d_ss / 2.0) ** 2))
    if x >= d_ll / 2.0:                     # would not be in the corridor
        return []
    px = x if lean == 1 else d_ll - x
    return [((px, d_ss / 2.0), (px, -d_ss / 2.0))]


def _midway_equidistant_candidates(d_ll, d_sl, d_ss):
    """C, equidistant mode (the v3 construction): both smalls on the
    perpendicular bisector, one each side of the axis, mean hub distance
    equal to d_sl."""

    def hub_dist(y):
        return math.hypot(d_ll / 2.0, y)

    def g(t):                                # decreasing on [0, d_ss/2]
        return hub_dist(t) + hub_dist(d_ss - t) - 2.0 * d_sl

    lo, hi = 0.0, d_ss / 2.0
    g_lo, g_hi = g(lo), g(hi)
    if g_hi > 0.0 or g_lo < 0.0:
        return []
    t = hi if abs(g_hi) < 1e-12 else brentq(g, lo, hi, xtol=1e-10, rtol=1e-14)
    return [(((d_ll / 2.0), t), ((d_ll / 2.0), -(d_ss - t)))]


def _split_candidates(d_ll, d_sl, d_ss):
    """D: one small per hub, pair symmetric about the perpendicular bisector."""
    x1 = (d_ll - d_ss) / 2.0
    if d_sl ** 2 < x1 ** 2:
        return []
    if SPLIT_INSIDE_ONLY and x1 < 0.0:
        return []
    y0 = math.sqrt(max(0.0, d_sl ** 2 - x1 ** 2))
    return [((x1, y0), (d_ll - x1, y0))]


def candidates(label, d_ll, d_sl, d_ss):
    if label == "A":
        return _hub_candidates(d_ll, d_sl, d_ss, hub=1)
    if label == "B":
        return _hub_candidates(d_ll, d_sl, d_ss, hub=2)
    if label in ("C1", "C2"):
        return _midway_balanced_candidates(d_ll, d_sl, d_ss,
                                           lean=1 if label == "C1" else 2)
    if label == "C":
        return _midway_equidistant_candidates(d_ll, d_sl, d_ss)
    if label == "D":
        return _split_candidates(d_ll, d_sl, d_ss)
    raise ValueError(f"Unknown archetype label: {label}")


def realized_distances(pos):
    l1, l2 = pos[BIG_NODES[0]], pos[BIG_NODES[1]]
    s1, s2 = pos[SMALL_NODES[0]], pos[SMALL_NODES[1]]
    d = {
        "d_LL": dist(l1, l2),
        "d_SS": dist(s1, s2),
        "S1_L1": dist(s1, l1), "S1_L2": dist(s1, l2),
        "S2_L1": dist(s2, l1), "S2_L2": dist(s2, l2),
    }
    for i in (1, 2):
        near = min(d[f"S{i}_L1"], d[f"S{i}_L2"])
        far = max(d[f"S{i}_L1"], d[f"S{i}_L2"])
        d[f"S{i}_near"], d[f"S{i}_far"] = near, far
        d[f"S{i}_centrality"] = near / far          # 1.0 = equidistant
    d["d_SL_mean"] = 0.5 * (d["S1_near"] + d["S2_near"])
    d["rho_comp"] = d["d_SS"] / (2.0 * min(d["S1_near"], d["S2_near"]))
    return d


def archetype_holds(label, d, ratio=NEARNESS_RATIO, c_min=C_MIN):
    """The qualitative condition that defines the archetype. A, B and D require
    the small clusters to be clearly attached (centrality <= ratio); C requires
    them to be roughly midway (centrality >= c_min)."""
    if label == "A":
        return d["S1_L1"] <= ratio * d["S1_L2"] and d["S2_L1"] <= ratio * d["S2_L2"]
    if label == "B":
        return d["S1_L2"] <= ratio * d["S1_L1"] and d["S2_L2"] <= ratio * d["S2_L1"]
    if label in ("C1", "C2"):
        return d["S1_centrality"] >= c_min and d["S2_centrality"] >= c_min
    if label == "C":                                 # exact equidistance
        return (abs(d["S1_L1"] - d["S1_L2"]) < 1e-6 * d["S1_L1"]
                and abs(d["S2_L1"] - d["S2_L2"]) < 1e-6 * d["S2_L1"])
    if label == "D":
        return d["S1_L1"] <= ratio * d["S1_L2"] and d["S2_L2"] <= ratio * d["S2_L1"]
    raise ValueError(label)


def build_topology(label, d_ll, d_sl, d_ss):
    """Return (positions, realized) or None if the archetype cannot realize the
    requested triple."""
    for s1, s2 in candidates(label, d_ll, d_sl, d_ss):
        pos = {BIG_NODES[0]: (0.0, 0.0), BIG_NODES[1]: (d_ll, 0.0),
               SMALL_NODES[0]: s1, SMALL_NODES[1]: s2}
        d = realized_distances(pos)
        if not archetype_holds(label, d):
            continue
        tol = 1e-6 * max(d_ll, d_sl, d_ss)
        assert abs(d["d_LL"] - d_ll) < tol
        assert abs(d["d_SS"] - d_ss) < tol
        assert abs(d["d_SL_mean"] - d_sl) < tol
        return pos, d
    return None


# ------------------------------------------------------- pooled LHS design --

def unit_to_params(u):
    return (
        D_LL_RANGE[0] + u[0] * (D_LL_RANGE[1] - D_LL_RANGE[0]),
        D_SL_RANGE[0] + u[1] * (D_SL_RANGE[1] - D_SL_RANGE[0]),
        D_SS_RANGE[0] + u[2] * (D_SS_RANGE[1] - D_SS_RANGE[0]),
    )


def feasible_pool(label, U):
    """Indices of the candidate pool this archetype can realize."""
    return np.asarray(
        [k for k in range(len(U))
         if build_topology(label, *unit_to_params(U[k])) is not None],
        dtype=int)


def _strata_deficit(col, n_strata, lo=0.0, hi=1.0):
    u = (col - lo) / max(hi - lo, 1e-12)
    bins = np.clip((u * n_strata).astype(int), 0, n_strata - 1)
    return np.abs(np.bincount(bins, minlength=n_strata) - 1).sum()


def pooled_objective(sub, n_strata, w_corr, groups=None, group_ranges=None,
                     w_arch=0.0):
    """Distance of the POOLED design from a Latin hypercube of the stated
    ranges, a de-correlation penalty, and - when groups are given - a penalty
    for any archetype that fails to spread over its own reachable range."""
    o1 = sum(_strata_deficit(sub[:, j], n_strata) for j in range(sub.shape[1]))
    c = np.corrcoef(sub, rowvar=False)
    obj = o1 + w_corr * np.abs(c[np.triu_indices(sub.shape[1], k=1)]).sum()

    if groups and w_arch:
        o2 = 0.0
        for lab, rows in groups.items():
            s, m = sub[rows], len(rows)
            lo, hi = group_ranges[lab]
            o2 += sum(_strata_deficit(s[:, j], m, lo[j], hi[j])
                      for j in range(s.shape[1]))
        obj += w_arch * o2
    return obj


def pooled_quality(sub, n_strata):
    """(strata deficit, max |correlation|) of the pooled design."""
    deficit = 0
    for j in range(sub.shape[1]):
        bins = np.clip((sub[:, j] * n_strata).astype(int), 0, n_strata - 1)
        deficit += n_strata - len(set(bins.tolist()))
    c = np.corrcoef(sub, rowvar=False)
    return deficit, float(np.abs(c[np.triu_indices(sub.shape[1], k=1)]).max())


def anneal_pooled(U, pools, labels, seed, n_iter, w_corr, w_arch=W_ARCH_SPREAD):
    """Choose one candidate per slot - the slot's archetype fixing which pool it
    may draw from - so that the N chosen points are as close as possible to a
    Latin hypercube of the three ranges, while each archetype still spreads over
    the range its own geometry can reach."""
    rng = np.random.default_rng(seed)
    N = len(labels)

    groups = {lab: [i for i, l in enumerate(labels) if l == lab]
              for lab in set(labels)}
    group_ranges = {lab: (U[pools[lab]].min(axis=0), U[pools[lab]].max(axis=0))
                    for lab in groups}

    def objective(sub):
        return pooled_objective(sub, N, w_corr, groups, group_ranges, w_arch)

    chosen, used = [], set()
    for lab in labels:                       # distinct starting points
        pool = pools[lab]
        for _ in range(200):
            k = int(pool[rng.integers(len(pool))])
            if k not in used:
                break
        chosen.append(k)
        used.add(k)
    chosen = np.asarray(chosen, dtype=int)

    cur = objective(U[chosen])
    best, best_chosen = cur, chosen.copy()

    t0, t_end = 1.0, 1e-3
    cool = (t_end / t0) ** (1.0 / max(1, n_iter))
    temp = t0
    for _ in range(n_iter):
        i = int(rng.integers(N))
        pool = pools[labels[i]]
        k = int(pool[rng.integers(len(pool))])
        if k in used:
            temp *= cool
            continue
        trial = chosen.copy()
        old = trial[i]
        trial[i] = k
        val = objective(U[trial])
        if val <= cur or rng.random() < math.exp(-(val - cur) / max(temp, 1e-12)):
            chosen, cur = trial, val
            used.discard(old)
            used.add(k)
            if val < best:
                best, best_chosen = val, chosen.copy()
        temp *= cool
        if best <= 1e-12:
            break
    return best_chosen


def build_design(rng_seed=SEED):
    """Return (labels, indices, U, pools) for the pooled design."""
    U = qmc.Sobol(d=3, scramble=True, seed=rng_seed).random(POOL_SIZE)
    labels = slot_labels()

    pools = {}
    for lab in sorted(set(labels)):
        pools[lab] = feasible_pool(lab, U)
        if len(pools[lab]) < labels.count(lab):
            raise RuntimeError(
                f"archetype {lab} has only {len(pools[lab])} feasible candidates "
                f"for {labels.count(lab)} slots - enlarge POOL_SIZE, relax C_MIN, "
                f"or widen the parameter ranges")

    N = len(labels)
    scored = []
    for r in range(POOLED_RESTARTS):
        idx = anneal_pooled(U, pools, labels, rng_seed + 1000 * r,
                            POOLED_ITER, POOLED_W_CORR)
        deficit, corr = pooled_quality(U[idx], N)
        scored.append((deficit, corr, idx))
        print(f"  restart {r + 1}/{POOLED_RESTARTS}: "
              f"pooled strata deficit {deficit}, max|corr| {corr:.2f}")

    # Orthogonality is a constraint, marginal coverage the objective: collinear
    # distance parameters make their relative importance in the decision trees
    # unstable, which is the failure that matters.
    ok = [s for s in scored if s[1] <= QC_MAX_CORR]
    best = min(ok, key=lambda s: (s[0], s[1])) if ok else \
        min(scored, key=lambda s: (s[1], s[0]))
    return labels, best[2], U, pools


# -------------------------------------------------------------- projection --

def haversine(lon1, lat1, lon2, lat2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2.0 * math.asin(math.sqrt(h)) * EARTH_R


def aeqd_inverse(x_km, y_km, lon0=LON0, lat0=LAT0):
    """Azimuthal-equidistant inverse: great-circle distance from the anchor is
    preserved exactly, unlike a flat km-per-degree conversion."""
    r = math.hypot(x_km, y_km)
    if r < 1e-12:
        return lon0, lat0
    az = math.atan2(x_km, y_km)                  # bearing from north
    delta = r / EARTH_R
    p0 = math.radians(lat0)
    lat = math.asin(math.sin(p0) * math.cos(delta)
                    + math.cos(p0) * math.sin(delta) * math.cos(az))
    lon = math.radians(lon0) + math.atan2(
        math.sin(az) * math.sin(delta) * math.cos(p0),
        math.cos(delta) - math.sin(p0) * math.sin(lat))
    return math.degrees(lon), math.degrees(lat)


def positions_to_lonlat(pos):
    """Project to lon/lat, then refine so the haversine distances used
    downstream reproduce the planar targets. Returns (coords, max residual km)."""
    coords = {n: aeqd_inverse(*pos[n]) for n in NODES}
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    target = [dist(pos[NODES[i]], pos[NODES[j]]) for i, j in pairs]

    def resid(vec):
        ll = {NODES[0]: (LON0, LAT0)}
        for k, n in enumerate(NODES[1:]):
            ll[n] = (vec[2 * k], vec[2 * k + 1])
        return [haversine(*ll[NODES[i]], *ll[NODES[j]]) - t
                for (i, j), t in zip(pairs, target)]

    x0 = np.array([v for n in NODES[1:] for v in coords[n]], dtype=float)
    if REFINE_LONLAT:
        x0 = least_squares(resid, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14).x
    err = float(np.max(np.abs(resid(x0))))
    out = {NODES[0]: (LON0, LAT0)}
    for k, n in enumerate(NODES[1:]):
        out[n] = (float(x0[2 * k]), float(x0[2 * k + 1]))
    return out, err


# ------------------------------------------------------------------- i / o --

def write_node_locations(lonlat, name, folder):
    rows = [{"Node": n, "lon": lonlat[n][0], "lat": lonlat[n][1], "alt": 0}
            for n in NODES]
    pd.DataFrame(rows).to_csv(folder / f"NodeLocations_{name}.csv",
                              sep=";", index=False)


def plot_topology(pos, archetype, name, folder):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    lx = [pos[n][0] for n in BIG_NODES]
    ly = [pos[n][1] for n in BIG_NODES]
    sx = [pos[n][0] for n in SMALL_NODES]
    sy = [pos[n][1] for n in SMALL_NODES]
    for b in BIG_NODES:
        for s in SMALL_NODES:
            ax.plot([pos[b][0], pos[s][0]], [pos[b][1], pos[s][1]],
                    color="#dcdfe6", lw=0.8, zorder=1)
    ax.plot(lx, ly, color="#b9bdc9", lw=1.2, zorder=1)
    ax.plot(sx, sy, color="#b9bdc9", lw=1.2, ls=":", zorder=1)
    ax.scatter(lx, ly, s=420, c=_UU_DARK, marker="o", zorder=3,
               edgecolors="white", linewidths=1.5, label="Large clusters")
    ax.scatter(sx, sy, s=180, c=_UU_YELLOW, marker="s", zorder=3,
               edgecolors=_UU_DARK, linewidths=1.2, label="Small clusters")
    for n, (x, y) in pos.items():
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(0, 14),
                    fontsize=8, ha="center", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_title(f"Archetype {archetype} - topology {name}")
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(folder / f"topology_{name}.png", dpi=140)
    plt.close(fig)


def plot_archetype_schematics(df, folder):
    fig, axes = plt.subplots(1, len(ARCHETYPES),
                             figsize=(4.0 * len(ARCHETYPES), 3.6))
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, ARCHETYPES):
        sub = df[df.archetype == a]
        row = sub.iloc[len(sub) // 2]
        pos = {n: (row[f"{n}_x"], row[f"{n}_y"]) for n in NODES}
        for b in BIG_NODES:
            for s in SMALL_NODES:
                ax.plot([pos[b][0], pos[s][0]], [pos[b][1], pos[s][1]],
                        color="#dcdfe6", lw=0.8, zorder=1)
        ax.plot([pos[n][0] for n in BIG_NODES], [pos[n][1] for n in BIG_NODES],
                color="#b9bdc9", lw=1.2, zorder=1)
        ax.scatter([pos[n][0] for n in BIG_NODES], [pos[n][1] for n in BIG_NODES],
                   s=260, c=_UU_DARK, marker="o", zorder=3, edgecolors="white")
        ax.scatter([pos[n][0] for n in SMALL_NODES], [pos[n][1] for n in SMALL_NODES],
                   s=120, c=_UU_YELLOW, marker="s", zorder=3, edgecolors=_UU_DARK)
        ax.set_title(f"{a}: {ARCHETYPE_NAME[a]}", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, ls="--", alpha=0.25)
        ax.set_xlabel("X [km]", fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "archetype_schematics.png", dpi=150)
    plt.close(fig)


def plot_diagnostics(df, folder):
    params = ["d_LL", "d_SL_mean", "d_SS"]
    labels = {"d_LL": "d_LL [km]", "d_SL_mean": "d_SL [km]", "d_SS": "d_SS [km]"}
    rng = {"d_LL": D_LL_RANGE, "d_SL_mean": D_SL_RANGE, "d_SS": D_SS_RANGE}
    colors = dict(zip(ARCHETYPES, ["#161D41", "#4A6FA5", "#C1666B", "#FFCD00"]))
    n = len(df)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    # top row: every point on its parameter axis, with the Latin strata drawn -
    # the pooled design should put exactly one point in each stripe
    for ax, p in zip(axes[0], params):
        lo, hi = rng[p]
        for e in np.linspace(lo, hi, n + 1):
            ax.axvline(e, color="#e8eaee", lw=0.6, zorder=0)
        for a in ARCHETYPES:
            sub = df[df.archetype == a]
            ax.scatter(sub[p], np.full(len(sub), a), s=42, color=colors[a],
                       edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(labels[p])
        ax.set_ylabel("archetype")
        ax.set_title(f"pooled Latin strata ({n})", fontsize=9)

    for ax, (px, py) in zip(axes[1], [("d_LL", "d_SL_mean"),
                                      ("d_LL", "d_SS"),
                                      ("d_SL_mean", "d_SS")]):
        for a in ARCHETYPES:
            sub = df[df.archetype == a]
            ax.scatter(sub[px], sub[py], s=34, color=colors[a], label=a,
                       edgecolors="white", linewidths=0.5)
        ax.set_xlabel(labels[px])
        ax.set_ylabel(labels[py])
        ax.set_xlim(*rng[px])
        ax.set_ylim(*rng[py])
        ax.grid(True, ls="--", alpha=0.25)
    axes[1][0].legend(fontsize=8, title="archetype")
    fig.suptitle("Pooled topology design: marginal strata and pairwise spread")
    fig.tight_layout()
    fig.savefig(folder / "design_diagnostics.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ report --

def max_normalised_gap(values, lo, hi):
    v = np.sort((np.asarray(values) - lo) / (hi - lo))
    return float(np.max(np.diff(np.concatenate([[0.0], v, [1.0]]))))


def coverage_report(df, pools, max_proj_err):
    params = {"d_LL": D_LL_RANGE, "d_SL_mean": D_SL_RANGE, "d_SS": D_SS_RANGE}
    n = len(df)
    out = ["Pooled topology design - realized coverage", "=" * 72,
           f"pool = {POOL_SIZE} scrambled Sobol points in the parameter box",
           f"selection = pooled Latin hypercube over all {n} topologies "
           f"({POOLED_RESTARTS} restarts x {POOLED_ITER} iterations)",
           f"archetype C mode = '{C_MODE}', centrality threshold = {C_MIN}",
           f"nearness ratio = {NEARNESS_RATIO}, hub side policy = '{HUB_SIDE}'",
           f"max haversine-vs-planar residual over all runs = {max_proj_err:.3f} km",
           "",
           "POOLED design - this is the property the study relies on", "-" * 72]

    for p, (lo, hi) in params.items():
        v = df[p].to_numpy()
        bins = np.clip(((v - lo) / (hi - lo) * n).astype(int), 0, n - 1)
        out.append(
            f"    {p:<10} {v.min():7.0f} - {v.max():7.0f} km"
            f"   strata filled {len(set(bins.tolist()))}/{n}"
            f"   max gap {max_normalised_gap(v, lo, hi):.3f}"
            f" (ideal {1 / (n + 1):.3f})")
    corr = df[["d_LL", "d_SL_mean", "d_SS"]].corr().to_numpy()
    off = [abs(corr[0, 1]), abs(corr[0, 2]), abs(corr[1, 2])]
    out.append(f"    max |correlation| between parameters: {max(off):.2f}")
    out.append("")
    out.append(df[["d_LL", "d_SL_mean", "d_SS", "rho_comp"]].corr().round(3).to_string())

    # Confounding check. If the archetype explains a large share of the variance
    # of a distance parameter, the decision tree cannot tell the two apart - it
    # would read "archetype D" and "wide small-small separation" as one signal.
    out += ["", "Archetype vs distances - confounding check", "-" * 72,
            "eta^2 = share of each parameter's variance explained by the",
            "archetype. Near 0 means the archetype contrast is clean; above",
            "~0.5 the two are entangled and the tree will confuse them.", ""]
    for p in params:
        v = df[p].to_numpy()
        grand = v.mean()
        ss_tot = float(((v - grand) ** 2).sum())
        ss_between = float(sum(
            len(g) * (g[p].mean() - grand) ** 2 for _, g in df.groupby("archetype")))
        eta2 = ss_between / ss_tot if ss_tot > 0 else 0.0
        flag = "  <-- entangled" if eta2 > 0.5 else ""
        out.append(f"    {p:<10} eta^2 = {eta2:.2f}{flag}")

    out += ["", "Per-archetype coverage - reported, not forced", "-" * 72,
            "An archetype spans only what its geometry allows. Comparisons",
            "within one archetype are valid over the range it actually covers;",
            "the pooled design above is what guarantees the stated ranges are",
            "sampled homogeneously.", ""]
    for a in ARCHETYPES:
        sub = df[df.archetype == a]
        labs = sorted({l for l in df[df.archetype == a].label})
        frac = sum(len(pools[l]) for l in labs) / POOL_SIZE
        out.append(f"{a} - {ARCHETYPE_NAME[a]}")
        out.append(f"    {len(sub)} topologies, "
                   f"{frac * 100:.1f}% of the box is feasible for it")
        for p, (lo, hi) in params.items():
            v = sub[p].to_numpy()
            out.append(f"    {p:<10} {v.min():7.0f} - {v.max():7.0f} km"
                       f"   spans {(v.max() - v.min()) / (hi - lo) * 100:5.1f}% of "
                       f"[{lo:.0f}, {hi:.0f}]")
        cen = sub[["S1_centrality", "S2_centrality"]].to_numpy()
        out.append(f"    centrality (d_near/d_far) {cen.min():.2f} - {cen.max():.2f}")
        out.append("")
    return "\n".join(out)


def methodology_note(df):
    n = len(df)
    return "\n".join([
        "Spatial topology design - generated description",
        "",
        f"{n} four-node configurations were generated, {N_PER_ARCHETYPE} for "
        f"each of the {len(ARCHETYPES)} archetypes. Each configuration is "
        "defined by three distance parameters - the large-large distance d_LL "
        f"in [{D_LL_RANGE[0]:.0f}, {D_LL_RANGE[1]:.0f}] km, the small-large "
        f"distance d_SL in [{D_SL_RANGE[0]:.0f}, {D_SL_RANGE[1]:.0f}] km and "
        f"the small-small distance d_SS in [{D_SS_RANGE[0]:.0f}, "
        f"{D_SS_RANGE[1]:.0f}] km - together with a discrete archetype fixing "
        "the qualitative arrangement of the two small clusters relative to the "
        "two large ones.",
        "",
        "Given an archetype, the node positions are obtained by solving the "
        "three distance constraints rather than by assigning coordinates: the "
        "archetype supplies the symmetry that reduces the remaining degree of "
        "freedom to a unique layout, so the requested distances are realized "
        "exactly and no additional random factor is introduced. The archetypes "
        "partition the arrangement continuum: a small cluster is treated as "
        f"attached to a hub when it is at most {NEARNESS_RATIO:.2f} times as "
        "far from it as from the other, and as midway otherwise.",
        "",
        "Because four coplanar nodes cannot carry three archetype-independent "
        "distances, each archetype can realize only part of the parameter box. "
        "The design is therefore constructed so that the topologies TOGETHER "
        "form a Latin hypercube of the three ranges: candidate points are "
        "generated as a scrambled Sobol sequence over the full ranges, each "
        "point is tested for geometric feasibility under each archetype, and "
        "points and archetype labels are assigned jointly by simulated "
        "annealing so that every parameter fills each of its strata once, the "
        "three parameters remain uncorrelated, and each archetype receives the "
        "same number of configurations. The stated ranges are thus sampled "
        "homogeneously without rescaling any infeasible draw and without "
        "widening any range; the coverage each individual archetype attains is "
        "reported separately.",
        "",
        "Node coordinates were produced with an azimuthal-equidistant inverse "
        "projection and refined so that the great-circle distances used by the "
        "optimisation model reproduce the sampled distances.",
    ])


# -------------------------------------------------------------------- main --

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print(f"Building pooled design: {N_PER_ARCHETYPE} per archetype, "
          f"{N_PER_ARCHETYPE * len(ARCHETYPES)} topologies, C_MODE='{C_MODE}'")
    labels, idx, U, pools = build_design()
    for lab in sorted(pools):
        print(f"  pool {lab}: {len(pools[lab])} feasible candidates "
              f"({len(pools[lab]) / POOL_SIZE * 100:.1f}% of the box)")

    wide, long, max_err = [], [], 0.0
    order = np.argsort([ARCHETYPES.index(public_archetype(l)) for l in labels],
                       kind="stable")
    for counter, slot in enumerate(order, start=1):
        label = labels[slot]
        a = public_archetype(label)
        d_ll, d_sl, d_ss = unit_to_params(U[idx[slot]])
        built = build_topology(label, d_ll, d_sl, d_ss)
        if built is None:
            raise RuntimeError(f"{label}: lost feasibility for "
                               f"({d_ll:.1f}, {d_sl:.1f}, {d_ss:.1f})")
        pos, d = built
        name = f"{counter:04d}"

        lonlat, err = positions_to_lonlat(pos)
        max_err = max(max_err, err)
        write_node_locations(lonlat, name, OUTPUT_FOLDER)
        if (counter - 1) % PLOT_EVERY == 0:
            plot_topology(pos, a, name, OUTPUT_FOLDER)

        row = {"topology": name, "archetype": a, "label": label,
               "archetype_name": ARCHETYPE_NAME[a],
               "d_LL_target": d_ll, "d_SL_target": d_sl, "d_SS_target": d_ss,
               "proj_residual_km": err, **d}
        for node in NODES:
            row[f"{node}_x"], row[f"{node}_y"] = pos[node]
            row[f"{node}_lon"], row[f"{node}_lat"] = lonlat[node]
        wide.append(row)

        for i, sn in enumerate(SMALL_NODES, start=1):
            long.append({
                "topology": name, "archetype": a, "small_cluster": sn,
                "d_nearest_large": d[f"S{i}_near"],
                "d_other_large": d[f"S{i}_far"],
                "centrality": d[f"S{i}_centrality"],
                "d_small_small": d["d_SS"],
                "d_large_large": d["d_LL"],
                "rho_comp": d["rho_comp"],
            })

        print(f"  {name} [{a}]: LL={d_ll:6.0f}  SL={d_sl:6.0f}  SS={d_ss:6.0f}"
              f"   centrality {d['S1_centrality']:.2f}   proj err {err:.3f} km")

    dfw = pd.DataFrame(wide)
    dfl = pd.DataFrame(long)
    dfw.to_csv(OUTPUT_FOLDER / "topology_parameters.csv", sep=";", index=False)
    dfl.to_csv(OUTPUT_FOLDER / "topology_nodes_long.csv", sep=";", index=False)

    plot_diagnostics(dfw, OUTPUT_FOLDER)
    plot_archetype_schematics(dfw, OUTPUT_FOLDER)

    report = coverage_report(dfw, pools, max_err)
    (OUTPUT_FOLDER / "topology_coverage.txt").write_text(report, encoding="utf-8")
    (OUTPUT_FOLDER / "methodology_note.txt").write_text(
        methodology_note(dfw), encoding="utf-8")

    print("\n" + report)
    print(f"\n{len(dfw)} topologies written to '{OUTPUT_FOLDER}'")


if __name__ == "__main__":
    main()
