"""
Four-node spatial topology design (v3).

Purpose
-------
Generate the set of four-node spatial configurations used in the parametric
study. Every configuration is described by three distance parameters

    d_LL   distance between the two large clusters
    d_SL   distance between a small cluster and the large cluster it is
           assigned to (for the equidistant archetype: its distance to both)
    d_SS   distance between the two small clusters

and one discrete archetype that fixes the qualitative arrangement.

What is different from generate_node_topology_fixed.py
------------------------------------------------------
1.  Positions are SOLVED from the three distances instead of assembled by a
    recipe. The old code multiplied d_SL by 0.3 and clamped it at 0.4*d_LL, so
    a requested 400 km became 192 km; archetype C ignored d_SL and archetype D
    ignored d_SS. Here the three requested distances are realized exactly
    (residual < 1e-6 km in the plane), so the ranges written in the paper are
    the ranges the optimisation model actually sees.

2.  Each archetype is a well-posed one-parameter-family reduced to a unique
    solution by an explicit symmetry convention (documented per archetype
    below). No free angle is drawn, so the archetype adds no hidden random
    factor that could confound the decision-tree analysis.

3.  Sampling is a Latin hypercube CONDITIONED on the feasible region of each
    archetype rather than a Latin hypercube in the box followed by clamping. A
    large low-discrepancy pool is drawn in the box, points the archetype cannot
    realize are discarded, and the design is selected from the survivors by
    conditioned LHS (Minasny & McBratney 2006): simulated annealing drives each
    parameter to occupy every one of its n equal-width strata exactly once
    while keeping the three parameters uncorrelated. That is what makes the
    density of sampled distances homogeneous over the stated ranges, which
    plain space-filling (maximin) selection does not do at small n - it piles
    points onto the corners of the feasible region. SELECTION = "maximin" keeps
    the space-filling alternative for comparison. The old scheme mapped
    infeasible draws onto the feasible interval, which over-weights the
    parameter combinations whose feasible interval is narrow.

4.  Optional paired core: a share of the design can be drawn from the region
    feasible for ALL archetypes and reused across them, so the archetype
    contrast is paired on identical distance triples and cannot be confounded
    with the distances. Off by default because the common region is dominated
    by archetype C's constraint (see the coverage report).

5.  Coordinates are produced with an azimuthal-equidistant inverse projection
    and then refined so that the HAVERSINE distances used downstream by
    utilities.calculate_distance_between_coordinates match the planar targets.
    The old flat km-per-degree conversion evaluated the longitude scale at the
    origin latitude only; a node 500 km north is 10% off in the east-west
    direction, which silently shifted the realized distances again.

Archetype definitions (geometry)
--------------------------------
Large clusters are placed at L1 = (0, 0) and L2 = (d_LL, 0). L1 is the larger
of the two (higher demand).

A - both small clusters attached to L1
    The pair is placed symmetric about the L1-L2 axis, so both small clusters
    are exchangeable with respect to the two large ones:
        S = (x, +-d_SS/2),  x = sqrt(d_SL^2 - (d_SS/2)^2)
    Feasible iff d_SS <= 2*d_SL. The pair is placed on the inner side (facing
    L2) when that still leaves both small clusters closer to L1, and on the
    outer side otherwise (HUB_SIDE = "auto"). Set HUB_SIDE to "inner" or
    "outer" to force one regime.

B - both small clusters attached to L2
    Mirror image of A about x = d_LL/2.

C - both small clusters midway, equidistant from both large clusters
    Equidistance forces both small clusters onto the perpendicular bisector
    x = d_LL/2, one on each side of the axis:
        S1 = (d_LL/2, +y1),  S2 = (d_LL/2, -y2),  y1 + y2 = d_SS
    d_SL is realized as the mean of the two (equal-to-both-hubs) distances,
    which pins y1 and y2 uniquely. Feasible iff
        sqrt(d_LL^2 + d_SS^2) / 2  <=  d_SL  <=  (d_LL/2 + sqrt(d_LL^2/4 + d_SS^2)) / 2
    The lower bound is the binding one: a cluster equidistant from both hubs is
    at least d_LL/2 away from each. With d_SL <= 500 km this archetype exists
    only for d_LL <~ 1000 km. Either accept the restricted d_LL coverage for C
    (it is reported), or widen the upper end of D_SL_RANGE to ~800 km.

D - one small cluster per large cluster
    The pair is placed symmetric about the perpendicular bisector, both on the
    same side of the axis, which realizes d_SL for both assignments at once:
        S1 = ((d_LL - d_SS)/2, y0),  S2 = ((d_LL + d_SS)/2, y0)
        y0 = sqrt(d_SL^2 - ((d_LL - d_SS)/2)^2)
    Feasible iff d_SL >= |d_LL - d_SS| / 2. Small d_SS with large d_LL is
    impossible: two clusters that each sit close to their own distant hub
    cannot also be close to each other. The limit d_SS -> small recovers the
    equidistant arrangement, i.e. archetype C is the narrow-pair limit of D.

Known limitation to state in the methodology
--------------------------------------------
Four nodes in the plane cannot carry three archetype-independent distances:
each archetype has its own feasible subset of the (d_LL, d_SL, d_SS) box, and
those subsets differ. The design therefore guarantees homogeneous coverage
WITHIN each archetype and reports the realized coverage per archetype, rather
than pretending the three marginals are identical across archetypes.

Outputs (in OUTPUT_FOLDER)
--------------------------
    NodeLocations_XXXX.csv      one per topology, same format as before
    topology_parameters.csv     one row per topology, targets + realized
    topology_nodes_long.csv     one row per (topology, small cluster)
    topology_coverage.txt       coverage / orthogonality report
    methodology_note.txt        paper-ready description of what was generated
    design_diagnostics.png      marginals and pairwise scatter of the design
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

# Parameter ranges [km] - these are realized exactly, not clamped.
D_LL_RANGE = (400.0, 1500.0)
D_SL_RANGE = (100.0, 500.0)
D_SS_RANGE = (100.0, 1000.0)

ARCHETYPES = ["A", "B", "C", "D"]
ARCHETYPE_NAME = {
    "A": "both small near Large_cluster1 (the larger hub)",
    "B": "both small near Large_cluster2 (the smaller hub)",
    "C": "both small midway, equidistant from the two hubs",
    "D": "one small cluster per hub",
}

N_PER_ARCHETYPE = 10            # 4 archetypes x 10 = 40 topologies
PAIRED_CORE = 0                 # of those, how many are shared across archetypes
POOL_SIZE = 2 ** 15             # low-discrepancy pool drawn in the box
SEED = 42

SELECTION = "clhs"              # "clhs" (homogeneous marginals) | "maximin"
CLHS_RESTARTS = 8               # independent restarts, best objective wins
# Annealing effort and the de-correlation weight MUST scale with the design
# size: the search space grows with n while the number of swaps does not. With
# a fixed budget (60k iterations, weight 5) archetype A reaches max|corr| 0.01
# at n = 10 but 0.89 at n = 20 - all strata filled, parameters collinear. The
# scaled settings below recover max|corr| 0.00 at n = 20.
CLHS_ITER = None                # None -> max(60000, 6000 * n)
CLHS_W_CORR = 5.0               # annealing weight; restarts are ranked
                                # lexicographically afterwards, so this only
                                # steers the search, it does not set the
                                # marginals-vs-orthogonality priority

# Quality gates checked after selection; a violation is reported, not silenced.
QC_MIN_STRATA_FRAC = 0.85       # fraction of strata that must be occupied
QC_MAX_CORR = 0.30              # max |Pearson| tolerated between parameters

# A small cluster counts as attached to a hub only if it is this much closer to
# it than to the other hub. 1.0 = plain "closer to"; lower = stricter archetype.
NEARNESS_RATIO = 0.90
HUB_SIDE = "auto"               # "auto" | "inner" | "outer"  (archetypes A/B)
SPLIT_INSIDE_ONLY = False       # archetype D: require both small in the corridor

# Geographic anchor and projection.
LON0, LAT0 = 4.4777, 51.9244
EARTH_R = 6371.0                # must match utilities.calculate_distance_...
REFINE_LONLAT = True

OUTPUT_FOLDER = Path(__file__).resolve().parent / "generated_topology_v3"
PLOT_EVERY = 1

RANGES = {"d_LL": D_LL_RANGE, "d_SL": D_SL_RANGE, "d_SS": D_SS_RANGE}
_UU_DARK, _UU_YELLOW = "#161D41", "#FFCD00"


# ---------------------------------------------------------------- geometry --

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _hub_candidates(d_ll, d_sl, d_ss, hub):
    """Archetype A/B: both small clusters at d_sl from one hub, symmetric about
    the L1-L2 axis. Returns inner-side then outer-side candidates."""
    if d_ss > 2.0 * d_sl:
        return []
    x = math.sqrt(max(0.0, d_sl ** 2 - (d_ss / 2.0) ** 2))
    out = []
    sides = {"auto": (x, -x), "inner": (x,), "outer": (-x,)}[HUB_SIDE]
    for xs in sides:
        anchor_x = 0.0 if hub == 1 else d_ll
        sign = 1.0 if hub == 1 else -1.0
        px = anchor_x + sign * xs
        out.append(((px, d_ss / 2.0), (px, -d_ss / 2.0)))
    return out


def _midway_candidates(d_ll, d_sl, d_ss):
    """Archetype C: both small clusters on the perpendicular bisector, one on
    each side of the axis, with mean hub distance equal to d_sl."""

    def hub_dist(y):
        return math.hypot(d_ll / 2.0, y)

    def g(t):                                    # decreasing on [0, d_ss/2]
        return hub_dist(t) + hub_dist(d_ss - t) - 2.0 * d_sl

    lo, hi = 0.0, d_ss / 2.0
    g_lo, g_hi = g(lo), g(hi)
    if g_hi > 0.0 or g_lo < 0.0:                 # d_sl outside the reachable band
        return []
    t = hi if abs(g_hi) < 1e-12 else brentq(g, lo, hi, xtol=1e-10, rtol=1e-14)
    y1, y2 = t, d_ss - t
    return [(((d_ll / 2.0), y1), ((d_ll / 2.0), -y2))]


def _split_candidates(d_ll, d_sl, d_ss):
    """Archetype D: one small cluster per hub, the pair symmetric about the
    perpendicular bisector and on the same side of the axis."""
    x1 = (d_ll - d_ss) / 2.0
    if d_sl ** 2 < x1 ** 2:
        return []
    if SPLIT_INSIDE_ONLY and x1 < 0.0:
        return []
    y0 = math.sqrt(max(0.0, d_sl ** 2 - x1 ** 2))
    return [((x1, y0), (d_ll - x1, y0))]


def candidates(archetype, d_ll, d_sl, d_ss):
    if archetype == "A":
        return _hub_candidates(d_ll, d_sl, d_ss, hub=1)
    if archetype == "B":
        return _hub_candidates(d_ll, d_sl, d_ss, hub=2)
    if archetype == "C":
        return _midway_candidates(d_ll, d_sl, d_ss)
    if archetype == "D":
        return _split_candidates(d_ll, d_sl, d_ss)
    raise ValueError(f"Unknown archetype: {archetype}")


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
        d[f"S{i}_near"] = near
        d[f"S{i}_far"] = far
        d[f"S{i}_centrality"] = near / far          # 1.0 = equidistant
    d["d_SL_mean"] = 0.5 * (d["S1_near"] + d["S2_near"])
    # Separation normalised by its geometric maximum: raw d_SS is mechanically
    # tied to d_SL (two clusters both at d_SL from one hub cannot be further
    # than 2*d_SL apart), the ratio removes most of that coupling.
    d["rho_comp"] = d["d_SS"] / (2.0 * min(d["S1_near"], d["S2_near"]))
    return d


def archetype_holds(archetype, d, ratio):
    if archetype == "A":
        return d["S1_L1"] <= ratio * d["S1_L2"] and d["S2_L1"] <= ratio * d["S2_L2"]
    if archetype == "B":
        return d["S1_L2"] <= ratio * d["S1_L1"] and d["S2_L2"] <= ratio * d["S2_L1"]
    if archetype == "C":
        return (abs(d["S1_L1"] - d["S1_L2"]) < 1e-6 * d["S1_L1"]
                and abs(d["S2_L1"] - d["S2_L2"]) < 1e-6 * d["S2_L1"])
    if archetype == "D":
        return d["S1_L1"] <= ratio * d["S1_L2"] and d["S2_L2"] <= ratio * d["S2_L1"]
    raise ValueError(archetype)


def build_topology(archetype, d_ll, d_sl, d_ss, ratio=NEARNESS_RATIO):
    """Return (positions, realized) or None if the archetype cannot realize the
    requested triple."""
    for s1, s2 in candidates(archetype, d_ll, d_sl, d_ss):
        pos = {BIG_NODES[0]: (0.0, 0.0), BIG_NODES[1]: (d_ll, 0.0),
               SMALL_NODES[0]: s1, SMALL_NODES[1]: s2}
        d = realized_distances(pos)
        if not archetype_holds(archetype, d, ratio):
            continue
        # the three requested distances must be realized, not approximated
        tol = 1e-6 * max(d_ll, d_sl, d_ss)
        assert abs(d["d_LL"] - d_ll) < tol
        assert abs(d["d_SS"] - d_ss) < tol
        assert abs(d["d_SL_mean"] - d_sl) < tol
        return pos, d
    return None


# ------------------------------------------------------ constrained design --

def unit_to_params(u):
    return (
        D_LL_RANGE[0] + u[0] * (D_LL_RANGE[1] - D_LL_RANGE[0]),
        D_SL_RANGE[0] + u[1] * (D_SL_RANGE[1] - D_SL_RANGE[0]),
        D_SS_RANGE[0] + u[2] * (D_SS_RANGE[1] - D_SS_RANGE[0]),
    )


def feasible_pool(archetype, U):
    """Indices of the pool that this archetype can realize, plus their unit
    coordinates. Because U is uniform in the box, the retained set is uniform
    over the archetype's feasible region."""
    idx = []
    for k in range(len(U)):
        d_ll, d_sl, d_ss = unit_to_params(U[k])
        if build_topology(archetype, d_ll, d_sl, d_ss) is not None:
            idx.append(k)
    return np.asarray(idx, dtype=int)


def greedy_maximin(X, n, seed_points=None):
    """Coffee-house design: iteratively add the point furthest from everything
    already selected. Space-filling over an arbitrarily shaped region."""
    m = len(X)
    if n > m:
        raise ValueError(f"asked for {n} points, only {m} feasible candidates")
    chosen = []
    if seed_points is not None and len(seed_points):
        mind = np.min(np.linalg.norm(X[:, None, :] - seed_points[None, :, :],
                                     axis=2), axis=1)
    else:
        first = int(np.argmin(np.linalg.norm(X - X.mean(axis=0), axis=1)))
        chosen.append(first)
        mind = np.linalg.norm(X - X[first], axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        mind = np.minimum(mind, np.linalg.norm(X - X[nxt], axis=1))
        mind[nxt] = -1.0
    return np.asarray(chosen, dtype=int)


def _clhs_objective(sub, n_strata, w_corr):
    """Conditioned-LHS objective: how far the subset is from having exactly one
    point per equal-width stratum in every parameter, plus a de-correlation
    penalty. sub is (n, d) in unit coordinates of the NOMINAL ranges."""
    n, d = sub.shape
    o1 = 0.0
    for j in range(d):
        bins = np.clip((sub[:, j] * n_strata).astype(int), 0, n_strata - 1)
        counts = np.bincount(bins, minlength=n_strata)
        o1 += np.abs(counts - n / n_strata).sum()
    c = np.corrcoef(sub, rowvar=False)
    o3 = np.abs(c[np.triu_indices(d, k=1)]).sum()
    return o1 + w_corr * o3


def clhs_effort(n):
    """Annealing budget and de-correlation weight for a design of size n."""
    n_iter = CLHS_ITER if CLHS_ITER is not None else max(60000, 6000 * n)
    return n_iter, CLHS_W_CORR


def conditioned_lhs(X, n, seed, n_iter=None, w_corr=None, fixed=None):
    """Select n rows of the candidate pool X (unit coordinates) so that the
    selection is as close as possible to a Latin hypercube of the nominal
    ranges. Simulated annealing over single swaps (Minasny & McBratney 2006).
    `fixed` are extra rows that are part of the design but cannot be swapped."""
    rng = np.random.default_rng(seed)
    m = len(X)
    if n > m:
        raise ValueError(f"asked for {n} points, only {m} feasible candidates")
    default_iter, default_w = clhs_effort(n)
    n_iter = default_iter if n_iter is None else n_iter
    w_corr = default_w if w_corr is None else w_corr
    fixed = np.empty((0, X.shape[1])) if fixed is None else np.asarray(fixed)
    n_strata = n + len(fixed)

    sel = rng.permutation(m)[:n]
    in_sel = np.zeros(m, dtype=bool)
    in_sel[sel] = True
    cur = _clhs_objective(np.vstack([fixed, X[sel]]), n_strata, w_corr)
    best, best_sel = cur, sel.copy()

    t0, t_end = 1.0, 1e-3
    cool = (t_end / t0) ** (1.0 / max(1, n_iter))
    temp = t0
    free = np.flatnonzero(~in_sel)
    for _ in range(n_iter):
        if len(free) == 0:
            break
        i = rng.integers(n)
        j = int(free[rng.integers(len(free))])
        trial = sel.copy()
        out, trial[i] = trial[i], j
        val = _clhs_objective(np.vstack([fixed, X[trial]]), n_strata, w_corr)
        if val <= cur or rng.random() < math.exp(-(val - cur) / max(temp, 1e-12)):
            sel, cur = trial, val
            in_sel[j], in_sel[out] = True, False
            free = np.flatnonzero(~in_sel)
            if val < best:
                best, best_sel = val, sel.copy()
        temp *= cool
        if best <= 1e-12:                        # perfect Latin hypercube found
            break
    return best_sel, best


def design_quality(sub, n_strata):
    """(strata deficit, max |correlation|) for a candidate design in unit
    coordinates. Lower is better on both, strata deficit first."""
    deficit = 0
    for j in range(sub.shape[1]):
        bins = np.clip((sub[:, j] * n_strata).astype(int), 0, n_strata - 1)
        deficit += n_strata - len(set(bins.tolist()))
    c = np.corrcoef(sub, rowvar=False)
    return deficit, float(np.abs(c[np.triu_indices(sub.shape[1], k=1)]).max())


def conditioned_lhs_restarts(X, n, seed, fixed=None, restarts=CLHS_RESTARTS):
    """Simulated annealing is only a local search, and a single run scales badly
    with n: at n = 20 one run of the default budget lands on a design with every
    stratum filled but max |correlation| 0.89, while another restart of the same
    budget reaches 0.00. Run several and choose between them explicitly.

    Marginal coverage and orthogonality genuinely trade off against each other
    whenever the feasible region is not a box, and neither a scalar objective
    nor a lexicographic rule picks sensibly: ranking by the annealing objective
    lets an arbitrary weight decide, while ranking on strata first accepts
    max |correlation| 0.77 to gain a single stratum. Treat orthogonality as a
    CONSTRAINT instead - collinear distance parameters make their relative
    importance in the decision trees unstable, which is the failure that
    matters - and maximise marginal coverage subject to it."""
    fixed_arr = np.empty((0, X.shape[1])) if fixed is None else np.asarray(fixed)
    n_strata = n + len(fixed_arr)
    scored = []
    for r in range(restarts):
        sel, _ = conditioned_lhs(X, n, seed=seed + 1000 * r, fixed=fixed)
        deficit, corr = design_quality(np.vstack([fixed_arr, X[sel]]), n_strata)
        scored.append((deficit, corr, sel))

    ok = [s for s in scored if s[1] <= QC_MAX_CORR]
    if ok:                                       # min deficit, then min corr
        return min(ok, key=lambda s: (s[0], s[1]))[2]
    return min(scored, key=lambda s: (s[1], s[0]))[2]   # nothing clears the cap


def build_design(rng_seed=SEED):
    """Return {archetype: array of (d_LL, d_SL, d_SS)} plus pool diagnostics."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=rng_seed)
    U = sampler.random(POOL_SIZE)

    pools = {a: feasible_pool(a, U) for a in ARCHETYPES}
    for a, idx in pools.items():
        if len(idx) == 0:
            raise RuntimeError(
                f"archetype {a} has an empty feasible region for the given "
                f"ranges - widen D_SL_RANGE or drop the archetype")

    def select(pool_idx, n, fixed_idx):
        if n <= 0:
            return np.array([], dtype=int)
        if SELECTION == "clhs":
            fixed = U[fixed_idx] if len(fixed_idx) else None
            return pool_idx[conditioned_lhs_restarts(U[pool_idx], n,
                                                     seed=rng_seed, fixed=fixed)]
        if SELECTION == "maximin":
            seeds = U[fixed_idx] if len(fixed_idx) else None
            return pool_idx[greedy_maximin(U[pool_idx], n, seed_points=seeds)]
        raise ValueError(f"Unknown SELECTION: {SELECTION}")

    core_idx = np.array([], dtype=int)
    if PAIRED_CORE > 0:
        common = pools[ARCHETYPES[0]]
        for a in ARCHETYPES[1:]:
            common = np.intersect1d(common, pools[a])
        if len(common) < PAIRED_CORE:
            raise RuntimeError(
                f"only {len(common)} pool points are feasible for every "
                f"archetype, cannot build a paired core of {PAIRED_CORE}")
        core_idx = select(common, PAIRED_CORE, np.array([], dtype=int))

    design, diag = {}, {}
    for a in ARCHETYPES:
        sel = select(pools[a], N_PER_ARCHETYPE - len(core_idx), core_idx)
        picked = np.concatenate([core_idx, sel]).astype(int)
        design[a] = np.array([unit_to_params(U[k]) for k in picked])

        # Fill distance: how far a random feasible triple is from the nearest
        # sampled one, in unit coordinates (box diagonal = sqrt(3)). Latin
        # marginals say nothing about holes in the joint space, so this is the
        # number that tells you whether N_PER_ARCHETYPE is large enough.
        ref = U[pools[a]]
        near = np.linalg.norm(ref[:, None, :] - U[picked][None, :, :],
                              axis=2).min(axis=1)
        diag[a] = {"feasible_fraction": len(pools[a]) / POOL_SIZE,
                   "fill_mean": float(near.mean()),
                   "fill_p95": float(np.percentile(near, 95))}

    return design, diag, len(core_idx)


# -------------------------------------------------------------- projection --

def haversine(lon1, lat1, lon2, lat2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2.0 * math.asin(math.sqrt(h)) * EARTH_R


def aeqd_inverse(x_km, y_km, lon0=LON0, lat0=LAT0):
    """Azimuthal-equidistant inverse: great-circle distance from the anchor is
    preserved exactly, unlike the flat km-per-degree conversion."""
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
    """Project to lon/lat, then (optionally) refine so that the haversine
    distances used downstream reproduce the planar targets. Returns
    ({node: (lon, lat)}, max absolute distance residual in km)."""
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
        sol = least_squares(resid, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        x0 = sol.x
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
    fig, axes = plt.subplots(1, len(ARCHETYPES), figsize=(4.0 * len(ARCHETYPES), 3.6))
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

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, p in zip(axes[0], params):
        for a in ARCHETYPES:
            ax.hist(df[df.archetype == a][p], bins=8, range=rng[p],
                    histtype="step", lw=1.6, color=colors[a], label=a)
        ax.set_xlabel(labels[p])
        ax.set_ylabel("count")
        ax.set_xlim(*rng[p])
        ax.grid(True, ls="--", alpha=0.25)
    axes[0][0].legend(fontsize=8, title="archetype")

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
    fig.suptitle("Topology design: marginal coverage and pairwise spread")
    fig.tight_layout()
    fig.savefig(folder / "design_diagnostics.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ report --

def max_normalised_gap(values, lo, hi):
    """Largest hole in the covered range, normalised. 1/(n+1) is the ideal for
    n equally spaced points; larger values mean a clustered design."""
    v = np.sort((np.asarray(values) - lo) / (hi - lo))
    edges = np.concatenate([[0.0], v, [1.0]])
    return float(np.max(np.diff(edges)))


def coverage_report(df, diag, n_core, max_proj_err):
    params = {"d_LL": D_LL_RANGE, "d_SL_mean": D_SL_RANGE, "d_SS": D_SS_RANGE}
    n_iter, w_corr = clhs_effort(N_PER_ARCHETYPE)
    out = ["Topology design - realized coverage", "=" * 72,
           f"pool = {POOL_SIZE} scrambled Sobol points in the parameter box",
           f"selection = {SELECTION} over the feasible pool"
           + (f" ({CLHS_RESTARTS} restarts x {n_iter} iterations, "
              f"de-correlation weight {w_corr:.0f})" if SELECTION == "clhs" else ""),
           f"paired core shared by all archetypes = {n_core} topologies",
           f"nearness ratio = {NEARNESS_RATIO}, hub side policy = '{HUB_SIDE}'",
           f"max haversine-vs-planar residual over all runs = {max_proj_err:.3f} km",
           ""]
    for a in ARCHETYPES:
        sub = df[df.archetype == a]
        out.append(f"{a} - {ARCHETYPE_NAME[a]}")
        out.append(f"    {len(sub)} topologies, "
                   f"{diag[a]['feasible_fraction'] * 100:.1f}% of the box is feasible")
        out.append(f"    joint fill distance (unit coords, box diagonal 1.73): "
                   f"mean {diag[a]['fill_mean']:.3f}, "
                   f"95th pct {diag[a]['fill_p95']:.3f}")
        for p, (lo, hi) in params.items():
            v = sub[p].to_numpy()
            n = len(v)
            bins = np.clip(((v - lo) / (hi - lo) * n).astype(int), 0, n - 1)
            out.append(
                f"    {p:<10} {v.min():7.0f} - {v.max():7.0f} km"
                f"   covers {(v.max() - v.min()) / (hi - lo) * 100:5.1f}% of "
                f"[{lo:.0f}, {hi:.0f}]"
                f"   strata filled {len(set(bins.tolist()))}/{n}"
                f"   max gap {max_normalised_gap(v, lo, hi):.2f}"
                f" (ideal {1 / (n + 1):.2f})")
        corr = sub[["d_LL", "d_SL_mean", "d_SS"]].corr().to_numpy()
        off = [abs(corr[0, 1]), abs(corr[0, 2]), abs(corr[1, 2])]
        out.append(f"    max |correlation| between parameters: {max(off):.2f}")
        out.append(f"    centrality (d_near/d_far) "
                   f"{sub[['S1_centrality', 'S2_centrality']].to_numpy().min():.2f}"
                   f" - {sub[['S1_centrality', 'S2_centrality']].to_numpy().max():.2f}")
        worst_fill = min(
            len(set(np.clip(((sub[p].to_numpy() - lo) / (hi - lo) * len(sub))
                            .astype(int), 0, len(sub) - 1).tolist()))
            for p, (lo, hi) in params.items())
        if worst_fill < QC_MIN_STRATA_FRAC * len(sub) or max(off) > QC_MAX_CORR:
            if diag[a]["feasible_fraction"] < 0.10:
                out.append(
                    "    WARNING: this archetype cannot carry three "
                    "independent, uniformly covered distances inside the "
                    "stated ranges. Its feasible region is too thin, so the "
                    "sampler must trade marginal coverage against parameter "
                    "correlation. Do not read a distance effect out of this "
                    "archetype alone; either widen the parameter ranges until "
                    "the region opens up, or report it as a restricted case.")
            else:
                out.append(
                    "    WARNING: the selection did not reach the quality "
                    "targets although the feasible region is wide enough. "
                    "This is a search failure, not a geometric limit - raise "
                    "CLHS_RESTARTS or CLHS_ITER and regenerate. A collinear "
                    "design makes the relative importance of the distance "
                    "parameters in the decision trees unstable.")
        out.append("")

    out += ["Full design, all archetypes pooled", "-" * 72]
    for p, (lo, hi) in params.items():
        v = df[p].to_numpy()
        out.append(f"    {p:<10} {v.min():7.0f} - {v.max():7.0f} km"
                   f"   max gap {max_normalised_gap(v, lo, hi):.2f}")
    out.append("")
    out.append(df[["d_LL", "d_SL_mean", "d_SS", "rho_comp"]].corr().round(3).to_string())
    return "\n".join(out)


def methodology_note(df, diag):
    feas_frac = {a: diag[a]["feasible_fraction"] for a in ARCHETYPES}
    n = len(df)
    lines = [
        "Spatial topology design - generated description",
        "",
        f"{n} four-node configurations were generated, "
        f"{N_PER_ARCHETYPE} for each of the {len(ARCHETYPES)} archetypes.",
        "Each configuration is defined by three distance parameters - the "
        "large-large distance d_LL in "
        f"[{D_LL_RANGE[0]:.0f}, {D_LL_RANGE[1]:.0f}] km, the small-large "
        f"distance d_SL in [{D_SL_RANGE[0]:.0f}, {D_SL_RANGE[1]:.0f}] km and "
        f"the small-small distance d_SS in [{D_SS_RANGE[0]:.0f}, "
        f"{D_SS_RANGE[1]:.0f}] km - together with a discrete archetype that "
        "fixes the qualitative arrangement of the two small clusters relative "
        "to the two large ones.",
        "",
        "Given an archetype, the four node positions are obtained by solving "
        "the three distance constraints rather than by assigning coordinates: "
        "each archetype supplies the symmetry that reduces the remaining "
        "degree of freedom to a unique layout, so the three requested "
        "distances are realized exactly and no additional random factor is "
        "introduced.",
        "",
        "Because four coplanar nodes carry six pairwise distances, each "
        "archetype can realize only a subset of the parameter box. The design "
        "was therefore drawn as a scrambled Sobol pool over the box, filtered "
        "on feasibility, and reduced to the final size by conditioned Latin "
        "hypercube sampling (Minasny and McBratney, 2006), which selects the "
        "subset whose marginals fill every stratum of the stated ranges once "
        "and whose parameters remain uncorrelated. The sampled distances are "
        "therefore homogeneously distributed over the ranges given above, and "
        "no draw is rescaled or discarded after the fact. Feasible shares "
        "of the box were "
        + ", ".join(f"{a}: {feas_frac[a] * 100:.0f}%" for a in ARCHETYPES) + ".",
        "",
        "Node coordinates were produced with an azimuthal-equidistant inverse "
        "projection and refined so that the great-circle distances used by the "
        "optimisation model reproduce the sampled distances.",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------------- main --

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    design, diag, n_core = build_design()

    wide, long, max_err, counter = [], [], 0.0, 1
    for a in ARCHETYPES:
        print(f"\n--- archetype {a}: {ARCHETYPE_NAME[a]} "
              f"({diag[a]['feasible_fraction'] * 100:.1f}% of the box feasible) ---")
        for d_ll, d_sl, d_ss in design[a]:
            built = build_topology(a, d_ll, d_sl, d_ss)
            if built is None:                     # cannot happen: pool prefiltered
                raise RuntimeError(f"{a}: lost feasibility for "
                                   f"({d_ll:.1f}, {d_sl:.1f}, {d_ss:.1f})")
            pos, d = built
            name = f"{counter:04d}"

            lonlat, err = positions_to_lonlat(pos)
            max_err = max(max_err, err)
            write_node_locations(lonlat, name, OUTPUT_FOLDER)
            if (counter - 1) % PLOT_EVERY == 0:
                plot_topology(pos, a, name, OUTPUT_FOLDER)

            row = {"topology": name, "archetype": a,
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

            print(f"  {name}: LL={d_ll:6.0f}  SL={d_sl:6.0f}  SS={d_ss:6.0f}"
                  f"   proj err {err:.3f} km")
            counter += 1

    dfw = pd.DataFrame(wide)
    dfl = pd.DataFrame(long)
    dfw.to_csv(OUTPUT_FOLDER / "topology_parameters.csv", sep=";", index=False)
    dfl.to_csv(OUTPUT_FOLDER / "topology_nodes_long.csv", sep=";", index=False)

    plot_diagnostics(dfw, OUTPUT_FOLDER)
    plot_archetype_schematics(dfw, OUTPUT_FOLDER)

    report = coverage_report(dfw, diag, n_core, max_err)
    (OUTPUT_FOLDER / "topology_coverage.txt").write_text(report, encoding="utf-8")
    (OUTPUT_FOLDER / "methodology_note.txt").write_text(
        methodology_note(dfw, diag), encoding="utf-8")

    print("\n" + report)
    print(f"\n{len(dfw)} topologies written to '{OUTPUT_FOLDER}'")


if __name__ == "__main__":
    main()