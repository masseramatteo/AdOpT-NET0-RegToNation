"""
Scenario design for the four-node campaign: constrained Latin hypercube of the
techno-economic parameters, paired with the per-economy topologies of
preprocess/generated_topology_v5 (design_manifest.csv).

Why a separate module
---------------------
The parallel runner differs between the local branch and working_snellius
(SLURM partitioning). Keeping the design logic here lets both import the same
file, so the campaign definition cannot drift between the two. Run it directly
(python sampling_design.py) to build the design and its QC report without
creating any model.

Design
------
1. Latin hypercube over the sampled parameters with scipy's centred-discrepancy
   optimisation (qmc.LatinHypercube, optimization="random-cd"): one point per
   stratum in every dimension, pairwise correlations driven to ~0.01.
2. Values from unit coordinates by linear interpolation between the anchor
   points of the grid (three equally spaced anchors -> uniform marginal).
3. Feasibility: annual hydrogen deliverable from grid electrolysis plus the
   import cap must cover demand,
        g + import_availability_ratio >= margin,
        g = (2*cap_large + 2*cap_small) / (D*1e6 / (8760*eta)).
   Infeasible points are repaired by coordinate swaps with feasible points
   (Petelet, Iooss, Asserin, Loredo 2010, "Latin hypercube sampling with
   inequality constraints", AStA Adv Stat Anal 94(4):325-339): the Latin
   property of every marginal is preserved exactly, only the joint distribution
   leaves the infeasible corner. Unlike the original cLHS (pairwise monotone
   constraints, compatibility-matrix permutation) this is a greedy swap repair,
   because the constraint is nonlinear in four variables.
   repair="legacy" (default) is the v5 / flat-LHS procedure, kept bit-identical.
   repair="balanced" randomises which constrained coordinate is swapped and then
   runs a feasibility-preserving within-column swap search (CD + correlation
   objective) to recover the space filling the repair destroyed.
4. Each economy is paired with the topologies listed for it in the manifest
   (8 per economy in v5). Spatial and economic parameters are independent by
   construction; the QC report measures it.

Outputs written into the results folder: sampling_qc.txt, economies.csv,
run_table.csv.
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

ETA_ELECTROLYZER = 0.655
CONSTRAINED_KEYS = ("total_demand_TWh", "electricity_availability_large",
                    "import_availability_ratio", "electricity_availability_small")
SPATIAL_COLS = ("d_LL", "d_SL_mean", "d_SS")
DEFAULT_TOPOLOGY_FOLDER = Path(__file__).resolve().parent / "preprocess" / "generated_topology_v5"


# ------------------------------------------------------------- feasibility --

def grid_adequacy(p):
    """g: annual grid-electrolysis H2 potential / annual demand (eta cancels)."""
    p_req = p["total_demand_TWh"] * 1e6 / 8760.0 / ETA_ELECTROLYZER
    return (2.0 * p["electricity_availability_large"]
            + 2.0 * p["electricity_availability_small"]) / p_req


def small_self_sufficiency(p):
    """k: small-cluster grid connection / the electricity need of one small node."""
    p_req = p["total_demand_TWh"] * 1e6 / 8760.0 / ETA_ELECTROLYZER
    sigma_d = 1.0 / (1.0 + p["demand_level_ratio"])
    return p["electricity_availability_small"] / (sigma_d * p_req / 2.0)


def is_feasible(p, margin):
    return grid_adequacy(p) + p["import_availability_ratio"] >= margin


# ----------------------------------------------------------------- mapping --

def unit_to_value(u, anchors):
    """Linear interpolation between sorted anchor values, u in [0, 1]."""
    values = sorted(anchors)
    x = u * (len(values) - 1)
    lo = min(int(np.floor(x)), len(values) - 2)
    w = x - lo
    return float(values[lo] * (1.0 - w) + values[lo + 1] * w)


# ------------------------------------------------------------ constrained --

REPAIR_MODES = ("legacy", "balanced")
GAMMA_WARN = 0.30          # infeasible share above which points pile on the frontier
PETELET_RHO_PER_GAMMA = 2.778   # Petelet et al. 2010 eq. 4, pairwise X1<X2, uniform marginals

# balanced-mode post-repair optimisation (feasibility-preserving column swaps)
POLISH_MAX_ITER = 30000
POLISH_NO_CHANGE = 3000    # stop after this many consecutive rejected swaps
POLISH_CORR_TARGET = 0.05  # scale of the correlation term in the objective
POLISH_W_CORR = 1.0


def centred_discrepancy(U):
    """Squared centred L2 discrepancy; identical to qmc.discrepancy(U, method='CD')."""
    n, d = U.shape
    z = np.abs(U - 0.5)
    t1 = (13.0 / 12.0) ** d
    t2 = np.prod(1.0 + 0.5 * z - 0.5 * z ** 2, axis=1).sum() * 2.0 / n
    acc = 0.0
    for i in range(n):                      # row-wise keeps memory O(n d)
        acc += _cd_pair_row(U, z, i).sum()
    return t1 - t2 + acc / n ** 2


def _cd_pair_row(U, z, i):
    """Pair terms of the CD sum between point i and every point."""
    return np.prod(1.0 + 0.5 * z[i] + 0.5 * z - 0.5 * np.abs(U[i] - U), axis=1)


def _cd_swap_delta(U, z, i1, i2, c):
    """Change of the squared CD when U[i1, c] and U[i2, c] are exchanged. O(n d)."""
    n = U.shape[0]

    def rows_contrib(UU, zz):
        r1, r2 = _cd_pair_row(UU, zz, i1), _cd_pair_row(UU, zz, i2)
        pair = 2.0 * (r1.sum() + r2.sum()) - r1[i1] - r2[i2] - 2.0 * r1[i2]
        single = (np.prod(1.0 + 0.5 * zz[i1] - 0.5 * zz[i1] ** 2)
                  + np.prod(1.0 + 0.5 * zz[i2] - 0.5 * zz[i2] ** 2))
        return pair / n ** 2 - 2.0 * single / n

    old = rows_contrib(U, z)
    U[i1, c], U[i2, c] = U[i2, c], U[i1, c]
    z[i1, c], z[i2, c] = z[i2, c], z[i1, c]
    new = rows_contrib(U, z)
    U[i1, c], U[i2, c] = U[i2, c], U[i1, c]          # restore; caller decides
    z[i1, c], z[i2, c] = z[i2, c], z[i1, c]
    return new - old


def _free_pair_mask(keys):
    constrained = {k for k in CONSTRAINED_KEYS if k in keys}
    d = len(keys)
    mask = np.zeros((d, d), dtype=bool)
    for a in range(d):
        for b in range(a + 1, d):
            if not (keys[a] in constrained and keys[b] in constrained):
                mask[a, b] = True
    return mask


def _polish(U, points, to_params, keys, margin, rng):
    """Greedy feasibility-preserving swap search (balanced mode).

    Exchanges two values inside one column (stays Latin), accepted only if
    every point stays feasible and J = CD/CD0 + w * maxcorr_free/target drops.
    Correlations are computed on U; with equally spaced anchors the value map
    is affine, so they equal the correlations of the parameter values.
    """
    n, d = U.shape
    constrained_cols = {keys.index(k) for k in CONSTRAINED_KEYS if k in keys}
    free = _free_pair_mask(keys)
    z = np.abs(U - 0.5)
    Xc = U - U.mean(axis=0)                 # column means invariant under swaps
    var = (Xc ** 2).sum(axis=0)
    C = Xc.T @ Xc                           # cross products, O(d) update per swap
    norm = np.sqrt(np.outer(var, var))

    def max_free_corr(CC):
        return float(np.abs(CC / norm)[free].max())

    cd = centred_discrepancy(U)
    cd0, corr0 = cd, max_free_corr(C)

    def objective(cd_, corr_):
        return cd_ / cd0 + POLISH_W_CORR * corr_ / POLISH_CORR_TARGET

    J = objective(cd, corr0)
    accepted, no_change, it = 0, 0, 0
    for it in range(1, POLISH_MAX_ITER + 1):
        if no_change >= POLISH_NO_CHANGE:
            break
        c = int(rng.integers(d))
        i1, i2 = (int(v) for v in rng.choice(n, size=2, replace=False))
        # correlation after the swap
        diff = Xc[i2, c] - Xc[i1, c]
        dC = diff * (Xc[i1] - Xc[i2])
        dC[c] = 0.0
        C_new = C.copy()
        C_new[c, :] += dC
        C_new[:, c] += dC
        corr_new = max_free_corr(C_new)
        cd_new = cd + _cd_swap_delta(U, z, i1, i2, c)
        J_new = objective(cd_new, corr_new)
        if J_new >= J:
            no_change += 1
            continue
        if c in constrained_cols:
            U[i1, c], U[i2, c] = U[i2, c], U[i1, c]
            p1, p2 = to_params(U[i1]), to_params(U[i2])
            if not (is_feasible(p1, margin) and is_feasible(p2, margin)):
                U[i1, c], U[i2, c] = U[i2, c], U[i1, c]
                no_change += 1
                continue
            points[i1], points[i2] = p1, p2
        else:
            U[i1, c], U[i2, c] = U[i2, c], U[i1, c]
            points[i1], points[i2] = to_params(U[i1]), to_params(U[i2])
        z[i1, c], z[i2, c] = z[i2, c], z[i1, c]
        Xc[i1, c], Xc[i2, c] = Xc[i2, c], Xc[i1, c]
        C, cd, J = C_new, cd_new, J_new
        accepted += 1
        no_change = 0
    return {"polish_iterations": it, "polish_accepted": accepted,
            "cd_after_repair": float(cd0), "cd_after_polish": float(cd),
            "maxcorr_free_after_repair": corr0, "maxcorr_free_after_polish": max_free_corr(C)}


def constrained_lhs_economies(param_grid, n, seed=42, margin=1.02, max_rounds=50,
                              repair="legacy"):
    """Return (list of parameter dicts, unit-cube array, qc dict).

    param_grid: {name: [anchor values]} for the numeric sampled parameters only.
    repair:
      "legacy"   - the v5 / flat-LHS procedure, kept bit-identical so those
                   designs can be regenerated: constrained coordinates tried
                   in the fixed CONSTRAINED_KEYS order (demand first).
      "balanced" - coordinate tried in random order per infeasible point (the
                   induced correlation is shared by all constrained variables),
                   then a feasibility-preserving swap search that recovers the
                   space filling and orthogonality lost in the repair.
    """
    if repair not in REPAIR_MODES:
        raise ValueError(f"repair must be one of {REPAIR_MODES}, got {repair!r}")
    keys = list(param_grid.keys())
    d = len(keys)
    U = qmc.LatinHypercube(d=d, optimization="random-cd", seed=seed).random(n)
    rng = np.random.default_rng(seed)
    cd_initial = float(qmc.discrepancy(U, method="CD"))

    def to_params(row):
        return {k: unit_to_value(row[i], param_grid[k]) for i, k in enumerate(keys)}

    points = [to_params(U[i]) for i in range(n)]
    n_infeasible_initial = sum(not is_feasible(p, margin) for p in points)
    gamma = n_infeasible_initial / n
    if gamma > GAMMA_WARN:
        warnings.warn(f"{gamma:.0%} of the LHS points are infeasible (> {GAMMA_WARN:.0%}): "
                      "the repair will pile points on the feasibility frontier "
                      "(Petelet et al. 2010). Consider widening the ranges or "
                      f"lowering margin={margin}.", stacklevel=2)

    # Petelet-style repair: exchange one coordinate between an infeasible point
    # and a feasible one so that both end up feasible. Exchanging the unit
    # coordinate exchanges the stratum, so every marginal stays Latin.
    constrained_present = [k for k in CONSTRAINED_KEYS if k in param_grid]
    swaps = 0
    swaps_per_key = {k: 0 for k in constrained_present}
    for _ in range(max_rounds):
        bad = [i for i in range(n) if not is_feasible(points[i], margin)]
        if not bad:
            break
        good = [i for i in range(n) if is_feasible(points[i], margin)]
        for i in bad:
            fixed = False
            if repair == "legacy":
                order = constrained_present
            else:
                order = [constrained_present[t] for t in rng.permutation(len(constrained_present))]
            for key in order:
                c = keys.index(key)
                for j in rng.permutation(good):
                    U[i, c], U[j, c] = U[j, c], U[i, c]
                    pi, pj = to_params(U[i]), to_params(U[j])
                    if is_feasible(pi, margin) and is_feasible(pj, margin):
                        points[i], points[j] = pi, pj
                        swaps += 1
                        swaps_per_key[key] += 1
                        fixed = True
                        break
                    U[i, c], U[j, c] = U[j, c], U[i, c]      # undo
                if fixed:
                    break
    bad = [i for i in range(n) if not is_feasible(points[i], margin)]
    if bad:
        raise RuntimeError(f"{len(bad)} economies still infeasible after repair; "
                           f"widen the ranges or lower margin={margin}")

    qc = {"n": n, "d": d, "keys": keys, "margin": margin, "seed": seed, "repair": repair,
          "infeasible_before_repair": n_infeasible_initial, "gamma": gamma,
          "swaps": swaps, "swaps_per_key": swaps_per_key, "cd_initial": cd_initial}
    if repair == "balanced":
        qc.update(_polish(U, points, to_params, keys, margin, rng))
        if any(not is_feasible(p, margin) for p in points):     # defensive
            raise RuntimeError("polish produced an infeasible point")
    return points, U, qc


# ------------------------------------------------------------- topologies --

def load_manifest(topology_folder=DEFAULT_TOPOLOGY_FOLDER):
    folder = Path(topology_folder)
    manifest = pd.read_csv(folder / "design_manifest.csv", sep=";", dtype=str)
    manifest["economy"] = manifest["economy"].astype(int)
    topo = pd.read_csv(folder / "topology_parameters.csv", sep=";",
                       dtype={"topology": str}).set_index("topology")
    return manifest, topo


def build_run_table(economies, manifest, topo, fixed_combo, n_economies_to_run=None):
    """One run per (economy, slot), economy-major. Returns list of parameter dicts."""
    slots = [c for c in manifest.columns if c != "economy"]
    n_run = len(economies) if n_economies_to_run is None else min(n_economies_to_run, len(economies))
    rows = []
    for i in range(n_run):
        econ_id = i + 1
        m = manifest.loc[manifest.economy == econ_id]
        if len(m) != 1:
            raise RuntimeError(f"economy {econ_id} not found exactly once in design_manifest.csv")
        m = m.iloc[0]
        for slot in slots:
            name = m[slot]
            p = {"scenario": name, "economy": econ_id, "slot": slot,
                 "archetype": str(topo.at[name, "archetype"])}
            p.update(fixed_combo)
            p.update(economies[i])
            rows.append(p)
    return rows


# ---------------------------------------------------------------------- qc --

def qc_report(economies, U, qc, run_rows, topo, corr_limit=None):
    keys, n = qc["keys"], qc["n"]
    if corr_limit is None:
        # Scale the orthogonality limit with the design size instead of fixing it.
        # The null SE of a Pearson r is 1/sqrt(n-3), and this test takes the MAX over
        # ~60 pairs, whose null expectation sits near 3 SE. A hard 0.05 was calibrated
        # at n=750, where it is 1.37 SE; at n=500 it drops to 1.11 SE and starts
        # failing on noise. It did, on 2026-09-10: the 500-economy design was rejected
        # for a single pair at 0.052 (1.16 SE, p ~ 0.25) while the feasibility repair
        # induces correlations up to 0.192 on constrained pairs that are accepted by
        # design - failing on 0.052 while accepting 0.192 is not a coherent criterion.
        # random-cd suppresses real correlations far below these limits, so the check
        # still catches genuine defects.
        corr_limit = max(0.05, 3.0 / np.sqrt(n - 3))
    lines = ["Sampling QC - constrained Latin hypercube + paired topologies", "=" * 72,
             f"economies {n}, dimensions {len(keys)}, seed {qc['seed']}, margin {qc['margin']}",
             f"repair mode {qc.get('repair', 'legacy')}",
             f"infeasible before repair {qc['infeasible_before_repair']} "
             f"({qc['infeasible_before_repair'] / n * 100:.1f}%), coordinate swaps {qc['swaps']}",
             "    swaps per column: " + ", ".join(f"{k} {v}" for k, v in qc.get("swaps_per_key", {}).items()),
             f"runs in table {len(run_rows)} "
             f"({len({r['economy'] for r in run_rows})} economies x "
             f"{len({r['slot'] for r in run_rows})} slots)", ""]
    failures = []

    lines += ["Latin property (points per stratum, ideal 1/1)", "-" * 72]
    for c, k in enumerate(keys):
        bins = np.clip((U[:, c] * n).astype(int), 0, n - 1)
        counts = np.bincount(bins, minlength=n)
        ok = counts.min() == 1 and counts.max() == 1
        lines.append(f"    {k:<34} min {counts.min()} max {counts.max()}  {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"Latin property violated for {k}")

    g = np.array([grid_adequacy(p) for p in economies])
    imp = np.array([p["import_availability_ratio"] for p in economies])
    k_small = np.array([small_self_sufficiency(p) for p in economies])
    feas_ok = bool(np.all(g + imp >= qc["margin"] - 1e-9))
    lines += ["", "Feasibility  g + import >= margin", "-" * 72,
              f"    all feasible: {'OK' if feas_ok else 'FAIL'}",
              f"    g  range {g.min():.2f} - {g.max():.2f}, median {np.median(g):.2f}, "
              f"share g < 1: {(g < 1).mean() * 100:.1f}%",
              f"    k  range {k_small.min():.2f} - {k_small.max():.2f}, median {np.median(k_small):.2f}, "
              f"share k < 1: {(k_small < 1).mean() * 100:.1f}%"]
    if not feas_ok:
        failures.append("infeasible economies present")

    df = pd.DataFrame(economies)
    lines += ["", "Coverage of the stated ranges", "-" * 72]
    for k in keys:
        lines.append(f"    {k:<34} {df[k].min():10.4g} - {df[k].max():10.4g}")

    corr = df[keys].corr().to_numpy()
    iu = np.triu_indices(len(keys), k=1)
    off = np.abs(corr[iu])
    constrained = [k for k in CONSTRAINED_KEYS if k in keys]
    free_pairs = [w for w in range(len(off))
                  if not (keys[iu[0][w]] in constrained and keys[iu[1][w]] in constrained)]
    lines += ["", f"Orthogonality, economic parameters (limit {corr_limit:.3f} = 3 SE at n={n})", "-" * 72,
              f"    max |corr| over {len(free_pairs)} unconstrained pairs: "
              f"{off[free_pairs].max():.3f}  {'OK' if off[free_pairs].max() < corr_limit else 'FAIL'}"]
    for w in np.argsort(off)[::-1][:5]:
        lines.append(f"        {keys[iu[0][w]]:<30} x {keys[iu[1][w]]:<30} {corr[iu[0][w], iu[1][w]]:+.3f}")
    lines.append("    constrained pairs (correlation induced by the feasible region, reported):")
    for a, b in itertools.combinations(constrained, 2):
        lines.append(f"        {a:<30} x {b:<30} {df[a].corr(df[b]):+.3f}")
    gamma = qc.get("gamma", qc["infeasible_before_repair"] / n)
    lines.append(f"    constraint intensity gamma (infeasible share) {gamma:.3f}; "
                 f"Petelet eq. 4 analogy rho ~ {PETELET_RHO_PER_GAMMA} gamma = "
                 f"{PETELET_RHO_PER_GAMMA * gamma:.3f} (pairwise-constraint relation, indicative only)")
    if off[free_pairs].max() >= corr_limit:
        failures.append("economic parameters correlated above limit")

    cd = qmc.discrepancy(U, method="CD")
    cd_plain = qmc.discrepancy(qmc.LatinHypercube(d=len(keys), seed=qc["seed"]).random(n), method="CD")
    lines += ["", "Space filling", "-" * 72,
              f"    centred L2 discrepancy {cd:.5f}  (plain random LHS, same seed: {cd_plain:.5f})"]
    if "cd_initial" in qc:
        stages = f"    CD stages: optimised LHS {qc['cd_initial']:.5f}"
        if "cd_after_repair" in qc:
            stages += (f" -> after repair {qc['cd_after_repair']:.5f}"
                       f" -> after polish {qc['cd_after_polish']:.5f}")
        lines.append(stages)
    if "polish_accepted" in qc:
        lines.append(f"    polish: {qc['polish_accepted']} swaps accepted in {qc['polish_iterations']} "
                     f"iterations; max |corr| free pairs {qc['maxcorr_free_after_repair']:.3f} -> "
                     f"{qc['maxcorr_free_after_polish']:.3f}")

    rt = pd.DataFrame(run_rows)
    sp = topo.loc[rt["scenario"], list(SPATIAL_COLS)].reset_index(drop=True)
    for a in sorted(rt["archetype"].unique()):
        sp[f"arch_{a}"] = (rt["archetype"] == a).astype(int).to_numpy()
    cross = pd.concat([rt[keys].reset_index(drop=True), sp], axis=1).corr().loc[keys, sp.columns]
    cmax = float(np.nanmax(np.abs(cross.to_numpy())))
    lines += ["", f"Spatial vs economic independence over {len(rt)} runs (limit {corr_limit})", "-" * 72,
              f"    max |corr(spatial, economic)|: {cmax:.3f}  {'OK' if cmax < corr_limit else 'FAIL'}"]
    if cmax >= corr_limit:
        failures.append("spatial parameters correlated with economic parameters")
    lines.append("    spatial coverage over the runs:")
    for c in SPATIAL_COLS:
        lines.append(f"        {c:<12} {sp[c].min():7.0f} - {sp[c].max():7.0f} km, "
                     f"{sp[c].round(6).nunique()} distinct values")
    lines.append(f"    archetype counts: {rt['archetype'].value_counts().sort_index().to_dict()}")

    lines += ["", "RESULT: " + ("PASS" if not failures else "FAIL - " + "; ".join(failures))]
    return "\n".join(lines), failures


LIST_VALUED = ("networks_new", "networks_existing", "small_cluster_new_technologies",
               "big_cluster_new_technologies", "small_cluster_existing_technologies",
               "big_cluster_existing_technologies")


def write_design(results_folder, economies, run_rows, report):
    folder = Path(results_folder)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(economies).assign(economy=lambda d: range(1, len(d) + 1)).to_csv(
        folder / "economies.csv", index=False)
    pd.DataFrame(run_rows).drop(columns=[c for c in LIST_VALUED if c in run_rows[0]]).to_csv(
        folder / "run_table.csv", index=False)
    (folder / "sampling_qc.txt").write_text(report, encoding="utf-8")


def build_campaign(param_grid, fixed_combo, results_folder,
                   topology_folder=DEFAULT_TOPOLOGY_FOLDER, n_economies=750,
                   n_economies_to_run=None, seed=42, margin=1.02, abort_on_fail=True,
                   repair="legacy"):
    """Economies -> pairing -> QC -> files. Returns the list of run parameter dicts.

    n_economies_to_run < n_economies gives a pilot that is a prefix of the full
    design (same economies, same topologies), so pilot runs are never wasted.
    repair: see constrained_lhs_economies ("legacy" reproduces v5).
    """
    economies, U, qc = constrained_lhs_economies(param_grid, n_economies, seed, margin,
                                                 repair=repair)
    manifest, topo = load_manifest(topology_folder)
    if len(manifest) < n_economies:
        raise RuntimeError(f"design_manifest.csv has {len(manifest)} economies, "
                           f"{n_economies} requested")
    # QC and design files always describe the FULL design; a pilot is then the
    # prefix of it (checking independence on 2 economies would be meaningless).
    run_rows_full = build_run_table(economies, manifest, topo, fixed_combo)
    report, failures = qc_report(economies, U, qc, run_rows_full, topo)
    if n_economies_to_run is not None and n_economies_to_run < n_economies:
        run_rows = [r for r in run_rows_full if r["economy"] <= n_economies_to_run]
        report += (f"\n\nPILOT: running economies 1-{n_economies_to_run} "
                   f"({len(run_rows)} of {len(run_rows_full)} runs); QC above refers to the full design")
    else:
        run_rows = run_rows_full
    write_design(results_folder, economies, run_rows_full, report)
    print(report)
    if failures and abort_on_fail:
        raise RuntimeError("sampling QC failed: " + "; ".join(failures))
    return run_rows


def load_campaign_definition():
    """Read the CAMPAIGN DEFINITION constants from the runner. Imports it when
    the environment has adopt_net0; otherwise parses the source with ast so the
    dry run also works in a bare python with only numpy/pandas/scipy."""
    names = ("PARAM_GRID_FOR_SAMPLING", "FIXED_PARAMS_GRID", "N_ECONOMIES",
             "FEASIBILITY_MARGIN", "LHS_SEED")
    optional = {"LHS_REPAIR": "legacy"}   # runners older than v6 have no LHS_REPAIR
    try:
        import FOUR_NODE_run_creation_and_gurobi_optimization_parallel as runner
        cfg = {n: getattr(runner, n) for n in names}
        cfg.update({n: getattr(runner, n, d) for n, d in optional.items()})
        return cfg
    except ImportError:
        import ast
        src = (Path(__file__).resolve().parent
               / "FOUR_NODE_run_creation_and_gurobi_optimization_parallel.py").read_text(encoding="utf-8")
        found = {}
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in set(names) | set(optional):
                found[node.targets[0].id] = ast.literal_eval(node.value)
        missing = set(names) - set(found)
        if missing:
            raise RuntimeError(f"could not read {missing} from the runner source")
        for n, d in optional.items():
            found.setdefault(n, d)
        return found


if __name__ == "__main__":
    # Dry run with the campaign grid: builds the design and the QC report only.
    cfg = load_campaign_definition()
    PARAM_GRID_FOR_SAMPLING, FIXED_PARAMS_GRID = cfg["PARAM_GRID_FOR_SAMPLING"], cfg["FIXED_PARAMS_GRID"]
    N_ECONOMIES, FEASIBILITY_MARGIN, LHS_SEED = cfg["N_ECONOMIES"], cfg["FEASIBILITY_MARGIN"], cfg["LHS_SEED"]
    fixed = {k: v[0] for k, v in FIXED_PARAMS_GRID.items()}
    out = Path(__file__).resolve().parent / "results" / "sampling_design_dry_run"
    rows = build_campaign(PARAM_GRID_FOR_SAMPLING, fixed, out, n_economies=N_ECONOMIES,
                          seed=LHS_SEED, margin=FEASIBILITY_MARGIN, abort_on_fail=False,
                          repair=cfg["LHS_REPAIR"])
    print(f"\n{len(rows)} runs; files in {out}")