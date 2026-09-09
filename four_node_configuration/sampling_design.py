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
   leaves the infeasible corner.
4. Each economy is paired with the topologies listed for it in the manifest
   (8 per economy in v5). Spatial and economic parameters are independent by
   construction; the QC report measures it.

Outputs written into the results folder: sampling_qc.txt, economies.csv,
run_table.csv.
"""

from __future__ import annotations

import itertools
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

def constrained_lhs_economies(param_grid, n, seed=42, margin=1.02, max_rounds=50):
    """Return (list of parameter dicts, unit-cube array, qc dict).

    param_grid: {name: [anchor values]} for the numeric sampled parameters only.
    """
    keys = list(param_grid.keys())
    d = len(keys)
    U = qmc.LatinHypercube(d=d, optimization="random-cd", seed=seed).random(n)
    rng = np.random.default_rng(seed)

    def to_params(row):
        return {k: unit_to_value(row[i], param_grid[k]) for i, k in enumerate(keys)}

    points = [to_params(U[i]) for i in range(n)]
    n_infeasible_initial = sum(not is_feasible(p, margin) for p in points)

    # Petelet-style repair: exchange one coordinate between an infeasible point
    # and a feasible one so that both end up feasible. Exchanging the unit
    # coordinate exchanges the stratum, so every marginal stays Latin.
    swaps = 0
    for _ in range(max_rounds):
        bad = [i for i in range(n) if not is_feasible(points[i], margin)]
        if not bad:
            break
        good = [i for i in range(n) if is_feasible(points[i], margin)]
        for i in bad:
            fixed = False
            for key in CONSTRAINED_KEYS:
                if key not in param_grid:
                    continue
                c = keys.index(key)
                for j in rng.permutation(good):
                    U[i, c], U[j, c] = U[j, c], U[i, c]
                    pi, pj = to_params(U[i]), to_params(U[j])
                    if is_feasible(pi, margin) and is_feasible(pj, margin):
                        points[i], points[j] = pi, pj
                        swaps += 1
                        fixed = True
                        break
                    U[i, c], U[j, c] = U[j, c], U[i, c]      # undo
                if fixed:
                    break
    bad = [i for i in range(n) if not is_feasible(points[i], margin)]
    if bad:
        raise RuntimeError(f"{len(bad)} economies still infeasible after repair; "
                           f"widen the ranges or lower margin={margin}")

    qc = {"n": n, "d": d, "keys": keys, "margin": margin, "seed": seed,
          "infeasible_before_repair": n_infeasible_initial, "swaps": swaps}
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

def qc_report(economies, U, qc, run_rows, topo, corr_limit=0.05):
    keys, n = qc["keys"], qc["n"]
    lines = ["Sampling QC - constrained Latin hypercube + paired topologies", "=" * 72,
             f"economies {n}, dimensions {len(keys)}, seed {qc['seed']}, margin {qc['margin']}",
             f"infeasible before repair {qc['infeasible_before_repair']} "
             f"({qc['infeasible_before_repair'] / n * 100:.1f}%), coordinate swaps {qc['swaps']}",
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
    lines += ["", f"Orthogonality, economic parameters (limit {corr_limit})", "-" * 72,
              f"    max |corr| over {len(free_pairs)} unconstrained pairs: "
              f"{off[free_pairs].max():.3f}  {'OK' if off[free_pairs].max() < corr_limit else 'FAIL'}"]
    for w in np.argsort(off)[::-1][:5]:
        lines.append(f"        {keys[iu[0][w]]:<30} x {keys[iu[1][w]]:<30} {corr[iu[0][w], iu[1][w]]:+.3f}")
    lines.append("    constrained pairs (correlation induced by the feasible region, reported):")
    for a, b in itertools.combinations(constrained, 2):
        lines.append(f"        {a:<30} x {b:<30} {df[a].corr(df[b]):+.3f}")
    if off[free_pairs].max() >= corr_limit:
        failures.append("economic parameters correlated above limit")

    cd = qmc.discrepancy(U, method="CD")
    cd_plain = qmc.discrepancy(qmc.LatinHypercube(d=len(keys), seed=qc["seed"]).random(n), method="CD")
    lines += ["", "Space filling", "-" * 72,
              f"    centred L2 discrepancy {cd:.5f}  (plain random LHS, same seed: {cd_plain:.5f})"]

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
                   n_economies_to_run=None, seed=42, margin=1.02, abort_on_fail=True):
    """Economies -> pairing -> QC -> files. Returns the list of run parameter dicts.

    n_economies_to_run < n_economies gives a pilot that is a prefix of the full
    design (same economies, same topologies), so pilot runs are never wasted.
    """
    economies, U, qc = constrained_lhs_economies(param_grid, n_economies, seed, margin)
    manifest, topo = load_manifest(topology_folder)
    if len(manifest) < n_economies:
        raise RuntimeError(f"design_manifest.csv has {len(manifest)} economies, "
                           f"{n_economies} requested")
    run_rows = build_run_table(economies, manifest, topo, fixed_combo, n_economies_to_run)
    report, failures = qc_report(economies, U, qc, run_rows, topo)
    write_design(results_folder, economies, run_rows, report)
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
    try:
        import FOUR_NODE_run_creation_and_gurobi_optimization_parallel as runner
        return {n: getattr(runner, n) for n in names}
    except ImportError:
        import ast
        src = (Path(__file__).resolve().parent
               / "FOUR_NODE_run_creation_and_gurobi_optimization_parallel.py").read_text(encoding="utf-8")
        found = {}
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
                found[node.targets[0].id] = ast.literal_eval(node.value)
        missing = set(names) - set(found)
        if missing:
            raise RuntimeError(f"could not read {missing} from the runner source")
        return found


if __name__ == "__main__":
    # Dry run with the campaign grid: builds the design and the QC report only.
    cfg = load_campaign_definition()
    PARAM_GRID_FOR_SAMPLING, FIXED_PARAMS_GRID = cfg["PARAM_GRID_FOR_SAMPLING"], cfg["FIXED_PARAMS_GRID"]
    N_ECONOMIES, FEASIBILITY_MARGIN, LHS_SEED = cfg["N_ECONOMIES"], cfg["FEASIBILITY_MARGIN"], cfg["LHS_SEED"]
    fixed = {k: v[0] for k, v in FIXED_PARAMS_GRID.items()}
    out = Path(__file__).resolve().parent / "results" / "sampling_design_dry_run"
    rows = build_campaign(PARAM_GRID_FOR_SAMPLING, fixed, out, n_economies=N_ECONOMIES,
                          seed=LHS_SEED, margin=FEASIBILITY_MARGIN, abort_on_fail=False)
    print(f"\n{len(rows)} runs; files in {out}")