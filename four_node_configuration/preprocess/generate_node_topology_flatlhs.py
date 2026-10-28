"""
Four-node spatial topology design - flat-LHS validation campaign (2026-10).

Why this design
---------------
The v5 campaign is paired: 8 topologies per economy, archetypes in equal shares, each archetype
sampled on its own feasible region. This design is the flat alternative over the SAME box, used
to validate the v5 trees out of sample:

  * one topology per economy (1500 economies, economy i -> topology F{i:04d})
  * the three distances are a Latin hypercube over the v5 box
        d_LL [400, 1500] km, d_SL [100, 500] km, d_SS [100, 1000] km
  * a triple is kept if at least one archetype can build it (85% of the box); the archetype is
    then drawn uniformly among those that can. The archetype is a construction rule here, not a
    sampled axis: the expected shares are A 25.8 / B 25.8 / C 0.7 / D 47.7 %.

Rejection is only on triples no archetype can build, so the distances stay close to Latin; the
realised stratum fill is reported. The economies are NOT made here: FOUR_NODE_run_flatlhs.py
builds them with the v5 procedure (sampling_design.constrained_lhs_economies, swap repair).

Outputs (generated_topology_flatlhs/): NodeLocations_F####.csv, topology_parameters.csv (v5
columns + feasible archetypes), topology_nodes_long.csv, design_manifest.csv (economy;flat),
topology_coverage.txt, a few layout PNGs.

Run:  python generate_node_topology_flatlhs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

import generate_node_topology_pooled as v4
from generate_node_topology_pooled import (
    ARCHETYPE_NAME, NODES, SMALL_NODES, D_LL_RANGE, D_SL_RANGE, D_SS_RANGE,
    build_topology, positions_to_lonlat, write_node_locations, plot_topology,
)

# ------------------------------------------------------------------ config --

N_RUNS = 1500
SEED = 2027                     # v5 used 42; economies use 2026 (runner side)
OVERSAMPLE = 1.5                # 85% of the box is feasible
PREFIX = "F"
PLOT_EVERY = 250                # individual layout PNGs (0 = none)
OUTPUT_FOLDER = Path(__file__).resolve().parent / "generated_topology_flatlhs"
RANGES = {"d_LL": D_LL_RANGE, "d_SL": D_SL_RANGE, "d_SS": D_SS_RANGE}
N_STRATA_REPORT = 30


def feasible_labels(d_ll, d_sl, d_ss):
    """Archetype -> build_topology label(s) that can realise the triple. C has two leans."""
    out = {}
    for a, labels in (("A", ["A"]), ("B", ["B"]), ("D", ["D"]), ("C", ["C1", "C2"])):
        ok = [lab for lab in labels if build_topology(lab, d_ll, d_sl, d_ss) is not None]
        if ok:
            out[a] = ok
    return out


def sample(seed=SEED):
    """Latin hypercube over the box; keep triples buildable by >= 1 archetype."""
    n_cand = int(np.ceil(N_RUNS * OVERSAMPLE))
    U = qmc.LatinHypercube(d=3, seed=seed).random(n_cand)
    rng = np.random.default_rng(seed)
    kept, n_tried = [], 0
    for u in U:
        n_tried += 1
        t = [lo + u[j] * (hi - lo) for j, (lo, hi) in enumerate(RANGES.values())]
        feas = feasible_labels(*t)
        if not feas:
            continue
        a = sorted(feas)[rng.integers(len(feas))]
        label = feas[a][rng.integers(len(feas[a]))]
        kept.append((u, t, a, label, feas))
        if len(kept) == N_RUNS:
            return kept, n_tried, n_cand
    raise RuntimeError(f"only {len(kept)} feasible triples of {n_cand}; raise OVERSAMPLE")


def coverage_report(dfw, U_kept, n_tried, n_cand, max_err):
    n = len(dfw)
    lines = ["Flat-LHS topology design - coverage", "=" * 72,
             f"topologies {n}, seed {SEED}, candidates tried {n_tried} of {n_cand} "
             f"(acceptance {n / n_tried:.3f})", ""]
    lines += ["Archetype shares (expected A .258 / B .258 / C .007 / D .477)", "-" * 72]
    for a, c in dfw.archetype.value_counts().sort_index().items():
        lines.append(f"    {a}: {c:5d}  ({c / n:.3f})")
    lines += ["", "Archetypes able to build the kept triple", "-" * 72]
    for k, c in dfw.n_feasible_archetypes.value_counts().sort_index().items():
        lines.append(f"    {k}: {c:5d}  ({c / n:.3f})")
    lines += ["", f"Stratum fill of the sampled distances ({N_STRATA_REPORT} strata, ideal "
              f"{n / N_STRATA_REPORT:.0f} each; rejection removes the infeasible corners)", "-" * 72]
    for j, k in enumerate(RANGES):
        counts = np.bincount(np.clip((U_kept[:, j] * N_STRATA_REPORT).astype(int), 0,
                                     N_STRATA_REPORT - 1), minlength=N_STRATA_REPORT)
        lines.append(f"    {k:5s} min {counts.min():3d} max {counts.max():3d}  "
                     f"empty {int((counts == 0).sum())}")
    lines += ["", "Realised distances (km)", "-" * 72]
    for c in ("d_LL", "d_SL_mean", "d_SS"):
        lines.append(f"    {c:10s} {dfw[c].min():7.0f} - {dfw[c].max():7.0f}, mean {dfw[c].mean():6.0f}")
    lines += ["", f"Max lon/lat projection residual: {max_err:.4f} km"]
    return "\n".join(lines)


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"flat-LHS design: {N_RUNS} topologies, one per economy")
    kept, n_tried, n_cand = sample()

    wide, long, manifest, max_err = [], [], [], 0.0
    for i, (u, (d_ll, d_sl, d_ss), a, label, feas) in enumerate(kept, start=1):
        pos, d = build_topology(label, d_ll, d_sl, d_ss)
        name = f"{PREFIX}{i:04d}"
        lonlat, err = positions_to_lonlat(pos)
        max_err = max(max_err, err)
        write_node_locations(lonlat, name, OUTPUT_FOLDER)
        if PLOT_EVERY and (i - 1) % PLOT_EVERY == 0:
            plot_topology(pos, v4.public_archetype(label), name, OUTPUT_FOLDER)
        manifest.append({"economy": i, "flat": name})
        r = {"topology": name, "economy": i, "slot": "flat", "archetype": a,
             "label": label, "archetype_name": ARCHETYPE_NAME[a],
             "d_LL_target": d_ll, "d_SL_target": d_sl, "d_SS_target": d_ss,
             "proj_residual_km": err, **d}
        for node in NODES:
            r[f"{node}_x"], r[f"{node}_y"] = pos[node]
            r[f"{node}_lon"], r[f"{node}_lat"] = lonlat[node]
        r["n_feasible_archetypes"] = len(feas)
        r["feasible_archetypes"] = "".join(sorted(feas))
        wide.append(r)
        for k, sn in enumerate(SMALL_NODES, start=1):
            long.append({"topology": name, "economy": i, "slot": "flat", "archetype": a,
                         "small_cluster": sn, "d_nearest_large": d[f"S{k}_near"],
                         "d_other_large": d[f"S{k}_far"], "centrality": d[f"S{k}_centrality"],
                         "d_small_small": d["d_SS"], "d_large_large": d["d_LL"],
                         "rho_comp": d["rho_comp"]})
        if i % 250 == 0:
            print(f"  {i}/{N_RUNS} written")

    dfw = pd.DataFrame(wide)
    dfw.to_csv(OUTPUT_FOLDER / "topology_parameters.csv", sep=";", index=False)
    pd.DataFrame(long).to_csv(OUTPUT_FOLDER / "topology_nodes_long.csv", sep=";", index=False)
    pd.DataFrame(manifest).to_csv(OUTPUT_FOLDER / "design_manifest.csv", sep=";", index=False)
    report = coverage_report(dfw, np.array([k[0] for k in kept]), n_tried, n_cand, max_err)
    (OUTPUT_FOLDER / "topology_coverage.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n{len(dfw)} topologies written to '{OUTPUT_FOLDER}'")


if __name__ == "__main__":
    main()
