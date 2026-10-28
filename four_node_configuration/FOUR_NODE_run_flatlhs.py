"""
Flat-LHS validation campaign (2026-10): 1500 runs, one per economy.

Reuses the v5 runner unchanged. Differences from the v5 campaign are only the inputs:
  * economies: 1500 from the SAME procedure as v5 (sampling_design.constrained_lhs_economies,
    Petelet swap repair, margin from the runner), new seed
  * topologies: preprocess/generated_topology_flatlhs (generate_node_topology_flatlhs.py),
    one per economy, archetype assigned among those that can build the triple
Fixed parameters, solver settings and model are imported from the v5 runner.

The QC orthogonality limit scales with n (max(0.05, 3 / sqrt(n - 3)), the rule on
working_snellius); the fixed 0.05 of this branch is too strict for 1500 independent runs.

Run:  python FOUR_NODE_run_flatlhs.py   (generate the topologies first)
"""

import os

# Must be set before workers spawn: utilities.load_scenario_nodes reads the topology folder
# from it, and the library thread pools must be limited before numpy/sklearn are imported.
os.environ["FOUR_NODE_TOPOLOGY_FOLDER"] = "generated_topology_flatlhs"
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from datetime import datetime
from pathlib import Path

import numpy as np

import sampling_design as sd
from FOUR_NODE_run_creation_and_gurobi_optimization_parallel import (
    FEASIBILITY_MARGIN, FIXED_PARAMS_GRID, PARAM_GRID_FOR_SAMPLING,
    ParallelCreationAndGurobiOptimizationRunner,
)

N_ECONOMIES = 1500
N_ECONOMIES_TO_RUN = None    # None = all; e.g. 20 for a pilot (prefix of the full design)
LHS_SEED = 2026              # v5 used 42 -> new economies
MAX_WORKERS = None           # set for the local server, e.g. 16
THREADS_PER_WORKER = None    # e.g. 3


def build_design(results_folder, topology_folder):
    """sampling_design.build_campaign with the n-scaled QC limit."""
    fixed = {k: v[0] for k, v in FIXED_PARAMS_GRID.items()}
    economies, U, qc = sd.constrained_lhs_economies(
        PARAM_GRID_FOR_SAMPLING, N_ECONOMIES, LHS_SEED, FEASIBILITY_MARGIN)
    manifest, topo = sd.load_manifest(topology_folder)
    if len(manifest) < N_ECONOMIES:
        raise RuntimeError(f"design_manifest.csv has {len(manifest)} economies, "
                           f"{N_ECONOMIES} requested")
    rows_full = sd.build_run_table(economies, manifest, topo, fixed)
    corr_limit = max(0.05, 3.0 / np.sqrt(N_ECONOMIES - 3))
    report, failures = sd.qc_report(economies, U, qc, rows_full, topo, corr_limit=corr_limit)
    rows = rows_full
    if N_ECONOMIES_TO_RUN is not None and N_ECONOMIES_TO_RUN < N_ECONOMIES:
        rows = [r for r in rows_full if r["economy"] <= N_ECONOMIES_TO_RUN]
        report += (f"\n\nPILOT: running economies 1-{N_ECONOMIES_TO_RUN} "
                   f"({len(rows)} of {len(rows_full)} runs); QC above refers to the full design")
    sd.write_design(results_folder, economies, rows_full, report)
    print(report)
    if failures:
        raise RuntimeError("sampling QC failed: " + "; ".join(failures))
    return rows


if __name__ == "__main__":
    base_path = Path(__file__).parent
    topology_folder = base_path / "preprocess" / os.environ["FOUR_NODE_TOPOLOGY_FOLDER"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = base_path / "results" / f"flatlhs_validation_{timestamp}"

    print(f"\n[SAMPLING] Flat-LHS validation: {N_ECONOMIES} economies x 1 topology, "
          f"seed {LHS_SEED}, topologies from {topology_folder.name}")
    combinations = build_design(results_folder, topology_folder)
    run_configs = [(f"parallel_run_{i:04d}", params) for i, params in enumerate(combinations, 1)]
    print(f"\n[OK] Total runs to execute: {len(run_configs)}")

    runner = ParallelCreationAndGurobiOptimizationRunner(
        base_path=base_path,
        max_workers=MAX_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
    )
    runner.run_parallel_optimization(run_configs=run_configs, results_base_folder=results_folder)

    print(f"\n[OK] Flat-LHS validation complete!")
    print(f"[RESULTS] Summary Excel: {results_folder / 'parallel_results_summary.xlsx'}")
