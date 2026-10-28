"""
One flat-LHS run in its own process: create + solve with the v5 runner functions (unchanged).

Started by FOUR_NODE_run_flatlhs.py, one process per run, so the watchdog there can kill and
retry a run that stalls before the solver starts (rare single-threaded hang inside the tsam
k-means / OpenBLAS call of read_data, seen in 2 of 20 pilot runs, 2026-10-28).

Usage:  python flatlhs_worker.py <job.json>
Writes <results>/_status/<run_id>.json with status SUCCESS / CREATION_FAILED / SOLVE_FAILED.
"""

import os

# Forced, not setdefault: without the limit concurrent tsam/KMeans reads stall
# (debug_kmeans2: 13/16 reads > 300 s without it, 0/16 with it).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
os.environ["FOUR_NODE_TOPOLOGY_FOLDER"] = "generated_topology_flatlhs"

import json
import sys
import time
from pathlib import Path


def write_status(results_folder, run_id, status, **info):
    folder = Path(results_folder) / "_status"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{run_id}.json").write_text(
        json.dumps({"run_id": run_id, "status": status, **info}, indent=2, default=str),
        encoding="utf-8")


def main(job_file):
    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    run_id, params = job["run_id"], job["params"]
    results_folder, base_path = Path(job["results_folder"]), Path(job["base_path"])

    from FOUR_NODE_run_creation_and_gurobi_optimization_parallel import (
        create_single_model, solve_single_model)

    t0 = time.time()
    _, input_path, results_path, params, error = create_single_model(
        (run_id, params, results_folder, base_path))
    if error:
        write_status(results_folder, run_id, "CREATION_FAILED", error=error,
                     seconds=time.time() - t0)
        return 2
    _, result_info, error = solve_single_model((run_id, input_path, results_path, params))
    if error:
        write_status(results_folder, run_id, "SOLVE_FAILED", error=error, seconds=time.time() - t0)
        return 3
    write_status(results_folder, run_id, "SUCCESS", seconds=time.time() - t0, **(result_info or {}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
