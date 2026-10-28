"""
Flat-LHS validation campaign (2026-10): 1500 runs, one per economy.

Reuses the v5 runner functions unchanged. Differences from the v5 campaign are only the inputs:
  * economies: 1500 from the SAME procedure as v5 (sampling_design.constrained_lhs_economies,
    Petelet swap repair, margin from the runner), new seed
  * topologies: preprocess/generated_topology_flatlhs (generate_node_topology_flatlhs.py),
    one per economy, archetype assigned among those that can build the triple
Fixed parameters, solver settings and model are imported from the v5 runner.

Execution: one process per run (flatlhs_worker.py: create + solve) under a watchdog instead of
the runner's ProcessPoolExecutor. A run that has not started the solver (no userData/<timestamp>
folder) STARTUP_TIMEOUT after launch is killed and retried. In the pilot 2 of 20 workers hung
single-threaded inside the tsam k-means (OpenBLAS) of read_data and a pool worker never frees up;
here a hang costs one retry. Solves themselves are never timed out by the watchdog.

Resume: set RESUME_FOLDER to an existing results folder; runs with a SUCCESS status are skipped.

The QC orthogonality limit scales with n (max(0.05, 3 / sqrt(n - 3)), the rule on
working_snellius); the fixed 0.05 of this branch is too strict for 1500 independent runs.

Run:  python FOUR_NODE_run_flatlhs.py   (generate the topologies first)
Monitor: <results>/progress.txt (rewritten every POLL seconds), <results>/watchdog.log
"""

import os

# Forced, not setdefault: without the limit concurrent tsam/KMeans reads stall
# (debug_kmeans2: 13/16 reads > 300 s without it, 0/16 with it). Workers inherit these.
os.environ["FOUR_NODE_TOPOLOGY_FOLDER"] = "generated_topology_flatlhs"
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sampling_design as sd

N_ECONOMIES = 1500
N_ECONOMIES_TO_RUN = None    # None = all; e.g. 20 for a pilot (prefix of the full design)
LHS_SEED = 2026              # v5 used 42 -> new economies
MAX_WORKERS = 22             # concurrent runs (48-core server: 22 x 2 = 44 cores)
THREADS_PER_WORKER = 2       # Gurobi Threads per run
STARTUP_TIMEOUT = 20 * 60    # s from launch to solver start (normal: 2-5 min)
MAX_ATTEMPTS = 3             # per run, counting the first
POLL = 30                    # s between watchdog checks
RESUME_FOLDER = None         # e.g. r"...\results\flatlhs_validation_20261028_200000"

# Read from the runner source without importing it (importing pulls in adopt_net0 here).
_CFG = sd.load_campaign_definition()
PARAM_GRID_FOR_SAMPLING = _CFG["PARAM_GRID_FOR_SAMPLING"]
FIXED_PARAMS_GRID = _CFG["FIXED_PARAMS_GRID"]
FEASIBILITY_MARGIN = _CFG["FEASIBILITY_MARGIN"]


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


class Watchdog:
    def __init__(self, base_path, results_folder, run_configs):
        self.base = Path(base_path)
        self.out = Path(results_folder)
        self.worker = self.base / "flatlhs_worker.py"
        self.queue = list(run_configs)               # (run_id, params)
        self.attempts = {rid: 0 for rid, _ in run_configs}
        self.running = {}                            # run_id -> (Popen, t_start, log handle)
        self.final = {}                              # run_id -> status dict
        self.kills = 0
        self.t0 = time.time()
        for sub in ("_jobs", "_logs", "_status", "_killed"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = str(self.base.parent) + os.pathsep + self.env.get("PYTHONPATH", "")

    def log(self, msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        with open(self.out / "watchdog.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def status_of(self, run_id):
        f = self.out / "_status" / f"{run_id}.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def solver_started(self, run_id):
        ud = self.out / run_id / "userData"
        return ud.exists() and any(p.is_dir() for p in ud.iterdir())

    def launch(self, run_id, params):
        self.attempts[run_id] += 1
        stale = self.out / run_id
        if stale.exists():                           # left by a killed or interrupted attempt
            try:
                shutil.move(str(stale), str(self.out / "_killed" /
                                            f"{run_id}_stale{self.attempts[run_id]}_{int(time.time())}"))
            except Exception as e:
                self.log(f"  could not move stale {run_id} folder aside: {e}")
        params = dict(params, threads=THREADS_PER_WORKER)
        job = self.out / "_jobs" / f"{run_id}.json"
        job.write_text(json.dumps({"run_id": run_id, "params": params,
                                   "results_folder": str(self.out), "base_path": str(self.base)},
                                  default=str), encoding="utf-8")
        logf = open(self.out / "_logs" / f"{run_id}_a{self.attempts[run_id]}.log", "w",
                    encoding="utf-8")
        p = subprocess.Popen([sys.executable, str(self.worker), str(job)], cwd=str(self.base),
                             env=self.env, stdout=logf, stderr=subprocess.STDOUT)
        self.running[run_id] = (p, time.time(), logf)

    def kill_and_requeue(self, run_id, params, waited):
        p, _, logf = self.running.pop(run_id)
        p.kill()
        p.wait()
        logf.close()
        self.kills += 1
        dest = self.out / "_killed" / f"{run_id}_a{self.attempts[run_id]}"
        try:
            shutil.move(str(self.out / run_id), str(dest))
        except Exception as e:                       # leave it; create_single_model overwrites
            self.log(f"  could not move {run_id} folder aside: {e}")
        if self.attempts[run_id] < MAX_ATTEMPTS:
            self.log(f"KILL {run_id}: no solver start after {waited / 60:.0f} min "
                     f"(attempt {self.attempts[run_id]}) -> requeued")
            self.queue.insert(0, (run_id, params))
        else:
            self.log(f"KILL {run_id}: no solver start after {waited / 60:.0f} min, "
                     f"attempt {self.attempts[run_id]} of {MAX_ATTEMPTS} -> GIVEN UP")
            self.final[run_id] = {"run_id": run_id, "status": "STARTUP_HANG",
                                  "attempts": self.attempts[run_id]}

    def progress(self, params_of):
        n = len(self.attempts)
        st = [s["status"] for s in self.final.values()]
        el = (time.time() - self.t0) / 3600
        ok = st.count("SUCCESS")
        lines = [f"{datetime.now():%Y-%m-%d %H:%M:%S}  elapsed {el:.2f} h",
                 f"total {n} | SUCCESS {ok} | failed {len(st) - ok} | running {len(self.running)} "
                 f"| queued {len(self.queue)} | watchdog kills {self.kills}"]
        if ok and el > 0:
            rate = ok / el
            lines.append(f"rate {rate:.1f} runs/h -> remaining ~{(n - len(st)) / rate:.1f} h")
        waiting = [rid for rid in self.running if not self.solver_started(rid)]
        lines.append(f"running, solver not yet started: {len(waiting)} {sorted(waiting)[:10]}")
        (self.out / "progress.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self):
        params_of = dict(self.queue)
        # resume: keep earlier successes
        for rid in list(params_of):
            s = self.status_of(rid)
            if s and s.get("status") == "SUCCESS":
                self.final[rid] = s
                self.queue = [q for q in self.queue if q[0] != rid]
        self.log(f"start: {len(self.queue)} runs to do, {len(self.final)} already done, "
                 f"{MAX_WORKERS} workers x {THREADS_PER_WORKER} threads, "
                 f"startup timeout {STARTUP_TIMEOUT / 60:.0f} min")
        while self.queue or self.running:
            while self.queue and len(self.running) < MAX_WORKERS:
                rid, params = self.queue.pop(0)
                self.launch(rid, params)
            time.sleep(POLL)
            for rid in list(self.running):
                p, t_start, logf = self.running[rid]
                if p.poll() is not None:
                    logf.close()
                    self.running.pop(rid)
                    s = self.status_of(rid) or {"run_id": rid, "status": "WORKER_DIED",
                                                "returncode": p.returncode}
                    s["attempts"] = self.attempts[rid]
                    if s["status"] == "SUCCESS":
                        self.final[rid] = s
                    elif self.attempts[rid] < MAX_ATTEMPTS:
                        # transient failures seen on this server: import errors from the
                        # network drive, license tokens; retry
                        self.log(f"{s['status']} {rid} (rc {p.returncode}, attempt "
                                 f"{self.attempts[rid]}): {s.get('error', '')} -> requeued")
                        (self.out / "_status" / f"{rid}.json").unlink(missing_ok=True)
                        self.queue.insert(0, (rid, params_of[rid]))
                    else:
                        self.final[rid] = s
                        self.log(f"{s['status']} {rid} (rc {p.returncode}, attempt "
                                 f"{self.attempts[rid]} of {MAX_ATTEMPTS}): {s.get('error', '')} "
                                 f"-> GIVEN UP")
                elif not self.solver_started(rid) and time.time() - t_start > STARTUP_TIMEOUT:
                    self.kill_and_requeue(rid, params_of[rid], time.time() - t_start)
            self.progress(params_of)
        self.progress(params_of)
        df = pd.DataFrame(list(self.final.values()))
        df.to_csv(self.out / "parallel_results_summary.csv", index=False, sep=";")
        df.to_excel(self.out / "parallel_results_summary.xlsx", index=False)
        ok = int((df.status == "SUCCESS").sum())
        self.log(f"finished: {ok}/{len(df)} SUCCESS, watchdog kills {self.kills}")


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent
    topology_folder = base_path / "preprocess" / os.environ["FOUR_NODE_TOPOLOGY_FOLDER"]
    if RESUME_FOLDER:
        results_folder = Path(RESUME_FOLDER)
        print(f"[RESUME] {results_folder}")
    else:
        results_folder = base_path / "results" / f"flatlhs_validation_{datetime.now():%Y%m%d_%H%M%S}"
        print(f"\n[SAMPLING] Flat-LHS validation: {N_ECONOMIES} economies x 1 topology, "
              f"seed {LHS_SEED}, topologies from {topology_folder.name}")
    # the design is rebuilt in both cases (deterministic); run_table.csv drops list-valued params
    combinations = build_design(results_folder, topology_folder)
    run_configs = [(f"parallel_run_{i:04d}", params) for i, params in enumerate(combinations, 1)]
    print(f"\n[OK] Total runs: {len(run_configs)}")
    Watchdog(base_path, results_folder, run_configs).run()
