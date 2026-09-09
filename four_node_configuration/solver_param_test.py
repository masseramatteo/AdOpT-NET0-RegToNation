"""
Gurobi parameter test on already-created four-node models.

Re-solves the input_data of a few runs of an existing campaign folder under
several solver settings, in parallel, and compares solve time, root time,
nodes, gap and NPV against the campaign's own solve of the same run (baseline
"A" = the settings in Run_optimization_four_nodes._configure_model).

Usage (from four_node_configuration/):
    python solver_param_test.py --source results/parallel_creation_test_20260909_145257
    python solver_param_test.py --source ... --runs 0001,0009 --settings B,C --workers 8
    python solver_param_test.py --source ... --dry-run          # list the jobs only

Output: results/solver_param_test_<timestamp>/<setting>/<run>/{input_data,userData}
        results/solver_param_test_<timestamp>/summary.csv and report.txt
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import multiprocessing as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # utilities, runner
sys.path.insert(0, str(HERE.parent))     # adopt_net0 (repo root)

# ---------------------------------------------------------------- settings --
# Each entry patches ConfigModel.json["solveroptions"] on top of the campaign
# settings (A). Keys are the adopt template keys; "threads" is per job.
TIME_LIMIT_H = 2.5

SETTINGS = {
    # A = campaign settings (Method -1 -> concurrent simplex at the root,
    #     MIPFocus 2, Cuts 2, no NoRel). Baseline read from the source logs;
    #     re-solved only for runs listed in --rerun-baseline.
    "A": {},
    # B: barrier at the root relaxation (crossover auto)
    "B": {"method": 2},
    # C: B + default focus and cut aggressiveness
    "C": {"method": 2, "mipfocus": 0, "cuts": -1},
    # D: C + NoRel heuristic 300 s before the root LP (needs the pass-through
    #    in adopt_net0/utilities.get_gurobi_parameters)
    "D": {"method": 2, "mipfocus": 0, "cuts": -1, "NoRelHeurTime": 300},
    # F: B with 6 Gurobi threads instead of 3 (barrier parallelises)
    "F": {"method": 2, "threads": 6},
}
DEFAULT_RUNS = ["0001", "0009", "0025", "0033"]
DEFAULT_SETTINGS = ["B", "C", "D", "F"]
DEFAULT_F_RUNS = ["0009", "0033"]          # threads test only on the slow ones
DEFAULT_RERUN_BASELINE = ["0033"]          # A re-solved where the source has no finished log


# --------------------------------------------------------------- log parse --
def parse_log(path: Path) -> dict:
    out = {}
    if not path or not path.exists():
        return out
    txt = path.read_text(errors="ignore")
    m = re.search(r"Root relaxation: objective [\d.e+-]+, (\d+) iterations, ([\d.]+) seconds", txt)
    if m:
        out["root_s"] = float(m.group(2))
    m = re.search(r"Barrier solved model in (\d+) iterations and ([\d.]+) seconds", txt)
    if m:
        out["barrier_s"] = float(m.group(2))
    m = re.search(r"Explored (\d+) nodes \((\d+) simplex iterations\) in ([\d.]+) seconds", txt)
    if m:
        out["nodes"] = int(m.group(1)); out["total_s"] = float(m.group(3))
    m = re.search(r"Best objective ([\d.e+-]+), best bound ([\d.e+-]+), gap ([\d.]+)%", txt)
    if m:
        out["obj"] = float(m.group(1)); out["gap_pct"] = float(m.group(3))
    if "Time limit reached" in txt:
        out["status"] = "time_limit"
    elif "Optimal solution found" in txt:
        out["status"] = "optimal"
    elif "interrupted" in txt:
        out["status"] = "interrupted"
    else:
        out["status"] = "running/unknown"
    t_first = None
    for mm in re.finditer(r"^[H*]?\s*\d+\s+\d+\s+.*?\s([\d.]+)%\s+[\d.-]+\s+(\d+)s\s*$", txt, re.M):
        t_first = int(mm.group(2)); break
    out["t_first_inc_s"] = t_first
    out["root_method"] = "barrier" if "barrier_s" in out else ("simplex" if "root_s" in out else None)
    return out


def find_log(run_folder: Path):
    logs = sorted(run_folder.glob("userData/*/solver_log.txt"))
    return logs[-1] if logs else None


def find_summary(run_folder: Path):
    js = sorted(run_folder.glob("userData/*/optimization_results_summary.json"))
    return js[-1] if js else None


# --------------------------------------------------------------------- job --
def solve_job(args):
    """Copy input_data, patch ConfigModel, solve with the runner's own
    solve_single_model. Runs in a spawned process."""
    label, run_id, src_run, dst_run, patch = args
    t0 = time.time()
    try:
        dst_run = Path(dst_run); src_run = Path(src_run)
        inp, out = dst_run / "input_data", dst_run / "userData"
        if not inp.exists():
            shutil.copytree(src_run / "input_data", inp)
        out.mkdir(parents=True, exist_ok=True)
        params = json.load(open(src_run / "run_params.json"))
        threads = int(patch.get("threads", params.get("threads", 3)))
        params["threads"] = threads
        # patch solver options + output path
        cfg_path = inp / "ConfigModel.json"
        cfg = json.load(open(cfg_path))
        so = cfg["solveroptions"]
        for k, v in patch.items():
            if k == "NoRelHeurTime":
                so["NoRelHeurTime"] = v
            elif k in so and isinstance(so[k], dict) and "value" in so[k]:
                so[k]["value"] = v
            else:
                so[k] = {"value": v}
        so["threads"]["value"] = threads
        so["timelim"]["value"] = TIME_LIMIT_H
        cfg["reporting"]["save_path"]["value"] = str(out)
        cfg["reporting"]["save_summary_path"]["value"] = str(out)
        json.dump(cfg, open(cfg_path, "w"), indent=4)

        from FOUR_NODE_run_creation_and_gurobi_optimization_parallel import solve_single_model
        res = solve_single_model((f"{label}_{run_id}", str(inp), str(out), params))
        err = res[-1] if isinstance(res, tuple) else None
        return label, run_id, time.time() - t0, err
    except Exception as e:  # noqa
        import traceback
        return label, run_id, time.time() - t0, f"{e}\n{traceback.format_exc()}"


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="campaign folder with created runs")
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    ap.add_argument("--settings", default=",".join(DEFAULT_SETTINGS))
    ap.add_argument("--f-runs", default=",".join(DEFAULT_F_RUNS), help="runs for the threads setting F")
    ap.add_argument("--rerun-baseline", default=",".join(DEFAULT_RERUN_BASELINE))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = Path(a.source)
    if not src.is_absolute():
        src = HERE / src
    runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    settings = [s.strip() for s in a.settings.split(",") if s.strip()]
    f_runs = [r.strip() for r in a.f_runs.split(",") if r.strip()]
    rerun_a = [r.strip() for r in a.rerun_baseline.split(",") if r.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = HERE / "results" / f"solver_param_test_{ts}"

    jobs = []
    for s in settings:
        for r in (f_runs if s == "F" else runs):
            jobs.append((s, r, src / f"parallel_run_{r}", dst / s / f"parallel_run_{r}", SETTINGS[s]))
    for r in rerun_a:
        jobs.append(("A", r, src / f"parallel_run_{r}", dst / "A" / f"parallel_run_{r}", SETTINGS["A"]))
    # order: 3-thread jobs first, 6-thread last -> the first wave fits the cores
    jobs.sort(key=lambda j: (int(j[4].get("threads", 3)) > 3, j[0], j[1]))

    threads_total = sum(int(j[4].get("threads", 3)) for j in jobs[:a.workers])
    print(f"[TEST] {len(jobs)} jobs, {a.workers} workers, first wave {threads_total} Gurobi threads, "
          f"time limit {TIME_LIMIT_H} h per job")
    for j in jobs:
        print(f"   {j[0]} run {j[1]} threads {j[4].get('threads', 3)} patch {j[4]}")
    print(f"[TEST] output: {dst}")
    for j in jobs:
        if not (j[2] / "input_data").exists():
            raise SystemExit(f"missing {j[2] / 'input_data'}")
    if a.dry_run:
        return

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "settings.json").write_text(json.dumps(SETTINGS, indent=2))
    t_start = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        futs = {ex.submit(solve_job, j): j for j in jobs}
        for f in as_completed(futs):
            label, run_id, wall, err = f.result()
            print(f"[DONE] {label} {run_id} wall {wall / 3600:.2f} h" + (f"  ERROR: {err}" if err else ""))
    print(f"[TEST] all jobs finished in {(time.time() - t_start) / 3600:.2f} h")
    write_report(src, dst, runs, settings, f_runs, rerun_a)


def write_report(src, dst, runs, settings, f_runs, rerun_a):
    import pandas as pd
    rows = []
    # baseline A from the source campaign logs (same machine, 3 threads)
    for r in runs:
        rf = src / f"parallel_run_{r}"
        d = parse_log(find_log(rf))
        p = json.load(open(rf / "run_params.json"))
        rows.append(dict(setting="A(source)", run=r, econ=p["economy"], price=round(p["electricity_price_avg"]),
                         threads=p.get("threads"), **d))
    for s in settings + (["A"] if rerun_a else []):
        for r in (f_runs if s == "F" else (rerun_a if s == "A" else runs)):
            rf = dst / s / f"parallel_run_{r}"
            d = parse_log(find_log(rf))
            p = json.load(open(src / f"parallel_run_{r}" / "run_params.json"))
            rows.append(dict(setting=s, run=r, econ=p["economy"], price=round(p["electricity_price_avg"]),
                             threads=SETTINGS[s].get("threads", 3), **d))
    df = pd.DataFrame(rows)
    base = df[df.setting == "A(source)"].set_index("run")
    df["speedup_vs_A"] = [round(base.total_s.get(r, float("nan")) / t, 2) if t else None
                          for r, t in zip(df.run, df.total_s)]
    df["npv_rel_diff_vs_A"] = [((o - base.obj.get(r, float("nan"))) / base.obj.get(r, float("nan")))
                               if o and r in base.index and base.obj.get(r) else None
                               for r, o in zip(df.run, df.obj)]
    cols = ["setting", "run", "econ", "price", "threads", "root_method", "root_s", "barrier_s",
            "t_first_inc_s", "nodes", "total_s", "gap_pct", "status", "speedup_vs_A", "npv_rel_diff_vs_A"]
    df = df[[c for c in cols if c in df.columns]].sort_values(["run", "setting"])
    df.to_csv(dst / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    txt = df.to_string(index=False)
    (dst / "report.txt").write_text(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()
