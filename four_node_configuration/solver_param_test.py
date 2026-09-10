"""
Gurobi parameter test on already-created four-node models.

Re-solves the input_data of a few runs of an existing campaign folder under
several solver settings, in parallel, and compares solve time, root time,
nodes, gap and NPV against the campaign's own solve of the same run.

The baseline is the source campaign's own log, on the same machine. Since the
2026-09-09 22:00 pilot the campaign already runs with Method=2, so the source
baseline IS setting B; "B" is kept in SETTINGS as an explicit control that the
source logs are comparable with jobs re-solved under test contention.

Second round (2026-09-10). The first round tested B/C/D/F, where C bundled
mipfocus=0 and cuts=-1 and came out a dead heat with B (14196 s vs 14247 s over
four runs) because the two knobs pull in opposite directions: mipfocus=0 found
the first incumbent 2.7-3.7x earlier, cuts=-1 lost the tight root bound on run
0025 (5 nodes -> 22). This round changes ONE parameter per setting so the two
effects can be read separately, on runs picked from the phase analysis of the
barrier pilot (136 runs: 2% root LP, 27% root cut loop, 71% tree; median 49% of
each run spent with no incumbent at all; median 13 200 dual simplex iterations
per node).

Usage (from four_node_configuration/):
    python solver_param_test.py --source results/parallel_creation_test_20260909_215958
    python solver_param_test.py --source ... --runs 0001,0110 --settings M0,K --workers 8
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
# Round 3 (2026-09-10 afternoon), against a hard deadline: Snellius launches
# tonight. The cap is set so that most jobs actually CLOSE, because the metric
# that matters is time to 1e-4 - what the campaign really does.
#
# A first draft of this round used a 1.5 h cap and gap_at_cap_pct as the metric.
# That was wrong twice over. (1) The gap does not descend smoothly: run 0110 at
# baseline hits 5% at 7539 s, 1% at 21613 s and closes at 21622 s - it sat
# between 5% and 1% for four hours and then collapsed in 9 seconds. A snapshot
# at a fixed time barely predicts the finish. (2) Worse, first incumbents on
# 0110/0016/0089 land at 6824/5635/8031 s, so at a 5400 s cap three of seven runs
# would have had no incumbent at all and an undefined gap - and the metric would
# have mechanically rewarded M0/MK, the settings that optimise time-to-first-
# incumbent, against B which keeps a tight bound but finds its solution late.
# That is measuring the hypothesis with itself.
TIME_LIMIT_H = 3.0

SETTINGS = {
    # A = the pre-2026-09-09 campaign settings (Method -1 -> concurrent simplex
    #     at the root). Kept for reference only, not in DEFAULT_SETTINGS.
    "A": {"method": -1},
    # B = current campaign settings: barrier at the root, MIPFocus 2, Cuts 2,
    #     NumericFocus 2, ScaleFlag 2, NodeMethod 1, Heuristics 0.05.
    #     Control: same settings as the source logs, re-solved under the test's
    #     own worker contention.
    "B": {"method": 2},
    # --- one knob at a time, all on top of B ---------------------------------
    # M0: MIPFocus 0 (balanced) instead of 2 (focus on the bound). Target: the
    #     no-incumbent phase, 49% of run time at the median.
    "M0": {"method": 2, "mipfocus": 0},
    # K:  Cuts 1 (moderate) instead of 2 (aggressive). Target: the root cut
    #     loop, 27% of campaign time. Not -1: on run 0025 the aggressive cuts
    #     bought a much tighter root bound (5 nodes vs 22).
    "K": {"method": 2, "cuts": 1},
    # N:  Gurobi default numerics instead of NumericFocus 2 + ScaleFlag 2,
    #     which were tuned in Dec 2025 on the 6.9 M-row model without typical
    #     days. Target: the 13 200 simplex iterations per node.
    #     WATCH the NPV column: the model has coefficients 6e-4 .. 4e8, this is
    #     the one setting that can trade accuracy for speed.
    "N": {"method": 2, "numericfocus": 0, "scaleflag": -1},
    # ND: barrier for the node LPs too. If branching an existence binary to 0
    #     wipes out 8760 h of flows, the parent basis is worth little and the
    #     node LP is effectively a fresh solve.
    "ND": {"method": 2, "nodemethod": 2},
    # HE: more time in heuristics (0.2 vs 0.05). Same target as M0, other road.
    "HE": {"method": 2, "heuristics": 0.2},
    # MK: round 2's two survivors together. They win on different runs (K on
    #     0090/0025, M0 on 0110/0016), which suggests they are complementary -
    #     but round 1 is the proof that two individually sensible knobs can
    #     cancel, so the combination is tested, never assumed.
    "MK": {"method": 2, "mipfocus": 0, "cuts": 1},
    # kept from round 1, not in DEFAULT_SETTINGS:
    # C: mipfocus and cuts together (the bundle that tied with B)
    "C": {"method": 2, "mipfocus": 0, "cuts": -1},
    # D: C + NoRel heuristic 300 s before the root LP (needs the pass-through
    #    in adopt_net0/utilities.get_gurobi_parameters)
    "D": {"method": 2, "mipfocus": 0, "cuts": -1, "NoRelHeurTime": 300},
    # F: B with 6 Gurobi threads instead of 3 (barrier parallelises)
    "F": {"method": 2, "threads": 6},
}

# Round 3 runs, all from the barrier pilot parallel_creation_test_20260909_215958.
# Round 2 used 5 runs and the B-vs-source spread showed run-to-run noise of about
# +/-50%, which is the same size as the effects being measured. So round 3 drops
# the two cheap guard runs (0001, 0025: together 8% of campaign compute) and
# spends the budget on 7 slow runs instead - the class that owns 61% of the
# compute - chosen to span price, cut-loop size and incumbent difficulty:
#   run   econ  price  base h  nodes  cut_loop  1st inc   why
#   0016    2     31    3.80     53     3563      5635    biggest cut loop (K's case)
#   0090   12    204    3.54    110     2611      1079    continuity with round 2; easy incumbent
#   0107   14     78    3.43     75     1361      2265    economy 14, the pathological one
#   0122   16     38    3.21     69     1771      3432    mid
#   0037    5    100    3.12     87     1230      2495    mid price
#   0113   15    162    3.00     70     1277      2879    mid-high price
#   0051    7     93    2.53     68     1991      3886    mid price, second economy
#   0089   12    204    2.23     75     1049      8031    latest incumbent of the pilot (M0's case)
# Seven distinct economies (2, 12, 14, 16, 5, 15, 7). Every run is <= 3.80 h at
# baseline, so most should CLOSE inside the 3 h cap and yield the real metric,
# time to 1e-4.
# 8 runs x 4 settings = 32 jobs on 16 workers = 2 waves. Two waves do NOT
# reintroduce round 2's confound: jobs are ordered by run, so all four settings
# of a given run sit in the same wave, and every comparison is within-run against
# B - i.e. under identical machine load.
# 0110 (6.01 h at baseline) stays out: nothing closes it inside 3 h, so it would
# spend four jobs to produce four capped results. Economy 14 is covered by 0107.
DEFAULT_RUNS = ["0016", "0090", "0107", "0122", "0037", "0113", "0051", "0089"]
DEFAULT_SETTINGS = ["B", "M0", "K", "MK"]
DEFAULT_F_RUNS = ["0110", "0016"]          # threads test only on the slow ones
DEFAULT_RERUN_BASELINE = []                # source logs are already Method=2


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
    # time to reach a given gap, from the node log: under the cap some jobs will
    # not close, and "gap X% reached at t" is comparable across settings anyway
    body = txt[txt.find("Expl Unexpl"):]
    pts = []
    for ln in body.splitlines():
        mt = re.search(r"(\d+)s\s*$", ln)
        mg = re.search(r"\s([\d.]+)%\s", ln)
        if mt and mg:
            pts.append((int(mt.group(1)), float(mg.group(1))))
    for tgt, key in ((5.0, "t_gap5_s"), (1.0, "t_gap1_s"), (0.5, "t_gap05_s"), (0.1, "t_gap01_s")):
        out[key] = next((t for t, g in pts if g <= tgt), None)
    # gap reached at the test's cap. The baseline logs come from the campaign,
    # which ran with a 50 h limit, so without this the capped test jobs would be
    # compared against a baseline that ran to optimality.
    cap = TIME_LIMIT_H * 3600
    at_cap = [g for t, g in pts if t <= cap]
    out["gap_at_cap_pct"] = at_cap[-1] if at_cap else None
    if out.get("total_s") and out["total_s"] <= cap and out.get("gap_pct") is not None:
        out["gap_at_cap_pct"] = out["gap_pct"]      # closed before the cap
    # Best dual bound reached. Unlike the gap this is monotone and defined from
    # the first node on, whether or not an incumbent exists - so for a job that
    # hits the cap it is the honest progress measure. write_report turns it into
    # a distance from the optimum already known from the pilot.
    bd = re.findall(r"^[H*]?\s*\d+\s+\d+\s+.*?\s([\d.]+e[+-]\d+)\s+[\d.]+%", txt, re.M)
    if not bd:
        bd = re.findall(r"\s([\d.]+e[+-]\d+)\s+[-\d.]+%\s", txt)
    out["best_bound"] = float(bd[-1]) if bd else None
    # end of the root cut loop = last "0 0" line of the node log
    cut_end = None
    for ln in body.splitlines():
        p = ln.split()
        mt = re.search(r"(\d+)s\s*$", ln)
        if mt and len(p) > 2 and p[0].lstrip("*H") == "0" and p[1] == "0":
            cut_end = int(mt.group(1))
    if cut_end is not None and out.get("root_s"):
        out["cut_loop_s"] = round(cut_end - out["root_s"])
    # Numerical health. Round 2 lesson: `numerical (trouble|issues)` was far too
    # broad - it matched the tail of Gurobi's advisory "...setting NumericFocus
    # parameter to avoid numerical issues", which is printed precisely BECAUSE
    # NumericFocus is off, so setting N looked broken when it was not. Match only
    # genuine trouble, and report the violation as a magnitude: every run of this
    # campaign, the production baseline included, ends with a max constraint
    # violation around 1e-5 against a 1e-6 tolerance (coefficients up to 4e8).
    # That is the mechanism behind the 0.1-0.3% NPV noise floor, so the number is
    # worth tracking; its mere presence is not a discriminator.
    out["numerical"] = 1 if re.search(r"numerical trouble", txt, re.I) else 0
    m = re.search(r"max constraint violation \(([\d.e+-]+)\)", txt)
    out["max_viol"] = float(m.group(1)) if m else None
    # wall-clock start, so the load confound stays auditable after the fact
    m = re.search(r"logging started \w{3} (\w{3} +\d+ \d+:\d+:\d+ \d{4})", txt)
    out["started"] = m.group(1) if m else None
    return out


def runs_for(setting: str, runs, f_runs):
    """A setting that overrides the thread count is tested on f_runs only."""
    return f_runs if "threads" in SETTINGS[setting] else runs


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
    ap.add_argument("--workers", type=int, default=16)   # 16 x 3 = 48 cores, one wave of 16 jobs
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
        for r in runs_for(s, runs, f_runs):
            jobs.append((s, r, src / f"parallel_run_{r}", dst / s / f"parallel_run_{r}", SETTINGS[s]))
    for r in rerun_a:
        jobs.append(("A", r, src / f"parallel_run_{r}", dst / "A" / f"parallel_run_{r}", SETTINGS["A"]))
    # Order by RUN first, then setting. Round 2 sorted by setting, which put all
    # of B/HE/K in the first wave and pushed N/ND 2-3 h later onto a machine that
    # had emptied out: setting was confounded with machine load (measured mean
    # concurrency 14.6 for B against 11.7 for ND). Sorting by run interleaves the
    # settings so every one of them spans the same load profile.
    # 6-thread jobs still go last so the first wave fits the cores.
    jobs.sort(key=lambda j: (int(j[4].get("threads", 3)) > 3, j[1], j[0]))

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
    # baseline from the source campaign logs (same machine, 3 threads). Since
    # 2026-09-09 22:00 the campaign runs Method=2, so this is B without the
    # test's own contention; the "B" setting below is the contended control.
    for r in runs:
        rf = src / f"parallel_run_{r}"
        d = parse_log(find_log(rf))
        p = json.load(open(rf / "run_params.json"))
        rows.append(dict(setting="base(source)", run=r, econ=p["economy"], price=round(p["electricity_price_avg"]),
                         threads=p.get("threads"), **d))
    for s in settings + (["A"] if rerun_a else []):
        for r in (rerun_a if s == "A" else runs_for(s, runs, f_runs)):
            rf = dst / s / f"parallel_run_{r}"
            d = parse_log(find_log(rf))
            p = json.load(open(src / f"parallel_run_{r}" / "run_params.json"))
            rows.append(dict(setting=s, run=r, econ=p["economy"], price=round(p["electricity_price_avg"]),
                             threads=SETTINGS[s].get("threads", 3), **d))
    df = pd.DataFrame(rows)
    base = df[df.setting == "base(source)"].set_index("run")
    df["speedup_vs_base"] = [round(base.total_s.get(r, float("nan")) / t, 2) if t else None
                             for r, t in zip(df.run, df.total_s)]
    # comparable across settings even when the cap is hit
    df["gap1_speedup"] = [round(base.t_gap1_s.get(r, float("nan")) / t, 2) if t else None
                          for r, t in zip(df.run, df.t_gap1_s)]
    df["npv_rel_diff_vs_base"] = [((o - base.obj.get(r, float("nan"))) / base.obj.get(r, float("nan")))
                                  if o and r in base.index and base.obj.get(r) else None
                                  for r, o in zip(df.run, df.obj)]
    # Distance of the dual bound from the optimum the pilot already established
    # for this run (objective is minimised, so bound <= z*). Monotone, defined
    # even with no incumbent: this is what to read for a job stopped at the cap.
    df["bound_to_opt_pct"] = [round(100 * (base.obj.get(r, float("nan")) - bd)
                                    / base.obj.get(r, float("nan")), 3)
                              if bd and r in base.index and base.obj.get(r) else None
                              for r, bd in zip(df.run, df.best_bound)]
    cols = ["setting", "run", "econ", "price", "threads", "started", "root_s", "cut_loop_s",
            "t_first_inc_s", "t_gap5_s", "t_gap1_s", "t_gap05_s", "t_gap01_s", "nodes", "total_s",
            "gap_pct", "gap_at_cap_pct", "bound_to_opt_pct", "status", "numerical", "max_viol",
            "speedup_vs_base", "gap1_speedup", "npv_rel_diff_vs_base"]
    df = df[[c for c in cols if c in df.columns]].sort_values(["run", "setting"])
    df.to_csv(dst / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    txt = df.to_string(index=False)

    # per-setting aggregate: which knob is worth adopting
    agg_cols = {c: "sum" for c in ("total_s", "t_gap1_s", "cut_loop_s", "t_first_inc_s") if c in df}
    if agg_cols:
        agg = df.groupby("setting").agg(n=("run", "size"), **{k: (k, v) for k, v in agg_cols.items()})
        b = agg.loc["base(source)"] if "base(source)" in agg.index else None
        if b is not None:
            for c in agg_cols:
                agg[c + "_x"] = (b[c] / agg[c]).round(2)
        txt += "\n\nPer-setting totals over the common runs (x = speedup vs base):\n" + agg.to_string()
        txt += ("\n\nRound 3 read, in order. (1) total_s for the jobs with status=optimal - this is "
                "the real metric, time to 1e-4, which is what the campaign does. Compare against "
                "setting B ON THE SAME RUN, never against the source logs: the test machine runs 16 "
                "hard jobs at once and is harsher than the pilot was. (2) For a job at the cap, read "
                "bound_to_opt_pct, not the gap: the bound is monotone and defined even before an "
                "incumbent exists, whereas the gap can sit flat for hours and then collapse in seconds "
                "(run 0110 at baseline: 5% at 7539 s, 1% at 21613 s, closed at 21622 s). (3) "
                "t_first_inc_s separately, as the thing M0/MK actually change. "
                "npv_rel_diff_vs_base is NOT usable for a capped job: the objective is minimised, so "
                "its incumbent is suboptimal and the drift measures suboptimality, not solver noise. "
                "max_viol is ~1e-5 for every setting including the baseline; only a value far out of "
                "that range, or numerical=1, is a red flag.")
    (dst / "report.txt").write_text(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()