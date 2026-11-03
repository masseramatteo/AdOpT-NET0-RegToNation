"""
Thread test (2026-11): are 2 Gurobi threads per run slower overall than 3?

The flat-LHS campaign ran 22 workers x 2 threads (44 cores); v5 on Snellius used 3 threads.
Here 45 of its economies are re-solved with 15 workers x 3 threads (45 cores, same machine
load) and compared with their 2-thread times from the campaign. Equal cores -> 3 threads gives
more runs/h only if the per-run speedup t2/t3 exceeds 22/15 = 1.47.

Selection: optimal campaign runs only, 9 bands of equal size on 2-thread work units, 5 runs at
random per band (SELECTION_SEED) -> easy to slow tail. The chosen list is fixed in
thread_test_selection.csv (next to this script) and copied to <results>/. Only these 45 runs are
launched, into a new results folder; the campaign folder is only read, never re-run.

Reuses FOUR_NODE_run_flatlhs unchanged (same design, worker and watchdog); only the worker and
thread counts are overridden. Run ids are kept, so parallel_run_XXXX is the same run in both folders.

Run:      python FOUR_NODE_run_flatlhs_threads.py
Analyse:  python FOUR_NODE_run_flatlhs_threads.py analyse <thread_test_results_folder>
          (also runs automatically at the end of the test)
Monitor:  <results>/progress.txt, <results>/watchdog.log
"""

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import FOUR_NODE_run_flatlhs as fl

REFERENCE_FOLDER = Path(__file__).resolve().parent / "results" / "flatlhs_validation_20261028_171519"
N_BANDS = 9
RUNS_PER_BAND = 5
SELECTION_SEED = 7
WORKERS_2, THREADS_2 = 22, 2       # campaign
WORKERS_3, THREADS_3 = 15, 3       # this test
RESUME_FOLDER = None               # e.g. r"...\results\flatlhs_threads3_20261103_120000"
# the 45 runs, chosen 2026-11-03 with select_runs() on the campaign logs; columns hold the
# 2-thread reference (sec, wu, nodes). Delete the file to reselect from REFERENCE_FOLDER.
SELECTION_FILE = Path(__file__).resolve().parent / "thread_test_selection.csv"

_EXPLORED = re.compile(r"Explored (\d+) nodes .* in ([0-9.]+) seconds \(([0-9.]+) work units\)")


def solver_stats(run_folder):
    """Nodes, seconds, work units and optimality from the latest solver_log.txt of a run."""
    logs = sorted(Path(run_folder).glob("userData/*/solver_log.txt"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    text = logs[-1].read_text(encoding="utf-8", errors="replace")
    m = _EXPLORED.findall(text)
    if not m:
        return None
    nodes, sec, wu = m[-1]
    return {"nodes": int(nodes), "sec": float(sec), "wu": float(wu),
            "optimal": "Optimal solution found" in text}


def reference_table(folder=REFERENCE_FOLDER):
    rows = []
    for run in sorted(Path(folder).glob("parallel_run_*")):
        s = solver_stats(run)
        if s:
            rows.append({"run_id": run.name, **s})
    return pd.DataFrame(rows)


def select_runs(ref):
    ok = ref[ref.optimal].copy()
    ok["band"] = pd.qcut(ok.wu, N_BANDS, labels=False)
    rng = np.random.default_rng(SELECTION_SEED)
    picks = [g.loc[rng.choice(g.index, RUNS_PER_BAND, replace=False)] for _, g in ok.groupby("band")]
    return pd.concat(picks).sort_values("run_id")


def analyse(test_folder):
    test_folder = Path(test_folder)
    sel = pd.read_csv(test_folder / "thread_test_selection.csv", sep=";")
    rows = []
    for _, r in sel.iterrows():
        s3 = solver_stats(test_folder / r.run_id)
        if s3 is None:
            continue
        rows.append({"run_id": r.run_id, "archetype": r.archetype, "band": r.band,
                     "sec_2": r.sec, "sec_3": s3["sec"], "wu_2": r.wu, "wu_3": s3["wu"],
                     "nodes_2": r.nodes, "nodes_3": s3["nodes"], "optimal_3": s3["optimal"]})
    df = pd.DataFrame(rows)
    df["speedup"] = df.sec_2 / df.sec_3
    df["s_per_wu_2"] = df.sec_2 / df.wu_2
    df["s_per_wu_3"] = df.sec_3 / df.wu_3
    df["wu_ratio"] = df.wu_3 / df.wu_2
    df.to_csv(test_folder / "thread_test_runs.csv", sep=";", index=False)

    breakeven = WORKERS_2 / WORKERS_3                                # 22/15
    # campaign throughput at equal cores: workers / mean solve time
    tp2 = WORKERS_2 / (df.sec_2.mean() / 3600)
    tp3 = WORKERS_3 / (df.sec_3.mean() / 3600)
    lines = [
        f"Thread test {datetime.now():%Y-%m-%d %H:%M}: {len(df)} runs, "
        f"{WORKERS_2}x{THREADS_2} (campaign) vs {WORKERS_3}x{THREADS_3} (test)",
        f"optimal at 3 threads: {int(df.optimal_3.sum())}/{len(df)}",
        "",
        f"speedup t2/t3: median {df.speedup.median():.2f}, geometric mean "
        f"{np.exp(np.log(df.speedup).mean()):.2f}, total time {df.sec_2.sum() / df.sec_3.sum():.2f}"
        f"  (break-even {breakeven:.2f})",
        f"work units t3/t2: median {df.wu_ratio.median():.2f}  (search path effect)",
        f"s per work unit: 2 threads {df.s_per_wu_2.median():.3f}, 3 threads {df.s_per_wu_3.median():.3f}"
        f"  (speed per unit of work)",
        f"runs/h at equal cores: 2 threads {tp2:.1f}, 3 threads {tp3:.1f}  "
        f"-> {'3' if tp3 > tp2 else '2'} threads better ({tp3 / tp2 - 1:+.0%} for 3)",
        "",
        "by difficulty band (2-thread work units):",
        df.groupby("band").agg(n=("run_id", "size"), sec_2_med=("sec_2", "median"),
                               sec_3_med=("sec_3", "median"),
                               speedup_med=("speedup", "median")).round(2).to_string(),
        "",
        "by archetype:",
        df.groupby("archetype").agg(n=("run_id", "size"),
                                    speedup_med=("speedup", "median")).round(2).to_string(),
    ]
    report = "\n".join(lines)
    (test_folder / "thread_test_report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    return df


def main():
    fl.MAX_WORKERS = WORKERS_3
    fl.THREADS_PER_WORKER = THREADS_3
    base_path = Path(fl.__file__).resolve().parent
    topology_folder = base_path / "preprocess" / fl.os.environ["FOUR_NODE_TOPOLOGY_FOLDER"]
    if RESUME_FOLDER:
        results_folder = Path(RESUME_FOLDER)
        print(f"[RESUME] {results_folder}")
    else:
        results_folder = base_path / "results" / f"flatlhs_threads3_{datetime.now():%Y%m%d_%H%M%S}"
    results_folder.mkdir(parents=True, exist_ok=True)

    sel_file = results_folder / "thread_test_selection.csv"
    if not sel_file.exists():
        if SELECTION_FILE.exists():                  # fixed list, committed with this script
            shutil.copy(SELECTION_FILE, sel_file)
        else:
            print(f"[SELECT] reading solver logs in {REFERENCE_FOLDER}")
            ref = reference_table()
            run_table = pd.read_csv(REFERENCE_FOLDER / "run_table.csv")
            ref["archetype"] = ref.run_id.map(
                lambda rid: run_table.archetype.iloc[int(rid.split("_")[-1]) - 1])
            select_runs(ref).to_csv(sel_file, sep=";", index=False)
    sel = pd.read_csv(sel_file, sep=";")
    print(f"[SELECT] {', '.join(sel.run_id)}")
    print(f"[SELECT] {len(sel)} runs; archetypes {sel.archetype.value_counts().to_dict()}")

    # same deterministic design as the campaign, filtered to the selected run ids
    combinations = fl.build_design(results_folder, topology_folder)
    run_configs = [(f"parallel_run_{i:04d}", params) for i, params in enumerate(combinations, 1)]
    wanted = set(sel.run_id)
    run_configs = [rc for rc in run_configs if rc[0] in wanted]
    print(f"\n[OK] Thread test: {len(run_configs)} runs, {WORKERS_3} workers x {THREADS_3} threads")
    fl.Watchdog(base_path, results_folder, run_configs).run()
    analyse(results_folder)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "analyse":
        analyse(sys.argv[2])
    else:
        main()
