"""
Near-optimal re-optimization comparison for four-node configuration.

Given a results folder with first-best optimization results:
1. Reads installed technologies and NPV (objective) from first-best runs
2. Re-runs each simulation excluding one installed technology at a time
3. Compares the new NPV vs the first-best NPV
4. Exports a comparison table (CSV + Excel)

Usage:
    python near_optimal_analysis.py
    Set RESULTS_FOLDER at the bottom of this file.
"""

import json
import sys
import h5py
import pandas as pd
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_PATH = Path(__file__).resolve().parent.parent  # four_node_configuration/
sys.path.insert(0, str(BASE_PATH))

# ── Technology mapping ────────────────────────────────────────────────────────
# Maps technology name → which params key it belongs to
TEC_PARAM_MAP = {
    "Electrolyzer_small":            "small_cluster_new_technologies",
    "Storage_H2_lowP":               "small_cluster_new_technologies",
    "Photovoltaic":                  "small_cluster_new_technologies",
    "Electrolyzer_big":              "big_cluster_new_technologies",
    "Storage_H2_highP":              "big_cluster_new_technologies",
    "hydrogenPipelineOnshore_lowP":  "networks_new",
    "hydrogenPipelineOnshore_highP": "networks_new",
}

# Technologies that should never be excluded (existing/fixed)
EXCLUDE_FROM_ANALYSIS = {"Storage_H2_Cavern", "Storage_H2_Cavern_existing", "Electrolyzer_big", "Storage_H2_highP"}

# Combinations to exclude together
# If any small-cluster tech is installed, exclude all three together (no small-cluster scenario)
COMBO_JOBS = [
    ("Electrolyzer_small", "Storage_H2_lowP", "Photovoltaic"),
]

# Trigger techs for each combo: combo runs if ANY of these is installed.
# If entry is None, falls back to COMBO_INDEPENDENT_OF_INSTALLATION logic on the full combo.
COMBO_TRIGGER_TECHS = [
    ("Electrolyzer_small",),  # only Electrolyzer_small triggers; Photovoltaic/Storage alone do not
]

# Minimum installed size (MW) to consider a technology as "installed" for trigger evaluation.
# Technologies not listed here use DEFAULT_INSTALLATION_THRESHOLD.
DEFAULT_INSTALLATION_THRESHOLD = 0.0
INSTALLATION_THRESHOLDS = {
    "Electrolyzer_small": 1.0,
}

# Networks to disable in the "no network" scenario (always run, regardless of installation)
NO_NETWORK_TECHS = ("hydrogenPipelineOnshore_lowP", "hydrogenPipelineOnshore_highP")

# Set to True to include network re-optimization jobs (single arc exclusions + no-network scenario)
REOPT_NETWORKS = False

# Technologies to skip in single-exclusion jobs (can still appear in COMBO_JOBS)
SKIP_SINGLE_EXCLUSIONS = {"Electrolyzer_small", "Storage_H2_lowP", "Photovoltaic", "hydrogenPipelineOnshore_lowP", "hydrogenPipelineOnshore_highP"}

# If True, combo job runs if at least ONE tech in the combo is installed.
# If False, combo job runs only if ALL techs in the combo are installed.
COMBO_INDEPENDENT_OF_INSTALLATION = True


# ─────────────────────────────────────────────────────────────────────────────
# READING RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def find_h5(run_folder):
    """Find the optimization_results.h5 file in a run folder."""
    for p in Path(run_folder).rglob("optimization_results.h5"):
        return p
    return None


def get_objective(run_folder):
    """Read objective value (NPV) from optimization_results_summary.json."""
    for p in Path(run_folder).rglob("optimization_results_summary.json"):
        with open(p) as f:
            data = json.load(f)
        val = data.get("objective_value")
        return val if val is not None else data.get("var_npv")
    return None


def get_installed_technologies(h5_path):
    """
    Return:
      installed_tecs  : set of installed technology names (size > 0)
      installed_arcs  : dict { network_name: set of (from_node, to_node) tuples }
      installed_sizes : dict { tec_name: { location: size } }
                        For node techs   → location is the node name
                        For network arcs → location is "from→to"
    Only returns technologies in TEC_PARAM_MAP and not in EXCLUDE_FROM_ANALYSIS.
    """
    installed_tecs  = set()
    installed_arcs  = {}   # { net_name: {(from, to), ...} }
    installed_sizes = {}   # { tec_name: { location: size_val } }

    with h5py.File(str(h5_path), "r") as f:
        if "design/nodes/period1" in f:
            for node in f["design/nodes/period1"].keys():
                for tec in f["design/nodes/period1"][node].keys():
                    if tec in TEC_PARAM_MAP and tec not in EXCLUDE_FROM_ANALYSIS:
                        size = f[f"design/nodes/period1/{node}/{tec}/size"][()]
                        size_val = float(size[0]) if hasattr(size, "__len__") else float(size)
                        if size_val > 0:
                            installed_tecs.add(tec)
                            installed_sizes.setdefault(tec, {})[node] = size_val

        if REOPT_NETWORKS and "design/networks/period1" in f:
            for net in f["design/networks/period1"].keys():
                if net in TEC_PARAM_MAP:
                    for arc in f[f"design/networks/period1/{net}"].keys():
                        size = f[f"design/networks/period1/{net}/{arc}/size"][()]
                        size_val = float(size[0]) if hasattr(size, "__len__") else float(size)
                        if size_val > 0:
                            installed_tecs.add(net)
                            parsed = _parse_arc_key(arc)
                            if parsed:
                                installed_arcs.setdefault(net, set()).add(parsed)
                                arc_label = f"{parsed[0]}→{parsed[1]}"
                                installed_sizes.setdefault(net, {})[arc_label] = size_val

    return installed_tecs, installed_arcs, installed_sizes


def read_first_best_results(results_folder):
    """
    Scan results_folder and return list of dicts with first-best run info.
    Keys: run_id, run_folder, objective, params, installed_tecs
    """
    results_folder = Path(results_folder)
    runs = []

    for run_folder in sorted(results_folder.iterdir()):
        if not run_folder.is_dir():
            continue

        params_file = run_folder / "run_params.json"
        if not params_file.exists():
            continue

        objective = get_objective(run_folder)
        if objective is None:
            print(f"  [SKIP] {run_folder.name}: no objective found")
            continue

        h5_path = find_h5(run_folder)
        if h5_path is None:
            print(f"  [SKIP] {run_folder.name}: no h5 file found")
            continue

        with open(params_file) as f:
            params = json.load(f)

        installed_tecs, installed_arcs, installed_sizes = get_installed_technologies(h5_path)

        runs.append({
            "run_id":           run_folder.name,
            "run_folder":       run_folder,
            "objective":        objective,
            "params":           params,
            "installed_tecs":   installed_tecs,
            "installed_arcs":   installed_arcs,    # { net: {(from,to), ...} }
            "installed_sizes":  installed_sizes,   # { tec: { location: size } }
        })

    print(f"[READ] Found {len(runs)} completed first-best runs in {results_folder.name}")
    return runs


# All possible node names — used to parse arc keys in the h5 (no separator)
ALL_NODES = ["Large_cluster1", "Large_cluster2", "Small_cluster1", "Small_cluster2"]


def _parse_arc_key(arc_key):
    """
    Parse an arc key like 'Large_cluster1Small_cluster2' into
    ('Large_cluster1', 'Small_cluster2') by trying all known node names.
    Returns (from_node, to_node) or None if parsing fails.
    """
    for node in ALL_NODES:
        if arc_key.startswith(node):
            to_node = arc_key[len(node):]
            if to_node in ALL_NODES:
                return node, to_node
    return None


# Network technologies are intentionally left in params — the solver re-optimises
# their arcs freely; we only block node-level investments.
NODE_PARAM_KEYS = {
    "small_cluster_new_technologies",
    "big_cluster_new_technologies",
}

# Maps each node-level technology to the nodes whose technology_data folder
# contains its JSON file.
TEC_NODE_MAP = {
    "Electrolyzer_small": ["Small_cluster1", "Small_cluster2"],
    "Storage_H2_lowP":    ["Small_cluster1", "Small_cluster2"],
    "Photovoltaic":       ["Small_cluster1", "Small_cluster2"],
    "Electrolyzer_big":   ["Large_cluster1", "Large_cluster2"],
    "Storage_H2_highP":   ["Large_cluster1", "Large_cluster2"],
}


def _disable_network_arcs_in_csv(input_data_path, net_name, arcs_to_disable):
    """
    Set connection = 0 (both directions) for specific (from_node, to_node) pairs
    in the network topology connection.csv of net_name.
    This prevents the solver from building those arcs at all.
    Called AFTER create_single_model so the file already exists.
    """
    csv_path = (
        Path(input_data_path)
        / "period1" / "network_topology" / "new" / net_name / "connection.csv"
    )
    if not csv_path.exists():
        print(f"  [WARN] connection.csv not found for {net_name}, skipping arc disable")
        return

    df = pd.read_csv(csv_path, sep=";", index_col=0)
    disabled = 0
    for from_node, to_node in arcs_to_disable:
        # Both directions (network is bidirectional)
        if from_node in df.index and to_node in df.columns:
            df.loc[from_node, to_node] = 0
            disabled += 1
        if to_node in df.index and from_node in df.columns:
            df.loc[to_node, from_node] = 0
            disabled += 1
    df.to_csv(csv_path, sep=";")
    print(f"  [ARC] Disabled {disabled} connection(s) in {net_name} connection.csv")


def _disable_all_network_arcs_in_csv(input_data_path, net_name):
    """Set ALL connections to 0 in a network's connection.csv (force no-network scenario)."""
    csv_path = (
        Path(input_data_path)
        / "period1" / "network_topology" / "new" / net_name / "connection.csv"
    )
    if not csv_path.exists():
        print(f"  [WARN] connection.csv not found for {net_name}, skipping")
        return
    df = pd.read_csv(csv_path, sep=";", index_col=0)
    df[:] = 0
    df.to_csv(csv_path, sep=";")
    print(f"  [ARC] Disabled ALL connections in {net_name} connection.csv")


def _disable_technology_in_json(input_data_path, tec_name):
    """
    Set size_max = 0 (and size_min = 0) in every node JSON for tec_name.
    Called AFTER create_single_model so the files already exist.
    """
    nodes = TEC_NODE_MAP.get(tec_name, [])
    for node in nodes:
        json_path = (
            Path(input_data_path)
            / "period1" / "node_data" / node / "technology_data"
            / f"{tec_name}.json"
        )
        if not json_path.exists():
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        data["size_min"] = 0
        data["size_max"] = 0
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# RE-OPTIMIZATION WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _reopt_worker(args):
    """
    Spawned worker: excludes one OR more technologies and re-solves.

    excluded_tecs is a tuple of technology names. For each:
      - Node technologies    → size_max=0 patched into the technology JSON.
      - Network technologies → ALL arcs disabled in connection.csv (entire
                               connection matrix zeroed, regardless of what
                               was installed in the first-best solution).

    Returns (run_id, label, new_objective, error_string_or_None).
    """
    run_id, params, excluded_tecs, installed_arcs, reopt_base_folder, base_path, gurobi_threads = args

    # Override Gurobi thread count if requested
    if gurobi_threads is not None:
        params = dict(params)
        params["threads"] = int(gurobi_threads)

    import sys
    sys.path.insert(0, str(base_path))
    from FOUR_NODE_run_creation_and_gurobi_optimization_parallel import (
        create_single_model, solve_single_model,
    )

    label = "+".join(sorted(excluded_tecs))
    reopt_id = f"{run_id}__excl_{label}"
    reopt_base_folder = Path(reopt_base_folder)

    try:
        # Create with full params so all JSON / topology files are written
        result = create_single_model((reopt_id, params, reopt_base_folder, base_path))
        _, input_path, results_path, _, error = result
        if error:
            return run_id, label, None, f"Creation failed: {error}"

        # Disable each excluded technology
        for tec in excluded_tecs:
            param_key  = TEC_PARAM_MAP.get(tec)
            is_network = param_key == "networks_new"
            if is_network:
                # Always zero the entire connection matrix — when a network is excluded,
                # all arcs must be forbidden regardless of what was installed.
                _disable_all_network_arcs_in_csv(input_path, tec)
            else:
                _disable_technology_in_json(input_path, tec)

        result = solve_single_model((reopt_id, input_path, results_path, params))
        _, result_info, error = result
        if error:
            return run_id, label, None, f"Solve failed: {error}"

        objective_value = (result_info or {}).get("objective_value")
        npv_str = f"{objective_value:.0f}" if objective_value is not None else "None"
        print(f"  [REOPT] {reopt_id}: NPV={npv_str}", flush=True)
        return run_id, label, objective_value, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return run_id, label, None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_reopt_comparison(results_folder, output_folder=None, local_reopt_folder=None,
                         max_workers=4, gurobi_threads=None):
    """
    For each first-best run: re-run excluding ALL installed node-level technologies
    at once, then compare the new NPV with the original NPV.

    Args:
        results_folder:     path to folder with first-best optimization results
        output_folder:      where to save the final comparison CSV/Excel
                            (default: results_folder/../reopt_comparison_<name>)
        local_reopt_folder: LOCAL folder for intermediate re-opt input/results data.
                            On Windows: keep short to avoid WinError 206 on UNC paths.
                            On Linux/HPC: use $TMPDIR for fast node-local I/O.
        max_workers:        parallel workers for re-optimization
        gurobi_threads:     Gurobi threads per worker (None = inherit from run_params.json)
    """
    results_folder = Path(results_folder)
    if output_folder is None:
        output_folder = results_folder.parent / f"reopt_comparison_{results_folder.name}"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if local_reopt_folder is None:
        local_reopt_folder = Path("C:/Temp/reopt_runs")
    reopt_folder = Path(local_reopt_folder)
    reopt_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"RE-OPTIMIZATION COMPARISON")
    print(f"  Results folder      : {results_folder}")
    print(f"  Output folder       : {output_folder}")
    print(f"  Local reopt folder  : {reopt_folder}  ← intermediate files stored here")
    print(f"  Max workers         : {max_workers}")
    print(f"  Gurobi threads      : {gurobi_threads}")
    print(f"{'='*70}\n", flush=True)

    # ── Step 1: Read first-best results ──────────────────────────────────────
    print("[STEP 1] Reading first-best results...")
    first_best = read_first_best_results(results_folder)
    if not first_best:
        print("[ERROR] No completed runs found.")
        return None

    # ── Step 2: Build re-optimization jobs (one per run × technology) ──────────
    print(f"\n[STEP 2] Building re-optimization jobs...")
    jobs = []
    for run in first_best:
        # Single-technology exclusions (only for installed techs)
        for tec in sorted(run["installed_tecs"]):
            if tec not in TEC_PARAM_MAP:
                continue
            if tec in SKIP_SINGLE_EXCLUSIONS:
                continue
            jobs.append((
                run["run_id"], run["params"], (tec,),
                run["installed_arcs"],
                reopt_folder, BASE_PATH, gurobi_threads,
            ))

        # Combination exclusions
        for i, combo in enumerate(COMBO_JOBS):
            trigger = COMBO_TRIGGER_TECHS[i] if i < len(COMBO_TRIGGER_TECHS) else None
            if trigger is not None:
                def _above_threshold(t_, sizes_):
                    threshold = INSTALLATION_THRESHOLDS.get(t_, DEFAULT_INSTALLATION_THRESHOLD)
                    return any(v > threshold for v in (sizes_.get(t_) or {}).values())
                condition = any(_above_threshold(t, run["installed_sizes"]) for t in trigger)
            else:
                condition = (
                    any(t in run["installed_tecs"] for t in combo)
                    if COMBO_INDEPENDENT_OF_INSTALLATION
                    else all(t in run["installed_tecs"] for t in combo)
                )
            if condition:
                jobs.append((
                    run["run_id"], run["params"], combo,
                    run["installed_arcs"],
                    reopt_folder, BASE_PATH, gurobi_threads,
                ))

        # No-network scenario: always run, force-disables all pipeline connections
        if REOPT_NETWORKS:
            jobs.append((
                run["run_id"], run["params"], NO_NETWORK_TECHS,
                run["installed_arcs"],
                reopt_folder, BASE_PATH, gurobi_threads,
            ))

    print(f"  First-best runs      : {len(first_best)}")
    print(f"  Re-optimization jobs : {len(jobs)}")
    for run in first_best:
        tecs = sorted(t for t in run["installed_tecs"] if t in TEC_PARAM_MAP)
        print(f"    {run['run_id']}: {tecs}")

    if not jobs:
        print("[WARNING] No jobs to run.")
        return None

    # ── Step 3: Run re-optimizations in parallel ──────────────────────────────
    print(f"\n[STEP 3] Running re-optimizations ({max_workers} workers)...", flush=True)
    reopt_results = []
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        future_to_job = {executor.submit(_reopt_worker, job): job for job in jobs}
        done = 0
        for future in as_completed(future_to_job):
            done += 1
            run_id, label, obj, error = future.result()
            reopt_results.append({
                "run_id":       run_id,
                "excluded_tec": label,
                "npv_reopt":    obj,
                "error":        error,
            })
            if error:
                print(f"  [{done}/{len(jobs)}] {run_id} excl {label}: ERROR — {error}", flush=True)
            else:
                print(f"  [{done}/{len(jobs)}] {run_id} excl {label}: NPV={obj}", flush=True)

    # ── Step 4: Build comparison table ───────────────────────────────────────
    print(f"\n[STEP 4] Building comparison table...")

    first_best_map = {r["run_id"]: r for r in first_best}
    rows = []

    for res in reopt_results:
        run       = first_best_map[res["run_id"]]
        npv_orig  = run["objective"]
        npv_reopt = res["npv_reopt"]

        delta_abs = (npv_reopt - npv_orig)              if npv_reopt is not None else None
        delta_pct = (delta_abs / abs(npv_orig) * 100)   if (npv_reopt is not None and npv_orig) else None

        # Build removed_tech_sizes string for each excluded technology
        excluded_labels = res["excluded_tec"].split("+")
        parts = []
        for tec in excluded_labels:
            sizes = run["installed_sizes"].get(tec)
            if sizes:
                size_str = ", ".join(f"{loc}={v:.1f}" for loc, v in sizes.items())
                parts.append(f"{tec}: {size_str}")
            else:
                parts.append(f"{tec}: not installed")
        removed_tech_sizes = " | ".join(parts)

        rows.append({
            "run_id":                  res["run_id"],
            "excluded_technology":     res["excluded_tec"],
            "npv_first_best":          npv_orig,
            "npv_reopt":               npv_reopt,
            "delta_npv":               delta_abs,
            "delta_npv_pct":           delta_pct,
            "installed_tecs_original": ", ".join(sorted(run["installed_tecs"])),
            "removed_tech_sizes":      removed_tech_sizes,
            "error":                   res["error"] or "",
        })

    df = pd.DataFrame(rows).sort_values(["run_id", "excluded_technology"])

    out_csv   = output_folder / "reopt_comparison.csv"
    out_excel = output_folder / "reopt_comparison.xlsx"
    df.to_csv(out_csv, sep=";", index=False)
    df.to_excel(out_excel, index=False)

    print(f"  Saved: {out_csv.name}  ({len(df)} rows)")
    print(f"  Saved: {out_excel.name}")

    # ── Step 4b: Copy reopt run folders to output folder ──────────────────────
    dest_reopt = output_folder / "reopt_runs"
    if reopt_folder.resolve() != dest_reopt.resolve():
        print(f"\n[STEP 4b] Copying reopt run folders to output folder...")
        if dest_reopt.exists():
            shutil.rmtree(dest_reopt)
        shutil.copytree(reopt_folder, dest_reopt)
        print(f"  Copied: {reopt_folder} → {dest_reopt}")
    else:
        print(f"\n[STEP 4b] Reopt runs already in output folder, skipping copy.")

    # ── Step 5: Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY — avg NPV change when each technology is excluded")
    print(f"{'='*70}")
    ok = df[df["error"] == ""]
    if not ok.empty:
        summary = (
            ok.groupby("excluded_technology")["delta_npv_pct"]
            .agg(["mean", "min", "max", "count"])
            .rename(columns={"mean": "avg_Δ%", "min": "min_Δ%", "max": "max_Δ%", "count": "n_runs"})
            .sort_values("avg_Δ%")
        )
        print(summary.to_string(float_format=lambda x: f"{x:+.2f}"))
    failed = df[df["error"] != ""]
    if not failed.empty:
        print(f"\n  Failed jobs: {len(failed)}")
        for _, row in failed.iterrows():
            print(f"    {row['run_id']} / {row['excluded_technology']}: {row['error']}")
    print(f"{'='*70}\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import platform

    if platform.system() == "Linux":
        # ── Snellius HPC entry point ──────────────────────────────────────────
        task_id      = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        home         = Path(os.environ["HOME"])
        base_results = home / "AdOpT-NET0-RegToNation/four_node_configuration/results/parallel_run_20260523_190614"

        TASK_FOLDERS = [
            base_results / "task_000",
            base_results / "task_001",
        ]
        RESULTS_FOLDER     = TASK_FOLDERS[task_id]
        MAX_WORKERS        = 63
        GUROBI_THREADS     = 3
        LOCAL_REOPT_FOLDER = Path(os.environ.get("TMPDIR", "/tmp")) / "reopt_runs"
        OUTPUT_FOLDER      = base_results / f"reopt_comparison_task_{task_id:03d}"

        print(f"[HPC] SLURM_ARRAY_TASK_ID = {task_id}")
        print(f"[HPC] Results folder      = {RESULTS_FOLDER}")
        print(f"[HPC] Output folder       = {OUTPUT_FOLDER}")
        print(f"[HPC] Temp reopt folder   = {LOCAL_REOPT_FOLDER}")
        print(f"[HPC] Workers             = {MAX_WORKERS} x {GUROBI_THREADS} threads", flush=True)

    else:
        # ── Windows / local entry point ───────────────────────────────────────
        RESULTS_FOLDER     = r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\3000_simulations_24_may_2026\Snellius_3000_simulations"
        MAX_WORKERS        = 15
        GUROBI_THREADS     = 3
        LOCAL_REOPT_FOLDER = r"C:\Temp\reopt_runs"
        OUTPUT_FOLDER      = None  # defaults to results_folder/../reopt_comparison_<name>

    run_reopt_comparison(
        results_folder=RESULTS_FOLDER,
        output_folder=OUTPUT_FOLDER,
        local_reopt_folder=LOCAL_REOPT_FOLDER,
        max_workers=MAX_WORKERS,
        gurobi_threads=GUROBI_THREADS,
    )