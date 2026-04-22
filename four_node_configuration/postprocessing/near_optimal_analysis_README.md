# Near-Optimal Re-Optimization Analysis

## Goal

For each first-best optimization run in a results folder, re-run the simulation
**excluding one installed technology at a time**, then compare the new NPV to the original.

---

## How to run

1. Set the parameters at the bottom of `near_optimal_analysis.py`:

```python
RESULTS_FOLDER     = r"\\server\...\4_simulations_for_rubustness"
MAX_WORKERS        = 4    # parallel processes
GUROBI_THREADS     = 12   # Gurobi threads per process (None = inherit from run_params.json)
LOCAL_REOPT_FOLDER = r"C:\Temp\reopt_runs"   # must be a short local path
```

2. Run:

```bash
python four_node_configuration/postprocessing/near_optimal_analysis.py
```

---

## Pipeline

### Step 1 — Read first-best results

For every subfolder in `RESULTS_FOLDER` that contains a `run_params.json`:
- Reads the **NPV** from `optimization_results_summary.json`
- Reads the **h5** file to find which technologies/arcs have `size > 0`
- Stores: `run_id`, `params`, `objective (NPV)`, `installed_tecs`, `installed_arcs`

### Step 2 — Build jobs

One job per **(run × installed technology)**.  
Example: 4 runs × 3 installed techs = **12 jobs**.

Each job carries: `run_id`, `params`, `excluded_tec`, `installed_arcs`, paths, `gurobi_threads`.

### Step 3 — Re-optimize in parallel

Each job runs in a **spawned subprocess** (`ProcessPoolExecutor`).  
The model is always created with **full params** so all JSON files exist.  
The exclusion is applied by patching input files **after** template creation but **before** solving:

| Technology type | Exclusion method |
|---|---|
| **Node** (Electrolyzer, Storage, PV) | `size_max = size_min = 0` patched into the technology JSON |
| **Network** (hydrogen pipeline) | `size_max_arcs = 0` set **only for the arcs that were installed** in the first-best, in `size_max_arcs.csv` |

### Step 4 — Build comparison table

One row per job, saved to `reopt_comparison.csv` and `reopt_comparison.xlsx` in the output folder.

| Column | Description |
|---|---|
| `run_id` | Original first-best run identifier |
| `excluded_technology` | Which technology was excluded |
| `npv_first_best` | NPV of the original run |
| `npv_reopt` | NPV of the re-optimized run |
| `delta_npv` | Absolute difference (reopt − first-best) |
| `delta_npv_pct` | % change relative to first-best |
| `installed_tecs_original` | All technologies installed in the first-best |
| `error` | Error message if the re-opt failed |

### Step 5 — Summary

Prints average `delta_npv_pct` per technology across all runs.  
A larger positive `Δ%` means the system could not compensate well → the technology is **more important**.

---

## Output files

| File | Location | Description |
|---|---|---|
| `reopt_comparison.csv` | Network output folder | Full comparison table (`;` separated) |
| `reopt_comparison.xlsx` | Network output folder | Same as above in Excel format |
| `reopt_runs/` | `LOCAL_REOPT_FOLDER` | Intermediate input/result data for each re-opt run |

> **Note:** intermediate files are stored locally (`C:\Temp\reopt_runs`) to avoid
> Windows `WinError 206` (path too long) when writing to a UNC network path.

---

## Key configuration constants

```python
# Technology → parameter key mapping
TEC_PARAM_MAP = {
    "Electrolyzer_small":            "small_cluster_new_technologies",
    "Storage_H2_lowP":               "small_cluster_new_technologies",
    "Photovoltaic":                  "small_cluster_new_technologies",
    "Electrolyzer_big":              "big_cluster_new_technologies",
    "Storage_H2_highP":              "big_cluster_new_technologies",
    "hydrogenPipelineOnshore_lowP":  "networks_new",
    "hydrogenPipelineOnshore_highP": "networks_new",
}

# Technologies never excluded (existing / fixed infrastructure)
EXCLUDE_FROM_ANALYSIS = {"Storage_H2_Cavern", "Storage_H2_Cavern_existing"}

# Node names used to parse arc keys from h5 (concatenated format: "NodeANodeB")
ALL_NODES = ["Large_cluster1", "Large_cluster2", "Small_cluster1", "Small_cluster2"]
```

---

## Thread / CPU usage

With `MAX_WORKERS = 4` and `GUROBI_THREADS = 12`:

```
4 processes × 12 Gurobi threads = 48 threads total
```

Set `GUROBI_THREADS = None` to inherit the thread count from the original `run_params.json`.

---

## What this script does NOT do (yet)

- No near-optimality threshold filtering
- No technology relevance flags (`_relevant` columns)

These can be added later using a `get_relevant_cases(threshold, df, tec_cols)` function
applied to the output `reopt_comparison.csv`.

