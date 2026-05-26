# Near-Optimal Analysis

Re-optimizes each first-best run with selected technologies excluded, then compares NPV to quantify the cost of removing small-cluster investments.

## What it does

1. Reads all first-best runs from a results folder (scans for `run_params.json` + `optimization_results.h5`)
2. For each run where `Electrolyzer_small > 1 MW`: re-solves with `Electrolyzer_small`, `Storage_H2_lowP`, and `Photovoltaic` all excluded (no small-cluster scenario)
3. Compares new NPV vs first-best NPV
4. Saves `reopt_comparison.csv` and `reopt_comparison.xlsx` + full reopt run folders

## Configuration (top of `near_optimal_analysis.py`)

| Variable | Purpose |
|---|---|
| `EXCLUDE_FROM_ANALYSIS` | Technologies never excluded (fixed/existing) |
| `COMBO_JOBS` | Groups of techs excluded together in one reopt |
| `COMBO_TRIGGER_TECHS` | Which techs in each combo must be installed to trigger it |
| `INSTALLATION_THRESHOLDS` | Min size (MW) to consider a tech as "installed" for trigger check |
| `SKIP_SINGLE_EXCLUSIONS` | Techs skipped for individual single-exclusion jobs |
| `REOPT_NETWORKS` | Whether to run network arc exclusion jobs (currently `False`) |

### Current logic

Combo triggers if `Electrolyzer_small > 1 MW` in either small cluster → excludes all three small-cluster techs:
- `Electrolyzer_small`
- `Storage_H2_lowP`
- `Photovoltaic`

Runs with only `Storage_H2_lowP` or `Photovoltaic` installed (no electrolyzer) → **no reopt job**.

## Output

```
results/parallel_run_YYYYMMDD_HHMMSS/
└── reopt_comparison_task_000/
    ├── reopt_comparison.csv       ← main results table
    ├── reopt_comparison.xlsx
    └── reopt_runs/                ← full reopt run folders (input + results)
        ├── parallel_run_0001__excl_Electrolyzer_small+...
        └── ...
```

### Output columns

| Column | Description |
|---|---|
| `run_id` | First-best run ID |
| `excluded_technology` | Tech(s) excluded (joined by `+`) |
| `npv_first_best` | Original NPV (€) |
| `npv_reopt` | Re-optimized NPV (€) |
| `delta_npv` | Absolute difference (€) |
| `delta_npv_pct` | Relative difference (%) |
| `removed_tech_sizes` | Installed sizes of excluded techs |
| `error` | Error message if solve failed |

## Running locally (Windows)

```python
# Set at bottom of near_optimal_analysis.py:
RESULTS_FOLDER     = r"\\soliscom.uu.nl\geo\SD\...\Snellius_3000_simulations"
MAX_WORKERS        = 15
GUROBI_THREADS     = 3
LOCAL_REOPT_FOLDER = r"C:\Temp\reopt_runs"
```

```bash
python four_node_configuration/postprocessing/near_optimal_analysis.py
```

## Running on Snellius (HPC)

Results must already be on the cluster at:
```
$HOME/AdOpT-NET0-RegToNation/four_node_configuration/results/parallel_run_20260523_190614/
├── task_000/   (~1500 runs)
└── task_001/   (~1500 runs)
```

Submit with:
```bash
cd $HOME/AdOpT-NET0-RegToNation
git pull
sbatch run_near_optimal.sh
```

- Runs as SLURM array job (`--array=0-1`): one fat_genoa node per task folder
- 63 workers × 3 Gurobi threads = 189 CPUs per node
- Intermediate files → `$TMPDIR/reopt_runs` (fast node-local NVMe)
- Output → `results/parallel_run_20260523_190614/reopt_comparison_task_000/` and `_001/`
- Logs → `logs/reopt_<jobid>_0.out` and `_1.out`

After job completes, download output folders from Snellius (Open OnDemand or `scp`).

## Expected run count

~2269 reopt jobs total across both task folders (runs with `Electrolyzer_small > 1 MW`).
Estimated wall time: ~1.5h per node at 48 workers.