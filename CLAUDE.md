# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## What this is

**AdOpT-NET0** — Python package for multi-energy system optimization. Models conversion technologies and networks for any carrier; optimizes design and operation using Pyomo (MILP/MINLP). Python ≥3.12.

The main research application lives in `four_node_configuration/` — a four-node hydrogen system study run at scale on Snellius HPC.

---

## AdOpT-NET0 package

### Entry point

`adopt_net0/modelhub.py` — `ModelHub` class:
1. `ModelHub.data` (`DataHandle`) loads input data from folder structure
2. Builds Pyomo model via `model_construction/`
3. Solves and writes results via `result_management/`

### Architecture

```
ModelComponent  (component.py)
├── Technology  (technologies/technology.py)
│   ├── genericTechnologies/  — conv1..4 (conversion types), res, stor, sink
│   └── specificTechnologies/ — combined_cycle, dac_adsorption, gas_turbine, heat_pump, hydro_open
└── Network     (networks/network.py)
    ├── genericNetworks/      — simple, electricity, fluid
    └── specificNetworks/     — fixed_pipeline
```

`conv1`–`conv4` encode different input/output performance relationship types (linear, piecewise, etc.).

Key modules:
- `adopt_net0/data_management/handle_input_data.py` — `DataHandle`: reads case study folder into `topology`, `time_series`, `technology_data`, `network_data`, `model_config`
- `adopt_net0/model_construction/` — builds Pyomo blocks: nodes, technologies, networks, balances, investment periods
- `adopt_net0/result_management/` — saves/reads HDF5 results (`tables` library)
- `adopt_net0/database/` — cost/data models for specific techs (PV, wind, heat pump, DAC, CO2 pipelines)

### Key conventions

- Pyomo components (sets, params, vars, constraints) added inside component classes, not in `model_construction/`
- Time sets per technology: `set_t_full`, `set_t_performance`, `set_t_global` — linked via `link_full_resolution_to_clustered`
- CCS: optional add-on injecting extra carriers/variables without changing base class
- Disjunctive programming (Pyomo GDP) for binary on/off; `big_m_transformation_required` controls reformulation

### Commands

```bash
poetry install --without dev
poetry run pytest --cov
poetry run pytest tests/test_technologies.py::test_name -v
black adopt_net0/
```

Tests use GLPK by default; Gurobi preferred locally.

---

## Four-node configuration study

### Purpose

Hydrogen system with 4 nodes: 2 large clusters (big electrolyzer, high-P storage) + 2 small clusters (small electrolyzer, low-P storage, PV). Pipelines connect nodes. ~3000 parametric runs exploring different archetypes.

### Key files

| File | Purpose |
|------|---------|
| `FOUR_NODE_run_creation_and_gurobi_optimization_parallel.py` | Creates model instances + solves in parallel (`ProcessPoolExecutor`, spawn). Entry point for batch runs. `create_single_model` / `solve_single_model` are importable workers. |
| `postprocessing/near_optimal_analysis.py` | Near-optimal re-optimization: excludes installed techs, re-solves, compares NPV. Config at top of file. |
| `postprocessing/extract_cost_comparison.py` | Cost breakdown comparison between first-best and reopt run. |
| `run_optimization.sh` | SLURM script for batch runs on Snellius |
| `run_near_optimal.sh` | SLURM script for near-optimal analysis on Snellius |

### Results folder structure

```
results/parallel_run_YYYYMMDD_HHMMSS/
├── task_000/
│   ├── parallel_run_0001/    ← one folder per run
│   │   ├── run_params.json
│   │   ├── optimization_results.h5
│   │   └── optimization_results_summary.json
│   └── parallel_results_summary.csv
└── task_001/
    └── ...
```

### Network drive

`X:` maps to `\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results` — always use `X:` to avoid UNC path issues.

### Technology mapping

| Technology | Cluster | Notes |
|---|---|---|
| `Electrolyzer_small`, `Storage_H2_lowP`, `Photovoltaic` | Small_cluster1/2 | `small_cluster_new_technologies` |
| `Electrolyzer_big`, `Storage_H2_highP` | Large_cluster1/2 | `big_cluster_new_technologies` |
| `hydrogenPipelineOnshore_lowP/highP` | network arcs | `networks_new` |
| `Storage_H2_Cavern`, `Storage_H2_Cavern_existing` | Large_cluster1 | existing, never excluded |

### Near-optimal analysis config (near_optimal_analysis.py)

- `EXCLUDE_FROM_ANALYSIS` — techs never excluded
- `COMBO_JOBS` — groups excluded together; paired with `COMBO_TRIGGER_TECHS` (which tech must be installed to trigger)
- `INSTALLATION_THRESHOLDS` — min MW to count as installed (default 0)
- `REOPT_NETWORKS` — enables no-network scenario; only triggers if `run["installed_arcs"]` non-empty
- `SKIP_SINGLE_EXCLUSIONS` — techs skipped for individual jobs

### HPC — Snellius

- Branch: `working_snellius`
- Partition: `fat_genoa` (192 CPU, 1440 GB RAM, factor 1.5×) — only available partition for CPU jobs
- Config: `63 workers × 3 Gurobi threads = 189 cores/node`
- Temp files: `$TMPDIR/reopt_runs` (node-local NVMe, fast I/O)
- Results saved to `$HOME/.../results/`, download via Open OnDemand or scp
- Submit: `sbatch run_optimization.sh` or `sbatch run_near_optimal.sh`
- Monitor: `squeue -u $USER`, `seff <jobid>` (after completion, shows peak RAM)
- Credits: `accinfo`
- Observed peak RAM: ~135 GB for 3000-run batch (fits in fat_genoa 1440 GB easily)