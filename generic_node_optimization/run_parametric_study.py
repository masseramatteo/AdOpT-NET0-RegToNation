"""
Run Parametric Study - Main Script
Executes multiple optimization runs varying input parameters

This script systematically explores the parameter space defined by:
- DEMAND LEVELS: ratio between big/small clusters
- TOTAL DEMAND: sum of all demands in TWh
- IMPORT AVAILABILITY: total import availability over total demand
- IMPORT COST: cost of H2 import relative to electricity
- ELECTRICITY PRICE: average cost of electricity
- ELECTRICITY AVAILABILITY: import limits for small clusters
"""

import json
import itertools
from pathlib import Path
import pandas as pd
from datetime import datetime
from optimization_runner import OptimizationRunner
import random
import numpy as np

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy not installed. Will use random sampling instead of LHC.")


def latin_hypercube_sample(param_grid, n_samples):
    """
    Generate parameter combinations using Latin Hypercube Sampling (LHC)

    LHC ensures better coverage of parameter space compared to random sampling:
    - Each parameter dimension is divided into n_samples intervals
    - One sample is taken from each interval
    - Provides more uniform distribution across parameter space

    Args:
        param_grid: Dictionary of parameters and their possible values
        n_samples: Number of samples to generate

    Returns:
        List of parameter dictionaries
    """

    # Separate continuous and discrete parameters
    param_names = []
    param_values = []
    param_types = []  # 'continuous' or 'discrete'

    for param_name, values in param_grid.items():
        param_names.append(param_name)
        param_values.append(values)

        # Determine if parameter is continuous (numeric) or discrete (list/categorical)
        if isinstance(values[0], (int, float)) and len(values) > 2:
            param_types.append('continuous')
        else:
            param_types.append('discrete')

    n_params = len(param_names)

    # Generate LHC samples in [0, 1]^n_params space
    sampler = qmc.LatinHypercube(d=n_params, seed=42)
    lhc_samples = sampler.random(n=n_samples)

    # Map LHC samples to actual parameter values
    combinations = []

    for sample in lhc_samples:
        param_dict = {}

        for i, (param_name, values, param_type) in enumerate(zip(param_names, param_values, param_types)):
            # Map [0, 1] to parameter range
            if param_type == 'continuous':
                # For continuous/numeric: interpolate between min and max
                min_val = min(values)
                max_val = max(values)
                # Round to nearest value in the list
                scaled_value = min_val + sample[i] * (max_val - min_val)
                param_dict[param_name] = min(values, key=lambda x: abs(x - scaled_value))
            else:
                # For discrete/categorical: map to index
                idx = int(sample[i] * len(values))
                if idx >= len(values):  # Edge case
                    idx = len(values) - 1
                param_dict[param_name] = values[idx]

        combinations.append(param_dict)

    return combinations


def main():
    base_path = Path(__file__).parent

    # =========================================================================
    # DEFINE PARAMETRIC GRID
    # =========================================================================

    param_grid = {
        # Geographic scenarios
        "scenarios": ["1751", "1752", "1753"],

        # DEMAND LEVELS: ratio big/small clusters
        # BIG clusters (2 nodes) vs SMALL clusters (4 nodes)
        # Values: [3, 5, 7, 10, 15, 20]
        "demand_level_ratio": [3, 5, 7, 10, 15, 20],

        # TOTAL DEMAND: sum of all demands in TWh annually
        # Values: [10, 20, 30, 50, 100] TWh
        "total_demand_TWh": [10, 20, 30, 50, 100],

        # IMPORT AVAILABILITY: total import / total demand
        # Values: [0.2, 0.5, 0.7, 1.0]
        "import_availability_ratio": [0.2, 0.5, 0.7, 1.0],

        # IMPORT COST: H2 import cost multiplier relative to electricity
        # (considering 65% efficiency: 1 MWh elec -> 0.65 MWh H2)
        # Values: [0.5, 1, 1.5, 2, 3, 5]
        "import_cost_multiplier": [0.5, 1, 1.5, 2, 3, 5],

        # ELECTRICITY PRICE: average EUR/MWh (fixed for BIG, varies for SMALL)
        # Values: [50, 100, 150, 200]
        "electricity_price_avg": [50, 100, 150, 200],

        # ELECTRICITY AVAILABILITY: for small clusters in MW
        # BIG clusters fixed at 2000 MW
        # Values: [30, 50, 100, 200, 400]
        "electricity_availability_small": [30, 50, 100, 200, 400],

        # Networks to test
        "networks": [
            ["hydrogenPipelineOnshore_lowP", "hydrogenPipelineOnshore_highP"],
            ["hydrogenPipelineOnshore_highP"]  # Only high pressure
        ],

        # Solver parameters
        "mipgap": [0.01],
        "time_limit": [100],
        "threads": [48]
    }

    # =========================================================================
    # GENERATE COMBINATIONS
    # =========================================================================

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"{'='*80}")
    print(f"PARAMETRIC STUDY SETUP")
    print(f"{'='*80}")
    print(f"Total theoretical combinations: {len(all_combinations)}")

    # Calculate expected combinations
    total_theoretical = 1
    for key, val_list in param_grid.items():
        print(f"  {key}: {len(val_list)} values")
        total_theoretical *= len(val_list)

    print(f"\nTotal combinations to test: {total_theoretical}")

    # Optional: Sample if too many combinations
    MAX_RUNS = 1000  # Limit total runs (change this value as needed)
    # Examples: 500 (fast), 2000 (detailed), 5000 (comprehensive), None (no limit)

    USE_LHC = True  # Use Latin Hypercube Sampling (better coverage than random)

    if MAX_RUNS is not None and len(all_combinations) > MAX_RUNS:
        print(f"\n⚠️  Too many combinations ({len(all_combinations)})")

        if USE_LHC and HAS_SCIPY:
            print(f"   Using Latin Hypercube Sampling for {MAX_RUNS} combinations...")
            print(f"   (Better parameter space coverage than random sampling)")
            combinations = latin_hypercube_sample(param_grid, MAX_RUNS)
        else:
            print(f"   Using random sampling for {MAX_RUNS} combinations...")
            random.seed(42)  # For reproducibility
            combinations = random.sample(all_combinations, MAX_RUNS)
    else:
        combinations = all_combinations

    print(f"\n✅ Will execute {len(combinations)} optimization runs")

    # Estimate time
    avg_solve_time = 100  # seconds per run (estimate)
    total_time_hours = (len(combinations) * avg_solve_time) / 3600
    print(f"⏱️  Estimated time: {total_time_hours:.1f} hours")

    # =========================================================================
    # CREATE RESULTS FOLDER
    # =========================================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = base_path / "results" / f"parametric_study_{timestamp}"
    results_folder.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_info = {
        "param_grid": param_grid,
        "total_theoretical_combinations": total_theoretical,
        "tested_combinations": len(combinations),
        "timestamp": timestamp,
        "sampled": len(all_combinations) > MAX_RUNS
    }

    with open(results_folder / "param_config.json", "w") as f:
        json.dump(config_info, f, indent=4)

    print(f"\n📁 Results folder: {results_folder}")

    # Ask for confirmation
    response = input("\n▶️  Press ENTER to start, or 'q' to quit: ")
    if response.lower() == 'q':
        print("❌ Aborted by user")
        return

    # =========================================================================
    # RUN OPTIMIZATIONS
    # =========================================================================

    print(f"\n{'='*80}")
    print(f"STARTING PARAMETRIC STUDY")
    print(f"{'='*80}\n")

    runner = OptimizationRunner(base_path)
    results_summary = []

    for idx, params in enumerate(combinations, 1):
        print(f"\n{'='*80}")
        print(f"RUN {idx}/{len(combinations)}")
        print(f"{'='*80}")
        print(f"Scenario: {params['scenarios']}")
        print(f"Total Demand: {params['total_demand_TWh']} TWh")
        print(f"Demand Ratio: {params['demand_level_ratio']}")
        print(f"Import Avail: {params['import_availability_ratio']}")
        print(f"Import Cost Mult: {params['import_cost_multiplier']}")
        print(f"Elec Price: {params['electricity_price_avg']} EUR/MWh")
        print(f"Elec Avail (small): {params['electricity_availability_small']} MW")
        print(f"{'='*80}\n")

        try:
            result = runner.run_optimization(
                run_id=f"run_{idx:04d}",
                params=params,
                results_base_folder=results_folder
            )

            results_summary.append({
                "run_id": f"run_{idx:04d}",
                "status": "SUCCESS",
                **params,
                **result
            })

            # Save intermediate results after each run
            df_summary = pd.DataFrame(results_summary)
            df_summary.to_csv(results_folder / "results_summary_partial.csv",
                            index=False, sep=';')

        except Exception as e:
            print(f"\n❌ ERROR during run {idx}")
            print(f"   Error message: {str(e)}")
            import traceback
            traceback.print_exc()

            results_summary.append({
                "run_id": f"run_{idx:04d}",
                "status": "FAILED",
                "error": str(e),
                **params,
                "objective_value": None,
                "solve_time": None,
                "termination_condition": "ERROR",
                "gap": None
            })

            # Save partial results
            df_summary = pd.DataFrame(results_summary)
            df_summary.to_csv(results_folder / "results_summary_partial.csv",
                            index=False, sep=';')

    # =========================================================================
    # SAVE FINAL RESULTS
    # =========================================================================

    print(f"\n{'='*80}")
    print(f"SAVING FINAL RESULTS")
    print(f"{'='*80}\n")

    df_summary = pd.DataFrame(results_summary)

    # Save as CSV and Excel
    df_summary.to_csv(results_folder / "results_summary.csv", index=False, sep=';')
    df_summary.to_excel(results_folder / "results_summary.xlsx", index=False)

    # Create analysis summary
    successful_runs = sum(1 for r in results_summary if r['status'] == 'SUCCESS')
    failed_runs = len(results_summary) - successful_runs

    summary_text = f"""
================================================================================
PARAMETRIC STUDY SUMMARY
================================================================================
Timestamp: {datetime.now()}
Results Folder: {results_folder}

EXECUTION STATISTICS
--------------------
Total Runs: {len(results_summary)}
Successful: {successful_runs}
Failed: {failed_runs}
Success Rate: {100*successful_runs/len(results_summary):.1f}%

PARAMETER SPACE
---------------
"""

    for key, values in param_grid.items():
        summary_text += f"{key}: {values}\n"

    summary_text += f"""
RESULTS FILES
-------------
- results_summary.csv: Full results in CSV format
- results_summary.xlsx: Full results in Excel format
- param_config.json: Parameter grid configuration
- run_XXXX/: Individual run folders with detailed results

================================================================================
"""

    (results_folder / "SUMMARY.txt").write_text(summary_text)

    print(summary_text)
    print(f"✅ Parametric study completed!")
    print(f"📊 Results saved in: {results_folder}")


if __name__ == "__main__":
    main()

