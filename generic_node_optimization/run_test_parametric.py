"""
Test Parametric Study - Quick Test Script
Esegue un numero ridotto di combinazioni per testare il sistema
"""

import json
import itertools
import time
from pathlib import Path
import pandas as pd
from datetime import datetime
from Run_optimization import OptimizationRunner


def main():
    base_path = Path(__file__).parent

    # =========================================================================
    # PARAMETRI RIDOTTI PER TEST
    # =========================================================================

    param_grid = {
        # Un solo scenario per test
        "scenarios": ["1751"],

        # Pochi valori per test rapido
        "demand_level_ratio": [10],
        "total_demand_TWh": [25],
        "import_availability_ratio": [0.6],
        "import_cost_multiplier": [2],
        "electricity_price_avg": [80],
        "electricity_availability_small": [80],
        "willingness_to_pay": [350],

        # Una sola configurazione network
        "networks": [
            ["hydrogenPipelineOnshore_lowP", "hydrogenPipelineOnshore_highP"]
        ],

        # Components in nodes
        "small_cluster_new_technologies": [["Electrolyzer_small", "Storage_H2_lowP"]],
        "big_cluster_new_technologies": [["Electrolyzer_big", "Storage_H2_highP"]],
        "existing_storage_technologies": [{"Storage_H2_Cavern": 100000}],

        "hydrogen_demand_small":    [{
        "SMALL1": {"Hydrogen use (TWh)": 0.95, "Capacity (MW)": 174},
        "SMALL2": {"Hydrogen use (TWh)": 0.64, "Capacity (MW)": 117},
        "SMALL3": {"Hydrogen use (TWh)": 0.48, "Capacity (MW)": 87},
        "SMALL4": {"Hydrogen use (TWh)": 0.18, "Capacity (MW)": 33}}],

        "hydrogen_demand_big":[{
        "BIG1": {"Hydrogen use (TWh)": 15},
        "BIG2": {"Hydrogen use (TWh)": 5}}],

        # Solver parameters
        "mipgap": [0.01],
        "time_limit": [50],
        "threads": [10]
    }

    # Genera combinazioni
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"{'='*80}")
    print(f"TEST PARAMETRIC STUDY")
    print(f"{'='*80}")
    print(f"Total combinations to test: {len(combinations)}")

    for key, val_list in param_grid.items():
        print(f"  {key}: {val_list}")

    # Calcola tempo stimato
    avg_solve_time = 10  # secondi
    total_time_min = (len(combinations) * avg_solve_time) / 60
    print(f"\n⏱️  Estimated time: {total_time_min:.1f} minutes")

    # =========================================================================
    # CREATE RESULTS FOLDER
    # =========================================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = base_path / "results" / f"test_parametric_{timestamp}"
    results_folder.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(results_folder / "param_config.json", "w") as f:
        json.dump({"param_grid": param_grid, "total_runs": len(combinations)}, f, indent=4)

    print(f"\n📁 Results folder: {results_folder}")


    # =========================================================================
    # RUN OPTIMIZATIONS
    # =========================================================================

    print(f"\n{'='*80}")
    print(f"STARTING TEST")
    print(f"{'='*80}\n")

    start_time = time.time()
    runner = OptimizationRunner(base_path)
    results_summary = []

    for idx, params in enumerate(combinations, 1):
        print(f"\n{'='*80}")
        print(f"TEST RUN {idx}/{len(combinations)}")
        print(f"{'='*80}")

        try:
            result = runner.run_optimization(
                run_id=f"test_run_{idx:04d}",
                params=params,
                results_base_folder=results_folder
            )

            # Guard: ensure result is a dict before using **result
            if result is None:
                print(f"⚠️  runner.run_optimization returned None for run {idx}; using empty result dict")
                result = {}
            elif not isinstance(result, dict):
                try:
                    result = dict(result)
                except Exception:
                    print(f"⚠️  Unexpected result type {type(result)} for run {idx}; using empty result dict")
                    result = {}

            results_summary.append({
                "run_id": f"test_run_{idx:04d}",
                "status": "SUCCESS",
                **params,
                **result
            })

        except Exception as e:
            print(f"\n❌ ERROR during test run {idx}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()

            results_summary.append({
                "run_id": f"test_run_{idx:04d}",
                "status": "FAILED",
                "error": str(e),
                **params
            })

    elapsed = time.time() - start_time

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    df_summary = pd.DataFrame(results_summary)
    df_summary.to_csv(results_folder / "test_results_summary.csv", index=False, sep=';')
    df_summary.to_excel(results_folder / "test_results_summary.xlsx", index=False)

    successful = sum(1 for r in results_summary if r['status'] == 'SUCCESS')

    summary_text = f"""
================================================================================
TEST PARAMETRIC STUDY SUMMARY
================================================================================

Total Test Runs: {len(results_summary)}
Successful: {successful}
Failed: {len(results_summary) - successful}
print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Average time per run: {elapsed/len(combinations):.1f}s")

Results folder: {results_folder}

If the test was successful, you can now run the full parametric study with:
    python run_parametric_study.py

================================================================================
"""

    (results_folder / "TEST_SUMMARY.txt").write_text(summary_text)

    print(summary_text)

    if successful == len(results_summary):
        print("✅ All test runs completed successfully!")
        print("   You can now run the full parametric study.")
    else:
        print(f"⚠️  {len(results_summary) - successful} test runs failed.")
        print("   Check the error messages above and fix issues before running the full study.")


if __name__ == "__main__":
    main()

