"""
Parallel Optimization Runner
Creates all models first, then solves them in parallel using multiprocessing
"""

import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pandas as pd
from datetime import datetime
from Run_optimization import OptimizationRunner


def create_single_model(args):
    """Create and prepare a single model without solving (runs in main process)"""
    run_id, params, results_base_folder, base_path = args

    print(f"📦 Creating model: {run_id}")

    try:
        runner = OptimizationRunner(base_path)
        run_folder = results_base_folder / run_id
        input_data_path = run_folder / "input_data"
        results_data_path = run_folder / "userData"

        input_data_path.mkdir(parents=True, exist_ok=True)
        results_data_path.mkdir(parents=True, exist_ok=True)

        # Save parameters
        with open(run_folder / "run_params.json", "w") as f:
            json.dump(params, f, indent=4)

        # Setup base
        import adopt_net0 as adopt

        adopt.create_optimization_templates(input_data_path)
        nodes = ["BIG1", "BIG2", "SMALL1", "SMALL2", "SMALL3", "SMALL4", "STORAGE"]

        runner.timesteps = 8760
        runner.wtp = params["willingness_to_pay"]
        runner.electricity_average_price = params["electricity_price_avg"]
        runner.electricity_availability_small = params["electricity_availability_small"]

        # Define topology
        runner._configure_topology(input_data_path, nodes)

        # Calculate derived parameters
        derived_params = runner._calculate_derived_parameters(params)
        params.update(derived_params)
        runner.h2_import_limit = params["h2_import_limit"]
        runner.h2_import_price = params["hydrogen_import_price"]

        # Configure model
        runner._configure_model(input_data_path, results_data_path, params)

        adopt.create_input_data_folder_template(input_data_path)

        # Load scenario nodes
        runner._load_scenario_nodes(input_data_path, nodes, params["scenarios"])

        # Define and characterize nodes
        from define_topology import define_nodes
        define_nodes(input_data_path, params)

        # Copy technology files
        adopt.copy_technology_data(input_data_path)
        adopt.copy_compressor_data(input_data_path)

        # Networks configuration
        runner._configure_networks(input_data_path, params["networks"])
        adopt.copy_network_data(input_data_path)

        from define_topology import add_new_network_H2
        add_new_network_H2(input_data_path, params["scenarios"])

        from define_components_spec import (
            define_hydrogen_pipeline2,
            define_hydrogen_storage,
            define_electrolyzers
        )
        define_hydrogen_pipeline2(input_data_path)
        define_hydrogen_storage(input_data_path)
        define_electrolyzers(input_data_path)

        # Load carrier data
        runner._load_carrier_data(input_data_path, nodes, params)

        print(f"✅ Model created: {run_id}")
        return run_id, str(input_data_path), str(results_data_path), params, None

    except Exception as e:
        print(f"❌ Error creating model {run_id}: {e}")
        import traceback
        traceback.print_exc()
        return run_id, None, None, params, str(e)


def solve_single_model(args):
    """Solve a single model (runs in separate process)"""
    run_id, input_data_path, results_data_path, params = args

    try:
        print(f"🔄 Solving: {run_id}")

        import adopt_net0 as adopt
        import pyomo.environ as pyo

        input_data_path = Path(input_data_path)
        results_data_path = Path(results_data_path)

        # Create ModelHub and read data
        m = adopt.ModelHub()
        m.read_data(input_data_path, start_period=0, end_period=1)

        # Solve
        m.quick_solve()

        # Extract results
        result_folder_path = Path(m.last_solve_info["result_folder_path"])

        # Save params info
        text = json.dumps(params, indent=4, ensure_ascii=False, default=str)
        (result_folder_path / "optimization_model_info.txt").write_text(text, encoding="utf-8")

        # Extract results - inline extraction since we can't import runner methods
        model = None
        if hasattr(m, "model"):
            if isinstance(m.model, dict):
                model = m.model.get("full") or next(iter(m.model.values()))
            else:
                model = m.model

        result_info = {}

        if model is not None:
            def get_value(component_name, pyo_type):
                try:
                    comp = getattr(model, component_name, None)
                    if comp is not None:
                        val = getattr(comp, "value", None)
                        if val is not None:
                            return float(val)
                        try:
                            first_idx = next(iter(comp))
                            return float(comp[first_idx])
                        except:
                            pass
                except Exception:
                    pass
                return None

            var_npv = get_value("var_npv", pyo.Var)
            para_total_demand = get_value("para_total_demand", pyo.Param)

            wtp = params.get("willingness_to_pay")
            if isinstance(wtp, (list, tuple)):
                wtp = wtp[0] if len(wtp) > 0 else None
            try:
                wtp = float(wtp) if wtp is not None else None
            except:
                wtp = None

            npv_over_demand = None
            if var_npv is not None and var_npv != 0 and para_total_demand is not None:
                npv_over_demand = var_npv / para_total_demand

            demand_times_wtp = None
            if para_total_demand is not None and wtp is not None:
                demand_times_wtp = para_total_demand * wtp

            objective_value = None
            try:
                for obj in model.component_objects(pyo.Objective, active=True):
                    objective_value = pyo.value(obj)
                    break
            except:
                pass

            result_info = {
                "var_npv": var_npv,
                "para_total_demand": para_total_demand,
                "npv_over_demand": npv_over_demand,
                "demand_times_wtp": demand_times_wtp,
                "willingness_to_pay": wtp,
                "objective_value": objective_value,
            }

            # Save to JSON
            result_file = result_folder_path / "optimization_results_summary.json"
            with open(result_file, "w") as f:
                json.dump(result_info, f, indent=4, default=str)

        print(f"✅ Solved: {run_id}")
        return run_id, result_info, None

    except Exception as e:
        print(f"❌ Error solving {run_id}: {e}")
        import traceback
        traceback.print_exc()
        return run_id, None, str(e)


class ParallelOptimizationRunner:
    """Run multiple optimizations in parallel"""

    def __init__(self, base_path, max_workers=None):
        self.base_path = Path(base_path)

        # Auto-detect optimal worker count
        if max_workers is None:
            cpu_count = mp.cpu_count()
            # Use half of logical processors, with min 2 and max 8
            max_workers = max(2, min(8, cpu_count // 2))
            print(f"💡 Auto-detected {cpu_count} logical processors → using {max_workers} parallel workers")

        self.max_workers = max_workers

    def run_parallel_optimization(self, run_configs, results_base_folder):
        """
        Run multiple optimizations in parallel

        Args:
            run_configs: List of (run_id, params) tuples
            results_base_folder: Base folder for results

        Returns:
            results_summary: List of dictionaries with results for each run
        """
        results_base_folder = Path(results_base_folder)
        results_base_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"PARALLEL OPTIMIZATION RUNNER")
        print(f"{'='*80}")
        print(f"Total runs: {len(run_configs)}")
        print(f"Max parallel workers: {self.max_workers}")
        print(f"Results folder: {results_base_folder}")
        print(f"{'='*80}\n")

        # =====================================================================
        # PHASE 1: CREATE ALL MODELS (SEQUENTIAL)
        # =====================================================================
        print(f"\n{'='*80}")
        print(f"PHASE 1: Creating {len(run_configs)} models (sequential)")
        print(f"{'='*80}\n")

        model_configs = []
        creation_errors = []

        for run_id, params in run_configs:
            args = (run_id, params, results_base_folder, self.base_path)
            run_id, input_path, results_path, params, error = create_single_model(args)

            if error:
                creation_errors.append({
                    "run_id": run_id,
                    "status": "CREATION_FAILED",
                    "error": error,
                    **params
                })
            else:
                model_configs.append((run_id, input_path, results_path, params))

        print(f"\n✅ Model creation complete: {len(model_configs)} models ready to solve")
        if creation_errors:
            print(f"⚠️  {len(creation_errors)} models failed to create")

        # =====================================================================
        # PHASE 2: SOLVE ALL MODELS (PARALLEL)
        # =====================================================================
        print(f"\n{'='*80}")
        print(f"PHASE 2: Solving {len(model_configs)} models in parallel")
        print(f"{'='*80}\n")

        start_time = time.time()
        results_summary = []
        solve_errors = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all solve jobs
            future_to_runid = {
                executor.submit(solve_single_model, config): config[0]
                for config in model_configs
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_runid):
                run_id = future_to_runid[future]
                completed += 1

                try:
                    run_id, result_info, error = future.result()

                    # Find original params
                    params = None
                    for config in model_configs:
                        if config[0] == run_id:
                            params = config[3]
                            break

                    if error:
                        solve_errors.append({
                            "run_id": run_id,
                            "status": "SOLVE_FAILED",
                            "error": error,
                            **(params or {})
                        })
                    else:
                        results_summary.append({
                            "run_id": run_id,
                            "status": "SUCCESS",
                            **(params or {}),
                            **(result_info or {})
                        })

                    print(f"Progress: {completed}/{len(model_configs)} completed")

                except Exception as e:
                    print(f"❌ Exception processing {run_id}: {e}")
                    # Find original params
                    params = None
                    for config in model_configs:
                        if config[0] == run_id:
                            params = config[3]
                            break

                    solve_errors.append({
                        "run_id": run_id,
                        "status": "SOLVE_FAILED",
                        "error": str(e),
                        **(params or {})
                    })

        elapsed = time.time() - start_time

        # Add creation errors to summary
        results_summary.extend(creation_errors)
        results_summary.extend(solve_errors)

        # =====================================================================
        # SAVE RESULTS SUMMARY
        # =====================================================================
        df_summary = pd.DataFrame(results_summary)
        df_summary.to_csv(results_base_folder / "parallel_results_summary.csv", index=False, sep=';')
        df_summary.to_excel(results_base_folder / "parallel_results_summary.xlsx", index=False)

        # =====================================================================
        # PRINT FINAL SUMMARY
        # =====================================================================
        successful = sum(1 for r in results_summary if r.get('status') == 'SUCCESS')
        failed = len(results_summary) - successful

        print(f"\n{'='*80}")
        print(f"PARALLEL OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total runs: {len(run_configs)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"  - Creation failures: {len(creation_errors)}")
        print(f"  - Solve failures: {len(solve_errors)}")
        print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Average time per run: {elapsed/len(run_configs):.1f}s")
        print(f"\n📁 Results saved to: {results_base_folder}")
        print(f"{'='*80}\n")

        if failed > 0:
            print("⚠️  Failed runs:")
            for item in (creation_errors + solve_errors):
                print(f"  {item['run_id']}: {item.get('error', 'Unknown error')}")

        return results_summary


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    import itertools

    base_path = Path(__file__).parent

    # Example parameter grid (small test)
    param_grid = {
        "scenarios": ["1751"],
        "demand_level_ratio": [10, 20],
        "total_demand_TWh": [25, 50],
        "import_availability_ratio": [0.5, 0.8],
        "import_cost_multiplier": [2],
        "electricity_price_avg": [80],
        "electricity_availability_small": [80],
        "willingness_to_pay": [250, 350],
        "networks": [["hydrogenPipelineOnshore_lowP", "hydrogenPipelineOnshore_highP"]],
        "small_cluster_new_technologies": [["Electrolyzer_small", "Storage_H2_lowP"]],
        "big_cluster_new_technologies": [["Electrolyzer_big", "Storage_H2_highP"]],
        "existing_storage_technologies": [{"Storage_H2_Cavern": 100000}],
        "hydrogen_demand_small": [{
            "SMALL1": {"Hydrogen use (TWh)": 0.95, "Capacity (MW)": 174},
            "SMALL2": {"Hydrogen use (TWh)": 0.64, "Capacity (MW)": 117},
            "SMALL3": {"Hydrogen use (TWh)": 0.48, "Capacity (MW)": 87},
            "SMALL4": {"Hydrogen use (TWh)": 0.18, "Capacity (MW)": 33}
        }],
        "hydrogen_demand_big": [{
            "BIG1": {"Hydrogen use (TWh)": 15},
            "BIG2": {"Hydrogen use (TWh)": 5}
        }],
        "mipgap": [0.01],
        "time_limit": [50],
        "threads": [48]
    }

    # Generate combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Prepare run configs
    run_configs = [(f"parallel_run_{i:04d}", params) for i, params in enumerate(combinations, 1)]

    print(f"Total combinations: {len(run_configs)}")

    # Create timestamp for results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = base_path / "results" / f"parallel_test_{timestamp}"

    # Run parallel optimization
    runner = ParallelOptimizationRunner(
        base_path=base_path,
        max_workers=4  # Adjust based on your system
    )

    results_summary = runner.run_parallel_optimization(
        run_configs=run_configs,
        results_base_folder=results_folder
    )

    print(f"\n✅ Parallel optimization complete!")
    print(f"📊 Summary Excel: {results_folder / 'parallel_results_summary.xlsx'}")

