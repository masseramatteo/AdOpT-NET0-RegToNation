"""
Parallel Optimization Runner with BENDERS DECOMPOSITION
Modified version that uses ModelHub_Benders instead of ModelHub
BOTH creates models AND solves them in parallel using multiprocessing
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
    """Create and prepare a single model without solving (runs in parallel process)"""
    run_id, params, results_base_folder, base_path = args

    print(f"[CREATE] Creating model: {run_id}")

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
        runner._configure_networks(input_data_path, params["networks_existing"], params["networks_new"])
        adopt.copy_network_data(input_data_path)

        # Import network creation functions
        from define_topology import (
            add_new_distribution_network,
            add_new_transmission_network,
            add_existing_distribution_network,
            add_existing_transmission_network
        )

        # Create networks based on configuration
        # NEW networks
        if params.get("networks_new") and params["networks_new"][0]:
            networks_new = params["networks_new"]

            # Check for new low pressure (distribution)
            if "hydrogenPipelineOnshore_lowP" in networks_new:
                print(f"[NETWORK] Creating new distribution network (lowP)")
                add_new_distribution_network(input_data_path, params["scenarios"])

            # Check for new high pressure (transmission)
            if "hydrogenPipelineOnshore_highP" in networks_new:
                print(f"[NETWORK] Creating new transmission network (highP)")
                add_new_transmission_network(input_data_path, params["scenarios"])

        # EXISTING networks
        if params.get("networks_existing") and params["networks_existing"][0]:
            networks_existing = params["networks_existing"]

            # Check for existing low pressure (distribution)
            if "hydrogenPipelineOnshore_lowP" in networks_existing:
                print(f"[NETWORK] Creating existing distribution network (lowP)")
                add_existing_distribution_network(input_data_path, params["scenarios"])

            # Check for existing high pressure (transmission)
            if "hydrogenPipelineOnshore_highP" in networks_existing:
                print(f"[NETWORK] Creating existing transmission network (highP)")
                add_existing_transmission_network(input_data_path, params["scenarios"])

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

        print(f"[SUCCESS] Model created: {run_id}")
        return run_id, str(input_data_path), str(results_data_path), params, None

    except Exception as e:
        print(f"[ERROR] Error creating model {run_id}: {e}")
        import traceback
        traceback.print_exc()
        return run_id, None, None, params, str(e)


def solve_single_model_benders(args):
    """
    Solve a single model using BENDERS DECOMPOSITION (runs in separate process)

    Modified version that uses ModelHub_Benders.solve_with_benders()
    """
    run_id, input_data_path, results_data_path, params = args

    try:
        print(f"[SOLVING BENDERS] {run_id}")

        # Import Gurobi and set per-process parameters
        try:
            import gurobipy as gp

            # Extract thread count from params (default to 1 if not specified)
            threads = int(params.get("threads", 1))

            gp.setParam("Threads", threads)
            print(f"  [CONFIG] Gurobi configured: {threads} threads per worker")

        except ImportError:
            print(f"  [WARNING] gurobipy not available for direct configuration")
        except Exception as e:
            print(f"  [WARNING] Could not configure Gurobi: {e}")

        import adopt_net0 as adopt
        import pyomo.environ as pyo

        input_data_path = Path(input_data_path)
        results_data_path = Path(results_data_path)

        # ====================================================================
        # BENDERS DECOMPOSITION: Use ModelHub_Benders instead of ModelHub
        # ====================================================================
        m = adopt.ModelHub_Benders()
        m.read_data(input_data_path, start_period=0, end_period=744)

        # Construct model and balances
        m.construct_model()
        m.construct_balances()

        # Get Benders parameters from params (with defaults)
        max_iterations = params.get("benders_max_iterations", 50)
        tolerance = params.get("benders_tolerance", 1e-4)
        verbose = params.get("benders_verbose", False)  # Set to False for parallel runs

        print(f"  [BENDERS] Starting decomposition:")
        print(f"    - Max iterations: {max_iterations}")
        print(f"    - Tolerance: {tolerance}")

        # Solve with Benders decomposition
        m.solve_with_benders(
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=verbose
        )

        # Extract Benders statistics
        benders_info = m.benders_info

        print(f"  [BENDERS] Completed:")
        print(f"    - Converged: {benders_info['converged']}")
        print(f"    - Iterations: {benders_info['iteration']}")
        print(f"    - Gap: {benders_info['gap']:.6f}")
        print(f"    - Final objective: {benders_info['upper_bound']:,.2f}")

        # Write results to HDF5 and summary files
        print(f"  [BENDERS] Writing results...")
        try:
            m.write_results()
            print(f"  [BENDERS] Results written successfully")
        except Exception as e:
            print(f"  [WARNING] Could not write results: {e}")
            import traceback
            traceback.print_exc()

        # Extract results
        result_folder_path = Path(m.last_solve_info.get("result_folder_path", results_data_path / "results"))
        result_folder_path.mkdir(parents=True, exist_ok=True)

        # Save params info with Benders statistics
        params_with_benders = params.copy()
        params_with_benders.update({
            "benders_converged": benders_info['converged'],
            "benders_iterations": benders_info['iteration'],
            "benders_gap": benders_info['gap'],
            "benders_lower_bound": benders_info['lower_bound'],
            "benders_upper_bound": benders_info['upper_bound'],
        })

        text = json.dumps(params_with_benders, indent=4, ensure_ascii=False, default=str)
        (result_folder_path / "optimization_model_info.txt").write_text(text, encoding="utf-8")

        # Extract results from model
        model = None
        if hasattr(m, "model"):
            if isinstance(m.model, dict):
                aggregation = m.info_solving_algorithms.get("aggregation_model", "full")
                model = m.model.get(aggregation) or next(iter(m.model.values()))
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

            objective_value = benders_info['upper_bound']  # Use Benders upper bound

            result_info = {
                "var_npv": var_npv,
                "para_total_demand": para_total_demand,
                "npv_over_demand": npv_over_demand,
                "demand_times_wtp": demand_times_wtp,
                "willingness_to_pay": wtp,
                "objective_value": objective_value,
                # Add Benders-specific info
                "benders_converged": benders_info['converged'],
                "benders_iterations": benders_info['iteration'],
                "benders_gap": benders_info['gap'],
                "benders_lower_bound": benders_info['lower_bound'],
                "benders_upper_bound": benders_info['upper_bound'],
                "benders_cuts_added": benders_info['cuts_added'],
                "benders_total_master_time": sum(benders_info['master_solve_times']),
                "benders_total_subproblem_time": sum(benders_info['subproblem_solve_times']),
            }

            # Save to JSON
            result_file = result_folder_path / "optimization_results_summary.json"
            with open(result_file, "w") as f:
                json.dump(result_info, f, indent=4, default=str)

        print(f"[SUCCESS] Solved with Benders: {run_id}")
        return run_id, result_info, None

    except Exception as e:
        print(f"[ERROR] Error solving {run_id}: {e}")
        import traceback
        traceback.print_exc()
        return run_id, None, str(e)


class ParallelBendersOptimizationRunner:
    """
    Run multiple optimizations in parallel using BENDERS DECOMPOSITION

    Modified version that uses ModelHub_Benders for all solves
    """

    @staticmethod
    def _get_strategy_config(cpu_count, strategy):
        """Get worker and thread configuration based on strategy"""
        target_total_threads = int(cpu_count * 0.90)

        if strategy == 'auto':
            max_workers = max(2, min(8, cpu_count // 2))
            threads_per_worker = max(1, target_total_threads // max_workers)
            description = "Auto-detect (balanced, 90% CPU target)"

        elif strategy == 'throughput':
            if cpu_count >= 40:
                threads_per_worker = max(3, cpu_count // 12)
            elif cpu_count >= 20:
                threads_per_worker = max(2, cpu_count // 8)
            else:
                threads_per_worker = max(2, cpu_count // 5)
            max_workers = max(2, target_total_threads // threads_per_worker)
            description = f"Throughput (many solves, {threads_per_worker} threads each)"

        elif strategy == 'heavy':
            if cpu_count >= 40:
                threads_per_worker = max(6, cpu_count // 8)
            elif cpu_count >= 20:
                threads_per_worker = max(5, cpu_count // 4)
            else:
                threads_per_worker = max(4, cpu_count // 3)
            max_workers = max(2, target_total_threads // threads_per_worker)
            description = f"Heavy problems (fewer solves, {threads_per_worker} threads each)"

        elif strategy == 'max_workers':
            max_workers = max(2, target_total_threads)
            threads_per_worker = 1
            description = f"Max workers ({max_workers} workers × 1 thread)"

        elif strategy == 'max_threads':
            threads_per_worker = max(2, target_total_threads)
            max_workers = 2
            description = f"Max threads (2 workers × {threads_per_worker} threads)"

        elif strategy == 'Personal_laptop_singles':
            threads_per_worker = 8
            max_workers = 1
            description = f"Max threads (1 workers × {threads_per_worker} threads)"

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return max_workers, threads_per_worker, description

    @staticmethod
    def prompt_strategy_selection():
        """Prompt user to select parallelization strategy"""
        cpu_count = mp.cpu_count()
        print(f"\n{'='*80}")
        print(f"BENDERS DECOMPOSITION - PARALLELIZATION STRATEGY")
        print(f"{'='*80}")
        print(f"Detected: {cpu_count} logical processors")
        print(f"\nAvailable strategies:\n")

        for strategy_key, strategy_name, strategy_num in [
            ('auto', 'Auto-detect', 'ENTER'),
            ('throughput', 'Throughput', '1'),
            ('heavy', 'Heavy problems', '2'),
        ]:
            workers, threads, desc = ParallelBendersOptimizationRunner._get_strategy_config(
                cpu_count, strategy_key
            )
            total = workers * threads
            pct = (total / cpu_count) * 100

            print(f"  [{strategy_num}] {strategy_name}:")
            print(f"          {desc}")
            print(f"          → {workers} workers × {threads} threads = {total} total ({pct:.0f}% CPU)")
            print()

        print(f"{'='*80}")

        while True:
            choice = input("Select strategy [ENTER/1/2]: ").strip().lower()
            if choice == '' or choice == 'auto':
                return 'auto', None, None
            elif choice == '1' or choice == 'throughput':
                return 'throughput', None, None
            elif choice == '2' or choice == 'heavy':
                return 'heavy', None, None
            else:
                print(f"Invalid choice. Please enter ENTER, 1, or 2.")

    def __init__(self, base_path, max_workers=None, threads_per_worker=None, strategy=None):
        """
        Initialize parallel Benders runner

        Args:
            base_path: Base path for the optimization project
            max_workers: Number of parallel workers
            threads_per_worker: Threads per Gurobi solve
            strategy: 'auto', 'throughput', 'heavy', or None (will prompt)
        """
        self.base_path = Path(base_path)

        cpu_count = mp.cpu_count()

        if max_workers is not None and threads_per_worker is not None:
            self.max_workers = max_workers
            self.threads_per_worker = threads_per_worker
            strategy_description = "Manual configuration"
        else:
            if strategy is None:
                result = self.prompt_strategy_selection()
                if isinstance(result, tuple):
                    strategy, _, _ = result

            max_workers, threads_per_worker, strategy_description = self._get_strategy_config(
                cpu_count, strategy
            )
            self.max_workers = max_workers
            self.threads_per_worker = threads_per_worker

        actual_total_threads = self.max_workers * self.threads_per_worker
        actual_utilization_pct = (actual_total_threads / cpu_count) * 100

        print(f"\n[CONFIG] Benders Parallel Configuration:")
        print(f"   - Strategy: {strategy_description}")
        print(f"   - Total logical processors: {cpu_count}")
        print(f"   - Parallel workers: {self.max_workers}")
        print(f"   - Threads per worker: {self.threads_per_worker}")
        print(f"   - Total thread usage: {actual_total_threads} (of {cpu_count})")
        print(f"   - CPU utilization: {actual_utilization_pct:.1f}%")

    def run_parallel_optimization(self, run_configs, results_base_folder):
        """
        Run multiple Benders optimizations in parallel

        Args:
            run_configs: List of (run_id, params) tuples
            results_base_folder: Base folder for results

        Returns:
            results_summary: List of dictionaries with results
        """
        results_base_folder = Path(results_base_folder)
        results_base_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"PARALLEL BENDERS DECOMPOSITION RUNNER")
        print(f"{'='*80}")
        print(f"Total runs: {len(run_configs)}")
        print(f"Max parallel workers: {self.max_workers}")
        print(f"Threads per worker: {self.threads_per_worker}")
        print(f"Results folder: {results_base_folder}")
        print(f"{'='*80}\n")

        total_start_time = time.time()

        # Inject threads parameter
        for run_id, params in run_configs:
            if "threads" not in params:
                params["threads"] = self.threads_per_worker

        # Phase 1: Create models in parallel
        print(f"\n{'='*80}")
        print(f"PHASE 1: Creating {len(run_configs)} models in PARALLEL")
        print(f"{'='*80}\n")

        model_configs = []
        creation_errors = []
        creation_start_time = time.time()

        ctx = mp.get_context('spawn')

        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as executor:
            future_to_runid = {
                executor.submit(create_single_model, (run_id, params, results_base_folder, self.base_path)): run_id
                for run_id, params in run_configs
            }

            completed = 0
            for future in as_completed(future_to_runid):
                run_id = future_to_runid[future]
                completed += 1

                try:
                    run_id, input_path, results_path, params, error = future.result()

                    if error:
                        creation_errors.append({
                            "run_id": run_id,
                            "status": "CREATION_FAILED",
                            "error": error,
                            **params
                        })
                    else:
                        model_configs.append((run_id, input_path, results_path, params))

                    print(f"Model creation progress: {completed}/{len(run_configs)} completed")

                except Exception as e:
                    print(f"[ERROR] Exception creating {run_id}: {e}")
                    creation_errors.append({
                        "run_id": run_id,
                        "status": "CREATION_FAILED",
                        "error": str(e),
                    })

        creation_elapsed = time.time() - creation_start_time
        print(f"\n[COMPLETE] Model creation: {len(model_configs)} models ready")
        print(f"   Creation time: {creation_elapsed:.1f}s")

        # Phase 2: Solve with Benders in parallel
        print(f"\n{'='*80}")
        print(f"PHASE 2: Solving {len(model_configs)} models with BENDERS")
        print(f"{'='*80}\n")

        solve_start_time = time.time()
        results_summary = []
        solve_errors = []

        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as executor:
            future_to_runid = {
                executor.submit(solve_single_model_benders, config): config[0]
                for config in model_configs
            }

            completed = 0
            for future in as_completed(future_to_runid):
                run_id = future_to_runid[future]
                completed += 1

                try:
                    run_id, result_info, error = future.result()

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

                    print(f"Solve progress: {completed}/{len(model_configs)} completed")

                except Exception as e:
                    print(f"[ERROR] Exception processing {run_id}: {e}")
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

        solve_elapsed = time.time() - solve_start_time
        total_elapsed = time.time() - total_start_time

        # Add errors to summary
        results_summary.extend(creation_errors)
        results_summary.extend(solve_errors)

        # Save results
        df_summary = pd.DataFrame(results_summary)
        df_summary.to_csv(results_base_folder / "benders_results_summary.csv", index=False, sep=';')
        df_summary.to_excel(results_base_folder / "benders_results_summary.xlsx", index=False)

        # Print summary
        successful = sum(1 for r in results_summary if r.get('status') == 'SUCCESS')
        failed = len(results_summary) - successful

        print(f"\n{'='*80}")
        print(f"PARALLEL BENDERS OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total runs: {len(run_configs)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"\n[TIMING]:")
        print(f"  - Creation: {creation_elapsed:.1f}s")
        print(f"  - Solving: {solve_elapsed:.1f}s")
        print(f"  - TOTAL: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print(f"\n[RESULTS] Saved to: {results_base_folder}")
        print(f"{'='*80}\n")

        return results_summary


# ============================================================================
# PARAMETER GENERATION (same as original)
# ============================================================================

def generate_parameter_combinations(param_grid, method='full'):
    """Generate parameter combinations"""
    import itertools

    if method == 'full':
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print(f"[SAMPLING] FULL GRID: {len(combinations)} combinations")
        return combinations
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# EXAMPLE USAGE WITH BENDERS
# ============================================================================
if __name__ == "__main__":
    base_path = Path(__file__).parent

    # Example parameter grid with BENDERS-specific parameters
    param_grid = {
        "scenarios": ["1751"],
        "demand_level_ratio": [15],
        "total_demand_TWh": [20],
        "import_availability_ratio": [0.4],
        "electricity_price_avg": [150],
        "electricity_availability_small": [100],
        "willingness_to_pay": [1075],
        "networks_new": [["hydrogenPipelineOnshore_lowP", "hydrogenPipelineOnshore_highP"]],
        "networks_existing": [[]],
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
        # BENDERS-specific parameters
        "benders_max_iterations": [50],  # Max Benders iterations
        "benders_tolerance": [1e-4],     # Convergence tolerance
        "benders_verbose": [False],      # False for parallel runs
    }

    combinations = generate_parameter_combinations(param_grid, method='full')
    run_configs = [(f"benders_run_{i:04d}", params) for i, params in enumerate(combinations, 1)]

    print(f"\n[OK] Total Benders runs: {len(run_configs)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = base_path / "results" / f"benders_parallel_{timestamp}"

    # Create Benders runner
    runner = ParallelBendersOptimizationRunner(
        base_path=base_path,
        strategy='Personal_laptop_singles'  # Will prompt for strategy
    )

    results_summary = runner.run_parallel_optimization(
        run_configs=run_configs,
        results_base_folder=results_folder
    )

    print(f"\n[OK] Parallel Benders optimization complete!")
    print(f"[RESULTS] Summary: {results_folder / 'benders_results_summary.xlsx'}")

