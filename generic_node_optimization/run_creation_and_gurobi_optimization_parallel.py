"""
Parallel Optimization Runner with Parallel Model Creation and Gurobi Environment Management
BOTH creates models AND solves them in parallel using multiprocessing
Uses Gurobi-recommended explicit environment management to avoid resource leaks
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


def solve_single_model(args):
    """
    Solve a single model (runs in separate process)

    Uses Gurobi's default environment per process with controlled thread limits.
    Each worker process (via 'spawn' context) has its own isolated Gurobi environment,
    preventing resource conflicts between parallel solves.
    """
    run_id, input_data_path, results_data_path, params = args

    try:
        print(f"[SOLVING] {run_id}")

        # Import Gurobi and set per-process parameters
        try:
            import gurobipy as gp

            # Extract thread count from params (default to 1 if not specified)
            threads = int(params.get("threads", 1))

            # Set Gurobi parameters for this process using the default environment
            # Since we use 'spawn' context, each process has its own default env
            # This prevents CPU oversubscription when running multiple workers
            gp.setParam("Threads", threads)

            # Optional: reduce console output (uncomment if needed)
            # gp.setParam("OutputFlag", 0)

            print(f"  [CONFIG] Gurobi default env configured: {threads} threads per worker")

        except ImportError:
            print(f"  [WARNING] gurobipy not available for direct configuration")
        except Exception as e:
            print(f"  [WARNING] Could not configure Gurobi: {e}")

        import adopt_net0 as adopt
        import pyomo.environ as pyo

        input_data_path = Path(input_data_path)
        results_data_path = Path(results_data_path)

        # Create ModelHub and read data
        # Pyomo/adopt will use the default Gurobi environment configured above
        m = adopt.ModelHub()
        m.read_data(input_data_path, start_period=0, end_period=8760)

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

        print(f"[SUCCESS] Solved: {run_id}")
        return run_id, result_info, None

    except Exception as e:
        print(f"[ERROR] Error solving {run_id}: {e}")
        import traceback
        traceback.print_exc()
        return run_id, None, str(e)


class ParallelCreationAndGurobiOptimizationRunner:
    """
    Run multiple optimizations in parallel with PARALLEL MODEL CREATION and Gurobi-aware process management

    This runner parallelizes BOTH phases:
    - Phase 1: Model creation in parallel
    - Phase 2: Model solving in parallel

    Uses the 'spawn' multiprocessing context to ensure child processes
    don't inherit the parent's Gurobi environment. Each worker process uses
    Gurobi's default environment (per-process), with controlled thread limits
    to prevent CPU oversubscription and optimize performance.
    """

    @staticmethod
    def _get_strategy_config(cpu_count, strategy):
        """Get worker and thread configuration based on strategy"""
        target_total_threads = int(cpu_count * 0.90)

        if strategy == 'auto':
            # Balanced: aim for ~4-6 threads per worker, adjust workers accordingly
            max_workers = max(2, min(8, cpu_count // 2))
            threads_per_worker = max(1, target_total_threads // max_workers)
            description = "Auto-detect (balanced, 85% CPU target)"

        elif strategy == 'throughput':
            # Throughput: maximize workers, use ~25-35% of cores per worker
            # Goal: many problems running, each with modest thread count
            if cpu_count >= 40:
                threads_per_worker = max(3, cpu_count // 12)  # ~3-4 threads
            elif cpu_count >= 20:
                threads_per_worker = max(2, cpu_count // 8)   # ~2-3 threads
            else:
                threads_per_worker = max(2, cpu_count // 5)   # ~2-3 threads
            max_workers = max(2, target_total_threads // threads_per_worker)
            description = f"Throughput (many solves, {threads_per_worker} threads each)"

        elif strategy == 'heavy':
            # Heavy: fewer workers, more threads each (~40-50% of cores per worker)
            # Goal: fewer problems running, each with more computational power
            if cpu_count >= 40:
                threads_per_worker = max(6, cpu_count // 8)   # ~6 threads
            elif cpu_count >= 20:
                threads_per_worker = max(5, cpu_count // 4)   # ~5 threads
            else:
                threads_per_worker = max(4, cpu_count // 3)   # ~4-5 threads
            max_workers = max(2, target_total_threads // threads_per_worker)
            description = f"Heavy problems (fewer solves, {threads_per_worker} threads each)"

        elif strategy == 'max_workers':
            # Max workers: use ~90% of cores as workers, minimal threads per worker
            max_workers = max(2, target_total_threads)
            threads_per_worker = 1  # Single-threaded solves for maximum parallelism
            description = f"Max workers (maximum parallelism, {max_workers} workers × 1 thread)"

        elif strategy == 'max_threads':
            # Max threads: use ~90% of cores for a single worker, minimize workers
            threads_per_worker = max(2, target_total_threads)
            max_workers = 2  # Minimal workers for maximum threads each
            description = f"Max threads (max power per solve, 2 workers × {threads_per_worker} threads)"

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return max_workers, threads_per_worker, description

    @staticmethod
    def prompt_strategy_selection():
        """Prompt user to select parallelization strategy"""
        cpu_count = mp.cpu_count()
        print(f"\n{'='*80}")
        print(f"PARALLELIZATION STRATEGY SELECTION")
        print(f"{'='*80}")
        print(f"Detected: {cpu_count} logical processors")
        print(f"\nAvailable strategies:\n")

        for strategy_key, strategy_name, strategy_num in [
            ('auto', 'Auto-detect', 'ENTER'),
            ('throughput', 'Throughput', '1'),
            ('heavy', 'Heavy problems', '2'),
            ('max_workers', 'Max Workers', '3'),
            ('max_threads', 'Max Threads', '4')
        ]:
            workers, threads, desc = ParallelCreationAndGurobiOptimizationRunner._get_strategy_config(
                cpu_count, strategy_key
            )
            total = workers * threads
            pct = (total / cpu_count) * 100

            print(f"  [{strategy_num}] {strategy_name}:")
            print(f"          {desc}")
            print(f"          → {workers} workers × {threads} threads = {total} total ({pct:.0f}% CPU)")

            if strategy_key == 'throughput':
                print(f"          Best for: Many problems, maximize throughput")
            elif strategy_key == 'heavy':
                print(f"          Best for: Very complex MILPs, minimize individual solve time")
            elif strategy_key == 'max_workers':
                print(f"          Best for: Maximum parallelism, many small/fast problems")
            elif strategy_key == 'max_threads':
                print(f"          Best for: Few very large problems, max power per solve")
            print()

        print(f"  [P] Manual configuration:")
        print(f"          Define custom workers and threads")
        print()

        print(f"{'='*80}")

        while True:
            choice = input("Select strategy [ENTER/1/2/3/4/P]: ").strip().lower()
            if choice == '' or choice == 'auto':
                return 'auto', None, None
            elif choice == '1' or choice == 'throughput':
                return 'throughput', None, None
            elif choice == '2' or choice == 'heavy':
                return 'heavy', None, None
            elif choice == '3' or choice == 'max_workers':
                return 'max_workers', None, None
            elif choice == '4' or choice == 'max_threads':
                return 'max_threads', None, None
            elif choice == 'p':
                # Manual configuration
                print(f"\n{'='*40}")
                print(f"MANUAL CONFIGURATION")
                print(f"{'='*40}")
                print(f"Available: {cpu_count} logical processors")

                while True:
                    try:
                        workers_input = input(f"Enter number of workers (1-{cpu_count}): ").strip()
                        manual_workers = int(workers_input)
                        if 1 <= manual_workers <= cpu_count:
                            break
                        else:
                            print(f"⚠️  Please enter a value between 1 and {cpu_count}")
                    except ValueError:
                        print(f"⚠️  Please enter a valid integer")

                while True:
                    try:
                        threads_input = input(f"Enter threads per worker (1-{cpu_count}): ").strip()
                        manual_threads = int(threads_input)
                        if 1 <= manual_threads <= cpu_count:
                            break
                        else:
                            print(f"⚠️  Please enter a value between 1 and {cpu_count}")
                    except ValueError:
                        print(f"⚠️  Please enter a valid integer")

                total = manual_workers * manual_threads
                pct = (total / cpu_count) * 100

                print(f"\n[OK] Manual configuration:")
                print(f"   -> {manual_workers} workers × {manual_threads} threads = {total} total ({pct:.0f}% CPU)")

                if total > cpu_count * 1.2:
                    print(f"   [WARNING] High CPU oversubscription ({pct:.0f}%)!")

                return 'manual', manual_workers, manual_threads
            else:
                print(f"Invalid choice '{choice}'. Please enter ENTER, 1, 2, 3, 4, or P.")

    def __init__(self, base_path, max_workers=None, threads_per_worker=None,
                 cpu_utilization_target=0.85, strategy=None):
        """
        Initialize parallel runner with Gurobi-aware configuration

        Args:
            base_path: Base path for the optimization project
            max_workers: Number of parallel workers (None = use strategy)
            threads_per_worker: Threads per Gurobi solve (None = use strategy)
            cpu_utilization_target: Target CPU utilization (0.0-1.0), default 0.85
                                   Only used if strategy='auto' or strategy=None
            strategy: 'auto', 'throughput', 'heavy', or None (will prompt user)
        """
        self.base_path = Path(base_path)

        # Auto-detect optimal worker count and threads
        cpu_count = mp.cpu_count()

        # If both max_workers and threads_per_worker are provided, use them directly
        if max_workers is not None and threads_per_worker is not None:
            self.max_workers = max_workers
            self.threads_per_worker = threads_per_worker
            strategy_description = "Manual configuration"
        else:
            # Use strategy-based configuration
            if strategy is None:
                # Prompt user for strategy selection
                result = self.prompt_strategy_selection()

                # Handle manual configuration (returns tuple)
                if isinstance(result, tuple):
                    strategy, manual_workers, manual_threads = result
                    if strategy == 'manual':
                        self.max_workers = manual_workers
                        self.threads_per_worker = manual_threads
                        strategy_description = "Manual configuration"
                    else:
                        # Get configuration based on strategy
                        max_workers, threads_per_worker, strategy_description = self._get_strategy_config(
                            cpu_count, strategy
                        )
                        self.max_workers = max_workers
                        self.threads_per_worker = threads_per_worker
                else:
                    # Old behavior for compatibility
                    strategy = result
                    max_workers, threads_per_worker, strategy_description = self._get_strategy_config(
                        cpu_count, strategy
                    )
                    self.max_workers = max_workers
                    self.threads_per_worker = threads_per_worker
            else:
                # Get configuration based on strategy
                max_workers, threads_per_worker, strategy_description = self._get_strategy_config(
                    cpu_count, strategy
                )
                self.max_workers = max_workers
                self.threads_per_worker = threads_per_worker

        # Calculate actual utilization
        actual_total_threads = self.max_workers * self.threads_per_worker
        actual_utilization_pct = (actual_total_threads / cpu_count) * 100

        print(f"\n[CONFIG] System configuration:")
        print(f"   - Strategy: {strategy_description}")
        print(f"   - Total logical processors: {cpu_count}")
        print(f"   - Parallel workers: {self.max_workers}")
        print(f"   - Threads per worker: {self.threads_per_worker}")
        print(f"   - Total thread usage: {actual_total_threads} (of {cpu_count})")
        print(f"   - CPU utilization: {actual_utilization_pct:.1f}%")

        if actual_total_threads > cpu_count:
            print(f"   [WARNING] CPU oversubscription ({actual_utilization_pct:.1f}%)!")
            print(f"       Consider reducing workers or threads_per_worker")
        elif actual_utilization_pct < 70:
            print(f"   [INFO] Conservative CPU usage ({actual_utilization_pct:.1f}%)")
            print(f"       Consider increasing workers or threads_per_worker for better performance")
        else:
            print(f"   [OK] Good CPU utilization balance")

    def run_parallel_optimization(self, run_configs, results_base_folder):
        """
        Run multiple optimizations in parallel with PARALLEL MODEL CREATION

        Both model creation and solving happen in parallel for maximum performance

        Args:
            run_configs: List of (run_id, params) tuples
            results_base_folder: Base folder for results

        Returns:
            results_summary: List of dictionaries with results for each run
        """
        results_base_folder = Path(results_base_folder)
        results_base_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"PARALLEL CREATION + GUROBI-ENV OPTIMIZATION RUNNER")
        print(f"{'='*80}")
        print(f"Total runs: {len(run_configs)}")
        print(f"Max parallel workers: {self.max_workers}")
        print(f"Threads per worker: {self.threads_per_worker}")
        print(f"Results folder: {results_base_folder}")
        print(f"{'='*80}\n")

        # Start total timer
        total_start_time = time.time()

        # Inject threads parameter into all run configs
        for run_id, params in run_configs:
            if "threads" not in params:
                params["threads"] = self.threads_per_worker

        # =====================================================================
        # PHASE 1: CREATE ALL MODELS (PARALLEL)
        # =====================================================================
        print(f"\n{'='*80}")
        print(f"PHASE 1: Creating {len(run_configs)} models in PARALLEL")
        print(f"Using 'spawn' context for clean process separation")
        print(f"{'='*80}\n")

        model_configs = []
        creation_errors = []

        creation_start_time = time.time()

        # Use spawn context for model creation
        ctx = mp.get_context('spawn')

        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as executor:
            # Submit all model creation jobs
            future_to_runid = {
                executor.submit(create_single_model, (run_id, params, results_base_folder, self.base_path)): run_id
                for run_id, params in run_configs
            }

            # Collect results as they complete
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
                        **{}
                    })

        creation_elapsed = time.time() - creation_start_time
        print(f"\n[COMPLETE] Model creation complete: {len(model_configs)} models ready to solve")
        print(f"   Creation time: {creation_elapsed:.1f}s ({creation_elapsed/60:.1f} min)")
        if creation_errors:
            print(f"[WARNING] {len(creation_errors)} models failed to create")

        # =====================================================================
        # PHASE 2: SOLVE ALL MODELS (PARALLEL)
        # =====================================================================
        print(f"\n{'='*80}")
        print(f"PHASE 2: Solving {len(model_configs)} models in PARALLEL")
        print(f"Using 'spawn' context for clean Gurobi environment per worker")
        print(f"{'='*80}\n")

        solve_start_time = time.time()
        results_summary = []
        solve_errors = []

        # Reuse spawn context (already created in Phase 1)
        # This ensures child processes don't inherit parent's Gurobi environment
        # Critical for proper license management and resource cleanup

        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as executor:
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

                    print(f"Solve progress: {completed}/{len(model_configs)} completed")

                except Exception as e:
                    print(f"[ERROR] Exception processing {run_id}: {e}")
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

        solve_elapsed = time.time() - solve_start_time
        total_elapsed = time.time() - creation_start_time

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
        print(f"\n[TIMING] Timing breakdown:")
        print(f"  - Model creation (parallel): {creation_elapsed:.1f}s ({creation_elapsed/60:.1f} min)")
        print(f"  - Model solving (parallel): {solve_elapsed:.1f}s ({solve_elapsed/60:.1f} min)")
        print(f"  - TOTAL TIME: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print(f"  - Average per run: {total_elapsed/len(run_configs):.1f}s")
        print(f"\n[SPEEDUP] Speedup estimate vs sequential:")
        print(f"  - If creation was sequential: ~{creation_elapsed * self.max_workers / 60:.1f} min")
        print(f"  - If solving was sequential: ~{solve_elapsed * self.max_workers / 60:.1f} min")
        print(f"  - Potential total sequential time: ~{(creation_elapsed + solve_elapsed) * self.max_workers / 60:.1f} min")
        print(f"\n[RESULTS] Results saved to: {results_base_folder}")
        print(f"{'='*80}\n")

        if failed > 0:
            print("[WARNING] Failed runs:")
            for item in (creation_errors + solve_errors):
                print(f"  {item['run_id']}: {item.get('error', 'Unknown error')}")

        return results_summary


# ============================================================================
# PARAMETER SAMPLING UTILITIES
# ============================================================================

def latin_hypercube_sampling(param_grid, n_samples=100, seed=None):
    """
    Generate parameter combinations using Latin Hypercube Sampling

    Args:
        param_grid: Dictionary with parameter names as keys and lists of values
        n_samples: Maximum number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        List of parameter dictionaries
    """
    if seed is not None:
        np.random.seed(seed)

    # Separate numeric and non-numeric parameters
    numeric_params = {}
    non_numeric_params = {}

    for key, values in param_grid.items():
        # Check if all values are numeric (int or float)
        if len(values) > 0 and all(isinstance(v, (int, float)) for v in values):
            numeric_params[key] = values
        else:
            non_numeric_params[key] = values

    # For numeric parameters, use LHS
    n_numeric = len(numeric_params)
    if n_numeric > 0:
        # Generate LHS samples in [0, 1]^n_numeric
        intervals = np.linspace(0, 1, n_samples + 1)
        lhs_samples = np.zeros((n_samples, n_numeric))

        for i in range(n_numeric):
            # Random permutation of intervals
            perm = np.random.permutation(n_samples)
            # Sample uniformly within each interval
            lhs_samples[:, i] = np.random.uniform(intervals[perm], intervals[perm + 1])

        # Map LHS samples to actual parameter values
        numeric_keys = list(numeric_params.keys())
        sampled_numeric = []

        for sample in lhs_samples:
            param_dict = {}
            for i, key in enumerate(numeric_keys):
                values = sorted(numeric_params[key])
                # Map [0, 1] to parameter range using interpolation
                idx_float = sample[i] * (len(values) - 1)
                idx_low = int(np.floor(idx_float))
                idx_high = min(idx_low + 1, len(values) - 1)

                # If very close to a discrete value, use it; otherwise interpolate
                if idx_low == idx_high:
                    param_dict[key] = values[idx_low]
                else:
                    # Linear interpolation
                    weight = idx_float - idx_low
                    param_dict[key] = values[idx_low] * (1 - weight) + values[idx_high] * weight

            sampled_numeric.append(param_dict)
    else:
        sampled_numeric = [{}] * n_samples

    # For non-numeric parameters, sample randomly or use all combinations if few
    if non_numeric_params:
        # Calculate total non-numeric combinations
        import itertools
        non_numeric_combinations = list(itertools.product(*non_numeric_params.values()))

        if len(non_numeric_combinations) <= n_samples:
            # Use all combinations
            sampled_non_numeric = [
                dict(zip(non_numeric_params.keys(), combo))
                for combo in non_numeric_combinations
            ]
        else:
            # Random sample without replacement
            indices = np.random.choice(len(non_numeric_combinations), n_samples, replace=False)
            sampled_non_numeric = [
                dict(zip(non_numeric_params.keys(), non_numeric_combinations[idx]))
                for idx in indices
            ]
    else:
        sampled_non_numeric = [{}] * n_samples

    # Combine numeric and non-numeric samples
    # If different lengths, cycle or truncate
    max_len = max(len(sampled_numeric), len(sampled_non_numeric))
    combinations = []

    for i in range(min(n_samples, max_len)):
        combined = {}
        if sampled_numeric:
            combined.update(sampled_numeric[i % len(sampled_numeric)])
        if sampled_non_numeric:
            combined.update(sampled_non_numeric[i % len(sampled_non_numeric)])
        combinations.append(combined)

    return combinations[:n_samples]


def generate_parameter_combinations(param_grid, method='full', max_samples=100, seed=42):
    """
    Generate parameter combinations using different methods

    Args:
        param_grid: Dictionary with parameter names as keys and lists of values
        method: 'full' for full grid, 'lhs' for Latin Hypercube Sampling
        max_samples: Maximum number of samples (only for LHS)
        seed: Random seed for reproducibility (only for LHS)

    Returns:
        List of parameter dictionaries
    """
    import itertools

    if method == 'full':
        # Full grid search (Cartesian product)
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print(f"[SAMPLING] Method: FULL GRID")
        print(f"   Total combinations: {len(combinations)}")
        return combinations

    elif method == 'lhs':
        # Latin Hypercube Sampling
        # Calculate potential full grid size
        full_size = 1
        for values in param_grid.values():
            full_size *= len(values)

        actual_samples = min(max_samples, full_size)
        combinations = latin_hypercube_sampling(param_grid, n_samples=actual_samples, seed=seed)

        print(f"[SAMPLING] Method: LATIN HYPERCUBE SAMPLING")
        print(f"   Full grid would be: {full_size} combinations")
        print(f"   LHS samples: {len(combinations)}")
        print(f"   Reduction: {(1 - len(combinations)/full_size)*100:.1f}%")
        return combinations

    else:
        raise ValueError(f"Unknown method: {method}. Use 'full' or 'lhs'")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    import itertools
    import numpy as np

    base_path = Path(__file__).parent

    # =======================================================
    #  Define Parameters
    # =======================================================
    param_grid = {
        "scenarios": ["1751"],
        "demand_level_ratio": [15],
        "total_demand_TWh": [20],
        "import_availability_ratio": [0.4],
        #"import_cost_multiplier": [2], # keep if fixed to wtp and see when it can be produced locally
        "electricity_price_avg": [150],
        "electricity_availability_small": [50],
        "willingness_to_pay": [300],
        "hydrogen_import_price": [50,100, 150, 200, 250, 300],
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
        # "threads" viene aggiunto dal runner
    }

    # ========================================================================
    # CHOOSE SAMPLING METHOD
    # ========================================================================
    # Option 1: Use Latin Hypercube Sampling (RECOMMENDED for large grids)
    # Limits to max_samples (e.g., 100) using smart sampling
    combinations = generate_parameter_combinations(
        param_grid,
        method='lhs',       # 'lhs' or 'full'
        max_samples=1,    # Maximum number of samples
        seed=42             # For reproducibility
    )

    # Option 2: Use full grid (all combinations)
    # Uncomment this to use all possible combinations
    # combinations = generate_parameter_combinations(param_grid, method='full')

    # Prepare run configs
    run_configs = [(f"parallel_run_{i:04d}", params) for i, params in enumerate(combinations, 1)]

    print(f"\n[OK] Total runs to execute: {len(run_configs)}")

    # Create timestamp for results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = base_path / "results" / f"parallel_creation_test_{timestamp}"

    # Create runner with explicit configuration
    # Option 1: Auto-detect everything (RECOMMENDED)
    runner = ParallelCreationAndGurobiOptimizationRunner(
        base_path=base_path,
        max_workers=None,  # Auto-detect (uses half of cores, max 8)
        threads_per_worker=None  # Auto-calculate based on workers
    )

    # Option 2: Explicit configuration examples
    #
    # For 14 cores (your laptop):
    # runner = ParallelCreationAndGurobiOptimizationRunner(
    #     base_path=base_path,
    #     max_workers=7,
    #     threads_per_worker=2
    # )
    #
    # For 48 cores (your VM):
    # - Conservative: 8 workers × 6 threads = 48 threads total
    # runner = ParallelCreationAndGurobiOptimizationRunner(
    #     base_path=base_path,
    #     max_workers=8,
    #     threads_per_worker=6
    # )
    #
    # - More parallel: 16 workers × 3 threads = 48 threads total
    # runner = ParallelCreationAndGurobiOptimizationRunner(
    #     base_path=base_path,
    #     max_workers=16,
    #     threads_per_worker=3
    # )

    results_summary = runner.run_parallel_optimization(
        run_configs=run_configs,
        results_base_folder=results_folder
    )

    print(f"\n[OK] Parallel creation + Gurobi-env optimization complete!")
    print(f"[RESULTS] Summary Excel: {results_folder / 'parallel_results_summary.xlsx'}")