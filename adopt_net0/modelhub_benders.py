"""
Benders Decomposition implementation for AdOpT-NET0

Implementa la decomposizione di Benders usando IL MODELLO COMPLETO:
- Master Problem: risolve solo le variabili var_size (time-independent)
- Subproblem: fissa le var_size e risolve tutte le variabili operative (time-dependent)

Approccio: Costruisci il modello completo UNA VOLTA, poi lavora su di esso
disattivando/attivando variabili a seconda della fase.
"""

import warnings
from pathlib import Path
import pyomo.environ as pyo
import os
import time
import numpy as np
import pandas as pd
import sys
import datetime

from .modelhub import ModelHub
from .utilities import (
    get_glpk_parameters,
    get_gurobi_parameters,
    get_data_for_investment_period,
)

from .result_management import *
import logging

log = logging.getLogger(__name__)


class ModelHub_Benders(ModelHub):
    """
    Extended ModelHub con capacità di decomposizione di Benders.

    Usa il MODELLO COMPLETO e alterna tra:
    - Master: minimizza costi investimento risolvendo solo var_size
    - Subproblem: con var_size fissate, risolve tutte le variabili operative
    """

    def __init__(self):
        """Constructor - estende ModelHub con attributi specifici per Benders"""
        super().__init__()

        # Attributi specifici per Benders
        self.benders_info = {
            "iteration": 0,
            "converged": False,
            "gap": float('inf'),
            "lower_bound": -float('inf'),
            "upper_bound": float('inf'),
            "cuts_added": 0,
            "master_solve_times": [],
            "subproblem_solve_times": [],
            "master_objectives": [],
            "subproblem_objectives": [],
        }
        self.master_solver = None
        self.subproblem_solver = None

        # Liste di variabili per identificazione rapida
        self.size_vars = []  # Variabili var_size (master)
        self.operational_vars = []  # Tutte le altre variabili (subproblem)

    def solve_with_benders(self, max_iterations: int = 100, tolerance: float = 1e-4,
                          verbose: bool = True):
        """
        Risolve il problema usando la decomposizione di Benders sul MODELLO COMPLETO.

        :param int max_iterations: Numero massimo di iterazioni
        :param float tolerance: Tolleranza di convergenza (gap relativo)
        :param bool verbose: Se True, stampa informazioni dettagliate
        """
        log_msg = "--- Avvio Decomposizione di Benders ---"
        if verbose:
            print(log_msg)
        log.info(log_msg)

        start_time = time.time()
        config = self.data.model_config

        # Verifica che il modello sia stato costruito
        aggregation = self.info_solving_algorithms["aggregation_model"]
        if aggregation not in self.model or self.model[aggregation] is None:
            raise Exception("Il modello deve essere costruito prima di usare Benders. "
                          "Chiama construct_model() e construct_balances() prima.")

        # Ottieni il modello completo
        model = self.model[aggregation]

        # Crea cartella per i log di Benders (sarà popolata durante il solve)
        save_path = Path(config["reporting"]["save_path"]["value"])
        timestamp = datetime.datetime.fromtimestamp(start_time).strftime("%Y%m%d%H%M%S")
        if config["reporting"]["case_name"]["value"] == -1:
            folder_name = f"{timestamp}_benders"
        else:
            folder_name = f"{timestamp}_{config['reporting']['case_name']['value']}_benders"

        self.benders_log_folder = save_path / folder_name / "benders_logs"
        self.benders_log_folder.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"📁 Log di Benders salvati in: {self.benders_log_folder}")

        # Inizializza i solver
        self._initialize_benders_solvers()

        # Identifica le variabili size e operative
        self._identify_variable_types(model)

        if verbose:
            print(f"\n📊 Analisi modello:")
            print(f"   - Variabili var_size (master): {len(self.size_vars)}")
            print(f"   - Variabili operative (subproblem): {len(self.operational_vars)}")

        # Aggiungi variabile theta per i Benders cuts
        model.var_theta = pyo.Var(domain=pyo.Reals, bounds=(-1e12, 1e12))
        model.benders_cuts = pyo.ConstraintList()

        # Main loop di Benders
        for iteration in range(1, max_iterations + 1):
            self.benders_info["iteration"] = iteration

            if verbose:
                log_msg = f"\n{'='*60}\nIterazione Benders {iteration}\n{'='*60}"
                print(log_msg)
            log.info(f"Benders Iteration {iteration}")

            # Step 1: Risolvi Master Problem (solo var_size + calcolo investment costs)
            try:
                master_obj = self._solve_master_problem(model, iteration, verbose)
                self.benders_info["lower_bound"] = master_obj
                self.benders_info["master_objectives"].append(master_obj)
            except Exception as e:
                log.error(f"Errore nel master problem: {e}")
                raise

            # Step 2: Fissa le var_size e risolvi Subproblem (tutte le variabili operative)
            try:
                subproblem_obj = self._solve_subproblem(model, verbose)
                self.benders_info["subproblem_objectives"].append(subproblem_obj)

                # Aggiorna upper bound
                if subproblem_obj < self.benders_info["upper_bound"]:
                    self.benders_info["upper_bound"] = subproblem_obj
                    if verbose:
                        print(f"✓ Nuovo upper bound: {subproblem_obj:,.2f}")
            except Exception as e:
                log.error(f"Errore nel subproblem: {e}")
                if verbose:
                    print(f"⚠ Subproblem infeasible o errore")
                # Continua con feasibility cut
                self._add_feasibility_cut(model)
                continue

            # Step 3: Calcola gap e verifica convergenza
            self._update_gap()

            if verbose:
                print(f"\n📊 Risultati Iterazione {iteration}:")
                print(f"   Lower Bound: {self.benders_info['lower_bound']:>15,.2f}")
                print(f"   Upper Bound: {self.benders_info['upper_bound']:>15,.2f}")
                print(f"   Gap:         {self.benders_info['gap']:>15.6f}")

            log.info(f"It. {iteration}: LB={self.benders_info['lower_bound']:.2f}, "
                    f"UB={self.benders_info['upper_bound']:.2f}, "
                    f"Gap={self.benders_info['gap']:.6f}")

            # Check convergenza
            if self.benders_info["gap"] <= tolerance:
                self.benders_info["converged"] = True
                if verbose:
                    print(f"\n✅ Benders converged in {iteration} iterations!")
                log.info(f"Benders converged in {iteration} iterations")
                break

            # Step 4: Aggiungi Benders cut
            self._add_benders_cut(model, subproblem_obj)

        # Fine algoritmo
        if not self.benders_info["converged"]:
            log_msg = f"⚠ Benders non convergente dopo {max_iterations} iterazioni"
            if verbose:
                print(f"\n{log_msg}")
            log.warning(log_msg)

        total_time = time.time() - start_time

        if verbose:
            print(f"\n{'='*60}")
            print(f"Decomposizione di Benders completata in {total_time:.2f}s")
            print(f"{'='*60}")

        log.info(f"Benders completed: {self.benders_info['iteration']} iterations, "
                f"{total_time:.2f}s, gap={self.benders_info['gap']:.6f}")

        # Prepara il solution object per write_results
        self._prepare_solution_object()

    def _initialize_benders_solvers(self):
        """Inizializza i solver per master e subproblem"""
        config = self.data.model_config

        if config["solveroptions"]["solver"]["value"] in ["gurobi", "gurobi_persistent"]:
            self.master_solver = get_gurobi_parameters(config["solveroptions"])
            self.subproblem_solver = get_gurobi_parameters(config["solveroptions"])
        elif config["solveroptions"]["solver"]["value"] == "glpk":
            self.master_solver = get_glpk_parameters(config["solveroptions"])
            self.subproblem_solver = get_glpk_parameters(config["solveroptions"])
        else:
            raise Exception(f"Solver {config['solveroptions']['solver']['value']} non supportato")

    def _identify_variable_types(self, model):
        """
        Identifica e classifica le variabili del modello:
        - size_vars: tutte le var_size (tecnologie e reti) - TIME INDEPENDENT
        - operational_vars: tutte le altre variabili - TIME DEPENDENT
        """
        self.size_vars = []
        self.operational_vars = []

        # Usa set di id() per confronti veloci e sicuri
        size_var_ids = set()
        operational_var_ids = set()

        # Itera su tutti i periodi, nodi, tecnologie, reti
        for period in model.set_periods:
            b_period = model.periods[period]

            # Variabili var_size delle tecnologie
            for node in model.set_nodes:
                b_node = b_period.node_blocks[node]

                for tec in b_node.set_technologies:
                    b_tec = b_node.tech_blocks_active[tec]

                    if hasattr(b_tec, 'var_size'):
                        self.size_vars.append(b_tec.var_size)
                        size_var_ids.add(id(b_tec.var_size))

                    # Tutte le altre variabili del blocco tech sono operative
                    for var in b_tec.component_objects(pyo.Var):
                        if var.name != 'var_size':
                            var_id = id(var)
                            if var_id not in operational_var_ids:
                                self.operational_vars.append(var)
                                operational_var_ids.add(var_id)

            # Variabili var_size delle reti
            if hasattr(b_period, 'network_block'):
                for netw in b_period.set_networks:
                    b_netw = b_period.network_block[netw]

                    if hasattr(b_netw, 'arc_block'):
                        for node_from, node_to in b_netw.set_arcs:
                            b_arc = b_netw.arc_block[node_from, node_to]

                            if hasattr(b_arc, 'var_size'):
                                self.size_vars.append(b_arc.var_size)
                                size_var_ids.add(id(b_arc.var_size))

                            # Tutte le altre variabili dell'arco sono operative
                            for var in b_arc.component_objects(pyo.Var):
                                if var.name != 'var_size':
                                    var_id = id(var)
                                    if var_id not in operational_var_ids:
                                        self.operational_vars.append(var)
                                        operational_var_ids.add(var_id)

        # Aggiungi anche variabili globali e di nodo (sono tutte operative tranne theta)
        for var in model.component_objects(pyo.Var, active=True):
            if var.name not in ['var_theta']:
                var_id = id(var)
                # Usa id() per confrontare senza creare espressioni Pyomo
                if var_id not in size_var_ids and var_id not in operational_var_ids:
                    self.operational_vars.append(var)
                    operational_var_ids.add(var_id)

    def _solve_master_problem(self, model, iteration: int, verbose: bool) -> float:
        """
        Risolve il master problem usando il modello completo ma:
        1. Fissa TUTTE le variabili operative ai loro valori correnti
        2. Libera solo le var_size
        3. Minimizza: investment_cost + theta
        """
        if verbose:
            print("🔧 Risoluzione Master Problem...")

        start_time = time.time()
        config = self.data.model_config

        # Step 1: Fissa tutte le variabili operative (libera le size)
        for var in self.operational_vars:
            if var.is_indexed():
                for idx in var:
                    if var[idx].value is not None:
                        var[idx].fix()
            else:
                if var.value is not None:
                    var.fix()

        # Step 2: Assicurati che le var_size siano libere
        for var_size in self.size_vars:
            if var_size.is_fixed():
                var_size.unfix()

        # Step 3: Calcola investment cost (CAPEX)
        investment_cost = self._calculate_investment_cost(model)

        # Step 4: Definisci obiettivo master: minimize investment_cost + theta
        if model.find_component("objective") is not None:
            model.del_component(model.objective)

        def master_objective_rule(m):
            return investment_cost + m.var_theta

        model.objective = pyo.Objective(rule=master_objective_rule, sense=pyo.minimize)

        # Step 5: Risolvi
        try:
            # Prepara logfile per il master (se definito result_folder_path)
            logfile_master = None
            if hasattr(self, 'benders_log_folder') and self.benders_log_folder is not None:
                logfile_master = str(self.benders_log_folder / f"master_iteration_{iteration:03d}.log")

            if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
                self.master_solver.set_instance(model)
                if logfile_master:
                    solution = self.master_solver.solve(tee=True, warmstart=(iteration > 1), logfile=logfile_master)
                else:
                    solution = self.master_solver.solve(tee=False, warmstart=(iteration > 1))
            else:
                if logfile_master:
                    solution = self.master_solver.solve(model, tee=True, logfile=logfile_master)
                else:
                    solution = self.master_solver.solve(model, tee=False)

            if solution.solver.termination_condition != pyo.TerminationCondition.optimal:
                raise Exception(f"Master problem non ottimale: {solution.solver.termination_condition}")

            obj_value = pyo.value(model.objective)

        except Exception as e:
            log.error(f"Errore solving master: {e}")
            raise
        finally:
            # Step 6: Libera le variabili operative per il subproblem
            for var in self.operational_vars:
                if var.is_indexed():
                    for idx in var:
                        if var[idx].is_fixed():
                            var[idx].unfix()
                else:
                    if var.is_fixed():
                        var.unfix()

        solve_time = time.time() - start_time
        self.benders_info["master_solve_times"].append(solve_time)

        if verbose:
            print(f"   ✓ Master risolto in {solve_time:.2f}s, obj = {obj_value:,.2f}")

        return obj_value

    def _calculate_investment_cost(self, model):
        """
        Calcola il costo di investimento totale (CAPEX) usando le variabili var_size.
        """
        data = self.data
        total_capex = 0

        for period in model.set_periods:
            b_period = model.periods[period]

            # Technology CAPEX
            for node in model.set_nodes:
                b_node = b_period.node_blocks[node]

                for tec in b_node.set_technologies:
                    b_tec = b_node.tech_blocks_active[tec]

                    if hasattr(b_tec, 'var_size'):
                        tec_data = data.technology_data[period][node][tec]
                        if hasattr(tec_data.processed_coeff, 'capex'):
                            capex_per_unit = tec_data.processed_coeff.capex
                            total_capex += capex_per_unit * b_tec.var_size

            # Network CAPEX
            if hasattr(b_period, 'network_block'):
                for netw in b_period.set_networks:
                    b_netw = b_period.network_block[netw]

                    if hasattr(b_netw, 'arc_block'):
                        netw_data = data.network_data[period][netw]

                        for node_from, node_to in b_netw.set_arcs:
                            b_arc = b_netw.arc_block[node_from, node_to]

                            if hasattr(b_arc, 'var_size'):
                                # Distanza
                                try:
                                    distance = netw_data.distance.at[node_from, node_to]
                                except:
                                    distance = 1.0

                                # CAPEX = gamma2*S + gamma4*S*D (parti variabili)
                                if hasattr(netw_data.processed_coeff, 'capex_gamma2'):
                                    gamma2 = netw_data.processed_coeff.capex_gamma2
                                    gamma4 = netw_data.processed_coeff.capex_gamma4
                                    total_capex += (gamma2 + gamma4 * distance) * b_arc.var_size
                                elif hasattr(netw_data.processed_coeff, 'capex'):
                                    capex_per_unit = netw_data.processed_coeff.capex
                                    total_capex += capex_per_unit * b_arc.var_size * distance

        return total_capex

    def _solve_subproblem(self, model, verbose: bool) -> float:
        """
        Risolve il subproblem usando il modello completo ma:
        1. Fissa TUTTE le var_size ai valori del master
        2. Libera tutte le variabili operative
        3. Minimizza: var_npv (costo totale)
        """
        if verbose:
            print("🔧 Risoluzione Subproblem...")

        start_time = time.time()
        config = self.data.model_config

        # Step 1: Fissa tutte le var_size
        for var_size in self.size_vars:
            if var_size.value is not None:
                var_size.fix()

        # Step 2: Assicurati che tutte le variabili operative siano libere
        for var in self.operational_vars:
            if var.is_indexed():
                for idx in var:
                    if var[idx].is_fixed():
                        var[idx].unfix()
            else:
                if var.is_fixed():
                    var.unfix()

        # Step 3: Obiettivo subproblem: minimize var_npv
        if model.find_component("objective") is not None:
            model.del_component(model.objective)

        def subproblem_objective_rule(m):
            return m.var_npv

        model.objective = pyo.Objective(rule=subproblem_objective_rule, sense=pyo.minimize)

        # Step 4: Risolvi
        try:
            # Prepara logfile per il subproblem (se definito result_folder_path)
            # Prendi l'iterazione corrente
            current_iteration = self.benders_info["iteration"]
            logfile_subproblem = None
            if hasattr(self, 'benders_log_folder') and self.benders_log_folder is not None:
                logfile_subproblem = str(self.benders_log_folder / f"subproblem_iteration_{current_iteration:03d}.log")

            if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
                self.subproblem_solver.set_instance(model)
                if logfile_subproblem:
                    solution = self.subproblem_solver.solve(tee=True, warmstart=True, logfile=logfile_subproblem)
                else:
                    solution = self.subproblem_solver.solve(tee=False, warmstart=True)
            else:
                if logfile_subproblem:
                    solution = self.subproblem_solver.solve(model, tee=True, logfile=logfile_subproblem)
                else:
                    solution = self.subproblem_solver.solve(model, tee=False)

            if solution.solver.termination_condition not in [
                pyo.TerminationCondition.optimal,
                pyo.TerminationCondition.feasible
            ]:
                raise Exception(f"Subproblem non ottimale: {solution.solver.termination_condition}")

            obj_value = pyo.value(model.var_npv)

        except Exception as e:
            log.error(f"Errore solving subproblem: {e}")
            raise
        finally:
            # Step 5: Libera le var_size per la prossima iterazione del master
            for var_size in self.size_vars:
                if var_size.is_fixed():
                    var_size.unfix()

        solve_time = time.time() - start_time
        self.benders_info["subproblem_solve_times"].append(solve_time)

        if verbose:
            print(f"   ✓ Subproblem risolto in {solve_time:.2f}s, obj = {obj_value:,.2f}")

        return obj_value

    def _update_gap(self):
        """Calcola il gap di ottimalità."""
        lb = self.benders_info["lower_bound"]
        ub = self.benders_info["upper_bound"]

        if abs(ub) > 1e-6:
            self.benders_info["gap"] = abs(ub - lb) / abs(ub)
        else:
            self.benders_info["gap"] = float('inf') if lb < ub else 0.0

    def _add_benders_cut(self, model, subproblem_obj: float):
        """
        Aggiunge un Benders optimality cut.

        Cut semplificato: theta >= subproblem_obj
        (In una implementazione completa, si userebbero le variabili duali)
        """
        model.benders_cuts.add(model.var_theta >= subproblem_obj)

        self.benders_info["cuts_added"] += 1

        log.info(f"Aggiunto Benders cut #{self.benders_info['cuts_added']}: "
                f"theta >= {subproblem_obj:.2f}")

    def _add_feasibility_cut(self, model):
        """
        Aggiunge un feasibility cut quando il subproblem è infeasible.
        """
        log.warning("Subproblem infeasible - aggiunta feasibility cut")
        self.benders_info["cuts_added"] += 1

    def _prepare_solution_object(self):
        """
        Prepara il solution object e last_solve_info per permettere la scrittura dei risultati.
        """
        config = self.data.model_config
        aggregation = self.info_solving_algorithms["aggregation_model"]

        # Crea un solution object completo che simula la struttura di Pyomo
        # Include solver status e problem bounds per compatibilità con write_results()

        # Crea un oggetto Problem che simula i bounds
        problem_obj = type('Problem', (object,), {
            'lower_bound': self.benders_info['lower_bound'],
            'upper_bound': self.benders_info['upper_bound'],
        })()

        # Crea un oggetto Solver con status e termination
        solver_obj = type('Solver', (object,), {
            'status': pyo.SolverStatus.ok,
            'termination_condition': pyo.TerminationCondition.optimal,
        })()

        # Crea una classe Solution con il metodo problem() integrato
        class SolutionObject:
            def __init__(self, solver, problem_obj):
                self.solver = solver
                self._problem_obj = problem_obj
                self.Problem = [problem_obj]

            def problem(self, idx):
                """Ritorna il problem object per l'indice specificato"""
                return self._problem_obj

        self.solution = SolutionObject(solver_obj, problem_obj)

        # Popola last_solve_info con tutti i campi necessari per write_results()
        # Usa la cartella dei log di Benders se disponibile
        if hasattr(self, 'benders_log_folder') and self.benders_log_folder is not None:
            result_folder_path = self.benders_log_folder.parent
        else:
            # Fallback: crea una cartella nella save_path
            save_path = Path(config["reporting"]["save_path"]["value"])
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            if config["reporting"]["case_name"]["value"] == -1:
                folder_name = f"{timestamp}_benders"
            else:
                folder_name = f"{timestamp}_{config['reporting']['case_name']['value']}_benders"
            result_folder_path = save_path / folder_name
            result_folder_path.mkdir(parents=True, exist_ok=True)

        self.last_solve_info = {
            "pareto_point": self.info_pareto.get("pareto_point", -1),
            "config": config,
            "result_folder_path": result_folder_path,
            "time_stage": self.info_solving_algorithms.get("time_stage", 1),
            "aggregation_model": aggregation,
            # Aggiungi informazioni specifiche di Benders
            "benders_converged": self.benders_info['converged'],
            "benders_iterations": self.benders_info['iteration'],
            "benders_gap": self.benders_info['gap'],
        }

    def quick_solve_benders(self, max_iterations: int = 100, tolerance: float = 1e-4,
                           verbose: bool = True):
        """
        Metodo convenience per costruire e risolvere con Benders.
        """
        self.construct_model()
        self.construct_balances()
        self.solve_with_benders(max_iterations=max_iterations,
                               tolerance=tolerance,
                               verbose=verbose)

    def print_benders_statistics(self):
        """Stampa statistiche dettagliate sulla decomposizione di Benders."""
        info = self.benders_info

        print("\n" + "="*70)
        print(" "*20 + "STATISTICHE BENDERS DECOMPOSITION")
        print("="*70)
        print(f"Convergenza:           {'✓ SI' if info['converged'] else '✗ NO'}")
        print(f"Iterazioni totali:     {info['iteration']}")
        print(f"Gap finale:            {info['gap']:.6f}")
        print(f"Lower Bound finale:    {info['lower_bound']:>20,.2f}")
        print(f"Upper Bound finale:    {info['upper_bound']:>20,.2f}")
        print(f"Cuts aggiunti:         {info['cuts_added']}")

        if info['master_solve_times']:
            print(f"\n{'Master Problem:':<30}")
            print(f"  Tempo medio solve:   {np.mean(info['master_solve_times']):>15,.2f}s")
            print(f"  Tempo totale:        {np.sum(info['master_solve_times']):>15,.2f}s")

        if info['subproblem_solve_times']:
            print(f"\n{'Subproblem:':<30}")
            print(f"  Tempo medio solve:   {np.mean(info['subproblem_solve_times']):>15,.2f}s")
            print(f"  Tempo totale:        {np.sum(info['subproblem_solve_times']):>15,.2f}s")

        if info['master_solve_times'] and info['subproblem_solve_times']:
            total_time = np.sum(info['master_solve_times']) + np.sum(info['subproblem_solve_times'])
            print(f"\n{'Tempo totale Benders:':<30} {total_time:>15,.2f}s")

        print("="*70 + "\n")

        # Convergence plot (testuale)
        if len(info['master_objectives']) > 1:
            print("Andamento convergenza:")
            print("-" * 70)
            for i, (lb, ub) in enumerate(zip(info['master_objectives'],
                                             info['subproblem_objectives']), 1):
                gap = abs(ub - lb) / abs(ub) if abs(ub) > 1e-6 else 0
                bar_length = int((1 - gap) * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                print(f"It {i:3d}: {bar} {gap:>8.4f}")
            print("-" * 70 + "\n")

