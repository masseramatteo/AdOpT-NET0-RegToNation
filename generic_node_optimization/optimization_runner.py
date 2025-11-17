"""
Optimization Runner for Parametric Studies
Handles single optimization runs with specified parameters
"""

import json
import shutil
from pathlib import Path
import pandas as pd
import adopt_net0 as adopt
from define_topology import define_nodes, add_new_network_H2
from define_components_spec import (
    define_hydrogen_pipeline2,
    define_hydrogen_storage,
    define_electrolyzers
)
from data_generator import generate_excel_data


class OptimizationRunner:
    def __init__(self, base_path):
        self.base_path = Path(base_path)

    def run_optimization(self, run_id, params, results_base_folder):
        """Run an optimization with given data and parameters"""

        print(f"New optimization: {run_id}")

        # Setup
        run_folder = results_base_folder / run_id
        input_data_path = run_folder / "input_data"
        results_data_path = run_folder / "userData"

        input_data_path.mkdir(parents=True, exist_ok=True)
        results_data_path.mkdir(parents=True, exist_ok=True)

        # Save parameters given for this run
        with open(run_folder / "run_params.json", "w") as f:
            json.dump(params, f, indent=4)

        print(f"   📁 Cartella run: {run_folder}")

        # Setup base
        adopt.create_optimization_templates(input_data_path)
        nodes = ["BIG1", "BIG2", "SMALL1", "SMALL2", "SMALL3", "SMALL4", "STORAGE"]

        # Configure topology
        self._configure_topology(input_data_path, nodes)

        # Calcola parametri derivati
        derived_params = self._calculate_derived_parameters(params)
        params.update(derived_params)

        # Configure model
        self._configure_model(input_data_path, results_data_path, params)

        adopt.create_input_data_folder_template(input_data_path)

        # Load scenario nodes
        self._load_scenario_nodes(input_data_path, nodes, params["scenarios"])

        # Define components
        define_nodes(input_data_path)
        adopt.copy_technology_data(input_data_path)
        adopt.copy_compressor_data(input_data_path)

        # Networks
        self._configure_networks(input_data_path, params["networks"])
        adopt.copy_network_data(input_data_path)
        add_new_network_H2(input_data_path, params["scenarios"])

        define_hydrogen_pipeline2(input_data_path)
        define_hydrogen_storage(input_data_path)
        define_electrolyzers(input_data_path)

        # ⭐ GENERA DATI EXCEL BASATI SUI PARAMETRI
        print(f"   📊 Generazione dati Excel...")
        generate_excel_data(input_data_path, params)

        # Load carrier data
        print(f"   📥 Caricamento carrier data...")
        self._load_carrier_data(input_data_path, nodes, params)

        # Solve
        print(f"   🔧 Costruzione e risoluzione modello...")
        m = adopt.ModelHub()
        m.read_data(input_data_path)
        m.quick_solve()

        # Extract results
        result_info = self._extract_results(m, run_folder, params)

        print(f"   ✅ Ottimizzazione completata!")
        print(f"   📈 Objective: {result_info.get('objective_value', 'N/A')}")
        print(f"   ⏱️  Time: {result_info.get('solve_time', 'N/A')}s")

        return result_info

    def _calculate_derived_parameters(self, params):
        """Calcola parametri derivati da quelli principali"""

        total_demand = params["total_demand_TWh"]
        import_availability = params["import_availability_ratio"]
        el_price = params["electricity_price_avg"]
        import_multiplier = params["import_cost_multiplier"]

        # H2 import limit (MW) based on total demand and availability
        # Total demand in MW medio = TWh * 1e6 / 8760
        average_demand_MW = total_demand * 1e6 / 8760
        h2_import_limit = average_demand_MW * import_availability

        # H2 import price: electricity price * ratio
        h2_import_price = el_price * import_multiplier

        return {
            "h2_import_limit": h2_import_limit,
            "hydrogen_import_price": h2_import_price,
            "el_import_limit": params["electricity_availability_small"]
        }

    def _configure_topology(self, input_data_path, nodes):
        """Configura Topology.json"""
        with open(input_data_path / "Topology.json", "r") as f:
            topology = json.load(f)
        topology["nodes"] = nodes
        topology["carriers"] = ["electricity", "hydrogen"]
        topology["investment_periods"] = ["period1"]
        with open(input_data_path / "Topology.json", "w") as f:
            json.dump(topology, f, indent=4)

    def _configure_model(self, input_data_path, results_data_path, params):
        """Configura ConfigModel.json"""
        with open(input_data_path / "ConfigModel.json", "r") as f:
            config = json.load(f)

        # Typical days
        config["optimization"]["typicaldays"]["N"]["value"] = 0
        config["optimization"]["typicaldays"]["method"]["value"] = 1

        # Solver parameters
        config["solveroptions"]["mipgap"]["value"] = params.get("mipgap", 0.01)
        config["solveroptions"]["mipfocus"]["value"] = params.get("mipfocus", 0)
        config["solveroptions"]["presolve"]["value"] = params.get("presolve", -1)
        config["solveroptions"]["heuristics"]["value"] = params.get("heuristics", 0.05)
        config["solveroptions"]["cuts"]["value"] = params.get("cuts", -1)
        config["solveroptions"]["lpwarmstart"]["value"] = params.get("lpwarmstart", 0)
        config["solveroptions"]["timelim"]["value"] = params.get("time_limit", 50)
        config["solveroptions"]["threads"]["value"] = params.get("threads", 48)

        # Reporting
        config["reporting"]["save_path"]["value"] = str(results_data_path)
        config["reporting"]["save_summary_path"]["value"] = str(results_data_path)

        # Objective
        config["optimization"]["objective"]["value"] = params.get("objective", "distance_willingness_to_pay")

        # Pressure settings
        config["performance"]["pressure"]["pressure_on"]["value"] = params.get("pressure_on", 1)
        config["performance"]["pressure"]["pressure_carriers"]["value"] = params.get("pressure_carriers", ["hydrogen"])

        with open(input_data_path / "ConfigModel.json", "w") as f:
            json.dump(config, f, indent=4)

    def _configure_networks(self, input_data_path, network_list):
        """Configura Networks.json"""
        with open(input_data_path / "period1" / "Networks.json", "r") as f:
            networks = json.load(f)
        networks["existing"] = []
        networks["new"] = network_list
        with open(input_data_path / "period1" / "Networks.json", "w") as f:
            json.dump(networks, f, indent=4)

    def _load_scenario_nodes(self, input_data_path, nodes, scenario):
        """Carica coordinate nodi da scenario"""
        scenario_file = self.base_path / "input_data" / "scenarios" / f"NodeLocations_{scenario}.csv"

        if not scenario_file.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_file}")

        scenario_nodes = pd.read_csv(scenario_file, sep=';')
        node_location = pd.read_csv(input_data_path / "NodeLocations.csv", sep=';', index_col=0)

        for _, row in scenario_nodes.iterrows():
            node_name = row['Node']
            if node_name in nodes:
                node_location.at[node_name, 'lon'] = row['lon']
                node_location.at[node_name, 'lat'] = row['lat']
                node_location.at[node_name, 'alt'] = row['alt']

        node_location.reset_index().to_csv(input_data_path / "NodeLocations.csv", sep=';', index=False)

    def _load_carrier_data(self, input_data_path, nodes, params):
        """Carica carrier data dai file Excel generati"""
        n_timestep = 8760

        for node in nodes:
            # Load data
            hourly_data = pd.read_excel(
                input_data_path / f"data/data_network_{node}.xlsx",
                header=0, nrows=n_timestep
            )

            connection_pressure_data = pd.read_excel(
                input_data_path / f"data/data_pressure_connections_{node}.xlsx",
                index_col=0
            )

            el_prices = pd.read_excel(
                input_data_path / f"data/electricity_prices_{node}.xlsx",
                header=0, nrows=n_timestep
            )['Electricity_Price_EUR_MWh']

            # Extract demands
            hydrogen_demand = hourly_data.iloc[:, 0]
            el_demand = hourly_data.iloc[:, 1]

            # Pressures
            h2_demand_pressure = float(connection_pressure_data.loc['demand', 'hydrogen'])
            h2_export_pressure = float(connection_pressure_data.loc['export', 'hydrogen'])
            h2_import_pressure = float(connection_pressure_data.loc['import', 'hydrogen'])
            h2_genprod_pressure = float(connection_pressure_data.loc['generic_production', 'hydrogen'])

            # Fill carrier data
            adopt.fill_carrier_data(input_data_path, value_or_data=el_demand,
                                  columns=['Demand'], carriers=['electricity'], nodes=[node])
            adopt.fill_carrier_data(input_data_path, value_or_data=hydrogen_demand,
                                  columns=['Demand'], carriers=['hydrogen'], nodes=[node])

            # Pressures
            adopt.fill_carrier_pressure_data(input_data_path, pressure_value_bar=h2_demand_pressure,
                                           connection=['Demand'], carriers=['hydrogen'], nodes=[node])
            adopt.fill_carrier_pressure_data(input_data_path, pressure_value_bar=h2_export_pressure,
                                           connection=['Export'], carriers=['hydrogen'], nodes=[node])
            adopt.fill_carrier_pressure_data(input_data_path, pressure_value_bar=h2_import_pressure,
                                           connection=['Import'], carriers=['hydrogen'], nodes=[node])
            adopt.fill_carrier_pressure_data(input_data_path, pressure_value_bar=h2_genprod_pressure,
                                           connection=['Generic production'], carriers=['hydrogen'], nodes=[node])

            # Electricity prices and limits
            adopt.fill_carrier_data(input_data_path, value_or_data=el_prices,
                                  columns=['Import price'], carriers=['electricity'], nodes=[node])
            adopt.fill_carrier_data(input_data_path, value_or_data=params.get("el_import_limit", 50),
                                  columns=['Import limit'], carriers=['electricity'], nodes=[node])

        # BIG nodes - hydrogen import availability
        for node in ["BIG1", "BIG2"]:
            adopt.fill_carrier_data(input_data_path, value_or_data=params.get("h2_import_limit", 1000),
                                  columns=['Import limit'], carriers=['hydrogen'], nodes=[node])
            adopt.fill_carrier_data(input_data_path, value_or_data=params.get("hydrogen_import_price", 270),
                                  columns=['Import price'], carriers=['hydrogen'], nodes=[node])
            adopt.fill_carrier_data(input_data_path, value_or_data=2000,  # BIG nodes have 2000 MW
                                  columns=['Import limit'], carriers=['electricity'], nodes=[node])

    def _extract_results(self, model, run_folder, params):
        """Estrae metriche chiave e salva risultati"""
        result_folder = model.last_solve_info.get("result_folder_path")

        if result_folder:
            # Copia risultati
            shutil.copytree(result_folder, run_folder / "model_results", dirs_exist_ok=True)

        # Salva info aggiuntive sulla run
        info_text = f"""
================================================================================
PARAMETRIC RUN INFORMATION
================================================================================
Run ID: {run_folder.name}
Timestamp: {pd.Timestamp.now()}

SCENARIO PARAMETERS
-------------------
Scenario: {params.get('scenarios', 'N/A')}
Networks: {params.get('networks', 'N/A')}

DEMAND PARAMETERS
-----------------
Total Demand: {params.get('total_demand_TWh', 'N/A')} TWh
Demand Level Ratio (Big/Small): {params.get('demand_level_ratio', 'N/A')}

IMPORT PARAMETERS
-----------------
Import Availability Ratio: {params.get('import_availability_ratio', 'N/A')}
H2 Import Limit: {params.get('h2_import_limit', 'N/A'):.2f} MW
H2 Import Price: {params.get('hydrogen_import_price', 'N/A'):.2f} EUR/MWh
Import Cost Multiplier: {params.get('import_cost_multiplier', 'N/A')}

ELECTRICITY PARAMETERS
----------------------
Electricity Price (avg): {params.get('electricity_price_avg', 'N/A')} EUR/MWh
Electricity Availability (small): {params.get('electricity_availability_small', 'N/A')} MW

SOLVER PARAMETERS
-----------------
MIP Gap: {params.get('mipgap', 'N/A')}
Time Limit: {params.get('time_limit', 'N/A')} seconds
Threads: {params.get('threads', 'N/A')}
Objective: {params.get('objective', 'N/A')}

SOLVER RESULTS
==============
Objective Value: {model.last_solve_info.get("objective_value", "N/A")}
Solve Time: {model.last_solve_info.get("solve_time", "N/A")} seconds
Termination Condition: {model.last_solve_info.get("termination_condition", "N/A")}
Gap: {model.last_solve_info.get("gap", "N/A")}
================================================================================
"""

        (run_folder / "run_summary.txt").write_text(info_text)

        return {
            "objective_value": model.last_solve_info.get("objective_value"),
            "solve_time": model.last_solve_info.get("solve_time"),
            "termination_condition": model.last_solve_info.get("termination_condition"),
            "gap": model.last_solve_info.get("gap"),
            "result_folder": str(run_folder / "model_results") if result_folder else "N/A"
        }

