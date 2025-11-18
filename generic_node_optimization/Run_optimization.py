"""
Optimization Runner with give input parameters
Handles single optimization runs with specified parameters
"""

import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
from define_topology import define_nodes
from define_topology import add_new_network_H2
from define_components_spec import define_hydrogen_pipeline1
from define_components_spec import define_hydrogen_pipeline2
from define_components_spec import define_hydrogen_storage
from define_components_spec import define_electrolyzers
from Change_data_excel import define_excel_data
from create_electricity_prices import create_electricity_prices

import adopt_net0 as adopt

class OptimizationRunner:
    def __init__(self, base_path):
        self.timesteps = None
        self.base_path = Path(base_path)
        self.wtp = None
        self.electricity_average_price = None
        self.electricity_availability_small = None
        self.h2_import_limit = None
        self.h2_import_price = None

    def run_optimization(self, run_id, params, results_base_folder):
        """Run an optimization with given data and parameters"""

        print(f"New optimization: {run_id}")

        #Setup paths:
        run_folder = results_base_folder / run_id
        input_data_path = run_folder / "input_data"
        results_data_path = run_folder / "userData"

        input_data_path.mkdir(parents=True, exist_ok=True)
        results_data_path.mkdir(parents=True, exist_ok=True)

        # Save parameters given for this run
        with open(run_folder / "run_params.json", "w") as f:
            json.dump(params, f, indent=4)

        # Setup base
        adopt.create_optimization_templates(input_data_path)
        nodes = ["BIG1", "BIG2", "SMALL1", "SMALL2", "SMALL3", "SMALL4", "STORAGE"]

        self.timesteps = 8760
        self.wtp = params["willingness_to_pay"]
        self.electricity_average_price = params["electricity_price_avg"]
        self.electricity_availability_small = params["electricity_availability_small"]

        #Define topology
        self._configure_topology(input_data_path, nodes)

        # Calcola parametri derivati
        derived_params = self._calculate_derived_parameters(params)
        params.update(derived_params)
        self.h2_import_limit = params["h2_import_limit"]
        self.h2_import_price = params["hydrogen_import_price"]

        # Configure model
        self._configure_model(input_data_path, results_data_path, params)

        adopt.create_input_data_folder_template(input_data_path)

        # Load scenario nodes
        self._load_scenario_nodes(input_data_path, nodes, params["scenarios"])

        # Define and characterize the nodes
        define_nodes(input_data_path, params)

        # Copy over technology files
        adopt.copy_technology_data(input_data_path)
        adopt.copy_compressor_data(input_data_path)

        # Networks congiguration
        self._configure_networks(input_data_path, params["networks"])
        adopt.copy_network_data(input_data_path)
        add_new_network_H2(input_data_path, params["scenarios"])

        define_hydrogen_pipeline2(input_data_path)
        define_hydrogen_storage(input_data_path)
        define_electrolyzers(input_data_path)

        # Load carrier data

        self._load_carrier_data(input_data_path, nodes, params)

        # Solve
        m = adopt.ModelHub()
        m.read_data(input_data_path, start_period=0, end_period=1)
        m.quick_solve()

        result_folder_path = m.last_solve_info["result_folder_path"]
        # serialize params to JSON (use default=str for non-JSON types)
        text = json.dumps(params, indent=4, ensure_ascii=False, default=str)

        # write to `optimization_model_info.txt`
        (result_folder_path / "optimization_model_info.txt").write_text(text, encoding="utf-8")

        # Extract results
        result_info = self._extract_results(m, run_folder, params)

        return result_info

    def _configure_topology(self, input_data_path, nodes):
        """Configure Topology.json"""
        with open(input_data_path / "Topology.json", "r") as f:
            topology = json.load(f)
        # Nodes
        topology["nodes"] = nodes
        # Carriers:
        topology["carriers"] = ["electricity", "hydrogen"]
        # Investment periods:
        topology["investment_periods"] = ["period1"]
        with open(input_data_path / "Topology.json", "w") as f:
            json.dump(topology, f, indent=4)

    def _configure_model(self, input_data_path, results_data_path, params):
        """Configure ConfigModel.json"""
        with open(input_data_path / "ConfigModel.json", "r") as f:
            configuration = json.load(f)

        # Set time aggregation settings:
        configuration["optimization"]["typicaldays"]["N"]["value"] = 0
        configuration["optimization"]["typicaldays"]["method"]["value"] = 1
        # Set MILP gap
        configuration["solveroptions"]["mipgap"]["value"] = 0.01
        configuration["solveroptions"]["mipfocus"]["value"] = 0
        configuration["solveroptions"]["presolve"]["value"] = -1
        configuration["solveroptions"]["heuristics"]["value"] = 0.05
        configuration["solveroptions"]["cuts"]["value"] = -1
        configuration["solveroptions"]["lpwarmstart"]["value"] = 0
        configuration["solveroptions"]["NoRelHeurTime"] = 0
        configuration["solveroptions"]["timelim"]["value"] = 50
        #configuration["solveroptions"]["threads"]["value"] = 48

        configuration["reporting"]["save_path"]["value"] = str(results_data_path)
        configuration["reporting"]["save_summary_path"]["value"] = str(results_data_path)

        # configuration["optimization"]["objective"]["value"] = "supply_willingness_to_pay"
        configuration["optimization"]["objective"]["value"] = "distance_willingness_to_pay"
        configuration["optimization"]["willingness_to_pay"]["value"] = self.wtp

        # Set pressure consideration
        configuration["performance"]["pressure"]["pressure_on"]["value"] = 1
        configuration["performance"]["pressure"]["pressure_carriers"]["value"] = ["hydrogen"]

        # Save json template
        with open(input_data_path / "ConfigModel.json", "w") as json_file:
            json.dump(configuration, json_file, indent=4)

    def _load_scenario_nodes(self, input_data_path, nodes, scenario):
        """Load coordinates for node from scenarios"""
        scenario_file = self.base_path / "input_data" / "scenarios" / f"NodeLocations_{scenario}.csv"

        if not scenario_file.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_file}")

        scenario_nodes = pd.read_csv(scenario_file, sep=';')
        print(scenario_nodes)
        # Create dictionaries from the scenario file
        node_lon = {}
        node_lat = {}
        node_alt = {}

        for _, row in scenario_nodes.iterrows():
            node_name = row['Node']
            if node_name in nodes:  # Only use nodes that exist in our system
                node_lon[node_name] = row['lon']
                node_lat[node_name] = row['lat']
                node_alt[node_name] = row['alt']

        # Load or create the main NodeLocations.csv
        node_location = pd.read_csv(input_data_path / "NodeLocations.csv", sep=';', index_col=0, header=0)

        for node in nodes:
            if node in node_lon:  # Only update if we have coordinates for this node
                node_location.at[node, 'lon'] = node_lon[node]
                node_location.at[node, 'lat'] = node_lat[node]
                node_location.at[node, 'alt'] = node_alt[node]

        node_location = node_location.reset_index()
        node_location.to_csv(input_data_path / "NodeLocations.csv", sep=';', index=False)

    def _configure_networks(self, input_data_path, network_list):
        """Configure Networks.json"""
        with open(input_data_path / "period1" / "Networks.json", "r") as f:
            networks = json.load(f)
        networks["existing"] = []
        networks["new"] = network_list
        with open(input_data_path / "period1" / "Networks.json", "w") as f:
            json.dump(networks, f, indent=4)

    def _load_carrier_data(self, input_data_path, nodes, params):
        """Carica carrier data dai file Excel generati"""
        n_timestep = self.timesteps

        data_folder = input_data_path/"data"
        define_excel_data(data_folder, params)
        create_electricity_prices(data_folder, nodes, params)

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
            adopt.fill_carrier_data(input_data_path, value_or_data=self.electricity_availability_small,
                                  columns=['Import limit'], carriers=['electricity'], nodes=[node])

        # BIG nodes - hydrogen import availability
        for node in ["BIG1", "BIG2"]:
            import_for_node = self.h2_import_limit/2
            adopt.fill_carrier_data(input_data_path, value_or_data=import_for_node,
                                  columns=['Import limit'], carriers=['hydrogen'], nodes=[node])
            adopt.fill_carrier_data(input_data_path, value_or_data=self.h2_import_price,
                                  columns=['Import price'], carriers=['hydrogen'], nodes=[node])
            adopt.fill_carrier_data(input_data_path, value_or_data=2000,  # BIG nodes have 2000 MW
                                  columns=['Import limit'], carriers=['electricity'], nodes=[node])
            # Use dynamic electricity prices for BIG nodes too
            adopt.fill_carrier_data(input_data_path, value_or_data=100, columns=['Import price'],
                                    carriers=['electricity'], nodes=[node])

    def _calculate_derived_parameters(self, params):
        """Calculate derived parameters based on input params"""

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

    def _extract_results(self, m, run_folder, params):
        """Extract results from the model after solving"""
        import pyomo.environ as pyo

        # Get the Pyomo model object from ModelHub
        model = None
        if hasattr(m, "model"):
            if isinstance(m.model, dict):
                model = m.model.get("full") or next(iter(m.model.values()))
            else:
                model = m.model

        if model is None:
            print("⚠️  Could not find Pyomo model object")
            return {}

        # Helper function to extract scalar values
        def get_value(component_name, pyo_type):
            try:
                comp = getattr(model, component_name, None)
                if comp is not None:
                    val = getattr(comp, "value", None)
                    if val is not None:
                        return float(val)
                    # If indexed, try first element
                    try:
                        first_idx = next(iter(comp))
                        return float(comp[first_idx])
                    except:
                        pass
            except Exception as e:
                print(f"⚠️  Error extracting {component_name}: {e}")
            return None

        # Extract values
        var_npv = get_value("var_npv", pyo.Var)
        para_total_demand = get_value("para_total_demand", pyo.Param)

        # Get WTP from params
        wtp = params.get("willingness_to_pay")
        if isinstance(wtp, (list, tuple)):
            wtp = wtp[0] if len(wtp) > 0 else None
        try:
            wtp = float(wtp) if wtp is not None else None
        except:
            wtp = None

        # Calculate derived metrics
        npv_over_demand = None
        if var_npv is not None and var_npv != 0 and para_total_demand is not None:
            npv_over_demand =var_npv / para_total_demand

        demand_times_wtp = None
        if para_total_demand is not None and wtp is not None:
            demand_times_wtp = para_total_demand * wtp

        # Extract objective value
        objective_value = None
        try:
            for obj in model.component_objects(pyo.Objective, active=True):
                objective_value = pyo.value(obj)
                break
        except Exception as e:
            print(f"⚠️  Error extracting objective: {e}")

        # Prepare result dictionary
        result_info = {
            "var_npv": var_npv,
            "para_total_demand": para_total_demand,
            "npv_over_demand": npv_over_demand,
            "demand_times_wtp": demand_times_wtp,
            "willingness_to_pay": wtp,
            "objective_value": objective_value,
        }

        # Print summary
        print("\n" + "="*80)
        print("RUN RESULTS SUMMARY")
        print("="*80)
        for key, val in result_info.items():
            print(f"  {key}: {val}")
        print("="*80 + "\n")

        # Save to JSON file
        result_folder_path = Path(m.last_solve_info["result_folder_path"])
        result_file = result_folder_path / "optimization_results_summary.json"
        with open(result_file, "w") as f:
            json.dump(result_info, f, indent=4, default=str)

        print(f"✅ Results saved to: {result_file}")

        return result_info
