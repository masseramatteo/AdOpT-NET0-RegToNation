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

import adopt_net0 as adopt

base_path =Path(__file__).parent
# Create folder for results
results_data_path = base_path/"userData"
results_data_path.mkdir(parents=True, exist_ok=True)

# Create input data path and optimization templates
input_data_path = base_path
input_data_path.mkdir(parents=True, exist_ok=True)

adopt.create_optimization_templates(input_data_path)
nodes = ["BIG1", "BIG2", 
         "SMALL1", "SMALL2", "SMALL3", "SMALL4", 
         "STORAGE"]
# Load json template
with open(input_data_path / "Topology.json", "r") as json_file:
    topology = json.load(json_file)
# Nodes
topology["nodes"] = nodes
# Carriers:
topology["carriers"] = ["electricity", "hydrogen"]
# Investment periods:
topology["investment_periods"] = ["period1"]
# Save json template
with open(input_data_path / "Topology.json", "w") as json_file:
    json.dump(topology, json_file, indent=4)

# Load json template
with open(input_data_path / "ConfigModel.json", "r") as json_file:
    configuration = json.load(json_file)
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
# configuration["solveroptions"]["threads"]["value"] = 48

configuration["reporting"]["save_path"]["value"] = str(results_data_path)
configuration["reporting"]["save_summary_path"]["value"] = str(results_data_path)

# configuration["optimization"]["objective"]["value"] = "supply_willingness_to_pay"
configuration["optimization"]["objective"]["value"] = "distance_willingness_to_pay"
configuration["optimization"]["willingness_to_pay"]["value"] = 350

# Set pressure consideration
configuration["performance"]["pressure"]["pressure_on"]["value"] = 1
configuration["performance"]["pressure"]["pressure_carriers"]["value"] = ["hydrogen"]

# Save json template
with open(input_data_path / "ConfigModel.json", "w") as json_file:
    json.dump(configuration, json_file, indent=4)

adopt.create_input_data_folder_template(input_data_path)

# Define node locations - read from generated scenario file
scenario_to_use = "1751"  # Start with first scenario for testing
scenario_node_file = base_path/"input_data"/"scenarios" / f"NodeLocations_{scenario_to_use}.csv"

if scenario_node_file.exists():
    # Read the generated scenario coordinates
    scenario_nodes = pd.read_csv(scenario_node_file, sep=';')
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
    
    # Verify all required nodes have coordinates
    missing_nodes = [node for node in nodes if node not in node_lon]
    if missing_nodes:
        # Use default coordinates for missing nodes
        for node in missing_nodes:
            node_lon[node] = 4.4777  # default lon
            node_lat[node] = 51.9244  # default lat
            node_alt[node] = 0        # default alt
else:
    raise FileNotFoundError(f"❌ Scenario file not found: {scenario_node_file}\n"
                          f"Please run 'generate_node_topology.py' first to generate the scenario files.\n"
                          f"Expected file: {scenario_node_file}")

# Load or create the main NodeLocations.csv
node_location = pd.read_csv(input_data_path / "NodeLocations.csv", sep=';', index_col=0, header=0)

for node in nodes:
    if node in node_lon:  # Only update if we have coordinates for this node
        node_location.at[node, 'lon'] = node_lon[node]
        node_location.at[node, 'lat'] = node_lat[node]
        node_location.at[node, 'alt'] = node_alt[node]

node_location = node_location.reset_index()
node_location.to_csv(input_data_path / "NodeLocations.csv", sep=';', index=False)

# adopt.show_available_technologies()
# adopt.show_available_networks()

# define and characterize the nodes
define_nodes(input_data_path)

# Copy over technology files
adopt.copy_technology_data(input_data_path)
adopt.copy_compressor_data(input_data_path)

# Add networks
with open(input_data_path / "period1" / "Networks.json", "r") as json_file:
    networks = json.load(json_file)
networks["existing"] = [] #"electricitySimple" "hydrogenPipelineOnshore_highP_big"
networks["new"] = ["hydrogenPipelineOnshore_lowP", #"hydrogenPipelineOnshore_lowP_small",
                   "hydrogenPipelineOnshore_highP" ] #"hydrogenTruck_highP"


with open(input_data_path / "period1" / "Networks.json", "w") as json_file:
    json.dump(networks, json_file, indent=4)

# define and characterize the hydrogen existing infrastructure (Hydrogen Backbone)
#add_existing_network_H2(input_data_path)

adopt.copy_network_data(input_data_path)

add_new_network_H2(input_data_path, scenario_to_use)

define_hydrogen_pipeline2(input_data_path)
define_hydrogen_storage(input_data_path)
define_electrolyzers(input_data_path)

n_timestep = 1

hourly_data = {}
connection_pressure_data = {}
electricity_prices = {}  # Add electricity prices dictionary
el_demand = {}
hydrogen_demand = {}
hydrogen_demand_pressure = {}
hydrogen_export_pressure = {}
hydrogen_import_pressure = {}
hydrogen_GenProd_pressure = {}

for node in nodes:
    # Read hourly data
    hourly_data[node] = pd.read_excel(
        input_data_path / f"data/data_network_{node}.xlsx",
        header=0,
        nrows=n_timestep
    )

    # Read connection pressure data
    connection_pressure_data[node] = pd.read_excel(
        input_data_path / f"data/data_pressure_connections_{node}.xlsx",
        index_col=0
    )

    # Read electricity prices for each node
    electricity_prices[node] = pd.read_excel(
        input_data_path / f"data/electricity_prices_{node}.xlsx",
        header=0,
        nrows=n_timestep
    )['Electricity_Price_EUR_MWh']  # Extract the price column

    # Extract demand data
    el_demand[node] = hourly_data[node].iloc[:, 1]
    hydrogen_demand[node] = hourly_data[node].iloc[:, 0]

for node in nodes:
    hydrogen_demand_pressure[node] = float(connection_pressure_data[node].loc['demand', 'hydrogen'])
    hydrogen_export_pressure[node] = float(connection_pressure_data[node].loc['export', 'hydrogen'])
    hydrogen_import_pressure[node] = float(connection_pressure_data[node].loc['import', 'hydrogen'])
    hydrogen_GenProd_pressure[node] = float(connection_pressure_data[node].loc['generic_production', 'hydrogen'])

    adopt.fill_carrier_data(input_data_path, value_or_data=el_demand[node], columns=['Demand'],
                            carriers=['electricity'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=hydrogen_demand[node], columns=['Demand'],
                            carriers=['hydrogen'], nodes=[node])

    adopt.fill_carrier_pressure_data(input_data_path, pressure_value_bar=hydrogen_demand_pressure[node],
                                     connection=['Demand'], carriers=['hydrogen'], nodes=[node])
    adopt.fill_carrier_pressure_data(input_data_path, pressure_value_bar=hydrogen_export_pressure[node],
                                     connection=['Export'], carriers=['hydrogen'], nodes=[node])
    adopt.fill_carrier_pressure_data(input_data_path, pressure_value_bar=hydrogen_import_pressure[node],
                                     connection=['Import'], carriers=['hydrogen'], nodes=[node])
    adopt.fill_carrier_pressure_data(input_data_path, pressure_value_bar=hydrogen_GenProd_pressure[node],
                                     connection=['Generic production'], carriers=['hydrogen'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=50, columns=['Import limit'], carriers=['electricity'],
                            nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=electricity_prices[node], columns=['Import price'],
                            carriers=['electricity'], nodes=[node])

# for node in ["Dongen", "Arnhem", "Venlo"]:
#     adopt.fill_carrier_data(input_data_path, value_or_data=200, columns=['Import limit'], carriers=['electricity'], nodes=[node])
#     adopt.fill_carrier_data(input_data_path, value_or_data=300, columns=['Import price'], carriers=['electricity'], nodes=[node])


for node in ["BIG1", "BIG2"]:
    adopt.fill_carrier_data(input_data_path, value_or_data=1000, columns=['Import limit'], carriers=['hydrogen'],
                            nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=270, columns=['Import price'], carriers=['hydrogen'],
                            nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=2000, columns=['Import limit'], carriers=['electricity'],
                            nodes=[node])
    # Use dynamic electricity prices for BIG nodes too
    adopt.fill_carrier_data(input_data_path, value_or_data=100, columns=['Import price'],
                            carriers=['electricity'], nodes=[node])

# Define climate data
# adopt.load_climate_data_from_api(input_data_path)

m = adopt.ModelHub()
m.read_data(input_data_path)
m.quick_solve()

result_folder_path = m.last_solve_info["result_folder_path"]
user_input = input(
    "What did you change for this optimization? "
    "Please describe the model and the changes with respect to the previous ones:\n"
)
extra_file = result_folder_path / "optimization_model_info.txt"
extra_file.write_text(user_input)

print(f"Your input has been saved in: {extra_file}")