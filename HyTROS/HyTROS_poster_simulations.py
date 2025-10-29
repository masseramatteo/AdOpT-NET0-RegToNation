import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
from define_topology import define_nodes
from define_topology import add_existing_network_H2
from define_topology import add_new_network_H2_all
from define_topology import add_new_network_H2_small
from define_topology import add_new_network_H2_small_and_EHB
from define_topology import add_new_network_H2_small_and_EHB_fixed
from define_components_spec import define_hydrogen_pipeline1
from define_components_spec import define_hydrogen_pipeline2
from define_components_spec import define_hydrogen_storage

import adopt_net0 as adopt

# Create folder for results
results_data_path = Path(r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\FirstWork simulations\userData")
results_data_path.mkdir(parents=True, exist_ok=True)

# Create input data path and optimization templates
input_data_path = Path(r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\FirstWork simulations")
input_data_path.mkdir(parents=True, exist_ok=True)

adopt.create_optimization_templates(input_data_path)
nodes = ["Roermond", "Dongen", "Oosterhout", "Maastricht", "Venlo", "Arnhem", "East_Groningen", "Betuwe", "Heerlen", "Dordrecht", "Wageningen", # small clusters
                     "Rotterdam", "Zeeland", "North_Sea", "North_Netherlands", "Chemelot", # big clusters
                     "Zuidwending"] #storage
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
configuration["solveroptions"]["mipgap"]["value"] = 0.02
configuration["solveroptions"]["mipfocus"]["value"] = 0
configuration["solveroptions"]["presolve"]["value"] = -1
configuration["solveroptions"]["heuristics"]["value"] = 0.05
configuration["solveroptions"]["cuts"]["value"] = -1
configuration["solveroptions"]["lpwarmstart"]["value"] = 0
configuration["solveroptions"]["NoRelHeurTime"] = 1500
configuration["solveroptions"]["timelim"]["value"] = 300

# Set pressure consideration
configuration["performance"]["pressure"]["pressure_on"]["value"] = 1
configuration["performance"]["pressure"]["pressure_carriers"]["value"]= ["hydrogen"]

# Save json template
with open(input_data_path / "ConfigModel.json", "w") as json_file:
    json.dump(configuration, json_file, indent=4)

adopt.create_input_data_folder_template(input_data_path)

# Define node locations
node_location = pd.read_csv(input_data_path / "NodeLocations.csv", sep=';', index_col=0, header=0)
node_lon = {
    'Roermond': 5.9875,
    'Dongen': 4.9459,
    'Oosterhout': 4.8617,
    'Maastricht': 5.6910,
    'Venlo': 6.1670,
    'Arnhem': 5.8987,
    'East_Groningen': 6.7261,
    'Betuwe': 5.5000,
    'Heerlen': 5.9815,
    'Dordrecht': 4.6783,
    'Wageningen': 5.6654,
    'Rotterdam': 4.4777,
    'Zeeland': 3.8497,
    'North_Sea': 4.8170,
    'North_Netherlands': 6.8262,
    'Chemelot': 5.8004,
    'Zuidwending': 6.9332
}
node_lat = {
    'Roermond': 51.1942,
    'Dongen': 51.6247,
    'Oosterhout': 51.6410,
    'Maastricht': 50.8514,
    'Venlo': 51.3670,
    'Arnhem': 51.9851,
    'East_Groningen': 53.1722,
    'Heerlen': 50.8837,
    'Dordrecht': 51.7958,
    'Wageningen': 51.9692,
    'Betuwe': 51.9167,
    'Rotterdam': 51.9244,
    'Zeeland': 51.4988,
    'North_Sea': 52.4330,
    'North_Netherlands': 53.454370,
    'Chemelot': 50.9756,
    'Zuidwending': 53.0955
}
node_alt = {
    'Roermond': 0,
    'Dongen': 0,
    'Oosterhout': 0,
    'Maastricht': 0,
    'Venlo': 0,
    'Arnhem': 0,
    'East_Groningen': 0,
    'Heerlen': 0,
    'Dordrecht': 0,
    'Wageningen': 0,
    'Betuwe': 0,
    'Rotterdam': 0,
    'Zeeland': 0,
    'North_Sea': 0,
    'North_Netherlands': 0,
    'Chemelot': 0,
    'Zuidwending': 0
}
for node in ["Roermond", "Dongen", "Oosterhout", "Maastricht", "Venlo", "Arnhem", "East_Groningen", "Betuwe", "Heerlen", "Dordrecht", "Wageningen", # small clusters
                     "Rotterdam", "Zeeland", "North_Sea", "North_Netherlands", "Chemelot", # big clusters
                     "Zuidwending"]:
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
networks["existing"] = [] #"electricitySimple", "hydrogenPipelineOnshore_highP_big"
networks["new"] = ["hydrogenPipelineOnshore_lowP", #"hydrogenPipelineOnshore_lowP_small",
                   "hydrogenPipelineOnshore_highP" ] #"hydrogenTruck_highP"


with open(input_data_path / "period1" / "Networks.json", "w") as json_file:
    json.dump(networks, json_file, indent=4)

# define and characterize the hydrogen existing infrastructure (Hydrogen Backbone)
#add_existing_network_H2(input_data_path)

adopt.copy_network_data(input_data_path)

add_new_network_H2_small_and_EHB(input_data_path)

define_hydrogen_pipeline2(input_data_path)
define_hydrogen_storage(input_data_path)

n_timestep = 1

hourly_data = {}
connection_pressure_data = {}
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

    # Extract demand data
    el_demand[node] = hourly_data[node].iloc[:, 1]
    hydrogen_demand[node] = hourly_data[node].iloc[:, 0]


for node in nodes:

    hydrogen_demand_pressure[node] = float(connection_pressure_data[node].loc['demand', 'hydrogen'])
    hydrogen_export_pressure[node] = float(connection_pressure_data[node].loc['export', 'hydrogen'])
    hydrogen_import_pressure[node] = float(connection_pressure_data[node].loc['import', 'hydrogen'])
    hydrogen_GenProd_pressure[node] = float(connection_pressure_data[node].loc['generic_production', 'hydrogen'])

    adopt.fill_carrier_data(input_data_path, value_or_data=el_demand[node], columns=['Demand'], carriers=['electricity'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=hydrogen_demand[node], columns=['Demand'], carriers=['hydrogen'], nodes=[node])

    adopt.fill_carrier_pressure_data(input_data_path, pressure_value_bar=hydrogen_demand_pressure[node], connection=['Demand'], carriers=['hydrogen'], nodes=[node])
    adopt.fill_carrier_pressure_data(input_data_path, pressure_value_bar=hydrogen_export_pressure[node], connection=['Export'], carriers=['hydrogen'], nodes=[node])
    adopt.fill_carrier_pressure_data(input_data_path, pressure_value_bar=hydrogen_import_pressure[node], connection=['Import'], carriers=['hydrogen'], nodes=[node])
    adopt.fill_carrier_pressure_data(input_data_path, pressure_value_bar=hydrogen_GenProd_pressure[node], connection=['Generic production'], carriers=['hydrogen'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=60, columns=['Import limit'], carriers=['electricity'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=150, columns=['Import price'], carriers=['electricity'], nodes=[node])

# for node in ["Dongen", "Arnhem", "Venlo"]:
#     adopt.fill_carrier_data(input_data_path, value_or_data=500, columns=['Import limit'], carriers=['electricity'], nodes=[node])
#     adopt.fill_carrier_data(input_data_path, value_or_data=300, columns=['Import price'], carriers=['electricity'], nodes=[node])


for node in ["Rotterdam", "Zeeland", "North_Sea", "North_Netherlands", "Chemelot"]:
    adopt.fill_carrier_data(input_data_path, value_or_data=1000, columns=['Import limit'], carriers=['hydrogen'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=150, columns=['Import price'], carriers=['hydrogen'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=5000, columns=['Import limit'], carriers=['electricity'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=150, columns=['Import price'], carriers=['electricity'], nodes=[node])

for node in ["Chemelot"]:
    adopt.fill_carrier_data(input_data_path, value_or_data=0, columns=['Import limit'], carriers=['hydrogen'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=800, columns=['Demand'], carriers=['hydrogen'], nodes=[node])

# Define climate data
#adopt.load_climate_data_from_api(input_data_path)

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