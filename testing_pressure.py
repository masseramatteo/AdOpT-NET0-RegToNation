import json
from pathlib import Path
import os
import pandas as pd
import numpy as np

import adopt_net0 as adopt

# Create folder for results
results_data_path = Path("./userData")
results_data_path.mkdir(parents=True, exist_ok=True)

# Create input data path and optimization templates
input_data_path = Path(r"C:\Users\Masse007\Documents\Code\CaseStudies\Adding_pressure")
input_data_path.mkdir(parents=True, exist_ok=True)

adopt.create_optimization_templates(input_data_path)

# Load json template
with open(input_data_path / "Topology.json", "r") as json_file:
    topology = json.load(json_file)
# Nodes
topology["nodes"] = ["Parma", "Utrecht", "Berlin"]
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
configuration["optimization"]["typicaldays"]["N"]["value"] = 30
configuration["optimization"]["typicaldays"]["method"]["value"] = 1
# Set MILP gap
configuration["solveroptions"]["mipgap"]["value"] = 0.02
# Set pressure consideration
configuration["performance"]["pressure"]["value"] = 1
configuration["performance"]["pressure"]["compressed_carrier"] = ["hydrogen"]

# Save json template
with open(input_data_path / "ConfigModel.json", "w") as json_file:
    json.dump(configuration, json_file, indent=4)

adopt.create_input_data_folder_template(input_data_path)

# Define node locations
node_location = pd.read_csv(
    input_data_path / "NodeLocations.csv", sep=";", index_col=0, header=0
)
node_lon = {"Parma": 5, "Utrecht": 5.25, "Berlin": 5.5}
node_lat = {"Parma": 50, "Utrecht": 52, "Berlin": 51}
node_alt = {"Parma": 0, "Utrecht": 0, "Berlin": 0}
for node in ["Parma", "Utrecht", "Berlin"]:
    node_location.at[node, "lon"] = node_lon[node]
    node_location.at[node, "lat"] = node_lat[node]
    node_location.at[node, "alt"] = node_alt[node]

node_location = node_location.reset_index()
node_location.to_csv(input_data_path / "NodeLocations.csv", sep=";", index=False)

adopt.show_available_technologies()

# Add required technologies for node 'Parma'
with open(
    input_data_path / "period1" / "node_data" / "Parma" / "Technologies.json", "r"
) as json_file:
    technologies = json.load(json_file)
technologies["new"] = ["Electrolyzer", "FuelCell"]

with open(
    input_data_path / "period1" / "node_data" / "Parma" / "Technologies.json", "w"
) as json_file:
    json.dump(technologies, json_file, indent=4)

# Add required technologies for node 'Utrecht'
with open(
    input_data_path / "period1" / "node_data" / "Utrecht" / "Technologies.json", "r"
) as json_file:
    technologies = json.load(json_file)
technologies["new"] = ["Storage_Battery", "WindTurbine_Onshore_4000", "Electrolyzer"]

with open(
    input_data_path / "period1" / "node_data" / "Utrecht" / "Technologies.json", "w"
) as json_file:
    json.dump(technologies, json_file, indent=4)

# Add required technologies for node 'Berlin'
with open(
    input_data_path / "period1" / "node_data" / "Berlin" / "Technologies.json", "r"
) as json_file:
    technologies = json.load(json_file)
technologies["new"] = ["Electrolyzer", "Storage_Battery"]
technologies["existing"] = {"FuelCell": 1000}

with open(
    input_data_path / "period1" / "node_data" / "Berlin" / "Technologies.json", "w"
) as json_file:
    json.dump(technologies, json_file, indent=4)

# Copy over technology files
adopt.copy_technology_data(input_data_path)

adopt.show_available_networks()

# Add networks
with open(input_data_path / "period1" / "Networks.json", "r") as json_file:
    networks = json.load(json_file)
networks["new"] = ["hydrogenPipelineOnshore_lowP", "electricitySimple"]
networks["existing"] = ["hydrogenPipelineOnshore_lowP"]

with open(input_data_path / "period1" / "Networks.json", "w") as json_file:
    json.dump(networks, json_file, indent=4)

# Make a new folder for the existing network
os.makedirs(
    input_data_path
    / "period1"
    / "network_topology"
    / "existing"
    / "hydrogenPipelineOnshore_lowP",
    exist_ok=True,
)

print("Existing network")
# Use the templates, fill and save them to the respective directory
# Connection
connection = pd.read_csv(
    input_data_path / "period1" / "network_topology" / "existing" / "connection.csv",
    sep=";",
    index_col=0,
)
connection.loc["Parma", "Utrecht"] = 1
connection.loc["Utrecht", "Parma"] = 0
connection.loc["Utrecht", "Berlin"] = 1
connection.loc["Berlin", "Utrecht"] = 1
connection.to_csv(
    input_data_path
    / "period1"
    / "network_topology"
    / "existing"
    / "hydrogenPipelineOnshore_lowP"
    / "connection.csv",
    sep=";",
)
print("Connection:", connection)

# Delete the template
os.remove(
    input_data_path / "period1" / "network_topology" / "existing" / "connection.csv"
)

# Distance
distance = pd.read_csv(
    input_data_path / "period1" / "network_topology" / "existing" / "distance.csv",
    sep=";",
    index_col=0,
)
distance.loc["Parma", "Utrecht"] = 50
distance.loc["Utrecht", "Parma"] = 50
distance.loc["Utrecht", "Berlin"] = 35
distance.loc["Berlin", "Utrecht"] = 35

distance.to_csv(
    input_data_path
    / "period1"
    / "network_topology"
    / "existing"
    / "hydrogenPipelineOnshore_lowP"
    / "distance.csv",
    sep=";",
)
print("Distance:", distance)

# Delete the template
os.remove(
    input_data_path / "period1" / "network_topology" / "existing" / "distance.csv"
)

# Size
size = pd.read_csv(
    input_data_path / "period1" / "network_topology" / "existing" / "size.csv",
    sep=";",
    index_col=0,
)
size.loc["Parma", "Utrecht"] = 500
size.loc["Utrecht", "Parma"] = 500
size.loc["Utrecht", "Berlin"] = 300
size.loc["Berlin", "Utrecht"] = 300
size.to_csv(
    input_data_path
    / "period1"
    / "network_topology"
    / "existing"
    / "hydrogenPipelineOnshore_lowP"
    / "size.csv",
    sep=";",
)
print("Size:", size)

# Delete the template
os.remove(input_data_path / "period1" / "network_topology" / "existing" / "size.csv")


print("New network")
# Make a new folder for the new network
os.makedirs(
    input_data_path
    / "period1"
    / "network_topology"
    / "new"
    / "hydrogenPipelineOnshore_lowP",
    exist_ok=True,
)
os.makedirs(
    input_data_path / "period1" / "network_topology" / "new" / "electricitySimple",
    exist_ok=True,
)

# max size arc
arc_size = pd.read_csv(
    input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv",
    sep=";",
    index_col=0,
)
arc_size.loc["Parma", "Utrecht"] = 1000
arc_size.loc["Utrecht", "Parma"] = 1000
arc_size.loc["Utrecht", "Berlin"] = 1000
arc_size.loc["Berlin", "Utrecht"] = 1000
arc_size.loc["Berlin", "Parma"] = 1500
arc_size.loc["Parma", "Berlin"] = 1500
arc_size.to_csv(
    input_data_path
    / "period1"
    / "network_topology"
    / "new"
    / "hydrogenPipelineOnshore_lowP"
    / "size_max_arcs.csv",
    sep=";",
)
arc_size.to_csv(
    input_data_path
    / "period1"
    / "network_topology"
    / "new"
    / "electricitySimple"
    / "size_max_arcs.csv",
    sep=";",
)
print("Max size per arc:", arc_size)

# Use the templates, fill and save them to the respective directory
# Connection
connection = pd.read_csv(
    input_data_path / "period1" / "network_topology" / "new" / "connection.csv",
    sep=";",
    index_col=0,
)
connection.loc["Parma", "Utrecht"] = 1
connection.loc["Utrecht", "Parma"] = 1
connection.loc["Utrecht", "Berlin"] = 1
connection.loc["Berlin", "Utrecht"] = 1
connection.loc["Berlin", "Parma"] = 1
connection.loc["Parma", "Berlin"] = 1
connection.to_csv(
    input_data_path
    / "period1"
    / "network_topology"
    / "new"
    / "hydrogenPipelineOnshore_lowP"
    / "connection.csv",
    sep=";",
)
connection.to_csv(
    input_data_path
    / "period1"
    / "network_topology"
    / "new"
    / "electricitySimple"
    / "connection.csv",
    sep=";",
)
print("Connection:", connection)

# Delete the template
os.remove(input_data_path / "period1" / "network_topology" / "new" / "connection.csv")

# Distance
distance = pd.read_csv(
    input_data_path / "period1" / "network_topology" / "new" / "distance.csv",
    sep=";",
    index_col=0,
)
distance.loc["Parma", "Utrecht"] = 50
distance.loc["Utrecht", "Parma"] = 50
distance.loc["Utrecht", "Berlin"] = 35
distance.loc["Berlin", "Utrecht"] = 35
distance.loc["Berlin", "Parma"] = 25
distance.loc["Parma", "Berlin"] = 25
distance.to_csv(
    input_data_path
    / "period1"
    / "network_topology"
    / "new"
    / "hydrogenPipelineOnshore_lowP"
    / "distance.csv",
    sep=";",
)
distance.to_csv(
    input_data_path
    / "period1"
    / "network_topology"
    / "new"
    / "electricitySimple"
    / "distance.csv",
    sep=";",
)
print("Distance:", distance)

# Delete the template
os.remove(input_data_path / "period1" / "network_topology" / "new" / "distance.csv")

# Delete the max_size_arc template
os.remove(
    input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv"
)


adopt.copy_network_data(input_data_path)

with open(
    input_data_path / "period1" / "network_data" / "hydrogenPipelineOnshore_lowP.json",
    "r",
) as json_file:
    network_data = json.load(json_file)

network_data["Economics"]["gamma2"] = 40000
network_data["Economics"]["gamma4"] = 300

with open(
    input_data_path / "period1" / "network_data" / "hydrogenPipelineOnshore_lowP.json",
    "w",
) as json_file:
    json.dump(network_data, json_file, indent=4)


# Read hourly data from Excel (Examplary demand profiles are provided with the package)

Parma_hourly_data = pd.read_excel(
    input_data_path / "data/data_network_Parma.xlsx", header=0, nrows=8760
)
Utrecht_hourly_data = pd.read_excel(
    input_data_path / "data/data_network_Utrecht.xlsx", header=0, nrows=8760
)
Berlin_hourly_data = pd.read_excel(
    input_data_path / "data/data_network_Berlin.xlsx", header=0, nrows=8760
)

# Save the hourly data to the carrier's file in the case study folder
# electricity demand and price
hourly_data = {
    "Parma": Parma_hourly_data,
    "Utrecht": Utrecht_hourly_data,
    "Berlin": Berlin_hourly_data,
}

el_demand = {}
hydrogen_demand = {}

for node in ["Parma", "Utrecht", "Berlin"]:
    el_demand[node] = hourly_data[node].iloc[:, 1]
    hydrogen_demand[node] = hourly_data[node].iloc[:, 0] * 3
    adopt.fill_carrier_data(
        input_data_path,
        value_or_data=el_demand[node],
        columns=["Demand"],
        carriers=["electricity"],
        nodes=[node],
    )
    adopt.fill_carrier_data(
        input_data_path,
        value_or_data=hydrogen_demand[node],
        columns=["Demand"],
        carriers=["hydrogen"],
        nodes=[node],
    )

    # Set import limits/cost
    adopt.fill_carrier_data(
        input_data_path,
        value_or_data=400,
        columns=["Import limit"],
        carriers=["hydrogen"],
        nodes=[node],
    )
    adopt.fill_carrier_data(
        input_data_path,
        value_or_data=20,
        columns=["Import limit"],
        carriers=["electricity"],
        nodes=[node],
    )
    adopt.fill_carrier_data(
        input_data_path,
        value_or_data=0.25,
        columns=["Import emission factor"],
        carriers=["electricity"],
        nodes=[node],
    )
    adopt.fill_carrier_data(
        input_data_path,
        value_or_data=20,
        columns=["Import price"],
        carriers=["hydrogen"],
        nodes=[node],
    )
    adopt.fill_carrier_data(
        input_data_path,
        value_or_data=120,
        columns=["Import price"],
        carriers=["electricity"],
        nodes=[node],
    )

# Define climate data
adopt.load_climate_data_from_api(input_data_path)

m = adopt.ModelHub()
m.read_data(input_data_path)
m.quick_solve()
