import math
import pandas as pd
from pathlib import Path
import os
import json

def calculate_distance_between_coordinates(lon1, lat1, lon2, lat2):
    """
    Calculate the distance between two points on Earth using the Haversine formula.
    Returns distance in kilometers.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in kilometers
    r = 6371

    return c * r


def calculate_distances_from_coordinates(input_data_path):
    """
    Calculate distances between all nodes based on their actual coordinates
    from the generated topology scenario.
    """
    # Read coordinates from the generated scenario file
    scenario_file = input_data_path / "NodeLocations.csv"

    if not scenario_file.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_file}")

    # Load node coordinates
    node_coords = pd.read_csv(scenario_file, sep=';')
    print(f" Calculating distances from scenario coordinates")

    # Create coordinate dictionary
    coords = {}
    for _, row in node_coords.iterrows():
        coords[row['index']] = (row['lon'], row['lat'])

    # Get all nodes
    all_nodes = list(coords.keys())

    # Create empty distance matrix
    distance_matrix = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)

    # Calculate distances between all pairs of nodes
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:  # Don't calculate distance from node to itself
                lon1, lat1 = coords[node1]
                lon2, lat2 = coords[node2]

                # Calculate distance using Haversine formula
                distance = calculate_distance_between_coordinates(lon1, lat1, lon2, lat2)
                distance_matrix.loc[node1, node2] = round(distance, 1)  # Round to 1 decimal place

    return distance_matrix

def load_scenario_nodes(input_data_path, nodes, scenario):
    """Load coordinates for node from scenarios"""
    # Build path relative to this file's location
    current_file = Path(__file__).resolve()
    four_node_folder = current_file.parent  # four_node_configuration folder
    scenario_file = four_node_folder / "preprocess" / "generated_topology_old" / f"NodeLocations_{scenario}.csv"

    if not scenario_file.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_file}")

    scenario_nodes = pd.read_csv(scenario_file, sep=';')
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

def define_nodes(input_data_path, params):
    # Add required technologies for SMALL cluster nodes
    small_nodes = ["Small_cluster1", "Small_cluster2"]
    small_new_tech_list = params["small_cluster_new_technologies"]
    small_existing_tech_list = params["small_cluster_existing_technologies"]

    for node in small_nodes:
        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "r") as json_file:
            technologies = json.load(json_file)

        technologies["new"] = small_new_tech_list
        technologies["existing"] = small_existing_tech_list

        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "w") as json_file:
            json.dump(technologies, json_file, indent=4)

    # Add required technologies for BIG cluster nodes
    big_nodes = ["Large_cluster"]
    big_new_tech_list = params["big_cluster_new_technologies"]
    big_existing_tech_list = params["big_cluster_existing_technologies"]

    for node in big_nodes:
        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "r") as json_file:
            technologies = json.load(json_file)

        technologies["new"] = big_new_tech_list
        technologies["existing"] = big_existing_tech_list

        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "w") as json_file:
            json.dump(technologies, json_file, indent=4)

def add_existing_distribution_network(input_data_path):
    """
    Create existing low pressure distribution network (hydrogenPipelineOnshore_lowP)
    Defines pre-existing hydrogen pipeline infrastructure for distribution
    """
    print("\n Creating Existing Distribution Network (Low Pressure)")

    # Create directory for existing distribution network
    os.makedirs(input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP",
                exist_ok=True)

    # Define node lists
    big_nodes = ["Large_cluster"]
    small_nodes = ["Small_cluster1", "Small_cluster2"]
    all_nodes = big_nodes + small_nodes

    # Connection matrix
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                             index_col=0)

    # Connect small nodes to each other (local distribution)
    for i, node1 in enumerate(small_nodes):
        for j, node2 in enumerate(small_nodes):
            if i != j:  # Avoid self-connection
                connection.loc[node1, node2] = 1
                connection.loc[node2, node1] = 1

    # Connect small nodes to large nodes (bidirectional)
    for small in small_nodes:
        for large in big_nodes:
            connection.loc[small, large] = 1
            connection.loc[large, small] = 1

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP" / "connection.csv",
        sep=";")
    print(f"  ✓ Connection matrix saved")

    # Distance matrix
    distance = calculate_distances_from_coordinates(input_data_path)
    print("  ✓ Using calculated distances from topology coordinates")

    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP" / "distance.csv",
        sep=";")
    print(f"  ✓ Distance matrix saved")

    # Size matrix (existing pipeline sizes)
    size = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "size.csv", sep=";",
                       index_col=0)

    # Set existing sizes for connections
    for i, node1 in enumerate(small_nodes):
        for j, node2 in enumerate(small_nodes):
            if i != j:  # Avoid self-connection
                size.loc[node1, node2] = 250
                size.loc[node2, node1] = 250


    for small in small_nodes:
        for large in big_nodes:
            size.loc[small, large] = 250
            size.loc[large, small] = 250

    size.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP" / "size.csv",
        sep=";")
    print(f"  ✓ Size matrix saved")
    print(f"✅ Existing distribution network created successfully\n")


def add_existing_transmission_network(input_data_path):
    """
    Create existing high pressure transmission network (hydrogenPipelineOnshore_highP)
    Defines pre-existing hydrogen pipeline infrastructure for transmission
    """
    print("\n Creating Existing Transmission Network (High Pressure)")

    # Create directory for existing transmission network
    os.makedirs(input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP",
                exist_ok=True)

    # Define node lists
    big_nodes = ["Large_cluster"]
    small_nodes = ["Small_cluster1", "Small_cluster2"]
    all_nodes = big_nodes + small_nodes

    # Connection matrix
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                             index_col=0)

    # Define existing connections for high pressure transmission network
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:
                connection.loc[node1, node2] = 1

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP" / "connection.csv",
        sep=";")
    print(f"  ✓ Connection matrix saved")

    # Distance matrix
    distance = calculate_distances_from_coordinates(input_data_path)
    print("  ✓ Using calculated distances from topology coordinates")

    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP" / "distance.csv",
        sep=";")
    print(f"  ✓ Distance matrix saved")

    # Size matrix (existing pipeline sizes)
    size = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "size.csv", sep=";",
                       index_col=0)

    # Set existing sizes for high pressure connections (typically larger)
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:
                size.loc[node1, node2] = 1000

    size.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP" / "size.csv",
        sep=";")
    print(f"  ✓ Size matrix saved")
    print(f"✅ Existing transmission network created successfully\n")


def add_new_distribution_network(input_data_path):
    """
    Create low pressure distribution network (hydrogenPipelineOnshore_lowP)
    Connects all nodes in a full mesh topology
    """
    print("\n Creating Distribution Network (Low Pressure)")

    # Create directory for distribution network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP",
                exist_ok=True)

    # Define node lists
    big_nodes = ["Large_cluster"]
    small_nodes = ["Small_cluster1", "Small_cluster2"]
    all_nodes = big_nodes + small_nodes

    # Load arc size template
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)

    # Same size for nodes to all other nodes
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:
                arc_size.loc[node1, node2] = 1000

    # Save arc size matrix
    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "size_max_arcs.csv",
        sep=";")
    print(f"  ✓ Arc sizes saved")

    # Load connection template
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)

    # Connect all nodes to all other nodes (full mesh)
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:
                connection.loc[node1, node2] = 1

    # Save connection matrix
    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "connection.csv",
        sep=";")
    print(f"  ✓ Connection matrix saved (full mesh)")


    distance = calculate_distances_from_coordinates(input_data_path)
    print("  ✓ Using calculated distances from topology coordinates")

    # Save distance matrix
    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "distance.csv",
        sep=";")
    print(f"  ✓ Distance matrix saved")
    print(f"Distribution network created successfully\n")


def add_new_transmission_network(input_data_path):
    """
    Create high pressure transmission network (hydrogenPipelineOnshore_highP)
    Connects all nodes in a full mesh topology
    """
    print("\n📍 Creating Transmission Network (High Pressure)")

    # Create directory for transmission network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP",
                exist_ok=True)

    # Define node lists
    big_nodes = ["Large_cluster"]
    small_nodes = ["Small_cluster1", "Small_cluster2"]
    all_nodes = big_nodes + small_nodes

    # Load arc size template
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)

    # Same size for nodes to all other nodes
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:
                arc_size.loc[node1, node2] = 1000

    # Save arc size matrix
    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "size_max_arcs.csv",
        sep=";")
    print(f"  ✓ Arc sizes saved")

    # Load connection template
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)

    # Connect all nodes to all other nodes (full mesh)
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:
                connection.loc[node1, node2] = 1

    # Save connection matrix
    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "connection.csv",
        sep=";")
    print(f"  ✓ Connection matrix saved (full mesh)")

    distance = calculate_distances_from_coordinates(input_data_path)
    print("  ✓ Using calculated distances from topology coordinates")

    # Save distance matrix
    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "distance.csv",
        sep=";")
    print(f"  ✓ Distance matrix saved")
    print(f"✅ Transmission network created successfully\n")

