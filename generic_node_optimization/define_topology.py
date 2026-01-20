import json
from pathlib import Path
import pandas as pd
import os
import math

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
    scenario_file = input_data_path/ "NodeLocations.csv"
    
    if not scenario_file.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_file}")
    
    # Load node coordinates
    node_coords = pd.read_csv(scenario_file, sep=';')
    print(f"📍 Calculating distances from scenario coordinates")
    
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

def define_nodes(input_data_path, params):
    # Add required technologies for SMALL cluster nodes
    small_nodes = ["SMALL1", "SMALL2", "SMALL3", "SMALL4"]
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
    big_nodes = ["BIG1", "BIG2"]
    big_new_tech_list = params["big_cluster_new_technologies"]
    big_existing_tech_list = params["big_cluster_existing_technologies"]

    
    for node in big_nodes:
        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "r") as json_file:
            technologies = json.load(json_file)

        technologies["new"] = big_new_tech_list
        technologies["existing"] = big_existing_tech_list

        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "w") as json_file:
            json.dump(technologies, json_file, indent=4)

    # Add required technologies for STORAGE node
    with open(input_data_path / "period1" / "node_data" / "STORAGE" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    storage_existing = params["existing_storage_technologies"]

    technologies["new"] = []
    technologies["existing"] = storage_existing

    with open(input_data_path / "period1" / "node_data" / "STORAGE" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

def add_existing_distribution_network(input_data_path, scenario_to_use):
    """
    Create existing low pressure distribution network (hydrogenPipelineOnshore_lowP)
    Defines pre-existing hydrogen pipeline infrastructure for distribution
    """
    print("\n📍 Creating Existing Distribution Network (Low Pressure)")

    # Create directory for existing distribution network
    os.makedirs(input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP",
                exist_ok=True)

    # Define node lists
    big_nodes = ["BIG1", "BIG2"]
    small_nodes = ["SMALL1", "SMALL2", "SMALL3", "SMALL4"]
    storage_node = ["STORAGE"]
    all_nodes = big_nodes + small_nodes + storage_node

    # Connection matrix
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                             index_col=0)

    # Define existing connections for low pressure distribution network
    # Connect small nodes to each other (local distribution)
    for i, node1 in enumerate(small_nodes):
        for j, node2 in enumerate(small_nodes):
            if i < j:  # Only upper triangle to avoid duplicates
                connection.loc[node1, node2] = 1
                connection.loc[node2, node1] = 1

    # Connect each small node to nearest big node
    connection.loc["SMALL1", "BIG1"] = 1
    connection.loc["BIG1", "SMALL1"] = 1
    connection.loc["SMALL2", "BIG1"] = 1
    connection.loc["BIG1", "SMALL2"] = 1
    connection.loc["SMALL3", "BIG2"] = 1
    connection.loc["BIG2", "SMALL3"] = 1
    connection.loc["SMALL4", "BIG2"] = 1
    connection.loc["BIG2", "SMALL4"] = 1

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP" / "connection.csv",
        sep=";")
    print(f"  ✓ Connection matrix saved")

    # Distance matrix
    try:
        distance = calculate_distances_from_coordinates(input_data_path)
        print("  ✓ Using calculated distances from topology coordinates")
    except FileNotFoundError as e:
        print(f"  ⚠️ {e}")
        print("  📍 Using template distances as fallback")
        distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "distance.csv", sep=";",
                               index_col=0)

    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP" / "distance.csv",
        sep=";")
    print(f"  ✓ Distance matrix saved")

    # Size matrix (existing pipeline sizes)
    size = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "size.csv", sep=";",
                       index_col=0)

    # Set existing sizes for connections (size 1 = standard existing pipeline)
    for i, node1 in enumerate(small_nodes):
        for j, node2 in enumerate(small_nodes):
            if i < j:
                size.loc[node1, node2] = 1
                size.loc[node2, node1] = 1

    # Small to big connections
    size.loc["SMALL1", "BIG1"] = 250
    size.loc["BIG1", "SMALL1"] = 250
    size.loc["SMALL2", "BIG1"] = 250
    size.loc["BIG1", "SMALL2"] = 250
    size.loc["SMALL3", "BIG2"] = 250
    size.loc["BIG2", "SMALL3"] = 250
    size.loc["SMALL4", "BIG2"] = 250
    size.loc["BIG2", "SMALL4"] = 250

    size.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP" / "size.csv",
        sep=";")
    print(f"  ✓ Size matrix saved")
    print(f"✅ Existing distribution network created successfully\n")


def add_existing_transmission_network(input_data_path, scenario_to_use):
    """
    Create existing high pressure transmission network (hydrogenPipelineOnshore_highP)
    Defines pre-existing hydrogen pipeline infrastructure for transmission
    """
    print("\n📍 Creating Existing Transmission Network (High Pressure)")

    # Create directory for existing transmission network
    os.makedirs(input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP",
                exist_ok=True)

    # Define node lists
    big_nodes = ["BIG1", "BIG2"]
    small_nodes = ["SMALL1", "SMALL2", "SMALL3", "SMALL4"]
    storage_node = ["STORAGE"]
    all_nodes = big_nodes + small_nodes + storage_node

    # Connection matrix
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                             index_col=0)

    # Define existing connections for high pressure transmission network
    # Connect big nodes to each other (backbone)
    connection.loc["BIG1", "BIG2"] = 1
    connection.loc["BIG2", "BIG1"] = 1

    # Connect storage to big nodes
    connection.loc["BIG1", "STORAGE"] = 1
    connection.loc["STORAGE", "BIG1"] = 1
    connection.loc["BIG2", "STORAGE"] = 1
    connection.loc["STORAGE", "BIG2"] = 1

    # # Connect some small nodes to big nodes for transmission access
    # connection.loc["SMALL1", "BIG1"] = 1
    # connection.loc["BIG1", "SMALL1"] = 1
    # connection.loc["SMALL3", "BIG2"] = 1
    # connection.loc["BIG2", "SMALL3"] = 1

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP" / "connection.csv",
        sep=";")
    print(f"  ✓ Connection matrix saved")

    # Distance matrix
    try:
        distance = calculate_distances_from_coordinates(input_data_path)
        print("  ✓ Using calculated distances from topology coordinates")
    except FileNotFoundError as e:
        print(f"  ⚠️ {e}")
        print("  📍 Using template distances as fallback")
        distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "distance.csv", sep=";",
                               index_col=0)

    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP" / "distance.csv",
        sep=";")
    print(f"  ✓ Distance matrix saved")

    # Size matrix (existing pipeline sizes)
    size = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "size.csv", sep=";",
                       index_col=0)

    # Set existing sizes for high pressure connections (typically larger)
    # Big to big backbone
    size.loc["BIG1", "BIG2"] = 500  # Size 2 = larger existing pipeline
    size.loc["BIG2", "BIG1"] = 500

    # Storage connections
    size.loc["BIG1", "STORAGE"] = 500
    size.loc["STORAGE", "BIG1"] = 500
    size.loc["BIG2", "STORAGE"] = 500
    size.loc["STORAGE", "BIG2"] = 500

    # # Small to big connections
    # size.loc["SMALL1", "BIG1"] = 500
    # size.loc["BIG1", "SMALL1"] = 500
    # size.loc["SMALL3", "BIG2"] = 500
    # size.loc["BIG2", "SMALL3"] = 500

    size.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP" / "size.csv",
        sep=";")
    print(f"  ✓ Size matrix saved")
    print(f"✅ Existing transmission network created successfully\n")


def add_new_distribution_network(input_data_path, scenario_to_use):
    """
    Create low pressure distribution network (hydrogenPipelineOnshore_lowP)
    Connects all nodes in a full mesh topology
    """
    print("\n📍 Creating Distribution Network (Low Pressure)")

    # Create directory for distribution network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP",
                exist_ok=True)

    # Define node lists
    big_nodes = ["BIG1", "BIG2"]
    small_nodes = ["SMALL1", "SMALL2", "SMALL3", "SMALL4"]
    storage_node = ["STORAGE"]
    all_nodes = big_nodes + small_nodes + storage_node

    # Load arc size template
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)

    # Set arc sizes for all node combinations
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:  # Don't connect node to itself
                if node1 in small_nodes and node2 in small_nodes:
                    arc_size.loc[node1, node2] = 1000  # Small to small
                elif (node1 in big_nodes and node2 in small_nodes) or (node1 in small_nodes and node2 in big_nodes):
                    arc_size.loc[node1, node2] = 1000  # Big to small or small to big
                elif node1 in big_nodes and node2 in big_nodes:
                    arc_size.loc[node1, node2] = 1000  # Big to big
                elif node1 == "STORAGE" or node2 == "STORAGE":
                    arc_size.loc[node1, node2] = 1000  # Storage connections
                else:
                    arc_size.loc[node1, node2] = 1000  # Default

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

    # Calculate and save distances
    try:
        distance = calculate_distances_from_coordinates(input_data_path)
        print("  ✓ Using calculated distances from topology coordinates")
    except FileNotFoundError as e:
        print(f"  ⚠️ {e}")
        print("  📍 Using template distances as fallback")
        distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "distance.csv", sep=";",
                               index_col=0)

    # Save distance matrix
    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "distance.csv",
        sep=";")
    print(f"  ✓ Distance matrix saved")
    print(f"✅ Distribution network created successfully\n")


def add_new_transmission_network(input_data_path, scenario_to_use):
    """
    Create high pressure transmission network (hydrogenPipelineOnshore_highP)
    Connects all nodes in a full mesh topology
    """
    print("\n📍 Creating Transmission Network (High Pressure)")

    # Create directory for transmission network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP",
                exist_ok=True)

    # Define node lists
    big_nodes = ["BIG1", "BIG2"]
    small_nodes = ["SMALL1", "SMALL2", "SMALL3", "SMALL4"]
    storage_node = ["STORAGE"]
    all_nodes = big_nodes + small_nodes + storage_node

    # Load arc size template
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)

    # Set arc sizes for all node combinations
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:  # Don't connect node to itself
                if node1 in small_nodes and node2 in small_nodes:
                    arc_size.loc[node1, node2] = 1000  # Small to small
                elif (node1 in big_nodes and node2 in small_nodes) or (node1 in small_nodes and node2 in big_nodes):
                    arc_size.loc[node1, node2] = 1000  # Big to small or small to big
                elif node1 in big_nodes and node2 in big_nodes:
                    arc_size.loc[node1, node2] = 1000  # Big to big
                elif node1 == "STORAGE" or node2 == "STORAGE":
                    arc_size.loc[node1, node2] = 1000  # Storage connections
                else:
                    arc_size.loc[node1, node2] = 1000  # Default

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

    # Calculate and save distances
    try:
        distance = calculate_distances_from_coordinates(input_data_path)
        print("  ✓ Using calculated distances from topology coordinates")
    except FileNotFoundError as e:
        print(f"  ⚠️ {e}")
        print("  📍 Using template distances as fallback")
        distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "distance.csv", sep=";",
                               index_col=0)

    # Save distance matrix
    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "distance.csv",
        sep=";")
    print(f"  ✓ Distance matrix saved")
    print(f"✅ Transmission network created successfully\n")


def add_new_network_H2(input_data_path):
    # Make a new folder for the new network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP",
                exist_ok=True)

    print("New network")

    # Define new node lists
    big_nodes = ["BIG1", "BIG2"]
    small_nodes = ["SMALL1", "SMALL2", "SMALL3", "SMALL4"]
    storage_node = ["STORAGE"]
    all_nodes = big_nodes + small_nodes + storage_node

    # max size arc
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)

    # Connect all possible combinations of nodes
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:  # Don't connect node to itself
                # Set arc size based on node types
                if node1 in small_nodes and node2 in small_nodes:
                    arc_size.loc[node1, node2] = 1000  # Small to small
                elif (node1 in big_nodes and node2 in small_nodes) or (node1 in small_nodes and node2 in big_nodes):
                    arc_size.loc[node1, node2] = 1000  # Big to small or small to big
                elif node1 in big_nodes and node2 in big_nodes:
                    arc_size.loc[node1, node2] = 1000  # Big to big (EHB style)
                elif node1 == "STORAGE" or node2 == "STORAGE":
                    arc_size.loc[node1, node2] = 1000  # Storage connections
                else:
                    arc_size.loc[node1, node2] = 1000  # Default

    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "size_max_arcs.csv",
        sep=";")

    # Create high pressure network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP",
                exist_ok=True)

    # For high pressure, create connections between big nodes and storage
    # Reset arc_size for high pressure network
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)

    # Connect all small nodes to each other and to the backbone
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:  # Don't connect node to itself
                # Set arc size based on node types
                if node1 in small_nodes and node2 in small_nodes:
                    arc_size.loc[node1, node2] = 1000  # Small to small
                elif (node1 in big_nodes and node2 in small_nodes) or (node1 in small_nodes and node2 in big_nodes):
                    arc_size.loc[node1, node2] = 1000  # Big to small or small to big
                elif node1 in big_nodes and node2 in big_nodes:
                    arc_size.loc[node1, node2] = 1000  # Big to big
                elif node1 == "STORAGE" or node2 == "STORAGE":
                    arc_size.loc[node1, node2] = 1000  # Storage connections
                else:
                    arc_size.loc[node1, node2] = 1000  # Default

    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "size_max_arcs.csv",
        sep=";")

    print("Max size per arc:", arc_size)

    # Delete the max_size_arc template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv")

    # Connection matrices
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)

    # Connect all nodes to all other nodes (full mesh for low pressure)
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:  # Don't connect node to itself
                connection.loc[node1, node2] = 1

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "connection.csv",
        sep=";")

    # High pressure connections (
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)

    # Connect all nodes to all other nodes (full mesh for high pressure)
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:  # Don't connect node to itself
                connection.loc[node1, node2] = 1

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "connection.csv",
        sep=";")

    print("Connection:", connection)

    # Delete the connection template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "connection.csv")

    # Calculate real distances from topology coordinates
    try:
        distance = calculate_distances_from_coordinates(input_data_path)
        print("✅ Using calculated distances from topology coordinates")
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        print("📍 Using template distances as fallback")
        # Load template distance matrix as fallback
        distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "distance.csv", sep=";",
                               index_col=0)

    # Save distance matrices for both networks
    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "distance.csv",
        sep=";")
    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "distance.csv",
        sep=";")

    print("Distance matrix saved for both networks")

    # Delete the distance template
    try:
        os.remove(input_data_path / "period1" / "network_topology" / "new" / "distance.csv")
    except FileNotFoundError:
        pass  # Template might not exist
