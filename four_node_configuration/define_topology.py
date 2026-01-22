import json
from pathlib import Path
import pandas as pd
import os
import math

def set_distance_between_nodes(input_data_path, params):
    """
    Set the following value in the input data:

    - Distance to shore for large and small cluster
    """

    # Process new networks
    if "networks_new" in params and params["networks_new"]:
        for network in params["networks_new"]:

            distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "distance.csv",
                                   sep=";",
                                   index_col=0)
            # Convert DataFrame to float BEFORE assignment
            distance = distance.astype(float)

            distance.loc["Large_cluster", "Small_cluster"] = distance_value
            distance.loc["Small_cluster", "Large_cluster"] = distance_value

            distance.to_csv(
                input_data_path / "period1" / "network_topology" / "new" /
                network / "distance.csv",
                sep=";")

            print(f"  ✓ Updated distances for new network: {network}")

    # Process existing networks
    if "networks_existing" in params and params["networks_existing"]:
        for network in params["networks_existing"]:

            distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "distance.csv",
                                   sep=";",
                                   index_col=0)

            # Convert DataFrame to float BEFORE assignment
            distance = distance.astype(float)

            distance.loc["Large_cluster", "Small_cluster"] = distance_value
            distance.loc["Small_cluster", "Large_cluster"] = distance_value

            distance.to_csv(
                input_data_path / "period1" / "network_topology" / "existing" /
                network / "distance.csv",
                sep=";")

            print(f"  ✓ Updated distances for existing network: {network}")


def add_existing_distribution_network(input_data_path):
    """
    Create existing low pressure distribution network (hydrogenPipelineOnshore_lowP)
    Defines pre-existing hydrogen pipeline infrastructure for distribution
    """
    print("Creating Existing Distribution Network (Low Pressure)")

    # Create directory for existing distribution network
    os.makedirs(input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP",
                exist_ok=True)

    # Connection matrix
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                             index_col=0)

    # Connect each small node to nearest big node
    connection.loc["Large_cluster", "Small_cluster"] = 1
    connection.loc["Small_cluster", "Large_cluster"] = 1

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP" / "connection.csv",
        sep=";")

    # Size matrix (existing pipeline sizes)
    size = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "size.csv", sep=";",
                       index_col=0)

    size.loc["Large_cluster", "Small_cluster"] = 100
    size.loc["Small_cluster", "Large_cluster"] = 100

    size.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP" / "size.csv",
        sep=";")

    print(f"Existing distribution network created successfully\n")

def add_existing_transmission_network(input_data_path):
    """
    Create existing high pressure transmission network (hydrogenPipelineOnshore_highP)
    Defines pre-existing hydrogen pipeline infrastructure for transmission
    """
    print("Creating Existing Transmission Network (High Pressure)")

    # Create directory for existing transmission network
    os.makedirs(input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP",
                exist_ok=True)

    # Connection matrix
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                             index_col=0)

    # Define existing connections for high pressure transmission network
    # Connect big nodes to each other (backbone)
    connection.loc["Large_cluster", "Small_cluster"] = 1
    connection.loc["Small_cluster", "Large_cluster"] = 1

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP" / "connection.csv",
        sep=";")

    # Size matrix (existing pipeline sizes)
    size = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "size.csv", sep=";",
                       index_col=0)

    size.loc["Large_cluster", "Small_cluster"] = 300
    size.loc["Small_cluster", "Large_cluster"] = 300

    size.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP" / "size.csv",
        sep=";")

    print(f"Existing transmission network created successfully\n")

def add_new_distribution_network(input_data_path):
    """
    Create low pressure distribution network (hydrogenPipelineOnshore_lowP)
    Connects all nodes in a full mesh topology
    """
    print("Creating Distribution Network (Low Pressure)")

    # Create directory for distribution network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP",
                exist_ok=True)

    # Load arc size template
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)

    arc_size.loc["Large_cluster", "Small_cluster"] = 500
    arc_size.loc["Small_cluster", "Large_cluster"] = 500


    # Save arc size matrix
    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "size_max_arcs.csv",
        sep=";")

    # Load connection template
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)

    connection.loc["Large_cluster", "Small_cluster"] = 1
    connection.loc["Small_cluster", "Large_cluster"] = 1

    # Save connection matrix
    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "connection.csv",
        sep=";")

    print(f"Distribution network created successfully\n")

def add_new_transmission_network(input_data_path):
    """
    Create high pressure transmission network (hydrogenPipelineOnshore_highP)
    Connects all nodes in a full mesh topology
    """
    print("Creating Transmission Network (High Pressure)")

    # Create directory for transmission network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP",
                exist_ok=True)

    # Load arc size template
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)

    arc_size.loc["Large_cluster", "Small_cluster"] = 1000
    arc_size.loc["Small_cluster", "Large_cluster"] = 1000

    # Save arc size matrix
    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "size_max_arcs.csv",
        sep=";")

    # Load connection template
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)

    connection.loc["Large_cluster", "Small_cluster"] = 1
    connection.loc["Small_cluster", "Large_cluster"] = 1

    # Save connection matrix
    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "connection.csv",
        sep=";")

    print(f"Transmission network created successfully\n")

