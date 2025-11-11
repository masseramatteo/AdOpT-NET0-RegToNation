import json
from pathlib import Path
import os
import pandas as pd
import numpy as np


def define_nodes(input_data_path):
    # Add required technologies for node 'Roermond' small cluster
    with open(input_data_path / "period1" / "node_data" / "Roermond" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = ["Electrolyzer", "Storage_H2"]
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "Roermond" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'Dongen' small cluster
    with open(input_data_path / "period1" / "node_data" / "Dongen" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = ["Electrolyzer", "Storage_H2"]
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "Dongen" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'Oosterhout' small cluster
    with open(input_data_path / "period1" / "node_data" / "Oosterhout" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = ["Electrolyzer", "Storage_H2"]
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "Oosterhout" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'Maastricht' small cluster
    with open(input_data_path / "period1" / "node_data" / "Maastricht" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = ["Electrolyzer", "Storage_H2"]
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "Maastricht" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'Venlo' small cluster
    with open(input_data_path / "period1" / "node_data" / "Venlo" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = ["Electrolyzer", "Storage_H2"]
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "Venlo" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'Arnhem' small cluster
    with open(input_data_path / "period1" / "node_data" / "Arnhem" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = ["Electrolyzer", "Storage_H2"]
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "Arnhem" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'East_Groningen' small cluster
    with open(input_data_path / "period1" / "node_data" / "East_Groningen" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = ["Electrolyzer", "Storage_H2"]
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "East_Groningen" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'Betuwe' small cluster
    with open(input_data_path / "period1" / "node_data" / "Betuwe" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = ["Electrolyzer", "Storage_H2"]
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "Betuwe" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'Heerlen' small cluster
    with open(input_data_path / "period1" / "node_data" / "Heerlen" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = ["Electrolyzer", "Storage_H2"]
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "Heerlen" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'Dordrecht' small cluster
    with open(input_data_path / "period1" / "node_data" / "Dordrecht" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = ["Electrolyzer", "Storage_H2"]
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "Dordrecht" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'Wageningen' small cluster
    with open(input_data_path / "period1" / "node_data" / "Wageningen" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = ["Electrolyzer", "Storage_H2"]
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "Wageningen" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'Rotterdam' big cluster
    with open(input_data_path / "period1" / "node_data" / "Rotterdam" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = []
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "Rotterdam" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'Zeeland' big cluster
    with open(input_data_path / "period1" / "node_data" / "Zeeland" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = []
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "Zeeland" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'North_Sea' big cluster
    with open(input_data_path / "period1" / "node_data" / "North_Sea" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = []
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "North_Sea" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'North_Netherlands' big cluster
    with open(input_data_path / "period1" / "node_data" / "North_Netherlands" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = []
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "North_Netherlands" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'Chemelot' big cluster
    with open(input_data_path / "period1" / "node_data" / "Chemelot" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = []
    technologies["existing"] = {}

    with open(input_data_path / "period1" / "node_data" / "Chemelot" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'Zuidwending' storage
    with open(input_data_path / "period1" / "node_data" / "Zuidwending" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)

    technologies["new"] = []
    technologies["existing"] = {"Storage_H2_Cavern": 100000}

    with open(input_data_path / "period1" / "node_data" / "Zuidwending" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

def add_existing_network_H2(input_data_path):
    # Make a new folder for the existing network
    os.makedirs(input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP_big",
                exist_ok=True)
    print("Existing network")

    # Use the templates, fill and save them to the respective directory
    # Connection

    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                             index_col=0)
    connection.loc["Rotterdam", "Zeeland"] = 1
    connection.loc["Zeeland", "Rotterdam"] = 1
    connection.loc["Rotterdam", "North_Sea"] = 1
    connection.loc["North_Sea", "Rotterdam"] = 1
    connection.loc["North_Sea", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "North_Sea"] = 1
    connection.loc["Zuidwending", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Chemelot"] = 1
    connection.loc["Chemelot", "Zuidwending"] = 1
    connection.loc["Zeeland", "Chemelot"] = 1
    connection.loc["Chemelot", "Zeeland"] = 1
    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP_big" / "connection.csv",
        sep=";")
    print("Connection:", connection)

    # Delete the template
    os.remove(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv")

    # Distance
    distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "distance.csv", sep=";",
                           index_col=0)
    distance.loc["Rotterdam", "Zeeland"] = 64
    distance.loc["Zeeland", "Rotterdam"] = 64
    distance.loc["Rotterdam", "North_Sea"] = 61
    distance.loc["North_Sea", "Rotterdam"] = 61
    distance.loc["North_Sea", "North_Netherlands"] = 176
    distance.loc["North_Netherlands", "North_Sea"] = 176
    distance.loc["Zuidwending", "North_Netherlands"] = 40
    distance.loc["North_Netherlands", "Zuidwending"] = 40
    distance.loc["Zuidwending", "Chemelot"] = 248
    distance.loc["Chemelot", "Zuidwending"] = 248
    distance.loc["Zeeland", "Chemelot"] = 148
    distance.loc["Chemelot", "Zeeland"] = 148
    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP_big" / "distance.csv",
        sep=";")
    print("Distance:", distance)

    # Delete the template
    os.remove(input_data_path / "period1" / "network_topology" / "existing" / "distance.csv")

    # Size
    size = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "size.csv", sep=";", index_col=0)
    size.loc["Rotterdam", "Zeeland"] = 1
    size.loc["Zeeland", "Rotterdam"] = 1
    size.loc["Rotterdam", "North_Sea"] = 1
    size.loc["North_Sea", "Rotterdam"] = 1
    size.loc["North_Sea", "North_Netherlands"] = 1
    size.loc["North_Netherlands", "North_Sea"] = 1
    size.loc["Zuidwending", "North_Netherlands"] = 1
    size.loc["North_Netherlands", "Zuidwending"] = 1
    size.loc["Zuidwending", "Chemelot"] = 1
    size.loc["Chemelot", "Zuidwending"] = 1
    size.loc["Zeeland", "Chemelot"] = 1
    size.loc["Chemelot", "Zeeland"] = 1
    size.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP_big" / "size.csv",
        sep=";")
    print("Size:", size)

    # Delete the template
    os.remove(input_data_path / "period1" / "network_topology" / "existing" / "size.csv")

def add_new_network_H2_all(input_data_path):
    # # Make a new folder for the new network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP",
                exist_ok=True)
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP",
                exist_ok=True)
    #os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP", exist_ok=True)

    print("New network")

    # # max size arc
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)
    # <editor-fold desc="Arc size connections pipeline">
    arc_size.loc["Roermond", "Dongen"] = 3
    arc_size.loc["Dongen", "Roermond"] = 3

    arc_size.loc["Roermond", "Oosterhout"] = 3
    arc_size.loc["Oosterhout", "Roermond"] = 3

    arc_size.loc["Roermond", "Maastricht"] = 3
    arc_size.loc["Maastricht", "Roermond"] = 3

    arc_size.loc["Roermond", "Venlo"] = 3
    arc_size.loc["Venlo", "Roermond"] = 3

    arc_size.loc["Roermond", "Arnhem"] = 3
    arc_size.loc["Arnhem", "Roermond"] = 3

    arc_size.loc["Roermond", "East_Groningen"] = 3
    arc_size.loc["East_Groningen", "Roermond"] = 3

    arc_size.loc["Roermond", "Betuwe"] = 3
    arc_size.loc["Betuwe", "Roermond"] = 3

    arc_size.loc["Roermond", "Heerlen"] = 3
    arc_size.loc["Heerlen", "Roermond"] = 3

    arc_size.loc["Roermond", "Dordrecht"] = 3
    arc_size.loc["Dordrecht", "Roermond"] = 3

    arc_size.loc["Roermond", "Wageningen"] = 3
    arc_size.loc["Wageningen", "Roermond"] = 3

    arc_size.loc["Roermond", "Rotterdam"] = 3
    arc_size.loc["Rotterdam", "Roermond"] = 3

    arc_size.loc["Roermond", "Zeeland"] = 3
    arc_size.loc["Zeeland", "Roermond"] = 3

    arc_size.loc["Roermond", "North_Sea"] = 3
    arc_size.loc["North_Sea", "Roermond"] = 3

    arc_size.loc["Roermond", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "Roermond"] = 3

    arc_size.loc["Roermond", "Chemelot"] = 3
    arc_size.loc["Chemelot", "Roermond"] = 3

    arc_size.loc["Roermond", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "Roermond"] = 3

    arc_size.loc["Dongen", "Oosterhout"] = 3
    arc_size.loc["Oosterhout", "Dongen"] = 3

    arc_size.loc["Dongen", "Maastricht"] = 3
    arc_size.loc["Maastricht", "Dongen"] = 3

    arc_size.loc["Dongen", "Venlo"] = 3
    arc_size.loc["Venlo", "Dongen"] = 3

    arc_size.loc["Dongen", "Arnhem"] = 3
    arc_size.loc["Arnhem", "Dongen"] = 3

    arc_size.loc["Dongen", "East_Groningen"] = 3
    arc_size.loc["East_Groningen", "Dongen"] = 3

    arc_size.loc["Dongen", "Betuwe"] = 3
    arc_size.loc["Betuwe", "Dongen"] = 3

    arc_size.loc["Dongen", "Heerlen"] = 3
    arc_size.loc["Heerlen", "Dongen"] = 3

    arc_size.loc["Dongen", "Dordrecht"] = 3
    arc_size.loc["Dordrecht", "Dongen"] = 3

    arc_size.loc["Dongen", "Wageningen"] = 3
    arc_size.loc["Wageningen", "Dongen"] = 3

    arc_size.loc["Dongen", "Rotterdam"] = 3
    arc_size.loc["Rotterdam", "Dongen"] = 3

    arc_size.loc["Dongen", "Zeeland"] = 3
    arc_size.loc["Zeeland", "Dongen"] = 3

    arc_size.loc["Dongen", "North_Sea"] = 3
    arc_size.loc["North_Sea", "Dongen"] = 3

    arc_size.loc["Dongen", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "Dongen"] = 3

    arc_size.loc["Dongen", "Chemelot"] = 3
    arc_size.loc["Chemelot", "Dongen"] = 3

    arc_size.loc["Dongen", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "Dongen"] = 3

    arc_size.loc["Oosterhout", "Maastricht"] = 3
    arc_size.loc["Maastricht", "Oosterhout"] = 3

    arc_size.loc["Oosterhout", "Venlo"] = 3
    arc_size.loc["Venlo", "Oosterhout"] = 3

    arc_size.loc["Oosterhout", "Arnhem"] = 3
    arc_size.loc["Arnhem", "Oosterhout"] = 3

    arc_size.loc["Oosterhout", "East_Groningen"] = 3
    arc_size.loc["East_Groningen", "Oosterhout"] = 3

    arc_size.loc["Oosterhout", "Betuwe"] = 3
    arc_size.loc["Betuwe", "Oosterhout"] = 3

    arc_size.loc["Oosterhout", "Heerlen"] = 3
    arc_size.loc["Heerlen", "Oosterhout"] = 3

    arc_size.loc["Oosterhout", "Dordrecht"] = 3
    arc_size.loc["Dordrecht", "Oosterhout"] = 3

    arc_size.loc["Oosterhout", "Wageningen"] = 3
    arc_size.loc["Wageningen", "Oosterhout"] = 3

    arc_size.loc["Oosterhout", "Rotterdam"] = 3
    arc_size.loc["Rotterdam", "Oosterhout"] = 3

    arc_size.loc["Oosterhout", "Zeeland"] = 3
    arc_size.loc["Zeeland", "Oosterhout"] = 3

    arc_size.loc["Oosterhout", "North_Sea"] = 3
    arc_size.loc["North_Sea", "Oosterhout"] = 3

    arc_size.loc["Oosterhout", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "Oosterhout"] = 3

    arc_size.loc["Oosterhout", "Chemelot"] = 3
    arc_size.loc["Chemelot", "Oosterhout"] = 3

    arc_size.loc["Oosterhout", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "Oosterhout"] = 3

    arc_size.loc["Maastricht", "Venlo"] = 3
    arc_size.loc["Venlo", "Maastricht"] = 3

    arc_size.loc["Maastricht", "Arnhem"] = 3
    arc_size.loc["Arnhem", "Maastricht"] = 3

    arc_size.loc["Maastricht", "East_Groningen"] = 3
    arc_size.loc["East_Groningen", "Maastricht"] = 3

    arc_size.loc["Maastricht", "Betuwe"] = 3
    arc_size.loc["Betuwe", "Maastricht"] = 3

    arc_size.loc["Maastricht", "Heerlen"] = 3
    arc_size.loc["Heerlen", "Maastricht"] = 3

    arc_size.loc["Maastricht", "Dordrecht"] = 3
    arc_size.loc["Dordrecht", "Maastricht"] = 3

    arc_size.loc["Maastricht", "Wageningen"] = 3
    arc_size.loc["Wageningen", "Maastricht"] = 3

    arc_size.loc["Maastricht", "Rotterdam"] = 3
    arc_size.loc["Rotterdam", "Maastricht"] = 3

    arc_size.loc["Maastricht", "Zeeland"] = 3
    arc_size.loc["Zeeland", "Maastricht"] = 3

    arc_size.loc["Maastricht", "North_Sea"] = 3
    arc_size.loc["North_Sea", "Maastricht"] = 3

    arc_size.loc["Maastricht", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "Maastricht"] = 3

    arc_size.loc["Maastricht", "Chemelot"] = 3
    arc_size.loc["Chemelot", "Maastricht"] = 3

    arc_size.loc["Maastricht", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "Maastricht"] = 3

    arc_size.loc["Venlo", "Arnhem"] = 3
    arc_size.loc["Arnhem", "Venlo"] = 3

    arc_size.loc["Venlo", "East_Groningen"] = 3
    arc_size.loc["East_Groningen", "Venlo"] = 3

    arc_size.loc["Venlo", "Betuwe"] = 3
    arc_size.loc["Betuwe", "Venlo"] = 3

    arc_size.loc["Venlo", "Heerlen"] = 3
    arc_size.loc["Heerlen", "Venlo"] = 3

    arc_size.loc["Venlo", "Dordrecht"] = 3
    arc_size.loc["Dordrecht", "Venlo"] = 3

    arc_size.loc["Venlo", "Wageningen"] = 3
    arc_size.loc["Wageningen", "Venlo"] = 3

    arc_size.loc["Venlo", "Rotterdam"] = 3
    arc_size.loc["Rotterdam", "Venlo"] = 3

    arc_size.loc["Venlo", "Zeeland"] = 3
    arc_size.loc["Zeeland", "Venlo"] = 3

    arc_size.loc["Venlo", "North_Sea"] = 3
    arc_size.loc["North_Sea", "Venlo"] = 3

    arc_size.loc["Venlo", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "Venlo"] = 3

    arc_size.loc["Venlo", "Chemelot"] = 3
    arc_size.loc["Chemelot", "Venlo"] = 3

    arc_size.loc["Venlo", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "Venlo"] = 3

    arc_size.loc["Arnhem", "East_Groningen"] = 3
    arc_size.loc["East_Groningen", "Arnhem"] = 3

    arc_size.loc["Arnhem", "Betuwe"] = 3
    arc_size.loc["Betuwe", "Arnhem"] = 3

    arc_size.loc["Arnhem", "Heerlen"] = 3
    arc_size.loc["Heerlen", "Arnhem"] = 3

    arc_size.loc["Arnhem", "Dordrecht"] = 3
    arc_size.loc["Dordrecht", "Arnhem"] = 3

    arc_size.loc["Arnhem", "Wageningen"] = 3
    arc_size.loc["Wageningen", "Arnhem"] = 3

    arc_size.loc["Arnhem", "Rotterdam"] = 3
    arc_size.loc["Rotterdam", "Arnhem"] = 3

    arc_size.loc["Arnhem", "Zeeland"] = 3
    arc_size.loc["Zeeland", "Arnhem"] = 3

    arc_size.loc["Arnhem", "North_Sea"] = 3
    arc_size.loc["North_Sea", "Arnhem"] = 3

    arc_size.loc["Arnhem", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "Arnhem"] = 3

    arc_size.loc["Arnhem", "Chemelot"] = 3
    arc_size.loc["Chemelot", "Arnhem"] = 3

    arc_size.loc["Arnhem", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "Arnhem"] = 3

    arc_size.loc["East_Groningen", "Betuwe"] = 3
    arc_size.loc["Betuwe", "East_Groningen"] = 3

    arc_size.loc["East_Groningen", "Heerlen"] = 3
    arc_size.loc["Heerlen", "East_Groningen"] = 3

    arc_size.loc["East_Groningen", "Dordrecht"] = 3
    arc_size.loc["Dordrecht", "East_Groningen"] = 3

    arc_size.loc["East_Groningen", "Wageningen"] = 3
    arc_size.loc["Wageningen", "East_Groningen"] = 3

    arc_size.loc["East_Groningen", "Rotterdam"] = 3
    arc_size.loc["Rotterdam", "East_Groningen"] = 3

    arc_size.loc["East_Groningen", "Zeeland"] = 3
    arc_size.loc["Zeeland", "East_Groningen"] = 3

    arc_size.loc["East_Groningen", "North_Sea"] = 3
    arc_size.loc["North_Sea", "East_Groningen"] = 3

    arc_size.loc["East_Groningen", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "East_Groningen"] = 3

    arc_size.loc["East_Groningen", "Chemelot"] = 3
    arc_size.loc["Chemelot", "East_Groningen"] = 3

    arc_size.loc["East_Groningen", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "East_Groningen"] = 3

    arc_size.loc["Betuwe", "Heerlen"] = 3
    arc_size.loc["Heerlen", "Betuwe"] = 3

    arc_size.loc["Betuwe", "Dordrecht"] = 3
    arc_size.loc["Dordrecht", "Betuwe"] = 3

    arc_size.loc["Betuwe", "Wageningen"] = 3
    arc_size.loc["Wageningen", "Betuwe"] = 3

    arc_size.loc["Betuwe", "Rotterdam"] = 3
    arc_size.loc["Rotterdam", "Betuwe"] = 3

    arc_size.loc["Betuwe", "Zeeland"] = 3
    arc_size.loc["Zeeland", "Betuwe"] = 3

    arc_size.loc["Betuwe", "North_Sea"] = 3
    arc_size.loc["North_Sea", "Betuwe"] = 3

    arc_size.loc["Betuwe", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "Betuwe"] = 3

    arc_size.loc["Betuwe", "Chemelot"] = 3
    arc_size.loc["Chemelot", "Betuwe"] = 3

    arc_size.loc["Betuwe", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "Betuwe"] = 3

    arc_size.loc["Heerlen", "Dordrecht"] = 3
    arc_size.loc["Dordrecht", "Heerlen"] = 3

    arc_size.loc["Heerlen", "Wageningen"] = 3
    arc_size.loc["Wageningen", "Heerlen"] = 3

    arc_size.loc["Heerlen", "Rotterdam"] = 3
    arc_size.loc["Rotterdam", "Heerlen"] = 3

    arc_size.loc["Heerlen", "Zeeland"] = 3
    arc_size.loc["Zeeland", "Heerlen"] = 3

    arc_size.loc["Heerlen", "North_Sea"] = 3
    arc_size.loc["North_Sea", "Heerlen"] = 3

    arc_size.loc["Heerlen", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "Heerlen"] = 3

    arc_size.loc["Heerlen", "Chemelot"] = 3
    arc_size.loc["Chemelot", "Heerlen"] = 3

    arc_size.loc["Heerlen", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "Heerlen"] = 3

    arc_size.loc["Dordrecht", "Wageningen"] = 3
    arc_size.loc["Wageningen", "Dordrecht"] = 3

    arc_size.loc["Dordrecht", "Rotterdam"] = 3
    arc_size.loc["Rotterdam", "Dordrecht"] = 3

    arc_size.loc["Dordrecht", "Zeeland"] = 3
    arc_size.loc["Zeeland", "Dordrecht"] = 3

    arc_size.loc["Dordrecht", "North_Sea"] = 3
    arc_size.loc["North_Sea", "Dordrecht"] = 3

    arc_size.loc["Dordrecht", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "Dordrecht"] = 3

    arc_size.loc["Dordrecht", "Chemelot"] = 3
    arc_size.loc["Chemelot", "Dordrecht"] = 3

    arc_size.loc["Dordrecht", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "Dordrecht"] = 3

    arc_size.loc["Wageningen", "Rotterdam"] = 3
    arc_size.loc["Rotterdam", "Wageningen"] = 3

    arc_size.loc["Wageningen", "Zeeland"] = 3
    arc_size.loc["Zeeland", "Wageningen"] = 3

    arc_size.loc["Wageningen", "North_Sea"] = 3
    arc_size.loc["North_Sea", "Wageningen"] = 3

    arc_size.loc["Wageningen", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "Wageningen"] = 3

    arc_size.loc["Wageningen", "Chemelot"] = 3
    arc_size.loc["Chemelot", "Wageningen"] = 3

    arc_size.loc["Wageningen", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "Wageningen"] = 3

    arc_size.loc["Rotterdam", "Zeeland"] = 3
    arc_size.loc["Zeeland", "Rotterdam"] = 3

    arc_size.loc["Rotterdam", "North_Sea"] = 3
    arc_size.loc["North_Sea", "Rotterdam"] = 3

    arc_size.loc["Rotterdam", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "Rotterdam"] = 3

    arc_size.loc["Rotterdam", "Chemelot"] = 3
    arc_size.loc["Chemelot", "Rotterdam"] = 3

    arc_size.loc["Rotterdam", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "Rotterdam"] = 3

    arc_size.loc["Zeeland", "North_Sea"] = 3
    arc_size.loc["North_Sea", "Zeeland"] = 3

    arc_size.loc["Zeeland", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "Zeeland"] = 3

    arc_size.loc["Zeeland", "Chemelot"] = 3
    arc_size.loc["Chemelot", "Zeeland"] = 3

    arc_size.loc["Zeeland", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "Zeeland"] = 3

    arc_size.loc["North_Sea", "North_Netherlands"] = 3
    arc_size.loc["North_Netherlands", "North_Sea"] = 3

    arc_size.loc["North_Sea", "Chemelot"] = 3
    arc_size.loc["Chemelot", "North_Sea"] = 3

    arc_size.loc["North_Sea", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "North_Sea"] = 3

    arc_size.loc["North_Netherlands", "Chemelot"] = 3
    arc_size.loc["Chemelot", "North_Netherlands"] = 3

    arc_size.loc["North_Netherlands", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "North_Netherlands"] = 3

    arc_size.loc["Chemelot", "Zuidwending"] = 3
    arc_size.loc["Zuidwending", "Chemelot"] = 3
    # </editor-fold>

    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "size_max_arcs.csv",
        sep=";")
    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "size_max_arcs.csv",
        sep=";")

    # <editor-fold desc="Arc size connections truck">
    # arc_size.loc["Roermond", "Dongen"] = 3
    # arc_size.loc["Dongen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Oosterhout"] = 3
    # arc_size.loc["Oosterhout", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Maastricht"] = 3
    # arc_size.loc["Maastricht", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Venlo"] = 3
    # arc_size.loc["Venlo", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Roermond"] = 3
    #
    # arc_size.loc["Dongen", "Oosterhout"] = 3
    # arc_size.loc["Oosterhout", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Maastricht"] = 3
    # arc_size.loc["Maastricht", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Venlo"] = 3
    # arc_size.loc["Venlo", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Dongen"] = 3
    #
    # arc_size.loc["Oosterhout", "Maastricht"] = 3
    # arc_size.loc["Maastricht", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Venlo"] = 3
    # arc_size.loc["Venlo", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Oosterhout"] = 3
    #
    # arc_size.loc["Maastricht", "Venlo"] = 3
    # arc_size.loc["Venlo", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Maastricht"] = 3
    #
    # arc_size.loc["Venlo", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Venlo"] = 3
    #
    # arc_size.loc["Arnhem", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Arnhem"] = 3
    #
    # arc_size.loc["East_Groningen", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "East_Groningen"] = 3
    #
    # arc_size.loc["Betuwe", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Betuwe"] = 3
    #
    # arc_size.loc["Heerlen", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Heerlen"] = 3
    #
    # arc_size.loc["Dordrecht", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Dordrecht"] = 3
    #
    # arc_size.loc["Wageningen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Wageningen"] = 3
    #
    # arc_size.loc["Rotterdam", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Rotterdam"] = 3
    #
    # arc_size.loc["Zeeland", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Zeeland"] = 3
    #
    # arc_size.loc["Zeeland", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Zeeland"] = 3
    #
    # arc_size.loc["Zeeland", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Zeeland"] = 3
    #
    # arc_size.loc["Zeeland", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Zeeland"] = 3
    #
    # arc_size.loc["North_Sea", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "North_Sea"] = 3
    #
    # arc_size.loc["North_Sea", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "North_Sea"] = 3
    #
    # arc_size.loc["North_Sea", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "North_Sea"] = 3
    #
    # arc_size.loc["North_Netherlands", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "North_Netherlands"] = 3
    #
    # arc_size.loc["North_Netherlands", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "North_Netherlands"] = 3
    #
    # arc_size.loc["Chemelot", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Chemelot"] = 3
    # # </editor-fold>
    #
    # arc_size.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "size_max_arcs.csv", sep=";")

    print("Max size per arc:", arc_size)

    # Delete the max_size_arc template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv")

    # # Connection
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)
    # <editor-fold desc="Connections">
    connection.loc["Roermond", "Dongen"] = 1
    connection.loc["Dongen", "Roermond"] = 1

    connection.loc["Roermond", "Oosterhout"] = 1
    connection.loc["Oosterhout", "Roermond"] = 1

    connection.loc["Roermond", "Maastricht"] = 1
    connection.loc["Maastricht", "Roermond"] = 1

    connection.loc["Roermond", "Venlo"] = 1
    connection.loc["Venlo", "Roermond"] = 1

    connection.loc["Roermond", "Arnhem"] = 1
    connection.loc["Arnhem", "Roermond"] = 1

    connection.loc["Roermond", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Roermond"] = 1

    connection.loc["Roermond", "Betuwe"] = 1
    connection.loc["Betuwe", "Roermond"] = 1

    connection.loc["Roermond", "Heerlen"] = 1
    connection.loc["Heerlen", "Roermond"] = 1

    connection.loc["Roermond", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Roermond"] = 1

    connection.loc["Roermond", "Wageningen"] = 1
    connection.loc["Wageningen", "Roermond"] = 1

    connection.loc["Roermond", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Roermond"] = 1

    connection.loc["Roermond", "Zeeland"] = 1
    connection.loc["Zeeland", "Roermond"] = 1

    connection.loc["Roermond", "North_Sea"] = 1
    connection.loc["North_Sea", "Roermond"] = 1

    connection.loc["Roermond", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Roermond"] = 1

    connection.loc["Roermond", "Chemelot"] = 1
    connection.loc["Chemelot", "Roermond"] = 1

    connection.loc["Roermond", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Roermond"] = 1

    connection.loc["Dongen", "Oosterhout"] = 1
    connection.loc["Oosterhout", "Dongen"] = 1

    connection.loc["Dongen", "Maastricht"] = 1
    connection.loc["Maastricht", "Dongen"] = 1

    connection.loc["Dongen", "Venlo"] = 1
    connection.loc["Venlo", "Dongen"] = 1

    connection.loc["Dongen", "Arnhem"] = 1
    connection.loc["Arnhem", "Dongen"] = 1

    connection.loc["Dongen", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Dongen"] = 1

    connection.loc["Dongen", "Betuwe"] = 1
    connection.loc["Betuwe", "Dongen"] = 1

    connection.loc["Dongen", "Heerlen"] = 1
    connection.loc["Heerlen", "Dongen"] = 1

    connection.loc["Dongen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Dongen"] = 1

    connection.loc["Dongen", "Wageningen"] = 1
    connection.loc["Wageningen", "Dongen"] = 1

    connection.loc["Dongen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Dongen"] = 1

    connection.loc["Dongen", "Zeeland"] = 1
    connection.loc["Zeeland", "Dongen"] = 1

    connection.loc["Dongen", "North_Sea"] = 1
    connection.loc["North_Sea", "Dongen"] = 1

    connection.loc["Dongen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Dongen"] = 1

    connection.loc["Dongen", "Chemelot"] = 1
    connection.loc["Chemelot", "Dongen"] = 1

    connection.loc["Dongen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Dongen"] = 1

    connection.loc["Oosterhout", "Maastricht"] = 1
    connection.loc["Maastricht", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Venlo"] = 1
    connection.loc["Venlo", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Arnhem"] = 1
    connection.loc["Arnhem", "Oosterhout"] = 1

    connection.loc["Oosterhout", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Betuwe"] = 1
    connection.loc["Betuwe", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Heerlen"] = 1
    connection.loc["Heerlen", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Wageningen"] = 1
    connection.loc["Wageningen", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Zeeland"] = 1
    connection.loc["Zeeland", "Oosterhout"] = 1

    connection.loc["Oosterhout", "North_Sea"] = 1
    connection.loc["North_Sea", "Oosterhout"] = 1

    connection.loc["Oosterhout", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Chemelot"] = 1
    connection.loc["Chemelot", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Oosterhout"] = 1

    connection.loc["Maastricht", "Venlo"] = 1
    connection.loc["Venlo", "Maastricht"] = 1

    connection.loc["Maastricht", "Arnhem"] = 1
    connection.loc["Arnhem", "Maastricht"] = 1

    connection.loc["Maastricht", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Maastricht"] = 1

    connection.loc["Maastricht", "Betuwe"] = 1
    connection.loc["Betuwe", "Maastricht"] = 1

    connection.loc["Maastricht", "Heerlen"] = 1
    connection.loc["Heerlen", "Maastricht"] = 1

    connection.loc["Maastricht", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Maastricht"] = 1

    connection.loc["Maastricht", "Wageningen"] = 1
    connection.loc["Wageningen", "Maastricht"] = 1

    connection.loc["Maastricht", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Maastricht"] = 1

    connection.loc["Maastricht", "Zeeland"] = 1
    connection.loc["Zeeland", "Maastricht"] = 1

    connection.loc["Maastricht", "North_Sea"] = 1
    connection.loc["North_Sea", "Maastricht"] = 1

    connection.loc["Maastricht", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Maastricht"] = 1

    connection.loc["Maastricht", "Chemelot"] = 1
    connection.loc["Chemelot", "Maastricht"] = 1

    connection.loc["Maastricht", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Maastricht"] = 1

    connection.loc["Venlo", "Arnhem"] = 1
    connection.loc["Arnhem", "Venlo"] = 1

    connection.loc["Venlo", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Venlo"] = 1

    connection.loc["Venlo", "Betuwe"] = 1
    connection.loc["Betuwe", "Venlo"] = 1

    connection.loc["Venlo", "Heerlen"] = 1
    connection.loc["Heerlen", "Venlo"] = 1

    connection.loc["Venlo", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Venlo"] = 1

    connection.loc["Venlo", "Wageningen"] = 1
    connection.loc["Wageningen", "Venlo"] = 1

    connection.loc["Venlo", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Venlo"] = 1

    connection.loc["Venlo", "Zeeland"] = 1
    connection.loc["Zeeland", "Venlo"] = 1

    connection.loc["Venlo", "North_Sea"] = 1
    connection.loc["North_Sea", "Venlo"] = 1

    connection.loc["Venlo", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Venlo"] = 1

    connection.loc["Venlo", "Chemelot"] = 1
    connection.loc["Chemelot", "Venlo"] = 1

    connection.loc["Venlo", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Venlo"] = 1

    connection.loc["Arnhem", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Arnhem"] = 1

    connection.loc["Arnhem", "Betuwe"] = 1
    connection.loc["Betuwe", "Arnhem"] = 1

    connection.loc["Arnhem", "Heerlen"] = 1
    connection.loc["Heerlen", "Arnhem"] = 1

    connection.loc["Arnhem", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Arnhem"] = 1

    connection.loc["Arnhem", "Wageningen"] = 1
    connection.loc["Wageningen", "Arnhem"] = 1

    connection.loc["Arnhem", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Arnhem"] = 1

    connection.loc["Arnhem", "Zeeland"] = 1
    connection.loc["Zeeland", "Arnhem"] = 1

    connection.loc["Arnhem", "North_Sea"] = 1
    connection.loc["North_Sea", "Arnhem"] = 1

    connection.loc["Arnhem", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Arnhem"] = 1

    connection.loc["Arnhem", "Chemelot"] = 1
    connection.loc["Chemelot", "Arnhem"] = 1

    connection.loc["Arnhem", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Arnhem"] = 1

    connection.loc["East_Groningen", "Betuwe"] = 1
    connection.loc["Betuwe", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Heerlen"] = 1
    connection.loc["Heerlen", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Wageningen"] = 1
    connection.loc["Wageningen", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Zeeland"] = 1
    connection.loc["Zeeland", "East_Groningen"] = 1

    connection.loc["East_Groningen", "North_Sea"] = 1
    connection.loc["North_Sea", "East_Groningen"] = 1

    connection.loc["East_Groningen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Chemelot"] = 1
    connection.loc["Chemelot", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "East_Groningen"] = 1

    connection.loc["Betuwe", "Heerlen"] = 1
    connection.loc["Heerlen", "Betuwe"] = 1

    connection.loc["Betuwe", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Betuwe"] = 1

    connection.loc["Betuwe", "Wageningen"] = 1
    connection.loc["Wageningen", "Betuwe"] = 1

    connection.loc["Betuwe", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Betuwe"] = 1

    connection.loc["Betuwe", "Zeeland"] = 1
    connection.loc["Zeeland", "Betuwe"] = 1

    connection.loc["Betuwe", "North_Sea"] = 1
    connection.loc["North_Sea", "Betuwe"] = 1

    connection.loc["Betuwe", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Betuwe"] = 1

    connection.loc["Betuwe", "Chemelot"] = 1
    connection.loc["Chemelot", "Betuwe"] = 1

    connection.loc["Betuwe", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Betuwe"] = 1

    connection.loc["Heerlen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Heerlen"] = 1

    connection.loc["Heerlen", "Wageningen"] = 1
    connection.loc["Wageningen", "Heerlen"] = 1

    connection.loc["Heerlen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Heerlen"] = 1

    connection.loc["Heerlen", "Zeeland"] = 1
    connection.loc["Zeeland", "Heerlen"] = 1

    connection.loc["Heerlen", "North_Sea"] = 1
    connection.loc["North_Sea", "Heerlen"] = 1

    connection.loc["Heerlen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Heerlen"] = 1

    connection.loc["Heerlen", "Chemelot"] = 1
    connection.loc["Chemelot", "Heerlen"] = 1

    connection.loc["Heerlen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Heerlen"] = 1

    connection.loc["Dordrecht", "Wageningen"] = 1
    connection.loc["Wageningen", "Dordrecht"] = 1

    connection.loc["Dordrecht", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Dordrecht"] = 1

    connection.loc["Dordrecht", "Zeeland"] = 1
    connection.loc["Zeeland", "Dordrecht"] = 1

    connection.loc["Dordrecht", "North_Sea"] = 1
    connection.loc["North_Sea", "Dordrecht"] = 1

    connection.loc["Dordrecht", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Dordrecht"] = 1

    connection.loc["Dordrecht", "Chemelot"] = 1
    connection.loc["Chemelot", "Dordrecht"] = 1

    connection.loc["Dordrecht", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Dordrecht"] = 1

    connection.loc["Wageningen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Wageningen"] = 1

    connection.loc["Wageningen", "Zeeland"] = 1
    connection.loc["Zeeland", "Wageningen"] = 1

    connection.loc["Wageningen", "North_Sea"] = 1
    connection.loc["North_Sea", "Wageningen"] = 1

    connection.loc["Wageningen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Wageningen"] = 1

    connection.loc["Wageningen", "Chemelot"] = 1
    connection.loc["Chemelot", "Wageningen"] = 1

    connection.loc["Wageningen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Wageningen"] = 1

    connection.loc["Rotterdam", "Zeeland"] = 1
    connection.loc["Zeeland", "Rotterdam"] = 1

    connection.loc["Rotterdam", "North_Sea"] = 1
    connection.loc["North_Sea", "Rotterdam"] = 1

    connection.loc["Rotterdam", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Rotterdam"] = 1

    connection.loc["Rotterdam", "Chemelot"] = 1
    connection.loc["Chemelot", "Rotterdam"] = 1

    connection.loc["Rotterdam", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Rotterdam"] = 1

    connection.loc["Zeeland", "North_Sea"] = 1
    connection.loc["North_Sea", "Zeeland"] = 1

    connection.loc["Zeeland", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Zeeland"] = 1

    connection.loc["Zeeland", "Chemelot"] = 1
    connection.loc["Chemelot", "Zeeland"] = 1

    connection.loc["Zeeland", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Zeeland"] = 1

    connection.loc["North_Sea", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "North_Sea"] = 1

    connection.loc["North_Sea", "Chemelot"] = 1
    connection.loc["Chemelot", "North_Sea"] = 1

    connection.loc["North_Sea", "Zuidwending"] = 1
    connection.loc["Zuidwending", "North_Sea"] = 1

    connection.loc["North_Netherlands", "Chemelot"] = 1
    connection.loc["Chemelot", "North_Netherlands"] = 1

    connection.loc["North_Netherlands", "Zuidwending"] = 1
    connection.loc["Zuidwending", "North_Netherlands"] = 1

    connection.loc["Chemelot", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Chemelot"] = 1

    # </editor-fold>
    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "connection.csv",
        sep=";")
    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "connection.csv",
        sep=";")
    #connection.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "connection.csv", sep=";")

    print("Connection:", connection)

    # Delete the template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "connection.csv")

    # # Distance
    distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "distance.csv", sep=";",
                           index_col=0)
    # <editor-fold desc="Distances">

    distance.loc["Roermond", "Dongen"] = 87
    distance.loc["Dongen", "Roermond"] = 87

    distance.loc["Roermond", "Oosterhout"] = 93
    distance.loc["Oosterhout", "Roermond"] = 93

    distance.loc["Roermond", "Maastricht"] = 43
    distance.loc["Maastricht", "Roermond"] = 43

    distance.loc["Roermond", "Venlo"] = 23
    distance.loc["Venlo", "Roermond"] = 23

    distance.loc["Roermond", "Arnhem"] = 88
    distance.loc["Arnhem", "Roermond"] = 88

    distance.loc["Roermond", "East_Groningen"] = 226
    distance.loc["East_Groningen", "Roermond"] = 226

    distance.loc["Roermond", "Betuwe"] = 87
    distance.loc["Betuwe", "Roermond"] = 87

    distance.loc["Roermond", "Heerlen"] = 35
    distance.loc["Heerlen", "Roermond"] = 35

    distance.loc["Roermond", "Dordrecht"] = 113
    distance.loc["Dordrecht", "Roermond"] = 113

    distance.loc["Roermond", "Wageningen"] = 89
    distance.loc["Wageningen", "Roermond"] = 89

    distance.loc["Roermond", "Rotterdam"] = 132
    distance.loc["Rotterdam", "Roermond"] = 132

    distance.loc["Roermond", "Zeeland"] = 152
    distance.loc["Zeeland", "Roermond"] = 152

    distance.loc["Roermond", "North_Sea"] = 160
    distance.loc["North_Sea", "Roermond"] = 160

    distance.loc["Roermond", "North_Netherlands"] = 258
    distance.loc["North_Netherlands", "Roermond"] = 258

    distance.loc["Roermond", "Chemelot"] = 28
    distance.loc["Chemelot", "Roermond"] = 28

    distance.loc["Roermond", "Zuidwending"] = 221
    distance.loc["Zuidwending", "Roermond"] = 221

    distance.loc["Dongen", "Oosterhout"] = 6
    distance.loc["Oosterhout", "Dongen"] = 6

    distance.loc["Dongen", "Maastricht"] = 100
    distance.loc["Maastricht", "Dongen"] = 100

    distance.loc["Dongen", "Venlo"] = 89
    distance.loc["Venlo", "Dongen"] = 89

    distance.loc["Dongen", "Arnhem"] = 77
    distance.loc["Arnhem", "Dongen"] = 77

    distance.loc["Dongen", "East_Groningen"] = 210
    distance.loc["East_Groningen", "Dongen"] = 210

    distance.loc["Dongen", "Betuwe"] = 50
    distance.loc["Betuwe", "Dongen"] = 50

    distance.loc["Dongen", "Heerlen"] = 109
    distance.loc["Heerlen", "Dongen"] = 109

    distance.loc["Dongen", "Dordrecht"] = 26
    distance.loc["Dordrecht", "Dongen"] = 26

    distance.loc["Dongen", "Wageningen"] = 63
    distance.loc["Wageningen", "Dongen"] = 63

    distance.loc["Dongen", "Rotterdam"] = 46
    distance.loc["Rotterdam", "Dongen"] = 46

    distance.loc["Dongen", "Zeeland"] = 77
    distance.loc["Zeeland", "Dongen"] = 77

    distance.loc["Dongen", "North_Sea"] = 90
    distance.loc["North_Sea", "Dongen"] = 90

    distance.loc["Dongen", "North_Netherlands"] = 240
    distance.loc["North_Netherlands", "Dongen"] = 240

    distance.loc["Dongen", "Chemelot"] = 93
    distance.loc["Chemelot", "Dongen"] = 93

    distance.loc["Dongen", "Zuidwending"] = 212
    distance.loc["Zuidwending", "Dongen"] = 212

    distance.loc["Oosterhout", "Maastricht"] = 105
    distance.loc["Maastricht", "Oosterhout"] = 105

    distance.loc["Oosterhout", "Venlo"] = 95
    distance.loc["Venlo", "Oosterhout"] = 95

    distance.loc["Oosterhout", "Arnhem"] = 81
    distance.loc["Arnhem", "Oosterhout"] = 81

    distance.loc["Oosterhout", "East_Groningen"] = 212
    distance.loc["East_Groningen", "Oosterhout"] = 212

    distance.loc["Oosterhout", "Betuwe"] = 54
    distance.loc["Betuwe", "Oosterhout"] = 54

    distance.loc["Oosterhout", "Heerlen"] = 115
    distance.loc["Heerlen", "Oosterhout"] = 115

    distance.loc["Oosterhout", "Dordrecht"] = 21
    distance.loc["Dordrecht", "Oosterhout"] = 21

    distance.loc["Oosterhout", "Wageningen"] = 66
    distance.loc["Wageningen", "Oosterhout"] = 66

    distance.loc["Oosterhout", "Rotterdam"] = 41
    distance.loc["Rotterdam", "Oosterhout"] = 41

    distance.loc["Oosterhout", "Zeeland"] = 72
    distance.loc["Zeeland", "Oosterhout"] = 72

    distance.loc["Oosterhout", "North_Sea"] = 88
    distance.loc["North_Sea", "Oosterhout"] = 88

    distance.loc["Oosterhout", "North_Netherlands"] = 241
    distance.loc["North_Netherlands", "Oosterhout"] = 241

    distance.loc["Oosterhout", "Chemelot"] = 99
    distance.loc["Chemelot", "Oosterhout"] = 99

    distance.loc["Oosterhout", "Zuidwending"] = 214
    distance.loc["Zuidwending", "Oosterhout"] = 214

    distance.loc["Maastricht", "Venlo"] = 66
    distance.loc["Venlo", "Maastricht"] = 66

    distance.loc["Maastricht", "Arnhem"] = 127
    distance.loc["Arnhem", "Maastricht"] = 127

    distance.loc["Maastricht", "East_Groningen"] = 268
    distance.loc["East_Groningen", "Maastricht"] = 268

    distance.loc["Maastricht", "Betuwe"] = 119
    distance.loc["Betuwe", "Maastricht"] = 119

    distance.loc["Maastricht", "Heerlen"] = 21
    distance.loc["Heerlen", "Maastricht"] = 21

    distance.loc["Maastricht", "Dordrecht"] = 126
    distance.loc["Dordrecht", "Maastricht"] = 126

    distance.loc["Maastricht", "Wageningen"] = 124
    distance.loc["Wageningen", "Maastricht"] = 124

    distance.loc["Maastricht", "Rotterdam"] = 146
    distance.loc["Rotterdam", "Maastricht"] = 146

    distance.loc["Maastricht", "Zeeland"] = 147
    distance.loc["Zeeland", "Maastricht"] = 147

    distance.loc["Maastricht", "North_Sea"] = 186
    distance.loc["North_Sea", "Maastricht"] = 186

    distance.loc["Maastricht", "North_Netherlands"] = 300
    distance.loc["North_Netherlands", "Maastricht"] = 300

    distance.loc["Maastricht", "Chemelot"] = 16
    distance.loc["Chemelot", "Maastricht"] = 16

    distance.loc["Maastricht", "Zuidwending"] = 264
    distance.loc["Zuidwending", "Maastricht"] = 264

    distance.loc["Venlo", "Arnhem"] = 71
    distance.loc["Arnhem", "Venlo"] = 71

    distance.loc["Venlo", "East_Groningen"] = 204
    distance.loc["East_Groningen", "Venlo"] = 204

    distance.loc["Venlo", "Betuwe"] = 77
    distance.loc["Betuwe", "Venlo"] = 77

    distance.loc["Venlo", "Heerlen"] = 55
    distance.loc["Heerlen", "Venlo"] = 55

    distance.loc["Venlo", "Dordrecht"] = 113
    distance.loc["Dordrecht", "Venlo"] = 113

    distance.loc["Venlo", "Wageningen"] = 75
    distance.loc["Wageningen", "Venlo"] = 75

    distance.loc["Venlo", "Rotterdam"] = 132
    distance.loc["Rotterdam", "Venlo"] = 132

    distance.loc["Venlo", "Zeeland"] = 161
    distance.loc["Zeeland", "Venlo"] = 161

    distance.loc["Venlo", "North_Sea"] = 150
    distance.loc["North_Sea", "Venlo"] = 150

    distance.loc["Venlo", "North_Netherlands"] = 236
    distance.loc["North_Netherlands", "Venlo"] = 236

    distance.loc["Venlo", "Chemelot"] = 50
    distance.loc["Chemelot", "Venlo"] = 50

    distance.loc["Venlo", "Zuidwending"] = 199
    distance.loc["Zuidwending", "Venlo"] = 199

    distance.loc["Arnhem", "East_Groningen"] = 143
    distance.loc["East_Groningen", "Arnhem"] = 143

    distance.loc["Arnhem", "Betuwe"] = 28
    distance.loc["Betuwe", "Arnhem"] = 28

    distance.loc["Arnhem", "Heerlen"] = 123
    distance.loc["Heerlen", "Arnhem"] = 123

    distance.loc["Arnhem", "Dordrecht"] = 86
    distance.loc["Dordrecht", "Arnhem"] = 86

    distance.loc["Arnhem", "Wageningen"] = 16
    distance.loc["Wageningen", "Arnhem"] = 16

    distance.loc["Arnhem", "Rotterdam"] = 98
    distance.loc["Rotterdam", "Arnhem"] = 98

    distance.loc["Arnhem", "Zeeland"] = 151
    distance.loc["Zeeland", "Arnhem"] = 151

    distance.loc["Arnhem", "North_Sea"] = 89
    distance.loc["North_Sea", "Arnhem"] = 89

    distance.loc["Arnhem", "North_Netherlands"] = 175
    distance.loc["North_Netherlands", "Arnhem"] = 175

    distance.loc["Arnhem", "Chemelot"] = 112
    distance.loc["Chemelot", "Arnhem"] = 112

    distance.loc["Arnhem", "Zuidwending"] = 142
    distance.loc["Zuidwending", "Arnhem"] = 142

    distance.loc["East_Groningen", "Betuwe"] = 162
    distance.loc["Betuwe", "East_Groningen"] = 162

    distance.loc["East_Groningen", "Heerlen"] = 260
    distance.loc["Heerlen", "East_Groningen"] = 260

    distance.loc["East_Groningen", "Dordrecht"] = 207
    distance.loc["Dordrecht", "East_Groningen"] = 207

    distance.loc["East_Groningen", "Wageningen"] = 152
    distance.loc["Wageningen", "East_Groningen"] = 152

    distance.loc["East_Groningen", "Rotterdam"] = 206
    distance.loc["Rotterdam", "East_Groningen"] = 206

    distance.loc["East_Groningen", "Zeeland"] = 270
    distance.loc["Zeeland", "East_Groningen"] = 270

    distance.loc["East_Groningen", "North_Sea"] = 152
    distance.loc["North_Sea", "East_Groningen"] = 152

    distance.loc["East_Groningen", "North_Netherlands"] = 32
    distance.loc["North_Netherlands", "East_Groningen"] = 32

    distance.loc["East_Groningen", "Chemelot"] = 252
    distance.loc["Chemelot", "East_Groningen"] = 252

    distance.loc["East_Groningen", "Zuidwending"] = 16
    distance.loc["Zuidwending", "East_Groningen"] = 16

    distance.loc["Betuwe", "Heerlen"] = 120
    distance.loc["Heerlen", "Betuwe"] = 120

    distance.loc["Betuwe", "Dordrecht"] = 58
    distance.loc["Dordrecht", "Betuwe"] = 58

    distance.loc["Betuwe", "Wageningen"] = 13
    distance.loc["Wageningen", "Betuwe"] = 13

    distance.loc["Betuwe", "Rotterdam"] = 70
    distance.loc["Rotterdam", "Betuwe"] = 70

    distance.loc["Betuwe", "Zeeland"] = 123
    distance.loc["Zeeland", "Betuwe"] = 123

    distance.loc["Betuwe", "North_Sea"] = 74
    distance.loc["North_Sea", "Betuwe"] = 74

    distance.loc["Betuwe", "North_Netherlands"] = 193
    distance.loc["North_Netherlands", "Betuwe"] = 193

    distance.loc["Betuwe", "Chemelot"] = 107
    distance.loc["Chemelot", "Betuwe"] = 107

    distance.loc["Betuwe", "Zuidwending"] = 163
    distance.loc["Zuidwending", "Betuwe"] = 163

    distance.loc["Heerlen", "Dordrecht"] = 136
    distance.loc["Dordrecht", "Heerlen"] = 136

    distance.loc["Heerlen", "Wageningen"] = 123
    distance.loc["Wageningen", "Heerlen"] = 123

    distance.loc["Heerlen", "Rotterdam"] = 156
    distance.loc["Rotterdam", "Heerlen"] = 156

    distance.loc["Heerlen", "Zeeland"] = 164
    distance.loc["Zeeland", "Heerlen"] = 164

    distance.loc["Heerlen", "North_Sea"] = 190
    distance.loc["North_Sea", "Heerlen"] = 190

    distance.loc["Heerlen", "North_Netherlands"] = 292
    distance.loc["North_Netherlands", "Heerlen"] = 292

    distance.loc["Heerlen", "Chemelot"] = 16
    distance.loc["Chemelot", "Heerlen"] = 16

    distance.loc["Heerlen", "Zuidwending"] = 254
    distance.loc["Zuidwending", "Heerlen"] = 254

    distance.loc["Dordrecht", "Wageningen"] = 70
    distance.loc["Wageningen", "Dordrecht"] = 70

    distance.loc["Dordrecht", "Rotterdam"] = 20
    distance.loc["Rotterdam", "Dordrecht"] = 20

    distance.loc["Dordrecht", "Zeeland"] = 66
    distance.loc["Zeeland", "Dordrecht"] = 66

    distance.loc["Dordrecht", "North_Sea"] = 71
    distance.loc["North_Sea", "Dordrecht"] = 71

    distance.loc["Dordrecht", "North_Netherlands"] = 235
    distance.loc["North_Netherlands", "Dordrecht"] = 235

    distance.loc["Dordrecht", "Chemelot"] = 120
    distance.loc["Chemelot", "Dordrecht"] = 120

    distance.loc["Dordrecht", "Zuidwending"] = 210
    distance.loc["Zuidwending", "Dordrecht"] = 210

    distance.loc["Wageningen", "Rotterdam"] = 82
    distance.loc["Rotterdam", "Wageningen"] = 82

    distance.loc["Wageningen", "Zeeland"] = 136
    distance.loc["Zeeland", "Wageningen"] = 136

    distance.loc["Wageningen", "North_Sea"] = 77
    distance.loc["North_Sea", "Wageningen"] = 77

    distance.loc["Wageningen", "North_Netherlands"] = 183
    distance.loc["North_Netherlands", "Wageningen"] = 183

    distance.loc["Wageningen", "Chemelot"] = 111
    distance.loc["Chemelot", "Wageningen"] = 111

    distance.loc["Wageningen", "Zuidwending"] = 152
    distance.loc["Zuidwending", "Wageningen"] = 152

    distance.loc["Rotterdam", "Zeeland"] = 64
    distance.loc["Zeeland", "Rotterdam"] = 64

    distance.loc["Rotterdam", "North_Sea"] = 61
    distance.loc["North_Sea", "Rotterdam"] = 61

    distance.loc["Rotterdam", "North_Netherlands"] = 232
    distance.loc["North_Netherlands", "Rotterdam"] = 232

    distance.loc["Rotterdam", "Chemelot"] = 140
    distance.loc["Chemelot", "Rotterdam"] = 140

    distance.loc["Rotterdam", "Zuidwending"] = 211
    distance.loc["Zuidwending", "Rotterdam"] = 211

    distance.loc["Zeeland", "North_Sea"] = 123
    distance.loc["North_Sea", "Zeeland"] = 123

    distance.loc["Zeeland", "North_Netherlands"] = 296
    distance.loc["North_Netherlands", "Zeeland"] = 296

    distance.loc["Zeeland", "Chemelot"] = 148
    distance.loc["Chemelot", "Zeeland"] = 148

    distance.loc["Zeeland", "Zuidwending"] = 275
    distance.loc["Zuidwending", "Zeeland"] = 275

    distance.loc["North_Sea", "North_Netherlands"] = 176
    distance.loc["North_Netherlands", "North_Sea"] = 176

    distance.loc["North_Sea", "Chemelot"] = 176
    distance.loc["Chemelot", "North_Sea"] = 176

    distance.loc["North_Sea", "Zuidwending"] = 160
    distance.loc["Zuidwending", "North_Sea"] = 160

    distance.loc["North_Netherlands", "Chemelot"] = 284
    distance.loc["Chemelot", "North_Netherlands"] = 284

    distance.loc["North_Netherlands", "Zuidwending"] = 41
    distance.loc["Zuidwending", "North_Netherlands"] = 41

    distance.loc["Chemelot", "Zuidwending"] = 248
    distance.loc["Zuidwending", "Chemelot"] = 248

    # </editor-fold>

    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "distance.csv",
        sep=";")
    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "distance.csv",
        sep=";")
    #distance.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "distance.csv", sep=";")
    print("Distance:", distance)

    # Delete the template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "distance.csv")

def add_new_network_H2_small(input_data_path):
    # # Make a new folder for the new network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP_big",
                exist_ok=True)
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP_big",
                exist_ok=True)
    #os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP", exist_ok=True)

    print("New network")

    # # max size arc
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)
    # <editor-fold desc="Arc size connections pipeline">
    arc_size.loc["Roermond", "Dongen"] = 2
    arc_size.loc["Dongen", "Roermond"] = 2

    arc_size.loc["Roermond", "Oosterhout"] = 2
    arc_size.loc["Oosterhout", "Roermond"] = 2

    arc_size.loc["Roermond", "Maastricht"] = 2
    arc_size.loc["Maastricht", "Roermond"] = 2

    arc_size.loc["Roermond", "Venlo"] = 2
    arc_size.loc["Venlo", "Roermond"] = 2

    arc_size.loc["Roermond", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Roermond"] = 2

    arc_size.loc["Roermond", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Roermond"] = 2

    arc_size.loc["Roermond", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Roermond"] = 2

    arc_size.loc["Roermond", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Roermond"] = 2

    arc_size.loc["Roermond", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Roermond"] = 2

    arc_size.loc["Roermond", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Roermond"] = 2

    arc_size.loc["Dongen", "Oosterhout"] = 2
    arc_size.loc["Oosterhout", "Dongen"] = 2

    arc_size.loc["Dongen", "Maastricht"] = 2
    arc_size.loc["Maastricht", "Dongen"] = 2

    arc_size.loc["Dongen", "Venlo"] = 2
    arc_size.loc["Venlo", "Dongen"] = 2

    arc_size.loc["Dongen", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Dongen"] = 2

    arc_size.loc["Dongen", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Dongen"] = 2

    arc_size.loc["Dongen", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Dongen"] = 2

    arc_size.loc["Dongen", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Dongen"] = 2

    arc_size.loc["Dongen", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Dongen"] = 2

    arc_size.loc["Dongen", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Dongen"] = 2

    arc_size.loc["Oosterhout", "Maastricht"] = 2
    arc_size.loc["Maastricht", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Venlo"] = 2
    arc_size.loc["Venlo", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Oosterhout"] = 2

    arc_size.loc["Maastricht", "Venlo"] = 2
    arc_size.loc["Venlo", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Maastricht"] = 2

    arc_size.loc["Maastricht", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Maastricht"] = 2

    arc_size.loc["Venlo", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Venlo"] = 2

    arc_size.loc["Venlo", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Venlo"] = 2

    arc_size.loc["Venlo", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Venlo"] = 2

    arc_size.loc["Venlo", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Venlo"] = 2

    arc_size.loc["Venlo", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Venlo"] = 2

    arc_size.loc["Venlo", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Venlo"] = 2

    arc_size.loc["Arnhem", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Arnhem"] = 2

    arc_size.loc["Arnhem", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Arnhem"] = 2

    arc_size.loc["Arnhem", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Arnhem"] = 2

    arc_size.loc["Arnhem", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Arnhem"] = 2

    arc_size.loc["Arnhem", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Arnhem"] = 2

    arc_size.loc["East_Groningen", "Betuwe"] = 2
    arc_size.loc["Betuwe", "East_Groningen"] = 2

    arc_size.loc["East_Groningen", "Heerlen"] = 2
    arc_size.loc["Heerlen", "East_Groningen"] = 2

    arc_size.loc["East_Groningen", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "East_Groningen"] = 2

    arc_size.loc["East_Groningen", "Wageningen"] = 2
    arc_size.loc["Wageningen", "East_Groningen"] = 2

    arc_size.loc["Betuwe", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Betuwe"] = 2

    arc_size.loc["Betuwe", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Betuwe"] = 2

    arc_size.loc["Betuwe", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Betuwe"] = 2

    arc_size.loc["Heerlen", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Heerlen"] = 2

    arc_size.loc["Heerlen", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Heerlen"] = 2

    arc_size.loc["Dordrecht", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Dordrecht"] = 2

    # connections with big clusters Chemelot

    arc_size.loc["Heerlen", "Chemelot"] = 2
    arc_size.loc["Chemelot", "Heerlen"] = 2

    arc_size.loc["Roermond", "Chemelot"] = 2
    arc_size.loc["Chemelot", "Roermond"] = 2

    arc_size.loc["Roermond", "Chemelot"] = 2
    arc_size.loc["Chemelot", "Roermond"] = 2

    arc_size.loc["Venlo", "Chemelot"] = 2
    arc_size.loc["Chemelot", "Venlo"] = 2

    # connections with big clusters Zeeland

    arc_size.loc["Dongen", "Zeeland"] = 2
    arc_size.loc["Zeeland", "Dongen"] = 2

    arc_size.loc["Oosterhout", "Zeeland"] = 2
    arc_size.loc["Zeeland", "Oosterhout"] = 2

    arc_size.loc["Dordrecht", "Zeeland"] = 2
    arc_size.loc["Zeeland", "Dordrecht"] = 2

    arc_size.loc["Betuwe", "Zeeland"] = 2
    arc_size.loc["Zeeland", "Betuwe"] = 2

    # connections with big clusters Rotterdam

    arc_size.loc["Dongen", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Dongen"] = 2

    arc_size.loc["Oosterhout", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Oosterhout"] = 2

    arc_size.loc["Dordrecht", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Dordrecht"] = 2

    arc_size.loc["Betuwe", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Betuwe"] = 2

    arc_size.loc["Wageningen", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Wageningen"] = 2

    # connections with big clusters North_Sea

    arc_size.loc["Dongen", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Dongen"] = 2

    arc_size.loc["Oosterhout", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Oosterhout"] = 2

    arc_size.loc["Dordrecht", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Dordrecht"] = 2

    arc_size.loc["Betuwe", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Betuwe"] = 2

    arc_size.loc["Wageningen", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Wageningen"] = 2

    arc_size.loc["Arnhem", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Arnhem"] = 2

    # connections with big clusters North_Netherlands

    arc_size.loc["East_Groningen", "North_Netherlands"] = 2
    arc_size.loc["North_Netherlands", "East_Groningen"] = 2

    arc_size.loc["Betuwe", "North_Netherlands"] = 2
    arc_size.loc["North_Netherlands", "Betuwe"] = 2

    arc_size.loc["Wageningen", "North_Netherlands"] = 2
    arc_size.loc["North_Netherlands", "Wageningen"] = 2

    arc_size.loc["Arnhem", "North_Netherlands"] = 2
    arc_size.loc["North_Netherlands", "Arnhem"] = 2

    # connections with big clusters Zuidwending

    arc_size.loc["East_Groningen", "Zuidwending"] = 2
    arc_size.loc["Zuidwending", "East_Groningen"] = 2

    arc_size.loc["Betuwe", "Zuidwending"] = 2
    arc_size.loc["Zuidwending", "Betuwe"] = 2

    arc_size.loc["Wageningen", "Zuidwending"] = 2
    arc_size.loc["Zuidwending", "Wageningen"] = 2

    arc_size.loc["Arnhem", "Zuidwending"] = 2
    arc_size.loc["Zuidwending", "Arnhem"] = 2

    # </editor-fold>

    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP_big" / "size_max_arcs.csv",
        sep=";")
    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP_big" / "size_max_arcs.csv",
        sep=";")

    # <editor-fold desc="Arc size connections truck">
    # arc_size.loc["Roermond", "Dongen"] = 3
    # arc_size.loc["Dongen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Oosterhout"] = 3
    # arc_size.loc["Oosterhout", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Maastricht"] = 3
    # arc_size.loc["Maastricht", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Venlo"] = 3
    # arc_size.loc["Venlo", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Roermond"] = 3
    #
    # arc_size.loc["Dongen", "Oosterhout"] = 3
    # arc_size.loc["Oosterhout", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Maastricht"] = 3
    # arc_size.loc["Maastricht", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Venlo"] = 3
    # arc_size.loc["Venlo", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Dongen"] = 3
    #
    # arc_size.loc["Oosterhout", "Maastricht"] = 3
    # arc_size.loc["Maastricht", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Venlo"] = 3
    # arc_size.loc["Venlo", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Oosterhout"] = 3
    #
    # arc_size.loc["Maastricht", "Venlo"] = 3
    # arc_size.loc["Venlo", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Maastricht"] = 3
    #
    # arc_size.loc["Venlo", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Venlo"] = 3
    #
    # arc_size.loc["Arnhem", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Arnhem"] = 3
    #
    # arc_size.loc["East_Groningen", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "East_Groningen"] = 3
    #
    # arc_size.loc["Betuwe", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Betuwe"] = 3
    #
    # arc_size.loc["Heerlen", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Heerlen"] = 3
    #
    # arc_size.loc["Dordrecht", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Dordrecht"] = 3
    #
    # arc_size.loc["Wageningen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Wageningen"] = 3
    #
    # arc_size.loc["Rotterdam", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Rotterdam"] = 3
    #
    # arc_size.loc["Zeeland", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Zeeland"] = 3
    #
    # arc_size.loc["Zeeland", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Zeeland"] = 3
    #
    # arc_size.loc["Zeeland", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Zeeland"] = 3
    #
    # arc_size.loc["Zeeland", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Zeeland"] = 3
    #
    # arc_size.loc["North_Sea", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "North_Sea"] = 3
    #
    # arc_size.loc["North_Sea", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "North_Sea"] = 3
    #
    # arc_size.loc["North_Sea", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "North_Sea"] = 3
    #
    # arc_size.loc["North_Netherlands", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "North_Netherlands"] = 3
    #
    # arc_size.loc["North_Netherlands", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "North_Netherlands"] = 3
    #
    # arc_size.loc["Chemelot", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Chemelot"] = 3
    # # </editor-fold>
    #
    # arc_size.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "size_max_arcs.csv", sep=";")

    print("Max size per arc:", arc_size)

    # Delete the max_size_arc template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv")

    # # Connection
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)
    # <editor-fold desc="Connections">
    connection.loc["Roermond", "Dongen"] = 1
    connection.loc["Dongen", "Roermond"] = 1

    connection.loc["Roermond", "Oosterhout"] = 1
    connection.loc["Oosterhout", "Roermond"] = 1

    connection.loc["Roermond", "Maastricht"] = 1
    connection.loc["Maastricht", "Roermond"] = 1

    connection.loc["Roermond", "Venlo"] = 1
    connection.loc["Venlo", "Roermond"] = 1

    connection.loc["Roermond", "Arnhem"] = 1
    connection.loc["Arnhem", "Roermond"] = 1

    connection.loc["Roermond", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Roermond"] = 1

    connection.loc["Roermond", "Betuwe"] = 1
    connection.loc["Betuwe", "Roermond"] = 1

    connection.loc["Roermond", "Heerlen"] = 1
    connection.loc["Heerlen", "Roermond"] = 1

    connection.loc["Roermond", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Roermond"] = 1

    connection.loc["Roermond", "Wageningen"] = 1
    connection.loc["Wageningen", "Roermond"] = 1

    connection.loc["Dongen", "Oosterhout"] = 1
    connection.loc["Oosterhout", "Dongen"] = 1

    connection.loc["Dongen", "Maastricht"] = 1
    connection.loc["Maastricht", "Dongen"] = 1

    connection.loc["Dongen", "Venlo"] = 1
    connection.loc["Venlo", "Dongen"] = 1

    connection.loc["Dongen", "Arnhem"] = 1
    connection.loc["Arnhem", "Dongen"] = 1

    connection.loc["Dongen", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Dongen"] = 1

    connection.loc["Dongen", "Betuwe"] = 1
    connection.loc["Betuwe", "Dongen"] = 1

    connection.loc["Dongen", "Heerlen"] = 1
    connection.loc["Heerlen", "Dongen"] = 1

    connection.loc["Dongen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Dongen"] = 1

    connection.loc["Dongen", "Wageningen"] = 1
    connection.loc["Wageningen", "Dongen"] = 1

    connection.loc["Oosterhout", "Maastricht"] = 1
    connection.loc["Maastricht", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Venlo"] = 1
    connection.loc["Venlo", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Arnhem"] = 1
    connection.loc["Arnhem", "Oosterhout"] = 1

    connection.loc["Oosterhout", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Betuwe"] = 1
    connection.loc["Betuwe", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Heerlen"] = 1
    connection.loc["Heerlen", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Wageningen"] = 1
    connection.loc["Wageningen", "Oosterhout"] = 1

    connection.loc["Maastricht", "Venlo"] = 1
    connection.loc["Venlo", "Maastricht"] = 1

    connection.loc["Maastricht", "Arnhem"] = 1
    connection.loc["Arnhem", "Maastricht"] = 1

    connection.loc["Maastricht", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Maastricht"] = 1

    connection.loc["Maastricht", "Betuwe"] = 1
    connection.loc["Betuwe", "Maastricht"] = 1

    connection.loc["Maastricht", "Heerlen"] = 1
    connection.loc["Heerlen", "Maastricht"] = 1

    connection.loc["Maastricht", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Maastricht"] = 1

    connection.loc["Maastricht", "Wageningen"] = 1
    connection.loc["Wageningen", "Maastricht"] = 1

    connection.loc["Venlo", "Arnhem"] = 1
    connection.loc["Arnhem", "Venlo"] = 1

    connection.loc["Venlo", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Venlo"] = 1

    connection.loc["Venlo", "Betuwe"] = 1
    connection.loc["Betuwe", "Venlo"] = 1

    connection.loc["Venlo", "Heerlen"] = 1
    connection.loc["Heerlen", "Venlo"] = 1

    connection.loc["Venlo", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Venlo"] = 1

    connection.loc["Venlo", "Wageningen"] = 1
    connection.loc["Wageningen", "Venlo"] = 1

    connection.loc["Arnhem", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Arnhem"] = 1

    connection.loc["Arnhem", "Betuwe"] = 1
    connection.loc["Betuwe", "Arnhem"] = 1

    connection.loc["Arnhem", "Heerlen"] = 1
    connection.loc["Heerlen", "Arnhem"] = 1

    connection.loc["Arnhem", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Arnhem"] = 1

    connection.loc["Arnhem", "Wageningen"] = 1
    connection.loc["Wageningen", "Arnhem"] = 1

    connection.loc["East_Groningen", "Betuwe"] = 1
    connection.loc["Betuwe", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Heerlen"] = 1
    connection.loc["Heerlen", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Wageningen"] = 1
    connection.loc["Wageningen", "East_Groningen"] = 1

    connection.loc["Betuwe", "Heerlen"] = 1
    connection.loc["Heerlen", "Betuwe"] = 1

    connection.loc["Betuwe", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Betuwe"] = 1

    connection.loc["Betuwe", "Wageningen"] = 1
    connection.loc["Wageningen", "Betuwe"] = 1

    connection.loc["Heerlen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Heerlen"] = 1

    connection.loc["Heerlen", "Wageningen"] = 1
    connection.loc["Wageningen", "Heerlen"] = 1

    connection.loc["Dordrecht", "Wageningen"] = 1
    connection.loc["Wageningen", "Dordrecht"] = 1

    # connections with big clusters Chemelot

    connection.loc["Heerlen", "Chemelot"] = 1
    connection.loc["Chemelot", "Heerlen"] = 1

    connection.loc["Roermond", "Chemelot"] = 1
    connection.loc["Chemelot", "Roermond"] = 1

    connection.loc["Venlo", "Chemelot"] = 1
    connection.loc["Chemelot", "Venlo"] = 1

    # connections with big clusters Zeeland

    connection.loc["Dongen", "Zeeland"] = 1
    connection.loc["Zeeland", "Dongen"] = 1

    connection.loc["Oosterhout", "Zeeland"] = 1
    connection.loc["Zeeland", "Oosterhout"] = 1

    connection.loc["Dordrecht", "Zeeland"] = 1
    connection.loc["Zeeland", "Dordrecht"] = 1

    connection.loc["Betuwe", "Zeeland"] = 1
    connection.loc["Zeeland", "Betuwe"] = 1

    # connections with big clusters Rotterdam

    connection.loc["Dongen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Dongen"] = 1

    connection.loc["Oosterhout", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Oosterhout"] = 1

    connection.loc["Dordrecht", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Dordrecht"] = 1

    connection.loc["Betuwe", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Betuwe"] = 1

    connection.loc["Wageningen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Wageningen"] = 1

    # connections with big clusters North_Sea

    connection.loc["Dongen", "North_Sea"] = 1
    connection.loc["North_Sea", "Dongen"] = 1

    connection.loc["Oosterhout", "North_Sea"] = 1
    connection.loc["North_Sea", "Oosterhout"] = 1

    connection.loc["Dordrecht", "North_Sea"] = 1
    connection.loc["North_Sea", "Dordrecht"] = 1

    connection.loc["Betuwe", "North_Sea"] = 1
    connection.loc["North_Sea", "Betuwe"] = 1

    connection.loc["Wageningen", "North_Sea"] = 1
    connection.loc["North_Sea", "Wageningen"] = 1

    connection.loc["Arnhem", "North_Sea"] = 1
    connection.loc["North_Sea", "Arnhem"] = 1

    # connections with big clusters North_Netherlands

    connection.loc["East_Groningen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "East_Groningen"] = 1

    connection.loc["Betuwe", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Betuwe"] = 1

    connection.loc["Wageningen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Wageningen"] = 1

    connection.loc["Arnhem", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Arnhem"] = 1

    # connections with big clusters Zuidwending

    connection.loc["East_Groningen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "East_Groningen"] = 1

    connection.loc["Betuwe", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Betuwe"] = 1

    connection.loc["Wageningen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Wageningen"] = 1

    connection.loc["Arnhem", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Arnhem"] = 1

    # </editor-fold>

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP_big" / "connection.csv",
        sep=";")
    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP_big" / "connection.csv",
        sep=";")
    #connection.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "connection.csv", sep=";")

    print("Connection:", connection)

    # Delete the template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "connection.csv")

    # # Distance
    distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "distance.csv", sep=";",
                           index_col=0)
    # <editor-fold desc="Distances">

    distance.loc["Roermond", "Dongen"] = 87
    distance.loc["Dongen", "Roermond"] = 87

    distance.loc["Roermond", "Oosterhout"] = 93
    distance.loc["Oosterhout", "Roermond"] = 93

    distance.loc["Roermond", "Maastricht"] = 43
    distance.loc["Maastricht", "Roermond"] = 43

    distance.loc["Roermond", "Venlo"] = 23
    distance.loc["Venlo", "Roermond"] = 23

    distance.loc["Roermond", "Arnhem"] = 88
    distance.loc["Arnhem", "Roermond"] = 88

    distance.loc["Roermond", "East_Groningen"] = 226
    distance.loc["East_Groningen", "Roermond"] = 226

    distance.loc["Roermond", "Betuwe"] = 87
    distance.loc["Betuwe", "Roermond"] = 87

    distance.loc["Roermond", "Heerlen"] = 35
    distance.loc["Heerlen", "Roermond"] = 35

    distance.loc["Roermond", "Dordrecht"] = 113
    distance.loc["Dordrecht", "Roermond"] = 113

    distance.loc["Roermond", "Wageningen"] = 89
    distance.loc["Wageningen", "Roermond"] = 89

    distance.loc["Roermond", "Rotterdam"] = 132
    distance.loc["Rotterdam", "Roermond"] = 132

    distance.loc["Roermond", "Zeeland"] = 152
    distance.loc["Zeeland", "Roermond"] = 152

    distance.loc["Roermond", "North_Sea"] = 160
    distance.loc["North_Sea", "Roermond"] = 160

    distance.loc["Roermond", "North_Netherlands"] = 258
    distance.loc["North_Netherlands", "Roermond"] = 258

    distance.loc["Roermond", "Chemelot"] = 28
    distance.loc["Chemelot", "Roermond"] = 28

    distance.loc["Roermond", "Zuidwending"] = 221
    distance.loc["Zuidwending", "Roermond"] = 221

    distance.loc["Dongen", "Oosterhout"] = 6
    distance.loc["Oosterhout", "Dongen"] = 6

    distance.loc["Dongen", "Maastricht"] = 100
    distance.loc["Maastricht", "Dongen"] = 100

    distance.loc["Dongen", "Venlo"] = 89
    distance.loc["Venlo", "Dongen"] = 89

    distance.loc["Dongen", "Arnhem"] = 77
    distance.loc["Arnhem", "Dongen"] = 77

    distance.loc["Dongen", "East_Groningen"] = 210
    distance.loc["East_Groningen", "Dongen"] = 210

    distance.loc["Dongen", "Betuwe"] = 50
    distance.loc["Betuwe", "Dongen"] = 50

    distance.loc["Dongen", "Heerlen"] = 109
    distance.loc["Heerlen", "Dongen"] = 109

    distance.loc["Dongen", "Dordrecht"] = 26
    distance.loc["Dordrecht", "Dongen"] = 26

    distance.loc["Dongen", "Wageningen"] = 63
    distance.loc["Wageningen", "Dongen"] = 63

    distance.loc["Dongen", "Rotterdam"] = 46
    distance.loc["Rotterdam", "Dongen"] = 46

    distance.loc["Dongen", "Zeeland"] = 77
    distance.loc["Zeeland", "Dongen"] = 77

    distance.loc["Dongen", "North_Sea"] = 90
    distance.loc["North_Sea", "Dongen"] = 90

    distance.loc["Dongen", "North_Netherlands"] = 240
    distance.loc["North_Netherlands", "Dongen"] = 240

    distance.loc["Dongen", "Chemelot"] = 93
    distance.loc["Chemelot", "Dongen"] = 93

    distance.loc["Dongen", "Zuidwending"] = 212
    distance.loc["Zuidwending", "Dongen"] = 212

    distance.loc["Oosterhout", "Maastricht"] = 105
    distance.loc["Maastricht", "Oosterhout"] = 105

    distance.loc["Oosterhout", "Venlo"] = 95
    distance.loc["Venlo", "Oosterhout"] = 95

    distance.loc["Oosterhout", "Arnhem"] = 81
    distance.loc["Arnhem", "Oosterhout"] = 81

    distance.loc["Oosterhout", "East_Groningen"] = 212
    distance.loc["East_Groningen", "Oosterhout"] = 212

    distance.loc["Oosterhout", "Betuwe"] = 54
    distance.loc["Betuwe", "Oosterhout"] = 54

    distance.loc["Oosterhout", "Heerlen"] = 115
    distance.loc["Heerlen", "Oosterhout"] = 115

    distance.loc["Oosterhout", "Dordrecht"] = 21
    distance.loc["Dordrecht", "Oosterhout"] = 21

    distance.loc["Oosterhout", "Wageningen"] = 66
    distance.loc["Wageningen", "Oosterhout"] = 66

    distance.loc["Oosterhout", "Rotterdam"] = 41
    distance.loc["Rotterdam", "Oosterhout"] = 41

    distance.loc["Oosterhout", "Zeeland"] = 72
    distance.loc["Zeeland", "Oosterhout"] = 72

    distance.loc["Oosterhout", "North_Sea"] = 88
    distance.loc["North_Sea", "Oosterhout"] = 88

    distance.loc["Oosterhout", "North_Netherlands"] = 241
    distance.loc["North_Netherlands", "Oosterhout"] = 241

    distance.loc["Oosterhout", "Chemelot"] = 99
    distance.loc["Chemelot", "Oosterhout"] = 99

    distance.loc["Oosterhout", "Zuidwending"] = 214
    distance.loc["Zuidwending", "Oosterhout"] = 214

    distance.loc["Maastricht", "Venlo"] = 66
    distance.loc["Venlo", "Maastricht"] = 66

    distance.loc["Maastricht", "Arnhem"] = 127
    distance.loc["Arnhem", "Maastricht"] = 127

    distance.loc["Maastricht", "East_Groningen"] = 268
    distance.loc["East_Groningen", "Maastricht"] = 268

    distance.loc["Maastricht", "Betuwe"] = 119
    distance.loc["Betuwe", "Maastricht"] = 119

    distance.loc["Maastricht", "Heerlen"] = 21
    distance.loc["Heerlen", "Maastricht"] = 21

    distance.loc["Maastricht", "Dordrecht"] = 126
    distance.loc["Dordrecht", "Maastricht"] = 126

    distance.loc["Maastricht", "Wageningen"] = 124
    distance.loc["Wageningen", "Maastricht"] = 124

    distance.loc["Maastricht", "Rotterdam"] = 146
    distance.loc["Rotterdam", "Maastricht"] = 146

    distance.loc["Maastricht", "Zeeland"] = 147
    distance.loc["Zeeland", "Maastricht"] = 147

    distance.loc["Maastricht", "North_Sea"] = 186
    distance.loc["North_Sea", "Maastricht"] = 186

    distance.loc["Maastricht", "North_Netherlands"] = 300
    distance.loc["North_Netherlands", "Maastricht"] = 300

    distance.loc["Maastricht", "Chemelot"] = 16
    distance.loc["Chemelot", "Maastricht"] = 16

    distance.loc["Maastricht", "Zuidwending"] = 264
    distance.loc["Zuidwending", "Maastricht"] = 264

    distance.loc["Venlo", "Arnhem"] = 71
    distance.loc["Arnhem", "Venlo"] = 71

    distance.loc["Venlo", "East_Groningen"] = 204
    distance.loc["East_Groningen", "Venlo"] = 204

    distance.loc["Venlo", "Betuwe"] = 77
    distance.loc["Betuwe", "Venlo"] = 77

    distance.loc["Venlo", "Heerlen"] = 55
    distance.loc["Heerlen", "Venlo"] = 55

    distance.loc["Venlo", "Dordrecht"] = 113
    distance.loc["Dordrecht", "Venlo"] = 113

    distance.loc["Venlo", "Wageningen"] = 75
    distance.loc["Wageningen", "Venlo"] = 75

    distance.loc["Venlo", "Rotterdam"] = 132
    distance.loc["Rotterdam", "Venlo"] = 132

    distance.loc["Venlo", "Zeeland"] = 161
    distance.loc["Zeeland", "Venlo"] = 161

    distance.loc["Venlo", "North_Sea"] = 150
    distance.loc["North_Sea", "Venlo"] = 150

    distance.loc["Venlo", "North_Netherlands"] = 236
    distance.loc["North_Netherlands", "Venlo"] = 236

    distance.loc["Venlo", "Chemelot"] = 50
    distance.loc["Chemelot", "Venlo"] = 50

    distance.loc["Venlo", "Zuidwending"] = 199
    distance.loc["Zuidwending", "Venlo"] = 199

    distance.loc["Arnhem", "East_Groningen"] = 143
    distance.loc["East_Groningen", "Arnhem"] = 143

    distance.loc["Arnhem", "Betuwe"] = 28
    distance.loc["Betuwe", "Arnhem"] = 28

    distance.loc["Arnhem", "Heerlen"] = 123
    distance.loc["Heerlen", "Arnhem"] = 123

    distance.loc["Arnhem", "Dordrecht"] = 86
    distance.loc["Dordrecht", "Arnhem"] = 86

    distance.loc["Arnhem", "Wageningen"] = 16
    distance.loc["Wageningen", "Arnhem"] = 16

    distance.loc["Arnhem", "Rotterdam"] = 98
    distance.loc["Rotterdam", "Arnhem"] = 98

    distance.loc["Arnhem", "Zeeland"] = 151
    distance.loc["Zeeland", "Arnhem"] = 151

    distance.loc["Arnhem", "North_Sea"] = 89
    distance.loc["North_Sea", "Arnhem"] = 89

    distance.loc["Arnhem", "North_Netherlands"] = 175
    distance.loc["North_Netherlands", "Arnhem"] = 175

    distance.loc["Arnhem", "Chemelot"] = 112
    distance.loc["Chemelot", "Arnhem"] = 112

    distance.loc["Arnhem", "Zuidwending"] = 142
    distance.loc["Zuidwending", "Arnhem"] = 142

    distance.loc["East_Groningen", "Betuwe"] = 162
    distance.loc["Betuwe", "East_Groningen"] = 162

    distance.loc["East_Groningen", "Heerlen"] = 260
    distance.loc["Heerlen", "East_Groningen"] = 260

    distance.loc["East_Groningen", "Dordrecht"] = 207
    distance.loc["Dordrecht", "East_Groningen"] = 207

    distance.loc["East_Groningen", "Wageningen"] = 152
    distance.loc["Wageningen", "East_Groningen"] = 152

    distance.loc["East_Groningen", "Rotterdam"] = 206
    distance.loc["Rotterdam", "East_Groningen"] = 206

    distance.loc["East_Groningen", "Zeeland"] = 270
    distance.loc["Zeeland", "East_Groningen"] = 270

    distance.loc["East_Groningen", "North_Sea"] = 152
    distance.loc["North_Sea", "East_Groningen"] = 152

    distance.loc["East_Groningen", "North_Netherlands"] = 32
    distance.loc["North_Netherlands", "East_Groningen"] = 32

    distance.loc["East_Groningen", "Chemelot"] = 252
    distance.loc["Chemelot", "East_Groningen"] = 252

    distance.loc["East_Groningen", "Zuidwending"] = 16
    distance.loc["Zuidwending", "East_Groningen"] = 16

    distance.loc["Betuwe", "Heerlen"] = 120
    distance.loc["Heerlen", "Betuwe"] = 120

    distance.loc["Betuwe", "Dordrecht"] = 58
    distance.loc["Dordrecht", "Betuwe"] = 58

    distance.loc["Betuwe", "Wageningen"] = 13
    distance.loc["Wageningen", "Betuwe"] = 13

    distance.loc["Betuwe", "Rotterdam"] = 70
    distance.loc["Rotterdam", "Betuwe"] = 70

    distance.loc["Betuwe", "Zeeland"] = 123
    distance.loc["Zeeland", "Betuwe"] = 123

    distance.loc["Betuwe", "North_Sea"] = 74
    distance.loc["North_Sea", "Betuwe"] = 74

    distance.loc["Betuwe", "North_Netherlands"] = 193
    distance.loc["North_Netherlands", "Betuwe"] = 193

    distance.loc["Betuwe", "Chemelot"] = 107
    distance.loc["Chemelot", "Betuwe"] = 107

    distance.loc["Betuwe", "Zuidwending"] = 163
    distance.loc["Zuidwending", "Betuwe"] = 163

    distance.loc["Heerlen", "Dordrecht"] = 136
    distance.loc["Dordrecht", "Heerlen"] = 136

    distance.loc["Heerlen", "Wageningen"] = 123
    distance.loc["Wageningen", "Heerlen"] = 123

    distance.loc["Heerlen", "Rotterdam"] = 156
    distance.loc["Rotterdam", "Heerlen"] = 156

    distance.loc["Heerlen", "Zeeland"] = 164
    distance.loc["Zeeland", "Heerlen"] = 164

    distance.loc["Heerlen", "North_Sea"] = 190
    distance.loc["North_Sea", "Heerlen"] = 190

    distance.loc["Heerlen", "North_Netherlands"] = 292
    distance.loc["North_Netherlands", "Heerlen"] = 292

    distance.loc["Heerlen", "Chemelot"] = 16
    distance.loc["Chemelot", "Heerlen"] = 16

    distance.loc["Heerlen", "Zuidwending"] = 254
    distance.loc["Zuidwending", "Heerlen"] = 254

    distance.loc["Dordrecht", "Wageningen"] = 70
    distance.loc["Wageningen", "Dordrecht"] = 70

    distance.loc["Dordrecht", "Rotterdam"] = 20
    distance.loc["Rotterdam", "Dordrecht"] = 20

    distance.loc["Dordrecht", "Zeeland"] = 66
    distance.loc["Zeeland", "Dordrecht"] = 66

    distance.loc["Dordrecht", "North_Sea"] = 71
    distance.loc["North_Sea", "Dordrecht"] = 71

    distance.loc["Dordrecht", "North_Netherlands"] = 235
    distance.loc["North_Netherlands", "Dordrecht"] = 235

    distance.loc["Dordrecht", "Chemelot"] = 120
    distance.loc["Chemelot", "Dordrecht"] = 120

    distance.loc["Dordrecht", "Zuidwending"] = 210
    distance.loc["Zuidwending", "Dordrecht"] = 210

    distance.loc["Wageningen", "Rotterdam"] = 82
    distance.loc["Rotterdam", "Wageningen"] = 82

    distance.loc["Wageningen", "Zeeland"] = 136
    distance.loc["Zeeland", "Wageningen"] = 136

    distance.loc["Wageningen", "North_Sea"] = 77
    distance.loc["North_Sea", "Wageningen"] = 77

    distance.loc["Wageningen", "North_Netherlands"] = 183
    distance.loc["North_Netherlands", "Wageningen"] = 183

    distance.loc["Wageningen", "Chemelot"] = 111
    distance.loc["Chemelot", "Wageningen"] = 111

    distance.loc["Wageningen", "Zuidwending"] = 152
    distance.loc["Zuidwending", "Wageningen"] = 152

    distance.loc["Rotterdam", "Zeeland"] = 64
    distance.loc["Zeeland", "Rotterdam"] = 64

    distance.loc["Rotterdam", "North_Sea"] = 61
    distance.loc["North_Sea", "Rotterdam"] = 61

    distance.loc["Rotterdam", "North_Netherlands"] = 232
    distance.loc["North_Netherlands", "Rotterdam"] = 232

    distance.loc["Rotterdam", "Chemelot"] = 140
    distance.loc["Chemelot", "Rotterdam"] = 140

    distance.loc["Rotterdam", "Zuidwending"] = 211
    distance.loc["Zuidwending", "Rotterdam"] = 211

    distance.loc["Zeeland", "North_Sea"] = 123
    distance.loc["North_Sea", "Zeeland"] = 123

    distance.loc["Zeeland", "North_Netherlands"] = 296
    distance.loc["North_Netherlands", "Zeeland"] = 296

    distance.loc["Zeeland", "Chemelot"] = 148
    distance.loc["Chemelot", "Zeeland"] = 148

    distance.loc["Zeeland", "Zuidwending"] = 275
    distance.loc["Zuidwending", "Zeeland"] = 275

    distance.loc["North_Sea", "North_Netherlands"] = 176
    distance.loc["North_Netherlands", "North_Sea"] = 176

    distance.loc["North_Sea", "Chemelot"] = 176
    distance.loc["Chemelot", "North_Sea"] = 176

    distance.loc["North_Sea", "Zuidwending"] = 160
    distance.loc["Zuidwending", "North_Sea"] = 160

    distance.loc["North_Netherlands", "Chemelot"] = 284
    distance.loc["Chemelot", "North_Netherlands"] = 284

    distance.loc["North_Netherlands", "Zuidwending"] = 41
    distance.loc["Zuidwending", "North_Netherlands"] = 41

    distance.loc["Chemelot", "Zuidwending"] = 248
    distance.loc["Zuidwending", "Chemelot"] = 248

    # </editor-fold>

    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP_big" / "distance.csv",
        sep=";")
    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP_big" / "distance.csv",
        sep=";")
    #distance.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "distance.csv", sep=";")
    print("Distance:", distance)

    # Delete the template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "distance.csv")

def add_new_network_H2_small_and_EHB(input_data_path):
    # # Make a new folder for the new network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP",
                exist_ok=True)
    #os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP", exist_ok=True)

    print("New network")

    # # max size arc
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)
    # <editor-fold desc="Arc size connections pipeline low P">
    arc_size.loc["Roermond", "Dongen"] = 2
    arc_size.loc["Dongen", "Roermond"] = 2

    arc_size.loc["Roermond", "Oosterhout"] = 2
    arc_size.loc["Oosterhout", "Roermond"] = 2

    arc_size.loc["Roermond", "Maastricht"] = 2
    arc_size.loc["Maastricht", "Roermond"] = 2

    arc_size.loc["Roermond", "Venlo"] = 2
    arc_size.loc["Venlo", "Roermond"] = 2

    arc_size.loc["Roermond", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Roermond"] = 2

    arc_size.loc["Roermond", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Roermond"] = 2

    arc_size.loc["Roermond", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Roermond"] = 2

    arc_size.loc["Roermond", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Roermond"] = 2

    arc_size.loc["Roermond", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Roermond"] = 2

    arc_size.loc["Roermond", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Roermond"] = 2

    arc_size.loc["Dongen", "Oosterhout"] = 2
    arc_size.loc["Oosterhout", "Dongen"] = 2

    arc_size.loc["Dongen", "Maastricht"] = 2
    arc_size.loc["Maastricht", "Dongen"] = 2

    arc_size.loc["Dongen", "Venlo"] = 2
    arc_size.loc["Venlo", "Dongen"] = 2

    arc_size.loc["Dongen", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Dongen"] = 2

    arc_size.loc["Dongen", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Dongen"] = 2

    arc_size.loc["Dongen", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Dongen"] = 2

    arc_size.loc["Dongen", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Dongen"] = 2

    arc_size.loc["Dongen", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Dongen"] = 2

    arc_size.loc["Dongen", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Dongen"] = 2

    arc_size.loc["Oosterhout", "Maastricht"] = 2
    arc_size.loc["Maastricht", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Venlo"] = 2
    arc_size.loc["Venlo", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Oosterhout"] = 2

    arc_size.loc["Maastricht", "Venlo"] = 2
    arc_size.loc["Venlo", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Maastricht"] = 2

    arc_size.loc["Maastricht", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Maastricht"] = 2

    arc_size.loc["Venlo", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Venlo"] = 2

    arc_size.loc["Venlo", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Venlo"] = 2

    arc_size.loc["Venlo", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Venlo"] = 2

    arc_size.loc["Venlo", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Venlo"] = 2

    arc_size.loc["Venlo", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Venlo"] = 2

    arc_size.loc["Venlo", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Venlo"] = 2

    arc_size.loc["Arnhem", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Arnhem"] = 2

    arc_size.loc["Arnhem", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Arnhem"] = 2

    arc_size.loc["Arnhem", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Arnhem"] = 2

    arc_size.loc["Arnhem", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Arnhem"] = 2

    arc_size.loc["Arnhem", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Arnhem"] = 2

    arc_size.loc["East_Groningen", "Betuwe"] = 2
    arc_size.loc["Betuwe", "East_Groningen"] = 2

    arc_size.loc["East_Groningen", "Heerlen"] = 2
    arc_size.loc["Heerlen", "East_Groningen"] = 2

    arc_size.loc["East_Groningen", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "East_Groningen"] = 2

    arc_size.loc["East_Groningen", "Wageningen"] = 2
    arc_size.loc["Wageningen", "East_Groningen"] = 2

    arc_size.loc["Betuwe", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Betuwe"] = 2

    arc_size.loc["Betuwe", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Betuwe"] = 2

    arc_size.loc["Betuwe", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Betuwe"] = 2

    arc_size.loc["Heerlen", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Heerlen"] = 2

    arc_size.loc["Heerlen", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Heerlen"] = 2

    arc_size.loc["Dordrecht", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Dordrecht"] = 2

    # connections with big clusters Chemelot

    arc_size.loc["Heerlen", "Chemelot"] = 2
    arc_size.loc["Chemelot", "Heerlen"] = 2

    arc_size.loc["Roermond", "Chemelot"] = 2
    arc_size.loc["Chemelot", "Roermond"] = 2

    arc_size.loc["Roermond", "Chemelot"] = 2
    arc_size.loc["Chemelot", "Roermond"] = 2

    arc_size.loc["Venlo", "Chemelot"] = 2
    arc_size.loc["Chemelot", "Venlo"] = 2

    # connections with big clusters Zeeland

    arc_size.loc["Dongen", "Zeeland"] = 2
    arc_size.loc["Zeeland", "Dongen"] = 2

    arc_size.loc["Oosterhout", "Zeeland"] = 2
    arc_size.loc["Zeeland", "Oosterhout"] = 2

    arc_size.loc["Dordrecht", "Zeeland"] = 2
    arc_size.loc["Zeeland", "Dordrecht"] = 2

    arc_size.loc["Betuwe", "Zeeland"] = 2
    arc_size.loc["Zeeland", "Betuwe"] = 2

    # connections with big clusters Rotterdam

    arc_size.loc["Dongen", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Dongen"] = 2

    arc_size.loc["Oosterhout", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Oosterhout"] = 2

    arc_size.loc["Dordrecht", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Dordrecht"] = 2

    arc_size.loc["Betuwe", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Betuwe"] = 2

    arc_size.loc["Wageningen", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Wageningen"] = 2

    # connections with big clusters North_Sea

    arc_size.loc["Dongen", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Dongen"] = 2

    arc_size.loc["Oosterhout", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Oosterhout"] = 2

    arc_size.loc["Dordrecht", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Dordrecht"] = 2

    arc_size.loc["Betuwe", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Betuwe"] = 2

    arc_size.loc["Wageningen", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Wageningen"] = 2

    arc_size.loc["Arnhem", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Arnhem"] = 2

    # connections with big clusters North_Netherlands

    arc_size.loc["East_Groningen", "North_Netherlands"] = 2
    arc_size.loc["North_Netherlands", "East_Groningen"] = 2

    arc_size.loc["Betuwe", "North_Netherlands"] = 2
    arc_size.loc["North_Netherlands", "Betuwe"] = 2

    arc_size.loc["Wageningen", "North_Netherlands"] = 2
    arc_size.loc["North_Netherlands", "Wageningen"] = 2

    arc_size.loc["Arnhem", "North_Netherlands"] = 2
    arc_size.loc["North_Netherlands", "Arnhem"] = 2

    # connections with big clusters Zuidwending

    arc_size.loc["East_Groningen", "Zuidwending"] = 2
    arc_size.loc["Zuidwending", "East_Groningen"] = 2

    arc_size.loc["Betuwe", "Zuidwending"] = 2
    arc_size.loc["Zuidwending", "Betuwe"] = 2

    arc_size.loc["Wageningen", "Zuidwending"] = 2
    arc_size.loc["Zuidwending", "Wageningen"] = 2

    arc_size.loc["Arnhem", "Zuidwending"] = 2
    arc_size.loc["Zuidwending", "Arnhem"] = 2

    # </editor-fold>

    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "size_max_arcs.csv",
        sep=";")

    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP",
                exist_ok=True)

    # <editor-fold desc="Arc size connections pipeline high P">

    arc_size.loc["Rotterdam", "Zeeland"] = 1
    arc_size.loc["Zeeland", "Rotterdam"] = 1
    arc_size.loc["Rotterdam", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Rotterdam"] = 1
    arc_size.loc["North_Sea", "North_Netherlands"] = 1
    arc_size.loc["North_Netherlands", "North_Sea"] = 1
    arc_size.loc["Zuidwending", "North_Netherlands"] = 1
    arc_size.loc["North_Netherlands", "Zuidwending"] = 1
    arc_size.loc["Zuidwending", "Chemelot"] = 1
    arc_size.loc["Chemelot", "Zuidwending"] = 1
    arc_size.loc["Zeeland", "Chemelot"] = 1
    arc_size.loc["Chemelot", "Zeeland"] = 1

    arc_size.loc["Roermond", "Dongen"] = 2
    arc_size.loc["Dongen", "Roermond"] = 2

    arc_size.loc["Roermond", "Oosterhout"] = 2
    arc_size.loc["Oosterhout", "Roermond"] = 2

    arc_size.loc["Roermond", "Maastricht"] = 2
    arc_size.loc["Maastricht", "Roermond"] = 2

    arc_size.loc["Roermond", "Venlo"] = 2
    arc_size.loc["Venlo", "Roermond"] = 2

    arc_size.loc["Roermond", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Roermond"] = 2

    arc_size.loc["Roermond", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Roermond"] = 2

    arc_size.loc["Roermond", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Roermond"] = 2

    arc_size.loc["Roermond", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Roermond"] = 2

    arc_size.loc["Roermond", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Roermond"] = 2

    arc_size.loc["Roermond", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Roermond"] = 2

    arc_size.loc["Dongen", "Oosterhout"] = 2
    arc_size.loc["Oosterhout", "Dongen"] = 2

    arc_size.loc["Dongen", "Maastricht"] = 2
    arc_size.loc["Maastricht", "Dongen"] = 2

    arc_size.loc["Dongen", "Venlo"] = 2
    arc_size.loc["Venlo", "Dongen"] = 2

    arc_size.loc["Dongen", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Dongen"] = 2

    arc_size.loc["Dongen", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Dongen"] = 2

    arc_size.loc["Dongen", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Dongen"] = 2

    arc_size.loc["Dongen", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Dongen"] = 2

    arc_size.loc["Dongen", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Dongen"] = 2

    arc_size.loc["Dongen", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Dongen"] = 2

    arc_size.loc["Oosterhout", "Maastricht"] = 2
    arc_size.loc["Maastricht", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Venlo"] = 2
    arc_size.loc["Venlo", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Oosterhout"] = 2

    arc_size.loc["Oosterhout", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Oosterhout"] = 2

    arc_size.loc["Maastricht", "Venlo"] = 2
    arc_size.loc["Venlo", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Maastricht"] = 2

    arc_size.loc["Maastricht", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Maastricht"] = 2

    arc_size.loc["Maastricht", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Maastricht"] = 2

    arc_size.loc["Venlo", "Arnhem"] = 2
    arc_size.loc["Arnhem", "Venlo"] = 2

    arc_size.loc["Venlo", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Venlo"] = 2

    arc_size.loc["Venlo", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Venlo"] = 2

    arc_size.loc["Venlo", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Venlo"] = 2

    arc_size.loc["Venlo", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Venlo"] = 2

    arc_size.loc["Venlo", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Venlo"] = 2

    arc_size.loc["Arnhem", "East_Groningen"] = 2
    arc_size.loc["East_Groningen", "Arnhem"] = 2

    arc_size.loc["Arnhem", "Betuwe"] = 2
    arc_size.loc["Betuwe", "Arnhem"] = 2

    arc_size.loc["Arnhem", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Arnhem"] = 2

    arc_size.loc["Arnhem", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Arnhem"] = 2

    arc_size.loc["Arnhem", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Arnhem"] = 2

    arc_size.loc["East_Groningen", "Betuwe"] = 2
    arc_size.loc["Betuwe", "East_Groningen"] = 2

    arc_size.loc["East_Groningen", "Heerlen"] = 2
    arc_size.loc["Heerlen", "East_Groningen"] = 2

    arc_size.loc["East_Groningen", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "East_Groningen"] = 2

    arc_size.loc["East_Groningen", "Wageningen"] = 2
    arc_size.loc["Wageningen", "East_Groningen"] = 2

    arc_size.loc["Betuwe", "Heerlen"] = 2
    arc_size.loc["Heerlen", "Betuwe"] = 2

    arc_size.loc["Betuwe", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Betuwe"] = 2

    arc_size.loc["Betuwe", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Betuwe"] = 2

    arc_size.loc["Heerlen", "Dordrecht"] = 2
    arc_size.loc["Dordrecht", "Heerlen"] = 2

    arc_size.loc["Heerlen", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Heerlen"] = 2

    arc_size.loc["Dordrecht", "Wageningen"] = 2
    arc_size.loc["Wageningen", "Dordrecht"] = 2

    # connections with big clusters Chemelot

    arc_size.loc["Heerlen", "Chemelot"] = 2
    arc_size.loc["Chemelot", "Heerlen"] = 2

    arc_size.loc["Roermond", "Chemelot"] = 2
    arc_size.loc["Chemelot", "Roermond"] = 2

    arc_size.loc["Roermond", "Chemelot"] = 2
    arc_size.loc["Chemelot", "Roermond"] = 2

    arc_size.loc["Venlo", "Chemelot"] = 2
    arc_size.loc["Chemelot", "Venlo"] = 2

    # connections with big clusters Zeeland

    arc_size.loc["Dongen", "Zeeland"] = 2
    arc_size.loc["Zeeland", "Dongen"] = 2

    arc_size.loc["Oosterhout", "Zeeland"] = 2
    arc_size.loc["Zeeland", "Oosterhout"] = 2

    arc_size.loc["Dordrecht", "Zeeland"] = 2
    arc_size.loc["Zeeland", "Dordrecht"] = 2

    arc_size.loc["Betuwe", "Zeeland"] = 2
    arc_size.loc["Zeeland", "Betuwe"] = 2

    # connections with big clusters Rotterdam

    arc_size.loc["Dongen", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Dongen"] = 2

    arc_size.loc["Oosterhout", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Oosterhout"] = 2

    arc_size.loc["Dordrecht", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Dordrecht"] = 2

    arc_size.loc["Betuwe", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Betuwe"] = 2

    arc_size.loc["Wageningen", "Rotterdam"] = 2
    arc_size.loc["Rotterdam", "Wageningen"] = 2

    # connections with big clusters North_Sea

    arc_size.loc["Dongen", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Dongen"] = 2

    arc_size.loc["Oosterhout", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Oosterhout"] = 2

    arc_size.loc["Dordrecht", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Dordrecht"] = 2

    arc_size.loc["Betuwe", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Betuwe"] = 2

    arc_size.loc["Wageningen", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Wageningen"] = 2

    arc_size.loc["Arnhem", "North_Sea"] = 2
    arc_size.loc["North_Sea", "Arnhem"] = 2

    # connections with big clusters North_Netherlands

    arc_size.loc["East_Groningen", "North_Netherlands"] = 2
    arc_size.loc["North_Netherlands", "East_Groningen"] = 2

    arc_size.loc["Betuwe", "North_Netherlands"] = 2
    arc_size.loc["North_Netherlands", "Betuwe"] = 2

    arc_size.loc["Wageningen", "North_Netherlands"] = 2
    arc_size.loc["North_Netherlands", "Wageningen"] = 2

    arc_size.loc["Arnhem", "North_Netherlands"] = 2
    arc_size.loc["North_Netherlands", "Arnhem"] = 2

    # connections with big clusters Zuidwending

    arc_size.loc["East_Groningen", "Zuidwending"] = 2
    arc_size.loc["Zuidwending", "East_Groningen"] = 2

    arc_size.loc["Betuwe", "Zuidwending"] = 2
    arc_size.loc["Zuidwending", "Betuwe"] = 2

    arc_size.loc["Wageningen", "Zuidwending"] = 2
    arc_size.loc["Zuidwending", "Wageningen"] = 2

    arc_size.loc["Arnhem", "Zuidwending"] = 2
    arc_size.loc["Zuidwending", "Arnhem"] = 2

    # </editor-fold>

    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "size_max_arcs.csv",
        sep=";")

    # <editor-fold desc="Arc size connections truck">
    # arc_size.loc["Roermond", "Dongen"] = 3
    # arc_size.loc["Dongen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Oosterhout"] = 3
    # arc_size.loc["Oosterhout", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Maastricht"] = 3
    # arc_size.loc["Maastricht", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Venlo"] = 3
    # arc_size.loc["Venlo", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Roermond"] = 3
    #
    # arc_size.loc["Dongen", "Oosterhout"] = 3
    # arc_size.loc["Oosterhout", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Maastricht"] = 3
    # arc_size.loc["Maastricht", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Venlo"] = 3
    # arc_size.loc["Venlo", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Dongen"] = 3
    #
    # arc_size.loc["Oosterhout", "Maastricht"] = 3
    # arc_size.loc["Maastricht", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Venlo"] = 3
    # arc_size.loc["Venlo", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Oosterhout"] = 3
    #
    # arc_size.loc["Maastricht", "Venlo"] = 3
    # arc_size.loc["Venlo", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Maastricht"] = 3
    #
    # arc_size.loc["Venlo", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Venlo"] = 3
    #
    # arc_size.loc["Arnhem", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Arnhem"] = 3
    #
    # arc_size.loc["East_Groningen", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "East_Groningen"] = 3
    #
    # arc_size.loc["Betuwe", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Betuwe"] = 3
    #
    # arc_size.loc["Heerlen", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Heerlen"] = 3
    #
    # arc_size.loc["Dordrecht", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Dordrecht"] = 3
    #
    # arc_size.loc["Wageningen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Wageningen"] = 3
    #
    # arc_size.loc["Rotterdam", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Rotterdam"] = 3
    #
    # arc_size.loc["Zeeland", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Zeeland"] = 3
    #
    # arc_size.loc["Zeeland", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Zeeland"] = 3
    #
    # arc_size.loc["Zeeland", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Zeeland"] = 3
    #
    # arc_size.loc["Zeeland", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Zeeland"] = 3
    #
    # arc_size.loc["North_Sea", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "North_Sea"] = 3
    #
    # arc_size.loc["North_Sea", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "North_Sea"] = 3
    #
    # arc_size.loc["North_Sea", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "North_Sea"] = 3
    #
    # arc_size.loc["North_Netherlands", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "North_Netherlands"] = 3
    #
    # arc_size.loc["North_Netherlands", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "North_Netherlands"] = 3
    #
    # arc_size.loc["Chemelot", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Chemelot"] = 3
    # # </editor-fold>
    #
    # arc_size.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "size_max_arcs.csv", sep=";")

    print("Max size per arc:", arc_size)

    # Delete the max_size_arc template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv")

    # # Connection
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)
    # <editor-fold desc="Connections low">

    #small infrastructure
    connection.loc["Roermond", "Dongen"] = 1
    connection.loc["Dongen", "Roermond"] = 1

    connection.loc["Roermond", "Oosterhout"] = 1
    connection.loc["Oosterhout", "Roermond"] = 1

    connection.loc["Roermond", "Maastricht"] = 1
    connection.loc["Maastricht", "Roermond"] = 1

    connection.loc["Roermond", "Venlo"] = 1
    connection.loc["Venlo", "Roermond"] = 1

    connection.loc["Roermond", "Arnhem"] = 1
    connection.loc["Arnhem", "Roermond"] = 1

    connection.loc["Roermond", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Roermond"] = 1

    connection.loc["Roermond", "Betuwe"] = 1
    connection.loc["Betuwe", "Roermond"] = 1

    connection.loc["Roermond", "Heerlen"] = 1
    connection.loc["Heerlen", "Roermond"] = 1

    connection.loc["Roermond", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Roermond"] = 1

    connection.loc["Roermond", "Wageningen"] = 1
    connection.loc["Wageningen", "Roermond"] = 1

    connection.loc["Dongen", "Oosterhout"] = 1
    connection.loc["Oosterhout", "Dongen"] = 1

    connection.loc["Dongen", "Maastricht"] = 1
    connection.loc["Maastricht", "Dongen"] = 1

    connection.loc["Dongen", "Venlo"] = 1
    connection.loc["Venlo", "Dongen"] = 1

    connection.loc["Dongen", "Arnhem"] = 1
    connection.loc["Arnhem", "Dongen"] = 1

    connection.loc["Dongen", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Dongen"] = 1

    connection.loc["Dongen", "Betuwe"] = 1
    connection.loc["Betuwe", "Dongen"] = 1

    connection.loc["Dongen", "Heerlen"] = 1
    connection.loc["Heerlen", "Dongen"] = 1

    connection.loc["Dongen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Dongen"] = 1

    connection.loc["Dongen", "Wageningen"] = 1
    connection.loc["Wageningen", "Dongen"] = 1

    connection.loc["Oosterhout", "Maastricht"] = 1
    connection.loc["Maastricht", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Venlo"] = 1
    connection.loc["Venlo", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Arnhem"] = 1
    connection.loc["Arnhem", "Oosterhout"] = 1

    connection.loc["Oosterhout", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Betuwe"] = 1
    connection.loc["Betuwe", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Heerlen"] = 1
    connection.loc["Heerlen", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Wageningen"] = 1
    connection.loc["Wageningen", "Oosterhout"] = 1

    connection.loc["Maastricht", "Venlo"] = 1
    connection.loc["Venlo", "Maastricht"] = 1

    connection.loc["Maastricht", "Arnhem"] = 1
    connection.loc["Arnhem", "Maastricht"] = 1

    connection.loc["Maastricht", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Maastricht"] = 1

    connection.loc["Maastricht", "Betuwe"] = 1
    connection.loc["Betuwe", "Maastricht"] = 1

    connection.loc["Maastricht", "Heerlen"] = 1
    connection.loc["Heerlen", "Maastricht"] = 1

    connection.loc["Maastricht", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Maastricht"] = 1

    connection.loc["Maastricht", "Wageningen"] = 1
    connection.loc["Wageningen", "Maastricht"] = 1

    connection.loc["Venlo", "Arnhem"] = 1
    connection.loc["Arnhem", "Venlo"] = 1

    connection.loc["Venlo", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Venlo"] = 1

    connection.loc["Venlo", "Betuwe"] = 1
    connection.loc["Betuwe", "Venlo"] = 1

    connection.loc["Venlo", "Heerlen"] = 1
    connection.loc["Heerlen", "Venlo"] = 1

    connection.loc["Venlo", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Venlo"] = 1

    connection.loc["Venlo", "Wageningen"] = 1
    connection.loc["Wageningen", "Venlo"] = 1

    connection.loc["Arnhem", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Arnhem"] = 1

    connection.loc["Arnhem", "Betuwe"] = 1
    connection.loc["Betuwe", "Arnhem"] = 1

    connection.loc["Arnhem", "Heerlen"] = 1
    connection.loc["Heerlen", "Arnhem"] = 1

    connection.loc["Arnhem", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Arnhem"] = 1

    connection.loc["Arnhem", "Wageningen"] = 1
    connection.loc["Wageningen", "Arnhem"] = 1

    connection.loc["East_Groningen", "Betuwe"] = 1
    connection.loc["Betuwe", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Heerlen"] = 1
    connection.loc["Heerlen", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Wageningen"] = 1
    connection.loc["Wageningen", "East_Groningen"] = 1

    connection.loc["Betuwe", "Heerlen"] = 1
    connection.loc["Heerlen", "Betuwe"] = 1

    connection.loc["Betuwe", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Betuwe"] = 1

    connection.loc["Betuwe", "Wageningen"] = 1
    connection.loc["Wageningen", "Betuwe"] = 1

    connection.loc["Heerlen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Heerlen"] = 1

    connection.loc["Heerlen", "Wageningen"] = 1
    connection.loc["Wageningen", "Heerlen"] = 1

    connection.loc["Dordrecht", "Wageningen"] = 1
    connection.loc["Wageningen", "Dordrecht"] = 1

    # connections with big clusters Chemelot

    connection.loc["Heerlen", "Chemelot"] = 1
    connection.loc["Chemelot", "Heerlen"] = 1

    connection.loc["Roermond", "Chemelot"] = 1
    connection.loc["Chemelot", "Roermond"] = 1

    connection.loc["Venlo", "Chemelot"] = 1
    connection.loc["Chemelot", "Venlo"] = 1

    # connections with big clusters Zeeland

    connection.loc["Dongen", "Zeeland"] = 1
    connection.loc["Zeeland", "Dongen"] = 1

    connection.loc["Oosterhout", "Zeeland"] = 1
    connection.loc["Zeeland", "Oosterhout"] = 1

    connection.loc["Dordrecht", "Zeeland"] = 1
    connection.loc["Zeeland", "Dordrecht"] = 1

    connection.loc["Betuwe", "Zeeland"] = 1
    connection.loc["Zeeland", "Betuwe"] = 1

    # connections with big clusters Rotterdam

    connection.loc["Dongen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Dongen"] = 1

    connection.loc["Oosterhout", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Oosterhout"] = 1

    connection.loc["Dordrecht", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Dordrecht"] = 1

    connection.loc["Betuwe", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Betuwe"] = 1

    connection.loc["Wageningen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Wageningen"] = 1

    # connections with big clusters North_Sea

    connection.loc["Dongen", "North_Sea"] = 1
    connection.loc["North_Sea", "Dongen"] = 1

    connection.loc["Oosterhout", "North_Sea"] = 1
    connection.loc["North_Sea", "Oosterhout"] = 1

    connection.loc["Dordrecht", "North_Sea"] = 1
    connection.loc["North_Sea", "Dordrecht"] = 1

    connection.loc["Betuwe", "North_Sea"] = 1
    connection.loc["North_Sea", "Betuwe"] = 1

    connection.loc["Wageningen", "North_Sea"] = 1
    connection.loc["North_Sea", "Wageningen"] = 1

    connection.loc["Arnhem", "North_Sea"] = 1
    connection.loc["North_Sea", "Arnhem"] = 1

    # connections with big clusters North_Netherlands

    connection.loc["East_Groningen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "East_Groningen"] = 1

    connection.loc["Betuwe", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Betuwe"] = 1

    connection.loc["Wageningen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Wageningen"] = 1

    connection.loc["Arnhem", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Arnhem"] = 1

    # connections with big clusters Zuidwending

    connection.loc["East_Groningen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "East_Groningen"] = 1

    connection.loc["Betuwe", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Betuwe"] = 1

    connection.loc["Wageningen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Wageningen"] = 1

    connection.loc["Arnhem", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Arnhem"] = 1

    # </editor-fold>

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "connection.csv",
        sep=";")
    #connection.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "connection.csv", sep=";")

    # # Connection
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)
    # <editor-fold desc="Connections high">

    #EHB

    connection.loc["Rotterdam", "Zeeland"] = 1
    connection.loc["Zeeland", "Rotterdam"] = 1
    connection.loc["Rotterdam", "North_Sea"] = 1
    connection.loc["North_Sea", "Rotterdam"] = 1
    connection.loc["North_Sea", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "North_Sea"] = 1
    connection.loc["Zuidwending", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Chemelot"] = 1
    connection.loc["Chemelot", "Zuidwending"] = 1
    connection.loc["Zeeland", "Chemelot"] = 1
    connection.loc["Chemelot", "Zeeland"] = 1

    # small infrastructure
    connection.loc["Roermond", "Dongen"] = 1
    connection.loc["Dongen", "Roermond"] = 1

    connection.loc["Roermond", "Oosterhout"] = 1
    connection.loc["Oosterhout", "Roermond"] = 1

    connection.loc["Roermond", "Maastricht"] = 1
    connection.loc["Maastricht", "Roermond"] = 1

    connection.loc["Roermond", "Venlo"] = 1
    connection.loc["Venlo", "Roermond"] = 1

    connection.loc["Roermond", "Arnhem"] = 1
    connection.loc["Arnhem", "Roermond"] = 1

    connection.loc["Roermond", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Roermond"] = 1

    connection.loc["Roermond", "Betuwe"] = 1
    connection.loc["Betuwe", "Roermond"] = 1

    connection.loc["Roermond", "Heerlen"] = 1
    connection.loc["Heerlen", "Roermond"] = 1

    connection.loc["Roermond", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Roermond"] = 1

    connection.loc["Roermond", "Wageningen"] = 1
    connection.loc["Wageningen", "Roermond"] = 1

    connection.loc["Dongen", "Oosterhout"] = 1
    connection.loc["Oosterhout", "Dongen"] = 1

    connection.loc["Dongen", "Maastricht"] = 1
    connection.loc["Maastricht", "Dongen"] = 1

    connection.loc["Dongen", "Venlo"] = 1
    connection.loc["Venlo", "Dongen"] = 1

    connection.loc["Dongen", "Arnhem"] = 1
    connection.loc["Arnhem", "Dongen"] = 1

    connection.loc["Dongen", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Dongen"] = 1

    connection.loc["Dongen", "Betuwe"] = 1
    connection.loc["Betuwe", "Dongen"] = 1

    connection.loc["Dongen", "Heerlen"] = 1
    connection.loc["Heerlen", "Dongen"] = 1

    connection.loc["Dongen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Dongen"] = 1

    connection.loc["Dongen", "Wageningen"] = 1
    connection.loc["Wageningen", "Dongen"] = 1

    connection.loc["Oosterhout", "Maastricht"] = 1
    connection.loc["Maastricht", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Venlo"] = 1
    connection.loc["Venlo", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Arnhem"] = 1
    connection.loc["Arnhem", "Oosterhout"] = 1

    connection.loc["Oosterhout", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Betuwe"] = 1
    connection.loc["Betuwe", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Heerlen"] = 1
    connection.loc["Heerlen", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Wageningen"] = 1
    connection.loc["Wageningen", "Oosterhout"] = 1

    connection.loc["Maastricht", "Venlo"] = 1
    connection.loc["Venlo", "Maastricht"] = 1

    connection.loc["Maastricht", "Arnhem"] = 1
    connection.loc["Arnhem", "Maastricht"] = 1

    connection.loc["Maastricht", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Maastricht"] = 1

    connection.loc["Maastricht", "Betuwe"] = 1
    connection.loc["Betuwe", "Maastricht"] = 1

    connection.loc["Maastricht", "Heerlen"] = 1
    connection.loc["Heerlen", "Maastricht"] = 1

    connection.loc["Maastricht", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Maastricht"] = 1

    connection.loc["Maastricht", "Wageningen"] = 1
    connection.loc["Wageningen", "Maastricht"] = 1

    connection.loc["Venlo", "Arnhem"] = 1
    connection.loc["Arnhem", "Venlo"] = 1

    connection.loc["Venlo", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Venlo"] = 1

    connection.loc["Venlo", "Betuwe"] = 1
    connection.loc["Betuwe", "Venlo"] = 1

    connection.loc["Venlo", "Heerlen"] = 1
    connection.loc["Heerlen", "Venlo"] = 1

    connection.loc["Venlo", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Venlo"] = 1

    connection.loc["Venlo", "Wageningen"] = 1
    connection.loc["Wageningen", "Venlo"] = 1

    connection.loc["Arnhem", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Arnhem"] = 1

    connection.loc["Arnhem", "Betuwe"] = 1
    connection.loc["Betuwe", "Arnhem"] = 1

    connection.loc["Arnhem", "Heerlen"] = 1
    connection.loc["Heerlen", "Arnhem"] = 1

    connection.loc["Arnhem", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Arnhem"] = 1

    connection.loc["Arnhem", "Wageningen"] = 1
    connection.loc["Wageningen", "Arnhem"] = 1

    connection.loc["East_Groningen", "Betuwe"] = 1
    connection.loc["Betuwe", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Heerlen"] = 1
    connection.loc["Heerlen", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Wageningen"] = 1
    connection.loc["Wageningen", "East_Groningen"] = 1

    connection.loc["Betuwe", "Heerlen"] = 1
    connection.loc["Heerlen", "Betuwe"] = 1

    connection.loc["Betuwe", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Betuwe"] = 1

    connection.loc["Betuwe", "Wageningen"] = 1
    connection.loc["Wageningen", "Betuwe"] = 1

    connection.loc["Heerlen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Heerlen"] = 1

    connection.loc["Heerlen", "Wageningen"] = 1
    connection.loc["Wageningen", "Heerlen"] = 1

    connection.loc["Dordrecht", "Wageningen"] = 1
    connection.loc["Wageningen", "Dordrecht"] = 1

    # connections with big clusters Chemelot

    connection.loc["Heerlen", "Chemelot"] = 1
    connection.loc["Chemelot", "Heerlen"] = 1

    connection.loc["Roermond", "Chemelot"] = 1
    connection.loc["Chemelot", "Roermond"] = 1

    connection.loc["Venlo", "Chemelot"] = 1
    connection.loc["Chemelot", "Venlo"] = 1

    # connections with big clusters Zeeland

    connection.loc["Dongen", "Zeeland"] = 1
    connection.loc["Zeeland", "Dongen"] = 1

    connection.loc["Oosterhout", "Zeeland"] = 1
    connection.loc["Zeeland", "Oosterhout"] = 1

    connection.loc["Dordrecht", "Zeeland"] = 1
    connection.loc["Zeeland", "Dordrecht"] = 1

    connection.loc["Betuwe", "Zeeland"] = 1
    connection.loc["Zeeland", "Betuwe"] = 1

    # connections with big clusters Rotterdam

    connection.loc["Dongen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Dongen"] = 1

    connection.loc["Oosterhout", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Oosterhout"] = 1

    connection.loc["Dordrecht", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Dordrecht"] = 1

    connection.loc["Betuwe", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Betuwe"] = 1

    connection.loc["Wageningen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Wageningen"] = 1

    # connections with big clusters North_Sea

    connection.loc["Dongen", "North_Sea"] = 1
    connection.loc["North_Sea", "Dongen"] = 1

    connection.loc["Oosterhout", "North_Sea"] = 1
    connection.loc["North_Sea", "Oosterhout"] = 1

    connection.loc["Dordrecht", "North_Sea"] = 1
    connection.loc["North_Sea", "Dordrecht"] = 1

    connection.loc["Betuwe", "North_Sea"] = 1
    connection.loc["North_Sea", "Betuwe"] = 1

    connection.loc["Wageningen", "North_Sea"] = 1
    connection.loc["North_Sea", "Wageningen"] = 1

    connection.loc["Arnhem", "North_Sea"] = 1
    connection.loc["North_Sea", "Arnhem"] = 1

    # connections with big clusters North_Netherlands

    connection.loc["East_Groningen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "East_Groningen"] = 1

    connection.loc["Betuwe", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Betuwe"] = 1

    connection.loc["Wageningen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Wageningen"] = 1

    connection.loc["Arnhem", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Arnhem"] = 1

    # connections with big clusters Zuidwending

    connection.loc["East_Groningen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "East_Groningen"] = 1

    connection.loc["Betuwe", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Betuwe"] = 1

    connection.loc["Wageningen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Wageningen"] = 1

    connection.loc["Arnhem", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Arnhem"] = 1

    # </editor-fold>

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "connection.csv",
        sep=";")
    # connection.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "connection.csv", sep=";")

    print("Connection:", connection)

    # Delete the template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "connection.csv")

    # # Distance
    distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "distance.csv", sep=";",
                           index_col=0)
    # <editor-fold desc="Distances">

    distance.loc["Roermond", "Dongen"] = 87
    distance.loc["Dongen", "Roermond"] = 87

    distance.loc["Roermond", "Oosterhout"] = 93
    distance.loc["Oosterhout", "Roermond"] = 93

    distance.loc["Roermond", "Maastricht"] = 43
    distance.loc["Maastricht", "Roermond"] = 43

    distance.loc["Roermond", "Venlo"] = 23
    distance.loc["Venlo", "Roermond"] = 23

    distance.loc["Roermond", "Arnhem"] = 88
    distance.loc["Arnhem", "Roermond"] = 88

    distance.loc["Roermond", "East_Groningen"] = 226
    distance.loc["East_Groningen", "Roermond"] = 226

    distance.loc["Roermond", "Betuwe"] = 87
    distance.loc["Betuwe", "Roermond"] = 87

    distance.loc["Roermond", "Heerlen"] = 35
    distance.loc["Heerlen", "Roermond"] = 35

    distance.loc["Roermond", "Dordrecht"] = 113
    distance.loc["Dordrecht", "Roermond"] = 113

    distance.loc["Roermond", "Wageningen"] = 89
    distance.loc["Wageningen", "Roermond"] = 89

    distance.loc["Roermond", "Rotterdam"] = 132
    distance.loc["Rotterdam", "Roermond"] = 132

    distance.loc["Roermond", "Zeeland"] = 152
    distance.loc["Zeeland", "Roermond"] = 152

    distance.loc["Roermond", "North_Sea"] = 160
    distance.loc["North_Sea", "Roermond"] = 160

    distance.loc["Roermond", "North_Netherlands"] = 258
    distance.loc["North_Netherlands", "Roermond"] = 258

    distance.loc["Roermond", "Chemelot"] = 28
    distance.loc["Chemelot", "Roermond"] = 28

    distance.loc["Roermond", "Zuidwending"] = 221
    distance.loc["Zuidwending", "Roermond"] = 221

    distance.loc["Dongen", "Oosterhout"] = 6
    distance.loc["Oosterhout", "Dongen"] = 6

    distance.loc["Dongen", "Maastricht"] = 100
    distance.loc["Maastricht", "Dongen"] = 100

    distance.loc["Dongen", "Venlo"] = 89
    distance.loc["Venlo", "Dongen"] = 89

    distance.loc["Dongen", "Arnhem"] = 77
    distance.loc["Arnhem", "Dongen"] = 77

    distance.loc["Dongen", "East_Groningen"] = 210
    distance.loc["East_Groningen", "Dongen"] = 210

    distance.loc["Dongen", "Betuwe"] = 50
    distance.loc["Betuwe", "Dongen"] = 50

    distance.loc["Dongen", "Heerlen"] = 109
    distance.loc["Heerlen", "Dongen"] = 109

    distance.loc["Dongen", "Dordrecht"] = 26
    distance.loc["Dordrecht", "Dongen"] = 26

    distance.loc["Dongen", "Wageningen"] = 63
    distance.loc["Wageningen", "Dongen"] = 63

    distance.loc["Dongen", "Rotterdam"] = 46
    distance.loc["Rotterdam", "Dongen"] = 46

    distance.loc["Dongen", "Zeeland"] = 77
    distance.loc["Zeeland", "Dongen"] = 77

    distance.loc["Dongen", "North_Sea"] = 90
    distance.loc["North_Sea", "Dongen"] = 90

    distance.loc["Dongen", "North_Netherlands"] = 240
    distance.loc["North_Netherlands", "Dongen"] = 240

    distance.loc["Dongen", "Chemelot"] = 93
    distance.loc["Chemelot", "Dongen"] = 93

    distance.loc["Dongen", "Zuidwending"] = 212
    distance.loc["Zuidwending", "Dongen"] = 212

    distance.loc["Oosterhout", "Maastricht"] = 105
    distance.loc["Maastricht", "Oosterhout"] = 105

    distance.loc["Oosterhout", "Venlo"] = 95
    distance.loc["Venlo", "Oosterhout"] = 95

    distance.loc["Oosterhout", "Arnhem"] = 81
    distance.loc["Arnhem", "Oosterhout"] = 81

    distance.loc["Oosterhout", "East_Groningen"] = 212
    distance.loc["East_Groningen", "Oosterhout"] = 212

    distance.loc["Oosterhout", "Betuwe"] = 54
    distance.loc["Betuwe", "Oosterhout"] = 54

    distance.loc["Oosterhout", "Heerlen"] = 115
    distance.loc["Heerlen", "Oosterhout"] = 115

    distance.loc["Oosterhout", "Dordrecht"] = 21
    distance.loc["Dordrecht", "Oosterhout"] = 21

    distance.loc["Oosterhout", "Wageningen"] = 66
    distance.loc["Wageningen", "Oosterhout"] = 66

    distance.loc["Oosterhout", "Rotterdam"] = 41
    distance.loc["Rotterdam", "Oosterhout"] = 41

    distance.loc["Oosterhout", "Zeeland"] = 72
    distance.loc["Zeeland", "Oosterhout"] = 72

    distance.loc["Oosterhout", "North_Sea"] = 88
    distance.loc["North_Sea", "Oosterhout"] = 88

    distance.loc["Oosterhout", "North_Netherlands"] = 241
    distance.loc["North_Netherlands", "Oosterhout"] = 241

    distance.loc["Oosterhout", "Chemelot"] = 99
    distance.loc["Chemelot", "Oosterhout"] = 99

    distance.loc["Oosterhout", "Zuidwending"] = 214
    distance.loc["Zuidwending", "Oosterhout"] = 214

    distance.loc["Maastricht", "Venlo"] = 66
    distance.loc["Venlo", "Maastricht"] = 66

    distance.loc["Maastricht", "Arnhem"] = 127
    distance.loc["Arnhem", "Maastricht"] = 127

    distance.loc["Maastricht", "East_Groningen"] = 268
    distance.loc["East_Groningen", "Maastricht"] = 268

    distance.loc["Maastricht", "Betuwe"] = 119
    distance.loc["Betuwe", "Maastricht"] = 119

    distance.loc["Maastricht", "Heerlen"] = 21
    distance.loc["Heerlen", "Maastricht"] = 21

    distance.loc["Maastricht", "Dordrecht"] = 126
    distance.loc["Dordrecht", "Maastricht"] = 126

    distance.loc["Maastricht", "Wageningen"] = 124
    distance.loc["Wageningen", "Maastricht"] = 124

    distance.loc["Maastricht", "Rotterdam"] = 146
    distance.loc["Rotterdam", "Maastricht"] = 146

    distance.loc["Maastricht", "Zeeland"] = 147
    distance.loc["Zeeland", "Maastricht"] = 147

    distance.loc["Maastricht", "North_Sea"] = 186
    distance.loc["North_Sea", "Maastricht"] = 186

    distance.loc["Maastricht", "North_Netherlands"] = 300
    distance.loc["North_Netherlands", "Maastricht"] = 300

    distance.loc["Maastricht", "Chemelot"] = 16
    distance.loc["Chemelot", "Maastricht"] = 16

    distance.loc["Maastricht", "Zuidwending"] = 264
    distance.loc["Zuidwending", "Maastricht"] = 264

    distance.loc["Venlo", "Arnhem"] = 71
    distance.loc["Arnhem", "Venlo"] = 71

    distance.loc["Venlo", "East_Groningen"] = 204
    distance.loc["East_Groningen", "Venlo"] = 204

    distance.loc["Venlo", "Betuwe"] = 77
    distance.loc["Betuwe", "Venlo"] = 77

    distance.loc["Venlo", "Heerlen"] = 55
    distance.loc["Heerlen", "Venlo"] = 55

    distance.loc["Venlo", "Dordrecht"] = 113
    distance.loc["Dordrecht", "Venlo"] = 113

    distance.loc["Venlo", "Wageningen"] = 75
    distance.loc["Wageningen", "Venlo"] = 75

    distance.loc["Venlo", "Rotterdam"] = 132
    distance.loc["Rotterdam", "Venlo"] = 132

    distance.loc["Venlo", "Zeeland"] = 161
    distance.loc["Zeeland", "Venlo"] = 161

    distance.loc["Venlo", "North_Sea"] = 150
    distance.loc["North_Sea", "Venlo"] = 150

    distance.loc["Venlo", "North_Netherlands"] = 236
    distance.loc["North_Netherlands", "Venlo"] = 236

    distance.loc["Venlo", "Chemelot"] = 50
    distance.loc["Chemelot", "Venlo"] = 50

    distance.loc["Venlo", "Zuidwending"] = 199
    distance.loc["Zuidwending", "Venlo"] = 199

    distance.loc["Arnhem", "East_Groningen"] = 143
    distance.loc["East_Groningen", "Arnhem"] = 143

    distance.loc["Arnhem", "Betuwe"] = 28
    distance.loc["Betuwe", "Arnhem"] = 28

    distance.loc["Arnhem", "Heerlen"] = 123
    distance.loc["Heerlen", "Arnhem"] = 123

    distance.loc["Arnhem", "Dordrecht"] = 86
    distance.loc["Dordrecht", "Arnhem"] = 86

    distance.loc["Arnhem", "Wageningen"] = 16
    distance.loc["Wageningen", "Arnhem"] = 16

    distance.loc["Arnhem", "Rotterdam"] = 98
    distance.loc["Rotterdam", "Arnhem"] = 98

    distance.loc["Arnhem", "Zeeland"] = 151
    distance.loc["Zeeland", "Arnhem"] = 151

    distance.loc["Arnhem", "North_Sea"] = 89
    distance.loc["North_Sea", "Arnhem"] = 89

    distance.loc["Arnhem", "North_Netherlands"] = 175
    distance.loc["North_Netherlands", "Arnhem"] = 175

    distance.loc["Arnhem", "Chemelot"] = 112
    distance.loc["Chemelot", "Arnhem"] = 112

    distance.loc["Arnhem", "Zuidwending"] = 142
    distance.loc["Zuidwending", "Arnhem"] = 142

    distance.loc["East_Groningen", "Betuwe"] = 162
    distance.loc["Betuwe", "East_Groningen"] = 162

    distance.loc["East_Groningen", "Heerlen"] = 260
    distance.loc["Heerlen", "East_Groningen"] = 260

    distance.loc["East_Groningen", "Dordrecht"] = 207
    distance.loc["Dordrecht", "East_Groningen"] = 207

    distance.loc["East_Groningen", "Wageningen"] = 152
    distance.loc["Wageningen", "East_Groningen"] = 152

    distance.loc["East_Groningen", "Rotterdam"] = 206
    distance.loc["Rotterdam", "East_Groningen"] = 206

    distance.loc["East_Groningen", "Zeeland"] = 270
    distance.loc["Zeeland", "East_Groningen"] = 270

    distance.loc["East_Groningen", "North_Sea"] = 152
    distance.loc["North_Sea", "East_Groningen"] = 152

    distance.loc["East_Groningen", "North_Netherlands"] = 32
    distance.loc["North_Netherlands", "East_Groningen"] = 32

    distance.loc["East_Groningen", "Chemelot"] = 252
    distance.loc["Chemelot", "East_Groningen"] = 252

    distance.loc["East_Groningen", "Zuidwending"] = 16
    distance.loc["Zuidwending", "East_Groningen"] = 16

    distance.loc["Betuwe", "Heerlen"] = 120
    distance.loc["Heerlen", "Betuwe"] = 120

    distance.loc["Betuwe", "Dordrecht"] = 58
    distance.loc["Dordrecht", "Betuwe"] = 58

    distance.loc["Betuwe", "Wageningen"] = 13
    distance.loc["Wageningen", "Betuwe"] = 13

    distance.loc["Betuwe", "Rotterdam"] = 70
    distance.loc["Rotterdam", "Betuwe"] = 70

    distance.loc["Betuwe", "Zeeland"] = 123
    distance.loc["Zeeland", "Betuwe"] = 123

    distance.loc["Betuwe", "North_Sea"] = 74
    distance.loc["North_Sea", "Betuwe"] = 74

    distance.loc["Betuwe", "North_Netherlands"] = 193
    distance.loc["North_Netherlands", "Betuwe"] = 193

    distance.loc["Betuwe", "Chemelot"] = 107
    distance.loc["Chemelot", "Betuwe"] = 107

    distance.loc["Betuwe", "Zuidwending"] = 163
    distance.loc["Zuidwending", "Betuwe"] = 163

    distance.loc["Heerlen", "Dordrecht"] = 136
    distance.loc["Dordrecht", "Heerlen"] = 136

    distance.loc["Heerlen", "Wageningen"] = 123
    distance.loc["Wageningen", "Heerlen"] = 123

    distance.loc["Heerlen", "Rotterdam"] = 156
    distance.loc["Rotterdam", "Heerlen"] = 156

    distance.loc["Heerlen", "Zeeland"] = 164
    distance.loc["Zeeland", "Heerlen"] = 164

    distance.loc["Heerlen", "North_Sea"] = 190
    distance.loc["North_Sea", "Heerlen"] = 190

    distance.loc["Heerlen", "North_Netherlands"] = 292
    distance.loc["North_Netherlands", "Heerlen"] = 292

    distance.loc["Heerlen", "Chemelot"] = 16
    distance.loc["Chemelot", "Heerlen"] = 16

    distance.loc["Heerlen", "Zuidwending"] = 254
    distance.loc["Zuidwending", "Heerlen"] = 254

    distance.loc["Dordrecht", "Wageningen"] = 70
    distance.loc["Wageningen", "Dordrecht"] = 70

    distance.loc["Dordrecht", "Rotterdam"] = 20
    distance.loc["Rotterdam", "Dordrecht"] = 20

    distance.loc["Dordrecht", "Zeeland"] = 66
    distance.loc["Zeeland", "Dordrecht"] = 66

    distance.loc["Dordrecht", "North_Sea"] = 71
    distance.loc["North_Sea", "Dordrecht"] = 71

    distance.loc["Dordrecht", "North_Netherlands"] = 235
    distance.loc["North_Netherlands", "Dordrecht"] = 235

    distance.loc["Dordrecht", "Chemelot"] = 120
    distance.loc["Chemelot", "Dordrecht"] = 120

    distance.loc["Dordrecht", "Zuidwending"] = 210
    distance.loc["Zuidwending", "Dordrecht"] = 210

    distance.loc["Wageningen", "Rotterdam"] = 82
    distance.loc["Rotterdam", "Wageningen"] = 82

    distance.loc["Wageningen", "Zeeland"] = 136
    distance.loc["Zeeland", "Wageningen"] = 136

    distance.loc["Wageningen", "North_Sea"] = 77
    distance.loc["North_Sea", "Wageningen"] = 77

    distance.loc["Wageningen", "North_Netherlands"] = 183
    distance.loc["North_Netherlands", "Wageningen"] = 183

    distance.loc["Wageningen", "Chemelot"] = 111
    distance.loc["Chemelot", "Wageningen"] = 111

    distance.loc["Wageningen", "Zuidwending"] = 152
    distance.loc["Zuidwending", "Wageningen"] = 152

    distance.loc["Rotterdam", "Zeeland"] = 64
    distance.loc["Zeeland", "Rotterdam"] = 64

    distance.loc["Rotterdam", "North_Sea"] = 61
    distance.loc["North_Sea", "Rotterdam"] = 61

    distance.loc["Rotterdam", "North_Netherlands"] = 232
    distance.loc["North_Netherlands", "Rotterdam"] = 232

    distance.loc["Rotterdam", "Chemelot"] = 140
    distance.loc["Chemelot", "Rotterdam"] = 140

    distance.loc["Rotterdam", "Zuidwending"] = 211
    distance.loc["Zuidwending", "Rotterdam"] = 211

    distance.loc["Zeeland", "North_Sea"] = 123
    distance.loc["North_Sea", "Zeeland"] = 123

    distance.loc["Zeeland", "North_Netherlands"] = 296
    distance.loc["North_Netherlands", "Zeeland"] = 296

    distance.loc["Zeeland", "Chemelot"] = 148
    distance.loc["Chemelot", "Zeeland"] = 148

    distance.loc["Zeeland", "Zuidwending"] = 275
    distance.loc["Zuidwending", "Zeeland"] = 275

    distance.loc["North_Sea", "North_Netherlands"] = 176
    distance.loc["North_Netherlands", "North_Sea"] = 176

    distance.loc["North_Sea", "Chemelot"] = 176
    distance.loc["Chemelot", "North_Sea"] = 176

    distance.loc["North_Sea", "Zuidwending"] = 160
    distance.loc["Zuidwending", "North_Sea"] = 160

    distance.loc["North_Netherlands", "Chemelot"] = 284
    distance.loc["Chemelot", "North_Netherlands"] = 284

    distance.loc["North_Netherlands", "Zuidwending"] = 41
    distance.loc["Zuidwending", "North_Netherlands"] = 41

    distance.loc["Chemelot", "Zuidwending"] = 248
    distance.loc["Zuidwending", "Chemelot"] = 248

    # </editor-fold>

    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "distance.csv",
        sep=";")
    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "distance.csv",
        sep=";")
    #distance.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "distance.csv", sep=";")
    print("Distance:", distance)

    # Delete the template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "distance.csv")

def add_new_network_H2_small_and_EHB_fixed(input_data_path):
    # # Make a new folder for the new network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP_big",
                exist_ok=True)
    #os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP", exist_ok=True)

    print("New network")

    # # max size arc
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)
    # <editor-fold desc="Arc size connections pipeline low P">
    arc_size.loc["Roermond", "Dongen"] = 1
    arc_size.loc["Dongen", "Roermond"] = 1

    arc_size.loc["Roermond", "Oosterhout"] = 1
    arc_size.loc["Oosterhout", "Roermond"] = 1

    arc_size.loc["Roermond", "Maastricht"] = 1
    arc_size.loc["Maastricht", "Roermond"] = 1

    arc_size.loc["Roermond", "Venlo"] = 1
    arc_size.loc["Venlo", "Roermond"] = 1

    arc_size.loc["Roermond", "Arnhem"] = 1
    arc_size.loc["Arnhem", "Roermond"] = 1

    arc_size.loc["Roermond", "East_Groningen"] = 1
    arc_size.loc["East_Groningen", "Roermond"] = 1

    arc_size.loc["Roermond", "Betuwe"] = 1
    arc_size.loc["Betuwe", "Roermond"] = 1

    arc_size.loc["Roermond", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Roermond"] = 1

    arc_size.loc["Roermond", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Roermond"] = 1

    arc_size.loc["Roermond", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Roermond"] = 1

    arc_size.loc["Dongen", "Oosterhout"] = 1
    arc_size.loc["Oosterhout", "Dongen"] = 1

    arc_size.loc["Dongen", "Maastricht"] = 1
    arc_size.loc["Maastricht", "Dongen"] = 1

    arc_size.loc["Dongen", "Venlo"] = 1
    arc_size.loc["Venlo", "Dongen"] = 1

    arc_size.loc["Dongen", "Arnhem"] = 1
    arc_size.loc["Arnhem", "Dongen"] = 1

    arc_size.loc["Dongen", "East_Groningen"] = 1
    arc_size.loc["East_Groningen", "Dongen"] = 1

    arc_size.loc["Dongen", "Betuwe"] = 1
    arc_size.loc["Betuwe", "Dongen"] = 1

    arc_size.loc["Dongen", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Dongen"] = 1

    arc_size.loc["Dongen", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Dongen"] = 1

    arc_size.loc["Dongen", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Dongen"] = 1

    arc_size.loc["Oosterhout", "Maastricht"] = 1
    arc_size.loc["Maastricht", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "Venlo"] = 1
    arc_size.loc["Venlo", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "Arnhem"] = 1
    arc_size.loc["Arnhem", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "East_Groningen"] = 1
    arc_size.loc["East_Groningen", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "Betuwe"] = 1
    arc_size.loc["Betuwe", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Oosterhout"] = 1

    arc_size.loc["Maastricht", "Venlo"] = 1
    arc_size.loc["Venlo", "Maastricht"] = 1

    arc_size.loc["Maastricht", "Arnhem"] = 1
    arc_size.loc["Arnhem", "Maastricht"] = 1

    arc_size.loc["Maastricht", "East_Groningen"] = 1
    arc_size.loc["East_Groningen", "Maastricht"] = 1

    arc_size.loc["Maastricht", "Betuwe"] = 1
    arc_size.loc["Betuwe", "Maastricht"] = 1

    arc_size.loc["Maastricht", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Maastricht"] = 1

    arc_size.loc["Maastricht", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Maastricht"] = 1

    arc_size.loc["Maastricht", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Maastricht"] = 1

    arc_size.loc["Venlo", "Arnhem"] = 1
    arc_size.loc["Arnhem", "Venlo"] = 1

    arc_size.loc["Venlo", "East_Groningen"] = 1
    arc_size.loc["East_Groningen", "Venlo"] = 1

    arc_size.loc["Venlo", "Betuwe"] = 1
    arc_size.loc["Betuwe", "Venlo"] = 1

    arc_size.loc["Venlo", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Venlo"] = 1

    arc_size.loc["Venlo", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Venlo"] = 1

    arc_size.loc["Venlo", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Venlo"] = 1

    arc_size.loc["Arnhem", "East_Groningen"] = 1
    arc_size.loc["East_Groningen", "Arnhem"] = 1

    arc_size.loc["Arnhem", "Betuwe"] = 1
    arc_size.loc["Betuwe", "Arnhem"] = 1

    arc_size.loc["Arnhem", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Arnhem"] = 1

    arc_size.loc["Arnhem", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Arnhem"] = 1

    arc_size.loc["Arnhem", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Arnhem"] = 1

    arc_size.loc["East_Groningen", "Betuwe"] = 1
    arc_size.loc["Betuwe", "East_Groningen"] = 1

    arc_size.loc["East_Groningen", "Heerlen"] = 1
    arc_size.loc["Heerlen", "East_Groningen"] = 1

    arc_size.loc["East_Groningen", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "East_Groningen"] = 1

    arc_size.loc["East_Groningen", "Wageningen"] = 1
    arc_size.loc["Wageningen", "East_Groningen"] = 1

    arc_size.loc["Betuwe", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Betuwe"] = 1

    arc_size.loc["Betuwe", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Betuwe"] = 1

    arc_size.loc["Betuwe", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Betuwe"] = 1

    arc_size.loc["Heerlen", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Heerlen"] = 1

    arc_size.loc["Heerlen", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Heerlen"] = 1

    arc_size.loc["Dordrecht", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Dordrecht"] = 1

    # connections with big clusters Chemelot
    arc_size.loc["Heerlen", "Chemelot"] = 1
    arc_size.loc["Chemelot", "Heerlen"] = 1

    arc_size.loc["Roermond", "Chemelot"] = 1
    arc_size.loc["Chemelot", "Roermond"] = 1

    arc_size.loc["Venlo", "Chemelot"] = 1
    arc_size.loc["Chemelot", "Venlo"] = 1

    # connections with big clusters Zeeland
    arc_size.loc["Dongen", "Zeeland"] = 1
    arc_size.loc["Zeeland", "Dongen"] = 1

    arc_size.loc["Oosterhout", "Zeeland"] = 1
    arc_size.loc["Zeeland", "Oosterhout"] = 1

    arc_size.loc["Dordrecht", "Zeeland"] = 1
    arc_size.loc["Zeeland", "Dordrecht"] = 1

    arc_size.loc["Betuwe", "Zeeland"] = 1
    arc_size.loc["Zeeland", "Betuwe"] = 1

    # connections with big clusters Rotterdam
    arc_size.loc["Dongen", "Rotterdam"] = 1
    arc_size.loc["Rotterdam", "Dongen"] = 1

    arc_size.loc["Oosterhout", "Rotterdam"] = 1
    arc_size.loc["Rotterdam", "Oosterhout"] = 1

    arc_size.loc["Dordrecht", "Rotterdam"] = 1
    arc_size.loc["Rotterdam", "Dordrecht"] = 1

    arc_size.loc["Betuwe", "Rotterdam"] = 1
    arc_size.loc["Rotterdam", "Betuwe"] = 1

    arc_size.loc["Wageningen", "Rotterdam"] = 1
    arc_size.loc["Rotterdam", "Wageningen"] = 1

    # connections with big clusters North_Sea
    arc_size.loc["Dongen", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Dongen"] = 1

    arc_size.loc["Oosterhout", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Oosterhout"] = 1

    arc_size.loc["Dordrecht", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Dordrecht"] = 1

    arc_size.loc["Betuwe", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Betuwe"] = 1

    arc_size.loc["Wageningen", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Wageningen"] = 1

    arc_size.loc["Arnhem", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Arnhem"] = 1

    # connections with big clusters North_Netherlands
    arc_size.loc["East_Groningen", "North_Netherlands"] = 1
    arc_size.loc["North_Netherlands", "East_Groningen"] = 1

    arc_size.loc["Betuwe", "North_Netherlands"] = 1
    arc_size.loc["North_Netherlands", "Betuwe"] = 1

    arc_size.loc["Wageningen", "North_Netherlands"] = 1
    arc_size.loc["North_Netherlands", "Wageningen"] = 1

    arc_size.loc["Arnhem", "North_Netherlands"] = 1
    arc_size.loc["North_Netherlands", "Arnhem"] = 1

    # connections with big clusters Zuidwending
    arc_size.loc["East_Groningen", "Zuidwending"] = 1
    arc_size.loc["Zuidwending", "East_Groningen"] = 1

    arc_size.loc["Betuwe", "Zuidwending"] = 1
    arc_size.loc["Zuidwending", "Betuwe"] = 1

    arc_size.loc["Wageningen", "Zuidwending"] = 1
    arc_size.loc["Zuidwending", "Wageningen"] = 1

    arc_size.loc["Arnhem", "Zuidwending"] = 1
    arc_size.loc["Zuidwending", "Arnhem"] = 1
    # </editor-fold>

    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP_big" / "size_max_arcs.csv",
        sep=";")

    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP_big",
                exist_ok=True)

    # <editor-fold desc="Arc size connections pipeline high P">

    arc_size.loc["Rotterdam", "Zeeland"] = 1
    arc_size.loc["Zeeland", "Rotterdam"] = 1
    arc_size.loc["Rotterdam", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Rotterdam"] = 1
    arc_size.loc["North_Sea", "North_Netherlands"] = 1
    arc_size.loc["North_Netherlands", "North_Sea"] = 1
    arc_size.loc["Zuidwending", "North_Netherlands"] = 1
    arc_size.loc["North_Netherlands", "Zuidwending"] = 1
    arc_size.loc["Zuidwending", "Chemelot"] = 1
    arc_size.loc["Chemelot", "Zuidwending"] = 1
    arc_size.loc["Zeeland", "Chemelot"] = 1
    arc_size.loc["Chemelot", "Zeeland"] = 1

    arc_size.loc["Roermond", "Dongen"] = 1
    arc_size.loc["Dongen", "Roermond"] = 1

    arc_size.loc["Roermond", "Oosterhout"] = 1
    arc_size.loc["Oosterhout", "Roermond"] = 1

    arc_size.loc["Roermond", "Maastricht"] = 1
    arc_size.loc["Maastricht", "Roermond"] = 1

    arc_size.loc["Roermond", "Venlo"] = 1
    arc_size.loc["Venlo", "Roermond"] = 1

    arc_size.loc["Roermond", "Arnhem"] = 1
    arc_size.loc["Arnhem", "Roermond"] = 1

    arc_size.loc["Roermond", "East_Groningen"] = 1
    arc_size.loc["East_Groningen", "Roermond"] = 1

    arc_size.loc["Roermond", "Betuwe"] = 1
    arc_size.loc["Betuwe", "Roermond"] = 1

    arc_size.loc["Roermond", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Roermond"] = 1

    arc_size.loc["Roermond", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Roermond"] = 1

    arc_size.loc["Roermond", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Roermond"] = 1

    arc_size.loc["Dongen", "Oosterhout"] = 1
    arc_size.loc["Oosterhout", "Dongen"] = 1

    arc_size.loc["Dongen", "Maastricht"] = 1
    arc_size.loc["Maastricht", "Dongen"] = 1

    arc_size.loc["Dongen", "Venlo"] = 1
    arc_size.loc["Venlo", "Dongen"] = 1

    arc_size.loc["Dongen", "Arnhem"] = 1
    arc_size.loc["Arnhem", "Dongen"] = 1

    arc_size.loc["Dongen", "East_Groningen"] = 1
    arc_size.loc["East_Groningen", "Dongen"] = 1

    arc_size.loc["Dongen", "Betuwe"] = 1
    arc_size.loc["Betuwe", "Dongen"] = 1

    arc_size.loc["Dongen", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Dongen"] = 1

    arc_size.loc["Dongen", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Dongen"] = 1

    arc_size.loc["Dongen", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Dongen"] = 1

    arc_size.loc["Oosterhout", "Maastricht"] = 1
    arc_size.loc["Maastricht", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "Venlo"] = 1
    arc_size.loc["Venlo", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "Arnhem"] = 1
    arc_size.loc["Arnhem", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "East_Groningen"] = 1
    arc_size.loc["East_Groningen", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "Betuwe"] = 1
    arc_size.loc["Betuwe", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Oosterhout"] = 1

    arc_size.loc["Oosterhout", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Oosterhout"] = 1

    arc_size.loc["Maastricht", "Venlo"] = 1
    arc_size.loc["Venlo", "Maastricht"] = 1

    arc_size.loc["Maastricht", "Arnhem"] = 1
    arc_size.loc["Arnhem", "Maastricht"] = 1

    arc_size.loc["Maastricht", "East_Groningen"] = 1
    arc_size.loc["East_Groningen", "Maastricht"] = 1

    arc_size.loc["Maastricht", "Betuwe"] = 1
    arc_size.loc["Betuwe", "Maastricht"] = 1

    arc_size.loc["Maastricht", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Maastricht"] = 1

    arc_size.loc["Maastricht", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Maastricht"] = 1

    arc_size.loc["Maastricht", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Maastricht"] = 1

    arc_size.loc["Venlo", "Arnhem"] = 1
    arc_size.loc["Arnhem", "Venlo"] = 1

    arc_size.loc["Venlo", "East_Groningen"] = 1
    arc_size.loc["East_Groningen", "Venlo"] = 1

    arc_size.loc["Venlo", "Betuwe"] = 1
    arc_size.loc["Betuwe", "Venlo"] = 1

    arc_size.loc["Venlo", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Venlo"] = 1

    arc_size.loc["Venlo", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Venlo"] = 1

    arc_size.loc["Venlo", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Venlo"] = 1

    arc_size.loc["Arnhem", "East_Groningen"] = 1
    arc_size.loc["East_Groningen", "Arnhem"] = 1

    arc_size.loc["Arnhem", "Betuwe"] = 1
    arc_size.loc["Betuwe", "Arnhem"] = 1

    arc_size.loc["Arnhem", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Arnhem"] = 1

    arc_size.loc["Arnhem", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Arnhem"] = 1

    arc_size.loc["Arnhem", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Arnhem"] = 1

    arc_size.loc["East_Groningen", "Betuwe"] = 1
    arc_size.loc["Betuwe", "East_Groningen"] = 1

    arc_size.loc["East_Groningen", "Heerlen"] = 1
    arc_size.loc["Heerlen", "East_Groningen"] = 1

    arc_size.loc["East_Groningen", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "East_Groningen"] = 1

    arc_size.loc["East_Groningen", "Wageningen"] = 1
    arc_size.loc["Wageningen", "East_Groningen"] = 1

    arc_size.loc["Betuwe", "Heerlen"] = 1
    arc_size.loc["Heerlen", "Betuwe"] = 1

    arc_size.loc["Betuwe", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Betuwe"] = 1

    arc_size.loc["Betuwe", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Betuwe"] = 1

    arc_size.loc["Heerlen", "Dordrecht"] = 1
    arc_size.loc["Dordrecht", "Heerlen"] = 1

    arc_size.loc["Heerlen", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Heerlen"] = 1

    arc_size.loc["Dordrecht", "Wageningen"] = 1
    arc_size.loc["Wageningen", "Dordrecht"] = 1

    # connections with big clusters Chemelot

    arc_size.loc["Heerlen", "Chemelot"] = 1
    arc_size.loc["Chemelot", "Heerlen"] = 1

    arc_size.loc["Roermond", "Chemelot"] = 1
    arc_size.loc["Chemelot", "Roermond"] = 1

    arc_size.loc["Roermond", "Chemelot"] = 1
    arc_size.loc["Chemelot", "Roermond"] = 1

    arc_size.loc["Venlo", "Chemelot"] = 1
    arc_size.loc["Chemelot", "Venlo"] = 1

    # connections with big clusters Zeeland

    arc_size.loc["Dongen", "Zeeland"] = 1
    arc_size.loc["Zeeland", "Dongen"] = 1

    arc_size.loc["Oosterhout", "Zeeland"] = 1
    arc_size.loc["Zeeland", "Oosterhout"] = 1

    arc_size.loc["Dordrecht", "Zeeland"] = 1
    arc_size.loc["Zeeland", "Dordrecht"] = 1

    arc_size.loc["Betuwe", "Zeeland"] = 1
    arc_size.loc["Zeeland", "Betuwe"] = 1

    # connections with big clusters Rotterdam

    arc_size.loc["Dongen", "Rotterdam"] = 1
    arc_size.loc["Rotterdam", "Dongen"] = 1

    arc_size.loc["Oosterhout", "Rotterdam"] = 1
    arc_size.loc["Rotterdam", "Oosterhout"] = 1

    arc_size.loc["Dordrecht", "Rotterdam"] = 1
    arc_size.loc["Rotterdam", "Dordrecht"] = 1

    arc_size.loc["Betuwe", "Rotterdam"] = 1
    arc_size.loc["Rotterdam", "Betuwe"] = 1

    arc_size.loc["Wageningen", "Rotterdam"] = 1
    arc_size.loc["Rotterdam", "Wageningen"] = 1

    # connections with big clusters North_Sea

    arc_size.loc["Dongen", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Dongen"] = 1

    arc_size.loc["Oosterhout", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Oosterhout"] = 1

    arc_size.loc["Dordrecht", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Dordrecht"] = 1

    arc_size.loc["Betuwe", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Betuwe"] = 1

    arc_size.loc["Wageningen", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Wageningen"] = 1

    arc_size.loc["Arnhem", "North_Sea"] = 1
    arc_size.loc["North_Sea", "Arnhem"] = 1

    # connections with big clusters North_Netherlands

    arc_size.loc["East_Groningen", "North_Netherlands"] = 1
    arc_size.loc["North_Netherlands", "East_Groningen"] = 1

    arc_size.loc["Betuwe", "North_Netherlands"] = 1
    arc_size.loc["North_Netherlands", "Betuwe"] = 1

    arc_size.loc["Wageningen", "North_Netherlands"] = 1
    arc_size.loc["North_Netherlands", "Wageningen"] = 1

    arc_size.loc["Arnhem", "North_Netherlands"] = 1
    arc_size.loc["North_Netherlands", "Arnhem"] = 1

    # connections with big clusters Zuidwending

    arc_size.loc["East_Groningen", "Zuidwending"] = 1
    arc_size.loc["Zuidwending", "East_Groningen"] = 1

    arc_size.loc["Betuwe", "Zuidwending"] = 1
    arc_size.loc["Zuidwending", "Betuwe"] = 1

    arc_size.loc["Wageningen", "Zuidwending"] = 1
    arc_size.loc["Zuidwending", "Wageningen"] = 1

    arc_size.loc["Arnhem", "Zuidwending"] = 1
    arc_size.loc["Zuidwending", "Arnhem"] = 1

    # </editor-fold>

    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP_big" / "size_max_arcs.csv",
        sep=";")

    # <editor-fold desc="Arc size connections truck">
    # arc_size.loc["Roermond", "Dongen"] = 3
    # arc_size.loc["Dongen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Oosterhout"] = 3
    # arc_size.loc["Oosterhout", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Maastricht"] = 3
    # arc_size.loc["Maastricht", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Venlo"] = 3
    # arc_size.loc["Venlo", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Roermond"] = 3
    #
    # arc_size.loc["Roermond", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Roermond"] = 3
    #
    # arc_size.loc["Dongen", "Oosterhout"] = 3
    # arc_size.loc["Oosterhout", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Maastricht"] = 3
    # arc_size.loc["Maastricht", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Venlo"] = 3
    # arc_size.loc["Venlo", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Dongen"] = 3
    #
    # arc_size.loc["Dongen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Dongen"] = 3
    #
    # arc_size.loc["Oosterhout", "Maastricht"] = 3
    # arc_size.loc["Maastricht", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Venlo"] = 3
    # arc_size.loc["Venlo", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Oosterhout"] = 3
    #
    # arc_size.loc["Oosterhout", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Oosterhout"] = 3
    #
    # arc_size.loc["Maastricht", "Venlo"] = 3
    # arc_size.loc["Venlo", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Maastricht"] = 3
    #
    # arc_size.loc["Maastricht", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Maastricht"] = 3
    #
    # arc_size.loc["Venlo", "Arnhem"] = 3
    # arc_size.loc["Arnhem", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Venlo"] = 3
    #
    # arc_size.loc["Venlo", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Venlo"] = 3
    #
    # arc_size.loc["Arnhem", "East_Groningen"] = 3
    # arc_size.loc["East_Groningen", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Arnhem"] = 3
    #
    # arc_size.loc["Arnhem", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Arnhem"] = 3
    #
    # arc_size.loc["East_Groningen", "Betuwe"] = 3
    # arc_size.loc["Betuwe", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "East_Groningen"] = 3
    #
    # arc_size.loc["East_Groningen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "East_Groningen"] = 3
    #
    # arc_size.loc["Betuwe", "Heerlen"] = 3
    # arc_size.loc["Heerlen", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Betuwe"] = 3
    #
    # arc_size.loc["Betuwe", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Betuwe"] = 3
    #
    # arc_size.loc["Heerlen", "Dordrecht"] = 3
    # arc_size.loc["Dordrecht", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Heerlen"] = 3
    #
    # arc_size.loc["Heerlen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Heerlen"] = 3
    #
    # arc_size.loc["Dordrecht", "Wageningen"] = 3
    # arc_size.loc["Wageningen", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Dordrecht"] = 3
    #
    # arc_size.loc["Dordrecht", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Dordrecht"] = 3
    #
    # arc_size.loc["Wageningen", "Rotterdam"] = 3
    # arc_size.loc["Rotterdam", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Wageningen"] = 3
    #
    # arc_size.loc["Wageningen", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Wageningen"] = 3
    #
    # arc_size.loc["Rotterdam", "Zeeland"] = 3
    # arc_size.loc["Zeeland", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Rotterdam"] = 3
    #
    # arc_size.loc["Rotterdam", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Rotterdam"] = 3
    #
    # arc_size.loc["Zeeland", "North_Sea"] = 3
    # arc_size.loc["North_Sea", "Zeeland"] = 3
    #
    # arc_size.loc["Zeeland", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "Zeeland"] = 3
    #
    # arc_size.loc["Zeeland", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "Zeeland"] = 3
    #
    # arc_size.loc["Zeeland", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Zeeland"] = 3
    #
    # arc_size.loc["North_Sea", "North_Netherlands"] = 3
    # arc_size.loc["North_Netherlands", "North_Sea"] = 3
    #
    # arc_size.loc["North_Sea", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "North_Sea"] = 3
    #
    # arc_size.loc["North_Sea", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "North_Sea"] = 3
    #
    # arc_size.loc["North_Netherlands", "Chemelot"] = 3
    # arc_size.loc["Chemelot", "North_Netherlands"] = 3
    #
    # arc_size.loc["North_Netherlands", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "North_Netherlands"] = 3
    #
    # arc_size.loc["Chemelot", "Zuidwending"] = 3
    # arc_size.loc["Zuidwending", "Chemelot"] = 3
    # # </editor-fold>
    #
    # arc_size.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "size_max_arcs.csv", sep=";")

    print("Max size per arc:", arc_size)

    # Delete the max_size_arc template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv")

    # # Connection
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)
    # <editor-fold desc="Connections low">

    #small infrastructure
    connection.loc["Roermond", "Dongen"] = 1
    connection.loc["Dongen", "Roermond"] = 1

    connection.loc["Roermond", "Oosterhout"] = 1
    connection.loc["Oosterhout", "Roermond"] = 1

    connection.loc["Roermond", "Maastricht"] = 1
    connection.loc["Maastricht", "Roermond"] = 1

    connection.loc["Roermond", "Venlo"] = 1
    connection.loc["Venlo", "Roermond"] = 1

    connection.loc["Roermond", "Arnhem"] = 1
    connection.loc["Arnhem", "Roermond"] = 1

    connection.loc["Roermond", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Roermond"] = 1

    connection.loc["Roermond", "Betuwe"] = 1
    connection.loc["Betuwe", "Roermond"] = 1

    connection.loc["Roermond", "Heerlen"] = 1
    connection.loc["Heerlen", "Roermond"] = 1

    connection.loc["Roermond", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Roermond"] = 1

    connection.loc["Roermond", "Wageningen"] = 1
    connection.loc["Wageningen", "Roermond"] = 1

    connection.loc["Dongen", "Oosterhout"] = 1
    connection.loc["Oosterhout", "Dongen"] = 1

    connection.loc["Dongen", "Maastricht"] = 1
    connection.loc["Maastricht", "Dongen"] = 1

    connection.loc["Dongen", "Venlo"] = 1
    connection.loc["Venlo", "Dongen"] = 1

    connection.loc["Dongen", "Arnhem"] = 1
    connection.loc["Arnhem", "Dongen"] = 1

    connection.loc["Dongen", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Dongen"] = 1

    connection.loc["Dongen", "Betuwe"] = 1
    connection.loc["Betuwe", "Dongen"] = 1

    connection.loc["Dongen", "Heerlen"] = 1
    connection.loc["Heerlen", "Dongen"] = 1

    connection.loc["Dongen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Dongen"] = 1

    connection.loc["Dongen", "Wageningen"] = 1
    connection.loc["Wageningen", "Dongen"] = 1

    connection.loc["Oosterhout", "Maastricht"] = 1
    connection.loc["Maastricht", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Venlo"] = 1
    connection.loc["Venlo", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Arnhem"] = 1
    connection.loc["Arnhem", "Oosterhout"] = 1

    connection.loc["Oosterhout", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Betuwe"] = 1
    connection.loc["Betuwe", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Heerlen"] = 1
    connection.loc["Heerlen", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Wageningen"] = 1
    connection.loc["Wageningen", "Oosterhout"] = 1

    connection.loc["Maastricht", "Venlo"] = 1
    connection.loc["Venlo", "Maastricht"] = 1

    connection.loc["Maastricht", "Arnhem"] = 1
    connection.loc["Arnhem", "Maastricht"] = 1

    connection.loc["Maastricht", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Maastricht"] = 1

    connection.loc["Maastricht", "Betuwe"] = 1
    connection.loc["Betuwe", "Maastricht"] = 1

    connection.loc["Maastricht", "Heerlen"] = 1
    connection.loc["Heerlen", "Maastricht"] = 1

    connection.loc["Maastricht", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Maastricht"] = 1

    connection.loc["Maastricht", "Wageningen"] = 1
    connection.loc["Wageningen", "Maastricht"] = 1

    connection.loc["Venlo", "Arnhem"] = 1
    connection.loc["Arnhem", "Venlo"] = 1

    connection.loc["Venlo", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Venlo"] = 1

    connection.loc["Venlo", "Betuwe"] = 1
    connection.loc["Betuwe", "Venlo"] = 1

    connection.loc["Venlo", "Heerlen"] = 1
    connection.loc["Heerlen", "Venlo"] = 1

    connection.loc["Venlo", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Venlo"] = 1

    connection.loc["Venlo", "Wageningen"] = 1
    connection.loc["Wageningen", "Venlo"] = 1

    connection.loc["Arnhem", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Arnhem"] = 1

    connection.loc["Arnhem", "Betuwe"] = 1
    connection.loc["Betuwe", "Arnhem"] = 1

    connection.loc["Arnhem", "Heerlen"] = 1
    connection.loc["Heerlen", "Arnhem"] = 1

    connection.loc["Arnhem", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Arnhem"] = 1

    connection.loc["Arnhem", "Wageningen"] = 1
    connection.loc["Wageningen", "Arnhem"] = 1

    connection.loc["East_Groningen", "Betuwe"] = 1
    connection.loc["Betuwe", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Heerlen"] = 1
    connection.loc["Heerlen", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Wageningen"] = 1
    connection.loc["Wageningen", "East_Groningen"] = 1

    connection.loc["Betuwe", "Heerlen"] = 1
    connection.loc["Heerlen", "Betuwe"] = 1

    connection.loc["Betuwe", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Betuwe"] = 1

    connection.loc["Betuwe", "Wageningen"] = 1
    connection.loc["Wageningen", "Betuwe"] = 1

    connection.loc["Heerlen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Heerlen"] = 1

    connection.loc["Heerlen", "Wageningen"] = 1
    connection.loc["Wageningen", "Heerlen"] = 1

    connection.loc["Dordrecht", "Wageningen"] = 1
    connection.loc["Wageningen", "Dordrecht"] = 1

    # connections with big clusters Chemelot

    connection.loc["Heerlen", "Chemelot"] = 1
    connection.loc["Chemelot", "Heerlen"] = 1

    connection.loc["Roermond", "Chemelot"] = 1
    connection.loc["Chemelot", "Roermond"] = 1

    connection.loc["Venlo", "Chemelot"] = 1
    connection.loc["Chemelot", "Venlo"] = 1

    # connections with big clusters Zeeland

    connection.loc["Dongen", "Zeeland"] = 1
    connection.loc["Zeeland", "Dongen"] = 1

    connection.loc["Oosterhout", "Zeeland"] = 1
    connection.loc["Zeeland", "Oosterhout"] = 1

    connection.loc["Dordrecht", "Zeeland"] = 1
    connection.loc["Zeeland", "Dordrecht"] = 1

    connection.loc["Betuwe", "Zeeland"] = 1
    connection.loc["Zeeland", "Betuwe"] = 1

    # connections with big clusters Rotterdam

    connection.loc["Dongen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Dongen"] = 1

    connection.loc["Oosterhout", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Oosterhout"] = 1

    connection.loc["Dordrecht", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Dordrecht"] = 1

    connection.loc["Betuwe", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Betuwe"] = 1

    connection.loc["Wageningen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Wageningen"] = 1

    # connections with big clusters North_Sea

    connection.loc["Dongen", "North_Sea"] = 1
    connection.loc["North_Sea", "Dongen"] = 1

    connection.loc["Oosterhout", "North_Sea"] = 1
    connection.loc["North_Sea", "Oosterhout"] = 1

    connection.loc["Dordrecht", "North_Sea"] = 1
    connection.loc["North_Sea", "Dordrecht"] = 1

    connection.loc["Betuwe", "North_Sea"] = 1
    connection.loc["North_Sea", "Betuwe"] = 1

    connection.loc["Wageningen", "North_Sea"] = 1
    connection.loc["North_Sea", "Wageningen"] = 1

    connection.loc["Arnhem", "North_Sea"] = 1
    connection.loc["North_Sea", "Arnhem"] = 1

    # connections with big clusters North_Netherlands

    connection.loc["East_Groningen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "East_Groningen"] = 1

    connection.loc["Betuwe", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Betuwe"] = 1

    connection.loc["Wageningen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Wageningen"] = 1

    connection.loc["Arnhem", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Arnhem"] = 1

    # connections with big clusters Zuidwending

    connection.loc["East_Groningen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "East_Groningen"] = 1

    connection.loc["Betuwe", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Betuwe"] = 1

    connection.loc["Wageningen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Wageningen"] = 1

    connection.loc["Arnhem", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Arnhem"] = 1

    # </editor-fold>

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP_big" / "connection.csv",
        sep=";")
    #connection.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "connection.csv", sep=";")

    # # Connection
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)
    # <editor-fold desc="Connections high">

    #EHB

    connection.loc["Rotterdam", "Zeeland"] = 1
    connection.loc["Zeeland", "Rotterdam"] = 1
    connection.loc["Rotterdam", "North_Sea"] = 1
    connection.loc["North_Sea", "Rotterdam"] = 1
    connection.loc["North_Sea", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "North_Sea"] = 1
    connection.loc["Zuidwending", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Chemelot"] = 1
    connection.loc["Chemelot", "Zuidwending"] = 1
    connection.loc["Zeeland", "Chemelot"] = 1
    connection.loc["Chemelot", "Zeeland"] = 1

    # small infrastructure
    connection.loc["Roermond", "Dongen"] = 1
    connection.loc["Dongen", "Roermond"] = 1

    connection.loc["Roermond", "Oosterhout"] = 1
    connection.loc["Oosterhout", "Roermond"] = 1

    connection.loc["Roermond", "Maastricht"] = 1
    connection.loc["Maastricht", "Roermond"] = 1

    connection.loc["Roermond", "Venlo"] = 1
    connection.loc["Venlo", "Roermond"] = 1

    connection.loc["Roermond", "Arnhem"] = 1
    connection.loc["Arnhem", "Roermond"] = 1

    connection.loc["Roermond", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Roermond"] = 1

    connection.loc["Roermond", "Betuwe"] = 1
    connection.loc["Betuwe", "Roermond"] = 1

    connection.loc["Roermond", "Heerlen"] = 1
    connection.loc["Heerlen", "Roermond"] = 1

    connection.loc["Roermond", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Roermond"] = 1

    connection.loc["Roermond", "Wageningen"] = 1
    connection.loc["Wageningen", "Roermond"] = 1

    connection.loc["Dongen", "Oosterhout"] = 1
    connection.loc["Oosterhout", "Dongen"] = 1

    connection.loc["Dongen", "Maastricht"] = 1
    connection.loc["Maastricht", "Dongen"] = 1

    connection.loc["Dongen", "Venlo"] = 1
    connection.loc["Venlo", "Dongen"] = 1

    connection.loc["Dongen", "Arnhem"] = 1
    connection.loc["Arnhem", "Dongen"] = 1

    connection.loc["Dongen", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Dongen"] = 1

    connection.loc["Dongen", "Betuwe"] = 1
    connection.loc["Betuwe", "Dongen"] = 1

    connection.loc["Dongen", "Heerlen"] = 1
    connection.loc["Heerlen", "Dongen"] = 1

    connection.loc["Dongen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Dongen"] = 1

    connection.loc["Dongen", "Wageningen"] = 1
    connection.loc["Wageningen", "Dongen"] = 1

    connection.loc["Oosterhout", "Maastricht"] = 1
    connection.loc["Maastricht", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Venlo"] = 1
    connection.loc["Venlo", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Arnhem"] = 1
    connection.loc["Arnhem", "Oosterhout"] = 1

    connection.loc["Oosterhout", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Betuwe"] = 1
    connection.loc["Betuwe", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Heerlen"] = 1
    connection.loc["Heerlen", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Oosterhout"] = 1

    connection.loc["Oosterhout", "Wageningen"] = 1
    connection.loc["Wageningen", "Oosterhout"] = 1

    connection.loc["Maastricht", "Venlo"] = 1
    connection.loc["Venlo", "Maastricht"] = 1

    connection.loc["Maastricht", "Arnhem"] = 1
    connection.loc["Arnhem", "Maastricht"] = 1

    connection.loc["Maastricht", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Maastricht"] = 1

    connection.loc["Maastricht", "Betuwe"] = 1
    connection.loc["Betuwe", "Maastricht"] = 1

    connection.loc["Maastricht", "Heerlen"] = 1
    connection.loc["Heerlen", "Maastricht"] = 1

    connection.loc["Maastricht", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Maastricht"] = 1

    connection.loc["Maastricht", "Wageningen"] = 1
    connection.loc["Wageningen", "Maastricht"] = 1

    connection.loc["Venlo", "Arnhem"] = 1
    connection.loc["Arnhem", "Venlo"] = 1

    connection.loc["Venlo", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Venlo"] = 1

    connection.loc["Venlo", "Betuwe"] = 1
    connection.loc["Betuwe", "Venlo"] = 1

    connection.loc["Venlo", "Heerlen"] = 1
    connection.loc["Heerlen", "Venlo"] = 1

    connection.loc["Venlo", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Venlo"] = 1

    connection.loc["Venlo", "Wageningen"] = 1
    connection.loc["Wageningen", "Venlo"] = 1

    connection.loc["Arnhem", "East_Groningen"] = 1
    connection.loc["East_Groningen", "Arnhem"] = 1

    connection.loc["Arnhem", "Betuwe"] = 1
    connection.loc["Betuwe", "Arnhem"] = 1

    connection.loc["Arnhem", "Heerlen"] = 1
    connection.loc["Heerlen", "Arnhem"] = 1

    connection.loc["Arnhem", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Arnhem"] = 1

    connection.loc["Arnhem", "Wageningen"] = 1
    connection.loc["Wageningen", "Arnhem"] = 1

    connection.loc["East_Groningen", "Betuwe"] = 1
    connection.loc["Betuwe", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Heerlen"] = 1
    connection.loc["Heerlen", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "East_Groningen"] = 1

    connection.loc["East_Groningen", "Wageningen"] = 1
    connection.loc["Wageningen", "East_Groningen"] = 1

    connection.loc["Betuwe", "Heerlen"] = 1
    connection.loc["Heerlen", "Betuwe"] = 1

    connection.loc["Betuwe", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Betuwe"] = 1

    connection.loc["Betuwe", "Wageningen"] = 1
    connection.loc["Wageningen", "Betuwe"] = 1

    connection.loc["Heerlen", "Dordrecht"] = 1
    connection.loc["Dordrecht", "Heerlen"] = 1

    connection.loc["Heerlen", "Wageningen"] = 1
    connection.loc["Wageningen", "Heerlen"] = 1

    connection.loc["Dordrecht", "Wageningen"] = 1
    connection.loc["Wageningen", "Dordrecht"] = 1

    # connections with big clusters Chemelot

    connection.loc["Heerlen", "Chemelot"] = 1
    connection.loc["Chemelot", "Heerlen"] = 1

    connection.loc["Roermond", "Chemelot"] = 1
    connection.loc["Chemelot", "Roermond"] = 1

    connection.loc["Venlo", "Chemelot"] = 1
    connection.loc["Chemelot", "Venlo"] = 1

    # connections with big clusters Zeeland

    connection.loc["Dongen", "Zeeland"] = 1
    connection.loc["Zeeland", "Dongen"] = 1

    connection.loc["Oosterhout", "Zeeland"] = 1
    connection.loc["Zeeland", "Oosterhout"] = 1

    connection.loc["Dordrecht", "Zeeland"] = 1
    connection.loc["Zeeland", "Dordrecht"] = 1

    connection.loc["Betuwe", "Zeeland"] = 1
    connection.loc["Zeeland", "Betuwe"] = 1

    # connections with big clusters Rotterdam

    connection.loc["Dongen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Dongen"] = 1

    connection.loc["Oosterhout", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Oosterhout"] = 1

    connection.loc["Dordrecht", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Dordrecht"] = 1

    connection.loc["Betuwe", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Betuwe"] = 1

    connection.loc["Wageningen", "Rotterdam"] = 1
    connection.loc["Rotterdam", "Wageningen"] = 1

    # connections with big clusters North_Sea

    connection.loc["Dongen", "North_Sea"] = 1
    connection.loc["North_Sea", "Dongen"] = 1

    connection.loc["Oosterhout", "North_Sea"] = 1
    connection.loc["North_Sea", "Oosterhout"] = 1

    connection.loc["Dordrecht", "North_Sea"] = 1
    connection.loc["North_Sea", "Dordrecht"] = 1

    connection.loc["Betuwe", "North_Sea"] = 1
    connection.loc["North_Sea", "Betuwe"] = 1

    connection.loc["Wageningen", "North_Sea"] = 1
    connection.loc["North_Sea", "Wageningen"] = 1

    connection.loc["Arnhem", "North_Sea"] = 1
    connection.loc["North_Sea", "Arnhem"] = 1

    # connections with big clusters North_Netherlands

    connection.loc["East_Groningen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "East_Groningen"] = 1

    connection.loc["Betuwe", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Betuwe"] = 1

    connection.loc["Wageningen", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Wageningen"] = 1

    connection.loc["Arnhem", "North_Netherlands"] = 1
    connection.loc["North_Netherlands", "Arnhem"] = 1

    # connections with big clusters Zuidwending

    connection.loc["East_Groningen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "East_Groningen"] = 1

    connection.loc["Betuwe", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Betuwe"] = 1

    connection.loc["Wageningen", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Wageningen"] = 1

    connection.loc["Arnhem", "Zuidwending"] = 1
    connection.loc["Zuidwending", "Arnhem"] = 1

    # </editor-fold>

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP_big" / "connection.csv",
        sep=";")
    # connection.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "connection.csv", sep=";")

    print("Connection:", connection)

    # Delete the template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "connection.csv")

    # # Distance
    distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "distance.csv", sep=";",
                           index_col=0)
    # <editor-fold desc="Distances">

    distance.loc["Roermond", "Dongen"] = 87
    distance.loc["Dongen", "Roermond"] = 87

    distance.loc["Roermond", "Oosterhout"] = 93
    distance.loc["Oosterhout", "Roermond"] = 93

    distance.loc["Roermond", "Maastricht"] = 43
    distance.loc["Maastricht", "Roermond"] = 43

    distance.loc["Roermond", "Venlo"] = 23
    distance.loc["Venlo", "Roermond"] = 23

    distance.loc["Roermond", "Arnhem"] = 88
    distance.loc["Arnhem", "Roermond"] = 88

    distance.loc["Roermond", "East_Groningen"] = 226
    distance.loc["East_Groningen", "Roermond"] = 226

    distance.loc["Roermond", "Betuwe"] = 87
    distance.loc["Betuwe", "Roermond"] = 87

    distance.loc["Roermond", "Heerlen"] = 35
    distance.loc["Heerlen", "Roermond"] = 35

    distance.loc["Roermond", "Dordrecht"] = 113
    distance.loc["Dordrecht", "Roermond"] = 113

    distance.loc["Roermond", "Wageningen"] = 89
    distance.loc["Wageningen", "Roermond"] = 89

    distance.loc["Roermond", "Rotterdam"] = 132
    distance.loc["Rotterdam", "Roermond"] = 132

    distance.loc["Roermond", "Zeeland"] = 152
    distance.loc["Zeeland", "Roermond"] = 152

    distance.loc["Roermond", "North_Sea"] = 160
    distance.loc["North_Sea", "Roermond"] = 160

    distance.loc["Roermond", "North_Netherlands"] = 258
    distance.loc["North_Netherlands", "Roermond"] = 258

    distance.loc["Roermond", "Chemelot"] = 28
    distance.loc["Chemelot", "Roermond"] = 28

    distance.loc["Roermond", "Zuidwending"] = 221
    distance.loc["Zuidwending", "Roermond"] = 221

    distance.loc["Dongen", "Oosterhout"] = 6
    distance.loc["Oosterhout", "Dongen"] = 6

    distance.loc["Dongen", "Maastricht"] = 100
    distance.loc["Maastricht", "Dongen"] = 100

    distance.loc["Dongen", "Venlo"] = 89
    distance.loc["Venlo", "Dongen"] = 89

    distance.loc["Dongen", "Arnhem"] = 77
    distance.loc["Arnhem", "Dongen"] = 77

    distance.loc["Dongen", "East_Groningen"] = 210
    distance.loc["East_Groningen", "Dongen"] = 210

    distance.loc["Dongen", "Betuwe"] = 50
    distance.loc["Betuwe", "Dongen"] = 50

    distance.loc["Dongen", "Heerlen"] = 109
    distance.loc["Heerlen", "Dongen"] = 109

    distance.loc["Dongen", "Dordrecht"] = 26
    distance.loc["Dordrecht", "Dongen"] = 26

    distance.loc["Dongen", "Wageningen"] = 63
    distance.loc["Wageningen", "Dongen"] = 63

    distance.loc["Dongen", "Rotterdam"] = 46
    distance.loc["Rotterdam", "Dongen"] = 46

    distance.loc["Dongen", "Zeeland"] = 77
    distance.loc["Zeeland", "Dongen"] = 77

    distance.loc["Dongen", "North_Sea"] = 90
    distance.loc["North_Sea", "Dongen"] = 90

    distance.loc["Dongen", "North_Netherlands"] = 240
    distance.loc["North_Netherlands", "Dongen"] = 240

    distance.loc["Dongen", "Chemelot"] = 93
    distance.loc["Chemelot", "Dongen"] = 93

    distance.loc["Dongen", "Zuidwending"] = 212
    distance.loc["Zuidwending", "Dongen"] = 212

    distance.loc["Oosterhout", "Maastricht"] = 105
    distance.loc["Maastricht", "Oosterhout"] = 105

    distance.loc["Oosterhout", "Venlo"] = 95
    distance.loc["Venlo", "Oosterhout"] = 95

    distance.loc["Oosterhout", "Arnhem"] = 81
    distance.loc["Arnhem", "Oosterhout"] = 81

    distance.loc["Oosterhout", "East_Groningen"] = 212
    distance.loc["East_Groningen", "Oosterhout"] = 212

    distance.loc["Oosterhout", "Betuwe"] = 54
    distance.loc["Betuwe", "Oosterhout"] = 54

    distance.loc["Oosterhout", "Heerlen"] = 115
    distance.loc["Heerlen", "Oosterhout"] = 115

    distance.loc["Oosterhout", "Dordrecht"] = 21
    distance.loc["Dordrecht", "Oosterhout"] = 21

    distance.loc["Oosterhout", "Wageningen"] = 66
    distance.loc["Wageningen", "Oosterhout"] = 66

    distance.loc["Oosterhout", "Rotterdam"] = 41
    distance.loc["Rotterdam", "Oosterhout"] = 41

    distance.loc["Oosterhout", "Zeeland"] = 72
    distance.loc["Zeeland", "Oosterhout"] = 72

    distance.loc["Oosterhout", "North_Sea"] = 88
    distance.loc["North_Sea", "Oosterhout"] = 88

    distance.loc["Oosterhout", "North_Netherlands"] = 241
    distance.loc["North_Netherlands", "Oosterhout"] = 241

    distance.loc["Oosterhout", "Chemelot"] = 99
    distance.loc["Chemelot", "Oosterhout"] = 99

    distance.loc["Oosterhout", "Zuidwending"] = 214
    distance.loc["Zuidwending", "Oosterhout"] = 214

    distance.loc["Maastricht", "Venlo"] = 66
    distance.loc["Venlo", "Maastricht"] = 66

    distance.loc["Maastricht", "Arnhem"] = 127
    distance.loc["Arnhem", "Maastricht"] = 127

    distance.loc["Maastricht", "East_Groningen"] = 268
    distance.loc["East_Groningen", "Maastricht"] = 268

    distance.loc["Maastricht", "Betuwe"] = 119
    distance.loc["Betuwe", "Maastricht"] = 119

    distance.loc["Maastricht", "Heerlen"] = 21
    distance.loc["Heerlen", "Maastricht"] = 21

    distance.loc["Maastricht", "Dordrecht"] = 126
    distance.loc["Dordrecht", "Maastricht"] = 126

    distance.loc["Maastricht", "Wageningen"] = 124
    distance.loc["Wageningen", "Maastricht"] = 124

    distance.loc["Maastricht", "Rotterdam"] = 146
    distance.loc["Rotterdam", "Maastricht"] = 146

    distance.loc["Maastricht", "Zeeland"] = 147
    distance.loc["Zeeland", "Maastricht"] = 147

    distance.loc["Maastricht", "North_Sea"] = 186
    distance.loc["North_Sea", "Maastricht"] = 186

    distance.loc["Maastricht", "North_Netherlands"] = 300
    distance.loc["North_Netherlands", "Maastricht"] = 300

    distance.loc["Maastricht", "Chemelot"] = 16
    distance.loc["Chemelot", "Maastricht"] = 16

    distance.loc["Maastricht", "Zuidwending"] = 264
    distance.loc["Zuidwending", "Maastricht"] = 264

    distance.loc["Venlo", "Arnhem"] = 71
    distance.loc["Arnhem", "Venlo"] = 71

    distance.loc["Venlo", "East_Groningen"] = 204
    distance.loc["East_Groningen", "Venlo"] = 204

    distance.loc["Venlo", "Betuwe"] = 77
    distance.loc["Betuwe", "Venlo"] = 77

    distance.loc["Venlo", "Heerlen"] = 55
    distance.loc["Heerlen", "Venlo"] = 55

    distance.loc["Venlo", "Dordrecht"] = 113
    distance.loc["Dordrecht", "Venlo"] = 113

    distance.loc["Venlo", "Wageningen"] = 75
    distance.loc["Wageningen", "Venlo"] = 75

    distance.loc["Venlo", "Rotterdam"] = 132
    distance.loc["Rotterdam", "Venlo"] = 132

    distance.loc["Venlo", "Zeeland"] = 161
    distance.loc["Zeeland", "Venlo"] = 161

    distance.loc["Venlo", "North_Sea"] = 150
    distance.loc["North_Sea", "Venlo"] = 150

    distance.loc["Venlo", "North_Netherlands"] = 236
    distance.loc["North_Netherlands", "Venlo"] = 236

    distance.loc["Venlo", "Chemelot"] = 50
    distance.loc["Chemelot", "Venlo"] = 50

    distance.loc["Venlo", "Zuidwending"] = 199
    distance.loc["Zuidwending", "Venlo"] = 199

    distance.loc["Arnhem", "East_Groningen"] = 143
    distance.loc["East_Groningen", "Arnhem"] = 143

    distance.loc["Arnhem", "Betuwe"] = 28
    distance.loc["Betuwe", "Arnhem"] = 28

    distance.loc["Arnhem", "Heerlen"] = 123
    distance.loc["Heerlen", "Arnhem"] = 123

    distance.loc["Arnhem", "Dordrecht"] = 86
    distance.loc["Dordrecht", "Arnhem"] = 86

    distance.loc["Arnhem", "Wageningen"] = 16
    distance.loc["Wageningen", "Arnhem"] = 16

    distance.loc["Arnhem", "Rotterdam"] = 98
    distance.loc["Rotterdam", "Arnhem"] = 98

    distance.loc["Arnhem", "Zeeland"] = 151
    distance.loc["Zeeland", "Arnhem"] = 151

    distance.loc["Arnhem", "North_Sea"] = 89
    distance.loc["North_Sea", "Arnhem"] = 89

    distance.loc["Arnhem", "North_Netherlands"] = 175
    distance.loc["North_Netherlands", "Arnhem"] = 175

    distance.loc["Arnhem", "Chemelot"] = 112
    distance.loc["Chemelot", "Arnhem"] = 112

    distance.loc["Arnhem", "Zuidwending"] = 142
    distance.loc["Zuidwending", "Arnhem"] = 142

    distance.loc["East_Groningen", "Betuwe"] = 162
    distance.loc["Betuwe", "East_Groningen"] = 162

    distance.loc["East_Groningen", "Heerlen"] = 260
    distance.loc["Heerlen", "East_Groningen"] = 260

    distance.loc["East_Groningen", "Dordrecht"] = 207
    distance.loc["Dordrecht", "East_Groningen"] = 207

    distance.loc["East_Groningen", "Wageningen"] = 152
    distance.loc["Wageningen", "East_Groningen"] = 152

    distance.loc["East_Groningen", "Rotterdam"] = 206
    distance.loc["Rotterdam", "East_Groningen"] = 206

    distance.loc["East_Groningen", "Zeeland"] = 270
    distance.loc["Zeeland", "East_Groningen"] = 270

    distance.loc["East_Groningen", "North_Sea"] = 152
    distance.loc["North_Sea", "East_Groningen"] = 152

    distance.loc["East_Groningen", "North_Netherlands"] = 32
    distance.loc["North_Netherlands", "East_Groningen"] = 32

    distance.loc["East_Groningen", "Chemelot"] = 252
    distance.loc["Chemelot", "East_Groningen"] = 252

    distance.loc["East_Groningen", "Zuidwending"] = 16
    distance.loc["Zuidwending", "East_Groningen"] = 16

    distance.loc["Betuwe", "Heerlen"] = 120
    distance.loc["Heerlen", "Betuwe"] = 120

    distance.loc["Betuwe", "Dordrecht"] = 58
    distance.loc["Dordrecht", "Betuwe"] = 58

    distance.loc["Betuwe", "Wageningen"] = 13
    distance.loc["Wageningen", "Betuwe"] = 13

    distance.loc["Betuwe", "Rotterdam"] = 70
    distance.loc["Rotterdam", "Betuwe"] = 70

    distance.loc["Betuwe", "Zeeland"] = 123
    distance.loc["Zeeland", "Betuwe"] = 123

    distance.loc["Betuwe", "North_Sea"] = 74
    distance.loc["North_Sea", "Betuwe"] = 74

    distance.loc["Betuwe", "North_Netherlands"] = 193
    distance.loc["North_Netherlands", "Betuwe"] = 193

    distance.loc["Betuwe", "Chemelot"] = 107
    distance.loc["Chemelot", "Betuwe"] = 107

    distance.loc["Betuwe", "Zuidwending"] = 163
    distance.loc["Zuidwending", "Betuwe"] = 163

    distance.loc["Heerlen", "Dordrecht"] = 136
    distance.loc["Dordrecht", "Heerlen"] = 136

    distance.loc["Heerlen", "Wageningen"] = 123
    distance.loc["Wageningen", "Heerlen"] = 123

    distance.loc["Heerlen", "Rotterdam"] = 156
    distance.loc["Rotterdam", "Heerlen"] = 156

    distance.loc["Heerlen", "Zeeland"] = 164
    distance.loc["Zeeland", "Heerlen"] = 164

    distance.loc["Heerlen", "North_Sea"] = 190
    distance.loc["North_Sea", "Heerlen"] = 190

    distance.loc["Heerlen", "North_Netherlands"] = 292
    distance.loc["North_Netherlands", "Heerlen"] = 292

    distance.loc["Heerlen", "Chemelot"] = 16
    distance.loc["Chemelot", "Heerlen"] = 16

    distance.loc["Heerlen", "Zuidwending"] = 254
    distance.loc["Zuidwending", "Heerlen"] = 254

    distance.loc["Dordrecht", "Wageningen"] = 70
    distance.loc["Wageningen", "Dordrecht"] = 70

    distance.loc["Dordrecht", "Rotterdam"] = 20
    distance.loc["Rotterdam", "Dordrecht"] = 20

    distance.loc["Dordrecht", "Zeeland"] = 66
    distance.loc["Zeeland", "Dordrecht"] = 66

    distance.loc["Dordrecht", "North_Sea"] = 71
    distance.loc["North_Sea", "Dordrecht"] = 71

    distance.loc["Dordrecht", "North_Netherlands"] = 235
    distance.loc["North_Netherlands", "Dordrecht"] = 235

    distance.loc["Dordrecht", "Chemelot"] = 120
    distance.loc["Chemelot", "Dordrecht"] = 120

    distance.loc["Dordrecht", "Zuidwending"] = 210
    distance.loc["Zuidwending", "Dordrecht"] = 210

    distance.loc["Wageningen", "Rotterdam"] = 82
    distance.loc["Rotterdam", "Wageningen"] = 82

    distance.loc["Wageningen", "Zeeland"] = 136
    distance.loc["Zeeland", "Wageningen"] = 136

    distance.loc["Wageningen", "North_Sea"] = 77
    distance.loc["North_Sea", "Wageningen"] = 77

    distance.loc["Wageningen", "North_Netherlands"] = 183
    distance.loc["North_Netherlands", "Wageningen"] = 183

    distance.loc["Wageningen", "Chemelot"] = 111
    distance.loc["Chemelot", "Wageningen"] = 111

    distance.loc["Wageningen", "Zuidwending"] = 152
    distance.loc["Zuidwending", "Wageningen"] = 152

    distance.loc["Rotterdam", "Zeeland"] = 64
    distance.loc["Zeeland", "Rotterdam"] = 64

    distance.loc["Rotterdam", "North_Sea"] = 61
    distance.loc["North_Sea", "Rotterdam"] = 61

    distance.loc["Rotterdam", "North_Netherlands"] = 232
    distance.loc["North_Netherlands", "Rotterdam"] = 232

    distance.loc["Rotterdam", "Chemelot"] = 140
    distance.loc["Chemelot", "Rotterdam"] = 140

    distance.loc["Rotterdam", "Zuidwending"] = 211
    distance.loc["Zuidwending", "Rotterdam"] = 211

    distance.loc["Zeeland", "North_Sea"] = 123
    distance.loc["North_Sea", "Zeeland"] = 123

    distance.loc["Zeeland", "North_Netherlands"] = 296
    distance.loc["North_Netherlands", "Zeeland"] = 296

    distance.loc["Zeeland", "Chemelot"] = 148
    distance.loc["Chemelot", "Zeeland"] = 148

    distance.loc["Zeeland", "Zuidwending"] = 275
    distance.loc["Zuidwending", "Zeeland"] = 275

    distance.loc["North_Sea", "North_Netherlands"] = 176
    distance.loc["North_Netherlands", "North_Sea"] = 176

    distance.loc["North_Sea", "Chemelot"] = 176
    distance.loc["Chemelot", "North_Sea"] = 176

    distance.loc["North_Sea", "Zuidwending"] = 160
    distance.loc["Zuidwending", "North_Sea"] = 160

    distance.loc["North_Netherlands", "Chemelot"] = 284
    distance.loc["Chemelot", "North_Netherlands"] = 284

    distance.loc["North_Netherlands", "Zuidwending"] = 41
    distance.loc["Zuidwending", "North_Netherlands"] = 41

    distance.loc["Chemelot", "Zuidwending"] = 248
    distance.loc["Zuidwending", "Chemelot"] = 248

    # </editor-fold>

    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP_big" / "distance.csv",
        sep=";")
    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP_big" / "distance.csv",
        sep=";")
    #distance.to_csv(input_data_path / "period1" / "network_topology" / "new" / "hydrogenTruck_highP" / "distance.csv", sep=";")
    print("Distance:", distance)

    # Delete the template
    os.remove(input_data_path / "period1" / "network_topology" / "new" / "distance.csv")

