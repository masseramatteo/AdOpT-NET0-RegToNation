import json
from pathlib import Path
import os
import pandas as pd
import numpy as np


def define_hydrogen_pipeline1(input_data_path):

    with open(input_data_path / "period1" / "network_data"/ "hydrogenPipelineOnshore_lowP_big.json", "r") as json_file:
        network_data = json.load(json_file)

    network_data["Performance"]["rated_capacity"] = 500

    network_data["Economics"]["gamma1"] = 0
    network_data["Economics"]["gamma3"] = 0

    network_data["Economics"]["gamma2"] = 0.50E6
    network_data["Economics"]["gamma4"] = 500

    network_data["Performance"]["bidirectional_network_precise"] = 1

    with open(input_data_path / "period1" / "network_data"/ "hydrogenPipelineOnshore_lowP_big.json", "w") as json_file:
        json.dump(network_data, json_file, indent=4)

    with open(input_data_path / "period1" / "network_data" / "hydrogenPipelineOnshore_highP_big.json", "r") as json_file:
        network_data = json.load(json_file)

    network_data["Performance"]["rated_capacity"] = 1.2

    network_data["Economics"]["gamma1"] = 0
    network_data["Economics"]["gamma3"] = 0

    network_data["Economics"]["gamma2"] = 1.250E6
    network_data["Economics"]["gamma4"] = 1200

    network_data["Performance"]["bidirectional_network_precise"] = 1

    with open(input_data_path / "period1" / "network_data" / "hydrogenPipelineOnshore_highP_big.json", "w") as json_file:
        json.dump(network_data, json_file, indent=4)

def define_hydrogen_pipeline2(input_data_path):

    with open(input_data_path / "period1" / "network_data"/ "hydrogenPipelineOnshore_lowP.json", "r") as json_file:
        network_data = json.load(json_file)

    network_data["size_min"] = 0
    network_data["size_max"] = 250

    network_data["Economics"]["gamma1"] = 100000
    network_data["Economics"]["gamma3"] = 0

    network_data["Economics"]["gamma2"] = 0
    network_data["Economics"]["gamma4"] = 4000

    network_data["Performance"]["bidirectional_network"] = 1
    network_data["Performance"]["bidirectional_network_precise"] = 1

    network_data["Performance"]["min_transport"] = 0

    with open(input_data_path / "period1" / "network_data"/ "hydrogenPipelineOnshore_lowP.json", "w") as json_file:
        json.dump(network_data, json_file, indent=4)

    with open(input_data_path / "period1" / "network_data" / "hydrogenPipelineOnshore_highP.json", "r") as json_file:
        network_data = json.load(json_file)

    network_data["size_min"] = 0
    network_data["size_max"] = 1000

    network_data["Economics"]["gamma1"] = 200000
    network_data["Economics"]["gamma3"] = 0

    network_data["Economics"]["gamma2"] = 0
    network_data["Economics"]["gamma4"] = 4400

    network_data["Performance"]["bidirectional_network"] = 1
    network_data["Performance"]["bidirectional_network_precise"] = 1

    network_data["Performance"]["min_transport"] = 0

    with open(input_data_path / "period1" / "network_data" / "hydrogenPipelineOnshore_highP.json", "w") as json_file:
        json.dump(network_data, json_file, indent=4)

def define_hydrogen_storage(input_data_path):
    with open(input_data_path / "period1" / "node_data" / "STORAGE" / "technology_data" / "Storage_H2_Cavern.json",
              "r") as json_file:
        cavern_data = json.load(json_file)

    cavern_data["Performance"]["allow_only_one_direction"] = 1
    cavern_data["Performance"]["allow_only_one_direction_precise"] = 1
    cavern_data["Flexibility"]["charge_rate"] = 0.5
    cavern_data["Flexibility"]["discharge_rate"] = 0.5

    with open(input_data_path / "period1" / "node_data" / "STORAGE" / "technology_data" / "Storage_H2_Cavern.json",
              "w") as json_file:
        json.dump(cavern_data, json_file, indent=4)

    for node in ["SMALL1", "SMALL2", "SMALL3", "SMALL4"]:
        with open(input_data_path / "period1" / "node_data" / node / "technology_data" / "Storage_H2_lowP.json",
                  "r") as json_file:
            storage_data = json.load(json_file)

        storage_data["Performance"]["allow_only_one_direction"] = 1
        storage_data["Performance"]["allow_only_one_direction_precise"] = 1
        storage_data["Flexibility"]["charge_rate"] = 0.8
        storage_data["Flexibility"]["discharge_rate"] = 0.8
        storage_data["Performance"]["performance"]["eta_in"] = 0.95
        storage_data["Performance"]["performance"]["eta_out"] = 0.95

        with open(input_data_path / "period1" / "node_data" / node / "technology_data" / "Storage_H2_lowP.json",
                  "w") as json_file:
            json.dump(storage_data, json_file, indent=4)

    for node in ["BIG1", "BIG2"]:
        with open(input_data_path / "period1" / "node_data" / node / "technology_data" / "Storage_H2_highP.json",
                  "r") as json_file:
            storage_data = json.load(json_file)

        storage_data["Performance"]["allow_only_one_direction"] = 1
        storage_data["Performance"]["allow_only_one_direction_precise"] = 1
        storage_data["Flexibility"]["charge_rate"] = 0.9
        storage_data["Flexibility"]["discharge_rate"] = 0.9
        storage_data["Performance"]["performance"]["eta_in"] = 0.95
        storage_data["Performance"]["performance"]["eta_out"] = 0.95

        with open(input_data_path / "period1" / "node_data" / node / "technology_data" / "Storage_H2_highP.json",
                  "w") as json_file:
            json.dump(storage_data, json_file, indent=4)

def define_electrolyzers(input_data_path):
    for node in ["BIG1", "BIG2"]:
        with open(input_data_path / "period1" / "node_data" / node / "technology_data" / "Electrolyzer_big.json",
                  "r") as json_file:
            electrolyzer_data = json.load(json_file)

        electrolyzer_data["size_min"] = 200
        electrolyzer_data["size_max"] = 3000

        with open(input_data_path / "period1" / "node_data" / node / "technology_data" / "Electrolyzer_big.json",
                  "w") as json_file:
            json.dump(electrolyzer_data, json_file, indent=4)

    for node in ["SMALL1", "SMALL2", "SMALL3", "SMALL4"]:
        with open(input_data_path / "period1" / "node_data" / node / "technology_data" / "Electrolyzer_small.json",
                  "r") as json_file:
            electrolyzer_data = json.load(json_file)

        electrolyzer_data["size_max"] = 400

        with open(input_data_path / "period1" / "node_data" / node / "technology_data" / "Electrolyzer_small.json",
                  "w") as json_file:
            json.dump(electrolyzer_data, json_file, indent=4)

