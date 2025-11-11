import json
from pathlib import Path
import os
import pandas as pd
import numpy as np


def define_hydrogen_pipeline1(input_data_path):

    with open(input_data_path / "period1" / "network_data"/ "hydrogenPipelineOnshore_lowP_big.json", "r") as json_file:
        network_data = json.load(json_file)

    # network_data["Performance"]["rated_capacity"] = 500
    #
    # network_data["Economics"]["gamma1"] = 188512200
    # network_data["Economics"]["gamma3"] = 150000
    #
    # network_data["Economics"]["gamma2"] = 0
    # network_data["Economics"]["gamma4"] = 0

    network_data["Performance"]["rated_capacity"] = 100

    network_data["Economics"]["gamma1"] = 0
    network_data["Economics"]["gamma3"] = 160000

    network_data["Economics"]["gamma2"] = 0
    network_data["Economics"]["gamma4"] = 0

    network_data["Performance"]["bidirectional_network"] = 1
    network_data["Performance"]["bidirectional_network_precise"] = 1

    with open(input_data_path / "period1" / "network_data"/ "hydrogenPipelineOnshore_lowP_big.json", "w") as json_file:
        json.dump(network_data, json_file, indent=4)

    with open(input_data_path / "period1" / "network_data" / "hydrogenPipelineOnshore_highP_big.json", "r") as json_file:
        network_data = json.load(json_file)

    # network_data["Performance"]["rated_capacity"] = 1200
    #
    # network_data["Economics"]["gamma1"] = 297325700
    # network_data["Economics"]["gamma3"] = 600000
    #
    # network_data["Economics"]["gamma2"] = 0
    # network_data["Economics"]["gamma4"] = 0

    network_data["Performance"]["rated_capacity"] = 1000

    network_data["Economics"]["gamma1"] = 0
    network_data["Economics"]["gamma3"] = 700000

    network_data["Economics"]["gamma2"] = 0
    network_data["Economics"]["gamma4"] = 0

    network_data["Performance"]["bidirectional_network"] = 1
    network_data["Performance"]["bidirectional_network_precise"] = 1

    with open(input_data_path / "period1" / "network_data" / "hydrogenPipelineOnshore_highP_big.json", "w") as json_file:
        json.dump(network_data, json_file, indent=4)

def define_hydrogen_pipeline2(input_data_path):

    with open(input_data_path / "period1" / "network_data"/ "hydrogenPipelineOnshore_lowP.json", "r") as json_file:
        network_data = json.load(json_file)

    network_data["size_min"] = 100
    network_data["size_max"] = 250

    network_data["Economics"]["gamma1"] = 0
    network_data["Economics"]["gamma3"] = 0

    network_data["Economics"]["gamma2"] = 0
    network_data["Economics"]["gamma4"] = 1700

    network_data["Performance"]["bidirectional_network"] = 0
    network_data["Performance"]["bidirectional_network_precise"] = 0

    # network_data["Performance"]["min_transport"] = 0.1429

    with open(input_data_path / "period1" / "network_data"/ "hydrogenPipelineOnshore_lowP.json", "w") as json_file:
        json.dump(network_data, json_file, indent=4)

    with open(input_data_path / "period1" / "network_data" / "hydrogenPipelineOnshore_highP.json", "r") as json_file:
        network_data = json.load(json_file)

    network_data["size_min"] = 500
    network_data["size_max"] = 1000

    network_data["Economics"]["gamma1"] = 0
    network_data["Economics"]["gamma3"] = 0

    network_data["Economics"]["gamma2"] = 0
    network_data["Economics"]["gamma4"] = 700

    network_data["Performance"]["bidirectional_network"] = 0
    network_data["Performance"]["bidirectional_network_precise"] = 0

    # network_data["Performance"]["min_transport"] = 0.1429

    with open(input_data_path / "period1" / "network_data" / "hydrogenPipelineOnshore_highP.json", "w") as json_file:
        json.dump(network_data, json_file, indent=4)

def define_hydrogen_storage(input_data_path):
    with open(input_data_path / "period1" / "node_data" / "Zuidwending" / "technology_data" / "Storage_H2_Cavern.json",
              "r") as json_file:
        cavern_data = json.load(json_file)

    cavern_data["Performance"]["allow_only_one_direction"] = 1
    cavern_data["Performance"]["allow_only_one_direction_precise"] = 1
    cavern_data["Flexibility"]["charge_rate"] = 0.5
    cavern_data["Flexibility"]["discharge_rate"] = 0.5

    with open(input_data_path / "period1" / "node_data" / "Zuidwending" / "technology_data" / "Storage_H2_Cavern.json",
              "w") as json_file:
        json.dump(cavern_data, json_file, indent=4)
