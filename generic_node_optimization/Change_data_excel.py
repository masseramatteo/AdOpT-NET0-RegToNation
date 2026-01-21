from pathlib import Path
import os
import pandas as pd
import numpy as np

from openpyxl import load_workbook

def define_excel_data(input_data_path, params):
    input_data_path.mkdir(parents=True, exist_ok=True)

    # Extract parameters

    total_demand = params["total_demand_TWh"]
    ratio = params["demand_level_ratio"]
    el_price_avg = params["electricity_price_avg"]
    el_avail_small = params["electricity_availability_small"]
    hydrogen_data_small = params["hydrogen_demand_small"]
    hydrogen_data_big = params["hydrogen_demand_big"]

    # Calculate demand for cluster
    # BIG clusters: 2 nodes with the ratio of total demand
    # SMALL clusters: 4 nodes with 1 parte each
    # Total parts = 2*ratio + 4*1
    total_parts = 2 * ratio + 4
    demand_per_part = total_demand / total_parts

    # BIG: one is double the other (2:1 ratio)
    big_total = 2 * ratio * demand_per_part
    big1_demand = (2/3) * big_total  # BIG1 with 2 parts
    big2_demand = (1/3) * big_total  # BIG2 with 1 part

    # SMALL: equally divided
    small_demand_each = demand_per_part
    small_demand_1 = small_demand_each * (1+0.5)
    small_demand_2 = small_demand_each * (1-0.1)
    small_demand_3 = small_demand_each * (1-0.4)
    small_demand_4 = small_demand_each * (1+0.0)

    hydrogen_data = {
        "BIG1": big1_demand,
        "BIG2": big2_demand,
        "SMALL1": small_demand_1,
        "SMALL2": small_demand_2,
        "SMALL3": small_demand_3,
        "SMALL4": small_demand_4,
    }


    # {
    #     "SMALL1": {"Hydrogen use (TWh)": 0.95, "Capacity (MW)": 174},
    #     "SMALL2": {"Hydrogen use (TWh)": 0.64, "Capacity (MW)": 117},
    #     "SMALL3": {"Hydrogen use (TWh)": 0.48, "Capacity (MW)": 87},
    #     "SMALL4": {"Hydrogen use (TWh)": 0.18, "Capacity (MW)": 33},
    # }

    # hydrogen_data_big = {
    #     "BIG1": {"Hydrogen use (TWh)": 15},
    #     "BIG2": {"Hydrogen use (TWh)": 5},
    # }

    nodes_small_cluster = ["SMALL1", "SMALL2", "SMALL3", "SMALL4"]
    node_big_cluster = ["BIG1","BIG2"]
    node_storage = ["STORAGE"]

    timesteps = 8760

    for node in node_storage:
        file_path = input_data_path / f"data_network_{node}.xlsx"

        df = pd.DataFrame({
            'Hydrogen': [0] * timesteps,
            'Electricity': [0] * timesteps
        })

        df.to_excel(file_path, index=False)

        # File data_pressure_connections
        file_path_pressure = input_data_path / f"data_pressure_connections_{node}.xlsx"
        df_pressure = pd.DataFrame({
            'A': ['demand', 'export', 'import', 'generic_production'],
            'hydrogen': [25, 40, 40, 0]
        })
        df_pressure.to_excel(file_path_pressure, index=False)

    for node in nodes_small_cluster:
        file_path = input_data_path / f"data_network_{node}.xlsx"

        # hydrogen_use_TWh = hydrogen_data_small[node]["Hydrogen use (TWh)"]
        hydrogen_use_TWh = hydrogen_data[node]
        capacity_MW = hydrogen_data_small[node]["Capacity (MW)"]

        # Average MW
        average_MW = hydrogen_use_TWh * 1e6 / timesteps  # TWh -> GWh -> MW

        # ±15% fluctuations
        seed = 42
        rng = np.random.default_rng(seed)
        fluctuation = rng.normal(loc=0.0, scale=0.15 * average_MW, size=timesteps)
        hydrogen_profile = average_MW #+ fluctuation

        # #  to keep lower than capacity
        # scale_factor = capacity_MW / hydrogen_profile.max()
        # hydrogen_profile *= scale_factor

        # Crea DataFrame
        df = pd.DataFrame({
            'Hydrogen': hydrogen_profile,
            'Electricity': [0] * timesteps
        })

        df.to_excel(file_path, index=False)

        # File data_pressure_connections
        file_path_pressure = input_data_path / f"data_pressure_connections_{node}.xlsx"
        df_pressure = pd.DataFrame({
            'A': ['demand', 'export', 'import', 'generic_production'],
            'hydrogen': [25, 40, 40, 0]
        })
        df_pressure.to_excel(file_path_pressure, index=False)

    for node in node_big_cluster:
        file_path = input_data_path / f"data_network_{node}.xlsx"

        hydrogen_use_TWh = hydrogen_data[node]
        average_MW = hydrogen_use_TWh * 1e6 / timesteps

        seed = 42
        rng = np.random.default_rng(seed)
        fluctuation = rng.normal(loc=0.0, scale=0.10 * average_MW, size=timesteps)
        hydrogen_profile = average_MW #+ fluctuation

        df = pd.DataFrame({
            'Hydrogen': hydrogen_profile,
            'Electricity': [0] * timesteps
        })

        # File data_pressure_connections
        file_path_pressure = input_data_path / f"data_pressure_connections_{node}.xlsx"
        df_pressure = pd.DataFrame({
            'A': ['demand', 'export', 'import', 'generic_production'],
            'hydrogen': [25, 40, 40, 0]
        })
        df_pressure.to_excel(file_path_pressure, index=False)

        df.to_excel(file_path, index=False)


