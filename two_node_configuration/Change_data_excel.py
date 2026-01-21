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

    small_demand = total_demand / (1 + ratio)
    large_demand = total_demand - small_demand


    hydrogen_data = {
        "Large_cluster": large_demand,
        "Small_cluster": small_demand
    }

    nodes_small_cluster = ["Small_cluster"]
    node_big_cluster = ["Large_cluster"]

    timesteps = 8760

    # input_data_path = Path(input_data_path)

    demand_example_folder = (input_data_path.parents[4] / "data")
    demand_example = demand_example_folder / "Example_hydrogen_demand.xlsx"

    df_example = pd.read_excel(
        demand_example,
        usecols=[0, 1, 2],
        header=0,
        thousands=','
    )

    if len(df_example) < 8760:
        raise ValueError(f"Example series too short: {len(df_example)} < 8760")
    df_example = df_example.iloc[:8760]

    # Extract each column
    chemicals_series = df_example.iloc[:, 0].astype(float).to_numpy()
    refineries_series = df_example.iloc[:, 1].astype(float).to_numpy()
    other_series = df_example.iloc[:, 2].astype(float).to_numpy()

    # Calculate means
    chemicals_mean = float(np.mean(chemicals_series))
    refineries_mean = float(np.mean(refineries_series))
    other_mean = float(np.mean(other_series))

    # Validate means
    if chemicals_mean == 0 or refineries_mean == 0 or other_mean == 0:
        raise ValueError("One or more mean values are 0, cannot normalize.")

    # Create fluctuation factors
    Chemicals_fluct_factors = chemicals_series / chemicals_mean
    Rafineries_fluct_factors = refineries_series / refineries_mean
    Other_fluct_factors = other_series / other_mean

    for node in nodes_small_cluster:
        file_path = input_data_path / f"data_network_{node}.xlsx"

        # hydrogen_use_TWh = hydrogen_data_small[node]["Hydrogen use (TWh)"]
        hydrogen_use_TWh = hydrogen_data[node]

        # Average MW
        average_MW = hydrogen_use_TWh * 1e6 / timesteps  # TWh -> GWh -> MW
        #
        # # ±15% fluctuations
        # seed = 42
        # rng = np.random.default_rng(seed)
        # fluctuation = rng.normal(loc=0.0, scale=0.15 * average_MW, size=timesteps)
        # hydrogen_profile = average_MW + fluctuation

        # Apply Other fluctuation factors
        hydrogen_profile = average_MW * Other_fluct_factors

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

        # seed = 42
        # rng = np.random.default_rng(seed)
        # fluctuation = rng.normal(loc=0.0, scale=0.10 * average_MW, size=timesteps)
        # hydrogen_profile = average_MW + fluctuation

        # Apply Chemicals fluctuation factors
        hydrogen_profile = average_MW * Chemicals_fluct_factors

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


