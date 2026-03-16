"""
Create electricity price files for each node with 8760 hourly values
- Small nodes: Higher congestion rate with daily variations (cheaper mid-day)
- Big nodes: More stable pricing with wind patterns (cheaper at night)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(__file__).parent

def create_electricity_prices(input_data_path, nodes, params):
    """
    Create electricity price data for each node.

    params keys consigliate:
      - electricity_price_avg: float
      - electricity_price_mode: "constant" oppure "fluctuating"
      - electricity_price_min / electricity_price_max (opzionali)
      - smoothing_window (opzionale)
    """

    input_data_path = Path(input_data_path)

    electricity_example_folder = BASE_PATH / "data"
    electricity_example = electricity_example_folder/"Example_electricity_prices.xlsx"

    example_series = pd.read_excel(
        electricity_example,
        usecols=[0],
        header=0
    ).iloc[:, 0].astype(float).to_numpy()

    if len(example_series) < 8760:
        raise ValueError(f"Example series too short: {len(example_series)} < 8760")
    example_series = example_series[:8760]

    example_mean = float(np.mean(example_series))
    if example_mean == 0:
        raise ValueError("Mean of example series is 0, cannot normalize.")
    example_fluct_factors = example_series / example_mean  # mean = 1

    base_price_small = float(params["electricity_price_avg"])
    base_price_big = float(params["electricity_price_avg"])

    price_min = float(params.get("electricity_price_min", 0))
    price_max = float(params.get("electricity_price_max", 400))
    window_size = int(params.get("smoothing_window", 0))  # 0 = no smoothing

    for node in nodes:
        # ---- manual mode selection (keep your structure) ----
        if node.startswith("SMALL"):
            mode = "fluctuating"  # change manually if you want
            base = base_price_small

        elif node.startswith("BIG"):
            mode = "constant"  # change manually if you want
            base = base_price_big

        elif node == "STORAGE":
            mode = "constant"  # change manually if you want
            base = base_price_big  # or params.get("storage_price_avg", base_price_big)

        else:
            mode = "constant"
            base = float(params["electricity_price_avg"])

        # ---- build series from chosen mode ----
        if mode == "constant":
            electricity_prices = np.full(8760, base, dtype=float)
        else:
            electricity_prices = base * example_fluct_factors  # same trend, new mean (=base)

        # smoothing (optional)
        if window_size and window_size > 1:
            electricity_prices = np.convolve(
                electricity_prices,
                np.ones(window_size) / window_size,
                mode="same"
            )

        # bounds
        electricity_prices = np.clip(electricity_prices, price_min, price_max)

        # export
        price_data = pd.DataFrame({
            "Hour": np.arange(1, 8761),
            "Electricity_Price_EUR_MWh": electricity_prices
        })
        output_file = input_data_path / f"electricity_prices_{node}.xlsx"
        price_data.to_excel(output_file, index=False)

        # stats
        price_changes = np.abs(np.diff(electricity_prices))
        print(f"Created electricity prices for {node} (mode={mode}): "
              f"avg={np.mean(electricity_prices):.2f}, "
              f"min={np.min(electricity_prices):.2f}, "
              f"max={np.max(electricity_prices):.2f}, "
              f"max Δh={np.max(price_changes):.2f}, "
              f"avg Δh={np.mean(price_changes):.2f}")
