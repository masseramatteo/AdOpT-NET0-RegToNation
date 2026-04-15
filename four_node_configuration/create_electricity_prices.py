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

    approach = params.get("electricity_profile_approach", "fixed_profile")
    base_price = float(params["electricity_price_avg"])
    price_min = float(params.get("electricity_price_min", 0))
    price_max = float(params.get("electricity_price_max", 400))
    window_size = int(params.get("smoothing_window", 0))  # 0 = no smoothing

    # ---- build fluctuation factors (mean = 1) ----
    if approach == "fixed_profile":
        electricity_example_folder = BASE_PATH / "data"
        electricity_example = electricity_example_folder / "Example_electricity_prices.xlsx"

        example_series = pd.read_excel(
            electricity_example,
            header=0
        )["E-MBT (ammonia)"].astype(float).to_numpy()  # E-MBT (ammonia)  Last paper 2040

        if len(example_series) < 8760:
            raise ValueError(f"Example series too short: {len(example_series)} < 8760")
        example_series = example_series[:8760]

        example_mean = float(np.mean(example_series))
        if example_mean == 0:
            raise ValueError("Mean of example series is 0, cannot normalize.")
        series = example_series / example_mean * base_price  # mean = base_price

    elif approach == "changing_profile":
        from price_profile_generation import (
            load_electricity_price_profile, fit_electricty_price_trends, generate_electricity_price_profile
        )

        data_dir = BASE_PATH / "data" / "european_wholesale_electricity_price_data_hourly"
        p = load_electricity_price_profile("Netherlands", data_dir=str(data_dir))
        p, fit_params = fit_electricty_price_trends(p, 2024)

        scaling_factors = {
            "trend": 1.0,
            "weekly_factor": 1.0,
            "hourly_factor": 1.0,
            "overall_factor": 1.0,
        }

        p_gen = generate_electricity_price_profile(fit_params, base_price, scaling_factors, 2024)
        series = p_gen["p"].to_numpy()[:8760]  # mean ≈ base_price

    else:
        raise ValueError(f"Unknown electricity_profile_approach: '{approach}'")

    # ---- rescale to target std in EUR/MWh ----
    target_std = float(params["electricity_standard_dev"])
    series_mean = float(np.mean(series))
    current_std = float(np.std(series))
    print(f"Input series:   avg={series_mean:.2f} EUR/MWh, std={current_std:.2f} EUR/MWh")
    series = (series - series_mean) * (target_std / current_std) + series_mean
    print(f"Output profile: avg={np.mean(series):.2f} EUR/MWh, std={np.std(series):.2f} EUR/MWh")
    fluct_factors = series / float(np.mean(series))  # mean = 1

    # ---- apply to each node ----
    for node in nodes:
        if node.startswith("Small"):
            mode = "fluctuating"
            base = base_price

        elif node.startswith("Large"):
            mode = "fluctuating"
            base = base_price

        elif node == "STORAGE":
            mode = "fluctuating"
            base = base_price

        else:
            mode = "fluctuating"
            base = base_price

        if mode == "constant":
            electricity_prices = np.full(8760, base, dtype=float)
        else:
            electricity_prices = base * fluct_factors  # same trend, new mean (=base)

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
        print(f"Created electricity prices for {node} (mode={mode}, approach={approach}): "
              f"avg={np.mean(electricity_prices):.2f}, "
              f"min={np.min(electricity_prices):.2f}, "
              f"max={np.max(electricity_prices):.2f}, "
              f"max Δh={np.max(price_changes):.2f}, "
              f"avg Δh={np.mean(price_changes):.2f}")

