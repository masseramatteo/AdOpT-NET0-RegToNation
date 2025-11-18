"""
Create electricity price files for each node with 8760 hourly values
- Small nodes: Higher congestion rate with daily variations (cheaper mid-day)
- Big nodes: More stable pricing with wind patterns (cheaper at night)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_electricity_prices(input_data_path, nodes, params):
    """
    Create electricity price data for each node with smooth, continuous variations

    Args:
        input_data_path: Path to input data folder
        nodes: List of node names
    """

    # Create data directory if it doesn't exist

    # Create hourly time series for full year (8760 hours)
    hours = np.arange(8760)
    days = hours // 24
    hour_of_day = hours % 24

    # Base prices
    base_price_small = params["electricity_price_avg"]  # EUR/MWh for small nodes
    base_price_big = params["electricity_price_avg"]    # EUR/MWh for big nodes (lower base due to better grid connection)

    for node in nodes:
        if node.startswith("SMALL"):
            # Small nodes: Higher congestion with smooth daily patterns
            # Create smooth daily variation using sinusoidal function
            # Peak price at 19h (evening), lowest at 13h (midday solar peak)
            daily_cycle = 0.85 + 0.3 * np.cos(2 * np.pi * (hour_of_day - 13) / 24)

            # Weekly pattern - weekends slightly cheaper (smooth transition)
            day_of_week = (days % 7)
            weekend_factor = 0.95 + 0.05 * np.cos(2 * np.pi * day_of_week / 7)

            # Seasonal variation - smooth winter peak
            day_of_year = days % 365
            seasonal_factor = 0.9 + 0.2 * np.cos(2 * np.pi * (day_of_year - 15) / 365)  # Peak in January

            # Add smooth congestion variations (longer cycles)
            congestion_cycle = 1 + 0.08 * np.sin(2 * np.pi * hours / (24 * 7))  # Weekly cycle
            congestion_cycle += 0.05 * np.sin(2 * np.pi * hours / (24 * 3))      # 3-day cycle

            # Gentle random walk for market variations (continuous, no jumps)
            np.random.seed(hash(node) % 2**32)
            random_walk = np.cumsum(np.random.normal(0, 0.002, 8760))  # Small incremental changes
            smooth_noise = 1 + 0.03 * np.tanh(random_walk)  # Bounded between ±3%

            electricity_prices = base_price_small * daily_cycle * weekend_factor * seasonal_factor * congestion_cycle * smooth_noise

        elif node.startswith("BIG"):
            # Big nodes: More stable with smooth wind patterns
            # Wind pattern - stronger at night, weaker in afternoon
            wind_cycle = 0.9 + 0.2 * np.cos(2 * np.pi * (hour_of_day - 15) / 24)  # Cheapest at 3am, peak at 3pm

            # Smoother seasonal variation
            day_of_year = days % 365
            seasonal_factor = 0.95 + 0.1 * np.cos(2 * np.pi * (day_of_year - 15) / 365)

            # Grid stability factor - very smooth variations
            stability_cycle = 1 + 0.03 * np.sin(2 * np.pi * hours / (24 * 14))  # Bi-weekly cycle

            # Minimal random variations for stable grid
            np.random.seed(hash(node) % 2**32)
            random_walk = np.cumsum(np.random.normal(0, 0.001, 8760))
            smooth_noise = 1 + 0.015 * np.tanh(random_walk)  # ±1.5% bounded variation

            electricity_prices = base_price_big * wind_cycle * seasonal_factor * stability_cycle * smooth_noise

        elif node == "STORAGE":
            # Storage node: Smooth arbitrage-based pricing
            # Follows renewable production patterns
            storage_cycle = 0.9 + 0.2 * np.cos(2 * np.pi * (hour_of_day - 14) / 24)  # Cheapest at 2pm, peak at 2am

            # Moderate seasonal variation
            day_of_year = days % 365
            seasonal_factor = 0.93 + 0.14 * np.cos(2 * np.pi * (day_of_year - 15) / 365)

            # Storage optimization cycles
            arbitrage_cycle = 1 + 0.04 * np.sin(2 * np.pi * hours / (24 * 5))  # 5-day optimization cycle

            # Moderate random variations
            np.random.seed(hash(node) % 2**32)
            random_walk = np.cumsum(np.random.normal(0, 0.0015, 8760))
            smooth_noise = 1 + 0.025 * np.tanh(random_walk)  # ±2.5% bounded variation

        # Apply smoothing filter to ensure no sudden jumps
        # Use a moving average to further smooth the prices
        window_size = 3  # 3-hour smoothing window
        electricity_prices_smooth = np.convolve(electricity_prices, np.ones(window_size)/window_size, mode='same')

        # Ensure prices stay within reasonable bounds
        electricity_prices_smooth = np.clip(electricity_prices_smooth, 50, 400)  # Min 50, Max 400 EUR/MWh

        # Create DataFrame
        price_data = pd.DataFrame({
            'Hour': range(1, 8761),
            'Electricity_Price_EUR_MWh': electricity_prices_smooth
        })

        # Save to Excel file
        output_file = input_data_path / f"electricity_prices_{node}.xlsx"
        price_data.to_excel(output_file, index=False)

        # Calculate price variation metrics
        price_changes = np.abs(np.diff(electricity_prices_smooth))
        max_hourly_change = np.max(price_changes)
        avg_hourly_change = np.mean(price_changes)

        # Print statistics
        print(f"Created smooth electricity prices for {node}:")
