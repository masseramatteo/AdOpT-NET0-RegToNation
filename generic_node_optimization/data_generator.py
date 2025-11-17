"""
Data Generator for Parametric Studies
Generates Excel files with demand profiles based on parametric study parameters
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_excel_data(input_data_path, params):
    """
    Genera i file Excel con dati di domanda basati sui parametri dello studio

    Parametri:
    - total_demand_TWh: domanda totale annuale in TWh
    - demand_level_ratio: ratio tra big e small clusters (big/small)
    - electricity_price_avg: prezzo medio elettricità EUR/MWh
    - electricity_availability_small: limite import elettricità per small nodes in MW
    """

    input_data_path = Path(input_data_path)
    data_folder = input_data_path / "data"
    data_folder.mkdir(parents=True, exist_ok=True)

    timesteps = 8760

    # Estrai parametri
    total_demand = params["total_demand_TWh"]
    ratio = params["demand_level_ratio"]
    el_price_avg = params["electricity_price_avg"]
    el_avail_small = params["electricity_availability_small"]

    # Calcola domande per cluster
    # BIG clusters: 2 nodi con ratio parte della domanda totale
    # SMALL clusters: 4 nodi con 1 parte ciascuno
    # Total parts = 2*ratio + 4*1
    total_parts = 2 * ratio + 4
    demand_per_part = total_demand / total_parts

    # BIG: uno doppio dell'altro (2:1 ratio)
    big_total = 2 * ratio * demand_per_part
    big1_demand = (2/3) * big_total  # BIG1 ha 2 parti
    big2_demand = (1/3) * big_total  # BIG2 ha 1 parte

    # SMALL: divisi equamente
    small_demand_each = demand_per_part

    hydrogen_data = {
        "BIG1": big1_demand,
        "BIG2": big2_demand,
        "SMALL1": small_demand_each,
        "SMALL2": small_demand_each,
        "SMALL3": small_demand_each,
        "SMALL4": small_demand_each,
    }

    print(f"\n📊 Generazione dati con parametri:")
    print(f"   Total Demand: {total_demand} TWh")
    print(f"   Demand Ratio (Big/Small): {ratio}")
    print(f"   Demand per part: {demand_per_part:.2f} TWh")
    print(f"   BIG1: {big1_demand:.2f} TWh")
    print(f"   BIG2: {big2_demand:.2f} TWh")
    print(f"   SMALL (each): {small_demand_each:.2f} TWh")

    nodes_small = ["SMALL1", "SMALL2", "SMALL3", "SMALL4"]
    nodes_big = ["BIG1", "BIG2"]
    node_storage = ["STORAGE"]

    # STORAGE node: zero demand
    for node in node_storage:
        _create_node_data(data_folder, node, 0, timesteps)
        _create_pressure_data(data_folder, node)
        _create_electricity_prices(data_folder, node, el_price_avg, variation=0)

    # SMALL nodes - variazione prezzi elettricità
    for node in nodes_small:
        h2_demand_TWh = hydrogen_data[node]
        _create_node_data(data_folder, node, h2_demand_TWh, timesteps, fluctuation=0.15)
        _create_pressure_data(data_folder, node)
        _create_electricity_prices(data_folder, node, el_price_avg, variation=0.15)

    # BIG nodes - prezzi fissi
    for node in nodes_big:
        h2_demand_TWh = hydrogen_data[node]
        _create_node_data(data_folder, node, h2_demand_TWh, timesteps, fluctuation=0.10)
        _create_pressure_data(data_folder, node)
        _create_electricity_prices(data_folder, node, el_price_avg, variation=0)  # Fixed price


def _create_node_data(data_folder, node, h2_demand_TWh, timesteps, fluctuation=0.10):
    """Crea file data_network_{node}.xlsx"""

    file_path = data_folder / f"data_network_{node}.xlsx"

    if h2_demand_TWh > 0:
        # Calcola profilo idrogeno
        average_MW = h2_demand_TWh * 1e6 / timesteps  # TWh -> GWh -> MWh -> MW

        # Aggiungi fluttuazioni casuali
        fluctuation_values = np.random.normal(0, fluctuation * average_MW, timesteps)
        hydrogen_profile = average_MW + fluctuation_values
        hydrogen_profile = np.maximum(hydrogen_profile, 0)  # No valori negativi
    else:
        hydrogen_profile = np.zeros(timesteps)

    df = pd.DataFrame({
        'Hydrogen': hydrogen_profile,
        'Electricity': np.zeros(timesteps)  # No electricity demand
    })

    df.to_excel(file_path, index=False)


def _create_pressure_data(data_folder, node):
    """Crea file data_pressure_connections_{node}.xlsx"""

    file_path = data_folder / f"data_pressure_connections_{node}.xlsx"

    df = pd.DataFrame({
        'A': ['demand', 'export', 'import', 'generic_production'],
        'hydrogen': [25, 40, 40, 0]
    })

    df.to_excel(file_path, index=False)


def _create_electricity_prices(data_folder, node, avg_price, variation=0.15):
    """Crea file electricity_prices_{node}.xlsx con variazioni giornaliere/stagionali"""

    file_path = data_folder / f"electricity_prices_{node}.xlsx"
    timesteps = 8760

    # Pattern base: variazioni giornaliere (normalizzato, media=1)
    hourly_pattern = np.array([
        0.7, 0.65, 0.6, 0.6, 0.65, 0.75,  # 0-5: notte (basso)
        0.9, 1.1, 1.2, 1.15, 1.1, 1.05,   # 6-11: mattina/picco
        1.0, 0.95, 0.9, 0.95, 1.0, 1.15,  # 12-17: pomeriggio
        1.3, 1.25, 1.15, 1.0, 0.9, 0.8    # 18-23: sera/picco serale
    ])

    # Replica pattern per anno intero
    daily_pattern = np.tile(hourly_pattern, 365)

    # Aggiungi variazione stagionale (inverno più caro, estate più economico)
    # +10% in inverno, -10% in estate
    seasonal_variation = 1 + 0.1 * np.sin(np.linspace(-np.pi/2, 3*np.pi/2, timesteps))

    # Calcola prezzi base
    prices = avg_price * daily_pattern * seasonal_variation

    # Aggiungi rumore casuale se richiesto (per small nodes)
    if variation > 0:
        noise = np.random.normal(1, variation, timesteps)
        prices *= noise

    prices = np.maximum(prices, 10)  # Minimo 10 EUR/MWh

    df = pd.DataFrame({
        'Electricity_Price_EUR_MWh': prices
    })

    df.to_excel(file_path, index=False)


if __name__ == "__main__":
    # Test della funzione
    test_params = {
        "total_demand_TWh": 20,
        "demand_level_ratio": 5,
        "electricity_price_avg": 100,
        "electricity_availability_small": 50
    }

    test_path = Path(__file__).parent / "test_data_generation"
    generate_excel_data(test_path, test_params)
    print(f"✅ Test completato! Dati generati in: {test_path}")

