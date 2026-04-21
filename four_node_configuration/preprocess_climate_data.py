"""
Pre-download JRC climate data for 4 fixed onshore representative locations.

Run this ONCE locally (where internet access is available).
The same 4 climate curves are reused for all scenarios (no per-scenario download).
Results saved to preprocess/climate_data/{node}.csv

In solve_single_model, these files are applied to the ClimateData.csv template
instead of calling the JRC API.
"""

import pandas as pd
from pathlib import Path

from adopt_net0.data_preprocessing.data_loading import import_jrc_climate_data

# ── Fixed onshore representative coordinates ─────────────────────────────────
# Large clusters: use scenario 0001 coordinates (confirmed working)
# Small clusters: onshore points in representative regions
NODES = {
    "Large_cluster1": {"lon": 4.4777,  "lat": 51.9244, "alt": 0},  # Netherlands
    "Large_cluster2": {"lon": 22.7799, "lat": 51.9244, "alt": 0},  # Poland
    "Small_cluster1": {"lon": 9.5000,  "lat": 56.0000, "alt": 0},  # Jutland, Denmark
    "Small_cluster2": {"lon": 7.0000,  "lat": 47.5000, "alt": 0},  # Alsace, France
}

BASE_PATH = Path(__file__).resolve().parent
CACHE_DIR = BASE_PATH / "preprocess" / "climate_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Download ─────────────────────────────────────────────────────────────────
for node, coords in NODES.items():
    cache_file = CACHE_DIR / f"{node}.csv"

    if cache_file.exists():
        print(f"[{node}] Already cached — skipping.")
        continue

    lon, lat, alt = coords["lon"], coords["lat"], coords["alt"]
    print(f"[{node}] Downloading... lon={lon} lat={lat}")
    data = import_jrc_climate_data(lon, lat, 2022, alt)
    data["dataframe"].to_csv(cache_file, sep=";", index=False)
    print(f"[{node}] Saved to {cache_file.relative_to(BASE_PATH)}")

print("\nDone. Commit preprocess/climate_data/ and push to Snellius.")