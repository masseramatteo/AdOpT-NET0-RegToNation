import math
import pandas as pd
from pathlib import Path
import os
import json

# ─────────────────────────────────────────────────────────────────────────────
# Default path to the PVGIS export file
# Columns (after skipping 8 metadata rows):
#   time       – YYYYMMDD:HHmm
#   G(i)       – Global irradiance on tilted surface  [W/m²]
#   H_sun      – Sun height (elevation angle)         [degrees]
#   T2m        – Air temperature at 2 m               [°C]
#   WS10m      – Wind speed at 10 m                   [m/s]
#   Int        – Interpolation flag (ignored)
# ─────────────────────────────────────────────────────────────────────────────
_PVGIS_DATA_FILE = (
    Path(__file__).parent / "preprocess" / "data" / "Timeseries_weather_data_solar.csv"
)

# Number of metadata lines at the top and footer lines at the bottom of the file
_PVGIS_SKIPROWS = 8
_PVGIS_SKIPFOOTER = 3


def load_climate_data_from_pvgis(input_data_path, year: int = 2015,
                                  pvgis_file: Path = None):
    """
    Read hourly climate data from a locally stored PVGIS export CSV and write
    the relevant columns into every node's ``ClimateData.csv``.

    The PVGIS file is the standard hourly-radiation export from
    https://re.jrc.ec.europa.eu/pvg_tools/en/ (CSV format).  It must contain
    the columns ``time``, ``G(i)``, ``T2m``, ``WS10m`` after skipping the
    8-line metadata header.

    Column mapping written to ClimateData.csv:
        ``G(i)``   → ``ghi``      (W/m²  – pvlib decomposes into DNI/DHI automatically)
        ``T2m``    → ``temp_air`` (°C)
        ``WS10m``  → ``ws10``     (m/s)

    pvlib's ModelChain will call its built-in Erbs decomposition model when
    ``dhi`` / ``dni`` are NaN, so no changes to ``res.py`` are needed.

    Args:
        input_data_path (str | Path): Path to the adopt_net0 input-data folder.
        year (int): Calendar year to extract from the PVGIS file (default 2015).
            Must be present in the file (2005–2023 in the default dataset).
        pvgis_file (Path | None): Path to the PVGIS CSV. Defaults to
            ``preprocess/data/Timeseries_weather_data_solar.csv``.
    """
    input_data_path = Path(input_data_path)
    if pvgis_file is None:
        pvgis_file = _PVGIS_DATA_FILE
    pvgis_file = Path(pvgis_file)

    if not pvgis_file.exists():
        raise FileNotFoundError(
            f"PVGIS data file not found: {pvgis_file}\n"
            "Download it from https://re.jrc.ec.europa.eu/pvg_tools/en/ "
            "(Hourly Data → CSV) and place it at that path."
        )

    # ── Parse PVGIS file ──────────────────────────────────────────────────────
    print(f"  [PVGIS] Reading {pvgis_file.name} …")
    raw = pd.read_csv(
        pvgis_file,
        skiprows=_PVGIS_SKIPROWS,
        skipfooter=_PVGIS_SKIPFOOTER,
        engine="python",
        dtype={"G(i)": float, "T2m": float, "WS10m": float},
    )

    # Parse timestamp: format is YYYYMMDD:HHmm  (e.g. 20150101:0011)
    raw["_dt"] = pd.to_datetime(raw["time"].astype(str), format="%Y%m%d:%H%M", errors="coerce")
    raw = raw[raw["_dt"].notna()]   # drop the 5 NaT rows (footer bleed-through guard)

    # Filter to requested year
    year_data = raw[raw["_dt"].dt.year == year].copy()
    if len(year_data) == 0:
        available = sorted(raw["_dt"].dt.year.dropna().unique().astype(int))
        raise ValueError(
            f"Year {year} not found in PVGIS file. Available years: {available}"
        )
    print(f"  [PVGIS] Loaded {len(year_data)} hourly rows for year {year}")

    # Build a clean DataFrame with the adopt/pvlib column names
    climate_values = pd.DataFrame({
        "ghi":      year_data["G(i)"].values.astype(float),
        "temp_air": year_data["T2m"].values.astype(float),
        "ws10":     year_data["WS10m"].values.astype(float),
    })
    # dhi / dni are left as NaN → pvlib ModelChain will decompose GHI via Erbs model

    # ── Write into each node's ClimateData.csv ────────────────────────────────
    topology_path = input_data_path / "Topology.json"
    if not topology_path.exists():
        raise FileNotFoundError(f"Topology.json not found at: {topology_path}")
    with open(topology_path, "r") as f:
        topology = json.load(f)

    updated = 0
    skipped = 0

    for period in topology["investment_periods"]:
        for node in topology["nodes"]:
            climate_path = input_data_path / period / "node_data" / node / "ClimateData.csv"
            if not climate_path.exists():
                print(f"  [CLIMATE] WARN: ClimateData.csv not found for {node}/{period}, skipping")
                skipped += 1
                continue

            climate_df = pd.read_csv(climate_path, sep=";", index_col=0)
            n = len(climate_df)

            for col in ["ghi", "temp_air", "ws10"]:
                if col in climate_df.columns:
                    climate_df[col] = climate_values[col].values[:n]

            climate_df.to_csv(climate_path, sep=";")
            updated += 1
            print(f"  [CLIMATE] {node}/{period} written with PVGIS year={year} data ✓")

    print(
        f"\n  [CLIMATE] Done → {updated} node(s) updated, {skipped} skipped"
    )


def calculate_distance_between_coordinates(lon1, lat1, lon2, lat2):
    """
    Calculate the distance between two points on Earth using the Haversine formula.
    Returns distance in kilometers.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in kilometers
    r = 6371

    return c * r


def calculate_distances_from_coordinates(input_data_path):
    """
    Calculate distances between all nodes based on their actual coordinates
    from the generated topology scenario.
    """
    # Read coordinates from the generated scenario file
    scenario_file = input_data_path / "NodeLocations.csv"

    if not scenario_file.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_file}")

    # Load node coordinates
    node_coords = pd.read_csv(scenario_file, sep=';')
    print(f" Calculating distances from scenario coordinates")

    # Create coordinate dictionary
    coords = {}
    for _, row in node_coords.iterrows():
        coords[row['index']] = (row['lon'], row['lat'])

    # Get all nodes
    all_nodes = list(coords.keys())

    # Create empty distance matrix
    distance_matrix = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)

    # Calculate distances between all pairs of nodes
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:  # Don't calculate distance from node to itself
                lon1, lat1 = coords[node1]
                lon2, lat2 = coords[node2]

                # Calculate distance using Haversine formula
                distance = calculate_distance_between_coordinates(lon1, lat1, lon2, lat2)
                distance_matrix.loc[node1, node2] = round(distance, 1)  # Round to 1 decimal place

    return distance_matrix

def load_scenario_nodes(input_data_path, nodes, scenario):
    """Load coordinates for node from scenarios"""
    # Build path relative to this file's location
    current_file = Path(__file__).resolve()
    four_node_folder = current_file.parent  # four_node_configuration folder
    scenario_file = four_node_folder / "preprocess" / "generated_topology" / f"NodeLocations_{scenario}.csv"

    if not scenario_file.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_file}")

    scenario_nodes = pd.read_csv(scenario_file, sep=';')
    # Create dictionaries from the scenario file
    node_lon = {}
    node_lat = {}
    node_alt = {}

    for _, row in scenario_nodes.iterrows():
        node_name = row['Node']
        if node_name in nodes:  # Only use nodes that exist in our system
            node_lon[node_name] = row['lon']
            node_lat[node_name] = row['lat']
            node_alt[node_name] = row['alt']

    # Load or create the main NodeLocations.csv
    node_location = pd.read_csv(input_data_path / "NodeLocations.csv", sep=';', index_col=0, header=0)

    for node in nodes:
        if node in node_lon:  # Only update if we have coordinates for this node
            node_location.at[node, 'lon'] = node_lon[node]
            node_location.at[node, 'lat'] = node_lat[node]
            node_location.at[node, 'alt'] = node_alt[node]

    node_location = node_location.reset_index()
    node_location.to_csv(input_data_path / "NodeLocations.csv", sep=';', index=False)

def define_nodes(input_data_path, params):
    # Add required technologies for SMALL cluster nodes
    small_nodes = ["Small_cluster1", "Small_cluster2"]
    small_new_tech_list = params["small_cluster_new_technologies"]
    small_existing_tech_list = params["small_cluster_existing_technologies"]

    for node in small_nodes:
        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "r") as json_file:
            technologies = json.load(json_file)

        technologies["new"] = small_new_tech_list
        technologies["existing"] = small_existing_tech_list

        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "w") as json_file:
            json.dump(technologies, json_file, indent=4)

    # Add required technologies for BIG cluster nodes
    big_nodes1 = ["Large_cluster1"]
    big_nodes2 = ["Large_cluster2"]
    big_new_tech_list = params["big_cluster_new_technologies"]
    big_existing_tech_list = params["big_cluster_existing_technologies"]

    for node in big_nodes1:
        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "r") as json_file:
            technologies = json.load(json_file)

        technologies["new"] = big_new_tech_list
        technologies["existing"] = big_existing_tech_list

        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "w") as json_file:
            json.dump(technologies, json_file, indent=4)

    for node in big_nodes2:
        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "r") as json_file:
            technologies = json.load(json_file)

        technologies["new"] = big_new_tech_list
        technologies["existing"] = {}

        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "w") as json_file:
            json.dump(technologies, json_file, indent=4)



def add_existing_distribution_network(input_data_path):
    """
    Create existing low pressure distribution network (hydrogenPipelineOnshore_lowP)
    Defines pre-existing hydrogen pipeline infrastructure for distribution
    """
    print("\n Creating Existing Distribution Network (Low Pressure)")

    # Create directory for existing distribution network
    os.makedirs(input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP",
                exist_ok=True)

    # Define node lists
    big_nodes = ["Large_cluster1", "Large_cluster2"]
    small_nodes = ["Small_cluster1", "Small_cluster2"]
    all_nodes = big_nodes + small_nodes

    # Connection matrix
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                             index_col=0)

    # Connect small nodes to each other (local distribution)
    for i, node1 in enumerate(small_nodes):
        for j, node2 in enumerate(small_nodes):
            if i != j:  # Avoid self-connection
                connection.loc[node1, node2] = 1
                connection.loc[node2, node1] = 1

    # Connect small nodes to large nodes (bidirectional)
    for small in small_nodes:
        for large in big_nodes:
            connection.loc[small, large] = 1
            connection.loc[large, small] = 1

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP" / "connection.csv",
        sep=";")
    print(f"  ✓ Connection matrix saved")

    # Distance matrix
    distance = calculate_distances_from_coordinates(input_data_path)
    print("  ✓ Using calculated distances from topology coordinates")

    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP" / "distance.csv",
        sep=";")
    print(f"  ✓ Distance matrix saved")

    # Size matrix (existing pipeline sizes)
    size = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "size.csv", sep=";",
                       index_col=0)

    # Set existing sizes for connections
    for i, node1 in enumerate(small_nodes):
        for j, node2 in enumerate(small_nodes):
            if i != j:  # Avoid self-connection
                size.loc[node1, node2] = 250
                size.loc[node2, node1] = 250


    for small in small_nodes:
        for large in big_nodes:
            size.loc[small, large] = 250
            size.loc[large, small] = 250

    size.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_lowP" / "size.csv",
        sep=";")
    print(f"  ✓ Size matrix saved")
    print(f"✅ Existing distribution network created successfully\n")


def add_existing_transmission_network(input_data_path):
    """
    Create existing high pressure transmission network (hydrogenPipelineOnshore_highP)
    Defines pre-existing hydrogen pipeline infrastructure for transmission
    """
    print("\n Creating Existing Transmission Network (High Pressure)")

    # Create directory for existing transmission network
    os.makedirs(input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP",
                exist_ok=True)

    # Define node lists
    big_nodes = ["Large_cluster1", "Large_cluster2"]
    small_nodes = ["Small_cluster1", "Small_cluster2"]
    all_nodes = big_nodes + small_nodes

    # Connection matrix
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                             index_col=0)

    # Define existing connections for high pressure transmission network
    # Connect big nodes to each other (backbone)
    connection.loc["Large_cluster1", "Large_cluster2"] = 1
    connection.loc["Large_cluster2", "Large_cluster1"] = 1

    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP" / "connection.csv",
        sep=";")
    print(f"  ✓ Connection matrix saved")

    # Distance matrix
    distance = calculate_distances_from_coordinates(input_data_path)
    print("  ✓ Using calculated distances from topology coordinates")

    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP" / "distance.csv",
        sep=";")
    print(f"  ✓ Distance matrix saved")

    # Size matrix (existing pipeline sizes)
    size = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "size.csv", sep=";",
                       index_col=0)

    # Set existing sizes for high pressure connections (typically larger)
    # Big to big backbone
    size.loc["Large_cluster1", "Large_cluster2"] = 1000
    size.loc["Large_cluster2", "Large_cluster1"] = 1000

    size.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "hydrogenPipelineOnshore_highP" / "size.csv",
        sep=";")
    print(f"  ✓ Size matrix saved")
    print(f"✅ Existing transmission network created successfully\n")


def add_new_distribution_network(input_data_path):
    """
    Create low pressure distribution network (hydrogenPipelineOnshore_lowP)
    Connects all nodes in a full mesh topology
    """
    print("\n Creating Distribution Network (Low Pressure)")

    # Create directory for distribution network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP",
                exist_ok=True)

    # Define node lists
    big_nodes = ["Large_cluster1", "Large_cluster2"]
    small_nodes = ["Small_cluster1", "Small_cluster2"]
    all_nodes = big_nodes + small_nodes

    # Load arc size template
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)

    # Same size for nodes to all other nodes
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:
                arc_size.loc[node1, node2] = 1000

    # Save arc size matrix
    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "size_max_arcs.csv",
        sep=";")
    print(f"  ✓ Arc sizes saved")

    # Load connection template
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)

    # Connect all nodes to all other nodes (full mesh)
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:
                connection.loc[node1, node2] = 1

    # Save connection matrix
    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "connection.csv",
        sep=";")
    print(f"  ✓ Connection matrix saved (full mesh)")


    distance = calculate_distances_from_coordinates(input_data_path)
    print("  ✓ Using calculated distances from topology coordinates")

    # Save distance matrix
    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_lowP" / "distance.csv",
        sep=";")
    print(f"  ✓ Distance matrix saved")
    print(f"Distribution network created successfully\n")


def add_new_transmission_network(input_data_path):
    """
    Create high pressure transmission network (hydrogenPipelineOnshore_highP)
    Connects all nodes in a full mesh topology
    """
    print("\n📍 Creating Transmission Network (High Pressure)")

    # Create directory for transmission network
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP",
                exist_ok=True)

    # Define node lists
    big_nodes = ["Large_cluster1", "Large_cluster2"]
    small_nodes = ["Small_cluster1", "Small_cluster2"]
    all_nodes = big_nodes + small_nodes

    # Load arc size template
    arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)

    # Same size for nodes to all other nodes
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:
                arc_size.loc[node1, node2] = 1000

    # Save arc size matrix
    arc_size.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "size_max_arcs.csv",
        sep=";")
    print(f"  ✓ Arc sizes saved")

    # Load connection template
    connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)

    # Connect all nodes to all other nodes (full mesh)
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i != j:
                connection.loc[node1, node2] = 1

    # Save connection matrix
    connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "connection.csv",
        sep=";")
    print(f"  ✓ Connection matrix saved (full mesh)")

    distance = calculate_distances_from_coordinates(input_data_path)
    print("  ✓ Using calculated distances from topology coordinates")

    # Save distance matrix
    distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "hydrogenPipelineOnshore_highP" / "distance.csv",
        sep=";")
    print(f"  ✓ Distance matrix saved")
    print(f"✅ Transmission network created successfully\n")

def tune_gurobi_model(model, output_dir, time_limit=-1, trials_per_setting=3):
    """
    Tunes a Gurobi model with basic settings to find optimal solver parameters.
    Parameter sets that Gurobi sees as an improvement are saved to tune0.prm, tune1.prm, etc.
    Parameter sets are stored in order of decreasing quality, with parameter set 0 being the best.

    Args:
        model: an instance of a Gurobi model (pyomo model with embedded Gurobi)
        output_dir: directory where to save tuning results (.prm files)
        time_limit: total number of seconds to spend tuning. Default of -1 will
                   choose a time limit automatically based on model size.
        trials_per_setting: number of trials to use per parameter set to reduce
                          the effects of randomness. Default is 3.

    Returns:
        Number of tuning results found
    """
    print("\n" + "="*80)
    print("GUROBI MODEL TUNING")
    print("="*80)
    print(f"Time limit: {time_limit}s (-1 = automatic)")
    print(f"Trials per setting: {trials_per_setting}")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set tuning parameters
    model.setParam('TuneTimeLimit', time_limit)
    model.setParam('TuneTrials', trials_per_setting)
    model.update()

    print("Starting tuning process...")
    print("(This may take a while depending on model size and time limit)\n")

    # Run tuning
    model.tune()

    # Get number of tuning results
    result_count = model.tuneResultCount

    print(f"\n✓ Tuning complete!")
    print(f"Found {result_count} improved parameter set(s)\n")

    if result_count == 0:
        print("⚠️  No improved parameter sets found.")
        print("The default parameters may already be optimal for this model.")
        return 0

    # Save each tuning result
    print("Saving tuning results:")
    for i in range(result_count):
        model.getTuneResult(i)
        param_file = output_path / f'tune{i}.prm'
        model.write(str(param_file))
        print(f"  [{i}] Saved to: {param_file}")

    print(f"\n✅ Best parameter set saved as: {output_path / 'tune0.prm'}")
    print("\nTo use the best parameters in future runs:")
    print("  1. Copy tune0.prm to your input data folder")
    print("  2. Load it before optimization with: model.read('tune0.prm')")
    print("="*80 + "\n")

    return result_count


def tune_gurobi_model(model, output_dir, time_limit=-1, trials_per_setting=3):
    """
    Tunes a Gurobi model with basic settings to find optimal solver parameters.
    Parameter sets that Gurobi sees as an improvement are saved to tune0.prm, tune1.prm, etc.
    Parameter sets are stored in order of decreasing quality, with parameter set 0 being the best.

    Args:
        model: an instance of a Gurobi model (pyomo model with embedded Gurobi)
        output_dir: directory where to save tuning results (.prm files)
        time_limit: total number of seconds to spend tuning. Default of -1 will
                   choose a time limit automatically based on model size.
        trials_per_setting: number of trials to use per parameter set to reduce
                          the effects of randomness. Default is 3.

    Returns:
        Number of tuning results found
    """
    print("\n" + "="*80)
    print("GUROBI MODEL TUNING")
    print("="*80)
    print(f"Time limit: {time_limit}s (-1 = automatic)")
    print(f"Trials per setting: {trials_per_setting}")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set tuning parameters
    model.setParam('TuneTimeLimit', time_limit)
    model.setParam('TuneTrials', trials_per_setting)
    model.update()

    print("Starting tuning process...")
    print("(This may take a while depending on model size and time limit)\n")

    # Run tuning
    model.tune()

    # Get number of tuning results
    result_count = model.tuneResultCount

    print(f"\n✓ Tuning complete!")
    print(f"Found {result_count} improved parameter set(s)\n")

    if result_count == 0:
        print("⚠️  No improved parameter sets found.")
        print("The default parameters may already be optimal for this model.")
        return 0

    # Save each tuning result
    print("Saving tuning results:")
    for i in range(result_count):
        model.getTuneResult(i)
        param_file = output_path / f'tune{i}.prm'
        model.write(str(param_file))
        print(f"  [{i}] Saved to: {param_file}")

    print(f"\n✅ Best parameter set saved as: {output_path / 'tune0.prm'}")
    print("\nTo use the best parameters in future runs:")
    print("  1. Copy tune0.prm to your input data folder")
    print("  2. Load it before optimization with: model.read('tune0.prm')")
    print("="*80 + "\n")

    return result_count


