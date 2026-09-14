import h5py
import json
import os
import pandas as pd
from pathlib import Path
import sys
import importlib.util

# Import utilities from parent directory
utilities_path = Path(__file__).parent.parent / "utilities.py"
spec = importlib.util.spec_from_file_location("utilities", utilities_path)
utilities = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utilities)
calculate_distance_between_coordinates = utilities.calculate_distance_between_coordinates

# ── Campaign to extract ──────────────────────────────────────────────────
# v5 paired design (Sept 2026): 8 topologies per economy, economy-major run ids.
OPTIMIZATION_FOLDER = (
    r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M"
    r"\AdOpT-NET0-RegToNation\four_node_configuration\results"
    r"\Snellius_simulations_10_09\6000_snellius_runs"
)

# ── Archetype configuration ──────────────────────────────────────────────
# Since the v5 campaign every run carries 'archetype' ('A'..'D'), 'economy' and
# 'slot' in its parameters, so nothing has to be derived. The fallback below is
# only used for pre-v5 campaigns, where scenarios were numbered sequentially
# (0001, 0002, ...) and every SCENARIOS_PER_ARCHETYPE consecutive scenarios
# belonged to the same archetype: 0001-0010 = Archetype 1, 0011-0020 = Archetype 2.
SCENARIOS_PER_ARCHETYPE = 10

# Topology design table of the v5 campaign. Joined on the run's 'scenario' so the
# geometry actually realized by the generator (d_LL, d_SS, d_SL_mean, centrality,
# compactness) is available to the decision trees next to the sampled economics.
TOPOLOGY_FOLDER = Path(__file__).parent.parent / "preprocess" / "generated_topology_v5"
# Columns taken from topology_parameters.csv (besides the identity columns)
TOPOLOGY_METRIC_COLUMNS = [
    "d_LL", "d_SS", "d_SL_mean", "d_LL_target", "d_SL_target", "d_SS_target",
    "S1_near", "S1_far", "S1_centrality", "S2_near", "S2_far", "S2_centrality",
    "rho_comp",
]

# Parameters that must stay strings: 'scenario' is a zero-padded topology id
# ("0001"), and int() would silently turn it into 1, breaking the join with
# topology_parameters.csv.
STRING_PARAMS = {
    "scenario", "slot", "archetype", "solar_anchor", "electricity_profile_approach",
    "status",
}

# Node-level technologies of the four-node study. Matching is done on these exact
# names so that the auto-generated compressors (hydrogen_Compressor_<a>_<b>, which
# contain 'Electrolyzer'/'Storage' in their name) do not end up in the technology
# buckets. Compressors are collected separately.
ELECTROLYZER_TECHS = {"Electrolyzer_big", "Electrolyzer_small"}
STORAGE_TECHS = {"Storage_H2_highP", "Storage_H2_lowP",
                 "Storage_H2_Cavern", "Storage_H2_Cavern_existing"}
PV_TECHS = {"Photovoltaic"}
COMPRESSOR_PREFIX = "hydrogen_Compressor"


def extract_optimization_results(h5_path):
    """
    Extract network sizes, electrolyzer sizes, storage sizes and PV sizes from the
    h5 file, plus the operation: hydrogen output of the electrolyzers, hydrogen
    input to storage, the hydrogen energy balance (import, network_inflow,
    network_outflow), and the electricity side (PV generation and curtailment,
    electricity consumed by the electrolyzers, grid import and its cost).

    All operational columns are ANNUAL totals: the model is solved on clustered
    typical days, so every series is weighted with k_means_specs/period1/factors
    before being summed (see _load_time_factors). Extractions made before this
    was fixed carry sums over the typical days instead, roughly a factor
    8760/(24*N_typical_days) too small, and must be regenerated.

    Args:
        h5_path: Path to the optimization_results.h5 file

    Returns:
        Dictionary with extracted data
    """
    results = {
        'networks': {},
        'electrolyzers': {},
        'storage': {},
        'operation': {},
        'photovoltaics': {},
        'electricity': {},
        'compressors': {},
    }

    with h5py.File(str(h5_path), 'r') as f:
        # Hourly weights of the clustered time steps (see _load_time_factors).
        # Every operational sum below is weighted with these, so the columns are
        # annual totals and not sums over typical days.
        factors = _load_time_factors(f)

        # Extract network sizes - individual arcs
        if 'design/networks/period1' in f:
            networks_group = f['design/networks/period1']

            for network_name in networks_group.keys():
                network_group = networks_group[network_name]

                # Track processed pairs to avoid bidirectional duplicates
                processed_pairs = set()

                for arc_name in network_group.keys():
                    arc_group = network_group[arc_name]
                    if 'size' in arc_group:
                        size_value = arc_group['size'][()]

                        # Parse arc name to identify nodes
                        node1, node2 = None, None

                        if '_to_' in arc_name:
                            parts = arc_name.split('_to_')
                            if len(parts) == 2:
                                node1, node2 = parts[0], parts[1]
                        else:
                            # Format: "Large_cluster1Small_cluster2"
                            known_nodes = ['Large_cluster1', 'Large_cluster2', 'Small_cluster1', 'Small_cluster2']
                            for node in known_nodes:
                                if arc_name.startswith(node):
                                    node1 = node
                                    remainder = arc_name[len(node):]
                                    if remainder in known_nodes:
                                        node2 = remainder
                                        break

                        if node1 and node2:
                            # Create sorted pair to avoid duplicates
                            node_pair = tuple(sorted([node1, node2]))
                            pair_key = (node_pair, network_name)

                            if pair_key not in processed_pairs:
                                processed_pairs.add(pair_key)
                                # Create standardized column name
                                column_key = f'{node_pair[0]}_to_{node_pair[1]}_{network_name}'
                                results['networks'][column_key] = size_value
                        else:
                            # Fallback: use arc name as-is
                            column_key = f'{arc_name}_{network_name}'
                            results['networks'][column_key] = size_value

        # Extract technology sizes (electrolyzers, storage, PV, compressors)
        # Matching is on the exact technology name: the compressors created by
        # adopt are called hydrogen_Compressor_<from>_<to> and would otherwise be
        # caught by a substring test on 'Electrolyzer' / 'Storage'.
        if 'design/nodes/period1' in f:
            nodes_group = f['design/nodes/period1']

            for node_name in nodes_group.keys():
                node_group = nodes_group[node_name]

                for tech_name in node_group.keys():
                    tech_group = node_group[tech_name]
                    if 'size' not in tech_group:
                        continue
                    size_value = _scalar(tech_group['size'][()])

                    if tech_name in ELECTROLYZER_TECHS:
                        results['electrolyzers'][f"{node_name}_{tech_name}"] = size_value
                    elif tech_name in STORAGE_TECHS:
                        results['storage'][f"{node_name}_{tech_name}"] = size_value
                    elif tech_name in PV_TECHS:
                        results['photovoltaics'][f"{node_name}_{tech_name}"] = size_value
                    elif tech_name.startswith(COMPRESSOR_PREFIX):
                        results['compressors'][f"{node_name}_{tech_name}"] = size_value
                    else:
                        # Unknown technology: keep it rather than lose it silently
                        results['electrolyzers'][f"{node_name}_{tech_name}"] = size_value

        # Extract operational data - technology_operation
        if 'operation/technology_operation/period1' in f:
            tech_op_group = f['operation/technology_operation/period1']

            for node_name in tech_op_group.keys():
                node_group = tech_op_group[node_name]

                # Track total hydrogen production and electrolyzer electricity use
                node_total_h2_production = 0
                node_total_el_to_electrolyzers = 0.0
                has_electrolyzer = False

                for tech_name in node_group.keys():
                    tech_group = node_group[tech_name]

                    # Electrolyzers: hydrogen produced and electricity consumed
                    if tech_name in ELECTROLYZER_TECHS:
                        has_electrolyzer = True
                        if 'hydrogen_output' in tech_group:
                            h2_output_sum = _annual_sum(tech_group['hydrogen_output'][()], factors)
                            results['operation'][f"{node_name}_{tech_name}_H2_output_sum"] = h2_output_sum
                            node_total_h2_production += h2_output_sum
                        if 'electricity_input' in tech_group:
                            el_in_sum = _annual_sum(tech_group['electricity_input'][()], factors)
                            results['electricity'][f"{node_name}_{tech_name}_electricity_input_sum"] = el_in_sum
                            node_total_el_to_electrolyzers += el_in_sum

                    # Storage: hydrogen input only
                    elif tech_name in STORAGE_TECHS:
                        if 'hydrogen_input' in tech_group:
                            results['operation'][f"{node_name}_{tech_name}_H2_input_sum"] = \
                                _annual_sum(tech_group['hydrogen_input'][()], factors)

                    # PV: generation, curtailment and realized capacity factor
                    elif tech_name in PV_TECHS:
                        if 'electricity_output' in tech_group:
                            results['electricity'][f"{node_name}_{tech_name}_electricity_output_sum"] = \
                                _annual_sum(tech_group['electricity_output'][()], factors)
                        if 'curtailment_electricity' in tech_group:
                            results['electricity'][f"{node_name}_{tech_name}_curtailment_sum"] = \
                                _annual_sum(tech_group['curtailment_electricity'][()], factors)
                        if 'cap_factor' in tech_group:
                            results['electricity'][f"{node_name}_{tech_name}_cap_factor_mean"] = \
                                _annual_mean(tech_group['cap_factor'][()], factors)

                # Add totals for this node
                if node_total_h2_production > 0:
                    results['operation'][f"{node_name}_TOTAL_H2_production"] = node_total_h2_production
                if has_electrolyzer:
                    results['electricity'][f"{node_name}_TOTAL_electrolyzer_electricity_input"] = \
                        node_total_el_to_electrolyzers

        # Extract operational data - energy_balance
        if 'operation/energy_balance/period1' in f:
            energy_balance_group = f['operation/energy_balance/period1']

            for node_name in energy_balance_group.keys():
                node_group = energy_balance_group[node_name]

                # Look for hydrogen carrier
                if 'hydrogen' in node_group:
                    h2_group = node_group['hydrogen']

                    # Import (only for Large_cluster)
                    if 'import' in h2_group:
                        results['operation'][f"{node_name}_hydrogen_import_sum"] = \
                            _annual_sum(h2_group['import'][()], factors)

                    # Network inflow
                    if 'network_inflow' in h2_group:
                        results['operation'][f"{node_name}_hydrogen_network_inflow_sum"] = \
                            _annual_sum(h2_group['network_inflow'][()], factors)

                    # Network outflow
                    if 'network_outflow' in h2_group:
                        results['operation'][f"{node_name}_hydrogen_network_outflow_sum"] = \
                            _annual_sum(h2_group['network_outflow'][()], factors)

                # Electricity carrier: how much grid electricity the node actually
                # used, what it cost, and how much came from the local PV. Needed
                # to tell a grid-limited node from a price-limited one.
                if 'electricity' in node_group:
                    el_group = node_group['electricity']

                    for key, label in (('import', 'import'),
                                       ('export', 'export'),
                                       ('technology_inputs', 'technology_inputs'),
                                       ('technology_outputs', 'technology_outputs'),
                                       ('generic_production', 'generic_production'),
                                       ('network_inflow', 'network_inflow'),
                                       ('network_outflow', 'network_outflow'),
                                       ('network_consumption', 'network_consumption'),
                                       ('compressor_input', 'compressor_input')):
                        if key in el_group:
                            results['electricity'][f"{node_name}_electricity_{label}_sum"] = \
                                _annual_sum(el_group[key][()], factors)

                    # Cost of the imported electricity (hourly price x hourly import)
                    if 'import' in el_group and 'import_price' in el_group:
                        imp = el_group['import'][()]
                        price = el_group['import_price'][()]
                        if imp.shape == price.shape:
                            results['electricity'][f"{node_name}_electricity_import_cost"] = \
                                _annual_sum(imp * price, factors)

    return results


def _scalar(value):
    """Return a plain scalar from an h5 value that may be a 1-element array."""
    return value[0] if hasattr(value, '__len__') else value


def _load_time_factors(f):
    """
    Weight of every time step of the solved model, in hours of the year.

    The model is solved on N_typical_days clustered days (20 x 24 = 480 steps in
    the v5 campaign), and each of those steps stands for as many hours as the
    number of days its cluster represents. k_means_specs/period1/factors holds
    exactly that, and sums to 8760. Summing a series without these weights
    over/under-counts every typical day by up to a factor 2.

    Returns None when the model was solved at full resolution (no clustering),
    in which case a plain sum is already the annual total.
    """
    if 'k_means_specs/period1/factors' not in f:
        return None
    factors = f['k_means_specs/period1/factors'][()]
    return factors.astype(float).ravel()


def _annual_sum(series, factors):
    """Annual total of an hourly series solved on clustered typical days."""
    values = series.astype(float).ravel()
    if factors is None or values.shape != factors.shape:
        return float(values.sum())
    return float((values * factors).sum())


def _annual_mean(series, factors):
    """Hour-weighted mean of a series solved on clustered typical days."""
    values = series.astype(float).ravel()
    if factors is None or values.shape != factors.shape:
        return float(values.mean())
    return float((values * factors).sum() / factors.sum())


def extract_node_distances(input_data_path):
    """
    Extract distances between all node pairs from NodeLocations.csv.

    Args:
        input_data_path: Path to the input_data folder containing NodeLocations.csv

    Returns:
        Dictionary with distance values for each node pair
    """
    node_locations_file = input_data_path / "NodeLocations.csv"

    if not os.path.exists(str(node_locations_file)):
        return {}

    # Read node locations
    node_locations = pd.read_csv(node_locations_file, sep=';')

    # Create coordinate dictionary
    coords = {}
    for _, row in node_locations.iterrows():
        node_name = row['index']
        coords[node_name] = (row['lon'], row['lat'])

    # Calculate distances between all node pairs
    distances = {}
    nodes = list(coords.keys())

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i < j:  # Only calculate once for each pair (avoid duplicates)
                lon1, lat1 = coords[node1]
                lon2, lat2 = coords[node2]

                # Calculate distance using Haversine formula from utilities
                distance_km = calculate_distance_between_coordinates(lon1, lat1, lon2, lat2)

                # Create sorted pair key for consistency
                node_pair = tuple(sorted([node1, node2]))
                distance_key = f"distance_{node_pair[0]}_to_{node_pair[1]}_km"
                distances[distance_key] = round(distance_km, 1)

    return distances


def _parse_json_value(value, key=None):
    """
    Convert a JSON value to its proper Python type.

    The parallel runner saves params with json.dumps(params, default=str),
    which converts numpy.float64 / numpy.int64 to strings like "12.345".
    This function converts those strings back to numbers so that the
    reconstructed data matches the original parallel_results_summary.xlsx.

    Keys listed in STRING_PARAMS are left untouched: 'scenario' is the
    zero-padded topology id ("0001") used to join topology_parameters.csv, and
    int() would turn it into 1.
    """
    if key in STRING_PARAMS:
        return value
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        # Try int first, then float
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def extract_run_params(result_subfolder):
    """
    Extract input parameters and solver results from JSON files saved in each
    run's result folder. This reconstructs the same data that the parallel runner
    saves into parallel_results_summary.xlsx.

    Reads from:
      - optimization_model_info.txt: input params (total_demand_TWh, electricity_price_avg, etc.)
      - optimization_results_summary.json: solver results (var_npv, objective_value, etc.)

    Args:
        result_subfolder: Path to the timestamp subfolder inside userData/
                          (where both files are stored)

    Returns:
        Dictionary with parameter + result values (empty dict if files not found)
    """
    result_subfolder = Path(result_subfolder)
    all_data = {}

    # 1) Read input parameters from optimization_model_info.txt
    info_file = result_subfolder / "optimization_model_info.txt"
    if os.path.isfile(str(info_file)):
        try:
            with open(str(info_file), 'r', encoding='utf-8') as f:
                params = json.loads(f.read())
            # Filter to only keep scalar values (skip lists, dicts, etc.)
            # Also convert numeric strings back to numbers (numpy types were
            # serialized as strings by json.dumps(default=str))
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool)):
                    all_data[key] = _parse_json_value(value, key)
                elif key == "solar_cf_realized_by_node" and isinstance(value, dict):
                    # Per-node realized PV capacity factor: flatten to scalars
                    # (the mean over the two small nodes is already a scalar param)
                    for node_name, cf in value.items():
                        if isinstance(cf, (int, float, str)):
                            all_data[f"solar_cf_realized_{node_name}"] = _parse_json_value(cf)
        except Exception as e:
            print(f"    ⚠️  Error reading params from {info_file}: {e}")
    else:
        print(f"    ⚠️  optimization_model_info.txt not found in {result_subfolder}")

    # 2) Read solver results from optimization_results_summary.json
    #    (var_npv, para_total_demand, npv_over_demand, demand_times_wtp,
    #     willingness_to_pay, objective_value)
    results_file = result_subfolder / "optimization_results_summary.json"
    if os.path.isfile(str(results_file)):
        try:
            with open(str(results_file), 'r', encoding='utf-8') as f:
                result_info = json.loads(f.read())
            for key, value in result_info.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    all_data[key] = _parse_json_value(value, key) if value is not None else None
        except Exception as e:
            print(f"    ⚠️  Error reading results from {results_file}: {e}")

    return all_data


def load_topology_parameters(topology_folder=TOPOLOGY_FOLDER):
    """
    Load the design table of the topology generator, indexed by topology id
    (the run's 'scenario'). Returns an empty dict if the file is not there, so
    campaigns generated with an older topology folder still extract.

    Returns:
        {topology_id: {column: value}} with the columns of TOPOLOGY_METRIC_COLUMNS
        plus 'archetype', 'economy', 'slot' when present.
    """
    topo_file = Path(topology_folder) / "topology_parameters.csv"
    if not os.path.isfile(str(topo_file)):
        print(f"  ⚠️  topology_parameters.csv not found in {topology_folder} "
              f"- topology metrics will be missing")
        return {}

    df = pd.read_csv(topo_file, sep=';', dtype={"topology": str, "slot": str,
                                                "archetype": str})
    keep = [c for c in TOPOLOGY_METRIC_COLUMNS if c in df.columns]
    identity = [c for c in ("archetype", "economy", "slot") if c in df.columns]

    table = {}
    for _, row in df.iterrows():
        entry = {c: row[c] for c in keep}
        # identity columns are prefixed so they never overwrite the run params
        for c in identity:
            entry[f"topology_{c}"] = row[c]
        table[str(row["topology"])] = entry

    print(f"  Loaded topology design table: {len(table)} topologies, "
          f"{len(keep)} metric columns")
    return table


def check_and_create_parallel_summary(optimization_folder, all_results):
    """
    Check if parallel_results_summary.xlsx exists and contains all runs.
    If missing or incomplete, (re)create it from the optimization_model_info.txt
    files found inside each run folder.

    This ensures the summary file is always available even when simulations were
    run across multiple server nodes that each produced separate partial files.

    Args:
        optimization_folder: Path to the optimization results folder
        all_results: Dictionary from extract_all_runs() (already contains 'params')

    Returns:
        DataFrame with the parallel results summary
    """
    optimization_path = Path(optimization_folder)
    summary_path = optimization_path / "parallel_results_summary.xlsx"

    n_runs = len(all_results)

    # Check if file exists and is complete
    existing_df = None
    if os.path.isfile(str(summary_path)):
        try:
            existing_df = pd.read_excel(summary_path)
            n_existing = len(existing_df)
            print(f"  Found existing parallel_results_summary.xlsx with {n_existing} runs")

            if n_existing >= n_runs:
                print(f"  ✓ File is complete ({n_existing} >= {n_runs} extracted runs)")
                return existing_df
            else:
                print(f"  ⚠️  File is incomplete ({n_existing} < {n_runs} extracted runs)")
        except Exception as e:
            print(f"  ⚠️  Error reading existing file: {e}")
    else:
        print(f"  parallel_results_summary.xlsx not found")

    # (Re)create from optimization_model_info.txt in each run
    print(f"  Recreating parallel_results_summary.xlsx from run folders...")

    summary_rows = []
    runs_without_params = 0

    for run_name, run_results in all_results.items():
        params = run_results.get('params', {})
        if not params:
            runs_without_params += 1

        row = {
            "run_id": run_name,
            "status": "SUCCESS",
            **params
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    if runs_without_params > 0:
        print(f"  ⚠️  {runs_without_params}/{n_runs} runs had no optimization_model_info.txt")

    # Save
    df_summary.to_csv(optimization_path / "parallel_results_summary.csv", index=False, sep=';')
    df_summary.to_excel(summary_path, index=False)
    print(f"  ✓ Created parallel_results_summary.xlsx with {len(df_summary)} runs")

    return df_summary


def extract_all_runs(optimization_folder):
    """
    Extract results from all parallel runs in the optimization folder.

    Args:
        optimization_folder: Path to the optimization folder

    Returns:
        Dictionary with results for each parallel run
    """
    optimization_path = Path(optimization_folder)

    if not os.path.exists(str(optimization_path)):
        raise ValueError(f"Path does not exist: {optimization_path}")

    all_results = {}

    # Design table of the topology generator, joined on the run's 'scenario'
    topology_table = load_topology_parameters()

    # Find all parallel_run_* folders (use os.listdir for UNC compatibility)
    parallel_runs = sorted([
        optimization_path / name
        for name in os.listdir(str(optimization_path))
        if os.path.isdir(str(optimization_path / name)) and name.startswith("parallel_run_")
    ])

    print(f"Found {len(parallel_runs)} parallel run folders\n")
    n_topology_missing = 0

    for run_folder in parallel_runs:
        # Navigate to userData/<timestamp_folder>/optimization_results.h5
        user_data = run_folder / "userData"

        if not os.path.isdir(str(user_data)):
            print(f"⚠️  Skipping {run_folder.name}: userData not found")
            continue

        # Find timestamp folder (use os.listdir for UNC compatibility).
        # Folder names are timestamps (YYYYMMDDHHMMSS-n) so they sort
        # chronologically; a re-solved run has more than one and the LAST one is
        # the result that counts. Folders holding an h5 win over empty ones.
        subfolders = sorted(
            user_data / name
            for name in os.listdir(str(user_data))
            if os.path.isdir(str(user_data / name))
        )

        if not subfolders:
            print(f"⚠️  Skipping {run_folder.name}: No timestamp folder found")
            continue

        with_h5 = [s for s in subfolders
                   if os.path.isfile(str(s / "optimization_results.h5"))]
        result_subfolder = with_h5[-1] if with_h5 else subfolders[-1]
        h5_file = result_subfolder / "optimization_results.h5"

        if os.path.isfile(str(h5_file)):
            print(f"✓ Processing {run_folder.name}")
            results = extract_optimization_results(h5_file)

            # Extract node distances from input_data folder
            input_data_path = run_folder / "input_data"
            if os.path.isdir(str(input_data_path)):
                distances = extract_node_distances(input_data_path)
                results['distances'] = distances
            else:
                results['distances'] = {}

            # Extract input parameters from optimization_model_info.txt
            results['params'] = extract_run_params(result_subfolder)

            # Join the topology design metrics on the scenario (topology id)
            scenario = results['params'].get('scenario')
            topo_entry = topology_table.get(str(scenario)) if topology_table else None
            if topo_entry:
                results['topology'] = dict(topo_entry)
            else:
                results['topology'] = {}
                if topology_table:
                    n_topology_missing += 1

            all_results[run_folder.name] = results
        else:
            print(f"⚠️  Skipping {run_folder.name}: h5 file not found at {h5_file}")

    if n_topology_missing:
        print(f"\n⚠️  {n_topology_missing}/{len(all_results)} runs had no matching "
              f"topology in topology_parameters.csv (pre-v5 campaign?)")

    return all_results


def create_summary_dataframe(all_results):
    """
    Create a single pandas DataFrame with one row per run and columns for each component.

    Args:
        all_results: Dictionary from extract_all_runs()

    Returns:
        DataFrame with columns for design sizes and operational sums
    """
    summary_data = []

    for run_name, run_results in all_results.items():
        row = {'run': run_name}

        # Archetype: campaigns from the v5 paired design (Sept 2026) store it in
        # the run parameters ('A'..'D'), together with 'economy' and 'slot'.
        # Older campaigns derive it from the scenario number:
        # 0001-0010 → Archetype 1, 0011-0020 → Archetype 2, etc.
        # (SCENARIOS_PER_ARCHETYPE scenarios per archetype)
        params = run_results.get('params', {})
        topology = run_results.get('topology', {})
        scenario = params.get('scenario')
        if params.get('archetype') is not None:
            row['archetype'] = str(params['archetype'])
        elif topology.get('topology_archetype') is not None:
            # Campaign without archetype in the params, but the topology id
            # resolves in topology_parameters.csv
            row['archetype'] = str(topology['topology_archetype'])
        elif scenario is not None:
            try:
                scenario_num = int(scenario)
                archetype_number = ((scenario_num - 1) // SCENARIOS_PER_ARCHETYPE) + 1
            except (ValueError, TypeError):
                archetype_number = scenario
            row['archetype'] = f'Archetype_{archetype_number}'
        else:
            row['archetype'] = 'Unknown'

        # Extract network sizes - now individual arcs
        for network_column, size in run_results['networks'].items():
            row[network_column] = size

        # Extract electrolyzer sizes
        for electrolyzer_name, size in run_results['electrolyzers'].items():
            row[electrolyzer_name] = size

        # Extract storage sizes
        for storage_name, size in run_results['storage'].items():
            row[storage_name] = size

        # Extract photovoltaic sizes
        for pv_name, size in run_results['photovoltaics'].items():
            row[pv_name] = size

        # Extract compressor sizes (design)
        for comp_name, size in run_results.get('compressors', {}).items():
            row[comp_name] = size

        # Extract operational data
        for op_name, value in run_results['operation'].items():
            row[op_name] = value

        # Extract electricity-side operation (PV output, curtailment,
        # electrolyzer consumption, grid import and its cost)
        for el_name, value in run_results.get('electricity', {}).items():
            row[el_name] = value

        # Extract topology design metrics (joined on scenario)
        for topo_name, value in run_results.get('topology', {}).items():
            row[topo_name] = value

        # Extract distances
        for distance_name, value in run_results.get('distances', {}).items():
            row[distance_name] = value

        # Extract input parameters (from optimization_model_info.txt)
        for param_name, value in run_results.get('params', {}).items():
            row[param_name] = value

        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Reorder columns for clarity
    # 'economy' and 'slot' identify the v5 paired design: 8 runs share an economy
    # (the unit of independence for grouped CV) and the slot names the pairing
    # (A_near / A_far / ...). They sit next to the archetype, not in the params.
    base_columns = [c for c in ['run', 'archetype', 'economy', 'slot', 'scenario']
                    if c in df.columns or c in ('run', 'archetype')]

    # Electrolyzer columns for all 4 nodes
    # ...existing code...
    electrolyzer_columns = [
        'Large_cluster1_Electrolyzer_big',
        'Large_cluster2_Electrolyzer_big',
        'Small_cluster1_Electrolyzer_small',
        'Small_cluster2_Electrolyzer_small'
    ]

    # Storage columns for all 4 nodes
    storage_columns = [
        'Large_cluster1_Storage_H2_highP',
        'Large_cluster1_Storage_H2_Cavern',
        'Large_cluster1_Storage_H2_Cavern_existing',
        'Large_cluster2_Storage_H2_highP',
        'Large_cluster2_Storage_H2_Cavern',
        'Small_cluster1_Storage_H2_lowP',
        'Small_cluster2_Storage_H2_lowP'
    ]

    # Operational columns - Electrolyzer H2 output for all 4 nodes
    electrolyzer_op_columns = [
        'Large_cluster1_Electrolyzer_big_H2_output_sum',
        'Large_cluster2_Electrolyzer_big_H2_output_sum',
        'Small_cluster1_Electrolyzer_small_H2_output_sum',
        'Small_cluster2_Electrolyzer_small_H2_output_sum'
    ]

    # Total H2 production per node
    node_total_h2_columns = [
        'Large_cluster1_TOTAL_H2_production',
        'Large_cluster2_TOTAL_H2_production',
        'Small_cluster1_TOTAL_H2_production',
        'Small_cluster2_TOTAL_H2_production'
    ]

    # Storage H2 input only for all storage types
    storage_op_columns = [
        # Large_cluster1 - H2 input
        'Large_cluster1_Storage_H2_highP_H2_input_sum',
        'Large_cluster1_Storage_H2_Cavern_H2_input_sum',
        'Large_cluster1_Storage_H2_Cavern_existing_H2_input_sum',
        # Large_cluster2 - H2 input
        'Large_cluster2_Storage_H2_highP_H2_input_sum',
        'Large_cluster2_Storage_H2_Cavern_H2_input_sum',
        # Small clusters - H2 input
        'Small_cluster1_Storage_H2_lowP_H2_input_sum',
        'Small_cluster2_Storage_H2_lowP_H2_input_sum'
    ]

    # Energy balance for all 4 nodes
    energy_balance_columns = [
        'Large_cluster1_hydrogen_import_sum',
        'Large_cluster1_hydrogen_network_inflow_sum',
        'Large_cluster1_hydrogen_network_outflow_sum',
        'Large_cluster2_hydrogen_import_sum',
        'Large_cluster2_hydrogen_network_inflow_sum',
        'Large_cluster2_hydrogen_network_outflow_sum',
        # H2 import exists only at the large nodes, but the balance writes a
        # zero series at the small ones - keep it so the block is symmetric
        'Small_cluster1_hydrogen_import_sum',
        'Small_cluster1_hydrogen_network_inflow_sum',
        'Small_cluster1_hydrogen_network_outflow_sum',
        'Small_cluster2_hydrogen_import_sum',
        'Small_cluster2_hydrogen_network_inflow_sum',
        'Small_cluster2_hydrogen_network_outflow_sum'
    ]

    # PV design columns (small clusters only in this study)
    pv_columns = [
        'Small_cluster1_Photovoltaic',
        'Small_cluster2_Photovoltaic',
        'Large_cluster1_Photovoltaic',
        'Large_cluster2_Photovoltaic',
    ]

    # Electricity-side operation: PV generation and curtailment, electrolyzer
    # consumption, grid import / export and the cost of the imported electricity.
    # Without these a grid-limited node cannot be told from a price-limited one.
    electricity_op_columns = []
    for _node in ['Large_cluster1', 'Large_cluster2', 'Small_cluster1', 'Small_cluster2']:
        electricity_op_columns += [
            f'{_node}_Photovoltaic_electricity_output_sum',
            f'{_node}_Photovoltaic_curtailment_sum',
            f'{_node}_Photovoltaic_cap_factor_mean',
            f'{_node}_Electrolyzer_big_electricity_input_sum',
            f'{_node}_Electrolyzer_small_electricity_input_sum',
            f'{_node}_TOTAL_electrolyzer_electricity_input',
            f'{_node}_electricity_import_sum',
            f'{_node}_electricity_import_cost',
            f'{_node}_electricity_export_sum',
            f'{_node}_electricity_technology_inputs_sum',
            f'{_node}_electricity_technology_outputs_sum',
            f'{_node}_electricity_generic_production_sum',
            f'{_node}_electricity_compressor_input_sum',
            f'{_node}_electricity_network_inflow_sum',
            f'{_node}_electricity_network_outflow_sum',
            f'{_node}_electricity_network_consumption_sum',
        ]

    # Topology design metrics joined from topology_parameters.csv
    topology_columns = ([c for c in TOPOLOGY_METRIC_COLUMNS if c in df.columns]
                        + [c for c in df.columns if c.startswith('topology_')])

    # Network arc columns - dynamically get all network columns
    # These are in format: Node1_to_Node2_NetworkType
    network_arc_columns = [col for col in df.columns
                          if col not in base_columns + electrolyzer_columns + storage_columns +
                          electrolyzer_op_columns + node_total_h2_columns + storage_op_columns + energy_balance_columns
                          and '_to_' in col and col.endswith(('highP', 'lowP', 'Onshore', 'Offshore'))]

    # Compressor design sizes: kept for completeness but pushed to the end, they
    # are model plumbing rather than a design decision of the study.
    compressor_columns = [col for col in df.columns if COMPRESSOR_PREFIX in col]

    # Distance columns - dynamically get all distance columns
    # These are in format: distance_Node1_to_Node2_km
    distance_columns = [col for col in df.columns if col.startswith('distance_') and col.endswith('_km')]

    # Input parameter columns (from optimization_model_info.txt)
    # These are the independent variables used for decision tree analysis.
    # Order = the sampled design of the v5 campaign, then the derived quantities.
    known_param_columns = [
        # sampled economics / demand
        'total_demand_TWh', 'demand_level_ratio', 'unbalance_ratio',
        'import_availability_ratio', 'electricity_price_avg',
        'electricity_standard_dev', 'hydrogen_import_price',
        # sampled grid connections
        'electricity_availability_small', 'electricity_availability_large',
        # sampled solar resource (target) and what the climate series realized
        'solar_cf_mean', 'solar_cf_realized_mean',
        'solar_cf_realized_Small_cluster1', 'solar_cf_realized_Small_cluster2',
        'solar_anchor', 'solar_anchor_scale', 'solar_cf_predicted',
        # sampled cost axes, new in the v5 campaign (Sept 2026)
        'capex_ratio_small_big', 'pipeline_cost_multiplier',
        # derived limits written by the runner
        'h2_import_limit1', 'h2_import_limit2',
        'el_import_limit_small', 'el_import_limit_large',
        # objective / solver outputs
        'var_npv', 'para_total_demand', 'npv_over_demand',
        'demand_times_wtp', 'willingness_to_pay', 'objective_value',
    ]
    param_columns = [col for col in known_param_columns if col in df.columns]
    # Also pick up any extra param columns not in the known list
    all_known_columns = (base_columns + param_columns + distance_columns +
                        topology_columns + electrolyzer_columns + storage_columns +
                        pv_columns + electricity_op_columns +
                        electrolyzer_op_columns + node_total_h2_columns +
                        storage_op_columns + energy_balance_columns + compressor_columns)
    extra_param_columns = [col for col in df.columns
                           if col not in all_known_columns
                           and '_to_' not in col
                           and not col.startswith('distance_')]

    # Only include columns that exist
    all_ordered_columns = (base_columns + param_columns + extra_param_columns +
                          distance_columns + topology_columns +
                          electrolyzer_columns + storage_columns + pv_columns +
                          electrolyzer_op_columns + node_total_h2_columns + storage_op_columns +
                          energy_balance_columns + electricity_op_columns +
                          network_arc_columns + compressor_columns)
    ordered_columns = [col for col in all_ordered_columns if col in df.columns]

    return df[ordered_columns]


def export_to_excel(df, output_path):
    """
    Export dataframe to Excel file.

    Args:
        df: DataFrame to export
        output_path: Path to output Excel file
    """
    df.to_excel(output_path, sheet_name='Results', index=False, engine='openpyxl')
    print(f"  ✓ Exported {len(df)} runs with {len(df.columns)-1} components")
    print(f"\n✓ Results exported to: {output_path}")


if __name__ == "__main__":
    # Campaign to extract (set OPTIMIZATION_FOLDER at the top of this file)
    optimization_folder = OPTIMIZATION_FOLDER

    print("="*80)
    print("EXTRACTING OPTIMIZATION RESULTS")
    print("="*80)
    print()

    # Extract all results
    all_results = extract_all_runs(optimization_folder)

    # Check and (re)create parallel_results_summary.xlsx if missing or incomplete
    print()
    print("="*80)
    print("CHECKING PARALLEL RESULTS SUMMARY")
    print("="*80)
    print()
    check_and_create_parallel_summary(optimization_folder, all_results)

    print()
    print("="*80)
    print("CREATING SUMMARY")
    print("="*80)
    print()

    # Create summary DataFrame
    df = create_summary_dataframe(all_results)

    # Print summary
    print(f"SUMMARY:")
    print(f"  Total runs: {len(df)}")
    print(f"  Components tracked:")
    for col in df.columns:
        if col != 'run':
            print(f"    - {col}")
    print()

    # Show first few rows as preview
    print("Preview of results:")
    print(df.head().to_string())
    print()

    # Export to Excel
    output_file = Path(optimization_folder) / "extracted_results.xlsx"
    print("="*80)
    print("EXPORTING TO EXCEL")
    print("="*80)
    print()
    export_to_excel(df, output_file)

