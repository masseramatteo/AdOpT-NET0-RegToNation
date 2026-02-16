import h5py
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


def extract_optimization_results(h5_path):
    """
    Extract network sizes, electrolyzer sizes, and storage sizes from h5 file.
    Also extract operational data: hydrogen output from electrolyzers, hydrogen input to storage,
    and energy balance data (import, network_inflow, network_outflow).

    Args:
        h5_path: Path to the optimization_results.h5 file

    Returns:
        Dictionary with extracted data
    """
    results = {
        'networks': {},
        'electrolyzers': {},
        'storage': {},
        'operation': {}
    }

    with h5py.File(h5_path, 'r') as f:
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

        # Extract electrolyzer sizes
        # ...existing code...
        if 'design/nodes/period1' in f:
            nodes_group = f['design/nodes/period1']

            for node_name in nodes_group.keys():
                node_group = nodes_group[node_name]

                # Look for electrolyzers
                for tech_name in node_group.keys():
                    if 'Electrolyzer' in tech_name:
                        tech_group = node_group[tech_name]
                        if 'size' in tech_group:
                            size_value = tech_group['size'][()]
                            results['electrolyzers'][f"{node_name}_{tech_name}"] = size_value[0] if hasattr(size_value, '__len__') else size_value

                    # Look for storage
                    if 'Storage' in tech_name:
                        tech_group = node_group[tech_name]
                        if 'size' in tech_group:
                            size_value = tech_group['size'][()]
                            results['storage'][f"{node_name}_{tech_name}"] = size_value[0] if hasattr(size_value, '__len__') else size_value

        # Extract operational data - technology_operation
        if 'operation/technology_operation/period1' in f:
            tech_op_group = f['operation/technology_operation/period1']

            for node_name in tech_op_group.keys():
                node_group = tech_op_group[node_name]

                # Track total hydrogen production per node
                node_total_h2_production = 0

                # Electrolyzer hydrogen output
                for tech_name in node_group.keys():
                    if 'Electrolyzer' in tech_name:
                        tech_group = node_group[tech_name]
                        if 'hydrogen_output' in tech_group:
                            h2_output = tech_group['hydrogen_output'][()]
                            h2_output_sum = h2_output.sum()
                            results['operation'][f"{node_name}_{tech_name}_H2_output_sum"] = h2_output_sum
                            # Add to node total
                            node_total_h2_production += h2_output_sum

                    # Storage hydrogen input only
                    if 'Storage' in tech_name:
                        tech_group = node_group[tech_name]
                        if 'hydrogen_input' in tech_group:
                            h2_input = tech_group['hydrogen_input'][()]
                            results['operation'][f"{node_name}_{tech_name}_H2_input_sum"] = h2_input.sum()

                # Add total hydrogen production for this node
                if node_total_h2_production > 0:
                    results['operation'][f"{node_name}_TOTAL_H2_production"] = node_total_h2_production

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
                        import_data = h2_group['import'][()]
                        results['operation'][f"{node_name}_hydrogen_import_sum"] = import_data.sum()

                    # Network inflow
                    if 'network_inflow' in h2_group:
                        inflow_data = h2_group['network_inflow'][()]
                        results['operation'][f"{node_name}_hydrogen_network_inflow_sum"] = inflow_data.sum()

                    # Network outflow
                    if 'network_outflow' in h2_group:
                        outflow_data = h2_group['network_outflow'][()]
                        results['operation'][f"{node_name}_hydrogen_network_outflow_sum"] = outflow_data.sum()

    return results


def extract_node_distances(input_data_path):
    """
    Extract distances between all node pairs from NodeLocations.csv.

    Args:
        input_data_path: Path to the input_data folder containing NodeLocations.csv

    Returns:
        Dictionary with distance values for each node pair
    """
    node_locations_file = input_data_path / "NodeLocations.csv"

    if not node_locations_file.exists():
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


def extract_all_runs(optimization_folder):
    """
    Extract results from all parallel runs in the optimization folder.

    Args:
        optimization_folder: Path to the optimization folder

    Returns:
        Dictionary with results for each parallel run
    """
    optimization_path = Path(optimization_folder)

    if not optimization_path.exists():
        raise ValueError(f"Path does not exist: {optimization_path}")

    all_results = {}

    # Find all parallel_run_* folders
    parallel_runs = sorted([d for d in optimization_path.iterdir()
                           if d.is_dir() and d.name.startswith("parallel_run_")])

    print(f"Found {len(parallel_runs)} parallel run folders\n")

    for run_folder in parallel_runs:
        # Navigate to userData/<timestamp_folder>/optimization_results.h5
        user_data = run_folder / "userData"

        if not user_data.exists():
            print(f"⚠️  Skipping {run_folder.name}: userData not found")
            continue

        # Find timestamp folder
        subfolders = [d for d in user_data.iterdir() if d.is_dir()]

        if not subfolders:
            print(f"⚠️  Skipping {run_folder.name}: No timestamp folder found")
            continue

        h5_file = subfolders[0] / "optimization_results.h5"

        if h5_file.exists():
            print(f"✓ Processing {run_folder.name}")
            results = extract_optimization_results(h5_file)

            # Extract node distances from input_data folder
            input_data_path = run_folder / "input_data"
            if input_data_path.exists():
                distances = extract_node_distances(input_data_path)
                results['distances'] = distances
            else:
                results['distances'] = {}

            all_results[run_folder.name] = results
        else:
            print(f"⚠️  Skipping {run_folder.name}: h5 file not found")

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

    for run_idx, (run_name, run_results) in enumerate(all_results.items()):
        row = {'run': run_name}

        # Assign archetype based on run index
        # Every 300 simulations = 1 archetype (10 cases * 30 simulations)
        archetype_number = (run_idx // 300) + 1
        row['archetype'] = f'Archetype_{archetype_number}'

        # Extract network sizes - now individual arcs
        for network_column, size in run_results['networks'].items():
            row[network_column] = size

        # Extract electrolyzer sizes
        for electrolyzer_name, size in run_results['electrolyzers'].items():
            row[electrolyzer_name] = size

        # Extract storage sizes
        for storage_name, size in run_results['storage'].items():
            row[storage_name] = size

        # Extract operational data
        for op_name, value in run_results['operation'].items():
            row[op_name] = value

        # Extract distances
        for distance_name, value in run_results.get('distances', {}).items():
            row[distance_name] = value

        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Reorder columns for clarity
    base_columns = ['run', 'archetype']

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
        'Small_cluster1_hydrogen_network_inflow_sum',
        'Small_cluster1_hydrogen_network_outflow_sum',
        'Small_cluster2_hydrogen_network_inflow_sum',
        'Small_cluster2_hydrogen_network_outflow_sum'
    ]

    # Network arc columns - dynamically get all network columns
    # These are in format: Node1_to_Node2_NetworkType
    network_arc_columns = [col for col in df.columns
                          if col not in base_columns + electrolyzer_columns + storage_columns +
                          electrolyzer_op_columns + node_total_h2_columns + storage_op_columns + energy_balance_columns
                          and '_to_' in col and col.endswith(('highP', 'lowP', 'Onshore', 'Offshore'))]

    # Distance columns - dynamically get all distance columns
    # These are in format: distance_Node1_to_Node2_km
    distance_columns = [col for col in df.columns if col.startswith('distance_') and col.endswith('_km')]

    # Only include columns that exist
    all_ordered_columns = (base_columns + distance_columns + electrolyzer_columns + storage_columns +
                          electrolyzer_op_columns + node_total_h2_columns + storage_op_columns +
                          energy_balance_columns + network_arc_columns)
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
    # Enter the path to your optimization folder
    optimization_folder = r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260130_094107"

    print("="*80)
    print("EXTRACTING OPTIMIZATION RESULTS")
    print("="*80)
    print()

    # Extract all results
    all_results = extract_all_runs(optimization_folder)

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

