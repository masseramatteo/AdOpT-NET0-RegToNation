import h5py
import pandas as pd
import json
from pathlib import Path


def extract_scenario_info(h5_path):
    """
    Extract comprehensive scenario information from h5 file and associated files.

    Args:
        h5_path: Path to the optimization_results.h5 file

    Returns:
        Dictionary with all scenario data
    """
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise ValueError(f"H5 file does not exist: {h5_path}")

    results = {
        'scenario_name': h5_path.parent.name,
        'mip_gap': None,
        'typical_days': None,
        'npv': None,
        'objective_value': None,
        'nodes': {},
        'h2_production': {},  # Total H2 production per node
        'h2_network_outflow': {}  # Total H2 network outflow per node
    }

    # Try to extract MIP gap and typical days from ConfigModel.json
    # ConfigModel.json is in: parallel_run_XXXX/input_data/ConfigModel.json
    # h5 file is in: parallel_run_XXXX/userData/TIMESTAMP/optimization_results.h5
    # So we need to go up 2 folders from h5, then into input_data
    parallel_run_folder = h5_path.parent.parent.parent  # Go up from TIMESTAMP -> userData -> parallel_run_XXXX
    config_path = parallel_run_folder / "input_data" / "ConfigModel.json"

    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                results['mip_gap'] = config.get("solveroptions", {}).get("mipgap", {}).get("value", None)
                results['typical_days'] = config.get("optimization", {}).get("typicaldays", {}).get("N", {}).get("value", None)
        except Exception as e:
            print(f"⚠️  Warning: Could not read ConfigModel.json: {e}")
    else:
        print(f"⚠️  Warning: ConfigModel.json not found at: {config_path}")


    # Extract node information from h5 file
    with h5py.File(h5_path, 'r') as f:
        # Extract NPV and objective value from summary group in h5 file
        if 'summary' in f:
            summary_group = f['summary']
            if 'total_npv' in summary_group:
                results['npv'] = summary_group['total_npv'][()]
            if 'objective' in summary_group:
                results['objective_value'] = summary_group['objective'][()]

        # Get list of nodes
        node_names = []
        if 'design/nodes/period1' in f:
            node_names = list(f['design/nodes/period1'].keys())

        # Initialize node data for expected nodes
        expected_nodes = ['Large_cluster1', 'Large_cluster2', 'Small_cluster1', 'Small_cluster2']
        for node_name in expected_nodes:
            if node_name not in results['nodes']:
                results['nodes'][node_name] = {
                    'technologies': {},
                    'exists': False
                }

        # Extract technologies for each node
        if 'design/nodes/period1' in f:
            nodes_group = f['design/nodes/period1']

            for node_name in nodes_group.keys():
                if node_name not in results['nodes']:
                    results['nodes'][node_name] = {'technologies': {}, 'exists': True}
                else:
                    results['nodes'][node_name]['exists'] = True

                node_group = nodes_group[node_name]

                # Extract all technologies and their sizes
                for tech_name in node_group.keys():
                    tech_group = node_group[tech_name]
                    if 'size' in tech_group:
                        size_value = tech_group['size'][()]
                        # Handle arrays
                        if hasattr(size_value, '__len__') and len(size_value) > 0:
                            size_value = size_value[0]
                        results['nodes'][node_name]['technologies'][tech_name] = size_value

        # Extract network connections
        results['networks'] = {}
        if 'design/networks/period1' in f:
            networks_group = f['design/networks/period1']

            for network_name in networks_group.keys():
                network_group = networks_group[network_name]

                # Get all arcs and their sizes
                arcs = {}
                for arc_name in network_group.keys():
                    arc_group = network_group[arc_name]
                    if 'size' in arc_group:
                        size_value = arc_group['size'][()]
                        arcs[arc_name] = size_value

                # Store network with all its arcs
                results['networks'][network_name] = {
                    'arcs': arcs,
                    'type': network_name  # e.g., hydrogenPipelineOnshore_highP
                }

        # Extract hydrogen production from electrolyzers
        if 'operation/technology_operation/period1' in f:
            tech_op_group = f['operation/technology_operation/period1']

            for node_name in tech_op_group.keys():
                node_group = tech_op_group[node_name]

                # Track total hydrogen production per node
                node_total_h2_production = 0

                # Sum all electrolyzer outputs for this node
                for tech_name in node_group.keys():
                    if 'Electrolyzer' in tech_name:
                        tech_group = node_group[tech_name]
                        if 'hydrogen_output' in tech_group:
                            h2_output = tech_group['hydrogen_output'][()]
                            node_total_h2_production += h2_output.sum()

                # Store total production for this node
                if node_total_h2_production > 0:
                    results['h2_production'][node_name] = node_total_h2_production

        # Extract hydrogen network outflow from energy_balance
        if 'operation/energy_balance/period1' in f:
            energy_balance_group = f['operation/energy_balance/period1']

            for node_name in energy_balance_group.keys():
                node_group = energy_balance_group[node_name]

                # Look for hydrogen carrier
                if 'hydrogen' in node_group:
                    h2_group = node_group['hydrogen']

                    # Network outflow
                    if 'network_outflow' in h2_group:
                        outflow_data = h2_group['network_outflow'][()]
                        results['h2_network_outflow'][node_name] = outflow_data.sum()

    return results


def create_comparison_dataframe(h5_paths):
    """
    Create a DataFrame comparing multiple scenarios.

    Args:
        h5_paths: List of paths to h5 files OR list of tuples (comment, path)

    Returns:
        DataFrame with one row per scenario
    """
    all_data = []

    for item in h5_paths:
        # Check if item is a tuple (comment, path) or just a path
        if isinstance(item, tuple):
            comment, h5_path = item
        else:
            comment = ""
            h5_path = item

        print(f"Processing: {h5_path}")
        try:
            scenario_data = extract_scenario_info(h5_path)

            # Build a flat dictionary for DataFrame
            row = {
                'comment': comment,
                'scenario': scenario_data['scenario_name'],
                'mip_gap': scenario_data['mip_gap'],
                'typical_days': scenario_data['typical_days'],
                'npv': scenario_data['npv'],
                'objective_value': scenario_data['objective_value']
            }

            # Add node information (only technologies, not exists flag)
            for node_name in ['Large_cluster1', 'Large_cluster2', 'Small_cluster1', 'Small_cluster2']:
                node_info = scenario_data['nodes'].get(node_name, {'exists': False, 'technologies': {}})

                if not node_info['exists']:
                    row[f'{node_name}_technologies'] = 'N/A'
                else:

                    # Create a summary of technologies
                    tech_list = []
                    for tech_name, size in node_info['technologies'].items():
                        if size > 0:
                            tech_list.append(f"{tech_name}({size:.2f})")

                    row[f'{node_name}_technologies'] = '; '.join(tech_list) if tech_list else 'None installed'

            # Add hydrogen production data for each node
            for node_name in ['Large_cluster1', 'Large_cluster2', 'Small_cluster1', 'Small_cluster2']:
                h2_prod = scenario_data['h2_production'].get(node_name, 0)
                row[f'{node_name}_H2_production'] = h2_prod

            # Add hydrogen network outflow data for each node
            for node_name in ['Large_cluster1', 'Large_cluster2', 'Small_cluster1', 'Small_cluster2']:
                h2_outflow = scenario_data['h2_network_outflow'].get(node_name, 0)
                row[f'{node_name}_H2_network_outflow'] = h2_outflow

            # Add network information - detailed per arc (only one direction for bidirectional)
            network_summary = []
            processed_pairs = set()  # Track processed node pairs to avoid duplicates

            for network_name, network_data in scenario_data['networks'].items():
                # Get each arc and its size
                for arc_name, arc_size in network_data['arcs'].items():
                    # Parse arc name to get nodes
                    # Format can be: "Large_cluster1Small_cluster2" or "Large_cluster1_to_Small_cluster2"
                    node1, node2 = None, None

                    if '_to_' in arc_name:
                        # Format: "node1_to_node2"
                        parts = arc_name.split('_to_')
                        if len(parts) == 2:
                            node1, node2 = parts[0], parts[1]
                    else:
                        # Format: "Large_cluster1Small_cluster2" or "Small_cluster1Large_cluster2"
                        # Need to split at the boundary between node names
                        # Known node names: Large_cluster1, Large_cluster2, Small_cluster1, Small_cluster2
                        known_nodes = ['Large_cluster1', 'Large_cluster2', 'Small_cluster1', 'Small_cluster2']

                        for node in known_nodes:
                            if arc_name.startswith(node):
                                node1 = node
                                remainder = arc_name[len(node):]
                                if remainder in known_nodes:
                                    node2 = remainder
                                    break

                    if node1 and node2:
                        # Create a sorted tuple to represent the pair (bidirectional)
                        node_pair = tuple(sorted([node1, node2]))
                        pair_key = (node_pair, network_name)

                        # Only process if we haven't seen this pair yet
                        if pair_key not in processed_pairs:
                            processed_pairs.add(pair_key)

                            # Create column name with sorted node names for consistency
                            column_name = f'{node_pair[0]}_to_{node_pair[1]}_{network_name}'
                            row[column_name] = arc_size

                            # Also add to summary string
                            if arc_size > 0:
                                network_summary.append(f"{node_pair[0]}↔{node_pair[1]}:{network_name}({arc_size:.2f})")
                    else:
                        # Fallback: if we couldn't parse, include it anyway
                        column_name = f'{arc_name}_{network_name}'
                        row[column_name] = arc_size
                        if arc_size > 0:
                            network_summary.append(f"{arc_name}:{network_name}({arc_size:.2f})")

            row['networks_summary'] = '; '.join(network_summary) if network_summary else 'No networks'

            all_data.append(row)
            print(f"  ✓ Successfully processed")

        except Exception as e:
            print(f"  ✗ Error processing {h5_path}: {e}")
            continue

    df = pd.DataFrame(all_data)

    # Reorder columns for better readability
    base_columns = ['comment', 'scenario', 'mip_gap', 'typical_days', 'npv', 'objective_value']

    # Only include technology columns, not _exists columns
    node_columns = []
    for node in ['Large_cluster1', 'Large_cluster2', 'Small_cluster1', 'Small_cluster2']:
        node_columns.append(f'{node}_technologies')

    # Add hydrogen production columns
    h2_production_columns = []
    for node in ['Large_cluster1', 'Large_cluster2', 'Small_cluster1', 'Small_cluster2']:
        h2_production_columns.append(f'{node}_H2_production')

    # Add hydrogen network outflow columns
    h2_outflow_columns = []
    for node in ['Large_cluster1', 'Large_cluster2', 'Small_cluster1', 'Small_cluster2']:
        h2_outflow_columns.append(f'{node}_H2_network_outflow')

    network_columns = ['networks_summary']

    # Add all network arc columns (format: arcname_networktype)
    # Get all columns that contain network information (excluding networks_summary)
    network_arc_columns = [col for col in df.columns
                          if col not in base_columns + node_columns + h2_production_columns + h2_outflow_columns + network_columns
                          and '_' in col and col.endswith(('highP', 'lowP', 'Onshore', 'Offshore'))]

    # Combine all columns, keeping only those that exist
    ordered_columns = base_columns + node_columns + h2_production_columns + h2_outflow_columns + network_columns + network_arc_columns
    ordered_columns = [col for col in ordered_columns if col in df.columns]

    return df[ordered_columns]


def export_to_excel(df, output_path):
    """
    Export DataFrame to Excel with formatting.

    Args:
        df: DataFrame to export
        output_path: Path to output Excel file
    """
    # First, save without formatting to avoid column name issues
    df.to_excel(output_path, sheet_name='Scenario Comparison', index=False, engine='openpyxl')

    print(f"\n✓ Exported {len(df)} scenarios to: {output_path}")


if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION: Add your h5 file paths here
    # ========================================================================

    h5_paths = [
        # IMPORTANT: Specify the full path to the optimization_results.h5 file
        # Format option 1 (with comment): ("Your description", r"path\to\optimization_results.h5")
        # Format option 2 (without comment): r"path\to\optimization_results.h5"
        ("no flactuations",
         r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260122_102337 - no flactuations\parallel_run_0002\userData\20260122102553-1\optimization_results.h5"),
        ("0 typical days, no storage",
         r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260122_103924 - fluctations no typical day\parallel_run_0002\userData\20260122104113-1\optimization_results.h5"),
        ("15 typical days, no storage",
         r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260122_134308 - 15 typical days\parallel_run_0002\userData\20260122134444-1\optimization_results.h5"),
        ("30 typical days, no storage storage",
         r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260122_135433 - 30 typical days\parallel_run_0002\userData\20260122135641-1\optimization_results.h5"),
        ("15 typical days, with storage", r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260123_093537 -15 typical days with storage\parallel_run_0002\userData\20260123093719-1\optimization_results.h5"),
        ("30 typical days, with storage",
         r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260122_140100 - 30 typical days with storage\parallel_run_0002\userData\20260122140428-1\optimization_results.h5"),
        ("15 typical days, with storage, network precise",
         r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260123_105916 - 15 strorage precise\parallel_run_0002\userData\20260123110106-1\optimization_results.h5"),
        ("10 typical days, with storage",
         r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260123_130520 - 10 typical days with storage\parallel_run_0002\userData\20260123130708-1\optimization_results.h5"),
        ("15 typical days, with storage higher gap 0.01",
         r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260123_134510 - 15 typical bigger gap\parallel_run_0002\userData\20260123134709-1\optimization_results.h5"),
        ("15 typical days, with storage higher gap 0.001 ",
         r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260123_140741 - 15 typical small bigger gap\parallel_run_0002\userData\20260123141048-1\optimization_results.h5"),
        ("5 typical days, with storage ",
         r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260123_151604 - 5 typical day with storage\parallel_run_0002\userData\20260123151822-1\optimization_results.h5"),

        # Add more paths as needed
    ]

    # Optional: Output file path (defaults to current directory)
    output_file = Path("scenario_comparison.xlsx")

    # ========================================================================
    # EXECUTION
    # ========================================================================

    print("="*80)
    print("SCENARIO COMPARISON TOOL")
    print("="*80)
    print()

    # Filter out any non-existent paths
    valid_paths = []
    for item in h5_paths:
        # Handle both tuple (comment, path) and plain path formats
        if isinstance(item, tuple):
            comment, path = item
            if Path(path).exists():
                valid_paths.append(item)
            else:
                print(f"⚠️  Warning: Path does not exist: {path}")
        else:
            path = item
            if Path(path).exists():
                valid_paths.append(path)
            else:
                print(f"⚠️  Warning: Path does not exist: {path}")

    if not valid_paths:
        print("\n❌ No valid h5 files found. Please check your paths.")
        exit(1)

    print(f"\nFound {len(valid_paths)} valid h5 files")
    print()

    # Extract data
    print("="*80)
    print("EXTRACTING SCENARIO DATA")
    print("="*80)
    print()

    df = create_comparison_dataframe(valid_paths)

    # Display summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"Total scenarios: {len(df)}")
    print()
    print("Preview:")
    print(df.to_string())
    print()

    # Export
    print("="*80)
    print("EXPORTING TO EXCEL")
    print("="*80)
    print()

    export_to_excel(df, output_file)

    print()
    print("="*80)
    print("DONE")
    print("="*80)
