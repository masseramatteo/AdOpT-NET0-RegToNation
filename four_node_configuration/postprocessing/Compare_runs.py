import h5py
import pandas as pd
import json
from pathlib import Path


def find_h5_files_in_results_folder(results_folder_path):
    """
    Automatically find all optimization_results.h5 files in a results folder.
    Searches through all parallel_run_XXXX subdirectories.

    Args:
        results_folder_path: Path to the results folder (e.g., parallel_creation_test_20260122_102337)

    Returns:
        List of paths to optimization_results.h5 files, sorted by run number
    """
    results_folder = Path(results_folder_path)

    if not results_folder.exists():
        raise ValueError(f"Results folder does not exist: {results_folder_path}")

    print(f"\n🔍 Scanning folder: {results_folder}")

    # Find all parallel_run_XXXX folders
    parallel_run_folders = sorted(results_folder.glob("parallel_run_*"))

    if not parallel_run_folders:
        print(f"⚠️  No parallel_run_XXXX folders found in {results_folder}")
        return []

    print(f"   Found {len(parallel_run_folders)} parallel_run folders")

    h5_files = []

    for run_folder in parallel_run_folders:
        # Look for userData folder
        userdata_folder = run_folder / "userData"

        if not userdata_folder.exists():
            print(f"   ⚠️  Skipping {run_folder.name}: no userData folder")
            continue

        # Find all timestamp folders (format: YYYYMMDDHHMMSS-N)
        timestamp_folders = list(userdata_folder.glob("*"))
        timestamp_folders = [f for f in timestamp_folders if f.is_dir()]

        if not timestamp_folders:
            print(f"   ⚠️  Skipping {run_folder.name}: no timestamp folders in userData")
            continue

        # Take the most recent timestamp folder (last one alphabetically)
        timestamp_folder = sorted(timestamp_folders)[-1]

        # Look for optimization_results.h5
        h5_file = timestamp_folder / "optimization_results.h5"

        if h5_file.exists():
            h5_files.append(h5_file)
            print(f"   ✓ {run_folder.name}: {h5_file.name}")
        else:
            print(f"   ⚠️  Skipping {run_folder.name}: optimization_results.h5 not found")

    print(f"\n   Total h5 files found: {len(h5_files)}")

    return h5_files


def find_h5_files_in_multiple_folders(folder_paths):
    """
    Find all h5 files in multiple results folders.

    Args:
        folder_paths: List of paths to results folders or list of tuples (comment, path)

    Returns:
        List of tuples (comment, h5_path) for all found h5 files
    """
    all_h5_files = []

    for item in folder_paths:
        # Check if item is a tuple (comment, path) or just a path
        if isinstance(item, tuple):
            base_comment, folder_path = item
        else:
            base_comment = ""
            folder_path = item

        folder_path = Path(folder_path)

        # Find all h5 files in this folder
        h5_files = find_h5_files_in_results_folder(folder_path)

        # Add to results with comment
        for h5_file in h5_files:
            # Extract run number from parallel_run_XXXX
            run_name = h5_file.parent.parent.parent.name  # e.g., parallel_run_0001

            if base_comment:
                comment = f"{base_comment} - {run_name}"
            else:
                comment = run_name

            all_h5_files.append((comment, str(h5_file)))

    return all_h5_files


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
        'scenario': None,  # Will be populated from run_params.json
        'mip_gap': None,
        'typical_days': None,
        'time_total': None,  # Total computation time
        'npv': None,
        'objective_value': None,
        'nodes': {},
        'h2_production': {},  # Total H2 production per node
        'h2_network_outflow': {}  # Total H2 network outflow per node
    }

    # Extract scenario and parameters from run_params.json
    # run_params.json is in: parallel_run_XXXX/run_params.json
    # h5 file is in: parallel_run_XXXX/userData/TIMESTAMP/optimization_results.h5
    parallel_run_folder = h5_path.parent.parent.parent  # Go up from TIMESTAMP -> userData -> parallel_run_XXXX
    run_params_path = parallel_run_folder / "run_params.json"

    if run_params_path.exists():
        try:
            with open(run_params_path, 'r') as f:
                run_params = json.load(f)
                # Extract scenario (e.g., "0035")
                results['scenario'] = run_params.get('scenario', None)
                # Also store all run parameters for reference
                results['run_params'] = run_params
                # Extract key parameters to top level for easier access
                results['mip_gap'] = run_params.get('mipgap', None)
                results['typical_days'] = run_params.get('N_typical_days', None)
        except Exception as e:
            print(f"⚠️  Warning: Could not read run_params.json: {e}")
    else:
        print(f"⚠️  Warning: run_params.json not found at: {run_params_path}")

    # Try to extract additional info from ConfigModel.json if needed
    # ConfigModel.json is in: parallel_run_XXXX/input_data/ConfigModel.json
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

    # -----------------------------------------------------------------
    # Profiling: look for memory_peak_create.json and memory_peak_solve.json
    # -----------------------------------------------------------------
    # Default profiling values
    results['create_avg_rss_mb'] = None
    results['create_peak_rss_mb'] = None
    results['solve_avg_rss_mb'] = None
    results['solve_peak_rss_mb'] = None
    # Default cpu profiling values
    results['create_cpu_avg_percent'] = None
    results['create_cpu_peak_percent'] = None
    results['create_cpu_time_s'] = None
    results['solve_cpu_avg_percent'] = None
    results['solve_cpu_peak_percent'] = None
    results['solve_cpu_time_s'] = None

    profiling_folder = parallel_run_folder / 'profiling'
    if profiling_folder.exists() and profiling_folder.is_dir():
        # Support multiple possible file names produced by profiling step
        candidates = {
            'create': ['memory_peak_create.json', 'proc_create.json', 'proc_create.json', 'memory_peak_create.json'],
            'solve': ['memory_peak_solve.json', 'proc_solve.json', 'proc_solve.json', 'memory_peak_solve.json']
        }

        # Helper to try reading a list of possible file names
        def _read_profile_file(folder, names):
            for nm in names:
                fp = folder / nm
                if fp.exists():
                    try:
                        with open(fp, 'r') as pf:
                            return json.load(pf)
                    except Exception as e:
                        print(f"⚠️  Warning: Could not read profiling file {fp}: {e}")
            return None

        # Read create profiling (if any)
        p_create = _read_profile_file(profiling_folder, candidates['create'])
        if p_create:
            results['create_avg_rss_mb'] = p_create.get('avg_rss_mb', results['create_avg_rss_mb'])
            results['create_peak_rss_mb'] = p_create.get('peak_rss_mb', results['create_peak_rss_mb'])
            # CPU metrics (some profiling files include these)
            results['create_cpu_avg_percent'] = p_create.get('cpu_avg_percent', results['create_cpu_avg_percent'])
            results['create_cpu_peak_percent'] = p_create.get('cpu_peak_percent', results['create_cpu_peak_percent'])
            results['create_cpu_time_s'] = p_create.get('cpu_time_s', results['create_cpu_time_s'])

        # Read solve profiling (if any)
        p_solve = _read_profile_file(profiling_folder, candidates['solve'])
        if p_solve:
            results['solve_avg_rss_mb'] = p_solve.get('avg_rss_mb', results['solve_avg_rss_mb'])
            results['solve_peak_rss_mb'] = p_solve.get('peak_rss_mb', results['solve_peak_rss_mb'])
            results['solve_cpu_avg_percent'] = p_solve.get('cpu_avg_percent', results['solve_cpu_avg_percent'])
            results['solve_cpu_peak_percent'] = p_solve.get('cpu_peak_percent', results['solve_cpu_peak_percent'])
            results['solve_cpu_time_s'] = p_solve.get('cpu_time_s', results['solve_cpu_time_s'])

    # Extract node information from h5 file
    with h5py.File(h5_path, 'r') as f:
        # Extract NPV, objective value, and time_total from summary group in h5 file
        if 'summary' in f:
            summary_group = f['summary']
            if 'total_npv' in summary_group:
                results['npv'] = summary_group['total_npv'][()]
            if 'objective' in summary_group:
                results['objective_value'] = summary_group['objective'][()]
            if 'time_total' in summary_group:
                results['time_total'] = summary_group['time_total'][()]

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
                'scenario': scenario_data.get('scenario', 'Unknown'),  # From run_params.json
                'scenario_name': scenario_data['scenario_name'],  # Timestamp folder name
                'mip_gap': scenario_data['mip_gap'],
                'typical_days': scenario_data['typical_days'],
                'time_total': scenario_data['time_total'],  # Total computation time
                'npv': scenario_data['npv'],
                'objective_value': scenario_data['objective_value']
            }

            # Add profiling metrics (if present)
            row['create_avg_rss_mb'] = scenario_data.get('create_avg_rss_mb', None)
            row['create_peak_rss_mb'] = scenario_data.get('create_peak_rss_mb', None)
            row['solve_avg_rss_mb'] = scenario_data.get('solve_avg_rss_mb', None)
            row['solve_peak_rss_mb'] = scenario_data.get('solve_peak_rss_mb', None)

            # Add CPU profiling metrics (if present)
            row['create_cpu_avg_percent'] = scenario_data.get('create_cpu_avg_percent', None)
            row['create_cpu_peak_percent'] = scenario_data.get('create_cpu_peak_percent', None)
            row['create_cpu_time_s'] = scenario_data.get('create_cpu_time_s', None)
            row['solve_cpu_avg_percent'] = scenario_data.get('solve_cpu_avg_percent', None)
            row['solve_cpu_peak_percent'] = scenario_data.get('solve_cpu_peak_percent', None)
            row['solve_cpu_time_s'] = scenario_data.get('solve_cpu_time_s', None)

            # Add all run parameters if available
            if 'run_params' in scenario_data:
                run_params = scenario_data['run_params']
                # Add key parameters
                for param_name in ['total_demand_TWh', 'demand_level_ratio', 'unbalance_ratio',
                                   'import_availability_ratio', 'electricity_price_avg',
                                   'electricity_availability_small', 'hydrogen_import_price',
                                   'willingness_to_pay', 'time_limit', 'threads']:
                    if param_name in run_params:
                        row[param_name] = run_params[param_name]

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
    base_columns = ['comment', 'scenario', 'mip_gap', 'typical_days', 'time_total', 'npv', 'objective_value']
    # Insert profiling columns after time_total
    profiling_columns = ['create_avg_rss_mb', 'create_peak_rss_mb', 'solve_avg_rss_mb', 'solve_peak_rss_mb']

    # Add CPU profiling columns
    cpu_profiling_columns = ['create_cpu_avg_percent', 'create_cpu_peak_percent', 'create_cpu_time_s',
                             'solve_cpu_avg_percent', 'solve_cpu_peak_percent', 'solve_cpu_time_s']

    # Add run parameter columns (from run_params.json)
    param_columns = ['total_demand_TWh', 'demand_level_ratio', 'unbalance_ratio',
                     'import_availability_ratio', 'electricity_price_avg',
                     'electricity_availability_small', 'hydrogen_import_price',
                     'willingness_to_pay', 'time_limit', 'threads']
    # Only include columns that exist in the dataframe
    param_columns = [col for col in param_columns if col in df.columns]

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
                          if col not in base_columns + param_columns + node_columns + h2_production_columns + h2_outflow_columns + network_columns
                          and '_' in col and col.endswith(('highP', 'lowP', 'Onshore', 'Offshore'))]

    # Combine all columns, keeping only those that exist
    ordered_columns = base_columns + profiling_columns + cpu_profiling_columns + param_columns + node_columns + h2_production_columns + h2_outflow_columns + network_columns + network_arc_columns
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
    # CONFIGURATION: Choose one of the two methods below
    # ========================================================================

    # METHOD 1: Manually specify individual h5 file paths
    # -----------------------------------------------------------------------
    USE_MANUAL_PATHS = False  # Set to True to use manual paths

    manual_h5_paths = [
        # Format option 1 (with comment): ("Your description", r"path\to\optimization_results.h5")
        # Format option 2 (without comment): r"path\to\optimization_results.h5"
        ("no flactuations",
         r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260122_102337 - no flactuations\parallel_run_0002\userData\20260122102553-1\optimization_results.h5"),
        ("0 typical days, no storage",
         r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260122_103924 - fluctations no typical day\parallel_run_0002\userData\20260122104113-1\optimization_results.h5"),
        # Add more paths as needed
    ]

    # METHOD 2: Automatically scan entire results folders
    # -----------------------------------------------------------------------
    # This will find ALL parallel_run_XXXX folders and their h5 files automatically
    USE_FOLDER_SCANNING = True  # Set to True to use automatic folder scanning

    results_folders = [
        # You can specify just the folder path, or add a comment like this:
        # Format option 1 (with comment): ("Description", r"path\to\results_folder")
        # Format option 2 (without comment): r"path\to\results_folder"

        ("Test run 1", r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_20260311_164411"),
        # Add more results folders as needed
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

    # Determine which method to use
    if USE_FOLDER_SCANNING:
        print("🔧 MODE: Automatic folder scanning")
        print("="*80)

        # Find all h5 files in the specified folders
        h5_paths = find_h5_files_in_multiple_folders(results_folders)

        if not h5_paths:
            print("\n❌ No h5 files found in the specified folders.")
            exit(1)

    elif USE_MANUAL_PATHS:
        print("🔧 MODE: Manual h5 file paths")
        print("="*80)
        h5_paths = manual_h5_paths
    else:
        print("❌ Error: Please set either USE_FOLDER_SCANNING or USE_MANUAL_PATHS to True")
        exit(1)

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
