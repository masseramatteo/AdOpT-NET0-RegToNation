import h5py
import pandas as pd
from pathlib import Path


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
        # Extract network sizes
        if 'design/networks/period1' in f:
            networks_group = f['design/networks/period1']

            for network_name in networks_group.keys():
                network_arcs = {}
                network_group = networks_group[network_name]

                for arc_name in network_group.keys():
                    arc_group = network_group[arc_name]
                    if 'size' in arc_group:
                        size_value = arc_group['size'][()]
                        network_arcs[arc_name] = size_value

                results['networks'][network_name] = network_arcs

        # Extract electrolyzer sizes
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

                # Electrolyzer hydrogen output
                for tech_name in node_group.keys():
                    if 'Electrolyzer' in tech_name:
                        tech_group = node_group[tech_name]
                        if 'hydrogen_output' in tech_group:
                            h2_output = tech_group['hydrogen_output'][()]
                            results['operation'][f"{node_name}_{tech_name}_H2_output_sum"] = h2_output.sum()

                    # Storage hydrogen input
                    if 'Storage_H2' in tech_name:
                        tech_group = node_group[tech_name]
                        if 'hydrogen_input' in tech_group:
                            h2_input = tech_group['hydrogen_input'][()]
                            results['operation'][f"{node_name}_{tech_name}_H2_input_sum"] = h2_input.sum()

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

    for run_name, run_results in all_results.items():
        row = {'run': run_name}

        # Extract network sizes (verify bidirectional arcs have same size)
        for network_name, arcs in run_results['networks'].items():
            arc_sizes = list(arcs.values())

            # Verify all arcs have the same size
            if len(arc_sizes) > 0:
                if len(set(arc_sizes)) > 1:
                    print(f"⚠️  Warning in {run_name}: {network_name} has different arc sizes: {arc_sizes}")

                # Use the first arc size (they should be equal)
                row[network_name] = arc_sizes[0]
            else:
                row[network_name] = 0.0

        # Extract electrolyzer sizes
        for electrolyzer_name, size in run_results['electrolyzers'].items():
            row[electrolyzer_name] = size

        # Extract storage sizes
        for storage_name, size in run_results['storage'].items():
            row[storage_name] = size

        # Extract operational data
        for op_name, value in run_results['operation'].items():
            row[op_name] = value

        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Reorder columns for clarity
    base_columns = ['run']
    network_columns = ['hydrogenPipelineOnshore_highP', 'hydrogenPipelineOnshore_lowP']
    electrolyzer_columns = ['Large_cluster_Electrolyzer_big', 'Small_cluster_Electrolyzer_small']
    storage_columns = ['Large_cluster_Storage_H2_highP', 'Small_cluster_Storage_H2_lowP']

    # Operational columns
    electrolyzer_op_columns = [
        'Large_cluster_Electrolyzer_big_H2_output_sum',
        'Small_cluster_Electrolyzer_small_H2_output_sum'
    ]
    storage_op_columns = [
        'Large_cluster_Storage_H2_highP_H2_input_sum',
        'Small_cluster_Storage_H2_lowP_H2_input_sum'
    ]
    energy_balance_columns = [
        'Large_cluster_hydrogen_import_sum',
        'Large_cluster_hydrogen_network_inflow_sum',
        'Large_cluster_hydrogen_network_outflow_sum',
        'Small_cluster_hydrogen_network_inflow_sum',
        'Small_cluster_hydrogen_network_outflow_sum'
    ]

    # Only include columns that exist
    all_ordered_columns = (base_columns + network_columns + electrolyzer_columns +
                          storage_columns + electrolyzer_op_columns + storage_op_columns +
                          energy_balance_columns)
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
    optimization_folder = r"C:\Users\Masse007\Documents\Code\AdOpT-NET0-RegToNation\two_node_configuration\results\parallel_creation_test_20251219_162930"

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

