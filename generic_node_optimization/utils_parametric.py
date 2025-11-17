"""
Utility Functions for Parametric Studies
Helper functions for managing and analyzing parametric study results
"""

import pandas as pd
from pathlib import Path
import json
import shutil


def list_parametric_studies(results_folder=None):
    """List all available parametric studies"""

    if results_folder is None:
        results_folder = Path(__file__).parent / "results"
    else:
        results_folder = Path(results_folder)

    if not results_folder.exists():
        print(f"❌ Results folder not found: {results_folder}")
        return []

    studies = [f for f in results_folder.iterdir()
               if f.is_dir() and (f.name.startswith("parametric_study_") or f.name.startswith("test_parametric_"))]

    if not studies:
        print("No parametric studies found.")
        return []

    print(f"\n{'='*80}")
    print(f"AVAILABLE PARAMETRIC STUDIES")
    print(f"{'='*80}\n")

    for i, study in enumerate(studies, 1):
        config_file = study / "param_config.json"
        summary_file = study / "results_summary.csv"

        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            total_runs = config.get("tested_combinations", "N/A")
        else:
            total_runs = "N/A"

        if summary_file.exists():
            df = pd.read_csv(summary_file, sep=';')
            completed = len(df)
            successful = sum(df['status'] == 'SUCCESS')
        else:
            completed = 0
            successful = 0

        print(f"{i}. {study.name}")
        print(f"   Runs: {completed}/{total_runs} (Success: {successful})")
        print(f"   Path: {study}")
        print()

    return studies


def get_study_info(study_path):
    """Get detailed information about a parametric study"""

    study_path = Path(study_path)

    if not study_path.exists():
        print(f"❌ Study not found: {study_path}")
        return None

    info = {
        "path": str(study_path),
        "name": study_path.name
    }

    # Load config
    config_file = study_path / "param_config.json"
    if config_file.exists():
        with open(config_file) as f:
            info["config"] = json.load(f)

    # Load results
    summary_file = study_path / "results_summary.csv"
    if summary_file.exists():
        df = pd.read_csv(summary_file, sep=';')
        info["total_runs"] = len(df)
        info["successful_runs"] = sum(df['status'] == 'SUCCESS')
        info["failed_runs"] = sum(df['status'] != 'SUCCESS')
        info["avg_objective"] = df[df['status'] == 'SUCCESS']['objective_value'].mean()
        info["avg_solve_time"] = df[df['status'] == 'SUCCESS']['solve_time'].mean()

    return info


def export_successful_runs(study_path, output_csv=None):
    """Export only successful runs to a separate CSV file"""

    study_path = Path(study_path)
    summary_file = study_path / "results_summary.csv"

    if not summary_file.exists():
        print(f"❌ Results file not found: {summary_file}")
        return None

    df = pd.read_csv(summary_file, sep=';')
    df_success = df[df['status'] == 'SUCCESS'].copy()

    if output_csv is None:
        output_csv = study_path / "successful_runs_only.csv"
    else:
        output_csv = Path(output_csv)

    df_success.to_csv(output_csv, index=False, sep=';')

    print(f"✅ Exported {len(df_success)} successful runs to: {output_csv}")
    return df_success


def compare_studies(study_paths):
    """Compare results from multiple parametric studies"""

    comparison = []

    for study_path in study_paths:
        study_path = Path(study_path)
        info = get_study_info(study_path)

        if info:
            comparison.append({
                "Study": study_path.name,
                "Total Runs": info.get("total_runs", 0),
                "Successful": info.get("successful_runs", 0),
                "Failed": info.get("failed_runs", 0),
                "Avg Objective": info.get("avg_objective", None),
                "Avg Solve Time (s)": info.get("avg_solve_time", None)
            })

    df_comparison = pd.DataFrame(comparison)

    print(f"\n{'='*80}")
    print(f"STUDY COMPARISON")
    print(f"{'='*80}\n")
    print(df_comparison.to_string(index=False))

    return df_comparison


def clean_failed_runs(study_path, confirm=True):
    """Remove folders for failed runs to save disk space"""

    study_path = Path(study_path)
    summary_file = study_path / "results_summary.csv"

    if not summary_file.exists():
        print(f"❌ Results file not found: {summary_file}")
        return

    df = pd.read_csv(summary_file, sep=';')
    failed_runs = df[df['status'] != 'SUCCESS']['run_id'].tolist()

    if not failed_runs:
        print("✅ No failed runs to clean.")
        return

    print(f"\nFound {len(failed_runs)} failed runs:")
    for run_id in failed_runs[:5]:  # Show first 5
        print(f"  - {run_id}")
    if len(failed_runs) > 5:
        print(f"  ... and {len(failed_runs) - 5} more")

    if confirm:
        response = input(f"\n⚠️  Delete {len(failed_runs)} failed run folders? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Cancelled.")
            return

    deleted = 0
    for run_id in failed_runs:
        run_folder = study_path / run_id
        if run_folder.exists():
            shutil.rmtree(run_folder)
            deleted += 1

    print(f"✅ Deleted {deleted} failed run folders.")


def get_parameter_ranges(study_path):
    """Get the actual range of parameters tested in a study"""

    study_path = Path(study_path)
    summary_file = study_path / "results_summary.csv"

    if not summary_file.exists():
        print(f"❌ Results file not found: {summary_file}")
        return None

    df = pd.read_csv(summary_file, sep=';')
    df_success = df[df['status'] == 'SUCCESS']

    param_columns = [
        'demand_level_ratio',
        'total_demand_TWh',
        'import_availability_ratio',
        'import_cost_multiplier',
        'electricity_price_avg',
        'electricity_availability_small'
    ]

    ranges = {}
    for col in param_columns:
        if col in df_success.columns:
            ranges[col] = {
                'min': df_success[col].min(),
                'max': df_success[col].max(),
                'unique_values': sorted(df_success[col].unique().tolist())
            }

    print(f"\n{'='*80}")
    print(f"PARAMETER RANGES (Successful Runs)")
    print(f"{'='*80}\n")

    for param, info in ranges.items():
        print(f"{param}:")
        print(f"  Range: [{info['min']}, {info['max']}]")
        print(f"  Unique values: {info['unique_values']}")
        print()

    return ranges


if __name__ == "__main__":
    # Example usage
    print("Parametric Study Utilities")
    print("="*80)

    studies = list_parametric_studies()

    if studies:
        print("\nSelect a study to get detailed info (or press Enter to skip):")
        choice = input(f"Enter number (1-{len(studies)}): ")

        if choice.strip():
            selected = studies[int(choice) - 1]
            info = get_study_info(selected)

            print(f"\n{'='*80}")
            print(f"STUDY INFO: {selected.name}")
            print(f"{'='*80}")
            for key, value in info.items():
                if key != "config":
                    print(f"{key}: {value}")

            # Get parameter ranges
            get_parameter_ranges(selected)

