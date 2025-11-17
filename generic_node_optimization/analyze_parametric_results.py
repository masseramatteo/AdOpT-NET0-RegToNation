"""
Analyze Parametric Study Results
Post-processing and visualization of parametric study results
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  Seaborn not installed. Heatmaps will use matplotlib instead.")


def load_results(results_folder):
    """Load results summary from parametric study"""
    results_path = Path(results_folder)
    summary_file = results_path / "results_summary.csv"

    if not summary_file.exists():
        raise FileNotFoundError(f"Results file not found: {summary_file}")

    df = pd.read_csv(summary_file, sep=';')
    print(f"✅ Loaded {len(df)} results from {results_path.name}")
    print(f"   Successful: {sum(df['status'] == 'SUCCESS')}")
    print(f"   Failed: {sum(df['status'] != 'SUCCESS')}")

    return df


def filter_successful(df):
    """Filter only successful runs"""
    df_success = df[df['status'] == 'SUCCESS'].copy()
    print(f"\n📊 Analyzing {len(df_success)} successful runs")
    return df_success


def analyze_by_parameter(df, param_name, output_folder):
    """Analyze results grouped by specific parameter"""

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby(param_name).agg({
        'objective_value': ['mean', 'std', 'min', 'max', 'count'],
        'solve_time': ['mean', 'std']
    }).round(2)

    print(f"\n{'='*60}")
    print(f"Analysis by {param_name}")
    print(f"{'='*60}")
    print(grouped)

    # Save to Excel
    grouped.to_excel(output_folder / f"analysis_by_{param_name}.xlsx")

    return grouped


def create_sensitivity_plot(df, param_name, output_folder):
    """Create sensitivity plot for specific parameter"""

    output_folder = Path(output_folder)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Objective value vs parameter
    df_plot = df.groupby(param_name)['objective_value'].mean()
    axes[0].plot(df_plot.index, df_plot.values, marker='o', linewidth=2)
    axes[0].set_xlabel(param_name, fontsize=12)
    axes[0].set_ylabel('Average Objective Value', fontsize=12)
    axes[0].set_title(f'Objective vs {param_name}', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Solve time vs parameter
    df_plot = df.groupby(param_name)['solve_time'].mean()
    axes[1].plot(df_plot.index, df_plot.values, marker='s', linewidth=2, color='orange')
    axes[1].set_xlabel(param_name, fontsize=12)
    axes[1].set_ylabel('Average Solve Time (s)', fontsize=12)
    axes[1].set_title(f'Solve Time vs {param_name}', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_folder / f"sensitivity_{param_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   📈 Saved: sensitivity_{param_name}.png")


def create_heatmap(df, param_x, param_y, metric='objective_value', output_folder='.'):
    """Create heatmap showing relationship between two parameters"""

    output_folder = Path(output_folder)

    # Pivot table
    pivot_table = df.pivot_table(
        values=metric,
        index=param_y,
        columns=param_x,
        aggfunc='mean'
    )

    plt.figure(figsize=(10, 8))

    if HAS_SEABORN:
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': metric})
    else:
        # Fallback to matplotlib
        im = plt.imshow(pivot_table, cmap='viridis', aspect='auto')
        plt.colorbar(im, label=metric)
        plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
        plt.yticks(range(len(pivot_table.index)), pivot_table.index)

    plt.title(f'{metric} - {param_y} vs {param_x}', fontsize=14)
    plt.xlabel(param_x, fontsize=12)
    plt.ylabel(param_y, fontsize=12)
    plt.tight_layout()

    filename = f"heatmap_{param_y}_vs_{param_x}_{metric}.png"
    plt.savefig(output_folder / filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   🔥 Saved: {filename}")


def identify_best_configurations(df, top_n=10):
    """Identify best configurations based on objective value"""

    df_sorted = df.sort_values('objective_value', ascending=True)
    best_configs = df_sorted.head(top_n)

    print(f"\n{'='*60}")
    print(f"TOP {top_n} CONFIGURATIONS")
    print(f"{'='*60}")

    columns_to_show = [
        'run_id', 'objective_value', 'solve_time',
        'total_demand_TWh', 'demand_level_ratio',
        'import_availability_ratio', 'import_cost_multiplier',
        'electricity_price_avg', 'electricity_availability_small'
    ]

    print(best_configs[columns_to_show].to_string(index=False))

    return best_configs


def main():
    """Main analysis function"""

    # Select results folder
    base_path = Path(__file__).parent
    results_base = base_path / "results"

    # List available studies
    studies = [f for f in results_base.iterdir() if f.is_dir() and f.name.startswith("parametric_study_")]

    if not studies:
        print("❌ No parametric studies found in results folder")
        return

    print("Available parametric studies:")
    for i, study in enumerate(studies, 1):
        print(f"  {i}. {study.name}")

    # Select study
    if len(studies) == 1:
        selected_study = studies[0]
        print(f"\n✅ Using study: {selected_study.name}")
    else:
        choice = input(f"\nSelect study (1-{len(studies)}): ")
        selected_study = studies[int(choice) - 1]

    # Load results
    df = load_results(selected_study)
    df_success = filter_successful(df)

    if len(df_success) == 0:
        print("❌ No successful runs to analyze")
        return

    # Create analysis folder
    analysis_folder = selected_study / "analysis"
    analysis_folder.mkdir(exist_ok=True)

    print(f"\n📁 Analysis output folder: {analysis_folder}")

    # Parameters to analyze
    parameters = [
        'demand_level_ratio',
        'total_demand_TWh',
        'import_availability_ratio',
        'import_cost_multiplier',
        'electricity_price_avg',
        'electricity_availability_small'
    ]

    # Analyze each parameter
    print(f"\n{'='*60}")
    print("SENSITIVITY ANALYSIS")
    print(f"{'='*60}")

    for param in parameters:
        if param in df_success.columns:
            analyze_by_parameter(df_success, param, analysis_folder)
            create_sensitivity_plot(df_success, param, analysis_folder)

    # Create heatmaps for key parameter combinations
    print(f"\n{'='*60}")
    print("HEATMAP ANALYSIS")
    print(f"{'='*60}")

    heatmap_pairs = [
        ('total_demand_TWh', 'demand_level_ratio'),
        ('import_availability_ratio', 'import_cost_multiplier'),
        ('electricity_price_avg', 'electricity_availability_small'),
        ('total_demand_TWh', 'import_availability_ratio')
    ]

    for param_x, param_y in heatmap_pairs:
        if param_x in df_success.columns and param_y in df_success.columns:
            create_heatmap(df_success, param_x, param_y, 'objective_value', analysis_folder)

    # Identify best configurations
    best_configs = identify_best_configurations(df_success, top_n=10)
    best_configs.to_excel(analysis_folder / "best_configurations.xlsx", index=False)

    print(f"\n✅ Analysis completed!")
    print(f"📊 Results saved in: {analysis_folder}")


if __name__ == "__main__":
    main()

