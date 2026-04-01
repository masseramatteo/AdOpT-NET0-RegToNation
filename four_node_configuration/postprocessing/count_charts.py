import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================================================================
# SETTINGS
# =============================================================================

# Add one or more folders here to compare them
results_folders = {
    # "New price, small": r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_1200_with_latest_electricity_fluctu_in_small",
    # "New price, large": r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_latest_prices_fluct_in_large",
    # "New price, all": r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_latest_prices_fluctuation_in_all",
    # "Old price, small": r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_old_prices_fluctuation_in_small_clusters",
    # "Old price, large": r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_old_prices_in_large_clusters",
    #"Old price, all": r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_old_prices_fluctuation_in_all_clusters"
}

# Binary variables to plot (display name -> column name that will be created)
# Set to True/False to include/exclude each variable
VARIABLES_TO_PLOT = {
    "Electrolyzer\nSmall cluster 1": "Electrolyzer_small1_installed",
    "Electrolyzer\nSmall cluster 2": "Electrolyzer_small2_installed",
    "Pipeline supply\nSmall cluster 1": "Pipeline_supply_small1",
    "Pipeline supply\nSmall cluster 2": "Pipeline_supply_small2",
}

# Set to True to also generate one chart per archetype in addition to the combined chart
ANALYZE_BY_ARCHETYPE = True

# Colors for yes/no bars
COLOR_YES = "#2196F3"
COLOR_NO  = "#BDBDBD"

# =============================================================================
# HELPER: load and preprocess one folder
# =============================================================================

def load_folder(folder_path: str) -> pd.DataFrame:
    extracted_results_path = os.path.join(folder_path, "extracted_results.xlsx")
    df_merged = pd.read_excel(extracted_results_path)

    # --- Derived binary dependent variables ---

    # Electrolyzer installed (size > 0)
    df_merged["Electrolyzer_small1_installed"] = (df_merged["Small_cluster1_Electrolyzer_small"] > 0).astype(int)
    df_merged["Electrolyzer_small2_installed"] = (df_merged["Small_cluster2_Electrolyzer_small"] > 0).astype(int)

    # Surplus in small clusters (hydrogen network outflow)
    df_merged["Surplus_in_small_cluster1"] = df_merged["Small_cluster1_hydrogen_network_outflow_sum"]
    df_merged["Surplus_in_small_cluster2"] = df_merged["Small_cluster2_hydrogen_network_outflow_sum"]

    # Pipeline supply: electrolyzer installed AND exports to large clusters
    df_merged["Pipeline_supply_small1"] = (
        (df_merged["Electrolyzer_small1_installed"] == 1) &
        (df_merged["Surplus_in_small_cluster1"] > 0)
    ).astype(int)
    df_merged["Pipeline_supply_small2"] = (
        (df_merged["Electrolyzer_small2_installed"] == 1) &
        (df_merged["Surplus_in_small_cluster2"] > 0)
    ).astype(int)

    # Pipeline type (highP / lowP)
    pipeline_connections = [
        ("Large_cluster1", "Small_cluster1"),
        ("Large_cluster1", "Small_cluster2"),
        ("Large_cluster2", "Small_cluster1"),
        ("Large_cluster2", "Small_cluster2"),
    ]
    for node_a, node_b in pipeline_connections:
        highp_col = f"{node_a}_to_{node_b}_hydrogenPipelineOnshore_highP"
        lowp_col  = f"{node_a}_to_{node_b}_hydrogenPipelineOnshore_lowP"
        df_merged[f"{node_a}_to_{node_b}_highP"] = df_merged[highp_col] if highp_col in df_merged.columns else 0
        df_merged[f"{node_a}_to_{node_b}_lowP"]  = df_merged[lowp_col]  if lowp_col  in df_merged.columns else 0

    df_merged["highP_total"] = sum(df_merged[f"{a}_to_{b}_highP"] for a, b in pipeline_connections)
    df_merged["lowP_total"]  = sum(df_merged[f"{a}_to_{b}_lowP"]  for a, b in pipeline_connections)

    return df_merged


# =============================================================================
# LOAD DATA
# =============================================================================

data = {}
for label, folder in results_folders.items():
    print(f"Loading: {label}")
    df = load_folder(folder)
    data[label] = df
    print(f"  -> {len(df)} successful runs")


# =============================================================================
# PLOT FUNCTION
# =============================================================================

def plot_count_chart(data_subset: dict, title: str, save_path: str) -> None:
    """Plot yes/no count bars for each variable, one group of bars per folder."""
    var_labels = list(VARIABLES_TO_PLOT.keys())
    var_cols   = list(VARIABLES_TO_PLOT.values())
    folder_labels = list(data_subset.keys())

    n_vars    = len(var_labels)
    n_folders = len(folder_labels)

    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 5), sharey=False)
    if n_vars == 1:
        axes = [axes]

    colors = [plt.get_cmap("tab10")(i) for i in range(10)]

    for ax, var_label, var_col in zip(axes, var_labels, var_cols):
        bar_width = 0.35 if n_folders > 1 else 0.5
        x = np.arange(2)  # [0 = No, 1 = Yes]

        for i, folder_label in enumerate(folder_labels):
            df = data_subset[folder_label]
            if var_col not in df.columns:
                print(f"  WARNING: column '{var_col}' not found for '{folder_label}', skipping.")
                continue

            n_yes  = int(df[var_col].sum())
            n_no   = int((df[var_col] == 0).sum())
            counts = [n_no, n_yes]

            offset = (i - (n_folders - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset,
                counts,
                width=bar_width,
                color=[COLOR_NO, COLOR_YES],
                label=folder_label if i == 0 else None,
                edgecolor="white",
                linewidth=0.5,
            )

            # Annotate counts on bars
            for bar, count in zip(bars, counts):
                pct = count / len(df) * 100 if len(df) > 0 else 0
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{count}\n({pct:.0f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(var_label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["No", "Yes"])
        ax.set_ylabel("Number of cases")
        ax.set_ylim(0, max((len(df) for df in data_subset.values()), default=1) * 1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend for multi-folder comparison
    if n_folders > 1:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=colors[i % len(colors)], label=lbl)
            for i, lbl in enumerate(folder_labels)
        ]
        fig.legend(handles=handles, loc="upper center", ncol=n_folders,
                   bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.suptitle(title, fontsize=13, y=1.05 if n_folders > 1 else 1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Chart saved: {save_path}")


# =============================================================================
# PLOT: combined (all runs)
# =============================================================================

save_dir = list(results_folders.values())[0]

plot_count_chart(
    data,
    title="Installation frequency across scenarios",
    save_path=os.path.join(save_dir, "count_charts.png"),
)

# =============================================================================
# PLOT: per archetype (optional)
# =============================================================================

if ANALYZE_BY_ARCHETYPE:
    # Collect archetypes present across all folders
    all_archetypes = sorted(set(
        arch
        for df in data.values()
        if "archetype" in df.columns
        for arch in df["archetype"].dropna().unique()
    ))

    if not all_archetypes:
        print("WARNING: ANALYZE_BY_ARCHETYPE=True but no 'archetype' column found in any folder.")
    else:
        print(f"\nArchetype analysis: found archetypes {all_archetypes}")
        for arch in all_archetypes:
            data_arch = {}
            for label, df in data.items():
                if "archetype" in df.columns:
                    df_arch = df[df["archetype"] == arch].copy()
                else:
                    df_arch = pd.DataFrame(columns=df.columns)
                data_arch[label] = df_arch
                print(f"  Archetype {arch} | {label}: {len(df_arch)} runs")

            plot_count_chart(
                data_arch,
                title=f"Installation frequency — Archetype {arch}",
                save_path=os.path.join(save_dir, f"count_charts_archetype_{arch}.png"),
            )

print("Done.")