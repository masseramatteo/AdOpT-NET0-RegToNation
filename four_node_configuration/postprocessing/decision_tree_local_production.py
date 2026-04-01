import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from typing import List
import os

# SETTINGS
do_preprocessing = 1
results_folder = r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_old_prices_fluctuation_in_small_clusters"

# results_folders = {
#     # "New price, small": r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_1200_with_latest_electricity_fluctu_in_small",
#     # "New price, large": r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_latest_prices_fluct_in_large",
#     # "New price, all": r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_latest_prices_fluctuation_in_all",
#     # "Old price, small": r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_old_prices_fluctuation_in_small_clusters",
#     # "Old price, large": r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_old_prices_in_large_clusters",
#     #"Old price, all": r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\parallel_creation_test_old_prices_fluctuation_in_all_clusters"
# }
# ============================================================================
# FEATURE SELECTION SWITCHES - Set to True/False to include/exclude features
# ============================================================================
INCLUDE_ARCHETYPE = False  # Set to False to exclude archetype from analysis
INCLUDE_DISTANCES = True   # Set to False to exclude all distances from analysis
ANALYZE_BY_ARCHETYPE = False  # Set to True to create separate decision trees for each archetype (1, 2, 3, 4)

# Dependent variables for 4-node configuration
dependent_vars_selection = ["Electrolyzer_small1_installed", "Electrolyzer_small2_installed"]
dependent_vars_size = ["Electrolyzer_small1_size", "Electrolyzer_small2_size"]
dependent_vars_operation = ["Surplus_in_small_cluster1", "Surplus_in_small_cluster2"]

# Independent variables for 4-node configuration
# Archetype replaces distance_between_nodes as categorical variable
# Add distances from each small cluster to the large clusters (using actual column names from Excel)

# Build independent variables list based on feature switches
independent_vars = []

# Add archetype if enabled
if INCLUDE_ARCHETYPE:
    independent_vars.append("archetype")

# Add distances if enabled (using simplified distance features)
if INCLUDE_DISTANCES:
    independent_vars.extend([
        "distance_from_large_cluster",  # Average distance from small clusters to nearest large
        "distance_Small_cluster1_to_Small_cluster2_km"  # Distance between the two small clusters
    ])

# Add all other features (economic and system parameters)
independent_vars.extend([
    "total_demand_TWh",
    "demand_level_ratio",
    "unbalance_ratio",
    "electricity_availability_small",
    "import_availability_ratio",
    "electricity_price_avg",
    "hydrogen_import_price"
])

print(f"\n{'='*80}")
print(f"FEATURE CONFIGURATION:")
print(f"  - Include Archetype: {INCLUDE_ARCHETYPE}")
print(f"  - Include Distances: {INCLUDE_DISTANCES}")
print(f"  - Total features: {len(independent_vars)}")
print(f"{'='*80}\n")

# =============================================================================
# Load and preprocess data
# =============================================================================
print("Loading data from Excel files...")

# Load data
extracted_results_path = os.path.join(results_folder, "extracted_results.xlsx")
df_merged = pd.read_excel(extracted_results_path)
print(f"Loaded {len(df_merged)} runs from extracted_results.xlsx")

# =============================================================================
# Create derived distance variables
# =============================================================================
print("\nCalculating distance to nearest large cluster for each small cluster...")

# For each small cluster, calculate the minimum distance to the nearest large cluster
df_merged['Small_cluster1_to_nearest_Large'] = df_merged[[
    'distance_Large_cluster1_to_Small_cluster1_km',
    'distance_Large_cluster2_to_Small_cluster1_km'
]].min(axis=1)

df_merged['Small_cluster2_to_nearest_Large'] = df_merged[[
    'distance_Large_cluster1_to_Small_cluster2_km',
    'distance_Large_cluster2_to_Small_cluster2_km'
]].min(axis=1)

# Calculate average distance from small clusters to nearest large cluster
df_merged['distance_from_large_cluster'] = (df_merged['Small_cluster1_to_nearest_Large'] +
                                              df_merged['Small_cluster2_to_nearest_Large']) / 2

print(f"\nDistance statistics:")
print(f"  Small_cluster1 to nearest Large: mean = {df_merged['Small_cluster1_to_nearest_Large'].mean():.2f} km, "
      f"min = {df_merged['Small_cluster1_to_nearest_Large'].min():.2f} km, "
      f"max = {df_merged['Small_cluster1_to_nearest_Large'].max():.2f} km")
print(f"  Small_cluster2 to nearest Large: mean = {df_merged['Small_cluster2_to_nearest_Large'].mean():.2f} km, "
      f"min = {df_merged['Small_cluster2_to_nearest_Large'].min():.2f} km, "
      f"max = {df_merged['Small_cluster2_to_nearest_Large'].max():.2f} km")
print(f"  Average distance from Large clusters: mean = {df_merged['distance_from_large_cluster'].mean():.2f} km, "
      f"min = {df_merged['distance_from_large_cluster'].min():.2f} km, "
      f"max = {df_merged['distance_from_large_cluster'].max():.2f} km")
print(f"  Distance between Small clusters: mean = {df_merged['distance_Small_cluster1_to_Small_cluster2_km'].mean():.2f} km, "
      f"min = {df_merged['distance_Small_cluster1_to_Small_cluster2_km'].min():.2f} km, "
      f"max = {df_merged['distance_Small_cluster1_to_Small_cluster2_km'].max():.2f} km")

# =============================================================================
# Create dependent variables
# =============================================================================
print("\nCreating dependent variables...")

# 1) Electrolyzer installed in Small_cluster1 and Small_cluster2: binary (1 if size > 0)
df_merged['Electrolyzer_small1_installed'] = (df_merged['Small_cluster1_Electrolyzer_small'] > 0).astype(int)
df_merged['Electrolyzer_small2_installed'] = (df_merged['Small_cluster2_Electrolyzer_small'] > 0).astype(int)

# 2) Electrolyzer sizes
df_merged['Electrolyzer_small1_size'] = df_merged['Small_cluster1_Electrolyzer_small']
df_merged['Electrolyzer_small2_size'] = df_merged['Small_cluster2_Electrolyzer_small']

# 3) Surplus in small clusters: hydrogen produced - hydrogen used = network outflow
df_merged['Surplus_in_small_cluster1'] = df_merged['Small_cluster1_hydrogen_network_outflow_sum']
df_merged['Surplus_in_small_cluster2'] = df_merged['Small_cluster2_hydrogen_network_outflow_sum']

# 4) Pipeline supply: binary (1 if electrolyzer installed AND exports to large clusters)
df_merged['Pipeline_supply_small1'] = ((df_merged['Electrolyzer_small1_installed'] == 1) &
                                        (df_merged['Surplus_in_small_cluster1'] > 0)).astype(int)
df_merged['Pipeline_supply_small2'] = ((df_merged['Electrolyzer_small2_installed'] == 1) &
                                        (df_merged['Surplus_in_small_cluster2'] > 0)).astype(int)

# 5) Pipeline_type: classificazione tra highP e lowP pipeline per ogni connessione
# Nomi delle colonne delle pipeline per connessioni specifiche
# Note: column names in Excel are like "Large_cluster1_to_Small_cluster1_hydrogenPipelineOnshore_highP"
pipeline_connections = [
    ('Large_cluster1', 'Small_cluster1'),
    ('Large_cluster1', 'Small_cluster2'),
    ('Large_cluster2', 'Small_cluster1'),
    ('Large_cluster2', 'Small_cluster2'),
    ('Large_cluster1', 'Large_cluster2'),
    ('Small_cluster1', 'Small_cluster2')
]

print(f"\nSearching for pipeline columns:")

for node_a, node_b in pipeline_connections:
    # Cerca le colonne per questa connessione specifica
    # Format: NodeA_to_NodeB_hydrogenPipelineOnshore_highP/lowP
    highp_col = f'{node_a}_to_{node_b}_hydrogenPipelineOnshore_highP'
    lowp_col = f'{node_a}_to_{node_b}_hydrogenPipelineOnshore_lowP'

    # Check if columns exist
    highp_exists = highp_col in df_merged.columns
    lowp_exists = lowp_col in df_merged.columns

    print(f"  Connection {node_a}_to_{node_b}:")
    print(f"    - High pressure column: {highp_col if highp_exists else 'NOT FOUND'}")
    print(f"    - Low pressure column: {lowp_col if lowp_exists else 'NOT FOUND'}")

    # Crea variabili per questa connessione
    if highp_exists:
        df_merged[f'{node_a}_to_{node_b}_highP'] = df_merged[highp_col]
    else:
        df_merged[f'{node_a}_to_{node_b}_highP'] = 0

    if lowp_exists:
        df_merged[f'{node_a}_to_{node_b}_lowP'] = df_merged[lowp_col]
    else:
        df_merged[f'{node_a}_to_{node_b}_lowP'] = 0

# Totali per high e low pressure (somma di tutte le connessioni)
# Focus only on connections between Small and Large clusters for the analysis
small_large_connections = [
    ('Large_cluster1', 'Small_cluster1'),
    ('Large_cluster1', 'Small_cluster2'),
    ('Large_cluster2', 'Small_cluster1'),
    ('Large_cluster2', 'Small_cluster2')
]

df_merged['highP_total'] = sum(df_merged[f'{node_a}_to_{node_b}_highP']
                                 for node_a, node_b in small_large_connections)
df_merged['lowP_total'] = sum(df_merged[f'{node_a}_to_{node_b}_lowP']
                               for node_a, node_b in small_large_connections)

print(f"\n  - Total highP: mean = {df_merged['highP_total'].mean():.2f}, max = {df_merged['highP_total'].max():.2f}")
print(f"  - Total lowP: mean = {df_merged['lowP_total'].mean():.2f}, max = {df_merged['lowP_total'].max():.2f}")

# Pipeline_type: 0 = nessuna pipeline, 1 = solo lowP, 2 = solo highP, 3 = entrambe
def classify_pipeline_type(row):
    has_highp = row['highP_total'] > 0
    has_lowp = row['lowP_total'] > 0

    if has_highp and has_lowp:
        return 3  # Both
    elif has_highp:
        return 2  # Only highP
    elif has_lowp:
        return 1  # Only lowP
    else:
        return 0  # No pipeline

df_merged['Pipeline_type'] = df_merged.apply(classify_pipeline_type, axis=1)

# Per il decision tree, creiamo una variabile binaria: highP vs lowP (escludendo "nessuna pipeline")
# 1 = highP preferred, 0 = lowP preferred (solo per casi dove almeno una pipeline è installata)
df_merged['HighP_vs_LowP'] = -1  # -1 = non applicabile (no pipeline installed)
mask_pipeline = df_merged['Pipeline_type'] > 0
df_merged.loc[mask_pipeline, 'HighP_vs_LowP'] = (df_merged.loc[mask_pipeline, 'highP_total'] >
                                                   df_merged.loc[mask_pipeline, 'lowP_total']).astype(int)

print("\nDependent variables created:")
print(f"  - Electrolyzer_small1_installed: {df_merged['Electrolyzer_small1_installed'].sum()} installed out of {len(df_merged)}")
print(f"  - Electrolyzer_small2_installed: {df_merged['Electrolyzer_small2_installed'].sum()} installed out of {len(df_merged)}")
print(f"  - Electrolyzer_small1_size: mean = {df_merged['Electrolyzer_small1_size'].mean():.2f} MW")
print(f"  - Electrolyzer_small2_size: mean = {df_merged['Electrolyzer_small2_size'].mean():.2f} MW")
print(f"  - Surplus_in_small_cluster1: mean = {df_merged['Surplus_in_small_cluster1'].mean():.2f} TWh")
print(f"  - Surplus_in_small_cluster2: mean = {df_merged['Surplus_in_small_cluster2'].mean():.2f} TWh")
print(f"  - Pipeline_supply_small1: {df_merged['Pipeline_supply_small1'].sum()} cases out of {len(df_merged)}")
print(f"  - Pipeline_supply_small2: {df_merged['Pipeline_supply_small2'].sum()} cases out of {len(df_merged)}")
print(f"  - Pipeline_type distribution: {df_merged['Pipeline_type'].value_counts().to_dict()}")
print(f"  - HighP_vs_LowP (where pipeline exists): {(df_merged['HighP_vs_LowP'] == 1).sum()} highP, {(df_merged['HighP_vs_LowP'] == 0).sum()} lowP")

# =============================================================================
# Preprocess data
# =============================================================================
if do_preprocessing:
    print("\nPreprocessing data...")

    df_filtered = df_merged.copy()
    print(f"Using {len(df_filtered)} runs")

    # Check for missing values in independent variables
    print("\nChecking missing values in independent variables:")
    for var in independent_vars:
        if var in df_filtered.columns:
            missing = df_filtered[var].isna().sum()
            print(f"  - {var}: {missing} missing values")
        else:
            print(f"  - {var}: COLUMN NOT FOUND!")

    # Handle archetype as categorical variable with one-hot encoding
    if INCLUDE_ARCHETYPE and 'archetype' in df_filtered.columns:
        print("\nEncoding archetype as categorical variable...")
        # Create dummy variables for archetype
        archetype_dummies = pd.get_dummies(df_filtered['archetype'], prefix='archetype', drop_first=False)
        df_filtered = pd.concat([df_filtered, archetype_dummies], axis=1)
        print(f"  Created {len(archetype_dummies.columns)} archetype dummy variables: {list(archetype_dummies.columns)}")
        print(f"  Distribution of archetypes in data:")
        for col in archetype_dummies.columns:
            count = df_filtered[col].sum()
            print(f"    {col}: {count} runs ({count/len(df_filtered)*100:.1f}%)")

        # Remove original archetype column from independent vars and add dummies
        independent_vars_processed = [v for v in independent_vars if v != 'archetype']
        independent_vars_processed.extend(archetype_dummies.columns.tolist())
        print(f"  Updated independent variables (now {len(independent_vars_processed)} features)")
    elif INCLUDE_ARCHETYPE and 'archetype' not in df_filtered.columns:
        independent_vars_processed = independent_vars.copy()
        print("\nWARNING: 'archetype' was set to be included but column not found in data")
    elif not INCLUDE_ARCHETYPE:
        # Remove archetype from independent vars if present
        independent_vars_processed = [v for v in independent_vars if v != 'archetype']
        print(f"\nArchetype excluded from analysis. Using {len(independent_vars_processed)} features.")
    else:
        independent_vars_processed = independent_vars.copy()
        print("\nWARNING: 'archetype' column not found in data")

    # Fill missing values with 0 or median
    numeric_cols = [v for v in independent_vars_processed if v in df_filtered.columns]
    df_filtered[numeric_cols] = df_filtered[numeric_cols].fillna(0)

    # Store preprocessed data
    data_preprocessed = {"cap": df_filtered, "independent_vars": independent_vars_processed}
    print(f"\nPreprocessing complete. Final dataset: {len(data_preprocessed['cap'])} runs")
else:
    data_preprocessed = {"cap": df_merged, "independent_vars": independent_vars}

# Split set
(data_train, data_test) = train_test_split(data_preprocessed["cap"],
                                                      test_size=0.1,
                                                      random_state=42)

# =============================================================================
# Archetype-specific analysis setup
# =============================================================================
# Create list of datasets to analyze: always include combined, optionally add archetype-specific
datasets_to_analyze = [
    ('combined', data_train, data_test, results_folder)
]

if ANALYZE_BY_ARCHETYPE and 'archetype' in data_preprocessed["cap"].columns:
    print("\n" + "="*80)
    print("ARCHETYPE-SPECIFIC ANALYSIS ENABLED")
    print("="*80)
    print("Will create separate decision trees for each archetype in addition to combined analysis...")

    archetypes = sorted(data_preprocessed["cap"]['archetype'].unique())
    print(f"Found {len(archetypes)} archetypes: {archetypes}")

    # Create datasets for each archetype
    archetype_data = {}
    for arch in archetypes:
        arch_mask_train = data_train['archetype'] == arch
        arch_mask_test = data_test['archetype'] == arch

        arch_results_folder = os.path.join(results_folder, f"archetype_{arch}")
        os.makedirs(arch_results_folder, exist_ok=True)

        archetype_data[arch] = {
            'train': data_train[arch_mask_train].copy(),
            'test': data_test[arch_mask_test].copy(),
            'results_folder': arch_results_folder
        }

        print(f"\nArchetype {arch}:")
        print(f"  Training: {len(archetype_data[arch]['train'])} runs")
        print(f"  Testing: {len(archetype_data[arch]['test'])} runs")
        print(f"  Results folder: {arch_results_folder}")

        # Add to datasets to analyze
        datasets_to_analyze.append((
            f'archetype_{arch}',
            archetype_data[arch]['train'],
            archetype_data[arch]['test'],
            arch_results_folder
        ))
else:
    print("\n" + "="*80)
    print("ARCHETYPE-SPECIFIC ANALYSIS DISABLED")
    print("Only combined analysis will be performed")
    print("="*80)

# Decision tree for local hydrogen production in small cluster

def _select_existing_columns(df, cols: List[str]):
    """Helper function to select only columns that exist in the dataframe."""
    existing = [c for c in cols if c in df.columns]
    if not existing:
        raise ValueError("None of the requested columns are present in the dataframe.")
    return existing


# =============================================================================
# MAIN ANALYSIS LOOP - Process each dataset (combined + archetype-specific)
# =============================================================================
for dataset_name, data_train, data_test, current_results_folder in datasets_to_analyze:

    print("\n" + "="*80)
    print("="*80)
    print(f"ANALYZING DATASET: {dataset_name.upper()}")
    print("="*80)
    print("="*80)

    if len(data_train) == 0 or len(data_test) == 0:
        print(f"\nWARNING: Skipping {dataset_name} - insufficient data")
        print(f"  Training samples: {len(data_train)}, Test samples: {len(data_test)}")
        continue

    # Prepare feature matrix with processed independent variables
    X_cols = _select_existing_columns(data_train, data_preprocessed["independent_vars"])
    X_train = data_train[X_cols].fillna(0)
    X_test = data_test[X_cols].fillna(0)

    print(f"\nDataset: {dataset_name}")
    print(f"Using {len(X_cols)} features: {X_cols}")
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"Results will be saved to: {current_results_folder}")

# =============================================================================
# 1) DECISION TREE: Electrolyzer Installation in Small_cluster1
# =============================================================================
print("\n" + "="*80)
print("DECISION TREE 1: Electrolyzer Installation in Small_cluster1")
print("="*80)

y_install_train = data_train["Electrolyzer_small1_installed"]
y_install_test = data_test["Electrolyzer_small1_installed"]

print(f"\nTarget distribution (train): {y_install_train.value_counts().to_dict()}")
print(f"Target distribution (test): {y_install_test.value_counts().to_dict()}")

# Train decision tree classifier
clf_install = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=0,
            class_weight="balanced"
        )
clf_install.fit(X_train, y_install_train)
pred_install = clf_install.predict(X_test)

# Metrics
print(f"\nAccuracy: {accuracy_score(y_install_test, pred_install):.4f}")
print("\nClassification Report:")
print(classification_report(y_install_test, pred_install))

# Feature importances
print("\nTop 10 Feature Importances:")
feature_importance = sorted(zip(X_cols, clf_install.feature_importances_),
                           key=lambda x: -x[1])
for i, (feature, importance) in enumerate(feature_importance[:10], 1):
    print(f"{i}. {feature}: {importance:.4f}")

# Export tree as text
tree_text_install = export_text(clf_install, feature_names=X_cols)
print("\nDecision Tree Structure:")
print(tree_text_install)

# Plot and save tree
plt.figure(figsize=(40, 20))
plot_tree(clf_install, feature_names=X_cols,
         class_names=["Not Installed", "Installed"],
         filled=True, rounded=True, fontsize=13)
plt.title("Decision Tree - Electrolyzer Installation in Small Cluster", fontsize=16)
plt.tight_layout()
save_path_1 = os.path.join(current_results_folder, "decision_tree_electrolyzer_install.png")
plt.savefig(save_path_1, dpi=300, bbox_inches="tight")
print(f"\nTree plot saved as: {save_path_1}")
plt.close()

# =============================================================================
# 1b) RANDOM FOREST: Electrolyzer Installation in Small_cluster1
# =============================================================================
print("\n" + "="*80)
print("RANDOM FOREST 1: Electrolyzer Installation in Small_cluster1")
print("="*80)

# Train Random Forest classifier
rf_install = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=3,
    min_samples_split=6,
    random_state=0,
    class_weight="balanced",
    n_jobs=-1
)
rf_install.fit(X_train, y_install_train)
pred_rf_install = rf_install.predict(X_test)

# Metrics
print(f"\nAccuracy: {accuracy_score(y_install_test, pred_rf_install):.4f}")
print("\nClassification Report:")
print(classification_report(y_install_test, pred_rf_install))

# Feature importances
print("\nTop 10 Feature Importances (Random Forest):")
feature_importance_rf = sorted(zip(X_cols, rf_install.feature_importances_),
                              key=lambda x: -x[1])
for i, (feature, importance) in enumerate(feature_importance_rf[:10], 1):
    print(f"{i}. {feature}: {importance:.4f}")

# Compare with Decision Tree
print("\nComparison Decision Tree vs Random Forest:")
print(f"  Decision Tree Accuracy: {accuracy_score(y_install_test, pred_install):.4f}")
print(f"  Random Forest Accuracy: {accuracy_score(y_install_test, pred_rf_install):.4f}")
print(f"  Improvement: {(accuracy_score(y_install_test, pred_rf_install) - accuracy_score(y_install_test, pred_install)):.4f}")

# Create comparison plot: Feature Importances DT vs RF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Decision Tree importances
dt_importances = pd.Series(clf_install.feature_importances_, index=X_cols).sort_values(ascending=False)
dt_importances[:10].plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_xlabel('Importance', fontsize=12)
ax1.set_title('Decision Tree - Top 10 Features\nElectrolyzer Installation (Small_cluster1)', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Random Forest importances
rf_importances = pd.Series(rf_install.feature_importances_, index=X_cols).sort_values(ascending=False)
rf_importances[:10].plot(kind='barh', ax=ax2, color='forestgreen')
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title('Random Forest - Top 10 Features\nElectrolyzer Installation (Small_cluster1)', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
save_path_rf1 = os.path.join(current_results_folder, "rf_comparison_electrolyzer_install_small1.png")
plt.savefig(save_path_rf1, dpi=300, bbox_inches="tight")
print(f"\nRandom Forest comparison plot saved as: {save_path_rf1}")
plt.close()

# =============================================================================
# 2) DECISION TREE: Pipeline Supply from Small to Large Cluster
# =============================================================================
print("\n" + "="*80)
print("DECISION TREE 2: Pipeline Supply from Small to Large Cluster")
print("="*80)

def pipeline_supply_label(df):
    """
    Create binary label: 1 if Large clusters are receiving hydrogen from Small clusters
    (checking if Large_cluster1_hydrogen_network_inflow_sum OR Large_cluster2_hydrogen_network_inflow_sum > threshold)
    """
    THRESHOLD = 0.1  # Threshold to consider supply as active

    # Check if inflow columns exist
    if "Large_cluster1_hydrogen_network_inflow_sum" not in df.columns and \
       "Large_cluster2_hydrogen_network_inflow_sum" not in df.columns:
        raise ValueError("Required columns for pipeline supply label are missing: Large_cluster1_hydrogen_network_inflow_sum and Large_cluster2_hydrogen_network_inflow_sum")

    # Check if either Large cluster is receiving hydrogen
    large1_receiving = False
    large2_receiving = False

    if "Large_cluster1_hydrogen_network_inflow_sum" in df.columns:
        large1_receiving = df["Large_cluster1_hydrogen_network_inflow_sum"] > THRESHOLD

    if "Large_cluster2_hydrogen_network_inflow_sum" in df.columns:
        large2_receiving = df["Large_cluster2_hydrogen_network_inflow_sum"] > THRESHOLD

    # Return 1 if at least one Large cluster is receiving hydrogen
    return (large1_receiving | large2_receiving).astype(int)

y_pipe_train = pipeline_supply_label(data_train)
y_pipe_test = pipeline_supply_label(data_test)

print(f"\nTarget distribution (train): {y_pipe_train.value_counts().to_dict()}")
print(f"Target distribution (test): {y_pipe_test.value_counts().to_dict()}")
print(f"\nLarge clusters receive hydrogen supply in {y_pipe_train.mean()*100:.1f}% of training runs")

# Show statistics of Large cluster hydrogen inflow
if "Large_cluster1_hydrogen_network_inflow_sum" in data_train.columns:
    print(f"\nLarge_cluster1 hydrogen inflow statistics:")
    print(f"  Mean: {data_train['Large_cluster1_hydrogen_network_inflow_sum'].mean():.2f}")
    print(f"  Max: {data_train['Large_cluster1_hydrogen_network_inflow_sum'].max():.2f}")
    print(f"  Non-zero count: {(data_train['Large_cluster1_hydrogen_network_inflow_sum'] > 0.1).sum()}")

if "Large_cluster2_hydrogen_network_inflow_sum" in data_train.columns:
    print(f"\nLarge_cluster2 hydrogen inflow statistics:")
    print(f"  Mean: {data_train['Large_cluster2_hydrogen_network_inflow_sum'].mean():.2f}")
    print(f"  Max: {data_train['Large_cluster2_hydrogen_network_inflow_sum'].max():.2f}")
    print(f"  Non-zero count: {(data_train['Large_cluster2_hydrogen_network_inflow_sum'] > 0.1).sum()}")

# Train decision tree classifier
clf_pipe = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=0,
            class_weight="balanced"
        )
clf_pipe.fit(X_train, y_pipe_train)
pred_pipe = clf_pipe.predict(X_test)

# Metrics
print(f"\nAccuracy: {accuracy_score(y_pipe_test, pred_pipe):.4f}")
print("\nClassification Report:")

# Determine available labels in test set to avoid classification report error
available_labels_test = sorted(y_pipe_test.unique())
target_names_map = {0: "No Supply", 1: "Supply to Large"}
target_names_filtered = [target_names_map[label] for label in available_labels_test]

print(classification_report(y_pipe_test, pred_pipe,
                           labels=available_labels_test,
                           target_names=target_names_filtered,
                           zero_division=0))

# Feature importances
print("\nTop 10 Feature Importances:")
feature_importance_pipe = sorted(zip(X_cols, clf_pipe.feature_importances_),
                                key=lambda x: -x[1])
for i, (feature, importance) in enumerate(feature_importance_pipe[:10], 1):
    print(f"{i}. {feature}: {importance:.4f}")

# Export tree as text
tree_text_pipe = export_text(clf_pipe, feature_names=X_cols)
print("\nDecision Tree Structure:")
print(tree_text_pipe)

# Plot and save tree
plt.figure(figsize=(40, 20))
plot_tree(clf_pipe, feature_names=X_cols,
         class_names=["No Supply", "Supply to Large"],
         filled=True, rounded=True, fontsize=13)
plt.title("Decision Tree - Pipeline Supply from Small to Large Cluster", fontsize=16)
plt.tight_layout()
save_path_2 = os.path.join(current_results_folder, "decision_tree_pipeline_supply_small_to_large.png")
plt.savefig(save_path_2, dpi=300, bbox_inches="tight")
print(f"\nTree plot saved as: {save_path_2}")
plt.close()

# =============================================================================
# 2b) RANDOM FOREST: Pipeline Supply from Small to Large Cluster
# =============================================================================
print("\n" + "="*80)
print("RANDOM FOREST 2: Pipeline Supply from Small to Large Cluster")
print("="*80)

# Train Random Forest classifier
rf_pipe = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=3,
    min_samples_split=6,
    random_state=0,
    class_weight="balanced",
    n_jobs=-1
)
rf_pipe.fit(X_train, y_pipe_train)
pred_rf_pipe = rf_pipe.predict(X_test)

# Metrics
print(f"\nAccuracy: {accuracy_score(y_pipe_test, pred_rf_pipe):.4f}")
print("\nClassification Report:")
print(classification_report(y_pipe_test, pred_rf_pipe,
                           labels=available_labels_test,
                           target_names=target_names_filtered,
                           zero_division=0))

# Feature importances
print("\nTop 10 Feature Importances (Random Forest):")
feature_importance_rf_pipe = sorted(zip(X_cols, rf_pipe.feature_importances_),
                                   key=lambda x: -x[1])
for i, (feature, importance) in enumerate(feature_importance_rf_pipe[:10], 1):
    print(f"{i}. {feature}: {importance:.4f}")

# Compare with Decision Tree
print("\nComparison Decision Tree vs Random Forest:")
print(f"  Decision Tree Accuracy: {accuracy_score(y_pipe_test, pred_pipe):.4f}")
print(f"  Random Forest Accuracy: {accuracy_score(y_pipe_test, pred_rf_pipe):.4f}")
print(f"  Improvement: {(accuracy_score(y_pipe_test, pred_rf_pipe) - accuracy_score(y_pipe_test, pred_pipe)):.4f}")

# Create comparison plot: Feature Importances DT vs RF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Decision Tree importances
dt_importances_pipe = pd.Series(clf_pipe.feature_importances_, index=X_cols).sort_values(ascending=False)
dt_importances_pipe[:10].plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_xlabel('Importance', fontsize=12)
ax1.set_title('Decision Tree - Top 10 Features\nPipeline Supply', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Random Forest importances
rf_importances_pipe = pd.Series(rf_pipe.feature_importances_, index=X_cols).sort_values(ascending=False)
rf_importances_pipe[:10].plot(kind='barh', ax=ax2, color='forestgreen')
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title('Random Forest - Top 10 Features\nPipeline Supply', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
save_path_rf2 = os.path.join(current_results_folder, "rf_comparison_pipeline_supply.png")
plt.savefig(save_path_rf2, dpi=300, bbox_inches="tight")
print(f"\nRandom Forest comparison plot saved as: {save_path_rf2}")
plt.close()

# # =============================================================================
# # 3) DECISION TREE: Pipeline Type Choice (High Pressure vs Low Pressure)
# # =============================================================================
# print("\n" + "="*80)
# print("DECISION TREE 3: Pipeline Type Choice (High Pressure vs Low Pressure)")
# print("="*80)
#
# # Filter only runs where at least one pipeline is installed
# data_train_pipeline = data_train[data_train['HighP_vs_LowP'] >= 0].copy()
# data_test_pipeline = data_test[data_test['HighP_vs_LowP'] >= 0].copy()
#
# print(f"\nFiltered to runs with pipeline installed:")
# print(f"  Training set: {len(data_train_pipeline)} runs")
# print(f"  Test set: {len(data_test_pipeline)} runs")
#
# if len(data_train_pipeline) > 10 and len(data_test_pipeline) > 0:
#     # Prepare features
#     X_train_pipe_type = data_train_pipeline[X_cols].fillna(0)
#     X_test_pipe_type = data_test_pipeline[X_cols].fillna(0)
#
#     y_pipe_type_train = data_train_pipeline['HighP_vs_LowP']
#     y_pipe_type_test = data_test_pipeline['HighP_vs_LowP']
#
#     print(f"\nTarget distribution (train): {y_pipe_type_train.value_counts().to_dict()}")
#     print(f"Target distribution (test): {y_pipe_type_test.value_counts().to_dict()}")
#
#     # Check if we have both classes
#     if len(y_pipe_type_train.unique()) > 1:
#         # Train decision tree classifier
#         clf_pipe_type = DecisionTreeClassifier(
#             max_depth=4,
#             min_samples_leaf=5,
#             min_samples_split=10,
#             random_state=0,
#             class_weight="balanced"
#         )
#         clf_pipe_type.fit(X_train_pipe_type, y_pipe_type_train)
#         pred_pipe_type = clf_pipe_type.predict(X_test_pipe_type)
#
#         # Metrics
#         print(f"\nAccuracy: {accuracy_score(y_pipe_type_test, pred_pipe_type):.4f}")
#         print("\nClassification Report:")
#         print(classification_report(y_pipe_type_test, pred_pipe_type,
#                                    target_names=["Low Pressure", "High Pressure"]))
#
#         # Feature importances
#         print("\nTop 10 Feature Importances:")
#         feature_importance_pipe_type = sorted(zip(X_cols, clf_pipe_type.feature_importances_),
#                                              key=lambda x: -x[1])
#         for i, (feature, importance) in enumerate(feature_importance_pipe_type[:10], 1):
#             print(f"{i}. {feature}: {importance:.4f}")
#
#         # Export tree as text
#         tree_text_pipe_type = export_text(clf_pipe_type, feature_names=X_cols)
#         print("\nDecision Tree Structure:")
#         print(tree_text_pipe_type)
#
#         # Plot and save tree
#         plt.figure(figsize=(40, 20))
#         plot_tree(clf_pipe_type, feature_names=X_cols,
#                  class_names=["Low Pressure", "High Pressure"],
#                  filled=True, rounded=True, fontsize=13)
#         plt.title("Decision Tree - Pipeline Type Choice (High Pressure vs Low Pressure)", fontsize=16)
#         save_path_3 = os.path.join(results_folder, "decision_tree_pipeline_type.png")
#         plt.savefig(save_path_3, dpi=300, bbox_inches="tight")
#         print(f"\nTree plot saved as: {save_path_3}")
#         plt.close()
#
#         # =============================================================================
#         # 3b) RANDOM FOREST: Pipeline Type Choice (High Pressure vs Low Pressure)
#         # =============================================================================
#         print("\n" + "="*80)
#         print("RANDOM FOREST 3: Pipeline Type Choice (High Pressure vs Low Pressure)")
#         print("="*80)
#
#         # Train Random Forest classifier
#         rf_pipe_type = RandomForestClassifier(
#             n_estimators=100,
#             max_depth=10,
#             min_samples_leaf=3,
#             min_samples_split=6,
#             random_state=0,
#             class_weight="balanced",
#             n_jobs=-1
#         )
#         rf_pipe_type.fit(X_train_pipe_type, y_pipe_type_train)
#         pred_rf_pipe_type = rf_pipe_type.predict(X_test_pipe_type)
#
#         # Metrics
#         print(f"\nAccuracy: {accuracy_score(y_pipe_type_test, pred_rf_pipe_type):.4f}")
#         print("\nClassification Report:")
#         print(classification_report(y_pipe_type_test, pred_rf_pipe_type,
#                                    target_names=["Low Pressure", "High Pressure"]))
#
#         # Feature importances
#         print("\nTop 10 Feature Importances (Random Forest):")
#         feature_importance_rf_pipe_type = sorted(zip(X_cols, rf_pipe_type.feature_importances_),
#                                                 key=lambda x: -x[1])
#         for i, (feature, importance) in enumerate(feature_importance_rf_pipe_type[:10], 1):
#             print(f"{i}. {feature}: {importance:.4f}")
#
#         # Compare with Decision Tree
#         print("\nComparison Decision Tree vs Random Forest:")
#         print(f"  Decision Tree Accuracy: {accuracy_score(y_pipe_type_test, pred_pipe_type):.4f}")
#         print(f"  Random Forest Accuracy: {accuracy_score(y_pipe_type_test, pred_rf_pipe_type):.4f}")
#         print(f"  Improvement: {(accuracy_score(y_pipe_type_test, pred_rf_pipe_type) - accuracy_score(y_pipe_type_test, pred_pipe_type)):.4f}")
#
#         # Create comparison plot: Feature Importances DT vs RF
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
#
#         # Decision Tree importances
#         dt_importances_type = pd.Series(clf_pipe_type.feature_importances_, index=X_cols).sort_values(ascending=False)
#         dt_importances_type[:10].plot(kind='barh', ax=ax1, color='steelblue')
#         ax1.set_xlabel('Importance', fontsize=12)
#         ax1.set_title('Decision Tree - Top 10 Features\nPipeline Type (High vs Low Pressure)', fontsize=14, fontweight='bold')
#         ax1.invert_yaxis()
#         ax1.grid(axis='x', alpha=0.3)
#
#         # Random Forest importances
#         rf_importances_type = pd.Series(rf_pipe_type.feature_importances_, index=X_cols).sort_values(ascending=False)
#         rf_importances_type[:10].plot(kind='barh', ax=ax2, color='forestgreen')
#         ax2.set_xlabel('Importance', fontsize=12)
#         ax2.set_title('Random Forest - Top 10 Features\nPipeline Type (High vs Low Pressure)', fontsize=14, fontweight='bold')
#         ax2.invert_yaxis()
#         ax2.grid(axis='x', alpha=0.3)
#
#         plt.tight_layout()
#         save_path_rf3 = os.path.join(results_folder, "rf_comparison_pipeline_type.png")
#         plt.savefig(save_path_rf3, dpi=300, bbox_inches="tight")
#         print(f"\nRandom Forest comparison plot saved as: {save_path_rf3}")
#         plt.close()
#
#     else:
#         print("\nWARNING: Only one class found in training data. Cannot train classifier.")
#         print(f"All runs use the same pipeline type: {'High Pressure' if y_pipe_type_train.iloc[0] == 1 else 'Low Pressure'}")
# else:
#     print("\nWARNING: Insufficient data for pipeline type classification.")
#     print("Not enough runs with pipeline installed to train the model.")

# =============================================================================
# 4) DECISION TREE: Small Cluster Outflow vs Large Cluster Outflow
# =============================================================================
print("\n" + "="*80)
print("DECISION TREE 4: Small Cluster Outflow vs Large Cluster Outflow")
print("="*80)

# Create binary target: 1 if Small outflow > Large outflow, 0 otherwise
def outflow_comparison_label(df):
    """
    Binary label: 1 if Small cluster outflow > Large cluster outflow
    """
    if "Small_cluster1_hydrogen_network_outflow_sum" in df.columns and \
       "Small_cluster2_hydrogen_network_outflow_sum" in df.columns and \
       "Large_cluster1_hydrogen_network_outflow_sum" in df.columns and \
       "Large_cluster2_hydrogen_network_outflow_sum" in df.columns:
        # Sum of both small clusters vs sum of both large clusters
        small_total = df["Small_cluster1_hydrogen_network_outflow_sum"] + df["Small_cluster2_hydrogen_network_outflow_sum"]
        large_total = df["Large_cluster1_hydrogen_network_outflow_sum"] + df["Large_cluster2_hydrogen_network_outflow_sum"]
        return (small_total > large_total).astype(int)
    else:
        raise ValueError("Required columns for outflow comparison are missing")

y_outflow_train = outflow_comparison_label(data_train)
y_outflow_test = outflow_comparison_label(data_test)

print(f"\nTarget distribution (train): {y_outflow_train.value_counts().to_dict()}")
print(f"Target distribution (test): {y_outflow_test.value_counts().to_dict()}")
print(f"\nSmall cluster outflow dominates in {y_outflow_train.mean()*100:.1f}% of training runs")

# Check if we have both classes
if len(y_outflow_train.unique()) > 1:
    # Train decision tree classifier
    clf_outflow = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=0,
            class_weight="balanced"
        )
    clf_outflow.fit(X_train, y_outflow_train)
    pred_outflow = clf_outflow.predict(X_test)

    # Metrics
    print(f"\nAccuracy: {accuracy_score(y_outflow_test, pred_outflow):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_outflow_test, pred_outflow,
                               labels=[0, 1],
                               target_names=["Large Outflow > Small", "Small Outflow > Large"],
                               zero_division=0))

    # Feature importances
    print("\nTop 10 Feature Importances:")
    feature_importance_outflow = sorted(zip(X_cols, clf_outflow.feature_importances_),
                                       key=lambda x: -x[1])
    for i, (feature, importance) in enumerate(feature_importance_outflow[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")

    # Export tree as text
    tree_text_outflow = export_text(clf_outflow, feature_names=X_cols)
    print("\nDecision Tree Structure:")
    print(tree_text_outflow)

    # Plot and save tree
    plt.figure(figsize=(40, 20))
    plot_tree(clf_outflow, feature_names=X_cols,
             class_names=["Large Outflow > Small", "Small Outflow > Large"],
             filled=True, rounded=True, fontsize=13)
    plt.title("Decision Tree - Small Cluster Outflow vs Large Cluster Outflow", fontsize=16)
    save_path_4 = os.path.join(current_results_folder, "decision_tree_outflow_comparison.png")
    plt.savefig(save_path_4, dpi=300, bbox_inches="tight")
    print(f"\nTree plot saved as: {save_path_4}")
    plt.close()

    # =============================================================================
    # 4b) RANDOM FOREST: Small Cluster Outflow vs Large Cluster Outflow
    # =============================================================================
    print("\n" + "="*80)
    print("RANDOM FOREST 4: Small Cluster Outflow vs Large Cluster Outflow")
    print("="*80)

    # Train Random Forest classifier
    rf_outflow = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=3,
        min_samples_split=6,
        random_state=0,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_outflow.fit(X_train, y_outflow_train)
    pred_rf_outflow = rf_outflow.predict(X_test)

    # Metrics
    print(f"\nAccuracy: {accuracy_score(y_outflow_test, pred_rf_outflow):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_outflow_test, pred_rf_outflow,
                               labels=[0, 1],
                               target_names=["Large Outflow > Small", "Small Outflow > Large"],
                               zero_division=0))

    # Feature importances
    print("\nTop 10 Feature Importances (Random Forest):")
    feature_importance_rf_outflow = sorted(zip(X_cols, rf_outflow.feature_importances_),
                                          key=lambda x: -x[1])
    for i, (feature, importance) in enumerate(feature_importance_rf_outflow[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")

    # Compare with Decision Tree
    print("\nComparison Decision Tree vs Random Forest:")
    print(f"  Decision Tree Accuracy: {accuracy_score(y_outflow_test, pred_outflow):.4f}")
    print(f"  Random Forest Accuracy: {accuracy_score(y_outflow_test, pred_rf_outflow):.4f}")
    print(f"  Improvement: {(accuracy_score(y_outflow_test, pred_rf_outflow) - accuracy_score(y_outflow_test, pred_outflow)):.4f}")

    # Create comparison plot: Feature Importances DT vs RF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Decision Tree importances
    dt_importances_outflow = pd.Series(clf_outflow.feature_importances_, index=X_cols).sort_values(ascending=False)
    dt_importances_outflow[:10].plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Importance', fontsize=12)
    ax1.set_title('Decision Tree - Top 10 Features\nSmall vs Large Cluster Outflow', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Random Forest importances
    rf_importances_outflow = pd.Series(rf_outflow.feature_importances_, index=X_cols).sort_values(ascending=False)
    rf_importances_outflow[:10].plot(kind='barh', ax=ax2, color='forestgreen')
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title('Random Forest - Top 10 Features\nSmall vs Large Cluster Outflow', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path_rf4 = os.path.join(current_results_folder, "rf_comparison_outflow.png")
    plt.savefig(save_path_rf4, dpi=300, bbox_inches="tight")
    print(f"\nRandom Forest comparison plot saved as: {save_path_rf4}")
    plt.close()

else:
    print("\nWARNING: Only one class found in data. Cannot train classifier.")
    if y_outflow_train.iloc[0] == 1:
        print("Small cluster ALWAYS has greater outflow than Large cluster.")
    else:
        print("Large cluster ALWAYS has greater outflow than Small cluster.")

# =============================================================================
# 5) DECISION TREE: Small Clusters Interconnection (Small1 to Small2)
# =============================================================================
print("\n" + "="*80)
print("DECISION TREE 5: Small Clusters Interconnection (Pipeline between Small Clusters)")
print("="*80)

# Create binary target: 1 if there is a pipeline connecting the two small clusters
# Check both high pressure and low pressure pipelines
THRESHOLD = 0.1  # Threshold to consider pipeline as installed (MW or capacity unit)

def small_clusters_connected_label(df):
    """
    Binary label: 1 if Small_cluster1 and Small_cluster2 are connected via pipeline
    (either high pressure or low pressure with capacity > THRESHOLD)
    """
    highp_col = 'Small_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_highP'
    lowp_col = 'Small_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_lowP'

    has_highp = False
    has_lowp = False

    if highp_col in df.columns:
        has_highp = df[highp_col] > THRESHOLD

    if lowp_col in df.columns:
        has_lowp = df[lowp_col] > THRESHOLD

    # Connected if either pipeline type exists
    return (has_highp | has_lowp).astype(int)

y_small_connect_train = small_clusters_connected_label(data_train)
y_small_connect_test = small_clusters_connected_label(data_test)

print(f"\nTarget distribution (train): {y_small_connect_train.value_counts().to_dict()}")
print(f"Target distribution (test): {y_small_connect_test.value_counts().to_dict()}")
print(f"\nSmall clusters are connected in {y_small_connect_train.mean()*100:.1f}% of training runs")

# Check capacities
highp_col = 'Small_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_highP'
lowp_col = 'Small_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_lowP'

if highp_col in data_train.columns:
    print(f"\nHigh Pressure pipeline capacity statistics:")
    print(f"  Mean: {data_train[highp_col].mean():.2f}")
    print(f"  Max: {data_train[highp_col].max():.2f}")
    print(f"  Non-zero count: {(data_train[highp_col] > THRESHOLD).sum()}")

if lowp_col in data_train.columns:
    print(f"\nLow Pressure pipeline capacity statistics:")
    print(f"  Mean: {data_train[lowp_col].mean():.2f}")
    print(f"  Max: {data_train[lowp_col].max():.2f}")
    print(f"  Non-zero count: {(data_train[lowp_col] > THRESHOLD).sum()}")

# Check if we have both classes
if len(y_small_connect_train.unique()) > 1:
    # Train decision tree classifier
    clf_small_connect = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=0,
            class_weight="balanced"
        )
    clf_small_connect.fit(X_train, y_small_connect_train)
    pred_small_connect = clf_small_connect.predict(X_test)

    # Metrics
    print(f"\nAccuracy: {accuracy_score(y_small_connect_test, pred_small_connect):.4f}")

    # Determine available labels in test set to avoid classification report error
    available_labels_test = sorted(y_small_connect_test.unique())
    target_names_map = {0: "Not Connected", 1: "Connected"}
    target_names_filtered = [target_names_map[label] for label in available_labels_test]

    print("\nClassification Report:")
    print(classification_report(y_small_connect_test, pred_small_connect,
                               labels=available_labels_test,
                               target_names=target_names_filtered,
                               zero_division=0))

    # Feature importances
    print("\nTop 10 Feature Importances:")
    feature_importance_small_connect = sorted(zip(X_cols, clf_small_connect.feature_importances_),
                                             key=lambda x: -x[1])
    for i, (feature, importance) in enumerate(feature_importance_small_connect[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")

    # Export tree as text
    tree_text_small_connect = export_text(clf_small_connect, feature_names=X_cols)
    print("\nDecision Tree Structure:")
    print(tree_text_small_connect)

    # Plot and save tree
    plt.figure(figsize=(40, 20))
    plot_tree(clf_small_connect, feature_names=X_cols,
             class_names=["Not Connected", "Connected"],
             filled=True, rounded=True, fontsize=13)
    plt.title("Decision Tree - Small Clusters Interconnection (Pipeline between Small Clusters)", fontsize=16)
    save_path_5 = os.path.join(current_results_folder, "decision_tree_small_clusters_connection.png")
    plt.savefig(save_path_5, dpi=300, bbox_inches="tight")
    print(f"\nTree plot saved as: {save_path_5}")
    plt.close()

    # =============================================================================
    # 5b) RANDOM FOREST: Small Clusters Interconnection
    # =============================================================================
    print("\n" + "="*80)
    print("RANDOM FOREST 5: Small Clusters Interconnection")
    print("="*80)

    # Train Random Forest classifier
    rf_small_connect = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=3,
        min_samples_split=6,
        random_state=0,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_small_connect.fit(X_train, y_small_connect_train)
    pred_rf_small_connect = rf_small_connect.predict(X_test)

    # Metrics
    print(f"\nAccuracy: {accuracy_score(y_small_connect_test, pred_rf_small_connect):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_small_connect_test, pred_rf_small_connect,
                               labels=available_labels_test,
                               target_names=target_names_filtered,
                               zero_division=0))

    # Feature importances
    print("\nTop 10 Feature Importances (Random Forest):")
    feature_importance_rf_small_connect = sorted(zip(X_cols, rf_small_connect.feature_importances_),
                                                key=lambda x: -x[1])
    for i, (feature, importance) in enumerate(feature_importance_rf_small_connect[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")

    # Compare with Decision Tree
    print("\nComparison Decision Tree vs Random Forest:")
    print(f"  Decision Tree Accuracy: {accuracy_score(y_small_connect_test, pred_small_connect):.4f}")
    print(f"  Random Forest Accuracy: {accuracy_score(y_small_connect_test, pred_rf_small_connect):.4f}")
    print(f"  Improvement: {(accuracy_score(y_small_connect_test, pred_rf_small_connect) - accuracy_score(y_small_connect_test, pred_small_connect)):.4f}")

    # Create comparison plot: Feature Importances DT vs RF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Decision Tree importances
    dt_importances_connect = pd.Series(clf_small_connect.feature_importances_, index=X_cols).sort_values(ascending=False)
    dt_importances_connect[:10].plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Importance', fontsize=12)
    ax1.set_title('Decision Tree - Top 10 Features\nSmall Clusters Interconnection', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Random Forest importances
    rf_importances_connect = pd.Series(rf_small_connect.feature_importances_, index=X_cols).sort_values(ascending=False)
    rf_importances_connect[:10].plot(kind='barh', ax=ax2, color='forestgreen')
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title('Random Forest - Top 10 Features\nSmall Clusters Interconnection', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path_rf5 = os.path.join(current_results_folder, "rf_comparison_small_clusters_connection.png")
    plt.savefig(save_path_rf5, dpi=300, bbox_inches="tight")
    print(f"\nRandom Forest comparison plot saved as: {save_path_rf5}")
    plt.close()

else:
    print("\nWARNING: Only one class found in data. Cannot train classifier.")
    if y_small_connect_train.iloc[0] == 1:
        print("Small clusters are ALWAYS connected via pipeline.")
    else:
        print("Small clusters are NEVER connected via pipeline.")

# =============================================================================
# 6) DECISION TREE: Full Network Connectivity (All Nodes Interconnected)
# =============================================================================
print("\n" + "="*80)
print("DECISION TREE 6: Full Network Connectivity (All Nodes Can Be Interconnected)")
print("="*80)

# Create binary target: 1 if all nodes are connected (forming a connected graph)
# We need to check if the network forms a connected graph where all 4 nodes can reach each other
# Nodes: Large_cluster1, Large_cluster2, Small_cluster1, Small_cluster2

def is_fully_connected(df):
    """
    Check if all 4 nodes form a connected network.
    A network is fully connected if you can reach any node from any other node.

    We check all possible connections between the 4 nodes:
    - Large_cluster1 <-> Large_cluster2
    - Large_cluster1 <-> Small_cluster1
    - Large_cluster1 <-> Small_cluster2
    - Large_cluster2 <-> Small_cluster1
    - Large_cluster2 <-> Small_cluster2
    - Small_cluster1 <-> Small_cluster2
    """
    THRESHOLD = 0.1

    # Define all possible connections (bidirectional)
    connections = [
        ('Large_cluster1', 'Large_cluster2'),
        ('Large_cluster1', 'Small_cluster1'),
        ('Large_cluster1', 'Small_cluster2'),
        ('Large_cluster2', 'Small_cluster1'),
        ('Large_cluster2', 'Small_cluster2'),
        ('Small_cluster1', 'Small_cluster2')
    ]

    # Build adjacency list for graph
    def build_graph(row):
        """Build adjacency list from pipeline capacities"""
        from collections import defaultdict
        graph = defaultdict(set)

        for node_a, node_b in connections:
            # Check both directions and both pipeline types
            for direction in [(node_a, node_b), (node_b, node_a)]:
                highp_col = f'{direction[0]}_to_{direction[1]}_hydrogenPipelineOnshore_highP'
                lowp_col = f'{direction[0]}_to_{direction[1]}_hydrogenPipelineOnshore_lowP'

                has_connection = False
                if highp_col in row.index and row[highp_col] > THRESHOLD:
                    has_connection = True
                if lowp_col in row.index and row[lowp_col] > THRESHOLD:
                    has_connection = True

                # If connection exists, add edge (bidirectional)
                if has_connection:
                    graph[direction[0]].add(direction[1])
                    graph[direction[1]].add(direction[0])

        return graph

    # Check connectivity using BFS/DFS
    def is_connected_graph(graph, nodes):
        """Check if all nodes are reachable from any starting node"""
        if not graph:
            return False

        # Start from first node
        start_node = nodes[0]
        visited = set()
        stack = [start_node]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            # Add neighbors to stack
            if node in graph:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)

        # Check if all nodes were visited
        return len(visited) == len(nodes)

    # Apply to each row
    nodes = ['Large_cluster1', 'Large_cluster2', 'Small_cluster1', 'Small_cluster2']

    if isinstance(df, pd.DataFrame):
        result = []
        for idx, row in df.iterrows():
            graph = build_graph(row)
            result.append(1 if is_connected_graph(graph, nodes) else 0)
        return pd.Series(result, index=df.index)
    else:
        # Single row (Series)
        graph = build_graph(df)
        return 1 if is_connected_graph(graph, nodes) else 0

y_full_connect_train = is_fully_connected(data_train)
y_full_connect_test = is_fully_connected(data_test)

print(f"\nTarget distribution (train): {y_full_connect_train.value_counts().to_dict()}")
print(f"Target distribution (test): {y_full_connect_test.value_counts().to_dict()}")
print(f"\nFull network connectivity achieved in {y_full_connect_train.mean()*100:.1f}% of training runs")

# Show detailed connection statistics
print("\n\nDetailed Connection Statistics (Training Set):")
all_connections = [
    ('Large_cluster1', 'Large_cluster2'),
    ('Large_cluster1', 'Small_cluster1'),
    ('Large_cluster1', 'Small_cluster2'),
    ('Large_cluster2', 'Small_cluster1'),
    ('Large_cluster2', 'Small_cluster2'),
    ('Small_cluster1', 'Small_cluster2')
]

for node_a, node_b in all_connections:
    highp_col = f'{node_a}_to_{node_b}_hydrogenPipelineOnshore_highP'
    lowp_col = f'{node_a}_to_{node_b}_hydrogenPipelineOnshore_lowP'

    count_highp = 0
    count_lowp = 0
    count_any = 0

    if highp_col in data_train.columns:
        count_highp = (data_train[highp_col] > 0.1).sum()
    if lowp_col in data_train.columns:
        count_lowp = (data_train[lowp_col] > 0.1).sum()

    if highp_col in data_train.columns or lowp_col in data_train.columns:
        has_highp = data_train[highp_col] > 0.1 if highp_col in data_train.columns else False
        has_lowp = data_train[lowp_col] > 0.1 if lowp_col in data_train.columns else False
        count_any = ((has_highp) | (has_lowp)).sum()

    print(f"  {node_a} <-> {node_b}:")
    print(f"    High-P: {count_highp} cases, Low-P: {count_lowp} cases, Any: {count_any} cases")

# Check if we have both classes
if len(y_full_connect_train.unique()) > 1:
    # Train decision tree classifier
    clf_full_connect = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=0,
        class_weight="balanced"
    )
    clf_full_connect.fit(X_train, y_full_connect_train)
    pred_full_connect = clf_full_connect.predict(X_test)

    # Metrics
    print(f"\nAccuracy: {accuracy_score(y_full_connect_test, pred_full_connect):.4f}")

    # Determine available labels in test set
    available_labels_test = sorted(y_full_connect_test.unique())
    target_names_map = {0: "Not Fully Connected", 1: "Fully Connected"}
    target_names_filtered = [target_names_map[label] for label in available_labels_test]

    print("\nClassification Report:")
    print(classification_report(y_full_connect_test, pred_full_connect,
                               labels=available_labels_test,
                               target_names=target_names_filtered,
                               zero_division=0))

    # Feature importances
    print("\nTop 10 Feature Importances:")
    feature_importance_full_connect = sorted(zip(X_cols, clf_full_connect.feature_importances_),
                                           key=lambda x: -x[1])
    for i, (feature, importance) in enumerate(feature_importance_full_connect[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")

    # Export tree as text
    tree_text_full_connect = export_text(clf_full_connect, feature_names=X_cols)
    print("\nDecision Tree Structure:")
    print(tree_text_full_connect)

    # Plot and save tree
    plt.figure(figsize=(40, 20))
    plot_tree(clf_full_connect, feature_names=X_cols,
             class_names=["Not Fully Connected", "Fully Connected"],
             filled=True, rounded=True, fontsize=13)
    plt.title("Decision Tree - Full Network Connectivity (All 4 Nodes Interconnected)", fontsize=16)
    save_path_6 = os.path.join(current_results_folder, "decision_tree_full_network_connectivity.png")
    plt.savefig(save_path_6, dpi=300, bbox_inches="tight")
    print(f"\nTree plot saved as: {save_path_6}")
    plt.close()

    # =============================================================================
    # 6b) RANDOM FOREST: Full Network Connectivity
    # =============================================================================
    print("\n" + "="*80)
    print("RANDOM FOREST 6: Full Network Connectivity")
    print("="*80)

    # Train Random Forest classifier
    rf_full_connect = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=3,
        min_samples_split=6,
        random_state=0,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_full_connect.fit(X_train, y_full_connect_train)
    pred_rf_full_connect = rf_full_connect.predict(X_test)

    # Metrics
    print(f"\nAccuracy: {accuracy_score(y_full_connect_test, pred_rf_full_connect):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_full_connect_test, pred_rf_full_connect,
                               labels=available_labels_test,
                               target_names=target_names_filtered,
                               zero_division=0))

    # Feature importances
    print("\nTop 10 Feature Importances (Random Forest):")
    feature_importance_rf_full_connect = sorted(zip(X_cols, rf_full_connect.feature_importances_),
                                               key=lambda x: -x[1])
    for i, (feature, importance) in enumerate(feature_importance_rf_full_connect[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")

    # Compare with Decision Tree
    print("\nComparison Decision Tree vs Random Forest:")
    print(f"  Decision Tree Accuracy: {accuracy_score(y_full_connect_test, pred_full_connect):.4f}")
    print(f"  Random Forest Accuracy: {accuracy_score(y_full_connect_test, pred_rf_full_connect):.4f}")
    print(f"  Improvement: {(accuracy_score(y_full_connect_test, pred_rf_full_connect) - accuracy_score(y_full_connect_test, pred_full_connect)):.4f}")

    # Create comparison plot: Feature Importances DT vs RF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Decision Tree importances
    dt_importances_full = pd.Series(clf_full_connect.feature_importances_, index=X_cols).sort_values(ascending=False)
    dt_importances_full[:10].plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Importance', fontsize=12)
    ax1.set_title('Decision Tree - Top 10 Features\nFull Network Connectivity', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Random Forest importances
    rf_importances_full = pd.Series(rf_full_connect.feature_importances_, index=X_cols).sort_values(ascending=False)
    rf_importances_full[:10].plot(kind='barh', ax=ax2, color='forestgreen')
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title('Random Forest - Top 10 Features\nFull Network Connectivity', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path_rf6 = os.path.join(current_results_folder, "rf_comparison_full_network_connectivity.png")
    plt.savefig(save_path_rf6, dpi=300, bbox_inches="tight")
    print(f"\nRandom Forest comparison plot saved as: {save_path_rf6}")
    plt.close()

else:
    print("\nWARNING: Only one class found in data. Cannot train classifier.")
    if y_full_connect_train.iloc[0] == 1:
        print("Network is ALWAYS fully connected.")
    else:
        print("Network is NEVER fully connected.")

# =============================================================================
# 7) DECISION TREE: Small Cluster Net Export (Outflow > Inflow)
# =============================================================================
print("\n" + "="*80)
print("DECISION TREE 7: Small Cluster Net Export (Outflow > Inflow)")
print("="*80)

def small_cluster_net_export_label(df):
    """
    Binary label: 1 if Small cluster exports more than it imports
    Checks if outflow > inflow for at least one small cluster
    """
    small1_net_export = (df["Small_cluster1_hydrogen_network_outflow_sum"] >
                         df["Small_cluster1_hydrogen_network_inflow_sum"])
    small2_net_export = (df["Small_cluster2_hydrogen_network_outflow_sum"] >
                         df["Small_cluster2_hydrogen_network_inflow_sum"])

    return (small1_net_export | small2_net_export).astype(int)

y_net_export_train = small_cluster_net_export_label(data_train)
y_net_export_test = small_cluster_net_export_label(data_test)

print(f"\nTarget distribution (train): {y_net_export_train.value_counts().to_dict()}")
print(f"Target distribution (test): {y_net_export_test.value_counts().to_dict()}")
print(f"\nSmall clusters are net exporters in {y_net_export_train.mean()*100:.1f}% of training runs")

# Statistics
print(f"\nSmall_cluster1 balance:")
print(f"  Avg inflow: {data_train['Small_cluster1_hydrogen_network_inflow_sum'].mean():.2f} TWh")
print(f"  Avg outflow: {data_train['Small_cluster1_hydrogen_network_outflow_sum'].mean():.2f} TWh")

print(f"\nSmall_cluster2 balance:")
print(f"  Avg inflow: {data_train['Small_cluster2_hydrogen_network_inflow_sum'].mean():.2f} TWh")
print(f"  Avg outflow: {data_train['Small_cluster2_hydrogen_network_outflow_sum'].mean():.2f} TWh")

if len(y_net_export_train.unique()) > 1:
    clf_net_export = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=0,
        class_weight="balanced"
    )
    clf_net_export.fit(X_train, y_net_export_train)
    pred_net_export = clf_net_export.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_net_export_test, pred_net_export):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_net_export_test, pred_net_export,
                               labels=[0, 1],
                               target_names=["Net Importer", "Net Exporter"],
                               zero_division=0))

    print("\nTop 10 Feature Importances:")
    feature_importance_net = sorted(zip(X_cols, clf_net_export.feature_importances_),
                                   key=lambda x: -x[1])
    for i, (feature, importance) in enumerate(feature_importance_net[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")

    tree_text_net = export_text(clf_net_export, feature_names=X_cols)
    print("\nDecision Tree Structure:")
    print(tree_text_net)

    plt.figure(figsize=(40, 20))
    plot_tree(clf_net_export, feature_names=X_cols,
             class_names=["Net Importer", "Net Exporter"],
             filled=True, rounded=True, fontsize=13)
    plt.title("Decision Tree - Small Cluster Net Export (Outflow > Inflow)", fontsize=16)
    save_path_7 = os.path.join(current_results_folder, "decision_tree_small_net_export.png")
    plt.savefig(save_path_7, dpi=300, bbox_inches="tight")
    print(f"\nTree plot saved as: {save_path_7}")
    plt.close()

    # Random Forest
    print("\n" + "="*80)
    print("RANDOM FOREST 7: Small Cluster Net Export")
    print("="*80)

    rf_net_export = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=3,
        min_samples_split=6,
        random_state=0,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_net_export.fit(X_train, y_net_export_train)
    pred_rf_net = rf_net_export.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_net_export_test, pred_rf_net):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_net_export_test, pred_rf_net,
                               labels=[0, 1],
                               target_names=["Net Importer", "Net Exporter"],
                               zero_division=0))

    print("\nTop 10 Feature Importances (Random Forest):")
    feature_importance_rf_net = sorted(zip(X_cols, rf_net_export.feature_importances_),
                                      key=lambda x: -x[1])
    for i, (feature, importance) in enumerate(feature_importance_rf_net[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    dt_imp_net = pd.Series(clf_net_export.feature_importances_, index=X_cols).sort_values(ascending=False)
    dt_imp_net[:10].plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Importance', fontsize=12)
    ax1.set_title('Decision Tree\nSmall Cluster Net Export', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    rf_imp_net = pd.Series(rf_net_export.feature_importances_, index=X_cols).sort_values(ascending=False)
    rf_imp_net[:10].plot(kind='barh', ax=ax2, color='forestgreen')
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title('Random Forest\nSmall Cluster Net Export', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path_rf7 = os.path.join(current_results_folder, "rf_comparison_small_net_export.png")
    plt.savefig(save_path_rf7, dpi=300, bbox_inches="tight")
    print(f"\nRandom Forest comparison plot saved as: {save_path_rf7}")
    plt.close()
else:
    print("\nWARNING: Only one class found. Cannot train decision tree.")

# =============================================================================
# 8) DECISION TREE: Small Cluster with Two Connections
# =============================================================================
print("\n" + "="*80)
print("DECISION TREE 8: Small Cluster with Two Connections (Including Small-Small Connection)")
print("="*80)

def small_cluster_two_connections_label(df):
    """
    Binary label: 1 if at least one Small cluster has two connections
    A Small cluster can have two connections if connected to:
    - Both Large clusters, OR
    - One Large cluster + the other Small cluster
    """
    # Connection between Small clusters
    small1_to_small2_highP = df["Small_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_highP"] > 0.1
    small1_to_small2_lowP = df["Small_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_lowP"] > 0.1
    small_to_small_connected = small1_to_small2_highP | small1_to_small2_lowP

    # Small_cluster1 connections to Large clusters
    small1_to_large1_highP = df["Large_cluster1_to_Small_cluster1_hydrogenPipelineOnshore_highP"] > 0.1
    small1_to_large1_lowP = df["Large_cluster1_to_Small_cluster1_hydrogenPipelineOnshore_lowP"] > 0.1
    small1_to_large2_highP = df["Large_cluster2_to_Small_cluster1_hydrogenPipelineOnshore_highP"] > 0.1
    small1_to_large2_lowP = df["Large_cluster2_to_Small_cluster1_hydrogenPipelineOnshore_lowP"] > 0.1

    small1_connected_to_large1 = small1_to_large1_highP | small1_to_large1_lowP
    small1_connected_to_large2 = small1_to_large2_highP | small1_to_large2_lowP
    small1_connected_to_small2 = small_to_small_connected

    # Count connections for Small_cluster1
    small1_num_connections = (small1_connected_to_large1.astype(int) +
                              small1_connected_to_large2.astype(int) +
                              small1_connected_to_small2.astype(int))
    small1_has_two_connections = small1_num_connections >= 2

    # Small_cluster2 connections to Large clusters
    small2_to_large1_highP = df["Large_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_highP"] > 0.1
    small2_to_large1_lowP = df["Large_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_lowP"] > 0.1
    small2_to_large2_highP = df["Large_cluster2_to_Small_cluster2_hydrogenPipelineOnshore_highP"] > 0.1
    small2_to_large2_lowP = df["Large_cluster2_to_Small_cluster2_hydrogenPipelineOnshore_lowP"] > 0.1

    small2_connected_to_large1 = small2_to_large1_highP | small2_to_large1_lowP
    small2_connected_to_large2 = small2_to_large2_highP | small2_to_large2_lowP
    small2_connected_to_small1 = small_to_small_connected

    # Count connections for Small_cluster2
    small2_num_connections = (small2_connected_to_large1.astype(int) +
                              small2_connected_to_large2.astype(int) +
                              small2_connected_to_small1.astype(int))
    small2_has_two_connections = small2_num_connections >= 2

    # At least one small cluster has two or more connections
    return (small1_has_two_connections | small2_has_two_connections).astype(int)

y_two_conn_train = small_cluster_two_connections_label(data_train)
y_two_conn_test = small_cluster_two_connections_label(data_test)

print(f"\nTarget distribution (train): {y_two_conn_train.value_counts().to_dict()}")
print(f"Target distribution (test): {y_two_conn_test.value_counts().to_dict()}")
print(f"\nAt least one Small cluster has two connections in {y_two_conn_train.mean()*100:.1f}% of training runs")

# Detailed statistics
small_to_small = ((data_train["Small_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_highP"] > 0.1) |
                  (data_train["Small_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_lowP"] > 0.1))

small1_to_large1 = ((data_train["Large_cluster1_to_Small_cluster1_hydrogenPipelineOnshore_highP"] > 0.1) |
                    (data_train["Large_cluster1_to_Small_cluster1_hydrogenPipelineOnshore_lowP"] > 0.1))
small1_to_large2 = ((data_train["Large_cluster2_to_Small_cluster1_hydrogenPipelineOnshore_highP"] > 0.1) |
                    (data_train["Large_cluster2_to_Small_cluster1_hydrogenPipelineOnshore_lowP"] > 0.1))
small2_to_large1 = ((data_train["Large_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_highP"] > 0.1) |
                    (data_train["Large_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_lowP"] > 0.1))
small2_to_large2 = ((data_train["Large_cluster2_to_Small_cluster2_hydrogenPipelineOnshore_highP"] > 0.1) |
                    (data_train["Large_cluster2_to_Small_cluster2_hydrogenPipelineOnshore_lowP"] > 0.1))

print(f"\nSmall-Small connection:")
print(f"  Small_cluster1 <-> Small_cluster2: {small_to_small.sum()} cases")

print(f"\nSmall_cluster1 connections:")
print(f"  Connected to Large_cluster1: {small1_to_large1.sum()} cases")
print(f"  Connected to Large_cluster2: {small1_to_large2.sum()} cases")
print(f"  Connected to Small_cluster2: {small_to_small.sum()} cases")
small1_num_conn = (small1_to_large1.astype(int) + small1_to_large2.astype(int) + small_to_small.astype(int))
print(f"  Total with 2+ connections: {(small1_num_conn >= 2).sum()} cases")

print(f"\nSmall_cluster2 connections:")
print(f"  Connected to Large_cluster1: {small2_to_large1.sum()} cases")
print(f"  Connected to Large_cluster2: {small2_to_large2.sum()} cases")
print(f"  Connected to Small_cluster1: {small_to_small.sum()} cases")
small2_num_conn = (small2_to_large1.astype(int) + small2_to_large2.astype(int) + small_to_small.astype(int))
print(f"  Total with 2+ connections: {(small2_num_conn >= 2).sum()} cases")

if len(y_two_conn_train.unique()) > 1:
    clf_two_conn = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=0,
        class_weight="balanced"
    )
    clf_two_conn.fit(X_train, y_two_conn_train)
    pred_two_conn = clf_two_conn.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_two_conn_test, pred_two_conn):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_two_conn_test, pred_two_conn,
                               labels=[0, 1],
                               target_names=["No Two Connections", "Has Two Connections"],
                               zero_division=0))

    print("\nTop 10 Feature Importances:")
    feature_importance_two_conn = sorted(zip(X_cols, clf_two_conn.feature_importances_),
                                        key=lambda x: -x[1])
    for i, (feature, importance) in enumerate(feature_importance_two_conn[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")

    tree_text_two_conn = export_text(clf_two_conn, feature_names=X_cols)
    print("\nDecision Tree Structure:")
    print(tree_text_two_conn)

    plt.figure(figsize=(40, 20))
    plot_tree(clf_two_conn, feature_names=X_cols,
             class_names=["No Two Connections", "Has Two Connections"],
             filled=True, rounded=True, fontsize=13)
    plt.title("Decision Tree - Small Cluster with Two Connections (to Both Large Clusters)", fontsize=16)
    save_path_8 = os.path.join(current_results_folder, "decision_tree_small_two_connections.png")
    plt.savefig(save_path_8, dpi=300, bbox_inches="tight")
    print(f"\nTree plot saved as: {save_path_8}")
    plt.close()

    # Random Forest
    print("\n" + "="*80)
    print("RANDOM FOREST 8: Small Cluster with Two Connections")
    print("="*80)

    rf_two_conn = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=3,
        min_samples_split=6,
        random_state=0,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_two_conn.fit(X_train, y_two_conn_train)
    pred_rf_two_conn = rf_two_conn.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_two_conn_test, pred_rf_two_conn):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_two_conn_test, pred_rf_two_conn,
                               labels=[0, 1],
                               target_names=["No Two Connections", "Has Two Connections"],
                               zero_division=0))

    print("\nTop 10 Feature Importances (Random Forest):")
    feature_importance_rf_two_conn = sorted(zip(X_cols, rf_two_conn.feature_importances_),
                                           key=lambda x: -x[1])
    for i, (feature, importance) in enumerate(feature_importance_rf_two_conn[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    dt_imp_two_conn = pd.Series(clf_two_conn.feature_importances_, index=X_cols).sort_values(ascending=False)
    dt_imp_two_conn[:10].plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Importance', fontsize=12)
    ax1.set_title('Decision Tree\nSmall Cluster Two Connections', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    rf_imp_two_conn = pd.Series(rf_two_conn.feature_importances_, index=X_cols).sort_values(ascending=False)
    rf_imp_two_conn[:10].plot(kind='barh', ax=ax2, color='forestgreen')
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title('Random Forest\nSmall Cluster Two Connections', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path_rf8 = os.path.join(current_results_folder, "rf_comparison_small_two_connections.png")
    plt.savefig(save_path_rf8, dpi=300, bbox_inches="tight")
    print(f"\nRandom Forest comparison plot saved as: {save_path_rf8}")
    plt.close()
else:
    print("\nWARNING: Only one class found. Cannot train decision tree.")
    if y_two_conn_train.iloc[0] == 1:
        print("At least one Small cluster ALWAYS has two connections.")
    else:
        print("Small clusters NEVER have two connections.")

# =============================================================================
# 9) DECISION TREE: High Pressure vs Low Pressure Pipeline Choice
# =============================================================================
print("\n" + "="*80)
print("DECISION TREE 9: High Pressure vs Low Pressure Pipeline Choice")
print("="*80)

def pipeline_pressure_choice_label(df):
    """
    Binary label for cases where pipelines exist:
    1 = High Pressure pipeline chosen
    0 = Low Pressure pipeline chosen

    Only considers cases where at least one pipeline exists.
    Returns NaN for cases without pipelines (will be filtered out).
    """
    # All possible pipeline connections (High P)
    highP_cols = [
        "Large_cluster1_to_Large_cluster2_hydrogenPipelineOnshore_highP",
        "Large_cluster1_to_Small_cluster1_hydrogenPipelineOnshore_highP",
        "Large_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_highP",
        "Large_cluster2_to_Small_cluster1_hydrogenPipelineOnshore_highP",
        "Large_cluster2_to_Small_cluster2_hydrogenPipelineOnshore_highP",
        "Small_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_highP"
    ]

    # All possible pipeline connections (Low P)
    lowP_cols = [
        "Large_cluster1_to_Large_cluster2_hydrogenPipelineOnshore_lowP",
        "Large_cluster1_to_Small_cluster1_hydrogenPipelineOnshore_lowP",
        "Large_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_lowP",
        "Large_cluster2_to_Small_cluster1_hydrogenPipelineOnshore_lowP",
        "Large_cluster2_to_Small_cluster2_hydrogenPipelineOnshore_lowP",
        "Small_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_lowP"
    ]

    # Sum of all high pressure pipelines
    highP_total = sum(df[col] for col in highP_cols if col in df.columns)
    # Sum of all low pressure pipelines
    lowP_total = sum(df[col] for col in lowP_cols if col in df.columns)

    # Mask for cases with at least one pipeline
    has_pipeline = (highP_total > 0.1) | (lowP_total > 0.1)

    # Label: 1 if predominantly high pressure, 0 if predominantly low pressure
    result = pd.Series(index=df.index, dtype=float)
    result[has_pipeline] = (highP_total[has_pipeline] > lowP_total[has_pipeline]).astype(int)
    result[~has_pipeline] = pd.NA

    return result

y_pipeline_pressure_train_full = pipeline_pressure_choice_label(data_train)
y_pipeline_pressure_test_full = pipeline_pressure_choice_label(data_test)

# Filter out cases without pipelines
train_with_pipeline = y_pipeline_pressure_train_full.notna()
test_with_pipeline = y_pipeline_pressure_test_full.notna()

X_train_pipeline = X_train[train_with_pipeline]
y_pipeline_pressure_train = y_pipeline_pressure_train_full[train_with_pipeline].astype(int)

X_test_pipeline = X_test[test_with_pipeline]
y_pipeline_pressure_test = y_pipeline_pressure_test_full[test_with_pipeline].astype(int)

print(f"\nFiltering to cases with pipelines:")
print(f"  Training: {len(X_train_pipeline)} out of {len(X_train)} runs have pipelines")
print(f"  Testing: {len(X_test_pipeline)} out of {len(X_test)} runs have pipelines")

print(f"\nTarget distribution (train): {y_pipeline_pressure_train.value_counts().to_dict()}")
print(f"Target distribution (test): {y_pipeline_pressure_test.value_counts().to_dict()}")
print(f"\nHigh Pressure chosen in {y_pipeline_pressure_train.mean()*100:.1f}% of training runs (with pipelines)")

# Detailed statistics
data_train_pipeline = data_train[train_with_pipeline]
highP_cols = [
    "Large_cluster1_to_Large_cluster2_hydrogenPipelineOnshore_highP",
    "Large_cluster1_to_Small_cluster1_hydrogenPipelineOnshore_highP",
    "Large_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_highP",
    "Large_cluster2_to_Small_cluster1_hydrogenPipelineOnshore_highP",
    "Large_cluster2_to_Small_cluster2_hydrogenPipelineOnshore_highP",
    "Small_cluster1_to_Small_cluster2_hydrogenPipelineOnshore_highP"
]
lowP_cols = [col.replace('_highP', '_lowP') for col in highP_cols]

print(f"\nPipeline statistics (cases with pipelines only):")
for i, (hp_col, lp_col) in enumerate(zip(highP_cols, lowP_cols)):
    connection_name = hp_col.replace('_hydrogenPipelineOnshore_highP', '')
    hp_count = (data_train_pipeline[hp_col] > 0.1).sum()
    lp_count = (data_train_pipeline[lp_col] > 0.1).sum()
    print(f"  {connection_name}:")
    print(f"    High P: {hp_count} cases, Low P: {lp_count} cases")

if len(y_pipeline_pressure_train.unique()) > 1 and len(X_train_pipeline) > 10:
    clf_pressure = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=0,
        class_weight="balanced"
    )
    clf_pressure.fit(X_train_pipeline, y_pipeline_pressure_train)
    pred_pressure = clf_pressure.predict(X_test_pipeline)

    print(f"\nAccuracy: {accuracy_score(y_pipeline_pressure_test, pred_pressure):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_pipeline_pressure_test, pred_pressure,
                               labels=[0, 1],
                               target_names=["Low Pressure", "High Pressure"],
                               zero_division=0))

    print("\nTop 10 Feature Importances:")
    feature_importance_pressure = sorted(zip(X_cols, clf_pressure.feature_importances_),
                                        key=lambda x: -x[1])
    for i, (feature, importance) in enumerate(feature_importance_pressure[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")

    tree_text_pressure = export_text(clf_pressure, feature_names=X_cols)
    print("\nDecision Tree Structure:")
    print(tree_text_pressure)

    plt.figure(figsize=(40, 20))
    plot_tree(clf_pressure, feature_names=X_cols,
             class_names=["Low Pressure", "High Pressure"],
             filled=True, rounded=True, fontsize=13)
    plt.title("Decision Tree - High Pressure vs Low Pressure Pipeline Choice", fontsize=16)
    save_path_9 = os.path.join(current_results_folder, "decision_tree_pipeline_pressure.png")
    plt.savefig(save_path_9, dpi=300, bbox_inches="tight")
    print(f"\nTree plot saved as: {save_path_9}")
    plt.close()

    # Random Forest
    print("\n" + "="*80)
    print("RANDOM FOREST 9: High Pressure vs Low Pressure Pipeline Choice")
    print("="*80)

    rf_pressure = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=3,
        min_samples_split=6,
        random_state=0,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_pressure.fit(X_train_pipeline, y_pipeline_pressure_train)
    pred_rf_pressure = rf_pressure.predict(X_test_pipeline)

    print(f"\nAccuracy: {accuracy_score(y_pipeline_pressure_test, pred_rf_pressure):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_pipeline_pressure_test, pred_rf_pressure,
                               labels=[0, 1],
                               target_names=["Low Pressure", "High Pressure"],
                               zero_division=0))

    print("\nTop 10 Feature Importances (Random Forest):")
    feature_importance_rf_pressure = sorted(zip(X_cols, rf_pressure.feature_importances_),
                                           key=lambda x: -x[1])
    for i, (feature, importance) in enumerate(feature_importance_rf_pressure[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    dt_imp_pressure = pd.Series(clf_pressure.feature_importances_, index=X_cols).sort_values(ascending=False)
    dt_imp_pressure[:10].plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Importance', fontsize=12)
    ax1.set_title('Decision Tree\nPipeline Pressure Choice', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    rf_imp_pressure = pd.Series(rf_pressure.feature_importances_, index=X_cols).sort_values(ascending=False)
    rf_imp_pressure[:10].plot(kind='barh', ax=ax2, color='forestgreen')
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title('Random Forest\nPipeline Pressure Choice', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path_rf9 = os.path.join(current_results_folder, "rf_comparison_pipeline_pressure.png")
    plt.savefig(save_path_rf9, dpi=300, bbox_inches="tight")
    print(f"\nRandom Forest comparison plot saved as: {save_path_rf9}")
    plt.close()
else:
    print("\nWARNING: Insufficient data or only one class found. Cannot train decision tree.")
    if len(X_train_pipeline) <= 10:
        print(f"Only {len(X_train_pipeline)} cases with pipelines in training set.")
    elif len(y_pipeline_pressure_train.unique()) == 1:
        if y_pipeline_pressure_train.iloc[0] == 1:
            print("High Pressure ALWAYS chosen when pipelines exist.")
        else:
            print("Low Pressure ALWAYS chosen when pipelines exist.")

    print("\n" + "="*80)
    print(f"DATASET '{dataset_name.upper()}' ANALYSIS COMPLETE")
    print("="*80)

# =============================================================================
# END OF MAIN LOOP
# =============================================================================

print("\n" + "="*80)
print("="*80)
print("ALL ANALYSES COMPLETE")
print("="*80)
print("="*80)

if ANALYZE_BY_ARCHETYPE and 'archetype' in data_preprocessed["cap"].columns:
    archetypes_list = sorted(data_preprocessed["cap"]['archetype'].unique())
    print("\nResults saved in:")
    print(f"  - Combined analysis: {results_folder}")
    for arch in archetypes_list:
        arch_folder = os.path.join(results_folder, f"archetype_{arch}")
        print(f"  - Archetype {arch}: {arch_folder}")
else:
    print(f"\nAll results saved in: {results_folder}")

