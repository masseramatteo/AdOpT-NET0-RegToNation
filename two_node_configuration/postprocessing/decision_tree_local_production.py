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
results_folder = r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\two_node_configuration\results\parallel_creation_test_20260106_112208"

dependent_vars_selection = ["Electrolyzer_small_installed"]
dependent_vars_size = ["Electrolyzer_small_size"]
dependent_vars_operation = ["Surplus_in_small_cluster"]

independent_vars = ["distance_between_nodes", "total_demand_TWh",
                    "demand_level_ratio", "import_availability_ratio",
                    "electricity_price_avg", "electricity_availability_small",
                    "hydrogen_import_price"]

# =============================================================================
# Load and preprocess data
# =============================================================================
print("Loading data from Excel files...")

# Load the two Excel files
parallel_summary_path = os.path.join(results_folder, "parallel_results_summary.xlsx")
extracted_results_path = os.path.join(results_folder, "extracted_results.xlsx")

df_summary = pd.read_excel(parallel_summary_path)
df_extracted = pd.read_excel(extracted_results_path)

print(f"Loaded {len(df_summary)} runs from parallel_results_summary.xlsx")
print(f"Loaded {len(df_extracted)} runs from extracted_results.xlsx")

# Merge the two dataframes on 'run_id' and 'run'
df_merged = pd.merge(df_summary, df_extracted, left_on='run_id', right_on='run', how='inner')
print(f"Merged data: {len(df_merged)} runs")

# =============================================================================
# Create dependent variables
# =============================================================================
print("\nCreating dependent variables...")

# 1) Electrolyzer_small_installed: binary (1 if size > 0)
df_merged['Electrolyzer_small_installed'] = (df_merged['Small_cluster_Electrolyzer_small'] > 0).astype(int)

# 2) Electrolyzer_small_size: the actual size
df_merged['Electrolyzer_small_size'] = df_merged['Small_cluster_Electrolyzer_small']

# 3) Surplus_in_small_cluster: se large cluster ha inflow da network, significa che small cluster esporta
# Surplus = hydrogen prodotto nel small - hydrogen usato nel small = network outflow
df_merged['Surplus_in_small_cluster'] = df_merged['Small_cluster_hydrogen_network_outflow_sum']

# 4) Pipeline_supply: binary (1 if electrolyzer installed AND exports to large cluster)
df_merged['Pipeline_supply'] = ((df_merged['Electrolyzer_small_installed'] == 1) &
                                 (df_merged['Surplus_in_small_cluster'] > 0)).astype(int)

# 5) Pipeline_type: classificazione tra highP e lowP pipeline
# Nomi esatti delle colonne delle pipeline
highp_col = 'hydrogenPipelineOnshore_highP'
lowp_col = 'hydrogenPipelineOnshore_lowP'

print(f"\nSearching for pipeline columns:")
print(f"  - Looking for high pressure: '{highp_col}'")
print(f"  - Looking for low pressure: '{lowp_col}'")

# Verifica presenza colonne
highp_exists = highp_col in df_merged.columns
lowp_exists = lowp_col in df_merged.columns

print(f"  - High pressure column found: {highp_exists}")
print(f"  - Low pressure column found: {lowp_exists}")

# Creiamo variabili per highP e lowP
if highp_exists:
    df_merged['highP_total'] = df_merged[highp_col]
    print(f"  - highP_total: mean = {df_merged['highP_total'].mean():.2f}, max = {df_merged['highP_total'].max():.2f}")
else:
    df_merged['highP_total'] = 0
    print("  WARNING: No highP column found")

if lowp_exists:
    df_merged['lowP_total'] = df_merged[lowp_col]
    print(f"  - lowP_total: mean = {df_merged['lowP_total'].mean():.2f}, max = {df_merged['lowP_total'].max():.2f}")
else:
    df_merged['lowP_total'] = 0
    print("  WARNING: No lowP column found")

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
print(f"  - Electrolyzer_small_installed: {df_merged['Electrolyzer_small_installed'].sum()} installed out of {len(df_merged)}")
print(f"  - Electrolyzer_small_size: mean = {df_merged['Electrolyzer_small_size'].mean():.2f} MW")
print(f"  - Surplus_in_small_cluster: mean = {df_merged['Surplus_in_small_cluster'].mean():.2f} TWh")
print(f"  - Pipeline_supply: {df_merged['Pipeline_supply'].sum()} cases out of {len(df_merged)}")
print(f"  - Pipeline_type distribution: {df_merged['Pipeline_type'].value_counts().to_dict()}")
print(f"  - HighP_vs_LowP (where pipeline exists): {(df_merged['HighP_vs_LowP'] == 1).sum()} highP, {(df_merged['HighP_vs_LowP'] == 0).sum()} lowP")

# =============================================================================
# Preprocess data
# =============================================================================
if do_preprocessing:
    print("\nPreprocessing data...")

    # Filter only successful runs
    df_filtered = df_merged[df_merged['status'] == 'SUCCESS'].copy()
    print(f"Filtered to {len(df_filtered)} successful runs")

    # Check for missing values in independent variables
    print("\nChecking missing values in independent variables:")
    for var in independent_vars:
        if var in df_filtered.columns:
            missing = df_filtered[var].isna().sum()
            print(f"  - {var}: {missing} missing values")
        else:
            print(f"  - {var}: COLUMN NOT FOUND!")

    # Fill missing values with 0 or median
    df_filtered[independent_vars] = df_filtered[independent_vars].fillna(0)

    # Store preprocessed data
    data_preprocessed = {"cap": df_filtered}
    print(f"\nPreprocessing complete. Final dataset: {len(data_preprocessed['cap'])} runs")
else:
    data_preprocessed = {"cap": df_merged}

# Split set
(data_train, data_test) = train_test_split(data_preprocessed["cap"],
                                                      test_size=0.1,
                                                      random_state=42)

# Decision tree for local hydrogen production in small cluster

def _select_existing_columns(df, cols: List[str]):
    """Helper function to select only columns that exist in the dataframe."""
    existing = [c for c in cols if c in df.columns]
    if not existing:
        raise ValueError("None of the requested columns are present in the dataframe.")
    return existing


# Prepare feature matrix
X_cols = _select_existing_columns(data_train, independent_vars)
X_train = data_train[X_cols].fillna(0)
X_test = data_test[X_cols].fillna(0)

print(f"\nUsing {len(X_cols)} features: {X_cols}")
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# =============================================================================
# 1) DECISION TREE: Electrolyzer Installation in Small Cluster
# =============================================================================
print("\n" + "="*80)
print("DECISION TREE 1: Electrolyzer Installation in Small Cluster")
print("="*80)

y_install_train = data_train["Electrolyzer_small_installed"]
y_install_test = data_test["Electrolyzer_small_installed"]

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
save_path_1 = os.path.join(results_folder, "decision_tree_electrolyzer_install.png")
plt.savefig(save_path_1, dpi=300, bbox_inches="tight")
print(f"\nTree plot saved as: {save_path_1}")
plt.close()

# =============================================================================
# 1b) RANDOM FOREST: Electrolyzer Installation in Small Cluster
# =============================================================================
print("\n" + "="*80)
print("RANDOM FOREST 1: Electrolyzer Installation in Small Cluster")
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
ax1.set_title('Decision Tree - Top 10 Features\nElectrolyzer Installation', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Random Forest importances
rf_importances = pd.Series(rf_install.feature_importances_, index=X_cols).sort_values(ascending=False)
rf_importances[:10].plot(kind='barh', ax=ax2, color='forestgreen')
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title('Random Forest - Top 10 Features\nElectrolyzer Installation', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
save_path_rf1 = os.path.join(results_folder, "rf_comparison_electrolyzer_install.png")
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
    Create binary label: 1 if electrolyzer is installed AND surplus > 0
    (meaning small cluster supplies hydrogen to large cluster via pipeline)
    """
    required = _select_existing_columns(df, ["Surplus_in_small_cluster", "Electrolyzer_small_installed"])
    if "Surplus_in_small_cluster" in df.columns and "Electrolyzer_small_installed" in df.columns:
        return ((df["Surplus_in_small_cluster"] > 0) & (df["Electrolyzer_small_installed"] == 1)).astype(int)
    else:
        raise ValueError("Required columns for pipeline supply label are missing")

y_pipe_train = pipeline_supply_label(data_train)
y_pipe_test = pipeline_supply_label(data_test)

print(f"\nTarget distribution (train): {y_pipe_train.value_counts().to_dict()}")
print(f"Target distribution (test): {y_pipe_test.value_counts().to_dict()}")

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
print(classification_report(y_pipe_test, pred_pipe))

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
         class_names=["No Pipeline Supply", "Pipeline Supply"],
         filled=True, rounded=True, fontsize=13)
plt.title("Decision Tree - Pipeline Supply from Small to Large Cluster", fontsize=16)
save_path_2 = os.path.join(results_folder, "decision_tree_pipeline_supply.png")
plt.savefig(save_path_2, dpi=300, bbox_inches="tight")
print(f"\nTree plot saved as: {save_path_2}")
print("\nTree plot saved as: decision_tree_pipeline_supply.png")
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
print(classification_report(y_pipe_test, pred_rf_pipe))

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
save_path_rf2 = os.path.join(results_folder, "rf_comparison_pipeline_supply.png")
plt.savefig(save_path_rf2, dpi=300, bbox_inches="tight")
print(f"\nRandom Forest comparison plot saved as: {save_path_rf2}")
plt.close()

# =============================================================================
# 3) DECISION TREE: Pipeline Type Choice (High Pressure vs Low Pressure)
# =============================================================================
print("\n" + "="*80)
print("DECISION TREE 3: Pipeline Type Choice (High Pressure vs Low Pressure)")
print("="*80)

# Filter only runs where at least one pipeline is installed
data_train_pipeline = data_train[data_train['HighP_vs_LowP'] >= 0].copy()
data_test_pipeline = data_test[data_test['HighP_vs_LowP'] >= 0].copy()

print(f"\nFiltered to runs with pipeline installed:")
print(f"  Training set: {len(data_train_pipeline)} runs")
print(f"  Test set: {len(data_test_pipeline)} runs")

if len(data_train_pipeline) > 10 and len(data_test_pipeline) > 0:
    # Prepare features
    X_train_pipe_type = data_train_pipeline[X_cols].fillna(0)
    X_test_pipe_type = data_test_pipeline[X_cols].fillna(0)

    y_pipe_type_train = data_train_pipeline['HighP_vs_LowP']
    y_pipe_type_test = data_test_pipeline['HighP_vs_LowP']

    print(f"\nTarget distribution (train): {y_pipe_type_train.value_counts().to_dict()}")
    print(f"Target distribution (test): {y_pipe_type_test.value_counts().to_dict()}")

    # Check if we have both classes
    if len(y_pipe_type_train.unique()) > 1:
        # Train decision tree classifier
        clf_pipe_type = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=0,
            class_weight="balanced"
        )
        clf_pipe_type.fit(X_train_pipe_type, y_pipe_type_train)
        pred_pipe_type = clf_pipe_type.predict(X_test_pipe_type)

        # Metrics
        print(f"\nAccuracy: {accuracy_score(y_pipe_type_test, pred_pipe_type):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_pipe_type_test, pred_pipe_type,
                                   target_names=["Low Pressure", "High Pressure"]))

        # Feature importances
        print("\nTop 10 Feature Importances:")
        feature_importance_pipe_type = sorted(zip(X_cols, clf_pipe_type.feature_importances_),
                                             key=lambda x: -x[1])
        for i, (feature, importance) in enumerate(feature_importance_pipe_type[:10], 1):
            print(f"{i}. {feature}: {importance:.4f}")

        # Export tree as text
        tree_text_pipe_type = export_text(clf_pipe_type, feature_names=X_cols)
        print("\nDecision Tree Structure:")
        print(tree_text_pipe_type)

        # Plot and save tree
        plt.figure(figsize=(40, 20))
        plot_tree(clf_pipe_type, feature_names=X_cols,
                 class_names=["Low Pressure", "High Pressure"],
                 filled=True, rounded=True, fontsize=13)
        plt.title("Decision Tree - Pipeline Type Choice (High Pressure vs Low Pressure)", fontsize=16)
        save_path_3 = os.path.join(results_folder, "decision_tree_pipeline_type.png")
        plt.savefig(save_path_3, dpi=300, bbox_inches="tight")
        print(f"\nTree plot saved as: {save_path_3}")
        plt.close()

        # =============================================================================
        # 3b) RANDOM FOREST: Pipeline Type Choice (High Pressure vs Low Pressure)
        # =============================================================================
        print("\n" + "="*80)
        print("RANDOM FOREST 3: Pipeline Type Choice (High Pressure vs Low Pressure)")
        print("="*80)

        # Train Random Forest classifier
        rf_pipe_type = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=3,
            min_samples_split=6,
            random_state=0,
            class_weight="balanced",
            n_jobs=-1
        )
        rf_pipe_type.fit(X_train_pipe_type, y_pipe_type_train)
        pred_rf_pipe_type = rf_pipe_type.predict(X_test_pipe_type)

        # Metrics
        print(f"\nAccuracy: {accuracy_score(y_pipe_type_test, pred_rf_pipe_type):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_pipe_type_test, pred_rf_pipe_type,
                                   target_names=["Low Pressure", "High Pressure"]))

        # Feature importances
        print("\nTop 10 Feature Importances (Random Forest):")
        feature_importance_rf_pipe_type = sorted(zip(X_cols, rf_pipe_type.feature_importances_),
                                                key=lambda x: -x[1])
        for i, (feature, importance) in enumerate(feature_importance_rf_pipe_type[:10], 1):
            print(f"{i}. {feature}: {importance:.4f}")

        # Compare with Decision Tree
        print("\nComparison Decision Tree vs Random Forest:")
        print(f"  Decision Tree Accuracy: {accuracy_score(y_pipe_type_test, pred_pipe_type):.4f}")
        print(f"  Random Forest Accuracy: {accuracy_score(y_pipe_type_test, pred_rf_pipe_type):.4f}")
        print(f"  Improvement: {(accuracy_score(y_pipe_type_test, pred_rf_pipe_type) - accuracy_score(y_pipe_type_test, pred_pipe_type)):.4f}")

        # Create comparison plot: Feature Importances DT vs RF
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Decision Tree importances
        dt_importances_type = pd.Series(clf_pipe_type.feature_importances_, index=X_cols).sort_values(ascending=False)
        dt_importances_type[:10].plot(kind='barh', ax=ax1, color='steelblue')
        ax1.set_xlabel('Importance', fontsize=12)
        ax1.set_title('Decision Tree - Top 10 Features\nPipeline Type (High vs Low Pressure)', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        # Random Forest importances
        rf_importances_type = pd.Series(rf_pipe_type.feature_importances_, index=X_cols).sort_values(ascending=False)
        rf_importances_type[:10].plot(kind='barh', ax=ax2, color='forestgreen')
        ax2.set_xlabel('Importance', fontsize=12)
        ax2.set_title('Random Forest - Top 10 Features\nPipeline Type (High vs Low Pressure)', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        save_path_rf3 = os.path.join(results_folder, "rf_comparison_pipeline_type.png")
        plt.savefig(save_path_rf3, dpi=300, bbox_inches="tight")
        print(f"\nRandom Forest comparison plot saved as: {save_path_rf3}")
        plt.close()

    else:
        print("\nWARNING: Only one class found in training data. Cannot train classifier.")
        print(f"All runs use the same pipeline type: {'High Pressure' if y_pipe_type_train.iloc[0] == 1 else 'Low Pressure'}")
else:
    print("\nWARNING: Insufficient data for pipeline type classification.")
    print("Not enough runs with pipeline installed to train the model.")

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
    if "Small_cluster_hydrogen_network_outflow_sum" in df.columns and "Large_cluster_hydrogen_network_outflow_sum" in df.columns:
        return (df["Small_cluster_hydrogen_network_outflow_sum"] > df["Large_cluster_hydrogen_network_outflow_sum"]).astype(int)
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
                               target_names=["Large Outflow > Small", "Small Outflow > Large"]))

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
    save_path_4 = os.path.join(results_folder, "decision_tree_outflow_comparison.png")
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
                               target_names=["Large Outflow > Small", "Small Outflow > Large"]))

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
    save_path_rf4 = os.path.join(results_folder, "rf_comparison_outflow.png")
    plt.savefig(save_path_rf4, dpi=300, bbox_inches="tight")
    print(f"\nRandom Forest comparison plot saved as: {save_path_rf4}")
    plt.close()

else:
    print("\nWARNING: Only one class found in data. Cannot train classifier.")
    if y_outflow_train.iloc[0] == 1:
        print("Small cluster ALWAYS has greater outflow than Large cluster.")
    else:
        print("Large cluster ALWAYS has greater outflow than Small cluster.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

