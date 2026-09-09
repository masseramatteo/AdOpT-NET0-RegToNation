"""
Shared configuration for the DT_Supply_Chain infrastructure-typology pipeline.

Reproduces the cluster-then-explain methodology of:
  - Baader et al. (2023) "Streamlining Energy Transition Scenarios to Key Policy Decisions"
  - Wiest et al. (2026) "Low-regret strategies for energy systems planning"
applied to the TYPE of final hydrogen infrastructure emerging from the 4-node MILP runs.

Decisions locked with the user (2026-08-04):
  - Clustering: standardize ALL features (Baader-faithful), plain k-Means.
  - Scope: combined across all 4 archetypes first, then compare.
  - Storage siting: exclude the existing (fixed) H2 Cavern; count newly-built storage only.
  - Outputs of interest: the 9 features defined in build_features.py, as proposed.
"""

import os

# Number of clusters k. Override per run with the DT_K environment variable, e.g.
#   $env:DT_K=9  (PowerShell)  so outputs land in DT_supplychain/k9/.
N_CLUSTERS = int(os.environ.get("DT_K", 4))

# --- Paths -------------------------------------------------------------------
RESULTS_FOLDER = (
    r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M"
    r"\AdOpT-NET0-RegToNation\four_node_configuration\results"
    r"\3000_simulations_24_may_2026\Snellius_3000_simulations"
)
EXTRACTED_RESULTS = os.path.join(RESULTS_FOLDER, "extracted_results.xlsx")

# Output directory for this pipeline (next to the scripts)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(OUTPUT_DIR, "infrastructure_features.csv")
# labels file is per-k so runs with different k don't clobber each other
CLUSTER_LABELS_CSV = os.path.join(OUTPUT_DIR, f"cluster_labels_k{N_CLUSTERS}.csv")
# Figures saved to the shared results folder, in a per-k subfolder
BASE_FIG_DIR = os.path.join(RESULTS_FOLDER, "DT_supplychain")
FIG_DIR = os.path.join(BASE_FIG_DIR, f"k{N_CLUSTERS}")

# --- Numerics ----------------------------------------------------------------
BUILD_EPS = 1.0           # a pipeline/tech counts as "built"/installed if size > 1 MW (paper convention)
RANDOM_STATE = 42

# --- Tree settings -----------------------------------------------------------
# N_CLUSTERS is set at the top (env-overridable). k=4 chosen for driver-tree
# predictability: accuracy sweep (experiments.py) shows the sampled inputs predict a
# 4-way typology at ~0.68 CV (baseline 0.38, RF ceiling 0.78), vs only ~0.47 for the
# 6-way typology. Trade-off: the standalone "Import-reliant" type dissolves into a
# within-cluster import gradient.
# Explaining tree leaves: = N_CLUSTERS is Baader-faithful; a little higher buys purity.
EXPLAIN_MAX_LEAF_NODES = N_CLUSTERS + 2
# Driver tree — control readability by LEAF BUDGET, not depth. A deep-but-narrow
# tree (max_leaf_nodes) gives far higher accuracy per leaf than a shallow wide one:
# at 16 leaves CV acc ~0.665 (k=4) vs ~0.593 for max_depth=4 at the same size.
DRIVER_MAX_DEPTH = None
DRIVER_MAX_LEAF_NODES = 16
DRIVER_MIN_SAMPLES_LEAF = 20
DRIVER_MIN_SAMPLES_SPLIT = 20
# Auto-tune the leaf budget per k: sweep DRIVER_LEAF_GRID, pick the SMALLEST leaf
# count whose CV accuracy is within DRIVER_TRADEOFF_TOL of the best in the grid
# (best readability/accuracy trade-off). Overrides DRIVER_MAX_LEAF_NODES per run.
DRIVER_AUTOTUNE = True
DRIVER_LEAF_GRID = [8, 10, 12, 14, 16, 18, 20]
DRIVER_TRADEOFF_TOL = 0.01

# RandomForest for robust (averaged) feature importances — separate figure only.
RF_PARAMS = dict(n_estimators=100, max_depth=10, min_samples_leaf=3,
                 min_samples_split=6, random_state=0, class_weight="balanced", n_jobs=-1)

# --- Cluster archetype names (keyed by k-Means / cluster_final id at k=4) -----
CLUSTER_NAMES = {
    0: "Centralized backbone",
    1: "Satellite / radial",
    2: "Local self-sufficient",
    3: "Small-small meshed",
}

# --- Feature groups ----------------------------------------------------------
# Continuous outputs of interest (unit-cancelling shares/intensities)
CONTINUOUS_OOI = [
    "local_self_sufficiency",   # min(small_prod / small_demand, 1): local demand covered by own production
    "import_intensity",         # total imported H2 / total demand
    "transport_intensity",      # total H2 moved over network / total demand (volume of transport)
    "storage_large_share",      # newly-built storage in large clusters / total newly-built storage
]
# Topology outputs of interest (binary / count)
TOPOLOGY_OOI = [
    "SS_pipe",
    "LL_pipe",
    "meshed",
    "n_pipelines",
    "highP_share",
]
OUTPUTS_OF_INTEREST = CONTINUOUS_OOI + TOPOLOGY_OOI

# Sampled input parameters used for the driver tree (Step 4)
DRIVER_INPUTS = [
    "total_demand_TWh",
    "demand_level_ratio",
    "unbalance_ratio",
    "electricity_availability_small",
    "electricity_availability_large",
    "import_availability_ratio",
    "electricity_price_avg",
    "electricity_standard_dev",
    "hydrogen_import_price",
    "solar_cf_realized_mean",
    "distance_Large_cluster1_to_Large_cluster2_km",
    "distance_Small_cluster1_to_Small_cluster2_km",
    "distance_from_large_cluster",   # derived in build_features
]

# --- UU brand colours (Joule style) -----------------------------------------
UU_NAVY = "#161D41"
UU_TEAL = "#63A593"
UU_YELLOW = "#FFCD00"
UU_RED = "#C00935"
UU_AMBER = "#b8900a"
# Ordered palette for up to 6 clusters
CLUSTER_PALETTE = [UU_NAVY, UU_TEAL, UU_YELLOW, UU_RED, UU_AMBER, "#6c757d"]


def name_cluster(p):
    """Rule-based descriptive name from a cluster's mean output-of-interest profile
    (a dict / pandas Series). Works for any k. Order matters — most distinctive
    topology features first."""
    if p["SS_pipe"] > 0.5:
        return "Small-small meshed"
    if p["meshed"] > 0.5:
        return "Dual-supply meshed"
    if p["import_intensity"] > 0.30:
        return "Import-reliant"
    if p["LL_pipe"] > 0.5:
        return ("Local self-sufficient" if p["local_self_sufficiency"] > 0.60
                else "Centralized backbone")
    if p["n_pipelines"] < 2.2:
        return "Minimal network"
    return "Satellite / radial"


def apply_joule_style():
    """Apply a clean matplotlib style consistent with Joule figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": UU_NAVY,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })