"""
SHAP + Partial Dependence analysis for electrolyzer installation decision.
Run after extract_design_sizes.py has produced extracted_results.xlsx.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_PATH = r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\AdOpT-NET0-RegToNation\four_node_configuration\results\checking_on_installation_small\1200_sim_to_try_less_solar\extracted_results.xlsx"

TARGET_COL    = "Small_cluster1_Electrolyzer_small"   # change to Small_cluster2 if needed
TARGET_LABEL  = "Small_cluster1"
INSTALL_THRESHOLD = 1.0   # MW

FEATURES = [
    "total_demand_TWh",
    "demand_level_ratio",
    "unbalance_ratio",
    "electricity_availability_small",
    "electricity_availability_large",
    "import_availability_ratio",
    "electricity_price_avg",
    "electricity_standard_dev",
    "hydrogen_import_price",
    "solar_cf_mean",
    "distance_from_large_cluster",
    "distance_Small_cluster1_to_Small_cluster2_km",
]

# Features to show PDP for (top candidates)
PDP_FEATURES = [
    "total_demand_TWh",
    "hydrogen_import_price",
    "electricity_price_avg",
    "distance_from_large_cluster",
    "solar_cf_mean",
    "electricity_availability_large",
]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_excel(RESULTS_PATH)
print(f"  {len(df)} runs loaded")

# Build target
df["y"] = (df[TARGET_COL].fillna(0) > INSTALL_THRESHOLD).astype(int)
print(f"  Install rate: {df['y'].mean():.1%}")

# Compute distance_from_large_cluster if missing
if "distance_from_large_cluster" not in df.columns:
    d_cols = [c for c in df.columns if "distance" in c and "Large" in c and c.endswith("_km")]
    if d_cols:
        df["distance_from_large_cluster"] = df[d_cols].min(axis=1)
        print(f"  Derived distance_from_large_cluster from: {d_cols}")

# Select features present in data
feature_cols = [f for f in FEATURES if f in df.columns]
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    print(f"  WARNING: features not found in data (skipped): {missing}")

X = df[feature_cols].fillna(0)
y = df["y"]

output_dir = Path(RESULTS_PATH).parent
print(f"  Output dir: {output_dir}\n")

# ── Train models ──────────────────────────────────────────────────────────────
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=3,
    min_samples_split=6, random_state=0, class_weight="balanced"
)
rf.fit(X, y)
cv_rf = cross_val_score(rf, X, y, cv=10, scoring="accuracy")
print(f"  RF CV accuracy: {cv_rf.mean():.4f} +/- {cv_rf.std():.4f}")

print("Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=0
)
gb.fit(X, y)
cv_gb = cross_val_score(gb, X, y, cv=10, scoring="accuracy")
print(f"  GB CV accuracy: {cv_gb.mean():.4f} +/- {cv_gb.std():.4f}\n")

# Pick best model for SHAP
best_model = rf if cv_rf.mean() >= cv_gb.mean() else gb
best_name  = "RandomForest" if best_model is rf else "GradientBoosting"
print(f"Using {best_name} for SHAP analysis\n")

# ── SHAP ──────────────────────────────────────────────────────────────────────
print("Computing SHAP values (TreeExplainer)...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X)

# For binary classification shap returns shape (n, features, 2) for RF
# We want class=1 (install)
if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
    sv = shap_values[:, :, 1]   # class 1
else:
    sv = shap_values

print("  SHAP values computed\n")

# 1) Beeswarm summary plot
print("Plotting SHAP beeswarm summary...")
fig, ax = plt.subplots(figsize=(9, 6))
shap.plots.beeswarm(sv, max_display=12, show=False)
plt.title(f"SHAP values — Electrolyzer install ({TARGET_LABEL})\n{best_name}", fontsize=12)
plt.tight_layout()
out = output_dir / f"shap_beeswarm_{TARGET_LABEL}.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# 2) Bar plot (mean |SHAP|)
print("Plotting SHAP bar (mean |SHAP|)...")
fig, ax = plt.subplots(figsize=(8, 5))
shap.plots.bar(sv, max_display=12, show=False)
plt.title(f"Mean |SHAP| — Electrolyzer install ({TARGET_LABEL})\n{best_name}", fontsize=12)
plt.tight_layout()
out = output_dir / f"shap_bar_{TARGET_LABEL}.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# 3) SHAP scatter for top features (shows direction)
print("Plotting SHAP scatter for key features...")
top_features = sorted(
    zip(feature_cols, np.abs(sv.values).mean(axis=0)),
    key=lambda x: -x[1]
)[:6]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, (feat, _) in zip(axes.flatten(), top_features):
    feat_idx = feature_cols.index(feat)
    ax.scatter(X[feat], sv.values[:, feat_idx],
               c=sv.values[:, feat_idx], cmap="coolwarm",
               alpha=0.4, s=10)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(feat, fontsize=8)
    ax.set_ylabel("SHAP value", fontsize=8)
    ax.set_title(feat, fontsize=9)

fig.suptitle(f"SHAP scatter — top 6 features ({TARGET_LABEL})", fontsize=11)
plt.tight_layout()
out = output_dir / f"shap_scatter_{TARGET_LABEL}.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── Partial Dependence Plots ───────────────────────────────────────────────────
pdp_cols = [f for f in PDP_FEATURES if f in feature_cols]
print(f"\nPlotting PDPs for: {pdp_cols}")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
PartialDependenceDisplay.from_estimator(
    best_model, X, pdp_cols,
    kind="average",        # average prediction (ICE = individual conditional)
    target=1,              # class 1 = install
    ax=axes.flatten()[:len(pdp_cols)],
    line_kw={"color": "steelblue", "lw": 2}
)
fig.suptitle(f"Partial Dependence Plots — install probability ({TARGET_LABEL})\n{best_name}", fontsize=11)
plt.tight_layout()
out = output_dir / f"pdp_{TARGET_LABEL}.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"SHAP mean |value| ranking ({best_name})")
print("="*60)
mean_abs = np.abs(sv.values).mean(axis=0)
for feat, val in sorted(zip(feature_cols, mean_abs), key=lambda x: -x[1]):
    direction = "+" if sv.values[:, feature_cols.index(feat)].mean() > 0 else "-"
    print(f"  {direction}  {feat:<45} {val:.4f}")

print("\nDone.")