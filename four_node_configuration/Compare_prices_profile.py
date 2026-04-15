"""
Compare FROM_SERIES vs FROM_PAPER electricity price profiles.

Run from the four_node_configuration folder:
    python compare_price_profiles.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from price_profile_generation import (
    load_electricity_price_profile,
    fit_electricty_price_trends,
    generate_electricity_price_profile,
)

# ── config ───────────────────────────────────────────────────────────────────
BASE_PATH  = Path(__file__).parent
DATA_DIR   = BASE_PATH / "data" / "european_wholesale_electricity_price_data_hourly"
COUNTRY    = "Netherlands"
SHAPE_YEAR = 2019   # year used to fit the Keles shape and the FROM_SERIES shape
GEN_YEAR   = 2024   # calendar year for the generated profile — must exist in CSV

# If True  → target mean and std are taken from SHAPE_YEAR historical data
#            (useful to see the pure shape without any level change)
# If False → target mean and std are set manually below
USE_SHAPE_YEAR_STATS = False

TARGET_MEAN = 100.0   # used only if USE_SHAPE_YEAR_STATS = False
TARGET_STD  = 30.0    # used only if USE_SHAPE_YEAR_STATS = False


# ── load data ─────────────────────────────────────────────────────────────────
p_hist     = load_electricity_price_profile(COUNTRY, data_dir=str(DATA_DIR))
p_shape    = p_hist[p_hist.index.year == SHAPE_YEAR]["p"].to_numpy()[:8760]
p_hist_gen = p_hist[p_hist.index.year == GEN_YEAR]["p"].to_numpy()[:8760]

# resolve target mean and std
if USE_SHAPE_YEAR_STATS:
    target_mean = float(np.mean(p_shape))
    target_std  = float(np.std(p_shape))
    stats_label = f"same as {SHAPE_YEAR} historical"
else:
    target_mean = TARGET_MEAN
    target_std  = TARGET_STD
    stats_label = "custom"

print(f"SHAPE_YEAR  {SHAPE_YEAR}: mean={np.mean(p_shape):.2f}, std={np.std(p_shape):.2f}")
print(f"Historical  {GEN_YEAR}: mean={np.mean(p_hist_gen):.2f}, std={np.std(p_hist_gen):.2f}")
print(f"Target ({stats_label}): mean={target_mean:.2f}, std={target_std:.2f}")

# ── FROM_SERIES ───────────────────────────────────────────────────────────────
series_from_series = (
    (p_shape - np.mean(p_shape))
    * (target_std / np.std(p_shape))
    + target_mean
)
print(f"FROM_SERIES: mean={np.mean(series_from_series):.2f}, std={np.std(series_from_series):.2f}")

# ── FROM_PAPER ────────────────────────────────────────────────────────────────
_, fit_params = fit_electricty_price_trends(p_hist, SHAPE_YEAR)

scaling_factors = {
    "trend": 1.0,
    "weekly_factor": 1.0,
    "hourly_factor": 1.0,
    "overall_factor": 1.0,
}
p_gen = generate_electricity_price_profile(fit_params, target_mean, scaling_factors, GEN_YEAR)
series_from_paper = p_gen["p"].to_numpy()[:8760]
series_from_paper = (
    (series_from_paper - np.mean(series_from_paper))
    * (target_std / np.std(series_from_paper))
    + target_mean
)
print(f"FROM_PAPER:  mean={np.mean(series_from_paper):.2f}, std={np.std(series_from_paper):.2f}")

# ── datetime index ────────────────────────────────────────────────────────────
date_rng       = pd.date_range(start=f"{GEN_YEAR}-01-01",   periods=8760, freq="1h")
date_rng_shape = pd.date_range(start=f"{SHAPE_YEAR}-01-01", periods=len(p_shape), freq="1h")
date_rng_gen   = pd.date_range(start=f"{GEN_YEAR}-01-01",   periods=len(p_hist_gen), freq="1h")

title_stats = f"mean={target_mean:.1f}, std={target_std:.1f} €/MWh  ({stats_label})"

# ════════════════════════════════════════════════════════════════════════════
# PLOT 1: full year — four stacked panels
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=False)
fig.patch.set_facecolor("white")
fig.suptitle(
    f"Electricity price profiles — {COUNTRY}  |  Shape year: {SHAPE_YEAR}\n{title_stats}",
    fontsize=12, fontweight="bold"
)

panels = [
    (p_hist_gen,         date_rng_gen,   f"Historical {GEN_YEAR}",                         "#555555"),
    (p_shape,            date_rng_shape, f"Historical {SHAPE_YEAR} (shape source)",         "#888888"),
    (series_from_series, date_rng,       f"FROM_SERIES ({SHAPE_YEAR} → target)",            "#1a6faf"),
    (series_from_paper,  date_rng,       f"FROM_PAPER — Keles ({SHAPE_YEAR} fit → target)", "#c0392b"),
]

for ax, (series, drng, label, color) in zip(axes, panels):
    ax.plot(drng, series, color=color, linewidth=0.4, alpha=0.85)
    ax.axhline(np.mean(series), color="black", linewidth=1.0, linestyle="--",
               label=f"Mean = {np.mean(series):.1f} €/MWh")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_title(f"{label}  |  std = {np.std(series):.1f} €/MWh",
                 fontsize=9, fontweight="bold", pad=4)
    ax.set_ylabel("Price (EUR/MWh)", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Month", fontsize=9)
plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# PLOT 2: seasonal weekly zoom — FROM_SERIES vs FROM_PAPER
# ════════════════════════════════════════════════════════════════════════════
seasons = {
    "Winter (Jan week 2)": (7  * 24, 7  * 24 + 168),
    "Spring (Apr week 2)": (98 * 24, 98 * 24 + 168),
    "Summer (Jul week 2)": (190* 24, 190* 24 + 168),
    "Autumn (Oct week 2)": (281* 24, 281* 24 + 168),
}

fig, axes = plt.subplots(len(seasons), 1, figsize=(14, 12), sharex=False)
fig.patch.set_facecolor("white")
fig.suptitle(
    f"Seasonal weekly zoom — FROM_SERIES vs FROM_PAPER  |  {COUNTRY}\n{title_stats}",
    fontsize=12, fontweight="bold"
)

for ax, (season_name, (i0, i1)) in zip(axes, seasons.items()):
    t = date_rng[i0:i1]
    ax.plot(t, series_from_series[i0:i1], color="#1a6faf", linewidth=1.2,
            alpha=0.9, label=f"FROM_SERIES ({SHAPE_YEAR})", zorder=2)
    ax.plot(t, series_from_paper[i0:i1],  color="#c0392b", linewidth=1.2,
            alpha=0.9, linestyle="--", label=f"FROM_PAPER — Keles ({SHAPE_YEAR})", zorder=1)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_title(season_name, fontsize=10, fontweight="bold", pad=4)
    ax.set_ylabel("Price (EUR/MWh)", fontsize=8)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# PLOT 3: price duration curves
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")

for series, label, color, ls in [
    (p_hist_gen,         f"Historical {GEN_YEAR}",       "#555555", "-"),
    (series_from_series, f"FROM_SERIES ({SHAPE_YEAR})",  "#1a6faf", "-"),
    (series_from_paper,  f"FROM_PAPER  ({SHAPE_YEAR})",  "#c0392b", "--"),
]:
    sorted_s = np.sort(series)[::-1]
    ax.plot(np.arange(1, len(sorted_s) + 1), sorted_s,
            color=color, linewidth=1.2, linestyle=ls, label=label)

ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.set_xlabel("Hours (sorted)", fontsize=10)
ax.set_ylabel("Price (EUR/MWh)", fontsize=10)
ax.set_title(
    f"Price duration curves — {COUNTRY}  |  {title_stats}",
    fontsize=11, fontweight="bold"
)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()