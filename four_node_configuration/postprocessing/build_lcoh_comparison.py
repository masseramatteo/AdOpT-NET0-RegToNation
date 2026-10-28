"""
Build an Excel comparison between the first-best 3000-run batch and a near-optimal
(re-optimisation) batch, adding the levelised cost of hydrogen (LCOH).

LCOH is defined as total system cost (objective value = NPV) divided by the total
hydrogen demand of the system. The demand is identical between the first-best run
and its re-optimised counterpart (same input data), so it is taken once from the
first-best batch.

Outputs <REOPT_DIR>/reopt_comparison_with_LCOH.xlsx
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

BASE = r"X:/3000_simulations_24_may_2026"
ORIG_DIR = os.path.join(BASE, "Snellius_3000_simulations")
REOPT_DIR = os.path.join(
    BASE, "reopt_comparison_Snellius_3000_simulations_small_electrolyzer_only"
)

# Existing comparison file produced by near_optimal_analysis.py (merged over tasks).
REOPT_COMPARISON_FILE = "reopt_comparison.xlsx"
OUTPUT_FILE = "reopt_comparison_with_LCOH.xlsx"

# Lower heating value of hydrogen, used to convert EUR/MWh into EUR/kg.
LHV_H2_MWH_PER_KG = 33.33 / 1000


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def load_first_best_summary(orig_dir: str) -> pd.DataFrame:
    """Return run_id -> objective value and total hydrogen demand for the first-best batch.

    The batch summary CSV is used when available because it is a single file read.
    Runs that are missing from it are filled in from the per-run summary JSON.
    """
    csv_path = os.path.join(orig_dir, "parallel_results_summary.csv")
    df = pd.read_csv(csv_path, sep=";")
    df = df[df["status"] == "SUCCESS"]

    return df[["run_id", "objective_value", "para_total_demand"]].rename(
        columns={
            "objective_value": "cost_first_best",
            "para_total_demand": "total_demand_MWh",
        }
    )


def read_run_summary(run_folder: str) -> dict | None:
    """Read the optimisation summary JSON nested under a run folder's userData/<stamp>/."""
    matches = glob.glob(
        os.path.join(run_folder, "userData", "*", "optimization_results_summary.json")
    )
    if not matches:
        return None
    with open(matches[0]) as fh:
        return json.load(fh)


def fill_missing_from_disk(
    df: pd.DataFrame, orig_dir: str, reopt_dir: str
) -> pd.DataFrame:
    """Fill cost/demand values that the batch summary CSV did not cover."""
    missing = df[df["total_demand_MWh"].isna() | df["cost_first_best"].isna()]
    for idx, row in missing.iterrows():
        summary = read_run_summary(os.path.join(orig_dir, row["run_id"]))
        if summary is None:
            # Demand is a pure input parameter, so the re-optimised run carries the
            # same value and can be used as a fallback.
            summary = read_run_summary(os.path.join(reopt_dir, row["reopt_folder"]))
        if summary is None:
            continue
        if pd.isna(df.at[idx, "total_demand_MWh"]):
            df.at[idx, "total_demand_MWh"] = summary["para_total_demand"]
        if pd.isna(df.at[idx, "cost_first_best"]):
            df.at[idx, "cost_first_best"] = summary["var_npv"]
    return df


def map_reopt_folders(reopt_dir: str) -> pd.DataFrame:
    """Map each run_id to the folder name of its re-optimised run."""
    folders = [
        d
        for d in os.listdir(reopt_dir)
        if d.startswith("parallel_run_") and os.path.isdir(os.path.join(reopt_dir, d))
    ]
    return pd.DataFrame(
        {"run_id": [d.split("__")[0] for d in folders], "reopt_folder": folders}
    )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    comparison = pd.read_excel(os.path.join(REOPT_DIR, REOPT_COMPARISON_FILE))
    print(f"Re-optimisation comparison rows: {len(comparison)}")

    first_best = load_first_best_summary(ORIG_DIR)
    print(f"First-best runs in batch summary: {len(first_best)}")

    df = comparison.merge(map_reopt_folders(REOPT_DIR), on="run_id", how="left")
    df = df.merge(first_best, on="run_id", how="left")
    df = fill_missing_from_disk(df, ORIG_DIR, REOPT_DIR)

    # The comparison file already stores both objective values; prefer them and keep the
    # batch-summary value only as a cross-check.
    df["cost_first_best"] = df["npv_first_best"]
    df["cost_reopt"] = df["npv_reopt"]

    df["delta_cost"] = df["cost_reopt"] - df["cost_first_best"]
    df["delta_cost_pct"] = 100 * df["delta_cost"] / df["cost_first_best"]

    df["lcoh_first_best_eur_per_MWh"] = df["cost_first_best"] / df["total_demand_MWh"]
    df["lcoh_reopt_eur_per_MWh"] = df["cost_reopt"] / df["total_demand_MWh"]
    df["delta_lcoh_eur_per_MWh"] = (
        df["lcoh_reopt_eur_per_MWh"] - df["lcoh_first_best_eur_per_MWh"]
    )
    df["delta_lcoh_pct"] = (
        100 * df["delta_lcoh_eur_per_MWh"] / df["lcoh_first_best_eur_per_MWh"]
    )

    df["lcoh_first_best_eur_per_kg"] = (
        df["lcoh_first_best_eur_per_MWh"] * LHV_H2_MWH_PER_KG
    )
    df["lcoh_reopt_eur_per_kg"] = df["lcoh_reopt_eur_per_MWh"] * LHV_H2_MWH_PER_KG
    df["delta_lcoh_eur_per_kg"] = df["delta_lcoh_eur_per_MWh"] * LHV_H2_MWH_PER_KG

    columns = [
        "run_id",
        "excluded_technology",
        "total_demand_MWh",
        "cost_first_best",
        "cost_reopt",
        "delta_cost",
        "delta_cost_pct",
        "lcoh_first_best_eur_per_MWh",
        "lcoh_reopt_eur_per_MWh",
        "delta_lcoh_eur_per_MWh",
        "delta_lcoh_pct",
        "lcoh_first_best_eur_per_kg",
        "lcoh_reopt_eur_per_kg",
        "delta_lcoh_eur_per_kg",
        "installed_tecs_original",
        "removed_tech_sizes",
        "error",
    ]
    out = df[columns].sort_values("delta_lcoh_pct", ascending=False)

    n_missing = int(out["total_demand_MWh"].isna().sum())
    if n_missing:
        print(f"WARNING: {n_missing} runs without a demand value (LCOH left empty)")

    stats = out[
        ["delta_cost_pct", "delta_lcoh_eur_per_MWh", "delta_lcoh_eur_per_kg"]
    ].describe()
    print(stats.to_string())

    output_path = os.path.join(REOPT_DIR, OUTPUT_FILE)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="comparison", index=False)
        stats.to_excel(writer, sheet_name="statistics")
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()