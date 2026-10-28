"""
How does the system rebuild itself when the small electrolyzer is taken away?

For the runs belonging to the four uncertainty draws whose cost penalty exceeds 5%,
this reads the first-best and the re-optimised HDF5 result files and extracts the
design (technology and pipeline sizes), the annual energy flows, and the cost
breakdown, so the substitution mechanism can be compared draw by draw.

Typical-day results are scaled to a full year with the k-means factors.

Writes <REOPT_DIR>/figures_design/design_substitution.xlsx
"""

from __future__ import annotations

import glob
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pandas as pd

BASE = r"X:/3000_simulations_24_may_2026"
ORIG_DIR = os.path.join(BASE, "Snellius_3000_simulations")
REOPT_DIR = os.path.join(
    BASE, "reopt_comparison_Snellius_3000_simulations_small_electrolyzer_only"
)
CLUSTER_FILE = os.path.join(REOPT_DIR, "figures_clusters", "cluster_analysis.xlsx")
OUT_DIR = os.path.join(REOPT_DIR, "figures_design")

# Which draws to read, selected by their median cost penalty. Override on the command
# line, e.g. `python extract_design_substitution.py 1 5` for the 1-5% band.
PENALTY_BAND = (5.0, float("inf"))


def band_tag(band: tuple[float, float]) -> str:
    low, high = band
    return f"{low:g}plus" if high == float("inf") else f"{low:g}to{high:g}"


def out_file(band: tuple[float, float]) -> str:
    return os.path.join(OUT_DIR, f"design_substitution_{band_tag(band)}.xlsx")

LARGE_NODES = ["Large_cluster1", "Large_cluster2"]
SMALL_NODES = ["Small_cluster1", "Small_cluster2"]
ALL_NODES = LARGE_NODES + SMALL_NODES

SUMMARY_COSTS = [
    "cost_capex_tecs",
    "cost_capex_netws",
    "cost_opex_tecs",
    "cost_opex_netws",
    "cost_imports",
    "cost_exports",
]

# h5py issues many small random reads, which are pathologically slow over the SMB share
# (~6 s per file, and far worse under concurrency). Copying the 2 MB file locally first and
# reading it from disk is roughly five times faster and degrades gracefully in parallel.
N_WORKERS = 6
N_RETRIES = 3
SCRATCH = os.path.join(tempfile.gettempdir(), "adopt_h5_scratch")


def scalar(group, path: str) -> float:
    """Read a scalar or length-1 dataset, returning NaN when it is absent."""
    try:
        value = group[path][()]
    except KeyError:
        return float("nan")
    return float(np.ravel(value)[0])


def arc_class(from_node: str, to_node: str) -> str:
    large_from = from_node.startswith("Large")
    large_to = to_node.startswith("Large")
    if large_from and large_to:
        return "large_large"
    if not large_from and not large_to:
        return "small_small"
    return "small_large"


def extract_via_local_copy(remote_path: str) -> dict:
    """Copy the result file to local scratch, read it there, then remove the copy.

    Network hiccups on the share surface as OSError or, from inside h5py, RuntimeError,
    so both are retried before giving up.
    """
    os.makedirs(SCRATCH, exist_ok=True)
    local_path = os.path.join(SCRATCH, f"{uuid.uuid4().hex}.h5")
    last_error: Exception | None = None
    try:
        for attempt in range(N_RETRIES):
            try:
                shutil.copyfile(remote_path, local_path)
                return extract(local_path)
            except (OSError, RuntimeError, KeyError) as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
    raise RuntimeError(f"failed after {N_RETRIES} attempts: {last_error}")


def extract(h5_path: str) -> dict:
    out: dict[str, float] = {}
    with h5py.File(h5_path, "r") as f:
        for key in SUMMARY_COSTS:
            out[key] = scalar(f, f"summary/{key}")

        # Annual weights for the typical days.
        factors = np.ravel(f["k_means_specs/period1/factors"][()]).astype(float)

        # --- design: technology sizes -------------------------------------------
        for node in ALL_NODES:
            base = f"design/nodes/period1/{node}"
            if base not in f:
                continue
            for tec in f[base]:
                if tec.startswith("hydrogen_Compressor"):
                    continue
                out[f"size|{tec}|{node}"] = scalar(f, f"{base}/{tec}/size")
                out[f"capex|{tec}|{node}"] = scalar(f, f"{base}/{tec}/capex_tot")

        # --- design: pipelines, aggregated by network and arc class -------------
        netw_base = "design/networks/period1"
        if netw_base in f:
            for netw in f[netw_base]:
                for arc in f[f"{netw_base}/{netw}"]:
                    grp = f[f"{netw_base}/{netw}/{arc}"]
                    from_node = grp["fromNode"][()].decode()
                    to_node = grp["toNode"][()].decode()
                    cls = arc_class(from_node, to_node)
                    for field, target in (
                        ("size", "size"),
                        ("total_flow", "flow"),
                        ("capex", "capex"),
                    ):
                        key = f"netw_{target}|{netw}|{cls}"
                        out[key] = out.get(key, 0.0) + scalar(grp, field)

        # --- operation: annual energy flows -------------------------------------
        for node in ALL_NODES:
            eb = f"operation/energy_balance/period1/{node}"
            if eb not in f:
                continue
            for carrier in ("hydrogen", "electricity"):
                if carrier not in f[eb]:
                    continue
                for field in ("import", "demand", "export"):
                    series = np.ravel(f[f"{eb}/{carrier}/{field}"][()]).astype(float)
                    out[f"annual|{carrier}_{field}|{node}"] = float(
                        np.sum(series * factors)
                    )

            top = f"operation/technology_operation/period1/{node}"
            if top not in f:
                continue
            for tec in f[top]:
                if tec.startswith("hydrogen_Compressor"):
                    continue
                if "hydrogen_output" in f[f"{top}/{tec}"]:
                    series = np.ravel(f[f"{top}/{tec}/hydrogen_output"][()]).astype(float)
                    out[f"annual|h2_out|{tec}|{node}"] = float(np.sum(series * factors))
    return out


def find_h5(folder: str) -> str | None:
    matches = glob.glob(os.path.join(folder, "userData", "*", "optimization_results.h5"))
    return matches[0] if matches else None


def extract_pair(run_id: str) -> dict | None:
    orig = find_h5(os.path.join(ORIG_DIR, run_id))
    reopt = find_h5(
        os.path.join(REOPT_DIR, f"{run_id}__excl_Electrolyzer_small")
    )
    if orig is None or reopt is None:
        print(f"  {run_id}: missing result file", flush=True)
        return None
    try:
        a = extract_via_local_copy(orig)
        b = extract_via_local_copy(reopt)
    except Exception as exc:  # a single unreadable run must not kill the batch
        print(f"  {run_id}: {exc}", flush=True)
        return None

    row: dict[str, object] = {"run_id": run_id}
    for key in sorted(set(a) | set(b)):
        row[f"fb__{key}"] = a.get(key, np.nan)
        row[f"ro__{key}"] = b.get(key, np.nan)
    return row


def main(band: tuple[float, float] = PENALTY_BAND) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    runs = pd.read_excel(CLUSTER_FILE, sheet_name="runs")
    draws = pd.read_excel(CLUSTER_FILE, sheet_name="draws")
    low, high = band
    selected = draws.loc[
        (draws["penalty_median_pct"] >= low) & (draws["penalty_median_pct"] < high),
        "draw",
    ].tolist()
    target = runs[runs["draw"].isin(selected)][
        ["run_id", "draw", "scenario", "delta_cost_pct", "delta_lcoh_eur_per_kg"]
    ].copy()
    print(
        f"penalty band [{low}, {high})%: draws {sorted(selected)} "
        f"-> {len(target)} run pairs to read"
    )

    started = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for done, row in enumerate(pool.map(extract_pair, target["run_id"]), start=1):
            if row is not None:
                rows.append(row)
            if done % 10 == 0 or done == len(target):
                rate = (time.time() - started) / done
                print(
                    f"  {done}/{len(target)} pairs, {rate:.1f} s/pair, "
                    f"~{rate * (len(target) - done) / 60:.0f} min left",
                    flush=True,
                )
    print(f"read {len(rows)} pairs of {len(target)}")

    data = target.merge(pd.DataFrame(rows), on="run_id", how="inner")

    # Deltas for every extracted quantity.
    for column in [c for c in data.columns if c.startswith("fb__")]:
        key = column[4:]
        reopt_col = f"ro__{key}"
        if reopt_col in data.columns:
            data[f"d__{key}"] = data[reopt_col] - data[column]

    path = out_file(band)
    data.to_excel(path, sheet_name="pairs", index=False)
    print(f"Written: {path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        main((float(sys.argv[1]), float(sys.argv[2])))
    else:
        main()