"""
Step 1 - Build the outputs-of-interest matrix (one row per scenario).

Reads extracted_results.xlsx and constructs the 9 infrastructure outputs of interest
plus the sampled driver inputs. All shares are computed from *_sum columns only, so
their (unknown/mixed) units cancel. Per-node H2 demand is reconstructed from the
annual flow balance:

    node_demand = production + import + network_inflow - network_outflow

(storage nets to ~0 over the year; verified 0 negative rows on this dataset).

Run:  python build_features.py
Out:  infrastructure_features.csv   (keyed by run/scenario id)
"""

import numpy as np
import pandas as pd

import config as C

NODES = ["Large_cluster1", "Large_cluster2", "Small_cluster1", "Small_cluster2"]
LARGE = ["Large_cluster1", "Large_cluster2"]
SMALL = ["Small_cluster1", "Small_cluster2"]

# 6 undirected arcs, as stored in the extracted columns (direction is arbitrary)
ARCS = [
    ("Large_cluster1", "Large_cluster2"),
    ("Large_cluster1", "Small_cluster1"),
    ("Large_cluster1", "Small_cluster2"),
    ("Large_cluster2", "Small_cluster1"),
    ("Large_cluster2", "Small_cluster2"),
    ("Small_cluster1", "Small_cluster2"),
]


def _arc_cols(a, b):
    base = f"{a}_to_{b}"
    return base + "_hydrogenPipelineOnshore_highP", base + "_hydrogenPipelineOnshore_lowP"


def build():
    df = pd.read_excel(C.EXTRACTED_RESULTS)
    print(f"Loaded {len(df)} runs from extracted_results.xlsx")

    # Keep successful runs if a status column exists (it may not in this extraction)
    if "status" in df.columns:
        n0 = len(df)
        df = df[df["status"] == "SUCCESS"].copy()
        print(f"Filtered to {len(df)} SUCCESS runs (of {n0})")

    out = pd.DataFrame(index=df.index)
    out["run"] = df["run"].values
    out["archetype"] = df["archetype"].values
    # v5 paired design: keep the grouping key and slot so downstream splits can
    # be grouped by economy (C.GROUP_COL)
    for _c in ("economy", "slot"):
        if _c in df.columns:
            out[_c] = df[_c].values

    # --- Production, imports, network flows, reconstructed demand -------------
    # NaN in a flow/production column means the technology was not installed
    # (no output column), i.e. the flow is genuinely 0 - fill accordingly so it
    # does not propagate NaN into the aggregate totals.
    _flow_cols = []
    for n in NODES:
        _flow_cols += [f"{n}_TOTAL_H2_production", f"{n}_hydrogen_import_sum",
                       f"{n}_hydrogen_network_inflow_sum", f"{n}_hydrogen_network_outflow_sum"]
    df[_flow_cols] = df[_flow_cols].fillna(0.0)

    prod = {n: df[f"{n}_TOTAL_H2_production"] for n in NODES}
    imp = {n: df[f"{n}_hydrogen_import_sum"] for n in NODES}
    inflow = {n: df[f"{n}_hydrogen_network_inflow_sum"] for n in NODES}
    outflow = {n: df[f"{n}_hydrogen_network_outflow_sum"] for n in NODES}
    demand = {n: prod[n] + imp[n] + inflow[n] - outflow[n] for n in NODES}

    total_prod = sum(prod[n] for n in NODES)
    total_dem = sum(demand[n] for n in NODES)
    total_imp = sum(imp[n] for n in NODES)
    total_transport = sum(outflow[n] for n in NODES)   # H2 moved over the network
    small_prod = sum(prod[n] for n in SMALL)
    small_dem = sum(demand[n] for n in SMALL)

    n_neg = (total_dem < -1e-6).sum() + (small_dem < -1e-6).sum()
    if n_neg:
        print(f"  WARNING: {n_neg} runs with negative reconstructed demand")

    # 1) local self-sufficiency: how much of small-cluster demand is covered by
    #    the small clusters' own production. Capped at 1.0 (100% coverage); the
    #    raw ratio (which can exceed 1 for net exporters) is kept as a diagnostic.
    _raw = _safe_div(small_prod, small_dem, fill=0.0)
    out["_local_prod_ratio_raw"] = _raw
    out["local_self_sufficiency"] = np.minimum(_raw, 1.0)
    # 2) import intensity
    out["import_intensity"] = _safe_div(total_imp, total_dem)
    # 3) transport intensity: volume of H2 moved over the network vs total demand
    out["transport_intensity"] = _safe_div(total_transport, total_dem)

    # 3) storage siting - NEWLY BUILT ONLY (exclude existing Cavern)
    _stor_cols = ["Large_cluster1_Storage_H2_highP", "Large_cluster2_Storage_H2_highP",
                  "Small_cluster1_Storage_H2_lowP", "Small_cluster2_Storage_H2_lowP"]
    df[_stor_cols] = df[_stor_cols].fillna(0.0)
    large_stor = df["Large_cluster1_Storage_H2_highP"] + df["Large_cluster2_Storage_H2_highP"]
    small_stor = df["Small_cluster1_Storage_H2_lowP"] + df["Small_cluster2_Storage_H2_lowP"]
    total_stor = large_stor + small_stor
    # if no storage built at all, share is undefined -> 0 (no large storage); track it
    out["storage_large_share"] = _safe_div(large_stor, total_stor, fill=0.0)
    out["_any_storage"] = (total_stor > C.BUILD_EPS).astype(int)

    # --- Topology ------------------------------------------------------------
    built = {}          # (a,b) -> bool series, arc built at all
    built_highp = {}    # (a,b) -> bool series, highP variant built
    for a, b in ARCS:
        hp_col, lp_col = _arc_cols(a, b)
        hp = df[hp_col] if hp_col in df.columns else 0.0
        lp = df[lp_col] if lp_col in df.columns else 0.0
        built[(a, b)] = (hp > C.BUILD_EPS) | (lp > C.BUILD_EPS)
        built_highp[(a, b)] = (hp > C.BUILD_EPS)

    def arc(a, b):
        # arcs stored one direction only; try both orders
        if (a, b) in built:
            return built[(a, b)]
        return built[(b, a)]

    def arc_hp(a, b):
        if (a, b) in built_highp:
            return built_highp[(a, b)]
        return built_highp[(b, a)]

    # 5) S-S pipeline present
    out["SS_pipe"] = arc("Small_cluster1", "Small_cluster2").astype(int)
    # 6) L-L pipeline present
    out["LL_pipe"] = arc("Large_cluster1", "Large_cluster2").astype(int)

    # per-small-cluster connection degree
    deg_s1 = (arc("Large_cluster1", "Small_cluster1").astype(int)
              + arc("Large_cluster2", "Small_cluster1").astype(int)
              + arc("Small_cluster1", "Small_cluster2").astype(int))
    deg_s2 = (arc("Large_cluster1", "Small_cluster2").astype(int)
              + arc("Large_cluster2", "Small_cluster2").astype(int)
              + arc("Small_cluster1", "Small_cluster2").astype(int))
    out["degree_small1"] = deg_s1
    out["degree_small2"] = deg_s2
    # 7) meshed vs satellite: any small cluster with degree > 1
    out["meshed"] = ((deg_s1 > 1) | (deg_s2 > 1)).astype(int)

    # 8) number of pipelines built (of 6 arcs)
    n_pipes = sum(built[(a, b)].astype(int) for a, b in ARCS)
    out["n_pipelines"] = n_pipes
    # 9) high-pressure share of built arcs
    n_hp = sum(built_highp[(a, b)].astype(int) for a, b in ARCS)
    out["highP_share"] = _safe_div(n_hp, n_pipes, fill=0.0)

    # --- Driver inputs (Step 4) ---------------------------------------------
    # derived compactness/distance feature (avg small->nearest large)
    s1_near = df[["distance_Large_cluster1_to_Small_cluster1_km",
                  "distance_Large_cluster2_to_Small_cluster1_km"]].min(axis=1)
    s2_near = df[["distance_Large_cluster1_to_Small_cluster2_km",
                  "distance_Large_cluster2_to_Small_cluster2_km"]].min(axis=1)
    df["distance_from_large_cluster"] = (s1_near + s2_near) / 2

    for col in C.DRIVER_INPUTS:
        if col in df.columns:
            out[col] = df[col].values
        else:
            print(f"  WARNING: driver input '{col}' not found")

    out = out.reset_index(drop=True)
    out.to_csv(C.FEATURES_CSV, index=False)
    print(f"\nSaved {len(out)} rows x {out.shape[1]} cols -> {C.FEATURES_CSV}")

    _report(out, total_stor)
    return out


def _safe_div(num, den, fill=np.nan):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(den) > 1e-12, num / den, fill)
    return r


def _report(out, total_stor):
    print("\n" + "=" * 70)
    print("OUTPUTS OF INTEREST - summary statistics")
    print("=" * 70)
    with pd.option_context("display.float_format", lambda v: f"{v:8.3f}"):
        print(out[C.OUTPUTS_OF_INTEREST].describe().T[["mean", "std", "min", "50%", "max"]])
    print(f"\nRuns with no storage built at all: {(out['_any_storage'] == 0).sum()} "
          f"(storage_large_share set to 0 for these)")
    print("\nTopology binary/count value counts:")
    for c in ["SS_pipe", "LL_pipe", "meshed", "n_pipelines"]:
        print(f"  {c}: {out[c].value_counts().sort_index().to_dict()}")
    print("\nArchetype distribution:")
    print(f"  {out['archetype'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    build()