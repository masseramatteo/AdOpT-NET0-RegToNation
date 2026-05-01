"""
Cost comparison between a first-best run and a re-optimized run.

Extracts from the h5 summary:
  - Tech CAPEX / OPEX
  - Network CAPEX / OPEX
  - Import cost (total + breakdown by carrier: electricity, hydrogen)
  - Total NPV (cost)

Usage:
    from extract_cost_comparison import compare_runs, print_table

    table = compare_runs(
        orig_run_folder  = r"X:\...\parallel_run_0092",
        reopt_run_folder = r"X:\...\parallel_run_0092__excl_Electrolyzer_small+Storage_H2_lowP",
    )
    print_table(table)

Or run directly:
    python extract_cost_comparison.py
    (edit ORIG_FOLDER and REOPT_FOLDER at the bottom)
"""

import h5py
import os
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# H5 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_h5(run_folder):
    """Return path to optimization_results.h5 inside a run folder."""
    ud = Path(run_folder) / "userData"
    if not ud.exists():
        raise FileNotFoundError(f"userData not found in {run_folder}")
    timestamps = [d for d in os.listdir(str(ud)) if os.path.isdir(str(ud / d))]
    if not timestamps:
        raise FileNotFoundError(f"No timestamp folder in {ud}")
    h5 = ud / timestamps[0] / "optimization_results.h5"
    if not h5.exists():
        raise FileNotFoundError(f"optimization_results.h5 not found at {h5}")
    return str(h5)


def _read_summary(h5_path):
    """Read all cost fields from the summary group."""
    with h5py.File(h5_path, "r") as f:
        s = f["summary"]
        return {
            "tec_capex":   float(s["cost_capex_tecs"][()]),
            "tec_opex":    float(s["cost_opex_tecs"][()]),
            "net_capex":   float(s["cost_capex_netws"][()]),
            "net_opex":    float(s["cost_opex_netws"][()]),
            "import_cost": float(s["cost_imports"][()]),
            "export_rev":  float(s["cost_exports"][()]),
            "total_npv":   float(s["total_npv"][()]),
        }


def _import_by_carrier(h5_path):
    """
    Break down import cost by carrier (electricity, hydrogen, ...).

    Uses import * import_price element-wise as a proxy for the fraction,
    then scales to match the summary total so units are consistent.
    """
    with h5py.File(h5_path, "r") as f:
        summary_total = float(f["summary/cost_imports"][()])
        carrier_raw = {}
        base = "operation/energy_balance/period1"
        if base not in f:
            return {}
        for node in f[base].keys():
            for carrier in f[f"{base}/{node}"].keys():
                g = f[f"{base}/{node}/{carrier}"]
                if "import" in g and "import_price" in g:
                    cost = float(np.sum(g["import"][()] * g["import_price"][()]))
                    carrier_raw[carrier] = carrier_raw.get(carrier, 0.0) + cost

    total_raw = sum(carrier_raw.values())
    if total_raw == 0:
        return {c: 0.0 for c in carrier_raw}
    return {c: v / total_raw * summary_total for c, v in carrier_raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison function
# ─────────────────────────────────────────────────────────────────────────────

def compare_runs(orig_run_folder, reopt_run_folder):
    """
    Compare cost breakdown between the original first-best run and a re-opt run.

    Args:
        orig_run_folder:  path to the original parallel_run_XXXX folder
        reopt_run_folder: path to the re-opt parallel_run_XXXX__excl_... folder

    Returns:
        List of dicts with keys: item, orig, reopt, delta, delta_pct
    """
    h5_orig  = _find_h5(orig_run_folder)
    h5_reopt = _find_h5(reopt_run_folder)

    orig_s  = _read_summary(h5_orig)
    reopt_s = _read_summary(h5_reopt)

    orig_imp  = _import_by_carrier(h5_orig)
    reopt_imp = _import_by_carrier(h5_reopt)

    all_carriers = sorted(set(list(orig_imp.keys()) + list(reopt_imp.keys())))

    def row(label, o, r):
        d = r - o
        pct = (d / abs(o) * 100) if o != 0 else None
        return {"item": label, "orig": o, "reopt": r, "delta": d, "delta_pct": pct}

    rows = [
        row("Tech CAPEX",    orig_s["tec_capex"],   reopt_s["tec_capex"]),
        row("Tech OPEX",     orig_s["tec_opex"],    reopt_s["tec_opex"]),
        row("Network CAPEX", orig_s["net_capex"],   reopt_s["net_capex"]),
        row("Network OPEX",  orig_s["net_opex"],    reopt_s["net_opex"]),
        row("Import cost",   orig_s["import_cost"], reopt_s["import_cost"]),
    ]

    for carrier in all_carriers:
        o = orig_imp.get(carrier, 0.0)
        r = reopt_imp.get(carrier, 0.0)
        rows.append(row(f"  - {carrier} import", o, r))

    rows.append(row("Export revenue", orig_s["export_rev"], reopt_s["export_rev"]))
    rows.append({"item": "---", "orig": None, "reopt": None, "delta": None, "delta_pct": None})
    rows.append(row("TOTAL NPV (cost)", orig_s["total_npv"], reopt_s["total_npv"]))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_table(rows):
    """Print comparison table to console."""
    print(f"\n{'Item':<25} {'Original (M euro)':>18} {'Reopt (M euro)':>15} {'Delta (M euro)':>15} {'Delta%':>10}")
    print("-" * 85)
    for r in rows:
        if r["item"] == "---":
            print("-" * 85)
            continue
        o   = f"{r['orig']/1e6:>18.3f}"   if r["orig"]  is not None else " " * 18
        re  = f"{r['reopt']/1e6:>15.3f}"  if r["reopt"] is not None else " " * 15
        d   = f"{r['delta']/1e6:>+15.3f}" if r["delta"] is not None else " " * 15
        pct = f"{r['delta_pct']:>9.2f}%"  if r["delta_pct"] is not None else "         -"
        print(f"{r['item']:<25} {o} {re} {d} {pct}")


def to_markdown_table(rows):
    """Return a markdown table string ready to paste into Obsidian."""
    lines = [
        "| Item | Original (M euro) | Reopt (M euro) | Delta (M euro) | Delta% |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["item"] == "---":
            lines.append("| | | | | |")
            continue
        o   = f"{r['orig']/1e6:.3f}"        if r["orig"]  is not None else ""
        re  = f"{r['reopt']/1e6:.3f}"       if r["reopt"] is not None else ""
        d   = f"{r['delta']/1e6:+.3f}"      if r["delta"] is not None else ""
        pct = f"{r['delta_pct']:.2f}%"      if r["delta_pct"] is not None else "-"
        lines.append(f"| {r['item']} | {o} | {re} | {d} | {pct} |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ORIG_FOLDER  = r"X:\new_limits_on_large_cluster_simulations\1600_simulations_with_new_constraints\parallel_run_0092"
    REOPT_FOLDER = r"X:\new_limits_on_large_cluster_simulations\reopt_comparison_1600_simulations_with_new_constraints_out_decentralized\reopt_runs\parallel_run_0092__excl_Electrolyzer_small+Storage_H2_lowP"

    rows = compare_runs(ORIG_FOLDER, REOPT_FOLDER)

    print_table(rows)

    print("\n\nMarkdown table:\n")
    print(to_markdown_table(rows))