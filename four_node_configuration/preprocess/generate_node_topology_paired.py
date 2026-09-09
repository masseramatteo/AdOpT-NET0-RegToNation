"""
Four-node spatial topology design (v5) - paired-by-economy design ("C2").

Why v5
------
v4 produces one pool of 40 topologies that every economic sample is crossed
with. For the decision trees that means 40 distinct distance triples in the
whole campaign, and the tree can only place a distance split between two of
them. v5 gives EVERY economic sample its own topologies, so distances become a
continuous sampled parameter like price or demand, while keeping a small paired
block inside each economy for within-economy contrasts.

Design per economy (8 topologies)
---------------------------------
    A_near, A_far   two triples, "near" from the lower half of d_SL and "far"
    B_near, B_far   from the upper half, SHARED between A and B (mirror
                    archetypes with the identical feasibility region), so
                    "attach to the larger vs the smaller hub" is a clean
                    paired contrast at fixed distances
    D_near, D_far   two own triples for the one-small-per-hub archetype
    C_near, C_far   two own triples for the midway archetype (lean to L1 / L2
                    alternated, balanced over the campaign)

D and C do NOT share the A/B triples: forcing D onto them would remove
"attached smalls with far-apart hubs" from A/B (D needs d_SL >= |d_LL-d_SS|/2),
and C is confined to d_LL <~ 1000 km by construction (midway means ~d_LL/2 from
each hub with d_SL <= 500).

Sampling
--------
For each archetype group a scrambled Sobol sequence over the full parameter box
is filtered for geometric feasibility and the first N_ECON*2 feasible points are
kept, i.e. a low-discrepancy sample UNIFORM over that archetype's feasible
region. Nothing is clamped or rescaled. Within a group the points are split at
the median d_SL into near / far halves and paired at random; pairs are then
assigned to economies by a random permutation, so distances are independent of
the economic parameters by construction.

Geometry, coordinates and archetype definitions are imported unchanged from v4
(generate_node_topology_pooled.py): closed-form node positions, centrality
threshold NEARNESS_RATIO = C_MIN = 0.90, azimuthal-equidistant projection with
least-squares refinement against the haversine distances used downstream.

Outputs (OUTPUT_FOLDER = generated_topology_v5/)
------------------------------------------------
    NodeLocations_XXXX.csv      one per topology (XXXX = 0001 .. 8*N_ECON)
    design_manifest.csv         one row per economy: the 8 topology names
    topology_parameters.csv     one row per topology, targets + realized
    topology_nodes_long.csv     one row per (topology, small cluster)
    topology_coverage.txt       pooled, per-archetype and per-pair coverage
    design_diagnostics.png      marginals and pairwise scatter per archetype
    methodology_note.txt        paper-ready description

The runner must pair economy i with the 8 topologies listed for it in
design_manifest.csv instead of crossing every economy with every topology.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

import generate_node_topology_pooled as v4
from generate_node_topology_pooled import (
    ARCHETYPE_NAME, ARCHETYPES, BIG_NODES, NODES, SMALL_NODES,
    D_LL_RANGE, D_SL_RANGE, D_SS_RANGE, NEARNESS_RATIO, C_MIN,
    build_topology, positions_to_lonlat, unit_to_params, write_node_locations,
    plot_topology, max_normalised_gap,
)

# ------------------------------------------------------------------ config --

N_ECON = 750                    # economies; runs = 8 * N_ECON
POOL_SIZE = 2 ** 17             # Sobol candidates over the box, per group
SEED = 42
SLOT_ORDER = ["A_near", "A_far", "B_near", "B_far",
              "D_near", "D_far", "C_near", "C_far"]
GROUPS = {"AB": "A", "D": "D", "C": "C1"}   # feasibility is tested with this label
N_STRATA_REPORT = 60            # bins for the pooled coverage report
PLOT_EVERY = 1000               # individual layout PNGs (0 = none)

OUTPUT_FOLDER = Path(__file__).resolve().parent / "generated_topology_v5"
RANGES = {"d_LL": D_LL_RANGE, "d_SL": D_SL_RANGE, "d_SS": D_SS_RANGE}


# ---------------------------------------------------------------- sampling --

def feasible_sobol(label, n_needed, seed):
    """First n_needed feasible points, in Sobol order, of a scrambled Sobol
    sequence over the box. Uniform over the archetype's feasible region."""
    U = qmc.Sobol(d=3, scramble=True, seed=seed).random(POOL_SIZE)
    keep = []
    for k in range(len(U)):
        if build_topology(label, *unit_to_params(U[k])) is not None:
            keep.append(k)
            if len(keep) == n_needed:
                break
    if len(keep) < n_needed:
        raise RuntimeError(f"group {label}: only {len(keep)} feasible of "
                           f"{POOL_SIZE} candidates, need {n_needed}; raise POOL_SIZE")
    frac = len(keep) / (k + 1)
    return U[keep], frac


def near_far_pairs(U, rng):
    """Split at the median d_SL and pair by rank: the i-th smallest of the
    lower half with the i-th smallest of the upper half, so every economy sees
    a consistent near-to-far step of about half the d_SL range.
    Returns array (n_pairs, 2) of row indices into U."""
    d_sl = U[:, 1]
    order = np.argsort(d_sl)
    half = len(U) // 2
    near, far = order[:half], order[half:2 * half]
    return np.column_stack([near, far])


def build_design(seed=SEED):
    """Return dict group -> (U_group, pairs assigned per economy, feasible frac)."""
    rng = np.random.default_rng(seed)
    design = {}
    for g_i, (grp, label) in enumerate(GROUPS.items()):
        U, frac = feasible_sobol(label, 2 * N_ECON, seed + 100 * g_i)
        pairs = near_far_pairs(U, rng)
        perm = rng.permutation(N_ECON)          # economy -> pair, random
        design[grp] = (U, pairs[perm], frac)
        print(f"  group {grp}: {len(U)} points, feasible share of the box "
              f"{frac * 100:.1f}%")
    return design


# ------------------------------------------------------------------ report --

def eta2(df, col):
    v = df[col].to_numpy()
    grand = v.mean()
    ss_tot = float(((v - grand) ** 2).sum())
    ss_b = float(sum(len(g) * (g[col].mean() - grand) ** 2
                     for _, g in df.groupby("archetype")))
    return ss_b / ss_tot if ss_tot > 0 else 0.0


def coverage_report(df, fracs, max_err):
    n = len(df)
    params = {"d_LL": D_LL_RANGE, "d_SL_mean": D_SL_RANGE, "d_SS": D_SS_RANGE}
    out = ["Paired topology design (v5) - realized coverage", "=" * 72,
           f"{N_ECON} economies x 8 topologies = {n} topologies",
           f"distinct distance triples = {df[['d_LL', 'd_SL_mean', 'd_SS']].round(6).drop_duplicates().shape[0]} "
           f"(A and B share theirs by design)",
           f"centrality threshold = {C_MIN}, nearness ratio = {NEARNESS_RATIO}",
           f"max haversine-vs-planar residual = {max_err:.3f} km", "",
           f"POOLED over all {n} rows - {N_STRATA_REPORT} strata per parameter",
           "-" * 72]
    for p, (lo, hi) in params.items():
        v = df[p].to_numpy()
        bins = np.clip(((v - lo) / (hi - lo) * N_STRATA_REPORT).astype(int),
                       0, N_STRATA_REPORT - 1)
        counts = np.bincount(bins, minlength=N_STRATA_REPORT)
        out.append(f"    {p:<10} {v.min():7.0f} - {v.max():7.0f} km"
                   f"   strata filled {int((counts > 0).sum())}/{N_STRATA_REPORT}"
                   f"   min/max rows per stratum {counts.min()}/{counts.max()}"
                   f"   max gap {max_normalised_gap(v, lo, hi):.4f}")
    c = df[["d_LL", "d_SL_mean", "d_SS"]].corr().to_numpy()
    out.append(f"    max |corr| pooled: "
               f"{max(abs(c[0, 1]), abs(c[0, 2]), abs(c[1, 2])):.2f}")
    out += ["", "Archetype vs distances - confounding check (eta^2)", "-" * 72]
    for p in params:
        e = eta2(df, p)
        out.append(f"    {p:<10} eta^2 = {e:.2f}{'  <-- entangled' if e > 0.5 else ''}")

    out += ["", "Per archetype - uniform over its own feasible region", "-" * 72]
    for a in ARCHETYPES:
        sub = df[df.archetype == a]
        grp = "AB" if a in "AB" else a
        out.append(f"{a} - {ARCHETYPE_NAME[a]}  ({len(sub)} rows, "
                   f"{fracs[grp] * 100:.1f}% of the box feasible)")
        for p, (lo, hi) in params.items():
            v = sub[p].to_numpy()
            bins = np.clip(((v - lo) / (hi - lo) * N_STRATA_REPORT).astype(int),
                           0, N_STRATA_REPORT - 1)
            out.append(f"    {p:<10} {v.min():7.0f} - {v.max():7.0f} km"
                       f"   spans {(v.max() - v.min()) / (hi - lo) * 100:5.1f}%"
                       f"   strata {int(len(set(bins.tolist())))}/{N_STRATA_REPORT}")
        cc = sub[["d_LL", "d_SL_mean", "d_SS"]].corr().to_numpy()
        out.append(f"    corr(LL,SL) {cc[0, 1]:+.2f}  corr(LL,SS) {cc[0, 2]:+.2f}"
                   f"  corr(SL,SS) {cc[1, 2]:+.2f}   (support-induced, reported not forced)")
        cen = sub[["S1_centrality", "S2_centrality"]].to_numpy()
        out.append(f"    centrality {cen.min():.2f} - {cen.max():.2f}")
        out.append("")

    out += ["Within-economy near/far contrast (d_SL far - d_SL near, km)", "-" * 72]
    for a in ARCHETYPES:
        near = df[(df.archetype == a) & (df.slot.str.endswith("near"))].sort_values("economy")
        far = df[(df.archetype == a) & (df.slot.str.endswith("far"))].sort_values("economy")
        diff = far.d_SL_mean.to_numpy() - near.d_SL_mean.to_numpy()
        out.append(f"    {a}: median {np.median(diff):6.0f}   p10 {np.percentile(diff, 10):6.0f}"
                   f"   min {diff.min():6.0f}")
    return "\n".join(out)


def plot_diagnostics(df, folder):
    params = ["d_LL", "d_SL_mean", "d_SS"]
    labels = {"d_LL": "d_LL [km]", "d_SL_mean": "d_SL [km]", "d_SS": "d_SS [km]"}
    rng_ = {"d_LL": D_LL_RANGE, "d_SL_mean": D_SL_RANGE, "d_SS": D_SS_RANGE}
    colors = dict(zip(ARCHETYPES, ["#161D41", "#4A6FA5", "#C1666B", "#FFCD00"]))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, p in zip(axes[0], params):
        lo, hi = rng_[p]
        bins = np.linspace(lo, hi, 31)
        ax.hist([df[df.archetype == a][p] for a in ARCHETYPES], bins=bins,
                stacked=True, color=[colors[a] for a in ARCHETYPES], label=ARCHETYPES)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(labels[p])
        ax.set_title("pooled marginal (stacked by archetype)", fontsize=9)
    axes[0][0].legend(fontsize=8)
    for ax, (px, py) in zip(axes[1], [("d_LL", "d_SL_mean"), ("d_LL", "d_SS"),
                                      ("d_SL_mean", "d_SS")]):
        for a in ARCHETYPES:
            sub = df[df.archetype == a].sample(min(400, (df.archetype == a).sum()),
                                               random_state=0)
            ax.scatter(sub[px], sub[py], s=6, color=colors[a], label=a, alpha=0.6)
        ax.set_xlabel(labels[px]); ax.set_ylabel(labels[py])
        ax.set_xlim(*rng_[px]); ax.set_ylim(*rng_[py])
        ax.grid(True, ls="--", alpha=0.25)
    fig.suptitle(f"v5 paired design: {N_ECON} economies x 8 topologies")
    fig.tight_layout()
    fig.savefig(folder / "design_diagnostics.png", dpi=150)
    plt.close(fig)


def methodology_note(df):
    return "\n".join([
        "Spatial topology design (v5) - generated description", "",
        f"Each of the {N_ECON} economic samples is paired with eight four-node "
        "configurations of its own: two per archetype, the two differing in the "
        "small-large distance (one drawn from the lower half of the d_SL range, "
        "one from the upper half). The two hub-attached archetypes (A, B) share "
        "the same two distance triples, so that attaching the small clusters to "
        "the larger or to the smaller hub is compared at identical distances; "
        "the split and midway archetypes receive their own triples because "
        "their feasible regions differ from that of the attached archetypes.",
        "",
        "Distances are sampled as a scrambled Sobol sequence over the full "
        f"ranges d_LL [{D_LL_RANGE[0]:.0f}, {D_LL_RANGE[1]:.0f}] km, d_SL "
        f"[{D_SL_RANGE[0]:.0f}, {D_SL_RANGE[1]:.0f}] km and d_SS "
        f"[{D_SS_RANGE[0]:.0f}, {D_SS_RANGE[1]:.0f}] km, filtered for geometric "
        "feasibility under each archetype and never clamped or rescaled; the "
        "sample is therefore uniform over each archetype's feasible region and "
        "the realized coverage per archetype is reported. Triples are assigned "
        "to economic samples by random permutation, so the spatial and the "
        "techno-economic parameters are independent by construction. Across the "
        f"campaign this yields {df[['d_LL', 'd_SL_mean', 'd_SS']].round(6).drop_duplicates().shape[0]} "
        "distinct distance triples, so distance enters the decision-tree "
        "analysis as a continuous parameter.",
        "",
        "Node positions are the closed-form solution of the three distance "
        "constraints under the archetype's symmetry convention (see v4), "
        "projected with an azimuthal-equidistant inverse projection and refined "
        "so that the great-circle distances used by the optimisation model "
        "reproduce the sampled distances.",
    ])


# -------------------------------------------------------------------- main --

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"v5 paired design: {N_ECON} economies x 8 = {8 * N_ECON} topologies")
    design = build_design()
    fracs = {g: design[g][2] for g in design}

    rng = np.random.default_rng(SEED + 7)
    wide, long, manifest, max_err = [], [], [], 0.0
    counter = 0
    for econ in range(1, N_ECON + 1):
        row_m = {"economy": econ}
        # slots -> (group, label, pair column)
        c_leans = ["C1", "C2"] if rng.random() < 0.5 else ["C2", "C1"]
        slots = [("A_near", "AB", "A", 0), ("A_far", "AB", "A", 1),
                 ("B_near", "AB", "B", 0), ("B_far", "AB", "B", 1),
                 ("D_near", "D", "D", 0), ("D_far", "D", "D", 1),
                 ("C_near", "C", c_leans[0], 0), ("C_far", "C", c_leans[1], 1)]
        for slot, grp, label, col in slots:
            U, pairs, _ = design[grp]
            u = U[pairs[econ - 1, col]]
            d_ll, d_sl, d_ss = unit_to_params(u)
            built = build_topology(label, d_ll, d_sl, d_ss)
            if built is None:
                raise RuntimeError(f"economy {econ} {slot}: {label} lost feasibility "
                                   f"for ({d_ll:.1f}, {d_sl:.1f}, {d_ss:.1f})")
            pos, d = built
            counter += 1
            name = f"{counter:04d}"
            a = v4.public_archetype(label)
            lonlat, err = positions_to_lonlat(pos)
            max_err = max(max_err, err)
            write_node_locations(lonlat, name, OUTPUT_FOLDER)
            if PLOT_EVERY and (counter - 1) % PLOT_EVERY == 0:
                plot_topology(pos, a, name, OUTPUT_FOLDER)
            row_m[slot] = name
            r = {"topology": name, "economy": econ, "slot": slot, "archetype": a,
                 "label": label, "archetype_name": ARCHETYPE_NAME[a],
                 "d_LL_target": d_ll, "d_SL_target": d_sl, "d_SS_target": d_ss,
                 "proj_residual_km": err, **d}
            for node in NODES:
                r[f"{node}_x"], r[f"{node}_y"] = pos[node]
                r[f"{node}_lon"], r[f"{node}_lat"] = lonlat[node]
            wide.append(r)
            for i, sn in enumerate(SMALL_NODES, start=1):
                long.append({"topology": name, "economy": econ, "slot": slot,
                             "archetype": a, "small_cluster": sn,
                             "d_nearest_large": d[f"S{i}_near"],
                             "d_other_large": d[f"S{i}_far"],
                             "centrality": d[f"S{i}_centrality"],
                             "d_small_small": d["d_SS"], "d_large_large": d["d_LL"],
                             "rho_comp": d["rho_comp"]})
        manifest.append(row_m)
        if econ % 100 == 0:
            print(f"  economy {econ}/{N_ECON} written")

    dfw, dfl, dfm = pd.DataFrame(wide), pd.DataFrame(long), pd.DataFrame(manifest)
    dfw.to_csv(OUTPUT_FOLDER / "topology_parameters.csv", sep=";", index=False)
    dfl.to_csv(OUTPUT_FOLDER / "topology_nodes_long.csv", sep=";", index=False)
    dfm.to_csv(OUTPUT_FOLDER / "design_manifest.csv", sep=";", index=False)
    plot_diagnostics(dfw, OUTPUT_FOLDER)
    report = coverage_report(dfw, fracs, max_err)
    (OUTPUT_FOLDER / "topology_coverage.txt").write_text(report, encoding="utf-8")
    (OUTPUT_FOLDER / "methodology_note.txt").write_text(methodology_note(dfw), encoding="utf-8")
    print("\n" + report)
    print(f"\n{len(dfw)} topologies written to '{OUTPUT_FOLDER}'")


if __name__ == "__main__":
    main()