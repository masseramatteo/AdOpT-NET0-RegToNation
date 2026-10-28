"""
Four-node spatial topology design - screening campaign on the small-large distance.

Why this design
---------------
In the v5 campaign the whole d_SL range [100, 500] km sits BEYOND the distance at
which a pipeline from the hub stops being cheaper than a local electrolyzer. On the
5992 solved runs the capex break-even sits between 60 and 320 km depending on the
two cost multipliers, while the sampled d_SL median is 382 km, so distance decides
almost nothing: the small cluster produces locally in 90.9% of runs and d_SL carries
no tree importance.

This design keeps the v5 box and OPENS IT DOWNWARD, so the sample straddles the
break-even instead of sitting above it:

                    v5                  screening
    d_LL       [400, 1500] km        [150,  500] km
    d_SL       [100,  500] km        [ 20,  300] km      <- the change that matters
    d_SS       [100, 1000] km        [ 40,  300] km

d_LL and d_SS follow d_SL down out of geometric necessity, not choice: with both
small clusters attached to the same hub the triangle inequality forces
d_SS <= 2*d_SL, and the midway archetype needs d_LL of the order of 2*d_SL.

Design per economy (3 topologies)
---------------------------------
Each economy gets ONE archetype and three topologies from that archetype's own
feasible region, ranked by d_SL and paired by rank:

    SL_low, SL_mid, SL_high

so the small-large distance is a WITHIN-ECONOMY contrast. In v5 the near/far
pairing was what made the distance effect visible at all - unpaired it was
invisible, paired it came out at -3.5 points with a CI clear of zero - and here
distance is the variable under test, so the design is built around that contrast.

One archetype per economy, not all four
---------------------------------------
Crossing every economy with all four archetypes would cost 12 runs per economy and
leave 54 economies, i.e. 2 replicates per cell of the 3x3x3 economic factorial.
One archetype per economy gives 216 economies and 8 replicates. The archetype
contrast itself is already measured on the v5 campaign over 749 paired economies;
what this campaign needs is power on the economic axes.

Per-archetype feasible regions
------------------------------
As in v5, each archetype is sampled uniformly over ITS OWN feasible region inside
the box - nothing is clamped or rescaled - so the realized d_SL ranges differ by
archetype and are reported rather than forced to match. Measured on this box:

    A, B   76.8% of the box feasible, d_SL  21-300 km
    C       3.4%                      d_SL  84-285 km
    D      74.7%                      d_SL  20-300 km

The midway archetype is the narrow one: placing both small clusters halfway
between the hubs ties d_SL to d_LL and d_SS, so only a thin slice of the box
realizes it. It is sampled, not excluded; its coverage is in topology_coverage.txt.

Geometry, coordinates and archetype definitions are imported unchanged from v4
(generate_node_topology_pooled.py), exactly as the v5 paired generator does.

Outputs (OUTPUT_FOLDER = generated_topology_screening/)
------------------------------------------------------
    NodeLocations_XXXX.csv      one per topology (XXXX = 0001 .. 3*N_ECON)
    design_manifest.csv         one row per economy: the 3 topology names
    topology_parameters.csv     one row per topology, targets + realized
    topology_nodes_long.csv     one row per (topology, small cluster)
    topology_coverage.txt       coverage per archetype and the d_SL contrast
    design_diagnostics.png      marginals and pairwise scatter per archetype
    methodology_note.txt        paper-ready description

The runner pairs economy i with the 3 topologies listed for it in
design_manifest.csv; sampling_design.build_run_table already reads whatever slot
columns the manifest has, so nothing changes there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

import generate_node_topology_pooled as v4
from generate_node_topology_pooled import (
    ARCHETYPE_NAME, NODES, SMALL_NODES, NEARNESS_RATIO, C_MIN,
    build_topology, positions_to_lonlat, write_node_locations, plot_topology,
)

# ------------------------------------------------------------------ config --

# The box. Deliberately NOT v4's D_*_RANGE constants: those are the v5 ranges and
# stay untouched so the v5 design remains reproducible from the same module.
BOX = {
    "d_LL": (150.0, 500.0),     # v5: (400, 1500)
    "d_SL": (20.0, 300.0),      # v5: (100, 500)
    "d_SS": (40.0, 300.0),      # v5: (100, 1000)
}

N_ECON = 216                     # economies; runs = 3 * N_ECON = 648
SLOT_ORDER = ["SL_low", "SL_mid", "SL_high"]
N_SLOTS = len(SLOT_ORDER)
# economies per archetype; must sum to N_ECON
ARCH_SHARE = {"A": 54, "B": 54, "C": 54, "D": 54}
# Pool label -> (label used for the feasibility test, archetypes drawn from it).
# A and B are mirror archetypes with the identical feasible region, so they share
# one pool and take disjoint slices of it. C is tested with the "C1" lean.
GROUPS = {"AB": ("A", ["A", "B"]), "C": ("C1", ["C"]), "D": ("D", ["D"])}
# C realizes only ~3.4% of the box, so it needs ~4800 candidates for its 162
# triples; the pool is sized for the worst archetype and the search stops early.
POOL_SIZE = 2 ** 15
SEED = 42
N_STRATA_REPORT = 24
PLOT_EVERY = 100                 # individual layout PNGs (0 = none)

OUTPUT_FOLDER = Path(__file__).resolve().parent / "generated_topology_screening"


def unit_to_params(u):
    """(d_LL, d_SL, d_SS) from a 3-D unit point, on THIS campaign's box."""
    return tuple(BOX[k][0] + u[i] * (BOX[k][1] - BOX[k][0])
                 for i, k in enumerate(("d_LL", "d_SL", "d_SS")))


def feasible_sobol(label, n_needed, seed):
    """First n_needed feasible points, in Sobol order, of a scrambled Sobol
    sequence over the box: a low-discrepancy sample UNIFORM over the archetype's
    feasible region. Nothing is clamped or rescaled."""
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
    return U[keep], len(keep) / (k + 1)


def tercile_triples(U):
    """Sort by d_SL, cut into three, pair by rank: the i-th smallest of the low
    third with the i-th of the middle and the i-th of the top third, so every
    economy sees a consistent low-to-high step across the archetype's range.
    Returns (n_triples, 3) of row indices into U."""
    order = np.argsort(U[:, 1])
    n = len(U) // N_SLOTS
    return np.column_stack([order[i * n:(i + 1) * n] for i in range(N_SLOTS)])


def build_design(seed=SEED):
    """group -> (U, triples, feasible share). One triple per economy of the group."""
    design = {}
    for g_i, (grp, (label, archs)) in enumerate(GROUPS.items()):
        n_econ = sum(ARCH_SHARE[a] for a in archs)
        U, frac = feasible_sobol(label, N_SLOTS * n_econ, seed + 100 * g_i)
        design[grp] = (U, tercile_triples(U), frac)
        print(f"  group {grp} (tested as {label}): {len(U)} points for {n_econ} "
              f"economies, feasible share of the box {frac * 100:.1f}%")
    return design


def assign_archetypes(rng):
    """One archetype per economy, balanced, in random order."""
    a = np.concatenate([np.repeat(k, v) for k, v in ARCH_SHARE.items()])
    if len(a) != N_ECON:
        raise RuntimeError(f"ARCH_SHARE sums to {len(a)}, expected N_ECON={N_ECON}")
    return a[rng.permutation(N_ECON)]


# ------------------------------------------------------------------ report --

def eta2(df, col):
    v = df[col].to_numpy()
    grand = v.mean()
    ss_tot = float(((v - grand) ** 2).sum())
    ss_b = float(sum(len(g) * (g[col].mean() - grand) ** 2
                     for _, g in df.groupby("archetype")))
    return ss_b / ss_tot if ss_tot > 0 else 0.0


def coverage_report(df, fracs, max_err):
    out = ["Screening design - d_SL opened downward, ranked in three within economy",
           "=" * 72,
           f"economies {N_ECON}, topologies {len(df)}, seed {SEED}",
           "box: " + ", ".join(f"{k} [{v[0]:.0f}, {v[1]:.0f}]" for k, v in BOX.items()),
           f"archetype shares {ARCH_SHARE}",
           f"centrality threshold = {C_MIN}, nearness ratio = {NEARNESS_RATIO}",
           f"max projection residual {max_err:.2f} km", ""]

    out += ["Feasible share of the box, per group", "-" * 72]
    for g, f in fracs.items():
        out.append(f"    {g}: {f * 100:5.1f}%")

    out += ["", "Realized d_SL per archetype and slot (km)", "-" * 72,
            f"    {'arch':<5}" + "".join(f"{s:>22}" for s in SLOT_ORDER)]
    for a, g in df.groupby("archetype"):
        row = f"    {a:<5}"
        for s in SLOT_ORDER:
            v = g[g.slot == s]["d_SL_mean"]
            row += f"{f'{v.min():.0f}-{v.max():.0f} (med {v.median():.0f})':>22}"
        out.append(row)

    out += ["", "Within-economy contrast (realized d_SL, km)", "-" * 72]
    for a, g in df.groupby("archetype"):
        piv = g.pivot_table(index="economy", columns="slot", values="d_SL_mean")
        d = (piv[SLOT_ORDER[-1]] - piv[SLOT_ORDER[0]]).dropna()
        out.append(f"    {a}: high - low  median {np.median(d):6.0f}  "
                   f"p10 {np.percentile(d, 10):6.0f}  p90 {np.percentile(d, 90):6.0f}")

    # The three slots of an economy are three independent triples, as near/far are
    # in v5, so d_LL and d_SS move with d_SL. Report by how much: the d_SL contrast
    # is only clean to the extent these stay flat. For the midway archetype they
    # cannot - its geometry ties d_SL to d_LL - and that has to be visible here
    # rather than discovered during the analysis.
    out += ["", "Co-movement of the other two distances over the same contrast (km)",
            "-" * 72,
            f"    {'arch':<5}{'d_SL high-low':>16}{'d_LL high-low':>28}{'d_SS high-low':>28}"]
    rng = np.random.default_rng(0)
    for a, g in df.groupby("archetype"):
        row = f"    {a:<5}"
        for col in ("d_SL_mean", "d_LL", "d_SS"):
            piv = g.pivot_table(index="economy", columns="slot", values=col)
            d = (piv[SLOT_ORDER[-1]] - piv[SLOT_ORDER[0]]).dropna().to_numpy()
            bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(1500)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sig = "*" if (lo > 0 or hi < 0) else " "
            row += (f"{d.mean():+8.1f}" if col == "d_SL_mean"
                    else f"{f'{d.mean():+7.1f} [{lo:+6.1f},{hi:+6.1f}]{sig}':>28}")
        out.append(row)
    out += ["    * = the co-moving distance shifts systematically with d_SL; the "
            "contrast is joint, not pure"]

    out += ["", "Marginal coverage of the sampled distances (all topologies)", "-" * 72]
    for col, key in (("d_LL", "d_LL"), ("d_SL_mean", "d_SL"), ("d_SS", "d_SS")):
        lo, hi = BOX[key]
        v = df[col]
        filled = len(np.unique(np.clip(((v - lo) / (hi - lo) * N_STRATA_REPORT).astype(int),
                                       0, N_STRATA_REPORT - 1)))
        out.append(f"    {col:10s}: {filled}/{N_STRATA_REPORT} strata filled, "
                   f"range {v.min():.0f}-{v.max():.0f} km")

    out += ["", "Archetype-distance confounding (eta^2 over topologies)", "-" * 72]
    for col in ("d_LL", "d_SL_mean", "d_SS"):
        e = eta2(df, col)
        out.append(f"    {col:10s}: {e:.3f}{'  (confounded)' if e > 0.25 else ''}")

    out += ["", f"RESULT: {len(df)} topologies, {df.economy.nunique()} economies "
                f"x {N_SLOTS} slots, {df.d_SL_mean.nunique()} distinct d_SL values"]
    return "\n".join(out)


def plot_diagnostics(df, folder):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    for ax, (col, key) in zip(axes[:3], (("d_LL", "d_LL"), ("d_SL_mean", "d_SL"),
                                         ("d_SS", "d_SS"))):
        for a, g in df.groupby("archetype"):
            ax.hist(g[col], bins=20, range=BOX[key], histtype="step", label=a, lw=1.5)
        ax.set_xlabel(f"{col} (km)"), ax.set_ylabel("topologies")
        ax.legend(fontsize=8), ax.grid(alpha=.3)
    for a, g in df.groupby("archetype"):
        axes[3].scatter(g.d_SL_mean, g.d_LL, s=6, alpha=.4, label=a)
    axes[3].set_xlabel("d_SL (km)"), axes[3].set_ylabel("d_LL (km)")
    axes[3].legend(fontsize=8), axes[3].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(folder / "design_diagnostics.png", dpi=130)
    plt.close(fig)


def methodology_note(df):
    per_arch = "; ".join(
        f"{a} {g.d_SL_mean.min():.0f}-{g.d_SL_mean.max():.0f} km"
        for a, g in df.groupby("archetype"))
    return (
        "Spatial topology design (screening campaign) - generated description\n\n"
        f"Each of the {N_ECON} economic samples is paired with {N_SLOTS} four-node "
        f"configurations of its own, drawn from the feasible region of the one "
        f"archetype assigned to that sample and ranked by the small-large distance, "
        f"so that the small-large distance enters as a within-economy contrast "
        f"(low / medium / high) with the economic parameters held constant.\n\n"
        f"Distances are sampled as a scrambled Sobol sequence over d_LL "
        f"[{BOX['d_LL'][0]:.0f}, {BOX['d_LL'][1]:.0f}] km, d_SL "
        f"[{BOX['d_SL'][0]:.0f}, {BOX['d_SL'][1]:.0f}] km and d_SS "
        f"[{BOX['d_SS'][0]:.0f}, {BOX['d_SS'][1]:.0f}] km, filtered for geometric "
        f"feasibility under each archetype and never clamped or rescaled; the sample "
        f"is therefore uniform over each archetype's feasible region and the realized "
        f"coverage per archetype is reported. Realized small-large distances: "
        f"{per_arch}.\n\n"
        f"The box is the one used for the main campaign, opened downward: the main "
        f"campaign sampled d_SL in [100, 500] km, which lies entirely above the "
        f"distance at which transporting hydrogen from the hub becomes cheaper than "
        f"producing it locally, so distance could not discriminate. The hub-hub and "
        f"small-small distances follow d_SL down out of geometric necessity: with both "
        f"small clusters attached to the same hub the triangle inequality forces "
        f"d_SS <= 2 d_SL, and the midway archetype requires a hub-hub distance of the "
        f"order of twice the small-large distance.\n\n"
        f"Node positions are the closed-form solution of the three distance constraints "
        f"under the archetype's symmetry convention, projected with an azimuthal-"
        f"equidistant inverse projection and refined so that the great-circle distances "
        f"used by the optimisation model reproduce the sampled distances.\n")


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"screening design: {N_ECON} economies x {N_SLOTS} = {N_SLOTS * N_ECON} topologies")
    print("  box: " + ", ".join(f"{k} {v}" for k, v in BOX.items()))
    design = build_design()
    fracs = {g: design[g][2] for g in design}

    rng = np.random.default_rng(SEED + 7)
    arch_of_econ = assign_archetypes(rng)

    # Hand each archetype a disjoint slice of its group's triples, so A and B
    # never sit on the same geometry. The triples come out of tercile_triples
    # ordered by rank, so they MUST be permuted first: slicing the ordered array
    # would give A the short half of every tercile and B the long half, which
    # would confound the "attach to the larger vs the smaller hub" contrast with
    # distance.
    triples_of_arch, taken = {}, {}
    for grp, (_, archs) in GROUPS.items():
        _, triples, _ = design[grp]
        triples = triples[rng.permutation(len(triples))]
        start = 0
        for a in archs:
            triples_of_arch[a] = triples[start:start + ARCH_SHARE[a]]
            taken[a] = 0
            start += ARCH_SHARE[a]

    wide, long, manifest, max_err = [], [], [], 0.0
    counter = 0
    for econ in range(1, N_ECON + 1):
        arch = arch_of_econ[econ - 1]
        grp = next(g for g, (_, archs) in GROUPS.items() if arch in archs)
        U = design[grp][0]
        rows = triples_of_arch[arch][taken[arch]]
        taken[arch] += 1
        # C is realized with the C1 lean; A, B, D are their own label
        label = "C1" if arch == "C" else arch

        row_m = {"economy": econ}
        for slot, row in zip(SLOT_ORDER, rows):
            d_ll, d_sl, d_ss = unit_to_params(U[row])
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
        if econ % 50 == 0:
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
