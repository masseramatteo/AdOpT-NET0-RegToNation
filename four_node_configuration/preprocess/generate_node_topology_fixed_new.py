"""
Four-node topology generator (v2).

Three parameters, all REALIZED distances rather than generator inputs:

    d_LL  distance between the two large clusters
    d_SL  distance from each small cluster to its assigned large cluster
    d_SS  distance between the two small clusters

Geometry
--------
L1 = (0,0), L2 = (d_LL, 0). Positions are SOLVED so that the three distances
hold exactly. Two small clusters in the plane have 4 degrees of freedom;
fixing d_SL for both and d_SS between them consumes 3, and the remaining
orientation is pinned by the attachment. No free angle, no nuisance draw.

Attachment (both->L1, both->L2, split) is a SAMPLING STRUCTURE that guarantees
coverage of the qualitatively distinct arrangements. It is not intended as an
analysis feature: the same LHS draws are reused across all three attachments,
so d_SL coverage is identical between them and attachment can be dropped from
the decision trees without confounding.

What changed relative to v1
---------------------------
1. Positions solved from the three distances, instead of assembled by a recipe.
   The v1 code multiplied d_SL by 0.3 and clamped it at 0.4*d_LL, so asking for
   400 km produced 192 km. It also discarded d_SL entirely in archetype C and
   d_SS entirely in archetype D.
2. Archetype C removed. Equidistance is the limiting case d_SL ~ d_LL/2, reached
   continuously; keeping it as a category confounded it with the distance range
   and left 0% of A/B/D observations above the 515 km threshold.
3. Infeasible parameter triples are MAPPED onto the feasible interval rather
   than rejected, so LHS stratification survives and no draw is discarded.
   Consequence: d_SS is uniform CONDITIONAL on d_LL and d_SL, not marginally
   uniform. This is geometrically unavoidable and must be stated in the paper.
4. d_SS floored at 50 km. v1 allowed separations down to ~1 km, which is not two
   clusters. This regime drove the small-small pipeline result (DT5).
5. d_LL floor lowered 400 -> 150 km. EU industrial cluster data (sEEnergies D5.1,
   three sectors, 50 km linkage) puts adjacent large clusters at a median
   nearest-neighbour distance of ~177 km; a 400 km floor excludes the
   Ruhr-Antwerp-Rotterdam spacing entirely.
6. Guard added for the split attachment (v1 had none): d_SL > d_LL/2 made the
   small clusters swap sides or fall outside the segment in ~23% of draws.
7. Centred-discrepancy optimised LHS instead of plain LHS.
8. One shared LHS design across attachments -> attachment is a paired contrast.
9. Replicate labelling on the topology design, so prevalence statistics can be
   reported with an internal accuracy estimate (Janssen 2013, replicated LHS).

Known limitation (state it in the methodology)
----------------------------------------------
Four nodes in a plane cannot have all three distances simultaneously
independent of the attachment. This design protects d_SL, because the
electrolyzer-siting threshold depends on it. d_SS coverage does differ between
the both-near attachments and split; d_SS-driven results should therefore be
reported either restricted to the both-near pair, or with per-attachment
coverage stated.

Outputs
-------
    NodeLocations_XXXX.csv    one per topology, unchanged format
    topology_parameters.csv   wide, one row per topology
    topology_nodes_long.csv   one row per (topology, small cluster)
    topology_coverage.txt     realized ranges per attachment, for the paper
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

from preprocess_utilities import cartesian_to_lonlat

# ----------------------------------------------------------------- config --
big_nodes = ["Large_cluster1", "Large_cluster2"]      # Large1 is the bigger one
small_nodes = ["Small_cluster1", "Small_cluster2"]

lon0, lat0 = 4.4777, 51.9244
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * math.cos(math.radians(lat0))

D_LL_RANGE = (150.0, 1500.0)     # large-large [km]   (was 400-1500)
D_SL_RANGE = (50.0, 900.0)       # small to its own large [km]   (was 100-500)
D_SS_FLOOR = 50.0                # two clusters closer than this are one cluster

ATTACHMENTS = ["both_L1", "both_L2", "split"]
N_PER_ATTACHMENT = 30
N_REPLICATES = 5                 # must divide N_PER_ATTACHMENT
OVERSAMPLE = 3                   # draw pool multiplier (split rejects ~45%)
SEED = 42
PLOT_EVERY = 10

OUTPUT_FOLDER = Path("generated_topology_v2")


# -------------------------------------------------------------- geometry --
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def solve_both_near(d_ll, d_sl, u_ss, anchor_left):
    """Both small clusters at distance d_sl from the same large cluster,
    separated by d_ss, placed symmetrically about the L1-L2 axis:

        S = (x, +-d_ss/2),  x = sqrt(d_sl^2 - (d_ss/2)^2)

    Feasibility: d_ss <= 2*d_sl (so x is real) and x < d_ll/2 (so the small
    clusters really are nearer their own anchor). The second condition puts a
    LOWER bound on d_ss when d_sl > d_ll/2.
    """
    lo = max(2.0 * math.sqrt(max(0.0, d_sl ** 2 - (d_ll / 2.0) ** 2)), D_SS_FLOOR)
    hi = 2.0 * d_sl
    if hi <= lo:
        return None
    d_ss = lo + u_ss * (hi - lo)
    x = math.sqrt(max(0.0, d_sl ** 2 - (d_ss / 2.0) ** 2))
    xx = x if anchor_left else d_ll - x
    return (xx, d_ss / 2.0), (xx, -d_ss / 2.0), d_ss


def solve_split(d_ll, d_sl, u_ss):
    """S1 at distance d_sl from L1, S2 at distance d_sl from L2, separated by
    d_ss. S1 is placed on its circle at angle theta; S2 is then the intersection
    of the circle about L2 with the circle of radius d_ss about S1. Theta is
    scanned rather than sampled, so the configuration is fully determined by the
    three distances.

    Feasibility: |d_LL - 2*d_SL| <= d_SS <= d_LL + 2*d_SL, plus the guard that
    S1 stays left of the midline and S2 right of it.
    """
    L2 = (d_ll, 0.0)
    lo = max(abs(d_ll - 2.0 * d_sl), D_SS_FLOOR)
    hi = min(d_ll + 2.0 * d_sl, 1.6 * d_ll)
    if hi <= lo:
        return None
    d_ss = lo + u_ss * (hi - lo)

    for deg in range(0, 180):
        th = math.radians(deg)
        s1 = (d_sl * math.cos(th), d_sl * math.sin(th))
        dc = dist(s1, L2)
        if not (abs(d_sl - d_ss) <= dc <= d_sl + d_ss):
            continue
        a = (d_sl ** 2 - d_ss ** 2 + dc ** 2) / (2.0 * dc)
        h = math.sqrt(max(0.0, d_sl ** 2 - a ** 2))
        ux, uy = (s1[0] - L2[0]) / dc, (s1[1] - L2[1]) / dc
        px, py = L2[0] + a * ux, L2[1] + a * uy
        s2 = (px - h * uy, py + h * ux)
        if abs(dist(s1, s2) - d_ss) > 1.0:
            continue
        if s1[0] < d_ll / 2.0 < s2[0]:          # guard: v1 had none
            return s1, s2, d_ss
    return None


def build_topology(attachment, d_ll, d_sl, u_ss):
    if attachment == "both_L1":
        r = solve_both_near(d_ll, d_sl, u_ss, anchor_left=True)
    elif attachment == "both_L2":
        r = solve_both_near(d_ll, d_sl, u_ss, anchor_left=False)
    elif attachment == "split":
        r = solve_split(d_ll, d_sl, u_ss)
    else:
        raise ValueError(f"Unknown attachment: {attachment}")
    if r is None:
        return None
    s1, s2, d_ss = r
    pos = {big_nodes[0]: (0.0, 0.0), big_nodes[1]: (d_ll, 0.0),
           small_nodes[0]: s1, small_nodes[1]: s2}
    return pos, d_ss


def realized_distances(pos):
    L1, L2 = pos[big_nodes[0]], pos[big_nodes[1]]
    s1, s2 = pos[small_nodes[0]], pos[small_nodes[1]]
    d = {"d_LL": dist(L1, L2), "d_SS": dist(s1, s2),
         "S1_L1": dist(s1, L1), "S1_L2": dist(s1, L2),
         "S2_L1": dist(s2, L1), "S2_L2": dist(s2, L2)}
    d["S1_nearest_large"] = min(d["S1_L1"], d["S1_L2"])
    d["S2_nearest_large"] = min(d["S2_L1"], d["S2_L2"])
    # Compactness: separation normalised by its geometric maximum. Raw d_SS is
    # mechanically tied to d_SL (two clusters both at d_SL from one hub cannot
    # be more than 2*d_SL apart), giving r ~ +0.70. The ratio removes most of
    # that -- prefer it as the tree feature.
    d["rho_comp"] = d["d_SS"] / (2.0 * min(d["S1_nearest_large"],
                                           d["S2_nearest_large"]))
    return d


# ------------------------------------------------------------------- i/o --
def write_node_locations(pos, name, folder):
    rows = []
    for node, (x_km, y_km) in pos.items():
        lon, lat = cartesian_to_lonlat(x_km, y_km, lon0, lat0,
                                       km_per_deg_lon, km_per_deg_lat)
        rows.append({"Node": node, "lon": lon, "lat": lat, "alt": 0})
    pd.DataFrame(rows).to_csv(folder / f"NodeLocations_{name}.csv",
                              sep=";", index=False)


def plot_topology(pos, attachment, name, folder):
    plt.figure(figsize=(8, 5.5))
    lx = [pos[n][0] for n in big_nodes]
    ly = [pos[n][1] for n in big_nodes]
    sx = [pos[n][0] for n in small_nodes]
    sy = [pos[n][1] for n in small_nodes]
    plt.plot(lx, ly, color="#c9ccd6", lw=1, zorder=1)
    plt.scatter(lx, ly, s=420, c="#161D41", marker="o", zorder=3,
                edgecolors="white", linewidths=1.5, label="Large clusters")
    plt.scatter(sx, sy, s=180, c="#FFCD00", marker="s", zorder=3,
                edgecolors="#161D41", linewidths=1.2, label="Small clusters")
    for n, (x, y) in pos.items():
        plt.text(x, y + 28, n, fontsize=8, ha="center", fontweight="bold")
    plt.grid(True, ls="--", alpha=.3)
    plt.title(f"{attachment} — topology {name}")
    plt.xlabel("X [km]"); plt.ylabel("Y [km]")
    plt.legend(); plt.axis("equal"); plt.tight_layout()
    plt.savefig(folder / f"topology_{name}.png", dpi=140)
    plt.close()


# ------------------------------------------------------------------ main --
def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    if N_PER_ATTACHMENT % N_REPLICATES:
        raise ValueError("N_REPLICATES must divide N_PER_ATTACHMENT")
    per_rep = N_PER_ATTACHMENT // N_REPLICATES

    # ONE centred-discrepancy optimised design, reused across attachments,
    # built as N_REPLICATES independent blocks so prevalence statistics can be
    # reported as mean +/- spread across replicates.
    # Split rejects ~45% of draws on feasibility, so draw an OVERSAMPLE and keep
    # the first N_PER_ATTACHMENT feasible ones. both_L1/both_L2 never reject, so
    # they use the first block untouched and remain exactly paired with each
    # other. Split is therefore paired with them only on its feasible subset --
    # state this in the methodology.
    blocks = [qmc.LatinHypercube(d=3, optimization="random-cd",
                                 seed=SEED + r).random(per_rep * OVERSAMPLE)
              for r in range(N_REPLICATES)]
    U = np.vstack(blocks)
    rep_id = np.repeat(np.arange(N_REPLICATES), per_rep * OVERSAMPLE)

    wide, long, skipped = [], [], {a: 0 for a in ATTACHMENTS}
    counter = 1

    for attachment in ATTACHMENTS:
        print(f"\n--- {attachment} ---")
        kept = 0
        for k in range(len(U)):
            if kept >= N_PER_ATTACHMENT:
                break
            u_ll, u_sl, u_ss = U[k]
            d_ll = D_LL_RANGE[0] + u_ll * (D_LL_RANGE[1] - D_LL_RANGE[0])
            d_sl = D_SL_RANGE[0] + u_sl * (D_SL_RANGE[1] - D_SL_RANGE[0])

            built = build_topology(attachment, d_ll, d_sl, u_ss)
            if built is None:
                skipped[attachment] += 1
                continue
            pos, d_ss = built
            dd = realized_distances(pos)
            name = f"{counter:04d}"

            # attachment must hold by construction — assert, don't warn
            if attachment == "both_L1":
                assert dd["S1_L1"] < dd["S1_L2"] and dd["S2_L1"] < dd["S2_L2"]
            elif attachment == "both_L2":
                assert dd["S1_L2"] < dd["S1_L1"] and dd["S2_L2"] < dd["S2_L1"]
            else:
                assert dd["S1_L1"] < dd["S1_L2"] and dd["S2_L2"] < dd["S2_L1"]

            # requested distances must be realized
            assert abs(dd["S1_nearest_large"] - d_sl) < 1.0
            assert abs(dd["S2_nearest_large"] - d_sl) < 1.0
            assert abs(dd["d_SS"] - d_ss) < 1.0

            write_node_locations(pos, name, OUTPUT_FOLDER)
            if (counter - 1) % PLOT_EVERY == 0:
                plot_topology(pos, attachment, name, OUTPUT_FOLDER)

            row = {"topology": name, "attachment": attachment,
                   "draw": k, "replicate": int(rep_id[k]), **dd}
            for node in big_nodes + small_nodes:
                row[f"{node}_x"], row[f"{node}_y"] = pos[node]
            wide.append(row)

            for i, sn in enumerate(small_nodes, start=1):
                long.append({
                    "topology": name, "attachment": attachment,
                    "replicate": int(rep_id[k]), "small_cluster": sn,
                    "d_nearest_large": dd[f"S{i}_nearest_large"],
                    "d_other_large": max(dd[f"S{i}_L1"], dd[f"S{i}_L2"]),
                    "d_small_small": dd["d_SS"],
                    "rho_comp": dd["rho_comp"],
                    "d_large_large": dd["d_LL"],
                })

            print(f"  {name}: LL={d_ll:6.0f}  SL={d_sl:6.0f}  SS={d_ss:6.0f}")
            counter += 1
            kept += 1

    dfw = pd.DataFrame(wide)
    dfl = pd.DataFrame(long)
    dfw.to_csv(OUTPUT_FOLDER / "topology_parameters.csv", sep=";", index=False)
    dfl.to_csv(OUTPUT_FOLDER / "topology_nodes_long.csv", sep=";", index=False)

    # ---- coverage report, for the methodology section --------------------
    lines = ["Realized coverage per attachment", "=" * 64]
    for att in ATTACHMENTS:
        m = dfl.attachment == att
        w = dfw.attachment == att
        lines += [
            f"\n{att}  ({w.sum()} topologies, {skipped[att]} draws infeasible, "
            f"{w.sum()/(w.sum()+skipped[att])*100:.0f}% of attempted draws feasible)",
            f"  d_SL  {dfl.d_nearest_large[m].min():6.0f} - {dfl.d_nearest_large[m].max():6.0f} km"
            f"   share > 515 km: {(dfl.d_nearest_large[m] > 515).mean()*100:5.1f}%",
            f"  d_SS  {dfw.d_SS[w].min():6.0f} - {dfw.d_SS[w].max():6.0f} km",
            f"  d_LL  {dfw.d_LL[w].min():6.0f} - {dfw.d_LL[w].max():6.0f} km",
        ]
    corr = dfl[["d_nearest_large", "d_small_small", "rho_comp",
                "d_large_large"]].corr()
    lines += ["\nFeature correlation (node-level long format)", corr.round(3).to_string()]
    report = "\n".join(lines)
    (OUTPUT_FOLDER / "topology_coverage.txt").write_text(report)
    print("\n" + report)
    print(f"\nGenerated {counter - 1} topologies in '{OUTPUT_FOLDER.resolve()}'")


if __name__ == "__main__":
    main()