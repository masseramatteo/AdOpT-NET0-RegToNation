import pandas as pd
import numpy as np
from math import cos, radians
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt

from preprocess_utilities import (
    generate_line_cluster,
    cartesian_to_lonlat,
    polar_to_cartesian,
    euclidean_distance,
    map_to_nearest,
    map_to_discrete_index,
    latin_hypercube
)

# === name of nodes ===
big_nodes = ["Large_cluster1", "Large_cluster2"]       # 2 big nodes
small_nodes = ["Small_cluster1", "Small_cluster2"]     # 2 small nodes

# === PARAMETRI DI BASE ===
N_big = len(big_nodes)
N_small = len(small_nodes)

# origine geografica (per lon/lat fittizie)
lon0 = 4.4777
lat0 = 51.9244

# conversione km <-> gradi (approssimata)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * cos(radians(lat0))


def is_valid_configuration(big_positions, small_positions, D_min):
    """
    True se:
      - ogni Small è almeno D_min da ciascun Large
      - Small1 e Small2 sono almeno D_min tra loro
    """
    # Small-Large
    for s in small_positions:
        for b in big_positions:
            if euclidean_distance(s, b) < D_min:
                return False

    # Small-Small
    if euclidean_distance(small_positions[0], small_positions[1]) < D_min:
        return False

    return True


def generate_scenario(
    D_big,
    D_min,
    s1_r, s1_theta_deg,
    s2_r, s2_theta_deg,
    scenario_name,
    output_folder
):
    """
    Genera un file CSV per un singolo scenario (se valido) con:
      - 2 Large distanti D_big
      - 2 Small su circonferenze centrate nel midpoint dei Large
    """

    # Large fissi sulla retta che li unisce (asse x)
    big_positions = generate_line_cluster(N_big, D_big)

    # Small in polari rispetto al centro (0,0)
    s1 = polar_to_cartesian(s1_r, s1_theta_deg)
    s2 = polar_to_cartesian(s2_r, s2_theta_deg)
    small_positions = [s1, s2]

    if not is_valid_configuration(big_positions, small_positions, D_min):
        return None

    positions_dict = {}
    for name, (x, y) in zip(big_nodes, big_positions):
        positions_dict[name] = (x, y)
    for name, (x, y) in zip(small_nodes, small_positions):
        positions_dict[name] = (x, y)

    node_location = []
    for node, (x_km, y_km) in positions_dict.items():
        lon, lat = cartesian_to_lonlat(x_km, y_km, lon0, lat0, km_per_deg_lon, km_per_deg_lat)
        node_location.append({"Node": node, "lon": lon, "lat": lat, "alt": 0})

    df = pd.DataFrame(node_location)
    df.to_csv(output_folder / f"NodeLocations_{scenario_name}.csv", sep=";", index=False)
    return positions_dict



def plot_nodes(positions_dict, scenario_name, output_folder):
    """
    Crea un grafico dei nodi: big nodes in blu grandi, small nodes in arancione piccoli.
    """
    plt.figure(figsize=(8, 7))

    # Big nodes
    big_x = [positions_dict[n][0] for n in big_nodes if n in positions_dict]
    big_y = [positions_dict[n][1] for n in big_nodes if n in positions_dict]
    plt.scatter(big_x, big_y, s=300, label="Big nodes", zorder=2)

    # Small nodes
    small_x = [positions_dict[n][0] for n in small_nodes if n in positions_dict]
    small_y = [positions_dict[n][1] for n in small_nodes if n in positions_dict]
    plt.scatter(small_x, small_y, s=120, label="Small nodes", zorder=2)

    # Etichette
    for n, (x, y) in positions_dict.items():
        plt.text(x, y + 2, n, fontsize=10, ha="center", va="bottom")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title(f"Topologia nodi - Scenario {scenario_name}")
    plt.xlabel("X [km]")
    plt.ylabel("Y [km]")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_folder / f"topology_{scenario_name}.png")
    plt.close()


# =========================
# PARAMETRI DISCRETI
# =========================

# Distanza tra i 2 Large
D_big_values = [500, 1000, 1500]

# Distanza minima (Small-Large e Small-Small)
D_min_values = [50]

# Radii discreti di campionamento (li filtriamo con d_small_max)
r_candidates = [50, 100, 150, 200, 300, 400, 500]

# Angoli discreti (gradi)
theta_small_values = [0, 45, 90, 135, 180, 225, 270, 315]

# === CARTELLA OUTPUT ===
output_folder = Path("generated_topology")
output_folder.mkdir(parents=True, exist_ok=True)

N_samples_per_setting = 100     # <-- quanti scenari vuoi per ogni (D_big, D_min)
oversample_factor = 6           # <-- genera di più per compensare quelli scartati dai vincoli
seed = 42

r_max = max(r_candidates)

scenario_id = 1
kept = 0
skipped = 0

for D_big in D_big_values:
    for D_min in D_min_values:

        # LHS in 4D: (u_r1, u_th1, u_r2, u_th2)
        n_try = N_samples_per_setting * oversample_factor
        H = latin_hypercube(n_try, 4, seed=seed + int(D_big*10) + int(D_min))

        seen = set()  # per evitare duplicati dopo mapping discreto

        count_setting = 0
        for u_r1, u_th1, u_r2, u_th2 in H:
            # mappa su discreto
            r1_cont = u_r1 * r_max
            r2_cont = u_r2 * r_max
            r1 = map_to_nearest(r_candidates, r1_cont)
            r2 = map_to_nearest(r_candidates, r2_cont)

            th1 = map_to_discrete_index(theta_small_values, u_th1)
            th2 = map_to_discrete_index(theta_small_values, u_th2)

            # opzionale: considera Small1/Small2 indistinguibili -> canonicalizza
            # key = tuple(sorted([(r1, th1), (r2, th2)]))
            # se invece sono distinti, usa:
            key = tuple(sorted([(r1, th1), (r2, th2)]))

            if key in seen:
                continue
            seen.add(key)

            scenario_name = f"{scenario_id:06d}"

            positions_dict = generate_scenario(
                D_big=D_big,
                D_min=D_min,
                s1_r=r1, s1_theta_deg=th1,
                s2_r=r2, s2_theta_deg=th2,
                scenario_name=scenario_name,
                output_folder=output_folder
            )

            if positions_dict is None:
                skipped += 1
            else:
                kept += 1
                count_setting += 1

                if kept % 10 == 0:
                    plot_nodes(positions_dict, scenario_name, output_folder)

                if count_setting >= N_samples_per_setting:
                    break

            scenario_id += 1

        print(f"(D_big={D_big}, D_min={D_min}) -> salvati {count_setting}/{N_samples_per_setting}")

print(f"\n📁 Output in: '{output_folder.resolve()}'")
print(f"Validi salvati: {kept}")
print(f"Scartati (vincoli): {skipped}")
