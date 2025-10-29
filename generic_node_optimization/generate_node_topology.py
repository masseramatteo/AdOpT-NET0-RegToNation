import pandas as pd
import numpy as np
from math import cos, sin, radians
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
import random

# === NOMI DEI NODI ===
big_nodes = ["BIG1", "BIG2"]  # 2 big nodes
small_nodes = ["SMALL1", "SMALL2", "SMALL3", "SMALL4"]  # 4 small nodes
# Possibility to have storage or not, and place it randomly
ENABLE_STORAGE = True  # set to False to disable storage node
STORAGE_NAME = "STORAGE"

# === PARAMETRI DI BASE ===
N_big = len(big_nodes)
N_small = len(small_nodes)

# origine geografica (per lon/lat fittizie)
lon0 = 4.4777
lat0 = 51.9244
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * cos(radians(lat0))

# === FUNZIONI DI SUPPORTO ===
def generate_line_cluster(n_nodes, spacing):
    total_length = (n_nodes - 1) * spacing
    start = - total_length / 2
    return np.array([[start + i * spacing, 0] for i in range(n_nodes)])

def generate_square_cluster(n_nodes, spacing):
    if n_nodes == 1:
        return np.array([[0,0]])
    elif n_nodes == 2:
        return np.array([[-spacing/2, 0], [spacing/2, 0]])
    else:
        half = spacing/2
        base = [
            [-half, -half],
            [-half,  half],
            [ half, -half],
            [ half,  half]
        ]
        return np.array(base[:n_nodes])

def rotate(points, theta_deg):
    theta = radians(theta_deg)
    R = np.array([[cos(theta), -sin(theta)],
                  [sin(theta),  cos(theta)]])
    return points @ R.T

def shift(points, dx, dy):
    return points + np.array([dx, dy])

def cartesian_to_lonlat(x_km, y_km):
    lon = lon0 + x_km / km_per_deg_lon
    lat = lat0 + y_km / km_per_deg_lat
    return lon, lat

def get_random_storage_position(positions_dict, min_distance=30):
    """Generate a random position for storage within the system bounds, ensuring minimum distance from other nodes"""
    all_positions = list(positions_dict.values())
    if not all_positions:
        return (0, 0)

    # Get bounds of existing positions
    x_coords = [pos[0] for pos in all_positions]
    y_coords = [pos[1] for pos in all_positions]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Add some margin
    margin = 50  # 50 km margin
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin

    # Try to find a position with minimum distance from all existing nodes
    max_attempts = 100
    for attempt in range(max_attempts):
        # Generate random position within bounds
        random_x = random.uniform(x_min, x_max)
        random_y = random.uniform(y_min, y_max)

        # Check distance from all existing nodes
        valid_position = True
        for existing_pos in all_positions:
            distance = np.sqrt((random_x - existing_pos[0])**2 + (random_y - existing_pos[1])**2)
            if distance < min_distance:
                valid_position = False
                break

        if valid_position:
            return (random_x, random_y)

    # If no valid position found after max attempts, place at a calculated offset
    # Find the center of all nodes and place storage at an offset
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)

    # Place storage at a 45-degree angle from center, at safe distance
    offset_distance = max(50, min_distance * 1.5)
    storage_x = center_x + offset_distance * np.cos(np.radians(45))
    storage_y = center_y + offset_distance * np.sin(np.radians(45))

    return (storage_x, storage_y)

def generate_scenario(D_big, D_small, D_between, theta_deg, scenario_name, output_folder):
    """
    Genera un file CSV per un singolo scenario con i parametri dati.
    """
    # cluster big
    big_positions = generate_line_cluster(N_big, D_big)

    # cluster small
    small_positions_local = generate_square_cluster(N_small, D_small)
    small_positions_rot = rotate(small_positions_local, theta_deg)
    small_positions = shift(small_positions_rot, D_between, 0)

    # costruisci dizionario
    positions_dict = {}
    for name, (x, y) in zip(big_nodes, big_positions):
        positions_dict[name] = (x, y)
    for name, (x, y) in zip(small_nodes, small_positions):
        positions_dict[name] = (x, y)

    # Add storage node randomly if enabled
    if ENABLE_STORAGE:
        # Always place storage at a separate position, never at existing nodes
        positions_dict[STORAGE_NAME] = get_random_storage_position(positions_dict)

    # converti in lon/lat
    node_location = []
    for node, (x_km, y_km) in positions_dict.items():
        lon, lat = cartesian_to_lonlat(x_km, y_km)
        node_location.append({"Node": node, "lon": lon, "lat": lat, "alt": 0})

    df = pd.DataFrame(node_location)
    df.to_csv(output_folder / f"NodeLocations_{scenario_name}.csv", sep=';', index=False)
    print(f"✅ Scenario '{scenario_name}' generato con D_big={D_big}, D_small={D_small}, D_between={D_between}, theta={theta_deg}")

def plot_nodes(positions_dict, scenario_name, output_folder):
    """
    Crea un grafico dei nodi: big nodes in blu grandi, small nodes in arancione piccoli.
    """
    plt.figure(figsize=(8, 7))
    # Big nodes
    big_x = [positions_dict[n][0] for n in big_nodes if n in positions_dict]
    big_y = [positions_dict[n][1] for n in big_nodes if n in positions_dict]
    plt.scatter(big_x, big_y, c='blue', s=300, label='Big nodes', zorder=2)

    # Small nodes
    small_x = [positions_dict[n][0] for n in small_nodes if n in positions_dict]
    small_y = [positions_dict[n][1] for n in small_nodes if n in positions_dict]
    plt.scatter(small_x, small_y, c='orange', s=120, label='Small nodes', zorder=2)

    # Storage node
    if STORAGE_NAME in positions_dict:
        storage_x, storage_y = positions_dict[STORAGE_NAME]
        plt.scatter(storage_x, storage_y, c='green', s=200, marker='s', label='Storage', zorder=2)

    # Etichette
    for n, (x, y) in positions_dict.items():
        plt.text(x, y+2, n, fontsize=10, ha='center', va='bottom')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f'Topologia nodi - Scenario {scenario_name}')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_folder / f"topology_{scenario_name}.png")
    plt.close()

# === LISTE DI PARAMETRI PER COMBINAZIONI ===
D_big_values = [200, 500, 1000]         # varia la distanza interna big
D_small_values = [50, 100, 1000, 2000]         # varia la distanza interna small
D_between_values = [30, 60, 120, 200, 500]      # varia distanza fra cluster
theta_big_values = [0, 45, 90]        # varia rotazione cluster big
theta_small_values = [0, 45, 90]      # varia rotazione cluster small
storage_values = [True, False]         # varia presenza storage

# === CARTELLA OUTPUT ===
output_folder = Path(r"\\soliscom.uu.nl\geo\SD\Energy and Resources\GazzaniGroup\Matteo M\FirstWork simulations\x_nodes_generation\input_data\scenarios")
output_folder.mkdir(parents=True, exist_ok=True)

# === LOOP SU TUTTE LE COMBINAZIONI ===
scenario_id = 1
for D_big, D_small, D_between, theta_big, theta_small, storage_enabled in product(D_big_values, D_small_values, D_between_values, theta_big_values, theta_small_values, storage_values):
    scenario_name = f"{scenario_id:03d}"

    # cluster big (with rotation)
    big_positions_local = generate_line_cluster(N_big, D_big)
    big_positions = rotate(big_positions_local, theta_big)

    # cluster small (with rotation e shift)
    small_positions_local = generate_square_cluster(N_small, D_small)
    small_positions_rot = rotate(small_positions_local, theta_small)
    small_positions = shift(small_positions_rot, D_between, 0)

    # costruisci dizionario posizioni
    positions_dict = {}
    for name, (x, y) in zip(big_nodes, big_positions):
        positions_dict[name] = (x, y)
    for name, (x, y) in zip(small_nodes, small_positions):
        positions_dict[name] = (x, y)

    # Add storage node if enabled for this scenario
    if ENABLE_STORAGE and storage_enabled:
        # Always place storage at a separate position, never at existing nodes
        positions_dict[STORAGE_NAME] = get_random_storage_position(positions_dict)

    # Genera scenario CSV (temporarily modify ENABLE_STORAGE for this scenario)
    original_enable_storage = ENABLE_STORAGE
    ENABLE_STORAGE = storage_enabled
    generate_scenario(D_big, D_small, D_between, theta_small, scenario_name, output_folder)
    ENABLE_STORAGE = original_enable_storage

    # Ogni 25 scenari, crea il plot
    if (scenario_id - 1) % 25 == 0:
        plot_nodes(positions_dict, scenario_name, output_folder)

    scenario_id += 1

print(f"\n📁 Generati {scenario_id - 1} scenari in '{output_folder}'")
print(f"Combinazioni: D_big={len(D_big_values)}, D_small={len(D_small_values)}, D_between={len(D_between_values)}, theta_big={len(theta_big_values)}, theta_small={len(theta_small_values)}, storage={len(storage_values)}")
print(f"Totale combinazioni: {len(D_big_values) * len(D_small_values) * len(D_between_values) * len(theta_big_values) * len(theta_small_values) * len(storage_values)}")
