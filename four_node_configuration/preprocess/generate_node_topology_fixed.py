import pandas as pd
from math import cos, radians
from pathlib import Path
import matplotlib.pyplot as plt

from preprocess_utilities import (
    cartesian_to_lonlat,
    latin_hypercube
)

# === name of nodes ===
# Large1 is the bigger one
big_nodes = ["Large_cluster1", "Large_cluster2"]       # 2 big nodes (Large1 is bigger)
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

# === CONFIGURATION TYPES ===
# Type A: Both Small near Large1
# Type B: Both Small near Large2
# Type C: Small clusters have similar distance to both Large nodes
# Type D: One Small near Large1, one Small near Large2


def generate_configuration_type_A(large1_pos, large2_pos, d_small_large, d_small_small):
    """
    Type A: Both Small clusters near Large1
    Small clusters positioned to be closer to Large1 than Large2
    Uses angle from Large1 that ensures they stay on Large1's side
    """
    import math

    # Calculate the distance between the two large nodes
    d_large = math.sqrt((large2_pos[0] - large1_pos[0])**2 + (large2_pos[1] - large1_pos[1])**2)

    # To ensure Small nodes are closer to Large1, position them:
    # - At distance d_small_large from Large1
    # - In a direction perpendicular to the Large1-Large2 line
    # - Making sure d_small_large < d_large/2 (otherwise they'll be closer to Large2)

    # If d_small_large is too large, clamp it
    max_safe_distance = d_large * 0.4  # Stay within 40% of distance to ensure closeness
    d_sl_safe = min(d_small_large, max_safe_distance)

    # Position small nodes perpendicular to the x-axis (since Large nodes are on x-axis)
    # at distance d_sl_safe from Large1
    small1_pos = (large1_pos[0] + d_sl_safe * 0.3, large1_pos[1] + d_small_small/2)
    small2_pos = (large1_pos[0] + d_sl_safe * 0.3, large1_pos[1] - d_small_small/2)

    return [small1_pos, small2_pos]


def generate_configuration_type_B(large1_pos, large2_pos, d_small_large, d_small_small):
    """
    Type B: Both Small clusters near Large2
    Small clusters positioned to be closer to Large2 than Large1
    """
    import math

    # Calculate the distance between the two large nodes
    d_large = math.sqrt((large2_pos[0] - large1_pos[0])**2 + (large2_pos[1] - large1_pos[1])**2)

    # Similar to Type A, but for Large2
    max_safe_distance = d_large * 0.4
    d_sl_safe = min(d_small_large, max_safe_distance)

    # Position small nodes near Large2
    small1_pos = (large2_pos[0] - d_sl_safe * 0.3, large2_pos[1] + d_small_small/2)
    small2_pos = (large2_pos[0] - d_sl_safe * 0.3, large2_pos[1] - d_small_small/2)

    return [small1_pos, small2_pos]


def generate_configuration_type_C(large1_pos, large2_pos, d_small_large, d_small_small):
    """
    Type C: Small clusters have similar distance to both Large nodes
    Positioned at the midpoint between Large nodes, offset perpendicular
    """
    midpoint_x = (large1_pos[0] + large2_pos[0]) / 2
    midpoint_y = (large1_pos[1] + large2_pos[1]) / 2

    small1_pos = (midpoint_x, midpoint_y + d_small_small/2)
    small2_pos = (midpoint_x, midpoint_y - d_small_small/2)

    return [small1_pos, small2_pos]


def generate_configuration_type_D(large1_pos, large2_pos, d_small_large, d_small_small):
    """
    Type D: One Small near Large1, one Small near Large2
    """
    small1_pos = (large1_pos[0] + d_small_large, large1_pos[1])
    small2_pos = (large2_pos[0] - d_small_large, large2_pos[1])

    return [small1_pos, small2_pos]


def generate_scenario(
    config_type,
    d_large_large,
    d_small_large,
    d_small_small,
    scenario_name,
    output_folder
):
    """
    Generate a scenario with fixed configuration type.

    Args:
        config_type: 'A', 'B', 'C', or 'D'
        d_large_large: Distance between Large1 and Large2
        d_small_large: Average distance between Small and nearest Large
        d_small_small: Distance between Small1 and Small2
        scenario_name: Name for the output file
        output_folder: Where to save the CSV

    Returns:
        positions_dict or None if invalid
    """
    import math

    # Position Large nodes on x-axis
    # Large1 at origin, Large2 at (d_large_large, 0)
    large1_pos = (0, 0)
    large2_pos = (d_large_large, 0)

    # Generate Small positions based on configuration type
    if config_type == 'A':
        small_positions = generate_configuration_type_A(large1_pos, large2_pos, d_small_large, d_small_small)
    elif config_type == 'B':
        small_positions = generate_configuration_type_B(large1_pos, large2_pos, d_small_large, d_small_small)
    elif config_type == 'C':
        small_positions = generate_configuration_type_C(large1_pos, large2_pos, d_small_large, d_small_small)
    elif config_type == 'D':
        small_positions = generate_configuration_type_D(large1_pos, large2_pos, d_small_large, d_small_small)
    else:
        raise ValueError(f"Unknown configuration type: {config_type}")

    # Build positions dictionary
    positions_dict = {
        big_nodes[0]: large1_pos,
        big_nodes[1]: large2_pos,
        small_nodes[0]: small_positions[0],
        small_nodes[1]: small_positions[1]
    }

    # Verify distances for types A and B
    if config_type in ['A', 'B']:
        for i, small_node in enumerate(small_nodes):
            small_pos = small_positions[i]

            # Calculate distances to both large nodes
            dist_to_large1 = math.sqrt((small_pos[0] - large1_pos[0])**2 + (small_pos[1] - large1_pos[1])**2)
            dist_to_large2 = math.sqrt((small_pos[0] - large2_pos[0])**2 + (small_pos[1] - large2_pos[1])**2)

            # Verify the configuration is correct
            if config_type == 'A' and dist_to_large1 > dist_to_large2:
                print(f"    ⚠️  WARNING: {small_node} is closer to Large2 ({dist_to_large2:.1f}km) than Large1 ({dist_to_large1:.1f}km) in Type A!")
            elif config_type == 'B' and dist_to_large2 > dist_to_large1:
                print(f"    ⚠️  WARNING: {small_node} is closer to Large1 ({dist_to_large1:.1f}km) than Large2 ({dist_to_large2:.1f}km) in Type B!")

    # Create CSV
    node_location = []
    for node, (x_km, y_km) in positions_dict.items():
        lon, lat = cartesian_to_lonlat(x_km, y_km, lon0, lat0, km_per_deg_lon, km_per_deg_lat)
        node_location.append({"Node": node, "lon": lon, "lat": lat, "alt": 0})

    df = pd.DataFrame(node_location)
    df.to_csv(output_folder / f"NodeLocations_{scenario_name}.csv", sep=";", index=False)

    return positions_dict



def plot_nodes(positions_dict, config_type, scenario_name, output_folder):
    """
    Create a plot of the nodes configuration.
    """
    plt.figure(figsize=(10, 8))

    # Large nodes
    large_x = [positions_dict[n][0] for n in big_nodes]
    large_y = [positions_dict[n][1] for n in big_nodes]
    plt.scatter(large_x, large_y, s=500, c='blue', marker='o', label="Large nodes",
                zorder=3, edgecolors='black', linewidths=2)

    # Small nodes
    small_x = [positions_dict[n][0] for n in small_nodes]
    small_y = [positions_dict[n][1] for n in small_nodes]
    plt.scatter(small_x, small_y, s=200, c='orange', marker='s', label="Small nodes",
                zorder=3, edgecolors='black', linewidths=2)

    # Labels
    for n, (x, y) in positions_dict.items():
        plt.text(x, y + 20, n, fontsize=11, ha="center", va="bottom", fontweight='bold')

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.title(f"Configuration Type {config_type} - Scenario {scenario_name}", fontsize=14, fontweight='bold')
    plt.xlabel("X [km]", fontsize=12)
    plt.ylabel("Y [km]", fontsize=12)
    plt.legend(fontsize=11)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_folder / f"topology_{scenario_name}.png", dpi=150)
    plt.close()


# =========================
# PARAMETER RANGES
# =========================

# Configuration types to generate
config_types = ['A', 'B', 'C', 'D']

# Parameter ranges (min, max) for Latin Hypercube Sampling
# Distance between Large1 and Large2 [km]
d_large_large_range = (400, 1500)

# Distance between Small clusters [km]
d_small_small_range = (100, 1000)

# Average distance from Small to nearest Large [km]
d_small_large_range = (100, 500)

# Number of samples per configuration type
n_samples_per_config = 10  # You can increase this for more scenarios

# Random seed for reproducibility
seed = 42

# === OUTPUT FOLDER ===
output_folder = Path("generated_topology")
output_folder.mkdir(parents=True, exist_ok=True)

# Generate scenarios using Latin Hypercube Sampling
scenario_counter = 1  # Global counter starting from 1
plot_every = 1  # Plot every Nth scenario

print("="*80)
print("GENERATING FIXED CONFIGURATION SCENARIOS WITH LHS")
print("="*80)
print(f"\nConfiguration types: {config_types}")
print(f"Large-Large distance range: {d_large_large_range} km")
print(f"Small-Small distance range: {d_small_small_range} km")
print(f"Small-Large distance range: {d_small_large_range} km")
print(f"Samples per configuration: {n_samples_per_config}")
print(f"\nTotal scenarios to generate: {len(config_types) * n_samples_per_config}")
print("="*80)
print()

for config_idx, config_type in enumerate(config_types):
    print(f"\n--- Generating Configuration Type {config_type} ---")

    config_descriptions = {
        'A': 'Both Small clusters near Large1',
        'B': 'Both Small clusters near Large2',
        'C': 'Small clusters equidistant from Large nodes',
        'D': 'One Small near Large1, one near Large2'
    }
    print(f"    {config_descriptions[config_type]}")

    # Generate Latin Hypercube samples for 3 parameters
    # Use different seed for each configuration type
    config_seed = seed + config_idx * 1000
    lhs_samples = latin_hypercube(n_samples_per_config, 3, seed=config_seed)

    count_for_config = 0

    for sample_idx, (u_ll, u_ss, u_sl) in enumerate(lhs_samples):
        # Map [0,1] samples to actual parameter ranges
        d_ll = d_large_large_range[0] + u_ll * (d_large_large_range[1] - d_large_large_range[0])
        d_ss = d_small_small_range[0] + u_ss * (d_small_small_range[1] - d_small_small_range[0])
        d_sl = d_small_large_range[0] + u_sl * (d_small_large_range[1] - d_small_large_range[0])

        # Create scenario name with sequential number (0001, 0002, ...)
        scenario_name = f"{scenario_counter:04d}"

        # Print detailed info for tracking
        print(f"    Scenario {scenario_name}: Type {config_type}, LL={int(d_ll)}km, SS={int(d_ss)}km, SL={int(d_sl)}km")

        try:
            positions_dict = generate_scenario(
                config_type=config_type,
                d_large_large=d_ll,
                d_small_large=d_sl,
                d_small_small=d_ss,
                scenario_name=scenario_name,
                output_folder=output_folder
            )

            if positions_dict is not None:
                count_for_config += 1
                scenario_counter += 1  # Increment global counter

                # Plot some scenarios for visualization
                if (scenario_counter - 1) % plot_every == 0 or (scenario_counter - 1) <= 4:
                    plot_nodes(positions_dict, config_type, scenario_name, output_folder)


        except Exception as e:
            print(f"    ⚠️  Error generating {scenario_name}: {e}")
            continue

    print(f"    ✓ Type {config_type} complete: {count_for_config} scenarios generated")

print()
print("="*80)
print("GENERATION COMPLETE")
print("="*80)
print(f"\n📁 Output folder: '{output_folder.resolve()}'")
print(f"✓ Total scenarios generated: {scenario_counter - 1}")
print(f"✓ CSV files: {scenario_counter - 1}")
print(f"✓ Plot files: ~{(scenario_counter - 1) // plot_every + 4}")
print()
print("Configuration summary:")
for config_type in config_types:
    print(f"  Type {config_type}: {n_samples_per_config} scenarios (LHS)")
print()
print("Parameter ranges explored:")
print(f"  d_large_large: {d_large_large_range[0]:.0f} - {d_large_large_range[1]:.0f} km")
print(f"  d_small_small: {d_small_small_range[0]:.0f} - {d_small_small_range[1]:.0f} km")
print(f"  d_small_large: {d_small_large_range[0]:.0f} - {d_small_large_range[1]:.0f} km")
print("="*80)
