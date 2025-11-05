# ==============================================================================
# Artificial Spin Ice Lattice Generator and Monte Carlo Simulator
# ==============================================================================
#
# DESCRIPTION:
# This script provides a comprehensive toolkit for generating, modifying, and
# simulating artificial spin ice systems on a Kagome lattice derived from a
# honeycomb structure.
#
# WORKFLOW:
# 1.  Generate Honeycomb Grid: Creates a grid of hexagonal centers.
# 2.  Modify Lattice: Selectively removes specified hexagons to create custom
#     finite geometries.
# 3.  Add Pinning Sites: Appends additional spin sites (pins) to the boundaries
#     (top, bottom, left, right) of the lattice. Pins can be 'fixed' or 'free'.
# 4.  Run Monte Carlo Simulation: Executes a Monte Carlo simulated annealing
#     process to find low-energy spin configurations for the generated lattice,
#     respecting the constraints of any fixed pins.
# 5.  Save Results: Stores the final lattice structure (edges, centers) and the
#     simulated spin configurations (TrainData.npy) in a structured directory.
#
# USAGE:
# - Configure parameters in the `main()` function (grid size, hexagons to remove,
#   simulation settings).
# - The script uses a TensorFlow 1.x compatible graph-based execution model.
#
# LIBRARIES:
# - TensorFlow (1.x API), NumPy, Pandas, Matplotlib
#
# EXTERNAL DEPENDENCIES:
# - EVAE_Modules.py & Other_Modules.py: Contain helper functions like get_rotmat,
#   get_dipbasis, MyClock, etc. These files are required for full functionality.
#
# ==============================================================================


# --- 1. Imports and Global Settings ---
try:
    from EVAE_Modules import *
    from Other_Modules import *

    print("Successfully imported CustomLib modules.")
except ImportError:
    print("Warning: CustomLib modules not found. Using dummy placeholder functions.")
    # Dummy functions for basic script execution without custom libs
    import numpy as np
    import tensorflow as tf
    import time
    import os

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import warnings

# --- TensorFlow and Matplotlib Configuration ---
tf.compat.v1.disable_eager_execution()
warnings.filterwarnings('ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try:
    matplotlib.use('TkAgg')
except ImportError:
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        matplotlib.use('Agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --- Global Variables ---
hex_centers = None  # Global storage for hexagon centers after removal
myClock = None  # Global timer instance


# ==============================================================================
# 2. Hexagonal Lattice Generation Functions
# ==============================================================================
def generate_flat_top_hexagon(center, a=1.0):
    x0, y0 = center
    angles_deg = np.array([60 * i for i in range(6)])
    angles_rad = np.deg2rad(angles_deg)
    x = x0 + a * np.cos(angles_rad)
    y = y0 + a * np.sin(angles_rad)
    return list(zip(x, y))


def offset_to_cartesian(col, row, a=1.0):
    x = a * 3 / 2 * col
    y = a * np.sqrt(3) * (row + 0.5 * (col % 2))
    return x, y


def generate_hexagon_grid(cols=10, rows=10, a=1.0):
    hex_centers_list = []
    for c in range(cols):
        for r in range(rows):
            center_x, center_y = offset_to_cartesian(c, r, a)
            hex_centers_list.append(((center_x, center_y), c, r))
    print(f"[GridGen] Generated {len(hex_centers_list)} hex centers ({cols}x{rows}).")
    return hex_centers_list


def remove_hex_by_colrow(hex_centers_list, col_target, row_target):
    initial_count = len(hex_centers_list)
    filtered_list = [item for item in hex_centers_list if not (item[1] == col_target and item[2] == row_target)]
    removed_count = initial_count - len(filtered_list)
    if removed_count > 0:
        print(f"[GridMod] Removed hex at (col={col_target}, row={row_target}). New count: {len(filtered_list)}")
    else:
        print(f"[GridMod] Hex at (col={col_target}, row={row_target}) not found.")
    return filtered_list


# ==============================================================================
# 3. Edge Key Generation
# ==============================================================================
def edge_key(v1, v2, precision=6):
    v1r = (round(v1[0], precision), round(v1[1], precision))
    v2r = (round(v2[0], precision), round(v2[1], precision))
    return tuple(sorted([v1r, v2r]))


# ==============================================================================
# 4. CSV Generation
# ==============================================================================
def save_unique_lattice_csv(hex_centers_list, a=1.0, filename="Original_Lattice.csv"):
    edges_dict = {}
    for item in hex_centers_list:  # Use hex_centers after removal
        center_coords = item[0]
        vertices = generate_flat_top_hexagon(center_coords, a)
        for i in range(6):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 6]
            key = edge_key(v1, v2)
            if key not in edges_dict:
                mid_x = (v1[0] + v2[0]) / 2.0
                mid_y = (v1[1] + v2[1]) / 2.0
                edges_dict[key] = (v1[0], v1[1], v2[0], v2[1], mid_x, mid_y, 0.0)
    rows_list = list(edges_dict.values())
    num_edges = len(rows_list)
    if num_edges == 0:
        print(f"[CSV] Warning: No edges found. CSV '{filename}' will be empty.")
        df = pd.DataFrame(columns=["X1", "Y1", "X2", "Y2", "XC", "YC", "SkFpIndex"])
    else:
        arr = np.array(rows_list, dtype=np.float32)
        df = pd.DataFrame({
            "X1": arr[:, 0], "Y1": arr[:, 1], "X2": arr[:, 2], "Y2": arr[:, 3],
            "XC": arr[:, 4], "YC": arr[:, 5], "SkFpIndex": arr[:, 6]
        })
    df.to_csv(filename, index=False, float_format='%.6f')
    print(f"[CSV] Saved '{filename}' with {num_edges} unique edges.")
    return filename


# ==============================================================================
# 5. Pinning Functions
# ==============================================================================
def _append_pins_aligned(input_csv, output_csv, a, pin_threshold, pin_edge_length, pin_interval, side):
    """Internal helper function to add aligned pins to a specified side."""
    try:
        data = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"[_append_pins] Error: Input CSV '{input_csv}' not found.")
        return None

    global hex_centers
    if hex_centers is None or len(hex_centers) == 0:
        raise ValueError(f"[_append_pins] Global 'hex_centers' is required for side '{side}'.")

    centers_arr = np.array([item[0] for item in hex_centers])
    if centers_arr.size == 0:
        raise ValueError(f"[_append_pins] No centers available for side '{side}'.")

    if pin_edge_length is None:
        pin_edge_length = a

    boundary_hexagons = []
    coord_idx = -1
    compare_func = None
    boundary_val = 0.0
    skfp_index = 2.0  # Default: free pin

    if side == 'top':
        coord_idx = 1
        boundary_val = centers_arr[:, coord_idx].max()
        compare_func = lambda coord, val, thresh: coord >= val - thresh
        skfp_index = 2.0  # Top pins are also free
        print(f"[Pinning] Top boundary found near y={boundary_val:.6f}")
    elif side == 'bottom':
        coord_idx = 1
        boundary_val = centers_arr[:, coord_idx].min()
        compare_func = lambda coord, val, thresh: coord <= val + thresh
        print(f"[Pinning] Bottom boundary found near y={boundary_val:.6f}")
    elif side == 'left':
        coord_idx = 0
        boundary_val = centers_arr[:, coord_idx].min()
        compare_func = lambda coord, val, thresh: coord <= val + thresh
        print(f"[Pinning] Left boundary found near x={boundary_val:.6f}")
    elif side == 'right':
        coord_idx = 0
        boundary_val = centers_arr[:, coord_idx].max()
        compare_func = lambda coord, val, thresh: coord >= val - thresh
        print(f"[Pinning] Right boundary found near x={boundary_val:.6f}")
    else:
        raise ValueError(f"Invalid side specified: {side}")

    boundary_hexagons = [item for item in hex_centers if compare_func(item[0][coord_idx], boundary_val, pin_threshold)]

    new_pins = []
    pinned_count = 0
    boundary_count = 0

    for idx, item in enumerate(boundary_hexagons):
        center_coords, c, r = item
        center_x, center_y = center_coords
        boundary_count += 1
        if boundary_count % pin_interval == 0:
            vertices = generate_flat_top_hexagon(center_coords, a)
            extreme_vertices = []
            if side == 'top':
                extreme_val = max(v[1] for v in vertices)
                extreme_vertices = [v for v in vertices if abs(v[1] - extreme_val) < 1e-6]
            elif side == 'bottom':
                extreme_val = min(v[1] for v in vertices)
                extreme_vertices = [v for v in vertices if abs(v[1] - extreme_val) < 1e-6]
            elif side == 'left':
                extreme_val = min(v[0] for v in vertices)
                extreme_vertices = [v for v in vertices if abs(v[0] - extreme_val) < 1e-6]
            elif side == 'right':
                extreme_val = max(v[0] for v in vertices)
                extreme_vertices = [v for v in vertices if abs(v[0] - extreme_val) < 1e-6]

            for v_extreme in extreme_vertices:
                vx, vy = v_extreme
                vec_x, vec_y = vx - center_x, vy - center_y
                norm = np.sqrt(vec_x ** 2 + vec_y ** 2)
                if norm < 1e-9: continue
                unit_x, unit_y = vec_x / norm, vec_y / norm
                p_out_x = vx + unit_x * pin_edge_length
                p_out_y = vy + unit_y * pin_edge_length
                p_mid_x, p_mid_y = (vx + p_out_x) / 2.0, (vy + p_out_y) / 2.0
                new_pins.append([vx, vy, p_out_x, p_out_y, p_mid_x, p_mid_y, skfp_index])
                pinned_count += 1

    pin_type = "fixed" if skfp_index == 1.0 else "free"
    print(
        f"[Pinning] Found {len(boundary_hexagons)} boundary hexagons on '{side}'. Added {pinned_count} {pin_type} pins (SkFp={skfp_index}).")
    df_new = pd.DataFrame(new_pins, columns=["X1", "Y1", "X2", "Y2", "XC", "YC", "SkFpIndex"])
    out_data = pd.concat([data, df_new], ignore_index=True)
    out_data.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"[Pinning] Saved '{output_csv}'. Total rows = {len(out_data)}")
    return output_csv


# Wrapper functions for each side
def append_top_side_pin_aligned(input_csv, output_csv, a=1.0, pin_threshold=1e-3, pin_edge_length=None, pin_interval=1):
    return _append_pins_aligned(input_csv, output_csv, a, pin_threshold, pin_edge_length, pin_interval, 'top')


def append_bottom_side_pin_aligned(input_csv, output_csv, a=1.0, pin_threshold=1e-3, pin_edge_length=None,
                                   pin_interval=1):
    return _append_pins_aligned(input_csv, output_csv, a, pin_threshold, pin_edge_length, pin_interval, 'bottom')


def append_left_side_pin_aligned(input_csv, output_csv, a=1.0, pin_threshold=1e-3, pin_edge_length=None,
                                 pin_interval=1):
    return _append_pins_aligned(input_csv, output_csv, a, pin_threshold, pin_edge_length, pin_interval, 'left')


def append_right_side_pin_aligned(input_csv, output_csv, a=1.0, pin_threshold=1e-3, pin_edge_length=None,
                                  pin_interval=1):
    return _append_pins_aligned(input_csv, output_csv, a, pin_threshold, pin_edge_length, pin_interval, 'right')


# ==============================================================================
# 6. CSV Loading and Visualization
# ==============================================================================
def load_structureCSV(filename):
    try:
        data = pd.read_csv(filename)
        edges = data[["X1", "Y1", "X2", "Y2"]].values.astype(np.float32)
        centers = data[["XC", "YC"]].values.astype(np.float32)
        skfp = data["SkFpIndex"].values.astype(np.float32)
        print(f"[CSV] Loaded {len(edges)} edges/spins from '{filename}'.")
        unique, counts = np.unique(skfp, return_counts=True)
        print(f"[CSV] SkFpIndex counts: {dict(zip(unique, counts))}")
        return edges, centers, skfp
    except FileNotFoundError:
        print(f"[Error] File not found: '{filename}'")
        return None, None, None
    except KeyError as e:
        print(f"[Error] Missing column {e} in '{filename}'")
        return None, None, None


def plot_lattice_pins_no_shift(csv_file, title="Lattice Structure with Pins"):
    """Visualizes the lattice structure and all pins from a CSV file."""
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"[Plot] Error: CSV file not found: '{csv_file}'")
        return
    except Exception as e:
        print(f"[Plot] Error reading CSV: {e}")
        return

    edges = data[["X1", "Y1", "X2", "Y2"]].values
    centers = data[["XC", "YC"]].values
    skfp = data["SkFpIndex"].values

    mask_orig = (skfp == 0.0)
    mask_top_fixed = (skfp == 1.0)  # NOTE: This was changed for fixed top pins
    mask_other_free = (skfp == 2.0)

    plt.figure(figsize=(12, 12))

    # Original Lattice
    for e in edges[mask_orig]:
        plt.plot([e[0], e[2]], [e[1], e[3]], 'g-', lw=0.7, alpha=0.8, label='_nolegend_')
    if np.any(mask_orig):
        plt.scatter(centers[mask_orig][:, 0], centers[mask_orig][:, 1], color='blue', s=15,
                    label=f'Original Spins ({np.sum(mask_orig)})', alpha=0.8)

    # Top Fixed Pins
    for e in edges[mask_top_fixed]:
        plt.plot([e[0], e[2]], [e[1], e[3]], 'r--', lw=1.5, label='_nolegend_')
    if np.any(mask_top_fixed):
        plt.scatter(centers[mask_top_fixed][:, 0], centers[mask_top_fixed][:, 1], color='red', s=35, marker='s',
                    label=f'Top Pins (Fixed, SkFp=1.0) ({np.sum(mask_top_fixed)})')

    # Other Free Pins
    for e in edges[mask_other_free]:
        plt.plot([e[0], e[2]], [e[1], e[3]], 'm-.', lw=1.5, label='_nolegend_')
    if np.any(mask_other_free):
        plt.scatter(centers[mask_other_free][:, 0], centers[mask_other_free][:, 1], color='magenta', s=35, marker='^',
                    label=f'Other Pins (Free, SkFp=2.0) ({np.sum(mask_other_free)})')

    plt.title(title, fontsize=14)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 7. Monte Carlo Simulation
# ==============================================================================
def run_simulation_with_pins(edges, centers, skfp, paramdict):
    """Runs a Monte Carlo simulation with support for fixed pin states."""
    print("[Simulation] Starting Monte Carlo Simulation...")
    tf.compat.v1.reset_default_graph()

    N = len(edges)
    if N == 0:
        print("[Error] No edges/spins found for simulation.")
        return None

    try:
        rotmat = get_rotmat(edges)
        nnidx = None  # Not using nearest-neighbor for long-range interactions
        print(f"[Simulation] Structure info processed. N={N}")
    except Exception as e:
        print(f"[Error] Failed to process structure info: {e}")
        return None

    fixed_mask = (skfp == 1.0)  # Mask for pins to be fixed (SkFpIndex == 1.0)
    num_fixed = np.sum(fixed_mask)
    fixed_indices = np.where(fixed_mask)[0]
    print(f"[Simulation] Total spins: {N}, Fixed spins (SkFp=1.0): {num_fixed}, Free spins: {N - num_fixed}")

    # Calculate target spin orientations ONLY for fixed pins
    target_states_np = np.zeros((N, 3), dtype=np.float32)
    print("[Simulation] Calculating target states for fixed pins (SkFp=1.0)...")
    for idx in fixed_indices:
        x1, y1, x2, y2 = edges[idx]
        inward_vec_x, inward_vec_y = x1 - x2, y1 - y2
        norm = np.sqrt(inward_vec_x ** 2 + inward_vec_y ** 2)
        if norm < 1e-9:
            print(f"Warning: Fixed pin {idx} has zero length. Setting target to (1,0,0).")
            target_states_np[idx, :] = [1.0, 0.0, 0.0]
        else:
            sx, sy = inward_vec_x / norm, inward_vec_y / norm
            target_states_np[idx, :] = [sx, sy, 0.0]

    batch_size = paramdict.get("BatchSize", 1)
    try:
        X_init_np = get_IC(N, rotmat=rotmat, batch_size=batch_size)  # Shape: (BS, N, 3)
        print(f"[Simulation] Initial random spins generated. Shape: {X_init_np.shape}")
    except Exception as e:
        print(f"[Error] Failed to generate initial conditions: {e}")
        return None

    # Apply the calculated target directions to the initial state for fixed pins
    target_states_init = np.tile(target_states_np[np.newaxis, :, :], (batch_size, 1, 1))
    init_mask = np.broadcast_to(fixed_mask[np.newaxis, :, np.newaxis], (batch_size, N, 1))
    X_init_np = np.where(init_mask, target_states_init, X_init_np)
    print("[Simulation] Initial state for fixed pins has been set.")

    # Build the TensorFlow graph
    try:
        X_ph = tf.compat.v1.placeholder(tf.float32, shape=X_init_np.shape, name="X_placeholder")
        tfX = tf.compat.v1.Variable(X_ph, dtype=tf.float32, name="Spins")
        dipBasis = tf.constant(get_dipbasis(centers, paramdict.get("DR", 1.0), nnidx), name="DipoleBasis")
        totalHeff = get_dipheff(tfX, paramdict.get("Dip", 1.0), dipBasis)
        dipEnergy = tf.reduce_mean(-tf.reduce_sum(input_tensor=tfX * totalHeff / 2.0, axis=-1),
                                   name="AverageDipoleEnergy")
        T = tf.compat.v1.placeholder(tf.float64, shape=[], name="Temperature")
        nexttfX_candidate = get_MultiFlip_MPMC_engine(tfX, T, totalHeff)

        # Pinning Logic: Keep fixed pins at their target state
        fixed_mask_tf = tf.constant(fixed_mask, dtype=tf.bool)
        fixed_mask_expanded = fixed_mask_tf[None, :, None]
        batch_dim_size = tf.shape(input=tfX)[0]
        fixed_mask_tiled_3d = tf.tile(fixed_mask_expanded, multiples=[batch_dim_size, 1, 3])

        target_states_tf_const = tf.constant(target_states_np[np.newaxis, :, :], dtype=tf.float32)
        target_states_batch_tf = tf.tile(target_states_tf_const, multiples=[batch_dim_size, 1, 1])

        next_with_pins = tf.compat.v1.where(fixed_mask_tiled_3d, target_states_batch_tf, nexttfX_candidate)

        update = tfX.assign(next_with_pins)
        init = tf.compat.v1.global_variables_initializer()
    except Exception as e:
        print(f"[Error] Failed during TensorFlow graph construction: {e}")
        return None

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    X_final = None

    # Run the TensorFlow session
    try:
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(init, feed_dict={X_ph: X_init_np})
            feed = {}
            global myClock
            if myClock is None: myClock = MyClock()
            myClock.init()

            total_iter = paramdict.get("Total_Iteration", 10000)
            sub_iter = paramdict.get("Sub_Iteration", 1000)
            print(f"[Simulation] Starting simulation loop for {total_iter} iterations...")

            for it in range(total_iter):
                Tvalue = get_Tfeed(it, paramdict)
                feed.update({T: Tvalue})
                sess.run(update, feed_dict=feed)

                if it % sub_iter == 0 or it == total_iter - 1:
                    energy_v = sess.run(dipEnergy, feed_dict=feed)
                    elapsed = myClock.tictoc()
                    elapsed_str = f"{elapsed:.3f}" if elapsed is not None else "N/A"
                    energy_str = f"{energy_v:.6f}" if np.isfinite(energy_v) else "Invalid"
                    print(f"Iter {it}/{total_iter} | T={Tvalue:.4f} | Avg E={energy_str} | Step Time: {elapsed_str}s")

            X_final = sess.run(tfX)
            if not np.all(np.isfinite(X_final)):
                print("!!! Warning: Final spin state contains NaN/Inf values !!!")
                return None
            print(f"[Simulation] Simulation finished. Final spin configuration shape = {X_final.shape}")
    except Exception as e:
        print(f"[Error] An unexpected error occurred during the session: {e}")

    return X_final


# ==============================================================================
# 8. Results Saving Function
# ==============================================================================
def save_simulation_results(base_dir, cols, rows, removed_hex_info,
                            edges, centers, skfp, sim_data, pin_interval, dr_value):
    """Saves all simulation results to a structured directory."""
    if sim_data is None:
        print("[Save] Error: Simulation data is None. Nothing to save.")
        return
    print("[Save] Saving simulation results...")

    try:
        nodes = np.unique(np.concatenate([edges[:, :2], edges[:, 2:]], axis=0), axis=0)
        rotmat = get_rotmat(edges)
        nnidx = get_nnidx(edges)  # This might be unused if DR is large
    except Exception as e:
        print(f"[Save] Warning: Error calculating rotmat/nnidx: {e}")
        rotmat, nnidx = None, None

    # Create a descriptive directory name
    dataset_savedir_name = f"Sim_Cols{cols}Rows{rows}_DR{dr_value:.2f}"
    if removed_hex_info:
        removed_str = "Rm_" + "_".join([f"c{c}r{r}" for c, r in removed_hex_info])
        dataset_savedir_name = f"Sim_Cols{cols}Rows{rows}_{removed_str}_DR{dr_value:.2f}"

    dataset_savedir_path = os.path.join(base_dir, dataset_savedir_name)
    mkdir(dataset_savedir_path)

    try:
        # Save numpy arrays
        np.save(os.path.join(dataset_savedir_path, "Edges.npy"), edges)
        np.save(os.path.join(dataset_savedir_path, "CenterOfEdges.npy"), centers)
        np.save(os.path.join(dataset_savedir_path, "Nodes.npy"), nodes)
        np.save(os.path.join(dataset_savedir_path, "Rotmat.npy"), rotmat)
        np.save(os.path.join(dataset_savedir_path, "NNIndices.npy"), nnidx)
        np.save(os.path.join(dataset_savedir_path, "SkFpBool.npy"), skfp)
        np.save(os.path.join(dataset_savedir_path, "TrainData.npy"), sim_data)

        # Save metadata
        num_samples, num_spins = sim_data.shape[0], sim_data.shape[1]
        with open(os.path.join(dataset_savedir_path, "Info.txt"), "w") as f:
            f.write(f"BatchSize = {num_samples}\n")
            f.write(f"NumberOfSpins = {num_spins}\n")
            f.write(f"RemovedHexagons = {removed_hex_info}\n")
            f.write(f"PinningInfo = All sides pinned, Top Fixed (SkFp=1.0), Others Free (SkFp=2.0)\n")
        print(f"[Save] All results saved to: {dataset_savedir_path}")
    except Exception as e:
        print(f"[Save] Error while saving files: {e}")


# ==============================================================================
# 9. Main Execution Logic
# ==============================================================================
def main():
    """Main execution function: configures and runs the full pipeline."""
    print("=" * 70)
    print("Hex Lattice Sim: Grid Removal + All Sides Aligned Pins (Top Fixed)")
    print("=" * 70)

    # --- Configuration ---
    base_dir = os.path.join(os.getcwd(), "D")
    mkdir(base_dir)
    lattice_a = 1.0 / np.sqrt(3)
    grid_cols = 10
    grid_rows = 10

    # List of (col, row) tuples for hexagons to be removed.
    hex_to_remove = []  # Empty list for no removal.

    pinning_interval = 1
    pin_edge_len = lattice_a

    paramdict = {
        "Total_Iteration": 10000, "Sub_Iteration": 1000, "Tstart": 3.0, "Tend": 0.01,
        "Annealing%": 0.95, "Dip": 1.0, "DR": 1000.00, "BatchSize": 3000
    }

    # --- Intermediate File Naming ---
    file_suffix = f"_C{grid_cols}R{grid_rows}_NoRemoval" if not hex_to_remove else f"_C{grid_cols}R{grid_rows}_Removed"
    original_lattice_csv = f"Intermediate_Original{file_suffix}.csv"
    lattice_with_top_pins_csv = f"Intermediate_TopPins{file_suffix}.csv"
    lattice_with_tb_pins_csv = f"Intermediate_TBPins{file_suffix}.csv"
    lattice_with_tbl_pins_csv = f"Intermediate_TBLPins{file_suffix}.csv"
    final_lattice_with_pins_csv = f"Final_Lattice_AllPins{file_suffix}.csv"

    global myClock, hex_centers
    myClock = MyClock()

    # --- Step A: Generate Grid ---
    print("\n--- Step A: Generating Grid ---")
    initial_hex_centers = generate_hexagon_grid(grid_cols, grid_rows, lattice_a)
    if not initial_hex_centers:
        print("Error: Grid generation failed.")
        return

    # --- Step B: Remove Specified Hexagons ---
    print("\n--- Step B: Removing Specified Hexagons ---")
    removed_hex_log = []
    hex_centers = list(initial_hex_centers)  # Update global variable
    if hex_to_remove:
        print(f"Attempting to remove: {hex_to_remove}")
        for col_r, row_r in hex_to_remove:
            original_count = len(hex_centers)
            hex_centers = remove_hex_by_colrow(hex_centers, col_r, row_r)
            if len(hex_centers) < original_count:
                removed_hex_log.append((col_r, row_r))
        print(f"Actually removed: {removed_hex_log}")
    else:
        print("No hexagons specified for removal.")

    # --- Step C: Save Initial Lattice (Post-Removal) ---
    print("\n--- Step C: Saving Initial Lattice ---")
    save_unique_lattice_csv(hex_centers, lattice_a, original_lattice_csv)

    # --- Step D: Append Pins Sequentially (Currently Disabled) ---
    print("\n--- Step D: Appending Pins (Currently Disabled) ---")
    # To enable pinning, uncomment the following lines and adjust the logic.
    # The current flow adds pins sequentially, saving an intermediate CSV at each step.
    current_csv = original_lattice_csv

    # # 1. Top Pins (Example)
    # current_csv = append_top_side_pin_aligned(
    #     input_csv=current_csv, output_csv=lattice_with_top_pins_csv,
    #     a=lattice_a, pin_edge_length=pin_edge_len, pin_interval=pinning_interval
    # )
    # if current_csv is None: print("Error adding top pins."); return

    # # 2. Bottom Pins
    # current_csv = append_bottom_side_pin_aligned(
    #     input_csv=current_csv, output_csv=lattice_with_tb_pins_csv,
    #     a=lattice_a, pin_edge_length=pin_edge_len, pin_interval=pinning_interval
    # )
    # if current_csv is None: print("Error adding bottom pins."); return

    # ... and so on for left and right sides.

    final_pinned_csv = current_csv  # Use the last valid CSV file.

    # --- Step E: Plot Final Lattice ---
    print("\n--- Step E: Plotting Final Lattice ---")
    removed_info_str = f"Removed: {removed_hex_log}" if removed_hex_log else "No Removal"
    plot_title = f"Lattice {grid_cols}x{grid_rows}, {removed_info_str}, Pinned (Top Fixed)"
    plot_lattice_pins_no_shift(final_pinned_csv, title=plot_title)

    # --- Step F: Run Simulation ---
    print("\n--- Step F: Running Simulation ---")
    edges_arr, centers_arr, skfp_arr = load_structureCSV(final_pinned_csv)
    if edges_arr is None:
        print("Error: Failed to load final lattice for simulation.")
        return
    sim_data = run_simulation_with_pins(edges_arr, centers_arr, skfp_arr, paramdict)
    if sim_data is None:
        print("Error: Simulation failed. Exiting.")
        return
    print(f"[Main] Final dataset shape = {sim_data.shape}")

    # --- Step G: Save Results ---
    print("\n--- Step G: Saving Results ---")
    save_simulation_results(base_dir, grid_cols, grid_rows, removed_hex_log,
                            edges_arr, centers_arr, skfp_arr,
                            sim_data, pinning_interval, paramdict["DR"])

    print("\n" + "=" * 70)
    print(f"Pipeline finished for {grid_cols}x{grid_rows} grid.")
    print("=" * 70)


# ==============================================================================
# 10. Execution Point
# ==============================================================================
if __name__ == "__main__":
    main()
