###############################################################################
# 1. Imports and Global Settings
###############################################################################
try:
    from CustomLib.EVAE_Modules import *
    from CustomLib.Other_Modules import *
    print("Successfully imported CustomLib.")
except ImportError:
    print("Warning: CustomLib not found. Using dummy placeholder functions.")
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

tf.compat.v1.disable_eager_execution()
warnings.filterwarnings('ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try: matplotlib.use('TkAgg')
except ImportError:
    try: matplotlib.use('Qt5Agg')
    except ImportError: matplotlib.use('Agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
hex_centers = None # Global variable for storing the list of hexagon centers after removal
myClock = None

###############################################################################
# 2. Hexagonal Lattice Generation Functions
###############################################################################
def generate_flat_top_hexagon(center, a=1.0):
    x0, y0 = center; angles_deg = np.array([60 * i for i in range(6)]); angles_rad = np.deg2rad(angles_deg)
    x = x0 + a * np.cos(angles_rad); y = y0 + a * np.sin(angles_rad)
    return list(zip(x, y))

def offset_to_cartesian(col, row, a=1.0):
    x = a * 3/2 * col; y = a * np.sqrt(3) * (row + 0.5 * (col % 2))
    return x, y

def generate_hexagon_grid(cols=10, rows=10, a=1.0):
    hex_centers_list = []
    for c in range(cols):
        for r in range(rows):
            center_x, center_y = offset_to_cartesian(c, r, a); hex_centers_list.append(((center_x, center_y), c, r))
    print(f"[generate_hexagon_grid] Generated {len(hex_centers_list)} hex centers ({cols}x{rows}).")
    return hex_centers_list

###############################################################################
# 2-B. Function to remove a specific hexagon by col/row index
###############################################################################
def remove_hex_by_colrow(hex_centers_list, col_target, row_target):
    initial_count = len(hex_centers_list)
    filtered_list = [item for item in hex_centers_list if not (item[1] == col_target and item[2] == row_target)]
    removed_count = initial_count - len(filtered_list)
    if removed_count > 0: print(f"[remove_hex] Removed hex at (col={col_target}, row={row_target}). New count: {len(filtered_list)}")
    else: print(f"[remove_hex] Hex at (col={col_target}, row={row_target}) not found.")
    return filtered_list

###############################################################################
# 3. Edge Key Generation Function
###############################################################################
def edge_key(v1, v2, precision=6):
    v1r = (round(v1[0], precision), round(v1[1], precision)); v2r = (round(v2[0], precision), round(v2[1], precision))
    return tuple(sorted([v1r, v2r]))

###############################################################################
# 4. Original CSV Generation Function
###############################################################################
def save_unique_lattice_csv(hex_centers_list, a=1.0, filename="Original_Lattice.csv"):
    edges_dict = {}
    for item in hex_centers_list: # Use hex_centers after removal
        center_coords = item[0]; vertices = generate_flat_top_hexagon(center_coords, a)
        for i in range(6):
            v1=vertices[i]; v2=vertices[(i+1)%6]; key=edge_key(v1,v2)
            if key not in edges_dict:
                mid_x=(v1[0]+v2[0])/2.0; mid_y=(v1[1]+v2[1])/2.0; edges_dict[key] = (v1[0],v1[1],v2[0],v2[1],mid_x,mid_y,0.0)
    rows_list=list(edges_dict.values()); num_edges=len(rows_list)
    if num_edges==0: print(f"[save_csv] Warning: No edges. CSV '{filename}' empty."); df=pd.DataFrame(columns=["X1","Y1","X2","Y2","XC","YC","SkFpIndex"])
    else: arr=np.array(rows_list, dtype=np.float32); df=pd.DataFrame({"X1":arr[:,0],"Y1":arr[:,1],"X2":arr[:,2],"Y2":arr[:,3],"XC":arr[:,4],"YC":arr[:,5],"SkFpIndex":arr[:,6]})
    df.to_csv(filename, index=False);
    df.to_csv(filename, index=False, float_format='%.4f')
    print(f"[save_csv] Saved '{filename}' with {num_edges} unique edges.")

    return filename

###############################################################################
# 5. Pin Addition Functions (Aligned - Top, Bottom, Left, Right)
###############################################################################
def _append_pins_aligned(input_csv, output_csv, a, pin_threshold, pin_edge_length, pin_interval, side):
    """ Internal helper function: adds aligned pins to a specified side """
    try: data = pd.read_csv(input_csv)
    except FileNotFoundError: print(f"[_append_pins] Error: Input CSV '{input_csv}' not found."); return None

    global hex_centers # Use the list of hexagon centers after removal
    if hex_centers is None or len(hex_centers) == 0: raise ValueError(f"[_append_pins] Global 'hex_centers' needed for side '{side}'.")
    centers_arr = np.array([item[0] for item in hex_centers]);
    if centers_arr.size == 0: raise ValueError(f"[_append_pins] No centers left for side '{side}'.")

    if pin_edge_length is None: pin_edge_length = a

    boundary_hexagons = []
    coord_idx = -1 # 0 for X, 1 for Y
    compare_func = None
    boundary_val = 0.0
    skfp_index = 2.0 # Default value: free pin

    if side == 'top':
        coord_idx = 1; boundary_val = centers_arr[:, coord_idx].max()
        compare_func = lambda coord, val, thresh: coord >= val - thresh
        skfp_index = 2.0 # Top pins are also free pins
        print(f"[append_pin_{side}] Top boundary near y={boundary_val:.6f}")
    elif side == 'bottom':
        coord_idx = 1; boundary_val = centers_arr[:, coord_idx].min()
        compare_func = lambda coord, val, thresh: coord <= val + thresh
        print(f"[append_pin_{side}] Bottom boundary near y={boundary_val:.6f}")
    elif side == 'left':
        coord_idx = 0; boundary_val = centers_arr[:, coord_idx].min()
        compare_func = lambda coord, val, thresh: coord <= val + thresh
        print(f"[append_pin_{side}] Left boundary near x={boundary_val:.6f}")
    elif side == 'right':
        coord_idx = 0; boundary_val = centers_arr[:, coord_idx].max()
        compare_func = lambda coord, val, thresh: coord >= val - thresh
        print(f"[append_pin_{side}] Right boundary near x={boundary_val:.6f}")
    else:
        raise ValueError(f"Invalid side specified: {side}")

    boundary_hexagons = [item for item in hex_centers if compare_func(item[0][coord_idx], boundary_val, pin_threshold)]

    new_pins = []; pinned_count = 0; boundary_count = 0

    for idx, item in enumerate(boundary_hexagons):
        center_coords, c, r = item; center_x, center_y = center_coords; boundary_count += 1
        if boundary_count % pin_interval == 0:
            vertices = generate_flat_top_hexagon(center_coords, a)
            extreme_vertices = []
            if side == 'top':
                extreme_val = max(v[1] for v in vertices); extreme_vertices = [v for v in vertices if abs(v[1] - extreme_val) < 1e-6]
            elif side == 'bottom':
                extreme_val = min(v[1] for v in vertices); extreme_vertices = [v for v in vertices if abs(v[1] - extreme_val) < 1e-6]
            elif side == 'left':
                extreme_val = min(v[0] for v in vertices); extreme_vertices = [v for v in vertices if abs(v[0] - extreme_val) < 1e-6]
            elif side == 'right':
                extreme_val = max(v[0] for v in vertices); extreme_vertices = [v for v in vertices if abs(v[0] - extreme_val) < 1e-6]

            for v_extreme in extreme_vertices:
                vx, vy = v_extreme; vec_x = vx - center_x; vec_y = vy - center_y; norm = np.sqrt(vec_x**2 + vec_y**2)
                if norm < 1e-9: continue
                unit_x = vec_x / norm; unit_y = vec_y / norm
                p_out_x = vx + unit_x * pin_edge_length; p_out_y = vy + unit_y * pin_edge_length
                p_x1, p_y1 = vx, vy; p_x2, p_y2 = p_out_x, p_out_y
                p_mid_x = (p_x1 + p_x2) / 2.0; p_mid_y = (p_y1 + p_y2) / 2.0
                new_pins.append([p_x1, p_y1, p_x2, p_y2, p_mid_x, p_mid_y, skfp_index])
                pinned_count += 1

    pin_type = "fixed" if skfp_index == 1.0 else "free"
    print(f"[append_pin_{side}] Found {len(boundary_hexagons)} boundary hex. Added {pinned_count} {pin_type} pins (SkFp={skfp_index}).")
    df_new = pd.DataFrame(new_pins, columns=["X1","Y1","X2","Y2","XC","YC","SkFpIndex"])
    out_data = pd.concat([data, df_new], ignore_index=True);
    out_data.to_csv(output_csv, index=False, float_format='%.4f')
    print(f"[append_pin_{side}] Saved '{output_csv}'. Total rows={len(out_data)}")
    return output_csv

# Wrapper functions for each direction
def append_top_side_pin_aligned(input_csv, output_csv, a=1.0, pin_threshold=1e-3, pin_edge_length=None, pin_interval=1):
    return _append_pins_aligned(input_csv, output_csv, a, pin_threshold, pin_edge_length, pin_interval, 'top')

def append_bottom_side_pin_aligned(input_csv, output_csv, a=1.0, pin_threshold=1e-3, pin_edge_length=None, pin_interval=1):
    return _append_pins_aligned(input_csv, output_csv, a, pin_threshold, pin_edge_length, pin_interval, 'bottom')

def append_left_side_pin_aligned(input_csv, output_csv, a=1.0, pin_threshold=1e-3, pin_edge_length=None, pin_interval=1):
    return _append_pins_aligned(input_csv, output_csv, a, pin_threshold, pin_edge_length, pin_interval, 'left')

def append_right_side_pin_aligned(input_csv, output_csv, a=1.0, pin_threshold=1e-3, pin_edge_length=None, pin_interval=1):
    return _append_pins_aligned(input_csv, output_csv, a, pin_threshold, pin_edge_length, pin_interval, 'right')


###############################################################################
# 6-B. CSV Loading Function
###############################################################################
def load_structureCSV(filename):
    # ... (Code identical to previous, including edge return) ...
    try:
        data = pd.read_csv(filename); edges = data[["X1","Y1","X2","Y2"]].values.astype(np.float32)
        centers = data[["XC","YC"]].values.astype(np.float32); skfp = data["SkFpIndex"].values.astype(np.float32)
        print(f"[load_csv] Loaded {len(edges)} edges/spins from '{filename}'.")
        unique, counts = np.unique(skfp, return_counts=True); print(f"[load_csv] SkFpIndex: {dict(zip(unique, counts))}")
        return edges, centers, skfp
    except FileNotFoundError: print(f"[load_csv] Error: File '{filename}' not found."); return None,None,None
    except KeyError as e: print(f"[load_csv] Error: Missing column {e}"); return None,None,None

###############################################################################
# 7. Visualization Function (Modified: Legend)
###############################################################################
def plot_lattice_pins_no_shift(csv_file, title="Lattice Structure with Pins"):
    """ Visualize the lattice structure and all pins from a CSV file """
    try: data=pd.read_csv(csv_file)
    except FileNotFoundError: print(f"[plot_lattice] Error: CSV '{csv_file}' not found."); return
    except Exception as e: print(f"[plot_lattice] Error reading CSV: {e}"); return

    edges=data[["X1","Y1","X2","Y2"]].values
    centers=data[["XC","YC"]].values
    skfp=data["SkFpIndex"].values

    mask_orig=(skfp==0.0)   # Original
    mask_top_fixed=(skfp==2.0)  # Top free pins
    mask_other_free=(skfp==2.0) # Bottom, left, right free pins

    plt.figure(figsize=(12,12)) # Size can be adjusted

    # Original lattice
    for e in edges[mask_orig]: plt.plot([e[0],e[2]],[e[1],e[3]], 'g-', lw=0.7, alpha=0.8, label='_nolegend_')
    if np.sum(mask_orig)>0: plt.scatter(centers[mask_orig][:,0], centers[mask_orig][:,1], color='blue', s=15, label=f'Original Spins ({np.sum(mask_orig)})', alpha=0.8)

    # Top fixed pins
    for e in edges[mask_top_fixed]: plt.plot([e[0],e[2]],[e[1],e[3]], 'r--', lw=1.5, label='_nolegend_')
    if np.sum(mask_top_fixed)>0: plt.scatter(centers[mask_top_fixed][:,0], centers[mask_top_fixed][:,1], color='red', s=35, marker='s', label=f'Top Pins (Fixed, SkFp=1.0) ({np.sum(mask_top_fixed)})')

    # Other free pins (bottom, left, right) - displayed with one color/marker
    for e in edges[mask_other_free]: plt.plot([e[0],e[2]],[e[1],e[3]], 'm-.', lw=1.5, label='_nolegend_')
    if np.sum(mask_other_free)>0: plt.scatter(centers[mask_other_free][:,0], centers[mask_other_free][:,1], color='magenta', s=35, marker='^', label=f'Other Pins (Free, SkFp=2.0) ({np.sum(mask_other_free)})')

    plt.title(title, fontsize=14)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

###############################################################################
# 8. Monte Carlo Simulation (Fixing target direction for each pin)
###############################################################################
# The run_simulation_with_pins function provided in the previous answer goes here (no modification needed).
def run_simulation_with_pins(edges, centers, skfp, paramdict):
    """ Run Monte Carlo simulation (3D spins, forcing each fixed pin (SkFp==1.0) to its calculated inward direction) """
    print("[run_simulation] Starting Monte Carlo Simulation (forcing pin-specific fixed states)...")
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.get_default_graph()

    N = len(edges)
    if N == 0: print("[run_simulation] Error: No edges/spins found."); return None

    try:
        rotmat = get_rotmat(edges)
        nnidx = get_nnidx(edges)
        nnidx = None
        print(f"[run_simulation] Structure info processed. N={N}")
    except Exception as e: print(f"[run_simulation] Error processing structure: {e}"); return None

    fixed_mask = (skfp == 1.0) # Mask for pins to be fixed (only SkFpIndex == 1.0)
    num_fixed = np.sum(fixed_mask)
    fixed_indices = np.where(fixed_mask)[0]
    print(f"[run_simulation] Total spins: {N}, Fixed spins (SkFp=1.0): {num_fixed}, Free spins: {N-num_fixed}")

    # --- Calculate the target direction for each *fixed* pin ---
    target_states_np = np.zeros((N, 3), dtype=np.float32) # Free spin positions will remain zero
    print("[run_simulation] Calculating target states ONLY for fixed pins (SkFp=1.0)...")
    for idx in fixed_indices: # Calculate only for fixed pin indices
        x1, y1, x2, y2 = edges[idx]
        inward_vec_x = x1 - x2; inward_vec_y = y1 - y2
        norm = np.sqrt(inward_vec_x**2 + inward_vec_y**2)
        if norm < 1e-9:
            print(f"Warning: Fixed pin {idx} has zero length vector. Setting target to (1,0,0).")
            target_states_np[idx, :] = [1.0, 0.0, 0.0]
        else:
            sx = inward_vec_x / norm; sy = inward_vec_y / norm
            target_states_np[idx, :] = [sx, sy, 0.0] # Calculated direction (inward)
    # ---------------------------------

    batch_size = paramdict.get("BatchSize", 1)
    try:
        X_init_np = get_IC(N, rotmat=rotmat, batch_size=batch_size) # (BS, N, 3)
        print(f"[run_simulation] Initial random spins generated. Shape: {X_init_np.shape}")
    except Exception as e: print(f"[run_simulation] Error generating initial conditions: {e}"); return None

    # --- Set initial state: apply calculated target directions only at fixed pin locations ---
    target_states_init = np.tile(target_states_np[np.newaxis, :, :], (batch_size, 1, 1))
    init_mask = np.broadcast_to(fixed_mask[np.newaxis, :, np.newaxis], (batch_size, N, 1))
    X_init_np = np.where(init_mask, target_states_init, X_init_np)
    print("[run_simulation] Initial state for fixed pins set to calculated target directions.")
    # ------------------------------------------

    try: # Construct TensorFlow graph
        X_ph = tf.compat.v1.placeholder(tf.float32, shape=X_init_np.shape, name="X_placeholder")
        tfX = tf.compat.v1.Variable(X_ph, dtype=tf.float32, name="Spins")

        dipBasis = tf.constant(get_dipbasis(centers, paramdict.get("DR", 1.0), nnidx), name="DipoleBasis")
        totalHeff = get_dipheff(tfX, paramdict.get("Dip", 1.0), dipBasis)
        dipEnergy = tf.reduce_mean(-tf.reduce_sum(input_tensor=tfX * totalHeff / 2.0, axis=-1), name="AverageDipoleEnergy")

        T = tf.compat.v1.placeholder(tf.float64, shape=[], name="Temperature")
        nexttfX_candidate = get_MultiFlip_MPMC_engine(tfX, T, totalHeff)

        # --- Spin fixing logic: use target state tensor for each *fixed* pin ---
        fixed_mask_tf = tf.constant(fixed_mask, dtype=tf.bool)
        fixed_mask_expanded = fixed_mask_tf[None, :, None]
        batch_dim_size = tf.shape(input=tfX)[0]
        fixed_mask_tiled_3d = tf.tile(fixed_mask_expanded, multiples=[batch_dim_size, 1, 3])

        # target_states_np contains target states only for fixed pins, others are zero
        target_states_tf_const = tf.constant(target_states_np[np.newaxis, :, :], dtype=tf.float32)
        target_states_batch_tf = tf.tile(target_states_tf_const, multiples=[batch_dim_size, 1, 1])

        # tf.where: if mask is True (fixed pin), use target_states_batch_tf value; if False (free spin), use nexttfX_candidate value
        next_with_pins = tf.compat.v1.where(fixed_mask_tiled_3d, target_states_batch_tf, nexttfX_candidate)
        # --- End of modification ---

        update = tfX.assign(next_with_pins)
        init = tf.compat.v1.global_variables_initializer()

    except Exception as e: print(f"[run_simulation] Error during TensorFlow graph construction: {e}"); return None

    config = tf.compat.v1.ConfigProto(); config.gpu_options.allow_growth = True
    X_final = None

    # --- Session execution part is identical to the previous ---
    try:
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(init, feed_dict={X_ph: X_init_np})
            feed = {}
            global myClock
            if myClock is None: myClock = MyClock();
            myClock.init()

            total_iter = paramdict.get("Total_Iteration", 10000)
            sub_iter = paramdict.get("Sub_Iteration", 1000)
            print(f"[run_simulation] Starting simulation loop for {total_iter} iterations...")

            check_ops = { # For checking NaN/Inf
                "energy": dipEnergy,
                "heff_finite": tf.reduce_all(tf.math.is_finite(totalHeff)),
                "next_cand_finite": tf.reduce_all(tf.math.is_finite(nexttfX_candidate)),
                "current_X_finite": tf.reduce_all(tf.math.is_finite(tfX))
            }

            for it in range(total_iter):
                Tvalue = get_Tfeed(it, paramdict)
                feed.update({T: Tvalue})

                if it % (sub_iter * 5) == 0:
                    try:
                        check_results = sess.run(check_ops, feed_dict=feed)
                        if not check_results["current_X_finite"]: print(f"!!! Iter {it}: NaN/Inf in current tfX BEFORE update !!!"); return None
                        if not check_results["heff_finite"]: print(f"!!! Iter {it}: NaN/Inf in totalHeff BEFORE update !!!"); return None
                    except tf.errors.InvalidArgumentError as check_ie:
                         print(f"!!! Iter {it}: NaN/Inf detected during check BEFORE update: {check_ie} !!!"); return None

                sess.run(update, feed_dict=feed)

                if it % sub_iter == 0 or it == total_iter - 1:
                    try:
                        check_results = sess.run(check_ops, feed_dict=feed)
                        energy_v = check_results["energy"]

                        if not check_results["current_X_finite"]: print(f"!!! Iter {it}: NaN/Inf in tfX AFTER update !!!"); return None
                        if not check_results["next_cand_finite"]: print(f"!!! Iter {it}: NaN/Inf in PREVIOUS nexttfX_candidate !!!"); return None

                    except tf.errors.InvalidArgumentError as check_ie:
                         print(f"!!! Iter {it}: NaN/Inf detected during check AFTER update: {check_ie} !!!"); return None

                    try: elapsed = myClock.tictoc(); elapsed_str = f"{elapsed:.3f}" if elapsed is not None else "N/A"
                    except Exception as clock_err: print(f"Warning: Error in myClock.tictoc(): {clock_err}"); elapsed_str = "Error"
                    energy_str = f"{energy_v:.6f}" if np.isfinite(energy_v) else "Invalid"
                    if energy_str == "Invalid": print(f"Warning: Invalid energy detected at iteration {it}.")

                    print(f"Iter {it}/{total_iter} | T={Tvalue:.4f} | Avg E={energy_str} | Step Time: {elapsed_str}s")

            X_final = sess.run(tfX)
            if not np.all(np.isfinite(X_final)): print("!!! Warning: Final spin state contains NaN/Inf !!!"); return None
            print(f"[run_simulation] Simulation finished. Final spin configuration shape = {X_final.shape}")

    except tf.errors.InvalidArgumentError as ie: print(f"[run_simulation] TensorFlow InvalidArgumentError: {ie}. Likely NaN/Inf.")
    except AttributeError as ae: print(f"[run_simulation] AttributeError during session: {ae}")
    except Exception as e: print(f"[run_simulation] An unexpected error occurred: {e}")

    return X_final


###############################################################################
# 9. Simulation Result Saving Function (Modified: Directory Name)
###############################################################################
def save_simulation_results(base_dir, cols, rows, removed_hex_info,
                            edges, centers, skfp, sim_data, pin_interval, dr_value):
    """ Save simulation results (including 4-side pin info) """
    if sim_data is None: print("[save_results] Error: Sim data is None."); return
    print("[save_results] Saving simulation results...")
    try:
        nodes = np.unique(np.concatenate([edges[:, :2], edges[:, 2:]], axis=0), axis=0)
        rotmat = get_rotmat(edges); nnidx = get_nnidx(edges)
    except Exception as e: print(f"[save_results] Warning: Error calc rotmat/nnidx: {e}."); rotmat, nnidx = None, None

    removed_str = "NoneRemoved"
    if removed_hex_info: removed_str = "Rm_" + "_".join([f"c{c}r{r}" for c, r in removed_hex_info])

    # Modify directory name: specify pin info
    dataset_savedir_name = (
        f"SimCols{cols}_Rows{rows}_remove{pin_interval}_DR{dr_value:.2f}"
    )
    dataset_savedir_path = os.path.join(base_dir, dataset_savedir_name); mkdir(dataset_savedir_path)

    try: # File saving
        np.save(os.path.join(dataset_savedir_path,"Edges"), edges)
        np.save(os.path.join(dataset_savedir_path,"CenterOfEdges"), centers)
        np.save(os.path.join(dataset_savedir_path,"Nodes"), nodes)
        np.save(os.path.join(dataset_savedir_path,"Rotmat"), rotmat)
        np.save(os.path.join(dataset_savedir_path,"NNIndices"), nnidx)
        np.save(os.path.join(dataset_savedir_path,"SkFpBool"), skfp)
        np.save(os.path.join(dataset_savedir_path, "TrainData.npy"), sim_data)

        num_samples = len(sim_data); num_spins = sim_data.shape[1]
        with open(os.path.join(dataset_savedir_path, "TrainDataLen.txt"), "w") as f:
            f.write(f"BatchSize = {num_samples}\n"); f.write(f"Spins (N) = {num_spins}\n")
            f.write(f"Removed Hexagons: {removed_hex_info}\n")
            f.write(f"Pinning: All sides, Top Fixed (SkFp=1.0), Others Free (SkFp=2.0)\n") # Add pin info
        print(f"[save_results] Saved results to: {dataset_savedir_path}")
    except Exception as e: print(f"[save_results] Error saving files: {e}")


###############################################################################
# 10. main function (Grid removal and 4-side pin addition)
###############################################################################
def main():
    """ Main execution function: generate grid, remove cells, add 4-side pins, simulate, save """
    print("="*70); print("Hex Lattice Sim: Grid Removal + All Sides Aligned Pins (Top Fixed)"); print("="*70)

    base_dir = os.path.join(os.getcwd(), "D") # New directory
    mkdir(base_dir)
    lattice_a = 1.0; grid_cols = 10 ; grid_rows = 10  # Example list of hexagons to remove
    # hex_to_remove = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,7),(0,8),(0,9),(0,10),(0,11),(0,12),
    #                  (1,0),(1,1),(1,2),(1,3),(1,8),(1,9),(1,10),(1,11),(1,12),(2,0),(2,1),(2,2),
    #                  (2,10),(2,11),(2,12),(3,0),(3,11),(3,12),(5,12),(7,12),(9,12),(11,12),(13,0),(13,11),
    #                  (13,12),(14,0),(14,1),(14,2),(14,10),(14,11),(14,12),(15,0),(15,1),(15,2),(15,3),(15,8),
    #                  (15,9),(15,10),(15,11),(15,12),
    #                  (16,0),(16,1),(16,2),(16,3),(16,4),(16,5),(16,7),(16,8),(16,9),(16,10),(16,11),(16,12)]
    # hex_to_remove = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,0),(1,1),(1,2),(1,4),(1,5),(1,6),(1,7),
    #                  (10,0),(10,1),(10,2),(10,4),(10,5),(10,6),(10,3),(9,0),(9,1),(9,2),(9,4),(9,5),(9,6)
    #                 ,(9,7),(2,0),(4,0),(6,0),(8,0),(2,1),(2,6),(8,1),(8,6)] # No removal
    # hex_to_remove = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(1,2),(1,5),(2,0),(2,1),(3,5),(4,0),(4,1),
    #                 (5,0),(5,1),(5,2),(5,5),(6,0),(6,1),(6,2),(6,3),(6,4)] # No removal
    hex_to_remove = [] # No removal
    pinning_interval = 1; pin_edge_len = lattice_a
    paramdict = {
        "Total_Iteration": 100000, "Sub_Iteration": 10000, "Tstart": 3.0, "Tend": 0.01,
        "Annealing%": 0.95, "Dip": 1.0, "DR": 1000.00, "BatchSize": 30000
    }
    # Intermediate file names
    file_suffix = f"_C{grid_cols}R{grid_rows}_Rm" if hex_to_remove else f"_C{grid_cols}R{grid_rows}_NR"
    original_lattice_csv        = f"Intermediate_Original{file_suffix}.csv"
    lattice_with_top_pins_csv   = f"Intermediate_TopPins{file_suffix}.csv"
    lattice_with_tb_pins_csv    = f"Intermediate_TBPins{file_suffix}.csv"
    lattice_with_tbl_pins_csv   = f"Intermediate_TBLPins{file_suffix}.csv"
    final_lattice_with_pins_csv = f"Final_Lattice_AllPins{file_suffix}.csv" # Final CSV

    global myClock; myClock = MyClock()
    global hex_centers # Global variable to store the list of hexagons after removal

    # Step A: Generate grid
    print("\n--- Step A: Generate Grid ---");
    initial_hex_centers = generate_hexagon_grid(grid_cols, grid_rows, lattice_a)
    if not initial_hex_centers: print("Error: Grid gen failed."); return

    # Step B: Remove specified hexagons
    print("\n--- Step B: Remove Specified Hexagons ---")
    removed_hex_log = []
    hex_centers = list(initial_hex_centers) # Update the global variable hex_centers here
    if hex_to_remove:
        print(f"Attempting to remove: {hex_to_remove}")
        for col_r, row_r in hex_to_remove:
            original_count = len(hex_centers)
            hex_centers = remove_hex_by_colrow(hex_centers, col_r, row_r)
            if len(hex_centers) < original_count: removed_hex_log.append((col_r, row_r))
        print(f"Actual removed: {removed_hex_log}")
    else:
        print("No hexagons specified for removal.")

    # Step C: Create original CSV (based on the removed grid)
    print("\n--- Step C: Save Initial Lattice (Post-Removal) ---");
    save_unique_lattice_csv(hex_centers, lattice_a, original_lattice_csv)

    # Step D: Add pins (in the order of Top -> Bottom -> Left -> Right)
    print("\n--- Step D: Append Aligned Pins Sequentially ---");
    # 1. Top (Fixed, SkFp=2.0)
    # current_csv = append_top_side_pin_aligned(input_csv=original_lattice_csv, output_csv=lattice_with_top_pins_csv, a=lattice_a, pin_edge_length=pin_edge_len, pin_interval=pinning_interval)
    # if current_csv is None: print("Error adding top pins."); return
    # # 2. Bottom (Free, SkFp=2.0)
    # current_csv = append_bottom_side_pin_aligned(input_csv=original_lattice_csv, output_csv=lattice_with_tb_pins_csv, a=lattice_a, pin_edge_length=pin_edge_len, pin_interval=pinning_interval)
    # if current_csv is None: print("Error adding bottom pins."); return
    # 3. Left (Free, SkFp=2.0)
    # current_csv = append_left_side_pin_aligned(input_csv=original_lattice_csv, output_csv=lattice_with_tbl_pins_csv, a=lattice_a, pin_edge_length=pin_edge_len, pin_interval=pinning_interval)
    # if current_csv is None: print("Error adding left pins."); return
    # # 4. Right (Free, SkFp=2.0)
    # final_pinned_csv = append_right_side_pin_aligned(input_csv=current_csv , output_csv=final_lattice_with_pins_csv, a=lattice_a, pin_edge_length=pin_edge_len, pin_interval=pinning_interval)
    # if final_pinned_csv is None: print("Error adding right pins."); return

    # # Step E: Visualize the final lattice
    print("\n--- Step E: Plot Final Lattice with All Pins ---");
    removed_info_str = f"Removed: {removed_hex_log}" if removed_hex_log else "No Removal"
    plot_title = f"Lattice {grid_cols}x{grid_rows}, {removed_info_str}, All Sides Pinned (Top Fixed)"
    plot_lattice_pins_no_shift(original_lattice_csv , title=plot_title) # Use the final CSV file

    # Step F: Run the simulation
    print("\n--- Step F: Run Simulation ---");
    edges_arr, centers_arr, skfp_arr = load_structureCSV(original_lattice_csv) # Use the final CSV file
    if edges_arr is None: print("Error: Failed loading final lattice."); return
    sim_data = run_simulation_with_pins(edges_arr, centers_arr, skfp_arr, paramdict)
    if sim_data is None: print("Error: Simulation failed. Exiting."); return
    print(f"[main] Final dataset shape = {sim_data.shape}")

    # Step G: Save the results
    print("\n--- Step G: Save Results ---");
    save_simulation_results(base_dir, grid_cols, grid_rows, removed_hex_log,
                            edges_arr, centers_arr, skfp_arr,
                            sim_data, pinning_interval, paramdict["DR"])

    print("\n" + "="*70); print(f"Pipeline for {grid_cols}x{grid_rows} grid with removal & all pins Finished!"); print("="*70)

###############################################################################
# 11. Execution Point
###############################################################################
if __name__ == "__main__":
    main()
