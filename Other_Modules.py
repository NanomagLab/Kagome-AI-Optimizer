import os
from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from datetime import datetime

# Note: The following imports require PyQt5 and qimage2ndarray to be installed.
# These are used for visualization and are not essential for the core optimization logic.
try:
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
    from PyQt5.QtCore import QPointF, Qt
    import qimage2ndarray
except ImportError:
    print("Warning: PyQt5 or qimage2ndarray not found. Visualization functions will not be available.")


class MyClock:
    """A simple class for timing code execution blocks."""
    def __init__(self):
        self.prev_time = datetime.now()

    def tick_tock(self):
        """Prints the time elapsed since the last call."""
        now_time = datetime.now()
        print(f"Time elapsed: {now_time - self.prev_time}")
        self.prev_time = now_time

# --- Model Loading Utilities ---

def get_best_model_path(load_path: str, mode: str = "MaxEp") -> str:
    """
    Finds the path to the best model checkpoint in a directory.

    Args:
        load_path: Directory containing model checkpoints.
        mode: Strategy to determine the best model.
              "MaxEp": Choose the model with the maximum epoch number.
              "MinValLoss": Choose the model with the minimum validation loss.

    Returns:
        The full path to the selected model's .index file.
    """
    index_files = [f for f in os.listdir(load_path) if f.endswith(".index")]
    if not index_files:
        raise FileNotFoundError(f"No model .index files found in {load_path}")

    best_file = ""
    if mode == "MaxEp":
        max_epoch = -1
        for f in index_files:
            try:
                epoch = int(f.split("_")[0][len("Ep"):])
                if epoch > max_epoch:
                    max_epoch = epoch
                    best_file = f
            except (ValueError, IndexError):
                continue
    elif mode == "MinValLoss":
        min_val_loss = float('inf')
        for f in index_files:
            try:
                val_loss = float(f.split("_")[1][len("ValLoss"):-len(".index")])
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    best_file = f
            except (ValueError, IndexError):
                continue
    
    if not best_file:
        raise ValueError(f"Could not determine the best model in {load_path} with mode '{mode}'")
        
    return os.path.join(load_path, best_file)

def load_model_parameters(evae_model, model_loaddir_path: str):
    """Loads weights from the best checkpoint into a Keras model."""
    best_model_path = get_best_model_path(model_loaddir_path, mode="MaxEp")
    # Path is truncated to remove ".index" before loading weights
    evae_model.load_weights(best_model_path[:-len(".index")])
    print(f"Model parameters loaded from: {os.path.basename(best_model_path)}")
    return evae_model

# --- Visualization Utilities (PyQt5 based) ---
# These functions are for creating visual representations of the spin ice.

def init_qimage(edges: np.ndarray, width: int, height: int, format=QImage.Format_RGB32) -> Tuple[QImage, float]:
    """Initializes a QImage canvas for plotting."""
    qimage = QImage(width, height, format)
    qimage.fill(QColor(Qt.white).rgb())
    canvas_size = min(width, height)
    max_coord = np.amax(np.abs(edges))
    plot_scale = canvas_size / (2 * (max_coord + 1))
    return qimage, plot_scale

def plot_structure(qimage: QImage, edges: np.ndarray, plot_scale: float, line_width: int = 3) -> QImage:
    """Draws the underlying lattice structure on a QImage."""
    painter = QPainter()
    painter.begin(qimage)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(Qt.gray, line_width))
    
    center_offset = QPointF(qimage.width() / 2., qimage.height() / 2.)
    
    for edge in edges:
        spx, spy, epx, epy = edge
        # Invert y-axis to match Cartesian coordinates from QImage's top-left origin
        start_qpoint = QPointF(spx, -spy) * plot_scale + center_offset
        end_qpoint = QPointF(epx, -epy) * plot_scale + center_offset
        painter.drawLine(start_qpoint, end_qpoint)
        
    painter.end()
    return qimage

def plot_spin(qimage: QImage, vectors: np.ndarray, colors: np.ndarray, edges: np.ndarray, plot_scale: float, line_width: int = 1) -> QImage:
    """Draws spin vectors on the lattice."""
    painter = QPainter()
    painter.begin(qimage)
    painter.setRenderHint(QPainter.Antialiasing)
    
    center_offset = QPointF(qimage.width() / 2., qimage.height() / 2.)
    
    for i in range(len(edges)):
        r, g, b = colors[i]
        painter.setPen(QPen(QColor(int(r), int(g), int(b)), line_width))
        
        center_x = (edges[i][0] + edges[i][2]) / 2.
        center_y = (edges[i][1] + edges[i][3]) / 2.
        vec_x, vec_y = vectors[i][0], vectors[i][1]
        
        spx = -vec_x / 2.7 + center_x
        spy = -vec_y / 2.7 + center_y
        epx = vec_x / 2.7 + center_x
        epy = vec_y / 2.7 + center_y

        # Invert y-axis
        start_qpoint = QPointF(spx, -spy) * plot_scale + center_offset
        end_qpoint = QPointF(epx, -epy) * plot_scale + center_offset
        
        painter.drawLine(start_qpoint, end_qpoint)
        
        # Draw arrowhead
        angle = np.arctan2(end_qpoint.y() - start_qpoint.y(), end_qpoint.x() - start_qpoint.x())
        arrow_size = 5
        arrow_p1 = end_qpoint - QPointF(np.cos(angle - np.pi / 6) * arrow_size, np.sin(angle - np.pi / 6) * arrow_size)
        arrow_p2 = end_qpoint - QPointF(np.cos(angle + np.pi / 6) * arrow_size, np.sin(angle + np.pi / 6) * arrow_size)
        painter.drawLine(end_qpoint, arrow_p1)
        painter.drawLine(end_qpoint, arrow_p2)
        
    painter.end()
    return qimage

# --- Data Conversion and Helper Utilities ---

def mkdir(*args):
    """Creates directories if they do not exist."""
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)

def binary2spin(binary_data: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    """Converts binary spin representation (+1/-1) to 3D Cartesian vectors."""
    binary_data = np.sign(binary_data) # Ensure values are strictly +1 or -1
    # Create a 3D vector along the local x-axis for each spin
    local_vectors = np.stack([binary_data, np.zeros_like(binary_data), np.zeros_like(binary_data)], axis=-1)
    # Rotate the local vectors to the global coordinate system
    # Matmul expects (..., N, M) x (..., M, K) -> (..., N, K)
    # Add batch dimension if not present
    if local_vectors.ndim == 2:
        local_vectors = np.expand_dims(local_vectors, axis=0)
    
    # Reshape for broadcasting: (batch, spins, 1, 3) @ (spins, 3, 3) -> (batch, spins, 1, 3)
    rotated_vectors = np.matmul(np.expand_dims(local_vectors, -2), np.expand_dims(rotmat, 0))
    return np.squeeze(rotated_vectors, axis=-2)

def spin2binary(spin_vectors: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    """Converts 3D Cartesian spin vectors back to the local binary representation."""
    rotmat_T = np.transpose(rotmat, [0, 2, 1])
    # Add batch dimension if not present
    if spin_vectors.ndim == 2:
        spin_vectors = np.expand_dims(spin_vectors, axis=0)
        
    # Rotate global vectors to local coordinate systems
    local_vectors = np.matmul(np.expand_dims(spin_vectors, -2), np.expand_dims(rotmat_T, 0))
    # Take the x-component (the first one) and determine its sign
    binary_data = np.sign(np.squeeze(local_vectors[..., :1], axis=(-2, -1)))
    return binary_data

# --- Physics Calculation Utilities ---

def get_dipbasis(coes: np.ndarray, dr: float, nnidx: np.ndarray = None) -> np.ndarray:
    """Calculates the dipolar interaction tensor (basis) for all spin pairs."""
    print(f"Calculating dipolar basis for {coes.shape[0]} spins...")
    # Calculate displacement vectors r_ij between all pairs of spin centers
    rijs = coes[:, np.newaxis, :] - coes[np.newaxis, :, :]
    
    a, b = rijs[..., 0], rijs[..., 1]
    a2, b2 = np.square(a), np.square(b)
    r_mag = np.sqrt(a2 + b2)
    
    # Avoid division by zero for i=j
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_r = 1. / r_mag
    np.fill_diagonal(inv_r, 0)
    
    # Apply distance cutoff or nearest-neighbor mask
    if nnidx is None:
        mask = (r_mag > 0) & (r_mag < dr)
    else:
        mask = np.zeros((len(coes), len(coes)), dtype=bool)
        for i in range(len(coes)):
            valid_neighbors = nnidx[i][nnidx[i] != -1]
            mask[i, valid_neighbors] = True
            
    inv_r *= mask
    
    inv_r3 = np.power(inv_r, 3)
    inv_r5 = np.power(inv_r, 5)
    
    # Dipolar tensor components (see standard literature for derivation)
    D_xx = -inv_r3 + 3. * a2 * inv_r5
    D_yy = -inv_r3 + 3. * b2 * inv_r5
    D_zz = -inv_r3
    D_xy = 3. * a * b * inv_r5
    
    # This structure seems specific to the project's Hamiltonian formulation
    diDBasisxforSxyz = D_xx
    diDBasisyforSxyz = D_yy
    diDBasiszforSxyz = D_zz
    diDBasisxforSyxz = D_xy
    diDBasisyforSyxz = D_xy
    diDBasiszforSyxz = np.zeros_like(a)
    
    diDBasisforSxyz = np.stack([diDBasisxforSxyz, diDBasisyforSxyz, diDBasiszforSxyz], 0)
    diDBasisforSyxz = np.stack([diDBasisxforSyxz, diDBasisyforSyxz, diDBasiszforSyxz], 0)
    return np.stack([diDBasisforSxyz, diDBasisforSyxz], 0)


# --- Genetic Algorithm Class ---

class GeneticAlgorithm3:
    """
    A Genetic Algorithm implementation for optimizing in a continuous latent space.
    Features rank-based adaptive methods for crossover and mutation.
    """
    def __init__(
            self,
            fitness_func: Callable,
            save_dir: str,
            dim: int = 128,
            num_samples: int = 5000,
            num_elite: int = 0, # Elitism is handled outside in the main loop
            probability_method: str = "linear",
            selection_method: str = "stochastic_remainder_selection",
            crossover_method: str = "rank_based_adaptive",
            mutation_method: str = "rank_based_adaptive",
            k1: float = 0.5
    ):
        """Initializes the Genetic Algorithm with specified strategies."""
        self.fitness_func = fitness_func
        self.save_dir = save_dir
        self.dim = dim
        self.num_samples = num_samples
        self.num_offsprings = num_samples - num_elite
        self.num_elite = num_elite
        self.k1 = k1 # Parameter for rank-based adaptive methods

        # Assign methods based on string identifiers
        if probability_method != "linear":
            raise ValueError("Only 'linear' probability_method is currently supported in this version.")
        self.compute_probability = self.linear_probability

        if selection_method != "stochastic_remainder_selection":
            raise ValueError("Only 'stochastic_remainder_selection' is currently supported.")
        self.selection = self.stochastic_remainder_selection

        if crossover_method != "rank_based_adaptive":
            raise ValueError("Only 'rank_based_adaptive' crossover is currently supported.")
        self.crossover = self.rank_based_adaptive_crossover
        
        if mutation_method not in ["rank_based_adaptive", "rank_based_adaptive_random", "rank_based_adaptive_hybrid"]:
            raise ValueError(f"Unsupported mutation_method: {mutation_method}")
        self.mutation_method_name = mutation_method
        if mutation_method == "rank_based_adaptive":
            self.mutation = self.rank_based_adaptive_mutation


    def linear_probability(self, old_generation, selection_pressure):
        """Calculates selection probability based on linear ranking."""
        self.fitness = self.fitness_func(old_generation)
        self.best_fitness = tf.reduce_max(self.fitness)
        self.average_fitness = tf.reduce_mean(self.fitness)
        self.worst_fitness = tf.reduce_min(self.fitness)

        # probability = (fitness - worst) / (best - worst) * (pressure - 1) + 1
        numerator = self.fitness - self.worst_fitness
        denominator = self.best_fitness - self.worst_fitness
        # Avoid division by zero if all fitness values are the same
        if denominator == 0:
            return tf.ones_like(self.fitness) / tf.cast(tf.shape(self.fitness)[0], tf.float32)

        prob_raw = numerator / denominator * (selection_pressure - 1.0) + 1.0
        return prob_raw / tf.reduce_sum(prob_raw)

    def stochastic_remainder_selection(self, probabilities):
        """Performs stochastic remainder selection to choose parents."""
        total_population_size = self.num_offsprings * 2
        expected_counts = probabilities * tf.cast(total_population_size, tf.float32)
        integer_parts = tf.floor(expected_counts)
        fractional_parts = expected_counts - integer_parts

        # Deterministically select individuals based on the integer part
        deterministic_selection = tf.repeat(tf.range(tf.size(probabilities)), tf.cast(integer_parts, tf.int32))
        remaining_count = total_population_size - tf.size(deterministic_selection)

        # Probabilistically select the rest based on fractional parts
        if tf.reduce_sum(fractional_parts) > 0:
            fractional_probabilities = fractional_parts / tf.reduce_sum(fractional_parts)
            indices = tf.range(tf.size(probabilities))
            probabilistic_selection = tf.random.categorical(tf.math.log([fractional_probabilities]), remaining_count)[0]
            probabilistic_selection = tf.gather(indices, probabilistic_selection)
            selected_indices = tf.concat([deterministic_selection, probabilistic_selection], axis=0)
        else:
            selected_indices = deterministic_selection
            
        return selected_indices

    def rank_based_adaptive_crossover(self, old_generation, parents_indices):
        """Performs crossover where the probability depends on the rank of the dominant parent."""
        parents_fitness = tf.gather(self.fitness, parents_indices)
        # Sort by fitness and pair up for crossover
        sorted_indices = tf.gather(parents_indices, tf.argsort(parents_fitness))[::2]
        parents_indices = tf.random.shuffle(sorted_indices)
        parents_fitness = tf.gather(self.fitness, parents_indices)

        parents_rank = tf.argsort(tf.argsort(parents_fitness))
        
        # Determine dominant and recessive parents for each pair based on rank
        is_first_parent_dominant = parents_rank[::2] > parents_rank[1::2]
        dominant_mask = tf.cast(is_first_parent_dominant, tf.float32)[:, tf.newaxis]
        
        parents = tf.gather(old_generation, parents_indices)
        dom_parents = parents[::2] * dominant_mask + parents[1::2] * (1. - dominant_mask)
        rec_parents = parents[1::2] * dominant_mask + parents[::2] * (1. - dominant_mask)
        
        # Crossover probability is higher for less fit parents
        dominant_rank_float = tf.cast(tf.maximum(parents_rank[::2], parents_rank[1::2]), tf.float32)
        crossover_prob = 2. * self.k1 * (self.num_offsprings - 1. - dominant_rank_float) / (self.num_offsprings - 1.)
        crossover_prob = tf.clip_by_value(crossover_prob, 0., self.k1)

        mask = tf.where(tf.random.uniform((self.num_offsprings // 2, self.dim)) < crossover_prob[:, tf.newaxis], 1., 0.)
        off1 = (1. - mask) * dom_parents + mask * rec_parents
        off2 = (1. - mask) * rec_parents + mask * dom_parents
        offsprings = tf.concat([off1, off2], axis=0)
        return offsprings[:self.num_offsprings]

    def rank_based_adaptive_mutation(self, new_generation, mutation_rate):
        """Performs mutation where the probability is higher for less fit individuals."""
        fitness = self.fitness_func(new_generation)
        ranks = tf.cast(tf.argsort(tf.argsort(fitness)), tf.float32)[..., tf.newaxis]
        
        # Mutation probability is higher for lower ranks (less fit)
        mut_prob = 2. * mutation_rate * (self.num_offsprings - 1. - ranks) / (self.num_offsprings - 1.)
        mut_prob = tf.clip_by_value(mut_prob, 0., mutation_rate)

        mutation_values = tf.random.normal(shape=tf.shape(new_generation))
        mask = tf.random.uniform(shape=tf.shape(new_generation)) < mut_prob
        
        mutated_generation = tf.where(mask, mutation_values, new_generation)
        return mutated_generation

    def step(self, old_generation, selection_pressure, mutation_rate):
        """Performs one step of the genetic algorithm: selection, crossover, mutation."""
        probability = self.compute_probability(old_generation, selection_pressure)
        parents_indices = self.selection(probability)
        new_generation = self.crossover(old_generation, parents_indices)
        new_generation = self.mutation(new_generation, mutation_rate)
        # Note: Elitism is handled in the main loop, not here.
        return new_generation

    def run(
            self,
            total_iteration: int,
            sub_iteration: int,
            initial_mutation_rate: float,
            final_mutation_rate: float,
            **kwargs # Pass-through for unused but provided arguments
    ):
        """
        Runs the genetic algorithm optimization loop.
        
        Note: The 'save_spin_edge_csv' function is expected to be passed via kwargs
              from the main execution script.
        """
        # Linearly anneal mutation rate over the first half of iterations
        half_iteration = total_iteration // 2
        linear_part = tf.linspace(initial_mutation_rate, final_mutation_rate, half_iteration)
        fixed_part = tf.fill([total_iteration - half_iteration], final_mutation_rate)
        mutation_rate_schedule = tf.concat([linear_part, fixed_part], axis=0)
        
        # A fixed selection pressure is used in this implementation
        selection_pressure = 3.0

        # Start with a random population in the latent space
        generation = tf.random.normal((self.num_samples, self.dim))

        start_time = time.time()
        for i, mutation_rate in enumerate(mutation_rate_schedule):
            generation = self.step(generation, selection_pressure, mutation_rate)

            if (i + 1) % sub_iteration == 0:
                elapsed_time = time.time() - start_time
                progress = (i + 1) / total_iteration
                total_estimated_time = elapsed_time / progress if progress > 0 else 0
                eta = total_estimated_time - elapsed_time

                print(
                    f"\rGA Progress: {i + 1}/{total_iteration} | "
                    f"Best Fitness: {self.best_fitness.numpy():.6f} | "
                    f"Avg Fitness: {self.average_fitness.numpy():.6f} | "
                    f"ETA: {eta:.2f}s", end=""
                )

        print("\nGenetic Algorithm run completed.")
        
        # Save the final generation of latent vectors
        final_output_path = os.path.join(self.save_dir, "final_latent_vectors.npy")
        mkdir(self.save_dir)
        np.save(final_output_path, generation.numpy())
        print(f"Final latent vectors saved at: {final_output_path}")

        return generation
