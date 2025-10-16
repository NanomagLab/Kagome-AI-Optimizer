import os
import numpy as np
import tensorflow as tf

# Import custom modules (Assuming these contain necessary functions like spin2binary, get_hmloss etc.)
from CustomLib.EVAE_Modules import *
from CustomLib.Other_Modules import *

# --- Main Logic ---

# 1. GPU Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU memory growth set to True.")
    except RuntimeError as e:
        print(f"GPU Memory Growth Error: {e}")

# 2. Model & Training Hyperparameters
batch_size = 500
encoder_hidden_units = [512, 384, 256]
decoder_hidden_units = [256, 384, 512]
encoding_dim = 128
use_batch_norm = True
learning_rate = 1e-3
beta_kl = 0.005  # Weight for the KL divergence term

# 3. Dataset and Physics Parameters
dataset_dir_name = "Kagome_552_DR1000.00"
dataset_dir_path = os.path.join(os.getcwd(), "D", dataset_dir_name)
dip = 1.0
dipRange = float(dataset_dir_name.split('DR')[-1])
print(f"Dipolar Constant (dip): {dip}, Dipolar Range (dipRange): {dipRange}")

# 4. Data Loading
x_train_initial_3d = np.load(os.path.join(dataset_dir_path, "TrainData.npy"))[:30000]
edges = np.load(os.path.join(dataset_dir_path, "Edges.npy"))
coes = np.load(os.path.join(dataset_dir_path, "CenterOfEdges.npy"))
rotmat = np.load(os.path.join(dataset_dir_path, "Rotmat.npy"))
nnidx = np.load(os.path.join(dataset_dir_path, "NNIndices.npy")) if dipRange == 0.0 else None
nbofspins = x_train_initial_3d.shape[1]
print(f"Dataset imported. Number of spins = {nbofspins}")

# 5. Preprocess Data
x_train = spin2binary(x_train_initial_3d, rotmat)
x_train1 = x_train.reshape(-1, nbofspins)
print("Binarized training data prepared. Shape:", x_train1.shape)

# 6. VAE Model Definition
# Encoder
encoder_inputs = tf.keras.Input(shape=(nbofspins,))
x = encoder_inputs
for units in encoder_hidden_units:
    x = tf.keras.layers.Dense(units)(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
z_mean = tf.keras.layers.Dense(encoding_dim, name="z_mean")(x)
z_log_var = tf.keras.layers.Dense(encoding_dim, name="z_log_var")(x)

# Sampling Layer
class Sampling(tf.keras.layers.Layer):
    """Reparameterization trick: uses z_mean and z_log_var to sample z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = tf.keras.Input(shape=(encoding_dim,))
x = latent_inputs
for units in decoder_hidden_units:
    x = tf.keras.layers.Dense(units)(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
decoder_outputs = tf.keras.layers.Dense(nbofspins, activation='tanh')(x)
decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

# VAE Model Class
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            recon_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(data, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

# 7. Initialize and Compile VAE
vae = VAE(encoder, decoder, beta=beta_kl)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1)

# 8. Define Fitness Function for GA
dipBasis = get_dipbasis(coes, dipRange, nnidx)
def fitness_func(latent_code):
    """Fitness function for GA: negative of the total dipolar energy."""
    decoded_spins = vae.decoder(latent_code)
    energy, _ = get_hmloss(decoded_spins, rotmat, dip, dipBasis)
    return -energy

# --- Main Optimization Loop ---
num_cycles = 200
deep_search_period = 20
num_lowest_to_keep = 50

for cycle in range(num_cycles):
    is_deep_search_cycle = (cycle + 1) % deep_search_period == 0
    if is_deep_search_cycle:
        ga_iterations, initial_mutation_rate, final_mutation_rate = 20000, 1.0, 0.01
        print(f"\n>>> Starting Deep Search Cycle {cycle + 1} ({ga_iterations} iterations) <<<")
    else:
        ga_iterations, initial_mutation_rate, final_mutation_rate = 2000, 1.0, 1.0
        print(f"\n--- Starting Training Cycle {cycle + 1}/{num_cycles} ---")

    # --- Stage 1: Train VAE on the current dataset ---
    vae.fit(x_train1, epochs=100, batch_size=batch_size, callbacks=[early_stopping], verbose=2)
    print(f"Cycle {cycle + 1}: VAE training complete.")

    # --- Stage 2: Explore latent space with Genetic Algorithm ---
    ga = GeneticAlgorithm3(
        fitness_func,
        dim=encoding_dim,
        num_samples=5000,
        num_elite=0,
        probability_method="linear",
        selection_method="stochastic_remainder_selection",
        crossover_method="rank_based_adaptive",
        mutation_method="rank_based_adaptive",
        k1=0.5
    )
    optimized_latent_codes = ga.run(
        generator=vae.decoder,
        total_iteration=ga_iterations,
        sub_iteration=100,
        # The following parameters might be needed by ga.run for internal logging or checks
        dataset_dir_name=dataset_dir_name, rotmat=rotmat, dip=dip, dipBasis=dipBasis,
        edges=edges, coes=coes, get_hmtotalloss=get_hmloss,
        initial_mutation_rate=initial_mutation_rate,
        final_mutation_rate=final_mutation_rate
    )

    # --- Stage 3: Analyze GA results and find elite configurations ---
    ga_generated_spins = vae.decoder(optimized_latent_codes)
    ga_spins_binary = tf.where(ga_generated_spins > 0, 1.0, -1.0)
    ga_energies, _ = get_hmloss(ga_spins_binary, rotmat, dip, dipBasis)
    ga_energies_np = ga_energies.numpy()

    min_energy_this_cycle = np.min(ga_energies_np)
    print(f"Cycle {cycle + 1} GA Results - Mean E: {np.mean(ga_energies_np):.6f}, Min E: {min_energy_this_cycle:.6f}")

    def find_lowest_indices(arr, n):
        """Efficiently finds the indices of the n smallest values in arr."""
        n_eff = min(n, len(arr))
        if n_eff == 0:
            return np.array([], dtype=int)
        indices = np.argpartition(arr, n_eff - 1)[:n_eff]
        return indices

    lowest_indices = find_lowest_indices(ga_energies_np, num_lowest_to_keep)
    
    # --- Stage 4: Update the training dataset for the next cycle (The Virtuous Cycle) ---
    x_train_selected_elites = tf.gather(ga_spins_binary, lowest_indices, axis=0)
    
    num_to_replace = x_train_selected_elites.shape[0]
    if num_to_replace > 0:
        # Replace the oldest data with the newly found elite states
        x_train1 = x_train1[num_to_replace:]
        x_train1 = np.concatenate((x_train1, x_train_selected_elites.numpy()), axis=0)
        np.random.shuffle(x_train1)
        print(f"Dataset updated with {num_to_replace} new elite configurations.")
    else:
        print("No new configurations to update dataset with.")


# --- Final step: After the loop, find and save the absolute best configuration ---
print("\nAll optimization cycles complete. Finding the final ground state...")

# Use the final trained decoder to generate candidates from the last GA run
final_spins_binary = tf.where(vae.decoder(optimized_latent_codes) > 0, 1.0, -1.0)
final_energies, _ = get_hmloss(final_spins_binary, rotmat, dip, dipBasis)
final_energies_np = final_energies.numpy()

# Find the single best configuration
best_index = np.argmin(final_energies_np)
best_energy = final_energies_np[best_index]
best_spin_binary = final_spins_binary[best_index]

print(f"Absolute best energy found: {best_energy:.8f}")

# Save the final best spin configuration to a file
best_spin_3d = binary2spin(best_spin_binary[tf.newaxis, ...], rotmat)[0].numpy()
final_result_dir = "final_results"
os.makedirs(final_result_dir, exist_ok=True)
final_save_path = os.path.join(final_result_dir, f"ground_state_{dataset_dir_name}.npy")
np.save(final_save_path, best_spin_3d)

print(f"Final ground state configuration saved to: {final_save_path}")
