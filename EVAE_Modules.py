import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from Other_Modules import get_best_model_path, get_dipbasis

def get_callbacks(save_path, save_best_only=True, save_weights_only=False, period=1):
    return tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=save_best_only, save_weights_only=save_weights_only, period=period)
def get_encoder(x, encoding_dim, hidden_units, activation, batchNorm_bl=False):
    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i], activation=None, name="EncoderDense" + str(i))(x)
        if batchNorm_bl: x = tf.keras.layers.BatchNormalization(name="EncoderBNDense" + str(i))(x)
        x = tf.keras.layers.LeakyReLU(name="EncoderActDense" + str(i))(x) if activation == "leakyrelu" else tf.keras.layers.Activation(activation, name="EncoderActDense" + str(i))(x)
    encoded = tf.keras.layers.Dense(encoding_dim + encoding_dim, activation=None, name="EncoderEnd")(x)
    return encoded
def get_decoder(z, nbofspins, hidden_units, activation, batchNorm_bl=False):
    x = tf.keras.layers.Layer(name="DecoderStart")(z)
    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i], activation=None, name="DecoderDense" + str(i))(x)
        if batchNorm_bl: x = tf.keras.layers.BatchNormalization(name="DecoderBNDense" + str(i))(x)
        x = tf.keras.layers.LeakyReLU(name="DecoderActDense" + str(i))(x) if activation == "leakyrelu" else tf.keras.layers.Activation(activation, name="DecoderActDense" + str(i))(x)
    output = tf.keras.layers.Dense(nbofspins, activation="tanh", name="DecoderEnd")(x)
    return output
def get_encoder_decoder_from_model(evae_model):
    evae_layers = evae_model.layers
    for i in range(len(evae_layers)):
        if evae_layers[i].name == "EncoderEnd":
            encoder_end_index = i
        if evae_layers[i].name == "DecoderStart":
            decoder_start_index = i
        elif evae_layers[i].name == "DecoderEnd":
            decoder_end_index = i
            break

    encoder_input = evae_model.input
    encoder_x = encoder_input
    for i in range(encoder_end_index+1):
        if evae_layers[i].name[:7] == "Encoder": encoder_x = evae_layers[i](encoder_x)
    encoder_output = encoder_x
    encoding_dim = encoder_x.shape[-1]//2
    encoder_model = tf.keras.models.Model(encoder_input, encoder_output)

    decoder_input = tf.keras.layers.Input(shape=(encoding_dim))
    decoder_x = decoder_input
    for i in range(decoder_start_index, decoder_end_index + 1):
        if evae_layers[i].name[:7] == "Decoder": decoder_x = evae_layers[i](decoder_x)
    decoder_output = decoder_x
    decoder_model = tf.keras.models.Model(decoder_input, decoder_output)
    return encoder_model, decoder_model, encoding_dim
def get_EVAE_network(nbofspins, encoding_dim, encoder_hidden_units, decoder_hidden_units, activation, batchNorm_bl=False):
    K.clear_session()
    input = tf.keras.layers.Input(shape=(nbofspins))
    encoded = get_encoder(input, encoding_dim, encoder_hidden_units, activation, batchNorm_bl = batchNorm_bl)
    z_mean, z_log_var = tf.keras.layers.Lambda(split_z, name='z_split')(encoded)
    z = tf.keras.layers.Lambda(samplingTF, output_shape=(encoding_dim,), name='z_Sampling')([z_mean, z_log_var])
    output = get_decoder(z, nbofspins, decoder_hidden_units, activation, batchNorm_bl = batchNorm_bl)
    evae_model = tf.keras.models.Model(input, output, name='EVAE_Model')
    return evae_model
def compile_EVAE_model(evae_model, learning_rate = 1e-3, alpha=1.0, beta=0.0, gamma=0.0, dip_params=[1.0, 0.], coes=None, rotmat=None, nnidx=None, rcloss_type="mse"):
    dip, dipRange = dip_params
    dipBasis = tf.constant(get_dipbasis(coes, dipRange, nnidx))
    rotmat = tf.constant(rotmat)

    input = evae_model.input
    output = evae_model.output
    z_mean, z_log_var = evae_model.get_layer('z_split').output

    reconstruction_loss = get_mse_rcloss(input, output) if rcloss_type == "mse" else get_bc_rcloss(input, output)
    kl_loss = get_klloss(z_mean, z_log_var)
    binary_outputE, hamiltonian_loss = get_hmloss(output, rotmat, dip, dipBasis)
    vae_loss = K.mean(alpha * reconstruction_loss + beta * kl_loss + gamma * hamiltonian_loss)
    evae_model.add_loss(vae_loss)

    spin_Amp = tf.abs(output)
    evae_model.add_metric(reconstruction_loss, name="RcLoss", aggregation='mean')
    evae_model.add_metric(kl_loss, name="KlLoss", aggregation='mean')
    evae_model.add_metric(hamiltonian_loss, name="HmLoss", aggregation='mean')
    evae_model.add_metric(binary_outputE, name="Eout", aggregation='mean')
    evae_model.add_metric(spin_Amp, name="Amp", aggregation='mean')
    evae_model.add_metric(K.abs(z_mean), name="abs_zmean", aggregation='mean')
    evae_model.add_metric(K.exp(0.5 * z_log_var), name="zstd", aggregation='mean')
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    evae_model.compile(optimizer=opt)
    return evae_model
def get_mse_rcloss(input, output):
    return K.mean(K.reshape(K.square(input-output), [K.shape(input)[0], -1]), axis=-1)
def get_bc_rcloss(input, output, epsilon=1.E-7):
    output = K.clip(output, -1. + epsilon, 1. - epsilon)
    zero_one_input = 0.5 * (1. + input)
    zero_one_output = 0.5 * (1. + output)
    return -K.mean(zero_one_input * K.log(zero_one_output) + (1. - zero_one_input) * K.log(1. - zero_one_output), axis=-1)
def get_klloss(z_mean, z_log_var):
    return 0.5 * K.sum(K.square(z_mean) + K.exp(z_log_var) - z_log_var - 1, axis=-1)
def get_hmloss(output, rotmat, dip, dipBasis):
    spin = binary2spinTF_notnorm(output, rotmat)
    dipHeff = get_dipheff(spin, dip, dipBasis)
    hmloss = -tf.reduce_sum(spin * dipHeff / 2., axis=-1)
    hmloss = tf.reduce_mean(tf.reshape(hmloss, [tf.shape(output)[0], -1]), axis=-1)
    norm_spin = binary2spinTF_norm(output, rotmat)
    norm_dipHeff = get_dipheff(norm_spin, dip, dipBasis)
    dipEnergy = -tf.reduce_sum(norm_spin * norm_dipHeff / 2., axis=-1)
    dipEnergy = tf.reduce_mean(tf.reshape(dipEnergy, [tf.shape(output)[0], -1]), axis=-1)
    return dipEnergy, hmloss
def get_energy(tfX, rotmat, dip, dipBasis, average=True):
    norm_spin = binary2spinTF_norm(tfX, rotmat)
    dipHeff = get_dipheff(norm_spin, dip, dipBasis)
    dipEnergy = -tf.reduce_sum(norm_spin * dipHeff / 2., axis=-1)
    return tf.reduce_mean(tf.reshape(dipEnergy, [tf.shape(tfX)[0], -1]), axis=-1) if average else tf.reshape(dipEnergy, [tf.shape(tfX)[0], -1])
def split_z(encoded):
    return tf.split(encoded, 2, axis=-1)
def samplingTF(args):
    batch = K.shape(args[0])[0]
    dim = K.int_shape(args[0])[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return args[0] + K.exp(0.5 * args[1]) * epsilon
def sampling(args):
    z_mean, z_log_var = args
    batch = np.shape(z_mean)[0]
    dim = np.shape(z_mean)[1]
    epsilon = np.random.normal(size=(batch, dim))
    return z_mean + np.exp(0.5 * z_log_var) * epsilon
def binary2spinTF(x, rotmat, norm=False):
    if norm: x = tf.sign(x)
    x = tf.stack([x, tf.zeros_like(x), tf.zeros_like(x)], axis=-1)
    x = tf.squeeze(tf.matmul(tf.expand_dims(x, -2), tf.expand_dims(rotmat, 0)), -2)
    return x
def binary2spinTF_notnorm(x, rotmat):
    x = tf.stack([x, tf.zeros_like(x), tf.zeros_like(x)], axis=-1)
    x = tf.squeeze(tf.matmul(tf.expand_dims(x, -2), tf.expand_dims(rotmat, 0)), -2)
    return x
def binary2spinTF_norm(x, rotmat):
    x = tf.sign(x)
    x = tf.stack([x, tf.zeros_like(x), tf.zeros_like(x)], axis=-1)
    x = tf.squeeze(tf.matmul(tf.expand_dims(x, -2), tf.expand_dims(rotmat, 0)), -2)
    return x
def get_dipheff(tfX, dip, dipBasis):
    sxyz, syxz = tf.transpose(tfX, [2, 1, 0]), tf.transpose(tf.reverse(tf.roll(tfX, shift=1, axis=-1), axis=[-1]), [2, 1, 0])
    dipHeff = dip * tf.transpose(tf.reduce_sum(tf.matmul(dipBasis, tf.stack([sxyz, syxz], axis=0)), axis=0), [2, 1, 0])
    return dipHeff
def get_MultiFlip_MPMC_engine(tfX, T, totalHeff):
    shape = tfX.get_shape().as_list()[:2] + [1]
    rtfX_sign = tf.cast(tf.random.uniform(shape=shape, minval=-1, maxval=1, dtype=tf.int32) * 2 + 1, tfX.dtype)
    rtfX = tfX * rtfX_sign

    emu = -tf.reduce_sum(input_tensor=tf.multiply(tfX, totalHeff), axis=-1, keepdims=True)
    enu = -tf.reduce_sum(input_tensor=tf.multiply(rtfX, totalHeff), axis=-1, keepdims=True)
    deltaE = tf.cast(emu - enu, tf.float64)
    acceptProb = tf.exp(deltaE / T)
    accept_R = tf.random.uniform(tf.shape(emu), 0., 1., dtype=tf.float64)
    deltaSign = tf.cast(1. - tf.nn.relu(-tf.sign(acceptProb - accept_R)), tfX.dtype)
    stepfornu, stepformu = deltaSign, 1. - deltaSign
    return tf.nn.l2_normalize(tfX * stepformu + rtfX * stepfornu, axis=-1)
def get_Singleflip_RMPMC_engine(tfX, T, totalHeff):
    j = tf.random.uniform(shape=[], minval=0, maxval=tfX.shape[1], dtype=tf.int32)
    jth_tfX = tfX[:, j:j + 1, :]
    jth_totalHeff = totalHeff[:, j:j + 1, :]

    rjth_tfX = -tf.identity(jth_tfX)
    emu = -tf.reduce_sum(input_tensor=tf.multiply(jth_tfX, jth_totalHeff), axis=-1, keepdims=True)
    enu = -tf.reduce_sum(input_tensor=tf.multiply(rjth_tfX, jth_totalHeff), axis=-1, keepdims=True)
    deltaE = tf.cast(emu - enu, tf.float64)
    acceptProb = tf.exp(deltaE / T)
    accept_R = tf.random.uniform(tf.shape(emu), 0., 1., dtype=tf.float64)
    deltaSign = tf.cast(1. - tf.nn.relu(-tf.sign(acceptProb - accept_R)), tfX.dtype)
    stepfornu, stepformu = deltaSign, 1. - deltaSign
    next_jth_tfX = jth_tfX * stepformu + rjth_tfX * stepfornu

    next_tfX = tf.reshape(tf.concat([tfX[:, :j, :], next_jth_tfX, tfX[:, j + 1:, :]], axis=1), tfX.shape)
    return tf.nn.l2_normalize(next_tfX, axis=-1)
