import os
from typing import Callable
import tensorflow.keras.backend as K
import cv2
import numpy as np
import pandas as pd
import qimage2ndarray
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import QPointF, Qt, QEventLoop, QTimer
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
import tensorflow as tf
import time

class MyClock():
    def init(self):
        self.prevtime = datetime.now()
    def tictoc(self):
        self.nowtime = datetime.now()
        print("Taken Time : %s" % (str(self.nowtime - self.prevtime)))
        self.prevtime = datetime.now()

def get_best_model_path(load_path, mode="MaxEp"):
    listdir_parent_dir = os.listdir(load_path)
    index_files = [name for name in listdir_parent_dir if name[-len(".index"):] == ".index"]
    if mode == "MaxEp":
        index, max_epoch = 0, 0
        for i, name in enumerate(index_files):
            epoch = int(name.split("_")[0][len("Ep"):])
            if max_epoch < epoch:
                max_epoch = epoch
                index = i
    elif mode == "MinValLoss":
        index, min_valloss = 0, 100
        for i, name in enumerate(index_files):
            valloss = float(name.split("_")[1][len("ValLoss"):-len(".index")])
            if min_valloss > valloss:
                min_valloss = valloss
                index = i
    return os.path.join(load_path, index_files[index])
def load_model_parameters(evae_model, model_loaddir_path):
    best_model_path = get_best_model_path(model_loaddir_path, mode="MaxEp")
    evae_model.load_weights(best_model_path[:-len(".index")])
    print(os.path.basename(best_model_path) + " Model Parameter loaded")
    return evae_model
def init_qimage(edges, canvasWindowWidth, canvasWindowHeight, format=QImage.Format_RGB32):
    def plotScaleChange(edges, canvasWindowWidth):
        max_v = np.amax(abs(edges))
        return canvasWindowWidth // (2 * (max_v + 1))
    qimage = QImage(canvasWindowWidth, canvasWindowHeight, format)
    qimage.fill(QColor(Qt.white).rgb())
    canvasWindowSize = np.min([canvasWindowWidth, canvasWindowHeight])
    plotScale = plotScaleChange(edges, canvasWindowSize)
    return qimage, plotScale
def plot_structure(qimage, edges, plotScale, lineWidth=3, position=False, circle_r = 2.):
    painter = QPainter()
    painter.begin(qimage)
    painter.setRenderHint(QPainter.Antialiasing)
    for i in range(len(edges)):
        painter.setPen(QPen(Qt.gray, lineWidth))
        spx, spy = edges[i][0], edges[i][1]
        epx, epy = edges[i][2], edges[i][3]
        spy, epy = -spy, -epy # to fit the ij corrd to xy corrd
        startQpoint = QPointF(spx, spy) * plotScale + QPointF(qimage.width() / 2., qimage.height() / 2.)
        endQpoint = QPointF(epx, epy) * plotScale + QPointF(qimage.width() / 2., qimage.height() / 2.)
        painter.drawLine(startQpoint, endQpoint)
        if position:
            painter.setBrush(Qt.black)
            painter.setPen(QPen(Qt.black, lineWidth))
            px, py = (edges[i][0]+edges[i][2])/2., (edges[i][1]+edges[i][3])/2.
            py = -py  # to fit the ij corrd to xy corrd
            poleQpoint = QPointF(px, py) * plotScale + QPointF(qimage.width() / 2., qimage.height() / 2.)
            painter.drawEllipse(poleQpoint, circle_r, circle_r)
    painter.end()
    return qimage
def plot_spin(qimage, vectors, colors, edges, plotScale, lineWidth=1, arrowHead=True, ahRatio=0.15, zero_plot=True):
    painter = QPainter()
    painter.begin(qimage)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(Qt.white)
    for i in range(len(edges)):
        painter.setPen(QPen(QColor(colors[i][0], colors[i][1], colors[i][2]), lineWidth))
        spx = -vectors[i][0] / 2.7 + (edges[i][0] + edges[i][2]) / 2.
        spy = -vectors[i][1] / 2.7 + (edges[i][1] + edges[i][3]) / 2.
        epx = vectors[i][0] / 2.7 + (edges[i][0] + edges[i][2]) / 2.
        epy = vectors[i][1] / 2.7 + (edges[i][1] + edges[i][3]) / 2.
        spy, epy = -spy, -epy # to fit the ij corrd to xy corrd
        rho = np.sqrt(vectors[i][0] ** 2 + vectors[i][1] ** 2)
        startQpoint = QPointF(spx, spy) * plotScale + QPointF(qimage.width() / 2., qimage.height() / 2.)
        endQpoint = QPointF(epx, epy) * plotScale + QPointF(qimage.width() / 2., qimage.height() / 2.)
        if zero_plot:
            painter.drawLine(startQpoint, endQpoint)
        else:
            if startQpoint!=endQpoint: painter.drawLine(startQpoint, endQpoint)
        if arrowHead:
            angle = np.arctan2(epy - spy, epx - spx)
            ah1_startQpoint = endQpoint + QPointF(-np.cos(angle) - np.sin(angle), -np.sin(angle) + np.cos(angle)) * ahRatio * rho * plotScale
            ah2_startQpoint = endQpoint + QPointF(-np.cos(angle) + np.sin(angle), -np.sin(angle) - np.cos(angle)) * ahRatio * rho * plotScale
            painter.drawLine(ah1_startQpoint, endQpoint)
            painter.drawLine(ah2_startQpoint, endQpoint)
    painter.end()
    return qimage
def plot_pole(qimage, total_nodes, poles, plotScale, pole_r_scale=2.):
    painter = QPainter()
    painter.begin(qimage)
    painter.setRenderHint(QPainter.Antialiasing)
    for i in range(len(total_nodes)):
        if poles[i] == 0: r, g, b = 128, 128, 128
        elif poles[i] > 0: r, g, b = 255, 0, 0
        elif poles[i] < 0: r, g, b = 0, 0, 255
        circle_r = np.abs(poles[i]*pole_r_scale)+2.
        painter.setBrush(QColor(r, g, b))
        px, py = total_nodes[i][0], total_nodes[i][1]
        py = -py # to fit the ij corrd to xy corrd
        poleQpoint = QPointF(px, py) * plotScale + QPointF(qimage.width() / 2., qimage.height() / 2.)
        painter.drawEllipse(poleQpoint, circle_r, circle_r)
    painter.end()
    return qimage
def plot_m(qimage, ms, colors, total_nodes, plotScale, lineWidth=1, arrowHead=True, ahRatio=0.1, zero_plot=True):
    painter = QPainter()
    painter.begin(qimage)
    painter.setRenderHint(QPainter.Antialiasing)
    for i in range(len(total_nodes)):
        painter.setPen(QPen(QColor(colors[i][0], colors[i][1], colors[i][2]), lineWidth))
        spx = -ms[i][0] / 4. + total_nodes[i][0]
        spy = -ms[i][1] / 4. + total_nodes[i][1]
        epx = ms[i][0] / 4. + total_nodes[i][0]
        epy = ms[i][1] / 4. + total_nodes[i][1]
        spy, epy = -spy, -epy  # to fit the ij corrd to xy corrd
        rho = np.sqrt(ms[i][0] ** 2 + ms[i][1] ** 2)
        startQpoint = QPointF(spx, spy) * plotScale + QPointF(qimage.width() / 2., qimage.height() / 2.)
        endQpoint = QPointF(epx, epy) * plotScale + QPointF(qimage.width() / 2., qimage.height() / 2.)
        if zero_plot:
            painter.drawLine(startQpoint, endQpoint)
        else:
            if startQpoint != endQpoint: painter.drawLine(startQpoint, endQpoint)
        if arrowHead:
            angle = np.arctan2(epy - spy, epx - spx)
            ah1_startQpoint = endQpoint + QPointF(-np.cos(angle) - np.sin(angle), -np.sin(angle) + np.cos(angle)) * ahRatio * rho * plotScale
            ah2_startQpoint = endQpoint + QPointF(-np.cos(angle) + np.sin(angle), -np.sin(angle) - np.cos(angle)) * ahRatio * rho * plotScale
            painter.drawLine(ah1_startQpoint, endQpoint)
            painter.drawLine(ah2_startQpoint, endQpoint)
    painter.end()
    return qimage
def spin2rgb(X):
    def normalize(v, axis=-1):
        norm = np.linalg.norm(v, ord=2, axis=axis, keepdims=True)
        return norm, np.nan_to_num(v / norm)
    def hsv2rgb(hsv):
        hsv = np.asarray(hsv)
        if hsv.shape[-1] != 3: raise ValueError("Last dimension of input array must be 3; " "shape {shp} was found.".format(shp=hsv.shape))
        in_shape = hsv.shape
        hsv = np.array(hsv, copy=False, dtype=np.promote_types(hsv.dtype, np.float32), ndmin=2)

        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        r, g, b = np.empty_like(h), np.empty_like(h), np.empty_like(h)

        i = (h * 6.0).astype(int)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        idx = i % 6 == 0
        r[idx], g[idx], b[idx] = v[idx], t[idx], p[idx]

        idx = i == 1
        r[idx], g[idx], b[idx] = q[idx], v[idx], p[idx]

        idx = i == 2
        r[idx], g[idx], b[idx] = p[idx], v[idx], t[idx]

        idx = i == 3
        r[idx], g[idx], b[idx] = p[idx], q[idx], v[idx]

        idx = i == 4
        r[idx], g[idx], b[idx] = t[idx], p[idx], v[idx]

        idx = i == 5
        r[idx], g[idx], b[idx] = v[idx], p[idx], q[idx]

        idx = s == 0
        r[idx], g[idx], b[idx] = v[idx], v[idx], v[idx]

        rgb = np.stack([r, g, b], axis=-1)
        return rgb.reshape(in_shape)
    norm, normed_X = normalize(X)
    norm = np.clip(norm, 0, 1)
    X = norm * normed_X
    sxmap, symap, szmap = np.split(X, 3, axis=-1)
    szmap = 0.5 * szmap + (norm / 2.)
    H = np.clip(-np.arctan2(sxmap, -symap) / (2 * np.pi) + 0.5, 0, 1)
    S = np.clip(2 * np.minimum(szmap, norm - szmap), 0, norm)
    V = np.clip(2 * np.minimum(norm, szmap + norm / 2.) - 1.5 * norm + 0.5, 0.5 - 0.5 * norm, 0.5 + 0.5 * norm)
    img = np.concatenate((H, S, V), axis=-1)
    for i, map in enumerate(img): img[i] = hsv2rgb(map)
    return img*255
def spin2black(X):
    return np.zeros_like(X)
def spin2red(X):
    return np.ones_like(X) * np.array([200., 0., 0.])
def spin2green(X):
    return np.ones_like(X) * np.array([0., 200., 0.])
def spin2blue(X):
    return np.ones_like(X) * np.array([0., 0., 200.])
def spin2gray(X):
    return np.ones_like(X)* 128
def gridplot_colorimgs(rgbs, gridx, gridy, figsizex=10, figsizey=10, interpolation="None", axisoff=False):
    fig = plt.figure(figsize=(figsizex, figsizey))
    gs = gridspec.GridSpec(gridx, gridy)
    gs.update(wspace=0.05, hspace=0.05)
    for i, rgb in enumerate(rgbs):
        ax = plt.subplot(gs[i])
        if axisoff: plt.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.imshow(rgb, interpolation=interpolation)
    plt.show()
    plt.close(fig)
def mkdir(*args):
    for arg in args:
        if not os.path.exists(arg) == True:
            os.makedirs(arg)
def load_structureCSV(load_path):
    data = pd.read_csv(load_path, index_col=0)
    if list(data.keys())[-1] == "SkFpIndex":
        np_data = data.values
        edges = np_data[:, :4].astype(np.float32)
        coes = np_data[:, 4:6].astype(np.float32)
        skfp = np_data[:, -1].astype(np.float32)
        return edges, coes, skfp
    else:
        np_data = data.values
        edges = np_data[:, :4].astype(np.float32)
        coes = np_data[:, 4:6].astype(np.float32)
        return edges, coes
def norm_binary(binary):
    test = -np.sign(binary)
    test = test * (test > 0.)
    test = 2.*(1. - test) - 1.
    return test
def binary2spin(x, rotmat, norm=False):
    if norm: x = norm_binary(x)
    x = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=-1)
    x = np.squeeze(np.matmul(np.expand_dims(x, -2), np.expand_dims(rotmat, 0)), -2)
    return x
def spin2binary(x, rotmat):
    rotmat_T = np.transpose(rotmat, [0, 2, 1])
    binarydata = norm_binary(np.squeeze(np.matmul(np.expand_dims(x, -2), np.expand_dims(rotmat_T, 0))[..., :1], (-2, -1)))
    return binarydata
def get_rotmat(edges):
    phi = np.arctan2(edges[:, 3] - edges[:, 1], edges[:, 2] - edges[:, 0])
    cosphi, sinphi, zeros = np.cos(phi), np.sin(phi), np.zeros_like(phi)
    rotmat = np.stack([np.stack([cosphi, -sinphi, zeros], -1), np.stack([sinphi, cosphi, zeros], -1), np.stack([zeros, zeros, zeros], -1)], -1)
    return rotmat

def get_nnidx(edges):
    edges = np.round(np.stack([edges[:, :2], edges[:, 2:]], axis=1), 1)
    ls_edges = edges.tolist()
    len_ls_edges = len(ls_edges)
    ls_nn_indices = [[i for i in range (len_ls_edges) if i!=j and (ls_edges[j][0] in ls_edges[i] or ls_edges[j][1] in ls_edges[i])] for j in range (len_ls_edges)]
    maxLength = max(len(sub) for sub in ls_nn_indices)
    nnidx = -1*np.ones([len_ls_edges, maxLength])
    for i in range (len_ls_edges):
        for j in range (len(ls_nn_indices[i])):
            nnidx[i][j] = ls_nn_indices[i][j]
    return nnidx

def get_dipbasis(coes, dr, nnidx=None):
    print("coes=",coes.shape)
    def get_rijs(coes):
        coes1 = np.tile(coes[None], [len(coes), 1, 1])
        coes2 = np.transpose(coes1, [1, 0, 2])
        return coes1 - coes2

    rijs = get_rijs(coes)
    a, b = rijs[..., 0], rijs[..., 1]
    ab = a * b
    a2, b2 = np.square(a), np.square(b)
    sqrta2b2 = np.sqrt(a2 + b2)
    rsqrta2b2 = 1. / sqrta2b2
    np.fill_diagonal(rsqrta2b2, 0)
    if nnidx is None:
        step = 1. - np.clip(-np.sign(dr - sqrta2b2), a_min=0, a_max=1)
        np.fill_diagonal(step, 0)
    else:
        step = np.array([[1 if i in nnidx[j] else 0 for i in range(len(coes))] for j in range(len(coes))])
    step = step.astype(np.float32)
    rsqrta2b2 *= step

    r3, r5 = np.power(rsqrta2b2, 3), np.power(rsqrta2b2, 5)

    diDBasisxforSxyz = -1. * r3 + 3. * a2 * r5
    diDBasisyforSxyz = -1. * r3 + 3. * b2 * r5
    diDBasiszforSxyz = -1. * r3
    diDBasisxforSyxz = 3. * ab * r5
    diDBasisyforSyxz = 3. * ab * r5
    diDBasiszforSyxz = np.zeros_like(a)
    diDBasisforSxyz = np.stack([diDBasisxforSxyz, diDBasisyforSxyz, diDBasiszforSxyz], 0)
    diDBasisforSyxz = np.stack([diDBasisxforSyxz, diDBasisyforSyxz, diDBasiszforSyxz], 0)
    return np.stack([diDBasisforSxyz, diDBasisforSyxz], 0)

def get_dippair(coes, dr, nnidx=None):
    def get_rijs(coes):
        coes1 = np.tile(coes[None], [len(coes), 1, 1])
        coes2 = np.transpose(coes1, [1, 0, 2])
        return coes1 - coes2

    rijs = get_rijs(coes)
    a, b = rijs[..., 0], rijs[..., 1]
    a2, b2 = np.square(a), np.square(b)
    sqrta2b2 = np.sqrt(a2 + b2)
    if nnidx is None:
        step = 1. - np.clip(np.sign(sqrta2b2 - dr), a_min=0, a_max=1)
        np.fill_diagonal(step, 0)
    else:
        step = np.array([[1 if i in nnidx[j] else 0 for i in range(len(coes))] for j in range(len(coes))])
    xi = np.identity(len(coes))
    return xi, step
def get_IC(length, rotmat, batch_size=1):
    def randIC(length, rotmat, batch_size=1):
        R, zeropad = np.sign(np.random.uniform(size=[batch_size, length, 1], low=-1., high=1.)), np.zeros([batch_size, length, 1])
        X = np.concatenate([R, zeropad, zeropad], axis=-1)
        X = np.transpose(np.matmul(np.transpose(X, [1, 0, 2]), rotmat), [1, 0, 2])
        return X
    X = randIC(length, rotmat, batch_size)
    return X
def penroseGroundIC(length, coes, filename="GS1030.csv"):
    load_path = os.path.join(os.getcwd(), "ShiGroundStates", filename)
    gs = pd.read_csv(load_path, header=None)
    np_gs = gs.values.astype(np.float32)
    X = np.zeros([1, length, 3], np.float32)
    for i in range(length):
        coex, coey = np.round(coes[i][0], 2), np.round(coes[i][1], 2)
        for test in np_gs:
            if np.round(test[0], 2) == coex and np.round(test[1], 2) == coey:
                X[0, i, 0] = np.cos(np.pi*test[2]/180.)
                X[0, i, 1] = np.sin(np.pi*test[2]/180.)
                X[0, i, 2] = 0.0
                break
    return X
def get_Tfeed(it, paramdict):
    if it < paramdict["Annealing%"] * paramdict["Total_Iteration"]:
        Tfeed = paramdict["Tstart"] - (paramdict["Tstart"] - paramdict["Tend"]) * (it / (paramdict["Annealing%"] * paramdict["Total_Iteration"]))
    else:
        Tfeed = paramdict["Tend"]
    return Tfeed
def get_ssc(binary_input):
    norm_binary = binary_input
    len_input = len(norm_binary)
    ssc_x_test = np.tile(norm_binary[None], [len_input, 1, 1]) * np.tile(norm_binary[:, None], [1, len_input, 1])
    ssc_x_test = np.abs(np.sum(ssc_x_test, -1))
    ssc_x_test /= norm_binary.shape[1]
    np.fill_diagonal(ssc_x_test, 0)
    ssc_mean = np.sum(ssc_x_test) / (len_input * (len_input - 1))
    ssc2_mean = np.sum(np.square(ssc_x_test)) / (len_input * (len_input - 1))
    ssc_std = np.sqrt(ssc2_mean - np.square(ssc_mean))
    return ssc_mean, ssc_std
def check_showskfpmode(skfpbl, showskbl=True, showfpbl=True, concat=True):
    skidx = np.reshape(np.argwhere(skfpbl == 1), [-1]) if showskbl else np.array([])
    fpidx = np.reshape(np.argwhere(skfpbl == 0), [-1]) if showfpbl else np.array([])
    return np.concatenate([skidx, fpidx], 0).astype(int) if concat else [skidx.astype(int), fpidx.astype(int)]
def save_spin_config(sc, edges, coes, save_path):
    df_sc = pd.DataFrame(sc[:, :2])
    df_edges = pd.DataFrame(edges)
    df_coes = pd.DataFrame(coes)
    df_total = pd.concat([df_sc, df_edges, df_coes], axis=1)
    df_total.columns = ["Sx", "Sy", "X1", "Y1", "X2", "Y2", "XC", "YC"]
    df_total.to_csv(save_path)
def get_diff_mode_color(b1, b2):
    cond = np.equal(b1, b2)[:, None]
    return np.where(cond, np.array([0, 0, 255]), np.array([255, 0, 0]))
def build_nnidx(edges):
    """
    edges.shape == (E,4):
      each row = [X1, Y1, X2, Y2], edge i is connected from (X1,Y1) to (X2,Y2).
    If they share at least one endpoint, they are considered 'adjacent'.
    nnidx.shape == (E,K): K is the maximum number of adjacent edges.
      nnidx[i] -> list of edge indices that are neighbors to edge i (if none, -1).
    """
    # Reshape to (E,2,2) and round coordinates appropriately
    edges_2d = np.round(np.stack([edges[:, :2], edges[:, 2:]], axis=1), 4)
    E_ = len(edges_2d)
    ls_edges = edges_2d.tolist()

    # For each edge j, find neighboring edges i (sharing an endpoint)
    ls_nn_indices = []
    for j in range(E_):
        points_j = ls_edges[j]  # [[x1_j,y1_j],[x2_j,y2_j]]
        nn_for_j = []
        for i in range(E_):
            if i == j:
                continue
            # Adjacent if any point (endpoint) is the same
            if (ls_edges[i][0] in points_j) or (ls_edges[i][1] in points_j):
                nn_for_j.append(i)
        ls_nn_indices.append(nn_for_j)

    # Fill an array initialized with -1 with the neighbor indices
    max_len = max(len(sub) for sub in ls_nn_indices)
    nnidx = -1 * np.ones([E_, max_len], dtype=int)
    for j, neighbors in enumerate(ls_nn_indices):
        for k, idx in enumerate(neighbors):
            nnidx[j, k] = idx

    return nnidx

def compute_selection_probability(losses, iteration, total_iterations, sp_min=1.0, sp_max=2.0):
    """
    Compute selection probabilities with rank-based method and increasing selection pressure.

    Args:
    losses: Tensor of loss values.
    iteration: Current iteration number.
    total_iterations: Total number of iterations.
    sp_min: Minimum selection pressure (default: 1.0).
    sp_max: Maximum selection pressure (default: 2.0).

    Returns:
    Tensor of selection probabilities.
    """
    n = tf.shape(losses)[0]
    ranks = tf.argsort(tf.argsort(losses)) + 1  # Lower loss gets higher rank
    ranks = tf.cast(ranks, tf.float32)

    # Dynamically adjust selection pressure
    sp = sp_min + (sp_max - sp_min) * (iteration / total_iterations)

    # Calculate probabilities
    probabilities = (1 / tf.cast(n, tf.float32)) * (
            sp - (2 * sp - 2) * (ranks - 1) / (tf.cast(n, tf.float32) - 1)
    )
    probabilities = tf.clip_by_value(probabilities, 0.0, 1.0)  # Ensure probabilities are non-negative
    probabilities /= tf.reduce_sum(probabilities)  # Normalize probabilities
    return probabilities


def get_Jhmloss(outputs, h_params, img_size=128):
    exJ, DMN, hextz = h_params
    totalW3x3, hext = get_w3x3([exJ, DMN, True]), get_extField([img_size, img_size, hextz])
    return cal_energy_tf([outputs, totalW3x3, hext])
def cal_energy_tf(args):
    tfX, totalW3x3, hext = args
    def getHeff(X, filter, strides=[1, 1, 1, 1], padding="VALID"):
        return tf.nn.conv2d(input=X, filters=filter, strides=strides, padding=padding)
    padded_input = pad_wrap(tfX, 1, 1)
    totalHeff3x3 = getHeff(padded_input, totalW3x3)
    return K.mean(tf.reshape(-tf.reduce_sum(tf.multiply(tfX, hext + totalHeff3x3 / 2.), axis=-1), [K.shape(tfX)[0], -1]), axis=-1)
def pad_wrap(tfX, pad_i, pad_j, boundary="Periodic"):
    if boundary == "Periodic":
        M1 = tf.concat([tfX[:, :, -pad_j:, :], tfX, tfX[:, :, 0:pad_j, :]], 2)
        M1 = tf.concat([M1[:, -pad_i:, :, :], M1, M1[:, 0:pad_i, :, :]], 1)
    elif boundary == "Not Periodic":
        M1 = tf.concat(
            [tf.zeros_like(tfX[:, :, -pad_j:, :]), tfX, tf.zeros_like(tfX[:, :, 0:pad_j, :])], 2)
        M1 = tf.concat([tf.zeros_like(M1[:, -pad_i:, :, :]), M1, tf.zeros_like(M1[:, 0:pad_i, :, :])], 1)
    elif boundary == "X Periodic":
        M1 = tf.concat([tfX[:, :, -pad_j:, :], tfX, tfX[:, :, 0:pad_j, :]], 2)
        M1 = tf.concat([tf.zeros_like(M1[:, -pad_i:, :, :]), M1, tf.zeros_like(M1[:, 0:pad_i, :, :])], 1)
    elif boundary == "Y Periodic":
        M1 = tf.concat(
            [tf.zeros_like(tfX[:, :, -pad_j:, :]), tfX, tf.zeros_like(tfX[:, :, 0:pad_j, :])], 2)
        M1 = tf.concat([M1[:, -pad_i:, :, :], M1, M1[:, 0:pad_i, :, :]], 1)
    return M1

def \
        get_w3x3(args):
    exJ, DMN, self_bl = args
    w3x3 = np.zeros([3, 3, 3, 3], dtype=np.float32)
    for i in range(3):
        for j in range(3):
            if (i == 0 and j == 1):
                w3x3[i, j, 0, 0] += exJ
                w3x3[i, j, 1, 1] += exJ
                w3x3[i, j, 2, 2] += exJ
                w3x3[i, j, 2, 1] += -DMN
                w3x3[i, j, 1, 2] += DMN
            if (i == 2 and j == 1):
                w3x3[i, j, 0, 0] += exJ
                w3x3[i, j, 1, 1] += exJ
                w3x3[i, j, 2, 2] += exJ
                w3x3[i, j, 2, 1] += DMN
                w3x3[i, j, 1, 2] += -DMN
            if (i == 1 and j == 0):
                w3x3[i, j, 0, 0] += exJ
                w3x3[i, j, 1, 1] += exJ
                w3x3[i, j, 2, 2] += exJ
                w3x3[i, j, 2, 0] += DMN
                w3x3[i, j, 0, 2] += -DMN
            if (i == 1 and j == 2):
                w3x3[i, j, 0, 0] += exJ
                w3x3[i, j, 1, 1] += exJ
                w3x3[i, j, 2, 2] += exJ
                w3x3[i, j, 2, 0] += -DMN
                w3x3[i, j, 0, 2] += DMN
            if (i == 1 and j == 1):
                if self_bl:
                    w3x3[i, j, 0, 0] += -4*exJ
                    w3x3[i, j, 1, 1] += -4*exJ
                    w3x3[i, j, 2, 2] += -4*exJ
    return w3x3

def get_extField(args):
    i_size, j_size, hextz = args
    tf_extH = tf.ones([1, i_size, j_size, 3], dtype=tf.float32)
    tf_extH *= np.array([0., 0., hextz])
    return tf_extH
def build_nnidx1(edges, decimal=4):
    """
    edges.shape == (E,4):
      each row = [X1, Y1, X2, Y2], edge i is connected from (X1,Y1) to (X2,Y2).
    If they share at least one endpoint, they are considered 'adjacent'.
    nnidx.shape == (E,K): K is the maximum number of adjacent edges.
      nnidx[i] -> list of edge indices that are neighbors to edge i (if none, -1).
    decimal: round float coordinates to the nth decimal place for comparison
    """
    # Reshape to (E,2,2) and round coordinates to the specified decimal place
    edges_2d = np.round(np.stack([edges[:, :2], edges[:, 2:]], axis=1), decimal)
    E_ = len(edges_2d)
    ls_edges = edges_2d.tolist()

    ls_nn_indices = []
    for j in range(E_):
        points_j = ls_edges[j]  # [[x1_j,y1_j],[x2_j,y2_j]]
        nn_for_j = []
        for i in range(E_):
            if i == j:
                continue
            # Adjacent if any point (endpoint) is the same
            if (ls_edges[i][0] in points_j) or (ls_edges[i][1] in points_j):
                nn_for_j.append(i)
        ls_nn_indices.append(nn_for_j)

    max_len = max(len(sub) for sub in ls_nn_indices)
    nnidx = -1 * np.ones([E_, max_len], dtype=int)
    for j, neighbors in enumerate(ls_nn_indices):
        for k, idx in enumerate(neighbors):
            nnidx[j, k] = idx

    return nnidx

class GeneticAlgorithm3:
    def __init__(
            self,
            fitness_func: Callable = None,
            save_dir=None,
            dim: int = 128,
            num_samples: int = 1000,
            num_elite: int = 10,
            probability_method: str = "linear",
            selection_method: str = "stochastic_remainder_selection",
            crossover_method: str = "rank_based_adaptive",
            mutation_method: str = "rank_based_adaptive",
            k1: float = 0.5
    ):
        """
        From the existing GA code:
          - Set num_samples to 5000 (default)
          - Add 'triple_phase' to probability_method
          - Maintain existing methods for selection, crossover, and mutation
        """
        self.save_dir = save_dir
        self.fitness_func = fitness_func
        self.dim = dim
        self.num_offsprings = num_samples - num_elite
        self.num_elite = num_elite
        self.k1 = k1

        # probability_method
        if probability_method == "triple_phase":
            self.compute_probability = self.triple_phase_probability
        elif probability_method == "linear":
            self.compute_probability = self.linear_probability
        elif probability_method == "boltzmann":
            self.compute_probability = self.boltzmann_probability
        else:
            raise ValueError("Unsupported probability_method: " + probability_method)

        # selection_method
        if selection_method == "roulette_wheel":
            self.selection = self.roulette_wheel_selection
        elif selection_method == "stochastic_remainder_selection":
            self.selection = self.stochastic_remainder_selection
        else:
            raise ValueError("Unsupported selection_method: " + selection_method)

        # crossover_method
        if crossover_method == "standard":
            self.crossover = self.standard_crossover
        elif crossover_method == "adaptive":
            print("Warning! (crossover_method = 'adaptive') not fully tested.")
            assert (self.num_offsprings % 2) == 0
            self.crossover = self.adaptive_crossover
        elif crossover_method == "rank_based_adaptive":
            assert (self.num_offsprings % 2) == 0
            self.crossover = self.rank_based_adaptive_crossover
        else:
            raise ValueError("Unsupported crossover_method: " + crossover_method)

        # mutation_method
        if mutation_method == "standard":
            self.mutation = self.gaussian_mutation
        elif mutation_method == "adaptive":
            print("Warning! (mutation_method = 'adaptive') not fully tested.")
            self.mutation = self.adaptive_mutation
        elif mutation_method == "rank_based_adaptive":
            self.mutation = self.rank_based_adaptive_mutation
        elif mutation_method == "rank_based_adaptive_random":
            self.mutation = self.rank_based_adaptive_mutation_random
        elif mutation_method == "rank_based_adaptive_hybrid":
            self.mutation = self.random_triple_group_mutation
        else:
            raise ValueError("Unsupported mutation_method: " + mutation_method)


    def linear_probability(self, old_generation, selection_pressure):
        """
        Existing linear_probability
        probability = (fitness - worst)/(best-worst)*(selection_pressure-1)+1
        """
        self.fitness = self.fitness_func(old_generation)
        self.best_fitness = tf.reduce_max(self.fitness)
        self.average_fitness = tf.reduce_mean(self.fitness)
        self.worst_fitness = tf.reduce_min(self.fitness)

        numerator = (self.fitness - self.worst_fitness)
        denominator = (self.best_fitness - self.worst_fitness)
        prob_raw = numerator / denominator * (selection_pressure - 1.) + 1.
        sum_pr = tf.reduce_sum(prob_raw)
        probability = prob_raw / sum_pr
        return probability

    # -------------------
    # selection methods
    # -------------------

    def stochastic_remainder_selection(self, probabilities):
        total_population_size = self.num_offsprings*2
        probabilities = probabilities / (tf.reduce_sum(probabilities))
        expected_counts = probabilities * tf.cast(total_population_size, tf.float32)
        integer_parts = tf.floor(expected_counts)
        fractional_parts = expected_counts - integer_parts

        deterministic_selection = tf.repeat(tf.range(tf.size(probabilities)), tf.cast(integer_parts, tf.int32))
        remaining_count = total_population_size - tf.size(deterministic_selection)

        frac_sum = tf.reduce_sum(fractional_parts)
        fractional_probabilities = fractional_parts / frac_sum
        indices = tf.range(tf.size(probabilities))
        probabilistic_selection = tf.random.categorical(tf.math.log([fractional_probabilities]), remaining_count)
        probabilistic_selection = tf.gather(indices, probabilistic_selection[0])

        selected_indices = tf.concat([deterministic_selection, probabilistic_selection], axis=0)
        return selected_indices

    # ----------------
    # crossover
    # ----------------


    def rank_based_adaptive_crossover(self, old_generation, parents_indices):
        parents_fitness = tf.gather(self.fitness, parents_indices)
        sorted_indices = tf.gather(parents_indices, tf.argsort(parents_fitness))[::2]
        parents_indices = tf.random.shuffle(sorted_indices)
        parents_fitness = tf.gather(self.fitness, parents_indices)

        parents_rank = tf.argsort(tf.argsort(parents_fitness))
        dominant_rank = tf.cast(tf.where(parents_rank[::2] > parents_rank[1::2],
                                         parents_rank[::2], parents_rank[1::2]), tf.float32)
        dominant_mask = tf.where(parents_rank[::2] > parents_rank[1::2], 1., 0.)[:, tf.newaxis]
        parents = tf.gather(old_generation, parents_indices)
        dom_parents = parents[::2] * dominant_mask + parents[1::2] * (1. - dominant_mask)
        rec_parents = parents[1::2] * dominant_mask + parents[::2] * (1. - dominant_mask)
        crossover_prob = 2. * self.k1 * (self.num_offsprings - 1. - dominant_rank) / (self.num_offsprings - 1.)
        crossover_prob = tf.clip_by_value(crossover_prob, 0., self.k1)

        mask = tf.where(tf.random.uniform((self.num_offsprings // 2, self.dim)) < crossover_prob[:, tf.newaxis], 1., 0.)
        off1 = (1. - mask) * dom_parents + mask * rec_parents
        off2 = (1. - mask) * rec_parents + mask * dom_parents
        offsprings = tf.concat([off1, off2], axis=0)
        return offsprings[:self.num_offsprings]
    # ----------------
    # mutation
    # ----------------

    def rank_based_adaptive_mutation(self, new_generation, mutation_rate):
        fitness = self.fitness_func(new_generation)
        ranks = tf.cast(tf.argsort(tf.argsort(fitness)), tf.float32)[..., tf.newaxis]
        mut_prob = 2. * mutation_rate * (self.num_offsprings - 1. - ranks) / (self.num_offsprings - 1.)
        mut_prob = tf.clip_by_value(mut_prob, 0., mutation_rate)

        mutation_values = tf.random.normal((self.num_offsprings, self.dim))
        mask = tf.random.uniform((self.num_offsprings, self.dim)) < mut_prob
        new_generation = tf.where(mask, mutation_values, new_generation)
        return new_generation

    def rank_based_adaptive_mutation_random(self, new_generation, mutation_rate):
        """
        After applying rank-based adaptive mutation to the entire offspring population,
        half of the individuals from the mutated_generation result are preserved,
        and the other half are randomly generated to return the final offspring population.
        """
        # Apply rank-based adaptive mutation
        fitness = self.fitness_func(new_generation)
        ranks = tf.cast(tf.argsort(tf.argsort(fitness)), tf.float32)[..., tf.newaxis]
        mut_prob = 2. * mutation_rate * (self.num_offsprings - 1. - ranks) / (self.num_offsprings - 1.)
        mut_prob = tf.clip_by_value(mut_prob, 0., mutation_rate)
        mutation_values = tf.random.normal((self.num_offsprings, self.dim))
        mask = tf.random.uniform((self.num_offsprings, self.dim)) < mut_prob
        mutated_generation = tf.where(mask, mutation_values, new_generation)

        # Half of the total offspring population is preserved from the mutated_generation,
        # and the other half is newly generated randomly.
        half = self.num_offsprings // 2
        half=tf.cast(half, tf.int32)  # or tf.int64
        preserved = mutated_generation[:half]  # Half from existing results
        forced_random = tf.random.normal((half, self.dim))  # Half randomly generated

        # Combine the two parts to create the final population of 20000 individuals
        final_population = tf.concat([forced_random, preserved], axis=0)
        return final_population

    # ----------------
    # step & elitism
    # ----------------
    def step(self, old_generation, selection_pressure, mutation_rate):
        # compute probability
        probability = self.compute_probability(old_generation, selection_pressure)
        # selection
        parents_indices = self.selection(probability)
        # crossover
        new_generation = self.crossover(old_generation, parents_indices)
        # mutation
        new_generation = self.mutation(new_generation, mutation_rate)
        # elitism
        new_generation = self.elitism(old_generation, new_generation)
        return new_generation

    def elitism(self, old_generation, new_generation):
        if self.num_elite > 0:
            elite_indices = tf.argsort(self.fitness)[-self.num_elite:]
            elite = tf.gather(old_generation, elite_indices)
        else:
            elite = tf.zeros_like(old_generation)[:0]
        new_generation = tf.concat([elite, new_generation], axis=0)
        return new_generation

    # ----------------
    # run()
    # ----------------
    def run(
            self,
            total_iteration: int = 20000,
            sub_iteration: int = 10,
            initial_selection_pressure: float = 3.,
            final_selection_pressure: float = 3.,
            initial_mutation_rate: float = 1.0,
            final_mutation_rate: float = 0.01,
            generator=None,
            rotmat=0, dip=0, dipBasis=0,edges=0,coes=0,
            dataset_dir_name=None,
            get_hmtotalloss=None,
            save_spin_edge_csv=None,
    ):

        selection_pressure_schedule = tf.linspace(initial_selection_pressure, final_selection_pressure, total_iteration)

        half_iteration = total_iteration // 2
        linear_part = tf.linspace(initial_mutation_rate, final_mutation_rate, half_iteration)
        fixed_part = tf.fill([total_iteration - half_iteration], tf.constant(final_mutation_rate, dtype=tf.float32))
        mutation_rate_schedule = tf.concat([linear_part, fixed_part], axis=0)

        # mutation_rate_schedule = tf.linspace(initial_mutation_rate, final_mutation_rate, total_iteration)

        generation = tf.random.normal((self.num_offsprings, self.dim))

        start_time = time.time()
        for i, (selection_pressure, mutation_rate) in enumerate(
                zip(selection_pressure_schedule, mutation_rate_schedule)
        ):
            generation = self.step(generation, selection_pressure, mutation_rate)
            # Save every 1000 iterations
            if (i + 1) % (total_iteration//10) == 0:
                checkpoint_path = os.path.join(self.save_dir, "final_outputs.npy")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                np.save(checkpoint_path, generation)
                energy=self.fitness_func(generation)


                def find_min_info(arr):
                    min_value = np.max(arr)
                    min_indices = np.where(arr == min_value)
                    return min_value, min_indices[0]

                min_val, min_idx = find_min_info(energy)
                print(f"Max value: {min_val}")
                print(f"Location of max value: {min_idx}")
                idx = min_idx[0]
                generation1 = generation[idx, None]
                x_train = generator(generation1)
                norm_xtrain=tf.sign(x_train)
                spin_3d = binary2spin(x_train, rotmat)
                normspin_3d=spin_3d / np.linalg.norm(spin_3d, axis=-1, keepdims=True)
                spin_3d = spin_3d[0]
                spin_3d = spin_3d/np.linalg.norm(spin_3d, axis=-1,keepdims=True)
                traincsv_dir_name = "train_csv"
                os.makedirs(traincsv_dir_name, exist_ok=True)
                output_csv_path = os.path.join(traincsv_dir_name, f"train{dataset_dir_name}.csv")
                # If there are multiple spin_3d, pass as a list like [spin_3dA, spin_3dB, ...]
                save_spin_edge_csv([spin_3d, spin_3d, spin_3d], edges, coes, output_csv_path)
                print(f"\nCheckpoint saved at iteration {i + 1}: {checkpoint_path}")

            if (i + 1) % sub_iteration == 0:
                correlation = tf.reduce_mean(generation[:1] * generation).numpy()
                STD = tf.reduce_mean(tf.math.reduce_std(generation, axis=0)).numpy()
                elapsed_time = time.time() - start_time
                progress = (i + 1) / total_iteration
                total_estimated_time = elapsed_time / progress
                eta = total_estimated_time - elapsed_time
                print(
                    f'{i + 1} / {total_iteration} | Averaged Fitness: {self.average_fitness.numpy():.9f} worst Fitness {self.worst_fitness.numpy():.9f} '
                    f'| Best Fitness: {self.best_fitness.numpy():.9f} | Correlation: {correlation:.4f} '
                    f'| STD: {STD} | Progress: {elapsed_time:.2f}s / {total_estimated_time:.2f}s | ETA: {eta:.2f}s'
                )

        # Final save (overwrite the final result)
        final_output_path = os.path.join(self.save_dir, "final_outputs.npy")
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        np.save(final_output_path, generation)
        print("\nFinal outputs saved at:", final_output_path)

        return generation
