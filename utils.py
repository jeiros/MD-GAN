import matplotlib.pyplot as plt

import numpy as np
import keras
import keras.backend as K
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from msmbuilder.preprocessing import MinMaxScaler
import mdtraj
from matplotlib import pyplot as plt


def plot_losses(losses):
    losses = np.array(losses)
    fig, ax = plt.subplots()
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    ax.set(ylabel='BCE', xlabel='Epoch')
    return fig, ax


def make_trajectory_trainable(traj_list):
    """
    Build a train/test splittable array of cartesian coordinates from a list
    of mdtraj.Trajectory objects

    Parameters
    ----------
    traj_list: list of mdtraj.Trajectory objects

    Returns
    -------
    data: np.array, shape=(frames, n_atoms, 3)
        A numpy array of the XYZ coordinates of all the frames in the list of
        trajs. Coordinates are squised from -1 to 1.
        Use a MinMaxScaler.inverse_transform to map them back to the original
        space.
    sc: MinMaxScaler, The scaler used to squish the coordinates.
    """
    frame00 = traj_list[0][0]
    trjs = [t.superpose(frame00) for t in traj_list]
    sc = MinMaxScaler(feature_range=(-1, 1))
    frames = []
    for t in trjs:
        for f in t:
            frames.append(f.xyz.reshape(frame00.n_atoms, 3))
    f_txx_sc = sc.fit_transform(frames)
    data = np.dstack(f_txx_sc)
    data = data.transpose(2, 0, 1)
    return data, sc


def fake_traj_from_samples(samples, top, scaler):
    fake_tr = samples[:, :, :, 0]
    fake_traj_orig_space = [scaler.inverse_transform(t) for t in fake_tr]
    fake_traj = mdtraj.Trajectory(fake_traj_orig_space, topology=top)
    fake_traj.center_coordinates()
    fake_traj.superpose(fake_traj, 0)
    return fake_traj


def scatter(arr, ax=None, scatter_kws=None):
    if ax is None:
        f, ax = plt.subplots()
    if scatter_kws is None:
        scatter_kws = {}
    ax.scatter(arr[:, 0], arr[:, 1], **scatter_kws)
    return ax


def make_latent_samples(n_samples, sample_dim):
    return np.random.normal(loc=0, scale=1, size=(n_samples, sample_dim))


def make_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable


def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])


def make_2dtraj_GAN(sample_size,
                    g_hidden_size,
                    d_hidden_size,
                    leaky_alpha,
                    g_learning_rate,
                    d_learning_rate):
    K.clear_session()

    generator = Sequential([
        Dense(g_hidden_size, input_shape=(sample_size,)),
        LeakyReLU(alpha=leaky_alpha),
        Dense(2),
        Activation('tanh')
    ], name='generator')

    discriminator = Sequential([
        Dense(d_hidden_size, input_shape=(2,)),
        LeakyReLU(alpha=leaky_alpha),
        Dense(1),
        Activation('sigmoid')
    ], name='discriminator')

    gan = Sequential([
        generator,
        discriminator
    ])

    discriminator.compile(optimizer=Adam(lr=d_learning_rate), loss='binary_crossentropy')
    gan.compile(optimizer=Adam(lr=g_learning_rate), loss='binary_crossentropy')

    return gan, generator, discriminator


def make_3dtraj_GAN(sample_size,
                    g_hidden_size,
                    d_hidden_size,
                    leaky_alpha,
                    g_learning_rate,
                    d_learning_rate):
    K.clear_session()

    generator = Sequential([
        Dense(g_hidden_size, input_shape=(sample_size,)),
        LeakyReLU(alpha=leaky_alpha),
        Dense(2),
        Activation('tanh')
    ], name='generator')

    discriminator = Sequential([
        Dense(d_hidden_size, input_shape=(2,)),
        LeakyReLU(alpha=leaky_alpha),
        Dense(1),
        Activation('sigmoid')
    ], name='discriminator')

    gan = Sequential([
        generator,
        discriminator
    ])

    discriminator.compile(optimizer=Adam(lr=d_learning_rate), loss='binary_crossentropy')
    gan.compile(optimizer=Adam(lr=g_learning_rate), loss='binary_crossentropy')

    return gan, generator, discriminator
