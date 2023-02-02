import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sys
sys.modules['keras'] = keras
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, LSTM, Flatten, Dense, Input, Reshape
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import sys
from functools import partial
import h5py
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model, to_categorical

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from data_prep import *
from auto_encoders import *
import os
import utils


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    # return z_mean + K.exp(0.5 * z_log_var) * epsilon
    return z_mean + 0*K.exp(0.5 * z_log_var) +  0*epsilon

def maximum_mean_discrepancy(x, y, kernel=utils.gaussian_kernel_matrix):

    """Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.
    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
    where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
                GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    """
    with tf.name_scope('MaximumMeanDiscrepancy'):
        # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))

        # We do not allow the loss to become negative.
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost



class SVA_ae():
    def __init__(self, s_ae, t_ae, n_classes, latent_dim=20, beta=0.005) -> None:
        self.s_input_shape = s_ae.input_shape
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.t_input_shape = t_ae.input_shape
        self.classifier = None
        self.s_ae = s_ae
        self.t_ae = t_ae
        self.beta = beta
        self.build_source_model()
        self.build_target_model()

    def build_classifier(self):

        model = Sequential()
        model.add(Dense(self.latent_dim, activation='relu', input_dim=self.latent_dim))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))

        feats = Input(shape=(self.latent_dim,))
        class_label = model(feats)
        self.classifier = Model(feats, class_label)
    
    def build_source_model(self):
        s_input_layer = Input(shape=self.s_input_shape, name='s_input')
        self.s_latent_space = self.s_ae.encoder(s_input_layer)
        if self.classifier is None:
            self.build_classifier()
        self.s_net_cls = Model(self.s_ae.encoder.input, self.classifier(self.s_ae.encoder.output))

    def build_target_model(self):
        t_input_layer = Input(shape=self.t_input_shape, name='s_input')
        self.t_latent_space = self.t_ae.encoder(t_input_layer)
        if self.classifier is None:
            self.build_classifier()
        self.t_net_cls = Model(self.t_ae.encoder.input, self.classifier(self.t_ae.encoder.output))

    def t_ae_loss(self, y_true, y_pred):
        sigmas = [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ]
        gaussian_kernel = partial(
        utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

        xent_loss =  binary_crossentropy(K.flatten(y_true), K.flatten(y_pred)) / self.t_total_pixel
        # kl_loss = - 0.25 * K.sum(1 + self.t_z_log_var - K.square(self.t_z_mean) - K.exp(self.t_z_log_var), axis=-1)
        mmd_loss = maximum_mean_discrepancy(self.s_ae.encoded, self.t_ae.encoded, kernel=gaussian_kernel)
        vae_loss = K.mean(xent_loss +  self.beta * mmd_loss)
        return vae_loss

if __name__=="__main__":
    #Loading Data
    eda_label = pd.read_csv("data/eda_labels.csv")
    x_eda, y_eda = eda_preprop(eda_label, data_loc='data/eda_data/')

    eeg_label = pd.read_csv("data/eeg_labels.csv")
    data_loc = 'data/Muse_data/'
    x_eeg, y_eeg = eeg_preprep(eeg_label, data_loc=data_loc)

    #pre-training Auto-Encoders.
    eda_ae = AutoEncoder_eda(input_shape=x_eda.shape[1:])
    opt = Adam(lr=0.001)
    eda_ae.autoencoder.compile(loss = 'mse',
                optimizer = opt)
    hist = eda_ae.autoencoder.fit(x_eda, x_eda,
                                    epochs=200,
                                    batch_size=128,
                                    shuffle=True,
                                    verbose=2)

    eeg_ae = AutoEncoder_eeg(input_shape=x_eeg.shape[1:])
    opt = Adam(lr=0.008)
    eeg_ae.autoencoder.compile(loss = 'mse',
                optimizer = opt)
    hist = eeg_ae.autoencoder.fit(x_eeg, x_eeg,
                                    epochs=200,
                                    batch_size=128,
                                    shuffle=True,
                                    verbose=2)


    x_train_s, x_test_s , y_train_s, y_test_s = train_test_split(x_eda, y_eda)
    x_train_t, x_test_t , y_train_t, y_test_t = train_test_split(x_eeg, y_eeg)
    batch_size = 128
    svae = SVA_ae(eda_ae, eeg_ae, n_classes=1)
    svae.s_net_cls.compile(loss = 'binary_crossentropy', optimizer = opt)
    svae.t_net_cls.compile(loss=svae.t_ae_loss, optimizer = opt)

    hist = svae.s_net_cls.fit(x_train_s, y_train_s,
                        epochs = 200,
                        batch_size = batch_size,
                        shuffle=True,
                        verbose=2)

    hist = svae.t_net_cls.fit(x_train_t, y_train_t,
                        epochs = 150,
                        batch_size = batch_size,
                        shuffle=True,
                        verbose=2)

    pred = svae.t_net_cls.predict(x_test_t)
    t_score = accuracy_score(y_test_t, pred)
    print(t_score)

