# -*- coding: utf-8 -*-

# Author: Bas van Stein <bas9112@gmail.com>

"""
Autoencoder based gaussian process regressor using gpytorch.

@author: basvasntein
"""

import logging
import os
import sys

import gpytorch
import keras
import numpy as np
import torch
from keras import layers, regularizers
from sklearn.preprocessing import OneHotEncoder

from .gpy import PytorchGaussianProcess


class AutoencoderGaussianProcess(PytorchGaussianProcess):
    def __init__(self, encoding_dim=10, encoder=None, levels=None, gpu="0", **kwargs):
        super(AutoencoderGaussianProcess, self).__init__(**kwargs)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        # for categorical levels/variable number
        # in the future, maybe implement binary/multi-value split
        if levels is not None:
            assert isinstance(levels, dict)
            self._levels = levels
            self._cat_idx = list(self._levels.keys())
            self._categories = [list(l) for l in self._levels.values()]
            # encode categorical variables to binary values
            self._enc = OneHotEncoder(categories=self._categories, sparse=False)
        self.encoder = encoder
        self.encoding_dim = encoding_dim
        self.is_fitted = False

    def build_encoder(self, shape):
        """Build a simple AutoEncoder dense network to get a latent space."""
        input_layer = keras.Input(shape=shape[1])
        # "encoded" is the encoded representation of the input
        encoded = layers.Dense(
            self.encoding_dim * 2,
            activation="relu",
            activity_regularizer=regularizers.l1(10e-5),
        )(input_layer)
        encoded = layers.Dense(self.encoding_dim, activation="relu")(encoded)

        # "decoded" is the lossy reconstruction of the input
        decoded = layers.Dense(self.encoding_dim * 2, activation="relu")(encoded)
        decoded = layers.Dense(shape[1], activation="sigmoid")(decoded)

        # This model maps an input to its reconstruction
        self.autoencoder = keras.Model(input_layer, decoded)
        self.encoder = keras.Model(input_layer, encoded)
        # we do not care about the decoder part
        self.autoencoder.compile(optimizer="adam", loss="mse")
        self._logger.info(self.autoencoder.summary())

    def scale_data(self, data):
        """
        Transforms the search space data to a numpy array scaled between 0 and 1.
        Categorical variables are one-hot encoded, and should be defined at the end
        of the search space
        """
        data = self._check_X(data)
        i = 0
        for t in self.search_space.var_type:
            b = self.search_space.bounds[i]
            if t != "N":
                data[:, i] = (data[:, i] - b[0]) / (b[1] - b[0])
            i += 1
        return data

    def train_encoder(self, search_space, epochs=50, sample_size=1000):
        "Builds the autoencoder model and trains the encoder using the search_space to generate samples"
        # get data from search_space
        self._logger.info("Sampling the search space")
        data = search_space.sampling(sample_size, method="LHS")
        self.search_space = search_space
        data = self.scale_data(data)
        # normalize the data

        self._logger.info(f"Fitting autoencoder with input shape {data.shape}")
        self.build_encoder(data.shape)
        self.autoencoder.fit(
            data,
            data,
            epochs=epochs,
            batch_size=256,
            shuffle=True,
            validation_split=0.1,
        )

    def fit(self, X, y, training_iter=50):
        "Fit the Gaussian Process model"
        assert self.encoder is not None
        X = self.scale_data(X)
        X = self.encoder.predict(X)
        y = y.ravel()
        self.y = y
        return super(AutoencoderGaussianProcess, self).fit(X, y, training_iter)

    def predict(self, X, **kwargs):
        """Predict regression target for `X`.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        # Check data
        X = self.scale_data(X)
        X = self.encoder.predict(X)
        return super(AutoencoderGaussianProcess, self).predict(X, **kwargs)
