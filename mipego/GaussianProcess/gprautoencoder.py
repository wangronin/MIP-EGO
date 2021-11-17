# -*- coding: utf-8 -*-

# Author: Bas van Stein <bas9112@gmail.com>

"""
Autoencoder based gaussian process regressor.

@author: basvasntein
"""

import numpy as np
from numpy import std, array, atleast_2d
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder
from .gpr import GaussianProcess
import keras
from keras import layers
from keras import regularizers
from keras.callbacks import TensorBoard

MACHINE_EPSILON = np.finfo(np.double).eps


class AutoencoderGaussianProcess(GaussianProcess):

    def __init__(self, encoding_dim=10, sample_size=10000, encoder=None, levels=None, **kwargs):
        super(AutoencoderGaussianProcess, self).__init__(**kwargs)

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
        self.sample_size = sample_size

    def build_encoder(self, shape):
        """Build a simple AutoEncoder dense network to get a latent space."""
        input_layer = keras.Input(shape=shape)
        # "encoded" is the encoded representation of the input
        encoded = layers.Dense(self.encoding_dim*2, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)

        # "decoded" is the lossy reconstruction of the input
        decoded = layers.Dense(self.encoding_dim*2, activation='relu')(encoded)
        decoded = layers.Dense(shape[0], activation='sigmoid')(decoded)

        # This model maps an input to its reconstruction
        self.autoencoder = keras.Model(input_layer, decoded)
        self.encoder = keras.Model(input_layer, encoded)
        #we do not care about the decoder part
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def train_encoder(self, search_space):
        "Builds the autoencoder model and trains the encoder using the search_space to generate samples"
        #get data from search_space
        data = search_space.sampling(self.sample_size, method='LHS')
        data = self._check_X(data)
        self.build_encoder(self, data.shape)
        self.autoencoder.fit(data, data,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_split=0.1,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    def _check_X(self, X):
        "Transform X to a one-hot encoded object"
        X_ = array(X, dtype=object)
        if hasattr(self, '_levels'):
            X_cat = X_[:, self._cat_idx]
            try:
                X_cat = self._enc.transform(X_cat)
            except:
                X_cat = self._enc.fit_transform(X_cat)
            X = np.c_[np.delete(X_, self._cat_idx, 1).astype(float), X_cat]
        return X

    def fit(self, X, y):
        "Fit the Gaussian Process model"
        assert self._encoder is not None
        X = self._check_X(X)
        X = self.encoder.predict(X)
        y = y.ravel()
        self.y = y
        self.is_fitted = True
        return super(AutoencoderGaussianProcess, self).fit(X, y)

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
        check_is_fitted(self)
        # Check data
        X = self._check_X(X)
        X = self.encoder.predict(X)
        #encode X
        return super(AutoencoderGaussianProcess, self).predict(X, **kwargs)
