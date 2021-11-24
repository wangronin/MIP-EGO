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
from .trend import constant_trend
import logging, sys, os

class LoggerFormatter(logging.Formatter):
    default_time_format = '%m/%d/%Y %H:%M:%S'
    default_msec_format = '%s,%02d'

    FORMATS = {
        logging.DEBUG : '%(asctime)s - [%(name)s.%(levelname)s] {%(pathname)s:%(lineno)d} -- %(message)s',
        logging.INFO : '%(asctime)s - [%(name)s.%(levelname)s] -- %(message)s',
        logging.WARNING : '%(asctime)s - [%(name)s.%(levelname)s] {%(name)s} -- %(message)s',
        logging.ERROR : '%(asctime)s - [%(name)s.%(levelname)s] {%(name)s} -- %(message)s',
        'DEFAULT' : '%(asctime)s - %(levelname)s -- %(message)s'
    }
    
    def __init__(self, fmt='%(asctime)s - %(levelname)s -- %(message)s'):
        LoggerFormatter.FORMATS['DEFAULT'] = fmt
        super().__init__(fmt=fmt, datefmt=None, style='%') 
    
    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        _fmt = self._style._fmt

        # Replace the original format with one customized by logging level
        self._style._fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])

        # Call the original formatter class to do the grunt work
        fmt = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = _fmt
        return fmt

MACHINE_EPSILON = np.finfo(np.double).eps


class AutoencoderGaussianProcess(GaussianProcess):

    def __init__(self, encoding_dim=10, encoder=None, levels=None, logger=None, gpu="0", optimizer="CMA", verbose=False, **kwargs):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=gpu 

        lb, ub = -0.1, 1.1
        mean = constant_trend(encoding_dim, beta=None)
        thetaL = 1e-10 * (ub - lb) * np.ones(encoding_dim)
        thetaU = 10 * (ub - lb) * np.ones(encoding_dim)
        theta0 = np.random.rand(encoding_dim) * (thetaU - thetaL) + thetaL
        
        # initialize the base GaussianProcess model using preset settings for the autoencoder output
        super(AutoencoderGaussianProcess, self).__init__(mean=mean, 
            corr='matern',
            theta0=theta0, 
            thetaL=thetaL, 
            thetaU=thetaU,
            nugget=0, 
            noise_estim=False,
            optimizer=optimizer, 
            wait_iter=3, 
            random_start=encoding_dim,
            likelihood='concentrated', 
            eval_budget=100 * encoding_dim)

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
        self.logger = logger
        self.verbose = verbose

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)
        fmt = LoggerFormatter()

        if self.verbose:
            # create console handler and set level to warning
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            self._logger.addHandler(ch)

        # create file handler and set level to debug
        if logger is not None:
            fh = logging.FileHandler(logger)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

        if hasattr(self, 'logger'):
            self._logger.propagate = False

    def build_encoder(self, shape):
        """Build a simple AutoEncoder dense network to get a latent space."""
        input_layer = keras.Input(shape=shape[1])
        # "encoded" is the encoded representation of the input
        encoded = layers.Dense(self.encoding_dim*2, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)

        # "decoded" is the lossy reconstruction of the input
        decoded = layers.Dense(self.encoding_dim*2, activation='relu')(encoded)
        decoded = layers.Dense(shape[1], activation='sigmoid')(decoded)

        # This model maps an input to its reconstruction
        self.autoencoder = keras.Model(input_layer, decoded)
        self.encoder = keras.Model(input_layer, encoded)
        #we do not care about the decoder part
        self.autoencoder.compile(optimizer='adam', loss='mse')
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
            if (t != "N"):
                data[:,i] = (data[:,i] - b[0]) / (b[1] - b[0])
            i+=1
        return data

    def train_encoder(self, search_space, epochs=50, sample_size=1000):
        "Builds the autoencoder model and trains the encoder using the search_space to generate samples"
        #get data from search_space
        self._logger.info("Sampling the search space")
        data = search_space.sampling(sample_size, method='LHS')
        self.search_space = search_space
        data = self.scale_data(data)
        #normalize the data

        self._logger.info(f"Fitting autoencoder with input shape {data.shape}")
        self.build_encoder(data.shape)
        self.autoencoder.fit(data, data,
                epochs=epochs,
                batch_size=256,
                shuffle=True,
                validation_split=0.1)

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
        assert self.encoder is not None
        X = self.scale_data(X)
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
        X = self.scale_data(X)
        X = self.encoder.predict(X)
        #encode X
        return super(AutoencoderGaussianProcess, self).predict(X, **kwargs)
