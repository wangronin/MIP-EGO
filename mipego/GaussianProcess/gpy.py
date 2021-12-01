# -*- coding: utf-8 -*-

# Author: Bas van Stein <bas9112@gmail.com>

"""
Gaussian process regressor using gpytorch.

@author: basvasntein
"""

import logging
import sys

import gpytorch
import numpy as np
import torch
from numpy import array


class LoggerFormatter(logging.Formatter):
    default_time_format = "%m/%d/%Y %H:%M:%S"
    default_msec_format = "%s,%02d"

    FORMATS = {
        logging.DEBUG: "%(asctime)s - [%(name)s.%(levelname)s] {%(pathname)s:%(lineno)d} -- %(message)s",
        logging.INFO: "%(asctime)s - [%(name)s.%(levelname)s] -- %(message)s",
        logging.WARNING: "%(asctime)s - [%(name)s.%(levelname)s] {%(name)s} -- %(message)s",
        logging.ERROR: "%(asctime)s - [%(name)s.%(levelname)s] {%(name)s} -- %(message)s",
        "DEFAULT": "%(asctime)s - %(levelname)s -- %(message)s",
    }

    def __init__(self, fmt="%(asctime)s - %(levelname)s -- %(message)s"):
        LoggerFormatter.FORMATS["DEFAULT"] = fmt
        super().__init__(fmt=fmt, datefmt=None, style="%")

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        _fmt = self._style._fmt

        # Replace the original format with one customized by logging level
        self._style._fmt = self.FORMATS.get(record.levelno, self.FORMATS["DEFAULT"])

        # Call the original formatter class to do the grunt work
        fmt = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = _fmt
        return fmt


MACHINE_EPSILON = np.finfo(np.double).eps


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PytorchGaussianProcess:
    def __init__(self, logger=None, likelihood=None, verbose=False, use_cuda=False, **kwargs):
        self.likelihood = likelihood
        if likelihood == None:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.verbose = verbose
        self.logger = logger
        self.is_fitted = False
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.use_cuda = use_cuda

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

        if hasattr(self, "logger"):
            self._logger.propagate = False

    def _check_X(self, X):
        "Transform X to a one-hot encoded object"
        X_ = array(X, dtype=object)
        if hasattr(self, "_levels"):
            X_cat = X_[:, self._cat_idx]
            try:
                X_cat = self._enc.transform(X_cat)
            except:
                X_cat = self._enc.fit_transform(X_cat)
            X = np.c_[np.delete(X_, self._cat_idx, 1).astype(float), X_cat]
        return X

    def fit(self, X, y, training_iter=50):
        "Fit the Gaussian Process model"
        y = torch.from_numpy(np.array(y, dtype="float64"))
        X = torch.from_numpy(np.array(X, dtype="float64"))
        self.y = y
        self.training_iter = training_iter
        self.model = ExactGPModel(X, y, self.likelihood)
        self._logger.info("Training the GPytorch exact inference model")

        if (self.use_cuda):
            X = X.cuda()
            y = y.cuda()
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.1
        )  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(self.training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(X)
            # Calc loss and backprop gradients
            loss = -mll(output, y)
            loss.backward()
            self._logger.info(
                "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                % (
                    i + 1,
                    self.training_iter,
                    loss.item(),
                    self.model.covar_module.base_kernel.lengthscale.item(),
                    self.model.likelihood.noise.item(),
                )
            )
            optimizer.step()
        self.is_fitted = True

    def predict(self, X, eval_MSE=False):
        """Predict regression target for `X`.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Returns
        eval_MSE : {bool} if true also the variance is returned.
        -------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted mean values.
        y_variance : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted variance values.
        """
        # Get into evaluation (predictive posterior) mode
        X = torch.from_numpy(np.array(X, dtype="float64"))
        if (self.use_cuda):
            X = X.cuda()
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_preds = self.likelihood(self.model(X))

        if (self.use_cuda):
            y_preds = y_preds.cpu()
        if eval_MSE:
            return y_preds.mean.detach().numpy(), y_preds.variance.detach().numpy()
        return y_preds.mean.detach().numpy()
