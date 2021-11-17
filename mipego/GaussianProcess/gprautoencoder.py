# -*- coding: utf-8 -*-

# Author: Bas van Stein <bas9112@gmail.com>

"""
Autoencoder based gaussian process regressor.

@author: basvasntein
"""

import numpy as np
from sklearn.gaussian_process import correlation_models as correlation
from sklearn.utils.validation import check_is_fitted
from .gpr import GaussianProcess

MACHINE_EPSILON = np.finfo(np.double).eps


class AutoencoderGaussianProcess(GaussianProcess):

    def __init__(self, **kwargs):
        super(AutoencoderGaussianProcess, self).__init__(**kwargs)
    

    def predict(self, X, **kwargs):
        super(AutoencoderGaussianProcess, self).predict(X, **kwargs)