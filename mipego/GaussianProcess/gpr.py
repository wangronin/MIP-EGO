# -*- coding: utf-8 -*-

# Author: Hao Wang 
# Email: wangronin@gmail.com

from __future__ import print_function

import pdb
import warnings 

import numpy as np
from numpy.random import uniform
from numpy import log, pi, log10, exp, dot, sqrt, array

from scipy import linalg
from scipy.linalg import cho_solve
from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.utils import check_random_state, check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

from ..optimizer import OnePlusOne_Cholesky_CMA
from .kernel import absolute_exponential, squared_exponential, generalized_exponential, \
    cubic, pure_nugget, matern
    
from .trend import BasisExpansionTrend, NonparametricTrend

# from libgpr import corr_grad_theta, matern

MACHINE_EPSILON = np.finfo(np.double).eps

def l1_cross_distances(X):
    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.

    Parameters
    ----------

    X: array_like
        An array with shape (n_samples, n_features)

    Returns
    -------

    D: array with shape (n_samples * (n_samples - 1) / 2, n_features)
        The array of componentwise L1 cross-distances.

    ij: arrays with shape (n_samples * (n_samples - 1) / 2, 2)
        The indices i and j of the vectors in X associated to the cross-
        distances in D: D[k] = np.abs(X[ij[k, 0]] - Y[ij[k, 1]]).
    """
    X = check_array(X)
    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0
    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):n_samples])

    return D, ij

def my_dot(x, y):
    n_row = x.shape[0]
    n_col = y.shape[1]
    res = np.zeros((n_row, n_col))
    for i in range(n_row):
        for j in range(n_col):
            res[i, j] = np.sum(x[i, :] * y[:, j])
    return res

# TODO: remove the dependences from sklearn
class GaussianProcess(BaseEstimator, RegressorMixin):
    """The Gaussian Process model class.

    Read more in the :ref:`User Guide <gaussian_process>`.

    Parameters
    ----------
    mean : string or callable, optional
        A meanession function returning an array of outputs of the linear
        meanession functional basis. The number of observations n_samples
        should be greater than the size p of this basis.
        Default assumes a simple constant meanession trend.
        Available built-in meanession models are::

            'constant', 'linear', 'quadratic'

    corr : string or callable, optional
        A stationary autocorrelation function returning the autocorrelation
        between two points x and x'.
        Default assumes a squared-exponential autocorrelation model.
        Built-in correlation models are::

            'absolute_exponential', 'squared_exponential',
            'generalized_exponential', 'cubic', 'linear', 'matern'

    beta0 : double array_like, optional
        The meanession weight vector to perform Ordinary Kriging (OK).
        Default assumes Universal Kriging (UK) so that the vector beta of
        meanession weights is estimated using the maximum likelihood
        principle.

    storage_mode : string, optional
        A string specifying whether the Cholesky decomposition of the
        correlation matrix should be stored in the class (storage_mode =
        'full') or not (storage_mode = 'light').
        Default assumes storage_mode = 'full', so that the
        Cholesky decomposition of the correlation matrix is stored.
        This might be a useful parameter when one is not interested in the
        MSE and only plan to estimate the BLUP, for which the correlation
        matrix is not required.

    verbose : boolean, optional
        A boolean specifying the verbose level.
        Default is verbose = False.

    theta0 : double array_like, optional
        An array with shape (n_features, ) or (1, ).
        The parameters in the autocorrelation model.
        If thetaL and thetaU are also specified, theta0 is considered as
        the starting point for the maximum likelihood estimation of the
        best set of parameters.
        Default assumes isotropic autocorrelation model with theta0 = 1e-1.

    thetaL : double array_like, optional
        An array with shape matching theta0's.
        Lower bound on the autocorrelation parameters for maximum
        likelihood estimation.
        Default is None, so that it skips maximum likelihood estimation and
        it uses theta0.

    thetaU : double array_like, optional
        An array with shape matching theta0's.
        Upper bound on the autocorrelation parameters for maximum
        likelihood estimation.
        Default is None, so that it skips maximum likelihood estimation and
        it uses theta0.

    normalize : boolean, optional
        Input X and observations y are centered and reduced wrt
        means and standard deviations estimated from the n_samples
        observations provided.
        Default is normalize = True so that data is normalized to ease
        maximum likelihood estimation.

    TODO: the nugget behaves differently than what is described here...
    nugget : double or ndarray, optional
        Introduce a nugget effect to allow smooth predictions from noisy
        data.  If nugget is an ndarray, it must be the same length as the
        number of data points used for the fit.
        The nugget is added to the diagonal of the assumed training covariance;
        in this way it acts as a Tikhonov regularization in the problem.  In
        the special case of the squared exponential correlation function, the
        nugget mathematically represents the variance of the input values.
        Default assumes a nugget close to machine precision for the sake of
        robustness (nugget = 10. * MACHINE_EPSILON).

    optimizer : string, optional
        A string specifying the optimization algorithm to be used.
        Default uses 'fmin_cobyla' algorithm from scipy.optimize.
        Available optimizers are::

            'fmin_cobyla', 'Welch'

        'Welch' optimizer is dued to Welch et al., see reference [WBSWM1992]_.
        It consists in iterating over several one-dimensional optimizations
        instead of running one single multi-dimensional optimization.

    random_start : int, optional
        The number of times the Maximum Likelihood Estimation should be
        performed from a random starting point.
        The first MLE always uses the specified starting point (theta0),
        the next starting points are picked at random according to an
        exponential distribution (log-uniform on [thetaL, thetaU]).
        Default does not use random starting point (random_start = 1).

    random_state: integer or numpy.RandomState, optional
        The generator used to shuffle the sequence of coordinates of theta in
        the Welch optimizer. If an integer is given, it fixes the seed.
        Defaults to the global numpy random number generator.


    Attributes
    ----------
    theta_ : array
        Specified theta OR the best set of autocorrelation parameters (the \
        sought maximizer of the reduced likelihood function).

    log_likelihood_ : array
        The optimal reduced likelihood function value.

    """
    _optimizer_types = ['BFGS', 'CMA']
    _correlation_types = {
        'absolute_exponential': absolute_exponential,
        'squared_exponential': squared_exponential,
        'generalized_exponential': generalized_exponential,
        'cubic': cubic,
        'matern': matern}
    
    _likelihood_functions = ['concentrated', 'restricted']

    # TODO: separater the kernel function from here
    def __init__(self, mean, corr='squared_exponential',
                 theta0=1e-1, thetaL=None, thetaU=None, sigma2=None, 
                 nugget=None, noise_estim=False, optimizer='BFGS', 
                 likelihood='concentrated', random_start=1, wait_iter=5,
                 eval_budget=None, random_state=None, verbose=False):

        self.mean = mean      # Prior mean function 
        self.corr = corr      # Prior correlation function 
        self.sigma2 = sigma2  # variance of the stationary process
        self.verbose = verbose
        self.corr_type = corr
        self.is_fitted = False 
        
        # hyperparameters: kernel function
        self.theta0 = np.array(theta0).flatten()
        self.thetaL = np.array(thetaL).flatten()
        self.thetaU = np.array(thetaU).flatten()
        
        if not (np.isfinite(self.thetaL).all() and np.isfinite(self.thetaU).all()):
            raise ValueError("all bounds are required finite.")
        
        # hyperparameter optimization parameters
        self.optimizer = optimizer
        self.random_start = random_start
        self.random_state = random_state
        self.wait_iter = wait_iter
        self.eval_budget = eval_budget

        self.noise_var = np.atleast_1d(nugget) if nugget else 0
        self.noise_estim = True if noise_estim else False
        self.noisy = True if self.noise_var or self.noise_estim else False

        # three cases to compute the log-likelihood function
        # TODO: verify: it seems the noisy case is the most useful one
        if not self.noisy:
            self.estimation_mode = 'noiseless'
        elif self.noise_estim:
            self.estimation_mode = 'noise_estim'
        else:
            self.estimation_mode = 'noisy'
        
        assert likelihood in self._likelihood_functions
        self.likelihood = likelihood  # or restricted
        
        # estimation mode for the trend
        if isinstance(self.mean, BasisExpansionTrend):
            self.mean_type = 'basis_expansion'
            self.estimate_trend = True if self.mean.beta is None else False
        elif isinstance(self.mean, NonparametricTrend):
            self.mean_type = 'nonparametric'
        
    def _check_data(self, X, y):
        # Force data to 2D numpy.array
        X, y = check_X_y(X, y, multi_output=False, y_numeric=True)
        y = y.reshape(-1, 1)
        n_samples, n_features = X.shape
        self.X, self.y = X, y

        # Run input checks
        self._check_params()

        # Calculate matrix of distances D between samples
        D, ij = l1_cross_distances(X)
        if (np.min(np.sum(D, axis=1)) == 0. and self.corr != pure_nugget):
            raise Exception("Multiple input features cannot have the same"
                            " target value.")
        self.D = D
        self.ij = ij

        if self.mean_type == 'basis_expansion' and self.estimate_trend:
            F = self.mean.F(X)
            p = F.shape[1]
            if p > n_samples:
                raise Exception(("Ordinary least squares problem is undetermined "
                                 "n_samples=%d must be greater than the "
                                 "meanession model size p=%d.") % (n_samples, p))
            self.F = F
    
    def sampling_prior(self, X):
        pass
    
    def sampling_posterior(self, X):
        pass
    
    def prior_cov(self, X1, X2=None, corr=False):
        # Check input shapes
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2) if X2 else X1

        n_eval_X1, _ = X1.shape
        n_eval_X2, _ = X2.shape
        n_samples, n_features = self.X.shape
        n_samples_y, n_targets = self.y.shape

        if X1.shape[1] != n_features or X2.shape[1] != n_features:
            raise ValueError(("The number of features in X (X.shape[1] = %d) "
                              "should match the number of features used "
                              "for fit() "
                              "which is %d.") % (X1.shape[1], n_features))

        # remember to normalize the inputs
        # if normalize:
            # X1 = (X1 - self.X_mean) / self.X_std
            # X2 = (X2 - self.X_mean) / self.X_std

        dx_new = manhattan_distances(X1, Y=X2, sum_over_features=False)
        R_prior = self.corr(self.theta_, dx_new).reshape(n_eval_X1, n_eval_X2)

        if corr:
            return R_prior
        else:
            C_prior = array([self.sigma2[i] * R_prior for i in range(n_targets)])
            C_prior = sqrt((C_prior ** 2.).sum(axis=0) / n_targets)

            return C_prior

    def fit(self, X, y):
        """
        The Gaussian Process model fitting method.

        Parameters
        ----------
        X : double array_like
            An array with shape (n_samples, n_features) with the input at which
            observations were made.

        y : double array_like
            An array with shape (n_samples, ) or shape (n_samples, n_targets)
            with the observations of the output to be predicted.

        Returns
        -------
        gp : self
            A fitted Gaussian Process model object awaiting data to perform
            predictions.
        """
        # Run input checks
        self.random_state = check_random_state(self.random_state)
        self._check_data(X, y)

        # Determine Gaussian Process model parameters
        if self.thetaL is not None and self.thetaU is not None:
            # Maximum Likelihood Estimation of the parameters
            if self.verbose:
                print("Maximum Likelihood Estimation of the hyperparameters...")
            self.par, self.log_likelihood_, env = self._optimize_hyperparameter()
            
            if np.isinf(self.log_likelihood_):
                # TODO: open design choice here:
                #   1) leave this exception handeling part to the program calls gpr
                #   2) handel it here and save the incident to a log
                print('Doubling theta upper bound. Increasing nugget...')
                self.thetaU *= 2
                self.noise_var *= 10
                self.par, self.log_likelihood_, env = self._optimize_hyperparameter()
                # raise Exception("Bad parameter region. Try increasing upper bound")

        # find a better name for noise_var
        self.theta_ = self.par['theta']
        self.noise_var = env['noise_var']
        self.sigma2 = env['sigma2']
        self.rho = env['rho'] 
        self.Yt = env['Yt']
        self.C = env['C']
        if self.mean_type == 'basis_expansion' and self.estimate_trend:
            self.Ft = env['Ft']
            self.G = env['G']
            self.Q = env['Q']

        # compute for beta and gamma
        self.compute_beta_gamma()
        self.is_fitted = True

        return self

    def update(self, X, y):
        # TODO: implement incremental training 
        self.fit(X, y)
        return self

    def predict(self, X, eval_MSE=False, batch_size=None):
        """
        This function evaluates the Gaussian Process model at x.

        Parameters
        ----------
        X : array_like
            An array with shape (n_eval, n_features) giving the point(s) at
            which the prediction(s) should be made.

        eval_MSE : boolean, optional
            A boolean specifying whether the Mean Squared Error should be
            evaluated or not.
            Default assumes evalMSE = False and evaluates only the BLUP (mean
            prediction).

        batch_size : integer, optional
            An integer giving the maximum number of points that can be
            evaluated simultaneously (depending on the available memory).
            Default is None so that all given points are evaluated at the same
            time.

        Returns
        -------
        y : array_like, shape (n_samples, ) or (n_samples, n_targets)
            An array with shape (n_eval, ) if the Gaussian Process was trained
            on an array of shape (n_samples, ) or an array with shape
            (n_eval, n_targets) if the Gaussian Process was trained on an array
            of shape (n_samples, n_targets) with the Best Linear Unbiased
            Prediction at x.

        MSE : array_like, optional (if eval_MSE == True)
            An array with shape (n_eval, ) or (n_eval, n_targets) as with y,
            with the Mean Squared Error at x.
        """
        assert hasattr(self, 'X')

        # Check input shapes
        # TODO: remove the support for multiple independent outputs
        X = check_array(X)
        n_eval, _ = X.shape
        n_samples, n_features = self.X.shape
        n_samples_y, n_targets = self.y.shape

        # Run input checks
        self._check_params()

        if X.shape[1] != n_features:
            raise ValueError(("The number of features in X (X.shape[1] = %d) "
                              "should match the number of features used "
                              "for fit() which is %d.") % (X.shape[1], n_features))

        if batch_size is None:
            # (evaluates all given points in a single batch run)
            # Initialize output
            y = np.zeros(n_eval)
            if eval_MSE:
                MSE = np.zeros(n_eval)

            # Get pairwise componentwise L1-distances to the input training set
            # TODO: remove calculations of distances from here
            dx = manhattan_distances(X, Y=self.X, sum_over_features=False)
            # Get meanession function and correlation
            r = self.corr(self.theta_, dx).reshape(n_eval, n_samples)

            # Predictor
            # TODO: get rid of n_targets in this class
            y = (self.mean(X) + my_dot(r, self.gamma)).reshape(n_eval, n_targets).ravel()
            
            # Mean Squared Error
            if eval_MSE:
                rt = linalg.solve_triangular(self.C, r.T, lower=True)

                if self.estimate_trend:  # Universal / Ordinary Kriging
                    f = self.mean.F(X)
                    u = linalg.solve_triangular(self.G.T, np.dot(self.Ft.T, rt) - f.T, lower=True)
                else:                  # simple Kriging
                    u = np.zeros((n_targets, n_eval))

                MSE = np.dot(self.sigma2.reshape(n_targets, 1),
                             (1. - (rt ** 2.).sum(axis=0) + 
                             (u ** 2.).sum(axis=0))[np.newaxis, :])
                MSE = np.sqrt((MSE ** 2.).sum(axis=0) / n_targets)

                # Mean Squared Error might be slightly negative depending on
                # machine precision: force to zero!
                MSE[MSE < 0.] = 0.
                MSE = MSE.ravel()

                return y, MSE
            else:
                return y
        else:
            # Memory management
            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")

            if eval_MSE:
                y, MSE = np.zeros(n_eval), np.zeros(n_eval)
                for k in range(max(1, n_eval / batch_size)):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size + 1, n_eval + 1])
                    y[batch_from:batch_to], MSE[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to],
                                     eval_MSE=eval_MSE, batch_size=None)

                return y, MSE

            else:

                y = np.zeros(n_eval)
                for k in range(max(1, n_eval / batch_size)):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size + 1, n_eval + 1])
                    y[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to],
                                     eval_MSE=eval_MSE, batch_size=None)
                return y

    def gradient(self, x):
        """
        Calculate the gradient of the posterior mean and variance
        Note that the nugget effect will not the change the computation below
        """

        check_is_fitted(self, 'X')
        # Check input shapes
        x = np.atleast_2d(x)
        n_eval, _ = x.shape
        n_samples, n_features = self.X.shape

        if _ != n_features:
            raise Exception('x does not have the right size!')

        if n_eval != 1:
            raise Exception('x must be a vector!')

        # trend and its Jacobian
        f = self.mean.F(x).reshape(-1, 1)
        f_dx = self.mean.Jacobian(x)
        
        # correlation and its Jacobian
        d = manhattan_distances(x, Y=self.X, sum_over_features=False)
        r = self.corr(self.theta_, d).reshape(n_eval, n_samples)
        r_dx = self.corr_dx(x, X=self.X, r=r).T
        
        # gradient of the posterior mean
        y_dx = dot(self.mean.beta.T, f_dx) + my_dot(self.gamma.T, r_dx)

        # auxiliary variable: rt = C^-1 * r
        rt = linalg.solve_triangular(self.C, r.T, lower=True)
        rt_dx = linalg.solve_triangular(self.C, r_dx, lower=True)

        mse_dx = -dot(rt.T, rt_dx)      # for Simple Kriging
        if self.estimate_trend:       # Universal / Ordinary Kriging
            # auxiliary variable: u = Ft^T * rt - f
            u = dot(self.Ft.T, rt) - f
            u_dx = dot(self.Ft.T, rt_dx) - f_dx
            Ft2inv = linalg.inv(dot(self.Ft.T, self.Ft))
            mse_dx += u.T.dot(Ft2inv).dot(u_dx)

        mse_dx = 2.0 * self.sigma2 * mse_dx
        return y_dx.T, mse_dx.T

    # TODO: move corr_dx, corr_grad_theta the kernel module
    # TODO: rename corr_grad_theta to something else
    def corr_dx(self, x, X=None, theta=None, r=None, nu=1.5):
        # Check input shapes
        x = np.atleast_2d(x)
        n_eval, _ = x.shape
        n_samples, n_features = self.X.shape

        if _ != n_features:
            raise Exception('x does not have the right sizeh!')

        if n_eval != 1:
            raise Exception('x must be a vector!')

        if self.theta_ is None:
            raise Exception('The model is not fitted yet!')

        X = np.atleast_2d(X)
        if X is None:
            X = self.X

        diff = (x - X).T

        if theta is None:
            theta = self.theta_

        # calculate the required variables if not given
        if r is None:
            r = self.corr(self.theta_, np.abs(diff).T)

        theta = theta.reshape(-1, 1)

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                if self.corr_type == 'squared_exponential':
                    grad = -2 * r * (theta * diff)

                elif self.corr_type == 'matern':
                    c = np.sqrt(3)
                    D = np.sqrt(np.sum(theta * diff ** 2., axis=0))

                    if nu == 0.5:
                        grad = - diff * theta / D * r
                    elif nu == 1.5:
                        grad = diff * theta / D
                        grad *= -3. * D * exp(-c * D)
                    elif nu == 2.5:
                        pass

                elif self.corr_type == 'absolute_exponential':
                    grad = -r * theta * np.sign(diff)
                elif self.corr_type == 'generalized_exponential':
                    pass
                elif self.corr_type == 'cubic':
                    pass
                elif self.corr_type == 'linear':
                    pass
            except Warning:
                grad = np.zeros((n_features, n_samples))

        return grad
    
    # def corr_grad_theta(self, theta, X, R, nu=1.5):
        # return corr_grad_theta(theta, X, R, nu=1.5)
    
    def corr_grad_theta(self, theta, X, R, nu=1.5):
        # Check input shapes
        X = np.atleast_2d(X)
        n_eval, _ = X.shape
        n_features = self.X.shape[1]

        if _ != n_features:
            raise Exception('x does not have the right size!')

        diff = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2.

        if self.corr_type == 'squared_exponential':
            grad = -diff * R[..., np.newaxis]

        elif self.corr_type == 'matern':
            c = np.sqrt(3)
            D = np.sqrt(np.sum(theta * diff, axis=-1))

            if nu == 0.5:
                grad = - diff * theta / D * R
            elif nu == 1.5:
                grad = -3 * np.exp(-c * D)[..., np.newaxis] * diff / 2.
            elif nu == 2.5:
                pass

        elif self.corr_type == 'absolute_exponential':
            grad = -np.sqrt(diff) * R[..., np.newaxis]
        elif self.corr_type == 'generalized_exponential':
            pass
        elif self.corr_type == 'cubic':
            pass
        elif self.corr_type == 'linear':
            pass

        return grad

    def correlation_matrix(self, theta, X=None):
        D = self.D
        ij = self.ij
        n_samples = self.X.shape[0]
        
        # calculate the correlation matrix R
        r = self.corr(theta, D)
        R = np.eye(n_samples)
        R[ij[:, 0], ij[:, 1]] = r
        R[ij[:, 1], ij[:, 0]] = r
        return R

    def compute_beta_gamma(self):
        if self.estimate_trend:
            # estimate the trend coefficients
            self.mean.beta = linalg.solve_triangular(self.G, my_dot(self.Q.T, self.Yt))
        self.gamma = linalg.solve_triangular(self.C.T, self.rho).reshape(-1, 1)

    def _compute_aux_var(self, R):
        # compute auxiliary variables
        # Cholesky decomposition of R: Note that this matrix R can be singular
        # change notation from 'C' to 'L': 'L'ower triagular component...
        try:
            L = linalg.cholesky(R, lower=True)
        except linalg.LinAlgError as e:
            raise e

        Yt = linalg.solve_triangular(L, self.y, lower=True)
        if self.estimate_trend:
            # compute rho: the residuals in linear model
            # ρ = Y - Fβ 
            # both Y and F are transformed by R^-1/2 such that the noise covariance is identity
            Ft = linalg.solve_triangular(L, self.F, lower=True)
            Q, G = linalg.qr(Ft, mode='economic')
            rho = Yt - Q.dot(Q.T).dot(Yt)
        else:
            rho = Yt - linalg.solve_triangular(L, self.mean(self.X), lower=True)
            Ft, Q, G = None, None, None

        return L, Ft, Yt, Q, G, rho
    
    def log_likelihood_restricted(self, par, env=None, eval_grad=False):
        """
        TODO: write this function in C or Cython
        The restricted log likelihood function
        It yields the unbiased estimation for the hyperparameters 
        """
        check_is_fitted(self, "X")
        
        n_samples, n_features = self.X.shape
        n_par = len(par)
        log_likelihood = -np.inf
        
        # the MLE estimation of sigma2 is not used here as it will make estimation of theta biased
        if self.estimation_mode == 'noiseless':
            theta, sigma2, noise_var = par[:-1], par[-1], 0
            
        elif self.estimation_mode == 'noisy':
            theta, sigma2, noise_var = par[:-1], par[-1], self.noise_var
            
        elif self.estimation_mode == 'noise_estim':
            theta, sigma2, noise_var = par[:-2], par[-2], par[-1]
            
        R0 = self.correlation_matrix(theta)
        total_var = sigma2 + noise_var
        C = sigma2 * R0 + noise_var * np.eye(n_samples)  # homoscedastic noises are assumed
        R = C / total_var
            
        try:
            L, Ft, Yt, Q, G, rho = self._compute_aux_var(R)
        except linalg.LinAlgError as e:
            warnings.warn("linear operation failed on {}".format(e))
            if eval_grad:
                return log_likelihood, np.zeros((n_par, 1))
            else:
                return log_likelihood
            
        if self.estimate_trend: 
            p = Ft.shape[1]
            log_likelihood = -0.5 * ((n_samples - p) * log(2 * pi * total_var) -
                                     np.log(np.linalg.det(self.F.T.dot(self.F))) + 
                                     2 * np.log(np.diag(L)).sum() + 
                                     np.log(np.diag(G).prod() ** 2) +
                                     rho.T.dot(rho) / total_var).sum()
        else:  # simple kriging
            log_likelihood = -0.5 * (n_samples * log(2 * pi * total_var) - 
                                     2 * np.log(np.diag(L)).sum() + 
                                     rho.T.dot(rho) / total_var).sum()
            
        if np.exp(log_likelihood) > 1:
            if self.verbose:
                warnings.warn("invalid log likelihood value: {}".format(log_likelihood))
            log_likelihood = -np.inf

        if eval_grad:
            # gradient calculation of the log-likelihood
            gamma_ = linalg.solve_triangular(L.T, rho).reshape(-1, 1) / total_var
            Cinv = cho_solve((L, True), np.eye(n_samples)) / total_var
            if self.estimate_trend:
                __ = linalg.solve_triangular(L.T, Q)
                term = __.dot(__.T)
    
            # Covariance: partial derivatives w.r.t. theta
            C_grad_tensor = total_var * self.corr_grad_theta(theta, self.X, R0)
            # Covariance: partial derivatives w.r.t. sigma2
            C_grad_tensor = np.concatenate([C_grad_tensor, R0[..., np.newaxis]], axis=2)
            
            if self.estimation_mode == 'noise_estim':
                # Covariance: partial derivatives w.r.t. noise_var
                C_grad_tensor = np.concatenate([C_grad_tensor, np.eye(n_samples)[..., np.newaxis]], 
                                                axis=2)
            # partial dderivatives of llf
            gradient = np.zeros((n_par, 1))
            for i in range(n_par):
                C_grad = C_grad_tensor[:, :, i]
                # TODO: beta hat is not considered in the partial derivatives
                if self.estimate_trend: 
                    gradient[i] = -0.5 * (np.sum(Cinv * C_grad) - 
                                          gamma_.T.dot(C_grad).dot(gamma_) - 
                                          np.sum(term * C_grad))
                else:  # simple kriging
                    gradient[i] = -0.5 * (np.sum(Cinv * C_grad) - 
                                          gamma_.T.dot(C_grad).dot(gamma_))
                    
        if env is not None:
            env['sigma2'] = sigma2
            env['noise_var'] = noise_var
            env['rho'] = rho
            env['Yt'] = Yt
            env['C'] = L
            if self.estimate_trend:
                env['Ft'] = Ft 
                env['G'] = G
                env['Q'] = Q
                
        if eval_grad:
            return log_likelihood, gradient
        else:
            return log_likelihood

    def log_likelihood_concentrated(self, par, env=None, eval_grad=False):
        r"""
        The concentrated log likelihood function
        The MLE estiation on sigma2 is biased downwards: \hat{sigma2} < sigma2
        
        Parameters concetrated out are:
            beta : coeffiencts in the basis expansion trend function
                replaced by the MLE estimate (GLS formula) in the likelihood 
            sigma2 : stationary variance of the process
                replace by the MLE estimate
        """
        check_is_fitted(self, "X")

        n_samples, n_features = self.X.shape
        n_par = len(par)
        log_likelihood = -np.inf
        if self.estimation_mode == 'noiseless':
            theta = par
            noise_var = 0

            R0 = self.correlation_matrix(theta)
            
            try:
                L, Ft, Yt, Q, G, rho = self._compute_aux_var(R0)
            except linalg.LinAlgError:
                if eval_grad:
                    return (log_likelihood, np.zeros((n_par, 1)))
                else:
                    return log_likelihood
            
            # TODO: experimental correction of the sigma2 estimation
            if Q is not None:
                k = np.linalg.matrix_rank(Q.dot(Q.T))
            else:
                k = 0
            sigma2 = (rho ** 2.).sum(axis=0) / (n_samples - k)
            log_likelihood = -0.5 * (n_samples * log(2. * pi * sigma2) + 
                                     2. * np.log(np.diag(L)).sum() + n_samples)

        elif self.estimation_mode == 'noise_estim':
            theta, alpha = par[:-1], par[-1]
            R0 = self.correlation_matrix(theta)
            R = alpha * R0 + (1 - alpha) * np.eye(n_samples)

            try:
                L, Ft, Yt, Q, G, rho = self._compute_aux_var(R)
            except linalg.LinAlgError:
                if eval_grad:
                    return (log_likelihood, np.zeros(n_par, 1))
                else:
                    return log_likelihood
                
            sigma2_total = (rho ** 2.).sum(axis=0) / n_samples
            sigma2, noise_var = alpha * sigma2_total, (1 - alpha) * sigma2_total
            log_likelihood = -0.5 * (n_samples * log(2. * pi * sigma2_total) + 
                                     2. * np.log(np.diag(L)).sum() + n_samples)

        elif self.estimation_mode == 'noisy':
            theta, sigma2 = par[:-1], par[-1]
            noise_var = self.noise_var
            sigma2_total = sigma2 + noise_var

            R0 = self.correlation_matrix(theta)
            C = sigma2 * R0 + noise_var * np.eye(n_samples)
            R = C / sigma2_total

            try:
                L, Ft, Yt, Q, G, rho = self._compute_aux_var(R)
            except linalg.LinAlgError:
                if eval_grad:
                    return (log_likelihood, np.zeros(n_par, 1)) 
                else: 
                    return log_likelihood

            log_likelihood = -0.5 * (n_samples * log(2. * pi * sigma2_total) + 
                                     2. * np.log(np.diag(L)).sum() + 
                                     np.dot(rho.T, rho) / sigma2_total)

        if env is not None:
            env['sigma2'] = sigma2
            env['noise_var'] = noise_var
            env['rho'] = rho
            env['Yt'] = Yt
            # TODO: change variable 'C' --> 'L'
            env['C'] = L
            env['Ft'] = Ft
            env['G'] = G
            env['Q'] = Q

        # for verificationn
        # TODO: remove this in the future
        if log_likelihood > 0:
            # warnings.warn("invalid log likelihood value: {}".format(log_likelihood))
            return (-np.inf, np.zeros((n_par, 1))) if eval_grad else -np.inf
            
        if not eval_grad:
            return log_likelihood[0]

        # gradient calculation of the log-likelihood
        gamma = linalg.solve_triangular(L.T, rho).reshape(-1, 1)
        
        Rinv = cho_solve((L, True), np.eye(n_samples))
        Rinv_upper = Rinv[np.triu_indices(n_samples, 1)]
        _upper = gamma.dot(gamma.T)[np.triu_indices(n_samples, 1)]

        llf_grad = np.zeros((n_par, 1))

        if self.estimation_mode == 'noiseless':
            # The grad tensor of R w.r.t. theta
            R_grad_tensor = self.corr_grad_theta(theta, self.X, R0)

            for i in range(n_par):
                R_grad_upper = R_grad_tensor[:, :, i][np.triu_indices(n_samples, 1)]

                llf_grad[i] = np.sum(_upper * R_grad_upper) / sigma2 \
                    - np.sum(Rinv_upper * R_grad_upper)

        elif self.estimation_mode == 'noise_estim':
            # The grad tensor of R w.r.t. theta
            R_grad_tensor = alpha * self.corr_grad_theta(theta, self.X, R0)

            # partial derivatives w.r.t theta's
            for i in range(n_par - 1):
                R_grad_upper = R_grad_tensor[:, :, i][np.triu_indices(n_samples, 1)]

                # Note that sigma2_total is used here
                llf_grad[i] = np.sum(_upper * R_grad_upper) / sigma2_total \
                    - np.sum(Rinv_upper * R_grad_upper)

            # partial derivatives w.r.t 'v'
            R_dv = R0 - np.eye(n_samples)
            llf_grad[n_par - 1] = -0.5 * (np.sum(Rinv * R_dv) - 
                                          np.dot(gamma.T, R_dv.dot(gamma)) / sigma2_total)

        elif self.estimation_mode == 'noisy':
            gamma_ = gamma / sigma2_total
            Cinv = Rinv / sigma2_total
            # Covariance: partial derivatives w.r.t. theta
            C_grad_tensor = sigma2_total * self.corr_grad_theta(theta, self.X, R0)
            
            # Covariance: partial derivatives w.r.t. sigma2
            C_grad_tensor = np.concatenate([C_grad_tensor, R0[..., np.newaxis]], axis=2)

            for i in range(n_par):
                C_grad = C_grad_tensor[:, :, i]
                llf_grad[i] = -0.5 * (np.sum(Cinv * C_grad) - 
                                      np.dot(gamma_.T, C_grad).dot(gamma_))
                
        return log_likelihood[0], llf_grad
    
    def _hyperparameter_bound(self, par_list):
        bounds = []
        # TODO: better estimation the upper and lowe bound of sigma2
        for k, name in enumerate(par_list):
            if name == 'theta':
                bounds.append(np.c_[self.thetaL, self.thetaU])
            elif name == 'sigma2':
                bounds.append(np.atleast_2d([1e-5, self.y.std() ** 2]))
            elif name == 'alpha':
                bounds.append(np.atleast_2d([1e-10, 1.0 - 1e-10]))
            elif name == 'noise_var':
                # TODO: implement this
                bounds.append(np.atleast_2d([1e-10, 1.0 - 1e-10]))
        bounds = np.concatenate(bounds, axis=0)
        return bounds
        
    def _optimize_hyperparameter(self):
        """
        optimization procedure to determine the hyperparameter:
            theta: corralation length
            sigma2: process variance for stationary GP
            noise_var: the variance(s) of the noise process
        -------
        """

        if self.verbose:
            print("The chosen optimizer is: " + str(self.optimizer))
            print('Estimation mode: {}'.format(self.estimation_mode))
            print('Likelihood function: {}'.format(self.likelihood))
            print("{} random restarts are specified.".format(self.random_start))
        
        par_list = ['theta']
        par_len = [len(self.thetaL)]
        if self.likelihood == 'restricted' or self.estimation_mode == 'noisy':
            # TODO: implement optimization for the heterogenous case
            par_list += ['sigma2']
            par_len.append(1)
        
        if self.estimation_mode == 'noise_estim':
            if self.likelihood == 'concentrated':
                par_list += ['alpha']
                par_len.append(1)
            elif self.likelihood == 'restricted':
                par_list += ['noise_var']
                par_len.append(1)
        
        bounds = self._hyperparameter_bound(par_list)
        log10bounds = log10(bounds)
        n_theta = len(self.thetaL)
        # if the model has been optimized before as the starting point,
        # then use the last optimized hyperparameters
        # supposed to be good for updating the model incrementally
        # TODO: validate this
        if hasattr(self, 'theta_'):
            log10theta0 = log10(self.theta_)
        else:
            log10theta0 = log10(self.theta0) if self.theta0 is not None else \
                np.random.uniform(log10(bounds)[:, 0], log10(bounds)[:, 1])

        if self.estimation_mode == 'noiseless' and self.likelihood == 'concentrated':
            log10param = log10theta0
        else:
            log10param = np.r_[log10theta0, uniform(log10bounds[n_theta:, 0], 
                                                    log10bounds[n_theta:, 1])]
            
        n_par = len(log10param)
        eval_budget = 200 * n_par if self.eval_budget is None else self.eval_budget
        llf_opt = np.inf
        
        def _obj_func(eval_grad=False):
            def func(log10param):
                self.eval_count += 1
                param = 10. ** np.array(log10param)
                if self.likelihood == 'concentrated':
                    __ = self.log_likelihood_concentrated(param, eval_grad=eval_grad)
                elif self.likelihood == 'restricted':
                    __ = self.log_likelihood_restricted(param, eval_grad=eval_grad)
                return tuple(-_ for _ in tuple(__))
            return func
        
        self.eval_count = 0
        # TODO: maybe adopt an ILS-like restarting heuristic?
        if self.optimizer == 'BFGS':            
            obj_func = _obj_func(eval_grad=True)
            wait_count = 0  # stagnation counter
            
            for iteration in range(self.random_start):
                if iteration != 0:
                    log10param = np.random.uniform(log10bounds[:, 0], log10bounds[:, 1])
                
                # TODO: may be expose the parameter of fmin_l_bfgs_b to the user
                param_opt_, llf_opt_, info = fmin_l_bfgs_b(obj_func, log10param, 
                                                           bounds=log10bounds,
                                                           maxfun=eval_budget)
                
                # TODO: verify this rule to determine the marginal improvement 
                # diff = (llf_opt - llf_opt_) / max(abs(llf_opt_), abs(llf_opt), 1)
                if iteration == 0:
                    param_opt = param_opt_
                    llf_opt = llf_opt_    
                # elif diff >= 1e7 * MACHINE_EPSILON:
                elif llf_opt_ <= llf_opt:
                    param_opt, llf_opt = param_opt_, llf_opt_
                    wait_count = 0
                else:
                    wait_count += 1

                if self.verbose:
                    print('restart {} takes {} evals'.format(iteration + 1, info['funcalls']))
                    print('best log likekihood value: {}'.format(-llf_opt))
                    if info["warnflag"] != 0:
                        warnings.warn("fmin_l_bfgs_b terminated abnormally with "
                                      "the state: {}".format(info))
                                      
                eval_budget -= info['funcalls']
                if eval_budget <= 0 or wait_count >= self.wait_iter:
                    break
        
        elif self.optimizer == 'CMA':   # IPOP-CMA-ES 
            obj_func = _obj_func()
            opt = {
                'sigma0': 0.25 * np.max(log10bounds[:, 1] - log10bounds[:, 0]),
                'max_FEs': eval_budget,
                'ftarget': np.inf,
                'lb': log10bounds[:, 0],
                'ub': log10bounds[:, 1]
            }
            
            # TODO: perphas use the BIPOP-CMA-ES in the future     
            optimizer = OnePlusOne_Cholesky_CMA(
                n_par, obj_func, log10param, minimize=False, **opt
            )
            param_opt, llf_opt, _ = optimizer.run()
            evalcount = optimizer.eval_count
            param_opt = param_opt.flatten()
            
            if self.verbose:
                print('{} evals, best log likekihood value: {}'.format(evalcount, -llf_opt))
        
        optimal_param = 10. ** param_opt
        env = {}
        if self.likelihood == 'concentrated':
            optimal_llf_value = self.log_likelihood_concentrated(optimal_param, env)
        elif self.likelihood == 'restricted':
            optimal_llf_value = self.log_likelihood_restricted(optimal_param, env)
        
        param = {}
        i = 0
        for k, name in enumerate(par_list):
            len_ = par_len[k]
            param[name] = optimal_param[i:i+len_]
            i += len_

        return param, optimal_llf_value, env

    def _check_params(self):
        # Check correlation model
        if not callable(self.corr):
            if self.corr in self._correlation_types:
                self.corr = self._correlation_types[self.corr]
            else:
                raise ValueError("corr should be one of %s or callable, "
                                 "%s was given." % (self._correlation_types.keys(), self.corr))

        # Check correlation parameters
        # self.theta0 = np.atleast_2d(self.theta0)
        lth = self.theta0.size

        if self.thetaL is not None and self.thetaU is not None:
            if self.thetaL.size != lth or self.thetaU.size != lth:
                raise ValueError("theta0, thetaL and thetaU must have the "
                                 "same length.")
            if np.any(self.thetaL <= 0) or np.any(self.thetaU < self.thetaL):
                raise ValueError("The bounds must satisfy O < thetaL <= "
                                 "thetaU.")

        elif self.thetaL is None and self.thetaU is None:
            if np.any(self.theta0 <= 0):
                raise ValueError("theta0 must be strictly positive.")

        elif self.thetaL is None or self.thetaU is None:
            raise ValueError("thetaL and thetaU should either be both or "
                             "neither specified.")

        # Force verbose type to bool
        self.verbose = bool(self.verbose)

        # Check nugget value
        # self.nugget = np.asarray(self.nugget)
        # if np.any(self.nugget) < 0.:
        #     raise ValueError("nugget must be positive or zero.")
        # if (n_samples is not None
        #         and self.nugget.shape not in [(), (n_samples,)]):
        #     raise ValueError("nugget must be either a scalar "
        #                      "or array of length n_samples.")

        # Check optimizer
        if self.optimizer not in self._optimizer_types:
            raise ValueError("optimizer should be one of %s" % self._optimizer_types)

        # Force random_start type to int
        self.random_start = int(self.random_start)
