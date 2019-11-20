# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:44:21 2017

@author: wangronin
"""

import pdb

import warnings
import numpy as np
from numpy import sqrt, exp, pi
from scipy.stats import norm
from abc import ABCMeta, abstractmethod
from .Bi_Objective import *

# warnings.filterwarnings("error")

# TODO: perphas also enable acquisition function engineering here?
# meaning the combination of the acquisition functions
class InfillCriteria:
    __metaclass__ = ABCMeta
    def __init__(self, model, plugin=None, minimize=True):
        assert hasattr(model, 'predict')
        self.model = model
        self.minimize = minimize
        # change maximization problem to minimization
        self.plugin = plugin if self.minimize else -plugin
        if self.plugin is None:
            self.plugin = np.min(model.y) if minimize else -np.max(self.model.y)
    
    @abstractmethod
    def __call__(self, X):
        raise NotImplementedError

    def _predict(self, X):
        y_hat, sd2 = self.model.predict(X, eval_MSE=True)
        sd = sqrt(sd2)
        if not self.minimize:
            y_hat = -y_hat
        return y_hat, sd

    def _gradient(self, X):
        y_dx, sd2_dx = self.model.gradient(X)
        if not self.minimize:
            y_dx = -y_dx
        return y_dx, sd2_dx

    def check_X(self, X):
        """Keep input as '2D' object 
        """
        return np.atleast_2d(X)
        # return [X] if not hasattr(X[0], '__iter__') else X

#TODO_CHRIS here add class HVI(InfillCriteria)
#Overwrite __init__(), _predict() and _gradient() and everything else to make it work
#with two surrogate models
#use lower confidence bound
class HVI(InfillCriteria):
    """
    Hyper Volume Improvement
    """
    def __init__(self, model=None, time_model=None, loss_model=None, plugin=None, minimize=True, alpha=0.1, solutions=None, n_left=None, max_iter=None, sol=None,ref_time=None,ref_loss=None):
        assert hasattr(time_model, 'predict')
        assert hasattr(loss_model, 'predict')
        self.time_model = time_model
        self.loss_model = loss_model
        self.minimize = minimize
        #self.alpha = alpha#CHRIS alpha for Lower Confidence Bound
        self.alpha = 0.1 + 0.8*(n_left/max_iter)#CHRIS variable alpha for Lower Confidence Bound
        self.solutions = solutions
        self.n_left = n_left
        self.max_iter = max_iter
        self.Solution = sol
        self.par = pareto(self.solutions)
        self.ref_time = ref_time
        self.ref_loss = ref_loss
        # change maximization problem to minimization
        self.plugin = plugin if self.minimize else -plugin
        if not self.minimize:
            print("Warning: HVI might not work correctly for maximization")#TODO_CHRIS make shure this does work for maximization
        if self.plugin is None:
            #self.plugin = np.min(model.y) if minimize else -np.max(self.model.y)
            self.plugin = np.min(time_model.y) + np.min(loss_model.y) if minimize else -np.max(self.time_model.y)-np.max(self.loss_model.y)#CHRIS take the sum of mins, because we need to do something

    def __call__(self, X, dx=False):
        X = self.check_X(X)
        y_hat, time_sd,loss_sd = self._predict(X)
        return y_hat

    def _predict(self, X):
        y_time_hat, time_sd2 = self.time_model.predict(X, eval_MSE=True)
        y_loss_hat, loss_sd2 = self.loss_model.predict(X, eval_MSE=True)
        if not self.minimize:
            y_time_hat = -y_time_hat
            y_loss_hat = -y_loss_hat
        #CHRIS use y_hat and sd2 to calculate LCB of expected time and loss values, pass these to s-metric and calculate hypervolume improvement
        y_time_hat = y_time_hat[0]
        time_sd2 = time_sd2[0]
        y_loss_hat = y_loss_hat[0]
        loss_sd2 = loss_sd2[0]
        time_sd = sqrt(time_sd2)
        loss_sd = sqrt(loss_sd2)
        if self.alpha > 1.0 or self.alpha < 0:
            print('error: alpha for Lower Confidence bound must be between 0.0 and 1.0')
            exit()
        elif self.alpha >= 0.5:
            exp_time, _ = norm.interval(self.alpha-(1.0-self.alpha),loc=y_time_hat,scale=time_sd)#CHRIS Lower Confidence Bound
            exp_loss, _ = norm.interval(self.alpha-(1.0-self.alpha),loc=y_loss_hat,scale=loss_sd)#CHRIS Lower Confidence Bound
        else:
            _,exp_time = norm.interval(1.0-2.0*self.alpha,loc=y_time_hat,scale=time_sd)#CHRIS Lower Confidence Bound
            _,exp_loss = norm.interval(1.0-2.0*self.alpha,loc=y_loss_hat,scale=loss_sd)#CHRIS Lower Confidence Bound
        expected = self.Solution(X[0])
        expected.time = exp_time
        expected.loss = exp_loss
        
        hyp_vol_imp = s_metric(expected, self.solutions, self.n_left,self.max_iter,ref_time=self.ref_time,ref_loss=self.ref_loss,par=self.par)
        
        return hyp_vol_imp, time_sd, loss_sd

    def _gradient(self, X):
        #CHRIS returned gradient is sum of gradient of both models
        print("HVI gradient() is called?")
        y_time_dx, sd2_time_dx = self.time_model.gradient(X)
        y_loss_dx, sd2_loss_dx = self.loss_model.gradient(X)
        y_dx, sd2_dx = y_time_dx + y_loss_dx, sd2_time_dx + sd2_loss_dx
        if not self.minimize:
            y_dx = -y_dx
        return y_dx, sd2_dx

class MONTECARLO(InfillCriteria):
    """
    Monte Carlo method, returns random value
    """
    def __call__(self, X, dx=False):
        if dx:
            return np.random.rand(),np.random.rand()
        return np.random.rand()

# TODO: test UCB implementation
class UCB(InfillCriteria):
    """
    Upper Confidence Bound 
    """
    def __init__(self, model, plugin=None, minimize=True, alpha=1e-10):
        super(UCB, self).__init__(model, plugin, minimize)#Xin Guo improvement UCB used to be EpsilonPI
        self.alpha = alpha

    def __call__(self, X, dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)

        try:
            f_value = y_hat + self.alpha * sd
        except Exception: # in case of numerical errors
            f_value = 0

        if dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2. * sd)
            try:
                f_dx = y_dx + self.alpha * sd_dx
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_value, f_dx 
        return f_value

class EI(InfillCriteria):
    """
    Expected Improvement
    """
    # perhaps separate the gradient computation here
    def __call__(self, X, dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)
        # if the Kriging variance is to small
        # TODO: check the rationale of 1e-6 and why the ratio if intended
        # TODO: implement a counterpart of 'sigma2' for randomforest
        #if hasattr(self.model, 'sigma2'):#Xin Guo improvement
        #    if sd / np.sqrt(self.model.sigma2) < 1e-6:
        #        return (np.array([0.]),  np.zeros((len(X[0]), 1))) if dx else 0.
        if sd < 1e-6:#Xin Guo improvement
            f_value = (np.array([0.]),  np.zeros((len(X[0]), 1))) if dx else np.array([0.])
            return f_value
    
        try:
            # TODO: I have save xcr_ becasue xcr * sd != xcr_ numerically
            # find out the cause of such an error, probably representation error...
            xcr_ = self.plugin - y_hat
            xcr = xcr_ / sd
            xcr_prob, xcr_dens = norm.cdf(xcr), norm.pdf(xcr)
            f_value = xcr_ * xcr_prob + sd * xcr_dens
        except Exception: # in case of numerical errors
            # IMPORTANT: always keep the output in the same type
            f_value = np.array([0.])

        if dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2. * sd)
            try:
                f_dx = -y_dx * xcr_prob + sd_dx * xcr_dens
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_value, f_dx 
        return f_value

class EpsilonPI(InfillCriteria):
    """
    epsilon-Probability of Improvement
    # TODO: verify the implementation
    """
    def __init__(self, model, plugin=None, minimize=True, epsilon=1e-10):
        super(EpsilonPI, self).__init__(model, plugin, minimize)
        self.epsilon = epsilon

    def __call__(self, X, dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)

        coef = 1 - self.epsilon if y_hat > 0 else (1 + self.epsilon)
        try:
            xcr_ = self.plugin - coef * y_hat 
            xcr = xcr_ / sd
            f_value = norm.cdf(xcr)
        except Exception:
            f_value = 0.

        if dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2. * sd)
            try:
                f_dx = -(coef * y_dx + xcr * sd_dx) * norm.pdf(xcr) / sd
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_value, f_dx 
        return f_value

class PI(EpsilonPI):
    """
    Probability of Improvement
    """
    def __init__(self, model, plugin=None, minimize=True):
        super(PI, self).__init__(model, plugin, minimize, epsilon=0)

class MGFI(InfillCriteria):
    """
    Moment-Generating Function of Improvement 
    My new acquisition function proposed in SMC'17 paper
    """
    def __init__(self, model, plugin=None, minimize=True, t=1):
        super(MGFI, self).__init__(model, plugin, minimize)
        self.t = t

    def __call__(self, X, dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)
        
        # if the Kriging variance is to small
        # TODO: check the rationale of 1e-6 and why the ratio if intended
        if np.isclose(sd, 0):
            return (np.array([0.]), np.zeros((len(X[0]), 1))) if dx else 0.

        try:
            y_hat_p = y_hat - self.t * sd ** 2.
            beta_p = (self.plugin - y_hat_p) / sd
            term = self.t * (self.plugin - y_hat - 1)
            f_ = norm.cdf(beta_p) * exp(term + self.t ** 2. * sd ** 2. / 2.)
        except Exception: # in case of numerical errors
            f_ = np.array([0.])

        if np.isinf(f_):
            f_ = np.array([0.])
            
        if dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2. * sd)

            try:
                term = exp(self.t * (self.plugin + self.t * sd ** 2. / 2 - y_hat - 1))
                m_prime_dx = y_dx - 2. * self.t * sd * sd_dx
                beta_p_dx = -(m_prime_dx + beta_p * sd_dx) / sd
        
                f_dx = term * (norm.pdf(beta_p) * beta_p_dx + \
                    norm.cdf(beta_p) * ((self.t ** 2) * sd * sd_dx - self.t * y_dx))
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_, f_dx
        return f_
        
class GEI(InfillCriteria):
    """
    Generalized Expected Improvement 
    """
    def __init__(self, model, plugin=None, minimize=True, g=1):
        super(GEI, self).__init__(model, plugin, minimize)
        self.g = g

    def __call__(self, X, dx=False):
        pass

if __name__ == '__main__':

    # TODO: diagnostic plot for the gradient of Infill-Criteria
    # goes to unittest
    from GaussianProcess.trend import linear_trend, constant_trend
    from GaussianProcess import GaussianProcess
    from GaussianProcess.utils import plot_contour_gradient
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from deap import benchmarks

    np.random.seed(123)
    
    plt.ioff()
    fig_width = 16
    fig_height = 16
    
    noise_var = 0.
    def fitness(X):
        X = np.atleast_2d(X)
        return np.array([benchmarks.schwefel(x)[0] for x in X]) + \
            np.sqrt(noise_var) * np.random.randn(X.shape[0])
        
    dim = 2
    n_init_sample = 10

    x_lb = np.array([-5] * dim)
    x_ub = np.array([5] * dim)

    X = np.random.rand(n_init_sample, dim) * (x_ub - x_lb) + x_lb
    y = fitness(X)

    thetaL = 1e-5 * (x_ub - x_lb) * np.ones(dim)
    thetaU = 10 * (x_ub - x_lb) * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    mean = linear_trend(dim, beta=None)
    model = GaussianProcess(mean=mean, corr='matern', theta0=theta0, thetaL=thetaL, thetaU=thetaU,
                            nugget=None, noise_estim=True, optimizer='BFGS', verbose=True,
                            wait_iter=3, random_start=10, eval_budget=50)
    
    model.fit(X, y)
    
    def grad(model):
        f = MGFI(model, t=10)
        def __(x):
            _, dx = f(x, dx=True)
            return dx
        return __
    
    t = 1
    infill = MGFI(model, t=t)
    infill_dx = grad(model)
    
    m = lambda x: model.predict(x)
    sd2 = lambda x: model.predict(x, eval_MSE=True)[1]

    m_dx = lambda x: model.gradient(x)[0]
    sd2_dx = lambda x: model.gradient(x)[1]
    
    if 1 < 2:
        fig0, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=False, sharex=False,
                                  figsize=(fig_width, fig_height),
                                  subplot_kw={'aspect': 'equal'}, dpi=100)
                                  
        gs1 = gridspec.GridSpec(1, 3)
        gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
    
        plot_contour_gradient(ax0, fitness, None, x_lb, x_ub, title='Noisy function',
                              n_level=20, n_per_axis=200)
        
        plot_contour_gradient(ax1, m, m_dx, x_lb, x_ub, title='GPR estimation',
                              n_level=20, n_per_axis=200)
                              
        plot_contour_gradient(ax2, sd2, sd2_dx, x_lb, x_ub, title='GPR variance',
                              n_level=20, n_per_axis=200)
        plt.tight_layout()
    
    fig1, ax3 = plt.subplots(1, 1, figsize=(fig_width, fig_height),
                             subplot_kw={'aspect': 'equal'}, dpi=100)
                             
    plot_contour_gradient(ax3, infill, infill_dx, x_lb, x_ub, title='Infill-Criterion',
                          is_log=True, n_level=50, n_per_axis=250)

    plt.tight_layout()
    plt.show()
