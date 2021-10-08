"""
Created on Mon Apr 23 17:16:39 2018

@author: Hao Wang
@email: wangronin@gmail.com

"""
from pdb import set_trace

from typing import Callable, Any, Tuple
import os, sys, dill, functools, logging, time

import numpy as np
import json, copy, re 
from copy import copy
from joblib import Parallel, delayed

from . import InfillCriteria
from . import BiObjective
from .base import baseBO
from .Solution import Solution
from .SearchSpace import SearchSpace
from .Surrogate import SurrogateAggregation

class BO(baseBO):
    def _create_acquisition(self, fun=None, par={}, return_dx=False):
        fun = fun if fun is not None else self._acquisition_fun
        if hasattr(getattr(InfillCriteria, fun), 'plugin'):
            if 'plugin' not in par:
                par.update({'plugin': self.fmin})
        
        return super()._create_acquisition(fun, par, return_dx)

    def pre_eval_check(self, X):
        """Check for the duplicated solutions as it is not allowed in noiseless cases
        """
        if not isinstance(X, Solution):
            X = Solution(X, var_name=self.var_names)
        
        N = X.N
        if hasattr(self, 'data'):
            X = X + self.data

        _ = []
        for i in range(N):
            x = X[i]
            idx = np.arange(len(X)) != i
            CON = np.all(np.isclose(np.asarray(X[idx][:, self.r_index], dtype='float'),
                                    np.asarray(x[self.r_index], dtype='float')), axis=1)
            INT = np.all(X[idx][:, self.i_index] == x[self.i_index], axis=1)
            CAT = np.all(X[idx][:, self.d_index] == x[self.d_index], axis=1)
            if not any(CON & INT & CAT):
                _ += [i]

        return X[_]

class ParallelBO(BO):
    # TODO: add other Parallelization options: 
    # 1) niching-based approach (my EVOLVE paper),
    # 2) bi-objective Pareto-front (PI vs. EI) (my WCCI '16 paper), and
    # 3) maybe QEI?
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.n_point > 1

        if self._acquisition_fun == 'MGFI':
            self._par_name = 't'
            # Log-normal distribution for `t` supported on [0, \infty)
            self._sampler = lambda x: np.exp(np.log(x['t']) + 0.5 * np.random.randn())
        elif self._acquisition_fun == 'UCB':
            self._par_name = 'alpha'
            # Logit-normal distribution for `alpha` supported on [0, 1]
            self._sampler = lambda x: 1 / (1 + np.exp((x['alpha'] * 4 - 2) \
                + 0.6 * np.random.randn())) 
        elif self._acquisition_fun == 'EpsilonPI':
            self._par_name = 'epsilon'
            self._sampler = None # TODO: implement this!
        else:
            raise NotImplementedError
        
        _criterion = getattr(InfillCriteria, self._acquisition_fun)()
        if self._par_name not in self._acquisition_par:
            self._acquisition_par[self._par_name] = getattr(_criterion, self._par_name)
        
    def _batch_arg_max_acquisition(self, n_point, return_dx):
        criteria = []
        for _ in range(n_point):
            _par = self._sampler(self._acquisition_par)
            _acquisition_par = copy(self._acquisition_par)
            _acquisition_par.update({self._par_name : _par})
            criteria.append(
                self._create_acquisition(par=_acquisition_par, return_dx=return_dx)
            )
        
        if self.n_job > 1:
            __ = Parallel(n_jobs=self.n_job)(
                delayed(self._argmax_restart)(c, logger=self._logger) for c in criteria
            )
        else:
            __ = [list(self._argmax_restart(_, logger=self._logger)) for _ in criteria]
        
        return tuple(zip(*__))


class BiObjectiveBO(ParallelBO):
    def __init__(self, model2=None, ref_obj1=0, ref_obj2=0, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        self.model2 = model2
        self.ref_obj1 = ref_obj1
        self.ref_obj2 = ref_obj2
        
        if self._acquisition_fun == 'HVI':
            self._par_name = 'alpha'
            # Logit-normal distribution for `alpha` supported on [0, 1]
            self._sampler = lambda x: 1 / (1 + np.exp((x['alpha'] * 4 - 2) \
                + 0.6 * np.random.randn())) 
        else:
            raise NotImplementedError
        _criterion = getattr(InfillCriteria, self._acquisition_fun)()
        if self._par_name not in self._acquisition_par:
            self._acquisition_par[self._par_name] = getattr(_criterion, self._par_name)
        

class AnnealingBO(ParallelBO):
    def __init__(self, t0=2, tf=1e-1, schedule='exp', *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        self.t0 = t0
        self.tf = tf
        self.schedule = schedule
        self._acquisition_par['t'] = t0
        
        # TODO: add supports for UCB and epsilon-PI
        max_iter = (self.max_FEs - self._DoE_size) / self.n_point
        if self.schedule == 'exp':                          # exponential
            self.alpha = (self.tf / t0) ** (1. / max_iter) 
            self._anealer = lambda t: t * self.alpha
        elif self.schedule == 'linear':
            self.eta = (t0 - self.tf) / max_iter            # linear
            self._anealer = lambda t: t - self.eta
        elif self.schedule == 'log':
            self.c = self.tf * np.log(max_iter + 1)         # logarithmic 
            self._anealer = lambda t: t * self.c / np.log(self.iter_count + 2)
        else:
            raise NotImplementedError
        
        def callback():
            self._acquisition_par['t'] = self._anealer(self._acquisition_par['t'])
        self._acquisition_callbacks += [callback]
    
class SelfAdaptiveBO(ParallelBO):
    def __init__(self, *argv, **kwargs):
        super.__init__(*argv, **kwargs)
        assert self.n_point > 1

    def _batch_arg_max_acquisition(self, n_point, return_dx):
        criteria = []
        _t_list = []
        N = int(n_point / 2)

        for _ in range(n_point):
            _t = np.exp(self._acquisition_par['t'] * np.random.randn())
            _t_list.append(_t)
            _acquisition_par = copy(self._acquisition_par)
            _acquisition_par.update({'t' : _t})
            criteria.append(
                self._create_acquisition(par=_acquisition_par, return_dx=return_dx)
            )
        
        if self.n_job > 1:
            __ = Parallel(n_jobs=self.n_job)(
                delayed(self._argmax_restart)(c, logger=self._logger) for c in criteria
            )
        else:
            __ = [list(self._argmax_restart(_, logger=self._logger)) for _ in criteria]
        
        # NOTE: this adaptation is different from what I did in the LeGO paper..
        idx = np.argsort(__[1])[::-1][:N]
        self._acquisition_par['t'] = np.mean([_t_list[i] for i in idx])
        return tuple(zip(*__))

class NoisyBO(ParallelBO):
    def pre_eval_check(self, X):
        if not isinstance(X, Solution):
            X = Solution(X, var_name=self.var_names)
        return X

    def _create_acquisition(self, fun=None, par={}, return_dx=False):
        if hasattr(getattr(InfillCriteria, self._acquisition_fun), 'plugin'):
            # use the model prediction to determine the plugin under noisy scenarios
            # TODO: add more options for determining the plugin value
            y_ = self.model.predict(self.data)
            plugin = np.min(y_) if self.minimize else np.max(y_)
            par.update({'plugin' : plugin})
        
        return super()._create_acquisition(par=par, return_dx=return_dx)

class PCABO(ParallelBO):
    def __init__(self):
        pass
