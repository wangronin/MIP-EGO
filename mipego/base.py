"""
Created on Mon Apr 23 17:16:39 2018

@author: Hao Wang
@email: wangronin@gmail.com

"""
import numpy as np
import os, sys, dill, functools, logging, time

from abc import ABC, abstractmethod
from copy import copy, deepcopy
from typing import Callable, Any, Tuple
from joblib import Parallel, delayed

from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

from . import InfillCriteria
from .Solution import Solution
from .SearchSpace import SearchSpace
from .utils import arg_to_int
from .misc import LoggerFormatter
from .optimizer import argmax_restart

class baseOptimizer(ABC):
    def __init__(self, verbose):
        self.verbose = verbose
        self.xopt = None 
        self.fopt = None 
        self.stop_dict = {}

    @abstractmethod
    def ask(self, n_point=None):
        """Get suggestions from the surrogate model of the optimizer.

        Args:
            n_point (int, optional): Desired number of parallel suggestions in the output. Defaults to None.

        Returns:
            list of dict. List of `n_point` suggestions to evaluate the objective
                function. Each suggestion is a dictionary where each key
                corresponds to a parameter being optimized.
        """
        return

    @abstractmethod
    def tell(self, X, y):
        """Feed an observation back to the surrogate model.

        Args:
            X (list): Places where the objective function has already been evaluated.
                Each suggestion is a dictionary where each key corresponds to a
                parameter being optimized.
            y (array): array-like, shape (n,)
                Corresponding values where objective has been evaluated.
        """
        return

    @abstractmethod
    def evaluate(self, X):
        """Evaluate the objective function for a given sample.

        Args:
            X (dict): Configuration to evaluate
        """
        return 

    @abstractmethod
    def check_stop(self):
        """Check the stop criteria.
        """
        return

    def step(self):
        """Perform one step in the optimization and update the surrogate model.
        """
        X = self.ask()    
        func_vals = self.evaluate(X)
        self.tell(X, func_vals)
    
    def run(self):
        """Perform a complete optimization run.

        Returns:
            list: the optimum configuration with the evaluation value and the stopping criterium.
        """
        while not self.check_stop():
            self.step()
        return self.xopt, self.fopt, self.stop_dict

    @property
    def logger(self):
        """Default logger object.

        Returns:
            logger: The class property.
        """
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


class baseBO(ABC):
    def __init__(
        self, 
        search_space: SearchSpace, 
        obj_fun: Callable,
        parallel_obj_fun: Callable = None,
        eq_fun: Callable = None, 
        ineq_fun: Callable = None, 
        model = None,
        eval_type: str = 'list',
        DoE_size: int = None, 
        n_point: int = 1,
        acquisition_fun: str = 'EI',
        acquisition_par: dict = {},
        acquisition_optimization: dict = {},
        ftarget: float = None,
        max_FEs: int = None, 
        minimize: bool = True, 
        n_job: int = 1,
        data_file: str = None, 
        verbose: bool = False, 
        random_seed: int = None, 
        logger: str = None,
        ):
        """The base class for Bayesian Optimization.

        Args:
            search_space (SearchSpace): The search space, an instance of `SearchSpace` class
            obj_fun (Callable): The objective function to optimize
            parallel_obj_fun (Callable, optional): The objective function that takes multiple solutions simultaneously and implement
                the parallelization by itself. Defaults to None.
            eq_fun (Callable, optional): The equality constraints, whose return value should have the same size as the 
                number of equality constraints. Defaults to None.
            ineq_fun (Callable, optional): The inequality constraints, whose return value should have the same size as the 
                number of inequality constraints. Defaults to None.
            model ([type], optional): The surrogate mode, which will be automatically created if not passed in. Defaults to None.
            eval_type (str, optional): type of arguments `obj_fun` or `parallel_obj_fun` takes: ['list', 'dict']. Defaults to 'list'
            DoE_size (int, optional): The size of inital Design of Experiment (DoE). Defaults to None.
            n_point (int, optional): The number of candidate solutions proposed using infill-criteria. Defaults to 1.
            acquisition_fun (str, optional): The acquisition function. Defaults to 'EI'.
            acquisition_par (dict, optional): Extra parameters to the acquisition function. Defaults to {}.
            acquisition_optimization (dict, optional): Additional parameters controlling the acquisition optimization. Defaults to {}.
            ftarget (float, optional): The target value to hit. Defaults to None.
            max_FEs (int, optional): The maximal number of function evaluations. Defaults to None.
            minimize (bool, optional): To minimize (True) or maximize (False). Defaults to True.
            n_job (int, optional): The number of allowable jobs for parallelizing the function evaluation 
                (if `parallel_obj_fun` is not specified) and Only Effective when n_point > 1. Defaults to 1.
            data_file (str, optional): The name of the file to store extra historical information during the run. Defaults to None.
            verbose (bool, optional): The verbosity. Defaults to False.
            random_seed (int, optional): The seed for pseudo-random number generators. Defaults to None.
            logger (str, optional): name of the logger file or a `logging.Logger` object. Defaults to None.
        """           
        self.obj_fun = obj_fun
        self.parallel_obj_fun = parallel_obj_fun
        self.h = eq_fun
        self.g = ineq_fun
        self.n_job = max(1, int(n_job))
        self.n_point = max(1, int(n_point))
        self.ftarget = ftarget 
        self.minimize = minimize
        self.verbose = verbose
        self.data_file = data_file
        self.max_FEs = int(max_FEs) if max_FEs else np.inf  

        self.search_space = search_space
        self.DoE_size = DoE_size
        self.acquisition_fun = acquisition_fun
        self._acquisition_par = acquisition_par
        # the callback functions executed after every call of `arg_max_acquisition`
        self._acquisition_callbacks = []  
        self.model = model
        self.logger = logger
        self.random_seed = random_seed
        self._set_internal_optimization(**acquisition_optimization)
        self._get_best = np.min if self.minimize else np.max
        self._eval_type = eval_type   
        self._init_flatfitness_trial = 2
        self._set_aux_vars()
        # self._check_params()
    
    @property
    def acquisition_fun(self):
        return self._acquisition_fun

    @acquisition_fun.setter
    def acquisition_fun(self, fun):
        if isinstance(fun, str):
            self._acquisition_fun = fun
        else:
            assert hasattr(fun, '__call__')
        self._acquisition_fun = fun

    @property
    def DoE_size(self):
        return self._DoE_size

    @DoE_size.setter
    def DoE_size(self, DoE_size):
        if DoE_size:
            if isinstance(DoE_size, str):
                self._DoE_size = int(eval(DoE_size))
            elif isinstance(DoE_size, (int, float)):
                self._DoE_size = int(DoE_size)
            else: 
                raise ValueError
        else:
            self._DoE_size = int(self.dim * 20)

    @property
    def random_seed(self):
        return self._random_seed
    
    @random_seed.setter
    def random_seed(self, seed):
        if seed:
            self._random_seed = int(seed)
            if self._random_seed:
                np.random.seed(self._random_seed)

    @property
    def search_space(self):
        return self._search_space

    @search_space.setter
    def search_space(self, search_space):
        self._search_space = search_space
        self.dim = len(self._search_space)
        self.var_names = self._search_space.var_name
        self.r_index = self._search_space.id_C       # indices of continuous variable
        self.i_index = self._search_space.id_O       # indices of integer variable
        self.d_index = self._search_space.id_N       # indices of categorical variable

        self.param_type = self._search_space.var_type
        self.N_r = len(self.r_index)
        self.N_i = len(self.i_index)
        self.N_d = len(self.d_index)

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        if isinstance(logger, logging.Logger):
            self._logger = logger
            self._logger.propagate = False
            return

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
    
    def _set_aux_vars(self):
        self.iter_count = 0
        self.eval_count = 0
        self.stop_dict = {}
        self.hist_f = []

        if self._eval_type == 'list':
            self._to_pheno = lambda x: x.tolist()
            self._to_geno = lambda x: Solution(x, var_name=self.var_names)
        elif self._eval_type == 'dict':
            self._to_pheno = lambda x: x.to_dict(space=self._search_space)
            self._to_geno = lambda x: Solution.from_dict(x, space=self._search_space)

    def _set_internal_optimization(self, **kwargs):
        if 'optimizer' in kwargs:
            self._optimizer = kwargs['optimizer']
        else:
            if self.N_d + self.N_i == 0 and hasattr(self.model, 'gradient'):
                self._optimizer = 'BFGS'
            else:
                self._optimizer = 'MIES'

        # NOTE: `AQ`: acquisition
        if 'max_FEs' in kwargs:
            self.AQ_max_FEs = arg_to_int(kwargs['max_FEs'])
        else:
            self.AQ_max_FEs = int(5e2 * self.dim) if self._optimizer == 'MIES' else \
                int(1e2 * self.dim)
        
        self.AQ_n_restart = int(5 * self.dim) if 'n_restart' not in kwargs \
            else arg_to_int(kwargs['n_restart'])
        self.AQ_wait_iter = 3 if 'wait_iter' not in kwargs \
            else arg_to_int(kwargs['wait_iter'])

        self._argmax_restart = functools.partial(
            argmax_restart, 
            search_space=self._search_space,
            h=self.h,
            g=self.g,
            eval_budget=self.AQ_max_FEs,
            n_restart=self.AQ_n_restart,
            wait_iter=self.AQ_wait_iter,
            optimizer=self._optimizer
        ) 

    def _check_params(self):
        # TODO: add more parameter check-ups
        if np.isinf(self.max_FEs):
            raise ValueError('max_FEs cannot be infinite')

        assert hasattr(InfillCriteria, self._acquisition_fun)
    
    def _compare(self, f1, f2):
        """Test if objecctive value f1 is better than f2
        """
        return f1 < f2 if self.minimize else f2 > f1

    def run(self):
        while not self.check_stop():
            self.step()
        return self.xopt, self.fopt, self.stop_dict

    def step(self):
        X = self.ask()

        t0 = time.time()
        func_vals = self.evaluate(X)
        self._logger.info('evaluation takes {:.4f}s'.format(time.time() - t0))

        self.tell(X, func_vals)

    def ask(self, n_point=None):
        if self.model.is_fitted:
            n_point = self.n_point if n_point is None else self.n_point
            X = self.arg_max_acquisition(n_point=n_point)
            X = self._search_space.round(X)  # round to precision if specified

            X = Solution(
                X, index=len(self.data) + np.arange(len(X)), 
                var_name=self.var_names
            )
            X = self.pre_eval_check(X)

            # TODO: handle the constrains when performing random sampling
            # draw the remaining ones randomly
            if len(X) < n_point:
                self._logger.warn(
                    "iteration {}: duplicated solution found " 
                    "by optimization! New points is taken from random "
                    "design".format(self.iter_count)
                )
                N = n_point - len(X)
                method = 'LHS' if N > 1 else 'uniform'
                s = self._search_space.sampling(N=N, method=method) 
                X = X.tolist() + s
                X = Solution(
                    X, index=len(self.data) + np.arange(len(X)), 
                    var_name=self.var_names
                )
                X = self._search_space.round(X)
        else: # initial DoE
            if not n_point:
                n_point = self._DoE_size
                
            X = self._search_space.round(
                self.create_DoE(n_point)
            )
        return self._to_pheno(X)
    
    def tell(self, X, y):
        """Feed an observation back to the surrogate model.

        Args:
            X (list): Places where the objective function has already been evaluated.
                Each suggestion is a dictionary where each key corresponds to a
                parameter being optimized.
            y (array): array-like, shape (n,)
                Corresponding values where objective has been evaluated.
        """
        X = self._to_geno(X)

        msg = 'initial DoE of size {}:'.format(len(X)) if self.iter_count == 0 else \
            'iteration {}, {} infill points:'.format(self.iter_count, len(X))
        self._logger.info(msg)

        _X = self._to_pheno(X)
        for i in range(len(X)):
            X[i].fitness = y[i]
            X[i].n_eval += 1
            self.eval_count += 1
            self._logger.info(
                '#{} - fitness: {}, solution: {}'.format(
                    i + 1, y[i], _X[i]
                )
            )

        X = self.post_eval_check(X)
        self.data = self.data + X if hasattr(self, 'data') else X

        if self.data_file is not None:
            X.to_csv(self.data_file, header=False, append=True)

        self.fopt = self._get_best(self.data.fitness)
        _xopt = self.data[np.where(self.data.fitness == self.fopt)[0][0]] 
        self.xopt = self._to_pheno(_xopt)
        if self._eval_type == 'dict':
            self.xopt = self.xopt[0]

        self._logger.info('fopt: {}'.format(self.fopt))   
        self._logger.info('xopt: {}'.format(self.xopt)) 

        if not self.model.is_fitted: 
            self._fBest_DoE = copy(self.fopt) # the best point in the DoE 
            self._xBest_DoE = copy(self.xopt)

        r2 = self.update_model()   
        self._logger.info('Surrogate model r2: {}\n'.format(r2))

        self.iter_count += 1
        self.hist_f.append(self.fopt)

    def create_DoE(self, n_point=None):
        """Create a design of experiments using Latin Hypercube Sampling.

        Args:
            n_point (int, optional): Number of points to sample. Defaults to None.

        Returns:
            Solution: Solution object with the design of experiment.
        """
        DoE = []
        while len(DoE) < n_point:
            DoE += self._search_space.sampling(n_point - len(DoE), method='LHS')
            DoE = self.pre_eval_check(DoE).tolist()
        
        return Solution(DoE, var_name=self.var_names)

    @abstractmethod
    def pre_eval_check(self, X):
        raise NotImplementedError
    
    def post_eval_check(self, X):
        _ = np.isnan(X.fitness) | np.isinf(X.fitness)
        if np.any(_):
            self._logger.warn('{} candidate solutions are removed '
                             'due to falied fitness evaluation: \n{}'.format(sum(_), str(X[_, :])))
            X = X[~_, :] 

        return X
        
    def evaluate(self, X):
        """Evaluate the candidate points and update evaluation info in the dataframe.

        Args:
            X (list): Samples to evaluate.

        Returns:
            list: evaluations of X.
        """        
        # Parallelization is handled by the objective function itself
        if self.parallel_obj_fun is not None:  
            func_vals = self.parallel_obj_fun(X)
        else: 
            if self.n_job > 1: # or by ourselves..
                func_vals = Parallel(n_jobs=self.n_job)(delayed(self.obj_fun)(x) for x in X)
            else:              # or sequential execution
                func_vals = [self.obj_fun(x) for x in X]
                
        return func_vals

    def update_model(self):
        """Update the surrogate model.

        Returns:
            float: The R^2 score of the model (not that this is on the training data).
        """
        # TODO: implement a proper model selection here 
        # TODO: in case r2 is really poor, re-fit the model or log-transform `fitness`?
        data = self.data
        fitness = data.fitness

        # to standardize the response values to prevent numerical overflow that might 
        # appear in the MGF-based acquisition function.
        # Standardization should make it easier to specify the GP prior, compared to 
        # rescaling values to the unit interval.
        fitness_ = (fitness - np.mean(fitness)) / np.std(fitness) \
            if self._acquisition_fun == 'MGFI' else fitness

        self.fmin, self.fmax = np.min(fitness_), np.max(fitness_)
        self.frange = self.fmax - self.fmin

        self.model.fit(data, fitness_)
        fitness_hat = self.model.predict(data)

        # TODO: need to report more performance metric for regression   
        r2 = r2_score(fitness_, fitness_hat)
        return r2

    def arg_max_acquisition(self, n_point=None, return_value=False):
        """Global Optimization of the acqusition function / Infill criterion

        Args:
            n_point (int, optional): Number of points to find. Defaults to None.
            return_value (bool, optional): Return only the points or also the acquisition values. Defaults to False.

        Returns:
            list: Candidate solutions in list with optional infill criterium values.
        """        
        self._logger.debug('acquisition optimziation...')
        t0 = time.time()
        n_point = self.n_point if n_point is None else int(n_point)
        return_dx = True if self._optimizer == 'BFGS' else False

        if n_point > 1:  # multi-point/batch sequential strategy
            candidates, values = self._batch_arg_max_acquisition(
                n_point=n_point, return_dx=return_dx
            )
        else:            # single-point strategy
            criteria = self._create_acquisition(par={}, return_dx=return_dx)
            candidates, values = self._argmax_restart(criteria, logger=self._logger)
            candidates, values = [candidates], [values]

        self._logger.debug(
            'acquisition optimziation takes {:.4f}s'.format(time.time() - t0)
        )
        for callback in self._acquisition_callbacks: callback()

        return (candidates, values) if return_value else candidates

    def _create_acquisition(self, fun=None, par={}, return_dx=False):
        fun = fun if fun is not None else self._acquisition_fun
        par = copy(self._acquisition_par) if not par else par
        par.update({'model' : self.model, 'minimize' : self.minimize})

        criterion = getattr(InfillCriteria, fun)(**par)
        return functools.partial(criterion, return_dx=return_dx)

    def _batch_arg_max_acquisition(self, n_point, return_dx):
        raise NotImplementedError

    def check_stop(self):
        if self.eval_count >= self.max_FEs:
            self.stop_dict['max_FEs'] = self.eval_count
        
        if self.ftarget is not None and hasattr(self, 'xopt'):
            if self._compare(self.fopt, self.ftarget):
                self.stop_dict['ftarget'] = self.fopt

        return bool(self.stop_dict)
    
    def save(self, filename):
        """Save the optimizer and all data for a later restart.

        Args:
            filename (string): Location of the file to save.
        """
        with open(filename, 'wb') as f:
            # NOTE: we need to dump `self.data` first. Otherwise, some 
            # attributes of it will be lost
            if hasattr(self, 'data'):
                self.data = dill.dumps(self.data)
            
            if len(self._logger.handlers) > 1:
                _ = [h for h in self._logger.handlers if isinstance(h, logging.FileHandler)]
                if (len(_) > 0):
                    _logger = _[0].baseFilename
                else:
                    _logger = None
            else: 
                _logger = None
                        
            logger = self._logger
            self._logger = _logger

            dill.dump(self, f)

            self._logger = logger
            if hasattr(self, 'data'):
                self.data = dill.loads(self.data)
        
    @classmethod
    def load(cls, filename):
        """Load the state from a file to restart.

        Args:
            filename (string): Path of the file.

        Returns:
            object: BO object.
        """
        with open(filename, 'rb') as f:
            obj = dill.load(f)
            if hasattr(obj, 'data'):
                obj.data = dill.loads(obj.data)

            obj.logger = obj._logger
        return obj
    
