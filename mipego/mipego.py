# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: wangronin
@email: wangronin@gmail.com

"""
from __future__ import division
from __future__ import print_function

#import pdb
import dill, functools, itertools, copyreg, logging
import numpy as np

import GPUtil as gp

import queue
import threading
import time
import copy

import json #to save and load data


from joblib import Parallel, delayed
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import r2_score

from .InfillCriteria import EI, PI, MGFI, HVI, MONTECARLO
from .optimizer import mies
from .utils import proportional_selection

from .Bi_Objective import * # added the Bi_Objective code by Christiaan Lamers

# TODO: remove the usage of pandas here change it to customized np.ndarray
# TODO: finalize the logging system
class Solution(np.ndarray):
    def __new__(cls, x, fitness=None, n_eval=0, index=None, var_name=None, loss=None,time=None):
        obj = np.asarray(x, dtype='object').view(cls)
        obj.fitness = fitness
        obj.loss = loss#CHRIS: added loss and time here for bi-objective
        obj.time = time
        obj.n_eval = n_eval
        obj.index = index
        obj.var_name = var_name
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        # Needed for array slicing
        self.fitness = getattr(obj, 'fitness', None)
        self.loss = getattr(obj, 'loss', None)# added loss and time here
        self.time = getattr(obj, 'time',None)
        self.n_eval = getattr(obj, 'n_eval', None)
        self.index = getattr(obj, 'index', None)
        self.var_name = getattr(obj, 'var_name', None)
    
    def to_dict(self):
        if self.var_name is None: return
        return {k : self[i] for i, k in enumerate(self.var_name)}     
    
    def __str__(self):
        return self.to_dict()
    

class mipego(object):
    """
    Generic Bayesian optimization algorithm
    """
    def __init__(self, search_space, obj_func, surrogate, second_surrogate=None, ftarget=None,
                 minimize=True, noisy=False, max_eval=None, 
                 infill='MGFI', t0=2, tf=1e-1, schedule=None,
                 n_init_sample=None, n_point=1, n_job=1, backend='multiprocessing',
                 n_restart=None, max_infill_eval=None, wait_iter=3, optimizer='MIES', 
                 log_file=None, data_file=None, verbose=False, random_seed=None,
                 available_gpus=[],bi_objective=False,
                 ref_time=3000.0, ref_loss=3.0, hvi_alpha=0.1, ignore_gpu=[],
                 **obj_func_params):
        """
        parameter
        ---------
            search_space : instance of SearchSpace type
            obj_func : callable,
                the objective function to optimize
            surrogate: surrogate model, currently support either GPR or random forest
            second_surrogate: second surrogate model for bi-objective optimization.
            minimize : bool,
                minimize or maximize
            noisy : bool,
                is the objective stochastic or not?
            max_eval : int,
                maximal number of evaluations on the objective function
            infill: string
                infill criterion used in optimization
                For bi-objective this should be set to HVI (hyper-volume indicator)
                Other options are MGFI, IE, and PI
                <TODO> add sanity check for this.
            t0=2, tf=1e-1, schedule=None
                Temperature parameters for the MGFI infill criterion.
            n_init_sample : int,
                the size of inital Design of Experiment (DoE),
                default: 20 * dim
            n_point : int,
                the number of candidate solutions proposed using infill-criteria,
                default : 1
            n_job : int,
                the number of jobs scheduled for parallelizing the evaluation. 
                Only Effective when n_point > 1 
            backend : str, 
                the parallelization backend, supporting: 'multiprocessing', 'MPI', 'SPARC'
            optimizer: str,
                the optimization algorithm for infill-criteria,
                supported options: 'MIES' (Mixed-Integer Evolution Strategy), 
                                   'BFGS' (quasi-Newtion for GPR)
            available_gpus: array:
                one dimensional array of GPU numbers to use for running on GPUs in parallel. Defaults to no gpus.
            bi_objective: boolean:
                False means one objective optimization (one surrogate), True: two-objective problem (two surrogates)
            ref_time: float
                reference value for the second objective (only used in bi-objective optimization)
            ref_loss: float
                reference value for the first objective (only used in bi-objective optimization)
            hvi_alpha
                allows variable lower confidence interval, only used in bi-objective optimization
            ignore_gpus: list
                list of GPU's that should not be used

            obj_func_params: dictionary
                additional variables to be passed to the objective function

        """
        self.verbose = verbose
        self.log_file = log_file
        self.data_file = data_file
        self._space = search_space
        self.var_names = self._space.var_name.tolist()
        self.obj_func = obj_func
        self.noisy = noisy
        self.time_surrogate = second_surrogate 
        self.loss_surrogate = surrogate
        self.async_time_surrogates = {}
        self.async_loss_surrogates = {}
        self.all_time_r2 = []
        self.all_loss_r2 = []
        self.n_point = n_point
        self.n_jobs = n_job
        self.available_gpus = available_gpus
        self._parallel_backend = backend
        self.ftarget = ftarget 
        self.infill = infill
        self.minimize = minimize
        self.max_iter = np.inf
        self.dim = len(self._space)
        self._best = min if self.minimize else max
        self.ignore_gpu = ignore_gpu
        self.surr_time_fit_hist = []
        self.surr_time_mies_hist = []
        self.surr_loss_fit_hist = []
        self.surr_loss_mies_hist = []
        self.time_between_gpu_hist = []#time in gpuworker() that a network is not trained on a gpu
        self.obj_func_params = obj_func_params;
        
        self.bi = bi_objective 
        self.hvi_alpha = hvi_alpha # allows variable lower confidence interval
        
        self.r_index = self._space.id_C       # index of continuous variable
        self.i_index = self._space.id_O       # index of integer variable
        self.d_index = self._space.id_N       # index of categorical variable

        self.param_type = self._space.var_type
        self.N_r = len(self.r_index)
        self.N_i = len(self.i_index)
        self.N_d = len(self.d_index)
       
        # parameter: objective evaluation
        # TODO: for noisy objective function, maybe increase the initial evaluations
        self.init_n_eval = 1      
        self.max_eval = int(max_eval) if max_eval else np.inf
        self.n_left = int(max_eval) if max_eval else np.inf # counts number of iterations left
        self.n_init_sample = self.dim * 20 if n_init_sample is None else int(n_init_sample)
        self_eval_hist = [] #TODO remove this and make it work
        self.eval_hist_time = [] # added time and loss history
        self.eval_hist_loss = []
        self.eval_hist_id = []
        self.iter_count = 0
        self.eval_count = 0
        self.ref_time = ref_time
        self.ref_loss = ref_loss
        
        # setting up cooling schedule
        if self.infill == 'MGFI':
            self.t0 = t0
            self.tf = tf
            self.t = t0
            self.schedule = schedule
            
            # TODO: find a nicer way to integrate this part
            # cooling down to 1e-1
            max_iter = self.max_eval - self.n_init_sample 
            if self.schedule == 'exp':                         # exponential
                self.alpha = (self.tf / t0) ** (1. / max_iter) 
            elif self.schedule == 'linear':
                self.eta = (t0 - self.tf) / max_iter           # linear
            elif self.schedule == 'log':
                self.c = self.tf * np.log(max_iter + 1)        # logarithmic 
            elif self.schedule == 'self-adaptive':
                raise NotImplementedError

        # paramter: acquisition function optimziation
        mask = np.nonzero(self._space.C_mask | self._space.O_mask)[0]
        self._bounds = np.array([self._space.bounds[i] for i in mask])             # bounds for continuous and integer variable
        # self._levels = list(self._space.levels.values())
        self._levels = np.array([self._space.bounds[i] for i in self._space.id_N]) # levels for discrete variable
        self._optimizer = optimizer
        # TODO: set this number smaller when using L-BFGS and larger for MIES
        self._max_eval = int(5e2 * self.dim) if max_infill_eval is None else max_infill_eval
        self._random_start = int(5 * self.dim) if n_restart is None else n_restart
        self._wait_iter = int(wait_iter)    # maximal restarts when optimal value does not change

        # Intensify: the number of potential configuations compared against the current best
        # self.mu = int(np.ceil(self.n_init_sample / 3))
        self.mu = 3
        
        # stop criteria
        self.stop_dict = {}
        self.hist_f = []
        self._check_params()

        # set the random seed
        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(self.random_seed)
        
        self._get_logger(self.log_file)
        
        # allows for pickling the objective function 
        copyreg.pickle(self._eval_one, dill.pickles) 
        copyreg.pickle(self.obj_func, dill.pickles) 

        # paralellize gpus
        self.init_gpus = True
        self.evaluation_queue = queue.Queue()
        
    
    def _get_logger(self, logfile):
        """
        When logfile is None, no records are written
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('- %(asctime)s [%(levelname)s] -- '
                                      '[- %(process)d - %(name)s] %(message)s')

        # create console handler and set level to warning
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # create file handler and set level to debug
        if logfile is not None:
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def _compare(self, f1, f2):
        """
        Test if perf1 is better than perf2
        """
        if self.minimize:
            return f1 < f2
        else:
            return f2 > f2
    
    def _remove_duplicate(self, data):
        """
        check for the duplicated solutions, as it is not allowed
        for noiseless objective functions
        """
        ans = []
        X = np.array([s.tolist() for s in self.data], dtype='object')
        for i, x in enumerate(data):
            CON = np.all(np.isclose(np.asarray(X[:, self.r_index], dtype='float'),
                                    np.asarray(x[self.r_index], dtype='float')), axis=1)
            INT = np.all(X[:, self.i_index] == x[self.i_index], axis=1)
            CAT = np.all(X[:, self.d_index] == x[self.d_index], axis=1)
            if not any(CON & INT & CAT):
                ans.append(x)
        return ans

    def _eval_gpu(self, x, gpu=0, runs=1):
        """
        evaluate one solution
        """

        # TODO: sometimes the obj_func take a dictionary as input...
        time_,loss_, n_eval = x.time,x.loss, x.n_eval
        # try:
            # ans = [self.obj_func(x.tolist()) for i in range(runs)]
        # except:
        #ans = [self.obj_func(x.to_dict(), gpu_no=gpu) for i in range(runs)]
        gpu_patch = gpu
        while True:
            ans = self.obj_func(x.to_dict(), gpu_no=gpu_patch, **self.obj_func_params)
            

            time_ans,loss_ans,success= ans[0],ans[1],ans[2]
            
            if success:
                break
            else:
                while True:
                    print('gpu ' + str(gpu_patch) + ' failed to give answer, searching for new gpu')
                    available_gpus_patch = gp.getAvailable()
                    for i in range(len(self.ignore_gpu)):
                        try:
                            available_gpus_patch.remove(self.ignore_gpu[i])
                        except:
                            pass
                    if len(available_gpus_patch) > 0:
                        gpu_patch = available_gpus_patch[0]
                        break
                    else:
                        print('no gpus available, waiting 60 seconds')
                        time.sleep(60)

        time_loc = np.sum(time_ans)
        loss_loc = np.sum(loss_ans)
        
        x.n_eval += runs
        x.time = time_loc / runs if time_ is None else (time_ * n_eval + time_loc) / x.n_eval
        x.loss = loss_loc / runs if loss_ is None else (loss_ * n_eval + loss_loc) / x.n_eval

        self.eval_hist_loss += [loss_ans] # added time and loss history
        self.eval_hist_time += [time_ans]
        self.eval_hist_id += [x.index] * runs
        
        return x, runs, time_loc, loss_loc, [x.index] * runs

    def _eval_one(self, x, runs=1):
        """
        evaluate one solution
        """
        # TODO: sometimes the obj_func take a dictionary as input...
        time_ = x.time
        loss_ = x.loss
        n_eval = x.n_eval

        ans = self.obj_func(x.to_dict(), **self.obj_func_params)
        time_ans,loss_ans,success = ans[0],ans[1],ans[2]
        
        time = np.sum(time_ans)
        loss = np.sum(loss_ans)
        
        x.n_eval += runs
        x.time = time / runs if time_ is None else (time_ * n_eval + time) / x.n_eval
        x.loss = loss / runs if loss_ is None else (loss_ * n_eval + loss) / x.n_eval

        self.eval_hist_loss += [x.loss]
        self.eval_hist_time += [x.time]
        self.eval_hist_id += [x.index] * runs
        
        return x, runs, time, loss, [x.index] * runs

    def evaluate(self, data, runs=1):
        """ Evaluate the candidate points and update evaluation info in the dataframe
        """
        if isinstance(data, Solution):
            self._eval_one(data)
        
        elif isinstance(data, list): 
            if self.n_jobs > 1:
                if self._parallel_backend == 'multiprocessing': # parallel execution using joblib
                    res = Parallel(n_jobs=self.n_jobs, verbose=False)(delayed(self._eval_one, check_pickle=False)(x) for x in data)
                    x, runs, hist_time, hist_loss, hist_id = zip(*res)
                    self.eval_count += sum(runs)
                    self.eval_hist_time += list(itertools.chain(*hist_time))
                    self.eval_hist_loss += list(itertools.chain(*hist_loss))
                    self.eval_hist_id += list(itertools.chain(*hist_id))
                    for i, k in enumerate(data):
                        data[i] = x[i].copy()
                elif self._parallel_backend == 'MPI': # parallel execution using MPI
                    # TODO: to use InstanceRunner here
                    pass
                elif self._parallel_backend == 'Spark': # parallel execution using Spark
                    pass        
            else:
                for x in data:
                    self._eval_one(x)
                    self.eval_count += 1

    def fit_and_assess(self, time_surrogate = None, loss_surrogate = None):
        while True:
            try:
                X = np.atleast_2d([s.tolist() for s in self.data])
                time_fitness = np.array([s.time for s in self.data])
                
                #TODO: is normalization really a good idea here? can be removed, or save scaling factor and give to s-metric (min and max)
                # normalization the response for numerical stability
                # e.g., for MGF-based acquisition function
                #_time_min, _time_max = np.min(time_fitness), np.max(time_fitness)
                #time_fitness_scaled = (time_fitness - _time_min) / (_time_max - _time_min) #Xin Guo improvement
                
                if len(time_fitness) == 1: # for the case n_init_sample=1 #Xin Guo improvement
                    time_fitness_scaled = time_fitness
                else:
                    time_min, time_max = np.min(time_fitness), np.max(time_fitness)
                    if not time_min == time_max: # for the case of flat fitness
                        time_fitness_scaled = (time_fitness - time_min) / (time_max - time_min)
                    else:
                        time_fitness_scaled = time_fitness
                
                # fit the time surrogate model
                if (time_surrogate is None):
                    start_timer = time.time()
                    self.time_surrogate.fit(X, time_fitness)
                    stop_timer = time.time()
                    self.surr_time_fit_hist.append(stop_timer-start_timer)
                    self.time_is_update = True
                    start_timer = time.time()
                    time_fitness_hat = self.time_surrogate.predict(X)
                    stop_timer = time.time()
                    self.surr_time_mies_hist.append(stop_timer-start_timer)
                else:
                    start_timer = time.time()
                    time_surrogate.fit(X,  time_fitness)
                    stop_timer = time.time()
                    self.surr_time_fit_hist.append(stop_timer-start_timer)
                    self.time_is_update = True
                    start_timer = time.time()
                    time_fitness_hat = time_surrogate.predict(X)
                    stop_timer = time.time()
                    self.surr_time_mies_hist.append(stop_timer-start_timer)
                
                
                
                loss_fitness = np.array([s.loss for s in self.data])
                
                # normalization the response for numerical stability
                # e.g., for MGF-based acquisition function
                #_loss_min, _loss_max = np.min(loss_fitness), np.max(loss_fitness) #Xin Guo improvement
                #loss_fitness_scaled = (loss_fitness - _loss_min) / (_loss_max - _loss_min)
                
                if len(loss_fitness) == 1: # for the case n_init_sample=1 #Xin Guo improvement
                    loss_fitness_scaled = loss_fitness
                else:
                    loss_min, loss_max = np.min(loss_fitness), np.max(loss_fitness)
                    if not loss_min == loss_max: # for the case of flat fitness
                        loss_fitness_scaled = (loss_fitness - loss_min) / (loss_max - loss_min)
                    else:
                        loss_fitness_scaled = loss_fitness
            
                # fit the loss surrogate model
                if (loss_surrogate is None):
                    start_timer = time.time()
                    self.loss_surrogate.fit(X, loss_fitness)
                    stop_timer = time.time()
                    self.surr_loss_fit_hist.append(stop_timer-start_timer)
                    self.loss_is_update = True
                    start_timer = time.time()
                    loss_fitness_hat = self.loss_surrogate.predict(X)
                    stop_timer = time.time()
                    self.surr_loss_mies_hist.append(stop_timer - start_timer)
                else:
                    start_timer = time.time()
                    loss_surrogate.fit(X, loss_fitness)
                    stop_timer = time.time()
                    self.surr_loss_fit_hist.append(stop_timer-start_timer)
                    self.loss_is_update = True
                    start_timer = time.time()
                    loss_fitness_hat = loss_surrogate.predict(X)
                    stop_timer = time.time()
                    self.surr_loss_mies_hist.append(stop_timer - start_timer)
                
                #TODO use s-metric to calculate fitness? this is just for logging, optimization (searching for candidate) takes place before this step, so what does surrogate.predict do? the fitting part is useful though
                #fitness_hat = surrogate.predict(X)
                #TODO, maybe it's usefull to cast time and loss variables to sms-ego fitness here
                
                time_r2 = r2_score(time_fitness, time_fitness_hat)
                loss_r2 = r2_score(loss_fitness, loss_fitness_hat)
                self.all_time_r2.append(time_r2)
                self.all_loss_r2.append(loss_r2)
                break
            except Exception as e:
                print("Error fitting model, retrying...")
                print(X)
                print(time_fitness)
                print(loss_fitness)
                print(e)
                time.sleep(15)
        # TODO: in case r2 is really poor, re-fit the model or transform the input? 
        # consider the performance metric transformation in SMAC
        self.logger.info('Surrogate model time_r2: {}'.format(time_r2))
        self.logger.info('Surrogate model loss_r2: {}'.format(loss_r2))
        return time_r2,loss_r2

    def select_candidate(self):
        self.is_update = False
        X, infill_value = self.arg_max_acquisition(plugin=None, time_surrogate=self.time_surrogate, loss_surrogate=self.loss_surrogate,data=self.data ,n_left=self.n_left,max_iter=self.max_iter)
        
        if self.n_point > 1:
            X = [Solution(x, index=len(self.data) + i, var_name=self.var_names) for i, x in enumerate(X)]
        else:
            X = [Solution(X, index=len(self.data), var_name=self.var_names)]
            
        X = self._remove_duplicate(X)
        # if the number of new design sites obtained is less than required,
        # draw the remaining ones randomly
        if len(X) < self.n_point:
            self.logger.warn("iteration {}: duplicated solution found " 
                                "by optimization! New points is taken from random "
                                "design".format(self.iter_count))
            N = self.n_point - len(X)
            if N > 1:
                s = self._space.sampling(N=N, method='LHS')
            else:      # To generate a single sample, only uniform sampling is feasible
                s = self._space.sampling(N=1, method='uniform')
            X += [Solution(x, index=len(self.data) + i, var_name=self.var_names) for i, x in enumerate(s)]
        
        candidates_id = [x.index for x in X]
        # for noisy fitness: perform a proportional selection from the evaluated ones   
        if self.noisy:
            #after evaluate run S-metric on all solutions to determine fitness
            for i in range(len(self.data)):# this is a bottleneck
                other_solutions = copy.deepcopy(self.data)
                del other_solutions[i]
                self.data[i].fitness = s_metric(self.data[i], other_solutions,self.n_left,self.max_iter,ref_time=self.ref_time,ref_loss=self.ref_loss)
            id_, fitness = zip([(i, d.fitness) for i, d in enumerate(self.data) if i != self.incumbent_id])
            __ = proportional_selection(fitness, self.mu, self.minimize, replacement=False)
            candidates_id.append(id_[__])
        
        # TODO: postpone the evaluate to intensify...
        self.evaluate(X, runs=self.init_n_eval)
        print("n_left")
        print(self.n_left)
        self.data += X
        # after evaluate run S-metric on all solutions to determine fitness
        for i in range(len(self.data)):# this is a bottleneck
            other_solutions = copy.deepcopy(self.data)
            del other_solutions[i]
            self.data[i].fitness = s_metric(self.data[i], other_solutions,self.n_left,self.max_iter,ref_time=self.ref_time,ref_loss=self.ref_loss)
        
        return candidates_id

    def intensify(self, candidates_ids):
        """
        intensification procedure for noisy observations (from SMAC)
        """
        # TODO: verify the implementation here
        maxR = 20 # maximal number of the evaluations on the incumbent
        for i, ID in enumerate(candidates_ids):
            r, extra_run = 1, 1
            conf = self.data.loc[i]
            self.evaluate(conf, 1)
            print(conf.to_frame().T)

            if conf.n_eval > self.incumbent_id.n_eval:
                self.incumbent_id = self.evaluate(self.incumbent_id, 1)
                extra_run = 0

            while True:
                if self._compare(self.incumbent_id.perf, conf.perf):
                    self.incumbent_id = self.evaluate(self.incumbent_id, 
                                                   min(extra_run, maxR - self.incumbent_id.n_eval))
                    print(self.incumbent_id.to_frame().T)
                    break
                if conf.n_eval > self.incumbent_id.n_eval:
                    self.incumbent_id = conf
                    if self.verbose:
                        print('[DEBUG] iteration %d -- new incumbent selected:' % self.iter_count)
                        print('[DEBUG] {}'.format(self.incumbent_id))
                        print('[DEBUG] with performance: {}'.format(self.incumbent_id.perf))
                        print()
                    break

                r = min(2 * r, self.incumbent_id.n_eval - conf.n_eval)
                self.data.loc[i] = self.evaluate(conf, r)
                print(self.conf.to_frame().T)
                extra_run += r
    
    def _initialize(self):
        """Generate the initial data set (DOE) and construct the surrogate model
        """
        self.logger.info('selected time_surrogate model: {}'.format(self.time_surrogate.__class__))
        self.logger.info('selected loss_surrogate model: {}'.format(self.loss_surrogate.__class__))
        self.logger.info('building the initial design of experiemnts...')

        samples = self._space.sampling(self.n_init_sample)
        self.data = [Solution(s, index=k, var_name=self.var_names) for k, s in enumerate(samples)]
        self.evaluate(self.data, runs=self.init_n_eval)
        
        # after evaluate run S-metric on all solutions to determine fitness
        for i in range(len(self.data)):# this is a bottleneck
            other_solutions = copy.deepcopy(self.data)
            del other_solutions[i]
            self.data[i].fitness = s_metric(self.data[i], other_solutions,self.n_left,self.max_iter,ref_time=self.ref_time,ref_loss=self.ref_loss)
        
        # set the initial incumbent
        fitness = np.array([s.fitness for s in self.data])

        self.incumbent_id = np.nonzero(fitness == self._best(fitness))[0][0]
        self.fit_and_assess()

    def gpuworker(self, q, gpu_no):
        "GPU worker function "

        self.async_time_surrogates[gpu_no] = copy.deepcopy(self.time_surrogate);
        self.async_loss_surrogates[gpu_no] = copy.deepcopy(self.loss_surrogate);
        while True:
            start_timer_1 = time.time()
            self.logger.info('GPU no. {} is waiting for task'.format(gpu_no))
            #print("Queue size before q.get()= " + str(q.qsize()))
            confs_ = q.get()
            #print("Queue size after q.get()= " + str(q.qsize()))
            time.sleep(gpu_no)
            

            self.logger.info('Evaluating:')
            self.logger.info(confs_.to_dict())
            stop_timer_1 = time.time()
            
            confs_ = self._eval_gpu(confs_, gpu_no)[0] #will write the result to confs_
            
            start_timer_2 = time.time()
            self.n_left -= 1
            if self.n_left < 0:
                self.n_left = 0
            self.iter_count += 1
            
            if self.data is None:
                self.data = [confs_]
            else: 
                self.data += [confs_]
            

            for i in range(len(self.data)):# this is a bottleneck
                other_solutions = copy.deepcopy(self.data)
                del other_solutions[i]
                self.data[i].fitness = s_metric(self.data[i], other_solutions,self.n_left,self.max_iter,ref_time=self.ref_time,ref_loss=self.ref_loss)

            q.task_done()

            #print "GPU no. {} is waiting for task on thread {}".format(gpu_no, gpu_no)
            if not self.check_stop():
                self.logger.info('Data size is {}'.format(len(self.data)))
                if len(self.data) >= self.n_init_sample:
                    self.fit_and_assess(time_surrogate = self.async_time_surrogates[gpu_no], loss_surrogate = self.async_loss_surrogates[gpu_no])
                    while True:
                        try:
                            X, infill_value = self.arg_max_acquisition(plugin=None, time_surrogate = self.async_time_surrogates[gpu_no], loss_surrogate=self.async_loss_surrogates[gpu_no],data=self.data ,n_left=self.n_left)# two surrogates are needed
                            confs_ = Solution(X, index=len(self.data)+q.qsize(), var_name=self.var_names)
                            break
                        except Exception as e:
                            print(e)
                            print("Error selecting candidate, retrying in 60 seconds...")
                            time.sleep(60)
                    q.put(confs_)
                else:
                    samples = self._space.sampling(1)
                    confs_ = Solution(samples[0], index=len(self.data)+q.qsize(), var_name=self.var_names)
                    #confs_ = self._to_dataframe(self._space.sampling(1))
                    if (q.empty()):
                        q.put(confs_)
                
            else:
                break
            stop_timer_2 = time.time()
            self.time_between_gpu_hist.append((stop_timer_1 - start_timer_1)+(stop_timer_2-start_timer_2))

        print('Finished thread {}'.format(gpu_no))

    def save_data(self,filename):
        conf_array = []
        fit_array = []
        time_array = []
        loss_array = []
        n_eval_array = []
        index_array = []
        name_array = []
        
        for i in range(len(self.data)):
            conf_array.append(self.data[i].to_dict())
            fit_array.append(self.data[i].fitness)
            time_array.append(self.data[i].time)
            loss_array.append(self.data[i].loss)
            n_eval_array.append(self.data[i].n_eval)
            index_array.append(self.data[i].index)
            name_array.append(self.data[i].var_name)
        data_array = [conf_array,fit_array,time_array,loss_array,n_eval_array,index_array,name_array,self.all_time_r2,self.all_loss_r2,self.surr_time_fit_hist, self.surr_time_mies_hist, self.surr_loss_fit_hist, self.surr_loss_mies_hist,self.time_between_gpu_hist]
        
        with open(filename + '.json', 'w') as outfile:
            json.dump(data_array,outfile)
        return

    def step(self):
        if not hasattr(self, 'data'):
           self._initialize()
        
        ids = self.select_candidate()
        if self.noisy:
            self.incumbent_id = self.intensify(ids)
        else:
            fitness = np.array([s.fitness for s in self.data])
            self.incumbent_id = np.nonzero(fitness == self._best(fitness))[0][0]

        self.incumbent = self.data[self.incumbent_id]
        
        # model re-training
        # TODO: test more control rules on model refitting
        # if self.eval_count % 2 == 0:
            # self.fit_and_assess()
        self.fit_and_assess()
        self.n_left -= 1
        if self.n_left < 0:
            self.n_left = 0
        self.iter_count += 1
        self.hist_f.append(self.incumbent.fitness)

        self.logger.info('iteration {}, current incumbent is:'.format(self.iter_count))
        self.logger.info(self.incumbent.to_dict())
        
        # save the iterative data configuration to csv
        # self.incumbent.to_csv(self.data_file, header=False, index=False, mode='a')
        return self.incumbent, self.incumbent.fitness

    def run(self,restart=False):
        if (len(self.available_gpus) > 0):

            if self.n_jobs > len(self.available_gpus):
                print("Not enough GPUs available for n_jobs")
                return 1,1 # changed "1" to "1,1". This avoids an error message

            #self.n_point = 1 #set n_point to 1 because we only do one evaluation at a time (async)# n_point is set to 1 at initialisation
            # initialize
            self.logger.info('selected time_surrogate model: {}'.format(self.time_surrogate.__class__))
            self.logger.info('selected loss_surrogate model: {}'.format(self.loss_surrogate.__class__))
            self.logger.info('building the initial design of experiments...')
            
            if not restart:
                samples = self._space.sampling(self.n_init_sample)
                datasamples = [Solution(s, index=k, var_name=self.var_names) for k, s in enumerate(samples)]
                self.data = None


                for i in range(self.n_init_sample):
                    self.evaluation_queue.put(datasamples[i])

                self.iter_count -= self.n_init_sample# because initial samples are in queue, counters count them as normal samples, so this needs to be coutered
                self.n_left += self.n_init_sample
            else:
                for i in range(self.n_jobs):
                    self.evaluation_queue.put(self.data[i-self.n_jobs])
                for i in range(self.n_jobs):
                    del self.data[-1]
                    del self.all_time_r2[-1]
                    del self.all_loss_r2[-1]
                    del self.surr_time_fit_hist[-1]
                    del self.surr_time_mies_hist[-1]
                    del self.surr_loss_fit_hist[-1]
                    del self.surr_loss_mies_hist[-1]
                    del self.time_between_gpu_hist[-1]
                self.n_left += self.n_jobs
                self.iter_count -= self.n_jobs
                self.eval_count -= self.n_jobs

            thread_dict = {}
            # launch threads for all GPUs
            for i in range(self.n_jobs):
                t = threading.Thread(target=self.gpuworker, args=(self.evaluation_queue,
                                                                  self.available_gpus[i],))
                t.setDaemon = True
                thread_dict[i] = t
                t.start()

            # wait for queue to be empty and all threads to finish
            self.evaluation_queue.join()
            threads = [thread_dict[a] for a in thread_dict]
            for thread in threads:
                thread.join()

            print('\n\n All threads should now be done. Finishing program...\n\n')

            self.stop_dict['n_eval'] = self.eval_count
            self.stop_dict['n_iter'] = self.iter_count

            return #self.incumbent, self.stop_dict #self.incumbent does not exist anymore

        else:

            while not self.check_stop():
                self.step()

            self.stop_dict['n_eval'] = self.eval_count
            self.stop_dict['n_iter'] = self.iter_count
            return #self.incumbent, self.stop_dict #self.incumbent does not exist anymore

    def check_stop(self):
        # TODO: add more stop criteria
        # unify the design purpose of stop_dict
        if self.iter_count >= self.max_iter:
            self.stop_dict['max_iter'] = True

        if self.eval_count >= self.max_eval:
            self.stop_dict['max_eval'] = True
        
        if self.ftarget is not None and hasattr(self, 'incumbent') and \
            self._compare(self.incumbent.perf, self.ftarget):
            self.stop_dict['ftarget'] = True
        return len(self.stop_dict)

    def _acquisition(self, plugin=None, dx=False, time_surrogate=None, loss_surrogate=None,data=None,n_left=None,max_iter=None):
        if plugin is None:
            # plugin = np.min(self.data.perf) if self.minimize else -np.max(self.data.perf)
            # Note that performance are normalized when building the surrogate
            plugin = 0 if self.minimize else -1
        #here two surrogate functions are needed
        if (time_surrogate is None):
            time_surrogate = self.time_surrogate;
        if (loss_surrogate is None):
            loss_surrogate = self.loss_surrogate;
        if (data is None):
            data = self.data
        if (n_left is None):
            n_left = self.n_left
        if (max_iter is None):
            max_iter = self.max_iter
            
        if self.n_point == 1: # sequential mode
            if self.infill == 'HVI':
                acquisition_func = HVI(time_model=time_surrogate, loss_model=loss_surrogate, plugin=plugin, minimize=self.minimize, solutions=data, n_left=n_left,max_iter=max_iter,sol=Solution,ref_time=self.ref_time,ref_loss=self.ref_loss, alpha=self.hvi_alpha)
            elif self.infill == 'MC':
                acquisition_func = MONTECARLO(model=time_surrogate, plugin=plugin, minimize=self.minimize)
            else:
                print("Error, only HVI and MC infill criterium work for this implementation")
        else:
            print("Error, n_point should be 1 for this implementation")
                
        return functools.partial(acquisition_func, dx=dx)
        
    def _annealling(self):
        if self.schedule == 'exp':  
             self.t *= self.alpha
        elif self.schedule == 'linear':
            self.t -= self.eta
        elif self.schedule == 'log':
            # TODO: verify this
            self.t = self.c / np.log(self.iter_count + 1 + 1)
        
    def arg_max_acquisition(self, plugin=None, time_surrogate=None, loss_surrogate=None,data=None ,n_left=None,max_iter=None):
        """
        Global Optimization on the acqusition function 
        """
        if self.verbose:
            self.logger.info('acquisition function optimziation...')
        
        dx = True if self._optimizer == 'BFGS' else False
        #two surrogate functions must be passed
        obj_func = [self._acquisition(plugin, dx=dx, time_surrogate=time_surrogate, loss_surrogate=loss_surrogate,n_left=n_left,max_iter=max_iter) for i in range(self.n_point)]

        if self.n_point == 1:
            candidates, values = self._argmax_multistart(obj_func[0])
        else:
            # parallelization using joblib
            res = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._argmax_multistart, check_pickle=False)(func) for func in obj_func)
            candidates, values = list(zip(*res))
        return candidates, values

    def _argmax_multistart(self, obj_func):
        # keep the list of optima in each restart for future usage
        xopt, fopt = [], []  
        eval_budget = self._max_eval
        best = -np.inf
        wait_count = 0

        for iteration in range(self._random_start):
            x0 = self._space.sampling(N=1, method='uniform')[0]
            
            # TODO: add IPOP-CMA-ES here for testing
            # TODO: when the surrogate is GP, implement a GA-BFGS hybrid algorithm
            if self._optimizer == 'BFGS':
                if self.N_d + self.N_i != 0:
                    raise ValueError('BFGS is not supported with mixed variable types.')
                # TODO: find out why: somehow this local lambda function can be pickled...
                # for minimization
                func = lambda x: tuple(map(lambda x: -1. * x, obj_func(x)))
                xopt_, fopt_, stop_dict = fmin_l_bfgs_b(func, x0, pgtol=1e-8,
                                                        factr=1e6, bounds=self._bounds,
                                                        maxfun=eval_budget)
                xopt_ = xopt_.flatten().tolist()
                fopt_ = -np.asscalar(fopt_)
                
                if stop_dict["warnflag"] != 0 and self.verbose:
                    self.logger.warn("L-BFGS-B terminated abnormally with the "
                                     " state: %s" % stop_dict)
                                
            elif self._optimizer == 'MIES':
                #here send to MIES optimizer that uses s-metric as obj_func
                opt = mies(self._space, obj_func, max_eval=eval_budget, minimize=False, verbose=False, plus_selection=False)
                xopt_, fopt_, stop_dict = opt.optimize()

            if fopt_ > best:
                best = fopt_
                wait_count = 0
                if self.verbose:
                    self.logger.info('restart : {} - funcalls : {} - Fopt : {}'.format(iteration + 1, 
                        stop_dict['funcalls'], fopt_))
            else:
                wait_count += 1

            eval_budget -= stop_dict['funcalls']
            xopt.append(xopt_)
            fopt.append(fopt_)
            
            if eval_budget <= 0 or wait_count >= self._wait_iter:
                break
        # maximization: sort the optima in descending order
        idx = np.argsort(fopt)[::-1]
        return xopt[idx[0]], fopt[idx[0]]

    def _check_params(self):
        assert hasattr(self.obj_func, '__call__')

        if np.isinf(self.max_eval) and np.isinf(self.max_iter):
            raise ValueError('max_eval and max_iter cannot be both infinite')
