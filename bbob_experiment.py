import os, sys
import numpy as np
from benchmark import fgeneric
from benchmark import bbobbenchmarks as bn
from time import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from mipego import BO, ContinuousSpace, OrdinalSpace, \
    NominalSpace, RandomForest
from mipego.GaussianProcess import GaussianProcess
from mipego.GaussianProcess.trend import constant_trend
from mipego.GaussianProcess import AutoencoderGaussianProcess, PytorchGaussianProcess
from tqdm import tqdm

np.random.seed(42)

class _GaussianProcessRegressor(GaussianProcessRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_fitted = False
    
    def fit(self, X, y):
        super().fit(X, y)
        self.is_fitted = True 
        return self

    def predict(self, X, eval_MSE=False):
        _ = super().predict(X=X, return_std=eval_MSE)

        if eval_MSE:
            y_, sd = _
            sd2 = sd ** 2
            return y_, sd2
        else:
            return _

def run_optimizer(
    optimizer, 
    dim, 
    fID, 
    instance, 
    logfile, 
    lb, 
    ub, 
    max_FEs, 
    data_path, 
    bbob_opt
    ):
    """Parallel BBOB/COCO experiment wrapper
    """
    # Set different seed for different processes
    start = time()
    seed = np.mod(int(start) + os.getpid(), 1000)
    np.random.seed(seed)
    
    data_path = os.path.join(data_path, str(instance))
    max_FEs = eval(max_FEs)

    f = fgeneric.LoggingFunction(data_path, **bbob_opt)
    f.setfun(*bn.instantiate(fID, iinstance=instance))

    opt = optimizer(dim, f.evalfun, f.ftarget, max_FEs, lb, ub, logfile)
    opt.run()

    f.finalizerun()
    with open(logfile, 'a') as fout:
        fout.write(
            "{} on f{} in {}D, instance {}: FEs={}, fbest-ftarget={:.4e}, " 
            "elapsed time [m]: {:.3f}\n".format(optimizer, fID, dim, 
            instance, f.evaluations, f.fbest - f.ftarget, (time() - start) / 60.)
        )

def test_BO(dim, obj_fun, ftarget, max_FEs, lb, ub, logfile):
    """GP BO"""
    space = ContinuousSpace([lb, ub]) * dim

    # kernel = 1.0 * Matern(length_scale=(1, 1), length_scale_bounds=(1e-10, 1e2))
    # model = _GaussianProcessRegressor(kernel=kernel, alpha=0, n_restarts_optimizer=30, normalize_y=False)

    mean = constant_trend(dim, beta=0)  # equivalent to Simple Kriging
    thetaL = 1e-5 * (ub - lb) * np.ones(dim)
    thetaU = 10 * (ub - lb) * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    model = GaussianProcess(
        mean=mean, corr='matern',
        theta0=theta0, thetaL=thetaL, thetaU=thetaU,
        noise_estim=False, nugget=0,
        optimizer='BFGS', wait_iter=5, random_start=10 * dim,
        eval_budget=200 * dim
    )

    return BO(
        search_space=space, 
        obj_fun=obj_fun, 
        model=model, 
        DoE_size=max_FEs*0.2,
        max_FEs=max_FEs, 
        verbose=False, 
        n_point=1,
        minimize=True,
        acquisition_fun='EI',
        ftarget=ftarget,
        logger=None
    )

def test_BO_sklearn(dim, obj_fun, ftarget, max_FEs, lb, ub, logfile):
    """Sklearn BO"""
    space = ContinuousSpace([lb, ub]) * dim

    kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-10, 1e5))
    model = _GaussianProcessRegressor(kernel=kernel, alpha=0, n_restarts_optimizer=30, normalize_y=False)
    
    return BO(
        search_space=space, 
        obj_fun=obj_fun, 
        model=model, 
        DoE_size=max_FEs*0.2,
        max_FEs=max_FEs, 
        verbose=False, 
        n_point=1,
        minimize=True,
        acquisition_fun='EI',
        ftarget=ftarget,
        logger=None
    )


def test_GPytorchBO(dim, obj_fun, ftarget, max_FEs, lb, ub, logfile):
    """GPytorch BO"""
    space = ContinuousSpace([lb, ub]) * dim
    model = PytorchGaussianProcess()

    return BO(
        search_space=space, 
        obj_fun=obj_fun, 
        model=model, 
        DoE_size=max_FEs*0.2,
        max_FEs=max_FEs, 
        verbose=False, 
        n_point=1,
        minimize=True,
        acquisition_fun='EI',
        ftarget=ftarget,
        logger=None
    )

def test_AUBO(dim, obj_fun, ftarget, max_FEs, lb, ub, logfile):
    """Autoencoder BO"""
    space = ContinuousSpace([lb, ub]) * dim
    model = AutoencoderGaussianProcess(verbose=True, gpu="13", encoding_dim=5)
    model.train_encoder(space, sample_size=dim*100)
    return BO(
        search_space=space, 
        obj_fun=obj_fun, 
        model=model, 
        DoE_size=max_FEs*0.2,
        max_FEs=max_FEs, 
        verbose=False, 
        n_point=1,
        minimize=True,
        acquisition_fun='EI',
        ftarget=ftarget,
        logger=None
    )

if __name__ == '__main__': 
    dims = [5,10,20,40] #
    fIDs = bn.nfreeIDs[6:]    # for all fcts
    instance = [1] * 10

    for algorithm in tqdm([test_BO_sklearn, test_GPytorchBO, test_AUBO]):
        #print("algorithm", algorithm.__doc__)

        opts = {
            'max_FEs': "100",
            'lb': -5,
            'ub': 5,
            'data_path': './bbob_data/%s'%algorithm.__name__
        }
        opts['bbob_opt'] = {
            'comments': 'max_FEs={0}'.format(opts['max_FEs']),
            'algid': algorithm.__name__ 
        }
        
        for dim in tqdm(dims, leave=False):
            opts['max_FEs'] = str(10*dim+50)
            for fID in tqdm(fIDs, leave=False):
                for i in tqdm(instance, leave=False):
                    run_optimizer(
                        algorithm, dim, fID, i, logfile='./log', **opts
                    )