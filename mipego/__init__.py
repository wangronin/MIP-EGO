import os, logging

from . import InfillCriteria, Surrogate
from .BayesOpt import BO, ParallelBO, NoisyBO, AnnealingBO
from .Solution import Solution
from .Surrogate import RandomForest
from .SearchSpace import SearchSpace, ContinuousSpace, OrdinalSpace, NominalSpace
from .Extension import OptimizerPipeline
from . import Bi_Objective

__all__ = [
    'BO', 'ParallelBO', 'NoisyBO', 'AnnealingBO', 'Solution',
    'InfillCriteria', 'Surrogate', 'SearchSpace', 'OrdinalSpace', 'ContinuousSpace', 
    'NominalSpace', 'RandomForest', 'OptimizerPipeline', 'Bi_Objective'
]

# To use `dill` for the pickling, which works for
# much more python objects
os.environ['LOKY_PICKLER'] = 'dill' 

verbose = {
    False : logging.NOTSET,
    'DEBUG' : logging.DEBUG,
    'INFO' : logging.INFO
}

# TODO: add an interface function `fmin`
