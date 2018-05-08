#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:57:47 2017

@author: wangronin
"""

import pdb

import numpy as np
from mipego import mipego
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

np.random.seed(123)

dim = 2
n_step = 100
n_init_sample = 15

def obj_func(x):
   x_r, x_i, x_d = np.array([x['C_0'],x['C_1']]), x['I'], x['N']
   if x_d == 'OK':
       tmp = 0
   else:
       tmp = 1
   return np.sum(x_r ** 2.) + abs(x_i - 10) / 123. + tmp * 2.

C = ContinuousSpace([-5, 5],'C') * 2
I = OrdinalSpace([-100, 100],'I')
N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E'], 'N')

search_space = C * I * N


model = RandomForest(levels=search_space.levels)
# model = RrandomForest(levels=search_space.levels, seed=1, max_features='sqrt')

opt = mipego(search_space, obj_func, model, max_iter=n_step, random_seed=None,
               n_init_sample=n_init_sample, n_point=1, n_job=1, minimize=True, 
               verbose=True, optimizer='MIES')

opt.run()
