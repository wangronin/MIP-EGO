#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:38:25 2017

@author: wangronin & Bas van Stein
"""

import pdb

import subprocess, os, sys
from subprocess import STDOUT, check_output
import numpy as np
import time

from BayesOpt import BayesOpt
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace
import re
import traceback
import time

np.random.seed(42)

#--------------------------- Configuration settings --------------------------------------
# TODO: implement parallel execution of model
n_step = 200
n_init_sample = 10
verbose = True
save = False
logfile = 'run-mnist2.log'
class obj_func(object):
    def __init__(self, program):
        self.program = program
        
    def __call__(self, cfg, gpu_no):
        time.sleep(30)
        gpu_no += 6
        print("calling program with gpu "+str(gpu_no))
        #env = os.environ.copy()
        #env["CUDA_VISIBLE_DEVICES"] = str(gpu_no)
        cmd = ['python3', self.program, '--cfg', str(cfg), str(gpu_no)]
        outs = ""
        outputval = 0
        try:
            outs = str(check_output(cmd,stderr=STDOUT, timeout=40000)) # 
            #print("printing file")
            if os.path.isfile(logfile): 
                with open(logfile,'a') as f_handle:
                    f_handle.write(outs)
            else:
                with open(logfile,'w') as f_handle:
                    f_handle.write(outs)
            #print("done printing file")
            #print(outs)
            outs = outs.split("\\n")
            
            outputval = 0
            for i in range(len(outs)-1,1,-1):
                if re.match("^\d+?\.\d+?$", outs[-i]) is None:
                    #do nothing
                    a=1
                else:
                    print(outs[-i])
                    outputval = -1 * float(outs[-i])
            
            if np.isnan(outputval):
                outputval = 0
        except subprocess.CalledProcessError as e:
            traceback.print_exc()
            print (e.output)
        except:
            print ("Unexpected error:")
            traceback.print_exc()
            print (outs)
            
            outputval = 0
        print(outputval)
        return outputval

objective = obj_func('./all-cnn.py')
activation_fun = ["softmax"]
activation_fun_conv = ["elu","relu","tanh","sigmoid","selu"]
# the number of neurons per each layer
#dense = OrdinalSpace([2, 512], 'dense') * 2
#conv_layers = OrdinalSpace([1, 5], 'conv_layers') 
filters = OrdinalSpace([10, 600], 'filters') * 7
#pool_size = OrdinalSpace([1, 5], 'pool') * 3
kernel_size = OrdinalSpace([1, 6], 'k') * 7
strides = OrdinalSpace([1, 5], 's') * 3
stack_sizes = OrdinalSpace([1, 5], 'stack') * 3
#optimizer = NominalSpace(["adam","nadam","rmsprop"], "optimizer") 
activation = NominalSpace(activation_fun_conv, "activation")  # activation function
activation_dense = NominalSpace(activation_fun, "activ_dense") # activation function for dense layer
step = NominalSpace([True, False], "step")  # step
global_pooling = NominalSpace([True, False], "global_pooling")  # global_pooling
drop_out = ContinuousSpace([1e-5, .9], 'dropout') * 4        # drop_out rate
lr_rate = ContinuousSpace([1e-4, 1.0e-0], 'lr')        # learning rate
l2_regularizer = ContinuousSpace([1e-5, 1e-2], 'l2')# l2_regularizer

search_space =  stack_sizes * strides * filters *  kernel_size * activation * activation_dense * drop_out * lr_rate * l2_regularizer * step * global_pooling #pool_size *

# use random forest as the surrogate model for now
model = RandomForest(levels=search_space.levels)
opt = BayesOpt(search_space, objective, model, max_iter=n_step, random_seed=666,
               n_init_sample=10, n_point=10, n_jobs=10, minimize=True, 
               verbose=True, debug=False, optimizer='MIES', resume_file="mnist4.pkl")

incumbent, stop_dict = opt.run()
print (incumbent, stop_dict)
