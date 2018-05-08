<p align="center"><img width=12.5% src="https://github.com/wangronin/MIP-EGO/blob/master/media/logo.png"></p>
<p align="center"><img width=60% src="https://github.com/wangronin/MIP-EGO/blob/master/media/textlogo.png"></p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/pypi/pyversions/mipego.svg)
![Python](https://img.shields.io/pypi/status/mipego.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![GitHub Issues](https://img.shields.io/github/issues/wangronin/MIP-EGO.svg)](https://github.com/wangronin/MIP-EGO/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

**MiP-EGO** *(Mixed integer, Parallel - Efficient Global Optimization)* is an optimization package that can be used to optimize Mixed integer optimization problems. A mixed-integer problem is one where some of the decision variables are constrained to be integer values or categorical values.  
Next to the classical mixed integer problems, Algorithm selection or algorithm parameter optimization can also be seen as a complex mixed-integer problem.

The advantage of MiP-EGO is that it uses a surrogate model (the EGO part) to learn from the evaluations it has made so far. Instead of *Gaussian Process Regression* like in standard *EGO*, the MiP-EGO uses Random Forests instead, since Random Forests can handle mixed integer data by default.  
The P in MiP-EGO stands for parallel, as this implementation has the additional feature that it can evaluate several solutions in parallel, which is extremely handy when an evaluation takes a long time and several machines are available.

For example, one use case would be to optimize an expensive (in time) simulation. There are four simulation licenses, so four simulations can be run at the same time. With MiP-EGO, all these four licenses can be fully utilized, speeding up the optimization procedure. Using a novel infill-criteria, the Moment Generating Function Based criterium, multiple points can be selected as candidate solutions. See the following paper for more detail about this criterium:  
WANG, Hao, et al. *A new acquisition function for Bayesian optimization based on the moment-generating function.* In: Systems, Man, and Cybernetics (SMC), 2017 IEEE International Conference on. IEEE, 2017. p. 507-512.


#### Async Parallel Optimization of Neural Network Architectures

MiP-EGO also supports asynchronous parallel optimization, currently this feature is in *Beta* and being used to optimize the architecture and parameters of deep neural networks. See Example 2 for more details.


## Install
```python
pip install mipego
```

## Usage

To use the optimizer you need to define an objective function, the search space and configure the optimizer. Below are two examples that describe most of the functionality.

### Example - Optimizing A Black-Box Function
In this example we optimize a mixed integer black box problem.

```python
import os
import numpy as np

#import our package, the surrogate model and the search space classes
from mipego import mipego
from mipego.surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

# The "black-box" objective function
def obj_func(x):
   x_r, x_i, x_d = np.array([x['C_0'],x['C_1']]), x['I'], x['N']
   if x_d == 'OK':
       tmp = 0
   else:
       tmp = 1
   return np.sum(x_r ** 2.) + abs(x_i - 10) / 123. + tmp * 2.


# First we need to define the Search Space
# the search space consists of two continues variable
# one ordinal (integer) variable
# and one categorical.
C = ContinuousSpace([-5, 5],'C') * 2 
#here we defined two variables at once using the same lower and upper bounds.
#One with label C_0, and the other with label C_1
I = OrdinalSpace([-100, 100],'I') # one integer variable with label I
N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E'], 'N')

#the search space is simply the product of the above variables
search_space = C * I * N

#next we define the surrogate model and the optimizer.
model = RandomForest(levels=search_space.levels)
opt = mipego(search_space, obj_func, model, 
                 minimize=True,     #the problem is a minimization problem.
                 max_eval=500,      #we evaluate maximum 500 times
                 max_iter=500,      #we have max 500 iterations
                 infill='EI',       #Expected improvement as criteria
                 n_init_sample=10,  #We start with 10 initial samples
                 n_point=1,         #We evaluate every iteration 1 time
                 n_job=1,           #  with 1 process (job).
                 optimizer='MIES',  #We use the MIES internal optimizer.
                 verbose=False, random_seed=None)


#and we run the optimization.
incumbent, stop_dict = opt.run()
```



### Example 2 - Optimizing A Neural Network
In this example we optimize a neural network architecture on the MNIST dataset.
The objective function in this case is [this file](https://github.com/wangronin/BayesianOptimization/blob/master/all-cnn.py) from the root repository directory.  
In the objective file the neural network architecture is defined and evaluated on the MNIST dataset.   
The code below shows how to set up the optimizer for this purpose using 4 GPUs asynchronously. 

```python
import os
import numpy as np
import subprocess, sys
from subprocess import STDOUT, check_output

#import our package, the surrogate model and the search space classes
from mipego import mipego
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

#some help packages
import re
import traceback
import time

#first lets define our objective function, 
#this is basically calling the file (all-cnn.py) and processes its output.
class obj_func(object):
    def __init__(self, program):
        self.program = program
        
    def __call__(self, cfg, gpu_no):
        print("calling program with gpu "+str(gpu_no))
        cmd = ['python3', self.program, '--cfg', str(cfg), str(gpu_no)]
        outs = ""
        outputval = 0
        try:
            #we use a timeout to cancel very long evaluations.
            outs = str(check_output(cmd,stderr=STDOUT, timeout=40000)) 
            outs = outs.split("\\n")
            
            outputval = 0
            for i in range(len(outs)-1,1,-1):
                if re.match("^\d+?\.\d+?$", outs[-i]) is not None:
                    print(outs[-i])
                    outputval = -1 * float(outs[-i])
            if np.isnan(outputval):
                outputval = 0 #default to 0.
        except subprocess.CalledProcessError as e:
            #exception handling
            traceback.print_exc()
            print (e.output)
        except:
            print ("Unexpected error:")
            traceback.print_exc()
            outputval = 0
        return outputval



objective = obj_func('./all-cnn.py')
activation_fun = ["softmax","sigmoid"] #activation function of the last layer.
activation_fun_conv = ["elu","relu","tanh","sigmoid","selu"]

#Next we define the search space.
filters = OrdinalSpace([10, 600], 'filters') * 7
kernel_size = OrdinalSpace([1, 6], 'k') * 7
strides = OrdinalSpace([1, 5], 's') * 3
stack_sizes = OrdinalSpace([1, 5], 'stack') * 3
activation = NominalSpace(activation_fun_conv, "activation")  
activation_dense = NominalSpace(activation_fun, "activ_dense") 

# to use step decay or not
step = NominalSpace([True, False], "step")  
#to use global pooling in the end or not.
global_pooling = NominalSpace([True, False], "global_pooling")

drop_out = ContinuousSpace([1e-5, .9], 'dropout') * 4 
lr_rate = ContinuousSpace([1e-4, 1.0e-0], 'lr')      #learning rate
l2_regularizer = ContinuousSpace([1e-5, 1e-2], 'l2') # l2_regularizer

search_space =  stack_sizes * strides * filters *  kernel_size * activation * activation_dense * drop_out * lr_rate * l2_regularizer * step * global_pooling 
 
#We will use the first 4 GPU's of the system.
available_gpus = [0,1,2,3] 

# use random forest as the surrogate model 
model = RandomForest(levels=search_space.levels)

#now define the optimizer.
opt = mipego(search_space, objective, model, 
                 minimize=True, max_eval=None, max_iter=500, 
                 infill='MGFI', n_init_sample=10, 
                 n_point=4, n_job=4, 
                 #4 GPU's, all evaluating 1 point at a time.
                 wait_iter=3, optimizer='MIES', 
                 verbose=False, random_seed=None,
                 available_gpus=available_gpus)


#run
incumbent, stop_dict = opt.run()
```



## Contributing
Please take a look at our [contributing](https://github.com/wangronin/BayesianOptimization/blob/master/CONTRIBUTING.md) guidelines if you're interested in helping!

#### Beta Features
- Async GPU execution
- Intermediate files to support restarts / resumes.
