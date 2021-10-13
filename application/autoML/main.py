"""
Created on 08 Okt. 2021

@author: Bas van Stein

This example shows how you can use MiP-EGO in order to perform hyper-parameter optimization for machine learning tasks.
"""

#import packages
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

#import our package, the surrogate model and the search space classes
from mipego import ParallelBO
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

# Load the dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# First we need to define the Search Space
# the search space consists of one continues variable
# one ordinal (integer) variable
# and two categorical (nominal) variables.
Cvar = ContinuousSpace([1.0, 20.0],'C') # one integer variable with label C
degree = OrdinalSpace([2,6], 'degree') 
gamma = NominalSpace(['scale', 'auto'], 'gamma') 
kernel = NominalSpace(['linear', 'poly', 'rbf', 'sigmoid'], 'kernel') 

#the complete search space is just the sum of the parameter spaces
search_space = Cvar + gamma + degree + kernel

#now we define the objective function (the model optimization)
def train_model(c):
    #define the model
    # We will use a Support Vector Classifier
    svm = SVC(kernel=c['kernel'], gamma=c['gamma'], C=c['C'], degree=c['degree'])
    cv = KFold(n_splits=4, shuffle=True, random_state=42)

    # Nested CV with parameter optimization
    cv_score = cross_val_score(svm, X=X_iris, y=y_iris, cv=cv)

    #by default mip-ego minimises, so we reverse the accuracy
    return -1 * np.mean(cv_score)


model = RandomForest(levels=search_space.levels)
opt = ParallelBO(
    search_space=search_space, 
    obj_fun=train_model, 
    model=model, 
    max_FEs=6, 
    DoE_size=5,    # the initial DoE size
    eval_type='dict',
    acquisition_fun='MGFI',
    acquisition_par={'t' : 2},
    n_job=3,       # number of processes
    n_point=3,     # number of the candidate solution proposed in each iteration
    verbose=True   # turn this off, if you prefer no output
)
xopt, fopt, stop_dict = opt.run()

print('xopt: {}'.format(xopt))
print('fopt: {}'.format(fopt))
print('stop criteria: {}'.format(stop_dict))