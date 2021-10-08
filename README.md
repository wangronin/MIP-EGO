<p align="center"><img width=12.5% src="https://github.com/wangronin/MIP-EGO/blob/master/media/logo.png"></p>
<p align="center"><img width=60% src="https://github.com/wangronin/MIP-EGO/blob/master/media/textlogo.png"></p>

<!-- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -->
<!-- ![Python](https://img.shields.io/pypi/pyversions/mipego.svg)
![Python](https://img.shields.io/pypi/status/mipego.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![GitHub Issues](https://img.shields.io/github/issues/wangronin/MIP-EGO.svg)](https://github.com/wangronin/MIP-EGO/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT) -->

# Mixed-Integer Parallel Efficient Global Optimization

A `Python` implementation of the Efficient Global Optimization (EGO) / Bayesian Optimization (BO) algorithm for decision spaces composed of either real, integer, catergorical variables, or a mixture thereof.

Underpinned by surrogate models, this algorithm iteratively proposes candidate solutions using the so-called **acquisition function** which balances exploration with exploitation, and updates the surrogate model with newly observed objective values.

![](media/BO-example.gif)

The project is structured as follows:

* `MIPEGO/base.py`: the base class of Bayesian Optimization.
* `MIPEGO/BayesOpt.py` contains several BO variants:
  * `BO`: noiseless + seqential
  * `ParallelBO`: noiseless + parallel (a.k.a. batch-sequential)
  * `AnnealingBO`: noiseless + parallel + annealling [WEB18]
  * `SelfAdaptiveBO`: noiseless + parallel + self-adaptive [WEB19]
  * `NoisyBO`: noisy + parallel
  * `PCABO`: noiseless + parallel + PCA-assisted dimensionality reduction [RaponiWBBD20] **[Under Construction]**
* `MIPEGO/InfillCriteria.py`: the implemetation of acquisition functions (see below for the list of implemented ones).
* `MIPEGO/Surrogate.py`: the implementation/wrapper of sklearn's random forests model.
* `MIPEGO/SearchSpace.py`: implementation of the search/decision space.

## Features

This implementation differs from alternative packages/libraries in the following features:

* **Parallelization**, also known as _batch-sequential optimization_, for which several different approaches are implemented here.
* **Moment-Generating Function of the improvment** (MGFI) [WvSEB17a] is a recently proposed acquistion function, which implictly controls the exploration-exploitation trade-off.
* **Mixed-Integer Evolution Strategy** for optimizing the acqusition function, which is enabled when the search space is a mixture of real, integer, and categorical variables.

### Acqusition Functions

The following infill-criteria are implemented in the library:

* _Expected Improvement_ (EI)
* Probability of Improvement (PI) / Probability of Improvement
* _Upper Confidence Bound_ (UCB)
* _Moment-Generating Function of Improvement_ (MGFI)
* _Generalized Expected Improvement_ (GEI) **[Under Construction]**

For sequential working mode, Expected Improvement is used by default. For parallelization mode, MGFI is enabled by default.

### Surrogate Model

The meta (surrogate)-model used in Bayesian optimization. The basic requirement for such a model is to provide the uncertainty quantification (either empirical or theorerical) for the prediction. To easily handle the categorical data, __random forest__ model is used by default. The implementation here is based the one in _scikit-learn_, with modifications on uncertainty quantification.

<!-- 
#### Async Parallel Optimization of Neural Network Architectures

MiP-EGO also supports asynchronous parallel optimization, currently this feature is in *Beta* and being used to optimize the architecture and parameters of deep neural networks. See Example 2 for more details. -->

## Installation

```python
pip install mipego
```

## Exemplary Use Case

To use the optimizer you need to define an objective function, the search space and configure the optimizer. Below are two examples that describe most of the functionality.

### Optimizing A Black-Box Function

In this example we optimize a mixed integer black box problem.

```python
import numpy as np
from MIPEGO import ParallelBO, ContinuousSpace, OrdinalSpace, NominalSpace, RandomForest

seed = 666
np.random.seed(seed)
dim_r = 2  # dimension of the real values

def obj_fun(x):
    x_r = np.array([x['continuous_%d'%i] for i in range(dim_r)])
    x_i = x['ordinal']
    x_d = x['nominal']
    _ = 0 if x_d == 'OK' else 1
    return np.sum(x_r ** 2) + abs(x_i - 10) / 123. + _ * 2

# Continuous variables can be specified as follows:
# a 2-D variable in [-5, 5]^2
# for 2 variables, the naming scheme is continuous0, continuous1
C = ContinuousSpace([-5, 5], var_name='continuous') * dim_r

# Integer (ordinal) variables can be specified as follows:
# The domain of integer variables can be given as with continuous ones
# var_name is optional
I = OrdinalSpace([5, 15], var_name='ordinal')

# Discrete (nominal) variables can be specified as follows:
# No lb, ub... a list of categories instead
N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], var_name='nominal')

# The whole search space can be constructed:
search_space = C + I + N

# Bayesian optimization also uses a Surrogate model
# For mixed variable type, the random forest is typically used
model = RandomForest(levels=search_space.levels)

opt = ParallelBO(
    search_space=search_space,
    obj_fun=obj_fun,
    model=model,
    max_FEs=50,
    DoE_size=3,    # the initial DoE size
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
```

### Neural Architecture Search

In this example we optimize a neural network architecture on the well-known `MNIST` dataset. The objective function is computed in [this file](https://github.com/wangronin/MIP-EGO/blob/master/application/neural-architecture-search/all-cnn.py).
In the objective file the neural network architecture is defined and evaluated on the MNIST dataset. In this example, the optimizer proposes 4 candidates architectures which are trained on 4 GPUs simultaneously.

## A brief Introduction to Bayesian Optimization

Bayesian Optimization [Moc74, JSW98] (BO) is a sequential optimization strategy originally proposed to solve the single-objective black-box optimiza-tion problem that is costly to evaluate. Here, we shall restrict our discussion to the single-objective case. BO typically starts with sampling an initial design of experiment (DoE) of size, X={x<sub>1</sub>,x<sub>2</sub>,...,x<sub>n</sub>}, which is usually generated by simple random sampling, Latin Hypercube Sampling [SWN03], or the more sophisticated low-discrepancy sequence [Nie88] (e.g., Sobol sequences). Taking the initial DoE X and its corresponding objective value, Y={f(x<sub>1</sub>), f(x<sub>2</sub>),..., f(x<sub>n</sub>)} ⊆ ℝ, we proceed to construct a statistical model M describing the probability distribution of the objective function conditioned onthe initial evidence, namely Pr(f|X,Y). In most application scenarios of BO, there is a lack of a priori knowledge about f and therefore nonparametric models (e.g., Gaussian process regression or random forest) are commonly chosen for M, which gives rise to a predictor f'(x) for all x ∈ X and an uncertainty quantification s'(x) that estimates, for instance, the mean squared error of the predic-tion E(f'(x)−f(x))<sup>2</sup>. Based on f' and s', promising points can be identified via the so-called acquisition function which balances exploitation with exploration of the optimization process.

## Contributing

Please take a look at our [contributing](https://github.com/wangronin/BayesianOptimization/blob/master/CONTRIBUTING.md) guidelines if you're interested in helping!

<!-- #### Beta Features

- Async GPU execution
- Intermediate files to support restarts / resumes. -->

## Cite Us

You can find our paper on [IEEE Explore](https://ieeexplore.ieee.org/abstract/document/8851720/) and on [Arxiv](https://arxiv.org/abs/1810.05526).  
When using MiP-EGO for your research, please cite us as follows:

```bibtex
@inproceedings{van2019automatic,
    title={Automatic Configuration of Deep Neural Networks with Parallel Efficient Global Optimization},
    author={van Stein, Bas and Wang, Hao and B{\"a}ck, Thomas},
    booktitle={2019 International Joint Conference on Neural Networks (IJCNN)},
    pages={1--7},
    year={2019},
    organization={IEEE}
}
```

## Reference

* [Moc74] Jonas Mockus. "On bayesian methods for seeking the extremum". In Guri I. Marchuk, editor, _Optimization Techniques, IFIP Technical Conference, Novosibirsk_, USSR, July 1-7, 1974, volume 27 of _Lecture Notes in Computer Science_, pages 400–404. Springer, 1974.
* [JSW98] Donald R. Jones, Matthias Schonlau, and William J. Welch. "Efficient global optimization of expensive black-box functions". _J. Glob. Optim._, 13(4):455–492, 1998.
* [SWN03] Thomas J. Santner, Brian J. Williams, and William I. Notz. "The Design and Analysis of Computer Experiments". _Springer series in statistics._ Springer, 2003.
* [Nie88] Harald Niederreiter. "Low-discrepancy and low-dispersion sequences". _Journal of number theory_, 30(1):51–70, 1988.
* [WvSEB17a] Hao Wang, Bas van Stein, Michael Emmerich, and Thomas Bäck. "A New Acquisition Function for Bayesian Optimization Based on the Moment-Generating Function". In _Systems, Man, and Cybernetics (SMC), 2017 IEEE International Conference on_, pages 507–512. IEEE, 2017.
* [WEB18] Hao Wang, Michael Emmerich, and Thomas Bäck. "Cooling Strategies for the Moment-Generating Function in Bayesian Global Optimization". In _2018 IEEE Congress on Evolutionary Computation_, CEC 2018, Rio de Janeiro, Brazil, July 8-13, 2018, pages 1–8. IEEE, 2018.
* [WEB19] Hao, Wang, Michael Emmerich, and Thomas Bäck. "Towards self-adaptive efficient global optimization". In _AIP Conference Proceedings_, vol. 2070, no. 1, p. 020056. AIP Publishing LLC, 2019.
* [RaponiWBBD20] Elena Raponi, Hao Wang, Mariusz Bujny, Simonetta Boria, and Carola Doerr: "High Dimensional Bayesian Optimization Assisted by Principal Component Analysis". In _International Conference on Parallel Problem Solving from Nature_, pp. 169-183. Springer, Cham, 2020.

<!-- MiP-EGO (Mixed integer, Parallel - Efficient Global Optimization) is an optimization package that can be used to optimize Mixed integer optimization problems. A mixed-integer problem is one where some of the decision variables are constrained to be integer values or categorical values.
Next to the classical mixed integer problems, Algorithm selection or algorithm parameter optimization can also be seen as a complex mixed-integer problem.

The advantage of MiP-EGO is that it uses a surrogate model (the EGO part) to learn from the evaluations it has made so far. Instead of Gaussian Process Regression like in standard EGO, the MiP-EGO uses Random Forests instead, since Random Forests can handle mixed integer data by default.
The P in MiP-EGO stands for parallel, as this implementation has the additional feature that it can evaluate several solutions in parallel, which is extremely handy when an evaluation takes a long time and several machines are available.

For example, one use case would be to optimize an expensive (in time) simulation. There are four simulation licenses, so four simulations can be run at the same time. With MiP-EGO, all these four licenses can be fully utilized, speeding up the optimization procedure. Using a novel infill-criteria, the Moment Generating Function Based criterium, multiple points can be selected as candidate solutions. See the following paper for more detail about this criterium:
WANG, Hao, et al. A new acquisition function for Bayesian optimization based on the moment-generating function. In: Systems, Man, and Cybernetics (SMC), 2017 IEEE International Conference on. IEEE, 2017. p. 507-512. -->
